// Checks the float attention kernels of zero/kernels/ops.cl (Gemma 4, and head 128 for Flint) against a CPU
// computation in double on random data: attn_fa_g (one row per sub-group, -DFA_ROWS, -DFA_SPLIT + attn_fa_gc) and
// attn_fa_gh (KB 1 and 2), heads of 128 and 256, global (causal) and sliding with a ring cache smaller than the
// positions, with and without attention sinks (-DSINKS, Flint). The forms that are meant to be bit-identical (attn_fa_g, FA_ROWS, attn_fa_gh) are compared byte for
// byte too. usage: test_fa [kernel_file]
#include "../src/clrt.h"
#include "../src/quants.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double rel_err(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i]; }
  return std::sqrt(num / std::max(den, 1e-30));
}

int main(int argc, char** argv) {
  const std::string file = argc > 1 ? argv[1] : "zero/kernels/ops.cl";
  const std::string src = read_file(file);
  ClRt cl;
  if (!cl.init()) return 1;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  int fails = 0;
  const int nh = 8, nkv = 2, hpg = nh / nkv, M = 16;
  for (int HD : {128, 256}) {
    for (int win : {0, 100})
    for (int sink : {0, 1}) {
      // keys of positions 0 .. pos0 + M - 1 written at slot p % cap; the rows are pos0 .. pos0 + M - 1
      const int pos0 = 290, cap = win ? 128 : 320, P = pos0 + M;
      std::vector<float> q((size_t)M * nh * HD), kf((size_t)P * nkv * HD), vf((size_t)P * nkv * HD);
      for (auto& x : q) x = nd(rng) * 0.25f;
      for (auto& x : kf) x = nd(rng) * 0.25f;
      for (auto& x : vf) x = nd(rng);
      std::vector<float> sinks(nh);
      for (auto& x : sinks) x = nd(rng) * 2.0f;
      std::vector<float> kc((size_t)nkv * cap * HD, 0.0f);
      std::vector<uint16_t> vc((size_t)nkv * cap * HD, 0);
      for (int p = 0; p < P; ++p)
        for (int g = 0; g < nkv; ++g)
          for (int d = 0; d < HD; ++d) {
            const size_t at = ((size_t)g * cap + p % cap) * HD + d;
            kc[at] = kf[((size_t)p * nkv + g) * HD + d];
            vc[at] = f32_to_fp16(vf[((size_t)p * nkv + g) * HD + d]);
            vf[((size_t)p * nkv + g) * HD + d] = fp16_to_f32(vc[at]);  // (the reference sees the f16 values)
          }
      // CPU reference
      std::vector<double> ref((size_t)M * nh * HD, 0.0);
      for (int m = 0; m < M; ++m) {
        const int pos = pos0 + m, lo = win > 0 ? std::max(0, pos - win + 1) : 0;
        for (int h = 0; h < nh; ++h) {
          const int g = h / hpg;
          std::vector<double> s(pos - lo + 1);
          double mx = -1e300;
          for (int p = lo; p <= pos; ++p) {
            double a = 0;
            for (int d = 0; d < HD; ++d) a += (double)q[((size_t)m * nh + h) * HD + d] * kf[((size_t)p * nkv + g) * HD + d];
            s[p - lo] = a;
            mx = std::max(mx, a);
          }
          if (sink) mx = std::max(mx, (double)sinks[h]);
          double z = sink ? std::exp(sinks[h] - mx) : 0.0;
          for (double& x : s) z += x = std::exp(x - mx);
          for (int p = lo; p <= pos; ++p)
            for (int d = 0; d < HD; ++d)
              ref[((size_t)m * nh + h) * HD + d] += s[p - lo] / z * vf[((size_t)p * nkv + g) * HD + d];
        }
      }
      cl_mem bq = cl.buffer(CL_MEM_READ_ONLY, q.size() * 4, q.data());
      cl_mem bk = cl.buffer(CL_MEM_READ_ONLY, kc.size() * 4, kc.data());
      cl_mem bv = cl.buffer(CL_MEM_READ_ONLY, vc.size() * 2, vc.data());
      cl_mem bo = cl.buffer(CL_MEM_READ_WRITE, (size_t)M * nh * HD * 4);
      cl_mem bs = cl.buffer(CL_MEM_READ_ONLY, sinks.size() * 4, sinks.data());
      int cp = cap, nn = nh, nk = nkv, pp = pos0, w = win, nlo = -1, nhi = -1;
      auto result = [&](int rows) {
        std::vector<float> o((size_t)rows * nh * HD);
        CL_CHECK(clEnqueueReadBuffer(cl.queue, bo, CL_TRUE, 0, o.size() * 4, o.data(), 0, nullptr, nullptr));
        return o;
      };
      auto check = [&](const char* name, const std::vector<float>& o, const std::vector<float>* same) {
        std::vector<double> r(ref.begin(), ref.begin() + o.size());
        const double er = rel_err(o, r);
        const bool bit = !same || (same->size() >= o.size() && !memcmp(same->data(), o.data(), o.size() * 4));
        const bool ok = er < 2e-6 && bit;
        printf("HD %3d %-8s%s %-22s rel err %.2e%s %s\n", HD, win ? "sliding" : "global", sink ? " sinks" : "      ", name, er,
               same ? (bit ? ", bit-identical" : ", NOT bit-identical") : "", ok ? "OK" : "FAIL");
        fails += !ok;
      };
      const std::string base = "-cl-std=CL3.0 -cl-mad-enable -DHD=" + std::to_string(HD) + (sink ? " -DSINKS" : "");
      // attn_fa_g, one row per sub-group
      const int gq = 2;
      std::vector<float> o_g;
      {
        cl_kernel k = cl.kernel(cl.build(src, base + " -DFA_G -DGQ=" + std::to_string(gq)), "attn_fa_g");
        int MM = M;
        set_args(k, 0, bq, bk, bv, cp, nn, nk, pp, MM, bo, w, nlo, nhi);
        if (sink) set_args(k, 12, bs);
        size_t gws[3] = {16, (size_t)(nh / gq), (size_t)M}, lws[3] = {16, 1, 1};
        CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 3, nullptr, gws, lws, 0, nullptr, nullptr));
        o_g = result(M);
        check("attn_fa_g", o_g, nullptr);
      }
      {  // FA_ROWS 4
        cl_kernel k = cl.kernel(cl.build(src, base + " -DFA_G -DGQ=" + std::to_string(gq) + " -DFA_ROWS=4"), "attn_fa_g");
        int MM = M;
        set_args(k, 0, bq, bk, bv, cp, nn, nk, pp, MM, bo, w, nlo, nhi);
        if (sink) set_args(k, 12, bs);
        size_t gws[3] = {16, (size_t)(nh / gq), (size_t)M}, lws[3] = {16, 1, 4};
        CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 3, nullptr, gws, lws, 0, nullptr, nullptr));
        check("attn_fa_g FA_ROWS 4", result(M), &o_g);
      }
      {  // FA_SPLIT + attn_fa_gc, the first 8 rows
        const int n = 8, last = pos0 + n - 1, lo0 = win > 0 ? std::max(0, pos0 - win + 1) : 0;
        const int nb = last / 16 - lo0 / 16 + 1, ns = std::min(64, (nb + 7) / 8);
        cl_program pr = cl.build(src, base + " -DFA_G -DFA_SPLIT -DGQ=" + std::to_string(gq));
        cl_mem part = cl.buffer(CL_MEM_READ_WRITE, (size_t)n * nh * 64 * (HD + 2) * 4);
        cl_kernel k = cl.kernel(pr, "attn_fa_g");
        int MM = n, sg = ns;
        set_args(k, 0, bq, bk, bv, cp, nn, nk, pp, MM, bo, w, nlo, nhi, sg, part);
        size_t gws[3] = {16, (size_t)(nh / gq), (size_t)n * ns}, lws[3] = {16, 1, 1};
        CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 3, nullptr, gws, lws, 0, nullptr, nullptr));
        cl_kernel kc2 = cl.kernel(pr, "attn_fa_gc");
        set_args(kc2, 0, part, nn, pp, MM, bo, w, nlo, nhi);
        if (sink) set_args(kc2, 8, bs);
        size_t g2[3] = {(size_t)HD, (size_t)nh, (size_t)n};
        CL_CHECK(clEnqueueNDRangeKernel(cl.queue, kc2, 3, nullptr, g2, nullptr, 0, nullptr, nullptr));
        check("attn_fa_g FA_SPLIT", result(n), nullptr);
      }
      // attn_fa_gh: 4 output dims per lane (head 128: 2 sub-groups, head 256: 4)
      const int fs = HD / 64, gd = HD / fs / 16, gr = std::clamp(64 / (hpg * gd), 1, 4);
      for (int kb : {1, 2}) {
        cl_kernel k = cl.kernel(cl.build(src, base + " -DFA_GH -DHPG=" + std::to_string(hpg) + " -DFS=" + std::to_string(fs) +
                                                  " -DGR=" + std::to_string(gr) + " -DKB=" + std::to_string(kb)),
                                "attn_fa_gh");
        int MM = M;
        set_args(k, 0, bq, bk, bv, cp, nn, nk, pp, MM, bo, w, nlo, nhi);
        if (sink) set_args(k, 12, bs);
        size_t gws[3] = {(size_t)16 * fs, (size_t)nkv, (size_t)(M + gr - 1) / gr}, lws[3] = {(size_t)16 * fs, 1, 1};
        CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 3, nullptr, gws, lws, 0, nullptr, nullptr));
        check(kb == 1 ? "attn_fa_gh KB 1" : "attn_fa_gh KB 2", result(M), &o_g);
      }
    }
  }
  // attn_prep_g, head 128 as Flint uses it: q / k RMS norms with weights, NEOX RoPE (base 10000) or none
  // (theta_base 0), q scaled by 1/sqrt(128), V as projected (has_v = 2, no norm); the f32 q and the float K / V rows
  for (int rope : {1, 0}) {
    const int HD = 128, Mr = 4, pos0 = 37, cap = 64, ldq = (nh + 2 * nkv) * HD;
    const float eps = 1e-6f, base = rope ? 10000.0f : 0.0f, qs = 1.0f / std::sqrt(128.0f);
    std::vector<float> qkv((size_t)Mr * ldq), qn(HD), kn(HD);
    for (auto& x : qkv) x = nd(rng);
    for (auto& x : qn) x = 1.0f + 0.3f * nd(rng);
    for (auto& x : kn) x = 1.0f + 0.3f * nd(rng);
    cl_program pr = cl.build(src, "-cl-std=CL3.0 -cl-mad-enable -DHD=128 -DSB=2");
    cl_mem bqkv = cl.buffer(CL_MEM_READ_ONLY, qkv.size() * 4, qkv.data());
    cl_mem bqn = cl.buffer(CL_MEM_READ_ONLY, qn.size() * 4, qn.data()), bkn = cl.buffer(CL_MEM_READ_ONLY, kn.size() * 4, kn.data());
    cl_mem bqf = cl.buffer(CL_MEM_READ_WRITE, (size_t)Mr * nh * HD * 4);
    cl_mem bkc = cl.buffer(CL_MEM_READ_WRITE, (size_t)nkv * cap * HD * 4), bvc = cl.buffer(CL_MEM_READ_WRITE, (size_t)nkv * cap * HD * 2);
    cl_mem dummy = cl.buffer(CL_MEM_READ_WRITE, (size_t)Mr * nh * HD * 4);
    cl_kernel k = cl.kernel(pr, "attn_prep_g");
    int ld = ldq, nn = nh, nk = nkv, hv = 2, MM = Mr, pp = pos0, uff = 0, cp = cap, f16 = 3;
    float ep = eps, bs = base, q_s = qs;
    cl_mem no_table = nullptr;
    set_args(k, 0, bqkv, ld, nn, nk, hv, bqn, bkn, MM, pp, ep, bs, bqn, uff, q_s, dummy, dummy, dummy, dummy, dummy, dummy,
             cp, no_table);
    set_args(k, 22, f16, bqf, bkc, bvc);
    size_t gws[2] = {(size_t)(nh + nkv) * 16, (size_t)Mr}, lws[2] = {16, 1};
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr));
    std::vector<float> qf((size_t)Mr * nh * HD), kcg((size_t)nkv * cap * HD);
    std::vector<uint16_t> vcg((size_t)nkv * cap * HD);
    CL_CHECK(clEnqueueReadBuffer(cl.queue, bqf, CL_TRUE, 0, qf.size() * 4, qf.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl.queue, bkc, CL_TRUE, 0, kcg.size() * 4, kcg.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl.queue, bvc, CL_TRUE, 0, vcg.size() * 2, vcg.data(), 0, nullptr, nullptr));
    // CPU: the same steps in double
    auto head = [&](const float* x, const float* w, int pos, double scale, std::vector<double>& out) {
      double ss = 0;
      for (int d = 0; d < HD; ++d) ss += (double)x[d] * x[d];
      const double r = 1.0 / std::sqrt(ss / HD + eps);
      std::vector<double> y(HD);
      for (int d = 0; d < HD; ++d) y[d] = x[d] * r * w[d];
      out = y;
      if (rope)
        for (int i = 0; i < HD / 2; ++i) {
          const double th = pos * std::pow(10000.0, -2.0 * i / HD), c = std::cos(th), s = std::sin(th);
          out[i] = y[i] * c - y[i + HD / 2] * s;
          out[i + HD / 2] = y[i + HD / 2] * c + y[i] * s;
        }
      for (double& v : out) v *= scale;
    };
    std::vector<float> gq, gk, gv;
    std::vector<double> rq, rk, rv;
    for (int m = 0; m < Mr; ++m) {
      const int pos = pos0 + m;
      for (int h = 0; h < nh; ++h) {
        std::vector<double> o;
        head(&qkv[(size_t)m * ldq + (size_t)h * HD], qn.data(), pos, qs, o);
        rq.insert(rq.end(), o.begin(), o.end());
        gq.insert(gq.end(), qf.begin() + ((size_t)m * nh + h) * HD, qf.begin() + ((size_t)m * nh + h + 1) * HD);
      }
      for (int g = 0; g < nkv; ++g) {
        std::vector<double> o;
        head(&qkv[(size_t)m * ldq + (size_t)(nh + g) * HD], kn.data(), pos, 1.0, o);
        rk.insert(rk.end(), o.begin(), o.end());
        const size_t at = ((size_t)g * cap + pos % cap) * HD;
        gk.insert(gk.end(), kcg.begin() + at, kcg.begin() + at + HD);
        for (int d = 0; d < HD; ++d) {
          const float v = qkv[(size_t)m * ldq + (size_t)(nh + nkv + g) * HD + d];
          rv.push_back(fp16_to_f32(f32_to_fp16(v)));
          gv.push_back(fp16_to_f32(vcg[at + d]));
        }
      }
    }
    const double eq = rel_err(gq, rq), ek = rel_err(gk, rk), ev = rel_err(gv, rv);
    const bool ok = eq < 1e-5 && ek < 1e-5 && ev == 0.0;
    printf("HD 128 attn_prep_g %-8s q rel err %.2e, k %.2e, v %.2e %s\n", rope ? "RoPE" : "no RoPE", eq, ek, ev, ok ? "OK" : "FAIL");
    fails += !ok;
  }
  printf(fails ? "test_fa: %d FAILED\n" : "test_fa: all passed\n", fails);
  return fails ? 1 : 0;
}
