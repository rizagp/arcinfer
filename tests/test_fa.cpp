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
  printf(fails ? "test_fa: %d FAILED\n" : "test_fa: all passed\n", fails);
  return fails ? 1 : 0;
}
