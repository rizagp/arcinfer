// Checks the hyper-connection kernels of zero/kernels/ops.cl (Qwen3.8 Flash Next) against a CPU computation on
// random data: hc_norm_quant, hc_lo_quant, hc_gate_mean_quant, hc_combine. usage: test_hc [kernel_file]
#include "../src/clrt.h"
#include "../src/quants.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static double rel_err(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i]; }
  return std::sqrt(num / std::max(den, 1e-30));
}
// int8 activations: elements whose dequantized value differs by more than one step from the CPU quantizer's
static int quant_mismatch(const std::vector<int8_t>& aq, const std::vector<float>& asc, const std::vector<float>& ref, int M,
                          int K, int Mv) {
  std::vector<int8_t> q2((size_t)M * K);
  std::vector<float> s2((size_t)(K / 32) * M);
  std::vector<uint16_t> a2((size_t)(K / 16) * M);
  quantize_act_cpu(ref.data(), M, K, q2.data(), s2.data(), a2.data(), true);
  int bad = 0;
  for (int m = 0; m < Mv; ++m)
    for (int k = 0; k < K; ++k) {
      const float g = aq[(size_t)m * K + k] * asc[(size_t)(k / 32) * M + m], c = q2[(size_t)m * K + k] * s2[(size_t)(k / 32) * M + m];
      if (std::fabs(g - c) > 1.01f * s2[(size_t)(k / 32) * M + m] + 1e-12f) ++bad;
    }
  return bad;
}

int main(int argc, char** argv) {
  const std::string file = argc > 1 ? argv[1] : "zero/kernels/ops.cl";
  const int E = 2560, HC = 4, HD = HC * E, LR = 320, KP = 512, M = 16, Mv = 7;
  const float eps = 1e-6f;
  ClRt cl;
  if (!cl.init()) return 1;
  cl_program prog = cl.build(read_file(file), "-cl-std=CL3.0 -cl-mad-enable");
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto rnd = [&](size_t n, float sc) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * sc; return v; };
  int fails = 0;

  // hc_norm_quant
  auto res = rnd((size_t)M * HD, 2.0f), w = rnd(HD, 0.3f);
  for (auto& x : w) x += 1.0f;
  std::vector<float> xn((size_t)M * HD);
  {
    cl_mem bres = cl.buffer(CL_MEM_READ_WRITE, res.size() * 4, res.data()), bw = cl.buffer(CL_MEM_READ_ONLY, w.size() * 4, w.data());
    cl_mem bxn = cl.buffer(CL_MEM_READ_WRITE, xn.size() * 4), baq = cl.buffer(CL_MEM_READ_WRITE, (size_t)M * HD);
    cl_mem basc = cl.buffer(CL_MEM_READ_WRITE, (size_t)(HD / 32) * M * 4), babs = cl.buffer(CL_MEM_READ_WRITE, (size_t)(HD / 16) * M * 2);
    cl_kernel k = cl.kernel(prog, "hc_norm_quant");
    int e = E, hc = HC, mm = M, mv = Mv, a32 = 1;
    float ep = eps;
    set_args(k, 0, bres, bw, e, hc, mm, mv, ep, bxn, baq, basc, babs, a32);
    size_t gws[2] = {256, (size_t)M}, lws[2] = {256, 1};
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr));
    std::vector<int8_t> aq((size_t)M * HD);
    std::vector<float> asc((size_t)(HD / 32) * M);
    CL_CHECK(clEnqueueReadBuffer(cl.queue, bxn, CL_TRUE, 0, xn.size() * 4, xn.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl.queue, baq, CL_TRUE, 0, aq.size(), aq.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl.queue, basc, CL_TRUE, 0, asc.size() * 4, asc.data(), 0, nullptr, nullptr));
    std::vector<double> ref((size_t)M * HD, 0.0);
    for (int m = 0; m < Mv; ++m)
      for (int c = 0; c < HC; ++c) {
        double ss = 0;
        for (int i = 0; i < E; ++i) ss += (double)res[(size_t)m * HD + c * E + i] * res[(size_t)m * HD + c * E + i];
        const double r = 1.0 / std::sqrt(ss / E + eps);
        for (int i = 0; i < E; ++i) ref[(size_t)m * HD + c * E + i] = res[(size_t)m * HD + c * E + i] * r * w[c * E + i];
      }
    const double er = rel_err(xn, ref);
    const int qb = quant_mismatch(aq, asc, xn, M, HD, Mv);
    printf("hc_norm_quant: xn rel err %.2e, int8 off by > 1 step: %d of %d %s\n", er, qb, Mv * HD, er < 1e-6 && qb == 0 ? "OK" : "FAIL");
    fails += !(er < 1e-6 && qb == 0);
  }
  // hc_lo_quant
  {
    auto lo = rnd((size_t)M * LR, 3.0f);
    cl_mem blo = cl.buffer(CL_MEM_READ_ONLY, lo.size() * 4, lo.data()), baq = cl.buffer(CL_MEM_READ_WRITE, (size_t)M * KP);
    cl_mem basc = cl.buffer(CL_MEM_READ_WRITE, (size_t)(KP / 32) * M * 4), babs = cl.buffer(CL_MEM_READ_WRITE, (size_t)(KP / 16) * M * 2);
    cl_kernel k = cl.kernel(prog, "hc_lo_quant");
    int lr = LR, kp = KP, hc = HC, mm = M, mv = Mv, a32 = 1;
    set_args(k, 0, blo, lr, kp, hc, mm, mv, baq, basc, babs, a32);
    size_t gws[2] = {(size_t)16 * KP / 256, (size_t)M}, lws[2] = {16, 1};
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr));
    std::vector<int8_t> aq((size_t)M * KP);
    std::vector<float> asc((size_t)(KP / 32) * M);
    CL_CHECK(clEnqueueReadBuffer(cl.queue, baq, CL_TRUE, 0, aq.size(), aq.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl.queue, basc, CL_TRUE, 0, asc.size() * 4, asc.data(), 0, nullptr, nullptr));
    std::vector<float> ref((size_t)M * KP, 0.0f);
    for (int m = 0; m < Mv; ++m)
      for (int i = 0; i < LR; ++i) { const float v = lo[(size_t)m * LR + i] / HC; ref[(size_t)m * KP + i] = v / (1.0f + std::exp(-v)); }
    const int qb = quant_mismatch(aq, asc, ref, M, KP, Mv);
    int pad_bad = 0;
    for (int m = 0; m < M; ++m)
      for (int i = (m < Mv ? LR : 0); i < KP; ++i) pad_bad += aq[(size_t)m * KP + i] != 0;
    printf("hc_lo_quant: int8 off by > 1 step: %d of %d, non-zero padding: %d %s\n", qb, Mv * KP, pad_bad, qb == 0 && !pad_bad ? "OK" : "FAIL");
    fails += !(qb == 0 && !pad_bad);
  }
  // hc_gate_mean_quant (xn from the first test)
  {
    auto gate = rnd((size_t)M * HD, 2.0f);
    cl_mem bxn = cl.buffer(CL_MEM_READ_ONLY, xn.size() * 4, xn.data()), bg = cl.buffer(CL_MEM_READ_ONLY, gate.size() * 4, gate.data());
    cl_mem bmix = cl.buffer(CL_MEM_READ_WRITE, (size_t)M * E * 4), baq = cl.buffer(CL_MEM_READ_WRITE, (size_t)M * E);
    cl_mem basc = cl.buffer(CL_MEM_READ_WRITE, (size_t)(E / 32) * M * 4), babs = cl.buffer(CL_MEM_READ_WRITE, (size_t)(E / 16) * M * 2);
    cl_kernel k = cl.kernel(prog, "hc_gate_mean_quant");
    int e = E, hc = HC, mm = M, mv = Mv, a32 = 1;
    set_args(k, 0, bxn, bg, e, hc, mm, mv, bmix, baq, basc, babs, a32);
    size_t gws[2] = {256, (size_t)M}, lws[2] = {256, 1};
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr));
    std::vector<float> mix((size_t)M * E), asc((size_t)(E / 32) * M);
    std::vector<int8_t> aq((size_t)M * E);
    CL_CHECK(clEnqueueReadBuffer(cl.queue, bmix, CL_TRUE, 0, mix.size() * 4, mix.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl.queue, baq, CL_TRUE, 0, aq.size(), aq.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl.queue, basc, CL_TRUE, 0, asc.size() * 4, asc.data(), 0, nullptr, nullptr));
    std::vector<double> ref((size_t)M * E, 0.0);
    for (int m = 0; m < Mv; ++m)
      for (int i = 0; i < E; ++i) {
        double acc = 0;
        for (int c = 0; c < HC; ++c) { const size_t j = (size_t)m * HD + c * E + i; acc += xn[j] / (1.0 + std::exp(-(double)gate[j])); }
        ref[(size_t)m * E + i] = acc / HC;
      }
    const double er = rel_err(mix, ref);
    const int qb = quant_mismatch(aq, asc, mix, M, E, Mv);
    printf("hc_gate_mean_quant: mixed rel err %.2e, int8 off by > 1 step: %d of %d %s\n", er, qb, Mv * E, er < 1e-5 && qb == 0 ? "OK" : "FAIL");
    fails += !(er < 1e-5 && qb == 0);
  }
  // hc_combine
  {
    auto out = rnd((size_t)M * E, 1.0f), inj = rnd((size_t)M * HC, 4.0f);
    cl_mem bres = cl.buffer(CL_MEM_READ_WRITE, res.size() * 4, res.data()), bo = cl.buffer(CL_MEM_READ_ONLY, out.size() * 4, out.data());
    cl_mem bi = cl.buffer(CL_MEM_READ_ONLY, inj.size() * 4, inj.data());
    cl_kernel k = cl.kernel(prog, "hc_combine");
    int e = E, hc = HC;
    set_args(k, 0, bres, bo, bi, e, hc);
    size_t gws[2] = {(size_t)HD, (size_t)Mv};
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
    std::vector<float> got(res.size());
    CL_CHECK(clEnqueueReadBuffer(cl.queue, bres, CL_TRUE, 0, got.size() * 4, got.data(), 0, nullptr, nullptr));
    std::vector<double> ref(res.begin(), res.end());
    for (int m = 0; m < Mv; ++m)
      for (int c = 0; c < HC; ++c) {
        const double wgt = 2.0 / (1.0 + std::exp(-(double)inj[(size_t)m * HC + c] / HC));
        for (int i = 0; i < E; ++i) ref[(size_t)m * HD + c * E + i] += wgt * out[(size_t)m * E + i];
      }
    const double er = rel_err(got, ref);
    printf("hc_combine: rel err %.2e %s\n", er, er < 1e-6 ? "OK" : "FAIL");
    fails += !(er < 1e-6);
  }
  printf(fails ? "SOME FAILED\n" : "ALL OK\n");
  return fails ? 1 : 0;
}
