// Checks the MoE routing kernels of zero/kernels/moe.cl (moe_route, moe_route1) against a CPU computation on random
// logits: softmax weights (Qwen) and sigmoid weights with an ungated shared FFN (-DSIGMOID -DSH_NOGATE, Flint), top
// NU of NE experts, weights normalized over the selection, the shared-expert gate, the per-expert counts.
// usage: test_route [kernel_file]
#include "../src/clrt.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const std::string file = argc > 1 ? argv[1] : "zero/kernels/moe.cl";
  const std::string src = read_file(file);
  ClRt cl;
  if (!cl.init()) return 1;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  int fails = 0;
  const int NE = 256, NU = 6, T = 7, ldl = NE + 1;
  for (int sig : {0, 1}) {
    std::vector<float> lg((size_t)T * ldl);
    for (auto& x : lg) x = nd(rng) * 2.0f;
    // CPU
    std::vector<int> rid((size_t)T * NU), rcount(NE, 0);
    std::vector<double> rw((size_t)T * NU), rg(T);
    for (int t = 0; t < T; ++t) {
      const float* l = &lg[(size_t)t * ldl];
      std::vector<int> idx(NE);
      std::iota(idx.begin(), idx.end(), 0);
      std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
      const double mx = l[idx[0]];
      double sum = 0;
      for (int k = 0; k < NU; ++k) {
        const int e = idx[k];
        rid[(size_t)t * NU + k] = e;
        rw[(size_t)t * NU + k] = sig ? 1.0 / (1.0 + std::exp(-(double)l[e])) : std::exp(l[e] - mx);
        sum += rw[(size_t)t * NU + k];
        ++rcount[e];
      }
      for (int k = 0; k < NU; ++k) rw[(size_t)t * NU + k] /= sum;
      rg[t] = sig ? 1.0 : 1.0 / (1.0 + std::exp(-(double)l[NE]));
    }
    const std::string opts = "-cl-std=CL3.0 -cl-mad-enable -DNE=" + std::to_string(NE) + " -DNU=" + std::to_string(NU) +
                             " -DTM=16" + (sig ? " -DSIGMOID -DSH_NOGATE" : "");
    cl_program pr = cl.build(src, opts);
    cl_mem blg = cl.buffer(CL_MEM_READ_ONLY, lg.size() * 4, lg.data());
    cl_mem bid = cl.buffer(CL_MEM_READ_WRITE, (size_t)T * NU * 4), bw = cl.buffer(CL_MEM_READ_WRITE, (size_t)T * NU * 4);
    cl_mem bg = cl.buffer(CL_MEM_READ_WRITE, (size_t)T * 4);
    std::vector<int> zero(NE, 0);
    cl_mem bc = cl.buffer(CL_MEM_READ_WRITE, NE * 4, zero.data());
    {
      cl_kernel k = cl.kernel(pr, "moe_route");
      int ld = ldl, tt = T;
      set_args(k, 0, blg, ld, tt, bid, bw, bg, bc);
      size_t gws[2] = {16, (size_t)T}, lws[2] = {16, 1};
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr));
      std::vector<int> id((size_t)T * NU), cnt(NE);
      std::vector<float> w((size_t)T * NU), g(T);
      CL_CHECK(clEnqueueReadBuffer(cl.queue, bid, CL_TRUE, 0, id.size() * 4, id.data(), 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl.queue, bw, CL_TRUE, 0, w.size() * 4, w.data(), 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl.queue, bg, CL_TRUE, 0, g.size() * 4, g.data(), 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl.queue, bc, CL_TRUE, 0, cnt.size() * 4, cnt.data(), 0, nullptr, nullptr));
      double ew = 0, eg = 0;
      for (size_t i = 0; i < w.size(); ++i) ew = std::max(ew, std::fabs(w[i] - rw[i]));
      for (int t = 0; t < T; ++t) eg = std::max(eg, std::fabs(g[t] - rg[t]));
      const bool ok = id == rid && cnt == rcount && ew < 1e-6 && eg < 1e-6;
      printf("moe_route  %-8s ids %s, counts %s, max weight err %.1e, gate err %.1e %s\n", sig ? "sigmoid" : "softmax",
             id == rid ? "same" : "DIFFER", cnt == rcount ? "same" : "DIFFER", ew, eg, ok ? "OK" : "FAIL");
      fails += !ok;
    }
    {  // one token: the first row
      cl_kernel k = cl.kernel(pr, "moe_route1");
      cl_mem tiles = cl.buffer(CL_MEM_READ_WRITE, (1 + 3 * NU) * 4), stok = cl.buffer(CL_MEM_READ_WRITE, NU * 4);
      cl_mem swt = cl.buffer(CL_MEM_READ_WRITE, NU * 4), inv = cl.buffer(CL_MEM_READ_WRITE, NU * 4);
      set_args(k, 0, blg, bg, tiles, stok, swt, inv);
      size_t g1 = 16, l1 = 16;
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 1, nullptr, &g1, &l1, 0, nullptr, nullptr));
      std::vector<int> tl(1 + 3 * NU);
      std::vector<float> w(NU), g(1);
      CL_CHECK(clEnqueueReadBuffer(cl.queue, tiles, CL_TRUE, 0, tl.size() * 4, tl.data(), 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl.queue, swt, CL_TRUE, 0, w.size() * 4, w.data(), 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl.queue, bg, CL_TRUE, 0, 4, g.data(), 0, nullptr, nullptr));
      bool ids_ok = tl[0] == NU;
      double ew = 0;
      for (int k2 = 0; k2 < NU; ++k2) {
        ids_ok = ids_ok && tl[1 + 3 * k2] == rid[k2];
        ew = std::max(ew, std::fabs(w[k2] - rw[k2]));
      }
      const double eg = std::fabs(g[0] - rg[0]);
      const bool ok = ids_ok && ew < 1e-6 && eg < 1e-6;
      printf("moe_route1 %-8s ids %s, max weight err %.1e, gate err %.1e %s\n", sig ? "sigmoid" : "softmax",
             ids_ok ? "same" : "DIFFER", ew, eg, ok ? "OK" : "FAIL");
      fails += !ok;
    }
  }
  printf(fails ? "test_route: %d FAILED\n" : "test_route: all passed\n", fails);
  return fails ? 1 : 0;
}
