// Benchmarks + validates the Q4_K x int8 DP4a GEMM kernel on the iGPU.
// usage: gemm_bench [M N K TM WGX grf256 iters]
#include "../src/clrt.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 512;
  int N = argc > 2 ? atoi(argv[2]) : 9216;
  int K = argc > 3 ? atoi(argv[3]) : 2560;
  int TM = argc > 4 ? atoi(argv[4]) : 16;
  int WGX = argc > 5 ? atoi(argv[5]) : 4;
  int grf256 = argc > 6 ? atoi(argv[6]) : 1;
  int iters = argc > 7 ? atoi(argv[7]) : 20;
  int WGY = argc > 8 ? atoi(argv[8]) : 1;
  const int SG = 16, nsb = K / 256, G = N / SG;
  if (K % 256 || N % SG || M % TM) { fprintf(stderr, "bad shape\n"); return 1; }

  std::mt19937 rng(42);
  std::vector<uint32_t> wq((size_t)G * nsb * 4 * 8 * SG), wh((size_t)G * nsb * 6 * SG);
  for (auto& x : wq) x = rng();
  std::uniform_real_distribution<float> ud(0.0005f, 0.002f);
  for (int g = 0; g < G; ++g)
    for (int s = 0; s < nsb; ++s)
      for (int l = 0; l < SG; ++l) {
        uint32_t* h = &wh[((size_t)g * nsb + s) * 6 * SG];
        float d = ud(rng), dm = ud(rng);
        memcpy(&h[0 * SG + l], &d, 4);
        memcpy(&h[1 * SG + l], &dm, 4);
        for (int f = 2; f < 6; ++f) h[f * SG + l] = rng() & 0x3F3F3F3Fu;  // 6-bit sc / mn
      }

  std::vector<int8_t> aq((size_t)M * K);
  std::vector<float> asc((size_t)nsb * M);
  std::vector<uint16_t> absum((size_t)nsb * 8 * M);
  for (auto& x : aq) x = (int8_t)((int)(rng() % 255) - 127);
  for (auto& x : asc) x = 0.01f + (rng() % 1000) * 4e-5f;
  for (int m = 0; m < M; ++m)
    for (int s = 0; s < nsb; ++s)
      for (int b = 0; b < 8; ++b) {
        int sum = 0;
        for (int i = 0; i < 32; ++i) sum += aq[(size_t)m * K + s * 256 + b * 32 + i];
        absum[((size_t)s * 8 + b) * M + m] = (uint16_t)(int16_t)sum;
      }

  ClRt cl;
  if (!cl.init()) return 1;
  std::string opts = "-cl-std=CL3.0 -cl-mad-enable -DTM=" + std::to_string(TM);
  if (grf256) opts += " -cl-intel-256-GRF-per-thread";
  if (argc > 9) opts += std::string(" ") + argv[9];  // extra kernel defines, e.g. "-DA_UNIFORM"
  auto prog = cl.build(read_file("kernels/gemm_q4k.cl"), opts);
  auto k = cl.kernel(prog, "gemm_q4k_q8");
  printf("TM=%d WGX=%d WGY=%d grf256=%d spill=%zu B\n", TM, WGX, WGY, grf256, cl.spill_bytes(k));

  auto bwq = cl.buffer(CL_MEM_READ_ONLY, wq.size() * 4, wq.data());
  auto bwh = cl.buffer(CL_MEM_READ_ONLY, wh.size() * 4, wh.data());
  auto baq = cl.buffer(CL_MEM_READ_ONLY, aq.size(), aq.data());
  auto basc = cl.buffer(CL_MEM_READ_ONLY, asc.size() * 4, asc.data());
  auto babs = cl.buffer(CL_MEM_READ_ONLY, absum.size() * 2, absum.data());
  auto bc = cl.buffer(CL_MEM_WRITE_ONLY, (size_t)M * N * 4);
  set_args(k, 0, bwq, bwh, baq, basc, babs, bc, M, N, K);

  size_t gws[2] = {(size_t)N, (size_t)(M / TM)};
  size_t lws[2] = {(size_t)(SG * WGX), (size_t)WGY};
  double best = 1e30, total = 0;
  for (int it = 0; it < iters + 3; ++it) {
    cl_event e;
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, &e));
    double ms = ClRt::event_ms(e);
    clReleaseEvent(e);
    if (it >= 3) { best = std::min(best, ms); total += ms; }
  }
  double ops = 2.0 * M * N * K;
  printf("M=%d N=%d K=%d  best %.3f ms (%.2f TOPS)  avg %.3f ms (%.2f TOPS)\n", M, N, K, best, ops / best * 1e-9,
         total / iters, ops / (total / iters) * 1e-9);

  // Validate a sample of outputs against a double-precision CPU reference.
  std::vector<float> C((size_t)M * N);
  CL_CHECK(clEnqueueReadBuffer(cl.queue, bc, CL_TRUE, 0, C.size() * 4, C.data(), 0, nullptr, nullptr));
  double max_rel = 0;
  for (int t = 0; t < 256; ++t) {
    int m = rng() % M, n = rng() % N, g = n / SG, l = n % SG;
    double ref = 0;
    for (int s = 0; s < nsb; ++s) {
      const uint32_t* h = &wh[((size_t)g * nsb + s) * 6 * SG];
      float d, dm;
      memcpy(&d, &h[0 * SG + l], 4);
      memcpy(&dm, &h[1 * SG + l], 4);
      for (int b = 0; b < 8; ++b) {
        int sc = (h[(2 + b / 4) * SG + l] >> (8 * (b % 4))) & 0xFF;
        int mn = (h[(4 + b / 4) * SG + l] >> (8 * (b % 4))) & 0xFF;
        int j = b / 2, hi = b % 2;
        for (int i = 0; i < 32; ++i) {
          uint32_t raw = wq[(((size_t)g * nsb + s) * 4 + j) * 8 * SG + (i / 4) * SG + l];
          int q = (raw >> (8 * (i % 4) + 4 * hi)) & 0xF;
          double w = (double)d * sc * q - (double)dm * mn;
          ref += w * asc[(size_t)s * M + m] * aq[(size_t)m * K + s * 256 + b * 32 + i];
        }
      }
    }
    double got = C[(size_t)m * N + n];
    max_rel = std::max(max_rel, std::fabs(got - ref) / (std::fabs(ref) + 1e-3));
  }
  printf("validation: max rel err = %.2e %s\n", max_rel, max_rel < 1e-3 ? "OK" : "FAIL");
  return 0;
}
