// Times the prompt GEMM (zero/kernels/gemm.cl, W8 weights as the just-in-time decoded IQ/K types use, per-32 activation
// scales) in isolation, with the engine's launch geometry, and compares a variant bitwise against the original.
// usage: gemm_w8_bench [M N K qtype iters] [variant defines] [variant kernel file]
//   qtype 100 = W8 per-16 scales, 101 = per-32 scales (default 100); shapes default to the 27B FFN up projection
//   env: GB_COOL=s idle seconds before each measurement (default 5), GB_WGY / GB_RG launch geometry, GB_TM tokens
//        per sub-group of the variant
#include "../src/clrt.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <thread>

static int env(const char* k, int d) { return getenv(k) ? atoi(getenv(k)) : d; }

int main(int argc, char** argv) {
  const int M = argc > 1 ? atoi(argv[1]) : 4096, N = argc > 2 ? atoi(argv[2]) : 17408, K = argc > 3 ? atoi(argv[3]) : 5120;
  const int qtype = argc > 4 ? atoi(argv[4]) : 100, iters = argc > 5 ? atoi(argv[5]) : 10;
  const std::string vdefs = argc > 6 ? argv[6] : "", vfile = argc > 7 ? argv[7] : "zero/kernels/gemm.cl";
  const int SG = 16, nsb = K / 256, G = N / SG, cool = env("GB_COOL", 5);
  if (K % 256 || N % SG || M % 128) { fprintf(stderr, "bad shape\n"); return 1; }
  std::mt19937 rng(42);
  // W8 weights [g][s][j][16 words][lane], headers [g][s][10][lane]: d, dmin, int8 scales (16 per superblock), mins
  std::vector<uint32_t> wq((size_t)G * nsb * 4 * 16 * SG), wh((size_t)G * nsb * 10 * SG);
  for (auto& x : wq) x = rng() & 0x7F7F7F7Fu;  // bytes in -127..127 after a sign flip below
  for (auto& x : wq) x ^= (rng() & 0x80808080u);
  std::uniform_real_distribution<float> ud(0.0005f, 0.002f);
  for (size_t gs = 0; gs < (size_t)G * nsb; ++gs)
    for (int l = 0; l < SG; ++l) {
      uint32_t* h = &wh[gs * 10 * SG];
      const float d = ud(rng);
      memcpy(&h[0 * SG + l], &d, 4);
      h[1 * SG + l] = 0;
      for (int f = 2; f < 6; ++f) h[f * SG + l] = (rng() & 0x3F3F3F3Fu) | 0x01010101u;
      for (int f = 6; f < 10; ++f) h[f * SG + l] = 0;
    }
  std::vector<int8_t> aq((size_t)M * K);
  for (auto& x : aq) x = (int8_t)((int)(rng() % 255) - 127);
  std::vector<float> asc((size_t)(K / 32) * M);
  for (auto& x : asc) x = 0.01f + (rng() % 1000) * 4e-5f;
  std::vector<uint16_t> absum((size_t)(K / 16) * M, 0);

  ClRt cl;
  if (!cl.init()) return 1;
  auto bwq = cl.buffer(CL_MEM_READ_ONLY, wq.size() * 4, wq.data());
  auto bwh = cl.buffer(CL_MEM_READ_ONLY, wh.size() * 4, wh.data());
  auto baq = cl.buffer(CL_MEM_READ_ONLY, aq.size(), aq.data());
  auto basc = cl.buffer(CL_MEM_READ_ONLY, asc.size() * 4, asc.data());
  auto babs = cl.buffer(CL_MEM_READ_ONLY, absum.size() * 2, absum.data());
  auto bc = cl.buffer(CL_MEM_READ_WRITE, (size_t)M * N * 4);

  struct Variant { std::string name, file, defs; int tm; };
  std::vector<Variant> vs = {{"original", "zero/kernels/gemm.cl", "", 16}};
  if (!vdefs.empty() || vfile != "zero/kernels/gemm.cl") vs.push_back({"variant", vfile, vdefs, env("GB_TM", 16)});
  std::vector<std::vector<float>> outs;
  for (const Variant& v : vs) {
    const std::string opts = "-cl-std=CL3.0 -cl-mad-enable -cl-intel-256-GRF-per-thread -DACT32 -DQTYPE=" +
                             std::to_string(qtype) + " -DTM=" + std::to_string(v.tm) + " " + v.defs;
    auto prog = cl.build(read_file(v.file), opts);
    cl_kernel k = cl.kernel(prog, "gemm");
    const int splits = K > 8192 ? (nsb + 16) / 17 : 1, rgroup = env("GB_RG", 4);
    int wgy = 1;
    while (wgy < env("GB_WGY", 8) && (M / v.tm) % (wgy * 2) == 0) wgy *= 2;
    size_t gws[2] = {(size_t)N * (M / v.tm / wgy), (size_t)wgy}, lws[2] = {16, (size_t)wgy};
    auto launch = [&](std::vector<cl_event>* ev) {
      for (int sp = 0; sp < splits; ++sp) {
        int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, acc = sp == 0 ? 0 : 1, ldc = N, coff = 0, MM = M,
            NN = N, KK = K;
        set_args(k, 0, bwq, bwh, baq, basc, babs, bc, MM, NN, KK, ldc, coff, acc, rgroup, s0, s1);
        cl_event e = nullptr;
        CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, ev ? &e : nullptr));
        if (ev) ev->push_back(e);
      }
    };
    launch(nullptr);
    clFinish(cl.queue);
    std::vector<float> c((size_t)M * N);
    CL_CHECK(clEnqueueReadBuffer(cl.queue, bc, CL_TRUE, 0, c.size() * 4, c.data(), 0, nullptr, nullptr));
    outs.push_back(c);
    std::this_thread::sleep_for(std::chrono::seconds(cool));
    std::vector<double> ms;
    for (int it = 0; it < iters; ++it) {
      std::vector<cl_event> ev;
      launch(&ev);
      clFinish(cl.queue);
      double t = 0;
      for (cl_event e : ev) { t += ClRt::event_ms(e); clReleaseEvent(e); }
      ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    const double ops = 2.0 * M * N * K;
    printf("%-8s %s TM=%d spill %zu B: median %.2f ms (min %.2f) -> %.2f TOPS (best %.2f)\n", v.name.c_str(),
           v.defs.c_str(), v.tm, cl.spill_bytes(k), ms[ms.size() / 2], ms[0], ops / ms[ms.size() / 2] / 1e9,
           ops / ms[0] / 1e9);
  }
  if (outs.size() == 2) {
    size_t nd = 0;
    double md = 0;
    for (size_t i = 0; i < outs[0].size(); ++i)
      if (outs[0][i] != outs[1][i]) {
        ++nd;
        md = std::max(md, (double)std::fabs(outs[0][i] - outs[1][i]) / (std::fabs(outs[0][i]) + 1e-6));
      }
    printf("variant vs original: %zu of %zu outputs differ (max relative %.2g)%s\n", nd, outs[0].size(), md,
           nd ? "" : " -> bitwise identical");
  }
  return 0;
}
