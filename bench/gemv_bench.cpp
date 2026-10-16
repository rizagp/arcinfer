// Times and validates the single-token GEMV (kernels/decode.cl "gemv" or a variant with the same arguments) on real
// weights of a GGUF model. A name with %d runs every layer that has the tensor in turn, so the weights stream from
// memory as in token generation instead of staying in the cache.
// usage: gemv_bench model.gguf tensor [defines] [kernel] [lanes_per_rowgroup] [iters] [kernel_file]
//   e.g. gemv_bench g12.gguf token_embd.weight "-DGROWS=2"
//        gemv_bench g12.gguf blk.%d.ffn_gate.weight "-DGROWS=1" gemv 16 20 scratch/decode_try.cl
#include "../src/clrt.h"
#include "../src/quants.h"
#include <algorithm>
#include <cmath>
#include <random>

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: gemv_bench model.gguf tensor [defines] [kernel] [lanes] [iters] [file]\n"); return 1; }
  const std::string tname = argv[2], defs = argc > 3 ? argv[3] : "", kname = argc > 4 ? argv[4] : "gemv";
  const int lanes = argc > 5 ? atoi(argv[5]) : 16, iters = argc > 6 ? atoi(argv[6]) : 10;
  const std::string file = argc > 7 ? argv[7] : "kernels/decode.cl";
  int grows = 4;
  if (auto p = defs.find("-DGROWS="); p != std::string::npos) grows = atoi(defs.c_str() + p + 8);

  GgufFile g(argv[1]);
  std::vector<const GgufTensor*> ts;
  if (tname.find("%d") != std::string::npos) {
    for (int l = 0; l < 1000; ++l) {
      char nm[256];
      snprintf(nm, sizeof nm, tname.c_str(), l);
      const GgufTensor* tp = g.find(nm);
      if (!tp) { if (l > 64) break; continue; }
      const GgufTensor& t = *tp;
      if (!ts.empty() && (t.type != ts[0]->type || t.dims != ts[0]->dims)) continue;  // one type and shape per run
      ts.push_back(&t);
    }
  } else {
    ts.push_back(&g.get(tname));
  }
  if (ts.empty()) { fprintf(stderr, "tensor not found\n"); return 1; }
  const uint32_t type = ts[0]->type;
  const int K = (int)ts[0]->dims[0], N = (int)ts[0]->dims[1], nsb = K / 256, M = 16;
  const size_t bb = gguf_sb_bytes(type), bp = raw_padded_sb_bytes(type);
  printf("%s: %s [%d x %d], %zu matrices, %.1f MB each\n", tname.c_str(), ggml_type_name(type), N, K, ts.size(),
         ts[0]->nbytes / 1e6);

  ClRt cl;
  if (!cl.init()) return 1;
  if (getenv("TOUCHALL")) {  // read the whole mapped file first, as the engine's load does
    uint64_t sum = 0;
    for (auto& t : g.tensors)
      for (uint64_t i = 0; i < t.nbytes; i += 4096) sum += t.data[i];
    printf("touched the file (%llu)\n", (unsigned long long)sum);
  }
  std::vector<cl_mem> raws, junk;
  if (const char* e = getenv("BIGALLOC_GB")) {  // simulate the engine's memory footprint (allocated before the weights)
    for (int i = 0; i < atoi(e) * 4; ++i) {
      cl_mem m = cl.buffer(CL_MEM_READ_WRITE, (size_t)256 << 20);
      const uint32_t z = 1;
      CL_CHECK(clEnqueueFillBuffer(cl.queue, m, &z, 4, 0, (size_t)256 << 20, 0, nullptr, nullptr));
      junk.push_back(m);
    }
    clFinish(cl.queue);
  }
  for (auto* t : ts) {
    std::vector<uint8_t> pad((size_t)N * nsb * bp + 256, 0);
    for (size_t i = 0; i < (size_t)N * nsb; ++i) gpu_raw_sb(t->type, t->data + i * bb, &pad[i * bp]);
    if (getenv("HOSTPTR")) {  // the engine's upload: CL_MEM_ALLOC_HOST_PTR + map
      cl_int err;
      cl_mem m = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, pad.size(), nullptr, &err);
      CL_CHECK(err);
      void* dst = clEnqueueMapBuffer(cl.queue, m, CL_TRUE, CL_MAP_WRITE, 0, pad.size(), 0, nullptr, nullptr, &err);
      CL_CHECK(err);
      memcpy(dst, pad.data(), pad.size());
      clEnqueueUnmapMemObject(cl.queue, m, dst, 0, nullptr, nullptr);
      raws.push_back(m);
    } else {
      raws.push_back(cl.buffer(CL_MEM_READ_ONLY, pad.size(), pad.data()));
    }
  }
  // activations: int8 per 32 with float scales (asc [K/32][M]), per-16 sums (absum [K/16][M]); token 0 only
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(K), xq(K);
  for (auto& v : x) v = nd(rng) * (1.0f + 3.0f * (rng() % 97 == 0));
  std::vector<int8_t> aq((size_t)M * K, 0);
  std::vector<float> asc((size_t)(K / 32) * M, 0.0f);
  std::vector<uint16_t> absum((size_t)(K / 16) * M, 0);
  for (int b = 0; b < K / 32; ++b) {
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[32 * b + i]));
    const float sc = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0.0f;
    asc[(size_t)b * M] = sc;
    for (int h = 0; h < 2; ++h) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) {
        const int k = 32 * b + 16 * h + i, q = (int)std::nearbyint(x[k] * inv);
        aq[k] = (int8_t)q;
        xq[k] = q * sc;
        sum += q;
      }
      absum[(size_t)(2 * b + h) * M] = (uint16_t)(int16_t)sum;
    }
  }
  auto tabs = build_iq_tables();
  cl_mem d_tabs = cl.buffer(CL_MEM_READ_ONLY, tabs.size() * 4, tabs.data());
  cl_mem d_aq = cl.buffer(CL_MEM_READ_ONLY, aq.size(), aq.data());
  cl_mem d_asc = cl.buffer(CL_MEM_READ_ONLY, asc.size() * 4, asc.data());
  cl_mem d_abs = cl.buffer(CL_MEM_READ_ONLY, absum.size() * 2, absum.data());
  cl_mem d_c = cl.buffer(CL_MEM_READ_WRITE, (size_t)N * 4);

  const std::string src = read_file(file);
  cl_program prog = cl.build(src, "-cl-std=CL3.0 -cl-mad-enable -DQT=" + std::to_string(type) + " " + defs);
  cl_kernel k = cl.kernel(prog, kname.c_str());
  printf("kernel %s %s: spill %zu bytes\n", kname.c_str(), defs.c_str(), cl.spill_bytes(k));
  const size_t lws = 256, gws = ((size_t)(N + grows - 1) / grows * lanes + lws - 1) / lws * lws;
  const int c_off = 0, acc = 0;
  auto launch = [&](cl_mem raw, cl_event* e) {
    set_args(k, 0, raw, N, K, d_tabs, d_aq, d_asc, d_abs, M, d_c, c_off, acc);
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 1, nullptr, &gws, &lws, 0, nullptr, e));
  };
  // correctness on the first matrix against the CPU dequantization
  launch(raws[0], nullptr);
  std::vector<float> c(N);
  CL_CHECK(clEnqueueReadBuffer(cl.queue, d_c, CL_TRUE, 0, (size_t)N * 4, c.data(), 0, nullptr, nullptr));
  if (const char* other = getenv("GB_CMP")) {  // GB_CMP=kernel[:defines]: bitwise comparison with another kernel
    std::string on = other, od = defs;
    if (const size_t colon = on.find(':'); colon != std::string::npos) { od = on.substr(colon + 1); on.resize(colon); }
    cl_program p2 = cl.build(src, "-cl-std=CL3.0 -cl-mad-enable -DQT=" + std::to_string(type) + " " + od);
    cl_kernel k2 = cl.kernel(p2, on.c_str());
    int g2 = 4;
    if (auto p = od.find("-DGROWS="); p != std::string::npos) g2 = atoi(od.c_str() + p + 8);
    const size_t gws2 = ((size_t)(N + g2 - 1) / g2 * lanes + lws - 1) / lws * lws;
    std::vector<float> c2(N);
    for (size_t m = 0; m < raws.size(); ++m) {
      launch(raws[m], nullptr);
      CL_CHECK(clEnqueueReadBuffer(cl.queue, d_c, CL_TRUE, 0, (size_t)N * 4, c.data(), 0, nullptr, nullptr));
      set_args(k2, 0, raws[m], N, K, d_tabs, d_aq, d_asc, d_abs, M, d_c, c_off, acc);
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k2, 1, nullptr, &gws2, &lws, 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl.queue, d_c, CL_TRUE, 0, (size_t)N * 4, c2.data(), 0, nullptr, nullptr));
      const size_t diff = memcmp(c.data(), c2.data(), (size_t)N * 4) ? 1 : 0;
      if (diff) {
        int nd = 0;
        for (int n = 0; n < N; ++n) nd += c[n] != c2[n];
        printf("matrix %zu: %d of %d outputs differ from %s\n", m, nd, N, on.c_str());
      }
    }
    printf("bitwise comparison with %s (%s) over %zu matrices done\n", on.c_str(), od.c_str(), raws.size());
    launch(raws[0], nullptr);
    CL_CHECK(clEnqueueReadBuffer(cl.queue, d_c, CL_TRUE, 0, (size_t)N * 4, c.data(), 0, nullptr, nullptr));
  }
  double maxerr = 0, maxref = 0;
  std::vector<float> row(K);
  const size_t rb = (size_t)nsb * bb;
  for (int n = 0; n < N; n += std::max(1, N / 4096)) {
    dequant_row(type, ts[0]->data + (size_t)n * rb, K, row.data());
    double s = 0;
    for (int i = 0; i < K; ++i) s += (double)row[i] * xq[i];
    maxerr = std::max(maxerr, std::fabs(s - c[n]));
    maxref = std::max(maxref, std::fabs(s));
  }
  printf("max |err| %.3g (max |ref| %.3g, rel %.2e)\n", maxerr, maxref, maxerr / std::max(maxref, 1e-30));
  // timing: every matrix in turn, iters rounds
  std::vector<double> per;
  for (int it = 0; it < iters + 1; ++it) {
    double ms = 0;
    for (auto r : raws) {
      cl_event e;
      launch(r, &e);
      ms += ClRt::event_ms(e);
      clReleaseEvent(e);
    }
    if (it > 0) per.push_back(ms / raws.size());
  }
  std::sort(per.begin(), per.end());
  const double med = per[per.size() / 2], bytes = (double)N * nsb * bb;
  printf("median %.3f ms per matrix (min %.3f) -> %.1f GB/s\n", med, per[0], bytes / med / 1e6);
  return 0;
}
