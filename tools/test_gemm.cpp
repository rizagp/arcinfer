// Validates repacking / JIT decoding + GEMM kernels on real GGUF tensors against a CPU reference, and times them.
// Each tensor is run through every applicable path: native (repacked) and JIT-W8, each with per-256 and
// per-32 (ACT32) activation scales.
// usage: test_gemm model.gguf M tensor_name [tensor_name ...]
#include "../src/clrt.h"
#include "../src/quants.h"
#include <algorithm>
#include <map>
#include <random>
#include <chrono>

struct Ctx {
  ClRt cl;
  std::string gsrc, dsrc, g8src;
  std::map<std::string, cl_kernel> cache;
  cl_mem tabs = nullptr;
  cl_kernel kernel(const std::string& file, const std::string& opts, const char* name) {
    auto key = file + opts + "|" + name;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto p = cl.build(file == "gemm" ? gsrc : file == "gemm8" ? g8src : dsrc, "-cl-std=CL3.0 -cl-mad-enable " + opts);
    return cache[key] = cl.kernel(p, name);
  }
};

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: test_gemm model.gguf M tensor...\n"); return 1; }
  GgufFile f(argv[1]);
  const int M = atoi(argv[2]);
  Ctx X;
  if (!X.cl.init()) return 1;
  X.gsrc = read_file("kernels/gemm.cl");
  X.dsrc = read_file("kernels/decode.cl");
  X.g8src = read_file("kernels/gemm8.cl");
  auto tabs = build_iq_tables();
  X.tabs = X.cl.buffer(CL_MEM_READ_ONLY, tabs.size() * 4, tabs.data());
  bool all_ok = true;

  for (int ti = 3; ti < argc; ++ti) {
    const GgufTensor& t = f.get(argv[ti]);
    const int K = (int)t.dims[0], N = (int)t.dims[1];
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t)M * K);
    for (auto& v : x) v = nd(rng);
    for (int i = 0; i < M; ++i) x[(size_t)i * K + (rng() % K)] *= 20.0f;  // a few outliers, like real activations

    // CPU reference rows for sampled outputs
    std::vector<std::pair<int, int>> samples;
    for (int i = 0; i < 192; ++i) samples.push_back({(int)(rng() % M), (int)(rng() % N)});
    std::map<int, std::vector<float>> wrows;
    for (auto& [m, n] : samples)
      if (!wrows.count(n)) {
        wrows[n].resize(K);
        dequant_row(t.type, t.data + (size_t)n * (t.nbytes / N), K, wrows[n].data());
      }

    for (int act32 = 0; act32 < 2; ++act32) {
      std::vector<int8_t> aq((size_t)M * K);
      std::vector<float> asc((size_t)(K / (act32 ? 32 : 256)) * M);
      std::vector<uint16_t> absum((size_t)(K / 16) * M);
      quantize_act_cpu(x.data(), M, K, aq.data(), asc.data(), absum.data(), act32);
      auto baq = X.cl.buffer(CL_MEM_READ_ONLY, aq.size(), aq.data());
      auto basc = X.cl.buffer(CL_MEM_READ_ONLY, asc.size() * 4, asc.data());
      auto babs = X.cl.buffer(CL_MEM_READ_ONLY, absum.size() * 2, absum.data());
      auto bc = X.cl.buffer(CL_MEM_READ_WRITE, (size_t)M * N * 4);
      const std::string aopt = std::string(getenv("ARC_GOPTS") ? std::string(" ") + getenv("ARC_GOPTS") : "") + (act32 ? " -DACT32" : "");

      for (int path = 0; path < 2; ++path) {  // 0 = native, 1 = JIT W8
        if (path == 0 && !is_native_type(t.type)) continue;
        if (path == 1 && (!is_w8_type(t.type) || !act32 || t.type == GGML_Q8_0)) continue;
        cl_mem bwq, bwh;
        double prep_ms = 0;
        int qtype;
        if (path == 0) {
          RepackedWeight rw;
          repack_rows(t.type, t.data, K, 0, N, rw);
          bwq = X.cl.buffer(CL_MEM_READ_ONLY, rw.wq.size() * 4, rw.wq.data());
          bwh = X.cl.buffer(CL_MEM_READ_ONLY, rw.wh.size() * 4, rw.wh.data());
          qtype = (int)t.type;
        } else {
          auto padded = pad_raw_rows(t.type, t.data, t.nbytes);
          auto braw = X.cl.buffer(CL_MEM_READ_ONLY, padded.size(), padded.data());
          bwq = X.cl.buffer(CL_MEM_READ_WRITE, w8_bytes_q(N, K));
          bwh = X.cl.buffer(CL_MEM_READ_WRITE, w8_bytes_h(N, K));
          const bool sbk = !getenv("ARC_DEC32") || w8_requant(t.type);
          auto kd = X.kernel("decode", "-DQT=" + std::to_string(t.type) + (getenv("ARC_G8") ? " -DGL=8" : ""), sbk ? "decode_w8_sb" : "decode_w8");
          set_args(kd, 0, braw, N, K, X.tabs, bwq, bwh);
          size_t lx = 256;
          while (N % lx) lx /= 2;
          size_t gws[2] = {(size_t)N, (size_t)K / (sbk ? 256 : 32)}, lws[2] = {sbk ? lx : 16, 1};
          for (int it = 0; it < 3; ++it) {
            cl_event e;
            CL_CHECK(clEnqueueNDRangeKernel(X.cl.queue, kd, 2, nullptr, gws, lws, 0, nullptr, &e));
            prep_ms = ClRt::event_ms(e);
            clReleaseEvent(e);
          }
          clReleaseMemObject(braw);
          qtype = w8_variant(t.type);
        }
        const int TN = getenv("ARC_TN") ? atoi(getenv("ARC_TN")) : 2, TMv = getenv("ARC_TM") ? atoi(getenv("ARC_TM")) : 16;
        const int WGYmax = getenv("ARC_WGY") ? atoi(getenv("ARC_WGY")) : 8;
        cl_kernel k;
        size_t gws[2], lws[2];
        int wgy = 1;
        if (path == 0) {
          k = X.kernel("gemm", "-cl-intel-256-GRF-per-thread -DCG=" + std::string(getenv("ARC_CG") ? getenv("ARC_CG") : "1") + " -DQTYPE=" + std::to_string(qtype) + aopt, "gemm");
          while (wgy < WGYmax && (M / 16) % (wgy * 2) == 0) wgy *= 2;
          gws[0] = (size_t)N * (M / 16 / wgy); gws[1] = wgy; lws[0] = 16 * (getenv("ARC_CG") ? atoi(getenv("ARC_CG")) : 1); lws[1] = wgy;
        } else if (!getenv("ARC_G8")) {
          k = X.kernel("gemm", "-cl-intel-256-GRF-per-thread -DTM=" + std::to_string(TMv) + " -DCG=" + std::string(getenv("ARC_CG") ? getenv("ARC_CG") : "1") + " -DQTYPE=" + std::to_string(100 + qtype) + aopt, "gemm");
          while (wgy < WGYmax && (M / TMv) % (wgy * 2) == 0) wgy *= 2;
          gws[0] = (size_t)N * (M / TMv / wgy); gws[1] = wgy; lws[0] = 16 * (getenv("ARC_CG") ? atoi(getenv("ARC_CG")) : 1); lws[1] = wgy;
        } else {
          k = X.kernel("gemm8", "-cl-intel-256-GRF-per-thread -DVARIANT=" + std::to_string(qtype) + " -DTN=" +
                                    std::to_string(TN) + " -DTM=" + std::to_string(TMv), "gemm8");
          while (wgy < WGYmax && (M / TMv) % (wgy * 2) == 0) wgy *= 2;
          gws[0] = (size_t)N / (8 * TN) * 8; gws[1] = M / TMv; lws[0] = 8; lws[1] = wgy;
        }
        int acc = 0, c_off = 0, rgroup = getenv("ARC_RG") ? atoi(getenv("ARC_RG")) : 1;
        const int kdiv = getenv("ARC_KDIV") ? atoi(getenv("ARC_KDIV")) : 1;  // split-K pieces
        auto set_split = [&](int sp) {
          int s0 = K / 256 * sp / kdiv, s1 = K / 256 * (sp + 1) / kdiv, ac = sp ? 1 : acc;
          set_args(k, 0, bwq, bwh, baq, basc, babs, bc, M, N, K, N, c_off, ac, rgroup, s0, s1);
        };
        if (path == 0 || !getenv("ARC_G8")) set_split(0);
        else set_args(k, 0, bwq, bwh, baq, basc, babs, bc, M, N, K, N, c_off, acc);
        double best = 1e30;
        for (int it = 0; it < 6; ++it) {
          cl_event e;
          CL_CHECK(clEnqueueNDRangeKernel(X.cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, &e));
          double t = ClRt::event_ms(e);
          for (int kk = 1; kk < kdiv; ++kk) { clReleaseEvent(e); set_split(kk); CL_CHECK(clEnqueueNDRangeKernel(X.cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, &e)); t += ClRt::event_ms(e); }
          if (kdiv > 1) set_split(0);
          best = std::min(best, t);
          clReleaseEvent(e);
        }
        std::vector<float> C((size_t)M * N);
        CL_CHECK(clEnqueueReadBuffer(X.cl.queue, bc, CL_TRUE, 0, C.size() * 4, C.data(), 0, nullptr, nullptr));
        double eq_num = 0, eq_den = 0, ef_num = 0, ef_den = 0;
        const int B = act32 ? 32 : 256;
        for (auto& [m, n] : samples) {
          const auto& w = wrows[n];
          double rq = 0, rf = 0;
          for (int i = 0; i < K; ++i) {
            rq += (double)w[i] * asc[(size_t)(i / B) * M + m] * aq[(size_t)m * K + i];
            rf += (double)w[i] * x[(size_t)m * K + i];
          }
          const double got = C[(size_t)m * N + n];
          eq_num += (got - rq) * (got - rq); eq_den += rq * rq;
          ef_num += (got - rf) * (got - rf); ef_den += rf * rf;
        }
        const double eq = std::sqrt(eq_num / eq_den), ef = std::sqrt(ef_num / ef_den);
        const bool ok = eq < (path == 1 && w8_requant(t.type) ? 1e-2 : 1e-4);  // requantized types are not exact
        all_ok &= ok;
        printf("%-26s %-7s %-6s %-5s N=%-6d K=%-5d | spill %4zu | prep %6.2f ms | gemm %7.3f ms %5.2f TOPS | q-ref err %.1e %s | f32-ref %.2e\n",
               t.name.c_str(), ggml_type_name(t.type), path ? "jit-w8" : "native", act32 ? "act32" : "act256", N, K, X.cl.spill_bytes(k), prep_ms,
               best, 2.0 * M * N * K / best * 1e-9, eq, ok ? "OK" : "FAIL", ef);
        clReleaseMemObject(bwq);
        clReleaseMemObject(bwh);
      }
      if (act32) {
        // GEMV on raw rows (token 0), checked against the same reference
        auto padded = pad_raw_rows(t.type, t.data, t.nbytes);
        auto braw = X.cl.buffer(CL_MEM_READ_ONLY, padded.size(), padded.data());
        const int GR = getenv("ARC_GROWS") ? atoi(getenv("ARC_GROWS")) : 4;
        auto kg = X.kernel("decode", "-DQT=" + std::to_string(t.type) + " -DGROWS=" + std::to_string(GR), "gemv");
        int c_off = 0, acc = 0;
        set_args(kg, 0, braw, N, K, X.tabs, baq, basc, babs, M, bc, c_off, acc);
        size_t gws = (size_t)(N + GR - 1) / GR * 16, lws = 256;
        gws = (gws + 255) / 256 * 256;
        double best = 1e30;
        for (int it = 0; it < 6; ++it) {
          cl_event e;
          CL_CHECK(clEnqueueNDRangeKernel(X.cl.queue, kg, 1, nullptr, &gws, &lws, 0, nullptr, &e));
          double t = ClRt::event_ms(e);
          best = std::min(best, t);
          clReleaseEvent(e);
        }
        std::vector<float> C(N);
        CL_CHECK(clEnqueueReadBuffer(X.cl.queue, bc, CL_TRUE, 0, (size_t)N * 4, C.data(), 0, nullptr, nullptr));
        double num = 0, den = 0;
        for (auto& [m, n] : samples) {
          const auto& w = wrows[n];
          double rq = 0;
          for (int i = 0; i < K; ++i) rq += (double)w[i] * asc[(size_t)(i / 32) * M] * aq[i];
          num += (C[n] - rq) * (C[n] - rq); den += rq * rq;
        }
        const double e = std::sqrt(num / den);
        all_ok &= e < 1e-4;
        printf("%-26s %-7s gemv   act32 N=%-6d K=%-5d | %7.3f ms %6.1f GB/s | q-ref err %.1e %s\n", t.name.c_str(),
               ggml_type_name(t.type), N, K, best, t.nbytes / best * 1e-6, e, e < 1e-4 ? "OK" : "FAIL");
        clReleaseMemObject(braw);
      }
      for (cl_mem b : {baq, basc, babs, bc}) clReleaseMemObject(b);
    }
  }
  printf("%s\n", all_ok ? "ALL OK" : "SOME FAILED");
  return all_ok ? 0 : 2;
}
