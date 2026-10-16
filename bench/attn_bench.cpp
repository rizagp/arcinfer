// Prefill attention micro-benchmark (kernels/ops.cl attn_q8) at any head shape, with an optional sliding window.
// Runs the int8 attention over an N-token prompt in chunks (like the engine) and checks sampled output rows
// against a CPU reference computed from the same int8 q/k/v.
// usage: attn_bench [N=11264] [chunk=4096] [extra build options...]
// env: ATT_NH (q heads, 24), ATT_NKV (kv heads, 4), ATT_HD (head dim, 256), ATT_WIN (window, 0 = causal),
//      ATT_HS / ATT_TPW (tiling; default as the engine), ATT_GRF128 (default register file),
//      ATT_P0 (only the queries from this position on: the cost at depth of a long prompt without running all of it)
// e.g. Gemma 4 12B global layers: ATT_NH=16 ATT_NKV=1 ATT_HD=512; its sliding layers: ATT_NKV=8 ATT_WIN=1024
#include "../src/clrt.h"
#include "../src/quants.h"
#include <cmath>
#include <random>
#include <cstring>

static int env(const char* k, int def) { return getenv(k) ? atoi(getenv(k)) : def; }

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 11264, chunk = argc > 2 ? atoi(argv[2]) : 4096;
  std::string extra;
  for (int i = 3; i < argc; ++i) extra += std::string(" ") + argv[i];
  const int nh = env("ATT_NH", 24), nkv = env("ATT_NKV", 4), hd = env("ATT_HD", 256), win = env("ATT_WIN", 0);
  const int hpg = nh / nkv, ctx = (N + 255) / 256 * 256, P0 = std::min(env("ATT_P0", 0), N - 1), NQ = N - P0;
  // tiling as Engine::attn_hs_for: largest of 3, 2, 1 dividing hpg with a power-of-two quotient <= 32
  int hs = 1;
  for (int h : {3, 2, 1})
    if (hpg % h == 0 && ((hpg / h) & (hpg / h - 1)) == 0 && hpg / h <= 32) { hs = h; break; }
  hs = env("ATT_HS", hs);
  const int nsp = hpg / hs, tpw = env("ATT_TPW", 32 / nsp);
  const int KB = env("ATT_SB", hd / 64), BD = hd / KB;  // scale blocks per row (ATT_SB=1: one per row)
  // ATT_PS=dim groups,heads,tokens: attn_ps (head dims split over sub-groups) instead of attn_q8
  int pns = 0, phg = 0, ptt = 0;
  if (getenv("ATT_PS")) sscanf(getenv("ATT_PS"), "%d,%d,%d", &pns, &phg, &ptt);
  ClRt cl;
  if (!cl.init()) return 1;
  std::string opts = std::string(getenv("ATT_GRF128") ? "" : "-cl-intel-256-GRF-per-thread ") +
                     "-cl-std=CL3.0 -cl-mad-enable -DHD=" + std::to_string(hd) + " -DHPG=" + std::to_string(hpg) +
                     " -DHS=" + std::to_string(hs) + " -DTPW=" + std::to_string(tpw) + " -DSB=" + std::to_string(KB) + extra;
  if (pns) opts += " -DPNS=" + std::to_string(pns) + " -DPHG=" + std::to_string(phg) + " -DPTT=" + std::to_string(ptt);
  auto prog = cl.build(read_file("kernels/ops.cl"), opts);
  cl_kernel k = cl.kernel(prog, pns ? "attn_ps" : "attn_q8");
  if (pns)
    printf("nh %d nkv %d hd %d win %d | attn_ps dim groups %d, %d heads x %d tokens per sub-group, TPW %d (%d work-items)%s: "
           "spill %zu bytes\n", nh, nkv, hd, win, pns, phg, ptt, tpw, 16 * (hpg / phg) * (tpw / ptt) * pns, extra.c_str(),
           cl.spill_bytes(k));
  else
    printf("nh %d nkv %d hd %d win %d | HS %d NSP %d TPW %d (%d work-items)%s: spill %zu bytes\n", nh, nkv, hd, win, hs, nsp,
           tpw, 16 * nsp * tpw, extra.c_str(), cl.spill_bytes(k));

  // random q/k/v (int8 + per-row scales) shaped like real data
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<int8_t> qi((size_t)NQ * nh * hd), kq((size_t)nkv * ctx * hd, 0), vq((size_t)nkv * ctx * hd, 0);
  std::vector<float> qsc((size_t)NQ * nh * KB), ksc((size_t)nkv * ctx * KB, 0.0f), vsc((size_t)nkv * ctx * KB, 0.0f);
  std::vector<float> kscb((size_t)nkv * ctx * KB, 0.0f), vscb((size_t)nkv * ctx * KB, 0.0f);  // [blk][KB][16]
  // blocked layouts of the kernels: K [blk][HD/4 words][16 keys] uints, V [blk][HD dims][16 keys] bytes
  std::vector<uint32_t> kqb((size_t)nkv * ctx * hd / 4, 0);
  std::vector<uint8_t> vqb((size_t)nkv * ctx * hd, 0);
  std::vector<float> v(hd);
  // rows with a few large dims (as Gemma's V rows), int8 with an absmax scale per 64-dim block
  auto fill_row = [&](int8_t* dst, float* sc, float mag) {
    for (int i = 0; i < hd; ++i) v[i] = nd(rng) * mag * (i % 97 == 5 ? 8.0f : 1.0f);
    for (int c = 0; c < KB; ++c) {
      float amax = 0.0f;
      for (int i = BD * c; i < BD * c + BD; ++i) amax = std::max(amax, std::fabs(v[i]));
      for (int i = BD * c; i < BD * c + BD; ++i) dst[i] = (int8_t)std::lrint(v[i] * 127.0f / amax);
      sc[c] = amax / 127.0f;
    }
  };
  for (size_t r = 0; r < (size_t)NQ * nh; ++r) fill_row(&qi[r * hd], &qsc[r * KB], 0.15f * std::sqrt(256.0f / hd));
  for (int g = 0; g < nkv; ++g)
    for (int t = 0; t < N; ++t) {
      int8_t* kr = &kq[((size_t)g * ctx + t) * hd];
      int8_t* vr = &vq[((size_t)g * ctx + t) * hd];
      fill_row(kr, &ksc[((size_t)g * ctx + t) * KB], 1.0f);
      fill_row(vr, &vsc[((size_t)g * ctx + t) * KB], 1.0f);
      const size_t blk = (size_t)g * (ctx / 16) + t / 16;
      for (int c = 0; c < KB; ++c) {
        kscb[(blk * KB + c) * 16 + t % 16] = ksc[((size_t)g * ctx + t) * KB + c];
        vscb[(blk * KB + c) * 16 + t % 16] = vsc[((size_t)g * ctx + t) * KB + c];
      }
      for (int w = 0; w < hd / 4; ++w) memcpy(&kqb[(blk * (hd / 4) + w) * 16 + t % 16], kr + 4 * w, 4);
      for (int d = 0; d < hd; ++d) vqb[(blk * hd + d) * 16 + t % 16] = (uint8_t)vr[d];
    }
  const size_t lsz = pns ? (size_t)16 * (hpg / phg) * (tpw / ptt) * pns : (size_t)16 * nsp * tpw;
  const int maxT = std::min(chunk, N);
  cl_mem b_qi = cl.buffer(CL_MEM_READ_ONLY, qi.size()), b_qsc = cl.buffer(CL_MEM_READ_ONLY, qsc.size() * 4);
  cl_mem b_kq = cl.buffer(CL_MEM_READ_ONLY, kqb.size() * 4, kqb.data()), b_ksc = cl.buffer(CL_MEM_READ_ONLY, kscb.size() * 4, kscb.data());
  cl_mem b_vq = cl.buffer(CL_MEM_READ_ONLY, vqb.size(), vqb.data()), b_vsc = cl.buffer(CL_MEM_READ_ONLY, vscb.size() * 4, vscb.data());
  std::vector<float> out((size_t)NQ * nh * hd);  // rows P0..N-1
  cl_mem b_out = cl.buffer(CL_MEM_READ_WRITE, (size_t)maxT * nh * hd * 4);

  double total = 0.0;
  for (int rep = 0; rep < 2; ++rep) {
    total = 0.0;
    for (int p0 = P0; p0 < N; p0 += chunk) {
      const int T = std::min(chunk, N - p0);
      clEnqueueWriteBuffer(cl.queue, b_qi, CL_TRUE, 0, (size_t)T * nh * hd, &qi[(size_t)(p0 - P0) * nh * hd], 0, nullptr, nullptr);
      clEnqueueWriteBuffer(cl.queue, b_qsc, CL_TRUE, 0, (size_t)T * nh * KB * 4, &qsc[(size_t)(p0 - P0) * nh * KB], 0, nullptr, nullptr);
      int cx = ctx, nhh = nh, pp = p0, TT = T, w = win, nlo = -1, nhi = -1;
      if (pns) set_args(k, 0, b_qi, b_qsc, b_kq, b_ksc, b_vq, b_vsc, cx, nhh, pp, TT, b_out, w);
      else set_args(k, 0, b_qi, b_qsc, b_kq, b_ksc, b_vq, b_vsc, cx, nhh, pp, TT, b_out, w, nlo, nhi);
      size_t gws[2] = {(size_t)(T + tpw - 1) / tpw * lsz, (size_t)nkv}, lws[2] = {lsz, 1};
      cl_event e;
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, &e));
      total += ClRt::event_ms(e);
      clReleaseEvent(e);
      if (rep == 1)
        clEnqueueReadBuffer(cl.queue, b_out, CL_TRUE, 0, (size_t)T * nh * hd * 4, &out[(size_t)(p0 - P0) * nh * hd], 0, nullptr, nullptr);
    }
  }
  if (pns && getenv("ATT_CMP")) {  // attn_ps against attn_q8 (default tiling) on the same data
    const int tq = 32 / nsp;
    const std::string o2 = std::string(getenv("ATT_GRF128") ? "" : "-cl-intel-256-GRF-per-thread ") +
                           "-cl-std=CL3.0 -cl-mad-enable -DHD=" + std::to_string(hd) + " -DHPG=" + std::to_string(hpg) +
                           " -DHS=" + std::to_string(hs) + " -DTPW=" + std::to_string(tq) + " -DSB=" + std::to_string(KB);
    cl_kernel k2 = cl.kernel(cl.build(read_file("kernels/ops.cl"), o2), "attn_q8");
    std::vector<float> o_q8((size_t)maxT * nh * hd);
    size_t ndiff = 0, nval = 0;
    double md = 0.0;
    for (int p0 = P0; p0 < N; p0 += chunk) {
      const int T = std::min(chunk, N - p0);
      clEnqueueWriteBuffer(cl.queue, b_qi, CL_TRUE, 0, (size_t)T * nh * hd, &qi[(size_t)(p0 - P0) * nh * hd], 0, nullptr, nullptr);
      clEnqueueWriteBuffer(cl.queue, b_qsc, CL_TRUE, 0, (size_t)T * nh * KB * 4, &qsc[(size_t)(p0 - P0) * nh * KB], 0, nullptr, nullptr);
      int cx = ctx, nhh = nh, pp = p0, TT = T, w = win, nlo = -1, nhi = -1;
      set_args(k2, 0, b_qi, b_qsc, b_kq, b_ksc, b_vq, b_vsc, cx, nhh, pp, TT, b_out, w, nlo, nhi);
      size_t gws[2] = {(size_t)(T + tq - 1) / tq * 16 * nsp * tq, (size_t)nkv}, lws[2] = {(size_t)16 * nsp * tq, 1};
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k2, 2, nullptr, gws, lws, 0, nullptr, nullptr));
      clEnqueueReadBuffer(cl.queue, b_out, CL_TRUE, 0, (size_t)T * nh * hd * 4, o_q8.data(), 0, nullptr, nullptr);
      for (size_t i = 0; i < (size_t)T * nh * hd; ++i) {
        const float a = out[(size_t)(p0 - P0) * nh * hd + i], b = o_q8[i];
        ndiff += memcmp(&a, &b, 4) != 0;
        md = std::max(md, (double)std::fabs(a - b));
        ++nval;
      }
    }
    printf("attn_ps vs attn_q8: %zu of %zu values differ, max abs difference %.3g%s\n", ndiff, nval, md,
           ndiff == 0 ? " (bit-identical)" : "");
  }
  double pairs = 0.0;  // (query, key) pairs
  for (int m = P0; m < N; ++m) pairs += win > 0 ? std::min(m + 1, win) : m + 1;
  const double macs = 2.0 * hd * nh * pairs;
  printf("N=%d chunk=%d queries %d..%d: %.1f ms per layer, %.2f T MAC/s effective\n", N, chunk, P0, N - 1, total,
         macs / (total * 1e-3) / 1e12);

  // sampled rows against the CPU reference
  double max_err = 0.0, max_ref = 0.0, se = 0.0, sr = 0.0;
  for (int s = 0; s < 48; ++s) {
    const int m = P0 + (s < 4 ? (s == 0 ? 0 : s == 1 ? NQ - 1 : s == 2 ? chunk - 1 : chunk) : (int)(rng() % NQ)), h = (int)(rng() % nh);
    if (m >= N) continue;
    const int g = h / hpg, t_lo = win > 0 ? std::max(0, m - win + 1) : 0;
    std::vector<double> sc(m + 1, 0.0);
    double mx = -1e30;
    for (int t = t_lo; t <= m; ++t) {
      double s2 = 0;
      for (int c = 0; c < KB; ++c) {
        int acc = 0;
        for (int d = BD * c; d < BD * c + BD; ++d) acc += qi[((size_t)(m - P0) * nh + h) * hd + d] * kq[((size_t)g * ctx + t) * hd + d];
        s2 += acc * (double)qsc[((size_t)(m - P0) * nh + h) * KB + c] * ksc[((size_t)g * ctx + t) * KB + c];
      }
      sc[t] = s2;
      mx = std::max(mx, sc[t]);
    }
    double L = 0.0;
    for (int t = t_lo; t <= m; ++t) { sc[t] = std::exp(sc[t] - mx); L += sc[t]; }
    for (int d = 0; d < hd; ++d) {
      double o = 0.0;
      for (int t = t_lo; t <= m; ++t) o += sc[t] * vq[((size_t)g * ctx + t) * hd + d] * (double)vsc[((size_t)g * ctx + t) * KB + d / BD];
      o /= L;
      const double e = o - out[((size_t)(m - P0) * nh + h) * hd + d];
      max_err = std::max(max_err, std::fabs(e));
      max_ref = std::max(max_ref, std::fabs(o));
      se += e * e;
      sr += o * o;
    }
  }
  printf("max abs error %.3g (max |ref| %.3g), rel rms error %.3g %s\n", max_err, max_ref, std::sqrt(se / sr),
         max_err < 1e-2 * std::max(1.0, max_ref) ? "OK" : "MISMATCH");
  return 0;
}
