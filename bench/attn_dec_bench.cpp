// Decode attention micro-benchmark: the engine's flash-decoding kernels (attn_dsplit, attn_split or attn_decode,
// then attn_combine) for T query rows at positions P-T+1 .. P against an int8 K/V cache, checked against a CPU
// reference computed from the same int8 values and scales.
// usage: attn_dec_bench [P=14000] [extra build options...]
// env: ATT_NH (q heads, 16), ATT_NKV (kv heads, 1), ATT_HD (head dim, 512), ATT_WIN (window, 0 = full),
//      ATT_SB (scale blocks per row, HD/64), ATT_T (query rows, 1), ATT_FILE (kernel source, zero/kernels/ops.cl),
//      ATT_ITERS (20), ATT_LAYERS (distinct caches cycled through so they stream from memory, 8),
//      ATT_DSPLIT=HG,NSUB (attn_dsplit with HG heads per sub-group, NSUB dim groups), ATT_OLD (the attn_split /
//      attn_decode path: ATT_SPLIT / ATT_DECODE force one, ATT_HS heads per sub-group), ATT_SMAX / ATT_TARGET /
//      ATT_SLEN (split length choice)
// e.g. Gemma 4 12B global layers: (defaults); its sliding layers: ATT_HD=256 ATT_NKV=8 ATT_WIN=1024
#include "../src/clrt.h"
#include <algorithm>
#include <cmath>
#include <random>

static int env(const char* k, int def) { return getenv(k) ? atoi(getenv(k)) : def; }

int main(int argc, char** argv) {
  const int P = argc > 1 ? atoi(argv[1]) : 14000;
  std::string extra;
  for (int i = 2; i < argc; ++i) extra += std::string(" ") + argv[i];
  const int nh = env("ATT_NH", 16), nkv = env("ATT_NKV", 1), hd = env("ATT_HD", 512), win = env("ATT_WIN", 0);
  const int hpg = nh / nkv, SB = env("ATT_SB", hd / 64), iters = env("ATT_ITERS", 20), nl = env("ATT_LAYERS", 8);
  const int T = env("ATT_T", 1), pos0 = P - T + 1;
  const int cap = (P + 1 + 255) / 256 * 256, nbc = cap / 16;
  ClRt cl;
  if (!cl.init()) return 1;
  const std::string src = read_file(getenv("ATT_FILE") ? getenv("ATT_FILE") : "zero/kernels/ops.cl");
  // (ATT_GRF128: the default register file, twice the threads per EU)
  const std::string base = std::string("-cl-std=CL3.0 -cl-mad-enable ") +
                           (getenv("ATT_GRF128") ? "" : "-cl-intel-256-GRF-per-thread ") + "-DHD=" + std::to_string(hd) +
                           " -DHPG=" + std::to_string(hpg) + " -DSB=" + std::to_string(SB) + extra;
  // kernel choice
  int hg = 0, nsub = 0;
  if (const char* e = getenv("ATT_DSPLIT")) sscanf(e, "%d,%d", &hg, &nsub);
  const bool dsplit = hg > 0;
  const int tr = T <= 1 ? 1 : T <= 2 ? 2 : 4;
  int hs = 1;
  for (int h : {3, 2, 1})
    if (hpg % h == 0 && ((hpg / h) & (hpg / h - 1)) == 0 && hpg / h <= 32) { hs = h; break; }
  if (hd >= 512 || SB > 1) hs = 1;  // as the engine's old rule
  hs = env("ATT_HS", hs);
  const int nsp = hpg / hs;
  int tpw = 1;
  while (tpw < T && tpw < 4 && 16 * nsp * tpw * 2 <= 512) tpw *= 2;
  const bool split_path = !dsplit && (getenv("ATT_DECODE") ? false : getenv("ATT_SPLIT") ? true : 16 * nsp * tpw >= 128);
  cl_kernel ka;
  int sg_per_split;
  if (dsplit) {
    ka = cl.kernel(cl.build(src, base + " -DHS=1 -DTPW=1 -DHG=" + std::to_string(hg) + " -DNSUB=" + std::to_string(nsub) +
                                     " -DTR=" + std::to_string(tr)), "attn_dsplit");
    sg_per_split = hpg / hg * nsub;
  } else {
    ka = cl.kernel(cl.build(src, base + " -DHS=" + std::to_string(hs) + " -DTPW=" + std::to_string(split_path ? tpw : 1)),
                   split_path ? "attn_split" : "attn_decode");
    sg_per_split = split_path ? nsp * tpw : nsp;
  }
  cl_kernel kc = cl.kernel(cl.build(src, "-cl-std=CL3.0 -cl-mad-enable"), "attn_combine");
  printf("nh %d nkv %d hd %d win %d SB %d | P %d T %d | %s %s: spill %zu bytes%s\n", nh, nkv, hd, win, SB, P, T,
         dsplit ? "attn_dsplit" : split_path ? "attn_split" : "attn_decode",
         dsplit ? ("HG " + std::to_string(hg) + " NSUB " + std::to_string(nsub) + " TR " + std::to_string(tr)).c_str()
                : ("HS " + std::to_string(hs) + " NSP " + std::to_string(nsp) + " TPW " + std::to_string(tpw)).c_str(),
         cl.spill_bytes(ka), extra.c_str());

  // int8 q (T rows), K, V with per-block scales, in the engine's layouts
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto i8 = [&](float s) { return (int8_t)std::clamp((int)std::lround(nd(rng) * s), -127, 127); };
  std::vector<int8_t> q((size_t)T * nh * hd);
  std::vector<float> qsc((size_t)T * nh * SB);
  for (auto& v : q) v = i8(40);
  for (auto& v : qsc) v = 0.004f + 0.002f * (rng() % 100) / 100.0f;
  const size_t kvb = (size_t)nkv * cap * hd, scb = (size_t)nkv * cap * SB;
  std::vector<std::vector<uint8_t>> kq(nl, std::vector<uint8_t>(kvb)), vq(nl, std::vector<uint8_t>(kvb));
  std::vector<std::vector<float>> ksc(nl, std::vector<float>(scb)), vsc(nl, std::vector<float>(scb));
  for (int l = 0; l < nl; ++l) {
    for (int g = 0; g < nkv; ++g)
      for (int t = 0; t < cap; ++t) {
        const size_t blk = (size_t)g * nbc + t / 16;
        const int kk = t % 16;
        for (int d = 0; d < hd; ++d) {
          kq[l][((blk * (hd / 4) + d / 4) * 16 + kk) * 4 + d % 4] = (uint8_t)i8(40);
          vq[l][(blk * hd + d) * 16 + kk] = (uint8_t)i8(40);
        }
        for (int c = 0; c < SB; ++c) {
          ksc[l][(blk * SB + c) * 16 + kk] = 0.003f + 0.002f * (rng() % 100) / 100.0f;
          vsc[l][(blk * SB + c) * 16 + kk] = 0.01f + 0.01f * (rng() % 100) / 100.0f;
        }
      }
  }
  auto K = [&](int l, int g, int t, int d) {
    const size_t blk = (size_t)g * nbc + t / 16;
    return (int8_t)kq[l][((blk * (hd / 4) + d / 4) * 16 + t % 16) * 4 + d % 4];
  };
  auto V = [&](int l, int g, int t, int d) { return (int8_t)vq[l][(((size_t)g * nbc + t / 16) * hd + d) * 16 + t % 16]; };
  auto S = [&](const std::vector<float>& s, int g, int t, int c) { return s[(((size_t)g * nbc + t / 16) * SB + c) * 16 + t % 16]; };

  cl_mem d_q = cl.buffer(CL_MEM_READ_ONLY, q.size(), q.data());
  cl_mem d_qsc = cl.buffer(CL_MEM_READ_ONLY, qsc.size() * 4, qsc.data());
  std::vector<cl_mem> d_kq, d_vq, d_ksc, d_vsc;
  for (int l = 0; l < nl; ++l) {
    d_kq.push_back(cl.buffer(CL_MEM_READ_ONLY, kvb, kq[l].data()));
    d_vq.push_back(cl.buffer(CL_MEM_READ_ONLY, kvb, vq[l].data()));
    d_ksc.push_back(cl.buffer(CL_MEM_READ_ONLY, scb * 4, ksc[l].data()));
    d_vsc.push_back(cl.buffer(CL_MEM_READ_ONLY, scb * 4, vsc[l].data()));
  }
  const size_t part_n = (size_t)4 * std::max(cap / 16 + 1, 512) * nh * (hd + 2);
  cl_mem d_part = cl.buffer(CL_MEM_READ_WRITE, part_n * 4);
  cl_mem d_out = cl.buffer(CL_MEM_READ_WRITE, (size_t)T * nh * hd * 4);

  // split choice as the engine: the longest split (<= smax) that still gives `target` sub-groups
  const int lo0 = win ? std::max(0, pos0 - win + 1) : 0;
  const int smax = env("ATT_SMAX", split_path ? 512 : dsplit ? 512 : 256), target = env("ATT_TARGET", split_path ? 256 : 512);
  int Sl = smax;
  while (Sl > 32 && ((P - lo0 / Sl * Sl) / Sl + 1) * nkv * sg_per_split < target) Sl /= 2;
  if (getenv("ATT_SLEN")) Sl = atoi(getenv("ATT_SLEN"));
  // attn_decode (one row per launch) picks its split per row; the others once for all rows
  auto launch = [&](int l, std::vector<cl_event>* ev) {
    int cp = cap, nn = nh, w = win, hdd = hd;
    auto enq = [&](cl_kernel k, int dims, const size_t* g, const size_t* lw) {
      cl_event e = nullptr;
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, dims, nullptr, g, lw, 0, nullptr, ev ? &e : nullptr));
      if (ev) ev->push_back(e);
    };
    auto combine = [&](int ns, int row, int prow) {
      int nsp_ = ns;
      set_args(kc, 0, d_part, nsp_, nn, d_out, row, hdd, prow);
      size_t g1 = (size_t)nh * hd, l1 = 64;
      enq(kc, 1, &g1, &l1);
    };
    if (dsplit || split_path) {
      const int kstart = lo0 / Sl * Sl, ns = (P - kstart) / Sl + 1;
      int p0 = pos0, mv = T, ks = kstart, sl = Sl, nsp_ = ns, q0 = 0;
      set_args(ka, 0, d_q, d_qsc, d_kq[l], d_ksc[l], d_vq[l], d_vsc[l], cp, nn, p0, mv, d_part, w, ks, sl, nsp_, q0);
      const int lsz = 16 * sg_per_split;
      size_t gws[2] = {(size_t)ns * lsz, (size_t)nkv}, lws[2] = {(size_t)lsz, 1};
      enq(ka, 2, gws, lws);
      for (int m = 0; m < T; ++m) combine(ns, m, m);
    } else {
      for (int m = 0; m < T; ++m) {
        const int p = pos0 + m, lo = win ? std::max(0, p - win + 1) : 0;
        int S = 256;
        while (S > 32 && ((p - lo / S * S) / S + 1) * nkv * nsp < 512) S /= 2;
        const int kstart = lo / S * S, ns = (p - kstart) / S + 1;
        int pp = p, row = m, ks = kstart, sl = S;
        set_args(ka, 0, d_q, d_qsc, d_kq[l], d_ksc[l], d_vq[l], d_vsc[l], cp, nn, pp, d_part, row, w, ks, sl);
        size_t gws[2] = {(size_t)ns * 16, (size_t)(nkv * nsp)}, lws[2] = {16, (size_t)nsp};
        enq(ka, 2, gws, lws);
        combine(ns, m, 0);
      }
    }
  };
  // correctness (layer 0, every row, a few heads)
  launch(0, nullptr);
  std::vector<float> out((size_t)T * nh * hd);
  CL_CHECK(clEnqueueReadBuffer(cl.queue, d_out, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));
  double maxerr = 0, maxref = 0;
  for (int m = 0; m < T; ++m) {
    const int p = pos0 + m, lo = win ? std::max(0, p - win + 1) : 0;
    for (int h = 0; h < nh; h += std::max(1, nh / 4)) {
      const int g = h / hpg;
      std::vector<double> s(p + 1, -1e300);
      double mx = -1e300;
      for (int t = lo; t <= p; ++t) {
        double a = 0;
        for (int c = 0; c < SB; ++c) {
          long acc = 0;
          for (int d = c * (hd / SB); d < (c + 1) * (hd / SB); ++d) acc += (long)q[((size_t)m * nh + h) * hd + d] * K(0, g, t, d);
          a += acc * (double)qsc[((size_t)m * nh + h) * SB + c] * S(ksc[0], g, t, c);
        }
        s[t] = a;
        mx = std::max(mx, a);
      }
      double z = 0;
      for (int t = lo; t <= p; ++t) z += std::exp(s[t] - mx);
      for (int d = 0; d < hd; ++d) {
        double o = 0;
        for (int t = lo; t <= p; ++t) o += std::exp(s[t] - mx) / z * V(0, g, t, d) * S(vsc[0], g, t, d / (hd / SB));
        maxerr = std::max(maxerr, std::fabs(o - out[((size_t)m * nh + h) * hd + d]));
        maxref = std::max(maxref, std::fabs(o));
      }
    }
  }
  printf("split length %d | max |err| %.3g (max |ref| %.3g, rel %.2e)\n", Sl, maxerr, maxref, maxerr / maxref);
  std::vector<double> tt;
  for (int it = 0; it < iters + 1; ++it) {
    double a = 0;
    for (int l = 0; l < nl; ++l) {
      std::vector<cl_event> ev;
      launch(l, &ev);
      for (auto e : ev) { a += ClRt::event_ms(e); clReleaseEvent(e); }
    }
    if (it) tt.push_back(a / nl);
  }
  std::sort(tt.begin(), tt.end());
  const double keys = P + 1 - lo0, bytes = keys * nkv * (2.0 * hd + 2 * SB * 4), med = tt[tt.size() / 2];
  printf("attention + combine %.1f us per layer -> %.1f GB/s of K/V\n", med * 1e3, bytes / med / 1e6);
  return 0;
}
