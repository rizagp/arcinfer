// CPU reference forward for Qwen3.5 (f32 activations, dequantized weights), mirroring llama.cpp's graph.
// Used to measure how close arcinferzero and llama.cpp are to the unquantized-activation result.
// usage: ref_forward model.gguf 1,2,3 [arcinferzero_top_logits_file]
#include "../src/gguf.h"
#include "../src/quants.h"
#include <cmath>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <map>
#include <mutex>

static GgufFile* F;
static std::string A;

// An arcinferzero package (zero/gguf_pack.h) holds its routed experts in the expert file next to it, in the GPU raw
// form of zero/expert_store.h (per expert one record: gate, up, down; each superblock as src/quants.h gpu_raw_sb
// writes it, rows of small-block types padded to a multiple of 256). The matrices of one layer are turned back into
// GGUF blocks once and kept until the next layer.
struct PackExperts {
  struct Layer { uint32_t type[3]; uint64_t part[3], rec, off; };
  FILE* f = nullptr;
  std::vector<Layer> layers;
  int cur_layer = -1;
  std::map<std::pair<int, int>, std::vector<uint8_t>> mats;  // (expert, matrix) -> GGUF blocks, N rows of K
  std::mutex mu;  // (the MoE loop runs tokens on several threads; a map node stays put while others are added)
};
static PackExperts* PX = nullptr;
static void open_package(const std::string& gguf) {
  if (!F->has("arcinferzero.package")) return;
  const std::string path = (std::filesystem::path(gguf).parent_path() / F->get_str("arcinferzero.experts_file")).string();
  PX = new PackExperts;
  PX->f = fopen(path.c_str(), "rb");
  if (!PX->f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  struct { uint32_t magic, version; uint64_t fp, layers, experts, bytes; } h{};
  if (fread(&h, sizeof h, 1, PX->f) != 1 || h.magic != 0x58435241u) { fprintf(stderr, "not an expert file: %s\n", path.c_str()); exit(1); }
  PX->layers.resize(h.layers);
  if (fread(PX->layers.data(), sizeof(PackExperts::Layer), h.layers, PX->f) != h.layers) { fprintf(stderr, "short expert file header\n"); exit(1); }
  fprintf(stderr, "ref_forward: experts from the package's expert file %s\n", path.c_str());
}
// one superblock from its GPU raw form back to GGUF bytes (the inverse of gpu_raw_sb)
static void gguf_sb(uint32_t type, const uint8_t* src, uint8_t* dst) {
  const size_t bb = gguf_sb_bytes(type);
  if (type == GGML_Q4_0 || type == GGML_Q8_0 || type == GGML_IQ4_NL || type == GGML_Q2_0) {
    const int qb = type == GGML_Q8_0 ? 32 : 16, nb = type == GGML_Q2_0 ? 4 : 8;
    for (int b = 0; b < nb; ++b) {
      memcpy(dst + (2 + qb) * b, src + 2 * b, 2);
      memcpy(dst + (2 + qb) * b + 2, src + 2 * nb + qb * b, qb);
    }
    return;
  }
  if (gpu_d_last(type)) {
    memcpy(dst, src + bb - 2, 2);
    memcpy(dst + 2, src, bb - 2);
    return;
  }
  memcpy(dst, src, bb);
}
// matrix m (0 gate, 1 up, 2 down) of expert e in layer il: N rows of K values as GGUF blocks (row bytes K / blk * bytes)
static const std::vector<uint8_t>& pack_matrix(int il, int m, int e, int N, int K) {
  std::lock_guard<std::mutex> lock(PX->mu);
  if (PX->cur_layer != il) { PX->mats.clear(); PX->cur_layer = il; }
  auto it = PX->mats.find({e, m});
  if (it != PX->mats.end()) return it->second;
  const PackExperts::Layer& L = PX->layers.at(il);
  const uint32_t ty = L.type[m];
  const size_t bb = gguf_sb_bytes(ty), bp = raw_padded_sb_bytes(ty);
  const int Kp = padded_k(K);
  const size_t nsb = (size_t)Kp / 256, raw_row = nsb * bp;
  int blk = 0, bytes = 0;
  ggml_block_info(ty, blk, bytes);
  const size_t row = (size_t)K / blk * bytes;  // (the GGUF row: the first K values of the padded one)
  uint64_t off = L.off + (uint64_t)e * L.rec;
  for (int i = 0; i < m; ++i) off += L.part[i];
  std::vector<uint8_t> raw(raw_row * N);
  _fseeki64(PX->f, (long long)off, SEEK_SET);
  if (fread(raw.data(), 1, raw.size(), PX->f) != raw.size()) { fprintf(stderr, "short read in the expert file\n"); exit(1); }
  std::vector<uint8_t> out(row * N), sb(bb * nsb);
  for (int n = 0; n < N; ++n) {
    for (size_t b = 0; b < nsb; ++b) gguf_sb(ty, raw.data() + n * raw_row + b * bp, sb.data() + b * bb);
    memcpy(out.data() + n * row, sb.data(), row);
  }
  return PX->mats[{e, m}] = std::move(out);
}

static int hp(const std::string& k, int def = -1) { return (int)F->get_int(A + "." + k, def); }

// int8 absmax rounding of v[0..n) in blocks of B; passes = 2: then the rounding residual rounded the same way and
// added (two int8 passes, values then residual: ~14 bits), as a two-pass engine kernel would see them
static void round_int8(float* v, size_t n, int B, int passes) {
  for (size_t b0 = 0; b0 < n; b0 += B) {
    const size_t b1 = std::min(n, b0 + B);
    float amax = 0;
    for (size_t i = b0; i < b1; ++i) amax = std::fmax(amax, std::fabs(v[i]));
    const float d = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0.0f;
    float ramax = 0;
    for (size_t i = b0; i < b1; ++i) {
      const float q = d * std::lrintf(v[i] * inv);
      ramax = std::fmax(ramax, std::fabs(v[i] - q));
    }
    const float d2 = ramax / 127.0f, inv2 = ramax > 0 ? 127.0f / ramax : 0.0f;
    for (size_t i = b0; i < b1; ++i) {
      const float q = d * std::lrintf(v[i] * inv);
      v[i] = passes >= 2 ? q + d2 * std::lrintf((v[i] - q) * inv2) : q;
    }
  }
}

// name matches env var `var`: "all", "1", or a comma-separated list of substrings of tensor names
static bool name_sel(const char* var, const std::string& name) {
  const char* sel = getenv(var);
  if (!sel) return false;
  const std::string s = sel;
  if (s == "all" || s == "1") return true;
  for (size_t a = 0; a <= s.size();) {
    const size_t b = std::min(s.find(',', a), s.size());
    if (b > a && name.find(s.substr(a, b - a)) != std::string::npos) return true;
    a = b + 1;
  }
  return false;
}

// Optional fake-quantization of matmul inputs (env REF_Q = "all" or substrings of tensor names, REF_QX = names left
// out, REF_QB = block size): int8 absmax per block, as arcinferzero does with 256-element blocks. REF_Q2 = "1" or names:
// two int8 passes for those.
static std::vector<float> fake_quant(const std::string& name, const std::vector<float>& X, int K) {
  if (!name_sel("REF_Q", name) || name_sel("REF_QX", name)) return X;
  const int B = getenv("REF_QB") ? atoi(getenv("REF_QB")) : 256;
  std::vector<float> q = X;
  round_int8(q.data(), q.size(), B, name_sel("REF_Q2", name) ? 2 : 1);
  (void)K;
  return q;
}

// Y[T][N] = X[T][K] * W^T ; W from GGUF (any supported type).
static void matmul(const std::string& name, const std::vector<float>& X0, int T, std::vector<float>& Y) {
  const GgufTensor& t = F->get(name);
  const int K = (int)t.dims[0], N = (int)t.dims[1];
  const std::vector<float> X = fake_quant(name, X0, K);
  const size_t rb = t.nbytes / N;
  const bool wq8 = name_sel("REF_WQ8", name);  // (REF_WQ8: the weight rows rounded as Q8_0, per 32 values)
  Y.assign((size_t)T * N, 0.0f);
#pragma omp parallel
  {
    std::vector<float> w(K);
#pragma omp for schedule(dynamic, 16)
    for (int n = 0; n < N; ++n) {
      dequant_row(t.type, t.data + (size_t)n * rb, K, w.data());
      if (wq8) round_int8(w.data(), (size_t)K, 32, 1);
      for (int i = 0; i < T; ++i) {
        double acc = 0;
        const float* x = &X[(size_t)i * K];
        for (int k = 0; k < K; ++k) acc += (double)w[k] * x[k];
        Y[(size_t)i * N + n] = (float)acc;
      }
    }
  }
}

static std::vector<float> vecf(const std::string& name) {
  const GgufTensor& t = F->get(name);
  std::vector<float> v(t.nelements());
  dequant_row(t.type, t.data, (int)v.size(), v.data());
  return v;
}

static void rmsnorm(float* x, int n, const float* w, float eps) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
  const float r = (float)(1.0 / std::sqrt(ss / n + eps));
  for (int i = 0; i < n; ++i) x[i] = x[i] * r * (w ? w[i] : 1.0f);
}
// (below -80 the result is 0 in float; /fp:fast turned x / (1 + exp(-x)) = x / inf into NaN there: Flash Next's mixers)
static float silu(float x) { return x < -80.0f ? 0.0f : x / (1.0f + std::exp(-x)); }

static float bf16f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// y[N] = W_e x for routed expert e of a tools/convert_gsq.py tensor pair (2-bit codes in the GPU layout of
// zero/kernels/moe.cl, bf16 scale per 128 inputs), decoded independently of the GPU kernels.
// Standard GGUF expert tensors ([n_expert][N][K] of any type with a dequantizer): row by row.
static void expert_mv(const std::string& name, int e, int N, int K, const float* x0, float* y) {
  const GgufTensor& q = F->get(name + ".weight");
  // (REF_Q: the expert inputs rounded as arcinferzero rounds them, like matmul's)
  const std::vector<float> xq = fake_quant(name, std::vector<float>(x0, x0 + K), K);
  const float* x = xq.data();
  if (!q.data && PX) {  // (a package: the expert file)
    const int il = std::stoi(name.substr(4, name.find('.', 4) - 4));
    const int m = name.find("gate") != std::string::npos ? 0 : name.find("_up_") != std::string::npos ? 1 : 2;
    const std::vector<uint8_t>& w8 = pack_matrix(il, m, e, N, K);
    int blk = 0, bytes = 0;
    ggml_block_info(q.type, blk, bytes);
    const size_t rb = (size_t)K / blk * bytes;
    std::vector<float> w(K);
    for (int n = 0; n < N; ++n) {
      dequant_row(q.type, w8.data() + (size_t)n * rb, K, w.data());
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += (double)w[k] * x[k];
      y[n] = (float)acc;
    }
    return;
  }
  if (!F->find(name + ".scale")) {
    const size_t rb = q.nbytes / ((size_t)q.dims[2] * N);
    std::vector<float> w(K);
    for (int n = 0; n < N; ++n) {
      dequant_row(q.type, q.data + ((size_t)e * N + n) * rb, K, w.data());
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += (double)w[k] * x[k];
      y[n] = (float)acc;
    }
    return;
  }
  const GgufTensor& s = F->get(name + ".scale");
  const uint32_t* cw = (const uint32_t*)q.data + (size_t)e * N * K / 16;
  const uint16_t* sw = (const uint16_t*)s.data + (size_t)e * N * K / 128;
  const int KG = K / 128;
  for (int n = 0; n < N; ++n) {
    const int g = n / 16, lane = n % 16;
    double acc = 0;
    for (int kg = 0; kg < KG; ++kg) {
      const float sc = bf16f(sw[((size_t)g * KG + kg) * 16 + lane]);
      double part = 0;
      for (int w = 0; w < 8; ++w) {
        const uint32_t word = cw[(((size_t)g * KG + kg) * 8 + w) * 16 + lane];
        for (int i = 0; i < 16; ++i) {
          const int code = (word >> (8 * (i % 4) + 2 * (i / 4))) & 3;
          part += (double)(code - 2) * x[kg * 128 + w * 16 + i];
        }
      }
      acc += part * sc;
    }
    y[n] = (float)acc;
  }
}

static float gelu_tanh(float x) { return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }

// Gemma 4 decoder (f32 activations, dequantized weights), mirroring llama.cpp's gemma4 graph: embeddings * sqrt(E),
// sandwich norms, q/k RMS norms, V = RMS-normed V (or raw K) without weight, NEOX RoPE (global layers: rope_freqs),
// attention scale 1, sliding window, shared KV layers, GeGLU FFN, layer output scale. Leaves the residual stream in x.
// REF_ATTQ: q, k and v rows rounded to int8 with a per-row absmax scale, as arcinferzero's attention stores them.
// REF_ATTQB = block size of the scales (default: the whole row).
// REF_ATTQ_HD = a head size: only the layers with that head size are rounded (Gemma 4: 512 global, 256 sliding).
static void fq_row(float* v, int n, bool is_v = false) {
  if (!getenv("REF_ATTQ")) return;
  if (getenv("REF_ATTQ_HD") && atoi(getenv("REF_ATTQ_HD")) != n) return;
  const bool row = is_v ? getenv("REF_ATTQV_ROW") != nullptr : getenv("REF_ATTQQK_ROW") != nullptr;
  const int B = getenv("REF_ATTQB") && !row ? atoi(getenv("REF_ATTQB")) : n;
  round_int8(v, (size_t)n, B, getenv("REF_ATTQ2") ? 2 : 1);  // (REF_ATTQ2=1: two int8 passes)
}
static void gemma4_layers(const std::vector<int>& toks, std::vector<float>& x) {
  const int T = (int)toks.size(), nl = hp("block_count"), E = hp("embedding_length"), nh = hp("attention.head_count");
  const int hd_full = hp("attention.key_length"), hd_swa = hp("attention.key_length_swa", hd_full);
  const int n_swa = hp("attention.sliding_window", 0), n_layer_kv = nl - hp("attention.shared_kv_layers", 0);
  const float eps = (float)F->get_float(A + ".attention.layer_norm_rms_epsilon", 1e-6);
  const double base = F->get_float(A + ".rope.freq_base", 1e6), base_swa = F->get_float(A + ".rope.freq_base_swa", 1e4);
  const auto* pat = F->get_ints(A + ".attention.sliding_window_pattern");
  const auto* kvh = F->get_ints(A + ".attention.head_count_kv");
  std::vector<float> rope_ff;
  if (F->find("rope_freqs.weight")) rope_ff = vecf("rope_freqs.weight");
  const GgufTensor& te = F->get("token_embd.weight");
  const int V = (int)te.dims[1];
  x.assign((size_t)T * E, 0.0f);
  for (int t = 0; t < T; ++t) {
    dequant_row(te.type, te.data + (size_t)toks[t] * (te.nbytes / V), E, &x[(size_t)t * E]);
    for (int i = 0; i < E; ++i) x[(size_t)t * E + i] *= std::sqrt((float)E);
  }
  // per-layer embeddings (E2B / E4B, as llama.cpp gemma3n / gemma4 and the engine's ple_inputs): per layer NPL values,
  // pl = (rmsnorm(proj(x) / sqrt(E)) * w + tok_embd_pl * sqrt(NPL)) / sqrt(2); at the end of layer il,
  // x = (x + rmsnorm(proj_il(gelu(inp_gate_il(x)) * pl[il])) * post_norm_il) * out_scale
  const int npl = hp("embedding_length_per_layer_input", 0);
  std::vector<float> pl;
  if (npl > 0) {
    matmul("per_layer_model_proj.weight", x, T, pl);  // [T][nl * npl]
    const GgufTensor& pt = F->get("per_layer_token_embd.weight");
    const size_t np = (size_t)nl * npl, prb = pt.nbytes / pt.dims[1];
    const std::vector<float> pw = vecf("per_layer_proj_norm.weight");
    std::vector<float> tok(np);
    for (int t = 0; t < T; ++t) {
      dequant_row(pt.type, pt.data + (size_t)toks[t] * prb, (int)np, tok.data());
      for (int g = 0; g < nl; ++g) {
        float* v = &pl[(size_t)t * np + (size_t)g * npl];
        for (int i = 0; i < npl; ++i) v[i] /= std::sqrt((float)E);
        rmsnorm(v, npl, pw.data(), eps);
        for (int i = 0; i < npl; ++i) v[i] = (v[i] + tok[(size_t)g * npl + i] * std::sqrt((float)npl)) / std::sqrt(2.0f);
      }
    }
  }
  std::vector<std::vector<float>> Ks(nl), Vs(nl);
  std::vector<int> hds(nl), nkvs(nl);
  for (int il = 0; il < nl; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const bool swa = pat && (*pat)[il] != 0;
    const int hd = swa ? hd_swa : hd_full, nkv = kvh ? (int)(*kvh)[il] : hp("attention.head_count_kv");
    hds[il] = hd;
    nkvs[il] = nkv;
    auto rope = [&](float* v, int pos) {
      const int half = hd / 2;
      for (int i = 0; i < half; ++i) {
        double th = pos * std::pow(swa ? base_swa : base, -2.0 * i / hd);
        if (!swa && !rope_ff.empty()) th /= rope_ff[i];
        const double c = std::cos(th), sn = std::sin(th), a = v[i], b = v[i + half];
        v[i] = (float)(a * c - b * sn);
        v[i + half] = (float)(a * sn + b * c);
      }
    };
    auto an = vecf(p + "attn_norm.weight");
    std::vector<float> h = x, q;
    for (int t = 0; t < T; ++t) rmsnorm(&h[(size_t)t * E], E, an.data(), eps);
    matmul(p + "attn_q.weight", h, T, q);
    auto qn = vecf(p + "attn_q_norm.weight");
    for (int t = 0; t < T; ++t)
      for (int hh = 0; hh < nh; ++hh) {
        float* v = &q[((size_t)t * nh + hh) * hd];
        rmsnorm(v, hd, qn.data(), eps);
        rope(v, t);
        fq_row(v, hd);
      }
    int src = il;
    if (il >= n_layer_kv) {
      src = n_layer_kv - (swa ? 2 : 1);
    } else {
      std::vector<float> k, v;
      matmul(p + "attn_k.weight", h, T, k);
      if (F->find(p + "attn_v.weight")) matmul(p + "attn_v.weight", h, T, v);
      else v = k;  // no V projection: V is the raw K projection
      auto kn = vecf(p + "attn_k_norm.weight");
      for (int t = 0; t < T; ++t)
        for (int g = 0; g < nkv; ++g) {
          float* kr = &k[((size_t)t * nkv + g) * hd];
          rmsnorm(kr, hd, kn.data(), eps);
          rope(kr, t);
          fq_row(kr, hd);
          rmsnorm(&v[((size_t)t * nkv + g) * hd], hd, nullptr, eps);
          fq_row(&v[((size_t)t * nkv + g) * hd], hd, true);
        }
      Ks[il] = std::move(k);
      Vs[il] = std::move(v);
    }
    const std::vector<float>&K = Ks[src], &Vv = Vs[src];
    const int nkv_s = nkvs[src];
    std::vector<float> att((size_t)T * nh * hd);
#pragma omp parallel for collapse(2)
    for (int t = 0; t < T; ++t)
      for (int hh = 0; hh < nh; ++hh) {
        const int g = hh / (nh / nkv_s), u0 = swa && n_swa > 0 ? std::max(0, t - n_swa + 1) : 0;
        const float* qv = &q[((size_t)t * nh + hh) * hd];
        std::vector<double> sc(t + 1);
        double mx = -1e300;
        for (int u = u0; u <= t; ++u) {
          double s2 = 0;
          for (int i = 0; i < hd; ++i) s2 += (double)qv[i] * K[((size_t)u * nkv_s + g) * hd + i];
          sc[u] = s2;  // attention scale 1
          mx = std::max(mx, s2);
        }
        double z = 0;
        for (int u = u0; u <= t; ++u) { sc[u] = std::exp(sc[u] - mx); z += sc[u]; }
        // REF_ATTP: P * vscale rounded to 8 bits per 16-key tile and 64-dim block, as arcinferzero's P*V does
        const bool pq = getenv("REF_ATTP") != nullptr;
        for (int i = 0; i < hd; ++i) {
          double acc = 0;
          if (!pq) {
            for (int u = u0; u <= t; ++u) acc += sc[u] * Vv[((size_t)u * nkv_s + g) * hd + i];
          } else {
            const int c0 = i / 64 * 64;
            for (int b0 = u0 / 16 * 16; b0 <= t; b0 += 16) {
              double w[16], vs[16], wmax = 0;
              for (int j = 0; j < 16; ++j) {
                const int u = b0 + j;
                w[j] = vs[j] = 0;
                if (u < u0 || u > t) continue;
                double am = 0;
                for (int d = c0; d < c0 + 64; ++d) am = std::max(am, (double)std::fabs(Vv[((size_t)u * nkv_s + g) * hd + d]));
                vs[j] = am / 127.0;
                w[j] = sc[u] * vs[j];
                wmax = std::max(wmax, w[j]);
              }
              if (wmax <= 0) continue;
              for (int j = 0; j < 16; ++j)
                if (vs[j] > 0)
                  acc += std::nearbyint(w[j] / wmax * 255.0) * (wmax / 255.0) *
                         (Vv[((size_t)(b0 + j) * nkv_s + g) * hd + i] / vs[j]);
            }
          }
          att[((size_t)t * nh + hh) * hd + i] = (float)(acc / z);
        }
      }
    std::vector<float> o, gg, uu, dd;
    matmul(p + "attn_output.weight", att, T, o);
    auto pa = vecf(p + "post_attention_norm.weight");
    for (int t = 0; t < T; ++t) {
      rmsnorm(&o[(size_t)t * E], E, pa.data(), eps);
      for (int i = 0; i < E; ++i) x[(size_t)t * E + i] += o[(size_t)t * E + i];
    }
    auto fn = vecf(p + "ffn_norm.weight");
    h = x;
    for (int t = 0; t < T; ++t) rmsnorm(&h[(size_t)t * E], E, fn.data(), eps);
    matmul(p + "ffn_gate.weight", h, T, gg);
    matmul(p + "ffn_up.weight", h, T, uu);
    for (size_t i = 0; i < gg.size(); ++i) gg[i] = gelu_tanh(gg[i]) * uu[i];
    matmul(p + "ffn_down.weight", gg, T, dd);
    if (F->find(p + "ffn_gate_inp.weight")) {
      // 26B-A4B (llama.cpp gemma4): dd = rmsnorm(mlp) * post_ffw_norm_1 + rmsnorm(experts) * post_ffw_norm_2; the
      // experts take rmsnorm(x) * pre_ffw_norm_2, the router rmsnorm(x) / sqrt(E) * ffn_gate_inp.scale; softmax over
      // the experts, top n_expert_used renormalized, times each expert's scale (ffn_down_exps.scale); GELU experts
      // whose gate and up rows are one tensor (gate first)
      const int NE = hp("expert_count"), NU = hp("expert_used_count");
      auto p1 = vecf(p + "post_ffw_norm_1.weight"), p2 = vecf(p + "post_ffw_norm_2.weight"),
           n2 = vecf(p + "pre_ffw_norm_2.weight"), rs = vecf(p + "ffn_gate_inp.scale"), es = vecf(p + "ffn_down_exps.scale");
      for (int t = 0; t < T; ++t) rmsnorm(&dd[(size_t)t * E], E, p1.data(), eps);
      std::vector<float> rin = x, ein = x, lg;
      for (int t = 0; t < T; ++t) {
        rmsnorm(&rin[(size_t)t * E], E, nullptr, eps);
        for (int i = 0; i < E; ++i) rin[(size_t)t * E + i] *= rs[i] / std::sqrt((float)E);
        rmsnorm(&ein[(size_t)t * E], E, n2.data(), eps);
      }
      matmul(p + "ffn_gate_inp.weight", rin, T, lg);
      ein = fake_quant(p + "ffn_gate_up_exps.weight", ein, E);  // (REF_Q: the experts' inputs rounded as the engine's)
      std::vector<std::vector<std::pair<int, float>>> by_e(NE);  // per expert: (token, weight)
      for (int t = 0; t < T; ++t) {
        const float* l = &lg[(size_t)t * NE];
        std::vector<int> idx(NE);
        for (int e = 0; e < NE; ++e) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + NU, idx.end(), [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
        double z = 0;
        for (int k = 0; k < NU; ++k) z += std::exp((double)l[idx[k]] - l[idx[0]]);
        for (int k = 0; k < NU; ++k) by_e[idx[k]].push_back({t, (float)(std::exp((double)l[idx[k]] - l[idx[0]]) / z * es[idx[k]])});
      }
      const GgufTensor& tgu = F->get(p + "ffn_gate_up_exps.weight");
      const GgufTensor& tdn = F->get(p + "ffn_down_exps.weight");
      const int K = (int)tgu.dims[0], R2 = (int)tgu.dims[1], Fe = R2 / 2, Kd = (int)tdn.dims[0], Nd = (int)tdn.dims[1];
      const size_t rbu = tgu.nbytes / ((size_t)R2 * NE), rbd = tdn.nbytes / ((size_t)Nd * NE);
      // row r of expert e's gate_up (gate rows 0 .. Fe-1, then up) or down matrix: from the GGUF, or for a package
      // from its expert file (gate, up and down there as three matrices of the record)
      auto gu_row = [&](int e, int r, float* w) {
        if (tgu.data) return dequant_row(tgu.type, tgu.data + ((size_t)e * R2 + r) * rbu, K, w);
        const int m = r < Fe ? 0 : 1, rr = r < Fe ? r : r - Fe;
        const std::vector<uint8_t>& mat = pack_matrix(il, m, e, Fe, K);
        dequant_row(tgu.type, mat.data() + (size_t)rr * (mat.size() / Fe), K, w);
      };
      auto dn_row = [&](int e, int n, float* w) {
        if (tdn.data) return dequant_row(tdn.type, tdn.data + ((size_t)e * Nd + n) * rbd, Kd, w);
        const std::vector<uint8_t>& mat = pack_matrix(il, 2, e, Nd, Kd);
        dequant_row(tdn.type, mat.data() + (size_t)n * (mat.size() / Nd), Kd, w);
      };
      std::vector<double> ysum((size_t)T * E, 0.0);
      std::mutex mu;
#pragma omp parallel for schedule(dynamic, 1)
      for (int e = 0; e < NE; ++e) {
        if (by_e[e].empty()) continue;
        const size_t nt = by_e[e].size();
        std::vector<float> w(std::max(K, Kd)), h(nt * Fe), up(nt * Fe);
        for (int r = 0; r < R2; ++r) {
          gu_row(e, r, w.data());
          for (size_t j = 0; j < nt; ++j) {
            const float* xv = &ein[(size_t)by_e[e][j].first * E];
            double acc = 0;
            for (int k = 0; k < K; ++k) acc += (double)w[k] * xv[k];
            (r < Fe ? h[j * Fe + r] : up[j * Fe + r - Fe]) = (float)acc;
          }
        }
        for (size_t j = 0; j < nt * Fe; ++j) h[j] = gelu_tanh(h[j]) * up[j];
        h = fake_quant(p + "ffn_down_exps.weight", h, Fe);
        std::vector<double> y(nt * Nd, 0.0);
        for (int n = 0; n < Nd; ++n) {
          dn_row(e, n, w.data());
          for (size_t j = 0; j < nt; ++j) {
            double acc = 0;
            for (int k = 0; k < Kd; ++k) acc += (double)w[k] * h[j * Fe + k];
            y[j * Nd + n] = acc * by_e[e][j].second;
          }
        }
        std::lock_guard<std::mutex> g(mu);
        for (size_t j = 0; j < nt; ++j)
          for (int n = 0; n < Nd; ++n) ysum[(size_t)by_e[e][j].first * E + n] += y[j * Nd + n];
      }
      for (int t = 0; t < T; ++t) {
        std::vector<float> m(E);
        for (int i = 0; i < E; ++i) m[i] = (float)ysum[(size_t)t * E + i];
        rmsnorm(m.data(), E, p2.data(), eps);
        for (int i = 0; i < E; ++i) dd[(size_t)t * E + i] += m[i];
      }
    }
    auto pf = vecf(p + "post_ffw_norm.weight");
    float scale = 1.0f;
    if (F->find(p + "layer_output_scale.weight")) scale = vecf(p + "layer_output_scale.weight")[0];
    for (int t = 0; t < T; ++t) {
      rmsnorm(&dd[(size_t)t * E], E, pf.data(), eps);
      for (int i = 0; i < E; ++i) x[(size_t)t * E + i] = (x[(size_t)t * E + i] + dd[(size_t)t * E + i]) * (npl > 0 ? 1.0f : scale);
    }
    if (npl > 0) {  // the per-layer embedding block, then the layer's output scale
      std::vector<float> g, y;
      matmul(p + "inp_gate.weight", x, T, g);  // [T][npl]
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < npl; ++i)
          g[(size_t)t * npl + i] = gelu_tanh(g[(size_t)t * npl + i]) * pl[((size_t)t * nl + il) * npl + i];
      matmul(p + "proj.weight", g, T, y);  // [T][E]
      const std::vector<float> pn = vecf(p + "post_norm.weight");
      for (int t = 0; t < T; ++t) {
        rmsnorm(&y[(size_t)t * E], E, pn.data(), eps);
        for (int i = 0; i < E; ++i) x[(size_t)t * E + i] = (x[(size_t)t * E + i] + y[(size_t)t * E + i]) * scale;
      }
    }
    fprintf(stderr, "\rlayer %d/%d", il + 1, nl);
  }
}

// Qwen3.8 Flash Next (qwen4exp), mirroring llama.cpp src/models/qwen4exp.cpp (see docs/flash-next-spec.md):
// hyper-connections (hc parallel residual streams mixed before and combined after each block), Gated DeltaNet with a
// sigmoid output gate, gated attention (dense: equal to QSA while every key block is selected, i.e. short prompts),
// 512-expert MoE + shared expert, the PLE n-gram hash embedding at its layer. Leaves the final mixer output (the
// model's output norm) in x [T][E].
static void qwen4exp_layers(const std::vector<int>& toks, std::vector<float>& x) {
  const int T = (int)toks.size(), nl = hp("block_count") - hp("nextn_predict_layers", 0), E = hp("embedding_length");
  const int HC = hp("hyper_connection.count"), HD = HC * E;
  const int nh = hp("attention.head_count"), nkv = hp("attention.head_count_kv"), hd = hp("attention.key_length");
  const int nrot = hp("rope.dimension_count"), ds = hp("ssm.state_size"), nk = hp("ssm.group_count"), nv = hp("ssm.time_step_rank");
  const int Di = hp("ssm.inner_size"), interval = hp("full_attention_interval", 4);
  const float eps = (float)F->get_float(A + ".attention.layer_norm_rms_epsilon", 1e-6);
  const double base = F->get_float(A + ".rope.freq_base", 1e7);
  const int Cd = 2 * nk * ds + Di;
  const GgufTensor& te = F->get("token_embd.weight");
  const int V = (int)te.dims[1];
  auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };

  // res[t][c][i]: HC copies of the embedding
  std::vector<float> res((size_t)T * HD);
  {
    std::vector<float> e(E);
    for (int t = 0; t < T; ++t) {
      dequant_row(te.type, te.data + (size_t)toks[t] * (te.nbytes / V), E, e.data());
      for (int c = 0; c < HC; ++c) memcpy(&res[((size_t)t * HC + c) * E], e.data(), E * 4);
    }
  }
  // per-stream RMS norm of v[t][c][:] with a [HC][E] weight (stream c uses w[c*E ..])
  auto stream_norm = [&](std::vector<float>& v, const std::vector<float>& w) {
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < HC; ++c) rmsnorm(&v[((size_t)t * HC + c) * E], E, w.data() + (size_t)c * E, eps);
  };
  // the mixer: block input [T][E] and (with an inject weight) the [T][HC] scatter logits
  auto hc_mix = [&](const std::string& pre, std::vector<float>& mixed, std::vector<float>* inject) {
    std::vector<float> xn = res, lo, gate;
    stream_norm(xn, vecf(pre + "_norm.weight"));
    matmul(pre + "_down.weight", xn, T, lo);
    for (auto& v : lo) v = silu(v / (float)HC);
    matmul(pre + "_up.weight", lo, T, gate);
    if (getenv("REF_DEBUG_LAST") && !inject) {
      int bx = 0, bl = 0, bg = 0;
      for (int i = 0; i < HD; ++i) bx += !std::isfinite(xn[(size_t)(T - 1) * HD + i]);
      for (int i = 0; i < hp("hyper_connection.low_rank"); ++i) bl += !std::isfinite(lo[(size_t)(T - 1) * hp("hyper_connection.low_rank") + i]);
      for (int i = 0; i < HD; ++i) bg += !std::isfinite(gate[(size_t)(T - 1) * HD + i]);
      fprintf(stderr, "\nmixer %s last row: non-finite xn %d lo %d gate %d (lo size %zu, T %d)\n", pre.c_str(), bx, bl, bg, lo.size(), T);
    }
    mixed.assign((size_t)T * E, 0.0f);
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < E; ++i) {
        double acc = 0;
        for (int c = 0; c < HC; ++c) {
          const size_t j = ((size_t)t * HC + c) * E + i;
          acc += xn[j] * sig(gate[j]);
        }
        mixed[(size_t)t * E + i] = (float)(acc / HC);
      }
    if (inject) matmul(pre + "_inject.weight", xn, T, *inject);
  };
  // res[t][c] += 2 sigmoid(inject[t][c] / HC) * out[t]
  auto hc_combine = [&](const std::vector<float>& out, const std::vector<float>& inject) {
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < HC; ++c) {
        const float w = (float)(2.0 * sig(inject[(size_t)t * HC + c] / (double)HC));
        for (int i = 0; i < E; ++i) res[((size_t)t * HC + c) * E + i] += w * out[(size_t)t * E + i];
      }
  };

  // PLE n-gram rows (host hash; the prompt starts a sequence: missing predecessors read as EOS)
  const auto* ple_layers = F->get_ints(A + ".ple.layers");
  const int ple_layer = ple_layers && !ple_layers->empty() ? (int)(*ple_layers)[0] : -1;
  std::vector<float> ple_emb;
  if (ple_layer >= 0) {
    const int ng = hp("ple.ngram_size"), per = hp("ple.heads_per_ngram"), nph = (ng - 1) * per, phd = hp("embedding_length_per_layer_input");
    const int64_t eos = hp("ple.eos_token_id");
    const auto* mult = F->get_ints(A + ".ple.layer_multipliers");
    const auto* off = F->get_ints(A + ".ple.head_offsets");
    const auto* voc = F->get_ints(A + ".ple.head_vocab_sizes");
    const GgufTensor& pt = F->get("per_layer_token_embd.weight");
    const size_t prb = pt.nbytes / pt.dims[1];
    ple_emb.assign((size_t)T * nph * phd, 0.0f);
    for (int t = 0; t < T; ++t) {
      std::vector<int64_t> ctx(ng);
      ctx[0] = toks[t];
      bool cut = false;
      for (int s = 1; s < ng; ++s) {
        const int64_t tk = cut || t - s < 0 ? -1 : toks[t - s];
        cut = cut || tk < 0 || tk == eos;
        ctx[s] = cut ? eos : tk;
      }
      for (int n = 2; n <= ng; ++n) {
        uint64_t mixed = (uint64_t)ctx[0] * (uint64_t)(*mult)[0];
        for (int j = 1; j < n; ++j) mixed ^= (uint64_t)ctx[j] * (uint64_t)(*mult)[j];
        for (int g = 0; g < per; ++g) {
          const int h = (n - 2) * per + g;
          const uint64_t row = mixed % (uint64_t)(*voc)[h] + (uint64_t)(*off)[h];
          dequant_row(pt.type, pt.data + row * prb, phd, &ple_emb[((size_t)t * nph + h) * phd]);
        }
      }
    }
  }

  std::vector<float> mixed, inject, out;
  // (REF_DUMP=file: the residual streams of the last token after each block, as runz ARC_DUMP_ROWS writes them)
  FILE* fdump = getenv("REF_DUMP") ? fopen(getenv("REF_DUMP"), "wb") : nullptr;
  auto dump_res = [&]() { if (fdump) fwrite(&res[(size_t)(T - 1) * HD], 4, HD, fdump); };
  for (int il = 0; il < nl; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    if (il == ple_layer) {
      // gate each stream by its match with the n-gram key, then a dilated depthwise causal conv
      std::vector<float> key, value;
      matmul(p + "ple_key.weight", ple_emb, T, key);
      matmul(p + "ple_value.weight", ple_emb, T, value);
      std::vector<float> query = res;
      stream_norm(key, vecf(p + "ple_norm_key.weight"));
      stream_norm(query, vecf(p + "ple_norm_query.weight"));
      std::vector<float> gated((size_t)T * HD);
      for (int t = 0; t < T; ++t)
        for (int c = 0; c < HC; ++c) {
          double s = 0;
          for (int i = 0; i < E; ++i) s += (double)key[((size_t)t * HC + c) * E + i] * query[((size_t)t * HC + c) * E + i];
          s /= std::sqrt((double)E);
          const double g = sig((s < 0 ? -1.0 : s > 0 ? 1.0 : 0.0) * std::sqrt(std::max(std::fabs(s), 1e-6)));
          for (int i = 0; i < E; ++i) gated[((size_t)t * HC + c) * E + i] = (float)(value[(size_t)t * E + i] * g);
        }
      std::vector<float> nrm = gated;
      stream_norm(nrm, vecf(p + "ple_norm_conv.weight"));
      const auto cw = vecf(p + "ple_conv1d.weight");  // [HD][kern]
      const int kern = hp("ple.conv_kernel"), dil = hp("ple.ngram_size");
      for (int t = 0; t < T; ++t)
        for (int ch = 0; ch < HD; ++ch) {
          double acc = 0;
          for (int k = 0; k < kern; ++k) {
            const int tt = t - (kern - 1 - k) * dil;
            if (tt >= 0) acc += (double)cw[(size_t)ch * kern + k] * nrm[(size_t)tt * HD + ch];
          }
          res[(size_t)t * HD + ch] += gated[(size_t)t * HD + ch] + silu((float)acc);
        }
    }
    hc_mix(p + "hc_attn", mixed, &inject);
    const std::vector<float>& h = mixed;
    out.assign((size_t)T * E, 0.0f);
    if ((il + 1) % interval != 0) {
      // Gated DeltaNet (as Qwen3.5, with a sigmoid output gate)
      std::vector<float> qkv, z, al, be;
      matmul(p + "attn_qkv.weight", h, T, qkv);
      matmul(p + "attn_gate.weight", h, T, z);
      matmul(p + "ssm_alpha.weight", h, T, al);
      matmul(p + "ssm_beta.weight", h, T, be);
      auto dt = vecf(p + "ssm_dt.bias"), sa = vecf(p + "ssm_a"), cw = vecf(p + "ssm_conv1d.weight"), sn = vecf(p + "ssm_norm.weight");
      std::vector<float> c((size_t)T * Cd);
      for (int t = 0; t < T; ++t)
        for (int ch = 0; ch < Cd; ++ch) {
          float acc = 0;
          for (int i = 0; i < 4; ++i) { int tt = t - 3 + i; if (tt >= 0) acc += cw[ch * 4 + i] * qkv[(size_t)tt * Cd + ch]; }
          c[(size_t)t * Cd + ch] = silu(acc);
        }
      for (int t = 0; t < T; ++t)
        for (int hh = 0; hh < 2 * nk; ++hh) {
          float* v = &c[(size_t)t * Cd + hh * ds];
          double ss = 0;
          for (int i = 0; i < ds; ++i) ss += (double)v[i] * v[i];
          const float r = (float)(1.0 / std::sqrt(ss + eps));
          for (int i = 0; i < ds; ++i) v[i] *= r;
        }
      std::vector<float> o((size_t)T * Di);
      std::vector<double> S((size_t)nv * ds * ds, 0.0);
      for (int t = 0; t < T; ++t)
        for (int hv = 0; hv < nv; ++hv) {
          const int kh = hv % nk;
          const float* q = &c[(size_t)t * Cd + kh * ds];
          const float* k = &c[(size_t)t * Cd + nk * ds + kh * ds];
          const float* v = &c[(size_t)t * Cd + 2 * nk * ds + hv * ds];
          const float a = al[(size_t)t * nv + hv] + dt[hv];
          const double g = (a > 20 ? a : std::log1p(std::exp((double)a))) * sa[hv];
          const double decay = std::exp(g), beta = sig(be[(size_t)t * nv + hv]);
          for (int jj = 0; jj < ds; ++jj) {
            double* s = &S[((size_t)hv * ds + jj) * ds];
            double sk = 0;
            for (int i = 0; i < ds; ++i) { s[i] *= decay; sk += s[i] * k[i]; }
            const double d = (v[jj] - sk) * beta;
            double oq = 0;
            for (int i = 0; i < ds; ++i) { s[i] += k[i] * d; oq += s[i] * q[i]; }
            o[(size_t)t * Di + hv * ds + jj] = (float)(oq / std::sqrt((double)ds));
          }
        }
      for (int t = 0; t < T; ++t)
        for (int hv = 0; hv < nv; ++hv) {
          float* ov = &o[(size_t)t * Di + hv * ds];
          rmsnorm(ov, ds, sn.data(), eps);
          for (int i = 0; i < ds; ++i) ov[i] *= (float)sig(z[(size_t)t * Di + hv * ds + i]);
        }
      matmul(p + "ssm_out.weight", o, T, out);
    } else {
      // gated attention, dense causal
      std::vector<float> qg, kk, vv;
      matmul(p + "attn_q.weight", h, T, qg);
      matmul(p + "attn_k.weight", h, T, kk);
      matmul(p + "attn_v.weight", h, T, vv);
      auto qn = vecf(p + "attn_q_norm.weight"), kn = vecf(p + "attn_k_norm.weight");
      std::vector<float> q((size_t)T * nh * hd);
      auto rope = [&](float* v, int pos) {
        const int hr = nrot / 2;
        for (int i = 0; i < hr; ++i) {
          const double th = pos * std::pow(base, -2.0 * i / nrot);
          const double cs = std::cos(th), sn = std::sin(th), a = v[i], b = v[i + hr];
          v[i] = (float)(a * cs - b * sn);
          v[i + hr] = (float)(a * sn + b * cs);
        }
      };
      for (int t = 0; t < T; ++t) {
        for (int hh = 0; hh < nh; ++hh) {
          float* d = &q[((size_t)t * nh + hh) * hd];
          memcpy(d, &qg[(size_t)t * nh * hd * 2 + hh * hd * 2], hd * 4);
          rmsnorm(d, hd, qn.data(), eps);
          rope(d, t);
          fq_row(d, hd);  // (REF_ATTQ: rounded as the engine's attention stores q / K / V)
        }
        for (int hh = 0; hh < nkv; ++hh) {
          float* d = &kk[(size_t)t * nkv * hd + hh * hd];
          rmsnorm(d, hd, kn.data(), eps);
          rope(d, t);
          fq_row(d, hd);
          if (getenv("REF_ATTQV")) fq_row(&vv[(size_t)t * nkv * hd + hh * hd], hd, true);  // (the engine keeps V f16)
        }
      }
      // QSA (llama.cpp qwen4exp build_qsa_sel): the indexer scores blocks of `ratio` cells (pooled raw keys, normed and
      // rotated at the block's first position) per query; a query attends to the cells of its top (top_k / ratio)
      // complete blocks and to the cells of its incomplete tail block. While every block is selected this is dense.
      std::vector<uint8_t> allow;  // [T][T], empty: dense
      const auto* cr = F->get_ints(A + ".attention.compress_ratios");
      const int ratio = cr && il < (int)cr->size() ? (int)(*cr)[il] : 0;
      if (ratio > 0 && F->find(p + "indexer.k_proj.weight") && !getenv("REF_DENSE")) {
        const int ih = hp("attention.indexer.head_count"), id = hp("attention.indexer.key_length");
        const int ntop = (getenv("REF_TOPK") ? atoi(getenv("REF_TOPK")) : hp("attention.indexer.top_k")) / ratio, nb = T / ratio;
        std::vector<float> kr, qi;
        matmul(p + "indexer.k_proj.weight", h, T, kr);
        matmul(p + "indexer.q_proj.weight", h, T, qi);
        auto ikn = vecf(p + "indexer.k_norm.weight"), iqn = vecf(p + "indexer.q_norm.weight");
        std::vector<float> pk((size_t)nb * id, 0.0f);
        for (int b = 0; b < nb; ++b) {
          float* d = &pk[(size_t)b * id];
          for (int j = 0; j < ratio; ++j)
            for (int i = 0; i < id; ++i) d[i] += kr[(size_t)(b * ratio + j) * id + i];
          for (int i = 0; i < id; ++i) d[i] /= (float)ratio;
          rmsnorm(d, id, ikn.data(), eps);
          rope(d, b * ratio);
        }
        for (int t = 0; t < T; ++t)
          for (int hh = 0; hh < ih; ++hh) {
            float* d = &qi[((size_t)t * ih + hh) * id];
            rmsnorm(d, id, iqn.data(), eps);
            rope(d, t);
          }
        allow.assign((size_t)T * T, 0);
#pragma omp parallel for
        for (int t = 0; t < T; ++t) {
          const int nv = (t + 1) / ratio;  // complete blocks visible to t
          std::vector<std::pair<double, int>> sc(nv);
          for (int b = 0; b < nv; ++b) {
            double acc = 0;
            for (int hh = 0; hh < ih; ++hh) {
              double dd = 0;
              for (int i = 0; i < id; ++i) dd += (double)qi[((size_t)t * ih + hh) * id + i] * pk[(size_t)b * id + i];
              acc += std::max(dd, 0.0);
            }
            sc[b] = {acc / std::sqrt((double)id), b};
          }
          // the top blocks, ties to the lower block (any choice is the model's: llama.cpp's top_k is unordered)
          if (t == T - 1 && getenv("REF_DBG_QSA") && atoi(getenv("REF_DBG_QSA")) == il) {
            FILE* f = fopen("qsa_dbg_ref.bin", "wb");
            for (auto& x : sc) { const float v = (float)x.first; fwrite(&v, 4, 1, f); }
            std::vector<std::pair<double, int>> s2 = sc;
            std::stable_sort(s2.begin(), s2.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            std::vector<float> on(nv, 0.0f);
            for (int k = 0; k < std::min(nv, ntop); ++k) on[s2[k].second] = 1.0f;
            fwrite(on.data(), 4, nv, f);
            fclose(f);
          }
          std::stable_sort(sc.begin(), sc.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
          for (int k = 0; k < std::min(nv, ntop); ++k)
            for (int j = 0; j < ratio; ++j) allow[(size_t)t * T + sc[k].second * ratio + j] = 1;
          for (int u = nv * ratio; u <= t; ++u) allow[(size_t)t * T + u] = 1;
        }
      }
      std::vector<float> att((size_t)T * nh * hd);
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
      for (int t = 0; t < T; ++t)
        for (int hh = 0; hh < nh; ++hh) {
          const int g = hh / (nh / nkv);
          std::vector<double> sc(t + 1);
          double mx = -1e300;
          for (int u = 0; u <= t; ++u) {
            if (!allow.empty() && !allow[(size_t)t * T + u]) { sc[u] = -1e300; continue; }
            double s = 0;
            for (int i = 0; i < hd; ++i) s += (double)q[((size_t)t * nh + hh) * hd + i] * kk[(size_t)u * nkv * hd + g * hd + i];
            sc[u] = s / std::sqrt((double)hd);
            mx = std::max(mx, sc[u]);
          }
          double zz = 0;
          for (auto& s : sc) { s = s <= -1e299 ? 0.0 : std::exp(s - mx); zz += s; }
          for (int i = 0; i < hd; ++i) {
            double acc = 0;
            for (int u = 0; u <= t; ++u) acc += sc[u] * vv[(size_t)u * nkv * hd + g * hd + i];
            const float gate = qg[(size_t)t * nh * hd * 2 + hh * hd * 2 + hd + i];
            att[((size_t)t * nh + hh) * hd + i] = (float)(acc / zz * sig(gate));
          }
        }
      matmul(p + "attn_output.weight", att, T, out);
    }
    hc_combine(out, inject);
    dump_res();

    // MoE: router softmax top-k (renormalized) + the sigmoid-gated shared expert
    hc_mix(p + "hc_ffn", mixed, &inject);
    if (getenv("REF_DBG_MOE") && atoi(getenv("REF_DBG_MOE")) == il) {
      FILE* f = fopen("mix_dbg_ref.bin", "wb");
      fwrite(&mixed[(size_t)(T - 1) * E], 4, E, f); fwrite(&inject[(size_t)(T - 1) * HC], 4, HC, f);
      fclose(f);
    }
    std::vector<float> g, u, d;
    matmul(p + "ffn_gate_shexp.weight", h, T, g);
    matmul(p + "ffn_up_shexp.weight", h, T, u);
    for (size_t i = 0; i < g.size(); ++i) g[i] = silu(g[i]) * u[i];
    matmul(p + "ffn_down_shexp.weight", g, T, d);
    const std::vector<float> d_sh = d;
    const int NE = hp("expert_count"), NU = hp("expert_used_count"), Fe = hp("expert_feed_forward_length");
    auto sgw = vecf(p + "ffn_gate_inp_shexp.weight");
    std::vector<float> rl;
    matmul(p + "ffn_gate_inp.weight", h, T, rl);
#pragma omp parallel for
    for (int t = 0; t < T; ++t) {
      const float* ht = &h[(size_t)t * E];
      double sg = 0;
      for (int i = 0; i < E; ++i) sg += (double)sgw[i] * ht[i];
      sg = sig(sg);
      float* dt = &d[(size_t)t * E];
      for (int i = 0; i < E; ++i) dt[i] *= (float)sg;
      std::vector<int> id(NE);
      for (int i = 0; i < NE; ++i) id[i] = i;
      const float* l = &rl[(size_t)t * NE];
      std::stable_sort(id.begin(), id.end(), [&](int a, int b) { return l[a] > l[b]; });
      double z = 0;
      for (int k = 0; k < NU; ++k) z += std::exp((double)l[id[k]] - l[id[0]]);
      std::vector<float> ge(Fe), ue(Fe), de(E);
      for (int k = 0; k < NU; ++k) {
        const int e = id[k];
        const float w = (float)(std::exp((double)l[e] - l[id[0]]) / z);
        expert_mv(p + "ffn_gate_exps", e, Fe, E, ht, ge.data());
        expert_mv(p + "ffn_up_exps", e, Fe, E, ht, ue.data());
        for (int i = 0; i < Fe; ++i) ge[i] = silu(ge[i]) * ue[i];
        expert_mv(p + "ffn_down_exps", e, E, Fe, ge.data(), de.data());
        for (int i = 0; i < E; ++i) dt[i] += w * de[i];
      }
    }
    if (getenv("REF_DBG_MOE") && atoi(getenv("REF_DBG_MOE")) == il) {
      const float* ht = &h[(size_t)(T - 1) * E];
      double sg = 0;
      for (int i = 0; i < E; ++i) sg += (double)sgw[i] * ht[i];
      const float sgf = (float)sig(sg);
      FILE* f = fopen("moe_dbg_ref.bin", "wb");
      fwrite(&d_sh[(size_t)(T - 1) * E], 4, E, f); fwrite(&sgf, 4, 1, f); fwrite(&d[(size_t)(T - 1) * E], 4, E, f);
      fclose(f);
    }
    hc_combine(d, inject);
    dump_res();
    fprintf(stderr, "\rlayer %d/%d", il + 1, nl);
  }
  if (fdump) fclose(fdump);
  // the final mixer is the output norm
  hc_mix("output_hc", mixed, nullptr);
  if (getenv("REF_DEBUG_LAST")) {  // (debug: non-finite values of the last row's streams / mixer output)
    int bad_r = 0, bad_m = 0;
    for (int i = 0; i < HD; ++i) bad_r += !std::isfinite(res[(size_t)(T - 1) * HD + i]);
    for (int i = 0; i < E; ++i) bad_m += !std::isfinite(mixed[(size_t)(T - 1) * E + i]);
    fprintf(stderr, "\nlast row: non-finite streams %d, mixer output %d\n", bad_r, bad_m);
  }
  x = mixed;
}

// Flint (llama.cpp flint): attention with q / k RMS norms per head and per-head sinks (one more logit in the softmax,
// with no value), scale 1/sqrt(head size); the sliding-window layers (a key is masked when q_pos - k_pos >= window)
// take NEOX RoPE over the whole head, the global layers no positions at all. FFN: SwiGLU, dense on the leading
// layers, then the experts chosen by sigmoid scores (top k of the scores; the chosen scores normalized to sum 1,
// times the scale) plus the shared SwiGLU FFN, added without a gate. Leaves the residual stream in x.
static void flint_layers(const std::vector<int>& toks, std::vector<float>& x) {
  const int T = (int)toks.size(), nl = hp("block_count"), E = hp("embedding_length"), nh = hp("attention.head_count"),
            nkv = hp("attention.head_count_kv"), hd = hp("attention.key_length"), nd = hp("leading_dense_block_count", 0),
            NE = hp("expert_count"), NU = hp("expert_used_count"), Fe = hp("expert_feed_forward_length"),
            win = hp("attention.sliding_window", 0);
  const float eps = (float)F->get_float(A + ".attention.layer_norm_rms_epsilon", 1e-6);
  const double base = F->get_float(A + ".rope.freq_base", 10000.0), wscale = F->get_float(A + ".expert_weights_scale", 1.0);
  const bool wnorm = F->get_int(A + ".expert_weights_norm", 0) != 0;
  const auto* swa_pat = F->get_ints(A + ".attention.sliding_window_pattern");
  const auto* rope_pat = F->get_ints(A + ".attention.rope_pattern");
  if (F->get_int(A + ".expert_gating_func", 2) != 2) throw std::runtime_error("flint: only sigmoid expert gating");
  for (int il = 0; il < nl; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const bool swa = swa_pat && il < (int)swa_pat->size() && (*swa_pat)[il] != 0;
    const bool roped = rope_pat ? il < (int)rope_pat->size() && (*rope_pat)[il] != 0 : swa;
    auto an = vecf(p + "attn_norm.weight");
    std::vector<float> h = x, q, k, v, o((size_t)T * nh * hd), out;
    for (int t = 0; t < T; ++t) rmsnorm(&h[(size_t)t * E], E, an.data(), eps);
    matmul(p + "attn_q.weight", h, T, q);
    matmul(p + "attn_k.weight", h, T, k);
    matmul(p + "attn_v.weight", h, T, v);
    auto qn = vecf(p + "attn_q_norm.weight"), kn = vecf(p + "attn_k_norm.weight"), sinks = vecf(p + "attn_sinks.weight");
    auto rope = [&](float* r, int pos) {
      const int half = hd / 2;
      for (int i = 0; i < half; ++i) {
        const double th = pos * std::pow(base, -2.0 * i / hd), cs = std::cos(th), sn = std::sin(th), a = r[i], b = r[i + half];
        r[i] = (float)(a * cs - b * sn);
        r[i + half] = (float)(a * sn + b * cs);
      }
    };
    for (int t = 0; t < T; ++t) {
      for (int hh = 0; hh < nh; ++hh) {
        float* r = &q[((size_t)t * nh + hh) * hd];
        rmsnorm(r, hd, qn.data(), eps);
        if (roped) rope(r, t);
      }
      for (int g = 0; g < nkv; ++g) {
        float* r = &k[((size_t)t * nkv + g) * hd];
        rmsnorm(r, hd, kn.data(), eps);
        if (roped) rope(r, t);
      }
    }
    const double scale = 1.0 / std::sqrt((double)hd);
#pragma omp parallel for collapse(2)
    for (int t = 0; t < T; ++t)
      for (int hh = 0; hh < nh; ++hh) {
        const int g = hh / (nh / nkv), u0 = swa && win > 0 ? std::max(0, t - win + 1) : 0;
        const float* qv = &q[((size_t)t * nh + hh) * hd];
        std::vector<double> sc(t + 1);
        double mx = sinks[hh];
        for (int u = u0; u <= t; ++u) {
          double s2 = 0;
          for (int i = 0; i < hd; ++i) s2 += (double)qv[i] * k[((size_t)u * nkv + g) * hd + i];
          sc[u] = s2 * scale;
          mx = std::max(mx, sc[u]);
        }
        double z = std::exp(sinks[hh] - mx);
        for (int u = u0; u <= t; ++u) { sc[u] = std::exp(sc[u] - mx); z += sc[u]; }
        for (int i = 0; i < hd; ++i) {
          double acc = 0;
          for (int u = u0; u <= t; ++u) acc += sc[u] * v[((size_t)u * nkv + g) * hd + i];
          o[((size_t)t * nh + hh) * hd + i] = (float)(acc / z);
        }
      }
    matmul(p + "attn_output.weight", o, T, out);
    for (size_t i = 0; i < x.size(); ++i) x[i] += out[i];
    // FFN
    auto fn = vecf(p + "ffn_norm.weight");
    h = x;
    for (int t = 0; t < T; ++t) rmsnorm(&h[(size_t)t * E], E, fn.data(), eps);
    std::vector<float> gt, up, d;
    const std::string sfx = il < nd ? ".weight" : "_shexp.weight";  // (MoE layers: the shared FFN first)
    matmul(p + "ffn_gate" + sfx, h, T, gt);
    matmul(p + "ffn_up" + sfx, h, T, up);
    for (size_t i = 0; i < gt.size(); ++i) gt[i] = silu(gt[i]) * up[i];
    matmul(p + "ffn_down" + sfx, gt, T, d);
    if (il >= nd) {
      std::vector<float> rl;
      matmul(p + "ffn_gate_inp.weight", h, T, rl);
#pragma omp parallel for
      for (int t = 0; t < T; ++t) {
        const float* ht = &h[(size_t)t * E];
        std::vector<double> pr(NE);
        for (int e = 0; e < NE; ++e) pr[e] = 1.0 / (1.0 + std::exp(-(double)rl[(size_t)t * NE + e]));
        std::vector<int> id(NE);
        for (int e = 0; e < NE; ++e) id[e] = e;
        std::stable_sort(id.begin(), id.end(), [&](int a, int b) { return pr[a] > pr[b]; });
        double z = 0;
        for (int kk = 0; kk < NU; ++kk) z += pr[id[kk]];
        std::vector<float> ge(Fe), ue(Fe), de(E);
        float* dt = &d[(size_t)t * E];
        for (int kk = 0; kk < NU; ++kk) {
          const int e = id[kk];
          const float w = (float)((wnorm ? pr[e] / z : pr[e]) * wscale);
          expert_mv(p + "ffn_gate_exps", e, Fe, E, ht, ge.data());
          expert_mv(p + "ffn_up_exps", e, Fe, E, ht, ue.data());
          for (int i = 0; i < Fe; ++i) ge[i] = silu(ge[i]) * ue[i];
          expert_mv(p + "ffn_down_exps", e, E, Fe, ge.data(), de.data());
          for (int i = 0; i < E; ++i) dt[i] += w * de[i];
        }
      }
    }
    for (size_t i = 0; i < x.size(); ++i) x[i] += d[i];
    fprintf(stderr, "\rlayer %d/%d", il + 1, nl);
  }
}

// Gemma's final logit soft-capping (cap * tanh(l / cap)); no-op for other models.
static void softcap(std::vector<float>& l) {
  const float c = (float)F->get_float(A + ".final_logit_softcapping", 0.0);
  if (c > 0)
    for (auto& v : l) v = c * std::tanh(v / c);
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: ref_forward model.gguf ids\n"); return 1; }
  GgufFile f(argv[1]);
  F = &f;
  A = f.get_str("general.architecture");
  open_package(argv[1]);
  std::vector<int> toks;
  { std::stringstream ss(argv[2]); std::string s; while (std::getline(ss, s, ',')) toks.push_back(std::stoi(s)); }
  const int T = (int)toks.size();
  const int nl = hp("block_count") - hp("nextn_predict_layers", 0), E = hp("embedding_length");
  const int nh = hp("attention.head_count"), nkv = hp("attention.head_count_kv"), hd = hp("attention.key_length");
  const int nrot = hp("rope.dimension_count"), ds = hp("ssm.state_size"), nk = hp("ssm.group_count"), nv = hp("ssm.time_step_rank");
  const int Di = hp("ssm.inner_size"), interval = hp("full_attention_interval", 4);
  const float eps = (float)f.get_float(A + ".attention.layer_norm_rms_epsilon", 1e-6);
  const float base = (float)f.get_float(A + ".rope.freq_base", 1e7);
  const int Cd = 2 * nk * ds + Di;
  const GgufTensor& te = f.get("token_embd.weight");
  const int V = (int)te.dims[1];

  std::vector<float> x((size_t)T * E), h, y, y2, y3;
  for (int t = 0; t < T; ++t) dequant_row(te.type, te.data + (size_t)toks[t] * (te.nbytes / V), E, &x[(size_t)t * E]);

  if (A == "gemma4") gemma4_layers(toks, x);
  if (A == "qwen4exp") qwen4exp_layers(toks, x);
  if (A == "flint") flint_layers(toks, x);
  for (int il = 0; il < nl && A != "gemma4" && A != "qwen4exp" && A != "flint"; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto an = vecf(p + "attn_norm.weight");
    h = x;
    for (int t = 0; t < T; ++t) rmsnorm(&h[(size_t)t * E], E, an.data(), eps);
    std::vector<float> out((size_t)T * E, 0.0f);
    if ((il + 1) % interval != 0) {
      // Gated DeltaNet
      std::vector<float> qkv, z, al, be;
      matmul(p + "attn_qkv.weight", h, T, qkv);
      matmul(p + "attn_gate.weight", h, T, z);
      matmul(p + "ssm_alpha.weight", h, T, al);
      matmul(p + "ssm_beta.weight", h, T, be);
      auto dt = vecf(p + "ssm_dt.bias"), sa = vecf(p + "ssm_a"), cw = vecf(p + "ssm_conv1d.weight"), sn = vecf(p + "ssm_norm.weight");
      std::vector<float> c((size_t)T * Cd);
      for (int t = 0; t < T; ++t)
        for (int ch = 0; ch < Cd; ++ch) {
          float acc = 0;
          for (int i = 0; i < 4; ++i) { int tt = t - 3 + i; if (tt >= 0) acc += cw[ch * 4 + i] * qkv[(size_t)tt * Cd + ch]; }
          c[(size_t)t * Cd + ch] = silu(acc);
        }
      for (int t = 0; t < T; ++t)
        for (int hh = 0; hh < 2 * nk; ++hh) {
          float* v = &c[(size_t)t * Cd + hh * ds];
          double ss = 0;
          for (int i = 0; i < ds; ++i) ss += (double)v[i] * v[i];
          const float r = (float)(1.0 / std::sqrt(ss + eps));
          for (int i = 0; i < ds; ++i) v[i] *= r;
        }
      std::vector<float> o((size_t)T * Di);
      std::vector<double> S((size_t)nv * ds * ds, 0.0);  // S[h][j][i]
      for (int t = 0; t < T; ++t)
        for (int hv = 0; hv < nv; ++hv) {
          const int kh = hv % nk;
          const float* q = &c[(size_t)t * Cd + kh * ds];
          const float* k = &c[(size_t)t * Cd + nk * ds + kh * ds];
          const float* v = &c[(size_t)t * Cd + 2 * nk * ds + hv * ds];
          const float a = al[(size_t)t * nv + hv] + dt[hv];
          const double g = (a > 20 ? a : std::log1p(std::exp((double)a))) * sa[hv];
          const double decay = std::exp(g), beta = 1.0 / (1.0 + std::exp(-(double)be[(size_t)t * nv + hv]));
          for (int jj = 0; jj < ds; ++jj) {
            double* s = &S[((size_t)hv * ds + jj) * ds];
            double sk = 0;
            for (int i = 0; i < ds; ++i) { s[i] *= decay; sk += s[i] * k[i]; }
            const double d = (v[jj] - sk) * beta;
            double oq = 0;
            for (int i = 0; i < ds; ++i) { s[i] += k[i] * d; oq += s[i] * q[i]; }
            o[(size_t)t * Di + hv * ds + jj] = (float)(oq / std::sqrt((double)ds));
          }
        }
      for (int t = 0; t < T; ++t)
        for (int hv = 0; hv < nv; ++hv) {
          float* ov = &o[(size_t)t * Di + hv * ds];
          rmsnorm(ov, ds, sn.data(), eps);
          for (int i = 0; i < ds; ++i) ov[i] *= silu(z[(size_t)t * Di + hv * ds + i]);
        }
      matmul(p + "ssm_out.weight", o, T, out);
    } else {
      std::vector<float> qg, kk, vv;
      matmul(p + "attn_q.weight", h, T, qg);
      matmul(p + "attn_k.weight", h, T, kk);
      matmul(p + "attn_v.weight", h, T, vv);
      auto qn = vecf(p + "attn_q_norm.weight"), kn = vecf(p + "attn_k_norm.weight");
      std::vector<float> q((size_t)T * nh * hd);
      auto rope = [&](float* v, int pos) {
        const int hr = nrot / 2;
        for (int i = 0; i < hr; ++i) {
          const double th = pos * std::pow((double)base, -2.0 * i / nrot);
          const double cs = std::cos(th), sn = std::sin(th), a = v[i], b = v[i + hr];
          v[i] = (float)(a * cs - b * sn);
          v[i + hr] = (float)(a * sn + b * cs);
        }
      };
      for (int t = 0; t < T; ++t) {
        for (int hh = 0; hh < nh; ++hh) {
          float* d = &q[((size_t)t * nh + hh) * hd];
          memcpy(d, &qg[(size_t)t * nh * hd * 2 + hh * hd * 2], hd * 4);
          rmsnorm(d, hd, qn.data(), eps);
          rope(d, t);
          fq_row(d, hd);  // (REF_ATTQ: rounded as the engine's attention stores q / K / V)
        }
        for (int hh = 0; hh < nkv; ++hh) {
          float* d = &kk[(size_t)t * nkv * hd + hh * hd];
          rmsnorm(d, hd, kn.data(), eps);
          rope(d, t);
          fq_row(d, hd);
          if (getenv("REF_ATTQV")) fq_row(&vv[(size_t)t * nkv * hd + hh * hd], hd, true);  // (the engine keeps V f16)
        }
      }
      std::vector<float> att((size_t)T * nh * hd);
      for (int t = 0; t < T; ++t)
        for (int hh = 0; hh < nh; ++hh) {
          const int g = hh / (nh / nkv);
          std::vector<double> sc(t + 1);
          double mx = -1e300;
          for (int u = 0; u <= t; ++u) {
            double s = 0;
            for (int i = 0; i < hd; ++i) s += (double)q[((size_t)t * nh + hh) * hd + i] * kk[(size_t)u * nkv * hd + g * hd + i];
            sc[u] = s / std::sqrt((double)hd);
            mx = std::max(mx, sc[u]);
          }
          double z = 0;
          for (auto& s : sc) { s = std::exp(s - mx); z += s; }
          for (int i = 0; i < hd; ++i) {
            double acc = 0;
            for (int u = 0; u <= t; ++u) acc += sc[u] * vv[(size_t)u * nkv * hd + g * hd + i];
            const float gate = qg[(size_t)t * nh * hd * 2 + hh * hd * 2 + hd + i];
            att[((size_t)t * nh + hh) * hd + i] = (float)(acc / z) / (1.0f + std::exp(-gate));
          }
        }
      matmul(p + "attn_output.weight", att, T, out);
    }
    for (size_t i = 0; i < x.size(); ++i) x[i] += out[i];
    auto pn = vecf(p + "post_attention_norm.weight");
    h = x;
    for (int t = 0; t < T; ++t) rmsnorm(&h[(size_t)t * E], E, pn.data(), eps);
    std::vector<float> g, u, d;
    const bool moe = A == "qwen35moe";
    const std::string sfx = moe ? "_shexp.weight" : ".weight";  // MoE: the shared expert first
    matmul(p + "ffn_gate" + sfx, h, T, g);
    matmul(p + "ffn_up" + sfx, h, T, u);
    for (size_t i = 0; i < g.size(); ++i) g[i] = silu(g[i]) * u[i];
    matmul(p + "ffn_down" + sfx, g, T, d);
    if (moe) {
      // sigmoid-gated shared expert + the top-k experts of the router softmax, weights renormalized
      const int NE = hp("expert_count"), NU = hp("expert_used_count"), Fe = hp("expert_feed_forward_length");
      auto sgw = vecf(p + "ffn_gate_inp_shexp.weight");
      std::vector<float> rl;
      matmul(p + "ffn_gate_inp.weight", h, T, rl);
#pragma omp parallel for
      for (int t = 0; t < T; ++t) {
        const float* ht = &h[(size_t)t * E];
        double sg = 0;
        for (int i = 0; i < E; ++i) sg += (double)sgw[i] * ht[i];
        sg = 1.0 / (1.0 + std::exp(-sg));
        float* dt = &d[(size_t)t * E];
        for (int i = 0; i < E; ++i) dt[i] *= (float)sg;
        std::vector<int> id(NE);
        for (int i = 0; i < NE; ++i) id[i] = i;
        const float* l = &rl[(size_t)t * NE];
        std::stable_sort(id.begin(), id.end(), [&](int a, int b) { return l[a] > l[b]; });
        double z = 0;
        for (int k = 0; k < NU; ++k) z += std::exp((double)l[id[k]] - l[id[0]]);
        std::vector<float> ge(Fe), ue(Fe), de(E);
        for (int k = 0; k < NU; ++k) {
          const int e = id[k];
          const float w = (float)(std::exp((double)l[e] - l[id[0]]) / z);
          expert_mv(p + "ffn_gate_exps", e, Fe, E, ht, ge.data());
          expert_mv(p + "ffn_up_exps", e, Fe, E, ht, ue.data());
          for (int i = 0; i < Fe; ++i) ge[i] = silu(ge[i]) * ue[i];
          expert_mv(p + "ffn_down_exps", e, E, Fe, ge.data(), de.data());
          for (int i = 0; i < E; ++i) dt[i] += w * de[i];
        }
      }
    }
    for (size_t i = 0; i < x.size(); ++i) x[i] += d[i];
    fprintf(stderr, "\rlayer %d/%d", il + 1, nl);
  }
  // (qwen4exp: no output norm, the final hyper-connection mixer is one)
  const std::vector<float> on = f.find("output_norm.weight") ? vecf("output_norm.weight") : std::vector<float>();
  auto out_norm = [&](float* v) { if (!on.empty()) rmsnorm(v, E, on.data(), eps); };
  if (getenv("REF_PPL")) {
    // perplexity-style scoring of this chunk: negative log-likelihood of tokens T/2+1 .. T-1
    std::vector<float> xs(x.begin() + (size_t)(T / 2) * E, x.end() - E), lg;
    const int R = T - 1 - T / 2;
    for (int r = 0; r < R; ++r) out_norm(&xs[(size_t)r * E]);
    matmul(f.find("output.weight") ? "output.weight" : "token_embd.weight", xs, R, lg);
    softcap(lg);
    double nll = 0;
    for (int r = 0; r < R; ++r) {
      const float* l = &lg[(size_t)r * V];
      double m2 = l[0], z2 = 0;
      for (int i = 0; i < V; ++i) m2 = std::max(m2, (double)l[i]);
      for (int i = 0; i < V; ++i) z2 += std::exp(l[i] - m2);
      nll += -(l[toks[T / 2 + r + 1]] - m2 - std::log(z2));
    }
    printf("\nppl_nll %.6f %d\n", nll, R);
    return 0;
  }
  std::vector<float> last(x.end() - E, x.end()), logits;
  out_norm(last.data());
  matmul(f.find("output.weight") ? "output.weight" : "token_embd.weight", last, 1, logits);
  softcap(logits);
  std::vector<int> idx(V);
  for (int i = 0; i < V; ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  double mx = logits[idx[0]], z = 0;
  for (float l : logits) z += std::exp(l - mx);
  printf("\nreference top-10:");
  for (int i = 0; i < 10; ++i) printf(" %d(%.4f)", idx[i], std::exp(logits[idx[i]] - mx) / z);
  printf("\n");
  // argv[3]: dump log-probs; argv[4]: compare against a dump (KL(dump || this)).
  if (argc > 4) {
    FILE* fi = fopen(argv[4], "rb");
    std::vector<float> ref(V);
    fread(ref.data(), 4, V, fi);
    fclose(fi);
    double kl = 0;
    for (int i = 0; i < V; ++i) kl += std::exp(ref[i]) * (ref[i] - (logits[i] - mx - std::log(z)));
    printf("KL(ref || this) = %.5f\n", kl);
  }
  if (argc > 3 && std::string(argv[3]) != "-") {
    FILE* fo = fopen(argv[3], "wb");
    std::vector<float> lp(V);
    if (getenv("REF_ALL")) {
      // log-probs of every position (T x V), for a KL averaged over the prompt
      std::vector<float> xs = x, all;
      for (int t = 0; t < T; ++t) out_norm(&xs[(size_t)t * E]);
      matmul(f.find("output.weight") ? "output.weight" : "token_embd.weight", xs, T, all);
      softcap(all);
      for (int t = 0; t < T; ++t) {
        const float* l = &all[(size_t)t * V];
        double m2 = l[0], z2 = 0;
        for (int i = 0; i < V; ++i) m2 = std::max(m2, (double)l[i]);
        for (int i = 0; i < V; ++i) z2 += std::exp(l[i] - m2);
        for (int i = 0; i < V; ++i) lp[i] = (float)(l[i] - m2 - std::log(z2));
        fwrite(lp.data(), 4, V, fo);
      }
    } else {
      for (int i = 0; i < V; ++i) lp[i] = (float)(logits[i] - mx - std::log(z));
      fwrite(lp.data(), 4, V, fo);
    }
    fclose(fo);
  }
  return 0;
}
