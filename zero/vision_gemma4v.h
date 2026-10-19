// arcinferzero: a copy of src/vision_gemma4v.h that builds on zero/engine.h (Level Zero).
// Gemma 4 vision encoder of the small models (mmproj projector type gemma4v: E2B / E4B) on the GPU.
// A ViT over 16x16 patches (no patch bias; pixels mapped to [-1, 1]), learned position embeddings looked up per
// column and row, 16 layers with RMS "sandwich" norms, per-head q / k RMS norms (v normed without weight), 2-D
// rotary positions (NEOX halves: columns, rows; theta 100), unscaled attention and a GELU-tanh gated FFN; every
// matrix product clamps its input and output to ranges stored in the mmproj ("clippable linears"). Then 3x3 average
// pooling (one token per 48x48 pixels) times sqrt(width), an RMS norm and the projection into the language model.
// Matches llama.cpp tools/mtmd (models/gemma4v.cpp, bicubic dyn-size preprocessing) except the FFN activation:
// Google's config says gelu_pytorch_tanh, which this follows (llama.cpp falls back to gelu_quick for these files).
//
// As VisionQwen3VL: f16 weights, f32 activations and accumulation (kernels/ops.cl vit_gemm_f16), f32 attention
// (vit_attn_rows); the CPU only decodes, resizes and lays out the image. Weights are uploaded on the first image.
#pragma once
#include "engine.h"
#include "../src/gguf.h"
#include "../src/image.h"
#include "../src/quants.h"
#include "../src/vision_gemma4.h"
#include <chrono>
#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

class VisionGemma4V : public GemmaImageEncoder {
 public:
  int n_embd = 0, n_ff = 0, n_layer = 0, n_head = 0, patch = 16, pool = 3, n_out = 0, pos_size = 0;
  float eps = 1e-6f;
  double encode_ms = 0;  // last image, GPU part

  VisionGemma4V(const std::string& path, Engine& eng) : f_(path), e_(eng) {
    if (f_.get_str("clip.vision.projector_type") != "gemma4v")
      throw std::runtime_error("mmproj: not a Gemma 4 vision projector (gemma4v)");
    n_embd = (int)f_.get_int("clip.vision.embedding_length", 0);
    n_ff = (int)f_.get_int("clip.vision.feed_forward_length", 0);
    n_layer = (int)f_.get_int("clip.vision.block_count", 0);
    n_head = (int)f_.get_int("clip.vision.attention.head_count", 12);
    patch = (int)f_.get_int("clip.vision.patch_size", 16);
    pool = (int)f_.get_int("clip.vision.projector.scale_factor", 3);
    n_out = (int)f_.get_int("clip.vision.projection_dim", 0);
    eps = (float)f_.get_float("clip.vision.attention.layer_norm_epsilon", 1e-6);
    if (const auto* m = f_.get_floats("clip.vision.image_mean"))
      for (int c = 0; c < 3 && c < (int)m->size(); ++c) mean_[c] = (float)(*m)[c];
    if (const auto* s = f_.get_floats("clip.vision.image_std"))
      for (int c = 0; c < 3 && c < (int)s->size(); ++c) std_[c] = (float)(*s)[c];
    if (n_embd % 16 || n_ff % 16 || n_out % 16 || n_embd / n_head != 64 || 3 * patch * patch % 16)
      throw std::runtime_error("mmproj: this Gemma 4 vision shape is not supported on the GPU (embedding " +
                               std::to_string(n_embd) + ", heads " + std::to_string(n_head) + ")");
    if (f_.find("v.std_bias") || f_.find("v.std_scale"))
      throw std::runtime_error("mmproj: Gemma 4 vision std_bias / std_scale are not supported yet");
    pos_size = (int)f_.get("v.position_embd.weight").dims[1];
    // Google's default budget is 280 tokens, 1120 the most it was trained with; small text needs the full 1120
    min_tokens = 1120;
    max_tokens = 1120;
  }

  int out_dim() const override { return n_out; }

  // GPU memory the encoder takes for images of up to max_tokens (weights + the buffers of reserve()).
  static size_t gpu_bytes(const GgufFile& g, int max_tokens) {
    const size_t E = g.get_int("clip.vision.embedding_length", 0), F = g.get_int("clip.vision.feed_forward_length", 0);
    const size_t P = g.get_int("clip.vision.patch_size", 16), pool = g.get_int("clip.vision.projector.scale_factor", 3);
    const size_t out = g.get_int("clip.vision.projection_dim", 0), K0 = 3 * P * P;
    // (a shape the encoder refuses: nothing to keep free; Gemma 4 26B-A4B's 72-dim heads)
    const int64_t nh = g.get_int("clip.vision.attention.head_count", 12);
    if (nh <= 0 || E % nh || E / nh != 64) return 0;
    const size_t R = ((size_t)max_tokens * pool * pool + 127) / 128 * 128;
    const size_t per = K0 * 4 + E * 4 + 8 + E * 4 + E * 4 + std::max(3 * E, 2 * F) * 4 + F * 4 + E * 4 +
                       std::max(E, out) * 4 + 3 * E * 2;
    return enc_weight_bytes(g, {"v.", "mm.input"}, {"v.position_embd.weight"}) + R * per;
  }

  // Tokens an image of this size becomes, and the resized size (multiples of 48 pixels).
  int plan(int w, int h, int& W, int& H) const {
    const int unit = patch * pool;
    const int64_t px = (int64_t)unit * unit;
    img::fit_size(w, h, unit, (int64_t)min_tokens * px, (int64_t)max_tokens * px, W, H);
    if ((int64_t)W * H > (int64_t)max_tokens * px) {  // (scaling up rounds up: round down again)
      const int W0 = W, H0 = H;
      img::fit_size(W0, H0, unit, 0, (int64_t)max_tokens * px, W, H);
    }
    return (W / unit) * (H / unit);
  }

  std::vector<float> encode(const ImageRGB& im, int& n_tok) override {
    load();
    int W, H;
    n_tok = plan(im.w, im.h, W, H);
    const ImageRGB r = img::resize_bicubic(im, W, H);
    const int px = W / patch, py = H / patch, N = px * py, E = n_embd, K0 = 3 * patch * patch;
    if (px > pos_size || py > pos_size) throw std::runtime_error("image too large for the position table");
    const int Mp = (N + 127) / 128 * 128;
    reserve(Mp, n_tok);
    // patches row-major as (channel, row, column) pixels in [-1, 1]; per patch the column and row embeddings
    std::vector<float> P((size_t)Mp * K0, 0.0f), A((size_t)N * E);
    std::vector<int> pos((size_t)Mp * 2, 0);
#pragma omp parallel for
    for (int p = 0; p < N; ++p) {
      const int y = p / px, x = p % px;
      float* a = &P[(size_t)p * K0];
      for (int c = 0; c < 3; ++c)
        for (int ky = 0; ky < patch; ++ky)
          for (int kx = 0; kx < patch; ++kx)
            a[(c * patch + ky) * patch + kx] =
                2.0f * ((r.px[((size_t)(y * patch + ky) * W + x * patch + kx) * 3 + c] / 255.0f - mean_[c]) / std_[c]) - 1.0f;
      const float* tx = &pos_table_[(size_t)x * E];
      const float* ty = &pos_table_[((size_t)pos_size + y) * E];
      float* o = &A[(size_t)p * E];
      for (int i = 0; i < E; ++i) o[i] = tx[i] + ty[i];
      pos[2 * p] = x;
      pos[2 * p + 1] = y;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto& q = e_.cl_.queue;
    CL_CHECK(clEnqueueWriteBuffer(q, bin_, CL_FALSE, 0, P.size() * 4, P.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(q, badd_, CL_FALSE, 0, A.size() * 4, A.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(q, bpos_, CL_FALSE, 0, pos.size() * 4, pos.data(), 0, nullptr, nullptr));

    gemm(bin_, K0, patch_w_, x_, E, 0, false, Mp, N);
    add(x_, E, badd_, N);
    const float log2_base = (float)std::log2(100.0);
    for (const Layer& l : L_) {
      // attention: norm, clamped q / k / v, per-head norms + rotary, unscaled attention, clamped output projection
      norm(x_, E, l.ln1, l.q.in_lo, l.q.in_hi, hn_, E, N);
      gemm(hn_, E, l.q.w, h_, 3 * E, 0, false, Mp, N);
      gemm(hn_, E, l.k.w, h_, 3 * E, E, false, Mp, N);
      gemm(hn_, E, l.v.w, h_, 3 * E, 2 * E, false, Mp, N);
      {
        cl_kernel k = kern("g4v_qkv", "");
        int nh = n_head, mp = Mp;
        float ep = eps, lb = log2_base;
        F8 cl = {{l.q.out_lo, l.q.out_hi, l.k.out_lo, l.k.out_hi, l.v.out_lo, l.v.out_hi, 0.0f, 0.0f}};
        set_args(k, 0, h_, nh, l.qn, l.kn, bpos_, ep, cl, lb, mp, q_, k_, v_);
        size_t g[2] = {(size_t)n_head * 16, (size_t)N}, lw[2] = {16, 1};
        e_.run("g4v_qkv", k, 2, g, lw);
      }
      {
        cl_kernel k = kern("vit_attn_rows", "-cl-intel-256-GRF-per-thread");
        int n = N, mp = Mp, nh = n_head;
        set_args(k, 0, q_, k_, v_, n, mp, nh, o_);
        const size_t rows = (size_t)8 * 4;
        size_t g[2] = {((size_t)N + rows - 1) / rows * rows, (size_t)n_head}, lw[2] = {rows, 1};
        e_.run("vit_attn", k, 2, g, lw);
      }
      clampc(o_, E, E, l.o.in_lo, l.o.in_hi, N);
      gemm(o_, E, l.o.w, t_, E, 0, false, Mp, N);
      norm_add(t_, l.post_attn, l.o.out_lo, l.o.out_hi, N);
      // FFN: norm, clamped gate / up, gelu_tanh(gate) * up, clamped down projection
      norm(x_, E, l.ln2, l.gate.in_lo, l.gate.in_hi, hn_, E, N);
      gemm(hn_, E, l.gate.w, h_, 2 * n_ff, 0, false, Mp, N);
      gemm(hn_, E, l.up.w, h_, 2 * n_ff, n_ff, false, Mp, N);
      {
        cl_kernel k = kern("g4v_geglu", "");
        int F = n_ff;
        F8 cl = {{l.gate.out_lo, l.gate.out_hi, l.up.out_lo, l.up.out_hi, l.down.in_lo, l.down.in_hi, 0.0f, 0.0f}};
        set_args(k, 0, h_, F, cl, g_);
        size_t g[2] = {(size_t)n_ff, (size_t)N};
        e_.run("g4v_geglu", k, 2, g, nullptr);
      }
      gemm(g_, n_ff, l.down.w, t_, E, 0, false, Mp, N);
      norm_add(t_, l.post_ffw, l.down.out_lo, l.down.out_hi, N);
    }
    // 3x3 average pooling times sqrt(E), RMS norm, projection
    const int Mq = (n_tok + 127) / 128 * 128;
    {
      cl_kernel k = kern("g4v_pool", "");
      int e = E, p = px, K = pool;
      float sc = std::sqrt((float)E);
      set_args(k, 0, x_, e, p, K, sc, t_);
      size_t g[2] = {(size_t)E, (size_t)n_tok};
      e_.run("g4v_pool", k, 2, g, nullptr);
    }
    norm(t_, E, ones_, -FLT_MAX, FLT_MAX, hn_, E, n_tok);
    gemm(hn_, E, proj_, o_, n_out, 0, false, Mq, n_tok);
    std::vector<float> out((size_t)n_tok * n_out);
    CL_CHECK(clEnqueueReadBuffer(q, o_, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));
    encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    e_.collect_profile();
    return out;
  }

 private:
  GgufFile f_;
  Engine& e_;
  float mean_[3] = {0.0f, 0.0f, 0.0f}, std_[3] = {1.0f, 1.0f, 1.0f};
  struct alignas(32) F8 {  // an OpenCL float8 kernel argument (clamp ranges)
    float v[8];
  };
  struct FW {  // an f16 weight [N][K] on the GPU
    cl_mem w = nullptr;
    int N = 0, K = 0;
  };
  struct CW {  // a clippable linear
    FW w;
    float in_lo = -FLT_MAX, in_hi = FLT_MAX, out_lo = -FLT_MAX, out_hi = FLT_MAX;
  };
  struct Layer {
    cl_mem ln1, ln2, post_attn, post_ffw, qn, kn;
    CW q, k, v, o, gate, up, down;
  };
  bool loaded_ = false;
  std::vector<Layer> L_;
  FW patch_w_, proj_;
  std::vector<float> pos_table_;
  int cap_ = 0, cap_tok_ = 0;
  cl_mem bin_ = nullptr, badd_ = nullptr, bpos_ = nullptr, x_ = nullptr, hn_ = nullptr, h_ = nullptr, g_ = nullptr,
         t_ = nullptr, o_ = nullptr, q_ = nullptr, k_ = nullptr, v_ = nullptr, ones_ = nullptr;

  std::vector<float> host(const std::string& name) const {
    const GgufTensor& t = f_.get(name);
    std::vector<float> v(t.nelements());
    dequant_row(t.type, t.data, (int64_t)v.size(), v.data());
    return v;
  }
  float scalar(const std::string& name, float dflt) const {
    if (!f_.find(name)) return dflt;
    return host(name)[0];
  }
  cl_mem f32(const std::string& name) {
    const std::vector<float> v = host(name);
    return e_.buf(v.size() * 4, v.data());
  }
  FW f16(const std::vector<float>& v, int N, int K) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = f32_to_fp16(v[i]);
    FW w;
    w.w = e_.buf(h.size() * 2, h.data());
    w.N = N;
    w.K = K;
    return w;
  }
  FW f16(const std::string& name) {
    const GgufTensor& t = f_.get(name);
    return f16(host(name), (int)t.dims[1], (int)t.dims[0]);
  }
  CW clw(const std::string& p) {  // p: "v.blk.N.attn_q"
    CW c;
    c.w = f16(p + ".weight");
    c.in_lo = scalar(p + ".input_min", -FLT_MAX);
    c.in_hi = scalar(p + ".input_max", FLT_MAX);
    c.out_lo = scalar(p + ".output_min", -FLT_MAX);
    c.out_hi = scalar(p + ".output_max", FLT_MAX);
    return c;
  }
  void load() {
    if (loaded_) return;
    {  // conv weight [E][3][16][16] = rows of (channel, row, column)
      const GgufTensor& t = f_.get("v.patch_embd.weight");
      patch_w_ = f16(host("v.patch_embd.weight"), (int)t.dims[3], 3 * patch * patch);
    }
    pos_table_ = host("v.position_embd.weight");
    for (int il = 0; il < n_layer; ++il) {
      const std::string p = "v.blk." + std::to_string(il) + ".";
      Layer l;
      l.ln1 = f32(p + "ln1.weight");
      l.ln2 = f32(p + "ln2.weight");
      l.post_attn = f32(p + "attn_post_norm.weight");
      l.post_ffw = f32(p + "ffn_post_norm.weight");
      l.qn = f32(p + "attn_q_norm.weight");
      l.kn = f32(p + "attn_k_norm.weight");
      l.q = clw(p + "attn_q");
      l.k = clw(p + "attn_k");
      l.v = clw(p + "attn_v");
      l.o = clw(p + "attn_out");
      l.gate = clw(p + "ffn_gate");
      l.up = clw(p + "ffn_up");
      l.down = clw(p + "ffn_down");
      // q / k / v share their input clamp (one normed copy), as do gate / up
      if (l.k.in_lo != l.q.in_lo || l.k.in_hi != l.q.in_hi || l.v.in_lo != l.q.in_lo || l.v.in_hi != l.q.in_hi ||
          l.up.in_lo != l.gate.in_lo || l.up.in_hi != l.gate.in_hi)
        throw std::runtime_error("mmproj: Gemma 4 vision layer " + std::to_string(il) + " has differing input clamps");
      L_.push_back(l);
    }
    {
      const std::vector<float> one(n_embd, 1.0f);  // (the weightless RMS norm before the projection)
      ones_ = e_.buf(one.size() * 4, one.data());
    }
    proj_ = f16("mm.input_projection.weight");
    if (proj_.K != n_embd || proj_.N != n_out) throw std::runtime_error("mmproj: unexpected Gemma 4 vision projection");
    loaded_ = true;
  }
  // buffers for Mp patch rows and n_tok output tokens (grown as needed; released with the engine)
  void reserve(int Mp, int n_tok) {
    const int Mq = (n_tok + 127) / 128 * 128;
    if (Mp <= cap_ && Mq <= cap_tok_) return;
    const size_t E = n_embd, F = n_ff, K0 = 3 * patch * patch, R = std::max(Mp, Mq);
    bin_ = e_.buf(R * K0 * 4);
    badd_ = e_.buf(R * E * 4);
    bpos_ = e_.buf(R * 8);
    x_ = e_.buf(R * E * 4);
    hn_ = e_.buf(R * E * 4);
    h_ = e_.buf(R * std::max<size_t>(3 * E, 2 * F) * 4);
    g_ = e_.buf(R * F * 4);
    t_ = e_.buf(R * E * 4);
    o_ = e_.buf(R * std::max<size_t>(E, n_out) * 4);
    q_ = e_.buf(R * E * 2);
    k_ = e_.buf(R * E * 2);
    v_ = e_.buf(R * E * 2);
    cap_ = std::max(cap_, Mp);
    cap_tok_ = std::max(cap_tok_, Mq);
  }
  cl_kernel kern(const char* name, const std::string& defs) { return e_.kernel("ops", "-DVIT " + defs, name); }
  void gemm(cl_mem A, int lda, const FW& w, cl_mem C, int ldc, int c_off, bool acc, int M, int n) {
    cl_kernel k = kern("vit_gemm_f16", "-cl-intel-256-GRF-per-thread");
    int ld = lda, K = w.K, lc = ldc, co = c_off, ac = acc ? 1 : 0, mv = n;
    set_args(k, 0, A, ld, w.w, K, C, lc, co, ac, mv);
    const int tiles = M / 16;
    int wgy = 8;
    while (tiles % wgy) wgy /= 2;
    size_t g[2] = {(size_t)w.N, (size_t)tiles}, lw[2] = {16, (size_t)wgy};
    e_.run("vit_gemm", k, 2, g, lw);
  }
  void norm(cl_mem x, int ldx, cl_mem w, float lo, float hi, cl_mem out, int ldo, int n) {
    cl_kernel k = kern("g4v_norm", "");
    int lx = ldx, E = n_embd, lo2 = ldo;
    float ep = eps, l = lo, h = hi;
    set_args(k, 0, x, lx, E, w, ep, l, h, out, lo2);
    size_t g[2] = {256, (size_t)n}, lw[2] = {256, 1};
    e_.run("g4v_norm", k, 2, g, lw);
  }
  void norm_add(cl_mem y, cl_mem w, float lo, float hi, int n) {
    cl_kernel k = kern("g4v_norm_add", "");
    int E = n_embd;
    float ep = eps, l = lo, h = hi;
    set_args(k, 0, y, E, w, ep, l, h, x_);
    size_t g[2] = {256, (size_t)n}, lw[2] = {256, 1};
    e_.run("g4v_norm", k, 2, g, lw);
  }
  void clampc(cl_mem x, int ld, int N, float lo, float hi, int n) {
    cl_kernel k = kern("g4v_clamp", "");
    int l = ld;
    float a = lo, b = hi;
    set_args(k, 0, x, l, a, b);
    size_t g[2] = {(size_t)N, (size_t)n};
    e_.run("g4v_clamp", k, 2, g, nullptr);
  }
  void add(cl_mem x, int N, cl_mem a, int n) {
    cl_kernel k = kern("vit_add", "");
    int nn = N, pr = 1;
    set_args(k, 0, x, nn, a, pr);
    size_t g[2] = {(size_t)N, (size_t)n};
    e_.run("vit_add", k, 2, g, nullptr);
  }
};
