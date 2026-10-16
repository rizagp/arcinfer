// Qwen3-VL vision encoder (mmproj projector type qwen3vl_merger: Qwen3.5 / 3.6 / 3.8 with vision) on the GPU.
// A ViT over 16x16 patches (two temporal patch kernels summed, as for a still image), learned position embeddings
// resized bilinearly (align corners) to the patch grid, 2-D rotary positions in every attention layer, then a
// LayerNorm and a 2x2 merger MLP into the language model's embedding space: one token per 32x32 pixels.
// Matches llama.cpp tools/mtmd (models/qwen3vl.cpp, bicubic dyn-size preprocessing); no deepstack layers.
//
// The CPU only decodes, resizes and lays out the image (milliseconds). The matrix products run in floating point
// (the mmproj's f16 weights, f32 activations and accumulation: kernels/ops.cl vit_gemm_f16): with the language
// model's int8 GEMM the encoder's output drifted 10-23 % from the f32 reference on photos (every layer's int8
// activations contributed), in f16 0.6-1.8 %. Attention is int8 (vit_attn_lr), norms and activations f32.
// Weights are uploaded on the first image, buffers grow with the largest image seen.
#pragma once
#include "engine.h"
#include "gguf.h"
#include "image.h"
#include "quants.h"
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

class VisionQwen3VL {
 public:
  int n_embd = 0, n_ff = 0, n_layer = 0, n_head = 0, patch = 16, merge = 2, n_out = 0, side = 0, hd_ = 64;
  // merged tokens per image (32x32 pixels each): smaller images are scaled up to min_tokens (the 4B misread short
  // text at ~240 tokens with some questions), larger ones down to max_tokens; llama.cpp allows 8..4096
  int min_tokens = 256, max_tokens = 2048;  // 2048: small text in large images (a 6750-px logo misread at 1024)
  float eps = 1e-6f;
  double encode_ms = 0;  // last image, GPU part

  VisionQwen3VL(const std::string& path, Engine& eng) : f_(path), e_(eng) {
    if (f_.get_str("clip.projector_type") != "qwen3vl_merger")
      throw std::runtime_error("mmproj: not a Qwen3-VL vision projector (qwen3vl_merger)");
    n_embd = (int)f_.get_int("clip.vision.embedding_length", 0);
    n_ff = (int)f_.get_int("clip.vision.feed_forward_length", 0);
    n_layer = (int)f_.get_int("clip.vision.block_count", 0);
    n_head = (int)f_.get_int("clip.vision.attention.head_count", 16);
    patch = (int)f_.get_int("clip.vision.patch_size", 16);
    merge = (int)f_.get_int("clip.vision.spatial_merge_size", 2);
    n_out = (int)f_.get_int("clip.vision.projection_dim", 0);
    eps = (float)f_.get_float("clip.vision.attention.layer_norm_epsilon", 1e-6);
    if (const auto* ds = f_.get_ints("clip.vision.is_deepstack_layers"))
      for (int64_t d : *ds)
        if (d) throw std::runtime_error("mmproj: Qwen3-VL deepstack layers are not supported");
    if (const auto* m = f_.get_floats("clip.vision.image_mean"))
      for (int c = 0; c < 3 && c < (int)m->size(); ++c) mean_[c] = (float)(*m)[c];
    if (const auto* s = f_.get_floats("clip.vision.image_std"))
      for (int c = 0; c < 3 && c < (int)s->size(); ++c) std_[c] = (float)(*s)[c];
    // the GPU kernels want 16-multiples, heads of 64 (4B) or 72 dims (9B: 8 SIMD lanes of 9)
    hd_ = n_head > 0 ? n_embd / n_head : 0;
    if (n_embd % 16 || n_ff % 16 || n_embd % n_head || (hd_ != 64 && hd_ != 72) || merge != 2 || 3 * patch * patch % 16)
      throw std::runtime_error("mmproj: this Qwen3-VL vision shape is not supported on the GPU yet (embedding " +
                               std::to_string(n_embd) + ", heads " + std::to_string(n_head) + ")");
    const GgufTensor& pe = f_.get("v.position_embd.weight");
    side = (int)std::lround(std::sqrt((double)pe.dims[1]));
  }

  // Whether this mmproj is a Qwen3-VL projector these kernels run (the shape checks of the constructor).
  static bool usable(const GgufFile& g) {
    if (g.get_str("clip.projector_type") != "qwen3vl_merger") return false;
    const int64_t e = g.get_int("clip.vision.embedding_length", 0), h = g.get_int("clip.vision.attention.head_count", 16);
    if (const auto* ds = g.get_ints("clip.vision.is_deepstack_layers"))
      for (int64_t d : *ds)
        if (d) return false;
    return e > 0 && e % 16 == 0 && h > 0 && e % h == 0 && (e / h == 64 || e / h == 72) &&
           g.get_int("clip.vision.spatial_merge_size", 2) == 2;
  }

  // GPU memory the encoder takes for images of up to max_tokens (weights + the buffers of reserve()).
  static size_t gpu_bytes(const GgufFile& g, int max_tokens) {
    const size_t E = g.get_int("clip.vision.embedding_length", 0), F = g.get_int("clip.vision.feed_forward_length", 0);
    const size_t P = g.get_int("clip.vision.patch_size", 16), nh = g.get_int("clip.vision.attention.head_count", 16);
    const size_t out = g.get_int("clip.vision.projection_dim", 0), mm0 = 4 * E, K0 = 3 * P * P;
    const size_t Mp = ((size_t)max_tokens * 4 + 127) / 128 * 128;
    const size_t per = K0 * 4 + E * 4 + 8 + E * 4 + E * 4 + std::max(std::max(3 * E, F), mm0) * 4 +
                       std::max(E, out) * 4 + 3 * E * 2 + 3 * nh * 4;
    // (the patch kernel's two temporal halves are summed into one; the position table stays on the host)
    return enc_weight_bytes(g, {"v.", "mm."}, {"v.patch_embd.weight.1", "v.position_embd.weight"}) + Mp * per;
  }

  // Merged tokens an image of this size becomes, and the resized size (multiples of 32 pixels).
  int plan(int w, int h, int& W, int& H) const {
    const int unit = patch * merge;
    img::fit_size(w, h, unit, (int64_t)min_tokens * unit * unit, (int64_t)max_tokens * unit * unit, W, H);
    return (W / unit) * (H / unit);
  }

  // Image -> n_tok x n_out embeddings (row-major), merged tokens in row-major order over a gw x gh grid.
  std::vector<float> encode(const ImageRGB& im, int& n_tok, int& gw, int& gh) {
    load();
    int W, H;
    n_tok = plan(im.w, im.h, W, H);
    const ImageRGB r = img::resize_bicubic(im, W, H);
    const int px = W / patch, py = H / patch, N = px * py, E = n_embd, K0 = 3 * patch * patch;
    gw = px / 2;
    gh = py / 2;
    const int Mp = (N + 127) / 128 * 128;
    reserve(Mp);
    // patches in the merger's order (2x2 groups contiguous), each as (channel, row, column) normalized pixels; the
    // patch bias plus the position embedding (bilinear, align corners) per patch; the rotary (row, column)
    std::vector<float> P((size_t)Mp * K0, 0.0f), A((size_t)N * E);
    std::vector<int> pos((size_t)Mp * 2, 0);
    const std::vector<float>& pb = patch_b_host_;
#pragma omp parallel for
    for (int g = 0; g < N / 4; ++g) {
      const int by = g / gw, bx = g % gw;
      for (int d = 0; d < 4; ++d) {
        const int p = g * 4 + d, y = 2 * by + d / 2, x = 2 * bx + d % 2;
        float* a = &P[(size_t)p * K0];
        for (int c = 0; c < 3; ++c)
          for (int ky = 0; ky < patch; ++ky)
            for (int kx = 0; kx < patch; ++kx)
              a[(c * patch + ky) * patch + kx] =
                  (r.px[((size_t)(y * patch + ky) * W + x * patch + kx) * 3 + c] / 255.0f - mean_[c]) / std_[c];
        const float sy = py > 1 ? (float)y * (side - 1) / (py - 1) : 0.0f, sx = px > 1 ? (float)x * (side - 1) / (px - 1) : 0.0f;
        const int y0 = (int)sy, x0 = (int)sx, y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
        const float fy = sy - y0, fx = sx - x0;
        const float* t00 = &pos_table_[((size_t)y0 * side + x0) * E];
        const float* t01 = &pos_table_[((size_t)y0 * side + x1) * E];
        const float* t10 = &pos_table_[((size_t)y1 * side + x0) * E];
        const float* t11 = &pos_table_[((size_t)y1 * side + x1) * E];
        float* o = &A[(size_t)p * E];
        for (int i = 0; i < E; ++i)
          o[i] = pb[i] + (1 - fy) * ((1 - fx) * t00[i] + fx * t01[i]) + fy * ((1 - fx) * t10[i] + fx * t11[i]);
        pos[2 * p] = y;
        pos[2 * p + 1] = x;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto& q = e_.cl_.queue;
    CL_CHECK(clEnqueueWriteBuffer(q, bin_, CL_FALSE, 0, P.size() * 4, P.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(q, badd_, CL_FALSE, 0, A.size() * 4, A.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(q, bpos_, CL_FALSE, 0, pos.size() * 4, pos.data(), 0, nullptr, nullptr));

    gemm(bin_, K0, patch_w_, x_, E, 0, false, Mp, N);
    add(x_, E, badd_, true, N);
    const float log2_base = (float)std::log2(10000.0);
    for (const Layer& l : L_) {
      ln(x_, l.ln1w, l.ln1b, Mp, N, hn_);
      gemm(hn_, E, l.qkv, h_, 3 * E, 0, false, Mp, N);
      if (att16_) {  // bias, 2-D rotary, f16 q / K / V; f32 attention
        cl_kernel k = kern("vit_qkv_rope", "-DQD=" + std::to_string(hd_));
        int nh = n_head, mp = Mp;
        set_args(k, 0, h_, l.qkv_b, bpos_, nh, log2_base, mp, q_, k_, v_);
        size_t g[2] = {(size_t)n_head * hd_ / 2, (size_t)N};
        e_.run("vit_qkv_rope", k, 2, g, nullptr);
        int n = N;
        if (getenv("ARC_VIT_ATTOLD")) {  // (A/B: the one-row-per-work-item kernel)
          cl_kernel k2 = kern("vit_attn", "-cl-intel-256-GRF-per-thread");
          set_args(k2, 0, q_, k_, v_, n, mp, nh, o_);
          size_t g2[2] = {(size_t)(N + 63) / 64 * 64, (size_t)n_head}, lw2[2] = {64, 1};
          e_.run("vit_attn", k2, 2, g2, lw2);
        } else {
          cl_kernel k2 = kern("vit_attn_rows", "-cl-intel-256-GRF-per-thread -DHD=" + std::to_string(hd_) + " " + ardefs_);
          set_args(k2, 0, q_, k_, v_, n, mp, nh, o_);
          const size_t rows = (size_t)8 * arsg_;
          size_t g2[2] = {((size_t)N + rows - 1) / rows * rows, (size_t)n_head}, lw2[2] = {rows, 1};
          e_.run("vit_attn", k2, 2, g2, lw2);
        }
      } else {
      {  // bias, 2-D rotary, int8 q / K / V
        cl_kernel k = kern("vit_qkv_rope8", attn_defs_);
        int nh = n_head, mp = Mp;
        set_args(k, 0, h_, l.qkv_b, bpos_, nh, log2_base, mp, q_, qsc_, k_, ksc_, v_, vsc_);
        size_t g[2] = {(size_t)n_head * 16, (size_t)N}, lw[2] = {16, 1};
        e_.run("vit_qkv_rope", k, 2, g, lw);
      }
      {
        cl_kernel k = kern("vit_attn_lr", attn_defs_ + " -cl-intel-256-GRF-per-thread");
        int n = N, mp = Mp, nh = n_head;
        set_args(k, 0, q_, qsc_, k_, ksc_, v_, vsc_, n, mp, nh, o_);
        const int rows = 64;  // 8 sub-groups of 8 query rows
        size_t g[2] = {(size_t)(N + rows - 1) / rows * rows, (size_t)n_head}, lw[2] = {(size_t)rows, 1};
        e_.run("vit_attn", k, 2, g, lw);
      }
      }
      gemm(o_, E, l.out, x_, E, 0, true, Mp, N);
      add(x_, E, l.out_b, false, N);
      ln(x_, l.ln2w, l.ln2b, Mp, N, hn_);
      gemm(hn_, E, l.up, h_, n_ff, 0, false, Mp, N);
      bias_act(h_, n_ff, l.up_b, true, N);
      gemm(h_, n_ff, l.down, x_, E, 0, true, Mp, N);
      add(x_, E, l.down_b, false, N);
    }
    // post-norm per patch, then 4 patches side by side (the same memory) through the merger
    ln(x_, post_w_, post_b_, Mp, N, o_);
    const int Mq = Mp / 4, nq = N / 4, E4 = 4 * E;
    gemm(o_, E4, mm0_, h_, mm0_.N, 0, false, Mq, nq);
    bias_act(h_, mm0_.N, mm0_b_, true, nq);
    gemm(h_, mm0_.N, mm2_, o_, n_out, 0, false, Mq, nq);
    bias_act(o_, n_out, mm2_b_, false, nq);
    std::vector<float> out((size_t)nq * n_out);
    CL_CHECK(clEnqueueReadBuffer(q, o_, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));
    encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    e_.collect_profile();
    return out;
  }

 private:
  GgufFile f_;
  Engine& e_;
  float mean_[3] = {0.5f, 0.5f, 0.5f}, std_[3] = {0.5f, 0.5f, 0.5f};
  struct FW {  // an f16 weight [N][K] on the GPU
    cl_mem w = nullptr;
    int N = 0, K = 0;
  };
  struct Layer {
    cl_mem ln1w, ln1b, ln2w, ln2b, qkv_b, out_b, up_b, down_b;
    FW qkv, out, up, down;
  };
  bool loaded_ = false;
  std::vector<Layer> L_;
  FW patch_w_, mm0_, mm2_;
  cl_mem post_w_ = nullptr, post_b_ = nullptr, mm0_b_ = nullptr, mm2_b_ = nullptr;
  std::vector<float> pos_table_, patch_b_host_;
  int cap_ = 0;  // rows the buffers hold
  cl_mem bin_ = nullptr, badd_ = nullptr, bpos_ = nullptr, x_ = nullptr, hn_ = nullptr, h_ = nullptr, o_ = nullptr,
         q_ = nullptr, k_ = nullptr, v_ = nullptr, qsc_ = nullptr, ksc_ = nullptr, vsc_ = nullptr, dummy_ = nullptr;
  // int8 attention: 64-key tiles, softmax denominator from the quantized weights (kernels/ops.cl vit_attn_lr)
  const std::string attn_defs_ = "-DVLR -DVKT=64 -DVCONS";
  bool att16_ = !getenv("ARC_VIT_ATT8");  // f32 attention (int8: 7-9 % error on photos, f32: 0.2-0.5 %)
  const int vcg_ = getenv("ARC_VCG") ? atoi(getenv("ARC_VCG")) : 1, vwgy_ = getenv("ARC_VWGY") ? atoi(getenv("ARC_VWGY")) : 8;
  const std::string vgdefs_ = getenv("ARC_VGDEFS") ? getenv("ARC_VGDEFS") : "";
  const int arsg_ = getenv("ARC_ARSG") ? atoi(getenv("ARC_ARSG")) : 4;  // sub-groups (of 8 query rows) per work-group
  const std::string ardefs_ = getenv("ARC_ARDEFS") ? getenv("ARC_ARDEFS") : "";

  std::vector<float> host(const std::string& name) const {
    const GgufTensor& t = f_.get(name);
    std::vector<float> v(t.nelements());
    dequant_row(t.type, t.data, (int64_t)v.size(), v.data());
    return v;
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
  void load() {
    if (loaded_) return;
    // the two temporal patch kernels see the same frame: their sum is the 2-D patch embedding
    {
      std::vector<float> w = host("v.patch_embd.weight"), w1 = host("v.patch_embd.weight.1");
      for (size_t i = 0; i < w.size(); ++i) w[i] += w1[i];
      patch_w_ = f16(w, n_embd, 3 * patch * patch);
    }
    patch_b_host_ = host("v.patch_embd.bias");
    pos_table_ = host("v.position_embd.weight");
    for (int il = 0; il < n_layer; ++il) {
      const std::string p = "v.blk." + std::to_string(il) + ".";
      Layer l;
      l.ln1w = f32(p + "ln1.weight");
      l.ln1b = f32(p + "ln1.bias");
      l.ln2w = f32(p + "ln2.weight");
      l.ln2b = f32(p + "ln2.bias");
      l.qkv = f16(p + "attn_qkv.weight");
      l.qkv_b = f32(p + "attn_qkv.bias");
      l.out = f16(p + "attn_out.weight");
      l.out_b = f32(p + "attn_out.bias");
      l.up = f16(p + "ffn_up.weight");
      l.up_b = f32(p + "ffn_up.bias");
      l.down = f16(p + "ffn_down.weight");
      l.down_b = f32(p + "ffn_down.bias");
      L_.push_back(l);
    }
    post_w_ = f32("v.post_ln.weight");
    post_b_ = f32("v.post_ln.bias");
    mm0_ = f16("mm.0.weight");
    mm0_b_ = f32("mm.0.bias");
    mm2_ = f16("mm.2.weight");
    mm2_b_ = f32("mm.2.bias");
    if (mm0_.K != 4 * n_embd || mm2_.N != n_out || mm0_.N % 16 || n_out % 16)
      throw std::runtime_error("mmproj: unexpected Qwen3-VL merger shapes");
    loaded_ = true;
  }
  // buffers for Mp patch rows (grown as needed; released with the engine)
  void reserve(int Mp) {
    if (Mp <= cap_) return;
    const size_t E = n_embd, F = std::max<size_t>(std::max<size_t>(3 * E, n_ff), mm0_.N), K0 = 3 * patch * patch;
    bin_ = e_.buf((size_t)Mp * K0 * 4);
    badd_ = e_.buf((size_t)Mp * E * 4);
    bpos_ = e_.buf((size_t)Mp * 8);
    x_ = e_.buf((size_t)Mp * E * 4);
    hn_ = e_.buf((size_t)Mp * E * 4);
    h_ = e_.buf((size_t)Mp * F * 4);
    o_ = e_.buf((size_t)Mp * std::max<size_t>(E, n_out) * 4);
    q_ = e_.buf((size_t)Mp * E * 2);  // (f16 for the f32 attention, int8 for vit_attn_lr)
    k_ = e_.buf((size_t)Mp * E * 2);
    v_ = e_.buf((size_t)Mp * E * 2);
    qsc_ = e_.buf((size_t)Mp * n_head * 4);
    ksc_ = e_.buf((size_t)Mp * n_head * 4);
    vsc_ = e_.buf((size_t)Mp * n_head * 4);
    if (!dummy_) dummy_ = e_.buf(64);
    cap_ = Mp;
  }
  cl_kernel kern(const char* name, const std::string& defs) { return e_.kernel("ops", "-DVIT " + defs, name); }
  // C[:, c_off : c_off + w.N] (+)= A[rows][K] * W^T for M (padded, a multiple of 16) rows, n of them stored
  void gemm(cl_mem A, int lda, const FW& w, cl_mem C, int ldc, int c_off, bool acc, int M, int n) {
    cl_kernel k = kern("vit_gemm_f16", "-cl-intel-256-GRF-per-thread " + vgdefs_);  // (128 registers spill: 2.2x slower)
    int ld = lda, K = w.K, lc = ldc, co = c_off, ac = acc ? 1 : 0, mv = n;
    set_args(k, 0, A, ld, w.w, K, C, lc, co, ac, mv);
    const int vtm = vgdefs_.find("-DVTM=") != std::string::npos ? atoi(vgdefs_.c_str() + vgdefs_.find("-DVTM=") + 6) : 16;
    const int tiles = M / vtm;
    int wgy = vwgy_, cg = vcg_;
    while (tiles % wgy) wgy /= 2;
    while ((w.N / 16) % cg) cg /= 2;
    size_t g[2] = {(size_t)w.N, (size_t)tiles}, lw[2] = {(size_t)16 * cg, (size_t)wgy};
    e_.run("vit_gemm", k, 2, g, lw);
  }
  // LayerNorm of the n valid rows into `out` (f32)
  void ln(cl_mem x, cl_mem w, cl_mem b, int Mp, int n, cl_mem out) {
    cl_kernel k = kern("vit_ln", "");
    int E = n_embd;
    float ep = eps;
    set_args(k, 0, x, E, w, b, ep, out);
    (void)Mp;
    size_t g[2] = {256, (size_t)n}, lw[2] = {256, 1};
    e_.run("vit_ln", k, 2, g, lw);
  }
  void bias_act(cl_mem x, int N, cl_mem b, bool gelu, int n) {
    cl_kernel k = kern("vit_bias_act", "");
    int nn = N, gl = gelu ? 1 : 0;
    set_args(k, 0, x, nn, b, gl);
    size_t g[2] = {(size_t)N, (size_t)n};
    e_.run("vit_bias_act", k, 2, g, nullptr);
  }
  void add(cl_mem x, int N, cl_mem a, bool per_row, int n) {
    cl_kernel k = kern("vit_add", "");
    int nn = N, pr = per_row ? 1 : 0;
    set_args(k, 0, x, nn, a, pr);
    size_t g[2] = {(size_t)N, (size_t)n};
    e_.run("vit_add", k, 2, g, nullptr);
  }
};
