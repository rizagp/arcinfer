// Gemma 4 "unified" vision embedder (mmproj projector type gemma4uv, e.g. Gemma 4 12B): there is no ViT; each
// 48x48 RGB patch goes through LayerNorm -> linear -> LayerNorm -> + learned 2D position embeddings ->
// LayerNorm -> RMSNorm -> input projection into the language model's embedding space, and the language model itself
// attends over the image tokens (non-causally in its sliding-window layers). Runs on the CPU (~0.1 TFLOP per
// full-size image). Matches llama.cpp tools/mtmd (models/gemma4uv.cpp, dyn_size preprocessing), except that small
// images are scaled up to 1120 tokens and large ones use up to 2048 by default (llama.cpp: 70..1120): a 550-pixel
// screenshot became ~140 tokens, one per 48x48 patch with ~10-pixel text, and the model misread words and invented
// numbers; scaled up 3x (1089 tokens) it read the same screenshots exactly.
#pragma once
#include "gguf.h"
#include "image.h"
#include "quants.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// An image encoder of a Gemma 4 model (the server holds either kind): image -> n_tok x out_dim() embeddings.
class GemmaImageEncoder {
 public:
  int min_tokens = 0, max_tokens = 0;  // tokens per image (smaller images are scaled up, larger ones down)
  virtual ~GemmaImageEncoder() = default;
  virtual int out_dim() const = 0;
  virtual std::vector<float> encode(const ImageRGB& im, int& n_tok) = 0;
};

class VisionGemma4 : public GemmaImageEncoder {
 public:
  int n_embd = 0, n_out = 0, patch = 48, pos_size = 0;
  float eps = 1e-6f;

  explicit VisionGemma4(const std::string& path) : f_(path) {
    min_tokens = 1120;
    max_tokens = 2048;
    if (f_.get_str("clip.vision.projector_type") != "gemma4uv")
      throw std::runtime_error("mmproj: only the Gemma 4 unified vision projector (gemma4uv) is supported");
    const int merge = (int)f_.get_int("clip.vision.projector.scale_factor", 3);
    patch = (int)f_.get_int("clip.vision.patch_size", 16) * merge;
    n_embd = (int)f_.get_int("clip.vision.embedding_length", 0);
    n_out = (int)f_.get_int("clip.vision.projection_dim", n_embd);
    eps = (float)f_.get_float("clip.vision.attention.layer_norm_epsilon", 1e-6);
    n1w_ = vec("v.patch_norm.1.weight");
    n1b_ = vec("v.patch_norm.1.bias");
    pew_ = vec("v.patch_embd.weight");
    peb_ = vec("v.patch_embd.bias");
    n2w_ = vec("v.patch_norm.2.weight");
    n2b_ = vec("v.patch_norm.2.bias");
    pos_ = vec("v.position_embd.weight");
    n3w_ = vec("v.patch_norm.3.weight");
    n3b_ = vec("v.patch_norm.3.bias");
    proj_ = vec("mm.input_projection.weight");
    pdim_ = 3 * patch * patch;
    pos_size_ = (int)(pos_.size() / 2 / n_embd);
    if ((int)n1w_.size() != pdim_ || (int)pew_.size() != pdim_ * n_embd || (int)proj_.size() != n_embd * n_out)
      throw std::runtime_error("mmproj: unexpected gemma4uv tensor shapes");
  }

  // Tokens an image of this size becomes, and the resized size.
  int plan(int w, int h, int& W, int& H) const {
    const int64_t px = (int64_t)patch * patch;
    img::fit_size(w, h, patch, (int64_t)min_tokens * px, (int64_t)max_tokens * px, W, H);
    // scaling up rounds each side up, which can pass max_tokens (when min is close to max): round down again
    if ((int64_t)W * H > (int64_t)max_tokens * px) {
      const int W0 = W, H0 = H;
      img::fit_size(W0, H0, patch, 0, (int64_t)max_tokens * px, W, H);
    }
    return (W / patch) * (H / patch);
  }

  // Image -> n_tok x n_out embeddings (row-major), patches in row-major order.
  int out_dim() const override { return n_out; }
  std::vector<float> encode(const ImageRGB& im, int& n_tok) override {
    int W, H;
    n_tok = plan(im.w, im.h, W, H);
    const ImageRGB r = img::resize_bicubic(im, W, H);
    const int nx = W / patch, ny = H / patch, N = nx * ny, K = pdim_, E = n_embd;
    // patch vectors (im2col order: channel, row, column), pixels scaled to [0, 1], then LayerNorm 1
    std::vector<float> A((size_t)N * K);
#pragma omp parallel for
    for (int p = 0; p < N; ++p) {
      const int py = p / nx, pxx = p % nx;
      float* a = &A[(size_t)p * K];
      for (int c = 0; c < 3; ++c)
        for (int ky = 0; ky < patch; ++ky)
          for (int kx = 0; kx < patch; ++kx)
            a[(c * patch + ky) * patch + kx] = r.px[((size_t)(py * patch + ky) * W + pxx * patch + kx) * 3 + c] / 255.0f;
      layer_norm(a, K, n1w_.data(), n1b_.data());
    }
    std::vector<float> X((size_t)N * E);
    matmul(A.data(), N, K, pew_.data(), E, X.data());
#pragma omp parallel for
    for (int p = 0; p < N; ++p) {
      float* x = &X[(size_t)p * E];
      for (int i = 0; i < E; ++i) x[i] += peb_[i];
      layer_norm(x, E, n2w_.data(), n2b_.data());
      const int col = std::min(p % nx, pos_size_ - 1), row = std::min(p / nx, pos_size_ - 1);
      const float* ex = &pos_[(size_t)col * E];
      const float* ey = &pos_[((size_t)pos_size_ + row) * E];
      for (int i = 0; i < E; ++i) x[i] += ex[i] + ey[i];
      layer_norm(x, E, n3w_.data(), n3b_.data());
      double ss = 0;
      for (int i = 0; i < E; ++i) ss += (double)x[i] * x[i];
      const float rr = 1.0f / std::sqrt((float)(ss / E) + eps);
      for (int i = 0; i < E; ++i) x[i] *= rr;
    }
    std::vector<float> out((size_t)N * n_out);
    matmul(X.data(), N, E, proj_.data(), n_out, out.data());
    return out;
  }

 private:
  GgufFile f_;
  std::vector<float> n1w_, n1b_, pew_, peb_, n2w_, n2b_, pos_, n3w_, n3b_, proj_;
  int pdim_ = 0, pos_size_ = 0;

  std::vector<float> vec(const std::string& name) const {
    const GgufTensor& t = f_.get(name);
    std::vector<float> v(t.nelements());
    dequant_row(t.type, t.data, (int64_t)v.size(), v.data());
    return v;
  }
  void layer_norm(float* x, int n, const float* w, const float* b) const {
    double m = 0, v = 0;
    for (int i = 0; i < n; ++i) m += x[i];
    m /= n;
    for (int i = 0; i < n; ++i) v += (x[i] - m) * (x[i] - m);
    const float r = 1.0f / std::sqrt((float)(v / n) + eps), mf = (float)m;
    for (int i = 0; i < n; ++i) x[i] = (x[i] - mf) * r * w[i] + b[i];
  }
  // C[N][M] = A[N][K] * W[M][K]^T, blocked over 32 rows of A so each weight row is reused from cache.
  static void matmul(const float* A, int N, int K, const float* W, int M, float* C) {
    const int nb = (N + 31) / 32;
#pragma omp parallel for schedule(dynamic, 1)
    for (int blk = 0; blk < nb * ((M + 255) / 256); ++blk) {
      const int b = blk % nb, mb = blk / nb;
      const int i0 = b * 32, i1 = std::min(N, i0 + 32), j0 = mb * 256, j1 = std::min(M, j0 + 256);
      for (int j = j0; j < j1; ++j) {
        const float* w = W + (size_t)j * K;
        for (int i = i0; i < i1; ++i) {
          const float* a = A + (size_t)i * K;
          float s = 0.0f;
          for (int k = 0; k < K; ++k) s += a[k] * w[k];
          C[(size_t)i * M + j] = s;
        }
      }
    }
  }
};
