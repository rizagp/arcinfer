// Checks a GPU vision encoder against an independent f32 CPU implementation written from the mmproj alone
// (natural patch order, explicit 2x2 merge): per output token cosine similarity and relative error.
// usage: vision_test model.gguf mmproj.gguf image.(png|jpg) [max_tokens [noref]]
//        (Qwen3-VL projector; the model only provides the OpenCL context, a small context is used)
#include "../src/engine.h"
#include "../src/vision_qwen3vl.h"
#include "../src/vision_gemma4v.h"
#include <cstdio>
#include <immintrin.h>

namespace {
const GgufFile* G = nullptr;
std::vector<float> T(const std::string& n) {
  const GgufTensor& t = G->get(n);
  std::vector<float> v(t.nelements());
  dequant_row(t.type, t.data, (int64_t)v.size(), v.data());
  return v;
}
// Emulation of the GPU's quantization in the reference (env VT_EMU, letters): 'w' = weights through Q8_0; the
// GEMM inputs to quantize to int8 per VT_BLK (default 32) values: p = patch, q = qkv, o = attention out, u = ffn up,
// d = ffn down, m = merger 0, n = merger 2. Shows which rounding the GPU's error comes from.
std::string EMU;
int EMU_BLK = 32;
// y[n][M] = x[n][K] * W[M][K]^T + b (f32 with 8 partial sums: vectorized, ~1e-6 relative)
void linear(std::vector<float> x, int n, int K, std::vector<float> W, const std::vector<float>& b, int M,
            std::vector<float>& y, char tag) {
  if (EMU.find('w') != std::string::npos) {
    const auto q8 = quantize_rows_q8_0(W.data(), M, K);
    dequant_row(GGML_Q8_0, q8.data(), (int64_t)M * K, W.data());
  }
  if (EMU.find('H') != std::string::npos) {  // 'H': weights and every GEMM input rounded to f16
    auto h = [](float v) { return fp16_to_f32(f32_to_fp16(v)); };
    for (float& v : W) v = h(v);
    for (float& v : x) v = h(v);
  }
  if (EMU.find(tag) != std::string::npos)
    for (size_t i = 0; i < (size_t)n * K; i += EMU_BLK) {
      float amax = 0;
      for (int k = 0; k < EMU_BLK; ++k) amax = std::max(amax, std::fabs(x[i + k]));
      const float sc = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0.0f;
      for (int k = 0; k < EMU_BLK; ++k) x[i + k] = std::nearbyint(x[i + k] * inv) * sc;
    }
  y.assign((size_t)n * M, 0.0f);
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < M; ++j) {
      const float* a = &x[(size_t)i * K];
      const float* w = &W[(size_t)j * K];
      float s8[8] = {};
      for (int k = 0; k < K; k += 8)
        for (int u = 0; u < 8; ++u) s8[u] += a[k + u] * w[k + u];
      double s = b.empty() ? 0.0 : b[j];
      for (int u = 0; u < 8; ++u) s += s8[u];
      y[(size_t)i * M + j] = (float)s;
    }
}
void layer_norm(std::vector<float>& x, int n, int E, const std::vector<float>& w, const std::vector<float>& b, float eps) {
  for (int i = 0; i < n; ++i) {
    float* r = &x[(size_t)i * E];
    double m = 0, v = 0;
    for (int k = 0; k < E; ++k) m += r[k];
    m /= E;
    for (int k = 0; k < E; ++k) v += (r[k] - m) * (r[k] - m);
    const double inv = 1.0 / std::sqrt(v / E + eps);
    for (int k = 0; k < E; ++k) r[k] = (float)((r[k] - m) * inv * w[k] + b[k]);
  }
}
float gelu(float x) { return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: vision_test model.gguf mmproj.gguf image [max_tokens [noref]]\n");
    return 1;
  }
  try {
    EngineOptions eo;
    eo.ctx = 4096;
    Engine eng(argv[1], eo);
    if (GgufFile(argv[2]).get_str("clip.vision.projector_type", "") == "gemma4v") {
      // Gemma 4 E2B / E4B: the GPU embeddings (n_tok x n_out f32) and the resized image's pixels go to
      // <image>.g4v.bin / <image>.g4v.rgb for tools/gemma4v_ref.py, the f32 reference
      VisionGemma4V vis(argv[2], eng);
      if (argc > 4) vis.min_tokens = vis.max_tokens = atoi(argv[4]);
      std::vector<uint8_t> bytes;
      {
        FILE* f = fopen(argv[3], "rb");
        if (!f) throw std::runtime_error("cannot open image");
        fseek(f, 0, SEEK_END);
        bytes.resize(ftell(f));
        fseek(f, 0, SEEK_SET);
        fread(bytes.data(), 1, bytes.size(), f);
        fclose(f);
      }
      const ImageRGB im = img::decode(bytes);
      int n_tok, W, H;
      vis.plan(im.w, im.h, W, H);
      std::vector<float> gpu = vis.encode(im, n_tok);
      gpu = vis.encode(im, n_tok);
      printf("image %dx%d -> %dx%d, %d tokens, GPU encoder %.1f ms\n", im.w, im.h, W, H, n_tok, vis.encode_ms);
      if (getenv("ARC_PROF")) {
        eng.profile = true;
        vis.encode(im, n_tok);
        for (const auto& [n, ms] : eng.prof_ms) printf("  %-16s %8.1f ms  x%d\n", n.c_str(), ms, eng.prof_n[n]);
      }
      const ImageRGB r = img::resize_bicubic(im, W, H);
      FILE* fo = fopen((std::string(argv[3]) + ".g4v.bin").c_str(), "wb");
      int hdr[3] = {n_tok, vis.n_out, 0};
      fwrite(hdr, 4, 3, fo);
      fwrite(gpu.data(), 4, gpu.size(), fo);
      fclose(fo);
      fo = fopen((std::string(argv[3]) + ".g4v.rgb").c_str(), "wb");
      int dim[2] = {W, H};
      fwrite(dim, 4, 2, fo);
      fwrite(r.px.data(), 1, r.px.size(), fo);
      fclose(fo);
      return 0;
    }
    VisionQwen3VL vis(argv[2], eng);
    if (argc > 4) vis.max_tokens = atoi(argv[4]);
    std::vector<uint8_t> bytes;
    {
      FILE* f = fopen(argv[3], "rb");
      if (!f) throw std::runtime_error("cannot open image");
      fseek(f, 0, SEEK_END);
      bytes.resize(ftell(f));
      fseek(f, 0, SEEK_SET);
      fread(bytes.data(), 1, bytes.size(), f);
      fclose(f);
    }
    const ImageRGB im = img::decode(bytes);
    int n_tok, gw, gh;
    std::vector<float> gpu = vis.encode(im, n_tok, gw, gh);  // (first call: includes uploading the weights)
    gpu = vis.encode(im, n_tok, gw, gh);
    printf("image %dx%d -> %d tokens (%d x %d merged grid), GPU encoder %.1f ms\n", im.w, im.h, n_tok, gw, gh, vis.encode_ms);
    if (getenv("ARC_PROF")) {  // per-kernel times of one more run
      eng.profile = true;
      vis.encode(im, n_tok, gw, gh);
      for (const auto& [n, ms] : eng.prof_ms) printf("  %-16s %8.1f ms  x%d\n", n.c_str(), ms, eng.prof_n[n]);
    }
    if (argc > 5 && std::string(argv[5]) == "noref") return 0;

    // ---- f32 CPU reference (cached next to the image as <image>.<tokens>.ref)
    const int nq = n_tok;
    std::vector<float> y1;
    const std::string refp = std::string(argv[3]) + "." + std::to_string(n_tok) + ".ref";
    if (FILE* rf = fopen(refp.c_str(), "rb")) {
      y1.resize((size_t)nq * vis.n_out);
      const bool ok = fread(y1.data(), 4, y1.size(), rf) == y1.size();
      fclose(rf);
      if (!ok) y1.clear();
    }
    auto reference = [&]() {
    std::vector<float> y1;
    GgufFile f(argv[2]);
    G = &f;
    const int E = vis.n_embd, F = vis.n_ff, nh = vis.n_head, P = vis.patch, side = vis.side;
    int W, H;
    vis.plan(im.w, im.h, W, H);
    const ImageRGB r = img::resize_bicubic(im, W, H);
    const int px = W / P, py = H / P, N = px * py;
    std::vector<float> w0 = T("v.patch_embd.weight"), w1 = T("v.patch_embd.weight.1"), pb = T("v.patch_embd.bias"),
                       pe = T("v.position_embd.weight");
    for (size_t i = 0; i < w0.size(); ++i) w0[i] += w1[i];
    std::vector<float> patches((size_t)N * 3 * P * P), x;
    for (int y = 0; y < py; ++y)
      for (int xx = 0; xx < px; ++xx)
        for (int c = 0; c < 3; ++c)
          for (int ky = 0; ky < P; ++ky)
            for (int kx = 0; kx < P; ++kx)
              patches[((size_t)(y * px + xx) * 3 + c) * P * P + ky * P + kx] =
                  (r.px[((size_t)(y * P + ky) * W + xx * P + kx) * 3 + c] / 255.0f - 0.5f) / 0.5f;
    linear(patches, N, 3 * P * P, w0, pb, E, x, 'p');
    for (int y = 0; y < py; ++y)  // position embedding: bilinear, align corners
      for (int xx = 0; xx < px; ++xx) {
        const double sy = py > 1 ? (double)y * (side - 1) / (py - 1) : 0, sx = px > 1 ? (double)xx * (side - 1) / (px - 1) : 0;
        const int y0 = (int)sy, x0 = (int)sx, y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
        const double fy = sy - y0, fx = sx - x0;
        for (int k = 0; k < E; ++k)
          x[(size_t)(y * px + xx) * E + k] += (float)((1 - fy) * ((1 - fx) * pe[((size_t)y0 * side + x0) * E + k] + fx * pe[((size_t)y0 * side + x1) * E + k]) +
                                                     fy * ((1 - fx) * pe[((size_t)y1 * side + x0) * E + k] + fx * pe[((size_t)y1 * side + x1) * E + k]));
      }
    for (int il = 0; il < vis.n_layer; ++il) {
      const std::string p = "v.blk." + std::to_string(il) + ".";
      std::vector<float> h = x, qkv, o((size_t)N * E), t, u;
      layer_norm(h, N, E, T(p + "ln1.weight"), T(p + "ln1.bias"), vis.eps);
      linear(h, N, E, T(p + "attn_qkv.weight"), T(p + "attn_qkv.bias"), 3 * E, qkv, 'q');
      const int HDv = E / nh, hf = HDv / 2, qt = HDv / 4;  // head size: 64 (4B) or 72 (9B)
      for (int i = 0; i < N; ++i) {  // 2-D rotary: row for pairs 0..qt-1, column for qt..hf-1, pairs (j, j + hf)
        const int y = i / px, xx = i % px;
        for (int hh = 0; hh < nh; ++hh)
          for (int part = 0; part < 2; ++part) {
            float* v = &qkv[(size_t)i * 3 * E + part * E + hh * HDv];
            for (int j = 0; j < hf; ++j) {
              const double th = (j < qt ? y : xx) * std::pow(10000.0, -(double)(j % qt) / qt);
              const float a = v[j], b = v[j + hf], c = (float)std::cos(th), s = (float)std::sin(th);
              v[j] = a * c - b * s;
              v[j + hf] = a * s + b * c;
            }
          }
      }
#pragma omp parallel for
      for (int i = 0; i < N; ++i)
        for (int hh = 0; hh < nh; ++hh) {
          const float* q = &qkv[(size_t)i * 3 * E + hh * HDv];
          std::vector<double> sc(N);
          double mx = -1e300, z = 0;
          for (int j = 0; j < N; ++j) {
            const float* kk = &qkv[(size_t)j * 3 * E + E + hh * HDv];
            double s = 0;
            for (int d = 0; d < HDv; ++d) s += (double)q[d] * kk[d];
            sc[j] = s / std::sqrt((double)HDv);
            mx = std::max(mx, sc[j]);
          }
          for (int j = 0; j < N; ++j) z += sc[j] = std::exp(sc[j] - mx);
          for (int d = 0; d < HDv; ++d) {
            double acc = 0;
            for (int j = 0; j < N; ++j) acc += sc[j] * qkv[(size_t)j * 3 * E + 2 * E + hh * HDv + d];
            o[(size_t)i * E + hh * HDv + d] = (float)(acc / z);
          }
        }
      linear(o, N, E, T(p + "attn_out.weight"), T(p + "attn_out.bias"), E, t, 'o');
      for (size_t k = 0; k < x.size(); ++k) x[k] += t[k];
      h = x;
      layer_norm(h, N, E, T(p + "ln2.weight"), T(p + "ln2.bias"), vis.eps);
      linear(h, N, E, T(p + "ffn_up.weight"), T(p + "ffn_up.bias"), F, u, 'u');
      for (float& v : u) v = gelu(v);
      linear(u, N, F, T(p + "ffn_down.weight"), T(p + "ffn_down.bias"), E, t, 'd');
      for (size_t k = 0; k < x.size(); ++k) x[k] += t[k];
    }
    layer_norm(x, N, E, T("v.post_ln.weight"), T("v.post_ln.bias"), vis.eps);
    std::vector<float> m4((size_t)nq * 4 * E), y0;
    for (int by = 0; by < py / 2; ++by)  // merged token (by, bx) = patches (2by+dy, 2bx+dx), dy then dx
      for (int bx = 0; bx < px / 2; ++bx)
        for (int d = 0; d < 4; ++d)
          memcpy(&m4[((size_t)(by * (px / 2) + bx) * 4 + d) * E], &x[(size_t)((2 * by + d / 2) * px + 2 * bx + d % 2) * E], E * 4);
    const std::vector<float> mw0 = T("mm.0.weight");
    linear(m4, nq, 4 * E, mw0, T("mm.0.bias"), (int)(mw0.size() / (4 * E)), y0, 'm');
    for (float& v : y0) v = gelu(v);
    const std::vector<float> mw2 = T("mm.2.weight");
    linear(y0, nq, (int)(mw0.size() / (4 * E)), mw2, T("mm.2.bias"), vis.n_out, y1, 'n');
    return y1;
    };
    if (y1.empty()) {
      y1 = reference();
      if (FILE* rf = fopen(refp.c_str(), "wb")) {
        fwrite(y1.data(), 4, y1.size(), rf);
        fclose(rf);
      }
    }
    const char* what = "GPU";
    if (getenv("VT_EMU")) {  // the reference with the GPU's rounding at the chosen places instead of the GPU
      EMU = getenv("VT_EMU");
      if (getenv("VT_BLK")) EMU_BLK = atoi(getenv("VT_BLK"));
      gpu = reference();
      what = "emulated";
    }

    double worst_cos = 1, sum_cos = 0, num = 0, den = 0;
    for (int i = 0; i < nq; ++i) {
      double ab = 0, aa = 0, bb = 0;
      for (int k = 0; k < vis.n_out; ++k) {
        const double a = gpu[(size_t)i * vis.n_out + k], b = y1[(size_t)i * vis.n_out + k];
        ab += a * b;
        aa += a * a;
        bb += b * b;
        num += (a - b) * (a - b);
        den += b * b;
      }
      const double c = ab / std::sqrt(aa * bb + 1e-30);
      worst_cos = std::min(worst_cos, c);
      sum_cos += c;
    }
    printf("%s vs f32 reference over %d tokens: mean cosine %.5f, worst %.5f, relative error %.4f\n", what, nq,
           sum_cos / nq, worst_cos, std::sqrt(num / den));
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
