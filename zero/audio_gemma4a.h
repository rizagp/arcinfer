// arcinferzero: a copy of src/audio_gemma4a.h that builds on zero/engine.h (Level Zero).
// Gemma 4 audio encoder of the small models (mmproj audio projector type gemma4a: E2B / E4B), the conformer on the
// GPU. Matches llama.cpp tools/mtmd (models/gemma4a.cpp, mtmd_audio_preprocessor_gemma4a):
//  - CPU: 16 kHz samples in clips of up to 30 s -> log-mel (semicausal padding, 320-sample Hann window in a 512-point
//    FFT, hop 160, magnitude, 128 HTK mel bins, ln(max(mel, 0.001))) -> two 3x3 stride-2 convolutions (128 and 32
//    channels, LayerNorm over channels, ReLU): 25 frames per second, 32 x 32 features each;
//  - GPU (f16 weights, f32 activations, kernels/ops.cl vit_gemm_f16 and a4_*): the input projection, then per
//    conformer block a half-step SiLU FFN, local causal self-attention (12 frames back, sinusoidal relative positions,
//    logits soft-capped at 50), a convolution module (GLU, causal depthwise conv of width 5, RMS norm, SiLU) and a
//    second half-step FFN, all with RMS norms and the mmproj's input / output clamps; then the output projection,
//    an RMS norm and the projection into the language model. One token per 40 ms.
#pragma once
#include "../src/audio_gemma4.h"
#include "engine.h"
#include "../src/gguf.h"
#include "../src/quants.h"
#include <cfloat>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

class AudioGemma4A : public GemmaAudioEncoder {
 public:
  int n_embd = 0, n_ff = 0, n_layer = 0, n_head = 0, n_mel = 128, n_out = 0;
  float eps = 1e-6f;
  double encode_ms = 0;

  AudioGemma4A(const std::string& path, Engine& eng) : f_(path), e_(eng) {
    if (f_.get_str("clip.audio.projector_type") != "gemma4a")
      throw std::runtime_error("mmproj: not a Gemma 4 audio conformer (gemma4a)");
    n_embd = (int)f_.get_int("clip.audio.embedding_length", 0);
    n_ff = (int)f_.get_int("clip.audio.feed_forward_length", 0);
    n_layer = (int)f_.get_int("clip.audio.block_count", 0);
    n_head = (int)f_.get_int("clip.audio.attention.head_count", 8);
    n_mel = (int)f_.get_int("clip.audio.num_mel_bins", 128);
    n_out = (int)f_.get("mm.a.input_projection.weight").dims[1];
    if (n_embd % 16 || n_ff % 16 || n_out % 16 || n_embd / n_head % 16 || n_embd / n_head > 128 || n_mel != 128)
      throw std::runtime_error("mmproj: this Gemma 4 audio shape is not supported");
    build_mel_filters();
  }

  int out_dim() const override { return n_out; }

  // GPU memory of the encoder (weights, the relative-position projections, buffers for a 30-second clip).
  static size_t gpu_bytes(const GgufFile& g) {
    const size_t E = g.get_int("clip.audio.embedding_length", 0), F = g.get_int("clip.audio.feed_forward_length", 0);
    const size_t L = g.get_int("clip.audio.block_count", 0), W = 1536, K0 = 1024, Mp = 768;
    const size_t per = K0 * 4 + E * 4 + std::max(E, W) * 4 + std::max(F, 3 * E) * 4 + 2 * std::max<size_t>(E, W) * 4;
    return enc_weight_bytes(g, {"a.", "mm.a."}, {"a.conv1d.0.weight", "a.conv1d.1.weight"}) + L * 13 * E * 4 + Mp * per;
  }

  // 16 kHz samples -> n_tok x n_out embeddings: 30-second clips encoded one after the other, tokens concatenated.
  std::vector<float> encode(const std::vector<float>& s, int& n_tok) override {
    load();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> out;
    n_tok = 0;
    const size_t clip = 30 * 16000;
    for (size_t off = 0; off < s.size(); off += clip) {
      int n = 0;
      const std::vector<float> e = encode_clip(s.data() + off, std::min(clip, s.size() - off), n);
      out.insert(out.end(), e.begin(), e.end());
      n_tok += n;
    }
    encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
  }

 private:
  GgufFile f_;
  Engine& e_;
  struct FW {
    cl_mem w = nullptr;
    int N = 0, K = 0;
  };
  struct CW {
    FW w;
    float in_lo = -FLT_MAX, in_hi = FLT_MAX, out_lo = -FLT_MAX, out_hi = FLT_MAX;
  };
  struct alignas(32) F8 {
    float v[8];
  };
  struct Layer {
    cl_mem ff_norm, ff_post, ff_norm1, ff_post1, attn_pre, attn_post, pds, prel, conv_pre, conv_dw, conv_post, ln2;
    CW ff_up, ff_down, ff_up1, ff_down1, q, k, v, o, pw1, pw2;
  };
  bool loaded_ = false;
  std::vector<Layer> L_;
  FW inp_, outp_, mmp_;
  cl_mem outp_b_ = nullptr, ones_ = nullptr;
  std::vector<float> c0w_, c0n_, c1w_, c1n_, filt_;  // subsampling convs and their channel norms; mel filters
  int c0_ = 128, c1_ = 32;
  int cap_ = 0;
  cl_mem bin_ = nullptr, x_ = nullptr, hn_ = nullptr, h_ = nullptr, t_ = nullptr, o_ = nullptr;

  std::vector<float> host(const std::string& name) const {
    const GgufTensor& t = f_.get(name);
    std::vector<float> v(t.nelements());
    dequant_row(t.type, t.data, (int64_t)v.size(), v.data());
    return v;
  }
  float scalar(const std::string& name, float dflt) const { return f_.find(name) ? host(name)[0] : dflt; }
  cl_mem f32(const std::vector<float>& v) { return e_.buf(v.size() * 4, v.data()); }
  cl_mem f32(const std::string& name) { return f32(host(name)); }
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
  CW clw(const std::string& p) {
    CW c;
    c.w = f16(p + ".weight");
    c.in_lo = scalar(p + ".input_min", -FLT_MAX);
    c.in_hi = scalar(p + ".input_max", FLT_MAX);
    c.out_lo = scalar(p + ".output_min", -FLT_MAX);
    c.out_hi = scalar(p + ".output_max", FLT_MAX);
    return c;
  }

  void build_mel_filters() {  // HTK mel scale, 0..8000 Hz, no area normalization; [n_mel][257]
    const int nb = 257;
    auto hz2mel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
    auto mel2hz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
    std::vector<double> hz(n_mel + 2);
    const double lo = hz2mel(0.0), hi = hz2mel(8000.0);
    for (int i = 0; i < n_mel + 2; ++i) hz[i] = mel2hz(lo + (hi - lo) * i / (n_mel + 1));
    filt_.assign((size_t)n_mel * nb, 0.0f);
    for (int m = 0; m < n_mel; ++m) {
      const double fl = hz[m], fc = hz[m + 1], fr = hz[m + 2];
      for (int k = 0; k < nb; ++k) {
        const double f = k * 16000.0 / 512.0;
        double w = 0.0;
        if (f >= fl && f <= fc) w = (f - fl) / std::max(1e-30, fc - fl);
        else if (f > fc && f <= fr) w = (fr - f) / std::max(1e-30, fr - fc);
        filt_[(size_t)m * nb + k] = (float)w;
      }
    }
  }

  // log-mel frames [T][128] of one clip (llama.cpp's semicausal padding and frame count)
  std::vector<float> mel(const float* x, size_t n, int& T) const {
    const int win = 320, fft = 512, hop = 160, nb = 257, pad_left = win / 2;
    const int64_t n_left = (int64_t)n + pad_left;
    T = (int)((n_left - (win + 1)) / hop + 1);
    if (T < 1) T = 1;
    const int64_t need = (int64_t)(T - 1) * hop + fft;
    const int64_t total = std::max<int64_t>(need - (int64_t)n, pad_left) + (int64_t)n;
    std::vector<float> p((size_t)std::max<int64_t>(total, need), 0.0f);
    std::copy(x, x + n, p.begin() + pad_left);
    std::vector<float> hann(win);
    for (int i = 0; i < win; ++i) hann[i] = 0.5f - 0.5f * std::cos(2.0f * 3.14159265358979f * i / win);
    std::vector<float> cs((size_t)nb * win), sn((size_t)nb * win);  // DFT of the 320 windowed samples (the rest are 0)
    for (int k = 0; k < nb; ++k)
      for (int i = 0; i < win; ++i) {
        const double a = 2.0 * 3.14159265358979323846 * (double)k * i / fft;
        cs[(size_t)k * win + i] = (float)std::cos(a);
        sn[(size_t)k * win + i] = (float)std::sin(a);
      }
    std::vector<float> out((size_t)T * n_mel);
#pragma omp parallel for
    for (int t = 0; t < T; ++t) {
      float fr[320], mag[257];
      for (int i = 0; i < win; ++i) {
        const size_t j = (size_t)t * hop + i;
        fr[i] = j < p.size() ? hann[i] * p[j] : 0.0f;
      }
      for (int k = 0; k < nb; ++k) {
        float re = 0.0f, im = 0.0f;
        const float* c = &cs[(size_t)k * win];
        const float* s = &sn[(size_t)k * win];
        for (int i = 0; i < win; ++i) {
          re += fr[i] * c[i];
          im -= fr[i] * s[i];
        }
        mag[k] = std::sqrt(re * re + im * im);
      }
      for (int m = 0; m < n_mel; ++m) {
        double sum = 0.0;
        const float* fw = &filt_[(size_t)m * nb];
        for (int k = 0; k < nb; ++k) sum += (double)mag[k] * fw[k];
        out[(size_t)t * n_mel + m] = (float)std::log(std::max(sum, 0.001));
      }
    }
    return out;
  }

  // 3x3 stride-2 convolution (padding 1) over (time, freq), then LayerNorm over channels (weight, no bias) and ReLU.
  // in [Ci][T][F] -> out [Co][T'][F']; w [Co][Ci][3 (time)][3 (freq)]
  static std::vector<float> conv_ln_relu(const std::vector<float>& in, int Ci, int T, int F, const std::vector<float>& w,
                                         const std::vector<float>& nw, int Co, int& To, int& Fo) {
    To = (T - 1) / 2 + 1;
    Fo = (F - 1) / 2 + 1;
    std::vector<float> out((size_t)Co * To * Fo);
#pragma omp parallel for
    for (int t = 0; t < To; ++t) {
      std::vector<float> v(Co);
      for (int f = 0; f < Fo; ++f) {
        for (int co = 0; co < Co; ++co) {
          float s = 0.0f;
          for (int ci = 0; ci < Ci; ++ci)
            for (int kt = 0; kt < 3; ++kt) {
              const int ti = 2 * t + kt - 1;
              if (ti < 0 || ti >= T) continue;
              for (int kf = 0; kf < 3; ++kf) {
                const int fi = 2 * f + kf - 1;
                if (fi < 0 || fi >= F) continue;
                s += in[((size_t)ci * T + ti) * F + fi] * w[(((size_t)co * Ci + ci) * 3 + kt) * 3 + kf];
              }
            }
          v[co] = s;
        }
        double mean = 0.0, var = 0.0;
        for (int co = 0; co < Co; ++co) mean += v[co];
        mean /= Co;
        for (int co = 0; co < Co; ++co) var += (v[co] - mean) * (v[co] - mean);
        const float r = (float)(1.0 / std::sqrt(var / Co + 1e-6));
        for (int co = 0; co < Co; ++co) out[((size_t)co * To + t) * Fo + f] = std::max(0.0f, (float)(v[co] - mean) * r * nw[co]);
      }
    }
    return out;
  }

  void load() {
    if (loaded_) return;
    c0w_ = host("a.conv1d.0.weight");
    c0n_ = host("a.conv1d.0.norm.weight");
    c1w_ = host("a.conv1d.1.weight");
    c1n_ = host("a.conv1d.1.norm.weight");
    c0_ = (int)f_.get("a.conv1d.0.weight").dims[3];
    c1_ = (int)f_.get("a.conv1d.1.weight").dims[3];
    inp_ = f16("a.input_projection.weight");
    if (inp_.K != c1_ * (n_mel / 4) || inp_.N != n_embd) throw std::runtime_error("mmproj: unexpected Gemma 4 audio input projection");
    // sinusoidal relative positions 12 .. 0 (rows 0 .. 12), [13][E]
    const int E = n_embd, R = 13, half = E / 2;
    std::vector<float> pe((size_t)R * E);
    const float inc = std::log(10000.0f) / std::max(half - 1, 1);
    for (int r = 0; r < R; ++r)
      for (int i = 0; i < half; ++i) {
        const float a = (float)(12 - r) * std::exp(-(float)i * inc);
        pe[(size_t)r * E + i] = std::sin(a);
        pe[(size_t)r * E + i + half] = std::cos(a);
      }
    for (int il = 0; il < n_layer; ++il) {
      const std::string p = "a.blk." + std::to_string(il) + ".";
      Layer l;
      l.ff_norm = f32(p + "ffn_norm.weight");
      l.ff_post = f32(p + "ffn_post_norm.weight");
      l.ff_norm1 = f32(p + "ffn_norm_1.weight");
      l.ff_post1 = f32(p + "ffn_post_norm_1.weight");
      l.attn_pre = f32(p + "attn_pre_norm.weight");
      l.attn_post = f32(p + "attn_post_norm.weight");
      l.pds = f32(p + "per_dim_scale.weight");
      // (llama.cpp: the GGUF's conv_norm / norm_conv names are swapped: "conv_norm" is the norm before the module)
      l.conv_pre = f32(p + "conv_norm.weight");
      l.conv_post = f32(p + "norm_conv.weight");
      l.conv_dw = f32(p + "conv_dw.weight");
      l.ln2 = f32(p + "ln2.weight");
      l.ff_up = clw(p + "ffn_up");
      l.ff_down = clw(p + "ffn_down");
      l.ff_up1 = clw(p + "ffn_up_1");
      l.ff_down1 = clw(p + "ffn_down_1");
      l.q = clw(p + "attn_q");
      l.k = clw(p + "attn_k");
      l.v = clw(p + "attn_v");
      l.o = clw(p + "attn_out");
      l.pw1 = clw(p + "conv_pw1");
      l.pw2 = clw(p + "conv_pw2");
      if (l.k.in_lo != l.q.in_lo || l.k.in_hi != l.q.in_hi || l.v.in_lo != l.q.in_lo || l.v.in_hi != l.q.in_hi)
        throw std::runtime_error("mmproj: Gemma 4 audio layer " + std::to_string(il) + " has differing q / k / v input clamps");
      {  // the relative-position projection is constant: computed here, [13][E]
        const std::vector<float> kr = host(p + "attn_k_rel.weight");  // [E][E]
        std::vector<float> pr((size_t)R * E, 0.0f);
        for (int r = 0; r < R; ++r)
          for (int o = 0; o < E; ++o) {
            double s = 0.0;
            for (int i = 0; i < E; ++i) s += (double)kr[(size_t)o * E + i] * pe[(size_t)r * E + i];
            pr[(size_t)r * E + o] = (float)s;
          }
        l.prel = f32(pr);
      }
      L_.push_back(l);
    }
    outp_ = f16("a.pre_encode.out.weight");
    outp_b_ = f32("a.pre_encode.out.bias");
    mmp_ = f16("mm.a.input_projection.weight");
    {
      const std::vector<float> one(std::max(outp_.N, n_embd), 1.0f);
      ones_ = f32(one);
    }
    if (outp_.K != n_embd || mmp_.K != outp_.N) throw std::runtime_error("mmproj: unexpected Gemma 4 audio output projections");
    loaded_ = true;
  }
  void reserve(int Mp) {
    if (Mp <= cap_) return;
    const size_t E = n_embd, F = std::max<size_t>(n_ff, 3 * E), W = std::max<size_t>(outp_.N, n_out);
    bin_ = e_.buf((size_t)Mp * inp_.K * 4);
    x_ = e_.buf((size_t)Mp * E * 4);
    hn_ = e_.buf((size_t)Mp * std::max<size_t>(E, W) * 4);
    h_ = e_.buf((size_t)Mp * F * 4);
    t_ = e_.buf((size_t)Mp * std::max<size_t>(E, W) * 4);
    o_ = e_.buf((size_t)Mp * std::max<size_t>(E, W) * 4);
    cap_ = Mp;
  }

  std::vector<float> encode_clip(const float* x, size_t n, int& n_tok) {
    int T = 0;
    const std::vector<float> m = mel(x, n, T);  // [T][128] = one input channel over (time, freq)
    int T1, F1, T2, F2;
    const std::vector<float> a1 = conv_ln_relu(m, 1, T, n_mel, c0w_, c0n_, c0_, T1, F1);
    const std::vector<float> a2 = conv_ln_relu(a1, c0_, T1, F1, c1w_, c1n_, c1_, T2, F2);
    const int N = T2, E = n_embd, K0 = c1_ * F2;
    n_tok = N;
    const int Mp = (N + 127) / 128 * 128;
    reserve(Mp);
    std::vector<float> in((size_t)Mp * K0, 0.0f);  // per frame: feature f * C + c (llama.cpp's flatten order)
    for (int t = 0; t < N; ++t)
      for (int f = 0; f < F2; ++f)
        for (int c = 0; c < c1_; ++c) in[(size_t)t * K0 + f * c1_ + c] = a2[((size_t)c * T2 + t) * F2 + f];
    auto& q = e_.cl_.queue;
    CL_CHECK(clEnqueueWriteBuffer(q, bin_, CL_TRUE, 0, in.size() * 4, in.data(), 0, nullptr, nullptr));
    gemm(bin_, K0, inp_, x_, E, 0, Mp, N);
    const int D = E / n_head;
    for (const Layer& l : L_) {
      ffn(l.ff_norm, l.ff_up, l.ff_down, l.ff_post, Mp, N);
      // attention
      norm(x_, E, l.attn_pre, l.q.in_lo, l.q.in_hi, hn_, E, N);
      gemm(hn_, E, l.q.w, h_, 3 * E, 0, Mp, N);
      gemm(hn_, E, l.k.w, h_, 3 * E, E, Mp, N);
      gemm(hn_, E, l.v.w, h_, 3 * E, 2 * E, Mp, N);
      {
        cl_kernel k = kern("a4_attn");
        int nh = n_head, d = D;
        float qs = (1.0f / std::sqrt((float)D)) / std::log(2.0f), ks = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f), cap = 50.0f;
        F8 cl = {{l.q.out_lo, l.q.out_hi, l.k.out_lo, l.k.out_hi, l.v.out_lo, l.v.out_hi, 0.0f, 0.0f}};
        set_args(k, 0, h_, nh, d, l.prel, l.pds, qs, ks, cap, cl, o_);
        size_t g[2] = {(size_t)n_head * 16, (size_t)N}, lw[2] = {16, 1};
        e_.run("a4_attn", k, 2, g, lw);
      }
      clampc(o_, E, E, l.o.in_lo, l.o.in_hi, N);
      gemm(o_, E, l.o.w, t_, E, 0, Mp, N);
      norm_axpy(t_, E, l.attn_post, l.o.out_lo, l.o.out_hi, 1.0f, N);
      // convolution module
      norm(x_, E, l.conv_pre, l.pw1.in_lo, l.pw1.in_hi, hn_, E, N);
      gemm(hn_, E, l.pw1.w, h_, 2 * E, 0, Mp, N);
      {
        cl_kernel k = kern("a4_glu_dw");
        int e = E, K = 5;
        float lo = l.pw1.out_lo, hi = l.pw1.out_hi;
        set_args(k, 0, h_, e, K, l.conv_dw, lo, hi, t_);
        size_t g[2] = {(size_t)E, (size_t)N};
        e_.run("a4_conv", k, 2, g, nullptr);
      }
      norm(t_, E, l.conv_post, -FLT_MAX, FLT_MAX, hn_, E, N);
      silu(hn_, E, l.pw2.in_lo, l.pw2.in_hi, N);
      gemm(hn_, E, l.pw2.w, t_, E, 0, Mp, N);
      clampc(t_, E, E, l.pw2.out_lo, l.pw2.out_hi, N);
      add(x_, E, t_, true, N);
      ffn(l.ff_norm1, l.ff_up1, l.ff_down1, l.ff_post1, Mp, N);
      norm(x_, E, l.ln2, -FLT_MAX, FLT_MAX, x_, E, N);
    }
    gemm(x_, E, outp_, t_, outp_.N, 0, Mp, N);
    add(t_, outp_.N, outp_b_, false, N);
    normw(t_, outp_.N, hn_, N);
    gemm(hn_, outp_.N, mmp_, o_, n_out, 0, Mp, N);
    std::vector<float> out((size_t)N * n_out);
    CL_CHECK(clEnqueueReadBuffer(q, o_, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));
    e_.collect_profile();
    return out;
  }

  // half-step FFN: x += 0.5 * rms(down(silu(up(rms(x) * nw)))) * pw
  void ffn(cl_mem nw, const CW& up, const CW& down, cl_mem pw, int Mp, int N) {
    const int E = n_embd;
    norm(x_, E, nw, up.in_lo, up.in_hi, hn_, E, N);
    gemm(hn_, E, up.w, h_, n_ff, 0, Mp, N);
    clampc(h_, n_ff, n_ff, up.out_lo, up.out_hi, N);
    silu(h_, n_ff, down.in_lo, down.in_hi, N);
    gemm(h_, n_ff, down.w, t_, E, 0, Mp, N);
    norm_axpy(t_, E, pw, down.out_lo, down.out_hi, 0.5f, N);
  }

  cl_kernel kern(const char* name) { return e_.kernel("ops", "-DVIT", name); }
  void gemm(cl_mem A, int lda, const FW& w, cl_mem C, int ldc, int c_off, int M, int n) {
    cl_kernel k = e_.kernel("ops", "-DVIT -cl-intel-256-GRF-per-thread", "vit_gemm_f16");
    int ld = lda, K = w.K, lc = ldc, co = c_off, ac = 0, mv = n;
    set_args(k, 0, A, ld, w.w, K, C, lc, co, ac, mv);
    const int tiles = M / 16;
    int wgy = 8;
    while (tiles % wgy) wgy /= 2;
    size_t g[2] = {(size_t)w.N, (size_t)tiles}, lw[2] = {16, (size_t)wgy};
    e_.run("a4_gemm", k, 2, g, lw);
  }
  void norm(cl_mem x, int E, cl_mem w, float lo, float hi, cl_mem out, int ldo, int n) {
    cl_kernel k = kern("g4v_norm");
    int lx = E, e = E, lo2 = ldo;
    float ep = eps, a = lo, b = hi;
    set_args(k, 0, x, lx, e, w, ep, a, b, out, lo2);
    size_t g[2] = {256, (size_t)n}, lw[2] = {256, 1};
    e_.run("a4_norm", k, 2, g, lw);
  }
  void normw(cl_mem x, int E, cl_mem out, int n) {  // weightless
    cl_kernel k = kern("g4v_norm");
    int lx = E, e = E, lo2 = E;
    float ep = eps, a = -FLT_MAX, b = FLT_MAX;
    set_args(k, 0, x, lx, e, ones_, ep, a, b, out, lo2);
    size_t g[2] = {256, (size_t)n}, lw[2] = {256, 1};
    e_.run("a4_norm", k, 2, g, lw);
  }
  void norm_axpy(cl_mem y, int E, cl_mem w, float lo, float hi, float s, int n) {
    cl_kernel k = kern("a4_norm_axpy");
    int e = E;
    float ep = eps, a = lo, b = hi, sc = s;
    set_args(k, 0, y, e, w, ep, a, b, sc, x_);
    size_t g[2] = {256, (size_t)n}, lw[2] = {256, 1};
    e_.run("a4_norm", k, 2, g, lw);
  }
  void silu(cl_mem x, int N, float lo, float hi, int n) {
    cl_kernel k = kern("a4_silu");
    int nn = N;
    float a = lo, b = hi;
    set_args(k, 0, x, nn, a, b);
    size_t g[2] = {(size_t)N, (size_t)n};
    e_.run("a4_silu", k, 2, g, nullptr);
  }
  void clampc(cl_mem x, int ld, int N, float lo, float hi, int n) {
    if (lo == -FLT_MAX && hi == FLT_MAX) return;
    cl_kernel k = kern("g4v_clamp");
    int l = ld;
    float a = lo, b = hi;
    set_args(k, 0, x, l, a, b);
    size_t g[2] = {(size_t)N, (size_t)n};
    e_.run("a4_clamp", k, 2, g, nullptr);
  }
  void add(cl_mem x, int N, cl_mem a, bool per_row, int n) {
    cl_kernel k = kern("vit_add");
    int nn = N, pr = per_row ? 1 : 0;
    set_args(k, 0, x, nn, a, pr);
    size_t g[2] = {(size_t)N, (size_t)n};
    e_.run("a4_add", k, 2, g, nullptr);
  }
};
