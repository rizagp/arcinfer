// Gemma 4 "unified" audio embedder (mmproj audio projector type gemma4ua, e.g. Gemma 4 12B): there is no audio
// encoder; the 16 kHz waveform is cut into 640-sample frames (40 ms, one token each, the last one zero-padded),
// every frame is RMS-normalized (no weight) and projected into the language model's embedding space, and the
// language model itself reads the frames (causally). Matches llama.cpp tools/mtmd (models/gemma4ua.cpp,
// mtmd_audio_preprocessor_gemma4ua). Runs on the CPU (~2.5 M multiply-adds per token).
#pragma once
#include "gguf.h"
#include "quants.h"
#include <cmath>
#include <string>
#include <vector>

// An audio encoder of a Gemma 4 model (the server holds either kind): 16 kHz samples -> n_tok x out_dim() embeddings.
class GemmaAudioEncoder {
 public:
  virtual ~GemmaAudioEncoder() = default;
  virtual int out_dim() const = 0;
  virtual std::vector<float> encode(const std::vector<float>& s, int& n_tok) = 0;
};

class AudioGemma4 : public GemmaAudioEncoder {
 public:
  int frame = 640, n_out = 0, rate = 16000;
  float eps = 1e-6f;

  explicit AudioGemma4(const std::string& path) : f_(path) {
    if (f_.get_str("clip.audio.projector_type") != "gemma4ua")
      throw std::runtime_error("mmproj: only the Gemma 4 unified audio projector (gemma4ua) is supported");
    const GgufTensor& t = f_.get("mm.a.input_projection.weight");
    frame = (int)t.dims[0];
    n_out = (int)t.dims[1];
    proj_.resize((size_t)frame * n_out);
    dequant_row(t.type, t.data, (int)proj_.size(), proj_.data());
  }

  // Tokens of a clip of n 16 kHz samples.
  int tokens(size_t n) const { return (int)((n + frame - 1) / frame); }

  // 16 kHz samples -> n_tok x n_out embeddings (row-major).
  int out_dim() const override { return n_out; }
  std::vector<float> encode(const std::vector<float>& s, int& n_tok) override {
    n_tok = tokens(s.size());
    std::vector<float> out((size_t)n_tok * n_out);
#pragma omp parallel for
    for (int t = 0; t < n_tok; ++t) {
      std::vector<float> x(frame, 0.0f);
      double ss = 0;
      for (int i = 0; i < frame; ++i) {
        const size_t k = (size_t)t * frame + i;
        x[i] = k < s.size() ? s[k] : 0.0f;
        ss += (double)x[i] * x[i];
      }
      const float r = (float)(1.0 / std::sqrt(ss / frame + eps));
      for (int i = 0; i < frame; ++i) x[i] *= r;
      float* o = &out[(size_t)t * n_out];
      for (int j = 0; j < n_out; ++j) {
        const float* w = &proj_[(size_t)j * frame];
        float acc = 0;
        for (int i = 0; i < frame; ++i) acc += w[i] * x[i];
        o[j] = acc;
      }
    }
    return out;
  }

 private:
  GgufFile f_;
  std::vector<float> proj_;  // [n_out][frame]
};
