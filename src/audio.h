// Audio input: decoding to 16 kHz mono float samples. WAV (PCM 8/16/24/32-bit or float32, any rate and channel
// count) is parsed here; other containers (mp3, m4a/aac, flac, wma, ...) go through Windows Media Foundation.
// Resampling: windowed-sinc low-pass interpolation (Blackman window).
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <shlwapi.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "ole32.lib")

namespace audio {

// RIFF/WAVE: returns false when the bytes are not a WAV file; throws on an unsupported WAV encoding.
inline bool parse_wav(const std::vector<uint8_t>& b, std::vector<float>& mono, int& rate) {
  auto u16 = [&](size_t o) { return (uint32_t)b[o] | (uint32_t)b[o + 1] << 8; };
  auto u32 = [&](size_t o) { return u16(o) | u16(o + 2) << 16; };
  if (b.size() < 12 || memcmp(b.data(), "RIFF", 4) || memcmp(b.data() + 8, "WAVE", 4)) return false;
  int fmt = 0, ch = 0, bits = 0;
  size_t p = 12;
  while (p + 8 <= b.size()) {
    const uint32_t len = u32(p + 4);
    const size_t body = p + 8, avail = std::min<size_t>(len, b.size() - body);
    if (!memcmp(b.data() + p, "fmt ", 4) && avail >= 16) {
      fmt = (int)u16(body);
      ch = (int)u16(body + 2);
      rate = (int)u32(body + 4);
      bits = (int)u16(body + 14);
      if (fmt == 0xFFFE && avail >= 26) fmt = (int)u16(body + 24);  // WAVE_FORMAT_EXTENSIBLE: sub-format GUID
    } else if (!memcmp(b.data() + p, "data", 4)) {
      if (ch <= 0 || rate <= 0) throw std::runtime_error("wav: data before fmt");
      const int bps = bits / 8;
      if (!((fmt == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (fmt == 3 && bits == 32)))
        throw std::runtime_error("wav: unsupported encoding (PCM 8/16/24/32-bit or float32 expected)");
      const size_t n = avail / ((size_t)bps * ch);
      mono.assign(n, 0.0f);
      const uint8_t* d = b.data() + body;
      for (size_t i = 0; i < n; ++i) {
        float acc = 0;
        for (int c = 0; c < ch; ++c) {
          const uint8_t* s = d + (i * ch + c) * bps;
          float v;
          if (fmt == 3) memcpy(&v, s, 4);
          else if (bits == 8) v = ((int)s[0] - 128) / 128.0f;
          else if (bits == 16) v = (int16_t)(s[0] | s[1] << 8) / 32768.0f;
          else if (bits == 24) v = (float)((int32_t)((uint32_t)s[0] << 8 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 24) >> 8) / 8388608.0f;
          else { int32_t x; memcpy(&x, s, 4); v = x / 2147483648.0f; }
          acc += v;
        }
        mono[i] = acc / ch;
      }
      return true;
    }
    p = body + len + (len & 1);
  }
  throw std::runtime_error("wav: no data chunk");
}

// Any container Media Foundation can read (mp3, m4a/aac, flac, wma, wav, ...), decoded to float, mixed to mono.
inline std::vector<float> decode_mf(const std::vector<uint8_t>& bytes, int& rate) {
  struct Com {
    bool ok;
    Com() { ok = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED)); MFStartup(MF_VERSION, MFSTARTUP_LITE); }
    ~Com() { MFShutdown(); if (ok) CoUninitialize(); }
  } com;
  auto check = [](HRESULT hr, const char* what) {
    if (FAILED(hr)) throw std::runtime_error(std::string("audio decoding failed (") + what + ")");
  };
  IStream* st = SHCreateMemStream(bytes.data(), (UINT)bytes.size());
  if (!st) throw std::runtime_error("audio: out of memory");
  IMFByteStream* bs = nullptr;
  IMFSourceReader* rd = nullptr;
  IMFMediaType* mt = nullptr;
  IMFMediaType* cur = nullptr;
  std::vector<float> mono;
  try {
    check(MFCreateMFByteStreamOnStream(st, &bs), "byte stream");
    check(MFCreateSourceReaderFromByteStream(bs, nullptr, &rd), "unrecognized audio format");
    check(MFCreateMediaType(&mt), "media type");
    mt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
    mt->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);
    check(rd->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, mt), "float output");
    check(rd->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, &cur), "output type");
    UINT32 sr = 0, ch = 0;
    cur->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &sr);
    cur->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &ch);
    if (!sr || !ch) throw std::runtime_error("audio: unknown sample rate or channel count");
    rate = (int)sr;
    for (;;) {
      DWORD flags = 0;
      IMFSample* s = nullptr;
      check(rd->ReadSample((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, nullptr, &flags, nullptr, &s), "read");
      if (s) {
        IMFMediaBuffer* mb = nullptr;
        if (SUCCEEDED(s->ConvertToContiguousBuffer(&mb))) {
          BYTE* p = nullptr;
          DWORD len = 0;
          if (SUCCEEDED(mb->Lock(&p, nullptr, &len))) {
            const size_t n = len / (4 * ch);
            const float* f = (const float*)p;
            for (size_t i = 0; i < n; ++i) {
              float acc = 0;
              for (UINT32 c = 0; c < ch; ++c) acc += f[i * ch + c];
              mono.push_back(acc / ch);
            }
            mb->Unlock();
          }
          mb->Release();
        }
        s->Release();
      }
      if (flags & (MF_SOURCE_READERF_ENDOFSTREAM | MF_SOURCE_READERF_ERROR)) break;
    }
  } catch (...) {
    if (cur) cur->Release();
    if (mt) mt->Release();
    if (rd) rd->Release();
    if (bs) bs->Release();
    st->Release();
    throw;
  }
  cur->Release();
  mt->Release();
  rd->Release();
  bs->Release();
  st->Release();
  return mono;
}

// Windowed-sinc resampling (low-pass at 0.97 x the lower Nyquist frequency, Blackman window, ~10 lobes per side).
inline std::vector<float> resample(const std::vector<float>& x, int from, int to) {
  if (from == to || x.empty()) return x;
  const double r = (double)to / from, fc = 0.5 * std::min(1.0, r) * 0.97;  // cutoff, cycles per input sample
  const int W = (int)std::ceil(10.0 / (2.0 * fc));                       // half-width in input samples
  const size_t n = (size_t)std::floor((double)x.size() * r);
  std::vector<float> y(n);
  const double pi = 3.14159265358979323846;
#pragma omp parallel for if (n > 100000)
  for (int64_t i = 0; i < (int64_t)n; ++i) {
    const double c = i / r;  // position in input samples
    const int64_t j0 = (int64_t)std::floor(c) - W + 1, j1 = (int64_t)std::floor(c) + W;
    double acc = 0, wsum = 0;
    for (int64_t j = j0; j <= j1; ++j) {
      const double d = c - j, a = 2 * fc * d;
      const double s = std::fabs(a) < 1e-12 ? 1.0 : std::sin(pi * a) / (pi * a);
      const double t = (d + W) / (2.0 * W);  // window position 0..1
      const double w = (t <= 0 || t >= 1) ? 0 : 0.42 - 0.5 * std::cos(2 * pi * t) + 0.08 * std::cos(4 * pi * t);
      const double h = 2 * fc * s * w;
      wsum += h;
      if (j >= 0 && j < (int64_t)x.size()) acc += h * x[(size_t)j];
    }
    y[(size_t)i] = (float)(wsum != 0 ? acc / wsum : 0);
  }
  return y;
}

// Any supported audio file -> 16 kHz mono samples.
inline std::vector<float> decode_16k(const std::vector<uint8_t>& bytes) {
  std::vector<float> s;
  int rate = 0;
  if (!parse_wav(bytes, s, rate)) s = decode_mf(bytes, rate);
  if (s.empty()) throw std::runtime_error("audio: no samples");
  // (a hostile or broken file: an odd sample rate makes the resampling window huge, an hours-long clip takes
  //  minutes of CPU and more context than any model has)
  if (rate < 1000 || rate > 384000) throw std::runtime_error("audio: unsupported sample rate " + std::to_string(rate));
  if (s.size() / (size_t)rate > 600) throw std::runtime_error("audio: longer than 10 minutes");
  return resample(s, rate, 16000);
}

}  // namespace audio
