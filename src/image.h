// Image input for vision models: decoding (PNG, JPEG, GIF, BMP, TIFF, WebP when the codec is installed) through
// the Windows Imaging Component, base64 data URLs, and a Pillow-compatible bicubic resize.
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#include <wincodec.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

struct ImageRGB {
  int w = 0, h = 0;
  std::vector<uint8_t> px;  // w*h*3, row-major RGB
};

namespace img {

inline std::vector<uint8_t> base64_decode(const std::string& s) {
  static int8_t T[256];
  static bool init = false;
  if (!init) {
    for (int i = 0; i < 256; ++i) T[i] = -1;
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) T[(uint8_t)a[i]] = (int8_t)i;
    T['-'] = 62;  // URL-safe variants
    T['_'] = 63;
    init = true;
  }
  std::vector<uint8_t> out;
  out.reserve(s.size() * 3 / 4);
  uint32_t acc = 0;
  int bits = 0;
  for (unsigned char c : s) {
    const int v = T[c];
    if (v < 0) continue;  // padding, whitespace
    acc = (acc << 6) | (uint32_t)v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back((uint8_t)(acc >> bits));
    }
  }
  return out;
}

// "data:image/png;base64,...." -> bytes. Remote URLs are not fetched.
inline std::vector<uint8_t> from_data_url(const std::string& url) {
  if (url.compare(0, 5, "data:") != 0) throw std::runtime_error("only data: image URLs are supported (no remote fetching)");
  const size_t comma = url.find(',');
  if (comma == std::string::npos || url.find(";base64", 0) > comma) throw std::runtime_error("image data URL must be base64");
  return base64_decode(url.substr(comma + 1));
}

inline ImageRGB decode(const std::vector<uint8_t>& bytes) {
  static thread_local bool com = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) || true;
  (void)com;
  IWICImagingFactory* f = nullptr;
  if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f))))
    throw std::runtime_error("WIC unavailable");
  IWICStream* st = nullptr;
  IWICBitmapDecoder* dec = nullptr;
  IWICBitmapFrameDecode* fr = nullptr;
  IWICFormatConverter* cv = nullptr;
  ImageRGB im;
  auto release = [&]() {
    if (cv) cv->Release();
    if (fr) fr->Release();
    if (dec) dec->Release();
    if (st) st->Release();
    f->Release();
  };
  HRESULT hr = f->CreateStream(&st);
  if (SUCCEEDED(hr)) hr = st->InitializeFromMemory((BYTE*)bytes.data(), (DWORD)bytes.size());
  if (SUCCEEDED(hr)) hr = f->CreateDecoderFromStream(st, nullptr, WICDecodeMetadataCacheOnDemand, &dec);
  if (SUCCEEDED(hr)) hr = dec->GetFrame(0, &fr);
  if (SUCCEEDED(hr)) hr = f->CreateFormatConverter(&cv);
  if (SUCCEEDED(hr)) hr = cv->Initialize(fr, GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
  UINT w = 0, h = 0;
  if (SUCCEEDED(hr)) hr = cv->GetSize(&w, &h);
  // (a small file can declare a huge picture: more than 16384 pixels a side or 64 megapixels is refused before the
  //  pixels are allocated; the encoders take ~1-4 megapixels anyway)
  if (SUCCEEDED(hr) && (w > 16384 || h > 16384 || (uint64_t)w * h > (64ull << 20))) hr = E_FAIL;
  if (SUCCEEDED(hr)) {
    im.w = (int)w;
    im.h = (int)h;
    im.px.resize((size_t)w * h * 3);
    hr = cv->CopyPixels(nullptr, w * 3, (UINT)im.px.size(), im.px.data());
  }
  release();
  if (FAILED(hr) || im.w <= 0 || im.h <= 0) throw std::runtime_error("cannot decode image");
  return im;
}

// Pillow-compatible separable resampling with the bicubic filter (a = -0.5), 8-bit fixed-point accumulation.
inline ImageRGB resize_bicubic(const ImageRGB& src, int W, int H) {
  if (src.w == W && src.h == H) return src;
  auto cubic = [](double x) {
    const double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
  };
  // per output coordinate: first input index and normalized weights (Pillow's precompute_coeffs)
  auto coeffs = [&](int in, int out, std::vector<int>& bounds, std::vector<double>& k, int& ksize) {
    const double scale = (double)in / out, fscale = std::max(1.0, scale), support = 2.0 * fscale;
    ksize = (int)std::ceil(support) * 2 + 1;
    bounds.assign((size_t)out * 2, 0);
    k.assign((size_t)out * ksize, 0.0);
    for (int x = 0; x < out; ++x) {
      const double center = (x + 0.5) * scale;
      int xmin = (int)(center - support + 0.5), xmax = (int)(center + support + 0.5);
      xmin = std::max(xmin, 0);
      xmax = std::min(xmax, in) - xmin;
      double ww = 0.0;
      for (int i = 0; i < xmax; ++i) {
        const double v = cubic((i + xmin - center + 0.5) / fscale);
        k[(size_t)x * ksize + i] = v;
        ww += v;
      }
      for (int i = 0; i < xmax; ++i) if (ww != 0.0) k[(size_t)x * ksize + i] /= ww;
      bounds[2 * x] = xmin;
      bounds[2 * x + 1] = xmax;
    }
  };
  auto clip8 = [](double v) { return (uint8_t)std::min(255.0, std::max(0.0, std::floor(v + 0.5))); };
  // horizontal pass
  ImageRGB tmp;
  tmp.w = W;
  tmp.h = src.h;
  tmp.px.resize((size_t)W * src.h * 3);
  {
    std::vector<int> b;
    std::vector<double> k;
    int ks;
    coeffs(src.w, W, b, k, ks);
#pragma omp parallel for
    for (int y = 0; y < src.h; ++y)
      for (int x = 0; x < W; ++x)
        for (int c = 0; c < 3; ++c) {
          double acc = 0.0;
          for (int i = 0; i < b[2 * x + 1]; ++i) acc += src.px[((size_t)y * src.w + b[2 * x] + i) * 3 + c] * k[(size_t)x * ks + i];
          tmp.px[((size_t)y * W + x) * 3 + c] = clip8(acc);
        }
  }
  ImageRGB dst;
  dst.w = W;
  dst.h = H;
  dst.px.resize((size_t)W * H * 3);
  {
    std::vector<int> b;
    std::vector<double> k;
    int ks;
    coeffs(src.h, H, b, k, ks);
#pragma omp parallel for
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int c = 0; c < 3; ++c) {
          double acc = 0.0;
          for (int i = 0; i < b[2 * y + 1]; ++i) acc += tmp.px[((size_t)(b[2 * y] + i) * W + x) * 3 + c] * k[(size_t)y * ks + i];
          dst.px[((size_t)y * W + x) * 3 + c] = clip8(acc);
        }
  }
  return dst;
}

// Target size that keeps the aspect ratio, with both sides multiples of `align` and the pixel count within
// [min_pixels, max_pixels] (llama.cpp mtmd's calc_size_preserved_ratio).
inline void fit_size(int w, int h, int align, int64_t min_pixels, int64_t max_pixels, int& W, int& H) {
  auto round_f = [&](double x) { return (int)std::round(x / align) * align; };
  auto ceil_f = [&](double x) { return (int)std::ceil(x / align) * align; };
  auto floor_f = [&](double x) { return (int)std::floor(x / align) * align; };
  H = std::max(align, round_f(h));
  W = std::max(align, round_f(w));
  if (max_pixels > 0 && (int64_t)H * W > max_pixels) {
    const double beta = std::sqrt((double)h * w / max_pixels);
    H = std::max(align, floor_f(h / beta));
    W = std::max(align, floor_f(w / beta));
  } else if (min_pixels > 0 && (int64_t)H * W < min_pixels) {
    const double beta = std::sqrt((double)min_pixels / ((double)h * w));
    H = ceil_f(h * beta);
    W = ceil_f(w * beta);
  }
}

}  // namespace img
