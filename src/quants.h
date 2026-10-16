// GGML quant block decoding (CPU reference) and repacking into arcinfer's GPU layouts.
//
// GPU layouts (all per 16-column group g and 256-element superblock s; "lane" = column within group):
//   wq: [g][s][j=0..3][DPJ][lane] uint   -- data for the 64-element chunk j
//   wh: [g][s][HDR][lane] uint           -- per-superblock scales
// Word w (0..15) of a chunk pairs with activation bytes 64j+4w..64j+4w+3. Words 0..7 come from the low
// nibbles ("lo"), 8..15 from the high nibbles ("hi") of the 8 packed uints.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "gguf.h"
#include "iq_tables.h"

inline float fp16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, f;
  if (exp == 0) {
    if (mant == 0) f = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) { mant <<= 1; --exp; }
      f = sign | (exp << 23) | ((mant & 0x3FF) << 13);
    }
  } else if (exp == 31) f = sign | 0x7F800000 | (mant << 13);
  else f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  float r;
  memcpy(&r, &f, 4);
  return r;
}

inline uint16_t f32_to_fp16(float x) {
  uint32_t f;
  memcpy(&f, &x, 4);
  uint32_t sign = (f >> 16) & 0x8000;
  int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = f & 0x7FFFFF;
  if (((f >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00 | (mant ? 0x200 : 0));
  if (exp >= 31) return (uint16_t)(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000;
    uint32_t shift = 14 - exp;
    uint32_t h = mant >> shift, rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) ++h;
    return (uint16_t)(sign | h);
  }
  uint32_t h = sign | (exp << 10) | (mant >> 13), rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
  return (uint16_t)h;
}

inline float bf16_to_f32(uint16_t h) {
  uint32_t f = (uint32_t)h << 16;
  float r;
  memcpy(&r, &f, 4);
  return r;
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// Decoded superblock: integer quants q[256] with the per-type scale structure.
struct DecodedSB {
  float d = 0, dmin = 0;   // Q4_K/Q5_K: d, dmin; Q6_K: d
  uint8_t sc[16] = {}, mn[8] = {};
  int8_t sc6[16] = {};     // Q6_K signed scales per 16
  float d8[8] = {};        // Q8_0 scale per 32
  int16_t q[256] = {};     // Q4_K: 0..15, Q5_K: 0..31, Q6_K: 0..63, Q8_0: -128..127, IQ*/Q2_K: W8 value
  // Generic "W8" view (all types except Q8_0): w[e] = wd * wsc[e/16] * (q[e] - qoff) - wdmin * wmn[e/16]
  float wd = 0, wdmin = 0;
  int8_t wsc[16] = {};
  uint8_t wmn[16] = {};
  int qoff = 0;
  bool has_min = false;
};

inline uint16_t rd16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
inline uint32_t rd32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }

inline void decode_sb(uint32_t type, const uint8_t* p, DecodedSB& o) {
  switch (type) {
    case GGML_Q4_K: case GGML_Q5_K: {
      uint16_t dh, mh;
      memcpy(&dh, p, 2);
      memcpy(&mh, p + 2, 2);
      o.d = fp16_to_f32(dh);
      o.dmin = fp16_to_f32(mh);
      const uint8_t* scales = p + 4;
      for (int b = 0; b < 8; ++b) get_scale_min_k4(b, scales, o.sc[b], o.mn[b]);
      o.wd = o.d; o.wdmin = o.dmin; o.has_min = true; o.qoff = 0;
      for (int b = 0; b < 16; ++b) { o.wsc[b] = (int8_t)o.sc[b / 2]; o.wmn[b] = o.mn[b / 2]; }
      const uint8_t* qh = type == GGML_Q5_K ? p + 16 : nullptr;
      const uint8_t* qs = type == GGML_Q5_K ? p + 48 : p + 16;
      for (int j = 0; j < 4; ++j)
        for (int l = 0; l < 32; ++l) {
          int lo = qs[32 * j + l] & 0xF, hi = qs[32 * j + l] >> 4;
          if (qh) {
            lo |= ((qh[l] >> (2 * j)) & 1) << 4;
            hi |= ((qh[l] >> (2 * j + 1)) & 1) << 4;
          }
          o.q[64 * j + l] = (int16_t)lo;
          o.q[64 * j + 32 + l] = (int16_t)hi;
        }
      return;
    }
    case GGML_Q6_K: {
      const uint8_t* ql = p;
      const uint8_t* qh = p + 128;
      const int8_t* sc = (const int8_t*)(p + 192);
      uint16_t dh;
      memcpy(&dh, p + 208, 2);
      o.d = fp16_to_f32(dh);
      for (int i = 0; i < 16; ++i) o.sc6[i] = sc[i];
      o.wd = o.d; o.wdmin = 0; o.has_min = false; o.qoff = 32;
      for (int i = 0; i < 16; ++i) o.wsc[i] = sc[i];
      for (int n = 0; n < 2; ++n)
        for (int l = 0; l < 32; ++l) {
          const uint8_t* L = ql + 64 * n;
          const uint8_t* H = qh + 32 * n;
          o.q[128 * n + l + 0] = (int16_t)((L[l] & 0xF) | (((H[l] >> 0) & 3) << 4));
          o.q[128 * n + l + 32] = (int16_t)((L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4));
          o.q[128 * n + l + 64] = (int16_t)((L[l] >> 4) | (((H[l] >> 4) & 3) << 4));
          o.q[128 * n + l + 96] = (int16_t)((L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4));
        }
      return;
    }
    case GGML_Q4_0: {  // 8 blocks of 32 (18 bytes: f16 d, low nibbles = 0..15, high = 16..31), as Q8_0 with q - 8
      for (int b = 0; b < 8; ++b) {
        const uint8_t* blk = p + 18 * b;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        o.d8[b] = fp16_to_f32(dh);
        for (int i = 0; i < 16; ++i) {
          o.q[32 * b + i] = (int16_t)((blk[2 + i] & 0xF) - 8);
          o.q[32 * b + 16 + i] = (int16_t)((blk[2 + i] >> 4) - 8);
        }
      }
      return;
    }
    case GGML_Q8_0: {
      for (int b = 0; b < 8; ++b) {
        const uint8_t* blk = p + 34 * b;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        o.d8[b] = fp16_to_f32(dh);
        for (int i = 0; i < 32; ++i) o.q[32 * b + i] = (int8_t)blk[2 + i];
      }
      return;
    }
    case GGML_Q2_K: {
      const uint8_t* sc = p;
      const uint8_t* qs = p + 16;
      o.wd = fp16_to_f32(rd16(p + 80)); o.wdmin = fp16_to_f32(rd16(p + 82)); o.has_min = true; o.qoff = 0;
      for (int b = 0; b < 16; ++b) { o.wsc[b] = (int8_t)(sc[b] & 0xF); o.wmn[b] = sc[b] >> 4; }
      for (int e = 0; e < 256; ++e) {
        const int n = e / 128, r = e % 128, j = r / 32, hf = (r % 32) / 16, l = r % 16;
        o.q[e] = (int16_t)((qs[32 * n + 16 * hf + l] >> (2 * j)) & 3);
      }
      return;
    }
    case GGML_IQ2_XXS: {
      const float d = fp16_to_f32(rd16(p));
      const uint8_t* qs = p + 2;
      o.wd = d / 8; o.has_min = false; o.qoff = 0;
      for (int ib = 0; ib < 8; ++ib) {
        const uint32_t a0 = rd32(qs + 8 * ib), a1 = rd32(qs + 8 * ib + 4);
        o.wsc[2 * ib] = o.wsc[2 * ib + 1] = (int8_t)(1 + 2 * (a1 >> 28));
        for (int l = 0; l < 4; ++l) {
          const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + ((a0 >> (8 * l)) & 0xFF));
          const uint8_t signs = ksigns_iq2xs[(a1 >> (7 * l)) & 127];
          for (int j = 0; j < 8; ++j) o.q[32 * ib + 8 * l + j] = (int16_t)(grid[j] * ((signs & kmask_iq2xs[j]) ? -1 : 1));
        }
      }
      return;
    }
    case GGML_IQ2_XS: {
      const float d = fp16_to_f32(rd16(p));
      const uint8_t* qs = p + 2;
      const uint8_t* scales = p + 66;
      o.wd = d / 8; o.has_min = false; o.qoff = 0;
      for (int ib = 0; ib < 8; ++ib) {
        o.wsc[2 * ib] = (int8_t)(1 + 2 * (scales[ib] & 0xF));
        o.wsc[2 * ib + 1] = (int8_t)(1 + 2 * (scales[ib] >> 4));
        for (int l = 0; l < 4; ++l) {
          const uint16_t q16 = rd16(qs + 2 * (4 * ib + l));
          const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (q16 & 511));
          const uint8_t signs = ksigns_iq2xs[q16 >> 9];
          for (int j = 0; j < 8; ++j) o.q[32 * ib + 8 * l + j] = (int16_t)(grid[j] * ((signs & kmask_iq2xs[j]) ? -1 : 1));
        }
      }
      return;
    }
    case GGML_IQ2_S: {
      const float d = fp16_to_f32(rd16(p));
      const uint8_t* qs = p + 2;
      const uint8_t* signs = qs + 32;
      const uint8_t* qh = p + 66;
      const uint8_t* scales = p + 74;
      o.wd = d / 8; o.has_min = false; o.qoff = 0;
      for (int ib = 0; ib < 8; ++ib) {
        o.wsc[2 * ib] = (int8_t)(1 + 2 * (scales[ib] & 0xF));
        o.wsc[2 * ib + 1] = (int8_t)(1 + 2 * (scales[ib] >> 4));
        for (int l = 0; l < 4; ++l) {
          const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)));
          for (int j = 0; j < 8; ++j)
            o.q[32 * ib + 8 * l + j] = (int16_t)(grid[j] * ((signs[4 * ib + l] & kmask_iq2xs[j]) ? -1 : 1));
        }
      }
      return;
    }
    case GGML_IQ3_XXS: {
      const float d = fp16_to_f32(rd16(p));
      const uint8_t* qs = p + 2;
      const uint8_t* ss = qs + 64;
      o.wd = d / 4; o.has_min = false; o.qoff = 0;
      for (int ib = 0; ib < 8; ++ib) {
        const uint32_t a = rd32(ss + 4 * ib);
        o.wsc[2 * ib] = o.wsc[2 * ib + 1] = (int8_t)(1 + 2 * (a >> 28));
        for (int l = 0; l < 4; ++l) {
          const uint8_t signs = ksigns_iq2xs[(a >> (7 * l)) & 127];
          const uint8_t* g1 = (const uint8_t*)(iq3xxs_grid + qs[8 * ib + 2 * l]);
          const uint8_t* g2 = (const uint8_t*)(iq3xxs_grid + qs[8 * ib + 2 * l + 1]);
          for (int j = 0; j < 4; ++j) {
            o.q[32 * ib + 8 * l + j] = (int16_t)(g1[j] * ((signs & kmask_iq2xs[j]) ? -1 : 1));
            o.q[32 * ib + 8 * l + j + 4] = (int16_t)(g2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1 : 1));
          }
        }
      }
      return;
    }
    case GGML_IQ3_S: {
      const float d = fp16_to_f32(rd16(p));
      const uint8_t* qs = p + 2;
      const uint8_t* qh = p + 66;
      const uint8_t* signs = p + 74;
      const uint8_t* scales = p + 106;
      o.wd = d; o.has_min = false; o.qoff = 0;
      for (int ib = 0; ib < 8; ++ib) {
        const int sc = 1 + 2 * ((scales[ib / 2] >> (4 * (ib % 2))) & 0xF);
        o.wsc[2 * ib] = o.wsc[2 * ib + 1] = (int8_t)sc;
        const uint8_t h = qh[ib];
        for (int l = 0; l < 4; ++l) {
          const uint8_t* g1 = (const uint8_t*)(iq3s_grid + (qs[8 * ib + 2 * l] | ((h << (8 - 2 * l)) & 256)));
          const uint8_t* g2 = (const uint8_t*)(iq3s_grid + (qs[8 * ib + 2 * l + 1] | ((h << (7 - 2 * l)) & 256)));
          const uint8_t sg = signs[4 * ib + l];
          for (int j = 0; j < 4; ++j) {
            o.q[32 * ib + 8 * l + j] = (int16_t)(g1[j] * ((sg & kmask_iq2xs[j]) ? -1 : 1));
            o.q[32 * ib + 8 * l + j + 4] = (int16_t)(g2[j] * ((sg & kmask_iq2xs[j + 4]) ? -1 : 1));
          }
        }
      }
      return;
    }
    case GGML_IQ1_M: {
      const uint8_t* qs = p;
      const uint8_t* qh = p + 32;
      const uint16_t sc[4] = {rd16(p + 48), rd16(p + 50), rd16(p + 52), rd16(p + 54)};
      const uint16_t su = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
      o.wd = fp16_to_f32(su) / 8; o.has_min = false; o.qoff = 0;
      for (int ib = 0; ib < 8; ++ib) {
        o.wsc[2 * ib] = (int8_t)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1);
        o.wsc[2 * ib + 1] = (int8_t)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
        const uint8_t h0 = qh[2 * ib], h1 = qh[2 * ib + 1];
        const int idx[4] = {qs[4 * ib] | ((h0 << 8) & 0x700), qs[4 * ib + 1] | ((h0 << 4) & 0x700),
                            qs[4 * ib + 2] | ((h1 << 8) & 0x700), qs[4 * ib + 3] | ((h1 << 4) & 0x700)};
        const int dl[4] = {(h0 & 0x08) ? -1 : 1, (h0 & 0x80) ? -1 : 1, (h1 & 0x08) ? -1 : 1, (h1 & 0x80) ? -1 : 1};
        for (int l = 0; l < 4; ++l) {
          const int8_t* grid = (const int8_t*)(iq1s_grid + idx[l]);
          for (int j = 0; j < 8; ++j) o.q[32 * ib + 8 * l + j] = (int16_t)(8 * grid[j] + dl[l]);
        }
      }
      return;
    }
    case GGML_IQ4_XS: {
      const float d = fp16_to_f32(rd16(p));
      const uint16_t sh = rd16(p + 2);
      const uint8_t* sl = p + 4;
      const uint8_t* qs = p + 8;
      o.wd = d; o.has_min = false; o.qoff = 0;
      for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        o.wsc[2 * ib] = o.wsc[2 * ib + 1] = (int8_t)(ls - 32);
        for (int j = 0; j < 16; ++j) {
          o.q[32 * ib + j] = kvalues_iq4nl[qs[16 * ib + j] & 0xF];
          o.q[32 * ib + 16 + j] = kvalues_iq4nl[qs[16 * ib + j] >> 4];
        }
      }
      return;
    }
    default: throw std::runtime_error("decode_sb: unsupported type");
  }
}

inline float dequant_value(uint32_t type, const DecodedSB& o, int e) {
  switch (type) {
    case GGML_Q4_K: case GGML_Q5_K: return o.d * o.sc[e / 32] * o.q[e] - o.dmin * o.mn[e / 32];
    case GGML_Q6_K: return o.d * o.sc6[e / 16] * (o.q[e] - 32);
    case GGML_Q8_0: case GGML_Q4_0: return o.d8[e / 32] * o.q[e];
    default: return o.wd * o.wsc[e / 16] * (o.q[e] - o.qoff) - (o.has_min ? o.wdmin * o.wmn[e / 16] : 0.0f);
  }
}

// Bytes of one 256-element superblock in GGUF.
inline int gguf_sb_bytes(uint32_t type) {
  int blk, bytes;
  ggml_block_info(type, blk, bytes);
  return bytes * (256 / blk);
}

// Per-chunk uints (DPJ) and header uints (HDR) per lane in the GPU layout.
inline void gpu_layout_dims(uint32_t type, int& dpj, int& hdr) {
  switch (type) {
    case GGML_Q4_K: dpj = 8; hdr = 6; return;
    case GGML_Q5_K: dpj = 10; hdr = 6; return;
    case GGML_Q6_K: dpj = 12; hdr = 5; return;
    case GGML_Q8_0: dpj = 16; hdr = 8; return;
    default: throw std::runtime_error(std::string("no GPU kernel for type ") + ggml_type_name(type));
  }
}

struct RepackedWeight {
  uint32_t type = 0;
  int N = 0, K = 0;  // output rows, input dim
  std::vector<uint32_t> wq, wh;
};

// Repacks rows [n0, n0+N) of a GGUF matrix (row length K) into the GPU layout.
inline void repack_rows(uint32_t type, const uint8_t* data, int K, int n0, int N, RepackedWeight& out) {
  const int SG = 16, nsb = K / 256, G = N / SG;
  if (K % 256 || N % SG) throw std::runtime_error("repack: K must be a multiple of 256 and N of 16");
  int dpj, hdr;
  gpu_layout_dims(type, dpj, hdr);
  const size_t row_bytes = (size_t)nsb * gguf_sb_bytes(type);
  out.type = type;
  out.N = N;
  out.K = K;
  out.wq.assign((size_t)G * nsb * 4 * dpj * SG, 0);
  out.wh.assign((size_t)G * nsb * hdr * SG, 0);
#pragma omp parallel for schedule(dynamic, 4)
  for (int g = 0; g < G; ++g) {
    DecodedSB o;
    for (int l = 0; l < SG; ++l) {
      const uint8_t* row = data + (size_t)(n0 + g * SG + l) * row_bytes;
      for (int s = 0; s < nsb; ++s) {
        decode_sb(type, row + (size_t)s * gguf_sb_bytes(type), o);
        uint32_t* H = &out.wh[((size_t)g * nsb + s) * hdr * SG];
        auto put_f = [&](int f, float v) { memcpy(&H[f * SG + l], &v, 4); };
        auto put_b4 = [&](int f, const uint8_t* b) {
          H[f * SG + l] = b[0] | (b[1] << 8) | (b[2] << 16) | ((uint32_t)b[3] << 24);
        };
        if (type == GGML_Q4_K || type == GGML_Q5_K) {
          put_f(0, o.d);
          put_f(1, o.dmin);
          put_b4(2, o.sc);
          put_b4(3, o.sc + 4);
          put_b4(4, o.mn);
          put_b4(5, o.mn + 4);
        } else if (type == GGML_Q6_K) {
          put_f(0, o.d);
          for (int f = 0; f < 4; ++f) put_b4(1 + f, (const uint8_t*)o.sc6 + 4 * f);
        } else {
          for (int b = 0; b < 8; ++b) put_f(b, o.d8[b]);
        }
        for (int j = 0; j < 4; ++j) {
          uint32_t* D = &out.wq[(((size_t)g * nsb + s) * 4 + j) * dpj * SG];
          const int16_t* q = o.q + 64 * j;
          auto put = [&](int idx, uint32_t v) { D[idx * SG + l] = v; };
          if (type == GGML_Q8_0) {
            for (int w = 0; w < 16; ++w)
              put(w, (uint8_t)q[4 * w] | ((uint8_t)q[4 * w + 1] << 8) | ((uint8_t)q[4 * w + 2] << 16) |
                         ((uint32_t)(uint8_t)q[4 * w + 3] << 24));
            continue;
          }
          // Low 4 bits: nibble words (lo half = elements 0..31 of the chunk, hi half = 32..63).
          for (int w = 0; w < 8; ++w) {
            uint32_t v = 0;
            for (int b = 0; b < 4; ++b)
              v |= (uint32_t)((q[4 * w + b] & 0xF) | ((q[32 + 4 * w + b] & 0xF) << 4)) << (8 * b);
            put(w, v);
          }
          if (type == GGML_Q5_K) {
            // bit 4: word w, byte b -> bit (8*b + w) of hb[half]
            for (int half = 0; half < 2; ++half) {
              uint32_t v = 0;
              for (int w = 0; w < 8; ++w)
                for (int b = 0; b < 4; ++b) v |= (uint32_t)((q[32 * half + 4 * w + b] >> 4) & 1) << (8 * b + w);
              put(8 + half, v);
            }
          } else if (type == GGML_Q6_K) {
            // bits 4..5: word w, byte b -> bits (8*b + 2*(w&3)) of hb[2*half + (w>>2)]
            for (int half = 0; half < 2; ++half)
              for (int t = 0; t < 2; ++t) {
                uint32_t v = 0;
                for (int w = 4 * t; w < 4 * t + 4; ++w)
                  for (int b = 0; b < 4; ++b) v |= (uint32_t)((q[32 * half + 4 * w + b] >> 4) & 3) << (8 * b + 2 * (w & 3));
                put(8 + 2 * half + t, v);
              }
          }
        }
      }
    }
  }
}

// Quantizes f32 rows [N][K] to GGUF Q8_0 blocks (so small f32 matrices can use the int8 GEMM path).
inline std::vector<uint8_t> quantize_rows_q8_0(const float* src, int N, int K) {
  std::vector<uint8_t> out((size_t)N * (K / 32) * 34);
  for (size_t b = 0; b < (size_t)N * (K / 32); ++b) {
    const float* v = src + b * 32;
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(v[i]));
    const float d = amax / 127.0f, inv = d > 0 ? 1.0f / d : 0.0f;
    uint8_t* blk = out.data() + b * 34;
    const uint16_t dh = f32_to_fp16(d);
    memcpy(blk, &dh, 2);
    for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)(int8_t)std::lrintf(v[i] * inv);
  }
  return out;
}

// Quantizes f32 rows [N][K] to GGUF Q4_0 blocks as llama.cpp's quantize_row_q4_0_ref: d = (the value of largest
// magnitude) / -8, q = min(15, (int)(v / d + 8.5)), element i in the low nibble of byte i, element i + 16 in the high.
inline std::vector<uint8_t> quantize_rows_q4_0(const float* src, int N, int K) {
  std::vector<uint8_t> out((size_t)N * (K / 32) * 18);
  for (size_t b = 0; b < (size_t)N * (K / 32); ++b) {
    const float* v = src + b * 32;
    float amax = 0, mx = 0;
    for (int i = 0; i < 32; ++i)
      if (std::fabs(v[i]) > amax) amax = std::fabs(v[i]), mx = v[i];
    const float d = mx / -8.0f, id = d != 0 ? 1.0f / d : 0.0f;
    uint8_t* blk = out.data() + b * 18;
    const uint16_t dh = f32_to_fp16(d);
    memcpy(blk, &dh, 2);
    for (int i = 0; i < 16; ++i) {
      const int q0 = std::min(15, (int)(v[i] * id + 8.5f)), q1 = std::min(15, (int)(v[i + 16] * id + 8.5f));
      blk[2 + i] = (uint8_t)(q0 | (q1 << 4));
    }
  }
  return out;
}

// Q3_K superblock (hmask[32], qs[64], 12 bytes of 6-bit scales, f16 d) to 256 floats, as llama.cpp.
inline void dequant_q3_k(const uint8_t* p, float* y) {
  const uint8_t *hm = p, *q = p + 32;
  uint16_t dh;
  memcpy(&dh, p + 108, 2);
  const float d_all = fp16_to_f32(dh);
  uint32_t aux[4];
  memcpy(aux, p + 96, 12);
  const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  const int8_t* scales = (const int8_t*)aux;
  int is = 0;
  uint8_t m = 1;
  for (int n = 0; n < 256; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      float dl = d_all * (scales[is++] - 32);
      for (int l = 0; l < 16; ++l) *y++ = dl * ((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
      dl = d_all * (scales[is++] - 32);
      for (int l = 0; l < 16; ++l) *y++ = dl * ((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
      shift += 2;
      m <<= 1;
    }
    q += 32;
  }
}

// Dequantizes one full row (K elements) of a GGUF matrix to float (CPU reference / embeddings).
inline void dequant_row(uint32_t type, const uint8_t* row, int K, float* out) {
  if (type == GGML_Q3_K) {
    for (int s = 0; s < K / 256; ++s) dequant_q3_k(row + (size_t)s * 110, out + (size_t)s * 256);
    return;
  }
  if (type == GGML_F32) { memcpy(out, row, (size_t)K * 4); return; }
  if (type == GGML_F16) { for (int i = 0; i < K; ++i) { uint16_t h; memcpy(&h, row + 2 * i, 2); out[i] = fp16_to_f32(h); } return; }
  if (type == GGML_BF16) { for (int i = 0; i < K; ++i) { uint16_t h; memcpy(&h, row + 2 * i, 2); out[i] = bf16_to_f32(h); } return; }
  if (type == GGML_Q4_0) {  // blocks of 32: f16 scale, low nibbles = elements 0..15, high nibbles = 16..31
    for (int b = 0; b < K / 32; ++b) {
      const uint8_t* p = row + (size_t)b * 18;
      uint16_t h;
      memcpy(&h, p, 2);
      const float d = fp16_to_f32(h);
      for (int j = 0; j < 16; ++j) {
        out[b * 32 + j] = d * (float)((p[2 + j] & 0x0F) - 8);
        out[b * 32 + 16 + j] = d * (float)((p[2 + j] >> 4) - 8);
      }
    }
    return;
  }
  if (type == GGML_Q8_0 && K % 256) {  // blocks of 32 (f16 scale + 32 int8), for rows that are not whole superblocks
    for (int b = 0; b < K / 32; ++b) {
      const uint8_t* p = row + (size_t)b * 34;
      uint16_t h;
      memcpy(&h, p, 2);
      const float d = fp16_to_f32(h);
      for (int j = 0; j < 32; ++j) out[b * 32 + j] = d * (float)(int8_t)p[2 + j];
    }
    return;
  }
  if (type == GGML_IQ4_NL) {  // blocks of 32: f16 scale, low nibbles = elements 0..15, high = 16..31, via the IQ4 table
    for (int b = 0; b < K / 32; ++b) {
      const uint8_t* p = row + (size_t)b * 18;
      uint16_t h;
      memcpy(&h, p, 2);
      const float d = fp16_to_f32(h);
      for (int j = 0; j < 16; ++j) {
        out[b * 32 + j] = d * (float)kvalues_iq4nl[p[2 + j] & 0x0F];
        out[b * 32 + 16 + j] = d * (float)kvalues_iq4nl[p[2 + j] >> 4];
      }
    }
    return;
  }
  if (type == GGML_Q2_0) {  // blocks of 64: f16 scale, element j = bits 2*(j%4) of byte j/4, value (q - 1) * d
    for (int b = 0; b < K / 64; ++b) {
      const uint8_t* p = row + (size_t)b * 18;
      uint16_t h;
      memcpy(&h, p, 2);
      const float d = fp16_to_f32(h);
      for (int j = 0; j < 64; ++j) out[b * 64 + j] = d * (float)(((p[2 + j / 4] >> (2 * (j % 4))) & 3) - 1);
    }
    return;
  }
  DecodedSB o;
  for (int s = 0; s < K / 256; ++s) {
    decode_sb(type, row + (size_t)s * gguf_sb_bytes(type), o);
    for (int e = 0; e < 256; ++e) out[s * 256 + e] = dequant_value(type, o, e);
  }
}

// CPU version of the activation quantizer (must match kernels/ops.cl quant_store16): int8 with absmax scale
// per 256 elements (asc [K/256][M]) or per 32 (act32: asc [K/32][M]); absum [K/16][M] = sums per 16.
inline void quantize_act_cpu(const float* x, int M, int K, int8_t* aq, float* asc, uint16_t* absum, bool act32 = false) {
  const int B = act32 ? 32 : 256;
  for (int m = 0; m < M; ++m)
    for (int b0 = 0; b0 < K; b0 += B) {
      const float* v = x + (size_t)m * K + b0;
      float amax = 0;
      for (int i = 0; i < B; ++i) amax = std::fmax(amax, std::fabs(v[i]));
      const float inv = amax > 0 ? 127.0f / amax : 0.0f;
      asc[(size_t)(b0 / B) * M + m] = amax / 127.0f;
      for (int i = 0; i < B; ++i) aq[(size_t)m * K + b0 + i] = (int8_t)std::lrintf(v[i] * inv);
    }
  for (int m = 0; m < M; ++m)
    for (int b = 0; b < K / 16; ++b) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) sum += aq[(size_t)m * K + b * 16 + i];
      absum[(size_t)b * M + m] = (uint16_t)(int16_t)sum;
    }
}

// Weight types the GEMM reads natively (repacked once at load). Everything else is decoded just in time
// on the GPU (kernels/decode.cl) into the W8 layout.
inline bool is_native_type(uint32_t t) { return t == GGML_Q4_K || t == GGML_Q5_K || t == GGML_Q6_K || t == GGML_Q8_0; }
// Small-block types (IQ4_NL: 32 elements / 18 bytes, Q2_0: 64 / 18) whose consecutive blocks are read as 256-element
// "superblocks" by the GPU (144 / 72 bytes). A row whose K is not a multiple of 256 (Qwen3.8 Flash Next down
// projections: K = 640) is padded to the next multiple with zero blocks (scale 0, dequantizing to zeros).
inline bool is_small_block_type(uint32_t t) { return t == GGML_IQ4_NL || t == GGML_Q2_0; }
inline int padded_k(int K) { return (K + 255) / 256 * 256; }
// N rows of K elements of a small-block type -> N rows of padded_k(K) elements (dst holds N * the padded row bytes)
inline void pad_small_block_rows(uint32_t type, const uint8_t* src, int N, int K, uint8_t* dst) {
  int blk = 0, bytes = 0;
  ggml_block_info(type, blk, bytes);
  const size_t rb = (size_t)K / blk * bytes, prb = (size_t)padded_k(K) / blk * bytes;
  for (int n = 0; n < N; ++n) {
    memcpy(dst + n * prb, src + n * rb, rb);
    memset(dst + n * prb + rb, 0, prb - rb);  // (scale 0: an fp16 zero, so the padded elements dequantize to 0)
  }
}

inline bool is_w8_type(uint32_t t) {
  switch (t) {
    case GGML_Q2_K: case GGML_Q3_K: case GGML_Q4_K: case GGML_Q5_K: case GGML_Q6_K: case GGML_IQ2_XXS: case GGML_IQ2_XS:
    case GGML_IQ2_S: case GGML_IQ3_XXS: case GGML_IQ3_S: case GGML_IQ1_M: case GGML_IQ4_XS: case GGML_Q8_0: case GGML_Q4_0:
    case GGML_IQ4_NL: case GGML_Q2_0: return true;
    default: return false;
  }
}
// W8x8 GEMM variant (kernels/gemm8.cl VARIANT 0..4) for a decoded GGUF type.
// Types with mins (Q2_K/Q4_K/Q5_K) are requantized to signed int8 by decode_w8_sb (no min term in the GEMM: 3x the
// speed of the scales + mins variants 2/3, about 0.5 % weight error). ARC_W8_MINS=1 selects the scales + mins form
// (experimental: currently less accurate end to end, see docs/review_notes.md).
inline bool w8_requant(uint32_t t) {
  static const bool mins = getenv("ARC_W8_MINS") != nullptr;
  return !mins && (t == GGML_Q2_K || t == GGML_Q4_K || t == GGML_Q5_K);
}
inline int w8_variant(uint32_t t) {
  if (t == GGML_Q8_0 || t == GGML_Q4_0 || t == GGML_IQ4_NL || t == GGML_Q2_0)
    return 4;  // per-32 f32 scales (the GEMM runs them as its native Q8_0 form)
  const bool sc32 = t == GGML_IQ4_XS || t == GGML_IQ3_S || t == GGML_IQ3_XXS || t == GGML_IQ2_XXS ||
                    t == GGML_Q4_K || t == GGML_Q5_K;
  const bool mins = !w8_requant(t) && (t == GGML_Q2_K || t == GGML_Q4_K || t == GGML_Q5_K);
  return (sc32 ? 1 : 0) + (mins ? 2 : 0);
}
// Bytes of the decoded W8x8 form of an N x K matrix (q + header).
inline size_t w8_bytes_q(int N, int K) { return (size_t)N * K; }
inline size_t w8_bytes_h(int N, int K) { return (size_t)N / 8 * (K / 256) * 10 * 8 * 4; }
// Codebooks for kernels/decode.cl, laid out at the TAB_* uint offsets defined there.
inline std::vector<uint32_t> build_iq_tables() {
  std::vector<uint32_t> t(8484, 0);
  auto put = [&](size_t off, const void* src, size_t bytes) { memcpy(t.data() + off, src, bytes); };
  put(0, iq2xxs_grid, sizeof(iq2xxs_grid));
  put(512, iq2xs_grid, sizeof(iq2xs_grid));
  put(1536, iq2s_grid, sizeof(iq2s_grid));
  put(3584, iq3xxs_grid, sizeof(iq3xxs_grid));
  put(3840, iq3s_grid, sizeof(iq3s_grid));
  put(4352, iq1s_grid, sizeof(iq1s_grid));
  put(8448, ksigns_iq2xs, sizeof(ksigns_iq2xs));
  put(8480, kvalues_iq4nl, sizeof(kvalues_iq4nl));
  return t;
}
// GPU raw form of a GGUF matrix: each 256-element superblock zero-padded to a multiple of 8 bytes, so a pair of
// superblocks is 16-byte aligned for sub-group block reads (kernels/decode.cl uses the padded size BBP).
// Q4_0, Q8_0 and the IQ2 / IQ3 codebook types are also regrouped (Q8_0: the 8 scales first, then 32-byte blocks;
// its GEMV read 34-byte blocks at 37 GB/s). IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S: the leading
// f16 scale moves behind the rest, so every block's fields (8-byte quant groups, 4-byte sign / scale words) are
// aligned for vector loads (kernels/decode.cl: their GEMVs went from 27-61 to ~64 GB/s).
// Q4_0: its 8 blocks of 18 bytes (f16 d + 16 bytes of nibbles) become the 8 scales followed by the
// 8 x 16 nibble bytes, so every block's quants are 16-byte aligned for vector loads (same 144 bytes).
inline int raw_padded_sb_bytes(uint32_t type) { return (gguf_sb_bytes(type) + 7) / 8 * 8; }
inline bool gpu_d_last(uint32_t type) {
  return type == GGML_IQ2_XXS || type == GGML_IQ2_XS || type == GGML_IQ2_S || type == GGML_IQ3_XXS || type == GGML_IQ3_S;
}
inline bool gpu_regrouped(uint32_t type) {
  return type == GGML_Q4_0 || type == GGML_Q8_0 || type == GGML_IQ4_NL || type == GGML_Q2_0 || gpu_d_last(type);
}
// One superblock in its GPU raw form (dst: raw_padded_sb_bytes(type) bytes).
inline void gpu_raw_sb(uint32_t type, const uint8_t* src, uint8_t* dst) {
  const size_t bb = gguf_sb_bytes(type), bp = raw_padded_sb_bytes(type);
  if (type == GGML_Q4_0 || type == GGML_Q8_0 || type == GGML_IQ4_NL || type == GGML_Q2_0) {
    // nb blocks (f16 d + quants) -> the nb scales, then the quants (Q4_0 / Q8_0 / IQ4_NL: 8 blocks of 32;
    // Q2_0: 4 blocks of 64, so its scales take 8 bytes)
    const int qb = type == GGML_Q8_0 ? 32 : 16, nb = type == GGML_Q2_0 ? 4 : 8;
    for (int b = 0; b < nb; ++b) {
      memcpy(dst + 2 * b, src + (2 + qb) * b, 2);
      memcpy(dst + 2 * nb + qb * b, src + (2 + qb) * b + 2, qb);
    }
    return;
  }
  if (gpu_d_last(type)) {  // d | rest  ->  rest | d | pad
    memcpy(dst, src + 2, bb - 2);
    memcpy(dst + bb - 2, src, 2);
    if (bp > bb) memset(dst + bb, 0, bp - bb);
    return;
  }
  memcpy(dst, src, bb);
  if (bp > bb) memset(dst + bb, 0, bp - bb);
}
inline std::vector<uint8_t> pad_raw_rows(uint32_t type, const uint8_t* data, size_t nbytes) {
  const size_t bb = gguf_sb_bytes(type), bp = raw_padded_sb_bytes(type), nsb = nbytes / bb;
  std::vector<uint8_t> out(nsb * bp + 256, 0);  // tail slack: block reads may run past the last superblock
  if (bb == bp && !gpu_regrouped(type)) {
    memcpy(out.data(), data, nbytes);
  } else {
    for (size_t i = 0; i < nsb; ++i) gpu_raw_sb(type, data + i * bb, out.data() + i * bp);
  }
  return out;
}