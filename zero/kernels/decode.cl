// Decoding of raw GGUF blocks, shared by:
//  - decode_w8: just-in-time conversion of a matrix into the "W8" GEMM layout (kernels/gemm.cl QTYPE 100..103)
//  - gemv:      single-token matrix-vector product directly on the raw rows (token generation, lm_head)
// Generic form of every type: w[e] = d * sc[e/16] * q[e] - dmin * mn[e/16] with int8 q, int8 sc, uint8 mn and
// per-256-superblock d, dmin (Q8_0 / Q4_0: d is the per-32 scale, sc = 1; a "superblock" is 8 GGUF blocks). Values are produced directly as packed
// int8x4 words with SWAR bit operations. Compiled per type with -DQT=<ggml type>.
// Codebooks come in one buffer (TAB_* uint offsets, filled by the host from src/iq_tables.h).

#define TAB_IQ2XXS 0
#define TAB_IQ2XS 512
#define TAB_IQ2S 1536
#define TAB_IQ3XXS 3584
#define TAB_IQ3S 3840
#define TAB_IQ1S 4352
#define TAB_KSIGNS 8448
#define TAB_KVAL 8480

#define Q4_0 2
#define Q8_0 8
// per-32 float scales (decoded into the W8 GEMM's VARIANT 4 layout: one f32 per 32-block in the header)
#define PER32 (QT == Q8_0 || QT == Q4_0)
// header uints per superblock and lane: W8 (d, dmin, 16 scales, 16 mins) or, for PER32, the 8 f32 block scales in
// the layout of the GEMM's native Q8_0 form (kernels/gemm.cl QTYPE 8), which reads the decoded weights directly
#if PER32
#define HSTR 8
#else
#define HSTR 10
#endif
#define Q2_K 10
#define Q3_K 11
#define Q4_K 12
#define Q5_K 13
#define Q6_K 14
#define IQ2_XXS 16
#define IQ2_XS 17
#define IQ3_XXS 18
#define IQ3_S 21
#define IQ2_S 22
#define IQ4_XS 23
#define IQ1_M 29

#if QT == Q2_K
#define BB 84
#elif QT == Q3_K
#define BB 110
#elif QT == Q4_K
#define BB 144
#elif QT == Q5_K
#define BB 176
#elif QT == Q6_K
#define BB 210
#elif QT == IQ2_XXS
#define BB 66
#elif QT == IQ2_XS
#define BB 74
#elif QT == IQ2_S
#define BB 82
#elif QT == IQ3_XXS
#define BB 98
#elif QT == IQ3_S
#define BB 110
#elif QT == IQ1_M
#define BB 56
#elif QT == IQ4_XS
#define BB 136
#elif QT == Q8_0
#define BB 272
#elif QT == Q4_0
#define BB 144
#else
#error "unsupported QT"
#endif
// superblock size in the padded GPU raw form (multiple of 8 bytes)
#define BBP ((BB + 7) / 8 * 8)

inline uint rd16(const uchar* p) { return (uint)p[0] | ((uint)p[1] << 8); }
inline uint rd32(const uchar* p) { return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24); }
inline float f16(uint h) { return vload_half(0, (const __private half*)&h); }
// codebook of this type (GRID_OFF/GRID_N uints) and its auxiliary table (ksigns or kvalues: AUX_OFF/AUX_N uints)
#if QT == IQ2_XXS
#define GRID_OFF TAB_IQ2XXS
#define GRID_N 512
#elif QT == IQ2_XS
#define GRID_OFF TAB_IQ2XS
#define GRID_N 1024
#elif QT == IQ2_S
#define GRID_OFF TAB_IQ2S
#define GRID_N 2048
#elif QT == IQ3_XXS
#define GRID_OFF TAB_IQ3XXS
#define GRID_N 256
#elif QT == IQ3_S
#define GRID_OFF TAB_IQ3S
#define GRID_N 512
#elif QT == IQ1_M
#define GRID_OFF TAB_IQ1S
#define GRID_N 4096
#endif
#if QT == IQ2_XXS || QT == IQ2_XS || QT == IQ3_XXS
#define AUX_OFF TAB_KSIGNS
#define AUX_N 32
#elif QT == IQ4_XS
#define AUX_OFF TAB_KVAL
#define AUX_N 4
#endif
inline uint grid4w(const uint* grid, uint idx) { return grid[idx]; }
inline uint2 grid8w(const uint* grid, uint idx) { return (uint2)(grid[2 * idx], grid[2 * idx + 1]); }
inline uint ksigns(const uint* aux, uint i) { return ((const uchar*)aux)[i]; }

// Applies 4 sign bits (bit k negates byte k) to 4 positive int8 values. No byte of g is 0, so the
// two's-complement increment never carries into the next byte.
inline uint apply_signs4(uint g, uint s4) {
  const uint m = ((s4 * 0x00204081u) & 0x01010101u) * 0xFFu;
  return (g ^ m) + (m & 0x01010101u);
}
// Per-byte (q - 8) for 4-bit unsigned values, without cross-byte borrows.
inline uint sub8_bytes(uint q) { return ((q | 0x80808080u) - 0x08080808u) ^ 0x80808080u; }
// Per-byte (q - 32) for 6-bit unsigned values, without cross-byte borrows.
inline uint sub32_bytes(uint q) { return ((q | 0x80808080u) - 0x20202020u) ^ 0x80808080u; }

inline void get_scale_min_k4(int j, const uchar* q, int* d, int* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// Decodes 32-element block ib of the superblock at p into 8 packed int8x4 words (elements 4k..4k+3 in w[k]).
inline void decode32(const uchar* p, int ib, const uint* grid, const uint* aux, uint* w, int* psc0, int* psc1,
                     int* pmn0, int* pmn1, float* pd, float* pdmin) {
  int sc0, sc1, mn0 = 0, mn1 = 0;
  float d, dmin = 0.0f;
#if QT == Q2_K
  d = f16(rd16(p + 80));
  dmin = f16(rd16(p + 82));
  sc0 = p[2 * ib] & 0xF; mn0 = p[2 * ib] >> 4;
  sc1 = p[2 * ib + 1] & 0xF; mn1 = p[2 * ib + 1] >> 4;
  {
    const uchar* qs = p + 16 + 32 * (ib / 4);
    const int sh = 2 * (ib % 4);
#pragma unroll
    for (int k = 0; k < 8; ++k) w[k] = (rd32(qs + 4 * k) >> sh) & 0x03030303u;
  }
#elif QT == Q3_K
  // hmask[32] | qs[64] | 12 bytes of 6-bit scales | f16 d. Block ib: 2-bit values (qs[32 (ib / 4) + e] >> 2 (ib % 4))
  // plus 4 * (bit ib of hmask[e]), minus 4; scale i (16 elements): low 4 bits from byte i % 8 (nibble i / 8), top 2
  // bits from byte 8 + i % 4 (bits 2 (i / 4)), minus 32
  d = f16(rd16(p + 108));
  {
    const uchar* sc = p + 96;
    const int i0 = 2 * ib, i1 = 2 * ib + 1;
    sc0 = (int)(((sc[i0 & 7] >> (4 * (i0 >> 3))) & 0xF) | (((sc[8 + (i0 & 3)] >> (2 * (i0 >> 2))) & 3) << 4)) - 32;
    sc1 = (int)(((sc[i1 & 7] >> (4 * (i1 >> 3))) & 0xF) | (((sc[8 + (i1 & 3)] >> (2 * (i1 >> 2))) & 3) << 4)) - 32;
    const uchar* qs = p + 32 + 32 * (ib / 4);
    const int sh = 2 * (ib % 4);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const uint v = ((rd32(qs + 4 * k) >> sh) & 0x03030303u) | (((rd32(p + 4 * k) >> ib) & 0x01010101u) << 2);
      w[k] = ((v | 0x80808080u) - 0x04040404u) ^ 0x80808080u;  // per byte v - 4
    }
  }
#elif QT == Q4_K || QT == Q5_K
  d = f16(rd16(p));
  dmin = f16(rd16(p + 2));
  get_scale_min_k4(ib, p + 4, &sc0, &mn0);
  sc1 = sc0; mn1 = mn0;
  {
    const uchar* qs = p + (QT == Q5_K ? 48 : 16) + 32 * (ib / 2);
    const int sh = 4 * (ib % 2);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      w[k] = (rd32(qs + 4 * k) >> sh) & 0x0F0F0F0Fu;
#if QT == Q5_K
      w[k] |= ((rd32(p + 16 + 4 * k) >> ib) & 0x01010101u) << 4;
#endif
    }
  }
#elif QT == Q6_K
  d = f16(rd16(p + 208));
  sc0 = (char)p[192 + 2 * ib];
  sc1 = (char)p[192 + 2 * ib + 1];
  {
    const int nn = ib / 4, quarter = ib % 4;
    const uchar* L = p + 64 * nn + ((quarter & 1) ? 32 : 0);
    const uchar* H = p + 128 + 32 * nn;
    const int lsh = quarter >= 2 ? 4 : 0, hsh = 2 * quarter;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const uint lo = (rd32(L + 4 * k) >> lsh) & 0x0F0F0F0Fu;
      const uint hi = ((rd32(H + 4 * k) >> hsh) & 0x03030303u) << 4;
      w[k] = sub32_bytes(lo | hi);
    }
  }
#elif QT == Q8_0
  {
    // GPU form (src/quants.h gpu_raw_sb): the 8 f16 scales, then 8 blocks of 32 int8
    d = f16(rd16(p + 2 * ib));
    sc0 = sc1 = 1;
#pragma unroll
    for (int k = 0; k < 8; ++k) w[k] = rd32(p + 16 + 32 * ib + 4 * k);
  }
#elif QT == Q4_0
  {
    // GPU form (src/quants.h gpu_raw_sb): 8 f16 scales, then 16 bytes per block with elements 0..15 in the low
    // nibbles and 16..31 in the high ones; w = d * (q - 8)
    const uchar* blk = p + 16 + 16 * ib;
    d = f16(rd16(p + 2 * ib));
    sc0 = sc1 = 1;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint v = rd32(blk + 4 * k);
      w[k] = sub8_bytes(v & 0x0F0F0F0Fu);
      w[k + 4] = sub8_bytes((v >> 4) & 0x0F0F0F0Fu);
    }
  }
#elif QT == IQ2_XXS
  // (IQ2 / IQ3 types in their GPU form, src/quants.h gpu_raw_sb: the f16 d last, every other field 2 bytes earlier)
  d = f16(rd16(p + BB - 2)) * 0.125f;
  {
    const uint a0 = rd32(p + 8 * ib), a1 = rd32(p + 8 * ib + 4);
    sc0 = sc1 = 1 + 2 * (a1 >> 28);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint2 g = grid8w(grid, (a0 >> (8 * l)) & 0xFF);
      const uint sg = ksigns(aux, (a1 >> (7 * l)) & 127);
      w[2 * l] = apply_signs4(g.s0, sg & 0xF);
      w[2 * l + 1] = apply_signs4(g.s1, sg >> 4);
    }
  }
#elif QT == IQ2_XS
  d = f16(rd16(p + BB - 2)) * 0.125f;
  {
    const uint scb = p[64 + ib];
    sc0 = 1 + 2 * (scb & 0xF);
    sc1 = 1 + 2 * (scb >> 4);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint q16 = rd16(p + 2 * (4 * ib + l));
      const uint2 g = grid8w(grid, q16 & 511);
      const uint sg = ksigns(aux, q16 >> 9);
      w[2 * l] = apply_signs4(g.s0, sg & 0xF);
      w[2 * l + 1] = apply_signs4(g.s1, sg >> 4);
    }
  }
#elif QT == IQ2_S
  d = f16(rd16(p + BB - 2)) * 0.125f;
  {
    const uint scb = p[72 + ib], h = p[64 + ib];
    sc0 = 1 + 2 * (scb & 0xF);
    sc1 = 1 + 2 * (scb >> 4);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint2 g = grid8w(grid, p[4 * ib + l] | ((h << (8 - 2 * l)) & 0x300));
      const uint sg = p[32 + 4 * ib + l];
      w[2 * l] = apply_signs4(g.s0, sg & 0xF);
      w[2 * l + 1] = apply_signs4(g.s1, sg >> 4);
    }
  }
#elif QT == IQ3_XXS
  d = f16(rd16(p + BB - 2)) * 0.25f;
  {
    const uint a = rd32(p + 64 + 4 * ib);
    sc0 = sc1 = 1 + 2 * (a >> 28);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint sg = ksigns(aux, (a >> (7 * l)) & 127);
      w[2 * l] = apply_signs4(grid4w(grid, p[8 * ib + 2 * l]), sg & 0xF);
      w[2 * l + 1] = apply_signs4(grid4w(grid, p[8 * ib + 2 * l + 1]), sg >> 4);
    }
  }
#elif QT == IQ3_S
  d = f16(rd16(p + BB - 2));
  {
    sc0 = sc1 = 1 + 2 * ((p[104 + ib / 2] >> (4 * (ib % 2))) & 0xF);
    const uint h = p[64 + ib];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint g1 = p[8 * ib + 2 * l] | ((h << (8 - 2 * l)) & 256);
      const uint g2 = p[8 * ib + 2 * l + 1] | ((h << (7 - 2 * l)) & 256);
      const uint sg = p[72 + 4 * ib + l];
      w[2 * l] = apply_signs4(grid4w(grid, g1), sg & 0xF);
      w[2 * l + 1] = apply_signs4(grid4w(grid, g2), sg >> 4);
    }
  }
#elif QT == IQ1_M
  {
    const uint s0 = rd16(p + 48), s1 = rd16(p + 50), s2 = rd16(p + 52), s3 = rd16(p + 54);
    const uint su = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
    d = f16(su) * 0.125f;
    const uint sc = ib / 2 == 0 ? s0 : ib / 2 == 1 ? s1 : ib / 2 == 2 ? s2 : s3;
    sc0 = 2 * ((sc >> (6 * (ib % 2) + 0)) & 7) + 1;
    sc1 = 2 * ((sc >> (6 * (ib % 2) + 3)) & 7) + 1;
    const uint h0 = p[32 + 2 * ib], h1 = p[32 + 2 * ib + 1];
    const uint idx[4] = {p[4 * ib] | ((h0 << 8) & 0x700), p[4 * ib + 1] | ((h0 << 4) & 0x700),
                         p[4 * ib + 2] | ((h1 << 8) & 0x700), p[4 * ib + 3] | ((h1 << 4) & 0x700)};
    const int dl[4] = {(h0 & 0x08) ? -1 : 1, (h0 & 0x80) ? -1 : 1, (h1 & 0x08) ? -1 : 1, (h1 & 0x80) ? -1 : 1};
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint2 g = grid8w(grid, idx[l]);
#pragma unroll
      for (int hw = 0; hw < 2; ++hw) {
        const uint gw = hw ? g.s1 : g.s0;
        uint o = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) o |= (uint)(uchar)(char)(8 * (char)((gw >> (8 * b)) & 0xFF) + dl[l]) << (8 * b);
        w[2 * l + hw] = o;
      }
    }
  }
#elif QT == IQ4_XS
  d = f16(rd16(p));
  {
    const uint sh = rd16(p + 2);
    const int ls = ((p[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    sc0 = sc1 = ls - 32;
    // a 16-byte table in local memory: measured faster than a 256-entry byte-pair table (SLM bank conflicts) and
    // than sub-group shuffles with a variable lane index
    const uchar* kv = (const uchar*)aux;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint b = rd32(p + 8 + 16 * ib + 4 * k);
      w[k] = kv[b & 0xF] | ((uint)kv[(b >> 8) & 0xF] << 8) | ((uint)kv[(b >> 16) & 0xF] << 16) | ((uint)kv[(b >> 24) & 0xF] << 24);
      w[4 + k] = kv[(b >> 4) & 0xF] | ((uint)kv[(b >> 12) & 0xF] << 8) | ((uint)kv[(b >> 20) & 0xF] << 16) |
                 ((uint)kv[(b >> 28) & 0xF] << 24);
    }
  }
#endif
  *psc0 = sc0; *psc1 = sc1; *pmn0 = mn0; *pmn1 = mn1; *pd = d; *pdmin = dmin;
}

// raw GGUF rows -> W8 GEMM layout: [g][s][j][w=0..15][lane] uint words and [g][s][10][lane] uint headers
// (d, dmin, sc[16] bytes, mn[16] bytes; Q8_0 stores the f32 scale of each 32-block in header word ib).
// One work-item per (row n, 32-element block); grid (N, K/32).
#ifndef GL
#define GL 16
#endif
// arcinferzero expert streaming (-DXCACHE): an expert's rows are not at e * rows in one buffer but in its record,
// wherever the expert cache put it; xaddr[e] holds the record's address, the matrix at byte offset `off` in it.
#ifdef XCACHE
#define XROW(base, off, e, NR, n, sb) ((__global const uchar*)xaddr[e] + (off) + ((size_t)(n) * nsb + (sb)) * BBP)
#define XBASE(base, off, e, NR) ((__global const uchar*)xaddr[e] + (off))
#define XARGS_W8 , __global const ulong* xaddr, int xe0, int xn, int xoff
#define XARGS_GU , __global const ulong* xaddr, int xog, int xou
#define XARGS_D , __global const ulong* xaddr, int xod
#else
#define XROW(base, off, e, NR, n, sb) ((base) + (((size_t)(e) * (NR) + (n)) * nsb + (sb)) * BBP)
#define XBASE(base, off, e, NR) ((base) + (size_t)(e) * (NR) * nsb * BBP)
#define XARGS_W8
#define XARGS_GU
#define XARGS_D
#endif
// row n, superblock sb of an expert matrix starting at b (its address read once per work-item: XBASE)
#define XR(b, n, sb) ((b) + ((size_t)(n) * nsb + (sb)) * BBP)
__kernel void decode_w8(__global const uchar* raw, int N, int K, __global const uint* tabs, __global uint* wq,
                        __global uint* wh) {
  const int n = get_global_id(0), b32 = get_global_id(1);
  const int nsb = K / 256, s = b32 / 8, ib = b32 % 8, g = n / GL, lane = n % GL;
  uint w[8];
  int sc0, sc1, mn0, mn1;
  float d, dmin;
#ifdef GRID_OFF
  const uint* grid = tabs + GRID_OFF;
#else
  const uint* grid = tabs;
#endif
#ifdef AUX_OFF
  const uint* aux = tabs + AUX_OFF;
#else
  const uint* aux = tabs;
#endif
  decode32(raw + ((size_t)n * nsb + s) * BBP, ib, grid, aux, w, &sc0, &sc1, &mn0, &mn1, &d, &dmin);
  __global uint* dst = wq + (((size_t)g * nsb + s) * 4 + ib / 2) * 16 * GL + (ib % 2) * 8 * GL + lane;
#pragma unroll
  for (int k = 0; k < 8; ++k) dst[k * GL] = w[k];
  __global uint* hd = wh + ((size_t)g * nsb + s) * HSTR * GL;
#if PER32
  hd[ib * GL + lane] = as_uint(d);
#else
  if (ib == 0) {
    hd[0 * GL + lane] = as_uint(d);
    hd[1 * GL + lane] = as_uint(dmin);
  }
  __global uchar* hb = (__global uchar*)hd;
  const int b0 = 2 * ib, b1 = 2 * ib + 1;
  hb[((2 + b0 / 4) * GL + lane) * 4 + b0 % 4] = (uchar)(char)sc0;
  hb[((2 + b1 / 4) * GL + lane) * 4 + b1 % 4] = (uchar)(char)sc1;
  hb[((6 + b0 / 4) * GL + lane) * 4 + b0 % 4] = (uchar)mn0;
  hb[((6 + b1 / 4) * GL + lane) * 4 + b1 % 4] = (uchar)mn1;
#endif
}

// Same conversion, one work-item per (row, superblock) with the codebooks staged in local memory once per
// work-group. grid (N, K/256), local (LSZ, 1) with LSZ dividing N.
__kernel void decode_w8_sb(__global const uchar* raw, int N, int K, __global const uint* tabs, __global uint* wq,
                           __global uint* wh XARGS_W8) {
#ifdef GRID_OFF
  __local uint lgrid[GRID_N];
  for (int i = get_local_id(0); i < GRID_N; i += get_local_size(0)) lgrid[i] = tabs[GRID_OFF + i];
  const uint* grid = lgrid;
#else
  const uint* grid = tabs;
#endif
#ifdef AUX_OFF
  __local uint laux[AUX_N];
  for (int i = get_local_id(0); i < AUX_N; i += get_local_size(0)) laux[i] = tabs[AUX_OFF + i];
  const uint* aux = laux;
#else
  const uint* aux = tabs;
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n = get_global_id(0), s = get_global_id(1);
  const int nsb = K / 256, g = n / GL, lane = n % GL;
  if (n >= N) return;
  // the superblock into registers with 8-byte vector loads, decoded from there (byte-wise reads of global memory
  // made decoding several times slower than reading the weights)
  uint sbu[BBP / 4];
  {
#ifdef XCACHE
    __global const uint* src = (__global const uint*)XROW(raw, xoff, xe0 + n / xn, 0, n % xn, s);
#else
    __global const uint* src = (__global const uint*)(raw + ((size_t)n * nsb + s) * BBP);
#endif
#pragma unroll
    for (int i = 0; i < BBP / 8; ++i) {
      const uint2 v = vload2(i, src);
      sbu[2 * i] = v.x;
      sbu[2 * i + 1] = v.y;
    }
  }
  const uchar* p = (const uchar*)sbu;
  __global uint* hd = wh + ((size_t)g * nsb + s) * HSTR * GL;
#if QT == Q2_K || QT == Q4_K || QT == Q5_K
  // Types with per-block mins (w = d*sc*q - dmin*m, q unsigned) are requantized to signed int8 so they can use
  // the min-free GEMM: one float D per superblock, an int8 scale per 16 (equal halves for Q4_K/Q5_K) chosen from
  // the analytic block range, q' = round(w / (D*sc')). The added error is ~1/16 of the original quantization step.
#if QT == Q4_K
  const float qmax = 15.0f;
#elif QT == Q5_K
  const float qmax = 31.0f;
#else
  const float qmax = 3.0f;
#endif
  uint wv[8][8];
  float fs[16], fm[16], am[16];
  float D = 0.0f;
  for (int ib = 0; ib < 8; ++ib) {
    int sc0, sc1, mn0, mn1;
    float d, dmin;
    decode32(p, ib, grid, aux, wv[ib], &sc0, &sc1, &mn0, &mn1, &d, &dmin);
    fs[2 * ib] = d * sc0; fs[2 * ib + 1] = d * sc1;
    fm[2 * ib] = dmin * mn0; fm[2 * ib + 1] = dmin * mn1;
  }
  for (int h = 0; h < 16; ++h) {
    am[h] = fmax(fabs(fm[h]), fabs(fs[h] * qmax - fm[h]));
    D = fmax(D, am[h]);
  }
  D /= 127.0f * 127.0f;
  uint scw[4] = {0, 0, 0, 0};
  for (int ib = 0; ib < 8; ++ib) {
    float step[2];
    for (int hh = 0; hh < 2; ++hh) {
      const int h = 2 * ib + hh;
      const int sq = D > 0.0f ? clamp((int)ceil(am[h] / (127.0f * D) - 1e-4f), 1, 127) : 1;
      scw[h / 4] |= (uint)sq << (8 * (h % 4));
      step[hh] = D * sq;
    }
    __global uint* dst = wq + (((size_t)g * nsb + s) * 4 + ib / 2) * 16 * GL + (ib % 2) * 8 * GL + lane;
    for (int k = 0; k < 8; ++k) {
      const int h = 2 * ib + k / 4;
      const float inv = step[k / 4] > 0.0f ? 1.0f / step[k / 4] : 0.0f;
      uint o = 0;
      for (int b = 0; b < 4; ++b) {
        const float v = fs[h] * (float)((wv[ib][k] >> (8 * b)) & 0xFF) - fm[h];
        o |= (uint)(clamp(convert_int_rte(v * inv), -127, 127) & 0xFF) << (8 * b);
      }
      dst[k * GL] = o;
    }
  }
  hd[0 * GL + lane] = as_uint(D);
  hd[1 * GL + lane] = 0u;
  for (int f = 0; f < 4; ++f) {
    hd[(2 + f) * GL + lane] = scw[f];
    hd[(6 + f) * GL + lane] = 0u;
  }
#else
  uint scw[4] = {0, 0, 0, 0}, mnw[4] = {0, 0, 0, 0};
  float d0 = 0.0f, dm0 = 0.0f;
  for (int ib = 0; ib < 8; ++ib) {
    uint w[8];
    int sc0, sc1, mn0, mn1;
    float d, dmin;
    decode32(p, ib, grid, aux, w, &sc0, &sc1, &mn0, &mn1, &d, &dmin);
    __global uint* dst = wq + (((size_t)g * nsb + s) * 4 + ib / 2) * 16 * GL + (ib % 2) * 8 * GL + lane;
#pragma unroll
    for (int k = 0; k < 8; ++k) dst[k * GL] = w[k];
#if PER32
    hd[ib * GL + lane] = as_uint(d);
#else
    const int b0 = 2 * ib, b1 = 2 * ib + 1;
    scw[b0 / 4] |= (uint)(uchar)(char)sc0 << (8 * (b0 % 4));
    scw[b1 / 4] |= (uint)(uchar)(char)sc1 << (8 * (b1 % 4));
    mnw[b0 / 4] |= (uint)(uchar)mn0 << (8 * (b0 % 4));
    mnw[b1 / 4] |= (uint)(uchar)mn1 << (8 * (b1 % 4));
    d0 = d;
    dm0 = dmin;
#endif
  }
#if !PER32
  hd[0 * GL + lane] = as_uint(d0);
  hd[1 * GL + lane] = as_uint(dm0);
#pragma unroll
  for (int f = 0; f < 4; ++f) {
    hd[(2 + f) * GL + lane] = scw[f];
    hd[(6 + f) * GL + lane] = mnw[f];
  }
#endif
#endif  // requantized types
}
// Same conversion for 16 rows of SL_NSB superblocks (the whole rows: K = 256 * SL_NSB) per work-group of
// 16 * SL_NSB: the rows' bytes (contiguous) are first copied to local memory with coalesced loads, then each
// work-item decodes one superblock from there (lane = row, so the W8 stores of a sub-group are contiguous).
// Types without mins only. grid (N / 16 * 16 * SL_NSB), local (16 * SL_NSB).
#ifdef SL_NSB
__kernel void decode_w8_sl(__global const uchar* raw, int N, int K, __global const uint* tabs, __global uint* wq,
                           __global uint* wh XARGS_W8) {
#ifdef GRID_OFF
  __local uint lgrid[GRID_N];
  for (int i = get_local_id(0); i < GRID_N; i += get_local_size(0)) lgrid[i] = tabs[GRID_OFF + i];
  const uint* grid = lgrid;
#else
  const uint* grid = tabs;
#endif
#ifdef AUX_OFF
  __local uint laux[AUX_N];
  for (int i = get_local_id(0); i < AUX_N; i += get_local_size(0)) laux[i] = tabs[AUX_OFF + i];
  const uint* aux = laux;
#else
  const uint* aux = tabs;
#endif
  __local uint sl[16 * SL_NSB * BBP / 4];
  const int lid = get_local_id(0), nsb = SL_NSB, n0 = get_group_id(0) * 16;
  {
#ifdef XCACHE
    __global const uint* src = (__global const uint*)XROW(raw, xoff, xe0 + n0 / xn, 0, n0 % xn, 0);
#else
    __global const uint* src = (__global const uint*)(raw + (size_t)n0 * nsb * BBP);
#endif
    for (int i = lid; i < 16 * SL_NSB * BBP / 4; i += 16 * SL_NSB) sl[i] = src[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = lid % 16, s = lid / 16, n = n0 + lane, g = n / GL;
  const uchar* p = (const uchar*)(sl + (lane * nsb + s) * (BBP / 4));
  __global uint* hd = wh + ((size_t)g * nsb + s) * HSTR * GL;
  uint scw[4] = {0, 0, 0, 0};
  float d0 = 0.0f;
  for (int ib = 0; ib < 8; ++ib) {
    uint w[8];
    int sc0, sc1, mn0, mn1;
    float d, dmin;
    decode32(p, ib, grid, aux, w, &sc0, &sc1, &mn0, &mn1, &d, &dmin);
    __global uint* dst = wq + (((size_t)g * nsb + s) * 4 + ib / 2) * 16 * GL + (ib % 2) * 8 * GL + lane;
#pragma unroll
    for (int k = 0; k < 8; ++k) dst[k * GL] = w[k];
#if PER32
    hd[ib * GL + lane] = as_uint(d);
#else
    const int b0 = 2 * ib, b1 = 2 * ib + 1;
    scw[b0 / 4] |= (uint)(uchar)(char)sc0 << (8 * (b0 % 4));
    scw[b1 / 4] |= (uint)(uchar)(char)sc1 << (8 * (b1 % 4));
    d0 = d;
#endif
  }
#if !PER32
  hd[0 * GL + lane] = as_uint(d0);
  hd[1 * GL + lane] = 0u;
#pragma unroll
  for (int f = 0; f < 4; ++f) {
    hd[(2 + f) * GL + lane] = scw[f];
    hd[(6 + f) * GL + lane] = 0u;
  }
#endif
}
#endif

// Single-token matrix-vector product on the padded raw rows (token generation / lm_head):
//   C[c_off + n] (+)= sum_k W[n][k] * a[k],  a = int8 activations of token 0 with per-32 scales (row stride M)
//   and per-16 sums. Each sub-group streams GROWS rows; per step it block-reads two superblocks (coalesced)
//   into local memory, then lane l decodes 32-block (l % 8) of superblock (l / 8) from there.
//   Codebooks are staged in local memory once per work-group. grid (ceil(N/GROWS)*16 rounded to 256), local 256.
#ifndef GROWS
#define GROWS 4
#endif
// -DSOFTCAP=c (output head of Gemma): results are soft-capped, c * tanh(v / c)
#ifdef SOFTCAP
#define HEAD_OUT(v) (SOFTCAP * tanh((v) * (1.0f / SOFTCAP)))
#else
#define HEAD_OUT(v) (v)
#endif

// Split-K (-DSPLITK=S, small matrices): work-group column g1 = get_group_id(1) takes superblock pairs
// [g1 * P / S, (g1 + 1) * P / S) of the row and writes its partial sums to C[g1 * N + n] (no c_off, no
// accumulation); splitk_sum adds the S partials into the output. With a few hundred to a few thousand rows a
// GEMV streams each row with too few sub-groups in flight to hide the memory latency (Qwen3.6 35B: 512-row
// matrices at 36 GB/s, 2048-row at 49, 8192-row at 62).
#ifdef SPLITK
#define SK_LO(nsb) (2 * ((int)get_group_id(1) * (((nsb) + 1) / 2) / SPLITK))
#define SK_HI(nsb) min((nsb), 2 * (((int)get_group_id(1) + 1) * (((nsb) + 1) / 2) / SPLITK))
#define SK_STORE(n, v) C[(size_t)get_group_id(1) * N + (n)] = (v)
#else
#define SK_LO(nsb) 0
#define SK_HI(nsb) (nsb)
#define SK_STORE(n, v) C[c_off + (n)] = accumulate ? C[c_off + (n)] + (v) : (v)
#endif
// C[c_off + n] (+)= sum of the S split-K partials p[s * N + n]. grid (N).
__kernel void splitk_sum(__global const float* p, int S, int N, __global float* C, int c_off, int accumulate) {
  const int n = get_global_id(0);
  if (n >= N) return;
  float v = 0.0f;
  for (int i = 0; i < S; ++i) v += p[(size_t)i * N + n];
  C[c_off + n] = accumulate ? C[c_off + n] + v : v;
}
#define WIN_U (2 * BBP / 4)          // uints per two-superblock window
#define WIN_CALLS ((WIN_U + 15) / 16)
__attribute__((intel_reqd_sub_group_size(16)))
// With -DROWIDX the kernel computes only the rows listed in rowidx[0..N) (compact output; MTP draft head).
__kernel void gemv(__global const uchar* raw, int N, int K, __global const uint* tabs, __global const char* aq,
                   __global const float* asc, __global const ushort* absum, int M, __global float* C, int c_off,
                   int accumulate
#ifdef ROWIDX
                   , __global const int* rowidx
#endif
                   ) {
  __local uint win_all[16][WIN_CALLS * 16];
  __local uint* win = win_all[get_sub_group_id() & 15];
#ifdef GRID_OFF
  __local uint lgrid[GRID_N];
  for (int i = get_local_id(0); i < GRID_N; i += get_local_size(0)) lgrid[i] = tabs[GRID_OFF + i];
  const uint* grid = lgrid;
#else
  const uint* grid = tabs;
#endif
#ifdef AUX_OFF
  __local uint laux[AUX_N];
  for (int i = get_local_id(0); i < AUX_N; i += get_local_size(0)) laux[i] = tabs[AUX_OFF + i];
  const uint* aux = laux;
#else
  const uint* aux = tabs;
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id(), r0 = (get_global_id(0) / 16) * GROWS;
  const int nsb = K / 256;
  float acc[GROWS];
#pragma unroll
  for (int r = 0; r < GROWS; ++r) acc[r] = 0.0f;
  for (int sp = SK_LO(nsb); sp < SK_HI(nsb); sp += 2) {
    const int b = sp * 8 + lane;  // this lane's 32-block within the row
    const bool has = sp + lane / 8 < nsb;
    const uint8 a8 = has ? vload8(0, (__global const uint*)(aq + 32 * b)) : (uint8)0;
    const float sa = has ? asc[(size_t)b * M] : 0.0f;
#if QT == Q2_K || QT == Q4_K || QT == Q5_K
    const int bs0 = has ? (short)absum[(size_t)(2 * b) * M] : 0, bs1 = has ? (short)absum[(size_t)(2 * b + 1) * M] : 0;
#endif
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      #ifdef ROWIDX
      const int n = rowidx[min(r0 + r, N - 1)];
      #else
      const int n = min(r0 + r, N - 1);
      #endif
      __global const uint* src = (__global const uint*)(raw + ((size_t)n * nsb + sp) * BBP);
#pragma unroll
      for (int c = 0; c < WIN_CALLS; ++c) win[c * 16 + lane] = intel_sub_group_block_read(src + c * 16);
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      uint w[8];
      int sc0, sc1, mn0, mn1;
      float d, dmin;
      decode32((const uchar*)win + (lane / 8) * BBP, lane % 8, grid, aux, w, &sc0, &sc1, &mn0, &mn1, &d, &dmin);
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      int lo = dot_4x8packed_ss_int(a8.s0, w[0]);
      lo += dot_4x8packed_ss_int(a8.s1, w[1]);
      lo += dot_4x8packed_ss_int(a8.s2, w[2]);
      lo += dot_4x8packed_ss_int(a8.s3, w[3]);
      int hi = dot_4x8packed_ss_int(a8.s4, w[4]);
      hi += dot_4x8packed_ss_int(a8.s5, w[5]);
      hi += dot_4x8packed_ss_int(a8.s6, w[6]);
      hi += dot_4x8packed_ss_int(a8.s7, w[7]);
#if QT == Q2_K || QT == Q4_K || QT == Q5_K
      acc[r] += sa * (d * (float)(sc0 * lo + sc1 * hi) - dmin * (float)(mn0 * bs0 + mn1 * bs1));
#else
      acc[r] += sa * d * (float)(sc0 * lo + sc1 * hi);
#endif
    }
  }
#pragma unroll
  for (int r = 0; r < GROWS; ++r) {
    const float v = HEAD_OUT(sub_group_reduce_add(acc[r]));
    const int n = r0 + r;
    if (lane == 0 && n < N) SK_STORE(n, v);
  }
}

// Activations of the TB tokens of gemv_mt / gemv_mt_v / gemv_glu. TB < MT_LAZY_MIN: loaded once per superblock pair
// into registers (MT_ACTS), each (row, token) dot reads them there (MT_ACT). TB >= MT_LAZY_MIN (default 4, the
// verification of 3..7 MTP drafts): held all together they spilled (TB 8: 8 x uint8 per lane, 128 GRF here; TB 4:
// 1-4 KB per thread with 4 rows per sub-group); instead each token's bytes are loaded
// where they are used (L1 hits: the same bytes for every row of the sub-group) and tokens beyond T are skipped.
// Same arithmetic for every (row, token): bit-identical results.
#ifndef MT_LAZY_MIN
#define MT_LAZY_MIN 4  // (TB 4 spilled 1-4 KB per thread too: 3-draft verification)
#endif
#if TB >= MT_LAZY_MIN
#define MT_ACTS
#define MT_ACTS_K
#define MT_ACT(t)                                                                       \
  if ((t) >= T) break;                                                                  \
  const uint8 at = has ? vload8(0, (__global const uint*)(aq + (size_t)(t) * K + 32 * b)) : (uint8)0; \
  const float st = has ? asc[(size_t)b * M + (t)] : 0.0f;
#define MT_ACT_K(t)                                                                     \
  MT_ACT(t)                                                                             \
  const int b0 = has ? (short)absum[(size_t)(2 * b) * M + (t)] : 0;                     \
  const int b1 = has ? (short)absum[(size_t)(2 * b + 1) * M + (t)] : 0;
#else
#define MT_ACTS                                                                         \
  uint8 a8[TB];                                                                         \
  float sa[TB];                                                                         \
  _Pragma("unroll") for (int t = 0; t < TB; ++t) {                                      \
    const bool ok = has && t < T;                                                       \
    a8[t] = ok ? vload8(0, (__global const uint*)(aq + (size_t)t * K + 32 * b)) : (uint8)0; \
    sa[t] = ok ? asc[(size_t)b * M + t] : 0.0f;                                         \
  }
#define MT_ACTS_K                                                                       \
  MT_ACTS                                                                               \
  int bs0[TB], bs1[TB];                                                                 \
  _Pragma("unroll") for (int t = 0; t < TB; ++t) {                                      \
    const bool ok = has && t < T;                                                       \
    bs0[t] = ok ? (short)absum[(size_t)(2 * b) * M + t] : 0;                            \
    bs1[t] = ok ? (short)absum[(size_t)(2 * b + 1) * M + t] : 0;                        \
  }
#define MT_ACT(t) const uint8 at = a8[t]; const float st = sa[t];
#define MT_ACT_K(t) MT_ACT(t) const int b0 = bs0[t], b1 = bs1[t];
#endif
#if QT == IQ4_XS || QT == Q6_K || QT == Q4_0 || QT == Q8_0 || defined(GRID_OFF)
// GEMVs that read the weights straight into registers (gemv_v: one token, gemv_mt_v: TB tokens): IQ4_XS, and Q6_K
// for output heads. As gemv / gemv_mt, lane l handles 32-element block (l % 8) of superblock (sp + l / 8), the 16
// lanes two contiguous superblocks; but each lane vector-loads its own bytes from global memory instead of the
// sub-group staging the window in local memory (block reads, two barriers, byte-wise rd32 from local memory): about
// half the instructions per weight, which matters where decoding, not memory, was the limit. IQ4_XS's 16 codebook
// values come from four local tables pre-shifted into byte 0..3, so a packed int8x4 word is four lookups ORed
// together. Same arithmetic as gemv / gemv_mt: bit-identical results. Measured (gemv_bench, weights streaming from
// memory): E4B IQ4_XS matrices 44-65 -> 67-70 GB/s, Q6_K output heads 66-67 -> 72 GB/s; Q4_K / Q5_K matrices were
// as fast or faster with gemv (their decoding is cheap and every lane would reload the shared header), so they
// stay there. Q4_0 (Google's QAT Gemma 4) in its regrouped GPU form: one 16-byte vector load per block.
#if QT == IQ4_XS
#define VB_TABLES 64
inline void vb_tables(__local uint* kt, __global const uint* tabs) {
  if (get_local_id(0) < 64) {
    const int t = get_local_id(0) / 16, v = get_local_id(0) % 16;
    kt[t * 16 + v] = ((tabs[TAB_KVAL + v / 4] >> (8 * (v % 4))) & 0xFFu) << (8 * t);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
#elif defined(GRID_OFF)
// codebook types (IQ1_M, IQ2_*, IQ3_*): the codebook (and the sign table) staged in local memory; each lane loads
// its superblock into registers with 8-byte vector loads and decodes its block there with decode32 (gemv decoded
// from a sub-group-staged local window with byte-wise reads: 27-61 GB/s on these types)
#ifdef AUX_OFF
#define VB_TABLES (GRID_N + AUX_N)
#else
#define VB_TABLES GRID_N
#endif
inline void vb_tables(__local uint* kt, __global const uint* tabs) {
  for (int i = get_local_id(0); i < GRID_N; i += get_local_size(0)) kt[i] = tabs[GRID_OFF + i];
#ifdef AUX_OFF
  for (int i = get_local_id(0); i < AUX_N; i += get_local_size(0)) kt[GRID_N + i] = tabs[AUX_OFF + i];
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
}
#else
#define VB_TABLES 1
inline void vb_tables(__local uint* kt, __global const uint* tabs) {}
#endif
// Block ib of the superblock at p, as decode32: 32 int8 weights packed in w (elements 4k..4k+3 in w[k]) and the
// scales of its two 16-element halves (dmin / mins: 0, kept for decode32's interface).
inline void vblock(__global const uchar* p, int ib, __local const uint* kt, uint* w, float* d, float* dmin, int* sc0,
                   int* sc1, int* mn0, int* mn1) {
  __global const uint* pu = (__global const uint*)p;
#if QT == IQ4_XS
  const uint2 h = vload2(0, pu);
  const uint4 q = vload4(0, (__global const uint*)(p + 8 + 16 * ib));
  *d = f16(h.x & 0xFFFFu);
  *dmin = 0.0f;
  *sc0 = *sc1 = (int)(((h.y >> (4 * ib)) & 0xFu) | (((h.x >> (16 + 2 * ib)) & 3u) << 4)) - 32;
  *mn0 = *mn1 = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint bk = k == 0 ? q.x : k == 1 ? q.y : k == 2 ? q.z : q.w;
    w[k] = kt[bk & 15u] | kt[16 + ((bk >> 8) & 15u)] | kt[32 + ((bk >> 16) & 15u)] | kt[48 + ((bk >> 24) & 15u)];
    w[4 + k] = kt[(bk >> 4) & 15u] | kt[16 + ((bk >> 12) & 15u)] | kt[32 + ((bk >> 20) & 15u)] | kt[48 + (bk >> 28)];
  }
#elif QT == Q6_K
  const int nn = ib / 4, quarter = ib % 4;
  const uint8 L = vload8(0, (__global const uint*)(p + 64 * nn + ((quarter & 1) ? 32 : 0)));
  const uint8 H = vload8(0, (__global const uint*)(p + 128 + 32 * nn));
  const uint sw = *(__global const ushort*)(p + 192 + 2 * ib);
  *d = f16(*(__global const ushort*)(p + 208));
  *dmin = 0.0f;
  *sc0 = (char)(sw & 0xFFu);
  *sc1 = (char)(sw >> 8);
  *mn0 = *mn1 = 0;
  const int lsh = quarter >= 2 ? 4 : 0, hsh = 2 * quarter;
#pragma unroll
  for (int k = 0; k < 8; ++k) w[k] = sub32_bytes(((L[k] >> lsh) & 0x0F0F0F0Fu) | (((H[k] >> hsh) & 0x03030303u) << 4));
#elif QT == Q8_0
  const uint8 q = vload8(0, pu + 4 + 8 * ib);
  *d = f16(((__global const ushort*)p)[ib]);
  *dmin = 0.0f;
  *sc0 = *sc1 = 1;
  *mn0 = *mn1 = 0;
  w[0] = q.s0; w[1] = q.s1; w[2] = q.s2; w[3] = q.s3; w[4] = q.s4; w[5] = q.s5; w[6] = q.s6; w[7] = q.s7;
#elif QT == Q4_0
  const uint4 q = vload4(0, pu + 4 + 4 * ib);
  *d = f16(((__global const ushort*)p)[ib]);
  *dmin = 0.0f;
  *sc0 = *sc1 = 1;
  *mn0 = *mn1 = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint qk = k == 0 ? q.x : k == 1 ? q.y : k == 2 ? q.z : q.w;
    w[k] = sub8_bytes(qk & 0x0F0F0F0Fu);
    w[4 + k] = sub8_bytes((qk >> 4) & 0x0F0F0F0Fu);
  }
#elif defined(GRID_OFF)
  // codebook types: the block's bytes with 16-bit vector loads (offsets 2 mod 4), codebook / signs from local memory
  __global const ushort* ps = (__global const ushort*)p;
  __local const uchar* ksg = (__local const uchar*)(kt + GRID_N);  // (types with a sign table)
  *dmin = 0.0f;
  *mn0 = *mn1 = 0;
#if QT == IQ3_XXS
  *d = f16(ps[(BB - 2) / 2]) * 0.25f;
  const ushort4 qv = as_ushort4(vload2(ib, pu));
  const uint a = pu[16 + ib];
  *sc0 = *sc1 = 1 + 2 * (int)(a >> 28);
  const uint qq[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint sg = ksg[(a >> (7 * l)) & 127];
    w[2 * l] = apply_signs4(kt[qq[l] & 0xFF], sg & 0xF);
    w[2 * l + 1] = apply_signs4(kt[qq[l] >> 8], sg >> 4);
  }
#elif QT == IQ2_XXS
  *d = f16(ps[(BB - 2) / 2]) * 0.125f;
  const uint2 qa = vload2(ib, pu);
  const uint a0 = qa.x, a1 = qa.y;
  *sc0 = *sc1 = 1 + 2 * (int)(a1 >> 28);
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint gi = (a0 >> (8 * l)) & 0xFF, sg = ksg[(a1 >> (7 * l)) & 127];
    w[2 * l] = apply_signs4(kt[2 * gi], sg & 0xF);
    w[2 * l + 1] = apply_signs4(kt[2 * gi + 1], sg >> 4);
  }
#elif QT == IQ3_S
  *d = f16(ps[(BB - 2) / 2]);
  const ushort4 qv = as_ushort4(vload2(ib, pu));
  const uint sgn = pu[18 + ib], h = (pu[16 + ib / 4] >> (8 * (ib % 4))) & 0xFF;
  *sc0 = *sc1 = 1 + 2 * (int)((pu[26] >> (4 * ib)) & 0xF);
  const uint qq[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint g1 = (qq[l] & 0xFF) | ((h << (8 - 2 * l)) & 256), g2 = (qq[l] >> 8) | ((h << (7 - 2 * l)) & 256);
    const uint sg = (sgn >> (8 * l)) & 0xFF;
    w[2 * l] = apply_signs4(kt[g1], sg & 0xF);
    w[2 * l + 1] = apply_signs4(kt[g2], sg >> 4);
  }
#elif QT == IQ2_XS
  *d = f16(ps[(BB - 2) / 2]) * 0.125f;
  const ushort4 qv = as_ushort4(vload2(ib, pu));
  const uint scb = (pu[16 + ib / 4] >> (8 * (ib % 4))) & 0xFF;
  *sc0 = 1 + 2 * (int)(scb & 0xF);
  *sc1 = 1 + 2 * (int)(scb >> 4);
  const uint qq[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint gi = qq[l] & 511, sg = ksg[qq[l] >> 9];
    w[2 * l] = apply_signs4(kt[2 * gi], sg & 0xF);
    w[2 * l + 1] = apply_signs4(kt[2 * gi + 1], sg >> 4);
  }
#elif QT == IQ2_S
  *d = f16(ps[(BB - 2) / 2]) * 0.125f;
  const uint qs4 = pu[ib], sg4 = pu[8 + ib];
  const uint h = (pu[16 + ib / 4] >> (8 * (ib % 4))) & 0xFF, scb = (pu[18 + ib / 4] >> (8 * (ib % 4))) & 0xFF;
  *sc0 = 1 + 2 * (int)(scb & 0xF);
  *sc1 = 1 + 2 * (int)(scb >> 4);
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint gi = ((qs4 >> (8 * l)) & 0xFF) | ((h << (8 - 2 * l)) & 0x300), sg = (sg4 >> (8 * l)) & 0xFF;
    w[2 * l] = apply_signs4(kt[2 * gi], sg & 0xF);
    w[2 * l + 1] = apply_signs4(kt[2 * gi + 1], sg >> 4);
  }
#else
  // (other codebook types, IQ1_M: decode32 from global memory, byte-wise)
#ifdef AUX_OFF
  decode32((const uchar*)p, ib, (const uint*)kt, (const uint*)(kt + GRID_N), w, sc0, sc1, mn0, mn1, d, dmin);
#else
  decode32((const uchar*)p, ib, (const uint*)kt, (const uint*)kt, w, sc0, sc1, mn0, mn1, d, dmin);
#endif
#endif
#endif
}
// decode_w8_sb for the types with a vector block decoder (vblock: the bytes loaded straight from global memory, no
// private copy of the superblock indexed byte by byte), scales + mins-free types only. grid (N, K/256), local (LSZ, 1)
// with LSZ dividing N and >= 64.
__kernel void decode_w8_vb(__global const uchar* raw, int N, int K, __global const uint* tabs, __global uint* wq,
                           __global uint* wh) {
  __local uint kt[VB_TABLES];
  vb_tables(kt, tabs);
  const int n = get_global_id(0), s = get_global_id(1);
  const int nsb = K / 256, g = n / GL, lane = n % GL;
  if (n >= N) return;
  __global const uchar* p = raw + ((size_t)n * nsb + s) * BBP;
  __global uint* hd = wh + ((size_t)g * nsb + s) * HSTR * GL;
  uint scw[4] = {0, 0, 0, 0};
  float d0 = 0.0f;
#pragma unroll
  for (int ib = 0; ib < 8; ++ib) {
    uint w[8];
    int sc0, sc1, mn0, mn1;
    float d, dmin;
    vblock(p, ib, kt, w, &d, &dmin, &sc0, &sc1, &mn0, &mn1);
    __global uint* dst = wq + (((size_t)g * nsb + s) * 4 + ib / 2) * 16 * GL + (ib % 2) * 8 * GL + lane;
#pragma unroll
    for (int k = 0; k < 8; ++k) dst[k * GL] = w[k];
#if PER32
    hd[ib * GL + lane] = as_uint(d);
#else
    const int b0 = 2 * ib, b1 = 2 * ib + 1;
    scw[b0 / 4] |= (uint)(uchar)(char)sc0 << (8 * (b0 % 4));
    scw[b1 / 4] |= (uint)(uchar)(char)sc1 << (8 * (b1 % 4));
    d0 = d;
#endif
  }
#if !PER32
  hd[0 * GL + lane] = as_uint(d0);
  hd[1 * GL + lane] = 0u;
#pragma unroll
  for (int f = 0; f < 4; ++f) {
    hd[(2 + f) * GL + lane] = scw[f];
    hd[(6 + f) * GL + lane] = 0u;
  }
#endif
}

// With -DROWIDX (as gemv): only the rows listed in rowidx[0..N), output compact (the MTP draft head).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_v(__global const uchar* raw, int N, int K, __global const uint* tabs, __global const char* aq,
                     __global const float* asc, __global const ushort* absum, int M, __global float* C, int c_off,
                     int accumulate
#ifdef ROWIDX
                     , __global const int* rowidx
#endif
                     ) {
  __local uint kt[VB_TABLES];
  vb_tables(kt, tabs);
  const int lane = get_sub_group_local_id(), r0 = (get_global_id(0) / 16) * GROWS;
  const int nsb = K / 256, ib = lane % 8, hs = lane / 8;
  float acc[GROWS];
#pragma unroll
  for (int r = 0; r < GROWS; ++r) acc[r] = 0.0f;
  for (int sp = SK_LO(nsb); sp < SK_HI(nsb); sp += 2) {
    const int b = sp * 8 + lane;
    const bool has = sp + hs < nsb;
    const int s = has ? sp + hs : sp;
    const uint8 a8 = has ? vload8(0, (__global const uint*)(aq + 32 * b)) : (uint8)0;
    const float sa = has ? asc[(size_t)b * M] : 0.0f;
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      uint w[8];
      float d, dmin;
      int sc0, sc1, mn0, mn1;
#ifdef ROWIDX
      const int row = rowidx[min(r0 + r, N - 1)];
#else
      const int row = min(r0 + r, N - 1);
#endif
      vblock(raw + ((size_t)row * nsb + s) * BBP, ib, kt, w, &d, &dmin, &sc0, &sc1, &mn0, &mn1);
      int lo = dot_4x8packed_ss_int(a8.s0, w[0]);
      lo += dot_4x8packed_ss_int(a8.s1, w[1]);
      lo += dot_4x8packed_ss_int(a8.s2, w[2]);
      lo += dot_4x8packed_ss_int(a8.s3, w[3]);
      int hi = dot_4x8packed_ss_int(a8.s4, w[4]);
      hi += dot_4x8packed_ss_int(a8.s5, w[5]);
      hi += dot_4x8packed_ss_int(a8.s6, w[6]);
      hi += dot_4x8packed_ss_int(a8.s7, w[7]);
      acc[r] += sa * d * (float)(sc0 * lo + sc1 * hi);
    }
  }
#pragma unroll
  for (int r = 0; r < GROWS; ++r) {
    const float v = HEAD_OUT(sub_group_reduce_add(acc[r]));
    const int n = r0 + r;
    if (lane == 0 && n < N) SK_STORE(n, v);
  }
}
#ifndef TB
#define TB 2
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_mt_v(__global const uchar* raw, int N, int K, __global const uint* tabs, __global const char* aq,
                        __global const float* asc, __global const ushort* absum, int M, int T, __global float* C,
                        int ldc, int c_off, int accumulate) {
  __local uint kt[VB_TABLES];
  vb_tables(kt, tabs);
  const int lane = get_sub_group_local_id(), r0 = (get_global_id(0) / 16) * GROWS;
  const int nsb = K / 256, ib = lane % 8, hs = lane / 8;
  float acc[GROWS][TB];
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int t = 0; t < TB; ++t) acc[r][t] = 0.0f;
  for (int sp = 0; sp < nsb; sp += 2) {
    const int b = sp * 8 + lane;
    const bool has = sp + hs < nsb;
    const int s = has ? sp + hs : sp;
    MT_ACTS
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      uint w[8];
      float d, dmin;
      int sc0, sc1, mn0, mn1;
      vblock(raw + ((size_t)min(r0 + r, N - 1) * nsb + s) * BBP, ib, kt, w, &d, &dmin, &sc0, &sc1, &mn0, &mn1);
#pragma unroll
      for (int t = 0; t < TB; ++t) {
        MT_ACT(t)
        int lo = dot_4x8packed_ss_int(at.s0, w[0]);
        lo += dot_4x8packed_ss_int(at.s1, w[1]);
        lo += dot_4x8packed_ss_int(at.s2, w[2]);
        lo += dot_4x8packed_ss_int(at.s3, w[3]);
        int hi = dot_4x8packed_ss_int(at.s4, w[4]);
        hi += dot_4x8packed_ss_int(at.s5, w[5]);
        hi += dot_4x8packed_ss_int(at.s6, w[6]);
        hi += dot_4x8packed_ss_int(at.s7, w[7]);
        acc[r][t] += st * d * (float)(sc0 * lo + sc1 * hi);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int t = 0; t < TB; ++t) {
      const float v = HEAD_OUT(sub_group_reduce_add(acc[r][t]));
      const int n = r0 + r;
      if (lane == 0 && n < N && t < T) {
        __global float* dst = C + (size_t)t * ldc + c_off + n;
        *dst = accumulate ? *dst + v : v;
      }
    }
}
#endif
#if QT == IQ4_XS || QT == Q4_0
// Gemma FFN input of the down projection for T <= TB tokens in one launch: y = gelu_tanh(x . gate) * (x . up),
// quantized per 32 values (act32 layout) into aq2 / asc2 / absum2 (not the input buffers: other work-groups are
// still reading those). A work-group (16 sub-groups) computes rows [32g, 32g + 32) of both matrices, a sub-group
// rows 32g + 2j, 32g + 2j + 1 of each (gemv_mt_v's arithmetic), then sub-group t quantizes token t's 32 values as
// quant_store16 would. Replaces gemm_gate + gemm_up + swiglu_quant (two launches per layer fewer; same values).
// grid (N / 32 * 256), local 256.
inline float gelu_tanh_v(float x) { return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_glu(__global const uchar* rg, __global const uchar* ru, int N, int K, __global const uint* tabs,
                       __global const char* aq, __global const float* asc, int M, int T, __global char* aq2,
                       __global float* asc2, __global ushort* absum2) {
  __local uint kt[VB_TABLES];
  __local float res[2][32][TB];
  vb_tables(kt, tabs);
  const int lane = get_sub_group_local_id(), sg = get_sub_group_id(), n0 = get_group_id(0) * 32;
  const int nsb = K / 256, ib = lane % 8, hs = lane / 8;
  float acc[4][TB];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int t = 0; t < TB; ++t) acc[r][t] = 0.0f;
  for (int sp = 0; sp < nsb; sp += 2) {
    const int b = sp * 8 + lane;
    const bool has = sp + hs < nsb;
    const int s = has ? sp + hs : sp;
    MT_ACTS
#pragma unroll
    for (int r = 0; r < 4; ++r) {  // gate rows 2sg, 2sg+1, then up rows 2sg, 2sg+1
      uint w[8];
      float d, dmin;
      int sc0, sc1, mn0, mn1;
      const int n = min(n0 + 2 * sg + (r & 1), N - 1);
      vblock((r < 2 ? rg : ru) + ((size_t)n * nsb + s) * BBP, ib, kt, w, &d, &dmin, &sc0, &sc1, &mn0, &mn1);
#pragma unroll
      for (int t = 0; t < TB; ++t) {
        MT_ACT(t)
        int lo = dot_4x8packed_ss_int(at.s0, w[0]);
        lo += dot_4x8packed_ss_int(at.s1, w[1]);
        lo += dot_4x8packed_ss_int(at.s2, w[2]);
        lo += dot_4x8packed_ss_int(at.s3, w[3]);
        int hi = dot_4x8packed_ss_int(at.s4, w[4]);
        hi += dot_4x8packed_ss_int(at.s5, w[5]);
        hi += dot_4x8packed_ss_int(at.s6, w[6]);
        hi += dot_4x8packed_ss_int(at.s7, w[7]);
        acc[r][t] += st * d * (float)(sc0 * lo + sc1 * hi);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int t = 0; t < TB; ++t) {
      const float v = sub_group_reduce_add(acc[r][t]);
      if (lane == 0) res[r / 2][2 * sg + (r & 1)][t] = v;
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg < T && sg < TB) {  // token sg: lane l holds values 2l, 2l + 1 of the block
    const int t = sg, k = n0 + 2 * lane;
    const float y0 = gelu_tanh_v(res[0][2 * lane][t]) * res[1][2 * lane][t];
    const float y1 = gelu_tanh_v(res[0][2 * lane + 1][t]) * res[1][2 * lane + 1][t];
    const float amax = sub_group_reduce_max(fmax(fabs(y0), fabs(y1)));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q0 = convert_int_rte(y0 * inv), q1 = convert_int_rte(y1 * inv);
    *(__global uchar2*)(aq2 + (size_t)t * N + k) = (uchar2)((uchar)q0, (uchar)q1);
    int sum = q0 + q1;  // per 16 values: lanes 0..7 and 8..15
    sum += intel_sub_group_shuffle_xor(sum, 1);
    sum += intel_sub_group_shuffle_xor(sum, 2);
    sum += intel_sub_group_shuffle_xor(sum, 4);
    if ((lane & 7) == 0) absum2[((size_t)n0 / 16 + lane / 8) * M + t] = (ushort)(short)sum;
    if (lane == 0) asc2[((size_t)n0 / 32) * M + t] = amax / 127.0f;
  }
}
#endif
// Matrix x few-tokens product on the padded raw rows (TB tokens share every weight read): used for small
// batches such as speculative-decoding verification. Same structure as gemv; activations of token t are row t
// (stride M) of the int8 layout. C[t * ldc + c_off + n] (+)= ... grid (ceil(N/GROWS)*16 rounded to 256), local 256.
#ifndef TB
#define TB 2
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_mt(__global const uchar* raw, int N, int K, __global const uint* tabs, __global const char* aq,
                      __global const float* asc, __global const ushort* absum, int M, int T, __global float* C,
                      int ldc, int c_off, int accumulate) {
  __local uint win_all[16][WIN_CALLS * 16];
  __local uint* win = win_all[get_sub_group_id() & 15];
#ifdef GRID_OFF
  __local uint lgrid[GRID_N];
  for (int i = get_local_id(0); i < GRID_N; i += get_local_size(0)) lgrid[i] = tabs[GRID_OFF + i];
  const uint* grid = lgrid;
#else
  const uint* grid = tabs;
#endif
#ifdef AUX_OFF
  __local uint laux[AUX_N];
  for (int i = get_local_id(0); i < AUX_N; i += get_local_size(0)) laux[i] = tabs[AUX_OFF + i];
  const uint* aux = laux;
#else
  const uint* aux = tabs;
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id(), r0 = (get_global_id(0) / 16) * GROWS;
  const int nsb = K / 256;
  float acc[GROWS][TB];
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int t = 0; t < TB; ++t) acc[r][t] = 0.0f;
  for (int sp = 0; sp < nsb; sp += 2) {
    const int b = sp * 8 + lane;
    const bool has = sp + lane / 8 < nsb;
    MT_ACTS_K
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      const int n = min(r0 + r, N - 1);
      __global const uint* src = (__global const uint*)(raw + ((size_t)n * nsb + sp) * BBP);
#pragma unroll
      for (int c = 0; c < WIN_CALLS; ++c) win[c * 16 + lane] = intel_sub_group_block_read(src + c * 16);
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      uint w[8];
      int sc0, sc1, mn0, mn1;
      float d, dmin;
      decode32((const uchar*)win + (lane / 8) * BBP, lane % 8, grid, aux, w, &sc0, &sc1, &mn0, &mn1, &d, &dmin);
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
      for (int t = 0; t < TB; ++t) {
        MT_ACT_K(t)
        int lo = dot_4x8packed_ss_int(at.s0, w[0]);
        lo += dot_4x8packed_ss_int(at.s1, w[1]);
        lo += dot_4x8packed_ss_int(at.s2, w[2]);
        lo += dot_4x8packed_ss_int(at.s3, w[3]);
        int hi = dot_4x8packed_ss_int(at.s4, w[4]);
        hi += dot_4x8packed_ss_int(at.s5, w[5]);
        hi += dot_4x8packed_ss_int(at.s6, w[6]);
        hi += dot_4x8packed_ss_int(at.s7, w[7]);
#if QT == Q2_K || QT == Q4_K || QT == Q5_K
        acc[r][t] += st * (d * (float)(sc0 * lo + sc1 * hi) - dmin * (float)(mn0 * b0 + mn1 * b1));
#else
        acc[r][t] += st * d * (float)(sc0 * lo + sc1 * hi);
#endif
      }
    }
  }
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int t = 0; t < TB; ++t) {
      const float v = HEAD_OUT(sub_group_reduce_add(acc[r][t]));
      const int n = r0 + r;
      if (lane == 0 && n < N && t < T) {
        __global float* dst = C + (size_t)t * ldc + c_off + n;
        *dst = accumulate ? *dst + v : v;
      }
    }
}

// ---- Mixture-of-experts on standard GGUF expert tensors (Unsloth / llama.cpp quantizations of Qwen3.5 / 3.6 MoE):
// the expert matrices [NE][N][K] of one type, in the GPU raw form. Same slots and tiles as kernels/moe.cl (tiles of
// up to TM slots of one expert); each sub-group computes GROWS rows of the tile's expert for its TM slots, decoding
// the weights as gemv does (two superblocks per step staged in local memory, lane l decodes 32-block l % 8 of the
// superblock l / 8). Only types without per-block mins (the IQ types, Q6_K, Q8_0, Q4_0).
// grid (ceil(rows / GROWS) * 16 rounded to 256, max_tiles), local (256, 1).
#ifdef MOE_Q
#ifndef TM
#define TM 8
#endif
#if QT == Q2_K || QT == Q4_K || QT == Q5_K
#define MOE_MIN 1  // (types with per-block mins: the activation sums per 16 are computed here, there is no absum)
#endif
// (types with a vector decoder, vblock: the block is read straight into registers instead of a staged window)
#if (QT == IQ4_XS || QT == Q6_K || QT == Q4_0 || QT == Q8_0 || defined(GRID_OFF)) && !defined(MOE_WINDOW)
#define MOE_VB 1
#endif
inline void moe_stage(__local uint* lgrid, __local uint* laux, __global const uint* tabs) {
#ifdef GRID_OFF
  for (int i = get_local_id(0); i < GRID_N; i += get_local_size(0)) lgrid[i] = tabs[GRID_OFF + i];
#endif
#ifdef AUX_OFF
  for (int i = get_local_id(0); i < AUX_N; i += get_local_size(0)) laux[i] = tabs[AUX_OFF + i];
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
}
#ifdef GRID_OFF
#define MOE_LGRID __local uint lgrid[GRID_N];
#define MOE_GRID ((const uint*)lgrid)
#else
#define MOE_LGRID __local uint lgrid[1];
#define MOE_GRID ((const uint*)tabs)
#endif
#ifdef AUX_OFF
#define MOE_LAUX __local uint laux[AUX_N];
#define MOE_AUX ((const uint*)laux)
#else
#define MOE_LAUX __local uint laux[1];
#define MOE_AUX ((const uint*)tabs)
#endif

// dot of the lane's decoded block w (sc0 / sc1 halves) with an int8 activation block a8
inline int moe_dot(uint8 a8, const uint* w, int sc0, int sc1) {
  int lo = dot_4x8packed_ss_int(a8.s0, w[0]);
  lo += dot_4x8packed_ss_int(a8.s1, w[1]);
  lo += dot_4x8packed_ss_int(a8.s2, w[2]);
  lo += dot_4x8packed_ss_int(a8.s3, w[3]);
  int hi = dot_4x8packed_ss_int(a8.s4, w[4]);
  hi += dot_4x8packed_ss_int(a8.s5, w[5]);
  hi += dot_4x8packed_ss_int(a8.s6, w[6]);
  hi += dot_4x8packed_ss_int(a8.s7, w[7]);
  return sc0 * lo + sc1 * hi;
}
// d * (scaled dot) - dmin * (mins x activation sums): the lane's block term before the activation scale
inline float moe_term(uint8 a8, const uint* w, float d, float dmin, int sc0, int sc1, int mn0, int mn1) {
#ifdef MOE_MIN
  const uint one = 0x01010101u;
  int slo = dot_4x8packed_ss_int(a8.s0, one), shi = dot_4x8packed_ss_int(a8.s4, one);
  slo += dot_4x8packed_ss_int(a8.s1, one);
  slo += dot_4x8packed_ss_int(a8.s2, one);
  slo += dot_4x8packed_ss_int(a8.s3, one);
  shi += dot_4x8packed_ss_int(a8.s5, one);
  shi += dot_4x8packed_ss_int(a8.s6, one);
  shi += dot_4x8packed_ss_int(a8.s7, one);
  return d * (float)moe_dot(a8, w, sc0, sc1) - dmin * (float)(mn0 * slo + mn1 * shi);
#else
  return d * (float)moe_dot(a8, w, sc0, sc1);
#endif
}

// h[slot][F] = silu(x W_gate_e^T) * (x W_up_e^T); x rows are the slots' tokens (aq [token][K], per-32 scales asc
// [K/32][M]); rg / ru: [NE * F] rows of K.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void moe_gateup_q(__global const uchar* rg, __global const uchar* ru, int F, int K, __global const uint* tabs,
                           __global const char* aq, __global const float* asc, int M, __global const int* tiles,
                           __global const int* stok, __global float* h XARGS_GU) {
  __local uint win_all[16][WIN_CALLS * 16];
  __local uint* win = win_all[get_sub_group_id() & 15];
  MOE_LGRID
  MOE_LAUX
  const int ti = get_group_id(1);
  if (ti >= tiles[0]) return;  // (uniform over the work-group)
#ifdef MOE_VB
  __local uint kt[VB_TABLES];
  vb_tables(kt, tabs);
#else
  moe_stage(lgrid, laux, tabs);
#endif
  __global const int* tp = tiles + 1 + 3 * ti;
  const int e = tp[0], r0 = tp[1], nr = tp[2];
  const int lane = get_sub_group_local_id(), n0 = (get_global_id(0) / 16) * GROWS, nsb = K / 256;
  __global const uchar* eg = XBASE(rg, xog, e, F);
  __global const uchar* eu = XBASE(ru, xou, e, F);
  float ag[GROWS][TM], au[GROWS][TM];
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int m = 0; m < TM; ++m) ag[r][m] = au[r][m] = 0.0f;
  for (int sp = 0; sp < nsb; sp += 2) {
    const int b = sp * 8 + lane;
    const bool has = sp + lane / 8 < nsb;
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      const int n = min(n0 + r, F - 1);
      // the gate and up blocks decoded once, then used for every slot of the tile
      uint wg[8], wu[8];
      int gs0, gs1, us0, us1, gm0 = 0, gm1 = 0, um0 = 0, um1 = 0;
      float dg, du, gdmin = 0.0f, udmin = 0.0f;
#ifdef MOE_VB
      {
        const int sbi = has ? sp + lane / 8 : sp;
        vblock(XR(eg, n, sbi), lane % 8, kt, wg, &dg, &gdmin, &gs0, &gs1, &gm0, &gm1);
        vblock(XR(eu, n, sbi), lane % 8, kt, wu, &du, &udmin, &us0, &us1, &um0, &um1);
      }
#else
#pragma unroll
      for (int gu = 0; gu < 2; ++gu) {
        __global const uint* src = (__global const uint*)(gu ? XR(eu, n, sp) : XR(eg, n, sp));
#pragma unroll
        for (int c = 0; c < WIN_CALLS; ++c) win[c * 16 + lane] = intel_sub_group_block_read(src + c * 16);
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        if (gu) decode32((const uchar*)win + (lane / 8) * BBP, lane % 8, MOE_GRID, MOE_AUX, wu, &us0, &us1, &um0, &um1, &du, &udmin);
        else decode32((const uchar*)win + (lane / 8) * BBP, lane % 8, MOE_GRID, MOE_AUX, wg, &gs0, &gs1, &gm0, &gm1, &dg, &gdmin);
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      }
#endif
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        if (m < nr) {
          const int tok = stok[r0 + m];
          const uint8 a8 = has ? vload8(0, (__global const uint*)(aq + (size_t)tok * K + 32 * b)) : (uint8)0;
          const float sa = has ? asc[(size_t)b * M + tok] : 0.0f;
          ag[r][m] += sa * moe_term(a8, wg, dg, gdmin, gs0, gs1, gm0, gm1);
          au[r][m] += sa * moe_term(a8, wu, du, udmin, us0, us1, um0, um1);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      const float g = sub_group_reduce_add(ag[r][m]), u = sub_group_reduce_add(au[r][m]);
      const int n = n0 + r;
      if (lane == 0 && n < F && m < nr) h[(size_t)(r0 + m) * F + n] = g / (1.0f + exp(-g)) * u;
    }
}

// y[slot][N] = weight(slot) * (h[slot] W_down_e^T); h quantized per slot (hq [slot][K], scales hsc [K/32][S]).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void moe_down_q(__global const uchar* rd, int N, int K, __global const uint* tabs, __global const char* hq,
                         __global const float* hsc, int S, __global const int* tiles, __global const float* swt,
                         __global float* y XARGS_D) {
  __local uint win_all[16][WIN_CALLS * 16];
  __local uint* win = win_all[get_sub_group_id() & 15];
  MOE_LGRID
  MOE_LAUX
  const int ti = get_group_id(1);
  if (ti >= tiles[0]) return;
#ifdef MOE_VB
  __local uint kt[VB_TABLES];
  vb_tables(kt, tabs);
#else
  moe_stage(lgrid, laux, tabs);
#endif
  __global const int* tp = tiles + 1 + 3 * ti;
  const int e = tp[0], r0 = tp[1], nr = tp[2];
  const int lane = get_sub_group_local_id(), n0 = (get_global_id(0) / 16) * GROWS, nsb = K / 256;
  __global const uchar* ed = XBASE(rd, xod, e, N);
  float acc[GROWS][TM];
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int m = 0; m < TM; ++m) acc[r][m] = 0.0f;
  for (int sp = 0; sp < nsb; sp += 2) {
    const int b = sp * 8 + lane;
    const bool has = sp + lane / 8 < nsb;
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      const int n = min(n0 + r, N - 1);
      uint w[8];
      int sc0, sc1, mn0, mn1;
      float d, dmin;
#ifdef MOE_VB
      vblock(XR(ed, n, (has ? sp + lane / 8 : sp)), lane % 8, kt, w, &d, &dmin, &sc0, &sc1,
             &mn0, &mn1);
#else
      __global const uint* src = (__global const uint*)XR(ed, n, sp);
#pragma unroll
      for (int c = 0; c < WIN_CALLS; ++c) win[c * 16 + lane] = intel_sub_group_block_read(src + c * 16);
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      decode32((const uchar*)win + (lane / 8) * BBP, lane % 8, MOE_GRID, MOE_AUX, w, &sc0, &sc1, &mn0, &mn1, &d, &dmin);
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);
#endif
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        if (m < nr) {
          const int row = r0 + m;
          const uint8 a8 = has ? vload8(0, (__global const uint*)(hq + (size_t)row * K + 32 * b)) : (uint8)0;
          const float sa = has ? hsc[(size_t)b * S + row] : 0.0f;
          acc[r][m] += sa * moe_term(a8, w, d, dmin, sc0, sc1, mn0, mn1);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      const float v = sub_group_reduce_add(acc[r][m]);
      const int n = n0 + r;
      if (lane == 0 && n < N && m < nr) y[(size_t)(r0 + m) * N + n] = swt[r0 + m] * v;
    }
}

#endif  // MOE_Q
