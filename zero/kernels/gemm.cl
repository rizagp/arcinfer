// C[M][ldc] (+)= A[M][K] * W[N][K]^T (written at column offset c_off): quantized weights x int8 activations,
// DP4a on Intel Xe-LPG. Compiled once per weight layout with -DQTYPE=<id> (and optionally -DACT32).
//
// Weight layouts (see src/quants.h and kernels/decode.cl), per 16-column group g and 256-superblock s:
//   native Q4_K(12) / Q5_K(13) / Q6_K(14) / Q8_0(8): repacked GGUF blocks
//   W8 (100..103): int8 q + per-16 int8 scales (+ per-16 mins), decoded just in time from any GGUF type.
//     100 = per-16 scales, 101 = per-32 scales, 102 = per-16 scales + mins, 103 = per-32 scales + mins
// Activations: aq [M][K] int8; asc [K/256][M] f32 (or [K/32][M] with -DACT32); absum [K/16][M] int16.
// accumulate: 0 = store, 1 = add to C, 2 = C = silu(C) * result (fused SwiGLU), 3 = gelu_tanh(C) * result.
//
// -DMOE_TILES (mixture-of-experts prompt chunks, -DACT32 only): W holds the decoded experts [e0, e1) stacked
// (N rows each); the work is the slot tiles (expert, first slot, rows) of those experts, tiles tstart[e0] ..
// tstart[e1] of the tile list (kernels/moe.cl moe_scan), one 16-column group x one tile per work-group, looping
// over the tiles with the launch's y size as stride. Activation row of slot r: rowmap[r] (use_map) or r itself;
// C row r (row stride ldc); rowscale[r] multiplies the result (use_scale: the routing weight).

#define SG 16
#ifndef TM
#define TM 16  // tokens per sub-group (16 or 32)
#endif
#define TMV (TM / 16)
#ifndef CG
#define CG 1  // 16-column groups per work-group
#endif
#define BCAST(x, i) intel_sub_group_shuffle((x), (i))
#ifdef DBG_NOBCAST
#define BCASTA(x, i) (x)
#else
#define BCASTA(x, i) BCAST(x, i)
#endif

#define Q4_K 12
#define Q5_K 13
#define Q6_K 14
#define Q8_0 8

#if QTYPE >= 100
#define W8 1
#define SC32 (QTYPE & 1)
#define HAS_MIN (QTYPE >= 102)
#define DPJ 16
#define HDR 10
#elif QTYPE == Q4_K
#define SC32 1
#define HAS_MIN 1
#define DPJ 8
#define HDR 6
#elif QTYPE == Q5_K
#define SC32 1
#define HAS_MIN 1
#define DPJ 10
#define HDR 6
#elif QTYPE == Q6_K
#define SC32 0
#define HAS_MIN 0
#define DPJ 12
#define HDR 5
#elif QTYPE == Q8_0
#define SC32 1
#define HAS_MIN 0
#define DPJ 16
#define HDR 8
#else
#error "unsupported QTYPE"
#endif

#ifndef TP  // tokens per step of the dot-product loop (divides TM)
#if defined(W8)
#define TP (SC32 && TM >= 16 ? 16 : 8)  // 27B prompt GEMM, IQ3: TP 16 +3-7 % vs one token at a time; 8 for 4 scales (16 spills)
#else
#define TP 1  // native types: Q8_0 is already scheduled well, the K types spill with TP > 1
#endif
#endif

#if QTYPE == Q4_K || QTYPE == Q5_K
#define DOT(a, w) dot_4x8packed_su_int((a), (w))
#else
#define DOT(a, w) dot_4x8packed_ss_int((a), (w))
#endif

// W8: int32 sum x weight scale (int8), activation sum (|b| <= 2048) x min (0..255) as 32x16 and 16x16-bit
// multiplies: one instruction each, where the 32x32-bit multiply is emulated (mul + mach). Same products.
// The native types keep the plain multiply (their compiled schedule depends on it; not measured yet).
#if defined(W8)
#define MULS(x, sc) ((x) * (short)(sc))
#define MULMIN(b, mn) ((short)(b) * (short)(mn))
#else
#define MULS(x, sc) ((sc) * (x))
#define MULMIN(b, mn) ((mn) * (b))
#endif

#define SBYTE(u, k) ((int)(char)(((u) >> (8 * (k))) & 0xFF))
#define UBYTE(u, k) ((int)(((u) >> (8 * (k))) & 0xFF))

// Per-byte (q - 32) for 6-bit unsigned values packed in a uint, without cross-byte borrows.
inline uint sub32_bytes(uint q) { return ((q | 0x80808080u) - 0x20202020u) ^ 0x80808080u; }

__attribute__((intel_reqd_sub_group_size(SG)))
#ifdef MOE_TILES
__kernel void gemm(__global const uint* restrict wq, __global const uint* restrict wh,
                   __global const char* restrict aq, __global const float* restrict asc,
                   __global const ushort* restrict absum, __global float* restrict C,
                   int M, int N, int K, int ldc, int accumulate, __global const int* restrict tiles,
                   __global const int* restrict tstart, int e0, int e1, __global const int* restrict rowmap,
                   int use_map, __global const float* restrict rowscale, int use_scale) {
  const int nsb = K / 256, s_begin = 0, s_end = nsb, lane = get_sub_group_local_id();
  const int gc = get_group_id(0), t_end = tstart[e1];
  for (int t = tstart[e0] + get_group_id(1); t < t_end; t += get_num_groups(1)) {
  const int e = tiles[1 + 3 * t], r0 = tiles[2 + 3 * t], nr = tiles[3 + 3 * t];
  const int g = (e - e0) * (N / SG) + gc;  // column group in the stacked decoded experts
  const int rl = r0 + min(lane, nr - 1);   // this lane's slot (rows past the tile repeat its last slot)
  const int tokl = use_map ? rowmap[rl] : rl;
#else
__kernel void gemm(__global const uint* restrict wq, __global const uint* restrict wh,
                   __global const char* restrict aq, __global const float* restrict asc,
                   __global const ushort* restrict absum, __global float* restrict C,
                   int M, int N, int K, int ldc, int c_off, int accumulate, int rgroup, int s_begin, int s_end) {
  // Work-group w covers CG*16 columns x (WGY * TM) tokens: its sub-groups share weight columns (along y) and
  // activation rows (along x) through L1. Groups of `rgroup` token rows are swept together column block by
  // column block, so a weight slice is reused by rgroup rows while it is still in L3.
  // grid (N * rows, WGY), local (16 * CG, WGY), rows = M / (TM * WGY); N/16 must be a multiple of CG.
  const int NC = N / (SG * CG), rows = M / (TM * (int)get_local_size(1));
  const int w = get_group_id(0), grp = w / (rgroup * NC), rem = w % (rgroup * NC);
  const int rin = min(rgroup, rows - grp * rgroup);
  const int g = (rem / rin) * CG + (int)get_local_id(0) / SG, row = grp * rgroup + rem % rin;
  const int m0 = (row * (int)get_local_size(1) + (int)get_local_id(1)) * TM;
  const int nsb = K / 256;
#endif

  __global const uint* wqp = wq + (size_t)g * nsb * (4 * DPJ * SG);
  __global const uint* whp = wh + (size_t)g * nsb * (HDR * SG);
#ifndef MOE_TILES
  __global const char* ap = aq + (size_t)m0 * K;
#endif

  float res[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) res[m] = 0.0f;

  // superblocks [s_begin, s_end) of K (split-K: later splits accumulate into C)
  for (int s = s_begin; s < s_end; ++s) {
    __global const uint* hp = whp + s * HDR * SG;
    // ---- per-superblock weight header
#if defined(W8)
    const uint2 hd = intel_sub_group_block_read2(hp);
    const float d = as_float(hd.s0), dmin = as_float(hd.s1);
    const uint4 hs4 = intel_sub_group_block_read4(hp + 2 * SG);
    const uint hsc[4] = {hs4.s0, hs4.s1, hs4.s2, hs4.s3};
#if HAS_MIN
    const uint4 hm4 = intel_sub_group_block_read4(hp + 6 * SG);
    const uint hmn[4] = {hm4.s0, hm4.s1, hm4.s2, hm4.s3};
#endif
#elif QTYPE == Q4_K || QTYPE == Q5_K
    const uint4 h = intel_sub_group_block_read4(hp);
    const uint2 h2 = intel_sub_group_block_read2(hp + 4 * SG);
    const float d = as_float(h.s0), dmin = as_float(h.s1);
#elif QTYPE == Q6_K
    const float d = as_float(intel_sub_group_block_read(hp));
    const uint4 h = intel_sub_group_block_read4(hp + SG);
    const uint hsc[4] = {h.s0, h.s1, h.s2, h.s3};
#else
    const uint8 h = intel_sub_group_block_read8(hp);
    const float d8[8] = {as_float(h.s0), as_float(h.s1), as_float(h.s2), as_float(h.s3),
                         as_float(h.s4), as_float(h.s5), as_float(h.s6), as_float(h.s7)};
#endif

#if defined(ACT32) && QTYPE != Q8_0
    // activation-scaled partial sums of this superblock; the weight scale d (and dmin) is applied once at the end
    float tq[TM];
#pragma unroll
    for (int m = 0; m < TM; ++m) tq[m] = 0.0f;
#if HAS_MIN
    float tmn[TM];
#pragma unroll
    for (int m = 0; m < TM; ++m) tmn[m] = 0.0f;
#endif
#endif
#ifndef ACT32
#if QTYPE == Q8_0
    float isum[TM];
#pragma unroll
    for (int m = 0; m < TM; ++m) isum[m] = 0.0f;
#else
    int isum[TM];
#pragma unroll
    for (int m = 0; m < TM; ++m) isum[m] = 0;
#endif
#endif

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      __global const uint* dp = wqp + (s * 4 + j) * DPJ * SG;
      // ---- decode this 64-element chunk into 16 int8x4 words: lo = elements 0..31, hi = 32..63
      uint lo[8], hi[8];
#if defined(W8) || QTYPE == Q8_0
#ifdef DBG_NOWLOAD
      const uint8 r0 = (uint8)(s * 7 + j) ^ (uint8)(get_sub_group_local_id()), r1 = r0 * 3u;
#else
      const uint8 r0 = intel_sub_group_block_read8(dp), r1 = intel_sub_group_block_read8(dp + 8 * SG);
#endif
      lo[0] = r0.s0; lo[1] = r0.s1; lo[2] = r0.s2; lo[3] = r0.s3; lo[4] = r0.s4; lo[5] = r0.s5; lo[6] = r0.s6; lo[7] = r0.s7;
      hi[0] = r1.s0; hi[1] = r1.s1; hi[2] = r1.s2; hi[3] = r1.s3; hi[4] = r1.s4; hi[5] = r1.s5; hi[6] = r1.s6; hi[7] = r1.s7;
#else
      const uint8 raw = intel_sub_group_block_read8(dp);
      const uint r[8] = {raw.s0, raw.s1, raw.s2, raw.s3, raw.s4, raw.s5, raw.s6, raw.s7};
#if QTYPE == Q5_K
      const uint2 hb = intel_sub_group_block_read2(dp + 8 * SG);
#elif QTYPE == Q6_K
      const uint4 hb4 = intel_sub_group_block_read4(dp + 8 * SG);
      const uint hb[4] = {hb4.s0, hb4.s1, hb4.s2, hb4.s3};
#endif
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        lo[w] = r[w] & 0x0F0F0F0Fu;
        hi[w] = (r[w] >> 4) & 0x0F0F0F0Fu;
#if QTYPE == Q5_K
        lo[w] |= ((hb.s0 >> w) & 0x01010101u) << 4;
        hi[w] |= ((hb.s1 >> w) & 0x01010101u) << 4;
#elif QTYPE == Q6_K
        lo[w] = sub32_bytes(lo[w] | (((hb[w >> 2] >> (2 * (w & 3))) & 0x03030303u) << 4));
        hi[w] = sub32_bytes(hi[w] | (((hb[2 + (w >> 2)] >> (2 * (w & 3))) & 0x03030303u) << 4));
#endif
      }
#endif

      // ---- integer weight scales for the 16-blocks 4j..4j+3 (or the 32-blocks 2j, 2j+1)
#if QTYPE == Q4_K || QTYPE == Q5_K
      const uint scw = (j < 2) ? h.s2 : h.s3;
      const int sc0 = UBYTE(scw, 2 * (j & 1)), sc2 = UBYTE(scw, 2 * (j & 1) + 1);
      const uint mnw = (j < 2) ? h2.s0 : h2.s1;
      const int mn0 = UBYTE(mnw, 2 * (j & 1)), mn2 = UBYTE(mnw, 2 * (j & 1) + 1);
      const int mn1 = mn0, mn3 = mn2;
#elif QTYPE == Q8_0
      const float dlo = d8[2 * j], dhi = d8[2 * j + 1];
#else
      const int sc0 = SBYTE(hsc[j], 0), sc1 = SBYTE(hsc[j], 1), sc2 = SBYTE(hsc[j], 2), sc3 = SBYTE(hsc[j], 3);
#if HAS_MIN
      const int mn0 = UBYTE(hmn[j], 0), mn1 = UBYTE(hmn[j], 1), mn2 = UBYTE(hmn[j], 2), mn3 = UBYTE(hmn[j], 3);
#endif
#endif

#ifdef ACT32
      // per-32 activation scales (lane = token) and, for min types, per-16 activation sums
      float avl[TMV], avh[TMV];
#pragma unroll
      for (int v = 0; v < TMV; ++v) {
#ifdef MOE_TILES
        avl[v] = asc[((size_t)s * 8 + 2 * j) * M + tokl];
        avh[v] = asc[((size_t)s * 8 + 2 * j + 1) * M + tokl];
#else
        avl[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + ((size_t)s * 8 + 2 * j) * M + m0 + 16 * v)));
        avh[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + ((size_t)s * 8 + 2 * j + 1) * M + m0 + 16 * v)));
#endif
      }
#if HAS_MIN
      // per-16 activation sums as int (shuffling 16-bit values is emulated and slow)
      int b0[TMV], b1[TMV], b2[TMV], b3[TMV];
#pragma unroll
      for (int v = 0; v < TMV; ++v) {
        b0[v] = (short)intel_sub_group_block_read_us(absum + ((size_t)s * 16 + 4 * j + 0) * M + m0 + 16 * v);
        b1[v] = (short)intel_sub_group_block_read_us(absum + ((size_t)s * 16 + 4 * j + 1) * M + m0 + 16 * v);
        b2[v] = (short)intel_sub_group_block_read_us(absum + ((size_t)s * 16 + 4 * j + 2) * M + m0 + 16 * v);
        b3[v] = (short)intel_sub_group_block_read_us(absum + ((size_t)s * 16 + 4 * j + 3) * M + m0 + 16 * v);
      }
#endif
#endif

      // TP tokens at a time: their activation words are loaded first, then their dot products run side by side
      // (2 or 4 * TP independent DP4a chains). Written out explicitly because for some kernels the compiler keeps
      // this source order, and one token at a time leaves 2 chains and 1-3 loads in flight (3x slower).
#pragma unroll
      for (int mg = 0; mg < TM; mg += TP) {
      uint aw[TP];
#pragma unroll
      for (int p = 0; p < TP; ++p) {
        const int m = mg + p;
#ifdef DBG_NOALOAD
        aw[p] = (uint)(m * 0x01010101 + s + j) ^ get_sub_group_local_id();
#elif defined(MOE_TILES)
        aw[p] = intel_sub_group_block_read((__global const uint*)(aq + (size_t)BCAST(tokl, m) * K + s * 256 + j * 64));
#else
        aw[p] = intel_sub_group_block_read((__global const uint*)(ap + (size_t)m * K + s * 256 + j * 64));
#endif
      }
#if SC32
      int c0[TP], c1[TP];
#pragma unroll
      for (int p = 0; p < TP; ++p) c0[p] = c1[p] = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w)
#pragma unroll
        for (int p = 0; p < TP; ++p) {
          c0[p] += DOT(BCASTA(aw[p], w), lo[w]);
          c1[p] += DOT(BCASTA(aw[p], w + 8), hi[w]);
        }
#else
      int c0[TP], c1[TP], c2[TP], c3[TP];
#pragma unroll
      for (int p = 0; p < TP; ++p) c0[p] = c1[p] = c2[p] = c3[p] = 0;
#pragma unroll
      for (int w = 0; w < 4; ++w)
#pragma unroll
        for (int p = 0; p < TP; ++p) {
          c0[p] += DOT(BCAST(aw[p], w), lo[w]);
          c1[p] += DOT(BCAST(aw[p], w + 4), lo[w + 4]);
          c2[p] += DOT(BCAST(aw[p], w + 8), hi[w]);
          c3[p] += DOT(BCAST(aw[p], w + 12), hi[w + 4]);
        }
#endif
#pragma unroll
      for (int p = 0; p < TP; ++p) {
        const int m = mg + p;
#if SC32
        const int acc0 = c0[p], acc1 = c1[p];
#if QTYPE == Q8_0
#ifdef ACT32
        res[m] = fma(BCAST(avl[m / 16], m % 16) * dlo, (float)acc0, fma(BCAST(avh[m / 16], m % 16) * dhi, (float)acc1, res[m]));
#else
        isum[m] = fma(dlo, (float)acc0, fma(dhi, (float)acc1, isum[m]));
#endif
#else
        const int plo = MULS(acc0, sc0), phi = MULS(acc1, sc2);
#endif
#else
        const int acc0 = c0[p], acc1 = c1[p], acc2 = c2[p], acc3 = c3[p];
        const int plo = MULS(acc0, sc0) + MULS(acc1, sc1), phi = MULS(acc2, sc2) + MULS(acc3, sc3);
#endif
#if QTYPE != Q8_0
#if defined(DBG_NOPOST)
        tq[m] += as_float(plo + phi);
#elif defined(ACT32)
        tq[m] = fma(BCAST(avl[m / 16], m % 16), (float)plo, fma(BCAST(avh[m / 16], m % 16), (float)phi, tq[m]));
#if HAS_MIN
        const int mlo = MULMIN(BCAST(b0[m / 16], m % 16), mn0) + MULMIN(BCAST(b1[m / 16], m % 16), mn1);
        const int mhi = MULMIN(BCAST(b2[m / 16], m % 16), mn2) + MULMIN(BCAST(b3[m / 16], m % 16), mn3);
        tmn[m] = fma(BCAST(avl[m / 16], m % 16), (float)mlo, fma(BCAST(avh[m / 16], m % 16), (float)mhi, tmn[m]));
#endif
#else
        isum[m] += plo + phi;
#endif
#endif
      }
      }
    }

#if defined(ACT32) && QTYPE != Q8_0
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#if HAS_MIN
      res[m] = fma(d, tq[m], fma(-dmin, tmn[m], res[m]));
#else
      res[m] = fma(d, tq[m], res[m]);
#endif
    }
#endif
#ifndef ACT32
    // ---- superblock epilogue: activation scale (and the min term)
    float av[TMV];
#pragma unroll
    for (int v = 0; v < TMV; ++v) av[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + (size_t)s * M + m0 + 16 * v)));
#if HAS_MIN
    int bs[16][TMV];
#pragma unroll
    for (int b = 0; b < 16; ++b)
#pragma unroll
      for (int v = 0; v < TMV; ++v) bs[b][v] = (short)intel_sub_group_block_read_us(absum + ((size_t)s * 16 + b) * M + m0 + 16 * v);
    int mnv[16];
#if QTYPE == Q4_K || QTYPE == Q5_K
#pragma unroll
    for (int b = 0; b < 16; ++b) mnv[b] = UBYTE((b < 8) ? h2.s0 : h2.s1, (b / 2) & 3);
#else
#pragma unroll
    for (int b = 0; b < 16; ++b) mnv[b] = UBYTE(hmn[b / 4], b & 3);
#endif
#endif
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      const float a_s = BCAST(av[m / 16], m % 16);
#if QTYPE == Q8_0
      res[m] = fma(a_s, isum[m], res[m]);
#else
      res[m] = fma(a_s * d, (float)isum[m], res[m]);
#if HAS_MIN
      int msum = 0;
#pragma unroll
      for (int b = 0; b < 16; ++b) msum += mnv[b] * BCAST(bs[b][m / 16], m % 16);
      res[m] = fma(-a_s * dmin, (float)msum, res[m]);
#endif
#endif
    }
#endif
  }

#ifdef MOE_TILES
  const float rsc = use_scale ? rowscale[rl] : 1.0f;
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    if (m >= nr) break;
    __global uint* cp = (__global uint*)(C + (size_t)(r0 + m) * ldc + gc * SG);
    float v = res[m] * BCAST(rsc, m);
#else
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    __global uint* cp = (__global uint*)(C + c_off + (size_t)(m0 + m) * ldc + g * SG);
    float v = res[m];
#endif
    if (accumulate == 1) {
      v += as_float(intel_sub_group_block_read(cp));
    } else if (accumulate == 2) {  // fused SwiGLU: C holds the gate projection, v the up projection
      const float g = as_float(intel_sub_group_block_read(cp));
      v *= g / (1.0f + exp(-g));
    } else if (accumulate == 3) {  // fused GeGLU (tanh GELU), Gemma
      const float g = as_float(intel_sub_group_block_read(cp));
      v *= 0.5f * g * (1.0f + tanh(0.7978845608f * (g + 0.044715f * g * g * g)));
    }
    intel_sub_group_block_write(cp, as_uint(v));
  }
#ifdef MOE_TILES
  }
#endif
}
