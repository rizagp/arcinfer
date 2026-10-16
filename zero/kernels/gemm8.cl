// C[M][ldc] (+)= A[M][K] * W[N][K]^T on the decoded "W8x8" layout, SIMD8 + signed DP4a (the fastest DP4a
// form on Xe-LPG). Activations always use per-32 scales (ACT32).
//
// Weights, per 8-column group g8 and 256-superblock s ("lane" = column within the group):
//   wq: [g8][s][j=0..3][w=0..15][lane] uint  int8x4 words, elements 64j+4w..64j+4w+3
//   wh: [g8][s][10][lane] uint               d f32, dmin f32, sc[16] int8 (4 uints), mn[16] uint8 (4 uints)
//                                            (VARIANT 4: f0..f7 = f32 scale of each 32-block, sc = 1)
// VARIANT: 0 per-16 scales, 1 per-32 scales, 2 per-16 + mins, 3 per-32 + mins, 4 per-32 f32 scales (Q8_0).
// Activations: aq [M][K] int8, asc [K/32][M] f32, absum [K/16][M] int16 (per-16 sums, used with mins).
// Each sub-group computes TN groups of 8 columns x TM tokens. grid (N/(8*TN)*8, M/TM).

#define SG 8
#ifndef TM
#define TM 16
#endif
#ifndef TN
#define TN 2
#endif
#define BC(x, i) intel_sub_group_shuffle((x), (i))
#define DOT(a, w) dot_4x8packed_ss_int((a), (w))
#define SBYTE(u, k) ((int)(char)(((u) >> (8 * (k))) & 0xFF))
#define UBYTE(u, k) ((int)(((u) >> (8 * (k))) & 0xFF))

#define SC32 (VARIANT == 1 || VARIANT == 3)
#define HAS_MIN (VARIANT == 2 || VARIANT == 3)
#define FSCALE (VARIANT == 4)

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm8(__global const uint* restrict wq, __global const uint* restrict wh,
                    __global const char* restrict aq, __global const float* restrict asc,
                    __global const ushort* restrict absum, __global float* restrict C,
                    int M, int N, int K, int ldc, int c_off, int accumulate) {
  const int lane = get_sub_group_local_id();
  const int g0 = (get_global_id(0) / SG) * TN;
  const int m0 = get_global_id(1) * TM;
  const int nsb = K / 256;
  __global const char* ap = aq + (size_t)m0 * K;

  float res[TN][TM];
#pragma unroll
  for (int t = 0; t < TN; ++t)
#pragma unroll
    for (int m = 0; m < TM; ++m) res[t][m] = 0.0f;

  for (int s = 0; s < nsb; ++s) {
    float d[TN], dmin[TN];
    uint hsc[TN][4], hmn[TN][4];
    float fs[TN][8];
#pragma unroll
    for (int t = 0; t < TN; ++t) {
      __global const uint* hp = wh + ((size_t)(g0 + t) * nsb + s) * 10 * SG;
#if FSCALE
      const uint8 h8 = intel_sub_group_block_read8(hp);
      fs[t][0] = as_float(h8.s0); fs[t][1] = as_float(h8.s1); fs[t][2] = as_float(h8.s2); fs[t][3] = as_float(h8.s3);
      fs[t][4] = as_float(h8.s4); fs[t][5] = as_float(h8.s5); fs[t][6] = as_float(h8.s6); fs[t][7] = as_float(h8.s7);
#else
      const uint2 hd = intel_sub_group_block_read2(hp);
      d[t] = as_float(hd.s0);
      dmin[t] = as_float(hd.s1);
      const uint4 h4 = intel_sub_group_block_read4(hp + 2 * SG);
      hsc[t][0] = h4.s0; hsc[t][1] = h4.s1; hsc[t][2] = h4.s2; hsc[t][3] = h4.s3;
#if HAS_MIN
      const uint4 m4 = intel_sub_group_block_read4(hp + 6 * SG);
      hmn[t][0] = m4.s0; hmn[t][1] = m4.s1; hmn[t][2] = m4.s2; hmn[t][3] = m4.s3;
#endif
#endif
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint w[TN][16];
#pragma unroll
      for (int t = 0; t < TN; ++t) {
        __global const uint* dp = wq + (((size_t)(g0 + t) * nsb + s) * 4 + j) * 16 * SG;
        const uint8 r0 = intel_sub_group_block_read8(dp), r1 = intel_sub_group_block_read8(dp + 8 * SG);
        w[t][0] = r0.s0; w[t][1] = r0.s1; w[t][2] = r0.s2; w[t][3] = r0.s3;
        w[t][4] = r0.s4; w[t][5] = r0.s5; w[t][6] = r0.s6; w[t][7] = r0.s7;
        w[t][8] = r1.s0; w[t][9] = r1.s1; w[t][10] = r1.s2; w[t][11] = r1.s3;
        w[t][12] = r1.s4; w[t][13] = r1.s5; w[t][14] = r1.s6; w[t][15] = r1.s7;
      }
      // activation scales of the two 32-blocks (lane = token within each half of the tile)
      float avl[TM / SG], avh[TM / SG];
#pragma unroll
      for (int v = 0; v < TM / SG; ++v) {
        avl[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + ((size_t)s * 8 + 2 * j) * M + m0 + v * SG)));
        avh[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + ((size_t)s * 8 + 2 * j + 1) * M + m0 + v * SG)));
      }
#if HAS_MIN
      ushort bsv[4][TM / SG];
#pragma unroll
      for (int b = 0; b < 4; ++b)
#pragma unroll
        for (int v = 0; v < TM / SG; ++v)
          bsv[b][v] = intel_sub_group_block_read_us(absum + ((size_t)s * 16 + 4 * j + b) * M + m0 + v * SG);
#endif

#pragma unroll
      for (int m = 0; m < TM; ++m) {
        __global const uint* arow = (__global const uint*)(ap + (size_t)m * K + s * 256 + j * 64);
        const uint a0 = intel_sub_group_block_read(arow), a1 = intel_sub_group_block_read(arow + SG);
        const float sl = BC(avl[m / SG], m % SG), sh = BC(avh[m / SG], m % SG);
#if HAS_MIN
        const int bs0 = (short)BC(bsv[0][m / SG], m % SG), bs1 = (short)BC(bsv[1][m / SG], m % SG);
        const int bs2 = (short)BC(bsv[2][m / SG], m % SG), bs3 = (short)BC(bsv[3][m / SG], m % SG);
#endif
#pragma unroll
        for (int t = 0; t < TN; ++t) {
#if SC32 || FSCALE
          int acc0 = 0, acc1 = 0;
#pragma unroll
          for (int x = 0; x < 8; ++x) {
            acc0 += DOT(BC(a0, x), w[t][x]);
            acc1 += DOT(BC(a1, x), w[t][8 + x]);
          }
#if FSCALE
          res[t][m] = fma(sl * fs[t][2 * j], (float)acc0, fma(sh * fs[t][2 * j + 1], (float)acc1, res[t][m]));
#else
          const int plo = SBYTE(hsc[t][j], 0) * acc0, phi = SBYTE(hsc[t][j], 2) * acc1;
#endif
#else
          int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#pragma unroll
          for (int x = 0; x < 4; ++x) {
            acc0 += DOT(BC(a0, x), w[t][x]);
            acc1 += DOT(BC(a0, x + 4), w[t][x + 4]);
            acc2 += DOT(BC(a1, x), w[t][8 + x]);
            acc3 += DOT(BC(a1, x + 4), w[t][12 + x]);
          }
          const int plo = SBYTE(hsc[t][j], 0) * acc0 + SBYTE(hsc[t][j], 1) * acc1;
          const int phi = SBYTE(hsc[t][j], 2) * acc2 + SBYTE(hsc[t][j], 3) * acc3;
#endif
#if !FSCALE
#if HAS_MIN
          const int mlo = UBYTE(hmn[t][j], 0) * bs0 + UBYTE(hmn[t][j], 1) * bs1;
          const int mhi = UBYTE(hmn[t][j], 2) * bs2 + UBYTE(hmn[t][j], 3) * bs3;
          res[t][m] = fma(sl, fma(d[t], (float)plo, -dmin[t] * (float)mlo),
                          fma(sh, fma(d[t], (float)phi, -dmin[t] * (float)mhi), res[t][m]));
#else
          res[t][m] = fma(sl * d[t], (float)plo, fma(sh * d[t], (float)phi, res[t][m]));
#endif
#endif
        }
      }
    }
  }

#pragma unroll
  for (int t = 0; t < TN; ++t)
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      __global uint* cp = (__global uint*)(C + c_off + (size_t)(m0 + m) * ldc + (g0 + t) * SG);
      float v = res[t][m];
      if (accumulate) v += as_float(intel_sub_group_block_read(cp));
      intel_sub_group_block_write(cp, as_uint(v));
    }
}
