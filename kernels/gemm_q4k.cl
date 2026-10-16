// C[M][N] = A[M][K] * W[N][K]^T for Q4_K weights x int8 activations, via DP4a.
// Target: Intel Xe-LPG iGPU (no XMX), SIMD16 sub-groups.
//
// Weight layout (repacked from GGUF Q4_K at load time), per 16-column group g and superblock s (256 K):
//   wq: [g][s][j=0..3][w=0..7][lane] uint  -- 32 B nibbles per column per 64-K chunk; low nibbles are
//                                              sub-block 2j, high nibbles sub-block 2j+1 (same as Q4_K)
//   wh: [g][s][f=0..5][lane] uint         -- d (f32), dmin (f32), sc[0..3], sc[4..7], mn[0..3], mn[4..7]
// Activation layout (produced by the quantize kernel):
//   aq:  [M][K] int8, asc: [K/256][M] f32 scale, absum: [K/256][8][M] int16 sum of each 32-block
// Dequantized weight = d*sc[b]*q - dmin*mn[b]; activation = asc*a.

#ifndef TM
#define TM 16
#endif
#define SG 16
#define TMV ((TM + SG - 1) / SG)

#define BCAST(x, i) intel_sub_group_shuffle((x), (i))

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_q4k_q8(__global const uint* restrict wq, __global const uint* restrict wh,
                          __global const char* restrict aq, __global const float* restrict asc,
                          __global const ushort* restrict absum, __global float* restrict C,
                          int M, int N, int K) {
  const int g = get_global_id(0) / SG;
  const int lane = get_sub_group_local_id();
  const int m0 = get_global_id(1) * TM;
  const int nsb = K / 256;

  __global const uint* wqp = wq + (size_t)g * nsb * (4 * 8 * SG);
  __global const uint* whp = wh + (size_t)g * nsb * (6 * SG);
  __global const char* ap = aq + (size_t)m0 * K;

  float res[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) res[m] = 0.0f;

  for (int s = 0; s < nsb; ++s) {
    const uint4 h = intel_sub_group_block_read4(whp + s * 6 * SG);
    const uint2 h2 = intel_sub_group_block_read2(whp + s * 6 * SG + 4 * SG);
    const float d = as_float(h.s0), dmin = as_float(h.s1);

    int isum[TM];
#pragma unroll
    for (int m = 0; m < TM; ++m) isum[m] = 0;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint8 raw = intel_sub_group_block_read8(wqp + (s * 4 + j) * 8 * SG);
      const uint r[8] = {raw.s0, raw.s1, raw.s2, raw.s3, raw.s4, raw.s5, raw.s6, raw.s7};
      uint lo[8], hi[8];
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        lo[w] = r[w] & 0x0F0F0F0Fu;
        hi[w] = (r[w] >> 4) & 0x0F0F0F0Fu;
      }
      const uint scw = (j < 2) ? h.s2 : h.s3;
      const int sc_lo = (scw >> (16 * (j & 1))) & 0xFF;
      const int sc_hi = (scw >> (16 * (j & 1) + 8)) & 0xFF;

#pragma unroll
      for (int m = 0; m < TM; ++m) {
#ifdef A_UNIFORM
        // Same address in every lane: loaded once as a scalar and used directly as a scalar operand.
        __global const uint* apw = (__global const uint*)(ap + (size_t)m * K + s * 256 + j * 64);
        const uint8 a0 = vload8(0, apw), a1 = vload8(1, apw);
        const uint al[8] = {a0.s0, a0.s1, a0.s2, a0.s3, a0.s4, a0.s5, a0.s6, a0.s7};
        const uint ah[8] = {a1.s0, a1.s1, a1.s2, a1.s3, a1.s4, a1.s5, a1.s6, a1.s7};
        int acc0 = 0, acc1 = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          acc0 += dot_4x8packed_su_int(al[w], lo[w]);
          acc1 += dot_4x8packed_su_int(ah[w], hi[w]);
        }
#else
        const uint a = intel_sub_group_block_read((__global const uint*)(ap + (size_t)m * K + s * 256 + j * 64));
        int acc0 = 0, acc1 = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          acc0 += dot_4x8packed_su_int(BCAST(a, w), lo[w]);
          acc1 += dot_4x8packed_su_int(BCAST(a, w + 8), hi[w]);
        }
#endif
        isum[m] += sc_lo * acc0 + sc_hi * acc1;
      }
    }

    // Superblock epilogue: apply activation scale and the min (offset) term.
    float av[TMV];
    ushort bs[8][TMV];
#pragma unroll
    for (int v = 0; v < TMV; ++v) {
      av[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + (size_t)s * M + m0 + v * SG)));
#pragma unroll
      for (int b = 0; b < 8; ++b)
        bs[b][v] = intel_sub_group_block_read_us(absum + ((size_t)s * 8 + b) * M + m0 + v * SG);
    }
    int mn[8];
#pragma unroll
    for (int b = 0; b < 8; ++b) mn[b] = (((b < 4) ? h2.s0 : h2.s1) >> (8 * (b & 3))) & 0xFF;

#pragma unroll
    for (int m = 0; m < TM; ++m) {
      int msum = 0;
#pragma unroll
      for (int b = 0; b < 8; ++b) msum += mn[b] * (int)(short)BCAST(bs[b][m / SG], m % SG);
      const float a_s = BCAST(av[m / SG], m % SG);
      res[m] = fma(a_s * d, (float)isum[m], res[m]);
      res[m] = fma(-a_s * dmin, (float)msum, res[m]);
    }
  }

#pragma unroll
  for (int m = 0; m < TM; ++m)
    intel_sub_group_block_write((__global uint*)(C + (size_t)(m0 + m) * N + g * SG), as_uint(res[m]));
}
