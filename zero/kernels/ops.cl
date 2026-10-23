#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Non-GEMM kernels of the forward passes (norms, activation quantization, Gated DeltaNet, RoPE, attention,
// Gemma 4 and MTP helpers). Activation quantization layout matches kernels/gemm.cl.

#define SG 16
// int8 attention: scale blocks per q/K/V row (see attn_q8); the engine passes -DSB to the attention kernels
#ifndef SB
#define SB 1
#endif
// Gemma 4's float attention: the K rows in f32
#define KC_T float

inline float silu(float x) { return x / (1.0f + exp(-x)); }

// Quantizes superblock s of row m to int8. Lane l holds v[i] = element s*256 + 16*l + i.
// Scale per 256 elements (asc [K/256][M]) or, with act32, per 32 elements (asc [K/32][M]).
// absum [K/16][M] holds the int8 sum of each 16-element block (for weight types with mins).
inline void quant_store16(const float* v, int s, int m, int M, int K, __global char* aq, __global float* asc,
                          __global ushort* absum, int act32) {
  const int lane = get_sub_group_local_id();
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmax(amax, fabs(v[i]));
  if (act32) amax = fmax(amax, intel_sub_group_shuffle_xor(amax, 1));
  else amax = sub_group_reduce_max(amax);
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  int sum = 0;
  uint pk[4] = {0, 0, 0, 0};
#ifdef Q2_AQ
  float res[16];  // (two passes: what the int8 values miss)
#endif
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int q = convert_int_rte(v[i] * inv);
    sum += q;
    pk[i / 4] |= (uint)(q & 0xFF) << (8 * (i % 4));
#ifdef Q2_AQ
    res[i] = v[i] - (float)q * (amax / 127.0f);
#endif
  }
  vstore4((uint4)(pk[0], pk[1], pk[2], pk[3]), 0, (__global uint*)(aq + (size_t)m * K + s * 256 + 16 * lane));
  absum[((size_t)s * 16 + lane) * M + m] = (ushort)(short)sum;
  if (act32) {
    if ((lane & 1) == 0) asc[((size_t)s * 8 + lane / 2) * M + m] = amax / 127.0f;
  } else if (lane == 0) {
    asc[(size_t)s * M + m] = amax / 127.0f;
  }
#ifdef Q2_AQ
  // the residual rounded the same way, in the same layout, Q2_AQ bytes / Q2_SC floats / Q2_AB shorts further on
  {
    float rmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) rmax = fmax(rmax, fabs(res[i]));
    if (act32) rmax = fmax(rmax, intel_sub_group_shuffle_xor(rmax, 1));
    else rmax = sub_group_reduce_max(rmax);
    const float rinv = rmax > 0.0f ? 127.0f / rmax : 0.0f;
    int rsum = 0;
    uint rk[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = convert_int_rte(res[i] * rinv);
      rsum += q;
      rk[i / 4] |= (uint)(q & 0xFF) << (8 * (i % 4));
    }
    __global char* aq2 = aq + Q2_AQ;
    __global float* asc2 = asc + Q2_SC;
    __global ushort* ab2 = absum + Q2_AB;
    vstore4((uint4)(rk[0], rk[1], rk[2], rk[3]), 0, (__global uint*)(aq2 + (size_t)m * K + s * 256 + 16 * lane));
    ab2[((size_t)s * 16 + lane) * M + m] = (ushort)(short)rsum;
    if (act32) {
      if ((lane & 1) == 0) asc2[((size_t)s * 8 + lane / 2) * M + m] = rmax / 127.0f;
    } else if (lane == 0) {
      asc2[(size_t)s * M + m] = rmax / 127.0f;
    }
  }
#endif
}

// y = rmsnorm(x[row_off + m]) * w, quantized for the next GEMM. Optionally also stores y as f32.
// Work-group: 256 items per row; grid (256, M).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void rmsnorm_quant(__global const float* x, int row_off, __global const float* w, int K, int M, int M_valid,
                            float eps, __global char* aq, __global float* asc, __global ushort* absum,
                            __global float* hout, int write_h, int act32) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  __global const float* xr = x + (size_t)(m + row_off) * K;
  float ss = 0.0f;
  if (valid)
    for (int i = get_local_id(0); i < K; i += get_local_size(0)) ss += xr[i] * xr[i];
  ss = work_group_reduce_add(ss);
  const float r = rsqrt(ss / K + eps);
  const int lane = get_sub_group_local_id();
  for (int s = get_sub_group_id(); s < K / 256; s += get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = valid ? xr[base + i] * r * w[base + i] : 0.0f;
    if (write_h)
#pragma unroll
      for (int i = 0; i < 16; ++i) hout[(size_t)m * K + base + i] = v[i];
    quant_store16(v, s, m, M, K, aq, asc, absum, act32);
  }
}

inline float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }

// y = silu(gate) * up (or gelu_tanh(gate) * up) from gu[m] = [gate(F) | up(F)], quantized (K = F).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void swiglu_quant(__global const float* gu, int F, int M, int M_valid, __global char* aq,
                           __global float* asc, __global ushort* absum, int act32, int fused, int gelu) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  const int lane = get_sub_group_local_id();
  // gu rows: [gate | up] (2F), or with `fused` the GEMM already wrote silu(gate) * up (F)
  __global const float* gr = gu + (size_t)m * (fused ? F : 2 * F);
  // superblocks over every sub-group along dim 0: grid (256, M) = one work-group per row; for a few rows the engine
  // launches (16 * F/256, M) with local (16, 1), one sub-group per superblock
  const int nsg = get_num_groups(0) * get_num_sub_groups();
  for (int s = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(); s < F / 256; s += nsg) {
    const int base = s * 256 + 16 * lane;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      v[i] = !valid ? 0.0f : fused ? gr[base + i] : (gelu ? gelu_tanh(gr[base + i]) : silu(gr[base + i])) * gr[F + base + i];
    quant_store16(v, s, m, M, F, aq, asc, absum, act32);
  }
}

// Causal depthwise conv (kernel 4) + SiLU over qkv channels, then L2-norm of each q/k head (128).
// q is additionally scaled by 1/sqrt(128) (the delta-net output scale). One sub-group per (128-channel
// group, CT consecutive tokens; CT = 1 measured fastest), with vector loads; groups [0, nqk) are q heads,
// [nqk, 2nqk) k heads, the rest v heads. grid (C/128*16, ceil(M/CT)).
#ifndef CT
#define CT 1
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gdn_conv(__global const float* qkv, __global const float* cstate, __global const float* cw, int C,
                       int nqk, int M_valid, float eps, __global float* out) {
  const int m0 = get_global_id(1) * CT, hg = get_global_id(0) / SG, lane = get_sub_group_local_id();
  const int c0 = hg * 128 + lane * 8;
  if (m0 >= M_valid) return;
  // taps: cw [c][4] -> wt[t] = weights of tap t for channels c0..c0+7
  const float16 wa = vload16(0, cw + (size_t)c0 * 4), wb = vload16(1, cw + (size_t)c0 * 4);
  const float8 wt0 = (float8)(wa.s0, wa.s4, wa.s8, wa.sc, wb.s0, wb.s4, wb.s8, wb.sc);
  const float8 wt1 = (float8)(wa.s1, wa.s5, wa.s9, wa.sd, wb.s1, wb.s5, wb.s9, wb.sd);
  const float8 wt2 = (float8)(wa.s2, wa.s6, wa.sa, wa.se, wb.s2, wb.s6, wb.sa, wb.se);
  const float8 wt3 = (float8)(wa.s3, wa.s7, wa.sb, wa.sf, wb.s3, wb.s7, wb.sb, wb.sf);
  float8 h[3];
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    const int tt = m0 - 3 + t;
    h[t] = tt >= 0 ? vload8(0, qkv + (size_t)tt * C + c0) : vload8(0, cstate + (size_t)(tt + 3) * C + c0);
  }
  const bool qk = hg < 2 * nqk;
  const float qscale = hg < nqk ? rsqrt(128.0f) : 1.0f;
  float8 xs[CT];
#pragma unroll
  for (int j = 0; j < CT; ++j) xs[j] = m0 + j < M_valid ? vload8(0, qkv + (size_t)(m0 + j) * C + c0) : (float8)0.0f;
#pragma unroll
  for (int j = 0; j < CT; ++j) {
    const int m = m0 + j;
    if (m >= M_valid) break;
    const float8 x = xs[j];
    const float8 acc = wt0 * h[0] + wt1 * h[1] + wt2 * h[2] + wt3 * x;
    float8 v = acc / (1.0f + exp(-acc));
    if (qk) {
      const float ss = sub_group_reduce_add(dot(v.lo, v.lo) + dot(v.hi, v.hi));
      v *= rsqrt(ss + eps) * qscale;
    }
    vstore8(v, 0, out + (size_t)m * C + c0);
    h[0] = h[1];
    h[1] = h[2];
    h[2] = x;
  }
}

// Keeps the last 3 raw qkv rows as conv history. grid (C).
__kernel void conv_state_update(__global const float* qkv, __global float* cstate, int C, int M_valid) {
  const int c = get_global_id(0);
  if (c >= C) return;
  const float old[3] = {cstate[c], cstate[C + c], cstate[2 * C + c]};
  float nw[3];
  for (int i = 0; i < 3; ++i) {
    const int t = M_valid - 3 + i;
    nw[i] = t >= 0 ? qkv[(size_t)t * C + c] : old[i + M_valid];
  }
  for (int i = 0; i < 3; ++i) cstate[(size_t)i * C + c] = nw[i];
}

// In place: ab[m] = [alpha | beta] (raw projections) -> [decay | beta] with
// decay = exp(softplus(alpha + dt_bias) * ssm_a), beta = sigmoid(beta). n = M_valid * 2 * nv.
__kernel void gdn_gates(__global float* ab, __global const float* dt_bias, __global const float* ssm_a, int nv, int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  const int c = i % (2 * nv);
  const float x = ab[i];
  if (c < nv) {
    const float a = x + dt_bias[c];
    ab[i] = exp((a > 20.0f ? a : log(1.0f + exp(a))) * ssm_a[c]);
  } else {
    ab[i] = 1.0f / (1.0f + exp(-x));
  }
}

// Gated delta rule, sequential over tokens. State S[h][j][i] (i = key dim) f32.
// v-head h uses k-head h % nk (ggml repeat semantics). SIMD8: each lane owns one full state column j (128
// floats in registers). q and k are block-read across the 8 lanes and used through constant-index lane
// broadcasts (free register regioning), so no reductions or redundant loads are needed. Next-token
// operands are prefetched one step ahead. Needs the 256-GRF build.
// x: conv output [M][C] = [q (nk*128) | k (nk*128) | v (nv*128)]; ab: [M][2*nv] = [decay | beta].
// grid: nv*128 work-items. The state is read from S and written to Sout (the same buffer: in place; a speculative
// verification pass writes elsewhere and keeps S for the replay of its accepted tokens).
// (-DNO_SG8, GPUs without 8-lane sub-groups (Xe2 / Xe3): 16 lanes, the same arithmetic in the same order; their
//  64-byte registers hold the 128-float column as Xe-LPG's 32-byte ones do at 8 lanes)
#ifdef NO_SG8
#define GW 16
#else
#define GW 8
#endif
#define BCW(x, i) intel_sub_group_shuffle((x), (i))
__attribute__((intel_reqd_sub_group_size(GW)))
__kernel void gdn_recur(__global const float* x, int C, __global const float* ab, int nk, int nv,
                        __global const float* S, int M_valid, __global float* o, __global float* Sout) {
  const int col = get_global_id(0), lane = get_sub_group_local_id();
  const int h = col / 128, j = col % 128, kh = h % nk, jb = j - lane;
  // the first token's operands are requested before the state, so their latency hides under the state loads
  const int qo = kh * 128, ko = nk * 128 + kh * 128, vo = 2 * nk * 128 + h * 128 + jb;
  uint8 kn0 = 0, kn1 = 0, qn0 = 0, qn1 = 0;
  uint vn = 0;
  float dn = 0.0f, bn = 0.0f;
  if (M_valid > 0) {
    kn0 = intel_sub_group_block_read8((__global const uint*)(x + ko));
    qn0 = intel_sub_group_block_read8((__global const uint*)(x + qo));
#if GW == 8
    kn1 = intel_sub_group_block_read8((__global const uint*)(x + ko + 64));
    qn1 = intel_sub_group_block_read8((__global const uint*)(x + qo + 64));
#endif
    vn = intel_sub_group_block_read((__global const uint*)(x + vo));
    dn = ab[h];
    bn = ab[nv + h];
  }
  float s[128];
#ifdef GDN_LINE_SWAP
  // lanes 2c and 2c+1 move the two 32-byte halves of one 64-byte line (message 0: the even lane's row, message 1:
  // the odd lane's row) and swap 8 floats with each other
  const int hf = lane & 1;
  const float8 hm = (float8)(as_float(-(uint)hf));  // all ones on odd lanes: bitselect(a, b, hm) = hf ? b : a
  __global const float* sp0 = S + ((size_t)h * 128 + (j & ~1)) * 128 + 8 * hf;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const float8 e0 = vload8(0, sp0 + 16 * q), e1 = vload8(0, sp0 + 128 + 16 * q);
    const float8 r = intel_sub_group_shuffle_xor(bitselect(e1, e0, hm), 1);
    const float8 lo = bitselect(e0, r, hm), hi = bitselect(r, e1, hm);
    s[16 * q] = lo.s0; s[16 * q + 1] = lo.s1; s[16 * q + 2] = lo.s2; s[16 * q + 3] = lo.s3;
    s[16 * q + 4] = lo.s4; s[16 * q + 5] = lo.s5; s[16 * q + 6] = lo.s6; s[16 * q + 7] = lo.s7;
    s[16 * q + 8] = hi.s0; s[16 * q + 9] = hi.s1; s[16 * q + 10] = hi.s2; s[16 * q + 11] = hi.s3;
    s[16 * q + 12] = hi.s4; s[16 * q + 13] = hi.s5; s[16 * q + 14] = hi.s6; s[16 * q + 15] = hi.s7;
  }
#else
  // 32 bytes per lane and message (vload16 = two d32x8 messages): all 16 state loads fit the 16 send tokens
  __global const float* sp = S + ((size_t)h * 128 + j) * 128;
#pragma unroll
  for (int i = 0; i < 128; i += 16) {
    const float16 v16 = vload16(i / 16, sp);
    s[i] = v16.s0; s[i + 1] = v16.s1; s[i + 2] = v16.s2; s[i + 3] = v16.s3;
    s[i + 4] = v16.s4; s[i + 5] = v16.s5; s[i + 6] = v16.s6; s[i + 7] = v16.s7;
    s[i + 8] = v16.s8; s[i + 9] = v16.s9; s[i + 10] = v16.sa; s[i + 11] = v16.sb;
    s[i + 12] = v16.sc; s[i + 13] = v16.sd; s[i + 14] = v16.se; s[i + 15] = v16.sf;
  }
#endif
  for (int t = 0; t < M_valid; ++t) {
    // (element i of q / k: register i / GW of lane i % GW, as the sub-group block reads lay them out)
    const float kr[16] = {as_float(kn0.s0), as_float(kn0.s1), as_float(kn0.s2), as_float(kn0.s3),
                          as_float(kn0.s4), as_float(kn0.s5), as_float(kn0.s6), as_float(kn0.s7),
                          as_float(kn1.s0), as_float(kn1.s1), as_float(kn1.s2), as_float(kn1.s3),
                          as_float(kn1.s4), as_float(kn1.s5), as_float(kn1.s6), as_float(kn1.s7)};
    const float qr[16] = {as_float(qn0.s0), as_float(qn0.s1), as_float(qn0.s2), as_float(qn0.s3),
                          as_float(qn0.s4), as_float(qn0.s5), as_float(qn0.s6), as_float(qn0.s7),
                          as_float(qn1.s0), as_float(qn1.s1), as_float(qn1.s2), as_float(qn1.s3),
                          as_float(qn1.s4), as_float(qn1.s5), as_float(qn1.s6), as_float(qn1.s7)};
    const float v = as_float(vn), decay = dn, beta = bn;
    if (t + 1 < M_valid) {
      __global const float* nrow = x + (size_t)(t + 1) * C;
      kn0 = intel_sub_group_block_read8((__global const uint*)(nrow + ko));
      qn0 = intel_sub_group_block_read8((__global const uint*)(nrow + qo));
#if GW == 8
      kn1 = intel_sub_group_block_read8((__global const uint*)(nrow + ko + 64));
      qn1 = intel_sub_group_block_read8((__global const uint*)(nrow + qo + 64));
#endif
      vn = intel_sub_group_block_read((__global const uint*)(nrow + vo));
      dn = ab[(size_t)(t + 1) * 2 * nv + h];
      bn = ab[(size_t)(t + 1) * 2 * nv + nv + h];
    }
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (int i = 0; i < 128; i += 4) {
      s[i] *= decay; s[i + 1] *= decay; s[i + 2] *= decay; s[i + 3] *= decay;
      a0 = fma(s[i], BCW(kr[i / GW], i % GW), a0);
      a1 = fma(s[i + 1], BCW(kr[(i + 1) / GW], (i + 1) % GW), a1);
      a2 = fma(s[i + 2], BCW(kr[(i + 2) / GW], (i + 2) % GW), a2);
      a3 = fma(s[i + 3], BCW(kr[(i + 3) / GW], (i + 3) % GW), a3);
    }
    const float d = (v - ((a0 + a1) + (a2 + a3))) * beta;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll
    for (int i = 0; i < 128; i += 4) {
      s[i] = fma(BCW(kr[i / GW], i % GW), d, s[i]);
      s[i + 1] = fma(BCW(kr[(i + 1) / GW], (i + 1) % GW), d, s[i + 1]);
      s[i + 2] = fma(BCW(kr[(i + 2) / GW], (i + 2) % GW), d, s[i + 2]);
      s[i + 3] = fma(BCW(kr[(i + 3) / GW], (i + 3) % GW), d, s[i + 3]);
      b0 = fma(s[i], BCW(qr[i / GW], i % GW), b0);
      b1 = fma(s[i + 1], BCW(qr[(i + 1) / GW], (i + 1) % GW), b1);
      b2 = fma(s[i + 2], BCW(qr[(i + 2) / GW], (i + 2) % GW), b2);
      b3 = fma(s[i + 3], BCW(qr[(i + 3) / GW], (i + 3) % GW), b3);
    }
    intel_sub_group_block_write((__global uint*)(o + (size_t)t * nv * 128 + h * 128 + jb),
                                as_uint((b0 + b1) + (b2 + b3)));
  }
#ifdef GDN_LINE_SWAP
  __global float* so0 = Sout + ((size_t)h * 128 + (j & ~1)) * 128 + 8 * hf;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const float8 lo = (float8)(s[16 * q], s[16 * q + 1], s[16 * q + 2], s[16 * q + 3], s[16 * q + 4], s[16 * q + 5],
                               s[16 * q + 6], s[16 * q + 7]);
    const float8 hi = (float8)(s[16 * q + 8], s[16 * q + 9], s[16 * q + 10], s[16 * q + 11], s[16 * q + 12],
                               s[16 * q + 13], s[16 * q + 14], s[16 * q + 15]);
    const float8 r = intel_sub_group_shuffle_xor(bitselect(hi, lo, hm), 1);
    vstore8(bitselect(lo, r, hm), 0, so0 + 16 * q);
    vstore8(bitselect(r, hi, hm), 0, so0 + 128 + 16 * q);
  }
#else
  __global float* so = Sout + ((size_t)h * 128 + j) * 128;
#pragma unroll
  for (int i = 0; i < 128; i += 16)
    vstore16((float16)(s[i], s[i + 1], s[i + 2], s[i + 3], s[i + 4], s[i + 5], s[i + 6], s[i + 7], s[i + 8], s[i + 9],
                       s[i + 10], s[i + 11], s[i + 12], s[i + 13], s[i + 14], s[i + 15]), i / 16, so);
#endif
}
#undef GW

// y = rmsnorm_head(o) * w * silu(z) per 128-dim head, quantized (K = nv*128). grid (256, M).
__attribute__((intel_reqd_sub_group_size(SG)))
#ifdef GATE_SIGMOID  // (Qwen3.8 Flash Next: the DeltaNet output gate is a sigmoid)
#define GDN_GATE(z) (1.0f / (1.0f + exp(-(z))))
#else
#define GDN_GATE(z) silu(z)
#endif
__kernel void gated_norm_quant(__global const float* o, __global const float* z, __global const float* w, int K,
                               int M, int M_valid, float eps, __global char* aq, __global float* asc,
                               __global ushort* absum, int act32) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  const int lane = get_sub_group_local_id();
  for (int s = get_sub_group_id(); s < K / 256; s += get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane, d0 = (16 * lane) & 127;
    float v[16];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      v[i] = valid ? o[(size_t)m * K + base + i] : 0.0f;
      ss += v[i] * v[i];
    }
    ss += intel_sub_group_shuffle_xor(ss, 1);
    ss += intel_sub_group_shuffle_xor(ss, 2);
    ss += intel_sub_group_shuffle_xor(ss, 4);
    const float r = rsqrt(ss / 128.0f + eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = valid ? v[i] * r * w[d0 + i] * GDN_GATE(z[(size_t)m * K + base + i]) : 0.0f;
    quant_store16(v, s, m, M, K, aq, asc, absum, act32);
  }
}


// Full-attention prep: per-head RMSNorm of q and k, NeoX RoPE on the first n_rot dims, q scaled by kq_scale.
// qkv row: [q|gate interleaved per head (nh*512) | k (nkv*256) | v (nkv*256)], row stride ldq.
// Writes q to qout [M][nh][256] and k, v (f16) to the cache [nkv][ctx][256] at pos0 + m. Also writes int8 copies of
// q (qi [M][nh][256], scale qsc [M][nh]) and k (kq [nkv][ctx][256], scale ksc [nkv][ctx]) for attn_q8.
// One sub-group per (head slot, token): slots [0, nh) are q heads, [nh, nh+nkv) k/v heads.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_prep(__global const float* qkv, int ldq, int nh, int nkv, __global const float* qn,
                        __global const float* kn, int M_valid, int pos0, float eps, float theta_scale, int n_rot,
                        float kq_scale, __global float* qout, __global half* kc, __global half* vc, int ctx,
                        __global char* qi, __global float* qsc, __global uint* kq, __global float* ksc,
                        __global uchar* vq, __global float* vsc, int f32_attn, int rope_shift,
                        __global const int* rpos, int use_rpos, int4 sect) {
  const int m = get_global_id(1), hs = get_global_id(0) / SG, lane = get_sub_group_local_id();
  if (m >= M_valid) return;
  const bool isq = hs < nh;
  __global const float* row = qkv + (size_t)m * ldq;
  __global const float* src = isq ? row + hs * 512 : row + nh * 512 + (hs - nh) * 256;
  __global const float* nw = isq ? qn : kn;
  float v[16];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    v[i] = src[16 * lane + i];
    ss += v[i] * v[i];
  }
  ss = sub_group_reduce_add(ss);
  const float r = rsqrt(ss / 256.0f + eps);
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] *= r * nw[16 * lane + i];

  // RoPE (n_rot = 64): pairs (e, e + n_rot/2) live in lanes l and l ^ 2 for l < 4. The rotary position is the cache
  // position plus rope_shift (text after images, whose tokens advance the rotary position by max(grid w, h) only);
  // with use_rpos the rows carry their own (t, h, w) positions (image tokens, Qwen3-VL interleaved M-RoPE: pair fi
  // takes h when fi % 3 == 1 and fi < 3 * sect.y, w when fi % 3 == 2 and fi < 3 * sect.z, else t)
  const int pos = pos0 + m, half_rot = n_rot / 2, rp_text = pos + rope_shift;
  float pv[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) pv[i] = intel_sub_group_shuffle_xor(v[i], 2);
  if (16 * lane < n_rot) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int e = 16 * lane + i, fi = e % half_rot;
      int rp = rp_text;
      if (use_rpos) {
        const int sector = fi % (sect.x + sect.y + sect.z + sect.w);
        const int comp = (sector % 3 == 1 && sector < 3 * sect.y) ? 1 : (sector % 3 == 2 && sector < 3 * sect.z) ? 2 : 0;
        rp = rpos[3 * m + comp];
      }
      const float theta = (float)rp * pow(theta_scale, (float)fi);
      const float c = cos(theta), sn = sin(theta);
      v[i] = e < half_rot ? v[i] * c - pv[i] * sn : v[i] * c + pv[i] * sn;
    }
  }
  if (isq)
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] *= kq_scale;
  const int kvh = hs - nh;
  __global const float* vs = row + nh * 512 + nkv * 256 + kvh * 256 + 16 * lane;
  if (f32_attn) {  // f32 q and f16 K/V rows for attn_fa
    if (isq) {
#pragma unroll
      for (int i = 0; i < 16; ++i) qout[((size_t)m * nh + hs) * 256 + 16 * lane + i] = v[i];
    } else {
      __global half* kd = kc + ((size_t)kvh * ctx + pos) * 256 + 16 * lane;
      __global half* vd = vc + ((size_t)kvh * ctx + pos) * 256 + 16 * lane;
#pragma unroll
      for (int i = 0; i < 16; ++i) { vstore_half(v[i], i, kd); vstore_half(vs[i], i, vd); }
    }
    return;
  }
  // int8 copy of the row (absmax scale per 64-dim block = 4 lanes)
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmax(amax, fabs(v[i]));
#pragma unroll
  for (int o = 1; o < 16 / SB; o <<= 1) amax = fmax(amax, intel_sub_group_shuffle_xor(amax, o));  // scale block
  float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  uint pk[4] = {0, 0, 0, 0};
#pragma unroll
  for (int i = 0; i < 16; ++i) pk[i / 4] |= (uint)(convert_int_rte(v[i] * inv) & 0xFF) << (8 * (i % 4));
  if (isq) {
    vstore4((uint4)(pk[0], pk[1], pk[2], pk[3]), 0, (__global uint*)(qi + ((size_t)m * nh + hs) * 256 + 16 * lane));
    if (lane % (16 / SB) == 0) qsc[((size_t)m * nh + hs) * SB + lane / (16 / SB)] = amax / 127.0f;
    return;
  }
  // K and V of this position go to 16-key blocks: K as [word 64][key 16] uints, V as [dim 256][key 16] bytes
  const size_t blk = (size_t)kvh * (ctx / 16) + pos / 16;
  const int kk = pos % 16;
#pragma unroll
  for (int c = 0; c < 4; ++c) kq[(blk * 64 + 4 * lane + c) * 16 + kk] = pk[c];
  float vv[16], vmax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) vv[i] = vs[i];
#pragma unroll
  for (int i = 0; i < 16; ++i) vmax = fmax(vmax, fabs(vv[i]));
#pragma unroll
  for (int o = 1; o < 16 / SB; o <<= 1) vmax = fmax(vmax, intel_sub_group_shuffle_xor(vmax, o));
  inv = vmax > 0.0f ? 127.0f / vmax : 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) vq[(blk * 256 + 16 * lane + i) * 16 + kk] = (uchar)convert_int_rte(vv[i] * inv);
  if (lane % (16 / SB) == 0) {  // scales [block][scale block][key]
    ksc[(blk * SB + lane / (16 / SB)) * 16 + kk] = amax / 127.0f;
    vsc[(blk * SB + lane / (16 / SB)) * 16 + kk] = vmax / 127.0f;
  }
}

// Flash-style causal attention. One sub-group per (group of GQ q heads sharing a kv head, token m).
// Keys go in blocks of 16: scores with lane = key (q broadcast from lanes), then P*V with lane = 16 head dims.
// Work-group = 16 consecutive tokens of one head group, so their K/V streams hit in L1.
// K/V rows beyond pos must be finite (the cache is zero-initialized); ctx must be a multiple of 16.
// GQ (q heads per sub-group) must divide nh/nkv. grid (16, nh/GQ, M).
#ifndef GQ
#define GQ 4
#endif
#define BC16(x, i) intel_sub_group_shuffle((x), (i))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_fa(__global const float* q, __global const half* kc, __global const half* vc, int ctx, int nh,
                      int nkv, int pos0, int M_valid, __global float* out) {
  const int lane = get_sub_group_local_id();
  const int hg = get_global_id(1), m = get_global_id(2);
  if (m >= M_valid) return;
  const int pos = pos0 + m, g = (hg * GQ) / (nh / nkv);
  __global const float* qp = q + ((size_t)m * nh + hg * GQ) * 256;
  __global const half* kb = kc + (size_t)g * ctx * 256;
  __global const half* vb = vc + (size_t)g * ctx * 256;
  float16 o[GQ];
  float mx[GQ], lp[GQ];
#pragma unroll
  for (int r = 0; r < GQ; ++r) { o[r] = 0.0f; mx[r] = -INFINITY; lp[r] = 0.0f; }

  const int nblk = pos / 16 + 1;
  for (int b = 0; b < nblk; ++b) {
    const int t0 = b * 16;
    float s[GQ];
#pragma unroll
    for (int r = 0; r < GQ; ++r) s[r] = 0.0f;
    __global const half* krow = kb + (size_t)(t0 + lane) * 256;
#pragma unroll 2
    for (int c = 0; c < 16; ++c) {
      const float16 kv = vload_half16(0, krow + 16 * c);
#pragma unroll
      for (int r = 0; r < GQ; ++r) {
        const float qc = as_float(intel_sub_group_block_read((__global const uint*)(qp + r * 256 + 16 * c)));
        float a = s[r];
        a = fma(BC16(qc, 0), kv.s0, a);  a = fma(BC16(qc, 1), kv.s1, a);
        a = fma(BC16(qc, 2), kv.s2, a);  a = fma(BC16(qc, 3), kv.s3, a);
        a = fma(BC16(qc, 4), kv.s4, a);  a = fma(BC16(qc, 5), kv.s5, a);
        a = fma(BC16(qc, 6), kv.s6, a);  a = fma(BC16(qc, 7), kv.s7, a);
        a = fma(BC16(qc, 8), kv.s8, a);  a = fma(BC16(qc, 9), kv.s9, a);
        a = fma(BC16(qc, 10), kv.sa, a); a = fma(BC16(qc, 11), kv.sb, a);
        a = fma(BC16(qc, 12), kv.sc, a); a = fma(BC16(qc, 13), kv.sd, a);
        a = fma(BC16(qc, 14), kv.se, a); a = fma(BC16(qc, 15), kv.sf, a);
        s[r] = a;
      }
    }
    const bool valid = t0 + lane <= pos;
#pragma unroll
    for (int r = 0; r < GQ; ++r) {
      const float sv = valid ? s[r] : -INFINITY;
      const float mn = fmax(mx[r], sub_group_reduce_max(sv));
      const float corr = exp(mx[r] - mn);
      const float p = valid ? exp(sv - mn) : 0.0f;
      lp[r] = lp[r] * corr + p;
      o[r] *= corr;
      mx[r] = mn;
      s[r] = p;
    }
#pragma unroll 2
    for (int j = 0; j < 16; ++j) {
      const float16 vv = vload_half16(0, vb + (size_t)(t0 + j) * 256 + 16 * lane);
#pragma unroll
      for (int r = 0; r < GQ; ++r) o[r] = fma((float16)sub_group_broadcast(s[r], j), vv, o[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < GQ; ++r) {
    const float L = sub_group_reduce_add(lp[r]);
    vstore16(o[r] / L, 0, out + ((size_t)m * nh + hg * GQ + r) * 256 + 16 * lane);
  }
}

// y = attn_out * sigmoid(gate), gate from the q|gate interleaved projection; quantized (K = nh*256).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_gate_quant(__global const float* att, __global const float* qkv, int ldq, int K, int M,
                              int M_valid, __global char* aq, __global float* asc, __global ushort* absum, int act32) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  const int lane = get_sub_group_local_id();
  for (int s = get_sub_group_id(); s < K / 256; s += get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane, h = base / 256, d = base % 256;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float g = valid ? qkv[(size_t)m * ldq + h * 512 + 256 + d + i] : 0.0f;
      v[i] = valid ? att[(size_t)m * K + base + i] / (1.0f + exp(-g)) : 0.0f;
    }
    quant_store16(v, s, m, M, K, aq, asc, absum, act32);
  }
}

// Causal attention, int8 throughout. Scores S = (q8 . k8) * qscale * kscale via DP4a, f32 online softmax, and
// P*V via DP4a as well: per 16-key tile, P * vscale (per key) is quantized to 8 bits with one scale per q row and
// dotted with the int8 V bytes (SageAttention-style block quantization of P). K and V live in 16-key blocks
// (K: [HD/4 words][16 keys] uints, V: [HD dims][16 keys] bytes; written by attn_prep*), so a tile is two
// contiguous copies of 4*HD uints into SLM shared by the whole work-group.
// HD = head dim (256 or 512). Optional sliding window `win` (a query at p sees keys p-win+1..p; 0 = unlimited)
// and cache capacity `cap` (positions; key position k lives in slot k % cap, so cap < context = ring buffer).
// Work-group = TPW consecutive tokens x NSP sub-groups; each sub-group handles HS of the HPG q heads that share
// the kv head, over all HD dims (lane l owns dims 16*i + l). grid (ceil(M/TPW) * 16*NSP*TPW, nkv),
// local (16*NSP*TPW, 1). Cache entries never written must be finite (zero-initialized).
#ifndef HPG
#define HPG 4
#endif
#ifndef HS
#define HS (HPG / 2)
#endif
#define NSP (HPG / HS)
#ifndef TPW
#define TPW 8
#endif
#ifndef HD
#define HD 256
#endif
// K/V tiles in SLM: a single buffer (two barriers per tile) leaves room for twice the work-groups per Xe-core of a
// double buffer (one barrier per tile, -DDOUBLE_BUF): HD=512 decode 1.5x faster, prefill the same or faster.
#ifdef DOUBLE_BUF
#define TILE_BUFS 2
#else
#define TILE_BUFS 1
#endif
#define KW (HD / 4)      // int8x4 words per key row
#define KC (HD / 64)     // q words per lane
#define DPL (HD / 16)    // output dims per lane
#define BLK_U (4 * HD)   // uints per 16-key block of K (and of V)

// Packs P (lane = key, values 0..255) of a 16-key tile into 4 words held by every lane: word w = keys 4w..4w+3.
inline uint4 pack_p(int q) {
  uint x = (uint)(q & 0xFF) << (8 * (get_sub_group_local_id() & 3));
  x |= intel_sub_group_shuffle_xor(x, 1);
  x |= intel_sub_group_shuffle_xor(x, 2);
  return (uint4)(intel_sub_group_shuffle(x, 0), intel_sub_group_shuffle(x, 4), intel_sub_group_shuffle(x, 8),
                 intel_sub_group_shuffle(x, 12));
}

#ifdef PRECISE_EXP
#define EXP2 exp2
#else
#define EXP2 native_exp2
#endif
// Scales: q [row][head][SB], K and V [cache block][SB][16 keys], SB scale blocks per row (-DSB): 1 = one per row
// (Qwen), KC = one per 64-dim block (Gemma 4, whose V rows have a few large dims that a per-row scale would let
// swamp the rest). The GEMM-like loops always go over the KC 64-dim chunks of a row.
// Scores of a 16-key tile for HS q heads (lane = key): int8 DP4a sums scaled by q and K scales -> sacc.
// KW_EXPR reads K word w of chunk c for this lane's key; sob = scale index of (tile, block 0, this lane).
#if SB == 1
#define QK_TILE(KW_EXPR, sob)                                                           \
  {                                                                                     \
    int acc[HS];                                                                        \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) acc[r] = 0;                          \
    _Pragma("unroll") for (int c = 0; c < KC; ++c)                                      \
      _Pragma("unroll") for (int w = 0; w < 16; ++w) {                                  \
        const uint kw = KW_EXPR;                                                        \
        _Pragma("unroll") for (int r = 0; r < HS; ++r)                                  \
          acc[r] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[r][c], w), kw);     \
      }                                                                                 \
    const float ks = ksc[sob];                                                          \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) sacc[r] = (float)acc[r] * qs[r][0] * ks; \
  }
#else
#define QK_TILE(KW_EXPR, sob)                                                           \
  _Pragma("unroll") for (int r = 0; r < HS; ++r) sacc[r] = 0.0f;                        \
  _Pragma("unroll") for (int c = 0; c < KC; ++c) {                                      \
    int acc[HS];                                                                        \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) acc[r] = 0;                          \
    _Pragma("unroll") for (int w = 0; w < 16; ++w) {                                    \
      const uint kw = KW_EXPR;                                                          \
      _Pragma("unroll") for (int r = 0; r < HS; ++r)                                    \
        acc[r] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[r][c], w), kw);       \
    }                                                                                   \
    const float ks = ksc[(sob) + c * 16];                                               \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) sacc[r] = fma((float)acc[r], qs[r][c] * ks, sacc[r]); \
  }
#endif

// Online-softmax step for one q head over a 16-key tile (lane = key). Scores are in the log2 domain (log2(e) is
// folded into the q scales). The accumulators are rescaled only when the running max grows (a uniform branch).
// Leaves pk = this key's weight relative to the running max and pb = exp2(tile max - running max) >= pk.
#define SOFTMAX_TILE(r, sv, valid)                                                      \
  {                                                                                     \
    const float tm = sub_group_reduce_max(sv);                                          \
    if (tm > mx[r]) {                                                                   \
      const float corr = EXP2(mx[r] - tm);                                              \
      lp[r] *= corr;                                                                    \
      _Pragma("unroll") for (int i = 0; i < DPL; ++i) o[r][i] *= corr;                  \
      mx[r] = tm;                                                                       \
    }                                                                                   \
    pk[r] = (valid) ? EXP2((sv) - mx[r]) : 0.0f;                                        \
    lp[r] += pk[r];                                                                     \
    pb[r] = EXP2(tm - mx[r]);                                                           \
  }

// P is quantized to 0..255 (unsigned x signed DP4a) or, with -DP7, to 0..127 (signed DP4a).
#ifdef P7
#define PQMAX 127.0f
#define PDOT(p, v) dot_4x8packed_ss_int(p, v)
#define PDOT_ACC(p, v, a) dot_acc_sat_4x8packed_ss_int(p, v, a)
#else
#define PQMAX 255.0f
#define PDOT(p, v) dot_4x8packed_us_int(p, v)
#define PDOT_ACC(p, v, a) dot_acc_sat_4x8packed_us_int(p, v, a)
#endif
// o[r][i] += P . V[dim 16*i + lane] for the HS heads; vt points at the tile's V block ([HD dims][4 words]).
// P * vscale(key, block) is quantized to 8 bits against the bound pb * max vscale of the tile and dotted with the
// int8 V bytes (SageAttention-style block quantization of P); with SB = KC once per 64-dim block (dims 16*i..,
// i in 4c..4c+3).
// The HS divisions run once (lane r for head r; the bounds are equal in every lane), and the dot products
// accumulate inside the DP4a with the heads interleaved (|sum| <= 16*255*128: never saturates). Same values;
// 4B prompt attention at 64K 936 -> 777 ms per layer, spill 576 -> 0 bytes (16 Oct).
#define PV_PACK(c, valid, sob)                                                          \
  const float vs = (valid) ? vsc[(sob) + (c) * 16] : 0.0f, vmt = sub_group_reduce_max(vs); \
  uint4 pw[HS];                                                                         \
  float ps[HS];                                                                         \
  {                                                                                     \
    float pmv = 0.0f;                                                                   \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) pmv = lane == r ? pb[r] * vmt : pmv; \
    const float invv = pmv > 0.0f ? PQMAX / pmv : 0.0f;                                 \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) {                                    \
      pw[r] = pack_p(convert_int_rte(pk[r] * vs * sub_group_broadcast(invv, r)));       \
      ps[r] = sub_group_broadcast(pmv, r) * (1.0f / PQMAX);                             \
    }                                                                                   \
  }
#define PV_DIM(vt, i)                                                                   \
  {                                                                                     \
    const uint4 v = vt[16 * (i) + lane];                                                \
    int a[HS];                                                                          \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) a[r] = PDOT(pw[r].x, v.x);           \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) a[r] = PDOT_ACC(pw[r].y, v.y, a[r]); \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) a[r] = PDOT_ACC(pw[r].z, v.z, a[r]); \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) a[r] = PDOT_ACC(pw[r].w, v.w, a[r]); \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) o[r][i] = fma((float)a[r], ps[r], o[r][i]);\
  }
#if SB == 1
#define PV_TILE(vt, sob, valid)                                                         \
  {                                                                                     \
    PV_PACK(0, valid, sob)                                                              \
    _Pragma("unroll") for (int i = 0; i < DPL; ++i) PV_DIM(vt, i)                       \
  }
#else
#define PV_TILE(vt, sob, valid)                                                         \
  _Pragma("unroll") for (int c = 0; c < KC; ++c) {                                      \
    PV_PACK(c, valid, sob)                                                              \
    _Pragma("unroll") for (int ii = 0; ii < 4; ++ii) PV_DIM(vt, 4 * c + ii)             \
  }
#endif

__attribute__((intel_reqd_sub_group_size(SG)))
#ifdef QSA  // Flash Next sparse attention: a query sees the keys of its selected blocks and of its tail block
#define QSA_ARGS , __global const uint* qmask, int qmw
// (block size QR; mask row `row`: bit b of word b / 32 = block b)
#define QSA_KEEP(row, qpos, key) ((key) >= ((qpos) + 1) / QR * QR || ((qmask[(size_t)(row) * qmw + ((key) / QR) / 32] >> (((key) / QR) & 31)) & 1u))
#else
#define QSA_ARGS
#endif
__kernel void attn_q8(__global const char* qi, __global const float* qsc, __global const uint* kq,
                      __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                      int nh, int pos0, int M_valid, __global float* out, int win, int nc_lo, int nc_hi QSA_ARGS) {
  // double-buffered tile: [K BLK_U uints | V BLK_U uints]; the next tile is fetched into registers while the current
  // one is processed. The per-key scales are read by every sub-group straight from the cache (L1): keeping them out
  // of SLM leaves an HD=512 tile pair at exactly 32 KB, so two work-groups fit an Xe-core's 64 KB.
  __local uint tile[TILE_BUFS][2 * BLK_U];
  const int lid = get_local_id(0), sg = lid / 16, lane = lid % 16, lsz = 16 * NSP * TPW;
  const int tl = sg / NSP, hh = sg % NSP, g = get_group_id(1);
  const int m = get_group_id(0) * TPW + tl;
  const bool active = m < M_valid;
  const int pos = pos0 + m, h0 = g * HPG + hh * HS;
  const int first = pos0 + (int)get_group_id(0) * TPW;
  const int last = pos0 + min(M_valid - 1, (int)get_group_id(0) * TPW + TPW - 1);
  const int lo = win > 0 ? pos - win + 1 : 0;  // first key this query sees
  // queries inside a non-causal span [nc_lo, nc_hi] (an image) also see the later keys of that span
  const int hi = (pos >= nc_lo && pos <= nc_hi) ? nc_hi : pos;

  uint qw[HS][KC];
  float qs[HS][SB];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    __global const uint* qp = (__global const uint*)(qi + ((size_t)(active ? m : 0) * nh + h0 + r) * HD);
#pragma unroll
    for (int c = 0; c < KC; ++c) qw[r][c] = active ? qp[c * 16 + lane] : 0u;
#pragma unroll
    for (int c = 0; c < SB; ++c) qs[r][c] = active ? qsc[((size_t)m * nh + h0 + r) * SB + c] * M_LOG2E_F : 0.0f;
  }
  float o[HS][DPL], mx[HS], lp[HS];
  float pk[HS], pb[HS], sacc[HS];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    mx[r] = -INFINITY;
    lp[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) o[r][i] = 0.0f;
  }

  const int nbc = cap / 16;                       // blocks in the cache
  const size_t blk0 = (size_t)g * nbc;
  const int last_hi = (nc_hi >= 0 && first <= nc_hi && last >= nc_lo) ? max(last, nc_hi) : last;
  const int b_first = win > 0 ? max(0, first - win + 1) / 16 : 0, b_last = last_hi / 16;
#define NL (2 * BLK_U / (16 * NSP * TPW))
  __global const uint* vqu = (__global const uint*)vq;
  uint rk[NL];
  {
    const size_t nb = (blk0 + b_first % nbc) * BLK_U;
#pragma unroll
    for (int j = 0; j < NL; ++j) rk[j] = j < NL / 2 ? kq[nb + lid + j * lsz] : vqu[nb + lid + (j - NL / 2) * lsz];
  }
  for (int b = b_first; b <= b_last; ++b) {
    const int t0 = b * 16;
    __local uint* tb = tile[b % TILE_BUFS];
    if (TILE_BUFS == 1 && b > b_first) barrier(CLK_LOCAL_MEM_FENCE);  // everyone is done with the previous tile
#pragma unroll
    for (int j = 0; j < NL; ++j) tb[lid + j * lsz] = rk[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (b < b_last) {
      const int bs = (b + 1) % nbc;
      const size_t nb = (blk0 + bs) * BLK_U;
#pragma unroll
      for (int j = 0; j < NL; ++j) rk[j] = j < NL / 2 ? kq[nb + lid + j * lsz] : vqu[nb + lid + (j - NL / 2) * lsz];
    }
    if (!active || t0 > hi || t0 + 15 < lo) continue;
    __local const uint* Kt = tb;
    __local const uint4* Vt = (__local const uint4*)(tb + BLK_U);
    const size_t sob = ((size_t)g * nbc + (size_t)(b % nbc)) * SB * 16 + lane;  // this tile's scales
    const int key = t0 + lane;
#ifdef QSA
    const bool valid = key <= hi && key >= lo && QSA_KEEP(m, pos, key);
    if (sub_group_all(!valid)) continue;  // (no key of the tile selected: nothing to add)
#else
    const bool valid = key <= hi && key >= lo;
#endif
#ifndef ABL_NOQK  // ablations (timing experiments only: wrong results)
#if HS == 4  // one address op per K load instead of two: 4 heads per sub-group spill 608 -> 0 bytes (3 heads: slower)
    __local const uint* KL = Kt + lane;
    QK_TILE(KL[(c * 16 + w) * 16], sob)
#else
    QK_TILE(Kt[(c * 16 + w) * 16 + lane], sob)
#endif
#else
#pragma unroll
    for (int r = 0; r < HS; ++r) sacc[r] = (float)(lane & 3) - (float)r + as_float(Kt[lane]) * 1e-30f;
#endif
#ifndef ABL_NOSM
#pragma unroll
    for (int r = 0; r < HS; ++r) {
      const float sv = valid ? sacc[r] : -INFINITY;
      SOFTMAX_TILE(r, sv, valid)
    }
#else
#pragma unroll
    for (int r = 0; r < HS; ++r) { pk[r] = valid ? sacc[r] * 0.01f + 0.5f : 0.0f; pb[r] = 1.0f; lp[r] += pk[r]; }
#endif
#ifndef ABL_NOPV
    PV_TILE(Vt, sob, valid)
#else
#pragma unroll
    for (int r = 0; r < HS; ++r) o[r][0] += pk[r] * as_float(Vt[lane].x) * 1e-30f;
#endif
  }
  if (active) {
#pragma unroll
    for (int r = 0; r < HS; ++r) {
      const float inv = 1.0f / sub_group_reduce_add(lp[r]);
      __global float* dst = out + ((size_t)m * nh + h0 + r) * HD + lane;
#pragma unroll
      for (int i = 0; i < DPL; ++i) dst[16 * i] = o[r][i] * inv;
    }
  }
}

// Prefill attention for long prompts with the head dims split over sub-groups. As attn_q8, a work-group covers TPW
// consecutive tokens of one kv head and shares every 16-key K/V tile through SLM; but a sub-group owns PTT tokens x
// PHG q heads (PRH rows) and only PDS = HD/PNS of the dims. It reads only its slice of each tile and holds q and the
// output of its dims only, so more rows fit in its registers and every byte read from SLM serves PRH rows (attn_q8:
// HS rows, SLM-bandwidth bound at long context; at HD 512 even HS = 2 spills). The partial scores of a row's PNS dim
// groups are summed through SLM (exact int sums with one scale per row: then the scores, the softmax and the output
// are bit-identical to attn_q8); every dim group runs the same online softmax and accumulates P*V for its dims.
// Full causal attention or a window (no non-causal spans). grid (ceil(M/TPW) * PLSZ, nkv), local (PLSZ, 1).
#ifdef PNS
#define PDS (HD / PNS)
#define PCS (PDS / 64)               // 64-dim chunks of a dim group
#define PNHG (HPG / PHG)             // head groups
#define PRH (PTT * PHG)              // (token, head) rows of a sub-group
#define PNRG (PNHG * (TPW / PTT))    // row groups of a work-group
#define PLSZ (16 * PNRG * PNS)       // work-items
#define PNL (2 * BLK_U / PLSZ)       // tile words fetched per work-item
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_ps(__global const char* qi, __global const float* qsc, __global const uint* kq,
                      __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                      int nh, int pos0, int M_valid, __global float* out, int win) {
  __local uint tile[2 * BLK_U];
  __local uint sp[PNRG][PNS][PRH][16];  // partial scores (int sums with SB = 1, else float bits)
  const int lid = get_local_id(0), sg = lid / 16, lane = lid % 16;
  const int dj = sg % PNS, rg = sg / PNS, hgi = rg % PNHG, tg = rg / PNHG, g = get_group_id(1);
  const int mb = (int)get_group_id(0) * TPW, h0 = g * HPG + hgi * PHG;
  // tokens of this sub-group: q rows mb + tg*PTT + t, positions pos0 + that, keys [tlo, thi]
  int tlo[PTT], thi[PTT];
  bool tact[PTT];
  int any_lo = INT_MAX, any_hi = -1;
#pragma unroll
  for (int t = 0; t < PTT; ++t) {
    const int m = mb + tg * PTT + t, pos = pos0 + m;
    tact[t] = m < M_valid;
    tlo[t] = win > 0 ? pos - win + 1 : 0;
    thi[t] = pos;
    if (tact[t]) {
      any_lo = min(any_lo, tlo[t]);
      any_hi = max(any_hi, thi[t]);
    }
  }
  uint qw[PRH][PCS];
  float qs[PRH][PCS];
#pragma unroll
  for (int x = 0; x < PRH; ++x) {
    const int t = x / PHG, r = x % PHG, m = tact[t] ? mb + tg * PTT + t : 0;
    __global const uint* qp = (__global const uint*)(qi + ((size_t)m * nh + h0 + r) * HD);
#pragma unroll
    for (int cc = 0; cc < PCS; ++cc) {
      const int c = dj * PCS + cc;
      qw[x][cc] = tact[t] ? qp[c * 16 + lane] : 0u;
      qs[x][cc] = tact[t] ? qsc[((size_t)m * nh + h0 + r) * SB + (SB == 1 ? 0 : c)] * M_LOG2E_F : 0.0f;
    }
  }
  float o[PRH][4 * PCS], mx[PRH], lp[PRH], pk[PRH], pb[PRH];
#pragma unroll
  for (int x = 0; x < PRH; ++x) {
    mx[x] = -INFINITY;
    lp[x] = 0.0f;
#pragma unroll
    for (int i = 0; i < 4 * PCS; ++i) o[x][i] = 0.0f;
  }

  const int nbc = cap / 16;
  const size_t blk0 = (size_t)g * nbc;
  const int first = pos0 + mb, last = pos0 + min(M_valid - 1, mb + TPW - 1);
  const int b_first = win > 0 ? max(0, first - win + 1) / 16 : 0, b_last = last / 16;
  __global const uint* vqu = (__global const uint*)vq;
  uint rk[PNL];
  {
    const size_t nb = (blk0 + b_first % nbc) * BLK_U;
#pragma unroll
    for (int j = 0; j < PNL; ++j) rk[j] = j < PNL / 2 ? kq[nb + lid + j * PLSZ] : vqu[nb + lid + (j - PNL / 2) * PLSZ];
  }
  __local const uint4* Vt = (__local const uint4*)(tile + BLK_U);
  for (int b = b_first; b <= b_last; ++b) {
    const int t0 = b * 16;
    if (b > b_first) barrier(CLK_LOCAL_MEM_FENCE);  // everyone is done with the previous tile and partial scores
#pragma unroll
    for (int j = 0; j < PNL; ++j) tile[lid + j * PLSZ] = rk[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (b < b_last) {
      const size_t nb = (blk0 + (b + 1) % nbc) * BLK_U;
#pragma unroll
      for (int j = 0; j < PNL; ++j) rk[j] = j < PNL / 2 ? kq[nb + lid + j * PLSZ] : vqu[nb + lid + (j - PNL / 2) * PLSZ];
    }
    // the same for every dim group of a row group: they write and read the partial scores together
    const bool work = any_hi >= t0 && any_lo <= t0 + 15;
    const size_t sob = (blk0 + (size_t)(b % nbc)) * SB * 16 + lane;  // this tile's scales
    const int key = t0 + lane;
    if (work) {
#if SB == 1
      int ia[PRH];
#pragma unroll
      for (int x = 0; x < PRH; ++x) ia[x] = 0;
#pragma unroll
      for (int cc = 0; cc < PCS; ++cc)
#pragma unroll
        for (int w = 0; w < 16; ++w) {
          const uint kw = tile[((dj * PCS + cc) * 16 + w) * 16 + lane];
#pragma unroll
          for (int x = 0; x < PRH; ++x) ia[x] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[x][cc], w), kw);
        }
#pragma unroll
      for (int x = 0; x < PRH; ++x) sp[rg][dj][x][lane] = (uint)ia[x];
#else
      float fa[PRH];
#pragma unroll
      for (int x = 0; x < PRH; ++x) fa[x] = 0.0f;
#pragma unroll
      for (int cc = 0; cc < PCS; ++cc) {
        const int c = dj * PCS + cc;
        int acc[PRH];
#pragma unroll
        for (int x = 0; x < PRH; ++x) acc[x] = 0;
#pragma unroll
        for (int w = 0; w < 16; ++w) {
          const uint kw = tile[(c * 16 + w) * 16 + lane];
#pragma unroll
          for (int x = 0; x < PRH; ++x) acc[x] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[x][cc], w), kw);
        }
        const float ks = ksc[sob + c * 16];
#pragma unroll
        for (int x = 0; x < PRH; ++x) fa[x] = fma((float)acc[x], qs[x][cc] * ks, fa[x]);
      }
#pragma unroll
      for (int x = 0; x < PRH; ++x) sp[rg][dj][x][lane] = as_uint(fa[x]);
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!work) continue;
    // full scores and the online softmax of every row that sees a key of this tile (the others get P = 0)
#pragma unroll
    for (int x = 0; x < PRH; ++x) {
      const int t = x / PHG;
      pk[x] = 0.0f;
      pb[x] = 0.0f;
      if (tact[t] && t0 <= thi[t] && t0 + 15 >= tlo[t]) {
        const bool valid = key <= thi[t] && key >= tlo[t];
#if SB == 1
        int s = 0;
#pragma unroll
        for (int j = 0; j < PNS; ++j) s += (int)sp[rg][j][x][lane];
        const float sc = (float)s * qs[x][0] * ksc[sob];
#else
        float sc = 0.0f;
#pragma unroll
        for (int j = 0; j < PNS; ++j) sc += as_float(sp[rg][j][x][lane]);
#endif
        const float sv = valid ? sc : -INFINITY;
        const float tm = sub_group_reduce_max(sv);
        if (tm > mx[x]) {
          const float corr = EXP2(mx[x] - tm);
          lp[x] *= corr;
#pragma unroll
          for (int i = 0; i < 4 * PCS; ++i) o[x][i] *= corr;
          mx[x] = tm;
        }
        pk[x] = valid ? EXP2(sv - mx[x]) : 0.0f;
        lp[x] += pk[x];
        pb[x] = EXP2(tm - mx[x]);
      }
    }
    // P * V for this sub-group's dims; P quantized against each token's bound as in PV_PACK
    uint4 pw[PRH];
    float pq[PRH];
#pragma unroll
    for (int cc = 0; cc < PCS; ++cc) {
      const int c = dj * PCS + cc;
      if (SB > 1 || cc == 0) {
        const float vsv = vsc[sob + (SB == 1 ? 0 : c * 16)];
        float vst[PTT], vmt[PTT];
#pragma unroll
        for (int t = 0; t < PTT; ++t) {
          vst[t] = key <= thi[t] && key >= tlo[t] ? vsv : 0.0f;
          vmt[t] = sub_group_reduce_max(vst[t]);
        }
#pragma unroll
        for (int x = 0; x < PRH; ++x) {
          const int t = x / PHG;
          const float pm = pb[x] * vmt[t];
          pw[x] = pack_p(convert_int_rte(pk[x] * vst[t] * (pm > 0.0f ? PQMAX / pm : 0.0f)));
          pq[x] = pm * (1.0f / PQMAX);
        }
      }
#pragma unroll
      for (int ii = 0; ii < 4; ++ii) {
        const uint4 v = Vt[16 * (4 * c + ii) + lane];
#pragma unroll
        for (int x = 0; x < PRH; ++x) {
          int a = PDOT(pw[x].x, v.x);
          a += PDOT(pw[x].y, v.y);
          a += PDOT(pw[x].z, v.z);
          a += PDOT(pw[x].w, v.w);
          o[x][4 * cc + ii] = fma((float)a, pq[x], o[x][4 * cc + ii]);
        }
      }
    }
  }
#pragma unroll
  for (int x = 0; x < PRH; ++x) {
    const int t = x / PHG, r = x % PHG;
    if (tact[t]) {
      const float inv = 1.0f / sub_group_reduce_add(lp[x]);
      __global float* dst = out + ((size_t)(mb + tg * PTT + t) * nh + h0 + r) * HD + lane;
#pragma unroll
      for (int cc = 0; cc < PCS; ++cc)
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) dst[16 * (4 * (dj * PCS + cc) + ii)] = o[x][4 * cc + ii] * inv;
    }
  }
}
#endif

// Flash-decoding for a few query rows (single tokens, speculative verification) with the tile scheme of attn_q8
// (q rows qrow0 + m of qi):
// a work-group covers one key split of one kv head for all M_valid (<= TPW) rows x NSP head slices, so each K/V
// tile is read from memory once into SLM and used by every q head of every row. Row m (position pos0 + m) sees
// keys [max(lo(m), split start), min(pos0 + m, split end)]. Partial results (max, sum, unnormalized o) go to
// part[((m * nsplit + split) * nh + h) * (HD + 2)]; attn_combine merges each row's splits.
// grid (nsplit * 16*NSP*TPW, nkv), local (16*NSP*TPW, 1). kstart and slen: multiples of 16.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_split(__global const char* qi, __global const float* qsc, __global const uint* kq,
                         __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                         int nh, int pos0, int M_valid, __global float* part, int win, int kstart, int slen, int nsplit,
                         int qrow0) {
  __local uint tile[TILE_BUFS][2 * BLK_U];
  const int lid = get_local_id(0), sg = lid / 16, lane = lid % 16, lsz = 16 * NSP * TPW;
  const int m = sg / NSP, hh = sg % NSP, g = get_group_id(1), split = get_group_id(0);
  const bool active = m < M_valid;
  const int pos = pos0 + m, h0 = g * HPG + hh * HS;
  const int s0 = kstart + split * slen, s1 = s0 + slen - 1;  // this split's keys
  const int lo = max(s0, win > 0 ? pos - win + 1 : 0), hi = min(s1, pos);

  uint qw[HS][KC];
  float qs[HS][SB];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    __global const uint* qp = (__global const uint*)(qi + ((size_t)(qrow0 + (active ? m : 0)) * nh + h0 + r) * HD);
#pragma unroll
    for (int c = 0; c < KC; ++c) qw[r][c] = active ? qp[c * 16 + lane] : 0u;
#pragma unroll
    for (int c = 0; c < SB; ++c) qs[r][c] = active ? qsc[((size_t)(qrow0 + m) * nh + h0 + r) * SB + c] * M_LOG2E_F : 0.0f;
  }
  float o[HS][DPL], mx[HS], lp[HS];
  float pk[HS], pb[HS], sacc[HS];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    mx[r] = -INFINITY;
    lp[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) o[r][i] = 0.0f;
  }
  const int nbc = cap / 16;
  const size_t blk0 = (size_t)g * nbc;
  const int lo_all = max(s0, win > 0 ? pos0 - win + 1 : 0), hi_all = min(s1, pos0 + M_valid - 1);
  const int b_first = max(lo_all, 0) / 16, b_last = hi_all / 16;
  __global const uint* vqu = (__global const uint*)vq;
  uint rk[NL];
  if (b_first <= b_last) {
    const size_t nb = (blk0 + b_first % nbc) * BLK_U;
#pragma unroll
    for (int j = 0; j < NL; ++j) rk[j] = j < NL / 2 ? kq[nb + lid + j * lsz] : vqu[nb + lid + (j - NL / 2) * lsz];
  }
  for (int b = b_first; b <= b_last; ++b) {
    const int t0 = b * 16;
    __local uint* tb = tile[b % TILE_BUFS];
    if (TILE_BUFS == 1 && b > b_first) barrier(CLK_LOCAL_MEM_FENCE);  // everyone is done with the previous tile
#pragma unroll
    for (int j = 0; j < NL; ++j) tb[lid + j * lsz] = rk[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (b < b_last) {
      const int bs = (b + 1) % nbc;
      const size_t nb = (blk0 + bs) * BLK_U;
#pragma unroll
      for (int j = 0; j < NL; ++j) rk[j] = j < NL / 2 ? kq[nb + lid + j * lsz] : vqu[nb + lid + (j - NL / 2) * lsz];
    }
    if (!active || t0 > hi || t0 + 15 < lo) continue;
    __local const uint* Kt = tb;
    __local const uint4* Vt = (__local const uint4*)(tb + BLK_U);
    const size_t sob = ((size_t)g * nbc + (size_t)(b % nbc)) * SB * 16 + lane;  // this tile's scales
    const int key = t0 + lane;
    const bool valid = key <= hi && key >= lo;
    QK_TILE(Kt[(c * 16 + w) * 16 + lane], sob)
#pragma unroll
    for (int r = 0; r < HS; ++r) {
      const float sv = valid ? sacc[r] : -INFINITY;
      SOFTMAX_TILE(r, sv, valid)
    }
    PV_TILE(Vt, sob, valid)  // (int8 V: attn_split is not used with --v4, see attn_decode_rows)
  }
  if (active) {
#pragma unroll
    for (int r = 0; r < HS; ++r) {
      __global float* dst = part + (((size_t)m * nsplit + split) * nh + h0 + r) * (HD + 2);
      const float L = sub_group_reduce_add(lp[r]);
      if (lane == 0) { dst[0] = mx[r]; dst[1] = L; }
#pragma unroll
      for (int i = 0; i < DPL; ++i) dst[2 + 16 * i + lane] = o[r][i];
    }
  }
}

// Single-token attention split over the key range (flash-decoding): one sub-group per (key split, kv head,
// head slice) computes, for HS q heads, the max score, softmax denominator and unnormalized P*V over its keys
// [kstart + split*slen, min(pos+1, kstart + (split+1)*slen)), restricted to the window, with the same int8 math
// as attn_q8. attn_combine merges the splits. grid (16 * nsplit, nkv * NSP), local (16, NSP): the head slices of a
// split share its K/V reads through L1.
// part: [nsplit][nh][HD + 2] = (max, sum, o[HD]). slen: multiple of 16; kstart: multiple of slen at or below the
// window start.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_decode(__global const char* qi, __global const float* qsc, __global const uint* kq,
                          __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                          int nh, int pos, __global float* part, int qrow, int win, int kstart, int slen QSA_ARGS
                         ) {
  const int lane = get_sub_group_local_id(), split = get_global_id(0) / SG;
  const int g = get_global_id(1) / NSP, h0 = g * HPG + (get_global_id(1) % NSP) * HS;
  const int lo = win > 0 ? max(0, pos - win + 1) : 0;
  const int k0 = max(kstart + split * slen, lo & ~15), k1 = min(pos + 1, kstart + (split + 1) * slen);
  uint qw[HS][KC];
  float qs[HS][SB];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    __global const uint* qp = (__global const uint*)(qi + ((size_t)qrow * nh + h0 + r) * HD);
#pragma unroll
    for (int c = 0; c < KC; ++c) qw[r][c] = qp[c * 16 + lane];
#pragma unroll
    for (int c = 0; c < SB; ++c) qs[r][c] = qsc[((size_t)qrow * nh + h0 + r) * SB + c] * M_LOG2E_F;
  }
  float o[HS][DPL], mx[HS], lp[HS];
  float pk[HS], pb[HS], sacc[HS];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    mx[r] = -INFINITY;
    lp[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) o[r][i] = 0.0f;
  }
  const int nbc = cap / 16;
  const size_t blk0 = (size_t)g * nbc;
  for (int t0 = k0; t0 < k1; t0 += 16) {
    const int bs = (t0 / 16) % nbc;
    const size_t b = blk0 + bs, sob = b * SB * 16 + lane;
    const int key = t0 + lane;
#ifdef QSA
    const bool valid = key < k1 && key >= lo && QSA_KEEP(qrow, pos, key);
    if (sub_group_all(!valid)) continue;  // (no key of the tile selected: its K/V are not read)
#else
    const bool valid = key < k1 && key >= lo;
#endif
    QK_TILE(intel_sub_group_block_read(kq + (b * KW + c * 16 + w) * 16), sob)
#pragma unroll
    for (int r = 0; r < HS; ++r) {
      const float sv = valid ? sacc[r] : -INFINITY;
      SOFTMAX_TILE(r, sv, valid)
    }
    __global const uint4* vt = vq + b * HD;
    PV_TILE(vt, sob, valid)
  }
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    __global float* dst = part + ((size_t)split * nh + h0 + r) * (HD + 2);
    const float L = sub_group_reduce_add(lp[r]);
    if (lane == 0) { dst[0] = mx[r]; dst[1] = L; }
#pragma unroll
    for (int i = 0; i < DPL; ++i) dst[2 + 16 * i + lane] = o[r][i];
  }
}

// Flash-decoding of up to TR query rows (single tokens, speculative verification) with the head dims split over the
// sub-groups of a work-group: a work-group covers one key split of one kv head; sub-group (head group hg, dim group
// dj) owns HG q heads of every row and the DS = HD/NSUB dims [dj*DS, (dj+1)*DS). Per 16-key tile each sub-group
// reads only its dims of K and V (every byte of the cache is read once for all rows and heads), computes partial
// scores, and the NSUB partials of a (row, head) are summed through SLM; the online softmax then runs in every dim
// group of the head group (same values, same order: identical results), and each sub-group accumulates P*V for its
// dims. Row m (position pos0 + m, q row qrow0 + m) sees keys [max(lo(m), split start), min(pos0 + m, split end)].
// Same int8 math as attn_decode. Partials go to part[((m * nsplit + split) * nh + h) * (HD + 2)] for attn_combine.
// Built with -DHG, -DNSUB (DS a multiple of 64) and -DTR. grid (nsplit * 16*NHG*NSUB, nkv), local (16*NHG*NSUB, 1).
// kstart and slen: multiples of 16.
#ifdef NSUB
#define DS (HD / NSUB)
#define CPS (DS / 64)
#define NHG (HPG / HG)
#ifndef TR
#define TR 1
#endif
#define RH (TR * HG)  // (row, head) pairs of a sub-group
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_dsplit(__global const char* qi, __global const float* qsc, __global const uint* kq,
                          __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                          int nh, int pos0, int M_valid, __global float* part, int win, int kstart, int slen,
                          int nsplit, int qrow0) {
  __local float sp[2][NHG][NSUB][RH][SG];
  const int lane = get_sub_group_local_id(), sg = get_sub_group_id(), hgi = sg / NSUB, dj = sg % NSUB;
  const int g = get_group_id(1), split = get_group_id(0), h0 = g * HPG + hgi * HG;
  const int s0 = kstart + split * slen, s1 = s0 + slen - 1;  // this split's keys
  const int lo_all = max(s0, win > 0 ? pos0 - win + 1 : 0), hi_all = min(s1, pos0 + M_valid - 1);
  uint qw[RH][CPS];
  float qs[RH][CPS];
#pragma unroll
  for (int x = 0; x < RH; ++x) {
    const int m = min(x / HG, M_valid - 1), r = x % HG;
    __global const uint* qp = (__global const uint*)(qi + ((size_t)(qrow0 + m) * nh + h0 + r) * HD);
#pragma unroll
    for (int cc = 0; cc < CPS; ++cc) {
      const int c = dj * CPS + cc;
      qw[x][cc] = qp[c * 16 + lane];
      qs[x][cc] = qsc[((size_t)(qrow0 + m) * nh + h0 + r) * SB + (SB == 1 ? 0 : c)] * M_LOG2E_F;
    }
  }
  float o[RH][4 * CPS], mx[RH], lp[RH], pk[RH], pb[RH];
#pragma unroll
  for (int x = 0; x < RH; ++x) {
    mx[x] = -INFINITY;
    lp[x] = 0.0f;
#pragma unroll
    for (int i = 0; i < 4 * CPS; ++i) o[x][i] = 0.0f;
  }
  const int nbc = cap / 16;
  const size_t blk0 = (size_t)g * nbc;
  int it = 0;
  for (int t0 = lo_all & ~15; t0 <= hi_all; t0 += 16, ++it) {
    const size_t b = blk0 + (t0 / 16) % nbc, sob = b * SB * 16 + lane;
    const int key = t0 + lane;
    // partial scores of this sub-group's dims
    float ps_[RH];
#pragma unroll
    for (int x = 0; x < RH; ++x) ps_[x] = 0.0f;
#pragma unroll
    for (int cc = 0; cc < CPS; ++cc) {
      const int c = dj * CPS + cc;
      int acc[RH];
#pragma unroll
      for (int x = 0; x < RH; ++x) acc[x] = 0;
#pragma unroll
      for (int w = 0; w < 16; ++w) {
        const uint kw = intel_sub_group_block_read(kq + (b * KW + c * 16 + w) * 16);
#pragma unroll
        for (int x = 0; x < RH; ++x) acc[x] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[x][cc], w), kw);
      }
      const float ks = ksc[sob + (SB == 1 ? 0 : c * 16)];
#pragma unroll
      for (int x = 0; x < RH; ++x) ps_[x] = fma((float)acc[x], qs[x][cc] * ks, ps_[x]);
    }
    __local float* spb = &sp[it & 1][hgi][0][0][0];
#pragma unroll
    for (int x = 0; x < RH; ++x) spb[(dj * RH + x) * SG + lane] = ps_[x];
    barrier(CLK_LOCAL_MEM_FENCE);
    // full scores and the online softmax of every (row, head) of the head group; a row with no key of this tile
    // (beyond its position, or before its window) is left as it is
#pragma unroll
    for (int x = 0; x < RH; ++x) {
      const int m = x / HG, pos = pos0 + m, lo = max(s0, win > 0 ? pos - win + 1 : 0), hi = min(s1, pos);
      pk[x] = 0.0f;
      pb[x] = 0.0f;
      if (m < M_valid && t0 <= hi && t0 + 15 >= lo) {
        const bool valid = key <= hi && key >= lo;
        float sv = 0.0f;
#pragma unroll
        for (int j = 0; j < NSUB; ++j) sv += spb[(j * RH + x) * SG + lane];
        sv = valid ? sv : -INFINITY;
        const float tm = sub_group_reduce_max(sv);
        if (tm > mx[x]) {
          const float corr = EXP2(mx[x] - tm);
          lp[x] *= corr;
#pragma unroll
          for (int i = 0; i < 4 * CPS; ++i) o[x][i] *= corr;
          mx[x] = tm;
        }
        pk[x] = valid ? EXP2(sv - mx[x]) : 0.0f;
        lp[x] += pk[x];
        pb[x] = EXP2(tm - mx[x]);
      }
    }
    // P * V for this sub-group's dims (P re-quantized per 64-dim block with SB > 1, as PV_PACK); keys outside every
    // row's range have pk = 0
    __global const uint4* vt = vq + b * HD;
#define DS_VSC(c) vsc[sob + (SB == 1 ? 0 : (c) * 16)]
#define DS_REPACK (SB > 1)
    uint4 pw[RH];
    float pq[RH];
#pragma unroll
    for (int cc = 0; cc < CPS; ++cc) {
      const int c = dj * CPS + cc;
      if (DS_REPACK || cc == 0) {
#ifdef ROWVMT
        // (-DROWVMT, Qwen: the bound of each row covers only the keys that row sees, so a row of a verification pass
        //  gets exactly the values of decoding its token alone: speculative decoding stays bit-identical)
        const float vraw = DS_VSC(c);
#pragma unroll
        for (int x = 0; x < RH; ++x) {
          const int m = x / HG, pos = pos0 + m, lo = max(s0, win > 0 ? pos - win + 1 : 0), hi = min(s1, pos);
          const float vs = m < M_valid && key >= lo && key <= hi ? vraw : 0.0f, vmt = sub_group_reduce_max(vs);
          const float pm = pb[x] * vmt;
          pw[x] = pack_p(convert_int_rte(pk[x] * vs * (pm > 0.0f ? PQMAX / pm : 0.0f)));
          pq[x] = pm * (1.0f / PQMAX);
        }
#else
        const bool used = key >= lo_all && key <= hi_all;  // the bound covers only keys some row sees
        const float vs = used ? DS_VSC(c) : 0.0f, vmt = sub_group_reduce_max(vs);
#pragma unroll
        for (int x = 0; x < RH; ++x) {
          const float pm = pb[x] * vmt;
          pw[x] = pack_p(convert_int_rte(pk[x] * vs * (pm > 0.0f ? PQMAX / pm : 0.0f)));
          pq[x] = pm * (1.0f / PQMAX);
        }
#endif
      }
#pragma unroll
      for (int ii = 0; ii < 4; ++ii) {
        const uint4 v = vt[64 * c + 16 * ii + lane];
#pragma unroll
        for (int x = 0; x < RH; ++x) {
          int a = PDOT(pw[x].x, v.x);
          a += PDOT(pw[x].y, v.y);
          a += PDOT(pw[x].z, v.z);
          a += PDOT(pw[x].w, v.w);
          o[x][4 * cc + ii] = fma((float)a, pq[x], o[x][4 * cc + ii]);
        }
      }
    }
  }
#undef DS_VSC
#undef DS_REPACK
#pragma unroll
  for (int x = 0; x < RH; ++x) {
    const int m = x / HG, r = x % HG;
    const float L = sub_group_reduce_add(lp[x]);
    if (m < M_valid) {
      __global float* dst = part + (((size_t)m * nsplit + split) * nh + h0 + r) * (HD + 2);
      if (dj == 0 && lane == 0) { dst[0] = mx[x]; dst[1] = L; }
#pragma unroll
      for (int cc = 0; cc < CPS; ++cc)
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) dst[2 + 64 * (dj * CPS + cc) + 16 * ii + lane] = o[x][4 * cc + ii];
    }
  }
}
#endif

// Merges the per-split partial results of attn_decode / attn_split (partials row prow) into output row orow.
// One work-item per (q head, output dim); grid (nh * hd), local 64 (hd: multiple of 64). The lanes of a sub-group
// take the split maxima and weights 16 at a time, then every lane sums its dim over the splits with the weights
// broadcast (independent loads, no data-dependent branches). Splits with no valid key (max = -inf) weigh 0.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_combine(__global const float* part, int nsplit, int nh, __global float* out, int orow, int hd,
                           int prow) {
  const int h = get_global_id(0) / hd, d = get_global_id(0) % hd, lane = get_sub_group_local_id();
  __global const float* p = part + ((size_t)prow * nsplit * nh + h) * (hd + 2);
  const size_t stride = (size_t)nh * (hd + 2);
  float M = -INFINITY;
  for (int s = lane; s < nsplit; s += SG) M = fmax(M, p[s * stride]);
  M = sub_group_reduce_max(M);
  float L = 0.0f, acc = 0.0f;
  for (int s0 = 0; s0 < nsplit; s0 += SG) {
    const int s = s0 + lane;
    const float m = s < nsplit ? p[s * stride] : -INFINITY;
    const float w = m == -INFINITY ? 0.0f : exp2(m - M);
    L = fma(s < nsplit ? p[s * stride + 1] : 0.0f, w, L);
    const int n = min(SG, nsplit - s0);
    __global const float* q = p + s0 * stride + 2 + d;
    if (n == SG) {
#pragma unroll
      for (int j = 0; j < SG; ++j) acc = fma(q[j * stride], sub_group_broadcast(w, j), acc);
    } else {
      for (int j = 0; j < n; ++j) acc = fma(q[j * stride], sub_group_broadcast(w, j), acc);
    }
  }
  out[((size_t)orow * nh + h) * hd + d] = acc / sub_group_reduce_add(L);
}

// MTP (nextn) block input: out[m] = [rmsnorm(e[m]) * ew | rmsnorm(h[m]) * hw] (f32, 2E wide). grid (256, n).
__kernel void mtp_concat(__global const float* e, __global const float* h, __global const float* ew,
                         __global const float* hw, int E, float eps, __global float* out) {
  const int m = get_group_id(1), l = get_local_id(0);
  __global const float* er = e + (size_t)m * E;
  __global const float* hr = h + (size_t)m * E;
  float se = 0.0f, sh = 0.0f;
  for (int i = l; i < E; i += get_local_size(0)) { se += er[i] * er[i]; sh += hr[i] * hr[i]; }
  se = work_group_reduce_add(se);
  sh = work_group_reduce_add(sh);
  const float re = rsqrt(se / E + eps), rh = rsqrt(sh / E + eps);
  __global float* o = out + (size_t)m * 2 * E;
  for (int i = l; i < E; i += get_local_size(0)) {
    o[i] = er[i] * re * ew[i];
    o[E + i] = hr[i] * rh * hw[i];
  }
}

// ---------------------------------------------------------------- Gemma 4
// RoPE angles of rows pos0 .. pos0 + M_valid - 1 for one layer type (head size hd, base, optional frequency
// factors): rt[m][i] = (cos, sin) of theta = pos * base^(-2i/hd) / ff[i], as attn_prep_g computes them, once per
// forward pass instead of in every layer. grid (hd / 2, M_valid).
__kernel void rope_table(__global float2* rt, int hd, int pos0, float theta_base, __global const float* ff,
                         int use_ff) {
  const int fi = get_global_id(0), m = get_global_id(1);
  float theta = (float)(pos0 + m) * pow(theta_base, -2.0f * (float)fi / (float)hd);
  if (use_ff) theta /= ff[fi];
  rt[(size_t)m * (hd / 2) + fi] = (float2)(cos(theta), sin(theta));
}

// n (a multiple of 8) consecutive floats into a register array with 32-byte vector loads: a lane's run of a row
// read as one access instead of n scalar ones (same values; the per-lane runs are 64-128 bytes apart, so scalar
// loads touched 16 cache lines per instruction)
#define LOADF(dst, p, n)                                                                  \
  _Pragma("unroll") for (int c_ = 0; c_ < (n) / 8; ++c_) {                              \
    const float8 t_ = vload8(0, (p) + 8 * c_);                                          \
    (dst)[8 * c_] = t_.s0; (dst)[8 * c_ + 1] = t_.s1; (dst)[8 * c_ + 2] = t_.s2;         \
    (dst)[8 * c_ + 3] = t_.s3; (dst)[8 * c_ + 4] = t_.s4; (dst)[8 * c_ + 5] = t_.s5;     \
    (dst)[8 * c_ + 6] = t_.s6; (dst)[8 * c_ + 7] = t_.s7;                               \
  }
// Attention inputs for one (token, head) per sub-group: RMS norm per head (weighted for q and k), NEOX RoPE over all
// HD dims (theta = pos * base^(-2i/HD) / ff[i] with optional frequency factors), V = RMS norm without weight of the
// V projection, or of the raw K projection when the layer has none (has_v = 0). int8 copies with a per-row scale:
// q -> qi/qsc, K/V -> the 16-key-block cache (slot = pos % cap). Lane l holds dims [VPL*l, VPL*l + VPL).
// qkv rows: [q nh*HD | k nkv*HD | v nkv*HD]. With rt (rope_table of these rows) the angles come from the table.
// grid (16*(nh+nkv), M), local (16, 1).
#define VPL (HD / 16)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_prep_g(__global const float* qkv, int ldq, int nh, int nkv, int has_v, __global const float* qn,
                          __global const float* kn, int M_valid, int pos0, float eps, float theta_base,
                          __global const float* ff, int use_ff, float q_scale, __global char* qi,
                          __global float* qsc, __global uint* kq, __global float* ksc, __global uchar* vq,
                          __global float* vsc, int cap, __global const float2* rt, int f16kv, __global float* qf,
                          __global KC_T* kc, __global half* vc) {
  const int m = get_global_id(1), hs = get_global_id(0) / SG, lane = get_sub_group_local_id();
  if (m >= M_valid) return;
  const bool isq = hs < nh;
  const int kvh = hs - nh, pos = pos0 + m;
  __global const float* row = qkv + (size_t)m * ldq;
  __global const float* src = isq ? row + (size_t)hs * HD : row + (size_t)(nh + kvh) * HD;
  float v[VPL];
  float ss = 0.0f;
  LOADF(v, src + VPL * lane, VPL)
#pragma unroll
  for (int i = 0; i < VPL; ++i) ss += v[i] * v[i];
  ss = sub_group_reduce_add(ss);
  float r = rsqrt(ss / HD + eps);
  __global const float* nw = isq ? qn : kn;
  // (norm weights, RoPE pairs and table in pieces of 8: whole-row copies spilled 4.6 KB per thread at HD 512)
#define PRP (VPL < 8 ? VPL : 8)
#pragma unroll
  for (int c0 = 0; c0 < VPL; c0 += PRP) {
    float nwv[PRP];
    LOADF(nwv, nw + VPL * lane + c0, PRP)
#pragma unroll
    for (int i = 0; i < PRP; ++i) v[c0 + i] *= r * nwv[i];
  }
  // NEOX RoPE: element e pairs with e +- HD/2, which lives in lane l ^ 8
  __global const float* rtl = (__global const float*)(rt + (size_t)m * (HD / 2) + (VPL * lane) % (HD / 2));
#pragma unroll
  for (int c0 = 0; c0 < VPL; c0 += PRP) {
    float pv[PRP], rtv[2 * PRP];  // this lane's (cos, sin) pairs: angles (VPL * lane) % (HD / 2) + c0 ..
#pragma unroll
    for (int i = 0; i < PRP; ++i) pv[i] = intel_sub_group_shuffle_xor(v[c0 + i], 8);
    if (rt) { LOADF(rtv, rtl + 2 * c0, 2 * PRP) }
#pragma unroll
    for (int i = 0; i < PRP; ++i) {
      const int e = VPL * lane + c0 + i, fi = e % (HD / 2);
      float c, sn;
      if (rt) {
        c = rtv[2 * i];
        sn = rtv[2 * i + 1];
      } else {
        float theta = (float)pos * pow(theta_base, -2.0f * (float)fi / (float)HD);
        if (use_ff) theta /= ff[fi];
        c = cos(theta);
        sn = sin(theta);
      }
      v[c0 + i] = e < HD / 2 ? v[c0 + i] * c - pv[i] * sn : v[c0 + i] * c + pv[i] * sn;
    }
  }
#undef PRP
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < VPL; ++i) {
    if (isq) v[i] *= q_scale;
    amax = fmax(amax, fabs(v[i]));
  }
#pragma unroll
  for (int o = 1; o < 16 / SB; o <<= 1) amax = fmax(amax, intel_sub_group_shuffle_xor(amax, o));  // scale block
  float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  uint pk[VPL / 4];
#pragma unroll
  for (int c = 0; c < VPL / 4; ++c) pk[c] = 0;
#pragma unroll
  for (int i = 0; i < VPL; ++i) pk[i / 4] |= (uint)(convert_int_rte(v[i] * inv) & 0xFF) << (8 * (i % 4));
  // (f16kv bit 0: the f32 query and the float K / V rows for attn_fa_g as well: kc / vc [kv head][cap][HD], slot =
  //  pos % cap)
  if (f16kv & 1) {
    if (isq) {
      __global float* qd = qf + ((size_t)m * nh + hs) * HD + VPL * lane;
#pragma unroll
      for (int i = 0; i < VPL; ++i) qd[i] = v[i];
    } else {
      __global KC_T* kd = kc + ((size_t)kvh * cap + pos % cap) * HD + VPL * lane;
#pragma unroll
      for (int i = 0; i < VPL; ++i) kd[i] = v[i];
    }
  }
  if (isq) {
    __global uint* dst = (__global uint*)(qi + ((size_t)m * nh + hs) * HD) + (VPL / 4) * lane;
#pragma unroll
    for (int c = 0; c < VPL / 4; ++c) dst[c] = pk[c];
    if (lane % (16 / SB) == 0) qsc[((size_t)m * nh + hs) * SB + lane / (16 / SB)] = amax / 127.0f;
    return;
  }
  const int slot = pos % cap, kk = slot % 16;
  const size_t blk = (size_t)kvh * (cap / 16) + slot / 16;
  // (f16kv bit 1: this layer keeps no int8 K / V copy, only the drafter's layers do)
  const bool i8 = !(f16kv & 2);
#pragma unroll
  for (int c = 0; c < VPL / 4; ++c)
    if (i8) kq[(blk * (HD / 4) + (VPL / 4) * lane + c) * 16 + kk] = pk[c];
  // V: the V projection, or the raw K projection, RMS-normalized without weight
  // (the raw K projection is read again from qkv, an input only: a register copy of it spilled at HD 512)
  float raw[VPL];
  if (has_v) {
    __global const float* vs = row + (size_t)(nh + nkv + kvh) * HD;
    ss = 0.0f;
    LOADF(raw, vs + VPL * lane, VPL)
#pragma unroll
    for (int i = 0; i < VPL; ++i) ss += raw[i] * raw[i];
    ss = sub_group_reduce_add(ss);
  } else {
    LOADF(raw, src + VPL * lane, VPL)
  }
  r = rsqrt(ss / HD + eps);
  float vmax = 0.0f;
#pragma unroll
  for (int i = 0; i < VPL; ++i) { raw[i] *= r; vmax = fmax(vmax, fabs(raw[i])); }
#pragma unroll
  for (int o = 1; o < 16 / SB; o <<= 1) vmax = fmax(vmax, intel_sub_group_shuffle_xor(vmax, o));
  inv = vmax > 0.0f ? 127.0f / vmax : 0.0f;
#pragma unroll
  for (int i = 0; i < VPL; ++i)
    if (i8) vq[(blk * HD + VPL * lane + i) * 16 + kk] = (uchar)convert_int_rte(raw[i] * inv);
  if (i8 && lane % (16 / SB) == 0) {  // scales [block][scale block][key]
    ksc[(blk * SB + lane / (16 / SB)) * 16 + kk] = amax / 127.0f;
    vsc[(blk * SB + lane / (16 / SB)) * 16 + kk] = vmax / 127.0f;
  }
  if (f16kv & 1) {
    __global half* vd = vc + ((size_t)kvh * cap + pos % cap) * HD + VPL * lane;
#pragma unroll
    for (int i = 0; i < VPL; ++i) vstore_half(raw[i], 0, vd + i);
  }
}

// Gemma 4 attention in float (two-pass mode, QUEUE F21.3): f32 queries (qf [M][nh][HD]) against f16 K / V rows
// (kc / vc [kv head][cap][HD], slot = position % cap, written by attn_prep_g), f32 scores, softmax and P.V; no
// 1/sqrt(d) (Gemma 4). Query position p0 + m sees the keys lo .. hi: lo = p - win + 1 (win > 0, sliding) or 0,
// hi = p, or nc_hi inside a non-causal span [nc_lo, nc_hi] (an image). Each row is computed alone (its result does
// not depend on the other rows of the launch): verification rows equal single tokens. One sub-group per (GQ q heads
// of one kv head, row); keys in blocks of 16, lane = key for the scores, lane = 16 dims per 256 for P.V.
// grid (16, nh / GQ, M), local (16, 1, 1).
// -DFA_ROWS=R (prompt chunks): a work-group of R consecutive rows, one sub-group each as above, loads every block of
// 16 keys and values once into local memory for all of them (each row read them all from global memory before:
// 4K-token prompt 14.7 s of attention); a row still takes exactly its own blocks with the same arithmetic, so the
// results are bit for bit those of R = 1. grid (16, nh / GQ, ceil(M / R) * R), local (16, 1, R).
#ifdef FA_G
#define FDC (HD / 256)  // 256-dim chunks of a row
// -DFA_SPLIT (generation / verification, <= 8 rows): flash-decoding. A row's key blocks are cut into splits of at
// least FA_BPS blocks, at most FA_SMAX of them, a function of that row's own range only (so a row gives the same
// result alone and in a verification launch); each split runs the loop below over its blocks and writes its
// unnormalized output, its max and its sum to part, and attn_fa_gc combines them. One row alone used to be 8
// sub-groups scanning every key: 26B at 14K context, 142 ms of attention per token.
#define FA_SMAX 64
#define FA_BPS 8
inline void fa_split_range(int lo, int hi, int sp, int* b0, int* b1) {  // blocks [b0, b1) of split sp (empty: b0 >= b1)
  const int nb = hi / 16 - lo / 16 + 1, ns = min(FA_SMAX, (nb + FA_BPS - 1) / FA_BPS), per = (nb + ns - 1) / ns;
  *b0 = lo / 16 + sp * per;
  *b1 = min(lo / 16 + (sp + 1) * per, hi / 16 + 1);
}
#ifndef FA_ROWS
#define FA_ROWS 1
#endif
#if FA_ROWS > 1
#define FA_KS (HD + 1)  // (a key's row stride in local memory: lane = key reads without bank conflicts)
#define FA_KW uint
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_fa_g(__global const float* qf, __global const KC_T* kc, __global const half* vc, int cap, int nh,
                        int nkv, int pos0, int M_valid, __global float* out, int win, int nc_lo, int nc_hi
#ifdef FA_SPLIT
                        , int sgrid, __global float* part
#endif
                        ) {
  const int lane = get_sub_group_local_id();
#ifdef FA_SPLIT
  const int hg = get_global_id(1), m = get_global_id(2) / sgrid, sp = get_global_id(2) % sgrid;
#else
  const int hg = get_global_id(1), m = get_global_id(2);
#endif
#if FA_ROWS > 1
  __local FA_KW sk[16 * FA_KS];
  __local ushort sv[16 * HD];
  const int mr = min(m, M_valid - 1);  // (rows past the launch take part in the loads; nothing is stored for them)
#else
  if (m >= M_valid) return;
  const int mr = m;
#endif
  const int pos = pos0 + mr, g = (hg * GQ) / (nh / nkv);
  const int lo = win > 0 ? max(0, pos - win + 1) : 0;
  const int hi = (pos >= nc_lo && pos <= nc_hi) ? nc_hi : pos;
  __global const float* qp = qf + ((size_t)mr * nh + hg * GQ) * HD;
  __global const KC_T* kb = kc + (size_t)g * cap * HD;
  __global const half* vb = vc + (size_t)g * cap * HD;
  float16 o[GQ][FDC];
  float mx[GQ], lp[GQ];
#pragma unroll
  for (int r = 0; r < GQ; ++r) {
    mx[r] = -INFINITY;
    lp[r] = 0.0f;
#pragma unroll
    for (int c = 0; c < FDC; ++c) o[r][c] = 0.0f;
  }
#if FA_ROWS > 1
  // the work-group's blocks: its first row's lo to its last row's hi (both grow with the position)
  const int m0 = get_group_id(2) * FA_ROWS, m1 = min(m0 + FA_ROWS, M_valid) - 1, q0 = pos0 + m0, q1 = pos0 + m1;
  const int blo = (win > 0 ? max(0, q0 - win + 1) : 0) / 16, bhi = ((q1 >= nc_lo && q1 <= nc_hi) ? nc_hi : q1) / 16;
  const int tid = get_local_id(2) * 16 + lane;
  for (int b = blo; b <= bhi; ++b) {
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = tid; i < 16 * HD; i += 16 * FA_ROWS) {
      const int kk = i / HD, d = i % HD;
      const size_t at = (size_t)((b * 16 + kk) % cap) * HD + d;
      sk[kk * FA_KS + d] = ((__global const FA_KW*)kb)[at];
      sv[kk * HD + d] = ((__global const ushort*)vb)[at];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (b < lo / 16 || b > hi / 16) continue;  // (uniform over the sub-group: its row's own blocks only)
#elif defined(FA_SPLIT)
  int bs0, bs1;
  fa_split_range(lo, hi, sp, &bs0, &bs1);
  if (bs0 >= bs1) return;  // (a split past this row's)
  for (int b = bs0; b < bs1; ++b) {
#else
  for (int b = lo / 16; b <= hi / 16; ++b) {
#endif
    const int t0 = b * 16, p = t0 + lane;
    const bool valid = p >= lo && p <= hi;
    float s[GQ];
#pragma unroll
    for (int r = 0; r < GQ; ++r) s[r] = 0.0f;
#if FA_ROWS > 1
    __local const KC_T* krow = (__local const KC_T*)sk + lane * FA_KS;
#else
    __global const KC_T* krow = kb + (size_t)(p % cap) * HD;
#endif
    for (int c = 0; c < HD / 16; ++c) {
      const float16 kv = valid ? vload16(0, krow + 16 * c) : (float16)0.0f;
#pragma unroll
      for (int r = 0; r < GQ; ++r) {
        const float qc = as_float(intel_sub_group_block_read((__global const uint*)(qp + r * HD + 16 * c)));
        float a = s[r];
        a = fma(BC16(qc, 0), kv.s0, a);  a = fma(BC16(qc, 1), kv.s1, a);
        a = fma(BC16(qc, 2), kv.s2, a);  a = fma(BC16(qc, 3), kv.s3, a);
        a = fma(BC16(qc, 4), kv.s4, a);  a = fma(BC16(qc, 5), kv.s5, a);
        a = fma(BC16(qc, 6), kv.s6, a);  a = fma(BC16(qc, 7), kv.s7, a);
        a = fma(BC16(qc, 8), kv.s8, a);  a = fma(BC16(qc, 9), kv.s9, a);
        a = fma(BC16(qc, 10), kv.sa, a); a = fma(BC16(qc, 11), kv.sb, a);
        a = fma(BC16(qc, 12), kv.sc, a); a = fma(BC16(qc, 13), kv.sd, a);
        a = fma(BC16(qc, 14), kv.se, a); a = fma(BC16(qc, 15), kv.sf, a);
        s[r] = a;
      }
    }
#pragma unroll
    for (int r = 0; r < GQ; ++r) {
      const float sv = valid ? s[r] : -INFINITY;
      const float mn = fmax(mx[r], sub_group_reduce_max(sv));
      const float corr = exp(mx[r] - mn);
      const float pr = valid ? exp(sv - mn) : 0.0f;
      lp[r] = lp[r] * corr + pr;
#pragma unroll
      for (int c = 0; c < FDC; ++c) o[r][c] *= corr;
      mx[r] = mn;
      s[r] = pr;
    }
    for (int j = 0; j < 16; ++j) {
      const int pj = t0 + j;
      if (pj < lo || pj > hi) continue;  // (the same for every lane)
#if FA_ROWS > 1
      __local const half* vrow = (__local const half*)sv + j * HD + 16 * lane;
#else
      __global const half* vrow = vb + (size_t)(pj % cap) * HD + 16 * lane;
#endif
#pragma unroll
      for (int c = 0; c < FDC; ++c) {
        const float16 vv = vload_half16(0, vrow + 256 * c);
#pragma unroll
        for (int r = 0; r < GQ; ++r) o[r][c] = fma((float16)sub_group_broadcast(s[r], j), vv, o[r][c]);
      }
    }
  }
  if (m >= M_valid) return;
#pragma unroll
  for (int r = 0; r < GQ; ++r) {
    const float L = sub_group_reduce_add(lp[r]);
#ifdef FA_SPLIT  // (this split's unnormalized output, then its sum and max)
    __global float* pp = part + (((size_t)m * nh + hg * GQ + r) * FA_SMAX + sp) * (HD + 2);
#pragma unroll
    for (int c = 0; c < FDC; ++c) vstore16(o[r][c], 0, pp + 256 * c + 16 * lane);
    if (lane == 0) {
      pp[HD] = L;
      pp[HD + 1] = mx[r];
    }
#else
#pragma unroll
    for (int c = 0; c < FDC; ++c)
      vstore16(o[r][c] / L, 0, out + ((size_t)m * nh + hg * GQ + r) * HD + 256 * c + 16 * lane);
#endif
  }
}
// The FA_SPLIT partials of each (row, head) combined: out = sum_s o_s e^(mx_s - mx) / sum_s L_s e^(mx_s - mx), the
// splits in order. grid (HD, nh, M).
__kernel void attn_fa_gc(__global const float* part, int nh, int pos0, int M_valid, __global float* out, int win,
                         int nc_lo, int nc_hi) {
  const int d = get_global_id(0), h = get_global_id(1), m = get_global_id(2);
  if (m >= M_valid) return;
  const int pos = pos0 + m;
  const int lo = win > 0 ? max(0, pos - win + 1) : 0;
  const int hi = (pos >= nc_lo && pos <= nc_hi) ? nc_hi : pos;
  __global const float* pp = part + ((size_t)m * nh + h) * FA_SMAX * (HD + 2);
  float mx = -INFINITY;
  for (int sp = 0; sp < FA_SMAX; ++sp) {
    int b0, b1;
    fa_split_range(lo, hi, sp, &b0, &b1);
    if (b0 >= b1) break;
    mx = fmax(mx, pp[(size_t)sp * (HD + 2) + HD + 1]);
  }
  float num = 0.0f, den = 0.0f;
  for (int sp = 0; sp < FA_SMAX; ++sp) {
    int b0, b1;
    fa_split_range(lo, hi, sp, &b0, &b1);
    if (b0 >= b1) break;
    __global const float* q = pp + (size_t)sp * (HD + 2);
    const float w = exp(q[HD + 1] - mx);
    num = fma(q[d], w, num);
    den = fma(q[HD], w, den);
  }
  out[((size_t)m * nh + h) * HD + d] = num / den;
}
#undef FDC
#endif  // FA_G

// attn_fa_g for prompt chunks of the wide-head layers (-DFA_GH, two-pass mode; Gemma 4's global layers, head 512, 8
// q heads per kv head): the same results bit for bit with each key row read once per row instead of once per pair of
// q heads. A work-group = one row x one kv head, FS sub-groups. Per step the FS sub-groups take FS consecutive blocks
// of 16 keys, one each, and score it for all HPG q heads (lane = key, each score the same fma chain over the dims as
// attn_fa_g), into local memory. Then every sub-group runs the blocks' softmax updates in order for all heads (the
// same operations, so the same values in each) and P.V for its HD / FS dims (lane = HD / FS / 16 dims). attn_fa_g
// holds HD / 16 output floats per lane per (row, head), which limited it to 2 heads per sub-group.
// grid (16 * FS, nkv, M), local (16 * FS, 1, 1).
#ifdef FA_GH
#define GD (HD / FS / 16)  // output dims per lane
#ifndef GR
#define GR 1  // query rows per work-group: each key / value row read once for GR rows x HPG heads
#endif
#define GRH (GR * HPG)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_fa_gh(__global const float* qf, __global const KC_T* kc, __global const half* vc, int cap, int nh,
                         int nkv, int pos0, int M_valid, __global float* out, int win, int nc_lo, int nc_hi) {
  __local float sc[FS][GRH][16];
  const int lane = get_sub_group_local_id(), sgi = get_sub_group_id();
  const int g = get_group_id(1), m0 = get_group_id(2) * GR;
  // each row's key range [lo, hi]; the work-group walks the blocks of their union
  int lor[GR], hir[GR];
#pragma unroll
  for (int r = 0; r < GR; ++r) {
    const int pos = pos0 + min(m0 + r, M_valid - 1);
    lor[r] = win > 0 ? max(0, pos - win + 1) : 0;
    hir[r] = (pos >= nc_lo && pos <= nc_hi) ? nc_hi : pos;
    if (m0 + r >= M_valid) lor[r] = 1, hir[r] = 0;  // (padding row: no keys)
  }
  int lo = lor[0], hi = hir[0];
#pragma unroll
  for (int r = 1; r < GR; ++r)
    if (hir[r] >= lor[r]) lo = min(lo, lor[r]), hi = max(hi, hir[r]);
  __global const KC_T* kb = kc + (size_t)g * cap * HD;
  __global const half* vb = vc + (size_t)g * cap * HD;
  float mx[GRH], lp[GRH];
#if GD == 4
  float4 o[GRH];
#elif GD == 8
  float8 o[GRH];
#else
  float16 o[GRH];
#endif
#pragma unroll
  for (int h = 0; h < GRH; ++h) {
    mx[h] = -INFINITY;
    lp[h] = 0.0f;
    o[h] = 0.0f;
  }
  const int b0 = lo / 16, b1 = hi / 16;
  for (int bs = b0; bs <= b1; bs += FS) {
    {  // scores of block bs + sgi
      const int b = bs + sgi, p = b * 16 + lane;
      const bool any = b <= b1 && p >= lo && p <= hi;
      __global const KC_T* krow = kb + (size_t)(p % cap) * HD;
      float s[GRH];
#pragma unroll
      for (int h = 0; h < GRH; ++h) s[h] = 0.0f;
      for (int c = 0; c < HD / 16; ++c) {
        const float16 kv = any ? vload16(0, krow + 16 * c) : (float16)0.0f;
#pragma unroll
        for (int h = 0; h < GRH; ++h) {
          const int r = h / HPG, m = min(m0 + r, M_valid - 1);
          const float qc = as_float(intel_sub_group_block_read((__global const uint*)(qf + ((size_t)m * nh + g * HPG + h % HPG) * HD + 16 * c)));
          float a = s[h];
          a = fma(BC16(qc, 0), kv.s0, a);  a = fma(BC16(qc, 1), kv.s1, a);
          a = fma(BC16(qc, 2), kv.s2, a);  a = fma(BC16(qc, 3), kv.s3, a);
          a = fma(BC16(qc, 4), kv.s4, a);  a = fma(BC16(qc, 5), kv.s5, a);
          a = fma(BC16(qc, 6), kv.s6, a);  a = fma(BC16(qc, 7), kv.s7, a);
          a = fma(BC16(qc, 8), kv.s8, a);  a = fma(BC16(qc, 9), kv.s9, a);
          a = fma(BC16(qc, 10), kv.sa, a); a = fma(BC16(qc, 11), kv.sb, a);
          a = fma(BC16(qc, 12), kv.sc, a); a = fma(BC16(qc, 13), kv.sd, a);
          a = fma(BC16(qc, 14), kv.se, a); a = fma(BC16(qc, 15), kv.sf, a);
          s[h] = a;
        }
      }
#pragma unroll
      for (int h = 0; h < GRH; ++h) {
        const int r = h / HPG;
        sc[sgi][h][lane] = b <= b1 && p >= lor[r] && p <= hir[r] ? s[h] : -INFINITY;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < FS && bs + i <= b1; ++i) {  // the blocks in order, as attn_fa_g takes them
      const int t0 = (bs + i) * 16, p = t0 + lane;
      float pr[GRH];
#pragma unroll
      for (int h = 0; h < GRH; ++h) {
        const int r = h / HPG;
        const bool valid = p >= lor[r] && p <= hir[r];
        const float sv = sc[i][h][lane];
        const float mn = fmax(mx[h], sub_group_reduce_max(sv));
        // (a block before a row's first key leaves it untouched: no exp(-inf + inf))
        const float corr = mn == -INFINITY ? 1.0f : exp(mx[h] - mn);
        pr[h] = valid ? exp(sv - mn) : 0.0f;
        lp[h] = lp[h] * corr + pr[h];
        o[h] *= corr;
        mx[h] = mn;
      }
      for (int j = 0; j < 16; ++j) {
        const int pj = t0 + j;
        if (pj < lo || pj > hi) continue;  // (the same for every lane)
        __global const half* vrow = vb + (size_t)(pj % cap) * HD + sgi * (HD / FS) + GD * lane;
#if GD == 4
        const float4 vv = vload_half4(0, vrow);
#pragma unroll
        for (int h = 0; h < GRH; ++h) o[h] = fma((float4)sub_group_broadcast(pr[h], j), vv, o[h]);
#elif GD == 8
        const float8 vv = vload_half8(0, vrow);
#pragma unroll
        for (int h = 0; h < GRH; ++h) o[h] = fma((float8)sub_group_broadcast(pr[h], j), vv, o[h]);
#else
        const float16 vv = vload_half16(0, vrow);
#pragma unroll
        for (int h = 0; h < GRH; ++h) o[h] = fma((float16)sub_group_broadcast(pr[h], j), vv, o[h]);
#endif
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#pragma unroll
  for (int h = 0; h < GRH; ++h) {
    const int m = m0 + h / HPG;
    const float L = sub_group_reduce_add(lp[h]);
    if (m < M_valid) {
      __global float* op = out + ((size_t)m * nh + g * HPG + h % HPG) * HD + sgi * (HD / FS) + GD * lane;
#if GD == 4
      vstore4(o[h] / L, 0, op);
#elif GD == 8
      vstore8(o[h] / L, 0, op);
#else
      vstore16(o[h] / L, 0, op);
#endif
    }
  }
}
#undef GD
#undef GRH
#endif  // FA_GH

// Single-token rows: x[m] = (x[m] + rmsnorm(t[m]) * w) * s (as add_rmsnorm), then the next GEMV input from the new
// x[m]: rmsnorm(x[m]) * w2 (or x[m] itself with w2_norm = 0) quantized like rmsnorm_quant. The new values stay in
// registers for the second step (sub-group j, lane l: elements 256 (j + 16 i) + 16 l .. +15; K <= 16384): reading
// them back from x could return the pre-update values from L1. grid (256, M_valid), local (256, 1).
#ifndef ANQ_MAX  // (superblocks per sub-group: the engine passes ceil(K / 256 / 16))
#define ANQ_MAX 4
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void add_norm_quant(__global float* x, __global const float* t, __global const float* w, int K, float eps,
                             float s, __global const float* w2, int w2_norm, int M, __global char* aq,
                             __global float* asc, __global ushort* absum, int act32) {
  const int m = get_group_id(1), lane = get_sub_group_local_id(), sg = get_sub_group_id(), nsg = get_num_sub_groups();
  __global const float* tr = t + (size_t)m * K;
  __global float* xr = x + (size_t)m * K;
  float ss = 0.0f;
  for (int i = get_local_id(0); i < K; i += get_local_size(0)) ss += tr[i] * tr[i];
  ss = work_group_reduce_add(ss);
  const float r = rsqrt(ss / K + eps);
  float xv[ANQ_MAX][16], s2 = 0.0f;
#pragma unroll
  for (int j = 0; j < ANQ_MAX; ++j) {
    const int sb = sg + j * nsg;
    if (sb < K / 256) {
      const int base = sb * 256 + 16 * lane;
      float xo[16], to[16], wo[16];
      LOADF(xo, xr + base, 16)
      LOADF(to, tr + base, 16)
      LOADF(wo, w + base, 16)
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float v = (xo[i] + to[i] * r * wo[i]) * s;
        xv[j][i] = v;
        s2 += v * v;
      }
      vstore8((float8)(xv[j][0], xv[j][1], xv[j][2], xv[j][3], xv[j][4], xv[j][5], xv[j][6], xv[j][7]), 0, xr + base);
      vstore8((float8)(xv[j][8], xv[j][9], xv[j][10], xv[j][11], xv[j][12], xv[j][13], xv[j][14], xv[j][15]), 0,
              xr + base + 8);
    }
  }
  s2 = work_group_reduce_add(s2);
  const float r2 = w2_norm ? rsqrt(s2 / K + eps) : 1.0f;
#pragma unroll
  for (int j = 0; j < ANQ_MAX; ++j) {
    const int sb = sg + j * nsg;
    if (sb < K / 256) {
      const int base = sb * 256 + 16 * lane;
      float v[16], w2v[16];
      if (w2_norm) { LOADF(w2v, w2 + base, 16) }
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = w2_norm ? xv[j][i] * r2 * w2v[i] : xv[j][i];
      quant_store16(v, sb, m, M, K, aq, asc, absum, act32);
    }
  }
}

// x[m] = (x[m] + rmsnorm(t[m]) * w) * s: Gemma's post-attention / post-FFN norm, residual add and layer output
// scale. grid (256, M_valid), local (256, 1).
__kernel void add_rmsnorm(__global float* x, __global const float* t, __global const float* w, int K, float eps,
                          float s) {
  const int m = get_group_id(1), l = get_local_id(0);
  __global const float* tr = t + (size_t)m * K;
  __global float* xr = x + (size_t)m * K;
  float ss = 0.0f;
  for (int i = l; i < K; i += get_local_size(0)) ss += tr[i] * tr[i];
  ss = work_group_reduce_add(ss);
  const float r = rsqrt(ss / K + eps);
  for (int i = l; i < K; i += get_local_size(0)) xr[i] = (xr[i] + tr[i] * r * w[i]) * s;
}

// Gemma 4 per-layer embeddings (E2B / E4B). pl rows hold the per-layer projection of the token embedding
// (n_layer groups of NPL values); tok rows the scaled per-layer token embeddings. In place:
// pl[g] = (rmsnorm(pl[g] * pre_scale) * w + tok[g]) * out_scale. grid (16 * n_layer, M), local (16, 1).
#define NPL 256
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ple_combine(__global float* pl, __global const float* tok, __global const float* w, int n_layer,
                          int M_valid, float pre_scale, float eps, float out_scale) {
  const int m = get_global_id(1), l = get_global_id(0) / SG, lane = get_sub_group_local_id();
  if (m >= M_valid) return;
  __global float* p = pl + ((size_t)m * n_layer + l) * NPL + lane * 16;
  __global const float* t = tok + ((size_t)m * n_layer + l) * NPL + lane * 16;
  float v[16], ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) { v[i] = p[i] * pre_scale; ss += v[i] * v[i]; }
  ss = sub_group_reduce_add(ss);
  const float r = rsqrt(ss / NPL + eps);
#pragma unroll
  for (int i = 0; i < 16; ++i) p[i] = (v[i] * r * w[lane * 16 + i] + t[i]) * out_scale;
}

// Per-layer input gate of layer `il`: v = gelu(g[m]) * pl[m][il], quantized for the following projection
// (K = NPL, one superblock). grid (16, M), local (16, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ple_gate_quant(__global const float* g, __global const float* pl, int il, int n_layer, int M,
                             int M_valid, __global char* aq, __global float* asc, __global ushort* absum, int act32) {
  const int m = get_global_id(1), lane = get_sub_group_local_id();
  float v[16];
  const bool valid = m < M_valid;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int e = lane * 16 + i;
    v[i] = valid ? gelu_tanh(g[(size_t)m * NPL + e]) * pl[((size_t)m * n_layer + il) * NPL + e] : 0.0f;
  }
  quant_store16(v, 0, m, M, NPL, aq, asc, absum, act32);
}

// Logit softcap of rows r of C (row stride ld, from column off): v = cap * tanh(v / cap), as the head GEMVs' HEAD_OUT
// (two-pass mode: after both passes). grid (ceil(N / 256) * 256, rows).
__kernel void softcap_rows(__global float* C, int ld, int off, int N, float cap) {
  const int n = get_global_id(0), r = get_global_id(1);
  if (n >= N) return;
  __global float* p = C + (size_t)r * ld + off + n;
  *p = cap * tanh(*p * (1.0f / cap));
}

// x[m] *= s (layer output scale applied after the per-layer-embedding block). grid (256, M_valid).
__kernel void scale_rows(__global float* x, int K, float s) {
  const int m = get_group_id(1);
  for (int i = get_local_id(0); i < K; i += get_local_size(0)) x[(size_t)m * K + i] *= s;
}

// ---------------------------------------------------------------- vision encoders (ViT on the GPU)
// Compiled only into the vision encoders' builds of this file (-DVIT): the language model never pays for them.
#ifdef VIT
// The matrix products go through the int8 GEMM (kernels/gemm.cl, Q8_0 weights, activations per 32); these kernels
// do what lies between them. Rows m >= M_valid are padding: quantized as zeros, not written otherwise.

// LayerNorm with weight and bias (two-pass mean / variance) of row m (E values, a multiple of 256), quantized for the
// next GEMM (quant != 0) and / or written as f32 to hout. grid (256, M), local (256, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void vit_ln_quant(__global const float* x, int E, __global const float* w, __global const float* b, float eps,
                           int M, int M_valid, __global char* aq, __global float* asc, __global ushort* absum,
                           int quant, __global float* hout, int write_h) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  __global const float* xr = x + (size_t)m * E;
  float s = 0.0f;
  if (valid)
    for (int i = get_local_id(0); i < E; i += get_local_size(0)) s += xr[i];
  const float mean = work_group_reduce_add(s) / E;
  float q = 0.0f;
  if (valid)
    for (int i = get_local_id(0); i < E; i += get_local_size(0)) q += (xr[i] - mean) * (xr[i] - mean);
  const float r = rsqrt(work_group_reduce_add(q) / E + eps);
  const int lane = get_sub_group_local_id();
  for (int sb = get_sub_group_id(); sb < E / 256; sb += get_num_sub_groups()) {
    const int base = sb * 256 + 16 * lane;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = valid ? (xr[base + i] - mean) * r * w[base + i] + b[base + i] : 0.0f;
    if (write_h && valid)
#pragma unroll
      for (int i = 0; i < 16; ++i) hout[(size_t)m * E + base + i] = v[i];
    if (quant) quant_store16(v, sb, m, M, E, aq, asc, absum, 1);
  }
}

// Row m of x (K values), plus bias (if has_b) and through gelu_tanh (if gelu), quantized for the next GEMM.
// grid (256, M), local (256, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void vit_quant(__global const float* x, int K, __global const float* bias, int has_b, int gelu, int M,
                        int M_valid, __global char* aq, __global float* asc, __global ushort* absum) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  const int lane = get_sub_group_local_id();
  __global const float* xr = x + (size_t)m * K;
  for (int sb = get_sub_group_id(); sb < K / 256; sb += get_num_sub_groups()) {
    const int base = sb * 256 + 16 * lane;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float t = valid ? xr[base + i] + (has_b ? bias[base + i] : 0.0f) : 0.0f;
      v[i] = gelu ? gelu_tanh(t) : t;
    }
    quant_store16(v, sb, m, M, K, aq, asc, absum, 1);
  }
}

// x[m][n] += a[n] (per_row 0: a bias) or a[m][n] (per_row 1), n < N. grid (N, M_valid).
__kernel void vit_add(__global float* x, int N, __global const float* a, int per_row) {
  const int n = get_global_id(0), m = get_global_id(1);
  x[(size_t)m * N + n] += a[per_row ? (size_t)m * N + n : n];
}

// Qwen3-VL vision attention input: qkv[m] = [q | k | v] (3E, E = nh * QD) plus its bias; q and k get the 2-D rotary
// positions of patch m (pos[m] = (row, column)): pair (i, i + QD/2) of each head turns by row * base^(-i/(QD/4)) for
// i < QD/4 and by column * base^(-(i - QD/4)/(QD/4)) after (Hugging Face Qwen3-VL, ggml GGML_ROPE_TYPE_VISION: two
// sections of QD/4). q is pre-scaled by 1/sqrt(QD). Output per head: [nh][Mp][QD] halves. QD = head size (-DQD, default
// 64; Qwen3.5 9B: 72). grid (nh * QD/2, M_valid).
#ifndef QD
#define QD 64
#endif
__kernel void vit_qkv_rope(__global const float* qkv, __global const float* bias, __global const int2* pos, int nh,
                           float log2_base, int Mp, __global half* q, __global half* k, __global half* v) {
  const int hf = QD / 2, qt = QD / 4;
  const int i = get_global_id(0) % hf, h = get_global_id(0) / hf, m = get_global_id(1), E = nh * QD;
  __global const float* row = qkv + (size_t)m * 3 * E + h * QD;
  __global const float* bb = bias + h * QD;
  const int2 p = pos[m];
  const float theta = (float)(i < qt ? p.x : p.y) * exp2(-log2_base * (float)(i % qt) / (float)qt);
  const float c = cos(theta), s = sin(theta), qs = rsqrt((float)QD);
  const size_t o = ((size_t)h * Mp + m) * QD;
  const float q0 = row[i] + bb[i], q1 = row[i + hf] + bb[i + hf];
  const float k0 = row[E + i] + bb[E + i], k1 = row[E + i + hf] + bb[E + i + hf];
  vstore_half((q0 * c - q1 * s) * qs, o + i, q);
  vstore_half((q0 * s + q1 * c) * qs, o + i + hf, q);
  vstore_half(k0 * c - k1 * s, o + i, k);
  vstore_half(k0 * s + k1 * c, o + i + hf, k);
  vstore_half(row[2 * E + i] + bb[2 * E + i], o + i, v);
  vstore_half(row[2 * E + i + hf] + bb[2 * E + i + hf], o + i + hf, v);
}

// LayerNorm with weight and bias of row m (any E) into out (f32). grid (256, M_valid), local (256, 1).
__kernel void vit_ln(__global const float* x, int E, __global const float* w, __global const float* b, float eps,
                     __global float* out) {
  const int m = get_group_id(1);
  __global const float* xr = x + (size_t)m * E;
  float s = 0.0f;
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) s += xr[i];
  const float mean = work_group_reduce_add(s) / E;
  float qv = 0.0f;
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) qv += (xr[i] - mean) * (xr[i] - mean);
  const float r = rsqrt(work_group_reduce_add(qv) / E + eps);
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) out[(size_t)m * E + i] = (xr[i] - mean) * r * w[i] + b[i];
}

// Vision-encoder GEMM in floating point: C[m][c_off + n] (+)= sum_k A[m][k] * W[n][k], A f32 [M][lda], W the
// mmproj's f16 weights [N][K], f32 accumulation. The ViT is far more sensitive to int8 activations than the
// language model (Qwen3-VL, photos: 10-23 % relative error with int8 GEMM inputs, 0.6-1.8 % in f16). Lane = column
// (16 per sub-group), VTM tokens per sub-group; the sub-groups of a work-group share the weight columns through L1.
// A token row is read as 16 floats per lane (one block read) and broadcast into the FMAs. grid (N, Mp / VTM),
// local (16, WGY); K a multiple of 16, rows m >= M_valid are computed but not stored.
#ifndef VTM
#define VTM 16
#endif
#ifdef NOSHUF  // (timing ablation: wrong results)
#define VSH(a, i) (a)
#else
#define VSH(a, i) intel_sub_group_shuffle((a), (i))
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void vit_gemm_f16(__global const float* A, int lda, __global const half* W, int K, __global float* C,
                           int ldc, int c_off, int acc, int M_valid) {
  const int n = get_global_id(0), m0 = get_global_id(1) * VTM;
  float r[VTM];
#pragma unroll
  for (int m = 0; m < VTM; ++m) r[m] = 0.0f;
  __global const half* wr = W + (size_t)n * K;
  for (int k0 = 0; k0 < K; k0 += 16) {
    const float16 w = vload_half16(0, wr + k0);
#pragma unroll
    for (int m = 0; m < VTM; ++m) {
      const float a = as_float(intel_sub_group_block_read((__global const uint*)(A + (size_t)(m0 + m) * lda + k0)));
      float s = r[m];
      s = fma(VSH(a, 0), w.s0, s);
      s = fma(VSH(a, 1), w.s1, s);
      s = fma(VSH(a, 2), w.s2, s);
      s = fma(VSH(a, 3), w.s3, s);
      s = fma(VSH(a, 4), w.s4, s);
      s = fma(VSH(a, 5), w.s5, s);
      s = fma(VSH(a, 6), w.s6, s);
      s = fma(VSH(a, 7), w.s7, s);
      s = fma(VSH(a, 8), w.s8, s);
      s = fma(VSH(a, 9), w.s9, s);
      s = fma(VSH(a, 10), w.sa, s);
      s = fma(VSH(a, 11), w.sb, s);
      s = fma(VSH(a, 12), w.sc, s);
      s = fma(VSH(a, 13), w.sd, s);
      s = fma(VSH(a, 14), w.se, s);
      s = fma(VSH(a, 15), w.sf, s);
      r[m] = s;
    }
  }
#pragma unroll
  for (int m = 0; m < VTM; ++m) {
    if (m0 + m >= M_valid) break;
    __global float* c = C + (size_t)(m0 + m) * ldc + c_off + n;
    *c = acc ? *c + r[m] : r[m];
  }
}


// x[m][n] = act(x[m][n] + b[n]) for the valid rows: the bias (and GELU tanh) after an f32 GEMM. grid (N, M_valid).
__kernel void vit_bias_act(__global float* x, int N, __global const float* b, int gelu) {
  const int n = get_global_id(0), m = get_global_id(1);
  float v = x[(size_t)m * N + n] + b[n];
  if (gelu) v = 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
  x[(size_t)m * N + n] = v;
}

// Vision attention in f32, lane = query row (SIMD8): each lane holds its row's q (64 values, pre-scaled by 1/8) and 64
// accumulators; a key's K and V rows (f16, 8 dims per lane) are read once per sub-group and broadcast into the FMAs,
// so the softmax needs no cross-lane reductions and no local memory. Tiles of FT keys, online softmax (natural
// exp, as vit_attn). Inputs as vit_qkv_rope writes them ([nh][Mp][64] f16). out[m][h*64 + d].
// grid (ceil(n / (8 * ARSG)) * 8 * ARSG, nh), local (8 * ARSG, 1); -cl-intel-256-GRF-per-thread.
// Tuned on the Arc iGPU (1008-token image, 24 layers): FT 16 -> 1118 ms (registers spill), FT 8 -> 704, FT 4 with
// native_exp -> 588 ms (2.7 TFLOPS; the error against the f32 reference unchanged at 0.2-0.6 %).
#ifndef FT
#define FT 4
#endif
// head size HD (default 64; Qwen3.5 9B's encoder: 72): lane l of the SIMD8 sub-group holds dims PL*l .. PL*l + PL-1
// (-DNO_SG8: 16 lanes, dims interleaved, ceil(HD / 16) per lane; the last block read may run past a row's HD values
//  into the next row or the buffer's slack, never used)
#ifndef HD
#define HD 64
#endif
#ifdef NO_SG8
#define VW 16
#else
#define VW 8
#endif
#define PL ((HD + VW - 1) / VW)
#if PL == 8 && VW == 8
#define KD(r, d) intel_sub_group_shuffle((r)[(d) % PL], (d) / PL)
#else
// other head sizes: interleaved, lane l holds dims l, l + VW, ... (one aligned sub-group block read per VW dims)
#define KD(r, d) intel_sub_group_shuffle((r)[(d) / VW], (d) % VW)
inline void kv_row(__global const half* p, float* r) {
#pragma unroll
  for (int i = 0; i < PL; ++i) r[i] = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(p + VW * i)));
}
#endif
#ifdef PRECISE_VEXP
#define AEXP exp
#else
#define AEXP native_exp
#endif
__attribute__((intel_reqd_sub_group_size(VW)))
__kernel void vit_attn_rows(__global const half* q, __global const half* k, __global const half* v, int n, int Mp,
                            int nh, __global float* out) {
  const int lane = get_sub_group_local_id(), h = get_group_id(1), m = get_global_id(0);
  const bool act = m < n;
  __global const half* qp = q + ((size_t)h * Mp + (act ? m : 0)) * HD;
#if PL == 8 && VW == 8
  float qr[HD];
#pragma unroll
  for (int d = 0; d < HD; ++d) qr[d] = vload_half(d, qp);
#endif
  // (other head sizes: q is read again per key from L1 instead of held: 72-dim q + accumulators spilled)
  float o[HD];
#pragma unroll
  for (int d = 0; d < HD; ++d) o[d] = 0.0f;
  float mx = -INFINITY, l = 0.0f;
  __global const half* kh = k + (size_t)h * Mp * HD;
  __global const half* vh = v + (size_t)h * Mp * HD;
  for (int j0 = 0; j0 < n; j0 += FT) {
    float s[FT], tm = -INFINITY;
#pragma unroll
    for (int jj = 0; jj < FT; ++jj) {
      const int key = min(j0 + jj, n - 1);
      float kr[PL];  // dims PL * lane ..
#if PL == 8 && VW == 8
      vstore8(vload_half8(lane, kh + (size_t)key * HD), 0, kr);
#else
      kv_row(kh + (size_t)key * HD, kr);
#endif
      float a0 = 0.0f, a1 = 0.0f;
#if PL == 8 && VW == 8
#pragma unroll
      for (int d = 0; d < HD; d += 2) {
        a0 = fma(qr[d], KD(kr, d), a0);
        a1 = fma(qr[d + 1], KD(kr, d + 1), a1);
      }
#else
#pragma unroll
      for (int c = 0; c < HD / 8; ++c) {
        const float8 qc = vload_half8(c, qp);
        a0 = fma(qc.s0, KD(kr, 8 * c), a0);
        a1 = fma(qc.s1, KD(kr, 8 * c + 1), a1);
        a0 = fma(qc.s2, KD(kr, 8 * c + 2), a0);
        a1 = fma(qc.s3, KD(kr, 8 * c + 3), a1);
        a0 = fma(qc.s4, KD(kr, 8 * c + 4), a0);
        a1 = fma(qc.s5, KD(kr, 8 * c + 5), a1);
        a0 = fma(qc.s6, KD(kr, 8 * c + 6), a0);
        a1 = fma(qc.s7, KD(kr, 8 * c + 7), a1);
      }
#endif
      s[jj] = j0 + jj < n ? a0 + a1 : -INFINITY;
      tm = fmax(tm, s[jj]);
    }
    if (tm > mx) {
      const float corr = AEXP(mx - tm);
      l *= corr;
#pragma unroll
      for (int d = 0; d < HD; ++d) o[d] *= corr;
      mx = tm;
    }
#pragma unroll
    for (int jj = 0; jj < FT; ++jj) {
      const float p = AEXP(s[jj] - mx);
      l += p;
      const int key = min(j0 + jj, n - 1);
      float vr[PL];
#if PL == 8 && VW == 8
      vstore8(vload_half8(lane, vh + (size_t)key * HD), 0, vr);
#else
      kv_row(vh + (size_t)key * HD, vr);
#endif
#ifndef NOPV
#pragma unroll
      for (int d = 0; d < HD; ++d) o[d] = fma(p, KD(vr, d), o[d]);
#else
      o[jj] += p * vr[jj % PL];
#endif
    }
  }
  if (act) {
    __global float* dst = out + (size_t)m * nh * HD + h * HD;
    const float inv = 1.0f / l;
#pragma unroll
    for (int d = 0; d < HD; ++d) dst[d] = o[d] * inv;
  }
}
// ---- Gemma 4 vision (mmproj gemma4v: the E2B / E4B encoder; zero/vision_gemma4v.h). RMS norms throughout, and
// "clippable" linears: each matrix product's input and output are clamped to ranges stored in the mmproj.

// out[m] = clamp(rms(x[m]) * w, lo, hi). grid (256, M_valid), local (256, 1).
__kernel void g4v_norm(__global const float* x, int ldx, int E, __global const float* w, float eps, float lo, float hi,
                       __global float* out, int ldo) {
  const int m = get_group_id(1);
  __global const float* xr = x + (size_t)m * ldx;
  float s = 0.0f;
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) s += xr[i] * xr[i];
  const float r = rsqrt(work_group_reduce_add(s) / E + eps);
  for (int i = get_local_id(0); i < E; i += get_local_size(0))
    out[(size_t)m * ldo + i] = clamp(xr[i] * r * w[i], lo, hi);
}

// Sandwich norm into the residual: x[m] += rms(clamp(y[m], lo, hi)) * w. grid (256, M_valid), local (256, 1).
__kernel void g4v_norm_add(__global const float* y, int E, __global const float* w, float eps, float lo, float hi,
                           __global float* x) {
  const int m = get_group_id(1);
  __global const float* yr = y + (size_t)m * E;
  float s = 0.0f;
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) {
    const float v = clamp(yr[i], lo, hi);
    s += v * v;
  }
  const float r = rsqrt(work_group_reduce_add(s) / E + eps);
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) x[(size_t)m * E + i] += clamp(yr[i], lo, hi) * r * w[i];
}

// x[m][n] = clamp(x[m][n], lo, hi) for n < N (row stride ld). grid (N, M_valid).
__kernel void g4v_clamp(__global float* x, int ld, float lo, float hi) {
  const int n = get_global_id(0), m = get_global_id(1);
  x[(size_t)m * ld + n] = clamp(x[(size_t)m * ld + n], lo, hi);
}

// Attention input: h[m] = [q | k | v] (3E, E = nh * 64), each clamped to its matrix's output range (cl: q lo, hi,
// k lo, hi, v lo, hi); per head an RMS norm (q, k with weights qn / kn, v without), then 2-D rotary positions in NEOX
// order: dims (i, i + 16), i < 16, turn by column * 100^(-i/16), dims (32 + i, 48 + i) by row * 100^(-i/16)
// (llama.cpp models/gemma4v.cpp). No 1/sqrt(64): Gemma 4's vision attention is unscaled. Output [nh][Mp][64] f16 as
// vit_attn_rows reads it. Lane l of a sub-group holds dims l, l + 16, l + 32, l + 48 of one head and row.
// grid (nh * 16, M_valid), local (16, 1).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void g4v_qkv(__global const float* h, int nh, __global const float* qn, __global const float* kn,
                      __global const int2* pos, float eps, float8 cl, float log2_base, int Mp, __global half* q,
                      __global half* k, __global half* v) {
  const int l = get_sub_group_local_id(), hh = get_global_id(0) / 16, m = get_global_id(1), E = nh * 64;
  const int2 p = pos[m];  // (column, row)
  const float f = exp2(-log2_base * (float)l / 16.0f), th1 = (float)p.x * f, th2 = (float)p.y * f;
  const float c1 = cos(th1), s1 = sin(th1), c2 = cos(th2), s2 = sin(th2);
  const size_t o = ((size_t)hh * Mp + m) * 64;
  const float los[3] = {cl.s0, cl.s2, cl.s4}, his[3] = {cl.s1, cl.s3, cl.s5};
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    __global const float* row = h + (size_t)m * 3 * E + t * E + hh * 64;
    float a0 = clamp(row[l], los[t], his[t]), a1 = clamp(row[l + 16], los[t], his[t]);
    float a2 = clamp(row[l + 32], los[t], his[t]), a3 = clamp(row[l + 48], los[t], his[t]);
    const float r = rsqrt(sub_group_reduce_add(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3) / 64.0f + eps);
    if (t < 2) {
      __global const float* nw = t == 0 ? qn : kn;
      a0 *= r * nw[l]; a1 *= r * nw[l + 16]; a2 *= r * nw[l + 32]; a3 *= r * nw[l + 48];
      const float b0 = a0 * c1 - a1 * s1, b1 = a0 * s1 + a1 * c1;
      const float b2 = a2 * c2 - a3 * s2, b3 = a2 * s2 + a3 * c2;
      a0 = b0; a1 = b1; a2 = b2; a3 = b3;
    } else {
      a0 *= r; a1 *= r; a2 *= r; a3 *= r;
    }
    __global half* dst = t == 0 ? q : t == 1 ? k : v;
    vstore_half(a0, o + l, dst);
    vstore_half(a1, o + l + 16, dst);
    vstore_half(a2, o + l + 32, dst);
    vstore_half(a3, o + l + 48, dst);
  }
}

// g4v_qkv for heads of G4V_HD dims (-DG4V_HD, a multiple of 4 up to 128; Gemma 4 26B-A4B: 72): the same values, lane l
// taking dims l, l + 16, ... of a head; the rotary partner of a dim (NEOX within each half: dims (i, i + G4V_HD / 4)) is
// read from the row again and normed with the same factor. Output [nh][Mp][G4V_HD] f16. grid (nh * 16, M_valid).
#ifndef G4V_HD
#define G4V_HD 64
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void g4v_qkv_hd(__global const float* h, int nh, __global const float* qn, __global const float* kn,
                         __global const int2* pos, float eps, float8 cl, float log2_base, int Mp, __global half* q,
                         __global half* k, __global half* v) {
  const int l = get_sub_group_local_id(), hh = get_global_id(0) / 16, m = get_global_id(1), E = nh * G4V_HD, Q = G4V_HD / 4;
  const int2 p = pos[m];  // (column, row)
  const size_t o = ((size_t)hh * Mp + m) * G4V_HD;
  const float los[3] = {cl.s0, cl.s2, cl.s4}, his[3] = {cl.s1, cl.s3, cl.s5};
  for (int t = 0; t < 3; ++t) {
    __global const float* row = h + (size_t)m * 3 * E + t * E + hh * G4V_HD;
    float ss = 0.0f;
    for (int d = l; d < G4V_HD; d += 16) {
      const float a = clamp(row[d], los[t], his[t]);
      ss += a * a;
    }
    const float r = rsqrt(sub_group_reduce_add(ss) / (float)G4V_HD + eps);
    __global half* dst = t == 0 ? q : t == 1 ? k : v;
    for (int d = l; d < G4V_HD; d += 16) {
      float a = clamp(row[d], los[t], his[t]) * r;
      if (t < 2) {
        __global const float* nw = t == 0 ? qn : kn;
        const int i = d % Q, second = d % (2 * Q) >= Q, pd = second ? d - Q : d + Q;
        const float b = clamp(row[pd], los[t], his[t]) * r * nw[pd];
        a *= nw[d];
        const float th = (float)(d < 2 * Q ? p.x : p.y) * exp2(-log2_base * (float)i / (float)Q);
        const float c = cos(th), s = sin(th);
        a = second ? b * s + a * c : a * c - b * s;
      }
      vstore_half(a, o + d, dst);
    }
  }
}

// x[t] = (x[t] - bias) * scale (Gemma 4 vision std_bias / std_scale, after the pooling). grid (E, n_tok).
__kernel void g4v_std(__global float* x, int E, __global const float* bias, __global const float* scale) {
  const int e = get_global_id(0), t = get_global_id(1);
  x[(size_t)t * E + e] = (x[(size_t)t * E + e] - bias[e]) * scale[e];
}

// FFN: out[m][n] = clamp(gelu_tanh(clamp(g, cl.s0, cl.s1)) * clamp(u, cl.s2, cl.s3), cl.s4, cl.s5) with
// h[m] = [gate | up] (2F); the last clamp is the down projection's input range. grid (F, M_valid).
__kernel void g4v_geglu(__global const float* h, int F, float8 cl, __global float* out) {
  const int n = get_global_id(0), m = get_global_id(1);
  const float g = clamp(h[(size_t)m * 2 * F + n], cl.s0, cl.s1), u = clamp(h[(size_t)m * 2 * F + F + n], cl.s2, cl.s3);
  const float ge = 0.5f * g * (1.0f + tanh(0.7978845608f * (g + 0.044715f * g * g * g)));
  out[(size_t)m * F + n] = clamp(ge * u, cl.s4, cl.s5);
}

// 3x3 (K x K) average pooling over the px x py patch grid (row-major patches), times sqrt(E):
// out[t][e], t over the (px / K) x (py / K) grid in row-major order. grid (E, n_tok).
__kernel void g4v_pool(__global const float* x, int E, int px, int K, float scale, __global float* out) {
  const int e = get_global_id(0), t = get_global_id(1), ox = px / K;
  const int ty = t / ox, tx = t % ox;
  float s = 0.0f;
  for (int dy = 0; dy < K; ++dy)
    for (int dx = 0; dx < K; ++dx) s += x[((size_t)(ty * K + dy) * px + tx * K + dx) * E + e];
  out[(size_t)t * E + e] = s / (float)(K * K) * scale;
}
// ---- Gemma 4 audio conformer (mmproj gemma4a: the E2B / E4B audio encoder; zero/audio_gemma4a.h)

// x[m][n] = clamp(silu(x[m][n]), lo, hi) for n < N. grid (N, M_valid).
__kernel void a4_silu(__global float* x, int N, float lo, float hi) {
  const int n = get_global_id(0), m = get_global_id(1);
  const float v = x[(size_t)m * N + n];
  x[(size_t)m * N + n] = clamp(v / (1.0f + exp(-v)), lo, hi);
}

// Half-step FFN residual: x[m] += s * rms(clamp(y[m], lo, hi)) * w. grid (256, M_valid), local (256, 1).
__kernel void a4_norm_axpy(__global const float* y, int E, __global const float* w, float eps, float lo, float hi,
                           float s, __global float* x) {
  const int m = get_group_id(1);
  __global const float* yr = y + (size_t)m * E;
  float ss = 0.0f;
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) {
    const float v = clamp(yr[i], lo, hi);
    ss += v * v;
  }
  const float r = rsqrt(work_group_reduce_add(ss) / E + eps);
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) x[(size_t)m * E + i] += s * clamp(yr[i], lo, hi) * r * w[i];
}

// Chunked local self-attention of the Gemma 4 audio conformer, written out per query: query t sees keys
// t - (W - 1) .. t (W = 12: llama.cpp's blocked mask allows 0 <= t - key < 12), score = q . k + q . p[12 - d]
// (d = t - key, p = the projected sinusoidal relative positions [13][E]), soft-capped (cap * tanh(s / cap)), softmax.
// qkv[m] = [q | k | v] (3E), each clamped to its output range (cl), q scaled by qs * pds[dim], k by ks.
// A sub-group per (head, query): lane l holds dims l + 16 j of the head (D = 128). out[m][h*D + d].
// grid (nh * 16, n), local (16, 1).
#define A4W 12
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void a4_attn(__global const float* qkv, int nh, int D, __global const float* p, __global const float* pds,
                      float qs, float ks, float cap, float8 cl, __global float* out) {
  const int l = get_sub_group_local_id(), h = get_global_id(0) / 16, t = get_global_id(1), E = nh * D, J = D / 16;
  float qv[8];
  for (int j = 0; j < J; ++j) {
    const int d = l + 16 * j;
    qv[j] = clamp(qkv[(size_t)t * 3 * E + h * D + d], cl.s0, cl.s1) * qs * pds[d];
  }
  float sc[A4W];
  float mx = -INFINITY;
  for (int w = 0; w < A4W; ++w) {
    const int key = t - w;  // d = w
    float s = 0.0f;
    if (key >= 0) {
      __global const float* kr = qkv + (size_t)key * 3 * E + E + h * D;
      __global const float* pr = p + (size_t)(12 - w) * E + h * D;
      for (int j = 0; j < J; ++j) {
        const int d = l + 16 * j;
        s += qv[j] * (clamp(kr[d], cl.s2, cl.s3) * ks + pr[d]);
      }
      s = sub_group_reduce_add(s);
      s = cap * tanh(s / cap);
      mx = fmax(mx, s);
    }
    sc[w] = s;
  }
  float den = 0.0f;
  for (int w = 0; w < A4W; ++w) {
    sc[w] = t - w >= 0 ? exp(sc[w] - mx) : 0.0f;
    den += sc[w];
  }
  for (int j = 0; j < J; ++j) {
    const int d = l + 16 * j;
    float o = 0.0f;
    for (int w = 0; w < A4W; ++w)
      if (t - w >= 0) o += sc[w] * clamp(qkv[(size_t)(t - w) * 3 * E + 2 * E + h * D + d], cl.s4, cl.s5);
    out[(size_t)t * E + h * D + d] = o / den;
  }
}

// Convolution module, after pointwise conv 1 (h[m] = [a | b], 2E, clamped to [lo, hi]): GLU a * sigmoid(b), then a
// causal depthwise conv of width K over time (w[c][j], j = K - 1 the current step). grid (E, n).
__kernel void a4_glu_dw(__global const float* h, int E, int K, __global const float* w, float lo, float hi,
                        __global float* out) {
  const int c = get_global_id(0), t = get_global_id(1);
  float s = 0.0f;
  for (int j = 0; j < K; ++j) {
    const int tt = t - (K - 1) + j;
    if (tt < 0) continue;
    const float a = clamp(h[(size_t)tt * 2 * E + c], lo, hi), b = clamp(h[(size_t)tt * 2 * E + E + c], lo, hi);
    s += w[(size_t)c * K + j] * (a / (1.0f + exp(-b)));
  }
  out[(size_t)t * E + c] = s;
}
#endif  // VIT

// ---- Qwen3.8 Flash Next (qwen4exp) hyper-connections (docs/flash-next-spec.md). The residual of a token is HC
// streams of E floats ([M][HC*E], stream c at c*E); E is a multiple of 256, so every superblock of a row lies in
// one stream.

// xn = per-stream RMS norm of res times w ([HC][E]), stored as f32 and quantized for the low-rank down projection
// (K = HC*E); source rows r0 + m. grid (256, M), local (256, 1); rows >= M_valid store zeros.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void hc_norm_quant(__global const float* res, int r0, __global const float* w, int E, int HC, int M,
                            int M_valid, float eps, __global float* xn, __global char* aq, __global float* asc,
                            __global ushort* absum, int act32) {
  const int m = get_group_id(1), HCE = HC * E, lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const bool valid = m < M_valid;
  __global const float* xr = res + (size_t)(r0 + m) * HCE;
  __local float rs[16];
  for (int c = 0; c < HC; ++c) {
    float ss = 0.0f;
    if (valid)
      for (int i = get_local_id(0); i < E; i += get_local_size(0)) ss += xr[c * E + i] * xr[c * E + i];
    ss = work_group_reduce_add(ss);
    if (get_local_id(0) == 0) rs[c] = rsqrt(ss / E + eps);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = sg; s < HCE / 256; s += get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane, c = (s * 256) / E;
    const float r = rs[c];
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = valid ? xr[base + i] * r * w[base + i] : 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) xn[(size_t)m * HCE + base + i] = v[i];
    quant_store16(v, s, m, M, HCE, aq, asc, absum, act32);
  }
}

// out[t][n] = W[n] . x[t] in f32 (rows of K floats; K a multiple of 64): a router whose choice int8 inputs would
// change (Gemma 4 26B-A4B: int8 router inputs alone move the reference by KL 0.035). One sub-group per (row, token).
// grid (16 * N, T), local (16, 1).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_f32(__global const float* x, int K, __global const float* w, int N, __global float* out, int ldo) {
  const int n = get_global_id(0) / 16, t = get_global_id(1), lane = get_sub_group_local_id();
  if (n >= N) return;
  __global const float4* xr = (__global const float4*)(x + (size_t)t * K);
  __global const float4* wr = (__global const float4*)(w + (size_t)n * K);
  float acc = 0.0f;
  for (int i = lane; i < K / 4; i += 16) {
    const float4 a = xr[i], b = wr[i];
    acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  }
  acc = sub_group_reduce_add(acc);
  if (lane == 0) out[(size_t)t * ldo + n] = acc;
}

// silu(lo / HC) of the low-rank activations ([M][LR], row stride LD), padded with zeros to KP (a multiple of 256) and quantized
// for the up projection. grid (16 * KP / 256, M), local (16, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void hc_lo_quant(__global const float* lo, int LR, int LD, int KP, int HC, int M, int M_valid, __global char* aq,
                          __global float* asc, __global ushort* absum, int act32) {
  const int m = get_global_id(1), s = get_group_id(0), lane = get_sub_group_local_id();
  const float inv_hc = 1.0f / (float)HC;
  float v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int k = s * 256 + 16 * lane + i;
    v[i] = m < M_valid && k < LR ? silu(lo[(size_t)m * LD + k] * inv_hc) : 0.0f;
  }
  quant_store16(v, s, m, M, KP, aq, asc, absum, act32);
}

// The rounding residual of hc_lo_quant (same values and scales: v - q * amax / 127), quantized for a second up pass
// that accumulates (ARC_HC_F32=4: two-pass int8 up input). grid (16 * KP / 256, M), local (16, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void hc_lo_quant_res(__global const float* lo, int LR, int LD, int KP, int HC, int M, int M_valid,
                              __global char* aq, __global float* asc, __global ushort* absum, int act32) {
  const int m = get_global_id(1), s = get_group_id(0), lane = get_sub_group_local_id();
  const float inv_hc = 1.0f / (float)HC;
  float v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int k = s * 256 + 16 * lane + i;
    v[i] = m < M_valid && k < LR ? silu(lo[(size_t)m * LD + k] * inv_hc) : 0.0f;
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmax(amax, fabs(v[i]));
  if (act32) amax = fmax(amax, intel_sub_group_shuffle_xor(amax, 1));
  else amax = sub_group_reduce_max(amax);
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f, d = amax / 127.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] -= (float)convert_int_rte(v[i] * inv) * d;
  quant_store16(v, s, m, M, KP, aq, asc, absum, act32);
}

// The block input: mixed = mean over the streams of xn * sigmoid(gate) ([M][E], f32 and quantized for the block's
// projections, K = E). grid (256, M), local (256, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void hc_gate_mean_quant(__global const float* xn, __global const float* gate, int E, int HC, int M,
                                 int M_valid, __global float* mixed, __global char* aq, __global float* asc,
                                 __global ushort* absum, int act32) {
  const int m = get_group_id(1), HCE = HC * E, lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const bool valid = m < M_valid;
  const float inv_hc = 1.0f / (float)HC;
  for (int s = sg; s < E / 256; s += get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane;
    // (each lane's 16 values with one vector load per stream: read value by value, a load instruction touched 16
    //  cache lines for 4 bytes each; Flash Next, 2048-row chunk: 45 -> 3.7 ms, bit-identical)
    float16 acc = (float16)0.0f;
    if (valid)
      for (int c = 0; c < HC; ++c) {
        const size_t j = (size_t)m * HCE + c * E + base;
        acc += vload16(0, xn + j) / (1.0f + exp(-vload16(0, gate + j)));
      }
    float v[16];
    vstore16(acc * inv_hc, 0, v);
    vstore16(acc * inv_hc, 0, mixed + (size_t)m * E + base);
    quant_store16(v, s, m, M, E, aq, asc, absum, act32);
  }
}

// The mixers of a prompt chunk with f32 activations (ARC_HC_F32; single tokens always take f32: hc_down1 / hc_up1):
// C[m][n] = W[n] . x_m for the Q8 superblock rows of hc_down / hc_up (272 bytes per 256 values: 8 f16 scales, one
// per 32 values, then the 256 int8 values in order; q8sb_dot reads them through the HC_XT order of its inputs). A lane per output column n, a
// work-group 256 columns x HCF_TM rows: the rows' 256 inputs of a superblock go through local memory once per
// work-group. lo_in: the inputs are silu(lo / HC) of the low-rank values (k < LR of rows of stride ld_in, zero up to
// K), else x rows of stride ld_in. grid (256 * ceil(N / 256), ceil(M / HCF_TM)), local (256, 1).
#ifndef HCF_TM
#define HCF_TM 16
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hc_gemm_f32(__global const float* x, int ld_in, int K, int lo_in, int LR, int HC,
                          __global const uchar* wq, int N, __global float* C, int ldc, int M) {
  __local float xs[HCF_TM * 256];
  const int li = get_local_id(0), n = get_global_id(0), m0 = get_group_id(1) * HCF_TM, nsb = K / 256;
  const float inv_hc = 1.0f / (float)HC;
  float acc[HCF_TM];
#pragma unroll
  for (int t = 0; t < HCF_TM; ++t) acc[t] = 0.0f;
  for (int sb = 0; sb < nsb; ++sb) {
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = li; i < HCF_TM * 256; i += 256) {
      const int t = i / 256, k = sb * 256 + i % 256, m = m0 + t;
      float v = 0.0f;
      if (m < M) {
        if (lo_in) v = k < LR ? silu(x[(size_t)m * ld_in + k] * inv_hc) : 0.0f;
        else v = x[(size_t)m * ld_in + k];
      }
      xs[i] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (n < N) {
      __global const uchar* p = wq + ((size_t)n * nsb + sb) * 272;
      for (int l = 0; l < 16; ++l) {
        if (lo_in && sb * 256 + 16 * l >= LR) break;  // (the up projection's zero padding past LR)
        const float d = vload_half(l / 2, (__global const half*)p);
        const uint4 q = vload4(0, (__global const uint*)(p + 16 + 16 * l));
        const uint w4[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const float w = d * (float)(char)((w4[j / 4] >> (8 * (j % 4))) & 0xFF);
          const int k = 16 * l + j;
#pragma unroll
          for (int t = 0; t < HCF_TM; ++t) acc[t] += w * xs[t * 256 + k];
        }
      }
    }
  }
  if (n < N)
#pragma unroll
    for (int t = 0; t < HCF_TM; ++t)
      if (m0 + t < M) C[(size_t)(m0 + t) * ldc + n] = acc[t];
}

// res[m][c] += 2 sigmoid(inject[m][c] / HC) * out[m] for rows < M_valid (inject logits: column off of rows of stride
// ld). grid (HC * E, M_valid).
__kernel void hc_combine(__global float* res, __global const float* out, __global const float* inject, int ld,
                         int off, int E, int HC) {
  const int j = get_global_id(0), m = get_global_id(1), c = j / E, i = j % E;
  const float w = 2.0f / (1.0f + exp(-inject[(size_t)m * ld + off + c] / (float)HC));
  res[(size_t)m * HC * E + j] += w * out[(size_t)m * E + i];
}

// ---- Flash Next PLE (n-gram embedding block of one layer; docs/flash-next-spec.md, llama.cpp qwen4exp)
// Per row m < M_valid and stream c: s = <rmsnorm(key_c) * wk_c, rmsnorm(res_c) * wq_c> / sqrt(E), gate =
// sigmoid(sign(s) * sqrt(max(|s|, 1e-6))), gated_c = value * gate, nrm_c = rmsnorm(gated_c) * wc_c.
// key [M][HC*E], value [M][E], res [M][HC*E]; gated, nrm [M][HC*E]. grid (256, M_valid), local (256, 1).
__kernel void ple_gate(__global const float* key, __global const float* value, __global const float* res,
                       __global const float* wk, __global const float* wq, __global const float* wc, int E, int HC,
                       float eps, __global float* gated, __global float* nrm) {
  const int m = get_group_id(1), HCE = HC * E, li = get_local_id(0), ls = get_local_size(0);
  __global const float* kr = key + (size_t)m * HCE;
  __global const float* qr = res + (size_t)m * HCE;
  __global const float* vr = value + (size_t)m * E;
  for (int c = 0; c < HC; ++c) {
    float kk = 0.0f, qq = 0.0f, kq = 0.0f;
    for (int i = li; i < E; i += ls) {
      const float a = kr[c * E + i], b = qr[c * E + i];
      kk += a * a;
      qq += b * b;
      kq += a * wk[c * E + i] * b * wq[c * E + i];
    }
    kk = work_group_reduce_add(kk);
    qq = work_group_reduce_add(qq);
    kq = work_group_reduce_add(kq);
    const float s = kq * rsqrt(kk / E + eps) * rsqrt(qq / E + eps) / sqrt((float)E);
    const float g = 1.0f / (1.0f + exp(-(s < 0.0f ? -1.0f : s > 0.0f ? 1.0f : 0.0f) * sqrt(fmax(fabs(s), 1e-6f))));
    float gg = 0.0f;
    for (int i = li; i < E; i += ls) {
      const float v = vr[i] * g;
      gated[(size_t)m * HCE + c * E + i] = v;
      gg += v * v;
    }
    gg = work_group_reduce_add(gg);
    const float r = rsqrt(gg / E + eps);
    for (int i = li; i < E; i += ls) nrm[(size_t)m * HCE + c * E + i] = vr[i] * g * r * wc[c * E + i];
  }
}

// res[m][ch] += gated[m][ch] + silu(sum_k w[ch][k] * nrm(m - (NTAP-1-k) * DIL)[ch]) for rows < T: a depthwise causal
// conv over the rows of this chunk and, before them, the history hist [HIST][HCE] (the last HIST rows of earlier
// positions, oldest first; zeros at the start of a sequence). grid (HCE, T).
__kernel void ple_conv_add(__global float* res, __global const float* gated, __global const float* nrm,
                           __global const float* hist, __global const float* w, int HCE, int NTAP, int DIL, int HIST) {
  const int ch = get_global_id(0), m = get_global_id(1);
  float acc = 0.0f;
  for (int k = 0; k < NTAP; ++k) {
    const int t = m - (NTAP - 1 - k) * DIL;
    const float v = t >= 0 ? nrm[(size_t)t * HCE + ch] : t + HIST >= 0 ? hist[(size_t)(t + HIST) * HCE + ch] : 0.0f;
    acc += w[(size_t)ch * NTAP + k] * v;
  }
  res[(size_t)m * HCE + ch] += gated[(size_t)m * HCE + ch] + acc / (1.0f + exp(-acc));
}

// hist = the last HIST rows of [hist; nrm[0..T)], in place (each work-item owns one channel). grid (HCE).
__kernel void ple_hist_update(__global float* hist, __global const float* nrm, int HCE, int T, int HIST) {
  const int ch = get_global_id(0);
  float h[16];
  for (int j = 0; j < HIST; ++j) {
    const int t = T - HIST + j;  // row of the combined sequence relative to the chunk start
    h[j] = t >= 0 ? nrm[(size_t)t * HCE + ch] : hist[(size_t)(t + HIST) * HCE + ch];
  }
  for (int j = 0; j < HIST; ++j) hist[(size_t)j * HCE + ch] = h[j];
}

// ---- Flash Next QSA (sparse attention, llama.cpp qwen4exp build_qsa_sel; docs/flash-next-spec.md). The indexer keys
// of blocks of QR consecutive positions are pooled (mean of the raw keys, RMS norm, RoPE at the block's first
// position); a query scores every complete block it can see (sum over the indexer heads of relu(q . k) / sqrt(dim))
// and attends to the keys of its top blocks and of its incomplete tail block. Indexer rows ixo [M][ldx]: raw key
// (QD floats) at column 0, the QH query heads after it. A lane holds dims lane + 16 j (QD = 128): RoPE pairs (i, i +
// n_rot / 2) are then in one lane.
#ifndef QD
#define QD 128
#endif
#ifndef QR  // (block size and indexer heads: the engine passes the model's)
#define QR 4
#endif
#ifndef QH
#define QH 4
#endif
#define QJ (QD / 16)
inline void qsa_norm_rope(float* v, __global const float* w, float eps, int p, float theta_scale, int n_rot) {
  const int lane = get_sub_group_local_id();
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < QJ; ++j) ss += v[j] * v[j];
  const float r = rsqrt(sub_group_reduce_add(ss) / QD + eps);
#pragma unroll
  for (int j = 0; j < QJ; ++j) v[j] *= r * w[lane + 16 * j];
  const int jh = n_rot / 32;  // (pairs (i, i + n_rot / 2): lane values j, j + jh; n_rot a multiple of 32)
#pragma unroll
  for (int j = 0; j < QJ; ++j)
    if (j < jh) {
      const float theta = (float)p * pow(theta_scale, (float)(lane + 16 * j));
      const float c = cos(theta), sn = sin(theta), a = v[j], b = v[j + jh];
      v[j] = a * c - b * sn;
      v[j + jh] = a * sn + b * c;
    }
}
// pooled keys of the blocks b0 .. b0 + get_num_groups(0) - 1 (complete within positions < pos0 + T); positions
// before pos0 come from the tail state (positions pos0 / QR * QR .. pos0 - 1). grid (16 * blocks), local (16).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qsa_pool(__global const float* ixo, int ldx, int pos0, __global const float* tail, __global const float* kn,
                       int b0, float eps, float theta_scale, int n_rot, __global float* pk) {
  const int b = b0 + get_group_id(0), lane = get_sub_group_local_id(), t0 = pos0 / QR * QR;
  float v[QJ];
#pragma unroll
  for (int j = 0; j < QJ; ++j) v[j] = 0.0f;
  for (int i = 0; i < QR; ++i) {
    const int p = b * QR + i;
#pragma unroll
    for (int j = 0; j < QJ; ++j) v[j] += p >= pos0 ? ixo[(size_t)(p - pos0) * ldx + lane + 16 * j] : tail[(p - t0) * QD + lane + 16 * j];
  }
#pragma unroll
  for (int j = 0; j < QJ; ++j) v[j] *= 1.0f / QR;
  qsa_norm_rope(v, kn, eps, b * QR, theta_scale, n_rot);
#pragma unroll
  for (int j = 0; j < QJ; ++j) pk[(size_t)b * QD + lane + 16 * j] = v[j];
}
// the tail state after the chunk: raw keys of positions (pos0 + T) / QR * QR .. pos0 + T - 1. grid (QD), one group.
__kernel void qsa_tail(__global const float* ixo, int ldx, int pos0, int T, __global float* tail) {
  const int d = get_global_id(0), t0 = pos0 / QR * QR, n0 = (pos0 + T) / QR * QR;
  float h[QR];
  for (int i = 0; i < QR - 1; ++i) {
    const int p = n0 + i;
    h[i] = p >= pos0 + T ? 0.0f : p >= pos0 ? ixo[(size_t)(p - pos0) * ldx + d] : tail[(p - t0) * QD + d];
  }
  for (int i = 0; i < QR - 1; ++i) tail[i * QD + d] = h[i];
}
// indexer queries: RMS norm + RoPE per head. grid (16 * QH, T), local (16, 1).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qsa_q(__global const float* ixo, int ldx, int pos0, __global const float* qn, float eps, float theta_scale,
                    int n_rot, __global float* qx) {
  const int m = get_global_id(1), hh = get_group_id(0), lane = get_sub_group_local_id();
  float v[QJ];
#pragma unroll
  for (int j = 0; j < QJ; ++j) v[j] = ixo[(size_t)m * ldx + QD + hh * QD + lane + 16 * j];
  qsa_norm_rope(v, qn, eps, pos0 + m, theta_scale, n_rot);
#pragma unroll
  for (int j = 0; j < QJ; ++j) qx[((size_t)m * QH + hh) * QD + lane + 16 * j] = v[j];
}
// scores of rows r0 .. r0 + R - 1 against the blocks they see (else -1). grid (blocks rounded to 64, R), local (64, 1).
__kernel void qsa_score(__global const float* qx, __global const float* pk, int r0, int pos0, __global float* sc, int lds) {
  const int b = get_global_id(0), i = get_global_id(1), m = r0 + i, nv = (pos0 + m + 1) / QR;
  __local float q[QH * QD];
  for (int k = get_local_id(0); k < QH * QD; k += get_local_size(0)) q[k] = qx[(size_t)m * QH * QD + k];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (b >= lds) return;
  float s = -1.0f;
  if (b < nv) {
    __global const float4* kp = (__global const float4*)(pk + (size_t)b * QD);
    s = 0.0f;
    for (int hh = 0; hh < QH; ++hh) {
      float d = 0.0f;
      for (int k = 0; k < QD / 4; ++k) {
        const float4 kv = kp[k];
        d += q[hh * QD + 4 * k] * kv.x + q[hh * QD + 4 * k + 1] * kv.y + q[hh * QD + 4 * k + 2] * kv.z + q[hh * QD + 4 * k + 3] * kv.w;
      }
      s += fmax(d, 0.0f);
    }
    s *= rsqrt((float)QD);
  }
  sc[(size_t)i * lds + b] = s;
}
// qsa_score for QRT rows at a time: each block's keys are read once for the QRT rows (qsa_score reads them once per
// row: the whole pooled key array per row, ~64 GB per layer of a 32K prompt). Per row and head the same sum in the
// same order (k ascending, then the heads), so the scores are the same. grid (blocks rounded to 64, ceil(R / QRT)),
// local (64, 1).
#ifndef QRT
#define QRT 8
#endif
__kernel void qsa_score_rt(__global const float* qx, __global const float* pk, int r0, int pos0, int R,
                           __global float* sc, int lds) {
  const int b = get_global_id(0), i0 = get_global_id(1) * QRT;
  __local float q[QRT * QH * QD];
  for (int k = get_local_id(0); k < QRT * QH * QD; k += get_local_size(0)) {
    const int r = k / (QH * QD);
    q[k] = i0 + r < R ? qx[(size_t)(r0 + i0 + r) * QH * QD + k % (QH * QD)] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (b >= lds) return;
  float d[QRT][QH];
#pragma unroll
  for (int r = 0; r < QRT; ++r)
#pragma unroll
    for (int hh = 0; hh < QH; ++hh) d[r][hh] = 0.0f;
  __global const float4* kp = (__global const float4*)(pk + (size_t)b * QD);
  for (int k = 0; k < QD / 4; ++k) {
    const float4 kv = kp[k];
#pragma unroll
    for (int r = 0; r < QRT; ++r)
#pragma unroll
      for (int hh = 0; hh < QH; ++hh) {
        __local const float* qq = q + (r * QH + hh) * QD + 4 * k;
        d[r][hh] += qq[0] * kv.x + qq[1] * kv.y + qq[2] * kv.z + qq[3] * kv.w;
      }
  }
#pragma unroll
  for (int r = 0; r < QRT; ++r) {
    const int i = i0 + r, nv = (pos0 + r0 + i + 1) / QR;
    if (i >= R) break;
    float s = -1.0f;
    if (b < nv) {
      s = 0.0f;
#pragma unroll
      for (int hh = 0; hh < QH; ++hh) s += fmax(d[r][hh], 0.0f);
      s *= rsqrt((float)QD);
    }
    sc[(size_t)i * lds + b] = s;
  }
}
// the mask of row r0 + i: its top `ntop` visible blocks (ties: the lower blocks), all of them when it sees no more.
// Exact selection by a radix select on the score bits (scores >= 0, so their bits sort as the values). mask
// [M][qmw] words, bit b = block b. grid (256, R), local (256, 1).
__kernel void qsa_select(__global const float* sc, int lds, int r0, int pos0, int ntop, __global uint* mask, int qmw) {
  const int i = get_group_id(1), m = r0 + i, lid = get_local_id(0), nv = (pos0 + m + 1) / QR;
  __global const uint* su = (__global const uint*)(sc + (size_t)i * lds);
  __local uint hist[256];
  __local uint sel[2];  // prefix, rank left
  uint prefix = 0, pmask = 0, k = (uint)ntop;
  const bool all = nv <= ntop;
  if (!all)
    for (int shift = 24; shift >= 0; shift -= 8) {
      hist[lid] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int b = lid; b < nv; b += 256) {
        const uint u = su[b];
        if ((u & pmask) == prefix) atomic_inc(&hist[(u >> shift) & 255]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lid == 0) {
        uint cum = 0;
        int bin = 255;
        for (; bin > 0; --bin) {
          if (cum + hist[bin] >= k) break;
          cum += hist[bin];
        }
        sel[0] = prefix | ((uint)bin << shift);
        sel[1] = k - cum;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      prefix = sel[0];
      k = sel[1];
      pmask |= 255u << shift;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  // prefix = the ntop-th largest score; k = how many of the scores equal to it are taken (the lowest blocks)
  const int nw = (nv + 31) / 32, per = (nw + 255) / 256, w0 = lid * per, w1 = min(nw, w0 + per);
  uint eq = 0;
  if (!all)
    for (int b = w0 * 32; b < min(nv, w1 * 32); ++b) eq += su[b] == prefix;
  uint before = work_group_scan_exclusive_add(eq);
  for (int w = w0; w < w1; ++w) {
    uint bits = 0;
    for (int j = 0; j < 32; ++j) {
      const int b = w * 32 + j;
      if (b >= nv) break;
      bool take = all;
      if (!all) {
        const uint u = su[b];
        take = u > prefix;
        if (u == prefix) { take = before < k; ++before; }
      }
      bits |= (take ? 1u : 0u) << j;
    }
    mask[(size_t)m * qmw + w] = bits;
  }
}

// ---- DFlash2 block drafter (z-lab/dflash model.py; zero/dflash.h). Rows of a block of DB positions.
// Grouped dynamic causal conv over the block rows: out[t][c] = sum_k (base[k][c] + dyn[t][half][k][g(c)]) x[t-k][c]
// (x rows before 0 are zero), dyn rows [2][K][E/G] of ldd floats. grid (E, rows).
__kernel void dfl_conv(__global const float* x, __global const float* dyn, int ldd, int half_, __global const float* base,
                       int E, int G, int K, __global float* out) {
  const int c = get_global_id(0), t = get_global_id(1), N = E / G, g = c / G;
  float acc = 0.0f;
  for (int k = 0; k < K && k <= t; ++k)
    acc += (base[k * E + c] + dyn[(size_t)t * ldd + (half_ * K + k) * N + g]) * x[(size_t)(t - k) * E + c];
  out[(size_t)t * E + c] = acc;
}
// RMS norm (w) + RoPE over all QD dims of each head, in place: rows [rows][ld], heads at h * QD. grid (16 * nh, rows).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dfl_norm_rope(__global float* x, int ld, __global const float* w, float eps, int pos0, float theta_scale) {
  const int m = get_global_id(1), hh = get_group_id(0), lane = get_sub_group_local_id();
  __global float* r = x + (size_t)m * ld + hh * QD;
  float v[QJ];
#pragma unroll
  for (int j = 0; j < QJ; ++j) v[j] = r[lane + 16 * j];
  qsa_norm_rope(v, w, eps, pos0 + m, theta_scale, QD);
#pragma unroll
  for (int j = 0; j < QJ; ++j) r[lane + 16 * j] = v[j];
}
// k, v rows [rows][KVW] (k normed and rotated) into the context rings [R][KVW] at positions pos0 + row. grid (KVW, rows).
__kernel void dfl_kv_store(__global const float* k, __global const float* v, int KVW, int pos0, int R,
                           __global float* rk, __global float* rv) {
  const int i = get_global_id(0), m = get_global_id(1);
  const size_t dst = (size_t)((pos0 + m) % R) * KVW + i;
  rk[dst] = k[(size_t)m * KVW + i];
  rv[dst] = v[(size_t)m * KVW + i];
}
// Block attention: query row i (position start + i) of head h over the context keys at positions > start + i - win
// (ring of R positions) and every block key (non-causal). q [rows][NH * QD], kb / vb [rows][NKV * QD] (k normed +
// rotated), out [rows][NH * QD]. grid (QD, rows * NH), local (QD, 1).
__kernel void dfl_attn(__global const float* q, __global const float* kb, __global const float* vb,
                       __global const float* rk, __global const float* rv, int R, int start, int rows, int win, int NH,
                       int NKV, __global float* out) {
  const int d = get_local_id(0), i = get_group_id(1) / NH, h = get_group_id(1) % NH, g = h / (NH / NKV);
  const int KVW = NKV * QD, lo = max(0, start + i - win + 1), nc = start - lo, nk = nc + rows;
  __local float qv[QD];
  __local float sc[4096 + 64];
  __local float red[QD];
  qv[d] = q[((size_t)i * NH + h) * QD + d] * rsqrt((float)QD);
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = -INFINITY;
  for (int k = d; k < nk; k += QD) {
    __global const float* kr = k < nc ? rk + (size_t)((lo + k) % R) * KVW + g * QD : kb + (size_t)(k - nc) * KVW + g * QD;
    float s = 0.0f;
    for (int e = 0; e < QD; ++e) s += qv[e] * kr[e];
    sc[k] = s;
    mx = fmax(mx, s);
  }
  mx = work_group_reduce_max(mx);
  float z = 0.0f;
  for (int k = d; k < nk; k += QD) {
    const float p = exp(sc[k] - mx);
    sc[k] = p;
    z += p;
  }
  z = work_group_reduce_add(z);
  barrier(CLK_LOCAL_MEM_FENCE);
  float acc = 0.0f;
  for (int k = 0; k < nk; ++k) {
    __global const float* vr = k < nc ? rv + (size_t)((lo + k) % R) * KVW + g * QD : vb + (size_t)(k - nc) * KVW + g * QD;
    acc += sc[k] * vr[d];
  }
  out[((size_t)i * NH + h) * QD + d] = acc / z;
}
// x += y over rows [rows][E]. grid (E, rows).
__kernel void dfl_add(__global float* x, __global const float* y, int E) {
  const size_t j = (size_t)get_global_id(1) * E + get_global_id(0);
  x[j] += y[j];
}

// ---- Flash Next mixers of a single-token pass, fused, with f32 activations (the prompt path quantizes them for the
// int8 GEMMs). Weights in the Q8_0 GPU raw form: superblocks of 8 f16 scales + 256 int8 (272 bytes). HCK = HC * E.
#ifndef HCK
#define HCK 10240
#endif
// The activations in local memory are stored per superblock transposed, element 16 l + j at 16 j + l (HC_XT), so that
// the lanes' reads of one step are consecutive: read as x[16 l + j] all 16 lanes hit the same bank (hc_up1 100 us).
#define HC_XT(k) (((k) & ~255) | (((k) & 15) << 4) | (((k) >> 4) & 15))
// dot of 256 weights (one superblock at p) with x[0..256) (HC_XT layout): lane l takes elements 16 l .. 16 l + 15
// (block l / 2)
inline float q8sb_dot(__global const uchar* p, __local const float* x) {
  const int lane = get_sub_group_local_id();
  const float d = vload_half(lane / 2, (__global const half*)p);
  const uint4 q = vload4(0, (__global const uint*)(p + 16 + 16 * lane));
  const uint w[4] = {q.x, q.y, q.z, q.w};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += (float)(char)((w[j / 4] >> (8 * (j % 4))) & 0xFF) * x[16 * j + lane];
  return d * acc;
}
// q8sb_dot for a superblock holding nv < 256 values (the rest zero padding): lanes past them load nothing (the up
// projection's 320 values in 512; hc_up1 905 -> 843 ms per 96 Flash Next tokens, bit-identical; returning 0 for
// those lanes instead was 731 ms but changed the compiled arithmetic and with it the results)
inline float q8sb_dot_v(__global const uchar* p, __local const float* x, int nv) {
  const int lane = get_sub_group_local_id();
  const bool on = 16 * lane < nv;  // (the same arithmetic for every lane, only the loads predicated)
  const float d = on ? vload_half(lane / 2, (__global const half*)p) : 0.0f;
  const uint4 q = on ? vload4(0, (__global const uint*)(p + 16 + 16 * lane)) : (uint4)0;
  const uint w[4] = {q.x, q.y, q.z, q.w};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += (float)(char)((w[j / 4] >> (8 * (j % 4))) & 0xFF) * x[16 * j + lane];
  return d * acc;
}
// Partial sums of lo[n] = W_down[n] . xn over K slice ks (HCK / KS values; n < N: the low-rank and the inject rows),
// xn = per-stream RMS norm of res row r0 times w; part [KS][N]; the streams' 1 / rms to rinv. grid (256 * ceil(N / 16),
// KS), local (256, 1): a sub-group per (row, slice), so enough threads run (one row per sub-group over all of K: 21
// work-groups, 3x slower than the int8 GEMV path).
#ifndef HC_KS
#define HC_KS 8
#endif
#ifndef HC_RT
#define HC_RT 1  // rows (single tokens: 1; the 2-8 rows of a verification pass, each computed as a single token is)
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hc_down1(__global const float* res, int r0, __global const float* w, int E, int HC, float eps,
                       __global const uchar* wd, int N, __global float* part, __global float* rinv) {
  __local float xs[HC_RT][HCK / HC_KS];
  __local float rs[HC_RT];
  const int li = get_local_id(0), ls = get_local_size(0), ks = get_group_id(1), SL = HC * E / HC_KS;
  const int k0 = ks * SL, c = k0 / E;  // (a slice lies in one stream: E is a multiple of SL)
  for (int r = 0; r < HC_RT; ++r) {
    __global const float* xr = res + (size_t)(r0 + r) * HC * E;
    float ss = 0.0f;
    for (int i = li; i < E; i += ls) {
      const float v = xr[c * E + i];
      ss += v * v;
    }
    ss = work_group_reduce_add(ss);
    if (li == 0) rs[r] = rsqrt(ss / E + eps);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int r = 0; r < HC_RT; ++r) {
    __global const float* xr = res + (size_t)(r0 + r) * HC * E;
    const float rr = rs[r];
    for (int i = li; i < SL; i += ls) xs[r][HC_XT(i)] = xr[k0 + i] * rr * w[k0 + i];
    if (get_group_id(0) == 0 && li == 0 && (k0 % E) == 0) rinv[r * HC + c] = rr;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n = get_group_id(0) * 16 + get_sub_group_id(), nsb = HC * E / 256, s0 = k0 / 256;
  if (n >= N) return;
  float acc[HC_RT];
#pragma unroll
  for (int r = 0; r < HC_RT; ++r) acc[r] = 0.0f;
  for (int sb = 0; sb < SL / 256; ++sb)
#pragma unroll
    for (int r = 0; r < HC_RT; ++r) acc[r] += q8sb_dot(wd + ((size_t)n * nsb + s0 + sb) * 272, xs[r] + sb * 256);
#pragma unroll
  for (int r = 0; r < HC_RT; ++r) {
    const float a = sub_group_reduce_add(acc[r]);
    if (get_sub_group_local_id() == 0) part[((size_t)r * HC_KS + ks) * N + n] = a;
  }
}
// mixed[i] = mean over the streams c of xn_c[i] * sigmoid(W_up[c E + i] . silu(lo / HC)), lo = the sum of the partial
// sums (LR values, the rest of the KP-wide input zero). grid (256 * ceil(E / 16)), local (256): a sub-group per i.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hc_up1(__global const float* res, int r0, __global const float* w, __global const float* rinv,
                     __global const float* part, int N, __global float* lo, int LR, int KP, int E, int HC,
                     __global const uchar* wu, __global float* mixed) {
  __local float xs[HC_RT][1024];
  const int li = get_local_id(0);
  for (int r = 0; r < HC_RT; ++r)
    for (int k = li; k < KP; k += get_local_size(0)) {
      float v = 0.0f;
      if (k < N)
        for (int q = 0; q < HC_KS; ++q) v += part[((size_t)r * HC_KS + q) * N + k];
      if (get_group_id(0) == 0 && k < N) lo[(size_t)r * N + k] = v;  // (the inject logits for hc_combine)
      v = k < LR ? v / (float)HC : 0.0f;
      xs[r][HC_XT(k)] = v / (1.0f + exp(-v));
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int i = get_group_id(0) * 16 + get_sub_group_id(), nsb = KP / 256;
  if (i >= E) return;
  float m[HC_RT];
#pragma unroll
  for (int r = 0; r < HC_RT; ++r) m[r] = 0.0f;
  for (int c = 0; c < HC; ++c) {
    const int n = c * E + i;
    float acc[HC_RT];
#pragma unroll
    for (int r = 0; r < HC_RT; ++r) acc[r] = 0.0f;
    for (int sb = 0; sb < nsb; ++sb)
#pragma unroll
      for (int r = 0; r < HC_RT; ++r) acc[r] += q8sb_dot_v(wu + ((size_t)n * nsb + sb) * 272, xs[r] + sb * 256, LR - sb * 256);
#pragma unroll
    for (int r = 0; r < HC_RT; ++r) {
      const float g = sub_group_reduce_add(acc[r]);
      __global const float* xr = res + (size_t)(r0 + r) * HC * E;
      m[r] += xr[n] * rinv[r * HC + c] * w[n] / (1.0f + exp(-g));
    }
  }
  if (get_sub_group_local_id() == 0)
#pragma unroll
    for (int r = 0; r < HC_RT; ++r) mixed[(size_t)r * E + i] = m[r] / (float)HC;
}
