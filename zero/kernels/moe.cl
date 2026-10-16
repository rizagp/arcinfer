// Mixture-of-experts FFN (Qwen3.5 / Qwen3.6 MoE): routing, grouping the (token, expert) pairs by expert ("slots"),
// the expert GEMMs on GSQ 2-bit weights and the residual update. Compiled with -DNE (experts), -DNU (experts per
// token) and -DTM (slots per tile: rows of one sub-group).
//
// Expert weights (tools/convert_gsq.py): q in {-2,-1,0,1} stored as q + 2 in 2 bits, a bf16 scale per 128 inputs.
// Per expert matrix [N][K]: codes [N/16][K/128][8][16] words (column group, input group, word, lane = output
// column); in a word, the code of input e (of its 16) sits at bit 8 * (e % 4) + 2 * (e / 4), so dec2(w, j) is the
// DP4a operand of inputs 4j .. 4j+3. Scales [N/16][K/128][16].
// Activations: int8 rows (K per row) with per-32 scales sc[(k / 32) * M + row] (rmsnorm_quant's act32 layout).

#define SG 16
#define BCAST(x, i) intel_sub_group_shuffle((x), (i))

inline float bf16f(ushort h) { return as_float((uint)h << 16); }
// the four signed bytes q = code - 2 of inputs 4j .. 4j+3 of a code word (no borrows: every byte stays below 0x82)
inline uint dec2(uint w, int j) { return (((w >> (2 * j)) & 0x03030303u) + 0x7E7E7E7Eu) ^ 0x80808080u; }

// One sub-group per token: softmax over the NE router logits, top NU (ties: lower expert), weights renormalized
// over the selection; counts the tokens of every expert. Row t of `logits` holds the shared-expert gate logit at
// column NE. grid (16, T), local (16, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void moe_route(__global const float* logits, int ldl, int T, __global int* ids, __global float* wts,
                        __global float* sgate, __global int* counts) {
  const int t = get_group_id(1), l = get_sub_group_local_id();
  if (t >= T) return;
  __global const float* lg = logits + (size_t)t * ldl;
  float v[NE / SG];
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < NE / SG; ++i) {
    v[i] = lg[i * SG + l];
    mx = fmax(mx, v[i]);
  }
  mx = sub_group_reduce_max(mx);
  int sel[NU];
  float p[NU], sum = 0.0f;
#pragma unroll
  for (int k = 0; k < NU; ++k) {
    float bv = -INFINITY;
    int bi = NE;
#pragma unroll
    for (int i = 0; i < NE / SG; ++i)
      if (v[i] > bv) { bv = v[i]; bi = i * SG + l; }
    const float gv = sub_group_reduce_max(bv);
    const int gi = sub_group_reduce_min(bv == gv ? bi : NE);
#pragma unroll
    for (int i = 0; i < NE / SG; ++i)
      if (i * SG + l == gi) v[i] = -INFINITY;
    sel[k] = gi;
    p[k] = exp(gv - mx);
    sum += p[k];
  }
#pragma unroll
  for (int k = 0; k < NU; ++k)
    if (l == k) {
      ids[t * NU + k] = sel[k];
      wts[t * NU + k] = p[k] / sum;
      atomic_inc(&counts[sel[k]]);
    }
  if (l == 0) sgate[t] = 1.0f / (1.0f + exp(-lg[NE]));
}

// Single token (TM = 1): routing straight into NU one-slot tiles, slot k = the k-th selected expert (no counts,
// scan or scatter). grid (16), local (16).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void moe_route1(__global const float* logits, __global float* sgate, __global int* tiles,
                         __global int* stok, __global float* swt, __global int* inv) {
  const int l = get_sub_group_local_id();
  float v[NE / SG];
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < NE / SG; ++i) {
    v[i] = logits[i * SG + l];
    mx = fmax(mx, v[i]);
  }
  mx = sub_group_reduce_max(mx);
  int sel[NU];
  float p[NU], sum = 0.0f;
#pragma unroll
  for (int k = 0; k < NU; ++k) {
    float bv = -INFINITY;
    int bi = NE;
#pragma unroll
    for (int i = 0; i < NE / SG; ++i)
      if (v[i] > bv) { bv = v[i]; bi = i * SG + l; }
    const float gv = sub_group_reduce_max(bv);
    const int gi = sub_group_reduce_min(bv == gv ? bi : NE);
#pragma unroll
    for (int i = 0; i < NE / SG; ++i)
      if (i * SG + l == gi) v[i] = -INFINITY;
    sel[k] = gi;
    p[k] = exp(gv - mx);
    sum += p[k];
  }
#pragma unroll
  for (int k = 0; k < NU; ++k)
    if (l == k) {
      tiles[1 + 3 * k] = sel[k];
      tiles[2 + 3 * k] = k;
      tiles[3 + 3 * k] = 1;
      stok[k] = 0;
      swt[k] = p[k] / sum;
      inv[k] = k;
    }
  if (l == 0) {
    tiles[0] = NU;
    sgate[0] = 1.0f / (1.0f + exp(-logits[NE]));
  }
}

// Slot offsets per expert (exclusive scan of the counts) and the tile list: tiles[0] = number of tiles, then
// (expert, first slot, rows) per tile of up to TM slots. Resets the counts for the next layer. One work-group of NE.
__kernel void moe_scan(__global int* counts, __global int* offs, __global int* cursor, __global int* tiles,
                       __global int* tstart) {
  __local int a[NE], b[NE];
  const int e = get_local_id(0);
  const int c = counts[e], nt = (c + TM - 1) / TM;
  a[e] = c;
  b[e] = nt;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int d = 1; d < NE; d <<= 1) {
    const int x = e >= d ? a[e - d] : 0, y = e >= d ? b[e - d] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    a[e] += x;
    b[e] += y;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const int off = a[e] - c, toff = b[e] - nt;
  offs[e] = off;
  cursor[e] = off;
  tstart[e] = toff;  // (first tile of each expert; tstart[NE] = the tile count)
  if (e == NE - 1) tiles[0] = tstart[NE] = b[e];
  for (int i = 0; i < nt; ++i) {
    __global int* tp = tiles + 1 + 3 * (toff + i);
    tp[0] = e;
    tp[1] = off + i * TM;
    tp[2] = min(TM, c - i * TM);
  }
  counts[e] = 0;
}

// Places every (token, expert) pair in its expert's slot range. grid (T * NU)
__kernel void moe_scatter(__global const int* ids, __global const float* wts, int n, __global int* cursor,
                          __global int* stok, __global float* swt, __global int* inv) {
  const int i = get_global_id(0);
  if (i >= n) return;
  const int s = atomic_inc(&cursor[ids[i]]);
  stok[s] = i / NU;
  swt[s] = wts[i];
  inv[i] = s;
}

// Dot products of TM activation rows with one 128-input group of 16 expert columns (one per lane): the rows'
// int8 values come in as two uints per row (a0: inputs 0..63, a1: 64..127, lane i holding 4i..4i+3); dw are the
// decoded weights (32 DP4a operands); returns the activation-scaled sum (weight scale not applied).
#define GROUP_DOT(dw, a0, a1, s0, s1, s2, s3, out)                                       \
  {                                                                                      \
    int i0 = 0, i1 = 0, i2 = 0, i3 = 0;                                                  \
    _Pragma("unroll") for (int w = 0; w < 8; ++w) {                                      \
      i0 += dot_4x8packed_ss_int(BCAST(a0, w), (dw)[w]);                                 \
      i1 += dot_4x8packed_ss_int(BCAST(a0, 8 + w), (dw)[8 + w]);                         \
      i2 += dot_4x8packed_ss_int(BCAST(a1, w), (dw)[16 + w]);                            \
      i3 += dot_4x8packed_ss_int(BCAST(a1, 8 + w), (dw)[24 + w]);                        \
    }                                                                                    \
    out = (s0) * (float)i0 + (s1) * (float)i1 + (s2) * (float)i2 + (s3) * (float)i3;    \
  }

// decoded DP4a operands of one lane's 8 code words (128 inputs): index 4 * word + j covers inputs 16 word + 4j ..
#define DECODE_GROUP(cw, dw)                                         \
  _Pragma("unroll") for (int w = 0; w < 8; ++w)                      \
    _Pragma("unroll") for (int j = 0; j < 4; ++j) (dw)[4 * w + j] = dec2((cw)[w], j);

// h[slot][F] = silu(x W_gate^T) * (x W_up^T) for one tile (up to TM slots of one expert) and 16 columns.
// x rows are the tokens of the slots (aq [token][K], asc per-32 scales, row stride M).
// grid (F, max_tiles), local (16, 1)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void moe_gateup(__global const uint* wg, __global const ushort* sg, __global const uint* wu,
                         __global const ushort* su, __global const char* aq, __global const float* asc, int M, int K,
                         int F, __global const int* tiles, __global const int* stok, __global float* h) {
  const int ti = get_group_id(1);
  if (ti >= tiles[0]) return;
  __global const int* tp = tiles + 1 + 3 * ti;
  const int e = tp[0], r0 = tp[1], nr = tp[2];
  const int g = get_group_id(0), l = get_sub_group_local_id(), KG = K / 128;
  const size_t wo = (size_t)e * F * K / 16 + (size_t)g * KG * 128, so = (size_t)e * F * K / 128 + (size_t)g * KG * 16;
  int tok[TM];
  float rg[TM], ru[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    tok[m] = stok[r0 + min(m, nr - 1)];
    rg[m] = 0.0f;
    ru[m] = 0.0f;
  }
  for (int kg = 0; kg < KG; ++kg) {
    const uint8 G = intel_sub_group_block_read8(wg + wo + kg * 128), U = intel_sub_group_block_read8(wu + wo + kg * 128);
    const uint gw[8] = {G.s0, G.s1, G.s2, G.s3, G.s4, G.s5, G.s6, G.s7};
    const uint uw[8] = {U.s0, U.s1, U.s2, U.s3, U.s4, U.s5, U.s6, U.s7};
    const float gs = bf16f(intel_sub_group_block_read_us(sg + so + kg * 16));
    const float us = bf16f(intel_sub_group_block_read_us(su + so + kg * 16));
    uint dg[32], du[32];
    DECODE_GROUP(gw, dg);
    DECODE_GROUP(uw, du);
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      __global const char* ap = aq + (size_t)tok[m] * K + kg * 128;
      const uint a0 = intel_sub_group_block_read((__global const uint*)ap);
      const uint a1 = intel_sub_group_block_read((__global const uint*)(ap + 64));
      __global const float* sp = asc + (size_t)kg * 4 * M + tok[m];
      const float s0 = sp[0], s1 = sp[M], s2 = sp[2 * M], s3 = sp[3 * M];
      float vg, vu;
      GROUP_DOT(dg, a0, a1, s0, s1, s2, s3, vg);
      GROUP_DOT(du, a0, a1, s0, s1, s2, s3, vu);
      rg[m] = fma(gs, vg, rg[m]);
      ru[m] = fma(us, vu, ru[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
    if (m < nr) {
      const float x = rg[m];
      h[(size_t)(r0 + m) * F + g * SG + l] = x / (1.0f + exp(-x)) * ru[m];
    }
}

// int8 rows with per-32 scales of h (slots x F), as rmsnorm_quant's act32 layout with row stride S.
// grid (16 * F / 256, n), local (16, 1): one sub-group per 256 values of a row.
#ifndef FV  // columns that hold values (Flash Next: 640 of each 768-wide row; the rest quantize as zeros)
#define FV F
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void moe_quant(__global const float* h, int F, int S, __global char* hq, __global float* hsc) {
  const int s = get_group_id(0), r = get_group_id(1), lane = get_sub_group_local_id();
  const int base = s * 256 + 16 * lane;
  float v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = base + i < FV ? h[(size_t)r * F + base + i] : 0.0f;
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmax(amax, fabs(v[i]));
  amax = fmax(amax, intel_sub_group_shuffle_xor(amax, 1));
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  uint pk[4] = {0, 0, 0, 0};
#pragma unroll
  for (int i = 0; i < 16; ++i) pk[i / 4] |= (uint)(convert_int_rte(v[i] * inv) & 0xFF) << (8 * (i % 4));
  vstore4((uint4)(pk[0], pk[1], pk[2], pk[3]), 0, (__global uint*)(hq + (size_t)r * F + base));
  if ((lane & 1) == 0) hsc[((size_t)s * 8 + lane / 2) * S + r] = amax / 127.0f;
}

// y[slot][N] = weight(slot) * (h[slot] W_down^T) for one tile and 16 columns. grid (N, max_tiles), local (16, 1)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void moe_down(__global const uint* wd, __global const ushort* sd, __global const char* hq,
                       __global const float* hsc, int S, int K, int N, __global const int* tiles,
                       __global const float* swt, __global float* y) {
  const int ti = get_group_id(1);
  if (ti >= tiles[0]) return;
  __global const int* tp = tiles + 1 + 3 * ti;
  const int e = tp[0], r0 = tp[1], nr = tp[2];
  const int g = get_group_id(0), l = get_sub_group_local_id(), KG = K / 128;
  const size_t wo = (size_t)e * N * K / 16 + (size_t)g * KG * 128, so = (size_t)e * N * K / 128 + (size_t)g * KG * 16;
  float r[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) r[m] = 0.0f;
  for (int kg = 0; kg < KG; ++kg) {
    const uint8 D = intel_sub_group_block_read8(wd + wo + kg * 128);
    const uint cw[8] = {D.s0, D.s1, D.s2, D.s3, D.s4, D.s5, D.s6, D.s7};
    const float ws = bf16f(intel_sub_group_block_read_us(sd + so + kg * 16));
    uint dw[32];
    DECODE_GROUP(cw, dw);
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      const int row = r0 + min(m, nr - 1);
      __global const char* ap = hq + (size_t)row * K + kg * 128;
      const uint a0 = intel_sub_group_block_read((__global const uint*)ap);
      const uint a1 = intel_sub_group_block_read((__global const uint*)(ap + 64));
      __global const float* sp = hsc + (size_t)kg * 4 * S + row;
      float v;
      GROUP_DOT(dw, a0, a1, sp[0], sp[S], sp[2 * S], sp[3 * S], v);
      r[m] = fma(ws, v, r[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
    if (m < nr) y[(size_t)(r0 + m) * N + g * SG + l] = swt[r0 + m] * r[m];
}

// x[t] += sigmoid-gated shared expert + the token's NU expert outputs. grid (N, T)
__kernel void moe_combine(__global float* x, __global const float* sh, __global const float* sgate,
                          __global const float* y, __global const int* inv, int N) {
  const int c = get_global_id(0), t = get_global_id(1);
  float v = 0.0f;
#pragma unroll
  for (int k = 0; k < NU; ++k) v += y[(size_t)inv[t * NU + k] * N + c];
  x[(size_t)t * N + c] += v + sgate[t] * sh[(size_t)t * N + c];
}
