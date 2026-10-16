// Micro-kernels to measure the real DP4a / FMA ceiling of the iGPU at sustained clocks.
#define NACC 16

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_dp4a(__global int* out, int iters, uint seed) {
  uint x = seed ^ get_global_id(0);
  int acc[NACC];
  uint b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) { acc[i] = 0; b[i] = x * (i + 1); }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] += dot_4x8packed_su_int(b[i], x);
    x += 0x01010101u;
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}

// Same, but one operand is a lane broadcast (as in the GEMM inner loop).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_dp4a_bcast(__global int* out, int iters, uint seed) {
  uint x = seed ^ get_global_id(0);
  int acc[NACC];
  uint b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) { acc[i] = 0; b[i] = x * (i + 1); }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] += dot_4x8packed_su_int(intel_sub_group_shuffle(x, i), b[i]);
    x += 0x01010101u;
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_fma_f16(__global half* out, int iters, float seed) {
  half x = (half)(seed + get_global_id(0) * 1e-3f);
  half acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = (half)(i * 1e-3f);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = fma(acc[i], x, (half)1e-4f);
  }
  half s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_fma_f32(__global float* out, int iters, float seed) {
  float x = seed + get_global_id(0) * 1e-3f;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = i * 1e-3f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = fma(acc[i], x, 1e-4f);
  }
  float s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}

// Variants to find which DP4a form issues at full rate.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_dp4a_ss(__global int* out, int iters, uint seed) {
  uint x = seed ^ get_global_id(0);
  int acc[NACC];
  uint b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) { acc[i] = 0; b[i] = x * (i + 1); }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] += dot_4x8packed_ss_int(b[i], x);
    x += 0x01010101u;
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_dp4a_accsat(__global int* out, int iters, uint seed) {
  uint x = seed ^ get_global_id(0);
  int acc[NACC];
  uint b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) { acc[i] = 0; b[i] = x * (i + 1); }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = dot_acc_sat_4x8packed_ss_int(b[i], x, acc[i]);
    x += 0x01010101u;
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void peak_dp4a_simd8(__global int* out, int iters, uint seed) {
  uint x = seed ^ get_global_id(0);
  int acc[NACC];
  uint b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) { acc[i] = 0; b[i] = x * (i + 1); }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] += dot_4x8packed_ss_int(b[i], x);
    x += 0x01010101u;
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}

// 32 independent accumulators: checks whether DP4a latency (not issue rate) is the limit.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_dp4a_32acc(__global int* out, int iters, uint seed) {
  uint x = seed ^ get_global_id(0);
  int acc[32];
  uint b[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) b[i] = x * (i + 1);
#pragma unroll
  for (int i = 0; i < 32; ++i) acc[i] = 0;
  for (int it = 0; it < iters / 2; ++it) {
#pragma unroll
    for (int i = 0; i < 32; ++i) acc[i] += dot_4x8packed_ss_int(b[i % 8], x + i);
    x += 0x01010101u;
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}
// Broadcast operand placement (signed DP4a), SIMD16 and SIMD8.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_ss_bc_a(__global int* out, int iters, uint seed) {
  uint x = seed ^ get_global_id(0);
  int acc[NACC];
  uint b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) { acc[i] = 0; b[i] = x * (i + 1); }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] += dot_4x8packed_ss_int(intel_sub_group_shuffle(x, i), b[i]);
    x += 0x01010101u;
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_ss_bc_b(__global int* out, int iters, uint seed) {
  uint x = seed ^ get_global_id(0);
  int acc[NACC];
  uint b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) { acc[i] = 0; b[i] = x * (i + 1); }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] += dot_4x8packed_ss_int(b[i], intel_sub_group_shuffle(x, i));
    x += 0x01010101u;
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void peak_ss_bc8(__global int* out, int iters, uint seed) {
  uint x = seed ^ get_global_id(0);
  int acc[NACC];
  uint b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) { acc[i] = 0; b[i] = x * (i + 1); }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] += dot_4x8packed_ss_int(b[i], intel_sub_group_shuffle(x, i % 8));
    x += 0x01010101u;
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}
// One shared broadcast value feeding several DP4a (as in a GEMM with TN columns per lane).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_ss_bc_shared(__global int* out, int iters, uint seed) {
  uint x = seed ^ get_global_id(0);
  int acc[NACC];
  uint b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) { acc[i] = 0; b[i] = x * (i + 1); }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; i += 4) {
      const uint bx = intel_sub_group_shuffle(x, i);
      acc[i] += dot_4x8packed_ss_int(b[i], bx);
      acc[i + 1] += dot_4x8packed_ss_int(b[i + 1], bx);
      acc[i + 2] += dot_4x8packed_ss_int(b[i + 2], bx);
      acc[i + 3] += dot_4x8packed_ss_int(b[i + 3], bx);
    }
    x += 0x01010101u;
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[get_global_id(0)] = s;
}