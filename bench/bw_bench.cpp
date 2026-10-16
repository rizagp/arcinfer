// Measures achievable GPU read bandwidth from system memory (the ceiling for token generation).
#include "../src/clrt.h"
#include <algorithm>

static const char* kSrc = R"CLC(
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rd_block(__global const uint* p, __global uint* out, int per_item) {
  // each sub-group streams a contiguous region with block reads
  const size_t sg = get_global_id(0) / 16;
  __global const uint* q = p + sg * (size_t)per_item * 16;
  uint acc = 0;
  for (int i = 0; i < per_item; i += 8) {
    const uint8 v = intel_sub_group_block_read8(q + (size_t)i * 16);
    acc ^= v.s0 ^ v.s1 ^ v.s2 ^ v.s3 ^ v.s4 ^ v.s5 ^ v.s6 ^ v.s7;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
__kernel void rd_vec(__global const uint4* p, __global uint* out, int per_item) {
  // each work-item reads a contiguous chunk with 16-byte loads
  const size_t g = get_global_id(0);
  __global const uint4* q = p + g * (size_t)per_item;
  uint4 acc = 0;
  for (int i = 0; i < per_item; ++i) acc ^= q[i];
  if (acc.x == 0x12345678u) out[0] = acc.y;
}
)CLC";

int main() {
  ClRt cl;
  if (!cl.init()) return 1;
  auto prog = cl.build(kSrc, "-cl-std=CL3.0");
  const size_t bytes = (size_t)1 << 30;
  auto buf = cl.buffer(CL_MEM_READ_WRITE, bytes);
  const uint32_t one = 1;
  CL_CHECK(clEnqueueFillBuffer(cl.queue, buf, &one, 4, 0, bytes, 0, nullptr, nullptr));
  auto out = cl.buffer(CL_MEM_READ_WRITE, 64);
  for (const char* name : {"rd_block", "rd_vec"}) {
    auto k = cl.kernel(prog, name);
    for (int per : {64, 256, 1024}) {
      size_t items = std::string(name) == "rd_block" ? bytes / 4 / per : bytes / 16 / per;
      int p = per;
      set_args(k, 0, buf, out, p);
      size_t lws = 64;
      double best = 1e30;
      for (int r = 0; r < 5; ++r) {
        cl_event e;
        CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 1, nullptr, &items, &lws, 0, nullptr, &e));
        best = std::min(best, ClRt::event_ms(e));
        clReleaseEvent(e);
      }
      printf("%-9s per-item %-5d %7.2f ms  %6.1f GB/s\n", name, per, best, bytes / best * 1e-6);
    }
  }
  return 0;
}
