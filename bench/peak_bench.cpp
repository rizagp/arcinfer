// Measures sustained DP4a / FMA throughput of the iGPU.
#include "../src/clrt.h"
#include <algorithm>

int main() {
  ClRt cl;
  if (!cl.init()) return 1;
  auto prog = cl.build(read_file("kernels/peak.cl"), "-cl-std=CL3.0 -cl-mad-enable");
  const size_t threads = 112 * 8 * 16 * 4;  // EUs x HW threads x SIMD16 lanes x oversubscription
  const int iters = 4096;
  auto out = cl.buffer(CL_MEM_READ_WRITE, threads * 4);
  struct { const char* name; double ops_per_iter; } ks[] = {
      {"peak_dp4a", 16 * 8.0}, {"peak_dp4a_bcast", 16 * 8.0}, {"peak_fma_f16", 16 * 2.0}, {"peak_fma_f32", 16 * 2.0},
      {"peak_dp4a_ss", 16 * 8.0}, {"peak_dp4a_accsat", 16 * 8.0}, {"peak_dp4a_simd8", 16 * 8.0}, {"peak_dp4a_32acc", 32 * 8.0 / 2},
      {"peak_ss_bc_a", 16 * 8.0}, {"peak_ss_bc_b", 16 * 8.0}, {"peak_ss_bc8", 16 * 8.0}, {"peak_ss_bc_shared", 16 * 8.0}};
  for (auto& kd : ks) {
    auto k = cl.kernel(prog, kd.name);
    cl_uint seed = 12345;
    float fseed = 0.5f;
    CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &out));
    CL_CHECK(clSetKernelArg(k, 1, sizeof(int), &iters));
    if (kd.name[5] == 'd') CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_uint), &seed));
    else CL_CHECK(clSetKernelArg(k, 2, sizeof(float), &fseed));
    size_t gws = threads, lws = 256;
    double best = 1e30;
    for (int r = 0; r < 8; ++r) {
      cl_event e;
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 1, nullptr, &gws, &lws, 0, nullptr, &e));
      best = std::min(best, ClRt::event_ms(e));
      clReleaseEvent(e);
    }
    double ops = (double)threads * iters * kd.ops_per_iter;
    printf("%-16s %8.3f ms  %6.2f T(FL)OPS\n", kd.name, best, ops / best * 1e-9);
  }
  return 0;
}
