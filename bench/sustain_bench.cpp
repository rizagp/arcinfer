// Runs the same DP4a peak kernel continuously and prints throughput per second, to observe
// power/thermal throttling of the iGPU under sustained load.
// usage: sustain_bench [seconds]
#include "../src/clrt.h"
#include <chrono>

int main(int argc, char** argv) {
  const int seconds = argc > 1 ? atoi(argv[1]) : 90;
  ClRt cl;
  if (!cl.init()) return 1;
  auto prog = cl.build(read_file("kernels/peak.cl"), "-cl-std=CL3.0 -cl-mad-enable");
  auto k = cl.kernel(prog, "peak_ss_bc_a");
  const size_t threads = 112 * 8 * 16 * 4;
  const int iters = 4096;
  auto out = cl.buffer(CL_MEM_READ_WRITE, threads * 4);
  cl_uint seed = 7;
  CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &out));
  CL_CHECK(clSetKernelArg(k, 1, sizeof(int), &iters));
  CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_uint), &seed));
  const double ops = (double)threads * iters * 16 * 8;
  auto t_start = std::chrono::steady_clock::now();
  int sec = 0;
  double acc_ms = 0;
  int n = 0;
  while (true) {
    size_t gws = threads, lws = 256;
    cl_event e;
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 1, nullptr, &gws, &lws, 0, nullptr, &e));
    acc_ms += ClRt::event_ms(e);
    ++n;
    clReleaseEvent(e);
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    if (el >= sec + 5) {
      printf("t=%3ds  %.2f TOPS\n", sec + 5, ops * n / acc_ms * 1e-9);
      fflush(stdout);
      sec += 5;
      acc_ms = 0;
      n = 0;
      if (sec >= seconds) break;
    }
  }
  return 0;
}
