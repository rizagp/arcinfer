// arcinferzero feasibility probe: an OpenCL C kernel compiled by the OpenCL driver, its native binary loaded into
// Level Zero; GPU read bandwidth of host / device / shared USM; and how much host memory the GPU can read beyond the
// "GPU memory" share of the Intel driver (the question SSD streaming of MoE experts depends on).
// usage: ze_probe [host GiB to try = 22]
#include "../src/clrt.h"
#include "ze_min.h"
#include <chrono>
#include <cstring>
#include <vector>

static const char* kSrc = R"CLC(
__kernel void bw(__global const uint4* src, ulong n, __global uint* out) {
  const size_t gid = get_global_id(0), gs = get_global_size(0);
  uint4 a = (uint4)(0);
  for (ulong i = gid; i < n; i += gs) a ^= src[i];
  out[gid] = a.x ^ a.y ^ a.z ^ a.w;
}
__kernel void fillk(__global uint* dst, ulong n) {
  for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) dst[i] = (uint)i * 2654435761u;
}
)CLC";

static ze_context_handle_t ctx;
static ze_device_handle_t dev;
static ze_command_list_handle_t cl0;
static ze_kernel_handle_t kbw, kfill;
static const uint32_t kGroups = 112 * 16, kWg = 256;  // 112 EUs x 16 work-groups of 256

static double gpu_read(void* p, size_t bytes, uint32_t* xr_out, int reps = 3) {
  static void* out = nullptr;
  if (!out) {
    ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    ZE_CHECK(zeMemAllocHost(ctx, &hd, (size_t)kGroups * kWg * 4, 64, &out));
  }
  const uint64_t n = bytes / 16;
  ZE_CHECK(zeKernelSetArgumentValue(kbw, 0, sizeof(void*), &p));
  ZE_CHECK(zeKernelSetArgumentValue(kbw, 1, sizeof(n), &n));
  ZE_CHECK(zeKernelSetArgumentValue(kbw, 2, sizeof(void*), &out));
  ZE_CHECK(zeKernelSetGroupSize(kbw, kWg, 1, 1));
  ze_group_count_t gc{kGroups, 1, 1};
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    ZE_CHECK(zeCommandListAppendLaunchKernel(cl0, kbw, &gc, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(cl0, UINT64_MAX));
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  uint32_t x = 0;
  const uint32_t* o = (const uint32_t*)out;
  for (size_t i = 0; i < (size_t)kGroups * kWg; ++i) x ^= o[i];
  if (xr_out) *xr_out = x;
  return bytes / best / 1e9;
}
static uint32_t cpu_fill_xor(uint32_t* p, size_t bytes) {
  uint32_t x = 0;
  for (size_t i = 0; i < bytes / 4; ++i) { p[i] = (uint32_t)i * 2654435761u; x ^= p[i]; }
  return x;
}

int main(int argc, char** argv) {
  const double try_gib = argc > 1 ? atof(argv[1]) : 22.0;
  // 1. compile with the OpenCL driver, take the native binary
  ClRt cl;
  if (!cl.init(false)) return 1;
  cl_program prog = cl.build(kSrc, "-cl-std=CL3.0");
  size_t bn = 0;
  clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bn), &bn, nullptr);
  std::vector<uint8_t> bin(bn);
  uint8_t* bp = bin.data();
  clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bp), &bp, nullptr);
  printf("OpenCL native binary: %zu bytes\n", bn);

  // 2. Level Zero: driver, device, context, immediate command list, module from that binary
  if (!ze_load()) { fprintf(stderr, "cannot load ze_loader.dll\n"); return 1; }
  ZE_CHECK(zeInit(1));
  uint32_t nd = 1;
  ze_driver_handle_t drv;
  ZE_CHECK(zeDriverGet(&nd, &drv));
  uint32_t ndev = 1;
  ZE_CHECK(zeDeviceGet(drv, &ndev, &dev));
  ze_device_properties_t dp{};
  dp.stype = ZE_STYPE_DEVICE_PROPERTIES;
  ZE_CHECK(zeDeviceGetProperties(dev, &dp));
  printf("Level Zero device: %s | %u slices x %u subslices x %u EUs, SIMD %u, %u threads/EU, %u MHz, max alloc %.2f GiB\n",
         dp.name, dp.numSlices, dp.numSubslicesPerSlice, dp.numEUsPerSubslice, dp.physicalEUSimdWidth, dp.numThreadsPerEU,
         dp.coreClockRate, dp.maxMemAllocSize / 1073741824.0);
  uint32_t nm = 1;
  ze_device_memory_properties_t mp{};
  mp.stype = ZE_STYPE_DEVICE_MEMORY_PROPERTIES;
  ZE_CHECK(zeDeviceGetMemoryProperties(dev, &nm, &mp));
  printf("device memory: %.2f GiB (%s)\n", mp.totalSize / 1073741824.0, mp.name);
  ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
  ZE_CHECK(zeContextCreate(drv, &cd, &ctx));
  ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0, 2, 0};
  ZE_CHECK(zeCommandListCreateImmediate(ctx, dev, &qd, &cl0));
  ze_module_desc_t md{ZE_STYPE_MODULE_DESC, nullptr, 1, bn, bin.data(), "", nullptr};
  ze_module_handle_t mod;
  ze_module_build_log_handle_t log = nullptr;
  const ze_result_t mr = zeModuleCreate(ctx, dev, &md, &mod, &log);
  if (mr != ZE_RESULT_SUCCESS) {
    size_t ls = 0;
    zeModuleBuildLogGetString(log, &ls, nullptr);
    std::string s(ls, 0);
    zeModuleBuildLogGetString(log, &ls, s.data());
    fprintf(stderr, "zeModuleCreate (native) failed 0x%x: %s\n", mr, s.c_str());
    return 1;
  }
  ze_kernel_desc_t kd{ZE_STYPE_KERNEL_DESC, nullptr, 0, "bw"};
  ZE_CHECK(zeKernelCreate(mod, &kd, &kbw));
  kd.pKernelName = "fillk";
  ZE_CHECK(zeKernelCreate(mod, &kd, &kfill));
  printf("native module loaded: OpenCL kernels run under Level Zero\n");

  // 3. bandwidth of the three kinds of USM (1 GiB each)
  const size_t G = (size_t)1 << 30;
  ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
  ze_device_mem_alloc_desc_t dd{ZE_STYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  {
    void* h = nullptr;
    ZE_CHECK(zeMemAllocHost(ctx, &hd, G, 4096, &h));
    const uint32_t want = cpu_fill_xor((uint32_t*)h, G);
    uint32_t got = 0;
    const double gbs = gpu_read(h, G, &got);
    printf("host USM   1 GiB: %6.1f GB/s  %s\n", gbs, got == want ? "(checksum OK)" : "(CHECKSUM WRONG)");
    ZE_CHECK(zeMemFree(ctx, h));
  }
  {
    void* d = nullptr;
    ZE_CHECK(zeMemAllocDevice(ctx, &dd, G, 4096, dev, &d));
    const uint64_t n = G / 4;
    ZE_CHECK(zeKernelSetArgumentValue(kfill, 0, sizeof(void*), &d));
    ZE_CHECK(zeKernelSetArgumentValue(kfill, 1, sizeof(n), &n));
    ZE_CHECK(zeKernelSetGroupSize(kfill, kWg, 1, 1));
    ze_group_count_t gc{kGroups, 1, 1};
    ZE_CHECK(zeCommandListAppendLaunchKernel(cl0, kfill, &gc, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(cl0, UINT64_MAX));
    printf("device USM 1 GiB: %6.1f GB/s\n", gpu_read(d, G, nullptr));
    ZE_CHECK(zeMemFree(ctx, d));
  }
  {
    void* s = nullptr;
    ZE_CHECK(zeMemAllocShared(ctx, &dd, &hd, G, 4096, dev, &s));
    const uint32_t want = cpu_fill_xor((uint32_t*)s, G);
    uint32_t got = 0;
    const double gbs = gpu_read(s, G, &got);
    printf("shared USM 1 GiB: %6.1f GB/s  %s\n", gbs, got == want ? "(checksum OK)" : "(CHECKSUM WRONG)");
    ZE_CHECK(zeMemFree(ctx, s));
  }

  // 4. host USM beyond the GPU memory share: allocate 1 GiB blocks, fill them on the CPU, then read the first and
  // the last on the GPU
  std::vector<void*> blocks;
  std::vector<uint32_t> want;
  for (int i = 0; i < (int)try_gib; ++i) {
    void* h = nullptr;
    if (zeMemAllocHost(ctx, &hd, G, 4096, &h) != ZE_RESULT_SUCCESS || !h) { printf("host USM: allocation %d failed\n", i + 1); break; }
    blocks.push_back(h);
    want.push_back(cpu_fill_xor((uint32_t*)h, G));
  }
  printf("host USM allocated: %zu GiB (GPU memory share %.2f GiB)\n", blocks.size(), mp.totalSize / 1073741824.0);
  for (size_t i : {(size_t)0, blocks.size() / 2, blocks.size() - 1}) {
    if (i >= blocks.size()) continue;
    uint32_t got = 0;
    const double gbs = gpu_read(blocks[i], G, &got, 1);
    printf("  GPU reads block %2zu: %6.1f GB/s  %s\n", i + 1, gbs, got == want[i] ? "(checksum OK)" : "(CHECKSUM WRONG)");
  }
  // all blocks in one pass (every one resident at once)
  {
    auto t0 = std::chrono::steady_clock::now();
    int ok = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
      uint32_t got = 0;
      gpu_read(blocks[i], G, &got, 1);
      ok += got == want[i];
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  GPU reads all %zu GiB: %.1f GB/s, %d blocks correct\n", blocks.size(), blocks.size() * (double)G / s / 1e9, ok);
  }
  for (void* h : blocks) zeMemFree(ctx, h);
  zeContextDestroy(ctx);
  return 0;
}
