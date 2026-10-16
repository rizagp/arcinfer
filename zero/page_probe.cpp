// Does the GPU page size matter for reading model weights? A buffer is allocated twice: as a plain device allocation
// (zeMemAllocDevice, the driver picks the pages) and through the virtual-memory API (zeVirtualMemReserve +
// zePhysicalMemCreate + zeVirtualMemMap) with the page size the driver reports for that size. Both are read by the same
// kernels: sequentially (coalesced, the GEMV pattern) and in 256-byte pieces at random offsets (expert rows spread over
// GBs, the pattern that misses the GPU's TLB most). usage: page_probe [MiB, default 2048]
#include "../src/clrt.h"
#include "ze_min.h"
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

static const char* kSrc = R"CLC(
// every work-item reads uint4s with a stride of the whole grid: a coalesced sweep of the buffer
__kernel void seq_read(__global const uint4* p, ulong n16, __global uint* out) {
  const size_t g = get_global_id(0), G = get_global_size(0);
  uint4 a = (uint4)(0);
  for (size_t i = g; i < n16; i += G) a ^= p[i];
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x12345678u) out[0] = 1;  // (keeps the reads)
}
// each sub-group of 16 reads 256 contiguous bytes at a pseudo-random 256-byte-aligned offset, `iters` times
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rand_read(__global const uint4* p, ulong nblk, int iters, __global uint* out) {
  const uint sg = get_global_id(0) / 16, lane = get_sub_group_local_id();
  uint h = sg * 2654435761u + 12345u;
  uint4 a = (uint4)(0);
  for (int it = 0; it < iters; ++it) {
    h ^= h << 13; h ^= h >> 17; h ^= h << 5;
    const ulong b = ((ulong)h * nblk) >> 32;  // block index in [0, nblk)
    a ^= p[b * 16 + lane];
  }
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x12345678u) out[0] = 1;
}
// as rand_read, but each pick reads `rows` consecutive 256-byte rows (a chunk of rows * 256 bytes)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void chunk_read(__global const uint4* p, ulong nchunk, int rows, int picks, __global uint* out) {
  const uint sg = get_global_id(0) / 16, lane = get_sub_group_local_id();
  uint h = sg * 2654435761u + 12345u;
  uint4 a = (uint4)(0);
  for (int it = 0; it < picks; ++it) {
    h ^= h << 13; h ^= h >> 17; h ^= h << 5;
    const ulong c = ((ulong)h * nchunk) >> 32;
    __global const uint4* q = p + c * rows * 16 + lane;
    for (int r = 0; r < rows; ++r) a ^= q[r * 16];
  }
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x12345678u) out[0] = 1;
}
)CLC";

#define ZE_STYPE_PHYSICAL_MEM_DESC 0x20u
typedef struct _ze_physical_mem_handle_t* ze_physical_mem_handle_t;
struct ze_physical_mem_desc_t { uint32_t stype; const void* pNext; uint32_t flags; size_t size; };
typedef ze_result_t(__stdcall* PFN_qps)(ze_context_handle_t, ze_device_handle_t, size_t, size_t*);
typedef ze_result_t(__stdcall* PFN_vres)(ze_context_handle_t, const void*, size_t, void**);
typedef ze_result_t(__stdcall* PFN_pcre)(ze_context_handle_t, ze_device_handle_t, ze_physical_mem_desc_t*, ze_physical_mem_handle_t*);
typedef ze_result_t(__stdcall* PFN_vmap)(ze_context_handle_t, const void*, size_t, ze_physical_mem_handle_t, size_t, uint32_t);
typedef ze_result_t(__stdcall* PFN_vunmap)(ze_context_handle_t, const void*, size_t);
typedef ze_result_t(__stdcall* PFN_pdes)(ze_context_handle_t, ze_physical_mem_handle_t);
typedef ze_result_t(__stdcall* PFN_vfree)(ze_context_handle_t, const void*, size_t);

int main(int argc, char** argv) {
  const size_t mib = argc > 1 ? (size_t)atoll(argv[1]) : 2048, bytes = mib << 20;
  ClRt cl;
  if (!cl.init(false) || !ze_load()) return 1;
  cl_program prog = cl.build(kSrc, "-cl-std=CL3.0");
  size_t bn = 0;
  clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bn), &bn, nullptr);
  std::vector<uint8_t> bin(bn);
  uint8_t* bp = bin.data();
  clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bp), &bp, nullptr);

  ZE_CHECK(zeInit(1));
  uint32_t n = 1;
  ze_driver_handle_t drv;
  ZE_CHECK(zeDriverGet(&n, &drv));
  ze_device_handle_t dev;
  n = 1;
  ZE_CHECK(zeDeviceGet(drv, &n, &dev));
  ze_context_handle_t ctx;
  ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
  ZE_CHECK(zeContextCreate(drv, &cd, &ctx));
  ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0, 0, 0};
  ze_command_list_handle_t cmd;
  ZE_CHECK(zeCommandListCreateImmediate(ctx, dev, &qd, &cmd));
  ze_module_desc_t md{ZE_STYPE_MODULE_DESC, nullptr, 1, bn, bin.data(), "", nullptr};
  ze_module_handle_t mod;
  ZE_CHECK(zeModuleCreate(ctx, dev, &md, &mod, nullptr));
  auto mk = [&](const char* name) {
    ze_kernel_desc_t kd{ZE_STYPE_KERNEL_DESC, nullptr, 0, name};
    ze_kernel_handle_t k;
    ZE_CHECK(zeKernelCreate(mod, &kd, &k));
    return k;
  };
  ze_kernel_handle_t kseq = mk("seq_read"), krand = mk("rand_read");

  HMODULE zl = GetModuleHandleA("ze_loader.dll");
  auto qps = (PFN_qps)GetProcAddress(zl, "zeVirtualMemQueryPageSize");
  auto vres = (PFN_vres)GetProcAddress(zl, "zeVirtualMemReserve");
  auto pcre = (PFN_pcre)GetProcAddress(zl, "zePhysicalMemCreate");
  auto vmap = (PFN_vmap)GetProcAddress(zl, "zeVirtualMemMap");
  auto vunmap = (PFN_vunmap)GetProcAddress(zl, "zeVirtualMemUnmap");
  auto pdes = (PFN_pdes)GetProcAddress(zl, "zePhysicalMemDestroy");
  auto vfree = (PFN_vfree)GetProcAddress(zl, "zeVirtualMemFree");
  if (!qps || !vres || !pcre || !vmap) { printf("virtual-memory API missing in ze_loader.dll\n"); return 1; }
  for (size_t s : {(size_t)4096, (size_t)65536, (size_t)2 << 20, (size_t)64 << 20, bytes}) {
    size_t ps = 0;
    const ze_result_t r = qps(ctx, dev, s, &ps);
    printf("page size for a %zu KiB allocation: %zu KiB (result 0x%x)\n", s >> 10, ps >> 10, r);
  }

  void* out = nullptr;
  ze_device_mem_alloc_desc_t dd{ZE_STYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ZE_CHECK(zeMemAllocDevice(ctx, &dd, 64, 64, dev, &out));

  auto run = [&](ze_kernel_handle_t k, uint32_t groups, uint32_t lsz) {
    ZE_CHECK(zeKernelSetGroupSize(k, lsz, 1, 1));
    ze_group_count_t gc{groups, 1, 1};
    const auto t0 = std::chrono::steady_clock::now();
    ZE_CHECK(zeCommandListAppendLaunchKernel(cmd, k, &gc, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(cmd, UINT64_MAX));
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };
  // (spans for the random reads: all of the buffer, and 256 MiB)
  auto measure = [&](const char* label, void* p) {
    const uint8_t pat = 0x5a;
    ZE_CHECK(zeCommandListAppendMemoryFill(cmd, p, &pat, 1, bytes, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(cmd, UINT64_MAX));
    const uint64_t n16 = bytes / 16;
    ZE_CHECK(zeKernelSetArgumentValue(kseq, 0, sizeof(p), &p));
    ZE_CHECK(zeKernelSetArgumentValue(kseq, 1, sizeof(n16), &n16));
    ZE_CHECK(zeKernelSetArgumentValue(kseq, 2, sizeof(out), &out));
    const uint32_t sgroups = 112 * 8 * 4;  // (work-groups of 256: enough threads to saturate memory)
    double best = 1e9;
    for (int r = 0; r < 5; ++r) best = std::min(best, run(kseq, sgroups, 256));
    printf("%-22s sequential: %6.1f GB/s\n", label, bytes / best / 1e9);
    for (size_t span : {bytes, std::min(bytes, (size_t)256 << 20)}) {
      const uint64_t nblk = span / 256;
      const int iters = 256;
      const uint32_t rgroups = 112 * 8 * 8;  // 256 work-items = 16 sub-groups per group
      ZE_CHECK(zeKernelSetArgumentValue(krand, 0, sizeof(p), &p));
      ZE_CHECK(zeKernelSetArgumentValue(krand, 1, sizeof(nblk), &nblk));
      ZE_CHECK(zeKernelSetArgumentValue(krand, 2, sizeof(iters), &iters));
      ZE_CHECK(zeKernelSetArgumentValue(krand, 3, sizeof(out), &out));
      best = 1e9;
      for (int r = 0; r < 5; ++r) best = std::min(best, run(krand, rgroups, 256));
      const double moved = (double)rgroups * 16 * iters * 256;
      printf("%-22s random 256 B over %5zu MiB: %6.1f GB/s\n", label, span >> 20, moved / best / 1e9);
    }
  };

  void* plain = nullptr;
  ZE_CHECK(zeMemAllocDevice(ctx, &dd, bytes, 4096, dev, &plain));
  if (argc > 2 && std::string(argv[2]) == "sweep") {  // page_probe MiB sweep: random reads over growing spans
    ze_kernel_handle_t kc = mk("chunk_read");
    const uint8_t pat = 0x5a;
    ZE_CHECK(zeCommandListAppendMemoryFill(cmd, plain, &pat, 1, bytes, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(cmd, UINT64_MAX));
    printf("random reads (GB/s) by span, per chunk size\n%10s %9s %9s %9s\n", "span MiB", "256 B", "4 KiB", "64 KiB");
    std::vector<size_t> spans;
    for (size_t s = 4; s <= mib; s *= 2) {
      spans.push_back(s);
      if (s >= 512 && s * 3 / 2 <= mib) spans.push_back(s * 3 / 2);  // (finer steps around the drop)
    }
    std::sort(spans.begin(), spans.end());
    for (size_t s : spans) {
      printf("%10zu", s);
      for (int rows : {1, 16, 256}) {
        const uint64_t nchunk = (s << 20) / (256ull * rows);
        const int picks = std::max(1, 256 / rows);  // (the same bytes per sub-group for every chunk size)
        const uint32_t groups = 112 * 8 * 8;
        ZE_CHECK(zeKernelSetArgumentValue(kc, 0, sizeof(plain), &plain));
        ZE_CHECK(zeKernelSetArgumentValue(kc, 1, sizeof(nchunk), &nchunk));
        ZE_CHECK(zeKernelSetArgumentValue(kc, 2, sizeof(rows), &rows));
        ZE_CHECK(zeKernelSetArgumentValue(kc, 3, sizeof(picks), &picks));
        ZE_CHECK(zeKernelSetArgumentValue(kc, 4, sizeof(out), &out));
        double best = 1e9;
        for (int r = 0; r < 4; ++r) best = std::min(best, run(kc, groups, 256));
        const double moved = (double)groups * 16 * picks * rows * 256;
        printf(" %9.1f", moved / best / 1e9);
      }
      printf("\n");
      fflush(stdout);
    }
    return 0;
  }
  measure("zeMemAllocDevice", plain);
  ZE_CHECK(zeMemFree(ctx, plain));

  size_t ps = 0;
  qps(ctx, dev, bytes, &ps);
  // (physical memory in pieces: one piece of the whole size is refused with UNSUPPORTED_SIZE)
  const size_t piece = argc > 2 ? (size_t)atoll(argv[2]) << 20 : (size_t)256 << 20;
  const size_t vbytes = (bytes + piece - 1) / piece * piece;
  void* vptr = nullptr;
  std::vector<ze_physical_mem_handle_t> phs;
  ze_result_t r = vres(ctx, nullptr, vbytes, &vptr);
  if (r != 0) printf("zeVirtualMemReserve: 0x%x\n", r);
  for (size_t off = 0; r == 0 && off < vbytes; off += piece) {
    ze_physical_mem_desc_t pd{ZE_STYPE_PHYSICAL_MEM_DESC, nullptr, 0, piece};
    ze_physical_mem_handle_t ph = nullptr;
    r = pcre(ctx, dev, &pd, &ph);
    if (r != 0) { printf("zePhysicalMemCreate (%zu MiB): 0x%x\n", piece >> 20, r); break; }
    phs.push_back(ph);
    r = vmap(ctx, (uint8_t*)vptr + off, piece, ph, 0, 1 /* read-write */);
    if (r != 0) printf("zeVirtualMemMap: 0x%x\n", r);
  }
  if (r == 0) {
    const std::string label = "virtual, " + std::to_string(ps >> 10) + " KiB pages";
    measure(label.c_str(), vptr);
  }
  if (vptr) vunmap(ctx, vptr, vbytes);
  for (auto ph : phs) pdes(ctx, ph);
  if (vptr) vfree(ctx, vptr, vbytes);
  return 0;
}
