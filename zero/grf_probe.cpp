// Does Level Zero run an OpenCL-compiled native kernel with the large register file (-cl-intel-256-GRF-per-thread)?
// A kernel that keeps 96 floats per work-item (192 GRF of 32 bytes at SIMD16) is built with and without the option,
// timed through OpenCL, a Level Zero immediate command list and a regular command list + queue, and its spill size is
// read from both APIs. usage: grf_probe
#include "../src/clrt.h"
#include "ze_min.h"
#include <chrono>
#include <cstring>
#include <vector>

static const char* kSrc = R"CLC(
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hog(__global float* out, int iters) {
  const int g = get_global_id(0);
  float a[96];
#pragma unroll
  for (int i = 0; i < 96; ++i) a[i] = (float)(g + i) * 0.001f;
  for (int k = 0; k < iters; ++k) {
#pragma unroll
    for (int i = 0; i < 96; ++i) a[i] = fma(a[i], 0.9999f, a[(i + 37) % 96] * 0.0001f);
  }
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < 96; ++i) s += a[i];
  out[g] = s;
}
)CLC";

typedef ze_result_t (*PFN_qcreate)(ze_context_handle_t, ze_device_handle_t, const ze_command_queue_desc_t*, void**);
struct ze_command_list_desc_t { uint32_t stype; const void* pNext; uint32_t ordinal; uint32_t flags; };
typedef ze_result_t (*PFN_lcreate)(ze_context_handle_t, ze_device_handle_t, const ze_command_list_desc_t*, ze_command_list_handle_t*);
typedef ze_result_t (*PFN_lclose)(ze_command_list_handle_t);
typedef ze_result_t (*PFN_qexec)(void*, uint32_t, ze_command_list_handle_t*, void*);
typedef ze_result_t (*PFN_qsync)(void*, uint64_t);

static const size_t kGlobal = 112 * 8 * 16 * 4, kLocal = 64;
static const int kIters = 4000;

static std::string read_all(const char* p) {
  FILE* f = fopen(p, "rb");
  if (!f) return "";
  std::string s;
  char b[65536];
  size_t n;
  while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
  fclose(f);
  return s;
}

int main(int argc, char** argv) {
  ClRt cl;
  if (!cl.init(false)) return 1;
  if (!ze_load()) return 1;
  if (argc >= 4) {  // grf_probe FILE.cl KERNEL "OPTIONS": spill of that kernel through OpenCL and Level Zero
    const std::string src = read_all(argv[1]);
    cl_program prog = cl.build(src, argv[3]);
    cl_int e2;
    cl_kernel ck = clCreateKernel(prog, argv[2], &e2);
    size_t spill = 0;
    clGetKernelWorkGroupInfo(ck, cl.device, 0x4109, sizeof(spill), &spill, nullptr);
    size_t bn = 0;
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bn), &bn, nullptr);
    std::vector<uint8_t> bin(bn);
    uint8_t* bp = bin.data();
    clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bp), &bp, nullptr);
    ZE_CHECK(zeInit(1));
    uint32_t n1 = 1;
    ze_driver_handle_t d1;
    ZE_CHECK(zeDriverGet(&n1, &d1));
    ze_device_handle_t v1;
    n1 = 1;
    ZE_CHECK(zeDeviceGet(d1, &n1, &v1));
    ze_context_handle_t c1;
    ze_context_desc_t cd1{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
    ZE_CHECK(zeContextCreate(d1, &cd1, &c1));
    ze_module_desc_t md{ZE_STYPE_MODULE_DESC, nullptr, 1, bn, bin.data(), "", nullptr};
    ze_module_handle_t mod;
    ZE_CHECK(zeModuleCreate(c1, v1, &md, &mod, nullptr));
    ze_kernel_desc_t kd{ZE_STYPE_KERNEL_DESC, nullptr, 0, argv[2]};
    ze_kernel_handle_t k;
    ZE_CHECK(zeKernelCreate(mod, &kd, &k));
    ze_kernel_properties_t kp{};
    kp.stype = 0x1e;
    zeKernelGetProperties(k, &kp);
    printf("%s: OpenCL spill %zu | Level Zero spill %u, private %u\n", argv[2], spill, kp.spillMemSize, kp.privateMemSize);
    return 0;
  }
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
  HMODULE zl = GetModuleHandleA("ze_loader.dll");
  auto qcreate = (PFN_qcreate)GetProcAddress(zl, "zeCommandQueueCreate");
  auto lcreate = (PFN_lcreate)GetProcAddress(zl, "zeCommandListCreate");
  auto lclose = (PFN_lclose)GetProcAddress(zl, "zeCommandListClose");
  auto qexec = (PFN_qexec)GetProcAddress(zl, "zeCommandQueueExecuteCommandLists");
  auto qsync = (PFN_qsync)GetProcAddress(zl, "zeCommandQueueSynchronize");
  ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0, 0, 0};
  ze_command_list_handle_t imm;
  ZE_CHECK(zeCommandListCreateImmediate(ctx, dev, &qd, &imm));
  void* q = nullptr;
  ZE_CHECK(qcreate(ctx, dev, &qd, &q));
  void* dout = nullptr;
  ze_device_mem_alloc_desc_t dd{ZE_STYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ZE_CHECK(zeMemAllocDevice(ctx, &dd, kGlobal * 4, 64, dev, &dout));
  cl_int err;
  cl_mem cout = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, kGlobal * 4, nullptr, &err);

  for (const char* opt : {"-cl-std=CL3.0", "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread"}) {
    cl_program prog = cl.build(kSrc, opt);
    cl_kernel ck = clCreateKernel(prog, "hog", &err);
    size_t spill = 0;
    clGetKernelWorkGroupInfo(ck, cl.device, 0x4109 /* CL_KERNEL_SPILL_MEM_SIZE_INTEL */, sizeof(spill), &spill, nullptr);
    int iters = kIters;
    clSetKernelArg(ck, 0, sizeof(cout), &cout);
    clSetKernelArg(ck, 1, sizeof(int), &iters);
    double ocl = 1e30;
    for (int r = 0; r < 4; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      clEnqueueNDRangeKernel(cl.queue, ck, 1, nullptr, &kGlobal, &kLocal, 0, nullptr, nullptr);
      clFinish(cl.queue);
      ocl = std::min(ocl, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    size_t bn = 0;
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bn), &bn, nullptr);
    std::vector<uint8_t> bin(bn);
    uint8_t* bp = bin.data();
    clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bp), &bp, nullptr);
    double zimm = 1e30, zreg = 1e30;
    uint32_t zspill = 0;
    for (const char* bf : {"", "-ze-opt-large-register-file"}) {
      ze_module_desc_t md{ZE_STYPE_MODULE_DESC, nullptr, 1, bn, bin.data(), bf, nullptr};
      ze_module_handle_t mod;
      ZE_CHECK(zeModuleCreate(ctx, dev, &md, &mod, nullptr));
      ze_kernel_desc_t kd{ZE_STYPE_KERNEL_DESC, nullptr, 0, "hog"};
      ze_kernel_handle_t k;
      ZE_CHECK(zeKernelCreate(mod, &kd, &k));
      ze_kernel_properties_t kp{};
      kp.stype = 0x1e;
      zeKernelGetProperties(k, &kp);
      zspill = kp.spillMemSize;
      ZE_CHECK(zeKernelSetGroupSize(k, (uint32_t)kLocal, 1, 1));
      ZE_CHECK(zeKernelSetArgumentValue(k, 0, sizeof(dout), &dout));
      ZE_CHECK(zeKernelSetArgumentValue(k, 1, sizeof(int), &iters));
      ze_group_count_t gc{(uint32_t)(kGlobal / kLocal), 1, 1};
      for (int r = 0; r < 4; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        ZE_CHECK(zeCommandListAppendLaunchKernel(imm, k, &gc, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListHostSynchronize(imm, UINT64_MAX));
        zimm = std::min(zimm, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      ze_command_list_desc_t ld{0xf, nullptr, 0, 0};
      ze_command_list_handle_t reg;
      ZE_CHECK(lcreate(ctx, dev, &ld, &reg));
      ZE_CHECK(zeCommandListAppendLaunchKernel(reg, k, &gc, nullptr, 0, nullptr));
      ZE_CHECK(lclose(reg));
      for (int r = 0; r < 4; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        ZE_CHECK(qexec(q, 1, &reg, nullptr));
        ZE_CHECK(qsync(q, UINT64_MAX));
        zreg = std::min(zreg, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      printf("%-48s | OpenCL %.2f ms spill %zu | L0 module flags '%s': spill %u, immediate %.2f ms, regular %.2f ms\n",
             opt, ocl, spill, bf, zspill, zimm, zreg);
      zimm = zreg = 1e30;
    }
  }
  return 0;
}
