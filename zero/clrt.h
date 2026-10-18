// arcinferzero: the subset of the OpenCL API the arcinfer engine uses, implemented on Level Zero, so that
// zero/engine.h (a copy of src/engine.h) runs unchanged on top of it. Buffers are device USM allocations (a sub-buffer is
// a pointer into its parent), commands go to one in-order immediate command list (the same order as arcinfer's in-order
// OpenCL queue, hence the same results), and kernels are the native binaries of the OpenCL compiler: built through
// OpenCL.dll (loaded under other names) and shared with arcinfer's kernel cache (%LOCALAPPDATA%\arcinfer\kernels).
#pragma once
#include "ze_min.h"
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint cl_bool;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_map_flags;
typedef cl_ulong cl_queue_properties;
typedef cl_uint cl_device_info;
typedef cl_uint cl_profiling_info;
typedef cl_uint cl_buffer_create_type;

struct ZMem {
  char* ptr;
  size_t size;
  bool owner;
};
struct ZKernel {
  ze_kernel_handle_t k;
  std::string name;
  uint32_t gx = 0, gy = 0, gz = 0;  // group size last set
};
struct ZProg { ze_module_handle_t m; bool indirect = false; std::string opts; };  // indirect: built with -DXCACHE (see kernel())
typedef ZMem* cl_mem;
typedef ZKernel* cl_kernel;
typedef ZProg* cl_program;
typedef struct _z_queue* cl_command_queue;
typedef struct _z_context* cl_context;
typedef struct _z_device* cl_device_id;
typedef struct _z_platform* cl_platform_id;
typedef struct _z_event* cl_event;

#define CL_SUCCESS 0
#define CL_TRUE 1
#define CL_FALSE 0
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_USE_HOST_PTR (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR (1 << 4)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_MAP_READ (1 << 0)
#define CL_MAP_WRITE (1 << 1)
#define CL_BUFFER_CREATE_TYPE_REGION 0x1220
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
struct cl_buffer_region { size_t origin, size; };

#define CL_CHECK(expr)                                                                       \
  do {                                                                                       \
    cl_int _e = (expr);                                                                      \
    if (_e != CL_SUCCESS) {                                                                  \
      fprintf(stderr, "arcinferzero error %d at %s:%d: %s\n", _e, __FILE__, __LINE__, #expr); \
      exit(1);                                                                               \
    }                                                                                        \
  } while (0)

// ---- the runtime state (one device, one context, one in-order immediate command list)
namespace zrt {
inline ze_context_handle_t ctx = nullptr;
inline ze_device_handle_t dev = nullptr;
inline ze_command_list_handle_t cl = nullptr;
inline std::unordered_set<ZMem*> mems;
inline uint64_t global_mem = 0;
inline uint64_t device_bytes = 0;  // device USM allocated by the engine
inline char* stage = nullptr;      // host USM bounce buffer for reads / writes
inline size_t stage_n = 0;
inline void sync() { ZE_CHECK(zeCommandListHostSynchronize(cl, UINT64_MAX)); }
// A second immediate command list for copies that run while `cl` executes kernels (the expert cache's prompt
// prefetch, xcache.h): on a copy engine when the device has one, else on another queue of the compute group.
inline ze_command_list_handle_t cl_copy = nullptr;
inline const char* cl_copy_kind = "";
inline ze_command_list_handle_t copy_list() {
  if (cl_copy) return cl_copy;
  uint32_t ng = 0;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(dev, &ng, nullptr));
  std::vector<ze_command_queue_group_properties_t> gp(ng);
  for (auto& g : gp) g.stype = ZE_STYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(dev, &ng, gp.data()));
  int ord = -1, idx = 0;
  for (uint32_t i = 0; i < ng && ord < 0; ++i)  // a copy-only group
    if ((gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) && !(gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE))
      ord = (int)i, cl_copy_kind = "copy engine";
  for (uint32_t i = 0; i < ng && ord < 0; ++i)  // else the compute group, its second queue when it has one
    if (gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)
      ord = (int)i, idx = gp[i].numQueues > 1 ? 1 : 0, cl_copy_kind = idx ? "second compute queue" : "compute queue";
  if (ord < 0) ord = 0;
  ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, (uint32_t)ord, (uint32_t)idx, 2 /* in order */,
                             2 /* asynchronous */, 0};
  ZE_CHECK(zeCommandListCreateImmediate(ctx, dev, &qd, &cl_copy));
  fprintf(stderr, "arcinferzero: expert copies on a %s (queue group %d, queue %d)\n", cl_copy_kind, ord, idx);
  return cl_copy;
}
// Events the copy list signals after a batch of copies: 0 and 1 (used alternately) the main list waits on before
// kernels read the copied memory (the wait, device scope, has the compute side invalidate its caches, which a host
// synchronization alone does not promise); 2 and 3 the copy thread waits on before it reuses a part of its staging.
inline ze_event_handle_t copy_ev[4] = {};
inline ze_event_handle_t copy_event(int i) {
  if (!copy_ev[0]) {
    ze_event_pool_desc_t pd{ZE_STYPE_EVENT_POOL_DESC, nullptr, 0x1 /* host visible */, 4};
    ze_event_pool_handle_t pool = nullptr;
    ZE_CHECK(zeEventPoolCreate(ctx, &pd, 1, &dev, &pool));
    for (uint32_t k = 0; k < 4; ++k) {
      ze_event_desc_t ed{ZE_STYPE_EVENT_DESC, nullptr, k, 0x4 /* signal: host scope */, 0x2 /* wait: device scope */};
      ZE_CHECK(zeEventCreate(pool, &ed, &copy_ev[k]));
    }
  }
  return copy_ev[i & 3];
}
// kernel timestamps for profiling (engine.profile: clEnqueueNDRangeKernel asked for an event): events come from pools
// of 4096 host-visible timestamp events and go back to a free list on clReleaseEvent
inline std::vector<ze_event_handle_t> ev_free;
inline double ev_ns_per_tick = 0;
inline uint64_t ev_mask = ~0ull;
inline ze_event_handle_t ev_get() {
  if (ev_free.empty()) {
    if (ev_ns_per_tick == 0) {
      ze_device_properties_t dp{};
      dp.stype = ZE_STYPE_DEVICE_PROPERTIES;  // (this stype: timerResolution in ns per tick)
      ZE_CHECK(zeDeviceGetProperties(dev, &dp));
      ev_ns_per_tick = dp.timerResolution ? (double)dp.timerResolution : 1.0;
      if (dp.kernelTimestampValidBits && dp.kernelTimestampValidBits < 64) ev_mask = (1ull << dp.kernelTimestampValidBits) - 1;
    }
    const uint32_t n = 4096;
    ze_event_pool_desc_t pd{ZE_STYPE_EVENT_POOL_DESC, nullptr, 0x1 | 0x4 /* host visible, kernel timestamps */, n};
    ze_event_pool_handle_t pool;
    ZE_CHECK(zeEventPoolCreate(ctx, &pd, 1, &dev, &pool));
    for (uint32_t i = 0; i < n; ++i) {
      ze_event_desc_t ed{ZE_STYPE_EVENT_DESC, nullptr, i, 0x4 /* host scope */, 0};
      ze_event_handle_t e;
      ZE_CHECK(zeEventCreate(pool, &ed, &e));
      ev_free.push_back(e);
    }
  }
  ze_event_handle_t e = ev_free.back();
  ev_free.pop_back();
  return e;
}
inline char* staging(size_t n) {
  if (n > stage_n) {
    if (stage) { sync(); zeMemFree(ctx, stage); }
    stage_n = std::max(n, (size_t)1 << 22);
    ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    void* p = nullptr;
    ZE_CHECK(zeMemAllocHost(ctx, &hd, stage_n, 4096, &p));
    stage = (char*)p;
  }
  return stage;
}
inline void* host_alloc(size_t n) {
  ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
  void* p = nullptr;
  ZE_CHECK(zeMemAllocHost(ctx, &hd, n, 4096, &p));
  return p;
}
// Called when the GPU runs out of memory (the share is shared with every other program: a browser or the desktop
// taking more makes a launch or allocation fail): frees some memory (the engine shrinks its expert cache) and returns
// true, or false when it cannot; the failed call is then retried.
inline bool (*oom_hook)() = nullptr;
inline void* device_alloc(size_t n) {
  ze_device_mem_alloc_desc_t dd{ZE_STYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void* p = nullptr;
  ze_result_t r = zeMemAllocDevice(ctx, &dd, std::max<size_t>(n, 64), 4096, dev, &p);
  for (int i = 0; r != ZE_RESULT_SUCCESS && i < 8 && oom_hook && oom_hook(); ++i)
    r = zeMemAllocDevice(ctx, &dd, std::max<size_t>(n, 64), 4096, dev, &p);
  if (r != ZE_RESULT_SUCCESS) {
    fprintf(stderr, "arcinferzero: device allocation of %.1f MB failed (0x%x; %.2f GiB allocated)\n", n / 1e6, r,
            device_bytes / 1073741824.0);
    exit(1);
  }
  device_bytes += n;
  return p;
}
// What Windows lets this process keep resident on the GPU now (WDDM budget, DXGI) and what it uses: the GPU share is
// an upper bound, the budget shrinks when the system is short of memory. Returns false without the numbers.
inline bool video_budget(uint64_t& budget, uint64_t& usage) {
  IDXGIFactory4* f = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&f))) return false;
  bool ok = false;
  IDXGIAdapter1* a1 = nullptr;
  for (UINT i = 0; !ok && f->EnumAdapters1(i, &a1) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 d;
    a1->GetDesc1(&d);
    IDXGIAdapter3* a3 = nullptr;
    if (d.VendorId == 0x8086 && SUCCEEDED(a1->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3))) {
      DXGI_QUERY_VIDEO_MEMORY_INFO m{};
      if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &m))) {
        budget = m.Budget;
        usage = m.CurrentUsage;
        ok = true;
      }
      a3->Release();
    }
    a1->Release();
  }
  f->Release();
  return ok;
}
inline int note_layer = -1, note_T = -1;  // (diagnostics: the layer and rows the engine is working on)
inline void print_memory() {
  if (note_layer >= 0) fprintf(stderr, "  (layer %d, %d rows)\n", note_layer, note_T);
  uint64_t b = 0, u = 0;
  MEMORYSTATUSEX ms{sizeof ms};
  GlobalMemoryStatusEx(&ms);
  if (video_budget(b, u))
    fprintf(stderr, "  GPU memory: Windows budget %.2f GiB, in use %.2f GiB (allocated by the engine %.2f GiB); RAM free %.2f GiB\n",
            b / 1073741824.0, u / 1073741824.0, device_bytes / 1073741824.0, ms.ullAvailPhys / 1073741824.0);
}
// memory the engine allocated itself (host USM, device pools) as a buffer kernels can take as an argument
inline ZMem* wrap(void* p, size_t n) {
  ZMem* m = new ZMem{(char*)p, n, false};
  mems.insert(m);
  return m;
}
struct Mapping { ZMem* m; size_t off, n; void* host; };
inline std::vector<Mapping> maps;

// OpenCL.dll, only to compile kernels (its native binaries load into Level Zero)
namespace rcl {
typedef cl_int(__stdcall* GetPlatformIDs)(cl_uint, void**, cl_uint*);
typedef cl_int(__stdcall* GetDeviceIDs)(void*, cl_ulong, cl_uint, void**, cl_uint*);
typedef cl_int(__stdcall* GetDeviceInfo)(void*, cl_uint, size_t, void*, size_t*);
typedef void*(__stdcall* CreateContext)(const intptr_t*, cl_uint, void* const*, void*, void*, cl_int*);
typedef void*(__stdcall* CreateProgramWithSource)(void*, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int(__stdcall* BuildProgram)(void*, cl_uint, void* const*, const char*, void*, void*);
typedef cl_int(__stdcall* GetProgramBuildInfo)(void*, void*, cl_uint, size_t, void*, size_t*);
typedef cl_int(__stdcall* GetProgramInfo)(void*, cl_uint, size_t, void*, size_t*);
typedef cl_int(__stdcall* ReleaseProgram)(void*);
inline GetPlatformIDs getPlatformIDs;
inline GetDeviceIDs getDeviceIDs;
inline GetDeviceInfo getDeviceInfo;
inline CreateContext createContext;
inline CreateProgramWithSource createProgramWithSource;
inline BuildProgram buildProgram;
inline GetProgramBuildInfo getProgramBuildInfo;
inline GetProgramInfo getProgramInfo;
inline ReleaseProgram releaseProgram;
inline void *device = nullptr, *context = nullptr;
inline bool load() {
  HMODULE h = LoadLibraryA("OpenCL.dll");
  if (!h) return false;
#define RCL(var, name) var = (decltype(var))(void*)GetProcAddress(h, name); if (!var) return false;
  RCL(getPlatformIDs, "clGetPlatformIDs") RCL(getDeviceIDs, "clGetDeviceIDs") RCL(getDeviceInfo, "clGetDeviceInfo")
  RCL(createContext, "clCreateContext") RCL(createProgramWithSource, "clCreateProgramWithSource")
  RCL(buildProgram, "clBuildProgram") RCL(getProgramBuildInfo, "clGetProgramBuildInfo")
  RCL(getProgramInfo, "clGetProgramInfo") RCL(releaseProgram, "clReleaseProgram")
#undef RCL
  cl_uint np = 0;
  if (getPlatformIDs(0, nullptr, &np) != 0 || !np) return false;
  std::vector<void*> plats(np);
  getPlatformIDs(np, plats.data(), nullptr);
  for (void* p : plats) {
    cl_uint nd = 0;
    if (getDeviceIDs(p, 1u << 2, 1, &device, &nd) == 0 && nd) break;
  }
  if (!device) return false;
  cl_int err = 0;
  context = createContext(nullptr, 1, &device, nullptr, nullptr, &err);
  return err == 0;
}
}  // namespace rcl
}  // namespace zrt

// ---- the OpenCL calls of the engine
inline cl_int clFinish(cl_command_queue) { zrt::sync(); return CL_SUCCESS; }
inline cl_int clFlush(cl_command_queue) { return CL_SUCCESS; }
inline cl_int clEnqueueCopyBuffer(cl_command_queue, cl_mem src, cl_mem dst, size_t so, size_t doff, size_t n, cl_uint,
                                  const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  if (!n) return CL_SUCCESS;
  if (so + n > src->size || doff + n > dst->size) {  // (refused as OpenCL does, not copied over what follows)
    fprintf(stderr, "arcinferzero: copy of %zu bytes past a buffer's end (source %zu + %zu of %zu, destination %zu + %zu of %zu; refused)\n",
            n, so, n, src->size, doff, n, dst->size);
    return -30;
  }
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::cl, dst->ptr + doff, src->ptr + so, n, nullptr, 0, nullptr));
  return CL_SUCCESS;
}
inline bool out_of_bounds(const char* what, cl_mem b, size_t off, size_t n) {
  if (b && off + n <= b->size) return false;
  fprintf(stderr, "arcinferzero: %s of %zu bytes at offset %zu past the end of a %zu-byte buffer (refused)\n", what, n,
          off, b ? b->size : (size_t)0);
  return true;
}
inline cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem b, cl_bool, size_t off, size_t n, void* host, cl_uint,
                                  const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  if (out_of_bounds("read", b, off, n)) return -30;
  char* s = zrt::staging(n);
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::cl, s, b->ptr + off, n, nullptr, 0, nullptr));
  zrt::sync();
  memcpy(host, s, n);
  return CL_SUCCESS;
}
inline cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem b, cl_bool, size_t off, size_t n, const void* host, cl_uint,
                                   const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  if (out_of_bounds("write", b, off, n)) return -30;
  char* s = zrt::staging(n);
  zrt::sync();  // (the bounce buffer may still be read by an earlier copy)
  memcpy(s, host, n);
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::cl, b->ptr + off, s, n, nullptr, 0, nullptr));
  zrt::sync();
  return CL_SUCCESS;
}
inline cl_int clEnqueueFillBuffer(cl_command_queue, cl_mem b, const void* pattern, size_t ps, size_t off, size_t n,
                                  cl_uint, const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  if (out_of_bounds("fill", b, off, n)) return -30;
  ZE_CHECK(zeCommandListAppendMemoryFill(zrt::cl, b->ptr + off, pattern, ps, n, nullptr, 0, nullptr));
  zrt::sync();  // (the pattern is the caller's)
  return CL_SUCCESS;
}
inline void* clEnqueueMapBuffer(cl_command_queue, cl_mem b, cl_bool, cl_map_flags flags, size_t off, size_t n, cl_uint,
                                const cl_event*, cl_event* ev, cl_int* err) {
  if (ev) *ev = nullptr;
  void* h = zrt::host_alloc(n);
  if (flags & CL_MAP_READ) {
    ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::cl, h, b->ptr + off, n, nullptr, 0, nullptr));
    zrt::sync();
  }
  zrt::maps.push_back({b, off, n, h});
  if (err) *err = CL_SUCCESS;
  return h;
}
inline cl_int clEnqueueUnmapMemObject(cl_command_queue, cl_mem b, void* p, cl_uint, const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  for (size_t i = 0; i < zrt::maps.size(); ++i)
    if (zrt::maps[i].host == p && zrt::maps[i].m == b) {
      ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::cl, b->ptr + zrt::maps[i].off, p, zrt::maps[i].n, nullptr, 0, nullptr));
      zrt::sync();
      zeMemFree(zrt::ctx, p);
      zrt::maps.erase(zrt::maps.begin() + i);
      return CL_SUCCESS;
    }
  return -30;
}
inline cl_mem clCreateBuffer(cl_context, cl_mem_flags flags, size_t n, void* host, cl_int* err) {
  ZMem* m = new ZMem{(char*)zrt::device_alloc(n), n, true};
  zrt::mems.insert(m);
  if ((flags & CL_MEM_COPY_HOST_PTR) && host) clEnqueueWriteBuffer(nullptr, m, CL_TRUE, 0, n, host, 0, nullptr, nullptr);
  if (err) *err = CL_SUCCESS;
  return m;
}
inline cl_mem clCreateSubBuffer(cl_mem parent, cl_mem_flags, cl_buffer_create_type, const void* region, cl_int* err) {
  const cl_buffer_region* r = (const cl_buffer_region*)region;
  if (r->origin + r->size > parent->size) {
    if (err) *err = -13;
    return nullptr;
  }
  ZMem* m = new ZMem{parent->ptr + r->origin, r->size, false};
  zrt::mems.insert(m);
  if (err) *err = CL_SUCCESS;
  return m;
}
inline cl_int clReleaseMemObject(cl_mem m) {
  if (!m) return CL_SUCCESS;
  if (m->owner) { zrt::sync(); zeMemFree(zrt::ctx, m->ptr); zrt::device_bytes -= m->size; }
  zrt::mems.erase(m);
  delete m;
  return CL_SUCCESS;
}
// a buffer argument (a registered cl_mem) becomes its USM pointer; anything else is passed as given
inline cl_int clSetKernelArg(cl_kernel k, cl_uint i, size_t sz, const void* v) {
  if (sz == sizeof(void*) && v) {
    ZMem* m = *(ZMem* const*)v;
    if (m && zrt::mems.count(m)) {
      void* p = m->ptr;
      return zeKernelSetArgumentValue(k->k, i, sizeof(p), &p) == ZE_RESULT_SUCCESS ? CL_SUCCESS : -51;
    }
  }
  return zeKernelSetArgumentValue(k->k, i, sz, v) == ZE_RESULT_SUCCESS ? CL_SUCCESS : -51;
}
inline cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel k, cl_uint dim, const size_t*, const size_t* g,
                                     const size_t* l, cl_uint, const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  uint32_t ls[3] = {1, 1, 1}, gs[3] = {1, 1, 1};
  for (cl_uint d = 0; d < dim; ++d) gs[d] = (uint32_t)g[d];
  if (l) {
    for (cl_uint d = 0; d < dim; ++d) ls[d] = (uint32_t)l[d];
  } else {
    ZE_CHECK(zeKernelSuggestGroupSize(k->k, gs[0], gs[1], gs[2], &ls[0], &ls[1], &ls[2]));
    // (Level Zero runs whole work-groups only: a group size that does not divide the global size would start
    //  work-items past it, and these kernels have no bounds checks; take the largest divisor not above the
    //  suggestion)
    for (int d = 0; d < 3; ++d)
      while (ls[d] > 1 && gs[d] % ls[d]) --ls[d];
  }
  if (ls[0] != k->gx || ls[1] != k->gy || ls[2] != k->gz) {
    ZE_CHECK(zeKernelSetGroupSize(k->k, ls[0], ls[1], ls[2]));
    k->gx = ls[0], k->gy = ls[1], k->gz = ls[2];
  }
  ze_group_count_t gc{(gs[0] + ls[0] - 1) / ls[0], (gs[1] + ls[1] - 1) / ls[1], (gs[2] + ls[2] - 1) / ls[2]};
  if (gs[0] % ls[0] || gs[1] % ls[1] || gs[2] % ls[2]) {
    // OpenCL would run a smaller last work-group; here the last group runs whole and its extra work-items would
    // read and write past the data. A launch like that is a bug in the caller (round the global size and give the
    // kernel a bounds check), so it fails loudly instead of corrupting memory.
    fprintf(stderr, "arcinferzero: %s: global size %u x %u x %u is not a multiple of the work-group %u x %u x %u\n",
            k->name.c_str(), gs[0], gs[1], gs[2], ls[0], ls[1], ls[2]);
    return -54;  // (CL_INVALID_WORK_GROUP_SIZE)
  }
  ze_event_handle_t sig = ev ? zrt::ev_get() : nullptr;
  if (ev) *ev = (cl_event)sig;
  ze_result_t r = zeCommandListAppendLaunchKernel(zrt::cl, k->k, &gc, sig, 0, nullptr);
  static const long fake_oom = getenv("ARC_FAKE_OOM") ? atol(getenv("ARC_FAKE_OOM")) : 0;  // (test: launch N "fails")
  static long launches = 0;
  if (fake_oom && ++launches % fake_oom == 0 && r == ZE_RESULT_SUCCESS) {
    zrt::sync();  // (the kernel was appended: let it finish, then report the failure as if it had not been)
    fprintf(stderr, "(test: launch %ld of %s reported as out of device memory)\n", launches, k->name.c_str());
    if (zrt::oom_hook && zrt::oom_hook()) return CL_SUCCESS;
  }
  for (int i = 0; r == (ze_result_t)0x70000003 && i < 8 && zrt::oom_hook && zrt::oom_hook(); ++i)  // (out of device memory)
    r = zeCommandListAppendLaunchKernel(zrt::cl, k->k, &gc, sig, 0, nullptr);
  if (r != ZE_RESULT_SUCCESS) {
    fprintf(stderr, "Level Zero error 0x%x launching %s (global %u x %u x %u, local %u x %u x %u)\n", r, k->name.c_str(),
            gs[0], gs[1], gs[2], ls[0], ls[1], ls[2]);
    if (ze_fail_hook) ze_fail_hook();
    fflush(stderr);
    _Exit(1);
  }
  static const bool sync_each = getenv("ARC_ZE_SYNC_EACH") != nullptr;  // (debug: which kernel loses the device)
  if (sync_each) {
    const ze_result_t s = zeCommandListHostSynchronize(zrt::cl, UINT64_MAX);
    if (s != ZE_RESULT_SUCCESS) {
      fprintf(stderr, "Level Zero error 0x%x running %s (global %u x %u x %u, local %u x %u x %u)\n", s, k->name.c_str(),
              gs[0], gs[1], gs[2], ls[0], ls[1], ls[2]);
      if (ze_fail_hook) ze_fail_hook();
      fflush(stderr);
      _Exit(1);
    }
  }
  return CL_SUCCESS;
}
inline cl_int clGetDeviceInfo(cl_device_id, cl_device_info p, size_t sz, void* v, size_t* ret) {
  if (p == CL_DEVICE_GLOBAL_MEM_SIZE && sz >= 8) { *(cl_ulong*)v = zrt::global_mem; if (ret) *ret = 8; return CL_SUCCESS; }
  return -30;
}
inline cl_int clWaitForEvents(cl_uint, const cl_event*) { zrt::sync(); return CL_SUCCESS; }
inline cl_int clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t sz, void* v, size_t*) {
  if (v && sz >= 8) *(cl_ulong*)v = 0;
  return CL_SUCCESS;
}
inline cl_int clReleaseEvent(cl_event e) {  // (only kernel events are real: back to the free list)
  if (e) { ZE_CHECK(zeEventHostReset((ze_event_handle_t)e)); zrt::ev_free.push_back((ze_event_handle_t)e); }
  return CL_SUCCESS;
}
inline cl_int clReleaseKernel(cl_kernel k) { if (k) { zeKernelDestroy(k->k); delete k; } return CL_SUCCESS; }
inline cl_int clReleaseProgram(cl_program p) { if (p) { zeModuleDestroy(p->m); delete p; } return CL_SUCCESS; }
inline cl_int clReleaseCommandQueue(cl_command_queue) { return CL_SUCCESS; }
inline cl_int clReleaseContext(cl_context) { return CL_SUCCESS; }

struct ClRt {
  std::string device_id, cache_dir;
  static uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
  }
  // (non-null placeholders: the engine passes them back to the calls above)
  cl_platform_id platform = (cl_platform_id)1;
  cl_device_id device = (cl_device_id)1;
  cl_context ctx = (cl_context)1;
  cl_command_queue queue = (cl_command_queue)1;

  bool init(bool = true) {
    if (!zrt::rcl::load()) { fprintf(stderr, "cannot load OpenCL.dll (needed to compile kernels)\n"); return false; }
    char name[256] = {}, drv[256] = {};
    zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102B, sizeof(name) - 1, name, nullptr);  // CL_DEVICE_NAME
    zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102D, sizeof(drv) - 1, drv, nullptr);   // CL_DRIVER_VERSION
    device_id = std::string(name) + "|" + drv;  // (the same key as arcinfer's kernel cache)
    if (const char* la = getenv("LOCALAPPDATA"); la && !getenv("ARC_NO_KCACHE")) cache_dir = std::string(la) + "\\arcinfer\\kernels";
    // a second engine in the process (the server switching models, after the first one freed its memory) keeps the
    // runtime: the context, its command lists and events, the staging buffer (all of zrt:: belongs to that context)
    if (zrt::ctx) return true;
    if (!ze_load()) { fprintf(stderr, "cannot load ze_loader.dll (Level Zero)\n"); return false; }
    ZE_CHECK(zeInit(1));
    uint32_t n = 1;
    ze_driver_handle_t drvh;
    ZE_CHECK(zeDriverGet(&n, &drvh));
    n = 1;
    ZE_CHECK(zeDeviceGet(drvh, &n, &zrt::dev));
    uint32_t nm = 1;
    ze_device_memory_properties_t mp{};
    mp.stype = ZE_STYPE_DEVICE_MEMORY_PROPERTIES;
    ZE_CHECK(zeDeviceGetMemoryProperties(zrt::dev, &nm, &mp));
    zrt::global_mem = mp.totalSize;
    ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
    ZE_CHECK(zeContextCreate(drvh, &cd, &zrt::ctx));
    ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 2 /* in order */, 2 /* asynchronous */, 0};
    ZE_CHECK(zeCommandListCreateImmediate(zrt::ctx, zrt::dev, &qd, &zrt::cl));
    ze_fail_hook = zrt::print_memory;
    return true;
  }

  // The native binary for (source, options): from the shared kernel cache, else compiled by OpenCL and saved there.
  cl_program build(const std::string& src, const std::string& opts) {
    std::string path;
    std::vector<uint8_t> bin;
    if (!cache_dir.empty()) {
      char hex[17];
      snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)fnv1a(src + "\x01" + opts + "\x01" + device_id));
      path = cache_dir + "\\" + hex + ".bin";
      std::ifstream f(path, std::ios::binary);
      if (f) bin.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    if (bin.empty()) {
      cl_int err = 0;
      const char* s = src.c_str();
      const size_t sn = src.size();
      void* prog = zrt::rcl::createProgramWithSource(zrt::rcl::context, 1, &s, &sn, &err);
      err = zrt::rcl::buildProgram(prog, 1, &zrt::rcl::device, opts.c_str(), nullptr, nullptr);
      if (err != 0) {
        size_t ln = 0;
        zrt::rcl::getProgramBuildInfo(prog, zrt::rcl::device, 0x1183, 0, nullptr, &ln);
        std::string log(ln, 0);
        zrt::rcl::getProgramBuildInfo(prog, zrt::rcl::device, 0x1183, ln, log.data(), nullptr);
        fprintf(stderr, "build log:\n%s\n", log.c_str());
        CL_CHECK(err);
      }
      size_t bn = 0;
      zrt::rcl::getProgramInfo(prog, 0x1165, sizeof(bn), &bn, nullptr);
      bin.resize(bn);
      uint8_t* bp = bin.data();
      zrt::rcl::getProgramInfo(prog, 0x1166, sizeof(bp), &bp, nullptr);
      zrt::rcl::releaseProgram(prog);
      if (!path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        const std::string tmp = path + ".tmp";
        { std::ofstream o(tmp, std::ios::binary); o.write((const char*)bin.data(), (std::streamsize)bin.size()); }
        std::filesystem::rename(tmp, path, ec);
      }
    }
    // The OpenCL build option -cl-intel-256-GRF-per-thread does not survive into a Level Zero module made from the
    // native binary: the kernels ran with the default register file and spilled (attn_decode at HD 256: 4 KB per
    // thread, 1.9 -> 3.6 ms per layer at 64K context). The module asks for the large register file itself.
    const bool big_grf = opts.find("-cl-intel-256-GRF-per-thread") != std::string::npos;
    ze_module_desc_t md{ZE_STYPE_MODULE_DESC, nullptr, 1 /* native */, bin.size(), bin.data(),
                        big_grf ? "-ze-opt-large-register-file" : "", nullptr};
    ZProg* p = new ZProg{};
    p->indirect = opts.find("-DXCACHE") != std::string::npos;
    p->opts = opts;
    ze_module_build_log_handle_t log = nullptr;
    const ze_result_t r = zeModuleCreate(zrt::ctx, zrt::dev, &md, &p->m, &log);
    if (r != ZE_RESULT_SUCCESS) {
      size_t ln = 0;
      zeModuleBuildLogGetString(log, &ln, nullptr);
      std::string s(ln, 0);
      zeModuleBuildLogGetString(log, &ln, s.data());
      fprintf(stderr, "zeModuleCreate failed 0x%x: %s\n", r, s.c_str());
      exit(1);
    }
    if (log) zeModuleBuildLogDestroy(log);
    return p;
  }
  cl_kernel kernel(cl_program prog, const char* name) {
    ZKernel* k = new ZKernel{};
    k->name = name;
    ze_kernel_desc_t kd{ZE_STYPE_KERNEL_DESC, nullptr, 0, name};
    const ze_result_t r = zeKernelCreate(prog->m, &kd, &k->k);
    if (r != ZE_RESULT_SUCCESS) { fprintf(stderr, "zeKernelCreate(%s) failed 0x%x\n", name, r); exit(1); }
    // The expert kernels (-DXCACHE) reach memory that is not one of their arguments (each expert's address comes from
    // a table): with indirect access the driver keeps every allocation resident for them; without it an expert in a
    // piece of the cache that Windows had paged out was a page fault ("device lost" now and then, in the server).
    // Only for them: on every kernel it made the driver touch every allocation per launch (27B: slower).
    if (prog->indirect) zeKernelSetIndirectAccess(k->k, 7 /* device | host | shared */);
    static const bool kprops = getenv("ARC_ZE_KPROPS") != nullptr;  // (debug: private / spill memory per kernel)
    if (kprops) {
      ze_kernel_properties_t kp{};
      kp.stype = (ze_structure_type_t)0x1e;
      if (zeKernelGetProperties(k->k, &kp) == ZE_RESULT_SUCCESS && (kp.privateMemSize || kp.spillMemSize))
        fprintf(stderr, "kernel %s: private %u bytes, spill %u bytes per work-item/thread [%s]\n", name, kp.privateMemSize,
                kp.spillMemSize, prog->opts.c_str());
    }
    return k;
  }
  size_t spill_bytes(cl_kernel) { return 0; }
  cl_mem buffer(cl_mem_flags flags, size_t bytes, const void* host = nullptr) {
    cl_int err;
    cl_mem m = clCreateBuffer(ctx, flags | (host ? CL_MEM_COPY_HOST_PTR : 0), bytes, (void*)host, &err);
    CL_CHECK(err);
    return m;
  }
  static double event_ms(cl_event e) {  // (kernel time from the Level Zero timestamps; call after clFinish)
    if (!e) return 0.0;
    ze_kernel_timestamp_result_t ts{};
    if (zeEventQueryKernelTimestamp((ze_event_handle_t)e, &ts) != ZE_RESULT_SUCCESS) return 0.0;
    const uint64_t t = (ts.context.kernelEnd - ts.context.kernelStart) & zrt::ev_mask;
    return t * zrt::ev_ns_per_tick * 1e-6;
  }
};

inline std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

template <typename T> inline void set_args(cl_kernel k, cl_uint i, const T& v) { CL_CHECK(clSetKernelArg(k, i, sizeof(T), &v)); }
template <typename T, typename... R> inline void set_args(cl_kernel k, cl_uint i, const T& v, const R&... rest) {
  set_args(k, i, v);
  set_args(k, i + 1, rest...);
}
