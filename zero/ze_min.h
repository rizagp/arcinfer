// arcinferzero: minimal Level Zero (oneAPI L0) declarations + a dynamic loader for ze_loader.dll (installed with the
// Intel graphics driver). Only what arcinferzero uses; the layouts follow ze_api.h (Level Zero spec 1.x).
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

typedef uint32_t ze_result_t;
#define ZE_RESULT_SUCCESS 0u
typedef struct _ze_driver_handle_t* ze_driver_handle_t;
typedef struct _ze_device_handle_t* ze_device_handle_t;
typedef struct _ze_context_handle_t* ze_context_handle_t;
typedef struct _ze_command_list_handle_t* ze_command_list_handle_t;
typedef struct _ze_module_handle_t* ze_module_handle_t;
typedef struct _ze_module_build_log_handle_t* ze_module_build_log_handle_t;
typedef struct _ze_kernel_handle_t* ze_kernel_handle_t;
typedef struct _ze_event_handle_t* ze_event_handle_t;
typedef struct _ze_event_pool_handle_t* ze_event_pool_handle_t;
typedef uint32_t ze_structure_type_t;

// structure types (ze_structure_type_t)
#define ZE_STYPE_DEVICE_PROPERTIES 0x3u
#define ZE_STYPE_DEVICE_MEMORY_PROPERTIES 0x7u
#define ZE_STYPE_CONTEXT_DESC 0xdu
#define ZE_STYPE_COMMAND_QUEUE_DESC 0xeu
#define ZE_STYPE_EVENT_POOL_DESC 0x10u
#define ZE_STYPE_EVENT_DESC 0x11u
#define ZE_STYPE_DEVICE_MEM_ALLOC_DESC 0x15u
#define ZE_STYPE_HOST_MEM_ALLOC_DESC 0x16u
#define ZE_STYPE_MODULE_DESC 0x1bu
#define ZE_STYPE_KERNEL_DESC 0x1du

struct ze_device_properties_t {
  ze_structure_type_t stype;
  void* pNext;
  uint32_t type, vendorId, deviceId, flags, subdeviceId, coreClockRate;
  uint64_t maxMemAllocSize;
  uint32_t maxHardwareContexts, maxCommandQueuePriority, numThreadsPerEU, physicalEUSimdWidth, numEUsPerSubslice,
      numSubslicesPerSlice, numSlices;
  uint64_t timerResolution;
  uint32_t timestampValidBits, kernelTimestampValidBits;
  uint8_t uuid[16];
  char name[256];
};
struct ze_device_memory_properties_t {
  ze_structure_type_t stype;
  void* pNext;
  uint32_t flags, maxClockRate, maxBusWidth;
  uint64_t totalSize;
  char name[256];
};
struct ze_context_desc_t { ze_structure_type_t stype; const void* pNext; uint32_t flags; };
struct ze_command_queue_desc_t {
  ze_structure_type_t stype;
  const void* pNext;
  uint32_t ordinal, index, flags, mode, priority;  // flags: 2 = in order; mode: 0 default, 1 synchronous, 2 asynchronous
};
struct ze_host_mem_alloc_desc_t { ze_structure_type_t stype; const void* pNext; uint32_t flags; };
struct ze_device_mem_alloc_desc_t { ze_structure_type_t stype; const void* pNext; uint32_t flags, ordinal; };
struct ze_module_constants_t { uint32_t numConstants; const uint32_t* pConstantIds; const void** pConstantValues; };
struct ze_module_desc_t {
  ze_structure_type_t stype;
  const void* pNext;
  uint32_t format;  // 0 SPIR-V, 1 native (the device binary the OpenCL compiler produces)
  size_t inputSize;
  const uint8_t* pInputModule;
  const char* pBuildFlags;
  const ze_module_constants_t* pConstants;
};
struct ze_kernel_desc_t { ze_structure_type_t stype; const void* pNext; uint32_t flags; const char* pKernelName; };
struct ze_kernel_properties_t {  // (stype 0x1e)
  ze_structure_type_t stype;
  void* pNext;
  uint32_t numKernelArgs, requiredGroupSize[3], requiredNumSubGroups, requiredSubgroupSize, maxSubgroupSize,
      maxNumSubgroups, localMemSize, privateMemSize, spillMemSize;
  uint8_t uuid[32];
};
struct ze_group_count_t { uint32_t groupCountX, groupCountY, groupCountZ; };
struct ze_event_pool_desc_t { ze_structure_type_t stype; const void* pNext; uint32_t flags, count; };
struct ze_event_desc_t { ze_structure_type_t stype; const void* pNext; uint32_t index, signal, wait; };
struct ze_kernel_timestamp_data_t { uint64_t kernelStart, kernelEnd; };
struct ze_kernel_timestamp_result_t { ze_kernel_timestamp_data_t global, context; };

#define ZE_FUNCS(X)                                                                                              \
  X(ze_result_t, zeInit, (uint32_t))                                                                             \
  X(ze_result_t, zeDriverGet, (uint32_t*, ze_driver_handle_t*))                                                  \
  X(ze_result_t, zeDeviceGet, (ze_driver_handle_t, uint32_t*, ze_device_handle_t*))                              \
  X(ze_result_t, zeDeviceGetProperties, (ze_device_handle_t, ze_device_properties_t*))                           \
  X(ze_result_t, zeDeviceGetMemoryProperties, (ze_device_handle_t, uint32_t*, ze_device_memory_properties_t*))    \
  X(ze_result_t, zeContextCreate, (ze_driver_handle_t, const ze_context_desc_t*, ze_context_handle_t*))          \
  X(ze_result_t, zeContextDestroy, (ze_context_handle_t))                                                        \
  X(ze_result_t, zeCommandListCreateImmediate,                                                                   \
    (ze_context_handle_t, ze_device_handle_t, const ze_command_queue_desc_t*, ze_command_list_handle_t*))        \
  X(ze_result_t, zeCommandListDestroy, (ze_command_list_handle_t))                                               \
  X(ze_result_t, zeCommandListHostSynchronize, (ze_command_list_handle_t, uint64_t))                             \
  X(ze_result_t, zeCommandListAppendLaunchKernel,                                                                \
    (ze_command_list_handle_t, ze_kernel_handle_t, const ze_group_count_t*, ze_event_handle_t, uint32_t,          \
     ze_event_handle_t*))                                                                                        \
  X(ze_result_t, zeCommandListAppendMemoryCopy,                                                                  \
    (ze_command_list_handle_t, void*, const void*, size_t, ze_event_handle_t, uint32_t, ze_event_handle_t*))     \
  X(ze_result_t, zeCommandListAppendBarrier, (ze_command_list_handle_t, ze_event_handle_t, uint32_t, ze_event_handle_t*)) \
  X(ze_result_t, zeCommandListAppendMemoryFill,                                                                  \
    (ze_command_list_handle_t, void*, const void*, size_t, size_t, ze_event_handle_t, uint32_t, ze_event_handle_t*)) \
  X(ze_result_t, zeKernelSuggestGroupSize, (ze_kernel_handle_t, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t*, uint32_t*)) \
  X(ze_result_t, zeMemAllocHost, (ze_context_handle_t, const ze_host_mem_alloc_desc_t*, size_t, size_t, void**))  \
  X(ze_result_t, zeMemAllocDevice,                                                                               \
    (ze_context_handle_t, const ze_device_mem_alloc_desc_t*, size_t, size_t, ze_device_handle_t, void**))        \
  X(ze_result_t, zeMemAllocShared,                                                                               \
    (ze_context_handle_t, const ze_device_mem_alloc_desc_t*, const ze_host_mem_alloc_desc_t*, size_t, size_t,     \
     ze_device_handle_t, void**))                                                                                \
  X(ze_result_t, zeMemFree, (ze_context_handle_t, void*))                                                        \
  X(ze_result_t, zeModuleCreate,                                                                                 \
    (ze_context_handle_t, ze_device_handle_t, const ze_module_desc_t*, ze_module_handle_t*,                      \
     ze_module_build_log_handle_t*))                                                                             \
  X(ze_result_t, zeModuleDestroy, (ze_module_handle_t))                                                          \
  X(ze_result_t, zeModuleBuildLogGetString, (ze_module_build_log_handle_t, size_t*, char*))                      \
  X(ze_result_t, zeModuleBuildLogDestroy, (ze_module_build_log_handle_t))                                        \
  X(ze_result_t, zeKernelCreate, (ze_module_handle_t, const ze_kernel_desc_t*, ze_kernel_handle_t*))             \
  X(ze_result_t, zeKernelGetProperties, (ze_kernel_handle_t, ze_kernel_properties_t*))                           \
  X(ze_result_t, zeKernelSetIndirectAccess, (ze_kernel_handle_t, uint32_t))                                      \
  X(ze_result_t, zeKernelDestroy, (ze_kernel_handle_t))                                                          \
  X(ze_result_t, zeKernelSetGroupSize, (ze_kernel_handle_t, uint32_t, uint32_t, uint32_t))                       \
  X(ze_result_t, zeKernelSetArgumentValue, (ze_kernel_handle_t, uint32_t, size_t, const void*))                  \
  X(ze_result_t, zeEventPoolCreate,                                                                              \
    (ze_context_handle_t, const ze_event_pool_desc_t*, uint32_t, ze_device_handle_t*, ze_event_pool_handle_t*))  \
  X(ze_result_t, zeEventCreate, (ze_event_pool_handle_t, const ze_event_desc_t*, ze_event_handle_t*))            \
  X(ze_result_t, zeEventHostSynchronize, (ze_event_handle_t, uint64_t))                                          \
  X(ze_result_t, zeEventHostReset, (ze_event_handle_t))                                                            X(ze_result_t, zeEventQueryKernelTimestamp, (ze_event_handle_t, ze_kernel_timestamp_result_t*))

#define ZE_DECL(ret, name, args) typedef ret(__stdcall* PFN_##name) args; inline PFN_##name name = nullptr;
ZE_FUNCS(ZE_DECL)
#undef ZE_DECL

inline bool ze_load() {
  HMODULE m = LoadLibraryA("ze_loader.dll");
  if (!m) return false;
#define ZE_LOAD(ret, name, args) \
  name = (PFN_##name)GetProcAddress(m, #name); \
  if (!name) { fprintf(stderr, "ze_loader.dll: %s missing\n", #name); return false; }
  ZE_FUNCS(ZE_LOAD)
#undef ZE_LOAD
  return true;
}

inline void (*ze_fail_hook)() = nullptr;  // (diagnostics printed before exiting on an error)
#define ZE_CHECK(x)                                                                                 \
  do {                                                                                              \
    const ze_result_t r_ = (x);                                                                     \
    if (r_ != ZE_RESULT_SUCCESS) {                                                                  \
      fprintf(stderr, "Level Zero error 0x%x at %s:%d: %s\n", r_, __FILE__, __LINE__, #x);          \
      if (ze_fail_hook) ze_fail_hook();                                                             \
      fflush(stderr);                                                                               \
      _Exit(1); /* (not exit(): after a lost device the driver's cleanup can hang the process) */   \
    }                                                                                               \
  } while (0)
