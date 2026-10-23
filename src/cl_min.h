// Minimal OpenCL 3.0 declarations + dynamic loader for OpenCL.dll.
// Only what arcinferzero needs; avoids depending on the Khronos SDK.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint cl_bool;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_map_flags;
typedef cl_bitfield cl_queue_properties_t;
typedef cl_ulong cl_queue_properties;
typedef intptr_t cl_context_properties;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_uint cl_program_build_info;
typedef cl_uint cl_program_info;
typedef cl_uint cl_kernel_work_group_info;
typedef cl_uint cl_profiling_info;

typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef struct _cl_event* cl_event;

#define CL_SUCCESS 0
#define CL_TRUE 1
#define CL_FALSE 0
#define CL_DEVICE_TYPE_GPU (1 << 2)

#define CL_PLATFORM_NAME 0x0902
#define CL_PLATFORM_VERSION 0x0901

#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_DEVICE_GLOBAL_MEM_CACHE_SIZE 0x101E
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1023
#define CL_DEVICE_NAME 0x102B
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_EXTENSIONS 0x1030
#define CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#define CL_DEVICE_SUB_GROUP_SIZES_INTEL 0x4108

#define CL_QUEUE_PROPERTIES 0x1093
#define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (1 << 0)
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)

#define CL_MEM_READ_WRITE (1 << 0)
#define CL_BUFFER_CREATE_TYPE_REGION 0x1220
struct cl_buffer_region { size_t origin, size; };
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_USE_HOST_PTR (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR (1 << 4)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_MAP_READ (1 << 0)
#define CL_MAP_WRITE (1 << 1)

#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_PROGRAM_BINARY_SIZES 0x1165
#define CL_PROGRAM_BINARIES 0x1166
#define CL_KERNEL_WORK_GROUP_SIZE 0x11B0
#define CL_KERNEL_PRIVATE_MEM_SIZE 0x11B4
#define CL_KERNEL_SPILL_MEM_SIZE_INTEL 0x4109
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283

#define CL_FUNCS(X)                                                                                        \
  X(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id*, cl_uint*))                                        \
  X(cl_int, clGetPlatformInfo, (cl_platform_id, cl_platform_info, size_t, void*, size_t*))                 \
  X(cl_int, clGetDeviceIDs, (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*))            \
  X(cl_int, clGetDeviceInfo, (cl_device_id, cl_device_info, size_t, void*, size_t*))                       \
  X(cl_context, clCreateContext,                                                                           \
    (const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*))                   \
  X(cl_command_queue, clCreateCommandQueueWithProperties,                                                  \
    (cl_context, cl_device_id, const cl_queue_properties*, cl_int*))                                       \
  X(cl_mem, clCreateBuffer, (cl_context, cl_mem_flags, size_t, void*, cl_int*))                              X(cl_mem, clCreateSubBuffer, (cl_mem, cl_mem_flags, cl_uint, const void*, cl_int*))                      \
  X(cl_int, clEnqueueWriteBuffer,                                                                          \
    (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*)) \
  X(cl_int, clEnqueueReadBuffer,                                                                           \
    (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*))       \
  X(cl_int, clEnqueueCopyBuffer,                                                                           \
    (cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*))      \
  X(cl_int, clEnqueueFillBuffer,                                                                           \
    (cl_command_queue, cl_mem, const void*, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*))  \
  X(void*, clEnqueueMapBuffer,                                                                             \
    (cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event*, cl_event*, \
     cl_int*))                                                                                             \
  X(cl_int, clEnqueueUnmapMemObject, (cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*)) \
  X(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char**, const size_t*, cl_int*))    \
  X(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void*, void*))         \
  X(cl_program, clCreateProgramWithBinary,                                                                 \
    (cl_context, cl_uint, const cl_device_id*, const size_t*, const unsigned char**, cl_int*, cl_int*))    \
  X(cl_int, clGetProgramInfo, (cl_program, cl_program_info, size_t, void*, size_t*))                       \
  X(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*)) \
  X(cl_kernel, clCreateKernel, (cl_program, const char*, cl_int*))                                         \
  X(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void*))                                     \
  X(cl_int, clGetKernelWorkGroupInfo,                                                                      \
    (cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*))                          \
  X(cl_int, clEnqueueNDRangeKernel,                                                                        \
    (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint,           \
     const cl_event*, cl_event*))                                                                          \
  X(cl_int, clGetEventProfilingInfo, (cl_event, cl_profiling_info, size_t, void*, size_t*))                \
  X(cl_int, clWaitForEvents, (cl_uint, const cl_event*))                                                   \
  X(cl_int, clReleaseEvent, (cl_event))                                                                    \
  X(cl_int, clFinish, (cl_command_queue))                                                                  \
  X(cl_int, clFlush, (cl_command_queue))                                                                   \
  X(cl_int, clReleaseMemObject, (cl_mem))                                                                  \
  X(cl_int, clReleaseKernel, (cl_kernel))                                                                  \
  X(cl_int, clReleaseProgram, (cl_program))                                                                \
  X(cl_int, clReleaseCommandQueue, (cl_command_queue))                                                     \
  X(cl_int, clReleaseContext, (cl_context))

#define CL_DECLARE(ret, name, args) typedef ret(__stdcall* PFN_##name) args; inline PFN_##name name = nullptr;
CL_FUNCS(CL_DECLARE)
#undef CL_DECLARE

inline bool cl_load() {
  HMODULE h = LoadLibraryA("OpenCL.dll");
  if (!h) return false;
#define CL_RESOLVE(ret, name, args)                       \
  name = (PFN_##name)(void*)GetProcAddress(h, #name);     \
  if (!name) { fprintf(stderr, "missing %s\n", #name); return false; }
  CL_FUNCS(CL_RESOLVE)
#undef CL_RESOLVE
  return true;
}

#define CL_CHECK(expr)                                                              \
  do {                                                                              \
    cl_int _e = (expr);                                                             \
    if (_e != CL_SUCCESS) {                                                         \
      fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", _e, __FILE__, __LINE__, #expr); \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)
