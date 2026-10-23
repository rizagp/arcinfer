// Small OpenCL runtime helpers: device selection, program build, timing.
#pragma once
#include "app_dir.h"
#include "cl_min.h"
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>

struct ClRt {
  std::string device_id, cache_dir;
  static uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
  }
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context ctx = nullptr;
  cl_command_queue queue = nullptr;

  bool init(bool profiling = true) {
    if (!cl_load()) { fprintf(stderr, "cannot load OpenCL.dll\n"); return false; }
    cl_uint np = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &np));
    std::vector<cl_platform_id> plats(np);
    CL_CHECK(clGetPlatformIDs(np, plats.data(), nullptr));
    for (auto p : plats) {
      cl_uint nd = 0;
      if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, &nd) == CL_SUCCESS && nd > 0) { platform = p; break; }
    }
    if (!platform) { fprintf(stderr, "no OpenCL GPU found\n"); return false; }
    cl_int err;
    ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, profiling ? (cl_queue_properties)CL_QUEUE_PROFILING_ENABLE : 0, 0};
    queue = clCreateCommandQueueWithProperties(ctx, device, props, &err);
    CL_CHECK(err);
    char name[256] = {}, drv[256] = {};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(drv) - 1, drv, nullptr);
    device_id = std::string(name) + "|" + drv;
    if (!getenv("ARC_NO_KCACHE")) cache_dir = app_dir() + "\\kernels";
    return true;
  }

  // Builds a program, or loads its binary from the kernel cache (%LOCALAPPDATA%\arcinferzero\kernels, keyed by the
  // source, the options, the device and the driver version): the driver compiles each variant only once.
  cl_program build(const std::string& src, const std::string& opts) {
    std::string path;
    if (!cache_dir.empty()) {
      const uint64_t h = fnv1a(src + "\x01" + opts + "\x01" + device_id);
      char hex[17];
      snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)h);
      path = cache_dir + "\\" + hex + ".bin";
      std::ifstream f(path, std::ios::binary);
      if (f) {
        std::vector<unsigned char> bin((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const unsigned char* bp = bin.data();
        const size_t bn = bin.size();
        cl_int st = CL_SUCCESS, err = CL_SUCCESS;
        cl_program p = bn ? clCreateProgramWithBinary(ctx, 1, &device, &bn, &bp, &st, &err) : nullptr;
        if (p && err == CL_SUCCESS && st == CL_SUCCESS && clBuildProgram(p, 1, &device, opts.c_str(), nullptr, nullptr) == CL_SUCCESS)
          return p;
        if (p) clReleaseProgram(p);
      }
    }
    cl_program prog = build_source(src, opts);
    if (!path.empty()) {
      size_t bn = 0;
      if (clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bn), &bn, nullptr) == CL_SUCCESS && bn) {
        std::vector<unsigned char> bin(bn);
        unsigned char* bp = bin.data();
        if (clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bp), &bp, nullptr) == CL_SUCCESS) {
          std::error_code ec;
          std::filesystem::create_directories(cache_dir, ec);
          const std::string tmp = path + ".tmp";
          std::ofstream o(tmp, std::ios::binary);
          o.write((const char*)bin.data(), (std::streamsize)bn);
          o.close();
          std::filesystem::rename(tmp, path, ec);
        }
      }
    }
    return prog;
  }

  cl_program build_source(const std::string& src, const std::string& opts) {
    cl_int err;
    const char* s = src.c_str();
    size_t n = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &s, &n, &err);
    CL_CHECK(err);
    err = clBuildProgram(prog, 1, &device, opts.c_str(), nullptr, nullptr);
    size_t logn = 0;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logn);
    if (logn > 2) {
      std::string log(logn, '\0');
      clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, logn, log.data(), nullptr);
      if (err != CL_SUCCESS || log.find("warning") != std::string::npos) fprintf(stderr, "build log:\n%s\n", log.c_str());
    }
    CL_CHECK(err);
    return prog;
  }

  cl_kernel kernel(cl_program prog, const char* name) {
    cl_int err;
    cl_kernel k = clCreateKernel(prog, name, &err);
    CL_CHECK(err);
    return k;
  }

  size_t spill_bytes(cl_kernel k) {
    cl_ulong v = 0;
    clGetKernelWorkGroupInfo(k, device, CL_KERNEL_SPILL_MEM_SIZE_INTEL, sizeof(v), &v, nullptr);
    return (size_t)v;
  }

  cl_mem buffer(cl_mem_flags flags, size_t bytes, const void* host = nullptr) {
    cl_int err;
    cl_mem m = clCreateBuffer(ctx, flags | (host ? CL_MEM_COPY_HOST_PTR : 0), bytes, (void*)host, &err);
    CL_CHECK(err);
    return m;
  }

  // Elapsed GPU time of an event in milliseconds.
  static double event_ms(cl_event e) {
    cl_ulong t0 = 0, t1 = 0;
    clWaitForEvents(1, &e);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
    return (t1 - t0) * 1e-6;
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
