// Prints the OpenCL GPU properties that matter for kernel design.
#include "../src/cl_min.h"
#include <string>
#include <vector>

static std::string dev_str(cl_device_id d, cl_device_info p) {
  size_t n = 0;
  clGetDeviceInfo(d, p, 0, nullptr, &n);
  std::string s(n, '\0');
  clGetDeviceInfo(d, p, n, s.data(), nullptr);
  if (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

template <typename T> static T dev_val(cl_device_id d, cl_device_info p) {
  T v{};
  clGetDeviceInfo(d, p, sizeof(T), &v, nullptr);
  return v;
}

int main() {
  if (!cl_load()) { fprintf(stderr, "cannot load OpenCL.dll\n"); return 1; }
  cl_uint np = 0;
  CL_CHECK(clGetPlatformIDs(0, nullptr, &np));
  std::vector<cl_platform_id> plats(np);
  CL_CHECK(clGetPlatformIDs(np, plats.data(), nullptr));
  for (auto p : plats) {
    char name[256] = {};
    clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
    printf("platform: %s\n", name);
    cl_uint nd = 0;
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
    std::vector<cl_device_id> devs(nd);
    clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
    for (auto d : devs) {
      printf("  device: %s\n", dev_str(d, CL_DEVICE_NAME).c_str());
      printf("  version: %s | driver %s\n", dev_str(d, CL_DEVICE_VERSION).c_str(), dev_str(d, CL_DRIVER_VERSION).c_str());
      printf("  compute units (EUs): %u\n", dev_val<cl_uint>(d, CL_DEVICE_MAX_COMPUTE_UNITS));
      printf("  max clock: %u MHz\n", dev_val<cl_uint>(d, CL_DEVICE_MAX_CLOCK_FREQUENCY));
      printf("  global mem: %.1f GiB | max alloc: %.1f GiB\n", dev_val<cl_ulong>(d, CL_DEVICE_GLOBAL_MEM_SIZE) / 1073741824.0,
             dev_val<cl_ulong>(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE) / 1073741824.0);
      printf("  local mem (SLM): %llu KiB | L3: %llu KiB\n",
             (unsigned long long)dev_val<cl_ulong>(d, CL_DEVICE_LOCAL_MEM_SIZE) / 1024,
             (unsigned long long)dev_val<cl_ulong>(d, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) / 1024);
      printf("  max work-group: %zu | host unified mem: %u\n", dev_val<size_t>(d, CL_DEVICE_MAX_WORK_GROUP_SIZE),
             dev_val<cl_bool>(d, CL_DEVICE_HOST_UNIFIED_MEMORY));
      size_t sgn = 0;
      clGetDeviceInfo(d, CL_DEVICE_SUB_GROUP_SIZES_INTEL, 0, nullptr, &sgn);
      std::vector<size_t> sgs(sgn / sizeof(size_t));
      clGetDeviceInfo(d, CL_DEVICE_SUB_GROUP_SIZES_INTEL, sgn, sgs.data(), nullptr);
      printf("  sub-group sizes:");
      for (auto s : sgs) printf(" %zu", s);
      printf("\n");
      std::string ext = dev_str(d, CL_DEVICE_EXTENSIONS);
      const char* want[] = {"cl_khr_fp16", "cl_khr_integer_dot_product", "cl_intel_subgroups", "cl_intel_subgroups_short",
                            "cl_intel_subgroups_char", "cl_intel_required_subgroup_size", "cl_intel_subgroup_matrix_multiply_accumulate",
                            "cl_intel_split_work_group_barrier", "cl_intel_subgroup_2d_block_io", "cl_khr_subgroup_shuffle",
                            "cl_intel_unified_shared_memory", "cl_intel_bfloat16_conversions", "cl_khr_subgroup_extended_types"};
      for (auto w : want) printf("  %-48s %s\n", w, ext.find(w) != std::string::npos ? "yes" : "-");
    }
  }
  return 0;
}
