// Prints the WDDM video memory budget and usage of this process on every adapter (DXGI QueryVideoMemoryInfo): what
// Windows lets a process keep resident on the GPU right now (for the Intel iGPU: the shared system memory segment).
#define NOMINMAX
#include <windows.h>
#include <dxgi1_4.h>
#include <cstdio>
#pragma comment(lib, "dxgi.lib")

int main() {
  IDXGIFactory4* f = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&f))) return 1;
  IDXGIAdapter1* a1 = nullptr;
  for (UINT i = 0; f->EnumAdapters1(i, &a1) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 d;
    a1->GetDesc1(&d);
    IDXGIAdapter3* a3 = nullptr;
    if (SUCCEEDED(a1->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3))) {
      DXGI_QUERY_VIDEO_MEMORY_INFO loc{}, non{};
      a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &loc);
      a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non);
      printf("%ls | dedicated %.2f GiB, shared %.2f GiB | local budget %.2f GiB (used %.2f) | non-local budget %.2f GiB (used %.2f)\n",
             d.Description, d.DedicatedVideoMemory / 1073741824.0, d.SharedSystemMemory / 1073741824.0,
             loc.Budget / 1073741824.0, loc.CurrentUsage / 1073741824.0, non.Budget / 1073741824.0, non.CurrentUsage / 1073741824.0);
      a3->Release();
    }
    a1->Release();
  }
  f->Release();
  return 0;
}
