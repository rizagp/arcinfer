// SSD read speed for streaming model weights: sequential and random block reads of a large file with the Windows file
// cache bypassed (FILE_FLAG_NO_BUFFERING), one thread or several in flight (overlapped I/O).
// usage: ssd_bench <file> [block MiB = 2] [total GiB = 4] [queue depth = 8]
#define NOMINMAX
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static double run(HANDLE h, uint64_t fsize, size_t block, uint64_t total, int qd, bool random) {
  const int n = (int)(total / block);
  std::vector<OVERLAPPED> ov(qd);
  std::vector<HANDLE> ev(qd);
  std::vector<void*> buf(qd);
  for (int i = 0; i < qd; ++i) {
    ev[i] = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    buf[i] = VirtualAlloc(nullptr, block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  }
  std::mt19937_64 rng(42);
  const uint64_t nblk = fsize / block;
  uint64_t seq = 0;
  auto next_off = [&]() -> uint64_t { return random ? (rng() % nblk) * block : (seq++ % nblk) * block; };
  auto t0 = std::chrono::steady_clock::now();
  int issued = 0, done = 0;
  auto issue = [&](int s) {
    ZeroMemory(&ov[s], sizeof(OVERLAPPED));
    const uint64_t off = next_off();
    ov[s].Offset = (DWORD)off;
    ov[s].OffsetHigh = (DWORD)(off >> 32);
    ov[s].hEvent = ev[s];
    ResetEvent(ev[s]);
    if (!ReadFile(h, buf[s], (DWORD)block, nullptr, &ov[s]) && GetLastError() != ERROR_IO_PENDING) {
      fprintf(stderr, "ReadFile failed: %lu\n", GetLastError());
      exit(1);
    }
    ++issued;
  };
  for (int s = 0; s < qd && issued < n; ++s) issue(s);
  while (done < n) {
    const DWORD w = WaitForMultipleObjects((DWORD)std::min(qd, issued), ev.data(), FALSE, INFINITE);
    const int s = (int)(w - WAIT_OBJECT_0);
    DWORD got = 0;
    GetOverlappedResult(h, &ov[s], &got, FALSE);
    ++done;
    if (issued < n) issue(s);
    else ResetEvent(ev[s]), ev[s] = CreateEventW(nullptr, TRUE, FALSE, nullptr);  // (keep it out of the wait set)
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  for (int i = 0; i < qd; ++i) VirtualFree(buf[i], 0, MEM_RELEASE);
  return (double)n * block / s / 1e9;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ssd_bench <file> [block MiB] [total GiB] [queue depth]\n"); return 1; }
  const size_t block = (size_t)((argc > 2 ? atof(argv[2]) : 2.0) * 1048576);
  const uint64_t total = (uint64_t)((argc > 3 ? atof(argv[3]) : 4.0) * 1073741824.0);
  const int qd = argc > 4 ? atoi(argv[4]) : 8;
  HANDLE h = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
  if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  LARGE_INTEGER sz;
  GetFileSizeEx(h, &sz);
  printf("file %.1f GB, block %.2f MiB, %.1f GiB per test\n", sz.QuadPart / 1e9, block / 1048576.0, total / 1073741824.0);
  for (int q : {1, qd}) {
    printf("  sequential, queue %2d: %6.2f GB/s\n", q, run(h, (uint64_t)sz.QuadPart, block, total, q, false));
    printf("  random,     queue %2d: %6.2f GB/s\n", q, run(h, (uint64_t)sz.QuadPart, block, total, q, true));
  }
  CloseHandle(h);
  return 0;
}
