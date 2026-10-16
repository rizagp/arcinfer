// Random reads from a file the way the expert cache does them (unbuffered, overlapped, page-aligned buffers):
// throughput by queue depth and record size, to tell the drive's limit from the engine's.
// usage: ssd_bench FILE [record_bytes=2232320] [seconds=4]
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ssd_bench FILE [record_bytes] [seconds]\n"); return 1; }
  const size_t rec = argc > 2 ? (size_t)atoll(argv[2]) / 4096 * 4096 : 2232320;
  const double secs = argc > 3 ? atof(argv[3]) : 4.0;
  HANDLE h = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
  if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  LARGE_INTEGER sz;
  GetFileSizeEx(h, &sz);
  const uint64_t n = (uint64_t)sz.QuadPart / rec;
  std::mt19937_64 rng(1);
  printf("%s: %.1f GB, records of %zu bytes\n", argv[1], sz.QuadPart / 1e9, rec);
  for (int qd : {1, 2, 4, 8, 16, 32}) {
    std::vector<char*> buf(qd);
    std::vector<OVERLAPPED> ov(qd);
    std::vector<HANDLE> ev(qd);
    for (int i = 0; i < qd; ++i) {
      buf[i] = (char*)VirtualAlloc(nullptr, rec, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      ev[i] = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    }
    auto issue = [&](int i) {
      const uint64_t off = (rng() % n) * rec;
      ZeroMemory(&ov[i], sizeof ov[i]);
      ov[i].Offset = (DWORD)off;
      ov[i].OffsetHigh = (DWORD)(off >> 32);
      ov[i].hEvent = ev[i];
      ResetEvent(ev[i]);
      if (!ReadFile(h, buf[i], (DWORD)rec, nullptr, &ov[i]) && GetLastError() != ERROR_IO_PENDING) {
        fprintf(stderr, "read failed %lu\n", GetLastError());
        exit(1);
      }
    };
    for (int i = 0; i < qd; ++i) issue(i);
    uint64_t bytes = 0, reads = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
      const DWORD w = WaitForMultipleObjects((DWORD)qd, ev.data(), FALSE, INFINITE);
      const int i = (int)(w - WAIT_OBJECT_0);
      DWORD got = 0;
      GetOverlappedResult(h, &ov[i], &got, FALSE);
      bytes += got;
      ++reads;
      issue(i);
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    WaitForMultipleObjects((DWORD)qd, ev.data(), TRUE, INFINITE);
    printf("QD %2d: %6.2f GB/s, %.2f ms per read\n", qd, bytes / t / 1e9, t * 1000 / reads * qd);
    for (int i = 0; i < qd; ++i) { VirtualFree(buf[i], 0, MEM_RELEASE); CloseHandle(ev[i]); }
  }
  CloseHandle(h);
  return 0;
}
