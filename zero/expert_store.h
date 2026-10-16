// arcinferzero: the routed experts of a MoE GGUF in a file of their own on the SSD, laid out for streaming:
//   %LOCALAPPDATA%\arcinfer\zero\<gguf name>-<fingerprint>.experts
// Every (layer, expert) is one record: its gate, up and down rows one after the other in the GPU raw form arcinfer
// uploads (src/quants.h gpu_raw_sb: regrouped / padded superblocks), the record padded to a multiple of 4 KB, so one
// unbuffered read (FILE_FLAG_NO_BUFFERING: no Windows file cache in RAM) brings a whole expert, ready for the GPU.
// The file is made from the GGUF when missing or when it does not match the GGUF (size, time, name: the fingerprint),
// and reused otherwise; it can be deleted at any time.
#pragma once
#include "../src/gguf.h"
#include "../src/quants.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

struct ExpertStore {
  struct LayerInfo {
    uint32_t type[3] = {};  // gate, up, down GGUF types
    uint64_t part[3] = {};  // bytes of each matrix of one expert (GPU raw form)
    uint64_t rec = 0;       // bytes of one expert record (multiple of 4096)
    uint64_t off = 0;       // file offset of the layer's expert 0
  };
  std::vector<LayerInfo> layers;  // by layer index; rec == 0: not a MoE layer
  int n_expert = 0;
  std::string path;
  uint64_t file_bytes = 0;
  double build_ms = 0;
  bool built = false;

  static constexpr uint32_t kMagic = 0x58435241u;  // "ARCX"
  static constexpr uint32_t kVersion = 1;
  static constexpr uint64_t kAlign = 4096;

  ~ExpertStore() {
    for (auto& s : slots_) if (s.ev) CloseHandle(s.ev);
    if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
  }

  // prefixes[i]: the GGUF tensor prefix of layer i ("blk.7.") when it has routed experts, else "".
  // fixed_path: write / use the expert file there (a package being made). A package GGUF ("arcinferzero.package")
  // has no expert data: its expert file next to it is used as it is (never rebuilt).
  void open_or_build(const GgufFile& f, const std::string& gguf_path, const std::vector<std::string>& prefixes, int n_exp,
                     const std::string& fixed_path = "") {
    n_expert = n_exp;
    const bool package = f.has("arcinferzero.package");
    namespace fs = std::filesystem;
    std::error_code ec;
    const uint64_t gsize = fs::file_size(gguf_path, ec);
    const auto gtime = fs::last_write_time(gguf_path, ec).time_since_epoch().count();
    const std::string name = fs::path(gguf_path).stem().string();
    uint64_t fp = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) { for (size_t i = 0; i < n; ++i) { fp ^= ((const uint8_t*)p)[i]; fp *= 1099511628211ull; } };
    mix(&gsize, 8);
    mix(&gtime, sizeof(gtime));
    mix(name.data(), name.size());
    mix(&kVersion, 4);
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)fp);
    const char* la = getenv("LOCALAPPDATA");
    const std::string dir = std::string(la ? la : ".") + "\\arcinfer\\zero";
    fs::create_directories(dir, ec);
    path = dir + "\\" + name + "-" + std::string(hex).substr(0, 8) + ".experts";
    if (!fixed_path.empty()) path = fixed_path;
    if (package) path = (fs::path(gguf_path).parent_path() / f.get_str("arcinferzero.experts_file")).string();

    // layout
    const char* names[3] = {"gate", "up", "down"};
    layers.assign(prefixes.size(), LayerInfo{});
    uint64_t off = kAlign * ((64 + prefixes.size() * 80 + kAlign - 1) / kAlign);  // after the header
    for (size_t l = 0; l < prefixes.size(); ++l) {
      if (prefixes[l].empty()) continue;
      LayerInfo& L = layers[l];
      uint64_t sum = 0;
      for (int i = 0; i < 3; ++i) {
        const GgufTensor& t = f.get(prefixes[l] + "ffn_" + names[i] + "_exps.weight");
        L.type[i] = t.type;
        // (a small-block type whose rows are not whole superblocks, Flash Next's K = 640 down rows, is padded with
        // zero blocks to padded_k: quants.h pad_small_block_rows)
        const uint64_t rows = t.dims[1], nsb = (uint64_t)padded_k((int)t.dims[0]) / 256;
        L.part[i] = rows * nsb * (uint64_t)raw_padded_sb_bytes(t.type);
        sum += L.part[i];
      }
      L.rec = (sum + kAlign - 1) / kAlign * kAlign;
      L.off = off;
      off += L.rec * n_expert;
    }
    file_bytes = off;

    if (!valid(fp, package)) {
      if (package)
        throw std::runtime_error("the package's expert file is missing or does not match: " + path +
                                 " (make the package again from the original GGUF: serverz --pack <gguf>)");
      build(f, prefixes, fp);
    }
    h_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                     FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (h_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
    for (auto& s : slots_) s.ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  }

  // ---- reads: up to kQD unbuffered reads in flight. dst must be 4 KB aligned (host USM is).
  struct Req { int layer, expert; void* dst; };
  void read(const std::vector<Req>& reqs) {
    size_t next = 0, done = 0;
    int busy = 0;
    auto issue = [&](int s) {
      const Req& r = reqs[next++];
      const LayerInfo& L = layers[r.layer];
      const uint64_t o = L.off + (uint64_t)r.expert * L.rec;
      Slot& sl = slots_[s];
      ZeroMemory(&sl.ov, sizeof sl.ov);
      sl.ov.Offset = (DWORD)o;
      sl.ov.OffsetHigh = (DWORD)(o >> 32);
      sl.ov.hEvent = sl.ev;
      ResetEvent(sl.ev);
      if (!ReadFile(h_, r.dst, (DWORD)L.rec, nullptr, &sl.ov) && GetLastError() != ERROR_IO_PENDING)
        throw std::runtime_error("expert read failed (" + std::to_string(GetLastError()) + ")");
      sl.busy = true;
      ++busy;
      bytes_read += L.rec;
    };
    for (int s = 0; s < kQD && next < reqs.size(); ++s) issue(s);
    while (done < reqs.size()) {
      HANDLE evs[kQD];
      int map[kQD], n = 0;
      for (int s = 0; s < kQD; ++s) if (slots_[s].busy) evs[n] = slots_[s].ev, map[n++] = s;
      const DWORD w = WaitForMultipleObjects((DWORD)n, evs, FALSE, INFINITE);
      const int s = map[w - WAIT_OBJECT_0];
      DWORD got = 0;
      if (!GetOverlappedResult(h_, &slots_[s].ov, &got, FALSE)) throw std::runtime_error("expert read failed");
      slots_[s].busy = false;
      --busy;
      ++done;
      if (next < reqs.size()) issue(s);
    }
    reads += reqs.size();
  }
  uint64_t reads = 0, bytes_read = 0;

 private:
  static constexpr int kQD = 16;
  struct Slot { OVERLAPPED ov{}; HANDLE ev = nullptr; bool busy = false; };
  Slot slots_[kQD];
  HANDLE h_ = INVALID_HANDLE_VALUE;

  struct Header {
    uint32_t magic, version;
    uint64_t fingerprint, layers, experts, bytes;
  };
  bool valid(uint64_t fp, bool any_fp = false) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) != file_bytes) return false;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    Header h{};
    const bool ok = fread(&h, sizeof h, 1, f) == 1 && h.magic == kMagic && h.version == kVersion && (any_fp || h.fingerprint == fp) &&
                    h.layers == layers.size() && h.experts == (uint64_t)n_expert && h.bytes == file_bytes;
    bool same = ok;
    for (size_t l = 0; same && l < layers.size(); ++l) {
      LayerInfo li{};
      same = fread(&li, sizeof li, 1, f) == 1 && memcmp(&li, &layers[l], sizeof li) == 0;
    }
    fclose(f);
    return same;
  }
  void build(const GgufFile& f, const std::vector<std::string>& prefixes, uint64_t fp) {
    const auto t0 = std::chrono::steady_clock::now();
    fprintf(stderr, "arcinferzero: writing the expert file %s (%.1f GB, once per model)\n", path.c_str(), file_bytes / 1e9);
    const std::string tmp = path + ".tmp";
    // unbuffered writes (4 KB multiples from page-aligned buffers): 15 GB through the Windows file cache filled the
    // RAM (3.9 GB left, paging) on the 32 GB laptop
    HANDLE w = CreateFileA(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (w == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot write " + tmp);
    auto put = [&](const void* p, uint64_t n) {
      while (n) {
        const DWORD c = (DWORD)std::min<uint64_t>(n, 1u << 30);
        DWORD got = 0;
        if (!WriteFile(w, p, c, &got, nullptr) || got != c) { CloseHandle(w); throw std::runtime_error("write failed: " + tmp); }
        p = (const uint8_t*)p + c;
        n -= c;
      }
    };
    // header (+ layer table), padded to the first layer's offset
    {
      uint64_t first = kAlign;
      for (auto& L : layers) if (L.rec) { first = L.off; break; }
      uint8_t* hb = (uint8_t*)VirtualAlloc(nullptr, first, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      Header h{kMagic, kVersion, fp, layers.size(), (uint64_t)n_expert, file_bytes};
      memcpy(hb, &h, sizeof h);
      memcpy(hb + sizeof h, layers.data(), layers.size() * sizeof(LayerInfo));
      put(hb, first);
      VirtualFree(hb, 0, MEM_RELEASE);
    }
    const char* names[3] = {"gate", "up", "down"};
    const int G = 32;  // experts converted per batch (in parallel)
    for (size_t l = 0; l < prefixes.size(); ++l) {
      if (prefixes[l].empty()) continue;
      const LayerInfo& L = layers[l];
      const GgufTensor* t[3];
      for (int i = 0; i < 3; ++i) t[i] = &f.get(prefixes[l] + "ffn_" + names[i] + "_exps.weight");
      struct Buf { uint8_t* p; ~Buf() { VirtualFree(p, 0, MEM_RELEASE); } } bb{(uint8_t*)VirtualAlloc(nullptr, (size_t)L.rec * G, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)};
      uint8_t* buf = bb.p;
      for (int e0 = 0; e0 < n_expert; e0 += G) {
        const int ne = std::min(G, n_expert - e0);
        memset(buf, 0, (size_t)ne * L.rec);
#pragma omp parallel for
        for (int j = 0; j < ne; ++j) {
          uint8_t* dst = buf + (size_t)j * L.rec;
          for (int i = 0; i < 3; ++i) {
            const uint32_t ty = L.type[i];
            const size_t bb = gguf_sb_bytes(ty), bp = raw_padded_sb_bytes(ty);
            const size_t nsb = L.part[i] / bp;  // superblocks of one expert matrix
            const int K = (int)t[i]->dims[0], rows = (int)t[i]->dims[1];
            const uint8_t* src;
            std::vector<uint8_t> padded;
            if (K % 256) {  // rows padded to whole superblocks first
              int blk = 0, bytes = 0;
              ggml_block_info(ty, blk, bytes);
              padded.resize(nsb * bb);
              pad_small_block_rows(ty, t[i]->data + (size_t)(e0 + j) * rows * (K / blk) * bytes, rows, K, padded.data());
              src = padded.data();
            } else {
              src = t[i]->data + (size_t)(e0 + j) * nsb * bb;
            }
            for (size_t s = 0; s < nsb; ++s) gpu_raw_sb(ty, src + s * bb, dst + s * bp);
            dst += L.part[i];
          }
        }
        put(buf, (uint64_t)ne * L.rec);
      }
      // the layer's GGUF pages leave the working set (they stay in the standby list, which Windows can reuse)
      SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
      fprintf(stderr, "\r  layer %zu / %zu", l + 1, prefixes.size());
    }
    fprintf(stderr, "\n");
    CloseHandle(w);
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) throw std::runtime_error("cannot rename " + tmp + ": " + ec.message());
    build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    built = true;
  }
};
