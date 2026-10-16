// GGUF v3 reader: memory-maps the file and exposes metadata + tensor views. A split model (name-00001-of-0000N.gguf,
// "split.count" > 1) is opened from its first part: the other parts are mapped too and their tensors added.
// A tensor listed under "arcinferzero.virtual.*" (an arcinferzero package: the routed experts live in its own
// expert file) has its name, type and shape but no data (data == nullptr).
#pragma once
// (see enc_weight_bytes below: the encoders' GPU footprint for --ctx auto)
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <initializer_list>
#include <unordered_map>
#include <variant>
#include <stdexcept>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

enum GgmlType : uint32_t {
  GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7, GGML_Q8_0 = 8,
  GGML_Q8_1 = 9, GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14,
  GGML_Q8_K = 15, GGML_IQ2_XXS = 16, GGML_IQ2_XS = 17, GGML_IQ3_XXS = 18, GGML_IQ1_S = 19, GGML_IQ4_NL = 20,
  GGML_IQ3_S = 21, GGML_IQ2_S = 22, GGML_IQ4_XS = 23, GGML_I8 = 24, GGML_I16 = 25, GGML_I32 = 26, GGML_IQ1_M = 29,
  GGML_BF16 = 30, GGML_MXFP4 = 39,
  GGML_Q2_0 = 42,  // blocks of 64: f16 scale + 2-bit codes, value (q - 1) * d (Qwen3.8 Flash Next down projections)
};

inline const char* ggml_type_name(uint32_t t) {
  switch (t) {
    case GGML_F32: return "F32"; case GGML_F16: return "F16"; case GGML_Q4_0: return "Q4_0";
    case GGML_Q4_1: return "Q4_1"; case GGML_Q5_0: return "Q5_0"; case GGML_Q5_1: return "Q5_1";
    case GGML_Q8_0: return "Q8_0"; case GGML_Q2_K: return "Q2_K"; case GGML_Q3_K: return "Q3_K";
    case GGML_Q4_K: return "Q4_K"; case GGML_Q5_K: return "Q5_K"; case GGML_Q6_K: return "Q6_K";
    case GGML_IQ4_NL: return "IQ4_NL"; case GGML_IQ4_XS: return "IQ4_XS"; case GGML_BF16: return "BF16";
    case GGML_IQ2_XXS: return "IQ2_XXS"; case GGML_IQ2_XS: return "IQ2_XS"; case GGML_IQ3_XXS: return "IQ3_XXS";
    case GGML_IQ1_S: return "IQ1_S"; case GGML_IQ3_S: return "IQ3_S"; case GGML_IQ2_S: return "IQ2_S"; case GGML_IQ1_M: return "IQ1_M";
    case GGML_MXFP4: return "MXFP4"; case GGML_Q2_0: return "Q2_0"; default: return "?";
  }
}

// (elements per block, bytes per block)
inline void ggml_block_info(uint32_t t, int& blk, int& bytes) {
  switch (t) {
    case GGML_F32: blk = 1; bytes = 4; return;
    case GGML_F16: case GGML_BF16: case GGML_I16: blk = 1; bytes = 2; return;
    case GGML_I8: blk = 1; bytes = 1; return;
    case GGML_I32: blk = 1; bytes = 4; return;
    case GGML_Q4_0: blk = 32; bytes = 18; return;
    case GGML_Q4_1: blk = 32; bytes = 20; return;
    case GGML_Q5_0: blk = 32; bytes = 22; return;
    case GGML_Q5_1: blk = 32; bytes = 24; return;
    case GGML_Q8_0: blk = 32; bytes = 34; return;
    case GGML_Q2_K: blk = 256; bytes = 84; return;
    case GGML_Q3_K: blk = 256; bytes = 110; return;
    case GGML_Q4_K: blk = 256; bytes = 144; return;
    case GGML_Q5_K: blk = 256; bytes = 176; return;
    case GGML_Q6_K: blk = 256; bytes = 210; return;
    case GGML_IQ4_NL: blk = 32; bytes = 18; return;
    case GGML_IQ4_XS: blk = 256; bytes = 136; return;
    case GGML_IQ2_XXS: blk = 256; bytes = 66; return;
    case GGML_IQ2_XS: blk = 256; bytes = 74; return;
    case GGML_IQ2_S: blk = 256; bytes = 82; return;
    case GGML_IQ3_XXS: blk = 256; bytes = 98; return;
    case GGML_IQ3_S: blk = 256; bytes = 110; return;
    case GGML_IQ1_S: blk = 256; bytes = 50; return;
    case GGML_IQ1_M: blk = 256; bytes = 56; return;
    case GGML_MXFP4: blk = 32; bytes = 17; return;
    case GGML_Q2_0: blk = 64; bytes = 18; return;
    default: throw std::runtime_error("unsupported ggml type " + std::to_string(t));
  }
}

struct GgufTensor {
  std::string name;
  uint32_t type = 0;
  std::vector<uint64_t> dims;  // dims[0] is the contiguous (innermost) dimension
  const uint8_t* data = nullptr;
  uint64_t nbytes = 0;
  uint64_t nelements() const { uint64_t n = 1; for (auto d : dims) n *= d; return n; }
};

using GgufValue = std::variant<int64_t, uint64_t, double, bool, std::string, std::vector<int64_t>, std::vector<double>,
                               std::vector<std::string>>;

class GgufFile {
 public:
  std::unordered_map<std::string, GgufValue> kv;
  std::vector<std::string> kv_order;
  // the stored GGUF type of every key (and of its elements, for arrays): kept for writing the metadata again
  std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> kv_types;
  std::string path;
  std::vector<GgufTensor> tensors;
  std::unordered_map<std::string, size_t> tensor_index;

  explicit GgufFile(const std::string& path_) : path(path_) {
    file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(file_, &sz);
    size_ = (uint64_t)sz.QuadPart;
    map_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map_) throw std::runtime_error("CreateFileMapping failed");
    base_ = (const uint8_t*)MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
    if (!base_) throw std::runtime_error("MapViewOfFile failed");
    parse();
    open_parts();
    add_virtual();
  }
  ~GgufFile() {
    if (base_) UnmapViewOfFile(base_);
    if (map_) CloseHandle(map_);
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
  }
  GgufFile(const GgufFile&) = delete;
  GgufFile& operator=(const GgufFile&) = delete;

  const GgufTensor* find(const std::string& name) const {
    auto it = tensor_index.find(name);
    return it == tensor_index.end() ? nullptr : &tensors[it->second];
  }
  const GgufTensor& get(const std::string& name) const {
    auto t = find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    return *t;
  }
  bool has(const std::string& key) const { return kv.count(key) > 0; }
  int64_t get_int(const std::string& key, int64_t def = -1) const {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    if (auto p = std::get_if<int64_t>(&it->second)) return *p;
    if (auto p = std::get_if<uint64_t>(&it->second)) return (int64_t)*p;
    if (auto p = std::get_if<bool>(&it->second)) return *p;
    return def;
  }
  double get_float(const std::string& key, double def = 0) const {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    if (auto p = std::get_if<double>(&it->second)) return *p;
    if (auto p = std::get_if<int64_t>(&it->second)) return (double)*p;
    if (auto p = std::get_if<uint64_t>(&it->second)) return (double)*p;
    return def;
  }
  std::string get_str(const std::string& key, const std::string& def = "") const {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    if (auto p = std::get_if<std::string>(&it->second)) return *p;
    return def;
  }
  const std::vector<std::string>* get_strs(const std::string& key) const {
    auto it = kv.find(key);
    return it == kv.end() ? nullptr : std::get_if<std::vector<std::string>>(&it->second);
  }
  const std::vector<int64_t>* get_ints(const std::string& key) const {
    auto it = kv.find(key);
    return it == kv.end() ? nullptr : std::get_if<std::vector<int64_t>>(&it->second);
  }
  const std::vector<double>* get_floats(const std::string& key) const {
    auto it = kv.find(key);
    return it == kv.end() ? nullptr : std::get_if<std::vector<double>>(&it->second);
  }

 private:
  HANDLE file_ = INVALID_HANDLE_VALUE, map_ = nullptr;
  const uint8_t* base_ = nullptr;
  uint64_t size_ = 0, pos_ = 0;

  std::vector<std::unique_ptr<GgufFile>> parts_;  // the other parts of a split model

  // name-00001-of-0000N.gguf: the parts 2..N, their tensors added to this one's index
  void open_parts() {
    const int64_t n = get_int("split.count", 1);
    if (n <= 1) return;
    const size_t at = path.rfind("-00001-of-");
    if (at == std::string::npos) return;  // (a later part opened on its own: only its own tensors)
    for (int64_t k = 2; k <= n; ++k) {
      char num[8];
      snprintf(num, sizeof num, "%05d", (int)k);
      std::string pp = path;
      pp.replace(at + 1, 5, num);
      parts_.push_back(std::make_unique<GgufFile>(pp));
      for (const GgufTensor& t : parts_.back()->tensors) {
        if (tensor_index.count(t.name)) continue;
        tensor_index[t.name] = tensors.size();
        tensors.push_back(t);
      }
    }
  }
  // tensors without data (arcinferzero packages): names, types and shapes from the metadata
  void add_virtual() {
    const auto* names = get_strs("arcinferzero.virtual.names");
    const auto* types = get_ints("arcinferzero.virtual.types");
    const auto* dims = get_ints("arcinferzero.virtual.dims");  // 4 per tensor (unused trailing dims = 0)
    if (!names || !types || !dims || types->size() != names->size() || dims->size() != names->size() * 4) return;
    for (size_t i = 0; i < names->size(); ++i) {
      if (tensor_index.count((*names)[i])) continue;
      GgufTensor t;
      t.name = (*names)[i];
      t.type = (uint32_t)(*types)[i];
      for (int d = 0; d < 4; ++d)
        if ((*dims)[i * 4 + d] > 0) t.dims.push_back((uint64_t)(*dims)[i * 4 + d]);
      int blk, bytes;
      ggml_block_info(t.type, blk, bytes);
      t.nbytes = t.nelements() / blk * bytes;
      tensor_index[t.name] = tensors.size();
      tensors.push_back(t);
    }
  }

  template <typename T> T rd() {
    if (pos_ + sizeof(T) > size_) throw std::runtime_error("gguf: truncated");
    T v;
    memcpy(&v, base_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }
  std::string rd_str() {
    uint64_t n = rd<uint64_t>();
    if (pos_ + n > size_) throw std::runtime_error("gguf: truncated string");
    std::string s((const char*)base_ + pos_, n);
    pos_ += n;
    return s;
  }
  int64_t rd_int(uint32_t t) {
    switch (t) {
      case 0: return rd<uint8_t>(); case 1: return rd<int8_t>(); case 2: return rd<uint16_t>();
      case 3: return rd<int16_t>(); case 4: return rd<uint32_t>(); case 5: return rd<int32_t>();
      case 7: return rd<uint8_t>(); case 10: return (int64_t)rd<uint64_t>(); case 11: return rd<int64_t>();
      default: throw std::runtime_error("gguf: not an int type");
    }
  }
  GgufValue rd_value(uint32_t t) {
    switch (t) {
      case 6: return (double)rd<float>();
      case 12: return rd<double>();
      case 7: return rd<uint8_t>() != 0;
      case 8: return rd_str();
      case 10: return rd<uint64_t>();
      case 9: {
        uint32_t et = rd<uint32_t>();
        uint64_t n = rd<uint64_t>();
        if (et == 8) { std::vector<std::string> v(n); for (auto& s : v) s = rd_str(); return v; }
        if (et == 6 || et == 12) { std::vector<double> v(n); for (auto& x : v) x = et == 6 ? rd<float>() : rd<double>(); return v; }
        std::vector<int64_t> v(n);
        for (auto& x : v) x = rd_int(et);
        return v;
      }
      default: return rd_int(t);
    }
  }
  void parse() {
    if (rd<uint32_t>() != 0x46554747) throw std::runtime_error("not a GGUF file");
    uint32_t ver = rd<uint32_t>();
    if (ver < 2) throw std::runtime_error("unsupported GGUF version");
    uint64_t nt = rd<uint64_t>(), nkv = rd<uint64_t>();
    for (uint64_t i = 0; i < nkv; ++i) {
      std::string key = rd_str();
      uint32_t t = rd<uint32_t>();
      uint32_t et = 0;
      if (t == 9) memcpy(&et, base_ + pos_, 4);  // (an array: its element type comes next)
      kv[key] = rd_value(t);
      kv_types[key] = {t, et};
      kv_order.push_back(key);
    }
    tensors.resize(nt);
    std::vector<uint64_t> offs(nt);
    for (uint64_t i = 0; i < nt; ++i) {
      auto& t = tensors[i];
      t.name = rd_str();
      uint32_t nd = rd<uint32_t>();
      t.dims.resize(nd);
      for (auto& d : t.dims) d = rd<uint64_t>();
      t.type = rd<uint32_t>();
      offs[i] = rd<uint64_t>();
      tensor_index[t.name] = i;
    }
    uint64_t align = (uint64_t)get_int("general.alignment", 32);
    uint64_t data_start = (pos_ + align - 1) / align * align;
    for (uint64_t i = 0; i < nt; ++i) {
      auto& t = tensors[i];
      int blk, bytes;
      ggml_block_info(t.type, blk, bytes);
      t.nbytes = t.nelements() / blk * bytes;
      t.data = base_ + data_start + offs[i];
      if (data_start + offs[i] + t.nbytes > size_) throw std::runtime_error("gguf: tensor out of range " + t.name);
    }
  }
};

// GPU bytes of the tensors whose names start with one of `prefixes`, as the GPU encoders upload them: matrices
// (2+ dims) as f16, vectors as f32; `skip` names stay on the host. (--ctx auto reserves the encoders' memory.)
inline size_t enc_weight_bytes(const GgufFile& g, std::initializer_list<const char*> prefixes,
                               std::initializer_list<const char*> skip = {}) {
  size_t b = 0;
  for (const auto& t : g.tensors) {
    bool in = false;
    for (const char* p : prefixes) in = in || t.name.rfind(p, 0) == 0;
    for (const char* s : skip) in = in && t.name != s;
    if (in) b += t.nelements() * (t.dims.size() >= 2 ? 2 : 4);
  }
  return b;
}
