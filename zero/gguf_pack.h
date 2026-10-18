// arcinferzero model packages: a MoE model kept without its original GGUF.
//
//   %LOCALAPPDATA%\arcinfer\zero\models\<name>\<name>.gguf      every tensor except the routed experts, all metadata
//                                                              (the experts listed as virtual tensors: name, type,
//                                                              shape) and "arcinferzero.package" = 1
//   %LOCALAPPDATA%\arcinfer\zero\models\<name>\<name>.experts  the routed experts in the expert file layout
//                                                              (expert_store.h), built from the source GGUF
//
// make_package() reads the source GGUF (one file or split parts, from any drive: a USB disk is fine) once and writes
// both; afterwards the source can be moved away. The engine opens the package GGUF like any model; its expert file
// sits next to it.
#pragma once
#include "../src/gguf.h"
#include "expert_store.h"
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace zpack {

inline bool is_expert_tensor(const std::string& name) {
  return name.find("_exps.") != std::string::npos;  // (blk.N.ffn_{gate,up,down}_exps.weight)
}
inline std::string packages_dir() {
  const char* la = getenv("LOCALAPPDATA");
  return std::string(la ? la : ".") + "\\arcinfer\\zero\\models";
}

// ---- a GGUF v3 writer for the metadata (with the stored types) and tensors
class Writer {
 public:
  explicit Writer(const std::string& path) {
    f_ = fopen(path.c_str(), "wb");
    if (!f_) throw std::runtime_error("cannot write " + path);
  }
  ~Writer() { if (f_) fclose(f_); }
  void put(const void* p, size_t n) {
    if (n && fwrite(p, 1, n, f_) != n) throw std::runtime_error("write failed");
    pos_ += n;
  }
  template <typename T> void put(T v) { put(&v, sizeof v); }
  void str(const std::string& s) {
    put<uint64_t>(s.size());
    put(s.data(), s.size());
  }
  void pad(uint64_t align) {
    static const char z[64] = {};
    while (pos_ % align) put(z, std::min<uint64_t>(64, align - pos_ % align));
  }
  uint64_t pos() const { return pos_; }
  void close() { if (f_) { fclose(f_); f_ = nullptr; } }

 private:
  FILE* f_ = nullptr;
  uint64_t pos_ = 0;
};

// one scalar of GGUF type t
inline void put_scalar(Writer& w, uint32_t t, const GgufValue& v, size_t i = 0) {
  auto as_int = [&]() -> int64_t {
    if (auto p = std::get_if<int64_t>(&v)) return *p;
    if (auto p = std::get_if<uint64_t>(&v)) return (int64_t)*p;
    if (auto p = std::get_if<bool>(&v)) return *p;
    if (auto p = std::get_if<std::vector<int64_t>>(&v)) return (*p)[i];
    return 0;
  };
  auto as_f = [&]() -> double {
    if (auto p = std::get_if<double>(&v)) return *p;
    if (auto p = std::get_if<std::vector<double>>(&v)) return (*p)[i];
    return (double)as_int();
  };
  switch (t) {
    case 0: w.put<uint8_t>((uint8_t)as_int()); break;
    case 1: w.put<int8_t>((int8_t)as_int()); break;
    case 2: w.put<uint16_t>((uint16_t)as_int()); break;
    case 3: w.put<int16_t>((int16_t)as_int()); break;
    case 4: w.put<uint32_t>((uint32_t)as_int()); break;
    case 5: w.put<int32_t>((int32_t)as_int()); break;
    case 6: w.put<float>((float)as_f()); break;
    case 7: w.put<uint8_t>((uint8_t)(as_int() != 0)); break;
    case 10: w.put<uint64_t>((uint64_t)as_int()); break;
    case 11: w.put<int64_t>(as_int()); break;
    case 12: w.put<double>(as_f()); break;
    case 8: {
      if (auto p = std::get_if<std::string>(&v)) w.str(*p);
      else if (auto q = std::get_if<std::vector<std::string>>(&v)) w.str((*q)[i]);
      else w.str("");
      break;
    }
    default: throw std::runtime_error("gguf writer: unsupported type " + std::to_string(t));
  }
}
inline void put_kv(Writer& w, const std::string& key, uint32_t t, uint32_t et, const GgufValue& v) {
  w.str(key);
  w.put<uint32_t>(t);
  if (t != 9) return put_scalar(w, t, v);
  w.put<uint32_t>(et);
  size_t n = 0;
  if (auto p = std::get_if<std::vector<int64_t>>(&v)) n = p->size();
  else if (auto p = std::get_if<std::vector<double>>(&v)) n = p->size();
  else if (auto p = std::get_if<std::vector<std::string>>(&v)) n = p->size();
  w.put<uint64_t>(n);
  for (size_t i = 0; i < n; ++i) put_scalar(w, et, v, i);
}

struct Result {
  std::string gguf, experts;
  uint64_t rest_bytes = 0, expert_bytes = 0;
  double seconds = 0;
};

// Writes the package of `source` (the first part of a split model, or the only file) under packages_dir().
inline Result make_package(const std::string& source, int n_layer_hint = 0) {
  namespace fs = std::filesystem;
  const auto t0 = std::chrono::steady_clock::now();
  GgufFile src(source);
  std::string name = fs::path(source).stem().string();
  const size_t sp = name.find("-00001-of-");
  if (sp != std::string::npos) name.resize(sp);
  const std::string dir = packages_dir() + "\\" + name;
  fs::create_directories(dir);
  Result r;
  r.gguf = dir + "\\" + name + ".gguf";
  r.experts = dir + "\\" + name + ".experts";

  // Order: the package GGUF goes to a .tmp, then the expert file is moved or written, then the .tmp takes the
  // package's name. A step that fails leaves no half package: the .tmp is removed, and a package GGUF only appears
  // once its expert file is complete (the engine refuses a package without one).
  const std::string arch = src.get_str("general.architecture");
  // (the engine's layer count: the MTP / nextn blocks are not counted, the extra entry below holds them; with
  //  block_count the 35B's expert file listed one layer more than the engine expects and was refused)
  const int n_layer = n_layer_hint > 0 ? n_layer_hint
                                       : (int)src.get_int(arch + ".block_count", 0) - (int)src.get_int(arch + ".nextn_predict_layers", 0);
  const int n_exp = (int)src.get_int(arch + ".expert_count", 0);
  if (n_exp <= 0) throw std::runtime_error("not a MoE model (no " + arch + ".expert_count)");
  if (n_layer <= 0 || n_layer > 4096) throw std::runtime_error("unexpected layer count " + std::to_string(n_layer) + " (" + arch + ".block_count)");
  // (the same layer list as the engine's: blocks 0..n_layer, the last one the MTP / nextn block when there is one)
  std::vector<std::string> pre(n_layer + 1);
  for (int il = 0; il <= n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    if (src.find(p + "ffn_gate_exps.weight")) pre[il] = p;
  }
  // ---- 1. the package GGUF: all metadata + every non-expert tensor; the experts as virtual tensors
  std::vector<const GgufTensor*> keep;
  std::vector<std::string> vnames;
  std::vector<int64_t> vtypes, vdims;
  for (const GgufTensor& t : src.tensors) {
    if (!t.data) continue;  // (already virtual)
    if (is_expert_tensor(t.name)) {
      vnames.push_back(t.name);
      vtypes.push_back(t.type);
      for (int d = 0; d < 4; ++d) vdims.push_back(d < (int)t.dims.size() ? (int64_t)t.dims[d] : 0);
    } else {
      keep.push_back(&t);
    }
  }
  const uint64_t align = (uint64_t)src.get_int("general.alignment", 32);
  const std::string tmp = r.gguf + ".tmp";
  auto drop_tmp = [&] { std::error_code e; fs::remove(tmp, e); };
  try {
    Writer w(tmp);
    std::vector<std::string> keys;
    for (const std::string& k : src.kv_order)
      if (k.rfind("split.", 0) != 0 && k.rfind("arcinferzero.", 0) != 0) keys.push_back(k);
    w.put<uint32_t>(0x46554747);
    w.put<uint32_t>(3);
    w.put<uint64_t>(keep.size());
    w.put<uint64_t>(keys.size() + 6);
    for (const std::string& k : keys) {
      const auto& ty = src.kv_types.at(k);
      put_kv(w, k, ty.first, ty.second, src.kv.at(k));
    }
    put_kv(w, "arcinferzero.package", 4, 0, GgufValue((int64_t)1));
    put_kv(w, "arcinferzero.experts_file", 8, 0, GgufValue(fs::path(r.experts).filename().string()));
    put_kv(w, "arcinferzero.source", 8, 0, GgufValue(fs::path(source).filename().string()));
    put_kv(w, "arcinferzero.virtual.names", 9, 8, GgufValue(vnames));
    put_kv(w, "arcinferzero.virtual.types", 9, 4, GgufValue(vtypes));
    put_kv(w, "arcinferzero.virtual.dims", 9, 11, GgufValue(vdims));
    uint64_t off = 0;
    for (const GgufTensor* t : keep) {
      w.str(t->name);
      w.put<uint32_t>((uint32_t)t->dims.size());
      for (uint64_t d : t->dims) w.put<uint64_t>(d);
      w.put<uint32_t>(t->type);
      w.put<uint64_t>(off);
      off += (t->nbytes + align - 1) / align * align;
    }
    w.pad(align);
    for (const GgufTensor* t : keep) {
      w.put(t->data, t->nbytes);
      w.pad(align);
    }
    r.rest_bytes = w.pos();
    w.close();
  } catch (...) {
    drop_tmp();
    throw;
  }
  // ---- 2. the experts, through the expert store, next to the package (the engine's expert file for the source
  //  moves here when it exists, see ExpertStore::open_or_build)
  try {
    ExpertStore st;
    st.open_or_build(src, source, pre, n_exp, r.experts);
    if (!st.built) fprintf(stderr, "arcinferzero: the expert file was already there: %s\n", r.experts.c_str());
    r.expert_bytes = st.file_bytes;
  } catch (...) {
    drop_tmp();
    throw;
  }
  std::error_code ec;
  fs::rename(tmp, r.gguf, ec);
  if (ec) {
    drop_tmp();
    throw std::runtime_error("cannot rename " + tmp + ": " + ec.message());
  }
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace zpack
