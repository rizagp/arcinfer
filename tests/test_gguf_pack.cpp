// CPU-only check of the GGUF writer used by arcinferzero packages (zero/gguf_pack.h): a GGUF is written again with
// every key (stored types kept) and tensor, read back, and compared. Usage: test_gguf_pack <in.gguf> <out.gguf>
#include "../zero/gguf_pack.h"
#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: test_gguf_pack in.gguf out.gguf\n"); return 2; }
  int fails = 0;
  {
    GgufFile src(argv[1]);
    const uint64_t align = (uint64_t)src.get_int("general.alignment", 32);
    zpack::Writer w(argv[2]);
    w.put<uint32_t>(0x46554747);
    w.put<uint32_t>(3);
    w.put<uint64_t>(src.tensors.size());
    w.put<uint64_t>(src.kv_order.size());
    for (const std::string& k : src.kv_order) {
      const auto& ty = src.kv_types.at(k);
      zpack::put_kv(w, k, ty.first, ty.second, src.kv.at(k));
    }
    uint64_t off = 0;
    for (const GgufTensor& t : src.tensors) {
      w.str(t.name);
      w.put<uint32_t>((uint32_t)t.dims.size());
      for (uint64_t d : t.dims) w.put<uint64_t>(d);
      w.put<uint32_t>(t.type);
      w.put<uint64_t>(off);
      off += (t.nbytes + align - 1) / align * align;
    }
    w.pad(align);
    for (const GgufTensor& t : src.tensors) {
      w.put(t.data, t.nbytes);
      w.pad(align);
    }
    w.close();
  }
  GgufFile a(argv[1]), b(argv[2]);
  if (a.kv_order != b.kv_order) { printf("FAIL key order\n"); ++fails; }
  for (const std::string& k : a.kv_order) {
    if (a.kv_types.at(k) != b.kv_types.at(k)) { printf("FAIL type of %s\n", k.c_str()); ++fails; }
    if (!(a.kv.at(k) == b.kv.at(k))) { printf("FAIL value of %s\n", k.c_str()); ++fails; }
  }
  if (a.tensors.size() != b.tensors.size()) { printf("FAIL tensor count\n"); ++fails; }
  for (size_t i = 0; i < a.tensors.size() && i < b.tensors.size(); ++i) {
    const GgufTensor &x = a.tensors[i], &y = b.tensors[i];
    if (x.name != y.name || x.type != y.type || x.dims != y.dims || x.nbytes != y.nbytes || memcmp(x.data, y.data, x.nbytes))
      { printf("FAIL tensor %s\n", x.name.c_str()); ++fails; }
  }
  printf("%zu keys, %zu tensors: %s\n", a.kv_order.size(), a.tensors.size(), fails ? "FAILED" : "identical");
  return fails ? 1 : 0;
}
