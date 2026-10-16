// Prints GGUF metadata (arrays truncated) and the tensor list with types and shapes.
// usage: gguf_dump model.gguf [max_layer_to_show]
#include "../src/gguf.h"
#include <cstdio>
#include <map>

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: gguf_dump model.gguf [max_layer]\n"); return 1; }
  int max_layer = argc > 2 ? atoi(argv[2]) : 4;
  GgufFile f(argv[1]);
  for (auto& key : f.kv_order) {
    auto& v = f.kv.at(key);
    printf("%-50s ", key.c_str());
    if (auto p = std::get_if<int64_t>(&v)) printf("%lld\n", (long long)*p);
    else if (auto p = std::get_if<uint64_t>(&v)) printf("%llu\n", (unsigned long long)*p);
    else if (auto p = std::get_if<double>(&v)) printf("%g\n", *p);
    else if (auto p = std::get_if<bool>(&v)) printf("%s\n", *p ? "true" : "false");
    else if (auto p = std::get_if<std::string>(&v)) printf("\"%.120s\"%s\n", p->c_str(), p->size() > 120 ? "..." : "");
    else if (auto p = std::get_if<std::vector<int64_t>>(&v)) {
      printf("[%zu ints]", p->size());
      for (size_t i = 0; i < p->size() && i < 48; ++i) printf(" %lld", (long long)(*p)[i]);
      printf("\n");
    } else if (auto p = std::get_if<std::vector<double>>(&v)) {
      printf("[%zu floats]", p->size());
      for (size_t i = 0; i < p->size() && i < 16; ++i) printf(" %g", (*p)[i]);
      printf("\n");
    } else if (auto p = std::get_if<std::vector<std::string>>(&v)) {
      printf("[%zu strings]", p->size());
      for (size_t i = 0; i < p->size() && i < 5; ++i) printf(" \"%s\"", (*p)[i].c_str());
      printf("\n");
    }
  }
  std::map<std::string, uint64_t> bytes_by_type;
  uint64_t total = 0;
  for (auto& t : f.tensors) {
    bytes_by_type[ggml_type_name(t.type)] += t.nbytes;
    total += t.nbytes;
    int layer = -1;
    if (sscanf(t.name.c_str(), "blk.%d.", &layer) == 1 && layer > max_layer) continue;
    printf("%-44s %-6s [", t.name.c_str(), ggml_type_name(t.type));
    for (size_t i = 0; i < t.dims.size(); ++i) printf("%s%llu", i ? ", " : "", (unsigned long long)t.dims[i]);
    printf("]\n");
  }
  printf("--- %zu tensors, %.2f GiB\n", f.tensors.size(), total / 1073741824.0);
  for (auto& [k, v] : bytes_by_type) printf("  %-6s %8.1f MiB (%.1f%%)\n", k.c_str(), v / 1048576.0, 100.0 * v / total);
  return 0;
}
