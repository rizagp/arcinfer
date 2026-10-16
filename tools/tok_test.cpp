// Checks the tokenizer against reference token ids (e.g. from llama-tokenize) and round-trips decode.
// usage: tok_test model.gguf text.txt ids.txt
#include "../src/tokenizer.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

static std::string slurp(const char* p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: tok_test model.gguf text.txt ids.txt\n"); return 1; }
  GgufFile f(argv[1]);
  Tokenizer tok(f);
  const std::string text = slurp(argv[2]);
  std::vector<int> ref;
  {
    std::stringstream ss(slurp(argv[3]));
    std::string s;
    while (std::getline(ss, s, ',')) if (!s.empty()) ref.push_back(std::stoi(s));
  }
  auto t0 = std::chrono::steady_clock::now();
  const std::vector<int> ids = tok.encode(text, true);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  size_t first_diff = std::min(ids.size(), ref.size());
  for (size_t i = 0; i < std::min(ids.size(), ref.size()); ++i)
    if (ids[i] != ref[i]) { first_diff = i; break; }
  const bool same = ids == ref;
  printf("chars %zu | ours %zu tokens (%.1f ms) | reference %zu tokens | %s", text.size(), ids.size(), ms, ref.size(),
         same ? "IDENTICAL" : "DIFFERENT");
  if (!same) {
    printf(" at %zu:", first_diff);
    for (size_t i = first_diff; i < first_diff + 6 && i < ids.size(); ++i) printf(" %d[%s]", ids[i], tok.piece(ids[i]).c_str());
    printf(" | ref:");
    for (size_t i = first_diff; i < first_diff + 6 && i < ref.size(); ++i) printf(" %d[%s]", ref[i], tok.piece(ref[i]).c_str());
  }
  printf(" | round-trip %s\n", tok.decode(ids) == text ? "OK" : "MISMATCH");
  return same ? 0 : 2;
}
