// Prints the CPU dequantization of one row of a GGUF tensor (first values, sum, sum of squares), to compare the
// decoders in src/quants.h with an independent implementation. usage: test_quant_rows model.gguf TENSOR [row] [n]
// With PAD=1 and a small-block type (IQ4_NL / Q2_0) it also checks pad_small_block_rows: the padded row must
// dequantize to the same K values followed by zeros.
#include "../src/gguf.h"
#include "../src/quants.h"
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: test_quant_rows model.gguf TENSOR [row] [n]\n"); return 1; }
  GgufFile f(argv[1]);
  const GgufTensor& t = f.get(argv[2]);
  const uint64_t row = argc > 3 ? strtoull(argv[3], nullptr, 10) : 0;
  const int n = argc > 4 ? atoi(argv[4]) : 8;
  int blk = 0, bytes = 0;
  ggml_block_info(t.type, blk, bytes);
  const int K = (int)t.dims[0];
  const size_t row_bytes = (size_t)K / blk * bytes;
  std::vector<float> v(K);
  dequant_row(t.type, t.data + row * row_bytes, K, v.data());
  double s = 0, s2 = 0;
  for (float x : v) { s += x; s2 += (double)x * x; }
  printf("%s %s K=%d row %llu: sum %.6f sumsq %.6f first:", argv[2], ggml_type_name(t.type), K, (unsigned long long)row, s, s2);
  for (int i = 0; i < n && i < K; ++i) printf(" %.6f", v[i]);
  printf("\n");
  if (getenv("PAD") && is_small_block_type(t.type)) {
    const int Kp = padded_k(K);
    std::vector<uint8_t> padded((size_t)Kp / blk * bytes);
    pad_small_block_rows(t.type, t.data + row * row_bytes, 1, K, padded.data());
    std::vector<float> p(Kp);
    dequant_row(t.type, padded.data(), Kp, p.data());
    int bad = 0;
    for (int i = 0; i < Kp; ++i) {
      const float want = i < K ? v[i] : 0.0f;
      if (p[i] != want) ++bad;
    }
    printf("padded to K=%d (%zu bytes): %s\n", Kp, padded.size(), bad ? "MISMATCH" : "identical + zeros");
    return bad ? 1 : 0;
  }
  return 0;
}
