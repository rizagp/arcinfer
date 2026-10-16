// Compares the residual stream after each half of every layer between decoding two tokens one by one and as one
// 2-token pass (the speculative-verification path), which must be bit-identical for MTP.
// usage: bitcheck model.gguf
#include "../src/engine.h"
#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: bitcheck model.gguf\n"); return 1; }
  EngineOptions eo;
  eo.ctx = 256;
  Engine eng(argv[1], eo);
  const int P[] = {248045, 846, 198, 213077, 173634, 172069, 23220, 30, 243381, 189040, 13, 248046, 198, 248045, 74455, 198};
  const int np = sizeof P / sizeof P[0], ab[2] = {248068, 271};
  eng.forward(P, np);
  eng.forward(&ab[0], 1);
  eng.dbg_rows = true;
  const std::vector<float> l1 = eng.forward(&ab[1], 1);
  auto one = eng.dbg_;
  eng.dbg_rows = false;
  eng.dbg_.clear();
  eng.reset();
  eng.forward(P, np);
  eng.dbg_rows = true;
  const std::vector<float> l2 = eng.forward(ab, 2);
  auto two = eng.dbg_;
  double ld = 0;
  for (size_t j = 0; j < l1.size(); ++j) ld = std::max(ld, (double)std::fabs(l1[j] - l2[j]));
  printf("logits: max diff %g\n", ld);
  int first = -1;
  for (size_t i = 0; i < one.size() && i < two.size(); ++i) {
    double md = 0;
    for (size_t j = 0; j < one[i].size(); ++j) md = std::max(md, (double)std::fabs(one[i][j] - two[i][j]));
    printf("layer %zu %s: max diff %g\n", i / 2, i % 2 ? "ffn " : "attn", md);
    if (md > 0 && first < 0) first = (int)i;
  }
  printf(first < 0 ? "bit-identical\n" : "first difference: layer %d %s\n", first / 2, first % 2 ? "ffn" : "attn");
  if (argc > 2) {  // with the MTP block: the speculative flow (draft, verify, reject / accept) against plain decoding
    EngineOptions em;
    em.ctx = 256;
    em.mtp = true;
    em.mtp_drafts = 1;
    Engine e2(argv[1], em);
    const size_t V = e2.hp.n_vocab;
    auto maxdiff = [&](const float* x, const float* y) {
      double d = 0;
      for (size_t j = 0; j < V; ++j) d = std::max(d, (double)std::fabs(x[j] - y[j]));
      return d;
    };
    e2.forward(P, np);
    e2.forward(&ab[0], 1);
    const std::vector<float> la = e2.forward(&ab[1], 1);
    e2.reset();
    e2.forward(P, np);
    e2.mtp_draft(ab[0], 1);
    e2.verify(ab[0], {ab[1] == 0 ? 1 : 0});  // a wrong draft
    e2.verify_done(0);
    const std::vector<int> d2 = e2.mtp_draft(ab[1], 1);
    const std::vector<float> lr = e2.verify(ab[1], d2);
    printf("rejected draft, then the next token: logits max diff %g\n", maxdiff(la.data(), lr.data()));
    e2.reset();
    e2.forward(P, np);
    e2.mtp_draft(ab[0], 1);
    const std::vector<float> lc = e2.verify(ab[0], {ab[1]});
    printf("accepted draft: logits max diff %g\n", maxdiff(la.data(), lc.data() + V));
  }
  return 0;
}
