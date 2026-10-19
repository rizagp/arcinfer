// arcinfer driver: feeds token ids, prints top logits and a greedy continuation; or benchmarks prefill.
// usage:
//   run model.gguf (--tokens 1,2,3 | --tokens-file ids.txt) [--n 16] [--prof] [--chunk N] [--ctx N]
//   run model.gguf --bench-pp 2048 [--prof]
// options: --attn-f32 (f32 attention scores), --act256 (per-256 activation scales), --budget GB,
//          --kl-ref file (KL against reference log-probs), --show (print the generated text)
#include "../src/full_speed.h"
#include "engine.h"  // arcinferzero (Level Zero)
#include "../src/tokenizer.h"
#include <cstdio>
#include <cwctype>
#include <fstream>
#include <sstream>
#include <tlhelp32.h>

static std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) if (!tok.empty() && tok != "\r" && tok != "\n") v.push_back(std::stoi(tok));
  return v;
}

static void top_k(const std::vector<float>& logits, int k, std::vector<std::pair<int, float>>& out) {
  std::vector<int> idx(logits.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  float mx = logits[idx[0]];
  double z = 0;
  for (float l : logits) z += std::exp((double)l - mx);
  out.clear();
  for (int i = 0; i < k; ++i) out.push_back({idx[i], (float)(std::exp((double)logits[idx[i]] - mx) / z)});
}

// Other processes on the iGPU (arcinfer, llama.cpp) share its time and its memory, so speeds measured while one of
// them runs are not representative: Gemma 4 E4B generated 8 tok/s with --mtp (23 alone) while a 27B server
// processed a prompt, and its load took 10-40 s instead of 2 s with an idle 27B server loaded.
static void warn_gpu_neighbours() {
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snap == INVALID_HANDLE_VALUE) return;
  PROCESSENTRY32W pe{};
  pe.dwSize = sizeof(pe);
  std::string found;
  for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
    if (pe.th32ProcessID == GetCurrentProcessId()) continue;
    std::string n;
    for (const wchar_t* c = pe.szExeFile; *c; ++c) n.push_back((char)towlower(*c));
    if (n == "server.exe" || n == "run.exe" || n.rfind("llama-", 0) == 0)
      found += (found.empty() ? "" : ", ") + n + " (pid " + std::to_string(pe.th32ProcessID) + ")";
  }
  CloseHandle(snap);
  if (!found.empty())
    fprintf(stderr, "note: other GPU processes are running: %s. They share the iGPU and its memory, so the speeds "
                    "measured now are not representative.\n", found.c_str());
}

static int run_main(int argc, char** argv);
int main(int argc, char** argv) {
  full_speed_process();
  try {
    return run_main(argc, argv);
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

static int run_main(int argc, char** argv) {
  const bool help = argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
  if (argc < 2 || help) {
    fprintf(stderr,
            "usage: runz model.gguf PROMPT [options]  feeds a prompt, prints the top next tokens and a greedy continuation\n"
            "       runz model.gguf --bench-pp N [--n N] prompt processing of N tokens, then generation speed\n"
            "prompt:\n"
            "  --tokens 1,2,3        token ids               --tokens-file F   token ids from a file (comma separated)\n"
            "  --text-file F         text, tokenized with the model's vocabulary (special tokens such as <|im_start|> parsed)\n"
            "  --dump-ids F          write the prompt's token ids to F and exit\n"
            "generation:\n"
            "  --n N                 tokens to generate (default 16)     --show   print the generated text\n"
            "  --dump-logits F       write the prompt logits and every generation step's (f32) to F, to compare runs bitwise\n"
            "  --mtp                 speculative decoding (MTP block / Gemma drafter)  --mtp-drafts N  drafts per step\n"
            "  --dflash DIR          speculative decoding with a DFlash2 block drafter (Qwen3.8 27B, Qwen3.6 35B-A3B)\n"
            "  --draft auto|mtp|dflash  with --dflash: both drafters picked per step (auto), the MTP block only, or\n"
            "                        DFlash2 only (the MTP block is not loaded)\n"
            "  --mtp-min-p P         draft a second token only above this MTP probability (default 0.5)\n"
            "engine:\n"
            "  --ctx N               context (default: what the prompt needs)     --chunk N   prompt tokens per step\n"
            "  --feed-step N         feed the prompt in calls of N tokens (tests resuming mid-prompt)\n"
            "  --budget GB           keep decoded weights resident up to GB\n"
            "  --attn-f32            f32 attention (f16 K/V)      --act256    one activation scale per 256 values (not 32)\n"
            "  --v4                  + a 4-bit copy of V that generation reads (Qwen, experimental; more memory)\n"
            "profiling:\n"
            "  --prof                time to first token and per-kernel profile of the prompt\n"
            "  --prof-gen            per-kernel profile of generation (ARC_PROF_LAYERS=8: per group of 8 layers)\n"
            "accuracy:\n"
            "  --kl-ref F            KL of the next-token distributions against an f32 reference (tools\\ref_forward)\n"
            "  --kl-decode           with --kl-ref: through the token-by-token generation path\n"
            "  --perplexity F        perplexity of a text file (llama-perplexity method: second half of each chunk scored)\n"
            "  --ppl-ctx N           chunk length (default 512)   --ppl-chunks N   chunks to score (default all)\n"
            "  --ppl-dump F          write the chunks' token ids to F (for ref_forward)\n"
            "  --ppl-kl-save F       save each scored token's top-64 log-probs   --ppl-kl F   KL against such a file\n"
            "  --ppl-score-last N    score only the last N positions of each chunk   --ppl-score-step S  feed those S at a\n"
            "                        time (S <= 8: through the generation / verification kernels)\n"
            "PROMPT is text (special tokens such as <|im_start|> parsed); a long one: --text-file (Windows limits a\n"
            "command line to ~32K characters).\n");
    return help ? 0 : 1;
  }
  std::string model = argv[1], tokens_arg, kl_ref, ppl_file, ppl_dump, text_prompt, dump_ids;
  std::string ppl_kl_save, ppl_kl;
  int ppl_ctx = 512, ppl_chunks = 0, ppl_score_last = 0, ppl_score_step = 0;
  int n_gen = 16, chunk = 0, ctx = 0, bench_pp = 0, feed_step = 0;
  EngineOptions eo;
  bool prof = false, show = false, prof_gen = false, kl_decode = false;
  std::string dump_logits;  // every generation step's logits (f32), to compare two builds / settings bitwise
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--tokens") tokens_arg = argv[++i];
    else if (a == "--tokens-file") {
      std::ifstream f(argv[++i]);
      std::stringstream ss;
      ss << f.rdbuf();
      tokens_arg = ss.str();
    }
    else if (a == "--text-file") {  // prompt text (special tokens parsed), tokenized with the model's vocabulary
      std::ifstream f(argv[++i], std::ios::binary);
      std::stringstream ss;
      ss << f.rdbuf();
      text_prompt = ss.str();
    }
    else if (a == "--dump-ids") dump_ids = argv[++i];  // write the prompt's token ids (comma separated) and exit
    else if (a == "--feed-step") feed_step = atoi(argv[++i]);
    else if (a == "--n") n_gen = atoi(argv[++i]);
    else if (a == "--chunk") chunk = atoi(argv[++i]);
    else if (a == "--ctx") ctx = atoi(argv[++i]);
    else if (a == "--bench-pp") bench_pp = atoi(argv[++i]);
    else if (a == "--prof") prof = true;
    else if (a == "--show") show = true;
    else if (a == "--dump-logits") dump_logits = argv[++i];
    else if (a == "--prof-gen") prof_gen = true;
    else if (a == "--act256") eo.act32_all = false;
    else if (a == "--attn-f32") eo.attn_int8 = false;
    else if (a == "--v4") eo.v4 = true;  // (4-bit V cache, Qwen)
    else if (a == "--mtp") eo.mtp = true;
    else if (a == "--dflash") { eo.dflash = argv[++i]; eo.mtp = true; }  // DFlash2 drafter folder
    else if (a == "--draft") {  // auto: DFlash2 and the MTP block, picked per step; mtp: the block only; dflash: DFlash2 only
      const std::string d = argv[++i];
      if (d == "mtp") eo.dflash.clear();
      else if (d == "dflash") eo.mtp_block = false;
      else if (d != "auto") { fprintf(stderr, "--draft: auto, mtp or dflash\n"); return 1; }
    }
    else if (a == "--mtp-drafts") eo.mtp_drafts = atoi(argv[++i]);
    else if (a == "--mtp-min-p") eo.mtp_min_p = (float)atof(argv[++i]);
    else if (a == "--kl-ref") kl_ref = argv[++i];
    else if (a == "--kl-decode") kl_decode = true;  // with --kl-ref of every position: feed the prompt token by token
    else if (a == "--perplexity") ppl_file = argv[++i];
    else if (a == "--ppl-dump") ppl_dump = argv[++i];
    else if (a == "--ppl-kl-save") ppl_kl_save = argv[++i];
    else if (a == "--ppl-score-last") ppl_score_last = atoi(argv[++i]);
    else if (a == "--ppl-score-step") ppl_score_step = atoi(argv[++i]);
    else if (a == "--ppl-kl") ppl_kl = argv[++i];
    else if (a == "--ppl-ctx") ppl_ctx = atoi(argv[++i]);
    else if (a == "--ppl-chunks") ppl_chunks = atoi(argv[++i]);
    else if (a == "--budget") eo.w8_budget_gb = atof(argv[++i]);
    else if (!a.empty() && a[0] != '-' && text_prompt.empty()) text_prompt = a;  // PROMPT (special tokens parsed)
    else { fprintf(stderr, "unknown option %s (--help: the options)\n", a.c_str()); return 1; }
  }
  std::vector<int> toks = parse_ids(tokens_arg);
  if (!text_prompt.empty()) {  // the vocabulary is read from the GGUF before the engine loads
    GgufFile g(model);
    toks = Vocab(g).encode(text_prompt);
  }
  if (!dump_ids.empty()) {
    FILE* fo = fopen(dump_ids.c_str(), "w");
    for (size_t i = 0; i < toks.size(); ++i) fprintf(fo, i ? ",%d" : "%d", toks[i]);
    fclose(fo);
    printf("%zu tokens -> %s\n", toks.size(), dump_ids.c_str());
    return 0;
  }
  const int need = std::max((bench_pp > 0 ? bench_pp : (int)toks.size()) + n_gen + 16, ppl_file.empty() ? 0 : ppl_ctx + 16);
  if (!ppl_file.empty() && chunk == 0) chunk = std::min(ppl_ctx, 8192);  // (longer chunks: fed in pieces of 8192)
  eo.ctx = std::max(ctx, need);
  eo.chunk = chunk;
  warn_gpu_neighbours();
  Engine eng(model, eo);
  Vocab vocab(eng.gguf());
  printf("loaded: %d layers, n_embd %d, vocab %d | GPU buffers %.2f GiB | W8 %s | chunk %d | ctx %d | load %.0f ms\n",
         eng.hp.n_layer, eng.hp.n_embd, eng.hp.n_vocab, eng.gpu_bytes / 1073741824.0,
         eng.w8_resident ? "resident" : "just-in-time", eng.max_chunk, eng.ctx, eng.load_ms);

  if (!ppl_file.empty()) {
    // Perplexity like llama-perplexity: split the tokenized text into ppl_ctx chunks, evaluate each from an
    // empty state and score the second half (logits at j predict token j+1, j in [ctx/2, ctx-1)).
    std::ifstream fin(ppl_file, std::ios::binary);
    std::stringstream ss;
    ss << fin.rdbuf();
    const std::vector<int> all = vocab.encode(ss.str(), false);
    const bool add_bos = eng.gguf().get_int("tokenizer.ggml.add_bos_token", 0) != 0;
    int nchunks = (int)(all.size() / ppl_ctx);
    if (ppl_chunks > 0) nchunks = std::min(nchunks, ppl_chunks);
    if (!ppl_dump.empty()) {  // write the chunks' token ids (one comma-separated line per chunk) for ref_forward
      FILE* fo = fopen(ppl_dump.c_str(), "w");
      for (int ch = 0; ch < nchunks; ++ch) {
        for (int j = 0; j < ppl_ctx; ++j) fprintf(fo, j ? ",%d" : "%d", all[(size_t)ch * ppl_ctx + j]);
        fprintf(fo, "\n");
      }
      fclose(fo);
      return 0;
    }
    double nll = 0;
    long count = 0;
    // (--ppl-kl-save F / --ppl-kl F: per scored token the base run's top 64 probabilities, and KL(base || this)
    //  over those 64 plus the rest of the mass as one bucket; also how often the top token is the same)
    constexpr int KT = 64;
    FILE* kls = ppl_kl_save.empty() ? nullptr : fopen(ppl_kl_save.c_str(), "wb");
    FILE* klr = ppl_kl.empty() ? nullptr : fopen(ppl_kl.c_str(), "rb");
    double kl_sum = 0, kl_max = 0;
    long same_top = 0;
    std::vector<float> lpv;
    std::vector<int> idx;
    auto t0 = std::chrono::steady_clock::now();
    for (int ch = 0; ch < nchunks; ++ch) {
      // like llama-perplexity: a model that expects BOS gets it as the first token of every chunk
      std::vector<int> chunk_toks(all.begin() + (size_t)ch * ppl_ctx, all.begin() + (size_t)(ch + 1) * ppl_ctx);
      if (add_bos && vocab.bos >= 0) chunk_toks[0] = vocab.bos;
      const int* tk = chunk_toks.data();
      eng.reset();
      // (fed in pieces of the chunk size: logits_at reads rows of the last piece only. --ppl-score-last N: only the
      //  last N positions are scored; --ppl-score-step S: those are fed S tokens at a time (S <= 8: the generation
      //  kernels), the rest in prompt chunks)
      const int piece = std::max(1, chunk);
      const int from = ppl_score_last > 0 ? std::max(1, ppl_ctx - 1 - ppl_score_last) : ppl_ctx / 2;
      for (int o = 0; o < ppl_ctx;) {
        const int n = o < from ? std::min(piece, from - o) : std::min(ppl_score_step > 0 ? ppl_score_step : piece, ppl_ctx - o);
        eng.forward(tk + o, n);
        for (int j = std::max(o, from); j < std::min(o + n, ppl_ctx - 1); ++j) {
          const auto& lg = eng.logits_at(j - o);
          const double mx = *std::max_element(lg.begin(), lg.end());
          double z = 0;
          for (float l : lg) z += std::exp((double)l - mx);
          nll += -((double)lg[tk[j + 1]] - mx - std::log(z));
          ++count;
          if (kls || klr) {
            const double lz = mx + std::log(z);
            if (kls) {
              idx.resize(lg.size());
              for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
              std::partial_sort(idx.begin(), idx.begin() + KT, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
              for (int k = 0; k < KT; ++k) {
                const float lp = (float)(lg[idx[k]] - lz);
                fwrite(&idx[k], 4, 1, kls);
                fwrite(&lp, 4, 1, kls);
              }
            } else {
              int id[KT];
              float bl[KT];
              for (int k = 0; k < KT; ++k) {
                if (fread(&id[k], 4, 1, klr) != 1 || fread(&bl[k], 4, 1, klr) != 1) { fprintf(stderr, "--ppl-kl: file too short\n"); return 1; }
              }
              double kl = 0, pb_top = 0, pq_top = 0;
              for (int k = 0; k < KT; ++k) {
                const double pb = std::exp((double)bl[k]), lq = lg[id[k]] - lz;
                kl += pb * (bl[k] - lq);
                pb_top += pb;
                pq_top += std::exp(lq);
              }
              const double tb = std::max(1e-12, 1 - pb_top), tq = std::max(1e-12, 1 - pq_top);
              kl += tb * std::log(tb / tq);
              kl_sum += kl;
              kl_max = std::max(kl_max, kl);
              same_top += (std::max_element(lg.begin(), lg.end()) - lg.begin()) == id[0];
            }
          }
        }
        o += n;
      }
      if (klr) printf("[%d] %.4f  KL %.5f  same top %.2f %%\n", ch + 1, std::exp(nll / count), kl_sum / count, 100.0 * same_top / count);
      else printf("[%d] %.4f\n", ch + 1, std::exp(nll / count));
      fflush(stdout);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("perplexity: %.4f over %d chunks of %d (%ld scored tokens, %zu tokens in file) in %.1f s\n", std::exp(nll / count),
           nchunks, ppl_ctx, count, all.size(), s);
    if (klr) printf("KL vs base: mean %.5f, max %.4f, same top token %.2f %%\n", kl_sum / count, kl_max, 100.0 * same_top / count);
    if (kls) fclose(kls);
    if (klr) fclose(klr);
    return 0;
  }
  if (bench_pp > 0) {
    std::vector<int> bt(bench_pp);
    for (int i = 0; i < bench_pp; ++i) bt[i] = 1000 + (i * 7919) % 50000;
    eng.forward(bt.data(), std::min(bench_pp, 64));  // warm-up (kernel JIT, page-in)
    // (ARC_BENCH_RUNS: prompt runs, default 3; long prompts: 1)
    const int runs = getenv("ARC_BENCH_RUNS") ? std::max(1, atoi(getenv("ARC_BENCH_RUNS"))) : 3;
    for (int r = 0; r < runs; ++r) {
      eng.reset();
      eng.prof_ms.clear();
      eng.prof_n.clear();
      eng.profile = prof && r == runs - 1;
      auto t0 = std::chrono::steady_clock::now();
      eng.forward(bt.data(), bench_pp);
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      printf("pp%d run %d: %.1f ms  ->  %.1f tokens/s\n", bench_pp, r, ms, bench_pp / ms * 1000.0);
    }
    eng.profile = prof_gen;  // (--prof-gen: the per-kernel profile below is of generation instead)
    if (prof_gen) { eng.prof_ms.clear(); eng.prof_n.clear(); prof = true; }
    int tok = 1000;
    const uint64_t b0 = eng.rd_bytes;
    std::vector<double> step_ms;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_gen; ++i) {
      const auto ts = std::chrono::steady_clock::now();
      const auto& lg = eng.forward(&tok, 1);
      tok = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
      step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("tg%d: %.1f ms  ->  %.2f tokens/s\n", n_gen, ms, n_gen / ms * 1000.0);
    {  // (a whole slow run or single slow steps)
      std::vector<double> st = step_ms;
      std::sort(st.begin(), st.end());
      printf("tg steps: min %.1f, median %.1f, max %.1f ms; first 8:", st.front(), st[st.size() / 2], st.back());
      for (int i = 0; i < 8 && i < (int)step_ms.size(); ++i) printf(" %.1f", step_ms[i]);
      printf("\n");
    }
    const double gb = (eng.rd_bytes - b0) / 1e9;
    printf("weights read: %.0f MB per token, %.1f GB/s (%.0f %% of ~75 GB/s)\n", gb * 1000 / n_gen, gb / ms * 1000,
           gb / ms * 1000 / 75 * 100);
  } else {
    if (toks.empty()) { fprintf(stderr, "no tokens\n"); return 1; }
    if (kl_decode) {
      // the single-token (decode) path at every prompt position against the reference log-probs of that position
      FILE* fi = fopen(kl_ref.c_str(), "rb");
      if (!fi) { fprintf(stderr, "cannot open %s\n", kl_ref.c_str()); return 1; }
      fseek(fi, 0, SEEK_END);
      std::vector<float> ref((size_t)ftell(fi) / 4);
      fseek(fi, 0, SEEK_SET);
      const size_t got = fread(ref.data(), 4, ref.size(), fi);
      fclose(fi);
      const size_t V = eng.hp.n_vocab;
      if (got != ref.size() || ref.size() != V * toks.size()) { fprintf(stderr, "reference must hold every position\n"); return 1; }
      double sum = 0, mxk = 0;
      int agree = 0;
      for (size_t r = 0; r < toks.size(); ++r) {
        const auto& l = eng.forward(&toks[r], 1);
        const double mx = *std::max_element(l.begin(), l.end());
        double z = 0;
        for (float v : l) z += std::exp((double)v - mx);
        const float* rr = ref.data() + r * V;
        double kl = 0;
        for (size_t i = 0; i < V; ++i)  // (a reference log-prob of -inf adds nothing)
          if (rr[i] > -1e30f) kl += std::exp((double)rr[i]) * (rr[i] - (l[i] - mx - std::log(z)));
        sum += kl;
        mxk = std::max(mxk, kl);
        agree += (std::max_element(l.begin(), l.end()) - l.begin()) == (std::max_element(rr, rr + V) - rr);
      }
      printf("KL decode path (ref || arcinfer) = %.5f (mean over %zu positions, max %.5f), top-1 agreement %d/%zu\n",
             sum / toks.size(), toks.size(), mxk, agree, toks.size());
      return 0;
    }
    // warm-up so kernel compilation is not counted in the time to first token
    { int w = toks[0]; eng.forward(&w, 1); eng.reset(); }
    // (ARC_PROMPT_WARM=N: N untimed runs of the whole prompt first, e.g. to fill the expert cache)
    for (int r = 0, nw = getenv("ARC_PROMPT_WARM") ? atoi(getenv("ARC_PROMPT_WARM")) : 0; r < nw; ++r) {
      auto tw = std::chrono::steady_clock::now();
      eng.forward(toks.data(), (int)toks.size());
      eng.reset();
      printf("warm-up prompt run %d: %.1f ms\n", r,
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count());
    }
    eng.profile = prof;
    auto t0 = std::chrono::steady_clock::now();
    // --feed-step N: the prompt goes through in pieces of N tokens (N <= 8: the single-token / small-batch kernels)
    if (feed_step > 0)
      for (size_t i = 0; i + feed_step < toks.size(); i += feed_step) eng.forward(toks.data() + i, feed_step);
    const size_t rest0 = feed_step > 0 ? (toks.size() - 1) / feed_step * feed_step : 0;
    // (ARC_DUMP_ROWS=file: the residual row of the last prompt token after each block, f32, for tools/ref_forward REF_DUMP)
    const char* dump_rows = getenv("ARC_DUMP_ROWS");
    eng.dbg_rows = dump_rows != nullptr;
    const auto& lg = eng.forward(toks.data() + rest0, (int)(toks.size() - rest0));
    if (dump_rows) {
      FILE* fd = fopen(dump_rows, "wb");
      for (const auto& r : eng.dbg_) fwrite(r.data(), 4, r.size(), fd);
      fclose(fd);
      eng.dbg_rows = false;
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("prompt: %zu tokens | time to first token %.1f ms (%.1f t/s)\n", toks.size(), ms, toks.size() / ms * 1000.0);
    std::vector<std::pair<int, float>> tk;
    top_k(lg, 10, tk);
    printf("top-10 next token:");
    for (auto& [id, p] : tk) printf(" %d(%.4f)", id, p);
    if (!kl_ref.empty()) {
      // KL(reference || arcinfer) over the full vocabulary; the reference file holds f32 log-probs of the last
      // position, or of every prompt position (then the KL is averaged over the prompt; single-chunk prompts).
      const size_t V = lg.size();
      auto kl_of = [&](const std::vector<float>& l, const float* ref) {
        const double mx = *std::max_element(l.begin(), l.end());
        double z = 0;
        for (float v : l) z += std::exp((double)v - mx);
        double kl = 0;
        for (size_t i = 0; i < V; ++i)  // (a reference log-prob of -inf adds nothing)
          if (ref[i] > -1e30f) kl += std::exp((double)ref[i]) * (ref[i] - (l[i] - mx - std::log(z)));
        return kl;
      };
      FILE* fi = fopen(kl_ref.c_str(), "rb");
      std::vector<float> ref;
      if (fi) {
        fseek(fi, 0, SEEK_END);
        ref.resize((size_t)ftell(fi) / 4);
        fseek(fi, 0, SEEK_SET);
        if (fread(ref.data(), 4, ref.size(), fi) != ref.size()) ref.clear();
        fclose(fi);
      }
      if (ref.size() == V) {
        printf("\nKL(ref || arcinfer) = %.5f", kl_of(lg, ref.data()));
      } else if (ref.size() == V * toks.size() && (int)toks.size() <= eng.max_chunk) {
        const std::vector<float> last(lg.begin(), lg.end());  // logits_at() overwrites the shared logits buffer
        double sum = 0, mx = 0;
        int agree = 0;  // (positions whose top token is the reference's)
        for (size_t r = 0; r < toks.size(); ++r) {
          const std::vector<float>& l = eng.logits_at((int)r);
          const float* rr = ref.data() + r * V;
          const double k = kl_of(l, rr);
          sum += k;
          mx = std::max(mx, k);
          agree += std::max_element(l.begin(), l.end()) - l.begin() == std::max_element(rr, rr + V) - rr;
        }
        printf("\nKL(ref || arcinfer) = %.5f (mean over %zu positions, max %.5f), top token agrees %d / %zu",
               sum / toks.size(), toks.size(), mx, agree, toks.size());
        std::copy(last.begin(), last.end(), const_cast<float*>(lg.data()));
      }
    }
    eng.profile = prof_gen;
    if (prof_gen) { eng.prof_ms.clear(); eng.prof_n.clear(); prof = true; }
    std::vector<int> gen;
    int tok = tk[0].first;
    const uint64_t rb0 = eng.rd_bytes;
    auto t1 = std::chrono::steady_clock::now();
    int steps = 0, accepted = 0, drafted = 0;
    if (eng.has_mtp) {
      // speculative decoding with the MTP block: up to two drafts per step, verified by one main-model pass
      const size_t V = eng.hp.n_vocab;
      auto argmax_row = [&](const std::vector<float>& l, int r) {
        return (int)(std::max_element(l.begin() + r * V, l.begin() + (r + 1) * V) - (l.begin() + r * V));
      };
      while ((int)gen.size() < n_gen) {
        gen.push_back(tok);
        if (tok == vocab.eos || (int)gen.size() >= n_gen) break;
        const std::vector<int> drafts = eng.mtp_draft(tok, eng.opt.mtp_drafts);
        const auto& lv = eng.verify(tok, drafts);
        int acc = 0, next = argmax_row(lv, 0);
        bool stop = false;
        while (acc < (int)drafts.size() && next == drafts[acc]) {
          gen.push_back(next);
          ++acc;
          if (next == vocab.eos || (int)gen.size() >= n_gen) { stop = true; break; }
          next = argmax_row(lv, acc);
        }
        eng.verify_done(acc);
        ++steps;
        drafted += (int)drafts.size();
        accepted += acc;
        if (stop) break;
        tok = next;
      }
    } else {
      FILE* dl = dump_logits.empty() ? nullptr : fopen(dump_logits.c_str(), "wb");
      struct Close { FILE*& f; ~Close() { if (f) fclose(f); } } close_dl{dl};
      if (dl) fwrite(lg.data(), sizeof(float), lg.size(), dl);  // the prompt's logits first
      for (int i = 0; i < n_gen; ++i) {
        gen.push_back(tok);
        if (tok == vocab.eos) break;
        const auto& l2 = eng.forward(&tok, 1);
        if (dl) fwrite(l2.data(), sizeof(float), l2.size(), dl);
        tok = (int)(std::max_element(l2.begin(), l2.end()) - l2.begin());
      }
    }
    double gms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    printf("\ngreedy:");
    for (int id : gen) printf(" %d", id);
    printf("\ngeneration: %zu tokens in %.1f ms (%.2f t/s)\n", gen.size(), gms, gen.size() / gms * 1000.0);
    if (!eng.expert_stats().empty()) printf("%s\n", eng.expert_stats().c_str());
    {
      const double gb = (eng.rd_bytes - rb0) / 1e9;
      printf("weights read: %.0f MB per pass, %.1f GB/s (%.0f %% of ~75 GB/s; drafter weights not counted)\n",
             gb * 1000 / std::max<size_t>(1, steps ? (size_t)steps : gen.size()), gb / gms * 1000, gb / gms * 1000 / 75 * 100);
    }
    if (steps) printf("MTP: %d drafts, %d accepted (%.1f %%), %.2f tokens per main-model pass (%d passes)\n", drafted,
                      accepted, 100.0 * accepted / std::max(drafted, 1), (double)gen.size() / steps, steps);
    if (show) printf("----- text -----\n%s\n----------------\n", vocab.decode(gen).c_str());
  }
  if (prof) {
    double total = 0;
    for (auto& [n, ms] : eng.prof_ms) total += ms;
    std::vector<std::pair<double, std::string>> rows;
    for (auto& [n, ms] : eng.prof_ms) rows.push_back({ms, n});
    std::sort(rows.rbegin(), rows.rend());
    printf("GPU time by kernel (last profiled forward): total %.1f ms\n", total);
    for (auto& [ms, n] : rows) printf("  %-20s %8.2f ms  %5.1f%%  (%d launches)\n", n.c_str(), ms, 100 * ms / total, eng.prof_n[n]);
  }
  return 0;
}
