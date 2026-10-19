# Optimization queue (started 2026-10-12)

Target: token generation using > 90 % of measured bandwidth (~75 GB/s, so > 67.5 GB/s average per token) for
every chat model in the Hugging Face cache, and vision / audio working on every model that has the encoder for it.
Status: ✅ done · 🔄 in progress · ⏳ queued · ⚠️ note · [~] partial

## Done (F1-F14)

Details: git history of QUEUE.md, CHANGELOG.md, docs/before-after.md.

Phase 1 (2026-10-12):
- #1 Q4_0 support (Google Gemma 4 QAT): done — E4B prefill 788 tok/s, generation 20.9 (drafter 26.8) tok/s
- #2 Bandwidth measurement tool: done — `run.exe` prints weights read and % of ~75 GB/s
- #3 Google mmproj detection: done — `gemma-4-E4B-it-mmproj.gguf` found; non-GSQ 35B marked not supported
- #4 Gemma 4 12B (Google) text / image / audio: done — generation 9.1 tok/s (85 % bandwidth), adjacent-string template crash fixed
- #5 Bandwidth efficiency stage 1: done — Q8_0 / IQ2 / IQ3 aligned blocks on the GPU (GEMV Q8_0 37→77 GB/s); rest in open items
- #6 E4B vision on the GPU: done — error vs f32 0.11–0.17 %, 1120 tokens 3.2 s
- #7 E4B audio on the GPU: done — transcription exact, 7.2 s of audio in 226 ms
- #8 Qwen3.5 9B vision: done — error 0.38 % vs f32
- #9 Qwen3.6 35B-A3B MoE kernels for standard GGUF types: done — generation 18.9 tok/s, prefill 465–480 tok/s
- #10 Regression tests (run_tests.ps1, Google Gemma references): done — all 12 checks pass

Phase 2 (2026-10-13 onward):
- F1 35B expert prefill with int8 GEMM: done — prefill 2048 tokens 476 → 642 tok/s (+35 %)
- F2 Long-context prefill attention: done (partial) — 4 q-heads per sub-group (attn_q8): 35B 9.4 → 8.4 s, 9B 8.2 → 7.0 s; rest in open items
- F3 35B generation expert kernels: done (partial) — kernels at practical limit; stacked DeltaNet GEMV: 35B 19.2 → 19.8 tok/s
- F4 MTP with 3 guesses (Qwen): done — 27B code 7.8 → 9.5 tok/s (+21 %)
- F6 35B vision: done 2026-10-14 — works via `35b-vision` (context 4,096 next to the encoder); rest in open items
- F8 MTP for 35B (`35b-mtp`, steps 8.1–8.7): done — bit-identical, generation 19.7 → 22.3 tok/s (+13 %)
- F9 Pi model list follows the vision setting: done via F10
- Vision on/off option (`pi-arc --vision off` / `arc-server -Vision off`): done 2026-10-13 — 9B 143K → 217K tokens
- F10 All models and variants in Pi /model automatically: done 2026-10-13 — 22 entries from 9 files, context per variant
- arcinferzero phase 1 (Level Zero probe, SSD bench, route simulation): done 2026-10-14 — LRU best, Windows minimum 14 GB, configurable expert cache
- arcinferzero phase 2 (Level Zero MoE engine, LRU cache, prefetch, SSD expert file): done 2026-10-14 — bit-identical, 35B 262K MTP 22.2 tok/s; second-image "device lost" fixed
- F12 server API stages 12.0–12.8, 12.10: done (code) 2026-10-15 — sampling options, logprobs / n, completions, response_format, Ollama, Anthropic, Responses, audio, launcher
- F13 13.1 validate all models on zero: done 2026-10-16 — all pass (see docs/arcinferzero-log.md); zero is the default in arc-server / pi-arc / launcher
- F14 7-draft MTP and DFlash2 for 27B: done — MTP 1–7 drafts identical (F16 E), DFlash2 drafter in use (F17, F19.7)
- arcinferzero package (`serverz --pack`): done 2026-10-15 — Gemma drafter GGUF re-copy identical (45 keys, 49 tensors)
- Not pursued (Phase 2 measurements): Level Zero gaps only (~1.6–2 %), chunked / f16 DeltaNet, split-K GEMV, per-type GEMV rules, gemv_glu for 27B

## Open items from F1-F14

- #5 (🔄) Bandwidth efficiency: remaining: non-weight kernels (DeltaNet recurrence, norm, attention), IQ1_M.
- Phase 1 (⚠️) Final speed measurements are taken after the 12B download finishes (CPU and GPU share power and heat).
- F2 (partial) Largest remaining cost for 35B on long prompts: per-chunk expert decode (~0.85 s × 11 chunks); needs one
  decode per layer for all chunks (next piece of work). See also F19 19.6 (⏸️).
- F5 (⏸️ not done) Q8_0 single copy in the GEMM layout (35B): after `decode_w8_sl` the Q8_0 decode per 2048 chunk is
  211 → 72 ms (~2 % of prefill), so a new GEMV for another layout is not worth it.
- F6 (next options) 35B vision context: Q8_0 encoder weights (saves ~0.45 GB), load the encoder only when there is an
  image, or raise the GPU budget.
- F7 (⏸️ not done, optional) 9B vision attention, head 72: split one query row across two lanes so the registers fit
  (the compiler still uses 128 registers for this kernel, outcome uncertain).
- F10 (optional note) If ~20 entries in /model feel too crowded, an option can be added to hide the "standard" variants.
- F11 (⏸️ postponed 2026-10-14) Int8 image-encoder weights, activations stay F32 (`vit_gemm` with Q8_0 weights unpacked
  to F16 in the kernel); 35B vision no longer depends on it. Note: a Q8_0 mmproj is currently also unpacked to F16 on the GPU.
- arcinferzero phase 3 (⏳) Qwen3.8 Flash Next (qwen4exp: QSA, n-gram table on SSD, 512 experts top-10). (F17 says Flash
  Next runs, commit 9788df0; this line was never closed.)
- F12 12.0–12.8, 12.10: marked "✅ code; overnight test" — the overnight test results are not recorded here.
- F12 12.9 (⏳) `/v1/embeddings` with EmbeddingGemma 2 (`unsloth/embeddinggemma-2-GGUF`, BF16 558 MB; optional BF16 mmproj
  982 MB for images/audio) + a document question-answering (RAG) feature option in the web UI; download with permission.
- F13 13.2 (🟡) Zero becomes the default: exes not renamed yet (`bin\server.exe` from the zero engine, `serverz.exe`
  removed, `runz.exe` → `run.exe`, OpenCL temporarily as `server-opencl.exe`).
- F13 13.3 (⏳) Daily use for 1–2 weeks.
- F13 13.4 (⏳) Remove the OpenCL path (src/engine.h OpenCL, old server.exe / run.exe).
- F14 Parallel drafting: the drafter (MTP / Gemma drafter) runs on a separate Level Zero queue, overlapping with
  main-model verification; first measure how much can overlap on the iGPU (one compute engine). (F16 E: parallel / tree
  drafts not yet.)
- arcinferzero package (⏳ overnight) 35B package: create, run without the original GGUF, results bit-identical to regular zero.
- arcinferzero package: Flash Next (DO NOT delete its GGUF: the user decides when, after backup): back up to flash drive
  (1 TB, exFAT/NTFS) → delete from the SSD → `--pack` from the flash drive; the `qwen4exp` architecture is handled separately.

## F15: two conversations at once and Gemma tool-call fixes (requested 2026-10-15)

Target (2026-10-15): E4B "at least good at Pi" (all Pi tools: read, bash, edit, write, web, browser).
- [x] Schema-constrained tool calls for Gemma 4 (`tools/tool_constraint.h`, 12 unit tests). In server-new.exe.
- [x] Qwen: a 3rd snapshot at the start of the assistant turn after the last user message (server-new.exe). The Qwen template drops
      the reasoning of old turns as soon as the user writes again, so every new message in Pi reprocessed everything from the end of the
      system prompt (this afternoon: 8K -> 4K cached, 10 s; at 57K: ~3 minutes). With this slot: only the last turn.
- [x] GPU test (night of 15 Oct, A1): Pi + Qwen 9B, two consecutive messages with a tool call in between: output
      identical to processing from scratch; new message 5.2 s vs 16.8 s without the snapshot.
- [x] GPU test (night of 15 Oct, A3): create_word.py edit test (copy in a mock folder) with E4B,
      ARC_TOOL_CONSTRAINT=0 vs 1: valid tool calls 2/10 → 9/10.
- [ ] Qwen XML format (`<tool_call><function=...><parameter=...>`) with the same constraint.
- [ ] Automated Pi tests: fixed tasks (read, run, small edit, large edit, write, web_search/fetch, browser) in an
      empty folder, score per model (E4B, 12B, 4B, 9B, 35B), before and after the constraint.
- [ ] If E4B is still not good enough: LoRA fine-tune on successful Pi conversations from large models (cloud GPU).

- [x] Web UI compaction: automatic summary when the context is nearly full / the server refuses, Compact button (CHANGELOG).
- [x] Raw tool-call text log: `ARC_LOG_TOOLCALLS=file.jsonl` (USAGE.md).
- [ ] Gemma tool calls: flatten nested `edits`; fill a missing `path` from the only file read
      (`read`) in that conversation (if more than one file: no guessing); `&amp;` in bash commands.
- [ ] Slots stage 1: several conversation caches, not split evenly: one pool, active conversations use what they
      need; when full, the least recently used slot is saved to the SSD (not discarded) and reloaded when it
      comes back. On Level Zero (F13).
- [ ] Slots stage 2 = batch decode (continuous batching, like llama.cpp `-np` / vLLM; user 16 Oct: batching is more general
      than just slots): several requests in one GPU step, weights read once (dense: total nearly 2x,
      MoE: less because experts differ); MTP/drafter per request; prefill interleaved. Stage 1 (cache) is its basis.
- [ ] Check: web UI in Pi's headless browser drops to 4.9 tok/s (suspected: GPU taken by browser rendering).

## Flash Next (qwen4exp): download finished 2026-10-15 13:02

File (HF cache, DO NOT delete): `~/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f920.../IQ3_S/`
part 1 54.8 GB (all weights), part 2 28.8 GB (only `per_layer_token_embd` [160, 320,001,536] IQ4_NL, n-gram table).
Header read (v3, 1224 tensors). Architecture to build:
- 48 layers, hidden 2560, 3 DeltaNet : 1 full attention (same pattern as Qwen3.6), MoE 512 experts, 10 active, ffn 640, + shared expert.
- Hyper-connections: 4 residual streams (10240 = 4 x 2560), low rank 320 (`hc_attn_*`, `hc_ffn_*`, `output_hc_*`).
- Full attention: sparse indexer (4 heads, key 128, top_k 2048) + `compress_ratios` 4 in the attention layers.
- N-gram PLE: hash table of 320 million rows x 160 (part 2), 1 row per token -> read directly from the SSD (no need for it on the GPU).
- New type: 42 = Q2_0 (block 64) in 9 `ffn_down_exps`; others IQ3_S/IQ3_XXS/IQ2_S/IQ4_NL/IQ4_XS/Q4_K..Q6_K/BF16.
- No MTP in this GGUF yet. BF16 mmproj 0.91 GB available (not downloaded, needs permission).
- Reference: upstream llama.cpp supports it (`-lm mmap --lazy-mode on`); GSQ/RCO on IST-DASLab's GitHub.
Disk: C: ~67 GB left. The expert package (~50 GB) fits but tightly -> prefer reading experts directly from the GGUF.

## F16: optimize the Level Zero kernels as far as possible (requested 2026-10-15, started that night)

"No shortcuts." Only the zero path (kernels/*.cl through zero/clrt.h). Baseline first (prefill/decode tok/s per model).
- [~] D1 Register spill (partial, see status below): `ARC_ZE_KPROPS=1` prints private/spill per kernel; all models + mmproj encoders; target 0 spill.
- [~] D2 Intel compile flags (256 GRF tested per kernel, see status; subgroup 32 not yet): large GRF (256 per thread / auto), subgroup 16 vs 32, option for buffers > 4 GB; test per kernel.
- [ ] D3 Grouping/tiling: tile & work-group shape for prefill GEMM, rows per subgroup for decode GEMV, block reads, SLM;
      autotune per matrix shape per model, results saved.
- [ ] D4 Prefill GEMM without XMX: this GPU (Core Ultra 5 125H, Arc Xe-LPG 112 EU) does NOT have XMX
      (matrix_multiply_accumulate and 2d_block_io are absent); maximize DP4a (already used) + register blocking + fp16.
- [~] D5 (35B decode dsplit and Qwen prefill done; Gemma sliding window / short context not yet) Attention for short context (~1-4K), medium (~16-32K), long (~128K-262K): split-K decode for long
      context, flash-attention-style prefill (SLM, fp16/DP4a; no XMX), Gemma sliding window, multi-row MTP verify;
      variant picked automatically per context length; error measured on long text too.
- [~] D6 mmproj (Qwen/Gemma images, Gemma audio) on Level Zero: works and identical (15 Oct); its kernels not optimized yet.
- [~] E (MTP 1–7 drafts identical done; parallel / tree drafts not yet) Parallel MTP drafting on Level Zero (part of F14): several drafts per step, one-pass verification, output
      identical; best number of drafts per model.
Correctness ("make sure the error rate is low"): a new accumulation order = not bit-identical with OpenCL. Reference = f32 logits
on fixed text (~2K tokens) per model; the current kernels' baseline error is measured first. Requirements for every change: mean KL
<= baseline and < 1e-3, top-1 >= baseline - 0.1 points, perplexity up < 0.1 %, greedy the same except near-ties, tool-call
tests pass, image/audio answers the same. Not met = not used (switch to the old kernel).

### Status, night of 15 October (updated continuously)

- [x] A1–A5 pass (Qwen turn snapshot, preserve-thinking, Gemma + Qwen tool-call constraint, session cache,
      web UI compacting). See docs/arcinferzero-log.md.
- [x] Zero baseline for all models (identical to OpenCL; 35B identical with the same chunk).
- [x] E: MTP 1–7 drafts correct (identical), replay rollback (1 state copy). 4–7 drafts not yet faster than 3:
      each draft = one pass through the MTP block + the full output head.
- [x] D1 GEMV TB=8 (4–7 drafts): spill 9 KB → ≤ 2.9 KB, 3.8 → 9.2 tok/s (27B, 5 drafts).
- [x] D1 decode attention: spill gone with 256 GRF; 35B uses attn_dsplit (1.16 → 0.94 ms/layer at 64K),
      verification still bit-identical. Correction: Level Zero DOES USE the 256 GRF flag; the earlier conclusion was wrong because
      my own env switch was empty (see log, CORRECTION section).
- [x] Default zero MTP drafts: 35B 3, 4B 3, 9B 2, 27B 3, Gemma 2. Fixes: JSON comma after the last key,
      `/api/tags` with `--no-swap`, Gemma attn_prep_g 256 GRF (E4B +1–2 %).
- [x] Gemma spill: attn_prep_g HD512 4.6 KB → 0, add_norm_quant 3.7 KB → 64 B (identical, E4B/12B slightly faster).
- [ ] Remaining spill: gdn_conv 1 KB (256 GRF: 0 spill, not faster), moe_gateup_q 2 KB (256 GRF not faster).
- [x] D5 Qwen prefill attention: attn_q8 256 GRF (already used by the engine) beats all attn_ps variants
      (64K: 4B/9B 926 vs 1,117 ms, 35B 923 vs 1,138, 27B 1,516 vs 1,669). No change.
- [ ] Cheaper drafts: a compact draft head (`ARC_DRAFT_VOCAB`) exists but is off (a loss for Indonesian);
      next ideas: a dynamic token list (tokens in the context + frequent ones), or parallel (tree) drafts per pass.
- [ ] The server is sometimes ±½ slower for its whole lifetime (35B: 26 → 12 tok/s, identical logs), especially when started
      right after another server exits; a 5 s pause helps. runz --bench-pp now prints per-step times (16 Oct:
      7 E4B 32K runs stable, the slowdown did not appear; suspected: one big pause, not a GPU that stays slow).
      17 Oct (F17): showed up again in runz 35B (about 1 in 5 runs, 14-16 vs 21-23 tok/s, identical work, expert
      hit 100 %). Profile: all dense weight kernels (lm_head, qkv, z, ssm_out, q, o, drafter) ~2x slower,
      expert kernels (1 GB pool) normal; slow runs also have the longest load. Suspected: physical page placement of the
      weight buffers. Tried, not used: 512 MB arena (35B +2 % and more stable, but 27B prompt -2..5 %),
      2 MB alignment, trimming the working set during load (no visible effect; the symptom does not appear every run).

## F17: weekend work, Friday 16 Oct night → Monday 19 Oct 08:00 (laptop left unattended)

Focus: 27B speed on arcinferzero. Standing rules: results bit-identical to before (or, for the drafter,
answers identical to running without the drafter), tests in a scratch folder, test server on port 8091 and always shut down, Windows
RAM >= 14 GB, GGUFs not deleted, Gemma drafter path unchanged, no downloads other than DFlash2
(already allowed), push to GitHub only when allowed. Every change is recorded in docs/arcinferzero-log.md.

| # | Work | Estimate | Target | Pass criteria |
|---|---|---|---|---|
| 17.1 | Faster IQ3_S / IQ3_XXS (and IQ2_*) GEMV: zero-specific GEMV bench per quantization type, codebook variants (SLM / registers / constants), how indices + signs are unpacked, rows per sub-group; summation order unchanged | 2–4 hours | 27B FFN 56–60 → ≥ 68 GB/s; 27B +5–10 % | identical logits (run vs runz) 4B, 9B, 27B, 35B; speed alternating 3x |
| 17.2 | Fuse small kernels: FFN gate + up in one kernel; small DeltaNet kernels (gates, conv, state update, norm) | 2–3 hours | gaps between kernels 4 % → ≤ 2 %; 27B +2–4 % | same as 17.1 |
| 17.3 | DFlash2 for 27B: (a) convert BF16 safetensors → GGUF (weights to Q8_0, selector + convolution as is); (b) CPU numpy reference from the z-lab/dflash code (model.py), checked with hidden states dumped from runz; (c) drafter on the GPU: fc + hidden_norm, 5 layers (two-tap convolution, non-causal sliding-2048 attention over context features, MLP), 27B output head for 7 positions, top-16 + selector; (d) save hidden states of layers 5/19/33/47/61 during verification; (e) block-8 verification using the existing 7-draft MTP path (DeltaNet replay rollback); (f) runz + server (`--drafter dflash`) | 15–25 hours | 27B ±9 → 12–15 tok/s | answers identical to running without the drafter (code + Indonesian, temperature 0); draft tokens = numpy reference; no crash over 20 consecutive requests |
| 17.4 | (if 17.3 is done) Tune DFlash2: effective block length, drafter quantization (Q8_0 vs Q4), sampling at temperature > 0 (rejection sampling as in the reference), long context | 3–6 hours | best speed per kind of answer | same as 17.3 |
| 17.5 | (if time remains) Cheaper MTP drafts for other models (dynamic token list); or start paged KV for slots | remaining time | — | identical |
| 17.6 | Full regression for all models (identity, images/audio, MTP, tool calls, API, Pi smoke), docs, report | 3–4 hours | all pass | — |

Rough schedule (adjusted to results; work that fails its criteria is dropped, not forced):
- Friday night: 17.1, then 17.2.
- Saturday: 17.3 a–c (conversion, numpy reference, drafter on the GPU).
- Sunday: 17.3 d–f (integration, server), 17.4.
- Sunday night – Monday 04:00: 17.5 if there is time; Monday 04:00–07:30: 17.6.
- Monday 07:30 stop testing; 07:40 tidy up: all test processes stopped, server LEFT OFF (user's decision),
  final report.

**Status (16 Oct night):**
- [x] Part A: GEMM with 16 tokens side by side (+3-7 %), attention P·V (-12..17 %); decode prefetch, split, GEMV sign
  trick tried, not used (docs/lessons §5).
- [x] Flash Next runs (engine, PLE, QSA, server): short-chat KL 0.0014; generation ±4 tok/s; commit 9788df0.
- [ ] Flash Next: validate QSA at 3,566 tokens vs the CPU reference; speed (cache 12 %, one pool); no MTP.
- [ ] DFlash2 for 27B: drafter code written (zero/engine.h dflash_*, --dflash DIR), not tested yet.

**New order (user decision 16 Oct midday): Flash Next first.**
1. Flash Next (qwen4exp) on arcinferzero, reusing the 35B path (SSD expert streaming + GPU cache, MoE routing,
   DeltaNet, tokenizer/template/server). New: Q2_0 type (id 42), hyper-connections (4 residual streams, low rank 320),
   sparse attention (top-2048 indexer, x4 compression), MoE 512 experts / 10 active + shared expert, n-gram PLE table
   (part 2, 26.8 GiB IQ4_NL) read from the SSD per token, mmproj if Qwen-VL family. Reference: upstream llama.cpp
   (qwen4exp) + our own f32 CPU reference. Experts read directly from the GGUF (C: 81 GB left). Estimate 20–30 hours.
   Pass criteria: CPU vs GPU reference (small KL, same greedy), sensible answers (code + Indonesian), server + Pi work,
   no crash over 20 requests; measure prompt and generation (estimate 5–8 tok/s, prompt 150–250 tok/s).
2. DFlash2 for 27B (17.3/17.4) right after Flash Next (user decision: Flash Next + DFlash2 first).
3. 17.1 + 17.2 (±6 hours) if time remains.
4. 17.6 regression + report (Monday 04:00–07:30).

Minimize error (user request): every stage is checked against the CPU reference before moving on; after changing shared
code (zero/engine.h, kernels, server) the existing models are retested (4B, 27B, 35B, E4B: logits bit-identical vs before,
images/audio, MTP); whatever fails is dropped. Pushing to GitHub is allowed (user, 16 Oct midday) for every stage that passes,
after checking for private paths / secrets / large files / CRLF.

Small models (4B, 9B, 12B, E4B) are only on D:\models (flash drive, sometimes disconnects after sleep; the model list skips it if
it does not respond within 2 s). Small-model regression is read from D:; if D: is disconnected, it is skipped and noted in the report (user: not
to be copied to C:). C: space 82 GB: enough even if Flash Next turns out to need a separate expert file (±47 GB).

Flash Next size (part 1 header): 1,223 tensors, 51.0 GiB: experts 46.8 GiB (512 x 48 layers, ±2 MiB per expert),
non-expert ±4.2 GiB (attn_qkv 0.66, output 0.49, ssm_out 0.41, attn_gate 0.35, token_embd 0.31, hc_* 4 x 0.29,
attn_q 0.24, shared expert 0.14, router 0.12, ...). Part 2: per_layer_token_embd IQ4_NL [160, 320,001,536] 26.8 GiB.

User decisions (16 Oct morning): start tonight when the user says so (the server is in use by the user until late afternoon); push to GitHub
for every stage that passes (changed 16 Oct midday; originally local commits only); Monday morning the server is left off; the laptop is kept awake by an
app during the work.

## F18: DeepSeek review follow-up + 35B long context (Saturday 17 Oct night, user asleep)

Source: `deepseek-v4.1-flash-evaluation-of-arcinfer.md` (sections 3-17), every finding checked against the code
first. Standing rules: logits / greedy text identical (or KL measured and reported), A/B alternated, one GPU job at a
time, no model GGUF deleted, no large new disk use without asking (Flash Next expert file = 53 GB), server off at the end.

| # | Task | Status | Notes |
|---|---|---|---|
| 18.1 | P0-1..4: HTTP parser, media lock, send timeout, empty prompt | ✅ | commit 2112264, tests/test_http.cpp |
| 18.2 | P1-4..6: sampler (presence penalty in greedy, logit_bias -100 bans, sample_allowed without a vocab-sized index), API metadata; P1-1 run_tests.ps1 for zero | ✅ | commit c697b40, tests/test_sampler.cpp |
| 18.3 | Level Zero launch safety (review 13/16): suggested group size divides the global size; explicit mismatches fail loudly; guards in conv_state_update / moe_combine; gemv odd-nsb over-read (tail slack 256 -> 1024 B); GEMM shape invariants; dfl_attn bound; ARC_KV4 validated | ✅ | commit 64a2441; 27B / 35B identical (plain, MTP, DFlash2), run_tests.ps1 20 passed |
| 18.4 | 35B UD-IQ4_XS: expert file, context sweep 32K / 64K / 96K / 131K / 262K (cache %, hit rate, SSD, tok/s, hybrid), prompts 250K / 125K, MTP-only and plain at 131K / 262K; same for Flash Next | ✅ | docs/speed-progress.md 2026-10-18 |
| 18.5 | Optimal context: 131,072 for 35B and Flash Next; auto context reserves 128K before the expert cache | ✅ | commit 64a2441; DFlash2 at 262K: no gain for code, -18 % Indonesian (auto stays at 131K, so not switched off) |
| 18.6 | 35B identity: plain vs MTP vs DFlash2 greedy; run_tests.ps1 (zero) with 27B + 35B | ✅ | ALL TESTS PASSED (20 passed, 7 skipped) |
| 18.7 | KV 4-bit accuracy (ARC_KV4 simulation, 8K / 32K) on 27B, 35B, Flash Next, with the f16-vs-int8 noise floor | ✅ | table in speed-progress; a real V-only 4-bit cache is the next long-context lever (35B at 250K: ~45 of 92 ms per step are K/V reads); not started (1-2 days: every attention kernel) |
| 18.8 | Decode attention at deep context (27B / 35B, 64K+): split tuning (ARC_ATTN_DEC_SG / SMAX) | ⚠️ | not needed: at 250K the extra ~45 ms per step equals reading 2.6 GB of int8 K/V at ~60 GB/s; only fewer bytes (4-bit KV) help |
| 18.9 | MOE_TILES balance (review N-3) | ⚠️ | not needed: a 4096-token chunk has ~512 equal 16-row tiles per 64-expert group and ny = that count + 64, so each work-group takes about one tile |
| 18.10 | RoPE pow table (P1-7) | ⚠️ | skipped: attn_prep is 0.3 % of a Flash Next prompt, ~0.5 % of generation |
| 18.13 | 4-bit V (review 18 / 18.7 option D, user: "PP terbaik, TG baik") | ✅ | opt-in `--v4`: int8 V for prompts + a 4-bit copy (per 64 dims, (n - 8) * amax / 7) for generation; 35B 125K prompt 321 -> 316 tok/s, generation at depth 72.8 -> 67.5 ms per step, expert cache 90 -> 87 %. session save -> restart -> load with --v4 tested 18 Oct (identical follow-up; an int8 server refuses the V4 files) |
| 18.11 | Flash Next QSA row tiling (N-5) | ✅ | commit ec25201: qsa_score_rt (8 rows), bit-identical; pp32768 +5..17 %, 125K prompt 78.6 -> 123.4 tok/s; --attn-f32 refused for QSA (it had no masks) |
| 18.12 | Docs: speed-progress (35B IQ4_XS, sweep), lessons (mach scan, KV 4-bit results), README 35B line | ✅ | |
| 18.14 | DeepSeek review re-check (agent, every item vs HEAD e85bd08) | 🔄 | done since: chunk-size line cap + chunk trailers, --host warning, --v4 / --mtp-drafts help, "one answer at a time" note. Left, by importance: kernel lane guards (the shim already fails loudly) + a non-multiple-dims test; writer thread so a slow client does not hold the engine lock; --v4 Flash Next beyond 8K (35B done: 32K KL 0.0124; 262K generation -8 % per step at 76 % cache); Ctrl+C session save / RAII writers; qsa_select histogram; GEMM / GEMV / moe_group autotune; merge kernels/ and zero/kernels/; split server.cpp / ops.cl; /props and /v1/models schema test; check-docs script; small test model |

## F19: prompt speed after V4 (review section 19, requested 2026-10-18)

| # | Task | Status | Notes |
|---|---|---|---|
| 19.1 | Flash Next prompt: where the GPU waits | ✅ | 11K prompt: wall 99 s, GPU kernels 58 s, expert reads (`ensure`, GPU idle) 29 s, PLE host rows 3 s (not the cause, review 19.2 C); 117 GB read from the SSD (2 layer-major segments of 8192, ~89 % of a layer's experts missing per segment) at 3.9 GB/s, hardly overlapped with the GPU. Host times now in the expert statistics line |
| 19.2 | Larger prefetch staging (ARC_XSTAGE) | ⚠️ | 256: prompt +5..11 % (A/B), generation -2..3 % (cache 22 -> 20 %); 512: +24 % once, generation -1..30 %; not made the default (staging is host RAM, the expert cache shrinks) |
| 19.3 | Overlap expert reads with the GPU without a larger staging (copy engine / ring) | ✅ | deep prefetch (`zero/xcache.h`, `ARC_XCOPY=0` turns it off): while the GPU runs layer L, a thread reads every missing expert of L+1 and the copy engine (second command list) copies it into its slot; the main list waits on an event. Logits bit for bit the same as before. Flash Next: 4096 tokens 107 -> 166 tok/s, 9000 tokens 116.9 -> 144.5, 32K (ctx 131K) 131.7 -> 162.6 tok/s (GPU idle on reads 62.4 -> 5.8 s); 35B 32K at ctx 131K (cache 90 %) 488.6 -> 557 tok/s; 35B at ctx 16K (cache 97 %) the same. Bugs found on the way: (1) a small pool (868 slots, 3 layers in a row) left the current layer less than a layer's worth of slots, and its next chunk's victim walk reached its own hits (rows 2048+ of layer 28 wrong, first visible at layer 29): victims now skip the call's hits, and the prefetch leaves the current layer NE slots of a shared pool; (2) prefetched experts went to the LRU front: with a cyclic layer walk that evicts the next layers' experts (35B, 97 % cached: 7637 reads instead of 846), now they go to the back like the prompt's misses; (3) segments under 2048 tokens leave ~25 % of the prefetched experts unused (808 tokens), so the deep prefetch needs >= 2048 tokens (`ARC_XDEEP_MIN`); (4) reads and copies pipelined in two halves of the staging (SSD 3.1 -> 3.9 GB/s). Not the causes: stale caches, record over-reads (zeroed tails, padded slots), event reuse, routing order |
| 19.4 | 35B chunk 8192 | ✅ | the default for streamed MoE (Flash Next keeps 2048). 35B at ctx 262K, chunk 2048 / 4096 / 8192 (deep prefetch on, cool-downs between runs): expert cache 83 / 81 / 77 %, 32K prompt 485 / 533 / 604 tok/s, 128K prompt 284 / 306 / 324, generation at 128K 13.0 / 12.8 / 12.6 tok/s (an earlier 6.95 was a hot laptop). Auto context stays 131K (it reserves the context first); ctx-auto key m13 |
| 19.5 | GEMM / moe_group autotune, 27B --budget | ✅ | the defaults are already the best: 27B pp4096 (RG 2/8, WGY 4/16, TM 32, split-K 9/34/off) and 35B pp8192 (MOE_G 32/128, RG, WGY) all the same or slower (27B 123-133 tok/s, 35B ~760). `ARC_GEMM_SPLIT` (superblocks per split-K piece) added for such sweeps |
| 19.6 | 35B: expert decode once per layer | ⏸️ | large (the down outputs of a segment's chunks, ~1 GB, would cost expert cache); chunk 8192 already takes much of it |
| 19.7 | Drafter choice per model (Pi's /model) | ✅ | a model with a DFlash2 drafter is also listed as `<id>-nodf` (MTP block only) and `<id>-df` (DFlash2 only, the MTP block not loaded: 27B auto context 69,632 -> 94,208); `--draft auto|mtp|dflash` (server, runz); greedy text identical in all modes. 35B, 200 tokens: Python code plain 20.2 / auto 45.1 / MTP 39.1 / DFlash2 49.9 tok/s, Indonesian auto 21.0 / MTP 21.9 / DFlash2 16.5 |
| 19.8 | Review sections 20-21 (deep prefetch lifetime, guards, build flags) | ✅ | a second engine in the process (the server's model switch) keeps the Level Zero context, lists, events and staging (before: a new context with the old one's copy list / events / staging buffer / timestamp events); the prefetch thread catches read errors (signals the event, drops its experts, join() rethrows); WaitForMultipleObjects / CreateEventW checked; slots without a key guarded; divisor knobs >= 1; /fp:precise for ref_forward, bitcheck, tests. Checked: 35B -> 35B-df -> Flash Next -> 35B in one server, answers right. Not changed (the review was wrong): event scope 0x4 is HOST, queue flag 2 is IN_ORDER, the thread syncs its copy list before it ends. Regression (2026-10-18, every run's logits bit for bit against the old path): 35B with a 41 % cache, Flash Next 13 %, ARC_XSTAGE=32, ARC_XDEEP_MIN=512, 40 Flash Next runs in a row, and ARC_FAKE_OOM with 2-6 cache shrinks per run on both paths (Flash Next 9000 / 4096, 35B): 56 of 56 the same, no hang; shrink() no longer lets a prefetch error out of a kernel launch, ensure() fails clearly with no free slot. Not tested: devices without a copy engine |


## F20: next work, waiting for the user's "mulai" (queued 2026-10-19)

Nothing here starts before the user says "mulai". Order as listed; each item gets its before / after row in
docs/before-after.md and a local commit (push only when asked).

| # | Task | Status | Notes |
|---|---|---|---|
| 20.1 | Flash Next accuracy: where its KL 0.020 comes from | ✅ first pass (2026-10-19) | the f32 reference (`tools/ref_forward`, now reading the package's expert file) with one group's inputs int8-rounded per 32 values, against the plain reference, 363 tokens of README text: HC mixers (`hc_`) KL 0.0605 (top token 89.0 %), attention + DeltaNet inputs (`attn_`) 0.0518 (89.0 %), routed experts (`exps`) 0.0250 (93.9 %), shared expert (`shexp`) 0.0210 (93.4 %); together about the 0.15 the engine had on a 337-token text. The engine's HC mixers are f32 for single tokens but int8 GEMMs for prompt chunks (`hc_mix`), so the prompt path carries that part. Not run yet: `indexer` (QSA), `ssm_out`, router, output head; the engine's own KL on this text (the test lost the GPU device: the user's 35B server was running at the same time and held the shared GPU memory; check every serverz / runz process before GPU tests) |
| 20.2 | Flash Next accuracy: higher precision only where it pays | ⏳ next | first candidate: prompt chunks' HC mixers in f32 as single tokens already are (small matrices); then finer scales for the `attn_` inputs; each measured as KL / top-token agreement against the reference and prompt speed, with no other GPU process running (the user's server stopped first, with permission) |
| 20.3 | Slots stage A: two conversations at once | ⏳ waiting | `ConvState` per slot (KV, DeltaNet states, snapshots, MTP / DFlash2 state, PLE / QSA tails), `Engine::bind()` swaps handles; server: per-slot cached_ / cuts / snapshots, a per-step lock instead of the request-long one, `--slots N` (default 1), `-Slots` in arc-server / pi-arc; tests: each slot bit-identical to running alone, two Pi sessions at once (empty folder, `--no-tools`); 3-4 days |
| 20.4 | Batch decode B1: dense Qwen (4B, 9B, 27B) | ⏳ waiting | several slots in one GPU step (the multi-row verification pass is the base), attention per row over its own slot's KV, DeltaNet state per row, scheduler with prefill interleaved; target 4B: 2 slots 1.6-1.9x, 4 slots 2.5-3.5x total; 4-6 days |
| 20.5 | Batch decode B2: Gemma 4 E4B | ⏳ waiting | sliding-window rings and PLE per slot; the Gemma drafter path unchanged; 2-3 days |
| 20.6 | Batch decode B3: MoE (35B, Flash Next) | ⏳ optional | the union of the slots' experts read once; Flash Next gains least (SSD-bound); 3-5 days |
| 20.7 | Flash Next drafting | ⏳ waiting | check Hugging Face for an MTP block in the original Qwen3.8 Flash Next checkpoint or a DFlash2 drafter for it; only a check, no download without asking |
| 20.8 | Copy 4B / E4B from the D: backups to the C: Hugging Face cache | ⏳ waiting | needed for 20.4 / 20.5 (D:\models is no longer scanned); the user decides when |
| 20.9 | Gemma 4 26B-A4B (MoE) support | ⏳ waiting | google/gemma-4-26B-A4B-it-qat-q4_0-gguf (14.4 GB + 1.2 GB mmproj, downloaded 2026-10-19); the Gemma 4 loader knows only the dense models (12B, E4B): add its routed experts (router, expert GEMMs / GEMVs, SSD streaming and packages through the Qwen MoE paths where the layout allows), sliding-window + global attention as now; check the tensor names and expert types first, accuracy against `tools/ref_forward` (Gemma 4 MoE there too), the Gemma drafter path unchanged |
