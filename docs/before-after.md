# Before and after

Every change that moves speed, memory or accuracy gets a row here with the numbers before and after, measured the
same way, and the check that the answers did not change. Newest first. The long story is in
[review.md](review.md) and [speed-progress.md](speed-progress.md); what each finding taught is in
[lessons-intel-arc-igpu.md](lessons-intel-arc-igpu.md).

**Test machine:** Intel Core Ultra 5 125H (Arc iGPU, 7 Xe-cores / 112 EUs, no XMX), 32 GB LPDDR5x-6400, Kingston
NVMe SSD (3.5-4.3 GB/s for 1-4 MB reads), Windows 11, nothing else on the GPU. Pauses between runs (the laptop
slows down by 5-10 % when hot); one run per cell unless noted. tok/s = tokens per second.

**How "answers did not change" is checked:** the logits of every prompt position are dumped (`runz --dump-logits`)
and compared byte for byte with the run before the change, or the generated greedy text is compared character for
character. "Bit-identical" below means exactly that, not "close". Changes that are allowed to change the numbers
(new quantization, 4-bit V) are measured as KL divergence against a reference instead.

## Where it stands (2026-10-20)

| Model | Prompt | Generation | Accuracy |
|---|---|---|---|
| Qwen3.6 35B-A3B UD-IQ4_XS (MoE, experts on the SSD) | 604 tok/s (32K prompt), 324 tok/s (128K) | 20.6 tok/s; Python code 49.9 with DFlash2, Indonesian 21.9 with MTP; 12.6 at 128K depth | KL 0.004-0.008 against an f32 reference (generation path) |
| Qwen3.8 27B (IQ3_S GSQ, dense) | ~135 tok/s (4K prompt) | 4.9 tok/s; code 18.1 with DFlash2 + MTP | KL 0.0003-0.0004 |
| Qwen3.8 Flash Next (IQ3_S, 512 experts on the SSD) | ~191 tok/s (4K, measured 2026-10-23), 163 (32K), 123 (125K) | 6-7 tok/s | KL 0.020 against the CPU reference with QSA (3.5K prompt); 0.048 on 363 tokens of README text (0.065 before) |
| Qwen3.5 4B | 911 tok/s (2K prompt) | 17.6 tok/s | KL 0.0021; perplexity 6.056 (llama.cpp 6.141) |
| Gemma 4 12B (Google QAT) | 102 tok/s (14K prompt) | 19.0 tok/s with its drafter | KL 0.0002 (two-pass mode, the only one since 2026-10-22) |
| Gemma 4 26B-A4B (Google QAT Q4_0, MoE, experts on the SSD) | 254 tok/s (4K prompt), 189 (14K) | ~25 tok/s with its drafter; 13.0 after a 14K prompt without | KL 0.0042 against the CPU reference (480 tokens; two-pass mode, the only one since 2026-10-22) |

Two conversations at once (`--slots 2`), both writing their answers, their next tokens in one engine pass (tok/s of
the two together): 27B prose 7.3 -> 10.5, code 11.6 -> 15.4; 35B 20.2 -> 24.7, 29.0 -> 30.7; Gemma 4 26B 19.8 -> 26.2
(taking turns -> batched; logits bit-identical to each slot alone).

Against llama.cpp on the same files and laptop (10 October, the project's first, OpenCL engine; the Level Zero engine gives the same
logits as that engine and the newer numbers in the table above):

| Model | Prompt, 2K tokens: arcinferzero / llama.cpp Vulkan / SYCL | 34K-token code prompt, time to first token | Generation after the 34K prompt |
|---|---|---|---|
| Qwen3.5 4B | 911 / 242 / 337 tok/s (**2.7-3.8x**) | 57.5 s / 376 s / 163 s | 15.7 / 10.6 / 8.2 tok/s |
| Qwen3.5 9B | 402 / 151 / 189 | 96 s / 461 s / 243 s | 8.6 / 7.3 / 6.4 |
| Qwen3.8 27B | 113 / 23 / 25 (**4.5-4.9x**) | 327 s / 2,418 s / 1,624 s | 3.7 / 2.1 / 1.1 |
| Gemma 4 12B | 240 / 96 / – | 240 s / 1,165 s / – | 6.1 / 4.6 / – |

## Accuracy, consistency and answer quality

**Accuracy against a full-precision reference.** `tools/ref_forward` is an independent f32 CPU implementation of
each model; the KL divergence of the next-token distributions (0 = identical, averaged over the prompt positions)
measures how far the int8 GPU arithmetic moves the model:

| Model | KL vs f32 | What it means |
|---|---:|---|
| Qwen3.6 35B-A3B (GSQ) | 0.0008 | the int8 engine and the f32 reference pick the same tokens with nearly the same probabilities |
| Qwen3.5 4B | 0.0021 | 0.0006 of it is int8 attention |
| Qwen3.8 27B (generation path) | 0.0003-0.0004 | run-to-run rounding noise is of the same size |
| Qwen3.6 35B-A3B UD-IQ4_XS (generation path) | 0.004-0.008 | a small change in attention can pick other experts |
| Gemma 4 12B (QAT, regression test) | 0.021 | Gemma 4 is ~7x more sensitive to int8 activations; one int8 scale per 64 V dims (instead of per row) removed the flipped predictions (code: 0.23 -> 0.018) |
| Qwen3.8 Flash Next (3.5K prompt, QSA) | 0.020 | the most sensitive model: every matrix group adds 0.02-0.10 |

**Perplexity against llama.cpp** (40 chunks of llama.cpp documentation, lower is better): Qwen3.5 4B 6.056 vs
6.141 (Vulkan), 9B 5.417 vs 5.499, 27B 4.274 vs 4.311. Speed did not cost quality: on the same GGUF files
arcinferzero's int8 path scores as well as llama.cpp's.

**Consistency.** The engine is deterministic: the same prompt gives the same logits, bit for bit, run after run.
That property is what every speed change is held to:

- arcinferzero (Level Zero) gives bit-identical logits to the OpenCL engine on every model.
- Speculative decoding (MTP block, DFlash2, both) gives exactly the text of plain decoding; a drift after ~40
  tokens on the 35B was found this way and fixed (the verification rows took another attention kernel).
- The SSD expert streaming gives the same logits whatever the cache size (41 % or 97 % of the 35B's experts,
  13-23 % of Flash Next's) and whether the GPU runs out of memory midway (the cache gives back memory in 1 GiB
  pieces): 56 of 56 regression runs identical, including 40 Flash Next prompts in a row.
- Tokenization equals `llama-tokenize`; chat templates render exactly like the GGUF's Jinja templates.

## Changes, newest first

| Date | Change | Before | After | Answers |
|---|---|---|---|---|
| 10-24 | Gemma 4 prompt attention, sliding layers (head 256, at most 8 q rows per work-group: 26B, 12B): each sub-group scores 2 blocks of 16 keys per step (attn_fa_gh -DKB=2), so each q load serves 32 keys; the same fma chain per key. E4B (16 rows) and the head-512 layers keep 1 (slower with 2); without the cooling pad, alternated | 11K-token code prompt: 26B sliding attention 8.6-8.7 s, TTFT 64.9 s; 12B 16.5 / 20.1 s, TTFT 120.3 / 124.7 s | 26B **7.1-7.2 s**, TTFT 63.3 s; 12B **13.4 / 14.0 s**, TTFT 119.0 / 120.7 s; E4B unchanged | logits bit-identical (26B, 12B, E4B; bitcheck of every model) |
| 10-24 | Gemma 4 prompt attention: the ring slot of a 16-key block computed once per block (cap is a multiple of 16) instead of a modulo per key, in attn_fa_g and in attn_fa_gh for head 512 (head 256 kept per key: 5 % slower otherwise); without the cooling pad, alternated with the committed kernels | 26B 13,921 tokens: TTFT 65.9-69.9 s, global attention 16.3-18.5 s | **64.4 / 64.5 s**, **15.2 s**; sliding attention unchanged (8.7 s); E4B unchanged | logits bit-identical (26B, E4B; same addresses) |
| 10-23 | Gemma 4 E4B (and E2B) prompts: the global layers' attention on the int8 K / V copy again (attn_ps, as before 2026-10-22), the sliding layers and generation stay float; laptop without the cooling pad, runs alternated | 13,921-token prompt 30.2 / 30.3 s (461 tok/s); 4K 648 tok/s; generation 14.1-14.3 tok/s; GPU 8.28 GiB | **22.6 / 22.7 s (615 tok/s)**, +33 %; **724 tok/s** (+12 %); 14.3 tok/s; 8.62 GiB, auto context unchanged (40,960) | KL 0.00075 -> **0.00088** (480 tokens, top token 476 / 480 both; f32 simulation 0.00081 -> 0.00145 with V and P rounded too); drafter text = plain; other models bit-identical; `ARC_G4_ATT8=0`: float |
| 10-23 | Flash Next prompt: the per-tile expert gate / up kernel reads the activation scales transposed to [token][K/32] (`asc_transpose` + -DASCT), one coalesced read per token instead of 16 lanes in 16 rows | 4K prompt (ids): gate / up 8.08 / 8.18 s GPU, 160.2 / 158.4 tok/s | **7.01 / 7.12 s**, **164.1 tok/s** (one run at 137 with a cold file cache) | bit-identical (Flash Next 4K logits; bitcheck of every model) |
| 10-23 | **Baseline with the laptop on a cooling pad** (no code change; runz, 64 tokens generated without a drafter, two runs each) | earlier single runs without the pad: 26B 14K TTFT 74-84 s, 12B 4K 73.7 tok/s (DeepSeek's audit run), E4B 14K 30.1 s | 26B **252 / 252 tok/s** (4K, gen 15.6-16.4), **210 / 210** (14K, TTFT 66 s, gen 14.0-14.2); 12B **137 / 137** (4K, gen 7.4-7.6); E4B **570 / 571** (4K, gen 15.0-15.3), 14K 30.5-34.5 s (the slower run right after loading from D:); Flash Next 162 / 162 (4K ids, gen 4.6-4.9), 3,291-token text 98-99 tok/s with the expert file out of Windows' file cache (174 when it was in it), gen 6.6-6.7 | – (measurement only) |
| 10-23 | Qwen3.8 Flash Next with the DFlash2 drafter (`--dflash DIR`; HC mean features, 8-row verification with single-token mixers, PLE / QSA rollback) | plain decoding, 1,097-token prompt, 128 tokens: 6.44 / 6.78 tok/s | **5.82 / 6.57 tok/s** (79 % accepted, 3.66 tokens per pass; ~370 expert misses per pass against ~76 per token): no gain, opt-in only | logits bit-identical to plain (128 tokens; 3,291-token prompt with QSA, 96 tokens); other models bit-identical |
| 10-23 | Gemma 4 E4B: the matmuls in one int8 pass again, the float attention kept (the user's KL limit is now 0.015) | KL 0.00000 (480 tokens, prompt and generation); 13,921-token prompt 45.4 s (306 tok/s); 200 tokens with the drafter 29.3 tok/s | KL **0.00075** prompt, **0.00093** generation (top token 476 / 474 of 480); **30.1 s (463 tok/s)**; **32.2 tok/s** | drafter text = plain; 26B / 12B / Qwen / Flash Next unchanged (bit-identical); `ARC_G4_Q2MM=1`: the two passes |
| 10-23 | Gemma 4 26B `--ctx auto`: the expert cache keeps at least 0.6 of the free memory (was 0.75) | auto context 32K-61K by variant; at 36K of depth with 49K: 11.4 tok/s (89 % of the experts cached) | **90K-131K**; with 131K: 10.5 tok/s (70 %), -8 % | same answers (memory split only) |
| 10-23 | Server start: the warm-up also runs a 512-token prompt (the large-chunk kernels' first use), not only one token | 35B with drafters, first 1,749-token request after the start 8.6 s (203 tok/s), later ones 3.5 s | **4.0-4.4 s** (439-444 tok/s); ready after 13 s | same answers; a model switched in for a request keeps the short warm-up |
| 10-23 | Gemma 4 sliding-window layers (head 256) in prompts: `attn_fa_gh` with 4 sub-groups and 4 rows per work-group instead of attn_fa_g with 32 rows | 26B 4K prompt 230 tok/s (sliding attention 4.17 s), 13,921 tokens 77.7 s; E4B sliding attention 4.3 s at 13,921 tokens | **254 tok/s** (2.26 s), **73.8 s**; **2.6 s**; with tonight's other two attention changes: 26B 14K 90.7 -> 73.8 s, E4B 83.5 -> 45-47 s, 12B 194-210 -> 136 s | logits bit-identical against attn_fa_g on 13,921-token prompts (26B, E4B, 12B: the windows in use) |
| 10-23 | Gemma 4 head-512 prompt attention: several query rows per work-group (`attn_fa_gh -DGR`), each key / value row read once for all of them; as many rows as fit 64 output floats per lane (E4B 4, 26B 2, 12B 1) | E4B 13,921-token prompt 59.8 s (after the 8 sub-groups) | **45-47 s** (+27 %; 83.5 s before both changes) | logits bit-identical (E4B 13,921 tokens; 26B, Qwen 35B / 27B, Flash Next 480-token checks) |
| 10-23 | Gemma 4 head-512 prompt attention (`attn_fa_gh`): 8 sub-groups per work-group, 4 output dims per lane (was 4 and 8); the 12B's 16 q heads per kv head now fit it too (64 floats per lane, no spill) | 26B 13,921-token prompt 90.7 s, 4K 220 tok/s; E4B 13,921 tokens 83.5 s; 12B 194-210 s (attn_fa_g) | **77.7 s** (+17 %), **230**; **71.1 s**; **138-144 s** (+40 %) | logits bit-identical (26B 480 tokens, E4B / 12B 13,921 tokens against the old kernel choice); Qwen 35B / 27B, Flash Next unchanged |
| 10-23 | Flash Next, opt-in experiment `ARC_FN_Q2=1`: the dense matmuls' inputs as two int8 passes (Gemma 4's mechanism; experts, attention and the Q8_0 HC weights unchanged) | KL 0.0483 (363 tokens, top token 331 / 363), generation path 0.0375; 4K prompt 191 tok/s | KL **0.0384** (337 / 363), generation path 0.0353; **160 tok/s** (-16 %) | not the default: the simulation shows the HC weights as Q8_0 alone cost 0.035, so < 0.01 needs every part precise (QUEUE F21.13, the user decides) |
| 10-22 | Gemma 4: the int8 K / V copy only for the layers the assistant drafter reads (the sources of the last sliding and global layers; none without the drafter) | context memory per 1K tokens: 26B 42.6 MB, E4B 34.3 MB; a saved 1,511-token 26B conversation 492 MB | **33.7** (31.5 without the drafter), **27.7**; **363 MB**; the freed memory goes to the 26B's expert cache | logits bit-identical (26B, E4B; with and without the drafter); session round trip identical; `--slots-check` passed (26B, both) |
| 10-22 | Gemma 4 E2B / E4B (shared KV layers): sliding-window layers keep the ring of window + chunk + 4096 like the others instead of the full context | E4B: ~115 KB per token in the sliding layers' caches, auto context 94,208 (110,592 without images); ctx 16K 8.84 GiB | **auto context 131,072** (the trained maximum); ctx 16K **7.96 GiB** | 13,921-token prompt in a 16K context (the ring wraps) + 32 tokens: logits bit-identical with and without the drafter; `--slots-check` passed |
| 10-22 | Gemma 4 two-pass mode with streamed experts: prompt chunk 2048 instead of 8192 (the sliding layers' float K / V rings hold window + chunk + 4096 rows) | 26B, ctx 16K: expert cache 39 %, generation after a 13,921-token prompt 9.9 tok/s; 4K prompt 216.5 tok/s | **77 %**, **13.0 tok/s**; 216.6 (same); TTFT of the 14K prompt the same (92 s) | chunk size does not change a prompt's logits here (moe_rg and the row-independent attention) |
| 10-22 | Gemma 4 two-pass mode, generation / verification attention: flash-decoding (`attn_fa_g -DFA_SPLIT` + `attn_fa_gc`: a row's keys in up to 64 splits of >= 8 blocks, chosen from the row alone) | 26B at 13,921 tokens of context: global attention 9.16 s per 64 tokens, generation 16.9 s (profiled); E4B 200 tokens 17.2 / 34.5 tok/s with the drafter | **1.01 s**, **9.6 s**; E4B **18.8 / 38.1** | KL decode 26B 0.00419 (unchanged; top-1 474 / 476 of 480), E4B 0.00000 (480 / 480); greedy text with the drafter = without (26B, E4B) |
| 10-22 | Gemma 4 two-pass mode, prompt attention of the head-512 layers: `attn_fa_gh` (a work-group per row and kv head: each block of keys scored for all 8 q heads with one read of the key rows, the output dims split over 4 sub-groups) | 26B: global attention 4K prompt 2.1 s, 13,921-token prompt 24.2 s (TTFT 83.5 s) | **1.5 s**; **18.2 s** (TTFT **77.4 s**) | prompt + generation logits bit-identical (480 tokens) |
| 10-22 | moe_rg (prompt experts): Q4_0 / Q8_0 blocks read with vblock instead of copying each superblock into registers (2.6 KB spill per thread gone) | 26B 4K prompt: 508-514 tok/s, two-pass mode 200 | **526**, **218** | two-pass prompt + generation logits bit-identical (480 tokens) |
| 10-22 | Gemma 4 two-pass mode, drafter verification (2-4 tokens): `gemv_mt_v -DQ2F` takes the residuals as further token slots, both passes in one launch | with the drafter, 200 tokens: E4B 22.9, 12B 11.3, 26B 23.2 tok/s (plain mode 42.5 / 21.9 / 26.8) | **34.5**, **19.0**, **25.2** | E4B logits with the drafter bit-identical to the two-launch form (120 tokens); 26B greedy text with the drafter = without (200 tokens) |
| 10-22 | Gemma 4 two-pass mode, prompt attention (`attn_fa_g`, F21.11 phase 4b): sliding layers 16 rows per work-group sharing each key block in local memory (`-DFA_ROWS`), global layers 2 q heads per sub-group (spill) | 26B, 4K prompt: 141-146 tok/s (attention: sliding 10.5 s, global 6.2 s) | **200** (4.5 s, 4.0 s) | prompt and generation logits bit-identical (480-token prompt + 16 tokens) |
| 10-22 | Gemma 4 two-pass mode: the experts of generation and verification (T <= 8) through the per-tile kernels with both passes fused (`moe_gateup_q` / `moe_down_q -DQ2F`, tiles of 4 / 8 slots; F21.11 phase 3) instead of two `moe_rg` passes each | 26B, 200 tokens: 9.63 tok/s, with the drafter 15.85 | **16.76**, **17.50** | KL decode path 0.00444 -> **0.00419**, top-1 475 -> 476 / 480; greedy text unchanged (200 tokens) and the drafter's equal to plain decoding; prompt chunks > 8 tokens still through `moe_rg` |
| 10-22 | Gemma 4 two-pass mode (`--q2`): one-token GEMVs read the weights once for both passes (`-DQ2F`, F21.11 phase 1) | 26B, 128 tokens, A/B alternated: 9.42 / 9.45 tok/s | **11.18 / 11.60** (+18-23 %); with the drafter 15.9 (200 tokens) | logits of 64 generated tokens bit-identical to the two-launch form (so KL unchanged, 0.0044); greedy text with the drafter = without (200 tokens) |
| 10-21 | Sampled tool calls: constrained sampling skips a negligible vocabulary tail (`tools/sampler.h`) | 35B, temperature 0.7 / top_k 20 (as Pi): short tool call 4.8 tok/s, Pi's first `monitor_start` call 2.45 (greedy 32.2) | **20.4**; a 1,240-token `write` call 24.3 (greedy 24.4) | the same tokens as before in all 9,000 seeded test draws (`test_sampler`); greedy unchanged (it never reaches the tail) |
| 10-20 | **Gemma 4 26B drafter** (Unsloth's assistant, 3 drafts) | server, 400-token answers: code 18.8 tok/s, Indonesian 19.4 | **27.0**, **21.7** | greedy text bit-identical to plain decoding (128 tokens) |
| 10-20 | Gemma 4 verification attention one row per launch | drafter logits off plain decoding (26B up to 2.2, text changed after 7 tokens; 12B 1.3; E4B 0.85); 12B short / 11K 21.3 / 14.8, E4B 47.4 / 30.9, 26B 11K 21.0 tok/s | bit-identical; 23.2 / 14.9, 48.9 / 30.0, 21.0 | 12B KL 0.02061 (unchanged), greedy = llama.cpp |
| 10-20 | Gemma 4 26B prompt: experts' gate / up through moe_rg (GELU epilogue) | 4K 451 tok/s, 8K 405 | **506**, **457** | KL 0.0985 -> 0.0993 (480 tokens, noise); same greedy text |
| 10-20 | MTP block + DFlash2 (both drafters, picked per step): a DFlash2 step now also writes its token's row of the MTP block's KV cache (it was left unwritten, so the MTP drafts after DFlash2 steps read stale memory) | 256-token answers, best of two: 35B prose 20.56 tok/s, code 37.45; 27B prose 8.91, code 15.76 | 21.01, 38.65; 8.78, 16.29 (noise level) | text identical; drafts now independent of what the slot held before (runz `--slots-check` 35B + DFlash2: was DIFFERENT, alone vs interleaved) |
| 10-20 | Flash Next: the two-pass int8 HC mixer input (`ARC_HC_F32=4`) is now the default (`ARC_HC_F32=0`: one pass) | KL 0.0647, top token 324 / 363; prompt 4096 tokens 168.4 tok/s, 9000: 148.3 | KL **0.0483**, top token **331 / 363**; 159.8, 138.6 | the user chose accuracy over the 5-7 % |
| 10-19 | **Gemma 4 26B-A4B** (MoE, Google QAT Q4_0) on arcinferzero: experts streamed from the SSD (13.2 GB expert file, written in 73 s) | not supported | prompt 450 tok/s (4K), 410 (8K); generation 19.6-20.6 tok/s; expert cache 85-93 % (runz), 72 % at the server's 106K context | KL 0.073 prompt / 0.068 decode path; f32 router (was 0.079 with an int8 router); greedy answers correct in English and Indonesian; images: embeddings against the numpy reference cosine >= 0.999997, a test picture described correctly |
| 10-19 | **Batched decoding for Gemma 4** (26B-A4B; `--slots 2`) and `--ctx auto` for slots of streamed MoE models: 128K in all, at least 32K per slot | two answers at once taking turns: prose 19.80 tok/s together, code 19.60; 2 slots of the 26B got 8,192 tokens each (the cache's 3/4 floor), 35B 110,592 | **26.15** (1.32x), **26.47** (1.35x); engine 2 / 3 slots 1.30 / 1.58x; 26B 2 x **32,768** (cache 65 %), 35B 2 x 65,536 (73 %); one slot unchanged (106,496 / 131,072) | bit-identical to each slot alone (runz `--slots-check`, 220 and 1,500-token prompts: the 1,024-token windows wrap); server texts equal batched / turns |
| 10-19 | **Batched decoding over slots** (`--slots 2`, Qwen): both requests' next tokens and drafts in one pass | two answers at once taking turns: 27B 7.32 tok/s together (prose), 11.63 (code); 35B 20.24, 28.97 | **10.48** (1.43x), **15.42** (1.33x); **24.65** (1.22x), **30.73** (1.06x); engine without drafts 27B 2 / 3 / 4 slots 1.78 / 2.43 / 3.04x, 35B 1.35 / 1.67 / 1.91x | bit-identical to each slot alone (runz `--slots-check`: 27B and 35B plain / MTP / DFlash2); server texts equal with and without batching; one slot unchanged (logits dumps of the previous build) |
| 10-19 | **Two conversations at once** (`--slots 2`): each slot its own KV cache and states, requests take turns step by step by GPU time | a second Pi window waits for the first one's whole answer; switching conversations goes through the session cache | both answer at the same time; a question during another slot's 46K-token prompt (35B) answered after **9.6 s** (first try, turns every step: 49 s); context per slot: 27B 49,152, 35B 110,592 (expert cache 67 %) | bit-identical alone vs interleaved (runz `--slots-check`: 27B, 35B + MTP / DFlash2, Flash Next 9000 tokens); server greedy text the same with 1 and 2 slots |
| 10-19 | Flash Next prompt chunks: the HC mixers' up input as **two int8 passes** (values + rounding residual), opt-in `ARC_HC_F32=4` | KL 0.0647 against the f32 reference (363 tokens, max 1.56), top token 324 / 363; prompt 4096 tokens 168.4 tok/s, 9000: 148.3 | KL **0.0483** (max 0.78), top token **331 / 363**; 159.8 (-5.1 %), 138.6 (-6.5 %); not the default (cost above 5 %) | default unchanged (same KL to the last digit); f32 mixers (`=1`) KL 0.0369 at -22 % |
| 10-19 | **MoE models as arcinferzero packages** (`serverz --pack`): the non-expert weights + the expert file, the GGUF deleted; the expert file the engine already wrote moves into the package instead of being written again | on disk: Flash Next GGUF 78.7 GiB + expert file 52.1 GiB; 35B GGUF 17.0 GiB + expert file 14.7 GiB | Flash Next **31.0 + 52.1 GiB** (pack 5 min 20 s), 35B **2.4 + 14.7 GiB** (1 min 22 s): about 65 GiB freed, the same speed | bit-identical to the GGUF (Flash Next 9000 / 4096-token prompts; 35B 9000-token prompt + 32 generation steps) |
| 10-18 | **Deep expert prefetch** on the GPU's copy engine: while the GPU runs layer L of a prompt, every missing expert of L+1 is read from the SSD and copied into its slot | Flash Next prompt 4096 tokens 107 tok/s, 9000: 116.9, 32K: 131.7 (GPU idle on SSD reads 62.4 s); 35B 32K at ctx 131K 488.6; 35B with a 41 % cache 541 | **166**, **144.5**, **162.6** (idle 5.8 s); **557**; **592** | bit-identical (56 regression runs) |
| 10-18 | 35B prompt chunks 4096 -> **8192** tokens (ctx 262K) | 32K prompt 533 tok/s, 128K 306; generation at 128K 12.8 tok/s; expert cache 81 % | **604**, **324**; 12.6; 77 % | not compared bit for bit (the chunk size groups the prompt arithmetic differently) |
| 10-18 | **Drafter per model in Pi's /model**: `-nodf` (MTP block only), `-df` (DFlash2 only, MTP block not loaded) | one setting: both drafters picked per step | 35B Python code: both 45.1 tok/s, **DFlash2 only 49.9**; Indonesian: both 21.0, **MTP only 21.9**; 27B `-df` context 69,632 -> **94,208** tokens | greedy text identical in every mode |
| 10-18 | Server **model switch** keeps one Level Zero context (copy list, events and staging of the old one were reused by the new model) | handles of a destroyed context after a switch | 35B -> 35B-df -> Flash Next -> 35B in one server, correct answers | – |
| 10-18 | `--v4`: a 4-bit copy of V for generation (opt-in) | 35B at 250K depth 92.5 ms per token (10.6 tok/s) | **84.9 ms (11.5 tok/s)**; 125K prompt 321 -> 316 tok/s; KV memory +25 % | KL 0.005-0.008 vs int8 with the int8 sink / recent windows |
| 10-18 | Flash Next QSA block scores: 8 query rows per work-group (`qsa_score_rt`) | 125K prompt 78.6 tok/s (26.5 min); 32K 131-146 | **123.4 (16.9 min)**; 153-154 | bit-identical |
| 10-18 | Automatic context reserves 128K tokens before sizing the expert cache | MoE models ~32-40K by default | **131,072** tokens for 35B and Flash Next (35B cache 81 %, generation as fast as at 64K) | – |
| 10-17 | Host work out of the speculative loop (one-pass top-16, vectorized float exp) | 35B code DFlash2 48.4 tok/s, 27B 16.9 (one code prompt, measured back to back) | **52.7**, **18.3** | identical |
| 10-17 | DFlash2 drafter weights in Q4_0 | 27B drafter layers 539 ms per answer | **296 ms**, 0.84 GiB less memory | drafts accepted as often |
| 10-16 | **DFlash2 block drafter** (7 tokens per guess) + MTP | 27B Python code 4.85 tok/s; 35B 15.7 | first version 11.5 / 44.4; **18.1** / **49.8** in the 17 Oct table (96-token answers; other prompts give 45-53 on the 35B, e.g. 49.9 with DFlash2 only on 18 Oct) | identical to plain decoding |
| 10-16 | arcinferzero (Level Zero) becomes the default engine | – | the same speed or faster on every model | bit-identical to OpenCL |
| 10-14 | 35B experts streamed from the SSD (arcinferzero) | context limited by the GPU memory | the full 262K context fits with images and MTP | bit-identical |
| 10-13 | MTP speculative decoding for the 35B (MoE in the MTP block, Q3_K / K-quant experts) | 19.7 tok/s | **22.3** | identical to plain decoding |
| 10-13 | MoE prompts through an int8 GEMM (experts decoded per group of 64) | 35B 2048-token prompt 476 tok/s | **642** | per-layer outputs equal to the old kernel within ~1e-7 |
| 10-12 | Just-in-time weight decoding with vector loads (27B prompts) | IQ3_S / IQ3_XXS / IQ4_XS decoding 5.8 s (11K prompt) | **2.2 s** | bit-identical |
| up to 10-12 | The prompt path: int8 DP4a GEMM, int8 attention end to end, layer-major prefill, split-K | 27B 11,240-token prompt: first version 116 s (llama.cpp Vulkan 590 s, SYCL 504 s) | **89 s** (OpenCL), now pp4096 ~135 tok/s | measured against the f32 reference |
| before 10-10 | Gemma 4: one int8 scale per 64 dims for q/K/V instead of per row | code KL 0.23 with flipped predictions | **0.018** | accuracy fix (Gemma's attention 1.5-1.8x slower) |

### What did not pay off (kept out on purpose)

- Larger expert staging by default (`ARC_XSTAGE=256`): prompts +5-11 % but generation -2-3 % (host RAM taken from
  the cache). Deep prefetch gets the overlap without it.
- GEMM tile / split-K / work-group autotune: the defaults were already the fastest on 27B and 35B.
- 35B expert decoding once per layer across chunks: ~1 GB of extra buffers would come out of the expert cache;
  8192-token chunks give most of the gain.
- Faster attention variants for Qwen prompts (`attn_ps`, other tilings): slower than `attn_q8` with 256 registers.
- More in [lessons-intel-arc-igpu.md](lessons-intel-arc-igpu.md#5-tried-and-not-kept).

## Adding a row

Measure before and after on the same build options with a pause in between, best of two when a single number is
noisy, and state the check: bit-identical logits (`runz --dump-logits A.bin` vs `B.bin`, `cmp`), identical greedy
text, or KL against the f32 reference / the int8 cache (`runz --ppl-kl-save` / `--ppl-kl`) for changes that are
meant to change the numbers.
