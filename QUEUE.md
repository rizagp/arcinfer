# Optimization queue (started 2026-10-12)

Target: token generation using > 90 % of measured bandwidth (~75 GB/s, so > 67.5 GB/s average per token) for
every chat model in the Hugging Face cache, and vision / audio working on every model that has the encoder for it.
Status: ✅ done · 🔄 in progress · ⏳ queued · ⚠️ note · [~] partial

| # | Task | Status | Notes |
|---|---|---|---|
| 1 | Q4_0 support (Google's Gemma 4 QAT): prefill, fast GEMV, fused GLU | ✅ | E4B: prefill 788 tok/s, generation 20.9 (drafter 26.8) tok/s |
| 2 | Measurement tool: weights read per token and bandwidth efficiency, per model | ✅ | `run.exe` prints `weights read: … GB/s (… % of ~75 GB/s)`. First measurements (12B download running, numbers fluctuate): 4B 60–67 %, 9B 72 %, E4B 68 %, 27B 36 % |
| 3 | Detect Google mmproj (`gemma-4-E4B-it-mmproj.gguf` does not start with `mmproj`) | ✅ | also: non-GSQ 35B marked as not supported yet |
| 4 | Gemma 4 12B (Google): test text, image, audio (gemma4ua) | ✅ | generation 9.1 tok/s (85 % bandwidth), drafter 17–20; image and audio work. Fixed: the Google template uses adjacent strings (`"a" "b"`), which minja does not support → server crashed at start |
| 5 | Bandwidth efficiency per model (measure → optimize the one furthest from target) | 🔄 | stage 1 done: Q8_0 and IQ2/IQ3 weights rearranged on the GPU (aligned blocks) → GEMV Q8_0 37→77, IQ2_XXS 27→60, IQ3_XXS 41→62 GB/s. Now: 4B 79 %, 9B 86 %, E4B 81 %, 12B 82 %, 27B / Swift 72 % (before 77/83/76/80/65 %). Remaining: non-weight kernels (DeltaNet recurrence, norm, attention), IQ1_M |
| 6 | E4B vision on the GPU (gemma4v: 16-layer ViT, width 768) | ✅ | error vs f32 reference 0.11–0.17 %; 1120 tokens 3.2 s (attention 2.8 TFLOPS, GEMM 2.2). Also fixed: per-layer input (PLE) for image tokens, E2B/E4B image tokens causal (llama.cpp) |
| 7 | E4B audio on the GPU (gemma4a: 12-layer conformer, width 1024) | ✅ | transcription exact word for word; 7.2 s of audio encoded in 226 ms (first request ~23 s: kernel compilation + weight upload) |
| 8 | Qwen3.5 9B vision (encoder width 1152, head dim 72) | ✅ | error 0.38 % vs f32; screenshot 0.75 s, 2025-token image 8.2 s (head-72 attention still ~40 % above ideal: register spill) |
| 9 | Qwen3.6 35B-A3B UD-Q3_K_S: MoE kernels for standard GGUF types | ✅ | works: generation 18.9 tok/s (58 %), prefill 465–480 tok/s, automatic context 61K (17 GiB); prefill result = token-by-token, answers correct |
| 10 | Regression tests: new references for Google Gemma, run_tests.ps1 | ✅ | all pass (12 checks); Google Gemma 12B KL 0.0206 |

Not included: Qwen3-ASR 1.7B (not needed), bge-small (embedding), laya (not a chat model).
⚠️ Final speed measurements are taken after the 12B download finishes (CPU and GPU share power and heat).

---

# Phase 2: next optimization plan (drawn up 2026-10-13)

Basis: per-kernel profile of each model. Summary of where things stand:
- Token generation for dense models (4B, 9B, 12B, E4B, 27B) is already close to the practical limit: 72–86 % bandwidth; the
  remaining time is in non-weight kernels. The main lever is MTP / drafter (27B: 4.5 → 7.9 tok/s), and that is already on.
- Dense-model prefill is almost entirely int8 GEMM at ~65 % of DP4a peak (practical limit). DeltaNet is only 4–9 %.
- Still far from the limit: **35B expert kernels** (prefill 0.04 TOPS, generation ~41 GB/s) and
  **attention at long context**.

Order by largest impact per effort. Every task: measure before and after (best-of-two / alternating A-B),
results must be identical (same greedy output, prefill = token-by-token), then run_tests.ps1.

| # | Task | Status | Expected result | Cost / risk |
|---|---|---|---|---|
| F1 | **35B expert prefill with int8 GEMM**: decode experts to int8 per 64 experts (`decode_w8`), then one grouped GEMM kernel (lane = column, shared activations, DP4a) for all tiles (expert × token) | ✅ prefill 2048 tokens **476 → 642 tok/s** (+35 %); per-layer results match the old kernel (difference ~1e-7). Experts are decoded to int8 per 64 experts with the new kernel `decode_w8_sl` (rows copied to local memory first: IQ3_XXS 1038 → 518 ms, IQ4_XS 433 → 303 ms per chunk), then `gemm.cl` in `MOE_TILES` mode. Used for chunks ≥ 1152 tokens (decoding all experts takes ~1 s per chunk; break-even at 1024 tokens). Staging 75 MB | expert work 2.7 → ~1.1 s per 2048 chunk; 35B prefill **476 → ~700–750 tok/s**; TTFT for long prompts down ~1/3 | a few hours; staging buffer ~70 MB → automatic context −~7K tokens |
| F2 | **Long-context prefill attention** (27B, 9B, 35B): first measure its share at 16K / 32K tokens, then improve the prefill attention kernel (larger tiles, K/V through local memory, int8 scores + f32 accumulation) | ✅ (partial) measured on a 21,590-token prompt: attention 35B 22 % (9.4 s), 9B 14 %, 27B 12 % (22 s). Fix: 4 q-heads per sub-group for Qwen when the GQA ratio is divisible by 4 (attn_q8): 35B 9.4 → 8.4 s, 9B 8.2 → 7.0 s, same results; 27B (6 per kv head) unchanged. `attn_ps` (split dim) tried: slower. Largest remaining cost for 35B on long prompts: per-chunk expert decode (~0.85 s × 11 chunks); needs one decode per layer for all chunks (next piece of work) | unknown; 27B at 11K used to spend ~6.4 s in attention, and its share keeps growing with context | medium; accuracy must be checked at long context (KL) |
| F3 | **35B generation: expert kernels** (~41 GB/s → target ~60): the down matrix has short rows (512) → several rows per sub-group with a combined reduction; gate/up: reduce per-tile overhead | ✅ (partial) the expert kernels are already at the practical limit: rows-per-sub-group variants ±2 %, down + combine fused 212 vs 211 ms (not used). Instead: DeltaNet qkv + z (+ ab) stacked into one GEMV during generation (`load_gdn_in`, no double copy): 35B **19.2 → 19.8 tok/s** (+3 %), 27B 4.75 → 4.8–5.0; 9B same (qkv / z types differ in many layers). Q8_0 through `gemv_v` tried: slower | 35B generation **18.8 → ~20 tok/s** (+6–7 %) | medium; no effect on memory |
| F4 | **MTP with 3 guesses per step** for Qwen (27B, 9B, 4B): the engine is currently limited to 2 | ✅ 27B code 7.8 → 9.5 tok/s (+21 %), text +2.5 %; 4B / 9B same → automatic default 3 for 27B-class, 2 for the others; 27B context 86K → 82K | 27B code: 7.9 → maybe ~8.5 tok/s; plain text less | small; results must be identical to running without MTP |
| F5 | **Q8_0 single copy in the GEMM layout** (35B): a new GEMV that reads the interleaved layout, so the per-chunk Q8_0 decode (0.22 s) goes away | ⏸️ not done: after the new decode kernel (`decode_w8_sl`) the Q8_0 decode per 2048 chunk is down 211 → 72 ms (~2 % of prefill), so a new GEMV for another layout is not worth it | 35B prefill +~5 % | medium; the new GEMV must be as fast as the current one |
| F6 | **35B vision**: the Unsloth repo has an mmproj (the same Qwen3-VL encoder as 4B/9B); the server does not load vision for MoE models yet | ✅ works (2026-10-14, Unsloth mmproj-BF16): text in the image read exactly, encoding 1,008 tokens ~3 s (first time ~60 s: kernel compilation), prefill ~400 tok/s, generation ~18 tok/s. But the encoder (~1.1–1.3 GB) barely fits next to 35B in 18.3 GiB: context with images only 4,096. So `35b` stays without vision (40,960), images only through `35b-vision`. Next options: Q8_0 encoder weights (saves ~0.45 GB), load the encoder only when there is an image, or raise the GPU budget | new feature: 35B can read images | small–medium; encoder memory (~1 GB) reduces context |
| F7 | (optional) **9B vision attention, head 72**: split one query row across two lanes so the registers fit | ⏸️ not done (optional; the compiler still uses 128 registers for this kernel, outcome uncertain) | 9B vision attention ~1.6 → ~1.1 s per large image | outcome uncertain (compiler) |
| F8 | (needs a decision) **MTP for 35B**: needs the MTP version of the file, Q3_K support, Q2_K/Q4_K experts, and MoE in the MTP block | ✅ `35b-mtp` (MTP repo file): results **bit-identical** to running without MTP, generation **19.7 → 22.3 tok/s** (+13 %, 83 % of guesses accepted), prefill −3.4 %, automatic context 40K (variant `35b-mtp-std` 98K). Added: Q3_K in the GPU kernels, experts with min (Q4_K / Q2_K / Q5_K) in the expert kernels, MoE in the MTP block, 1 guess by default for MoE (1 and 2 equally fast; saves one state copy). Fixed: 35B verification attention now goes through the same kernel as single tokens (before, MTP text diverged after ~40 tokens) | 35B generation ~1.5× | large; Q3_K_M context drops to ~40–50K |

Already tried / measured and **not** pursued:
- Level Zero (gaps between kernels only ~1.6–2 %), out-of-order queue + events.
- "Chunked" DeltaNet (DeltaNet is only 4–9 % of prefill), f16 DeltaNet state (saves ~1.5 %, risk of accumulating error).
- Split-K GEMV for small matrices (not faster overall), per-type GEMV rules from single-matrix benchmarks
  (not faster inside the model), gemv_glu for 27B (only 24/64 layers fit, ~0.3 %).

Suggested work order (updated 2026-10-13, 35B MTP file downloading): **F8 → F1 → F4 → F3 → F2 → F5 → F9 → F6**, then F7.

## F8 (priority 1): MTP for Qwen3.6 35B-A3B, details

Goal: 35B generation 18.8 → ~27–30 tok/s with the MTP block, results identical to running without MTP, prefill not slower
by more than ~5 %.

| Step | Content | Notes |
|---|---|---|
| 8.1 | Inspect the downloaded file: `blk.40.*` tensors (MTP block, `nextn.*`), their expert types, size; context estimate | MTP repo file: main layers same as the non-MTP version, MTP block ±0.5 GB |
| 8.2 | **Q3_K type**: GPU decoder (`decode32`), CPU reference, decode to int8 for prefill, GEMV; kernel tests in `test_gemm` | not supported at all yet; in the MTP file it is used by the MTP-block experts |
| 8.3 | **Typed experts with a minimum value (Q2_K, Q4_K)** in the MoE kernels: activation sums per 16 (`absum`) for gate/up and down (`moe_quant` writes them too) | the MoE kernels currently reject these types |
| 8.4 | **MoE version of the MTP block**: load the router, shared expert and experts of the MTP block (`load_moe`), then run the MoE layer in the MTP path (`mtp_rows`) — for the prompt (batch) and for guessing tokens | the 4B/9B/27B MTP blocks use a plain FFN |
| 8.5 | Verify 2–3 tokens at once in the main MoE layers (small tiles) + DeltaNet state rollback (already exists) | measure: verification speed vs 1 token |
| 8.6 | **Test**: greedy identical with/without MTP; prefill with MTP ≤ 5 % slower (alternating A/B); acceptance rate and tok/s on code and plain-text prompts | pass criteria |
| 8.7 | Alias: the file name is the same as the non-MTP version (`Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`), so it is distinguished by repo name → `35b-mtp` (id `qwen3.6-35b-a3b-mtp`); Pi list, CHANGELOG, README | `35b` / `35b-s` unchanged |

Risk: Q3_K_M context with MTP drops to ~40–50K tokens (from 94K); verifying several tokens in the MoE layers
reads more experts at once (different tokens pick different experts), so the speedup may be below 27B's.

## New small tasks

| # | Task | Status | Expected result |
|---|---|---|---|
| F9 | Pi model list follows the vision setting: `pi-arc --vision off` → model marked text-only and the displayed context is longer | ✅ (via F10) | /model consistency; no speed effect |
| F11 | **Int8 image-encoder weights, activations stay F32**: a `vit_gemm` variant with Q8_0 weights unpacked to F16 in the kernel; test the difference vs the f32 reference and small text in test images; used if there is no degradation | ⏸️ postponed (2026-10-14): arcinferzero (35B experts in GPU cache + SSD) leaves room for the 0.9 GB encoder, so 35B vision no longer depends on F11. Note: a Q8_0 mmproj is currently also unpacked to F16 on the GPU | 35B encoder ~0.9 → ~0.48 GB, `35b-vision` 4K → ~40K tokens; 4B / 9B a bit more context | ~1 day; full int8 earlier (activations too) was off by 10–23 %, so activations must stay F32 |
| ✅ | Vision on/off option: `pi-arc --vision off` / `arc-server -Vision off` (done 2026-10-13) | ✅ | Qwen3.5 9B: 143K → 217K tokens with vision off; switching models via /model follows the setting used when the server was started |

## F10: all models and their variants show up automatically in Pi /model

Request: when Pi is opened, /model shows every downloaded model with all its variants: standard, with
MTP, with vision, with both. Without editing anything when a new model arrives.

Already in place (basis):
- The server scans the Hugging Face cache + `%LOCALAPPDATA%\arcinfer\models` on every start; `pi-arc` / `arc-server`
  rewrite Pi's model list from that scan.
- Every file has a unique id (2026-10-13): a file with the same name from another repo gets `-mtp` if it carries an
  MTP block, otherwise `-2`, `-3`; old ids (`qwen3.5-4b`, `35b`, ...) do not change.

Design:

| Step | Content |
|---|---|
| 10.1 | **Capabilities per file** from its contents, not from a name table: MTP block (`nextn_predict_layers` or a Gemma `mtp-*.gguf` drafter in the same folder), image / audio encoder (an mmproj in the same folder that the engine can run), architecture supported or not |
| 10.2 | **Variants per file**: only combinations that actually exist. Id = base id + suffix: `<id>` (standard: no MTP, no vision), `<id>-mtp`, `<id>-vision`, `<id>-mtp-vision`. Examples: Qwen3.5 9B → 4 variants; Qwen3.8 27B (MTP, no mmproj) → 2; non-MTP Qwen3.6 35B (no local mmproj) → 1 |
| 10.3 | **Automatic context per variant**: computed separately (vision off → no encoder reserve; MTP off → no MTP block), stored in the ctx-auto cache as now; shown in the variant name |
| 10.4 | **Names in Pi**: easy to read, e.g. `Qwen3.5 9B · MTP · vision · 143K`; sorted per model, most complete variant first; `text`/`image` input according to the variant |
| 10.5 | **Server**: picking a variant via /model = reloading the same file with different settings (mmproj on/off, MTP on/off); if only the variant differs from the one running, it is still reloaded (10–70 s) |
| 10.6 | **`pi-arc` / `arc-server`**: accept variant ids (`pi-arc 9b-vision`, `pi-arc 35b-mtp`); old short names = that model's default variant (default: MTP + vision when available, same as now) |
| 10.7 | **Test**: every variant appears in /model, switching variants via /model really changes the settings (check `/props`: vision, MTP, n_ctx), old ids still work, a newly downloaded model appears without code edits |

Notes:
- Number of entries: with the current 9 files, about 20 entries. If that feels too crowded, an option can be added to
  hide the "standard" variants.
- Scanning and computing context per variant takes a few seconds per new variant (only once, then cached).
- The `--vision off` / `--no-mtp` options of `pi-arc` stay, equivalent to picking the corresponding variant.

**F10 status: ✅ done (2026-10-13).** 22 entries from 9 files; old ids unchanged (= most complete variant); other variants
`-mtp`, `-vision`, `-std`; context per variant (e.g. 27B: 86K with MTP, 123K without; 9B: 143K full, 262K
standard); switching variants via /model = reloading with the variant's settings (tested: vision and MTP follow);
`pi-arc --vision off` / `--no-mtp` picks the matching variant.

Ordering with other tasks: F10 is small–medium and does not depend on F8. Proposal: do F10 while the 35B MTP file
downloads, then F8 (which automatically adds the `35b-mtp` variant).

## arcinferzero: MoE with experts in GPU cache + SSD (Level Zero), started 2026-10-14

- Phase 1 ✅ Level Zero test (`zero/ze_probe.cpp`): OpenCL kernel binaries can be loaded by Level Zero; device memory 69.9 GB/s,
  host 60.5 GB/s; host memory above the GPU budget can be used but only at 9–21 GB/s (Windows moves the memory around),
  so the cache stays within the GPU budget. Kingston NVMe SSD (`bench/ssd_bench.cpp`): ~3.5–4.2 GB/s for random reads
  of 1–4 MB with 16 requests in flight.
- Recording of 35B expert routes (`ARC_DUMP_ROUTES`, 3 sessions x 400 tokens) + simulation (`zero/cache_sim.py`): LRU is best
  (pinning popular experts is worse); hit rate 99.8 % (cache 88 %), 98 % (70 %), 95 % (50 %), 87 % (30 %).
- Decision: **Windows minimum 14 GB**; expert cache size **is configurable** — the default is computed automatically so the
  server total <= RAM - 14 GB (35B at 262K: cache ~10.8 GiB ≈ 83 %, ~19.5 tok/s); `--expert-cache-gb` / `--expert-cache-pct`,
  `--windows-reserve-gb`; no cache in RAM; direct SSD reads (FILE_FLAG_NO_BUFFERING).
- Expert file on the SSD (`%LOCALAPPDATA%\arcinfer\zero\<model>.experts`): each expert whole, sequential, 4 KB aligned, in a
  GPU-ready layout (blocks as in arcinfer), so one expert = one direct SSD read with no conversion. Missing →
  created automatically the first time the model runs; present and matching (size, date, GGUF fingerprint) → used; not matching →
  recreated. Can be deleted at any time (35B ~14 GB, Flash Next ~40–45 GB).
- Phase 2 results (2026-10-14), `zero/` (`bin/runz.exe`, `bin/serverz.exe`; regular arcinfer unchanged):
  - OpenCL→Level Zero shim (`zero/clrt.h`) bit-identical to arcinfer and faster (35B without streaming: TTFT
    6.45 → 4.07 s, generation 16.9 → 18.3 tok/s; 27B identical, prefill the same).
  - Expert streaming (`zero/expert_store.h`, `zero/xcache.h`, `-DXCACHE` kernels in `zero/kernels/decode.cl`): expert
    file 15.1 GB (41 layers including MTP) created once in 144 s; results **bit-identical** to arcinfer (chunk
    2048, logits of 48 tokens exactly the same).
  - Cache: one LRU list per record size; the prompt uses the tail of the list as a scratch area (does not
    sweep the cache); host staging 2 x 64 records (244 MiB); prefetch of the next layer in a thread while the GPU works;
    cache contents saved (`.lru`) and reloaded at start (10.6 GB in ~6 s).
  - 35B, context 262K, chunk 4096 (p1, 2216 tokens, generating 256 tokens): **MTP 22.2 tok/s** (cache 79 %, hit 97.6 %),
    without MTP 19.3 tok/s (hit 98.6 %), TTFT 3.3–3.5 s (~640–680 tok/s; arcinfer 4.1–6.5 s). Full cache: MTP
    23.9 tok/s.
  - Memory limits: GPU budget − 1.4 GiB (1.13 GiB left → kernels fail: out of device memory) and RAM − 14 GiB.
  - Zero server: chat, tool calls, images + MTP together (context 262K, cache ~70 %) work; **still an occasional
    "device lost" crash on the second image** — ✅ fixed 2026-10-14 (zeKernelSetIndirectAccess for the
    XCACHE kernels; 5/5 server sequences pass).
- Phase 2 ✅ (original plan; results above) MoE engine on Level Zero + LRU cache + prefetch from the router; test 35B 262K, results must be identical. Also
  **with MTP**: generation target **~21 tok/s** at 262K (arcinfer currently 22.3 tok/s but only 40K); the MTP block ~0.46 GB
  lives in GPU memory too (expert cache ~80 %), the hit rate of verifying 2 tokens per step is measured as well.
- Prefill: **chunk 4096** (experts not in the cache are read from the SSD once per chunk; per token half of
  chunk 2048; buffer +~0.3 GB). Experts for the prompt are streamed through a separate scratch buffer, not into the LRU cache,
  so one long prompt does not sweep the cache that generation uses.
- Phase 3 ⏳ Qwen3.8 Flash Next (qwen4exp: QSA, n-gram table on SSD, 512 experts top-10).


## F12: complete server API (requested 2026-10-15, all stages; no --api-key)

All on the server side (`tools/server.cpp`); `zero/serverz.cpp` is generated from it automatically at build time so both
servers are always the same. Each stage is tested with real clients (Python openai / ollama / anthropic packages, curl) and
existing features are checked to stay the same (Pi, tool calls, images, MTP).

| Stage | Content | Status |
|---|---|---|
| 12.0 | serverz.cpp generated from server.cpp at build time | ✅ (code; overnight test) |
| 12.1 | `frequency_penalty`, `repeat_penalty`, `logit_bias`, `tool_choice` (none / auto / required / a specific function) | ✅ code (tool_choice through a forced tool-call opener); overnight test |
| 12.2 | `logprobs` + `top_logprobs`; `n` > 1 (prompt state snapshot) | ✅ code (n > 1 without streaming); overnight test |
| 12.3 | `/v1/completions` (raw prompt, code autocomplete) | ✅ code; overnight test |
| 12.4 | `response_format`: `json_object`, `json_schema` (constrained sampling, JSON always valid) | ✅ code + 26 CPU unit tests; overnight model test |
| 12.5 | Ollama API: `/api/chat`, `/api/generate`, `/api/tags`, `/api/show` (NDJSON) | ✅ code (+ /api/tags, /api/show, /api/ps, /api/version); overnight test |
| 12.6 | `/v1/messages` (Anthropic: tool_use, streaming events) | ✅ code (+ /v1/messages/count_tokens); overnight test |
| 12.7 | `/v1/responses` (OpenAI Responses API, Codex CLI) | ✅ code; overnight test |
| 12.8 | `/v1/audio/transcriptions` (multipart upload, Gemma audio encoder) | ✅ code (+ /v1/audio/translations); overnight test |
| 12.9 | `/v1/embeddings` with EmbeddingGemma 2 (`unsloth/embeddinggemma-2-GGUF`, BF16 558 MB; optional BF16 mmproj 982 MB for images/audio) + a document question-answering (RAG) feature option in the web UI | ⏳ tomorrow (2026-10-16), download with permission; user is downloading Qwen3.8 Flash Next |
| 12.10 | Launcher: base URL + copy button; help; README / USAGE.md | ✅ (launcher: API address per format + Copy; docs); overnight display test |

## F13: arcinferzero becomes the only engine (decision 2026-10-15)

Once zero passes everything, OpenCL is no longer used to run models (OpenCL.dll is still used only
to compile kernels).

| Step | Content | Status |
|---|---|---|
| 13.1 | (Gemma + drafter move too: approved 2026-10-15, drafter code unchanged) Validate all models on zero: 4B, 9B, 27B, 35B, Gemma 12B / E4B (bit-identical vs arcinfer, speed alternating 3x, MTP, Gemma drafter, images, audio, Pi, tool calls) | ✅ 2026-10-16 night: all pass (identical, MTP, images/audio, Pi, tool calls); see docs/arcinferzero-log.md |
| 13.2 | Zero becomes the default: `bin\server.exe` built from the zero engine, `serverz.exe` removed (likewise `runz.exe` → `run.exe`); OpenCL temporarily as `server-opencl.exe` (`--opencl`); `--zero` not needed | 🟡 2026-10-16: zero is the default in arc-server / pi-arc / launcher (`-OpenCL` / `--opencl` for the old engine); exes not renamed yet |
| 13.3 | Daily use for 1–2 weeks | ⏳ |
| 13.4 | Remove the OpenCL path (src/engine.h OpenCL, old server.exe / run.exe) | ⏳ |

## F14: parallel drafting and DFlash on Level Zero (after F13)

Requested 2026-10-15: stabilize Level Zero first (F13), then:
- Parallel drafting: the drafter (MTP / Gemma drafter) runs on a separate Level Zero queue, overlapping with
  main-model verification; first measure how much can overlap on the iGPU (one compute engine).
- DFlash (block drafter, e.g. z-lab/Qwen3.5-27B-DFlash): one step guesses a block of tokens; main target 27B
  (currently 5–7 tok/s). Download models only with permission.

F14 details (2026-10-15, from config.json + README of `z-lab/Qwen3.8-27B-DFlash2`, mirror `incoai/...`; weights not downloaded yet):
- DFlash2 drafter: 5 Qwen3-style layers (hidden 5120, 32 heads / 8 KV, head 128, FFN 17408, sliding window 2048,
  non-causal), input = hidden states of main-model layers 5 / 19 / 33 / 47 / 61, blocks of 8 tokens (7 guesses), top-16
  candidates per position + selector (rank 256), two-tap convolution (kernel 2, group 16); 3.85 GB BF16 (~1.9B),
  Q8 ~2 GB / Q4 ~1.1 GB. Apache-2.0.
- Official table (H200): tokens accepted per step MTP-7 vs DFlash2: GSM8K 5.02 / 5.46; HumanEval 3.91 / 4.39;
  MT-Bench 3.74 / 4.10. Qwen3.8 has MTP with 7 guesses; arcinfer currently at most 3.
- Order: (1) 27B MTP up to 7 guesses (long-block verification + DeltaNet state rollback without 7 full copies),
  (2) DFlash2 on top of that (download 3.85 GB with permission). Rough estimate for 27B: 10–12 tok/s if ~4 tokens are accepted.

## arcinferzero package (2026-10-15): MoE models without the original GGUF

- `serverz --pack <gguf>` (part 1 if split): `%LOCALAPPDATA%\arcinfer\zero\models\<name>\<name>.gguf` (non-expert
  weights + all metadata + experts as "virtual" tensors) and `<name>.experts` (expert file). The source may
  be on a flash drive; afterwards the original GGUF may be moved. The package shows up in the model list (serverz) and only runs
  with streaming.
- GGUF reader: split models (`-00001-of-0000N`) are read as one; metadata types are kept so they can be rewritten.
- Test: re-copying the Gemma drafter GGUF is identical (45 keys, 49 tensors). ⏳ overnight: 35B package (create, run without the
  original GGUF, results bit-identical to regular zero). Flash Next (DO NOT delete its GGUF: the user decides when, after backup): download (IQ3_S 83.6 GB, 2 parts) → back up to flash drive
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
| 19.4 | 35B chunk 8192 for contexts <= 32K | ⏳ | review 19.6: costs ~25-30K tokens of auto context |
| 19.5 | GEMM / moe_group autotune, 27B --budget | ✅ | the defaults are already the best: 27B pp4096 (RG 2/8, WGY 4/16, TM 32, split-K 9/34/off) and 35B pp8192 (MOE_G 32/128, RG, WGY) all the same or slower (27B 123-133 tok/s, 35B ~760). `ARC_GEMM_SPLIT` (superblocks per split-K piece) added for such sweeps |
| 19.6 | 35B: expert decode once per layer | ⏳ | large |

