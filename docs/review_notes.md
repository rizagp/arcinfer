# arcinfer review — working notes (numbers are filled in from bench/results)

> These notes cover the OpenCL engine up to 15 October 2026. Since 16 October the default engine is arcinferzero
> (Level Zero, bit-identical output); see [arcinferzero-log.md](arcinferzero-log.md) and
> [performance-zero-2026-10-16.md](performance-zero-2026-10-16.md).

## Setup
- Laptop: Acer, Intel Core Ultra 5 125H (Meteor Lake), Arc iGPU 7 Xe-cores / 112 EUs @ 2.2 GHz, no XMX.
- Memory: 32 GB LPDDR5x-6400 (128-bit, 102 GB/s theoretical; 77 GB/s measured GPU block-read bandwidth).
- OS: Windows 11 Home 10.0.26200, Intel graphics driver 32.0.101.9033 (OpenCL 3.0 NEO, Level Zero 1.17).
- Measured iGPU ceilings (bench/peak_bench, bench/sustain_bench, bench/bw_bench):
  FP32 FMA 3.88 TFLOPS, FP16 FMA 7.32 TFLOPS, DP4a signed SIMD16 11.2–13.3 TOPS (sustained 13.0 for 2 min),
  DP4a signed x unsigned 9.1 TOPS, memory read 77 GB/s.
- The laptop throttles under sustained GPU load (5–8 % slower after several minutes); the benchmark scripts idle
  the GPU for 30 s before every measurement.

## Engines compared
- arcinfer (this project), OpenCL, int8 DP4a everywhere (GEMM, attention), JIT W8 decoding.
- llama.cpp b11527 official Windows build, Vulkan backend (`llama-b11527-bin-win-vulkan-x64.zip`).
- llama.cpp b11527 official Windows build, SYCL backend (`llama-b11527-bin-win-sycl-x64.zip`).
  Needs `GGML_SYCL_ENABLE_VMM=0` on this driver (otherwise the VMM pool fails with UR_RESULT_ERROR_INVALID_VALUE).
  Other SYCL options (FORCE_MMQ, ENABLE_DNN, ENABLE_GRAPH, ENABLE_FUSION, ENABLE_ESIMD) measured: no effect on
  pp2048 (314–318 t/s on the 4B) or tg; ENABLE_OPT=0 halves tg. Defaults kept.
- llama.cpp settings: `-ngl 99 -fa on -ub <best of 512/1024/2048 per model and backend> -b 2048`.

## Models
- Qwen3.5-4B UD-Q4_K_XL (Unsloth), 2.78 GiB.
- Qwen3.5-9B UD-Q4_K_XL (Unsloth), 5.71 GiB.
- Qwen3.8-27B GSQ-RCO IQ3_S-mtp (ISTA-DASLab), 11.28 GiB, per-tensor mixed IQ1_M..Q6_K
  (IQ3_S 30 %, IQ4_XS 25 %, IQ3_XXS 17 %, Q4_K 13 %, IQ2_S 7 %, rest small).
- Qwen3.6-35B-A3B 2-bit GSQ (ISTA-DASLab; safetensors, compressed-tensors pack-quantized), converted with
  tools/convert_gsq.py into a 10.92 GiB GGUF (arch qwen35moe): 2-bit routed experts {-2,-1,0,1} x bf16 scale per
  128 (8.2 GiB, exact codes), int8 attention / DeltaNet / shared experts -> Q8_0 (same int8 values, f16 scales,
  max rel. difference 2.7e-4), int8 per-channel lm_head -> Q8_0. llama.cpp cannot load this checkpoint; its closest
  GGUF is Unsloth's UD-IQ2_M (11.5 GB, IQ2_XXS gate/up and IQ3_XXS down experts with an imatrix), used only as a
  speed reference.
- Gemma 4 12B QAT Q4_K_M (community quantization) (+ mmproj BF16, + assistant drafter mtp-gemma-4-12B-it,
  423 M params, Q4_0) and Gemma 4 E4B "Aggressive" IQ4_XS (+ drafter, 4 heads, window 512).

## Workloads
- pp512 / pp2048 / tg64 (synthetic, llama-bench style).
- Time to first token on real "scan this codebase" prompts built from local projects:
  supreme-search 11,240 tokens; supreme-search 35,359 tokens; portal-evaluasi (an internal web app) 33,937 tokens
  (Gemma tokenizer: 13,921 and 35,957 tokens for the 11K and 34K texts).
- Token generation speed right after those prompts (long-context decoding).
- Perplexity: 40 x 512-token chunks of the llama.cpp docs (500 K chars), second half of each chunk scored.

## Accuracy notes
- Engine vs independent f32 CPU reference (tools/ref_forward), 4B, 20-token prompt, KL averaged over all 20
  positions: 0.0021 (int8 KV cache + int8 attention), 0.00056 with `--attn-f32` (f16 KV, f32 attention).
  A single-position KL is not a stable metric: a 1e-7 change in an intermediate value flips int8 roundings and moves
  it between 0.0001 and 0.004.
- Perplexity (4B, 8 x 2048 tokens of llama.cpp docs): 5.4749 f32 attention, 5.4759 int8 attention,
  5.4779 int8 attention + requantized Q4_K (current default). Differences < 0.06 %.
- Greedy continuations identical to llama.cpp on 4B and 9B test prompts (short prompts).
- Tokenizer identical to llama-tokenize on 11 K / 34 K / 35 K token prompts.
- Chat template rendering identical to the GGUF Jinja templates on 14 cases (Qwen3.5 + Qwen3.8 variants).

## Progress of the 27B 11 K-token time to first token (supreme-search prompt, 11,240 tokens)
| step | TTFT | main change |
|---|---|---|
| first working version | 116 s | DP4a GEMM, JIT W8, f32 attention |
| int8 scores, SLM tiles | 112 s | attention 33 -> 19.6 s |
| int8 P·V, int8 V cache, head split | 101 s | attention 19.6 -> 5.2 s |
| split-K + requantized Q4_K | 94 s | ffn_down 7.3 -> 8.8 TOPS, Q4_K 6.1 -> 8.8 TOPS, -0.85 GiB |
| layer-major prefill | 91 s | W8 decoding 6.9 -> 2.2 s |
| fused SwiGLU, vectorized conv | ~89 s (thermally limited runs ~91 s) | swiglu 2.5 -> 1.3 s, conv 2.0 -> 1.0 s |

Remaining profile (11 K tokens, GPU time ~89 s): GEMMs ~70 s (~8–9 TOPS, ~70 % of the DP4a peak; ablations show
the loop itself tops out at 10.6 TOPS without any memory traffic), attention 5.3 s, W8 decode 2.2–2.6 s,
DeltaNet recurrence 2.0 s, conv 1.0 s, SwiGLU quantization 1.3 s, norms 1.5 s. A < 60 s TTFT would need the
GEMMs at ~12 TOPS (> 90 % of peak), which this iGPU cannot sustain with DP4a.

## Optimizations that mattered (measured)
- int8 DP4a GEMM instead of fp16: ~6–9 TOPS vs llama.cpp's ~1.5 TOPS effective.
- W8 just-in-time decoding lets one GEMM kernel serve every IQ/K type; per-32 activation scales (KL 0.0014 -> 0.00008).
- Requantizing types with block minimums (Q2_K/Q4_K/Q5_K) to signed int8 W8: Q4_K GEMM 6.1 -> 8.8 TOPS; also lifts
  the 4B (mostly Q4_K) from 662 to ~900 t/s pp2048.
- Work-group sharing of weight tiles across 8 token tiles: 4.1 -> 6.7 TOPS (DRAM traffic).
- Split-K for K > 8192 (FFN down projection): 7.3 -> 8.8 TOPS; the activation slab of a token-row group then fits in L3.
- Chunk rows padded to 128: avoids a 40 % slow path on the last prompt chunk.
- Layer-major prefill for JIT models: each matrix decoded once per prompt instead of once per 4096-token chunk.
- Gated DeltaNet recurrence with a whole state column per SIMD8 lane: 424 ms -> 145 ms (4B, 2.6 K tokens).
- Attention (27B, 11 K tokens, 16 attention layers):
  - int8 scores with SLM key tiles: 33 s -> 19.6 s;
  - int8 P·V (P·vscale quantized to 8 bits per 16-key tile), int8 V cache in 16-key blocks, heads split across sub-groups (no duplicated QK, no register spills): 19.1 -> 7.4 s;
  - 16 tokens per work-group, double-buffered tiles: 7.4 -> 5.85 s;
  - lazy rescaling + log2-domain softmax + one P-scale bound per tile: 5.85 -> 5.1 s.
- Flash-decoding (split-K) for single-token attention at long context.
- GEMV on padded raw blocks with coalesced block reads into SLM + codebooks in SLM: 11–49 GB/s -> 57–77 GB/s.
- Fused SwiGLU in the up-projection GEMM epilogue: halves the f32 traffic of the SwiGLU/quantization pass.

## Things that did not work (kept as notes)
- SIMD8 W8 GEMM (despite 13.7 TOPS DP4a microbenchmark): slower in practice.
- f16 P·V accumulation: no speed-up, worse KL.
- SLM staging of q/k for the DeltaNet recurrence: SLM bandwidth bound, 4x slower.
- W8 kernel variant with per-block mins (Q4_K/Q5_K): 2.4 TOPS; replaced by requantization.
- GEMM work-groups spanning several 16-column groups (activation sharing in L1): no gain once the row grouping was tuned.
- GEMM ablations: removing lane broadcasts or the scale epilogue changes < 3 %; without any loads the loop reaches
  10.6 TOPS, i.e. the kernel is issue-bound, not memory-bound.
- 7-bit P (signed DP4a) instead of 8-bit P (unsigned x signed DP4a): same speed, 1.8x the error.
- Several tokens per sub-group in the DeltaNet conv (register sliding window): 3x slower than one token per
  sub-group with vector loads (too few sub-groups in flight).

## Qwen3.6-35B-A3B (mixture of experts, GSQ 2-bit)
- Format checks against the Unsloth GGUF of the same model (gguf-py dequantization): int8 tensors correlate 0.9993;
  2-bit codes LSB-first with offset 2 correlate 0.78–0.88 (both are 2–3-bit approximations of the same weights);
  MSB-first or offset 1.5 give ~0 / 0.65–0.75. llama.cpp's transforms (norm + 1, A = -exp(A_log), DeltaNet V heads
  grouped -> tiled) reproduce the Unsloth file's F32 tensors exactly. The converted expert layout decodes back to
  the checkpoint values with zero difference.
- Engine: router GEMM (+ shared-expert gate row), top-8 softmax renormalized, counting sort of (token, expert)
  pairs, tiles of 16 pairs of one expert, DP4a GEMMs on the 2-bit codes (decoded to int8 in registers, SwiGLU
  epilogue), routing weight on the down projection, combine with the sigmoid-gated shared expert; a single token
  routes straight into one-slot tiles.
- KL vs an f32 CPU reference that decodes the expert layout independently: 0.00077 (20 positions); answer
  "Jakarta" to the capital test prompt.
- Speed (preliminary, see results.jsonl): pp2048 925 t/s, 11K-token code prompt 14.1 s, decode ~15 t/s at 11K
  context (int8 attention/DeltaNet weights dominate: 1.5 GB of the ~2.3 GB read per token).
- Perplexity on the eval corpus 5.61 (arcinfer, GSQ) vs 5.17 (llama.cpp Vulkan, Unsloth UD-IQ2_M); the model card
  reports the GSQ checkpoint 2–4 points below the base model on GPQA / IFEval / MMLU-Pro.
- The checkpoint's MTP block has no routed experts, so this model runs without speculative decoding.

## Gemma 4 (12B and E4B)
- Text and vision verified against llama.cpp: greedy continuations identical (12B, E4B), top-10 probabilities equal
  to 4 decimals on chat prompts; images described correctly (1100-token 4K image encoded in 0.87 s on the CPU).
- Prefill attention at head size 512 (global layers, 16 q heads per kv head): the double-buffered K+V tiles
  (2 x 16 KB) plus the 256 B of scales in SLM let only one work-group run per Xe-core; reading the scales from L1
  instead: 1127 -> 649 ms per layer (11K tokens).
- Decode attention: the key split adapts to the range (>= 512 sub-groups; the 1024-key window of the sliding
  layers used only 32), the head slices of a split share a work-group (L1), one head per sub-group at head size 512,
  and attn_split processes several query rows per K/V tile (verification passes). Global layers at 13.9K context:
  10.7 -> 3.7 ms per token.
- lm_head GEMV (262144 x 3840 Q6_K, 15 superblocks per row): 2 rows per sub-group instead of 4: ~27 -> ~22 ms per
  token (noisy measurement).
- Assistant drafter (gemma4-assistant, as llama.cpp): reads the main model's KV cache of its last sliding and last
  global layer, same position for every chained draft; the input is [embedding(tok) * sqrt(E) | main hidden after the
  output norm]. Decode 6.3 -> 8.2–8.7 t/s at a 13.9K-token prompt (63–75 % of drafts accepted), 7.6 -> 11.3 t/s at
  short context; E4B 16.1 -> 21.4 t/s. Greedy output identical. A third draft per pass: +1.5 % only.
- Persistent kernel binary cache (%LOCALAPPDATA%\arcinfer\kernels): no compile stalls after the first use of a
  variant (a fresh variant costs 1–3 s, which had made the first speculative step look slow).
- Accuracy finding (f32 reference for Gemma 4 added to tools/ref_forward): with one int8 scale per q/K/V row,
  Gemma 4's next-token distributions drifted from the reference as the context grew (KL 0.08 over a 120-token raw
  text, a few positions flipping completely: KL 27 at one position of a 160-token code prompt). Simulating the
  quantization in the f32 reference located the cause: Gemma's V rows (RMS-normalized V / raw K projections without
  weight) contain a few large dims, so a per-row scale crushes the others. With one scale per 64 dims for q, K and V
  (median KL 0.0071 -> 0.0017 in simulation) the engine's KL on that text dropped to 0.035 and the flips vanished.
  Its cost is Gemma-only (Qwen keeps per-row scales, KL 0.0021 there): attention 1.5–1.8x slower for Gemma.
  The remaining engine error (KL ~0.018–0.03) matches the simulation of its int8 design (activations per 32: 0.009,
  attention per 64: 0.006, 8-bit P: 0.006); Gemma 4 is about 7x more sensitive to int8 activations than Qwen3.5.
- Gemma's raw-text perplexity is not a usable accuracy metric on this corpus: on 4 chunks the f32-precision CPU
  build of llama.cpp gives 130, Vulkan 120, SYCL 113, arcinfer 111 (per chunk the engines differ by up to ±20 %).

## Memory and context (16.5 GiB of GPU memory)
- Layer-major prefill now works in segments of 16,384 tokens (bounded residual-stream buffer instead of one sized
  for the whole context; each matrix decoded once per segment, <1 % of the prefill time).
- Largest contexts that fit (GPU memory incl. the server's two DeltaNet snapshots): 27B 49,152 (~16.0 GiB);
  35B-A3B 262,144 (14.8 GiB); 9B 262,144 (12.4 GiB); 4B 262,144 (11.9 GiB); Gemma 4 12B 262,144 (12.7 GiB);
  Gemma 4 E4B 131,072 (10.8 GiB, its trained maximum). Pi's maxTokens = the window (Pi clamps to what is left).

## Memory and context now (17.44 GiB, 2026-10-11)
- The GPU memory share is a driver setting the user changes (16.5 -> 18.34 -> 20 (system became unresponsive) ->
  17.44 GiB). `--ctx auto` (default) sizes each model at load: dry load (allocations added up, nothing uploaded,
  seconds), budget = device - 0.95 GiB (desktop, other apps) - 0.22 GiB (driver) - snapshots, multiple of 4096,
  cached in %LOCALAPPDATA%\arcinfer\ctx-auto.txt per file, chunk, MTP and device memory.
- At 17.44 GiB (dry probe = real load, 27B 15.89 GiB both): 27B 57,344 (15.89 GiB; base 13.57); 35B-A3B 262,144
  (14.50); 9B 262,144 (12.05); 4B 262,144 (11.64); Gemma 4 12B 262,144 (12.72); E4B 131,072 (10.75).
- Prompt chunk 4096 kept for every model (time to first token first; 1024-token chunks free 1.1 GiB on the 27B for
  ~8K more context at ~4 % slower 11K prompts).
- Later the same day the setting went back to 18.3 GiB: 27B 86,016 (16.84 GiB, load 28–33 s); the other models
  unchanged (trained maxima, same GPU memory). 118,784 with --chunk 1024.

## System prompts per project folder, one-command setup (2026-10-11, late)
- Pi writes the project folder into its system prompt (`<cwd>` section, pi-coding-agent core/system-prompt.js);
  the server reads it from the first message and files the saved state as
  `<folder>-<folder hash>-<prompt hash>-<tokens>.arcp`. Folder spelling normalized (case, `\` vs `/`). Per folder
  the 2 latest versions (two variants used in turn both stay: Pi run directly vs pi-arc gave 4,120 vs 4,146
  tokens in the arcinfer folder); folder-less prompts: the 8 latest; all models' files under --prefix-cache-gb
  (default 8). Before: 8 per model across all folders, so a 9th project evicted another's 40 s of work.
- Checked on a separate 4B server (own LOCALAPPDATA, kernel cache copied): grouping, 2 versions per folder,
  load after restart 0.07 s (60 MB, 584 tokens), a pre-change file adopted by rename, the size cap (LRU across
  models); real Pi: `folder_check_proj-f9a64d8e-...-524.arcp` for a 524-token no-tools prompt.
- Not merged into it: the session cache. A saved state can only be reused by a request that starts with exactly
  its tokens, so a folder's cache covers the system prompt its sessions share, not one session's conversation for
  another. Gap: the conversation held when the server stops is not saved (arc-server stop kills the process).
- setup.cmd: requirements (vswhere finds Visual Studio 2026 Build Tools; the component-ID query returns nothing
  there, so every installation is checked for vcvars64.bat), build when stale (log in build.log), clinfo checks,
  model downloads with hf_hub_download (HF_HUB_CACHE / HF_HOME as the server reads them), 35B conversion, Pi.
  Windows PowerShell 5.1 pitfalls met: a native program's redirected stderr is fatal under ErrorAction Stop;
  `@(@("x"), ...)` style candidate lists flatten; scripts must stay ASCII.

## Prompts at 90 % of the context (2026-10-10 evening; details in CHANGELOG.md)
- 27B 81,100 tokens: 16.4 min to the first token, 4.92 tok/s after it; E4B 117,965: 11.9 min; 4B 235,929: 27.3 min.
  Attention share at that depth: ~38 % (27B, 81K), ~half (4B, 236K).
- Gemma 4 12B global layers: `attn_ps` (head dims split over sub-groups, partial scores summed in SLM, no spills):
  1.25–1.4x at every length; Qwen keeps `attn_q8` (at its register limit; attn_ps bit-identical but slower).
- Qwen KV scale buffers were sized per 64 dims although Qwen uses one scale per row: 1,632 B/token less.

## Pi on the 27B (2026-10-11, real Pi, times from the server log)
- System prompt: ~4,100 tokens (tools, skills, cwd, AGENTS.md: one per project folder). First time 40.2 s;
  new session on a running server 3.9 s; after a server restart 5.3 s (state loaded from disk in 0.53 s, 290 MB);
  after /model to the 4B and back: switch 37.1 s, then 6.8 s (loaded in 1.19 s). Before the in-memory snapshot
  every new session took ~36 s; before the disk copy every restart / model switch paid the 40 s again.
- Session switching (session cache on): going back to a 6.6–7.5K-token session 55–64 s -> seconds.
- Esc (client closes the connection): before, prompt processing ran to its end (7K tokens: GPU busy 73 s more,
  the next request waited). Now: socket polled (WSAPoll + MSG_PEEK) after every layer and every token; cancelled
  0.5–0.9 s after the client left. The block in progress is rolled back (27B: DeltaNet states from a host copy
  taken at the block start, 157 MB, ~20 ms; Gemma: rewind in its rings when the block was <= ~8K tokens, else
  reset). Exactness: greedy output after a cancelled prompt identical to a never-cancelled run (27B, Gemma 12B).
- Pi drops aborted/failed replies (pi-ai transform-messages), so the next request diverges inside
  `<|im_start|>assistant`; the resume snapshot moved from after the header to its start: next message after an
  abort 21.5 s (cached 0) -> 5.0 s (cached 2,518 of 2,536). In real Pi: 5,594 of 5,620 cached, 5.5 s.
- Use-case run (scratch git project, Pi's tools + 2 extensions + local stdio MCP server, thinking low): parallel
  reads, edit, write with backslashes/quotes/braces/Unicode, 3,000-line output (Pi truncates; 104 s to process),
  failing commands, MCP via codemode, git commit, follow-up: 27 replies, 21 tool calls, 0 unparsed arguments,
  0 errored replies; 6 tool results flagged as errors were command failures the model handled (failing test,
  cp1252 console and an emoji, its own codemode script bug, a missing file on purpose, missing .gitignore, git
  without an identity). report.py output byte-exact; tests pass; README stock section from MCP data.
- Per step: 5–7 s prompt processing at 5–27K tokens of session; generation 5–6 tok/s dominates (a 770-token
  reply ~2.5 min). Floor: ~3 s for a handful of new tokens (15 tokens: 2.8–3.8 s), cause not investigated.
- Cold start: the first request after a load pays first-touch costs for the weights (a 15-token request after a
  cancelled first prompt: 27.5 s; warm: 2.8 s).

## Bugs found during the review work
- Embedding upload race: with >= 3 prompt chunks in one `forward()` call, the non-blocking upload of a chunk could
  read the host buffer after the next chunk had overwritten it (the server was not affected because each of its
  calls ended with a blocking read; `run.exe` on long prompts without `--prof` was). Fixed with a blocking upload.
- Chat template: tool-call arguments in the history were dropped (dangling reference to a temporary JSON object).
- kernels/ops.cl: the sub-group-size attribute of swiglu_quant ended up on the gelu_tanh helper added above it, so
  the program failed to build for every model after the Gemma GeGLU change.
- Server: Gemma 4 non-streaming responses had empty content (the parser's state only advanced while streaming).
- install-pi.ps1: a function parameter named `$input` (a PowerShell automatic variable) wrote
  `{"Current": null}` as every model's input types.
- arc-server stop: a server stopped via its state file was found again by the port query and stopped twice (error).
- attn_decode_rows could pick up to 32 query rows per launch while the partials buffer holds 4 (only reachable with
  more than 4 verification rows; capped).
- The W8 "scales + mins" path (exact Q4_K/Q5_K) is 3x slower in the GEMM and was not more accurate end to end; the
  requantized form stays the default (ARC_W8_MINS=1 for experiments).
- (2026-10-10/11) Gemma tool-argument parser looped forever on a stray `}` in an array; UTF-8 characters split over
  two tokens were streamed as two invalid pieces; the Engine never released its GPU buffers (the --ctx auto probe
  load stayed resident and made the real load page); E4B was advertised as image-capable without a supported
  projector; Pi model entries written with `input` as a string (Pi then rejected the provider).
- (2026-10-11) A cancelled prompt left the server processing it to the end; the resume snapshot sat two tokens past
  where Pi's request after an abort diverges (whole conversation reprocessed); the regression tests' server wrote
  into the user's system-prompt cache (now `--no-prefix-cache`).
- (2026-10-12) pi-arc opened Gemma 4 E4B while the server ran Swift: `last-model.txt` holds the path with mixed
  separators as `--list-models` writes it, arc-server compared it as text, found no model and started the file under
  a name made from the file name; that name was not in `/v1/models`, whose first entry (`gemma4-e4b`) pi-arc took
  as the running model. Server: paths normalized, model files compared as files, the loaded model always listed
  first; scripts: paths compared as full paths, the running model is the entry marked `loaded`.

## Gemma 4 E4B generation and long prompts (2026-10-12)
- One generated token (E4B IQ4_XS, short context, `run --prof-gen` on bench/prompts/gemma-code160): 56.5 ms of GPU
  time, 61 ms wall (16.4-17.1 tok/s). Bytes read per token: 2.74 GB (layers 2,135 MB, Q6_K output head 550 MB,
  BF16 per-layer projection 55 MB; the 2.3 GB per-layer embedding table only by rows) -> ~37 ms floor at 75 GB/s
  (bw_bench block reads 74-76 GB/s).
- GEMV per matrix (gemv_bench, weights streaming over all layers): FFN gate 64.7 GB/s, down 57.5, attention q 44.5,
  output 48.1, the 256-row per-layer-embedding matrices 21-24. Row-grouping sweeps (GROWS 1-8) did not help.
- Ablations of `gemv` on the FFN matrices: as is 0.225 / 0.261 ms; without the IQ4_XS decoding 0.202 / 0.200 ms
  (~70 GB/s); without the memory loads 0.168 / 0.183 ms; without both 0.039 / 0.059. Decoding (byte-wise rd32 from
  the local-memory window, 32 table lookups and shift / or packing per 32 weights, ~8 instructions per weight) was
  as expensive as reading, and the two overlapped only partly.
- `gemv_v` (lane-wise vector loads from global memory, no window, no barriers; four pre-shifted codebook tables, a
  word = four lookups ORed): gate 70.1, down 69.2, q 67.2, output 69.4 GB/s; small matrices 17 -> 11 us. Q6_K output
  heads 66-67 -> 72 GB/s (E4B, Qwen3.5 4B, Gemma 4 12B). Q4_K / Q5_K were slower with it (Qwen 4B gate 72.3 ->
  59.8: every lane reloads the shared 16-byte header, and their decoding was cheap), so they stay on `gemv`.
  Bitwise identical to `gemv` on every layer's matrix (gemv_bench `GB_CMP`).
- `gemv_glu`: gate and up rows of a 32-value block in one work-group, gelu_tanh(gate) * up quantized in the epilogue
  into a second activation buffer (other work-groups still read the input); replaces three launches per layer.
- Result (same session, 128 tokens): 17.2 -> 21.2 tok/s plain, with the drafter (`run --mtp`, 67 of 115 drafts,
  2.10 tokens per pass) ~23 -> 30.7; through the server at temperature 0.7: 30.9 tok/s. GPU time per token 56.5 ->
  46.6 ms. Logits of 64 generation steps bitwise identical (`run --dump-logits`, old path via ARC_NO_GEMV_V=1
  ARC_NO_GLU=1).
- Left at short context (46.6 ms GPU, 48.5 ms wall vs the 37 ms floor): add_norm_quant 126 launches x 16 us,
  attn_prep 44 x 27 us (scattered byte writes into the V cache blocks), per-layer-embedding kernels ~2.3 ms, the
  q / o GEMVs at ~53 GB/s in the engine (67-69 in isolation: launch ramp between small kernels). Vector loads in
  attn_prep_g / add_norm_quant: only ~5 %.
- Long prompts: 13,921-token code prompt 23.8 s, of which global attention (7 layers, head 512, 4 q heads per kv
  head) 6.2 s at ~11 % of the int8 peak: attn_q8 spilled 2 KB of registers. attn_bench sweep (ATT_NH=8 ATT_NKV=2
  ATT_HD=512): attn_q8 best 844-870 ms per layer, attn_ps PNS 4 / PHG 4 / PTT 1 / TPW 8 619 ms, no spills; 512
  work-items is the device maximum. Same error against the CPU reference (rel. rms 0.00379), not bit-identical
  (32.8 M of 57 M values differ, max 0.032 of 24.3). End to end: 23.5 -> 21.8 s (591 -> 638 t/s); perplexity over
  6 x 4,096 tokens 88.44 -> 88.34; the greedy continuation of that prompt diverges at token 13 (near-tie).
- pp2048: E4B 789 t/s, Qwen3.5 4B 880 t/s; E4B does ~27 % more arithmetic per token (wider FFN, head-512 global
  attention), so per operation they are on par. The ~350 t/s seen in Pi most likely came from sessions that
  shared the GPU with other work and from short follow-up turns, where fixed costs dominate the rate (not re-measured
  in Pi).
- Regression tests (Qwen3.5 4B, Gemma 4 12B): all pass, Gemma KL 0.0188.

## Qwen3.5 images on the GPU, weight decoding, 9B resident (2026-10-12, later)
- Qwen3-VL encoder (mmproj qwen3vl_merger, Qwen3.5 4B): with int8 GEMMs 10-23 % relative error vs an f32 CPU
  reference on photos (synthetic image 4 %). CPU emulation per rounding (vision_test VT_EMU): int8 activations of
  patch / attention-out / ffn up / ffn down 6.7-7.9 % each, qkv 2.1 %, merger 0.5 %, Q8_0 weights 3.7 %; f16 weights
  and inputs 0.6-1.8 %. GPU: f16 weights + f32 activations (vit_gemm_f16, 2.4 TFLOPS; 128 registers spill: 2.2x
  slower) with int8 attention 7.6-9.5 %, with f32 attention 0.2-0.5 %.
- f32 attention: one query row per lane (vit_attn_rows), 4-key tiles, native_exp: 1008 tokens 1118 -> 588 ms; 4096
  tokens 32.4 -> 10.4 s (whole encoder 36.1 -> 15.3 s; llama.cpp Vulkan 71 s for the same image). Tried and not
  kept: f16 (half2) GEMM (no faster, error 0.8-2.4 %), larger key tiles (spills).
- M-RoPE: sections 11/11/10 interleaved; image tokens (t, h, w) = start + shift + (0, row, column); text after an
  image at + max(gw, gh) (HF get_rope_index). Text output bitwise unchanged (KL identical to 5 decimals).
- Reading tests: "HELLO 42" read at 273+ tokens; a 6750 x 6750 logo "JB Logam Mulia" misread as "Logan" at 1024
  tokens by both arcinfer and llama.cpp, read right at 2048 and 4096 -> default 256..2048 tokens per image.
- Unsloth Studio's model picker shows no vision mark for any connected local server; image upload works.
- decode_w8 per superblock with vector loads (it read global memory byte by byte): 27B 11K prompt IQ3_S / IQ4_XS /
  IQ3_XXS decoding 5.8 -> 2.2 s, prompt logits bitwise identical.
- GEMM in isolation (bench/gemm_w8_bench, 27B shapes): W8 per-32 8.3-8.6 TOPS, per-16 7.1; no loads 9.7, no loads
  and no scale epilogue 11.0; signed DP4a with a broadcast operand peaks at 13.2. Pipelined loads, 32 tokens per
  sub-group and 24-bit multiplies did not help. In the model ~6.8-7.7 TOPS (heat, per-16 types, kernels between).
- Qwen3.5 9B: GEMMs 76 % of an 11K prompt at ~7.7 TOPS; it decoded its weights on every request (fixed 9 GB budget).
  --ctx auto now keeps them resident when >= 128K context still fits: 83-token prompt 1.05 -> 0.27 s, context
  262,144 -> 217,088.
- Benchmark cells left "not run" (llama.cpp SYCL on Gemma 4 12B, Gemma Vulkan perplexity, UD-IQ2_M for the 35B):
  llama.cpp's prompt processing on this iGPU made those runs hours long, and they would not have changed the picture.
