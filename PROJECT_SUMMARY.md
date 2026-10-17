# arcinfer — project summary

A from-scratch LLM inference engine for the **Intel Arc integrated GPU** of a Core Ultra 5 125H laptop (Meteor Lake,
7 Xe-cores / 112 EUs, no XMX, 32 GB LPDDR5x), in C++ and OpenCL. It loads GGUF files directly, serves an
OpenAI-compatible API with a web UI, and plugs into the **Pi** coding agent with one command. Everything runs on the
iGPU; the design goal from the start was fast prompt processing (time to first token) for long, code-heavy prompts.

Since 2026-10-16 the default engine is **arcinferzero** (`zero/`): the same OpenCL C kernels run through Level Zero,
bit-identical to the OpenCL engine, with mixture-of-experts weights streamed from the SSD (Qwen3.6 35B-A3B: 262K
context with images and MTP), MTP up to 7 drafts and a few faster kernels. Speeds:
[docs/performance-zero-2026-10-16.md](docs/performance-zero-2026-10-16.md); log: [docs/arcinferzero-log.md](docs/arcinferzero-log.md).

Detailed history: [CHANGELOG.md](CHANGELOG.md). Results and the story of the build: [docs/review.md](docs/review.md),
working notes in [docs/review_notes.md](docs/review_notes.md). Usage guide: [USAGE.md](USAGE.md).

---

## 1. The engine

**Measured ceilings of the hardware** (bench/peak_bench, bw_bench): DP4a int8 11.2–13.3 TOPS, FP16 7.3 TFLOPS,
FP32 3.9 TFLOPS, GPU memory reads ~75–77 GB/s. Everything below is shaped around two facts: int8 DP4a is the fastest
arithmetic this iGPU has, and token generation can never be faster than reading the weights once per token.

| Area | What was built |
|---|---|
| Matrix products (prompts) | int8 DP4a GEMM (6–9 TOPS; llama.cpp's backends reach ~1.5 TOPS effective here). Every GGUF weight type is decoded "just in time" to int8 (W8) so one tuned GEMM serves all of them; types with block minimums are requantized to signed int8 (Q4_K 6.1 -> 8.8 TOPS). Split-K for long rows, work-group sharing of weight tiles, layer-major prefill (each matrix decoded once per prompt segment). |
| Matrix-vector products (generation) | GEMVs straight on the raw GGUF blocks (57–77 GB/s), small-batch variants for speculative verification, fused epilogues (SwiGLU, Gemma's logit soft-capping). |
| Attention | int8 everywhere: int8 KV cache in 16-key blocks, DP4a scores, online softmax in the log2 domain, int8 P·V with per-tile P quantization; split-head variants for head size 512; flash-decoding with adaptive key splits for generation; sliding windows and ring buffers. 27B, 11K tokens: attention 33 s -> 5.1 s. |
| Hybrid / special layers | Qwen3.5/3.8 Gated DeltaNet (recurrence 424 -> 145 ms), Gemma 4 sliding + global layers, per-layer embeddings (E4B), Qwen3.6-35B-A3B mixture of experts with ISTA-DASLab's 2-bit GSQ experts (own converter, `tools/convert_gsq.py`). |
| Speculative decoding | Qwen's built-in MTP block and the Gemma 4 assistant drafters; output identical to plain decoding. |
| Weight types | Q8_0, Q2_K–Q6_K, IQ1_M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS, F16/BF16/F32, GSQ 2-bit. |

**Accuracy** is checked against an independent f32 CPU reference (`tools/ref_forward`) and llama.cpp: KL 0.002 on
Qwen 4B, Gemma 4 greedy output equal to llama.cpp, tokenizer and chat templates identical. One finding along the way:
Gemma 4's V rows have a few large dims, so int8 scales per 64 dims (instead of per row) were needed (KL on code
0.23 -> 0.018).

**Milestone:** Qwen3.8 27B, 11,240-token code prompt: time to first token 116 s (first working version) -> ~89 s.

## 2. Models it runs

| Model | Notes |
|---|---|
| Qwen3.8 27B GSQ-RCO and Swift 1.5 (IQ3_S mix) | context 86,016 tokens at the current 18.3 GiB GPU share; MTP |
| Qwen3.5 4B / 9B (Unsloth UD-Q4_K_XL) | 262,144 tokens; MTP |
| Qwen3.6-35B-A3B (GSQ 2-bit, converted) | 11K prompt in ~14 s, ~15 tok/s |
| Gemma 4 12B QAT (Q4_K_M) | image input (CPU encoder), assistant drafter |
| Gemma 4 E4B (IQ4_XS) | assistant drafter; 131,072 tokens |

## 3. The server and Pi

- **OpenAI API + web UI** (`bin\server.exe`): streaming, reasoning content, tool calls streamed while they are
  generated (strings, arrays and objects — Pi's `edit` streams), `--ctx auto` sizing each model to the GPU memory,
  model switching on request (Pi's `/model`), `/v1/models` with per-model image capability.
- **Caching:** in-memory conversation reuse, system prompts saved on disk **per project folder** (a new Pi session
  on the 27B: 40 s -> ~4–5 s to the first token, also after a restart), optional session cache for switching between
  conversations.
- **Esc works:** a closed connection cancels the request within one layer (next request after an abort: 78 s ->
  4 s), and the conversation resumes from just before the aborted reply.
- **Pi integration:** `pi-arc` starts the server and Pi in one command, `arc-server` manages it, Pi's model list is
  synced from the models on disk at every start. All 17 of Pi's tools (files, bash, web, browser, MCP codemode)
  verified with streaming; a real-Pi use-case run (27 replies, 21 tool calls) without errors.
- **Setup:** `setup.cmd` checks requirements, builds, downloads models, connects Pi. Every option is documented in
  `--help`, README, and the Indonesian guide.

## 4. Recent work (10–12 October 2026)

| Date | Work | Result |
|---|---|---|
| 10 Oct | Gemma 4 generation tuning (head allocation, GPU soft-capping, RoPE table, fused norms, decode attention) | 12B 7.8 -> 8.1 tok/s; E4B global-layer decode attention 2.4x faster |
| 10 Oct | Prompts at 90 % of the context, streamed tool calls | 27B 81K-token prompt measured; tool arguments stream |
| 11 Oct | Session cache, one-command Pi, model switching, system prompt on disk, per-folder cache, Esc cancellation, setup command, help texts | see section 3 |
| 11 Oct | Pi's edit streaming, all Pi tools checked (boolean `True` fix), model list follows the disk | 5 failing browser tools fixed |
| 11 Oct | Image input published per model (`/v1/models` capabilities) | Gemma 4 12B shows as vision-capable in clients |
| 12 Oct | **pi-arc model mix-up fixed** (mixed `/` and `\` in stored paths made the server run Swift under an unknown name, so Pi opened Gemma E4B) | pi-arc opens the model it started |
| 12 Oct | **web_search** in Pi: DuckDuckGo is blocked on this network and Startpage returns nothing; Yandex enabled in Pi's settings | results in ~1 s |
| 12 Oct | **E4B generation**: new GEMV for IQ4_XS (decoding was as costly as reading), Q6_K output heads, fused FFN (gate + up + GELU + quantize) | **17.2 -> 21.2 tok/s**, with the drafter **~23 -> 30.7** (31 through the server); logits bitwise identical |
| 12 Oct | **E4B long prompts**: split-head attention for its head-512 global layers | 14K-token prompt 23.5 -> 21.8 s (591 -> 638 tok/s), perplexity unchanged |
| 12 Oct | **Images for Qwen3.5 4B** on the GPU (f16 encoder, M-RoPE), web UI copy/download and auto-scroll | see section 5 |
| 12 Oct | **Weight decoding** with vector loads; 9B keeps decoded weights resident | 27B decode 5.8 -> 2.2 s per 11K prompt; 9B short prompt 1.05 -> 0.27 s |

All regression tests pass (`tests\run_tests.ps1`: kernels, tokenizer, templates, KL vs reference, greedy vs
llama.cpp, MTP / drafter identity, API).

## 5. Images

- **Qwen3.5 4B** reads images, encoded on the GPU (`src/vision_qwen3vl.h`): f16 weights with f32 activations and
  attention (int8 drifted 10-23 % from the f32 reference on photos; now 0.2-0.6 %), M-RoPE positions in the language
  model. Works in Pi (read tool), the web UI (Media) and the API. A typical image encodes in ~0.4 s, 2048 tokens in
  ~5 s, 4096 in ~15 s (llama.cpp: 71 s). Images are scaled to 256-2048 tokens.
- **Gemma 4 12B** reads images on the CPU encoder. Not yet: Qwen3.5 9B (wider encoder), Gemma 4 E4B, the 27B models
  (no projector).

## 6. Room left (measured)

- **Generation:** E4B now spends ~10 ms per token above its 37 ms memory floor (small per-layer kernels ~5 ms,
  matrix kernels below 75 GB/s ~4 ms); fusing more could reach ~23–24 tok/s plain, ~33–35 with the drafter.
- **Prompts:** the GEMM is at its practical ceiling (~65 % of the measured DP4a peak; the experiments of 12 October
  did not move it). Long-context attention is the remaining lever (~15-20 % on 50K+ prompts).
