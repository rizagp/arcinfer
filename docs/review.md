# arcinfer review: how the speed developed

arcinfer is a from-scratch LLM inference engine for Intel's integrated Arc graphics, built on one laptop to shorten
the wait before a model starts answering a long prompt. It runs Qwen3.5, Qwen3.6 and Qwen3.8 (dense and mixture of
experts) and Gemma 4 with image and audio input, serves an OpenAI-, Anthropic- and Ollama-compatible API, and plugs
into the Pi coding agent. This page tells the story of the build from the first working forward pass to
17 October 2026, with the emphasis on how prompt and generation speed developed and what caused each step.

Two engines share the kernels and the server code: **arcinfer** (`src/`, OpenCL), which the work up to 15 October was
done on, and **arcinferzero** (`zero/`), which runs the same kernels through Level Zero, is bit-identical to the
OpenCL engine on every model, streams mixture-of-experts weights from the SSD, and has been the default since
16 October.

Sources: [CHANGELOG.md](../CHANGELOG.md), [review_notes.md](review_notes.md) (working notes),
[arcinferzero-log.md](arcinferzero-log.md), [speed-progress.md](speed-progress.md) (the speed tables by date),
[lessons-intel-arc-igpu.md](lessons-intel-arc-igpu.md), [QUEUE.md](../QUEUE.md),
the README, `bench/results/results.jsonl` and the git log. Every figure below comes from one of them.

## The hardware

| | |
|---|---|
| Laptop | Acer, Intel Core Ultra 5 125H (Meteor Lake) |
| GPU | Arc iGPU, Xe-LPG, 7 Xe-cores / 112 EUs at 2.2 GHz; **no XMX** matrix engines (no DPAS, no 2D block IO) |
| Memory | 32 GB LPDDR5x-6400 shared with the CPU: 102 GB/s theoretical, 75–77 GB/s measured GPU reads |
| GPU memory share | a driver setting (Intel Graphics Software): 16.5 GiB during the 10 October benchmark, later 17.44, 18.3 GiB |
| OS, driver | Windows 11 Home 10.0.26200, Intel graphics driver 32.0.101.9033 (OpenCL 3.0 NEO, Level Zero 1.17) |

Measured ceilings (`bench/peak_bench`, `bench/sustain_bench`, `bench/bw_bench`): int8 DP4a 11.2–13.3 TOPS (13.0
sustained for 2 minutes), FP16 FMA 7.32 TFLOPS, FP32 FMA 3.88 TFLOPS, memory reads 77 GB/s. The laptop throttles
under sustained load (5–8 % slower after several minutes), so the benchmarks idle the GPU for 30 s first and take the
best of repeated runs.

Two facts shaped everything. Prompt processing is compute-bound, and the fastest arithmetic this GPU has is the int8
DP4a instruction, so every prompt matrix product is built on it. Generation is memory-bound: a token can never be
produced faster than the weights can be read once, about 75 GB/s divided by the bytes read per token (for the
~11 GB Qwen3.8 27B, about 6.5 tok/s without speculative decoding).

## Why another engine

Coding agents such as Pi send long prompts: the system prompt, tool definitions and whole source files. On this
laptop, llama.cpp (b11527, Vulkan and SYCL) processed a 27B model's prompt at 20 to 40 tokens per second, so an
11,240-token code prompt took 504–590 s before the first word appeared. The yardstick through most of the build was
the time to first token of Qwen3.8 27B on that prompt.

## Timeline

The git repository starts on 16 October 2026 with a single import commit (`54b9f26`); the earlier dates come from
CHANGELOG.md and the working notes. The first ten stages are recorded only as "before 10 October".

| Date | Phase | Main results |
|---|---|---|
| before 10 Oct | 1. Qwen3.5 forward pass on the iGPU | GGUF loading, DP4a GEMM, every weight type decoded just in time (JIT) into one int8 "W8" form, f32 attention, the Gated DeltaNet recurrence. 27B, 11K prompt: 116 s |
| before 10 Oct | 2. int8 attention | DP4a scores over key tiles in local memory, int8 P·V with P quantized per 16-key tile, int8 KV cache, heads split across sub-groups. Attention 33 → 5.2 s, TTFT 116 → 101 s |
| before 10 Oct | 3. GEMM and pipeline | split-K, Q4_K requantized to signed int8, weight tiles shared across token tiles, layer-major prefill, SwiGLU in the GEMM epilogue. 101 → ~89 s; GEMMs at 8–9 TOPS |
| before 10 Oct | 4. Server for agents | OpenAI-compatible API (streaming, reasoning, tool calls), prefix reuse, web UI, `pi-arc` / `arc-server` |
| before 10 Oct | 5. MTP speculative decoding | drafts from the Qwen GGUFs' own multi-token-prediction block. 27B generation 4.1 → 5.3–5.8 tok/s |
| before 10 Oct | 6. Gemma 4 | sliding-window and global layers, shared KV layers, per-layer embeddings (E4B), Jinja templates through minja, 12B image input |
| before 10 Oct | 7. Qwen3.6 35B-A3B, GSQ 2-bit | converter (`tools/convert_gsq.py`) and MoE kernels on the 2-bit codes. 11K prompt ~14 s, ~15 tok/s |
| before 10 Oct | 8–10. Gemma 4 at long context, accuracy audit, memory | head-512 attention, assistant drafters, persistent kernel cache; one V scale per 64 dims for Gemma; prefill in 16K segments, largest context that fits for every model |
| 10 Oct | Benchmark against llama.cpp; Gemma 4 generation | see the comparison section; output head allocated first, decode-attention kernel `attn_dsplit`, prompts at 90 % of the context |
| 11 Oct | Working in Pi all day | session cache, system prompt kept on disk per project folder, Esc cancels within a layer, `setup.cmd` |
| 12 Oct | Bandwidth work, images | GEMV raw blocks regrouped for vector loads, E4B 17.2 → 21.2 tok/s, JIT decoding 2.7x faster, Google's QAT Q4_0 Gemma files, Qwen3.5 images on the GPU, 35B from standard GGUFs |
| 13 Oct | Qwen3.6 35B | experts through the int8 GEMM (prompt 476 → 642 tok/s), MTP for the MoE (19.7 → 22.3 tok/s), Q8_0 stored once (context 61K → 209K) |
| 14 Oct | arcinferzero | OpenCL kernels on Level Zero (35B TTFT 6.45 → 4.07 s), experts streamed from the SSD: 35B at 262K context with MTP |
| 15 Oct | Full API; overnight Level Zero tests | Anthropic / Ollama / Responses formats, structured output, constrained tool calls; Level Zero baseline for every model, MTP up to 7 drafts |
| 16 Oct | arcinferzero default; Flash Next; DFlash2 | register-spill fixes, GEMM with 16 tokens side by side (27B prompt 2K 110 → 130 tok/s), Qwen3.8 Flash Next running, DFlash2 block drafter (27B code 4.85 → 11.5 tok/s), token-outer verification GEMVs |
| 17 Oct | Speculative decoding and MoE prompts | more drafts by default (27B 7, 35B 5), DFlash2 drafter in Q4_0, `moe_rg`, Flash Next 7.3 tok/s and 8K prompt 165 tok/s, kernel warm-up |

## Speed now (17 October, arcinferzero)

Speculative decoding through the server, greedy, answers identical to plain decoding (README):

| Model, text | plain | MTP alone | DFlash2 + MTP |
|---|---:|---:|---:|
| Qwen3.8 27B, Python code | 4.91 tok/s | 15.5 | 18.1 |
| Qwen3.8 27B, Indonesian prose | 4.88 | 7.0 | 6.8 |
| Qwen3.6 35B-A3B, Python code | 21.2 | 39.3 | 49.8 (99 % of drafts accepted) |
| Qwen3.6 35B-A3B, Indonesian prose | 21.0 | 22.7 | 22.7 |

| Model | Prompt | Generation |
|---|---|---|
| Qwen3.8 27B (IQ3_S GSQ) | pp4096 ~135 tok/s, pp16384 ~130 | 4.9 tok/s (207 ms per token; 231 ms at 32K context) |
| Qwen3.6 35B-A3B (UD-Q3_K_M) | pp4096 ~795 tok/s, pp8192 ~757 | ~21 tok/s |
| Qwen3.8 Flash Next (IQ3_S GSQ, experts streamed from the SSD) | pp8192 ~165 tok/s | 7.3 tok/s |

The other models' latest numbers are in their tables below (Level Zero, 15–16 October). All speed tables by date:
[speed-progress.md](speed-progress.md).

## Model by model

Each table lists the milestones the sources record, oldest first. "Prompt" is prompt processing (tok/s, or time to
first token where that was measured); "Generation" is tok/s. Numbers from different milestones were measured on
different prompts and lengths, so compare within a row or along one kind of measurement.

### Qwen3.8 27B (ISTA-DASLab GSQ-RCO IQ3_S-mtp, 11.28 GiB, mixed IQ1_M–Q6_K)

The 11,240-token code prompt, time to first token, through the OpenCL stages:

| Step | TTFT | Main change |
|---|---:|---|
| first working version | 116 s | DP4a GEMM, JIT W8 decoding, f32 attention |
| int8 scores, local-memory key tiles | 112 s | attention 33 → 19.6 s |
| int8 P·V, int8 V cache, head split | 101 s | attention 19.6 → 5.2 s |
| split-K, requantized Q4_K | 94 s | ffn_down 7.3 → 8.8 TOPS, Q4_K 6.1 → 8.8 TOPS, −0.85 GiB |
| layer-major prefill | 91 s | W8 decoding 6.9 → 2.2 s (each matrix decoded once per prompt) |
| fused SwiGLU, vectorized conv | ~89 s | SwiGLU 2.5 → 1.3 s, conv 2.0 → 1.0 s (thermally limited runs ~91 s) |

The remaining profile at that point (~89 s of GPU time): GEMMs ~70 s at 8–9 TOPS (~70 % of the DP4a peak), attention
5.3 s, W8 decoding 2.2–2.6 s, DeltaNet recurrence 2.0 s, norms 1.5 s, SwiGLU quantization 1.3 s, conv 1.0 s.

All milestones:

| Date | Prompt | Generation | Main change |
|---|---|---|---|
| before 10 Oct | 11K prompt 116 → ~89 s | 4.1, with MTP 5.3–5.8 | the stages above; MTP head |
| 10 Oct (benchmark) | 11K 97.1 s, 34K 326.6 s; pp2048 113 | tg64 4.1–4.2; 4.0 after the 11K prompt, 3.7 after 34K | — |
| 10 Oct (evening) | 81,100-token prompt in 16.4 min (82.5 tok/s); 11K 90.1 s | 4.92 with MTP after the 81K prompt | Qwen KV cache drops an unused scale buffer (1,632 B per token) |
| 12 Oct | JIT decoding per 11K prompt 5.8 → 2.2 s | 4.3 → 4.7 (65 → 72 % of the bandwidth) | `decode_w8_sb` vector loads; raw blocks regrouped for aligned vector loads |
| 13 Oct | — | 4.75 → 4.8–5.0; MTP on code 4.5 → 7.9 (2 drafts, 91 % accepted) | DeltaNet qkv + z in one GEMV |
| 13 Oct | — | MTP on code 7.8 → 9.5 (+21 %) | 3 MTP drafts (the OpenCL engine allowed 2) |
| 15 Oct (Level Zero) | 2K 110, 16K 103, 65K 80 tok/s | 5.1 / 4.7 / 4.0; server 5.0, MTP 9.1 (code) | arcinferzero, bit-identical to OpenCL |
| 16 Oct | — | 5 drafts 3.8 → 9.2 | verification GEMVs load each token's activations where used (no register spill) |
| 16 Oct | 2K 110 → 130, 16K 103 → 125 tok/s | — | GEMM: activation words of 16 tokens loaded first, their DP4a chains side by side (gate 8.05 → 8.30 TOPS, down 6.62 → 7.0) |
| 16 Oct | attention, last 4K chunk at 64K: 1,515 → 1,335 ms per layer | — | P·V divisions once per sub-group, DP4a accumulating in place |
| 16 Oct (night) | — | DFlash2: code 4.85 → 11.5 (89 % of drafts, 6.9 tokens per pass); Indonesian 5.6 with the MTP fallback | DFlash2 block drafter |
| 16 Oct (night) | — | MTP code 8.9 → 12.7, Indonesian 6.3 → 7.3; DFlash2 code 11.5 → 16.2 | token-outer multi-token GEMVs (8-row FFN 23 → 39–43 GB/s) |
| 16 Oct (night) | — | MTP code 12.6 → 15.2 | 7 MTP drafts by default (was 3) |
| 17 Oct | — | DFlash2 code 16.9 → 17.2 → 18.3; Indonesian hybrid 6.2 → 6.5; MTP Indonesian 6.95 → 7.2 | drafter in Q4_0, one-pass top-16 selector, fewer DFlash2 probes on dense models, 4 rows per sub-group in 2–5-token GEMVs |
| 17 Oct (now) | pp4096 ~135 tok/s | plain 4.91; code MTP 15.5, DFlash2 + MTP 18.1; Indonesian 7.0 / 6.8 | — |

Bandwidth share during generation: the first measurement with the new byte counter was 36 % (QUEUE item 2, taken
while a model download ran); the 12 October regrouping took it from 65 to 72 %. On 16 October the Level Zero
per-kernel profile put the gap in decoding: IQ4_XS / Q4_K matrices read at ≈62 GB/s, IQ3_S / IQ3_XXS at 51–55,
IQ2 at 47–52, IQ1_M at ≈33, and the 27B's FFN is mostly IQ3 (≈5 GB per token); ≈940 kernels per token leave ~4 %
of gaps between them.

Context: 40,960 tokens before the segmented prefill, 49,152 at 16.5 GiB, 57,344 at 17.44 GiB, 86,016 at 18.3 GiB
with 4,096-token chunks (118,784 with 1,024-token chunks, ~4 % slower on an 11K prompt).

### Qwen3.6 35B-A3B (mixture of experts: 256 experts, 8 active)

The first version ran ISTA-DASLab's GSQ checkpoint (2-bit routed experts, {−2, −1, 0, 1} times a scale per 128,
converted into a 10.92 GiB GGUF). From 12 October it also runs Unsloth's standard GGUFs (UD-Q3_K_S, UD-Q3_K_M and
the file of Unsloth's MTP repository, which adds the MTP block; the GSQ checkpoint's MTP block has no routed experts).
Only about 3 B of the 35 B parameters are active per token, so it generates about four times as fast as the 27B, but
in a 2,048-token chunk each expert sees only ~64 tokens, which is hard on the GEMM.

| Date | Prompt | Generation | Main change |
|---|---|---|---|
| before 10 Oct | 11K prompt ~14 s; pp2048 925 (preliminary) | ~15 at 11K context | GSQ converter; routing, counting sort of (token, expert) pairs, DP4a GEMMs on the 2-bit codes |
| 10 Oct (benchmark, GSQ) | 11K 14.3 s, 34K 60.6 s; pp2048 908.6 | tg64 15.7; 15.3 after the 11K prompt | — |
| 12 Oct (UD-Q3_K_S) | pp2048 473 (150 with 8-slot tiles) | 17–18 | expert kernels on the raw rows of standard GGUF types |
| 13 Oct | — | 17.4–18.5 → 18.2–18.8 | router and shared expert in one GEMV |
| 13 Oct (UD-Q3_K_M) | pp2048 476 → 642 | — | experts through the int8 GEMM, rows decoded via local memory (`decode_w8_sl`: IQ3_XXS 1,038 → 518 ms per chunk) |
| 13 Oct | — | 19.2 → 19.8 | DeltaNet qkv + z in one GEMV |
| 13 Oct (MTP file) | prompts −3.4 % | 19.7 → 22.3 with MTP (83 % accepted) | MoE in the MTP block, Q3_K on the GPU |
| 13 Oct | 21,590-token prompt: attention 9.4 → 8.4 s; whole prompt 44 s (493 tok/s) | — | 4 q heads per sub-group in prompt attention |
| 14 Oct (Level Zero) | TTFT 6.45 → 4.07 s | 16.9 → 18.3 | the OpenCL kernels on Level Zero |
| 14 Oct (streaming) | 2,216-token prompt 3.3–3.5 s at 262K context | 19.3, MTP 22.2 (cache 79 %, 97.6 % hits) | experts streamed from the SSD into a GPU LRU cache |
| 15 Oct (baseline) | 2K 660, 16K 679, 131K 254, 262K 146 tok/s | 20.1 / 18.8 / 13.8 / 10.1; server 21.0, MTP 26.8 | — |
| 16 Oct | — | decode attention at 64K 1.16 → 0.94 ms per layer; MTP mean 23.9 → 26.1 | `attn_dsplit` for the 35B shape; 3 MTP drafts (was 1) |
| 16 Oct | 651–654 tok/s | 19.9 | expert-cache pools grouped by record size (all experts cached again) |
| 16 Oct (night) | — | 19.9 → 21.5; MTP code 31.5 → 34.7, Indonesian 22.3 → 24.3; DFlash2 code 44.4 → 46.0 | token-outer verification GEMVs |
| 16 Oct (night) | — | MTP code 34.2 → 37.7 | 5 MTP drafts by default |
| 17 Oct | 8K prompt 620 → 750 tok/s | — | the server reports the engine's 4,096-token chunks (was 2,048) |
| 17 Oct | — | MTP 37.5 → 39.1; DFlash2 48.4 → 52.7, 49.4 → 53.3 | one-pass top-16 selector; vectorized float exp for the drafts' confidence |
| 17 Oct | 384–1,152-token prompts +3–14 % (768 tokens 449 → 511) | — | `moe_rg`: the MoE prompt GEMM on raw expert rows, bit-identical to decode + GEMM |
| 17 Oct (now) | ~795 tok/s at 4,096-token chunks | plain 21.2; code MTP 39.3, DFlash2 + MTP 49.8; Indonesian 22.7 / 22.7 | — |

Context with MTP: 40,960 tokens on the OpenCL engine (4,096 with images as well); 262,144 with images and MTP on
arcinferzero, with Windows keeping 14 GB of RAM. The expert file (15.1 GB) is written once in 144 s; the cache's
contents are saved and loaded again at start (10.6 GB in ~6 s).

### Qwen3.5 4B (Unsloth UD-Q4_K_XL, 2.78 GiB)

| Date | Prompt | Generation | Main change |
|---|---|---|---|
| before 10 Oct | pp2048 662 → ~900; DeltaNet recurrence 424 → 145 ms (2.6K tokens) | — | Q4_K requantized to signed int8 (the 4B is mostly Q4_K); a 128-float state column per SIMD8 lane |
| 10 Oct (benchmark) | pp2048 911; 11K 14.3 s, 34K 57.5 s | tg64 17.1–17.6; 16.1 after the 11K prompt | — |
| 10 Oct (evening) | 235,929-token prompt in 27.3 min (144 tok/s) | 9.0 with MTP after it | — |
| 12 Oct | pp2048 880 | bandwidth 77 → 79 %; Q6_K output head 66–67 → 72 GB/s | raw blocks regrouped; `gemv_v` for Q6_K heads |
| 15 Oct (Level Zero) | 2K 862, 16K 697, 131K 281 tok/s | 20.2 / 18.7 / 9.5; server 20.6, MTP 29.2 | — |
| 16 Oct | prompt attention, last 4K chunk at 64K: 937 → 775 ms per layer (4B / 9B / 35B shape) | MTP mean 22.9 → 24.8; server plain 20.5, MTP code 30.4, Indonesian 25.9 | 3 MTP drafts (was 2); P·V in place |

### Qwen3.5 9B (Unsloth UD-Q4_K_XL, 5.71 GiB)

| Date | Prompt | Generation | Main change |
|---|---|---|---|
| 10 Oct (benchmark) | pp2048 402; 11K 26.8 s, 34K 96.0 s | tg64 10.3; 10.0 after the 11K prompt | — |
| 12 Oct | 83-token prompt 1.05 → 0.27 s to the first token; 11K 27.4 → 26.6 s | bandwidth 83 → 86 % | decoded weights kept resident when ≥ 128K context still fits (it decoded every matrix on every request) |
| 13 Oct | 21,590-token prompt: attention 8.2 → 7.0 s | — | 4 q heads per sub-group |
| 15 Oct (Level Zero) | 2K 399, 16K 388, 131K 222 tok/s | 12.1 / 11.5 / 8.8; server 12.2, MTP 17.7 | — |
| 16 Oct | — | server plain 12.1, MTP (2 drafts) code 16.7, Indonesian 15.0 | — |

The 9B's 11K prompt profile (12 October): GEMMs 76 % of the time at ~7.7 TOPS, the kernel's practical ceiling.

### Gemma 4 12B

The first measurements used a community Q4_K_M quantization; from 12 October Google's QAT Q4_0 file.

| Date | Prompt | Generation | Main change |
|---|---|---|---|
| before 10 Oct | prefill attention at head size 512: 1,127 → 649 ms per layer (11K) | drafter: 7.6 → 11.3 at short context, 6.3 → 8.2–8.7 after a 13.9K prompt | scales read from L1 instead of local memory; assistant drafter |
| 10 Oct (benchmark, Q4_K_M) | pp2048 240; 13,921-token prompt 69.1 s, 35,957 tokens 239.9 s | tg64 7.5; 6.4 after the 13.9K prompt | — |
| 10 Oct | — | 7.79–7.83 → 8.09–8.16; drafter 8.74 (50 % accepted); 134 → 123 ms GPU time per token | output head allocated first (it was read at half speed), soft-capping on the GPU, RoPE table, fused norms, `attn_dsplit` (global layer at 14K 742 → 310 µs) |
| 10 Oct (evening) | global-layer prompt attention 1.25–1.4x faster | — | `attn_ps`: head dims split over sub-groups, no register spill |
| 12 Oct (Google QAT Q4_0) | — | 9.1 (85 % of the bandwidth), drafter 17–20 | Q4_0 regrouped for vector loads |
| 15 Oct (Level Zero) | 2K 271, 16K 218, 131K 86 tok/s | 9.5 / 9.2 / 7.4; server 9.8, drafter 21.1 | — |
| 16 Oct | 2K 271 → 289, 16K 218 → 228 | 9.55 / 9.24 | register spills removed (`attn_prep_g`, `add_norm_quant`) |

### Gemma 4 E4B

Community IQ4_XS file first, Google's QAT Q4_0 from 12 October.

| Date | Prompt | Generation | Main change |
|---|---|---|---|
| before 10 Oct | — | drafter 16.1 → 21.4 | assistant drafter |
| 10 Oct | global decode attention at 14K 1,008 → 420 µs per layer | 16.9; drafter 17.4–18.1 on a story prompt through the server | Gemma decode work (see the 12B) |
| 10 Oct (evening) | 117,965-token prompt in 11.9 min (165.5 tok/s) | 11.6 with the drafter after it | — |
| 12 Oct (IQ4_XS) | 13,921-token prompt 23.5 → 21.8 s (591 → 638 tok/s); pp2048 789 | 17.2 → 21.2; drafter ~23 → 30.7 | `gemv_v` (decoding the 4-bit codebook had cost as much as reading it), `gemv_glu` fused FFN, `attn_ps` for its head-512 global layers |
| 12 Oct (Google QAT Q4_0) | 788 tok/s | 20.9, drafter 26.8 | Q4_0 support |
| 15 Oct (Level Zero) | 2K 733, 16K 571, 131K 209 tok/s | 21.0 / 19.6 / 9.5; server 21.6, drafter 39.6 | — |
| 16 Oct | 2K 733 → 827, 16K 571 → 625 | 21.5 / 20.1 | register spills removed |

### Qwen3.8 Flash Next (ISTA-DASLab GSQ-RCO IQ3_S, arcinferzero only)

A new architecture (`qwen4exp`): 4 hyper-connection residual streams with mixers before every block, Gated DeltaNet
with a sigmoid gate, gated attention with QSA sparse attention, 512 experts (10 active) streamed from the SSD, and a
26.8 GiB PLE n-gram table in the GGUF's second part read per token. The expert file is 56 GB, so only 12–22 % of the
experts fit in the GPU cache.

| Date | Prompt | Generation | Main change |
|---|---|---|---|
| 16 Oct (evening) | 2,048 tokens ~87 tok/s | ~4 (4.28), cache 7.9 GiB = 12 % | first GPU version; one cache pool of the largest record size |
| 16 Oct (night) | unchanged | 5.45 → 5.9–6.2, then 6.85 | pools grouped by record size (12 → 16 %), dense W8 copies released for the cache (16 → 22 %, hits 80 → 85.5 %), MoE decode kernels reworked, fused single-token mixers (25.1 → 17.4 ms per token) |
| 17 Oct | — | 6.6 → 7.3 | a token's cached experts run while its missing ones are read from the SSD |
| 17 Oct | 8K prompt 98 → 127 tok/s | — | the prompt mixer's gated mean with vector loads (45 → 3.7 ms per 2,048-row chunk) |
| 17 Oct (now) | 8K prompt 145 → 165 tok/s | 7.3 | down projection through `moe_rg` |

The SSD reads random 2 MB records at 3.6–3.8 GB/s at any queue depth (`zero/ssd_bench.cpp`), and the engine is at
that limit.

### Images and audio

| Encoder | Result |
|---|---|
| Qwen3-VL (Qwen3.5 4B), 12 Oct | f16 weights with f32 activations and attention (int8 drifted 10–23 % from the f32 reference): 0.2–0.6 % error; attention with one query row per lane: 1,008 tokens 1,118 → 588 ms, 4,096 tokens 36.1 → 15.3 s (llama.cpp Vulkan: 71 s) |
| Qwen3.5 9B (head size 72), 12 Oct | attention for a 1,222-token image 6.0 → 1.6 s; 0.38 % from the f32 reference |
| Gemma 4 E4B ViT, 12 Oct | 0.11–0.17 % from an f32 reference; 1,120-token image in 3.2 s |
| Gemma 4 E4B audio conformer, 12 Oct | 7.2 s of speech encoded in 226 ms, transcribed word for word |
| Qwen3.6 35B-A3B, 14 Oct | 1,008 image tokens in ~3 s |

## Comparison with llama.cpp

The comparison was run once, on 10 October, with the OpenCL engine against llama.cpp b11527's official Windows
builds (Vulkan and SYCL; flash attention on, the best micro-batch of 512 / 1,024 / 2,048 per model and backend; SYCL
needs `GGML_SYCL_ENABLE_VMM=0` on this driver). The prompts are scans of two local projects, 11,240 and 33,937 tokens
for the Qwen tokenizer (13,921 and 35,957 for Gemma's), sent with the same token ids. llama.cpp SYCL on Gemma 4 12B,
Gemma's Vulkan perplexity and a llama.cpp reference for the 35B (Unsloth's UD-IQ2_M) were not run: llama.cpp's long
prompts on this iGPU would have taken hours more (the 27B's 34K prompt alone took 40 minutes on Vulkan). llama.cpp
was not measured again after that day, while arcinfer kept getting faster (the tables above).

The tables are written by `bench/review_data.py` from `bench/results/results.jsonl`.

<!-- DATA-BEGIN -->
Measured 2026-10-10 with the OpenCL engine; best of the repeated runs per engine. llama.cpp cannot load the GSQ 2-bit Qwen3.6 35B-A3B, so its row has arcinfer only.

Time to first token (seconds) on the code prompts, and generation speed right after the prompt (tok/s):

| Model | Prompt | Tokens | arcinfer TTFT | llama.cpp Vulkan TTFT | llama.cpp SYCL TTFT | arcinfer gen | llama.cpp Vulkan gen | llama.cpp SYCL gen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B | code 11K | 11,240 | 14.3 | 69.0 | 41.7 | 16.13 | 13.62 | 12.77 |
| Qwen3.5 4B | code 34K | 33,937 | 57.5 | 376.1 | 162.9 | 15.71 | 10.64 | 8.15 |
| Qwen3.5 9B | code 11K | 11,240 | 26.8 | 97.1 | 67.0 | 9.96 | 8.52 | 8.94 |
| Qwen3.5 9B | code 34K | 33,937 | 96.0 | 461.3 | 243.1 | 8.63 | 7.26 | 6.36 |
| Qwen3.8 27B | code 11K | 11,240 | 97.1 | 589.6 | 503.9 | 4.00 | 2.36 | 1.78 |
| Qwen3.8 27B | code 34K | 33,937 | 326.6 | 2,417.5 | 1,623.6 | 3.70 | 2.06 | 1.09 |
| Gemma 4 12B | code 11K | 13,921 | 69.1 | 249.7 | not run | 6.41 | 5.51 | not run |
| Gemma 4 12B | code 34K | 35,957 | 239.9 | 1,164.6 | not run | 6.10 | 4.60 | not run |
| Qwen3.6 35B-A3B (GSQ 2-bit) | code 11K | 11,240 | 14.3 | not run | not run | 15.27 | not run | not run |
| Qwen3.6 35B-A3B (GSQ 2-bit) | code 34K | 33,937 | 60.6 | not run | not run | 14.21 | not run | not run |

Synthetic throughput (llama-bench style; tok/s): prompt processing over 512 and 2,048 tokens, generation of 64 tokens after a short prompt:

| Model | arcinfer pp512 | llama.cpp Vulkan pp512 | llama.cpp SYCL pp512 | arcinfer pp2048 | llama.cpp Vulkan pp2048 | llama.cpp SYCL pp2048 | arcinfer tg64 | llama.cpp Vulkan tg64 | llama.cpp SYCL tg64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B | 919.2 | 263.9 | 295.3 | 911.3 | 241.6 | 337.4 | 17.6 | 16.2 | 17.9 |
| Qwen3.5 9B | 317.0 | 158.4 | 161.5 | 401.6 | 150.8 | 189.0 | 10.3 | 9.5 | 11.1 |
| Qwen3.8 27B | 82.5 | 24.2 | 40.5 | 113.2 | 22.9 | 24.9 | 4.2 | 2.8 | 2.2 |
| Gemma 4 12B | 202.7 | 125.1 | not run | 240.1 | 95.6 | not run | 7.5 | 7.0 | not run |
| Qwen3.6 35B-A3B (GSQ 2-bit) | 894.4 | not run | not run | 908.6 | not run | not run | 15.7 | not run | not run |

Perplexity (40 chunks of 512 tokens of llama.cpp documentation, second half of each chunk scored; lower is better). Gemma 4's raw-text perplexity is not a usable accuracy metric on this corpus (see Accuracy); the 35B row is the GSQ file, which llama.cpp cannot load:

| Model | arcinfer | llama.cpp Vulkan | llama.cpp SYCL |
|---|---:|---:|---:|
| Qwen3.5 4B | 6.0562 | 6.1410 | 6.1380 |
| Qwen3.5 9B | 5.4172 | 5.4987 | 5.4969 |
| Qwen3.8 27B | 4.2738 | 4.3114 | 4.3104 |
| Gemma 4 12B | 50.8016 | not run | not run |
| Qwen3.6 35B-A3B (GSQ 2-bit) | 5.6116 | not run | not run |
<!-- DATA-END -->

In short: the 27B's 11K prompt took 97 s against 590 s (Vulkan) and 504 s (SYCL), about 6x; its 34K prompt 5.5
minutes against 40 minutes on Vulkan (7.4x); Gemma 4 12B's 35,957-token prompt 240 s against 1,165 s (4.9x); the 4B's
pp2048 911 against 242 and 337 tok/s. Generation is limited by memory bandwidth and was much closer at short context
(4B tg64 17.6 vs 16.2 / 17.9 tok/s), but after a long prompt the int8 KV cache and DP4a attention keep it up: the 27B
generated 3.7 tok/s at 34K tokens against 2.06 (Vulkan) and 1.09 (SYCL).

## Speculative decoding

All drafting in arcinfer keeps the output: the drafts are verified in one main-model pass and the next token is still
taken from the main model's distribution. Every change was accepted only with greedy answers identical to plain
decoding.

**MTP (Qwen).** Qwen GGUFs carry a multi-token-prediction block. On the OpenCL engine it drafted up to 2, later 3
tokens per pass, with the DeltaNet states copied for the rollback after a rejected draft. arcinferzero (15 October)
replaced the copies: a verification pass writes its states to a second buffer and keeps its inputs, and a rejection
replays the recurrence over the accepted tokens only (one state copy instead of one per draft, no 150 MB copies per
step). That allowed up to 7 drafts, but at first more than 3 were slower, because the 8-token verification GEMVs held
all activations in registers and spilled. The sweep on the 27B (code answer, 400 tokens, before the fix):

| Drafts | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tok/s | 7.45 | 8.31 | 9.36 | 3.06 | 3.83 | 4.17 | 3.89 |
| drafts accepted | 192/208 | 257/280 | 283/333 | 298/376 | 306/405 | 314/417 | 314/456 |

After the spill fix (27B, 5 drafts: 3.8 → 9.2 tok/s) more drafts were no faster than 3, since every draft runs the
MTP block with its full output head (mean of the code and Indonesian answers, tok/s, 16 October):

| Model | 1 | 2 | 3 | 4 | 5 | 7 | default then |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5 4B | – | 22.9 | 24.8 | 25.2 | – | – | 3 (was 2) |
| Qwen3.5 9B | – | 15.8 | 15.6 | 15.9 | – | – | 2 |
| Qwen3.8 27B | – | – | 8.3 | – | 8.2 | 8.2 | 3 |
| Qwen3.6 35B | 23.9 | – | 26.1 | – | – | – | 3 (was 1) |

The token-outer GEMVs of 16 October (each token's activations loaded once for all of a sub-group's rows) made an
8-row 27B pass 487 → ~350 ms, and the defaults moved to 7 drafts for the 27B and 5 for the 35B.

**Gemma 4 assistant drafter.** Gemma 4 drafts with a separate small model that reads the main model's KV cache
instead of keeping its own. Its code was deliberately left unchanged after its introduction; its gains came from the
main model's faster verification. 12B: 8.74 tok/s on 10 October (50 % accepted), 17–20 with the Google QAT file,
21.1 through the server on 15 October. E4B: ~23 → 30.7 on 12 October, 39.6 through the server on 15 October. A third
draft per pass gave 2.67 instead of 2.25 tokens per pass but only +1.5 % overall.

**DFlash2 (16–17 October).** z-lab's DFlash2 block drafter for Qwen3.8 27B (and incoai's for Qwen3.6 35B): 5 layers
fed by the main model's hidden states after layers 5 / 19 / 33 / 47 / 61, which draft a block of 7 tokens in one pass
(8 rows: the token and 7 masks), followed by a candidate selector over the top 16. With the MTP block loaded too,
each step picks the drafter: DFlash2 while its drafts are accepted, else MTP. Its progress on code answers:

| Change | 27B code | 35B code |
|---|---:|---:|
| first version (16 Oct) | 4.85 → 11.5 tok/s | 44.4 |
| token-outer verification GEMVs | 16.2 | 46.0 |
| drafter in Q4_0 (was Q8_0; 0.84 GiB less) | 16.9 → 17.2 | — |
| one-pass top-16 selector | 16.9 → 18.3 | 48.4 → 52.7 |
| vectorized float exp for the drafts' confidence | unchanged | 49.4 → 53.3 |
| README (17 Oct) | 18.1 | 49.8 |

The drafters are trained mostly on English and code: on Indonesian prose DFlash2 takes ~11 % of its drafts, and the
MTP block does most of the work (27B 6.8 tok/s with both, 7.0 with MTP alone).

## Accuracy and correctness

Speed only counted if the answers stayed the same. The checks:

- An independent f32 CPU implementation of every architecture (`tools/ref_forward`); the engine's next-token
  distributions are compared with it by KL divergence averaged over all positions of a prompt, through the prompt
  path (`run --kl-ref`) and token by token through the generation path (`--kl-decode`).
- Greedy continuations identical to llama.cpp on the 4B, 9B, Gemma 4 12B and E4B test prompts; tokenizer output
  identical to `llama-tokenize`; chat templates rendered exactly like the GGUF Jinja templates.
- Kernel changes either bit-identical to the previous kernel (checked by dumping logits) or measured against the
  reference. arcinferzero produces bit-identical logits to the OpenCL engine on every model (the 35B with the same
  prompt chunking); images and audio give the same answers.
- Speculative decoding only accepted when the output equals plain decoding. This caught a real bug: the 35B's
  two-token verification used an attention kernel that split the keys differently from the single-token kernel, so
  MTP output drifted after ~40 tokens of a 2K prompt; verification rows now take the same kernel.

Mean KL against the f32 reference:

| Model | Prompt | KL | Note |
|---|---|---:|---|
| Qwen3.5 4B | chat, 20 positions | 0.0021 | int8 activations, attention and KV cache (0.0006 with f32 attention) |
| Qwen3.6 35B-A3B (GSQ) | chat, 20 positions | 0.0008 | the reference decodes the 2-bit expert layout on its own |
| Gemma 4 12B (Q4_K_M) | chat, 20 positions | 0.029 | 0.032 with one V scale per row |
| Gemma 4 12B (Q4_K_M) | code, 160 positions | 0.018 | 0.23 with one V scale per row, including a flipped prediction (KL 27 at one position) |
| Gemma 4 12B (Q4_K_M) | raw text, 120 positions | 0.035 | 0.080 with one V scale per row |
| Gemma 4 12B (Q4_K_M) | code, 160 positions, token by token | 0.013 | after the 10 October decode work (0.020 before) |
| Gemma 4 12B (Google QAT) | regression test | 0.0206 | greedy equal to llama.cpp |
| Qwen3.8 Flash Next | short chat prompt | 0.0014 | "Jakarta" 0.92 vs 0.93 |
| Qwen3.8 Flash Next | 337-token raw text | 0.15 | the reference with int8 activations gives 0.13 |
| Qwen3.8 Flash Next | 3,566 tokens, QSA | 0.020 | against the CPU reference with QSA (dense attention 0.043) |

Generation path on arcinferzero (`runz --kl-decode`, ~250-token texts; old decode attention vs the new split
kernel): 4B 0.00064 / 0.00067 vs 0.00062 / 0.00064, 9B 0.00069 vs 0.00060, 35B 0.00774 / 0.00444 vs 0.00811 /
0.00436, 27B 0.00030–0.00034 vs 0.00032–0.00040. Changing only the rounding of the 27B's scores moved its KL
0.000355 → 0.000414, so these differences are within noise.

The audit that mattered most: on Gemma 4 the drift from the reference grew with the context, up to complete flips of
the predicted token. Re-running the f32 reference with each piece of arcinfer's int8 rounding simulated in turn
pointed at the V cache: Gemma's V rows hold a few large values, and one scale per row left the rest with too few
levels. One scale per 64 dims removed the flips and halved the drift, at 1.5–1.8x the attention time for Gemma only.
Gemma 4 is about 7x more sensitive to int8 activations than Qwen3.5 (activations rounded to int8 alone, in the f32
reference: KL 0.009 vs 0.001). Flash Next is more sensitive still: every matrix group adds 0.02–0.10. The same method
decided the vision encoders: the Qwen3-VL encoder drifted 10–23 % with int8 activations, so it runs in floating point.

Perplexity in the 10 October benchmark (see the comparison tables) was 6.06 / 5.42 / 4.27 for the 4B / 9B / 27B on
arcinfer against 6.14 / 5.50 / 4.31 on llama.cpp Vulkan. On the 4B, int8 attention and requantized Q4_K together
changed perplexity by less than 0.06 % (5.4749 f32 attention, 5.4779 the default). Gemma's raw-text perplexity is not
usable on this corpus: on 4 chunks the f32 CPU build of llama.cpp gives 130, Vulkan 120, SYCL 113, arcinfer 111.

## What did not work

The full list with measurements is in [lessons-intel-arc-igpu.md](lessons-intel-arc-igpu.md#5-tried-and-not-kept).
The ones that shaped the design:

- **SIMD8 GEMM**: its DP4a microbenchmark peaks higher (13.7 TOPS), but the real kernel was 28 % slower; the GEMM is
  limited by the loads, lane broadcasts and scale math around the DP4a.
- **Fewer instructions in the GEMM**: removing an emulated 32-bit multiply cut the loop by ~10 % and made it 3x
  slower, because the compiler then kept only 2 independent DP4a chains in flight. Writing the interleave into the
  source (16 tokens side by side) was what helped.
- **Larger tiles** (32 tokens per sub-group, also with 256 registers): 2.4 vs 8.1 TOPS; too few threads in flight.
- **f16 P·V** (no faster, less accurate) and **7-bit P** (same speed, 1.8x the error).
- **Exact Q4_K block minimums** in the GEMM: 3x slower and not more accurate end to end.
- **Level Zero for kernel gaps alone** (13 October): gaps between kernels were only 1.6–2 %. Level Zero was adopted a
  day later for a different reason, memory control for streaming experts, and turned out faster as well.
- **More MTP drafts before the GEMV fix**, a fixed draft vocabulary (faster on English code, slower on Indonesian),
  other MTP confidence cut-offs.
- **Expert-cache tricks**: prefetching from a routing guess (SSD +55 %, no faster), a per-layer LRU (80 % vs 82 %
  hits), split SSD reads, a weight arena against the slow-run variance.
- **`attn_dsplit` for Qwen on the OpenCL engine**: up to 2x faster verification, but not bit-identical to
  single-token decoding, which the MTP test requires (on arcinferzero a variant with fixed key splits is used).

## Open issues

- **The original goal was not reached**: an 11,000-token prompt on the 27B in under a minute. On the OpenCL engine it
  took ~90 s on a cool laptop (97 s in the benchmark); the GEMMs alone took ~70 s, and 60 s would need more than 90 %
  of the DP4a peak. The 11K prompt has not been timed again on arcinferzero since its faster GEMM.
- **Slow-run variance**: about one run in five on the 35B, every dense-weight kernel runs ~2x slower with identical
  work (the expert cache stays at full speed); a server started right after another one stops sometimes ran at half
  speed for its whole life (`arc-server stop` now waits 5 s). Suspected: physical page placement of the weight
  buffers. Not solved.
- **27B generation** stays at ~72 % of the bandwidth: the IQ3 / IQ2 codebook decoding is not hidden behind the
  memory reads.
- **Flash Next**: no MTP block and no image input; generation depends on the SSD (cache 12–22 % of the experts);
  the README still lists long-context QSA as not validated beyond the 3,566-token check.
- **Register spills left**: `gdn_conv` 1 KB and `moe_gateup_q` 2 KB (256 registers remove them but are not faster).
- **Not done yet**: cheaper MTP drafts (dynamic token list or draft tree), several conversations at once and batched
  decoding, `/v1/embeddings`, a 4-bit KV cache, removing the OpenCL engine (arcinferzero is the default; the OpenCL
  path is kept with `--opencl` until it has been used daily for a while).
- From the OpenCL-era review, not re-investigated: on the 27B each request took ~3 s before the first token even for a
  handful of new tokens (15 tokens: 2.8–3.8 s); a cancelled Gemma 4 prompt can only be rolled back while its
  sliding-window rings still hold the positions before the block (blocks up to ~8K tokens).

## Method

- Models: Qwen3.5 4B and 9B UD-Q4_K_XL (Unsloth); Qwen3.8 27B GSQ-RCO IQ3_S-mtp (ISTA-DASLab); Gemma 4 12B and E4B
  (community Q4_K_M / IQ4_XS until 12 October, then Google's QAT Q4_0, with Unsloth's drafters); Qwen3.6 35B-A3B GSQ
  2-bit (ISTA-DASLab, converted) and Unsloth's UD-Q3_K_S / UD-Q3_K_M / MTP files; Qwen3.8 Flash Next GSQ-RCO IQ3_S;
  DFlash2 drafters from z-lab and incoai.
- Comparison with llama.cpp: `bench/run_bench.ps1`, raw results in `bench/results/results.jsonl`, tables written by
  `bench/review_data.py`. Each test preceded by 30 s of idle GPU; throughput the best of repeated runs.
- Perplexity: 40 chunks of 512 tokens from 500 KB of llama.cpp documentation, second half of each chunk scored (the
  llama-perplexity method).
- Level Zero numbers: `runz --bench-pp` (a prompt of N tokens, then 64 generated tokens) and server answers of
  400 tokens at temperature 0 to a Python code question and an Indonesian explanation of TCP / UDP. Generation after a
  long prompt is sometimes very noisy (one E4B run at 32K dropped to 10.8 from 17.9 tok/s; seven repeats were steady
  at 18.5–19.0).
- Kernels: `bench/gemv_bench`, `bench/attn_bench`, `bench/attn_dec_bench`, `bench/gemm_w8_bench`, `bench/moe_bench`
  on real model tensors, checked against CPU references; regression tests `tests/run_tests.ps1`.
