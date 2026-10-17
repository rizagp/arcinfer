# arcinfer speed log

Performance test report for arcinfer, 10–14 October 2026: prompt processing (prefill) and token generation for models from 4B to 35B on the Intel Arc iGPU, from the first version to the current state.

**Running a 35B model on a laptop without a discrete graphics card.** This report summarizes the performance tests of arcinfer, an inference engine built specifically for the Intel Arc iGPU, from its first version to its current state. The tests cover prefill (prompt processing) speed and token generation speed on six models, from Qwen3.5 4B to Qwen3.6 35B-A3B.

Every measurement was taken on the same machine. Data sources: the arcinfer CHANGELOG, QUEUE, review_notes and bench/results.

Note: this report uses the OpenCL engine (10–14 October). Since 16 October the default engine is arcinferzero (the same kernels through Level Zero, bit-identical results); the latest numbers are in [performance-zero-2026-10-16.md](performance-zero-2026-10-16.md).

## Test platform

| Item | Value |
|---|---|
| CPU | Core Ultra 5 125H |
| iGPU | Arc, 112 EU (7 Xe-cores, 2.2 GHz) |
| Matrix units | No XMX (DP4a int8 only) |
| int8 peak | 13 TOPS (DP4a, measured) |
| Memory | 32 GB LPDDR5x |
| Bandwidth | ~75 GB/s measured (102 theoretical) |
| GPU allocation | 18.3 GiB (shared with Windows) |
| API | OpenCL 3.0 |

## The main limit: memory bandwidth

During token generation the model reads all of its weights once for every generated token. With a measured memory bandwidth of about 75 GB/s, the upper bound on generation speed is simply `75 GB/s ÷ weight size`. For example, Qwen3.8 27B with about 11 GB of weights can reach only about 6.5 tokens per second in theory, however good the kernels are.

So the most useful measure for generation is not tokens per second alone but **the share of bandwidth actually used**. In the first measurement Qwen3.8 27B used only 36 % of the bandwidth. The rest was lost in three places:

- **Decoding the quantization format:** formats such as IQ3_S, IQ2_XXS and IQ4_XS need extra compute to unpack into weight values.
- **Unaligned reads:** most blocks start at addresses that are not a multiple of 4 bytes, so they cannot be read with vector instructions.
- **Kernel launch overhead:** too many small kernels, each with a fixed cost.

### Bandwidth use during token generation (% of ~75 GB/s)

| Model | First measurement | After the GEMV optimizations |
|---|---|---|
| Qwen3.5 4B | 64 % | 79 % |
| Qwen3.5 9B | 72 % | 86 % |
| Gemma 4 E4B | 68 % | 81 % |
| Gemma 4 12B | 80 % | 82 % |
| Qwen3.8 27B | 36 % | 72 % |
| Qwen3.6 35B | – | 58 % |

First measurement: QUEUE item 2. After the optimizations: quantization blocks re-laid-out so they can be read with aligned vector loads (12–13 October). Qwen3.6 35B has only one data point because the MoE expert kernel reads small, scattered blocks.

The largest gain came from a step that looks simple: **re-laying-out the bytes**. GGUF puts the `f16` scale at the start of every block, so the quantized data after it lands on an address that is not a multiple of 4 bytes. Moving the scale to the end (IQ2/IQ3) or gathering the scales at the start of the superblock (Q8_0, Q4_0) lets every GPU lane load its block in one vector instruction. The effect on GEMV speed per quantization type:

### GEMV speed per quantization type (GB/s)

| Type | Before | After |
|---|---|---|
| Q8_0 | 37 | 77 |
| IQ2_XXS | 27 | 60 |
| IQ3_XXS | 41 | 62 |
| IQ3_S | 61 | 66 |
| IQ4_XS (E4B q) | 44 | 67 |
| Q6_K (head) | 66 | 72 |

Source: CHANGELOG 12 October (GEMV bandwidth) and review_notes (IQ4_XS gemv_v on E4B: q matrix 44 → 67 GB/s).

## Token generation: from the first version to now

The table below compares generation speed in the first measurement with the current state. For models that have an MTP block (Qwen) or a drafter (Gemma), the result with speculative decoding is shown too: a small model guesses several tokens at once and the main model verifies them in one pass. The output stays bit-identical to plain decoding, only faster.

### Token generation speed (tokens/s)

| Model | First measurement | Now, without MTP | Now, with MTP / drafter |
|---|---|---|---|
| Qwen3.5 4B | 17.1 | 20.4 | – |
| Qwen3.5 9B | 10.3 | 12.0 | – |
| Gemma 4 E4B | 16.9 | 21.2 | 30.7 |
| Gemma 4 12B | 7.5 | 9.1 | 18.5 |
| Qwen3.8 27B | 4.1 | 4.9 | 9.5 |
| Qwen3.6 35B | 17.5 | 19.8 | 22.3 |

Qwen3.8 27B with MTP reaches 9.5 tok/s on code (3 drafts); on general text the number is lower. Gemma 4 12B and E4B: the first measurement used a community quantization file that was later replaced by Google's official QAT file (Q4_0). Qwen3.6 35B: the first measurement used the standard Unsloth file (12 October); the earlier 2-bit GSQ version was around 15 tok/s.

## Prefill: getting the most out of int8 compute

Prefill behaves differently from generation. Thousands of tokens are processed together, so the weights are reused many times and the limit moves from memory bandwidth to compute. This iGPU has no matrix units (XMX) and relies only on the `DP4a` instruction, four int8 multiplies in one step, with a peak of about 13 TOPS. arcinfer converts all weights to int8 (W8), quantizes the activations per 32 values, and runs them through its own DP4a GEMM kernels, which reach 6–9 TOPS.

For comparison, here are results against llama.cpp (build b11527, Vulkan and SYCL backends, flash attention on, best ubatch) with the same prompts:

| Speedup | Measurement |
|---|---|
| 6× | Qwen3.8 27B, 11,240-token prompt: 97 s, against 590 s (Vulkan) and 504 s (SYCL) |
| 7.4× | Qwen3.8 27B, 33,937-token prompt: 5.5 minutes, against 40 minutes (Vulkan) |
| 3.8× | Qwen3.5 4B pp2048: 911 tok/s, against 242 (Vulkan) and 337 (SYCL) |
| 4.9× | Gemma 4 12B, 35,957-token prompt: 240 s, against 1,165 s (Vulkan) |

### Prefill speed pp2048 (tokens/s)

| Model | arcinfer | llama.cpp Vulkan | llama.cpp SYCL |
|---|---|---|---|
| Qwen3.5 4B | 911 | 242 | 337 |
| Qwen3.5 9B | 402 | 151 | 189 |
| Gemma 4 12B | 240 | 96 | – |
| Qwen3.8 27B | 113 | 23 | 25 |
| Qwen3.6 35B | 642 | – | – |

Source: bench/results/results.jsonl, 10 October (first test). Gemma 4 12B used a community Q4_K_M quantization file; SYCL was not tested. Qwen3.6 35B: llama.cpp cannot load the GSQ file, so only the arcinfer number for the Unsloth Q3_K_M file is shown (13 October).

The lead also shows in generation after a long prompt. At 34K tokens of context Qwen3.8 27B still generates 3.70 tok/s, against 2.06 tok/s for Vulkan and 1.09 tok/s for SYCL. The difference comes from int8 attention: keys and values are stored as int8 with a per-row scale, and both the `Q·K` scores and `P·V` also use DP4a.

## Qwen3.8 27B: time to first token down from 116 to 89 seconds

The 11,240-token prompt on Qwen3.8 27B has been the reference since the start of development. Each step below is one optimization that proved to work:

| Time to first token | Step |
|---|---|
| 116 s | first version with correct output |
| 112 s | int8 attention scores and tiles in local memory |
| 101 s | int8 P·V, int8 V cache, head splitting |
| 94 s | split-K and Q4_K conversion to signed int8 |
| 91 s | per-layer prefill (W8 decode only once) |
| 89 s | fused SwiGLU and vectorized convolution |

Attention alone went from **33 seconds to 5.1 seconds**. This was done in stages: int8 scores with the key tile in local memory, then int8 `P·V` (probabilities quantized to 8 bits per block, following the SageAttention approach), then splitting heads across sub-groups.

## Qwen3.6 35B-A3B: mixture of experts on an iGPU

Qwen3.6 35B has 256 experts in every layer, but only 8 are active for each token. For generation this is a big advantage, since only about 2.4 GB is read per token, so the 35B runs about 4 times faster than the 27B. For prefill it is the opposite: each expert gets only about 64 tokens per 2,048-token prompt chunk, too few for GEMM to work efficiently. The optimizations done so far:

| Optimization (Qwen3.6 35B) | Before | After |
|---|---|---|
| Q8_0 stored once and decoded per prompt chunk | context 61,440 | 208,896 (Q3_K_S) |
| Router and shared expert fused into one GEMV | 17.4–18.5 tok/s | 18.2–18.8 tok/s |
| Experts run with int8 GEMM (decode per 64 experts, then GEMM per tile) | 476 tok/s | 642 tok/s (pp2048) |
| IQ3_XXS decode through local memory (decode_w8_sl) | 1,038 ms | 518 ms per chunk |
| DeltaNet qkv and z fused into one GEMV | 19.2 tok/s | 19.8 tok/s |
| MTP block enabled (Unsloth MTP repo file, Q3_K / Q4_K on the GPU) | 19.7 tok/s | 22.3 tok/s (83 % of drafts accepted) |
| MoE buffers share memory with the attention buffers | context 86K | 102,400 |
| Prompt attention with 4 heads per sub-group (21,590 tokens) | 9.4 s | 8.4 s |

## Accuracy: speed must not cost correctness

> An optimization that changes the output, however slightly, is not accepted.

Every optimization on the decode path is checked in two ways:

- **KL divergence against an f32 reference:** the token distribution is compared with a full-precision CPU computation. Results: Qwen3.5 4B 0.0021, Qwen3.6 35B (GSQ) 0.0008, Gemma 4 12B QAT 0.021.
- **Bit identity for MTP:** output with speculative decoding must be exactly the same as plain decoding on a 2K-token prompt.

The second check caught a well-hidden bug on the 35B. Two-token verification used an attention kernel that split the keys differently from the one-token kernel. As a result the text started to diverge after about 40 tokens, and only on long prompts. After the fix, the 35B MTP output is bit-identical again.

## Long context

Automatic context uses as much of the remaining GPU memory as possible after the weights, the work buffers and about 1.2 GiB kept free for Windows. Results at the extremes:

| Model | Prompt length | Processing time | Generation afterwards |
|---|---|---|---|
| Qwen3.5 4B | 235,929 tokens | 27.3 minutes (144 tok/s) | 9.0 tok/s |
| Gemma 4 E4B | 117,965 tokens | 11.9 minutes (165 tok/s) | 11.6 tok/s |
| Qwen3.8 27B | 81,100 tokens | 16.4 minutes (82 tok/s) | 4.9 tok/s |
| Qwen3.6 35B | 21,590 tokens | 44 s (493 tok/s) | – |

When used as an agent (Pi), these numbers are rarely felt directly. The server keeps the running conversation, so each turn processes only the new tokens. The per-project system prompt is also cached on disk, so a new Qwen3.8 27B session on a project that was opened before can start in about 5 seconds, against about 40 seconds before.

## Optimizations that did not pay off

Not every idea worked. These approaches were tried and dropped, with the reason:

- **Level Zero instead of OpenCL.** The gap between kernels was only 1.6–2 %, not worth rewriting the runtime.
- **SIMD8 GEMM.** Reached 13.7 TOPS in a microbenchmark, but was slower in the model.
- **P·V in f16.** No speedup, and the KL got worse.
- **32-key attention tile.** 14–20 % slower because of register spills.
- **Split-K in GEMV.** 512-row matrices were 10–15 % faster, but other matrices slowed down, so no gain overall.
- **W8 with a min value.** Only 2.4 TOPS, three times slower than converting to signed int8.
- **Byte-pair LUT for IQ4_XS.** Dropped from 66 to 55 GB/s; a small table in registers is still better.
- **Fusing down and combine (35B).** 212 ms against 211 ms. Fewer kernels are not automatically faster.
- **Q8_0 through gemv_v.** Slower than GEMV with a local-memory window.
- **Per-chunk DeltaNet.** Only 4–9 % of prefill time, so the extra complexity is not worth it.
- **attn_ps for the Qwen configurations.** Only 1.0–1.6 T MAC/s, below attn_q8 at about 2.2.
- **A third MTP draft on the 4B and the MoE.** No speed gain. Defaults: 3 drafts for the 27B, 1 for the MoE, and 2 for the other models.

## Next steps

Two priorities matter most to keep long-context performance up:

- **4-bit KV cache:** halves the data read per token at deep context and doubles the maximum context, provided its accuracy passes the tests.
- **Decode the 35B experts once per layer:** today about 22 % of the 35B prefill time is spent decoding the same weights again for every prompt chunk.

Methodology note: generation numbers are the best of two runs, because the first run after another process stops tends to be slower, and the iGPU lowers its clock by about 5–8 % under sustained load. During this period some models also changed files (Gemma community quantization to Google QAT; 35B GSQ to Unsloth), and the tables note where the first measurement came from a different file.
