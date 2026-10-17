# arcinfer speed progress

One file that tracks the speed of arcinfer over time on the same laptop (Core Ultra 5 125H, Arc Xe-LPG 112 EU, no
XMX, DP4a, ~75 GB/s, 32 GB LPDDR5x-6400, Windows 11). The newest measurements come first; each dated section keeps
what was measured then; where a later change overturned a statement, a **Later:** note says so. Add a new dated
section at the top when the speed changes, and update the current table.
The story behind the numbers: [review.md](review.md); what helped and what did not: [lessons-intel-arc-igpu.md](lessons-intel-arc-igpu.md).

## Current speed (2026-10-17, arcinferzero)

Greedy answers of 96 tokens, identical to plain decoding in every column (`runz`, warm kernels):

| Model, text | plain | MTP alone | DFlash2 + MTP |
|---|---|---|---|
| Qwen3.8 27B, Python code | 4.91 tok/s | 15.5 | 18.1 |
| Qwen3.8 27B, Indonesian prose | 4.88 | 7.0 | 6.8 |
| Qwen3.6 35B-A3B, Python code | 21.2 | 39.3 | 49.8 |
| Qwen3.6 35B-A3B, Indonesian prose | 21.0 | 22.7 | 22.7 |

| Model | Prompt | Generation |
|---|---|---|
| Qwen3.8 27B (IQ3_S GSQ) | pp4096 ~135 tok/s, pp16384 ~130 | 4.9 tok/s (207 ms per token; 231 ms at 32K context) |
| Qwen3.6 35B-A3B (UD-Q3_K_M) | pp4096 ~795 tok/s, pp8192 ~757 | ~21 tok/s |
| Qwen3.8 Flash Next (IQ3_S GSQ, experts streamed from the SSD) | pp8192 ~165 tok/s | 7.3 tok/s |

These are `runz` runs with a short context. Through the server the MoE models also hold the image encoder, the
DFlash2 drafter and a 32-40K context, which leaves a smaller expert cache: the 35B (cache 77 %) answers code at about
32 tok/s and Indonesian prose at about 18. Since 17 October the automatic context gives the expert cache priority
(up to 90 % of the experts) because a small cache makes generation pause for SSD reads; before, the 35B took a
253,952-token context with a 60 % cache and generated 10-15 tok/s.

## 2026-10-18: long contexts with the experts streamed from the SSD

`runz`, no image encoder loaded (the server also holds one: about 1 GB less expert cache). Generation: 128-token
answers right after loading (short prompt); "hit" is the expert cache hit rate during generation; the SSD column is
everything read from the expert file, mostly the cache fill at load. Long prompts: synthetic tokens
(`--bench-pp`, one run), then 64 tokens of plain decoding at that depth. Pauses between runs (the laptop throttles
when hot).

### Qwen3.6 35B-A3B UD-IQ4_XS (18.2 GB; experts 14.7 GiB in the expert file)

The file: expert `down` IQ4_XS (37 layers) / Q6_K (3), `gate` and `up` IQ3_S (39 of 40 layers), dense parts Q8_0,
output Q6_K. The expert file (15.8 GB) is written once at the first load.

| Context | Expert cache (DFlash2 + MTP) | Code, DFlash2 + MTP | Indonesian, DFlash2 + MTP | Code, MTP only | Indonesian, MTP only | Plain |
|---|---|---|---|---|---|---|
| 32,768 | 12.96 GiB (88 %) | 39.0 * | 19.6 * | – | – | – |
| 65,536 | 12.61 GiB (86 %) | 48.6 | 21.7 | – | – | – |
| 98,304 | 12.26 GiB (83 %) | 48.9 | 21.6 | – | – | – |
| 131,072 | 11.91 GiB (81 %), hit 100 % | 47.5 | 20.7 | 34.6 (cache 87 %) | 22.0 | 21.2 (90 %) |
| 262,144 | 10.48 GiB (71 %), hit 93-96 % | 36.5 | 19.2 | 36.6 (78 %) | 23.4 | 21.2 (81 %) |

\* the first run after the expert file was written and the kernels compiled.

| Context | Prompt | Generation at that depth (plain) |
|---|---|---|
| 131,072 (prompt 125,000) | 321 tok/s (6.5 min) | 13.5 tok/s (median step 72.8 ms) |
| 262,144 (prompt 250,000) | 193 tok/s (21.6 min) | 10.6 tok/s (median step 92.5 ms) |

Against the OpenCL engine with UD-Q3_K_M (section "Prompt and generation at different context lengths" below):
prompts +27 % at 131K (253.5 -> 321) and +32 % at 262K (146 -> 193); generation at depth about the same
(13.8 / 10.1 then).

- The 35B's KV cache is small (10 attention layers, 2 KV heads): from 32K to 262K the expert cache only drops from
  88 % to 71 % (81 % without drafters), so short answers stay fast up to 131K.
- At depth, generation is bound by **reading the KV cache**, not by the SSD: a step takes ~48 ms at a short context
  and 92 ms after 250K tokens; the difference matches reading ~2.6 GB of int8 K/V per token at ~60 GB/s. A 4-bit V
  (or K and V) cache would cut that directly (see the accuracy section below).
- DFlash2 at 262K: no gain for code (36.5 vs 36.6 MTP only) and a loss for Indonesian (19.2 vs 23.4): its 8-row
  verification touches more experts while the cache is smaller. At 131K it still pays for code (47.5 vs 34.6).
- **Recommended context: 131,072** (cache 81 % with DFlash2 + MTP, hit 100 % in these runs, the same speed as 64K).
  262,144 works, but DFlash2 should then be off, and long sessions run close to the 60 % cache where generation used
  to stall (server, 17 Oct).

### Qwen3.8 Flash Next IQ3_S (83.6 GB; experts 52.1 GiB, expert file rebuilt 18 Oct)

| Context | Expert cache | Hit (code / Indonesian) | Code | Indonesian |
|---|---|---|---|---|
| 32,768 | 11.55 GiB (22 %) | 77.4 / 82.8 % | 5.99 tok/s | 6.87 |
| 65,536 | 11.12 GiB (21 %) | 76.2 / 81.9 % | 5.99 | 6.66 |
| 98,304 | 10.68 GiB (20 %) | 75.0 / 80.9 % | 5.90 | 6.60 |
| 131,072 | 10.24 GiB (20 %) | 74.2 / 79.9 % | 5.85 | 6.44 |
| 262,144 | 8.44 GiB (16 %) | 70.8 / 78.0 % | 5.47 | 6.23 |

| Context | Prompt | Generation at that depth (plain) |
|---|---|---|
| 131,072 (prompt 125,000) | 78.6 tok/s (26.5 min) | 3.35 tok/s (median step 190 ms) |
| 262,144 (prompt 250,000) | 37.8 tok/s (110 min) | 3.97 tok/s (median step 240 ms) |

- The cache is a fifth of the experts at any context, so the context costs little: from 32K to 131K short answers
  lose 2-6 %, at 262K 9-10 %.
- Long prompts are the limit: 165 tok/s at 8K, 79 at 125K, 38 at 250K. The slowdown grows faster than the length,
  i.e. attention / QSA dominates long prompts (profile pending, review N-5).
- **Recommended context: 131,072** (−2 to −6 % against 32K); 262,144 only for occasional very long documents.

## 2026-10-17: speculative decoding, MoE prompts, Flash Next

| Change | Before | After |
|---|---|---|
| Token-outer multi-token GEMVs (verification rows) | 27B FFN, 8 rows: 23 GB/s | 39-43 GB/s |
| Default MTP drafts 27B 3 -> 7, 35B 3 -> 5 | 27B code 12.6 tok/s, 35B code 34.2 | 15.2, 37.7 |
| 2-5-token GEMVs: 4 rows per sub-group for codebook types / long rows | 27B Indonesian with MTP 6.95 tok/s | 7.2 |
| DFlash2 drafter weights Q4_0 (was Q8_0) | 27B code 16.9 tok/s | 17.2 (drafter 0.84 GiB smaller) |
| DFlash2 selector: one-pass top-16 instead of nth_element | 35B code 48.4 tok/s, 27B 16.9 | 52.7, 18.3 |
| Draft confidence with a vectorized float exp | 35B code DFlash2 49.4 tok/s, MTP 37.5 | 53.3, 39.1 |
| DFlash2 probes backed off on dense models | 27B Indonesian hybrid 6.2 tok/s | 6.5 |
| Cached experts computed while missing ones are read from the SSD | Flash Next generation 6.6 tok/s | 7.3 |
| Flash Next prompt mixer (gated mean) with vector loads | 45 ms per 2048-row chunk; pp8192 98 tok/s | 3.7 ms; 127 |
| moe_rg (MoE GEMM on raw expert rows) for Flash Next's down projection | pp8192 145 tok/s | 165 |
| moe_rg for 35B 384-1152-token chunks | 412 / 440 / 449 / 486 / 529 tok/s (384 / 512 / 768 / 1024 / 1152) | 424 / 459 / 511 / 520 / 548 |
| serverz reports 4096-token chunks for streamed MoE (was 2048) | 35B pp8192 620 tok/s | 750 |
| Automatic context: expert cache first for streamed MoE | 35B server: 253,952 tokens, 60 % cache, 10-15 tok/s | 32,768-40,960 tokens, 77 % cache, ~32 / 18 tok/s (code / Indonesian) |
| Automatic context: Flash Next | 4,096 tokens (the 60 % cache reservation did not fit) | 32,768 |

Details of each change: [arcinferzero-log.md](arcinferzero-log.md) (16-17 October entries).

## 2026-10-16: arcinferzero (Level Zero) becomes the default engine

arcinfer, overnight results, 15 → 16 October 2026.

Tests ran from 17:00 to 06:05. Since this morning zero is the default engine; the old OpenCL engine is still available with `--opencl`. Every number here comes from real runs on this laptop.

Hardware: Core Ultra 5 125H, Arc Xe-LPG 112 EU, no XMX (DP4a), 32 GB RAM.


### Summary

| Measurement | Result | Context |
|---|---|---|
| Qwen3.6 35B, MTP 3 drafts (code) | 31.5 tok/s | 19.9 without MTP, identical answers |
| Gemma E4B, 2K prompt | 827 tok/s | 733 the evening before (+13 %) |
| Qwen3.8 27B, MTP (code) | 9.1 tok/s | 5.1 without MTP; 5 drafts used to be 3.8 |
| Match with OpenCL | 6 / 6 models | bit-identical logits, same image and audio answers |

### Answer speed through the server

400-token answers at temperature 0 to two questions: Python code, and an explanation of TCP/UDP in Indonesian. Answers with MTP are always exactly the same as without it. Values in tok/s.

| Model | Drafts | Without MTP | MTP, code answer | MTP, Indonesian answer |
|---|---|---|---|---|
| Qwen3.5 4B | 3 drafts | 20.5 | 30.4 | 25.9 |
| Qwen3.5 9B | 2 drafts | 12.1 | 16.7 | 15.0 |
| Qwen3.8 27B | 3 drafts | 5.1 | 9.1 | 7.4 |
| Qwen3.6 35B | 3 drafts | 19.9 | 31.5 | 24.6 |
| Gemma E4B | drafter | 21.6 | 39.6 | – |
| Gemma 12B | drafter | 9.8 | 21.1 | – |

Gemma uses its assistant drafter (not MTP) and was measured with the code question only (baseline, evening of 15 Oct). The Qwen numbers come from the last runs with the final build.

### Number of MTP drafts

Mean of the code and Indonesian answers, tok/s. More drafts are not automatically faster: every draft runs the MTP block with the full output head.

| Model | 1 | 2 | 3 | 4 | 5 | 7 | Default |
|---|---|---|---|---|---|---|---|
| Qwen3.5 4B | – | 22.9 | 24.8 | 25.2 | – | – | 3 (was 2) |
| Qwen3.5 9B | – | 15.8 | 15.6 | 15.9 | – | – | 2 |
| Qwen3.8 27B | – | – | 8.3 | – | 8.2 | 8.2 | 3 |
| Qwen3.6 35B | 23.9 | – | 26.1 | – | – | – | 3 (was 1) |

**Later (17 Oct):** with the token-outer multi-token GEMVs more drafts did pay off; the defaults became 7 for the
27B and 5 for the 35B (code 12.6 -> 15.2 and 34.2 -> 37.7 tok/s), 3 for the 4B and 2 for the 9B.

### Prompt and generation at different context lengths

`runz --bench-pp`: a prompt of N tokens, then 64 generated tokens. Prompt = tok/s processing the prompt; gen = tok/s generating after it. The "later" columns are a second, later measurement: for Gemma, evening of 15 Oct → morning of 16 Oct; for the Qwen3.8 27B prompt, morning → afternoon of 16 Oct (faster prompt GEMM; generation not re-measured). "–" means not re-measured.

| Model | Context | Prompt | Gen | Prompt, later | Gen, later |
|---|---|---|---|---|---|
| Qwen3.5 4B | 2K | 862.5 | 20.2 | – | – |
| | 16K | 696.9 | 18.7 | – | – |
| | 131K | 280.5 | 9.5 | – | – |
| Qwen3.5 9B | 2K | 399.1 | 12.1 | – | – |
| | 16K | 387.9 | 11.5 | – | – |
| | 131K | 221.9 | 8.8 | – | – |
| Qwen3.8 27B | 2K | 110.2 | 5.1 | 130.1 | – |
| | 16K | 102.8 | 4.7 | 125.4 | – |
| | 64K | 79.6 | 4.0 | – | – |
| Qwen3.6 35B | 2K | 660.0 | 20.1 | – | – |
| | 16K | 678.7 | 18.8 | – | – |
| | 131K | 253.5 | 13.8 | – | – |
| | 262K | 146.1 | 10.1 | – | – |
| Gemma E4B | 2K | 733.3 | 21.0 | 827.5 | 21.5 |
| | 16K | 571.1 | 19.6 | 624.9 | 20.1 |
| | 131K | 209.4 | 9.5 | 197.2 | 9.4 |
| Gemma 12B | 2K | 270.6 | 9.5 | 289.4 | 9.6 |
| | 16K | 218.4 | 9.2 | 227.7 | 9.2 |
| | 131K | 86.1 | 7.4 | 86.5 | 7.6 |

Generation after a long prompt is sometimes very noisy (one E4B run at 32K dropped to 10.8 from 17.9 tok/s). Seven repeat runs were steady at 18.5–19.0, so the E4B drop at 131K in a single run (209 → 197) is not treated as a regression yet.

### Kernels improved

| Kernel | Before | After |
|---|---|---|
| 35B generation attention, 64K | 1.16 ms | 0.94 ms (improved) |
| GEMV for 4–7 drafts (27B, 5 drafts) | 3.8 tok/s | 9.2 tok/s (improved) |
| attn_prep_g Gemma HD512, spill | 4,608 B | 0 (improved) |
| add_norm_quant Gemma, spill | 3,712 B | 64 B (improved) |
| gdn_conv Qwen, spill | 1,024 B | 1,024 B |
| moe_gateup_q 35B, spill | 2 KB | 2 KB |

The last two spills go away with 256 registers, but that is not faster, so they stay.

### Qwen prompt attention

The last 4K chunk at 64K depth, ms per layer. The current kernel (256 registers) against the best variant that splits the head dimensions.

| Model | Heads | attn_q8 (in use) | best attn_ps |
|---|---|---|---|
| 4B / 9B | 16 q / 4 kv | 926 ms | 1,117 ms |
| 35B | 16 q / 2 kv | 923 ms | 1,138 ms |
| 27B | 24 q / 4 kv | 1,516 ms | 1,669 ms |

### Correctness

| Test | Result |
|---|---|
| Logits, zero vs OpenCL, 11K tokens (4B, 9B, 27B, E4B, 12B) | bit-identical |
| Qwen3.6 35B with the same prompt chunking | bit-identical |
| Gemma E4B / 12B after this morning's spill fixes | bit-identical |
| Images (E4B, 12B, 9B) and audio (E4B, 12B) | same answers |
| MTP with 1–7 drafts, every Qwen model | same answers |
| Gemma tool calls with the constraint (stress test) | 9/10 valid (was 2/10) |
| API formats (OpenAI, Anthropic, Ollama, Responses, JSON schema, logprobs) | all working |
| API unit tests and tool-call streaming | all passed |
| New 35B generation attention, KL vs the f32 reference | within noise (flagged for attention) |

### Notes

- Correction: during the night I first concluded that Level Zero ignores the 256-register flag. That was wrong; my own environment switch came out empty. Every number on this page was measured again with the correct build.
- A server started right after another one stops sometimes runs at half speed. `arc-server stop` now waits 5 seconds, and runz logs the time of every step so the cause can be seen if it happens again.
- Not done yet: cheaper MTP drafts (a dynamic token list or a draft tree), and the two small spills above.
  **Later:** a fixed draft vocabulary was tried on 17 Oct and dropped (Indonesian tokens lie beyond id 131,072); the
  host-side cost of drafting was cut instead (one-pass top-K, vectorized softmax: 35B code 44 -> 53 tok/s).

## 2026-10-10 to 2026-10-14: the OpenCL engine, from the first version

Performance test report for arcinfer, 10–14 October 2026: prompt processing (prefill) and token generation for models from 4B to 35B on the Intel Arc iGPU, from the first version to the current state.

**Running a 35B model on a laptop without a discrete graphics card.** This report summarizes the performance tests of arcinfer, an inference engine built specifically for the Intel Arc iGPU, from its first version to its current state. The tests cover prefill (prompt processing) speed and token generation speed on six models, from Qwen3.5 4B to Qwen3.6 35B-A3B.

Every measurement was taken on the same machine. Data sources: the arcinfer CHANGELOG, QUEUE, review_notes and bench/results.


### Test platform

| Item | Value |
|---|---|
| CPU | Core Ultra 5 125H |
| iGPU | Arc, 112 EU (7 Xe-cores, 2.2 GHz) |
| Matrix units | No XMX (DP4a int8 only) |
| int8 peak | 13 TOPS (DP4a, measured) |
| Memory | 32 GB LPDDR5x |
| Bandwidth | ~75 GB/s measured (102 theoretical) |
| GPU allocation | 18.3 GiB (shared with Windows) |
| API | OpenCL 3.0 (Level Zero from 16 October) |

### The main limit: memory bandwidth

During token generation the model reads all of its weights once for every generated token. With a measured memory bandwidth of about 75 GB/s, the upper bound on generation speed is simply `75 GB/s ÷ weight size`. For example, Qwen3.8 27B with about 11 GB of weights can reach only about 6.5 tokens per second in theory, however good the kernels are.

So the most useful measure for generation is not tokens per second alone but **the share of bandwidth actually used**. In the first measurement Qwen3.8 27B used only 36 % of the bandwidth. The rest was lost in three places:

- **Decoding the quantization format:** formats such as IQ3_S, IQ2_XXS and IQ4_XS need extra compute to unpack into weight values.
- **Unaligned reads:** most blocks start at addresses that are not a multiple of 4 bytes, so they cannot be read with vector instructions.
- **Kernel launch overhead:** too many small kernels, each with a fixed cost.

#### Bandwidth use during token generation (% of ~75 GB/s)

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

#### GEMV speed per quantization type (GB/s)

| Type | Before | After |
|---|---|---|
| Q8_0 | 37 | 77 |
| IQ2_XXS | 27 | 60 |
| IQ3_XXS | 41 | 62 |
| IQ3_S | 61 | 66 |
| IQ4_XS (E4B q) | 44 | 67 |
| Q6_K (head) | 66 | 72 |

Source: CHANGELOG 12 October (GEMV bandwidth) and review_notes (IQ4_XS gemv_v on E4B: q matrix 44 → 67 GB/s).

### Token generation: from the first version to now

The table below compares generation speed in the first measurement with the current state. For models that have an MTP block (Qwen) or a drafter (Gemma), the result with speculative decoding is shown too: a small model guesses several tokens at once and the main model verifies them in one pass. The output stays bit-identical to plain decoding, only faster.

#### Token generation speed (tokens/s)

| Model | First measurement | Now, without MTP | Now, with MTP / drafter |
|---|---|---|---|
| Qwen3.5 4B | 17.1 | 20.4 | – |
| Qwen3.5 9B | 10.3 | 12.0 | – |
| Gemma 4 E4B | 16.9 | 21.2 | 30.7 |
| Gemma 4 12B | 7.5 | 9.1 | 18.5 |
| Qwen3.8 27B | 4.1 | 4.9 | 9.5 |
| Qwen3.6 35B | 17.5 | 19.8 | 22.3 |

Qwen3.8 27B with MTP reaches 9.5 tok/s on code (3 drafts); on general text the number is lower. Gemma 4 12B and E4B: the first measurement used a community quantization file that was later replaced by Google's official QAT file (Q4_0). Qwen3.6 35B: the first measurement used the standard Unsloth file (12 October); the earlier 2-bit GSQ version was around 15 tok/s.

### Prefill: getting the most out of int8 compute

Prefill behaves differently from generation. Thousands of tokens are processed together, so the weights are reused many times and the limit moves from memory bandwidth to compute. This iGPU has no matrix units (XMX) and relies only on the `DP4a` instruction, four int8 multiplies in one step, with a peak of about 13 TOPS. arcinfer converts all weights to int8 (W8), quantizes the activations per 32 values, and runs them through its own DP4a GEMM kernels, which reach 6–9 TOPS.

For comparison, here are results against llama.cpp (build b11527, Vulkan and SYCL backends, flash attention on, best ubatch) with the same prompts:

| Speedup | Measurement |
|---|---|
| 6× | Qwen3.8 27B, 11,240-token prompt: 97 s, against 590 s (Vulkan) and 504 s (SYCL) |
| 7.4× | Qwen3.8 27B, 33,937-token prompt: 5.5 minutes, against 40 minutes (Vulkan) |
| 3.8× | Qwen3.5 4B pp2048: 911 tok/s, against 242 (Vulkan) and 337 (SYCL) |
| 4.9× | Gemma 4 12B, 35,957-token prompt: 240 s, against 1,165 s (Vulkan) |

#### Prefill speed pp2048 (tokens/s)

| Model | arcinfer | llama.cpp Vulkan | llama.cpp SYCL |
|---|---|---|---|
| Qwen3.5 4B | 911 | 242 | 337 |
| Qwen3.5 9B | 402 | 151 | 189 |
| Gemma 4 12B | 240 | 96 | – |
| Qwen3.8 27B | 113 | 23 | 25 |
| Qwen3.6 35B | 642 | – | – |

Source: bench/results/results.jsonl, 10 October (first test). Gemma 4 12B used a community Q4_K_M quantization file; SYCL was not tested. Qwen3.6 35B: llama.cpp cannot load the GSQ file, so only the arcinfer number for the Unsloth Q3_K_M file is shown (13 October).

The lead also shows in generation after a long prompt. At 34K tokens of context Qwen3.8 27B still generates 3.70 tok/s, against 2.06 tok/s for Vulkan and 1.09 tok/s for SYCL. The difference comes from int8 attention: keys and values are stored as int8 with a per-row scale, and both the `Q·K` scores and `P·V` also use DP4a.

### Qwen3.8 27B: time to first token down from 116 to 89 seconds

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

### Qwen3.6 35B-A3B: mixture of experts on an iGPU

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

### Accuracy: speed must not cost correctness

> An optimization that changes the output, however slightly, is not accepted.

Every optimization on the decode path is checked in two ways:

- **KL divergence against an f32 reference:** the token distribution is compared with a full-precision CPU computation. Results: Qwen3.5 4B 0.0021, Qwen3.6 35B (GSQ) 0.0008, Gemma 4 12B QAT 0.021.
- **Bit identity for MTP:** output with speculative decoding must be exactly the same as plain decoding on a 2K-token prompt.

The second check caught a well-hidden bug on the 35B. Two-token verification used an attention kernel that split the keys differently from the one-token kernel. As a result the text started to diverge after about 40 tokens, and only on long prompts. After the fix, the 35B MTP output is bit-identical again.

### Long context

Automatic context uses as much of the remaining GPU memory as possible after the weights, the work buffers and about 1.2 GiB kept free for Windows (**Later:** for MoE models with streamed experts the expert cache now comes first, see the current section). Results at the extremes:

| Model | Prompt length | Processing time | Generation afterwards |
|---|---|---|---|
| Qwen3.5 4B | 235,929 tokens | 27.3 minutes (144 tok/s) | 9.0 tok/s |
| Gemma 4 E4B | 117,965 tokens | 11.9 minutes (165 tok/s) | 11.6 tok/s |
| Qwen3.8 27B | 81,100 tokens | 16.4 minutes (82 tok/s) | 4.9 tok/s |
| Qwen3.6 35B | 21,590 tokens | 44 s (493 tok/s) | – |

When used as an agent (Pi), these numbers are rarely felt directly. The server keeps the running conversation, so each turn processes only the new tokens. The per-project system prompt is also cached on disk, so a new Qwen3.8 27B session on a project that was opened before can start in about 5 seconds, against about 40 seconds before.

### Optimizations that did not pay off

Not every idea worked. These approaches were tried and dropped, with the reason:

- **Level Zero instead of OpenCL.** The gap between kernels was only 1.6–2 %, not worth rewriting the runtime.
  **Later: overturned.** Level Zero was built anyway (arcinferzero) and became the default engine on 16 October:
  bit-identical results, MoE experts streamed from the SSD with a GPU cache (35B at long contexts, Flash Next at all),
  explicit control of memory and queues, and many kernels made faster on it since (see the 16 and 17 October
  sections). The 1.6-2 % figure measured only the launch gaps, not what the runtime made possible.
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
  **Later: overturned** by faster multi-token verification: defaults now 7 (27B), 5 (35B), 3 (4B), 2 (9B).

### Next steps

Two priorities matter most to keep long-context performance up:

- **4-bit KV cache:** halves the data read per token at deep context and doubles the maximum context, provided its accuracy passes the tests.
- **Decode the 35B experts once per layer:** today about 22 % of the 35B prefill time is spent decoding the same weights again for every prompt chunk.
  **Later:** partly addressed on 17 Oct: 4096-token chunks (decode share 29 -> 17 %), and moe_rg, which reads the raw
  expert rows without the decode pass, for 320-1279-token chunks. Decoding once per layer across chunks is still
  open (it needs per-chunk MoE buffers of ~1 GB).

Methodology note: generation numbers are the best of two runs, because the first run after another process stops tends to be slower, and the iGPU lowers its clock by about 5–8 % under sustained load. During this period some models also changed files (Gemma community quantization to Google QAT; 35B GSQ to Unsloth), and the tables note where the first measurement came from a different file.
