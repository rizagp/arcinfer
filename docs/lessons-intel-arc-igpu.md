# LLM inference on an Intel Arc iGPU: what made it fast, and what did not

Lessons from building arcinfer, a from-scratch inference engine for the integrated Arc GPU of a Meteor Lake laptop,
October 2026. Every number below was measured on one machine; your hardware, driver and models will differ, so treat
them as starting points, not promises. Sources: [CHANGELOG.md](../CHANGELOG.md), [review_notes.md](review_notes.md),
[arcinferzero-log.md](arcinferzero-log.md).

**Machine:** Intel Core Ultra 5 125H, Arc iGPU (Xe-LPG, 7 Xe-cores / 112 EUs, **no XMX matrix engines**), 32 GB
LPDDR5x-6400 shared with the CPU, Windows 11, Intel driver 32.0.101.9033. Engines: OpenCL 3.0, later the same OpenCL C
kernels run through Level Zero.

## 1. Know the hardware first (measured)

| What | Measured | Consequence |
|---|---|---|
| int8 dot products (DP4a, signed x signed, broadcast operand) | 11–13 TOPS peak | the fastest arithmetic this GPU has: build the matrix products on it |
| FP16 / FP32 FMA | 7.3 / 3.9 TFLOPS | fp16 GEMM is about half of int8 |
| GPU memory reads | 75–77 GB/s | generation can never beat "read every weight once per token" |
| XMX / DPAS / 2D block IO | **not present** on Xe-LPG | tutorials and tools built for Arc discrete GPUs, Lunar Lake or Panther Lake (XMX tiles, joint_matrix) do not apply |
| GPU cache | ±4 MiB (random reads ±900 GB/s up to 4 MiB, RAM speed from ±32 MiB) | keep data that many work-groups reuse under ~4 MiB |
| GPU page size | 2 MiB for large allocations | random 256-byte reads spread over more than ~1 GiB drop to ±21 GB/s; reads of >= 4 KiB contiguous stay at ±75 GB/s |
| Level Zero virtual memory (`zeVirtualMemMap`) | refused on this Windows driver | page-level tricks (custom pages, read-only mappings) are not available |
| Sustained load | 5–8 % slower after minutes (heat) | benchmark with idle gaps; take the best of two runs |

Measure these yourself (`bench/peak_bench`, `bench/bw_bench`, `zero/page_probe.cpp`) before believing any document,
including generated "blueprints" that quote bandwidths or features this chip does not have.

## 2. What made prompts fast

Prompt processing (time to first token) is compute-bound; on this iGPU llama.cpp's backends were the bottleneck.

| Change | Effect |
|---|---|
| **int8 DP4a GEMM** (activations int8 with a scale per 32, weights int8) instead of fp16 | ~6–9 TOPS vs ~1.5 TOPS effective in llama.cpp's backends; prompts 2.1–4.9x faster than llama.cpp Vulkan / SYCL (table below) |
| **Just-in-time weight decoding** of every GGUF type into one int8 "W8" layout | one tuned GEMM serves all IQ / K quant types |
| Requantizing types with block minimums (Q4_K, Q5_K, Q2_K) to signed int8 | Q4_K GEMM 6.1 -> 8.8 TOPS; the exact "scales + mins" kernel was 3x slower and not more accurate end to end |
| Work-groups sharing weight tiles across 8 token tiles | 4.1 -> 6.7 TOPS |
| Split-K for long reductions (K > 8192) so the activation slab stays in cache | 7.3 -> 8.8 TOPS |
| **Layer-major prefill** (all chunks of the prompt through a layer before the next) | each matrix decoded once per prompt instead of once per chunk: W8 decoding 6.9 -> 2.2 s (27B, 11K tokens) |
| int8 attention end to end (int8 K/V cache in 16-key blocks, DP4a scores, int8 P·V, log2 softmax with lazy rescaling) | attention 33 -> 5.1 s (27B, 11K tokens) |
| Gemma 4: an int8 scale per 64 dims for q/K/V instead of per row | KL on code 0.23 -> 0.018 (a few large V dims swamped the row scale) |
| Splitting head dims over sub-groups where registers spilled (Gemma head 512) | 1.25–1.4x |
| **Large register file (`-cl-intel-256-GRF-per-thread`) on kernels that spilled** | Qwen prompt attention, last 4K-token chunk at 64K depth: 1,276 -> 926 ms per layer; Gemma E4B prompts 733 -> 827 tok/s after removing the last big spills |
| Gated DeltaNet recurrence with a whole 128-float state column per SIMD8 lane | 424 -> 145 ms (4B, 2.6K tokens) |
| Vision encoders on the GPU (tiled attention, native_exp) | 1008-token image 1118 -> 588 ms |

Prompt speed, tokens/s (10 October, OpenCL engine, same laptop; llama.cpp b11527 official builds, flash attention,
best micro-batch per model):

| Model | arcinfer pp2048 | llama.cpp Vulkan | llama.cpp SYCL |
|---|---|---|---|
| Qwen3.5 4B | 911 | 242 | 337 |
| Qwen3.5 9B | 402 | 151 | 189 |
| Qwen3.8 27B | 113 | 23 | 25 |
| Gemma 4 12B | 240 | 96 | — |

Generation speed was similar to llama.cpp at that point (4B 17.6 vs 16.2 / 17.9 tok/s); it improved later (section 3).

## 3. What made generation fast

Generation is memory-bound: the goal is reading the weights at close to 75 GB/s, then reading them fewer times.

| Change | Effect |
|---|---|
| GEMVs straight on the raw GGUF blocks (no decode pass), block reads, codebooks in local memory | 11–49 -> 57–77 GB/s |
| Raw blocks regrouped once at load for aligned vector loads (Q8_0 scales first; IQ2/IQ3 scale moved last) | Q8_0 GEMV 37 -> 77 GB/s, IQ2_XXS 27 -> 60, IQ3_XXS 41 -> 62 |
| IQ4_XS: lanes vector-load their own bytes; 16-entry table pre-shifted per byte | E4B attention q 44 -> 67 GB/s, output head 48 -> 69; E4B 17 -> 21 tok/s |
| Allocating the output head before all other weights | a late allocation read at half speed (30 GB/s) |
| Fusing small per-token kernels (residual + norms + quantization), together with the IQ4_XS GEMVs above | E4B GPU time 56.5 -> 46.6 ms per token |
| Flash-decoding (keys split over work-groups) and splitting head dims across sub-groups | Gemma head 512 at 14K: 0.74 -> 0.31 ms per layer; Qwen3.6 35B at 64K: 1.16 -> 0.94 ms |
| **Speculative decoding with the model's own MTP head** (Qwen) or Gemma's assistant drafter, verified in one pass; output unchanged | Qwen3.6 35B 19.9 -> 31.5 tok/s (code), Qwen3.8 27B 5.1 -> 9.1, Gemma E4B 21.6 -> 39.6, 12B 9.8 -> 21.1 |
| Verification GEMVs loading each token's activations where used (instead of holding 4–8 tokens in registers) | 27B with 5 drafts 3.8 -> 9.2 tok/s (the register version spilled) |
| Rollback of rejected drafts by replaying the recurrence over the accepted tokens (one state copy, not one per draft) | 1–7 drafts with identical output, no 150 MB copies per step |

Per-token share of the bandwidth reached: Qwen3.5 4B 79 %, 9B 86 %, Gemma E4B 81 %, 12B 82 %, Qwen3.8 27B 72 %.
The 27B's remaining gap is decoding cost: its IQ3_S / IQ3_XXS / IQ2 weights read at 51–55 GB/s-equivalent while
IQ4_XS / Q4_K reach ±62 in the same profile; codebook types need more arithmetic per byte than this GPU hides.

## 4. Fitting big models and long contexts

| Change | Effect |
|---|---|
| **MoE expert streaming from the SSD** (Level Zero): only the experts in use stay in a GPU cache (LRU, prefetch while the GPU works) | Qwen3.6 35B-A3B at the full 262K context together with images and MTP (the all-resident engine fit ~40K), Windows keeps 14 GB of RAM |
| Prompt chunks of 4,096 with the experts of a chunk streamed through a separate buffer | a long prompt does not sweep the cache that generation uses |
| Automatic context from the GPU memory the driver currently offers | the budget changes with other apps; computing it per start avoids out-of-memory crashes |
| Prefix cache (system prompt state on disk per project), session cache (whole conversations to disk), a snapshot at the start of the last assistant turn | a new message in a long Pi session: 5.2 s instead of 16.8 s (9B); a new session reuses the system prompt in ~5 s instead of ~40 s (27B) |

## 5. Tried and not kept

| Idea | Result |
|---|---|
| fp16 P·V accumulation in attention | no speed-up, worse error |
| SIMD8 W8 GEMM (its DP4a microbenchmark peaks at 13.7 TOPS vs 13.2 for SIMD16) | 28 % slower: 27B ffn_gate 7.90 -> 5.69 TOPS, ffn_down 6.62 -> 4.74 (re-measured 16 Oct). The ALUs are 8 lanes wide, so SIMD16 loses almost nothing in pure arithmetic, while the GEMM is limited by the loads, lane broadcasts and scale math around the DP4a, which SIMD8 needs twice as many instructions for. A peak microbenchmark measures the part that is not the bottleneck |
| Software-pipelined GEMM loop, 32 tokens per sub-group, 24-bit multiplies | 7.0, 2.6 TOPS, no change |
| SLM staging of q/k for the DeltaNet recurrence | 4x slower (local memory bandwidth) |
| Several tokens per sub-group in the DeltaNet conv | 3x slower (too few sub-groups in flight) |
| 256-entry byte-pair table for IQ4_XS | slower (local-memory bank conflicts, 66 -> 55 GB/s) |
| Split-K GEMV for small matrices | 10–15 % faster on 512-row matrices, slower on large ones: no gain overall |
| 256 registers for kernels with small spills (DeltaNet conv 1 KB, MoE gate/up 2 KB) | spill gone, not faster: fewer threads in flight cost as much |
| Splitting head dims for Qwen prompt attention (head 256) once 256 registers were used | slower than the plain kernel at every shape (926 vs 1,117 ms) |
| More than 3 MTP drafts per step | not faster: each draft runs the MTP block with the full output head |
| A smaller draft output head (first N token ids) | faster for English code, slower for Indonesian (its tokens lie beyond 65,536) |
| int8 vision encoder weights and activations | 10–23 % off the f32 reference: activations must stay f32 |
| Opting out of Windows power throttling (EcoQoS) | no measurable effect |
| Larger GPU pages / Level Zero virtual memory | the driver already uses 2 MiB pages; mapping physical memory is refused |

## 6. Mistakes worth avoiding

- **Check what the compiler actually received.** I once concluded that Level Zero ignores the 256-register flag;
  my own environment switch had come out empty, so the flag was never passed. Print each kernel's build options and
  spill size (`ARC_ZE_KPROPS=1`) before blaming a driver.
- **An identical test is not proof of a memory-ordering hazard's absence.** Reading back values a work-item just
  wrote passed every test but can return stale cache lines; keep such values in registers.
- **Do not measure next to another GPU process.** A CPU reference running alongside, a second server, or a server
  started while Windows was still releasing the previous one's memory (half speed for its whole life) all distorted
  numbers. Wait a few seconds between runs, alternate A/B runs, and take the best of two.
- **Small-matrix benchmarks lie about GB/s**: kernel launch time dominates; measure on matrices that stream from memory.
- **Logit differences at ~0.0003 KL are noise** from int8 rounding flips (±15–30 % run to run); judge accuracy over
  many positions and prompts.

## 7. Correctness discipline that kept this sane

- An independent f32 CPU reference for every architecture (`tools/ref_forward`), compared by KL over all prompt
  positions, and greedy output compared with llama.cpp.
- Every kernel change either bit-identical to the previous one (checked by dumping logits) or measured against the
  reference; Level Zero and OpenCL engines produce bit-identical logits on every model.
- Speculative decoding only accepted when the answer is identical to plain decoding.
- Per-kernel tests on real model tensors against the CPU dequantization (`tools/test_gemm`, `bench/gemv_bench`).
