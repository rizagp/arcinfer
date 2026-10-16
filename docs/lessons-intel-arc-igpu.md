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
| Attention P·V: per-head divisions done once (lane r for head r) and the DP4a accumulating in place (`dot_acc_sat`, heads interleaved), found by reading the assembly with ocloc | same values; last 4K chunk at 64K depth: 4B/9B/35B shape 937 -> 775 ms per layer (spill 576 -> 0 B), 27B 1515 -> 1335 ms |
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
| **DFlash2 block drafter** (8 rows per draft pass, 7 drafts verified at once; switches to the MTP block when its drafts are rejected) | 27B Python code 4.85 -> 11.5 tok/s (MTP alone ~8.9; 89 % of drafts taken); 35B code 15.7 -> 44.4 (MTP 31.5; 98 % taken); Indonesian prose ~MTP (27B 5.6, 35B 22.5: the drafters are weak outside English / code) |
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
| Expert cache with one pool of the largest record size when a model mixes more than two record sizes (Qwen3.8 Flash Next: 512 experts, 5 type mixes) | separate pools each needed a whole layer and ran out of GPU memory; one pool runs it with a 7.9 GiB cache (12 % of a 56 GB expert file), ~4 tok/s |
| Prompt chunks of 4,096 with the experts of a chunk streamed through a separate buffer | a long prompt does not sweep the cache that generation uses |
| Automatic context from the GPU memory the driver currently offers | the budget changes with other apps; computing it per start avoids out-of-memory crashes |
| Prefix cache (system prompt state on disk per project), session cache (whole conversations to disk), a snapshot at the start of the last assistant turn | a new message in a long Pi session: 5.2 s instead of 16.8 s (9B); a new session reuses the system prompt in ~5 s instead of ~40 s (27B) |

## 5. Tried and not kept

| Idea | Result |
|---|---|
| fp16 P·V accumulation in attention | no speed-up, worse error |
| SIMD8 W8 GEMM (its DP4a microbenchmark peaks at 13.7 TOPS vs 13.2 for SIMD16) | 28 % slower: 27B ffn_gate 7.90 -> 5.69 TOPS, ffn_down 6.62 -> 4.74 (re-measured 16 Oct). The ALUs are 8 lanes wide, so SIMD16 loses almost nothing in pure arithmetic, while the GEMM is limited by the loads, lane broadcasts and scale math around the DP4a, which SIMD8 needs twice as many instructions for. A peak microbenchmark measures the part that is not the bottleneck |
| Software-pipelined GEMM loop, 32 tokens per sub-group, 24-bit multiplies | 7.0, 2.6 TOPS, no change |
| 32 tokens per sub-group again with 256 registers (no spill), or two column groups per work-group | still 2.4 vs 8.1 TOPS (27B ffn_gate, 16 Oct); CG=2 within 1 %: the slowdown was never the spill, larger tiles leave too few threads in flight |
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
| LSC prefetch of the next K/V tiles in decode attention (`prefetch()`, 1–4 tiles ahead) | 37 % slower (27B, 32K: 1.12 -> 1.53 ms per layer): the extra messages compete with the real loads |
| Smaller or larger key splits for decode attention (64–512 keys per sub-group, to fill the last wave) | within 3 % at every shape (27B, 4B, 35B at 32K); the wave-count estimate from the assembly did not hold |
| IQ2/IQ3 GEMV: sign byte computed (`i | parity(i) << 7`) instead of read from a local-memory table, and the sign mask without the 32-bit multiply | gemv_v (the kernel the engine uses for these types) equal or slower: IQ2_XS 63 -> 60 GB/s; only the unused plain gemv got faster. Fewer instructions again did not mean faster |
| Prefetching the next layer's experts from a routing guess (its router on this layer's FFN input) | Flash Next, 16 % cache: 37 % of the guesses used, 48 % of the misses caught, SSD reads +55 %, no faster (the SSD is the limit, and wrong guesses take its time) |
| A per-layer LRU for the expert cache (against the cyclic layer order of decoding) | a recorded Flash Next trace: 80 % hits vs 82 % for the global LRU (Belady's optimum: 90 %) |
| Layer-major prompt segments of 32K instead of 16K for narrow models (experts decoded once per segment) | 35B, 40K prompt: 530 -> 521 tok/s (the larger residual buffer costs expert-cache room) |
| Double-buffered K/V tiles, 8 or 32 tokens per work-group in prompt attention | within 4 % (35B shape at 32K) |
| Running the next expert group's decode to W8 beside the current group's MoE GEMM, on the second compute engine (Level Zero queue group 1) | a 1 GiB stream kernel and a DP4a kernel: 109 ms one after the other, 89 ms on two engines; the MoE prompt GEMM itself is ALU-bound (35B, 4K chunk: 930 -> 675 ms with no loads at all), so at most ~7 % back for the cross-engine synchronization |
| A draft head over a fixed vocabulary prefix (`ARC_DRAFT_VOCAB`) with 7 MTP drafts | 27B: first 131,072 ids, code 15.1 -> 16.3 tok/s but Indonesian 6.9 -> 5.8 (its words lie beyond id 131,072); 65,536 slower for both |
| Other MTP confidence cut-offs (`--mtp-min-p`) | 27B, 0.3 / 0.5 / 0.7 / 0.85: code 14.6 / 15.3 / 14.0 / 13.9, Indonesian 6.6 / 6.9 / 6.8 / 6.6 tok/s: 0.5 stays |
| Loading the DFlash2 drafter before the main weights | 27B: its lm_head back from 21 to 10 ms per pass, but other weights slowed and the wall-clock speed was no better (6.1-6.3 vs 6.6 tok/s); which buffers land on slow pages varies with the load (a 35B hybrid run is ~20-30 % slower about one time in six, with identical work) |
| Larger GPU pages / Level Zero virtual memory | the driver already uses 2 MiB pages; mapping physical memory is refused |

## 6. Compiler flags that this driver honors (tested 16 Oct, driver 32.0.101.9033)

Options go to clBuildProgram (the same binaries then run through Level Zero). The driver rejects unknown options
with a build error, so "accepted" means recognized. Tested on a kernel that spills 1.7 KB at the default:

| Option | Accepted | Effect seen |
|---|---|---|
| `-cl-intel-256-GRF-per-thread` / `-ze-opt-large-register-file` | yes | spill 1728 -> 0 B; 4 threads per EU instead of 7 |
| `-cl-intel-reqd-eu-thread-count=4` | yes | same as 256 GRF (spill 0); 5, 6, 7, 8 behave like the default: Xe-LPG has only two register modes, no middle ground |
| `-cl-intel-128-GRF-per-thread`, `-cl-intel-num-thread-per-eu N` | yes | the default |
| `-cl-intel-enable-auto-large-GRF-mode` | yes | no change: the heuristic kept 128 GRF on a spilling kernel; choose per kernel yourself |
| `-cl-intel-no-prera-scheduling`, `-cl-uniform-work-group-size`, `-cl-no-subgroup-ifp`, `-cl-fast-relaxed-math` | yes | no change in spill; speed effect must be measured per kernel (`-cl-fast-relaxed-math` changes results) |
| `-cl-intel-greater-than-4GB-buffer-required` | yes | slightly different code (spill 1728 -> 1664 B) |
| `-igc_opts '...'` | yes | passes internal IGC flags; the dump flags (`ShaderDumpEnable`) do nothing on the release driver |
| `-cl-intel-fail-on-spill`, `-cl-intel-no-spill`, `-cl-intel-exp-register-file-size=N` | **rejected** | internal-only options |
| `IGC_*` environment variables | ignored | no shader dumps on the release driver |

Sub-group size is fixed in the source with `__attribute__((intel_reqd_sub_group_size(16)))`. To read the generated
assembly you need Intel's offline compiler `ocloc` (not shipped with the driver). Practical rule: keep 128 GRF where a
kernel does not spill (more threads hide memory latency); use 256 GRF per kernel where it spills and measure, since a
small spill can cost less than losing 3 of 7 threads per EU.

**Reading the assembly** (`tools/kernel_isa.py`: Intel's `ocloc` compiles for Meteor Lake and disassembles; the
driver's own binaries in arcinfer's kernel cache can be disassembled the same way, and they differ from ocloc's newer
compiler). The prompt GEMM's hot loop (W8 per-32 scales, 256 GRF) is 1,957 instructions per superblock, 52 % DP4a;
136 `mach` + extra `mul` come from `sc * acc`, an int32 multiply this GPU emulates. Removing it (a 16-bit scale, or
the multiply done in float, both bit-identical) cut the loop to 1,751 instructions — and made the GEMM **3x slower**
(8.2 -> 2.5-2.8 TOPS): the driver's scheduler then interleaved only 2 independent DP4a chains instead of ~5, so each
DP4a waited on the one before it (`{I@2}` instead of `{I@5}` dependencies). Instruction count is not the metric;
the number of independent chains in flight is. Any GEMM change has to be measured, not counted.
Why the scheduler did it: for this large unrolled loop it kept the **source order** (one token at a time: one
activation load, its 16 DP4a, its scale math), so only 2 chains and 1-3 loads were in flight. Writing the interleave
into the source fixes it: load the activation words of 16 tokens first, then run their dot products side by side
(32 chains), then each token's scale math. With the 16-bit scale multiply this gives 1,742 instructions, ~900 of
1,024 DP4a without a dependency wait, no spill, and is bit-identical: 27B GEMM 8.05 -> 8.30 TOPS (gate), 6.62 -> 7.0
(down), 27B prompt 124 -> 127 tok/s; 35B unchanged (its prompt is not GEMM-bound). Rule: if the compiler keeps
source order, the source order must already be the schedule you want.

## 7. Mistakes worth avoiding

- **Check what the compiler actually received.** I once concluded that Level Zero ignores the 256-register flag;
  my own environment switch had come out empty, so the flag was never passed. Print each kernel's build options and
  spill size (`ARC_ZE_KPROPS=1`) before blaming a driver.
- **An identical test is not proof of a memory-ordering hazard's absence.** Reading back values a work-item just
  wrote passed every test but can return stale cache lines; keep such values in registers.
- **Do not measure next to another GPU process.** A CPU reference running alongside, a second server, or a server
  started while Windows was still releasing the previous one's memory (half speed for its whole life) all distorted
  numbers. Wait a few seconds between runs, alternate A/B runs, and take the best of two.
- **The first run after a kernel source change includes JIT compiles.** A 27B generation right after a 32K prompt
  took 18.4 s for its first step (a dozen decode-attention variants compiled), then 238 ms per step; averages over
  short generations hide this, so report the median step (runz prints it).
- **Small-matrix benchmarks lie about GB/s**: kernel launch time dominates; measure on matrices that stream from memory.
- **Logit differences at ~0.0003 KL are noise** from int8 rounding flips (±15–30 % run to run); judge accuracy over
  many positions and prompts.

## 8. Correctness discipline that kept this sane

- An independent f32 CPU reference for every architecture (`tools/ref_forward`), compared by KL over all prompt
  positions, and greedy output compared with llama.cpp.
- Every kernel change either bit-identical to the previous one (checked by dumping logits) or measured against the
  reference; Level Zero and OpenCL engines produce bit-identical logits on every model.
- Speculative decoding only accepted when the answer is identical to plain decoding.
- Per-kernel tests on real model tensors against the CPU dequantization (`tools/test_gemm`, `bench/gemv_bench`).
