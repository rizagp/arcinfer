# LLM inference on an Intel Arc iGPU: what made it fast, and what did not

Lessons from building arcinfer, a from-scratch inference engine for the integrated Arc GPU of a Meteor Lake laptop,
from the first working forward pass (early October 2026) to the Level Zero engine of 17 October 2026. Every number
below was measured on one machine; your hardware, driver and models will differ, so treat them as starting points,
not promises. Sources: [CHANGELOG.md](../CHANGELOG.md), [review.md](review.md), [review_notes.md](review_notes.md),
[speed-progress.md](speed-progress.md), [arcinferzero-log.md](arcinferzero-log.md), [QUEUE.md](../QUEUE.md); how to
run it: [USAGE.md](../USAGE.md).

**Machine:** Intel Core Ultra 5 125H, Arc iGPU (Xe-LPG, 7 Xe-cores / 112 EUs at 2.2 GHz, **no XMX matrix engines**),
32 GB LPDDR5x-6400 shared with the CPU, Windows 11, Intel driver 32.0.101.9033. Engines: OpenCL 3.0 first, then
(since 16 October the default) the same OpenCL C kernels run through Level Zero, with MoE experts streamed from the
SSD.

**Where it ended up (17 October, Level Zero engine; greedy answers identical to plain decoding):**

| Model | Prompt | Generation: plain / MTP / DFlash2 + MTP (Python code) |
|---|---|---|
| Qwen3.8 27B (IQ3_S GSQ, dense) | pp4096 ~135 tok/s (llama.cpp Vulkan pp2048: 23) | 4.91 / 15.5 / 18.1 tok/s |
| Qwen3.6 35B-A3B (UD-Q3_K_M, MoE) | pp4096 ~795 tok/s | 21.2 / 39.3 / 49.8 tok/s |
| Qwen3.8 Flash Next (512 experts streamed from the SSD) | pp8192 ~165 tok/s | 7.3 tok/s |

On Indonesian prose the drafters help far less (27B 4.88 / 7.0 / 6.8, 35B 21.0 / 22.7 / 22.7 tok/s): they are
trained mostly on English and code.

## 1. Know the hardware first (measured)

| What | Measured | Consequence |
|---|---|---|
| int8 dot products (DP4a, signed x signed, broadcast operand) | 11.2–13.3 TOPS peak, 13.0 sustained for 2 minutes; signed x unsigned only 9.1 | the fastest arithmetic this GPU has: build the matrix products on it, in the signed x signed form |
| FP16 / FP32 FMA | 7.32 / 3.88 TFLOPS | fp16 GEMM is about half of int8 |
| GPU memory reads | 75–77 GB/s (102 GB/s theoretical) | generation can never beat "read every weight once per token" (27B, ~11 GB: ~6.5 tok/s without drafting) |
| XMX / DPAS / 2D block IO | **not present** on Xe-LPG | tutorials and tools built for Arc discrete GPUs, Lunar Lake or Panther Lake (XMX tiles, joint_matrix) do not apply |
| GPU cache | ±4 MiB (random reads ±900 GB/s up to 4 MiB, RAM speed from ±32 MiB) | keep data that many work-groups reuse under ~4 MiB |
| GPU page size | 2 MiB for large allocations (64 KiB for small ones) | random 256-byte reads spread over more than ~1 GiB drop to ±21 GB/s; reads of >= 4 KiB contiguous stay at ±75 GB/s |
| Level Zero virtual memory (`zeVirtualMemMap`) | refused on this Windows driver | page-level tricks (custom pages, read-only mappings, paged KV through page tables) are not available; use block tables in the kernel |
| Host memory used by the GPU beyond its share | 9–21 GB/s (Windows moves it around) | keep everything hot inside the GPU share |
| GPU memory share | a driver setting: 16.5, 18.34, 20 (Windows became unresponsive), 17.44, 18.3 GiB over these weeks; the desktop and apps hold 0.7–1.0 GiB of it, the driver ±0.22 GiB on top | compute the context at every start; keep ~1.4 GiB free for the driver (1.13 GiB left: out of device memory at kernel launch); Windows needs >= 14 GB of the 32 GB RAM |
| NVMe SSD (Kingston) | random 1–4 MB reads 3.5–4.2 GB/s with 16 requests in flight; random 2 MB reads top out at 3.6–3.8 GB/s at any queue depth | the ceiling for anything streamed from disk (MoE experts) |
| Sustained load | 5–8 % slower after minutes (heat); back-to-back runs 5–10 % slower than the first | benchmark with idle gaps (30 s before each measurement); take the best of two runs |
| The iGPU is shared with every process | E4B 23 tok/s alone, 8 tok/s while a 27B server processed a prompt | nothing else on the GPU while measuring |

Measure these yourself (`bench/peak_bench`, `bench/bw_bench`, `bench/ssd_bench`, `zero/page_probe.cpp`) before
believing any document, including generated "blueprints" that quote bandwidths or features this chip does not have.

## 2. What made prompts fast

Prompt processing (time to first token) is compute-bound; on this iGPU llama.cpp's backends were the bottleneck. The
yardstick through most of the work was the 27B's 11,240-token code prompt: 116 s in the first working version, ~89 s
after the steps below (llama.cpp: 590 s Vulkan, 504 s SYCL).

| Change | Effect |
|---|---|
| **int8 DP4a GEMM** (activations int8 with a scale per 32, weights int8) instead of fp16 | ~6–9 TOPS vs ~1.5 TOPS effective in llama.cpp's backends; prompts 2.1–4.9x faster than llama.cpp Vulkan / SYCL (table below); per-32 activation scales instead of coarser ones: KL 0.0014 -> 0.00008 |
| **Just-in-time weight decoding** of every GGUF type into one int8 "W8" layout | one tuned GEMM serves all IQ / K quant types, and a 27B only has to fit in its compressed form |
| Requantizing types with block minimums (Q4_K, Q5_K, Q2_K) to signed int8 | Q4_K GEMM 6.1 -> 8.8 TOPS (Qwen3.5 4B pp2048 662 -> ~900 tok/s); the exact "scales + mins" kernel ran 2.4 TOPS, 3x slower, and was not more accurate end to end (perplexity 5.4759 -> 5.4779) |
| Work-groups sharing weight tiles across 8 token tiles | 4.1 -> 6.7 TOPS |
| Split-K for long reductions (K > 8192) so the activation slab stays in cache | 7.3 -> 8.8 TOPS |
| Chunk rows padded to 128 | avoids a 40 % slow path on the last prompt chunk |
| **Layer-major prefill** (all chunks of the prompt through a layer before the next; segments of 16K tokens bound the residual buffer) | each matrix decoded once per segment instead of once per chunk: W8 decoding 6.9 -> 2.2 s (27B, 11K tokens) |
| W8 decoding: each superblock vector-loaded into registers (the per-32-block kernel read global memory byte by byte) | 27B 11K prompt, IQ3_S / IQ3_XXS / IQ4_XS decoding 5.8 -> 2.2 s, bit-identical |
| Decoded weights kept resident when the context still fits (>= min(128K, trained)) instead of a fixed budget | Qwen3.5 9B: 83-token prompt 1.05 -> 0.27 s to the first token |
| Fused SwiGLU in the up-projection GEMM epilogue; vectorized DeltaNet conv | swiglu 2.5 -> 1.3 s, conv 2.0 -> 1.0 s (27B, 11K) |
| int8 attention end to end (int8 K/V cache in 16-key blocks, DP4a scores, int8 P·V with P per 16-key tile, heads split over sub-groups, log2 softmax with lazy rescaling) | attention 33 -> 5.1 s (27B, 11K tokens) |
| Gemma 4: an int8 scale per 64 dims for q/K/V instead of per row | KL on code 0.23 -> 0.018 (a few large V dims swamped the row scale); costs Gemma's attention 1.5–1.8x |
| Gemma head-512 prompt attention: K/V scales read from L1 instead of local memory (32 KB of tiles + scales allowed one work-group per Xe-core) | 1,127 -> 649 ms per layer (11K tokens) |
| Splitting head dims over sub-groups where registers spilled (`attn_ps`, Gemma head 512) | 1.25–1.4x (12B global layers); E4B 13,921-token prompt 23.5 -> 21.8 s |
| Qwen prompt attention: 4 q heads per sub-group where the GQA ratio allows | 21,590-token prompt: 35B 9.4 -> 8.4 s, 9B 8.2 -> 7.0 s of attention |
| **Large register file (`-cl-intel-256-GRF-per-thread`) on kernels that spilled** | Qwen prompt attention, last 4K-token chunk at 64K depth: 1,276 -> 926 ms per layer; Gemma E4B prompts 733 -> 827 tok/s after removing the last big spills |
| Attention P·V: per-head divisions done once (lane r for head r) and the DP4a accumulating in place (`dot_acc_sat`, heads interleaved), found by reading the assembly with ocloc | same values; last 4K chunk at 64K depth: 4B/9B/35B shape 937 -> 775 ms per layer (spill 576 -> 0 B), 27B 1515 -> 1335 ms |
| GEMM loop with 16 tokens' dot products side by side in the source (section 6) | bit-identical, GEMM +3–7 %; 27B prompt 2K 110 -> 130, 16K 103 -> 125 tok/s |
| Gated DeltaNet recurrence with a whole 128-float state column per SIMD8 lane | 424 -> 145 ms (4B, 2.6K tokens) |
| **MoE prompts through the int8 GEMM**: per group of 64 experts the raw rows decoded to W8, then one GEMM over that group's slot tiles (chunks >= 1152 tokens) | Qwen3.6 35B 2048-token prompt 476 -> 642 tok/s |
| W8 decoding of whole expert rows staged in local memory with coalesced loads (`decode_w8_sl`) | IQ3_XXS 1038 -> 518 ms, IQ4_XS 433 -> 303 ms per 2048-token chunk; Q8_0 through its vector block decoder 211 -> 72 ms |
| **moe_rg**: the MoE GEMM on raw expert rows, each lane decoding its row's superblocks in registers (320–1279-token chunks) | equals decode + GEMM bit for bit, without the decode pass; 35B 384–1152-token prompts +3–14 %, and its logits no longer depend on the chunk size |
| 4096-token chunks for streamed MoE (the experts a chunk misses are read once per chunk) | 35B 8K prompt 620 -> 750 tok/s |
| The same kernels through Level Zero instead of OpenCL | bit-identical, 35B time to first token 6.45 -> 4.07 s, generation 16.9 -> 18.3 tok/s, model load 4.3 -> 2.3 s (4B) |
| Flash Next prompt mixer (gated mean) with vector loads instead of one value per lane per load | 45 -> 3.7 ms per 2048-row chunk; 8K prompt 98 -> 127 tok/s (with moe_rg for its down projection: 165) |
| Vision encoders on the GPU in f16 weights / f32 activations, attention with one query row per lane, 4-key tiles, native_exp | 1008-token image 1118 -> 588 ms; 4096 tokens 36.1 -> 15.3 s for the whole encoder (llama.cpp Vulkan 71 s); 72-dim heads with interleaved sub-group block reads 6.0 -> 1.6 s |

Prompt speed, tokens/s (10 October, OpenCL engine, same laptop; llama.cpp b11527 official builds, flash attention,
best micro-batch per model):

| Model | arcinfer pp2048 | llama.cpp Vulkan | llama.cpp SYCL |
|---|---|---|---|
| Qwen3.5 4B | 911 | 242 | 337 |
| Qwen3.5 9B | 402 | 151 | 189 |
| Qwen3.8 27B | 113 | 23 | 25 |
| Gemma 4 12B | 240 | 96 | — |

Long prompts widened the gap: the 27B's 33,937-token prompt took 5.5 minutes against 40 minutes on Vulkan, and
generation at 34K context ran 3.70 tok/s against 2.06 (Vulkan) and 1.09 (SYCL), thanks to int8 attention.

Where the ceiling is: the GEMM alone measures 8.3–8.6 TOPS (per-32 scales), 9.7 without loads and 11.0 without loads
or the scale epilogue, against 13.2 for the bare DP4a; in a long prompt it runs ~6.8–7.7 TOPS (heat, per-16-scale
types, the kernels in between). The 27B's 11K prompt never reached the original under-a-minute goal: the GEMMs alone
take ~70 s, and 60 s would need more than 90 % of the DP4a peak. The 35B MoE prompt GEMM is ALU-bound too (4K chunk
930 -> 675 ms with no loads at all).

## 3. What made generation fast

Generation is memory-bound: the goal is reading the weights at close to 75 GB/s, then reading them fewer times.

| Change | Effect |
|---|---|
| GEMVs straight on the raw GGUF blocks (no decode pass), block reads, codebooks in local memory | 11–49 -> 57–77 GB/s |
| Raw blocks regrouped once at load for aligned vector loads (Q8_0 / Q4_0 scales first; IQ2/IQ3 scale moved last) | Q8_0 GEMV 37 -> 77 GB/s, IQ2_XXS 27 -> 60, IQ3_XXS 41 -> 62, IQ3_S 61 -> 66; 27B 36 % -> 72 % of the bandwidth over the whole project |
| IQ4_XS: lanes vector-load their own bytes (no local-memory window, no barriers); 16-entry table pre-shifted per byte, so a packed word is four lookups ORed | ablations showed decoding cost as much as reading; E4B attention q 44 -> 67 GB/s, output head 48 -> 69, Q6_K output heads 66-67 -> 72; E4B 17 -> 21 tok/s |
| Allocating the output head before all other weights | a late allocation read at half speed (Gemma 12B head: 21–28 ms in the engine vs 13–14 ms alone) |
| Fusing small per-token kernels (residual + norms + quantization; Gemma's gate + up + GELU + quantization in one GEMV), together with the IQ4_XS GEMVs above | E4B GPU time 56.5 -> 46.6 ms per token |
| Gemma 12B: RoPE angles once per pass from a table (attn_prep 3.4 -> 1.9 ms per token), one sub-group per 256-value block in swiglu_quant (2.4 -> 1.2 ms), with the head-first allocation and the soft-capping on the GPU | GPU time per token 139–144 -> 123 ms |
| Stacking small matrices of one input into one GEMV (views, nothing stored twice): router + shared-expert gate/up; DeltaNet qkv + z (+ alpha / beta) | each launch has a fixed cost: 35B 17.4–18.5 -> 18.2–18.8 tok/s, then 19.2 -> 19.8 |
| Gemma's logit soft-capping in the output GEMV's epilogue; the server sampler's top-k from tokens within 40 temperatures of the best logit (one pass, same distribution) | no host loop over 262K logits per token (whose OpenMP threads kept spinning between tokens) |
| Flash-decoding (keys split over work-groups), splitting head dims across sub-groups, single-buffered K/V tiles (twice the work-groups per Xe-core), a branch-free split combine (26 -> 5 µs) | Gemma head 512 at 14K: 0.74 -> 0.31 ms per layer (13.9K, global layers: 10.7 -> 3.7 ms per token); Qwen3.6 35B at 64K: 1.16 -> 0.94 ms |
| DeltaNet recurrence state moved as vload16 / vstore16 (16 x 32-byte messages in flight instead of 32 x 16-byte in two rounds), the first token's operands requested before the state | bit-identical; 35B gdn_recur 316 -> 226–254 ms per 96 tokens (-20–28 % at T = 1) |
| **Speculative decoding with the model's own MTP head** (Qwen) or Gemma's assistant drafter, verified in one pass; output unchanged | Qwen3.6 35B 19.9 -> 31.5 tok/s (code, 3 drafts), Qwen3.8 27B 5.1 -> 9.1, Gemma E4B 21.6 -> 39.6, 12B 9.8 -> 21.1 (16 October) |
| Rollback of rejected drafts by replaying the recurrence over the accepted tokens (one state copy, not one per draft) | 1–7 drafts with identical output, no 150 MB copies per step |
| Verification GEMVs loading each token's activations where used (instead of holding 4–8 tokens in registers) | 27B with 5 drafts 3.8 -> 9.2 tok/s (the register version spilled up to 9 KB per thread) |
| **Token-outer multi-token GEMVs**: each token's activations loaded once for all of a sub-group's rows, no per-token branches, one DP4a chain per row, built for exactly T tokens with 256 registers | bit-identical; 8 rows: 27B IQ3_XXS gate 23.4 -> 42.9 GB/s, IQ3_S down 22.6 -> 38.9; an 8-row 27B pass 487 -> ~350 ms; 27B DFlash2 code 11.5 -> 16.2 tok/s, MTP code 8.9 -> 12.7; more drafts then paid off (default 27B 3 -> 7: 12.6 -> 15.2; 35B 3 -> 5: 34.2 -> 37.7) |
| **GEMV rows per sub-group chosen by K and token count**: 4 rows for codebook types and long rows at 2–5 tokens, one row for Q6_K, the old rule at 6–8 tokens; Gemma's 262K-row head 2 rows instead of 4 | 27B, 3 tokens: gate IQ3_XXS 57.8 -> 64.7 GB/s, down Q4_K 57.5 -> 70.6 (27B Indonesian with MTP 6.95 -> 7.2 tok/s); Gemma 12B head ~27 -> ~22 ms per token (noisy) |
| **DFlash2 block drafter** (8 rows per draft pass, 7 drafts verified at once; switches to the MTP block when its drafts are rejected) | first version: 27B Python code 4.85 -> 11.5 tok/s (89 % of drafts taken), 35B code 15.7 -> 44.4 (98 % taken); after the changes below 27B 18.1, 35B 49.8; Indonesian prose ~MTP (the drafters are weak outside English / code) |
| DFlash2 drafter weights in Q4_0 instead of Q8_0 | drafter layers ~45 % less time (27B: 539 -> 296 ms per answer), 0.84 GiB less memory, drafts accepted as often |
| **Host work out of the speculative loop**: the selector's top-16 in one pass instead of `nth_element` over the 248K vocabulary (~2 ms per draft position); draft confidence with a vectorized float exp (blocks summed in double) instead of one double exp per vocabulary entry (2–3 ms per draft) | the GPU waited for the CPU: 35B code 48.4 -> 52.7 and 49.4 -> 53.3 tok/s (+8–9 %), MTP 37.5 -> 39.1; 27B 16.9 -> 18.3 |
| DFlash2 + MTP hybrid backs off its DFlash2 probes on dense models (every 8 -> up to 32 steps) | DFlash2's 8-row pass costs more than an MTP pass: 27B Indonesian 6.2 -> 6.5 tok/s; MoE models keep probing every 8 steps |
| **One token's cached experts computed while its missing ones are read from the SSD** (the reads never evict experts of the running layer) | bit-identical; Flash Next (22 % cached, 84.5 % hits) 6.6 -> 7.3 tok/s |
| Flash Next: expert-cache pools grouped by record size, the dense weights' W8 copy freed for the cache (16 -> 22 %, hits 80 -> 86 %), MoE single-token kernels reworked, single-token mixers fused with their K split over 8 slices (hc_down1 255 -> 76 µs) | generation 4.28 -> 6.85 tok/s (25.1 -> 17.4 ms per token in the reworked parts), then 7.3 with the overlap above |

Per-token share of the bandwidth reached: Qwen3.5 4B 79 %, 9B 86 %, Gemma E4B 81 %, 12B 82 %, Qwen3.8 27B 72 %.
The 27B's remaining gap is decoding cost: its IQ3_S / IQ3_XXS / IQ2 weights read at 51–55 GB/s-equivalent while
IQ4_XS / Q4_K reach ±62 in the same profile; codebook types need more arithmetic per byte than this GPU hides. The
MoE expert kernels read small scattered blocks: the 35B's gate/up IQ3_XXS runs 51 GB/s alone, ±35 in the model.

## 4. Fitting big models and long contexts

| Change | Effect |
|---|---|
| **MoE expert streaming from the SSD** (Level Zero): only the experts in use stay in a GPU cache (LRU, the next layer's misses read on a thread while the GPU works) | Qwen3.6 35B-A3B at the full 262K context together with images and MTP (the all-resident engine fit ~40K; with images 4K), server with images ~17.5 GB instead of ~34 GB, Windows keeps 14 GB of RAM; MTP 22.2 tok/s at 262K (cache 79 %, hits 97.6 %) |
| Expert file written once from the GGUF: one GPU-ready, 4 KB-aligned record per expert, read with unbuffered I/O | writing 15 GB through the Windows file cache (with the GGUF pages still resident) left 3.9 GB of RAM free and stalled the laptop; unbuffered writes and releasing the GGUF pages per layer fixed it |
| Plain global LRU for the expert cache (from recorded routes and `zero/cache_sim.py`) | pinning popular experts was worse; hits 99.8 % at 88 % cached, 98 % at 70 %, 95 % at 50 %, 87 % at 30 % (35B) |
| Cache contents saved at exit (`.lru`) and loaded at start | 10.6 GB in ~6 s; first prompt 3.4 -> 0.6–1.1 s |
| Expert cache with one pool of the largest record size when a model mixes more than two record sizes (Qwen3.8 Flash Next: 512 experts, 5 type mixes) | separate pools each needed a whole layer and ran out of GPU memory; one pool runs it with a 7.9 GiB cache (12 % of a 56 GB expert file) |
| Prompt chunks of 4,096 with the experts of a chunk streamed through a separate buffer | a long prompt does not sweep the cache that generation uses |
| Out of GPU memory at launch: the expert cache gives back 1 GiB and retries | the GPU share is shared with every program (24 simulated failures, results identical) |
| Q8_0 kept only in its raw form for MoE / just-in-time models (decoded per chunk) | 35B UD-Q3_K_S GPU 16.47 -> 15.04 GiB, load 62 -> 28 s, context 61,440 -> 208,896 tokens; prompts ~4 % slower |
| MoE buffers aliased over buffers only used in the attention / DeltaNet half of a layer | 167 MB: 35B context 86K -> 102K |
| 1024-token instead of 4096-token prompt chunks (27B) | frees 1.1 GiB: context 90,112 -> 118,784, 11K prompts ~4 % slower |
| Buffers sized from what each model really uses (the Qwen KV scale buffer was sized per 64 dims, Qwen uses one per row) | 1,632 bytes per token less |
| **Automatic context** from a dry load that only adds up the GPU allocations (seconds, nothing uploaded), counting image / audio encoders | the budget changes with the driver setting and other apps; computing it per start avoids out-of-memory crashes and paging; Qwen3.5 9B with its vision encoder reserved 217,088 -> 143,360 tokens |
| Prefix cache (system prompt state on disk per project), session cache (whole conversations to disk), a snapshot at the start of the last assistant turn | a new message in a long Pi session: 5.2 s instead of 16.8 s (9B); a new session reuses the system prompt in ~5 s instead of ~40 s (27B) |

## 5. Tried and not kept

| Idea | Result |
|---|---|
| fp16 P·V accumulation in attention | no speed-up, worse error |
| 7-bit P (signed DP4a) instead of 8-bit P (unsigned x signed) | same speed, 1.8x the error |
| SIMD8 W8 GEMM (its DP4a microbenchmark peaks at 13.7 TOPS vs 13.2 for SIMD16) | 28 % slower: 27B ffn_gate 7.90 -> 5.69 TOPS, ffn_down 6.62 -> 4.74 (re-measured 16 Oct). The ALUs are 8 lanes wide, so SIMD16 loses almost nothing in pure arithmetic, while the GEMM is limited by the loads, lane broadcasts and scale math around the DP4a, which SIMD8 needs twice as many instructions for. A peak microbenchmark measures the part that is not the bottleneck |
| 32 tokens per sub-group in the GEMM, also with 256 registers (no spill), or two column groups per work-group | 2.6 TOPS at first; 2.4 vs 8.1 TOPS with 256 registers (27B ffn_gate, 16 Oct); CG=2 within 1 %: the slowdown was never the spill, larger tiles leave too few threads in flight |
| Software-pipelined GEMM loop; 24-bit multiplies in the scale step | 7.0 TOPS (it loses the unrolled superblock); no change |
| GEMM work-groups spanning several 16-column groups (activation sharing in L1); removing lane broadcasts or the scale epilogue | no gain once the row grouping was tuned; < 3 % |
| SLM staging of q/k for the DeltaNet recurrence | 4x slower (local memory bandwidth) |
| Several tokens per sub-group in the DeltaNet conv | 3x slower (too few sub-groups in flight) |
| A T = 1 DeltaNet recurrence with 4 lanes per state column and 128 registers | bit-identical but slower (35B 248–341 vs 226–254 ms per 96 tokens) |
| Chunked DeltaNet; f16 DeltaNet state | DeltaNet is only 4–9 % of a prompt; f16 saves ~1.5 % at the risk of growing error |
| 256-entry byte-pair table for IQ4_XS; register shuffles with a variable lane index | slower (local-memory bank conflicts, 66 -> 55 GB/s); shuffles halved the speed |
| Q4_K / Q5_K through the vector-load GEMV (`gemv_v`); Q8_0 through it | slower: every lane reloads the shared 16-byte header and their decoding is cheap (Qwen 4B gate 72.3 -> 59.8 GB/s); Q8_0 also slower than the local-memory window GEMV |
| Split-K GEMV for small matrices | 10–15 % faster on 512-row matrices, slower on large ones: no gain overall |
| Per-type GEMV rows / kernels tuned on single repeated matrices; GEMV row-grouping sweeps (1–8) for E4B's 256-row matrices | no gain in the model |
| MoE down projection + combine fused for single tokens | 212 vs 211 ms |
| gemv_glu (fused gate + up + activation) for the 27B | only 24 of 64 layers have matching types, ~0.3 % |
| 256 registers for kernels with small spills (DeltaNet conv 1 KB, MoE gate/up 2 KB) | spill gone, not faster: fewer threads in flight cost as much |
| 32-key tiles in prompt attention | the extra live state spills (2–3 KB): 14–20 % slower for Qwen |
| Splitting head dims for Qwen prompt attention (head 256, `attn_ps`) | bit-identical but slower: 1.0–1.6 vs 1.9–2.2 T MAC/s at first (splitting the dims doubles the per-row softmax and P packing work); once 256 registers were used, slower at every shape (926 vs 1,117 ms) |
| Double-buffered K/V tiles, 8 or 32 tokens per work-group in prompt attention | within 4 % (35B shape at 32K) |
| The head-dim-split decode attention (`attn_dsplit`) for Qwen verification rows (10 Oct) | up to 2x faster for 2-row passes, but rows no longer bit-identical to single-token decoding; later kept for the 35B only, with fixed key splits and a per-row P bound |
| LSC prefetch of the next K/V tiles in decode attention (`prefetch()`, 1–4 tiles ahead) | 37 % slower (27B, 32K: 1.12 -> 1.53 ms per layer): the extra messages compete with the real loads |
| Smaller or larger key splits for decode attention (64–512 keys per sub-group, to fill the last wave) | within 3 % at every shape (27B, 4B, 35B at 32K); the wave-count estimate from the assembly did not hold |
| IQ2/IQ3 GEMV: sign byte computed (`i | parity(i) << 7`) instead of read from a local-memory table, and the sign mask without the 32-bit multiply | gemv_v (the kernel the engine uses for these types) equal or slower: IQ2_XS 63 -> 60 GB/s; only the unused plain gemv got faster. Fewer instructions again did not mean faster |
| More than 3 MTP drafts per step, before the token-outer GEMVs | not faster (27B 3 / 5 / 7: 8.3 / 8.2 / 8.2 tok/s): each draft runs the MTP block with the full output head; a third Gemma draft per pass: +1.5 % |
| A draft head over a fixed vocabulary prefix (`ARC_DRAFT_VOCAB`) | 27B with 7 MTP drafts, first 131,072 ids: code 15.1 -> 16.3 tok/s but Indonesian 6.9 -> 5.8 (its words lie beyond id 131,072); 65,536 slower for both |
| Other MTP confidence cut-offs (`--mtp-min-p`) | 27B, 0.3 / 0.5 / 0.7 / 0.85: code 14.6 / 15.3 / 14.0 / 13.9, Indonesian 6.6 / 6.9 / 6.8 / 6.6 tok/s: 0.5 stays |
| Loading the DFlash2 drafter before the main weights | 27B: its lm_head back from 21 to 10 ms per pass, but other weights slowed and the wall-clock speed was no better (6.1–6.3 vs 6.6 tok/s) |
| Running the next expert group's decode to W8 beside the current group's MoE GEMM, on the second compute engine (Level Zero queue group 1) | a 1 GiB stream kernel and a DP4a kernel: 109 ms one after the other, 89 ms on two engines; the MoE prompt GEMM itself is ALU-bound, so at most ~7 % back for the cross-engine synchronization |
| Prefetching the next layer's experts from a routing guess (its router on this layer's FFN input) | Flash Next, 16 % cache: 37 % of the guesses used, 48 % of the misses caught, SSD reads +55 %, no faster (the SSD is the limit, and wrong guesses take its time) |
| A per-layer LRU for the expert cache (against the cyclic layer order of decoding) | a recorded Flash Next trace: 80 % hits vs 82 % for the global LRU (Belady's optimum: 90 %) |
| Reading a token's few missing expert records in 512 KB pieces, so that more reads are in flight | ssd_bench: one 2.2 MB record 0.88 ms alone, ~0.53 ms as four concurrent pieces; Flash Next generation equal (7.19 vs 7.19 tok/s), SSD time slightly up |
| Layer-major prompt segments of 32K instead of 16K for narrow models (experts decoded once per segment) | 35B, 40K prompt: 530 -> 521 tok/s (the larger residual buffer costs expert-cache room) |
| moe_rg outside its 320–1279-token range | 192 / 256 tokens 268 / 359 vs 324 / 369 tok/s with the per-tile kernels; 1536 / 2048 / 4096 tokens 565 / 571 / 583 vs 595 / 657 / 795 with decode + GEMM (rows decoded once per 16-token tile); 32-slot tiles spill and are slower still (2048: 523); consecutive tiles of one column group in consecutive work-groups: 512 tokens 2 % faster, 2048 tokens 25 % slower |
| moe_rg for Flash Next's gate / up as well (its down projection uses it) | per type at 2048-token chunks: IQ3_XXS 1240 -> 969 ms, IQ3_S 718 -> 570, but IQ2_S 1411 -> 2289; KL to the CPU reference 0.0191 -> 0.0231 (max 0.27 -> 0.49) on the 337-token text |
| Local-memory bank conflicts in the Flash Next mixers (activations read as x[16 l + j]) | transposed: hc_up1 / hc_down1 only 2–3 % faster (kept, bit-identical) |
| Dense weights carved from 512 MB device allocations (an arena), 2 MiB alignment, or trimming the working set at load, against the runs where every dense-weight kernel is ~2x slower | arena: 35B DFlash2 hybrid 22.7 -> 23.2 tok/s and steadier, but 27B prompts 2–5 % slower; alignment and trimming: no visible change |
| Q8_0 GEMV reading the GEMM's interleaved layout (one copy instead of decoding per chunk) | not worth it once Q8_0 decoding per 2048-token chunk fell to 72 ms (~2 % of a prompt) |
| Requantizing Q6_K's per-16 scales to per-32 for the GEMM | per-16 matrices cost only ~4 % more |
| f16 (half2) GEMM or larger key tiles in the vision encoder; int8 vision weights and activations | no faster, error 0.8–2.4 %; spills; 10–23 % off the f32 reference: activations must stay f32 |
| Opting out of Windows power throttling (EcoQoS) | no measurable effect |
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

**Level Zero:** the OpenCL option `-cl-intel-256-GRF-per-thread` does not survive into a Level Zero module made from
the native binary; the module has to ask for it itself (`-ze-opt-large-register-file` in `zeModuleCreate`). Without
that, decode attention at head size 256 spilled 4 KB per thread and took 3.6 instead of 1.9 ms per layer at 64K.

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
source order, the source order must already be the schedule you want. The same study gave the token-outer
verification GEMVs (rows' dot chains written interleaved, no per-token branches) and the DeltaNet state loads as
vload16 (fewer, larger messages in flight), both in section 3.

**The compiler's arithmetic is part of the result.** The kernels build with `-cl-mad-enable`, so the compiler may
contract a multiply and an add into one fused operation, and its choices depend on the surrounding code. In Flash
Next's hc_up1, making the lanes past the zero padding return 0 early was the fastest version (731 ms per 96 tokens)
but changed the compiled arithmetic and with it the results; keeping the same arithmetic on every lane and only
predicating the loads gave 905 -> 843 ms, bit-identical. Check bit identity after adding a branch, even one that
"cannot" change a value.

## 7. Mistakes worth avoiding

**Memory access**

- **Never let each lane load its own scattered scalars.** Flash Next's gated-mean mixer read each lane's 16 values
  one by one: every load instruction touched 16 cache lines. Vector loads made it 45 -> 3.7 ms per 2048-row chunk
  (12x). The same pattern made the W8 decoder that read global memory byte by byte 2.7x slower than the
  vector-loading one, and a vector block decoder slower than the superblock one for codebook types (each lane its own
  superblock: 16 cache lines per load instruction).
- **An identical test is not proof of a memory-ordering hazard's absence.** Reading back values a work-item just
  wrote passed every test but can return stale cache lines; keep such values in registers. Likewise a non-blocking
  upload of a prompt chunk could read the host buffer after the next chunk overwrote it (only with >= 3 chunks in one
  call, only in `run.exe`).
- **Level Zero kernels use raw pointers.** A read past an allocation that OpenCL clamped harmlessly is a GPU page
  fault ("device lost") under Level Zero: give buffers that block reads may overrun some slack (+64 KB per expert
  cache piece). Kernels that reach memory not passed as an argument (expert addresses from a table) need
  `zeKernelSetIndirectAccess`, or the driver may leave that memory paged out (only ~9.9 of 15.5 GiB were resident;
  random crashes in about half of the server test sequences). Set it only on those kernels: on every kernel it made
  the driver touch every allocation per launch, and the 27B got slower.
- **Release what you allocate.** The engine never freed its GPU buffers, so the dry load that sizes the context
  stayed resident and made the real load page.

**Measurement**

- **Check what the compiler actually received.** I once concluded that Level Zero ignores the 256-register flag;
  my own environment switch had come out empty, so the flag was never passed. Print each kernel's build options and
  spill size (`ARC_ZE_KPROPS=1`) before blaming a driver.
- **Run-to-run variance is 10–30 %, so A/B tests must alternate and repeat.** About one 35B run in five ran 14–16
  instead of 21–23 tok/s with identical work and 100 % expert hits: every dense-weight kernel (output head, qkv, z,
  ssm_out, q, o, drafter) was ~2x slower while the expert kernels (1 GB pool chunks) kept full speed, and those runs
  also loaded slowest. The suspect is where the weight buffers land in physical memory; nothing tried removed it. A
  server started while Windows was still releasing the previous one's memory ran at half speed for its whole life
  (35B + MTP 11.9 vs 26.5 tok/s, identical logs): wait a few seconds between runs. A 32K-context E4B run once read
  10.8 instead of 17.9 tok/s, seven repeats 18.5–19.0. Treat a single slow or fast run as noise until it repeats.
- **Do not measure next to another GPU process.** A CPU reference running alongside, a second server, or an idle
  loaded server (an E4B load took 13–43 s instead of 2 s) all distorted numbers. `run.exe` warns when it sees one.
- **The first run after a kernel source change includes JIT compiles.** A 27B generation right after a 32K prompt
  took 18.4 s for its first step (a dozen decode-attention variants compiled), then 238 ms per step; after an
  update, the 27B with DFlash2 needed 37 s for a 27-token prompt and then ran 1–3 tok/s while kernels compiled; a
  first image took ~60 s instead of ~3 s. Report the median step (runz prints it), warm up before timing, and in a
  server compile ahead: arcinfer now runs a short prompt, a token and a verification pass of every draft count once
  per model, options and kernel sources, with a marker in the kernel cache so later starts skip it (8 s once for the
  27B, 18 s for the 35B). A persistent kernel binary cache removes the stalls after the first use.
- **The first request after a load also pays first-touch costs** for the weights (27B: 27.5 s for a 15-token request
  that takes 2.8 s warm).
- **A profile changes what it measures and hides what is not a kernel.** With an event on every launch the 27B ran
  4.29 instead of 4.8 tok/s, and per-kernel GPU times show nothing of the host: the speculative loop's
  `nth_element` and scalar double exp cost 8–9 % while the GPU waited, and compiles or a driver pause look like a
  slow GPU. Compare the wall time per step with the summed GPU time (27B: 233 vs 223 ms, so ~4 % gaps) before
  optimizing kernels.
- **Benchmarks of one kernel can mislead.** Small-matrix benchmarks lie about GB/s (kernel launch time dominates;
  measure on matrices that stream from memory); a GEMV row choice that won in the bench was 2 % slower in the model;
  a peak DP4a microbenchmark favoured SIMD8, which was 28 % slower in the GEMM. Confirm every win end to end.
- **Estimates from one component can be wrong in either direction.** Level Zero was first dismissed because the gaps
  between kernels were only 1.6–2 %; run end to end it was clearly faster (35B time to first token 6.45 -> 4.07 s).
- **Logit differences at ~0.0003 KL are noise** from int8 rounding flips (±15–30 % run to run); judge accuracy over
  many positions and prompts. Changing only the 27B's score rounding moved its KL 0.000355 -> 0.000414.

**Process**

- **References have bugs too.** The CPU reference decoded only 512 of 640 values of a Q8_0 row and produced NaN in
  a silu built with `/fp:fast` (x / inf); per-block residual dumps from both sides found both.
- **Test scripts must stop their servers even when they fail.** A failed overnight script left a ~14 GB server
  running and the laptop hung by morning.

## 8. Correctness discipline that kept this sane

- An independent f32 CPU reference for every architecture (`tools/ref_forward`), compared by KL over all prompt
  positions (`run --kl-decode` for the generation path), and greedy output compared with llama.cpp; tokenizer and
  chat templates checked against llama-tokenize and the GGUF's Jinja templates.
- Every kernel change either bit-identical to the previous one (checked by dumping logits with `run --dump-logits`,
  and per matrix with `bench/gemv_bench GB_CMP`) or measured against the reference; Level Zero and OpenCL engines
  produce bit-identical logits on every model. When results may change, the bar was: mean KL <= baseline and
  < 1e-3, top-1 agreement >= baseline - 0.1 point, perplexity up < 0.1 %, greedy equal except at near-ties.
- Speculative decoding only accepted when the answer is identical to plain decoding, which requires verification
  rows bit-identical to single-token decoding: a 35B verification kernel that split the keys differently drifted
  after ~40 tokens of a 2K prompt, and only on long prompts.
- MoE results depend on how a prompt is chunked (per-tile kernels, decode + GEMM and single-token paths sum in
  different orders): compare engines with the same chunk sizes, and prefer kernels whose results do not depend on it
  (moe_rg equals decode + GEMM bit for bit).
- Locate an accuracy problem by simulating each rounding in the f32 reference: it pinned Gemma's drift on the
  per-row V scale (median KL 0.0071 -> 0.0017 in simulation with per-64 scales), showed that a vision encoder needs
  f32 activations, and that Gemma 4 is ~7x and Flash Next even more sensitive to int8 activations than Qwen3.5
  (Flash Next 337-token text: engine KL 0.15, reference with int8 activations 0.13).
- A single-position or 20-position KL is not a metric: one near-tied position swung a Gemma test between 0.009 and
  0.043 across builds whose intermediate values differed by at most 1e-6. Raw-text perplexity did not rank engines
  on Gemma either (130 / 120 / 113 / 111 across four engines).
- Per-kernel tests on real model tensors against the CPU dequantization (`tools/test_gemm`, `bench/gemv_bench`).

## 9. Serving an agent (Pi) on this laptop

- Put recurrent-state snapshots where the client's next request diverges: Pi drops an aborted reply, so the resume
  point moved to the start of the assistant header (next message 21.5 -> 5.0 s); Qwen's template drops earlier
  reasoning once the user writes again, so a third snapshot at the start of the last assistant turn (5.2 instead of
  16.8 s on the 9B).
- Cancel within one layer when the client hangs up (the socket polled after every layer and token): the GPU was busy
  73 s after Esc on a 7K-token prompt, now ~1 s, with the block in progress rolled back exactly.
- Constrain tool calls to the tool's schema while they are written: Gemma E4B under stress, 2/10 -> 9/10 well-formed
  calls; a call that is already well formed is sampled exactly as before (1–2 of ~1,100 tokens corrected).
- Stream tool-call arguments as they are generated (a 2,368-token `write` showed nothing for 6 minutes), and never
  split a UTF-8 character across stream pieces or JSON strings.
