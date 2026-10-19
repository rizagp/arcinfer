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

## Where it stands (2026-10-18)

| Model | Prompt | Generation | Accuracy |
|---|---|---|---|
| Qwen3.6 35B-A3B UD-IQ4_XS (MoE, experts on the SSD) | 604 tok/s (32K prompt), 324 tok/s (128K) | 20.6 tok/s; Python code 49.9 with DFlash2, Indonesian 21.9 with MTP; 12.6 at 128K depth | KL 0.004-0.008 against an f32 reference (generation path) |
| Qwen3.8 27B (IQ3_S GSQ, dense) | ~135 tok/s (4K prompt) | 4.9 tok/s; code 18.1 with DFlash2 + MTP | KL 0.0003-0.0004 |
| Qwen3.8 Flash Next (IQ3_S, 512 experts on the SSD) | 166 tok/s (4K), 163 (32K), 123 (125K) | 6-7 tok/s | KL 0.020 against the CPU reference with QSA (3.5K prompt) |
| Qwen3.5 4B | 911 tok/s (2K prompt) | 17.6 tok/s | KL 0.0021; perplexity 6.056 (llama.cpp 6.141) |
| Gemma 4 12B (Google QAT) | 240 tok/s (2K prompt) | 7.5 tok/s | KL 0.021; greedy text equal to llama.cpp |

Against llama.cpp on the same files and laptop (10 October, arcinfer's OpenCL engine; arcinferzero gives the same
logits as that engine and the newer numbers in the table above):

| Model | Prompt, 2K tokens: arcinfer / llama.cpp Vulkan / SYCL | 34K-token code prompt, time to first token | Generation after the 34K prompt |
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
arcinfer's int8 path scores as well as llama.cpp's.

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
