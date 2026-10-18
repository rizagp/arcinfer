# arcinferzero work log (14 October 17:00 – 15 October morning)

> Archived development log (14-17 October). The current state and every later change with its numbers:
> [before-after.md](../before-after.md); the specification of Flash Next: [flash-next-spec.md](../flash-next-spec.md).

Goal: a new engine `arcinferzero` (folder `zero/`, the old arcinfer is not changed) for MoE models. Experts are stored on
the SSD, and only frequently used experts are kept in GPU memory (LRU cache). Target: Qwen3.6 35B with a 262K context
and MTP at about 21 tok/s, results bit-identical to arcinfer, and Windows still gets at least 14 GB of RAM.

## Summary of results

| | arcinfer (old) | arcinferzero |
|---|---|---|
| Maximum 35B context (MTP) | 40,960 | 262,144 |
| 35B context with images + MTP | 4,096 | 262,144 |
| 35B generation with MTP | 22.3 tok/s (40K context) | 22.2 tok/s (262K context) |
| 35B generation without MTP | 16.9–18.3 tok/s | 19.3 tok/s |
| Time to first token (2,216-token prompt) | 4.1–6.5 s | 3.3–3.5 s |
| 35B server + images memory | ~34 GB (only ~1 GB of RAM left free, load 426 s) | ~17.5 GB (Windows gets 14 GiB, load ~10 s) |
| Results (logits) | reference | bit-identical |

## Order of work

### 14 October afternoon

1. **Testing Level Zero as a replacement for OpenCL** (`zero/clrt.h`). The arcinfer kernels run through Level Zero
   unchanged. Bit-identical results and faster: 35B without streaming TTFT 6.45 → 4.07 s, generation 16.9 → 18.3 tok/s.
   27B also identical, same prefill (~110 tok/s).
2. **Expert file on the SSD** (`zero/expert_store.h`): `%LOCALAPPDATA%\arcinfer\zero\<model>-<id>.experts`. One expert
   = one GPU-ready record, 4 KB aligned, read directly without the Windows cache. 35B: 15.1 GB for 41 layers (including
   MTP). Built once (144 s), then reused. Rebuilt only if the GGUF changes. Safe to delete.
3. **Expert cache on the GPU** (`zero/xcache.h`) and `-DXCACHE` versions of the expert kernels (each expert's address
   is taken from a table). First test: bit-identical to arcinfer, generation 19.8 tok/s.
4. **Error #1: RAM ran out while building the expert file** (only 3.9 GB of RAM left free, laptop slowed down). Cause:
   the 15 GB file went through the Windows cache and the GGUF pages stayed resident in RAM. Fix: write without the
   Windows cache, release the GGUF pages after each layer and after load.
5. **Error #2: cache installed at 100 %.** The minimum share per expert size group was too large. Fixed.
6. **262K context + 4096 chunk.** At first the chunk fell to 512 (TTFT 5.7 s). Fixed to 4096; long prompts are
   processed per layer (layer-major) so that experts are read once per layer, and the next layer's experts are read
   from the SSD in a separate thread while the GPU works (prefetch). TTFT dropped to ~3.4 s.
7. **New cache layout**: the 976 MiB temporary area was replaced by a 244 MiB staging area. Prompts use the back end of
   the LRU list, so they do not sweep the cache. The memory saved goes to the cache. Result: MTP 22.3 tok/s at 262K
   (target of 21 reached). Upper bound with a full cache: 23.9 tok/s.
8. **Error #3: GPU out of memory** (cache 84 %, 1.13 GiB of the GPU share left). Fixed by leaving 1.4 GiB for the
   driver. As a result the cache is ~79–82 % (without images) or ~70 % (with the image encoder reserve).
9. **Zero server** (`zero/serverz.cpp`): chat, tool calls, images + MTP working together. The cache contents are saved
   (`.lru`) and reloaded at start (10.6 GB in ~6 s), so the first prompt is fast (3.4 → 0.6–1.1 s).
   Zero's automatic context uses its own record (previously it inherited arcinfer's 4,096).

### 14 October night: hunting the "device lost" crash

The zero server sometimes crashed midway (in about half of the chat → tool call → several images test sequences).
Plain arcinfer with the same sequence did not crash. Traced with new debug tools: sync per kernel
(`ARC_ZE_SYNC_EACH=1`), expert address check (`ARC_XCHECK=1`), Windows memory budget, kernel scratch size.
Findings and fixes:

| Finding | Fix |
|---|---|
| A kernel reads slightly past the end of the last expert. 35B records are exact multiples of 4 KB, so there is no slack at the end of a 1 GiB block | Every cache block gets a 64 KB reserve |
| Address table entries not yet filled were 0 (expert ID out of range if the router gives NaN) | All entries always hold a valid address |
| The shim did not check buffer bounds like OpenCL. The zero server and arcinfer shared the system prompt cache folder | Bounds are checked (transfers rejected as in OpenCL), zero uses its own `zero\prefix` and `zero\sessions` folders |
| The GPU share is shared with other applications (Chrome, desktop) | If the GPU runs out of memory, the cache is shrunk by 1 GiB and retried. Tested with 24 simulated failures, results still identical |
| **Main cause:** the expert kernels access cache memory through addresses in a table, not kernel arguments. Level Zero requires the "indirect access" permission; without it that memory is not guaranteed to be ready (only ~9.9 of 15.5 GiB was resident) | `zeKernelSetIndirectAccess` set on all kernels. Speed unchanged (20.1 tok/s), still bit-identical |

After the last fix: **5 out of 5** server test sequences passed (previously about half crashed). The next full
sequence also passed: chat, tool call `get_weather({"city":"Jakarta"})`, three images (image answers exactly the same
as plain arcinfer).

### 15 October morning

- The laptop hung for a while. The cause: the overnight Pi test script failed midway, so the zero server (~14 GB) was
  not shut down and kept using RAM until morning. The laptop restarted at about 07:42. No test processes are running
  now. From now on, test scripts always shut down the server at the end, including when the test fails.
- (The mouse moving by itself did not come from this work: no screen, mouse or browser control tools were used. Pi
  tests always run with `--no-tools --no-mcp`.)
- Options changed as requested: **`--zero`** for arcinferzero (`pi-arc 35b --zero`, `arc-server start -zero`).
  Without `--zero` it is always plain arcinfer; no choice is remembered. The launcher has an "Engine" field.
- Help updated: `pi-arc --help`, `arc-server help`, `bin\serverz.exe --help` (options `--expert-cache-gb`,
  `--expert-cache-pct`, `--windows-reserve-gb`, `--no-stream`).
- Documents: CHANGELOG, QUEUE.md, USAGE.md (arcinferzero section).

## Not finished yet

- Automated Pi test with the zero server (the script failed overnight, not because of the engine). A direct try with
  `pi-arc 35b --zero` can replace it.
- 27B with zero: tried directly through Pi (15 October) and works well (automatic context 81,920 with MTP, same as arcinfer; the first request is slow because of the one-time kernel compilation). Still missing: a clean speed comparison. 27B is not MoE, so there is no streaming and
  its context is the same as arcinfer's. Only bit-identity and speed were tested.
- Optional: automatically delete old expert files when the GGUF is replaced by a new version.

## How to use

```
pi-arc 35b --zero          # arcinferzero
pi-arc 35b                 # plain arcinfer
arc-server stop            # stop the server when done
```

Files: `zero/` (engine, Level Zero shim, expert cache, kernels), `bin\runz.exe`, `bin\serverz.exe`.
Data: `%LOCALAPPDATA%\arcinfer\zero\` (15 GB expert file, `.lru` cache list, automatic context, prefix, sessions).

## Night of 15–16 October: tests and optimization in zero

User's server at the start: Qwen3.6 35B (OpenCL, 262K context), stopped at 17:00, restored in the morning.
All tests on `bin\serverz.exe` (port 8091/8090), `--no-prefix-cache`, test servers always stopped.

### A1. Turn-start snapshot (Qwen)

A Pi-like conversation: system prompt ±1,240 tokens, two user turns with tool calls (tool results ±1,650 tokens), then
a new message. The new message must continue from the start of the previous turn.

| Model | New message: cached | Prompt time | Cold server | Result vs cold server |
|---|---|---|---|---|
| 9B | 4,679 / 6,376 | 5.2 s | 16.8 s | identical (text + logprob) |
| 4B | 4,679 / 6,376 | 2.9 s | 9.7 s | identical |
| 35B | 4,679 / 6,376 | 3.4 s | 11.6 s | same text, different logprob |

Without this snapshot the new message starts from 1,242 (the end of the previous first turn is lost).
35B: the logprob difference does not come from the snapshot. MoE results do depend on how the prompt is chunked: chunks
≤ 8 tokens use decode attention + unfused gate/up, ≥ 1,152 tokens the expert GEMM, in between the per-tile kernel
(runz `--feed-step 7`: 0.1809 vs 0.1712 for the top token; cold server chunk 1024 vs 2048 also differs slightly).
Cold server twice: identical. Without streaming: the same difference. Which one is closer to f32 is measured in the baseline.

Bug found and fixed: `logprobs` sent tokens that are partial UTF-8 characters as raw bytes (broken JSON). Now the
text is replaced with U+FFFD, the original bytes stay in `bytes`.

### A2. `--preserve-thinking`

35B: new message cached 4,662 / 4,720, prompt 0.6 s (without the option: 1,242 cached, 5.0 s). 9B: warning "this
model's chat template has no such option", behavior unchanged.

### A3. Gemma tool call constraint (E4B, real Pi)

Task: read a script, run it, add a new section with the edit tool, run it again (test folder, read/bash/edit/write).

| Constraint | Succeeded | Edit calls | Failed | Time |
|---|---|---|---|---|
| off | 3/3 | 10 | 7 | 87–226 s |
| on | 3/3 | 5 | 0 | 88–112 s |

The failures with it off were all about content (`oldText` not unique / not matching), not format; the constraint did
not change a single token (0 corrections), so tool calls that are already correct are not disturbed. Format under pressure: see A3b.

### A3b. Tool call format under pressure (E4B, temperature 1.3, large edit forced via tool_choice)

| Constraint | Valid | Broken |
|---|---|---|
| off | 2/10 | 8 (wrong edits content 3, unknown keys 3, missing path 1, no call 1) |
| on | 9/10 | 1 (did not finish, 3,000-token limit) |

The constraint corrects 1–2 tokens out of ±1,100 per call (log: "tool call: 2 of 1237 tokens corrected").

### A4. Session cache (9B, Pi + web UI alternating)

Back to the Pi conversation after the web UI: loaded from disk in 0.17 s, 4,679 tokens reused, answer identical
to a cold server (15.9 s prompt). The short web UI conversation (597 tokens) is deliberately not saved (1,024-token
limit: reprocessing it is cheaper).

### A5. Web UI compacting (4B, 4,096 context)

Automatic at 72 %: 10 messages → summary, context drops to 27 %. The first 4B summary lost one code word
(DELTA); the summary instruction was tightened (list, every item copied exactly, temperature 0) → all code words kept,
also after a second compaction (8/8 correct). Stop while summarizing: conversation intact, the server cancels within 0.6 s.
Compact button: works. 4B summaries are sometimes long (±90 s) → instruction "one line per item".

### B. Zero baseline before kernel optimization (runz --bench-pp; server with 400-token answers)

Prefill / decode in tokens per second at short (2K), medium (16K) and long (131K) contexts:

| Model | Identical to OpenCL | Prefill 2K / 16K / 131K | Decode 2K / 16K / 131K | Server without / with MTP |
|---|---|---|---|---|
| Qwen3.5 4B | yes | 862 / 697 / 281 | 20.2 / 18.7 / 9.5 | 20.6 / 29.2 |
| Qwen3.5 9B | yes | 399 / 388 / 222 | 12.1 / 11.5 / 8.8 | 12.2 / 17.7 |
| Gemma 4 E4B | yes | 733 / 571 / 209 | 21.0 / 19.6 / 9.5 | 21.6 / 39.6 (drafter) |
| Gemma 4 12B | yes | 271 / 218 / 86 | 9.5 / 9.2 / 7.4 | 9.8 / 21.1 (drafter) |
| Qwen3.6 35B | no (see below) | 660 / 679 / 254 | 20.1 / 18.8 / 13.8 | (to follow) |

At long context decode drops to ±½ and prefill to ⅓–¼: attention. Estimate from the 4B: at 131K decode attention
uses ±30 GB/s of 77 GB/s, prefill attention ±28 % of the DP4a peak. GEMM is already at ±70 % of peak (review notes).
35B not identical run vs runz: suspected chunking (runz streaming 4096, run MoE 2048; MoE results depend on the
chunks, see A1), checked with `--chunk 2048`.

Baseline continued:

| Model | Identical to OpenCL | Prefill 2K / 16K / long | Decode 2K / 16K / long | Server without / with MTP |
|---|---|---|---|---|
| Qwen3.6 35B | yes, with the same chunk | 660 / 679 / 254 (131K), 146 (262K) | 20.1 / 18.8 / 13.8 (131K), 10.1 (262K) | 21.0 / 26.8 |
| Qwen3.8 27B | yes | 110 / 103 / 80 (65K) | 5.1 / 4.7 / 4.0 (65K) | 5.0 / 9.1 |

35B: run (MoE chunk 2048) vs runz (streaming: chunk 4096) differ; `--chunk 2048` on both: bit-identical.

### E. MTP up to 7 drafts (27B, server, temperature 0, two 400-token answers)

DeltaNet rollback replaced: the verification pass writes the state to `S_next` and keeps its inputs; when a draft is
rejected → the recurrence is replayed from `S` for the accepted tokens only (one state copy, not 3 or 7; no 150 MB
copy per step). All draft counts 1–7: answers identical to without MTP.

| Drafts | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| tok/s (code) | 7.45 | 8.31 | 9.36 | 3.06 | 3.83 | 4.17 | 3.89 |
| drafts accepted | 192/208 | 257/280 | 283/333 | 298/376 | 306/405 | 314/417 | 314/456 |

Without MTP 5.07. From 4 drafts (5 verification rows) the engine uses the multi-token GEMV TB=8: activations of 8 tokens
in registers (±128 GRF on this GPU) → spill. Fixed: see D1.

### D1. Register spill (ARC_ZE_KPROPS=1, 27B with 5 MTP drafts)

| Kernel | Spill per thread | Notes |
|---|---|---|
| `attn_split` (decode attention, verification rows) | 11,232 B | caused by my switch bug (see CORRECTION): 0 with flag 256 |
| `attn_decode` (decode attention, 1 token) | 7,456 B | same: 0 with flag 256 |
| `gemv_mt` TB=8 | up to 9,216 B → up to 2,912 B | fixed: activations loaded per token |
| `gemv_mt_v` TB=8 | up to 5,024 B → up to 1,248 B | same |
| `gdn_conv` | 1,024 B | not yet |
| `attn_prep` | 192 B | not yet |

TB=8 (4–7 drafts): 128 GRF is faster than 256 GRF (27B, 5 drafts: 9.2 vs 7.6 tok/s; before the fix 3.8).
Rows per sub-group 1/2/4: the same (9.2–9.45). With this fix 5–7 drafts = 3 drafts (±9.2 tok/s): verification
steps drop 117 → 94 → 86 per 400 tokens, but each draft adds one pass through the MTP block including the
248K-vocabulary output head (±1 GB read per draft on the 27B).

System problem found: a server started right after another server exits sometimes runs at half
speed for its whole lifetime (35B + MTP: 11.9 vs 26.5 tok/s, identical logs). A 5-second pause: normal. `arc-server stop`
now waits for the process to exit + 5 seconds; so does the test harness.

### CORRECTION (01:30): "Level Zero does not use 256-register mode" is WRONG, the cause was my bug

The test switch `ARC_ATTN_DEC_GRF` that I added that afternoon (a static `std::string` from a ternary operator) was empty
at runtime, so since the 20:24 build zero's decode attention kernels were built WITHOUT `-cl-intel-256-GRF-per-thread`.
Evidence: `ARC_ZE_KPROPS` now prints each program's build options; `attn_decode` was recorded without the flag. A new
probe (`zero/grf_probe.cpp`): a 192-GRF kernel with flag 256 does not spill and is equally fast in OpenCL, Level Zero
immediate and regular command lists (23 ms; without the flag 212–1,429 ms). Level Zero honors that mode. The switch was rewritten explicitly.
Affected: zero decode attention measurements after 20:24 (27B attn_decode/attn_split spill in table D1,
"zero 13.1 vs OpenCL 15.0 tok/s at 64K", the register-128 sweep, the dsplit comparison). The baseline (17:02 binary) is not.
KL results remain valid (spill does not change the arithmetic).

Re-measured with correct registers: 4B decode 32K old path 17.1–17.3 vs dsplit 17.1 (equal); 35B 32K 18.4 vs
18.5. Decode attention is indeed already close to the bandwidth limit. 256-register bench at 64K: dsplit wins 23 % only for
the 35B shape (8 heads per kv head: 1.16 → 0.94 ms per layer), 4B/9B +7 %, 27B +4 %. Decision: dsplit (4 heads per
sub-group, dimension split in 4, 1 row per launch, 256 chunks, P scale per row) only for 35B; the others
keep the old path (reads K/V for the 4 verification rows once). `ARC_QWEN_DSPLIT=0/1` disables / forces it.

Decode error (mean KL against CPU f32, runz --kl-decode; texts of ±250 tokens; top-1 = same top token):

| Model | Old path | New path (dsplit) |
|---|---|---|
| 4B (2 texts) | 0.00064 / 0.00067; top-1 245+249 | 0.00062 / 0.00064; top-1 245+249 |
| 9B | 0.00069; top-1 246 | 0.00060; top-1 247 |
| 35B (2 texts) | 0.00774 / 0.00444; top-1 245+240 | 0.00811 / 0.00436; top-1 242+243 |
| 27B (3 texts) | 0.00030 / 0.00030 / 0.00034; top-1 702/709 | 0.00040 / 0.00034 / 0.00032; top-1 700/709 |

Measurement noise at this level: changing only the 27B score rounding (exact int sum vs float per 64 dimensions)
shifts KL 0.000355 → 0.000414 and top-1 700 → 699. The old/new difference is within that noise (its direction also
alternates between texts and models), so the new path is used for all Qwen models; `ARC_QWEN_DSPLIT=0` restores the old one.

MTP with the new decode path (server, temperature 0, answers identical to without MTP in all cases):
4B 3 drafts 27.5 / 19.3 tok/s; 9B 3 drafts 18.5 / 15.1; 35B 1 draft 26.3 / 22.9, 3 drafts 30.7 / 23.4; 27B 3 drafts
7.9 / 6.3 and 5 drafts 9.4 / 7.0 (these 9B/27B numbers were disturbed by the f32 reference computed at the same time on the CPU).

### D5 Qwen prefill attention, default registers (attn_bench ATT_GRF128, last 4K chunk at 64K depth)

4B/9B: attn_q8 1,276 ms (spill 4,192 B) → attn_ps PNS 1, PHG 2, PTT 1, TPW 32: 1,166 ms (+9 %), bit-identical
(ATT_CMP). The PNS 2/4 variants are slower with default registers.

MTP re-measured with the correct build (server, temperature 0, average of code + Indonesian answers, all identical):
4B 2 / 3 / 4 drafts: 22.9 / 24.8 / 25.2 tok/s; 9B 2 / 3 / 4: 15.8 / 15.6 / 15.9; 35B 1 / 3: 23.9 / 26.1;
27B 3 / 5 / 7: 8.3 / 8.2 / 8.2. Zero defaults: 4B 2 → 3, 35B 1 → 3 (9B 2, 27B 3, Gemma unchanged).

### Zero becomes the default engine (16 Oct, ~02:30)

`arc-server`, `pi-arc` and the launcher use serverz by default; `-OpenCL` / `--opencl` for the old engine. Basis:
all of tonight's tests passed (identical to OpenCL, images/audio, MTP, Pi, tool calls, API format).

### D5 Qwen prefill attention with 256 GRF (correct build; last 4K chunk at 64K depth)

| Shape | attn_q8 (256 GRF, = engine) | best attn_ps | 
|---|---|---|
| 4B/9B (16/4) | 926 ms (spill 576 B) | 1,117 ms (PS 1,4,1 TPW 32) |
| 35B (16/2) | 923 ms (spill 576 B) | 1,138 ms (PS 1,4,1 TPW 16) |
| 27B (24/4) | 1,516 ms (spill 0) | 1,669 ms (PS 1,3,1 TPW 8) |

With 256 GRF attn_q8 wins in all shapes (attn_ps only wins against 128-GRF attn_q8, 1,276 ms). The engine
already builds attn_q8 with 256 GRF, so there is no change; `ARC_QWEN_PS` stays off by default.

### add_norm_quant spill (Gemma) 3,712 B → 0 (16 Oct, 03:00–04:50)

The second pass rereads the values that the work-item itself just wrote, instead of holding the whole row in registers.
E4B and 12B logits bit-identical to OpenCL (11K-token dump). runz --bench-pp (vs the 15 Oct afternoon baseline, which also
did not yet use the 256-GRF attn_prep_g): E4B 2K 733 → 814 tok/s, 16K 571 → 620, 131K 209 → 197 (a single run);
12B 2K 271 → 286, 16K 218 → 224, 131K 86.1 → 86.5; generate the same or slightly up.
Alternating old/new A/B, E4B: 2K identical (pp ~820, tg 21.4 for both); 64K pp the same (~330), 32K pp the same (~470).
Generate after a long prompt is very noisy: 32K old 10.8 then 17.9 tok/s, new 18.3 / 18.4; 64K old
10.1 / 9.9, new 8.9 / 8.9. This is the same "sometimes ½ speed" noise (this kernel does not depend on context length and
at 2K there is no difference), not a regression.

### gdn_conv (Qwen) 1 KB spill (16 Oct, 04:50–05:10)

256 GRF removes its spill, but 9B runz --bench-pp alternating 3 rounds: 2K 435/430/417 (without) vs 432/419/414
(256), 16K 390/366/365 vs 389/362/390: the same within noise. Reading the weights per channel (vload4) actually gives 1,792 B. Default
unchanged; `ARC_CONV_GRF=256` for experiments.

### add_norm_quant correction + attn_prep_g HD512 spill (16 Oct, 05:10–05:25)

The "reread from x" version (03:00) was reverted even though identical: a kernel comment notes that reading x just written
can return a stale value from L1 (which may show up only occasionally). Instead: values stay in registers, but `ANQ_MAX`
(superblocks per sub-group) is passed by the engine = ceil(n_embd / 256 / 16): E4B and 12B 1 → spill 3,712 → 64 B.
attn_prep_g: norm weights, RoPE pairs and the RoPE table are processed 8 elements at a time, and raw K (V for layers without a
V projection) is reread from qkv (input only, not written by this kernel): HD 512 spill 4,608 → 0 (256 GRF).
E4B / 12B logits bit-identical to OpenCL. runz --bench-pp: E4B 2K 827 tok/s (tg 21.5), 16K 625 (tg 20.1);
12B 2K 289 (9.55), 16K 228 (9.24).

### Generate sometimes at ½ speed: per-step recorder (16 Oct, 05:30–05:45)

`runz --bench-pp` now prints the time per generate step (min / median / max, first 8 steps) and accepts
`--prof-gen`. E4B 32K, 7 times (1 or 3 prompt runs, 128 tokens): 18.5–18.96 tok/s, median 52.4 ms, first step
91–231 ms; the slowdown did not appear. The earlier slow run (64 tokens, 10.8 vs 17.9 tok/s = ±2.4 s extra) fits better
with one large pause (e.g. WDDM moving memory) than with a GPU that is slow for the whole run; to be looked at again with
this recorder if it shows up.

### GPU memory page size (16 Oct, midday; `zero/page_probe.cpp`)

Prompted by the document "Intel GPU Level Zero Custom Kernel Blueprint" (4 KB pages, read-only virtual memory, paged KV
via zePhysicalMemCreate). Results on this driver:
- zeVirtualMemQueryPageSize: 64 KiB for small allocations, 2 MiB for allocations >= 2 MiB.
- zeMemAllocDevice: sequential reads 67–77 GB/s; random 256 B reads in an area <= 1 GiB ±78 GB/s, at 1.5–2 GiB 22–24 GB/s
  (±3.5x slower; guess: GPU TLB coverage of ±1 GiB with 2 MiB pages).
- zeVirtualMemReserve + zePhysicalMemCreate succeed, but zeVirtualMemMap is rejected (0x7800000a, UNSUPPORTED_SIZE)
  for chunks of 2, 64 and 256 MiB: Level Zero virtual memory cannot be used on this Windows. Paged KV (stage 1/2
  slots, tree drafts) has to go through a block table in the kernel, not physical page mapping.
- arcinfer reads weights sequentially and 35B experts in whole blocks (hundreds of KB), so the ±1 GiB limit for small random reads
  has almost no effect; no change.

Random read sweep (`page_probe 3072 sweep`; GB/s, a single zeMemAllocDevice allocation; a single 4 GiB allocation is rejected by the driver):

| Area | 256 B | 4 KiB | 64 KiB |
|---|---|---|---|
| 4 MiB | 902 | 934 | 951 |
| 8 MiB | 156 | 158 | 266 |
| 16 MiB | 104 | 104 | 228 |
| 32 MiB | 89 | 89 | 99 |
| 64 MiB – 1 GiB | 77–83 | 78–83 | 77–89 |
| 1.5 GiB | 23.7 | 75.7 | 76.0 |
| 2 GiB | 21.7 | 75.3 | 76.6 |
| 3 GiB | 20.7 | 74.9 | 76.2 |

GPU cache ±4 MiB (±900 GB/s); from ±32 MiB RAM speed (±78 GB/s); above ±1 GiB only random 256 B reads drop
(±21 GB/s), chunks >= 4 KiB stay at ±75. arcinfer's weights, experts and KV blocks (16 keys x 256 dim = 4 KiB per head)
are all >= 4 KiB contiguous: not affected. For paged KV later: pages >= 4 KiB.

### 27B: where the time goes (16 Oct, midday; runz --prof-gen, the per-kernel profile now uses Level Zero timestamps)

Previously `--prof-gen` in zero was always 0 ms (clrt.h had no timer); now events come from a kernel timestamp pool.
27B, generate (profiling on: 4.29 tok/s, without profiling 4.8): GPU time 223 ms / step 233 ms -> gaps between kernels ±4 %
(±940 kernels per token). Spill: attn_prep 192 B (0.5 % of the time), gdn_conv 1 KB (0.6 %): insignificant.
Effective GB/s per role (11.3 GB/token): attn_qkv 76, output 73.5, attn_gate 73; ffn_down 60, ffn_up 57, ffn_gate 56,
ssm_out 51. Per quantization type (per-layer profile, absolute numbers lower because of the profiling overhead): IQ4_XS / Q4_K
≈ 62, IQ3_S / IQ3_XXS ≈ 51–55, IQ2_* ≈ 47–52, IQ1_M ≈ 33: codebook decoding (IQ3/IQ2) is not yet hidden behind memory read time.
The 27B FFN is dominated by IQ3_S / IQ3_XXS (±5 GB). Estimate: IQ3 decoding on par with IQ4_XS +5–10 %, merging small kernels +2–4 %.

## Qwen3.8 Flash Next (qwen4exp), from 16 Oct midday

Specification: docs/flash-next-spec.md (still in docs/) (from llama.cpp src/models/qwen4exp.cpp). Done and tested (16 Oct 11:40–12:10):
- Loader: type Q2_0 (42) and CPU decoding of IQ4_NL / Q2_0; matches an independent Python decoder exactly (5 rows,
  including a patched n-gram table row). gguf_dump reads 1,224 tensors / 77.9 GiB (2 parts).
- CPU reference (tools/ref_forward, qwen4exp): hyper-connections, DeltaNet with a sigmoid gate, dense attention (same
  as QSA as long as all blocks are selected), MoE 512/10 + shared expert, n-gram PLE. "Apa ibu kota Indonesia? Jawab
  singkat." -> "Jakarta" (p 0.89), greedy "Jakarta<|im_end|>"; 20 tokens in 26 s on the CPU.
- K = 640 (`ffn_down`): IQ4_NL / Q2_0 rows padded with zero blocks to 768 (pad_small_block_rows, test PAD=1).
- GPU (zero/kernels/decode.cl): IQ4_NL (like Q4_0 via the kvalues table) and Q2_0 (72-byte superblock, 2-bit SWAR
  unpack) in decode32 / vblock; per-32 scales -> native Q8_0-shaped GEMM (engine + test_gemm). Flash Next weights:
  gemv, gemv_v, JIT-W8 + GEMM all OK (err 6e-8..1e-7 vs the CPU reference), no spill.
- Hyper-connection kernels (ops.cl: hc_norm_quant, hc_lo_quant (silu(lo/4), K 320 padded to 512),
  hc_gate_mean_quant, hc_combine): tests/test_hc.cpp vs CPU, err 2e-8..6e-8, int8 never off by > 1 step.
Next: the qwen4exp engine in zero/engine.h (BF16 hc_* / router weights as Q8_0 with padded K, layer flow,
512-expert MoE streaming directly from the GGUF + down row padding, PLE on the host), then QSA for long context.

### Flash Next runs on the GPU (16 Oct afternoon-night)

- qwen4exp engine in zero/engine.h: 4-stream residual (x_ [M][4x2560]), hc mixer before each block (per-stream norm,
  down + inject stacked into one 336-row Q8_0 matrix, silu(lo/4), up with K 320 padded to 512, gated mean),
  block output to hco_ then hc_combine; DeltaNet sigmoid gate (-DGATE_SIGMOID); MoE 512/10, F 640 stored per row as
  768 (HLD/FV in moe_gateup_q / moe_quant); shared expert padded to 768 rows; PLE on the host (n-gram hash, 16
  IQ4_NL rows from part 2) + ple_gate / ple_conv_add / ple_hist_update on the GPU; PLE conv history and the last 2 tokens
  stored in the layer 1 recurrent state (included in snapshots / sessions / cancel).
- QSA: indexer (k + q stacked), block keys pooled (mean of 4 keys, norm, RoPE at the block's start position) per
  attention layer, block tail in the state; scores + radix select of the top 512 blocks per row -> mask; attn_q8 / attn_decode
  with -DQSA (unselected tiles are skipped). Below ~2K tokens it stays dense (identical). Top-k 64 test: GPU vs
  CPU scores correlation 0.9998, layer 3 block selection exactly the same.
- Accuracy: short chat prompt KL 0.0014. 337-token text KL 0.15 = same as the CPU reference with int8-quantized
  activations (0.13); this model is sensitive to int8 in every matrix (hc 0.098, attn 0.087, ssm 0.052, router 0.034, shexp 0.017 when
  each is alone). CPU reference bug found via per-block dumps: Q8_0 dequant_row with K 640 produced only 512 values.
- Server: correct answers (Tokyo, an is_prime function). Generate ±4 tok/s, expert cache 7.9 GiB (12 %; one pool because of
  5 record sizes: separate pools each need 1 full layer -> GPU out of memory). Expert file 56 GB.
- 27B / 35B logits bit-identical after all the changes.

### DFlash2 for 27B (16 Oct night)

- Architecture from z-lab/dflash model.py (read through a summary, tensors from the safetensors header): context = 27B hidden
  states after layers 5/19/33/47/61, fc (25600 -> 5120) + hidden_norm, then K/V for each drafter layer (ring 4096,
  window 2048). Draft: block of 8 rows [token, mask x7], 5 pre-norm layers, 2-tap dynamic convolution grouped by 16
  around attention and MLP (prepare: kernel half 0 from the input; finish: half 1 on the output), non-causal
  attention to context + block, 27B head, top-16 greedy-path selector (logit + <pred[prev] * proj(h), succ[cand]>).
- fc is computed incrementally per target layer (5120x5120 GEMM slice, accumulated) so the features of the 5 layers are not stored.
- Results (greedy, identical to without the drafter): code 4.85 -> 11.5 tok/s (89 % of drafts accepted, 6.9 tokens/pass);
  Indonesian text with DFlash2 alone 2.9 tok/s (11 % accepted) -> combined mode with MTP: 5.6 (MTP alone 5.8-6.3).
- The drafter takes ±13 % of the time (dfl 7.9 %, head 4.7 %); the rest is the 27B 8-row verification (±480 ms per pass).

### Flash Next faster (16 Oct night)

- Generate 4.28 -> 6.85 tok/s: expert cache pools grouped by record size (12 -> 16 %), W8 copies of the dense weights
  released for the cache (16 -> 22 %, hit 80 -> 86 %), MoE TM 1 kernel reworked (agent, ISA study), single-token hc
  mixer fused (hc_down1 with K split into 8 pieces + hc_up1, f32): 25.1 -> 17.4 ms per token.
- This laptop's SSD: 2 MB random reads at most 3.6-3.8 GB/s at every QD (zero/ssd_bench.cpp); the engine is already at that limit.
- Guessing the next layer's routing for prefetch (ARC_XGUESS=1): 37 % of guesses used, 48 % of misses caught,
  SSD reads +55 %, not faster -> off. Simulation (zero/cache_sim.py): global LRU 82 % > per-layer LRU 80 %,
  Belady 90 % at a 16 % cache.
- QSA validated on 3,566 tokens: KL vs the CPU reference with QSA 0.020 (dense attention 0.043). The CPU reference briefly gave NaN:
  silu with /fp:fast (x / inf) in the mixer -> fixed in tools/ref_forward.
- 35B: moe_down kernel -20 % (bit-identical); bench/moe_bench: IQ3_XXS gate/up 51 GB/s alone, ±35 in the model.

### PP / TG / speculative decoding optimization (16-17 Oct night, F17)

- Token-outer multi-token GEMV (speculative verification): 8-row 27B FFN 23 -> 39-43 GB/s. New default MTP drafts:
  27B 7, 35B 5 (27B code 12.6 -> 15.2 tok/s, 35B 34.2 -> 37.7; Indonesian prose the same). Rows-per-sub-group rule
  for 2-5 tokens: 27B Indonesian prose + MTP 6.95 -> 7.2.
- DFlash2: Q4_0 drafter weights (half the bytes, 0.84 GiB saved on the 27B, same draft acceptance), single-pass top-16
  selector (instead of nth_element over 248K tokens, ±2 ms per position on the CPU), draft confidence softmax with vectorized
  float exp. Code: 27B 16.9 -> 18.3 tok/s, 35B 44 -> 53. Combined mode delays the DFlash2 probe on dense models
  (8 -> up to 32 steps): 27B Indonesian prose 6.2 -> 6.5.
- Flash Next: experts already in the cache are processed by the GPU while the missing ones are read from the SSD (6.6 -> 7.3 tok/s); prompt hc
  mixer (gated mean) with vector loads: 45 -> 3.7 ms per chunk; prompt down projection through moe_rg: 8K prompt
  98 -> 165 tok/s, KL to the CPU reference 0.0221 -> 0.0191.
- moe_rg (new): int8 MoE GEMM directly from raw expert rows (decoded in registers), result = decode + GEMM bit
  for bit; 35B prompts of 384-1152 tokens +3..14 % and logits no longer depend on the chunk size.
- serverz reports chunk 4096 for streaming MoE (35B 8K: 620 -> 750 tok/s compared to chunk 2048).
- Tried, not used (docs/lessons, section 5): overlapping two engines, a draft token list, an allocation arena, trimming the
  working set during load, split SSD reads, 32-slot moe_rg tiles, a swapped grid. Run-to-run variation of 10-30 % (sometimes all
  dense weights 2x slower) is recorded in QUEUE F16.
