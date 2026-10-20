# Changelog

## 2026-10-19: two conversations at once, Gemma 4 26B-A4B, Flash Next accuracy knob

- **Gemma 4 26B-A4B** (Google QAT Q4_0; arcinferzero, `pi-arc g26`): the dense FFN and 128 routed experts side by
  side, experts streamed from the SSD like the 35B's (gate / up from one tensor; GELU; the router in f32 on its own
  input norm; a scale per expert). Prompt 450 tok/s (4K), generation ~20 tok/s; KL 0.07 against the CPU reference
  (`tools/ref_forward` now runs its experts), the size of int8 activations' own effect. Images too: its vision
  encoder's 72-dim heads (`g4v_qkv_hd`) and std_bias / std_scale; the embeddings match `tools/gemma4v_ref.py` (cosine
  >= 0.999997; `bin\vision_testz.exe` writes them for arcinferzero). `--ctx auto` keeps no memory for an encoder it
  cannot run.
- The server prints why a model cannot load instead of stopping silently.

- **Conversation slots** (arcinferzero; `serverz --slots N`, `arc-server -Slots N`, `pi-arc --slots N`): the server
  holds N conversations, each with its own KV caches, DeltaNet / conv states, QSA keys, MTP / DFlash2 state and
  snapshots (`Engine::add_slots` / `bind`: the state buffers are re-pointed to the slot's memory, no copies).
  Requests in different slots take turns step by step by GPU time used, so two Pi sessions answer at the same time;
  a short question during another slot's 46K-token prompt: 49 -> 9.6 s. Logits bit-identical alone and interleaved
  (`runz --slots-check`: 27B, 35B with MTP and DFlash2, Flash Next with QSA); one slot unchanged.
- **Batched decoding for Gemma 4** too (26B-A4B: two answers at once 19.8 -> 26.2 tok/s); `--ctx auto` with slots on
  streamed MoE models aims at 128K in all and keeps at least 32K per slot (26B 2 x 32,768 instead of 2 x 8,192).
- **Batched decoding over slots** (Qwen dense and MoE): requests generating at the same time run their next tokens
  and drafts in one engine pass (`Engine::forward_batch` / `verify_batch`; every slot drafts in its own state, each
  settles its own drafts). Two answers at once: 27B 7.3 -> 10.5 tok/s together (prose), 11.6 -> 15.4 (code); 35B
  20.2 -> 24.7, 29.0 -> 30.7. Logits bit-identical to each slot alone; `runz --bench-batch N` times it (27B, no
  drafts: 2 / 3 / 4 slots 1.78 / 2.43 / 3.04x).

- `ARC_HC_F32` (arcinferzero, Flash Next prompts; `4` the default since 20 Oct, `0` the old single int8 pass): the hyper-connection mixers of prompt chunks with more
  precision. `4` takes the up projection's input as two int8 passes (values, then their rounding residual): KL
  against the f32 reference 0.0647 -> 0.0483, top token 324 -> 331 of 363, prompts 5-7 % slower. `1` f32 for both
  projections (KL 0.0369, about 22 % slower), `2` / `3` the down / up projection only. The default (int8) is unchanged.
- `tools/ref_forward` reads arcinferzero packages; `runz --kl-ref` reports top-token agreement.

## 2026-10-18: SSD expert prefetch on the copy engine, 35B chunks of 8192, drafter per model, long contexts

- **Deep expert prefetch** (arcinferzero, prompts of >= 2048 tokens): while the GPU runs a layer, a thread reads every
  missing expert of the next layer from the SSD and the GPU's copy engine copies them into their cache slots; the
  kernels wait on an event. Flash Next prompts 4096 tokens 107 -> 166 tok/s, 9000 116.9 -> 144.5, 32K 131.7 -> 162.6;
  35B 32K at ctx 131K 488.6 -> 557. Logits bit-identical; 56 regression runs (small caches, simulated GPU
  out-of-memory, 40 prompts in a row) identical. `ARC_XCOPY=0` turns it off.
- **Qwen3.6 35B prompts in 8192-token chunks** (streamed MoE): 32K prompt 533 -> 604 tok/s, 128K 306 -> 324,
  generation about the same (ctx 262K; expert cache 81 -> 77 %).
- **Drafter per model** in Pi's `/model` and `pi-arc`: `<id>-nodf` drafts with the MTP block only, `<id>-df` with
  DFlash2 only (the MTP block is not loaded: 27B context 69,632 -> 94,208). 35B: Python code fastest with `-df`
  (49.9 tok/s), Indonesian with `-nodf` (21.9). `--draft auto|mtp|dflash` for the server and `runz`.
- `--v4`: a 4-bit copy of V for generation, opt-in (35B at 250K depth 10.6 -> 11.5 tok/s, KV +25 %); the
  automatic context gives the 35B and Flash Next 131,072 tokens; Flash Next's QSA scores 8 rows per work-group
  (125K prompt 78.6 -> 123.4 tok/s).
- Fixes: a server switching models now keeps one Level Zero context (the new model used the old context's copy
  list, events and staging buffer); the expert prefetch thread's read errors no longer end the process; GPU
  out-of-memory during a prefetch is handled; numeric `ARC_*` knobs that divide are kept >= 1; `/fp:precise` for the
  accuracy tools.
- `scripts/hf-download.py` keeps trying until a download is complete (a router that drops out is waited out), with
  no speed limit by default and several files per command.
- Docs: [docs/before-after.md](docs/before-after.md) (every change with its numbers before and after, accuracy and
  consistency), a new README overview.

## 2026-10-16 (night): DFlash2 block drafter for Qwen3.8 27B

- `--dflash DIR` (runz, serverz) loads a z-lab DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2): 5 layers fed by the
  target's hidden states after layers 5 / 19 / 33 / 47 / 61 (fc + norm into a context K/V ring of 4096 positions,
  attention window 2048), a block of 8 rows [token, mask x 7] per draft with two-tap grouped dynamic convolutions
  around attention and MLP, the target's output head and the candidate selector (top 16, greedy bilinear path).
  Weights BF16 -> Q8_0 on the GPU (~1.9 GB), selector codebooks read from the mapped file.
- With the model's MTP block also loaded (the default with `--dflash`), each step picks the drafter: DFlash2 while
  its recent accepted drafts per pass stay >= 2 (`ARC_DFL_MIN`), else the MTP block (3 drafts), with a DFlash2 step
  every 8th to re-check. `ARC_DFL_P` (default 0.3) drops a DFlash2 path's tail below that probability.
- 27B, greedy, answers identical to plain decoding: Python code 4.85 -> 11.5 tok/s (MTP alone ~8.9; 89 % of drafts
  taken, 6.9 tokens per main-model pass); Indonesian prose 4.67 -> 5.6 (MTP alone 5.8-6.3; DFlash2 takes ~11 % of
  its drafts there: its training data is mostly English and code).

## 2026-10-16 (evening): Qwen3.8 Flash Next on arcinferzero

- New architecture `qwen4exp` (Qwen3.8 Flash Next, ISTA-DASLab GSQ-RCO IQ3_S, split GGUF of 2 parts) in arcinferzero:
  4 hyper-connection streams with their mixers before every block, Gated DeltaNet with a sigmoid output gate, gated
  attention, 512-expert MoE (10 active) streamed from the SSD with 640-wide experts (down rows padded to 768), the
  PLE n-gram block (rows read from the 26.8 GiB table in part 2 per token, its conv history kept with the recurrent
  state) and QSA sparse attention (indexer, pooled block keys, top 512 blocks per query; dense below ~2K tokens).
- Accuracy against the CPU reference (`tools/ref_forward`, which now covers QSA and dumps per-block residuals):
  short chat prompt KL 0.0014 ("Jakarta" 0.92 vs 0.93). On a 337-token raw text the KL is 0.15, the same as the
  reference with int8 activations (0.13): this model is more sensitive to int8 activation quantization than the
  others (every matrix group adds 0.02-0.10).
- Speed (first numbers, 32 GB laptop, expert cache 7.9 GiB = 12 %): generation ~4 tok/s, short prompts 3-6 tok/s
  while the cache is cold. The expert file (56 GB) is written once on the first start (~9.5 minutes).
- Expert cache: with more than two record sizes (Flash Next has 5 type mixes) one pool of the largest record serves
  every layer; separate pools each needed a whole layer and ran out of GPU memory.
- `src/quants.h`: Q8_0 rows whose length is not a multiple of 256 dequantize fully (the CPU reference read only the
  first 512 of 640 values of Flash Next's last shared expert).
- 27B and 35B logits unchanged (bit-identical).

## 2026-10-16 (afternoon): extra model folders, kernel timings on Level Zero

- Models can live on another drive: `%LOCALAPPDATA%\arcinfer\model-dirs.txt` (one folder per line) or
  `ARC_MODEL_DIRS` (`;`-separated) adds folders in the Hugging Face cache layout or with `.gguf` files directly in
  them. The Hugging Face cache is searched first, so a model kept in both places still loads from there.
- `runz --prof-gen` on arcinferzero now reports per-kernel times (Level Zero kernel timestamps; before: 0 ms).
- `zero/page_probe.cpp`: GPU page size and random-read speed by span (2 MiB pages, ~4 MiB GPU cache, random 256 B
  reads slow down above ~1 GiB; the Level Zero virtual-memory API is refused on this Windows driver).

## 2026-10-16: arcinferzero is the default engine; MTP up to 7 drafts

- `arc-server`, `pi-arc` and the launcher start arcinferzero (Level Zero) by default; `--opencl` / `-OpenCL` (launcher:
  Engine "arcinfer") runs the OpenCL server as before. `--zero` is still accepted. Tested the night before: identical
  logits to OpenCL for every model (35B with the same prompt chunking), images and audio identical (E4B, 12B, 9B),
  every API format, Pi, tool calls, MTP identical to plain decoding.
- MTP drafts up to 7 on arcinferzero (`--mtp-drafts 1..7`): a verification pass writes its DeltaNet states to a second
  buffer and keeps its inputs; rejected drafts are undone by replaying the recurrence over the accepted tokens (one
  state copy instead of one per draft, no 150 MB copies per step). Default drafts: 35B 1 -> 3 (23.9 -> 26.1 tok/s),
  4B 2 -> 3 (22.9 -> 24.8); 9B 2, 27B 3 (5 and 7 are not faster yet: each draft runs the MTP block with the full
  output head).
- Multi-token GEMVs (verification of 3..7 drafts) load each token's activations where they are used instead of
  holding all of them in registers: 4-7 drafts no longer spill (27B, 5 drafts: 3.8 -> 9.2 tok/s).
- Qwen3.6 35B decode attention: attn_dsplit (4 q heads per sub-group, dims split over 4), 1.16 -> 0.94 ms per layer at
  64K; verification rows stay bit-identical to single-token decoding (fixed key splits, per-row P bound).
- Gemma kernels without register spills: attention prep with the large register file and its norm / RoPE work in
  pieces of 8 (8.8 KB -> 0 per thread), the post-attention / post-FFN norm sized to the model (3.7 KB -> 64 B);
  identical logits. E4B prompt 2K 733 -> 827 tok/s, 16K 571 -> 625; 12B 271 -> 289, 218 -> 228.
- Fixes: constrained JSON / tool calls no longer allow a comma after the last allowed key (a `json_schema` answer
  dead-ended as `{"name": "x", "age": 34,`); `/api/tags` lists the loaded model with `--no-swap`.
- `arc-server stop` waits for the process to end plus 5 s: a server started while Windows was still releasing the
  previous one's memory sometimes ran at half speed for its whole life. The server and runz opt out of Windows power
  throttling (EcoQoS) as a precaution (no effect measured).
- Diagnostics: `ARC_ZE_KPROPS=1` prints each kernel's spill and its build options; `zero/grf_probe.cpp` (also
  `grf_probe FILE.cl KERNEL "OPTIONS"`: one kernel's spill); `runz --bench-pp` prints the time of each generation step
  (min / median / max) and takes `--prof-gen`; `ARC_CONV_GRF=256` (experiment: no spill, not faster).
- `ARC_DRAFT_VOCAB=N` (off): the drafts' output head over the first N token ids only; faster for English code, slower
  for Indonesian (its tokens lie beyond 65536).

## 2026-10-15 (night): Level Zero tests, Qwen tool-call constraint, saving the held conversation

- Constrained tool calls also for Qwen's XML format (`<tool_call><function=..><parameter=..>`): the function and
  parameter names are the tool's own, every required parameter is there before `</function>`, non-string values are
  JSON of their schema type, string values cannot contain `</parameter>` / `<parameter=` (the parser would cut them
  there). Qwen: only after the reasoning (it never calls tools inside it). Gemma stress test (E4B, temperature 1.3, a
  long edit forced through tool_choice): 2/10 well-formed calls without the constraint, 9/10 with it.
- `POST /sessions/save`: the held conversation goes to the session cache and the call returns once it is on disk;
  `arc-server stop` / `restart` call it first, and the server does it before a model switch, so continuing that
  conversation afterwards loads it instead of processing it again.
- `ARC_LOG_TOOLCALLS=file`: every answer that writes a tool call is appended as a JSON line (raw text, parsed calls,
  tokens the constraint corrected), to see what a model wrote when a call fails.
- Fix: `logprobs` sent tokens that are part of a UTF-8 character as raw bytes, making the JSON invalid; the text is
  now valid UTF-8 (U+FFFD for a partial character), `bytes` keeps the exact bytes.
- Web UI compaction: the summary request asks for a list that copies every specific item exactly (temperature 0);
  with the first wording Qwen 4B dropped one of seven code words, now none.

## 2026-10-15: web UI compaction

- `--preserve-thinking` (server; `pi-arc --preserve-thinking`, `arc-server -PreserveThinking`): Qwen3.6's template
  option, earlier turns keep their reasoning in the prompt (requests can set `chat_template_kwargs.preserve_thinking`).
  The prompt then only grows, so a new message reuses the whole cache; the context fills sooner. Only for models
  whose template has the option (Qwen3.6 35B); Qwen3.8 27B always keeps the reasoning.

- Qwen: a third recurrent-state snapshot at the start of the assistant turn that follows the user's last message.
  Qwen's chat template drops the reasoning of earlier turns once the user writes again, so every new message in a
  Pi session diverged right after the previous user message and was processed again from the end of the system
  prompt; now only the last turn is processed again. In `server-new.exe` until tested on the GPU.

- Constrained tool calls (Gemma 4; `tools/tool_constraint.h`): while the model writes a tool call, only tokens that
  keep the call well formed and fitting the tool's parameter schema are drawn: required arguments cannot be left
  out (Pi's edit without `path`), keys are the tool's own (no code as a key, no `edits` inside an edit), strings use
  `<|"|>`, enum values and integers are respected, and the call ends with `<tool_call|>`. A call that is already well
  formed is sampled exactly as before. The request log shows "tool call: N of M tokens corrected" when the
  constraint changed something; `ARC_TOOL_CONSTRAINT=0` turns it off. In `server-new.exe` until tested on the GPU.

- The disk session cache is on by default (16 GB; `pi-arc --session-cache 0` turns it off): a Pi conversation that
  another session or the web UI pushed out of the GPU is loaded from disk in seconds instead of processed again.

- The web UI shows the context the conversation uses ("context 12.3K / 40.9K") and compacts it: when the next
  request would nearly fill the context, or the server answers `exceed_context_size_error`, the earlier messages are
  summarized by the model (thinking off, the cached prefix reused) and the conversation goes on with that summary
  plus the latest exchange. A history too long to summarize at once is summarized in parts. The Compact button does
  it on demand; Stop cancels it and keeps the conversation. The old messages stay on screen, dimmed, with the
  summary under "Earlier messages compacted".

## 2026-10-15: the full API (OpenAI, Anthropic, Ollama formats), structured output, one server source

- One server source: `zero/serverz.cpp` is `tools/server.cpp` built with `ARC_ZERO` (Level Zero engine, its own
  data folder `%LOCALAPPDATA%\arcinfer\zero`); `setup.ps1` builds both, and `runz`.
- Sampling: `frequency_penalty`, `repeat_penalty` / `repeat_last_n`, `logit_bias`; `logprobs` / `top_logprobs`;
  `n` > 1 (without streaming).
- Structured output: `response_format` `json_object` / `json_schema` constrains sampling with a JSON-schema prefix
  checker (`tools/json_constraint.h`), so the answer always parses and fits; `tool_choice` `required` / a named
  function starts the answer with a tool call.
- Other formats on the same handler (`tools/api_compat.h`): OpenAI `/v1/completions` (raw prompt) and
  `/v1/responses`; Anthropic `/v1/messages` (+ `count_tokens`); Ollama `/api/chat`, `/api/generate`, `/api/tags`,
  `/api/show`, `/api/ps`, `/api/version`; `/v1/audio/transcriptions` and `/translations` (Gemma 4 audio).
- `tests/test_api_compat.cpp`: 44 CPU-only checks of the JSON checker and the request translation (all pass).
- Launcher: the API address for each format with a Copy button. `pi-arc --zero` / `--no-stream`, `arc-server -Zero`
  / `-NoStream`; startup messages name the engine.
- Not yet: `/v1/embeddings` (needs an embedding model architecture the engine does not have).

## 2026-10-14: arcinferzero: Qwen3.6 35B with its experts streamed from the SSD (Level Zero)

- New engine in `zero/` (arcinfer itself is unchanged): `zero/clrt.h` runs arcinfer's OpenCL kernels under Level
  Zero (native binaries from the shared kernel cache, USM buffers). Same results bit for bit, faster: 35B TTFT 6.45 ->
  4.07 s, generation 16.9 -> 18.3 tok/s; load 4.3 -> 2.3 s (4B).
- Routed experts live in `%LOCALAPPDATA%\arcinfer\zero\<model>-<id>.experts` (GPU block form, one 4 KB-aligned record
  per expert; made once from the GGUF, 15.1 GB / 144 s for 35B; remade when the GGUF changes; may be deleted) and are
  read with unbuffered I/O (no Windows file cache) into an LRU cache on the GPU. The expert kernels take each
  expert's address from a table (`-DXCACHE`); prompts use the back of the LRU list as their transient area and read
  the next layer's missing experts on a thread while the GPU works; the cache's contents are saved (`.lru`) and loaded
  again at start (10.6 GB in ~6 s).
- Cache size: automatic so the process stays within RAM - 14 GiB (Windows) and the GPU share - 1.4 GiB;
  `--expert-cache-gb`, `--expert-cache-pct`, `--windows-reserve-gb`, `--no-stream`. When the GPU runs out of memory
  anyway (the share is shared with every program), the cache gives back 1 GiB and the launch is retried.
- Qwen3.6 35B at 262K context, 4096-token chunks: MTP 22.2 tok/s (cache 79 %, hit 97.6 %), without MTP 19.3 tok/s,
  time to first token 3.3-3.5 s for 2,216 tokens (arcinfer: 4.1-6.5 s at 40K context); output identical to arcinfer.
  With images and MTP together: 262K context, cache ~70 %.
- `bin\runz.exe` / `bin\serverz.exe` (build `zero\runz.cpp` / `zero\serverz.cpp`); `arc-server -Zero`,
  `pi-arc --zero` and the launcher's Engine field choose it; without them the usual arcinfer runs.

## 2026-10-14: Qwen3.6 35B images; projectors that leave too little context are opt-in

- Qwen3.6 35B-A3B reads images with Unsloth's `mmproj-BF16.gguf` (Qwen3-VL, 27 layers, 72-dim heads): text in a
  1920x1080 screenshot quoted exactly; 1,008 image tokens encoded in ~3 s (~60 s the first time: kernel compile).
- `--ctx auto` now counts the image encoder of MoE models too (it reserved nothing). For 35B the encoder (~1.1-1.3 GB)
  leaves only 4,096 tokens at 18.3 GiB, so a projector that cuts a model's auto context below 16K (to less than
  half of it) is no longer part of the default variant: `35b` stays MTP without images (40,960 tokens), images go
  through `35b-vision`. Other models keep their variants.

## 2026-10-14: launcher window, GPU memory by part, monitoring

- `arc-launcher.cmd` / `scripts/arc-launcher.ps1`: a window with every model and variant (MTP, images, automatic
  context, GPU memory at that context, compatibility), how the automatic context was counted for the selected one,
  a field for every server option, and the running server's state, GPU memory by part, requests and speeds. Start /
  Stop go through `arc-server`; the server keeps running when the window closes. Every field starts at the selected
  model's best value (`--list-models` "defaults": chunk, MTP drafts, image tokens, ...); changed fields are marked
  and are the only ones passed on; "Set as default" saves them and the default model, "Reset options" clears them.
- `arc-server.ps1`: every server option (`-MtpDrafts`, `-ThinkBudget`, `-ImageMinTokens` / `-ImageMaxTokens`,
  `-BudgetGb`, `-AttnF32`, `-NoPrefixCache`, `-PrefixCacheGb`, `-NoSwap`, `-ListenHost`, `-SessionDir`); a running
  server is restarted when one given differs (a plain `start`, as pi-arc does, keeps it).
- The engine counts its GPU memory by part (weights, MTP block / drafter, working buffers, context); `server.exe
  --list-models` gives each variant's counts as `--ctx auto` used them ("memory": device, kept free, snapshots,
  encoders, weights, mtp, buffers, context, context per 1K tokens), `/props` the loaded model's ("memory") and
  monitoring ("stats": requests, busy, filled context, the last request's timings). `/props` reports audio when the
  model has an audio encoder (it said false). The ctx-auto cache key carries layout version m9.

## 2026-10-13: Qwen3.6 35B: prompt experts through the int8 GEMM, MTP, more context; Qwen prompt attention

- MoE prompt chunks of >= 1152 tokens run the experts through the int8 GEMM (kernels/gemm.cl `MOE_TILES`): per group
  of 64 experts the raw rows are decoded to W8, then one GEMM over that group's slot tiles. Qwen3.6 35B UD-Q3_K_M,
  2048-token prompt: 476 -> 642 tok/s; per-layer outputs equal to the per-tile kernels (differences ~1e-7).
- `decode_w8_sl`: whole rows (up to 16 superblocks) are copied to local memory with coalesced loads, then decoded
  from there (types without mins). IQ3_XXS 1038 -> 518 ms, IQ4_XS 433 -> 303 ms per 2048-token chunk; Q8_0 / Q4_0
  use their vector block decoder (`decode_w8_vb`): Q8_0 211 -> 72 ms.
- MoE buffers share memory with buffers only used in the attention / DeltaNet half of a layer (expert activations
  over z_, expert outputs over conv_out_ / o_ / qrope_): 167 MB. Automatic context: `35b` 86K -> 102K (94K before
  the GEMM staging), `35b-s` 217K.
- Single tokens: DeltaNet qkv + z (+ alpha / beta when Q8_0) stacked into one GEMV (views, nothing stored twice).
  35B 19.2 -> 19.8 tok/s, 27B 4.75 -> 4.8-5.0, output identical; 9B unchanged (qkv / z types differ in most layers).
- MTP for Qwen3.6 35B (Unsloth's MTP repository, `35b-mtp`): MoE in the MTP block, Q3_K on the GPU (decode32), expert
  types with mins (Q4_K / Q2_K / Q5_K) in the expert kernels (activation sums computed in the kernel). Output
  bit-identical to plain decoding; 19.7 -> 22.3 tok/s (83 % of drafts accepted), prompts -3.4 %, automatic context
  40,960 (`35b-mtp-std`: 98,304). MoE models default to 1 draft per step (1 and 2 equally fast; one DeltaNet state
  copy less). Verification rows of models whose single token takes `attn_decode` (35B: 2 kv heads) now take it as
  well, row by row: `attn_split` split the keys differently and MTP output drifted after ~40 tokens of a 2K prompt.
- Qwen prompt attention (`attn_q8`): 4 q heads per sub-group where the GQA ratio allows. 21,590-token prompt:
  35B 9.4 -> 8.4 s, 9B 8.2 -> 7.0 s of attention, same output; 27B (6 q heads per kv head) unchanged.
- Server: Qwen3-VL projectors also load for MoE models (a projector whose output width does not match the model is
  refused); no 35B mmproj was available to test with.
- tools/bitcheck: residual stream per layer and logits of single-token vs 2-token passes and of the MTP flow
  (draft, verify, reject / accept); tools/ref_forward reads standard GGUF experts.
- Tried and left off: Q8_0 GEMV through `gemv_v` (slower), down + combine fused for single tokens (212 vs 211 ms),
  `attn_ps` for Qwen shapes (slower), Q4_0 requantization of the MTP block's experts (replaced by native Q3_K / Q4_K).
- The ctx-auto cache key carries layout version m8.

## 2026-10-13: Qwen3.6 35B: both Unsloth files selectable, router + shared expert in one GEMV

- `35b` = UD-Q3_K_M (IQ4_XS down experts; automatic context 94,208), `35b-s` = UD-Q3_K_S (204,800); both in Pi's
  /model (several files of one repository are named with their file), and the server switches between them.
- MoE layers keep the router rows (padded to 384) and the shared expert's gate and up rows in one Q8_0 buffer:
  a single token runs one GEMV over the 1,408 rows instead of three small ones (each paid a launch's fixed cost);
  the three matrices are sub-buffers of it for prompts, so nothing is stored twice. Qwen3.6 35B UD-Q3_K_M
  generation 17.4 / 17.8 / 18.5 -> 18.2 / 18.8 / 18.8 tok/s (alternating runs), output identical.
- Tried and left off: split-K GEMV for matrices with few rows (`ARC_SPLITK`): 512-row matrices 10-15 % faster,
  larger ones slower, no gain overall; per-type GEMV rows / kernels tuned on single repeated matrices: no gain in
  the model (Qwen3.8 27B). Qwen3.8 27B with its MTP block measured 4.5 -> 7.9 tok/s on code (2 drafts, 91 %
  accepted).
- The ctx-auto cache key carries layout version m4.

## 2026-10-13: memory: Q8_0 stored once, encoders reserved by --ctx auto; `35b` alias for Unsloth's GGUF

- Q8_0 matrices keep their repacked GEMM copy only with resident weights of a dense model; MoE models and
  just-in-time models decode the raw rows into the same form per prompt chunk (decode_w8, PER32). Qwen3.6-35B-A3B
  UD-Q3_K_S: GPU 16.47 -> 15.04 GiB, load 62 -> 28 s, automatic context 61,440 -> 208,896 tokens (a 24K-token prompt
  at 356 tok/s); prompts ~4 % slower (decoding per chunk), results identical.
- `--ctx auto` keeps the GPU encoders' memory free (weights + buffers for the largest image / a 30-second clip; they
  upload on first use and were not counted): Qwen3.5 9B 217,088 -> 143,360 tokens (~1.4 GB for the vision encoder at
  2048 image tokens), Gemma 4 E4B unchanged at 131,072 with 1.6 GB kept. The ctx-auto cache key carries an engine
  memory-layout version ("m2"), so stale contexts are computed again.
- `qwen3.6-35b-a3b` / `35b` (`pi-arc 35b`) also name Unsloth's `Qwen3.6-35B-A3B-UD-Q3_K_S.gguf`.
- Qwen3.5 9B vision attention (72-dim heads): q read again per key tile instead of held in registers (spills
  295 -> 164); speed unchanged at ~1.9 TFLOPS (64-dim heads: 2.8). The compiler keeps 128 registers per thread for
  this kernel despite -cl-intel-256-GRF-per-thread; going further needs another kernel layout.

## 2026-10-12: GEMV bandwidth, Qwen3.5 9B images, Qwen3.6-35B-A3B from standard GGUFs

- GPU raw forms regrouped for aligned vector loads: Q8_0 (the 8 scales first; GEMV 37 -> 77 GB/s), IQ2_XXS /
  IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S (the f16 scale moved behind the quants; IQ2_XXS 27 -> 60, IQ3_XXS 41 -> 62,
  IQ3_S 61 -> 66 GB/s); `gemv_v` vector decoders for the codebook types where faster. Token generation, share of
  the measured ~75 GB/s: Qwen3.5 4B 77 -> 79 %, 9B 83 -> 86 %, Gemma 4 E4B 76 -> 81 %, 12B 80 -> 82 %, Qwen3.8 27B
  65 -> 72 % (4.3 -> 4.7 tok/s). All regression tests pass.
- Qwen3.5 9B images (head size 72, width 1152): `vit_attn_rows` / `vit_qkv_rope` take the head size as a build
  option (72-dim heads read K / V with interleaved sub-group block reads: 6.0 -> 1.6 s of attention for a 1222-token
  image), a LayerNorm for any width; 0.38 % from the f32 reference (tools/vision_test.cpp, generalized).
- Qwen3.6-35B-A3B from standard GGUFs (Unsloth UD-Q3_K_S: IQ3_XXS / IQ3_S / IQ4_XS experts): `moe_gateup_q` /
  `moe_down_q` (kernels/decode.cl) on the experts' raw rows, each decoded weight serving a tile of 16 slots;
  generation 17-18 tok/s, 2048-token prompts 473 tok/s (150 with 8-slot tiles re-decoding per tile), automatic
  context 61K at 17 GiB. Prefill and token-by-token agree (top-1 0.9718 / 0.9712).

## 2026-10-12: Google's Gemma 4 QAT models (Q4_0), E4B images and audio on the GPU

- **Q4_0** weights (Google's QAT GGUFs): the GPU raw form regroups each 8-block superblock (8 scales, then the
  16-byte nibble groups, same 144 bytes) so every block is one aligned vector load. Token generation reads them in
  `gemv_v` / `gemv_mt_v` / the fused `gemv_glu` at 66-73 GB/s (bit-identical to `gemv`, ~1e-7 from the CPU
  reference); prompts decode them losslessly into the GEMM's native Q8_0 form (per-32 f32 scales; there is no such
  W8 variant: code 104 was read as per-16 + mins). Google Gemma 4 12B: 9.1 tok/s (85 % of the measured bandwidth),
  17-20 with the drafter; KL 0.0206 vs its f32 reference, greedy = llama.cpp's. E4B: 20.9 tok/s, 26.8 with the
  drafter, prompts 788 tok/s.
- Google's GGUFs omit `attn_k_norm` on shared-KV layers (they compute no K): optional there.
- Google's chat template (2026-07-09) concatenates adjacent string literals (`"a " "b"`), which minja cannot parse:
  the server crashed at start. Adjacent literals inside `{{ }}` / `{% %}` are now joined first; renders identical
  to jinja2 (tests/test_chat_jinja.py, new tests/data/gemma4_google_template.jinja).
- **E2B / E4B images** (mmproj gemma4v, src/vision_gemma4v.h): a 16-layer ViT on the GPU (f16 weights, f32
  activations, clamped linears, RMS sandwich norms, NEOX 2-D rotary, unscaled attention, 3x3 pooling), 0.11-0.17 %
  from an f32 numpy reference (tools/gemma4v_ref.py); a 1120-token image in 3.2 s. The FFN uses gelu_tanh as
  Google's config says (llama.cpp falls back to gelu_quick for these files). In the language model, image rows take
  the per-layer embedding of token 0 and stay causal (llama.cpp: E2B / E4B are always causal).
- **E2B / E4B audio** (mmproj gemma4a, src/audio_gemma4a.h): log-mel and the subsampling convolutions on the CPU, the
  12-layer conformer on the GPU; 7.2 s of speech encoded in 226 ms and transcribed word for word.
- `--image-min-tokens` / `--image-max-tokens` for Gemma 4 12B (default 1120..2048: small screenshots were read
  wrongly at ~140 tokens); Gemma models with an image projector keep one image per prompt step (`min_chunk`).
- Model list: projectors named `*-mmproj.gguf` (Google) are no longer listed as models; Qwen3.6-35B-A3B GGUFs
  without GSQ experts are listed as unsupported (they failed to load). setup.ps1 / docs: g12 and e4b now download
  Google's QAT files plus Unsloth's drafters (copied next to the model).
- `run.exe` prints the weight bytes the GEMVs read per token and the resulting GB/s.

## 2026-10-12: Gemma 4 12B reads small text in images (1120..2048 tokens per image)

- Gemma 4 12B images now become 1120 tokens by default (`--image-min-tokens` default 1120, was 70 as in
  llama.cpp): its encoder turns each 48x48-pixel patch into one token with a single linear layer, so a ~550-pixel
  screenshot became ~140 tokens with ~10-pixel text per patch, and the model misread words ("BORENK", "AEROWR") and
  invented numbers in a spec table. Scaled up to 1089 tokens it read both test screenshots exactly (five 5x3
  tables, part numbers, prices). Large images may now use up to 2048 tokens (`--image-max-tokens` default 2048,
  was 1120, the most Gemma 4 was trained with); at 2025 tokens it still read the logo and the 4x-upscaled table
  screenshot exactly. Cost: 1120 tokens ~0.7 s encoder (CPU) + ~7 s prefill; 2025 tokens ~1.5 s + ~11 s.
- `--image-min-tokens` / `--image-max-tokens` apply to Gemma 4 12B too (before: Qwen only); scaling up no longer
  rounds past the maximum.
- Image placeholder ids in the prefix cache include the token count, so a saved conversation whose image was
  encoded with other limits is not matched against the new image tokens.

## 2026-10-12: Qwen3.5 9B keeps its decoded weights (short prompts ~4x sooner)

- `--ctx auto` now also chooses whether the decoded int8 (W8) weights stay on the GPU: they do when that still
  leaves at least min(128K, trained) tokens of context; otherwise they are decoded just in time, with the largest
  context, as before. Before, a fixed 9 GB budget decided, so the 9B (raw + W8 12.4 GB) decoded every matrix again
  on every request. Qwen3.5 9B profile (11,240-token prompt, 27.4 s): GEMMs 76 % at ~7.7 TOPS (the kernel's
  practical ceiling; its Q6_K per-16 matrices cost only ~4 % more than the per-32 ones, not worth requantizing),
  decoding 1.4 s. Resident: 83-token prompt 1.05 -> 0.27 s to the first token, 11K prompt 27.4 -> 26.6 s; context
  262,144 -> 217,088 at 18.3 GiB (17.0 GiB used). The other models keep their mode and context (4B, E4B resident;
  12B, 27B just in time: resident would not fit). An explicit `--budget` keeps the old rule.
- The `ctx-auto.txt` cache key now includes this choice (before, a `--budget` run reused a context computed for
  just-in-time weights: the 9B with resident weights then took 17.8 GiB).

## 2026-10-12: image input for Qwen3.5 4B (in Pi, the web UI and the API)

- Qwen3.5 4B reads images: its `mmproj-F16.gguf` (Qwen3-VL projector) is found next to the model and runs on the GPU
  (`src/vision_qwen3vl.h`). `--list-models` and Pi's model list mark it `text, image`; Pi's `read` tool, the web
  UI's Media button and OpenAI `image_url` parts work. Checked in real Pi: it read the text of a test image exactly
  ("HELLO 42") and described a photo correctly (ship, terminal and crane names), 2,320-token prompt with two images
  in 2.2 s.
- Encoder precision: with the language model's int8 GEMMs the encoder drifted 10-23 % from an f32 CPU reference on
  photos. A CPU emulation of each rounding (`vision_test`, `VT_EMU`) showed every layer's int8 activations
  contributing, and int8 attention a further 7-9 %. The encoder now runs f16 weights with f32 activations
  (`vit_gemm_f16`, 2.4 TFLOPS) and f32 attention: 0.2-0.5 % error, worst token cosine 0.999. A typical image
  (~250 tokens) encodes in ~0.4 s, a full-size 1080p screenshot (1008 tokens) in ~2.3 s; the first image after a
  start also uploads the encoder (~3 s).
- Language model: M-RoPE (Qwen3-VL interleaved sections 11/11/10): image tokens get (t, h, w) = start + (0, row,
  column) of the merged-token grid, and the text after an image runs max(grid w, h) positions on instead of the
  token count (`Engine::rope_shift`, per-row positions in `attn_prep`). Text-only output is bitwise unchanged
  (regression tests: same KL to 5 decimals).
- `/v1/models` also carries llama-server's own list: a `models` array whose `capabilities` hold `multimodal` for
  the models with an image encoder (next to OpenAI's `data` with `capabilities`, `input_modalities`, `type: vlm`),
  for clients that treat arcinfer as a llama.cpp server.
- Encoder attention: one query row per lane (SIMD8) with q and the accumulators in registers, K / V rows broadcast,
  4-key tiles, native_exp (`vit_attn_rows`): 1008-token image 1118 -> 588 ms (2.7 TFLOPS), 4096 tokens 36.1 ->
  15.3 s, 2048 ~11 -> 5.0 s; error against the f32 reference unchanged (0.2-0.6 %).
- Images are scaled to 256-2048 tokens (32x32 pixels each; `--image-min-tokens`, `--image-max-tokens`): at 1024 a
  6750-pixel logo was misread ("Logan" for "Logam", llama.cpp at 1024 too), at 2048 read right (~8 s to encode). Not yet:
  Qwen3.5 9B (its encoder is 1152 wide with 72-dim heads), Gemma 4 E4B, the 12B's encoder on the GPU.

## 2026-10-12: just-in-time weight decoding 2.7x faster (27B prompts); GEMM measured against its ceilings

- `decode_w8_sb` (the W8 conversion of models whose weights do not stay decoded, e.g. Qwen3.8 27B) loads each
  superblock into registers with vector loads and decodes it from there, and is now used for every type. The
  per-32-block kernel used for IQ3_S / IQ3_XXS / IQ4_XS read global memory byte by byte. 27B, 11,240-token prompt:
  those three types 5.8 -> 2.2 s (all decoding ~6.3 -> ~2.6 s, ~3.5 % of the time to first token). The prompt
  logits are bitwise identical (6,615-token prompt; `ARC_W8_SB=0` selects the old kernel). It also shortens model
  loads that decode at load time (E4B).
- `run --dump-logits` writes the prompt's logits too; `ARC_PROF_W8TYPE=1` profiles decoding per weight type.
- GEMM, measured in isolation (new `bench/gemm_w8_bench`, 27B shapes, bitwise comparison with the original):
  W8 with per-32 scales 8.3-8.6 TOPS, per-16 7.1; no-load ablation 9.7, no loads and no scale epilogue 11.0, against
  13.2 TOPS for signed DP4a with a broadcast operand (`peak_ss_bc_a`; the 7.1 TOPS `peak_dp4a_bcast` is the slower
  signed x unsigned form). Tried and not kept: a software-pipelined chunk loop (identical results, 7.0 TOPS: it
  loses the unrolled superblock), 32 tokens per sub-group (2.6 TOPS), 24-bit multiplies in the scale step (no
  change). In the model the 27B's GEMMs run at ~6.8 TOPS over a 100 s prompt: sustained heat, the per-16-scale
  types (about a fifth of its weights) and the kernels in between, not the kernel's shapes (all 8.3-8.4 alone).
  Experiment switches: `ARC_GEMM_DEFS`, `ARC_GEMM_TM`, `ARC_GEMM_WGY`, `ARC_GEMM_RG`.

## 2026-10-12: web UI follows the reply; Copy / Download on code blocks

- Every code block in a reply has **Copy** and **Download** buttons, above and below it (no scrolling to either end
  of a long block), and shows its language. Download names the file
  from the block's language (```` ```html ```` -> `code.html`, `py`, `js`, `ps1`, ...; `code.txt` when none). The
  copied / saved text is the code exactly as written (HTML special characters included).

- The chat view stays at the newest output whenever the thread grows (streamed text, the thinking box appearing or
  opened, images loading, tool calls, the stats line). Before, it only scrolled when a new piece of text arrived,
  and it pulled the view back down on every piece even when you had scrolled up to read. Now scrolling up (wheel,
  touch, scrollbar, PageUp / Home / arrow keys outside text boxes) pauses following at once and shows a "↓ Latest"
  button; scrolling back to the bottom, the button, or sending a message resumes it. Checked in the browser against
  the server: 0 px behind while following (also with the thinking box opened mid-stream), the view stays put while
  paused.

## 2026-10-12: Gemma 4 E4B generation 17 -> 21 tok/s (31 with the drafter), faster long prompts

Profile of one generated token (E4B IQ4_XS, short context): 56.5 ms of GPU time; the matrices (2.74 GB read per
token) would take ~37 ms at the 75 GB/s the iGPU reads. The IQ4_XS GEMVs ran at 44-65 GB/s, and ablations showed
why: without its memory loads the kernel still took 75-90 % of its time, i.e. decoding the 4-bit codebook indices
(byte-wise reads from local memory, a table lookup and shift / or packing per weight) cost as much as reading them.

- **`gemv_v` / `gemv_mt_v`** (`kernels/decode.cl`): each lane vector-loads its block's header and indices straight
  from global memory (no local-memory window, no barriers), and IQ4_XS's codebook comes from four local tables
  pre-shifted into byte 0..3, so a packed int8x4 word is four lookups ORed together (~8 -> ~4 instructions per
  weight). Used for IQ4_XS and for Q6_K output heads; Q4_K / Q5_K stay on `gemv` (their decoding is cheap, and there
  `gemv` was as fast or faster). gemv_bench, weights streaming from memory: E4B FFN gate 65 -> 70 GB/s, down 57 ->
  69, attention q 44 -> 67, output 48 -> 69; Q6_K output heads (E4B, Qwen3.5 4B) 66-67 -> 72 GB/s. Results are
  bit-identical to `gemv` / `gemv_mt` (new `GB_CMP` mode of `bench/gemv_bench`: every matrix compared bitwise).
- **`gemv_glu`**: Gemma's FFN for 1-8 tokens in one launch: gate and up rows of each 32-value block in the same
  work-group, then `gelu_tanh(gate) * up` quantized per 32 values in the epilogue (the same formulas as
  `swiglu_quant`), into a second activation buffer that the down projection reads. Replaces gemm_gate + gemm_up +
  swiglu_quant (two launches per layer fewer).
- `attn_prep_g` / `add_norm_quant`: vector loads for each lane's run of values (same arithmetic; ~5 % faster).
- Result, E4B on `bench/prompts/gemma-code160` (128 tokens, same session): 17.2 -> 21.2 tok/s without drafting,
  ~23 -> 30.7 tok/s with the assistant drafter (`run --mtp`; the drafter itself is unchanged); through the server
  with sampling (temperature 0.7) 30.9 tok/s. GPU time per token 56.5 -> 46.6 ms. The logits of 64 generation steps
  are bitwise identical to before (new `run --dump-logits F`; `ARC_NO_GEMV_V=1` / `ARC_NO_GLU=1` select the old
  path for such comparisons).
- **Long prompts**: Gemma 4 E4B's global layers (head size 512, 4 q heads per kv head) now use `attn_ps` (head dims
  split over sub-groups) like the 12B's, with 8 tokens per work-group: attn_q8 spilled 2 KB of registers there.
  `bench/attn_bench` at 13,921 tokens: 870 -> 619 ms per layer (same error against the CPU reference). E4B, 13,921-
  token prompt: time to first token 23.5 -> 21.8 s (591 -> 638 t/s). Not bit-identical (partial scores summed in
  another order): perplexity over 6 chunks of 4,096 tokens 88.44 -> 88.34.
- Prompt speed in context: E4B does ~27 % more arithmetic per token than Qwen3.5 4B (wider FFN, head-512 global
  attention); pp2048 789 t/s (Qwen 4B: 880).
- `tests/run_tests.ps1` (Qwen3.5 4B, Gemma 4 12B): all checks pass (Gemma KL 0.0188, greedy = llama.cpp, drafter
  and MTP output identical).

## 2026-10-12: pi-arc opens the model it started

- Fixed: pi-arc started Swift but Pi opened Gemma 4 E4B, shown with Swift's context (86,016). `last-model.txt`
  holds the path as `server.exe --list-models` wrote it, with mixed separators (`C:\Users\me/.cache/huggingface/
  hub\models--...`); arc-server compared it to the resolved path as text, found no model and started the file under
  a name made from the file name (`swift-...-iq3_s-mtp`). That name was not in `/v1/models`, whose first entry was
  then `gemma4-e4b`; pi-arc and arc-server took the first entry as the running model, so Pi was given
  `gemma4-e4b`, its list got the running context, and Pi's first request made the server switch to the E4B.
  - server: the scanned paths have one kind of separator; model files are compared as files
    (`std::filesystem::equivalent`), so the loaded model is listed first and marked `loaded` however its path was
    spelled, and a request for its catalog id does not load it again. A model file outside the scanned folders is
    listed first under the alias it was started with.
  - arc-server / pi-arc: model paths compared as full paths; the running model is the entry marked `loaded`.
- web_search on networks where DuckDuckGo is blocked (Indonesia): pi-unsloth-webtools searches DuckDuckGo and
  Startpage by default, and from here DuckDuckGo times out while Startpage returns nothing, so every search ended
  in "No results found" (or was cancelled while waiting). Yandex answers (5 results in ~1 s), so Pi's
  `settings.json` now has `"webSearch": {"engines": ["yandex", "startpage", "duckduckgo"]}` (Pi's `/search-config`
  changes it). Not an arcinfer change: the model's `web_search` calls were well formed.

## 2026-10-11 (late night): image input published per model

- `/v1/models` says per model whether it takes images, in the fields clients read: `capabilities` (llama-server /
  Ollama style: `completion`, `tools`, `vision`, `multimodal`), `architecture.input_modalities` (OpenRouter style),
  `type` (`vlm` / `llm`, LM Studio style), `modalities`; `/props` adds llama-server's `modalities` block. Before,
  only `/props` had a `vision` flag, for the loaded model only, so a client such as Unsloth Studio could not see
  that `gemma4-12b` reads images. Only models whose image encoder arcinfer runs are marked: Gemma 4 12B
  (`gemma4uv`). Qwen3.5 4B / 9B (`qwen3vl_merger`) and Gemma 4 E4B (`gemma4v`) ship an mmproj whose encoder is
  not implemented yet, so they stay text-only.

## 2026-10-11 (late night): Pi's edit streams; Pi's model list follows the disk

- Tool calls: array and object arguments now stream while they are generated, as the model's JSON text kept as
  written (`chat::json_prefix` cuts it where strict JSON can continue: never inside an escape, a number, a literal
  or a UTF-8 character). Before, only string arguments streamed, and Pi's `edit` (its whole change is the `edits`
  array) arrived in one piece at the end. Measured on Swift 1.5 Qwen3.8 27B: a 282-character edit arrived in 95
  pieces over 14 s instead of at once. If the model's JSON is not strict (e.g. a raw line break inside a string),
  the part already sent is closed and the parameter is repeated with the whole-response value; JSON readers take the
  last one, so Pi sees exactly what it saw before. `tools/tool_stream_test`: new cases (streams before the array is
  closed, non-strict JSON, numbers and literals), 20,000 random responses; regression tests pass.
- All of Pi's tools checked (pi-coding-agent 1.1 with pi-unsloth-webtools, pi-browser-control and MCP `codemode`:
  17 tools, their parameter types captured from Pi's own request): `tools/tool_stream_test` now calls each of them
  as Qwen writes it and checks the final types against the schema and that every long string / array value
  streams before its parameter closes (bash command, edit edits, write content, browser_type text, codemode
  code, ...); numbers and booleans arrive when their parameter closes. Boolean parameters also accept Python-style
  `True` / `False` (before: passed on as the string "True", which Pi's validation rejects).
- Live, Swift 1.5 Qwen3.8 27B with Pi's real tool list (one streamed request per tool, nothing executed): all 17
  tools give arguments that match their schema; long values stream (write: 282 pieces over 54 s, edit: 99 over
  19 s, codemode: 34, bash: 22). Before the boolean fix 5 of them failed: this model writes `True` for
  browser_navigate, browser_screenshot, browser_click, browser_type and browser_console_logs.
- `arc-server start` / `restart` bring Pi's model list up to date (they rescan the Hugging Face cache), also when
  the server is already running, so a deleted model leaves Pi's `/model` without running `pi-arc` (which no longer
  writes the list itself). Checked with a model deleted from a temporary cache.

## 2026-10-11 (late night): help for every option, README brought up to date

- Every command documents all of its options: `pi-arc --help` (aliases, `-c` / `-r`, setup, what happens to the
  caches when you continue a session, all server options and files), `arc-server help`, `bin\server.exe --help`,
  `bin\run.exe --help` (all 24 options; before: a one-line usage) and `setup.cmd --help` (new: steps, models,
  what it does not install). `--help` exits with 0 everywhere.
- `pi-arc --model <text>` (or `--model arcinfer/<id>`, as in Pi) chooses the model like `-Model`; before, it went
  to Pi next to the `--model` pi-arc sets.
- README: "Using Pi" rewritten (model choice and `/model`, contexts per GPU memory setting, `--ctx auto`, a table of
  what the server processes for each way of starting or continuing a session, Esc), the Pi commands, the server's
  model switching and `--list-models`, the prompt cache under "How it works", troubleshooting (compiler search,
  Windows slow after raising the GPU memory, the first session in a project), layout.

## 2026-10-11 (late night): system prompts kept per project folder

- Pi's saved system prompts are grouped by project folder, read from the `<cwd>` section of Pi's system prompt
  (`<folder>-<folder hash>-<prompt hash>-<tokens>.arcp`; the folder's spelling normalized, so `C:/X` and `c:\x` are
  one folder). Each folder keeps its 2 latest versions (an `AGENTS.md` edit or a Pi update replaces the oldest;
  two variants used in turn both stay), and other folders do not replace them; only the size limit below removes a
  folder's files, least recently used first. Before: the 8 used last per model,
  across all folders. Prompts without a folder (other clients): the 8 used last, as before. All models' files
  together stay under `--prefix-cache-gb` (default 8; least recently used first). Files saved before this are
  renamed into their folder's group the first time they are used (no reprocessing).
- Checked on an isolated 4B server: per-folder names, 2 versions per folder, other folders untouched, load after a
  restart (0.07 s), a pre-change file adopted, the size cap; and in real Pi (file named after its folder).
- Not changed: the session cache (whole conversations, `--session-cache`) stays separate and off by default. Only a
  request that starts with exactly the saved tokens can reuse a saved state, so a folder's cache can cover the
  system prompt its sessions share but not one session's conversation for another.

## 2026-10-11 (late night): one setup command

- `setup.cmd` (`setup.ps1`): checks the requirements (Windows, Intel Arc GPU, Visual Studio C++ compiler, Python,
  Pi; says where to get what is missing), builds `server.exe` / `run.exe` / `clinfo.exe` when missing or older than
  the sources (compiler output in `%LOCALAPPDATA%\arcinfer\build.log`, shown on failure), checks the GPU with
  `clinfo` (OpenCL 3.0, DP4a, sub-groups, GPU memory), downloads the chosen models (`-Models 27b,4b`, `all`, `none`,
  or asked; Gemma with drafter and projector; the 35B downloaded and converted, with the 4B for its tokenizer),
  installs `huggingface_hub` / `numpy` with pip after asking, connects Pi, and runs the regression tests with
  `-Tests`. Safe to run again: done steps are skipped.
- `build.ps1` finds the compiler with `vswhere` (any Visual Studio / Build Tools with `vcvars64.bat`;
  `$env:ARC_VCVARS` overrides) instead of a fixed path, and builds with `_CRT_SECURE_NO_WARNINGS` (no
  getenv / fopen deprecation notes).
- `install-pi.ps1` writes Pi's model list from `server.exe --list-models` (every model found, each with its
  context) instead of a fixed list with fixed contexts; `scripts\pi-models.ps1` holds that code for it and `pi-arc`.

## 2026-10-11 (night): Esc in Pi stops the server right away

- A client that closes its connection (Pi's Esc) cancels the request: the server polls the socket (no write needed)
  after every layer of prompt processing and before every generated token. Before, a prompt kept being processed to
  its end (a 7K-token prompt on the 27B kept the GPU ~73 s after Esc; 50K tokens: ~10 minutes) and the next request
  waited for it.
- Prompt processing stops within one layer (`Engine::cancel`): the block it was in (layer-major segment or chunk) is
  undone (DeltaNet states from a host copy taken at the block start, 27B: 157 MB, ~20 ms per block; Gemma moves back
  in its sliding-window rings, else resets), complete blocks stay cached.
- Pi drops an aborted (or failed) reply and sends the conversation without it, so the next request diverges inside
  the `<|im_start|>assistant` header. The resume snapshot was right after that header: the whole conversation after
  the system prompt was processed again (2.5K-token conversation: 21.5 s). It is now taken at the header's start:
  5.0 s.
- Checked in real Pi on the 27B (a scratch project with Pi's normal tools and extensions plus a local MCP server):
  parallel tool calls, edits, files with backslashes, quotes, braces and Unicode, bash with long output and failing
  commands, MCP through codemode, git, follow-ups: 27 replies, 21 tool calls, all arguments valid, no errors from the
  server or Pi; files on disk exactly as asked. The regression tests' server runs with `--no-prefix-cache`.
- `pi-arc --ctx auto` / `arc-server -Ctx auto` work (pi-arc failed on a non-number): auto, and a server running a
  fixed context is restarted with auto; without the option a running server keeps its context. Bad values give a
  message instead of a PowerShell error.
- Help: `server.exe --help` (exit code 0) lists every option with its default; `pi-arc --help` has a WORKING IN PI
  section (Esc, new sessions, project folders); `arc-server help` says where saved system prompts are.
- `docs/review.html` and `docs/review_notes.md` (now docs/archive/) updated: stages 12–14, contexts at 17.4 GiB, Pi on the 27B with
  today's measurements and the use-case run, two new limits.

## 2026-10-11 (evening): system prompt kept on disk

- The state after a client's system prompt is saved to `%LOCALAPPDATA%\arcinfer\prefix\<model>` the first time it is
  processed and loaded when a request starts with it while the server holds something else (server start, model
  switch, a conversation with another system prompt). Pi's system prompt contains the project folder and its
  `AGENTS.md`, so each project has its own file; the 8 used last per model are kept. `--no-prefix-cache` turns it
  off. Measured in Pi, Qwen3.8 27B, 4146-token system prompt: first new session after a server restart 40.2 s ->
  5.3 s to the first token (file loaded in 0.53 s); the file (290 MB) is written in the background.
- The state is saved without the duplicate host snapshot at the same position (157 MB less per file; the snapshot
  is taken again after loading). Works for Gemma 4 too (its system turn is recognised the same way).
- `--ctx auto` measures with a dry load that only adds up the GPU allocations (seconds, nothing uploaded), so
  `--list-models` reports every model's context and Pi's `/model` list gets each model's own context.

## 2026-10-11 (later): one command for Pi, models switched from inside Pi

- `pi-arc` starts Pi right away on the model the server runs, else the one used last (`--pick`: choose from the
  list of every supported GGUF model in the Hugging Face cache, by full name). It registers all those models in
  Pi's `models.json`, so Pi's `/model` lists them.
- The server switches models itself: a request whose `model` is another model of `server.exe --list-models` (its
  id, `org/repo` name or file) unloads the current one and loads that one (`--no-swap` turns this off).
  `/v1/models` lists them all, the loaded one first. E4B <-> 4B: 4-9 s; GPU memory is the same after any number of
  switches (the engine now releases its buffers, kernels and context: the `--ctx auto` measuring load used to stay
  in memory).
- `--ctx auto` for every model by default (the context follows the GPU memory setting); `--list-models` reports each
  model's id, shortcut, image support (a `gemma4uv` projector: Gemma 4 12B, not the E4B's) and computed context.
- Disk session cache off by default (`pi-arc --session-cache 16`); `pi-arc --help` documents every option.

## 2026-10-11: session cache (switching between conversations), 27B back to 4096-token chunks

### Saved conversations

The server held one conversation: a new Pi session, or going back to an older one, processed the whole prompt again
(Qwen also its ~4.2K-token system prompt: the DeltaNet state can only be resumed where a snapshot was taken). Pi on
the 27B, two sessions taking turns: every first request had 0 tokens cached, 36 s for a new session, 55–64 s to go
back to a 6.6–7.5K-token one.

- `Engine::save_state` / `load_state`: the KV cache (int8 blocks and scales of every layer with its own cache; for
  sliding-window layers only the window before the resume point), the MTP block's cache and hidden state, the DeltaNet
  states and snapshots; copied in batches of up to 256 MB.
- Server: when a request does not continue the held conversation's last prompt, that conversation is saved to
  `%LOCALAPPDATA%\arcinfer\sessions\<model>-<size>\` first; a saved conversation that shares at least 512 more
  reusable tokens with the request is loaded. `--session-cache-gb` (default 0 = off; least recently used first),
  `--session-dir`; `GET /sessions` (size), `DELETE /sessions` (clear); `arc-server status` / `arc-server clear-cache`.
- Qwen: a recurrent-state snapshot at the end of the system prompt, kept in host memory (no GPU memory), so a new
  conversation with the same system prompt processes only its own messages.
- Answers are identical with and without other conversations in between (greedy; Qwen3.5 4B, Gemma 4 E4B, Gemma 4
  12B). Saved size for ~4.5K tokens: 4B 225 MB, E4B 49 MB, 12B 215 MB; saving 0.1–0.9 s, loading 0.05–0.3 s.

### Qwen3.8 27B: 86,016 tokens with 4096-token chunks

Default changed from 118,784 tokens with 1024-token chunks to 86,016 with 4096 (the largest that fits with the same
~1 GiB left for other apps). `arc-server start -Model 27b -Ctx 118784 -Chunk 1024` (or `pi-arc --ctx 118784
--chunk 1024`) gives the larger context; Pi's configured window stays 86,016.

## 2026-10-10 (evening): prompts at 90 % of the context

GPU memory is now 18.34 GiB (`CL_DEVICE_GLOBAL_MEM_SIZE`, was 16.5); the desktop and other apps hold 0.7–1.0 GiB of
it, and the driver ~0.22 GiB on top of the server's own buffers (Windows counter `GPU Process Memory(*)\Shared Usage`).

### Measured: a prompt of 90 % of the default context (arcinfer's own sources, `arc-server` settings)

| model | context | prompt | prompt processing | time to first token | generation after it |
|---|---|---|---|---|---|
| Qwen3.8 27B (before) | 90,112 | 81,100 | 82.5 t/s | 16.4 min | 4.92 t/s (MTP) |
| Qwen3.8 27B (2048-token chunks) | 106,496 | 95,846 | 78.6 t/s | 20.3 min | 4.55 t/s (MTP) |
| Qwen3.8 27B (now, 1024-token chunks) | 118,784 | 106,906 | 72.8 t/s | 24.5 min | 4.59 t/s (MTP) |
| Gemma 4 E4B | 131,072 | 117,965 | 165.5 t/s | 11.9 min | 11.6 t/s (drafter) |
| Qwen3.5 4B | 262,144 | 235,929 | 144 t/s | 27.3 min | 9.0 t/s (MTP) |

(35B-A3B, 9B and Gemma 4 12B were not run to the end.) At that depth attention takes a large share: ~38 % of the
27B's time at 81K, about half of the 4B's at 236K, most of the Gemma 4 12B's.

### Prompt attention (`bench/attn_bench`, new `ATT_P0`: only the queries from a position on, i.e. one chunk at depth)

- `attn_q8` at depth: 1.9–2.2 T MAC/s for the Qwen shapes (head 256), 0.87–0.91 for Gemma's head-512 global layers,
  which spill 1984 bytes of registers. Ablations on the 27B shape (compile-time `ABL_NOQK` / `ABL_NOSM` / `ABL_NOPV`):
  tile loads + barriers 19 %, QK 24 %, P·V with P packing 33 %, online softmax 19 %.
- **Gemma 4 12B global layers now use `attn_ps`** (new, `kernels/ops.cl`): the head dims are split over sub-groups
  (4 groups of 128 dims, 4 q heads each, 2 tokens per work-group) and partial scores are summed through SLM, so no
  registers spill: 1.25–1.4x faster at every prompt length (0.96 → 1.34 T MAC/s at 4K, 0.87 → 1.10 at 102K). KL vs
  the f32 reference on 160 code positions 0.0174 (0.0180 before); greedy output and drafter tests unchanged.
  `ARC_NO_PS=1` switches back. With one scale per row (Qwen) `attn_ps` is bit-identical to `attn_q8`
  (`ATT_CMP=1` checks it), but it is slower there (1.0–1.6 T MAC/s): splitting the dims doubles the per-row softmax
  and P packing work, which already take ~40 %. E4B's global layers (4 q heads) gain nothing either.
- Tried and removed: 32-key tiles (one max reduction and barrier pair per 32 keys): the extra live state spills
  (2–3 KB) and is 14–20 % slower for Qwen; 4 heads per sub-group: +19 % on the 4B / 35B shapes but spills.
  `attn_q8` is at the register limit of the Qwen shapes.

### Qwen3.8 27B: context 90,112 → 118,784 tokens

- The Qwen KV scale buffers were sized for one scale per 64 dims while Qwen uses one per row: 1,632 bytes per token
  less (36,992 → 35,360; −0.14 GiB at 90K).
- `arc-server` runs the 27B with 1024-token prompt chunks (`--chunk 1024`), which frees 1.1 GiB of activation
  buffers (GPU buffers at 106,496 tokens: 4096-token chunks ~17.5, 2048 16.78, 1024 16.41 GiB). Same outputs. Time
  to first token, 11,240 tokens: 90.1 s (4096), 91.9 s (2048), 94.0 s (1024); 33,937 tokens: 321.6 s (2048),
  322.2 s (1024); at 96K–107K the per-16K-segment times equal those of 4096-token chunks.
- At 118,784 the server holds 17.29 GiB; with 0.88 GiB of other apps that is 18.17 of 18.34 GiB. The previous
  90,112 with 4096-token chunks needs 17.62 GiB: over the budget once other apps hold more than ~0.7 GiB (paging).
- Not done: switching the chunk size during a session (2048 while the context is small). The KV cache is
  allocated for the whole context at load, so the memory for larger chunk buffers is never free; it would need a
  growing KV cache, for ~2 s on an 11K-token first prompt and nothing on long prompts or ordinary agent turns.

### Streaming

- A character whose UTF-8 bytes are split over two tokens (e.g. emoji) was streamed in two invalid pieces (the
  client shows two replacement characters). `emit` now holds back an incomplete character at the end of the text
  (Qwen and Gemma parsers); `tools/tool_stream_test` checks every delta byte by byte.

## 2026-10-10 (later): streamed tool calls

- **Tool calls are streamed while they are generated** (`tools/server.cpp`, `chat::partial_tool_calls`,
  `gemma4::partial_tool_calls`). Before, the server held a tool call back until the end of the response and sent it
  in one piece, so Pi showed nothing while the model wrote a file through its `write` tool (one call: 2,368 tokens,
  6 minutes without output). Now the first delta of a call carries its id and name, and the following ones the
  arguments JSON piece by piece, as OpenAI and llama-server send them. String parameters grow token by token;
  Qwen's non-string parameters (e.g. the `edits` array of Pi's `edit` tool) appear once the parameter is closed,
  because the model's JSON may still turn out invalid (it then becomes a string, as before). Gemma 4 delimits its
  strings, so all of its arguments stream. The pieces of a call only grow and end as the arguments of the
  whole-response parser, so non-streaming responses are unchanged.
- `tools/tool_stream_test.cpp` checks this for every prefix of hand-written and 3,000 random responses (both
  formats); part of `tests/run_tests.ps1`. The web UI and `tests/test_tools.py` join the pieces by index.
- Gemma 4 tool-call parser: an array closed by a stray `}` (`call:x{a:[1}`) made it loop forever while allocating
  memory (the server hung). A stray `}` now ends the array.
- `run.exe` warns when other GPU processes (`server.exe`, `run.exe`, `llama-*`) are running. This explains the
  "`run.exe --mtp` on E4B at 3–5 tok/s" below: alone, `run.exe --mtp` gives 23 tok/s on E4B (17 without drafting;
  20.5 after a 13.9K-token prompt), but with a 27B server processing a prompt at the same time it gave 8 tok/s
  (5–6 without drafting), and with an idle 27B server loaded the E4B load took 13–43 s instead of 2 s. The iGPU
  and its 16.5 GiB are shared between processes.

## 2026-10-10: token generation for Gemma 4 (12B and E4B)

Goal: faster generation on Gemma 4 12B and E4B, at short and long context, without changing the model's output.
The Gemma 4 assistant drafter (speculative decoding) was deliberately left as it was.

### Where the time went (Gemma 4 12B, before)

Profile of a single-token step after a short prompt: 134 ms per token (7.45 tok/s), 139–144 ms of GPU time when
profiled. The 7.4 GB of weights would take ~100 ms at the 72–77 GB/s the iGPU reads from memory, so ~30 % was
overhead:

| part | ms / token | note |
|---|---|---|
| FFN GEMVs (gate, up, down) | 78 | 61–71 GB/s, close to the limit |
| lm_head (262,144 × 3840, Q6_K, 826 MB) | 21–28 | **30 GB/s**, half the speed of the same kernel in isolation |
| attention GEMVs (q, k, v, o) | 24 | 50–60 GB/s |
| small kernels (attn_prep, swiglu, norms, attention, combine) | ~11 | attn_prep alone 55 µs per layer |

At 13.9K context the step grew to 157 ms: decode attention of the 8 global layers (head size 512, 16 q heads on
one kv head) ran at only ~20 GB/s, 0.7 ms per layer.

### Changes

1. **Output head allocated first** (`src/engine.h`, `load_weights`). The GPU read the 826 MB head buffer at about
   half speed when it was allocated after gigabytes of other weights (the same kernel on the same data took
   13–14 ms in isolation, 21–28 ms in the engine; loading the head first gives 14 ms in the engine). Applies to
   every architecture.
2. **Gemma's final logit soft-capping on the GPU.** `c·tanh(x/c)` is now the epilogue of the head's GEMV
   (`-DSOFTCAP`, `kernels/decode.cl`), instead of a host loop over 262K logits per token whose OpenMP worker
   threads kept spinning between tokens.
3. **RoPE table once per pass** (`rope_table` kernel). `attn_prep_g` used to compute `pow`, `sin` and `cos` for
   every element in all 48 layers; the angles are now computed once per pass for the sliding and the global layer
   type and read from a table (bit-identical values). attn_prep: 3.4 → 1.9 ms per token.
4. **Fused residual + norm + quantization** (`add_norm_quant`). For single-token and verification passes, the
   post-attention / post-FFN sandwich norm with the residual add and the next block's input norm and int8
   quantization run as one kernel (2 launches fewer per layer). Output is identical to the two separate kernels.
5. **Quantization of the SwiGLU and attention outputs** for single-token passes: one sub-group per 256-value block
   of the valid rows instead of one work-group per padded row (swiglu_quant 2.4 → 1.2 ms per token).
6. **Decode attention**:
   - New kernel `attn_dsplit` (`kernels/ops.cl`): a work-group covers one key split of one kv head; its sub-groups
     split the head dimensions (64 dims each) and the q heads, so every byte of the K/V cache is read once for all
     heads and up to 4 query rows. Partial scores of the dim groups are summed through local memory; softmax and
     P·V use the same int8 math as before. Used for Gemma 4 (not for the drafter's own layers, not for Qwen).
   - K/V tiles in local memory are single-buffered (`attn_q8`, `attn_split`): twice the work-groups per Xe-core.
     Decode with head size 512: 1.5× faster; prompt processing unchanged or up to 4 % faster.
   - New `attn_combine`: the lanes of a sub-group take the per-split maxima 16 at a time, no data-dependent
     branch; 3–5× faster (26 → 5 µs for 33 splits).

   Attention + combine per layer, `bench/attn_dec_bench` at 14K context:

   | layer type | rows | before | after |
   |---|---|---|---|
   | 12B global (16 q / 1 kv, head 512) | 1 | 742 µs | 310 µs |
   | 12B global, 36K context | 1 | 1,818 µs | 803 µs |
   | 12B sliding (window 1024) | 1 | 125–220 µs | 73 µs |
   | E4B global (8 q / 2 kv, head 512) | 1 | 1,008 µs | 420 µs |
   | E4B sliding (window 512) | 1 | 56 µs | 17 µs |
   | 12B global, verification | 2 | 779 µs | 520 µs |
   | E4B global, verification | 2 | 944 µs | 465 µs |
   | 12B sliding, verification | 2 | 142 µs | 78 µs |

   At 14K context this saves about 6–8 ms per token on the 12B (8 global + 40 sliding layers, plus attn_prep) and
   ~5.5 ms on the E4B (7 global + 35 sliding layers); at 36K the 12B's global layers alone save 8 ms.
7. **Server sampler** (`tools/server.cpp`): the top-k candidates are taken from the tokens within 40 temperatures
   of the best logit (one pass) instead of building and partially sorting a pair for each of the 262K vocabulary
   entries per token; when fewer than k candidates are left after penalties it falls back to the full vocabulary,
   so the sampled distribution is exactly the same.

### Results

- Gemma 4 12B, short context, 128 tokens, same session A/B: 7.79–7.83 → 8.09–8.16 tok/s without drafting
  (a later run on a warm laptop: 7.66); with the assistant drafter 8.74 tok/s (50 % of drafts accepted,
  1.94 tokens per main-model pass). GPU time per token 139–144 → 123 ms (profiled).
- Gemma 4 E4B, short context: 16.9 tok/s without drafting; with the drafter through the server 17.4–18.1 tok/s on a
  story prompt (40 % of drafts accepted; code accepts more).
- Long-context end-to-end generation after these changes was not measured; the per-layer attention savings above
  were. Before them, the 12B generated 6.4 tok/s after a 13.9K-token prompt (llama.cpp Vulkan 5.5).
- At short context the end-to-end gain is within the laptop's run-to-run variation (±4 %); the gains are at long
  context, where attention grows.

### Accuracy

- `tests/run_tests.ps1` with Qwen3.5-4B and Gemma 4 12B: all 10 checks pass (Gemma KL vs the f32 reference
  0.02927, greedy equal to llama.cpp, drafter output identical, Qwen MTP output identical).
- New `run --kl-decode`: feeds a prompt token by token through the generation path and reports the KL against the
  f32 reference at every position. Gemma 4 12B, 160 code positions: 0.013 (0.020 before the changes); 120 raw-text
  positions: 0.029 (0.034 before). A 20-position test swung between 0.009 and 0.043 between builds whose
  intermediate values differ by at most 1e-6: one near-tied position flips. Use the longer references.

### Tried and not kept

- IQ4_XS (E4B) weight decoding: a 256-entry byte-pair lookup table was slower (local-memory bank conflicts,
  66 → 55 GB/s), and register shuffles with a variable lane index halved the speed. The 16-byte table stays.
- `attn_dsplit` for Qwen: up to 2× faster for 2-row verification, but the verification rows are then no longer
  bit-identical to single-token decoding, which the MTP regression test requires. Qwen keeps the previous kernels.
- Drafter changes (native Q4_0 GEMV for the drafter's weights, which halves their bytes; draft argmax on the GPU
  instead of reading 1 MB of logits per draft): written and tested, then reverted on request. The drafter runs as
  before.

### New tools

- `bench/gemv_bench.cpp`: the single-token GEMV on real tensors of a GGUF (all layers in turn, so weights stream
  from memory), checked against the CPU dequantization; variants via build defines or another kernel file.
- `bench/attn_dec_bench.cpp`: decode attention (attn_dsplit / attn_split / attn_decode + combine) for 1–4 query
  rows at any head shape and window, checked against a CPU reference.
- `run --kl-decode` (with `--kl-ref` of every position) and `ARC_PROF_LAYERS=n` (profile per layer group).

### Known issues

- ~~`bin\run.exe --mtp` on Gemma 4 E4B generates at only 3–5 tok/s~~: not reproducible on an otherwise idle GPU
  (23 tok/s); the slow runs shared the GPU with other processes (see the entry above).
- E4B still has headroom: its q/o projections (IQ4_XS, K = 2560) run at 38–49 GB/s, the per-layer-embedding
  projections (256 rows, or K = 256) at 11–22 GB/s, and its small kernels take ~7 ms per token.
- Gemma 4 audio input (12B): the code compiles and is loaded when the mmproj has the audio projector; not yet tested.

## Earlier in this round (before 2026-10-10)

- Qwen3.6-35B-A3B, ISTA-DASLab GSQ 2-bit: `tools/convert_gsq.py` (safetensors → GGUF with the exact 2-bit expert
  codes) and `kernels/moe.cl` (routing, counting sort, DP4a GEMMs on the 2-bit codes). 11K-token prompt in ~14 s,
  ~15 tok/s generation, KL vs f32 0.0008.
- `gelu_tanh` attribute fix in `kernels/ops.cl`.
- Gemma 4 at long context: attention for head size 512 and windows, the assistant drafters, a persistent kernel
  binary cache (`%LOCALAPPDATA%\arcinfer\kernels`), lm_head tuning.
- Gemma 4 accuracy: one int8 scale per 64 dims for q/K/V (was one per row): KL on code 0.23 → 0.018.
- Segmented layer-major prefill (16K tokens) and the largest contexts that fit 16.5 GiB for every model
  (`scripts/arc-server.ps1`, `scripts/install-pi.ps1`).
- Review page `docs/review.html` (EN/ID) with `bench/review_data.py`; prompt benchmark `bench/prompt_bench.py`.
