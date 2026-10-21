# F21.11 — Q2 speed work log / handoff

Status: **planned, not implemented**. This file is the shared work log: the agent doing the code writes the
implementation notes below; Claude adds measured results in the same file.

## Why this file

The accuracy target is already met with `ARC_Q2=1`:

| Model | default KL | Q2 KL |
|---|---:|---:|
| Gemma 4 26B | 0.0985 | **0.00446** |
| Gemma 4 12B | 0.0327 | **0.00026** |
| Gemma 4 E4B | 0.00100 | **0.00003** |

The remaining problem is speed: Q2 generation was 1.8–3.2 tok/s before the GELU fix and ~7.9–9.3 tok/s on the
26B after it (to be re-measured). This work must not change the quantizer, the model weights, or the Q2 KL floor.

## Budget / scope

With a remaining API budget of about **$2**, the full F21.11 plan (fused GEMV + fused GEMM + fused MoE + flash
decoding attention + reduced KV memory + server cache keys) is **not realistically completable and verifiable**.
A realistic single increment is:

1. **Q2 cache-key/state safety** (small, required before server use), or
2. **fused single-token GEMV** (highest generation-speed impact).

Do not start with the attention rewrite; it is the largest part and needs GPU measurements.

## Phase 0 — make Q2 explicit (required before server use)

Current state:

- `zero/engine.h:q2_init()` reads `getenv("ARC_Q2")` directly.
- `tools/server.cpp:ctx_key()` does not include Q2.
- session/prefix state layout does not distinguish Q2.
- Q2 adds f32 K / f16 V rows and residual activation buffers; `ctx_bytes`/`slot_bytes` already account for some of
  this, but the saved-state header must too.

Changes:

- add `bool q2` to `EngineOptions`; initialize from `--q2` / `ARC_Q2`;
- keep the env fallback for `runz` compatibility;
- add `q2` to `ctx_key` and to the session/prefix layout suffix (`-q2`);
- make `save_state`/`load_state` reject a header whose Q2 flag differs;
- include the f32 K / f16 V segments in Q2 save/load (today `save_state` saves the int8 cache);
- add a server option `--q2` and document that Q2 is still opt-in;
- no speed claim from this phase.

Acceptance: Q2 state cannot be loaded by a non-Q2 server; non-Q2 files are byte-for-byte unchanged.

## Phase 1 — fused single-token GEMV (best speed/effort)

Today `matmul()` runs `matmul1()` twice:

```cpp
matmul1(name, w, Mp, C, ldc, c_off, acc, act32);   // value pass
aq_ = from(aq_, q2_aq_); ...                       // residual pass
matmul1(name, w, Mp, C, ldc, c_off, 1, act32);
```

For `cur_T_ == 1`, `matmul1()` reaches `gemv` / `gemv_v` in `zero/kernels/decode.cl`. Each pass reads every weight
row again.

Implement:

- add Q2 activation arguments to `gemv` and `gemv_v`: residual `aq2`, `asc2`, `absum2`, and a runtime `q2` flag;
- inside the superblock loop, load both activation words/scales and keep two accumulators;
- for `Q2_K/Q4_K/Q5_K`, use the residual `absum2` for the min correction;
- write `HEAD_OUT(reduce(acc_value) + reduce(acc_residual))`. (Checked 2026-10-21: the two-launch form did cap each
  pass, `HEAD_OUT(v1) + HEAD_OUT(v2)`, a bug; fixed: in Q2 the head's passes run without the cap and `softcap_rows`
  caps the sum. KL 26B 0.00446 -> 0.00443, top token 473 -> 475 / 480; 12B 0.00026 -> 0.00019. A fused head GEMV
  must cap once, after both accumulators are added.)
- in `matmul()`, when `q2_ && !q2_off_ && cur_T_ == 1`, call `matmul1(..., q2=true)` once and skip the second launch;
- update the direct `gemv` calls (the MTP draft head at `zero/engine.h:5192`) with Q2=false arguments;
- leave `gemv_mt` / `gemv_mt_v` (T=2..8 verification) on the two-launch path for this phase.

Expected: the weight traffic for single-token matrices drops from 2× to 1×; this is the main generation cost.
Target is not a promise: 26B ~14–17 tok/s at 4K if attention is not dominant; measure before/after.

Acceptance:

- KL against `tools/ref_forward` ≤ 0.01 and not worse than the current Q2 result;
- greedy text identical to current Q2 (or to the f32 reference where current Q2 differs);
- MTP/DFlash2 verification remains bit-identical;
- A/B runs alternate Q2 fused vs Q2 two-launch; two runs per mode, report both.

## Phase 2 — fused GEMM (prompt)

Add a Q2 variant of the prompt GEMM that reads a weight tile once and accumulates value + residual. Do this after
GEMV and after a GPU measurement; do not change the tiling defaults until measured.

## Phase 3 — fused MoE

`moe_rg` currently runs value and residual passes for gate/up/down. Fuse at the expert tile level; preserve
`moe_gelu_mul` and the GELU epilogue. This matters for 26B, but is larger than Phase 1.

## Phase 4 — attention

- split keys over sub-groups, partial softmax + combine for `attn_fa_g` generation;
- tile prompt rows;
- keep one row per launch for verification identity;
- then reduce KV memory: int8 KV only for the drafter layers, K f32 for the main model, V f16;
- update `ctx_bytes`/`slot_bytes` and `--ctx auto` before enabling by default.

## Measurements Claude should add here

For 26B, 12B, E4B at 4K and 8K prompt, 128–200 generated tokens:

| Model | Mode | Prompt tok/s | Generation tok/s | KL | Greedy |
|---|---|---:|---:|---:|---|
| 26B | default | | | 0.0985 | |
| 26B | Q2 two-launch | | | 0.00446 | |
| 26B | Q2 fused GEMV | | | | |
| 12B | default | | | 0.0327 | |
| 12B | Q2 fused GEMV | | | | |
| E4B | default | | | 0.00100 | |
| E4B | Q2 fused GEMV | | | | |

Also record:

- `runz --slots-check` with `ARC_Q2=1` on 26B/12B;
- MTP/DFlash2 identity;
- `slot bytes: N real, M counted: OK`;
- VRAM / `--ctx auto`;
- any regression on Qwen 27B/35B/Flash Next.

## Decision rule

Do not make Q2 the default until:

- KL ≤ 0.01 for all three Gemma models;
- Q2 generation is within roughly 10–15 % of the default mode, or the user explicitly accepts the trade-off;
- session/prefix cache files are Q2-safe;
- MTP/DFlash2 and slots tests pass on GPU.

Until then keep `ARC_Q2` opt-in and keep the fast default unchanged.
