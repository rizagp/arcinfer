# Qwen3.8 Flash Next (qwen4exp) — implementation spec for arcinferzero

Source: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S, 2 parts (C: Hugging Face cache; backup on D:).
Reference: llama.cpp `src/models/qwen4exp.cpp` + `conversion/qwen4exp.py` (read 2026-10-16). Notes written before any
code, so the implementation can be checked against them.

Status 2026-10-23: implemented in arcinferzero (README "Supported models"). KL against the f32 reference 0.048 on 363
tokens (above the 0.015 limit: QUEUE.md F21.13, where the error comes from); prompts 160-190 tok/s at 4K (F21.15);
generation 6-7 tok/s. incoai's DFlash2 drafter (trained on layers 5 / 15 / 25 / 35 / 45 averaged over the 4 HC
streams) is downloaded but not supported yet (F21.16).

## Hyperparameters (GGUF metadata)

| Key | Value |
|---|---|
| layers | 48: linear attention (Gated DeltaNet) except every 4th (3, 7, …, 47) = full attention |
| n_embd | 2560; hyper-connections: 4 streams (hc_dim 10240), low rank 320 |
| attention | 24 q heads, 2 kv heads, head 256, q+gate interleaved per head in `attn_q` [2560, 12288], q/k RMS norm, IMRoPE sections [11, 11, 10, 0], n_rot 64, base 1e7, sigmoid output gate |
| QSA (sparse attention) | indexer 4 heads x 128, top_k 2048, compress (pool) ratio 4 on every attention layer |
| DeltaNet | 16 k heads, 48 v heads, head 128 (conv dim 10240, d_inner 6144), conv kernel 4; **output gate sigmoid, not silu** (the one difference from Qwen3.5) |
| MoE | 512 experts, 10 used, softmax gating, top-k weights renormalized; expert FFN 640; shared expert FFN 640 with sigmoid gate `ffn_gate_inp_shexp` |
| PLE | layer 1 only; n-gram 3 (2- and 3-grams), 8 heads per n-gram = 16 heads x 160 = 2560; table `per_layer_token_embd` IQ4_NL [160, 320,001,536] in part 2 (26.8 GiB) |
| vocab | 248,320; eos 248046; no MTP block in this GGUF |
| eps | 1e-6 |

## Forward pass

- `res` = 4 copies of the token embedding: [n_embd, 4] per token.
- Layer il: (PLE at il = 1) → `hc_mix(res, hc_attn_*)` → DeltaNet or attention → `hc_combine` → `hc_mix(res,
  hc_ffn_*)` → MoE + shared expert → `hc_combine`.
- End: `hc_mix(res, output_hc_*)` without inject = the output norm (there is no other) → `output` (Q6_K).

`hc_mix(x [n_embd,4], w_norm [n_embd,4], W_down [10240→320], W_up [320→10240], W_inject [10240→4])`:
- `xn` = RMS norm of each stream separately (over its 2560), times `w_norm` (already 1 + w).
- `lo = silu((W_down · xn) / 4)`; `gate = W_up · lo` [10240].
- `mixed = mean over the 4 streams of xn_c * sigmoid(gate_c)` → [2560] block input.
- `inject = W_inject · xn` [4].

`hc_combine(res, out, inject)`: `w_c = 2 * sigmoid(inject_c / 4)`; `res_c += w_c * out` (every stream gets the
same block output, scaled by its own weight).

Attention (full layers), as Qwen3.6: `attn_q` → q and gate per head (q = first 256, gate = next 256), q_norm / k_norm
RMS, IMRoPE on 64 dims, softmax attention scale 1/16, output * sigmoid(gate), `attn_output`.

QSA (only matters past ~2,048 visible tokens: with n_pool <= 512 every block is selected, so short contexts are plain
causal attention):
- indexer key per token: `k_raw = index_k_proj · x` [128] (raw, cached).
- blocks of 4 consecutive cells (by sequence order): pooled key = mean of the 4 raw keys → RMS norm (index_k_norm) →
  RoPE at the block's first position.
- indexer query: `index_q_proj · x` → [4 heads, 128] → RMS norm → RoPE.
- score(block) = sum over heads of relu(q_h · k_block) / sqrt(128) (lightning indexer with unit weights); causal mask.
- attend to the cells of the top 512 blocks + the incomplete tail (last < 4 cells) — dense attention restricted to
  those cells.

DeltaNet: as Qwen3.5 (qkv conv 4 taps + silu, l2 norm q/k, beta = sigmoid, gate = softplus(alpha + dt_bias) *
ssm_a), then RMS norm (ssm_norm) * **sigmoid(z)**, `ssm_out`.

MoE: router `ffn_gate_inp` (BF16) → softmax over 512 → top 10 → renormalize; experts silu(gate)·up → down; plus
shared expert (gate/up/down 640) * sigmoid(`ffn_gate_inp_shexp` · x).

PLE (layer 1, applied to `res` before the layer's hc_mix):
- 16 row indices per token from a host-side hash: context `t[p], t[p-1], t[p-2]` (missing or EOS predecessors cut
  the window: everything at or before an EOS reads as EOS; the token's own EOS does not cut). For n = 2..3:
  `mixed = (t0*m0) ^ (t1*m1) [^ (t2*m2)]` (uint64), heads h = (n-2)*8 + g: `row = mixed % vocab[h] + offset[h]`.
- emb = 16 table rows of 160 → [2560].
- `key = ple_key · emb` [10240], `value = ple_value · emb` [2560].
- key and query (= res) RMS-normed per stream with `ple_norm_key` / `ple_norm_query`; `s_c = (key_c · query_c) /
  sqrt(2560)`; gate_c = sigmoid(sign(s) * sqrt(max(|s|, 1e-6))).
- gated_c = value * gate_c; normalized = per-stream RMS norm (`ple_norm_conv`).
- depthwise causal conv over 10240 channels, 4 taps, dilation 3 (history 9 tokens, a recurrent state),
  `out[c,t] = sum_k w[k,c] * x[c, t - (3-k)*3]`, then silu.
- res += gated + conv_out.

## Weight types

BF16 (30) for router, hc_*, indexer, ple_key/value; F32 norms; F16 ple_conv1d; Q4_K / Q5_K / Q6_K / IQ4_XS / IQ3_XXS /
IQ2_S for dense and expert gate/up; **IQ4_NL (20)** and **Q2_0 (42)** for expert / shared `ffn_down`.

- IQ4_NL: block 32, f16 d + 16 bytes of nibbles; value = d * kvalues_iq4nl[nibble] (the IQ4_XS table), low nibbles =
  elements 0..15, high = 16..31.
- Q2_0: block 64, f16 d + 16 bytes; element j = bits (2*(j%4)) of byte j/4; value = (q - 1) * d, q in 0..3.

## Risks / design points

1. **K = 640** for every `ffn_down` (expert and shared): arcinfer assumes K % 256 == 0 (superblocks, activation scales
   per 256 or 32). Decision (2026-10-16): treat consecutive small blocks as a 256-element "superblock" (IQ4_NL: 8 x 18
   = 144 bytes; Q2_0: 4 x 18 = 72 bytes; both multiples of 8 for block reads) and pad each down row from K = 640 to 768
   with zero blocks (scale 0) when the experts are copied into the GPU cache: +20 % for the down projections only
   (converting to Q8_0 instead would grow the experts to ±51 GB). The down input (silu(gate)·up, 640) is padded with
   zeros to 768 as well, so the products are unchanged. GPU decoders for QT 20 / 42 read these padded rows.
2. Expert streaming: 46.8 GiB of experts, ±2 MiB each, read directly from the GGUF (no separate expert file); GPU cache
   ±12 GiB (25 %). Per token ±0.94 GB of experts; a 4,096-token prompt chunk touches nearly all of them.
3. PLE table rows (160 x IQ4_NL = 90 bytes) are read from the SSD per token on the CPU: 16 random reads per token.
4. QSA state: a pooled-key cache per attention layer (blocks of 4) next to the KV cache.
5. Hyper-connections change the residual from [2560] to [4, 2560] everywhere, including snapshots / rollback and the
   session cache.

## Order of work

1. Loader: types 20 and 42 (CPU dequant + gguf_dump), split GGUF, tensor map; CPU f32 reference of the forward pass
   for a few tokens (numpy or C++, per layer dumps) — the yardstick for everything below.
2. GPU: hc_mix / hc_combine kernels, sigmoid-gated DeltaNet output, dense attention (short contexts), MoE 512 / top-10
   with K = 640 down, shared expert; expert streaming from the GGUF.
3. PLE (host hash, SSD rows, conv state).
4. QSA for long contexts.
5. Server: chat template (Qwen3.8 style, reasoning effort), tool calls, sessions; speed and acceptance tests.
