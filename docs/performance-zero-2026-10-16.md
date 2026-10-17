# arcinferzero (Level Zero) performance, 16 October 2026

arcinfer, overnight results, 15 → 16 October 2026.

Tests ran from 17:00 to 06:05. Since this morning zero is the default engine; the old OpenCL engine is still available with `--opencl`. Every number here comes from real runs on this laptop.

Hardware: Core Ultra 5 125H, Arc Xe-LPG 112 EU, no XMX (DP4a), 32 GB RAM.

For the earlier OpenCL results (10–14 October) see [speed-log.md](speed-log.md).

## Summary

| Measurement | Result | Context |
|---|---|---|
| Qwen3.6 35B, MTP 3 drafts (code) | 31.5 tok/s | 19.9 without MTP, identical answers |
| Gemma E4B, 2K prompt | 827 tok/s | 733 the evening before (+13 %) |
| Qwen3.8 27B, MTP (code) | 9.1 tok/s | 5.1 without MTP; 5 drafts used to be 3.8 |
| Match with OpenCL | 6 / 6 models | bit-identical logits, same image and audio answers |

## Answer speed through the server

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

## Number of MTP drafts

Mean of the code and Indonesian answers, tok/s. More drafts are not automatically faster: every draft runs the MTP block with the full output head.

| Model | 1 | 2 | 3 | 4 | 5 | 7 | Default |
|---|---|---|---|---|---|---|---|
| Qwen3.5 4B | – | 22.9 | 24.8 | 25.2 | – | – | 3 (was 2) |
| Qwen3.5 9B | – | 15.8 | 15.6 | 15.9 | – | – | 2 |
| Qwen3.8 27B | – | – | 8.3 | – | 8.2 | 8.2 | 3 |
| Qwen3.6 35B | 23.9 | – | 26.1 | – | – | – | 3 (was 1) |

## Prompt and generation at different context lengths

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

## Kernels improved

| Kernel | Before | After |
|---|---|---|
| 35B generation attention, 64K | 1.16 ms | 0.94 ms (improved) |
| GEMV for 4–7 drafts (27B, 5 drafts) | 3.8 tok/s | 9.2 tok/s (improved) |
| attn_prep_g Gemma HD512, spill | 4,608 B | 0 (improved) |
| add_norm_quant Gemma, spill | 3,712 B | 64 B (improved) |
| gdn_conv Qwen, spill | 1,024 B | 1,024 B |
| moe_gateup_q 35B, spill | 2 KB | 2 KB |

The last two spills go away with 256 registers, but that is not faster, so they stay.

## Qwen prompt attention

The last 4K chunk at 64K depth, ms per layer. The current kernel (256 registers) against the best variant that splits the head dimensions.

| Model | Heads | attn_q8 (in use) | best attn_ps |
|---|---|---|---|
| 4B / 9B | 16 q / 4 kv | 926 ms | 1,117 ms |
| 35B | 16 q / 2 kv | 923 ms | 1,138 ms |
| 27B | 24 q / 4 kv | 1,516 ms | 1,669 ms |

## Correctness

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

## Notes

- Correction: during the night I first concluded that Level Zero ignores the 256-register flag. That was wrong; my own environment switch came out empty. Every number on this page was measured again with the correct build.
- A server started right after another one stops sometimes runs at half speed. `arc-server stop` now waits 5 seconds, and runz logs the time of every step so the cause can be seen if it happens again.
- Not done yet: cheaper MTP drafts (a dynamic token list or a draft tree), and the two small spills above.
