# Using arcinfer

arcinfer is an LLM inference engine built specifically for this laptop (Intel Core Ultra 5 125H, integrated Intel
Arc GPU). Its focus is processing long prompts quickly, for example when an agent such as Pi reads many code files.
It runs GGUF models directly from the Hugging Face cache and serves an OpenAI-compatible API, so it can be used by
Pi, by other apps, or through its built-in web UI.

## 1. Setup (once)

Open PowerShell or Terminal in the arcinfer folder, then run:

```powershell
.\setup.cmd
```

The setup is safe to run again (steps that are already done are skipped). It:

- checks the requirements: 64-bit Windows, an Intel Arc GPU, Visual Studio Build Tools (C++), Python and Pi, and
  says how to install whatever is missing;
- builds `bin\server.exe`, `run.exe` and `clinfo.exe` when they are missing or older than their sources, then
  checks the GPU (OpenCL 3.0, DP4a, GPU memory);
- asks which models to download into the Hugging Face cache (`.\setup.cmd -Models 27b,4b` without asking; Gemma
  comes with its drafter and image projector; the 35B is downloaded and then converted). `huggingface_hub` /
  `numpy` are installed with pip when needed (it asks first);
- adds the `arcinfer` provider to `~/.pi/agent/models.json` with every model it finds and each one's context
  (other providers are left unchanged, the previous file is kept as `models.json.bak`), and installs the `pi-arc`
  and `arc-server` commands into `~/.pi/agent/bin` (already on the PATH), so they can be called from any folder.

`.\setup.cmd -Tests` also runs the regression tests.

Available models (setup downloads them into `%USERPROFILE%\.cache\huggingface\hub`):

| Command | Model | GGUF file |
|---|---|---|
| `pi-arc 27b` | Qwen3.8 27B | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` → `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |
| `pi-arc 35b` | Qwen3.6 35B-A3B (MoE) with MTP and images, ~131K context automatically (262K with `--ctx 262144`) | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` → `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (or `UD-Q3_K_M.gguf`) + `mmproj-BF16.gguf` (downloaded by hand with `hf download`); if it is not there: the GSQ 2-bit version `ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ`, converted (see below; without MTP) |
| `pi-arc g12` | Gemma 4 12B (+ images) | `google/gemma-4-12B-it-qat-q4_0-gguf` → `gemma-4-12b-it-qat-q4_0.gguf` + `mmproj-gemma-4-12b-it-qat-q4_0.gguf`; drafter `mtp-gemma-4-12B-it.gguf` from `unsloth/gemma-4-12B-it-qat-GGUF`, copied into the snapshot folder of Google's model |
| `pi-arc e4b` | Gemma 4 E4B | `google/gemma-4-E4B-it-qat-q4_0-gguf` → `gemma-4-E4B_q4_0-it.gguf`; drafter `mtp-gemma-4-E4B-it.gguf` from `unsloth/gemma-4-E4B-it-qat-GGUF`, copied into the snapshot folder of Google's model |
| `pi-arc 9b` | Qwen3.5 9B | `unsloth/Qwen3.5-9B-MTP-GGUF` → `Qwen3.5-9B-UD-Q4_K_XL.gguf` |
| `pi-arc 4b` | Qwen3.5 4B | `unsloth/Qwen3.5-4B-MTP-GGUF` → `Qwen3.5-4B-UD-Q4_K_XL.gguf` |
| (no short alias) | Qwen3.8 Flash Next (MoE with 512 experts, arcinferzero only) | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` → folder `IQ3_S`, 2 files (54.8 + 28.8 GB): `hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include "IQ3_S/*"` (not downloaded by setup) |
| (optional) | DFlash2 drafters for the 27B / 35B (see section 3) | `hf download z-lab/Qwen3.8-27B-DFlash2` (3.85 GB) / `hf download incoai/Qwen3.6-35B-A3B-DFlash2` (1.05 GB) |

The `35b` choice in `setup.cmd` downloads ISTA-DASLab's GSQ 2-bit version. This version is not available as a
GGUF, so it is converted once (~2.5 minutes, 10.9 GB result in `%LOCALAPPDATA%\arcinfer\models`):

```powershell
python tools\convert_gsq.py "<HF cache>\models--ISTA-DASLab--Qwen3.6-35B-A3B-2Bit-GSQ\snapshots\<rev>" `
  "$env:LOCALAPPDATA\arcinfer\models\Qwen3.6-35B-A3B-GSQ2-arc.gguf" --vocab-from "<any Qwen3.5/3.6 GGUF>"
```

## 2. Using Pi

1. Open a terminal in the project folder you want to work on. In File Explorer: open the folder, right-click an
   empty area, then choose **Open in Terminal**.
2. Run `pi-arc`. On first use, the server is started in the background first (5–40 s until "Ready" appears), then
   Pi opens.
3. Type your requests in Pi as usual.

| Command | What it does |
|---|---|
| `pi-arc` | The model that is running or was used last (the first time: choose from a list) |
| `pi-arc --pick` | Choose a model from the list of every downloaded model, by full name |
| `pi-arc 27b` | Qwen3.8 27B: very smart, slowest |
| `pi-arc 35b` | Qwen3.6 35B-A3B: smart and fast (mixture of experts, only ~3B active parameters per token) |
| `pi-arc g12` | Gemma 4 12B: reads images (attach an image in Pi or the web UI) |
| `pi-arc e4b` | Gemma 4 E4B: small and fast |
| `pi-arc 9b` | Qwen3.5 9B: faster, medium quality |
| `pi-arc 4b` | Qwen3.5 4B: fastest |
| `pi-arc --think off` | No thinking (reasoning) phase: much faster answers |
| `pi-arc --think medium` | Thinks longer for hard tasks (`off` / `minimal` / `low` / `medium` / `high`; default `low`) |
| `pi-arc 9b --think off` | A fast combination for everyday questions |
| `pi-arc -c` | Continue the last session in this folder (`pi-arc -r`: choose a session) |
| `pi-arc -p "explain app/auth.py"` | Other Pi arguments still work, written after the pi-arc options |
| `pi-arc --ctx 32768` | A different context size (default = the largest that fits, see section 5); `pi-arc --ctx auto` goes back to automatic |
| `pi-arc --no-mtp` | No speculative decoding (MTP / drafter) |
| `pi-arc --help` | Every pi-arc option, the server and the files it uses |

Inside Pi, `/model` shows every model; choosing one makes the server switch to that model (5–40 s), with its own
context. The next `pi-arc` uses the model chosen last.

The server keeps running after Pi is closed. The next session is ready right away without loading the model again,
and a repeated prompt (for example Pi's system prompt) comes from the cache. If `pi-arc` is called with another
model, the server switches models automatically.

Pi's system prompt is also saved to disk (`%LOCALAPPDATA%\arcinfer\prefix`), so after the server restarts or
switches models, a new Pi session still starts within a few seconds (27B: ~5 s instead of ~40 s). Pi's system
prompt contains the project folder and its `AGENTS.md`, so the saved prompts are grouped per project folder: each
folder keeps its 2 latest versions and is not pushed out by other folders (27B: ~290 MB per file). All models
together are limited to 8 GB (the least recently used go first; change it with `--prefix-cache-gb`).

A new session always starts empty: Pi only sends its system prompt and your message, so the model does not see the
conversation of an earlier session. To continue a conversation, use `pi-arc -c` or `pi-arc -r`. What the server
processes then:

| Situation | What is processed (27B) |
|---|---|
| right after Pi is closed | only the new message (the server still holds that conversation) |
| after another session or project | only the new message: the session cache (on by default, 8 GB) saves the conversation that was left to disk and loads it again in seconds (`pi-arc --session-cache 0` turns it off) |
| after `arc-server stop` / `restart` or a model switch | only the new message: the conversation being held is saved to disk first and then loaded again (lost only if the server process is killed) |

Pressing Esc in Pi stops the server right away (while reading the prompt: within ~1 s; while writing the answer:
before the next token), so the next message does not have to wait. The next message also does not process the
whole conversation again: only the new message is processed.

## 3. Managing the server

**Through the window:** double-click `arc-launcher.cmd` in the project folder. The window shows:
- every model and its variants (MTP, images, auto context, VRAM at that context, compatible or not);
- for the selected model: how the auto context is counted (GPU memory minus what is kept for Windows, snapshots,
  image encoder, weights, MTP block and buffers, then the rest divided by the memory needed per 1,000 tokens);
- the server option fields: the main ones are shown directly (context, MTP drafts, thinking budget, prompt cache,
  access from other computers); the rest (port, chunk, image tokens, W8 weights, cache folders and sizes,
  `--no-swap`, f32 attention) appear after ticking "Show advanced options";
- monitoring of the running server: model, filled context, VRAM per part, number of requests, the last prompt and
  answer speed, the percentage of MTP drafts accepted.

All fields are filled in automatically with the best values for the selected model, so just choose a model and
press Start. Fields you change turn yellow and only those are sent to the server. "Set as default" saves the
changed fields and makes the selected model the default one (also for `pi-arc` without a model name); "Reset
options" goes back to the best values. The Start / Stop buttons use `arc-server`, so the server keeps running after
the window is closed. (The window text is in English: prompt cache = saved conversations, prefix cache = saved
system prompt.)

The thinking budget in the launcher only applies to clients that do not send their own budget (web UI, other
apps). Pi sends its own budget according to its `/thinking` level: minimal = thinking off, low 2,048, medium 8,192,
high/xhigh 16,384 tokens.

**Through the terminal:**

```powershell
arc-server status              # whether the server is running, which model is loaded, size of the conversation cache
arc-server stop                # stop the server and free the GPU memory (12–16 GB depending on the model)
arc-server clear-cache         # delete every saved conversation (session cache; on by default, turn off with pi-arc --session-cache 0)
arc-server start -Model 9b     # start with a particular model without opening Pi
arc-server start -Model 27b -Ctx 32768 -NoMtp
arc-server start -Model 27b -Ctx auto   # back to the automatic context (restarts a server with a fixed context)
arc-server restart             # restart, for example after the GPU memory setting was changed
arc-server start -Model 35b -MtpDrafts 2 -ThinkBudget 4096 -NoSwap   # other options (see arc-server help)
arc-server help                # every option
```

The server log is `%LOCALAPPDATA%\arcinfer\server.log`. The server uses port 8090.

**arcinferzero (the default engine since 16 Oct 2026):** the server runs through Level Zero (Intel's GPU driver).
Its output is identical to the old OpenCL engine (tested on every model, images, audio, MTP, tool calls, Pi). For
MoE models (Qwen3.6 35B) the experts are kept on the SSD and only the frequently used experts are in GPU memory (a
cache): the context can reach 262K together with images and MTP, and Windows still keeps at least 14 GB of RAM.
The automatic context gives the expert cache priority (up to 90 % of the experts; 35B ~40K tokens), because a small
cache makes generation pause for SSD reads; choose more with `--ctx`. The
old OpenCL engine is still there: the "Engine" field in the launcher, or:

```powershell
arc-server start -Model 35b          # arcinferzero (default)
arc-server start -Model 35b -OpenCL  # old OpenCL engine
pi-arc 35b                           # arcinferzero
pi-arc 35b --opencl                  # old OpenCL engine (the running server is replaced)
pi-arc 35b --no-stream               # Level Zero without the SSD cache: every expert on the GPU (context as small as OpenCL)
```

`-Zero` / `--zero` are still accepted (they change nothing).

On first use, arcinferzero writes the expert file once (35B: 15 GB, about 2.5 minutes) to
`%LOCALAPPDATA%\arcinfer\zero\`. This file can be deleted at any time; when it is missing, it is written again
automatically. The cache size is set automatically (it can be changed: server options `--expert-cache-gb`,
`--expert-cache-pct`, `--windows-reserve-gb`). If GPU memory suddenly runs short because of other apps (e.g.
Chrome), the cache shrinks itself by 1 GB per step instead of crashing.

arcinferzero notes:
- The SSD cache is only for MoE models (35B and Qwen3.8 Flash Next; the experts of the 35B's MTP layer are in the same file). Dense models (27B, 9B, 4B, Gemma)
  only switch engine to Level Zero: context and memory are the same as on the OpenCL engine.
- Expert files are never deleted automatically: the next start uses them right away (35B about 10 s, including
  filling the GPU cache from the last list). They are rewritten only when the GGUF changes; files of an old version
  have to be deleted by hand from `%LOCALAPPDATA%\arcinfer\zero\`.
- GPU kernels are compiled once and then saved (`%LOCALAPPDATA%\arcinfer\kernels`): the first request of a model on
  zero, the first image, or the first use after an Intel driver update feels slower, after that it is normal.

**DFlash2 drafter (`--dflash`, arcinferzero only: `serverz.exe` / `runz.exe`).** This drafter guesses 7 tokens at
once from the main model's hidden state, then the main model checks them in a single pass. Download the drafter
(table in section 1), then pass its folder (the snapshot folder containing `config.json` and `model.safetensors`)
with `--dflash DIR`. The model's MTP block is still used: at each step the server picks DFlash2 while many of its
drafts are accepted, otherwise MTP. Results (greedy, answers identical to running without a drafter):

| Model, kind of text | No drafts | MTP alone | DFlash2 + MTP |
|---|---|---|---|
| Qwen3.8 27B, Python code | 4.91 tokens/s | 15.5 | 18.1 |
| Qwen3.8 27B, Indonesian text | 4.88 | 7.0 | 6.8 |
| Qwen3.6 35B-A3B, Python code | 21.2 | 39.3 | 49.8 (99 % of drafts accepted) |
| Qwen3.6 35B-A3B, Indonesian text | 21.0 | 22.7 | 22.7 |

This drafter was trained mostly on English text and code, so for Indonesian it adds almost no speed (MTP does most
of the work).

**4-bit V copy for long answers at long contexts (`--v4`, arcinferzero, Qwen models, experimental).**
`arc-server start -Model 35b -V4` (or `serverz.exe ... --v4`) keeps the normal int8 attention cache for reading
prompts and adds a 4-bit copy of its V half that generation reads. Prompts stay as fast as without it; generation
deep in a long context is ~7 % faster (35B after a 125K-token prompt: 13.5 -> 14.3 tokens/s), because each new token
reads half the V bytes. It costs memory instead of saving it: the attention cache grows by ~25 %, so the expert
cache of the MoE models gets a little smaller (35B at 131K: 90 -> 87 % of the experts). The answers change slightly
(4-bit values; 35B: perplexity +0.4 %), MTP and DFlash2 still give the same text as plain decoding. Below ~64K
tokens of context it brings nothing; leave it off unless long answers at very long contexts are the main use.

**Qwen3.8 Flash Next (arcinferzero only).** A MoE model with 512 experts fetched from the SSD (like the 35B). Run
`serverz.exe` with `-m` pointing to the first file (`...-IQ3_S-00001-of-00002.gguf`); the second file must be in the
same folder. The first start writes a 56 GB expert file to `%LOCALAPPDATA%\arcinfer\zero\` (~9.5 minutes); later
starts take ~13–25 s. It writes ~6–7 tokens/s at any context up to 131K (the expert cache holds about a fifth of
the experts); prompts: ~150 tokens/s at 8–32K, ~120 at 125K. The difference from the CPU reference is small (KL
0.0014 on a short chat prompt). This model has no MTP block and cannot read images. Its sparse long-context
attention (QSA) works up to the full 262,144 tokens (measured 18 Oct, docs/speed-progress.md).

Stop the server (`arc-server stop`) before using the laptop for other heavy GPU work (games, video editing), or
before moving or rebuilding the arcinfer folder.

## 4. Web UI and other apps

While the server is running:

- **Web UI**: open http://127.0.0.1:8090/ in a browser to chat directly (text files can be attached with
  drag-and-drop; images too for Gemma 4 12B / E4B, Qwen3.5 4B / 9B and Qwen3.6 35B: the **Image** button,
  drag-and-drop, or paste).
- **OpenAI API**: base URL `http://127.0.0.1:8090/v1`, any API key (for example `arcinfer`), model `qwen3.8-27b`
  (or the model currently loaded; other models in the list are loaded automatically). Supports streaming,
  `reasoning_content`, tool calls, `usage`, repetition penalties (`frequency_penalty`, `repeat_penalty`),
  `logit_bias`, `logprobs`, several answers at once (`n`), guaranteed-valid JSON answers (`response_format` with
  `json_schema`), and `tool_choice` (forcing a tool call).
- **Other formats** (same models and features):
  - **Ollama**: apps that have an "Ollama" option (e.g. Open WebUI, many editor plugins) just need to point to
    `http://127.0.0.1:8090` (`/api/chat`, `/api/generate`, `/api/tags`, ...).
  - **Anthropic**: `http://127.0.0.1:8090/v1/messages` (Claude format, including tool_use and thinking).
  - **OpenAI Responses**: `/v1/responses` (used e.g. by Codex CLI); the old **Completions**: `/v1/completions`
    (raw prompt, for code autocomplete).
  - **Audio transcription**: `/v1/audio/transcriptions` (upload an audio file, the result is text; needs Gemma 4
    12B / E4B, which have an audio encoder).
- Embeddings (`/v1/embeddings`) are not available yet: they need an embedding model with an architecture the engine
  does not support yet.
- **Web UI: automatic compaction.** The top right shows the context use ("context 12.3K / 40.9K"). If the next
  message would make the context almost full, the older messages are summarized by the model first, then the
  conversation continues with the summary + the last exchanges. The **Compact** button does this at any time;
  **Stop** cancels it.
- **Well-formed tool calls** (Gemma 4 and Qwen): while the model writes a tool call, the server only picks tokens
  that keep the call valid for the tool's schema (required arguments such as `path` cannot be left out, field names
  only those in the schema). The content is still the model's choice. Turn it off with the variable
  `ARC_TOOL_CONSTRAINT=0`. `ARC_LOG_TOOLCALLS=file.jsonl` logs the raw text of every tool call (to investigate
  failed tool calls).
- **Qwen3.6 35B: `pi-arc 35b --preserve-thinking`**: the thinking of earlier turns stays in the prompt (an official
  option of the Qwen3.6 template; Qwen3.8 27B always does this). Follow-up messages do not process anything again;
  the context fills faster.

Example from PowerShell:

```powershell
$body = @{ model = "qwen3.8-27b"; reasoning_effort = "none"
           messages = @(@{ role = "user"; content = "Hi! Explain arcinfer in one sentence." }) } | ConvertTo-Json -Depth 4
(Invoke-RestMethod http://127.0.0.1:8090/v1/chat/completions -Method Post -ContentType "application/json" -Body $body).choices[0].message.content
```

From Python (the `openai` package):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="arcinfer")
r = client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": "Hi!"}],
                                   extra_body={"reasoning_effort": "none"})
print(r.choices[0].message.content)
```

## 5. Context size, and when a conversation gets long

Each model uses the largest context that fits the GPU memory currently available (set in Intel Graphics Software,
~18–19 GiB on this laptop; the amount the driver offers changes with what other apps run), with ~1 GiB left for
Windows and other apps, or the model's training limit. The context adjusts itself when that setting is changed (run
`arc-server restart` afterwards). The launcher window shows the current numbers. Examples:

| Model | Default context |
|---|---|
| Qwen3.8 27B | ~41K tokens with images + MTP (larger without images: `27b-mtp`, or without both: `27b-std`) |
| Qwen3.6 35B-A3B | 131,072 automatically (images + MTP + DFlash2, ~80 % of the experts cached); up to 262,144 with `--ctx` (smaller expert cache) |
| Qwen3.8 Flash Next | 131,072 automatically; up to 262,144 with `--ctx` |
| Qwen3.5 9B / 4B | 262,144 tokens (full) |
| Gemma 4 12B | 262,144 tokens (full) |
| Gemma 4 E4B | 131,072 tokens (full) |

Do not set the GPU memory too high: at 20 GB (of 32 GB) the model leaves too little memory for Windows, so the
laptop becomes slow or stops responding.

The output limit (`maxTokens` in Pi) equals the context size: Pi itself clamps it to the room left. Very long
prompts still take time (see section 6), so a large context is mainly useful for long sessions that use the prefix
cache.

Pi handles a full context, not the server. When the conversation passes `context size − 16384` tokens, Pi
summarizes the older part and keeps the latest ~20,000 tokens. If a single request still exceeds the context, the
server sends the same error as llama.cpp ("the request exceeds the available context size"). Pi recognizes it,
then summarizes and retries automatically. You can also summarize at any time with the `/compact` command in Pi.

This summary takes time (a few minutes on the 27B), because the model has to read the part being summarized and
then process the new conversation again. The server cannot simply drop old tokens: the DeltaNet layers of
Qwen3.5/3.8 keep a state that cannot "forget" individual tokens.

## 6. How fast

Measured on this laptop with arcinferzero, 15–16 October 2026 (details and the speed history:
`docs/speed-progress.md`; comparison with llama.cpp: `docs/review.md`). Prompt speed = tokens per second
while reading a 16K-token prompt; writing speed = a 400-token answer through the server (code / Indonesian text;
MTP does not change the content of the answer).

| Model | Reading the prompt (16K) | Writing without MTP | Writing with MTP |
|---|---|---|---|
| Qwen3.6 35B-A3B | ~680 tokens/s | ~20 tokens/s | ~31 / 25 tokens/s |
| Qwen3.5 4B | ~700 tokens/s | ~20 tokens/s | ~30 / 26 tokens/s |
| Gemma 4 E4B | ~625 tokens/s | ~22 tokens/s | ~40 tokens/s (code) |
| Qwen3.5 9B | ~390 tokens/s | ~12 tokens/s | ~17 / 15 tokens/s |
| Gemma 4 12B | ~228 tokens/s | ~10 tokens/s | ~21 tokens/s (code) |
| Qwen3.8 27B | ~103 tokens/s | ~5 tokens/s | ~9 / 7.4 tokens/s |

Very long prompts are read more slowly (attention gets heavier): at 131K tokens the 35B ~250 tokens/s, the 4B ~280,
E4B ~200.

Tips:

- The longest part is usually the thinking phase. `--think off` or `--think minimal` speeds up simple answers.
- Follow-up requests in the same session only process the new tokens (prefix cache), so they are much faster than
  the first request.
- A new session in a project used before is fast: Pi's system prompt comes from the cache (memory or disk), so only
  your message is processed. The first session in a new project folder processes its system prompt once (27B:
  ~40 s).
- Going back to an old session after another session (or the web UI) loads its conversation from disk in seconds:
  the session cache is on by default (8 GB in `%LOCALAPPDATA%\arcinfer\zero\sessions`, delete with
  `arc-server clear-cache`, turn off with `pi-arc --session-cache 0`). `arc-server stop` / `restart` and a model
  switch first save the conversation the server holds, so afterwards that conversation is loaded too, not processed
  again.
- A new message in the same session (Qwen 4B / 9B / 35B) only processes the last turn again: the Qwen template drops
  the thinking of earlier turns as soon as you write again, and the server keeps the starting point of that turn
  (arcinferzero).
- Stop the server with `arc-server stop` (or the Stop button in the launcher), not by killing its process: only
  that way is the conversation being held saved first.
- A hot laptop slows down by 5–10 %. Do not run other GPU work at the same time.

## 7. Moving the arcinfer folder

1. `arc-server stop`
2. Move the `arcinfer` folder to its new location (File Explorer or `mv`).
3. Run setup again from the new location: `<new-folder>\setup.cmd -Models none`

Only the `pi-arc` / `arc-server` shortcut commands store the folder location; everything else uses relative paths.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pi-arc` is not recognized | Run `setup.cmd` (section 1), then open a new terminal. |
| Setup fails at the build step | Visual Studio Build Tools (C++) is not installed; setup shows how to install it. The compiler output is in `%LOCALAPPDATA%\arcinfer\build.log`. |
| "The server stopped while loading" | Look at the log it shows. Usually there is not enough GPU memory: close heavy apps, or use `--ctx 32768` / a smaller model. |
| Windows is slow or stops responding while a model loads | The GPU memory is set too high (section 5): lower it in Intel Graphics Software, then `arc-server restart`. |
| `web_search` in Pi answers "No results found" or never finishes | Its default search engines are DuckDuckGo and Startpage; DuckDuckGo is blocked in Indonesia, so neither returns results. Add Yandex in `%USERPROFILE%\.pi\agent\settings.json`: `"webSearch": {"engines": ["yandex", "startpage", "duckduckgo"]}` (or with the `/search-config` command in Pi). |
| A new Pi session takes ~40 s before answering | Normal for the first session in a new project folder (the system prompt is processed once); later sessions in that folder are fast. |
| "Model file ... not found" | The model has not been downloaded to the Hugging Face cache yet (see the table in section 1). |
| Models moved to another drive | Write the folders (one per line) in `%LOCALAPPDATA%\arcinfer\model-dirs.txt`, e.g. `D:\models` (a Hugging Face cache layout `models--...`, or `.gguf` files directly in that folder). The Hugging Face cache on C: is still searched first. If that drive is removed, its models disappear from the list until the drive is connected again. |
| Pi stops with a context error | Type `/compact` in Pi, or start a new session. |
| The answer is very slow at the start | Normal for the first large prompt (see section 6); later ones use the cache. |
| Want to see what is happening | `arc-server status`, and the log in `%LOCALAPPDATA%\arcinfer\server.log` (prompt processing time, speed, MTP statistics per request). |

## 9. Rebuilding from source

Needs Visual Studio 2026 Build Tools (MSVC). From the arcinfer folder, `.\setup.cmd -Models none` rebuilds only what
is needed (and stops the server first when `server.exe` is rebuilt too). Or by hand:

```powershell
arc-server stop
.\build.ps1 zero\serverz.cpp zero\runz.cpp     # arcinferzero (default)
.\build.ps1 tools\server.cpp tools\run.cpp     # OpenCL engine
```

GPU kernels (`zero\kernels\*.cl` for arcinferzero, `kernels\*.cl` for the OpenCL engine) are compiled by the driver
when the server starts, so kernel changes do not need a build.
Compiled kernels are saved in `%LOCALAPPDATA%\arcinfer\kernels`, so only the first use of each variant waits for
compilation.
Regression tests: `.\tests\run_tests.ps1 -Model4B <path to 4B gguf> -Model27B <path to 27B gguf> -ModelGemma <path to Gemma 12B> -Model35B <path to 35B gguf>`.
