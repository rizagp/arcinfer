# arcinfer

A from-scratch LLM inference engine for **Intel integrated GPUs (Xe-LPG / Meteor Lake "Arc Graphics")** that runs on
**Level Zero** (Intel's low-level GPU API; the original OpenCL engine is kept as an option), written in C++, focused on
**fast prompt processing (time to first token)** for long prompts such as scanning a codebase. It loads GGUF files directly and serves an **OpenAI-, Anthropic- and Ollama-compatible
API with a web UI**, and it plugs into the [Pi](https://pi.dev) coding agent with one command.

Two engines share the kernels and the server code:

- **arcinferzero** (`zero/`, `bin\serverz.exe`, `bin\runz.exe`; the default since 2026-10-16) runs the
  OpenCL-compiled kernels through **Level Zero**. Its output is bit-identical to the OpenCL engine on every model.
  On top it streams mixture-of-experts weights from the SSD (Qwen3.6 35B-A3B: only the experts in use stay in GPU
  memory; the full 262K context fits together with images and MTP, though the automatic context keeps most
  experts cached first, ~40K tokens, for steady generation) and supports up to 7 MTP drafts per step.
- **arcinfer** (`src/`, `bin\server.exe`, `bin\run.exe`): the original OpenCL engine, still available with
  `arc-server start -OpenCL` / `pi-arc --opencl` (launcher: Engine "arcinfer").

Target hardware it was built and tuned on: Intel Core Ultra 5 125H (Arc iGPU, 7 Xe-cores / 112 EUs, no XMX),
32 GB LPDDR5x-6400, Windows 11.

- **What made LLM inference fast on this iGPU, and what did not:** [docs/lessons-intel-arc-igpu.md](docs/lessons-intel-arc-igpu.md)
- Results and the story of the build, with the speed from the first version to now: [docs/review.md](docs/review.md)
- Speed progress from the first version to now (tracked by date): [docs/speed-progress.md](docs/speed-progress.md)
- arcinferzero development log: [docs/arcinferzero-log.md](docs/arcinferzero-log.md)
- What changed recently: [CHANGELOG.md](CHANGELOG.md)
- Usage guide (step by step): [USAGE.md](USAGE.md)

## Supported models

| GGUF architecture | Models | Notes |
|---|---|---|
| `qwen35` | Qwen3.5 (0.8B–27B dense), Qwen3.6, Qwen3.8 27B | hybrid Gated DeltaNet + gated attention; MTP speculative decoding; image input with a Qwen3-VL `mmproj` next to the model (encoded on the GPU; Qwen3.5 4B and 9B) |
| `qwen35moe` | Qwen3.6-35B-A3B: standard GGUFs (e.g. Unsloth UD-IQ4_XS / UD-Q3_K_M) or ISTA-DASLab GSQ 2-bit | mixture of experts (256 experts, 8 active); standard GGUF experts (IQ / Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0) load as they are; GSQ converted from the safetensors checkpoint with `tools/convert_gsq.py`; Unsloth's MTP repository adds speculative decoding and its `mmproj` image input (arcinferzero: both together; the automatic context reserves 128K tokens of context, then sizes the expert cache: 131K tokens with ~80 % of the UD-IQ4_XS experts cached; up to 262K with `--ctx`, with a smaller cache; context sweep in docs/speed-progress.md) |
| `gemma4` | Gemma 4 12B, Gemma 4 E4B (Google's QAT Q4_0 GGUFs) | sliding-window + global attention, per-layer embeddings (E4B); image and audio input (`mmproj`: 12B gemma4uv / gemma4ua, E4B gemma4v ViT and gemma4a conformer on the GPU); speculative decoding with the Gemma 4 assistant drafters (`mtp-*.gguf`) |
| `qwen4exp` | Qwen3.8 Flash Next (ISTA-DASLab GSQ-RCO IQ3_S, GGUF in 2 parts) | **arcinferzero only**; 4 hyper-connection streams, Gated DeltaNet + gated attention with QSA sparse attention, 512 experts (10 active) streamed from the SSD, PLE n-gram table read from part 2 per token; no MTP block, no image input; long-context QSA not validated yet |

Weight types: `Q4_0`, `Q8_0`, `Q2_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
`IQ4_XS` (mixed per tensor, e.g. Unsloth dynamic or ISTA-DASLab GSQ-RCO quants), plus `F32/F16/BF16` small matrices;
GSQ 2-bit experts (`tools/convert_gsq.py`); `Q4_0` drafter matrices (requantized to Q8_0 at load).

## Installation

Everything below runs in **Windows PowerShell** from the project folder (the folder that contains `build.ps1`).
If PowerShell refuses to run the scripts ("running scripts is disabled on this system"), start them as
`powershell -ExecutionPolicy Bypass -File <script> <arguments>`, or allow local scripts once for your user with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Quick setup

Once the graphics driver, Visual Studio Build Tools (C++), Python and optionally Pi are installed (steps 1–3 and
9.1 below), one command does the rest:

```powershell
.\setup.cmd                    # checks, builds, asks which models to download, connects Pi
.\setup.cmd -Models 27b,4b     # download these without asking (27b 35b 9b 4b g12 e4b, all, none)
.\setup.cmd -Tests             # also build the test tools and run the regression tests (needs the 4B)
```

It checks the requirements (and says where to get what is missing), builds `bin\server.exe`, `run.exe` and
`clinfo.exe` when they are missing or older than the sources, checks the GPU (OpenCL 3.0, DP4a, sub-groups, GPU
memory), downloads the chosen models into the Hugging Face cache (Gemma with its drafter and image projector; the
35B is downloaded and converted), installs `huggingface_hub` / `numpy` with pip when needed (it asks first), and
connects Pi (`pi-arc`, `arc-server`, Pi's model list with each model's context). Running it again only does what
is missing. Steps 1–10 describe the same by hand.

### 1. Requirements

| | Needed | Tested with |
|---|---|---|
| OS | Windows 11, 64-bit | Windows 11 Home 25H2 (build 26200) |
| GPU | Intel Core Ultra (Meteor Lake) with Intel Arc integrated graphics (Xe-LPG). The engine needs OpenCL 3.0 with `cl_intel_subgroups` and `cl_khr_integer_dot_product` (DP4a). Other GPUs are not supported. | Core Ultra 5 125H, Arc iGPU 112 EU |
| Memory | 16 GB works for the 4B / E4B models; 32 GB for the 12B–35B models and long contexts. The iGPU can use part of the RAM (`bin\clinfo.exe`, "global mem": 16.5 GiB on 32 GB by default, set in Intel Graphics Software; 18.3 GiB on the test laptop now), shared with the desktop and other apps (~0.7 GiB). Each model's context follows that setting. | 32 GB LPDDR5x-6400 |
| CPU | AVX2 (every Core Ultra has it; the build uses `/arch:AVX2`) | |
| Graphics driver | Intel Arc & Iris Xe graphics driver (it contains the OpenCL runtime; nothing else to install) | 32.0.101.9033 |
| Compiler | Visual Studio 2026 Build Tools with the **Desktop development with C++** workload (MSVC + Windows SDK) | VS Build Tools 18 |
| Python | Python 3.10 or newer with `huggingface_hub` (model downloads) and `numpy` (only for converting the 35B model) | Python 3.14, huggingface_hub 2.1 |
| Pi (optional) | the Pi coding agent, which runs on Node.js | Node.js 24 |
| Disk | 3–13 GB per model | |

### 2. Graphics driver

Install or update the Intel graphics driver from Intel's download center ("Intel Arc & Iris Xe Graphics - Windows")
or with the Intel Driver & Support Assistant, then restart. The OpenCL runtime comes with the driver; there is no
SDK to install: arcinfer loads `OpenCL.dll` at run time and the driver compiles the kernels.

### 3. Compiler

Install **Visual Studio 2026 Build Tools** (free, from visualstudio.microsoft.com, "Tools for Visual Studio" →
"Build Tools for Visual Studio 2026") and select the workload **Desktop development with C++** (it includes MSVC
and the Windows SDK). `build.ps1` finds the compiler itself (`vswhere`: any Visual Studio or Build Tools
installation with `VC\Auxiliary\Build\vcvars64.bat`, Community and Professional included); to use a particular
one, set `$env:ARC_VCVARS` to its `vcvars64.bat`.

### 4. Get the code

Copy or clone the `arcinfer` folder anywhere (the scripts find everything relative to their own location). It
already contains everything it needs to build (`third_party\` holds minja and nlohmann/json).

### 5. Build

```powershell
.\build.ps1 zero\serverz.cpp zero\runz.cpp tools\server.cpp tools\run.cpp tools\clinfo.cpp
```

This takes a few minutes and writes `bin\serverz.exe` / `bin\runz.exe` (arcinferzero: the HTTP server and the
command-line tool for runs, benchmarks and accuracy checks), `bin\server.exe` / `bin\run.exe` (the same on the
OpenCL engine) and `bin\clinfo.exe`. `zero\serverz.cpp` compiles `tools\server.cpp` on the Level Zero engine, so
both servers have the same API. Run the programs from the project folder, because they read `zero\kernels\*.cl`
(arcinferzero), `kernels\*.cl` (OpenCL engine) and `webui\` from there.

Check that the GPU is usable:

```powershell
.\bin\clinfo.exe
```

It should list `Intel(R) Arc(TM) Graphics`, `OpenCL 3.0 NEO`, `cl_khr_integer_dot_product yes`,
`cl_intel_subgroups yes` and sub-group sizes `8 16 32`.

Optional tools (tests and benchmarks):

```powershell
.\build.ps1 tools\test_gemm.cpp tools\tok_test.cpp tools\chat_render.cpp tools\ref_forward.cpp tools\gguf_dump.cpp tools\tool_stream_test.cpp
.\build.ps1 bench\gemv_bench.cpp bench\attn_dec_bench.cpp bench\attn_bench.cpp bench\bw_bench.cpp
```

### 6. Download models

Install the Hugging Face command-line tool once:

```powershell
python -m pip install -U huggingface_hub
```

Then download the models you want. `hf download` puts them in the Hugging Face cache
(`%USERPROFILE%\.cache\huggingface\hub`), which is where `arc-server` / `pi-arc` look for them, so no paths have to
be configured. For the Gemma models, the drafter (`mtp-*.gguf`) and the image projector (`mmproj-*.gguf`) must sit
in the same folder as the model: download them from the same repository in the same command.

| Alias | Model | Command | Size |
|---|---|---|---|
| `27b` | Qwen3.8 27B GSQ-RCO (with MTP head) | `hf download ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | 12.1 GB |
| `9b` | Qwen3.5 9B (with MTP head) | `hf download unsloth/Qwen3.5-9B-MTP-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf` | 6.1 GB |
| `4b` | Qwen3.5 4B (with MTP head) | `hf download unsloth/Qwen3.5-4B-MTP-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf` | 3.0 GB |
| `g12` | Gemma 4 12B QAT (Google, Q4_0) + image projector; drafter from Unsloth, copied into the model's snapshot folder | `hf download google/gemma-4-12B-it-qat-q4_0-gguf` and `hf download unsloth/gemma-4-12B-it-qat-GGUF mtp-gemma-4-12B-it.gguf` | 7.0 + 0.2 + 0.3 GB |
| `e4b` | Gemma 4 E4B QAT (Google, Q4_0); drafter from Unsloth, copied into the model's snapshot folder | `hf download google/gemma-4-E4B-it-qat-q4_0-gguf gemma-4-E4B_q4_0-it.gguf` and `hf download unsloth/gemma-4-E4B-it-qat-GGUF mtp-gemma-4-E4B-it.gguf` | 5.2 + 0.1 GB |
| `35b` | Qwen3.6 35B-A3B, 2-bit GSQ | see step 7 (download 12.6 GB, then convert) | 11.7 GB |
| | Qwen3.8 Flash Next (arcinferzero only, no short alias) | `hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include "IQ3_S/*"` | 54.8 + 28.8 GB |
| | DFlash2 drafters (optional, see `--dflash` below) | `hf download z-lab/Qwen3.8-27B-DFlash2` / `hf download incoai/Qwen3.6-35B-A3B-DFlash2` | 3.85 / 1.05 GB |

With their `mmproj` file next to the model, Gemma 4 12B and E4B read images and audio, and Qwen3.5 4B / 9B and
Qwen3.6 35B read images (all encoders run on the GPU).

### 7. Qwen3.6-35B-A3B (optional): convert the GSQ checkpoint

ISTA-DASLab publishes this model only as a safetensors checkpoint (compressed-tensors: 2-bit routed experts
`{-2,-1,0,1} x bf16 scale per 128`, int8 attention / DeltaNet / shared experts / lm_head). Convert it once
(about 2.5 minutes; it needs `numpy`, and the tokenizer metadata comes from any GGUF with the same vocabulary, e.g.
the 4B from step 6):

```powershell
python -m pip install -U numpy
hf download ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ
$snap = (Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--ISTA-DASLab--Qwen3.6-35B-A3B-2Bit-GSQ\snapshots" | Select-Object -First 1).FullName
$vocab = (Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -Filter "Qwen3.5-4B-UD-Q4_K_XL.gguf" | Select-Object -First 1).FullName
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\arcinfer\models" | Out-Null
python tools\convert_gsq.py $snap "$env:LOCALAPPDATA\arcinfer\models\Qwen3.6-35B-A3B-GSQ2-arc.gguf" --vocab-from $vocab
```

The int8 matrices become Q8_0 with the same int8 values (f16 scales), the experts keep their exact 2-bit codes in
the layout of `kernels/moe.cl`, and the tensor transforms follow llama.cpp's converter (DeltaNet V-head order,
norm weights + 1, A = -exp(A_log)). The checkpoint's MTP block has no routed experts, so this model runs without
speculative decoding. `arc-server` / `pi-arc` find the converted file in `%LOCALAPPDATA%\arcinfer\models`.

### 8. First run

```powershell
.\scripts\arc-server.ps1 start -Model e4b      # or 27b, 35b, 9b, 4b, g12
```

This starts arcinferzero; add `-OpenCL` for the OpenCL engine. The first start of each model compiles its GPU
kernels (up to a minute or two); they are cached in `%LOCALAPPDATA%\arcinfer\kernels`, so later starts take 5–40 s
(mostly reading the model file). The first arcinferzero start of a mixture-of-experts model also writes its expert
file once (35B: 15 GB, about 2.5 minutes; Qwen3.8 Flash Next: 56 GB, about 9.5 minutes, later loads ~13–25 s) to
`%LOCALAPPDATA%\arcinfer\zero\`; it can be deleted at any time and is written again when missing. Then open the web
UI at http://127.0.0.1:8090/ or call the OpenAI API at `http://127.0.0.1:8090/v1`. `arc-server status` shows what
is running and `arc-server stop` frees the GPU memory.

**Launcher window.** `arc-launcher.cmd` (double-click it in the project folder) opens a window (English):

- **Model list:** every model and variant in the Hugging Face cache with MTP, images, auto context, GPU memory at
  that context, file size, its id for Pi's `/model` and whether arcinfer supports it; the running one is green.
- **Options:** filled in with the best values for the selected model (what the server picks itself), so selecting a
  model and pressing Start is enough. Shown: context, MTP drafts, thinking budget (for clients that send none; Pi
  sends its own), prompt cache and network access; "Show advanced options" adds port, prompt chunk, image tokens,
  resident W8 weights, the cache folders and sizes, `--no-swap` and f32 attention. A field you change turns yellow
  and only those are passed on. "Set as default" keeps them (`%LOCALAPPDATA%\arcinfer\launcher.json`) and makes the
  selected model the default one (also pi-arc's); "Reset options" goes back to the best values.
- **Selected model:** what it supports and how its auto context was counted: GPU memory, minus what is kept for
  Windows, state snapshots, encoders, weights, MTP block and working buffers, divided by the context's memory per
  1,000 tokens, at most the model's max context length.
- **Running server:** model, filled context, GPU memory by part (weights, context, MTP, buffers, encoders),
  requests, the last prompt / answer speed and MTP acceptance.
- **Start / Stop server**, refresh, web UI, log. Start and Stop go through `arc-server`, so the server keeps running
  after the window is closed.

The same numbers come from `server.exe --list-models` ("memory", "defaults") and the server's `/props` ("memory",
"stats").

### 9. Pi (optional)

1. Install Pi by following the instructions on https://pi.dev and run it once, so that `%USERPROFILE%\.pi\agent`
   exists (with `bin` on the PATH).
2. Add arcinfer to Pi (one time, `setup.cmd` does it too; it builds `bin\server.exe` if it is missing, adds the
   `arcinfer` provider with every model it finds and its context for the current GPU memory to
   `~/.pi/agent/models.json` (the previous file is kept as `models.json.bak`; `pi-arc` refreshes the list), and
   installs the `pi-arc` and `arc-server` commands):

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\install-pi.ps1
   ```

3. Open a new terminal and start Pi from any project folder:

   ```powershell
   pi-arc                  # the model the server runs or used last (the first time: choose from a list)
   pi-arc --pick           # choose the model from the list of every downloaded model
   pi-arc 27b              # Qwen3.8 27B;  35b = Qwen3.6 35B-A3B (mixture of experts, much faster);
                           # 35b-mtp = the file of Unsloth's MTP repository (~13 % faster generation, 131K context)
   pi-arc g12              # Gemma 4 12B (also reads images);  e4b = Gemma 4 E4B;  9b / 4b = Qwen3.5 9B / 4B
   pi-arc -c               # continue the last session in this folder;  pi-arc -r = choose a session
   pi-arc --think off      # no reasoning; also minimal / low (default) / medium / high
   pi-arc --help           # every option
   arc-server status       # what is running;  arc-server stop = free the GPU memory
   arc-server clear-cache  # delete saved conversations (session cache: on by default, 8 GB; pi-arc --session-cache 0 = off)
   ```

   Inside Pi, `/model` switches between all of them (the server loads the chosen one).

### 10. Check the installation (optional)

With the test tools from step 5 built, the regression tests compare every kernel with a CPU reference, the
tokenizer and chat templates with the reference output, the model's next-token distributions with an f32 CPU
implementation, and the server API:

```powershell
powershell -ExecutionPolicy Bypass -File tests\run_tests.ps1               # arcinferzero, models found automatically
powershell -ExecutionPolicy Bypass -File tests\run_tests.ps1 -SkipGpu      # only the checks without a GPU / model
powershell -ExecutionPolicy Bypass -File tests\run_tests.ps1 -ListModels   # which model each check would use
```

It runs the model-free checks first, then per model found (27B, 35B, Flash Next, 4B, Gemma 4 12B) the KL and greedy
checks and MTP / DFlash2 = plain decoding, then a server smoke test; a missing model is a SKIP, not a failure
(`-Engine opencl` or `both` for the OpenCL engine). It ends with `ALL TESTS PASSED`. Stop `arc-server` first: the
test uses port 8091 and needs the GPU memory.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot load OpenCL.dll` / `no OpenCL GPU found` | Install or update the Intel graphics driver (step 2) and restart. |
| `build failed` right away / `No Visual Studio C++ compiler found` | Visual Studio's C++ workload is missing (step 3); to use a particular installation, set `$env:ARC_VCVARS` to its `vcvars64.bat`. `setup.cmd` checks this. The compiler output of `setup.cmd` is in `%LOCALAPPDATA%\arcinfer\build.log`. |
| `running scripts is disabled on this system` | Run the script through `powershell -ExecutionPolicy Bypass -File ...` (see the top of this section). |
| `Model file ... not found` | Download it with `hf download` (step 6); `arc-server` searches `%LOCALAPPDATA%\arcinfer\models` and the Hugging Face cache. |
| Models moved to another drive | List the folders, one per line, in `%LOCALAPPDATA%\arcinfer\model-dirs.txt` (or `ARC_MODEL_DIRS`, `;`-separated): a Hugging Face cache layout (`models--...`) or `.gguf` files directly in the folder. The Hugging Face cache is searched first, so a model kept in both places loads from there. |
| The first start of a model takes minutes | The driver compiles the kernels once; later starts use `%LOCALAPPDATA%\arcinfer\kernels`. |
| The server stops while loading, or another model fails to start | Another arcinfer server or a llama.cpp process holds the GPU memory: `arc-server stop`, close other GPU programs. The log is `%LOCALAPPDATA%\arcinfer\server.log`. |
| Gemma does not use its drafter | `mtp-gemma-4-*-it.gguf` must be in the same folder as the model GGUF (download it from the same repository). |
| Generation gets slower after a few minutes | The laptop throttles under sustained load (5–10 %); plug in the charger and use a performance power mode. |
| Windows gets slow or stops responding with a model loaded | The GPU memory setting (Intel Graphics Software) is so high that the model leaves too little RAM for Windows (seen at 20 GB on a 32 GB laptop). Lower it; the contexts follow the setting (`arc-server restart`). |
| Pi's `web_search` answers "No results found" or never returns | Its default engines are DuckDuckGo and Startpage; where DuckDuckGo is blocked (Indonesia) neither returns results. Add Yandex in `%USERPROFILE%\.pi\agent\settings.json`: `"webSearch": {"engines": ["yandex", "startpage", "duckduckgo"]}` (or Pi's `/search-config`). |
| A new Pi session takes ~40 s before answering | The first session in a project folder processes Pi's system prompt once; later sessions there reuse it. `%LOCALAPPDATA%\arcinfer\server.log` shows it (`system-prompt cache: saved / loaded ... for <folder>`). |

## Run the server by hand

```powershell
.\bin\serverz.exe -m C:\path\to\model.gguf --port 8080 --ctx auto --mtp --alias qwen3.8-27b
.\bin\serverz.exe --help           # every option with its default
.\bin\serverz.exe --list-models    # the models it can load (Hugging Face cache, %LOCALAPPDATA%\arcinfer\models), as JSON
```

`bin\server.exe` (OpenCL engine) takes the same options except the arcinferzero ones (`--mtp-drafts` above 3,
`--expert-cache-gb`, `--expert-cache-pct`, `--windows-reserve-gb`, `--no-stream`, `--dflash`) and does not load
Qwen3.8 Flash Next.

- Web UI: http://127.0.0.1:8080/
- OpenAI API: `POST /v1/chat/completions` (streaming, `reasoning_content`, tool calls streamed while they are
  generated, string arguments and array / object arguments alike (Pi's `edit` puts its whole change in an array),
  `usage`),
  `GET /v1/models` (every model of `--list-models`, the loaded one first), plus `GET /health`, `GET /props`,
  `POST /tokenize`, `GET` / `DELETE /sessions` (session cache size / clear).
- Sampling and output (chat): `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
  `repeat_penalty` / `repeat_last_n`, `logit_bias`, `seed`, `stop`, `max_tokens`; `logprobs` / `top_logprobs`;
  `n` > 1 (without streaming); `response_format` `json_object` / `json_schema` (sampling is constrained, so the answer
  always parses and fits the schema: types, properties / required / additionalProperties, items, enum, const,
  anyOf / oneOf, `$ref`); `tool_choice` `none` / `auto` / `required` / a named function (the answer starts with a
  call).
- Other API formats, on the same models and features (`tools/api_compat.h`):
  - OpenAI `POST /v1/completions` (a raw prompt, no chat template: code completion) and `POST /v1/responses`
    (Responses API: input items, function calls and their outputs, reasoning, streamed events).
  - Anthropic `POST /v1/messages` (system, image and tool_use / tool_result blocks, thinking, streamed events) and
    `POST /v1/messages/count_tokens`.
  - Ollama `POST /api/chat`, `POST /api/generate` (NDJSON streams, `images`, `tools`, `think`, `format`, `options`),
    `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `GET /api/version`: apps with an Ollama setting connect to
    `http://127.0.0.1:8090`.
  - OpenAI audio: `POST /v1/audio/transcriptions` and `/v1/audio/translations` (multipart upload; a model with an
    audio encoder, Gemma 4 12B / E4B).
- Model switching: a request whose `model` is another model of `--list-models` (its id, `org/repo` name or file)
  makes the server unload the current model and load that one (context auto, same options); `--no-swap` keeps the
  loaded model.
- Thinking control: `chat_template_kwargs.enable_thinking`, `enable_thinking`, or `reasoning_effort`
  (`none`/`low`/`medium`/`high`); a reasoning budget (`thinking_token_budget`, `thinking_budget_tokens` or
  `reasoning_budget`, default `--think-budget`) closes the `<think>` block after that many tokens.
- Prefix reuse: consecutive requests that share a prefix (agent loops) only process the new tokens.
- Cancelling: when the client closes the connection (Pi: Esc), the server stops within a layer during prompt
  processing (27B: ~1 s) and before the next token during generation, so the next request starts right away. The
  prompt blocks that were complete stay processed. A reply the client drops after an abort or error costs nothing
  on the next request: the resume point sits at the start of the reply's turn header.
- System-prompt cache: the state after a client's system prompt (Pi: ~4K tokens; it contains the project folder
  and its `AGENTS.md`, so each project has its own) is saved to `%LOCALAPPDATA%\arcinfer\prefix\<model>` the first
  time it is processed, written in the background, and loaded when a request starts with it while the server holds
  something else: after a server start, a model switch or another project's conversation. Pi's are kept per project
  folder (read from the `<cwd>` section of its system prompt): each folder keeps its 2 latest versions, which other
  folders do not replace (27B: ~290 MB each); other clients' prompts: the 8 used last. All models together
  stay under `--prefix-cache-gb` (default 8; least recently used go first); `--no-prefix-cache` turns it off.
  Files: `<folder>-<folder hash>-<prompt hash>-<tokens>.arcp`.
- Session cache: when a request belongs to another conversation (a new Pi session, or switching back to an older
  one), the server saves the conversation it holds to disk (KV cache, DeltaNet states; `%LOCALAPPDATA%\arcinfer\sessions`)
  and loads a saved one that matches the request, instead of processing its whole prompt again. New conversations
  resume after the system prompt they share (Qwen: a recurrent-state snapshot there, kept in host memory). Saving
  and loading take about a second; outputs are identical to never switching. `--session-cache-gb N` (the server's default 0 = off; arc-server / pi-arc start it with 8 GB, `pi-arc --session-cache 0` = off;
  the least recently used conversations go first); `arc-server clear-cache` deletes them all. `arc-server stop` /
  `restart` and a model switch save the conversation the server holds first (`POST /sessions/save`), so it is
  loaded afterwards instead of processed again; only a server that is killed loses it.
- Qwen turn snapshot (arcinferzero): the Qwen templates drop the reasoning of earlier turns as soon as the user
  writes again, so the server also snapshots the start of the last assistant turn and a new message only processes
  that turn again, not the whole conversation. `--preserve-thinking` (Qwen3.6; Qwen3.8 always does) keeps earlier
  reasoning in the prompt instead: nothing is processed again, the context fills faster.
- `--mtp`: speculative decoding. Qwen3.5/3.6/3.8: the model's own MTP (multi-token prediction) block in the GGUF
  (`nextn` tensors). Gemma 4: the assistant drafter `mtp-*.gguf` next to the model (found automatically), which
  reads the main model's KV cache. The drafts are verified in one main-model pass and the next token is still
  sampled from the model's distribution, so the output is unchanged (checked: identical answers at every draft
  count). `--mtp-drafts N`: arcinferzero 1–7 (default 7 for the 27B, 5 for the 35B, 3 for the 4B, 2 for the 9B and
  Gemma), the OpenCL engine up to 2 (Qwen) / 3 (Gemma).
- `--v4` (arcinferzero, Qwen; experimental, off by default): a 4-bit copy of the attention cache's V half that
  generation and draft verification read, next to the int8 cache that prompts read. Prompts keep their speed;
  generation deep in a long context is ~7 % faster (35B after 125K tokens: median step 72.8 -> 67.5 ms); the
  attention cache needs ~25 % more memory (35B at 131K: expert cache 90 -> 87 %); 35B perplexity +0.4 %. Useful
  only for long answers at contexts well above 64K (`docs/speed-progress.md`, 2026-10-18).
- `--dflash DIR` (arcinferzero: `serverz`, `runz`): speculative decoding with a DFlash2 block drafter, `DIR` being
  the folder with its `config.json` and `model.safetensors` (the snapshot folder of `hf download
  z-lab/Qwen3.8-27B-DFlash2` for the 27B, `hf download incoai/Qwen3.6-35B-A3B-DFlash2` for the 35B). It drafts a
  block of 7 tokens per pass from the main model's hidden states. The model's MTP block stays loaded and each step
  picks the drafter: DFlash2 while its drafts are accepted, else MTP. Greedy, answers identical to plain decoding:

  | Model, text | plain | MTP alone | DFlash2 + MTP |
  |---|---|---|---|
  | Qwen3.8 27B, Python code | 4.91 tok/s | 15.5 | 18.1 |
  | Qwen3.8 27B, Indonesian prose | 4.88 | 7.0 | 6.8 |
  | Qwen3.6 35B-A3B, Python code | 21.2 | 39.3 | 49.8 (99 % of drafts accepted) |
  | Qwen3.6 35B-A3B, Indonesian prose | 21.0 | 22.7 | 22.7 |

  The drafters are trained mostly on English and code; on other text the MTP block does most of the work.
- Images (`image_url` content parts, data URLs) and audio (`input_audio`, `/v1/audio/transcriptions`) when the
  model's `mmproj-*.gguf` is next to it (or given with `--mmproj`): images on Gemma 4 12B / E4B, Qwen3.5 4B / 9B and
  Qwen3.6 35B, audio on Gemma 4 12B / E4B; the encoders run on the GPU.
- Constrained tool calls (Gemma 4 and Qwen): while the model writes a tool call, only tokens that keep the call
  valid for the tool's JSON schema are allowed (required arguments such as `path` cannot be left out, only known
  keys); the content stays the model's choice. `ARC_TOOL_CONSTRAINT=0` turns it off; `ARC_LOG_TOOLCALLS=file.jsonl`
  logs the raw text of every tool call.
- During a long prompt, streaming responses carry `: prefill N/M` progress comments (also shown in the web UI).
- Qwen3.8 Flash Next (arcinferzero only): pass the first part (`...-IQ3_S-00001-of-00002.gguf`) with `-m`; part 2
  must sit next to it. First numbers on the 32 GB laptop: generation ~4 tok/s, short prompts slow while the expert
  cache is cold (cache 7.9 GiB = 12 % of the experts); KL vs the CPU reference 0.0014 on a short chat prompt. No
  MTP block, no image input, long-context QSA still to be validated.

## Using Pi

`pi-arc` (step 9, or `setup.cmd`) starts the server in the background if it is not running, then Pi. The server
stays loaded between Pi sessions, so only the first start pays the model load (5–40 s). `pi-arc --help` lists
every option.

- **Model:** the one the server runs, else the one used last, else a numbered list of every model in the Hugging
  Face cache by full name (`pi-arc --pick` always asks; `pi-arc 35b` or `pi-arc -Model swift` chooses directly).
  Every model is in Pi's `/model` list; choosing another one makes the server unload the current model and load
  that one (5–40 s) with its own context, and the next `pi-arc` starts with it. The list is rescanned on every
  `pi-arc` and `arc-server start` / `restart`: a model deleted from the Hugging Face cache disappears from it.
- **Context:** each model gets the largest context that fits the GPU memory currently set (Intel Graphics
  Software), with ~1 GiB left for Windows and other apps, at most its max context length (the GGUF's
  `context_length`). The GPU memory the driver offers changes with what else runs, so the numbers move; the launcher
  window shows how each one is counted. Examples at ~18–19 GiB: Qwen3.6 35B-A3B 40,960 with images, MTP and DFlash2
  (arcinferzero: experts streamed from the SSD, the automatic context leaves them up to 90 % of the cache; 262,144 fits
  with `--ctx 262144`, but generation then waits for SSD reads); Qwen3.8 Flash Next 40,960; Qwen3.5 4B / 9B and Gemma 4
  12B 262,144; Gemma 4 E4B 131,072; Qwen3.8 27B 40,960 with images, MTP and the DFlash2 drafter, more without them. `pi-arc --ctx N` sets a context, `pi-arc --ctx auto` goes back to the automatic one (restarting a server
  that runs a fixed context). `--chunk 1024` frees 1.1 GiB on the 27B for more context (118,784 at 18.3 GiB;
  prompts of ~11K tokens ~4 % slower, from ~34K on no difference). Pi's `maxTokens` is the whole window: Pi clamps
  each request to the room left (window − prompt − 4096).
- Reasoning effort `pi-arc --think off|minimal|low|medium|high` (default low; Pi's `/thinking` changes it). Pi
  sends the effort and a thinking budget per level, so the server's `--think-budget` does not apply to it:

  | Pi level | budget Pi sends | server |
  |---|---|---|
  | off | - | no reasoning |
  | minimal | 1,024 | reasoning off (minimal counts as off) |
  | low | 2,048 | effort low, `<think>` closed after 2,048 tokens |
  | medium | 8,192 | effort medium, closed after 8,192 |
  | high / xhigh | 16,384 | effort xhigh, closed after 16,384 (Pi treats xhigh as high) |

  The effort reaches the chat template only where it uses one (Qwen3.8 27B); the budget applies to every model.
  MTP speculative decoding is on by default (`--no-mtp` to disable).

### Sessions, caching and Esc

What the server has to process (times on the 27B; see "Run the server by hand" for how the caches work):

| You | The server processes |
|---|---|
| keep working in a session (agent loop) | only what is new: 5–7 s before the model starts writing, even 27K tokens into a session |
| open a new session (`pi-arc`, `/new`) in a project used before | only your message: the project's ~4K-token system prompt comes from memory or disk (~4–5 s instead of ~40 s), also after a server restart or a model switch |
| open the first session in a new project folder | its system prompt once (~40 s); it is then kept for that folder |
| continue a session (`pi-arc -c`, `pi-arc -r`, `/resume`) right after closing it | only your new message (the server still holds that conversation) |
| ... after another session or project used the server | only the new message: the session cache (on by default) loads it from disk in seconds |
| ... after `arc-server restart` / `stop` or a model switch | only the new message: the held conversation is saved first and loaded from disk |
| press Esc | nothing more: the server stops within ~1 s while reading the prompt, before the next token while writing; your next message only processes what is new |

A new session starts empty: Pi sends only its system prompt and your message, so the model does not see an earlier
session. To continue one, use `pi-arc -c` (the last session in this folder) or `pi-arc -r` (choose).

When a conversation gets long, Pi compacts it (summarizes the older part and keeps the last ~20 K tokens) once it
passes `contextWindow − 16384` tokens, using the token usage the server reports. If a single request still exceeds
the context, the server answers with llama-server's `exceed_context_size_error` ("the request exceeds the available
context size"), which Pi recognizes: it compacts and retries. The server itself does not drop old tokens (no
context shift): the Qwen3.5/3.8 DeltaNet layers keep a recurrent state that cannot forget individual tokens.

Manual configuration, if you prefer: add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "arcinfer": {
      "baseUrl": "http://127.0.0.1:8090/v1",
      "api": "openai-completions",
      "apiKey": "arcinfer",
      "models": [
        { "id": "qwen3.8-27b", "reasoning": true, "input": ["text"], "contextWindow": 86016, "maxTokens": 86016,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "compat": { "thinkingFormat": "qwen", "supportsReasoningEffort": true,
                      "thinkingTokenBudgetField": "thinking_token_budget", "maxTokensField": "max_tokens" } }
      ]
    }
  }
}
```

then `pi --model arcinfer/qwen3.8-27b --thinking low` (Pi sends the reasoning effort and budget; without an
effort the Qwen3.8 template thinks at `xhigh`, which can take very long on a laptop).

## CLI

`bin\runz.exe` (arcinferzero) and `bin\run.exe` (OpenCL engine) take the same arguments; `runz --bench-pp` also
prints the time of each generation step.

```powershell
.\bin\run.exe model.gguf --bench-pp 2048 --n 64                       # prompt-processing and generation speed
.\bin\run.exe model.gguf --tokens-file ids.txt --n 64 --show --prof   # time to first token + kernel profile
.\bin\run.exe model.gguf --text-file prompt.txt --n 128 --prof-gen    # generation speed + per-kernel profile
.\bin\run.exe model.gguf --perplexity corpus.txt --ppl-chunks 40      # perplexity (llama-perplexity method)
.\bin\run.exe model.gguf --tokens-file ids.txt --kl-ref ref.bin       # KL vs an f32 reference (tools\ref_forward)
.\bin\run.exe model.gguf --tokens-file ids.txt --kl-ref ref.bin --kl-decode   # the same, token by token
```

`--mtp` turns on speculative decoding, `--ctx N` sets the context; with `ARC_PROF_LAYERS=8` the profile is split
into groups of 8 layers.

## How it works

- **GEMM on DP4a.** The Arc iGPU has no matrix engines; its fastest arithmetic is signed int8 DP4a
  (~13 TOPS measured). Activations are quantized to int8 with a scale per 32 values; weights are multiplied in
  an int8 "W8" form (int8 values, int8 scale per 16 or 32 values, float scale per 256). SIMD16 sub-groups compute
  16 columns x 16 tokens with constant-index lane broadcasts, work-groups share weight tiles across 8 token tiles
  and are swizzled so weight slices are reused from L3; long reductions (K > 8192) are split so the activation
  slab stays in L3. The fused SwiGLU (silu(gate)·up) is applied in the epilogue of the up-projection GEMM.
- **Just-in-time weight decoding.** Every GGUF quant type is decoded into W8 on the GPU: IQ/K types without
  block minimums losslessly; types with minimums (Q2_K/Q4_K/Q5_K) are requantized to signed int8 with a per-16/32
  scale (adds ~1/10 of their own quantization noise; +0.05% perplexity on Qwen3.5-4B), which lets them use the
  fast min-free kernel (Q4_K: 6.1 -> 8.8 TOPS). Small models keep the decoded form resident; large ones (e.g. 27B)
  decode just in time, and a long prompt is processed **layer by layer** (all chunks of the prompt go through a
  layer before the next one), so each matrix is decoded once per prompt and only the raw GGUF bytes plus one
  layer of decoded weights have to fit in memory.
- **Token generation** reads the raw blocks directly (padded to 8-byte multiples for coalesced sub-group block
  reads, staged through local memory, codebooks in local memory) and decodes them with SWAR bit tricks; the large
  matrices run at 60–75 GB/s of the 77 GB/s the iGPU can read from memory. The output head is allocated before all
  other weights (a buffer allocated late is read at about half speed), Gemma's logit soft-capping runs in the head's
  epilogue, and for single-token passes the residual add, the sandwich norm and the next input's norm and int8
  quantization are one kernel.
- **Gated DeltaNet** recurrence: each SIMD8 lane keeps a full 128-float state column in registers; q/k are
  block-read and broadcast, so no reductions are needed.
- **Mixture of experts** (`kernels/moe.cl`): the router GEMM gives the top-8 experts per token (softmax,
  renormalized) and the sigmoid gate of the shared expert; the (token, expert) pairs are grouped by expert with a
  counting sort, and tiles of 16 pairs of one expert run DP4a GEMMs straight on the 2-bit codes (decoded to int8
  in registers, SwiGLU in the epilogue, routing weight on the down projection). A single token skips the sort.
- **Attention** is int8 end to end: the KV cache holds int8 K and V (stored in 16-key blocks); scores are DP4a,
  the online softmax is f32 (log2 domain, rescaling only when the running max grows), and P·V is DP4a too, with
  P·vscale quantized to 8 bits per 16-key tile (SageAttention-style). Qwen uses one scale per q/K/V row; Gemma 4
  one per 64 dims (its V rows have a few large dims that a row scale lets swamp the rest: measured against an f32
  reference). In prefill, K/V tiles are shared through local memory by all GQA heads of several tokens; for
  Gemma 4 12B's global layers (head size 512, 16 q heads per kv head) the sub-groups also split the head dimensions
  and sum partial scores through local memory (`attn_ps`), which avoids register spills (1.25–1.4x faster). Decoding
  splits the keys over many work-groups (flash-decoding) and merges the partial results; for Gemma 4 each
  work-group also splits the head dimensions over its sub-groups (`attn_dsplit`), so every K/V byte is read once
  for all heads and all rows of a verification pass (head size 512 at 14K context: 0.74 → 0.31 ms per layer).
  RoPE angles are computed once per pass.
- **Prompt cache.** The server holds one conversation (KV cache, DeltaNet states and the tokens they stand for)
  and processes only what a request adds to the prefix it shares. Attention K/V are stored by position, so moving
  back is free, but the DeltaNet states can only be restored where a snapshot was taken: at the end of the system
  prompt (host memory, and on disk per project folder), at the start of the last `<|im_start|>assistant` header
  (where the next turn diverges, also after an aborted reply the client drops) and at the end of the last prompt
  (GPU). A request resumes from the latest one inside its shared prefix. Gemma 4 has no recurrent state and moves
  back to any position its sliding-window rings still cover. A cancelled prompt rolls the block in progress back
  (DeltaNet states from a host copy taken at the block start) and keeps the blocks before it.
- Compiled kernels are cached in `%LOCALAPPDATA%\arcinfer\kernels` (keyed by source, options and driver), so
  only the first use of a kernel variant pays the driver's compile time.
- Prompts longer than 16,384 tokens go through the layer-major prefill in segments of 16,384, which keeps the
  residual-stream buffer small and lets the long contexts above fit.
- Tokenizer (byte-level BPE with the Qwen pre-tokenizer), chat templates (ChatML + `<think>` + XML tool calls for
  Qwen, the GGUF's Jinja template through minja for Gemma 4) and sampling are implemented natively; the tokenizer
  output is identical to llama.cpp's.

## Layout

```
src/        engine.h (forward passes), quants.h (GGUF decoding/repacking), tokenizer.h, chat.h (Qwen),
            chat_gemma4.h + chat_jinja.h (Gemma 4, minja templates), image.h + vision_gemma4.h / vision_qwen3vl.h (image input),
            audio.h + audio_gemma4.h (audio input), json.h, http.h, gguf.h, clrt.h / cl_min.h (OpenCL loader,
            kernel cache)
kernels/    gemm.cl (DP4a GEMM), decode.cl (JIT decode + GEMV), ops.cl (norms, DeltaNet, attention, quantization),
            moe.cl (mixture-of-experts routing and 2-bit expert GEMMs)
zero/       arcinferzero: engine.h (Level Zero engine), clrt.h (OpenCL-style calls on Level Zero), xcache.h +
            expert_store.h (MoE experts on the SSD, GPU cache), serverz.cpp / runz.cpp, kernels/ (its own copy of the
            kernels), vision / audio encoders, grf_probe.cpp (register-spill probe)
tools/      server.cpp, run.cpp, test_gemm.cpp (kernel validation), ref_forward.cpp (f32 CPU reference: Qwen3.5,
            Qwen3.5-MoE, Gemma 4), convert_gsq.py (ISTA GSQ checkpoint -> GGUF), tok_test.cpp, chat_render.cpp,
            gguf_dump.cpp, clinfo.cpp
bench/      micro-benchmarks (gemv_bench, attn_dec_bench, attn_bench, gemm_bench, bw_bench, ...), run_bench.ps1
            (comparison with llama.cpp), prompt_bench.py, review_data.py (review data)
scripts/    arc-server.ps1 (server in the background), arc-launcher.ps1 (launcher window, arc-launcher.cmd),
            pi-arc.ps1 (Pi launcher), install-pi.ps1 (commands and Pi's model list), pi-models.ps1 (writes that list)
tests/      run_tests.ps1 and its reference data
webui/      index.html
docs/       review.md (results), speed-progress.md (speed by date), lessons-intel-arc-igpu.md,
            arcinferzero-log.md, review_notes.md (working notes)
setup.cmd   setup.ps1: checks, builds, downloads models, connects Pi;  build.ps1: builds one or more tools
```

## Validation

- `tools/test_gemm` checks every weight type and kernel path (native, JIT W8, GEMV) against a CPU reference on
  real model tensors; `bench/gemv_bench` and `bench/attn_dec_bench` check the generation kernels the same way.
- `tools/ref_forward` is an independent f32 CPU implementation of the model; `run --kl-ref` reports the KL
  divergence of the engine's next-token distributions against it (averaged over the prompt positions), and
  `--kl-decode` does the same through the token-by-token generation path.
- `tests/run_tests.ps1` runs all of the above plus chat-template rendering against the GGUF Jinja templates and an
  OpenAI API test (streaming tool calls, prefix reuse); with `-ModelGemma` / `-Model35B` also the Gemma 4 and
  35B-A3B paths (KL vs the f32 reference, greedy equal to llama.cpp, drafter output identical).
- Mean KL of the next-token distributions vs the f32 reference: Qwen3.5-4B 0.0021, Qwen3.6-35B-A3B 0.0008,
  Gemma 4 12B 0.029 (Gemma 4 is about 7x more sensitive to int8 rounding: int8 activations alone give ~0.009).
- `tools/tok_test` compares tokenization with reference token ids.
- `tools/tool_stream_test` checks the streamed tool calls: for every prefix of a response the arguments sent so far
  are a prefix of the final ones, and the final ones equal the whole-response parse.
- Speeds measured while another GPU process runs (an arcinfer server, a stray `run.exe`, llama.cpp) are not
  representative: they share the iGPU and its memory. `run.exe` prints a note when it finds one.

## License

arcinfer is licensed under the [Apache License 2.0](LICENSE).

IQ codebook tables and Unicode category tables are generated from ggml / llama.cpp sources (MIT License).
`third_party/` contains minja (MIT) and nlohmann/json (MIT).
