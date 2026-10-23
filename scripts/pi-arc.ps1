# Pi on arcinferzero in one command: starts the arcinferzero server in the background if needed (the running model, else
# the one used last, else a list of every supported GGUF model in the Hugging Face cache by its full name), writes
# every model with its context into Pi's model list (so Pi's /model switches between them), then starts Pi.
# Other arguments go to Pi.
#
#   pi-arc                          the running / last model (the first time: choose from the list)
#   pi-arc --pick                   choose from the list
#   pi-arc 35b  /  pi-arc -Model swift   a shortcut (27b 35b 35b-s 35b-mtp flash 9b 4b g12 e4b g26, with -mtp /
#                                   -vision / -std / -nodf / -df for a variant) or part of a model's name
#   pi-arc -c  /  pi-arc -r         continue the last Pi session in this folder / choose one (Pi's options)
#   pi-arc --think off              no reasoning (fastest answers); also minimal / low (default) / medium / high /
#                                   xhigh / max
#   pi-arc -p "explain src/x"       any Pi arguments after the launcher options
#   pi-arc --ctx auto|N --chunk N   context (default auto: the largest that fits the GPU memory) / chunk size
#   pi-arc --session-cache 0        do not keep conversations on disk when switching (default 8 GB)
#   pi-arc --help / --help-all      the short help / every option
#   arc-server stop                 stop the background server when you are done (frees the GPU memory)
#
# (No param() block on purpose: PowerShell would otherwise try to bind Pi's own options such as -p or --tools.)
$ErrorActionPreference = "Stop"
$shortcuts = @("27b", "35b", "35b-s", "35b-m", "35b-mtp", "flash", "9b", "4b", "g12", "e4b", "g26")
# (a shortcut with a variant suffix: 27b-std, 9b-vision, 35b-mtp-std, g12-mtp, ...)
function Test-Shortcut([string]$s) { $s = $s.ToLower(); return ($shortcuts | Where-Object { $s -match "^$([regex]::Escape($_))(-mtp|-vision|-std)?(-df|-nodf)?$" }).Count -gt 0 }
$model = ""; $pick = $false; $thinking = "low"; $noMtp = $false; $ctx = ""; $chunk = 0; $sessGb = $null; $vision = ""; $noStream = $false; $preserve = $false; $slots = 0
$piArgs = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $args.Count; $i++) {
  $a = [string]$args[$i]
  if ($a -in "--help-all", "-HelpAll") {
    Write-Host @'
pi-arc: the Pi coding agent on the local arcinferzero server (Intel Arc iGPU), every option.

USAGE
  pi-arc [model] [launcher options] [Pi options and message]

  pi-arc                         start Pi right away: on the model the server runs, else the one used last
                                 (the first time: choose from the list)
  pi-arc --pick                  choose the model from the list first
  pi-arc 35b                     start Pi on a known model
  pi-arc -c                      continue the last Pi session in this folder (pi-arc -r: choose a session)
  pi-arc -Model swift --think off -p "summarize README.md"

SETUP
  setup.cmd                      (in the arcinferzero folder) checks the requirements, builds, downloads models and
                                 connects Pi; setup.cmd --help lists its options

CHANGING THE MODEL INSIDE PI
  /model                         every model below is in Pi's list (provider "arcinferzero"); choosing one makes
                                 the server unload the current model and load that one (5-40 s), with that
                                 model's own context. The next pi-arc starts with the model you switched to.

WORKING IN PI
  Esc                            stops the server right away: while it reads the prompt within ~1 s, while it
                                 writes before the next token. Your next message only processes what is new.
  next step in a session         only the new tokens are processed (the server holds the conversation).
  new session (/new, pi-arc)     reuses Pi's system prompt: in memory, and from disk after a server restart or
                                 a model switch (27B: ~5 s to the first token instead of ~40 s).
  another project folder         Pi's system prompt contains the folder and its AGENTS.md, so the first session
                                 in a project processes it once (27B: ~40 s); later ones reuse it. Kept per
                                 folder in %LOCALAPPDATA%\arcinferzero\zero\prefix: its 2 latest versions, which other
                                 projects do not replace; past 8 GB in all the least recently used go first.
  going back to an old session   (pi-arc -c, pi-arc -r, /resume) right after closing it: only the new message.
                                 After other sessions (or the web UI): loaded from disk in seconds (session
                                 cache, on by default; --session-cache 0: the whole conversation again). After a server restart:
                                 the whole conversation again (the one held when the server stops is not saved).

MODEL
  --pick, -Pick                  numbered list of every supported GGUF model in the Hugging Face cache
                                 (%USERPROFILE%\.cache\huggingface\hub) and %LOCALAPPDATA%\arcinferzero\models,
                                 by full name (e.g. ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF); Enter keeps the
                                 model that is already running. Image projectors (mmproj-*) and drafters
                                 (mtp-*) next to a model are loaded with it, not listed.
  27b 35b 35b-s 35b-mtp flash 9b 4b g12 e4b g26
                                 shortcuts, as the first argument: Qwen3.8 27B; Qwen3.6 35B-A3B (35b: UD-IQ4_XS
                                 from Unsloth's MTP repository, or UD-Q3_K_M / the GSQ file when that is the one
                                 downloaded; 35b-s: UD-Q3_K_S, longer context; 35b-mtp: UD-Q3_K_M of the MTP
                                 repository); flash: Qwen3.8 Flash Next; Qwen3.5 9B / 4B; Gemma 4 12B / E4B /
                                 26B-A4B (g26: mixture of experts on the SSD, reads images)
  <shortcut>-mtp / -vision / -std  a variant of the same file: MTP without images, images without MTP, neither
                                 (e.g. 9b-std: the most context, 27b-std, 35b-mtp-std). A shortcut must come
                                 before Pi's options and message; any model's id from --pick works too.
  <shortcut>-nodf / -df          a model with a DFlash2 drafter (27b, 35b): drafts with the MTP block only /
                                 DFlash2 only (default: both, picked per step), e.g. 35b-nodf
  -Model <text>                  the model whose id, name or file contains <text>, or a path to a .gguf file.
                                 Also --model <text> (or arcinferzero/<id>, as in Pi), --arc-model, --model-name.

LAUNCHER OPTIONS
  --think <level>                reasoning effort: off, minimal, low (default), medium, high, xhigh, max
                                 (also --thinking, -Thinking; Pi's /thinking changes it later). Pi sends a
                                 thinking budget per level: minimal = reasoning off on arcinferzero, low 2,048,
                                 medium 8,192, high / xhigh / max 16,384 tokens; the <think> block is closed there.
  --no-stream                    MoE models without the SSD expert cache: every expert on the GPU (a smaller
                                 context). Normally MoE models (35B, Flash Next, Gemma 4 26B) keep their experts,
                                 MTP layer included, on the SSD and the most used ones in a GPU cache, and read the
                                 next layer's missing experts while the GPU computes: 131,072 tokens of context by
                                 default, up to 262,144 with --ctx. The expert file (35B: 16 GB, Flash Next: 56 GB,
                                 Gemma 4 26B: 13 GB) is written once and reused.
  --vision on|off, --no-vision   image / audio input (default on); off: the projector is not loaded and the
                                 automatic context is longer (Qwen3.5 9B: ~143K -> ~217K tokens)
  --preserve-thinking            Qwen3.6 35B: earlier turns keep their reasoning in the prompt (Qwen3.8 27B always
                                 does). The model sees why it took its earlier steps; the context fills sooner.
  --no-mtp, -NoMtp               no speculative decoding (default: on when the model has an MTP block or a
                                 Gemma drafter; same output, faster generation)
  --ctx <N|auto>, -Ctx           context size in tokens, or auto = the largest that fits the GPU memory
                                 currently set (bin\clinfo.exe, "global mem") less 1.2 GiB for the driver,
                                 at most the model's max context length. Computed for every model in
                                 a few seconds (a dry load that only adds up the memory), remembered in
                                 %LOCALAPPDATA%\arcinferzero\zero\ctx-auto.txt, recomputed when the GPU memory changes.
                                 For example (2026-10-18, 21.5 GiB GPU budget): 27b 69,632 (images, MTP and
                                 DFlash2), 27b-df 94,208, 27b-std 188,416; 35b and flash 131,072 (the expert cache
                                 gets the rest). --pick shows every model's value.
                                 Without --ctx: a running server keeps its context, a new one starts with auto.
                                 --ctx auto also restarts a server that runs a fixed context (e.g. one started
                                 with --ctx 32768).
  --chunk <N>, -Chunk            prompt chunk size in tokens (default per model: 27B 4096; 35B and Gemma 4 26B
                                 8192; Flash Next 2048; 512 when the decoded weights stay resident, as on the 4B /
                                 9B / E4B; Gemma with images at least 2048). Smaller chunks need less GPU
                                 memory, which the auto context turns into more context: 27B with
                                 --chunk 1024 -> ~65K tokens at 17.4 GiB, 118,784 at 18.3 GiB; ~11K-token
                                 prompts ~4 % slower, long prompts the same speed.
  --session-cache <GB>           keep conversations on disk (%LOCALAPPDATA%\arcinferzero\zero\sessions) when you switch
  (also -SessionCache)           Pi sessions: going back to one loads it in seconds instead of processing its
                                 whole prompt again (default: 8 GB; 0 = off). New sessions always reuse the
                                 system prompt (in memory; from disk after a restart or a model switch).
  --slots <N>, -Slots            the server holds N conversations at once (default 1): start the first Pi with
                                 --slots 2, a second Pi (another terminal, plain pi-arc) shares the server; each
                                 keeps its own KV cache and they take turns step by step; while both write their
                                 answers, their next tokens run in one pass (27B prose 7.3 -> 10.5 tok/s together,
                                 35B 20.2 -> 24.7). Each slot gets the context that fits N of them (27B: 2 x 49,152
                                 tokens, 35B: 2 x 65,536, g26: 2 x 32,768)
  --help, -h, -?, /?, help       the short help;  --help-all: this text

  A launcher option that differs from what the running server uses restarts the server (loading takes
  5-40 s). Without launcher options the running server is kept as it is.

PI OPTIONS (passed to Pi unchanged; all of them: pi --help)
  -c, --continue                 continue the last Pi session in this folder
  -r, --resume                   choose a session of this folder to resume
  --session <path|id>            open a specific session (file or part of its id)
  --fork <path|id>               copy a session into a new one and continue there
  -n, --name <name>              name the session (shown in -r)
  --no-session                   do not save the session
  -p, --print                    answer the message and exit (no interactive screen)
  "message" / @file              a first message; @file attaches a file (an image when the model has vision)
  -t, --tools <list>             only these tools (e.g. read,grep,find,ls = read-only); --no-tools: none
  -xt, --exclude-tools <list>    every tool except these
  -nc, --no-context-files        do not load AGENTS.md / CLAUDE.md (a shorter system prompt)
  --system-prompt / --append-system-prompt <text>  replace / extend Pi's system prompt
  --export <session> [file.html] write a session as HTML and exit
  Resuming a session the server still holds is instant; otherwise its whole conversation is processed again
  (unless the session cache saved it; on by default).
  pi-arc gives Pi --model arcinferzero/<id> and --thinking itself. Pi's model list (provider "arcinferzero") is written
  to %USERPROFILE%\.pi\agent\models.json: every model the server finds, each with its context for the current
  GPU memory (the running one's as the server reports it); other providers are kept. It is rescanned on every
  pi-arc and arc-server start: a model deleted from the Hugging Face cache leaves Pi's /model.

SERVER COMMANDS (arc-server)
  arc-server start [options]     start the server or switch the model; same model choice as pi-arc
  arc-server restart [options]   restart with other options
  arc-server status              running model, its file, context and chunk size, saved conversations
  arc-server stop                stop the server and free the GPU memory
  arc-server clear-cache         delete all saved conversations (session cache)
  arc-server help                its own options:
    -Model <shortcut|text|path>  -Ctx <N|auto>  -Chunk <N>  -SessionCacheGb <N>  -NoMtp  -Vision on|off  -NoStream
    -Slots <N>
    -Port <N> (default 8090)  -MtpDrafts <1..7>  -ThinkBudget <N>  -ImageMinTokens / -ImageMaxTokens <N>
    -BudgetGb <GB>  -AttnF32  -NoPrefixCache  -PrefixCacheGb <N>  -NoSwap  -ListenHost <addr>  -SessionDir <dir>
    -Quiet
  arc-launcher.cmd (project folder)  a window: every model and variant with its memory and automatic context, a
                                 field for every option, Start / Stop, and the running server's GPU memory,
                                 requests and speeds (the server keeps running when it is closed)

SERVER DIRECTLY (bin\serverz.exe; arc-server passes these for you; details:
bin\serverz.exe --help)
  -m, --model <model.gguf>       model file              --port <N>          port (default 8080)
  -c, --ctx <N|auto>             context                 --chunk <N>         prompt chunk size
  --alias <name>                 model id in the API     --mtp               speculative decoding
  --mtp-drafts <N>               drafts per step (27B 7, 35B 5, 4B 3, else 2; 7 with DFlash2)
  --draft auto|mtp|dflash        with a DFlash2 drafter: both (auto), the MTP block only, DFlash2 only (the
                                 -nodf / -df variants)
  --think-budget <N>             close reasoning after N tokens
                                                                             (arc-server: 8192)
  --mmproj <file> / --no-mmproj  image projector (Gemma 4 12B; found next to the model by default)
  --session-cache-gb <N>         disk session cache      --session-dir <dir> where it is kept
  --no-prefix-cache              do not keep system prompts on disk
  --prefix-cache-gb <N>          disk space for them, all models together (default 8)
  --no-swap                      keep the loaded model (requests for another model are not switched to)
  --host <addr>                  listen address (default 127.0.0.1)
  --attn-f32                     f32 attention (tests)   --budget <GB>       keep decoded weights resident up to GB
  --slots <N>                    hold N conversations at once, taking turns step by step
  --webui <dir>                  web UI folder           --kernels <dir>     kernel folder (zero\kernels)
  --list-models                  the models arc-server's list shows, as JSON
  --help, -h                     every option with its default
  API: http://127.0.0.1:8090/v1 (OpenAI chat completions), web UI http://127.0.0.1:8090/,
       /health, /props (also GPU memory by part and request stats), /tokenize, /sessions (GET size, DELETE clear)

FILES
  %LOCALAPPDATA%\arcinferzero\server.log      server log: load time, context, per-request prompt / generation speed,
                                          system-prompt cache loads and saves
  %LOCALAPPDATA%\arcinferzero\server.json     what arc-server started (model, context, chunk)
  %LOCALAPPDATA%\arcinferzero\last-model.txt  the model started last (pi-arc starts it again by default)
  %LOCALAPPDATA%\arcinferzero\kernels         compiled GPU kernels (cache)
  %LOCALAPPDATA%\arcinferzero\zero\          arcinferzero's data:
    ctx-auto.txt                          computed contexts per model and GPU memory
    sessions                              saved conversations (--session-cache)
    prefix                                saved system prompts, per project folder (Pi's differs per
                                          project: it contains the folder and its AGENTS.md)
    *.experts / *.lru                     MoE expert files (written once per model) and their cache state
  %LOCALAPPDATA%\arcinferzero\build.log       compiler output of the last setup.cmd build
'@
    exit 0
  }
  if ($a -in "--help", "-h", "-?", "/?", "help") {
    Write-Host @'
pi-arc: the Pi coding agent on the local arcinferzero server (Intel Arc iGPU).

  pi-arc                         start Pi on the model the server runs, else the one used last
  pi-arc -c                      continue the last Pi session in this folder (pi-arc -r: choose a session)
  pi-arc 35b                     start on a model: 35b, 27b, flash (Qwen3.8 Flash Next), 9b, 4b, g12, e4b, g26
  pi-arc 35b-nodf / 35b-df       drafting with the MTP block only (prose, Indonesian) / DFlash2 only (code)
  pi-arc --pick                  choose from the list of every model on disk
  pi-arc --think off             no reasoning (fastest answers); minimal / low (default) / medium / high
  pi-arc -p "summarize x.md"     answer one message and exit (any other Pi option works too)
  arc-server stop                stop the background server (frees the GPU memory)

In Pi: /model switches the model (5-40 s), Esc stops the answer, /new starts a new session, /compact
shortens a long conversation (the details it drops are gone for the model).

  pi-arc --help-all              every option: models and variants, context, chunk size, caches, server
'@
    exit 0
  }
  elseif ((Test-Shortcut $a) -and $piArgs.Count -eq 0) { $model = $(if ($a.ToLower() -eq "35b-m") { "35b" } else { $a.ToLower() }) }
  elseif ($a -in "--pick", "-Pick") { $pick = $true }
  elseif ($a -in "--session-cache", "-SessionCache") { $sessGb = [double]$args[++$i] }
  elseif ($a -in "--think", "--thinking", "-Thinking") { $thinking = [string]$args[++$i] }
  elseif ($a -in "--arc-model", "-Model", "--model-name", "--model") {  # (--model: Pi's own option, taken here so
    $model = ([string]$args[++$i]) -replace '^arcinfer(zero)?/', ''   #  that Pi gets only the one pi-arc sets)
  }
  elseif ($a -in "--ctx", "-Ctx") {
    $ctx = ([string]$args[++$i]).Trim().ToLower()
    if ($ctx -ne "auto" -and $ctx -notmatch '^\d+$') { Write-Host "pi-arc: --ctx takes a number of tokens or auto"; exit 1 }
  }
  elseif ($a -in "--chunk", "-Chunk") { $chunk = [int]$args[++$i] }
  elseif ($a -in "--no-mtp", "-NoMtp") { $noMtp = $true }
  elseif ($a -in "--no-vision", "-NoVision") { $vision = "off" }
  elseif ($a -in "--vision", "-Vision" -and $i + 1 -lt $args.Count) { $i++; $vision = ([string]$args[$i]).ToLower() }
  elseif ($a -in "--preserve-thinking", "-PreserveThinking") { $preserve = $true }
  elseif ($a -in "--no-stream", "-NoStream") { $noStream = $true }
  elseif ($a -in "--slots", "-Slots") { $slots = [int]$args[++$i] }  # (conversations the server holds at once)
  else { $piArgs.Add($a) }
}
# Which model: the one the server is running, else the one started last, else ask (--pick: always ask).
. (Join-Path $PSScriptRoot "app-dir.ps1")
$stateDir = Get-AppDir  # (%LOCALAPPDATA%\arcinferzero; renamed from arcinfer once)
$base = "http://127.0.0.1:8090"
# the id of the loaded model: the entry marked loaded (the server lists every model it can switch to), else the first
function Get-LoadedId([int]$TimeoutSec) {
  $d = @((Invoke-RestMethod "$base/v1/models" -TimeoutSec $TimeoutSec).data)
  $l = @($d | Where-Object { $_.loaded -eq $true })
  if ($l.Count) { return $l[0].id } else { return $d[0].id }
}
if (-not $model -and -not $pick) {
  $running = try { Get-LoadedId 2 } catch { $null }
  $last = Join-Path $stateDir "last-model.txt"
  if ($running) { $model = $running }
  elseif (Test-Path $last) {
    $lp = (Get-Content $last -Raw).Trim()
    if ($lp -and (Test-Path $lp)) { $model = $lp }
  }
}
$sessArg = @{}; if ($null -ne $sessGb) { $sessArg.SessionCacheGb = $sessGb }  # (not given: the server's default, 8 GB)
if ($preserve) { $sessArg.PreserveThinking = $true }  # (only when given: a plain start keeps the running server)
if ($slots -gt 0) { $sessArg.Slots = $slots }  # (only when given, as above)
& (Join-Path $PSScriptRoot "arc-server.ps1") start -Model $model -Ctx $ctx -Chunk $chunk @sessArg -NoMtp:$noMtp -Vision $vision -NoStream:$noStream
if (-not $?) { exit 1 }

# (arc-server start has brought Pi's model list up to date: every model on disk, each with its own context)
$id = Get-LoadedId 5

$pi = Join-Path $env:USERPROFILE ".pi\agent\bin\pi.cmd"
if (-not (Test-Path $pi)) { $pi = "pi" }
& $pi --model "arcinferzero/$id" --thinking $thinking @piArgs
exit $LASTEXITCODE
