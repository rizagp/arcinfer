# Starts / stops the arcinferzero server in the background for Pi and other OpenAI-compatible clients.
#
#   arc-server start                  asks which model to load: every supported GGUF model in the Hugging Face cache
#                                     (and %LOCALAPPDATA%\arcinferzero\models), listed by its full name; Enter keeps the
#                                     running one. Waits until the server is ready.
#   arc-server start -Model 27b       the same without asking: a shortcut (27b 35b 35b-s 35b-mtp flash 9b 4b g12 e4b g26,
#                                     with -mtp / -vision / -std / -nodf / -df for a variant), a model id, part of a model's
#                                     name (e.g. swift) or a path to a .gguf file
#        [-SessionCacheGb N]          save conversations to disk when switching (N GB; default 8, 0 = off)
#        [-Vision on|off]             image / audio encoders (default on; off: no mmproj, more context)
#        [-Ctx N|auto] [-Chunk N] [-NoMtp]  context (default: auto, the largest that fits the GPU memory),
#                                     prompt chunk size, no speculative decoding
#        [-NoStream]                  MoE experts on the GPU instead of streamed from the SSD (bin\serverz.exe,
#                                     Level Zero: the engine; the OpenCL one, bin\server.exe, was removed 2026-10-20)
#   arc-server stop                   stop the server
#   arc-server status                 show what is running and the saved conversations
#   arc-server clear-cache            delete the saved conversations (session cache) of all models
#
# The server keeps running after Pi exits, so the next session skips the model load and reuses the prompt cache.
# Log: %LOCALAPPDATA%\arcinferzero\server.log
param(
  [ValidateSet("start", "stop", "status", "restart", "clear-cache", "help", "--help", "-h")] [string]$Action = "start",
  [string]$Model = "",
  [string]$Ctx = "",
  [int]$Chunk = 0,
  [double]$SessionCacheGb = 8,
  [int]$Port = 8090,
  [switch]$NoMtp,
  [ValidateSet("", "on", "off")] [string]$Vision = "",
  [switch]$NoStream,
  [int]$Slots = 0,
  # the other server options (0 / empty / off = the server's default)
  [int]$MtpDrafts = 0,
  [int]$ThinkBudget = 8192,
  [int]$ImageMinTokens = 0,
  [int]$ImageMaxTokens = 0,
  [double]$BudgetGb = 0,
  [switch]$AttnF32,
  [switch]$PreserveThinking,
  [switch]$NoPrefixCache,
  [double]$PrefixCacheGb = -1,
  [switch]$NoSwap,
  [string]$ListenHost = "",
  [string]$SessionDir = "",
  [switch]$Quiet
)
$ErrorActionPreference = "Stop"
# -Ctx: a number of tokens, or auto (which also restarts a server running a fixed context); not given (or 0): a
# running server is kept as it is, a new one starts with auto
$Ctx = "$Ctx".Trim().ToLower()
if ($Ctx -eq "0") { $Ctx = "" }
if ($Ctx -and $Ctx -ne "auto" -and $Ctx -notmatch '^\d+$') { throw "-Ctx: a number of tokens or auto (got '$Ctx')" }
$root = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "app-dir.ps1")
$stateDir = Get-AppDir  # (%LOCALAPPDATA%\arcinferzero; renamed from arcinfer once)
New-Item -ItemType Directory -Force $stateDir | Out-Null
# the engine: arcinferzero (Level Zero); -NoStream keeps every expert on the GPU
$Engine = if ($NoStream) { "zero-nostream" } else { "zero" }
# ($moe: the model has routed experts, from serverz --list-models; only those keep their experts on the SSD)
function Engine-Text([string]$e, [bool]$moe = $false) {
  switch ($e) {
    "zero" { if ($moe) { "arcinferzero (Level Zero; experts on the SSD with a GPU cache)" } else { "arcinferzero (Level Zero)" } }
    "zero-nostream" { "arcinferzero without the SSD cache (Level Zero; every weight on the GPU)" }
    default { "an older server ($e)" }
  }
}
$exe = Join-Path $root "bin\serverz.exe"
$log = Join-Path $stateDir "server.log"
$stateFile = Join-Path $stateDir "server.json"
$base = "http://127.0.0.1:$Port"
# saved conversations (server --session-cache-gb, default 8 GB)
$sessDir = Join-Path $stateDir "zero\sessions"

# The models come from `serverz.exe --list-models` (Hugging Face cache + %LOCALAPPDATA%\arcinferzero\models): name, file,
# alias (= the model id of the API and Pi), shortcut, the context --ctx auto chose for the current GPU memory. The
# context is auto unless -Ctx is given: the largest that fits the GPU memory currently set less 1.2 GiB for the
# driver, at most the trained one (Intel Graphics Software can change the GPU memory; it is then
# computed again). Smaller prompt chunks (-Chunk 1024) free ~1.1 GiB on the 27B for more context.
$lastFile = Join-Path $stateDir "last-model.txt"  # the model started last (pi-arc starts it again by default)

function Say($msg) { if (-not $Quiet) { Write-Host $msg } }

# The id of the model the server has loaded: the entry marked loaded (the server lists every model it can switch
# to), else the first (servers that list only the loaded one).
function Get-Running {
  try { $d = @((Invoke-RestMethod "$base/v1/models" -TimeoutSec 2).data) } catch { return $null }
  $l = @($d | Where-Object { $_.loaded -eq $true })
  if ($l.Count) { return $l[0].id } else { return $d[0].id }
}

# Model files compared by full path: serverz.exe lists Hugging Face cache paths with mixed separators
# (C:\Users\me/.cache/huggingface/hub\models--...), a resolved path has only backslashes.
function Same-Path($a, $b) {
  if (-not $a -or -not $b) { return $false }
  try { return [IO.Path]::GetFullPath($a) -eq [IO.Path]::GetFullPath($b) } catch { return $a -eq $b }
}

function Stop-Arc {
  $stopped = $false
  # the conversation the server holds goes to the session cache first (when it has one), so that continuing it after
  # the restart loads it instead of processing it again
  try { $null = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/sessions/save" -TimeoutSec 60 } catch {}
  if (Test-Path $stateFile) {
    $st = Get-Content $stateFile -Raw | ConvertFrom-Json
    $p = Get-Process -Id $st.pid -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq "serverz") { Stop-Process -Id $p.Id -Force; $stopped = $true }
    Remove-Item $stateFile -Force
  }
  # a server started by hand on the same port
  Get-CimInstance Win32_Process -Filter "Name = 'serverz.exe'" |
    Where-Object { $_.ExecutablePath -like "$root\bin\serverz.exe" -and $_.CommandLine -match "--port\s+$Port\b" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $stopped = $true }
  # A new server that allocates its GPU memory while Windows is still releasing the old one's can run at half speed
  # for its whole life (measured: Qwen3.6 35B on arcinferzero 26.4 -> 11.9 tok/s when started at once, normal with
  # 5 s in between): the process ends first, then a short pause.
  if ($stopped) {
    Get-Process -Name server, serverz -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$root\bin\server*.exe" } |
      Wait-Process -Timeout 20 -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
    Say "arcinferzero server stopped."
  }
  return $stopped
}

# The models serverz.exe finds (Hugging Face cache + arcinferzero's model folder), with their alias.
function Get-Models {
  $all = @(& $exe --list-models | ConvertFrom-Json | ForEach-Object { $_ })  # (Windows PowerShell: unroll the array)
  foreach ($m in $all) { $m | Add-Member ctx 0 -Force }  # (only -Ctx sets a fixed context here)
  return $all
}

# Pi's model list (~/.pi/agent/models.json, provider "arcinferzero") follows what is on disk: a model deleted from the
# Hugging Face cache leaves Pi's /model, a new one appears; each with its context (the running one's from the server)
function Sync-PiModels {
  if (-not (Test-Path (Join-Path $env:USERPROFILE ".pi\agent"))) { return }
  . (Join-Path $PSScriptRoot "pi-models.ps1")
  $cfgPath = Join-Path $env:USERPROFILE ".pi\agent\models.json"
  $before = if (Test-Path $cfgPath) { (Get-Item $cfgPath).LastWriteTime } else { $null }
  $id = Get-Running
  $n = 0
  if ($id) { try { $n = [int](Invoke-RestMethod "$base/props" -TimeoutSec 5).n_ctx } catch {} }
  try { $ids = @(Update-PiModels $exe $base $id $n) } catch { Say "  (Pi's model list not updated: $_)"; return }
  if ((Test-Path $cfgPath) -and (Get-Item $cfgPath).LastWriteTime -ne $before) { Say "Pi's model list updated: $($ids.Count) models" }
}

function Pick-Model($models, $runningPath, $runningAlias) {
  $ok = @($models | Where-Object { $_.supported })
  Write-Host ""
  Write-Host "Models (Hugging Face cache and $stateDir\models):"
  for ($i = 0; $i -lt $ok.Count; $i++) {
    $m = $ok[$i]
    $tags = @("{0:N1} GB" -f ($m.bytes / 1e9))
    if ($m.ctx_train) { $tags += "max context length {0:N0}" -f $m.ctx_train }
    if ($m.tags) { $tags += $m.tags }
    if ($m.ctx) { $tags += "auto context {0:N0}" -f $m.ctx }
    if ($m.short) { $tags += "shortcut $($m.short)" } else { $tags += "id $($m.alias)" }
    $mark = if ($runningAlias -and $m.alias -eq $runningAlias) { "  <- running" } else { "" }
    Write-Host ("  {0,2}) {1}" -f ($i + 1), $m.name) -NoNewline -ForegroundColor Cyan
    Write-Host ("   {0}  [{1}]{2}" -f $m.file, ($tags -join ", "), $mark)
  }
  $skip = @($models | Where-Object { -not $_.supported })
  if ($skip.Count) { Write-Host ("  (not supported by arcinferzero: " + (($skip | ForEach-Object { "$($_.name) [$($_.arch)]" }) -join ", ") + ")") }
  $prompt = if ($runningPath) { "Model number (Enter = keep the running model)" } else { "Model number" }
  while ($true) {
    $a = Read-Host $prompt
    if (-not $a -and $runningPath) { return $null }
    $n = 0
    if ([int]::TryParse($a, [ref]$n) -and $n -ge 1 -and $n -le $ok.Count) { return $ok[$n - 1] }
    Write-Host "Type a number from 1 to $($ok.Count)."
  }
}

if ($Action -in "help", "--help", "-h") {
  Write-Host @"
arc-server: the arcinferzero server in the background (port $Port), for Pi and other OpenAI-compatible clients.

  arc-server start [options]     start, or switch to another model; waits until ready. Also brings Pi's model
                                 list up to date (a model deleted from the Hugging Face cache leaves it, a new
                                 one appears), even when the server is already running.
  arc-server restart [options]   restart with new options
  arc-server status              running model, its context and chunk size, saved conversations
  arc-server stop                stop the server and free the GPU memory
  arc-server clear-cache         delete saved conversations (session cache). Saved system prompts (per project
                                 folder, 8 GB in all, reused across restarts) are in %LOCALAPPDATA%\arcinferzero\zero\prefix;
                                 deleting that folder is safe while the server is stopped.
  arc-server help                this text (also --help, -h)

Options for start / restart:
  -Model <shortcut|text|path>    27b 35b 35b-s 35b-mtp flash 9b 4b g12 e4b g26; a variant with -mtp (MTP, no images),
                                 -vision (images, no MTP) or -std (neither), e.g. 9b-std, 27b-std; a model id
                                 (as in Pi's /model), part of a model's name (e.g. swift) or a .gguf path.
                                 -nodf / -df (models with a DFlash2 drafter): the MTP block only / DFlash2 only.
                                 Without it: a numbered list of the models in the Hugging Face cache.
  -Ctx <N|auto>                  context in tokens, or auto = the largest that fits the GPU memory currently
                                 set, less 1.2 GiB for the driver (computed on the first start
                                 of a model, remembered, recomputed when the GPU memory changes). Not given:
                                 a running server keeps its context, a new one starts with auto. -Ctx auto
                                 also restarts a server that runs a fixed context.
  -Chunk <N>                     prompt chunk size (default per model: 27B 4096; 35B and Gemma 4 26B 8192; Flash
                                 Next 2048; 4B / 9B / E4B 512; Gemma with images at least 2048); smaller = less
                                 GPU memory, more context
  -SessionCacheGb <N>            save conversations to disk when switching sessions (default 8 GB, 0 = off)
  -NoMtp                         no speculative decoding
  -Slots <N>                     hold N conversations at once (default 1): two Pi sessions each keep their own
                                 KV cache and states and take turns step by step (a prompt piece, a token), so
                                 neither waits for the other's whole answer; while both write, their next tokens
                                 run in one pass. Each slot gets the
                                 context that fits N of them (-Ctx auto; 27B: 2 x 49,152, 35B: 2 x 65,536, g26 2 x 32,768)
  -NoStream                      every expert on the GPU: no expert file, no SSD reads; MoE models then get a
                                 smaller context. Without it, MoE models (35B, Flash Next, Gemma 4 26B) keep their
                                 experts (MTP layer included) on the SSD and the most used ones in a GPU cache
                                 and read the next layer's missing experts while the GPU computes (35B and Flash
                                 Next: 131,072 tokens of context by default, up to 262,144 with -Ctx; Windows keeps
                                 14 GB). The first start writes the expert file (35B: 16 GB, ~2.5 min; Flash Next:
                                 56 GB, ~9.5 min; Gemma 4 26B: 13 GB, ~1 min) to %LOCALAPPDATA%\arcinferzero\zero (safe
                                 to delete: it is written again).
  -Vision on|off                 image / audio input (default on). off: the projector (mmproj) is not loaded and
                                 --ctx auto keeps no GPU memory for it, so the context is longer
  -Port <N>                      port (default 8090; also for status / stop)
  -MtpDrafts <1..7>              drafts per step (default: 7 for Qwen 27B-class and with DFlash2, 5
                                 for MoE, 3 for 4B, 2 otherwise)
  -ThinkBudget <N>               default reasoning budget in tokens (default 8192; requests can override it)
  -ImageMinTokens / -ImageMaxTokens <N>  tokens per image (defaults: Qwen 256..2048, Gemma 4 12B 1120..2048)
  -BudgetGb <GB>                 keep decoded weights resident up to GB (default: chosen with the context)
  -AttnF32                       f32 attention (tests)
  -PreserveThinking              earlier turns keep their reasoning (Qwen3.6 35B; Qwen3.8 27B always does)
  -NoPrefixCache                 do not keep system prompts on disk;  -PrefixCacheGb <N>  their disk space (8)
  -NoSwap                        the server keeps its model (requests for another model do not switch)
  -ListenHost <addr>             listen address (default 127.0.0.1; 0.0.0.0 = other computers too)
  -SessionDir <folder>           where saved conversations go (default %LOCALAPPDATA%\arcinferzero\zero\sessions)
  -Quiet                         no messages
A running server is kept when the model and every option given match; otherwise it is restarted with them.

A window for all of this: arc-launcher.cmd in the project folder (models with their memory and automatic
context, every option filled in with the best value, Start / Stop, the running server's memory and speeds).

The server is started as bin\serverz.exe with --ctx, --mtp (when the model has
an MTP block or a Gemma drafter), --draft for the -nodf / -df variants, --think-budget 8192 and the options above;
its other options (bin\serverz.exe --help) keep their defaults.

Log: $log
Everything (Pi launcher, server options, files): pi-arc --help
"@
  return
}

switch ($Action) {
  "status" {
    $id = Get-Running
    if ($id) {
      $p = Invoke-RestMethod "$base/props" -TimeoutSec 2
      Say "arcinferzero server running on $base : model $id"
      if (Test-Path $lastFile) { Say ("  " + (Get-Content $lastFile -Raw).Trim()) }  # (also after switching in Pi)
      Say ("  " + ($p | ConvertTo-Json -Compress -Depth 3))
    } else { Say "arcinferzero server is not running." }
    $files = @(Get-ChildItem $sessDir -Recurse -Filter *.arcs -ErrorAction SilentlyContinue)
    Say ("Saved conversations: {0} ({1:N1} GiB) in {2}" -f $files.Count, (($files | Measure-Object Length -Sum).Sum / 1GB), $sessDir)
    return
  }
  "clear-cache" {
    $n = 0; $bytes = 0
    if (Get-Running) {  # the running server forgets its saved conversations and deletes their files
      try { $r = Invoke-RestMethod -Method Delete "$base/sessions" -TimeoutSec 600; $n += $r.deleted; $bytes += $r.freed_bytes } catch {}
    }
    $files = @(Get-ChildItem $sessDir -Recurse -Filter *.arcs -ErrorAction SilentlyContinue)  # other models'
    $n += $files.Count; $bytes += ($files | Measure-Object Length -Sum).Sum
    $files | ForEach-Object { [IO.File]::Delete($_.FullName) }
    Say ("Deleted {0} saved conversations ({1:N1} GiB)." -f $n, ($bytes / 1GB))
    return
  }
  "stop" { if (-not (Stop-Arc)) { Say "arcinferzero server is not running." }; return }
}

# ---- start / restart
if (-not (Test-Path $exe)) { throw "$exe not found: build it with .\build.ps1 zero\serverz.cpp (or run setup)" }
$running = Get-Running
$st = if ($running -and (Test-Path $stateFile)) { Get-Content $stateFile -Raw | ConvertFrom-Json } else { $null }
$models = Get-Models
if (-not $Model) {
  $m = Pick-Model $models $(if ($st) { $st.path } else { $null }) $running
  if (-not $m) { Say "Keeping $running."; return }
} elseif (Test-Path $Model -PathType Leaf) {
  $full = (Resolve-Path $Model).Path
  $m = $models | Where-Object { Same-Path $_.path $full } | Select-Object -First 1
  if (-not $m) { $m = [pscustomobject]@{ name = $full; file = Split-Path $full -Leaf; path = $full; mtp = $true; alias = [IO.Path]::GetFileNameWithoutExtension($full).ToLower(); ctx = 0 } }
} else {
  $m = $models | Where-Object { $_.supported -and ($_.short -eq $Model.ToLower() -or $_.alias -eq $Model.ToLower()) } | Select-Object -First 1
  if (-not $m) {
    $hits = @($models | Where-Object { $_.supported -and ($_.name + " " + $_.file) -like "*$Model*" })
    # (the variants of one file all match its name: then the file's main entry, the one listed first)
    $files = @($hits | ForEach-Object { $_.path } | Select-Object -Unique)
    if ($files.Count -eq 1) { $hits = @($hits[0]) }
    if ($hits.Count -ne 1) {
      $pool = if ($hits.Count) { $hits } else { @($models | Where-Object { $_.supported }) }
      $list = ($pool | ForEach-Object { "  {0,-34} {1,-14} {2}" -f $_.alias, $(if ($_.short) { $_.short } else { "-" }), $_.file }) -join "`n"
      $what = if ($hits.Count) { "matches $($hits.Count) models" } else { "matches no model" }
      throw "'$Model' $what. Use an id or a shortcut:`n  id                                 shortcut       file`n$list"
    }
    $m = $hits[0]
  }
}
# -Vision off / -NoMtp: the variant of the same file without the image projector / the MTP block (every
# combination a file supports is listed by the server: <id>, <id>-mtp, <id>-vision, <id>-std)
if (($Vision -eq "off" -and $m.vision) -or ($NoMtp -and $m.mtp) -or ($Vision -eq "on" -and -not $m.vision)) {
  $wantVision = $(if ($Vision -eq "off") { $false } elseif ($Vision -eq "on") { $true } else { [bool]$m.vision })
  $wantMtp = [bool]$m.mtp -and -not $NoMtp
  $v = $models | Where-Object { $_.supported -and (Same-Path $_.path $m.path) -and [bool]$_.vision -eq $wantVision -and [bool]$_.mtp -eq $wantMtp } | Select-Object -First 1
  if ($v) { $m = $v }
}
# a restart (another model, other options) keeps the running server's slots unless -Slots is given: pi-arc on
# another model dropped them (and the second Pi / client then waited for the only slot)
if (-not $PSBoundParameters.ContainsKey("Slots") -and $st -and "$($st.extra)" -match "--slots (\d+)") { $Slots = [int]$Matches[1] }
$ctxArg = if ($Ctx) { $Ctx } elseif ($Slots -gt 1) { "auto" } elseif ($m.ctx) { "$($m.ctx)" } else { "auto" }  # (the listed context is for one slot)
$novision = $(if ($m.vision -eq $false) { "1" } else { "0" })
# the other options, as server arguments (also kept in the state file: a change restarts the server)
$extra = @()
if ($MtpDrafts -gt 0) { $extra += @("--mtp-drafts", $MtpDrafts) }
if ($Slots -gt 1) { $extra += @("--slots", $Slots) }
if ($ImageMinTokens -gt 0) { $extra += @("--image-min-tokens", $ImageMinTokens) }
if ($ImageMaxTokens -gt 0) { $extra += @("--image-max-tokens", $ImageMaxTokens) }
if ($BudgetGb -gt 0) { $extra += @("--budget", $BudgetGb) }
if ($AttnF32) { $extra += "--attn-f32" }
if ($PreserveThinking) { $extra += "--preserve-thinking" }
if ($NoPrefixCache) { $extra += "--no-prefix-cache" }
if ($PrefixCacheGb -ge 0) { $extra += @("--prefix-cache-gb", $PrefixCacheGb) }
if ($NoSwap) { $extra += "--no-swap" }
if ($ListenHost) { $extra += @("--host", $ListenHost) }
if ($SessionDir) { $extra += @("--session-dir", "`"$SessionDir`"") }
$extraKey = "$($extra -join ' ')|think $ThinkBudget"
# (compared with the running server's only when given: pi-arc's plain start keeps a server started with other options)
$extraGiven = @("MtpDrafts", "ThinkBudget", "ImageMinTokens", "ImageMaxTokens", "BudgetGb", "AttnF32", "PreserveThinking", "NoPrefixCache",
  "PrefixCacheGb", "NoSwap", "ListenHost", "SessionDir", "Slots" | Where-Object { $PSBoundParameters.ContainsKey($_) }).Count -gt 0
# the running server is kept unless an option given now differs from what it runs with (empty / 0 = not given)
$same = $running -eq $m.alias -and (-not $Ctx -or ($st -and "$($st.ctx)" -eq $ctxArg)) -and
  ($Chunk -eq 0 -or ($st -and "$($st.chunk)" -eq "$Chunk")) -and (-not $PSBoundParameters.ContainsKey("SessionCacheGb") -or ($st -and "$($st.sess)" -eq "$SessionCacheGb")) -and
  (-not $extraGiven -or ($st -and "$(if ($st.extra) { $st.extra } else { '|think 8192' })" -eq $extraKey)) -and
  (-not $st -or "$(if ($st.engine) { $st.engine } else { 'opencl' })" -eq $Engine)
if ($Action -eq "start" -and $same) {
  Say "Server already running: $running on $base"
  Say "  engine: $(Engine-Text $Engine ([bool]$m.moe))"
  Sync-PiModels
  return
}
if ($running) {
  $was = if ($st -and $st.engine) { $st.engine } else { "opencl" }  # (a server.json without engine: the old OpenCL one)
  Say "Switching the server: $running ($(Engine-Text $was))"
  Say "                  to: $($m.alias) ($(Engine-Text $Engine ([bool]$m.moe)))"
  Stop-Arc | Out-Null
}
elseif ($Action -eq "restart") { Stop-Arc | Out-Null }

$useMtp = [bool]$m.mtp  # (the variant's setting; -NoMtp picked the variant without it)
$srvArgs = @("-m", "`"$($m.path)`"", "--port", $Port, "--ctx", $ctxArg, "--alias", $m.alias, "--think-budget", $ThinkBudget) + $extra
if ($useMtp) { $srvArgs += "--mtp" }
if ($useMtp -and $m.draft -and $m.draft -ne "auto") { $srvArgs += @("--draft", $m.draft) }  # (-nodf: MTP block only, -df: DFlash2 only)
if ($NoStream) { $srvArgs += "--no-stream" }
if ($Chunk -gt 0) { $srvArgs += @("--chunk", $Chunk) }
if ($SessionCacheGb -gt 0) { $srvArgs += @("--session-cache-gb", $SessionCacheGb) }  # saved conversations on disk (default 8 GB, 0 = off)
if ($novision -eq "1") { $srvArgs += "--no-mmproj" }
Say "Starting $(if ($NoStream) { 'arcinferzero (no SSD cache)' } else { 'arcinferzero' }): $($m.name) as $($m.alias), context $ctxArg, port $Port$(if ($useMtp) { ', speculative decoding' })"
Say "  engine: $(Engine-Text $Engine ([bool]$m.moe))"
if (-not $NoStream) {
  Say "  (MoE models: the first start writes the expert file to $stateDir\zero, 35B: 16 GB / ~2.5 min, Flash Next: 56 GB / ~9.5 min, once)"
}
Say "  (loading takes 5-40 s; log: $log)"
$p = Start-Process -FilePath $exe -ArgumentList $srvArgs -WorkingDirectory $root -WindowStyle Hidden -PassThru `
  -RedirectStandardError $log -RedirectStandardOutput (Join-Path $stateDir "server.out.log")
Set-Content $lastFile $m.path -Encoding utf8
@{ pid = $p.Id; engine = $Engine; model = $m.name; path = $m.path; alias = $m.alias; ctx = $ctxArg; chunk = $(if ($Chunk -gt 0) { "$Chunk" } else { "default" }); sess = "$SessionCacheGb"; novision = $novision; extra = $extraKey; port = $Port; started = (Get-Date).ToString("s") } |
  ConvertTo-Json | Set-Content $stateFile -Encoding utf8

$t0 = Get-Date
while ($true) {
  Start-Sleep -Milliseconds 700
  if ($p.HasExited) {
    Write-Host "The server stopped while loading. Last log lines:" -ForegroundColor Red
    Get-Content $log -Tail 15 | ForEach-Object { Write-Host "  $_" }
    Remove-Item $stateFile -Force -ErrorAction SilentlyContinue
    exit 1
  }
  try { if ((Invoke-RestMethod "$base/health" -TimeoutSec 2).status -eq "ok") { break } } catch {}
  if (((Get-Date) - $t0).TotalSeconds -gt 600) { throw "The server did not become ready within 10 minutes; see $log" }
}
$nctx = (Invoke-RestMethod "$base/props" -TimeoutSec 5).n_ctx
Say ("Ready after {0:N0} s: {1}, context {2:N0} (web UI {3}/)" -f ((Get-Date) - $t0).TotalSeconds, $m.alias, $nctx, $base)
Say "  engine: $(Engine-Text $Engine ([bool]$m.moe))"
if ($true) {
  # the engine's own lines: the expert cache (MoE) or that the model has no experts to stream
  $zl = @(Get-Content $log -ErrorAction SilentlyContinue | Where-Object { $_ -match "^arcinferzero: (expert cache|cache filled|expert file written|memory limit)" })
  if ($zl.Count) {
    foreach ($l in $zl) { Say ("  " + ($l -replace "^arcinferzero: ", "") -replace ", file .*$", "") }
  } elseif (-not $NoStream) {
    Say "  dense model: no MoE experts, so no SSD cache"
  }
}
Sync-PiModels
