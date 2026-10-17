# arcinfer regression tests. Run from any folder (the script switches to the project folder itself).
#
# Usage examples:
#   .\tests\run_tests.ps1 -SkipGpu                       # model-free CPU checks only (no GPU, no model)
#   .\tests\run_tests.ps1                                # default engine (zero = arcinferzero), auto-detected models
#   .\tests\run_tests.ps1 -Engine opencl -Model4B D:\models\Qwen3.5-4B-Q4_K_M.gguf
#   .\tests\run_tests.ps1 -Engine both -Model27B <27b.gguf> -DFlash27B <z-lab Qwen3.8-27B-DFlash2 dir>
#   .\tests\run_tests.ps1 -Model35B <35b-a3b GSQ gguf> -Ref35B tests\data\ref_35b_capital_all.bin
#   .\tests\run_tests.ps1 -SkipGpu -PackGguf <small.gguf>  # also round-trips a GGUF through zero/gguf_pack.h
#   .\tests\run_tests.ps1 -NoAutoDetect -Model4B <4b.gguf> # only the models given on the command line
#
# Parameters:
#   -Engine zero|opencl|both   zero: bin\runz.exe + bin\serverz.exe (default), opencl: bin\run.exe + bin\server.exe
#   -Model4B / -Model27B / -Model35B / -ModelGemma / -ModelFlashNext   GGUF files (Flash Next: the -00001-of- shard)
#   -DFlash27B / -DFlash35B    DFlash2 drafter folders (containing model.safetensors), zero engine only
#   -Ref35B <file>             opt-in 35B KL check. tests\data\ref_35b_capital_all.bin is the f32 reference of the
#                              Qwen3.6-35B-A3B GSQ file (tools\convert_gsq.py), not of Unsloth's UD-IQ4_XS: only pass it
#                              together with the matching -Model35B. A missing file is created with ref_forward.
#   -PackGguf <file>           small GGUF for bin\test_gguf_pack.exe (output goes to %TEMP% and is deleted)
#   -SkipGpu                   run only the model-free CPU checks
#   -ListModels                print the models every check would use and exit (runs nothing)
#   -Port <n>                  server smoke-test port (default 8091)
#
# Models that are not given are searched in %USERPROFILE%\.cache\huggingface\hub\models--*\snapshots and in the
# folders listed one per line in %LOCALAPPDATA%\arcinfer\model-dirs.txt (# starts a comment; missing folders, e.g.
# an unplugged drive, are ignored). A check whose model is missing prints [SKIP] and does not fail.
#
# Checks: model-free unit tests (HTTP parser, API compatibility, expert-cache pools, tool-call streaming, chat
# template rendering vs the GGUF Jinja templates, hyper-connection kernels), every quant type + kernel path of the
# OpenCL engine against a CPU reference, tokenizer vs reference ids, next-token KL vs the f32 CPU reference, greedy
# continuation, MTP / DFlash2 speculative decoding = plain greedy, and an OpenAI API smoke test with streaming tool
# calls, prefix reuse and malformed requests.
param(
  [ValidateSet("zero", "opencl", "both")] [string]$Engine = "zero",
  [string]$Model4B = "",
  [string]$Model27B = "",
  [string]$Model35B = "",
  [string]$ModelGemma = "",
  [string]$ModelFlashNext = "",
  [string]$DFlash27B = "",
  [string]$DFlash35B = "",
  [string]$Ref35B = "",
  [string]$PackGguf = "",
  [string]$Data = "tests\data",
  [int]$Port = 8091,
  [int]$ServerWait = 600,
  [switch]$SkipGpu,
  [switch]$NoAutoDetect,
  [switch]$ListModels
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# resolve the user's relative paths against the folder the script was started from, before switching to the project
function Resolve-Arg([string]$p) {
  if (-not $p) { return "" }
  if (Test-Path -LiteralPath $p) { return (Resolve-Path -LiteralPath $p).Path }
  return $p  # kept as given: the check reports it as missing
}
$Model4B = Resolve-Arg $Model4B; $Model27B = Resolve-Arg $Model27B; $Model35B = Resolve-Arg $Model35B
$ModelGemma = Resolve-Arg $ModelGemma; $ModelFlashNext = Resolve-Arg $ModelFlashNext
$DFlash27B = Resolve-Arg $DFlash27B; $DFlash35B = Resolve-Arg $DFlash35B; $PackGguf = Resolve-Arg $PackGguf
if ($Ref35B -and -not [System.IO.Path]::IsPathRooted($Ref35B)) { $Ref35B = Join-Path (Get-Location).Path $Ref35B }
if (-not [System.IO.Path]::IsPathRooted($Data)) {
  $Data = $(if (Test-Path -LiteralPath $Data) { (Resolve-Path -LiteralPath $Data).Path } else { Join-Path $root $Data })
}
Push-Location $root

$script:fail = 0; $script:pass = 0; $script:skip = 0
function Check($name, $ok, $detail = "") {
  if ($ok) { $script:pass++ } else { $script:fail++ }
  Write-Host ("[{0}] {1} {2}" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $name, $detail)
}
function Skip($name, $reason) {
  $script:skip++
  Write-Host ("[SKIP] {0} ({1})" -f $name, $reason)
}
function Section($title) { Write-Host ""; Write-Host "== $title" }

# runs a native program and returns stdout + stderr as one string. Windows PowerShell 5.1 turns stderr lines into
# terminating errors under ErrorActionPreference=Stop, hence the local Continue.
function Run-Exe([string]$exe, [string[]]$argv = @()) {
  $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
  try { $o = & $exe @argv 2>&1 | ForEach-Object { "$_" } | Out-String } finally { $ErrorActionPreference = $old }
  $script:exitCode = $LASTEXITCODE
  return $o
}
function Get-KL([string]$o) {
  $m = [regex]::Match($o, "KL\(ref \|\| arcinfer\) = ([\d.]+)")
  if (-not $m.Success) { return $null }
  return [double]::Parse($m.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}
function Get-Greedy([string]$o) { return [regex]::Match($o, "(?m)^greedy:[^\r\n]*").Value.Trim() }
function Make-Ref([string]$model, [string]$ids, [string]$ref) {
  Write-Host "   creating the f32 reference $ref with bin\ref_forward.exe (slow)"
  $env:REF_ALL = "1"
  try { Run-Exe .\bin\ref_forward.exe @($model, $ids, $ref) | Out-Null } finally { $env:REF_ALL = $null }
}

# ---------------------------------------------------------------------------------------------------------------
# model discovery
$searchRoots = @()
$hub = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
if (Test-Path -LiteralPath $hub) {
  $searchRoots += @(Get-ChildItem -LiteralPath $hub -Directory -Filter "models--*" -ErrorAction SilentlyContinue |
    ForEach-Object { Join-Path $_.FullName "snapshots" } | Where-Object { Test-Path -LiteralPath $_ })
}
$dirsFile = Join-Path $env:LOCALAPPDATA "arcinfer\model-dirs.txt"
if (Test-Path -LiteralPath $dirsFile) {
  foreach ($line in Get-Content -LiteralPath $dirsFile) {
    $d = $line.Trim()
    if (-not $d -or $d.StartsWith("#")) { continue }
    if (Test-Path -LiteralPath $d) { $searchRoots += $d } else { Write-Host "   (model folder $d not found, ignored)" }
  }
}
$script:allFiles = $null
function All-Files {
  if ($null -eq $script:allFiles) {
    $script:allFiles = @(foreach ($r in $searchRoots) {
      Get-ChildItem -LiteralPath $r -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -eq ".gguf" -or $_.Name -eq "model.safetensors" }
    })
  }
  return $script:allFiles
}
function Find-Gguf([string[]]$patterns) {
  foreach ($p in $patterns) {
    $f = All-Files | Where-Object { $_.Extension -eq ".gguf" -and $_.Name -like $p -and $_.Name -notlike "mmproj*" } |
      Select-Object -First 1
    if ($f) { return $f.FullName }
  }
  return ""
}
function Find-DFlash([string[]]$patterns) {  # a folder whose path matches and that holds model.safetensors
  foreach ($p in $patterns) {
    $f = All-Files | Where-Object { $_.Name -eq "model.safetensors" -and $_.DirectoryName -like $p } | Select-Object -First 1
    if ($f) { return $f.DirectoryName }
  }
  return ""
}
if (-not $NoAutoDetect -and (-not $SkipGpu -or $ListModels)) {
  if (-not $Model4B) { $Model4B = Find-Gguf @("Qwen3.5-4B*.gguf", "qwen3.5*4b*.gguf") }
  if (-not $Model27B) { $Model27B = Find-Gguf @("Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf") }
  if (-not $Model35B) { $Model35B = Find-Gguf @("Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf") }
  if (-not $ModelFlashNext) { $ModelFlashNext = Find-Gguf @("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf") }
  if (-not $ModelGemma) { $ModelGemma = Find-Gguf @("gemma-4-12b*qat*q4_0*.gguf", "gemma-4-12b*qat*.gguf") }
  if (-not $DFlash27B) { $DFlash27B = Find-DFlash @("*Qwen3.8-27B-DFlash2*") }
  if (-not $DFlash35B) { $DFlash35B = Find-DFlash @("*Qwen3.6-35B-A3B-DFlash2*") }
}
function Have([string]$p) { return $p -and (Test-Path -LiteralPath $p) }
function Why([string]$p, [string]$what) { if ($p) { return "$what not found: $p" } else { return "no $what" } }

$engines = $(if ($Engine -eq "both") { @("zero", "opencl") } else { @($Engine) })
$bins = @{ zero = @{ run = ".\bin\runz.exe"; server = ".\bin\serverz.exe" }; opencl = @{ run = ".\bin\run.exe"; server = ".\bin\server.exe" } }
Write-Host ("arcinfer tests: engine {0}{1}" -f ($engines -join " + "), $(if ($SkipGpu) { ", model-free CPU checks only (-SkipGpu)" } else { "" }))
if (-not $SkipGpu -or $ListModels) {
  foreach ($kv in @(@("4B", $Model4B), @("27B", $Model27B), @("35B-A3B", $Model35B), @("Flash Next", $ModelFlashNext),
                    @("Gemma 4 12B", $ModelGemma), @("DFlash2 27B", $DFlash27B), @("DFlash2 35B", $DFlash35B))) {
    Write-Host ("   {0,-12} {1}" -f $kv[0], $(if (Have $kv[1]) { $kv[1] } elseif ($kv[1]) { "(missing: $($kv[1]))" } else { "(none)" }))
  }
}

if ($ListModels) { Pop-Location; exit 0 }

# ---------------------------------------------------------------------------------------------------------------
# 1. model-free checks (CPU, plus the hyper-connection kernel test on random data)
Section "model-free checks"
function Unit($name, $exe, [string[]]$argv, $pattern) {
  if (-not (Test-Path -LiteralPath $exe)) { Skip $name "$exe not built"; return }
  $o = Run-Exe $exe $argv
  $ok = $o -match $pattern -and $script:exitCode -eq 0
  if (-not $ok) { Write-Host $o.TrimEnd() }
  Check $name $ok
}
Unit "HTTP parser (malformed requests)" .\bin\test_http.exe @() "test_http: all passed"
Unit "API compatibility" .\bin\test_api_compat.exe @() "all passed"
Unit "expert cache pools" .\bin\test_xcache_pools.exe @() "ALL OK"
if (Test-Path .\bin\test_sampler.exe) { Unit "sampler" .\bin\test_sampler.exe @() "ALL OK" }
Unit "tool call streaming" .\bin\tool_stream_test.exe @("3000") "ALL OK"

# chat template rendering vs Jinja (needs python + jinja2)
$tq35 = Join-Path $Data "qwen35_template.jinja"; $tq38 = Join-Path $Data "qwen38_template.jinja"
if (-not ((Test-Path $tq35) -and (Test-Path $tq38))) { Skip "chat template (14 cases)" "templates missing in $Data" }
elseif (-not (Test-Path .\bin\chat_render.exe)) { Skip "chat template (14 cases)" "bin\chat_render.exe not built" }
elseif (-not (Get-Command python -ErrorAction SilentlyContinue)) { Skip "chat template (14 cases)" "python not found" }
else {
  $o = Run-Exe python @("tests\test_chat_render.py", "$root\bin\chat_render.exe", $tq35, $tq38)
  if ($o -notmatch "ALL IDENTICAL") { Write-Host $o.TrimEnd() }
  Check "chat template (14 cases)" ($o -match "ALL IDENTICAL")
}

# GGUF repacking (zero/gguf_pack.h) must reproduce every key and tensor
if (-not $PackGguf) { Skip "GGUF pack round trip" "no -PackGguf" }
elseif (-not (Test-Path -LiteralPath $PackGguf)) { Skip "GGUF pack round trip" "not found: $PackGguf" }
else {
  $out = Join-Path $env:TEMP ("arcinfer_pack_test_{0}.gguf" -f $PID)
  try { Unit "GGUF pack round trip ($([System.IO.Path]::GetFileName($PackGguf)))" .\bin\test_gguf_pack.exe @($PackGguf, $out) "identical" }
  finally { Remove-Item -LiteralPath $out -Force -ErrorAction SilentlyContinue }
}

# hyper-connection kernels of zero/kernels/ops.cl (Flash Next) vs a CPU computation on random data (uses the GPU)
if ($SkipGpu) { Skip "hyper-connection kernels (zero)" "-SkipGpu" }
elseif ($engines -notcontains "zero") { Skip "hyper-connection kernels (zero)" "engine $Engine" }
else { Unit "hyper-connection kernels (zero)" .\bin\test_hc.exe @() "ALL OK" }

if ($SkipGpu) {
  Pop-Location
  Write-Host ""
  $sk = $(if ($script:skip) { ", $($script:skip) skipped" } else { "" })
  if ($script:fail -eq 0) { Write-Host "ALL TESTS PASSED ($($script:pass) passed$sk)" } else { Write-Host "$($script:fail) TEST(S) FAILED ($($script:pass) passed$sk)"; exit 1 }
  exit 0
}

try {
# ---------------------------------------------------------------------------------------------------------------
# 2. OpenCL kernels: GEMM (native + JIT W8), GEMV for every weight type present (bin\test_gemm.exe uses src/clrt.h)
if ($engines -contains "opencl") {
  Section "OpenCL kernels"
  if (-not (Have $Model4B)) { Skip "kernels (4B types) (OpenCL kernels)" (Why $Model4B "4B model") }
  else {
    Write-Host "   model: $Model4B"
    $t4 = Run-Exe .\bin\test_gemm.exe @($Model4B, "64", "blk.0.ffn_gate.weight", "blk.2.ffn_gate.weight", "blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "token_embd.weight")
    Check "kernels (4B types) (OpenCL kernels)" ($t4 -match "ALL OK")
  }
  if (-not (Have $Model27B)) { Skip "kernels (27B IQ types) (OpenCL kernels)" (Why $Model27B "27B model") }
  else {
    Write-Host "   model: $Model27B"
    $t27 = Run-Exe .\bin\test_gemm.exe @($Model27B, "64", "blk.0.attn_gate.weight", "blk.0.ffn_down.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.1.attn_gate.weight", "blk.1.attn_qkv.weight", "blk.1.ssm_out.weight", "blk.7.attn_q.weight", "blk.13.ffn_gate.weight", "blk.64.ffn_down.weight")
    Check "kernels (27B IQ types) (OpenCL kernels)" ($t27 -match "ALL OK")
  }
}

# 3. tokenizer vs reference ids (llama-tokenize --no-escape); engine independent
Section "tokenizer"
$prompts = @(Get-ChildItem $Data -Filter "*.prompt.txt" -ErrorAction SilentlyContinue)
if ($prompts.Count -eq 0) { Skip "tokenizer" "no *.prompt.txt in $Data" }
elseif (-not (Have $Model4B)) { Skip "tokenizer" (Why $Model4B "4B model") }
else {
  Write-Host "   model: $Model4B"
  foreach ($p in $prompts) {
    $idf = $p.FullName -replace "\.prompt\.txt$", ".ids.txt"
    if (-not (Test-Path $idf)) { continue }
    $o = Run-Exe .\bin\tok_test.exe @($Model4B, $p.FullName, $idf)
    Check "tokenizer $($p.Name)" ($o -match "IDENTICAL" -and $o -match "round-trip OK")
  }
}

# the 20-token Qwen chat prompt shared by the Qwen checks
$ids = "248045,846,198,213077,173634,172069,23220,30,243381,189040,13,248046,198,248045,74455,198,248068,271,248069,271"
$greedy4b = "greedy: 40 208756 5899 23220 36316 2972 195793 159034 248046"

foreach ($eng in $engines) {
  $run = $bins[$eng].run; $tag = "[$eng]"
  Section "model checks, engine $eng ($run)"
  if (-not (Test-Path -LiteralPath $run)) { Skip "$tag model checks" "$run not built"; continue }

  # 4. Qwen3.5 4B: next-token distributions vs the f32 CPU reference, averaged over all 20 prompt positions (a single
  #    position is too sensitive to individual int8 rounding flips). int8 K/V cache + int8 activations: ~0.002.
  if (-not (Have $Model4B)) { Skip "$tag 4B KL / greedy / MTP" (Why $Model4B "4B model") }
  else {
    Write-Host "   4B: $Model4B"
    $ref = Join-Path $Data "ref_4b_capital_all.bin"
    if (-not (Test-Path $ref)) { Make-Ref $Model4B $ids $ref }
    $o = Run-Exe $run @($Model4B, "--tokens", $ids, "--n", "9", "--kl-ref", $ref)
    $kl = Get-KL $o
    Check "$tag KL vs f32 reference (mean of 20 positions)" ($null -ne $kl -and $kl -lt 0.005) ("KL = {0}" -f $(if ($null -eq $kl) { "(no KL line)" } else { $kl }))
    Check "$tag greedy continuation" ($o -match $greedy4b) (Get-Greedy $o)
    # 4b. speculative decoding with the MTP block must reproduce the plain greedy continuation
    $o = Run-Exe $run @($Model4B, "--tokens", $ids, "--n", "9", "--mtp")
    $acc = [regex]::Match($o, "MTP: (\d+) drafts, (\d+) accepted")
    Check "$tag MTP speculative decoding (greedy identical)" ($o -match $greedy4b) ("drafts {0}, accepted {1}" -f $acc.Groups[1].Value, $acc.Groups[2].Value)
  }

  # 4c. Gemma 4 12B: next-token distributions vs its f32 CPU reference (ref_forward), greedy = llama.cpp's, and the
  #     assistant drafter must reproduce the greedy continuation. Gemma 4 is ~7x more sensitive to int8 rounding than
  #     Qwen (an f32 reference with int8 activations alone gives KL ~0.009), hence the looser bound.
  if (-not (Have $ModelGemma)) { Skip "$tag Gemma 4 KL / greedy / drafter" (Why $ModelGemma "Gemma 4 12B model") }
  else {
    Write-Host "   Gemma 4 12B: $ModelGemma"
    $gids = "2,2,105,2364,107,155122,528,1156,23974,3217,506,7217,563,3730,236761,106,107,105,4368,107"
    $gref = Join-Path $Data "ref_gemma12b_google_chat_all.bin"  # (f32 reference of Google's QAT Q4_0 12B)
    if (-not (Test-Path $gref)) { Make-Ref $ModelGemma $gids $gref }
    $o = Run-Exe $run @($ModelGemma, "--tokens", $gids, "--n", "12", "--kl-ref", $gref)
    $kl = Get-KL $o
    Check "$tag Gemma 4 KL vs f32 reference (mean of 20 positions)" ($null -ne $kl -and $kl -lt 0.05) ("KL = {0}" -f $(if ($null -eq $kl) { "(no KL line)" } else { $kl }))
    $greedy = "greedy: 100 45518 107 101 818 7217 7412 3730 1547 529 496 20284"
    Check "$tag Gemma 4 greedy = llama.cpp" ($o -match $greedy) (Get-Greedy $o)
    $o = Run-Exe $run @($ModelGemma, "--tokens", $gids, "--n", "12", "--mtp")
    Check "$tag Gemma 4 assistant drafter (greedy identical)" ($o -match $greedy) ([regex]::Match($o, "MTP: [^\r\n]*").Value)
  }

  # 4d. Qwen3.6-35B-A3B GSQ (tools\convert_gsq.py): MoE path vs its f32 CPU reference. Opt-in: the reference in
  #     tests\data was made from the GSQ file, not from the auto-detected UD-IQ4_XS.
  if (-not $Ref35B) { Skip "$tag 35B-A3B MoE KL vs f32 reference" "opt-in: -Ref35B <file>" }
  elseif (-not (Have $Model35B)) { Skip "$tag 35B-A3B MoE KL vs f32 reference" (Why $Model35B "35B-A3B model") }
  else {
    Write-Host "   35B-A3B: $Model35B, reference $Ref35B"
    if (-not (Test-Path -LiteralPath $Ref35B)) { Make-Ref $Model35B $ids $Ref35B }
    $o = Run-Exe $run @($Model35B, "--tokens", $ids, "--n", "4", "--kl-ref", $Ref35B)
    $kl = Get-KL $o
    Check "$tag 35B-A3B MoE KL vs f32 reference (mean of 20 positions)" ($null -ne $kl -and $kl -lt 0.005) ("KL = {0}" -f $(if ($null -eq $kl) { "(no KL line)" } else { $kl }))
  }

  # 4e. every available Qwen model: MTP and DFlash2 speculative decoding must reproduce the plain greedy continuation
  $qwen = @(
    @{ name = "27B"; model = $Model27B; dflash = $DFlash27B; dname = "-DFlash27B" },
    @{ name = "35B-A3B"; model = $Model35B; dflash = $DFlash35B; dname = "-DFlash35B" },
    @{ name = "Flash Next"; model = $ModelFlashNext; dflash = ""; dname = "" },
    @{ name = "4B"; model = $Model4B; dflash = ""; dname = "" })
  foreach ($q in $qwen) {
    $n = "$tag $($q.name)"
    if (-not (Have $q.model)) { Skip "$n MTP / DFlash2 = greedy" (Why $q.model "$($q.name) model"); continue }
    if ($eng -eq "opencl" -and $q.name -eq "Flash Next") { Skip "$n MTP = greedy" "the OpenCL engine has no hyper-connections"; continue }
    Write-Host "   $($q.name): $($q.model)"
    $o = Run-Exe $run @($q.model, "--tokens", $ids, "--n", "16")
    $g0 = Get-Greedy $o
    if (-not $g0) { Write-Host $o.TrimEnd() }
    Check "$n plain greedy" ([bool]$g0) $g0
    if (-not $g0) { continue }
    $o = Run-Exe $run @($q.model, "--tokens", $ids, "--n", "16", "--mtp")
    $g1 = Get-Greedy $o
    Check "$n MTP greedy identical" ($g1 -eq $g0) $(if ($g1 -eq $g0) { [regex]::Match($o, "MTP: [^\r\n]*").Value } else { $g1 })
    if (-not $q.dname) { continue }
    if ($eng -eq "opencl") { Skip "$n DFlash2 greedy identical" "the OpenCL engine has no DFlash2"; continue }
    if (-not (Have $q.dflash)) { Skip "$n DFlash2 greedy identical" (Why $q.dflash "DFlash2 drafter ($($q.dname))"); continue }
    Write-Host "   $($q.name) DFlash2: $($q.dflash)"
    $o = Run-Exe $run @($q.model, "--tokens", $ids, "--n", "16", "--dflash", $q.dflash)
    $g2 = Get-Greedy $o
    Check "$n DFlash2 greedy identical" ($g2 -eq $g0) $(if ($g2 -eq $g0) { [regex]::Match($o, "MTP: [^\r\n]*").Value } else { $g2 })
  }
}

# ---------------------------------------------------------------------------------------------------------------
# 5. server: streaming chat with a tool call, a follow-up that must reuse the cached prefix, malformed requests

# raw HTTP exchange (status line + headers + body as text); the send side is closed so the server ends the reply
function Raw-Http([int]$port, [string]$req) {
  $c = New-Object System.Net.Sockets.TcpClient
  try {
    $c.ReceiveTimeout = 30000; $c.SendTimeout = 10000
    $c.Connect("127.0.0.1", $port)
    $s = $c.GetStream()
    $b = [System.Text.Encoding]::UTF8.GetBytes($req)
    $s.Write($b, 0, $b.Length); $s.Flush()
    $c.Client.Shutdown([System.Net.Sockets.SocketShutdown]::Send)
    $ms = New-Object System.IO.MemoryStream
    $buf = New-Object byte[] 4096
    try { while (($n = $s.Read($buf, 0, $buf.Length)) -gt 0) { $ms.Write($buf, 0, $n) } } catch {}
    return [System.Text.Encoding]::UTF8.GetString($ms.ToArray())
  } catch { return "<no connection: $($_.Exception.Message)>" } finally { $c.Close() }
}
function Health([int]$port) {
  try { return (Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 5).status -eq "ok" } catch { return $false }
}
function Status-Line([string]$r) { return ($r -split "`r?`n")[0] }

foreach ($eng in $engines) {
  $exe = $bins[$eng].server; $tag = "[$eng]"
  Section "server smoke test, engine $eng ($exe)"
  $sm = @($Model4B, $Model27B, $Model35B) | Where-Object { Have $_ } | Select-Object -First 1
  if (-not (Test-Path -LiteralPath $exe)) { Skip "$tag server smoke test" "$exe not built"; continue }
  if (-not $sm) { Skip "$tag server smoke test" "no 4B, 27B or 35B-A3B model"; continue }
  Write-Host "   model: $sm, port $Port"
  $srv = Start-Process -FilePath $exe -ArgumentList @("-m", "`"$sm`"", "--port", $Port, "--ctx", "8192", "--alias", "test", "--no-prefix-cache") -PassThru -WindowStyle Hidden
  try {
    $up = $false
    for ($i = 0; $i -lt $ServerWait -and -not $srv.HasExited; $i++) {
      Start-Sleep -Milliseconds 1000
      if (Health $Port) { $up = $true; break }
    }
    Check "$tag server /health" $up $(if ($srv.HasExited) { "(server exited, code $($srv.ExitCode))" } elseif (-not $up) { "(no answer in $ServerWait s)" } else { "" })
    if (-not $up) { continue }
    $o = Run-Exe python @("tests\test_tools.py", "http://127.0.0.1:$Port")
    Check "$tag API tool call" ($o -match "turn 1: finish=tool_calls")
    Check "$tag API prefix reuse" ($o -match "cache_n=([1-9]\d*)")

    $body = '{"model":"test","prompt":"","max_tokens":4}'
    $r = Raw-Http $Port ("POST /v1/completions HTTP/1.1`r`nHost: 127.0.0.1`r`nContent-Type: application/json`r`nContent-Length: {0}`r`n`r`n{1}" -f [System.Text.Encoding]::UTF8.GetByteCount($body), $body)
    Check "$tag empty prompt -> 400" ((Status-Line $r) -match "^HTTP/1\.[01] 400") (Status-Line $r)
    Check "$tag /health after the empty prompt" (Health $Port)

    $r = Raw-Http $Port "POST /v1/completions HTTP/1.1`r`nHost: 127.0.0.1`r`nContent-Type: application/json`r`nContent-Length: abc`r`n`r`n"
    Check "$tag Content-Length abc -> 400" ((Status-Line $r) -match "^HTTP/1\.[01] 400") (Status-Line $r)
    Check "$tag /health after the malformed Content-Length" (Health $Port)
  } finally {
    if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue }
  }
}
} finally {
  Pop-Location
}

Write-Host ""
$sk = $(if ($script:skip) { ", $($script:skip) skipped" } else { "" })
if ($script:fail -eq 0) { Write-Host "ALL TESTS PASSED ($($script:pass) passed$sk)" } else { Write-Host "$($script:fail) TEST(S) FAILED ($($script:pass) passed$sk)"; exit 1 }
