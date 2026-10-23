# arcinfer setup: everything needed on this laptop, in one command. Safe to run again: done steps are skipped.
#
#   setup                        check the requirements, build, choose models to download, connect Pi
#   setup -Models 27b,4b         download these without asking (27b 35b 9b 4b g12 e4b, all, none)
#   setup -Tests                 also build the test tools and run the regression tests (needs the 4B model)
#   setup -NoPi                  leave Pi alone
#   setup -Yes                   do not ask (pip installs, downloads of the models given with -Models)
#   setup -Help                  all options
#
# Steps: 1 requirements (Windows, Intel Arc GPU, Visual Studio C++ compiler, Python, Pi)  2 build bin\serverz.exe,
# runz.exe, clinfo.exe when missing or older than the sources  3 GPU check (OpenCL 3.0, DP4a, sub-groups, memory)
# 4 models (Hugging Face cache; the 35B is converted)  5 Pi (pi-arc / arc-server commands, Pi's model list)
# 6 regression tests (-Tests).
param([string[]]$Models = @(), [switch]$Tests, [switch]$NoPi, [switch]$Yes, [switch]$Help)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$bin = Join-Path $root "bin"
$Models = @(($Models -join ",") -split "[,;\s]+" | Where-Object { $_ } | ForEach-Object { $_.ToLower() })
if ($Help -or ($Models | Where-Object { $_ -in "--help", "-h", "help", "/?", "-?" })) {
  Write-Host @'
setup: everything arcinfer needs on this laptop, in one command. Safe to run again: done steps are skipped.

USAGE
  setup                          check, build, ask which models to download, connect Pi
  setup -Models 27b,4b           download these without asking: 27b 35b 9b 4b g12 e4b, or all / none
  setup -Tests                   also build the test tools and run the regression tests (needs the 4B model)
  setup -NoPi                    leave Pi alone
  setup -Yes                     do not ask: pip installs and downloads go ahead (with -Models: those models;
                                 without: none are downloaded)
  setup -Help, --help, -h        this text

STEPS
  1 requirements   64-bit Windows, an Intel Arc GPU and its driver, the Visual Studio C++ compiler (vswhere;
                   $env:ARC_VCVARS = a vcvars64.bat overrides), Python 3.10+ (downloads), Pi (optional).
                   Missing ones: where to get them.
  2 build          bin\serverz.exe, runz.exe, clinfo.exe when missing or older than their sources; the compiler
                   output goes to %LOCALAPPDATA%\arcinfer\build.log (shown when a build fails). A running server
                   is stopped first if serverz.exe has to be rebuilt.
  3 GPU check      bin\clinfo.exe: OpenCL device, DP4a (cl_khr_integer_dot_product), Intel sub-groups, GPU memory
  4 models         into the Hugging Face cache (HF_HUB_CACHE / HF_HOME respected): Gemma with its drafter and
                   image projector; the 35B downloaded and converted (tools\convert_gsq.py, with the 4B for its
                   tokenizer) into %LOCALAPPDATA%\arcinfer\models. huggingface_hub / numpy are installed with
                   pip after asking; the free disk space is checked. Then: every model the server finds, with
                   its context for the current GPU memory.
  5 Pi             scripts\install-pi.ps1: the pi-arc and arc-server commands, Pi's model list
  6 tests          with -Tests: tests\run_tests.ps1 (unit tests, templates, accuracy vs f32, MTP / DFlash2, slots, API)

MODELS
  27b  Qwen3.8 27B (ISTA-DASLab GSQ-RCO IQ3_S, MTP)        12.1 GB
  35b  Qwen3.6 35B-A3B 2-bit GSQ (converted)              12.6 GB download, 11.7 GB converted
  9b   Qwen3.5 9B (Unsloth UD-Q4_K_XL, MTP)                6.1 GB
  4b   Qwen3.5 4B (Unsloth UD-Q4_K_XL, MTP)                3.0 GB
  g12  Gemma 4 12B (Google QAT Q4_0) + image projector + drafter  7.4 GB
  e4b  Gemma 4 E4B (Google QAT Q4_0) + drafter                     5.2 GB

Not installed by setup: the graphics driver, Visual Studio Build Tools, Python and Pi (vendor installers).
'@
  exit 0
}

function Head($n, $t) { Write-Host ""; Write-Host "[$n/6] $t" -ForegroundColor Cyan }
function Ok($m) { Write-Host "  ok    $m" }
function Note($m) { Write-Host "  note  $m" -ForegroundColor Yellow }
function Bad($m, $fix) { Write-Host "  FAIL  $m" -ForegroundColor Red; if ($fix) { Write-Host "        $fix" }; exit 1 }
function Ask($q, [bool]$def = $true) {
  if ($Yes) { return $true }
  $a = Read-Host "  $q $(if ($def) { '[Y/n]' } else { '[y/N]' })"
  if (-not $a) { return $def }
  return $a -match '^(y|yes|ya)$'
}

Write-Host "arcinfer setup ($root)"

# ---------------------------------------------------------------- 1. requirements
Head 1 "Requirements"
if (-not [Environment]::Is64BitOperatingSystem) { Bad "64-bit Windows is required." }
$os = Get-CimInstance Win32_OperatingSystem
Ok "$($os.Caption) $($os.Version)"
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
Ok "CPU: $cpu"
$ramGb = [Math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
if ($ramGb -lt 15) { Note "RAM $ramGb GB: enough for the 4B / E4B models only (32 GB for the 12B-35B models)" } else { Ok "RAM: $ramGb GB" }
$gpus = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Intel" })
if (-not $gpus) { Bad "No Intel GPU found. arcinfer needs an Intel Core Ultra with Arc integrated graphics." }
foreach ($g in $gpus) { Ok "GPU: $($g.Name), driver $($g.DriverVersion)" }
if (-not ($gpus | Where-Object { $_.Name -match "Arc" })) { Note "only Intel Arc integrated graphics (Xe-LPG tested; Xe2 / Xe3 reported to work) is supported" }

# Visual Studio C++ compiler (the same search as build.ps1)
$vcvars = $null
if ($env:ARC_VCVARS -and (Test-Path $env:ARC_VCVARS)) { $vcvars = $env:ARC_VCVARS }
$vw = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not $vcvars -and (Test-Path $vw)) {
  foreach ($p in @(& $vw -all -products * -prerelease -property installationPath)) {
    $v = Join-Path $p "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $v) { $vcvars = $v; break }
  }
}
if (-not $vcvars) {
  Bad "No Visual Studio C++ compiler found." ("Install 'Build Tools for Visual Studio' (https://visualstudio.microsoft.com/downloads/, " +
    "under 'Tools for Visual Studio') with the workload 'Desktop development with C++', then run setup again.")
}
Ok "C++ compiler: $(Split-Path (Split-Path (Split-Path (Split-Path $vcvars))))"

# Python (model downloads, 35B conversion)
$py = $null  # the command line that runs Python, e.g. "python" or "py -3"
foreach ($c in "python", "py -3") {
  try {
    [string[]]$parts = $c -split " "
    $rest = @($parts | Select-Object -Skip 1)
    $v = "$(& $parts[0] @rest -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null)".Trim()
    if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(\d+)' -and [int]$Matches[1] -ge 10) { $py = $parts; Ok "Python $v"; break }
  } catch {}
}
if (-not $py) { Note "Python 3.10+ not found: needed to download models (https://www.python.org/downloads/, tick 'Add to PATH')" }
function Py { $rest = @($py | Select-Object -Skip 1); & $py[0] @rest @args }
# (Windows PowerShell turns a native program's redirected stderr into errors, fatal under "Stop": local Continue)
function PyHas($module) { $ErrorActionPreference = "Continue"; Py -c "import $module" 2>$null | Out-Null; return $LASTEXITCODE -eq 0 }

$piBin = Join-Path $env:USERPROFILE ".pi\agent\bin"
$hasPi = Test-Path $piBin
if ($hasPi) { Ok "Pi: $piBin" } elseif (-not $NoPi) { Note "Pi not found (optional): https://pi.dev, run it once, then run setup again" }

# ---------------------------------------------------------------- 2. build
Head 2 "Build"
# compiler output goes to %LOCALAPPDATA%\arcinfer\build.log; shown only when a build fails
$buildLog = Join-Path $env:LOCALAPPDATA "arcinfer\build.log"
New-Item -ItemType Directory -Force (Split-Path $buildLog) | Out-Null
function Build([string[]]$srcs) {
  $ErrorActionPreference = "Continue"
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "build.ps1") @srcs *> $buildLog
  if ($LASTEXITCODE -ne 0) {
    Get-Content $buildLog | Where-Object { $_ -match "error|failed" } | Select-Object -First 15 | ForEach-Object { Write-Host "        $_" }
    Bad "build failed" "full compiler output: $buildLog"
  }
  Ok "built $($srcs -join ', ')"
}
$newestHeader = (Get-ChildItem (Join-Path $root "src") -Filter *.h | Measure-Object LastWriteTime -Maximum).Maximum
# (arcinferzero: zero\serverz.cpp is tools\server.cpp on zero\engine.h, so it depends on both and on zero\*.h)
$newestZero = (Get-ChildItem (Join-Path $root "zero") -Filter *.h | Measure-Object LastWriteTime -Maximum).Maximum
$need = @()
foreach ($t in "tools\clinfo", "zero\serverz", "zero\runz") {
  $name = Split-Path $t -Leaf
  $exe = Join-Path $bin "$name.exe"
  $src = Get-Item (Join-Path $root "$t.cpp")
  $deps = @($newestHeader, $src.LastWriteTime)
  if ($t -like "zero*") { $deps += $newestZero }
  if ($t -eq "zero\serverz") { $deps += (Get-Item (Join-Path $root "tools\server.cpp")).LastWriteTime }
  $stale = -not (Test-Path $exe) -or @($deps | Where-Object { (Get-Item $exe).LastWriteTime -lt $_ }).Count -gt 0
  if ($stale) { $need += "$t.cpp" } else { Ok "bin\$name.exe is up to date" }
}
if ($need) {
  $running = @(Get-Process server, serverz -ErrorAction SilentlyContinue | Where-Object { $_.Path -like (Join-Path $bin "server*.exe") })
  if ($running.Count -and $need -contains "zero\serverz.cpp") {
    Note "stopping the running arcinfer server to rebuild it"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\arc-server.ps1") stop | Out-Null
  }
  Write-Host "  building $($need -join ', ') (about a minute)..."
  Build $need
}

# ---------------------------------------------------------------- 3. GPU check
Head 3 "GPU check (bin\clinfo.exe)"
$ErrorActionPreference = "Continue"
$ci = @(& (Join-Path $bin "clinfo.exe") 2>&1 | ForEach-Object { "$_" })
$ErrorActionPreference = "Stop"
$ciText = $ci -join "`n"
if ($LASTEXITCODE -ne 0 -or $ciText -notmatch "device:") {
  Bad "No OpenCL GPU found ($($ci | Select-Object -First 1))" "Install or update the Intel graphics driver (Intel Arc & Iris Xe Graphics, or Intel Driver & Support Assistant), restart, run setup again."
}
$dev = ($ci | Where-Object { $_ -match "device:" } | Select-Object -First 1).Trim()
Ok $dev
foreach ($ext in "cl_khr_integer_dot_product", "cl_intel_subgroups", "cl_intel_required_subgroup_size") {
  if ($ciText -notmatch "$ext\s+yes") { Bad "the GPU driver lacks $ext" "Update the Intel graphics driver." }
}
Ok "OpenCL: DP4a (cl_khr_integer_dot_product) and Intel sub-groups available"
if ($ciText -match "global mem: ([\d.]+) GiB") {
  Ok "GPU memory: $($Matches[1]) GiB (Intel Graphics Software can change it; each model's context follows it)"
}

# ---------------------------------------------------------------- 4. models
Head 4 "Models"
$hub = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE } elseif ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" } else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
$arcModels = Join-Path $env:LOCALAPPDATA "arcinfer\models"
$gsq35 = Join-Path $arcModels "Qwen3.6-35B-A3B-GSQ2-arc.gguf"
$catalog = @(
  [pscustomobject]@{ key = "27b"; name = "Qwen3.8 27B (best answers, slowest; MTP)"; repo = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"; gb = 12.1
    files = @("Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf") },
  [pscustomobject]@{ key = "35b"; name = "Qwen3.6 35B-A3B 2-bit (fast mixture of experts; converted)"; repo = "ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ"; gb = 12.6
    files = @() },
  [pscustomobject]@{ key = "9b"; name = "Qwen3.5 9B (MTP)"; repo = "unsloth/Qwen3.5-9B-MTP-GGUF"; gb = 6.1
    files = @("Qwen3.5-9B-UD-Q4_K_XL.gguf") },
  [pscustomobject]@{ key = "4b"; name = "Qwen3.5 4B (fastest; MTP)"; repo = "unsloth/Qwen3.5-4B-MTP-GGUF"; gb = 3.0
    files = @("Qwen3.5-4B-UD-Q4_K_XL.gguf") },
  # Google's QAT Q4_0 release has no drafter: Unsloth's copy of the assistant drafter is put next to the model
  [pscustomobject]@{ key = "g12"; name = "Gemma 4 12B (reads images; drafter)"; repo = "google/gemma-4-12B-it-qat-q4_0-gguf"; gb = 7.4
    files = @("gemma-4-12b-it-qat-q4_0.gguf", "mmproj-gemma-4-12b-it-qat-q4_0.gguf", "mtp-gemma-4-12B-it.gguf")
    drafter = @{ repo = "unsloth/gemma-4-12B-it-qat-GGUF"; file = "mtp-gemma-4-12B-it.gguf" } },
  # (its image projector, gemma4v, is not supported yet: not downloaded)
  [pscustomobject]@{ key = "e4b"; name = "Gemma 4 E4B (drafter)"; repo = "google/gemma-4-E4B-it-qat-q4_0-gguf"; gb = 5.2
    files = @("gemma-4-E4B_q4_0-it.gguf", "mtp-gemma-4-E4B-it.gguf")
    drafter = @{ repo = "unsloth/gemma-4-E4B-it-qat-GGUF"; file = "mtp-gemma-4-E4B-it.gguf" } }
)
function Present($m) {
  if ($m.key -eq "35b") { return Test-Path $gsq35 }
  $snaps = Join-Path $hub ("models--" + ($m.repo -replace "/", "--") + "\snapshots")
  if (-not (Test-Path $snaps)) { return $false }
  foreach ($f in $m.files) { if (-not (Get-ChildItem $snaps -Recurse -Filter $f -ErrorAction SilentlyContinue)) { return $false } }
  return $true
}
foreach ($m in $catalog) { $m | Add-Member have (Present $m) }
foreach ($m in $catalog) {
  Write-Host ("  {0} {1,-4} {2,-60} {3,5:N1} GB" -f $(if ($m.have) { "[x]" } else { "[ ]" }), $m.key, $m.name, $m.gb)
}
$want = @()
if ($Models) {
  if ($Models -contains "all") { $want = @($catalog | ForEach-Object { $_.key }) }
  elseif ($Models -notcontains "none") {
    foreach ($k in $Models) { if (-not ($catalog | Where-Object { $_.key -eq $k })) { Bad "unknown model '$k'" "use 27b 35b 9b 4b g12 e4b, all or none" } }
    $want = $Models
  }
} elseif (-not $Yes) {
  $a = Read-Host "  Models to download (e.g. '27b 4b'; Enter = none)"
  $want = @(($a -split "[,;\s]+") | Where-Object { $_ } | ForEach-Object { $_.ToLower() })
  foreach ($k in $want) { if (-not ($catalog | Where-Object { $_.key -eq $k })) { Bad "unknown model '$k'" "use 27b 35b 9b 4b g12 e4b" } }
}
$todo = @($catalog | Where-Object { $want -contains $_.key -and -not $_.have })
if ($todo | Where-Object { $_.key -eq "35b" }) {  # its tokenizer comes from the 4B GGUF
  $m4 = $catalog | Where-Object { $_.key -eq "4b" }
  $m4Queued = @($todo | Where-Object { $_.key -eq "4b" }).Count -gt 0
  if (-not $m4.have -and -not $m4Queued) { Note "the 35B conversion takes its tokenizer from the 4B model: downloading it too"; $todo = @($m4) + $todo }
}
if ($todo) {
  if (-not $py) { Bad "downloading models needs Python 3.10+" "Install Python (https://www.python.org/downloads/, tick 'Add to PATH'), then run setup again." }
  if (-not (PyHas "huggingface_hub")) {
    if (-not (Ask "Install the Python package huggingface_hub (pip)?")) { Bad "huggingface_hub is needed to download models" }
    Py -m pip install -U huggingface_hub
    if ($LASTEXITCODE -ne 0) { Bad "pip install huggingface_hub failed" }
  }
  $gb = ($todo | Measure-Object gb -Sum).Sum
  $drive = Get-PSDrive ((Split-Path $hub -Qualifier).TrimEnd(":")) -ErrorAction SilentlyContinue
  if ($drive -and $drive.Free / 1GB -lt $gb + 5) { Bad ("not enough disk space: {0:N0} GB needed, {1:N0} GB free on {2}:" -f ($gb + 5), ($drive.Free / 1GB), $drive.Name) }
  if (-not $Yes -and -not $Models) { if (-not (Ask ("Download {0} ({1:N1} GB)?" -f (($todo | ForEach-Object { $_.key }) -join ", "), $gb))) { $todo = @() } }
  foreach ($m in $todo) {
    Write-Host "  downloading $($m.key): $($m.repo)"
    if ($m.key -eq "35b") {
      if (-not (PyHas "numpy")) { Py -m pip install -U numpy; if ($LASTEXITCODE -ne 0) { Bad "pip install numpy failed" } }
      $snap = Py -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" $m.repo
      if ($LASTEXITCODE -ne 0) { Bad "download failed: $($m.repo)" }
      $snap = @($snap)[-1].Trim()
      $vocab = (Get-ChildItem $hub -Recurse -Filter "Qwen3.5-4B-UD-Q4_K_XL.gguf" | Select-Object -First 1).FullName
      New-Item -ItemType Directory -Force $arcModels | Out-Null
      Write-Host "  converting to $gsq35 (about 3 minutes)..."
      Py (Join-Path $root "tools\convert_gsq.py") $snap $gsq35 --vocab-from $vocab
      if ($LASTEXITCODE -ne 0) { Bad "conversion failed" }
    } else {
      [string[]]$own = $m.files
      if ($m.drafter) { $own = @($m.files | Where-Object { $_ -ne $m.drafter.file }) }
      [string[]]$dl = @("-c", "import sys; from huggingface_hub import hf_hub_download; print([hf_hub_download(sys.argv[1], f) for f in sys.argv[2:]][0])", $m.repo) + $own
      $first = Py @dl
      if ($LASTEXITCODE -ne 0) { Bad "download failed: $($m.repo)" }
      if ($m.drafter) {  # the engine looks for the drafter in the model's folder
        $dpath = Py -c "import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(sys.argv[1], sys.argv[2]))" $m.drafter.repo $m.drafter.file
        if ($LASTEXITCODE -ne 0) { Bad "download failed: $($m.drafter.repo)" }
        Copy-Item (@($dpath)[-1].Trim()) (Join-Path (Split-Path (@($first)[-1].Trim())) $m.drafter.file) -Force
      }
    }
    Ok "$($m.key) downloaded"
  }
}
# what the server finds, with each model's context for the current GPU memory (computed in seconds, remembered)
$ErrorActionPreference = "Continue"
$found = @(& (Join-Path $bin "serverz.exe") --list-models 2>$null | ConvertFrom-Json | ForEach-Object { $_ }) | Where-Object { $_.supported }
$ErrorActionPreference = "Stop"
if ($found) {
  Write-Host "  The server finds:"
  foreach ($f in $found) { Write-Host ("    {0,-32} context {1,7:N0} tokens   ({2})" -f $f.alias, $f.ctx, $f.name) }
} else {
  Note "no model yet: run setup again with -Models (e.g. setup -Models 4b)"
}

# ---------------------------------------------------------------- 5. Pi
Head 5 "Pi"
if ($NoPi) { Ok "skipped (-NoPi)" }
elseif (-not $hasPi) { Note "Pi not installed: https://pi.dev; afterwards run setup again (or scripts\install-pi.ps1)" }
else {
  $ErrorActionPreference = "Continue"
  $out = @(& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\install-pi.ps1") 2>&1 | ForEach-Object { "$_" })
  $code = $LASTEXITCODE
  $ErrorActionPreference = "Stop"
  if ($code -ne 0) { $out | ForEach-Object { Write-Host "        $_" }; Bad "connecting Pi failed" }
  $out | Where-Object { $_.Trim() } | ForEach-Object { Ok $_ }
}

# ---------------------------------------------------------------- 6. regression tests
Head 6 "Regression tests"
if (-not $Tests) { Ok "skipped (setup -Tests runs them: kernels, tokenizer, templates, accuracy vs f32, MTP, API)" }
else {
  $m4 = (Get-ChildItem $hub -Recurse -Filter "Qwen3.5-4B-UD-Q4_K_XL.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
  if (-not $m4) { Note "the tests need the 4B model: setup -Models 4b -Tests" }
  else {
    [string[]]$tools = @("tools\tok_test.cpp", "tools\chat_render.cpp", "tools\ref_forward.cpp", "tools\tool_stream_test.cpp",
                         "tests\test_http.cpp", "tests\test_api_compat.cpp", "tests\test_xcache_pools.cpp", "tests\test_sampler.cpp",
                         "tests\test_hc.cpp", "tests\test_quant_rows.cpp", "tests\test_web_tools.cpp")
    Write-Host "  building the test tools..."
    Build $tools
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\arc-server.ps1") stop | Out-Null  # (GPU memory)
    $g = (Get-ChildItem $hub -Recurse -Filter "gemma-4-12b-it-qat-q4_0.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
    $targs = @{ Model4B = $m4 }; if ($g) { $targs.ModelGemma = $g }
    Push-Location $root
    try { & (Join-Path $root "tests\run_tests.ps1") @targs } finally { Pop-Location }
  }
}

# ---------------------------------------------------------------- done
Write-Host ""
Write-Host "Done." -ForegroundColor Green
if ($hasPi -and -not $NoPi) {
  Write-Host "  Open a new terminal, go to a project folder and run:  pi-arc        (pi-arc --help: all options)"
}
Write-Host "  Server without Pi:  scripts\arc-server.ps1 start   (web UI http://127.0.0.1:8090/, API http://127.0.0.1:8090/v1)"
Write-Host "  The first start of a model compiles its GPU kernels once (a minute or two)."
