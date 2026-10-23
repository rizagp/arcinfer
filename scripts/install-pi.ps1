# One-time setup of arcinferzero for Pi on this laptop (setup.cmd runs it too):
#   - builds bin\serverz.exe if it is missing,
#   - adds the "arcinferzero" provider to ~/.pi/agent/models.json: every model serverz.exe finds (Hugging Face cache and
#     %LOCALAPPDATA%\arcinferzero\models) with its context for the current GPU memory (other providers are kept; the
#     previous file is saved as models.json.bak). pi-arc refreshes the list on every start.
#   - installs the `pi-arc` and `arc-server` commands into ~/.pi/agent/bin (already on PATH).
# Run:  powershell -ExecutionPolicy Bypass -File scripts\install-pi.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$piDir = Join-Path $env:USERPROFILE ".pi\agent"
$binDir = Join-Path $piDir "bin"
if (-not (Test-Path $binDir)) { throw "Pi is not installed ($binDir not found): install it from https://pi.dev and run it once." }

# 1. server binary
$exe = Join-Path $root "bin\serverz.exe"
if (-not (Test-Path $exe)) {
  Write-Host "Building bin\serverz.exe..."
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "build.ps1") zero\serverz.cpp
  if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

# 2. Pi model configuration
. (Join-Path $PSScriptRoot "pi-models.ps1")
$ids = @(Update-PiModels $exe "http://127.0.0.1:8090")
$cfgPath = Join-Path $piDir "models.json"
if ($ids.Count) { Write-Host "Pi models: $(($ids | ForEach-Object { "arcinferzero/$_" }) -join ', ')  ($cfgPath)" }
else { Write-Host "Pi models: none yet (download a model, then run this again or just start pi-arc)" }

# 3. commands on PATH
foreach ($name in "pi-arc", "arc-server") {
  $target = Join-Path $root "scripts\$name.ps1"
  "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"$target`" %*`r`n" | Set-Content (Join-Path $binDir "$name.cmd") -Encoding ascii -NoNewline
}
Write-Host "Commands installed: pi-arc, arc-server  ($binDir)"
