# Prompt processing and generation speed at growing context depth, for every model given (or the usual ones in the
# Hugging Face cache). For each depth N: one N-token prompt (average prompt tok/s), then 32 generated tokens at that
# depth (generation tok/s). Plain decoding (no MTP), as runz.exe --bench-pp.
#   .\bench\long_ctx.ps1 [-Depths 512,16384,32768,65536] [-Models path1,path2] [-Out bench\results\long_ctx.txt]
param(
  [int[]]$Depths = @(512, 16384, 32768, 65536),
  [string[]]$Models = @(),
  [string]$Out = "bench\results\long_ctx.txt",
  [int]$Gen = 32
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$hub = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
if (-not $Models.Count) {
  $names = "Qwen3.5-4B-UD-Q4_K_XL.gguf", "Qwen3.5-9B-UD-Q4_K_XL.gguf", "gemma-4-E4B_q4_0-it.gguf",
    "gemma-4-12b-it-qat-q4_0.gguf", "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf", "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
  foreach ($n in $names) {
    $f = Get-ChildItem -Recurse $hub -Filter $n -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -notmatch "MTP-GGUF\\snapshots\\5bc3" -and $_.FullName -notmatch "ukisai" } | Select-Object -First 1
    if ($f) { $Models += $f.FullName }
  }
}
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
"# long-context speed $(Get-Date -Format s)  (prompt tok/s = average over the whole prompt; gen = at that depth)" | Add-Content $Out
$env:ARC_BENCH_RUNS = "1"
foreach ($m in $Models) {
  $name = Split-Path $m -Leaf
  foreach ($d in $Depths) {
    $o = & .\bin\runz.exe $m --bench-pp $d --n $Gen 2>&1 | Out-String
    $pp = [regex]::Match($o, "pp\d+ run 0: ([\d.]+) ms\s+->\s+([\d.]+) tokens/s")
    $tg = [regex]::Match($o, "tg\d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s")
    $line = if ($pp.Success -and $tg.Success) {
      "{0,-40} depth {1,6}: prompt {2,8:N1} tok/s ({3,6:N1} s)   gen {4,6:N2} tok/s" -f $name, $d, [double]$pp.Groups[2].Value, ([double]$pp.Groups[1].Value / 1000), [double]$tg.Groups[1].Value
    } else {
      "{0,-40} depth {1,6}: FAILED  {2}" -f $name, $d, (($o -split "`n" | Select-Object -Last 3) -join " | ")
    }
    $line | Add-Content $Out
    Write-Host $line
  }
}
Remove-Item env:ARC_BENCH_RUNS
