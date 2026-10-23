# Full comparison run: tuning of llama.cpp micro-batch size, then every model x engine x test.
# Results: bench\results\results.jsonl (one JSON object per measurement). Summarize with bench\report.py.
param(
  [string]$PromptDir = "",
  [string]$Corpus = "",
  [string[]]$Models = @("4b", "9b", "27b"),
  [string[]]$Engines = @("arcinferzero", "vulkan", "sycl"),
  [switch]$SkipTuning
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $root "bench\results"
New-Item -ItemType Directory -Force $outDir | Out-Null
$out = Join-Path $outDir "results.jsonl"
$hub = "$env:USERPROFILE\.cache\huggingface\hub"
$paths = @{
  "4b"  = (Get-ChildItem $hub -Recurse -Filter "Qwen3.5-4B-UD-Q4_K_XL.gguf" | Select-Object -First 1).FullName
  "9b"  = (Get-ChildItem $hub -Recurse -Filter "Qwen3.5-9B-UD-Q4_K_XL.gguf" | Select-Object -First 1).FullName
  "27b" = (Get-ChildItem $hub -Recurse -Filter "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" | Select-Object -First 1).FullName
}
$env:GGML_SYCL_ENABLE_VMM = "0"  # see run_bench.ps1

# 1. llama.cpp tuning per model and backend: micro-batch size for pp2048 with flash attention on (on the 4B model
#    flash attention off is measured as well). SYCL environment options were checked separately: defaults are best.
$ub = @{}
foreach ($m in $Models) {
  foreach ($be in @("vulkan", "sycl") | Where-Object { $Engines -contains $_ }) {
    $ub["$m/$be"] = 512
    if ($SkipTuning) { continue }
    $exe = "$env:USERPROFILE\llama-builds\b11835-$be\llama-bench.exe"
    $fa = if ($m -eq "4b") { "0,1" } else { "1" }
    Start-Sleep -Seconds 30
    $o = & $exe -m $paths[$m] -ngl 99 -fa $fa -ub "512,1024,2048" -b 2048 -p 2048 -n 0 -r 1 -o json 2>$null | Out-String
    $rows = $o | ConvertFrom-Json
    foreach ($r in $rows) {
      @{ engine = $be; test = "tune"; model = $m; n_ubatch = $r.n_ubatch; flash_attn = $r.flash_attn; tps = $r.avg_ts; time = (Get-Date).ToString("s") } |
        ConvertTo-Json -Compress | Add-Content $out -Encoding utf8
    }
    $best = $rows | Where-Object { $_.flash_attn } | Sort-Object avg_ts -Descending | Select-Object -First 1
    $ub["$m/$be"] = [int]$best.n_ubatch
    Write-Host "$m $be best: ub=$($best.n_ubatch) -> $([math]::Round($best.avg_ts, 1)) t/s"
  }
}

# 2. all models x engines
$prompts = @()
if ($PromptDir) { $prompts = @(Get-ChildItem $PromptDir -Filter "*.ids.txt" | Sort-Object Length | ForEach-Object FullName) }
foreach ($m in $Models) {
  foreach ($e in $Engines) {
    $u = if ($ub.ContainsKey("$m/$e")) { $ub["$m/$e"] } else { 512 }
    & (Join-Path $PSScriptRoot "run_bench.ps1") -Model $paths[$m] -Name $m -Engines $e -Tests "pp", "ttft", "ppl" `
      -PromptFiles $prompts -Corpus $Corpus -Out $out -Ubatch $u
  }
}
python (Join-Path $PSScriptRoot "report.py") $out
