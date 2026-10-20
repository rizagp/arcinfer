# Benchmarks arcinfer against llama.cpp (official Vulkan and SYCL builds) on the same models, prompts and text.
# Every measurement is appended as one JSON object per line to $Out.
#
#   .\bench\run_bench.ps1 -Model <gguf> -Name 4b -Engines arcinfer,vulkan,sycl -Tests pp,ttft,ppl -Out results.jsonl
#
# Tests:  pp   = synthetic prompt processing (pp512, pp2048) + token generation (tg64), llama-bench style
#         ttft = time to first token on real code prompts (token-id files) + generation speed at that context
#         ppl  = perplexity on a text corpus (ctx 512, second half of each chunk scored, like llama-perplexity)
param(
  [Parameter(Mandatory)] [string]$Model,
  [Parameter(Mandatory)] [string]$Name,
  [string[]]$Engines = @("arcinfer", "vulkan", "sycl"),
  [string[]]$Tests = @("pp", "ttft", "ppl"),
  [string[]]$PromptFiles = @(),
  [string]$Corpus = "",
  [int]$PplChunks = 40,
  [string]$Out = "bench_results.jsonl",
  [string]$LlamaRoot = "$env:USERPROFILE\llama-builds",
  [string]$LlamaBuild = "b11527",
  [int]$Ubatch = 512,
  [int]$Repeats = 2,
  [int]$Cooldown = 30   # seconds of idle GPU before each measurement (the laptop throttles under sustained load)
)
$ErrorActionPreference = "Stop"
# The official SYCL build fails to map its virtual-memory pool on this driver (UR_RESULT_ERROR_INVALID_VALUE);
# the legacy pool works and is not slower (measured).
$env:GGML_SYCL_ENABLE_VMM = "0"
function Cool { if ($Cooldown -gt 0) { Start-Sleep -Seconds $Cooldown } }
$arc = Join-Path $PSScriptRoot "..\bin\runz.exe"
$arcDir = Join-Path $PSScriptRoot ".."

function Emit($obj) {
  $obj["model"] = $Name
  $obj["time"] = (Get-Date).ToString("s")
  ($obj | ConvertTo-Json -Compress -Depth 5) | Add-Content -Path $Out -Encoding utf8
  Write-Host ("  -> " + ($obj | ConvertTo-Json -Compress -Depth 5))
}

function Llama($engine, $exe) { Join-Path $LlamaRoot "$LlamaBuild-$engine\$exe" }

function Start-LlamaServer($engine, $ctx, $port) {
  $exe = Llama $engine "llama-server.exe"
  $log = Join-Path $env:TEMP "arcbench-$engine-server.log"
  $args = @("-m", $Model, "--port", $port, "-c", $ctx, "-ngl", "99", "-fa", "on", "-ub", $Ubatch, "-b", "2048", "-np", "1",
            "--ctx-checkpoints", "0", "--cache-ram", "0", "--no-warmup")
  $p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardError $log -RedirectStandardOutput "$log.out"
  for ($i = 0; $i -lt 600; $i++) {
    Start-Sleep -Milliseconds 1000
    try { if ((Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2).status -eq "ok") { return $p } } catch {}
    if ($p.HasExited) { break }
  }
  throw "llama-server ($engine) failed to start; see $log"
}

foreach ($engine in $Engines) {
  Write-Host "=== $Name / $engine ==="
  # ---------------------------------------------------------------- synthetic pp / tg
  if ($Tests -contains "pp") {
    if ($engine -eq "arcinfer") {
      foreach ($n in 512, 2048) {
        Cool
        Push-Location $arcDir
        $o = & $arc $Model --bench-pp $n --n 64 2>&1 | Out-String
        Pop-Location
        $runs = [regex]::Matches($o, "pp$n run \d: [\d.]+ ms\s+->\s+([\d.]+) tokens/s") | ForEach-Object { [double]$_.Groups[1].Value }
        $tg = [regex]::Match($o, "tg64: [\d.]+ ms\s+->\s+([\d.]+) tokens/s").Groups[1].Value
        Emit @{ engine = "arcinfer"; test = "pp$n"; tps = ($runs | Select-Object -Skip 1 | Measure-Object -Maximum).Maximum }
        if ($n -eq 512) { Emit @{ engine = "arcinfer"; test = "tg64"; tps = [double]$tg } }
      }
    } else {
      $exe = Llama $engine "llama-bench.exe"
      Cool
      $o = & $exe -m $Model -ngl 99 -fa 1 -ub $Ubatch -b 2048 -p "512,2048" -n 64 -r $Repeats -o json 2>$null | Out-String
      $j = $o | ConvertFrom-Json
      foreach ($r in $j) {
        $t = if ($r.n_prompt -gt 0) { "pp$($r.n_prompt)" } else { "tg$($r.n_gen)" }
        Emit @{ engine = $engine; test = $t; tps = [double]$r.avg_ts }
      }
    }
  }
  # ---------------------------------------------------------------- time to first token on real prompts
  if ($Tests -contains "ttft") {
    foreach ($pf in $PromptFiles) {
      $ids = (Get-Content $pf -Raw).Trim()
      $n = ($ids -split ",").Count
      $tag = [IO.Path]::GetFileNameWithoutExtension($pf) -replace "\.ids$", ""
      Cool
      if ($engine -eq "arcinfer") {
        Push-Location $arcDir
        $o = & $arc $Model --tokens-file $pf --n 32 --ctx ($n + 64) 2>&1 | Out-String
        Pop-Location
        $ttft = [regex]::Match($o, "time to first token ([\d.]+) ms").Groups[1].Value
        $g = [regex]::Match($o, "generation: (\d+) tokens in ([\d.]+) ms \(([\d.]+) t/s\)")
        Emit @{ engine = "arcinfer"; test = "ttft"; prompt = $tag; n_prompt = $n; ttft_s = [double]$ttft / 1000;
                pp_tps = $n / ([double]$ttft / 1000); tg_tps = [double]$g.Groups[3].Value; tg_n = [int]$g.Groups[1].Value }
      } else {
        $port = 8097
        $p = Start-LlamaServer $engine ($n + 128) $port
        try {
          $body = @{ prompt = @($ids -split "," | ForEach-Object { [int]$_ }); n_predict = 32; temperature = 0; cache_prompt = $false } | ConvertTo-Json -Depth 3 -Compress
          $t0 = Get-Date
          $res = Invoke-RestMethod "http://127.0.0.1:$port/completion" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 7200
          $wall = ((Get-Date) - $t0).TotalSeconds
          Emit @{ engine = $engine; test = "ttft"; prompt = $tag; n_prompt = $n; ttft_s = $res.timings.prompt_ms / 1000;
                  pp_tps = $res.timings.prompt_per_second; tg_tps = $res.timings.predicted_per_second; tg_n = $res.timings.predicted_n; wall_s = $wall }
        } finally {
          if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
          Start-Sleep -Seconds 3
        }
      }
    }
  }
  # ---------------------------------------------------------------- perplexity
  if ($Tests -contains "ppl" -and $Corpus) {
    Cool
    if ($engine -eq "arcinfer") {
      Push-Location $arcDir
      $o = & $arc $Model --perplexity $Corpus --ppl-ctx 512 --ppl-chunks $PplChunks 2>&1 | Out-String
      Pop-Location
      $m = [regex]::Match($o, "perplexity: ([\d.]+) over (\d+) chunks .* in ([\d.]+) s")
      Emit @{ engine = "arcinfer"; test = "ppl"; ppl = [double]$m.Groups[1].Value; chunks = [int]$m.Groups[2].Value; seconds = [double]$m.Groups[3].Value }
    } else {
      $exe = Llama $engine "llama-perplexity.exe"
      $t0 = Get-Date
      $o = & $exe -m $Model -f $Corpus -c 512 --chunks $PplChunks -ngl 99 -fa on -b 512 2>&1 | Out-String
      $m = [regex]::Match($o, "Final estimate: PPL = ([\d.]+)")
      Emit @{ engine = $engine; test = "ppl"; ppl = [double]$m.Groups[1].Value; chunks = $PplChunks; seconds = ((Get-Date) - $t0).TotalSeconds }
    }
  }
}
