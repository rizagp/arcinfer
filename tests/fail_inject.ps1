# Error paths of the server's generation steps (ARC_FAIL_PASS: the engine's N-th verification or batched pass throws
# after its layers wrote their state). A server with 2 slots and MTP:
#   reference   two requests at once (batched), no failure: their answers
#   batched     the same two requests, a batched pass fails: both get a 500, /health stays up, and the same two
#               requests afterwards give the reference answers (the slots started over)
#   streaming   one streamed request, its pass fails: the stream ends with an error event; afterwards the same
#               request (not streamed) gives the reference answer
# usage: .\tests\fail_inject.ps1 -Model <qwen 4b gguf with MTP> [-Port 8094]
# Prints PASS / FAIL lines; exit code 0 when all passed. Called by tests\run_tests.ps1 for the 4B.
param(
  [Parameter(Mandatory)] [string]$Model,
  [int]$Port = 8094
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "bin\serverz.exe"
$log = Join-Path $env:TEMP "arcinferzero-fail-inject.log"
$base = "http://127.0.0.1:$Port"
$failed = 0
function Check([string]$name, [bool]$ok, [string]$detail = "") {
  if ($ok) { Write-Host "[PASS] $name $detail" } else { Write-Host "[FAIL] $name $detail"; $script:failed++ }
}
function Start-Srv([string]$fail) {
  if ($fail) { $env:ARC_FAIL_PASS = $fail } else { Remove-Item Env:ARC_FAIL_PASS -ErrorAction SilentlyContinue }
  $p = Start-Process -FilePath $exe -ArgumentList @("-m", "`"$Model`"", "--port", "$Port", "--slots", "2", "--ctx", "4096", "--mtp", "--no-mmproj") `
    -PassThru -WindowStyle Hidden -RedirectStandardError $log -RedirectStandardOutput "$log.out" -WorkingDirectory $root
  Remove-Item Env:ARC_FAIL_PASS -ErrorAction SilentlyContinue
  for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Milliseconds 500
    try { if ((Invoke-WebRequest -UseBasicParsing "$base/health" -TimeoutSec 2).StatusCode -eq 200) { return $p } } catch {}
    if ($p.HasExited) { break }
  }
  throw "the server did not start (log: $log)"
}
function Stop-Srv($p) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit(10000) | Out-Null }; Start-Sleep -Seconds 2 }
function Body([string]$q, [bool]$stream = $false) {
  @{ max_tokens = 48; temperature = 0; stream = $stream; chat_template_kwargs = @{ enable_thinking = $false }
     messages = @(@{ role = "user"; content = $q }) } | ConvertTo-Json -Depth 5 -Compress
}
# the requests at once (each in its own job): status code and answer text
function Ask-All([string[]]$qs) {
  $jobs = foreach ($q in $qs) {
    Start-Job -ArgumentList "$base/v1/chat/completions", (Body $q) -ScriptBlock {
      param($url, $b)
      try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri $url -Method Post -Body $b -ContentType "application/json" -TimeoutSec 300
        $j = $r.Content | ConvertFrom-Json
        [pscustomobject]@{ code = [int]$r.StatusCode; text = [string]$j.choices[0].message.content }
      } catch {
        $c = 0
        if ($_.Exception.Response) { $c = [int]$_.Exception.Response.StatusCode }
        [pscustomobject]@{ code = $c; text = $_.ErrorDetails.Message }
      }
    }
  }
  $jobs | Wait-Job -Timeout 300 | Out-Null
  $out = foreach ($j in $jobs) { Receive-Job $j }
  $jobs | Remove-Job -Force
  return @($out)
}
function Healthy { try { (Invoke-WebRequest -UseBasicParsing "$base/health" -TimeoutSec 5).StatusCode -eq 200 } catch { $false } }

$qs = @("Name three large cities in West Java, one line each.", "Write a Python function that reverses a string.")
$p = $null
try {
  $p = Start-Srv ""
  $ref = @(Ask-All $qs)
  Check "reference answers" (($ref | Where-Object { $_.code -eq 200 -and $_.text }).Count -eq 2) ("lengths " + (($ref | ForEach-Object { $_.text.Length }) -join " / "))
  Stop-Srv $p

  # (pass 6: inside the batched generation of both; their prompts and first steps run before it)
  $p = Start-Srv "6"
  $r = @(Ask-All $qs)
  Check "batched pass fails: both requests get 500" (($r | Where-Object { $_.code -eq 500 }).Count -eq 2) ("codes " + (($r | ForEach-Object { $_.code }) -join " / "))
  Check "batched pass fails: the message names the failure" (($r | Where-Object { $_.text -match "injected failure" }).Count -eq 2)
  Check "/health after the failure" (Healthy)
  $r2 = @(Ask-All $qs)
  $same = $r2.Count -eq 2 -and $r2[0].code -eq 200 -and $r2[1].code -eq 200 -and $r2[0].text -eq $ref[0].text -and $r2[1].text -eq $ref[1].text
  Check "after the failure: the same answers as the reference" $same
  Stop-Srv $p

  $p = Start-Srv "3"
  $sb = Body $qs[0] $true
  $raw = try { (Invoke-WebRequest -UseBasicParsing -Uri "$base/v1/chat/completions" -Method Post -Body $sb -ContentType "application/json" -TimeoutSec 300).Content } catch { "EXCEPTION " + $_.Exception.Message }
  Check "streamed request: the stream ends with an error event" ($raw -match 'data: \{"error":\{"message":"[^"]*injected failure') (($raw -split "`n" | Where-Object { $_ -match "error" } | Select-Object -First 1))
  Check "/health after the streamed failure" (Healthy)
  $r3 = @(Ask-All @($qs[0]))  # (@(): Windows PowerShell 5.1 unrolls one result, and its objects have no .Count)
  Check "after the streamed failure: the reference answer" ($r3.Count -eq 1 -and $r3[0].code -eq 200 -and $r3[0].text -eq $ref[0].text) `
    $(if ($r3.Count -ne 1 -or $r3[0].code -ne 200 -or $r3[0].text -ne $ref[0].text) { "(code $($r3[0].code): '$($r3[0].text)' vs '$($ref[0].text)')" } else { "" })
} finally {
  Stop-Srv $p
}
if ($failed) { Write-Host "$failed check(s) failed (server log: $log)"; exit 1 }
Write-Host "error paths: all passed"
exit 0
