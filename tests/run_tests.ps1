# arcinfer regression tests. Run from the project folder:
#   .\tests\run_tests.ps1 -Model4B <qwen3.5-4b.gguf> [-Model27B <qwen3.8-27b.gguf>] [-ModelGemma <gemma4-12b.gguf>]
#                          [-Model35B <qwen3.6-35b-a3b gsq2 gguf>] [-Data <dir with prompts/refs>]
# Checks: every quant type + kernel path against a CPU reference, tokenizer vs reference ids, chat template
# rendering vs the GGUF Jinja templates, next-token KL vs the f32 CPU reference, and an OpenAI API smoke test
# with streaming tool calls and prefix reuse.
param(
  [Parameter(Mandatory)] [string]$Model4B,
  [string]$Model27B = "",
  [string]$ModelGemma = "",
  [string]$Model35B = "",
  [string]$Data = "tests\data",
  [int]$Port = 8091
)
$ErrorActionPreference = "Stop"
$fail = 0
function Check($name, $ok, $detail = "") {
  $script:fail += [int](-not $ok)
  Write-Host ("[{0}] {1} {2}" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $name, $detail)
}

# 1. kernels: GEMM (native + JIT W8), GEMV for every weight type present
$t4 = .\bin\test_gemm.exe $Model4B 64 blk.0.ffn_gate.weight blk.2.ffn_gate.weight blk.0.attn_qkv.weight blk.0.ssm_out.weight token_embd.weight 2>&1 | Out-String
Check "kernels (4B types)" ($t4 -match "ALL OK")
if ($Model27B) {
  $t27 = .\bin\test_gemm.exe $Model27B 64 blk.0.attn_gate.weight blk.0.ffn_down.weight blk.0.ffn_gate.weight blk.0.ffn_up.weight blk.1.attn_gate.weight blk.1.attn_qkv.weight blk.1.ssm_out.weight blk.7.attn_q.weight blk.13.ffn_gate.weight blk.64.ffn_down.weight 2>&1 | Out-String
  Check "kernels (27B IQ types)" ($t27 -match "ALL OK")
}

# 1b. tool-call streaming: the streamed arguments only grow and end as the whole-response parse (no model needed)
if (Test-Path .\bin\tool_stream_test.exe) {
  $o = .\bin\tool_stream_test.exe 3000 2>&1 | Out-String
  Check "tool call streaming" ($o -match "ALL OK")
}

# 2. tokenizer vs reference ids (llama-tokenize --no-escape)
foreach ($p in Get-ChildItem $Data -Filter "*.prompt.txt" -ErrorAction SilentlyContinue) {
  $ids = $p.FullName -replace "\.prompt\.txt$", ".ids.txt"
  if (-not (Test-Path $ids)) { continue }
  $o = .\bin\tok_test.exe $Model4B $p.FullName $ids 2>&1 | Out-String
  Check "tokenizer $($p.Name)" ($o -match "IDENTICAL" -and $o -match "round-trip OK")
}

# 3. chat template rendering vs Jinja (needs python + jinja2)
$tq35 = Join-Path $Data "qwen35_template.jinja"; $tq38 = Join-Path $Data "qwen38_template.jinja"
if ((Test-Path $tq35) -and (Test-Path $tq38)) {
  $o = python tests\test_chat_render.py "$PWD\bin\chat_render.exe" $tq35 $tq38 2>&1 | Out-String
  Check "chat template (14 cases)" ($o -match "ALL IDENTICAL")
}

# 4. next-token distributions vs the f32 CPU reference, averaged over all 20 prompt positions (a single position
#    is too sensitive to individual int8 rounding flips). int8 K/V cache + int8 activations: ~0.002.
$ref = Join-Path $Data "ref_4b_capital_all.bin"
$ids = "248045,846,198,213077,173634,172069,23220,30,243381,189040,13,248046,198,248045,74455,198,248068,271,248069,271"
if (-not (Test-Path $ref)) { $env:REF_ALL = "1"; .\bin\ref_forward.exe $Model4B $ids $ref | Out-Null; $env:REF_ALL = $null }
$o = .\bin\run.exe $Model4B --tokens $ids --n 9 --kl-ref $ref 2>&1 | Out-String
$kl = [double]([regex]::Match($o, "KL\(ref \|\| arcinfer\) = ([\d.]+)").Groups[1].Value)
Check "KL vs f32 reference (mean of 20 positions)" ($kl -lt 0.005) ("KL = {0}" -f $kl)
Check "greedy continuation" ($o -match "greedy: 40 208756 5899 23220 36316 2972 195793 159034 248046")

# 4b. speculative decoding with the MTP block must reproduce the plain greedy continuation
$o = .\bin\run.exe $Model4B --tokens $ids --n 9 --mtp 2>&1 | Out-String
$acc = [regex]::Match($o, "MTP: (\d+) drafts, (\d+) accepted")
Check "MTP speculative decoding (greedy identical)" ($o -match "greedy: 40 208756 5899 23220 36316 2972 195793 159034 248046") `
  ("drafts {0}, accepted {1}" -f $acc.Groups[1].Value, $acc.Groups[2].Value)

# 4c. Gemma 4 12B: next-token distributions vs its f32 CPU reference (ref_forward), greedy = llama.cpp's, and the
#     assistant drafter must reproduce the greedy continuation. Gemma 4 is ~7x more sensitive to int8 rounding than
#     Qwen (an f32 reference with int8 activations alone gives KL ~0.009), hence the looser bound.
if ($ModelGemma) {
  $gids = "2,2,105,2364,107,155122,528,1156,23974,3217,506,7217,563,3730,236761,106,107,105,4368,107"
  $gref = Join-Path $Data "ref_gemma12b_google_chat_all.bin"  # (f32 reference of Google's QAT Q4_0 12B)
  if (-not (Test-Path $gref)) { $env:REF_ALL = "1"; .\bin\ref_forward.exe $ModelGemma $gids $gref | Out-Null; $env:REF_ALL = $null }
  $o = .\bin\run.exe $ModelGemma --tokens $gids --n 12 --kl-ref $gref 2>&1 | Out-String
  $kl = [double]([regex]::Match($o, "KL\(ref \|\| arcinfer\) = ([\d.]+)").Groups[1].Value)
  Check "Gemma 4 KL vs f32 reference (mean of 20 positions)" ($kl -lt 0.05) ("KL = {0}" -f $kl)
  $greedy = "greedy: 100 45518 107 101 818 7217 7412 3730 1547 529 496 20284"
  Check "Gemma 4 greedy = llama.cpp" ($o -match $greedy)
  $o = .\bin\run.exe $ModelGemma --tokens $gids --n 12 --mtp 2>&1 | Out-String
  Check "Gemma 4 assistant drafter (greedy identical)" ($o -match $greedy) ([regex]::Match($o, "MTP: .*").Value)
}
# 4d. Qwen3.6-35B-A3B GSQ (tools\convert_gsq.py): MoE path vs its f32 CPU reference
if ($Model35B) {
  $qref = Join-Path $Data "ref_35b_capital_all.bin"
  if (-not (Test-Path $qref)) { $env:REF_ALL = "1"; .\bin\ref_forward.exe $Model35B $ids $qref | Out-Null; $env:REF_ALL = $null }
  $o = .\bin\run.exe $Model35B --tokens $ids --n 4 --kl-ref $qref 2>&1 | Out-String
  $kl = [double]([regex]::Match($o, "KL\(ref \|\| arcinfer\) = ([\d.]+)").Groups[1].Value)
  Check "35B-A3B MoE KL vs f32 reference (mean of 20 positions)" ($kl -lt 0.005) ("KL = {0}" -f $kl)
}

# 5. server: streaming chat with a tool call, then a follow-up that must reuse the cached prefix
$srv = Start-Process -FilePath .\bin\server.exe -ArgumentList @("-m", $Model4B, "--port", $Port, "--ctx", "8192", "--alias", "test", "--no-prefix-cache") -PassThru -WindowStyle Hidden
try {
  for ($i = 0; $i -lt 120; $i++) { Start-Sleep -Milliseconds 1000; try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq "ok") { break } } catch {} }
  $o = python tests\test_tools.py "http://127.0.0.1:$Port" 2>&1 | Out-String
  Check "API tool call" ($o -match "turn 1: finish=tool_calls")
  Check "API prefix reuse" ($o -match "cache_n=([1-9]\d*)")
} finally {
  Stop-Process -Id $srv.Id -Force
}

if ($fail -eq 0) { Write-Host "ALL TESTS PASSED" } else { Write-Host "$fail TEST(S) FAILED"; exit 1 }
