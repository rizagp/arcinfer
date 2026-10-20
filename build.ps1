# Builds one or more arcinfer targets with MSVC (Visual Studio Build Tools with the C++ workload).
# Usage: .\build.ps1 tools\clinfo.cpp [bench\gemm_bench.cpp ...]
# The compiler is found with vswhere (any Visual Studio / Build Tools installation that has vcvars64.bat);
# $env:ARC_VCVARS = the path of a vcvars64.bat overrides it.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Sources)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$vw = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vw) { $env:PATH = "$(Split-Path $vw);$env:PATH" }  # (vcvars64.bat calls vswhere too)
function Find-VcVars {
  if ($env:ARC_VCVARS -and (Test-Path $env:ARC_VCVARS)) { return $env:ARC_VCVARS }
  if (Test-Path $vw) {
    foreach ($p in @(& $vw -all -products * -prerelease -property installationPath)) {
      $v = Join-Path $p "VC\Auxiliary\Build\vcvars64.bat"
      if (Test-Path $v) { return $v }
    }
  }
  $v = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  if (Test-Path $v) { return $v }
  return $null
}
$vcvars = Find-VcVars
if (-not $vcvars) {
  throw "No Visual Studio C++ compiler found: install Visual Studio Build Tools with the 'Desktop development with C++' workload (README, step 3), or set `$env:ARC_VCVARS to a vcvars64.bat."
}
$out = Join-Path $root "bin"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($src in $Sources) {
  $full = Join-Path $root $src
  $name = [IO.Path]::GetFileNameWithoutExtension($full)
  # (_CRT_SECURE_NO_WARNINGS: no "getenv / fopen may be unsafe" deprecation notes)
  # (the f32 references and checks, ref_forward / bitcheck / tests / vision_test(z): /fp:precise; the rest /fp:fast;
  #  /utf-8: the sources are UTF-8, their string literals stay byte for byte)
  $fp = if ($name -match '^(ref_forward|bitcheck|test_|vision_test)') { '/fp:precise' } else { '/fp:fast' }
  $cmd = "`"$vcvars`" >nul && cl /nologo /utf-8 /std:c++20 /O2 /Oi /arch:AVX2 $fp /EHsc /MD /W3 /openmp:llvm /D_CRT_SECURE_NO_WARNINGS /I`"$root\third_party`" " +
         "/Fe`"$out\$name.exe`" /Fo`"$out\\`" `"$full`""
  cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "build failed: $src" }
}
