# The data folder %LOCALAPPDATA%\arcinferzero (kernel cache, model packages, caches, logs, settings), as the exes find
# it (src/app_dir.h). Until 2026-10-24 the project was called arcinfer and the folder %LOCALAPPDATA%\arcinfer: when only
# that one exists it is renamed once (the same drive: instant). If the rename fails (a file held open by a running
# older server) the old folder is used as it is. Dot-sourced by setup.ps1, the scripts and tests\run_tests.ps1.
function Get-AppDir {
  $now = Join-Path $env:LOCALAPPDATA "arcinferzero"
  $old = Join-Path $env:LOCALAPPDATA "arcinfer"
  if (-not (Test-Path -LiteralPath $now) -and (Test-Path -LiteralPath $old -PathType Container)) {
    try { Rename-Item -LiteralPath $old -NewName "arcinferzero" -ErrorAction Stop } catch { return $old }
  }
  return $now
}
