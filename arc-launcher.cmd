@echo off
rem arcinfer launcher window (scripts\arc-launcher.ps1): models, options, server on / off.
where pwsh >nul 2>nul
if %errorlevel%==0 (
  start "" pwsh -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -STA -File "%~dp0scripts\arc-launcher.ps1"
) else (
  start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -STA -File "%~dp0scripts\arc-launcher.ps1"
)
