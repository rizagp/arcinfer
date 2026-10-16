@echo off
rem arcinfer setup (see setup.ps1): setup, setup -Models 27b,4b, setup -Tests, setup -NoPi
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
