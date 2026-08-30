@echo off
setlocal EnableExtensions DisableDelayedExpansion
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Launch_DaguandanAssistant.ps1" %*
exit /b %ERRORLEVEL%
