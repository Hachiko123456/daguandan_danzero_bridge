@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_release.ps1"
if errorlevel 1 (
  echo.
  echo Packaging failed. Keep the error details above, then press any key to close.
  pause >nul
  exit /b 1
)
echo.
echo Packaging complete. Release: release\dist\DaguandanAssistant
echo Archive: release\DaguandanAssistant.zip
pause
