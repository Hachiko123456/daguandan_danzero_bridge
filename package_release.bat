@echo off
setlocal
cd /d "%~dp0"
if "%~2"=="" (
  echo Usage: package_release.bat RELEASE_ROOT WHEELHOUSE_ROOT
  echo Both directories must be explicit and external to the source checkout.
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_release.ps1" -ReleaseRoot "%~1" -WheelhouseRoot "%~2"
if errorlevel 1 (
  echo.
  echo Packaging failed. Keep the error details above, then press any key to close.
  pause >nul
  exit /b 1
)
echo.
echo Packaging complete. Release root: %~1
pause
