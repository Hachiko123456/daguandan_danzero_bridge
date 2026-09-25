@echo off
setlocal
cd /d "%~dp0"

if /I "%~1"=="--help" goto :usage
if /I "%~1"=="-h" goto :usage

set "ReleaseRoot=%~1"
set "WheelhouseRoot=%~2"
set "OverwriteFlag="

if "%~1"=="" (
  set "ReleaseRoot=%~dp0release\current"
  set "OverwriteFlag=-OverwriteExisting"
)

if "%~2"=="" (
  set "WheelhouseRoot=%LOCALAPPDATA%\Daguandan\wheelhouse"
)

if defined OverwriteFlag (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_release.ps1" -ReleaseRoot "%ReleaseRoot%" -WheelhouseRoot "%WheelhouseRoot%" -OverwriteExisting
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_release.ps1" -ReleaseRoot "%ReleaseRoot%" -WheelhouseRoot "%WheelhouseRoot%"
)
if errorlevel 1 (
  echo.
  echo Packaging failed. Keep the error details above, then press any key to close.
  pause >nul
  exit /b 1
)

echo.
echo Packaging complete. Release root: %ReleaseRoot%
pause
exit /b 0

:usage
echo Usage: package_release.bat [RELEASE_ROOT [WHEELHOUSE_ROOT]]
echo.
echo Defaults:
echo   RELEASE_ROOT   %~dp0release\current
echo   WHEELHOUSE_ROOT %%LOCALAPPDATA%%\Daguandan\wheelhouse
echo.
echo When RELEASE_ROOT is omitted, the default current release directory is rebuilt
echo automatically by passing -OverwriteExisting to scripts\package_release.ps1.
echo When RELEASE_ROOT is provided explicitly, existing output is not overwritten by default.
echo.
echo Examples:
echo   package_release.bat
echo   package_release.bat C:\DaguandanBuilds\candidate-20260831-001
echo   package_release.bat C:\DaguandanBuilds\candidate-20260831-001 C:\DaguandanBuildInputs\wheelhouse-cp312-win_amd64
exit /b 0
