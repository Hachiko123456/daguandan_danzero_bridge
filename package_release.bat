@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

if /I "%~1"=="--help" goto :usage
if /I "%~1"=="-h" goto :usage

set "ReleaseRoot=%~1"
set "WheelhouseRoot=%~2"
set "OverwriteFlag="
set "AllowDirtyFlag="
set "DefaultWheelhouse=0"
set "NeedPrepareWheelhouse=0"

if "%~1"=="" (
  set "ReleaseRoot=%~dp0release\current"
  set "OverwriteFlag=-OverwriteExisting"
  set "AllowDirtyFlag=-AllowDirtyDevelopmentBuild"
)

if "%~2"=="" (
  set "WheelhouseRoot=%LOCALAPPDATA%\Daguandan\wheelhouse"
  set "DefaultWheelhouse=1"
)

for %%I in ("%ReleaseRoot%") do set "ReleaseOutputDirectory=%%~fI"

echo [1/3] 检查 wheelhouse
echo Wheelhouse: "%WheelhouseRoot%"
if "%DefaultWheelhouse%"=="1" (
  if not exist "%WheelhouseRoot%\" (
    echo Default wheelhouse does not exist; dependencies will be prepared.
    set "NeedPrepareWheelhouse=1"
  ) else if not exist "%WheelhouseRoot%\.daguandan-wheelhouse-root" (
    echo Default wheelhouse is missing the ownership marker; dependencies will be prepared.
    set "NeedPrepareWheelhouse=1"
  ) else if not exist "%WheelhouseRoot%\wheelhouse.candidate.lock.json" (
    echo Default wheelhouse is missing the completion lock file; dependencies will be prepared.
    set "NeedPrepareWheelhouse=1"
  ) else (
    echo Default wheelhouse marker and completion lock are present.
  )
) else (
  echo Explicit wheelhouse was provided; it will not be replaced or prepared by this launcher.
)

echo.
echo [2/3] 准备依赖
if "%NeedPrepareWheelhouse%"=="1" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\prepare_release_wheelhouse.ps1" -WheelhouseRoot "%WheelhouseRoot%"
  set "StageExitCode=!ERRORLEVEL!"
  if not "!StageExitCode!"=="0" (
    echo.
    echo [2/3] 准备依赖 failed with exit code !StageExitCode!.
    echo Packaging aborted before building the release package.
    echo Press any key to close.
    pause >nul
    exit /b !StageExitCode!
  )
) else (
  echo Dependencies are already prepared or an explicit wheelhouse was supplied; skipping prepare step.
)

echo.
echo [3/3] 构建发布包
echo Release output directory: "%ReleaseOutputDirectory%"
echo Verified build caches are reused automatically; the previous release is kept until success.
if defined OverwriteFlag (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_release.ps1" -ReleaseRoot "%ReleaseRoot%" -WheelhouseRoot "%WheelhouseRoot%" -OverwriteExisting %AllowDirtyFlag%
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_release.ps1" -ReleaseRoot "%ReleaseRoot%" -WheelhouseRoot "%WheelhouseRoot%"
)
set "StageExitCode=!ERRORLEVEL!"
if not "!StageExitCode!"=="0" (
  echo.
  echo [3/3] 构建发布包 failed with exit code !StageExitCode!.
  echo Keep the error details above, then press any key to close.
  pause >nul
  exit /b !StageExitCode!
)

echo.
echo Packaging complete.
echo Release output directory: "%ReleaseOutputDirectory%"
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
echo automatically by passing -OverwriteExisting and -AllowDirtyDevelopmentBuild to
echo scripts\package_release.ps1. This default development build tolerates untracked
echo diagnostic materials in the working tree; explicit ReleaseRoot builds remain strict.
echo When WHEELHOUSE_ROOT is omitted, this launcher checks the default wheelhouse
echo and runs scripts\prepare_release_wheelhouse.ps1 before packaging if the
echo ownership marker or completion lock file is missing.
echo When RELEASE_ROOT or WHEELHOUSE_ROOT is provided explicitly, the launcher
echo preserves that explicit directory and does not overwrite or replace it.
echo.
echo Stages:
echo   [1/3] 检查 wheelhouse
echo   [2/3] 准备依赖
echo   [3/3] 构建发布包
echo.
echo Examples:
echo   package_release.bat
echo   package_release.bat C:\DaguandanBuilds\candidate-20260831-001
echo   package_release.bat C:\DaguandanBuilds\candidate-20260831-001 C:\DaguandanBuildInputs\wheelhouse-cp312-win_amd64
exit /b 0
