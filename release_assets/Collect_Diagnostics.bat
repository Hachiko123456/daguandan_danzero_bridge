@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

for /f %%I in ('powershell.exe -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "DIAG_STAMP=%%I"
rem Match Python's explicit override precedence. Defaults stay beside this EXE.
set "DIAG_ROOT=%~dp0logs\diagnostics"
if not "%DAGUANDAN_DATA_ROOT%"=="" set "DIAG_ROOT=%DAGUANDAN_DATA_ROOT%\diagnostics"
if not "%DAGUANDAN_DIAGNOSTICS_DIR%"=="" set "DIAG_ROOT=%DAGUANDAN_DIAGNOSTICS_DIR%"
if not "%DAGUANDAN_DIAGNOSTICS_ROOT%"=="" set "DIAG_ROOT=%DAGUANDAN_DIAGNOSTICS_ROOT%"
rem Fail before mkdir if the configured root is relative, inside immutable
rem resources, or traverses a link. No AppData/TEMP fallback is permitted.
set "DIAG_APP_ROOT=%~dp0"
powershell.exe -NoProfile -Command "$ErrorActionPreference='Stop'; $p=$env:DIAG_ROOT; if ([IO.Path]::GetPathRoot($p).Length -lt 3) { exit 1 }; $p=[IO.Path]::GetFullPath($p).TrimEnd('\'); if ($p -eq [IO.Path]::GetPathRoot($p).TrimEnd('\')) { exit 1 }; $app=[IO.Path]::GetFullPath($env:DIAG_APP_ROOT).TrimEnd('\'); $logs=$app+'\logs'; if (($p.Equals($app,[StringComparison]::OrdinalIgnoreCase) -or $p.StartsWith($app+'\',[StringComparison]::OrdinalIgnoreCase)) -and -not ($p.Equals($logs,[StringComparison]::OrdinalIgnoreCase) -or $p.StartsWith($logs+'\',[StringComparison]::OrdinalIgnoreCase))) { exit 1 }; for ($q=$p; $q; $q=[IO.Path]::GetDirectoryName($q)) { if (Test-Path -LiteralPath $q) { if ((Get-Item -LiteralPath $q -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { exit 1 } } }" >nul 2>&1
if errorlevel 1 (
  echo Invalid or unsafe diagnostics directory: "%DIAG_ROOT%"
  echo Use an absolute external path or this application's logs directory. No fallback was used.
  pause
  exit /b 1
)
set "DAGUANDAN_DIAGNOSTICS_ROOT=%DIAG_ROOT%"
set "MANUAL_ROOT=%DIAG_ROOT%\manual_%DIAG_STAMP%"
mkdir "%MANUAL_ROOT%" >nul 2>&1
if not exist "%MANUAL_ROOT%\" (
  echo Unable to create diagnostics directory: "%MANUAL_ROOT%"
  echo Check permissions or explicitly set DAGUANDAN_DIAGNOSTICS_ROOT. No fallback was used.
  pause
  exit /b 1
)

set "DIAG_ROOT=%MANUAL_ROOT%"
set "DOCTOR_REPORT=%DIAG_ROOT%\doctor.json"
set "LAUNCHER_LOG=%DIAG_ROOT%\launcher.log"
call :run_doctor

set "SUPPORT_DIR=%DAGUANDAN_DIAGNOSTICS_ROOT%\support"
mkdir "%SUPPORT_DIR%" >nul 2>&1
set "SUPPORT_ZIP=%SUPPORT_DIR%\support_%DIAG_STAMP%.zip"
echo.
echo Exporting a sanitized support bundle without screenshots...
rem The resulting support ZIP is sanitized and image-free by default.
"%~dp0DaguandanAssistant.exe" --export-support "%SUPPORT_ZIP%"
set "EXPORT_EXIT=%ERRORLEVEL%"
if "%EXPORT_EXIT%"=="0" (
  echo Support bundle: %SUPPORT_ZIP%
) else (
  echo Support export failed with exit code %EXPORT_EXIT%.
)

echo.
echo Doctor finished with exit code %DOCTOR_EXIT%.
echo Report directory:
echo %DIAG_ROOT%
echo.
echo The default bundle contains no screenshots. To include sensitive images,
echo run the executable with --include-support-images after explicit approval.
pause
if not "%DOCTOR_EXIT%"=="0" exit /b %DOCTOR_EXIT%
exit /b %EXPORT_EXIT%

:run_doctor
set "EXE_PRESENT=false"
set "MANIFEST_PRESENT=false"
if exist "%~dp0DaguandanAssistant.exe" set "EXE_PRESENT=true"
if exist "%~dp0build_manifest.json" set "MANIFEST_PRESENT=true"
>"%LAUNCHER_LOG%" echo schema=guandan.diagnostics-launcher/1
>>"%LAUNCHER_LOG%" echo executable_present=%EXE_PRESENT%
>>"%LAUNCHER_LOG%" echo build_manifest_present=%MANIFEST_PRESENT%
>>"%LAUNCHER_LOG%" echo diagnostics_root=%DAGUANDAN_DIAGNOSTICS_ROOT%
>>"%LAUNCHER_LOG%" echo default_diagnostics_subdirectory=logs\diagnostics
>>"%LAUNCHER_LOG%" echo doctor_report=doctor.json
if not exist "%~dp0DaguandanAssistant.exe" (
  set "DOCTOR_EXIT=2"
  >>"%LAUNCHER_LOG%" echo doctor_exit_code=2
  exit /b 0
)
"%~dp0DaguandanAssistant.exe" --doctor --doctor-output "%DOCTOR_REPORT%"
set "DOCTOR_EXIT=%ERRORLEVEL%"
>>"%LAUNCHER_LOG%" echo doctor_exit_code=%DOCTOR_EXIT%
exit /b 0
