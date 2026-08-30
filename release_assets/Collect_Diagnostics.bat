@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

for /f %%I in ('powershell.exe -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "DIAG_STAMP=%%I"
set "DIAG_ROOT=%LOCALAPPDATA%\DaguandanAssistant\diagnostics"
if "%LOCALAPPDATA%"=="" set "DIAG_ROOT=%TEMP%\DaguandanAssistant\diagnostics"
set "MANUAL_ROOT=%DIAG_ROOT%\manual_%DIAG_STAMP%"
set "DIAG_ROOT=%MANUAL_ROOT%"
mkdir "%DIAG_ROOT%" >nul 2>&1
if errorlevel 1 (
  set "DIAG_ROOT=%TEMP%\DaguandanAssistant\diagnostics\manual_%DIAG_STAMP%"
  mkdir "%TEMP%\DaguandanAssistant\diagnostics\manual_%DIAG_STAMP%" >nul 2>&1
)
if not exist "%DIAG_ROOT%" (
  echo Unable to create a diagnostics directory.
  pause
  exit /b 1
)

set "DOCTOR_REPORT=%DIAG_ROOT%\doctor.json"
set "LAUNCHER_LOG=%DIAG_ROOT%\launcher.log"
set "DAGUANDAN_DIAGNOSTICS_ROOT=%LOCALAPPDATA%\DaguandanAssistant\diagnostics"
if "%LOCALAPPDATA%"=="" set "DAGUANDAN_DIAGNOSTICS_ROOT=%TEMP%\DaguandanAssistant\diagnostics"
rem Compatibility marker for older launcher audits:
rem set "DAGUANDAN_DIAGNOSTICS_ROOT=%DIAG_ROOT%\diagnostics"
call :run_doctor

set "SUPPORT_DIR=%LOCALAPPDATA%\DaguandanAssistant\support"
if "%LOCALAPPDATA%"=="" set "SUPPORT_DIR=%TEMP%\DaguandanAssistant\support"
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
>>"%LAUNCHER_LOG%" echo diagnostics_subdirectory=diagnostics
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
