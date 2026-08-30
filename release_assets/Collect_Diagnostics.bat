@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

for /f %%I in ('powershell.exe -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "DIAG_STAMP=%%I"
set "DIAG_ROOT=%LOCALAPPDATA%\DaguandanAssistant\diagnostics\manual_%DIAG_STAMP%"
if "%LOCALAPPDATA%"=="" set "DIAG_ROOT=%TEMP%\DaguandanAssistant\diagnostics\manual_%DIAG_STAMP%"
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
set "DAGUANDAN_DIAGNOSTICS_ROOT=%DIAG_ROOT%\diagnostics"
mkdir "%DAGUANDAN_DIAGNOSTICS_ROOT%" >nul 2>&1
if not exist "%DAGUANDAN_DIAGNOSTICS_ROOT%" (
  echo Unable to create the startup diagnostics subdirectory.
  pause
  exit /b 1
)
call :run_doctor

echo.
echo Doctor finished with exit code %DOCTOR_EXIT%.
echo Report directory:
echo %DIAG_ROOT%
echo.
echo This launcher does not copy raw logs or create a support ZIP.
echo Use the application's approved support export when it becomes available.
pause
exit /b %DOCTOR_EXIT%

:run_doctor
set "EXE_PRESENT=false"
set "MANIFEST_PRESENT=false"
if exist "%~dp0DaguandanAssistant.exe" set "EXE_PRESENT=true"
if exist "%~dp0build_manifest.json" set "MANIFEST_PRESENT=true"
>"%LAUNCHER_LOG%" echo schema=guandan.diagnostics-launcher/1
>>"%LAUNCHER_LOG%" echo executable_present=%EXE_PRESENT%
>>"%LAUNCHER_LOG%" echo build_manifest_present=%MANIFEST_PRESENT%
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
