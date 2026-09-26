@echo off
setlocal EnableExtensions DisableDelayedExpansion
if /i "%~1"=="--help" goto help
if /i "%~1"=="/?" goto help
if not "%~2"=="" goto usage_error
set "IMAGE_OPTION="
if /i "%~1"=="--problem-no-images" set "IMAGE_OPTION=--problem-no-images"
if not "%~1"=="" if not defined IMAGE_OPTION goto usage_error

rem Display only: the EXE validates/resolves these unchanged environment values.
rem Never create a fresh doctor run or replace the selected diagnostics root.
set "DIAG_ROOT=%~dp0diagnostics"
if not "%DAGUANDAN_DATA_ROOT%"=="" set "DIAG_ROOT=%DAGUANDAN_DATA_ROOT%\diagnostics"
if not "%DAGUANDAN_DIAGNOSTICS_DIR%"=="" set "DIAG_ROOT=%DAGUANDAN_DIAGNOSTICS_DIR%"
if not "%DAGUANDAN_DIAGNOSTICS_ROOT%"=="" set "DIAG_ROOT=%DAGUANDAN_DIAGNOSTICS_ROOT%"
if not exist "%~dp0DaguandanAssistant.exe" goto missing_exe

echo Export an existing problem case, configuration, run and session evidence.
echo Nothing is uploaded. Screenshots can contain private game/window content.
if defined IMAGE_OPTION goto export
choice /C YN /N /M "Include screenshots in the local ZIP? [Y/N] "
if errorlevel 255 goto cancelled
if errorlevel 2 set "IMAGE_OPTION=--problem-no-images"
if not errorlevel 1 goto cancelled

:export
echo.
echo Exporting existing evidence without starting the GUI or running doctor...
"%~dp0DaguandanAssistant.exe" --export-problem %IMAGE_OPTION%
set "EXPORT_EXIT=%ERRORLEVEL%"
if not "%EXPORT_EXIT%"=="0" goto export_failed
echo.
echo Export completed. Problem ZIP folder:
echo "%DIAG_ROOT%\exports"
echo Filename: DaguandanAssistant_problem_*.zip
echo Check the ZIP inventory for missing, excluded or budget-limited evidence.
echo Review it before sharing. Nothing is uploaded.
pause
exit /b 0

:missing_exe
set "EXPORT_EXIT=2"
echo DaguandanAssistant.exe is missing beside this script.
goto recovery

:export_failed
echo Export failed or the EXE could not run. Exit code: %EXPORT_EXIT%

:recovery
echo No problem ZIP was created by this script.
echo If the EXE is missing or cannot run, manually copy the existing diagnostics folder:
echo "%DIAG_ROOT%"
echo Keep any old logs folder too; do not move or delete the original evidence.
echo If the path is invalid or unwritable, fix permissions or the explicit override.
echo No AppData/TEMP fallback is used. Copied evidence may contain private content.
echo Nothing is uploaded.
pause
exit /b %EXPORT_EXIT%

:cancelled
echo Export cancelled. No problem ZIP was created by this script.
exit /b 2

:usage_error
echo Unknown option. Use Collect_Diagnostics.bat --help for help.
exit /b 2

:help
echo Double-click Collect_Diagnostics.bat to export an existing problem case.
echo Answer Y to include screenshots, or N to omit them for privacy.
echo Optional: Collect_Diagnostics.bat --problem-no-images
echo Help: Collect_Diagnostics.bat --help
echo The local EXE uses headless --export-problem even if the GUI cannot start.
echo Default output: diagnostics\exports\DaguandanAssistant_problem_*.zip
echo Explicit diagnostics/DATA_ROOT overrides still take precedence.
echo Missing evidence and budget exclusions are recorded, not invented.
echo Nothing is uploaded. Review the ZIP before sharing.
exit /b 0
