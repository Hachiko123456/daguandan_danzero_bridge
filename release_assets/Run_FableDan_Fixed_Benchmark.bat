@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run_FableDan_Fixed_Benchmark.ps1"
set "BENCHMARK_EXIT=%ERRORLEVEL%"
if not "%BENCHMARK_EXIT%"=="0" echo Benchmark failed with exit code %BENCHMARK_EXIT%.
exit /b %BENCHMARK_EXIT%
