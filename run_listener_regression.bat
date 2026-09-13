@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "scripts\run_listener_regression.py" %*
) else (
  py -3.12 "scripts\run_listener_regression.py" %*
)
exit /b %errorlevel%