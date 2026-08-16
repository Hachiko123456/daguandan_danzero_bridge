@echo off
setlocal
cd /d "%~dp0"

echo Running the fixed FableDan benchmark: 200 games, fixed seed, no tribute, seat swapping.
echo This may take several minutes. Do not close this window.
DaguandanAssistant.exe --fabledan-fixed-benchmark
if errorlevel 1 (
  echo.
  echo Benchmark failed. Confirm that data\profiles\tencent_daguandan\models\best.npz exists.
  pause
  exit /b 1
)

echo.
powershell.exe -NoProfile -Command "$p = Get-ChildItem -LiteralPath 'data\profiles\tencent_daguandan\models\benchmarks' -Filter 'fabledan-standard-no-tribute-v1-*.json' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($null -eq $p) { throw 'Benchmark result file was not created.' }; Get-Content -LiteralPath $p.FullName -Raw"
echo.
echo Benchmark complete. The JSON result is under data\profiles\tencent_daguandan\models\benchmarks.
pause
