[CmdletBinding()]
param(
    [Parameter()]
    [string] $ExecutablePath = "",

    [Parameter()]
    [string] $OutputPath = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($ExecutablePath)) {
    $ExecutablePath = Join-Path $PSScriptRoot "DaguandanAssistant.exe"
}
$executable = [System.IO.Path]::GetFullPath($ExecutablePath)
if (-not [System.IO.File]::Exists($executable)) {
    Write-Error "DaguandanAssistant executable is missing: $executable"
    exit 2
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $runtimeRoot = $env:DAGUANDAN_DATA_ROOT
    if ([string]::IsNullOrWhiteSpace($runtimeRoot)) {
        if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
            Write-Error "LOCALAPPDATA is unavailable."
            exit 2
        }
        $runtimeRoot = Join-Path $env:LOCALAPPDATA "DaguandanAssistant"
    }
    $benchmarkRoot = Join-Path ([System.IO.Path]::GetFullPath($runtimeRoot)) "benchmarks"
    [System.IO.Directory]::CreateDirectory($benchmarkRoot) | Out-Null
    $name = "fabledan-standard-no-tribute-v1-{0}-{1}.json" -f (
        [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
    ), ([Guid]::NewGuid().ToString("N").Substring(0, 8))
    $OutputPath = Join-Path $benchmarkRoot $name
}
$output = [System.IO.Path]::GetFullPath($OutputPath)
if ([System.IO.File]::Exists($output) -or [System.IO.Directory]::Exists($output)) {
    Write-Error "Benchmark output must be a new file: $output"
    exit 2
}
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($output)) | Out-Null

Write-Output "Running fixed FableDan benchmark (200 games, fixed seed, no tribute, seat swapping)."
$cliOutput = @(& $executable --fabledan-fixed-benchmark --benchmark-output $output 2>&1)
$exitCode = $LASTEXITCODE
$cliOutput | ForEach-Object { Write-Output $_ }
if ($exitCode -ne 0) {
    Write-Error "Benchmark executable failed with exit code $exitCode."
    exit $exitCode
}
if (-not [System.IO.File]::Exists($output)) {
    Write-Error "Benchmark reported success but the exact output file is missing."
    exit 3
}
try {
    $result = Get-Content -LiteralPath $output -Raw -Encoding UTF8 | ConvertFrom-Json
}
catch {
    Write-Error "Benchmark output is not valid UTF-8 JSON."
    exit 3
}
if ($result.schema -ne "fabledan.fixed-benchmark/1") {
    Write-Error "Benchmark output schema is invalid."
    exit 3
}
Write-Output "Verified benchmark output: $output"
Get-Content -LiteralPath $output -Raw -Encoding UTF8 | Write-Output
exit 0
