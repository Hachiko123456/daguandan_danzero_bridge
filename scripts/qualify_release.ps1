[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $ReleaseRoot,
    [Parameter(Mandatory = $true)] [string] $Wheelhouse,
    [Parameter()] [string] $Output = "",
    [Parameter()] [string] $WorkRoot = "",
    [Parameter()] [string] $Session = "",
    [Parameter()] [string] $BaselineSummary = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not [System.IO.File]::Exists($python)) {
    throw "Qualification requires the repository .venv Python launcher."
}
$arguments = @(
    (Join-Path $projectRoot "scripts\qualify_release.py"),
    "--release-root", $ReleaseRoot,
    "--wheelhouse", $Wheelhouse
)
if (-not [string]::IsNullOrWhiteSpace($Output)) { $arguments += @("--output", $Output) }
if (-not [string]::IsNullOrWhiteSpace($WorkRoot)) { $arguments += @("--work-root", $WorkRoot) }
if (-not [string]::IsNullOrWhiteSpace($Session)) { $arguments += @("--session", $Session) }
if (-not [string]::IsNullOrWhiteSpace($BaselineSummary)) { $arguments += @("--baseline-summary", $BaselineSummary) }
& $python -I @arguments
exit $LASTEXITCODE
