[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $ReleaseRoot,
    [Parameter(Mandatory = $true)] [string] $Wheelhouse,
    [Parameter(Mandatory = $true)] [string] $Output,
    [Parameter(Mandatory = $true)] [string] $WorkRoot,
    [Parameter(Mandatory = $true)] [string] $Session,
    [Parameter(Mandatory = $true)] [string] $BaselineSummary,
    [Parameter(Mandatory = $true)] [string] $ReproSupport,
    [Parameter(Mandatory = $true)] [string] $ReproTruth,
    [Parameter(Mandatory = $true)] [string] $ReferenceReproReport,
    [Parameter(Mandatory = $true)] [string] $BaselineBundle,
    [Parameter(Mandatory = $true)] [string] $BaselineAuth
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
    "--wheelhouse", $Wheelhouse,
    "--output", $Output,
    "--work-root", $WorkRoot,
    "--session", $Session,
    "--baseline-summary", $BaselineSummary,
    "--repro-support", $ReproSupport,
    "--repro-truth", $ReproTruth,
    "--reference-repro-report", $ReferenceReproReport,
    "--baseline-bundle", $BaselineBundle,
    "--baseline-auth", $BaselineAuth
)
& $python -I @arguments
exit $LASTEXITCODE
