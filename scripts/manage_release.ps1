[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("install", "activate", "rollback", "status", "register-legacy-baseline", "create-baseline-auth")]
    [string] $Command,

    [Parameter()]
    [string] $RuntimeRoot = "",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not [System.IO.File]::Exists($python)) {
    throw "Source release management requires .venv\Scripts\python.exe."
}
$commandArguments = @((Join-Path $projectRoot "scripts\manage_release.py"))
if (-not [string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $commandArguments += @("--runtime-root", $RuntimeRoot)
}
$commandArguments += $Command
$commandArguments += $Arguments
& $python @commandArguments
exit $LASTEXITCODE
