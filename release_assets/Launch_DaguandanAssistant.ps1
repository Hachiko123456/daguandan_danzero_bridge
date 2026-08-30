[CmdletBinding()]
param(
    [Parameter()]
    [string] $RuntimeRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    if (-not [string]::IsNullOrWhiteSpace($env:DAGUANDAN_DATA_ROOT)) {
        $RuntimeRoot = $env:DAGUANDAN_DATA_ROOT
    }
    else {
        $RuntimeRoot = Join-Path $env:LOCALAPPDATA "DaguandanAssistant"
    }
}
$installRoot = Join-Path ([System.IO.Path]::GetFullPath($RuntimeRoot)) "install"
$marker = Join-Path $installRoot ".daguandan-install-root.json"
$activePath = Join-Path $installRoot "active.json"
if (-not [System.IO.File]::Exists($marker) -or -not [System.IO.File]::Exists($activePath)) {
    throw "No verified active DaguandanAssistant release is installed."
}
$active = Get-Content -LiteralPath $activePath -Raw | ConvertFrom-Json
if ($active.schema -ne "guandan.active-release/1") {
    throw "The active release pointer is invalid."
}
$versionRoot = Join-Path (Join-Path $installRoot "versions") ([string] $active.version_directory)
$executable = [System.IO.Path]::GetFullPath((Join-Path $versionRoot ([string] $active.executable_relative)))
$prefix = [System.IO.Path]::GetFullPath($versionRoot).TrimEnd('\') + "\"
if (-not $executable.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "The active executable escaped its version directory."
}
if (-not [System.IO.File]::Exists($executable)) {
    throw "The active executable is missing."
}
$receiptPath = Join-Path $versionRoot "install_receipt.json"
if (-not [System.IO.File]::Exists($receiptPath)) {
    throw "The active release receipt is missing."
}
$receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
if (
    ([string] $receipt.release_id) -ne ([string] $active.release_id) -or
    ([string] $receipt.executable_relative) -ne ([string] $active.executable_relative)
) {
    throw "The active pointer disagrees with its immutable install receipt."
}
$actualHash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne ([string] $receipt.executable_sha256).ToLowerInvariant()) {
    throw "The active executable failed its install-receipt hash check."
}
$env:DAGUANDAN_DATA_ROOT = [System.IO.Path]::GetFullPath($RuntimeRoot)
& $executable @args
exit $LASTEXITCODE
