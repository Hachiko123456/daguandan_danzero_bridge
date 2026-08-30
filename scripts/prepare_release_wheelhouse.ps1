[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $WheelhouseRoot,

    [Parameter()]
    [string] $Python = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wheelhouse = [System.IO.Path]::GetFullPath($WheelhouseRoot)
$filesystemRoot = [System.IO.Path]::GetPathRoot($wheelhouse)
if ($wheelhouse.TrimEnd('\') -eq $filesystemRoot.TrimEnd('\')) {
    throw "WheelhouseRoot must not be a filesystem root."
}
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $projectRoot ".venv\Scripts\python.exe"
}
if (-not [System.IO.File]::Exists($Python)) {
    throw "Python executable not found: $Python"
}
[System.IO.Directory]::CreateDirectory($wheelhouse) | Out-Null
$attributes = [System.IO.File]::GetAttributes($wheelhouse)
if (($attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "WheelhouseRoot cannot be a reparse point."
}
$marker = Join-Path $wheelhouse ".daguandan-wheelhouse-root"
if (-not [System.IO.File]::Exists($marker)) {
    $existing = @(Get-ChildItem -LiteralPath $wheelhouse -Force)
    if ($existing.Count -gt 0) {
        throw "Existing non-empty wheelhouse lacks its ownership marker."
    }
    [System.IO.File]::WriteAllText(
        $marker,
        "guandan.wheelhouse-root/1`r`n",
        [System.Text.UTF8Encoding]::new($false)
    )
}
if ((Get-Content -LiteralPath $marker -Raw).Trim() -ne "guandan.wheelhouse-root/1") {
    throw "Wheelhouse ownership marker is invalid."
}

Write-Host "Downloading the explicitly pinned release wheels..." -ForegroundColor Cyan
& $Python -m pip download `
    --isolated `
    --ignore-requires-python `
    --only-binary=:all: `
    --no-deps `
    --dest $wheelhouse `
    -r (Join-Path $projectRoot "requirements-release.in")
if ($LASTEXITCODE -ne 0) {
    throw "Pinned wheel download failed with exit code $LASTEXITCODE."
}

$candidateRequirements = Join-Path $wheelhouse "requirements-release.candidate.lock"
$candidateWheelhouse = Join-Path $wheelhouse "wheelhouse.candidate.lock.json"
& $Python (Join-Path $projectRoot "scripts\generate_release_lock.py") `
    --wheelhouse $wheelhouse `
    --requirements-lock $candidateRequirements `
    --wheelhouse-lock $candidateWheelhouse
if ($LASTEXITCODE -ne 0) {
    throw "Candidate wheelhouse lock generation failed."
}

if ((Get-FileHash -LiteralPath $candidateRequirements -Algorithm SHA256).Hash -ne `
    (Get-FileHash -LiteralPath (Join-Path $projectRoot "requirements-release.lock") -Algorithm SHA256).Hash) {
    throw "Downloaded wheels do not match the committed requirements lock. Review candidate lock files."
}
if ((Get-FileHash -LiteralPath $candidateWheelhouse -Algorithm SHA256).Hash -ne `
    (Get-FileHash -LiteralPath (Join-Path $projectRoot "wheelhouse.lock.json") -Algorithm SHA256).Hash) {
    throw "Downloaded wheels do not match the committed wheelhouse lock. Review candidate lock files."
}

& $Python (Join-Path $projectRoot "scripts\verify_release_inputs.py") `
    --project-root $projectRoot `
    --wheelhouse $wheelhouse `
    --python $Python
if ($LASTEXITCODE -ne 0) {
    throw "Wheelhouse verification failed."
}
Write-Host "Verified offline wheelhouse: $wheelhouse" -ForegroundColor Green
