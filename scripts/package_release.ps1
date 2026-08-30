[CmdletBinding()]
param(
    [Parameter()]
    [string] $ReleaseRoot = "",

    [Parameter()]
    [switch] $AllowDirtyDevelopmentBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-ProjectPython {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $Arguments
    )

    if (Test-Path -LiteralPath $script:venvPython) {
        & $script:venvPython @Arguments
    }
    else {
        & py -3.12 @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit code $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

function Get-ExistingPathAttributes {
    param(
        [Parameter(Mandatory = $true)]
        [string] $LiteralPath
    )

    try {
        return [System.IO.File]::GetAttributes($LiteralPath)
    }
    catch [System.IO.FileNotFoundException] {
        return $null
    }
    catch [System.IO.DirectoryNotFoundException] {
        return $null
    }
}

function Assert-NoReparsePathChain {
    param(
        [Parameter(Mandatory = $true)]
        [string] $LiteralPath
    )

    $current = [System.IO.Path]::GetFullPath($LiteralPath)
    while (-not [string]::IsNullOrWhiteSpace($current)) {
        $attributes = Get-ExistingPathAttributes -LiteralPath $current
        if (
            $null -ne $attributes -and
            ($attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
        ) {
            throw "Packaging path traverses a symlink, junction, or reparse point: $current"
        }
        $parent = [System.IO.Directory]::GetParent($current)
        if ($null -eq $parent -or $parent.FullName -eq $current) {
            break
        }
        $current = $parent.FullName
    }
}

function Assert-NoReparseTree {
    param(
        [Parameter(Mandatory = $true)]
        [string] $LiteralPath
    )

    Assert-NoReparsePathChain -LiteralPath $LiteralPath
    if ([System.IO.File]::Exists($LiteralPath)) {
        throw "Managed release directory target is a file: $LiteralPath"
    }
    if (-not [System.IO.Directory]::Exists($LiteralPath)) {
        return
    }

    $pending = [System.Collections.Generic.Stack[System.IO.DirectoryInfo]]::new()
    $pending.Push([System.IO.DirectoryInfo]::new($LiteralPath))
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        if (($directory.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Managed release directory is a reparse point: $($directory.FullName)"
        }
        foreach ($entry in $directory.EnumerateFileSystemInfos()) {
            if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Managed release tree contains a symlink, junction, or reparse point: $($entry.FullName)"
            }
            if (($entry.Attributes -band [System.IO.FileAttributes]::Directory) -ne 0) {
                $pending.Push([System.IO.DirectoryInfo] $entry)
            }
        }
    }
}

function Resolve-ManagedChildPath {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Root,

        [Parameter(Mandatory = $true)]
        [string] $Child
    )

    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    $resolvedChild = [System.IO.Path]::GetFullPath($Child)
    $prefix = "$resolvedRoot\"
    if (-not $resolvedChild.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Managed release path must stay below ReleaseRoot: $resolvedChild"
    }
    return $resolvedChild
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$appName = "DaguandanAssistant"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$profileSource = Join-Path $projectRoot "data\profiles\tencent_daguandan"
$modelSource = Join-Path $profileSource "models\best.npz"
$danzeroWeightsSource = Join-Path $projectRoot "src\daguandan_bridge\danzero\_vendor\guandan_rlcard\baselines\danzero\q_network.ckpt"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required to verify that a release build uses a clean source tree."
}
$gitStatus = @(& git -C $projectRoot status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw "Git could not inspect the source tree before packaging."
}
$sourceTreeDirty = $gitStatus.Count -gt 0
if ($sourceTreeDirty -and -not $AllowDirtyDevelopmentBuild) {
    throw "Source tree is dirty. Commit or stash changes before a release build, or pass -AllowDirtyDevelopmentBuild for an explicitly non-release development build."
}
if ($sourceTreeDirty) {
    Write-Warning "Creating an explicitly allowed dirty development build; build_manifest.json will record source.dirty=true."
}

if ([string]::IsNullOrWhiteSpace($ReleaseRoot)) {
    $releaseRoot = Join-Path $projectRoot "artifacts\release"
}
else {
    $releaseRoot = [System.IO.Path]::GetFullPath($ReleaseRoot)
}
$filesystemRoot = [System.IO.Path]::GetPathRoot($releaseRoot)
if (
    [string]::IsNullOrWhiteSpace($filesystemRoot) -or
    $releaseRoot.TrimEnd('\') -eq $filesystemRoot.TrimEnd('\') -or
    $releaseRoot.TrimEnd('\') -eq $projectRoot.TrimEnd('\')
) {
    throw "ReleaseRoot must name a dedicated directory, not a filesystem or project root: $releaseRoot"
}
$ownershipMarkerName = ".daguandan-release-root"
$ownershipMarkerValue = "guandan.package-release-root/1"
Assert-NoReparsePathChain -LiteralPath $releaseRoot
if ([System.IO.File]::Exists($releaseRoot)) {
    throw "ReleaseRoot is a file, not a dedicated directory: $releaseRoot"
}
[System.IO.Directory]::CreateDirectory($releaseRoot) | Out-Null
Assert-NoReparsePathChain -LiteralPath $releaseRoot
$ownershipMarkerPath = Join-Path $releaseRoot $ownershipMarkerName
$releaseEntries = @(Get-ChildItem -LiteralPath $releaseRoot -Force)
if ($releaseEntries.Count -eq 0) {
    [System.IO.File]::WriteAllText(
        $ownershipMarkerPath,
        "$ownershipMarkerValue`r`n",
        [System.Text.UTF8Encoding]::new($false)
    )
}
elseif (-not [System.IO.File]::Exists($ownershipMarkerPath)) {
    throw "Existing non-empty ReleaseRoot is not owned by this packaging script (missing $ownershipMarkerName): $releaseRoot"
}
Assert-NoReparsePathChain -LiteralPath $ownershipMarkerPath
if ((Get-Content -LiteralPath $ownershipMarkerPath -Raw).Trim() -ne $ownershipMarkerValue) {
    throw "ReleaseRoot ownership marker is invalid: $ownershipMarkerPath"
}

$distPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "dist")
$workPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "build")
$specPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "spec")
$payloadPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "payload")
$bundlePath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $distPath $appName)
$archivePath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "DaguandanAssistant.zip")
$archiveChecksumPath = Resolve-ManagedChildPath -Root $releaseRoot -Child "$archivePath.sha256"
$releaseRecordPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "DaguandanAssistant.release.json")
$buildManifestPath = Join-Path $bundlePath "build_manifest.json"
$buildManifestGenerator = Join-Path $projectRoot "scripts\generate_build_manifest.py"

$managedDirectories = @($distPath, $workPath, $specPath, $payloadPath)
foreach ($path in $managedDirectories) {
    Assert-NoReparseTree -LiteralPath $path
}
$managedFiles = @($archivePath, $archiveChecksumPath, $releaseRecordPath)
foreach ($path in $managedFiles) {
    Assert-NoReparsePathChain -LiteralPath $path
    if ([System.IO.Directory]::Exists($path)) {
        throw "Managed release file target is a directory: $path"
    }
}

if (-not (Test-Path -LiteralPath $venvPython) -and -not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Neither .venv\\Scripts\\python.exe nor Python Launcher (py) was found. Install Python 3.12 first."
}

$requiredFiles = @(
    (Join-Path $projectRoot "run.py"),
    (Join-Path $projectRoot "app.ico"),
    (Join-Path $profileSource "profile.json"),
    (Join-Path $profileSource "regions_config.json"),
    (Join-Path $profileSource "templates_config.json"),
    (Join-Path $profileSource "templates"),
    $modelSource,
    $danzeroWeightsSource,
    $buildManifestGenerator,
    (Join-Path $projectRoot "release_assets\MODEL_REPLACEMENT.txt"),
    (Join-Path $projectRoot "release_assets\Run_FableDan_Fixed_Benchmark.bat"),
    (Join-Path $projectRoot "release_assets\Collect_Diagnostics.bat")
)
foreach ($requiredFile in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $requiredFile)) {
        throw "Required release file is missing: $requiredFile"
    }
}

Write-Host "[1/5] Installing or updating PyInstaller..." -ForegroundColor Cyan
Invoke-ProjectPython -m pip install --upgrade "PyInstaller==6.21.0"

Write-Host "[2/5] Preparing the minimum runtime data..." -ForegroundColor Cyan
foreach ($path in $managedDirectories) {
    # Recheck immediately before each recursive deletion to narrow the race
    # window after the all-target preflight above.
    Assert-NoReparseTree -LiteralPath $path
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}
foreach ($path in $managedFiles) {
    Assert-NoReparsePathChain -LiteralPath $path
    if ([System.IO.Directory]::Exists($path)) {
        throw "Managed release file target became a directory: $path"
    }
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force
    }
}

$payloadProfile = Join-Path $payloadPath "data\profiles\tencent_daguandan"
New-Item -ItemType Directory -Force -Path "$payloadProfile\models\danzero" | Out-Null
Copy-Item -LiteralPath (Join-Path $profileSource "profile.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "regions_config.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "templates_config.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "templates") -Destination $payloadProfile -Recurse
Copy-Item -LiteralPath $modelSource -Destination (Join-Path $payloadProfile "models\best.npz")
Copy-Item -LiteralPath $danzeroWeightsSource -Destination (Join-Path $payloadProfile "models\danzero\q_network.ckpt")

Write-Host "[3/5] Building the portable Windows release..." -ForegroundColor Cyan
$pyinstallerArguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--noconsole",
    "--noupx",
    "--name", $appName,
    "--icon", (Join-Path $projectRoot "app.ico"),
    "--paths", (Join-Path $projectRoot "src"),
    "--distpath", $distPath,
    "--workpath", $workPath,
    "--specpath", $specPath,
    "--collect-submodules", "daguandan_bridge",
    "--collect-submodules", "rlcard",
    "--collect-data", "rlcard",
    "--collect-data", "qfluentwidgets",
    "--collect-data", "qframelesswindow",
    "--hidden-import", "torch",
    "--hidden-import", "cv2",
    "--hidden-import", "mss",
    "--hidden-import", "win32gui",
    "--hidden-import", "win32ui",
    "--hidden-import", "pythoncom",
    "--hidden-import", "pywintypes",
    (Join-Path $projectRoot "run.py")
)
Invoke-ProjectPython @pyinstallerArguments

if (-not (Test-Path -LiteralPath (Join-Path $bundlePath "$appName.exe"))) {
    throw "PyInstaller did not produce $appName.exe."
}

# PyInstaller can discover ICU from this workstation's Poppler runtime. Those
# ICU 78 binaries export version-suffixed symbols and shadow Windows' ICU,
# while Qt6Core requires the system ICU ABI. Keep them out of the portable
# bundle so Qt resolves the compatible Windows copies instead of failing while
# importing PySide6.QtGui.
$incompatibleIcuFiles = @(
    (Join-Path $bundlePath "_internal\icuuc.dll"),
    (Join-Path $bundlePath "_internal\icudt78.dll")
)
foreach ($file in $incompatibleIcuFiles) {
    if (Test-Path -LiteralPath $file) {
        Remove-Item -LiteralPath $file -Force
    }
    if (Test-Path -LiteralPath $file) {
        throw "Incompatible ICU dependency remains in release bundle: $file"
    }
}

# Fail closed if a differently named copy of the same ICU 78 runtime entered
# through PATH or another build-machine dependency.
$remainingIncompatibleIcu = @(
    Get-ChildItem -LiteralPath $bundlePath -Filter "icu*.dll" -Recurse -File | Where-Object {
        $name = $_.Name.ToLowerInvariant()
        $originalName = ([string] $_.VersionInfo.OriginalFilename).ToLowerInvariant()
        $name -eq "icuuc.dll" -or
        $name -match '^icu.*78\.dll$' -or
        $originalName -match '^icu.*78\.dll$'
    }
)
if ($remainingIncompatibleIcu.Count -gt 0) {
    $relativePaths = @(
        $remainingIncompatibleIcu | ForEach-Object {
            $_.FullName.Substring($bundlePath.Length).TrimStart('\')
        }
    )
    throw "Conflicting ICU dependencies remain in release bundle: $($relativePaths -join ', ')"
}

Write-Host "[4/5] Copying runtime resources and creating the build manifest..." -ForegroundColor Cyan
Copy-Item -LiteralPath (Join-Path $payloadPath "data") -Destination $bundlePath -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot "app.ico") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\MODEL_REPLACEMENT.txt") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\Run_FableDan_Fixed_Benchmark.bat") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\Collect_Diagnostics.bat") -Destination $bundlePath
New-Item -ItemType Directory -Force -Path (Join-Path $bundlePath "logs") | Out-Null

$bundledFableDanModel = Join-Path $bundlePath "data\profiles\tencent_daguandan\models\best.npz"
$bundledDanZeroModel = Join-Path $bundlePath "data\profiles\tencent_daguandan\models\danzero\q_network.ckpt"
foreach ($model in @($bundledFableDanModel, $bundledDanZeroModel)) {
    if (-not (Test-Path -LiteralPath $model)) {
        throw "Expected replaceable model missing from release: $model"
    }
}

Invoke-ProjectPython $buildManifestGenerator create `
    --project-root $projectRoot `
    --bundle-root $bundlePath `
    --output $buildManifestPath `
    --executable-name "$appName.exe" `
    --profile-name "tencent_daguandan"
Invoke-ProjectPython $buildManifestGenerator verify `
    --bundle-root $bundlePath `
    --manifest $buildManifestPath `
    --strict

Write-Host "[5/5] Creating the archive, checksum, and release record..." -ForegroundColor Cyan
$archiveCreated = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
    try {
        Assert-NoReparsePathChain -LiteralPath $archivePath
        if ([System.IO.Directory]::Exists($archivePath)) {
            throw "Managed release archive target became a directory: $archivePath"
        }
        if (Test-Path -LiteralPath $archivePath) {
            Remove-Item -LiteralPath $archivePath -Force
        }
        Compress-Archive -LiteralPath $bundlePath -DestinationPath $archivePath -Force
        $archiveCreated = $true
        break
    }
    catch {
        if ($attempt -eq 3) {
            throw
        }
        Write-Host "Archive attempt $attempt failed because a file is temporarily locked; retrying..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
    }
}
if (-not $archiveCreated) {
    throw "The release archive could not be created."
}

Invoke-ProjectPython $buildManifestGenerator release-record `
    --manifest $buildManifestPath `
    --archive $archivePath `
    --record $releaseRecordPath `
    --checksum $archiveChecksumPath

$size = (Get-ChildItem -LiteralPath $bundlePath -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Complete: {0}" -f $bundlePath) -ForegroundColor Green
Write-Host ("Archive: {0}" -f $archivePath) -ForegroundColor Green
Write-Host ("Archive SHA256: {0}" -f $archiveChecksumPath) -ForegroundColor Green
Write-Host ("Release record: {0}" -f $releaseRecordPath) -ForegroundColor Green
Write-Host ("Uncompressed size: {0:N1} MB" -f $size) -ForegroundColor Green
