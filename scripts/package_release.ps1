[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ReleaseRoot,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $WheelhouseRoot,

    [Parameter()]
    [string] $BootstrapPython = "",

    [Parameter()]
    [switch] $AllowDirtyDevelopmentBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-ExistingPathAttributes {
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
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
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
    $current = [System.IO.Path]::GetFullPath($LiteralPath)
    while (-not [string]::IsNullOrWhiteSpace($current)) {
        $attributes = Get-ExistingPathAttributes -LiteralPath $current
        if (
            $null -ne $attributes -and
            ($attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
        ) {
            throw "Path traverses a symlink, junction, or reparse point: $current"
        }
        $parent = [System.IO.Directory]::GetParent($current)
        if ($null -eq $parent -or $parent.FullName -eq $current) {
            break
        }
        $current = $parent.FullName
    }
}

function Assert-NoReparseTree {
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
    Assert-NoReparsePathChain -LiteralPath $LiteralPath
    if (-not [System.IO.Directory]::Exists($LiteralPath)) {
        throw "Required directory does not exist: $LiteralPath"
    }
    $pending = [System.Collections.Generic.Stack[System.IO.DirectoryInfo]]::new()
    $pending.Push([System.IO.DirectoryInfo]::new($LiteralPath))
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        if (($directory.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Directory is a reparse point: $($directory.FullName)"
        }
        foreach ($entry in $directory.EnumerateFileSystemInfos()) {
            if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Tree contains a symlink, junction, or reparse point: $($entry.FullName)"
            }
            if (($entry.Attributes -band [System.IO.FileAttributes]::Directory) -ne 0) {
                $pending.Push([System.IO.DirectoryInfo] $entry)
            }
        }
    }
}

function Test-IsSameOrBelow {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Root
    )
    $resolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    return (
        $resolvedPath.Equals($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        $resolvedPath.StartsWith(
            "$resolvedRoot\",
            [System.StringComparison]::OrdinalIgnoreCase
        )
    )
}

function Assert-DisjointRoots {
    param(
        [Parameter(Mandatory = $true)][string] $First,
        [Parameter(Mandatory = $true)][string] $Second,
        [Parameter(Mandatory = $true)][string] $Description
    )
    if (
        (Test-IsSameOrBelow -Path $First -Root $Second) -or
        (Test-IsSameOrBelow -Path $Second -Root $First)
    ) {
        throw "$Description must be disjoint: $First ; $Second"
    }
}

function Resolve-ManagedChildPath {
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)][string] $Child
    )
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    $resolvedChild = [System.IO.Path]::GetFullPath($Child)
    if (-not $resolvedChild.StartsWith(
        "$resolvedRoot\",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Managed release path must stay below ReleaseRoot: $resolvedChild"
    }
    return $resolvedChild
}

function Invoke-PythonCommand {
    param(
        [Parameter(Mandatory = $true)][string] $Python,
        [Parameter(ValueFromRemainingArguments = $true)][string[]] $Arguments
    )
    & $Python -I @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit code $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

function Invoke-CleanPython {
    param(
        [Parameter(Mandatory = $true)][string] $Python,
        [Parameter(Mandatory = $true)][string] $ScriptsPath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]] $Arguments
    )
    $removedNames = @(
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QML2_IMPORT_PATH",
        "TCL_LIBRARY",
        "TK_LIBRARY",
        "JAVA_HOME",
        "JDK_HOME",
        "CLASSPATH",
        "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_EXE",
        "_CE_CONDA",
        "_CE_M",
        "POPPLER_PATH"
    )
    $managedNames = @(
        $removedNames + @(
            "PATH",
            "PYTHONNOUSERSITE",
            "PIP_NO_INDEX",
            "PIP_DISABLE_PIP_VERSION_CHECK",
            "PIP_CONFIG_FILE",
            "PYINSTALLER_CONFIG_DIR",
            "TMP",
            "TEMP"
        )
    )
    $saved = @{}
    foreach ($name in $managedNames) {
        $saved[$name] = [System.Environment]::GetEnvironmentVariable($name, "Process")
    }
    try {
        foreach ($name in $removedNames) {
            [System.Environment]::SetEnvironmentVariable($name, $null, "Process")
        }
        $cleanPath = @(
            $ScriptsPath,
            $script:pythonBaseRoot,
            (Join-Path $script:pythonBaseRoot "Scripts"),
            (Join-Path $env:SystemRoot "System32"),
            $env:SystemRoot
        ) -join ";"
        [System.Environment]::SetEnvironmentVariable("PATH", $cleanPath, "Process")
        [System.Environment]::SetEnvironmentVariable("PYTHONNOUSERSITE", "1", "Process")
        [System.Environment]::SetEnvironmentVariable("PIP_NO_INDEX", "1", "Process")
        [System.Environment]::SetEnvironmentVariable("PIP_DISABLE_PIP_VERSION_CHECK", "1", "Process")
        [System.Environment]::SetEnvironmentVariable("PIP_CONFIG_FILE", "NUL", "Process")
        [System.Environment]::SetEnvironmentVariable(
            "PYINSTALLER_CONFIG_DIR",
            $script:pyinstallerConfigPath,
            "Process"
        )
        [System.Environment]::SetEnvironmentVariable("TMP", $script:tempPath, "Process")
        [System.Environment]::SetEnvironmentVariable("TEMP", $script:tempPath, "Process")
        & $Python -I @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Clean Python command failed (exit code $LASTEXITCODE): $($Arguments -join ' ')"
        }
    }
    finally {
        foreach ($name in $managedNames) {
            [System.Environment]::SetEnvironmentVariable($name, $saved[$name], "Process")
        }
    }
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$appName = "DaguandanAssistant"
$releaseRoot = [System.IO.Path]::GetFullPath($ReleaseRoot)
$wheelhouseRoot = [System.IO.Path]::GetFullPath($WheelhouseRoot)
$filesystemRoot = [System.IO.Path]::GetPathRoot($releaseRoot)
if (
    [string]::IsNullOrWhiteSpace($filesystemRoot) -or
    $releaseRoot.TrimEnd('\') -eq $filesystemRoot.TrimEnd('\')
) {
    throw "ReleaseRoot must be a unique dedicated directory, not a filesystem root."
}
if ([System.IO.File]::Exists($releaseRoot) -or [System.IO.Directory]::Exists($releaseRoot)) {
    throw "ReleaseRoot must be unique and must not already exist: $releaseRoot"
}
Assert-NoReparsePathChain -LiteralPath $releaseRoot
Assert-NoReparseTree -LiteralPath $wheelhouseRoot
Assert-DisjointRoots -First $releaseRoot -Second $projectRoot -Description "ReleaseRoot and project root"
Assert-DisjointRoots -First $wheelhouseRoot -Second $projectRoot -Description "WheelhouseRoot and project root"
Assert-DisjointRoots -First $releaseRoot -Second $wheelhouseRoot -Description "ReleaseRoot and WheelhouseRoot"

if ([string]::IsNullOrWhiteSpace($BootstrapPython)) {
    $BootstrapPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
}
$bootstrapPython = [System.IO.Path]::GetFullPath($BootstrapPython)
if (-not [System.IO.File]::Exists($bootstrapPython)) {
    throw "Hash-locked bootstrap Python executable not found: $bootstrapPython"
}
Assert-NoReparsePathChain -LiteralPath $bootstrapPython

$gitCommand = Get-Command git -ErrorAction SilentlyContinue
if ($null -eq $gitCommand) {
    throw "Git is required to capture release source identity."
}
$gitStatus = @(& $gitCommand.Source -C $projectRoot status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw "Git could not inspect the source tree before packaging."
}
$sourceTreeDirty = $gitStatus.Count -gt 0
if ($sourceTreeDirty -and -not $AllowDirtyDevelopmentBuild) {
    throw "Source tree is dirty. Commit or stash changes before a release build."
}
if ($sourceTreeDirty) {
    Write-Warning "Creating an explicitly allowed dirty development build."
}

$requiredFiles = @(
    (Join-Path $projectRoot "run.py"),
    (Join-Path $projectRoot "app.ico"),
    (Join-Path $projectRoot "requirements-release.in"),
    (Join-Path $projectRoot "requirements-release.lock"),
    (Join-Path $projectRoot "release_toolchain.lock.json"),
    (Join-Path $projectRoot "wheelhouse.lock.json"),
    (Join-Path $projectRoot "scripts\verify_release_inputs.py"),
    (Join-Path $projectRoot "scripts\audit_frozen_bundle.py"),
    (Join-Path $projectRoot "scripts\generate_build_manifest.py"),
    (Join-Path $projectRoot "release_assets\MODEL_REPLACEMENT.txt"),
    (Join-Path $projectRoot "release_assets\Run_FableDan_Fixed_Benchmark.bat"),
    (Join-Path $projectRoot "release_assets\Collect_Diagnostics.bat")
)
$profileSource = Join-Path $projectRoot "data\profiles\tencent_daguandan"
$modelSource = Join-Path $profileSource "models\best.npz"
$danzeroWeightsSource = Join-Path $projectRoot "src\daguandan_bridge\danzero\_vendor\guandan_rlcard\baselines\danzero\q_network.ckpt"
$requiredFiles += @(
    (Join-Path $profileSource "profile.json"),
    (Join-Path $profileSource "regions_config.json"),
    (Join-Path $profileSource "templates_config.json"),
    (Join-Path $profileSource "templates"),
    $modelSource,
    $danzeroWeightsSource
)
foreach ($requiredFile in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $requiredFile)) {
        throw "Required release input is missing: $requiredFile"
    }
}

Write-Host "[1/7] Verifying committed locks and the external wheelhouse..." -ForegroundColor Cyan
Invoke-PythonCommand $bootstrapPython `
    (Join-Path $projectRoot "scripts\verify_release_inputs.py") `
    --project-root $projectRoot `
    --wheelhouse $wheelhouseRoot `
    --python $bootstrapPython

[System.IO.Directory]::CreateDirectory($releaseRoot) | Out-Null
Assert-NoReparsePathChain -LiteralPath $releaseRoot
$ownershipMarkerPath = Join-Path $releaseRoot ".daguandan-release-root"
[System.IO.File]::WriteAllText(
    $ownershipMarkerPath,
    "guandan.package-release-root/2`r`n",
    [System.Text.UTF8Encoding]::new($false)
)

$distPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "dist")
$workPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "build")
$specPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "spec")
$payloadPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "payload")
$buildEnvPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "build-env")
$script:tempPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "temp")
$script:pyinstallerConfigPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "pyinstaller-config")
$bundlePath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $distPath $appName)
$archivePath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "DaguandanAssistant.zip")
$archiveChecksumPath = Resolve-ManagedChildPath -Root $releaseRoot -Child "$archivePath.sha256"
$releaseRecordPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "DaguandanAssistant.release.json")
$sourceIdentityPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "source_identity.json")
$releaseInputAuditPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "release_input_audit.json")
$buildManifestPath = Join-Path $bundlePath "build_manifest.json"
$nativeAuditPath = Join-Path $bundlePath "native_dependency_audit.json"
$bundledInputAuditPath = Join-Path $bundlePath "release_input_audit.json"

foreach ($directory in @($distPath, $workPath, $specPath, $payloadPath, $script:tempPath, $script:pyinstallerConfigPath)) {
    [System.IO.Directory]::CreateDirectory($directory) | Out-Null
    Assert-NoReparsePathChain -LiteralPath $directory
}

Invoke-PythonCommand $bootstrapPython `
    (Join-Path $projectRoot "scripts\generate_build_manifest.py") `
    source-identity `
    --project-root $projectRoot `
    --output $sourceIdentityPath

$script:pythonBaseRoot = (& $bootstrapPython -I -c "import sys; print(sys.base_prefix)").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($script:pythonBaseRoot)) {
    throw "Could not resolve the locked CPython base prefix."
}
Assert-NoReparsePathChain -LiteralPath $script:pythonBaseRoot

Write-Host "[2/7] Creating a fresh isolated build environment..." -ForegroundColor Cyan
Invoke-CleanPython $bootstrapPython (Split-Path -Parent $bootstrapPython) `
    -m venv $buildEnvPath
$buildPython = Join-Path $buildEnvPath "Scripts\python.exe"
if (-not [System.IO.File]::Exists($buildPython)) {
    throw "Fresh build environment did not create python.exe."
}
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") `
    -m pip install `
    --isolated `
    --no-index `
    --no-cache-dir `
    --disable-pip-version-check `
    --require-hashes `
    --ignore-requires-python `
    --no-deps `
    --find-links $wheelhouseRoot `
    -r (Join-Path $projectRoot "requirements-release.lock")
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") `
    (Join-Path $projectRoot "scripts\verify_release_inputs.py") `
    --project-root $projectRoot `
    --wheelhouse $wheelhouseRoot `
    --python $buildPython `
    --verify-installed `
    --output $releaseInputAuditPath

Write-Host "[3/7] Preparing immutable seed resources..." -ForegroundColor Cyan
$payloadProfile = Join-Path $payloadPath "data\profiles\tencent_daguandan"
[System.IO.Directory]::CreateDirectory((Join-Path $payloadProfile "models\danzero")) | Out-Null
Copy-Item -LiteralPath (Join-Path $profileSource "profile.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "regions_config.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "templates_config.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "templates") -Destination $payloadProfile -Recurse
Copy-Item -LiteralPath $modelSource -Destination (Join-Path $payloadProfile "models\best.npz")
Copy-Item -LiteralPath $danzeroWeightsSource -Destination (Join-Path $payloadProfile "models\danzero\q_network.ckpt")

Write-Host "[4/7] Building the frozen application from the clean environment..." -ForegroundColor Cyan
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
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") @pyinstallerArguments
$executablePath = Join-Path $bundlePath "$appName.exe"
if (-not [System.IO.File]::Exists($executablePath)) {
    throw "PyInstaller did not produce $appName.exe."
}

Write-Host "[5/7] Adding resources and running the fail-closed native audit..." -ForegroundColor Cyan
Copy-Item -LiteralPath (Join-Path $payloadPath "data") -Destination $bundlePath -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot "app.ico") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\MODEL_REPLACEMENT.txt") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\Run_FableDan_Fixed_Benchmark.bat") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\Collect_Diagnostics.bat") -Destination $bundlePath
Copy-Item -LiteralPath $releaseInputAuditPath -Destination $bundledInputAuditPath
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") `
    (Join-Path $projectRoot "scripts\audit_frozen_bundle.py") `
    --bundle-root $bundlePath `
    --work-root $workPath `
    --venv-root $buildEnvPath `
    --python-root $script:pythonBaseRoot `
    --project-root $projectRoot `
    --output $nativeAuditPath

Write-Host "[6/7] Creating and strictly verifying the build manifest..." -ForegroundColor Cyan
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") `
    (Join-Path $projectRoot "scripts\generate_build_manifest.py") `
    create `
    --project-root $projectRoot `
    --bundle-root $bundlePath `
    --output $buildManifestPath `
    --executable-name "$appName.exe" `
    --profile-name "tencent_daguandan" `
    --source-identity $sourceIdentityPath `
    --release-input-audit $bundledInputAuditPath `
    --native-audit $nativeAuditPath
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") `
    (Join-Path $projectRoot "scripts\generate_build_manifest.py") `
    verify `
    --bundle-root $bundlePath `
    --manifest $buildManifestPath `
    --strict

Write-Host "[7/7] Creating the archive, checksum, and release record..." -ForegroundColor Cyan
$archiveCreated = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
    try {
        Assert-NoReparsePathChain -LiteralPath $archivePath
        if ([System.IO.File]::Exists($archivePath)) {
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
        Start-Sleep -Seconds 3
    }
}
if (-not $archiveCreated) {
    throw "The release archive could not be created."
}
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") `
    (Join-Path $projectRoot "scripts\generate_build_manifest.py") `
    release-record `
    --manifest $buildManifestPath `
    --archive $archivePath `
    --record $releaseRecordPath `
    --checksum $archiveChecksumPath

$size = (Get-ChildItem -LiteralPath $bundlePath -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Complete bundle: {0}" -f $bundlePath) -ForegroundColor Green
Write-Host ("Archive: {0}" -f $archivePath) -ForegroundColor Green
Write-Host ("Archive SHA256: {0}" -f $archiveChecksumPath) -ForegroundColor Green
Write-Host ("Release record: {0}" -f $releaseRecordPath) -ForegroundColor Green
Write-Host ("Uncompressed size: {0:N1} MB" -f $size) -ForegroundColor Green
