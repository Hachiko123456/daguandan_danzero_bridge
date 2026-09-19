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
    [switch] $AllowDirtyDevelopmentBuild,

    [Parameter()]
    [switch] $OverwriteExisting,

    [Parameter()]
    [switch] $Compact
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

function Clear-ManagedReleaseRoot {
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)][bool] $Overwrite
    )
    $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    if (-not [System.IO.Directory]::Exists($fullRoot)) {
        return
    }
    if (-not $Overwrite) {
        throw "ReleaseRoot already exists. Use -OverwriteExisting only for a managed current release root: $fullRoot"
    }
    Assert-NoReparseTree -LiteralPath $fullRoot
    $markerPath = Join-Path $fullRoot ".daguandan-release-root"
    if (-not [System.IO.File]::Exists($markerPath)) {
        throw "Existing ReleaseRoot is not owned by this packaging system: $fullRoot"
    }
    if ((Get-Content -LiteralPath $markerPath -Raw).Trim() -ne "guandan.package-release-root/2") {
        throw "Existing ReleaseRoot ownership marker is invalid: $fullRoot"
    }
    $managed = @(
        "build", "build-env", "dist", "payload", "spec", "temp", "pyinstaller-config",
        "DaguandanAssistant.zip", "DaguandanAssistant.zip.sha256", "DaguandanAssistant.release.json",
        "source_identity.json", "release_input_audit.json", "bootstrap_python_audit.json"
    )
    foreach ($entry in Get-ChildItem -LiteralPath $fullRoot -Force) {
        if ($entry.Name -eq ".daguandan-release-root") { continue }
        if ($managed -notcontains $entry.Name) {
            throw "Existing ReleaseRoot contains unmanaged content; refusing to delete: $($entry.FullName)"
        }
    }
    foreach ($name in $managed) {
        $candidate = Join-Path $fullRoot $name
        if (Test-Path -LiteralPath $candidate) {
            Remove-Item -LiteralPath $candidate -Recurse -Force
        }
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
    & $Python -I -S @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit code $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

function Invoke-CleanPython {
    param(
        [Parameter(Mandatory = $true)][string] $Python,
        [Parameter(Mandatory = $true)][string] $ScriptsPath,
        [Parameter()][switch] $NoSite,
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
        if ($NoSite) {
            & $Python -I -S @Arguments
        }
        else {
            & $Python -I @Arguments
        }
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
$projectCurrentReleaseRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "release\current"))
$isProjectCurrentRelease = $releaseRoot.TrimEnd('\').Equals($projectCurrentReleaseRoot.TrimEnd('\'), [System.StringComparison]::OrdinalIgnoreCase)
if (
    [string]::IsNullOrWhiteSpace($filesystemRoot) -or
$releaseRoot.TrimEnd('\') -eq $filesystemRoot.TrimEnd('\')
) {
    throw "ReleaseRoot must not be a filesystem root."
}
if ($isProjectCurrentRelease) {
    if (-not $OverwriteExisting -and (Test-Path -LiteralPath $releaseRoot)) {
        throw "In-project ReleaseRoot already exists. Use -OverwriteExisting: $releaseRoot"
    }
    if (Test-Path -LiteralPath $releaseRoot) {
        Clear-ManagedReleaseRoot -Root $releaseRoot -Overwrite $OverwriteExisting
    }
    else {
        [System.IO.Directory]::CreateDirectory($releaseRoot) | Out-Null
    }
    Assert-NoReparsePathChain -LiteralPath $releaseRoot
}
else {
    if (Test-Path -LiteralPath $releaseRoot) {
        if (-not $OverwriteExisting) {
            throw "ReleaseRoot already exists. Use -OverwriteExisting: $releaseRoot"
        }
        Clear-ManagedReleaseRoot -Root $releaseRoot -Overwrite $true
    }
    Assert-NoReparsePathChain -LiteralPath $releaseRoot
}
Assert-NoReparseTree -LiteralPath $wheelhouseRoot
if (-not $isProjectCurrentRelease) {
    Assert-DisjointRoots -First $releaseRoot -Second $projectRoot -Description "ReleaseRoot and project root"
}
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
$script:pythonBaseRoot = (& $bootstrapPython -I -S -c "import sys; print(sys.base_prefix)").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($script:pythonBaseRoot)) {
    throw "Could not resolve the locked CPython base prefix without site initialization."
}
Assert-NoReparsePathChain -LiteralPath $script:pythonBaseRoot

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
    (Join-Path $projectRoot "python_runtime.lock.json"),
    (Join-Path $projectRoot "wheelhouse.lock.json"),
    (Join-Path $projectRoot "scripts\verify_release_inputs.py"),
    (Join-Path $projectRoot "scripts\audit_bootstrap_python.py"),
    (Join-Path $projectRoot "scripts\audit_frozen_bundle.py"),
    (Join-Path $projectRoot "scripts\generate_build_manifest.py"),
    (Join-Path $projectRoot "release_assets\MODEL_REPLACEMENT.txt"),
    (Join-Path $projectRoot "release_assets\Run_FableDan_Fixed_Benchmark.bat"),
    (Join-Path $projectRoot "release_assets\Run_FableDan_Fixed_Benchmark.ps1"),
    (Join-Path $projectRoot "release_assets\Collect_Diagnostics.bat"),
    (Join-Path $projectRoot "release_assets\Clean-DaguandanDiagnostics.ps1"),
    (Join-Path $projectRoot "release_assets\Launch_DaguandanAssistant.bat"),
    (Join-Path $projectRoot "release_assets\Launch_DaguandanAssistant.ps1"),
    (Join-Path $projectRoot "scripts\pyinstaller_live_v2_collection.json")
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
$bootstrapAuditPath = Resolve-ManagedChildPath -Root $releaseRoot -Child (Join-Path $releaseRoot "bootstrap_python_audit.json")
$buildManifestPath = Join-Path $bundlePath "build_manifest.json"
$nativeAuditPath = Join-Path $bundlePath "native_dependency_audit.json"
$bundledInputAuditPath = Join-Path $bundlePath "release_input_audit.json"
$bundledBootstrapAuditPath = Join-Path $bundlePath "bootstrap_python_audit.json"

foreach ($directory in @($distPath, $workPath, $specPath, $payloadPath, $script:tempPath, $script:pyinstallerConfigPath)) {
    [System.IO.Directory]::CreateDirectory($directory) | Out-Null
    Assert-NoReparsePathChain -LiteralPath $directory
}

Invoke-PythonCommand $bootstrapPython `
    (Join-Path $projectRoot "scripts\generate_build_manifest.py") `
    source-identity `
    --project-root $projectRoot `
    --output $sourceIdentityPath
$sourceIdentityVerifyArguments = @(
    (Join-Path $projectRoot "scripts\generate_build_manifest.py"),
    "verify-source-identity",
    "--project-root", $projectRoot,
    "--expected", $sourceIdentityPath
)
if ($AllowDirtyDevelopmentBuild) {
    $sourceIdentityVerifyArguments += "--allow-dirty"
}
Invoke-PythonCommand $bootstrapPython @sourceIdentityVerifyArguments

Write-Host "[2/7] Creating a fresh isolated build environment..." -ForegroundColor Cyan
$bootstrapAuditArguments = @(
    (Join-Path $projectRoot "scripts\audit_bootstrap_python.py"),
    "--python-root", $script:pythonBaseRoot,
    "--runtime-lock", (Join-Path $projectRoot "python_runtime.lock.json"),
    "--output", $bootstrapAuditPath
)
Invoke-CleanPython `
    -Python $bootstrapPython `
    -ScriptsPath (Split-Path -Parent $bootstrapPython) `
    -NoSite `
    -Arguments $bootstrapAuditArguments
Invoke-CleanPython `
    -Python $bootstrapPython `
    -ScriptsPath (Split-Path -Parent $bootstrapPython) `
    -NoSite `
    -Arguments @("-m", "venv", $buildEnvPath)
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
$liveV2CollectionPath = Join-Path $projectRoot "scripts\pyinstaller_live_v2_collection.json"
$liveV2Collection = Get-Content -LiteralPath $liveV2CollectionPath -Raw | ConvertFrom-Json
if ($liveV2Collection.schema -ne "guandan.pyinstaller-live-v2-collection/1") {
    throw "Unsupported live-v2 PyInstaller collection schema."
}
$liveV2HiddenImports = @(
    $liveV2Collection.hidden_imports |
        ForEach-Object { [string] $_ } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)
if ($liveV2HiddenImports.Count -eq 0) {
    throw "The live-v2 PyInstaller hidden-import inventory is empty."
}
if (($liveV2HiddenImports | Sort-Object -Unique).Count -ne $liveV2HiddenImports.Count) {
    throw "The live-v2 PyInstaller hidden-import inventory contains duplicates."
}
foreach ($module in $liveV2HiddenImports) {
    if ($module -notmatch '^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$') {
        throw "Invalid live-v2 PyInstaller hidden import: $module"
    }
}
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
    "--hidden-import", "pywintypes"
)
foreach ($module in $liveV2HiddenImports) {
    $pyinstallerArguments += @("--hidden-import", $module)
}
$pyinstallerArguments += (Join-Path $projectRoot "run.py")
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
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\Run_FableDan_Fixed_Benchmark.ps1") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\Collect_Diagnostics.bat") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\Clean-DaguandanDiagnostics.ps1") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\Launch_DaguandanAssistant.bat") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\Launch_DaguandanAssistant.ps1") -Destination $bundlePath
if ($AllowDirtyDevelopmentBuild) {
    [System.IO.File]::WriteAllText(
        (Join-Path $bundlePath "DEVELOPMENT_BUILD_NOT_FORMALLY_QUALIFIED.txt"),
        "This bundle was built from an explicitly allowed dirty working tree.`r`n" +
        "It is for isolated development validation only and is not a formally qualified release.`r`n",
        [System.Text.UTF8Encoding]::new($false)
    )
}
Copy-Item -LiteralPath $releaseInputAuditPath -Destination $bundledInputAuditPath
Copy-Item -LiteralPath $bootstrapAuditPath -Destination $bundledBootstrapAuditPath
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") `
    (Join-Path $projectRoot "scripts\audit_frozen_bundle.py") `
    --bundle-root $bundlePath `
    --work-root $workPath `
    --venv-root $buildEnvPath `
    --python-root $script:pythonBaseRoot `
    --project-root $projectRoot `
    --output $nativeAuditPath

# PyInstaller imports the source tree for an extended period.  Re-prove the
# tracked commit/tree/status before signing those bytes into the manifest.
Invoke-PythonCommand $bootstrapPython @sourceIdentityVerifyArguments

Write-Host "[6/7] Creating and verifying the build manifest..." -ForegroundColor Cyan
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
$manifestVerifyArguments = @(
    (Join-Path $projectRoot "scripts\generate_build_manifest.py"),
    "verify",
    "--bundle-root", $bundlePath,
    "--manifest", $buildManifestPath
)
if (-not $AllowDirtyDevelopmentBuild) {
    $manifestVerifyArguments += "--strict"
}
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") @manifestVerifyArguments

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

# A build hook or concurrent editor must not be able to modify tracked source
# after manifest creation and still publish a formally qualified archive.
Invoke-PythonCommand $bootstrapPython @sourceIdentityVerifyArguments

$size = (Get-ChildItem -LiteralPath $bundlePath -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
if ($Compact) {
    foreach ($managedDirectory in @($distPath, $workPath, $buildEnvPath, $payloadPath, $specPath, $script:tempPath, $script:pyinstallerConfigPath)) {
        if (Test-Path -LiteralPath $managedDirectory) {
            Remove-Item -LiteralPath $managedDirectory -Recurse -Force
        }
    }
    Write-Host "Compact mode removed build intermediates; archive and release records remain." -ForegroundColor Green
}
Write-Host ("Complete bundle: {0}" -f $bundlePath) -ForegroundColor Green
Write-Host ("Archive: {0}" -f $archivePath) -ForegroundColor Green
Write-Host ("Archive SHA256: {0}" -f $archiveChecksumPath) -ForegroundColor Green
Write-Host ("Release record: {0}" -f $releaseRecordPath) -ForegroundColor Green
Write-Host ("Uncompressed size: {0:N1} MB" -f $size) -ForegroundColor Green
