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

function Assert-ManagedReleaseRoot {
    param(
        [Parameter(Mandatory = $true)][string] $Root
    )
    $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    if (-not [System.IO.Directory]::Exists($fullRoot)) {
        if (Test-Path -LiteralPath $fullRoot) { throw "Managed release root is not a directory: $fullRoot" }
        return
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
        "source_identity.json", "release_input_audit.json", "bootstrap_python_audit.json", "build_metrics.json"
    )
    foreach ($entry in Get-ChildItem -LiteralPath $fullRoot -Force) {
        if ($entry.Name -eq ".daguandan-release-root") { continue }
        if ($managed -notcontains $entry.Name) {
            throw "Existing ReleaseRoot contains unmanaged content; refusing to delete: $($entry.FullName)"
        }
    }
}

function Clear-ManagedReleaseRoot {
    param([Parameter(Mandatory=$true)][string] $Root, [Parameter(Mandatory=$true)][bool] $Overwrite)
    if (-not $Overwrite) { throw "Explicit overwrite permission is required." }
    Assert-ManagedReleaseRoot -Root $Root
    $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    if ($fullRoot -eq [IO.Path]::GetPathRoot($fullRoot).TrimEnd('\')) { throw "Refuse root deletion." }
    foreach ($entry in Get-ChildItem -LiteralPath $fullRoot -Force) {
        if ($entry.Name -eq '.daguandan-release-root') { continue }
        $candidate = [IO.Path]::GetFullPath($entry.FullName)
        if ([IO.Path]::GetDirectoryName($candidate) -ne $fullRoot) { throw "Cleanup escaped managed root." }
        Remove-Item -LiteralPath $candidate -Recurse -Force
    }
}

function Ensure-OwnedBuildDirectory {
    param([string] $Root, [string] $Marker, [string] $Identity)
    Assert-NoReparsePathChain -LiteralPath $Root
    $markerPath = Join-Path $Root $Marker
    if (Test-Path -LiteralPath $Root) {
        if (-not [IO.Directory]::Exists($Root)) { throw "Build root is not a directory: $Root" }
        Assert-NoReparsePathChain -LiteralPath $markerPath
        if (Test-Path -LiteralPath $markerPath) {
            if ([IO.File]::ReadAllText($markerPath).Trim() -ne $Identity) { throw "Build root belongs to another owner: $Root" }
        } elseif (@(Get-ChildItem -LiteralPath $Root -Force).Count -gt 0) {
            throw "Refusing unowned build directory: $Root"
        }
    }
    [IO.Directory]::CreateDirectory($Root) | Out-Null
    [IO.File]::WriteAllText($markerPath, $Identity, [Text.UTF8Encoding]::new($false))
}

function Reset-OwnedCacheDirectory {
    param([string] $Directory, [string] $CacheRoot)
    $directoryPath = Resolve-ManagedChildPath -Root $CacheRoot -Child $Directory
    Assert-NoReparsePathChain -LiteralPath $directoryPath
    if (Test-Path -LiteralPath $directoryPath) {
        Assert-NoReparseTree -LiteralPath $directoryPath
        Remove-Item -LiteralPath $directoryPath -Recurse -Force
    }
    $receipt = "$directoryPath.receipt.json"
    Assert-NoReparsePathChain -LiteralPath $receipt
    if (Test-Path -LiteralPath $receipt) { Remove-Item -LiteralPath $receipt -Force }
    [IO.Directory]::CreateDirectory($directoryPath) | Out-Null
}

function Publish-ManagedRelease {
    param([Parameter(Mandatory=$true)][string] $Stage, [Parameter(Mandatory=$true)][string] $Destination)
    $stagePath = [IO.Path]::GetFullPath($Stage).TrimEnd('\')
    $finalPath = [IO.Path]::GetFullPath($Destination).TrimEnd('\')
    $previousPath = "$finalPath.previous"
    if ($finalPath -eq [IO.Path]::GetPathRoot($finalPath).TrimEnd('\')) { throw "Refuse publishing at filesystem root." }
    Assert-DisjointRoots -First $stagePath -Second $finalPath -Description 'Stage and destination'
    Assert-DisjointRoots -First $stagePath -Second $previousPath -Description 'Stage and previous'
    Assert-NoReparsePathChain -LiteralPath $finalPath
    Assert-NoReparsePathChain -LiteralPath $previousPath
    Assert-ManagedReleaseRoot -Root $stagePath
    if (-not (Test-Path -LiteralPath (Join-Path $stagePath '.daguandan-release-root'))) { throw 'Stage is not owned.' }
    if (Test-Path -LiteralPath $finalPath) { Assert-ManagedReleaseRoot -Root $finalPath }
    if (Test-Path -LiteralPath $previousPath) {
        Assert-ManagedReleaseRoot -Root $previousPath
        Clear-ManagedReleaseRoot -Root $previousPath -Overwrite $true
        Remove-Item -LiteralPath (Join-Path $previousPath '.daguandan-release-root') -Force
        Remove-Item -LiteralPath $previousPath -Force
    }
    $movedOld = $false
    try {
        if (Test-Path -LiteralPath $finalPath) {
            Move-Item -LiteralPath $finalPath -Destination $previousPath
            $movedOld = $true
        }
        Move-Item -LiteralPath $stagePath -Destination $finalPath
    } catch {
        if ($movedOld -and -not (Test-Path -LiteralPath $finalPath) -and (Test-Path -LiteralPath $previousPath)) {
            Move-Item -LiteralPath $previousPath -Destination $finalPath
        }
        throw
    }
}

function Complete-BuildStage {
    if ($null -ne $script:stageWatch) {
        $script:stageWatch.Stop()
        $seconds = [Math]::Round($script:stageWatch.Elapsed.TotalSeconds, 3)
        $script:stageTimings.Add([ordered]@{name=$script:stageName; seconds=$seconds})
        Write-Host ("[done] {0}: {1:N1}s (total {2:N1}s)" -f $script:stageName, $seconds, $script:buildWatch.Elapsed.TotalSeconds) -ForegroundColor DarkCyan
        $script:stageWatch = $null
    }
}
function Start-BuildStage {
    param([int] $Number, [string] $Name)
    Complete-BuildStage
    $script:stageName = $Name
    $script:stageWatch = [Diagnostics.Stopwatch]::StartNew()
    Write-Host ("[{0}/8] {1}" -f $Number, $Name) -ForegroundColor Cyan
}
function Write-BuildMetrics {
    param([string] $Root, [string] $Status, [string] $Failure = '')
    $document = [ordered]@{
        schema='guandan.release-build-metrics/1'; status=$Status;
        started_at_utc=$script:buildStarted; total_seconds=[Math]::Round($script:buildWatch.Elapsed.TotalSeconds,3);
        environment_cache_hit=$script:environmentCacheHit; work_cache_hit=$script:workCacheHit;
        cache_enabled=$script:cacheEnabled; stages=@($script:stageTimings.ToArray()); error=$Failure
    }
    $path = Join-Path $Root 'build_metrics.json'
    Assert-NoReparsePathChain -LiteralPath $path
    [IO.File]::WriteAllText($path, ($document | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
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
    & $Python -I -B -S @Arguments
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
            & $Python -I -B -S @Arguments
        }
        else {
            & $Python -I -B @Arguments
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

$script:buildWatch = [Diagnostics.Stopwatch]::StartNew()
$script:buildStarted = [DateTime]::UtcNow.ToString('o')
$script:stageTimings = New-Object 'System.Collections.Generic.List[object]'
$script:stageWatch = $null
$script:environmentCacheHit = $false
$script:workCacheHit = $false
$script:cacheEnabled = [bool]$AllowDirtyDevelopmentBuild
$cacheLock = $null
$requestedReleaseRoot = $ReleaseRoot
$stagingReleaseRoot = $null
try {
Start-BuildStage 1 'Validate inputs and preserve the previous release'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$appName = 'DaguandanAssistant'
$finalReleaseRoot = [IO.Path]::GetFullPath($requestedReleaseRoot).TrimEnd('\')
$wheelhouseRoot = [IO.Path]::GetFullPath($WheelhouseRoot)
if ([string]::IsNullOrWhiteSpace([IO.Path]::GetPathRoot($finalReleaseRoot)) -or
    $finalReleaseRoot -eq [IO.Path]::GetPathRoot($finalReleaseRoot).TrimEnd('\')) {
    throw 'ReleaseRoot must not be a filesystem root.'
}
$projectCurrentReleaseRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'release\current'))
$isProjectCurrentRelease = $finalReleaseRoot.Equals($projectCurrentReleaseRoot, [StringComparison]::OrdinalIgnoreCase)
Assert-NoReparsePathChain -LiteralPath $finalReleaseRoot
$cacheRoot = Join-Path $projectRoot '.cache\release'
Ensure-OwnedBuildDirectory -Root $cacheRoot -Marker '.daguandan-release-cache' -Identity 'guandan.release-build-cache/1'
$cacheLockPath = Join-Path $cacheRoot '.build.lock'
Assert-NoReparsePathChain -LiteralPath $cacheLockPath
try { $cacheLock = [IO.File]::Open($cacheLockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None) }
catch { throw "Another release build is active or the build lock is unavailable: $cacheLockPath" }
$previousRoot = "$finalReleaseRoot.previous"
if (-not (Test-Path -LiteralPath $finalReleaseRoot) -and (Test-Path -LiteralPath $previousRoot)) {
    Assert-ManagedReleaseRoot -Root $previousRoot
    Move-Item -LiteralPath $previousRoot -Destination $finalReleaseRoot
    Write-Host 'Restored the previous release after an interrupted publication.'
}
if (Test-Path -LiteralPath $finalReleaseRoot) {
    if (-not $OverwriteExisting) { throw "ReleaseRoot exists; explicit overwrite permission is required: $finalReleaseRoot" }
    Assert-ManagedReleaseRoot -Root $finalReleaseRoot
}
$stagingBase = Join-Path (Split-Path -Parent $finalReleaseRoot) '.staging'
Ensure-OwnedBuildDirectory -Root $stagingBase -Marker '.daguandan-release-staging' -Identity 'guandan.release-staging/1'
$stagingName = [IO.Path]::GetFileName($finalReleaseRoot) + '-' + [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$stagingReleaseRoot = Resolve-ManagedChildPath -Root $stagingBase -Child (Join-Path $stagingBase $stagingName)
[IO.Directory]::CreateDirectory($stagingReleaseRoot) | Out-Null
[IO.File]::WriteAllText((Join-Path $stagingReleaseRoot '.daguandan-release-root'), 'guandan.package-release-root/2', [Text.UTF8Encoding]::new($false))
Assert-NoReparseTree -LiteralPath $wheelhouseRoot
if (-not $isProjectCurrentRelease) {
    Assert-DisjointRoots -First $finalReleaseRoot -Second $projectRoot -Description "ReleaseRoot and project root"
}
Assert-DisjointRoots -First $wheelhouseRoot -Second $projectRoot -Description "WheelhouseRoot and project root"
Assert-DisjointRoots -First $finalReleaseRoot -Second $wheelhouseRoot -Description "ReleaseRoot and WheelhouseRoot"
Assert-DisjointRoots -First $stagingReleaseRoot -Second $wheelhouseRoot -Description "Staging and WheelhouseRoot"

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
    (Join-Path $projectRoot "scripts\release_build_cache.py"),
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

Write-Host "Verifying committed locks and the external wheelhouse..."
Invoke-PythonCommand $bootstrapPython `
    (Join-Path $projectRoot "scripts\verify_release_inputs.py") `
    --project-root $projectRoot `
    --wheelhouse $wheelhouseRoot `
    --python $bootstrapPython

[System.IO.Directory]::CreateDirectory($stagingReleaseRoot) | Out-Null
Assert-NoReparsePathChain -LiteralPath $stagingReleaseRoot
$ownershipMarkerPath = Join-Path $stagingReleaseRoot ".daguandan-release-root"
[System.IO.File]::WriteAllText(
    $ownershipMarkerPath,
    "guandan.package-release-root/2`r`n",
    [System.Text.UTF8Encoding]::new($false)
)

$distPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "dist")
$workPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "build")
$specPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "spec")
$payloadPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "payload")
$buildEnvPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "build-env")
$script:tempPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "temp")
$script:pyinstallerConfigPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "pyinstaller-config")
$bundlePath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $distPath $appName)
$archivePath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "DaguandanAssistant.zip")
$archiveChecksumPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child "$archivePath.sha256"
$releaseRecordPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "DaguandanAssistant.release.json")
$sourceIdentityPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "source_identity.json")
$releaseInputAuditPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "release_input_audit.json")
$bootstrapAuditPath = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "bootstrap_python_audit.json")
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

Start-BuildStage 2 "Validate or create isolated build environment"
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
$cacheHelper = Join-Path $projectRoot 'scripts\release_build_cache.py'
if ($script:cacheEnabled) {
    $keyPath = Join-Path $script:tempPath 'cache_keys.json'
    Invoke-PythonCommand $bootstrapPython $cacheHelper key --project-root $projectRoot --python-root $script:pythonBaseRoot --output $keyPath | Out-Null
    $keys = Get-Content -LiteralPath $keyPath -Raw | ConvertFrom-Json
    if ($keys.environment_key -notmatch '^[0-9a-f]{64}$' -or $keys.work_key -notmatch '^[0-9a-f]{64}$') { throw 'Invalid cache keys.' }
    $buildEnvPath = Join-Path $cacheRoot ("envs\" + $keys.environment_key.Substring(0,20) + '\env')
    $cachedWorkRoot = Join-Path $cacheRoot ("work\" + $keys.work_key.Substring(0,20) + '\work')
    $envDecisionPath = Join-Path $script:tempPath 'environment_cache.json'
    Invoke-PythonCommand $bootstrapPython $cacheHelper inspect --directory $buildEnvPath --kind environment --key $keys.environment_key --output $envDecisionPath | Out-Null
    $decision = Get-Content -LiteralPath $envDecisionPath -Raw | ConvertFrom-Json
    $script:environmentCacheHit = [bool]$decision.hit
    Write-Host ("Environment cache: {0} ({1})" -f $(if ($decision.hit) {'HIT'} else {'MISS'}), $decision.reason)
    if (-not $script:environmentCacheHit) { Reset-OwnedCacheDirectory -Directory $buildEnvPath -CacheRoot $cacheRoot }
    $workDecisionPath = Join-Path $script:tempPath 'work_cache.json'
    Invoke-PythonCommand $bootstrapPython $cacheHelper inspect --directory $cachedWorkRoot --kind work --key $keys.work_key --output $workDecisionPath | Out-Null
    $workDecision = Get-Content -LiteralPath $workDecisionPath -Raw | ConvertFrom-Json
    $script:workCacheHit = [bool]$workDecision.hit -and $script:environmentCacheHit
    Write-Host ("PyInstaller work cache: {0} ({1})" -f $(if ($script:workCacheHit) {'HIT'} else {'MISS'}), $workDecision.reason)
    if (-not $script:workCacheHit) { Reset-OwnedCacheDirectory -Directory $cachedWorkRoot -CacheRoot $cacheRoot }
    $workPath = Join-Path $cachedWorkRoot 'build'
    $specPath = Join-Path $cachedWorkRoot 'spec'
    $script:pyinstallerConfigPath = Join-Path $cachedWorkRoot 'pyinstaller-config'
    foreach ($directory in @($workPath, $specPath, $script:pyinstallerConfigPath)) { [IO.Directory]::CreateDirectory($directory) | Out-Null }
}
if (-not $script:environmentCacheHit) {
    Invoke-CleanPython -Python $bootstrapPython -ScriptsPath (Split-Path -Parent $bootstrapPython) -NoSite -Arguments @('-m','venv',$buildEnvPath)
    $buildPython = Join-Path $buildEnvPath 'Scripts\python.exe'
    if (-not [IO.File]::Exists($buildPython)) { throw 'Build environment did not create python.exe.' }
    Invoke-CleanPython $buildPython (Join-Path $buildEnvPath 'Scripts') `
        -m pip install --isolated --no-index --no-cache-dir --no-compile --disable-pip-version-check --require-hashes `
        --ignore-requires-python --no-deps --find-links $wheelhouseRoot -r (Join-Path $projectRoot 'requirements-release.lock')
} else {
    $buildPython = Join-Path $buildEnvPath 'Scripts\python.exe'
    Write-Host 'Validated environment reused; skipping virtualenv creation and dependency installation.' -ForegroundColor Green
}
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") `
    (Join-Path $projectRoot "scripts\verify_release_inputs.py") `
    --project-root $projectRoot `
    --wheelhouse $wheelhouseRoot `
    --python $buildPython `
    --verify-installed `
    --output $releaseInputAuditPath

# RLCard expands packaged jsondata.zip on its first import. Materialize that
# deterministic data BEFORE collect-data runs, not halfway through Analysis.
# Otherwise the next identical build sees a changed _input_datas and rebuilds.
Write-Host "Preparing deterministic RLCard data before module collection..."
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") -c "import rlcard.envs"

Start-BuildStage 3 "Copy fresh immutable seed resources"
$payloadProfile = Join-Path $payloadPath "data\profiles\tencent_daguandan"
[System.IO.Directory]::CreateDirectory((Join-Path $payloadProfile "models\danzero")) | Out-Null
Copy-Item -LiteralPath (Join-Path $profileSource "profile.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "regions_config.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "templates_config.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "templates") -Destination $payloadProfile -Recurse
Copy-Item -LiteralPath $modelSource -Destination (Join-Path $payloadProfile "models\best.npz")
Copy-Item -LiteralPath $danzeroWeightsSource -Destination (Join-Path $payloadProfile "models\danzero\q_network.ckpt")

Start-BuildStage 4 "Build frozen application into fresh staging directory"
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
if (-not $script:workCacheHit) { $pyinstallerArguments += '--clean' }
foreach ($module in $liveV2HiddenImports) {
    $pyinstallerArguments += @("--hidden-import", $module)
}
$pyinstallerArguments += (Join-Path $projectRoot "run.py")
Invoke-CleanPython $buildPython (Join-Path $buildEnvPath "Scripts") @pyinstallerArguments
$executablePath = Join-Path $bundlePath "$appName.exe"
if (-not [System.IO.File]::Exists($executablePath)) {
    throw "PyInstaller did not produce $appName.exe."
}

Start-BuildStage 5 "Add resources and audit native dependencies"
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

Start-BuildStage 6 "Create and verify build manifest"
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

Start-BuildStage 7 "Create archive, checksum, and release record"
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
if ($script:cacheEnabled) {
    Invoke-PythonCommand $bootstrapPython $cacheHelper seal --directory $buildEnvPath --kind environment --key $keys.environment_key --output (Join-Path $script:tempPath 'environment_seal.json') | Out-Null
    Invoke-PythonCommand $bootstrapPython $cacheHelper seal --directory $cachedWorkRoot --kind work --key $keys.work_key --output (Join-Path $script:tempPath 'work_seal.json') | Out-Null
}
if ($Compact) {
    foreach ($managedDirectory in @($distPath, $payloadPath, $script:tempPath)) {
        $safe = Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child $managedDirectory
        if (Test-Path -LiteralPath $safe) { Assert-NoReparseTree -LiteralPath $safe; Remove-Item -LiteralPath $safe -Recurse -Force }
    }
}
Start-BuildStage 8 'Publish validated release (previous release retained)'
Write-BuildMetrics -Root $stagingReleaseRoot -Status 'validated'
Publish-ManagedRelease -Stage $stagingReleaseRoot -Destination $finalReleaseRoot
$stagingReleaseRoot = $finalReleaseRoot
Complete-BuildStage
Write-BuildMetrics -Root $stagingReleaseRoot -Status 'passed'
Write-Host ("Complete bundle: {0}" -f (Join-Path $stagingReleaseRoot "dist\$appName")) -ForegroundColor Green
Write-Host ("Archive: {0}" -f (Join-Path $stagingReleaseRoot 'DaguandanAssistant.zip')) -ForegroundColor Green
Write-Host ("Build timings: {0}" -f (Join-Path $stagingReleaseRoot 'build_metrics.json')) -ForegroundColor Green
Write-Host ("Total: {0:N1}s; uncompressed size: {1:N1} MB" -f $script:buildWatch.Elapsed.TotalSeconds, $size) -ForegroundColor Green
} catch {
    Complete-BuildStage
    if ($stagingReleaseRoot -and (Test-Path -LiteralPath $stagingReleaseRoot)) {
        try { Write-BuildMetrics -Root $stagingReleaseRoot -Status 'failed' -Failure $_.Exception.Message } catch { Write-Warning 'Could not write failure timing report.' }
    }
    throw
} finally {
    if ($null -ne $cacheLock) { $cacheLock.Dispose() }
}
