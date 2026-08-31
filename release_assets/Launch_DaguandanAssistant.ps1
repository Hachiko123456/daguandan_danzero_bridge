[CmdletBinding()]
param(
    [Parameter()]
    [string] $RuntimeRoot = "",

    [Parameter()]
    [switch] $VerifyOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$legacyMutablePrefixes = @(
    "logs/**",
    "reports/**",
    "diagnostics/**",
    "data/profiles/*/sessions/**",
    "data/profiles/*/screenshots/**",
    "data/profiles/*/diagnostics/**",
    "data/profiles/*/truth_log_batch_reports/**",
    "data/profiles/*/models/benchmarks/**"
)

function Get-ExistingPathAttributes {
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
    try { return [System.IO.File]::GetAttributes($LiteralPath) }
    catch [System.IO.FileNotFoundException] { return $null }
    catch [System.IO.DirectoryNotFoundException] { return $null }
}

function Assert-NoReparsePathChain {
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
    $current = [System.IO.Path]::GetFullPath($LiteralPath)
    while (-not [string]::IsNullOrWhiteSpace($current)) {
        $attributes = Get-ExistingPathAttributes -LiteralPath $current
        if (
            $null -ne $attributes -and
            ($attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
        ) { throw "Managed launch path traverses a reparse point: $current" }
        $parent = [System.IO.Directory]::GetParent($current)
        if ($null -eq $parent -or $parent.FullName -eq $current) { break }
        $current = $parent.FullName
    }
}

function Assert-NoReparseTree {
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
    Assert-NoReparsePathChain -LiteralPath $LiteralPath
    $pending = [System.Collections.Generic.Stack[System.IO.DirectoryInfo]]::new()
    $pending.Push([System.IO.DirectoryInfo]::new($LiteralPath))
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($entry in $directory.EnumerateFileSystemInfos()) {
            if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Installed release contains a reparse point: $($entry.FullName)"
            }
            if (($entry.Attributes -band [System.IO.FileAttributes]::Directory) -ne 0) {
                $pending.Push([System.IO.DirectoryInfo] $entry)
            }
        }
    }
}

function Assert-SafeSegment {
    param(
        [Parameter(Mandatory = $true)][string] $Value,
        [Parameter(Mandatory = $true)][string] $Field
    )
    if (
        $Value.Length -lt 1 -or $Value.Length -gt 128 -or
        $Value -notmatch '^[A-Za-z0-9._-]+$' -or
        $Value -eq '.' -or $Value -eq '..' -or
        $Value.EndsWith('.') -or $Value.EndsWith(' ') -or
        $Value -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$'
    ) { throw "$Field is not a safe single path segment." }
}

function Resolve-SafeRelative {
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)][string] $Relative,
        [Parameter(Mandatory = $true)][string] $Field
    )
    if (
        [string]::IsNullOrWhiteSpace($Relative) -or
        $Relative.Contains('\') -or $Relative.Contains(':') -or
        $Relative.StartsWith('/')
    ) { throw "$Field is not a portable relative path." }
    $parts = $Relative.Split('/')
    if ($parts.Count -eq 0 -or ($parts | Where-Object { $_ -eq '' -or $_ -eq '.' -or $_ -eq '..' })) {
        throw "$Field contains an unsafe path segment."
    }
    $rootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    $candidate = $rootPath
    foreach ($part in $parts) { $candidate = Join-Path $candidate $part }
    $candidate = [System.IO.Path]::GetFullPath($candidate)
    if (-not $candidate.StartsWith("$rootPath\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Field escaped its managed root."
    }
    Assert-NoReparsePathChain -LiteralPath $candidate
    return $candidate
}

function Read-JsonObject {
    param(
        [Parameter(Mandatory = $true)][string] $LiteralPath,
        [Parameter(Mandatory = $true)][string] $Label
    )
    if (-not [System.IO.File]::Exists($LiteralPath)) { throw "$Label is missing." }
    try { return Get-Content -LiteralPath $LiteralPath -Raw -Encoding UTF8 | ConvertFrom-Json }
    catch { throw "$Label is not valid UTF-8 JSON." }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
    $stream = [System.IO.File]::OpenRead($LiteralPath)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $algorithm.ComputeHash($stream)
        return ([System.BitConverter]::ToString($bytes)).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
}

function Get-RelativePortablePath {
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)][string] $Path
    )
    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $pathFull = [System.IO.Path]::GetFullPath($Path)
    $rootUri = [System.Uri]::new($rootFull)
    $pathUri = [System.Uri]::new($pathFull)
    return [System.Uri]::UnescapeDataString($rootUri.MakeRelativeUri($pathUri).ToString()).Replace('\', '/')
}

function Test-LegacyMutablePath {
    param([Parameter(Mandatory = $true)][string] $Relative)
    $portable = $Relative.Replace('\', '/')
    if ($portable -match '^(?i:logs|reports|diagnostics)/.+') { return $true }
    if ($portable -match '^(?i:data/profiles/[^/]+/(sessions|screenshots|diagnostics|truth_log_batch_reports))/.+') {
        return $true
    }
    if ($portable -match '^(?i:data/profiles/[^/]+/models/benchmarks)/.+') { return $true }
    return $false
}

function Assert-ManifestFile {
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)] $Record,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][System.Collections.Generic.HashSet[string]] $Declared
    )
    $relative = [string] $Record.path
    $path = Resolve-SafeRelative -Root $Root -Relative $relative -Field "manifest file"
    if (-not [System.IO.File]::Exists($path)) {
        throw "A build-manifest file is missing: $relative"
    }
    if (([System.IO.FileInfo]::new($path)).Length -ne [Int64] $Record.bytes) {
        throw "A build-manifest file size changed: $relative"
    }
    if ((Get-Sha256 -LiteralPath $path) -ne ([string] $Record.sha256).ToLowerInvariant()) {
        throw "A build-manifest file hash changed: $relative"
    }
    if (-not $Declared.Add($relative.Replace('\', '/'))) {
        throw "The build manifest contains a duplicate path: $relative"
    }
}

if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    if (-not [string]::IsNullOrWhiteSpace($env:DAGUANDAN_DATA_ROOT)) {
        $RuntimeRoot = $env:DAGUANDAN_DATA_ROOT
    }
    else { $RuntimeRoot = Join-Path $env:LOCALAPPDATA "DaguandanAssistant" }
}
$runtimePath = [System.IO.Path]::GetFullPath($RuntimeRoot)
$filesystemRoot = [System.IO.Path]::GetPathRoot($runtimePath)
if ($runtimePath.TrimEnd('\') -eq $filesystemRoot.TrimEnd('\')) {
    throw "RuntimeRoot cannot be a filesystem root."
}
Assert-NoReparsePathChain -LiteralPath $runtimePath
$installRoot = Join-Path $runtimePath "install"
$marker = Join-Path $installRoot ".daguandan-install-root.json"
$activePath = Join-Path $installRoot "active.json"
if (-not [System.IO.File]::Exists($marker) -or -not [System.IO.File]::Exists($activePath)) {
    throw "No verified active DaguandanAssistant release is installed."
}
$active = Read-JsonObject -LiteralPath $activePath -Label "active release pointer"
if ($active.schema -ne "guandan.active-release/1") { throw "The active release pointer schema is invalid." }
$releaseId = [string] $active.release_id
$versionDirectory = [string] $active.version_directory
Assert-SafeSegment -Value $releaseId -Field "release_id"
Assert-SafeSegment -Value $versionDirectory -Field "version_directory"
if ($releaseId -ne $versionDirectory) { throw "The active release id and version directory disagree." }
$versionsRoot = Join-Path $installRoot "versions"
$versionRoot = Join-Path $versionsRoot $versionDirectory
Assert-NoReparseTree -LiteralPath $versionRoot
$receiptPath = Join-Path $versionRoot "install_receipt.json"
$receipt = Read-JsonObject -LiteralPath $receiptPath -Label "install receipt"
if (
    ([string] $receipt.release_id) -ne $releaseId -or
    ([string] $receipt.executable_relative) -ne ([string] $active.executable_relative)
) { throw "The active pointer disagrees with its immutable install receipt." }
$executable = Resolve-SafeRelative -Root $versionRoot -Relative ([string] $receipt.executable_relative) -Field "executable_relative"
if (-not [System.IO.File]::Exists($executable)) { throw "The active executable is missing." }
if ((Get-Sha256 -LiteralPath $executable) -ne ([string] $receipt.executable_sha256).ToLowerInvariant()) {
    throw "The active executable failed its install-receipt hash check."
}
$bundleRoot = Split-Path -Parent $executable

if ($receipt.schema -eq "guandan.installed-release/1") {
    $manifestPath = Resolve-SafeRelative -Root $versionRoot -Relative ([string] $receipt.manifest_relative) -Field "manifest_relative"
    if ((Get-Sha256 -LiteralPath $manifestPath) -ne ([string] $receipt.manifest_sha256).ToLowerInvariant()) {
        throw "The installed build manifest hash changed."
    }
    $manifest = Read-JsonObject -LiteralPath $manifestPath -Label "build manifest"
    if (
        $manifest.schema -ne "guandan.build-manifest/1" -or
        ([string] $manifest.build_id) -ne ([string] $active.build_id) -or
        ([string] $manifest.build_id) -ne ([string] $receipt.build_id)
    ) { throw "The build manifest identity disagrees with the active release." }
    $declared = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($record in $manifest.bundle_tree.files) {
        Assert-ManifestFile -Root $bundleRoot -Record $record -Declared $declared
    }
    if ($declared.Count -ne [int] $manifest.bundle_tree.file_count) {
        throw "The build manifest file count is inconsistent."
    }
    $manifestFullPath = [System.IO.Path]::GetFullPath($manifestPath)
    foreach ($file in [System.IO.Directory]::EnumerateFiles($bundleRoot, '*', [System.IO.SearchOption]::AllDirectories)) {
        if ([System.IO.Path]::GetFullPath($file) -eq $manifestFullPath) { continue }
        $relative = Get-RelativePortablePath -Root $bundleRoot -Path $file
        if (-not $declared.Contains($relative)) { throw "The installed bundle contains an undeclared file: $relative" }
    }
    $critical = @(
        "DaguandanAssistant.exe",
        "native_dependency_audit.json",
        "data/profiles/tencent_daguandan/profile.json",
        "data/profiles/tencent_daguandan/regions_config.json",
        "data/profiles/tencent_daguandan/templates_config.json",
        "data/profiles/tencent_daguandan/models/best.npz",
        "data/profiles/tencent_daguandan/models/danzero/q_network.ckpt"
    )
    foreach ($relative in $critical) {
        if (-not $declared.Contains($relative)) { throw "Critical release resource is undeclared: $relative" }
    }
    if (-not ($declared | Where-Object { $_.StartsWith('data/profiles/tencent_daguandan/templates/', [System.StringComparison]::OrdinalIgnoreCase) })) {
        throw "The release manifest contains no recognition templates."
    }
    $nativePath = Resolve-SafeRelative -Root $versionRoot -Relative ([string] $receipt.native_audit_relative) -Field "native_audit_relative"
    if ((Get-Sha256 -LiteralPath $nativePath) -ne ([string] $receipt.native_audit_sha256).ToLowerInvariant()) {
        throw "The native dependency audit hash changed."
    }
    $native = Read-JsonObject -LiteralPath $nativePath -Label "native dependency audit"
    if ($native.schema -ne "guandan.native-dependency-audit/1" -or $native.status -ne "PASS") {
        throw "The native dependency audit is missing or did not pass."
    }
}
elseif ($receipt.schema -eq "guandan.legacy-baseline/1") {
    if ($receipt.baseline -ne $true -or [string]::IsNullOrWhiteSpace([string] $receipt.baseline_auth_sha256)) {
        throw "The legacy baseline is not externally preauthorized."
    }
    $approvedRoot = Resolve-SafeRelative -Root $versionRoot -Relative ([string] $receipt.approved_artifact_relative) -Field "approved_artifact_relative"
    Assert-NoReparseTree -LiteralPath $approvedRoot
    $declared = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($record in $receipt.artifact_files) {
        Assert-ManifestFile -Root $approvedRoot -Record $record -Declared $declared
    }
    foreach ($file in [System.IO.Directory]::EnumerateFiles($approvedRoot, '*', [System.IO.SearchOption]::AllDirectories)) {
        $relative = Get-RelativePortablePath -Root $approvedRoot -Path $file
        if (-not $declared.Contains($relative)) { throw "The legacy baseline contains an undeclared file: $relative" }
    }
    $mutableProperty = $receipt.PSObject.Properties['legacy_mutable_prefixes']
    if ($null -eq $mutableProperty) { throw "The legacy mutable path policy is missing." }
    $receiptMutablePrefixes = @($mutableProperty.Value | ForEach-Object { [string] $_ })
    if ($receiptMutablePrefixes.Count -ne $legacyMutablePrefixes.Count) {
        throw "The legacy mutable path policy changed."
    }
    for ($index = 0; $index -lt $legacyMutablePrefixes.Count; $index++) {
        if (-not [string]::Equals(
            $receiptMutablePrefixes[$index],
            $legacyMutablePrefixes[$index],
            [System.StringComparison]::Ordinal
        )) { throw "The legacy mutable path policy changed." }
    }
    $runDeclared = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($record in $receipt.artifact_files) {
        $relative = [string] $record.path
        if (Test-LegacyMutablePath -Relative $relative) { continue }
        Assert-ManifestFile -Root $bundleRoot -Record $record -Declared $runDeclared
    }
    foreach ($file in [System.IO.Directory]::EnumerateFiles($bundleRoot, '*', [System.IO.SearchOption]::AllDirectories)) {
        $relative = Get-RelativePortablePath -Root $bundleRoot -Path $file
        if (Test-LegacyMutablePath -Relative $relative) { continue }
        if (-not $runDeclared.Contains($relative)) {
            throw "The legacy run copy contains an undeclared immutable file: $relative"
        }
    }
}
else { throw "The installed release receipt schema is unsupported." }

$dataPointerPath = Join-Path $runtimePath "data\v1\active.json"
if ($null -eq $active.data_pointer) {
    if ([System.IO.File]::Exists($dataPointerPath)) {
        throw "The release pointer expects no active data generation, but one is published."
    }
}
else {
    $dataPointer = Read-JsonObject -LiteralPath $dataPointerPath -Label "active data pointer"
    if ((Get-Sha256 -LiteralPath $dataPointerPath) -ne ([string] $active.data_pointer_sha256).ToLowerInvariant()) {
        throw "The active data pointer hash disagrees with the release pointer."
    }
    foreach ($field in @('schema', 'data_schema', 'build_id', 'generation_id')) {
        if ([string] $dataPointer.$field -ne [string] $active.data_pointer.$field) {
            throw "The release/data pointers disagree on $field."
        }
    }
    Assert-SafeSegment -Value ([string] $dataPointer.build_id) -Field "data build_id"
    Assert-SafeSegment -Value ([string] $dataPointer.generation_id) -Field "data generation_id"
    $markerPath = Join-Path $runtimePath ("data\v1\generations\{0}\runtime_layout.json" -f [string] $dataPointer.generation_id)
    Assert-NoReparsePathChain -LiteralPath $markerPath
    if ((Get-Sha256 -LiteralPath $markerPath) -ne ([string] $active.data_marker_sha256).ToLowerInvariant()) {
        throw "The active data generation marker hash changed."
    }
    $generation = Read-JsonObject -LiteralPath $markerPath -Label "data generation marker"
    if (
        $generation.schema -ne "guandan.user-data-generation/1" -or
        [string] $generation.build_id -ne [string] $dataPointer.build_id -or
        [string] $generation.generation_id -ne [string] $dataPointer.generation_id
    ) { throw "The active data generation marker disagrees with its pointer." }
}

if ($VerifyOnly) {
    Write-Output "Verified active DaguandanAssistant release: $releaseId"
    exit 0
}
$env:DAGUANDAN_DATA_ROOT = $runtimePath
& $executable @args
exit $LASTEXITCODE
