#requires -version 5.1
<#
Preview by default. -Apply permanently deletes OLD diagnostic incident media.
JSON, logs, session recordings, models, configs, ZIPs and user exports stay intact.
Close the assistant first and back up incidents needed for investigation.
RunsRoot is optional for administrators/tests; it must retain the exact
DaguandanAssistant\diagnostics\runs suffix. Junctions/symlinks are rejected.
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [ValidateRange(1, 1000)][int]$KeepPerRun = 3,
    [string]$RunsRoot = (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'DaguandanAssistant\diagnostics\runs')
)
$ErrorActionPreference = 'Stop'
$cleanRoot = [IO.Path]::GetFullPath($RunsRoot).TrimEnd('\', '/')
if ($cleanRoot -notmatch '[\\/]DaguandanAssistant[\\/]diagnostics[\\/]runs$') {
    throw 'Refusing root: it must end with DaguandanAssistant\diagnostics\runs.'
}
$reparseAttribute = [IO.FileAttributes]::ReparsePoint

function Assert-SafePath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    if ($full -ne $cleanRoot -and -not $full.StartsWith(
        $cleanRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside cleanup root: $full"
    }
    $part = $full
    while ($part) {
        $entry = Get-Item -LiteralPath $part -Force -ErrorAction Stop
        if (($entry.Attributes -band $reparseAttribute) -ne 0) {
            throw "Link or junction rejected: $part"
        }
        $parent = [IO.Directory]::GetParent($part)
        if ($null -eq $parent) { break }
        $part = $parent.FullName
    }
}

function Assert-AppStopped {
    $busy = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $_.Name -like 'DaguandanAssistant*.exe' -or
        (($_.Name -match '^(python|pythonw|pypy)[\d.]*\.exe$') -and
         ([string]::IsNullOrWhiteSpace($_.CommandLine) -or
          $_.CommandLine -match '(?i)(-m\s+daguandan_bridge(?:\s|\.)|[\\/]daguandan_bridge[\\/].*\.py|daguandan_danzero_bridge[\\/]run\.py|(?:^|\s)"?(?:\.[\\/])?run\.py(?:"|\s|$)|(?:run_gui|launch_gui|live_assistant)\.py)'))
    })
    if ($busy.Count -gt 0) {
        throw 'Close the assistant and its Python GUI processes first. No files deleted.'
    }
}

if (-not (Test-Path -LiteralPath $cleanRoot -PathType Container)) {
    Write-Host "No diagnostics directory: $cleanRoot"
    return
}
Assert-SafePath $cleanRoot
if ($Apply) { Assert-AppStopped }
$targets = New-Object 'System.Collections.Generic.List[object]'
$runs = @(Get-ChildItem -LiteralPath $cleanRoot -Directory -Force | Where-Object {
    ($_.Attributes -band $reparseAttribute) -eq 0
})
foreach ($run in $runs) {
    $incidentsPath = Join-Path $run.FullName 'opening\incidents'
    if (-not (Test-Path -LiteralPath $incidentsPath -PathType Container)) { continue }
    Assert-SafePath $incidentsPath
    $finished = @(Get-ChildItem -LiteralPath $incidentsPath -Directory -Force |
        Where-Object {
            $_.Name -match '^OPEN-[A-Za-z0-9-]+$' -and
            ($_.Attributes -band $reparseAttribute) -eq 0 -and
            (Test-Path -LiteralPath (Join-Path $_.FullName 'incident.json') -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName 'opening_evidence.json') -PathType Leaf)
        } | Sort-Object LastWriteTimeUtc -Descending)

    # Newer incidents can reference shared media in an older completed incident.
    # Preserve those owners too so retaining three usable incidents is truthful.
    $protected = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $pending = New-Object 'System.Collections.Generic.Queue[string]'
    foreach ($incident in @($finished | Select-Object -First $KeepPerRun)) {
        if ($protected.Add($incident.Name)) { $pending.Enqueue($incident.Name) }
    }
    while ($pending.Count -gt 0) {
        $owner = $pending.Dequeue()
        $manifestPath = Join-Path (Join-Path $incidentsPath $owner) 'opening_evidence.json'
        Assert-SafePath $manifestPath
        try {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($frame in @($manifest.frames)) {
                $reference = [string]$frame.media_reference.incident_id
                if ($reference -match '^OPEN-[A-Za-z0-9-]+$' -and $protected.Add($reference)) {
                    $refManifest = Join-Path (Join-Path $incidentsPath $reference) 'opening_evidence.json'
                    if (Test-Path -LiteralPath $refManifest -PathType Leaf) { $pending.Enqueue($reference) }
                }
            }
        } catch {
            throw "Cannot verify retained incident media references: $manifestPath; $($_.Exception.Message)"
        }
    }
    foreach ($incident in $finished) {
        if ($protected.Contains($incident.Name)) { continue }
        foreach ($subdir in @('frames', 'roi')) {
            $imageDir = Join-Path $incident.FullName $subdir
            if (-not (Test-Path -LiteralPath $imageDir -PathType Container)) { continue }
            Assert-SafePath $imageDir
            foreach ($file in @(Get-ChildItem -LiteralPath $imageDir -File -Force)) {
                if (($file.Attributes -band $reparseAttribute) -ne 0) { continue }
                if ($file.Extension.ToLowerInvariant() -notin @('.png', '.jpg', '.jpeg', '.bmp', '.avi', '.mp4')) { continue }
                $targets.Add([pscustomobject]@{
                    Path = $file.FullName
                    Bytes = $file.Length
                    Stamp = $file.LastWriteTimeUtc.Ticks
                })
            }
        }
    }
}
[long]$planned = ($targets | Measure-Object Bytes -Sum).Sum
Write-Host ("Root: {0}; candidates: {1} files, {2:N3} GiB" -f $cleanRoot, $targets.Count, ($planned / 1GB))
$targets | Select-Object -First 15 Path, Bytes | Format-Table -AutoSize
if (-not $Apply) {
    Write-Host 'PREVIEW ONLY. Nothing deleted. Use -Apply for permanent deletion (not Recycle Bin).'
    return
}
Assert-AppStopped
[long]$removedBytes = 0
$removedCount = 0
$failedCount = 0
foreach ($target in $targets) {
    try {
        Assert-SafePath $target.Path
        $now = Get-Item -LiteralPath $target.Path -Force
        if ($now.PSIsContainer -or $now.Length -ne $target.Bytes -or
            $now.LastWriteTimeUtc.Ticks -ne $target.Stamp) {
            throw 'File changed after enumeration; deletion refused.'
        }
        Remove-Item -LiteralPath $target.Path -Force -ErrorAction Stop
        $removedBytes += $target.Bytes
        $removedCount++
    } catch {
        $failedCount++
        Write-Warning ("Not deleted: {0}; {1}" -f $target.Path, $_.Exception.Message)
    }
}
Write-Host ("Deleted permanently (not recoverable here): {0} files, {1:N3} GiB; failed/skipped: {2}" -f
    $removedCount, ($removedBytes / 1GB), $failedCount)
if ($failedCount -gt 0) { exit 1 }
