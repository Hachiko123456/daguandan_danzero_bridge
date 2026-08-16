[CmdletBinding()]
param()

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

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$appName = "DaguandanAssistant"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$profileSource = Join-Path $projectRoot "data\profiles\tencent_daguandan"
$modelSource = Join-Path $profileSource "models\best.npz"
$danzeroWeightsSource = Join-Path $projectRoot "src\daguandan_bridge\danzero\_vendor\guandan_rlcard\baselines\danzero\q_network.ckpt"
$releaseRoot = Join-Path $projectRoot "release"
$distPath = Join-Path $releaseRoot "dist"
$workPath = Join-Path $releaseRoot "build"
$specPath = Join-Path $releaseRoot "spec"
$payloadPath = Join-Path $releaseRoot "payload"
$bundlePath = Join-Path $distPath $appName
$archivePath = Join-Path $releaseRoot "DaguandanAssistant.zip"

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
    (Join-Path $projectRoot "release_assets\MODEL_REPLACEMENT.txt"),
    (Join-Path $projectRoot "release_assets\Run_FableDan_Fixed_Benchmark.bat")
)
foreach ($requiredFile in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $requiredFile)) {
        throw "Required release file is missing: $requiredFile"
    }
}

Write-Host "[1/4] Installing or updating PyInstaller..." -ForegroundColor Cyan
Invoke-ProjectPython -m pip install --upgrade "PyInstaller==6.21.0"

Write-Host "[2/4] Preparing the minimum runtime data..." -ForegroundColor Cyan
foreach ($path in @($distPath, $workPath, $specPath, $payloadPath)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}

$payloadProfile = Join-Path $payloadPath "data\profiles\tencent_daguandan"
New-Item -ItemType Directory -Force -Path "$payloadProfile\models\danzero" | Out-Null
Copy-Item -LiteralPath (Join-Path $profileSource "profile.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "regions_config.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "templates_config.json") -Destination $payloadProfile
Copy-Item -LiteralPath (Join-Path $profileSource "templates") -Destination $payloadProfile -Recurse
Copy-Item -LiteralPath $modelSource -Destination (Join-Path $payloadProfile "models\best.npz")
Copy-Item -LiteralPath $danzeroWeightsSource -Destination (Join-Path $payloadProfile "models\danzero\q_network.ckpt")

Write-Host "[3/4] Building the portable Windows release..." -ForegroundColor Cyan
$pyinstallerArguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--noconsole",
    "--name", $appName,
    "--icon", (Join-Path $projectRoot "app.ico"),
    "--paths", (Join-Path $projectRoot "src"),
    "--distpath", $distPath,
    "--workpath", $workPath,
    "--specpath", $specPath,
    "--collect-submodules", "daguandan_bridge",
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

Write-Host "[4/4] Copying replaceable models and templates, then creating the archive..." -ForegroundColor Cyan
Copy-Item -LiteralPath (Join-Path $payloadPath "data") -Destination $bundlePath -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot "app.ico") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\MODEL_REPLACEMENT.txt") -Destination $bundlePath
Copy-Item -LiteralPath (Join-Path $projectRoot "release_assets\Run_FableDan_Fixed_Benchmark.bat") -Destination $bundlePath
New-Item -ItemType Directory -Force -Path (Join-Path $bundlePath "logs") | Out-Null

$bundledFableDanModel = Join-Path $bundlePath "data\profiles\tencent_daguandan\models\best.npz"
$bundledDanZeroModel = Join-Path $bundlePath "data\profiles\tencent_daguandan\models\danzero\q_network.ckpt"
foreach ($model in @($bundledFableDanModel, $bundledDanZeroModel)) {
    if (-not (Test-Path -LiteralPath $model)) {
        throw "Expected replaceable model missing from release: $model"
    }
}

$archiveCreated = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
    try {
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

$size = (Get-ChildItem -LiteralPath $bundlePath -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Complete: {0}" -f $bundlePath) -ForegroundColor Green
Write-Host ("Archive: {0}" -f $archivePath) -ForegroundColor Green
Write-Host ("Uncompressed size: {0:N1} MB" -f $size) -ForegroundColor Green
