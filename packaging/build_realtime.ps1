param(
    [ValidateSet("OCR", "YOLO")]
    [string]$Variant = "OCR",
    [ValidateSet("cycling", "speed_skating")]
    [string]$SportProfile = "cycling",
    [string]$ModelPath,
    [string]$SourcePath,
    [string]$OcrModelsPath = "$HOME\.paddlex\official_models",
    [string]$FfmpegPath
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Artifacts = Join-Path $RepoRoot "artifacts"
$DistRoot = Join-Path $Artifacts "dist"
$BuildRoot = Join-Path $Artifacts "build"
$Spec = Join-Path $PSScriptRoot "VideoPipeRealtime$Variant.spec"
$AppName = "VideoPipeRealtime$Variant"
$AppDir = Join-Path $DistRoot $AppName
$ResolvedModel = if ($ModelPath) { (Resolve-Path -LiteralPath $ModelPath).Path } else { $null }
$ResolvedSource = if ($SourcePath) { (Resolve-Path -LiteralPath $SourcePath).Path } else { $null }
$ResolvedFfmpeg = if ($FfmpegPath) {
    (Resolve-Path -LiteralPath $FfmpegPath).Path
} else {
    (Get-Command ffmpeg -ErrorAction Stop).Source
}
$env:VIDEOPIPE_FFMPEG = $ResolvedFfmpeg

if ($Variant -eq "OCR") {
    $env:VIDEOPIPE_OCR_MODELS = (Resolve-Path -LiteralPath $OcrModelsPath).Path
}

Push-Location $RepoRoot
try {
    python -m PyInstaller --noconfirm --clean --distpath $DistRoot --workpath $BuildRoot $Spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
    Remove-Item Env:VIDEOPIPE_FFMPEG -ErrorAction SilentlyContinue
}

$ModelArg = $null
if ($ResolvedModel) {
    $ModelName = if ([IO.Path]::GetExtension($ResolvedModel) -eq ".engine") { "best.engine" } else { "best.pt" }
    Copy-Item -LiteralPath $ResolvedModel -Destination (Join-Path $AppDir $ModelName) -Force
    $ModelArg = $ModelName
}

$SourceArg = $null
if ($ResolvedSource) {
    $SourceName = [IO.Path]::GetFileName($ResolvedSource)
    Copy-Item -LiteralPath $ResolvedSource -Destination (Join-Path $AppDir $SourceName) -Force
    $SourceArg = $SourceName
}

if ($ModelArg) {
    $ModeArg = if ($Variant -eq "YOLO") { " --yolo-only" } else { " --ocr-cpu-threads 1" }
    $SourceArgText = if ($SourceArg) { " --source `"$SourceArg`" --auto-start" } else { "" }
    $Launcher = @"
@echo off
cd /d "%~dp0"
set "RACE_DIR=%~1"
if not defined RACE_DIR set "RACE_DIR=RaceData"
$AppName.exe --model "$ModelArg"$SourceArgText --output "%RACE_DIR%" --sport-profile $SportProfile$ModeArg
"@
    Set-Content -LiteralPath (Join-Path $AppDir "Start-$Variant-$SportProfile.cmd") -Value $Launcher -Encoding ASCII
}

Write-Host "Built: $AppDir"
