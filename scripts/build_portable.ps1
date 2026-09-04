# Assemble a self-contained portable folder that a friend can extract and
# run WITHOUT installing anything on their end - no Python, no venv, no
# Visual C++ redistributable prompt. Everything is bundled.
#
# Output:  dist/portable/                (folder ready for `start.bat`)
#          dist/SermonTranslation.zip    (ship this to the friend)
#
# Layout inside the portable folder:
#   python/                <- copy of the developer's Python 3.14 install
#   sermon/                <- application source
#   models/                <- STT + MT + TTS models (~1 GB)
#   certs/                 <- self-signed cert (regenerated on first run
#                             for the friend's own LAN IP)
#   start.bat              <- what the friend double-clicks
#   README-FRIEND.md       <- one-page instructions
#
# We deliberately don't use a venv. Venvs have hardcoded paths to their base
# interpreter and don't survive being moved to another machine. Instead we
# bundle a full Python install and install every dependency into its
# site-packages directly. Straight-line, no relocation magic.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\build_portable.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host ""
Write-Host "=== Building portable Sermon Translation ===" -ForegroundColor Cyan

# --- prerequisites --------------------------------------------------------
$basePy = "$env:LOCALAPPDATA\Python\pythoncore-3.14-64"
if (-not (Test-Path "$basePy\python.exe")) {
    Write-Host "Base Python 3.14 not found at $basePy" -ForegroundColor Red
    Write-Host "Adjust the path at the top of this script if your Python lives elsewhere."
    exit 1
}
foreach ($p in @("models", "sermon", "run.py", "vocabulary.txt", "phrases.yo.txt")) {
    if (-not (Test-Path $p)) {
        Write-Host "Required item missing: $p (run this from the project root)" -ForegroundColor Red
        exit 1
    }
}

# --- staging tree ---------------------------------------------------------
$staging = "dist\portable"
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
New-Item -ItemType Directory -Path $staging | Out-Null

# 1. Python runtime (~140 MB) ---------------------------------------------
Write-Host "  [1/5] Copying Python 3.14 runtime..." -NoNewline
$stagePy = "$staging\python"
robocopy "$basePy" "$stagePy" /E /NFL /NDL /NJH /NJS /NP /XD "__pycache__" | Out-Null
if ($LASTEXITCODE -ge 8) { Write-Host " robocopy failed"; exit 1 }
Write-Host " done" -ForegroundColor Green

# 2. Install dependencies straight into that Python's site-packages -------
# NOT a venv - just pip --target so everything imports without env-activation.
Write-Host "  [2/5] Installing dependencies into bundled Python..."
$stagePyExe = "$stagePy\python.exe"
& $stagePyExe -m pip install --upgrade pip --quiet 2>&1 | Out-Null
& $stagePyExe -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "  pip install failed" -ForegroundColor Red; exit 1 }
Write-Host "        deps installed" -ForegroundColor Green

# 3. Source code ----------------------------------------------------------
Write-Host "  [3/5] Copying source..." -NoNewline
Copy-Item -Path "sermon" -Destination "$staging\sermon" -Recurse
Copy-Item -Path "run.py", "vocabulary.txt", "phrases.yo.txt" -Destination $staging
# Strip __pycache__ from the copied source
Get-ChildItem -Path "$staging\sermon" -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
Write-Host " done" -ForegroundColor Green

# 4. Models (~1 GB) -------------------------------------------------------
Write-Host "  [4/5] Copying models (this is the big one, ~1 GB)..." -NoNewline
robocopy "models" "$staging\models" /E /NFL /NDL /NJH /NJS /NP `
    /XD "__pycache__" ".cache" "_download" | Out-Null
Write-Host " done" -ForegroundColor Green

# 5. Launcher + README ----------------------------------------------------
Write-Host "  [5/5] Writing launcher + README..."

$startBat = @'
@echo off
REM Sermon Translation - portable launcher.
REM Double-click me. First time takes ~30s to load models. Then open the
REM URLs printed below on the phones you want to use.

setlocal
cd /d "%~dp0"

if not exist "python\python.exe" (
    echo Python runtime not found alongside this launcher.
    echo The portable folder appears incomplete - re-extract the zip.
    pause
    exit /b 1
)

REM Force the app to look for models beside this launcher, not in the
REM developer's original tree.
set SERMON_MODELS_DIR=%cd%\models

python\python.exe run.py --serve --source browser --tls --stt parakeet-0.6b
endlocal
pause
'@
Set-Content -Path "$staging\start.bat" -Value $startBat -Encoding ascii

$readme = @'
# Sermon Translation - portable build

## What this is
Live English -> French / Spanish sermon translation, running entirely on
this laptop. Phones on the same Wi-Fi listen to the translations in real
time. No internet needed once it is running.

## Run it
1. Double-click `start.bat`.
2. First launch takes about 30 seconds while models load. A terminal
   window opens - leave it open, the app stops when you close it.
3. Look at the terminal for two URLs like:

       SERVER   listener page:
                https://192.168.x.x:8443/
       SPEAKER  page for the preacher:
                https://192.168.x.x:8443/speaker

4. **Speaker's phone**: open the /speaker URL, tap Advanced -> Proceed
   (the certificate is self-signed), tap "Tap to start speaking",
   grant mic access.
5. **Listeners' phones**: open the / URL, tap Advanced -> Proceed,
   pick French or Spanish, tap "Start listening".

## When the Windows firewall prompts
Click **Allow** on both Private and Public networks. The app only serves
your local Wi-Fi - nothing goes to the internet.

## To stop
Close the terminal window that opened when you double-clicked start.bat.

## If phones cannot reach the URL
- All devices must be on the SAME Wi-Fi network as this laptop.
- Some routers block devices from talking to each other (called
  "AP isolation"). If yours does, turn on Windows Mobile Hotspot
  (Settings -> Mobile hotspot -> On) and have phones join that instead.

## System requirements
- Windows 10 or 11, 64-bit
- 4 GB free RAM
- 2 GB free disk space
- Nothing else needs installing
'@
Set-Content -Path "$staging\README-FRIEND.md" -Value $readme -Encoding utf8

# Also include the Silero VAD model (used by the Parakeet path)
if (Test-Path "models\silero_vad.onnx") {
    Copy-Item -Path "models\silero_vad.onnx" -Destination "$staging\models\" -Force
}

Write-Host ""
Write-Host "=== Portable folder assembled ===" -ForegroundColor Green
$size = (Get-ChildItem $staging -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("  location: {0}" -f (Resolve-Path $staging))
Write-Host ("  size    : {0:N0} MB" -f $size)

# --- zip it ---------------------------------------------------------------
$zip = "dist\SermonTranslation.zip"
if (Test-Path $zip) { Remove-Item $zip }
Write-Host ""
Write-Host "Compressing to $zip (this takes a couple of minutes)..." -NoNewline
Compress-Archive -Path "$staging\*" -DestinationPath $zip -CompressionLevel Optimal
$zsize = (Get-Item $zip).Length / 1MB
Write-Host " done" -ForegroundColor Green
Write-Host ("  {0:N0} MB - send this to your friend" -f $zsize)
Write-Host ""
Write-Host "Test it locally first by extracting to a fresh folder and running start.bat." -ForegroundColor Yellow
