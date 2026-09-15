# Phase 1 setup. Run from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Creates .venv, installs the runtime dependencies, and downloads the models.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host ""
Write-Host "=== Neo Sermon - setup ===" -ForegroundColor Cyan
Write-Host ""

# --- 1. locate a real Python (not the Microsoft Store stub) -----------------
$probe = "import sys;print('%d.%d' % sys.version_info[:2])"
$candidates = @(
    @{ exe = "py";     args = @("-3")    },
    @{ exe = "python"; args = @()        }
)

$pyExe = $null; $pyArgs = @(); $pyver = $null
foreach ($c in $candidates) {
    if (-not (Get-Command $c.exe -ErrorAction SilentlyContinue)) { continue }
    # Do not judge by install path: on Windows 11 both `python` and `py` are
    # shims under WindowsApps, and they resolve to a real interpreter once one
    # is installed. Probe instead - the Store placeholder fails this and the
    # real interpreter prints a version.
    $ver = $null
    try { $ver = & $c.exe @($c.args + @("-c", $probe)) 2>$null } catch { continue }
    if ($LASTEXITCODE -eq 0 -and "$ver".Trim() -match "^\d+\.\d+$") {
        $pyExe = $c.exe; $pyArgs = $c.args; $pyver = ("$ver").Trim(); break
    }
}

if (-not $pyExe) {
    Write-Host "No usable Python found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Python 3.11 (64-bit) from:"
    Write-Host "  https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    Write-Host "During install, TICK 'Add python.exe to PATH'."
    Write-Host "Then open a NEW terminal and run this script again."
    exit 1
}
Write-Host "Python $pyver found ($pyExe $pyArgs)" -ForegroundColor Green
if ($pyver -notmatch "^3\.(9|10|11|12|13|14)$") {
    Write-Host "Warning: Python $pyver is outside the tested range (3.9-3.14)." -ForegroundColor Yellow
}

# --- 2. virtual environment -------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv) ..."
    & $pyExe @($pyArgs + @("-m", "venv", ".venv"))
    if ($LASTEXITCODE -ne 0) { Write-Host "venv creation failed" -ForegroundColor Red; exit 1 }
} else {
    Write-Host ".venv already exists, reusing it."
}

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "$venvPy missing" -ForegroundColor Red; exit 1 }

# --- 3. dependencies --------------------------------------------------------
Write-Host ""
Write-Host "Installing dependencies (about 400 MB, a few minutes) ..."
& $venvPy -m pip install --upgrade pip setuptools wheel
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed" -ForegroundColor Red; exit 1 }

# --- 4. models --------------------------------------------------------------
Write-Host ""
Write-Host "Downloading models ..."
& $venvPy scripts\fetch_models.py
if ($LASTEXITCODE -ne 0) { Write-Host "model download failed" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Activate the environment and run the diagnostics:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python run.py doctor"
Write-Host ""
Write-Host "Then start live translation:"
Write-Host "  python run.py"
Write-Host ""
