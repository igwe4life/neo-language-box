@echo off
REM Launcher that needs no venv activation and no PowerShell execution policy
REM change. Passes every argument straight through to run.py.
REM
REM   start.bat            live translation
REM   start.bat doctor     full diagnostics
REM   start.bat mic        microphone level meter
REM   start.bat devices    list audio devices
REM   start.bat --model tiny.en

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   The virtual environment is missing.
    echo   Run this first:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
    echo.
    exit /b 1
)

".venv\Scripts\python.exe" run.py %*
endlocal
