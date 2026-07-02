@echo off
REM ============================================================
REM  OpenFlo one-time setup (Windows).
REM  Creates a local .venv and installs OpenFlo + every
REM  dependency, so openflo-gui.bat can launch. Safe to re-run
REM  (idempotent: re-uses the venv, re-installs/updates deps).
REM
REM  Double-click this file, or run it from a terminal.
REM  Pass "auto" as the first argument to skip the closing pause
REM  (openflo-gui.bat does this when it bootstraps on first run).
REM ============================================================
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"

REM Locate a Python: prefer the py launcher (picks the newest 3.x),
REM then a python on PATH. pip enforces the >=3.11 floor on install.
set "PYEXE="
py -3 --version >nul 2>&1 && set "PYEXE=py -3"
if not defined PYEXE python --version >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE (
    echo [OpenFlo setup] No Python found.
    echo                 Install Python 3.11+ from https://python.org and re-run.
    if not "%~1"=="auto" pause
    exit /b 1
)
echo [OpenFlo setup] Using Python: %PYEXE%

if not exist "%ROOT%.venv\Scripts\python.exe" (
    echo [OpenFlo setup] Creating virtual environment in .venv ...
    %PYEXE% -m venv "%ROOT%.venv"
    if errorlevel 1 (
        echo [OpenFlo setup] Failed to create the virtual environment.
        if not "%~1"=="auto" pause
        exit /b 1
    )
)

set "VPY=%ROOT%.venv\Scripts\python.exe"
echo [OpenFlo setup] Upgrading pip ...
"%VPY%" -m pip install --upgrade pip
echo [OpenFlo setup] Installing OpenFlo + dependencies (this can take a few minutes) ...
"%VPY%" -m pip install -e ".[gui]"
if errorlevel 1 (
    echo [OpenFlo setup] Dependency install failed. See the messages above.
    if not "%~1"=="auto" pause
    exit /b 1
)

echo.
echo [OpenFlo setup] Done. Launch OpenFlo with:  openflo-gui.bat
if not "%~1"=="auto" pause
endlocal
