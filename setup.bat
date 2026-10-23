@echo off
REM ============================================================
REM  OpenFlo one-time setup (Windows).
REM  Creates a local .venv and installs OpenFlo + every
REM  dependency, so openflo-gui.bat can launch. Safe to re-run
REM  (idempotent: re-uses the venv, re-installs/updates deps).
REM
REM  No Python? No problem: when no Python 3.11+ is found, setup
REM  uses uv (https://github.com/astral-sh/uv), which downloads
REM  its own Python. A uv already on PATH is used as is;
REM  otherwise a pinned uv release is downloaded into .tools\uv
REM  and its SHA-256 is checked against the hash below before it
REM  is unpacked. A mismatch stops setup.
REM
REM  Double-click this file, or run it from a terminal.
REM  Pass "auto" as the first argument to skip the closing pause
REM  and the shortcut step (openflo-gui.bat does this when it
REM  bootstraps on first run).
REM ============================================================
setlocal EnableExtensions
set "ROOT=%~dp0"
set "MODE=%~1"
cd /d "%ROOT%"

REM ---- Pinned uv release (used only when no Python 3.11+ exists) ----
REM Hashes copied from the release's <asset>.sha256 files. setup.sh pins
REM the same version and lists the same hashes; tests check they agree.
set "UV_VERSION=0.12.18"
set "UV_SHA256_X64=cae6a3bc25239f83dffb467a4b180508d9da23986c04639ebfa44e43e6a84bff"
set "UV_SHA256_ARM64=17f27b1c64eacc757ae603579f116a014881e486c5e79ae81877980d4699e943"
set "UV_BASE_URL=https://github.com/astral-sh/uv/releases/download"
set "UV_PYTHON_VERSION=3.12"

set "VPY=%ROOT%.venv\Scripts\python.exe"
if exist "%VPY%" goto :install

REM ---- Locate a Python 3.11+: the py launcher (newest 3.x), then python ----
set "PYCHECK=import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
set "PYEXE="
py -3 -c "%PYCHECK%" >nul 2>&1 && set "PYEXE=py -3"
if not defined PYEXE python -c "%PYCHECK%" >nul 2>&1 && set "PYEXE=python"
if defined PYEXE goto :make_venv

REM Say why an installed-but-old Python is not used.
py -3 --version >nul 2>&1 && echo [OpenFlo setup] Ignoring "py -3": OpenFlo needs Python 3.11+.
python --version >nul 2>&1 && echo [OpenFlo setup] Ignoring "python" on PATH: OpenFlo needs Python 3.11+.
goto :use_uv

:make_venv
echo [OpenFlo setup] Using Python: %PYEXE%
echo [OpenFlo setup] Creating virtual environment in .venv ...
%PYEXE% -m venv "%ROOT%.venv"
if errorlevel 1 (
    echo [OpenFlo setup] Failed to create the virtual environment.
    goto :fail
)
goto :install

:use_uv
echo [OpenFlo setup] No Python 3.11+ found; using uv to provide Python %UV_PYTHON_VERSION%.
set "UV="
where uv >nul 2>&1 && set "UV=uv"
if defined UV (
    echo [OpenFlo setup] Using uv on PATH.
) else (
    call :fetch_uv
)
if not defined UV goto :fail
"%UV%" python install %UV_PYTHON_VERSION%
if errorlevel 1 (
    echo [OpenFlo setup] uv could not install Python %UV_PYTHON_VERSION%.
    goto :fail
)
REM --seed puts pip in the venv so the install below is the same as ever.
"%UV%" venv --seed --python %UV_PYTHON_VERSION% "%ROOT%.venv"
if errorlevel 1 (
    echo [OpenFlo setup] uv could not create the virtual environment.
    goto :fail
)

:install
if not exist "%VPY%" (
    echo [OpenFlo setup] The virtual environment has no python.exe.
    goto :fail
)
echo [OpenFlo setup] Upgrading pip ...
"%VPY%" -m pip install --upgrade pip
echo [OpenFlo setup] Installing OpenFlo + dependencies (this can take a few minutes) ...
"%VPY%" -m pip install -e ".[gui]"
if errorlevel 1 (
    echo [OpenFlo setup] Dependency install failed. See the messages above.
    goto :fail
)

echo.
echo [OpenFlo setup] Done. Launch OpenFlo with:  openflo-gui.bat
if /i not "%MODE%"=="auto" call :make_shortcuts
if /i not "%MODE%"=="auto" pause
endlocal
exit /b 0

:fail
if /i not "%MODE%"=="auto" pause
endlocal
exit /b 1

REM ============================================================
REM  :fetch_uv - download the pinned uv zip for this machine,
REM  verify its SHA-256, unpack to .tools\uv. Sets UV on success.
REM ============================================================
:fetch_uv
set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"
set "UV_ASSET="
if /i "%ARCH%"=="AMD64" set "UV_ASSET=uv-x86_64-pc-windows-msvc.zip"
if /i "%ARCH%"=="AMD64" set "UV_SHA256=%UV_SHA256_X64%"
if /i "%ARCH%"=="ARM64" set "UV_ASSET=uv-aarch64-pc-windows-msvc.zip"
if /i "%ARCH%"=="ARM64" set "UV_SHA256=%UV_SHA256_ARM64%"
if not defined UV_ASSET (
    echo [OpenFlo setup] No pinned uv build for a %ARCH% processor.
    echo                 Install Python 3.11+ from https://python.org and re-run.
    exit /b 1
)
set "UV_DIR=%ROOT%.tools\uv"
set "UV_STAMP=%ROOT%.tools\uv\uv-%UV_VERSION%.ok"
if exist "%UV_DIR%\uv.exe" if exist "%UV_STAMP%" (
    set "UV=%UV_DIR%\uv.exe"
    exit /b 0
)
set "UV_ZIP=%ROOT%.tools\%UV_ASSET%"
set "UV_URL=%UV_BASE_URL%/%UV_VERSION%/%UV_ASSET%"
echo [OpenFlo setup] Downloading uv %UV_VERSION% (%UV_ASSET%) ...
REM PowerShell reads the paths/hash from the environment (no quoting games).
REM Exit 2 = SHA-256 mismatch: the zip is deleted and nothing is unpacked.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072; New-Item -ItemType Directory -Force -Path (Split-Path -Parent $env:UV_ZIP) | Out-Null; Invoke-WebRequest -UseBasicParsing -Uri $env:UV_URL -OutFile $env:UV_ZIP; $h = (Get-FileHash -Algorithm SHA256 -LiteralPath $env:UV_ZIP).Hash.ToLower(); if ($h -ne $env:UV_SHA256.ToLower()) { Remove-Item -LiteralPath $env:UV_ZIP -Force; Write-Host ('[OpenFlo setup] SHA-256 mismatch for ' + $env:UV_ASSET + ': expected ' + $env:UV_SHA256 + ', got ' + $h + '.'); exit 2 }; Write-Host '[OpenFlo setup] Checksum OK.'; if (Test-Path -LiteralPath $env:UV_DIR) { Remove-Item -LiteralPath $env:UV_DIR -Recurse -Force }; Expand-Archive -LiteralPath $env:UV_ZIP -DestinationPath $env:UV_DIR -Force; Remove-Item -LiteralPath $env:UV_ZIP -Force; exit 0"
if errorlevel 2 (
    echo [OpenFlo setup] Downloaded uv failed the SHA-256 check. Refusing to continue.
    exit /b 1
)
if errorlevel 1 (
    echo [OpenFlo setup] Could not download or unpack uv. Check the network, or
    echo                 install Python 3.11+ from https://python.org and re-run.
    exit /b 1
)
if not exist "%UV_DIR%\uv.exe" (
    echo [OpenFlo setup] uv.exe missing after unpacking.
    exit /b 1
)
>"%UV_STAMP%" echo %UV_VERSION%
set "UV=%UV_DIR%\uv.exe"
exit /b 0

REM ============================================================
REM  :make_shortcuts - Desktop and Start-menu shortcuts to
REM  openflo-gui.bat (interactive runs only). Failure is harmless.
REM ============================================================
:make_shortcuts
set "OPENFLO_ROOT=%ROOT%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ws = New-Object -ComObject WScript.Shell; foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) { $lnk = $ws.CreateShortcut((Join-Path $dir 'OpenFlo.lnk')); $lnk.TargetPath = (Join-Path $env:OPENFLO_ROOT 'openflo-gui.bat'); $lnk.WorkingDirectory = $env:OPENFLO_ROOT; $lnk.WindowStyle = 7; $lnk.Description = 'OpenFlo flow cytometry analysis'; $lnk.Save() }" >nul 2>&1
if errorlevel 1 (
    echo [OpenFlo setup] Could not create shortcuts; launch with openflo-gui.bat.
) else (
    echo [OpenFlo setup] Added "OpenFlo" shortcuts to the Desktop and the Start menu.
)
exit /b 0
