@echo off
REM ============================================================
REM  OpenFlo one-time setup (Windows).
REM  Creates a local .venv and installs OpenFlo + every
REM  dependency, so openflo-gui.bat can launch. Safe to re-run
REM  (idempotent: re-uses the venv, re-installs/updates deps).
REM
REM  OpenFlo supports Python 3.11 to 3.14. A newer Python (3.15+)
REM  is skipped with a message saying why: FlowUtils, which does
REM  the logicle transforms, has no download for it yet, so pip
REM  would have to compile it and that fails without C build tools.
REM
REM  No Python? No problem: when no Python 3.11 to 3.14 is found, setup
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

REM ---- Supported Pythons ----
REM Oldest and newest supported 3.x minor. setup.sh and the README say the
REM same; tests check the three agree. Raise PY_NEWEST_MINOR when FlowUtils
REM publishes wheels for the next Python (and add it to the py list below).
set "PY_OLDEST_MINOR=11"
set "PY_NEWEST_MINOR=14"
set "PYCHECK=import sys; sys.exit(0 if (3, %PY_OLDEST_MINOR%) <= sys.version_info[:2] <= (3, %PY_NEWEST_MINOR%) else 1)"
set "PYNEW=import sys; sys.exit(0 if sys.version_info[:2] > (3, %PY_NEWEST_MINOR%) else 1)"
set "TOO_NEW_1=OpenFlo supports Python 3.%PY_OLDEST_MINOR% to 3.%PY_NEWEST_MINOR% for now. FlowUtils, which does"
set "TOO_NEW_2=the logicle transforms, has no download for newer Pythons yet, so installing it would need a C compiler."

REM ---- Pinned uv release (used only when no supported Python exists) ----
REM Hashes copied from the release's <asset>.sha256 files. setup.sh pins
REM the same version and lists the same hashes; tests check they agree.
set "UV_VERSION=0.12.18"
set "UV_SHA256_X64=cae6a3bc25239f83dffb467a4b180508d9da23986c04639ebfa44e43e6a84bff"
set "UV_SHA256_ARM64=17f27b1c64eacc757ae603579f116a014881e486c5e79ae81877980d4699e943"
set "UV_BASE_URL=https://github.com/astral-sh/uv/releases/download"
set "UV_PYTHON_VERSION=3.12"

set "VPY=%ROOT%.venv\Scripts\python.exe"
if exist "%VPY%" goto :install

REM ---- Locate a supported Python ----
REM Probing "py -3.14" etc. must never install that Python: the Python
REM install manager does so by default for a version it lacks, with its
REM prompt hidden by >nul, and setup would look hung. setlocal keeps both
REM settings local to this script.
set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"
set "PYLAUNCHER_ALLOW_INSTALL="
REM The py launcher's default (newest 3.x) first; if that is too new, each
REM supported version by name, newest first; then python on PATH.
set "PYEXE="
py -3 -c "%PYCHECK%" >nul 2>&1 && set "PYEXE=py -3"
if not defined PYEXE py -3.14 -c "%PYCHECK%" >nul 2>&1 && set "PYEXE=py -3.14"
if not defined PYEXE py -3.13 -c "%PYCHECK%" >nul 2>&1 && set "PYEXE=py -3.13"
if not defined PYEXE py -3.12 -c "%PYCHECK%" >nul 2>&1 && set "PYEXE=py -3.12"
if not defined PYEXE py -3.11 -c "%PYCHECK%" >nul 2>&1 && set "PYEXE=py -3.11"
if not defined PYEXE python -c "%PYCHECK%" >nul 2>&1 && set "PYEXE=python"
if defined PYEXE goto :make_venv

REM Say why an installed Python is not used: too new, or too old.
py -3 -c "%PYNEW%" >nul 2>&1 && goto :py_launcher_too_new
py -3 --version >nul 2>&1 && echo [OpenFlo setup] Ignoring "py -3": OpenFlo needs Python 3.%PY_OLDEST_MINOR%+.
goto :check_path_python
:py_launcher_too_new
echo [OpenFlo setup] Ignoring "py -3": it is newer than Python 3.%PY_NEWEST_MINOR%.
echo                 %TOO_NEW_1%
echo                 %TOO_NEW_2%
:check_path_python
python -c "%PYNEW%" >nul 2>&1 && goto :path_python_too_new
python --version >nul 2>&1 && echo [OpenFlo setup] Ignoring "python" on PATH: OpenFlo needs Python 3.%PY_OLDEST_MINOR%+.
goto :use_uv
:path_python_too_new
echo [OpenFlo setup] Ignoring "python" on PATH: it is newer than Python 3.%PY_NEWEST_MINOR%.
echo                 %TOO_NEW_1%
echo                 %TOO_NEW_2%
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
echo [OpenFlo setup] No Python 3.%PY_OLDEST_MINOR% to 3.%PY_NEWEST_MINOR% found; using uv to provide Python %UV_PYTHON_VERSION%.
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
REM An existing .venv may have been made with a Python that is too new (or
REM old). Say so plainly instead of letting pip fail on a compiler error.
"%VPY%" -c "%PYCHECK%" >nul 2>&1 && goto :venv_ok
"%VPY%" -c "%PYNEW%" >nul 2>&1 && goto :venv_too_new
echo [OpenFlo setup] The existing .venv uses a Python OpenFlo does not support.
echo                 OpenFlo needs Python 3.%PY_OLDEST_MINOR% to 3.%PY_NEWEST_MINOR%.
echo                 Delete the .venv folder and run setup again.
goto :fail
:venv_too_new
echo [OpenFlo setup] The existing .venv uses a Python newer than 3.%PY_NEWEST_MINOR%.
echo                 %TOO_NEW_1%
echo                 %TOO_NEW_2%
echo                 Delete the .venv folder and run setup again; setup will
echo                 pick a supported Python or fetch one.
goto :fail
:venv_ok
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
    echo                 Install Python 3.11 to 3.14 from https://python.org and re-run.
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
    echo                 install Python 3.11 to 3.14 from https://python.org and re-run.
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
