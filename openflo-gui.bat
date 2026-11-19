@echo off
REM ============================================================
REM  Quick-launch the OpenFlo gate-editor GUI on Windows.
REM  Double-click this file, or run it from a terminal.
REM
REM  scripts\launch-gui.ps1 starts it via PowerShell's
REM  Start-Process with stdio redirected to a log file. This
REM  matters: under pythonw (no console) a detached process with
REM  NO stdio handles aborts on startup ("nothing opens") - giving
REM  it real handles (the log) fixes that.
REM
REM  Before launching, it checks that the .venv's Python can
REM  start. A .venv records where its base Python lives; when
REM  that Python is gone (uninstalled, upgraded or moved by uv
REM  or the installer, or the folder was copied from another
REM  computer), pythonw.exe fails with the Python Launcher's
REM  "Unable to create process" box. The check runs the console
REM  python.exe instead, which fails quietly; the broken .venv is
REM  then renamed to .venv-broken-N and setup.bat makes a new one.
REM  If that fails too, a window says what to do, in text that
REM  can be copied. When only
REM  pythonw.exe cannot start (an Application Control policy can
REM  block it and allow python.exe), launch-gui.ps1 uses python.exe.
REM ============================================================
setlocal
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv\Scripts"

REM Make the src-layout package importable even without `pip install`.
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"

REM A file to open (a double-clicked .flowsession / .flowproject; see
REM setup.bat filetypes) rides in the environment, so its spaces and
REM braces need no escaping on the PowerShell command line.
set "OPENFLO_OPEN=%~1"

if not exist "%VENV%\python.exe" goto :setup
"%VENV%\python.exe" -c "import sys" >nul 2>&1 && goto :launch

echo [OpenFlo] The Python this copy of OpenFlo was set up with cannot start.
echo [OpenFlo] Setting OpenFlo up again; the old environment is kept as .venv-broken-*.
set "BROKEN=.venv-broken-%RANDOM%"
ren "%ROOT%.venv" "%BROKEN%" >nul 2>&1
if exist "%ROOT%.venv" (
    set "OPENFLO_MSG=OpenFlo could not start: the Python in its .venv folder cannot run, and the folder could not be renamed (is OpenFlo or a terminal using it?). Close OpenFlo, delete the .venv folder in the OpenFlo folder (below) and open OpenFlo again."
    goto :fail
)

:setup
REM First launch, or a rebuilt .venv: install (visible console), then continue.
echo [OpenFlo] Installing OpenFlo's Python and dependencies, please wait...
call "%ROOT%setup.bat" auto
set "OPENFLO_MSG=OpenFlo could not start: setting up its Python failed. Run setup.bat in the OpenFlo folder (below) to see why."
if not exist "%VENV%\python.exe" goto :fail
"%VENV%\python.exe" -c "import sys" >nul 2>&1 || goto :fail

:launch
REM launch-gui.ps1 starts the venv's console-less pythonw.exe, or its
REM python.exe when pythonw cannot start (see that file).
set "OPENFLO_ROOT=%ROOT%"
set "OPENFLO_VENV=%VENV%"
set "OPENFLO_LOG=%TEMP%\openflo-gui.log"
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%ROOT%scripts\launch-gui.ps1"
if not errorlevel 1 goto :started
set "OPENFLO_MSG=OpenFlo could not start its Python. What went wrong is in openflo-gui.log.launch in your TEMP folder."
goto :fail

:started

endlocal
exit /b 0

:fail
echo [OpenFlo] %OPENFLO_MSG%
set "OPENFLO_ROOT=%ROOT%"
REM launch-error.ps1 reads the text from the environment (no quoting games),
REM shows it in a window where it can be selected and copied, and saves it
REM to %TEMP%\openflo-launch-error.txt.
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%ROOT%scripts\launch-error.ps1"
REM When the window cannot open either (a policy that keeps PowerShell from
REM loading Windows Forms), show the saved text and keep this console open.
if errorlevel 1 (
    start "" notepad "%TEMP%\openflo-launch-error.txt"
    pause
)
endlocal
exit /b 1
