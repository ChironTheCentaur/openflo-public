@echo off
REM ============================================================
REM  Quick-launch the OpenFlo gate-editor GUI on Windows.
REM  Double-click this file, or run it from a terminal.
REM
REM  Launches via PowerShell's Start-Process with stdio
REM  redirected to a log file. This matters: under pythonw
REM  (no console) a detached process with NO stdio handles
REM  aborts on startup ("nothing opens") - giving it real
REM  handles (the log) fixes that and leaves no console window.
REM ============================================================
setlocal
set "ROOT=%~dp0"

REM Make the src-layout package importable even without `pip install`.
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"

REM First launch with no venv? Bootstrap it (visible console), then continue.
if not exist "%ROOT%.venv\Scripts\pythonw.exe" (
    echo [OpenFlo] First launch - installing dependencies, please wait...
    call "%ROOT%setup.bat" auto
)

REM Prefer the venv's console-less pythonw, then system pythonw.
set "PY=%ROOT%.venv\Scripts\pythonw.exe"
if not exist "%PY%" set "PY=pythonw"

set "LOG=%TEMP%\openflo-gui.log"

REM A file to open (a double-clicked .flowsession / .flowproject; see
REM setup.bat filetypes) rides in the environment, so its spaces and
REM braces need no escaping on the PowerShell command line below.
set "OPENFLO_OPEN=%~1"

powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "$a = @('-m','openflo.gui'); if ($env:OPENFLO_OPEN) { $q = [string][char]34; $a += $q + $env:OPENFLO_OPEN + $q }; Start-Process -FilePath '%PY%' -ArgumentList $a -WorkingDirectory '%ROOT%' -RedirectStandardOutput '%LOG%' -RedirectStandardError '%LOG%.err'"

endlocal
