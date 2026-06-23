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

REM Prefer the venv's console-less pythonw, then system pythonw.
set "PY=%ROOT%.venv\Scripts\pythonw.exe"
if not exist "%PY%" set "PY=pythonw"

set "LOG=%TEMP%\openflo-gui.log"

powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "Start-Process -FilePath '%PY%' -ArgumentList '-m','openflo.gui' -WorkingDirectory '%ROOT%' -RedirectStandardOutput '%LOG%' -RedirectStandardError '%LOG%.err'"

endlocal
