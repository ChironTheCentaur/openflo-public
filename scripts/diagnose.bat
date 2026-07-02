@echo off
REM OpenFlo install health check (Windows). Double-click or run from a prompt.
REM Reports whether dependencies, engines, and core behaviour are within norms.
REM Use this when the GUI won't start or behaves oddly.
setlocal
where python >nul 2>nul && (set PY=python) || (set PY=py)
echo Running OpenFlo diagnostics with %PY% ...
echo.
%PY% -m openflo.diagnostics %*
echo.
echo (exit code %ERRORLEVEL%: 0 = healthy, non-zero = issues found)
pause
endlocal
