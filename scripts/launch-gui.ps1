# Starts the OpenFlo GUI from the .venv; called by openflo-gui.bat, which
# sets OPENFLO_ROOT, OPENFLO_VENV (the .venv\Scripts folder) and OPENFLO_LOG.
#
# The venv's pythonw.exe is a small redirector that starts the base
# Python's pythonw.exe (the "home" in pyvenv.cfg). When that base pythonw
# cannot start (an Application Control policy that blocks it but allows
# python.exe, or a missing file), the redirector shows the Python
# Launcher's "Unable to create process" box. So the base pythonw is tried
# first on its own, where a failure is quiet; if it fails, OpenFlo starts
# with the venv's python.exe instead, which opens a console window beside
# it (output still goes to the log).
$ErrorActionPreference = 'Stop'
$venv = $env:OPENFLO_VENV
$py = Join-Path $venv 'pythonw.exe'
try {
    if (-not (Test-Path -LiteralPath $py)) { throw 'no pythonw.exe in the venv' }
    $cfg = Join-Path (Split-Path -Parent $venv) 'pyvenv.cfg'
    $line = Get-Content -LiteralPath $cfg | Where-Object { $_ -match '^\s*home\s*=' } | Select-Object -First 1
    if (-not $line) { throw 'pyvenv.cfg names no home' }
    $baseHome = ($line -replace '^\s*home\s*=\s*', '').Trim()
    $probe = Start-Process -FilePath (Join-Path $baseHome 'pythonw.exe') -ArgumentList '-c', 'pass' -Wait -PassThru -WindowStyle Hidden
    if ($probe.ExitCode -ne 0) { throw "pythonw exited with $($probe.ExitCode)" }
} catch {
    $py = Join-Path $venv 'python.exe'
    Add-Content -LiteralPath ($env:OPENFLO_LOG + '.launch') -Value ("pythonw cannot start ($_); using python.exe")
}
# A file to open (OPENFLO_OPEN, set by the bat) is passed quoted, so its
# spaces stay one argument.
$a = @('-m', 'openflo.gui')
if ($env:OPENFLO_OPEN) { $q = [string][char]34; $a += $q + $env:OPENFLO_OPEN + $q }
try {
    Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $env:OPENFLO_ROOT `
        -RedirectStandardOutput $env:OPENFLO_LOG -RedirectStandardError ($env:OPENFLO_LOG + '.err')
} catch {
    # The bat sees exit code 1 and opens the error window.
    Add-Content -LiteralPath ($env:OPENFLO_LOG + '.launch') -Value ("$py could not start: $_")
    exit 1
}
