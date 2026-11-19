"""openflo-gui.bat starts OpenFlo from a healthy .venv and recovers from a
broken one (its base Python gone: uninstalled, moved by uv or the installer,
or the folder copied from another computer) instead of leaving the user at
the Python Launcher's "Unable to create process" box.

cmd.exe cannot run here, so these read the script; the behavior was checked
on Windows against a .venv whose pythonw an Application Control policy
blocks (see the pull request).
"""
import re
from pathlib import Path

import pytest

BAT = Path(__file__).resolve().parents[1] / 'openflo-gui.bat'
PS1 = BAT.parent / 'scripts' / 'launch-gui.ps1'


def _code() -> list[str]:
    """The script's commands, comments and echo lines left out."""
    lines = [ln.strip() for ln in BAT.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.upper().startswith(('REM', '@ECHO', 'ECHO'))]


def _index(pattern: str) -> int:
    code = _code()
    hits = [i for i, ln in enumerate(code) if re.search(pattern, ln)]
    assert hits, f'openflo-gui.bat has no line matching {pattern!r}'
    return hits[0]


def test_the_venv_python_is_checked_with_the_console_python_before_launch():
    # The console python.exe fails quietly; pythonw.exe would show the
    # launcher's error box.
    check = _index(r'^"%VENV%\\python\.exe" -c "import sys" >nul 2>&1 && goto :launch$')
    assert check < _index(r'^:launch$') < _index(r'launch-gui\.ps1"$')


def test_a_broken_venv_is_renamed_then_set_up_again():
    code = _code()
    check = _index(r'"import sys" >nul 2>&1 && goto :launch$')
    ren = _index(r'^ren "%ROOT%\.venv" "%BROKEN%"')
    assert code[_index(r'^set "BROKEN=')] == 'set "BROKEN=.venv-broken-%RANDOM%"'
    assert check < ren < _index(r'^:setup$') < _index(r'^call "%ROOT%setup\.bat" auto$')
    assert not any(re.match(r'(rd|rmdir|del)\b', ln, re.I) for ln in code), 'never deletes the .venv'
    # A rename that did not happen (folder in use) stops with a message.
    assert re.match(r'^if exist "%ROOT%\.venv" \($', code[ren + 1])


def test_the_rebuilt_venv_is_checked_again_before_launch():
    setup = _index(r'^call "%ROOT%setup\.bat" auto$')
    assert setup < _index(r'^if not exist "%VENV%\\python\.exe" goto :fail$') < _index(r'^:launch$')
    assert setup < _index(r'^"%VENV%\\python\.exe" -c "import sys" >nul 2>&1 \|\| goto :fail$') < _index(r'^:launch$')


def test_the_launch_hands_the_paths_to_launch_gui_ps1():
    code = _code()
    launch = _index(r'^:launch$')
    for ln in ('set "OPENFLO_ROOT=%ROOT%"', 'set "OPENFLO_VENV=%VENV%"',
               'set "OPENFLO_LOG=%TEMP%\\openflo-gui.log"'):
        assert ln in code[launch:], ln
    assert _index(r'-File "%ROOT%scripts\\launch-gui\.ps1"$') > launch
    assert not any(re.search(r'set "PY=pythonw"|set "PY=python"', ln) for ln in code), \
        'a bare pythonw on PATH may be the py launcher or another Python'


def _ps1() -> list[str]:
    lines = [ln.strip() for ln in PS1.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith('#')]


def _ps1_index(pattern: str) -> int:
    hits = [i for i, ln in enumerate(_ps1()) if re.search(pattern, ln)]
    assert hits, f'launch-gui.ps1 has no line matching {pattern!r}'
    return hits[0]


def test_launch_tries_the_base_pythonw_quietly_before_using_the_venv_pythonw():
    # The base pythonw named by pyvenv.cfg's home is started directly, where
    # a policy block or a missing file throws instead of showing the
    # launcher's box; only then is the venv's redirector used.
    code = _ps1()
    assert "$ErrorActionPreference = 'Stop'" in code
    try_ = _ps1_index(r'^try \{$')
    home = _ps1_index(r"-match '\^\\s\*home\\s\*='")
    probe = _ps1_index(r"Start-Process -FilePath \(Join-Path \$baseHome 'pythonw\.exe'\) "
                       r"-ArgumentList '-c', 'pass' -Wait -PassThru")
    exitcheck = _ps1_index(r'if \(\$probe\.ExitCode -ne 0\) \{ throw')
    catch = _ps1_index(r'^\} catch \{$')
    assert try_ < home < probe < exitcheck < catch
    assert code[catch + 1] == "$py = Join-Path $venv 'python.exe'"
    assert _ps1_index(r"^\$py = Join-Path \$venv 'pythonw\.exe'$") < try_


def test_launch_logs_why_it_fell_back_and_redirects_output():
    code = _ps1()
    catch = _ps1_index(r'^\} catch \{$')
    assert 'using python.exe' in code[catch + 2]
    args = _ps1_index(r"^\$a = @\('-m', 'openflo\.gui'\)$")
    start = _ps1_index(r"^Start-Process -FilePath \$py -ArgumentList \$a ")
    assert catch < args < start
    assert '-RedirectStandardOutput $env:OPENFLO_LOG' in code[start + 1]
    assert "-RedirectStandardError ($env:OPENFLO_LOG + '.err')" in code[start + 1]
    # A start that fails is logged and ends the script with 1, so the bat
    # opens the error window instead of closing with nothing on screen.
    assert code[start - 1] == 'try {'
    assert code[start + 2] == '} catch {'
    assert "Add-Content -LiteralPath ($env:OPENFLO_LOG + '.launch')" in code[start + 3]
    assert '-Value ("$py could not start: $_")' in code[start + 3]   # names why
    assert code[start + 4] == 'exit 1'


def test_a_double_clicked_file_reaches_the_app():
    # The file rides in OPENFLO_OPEN, set before any jump, so a first launch
    # (setup) and a normal one both pass it on; the script quotes it.
    code = _code()
    opn = code.index('set "OPENFLO_OPEN=%~1"')
    assert opn < min(i for i, ln in enumerate(code) if 'goto' in ln)
    ps = _ps1()
    args = _ps1_index(r"^\$a = @\('-m', 'openflo\.gui'\)$")
    assert ps[args + 1] == ("if ($env:OPENFLO_OPEN) { $q = [string][char]34; "
                            "$a += $q + $env:OPENFLO_OPEN + $q }")


def test_failure_shows_copyable_text_and_exits_non_zero():
    code = _code()
    launch = _index(r'-File "%ROOT%scripts\\launch-gui\.ps1"$')
    assert code[launch + 1] == 'if not errorlevel 1 goto :started'
    assert code[launch + 3] == 'goto :fail'
    fail = _index(r'^:fail$')
    show = _index(r'-File "%ROOT%scripts\\launch-error\.ps1"$')
    assert fail < show
    assert 'set "OPENFLO_ROOT=%ROOT%"' in code[fail:show]   # for Open folder
    # When the window cannot open (no Windows Forms under a script policy),
    # Notepad shows the saved text and the console stays open.
    assert code[show + 1:show + 7] == [
        'if errorlevel 1 (',
        'start "" notepad "%TEMP%\\openflo-launch-error.txt"',
        'pause', ')', 'endlocal', 'exit /b 1']
    # Every way into :fail sets the message first.
    for i, ln in enumerate(code):
        if 'goto :fail' in ln:
            assert any(c.startswith('set "OPENFLO_MSG=') for c in code[max(0, i - 3):i + 1]), ln


def test_the_failure_window_text_can_be_selected_copied_and_found_again():
    # Sky, 11-18: a message box's text cannot be selected.
    ps = [ln.strip() for ln in (BAT.parent / 'scripts' / 'launch-error.ps1').read_text().splitlines()]
    text = '\n'.join(ps)
    assert "$file = Join-Path $env:TEMP 'openflo-launch-error.txt'" in ps
    assert '$msg = $env:OPENFLO_MSG + $nl + $nl' in ps
    assert "if ($env:OPENFLO_ROOT) { $msg += 'OpenFlo folder: ' + $env:OPENFLO_ROOT.TrimEnd('\\') + $nl }" in ps
    assert "$msg += 'Report the problem: https://www.getopenflo.com/feedback'" in ps
    assert 'Set-Content -LiteralPath $file -Value $msg -Encoding UTF8' in ps
    assert '$box = New-Object System.Windows.Forms.RichTextBox' in ps
    assert '$box.ReadOnly = $true' in ps and '$box.Multiline = $true' in ps
    # The box counts a line break as one character: links are placed by
    # the box's own Find on "`n" text (Sky, 11-18: "`r`n" offsets drifted).
    assert '$box.Text = $msg -replace "`r`n", "`n"' in ps
    assert 'foreach ($m in [regex]::Matches($box.Text, $pattern)) {' in ps
    assert '$at = $box.Find($target, $from, [System.Windows.Forms.RichTextBoxFinds]::MatchCase)' in ps
    assert '$box.Select($at, $target.Length)' in ps
    assert '$links += [pscustomobject]@{ Start = $at; End = $at + $target.Length; Target = $target }' in ps
    # Sky, 11-18: links and paths blue, underlined and clickable.
    assert '$pattern = \'https?://[^\\s<>"]+[^\\s<>".,;:)]|[A-Za-z]:\\\\[^\\r\\n]+\'' in ps
    assert '$box.SelectionColor = [System.Drawing.Color]::FromArgb(11, 92, 173)' in ps
    assert '$box.SelectionFont = New-Object System.Drawing.Font($box.Font, [System.Drawing.FontStyle]::Underline)' in ps
    assert "if ($target -match '^https?://') { Start-Process $target }" in ps
    assert 'if ($l -and $box.SelectionLength -eq 0) { Open-Link $l.Target } })' in ps
    assert '$box.Add_MouseClick({ param($s, $e)' in ps
    assert '$open.Add_Click({ Start-Process explorer.exe -ArgumentList "`"$($env:OPENFLO_ROOT.TrimEnd(\'\\\'))`"" })' in ps
    for b in ('$ok', '$copy', '$open'):
        assert f'$buttons.Controls.Add({b})' in ps, b
    assert "$copy.Add_Click({ [System.Windows.Forms.Clipboard]::SetText($box.Text) })" in ps
    assert '[void]$form.ShowDialog()' in ps
    assert 'MessageBox' not in text


@pytest.mark.checkout_only
def test_broken_venvs_are_ignored_by_git():
    gi = (BAT.parent / '.gitignore').read_text().splitlines()
    assert '.venv-broken-*/' in gi
