"""The setup scripts' no-Python fallback to uv.

setup.sh runs here for real, in a copy of the checkout, under a PATH built
for each test: every system tool except python/pip/uv/curl/wget, plus stubs.
The stubs log their arguments and never touch the network, so nothing is
downloaded and nothing is installed. setup.bat cannot run on this platform;
it is checked statically against setup.sh.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SETUP_SH = REPO / 'setup.sh'
SETUP_BAT = REPO / 'setup.bat'

_HIDDEN = re.compile(r'^(python.*|pip.*|uv|uvx|curl|wget|uname)$')

needs_bash = pytest.mark.skipif(
    sys.platform == 'win32' or shutil.which('bash') is None,
    reason='setup.sh needs bash')


# ── Harness ───────────────────────────────────────────────────────────────────

def _system_bin(dest: Path) -> Path:
    """A directory linking every tool on PATH except the hidden ones."""
    dest.mkdir()
    for d in os.environ.get('PATH', '').split(os.pathsep):
        if not d or not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            src = os.path.join(d, name)
            if (_HIDDEN.match(name) or (dest / name).exists()
                    or not os.access(src, os.X_OK) or os.path.isdir(src)):
                continue
            (dest / name).symlink_to(src)
    return dest


def _stub(stubs: Path, name: str, body: str) -> None:
    p = stubs / name
    p.write_text('#!/usr/bin/env bash\n' + body)
    p.chmod(0o755)


# A fake interpreter for a venv: logs `venv-python <args>` and succeeds.
_VENV_PY = 'echo "venv-python $*" >> "$STUB_LOG"\nexit 0\n'


def _make_venv_python_cmd(var: str) -> str:
    """Shell that writes a fake <venv>/bin/python into the dir in $var."""
    return (f'mkdir -p "${var}/bin"\n'
            f'cat > "${var}/bin/python" <<\'PYEOF\'\n#!/usr/bin/env bash\n'
            f'{_VENV_PY}PYEOF\n'
            f'chmod +x "${var}/bin/python"\n')


def _stub_uv(stubs: Path) -> None:
    _stub(stubs, 'uv',
          'echo "uv $*" >> "$STUB_LOG"\n'
          'if [ "$1" = venv ]; then\n'
          '  for a in "$@"; do last="$a"; done\n'
          + _make_venv_python_cmd('last') +
          'fi\nexit 0\n')


def _fake_python_body(name: str, version: str) -> str:
    """Body of a fake interpreter that logs `<name> <args>`, reports
    `version`, and runs -c code in the real interpreter with
    sys.version_info faked, so setup.sh's own version check decides."""
    vi = tuple(int(x) for x in version.split('.')) + ('final', 0)
    return (f'echo "{name} $*" >> "$STUB_LOG"\n'
            f'if [ "$1" = --version ]; then echo "Python {version}"; exit 0; fi\n'
            'if [ "$1" = -c ]; then\n'
            f'  exec "{sys.executable}" -I -c "import sys; '
            f'sys.version_info = {vi!r}; $2"\n'
            'fi\n')


def _stub_python3(stubs: Path, version: str, name: str = 'python3') -> None:
    """A `name` on PATH that reports `version` and can make a venv."""
    _stub(stubs, name,
          _fake_python_body(name, version) +
          'if [ "$1" = -m ] && [ "$2" = venv ]; then\n'
          '  last="$3"\n'
          + _make_venv_python_cmd('last') +
          '  exit 0\nfi\nexit 0\n')


def _existing_venv(stubs: Path, version: str) -> None:
    """A .venv left by an earlier setup, whose python reports `version`."""
    vbin = stubs.parent / 'checkout' / '.venv' / 'bin'
    vbin.mkdir(parents=True)
    _stub(vbin, 'python',
          _fake_python_body('venv-python', version) + 'exit 0\n')


def _stub_uname(stubs: Path, system: str, machine: str) -> None:
    _stub(stubs, 'uname',
          f'case "$1" in -s) echo {system};; -m) echo {machine};; '
          f'*) echo {system};; esac\n')


def _stub_downloader(stubs: Path, tool: str, payload: bytes) -> None:
    """curl/wget stub: logs the URL and writes `payload` to the -o/-O file."""
    (stubs / 'payload.bin').write_bytes(payload)
    flag = '-o' if tool == 'curl' else '-O'
    _stub(stubs, tool,
          f'echo "{tool} $*" >> "$STUB_LOG"\n'
          'out=""; prev=""\n'
          'for a in "$@"; do\n'
          f'  [ "$prev" = "{flag}" ] && out="$a"\n'
          '  prev="$a"\n'
          'done\n'
          f'cp "{stubs}/payload.bin" "$out"\n')


class Run:
    def __init__(self, proc: subprocess.CompletedProcess, log: Path,
                 root: Path):
        self.code = proc.returncode
        self.out = proc.stdout + proc.stderr
        self.log = log.read_text().splitlines() if log.exists() else []
        self.root = root


@pytest.fixture
def sandbox(tmp_path):
    """Copy of setup.sh in a fake checkout, a stub dir and a clean PATH."""
    root = tmp_path / 'checkout'
    root.mkdir()
    shutil.copy(SETUP_SH, root / 'setup.sh')
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    sysbin = _system_bin(tmp_path / 'sysbin')
    log = tmp_path / 'calls.log'
    bash = shutil.which('bash')

    def run(**extra_env) -> Run:
        env = {
            'PATH': f'{stubs}{os.pathsep}{sysbin}',
            'HOME': str(tmp_path / 'home'),
            'STUB_LOG': str(log),
            'OPENFLO_UV_BASE_URL': 'https://downloads.invalid/uv',
            'LC_ALL': 'C',
        }
        env.update(extra_env)
        proc = subprocess.run([bash, str(root / 'setup.sh')], env=env,
                              capture_output=True, text=True, timeout=60)
        return Run(proc, log, root)

    _stub_uname(stubs, 'Linux', 'x86_64')
    return stubs, run


def _pinned_sh_version() -> str:
    m = re.search(r'^UV_VERSION="([^"]+)"', SETUP_SH.read_text(), re.M)
    assert m, 'setup.sh pins no UV_VERSION'
    return m.group(1)


def _sh_hashes() -> dict[str, str]:
    rows = re.findall(r'^\s*(uv-[\w.-]+?)\)\s*echo "([0-9a-f]{64})"',
                      SETUP_SH.read_text(), re.M)
    return dict(rows)


# ── setup.sh, run with stubs ──────────────────────────────────────────────────

@needs_bash
def test_no_python_uses_uv_on_path(sandbox):
    stubs, run = sandbox
    _stub_uv(stubs)
    r = run()
    assert r.code == 0, r.out
    assert 'uv python install 3.12' in r.log
    venv_calls = [c for c in r.log if c.startswith('uv venv')]
    assert len(venv_calls) == 1
    assert '--python 3.12' in venv_calls[0] and '--seed' in venv_calls[0]
    assert venv_calls[0].endswith('/.venv')
    assert 'venv-python -m pip install -e .[gui]' in r.log
    # uv was on PATH, so nothing was downloaded.
    assert not (r.root / '.tools').exists()
    assert r.log.index('uv python install 3.12') < r.log.index(venv_calls[0])


@needs_bash
def test_python_older_than_311_is_rejected_for_uv(sandbox):
    stubs, run = sandbox
    _stub_python3(stubs, '3.9.18')
    _stub_uv(stubs)
    r = run()
    assert r.code == 0, r.out
    assert 'needs Python 3.11+' in r.out
    assert not any(c.startswith('python3 -m venv') for c in r.log)
    assert 'uv python install 3.12' in r.log
    assert 'venv-python -m pip install -e .[gui]' in r.log


@needs_bash
def test_python_315_is_skipped_with_a_clear_message(sandbox):
    stubs, run = sandbox
    _stub_python3(stubs, '3.15.0')
    _stub_uv(stubs)
    r = run()
    assert r.code == 0, r.out
    assert 'Ignoring' in r.out and 'Python 3.15.0' in r.out
    assert 'supports Python 3.11 to 3.14' in r.out and 'FlowUtils' in r.out
    assert 'needs Python 3.11+' not in r.out
    assert not any(c.startswith('python3 -m venv') for c in r.log)
    assert 'uv python install 3.12' in r.log
    assert 'venv-python -m pip install -e .[gui]' in r.log


@needs_bash
@pytest.mark.parametrize('newer', ['3.15.0', '3.16.1'])
def test_too_new_python3_falls_back_to_a_supported_versioned_python(
        sandbox, newer):
    stubs, run = sandbox
    _stub_python3(stubs, newer)
    _stub_python3(stubs, '3.13.2', name='python3.13')
    _stub_python3(stubs, '3.11.9', name='python3.11')
    _stub_uv(stubs)
    r = run()
    assert r.code == 0, r.out
    assert f'Python {newer}' in r.out
    assert 'supports Python 3.11 to 3.14' in r.out
    # The newest supported one wins; uv is not needed.
    assert any(c.startswith('python3.13 -m venv') for c in r.log), r.log
    assert not any(c.startswith('python3.11 -m venv') for c in r.log)
    assert not any(c.startswith('uv ') for c in r.log)
    assert 'venv-python -m pip install -e .[gui]' in r.log


@needs_bash
@pytest.mark.parametrize('version', ['3.11.0', '3.14.0', '3.14.9'])
def test_supported_edges_are_used(sandbox, version):
    stubs, run = sandbox
    _stub_python3(stubs, version)
    _stub_uv(stubs)
    r = run()
    assert r.code == 0, r.out
    assert 'Ignoring' not in r.out
    assert any(c.startswith('python3 -m venv') for c in r.log)
    assert not any(c.startswith('uv ') for c in r.log)


@needs_bash
def test_existing_venv_on_315_stops_with_a_clear_message(sandbox):
    stubs, run = sandbox
    _existing_venv(stubs, '3.15.0')
    _stub_uv(stubs)
    r = run()
    assert r.code != 0, r.out
    assert 'existing .venv uses Python 3.15.0' in r.out
    assert 'supports Python 3.11 to 3.14' in r.out
    assert 'Delete the .venv folder' in r.out
    # pip never runs, so no compiler error can follow.
    assert not any('pip' in c for c in r.log), r.log


@needs_bash
def test_existing_venv_on_supported_python_installs(sandbox):
    stubs, run = sandbox
    _existing_venv(stubs, '3.12.4')
    _stub_uv(stubs)
    r = run()
    assert r.code == 0, r.out
    assert not any(c.startswith('uv ') for c in r.log)
    assert 'venv-python -m pip install -e .[gui]' in r.log


@needs_bash
def test_good_python_keeps_the_old_path(sandbox):
    stubs, run = sandbox
    _stub_python3(stubs, '3.12.4')
    _stub_uv(stubs)
    r = run()
    assert r.code == 0, r.out
    assert any(c.startswith('python3 -m venv') for c in r.log)
    assert not any(c.startswith('uv ') for c in r.log)
    assert 'venv-python -m pip install -e .[gui]' in r.log


@needs_bash
@pytest.mark.parametrize('tool', ['curl', 'wget'])
@pytest.mark.parametrize('system,machine,asset', [
    ('Linux', 'x86_64', 'uv-x86_64-unknown-linux-gnu.tar.gz'),
    ('Linux', 'aarch64', 'uv-aarch64-unknown-linux-gnu.tar.gz'),
    ('Darwin', 'x86_64', 'uv-x86_64-apple-darwin.tar.gz'),
    ('Darwin', 'arm64', 'uv-aarch64-apple-darwin.tar.gz'),
])
def test_wrong_checksum_stops_setup(sandbox, tool, system, machine, asset):
    stubs, run = sandbox
    _stub_uname(stubs, system, machine)
    # Generated bytes: a tarball whose hash cannot match the pinned one.
    _stub_downloader(stubs, tool, os.urandom(4096))
    r = run()
    assert r.code != 0, r.out
    assert 'SHA-256 mismatch' in r.out and 'Refusing to continue' in r.out
    url = f'https://downloads.invalid/uv/{_pinned_sh_version()}/{asset}'
    assert any(c.startswith(tool) and url in c for c in r.log), r.log
    # Nothing unpacked, nothing left behind, nothing installed.
    assert not (r.root / '.tools' / 'uv').exists()
    assert list((r.root / '.tools').iterdir()) == []
    assert not (r.root / '.venv').exists()
    assert not any(c.startswith('venv-python') for c in r.log)


@needs_bash
def test_sh_pins_a_hash_for_every_asset_it_can_pick():
    hashes = _sh_hashes()
    for arch in ('x86_64', 'aarch64'):
        for os_ in ('apple-darwin', 'unknown-linux-gnu'):
            assert f'uv-{arch}-{os_}.tar.gz' in hashes
    assert len(set(hashes.values())) == len(hashes)


# ── setup.bat, static ─────────────────────────────────────────────────────────

def _bat() -> str:
    return SETUP_BAT.read_text()


def _bat_var(name: str) -> str:
    m = re.search(rf'^set "{name}=([^"]*)"', _bat(), re.M)
    assert m, f'setup.bat sets no {name}'
    return m.group(1)


def test_bat_pins_same_uv_as_sh():
    assert _bat_var('UV_VERSION') == _pinned_sh_version()
    hashes = _sh_hashes()
    assert _bat_var('UV_SHA256_X64') == hashes['uv-x86_64-pc-windows-msvc.zip']
    assert (_bat_var('UV_SHA256_ARM64')
            == hashes['uv-aarch64-pc-windows-msvc.zip'])
    bat = _bat()
    # Each architecture gets its own asset and its own hash.
    assert re.search(r'"AMD64" set "UV_ASSET=uv-x86_64-pc-windows-msvc\.zip"',
                     bat)
    assert re.search(r'"AMD64" set "UV_SHA256=%UV_SHA256_X64%"', bat)
    assert re.search(r'"ARM64" set "UV_ASSET=uv-aarch64-pc-windows-msvc\.zip"',
                     bat)
    assert re.search(r'"ARM64" set "UV_SHA256=%UV_SHA256_ARM64%"', bat)


def test_bat_verifies_hash_and_refuses_on_mismatch():
    bat = _bat()
    assert 'Get-FileHash -Algorithm SHA256' in bat
    # Mismatch: delete the zip and exit 2 before Expand-Archive runs.
    m = re.search(r"if \(\$h -ne \$env:UV_SHA256\.ToLower\(\)\) \{([^}]*)\}",
                  bat)
    assert m, 'no hash comparison'
    assert 'exit 2' in m.group(1) and 'Remove-Item' in m.group(1)
    assert bat.index(m.group(0)) < bat.index('Expand-Archive')
    # The batch side turns exit 2 into a refusal.
    assert re.search(r'if errorlevel 2 \(\s*\n\s*echo [^\n]*Refusing to '
                     r'continue\.\s*\n\s*exit /b 1', bat)


def test_bat_falls_back_to_uv_and_installs_as_before():
    bat = _bat()
    assert re.search(r'^where uv ', bat, re.M)
    assert re.search(r'^\s*call :fetch_uv', bat, re.M)
    assert '"%UV%" python install %UV_PYTHON_VERSION%' in bat
    assert '"%UV%" venv --seed --python %UV_PYTHON_VERSION% "%ROOT%.venv"' in bat
    assert _bat_var('UV_PYTHON_VERSION') == '3.12'
    assert '"%VPY%" -m pip install -e ".[gui]"' in bat
    # Shortcuts only for interactive runs.
    assert 'if /i not "%MODE%"=="auto" call :make_shortcuts' in bat


@pytest.mark.checkout_only
def test_bat_checked_out_with_crlf():
    # cmd.exe can miss goto/call labels in LF-only files.
    attrs = (REPO / '.gitattributes').read_text()
    assert re.search(r'^\*\.bat\s+text\s+eol=crlf\s*$', attrs, re.M)


# ── Supported Python range: one answer everywhere ─────────────────────────────

def _sh_var(name: str) -> str:
    m = re.search(rf'^{name}=(\d+)$', SETUP_SH.read_text(), re.M)
    assert m, f'setup.sh sets no {name}'
    return m.group(1)


def test_supported_range_agrees_across_scripts_and_readme():
    oldest, newest = _sh_var('PY_OLDEST_MINOR'), _sh_var('PY_NEWEST_MINOR')
    assert (oldest, newest) == ('11', '14')
    assert _bat_var('PY_OLDEST_MINOR') == oldest
    assert _bat_var('PY_NEWEST_MINOR') == newest
    readme = (REPO / 'README.md').read_text(encoding='utf-8')
    assert f'**Supported Python: 3.{oldest} to 3.{newest}.**' in readme
    assert f'Python 3.{int(newest) + 1} is not supported yet' in readme


def test_requires_python_is_not_capped():
    # A cap makes pip on a newer Python quietly pick an old OpenFlo release
    # (one without the cap) instead of saying anything; we keep it open and
    # tell people in the setup scripts and the README instead.
    text = (REPO / 'pyproject.toml').read_text(encoding='utf-8')
    m = re.search(r'^requires-python\s*=\s*"([^"]*)"', text, re.M)
    assert m and m.group(1) == '>=3.11'


def _run_bat_check(var: str, version: tuple[int, int]) -> bool:
    """Run setup.bat's PYCHECK or PYNEW code, expanded as cmd would expand
    it, under a faked sys.version_info. True when it exits 0."""
    code = _bat_var(var)
    for name in ('PY_OLDEST_MINOR', 'PY_NEWEST_MINOR'):
        code = code.replace(f'%{name}%', _bat_var(name))
    assert '%' not in code, code
    vi = version + (0, 'final', 0)
    proc = subprocess.run(
        [sys.executable, '-I', '-c',
         f'import sys; sys.version_info = {vi!r}; {code}'])
    return proc.returncode == 0


@pytest.mark.parametrize('minor,ok,too_new', [
    (10, False, False), (11, True, False), (12, True, False),
    (14, True, False), (15, False, True), (16, False, True),
])
def test_bat_version_checks(minor, ok, too_new):
    assert _run_bat_check('PYCHECK', (3, minor)) is ok
    assert _run_bat_check('PYNEW', (3, minor)) is too_new


def test_bat_tries_each_supported_version_by_name():
    bat = _bat()
    oldest = int(_bat_var('PY_OLDEST_MINOR'))
    newest = int(_bat_var('PY_NEWEST_MINOR'))
    tried = re.findall(r'^if not defined PYEXE py -3\.(\d+) -c "%PYCHECK%" '
                       r'>nul 2>&1 && set "PYEXE=py -3\.(\d+)"$', bat, re.M)
    assert [int(a) for a, _ in tried] == list(range(newest, oldest - 1, -1))
    assert all(a == b for a, b in tried)
    # Default launcher first, PATH python last, all before the uv fallback.
    assert bat.index('py -3 -c "%PYCHECK%"') < bat.index(f'py -3.{newest} -c')
    path_py = bat.index('python -c "%PYCHECK%"')
    assert bat.index(f'py -3.{oldest} -c') < path_py
    assert path_py < bat.index('\n:use_uv\n')


def test_bat_explains_a_too_new_python():
    bat = _bat()
    msg = _bat_var('TOO_NEW_1') + ' ' + _bat_var('TOO_NEW_2')
    msg = msg.replace('%PY_OLDEST_MINOR%', '11').replace('%PY_NEWEST_MINOR%', '14')
    assert 'supports Python 3.11 to 3.14' in msg and 'FlowUtils' in msg
    # Both the launcher and PATH python get the explanation when too new.
    for label in ('py_launcher_too_new', 'path_python_too_new'):
        assert re.search(rf'-c "%PYNEW%" >nul 2>&1 && goto :{label}$', bat, re.M)
        block = bat.split(f'\n:{label}\n', 1)[1].split('\n:', 1)[0]
        assert '%TOO_NEW_1%' in block and '%TOO_NEW_2%' in block


def test_bat_checks_an_existing_venv_before_pip():
    install = _bat().split('\n:install\n', 1)[1]
    check = install.index('"%VPY%" -c "%PYCHECK%" >nul 2>&1 && goto :venv_ok')
    too_new = install.index(
        '"%VPY%" -c "%PYNEW%" >nul 2>&1 && goto :venv_too_new')
    assert check < too_new < install.index('-m pip install')
    # Between the check and :venv_ok, every path ends in :fail.
    unsupported = install[check:install.index('\n:venv_ok\n')]
    assert unsupported.count('goto :fail') == 2
    block = unsupported.split('\n:venv_too_new\n', 1)[1]
    assert '%TOO_NEW_1%' in block and 'Delete the .venv folder' in block


def test_bat_too_old_paths_skip_the_too_new_text():
    bat = _bat()
    # A too-old py -3 says "needs 3.11+" and jumps past the too-new text;
    # a too-old PATH python jumps straight to uv.
    for old_line, jump, label in (
            ('py -3 --version', 'goto :check_path_python',
             ':py_launcher_too_new'),
            ('python --version', 'goto :use_uv', ':path_python_too_new')):
        start = bat.index(old_line)
        between = bat[start:bat.index('\n' + label + '\n', start)]
        assert 'needs Python 3.%PY_OLDEST_MINOR%+' in between
        assert between.rstrip().endswith(jump), between


def test_bat_py_probes_cannot_install_a_python():
    bat = _bat()
    first_probe = bat.index('py -3 -c "%PYCHECK%"')
    for line in ('set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"',
                 'set "PYLAUNCHER_ALLOW_INSTALL="'):
        assert re.search(rf'^{re.escape(line)}$', bat, re.M), line
        assert bat.index(line) < first_probe
    assert bat.index('setlocal') < first_probe


@needs_bash
def test_existing_venv_on_310_stops_with_a_clear_message(sandbox):
    stubs, run = sandbox
    _existing_venv(stubs, '3.10.14')
    _stub_uv(stubs)
    r = run()
    assert r.code != 0, r.out
    assert 'existing .venv uses Python 3.10.14' in r.out
    assert 'needs Python 3.11 to 3.14' in r.out
    assert 'FlowUtils' not in r.out
    assert not any('pip' in c for c in r.log), r.log


@needs_bash
def test_too_new_python3_falls_back_to_the_oldest_supported_name(sandbox):
    stubs, run = sandbox
    _stub_python3(stubs, '3.15.0')
    _stub_python3(stubs, '3.11.9', name='python3.11')
    _stub_uv(stubs)
    r = run()
    assert r.code == 0, r.out
    assert any(c.startswith('python3.11 -m venv') for c in r.log), r.log
    assert not any(c.startswith('uv ') for c in r.log)


def test_readme_manual_routes_state_the_range():
    readme = (REPO / 'README.md').read_text(encoding='utf-8')
    assert 'routes below need Python 3.11 to 3.14.' in readme
    assert 'Python 3.11+' not in readme
