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


def _stub_python3(stubs: Path, version: str) -> None:
    """A python3 that reports `version`. Its -c code runs in the real
    interpreter with sys.version_info faked, so setup.sh's own version
    check decides."""
    vi = tuple(int(x) for x in version.split('.')) + ('final', 0)
    _stub(stubs, 'python3',
          'echo "python3 $*" >> "$STUB_LOG"\n'
          f'if [ "$1" = --version ]; then echo "Python {version}"; exit 0; fi\n'
          'if [ "$1" = -c ]; then\n'
          f'  exec "{sys.executable}" -I -c "import sys; '
          f'sys.version_info = {vi!r}; $2"\n'
          'fi\n'
          'if [ "$1" = -m ] && [ "$2" = venv ]; then\n'
          '  last="$3"\n'
          + _make_venv_python_cmd('last') +
          '  exit 0\nfi\nexit 0\n')


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
    assert 'sys.version_info >= (3, 11)' in bat
    assert re.search(r'^where uv ', bat, re.M)
    assert re.search(r'^\s*call :fetch_uv', bat, re.M)
    assert '"%UV%" python install %UV_PYTHON_VERSION%' in bat
    assert '"%UV%" venv --seed --python %UV_PYTHON_VERSION% "%ROOT%.venv"' in bat
    assert _bat_var('UV_PYTHON_VERSION') == '3.12'
    assert '"%VPY%" -m pip install -e ".[gui]"' in bat
    # Shortcuts only for interactive runs.
    assert 'if /i not "%MODE%"=="auto" call :make_shortcuts' in bat


def test_bat_checked_out_with_crlf():
    # cmd.exe can miss goto/call labels in LF-only files.
    attrs = (REPO / '.gitattributes').read_text()
    assert re.search(r'^\*\.bat\s+text\s+eol=crlf\s*$', attrs, re.M)
