"""Help > Check for updates: the editor flow (editor_update.UpdateMixin, 10 %
covered) and the parts of openflo.update that tests/test_update.py leaves
out (run_update, current_version, install-kind detection on a real repo).

Network and package managers are never touched: check_for_update / run_update
are replaced where the flow is under test, and run_update's own subprocess is
pointed at `python -c` or at a throw-away `git init` repo.
"""
from __future__ import annotations

import subprocess
import sys
import types

import pytest

from openflo import editor_update, update
from openflo.editor_update import UpdateMixin


class _Ed:
    """Tk-free stand-in for the editor: status bar + an `after` queue."""

    def __init__(self):
        self.status = []
        self.status_var = types.SimpleNamespace(set=self.status.append)
        self.queued = []
        self._ran = 0

    @property
    def status_ran(self):
        # at least one callback has run since the last drain started
        r, self._ran = self._ran, 0
        return r

    def after(self, _ms, fn):
        def run():
            self._ran += 1
            fn()
        self.queued.append(run)

    _check_for_updates = UpdateMixin._check_for_updates
    _on_update_checked = UpdateMixin._on_update_checked
    _offer_update = UpdateMixin._offer_update
    _run_update = UpdateMixin._run_update
    _on_update_done = UpdateMixin._on_update_done

    def drain(self, timeout=10.0, until=None):
        """Run queued `after` callbacks; wait for the worker thread to queue
        one first, and with `until`, keep going until it holds."""
        import time
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            while self.queued:
                self.queued.pop(0)()
            if until is None and not self.queued and self.status_ran:
                return
            if until is not None and until():
                return
            time.sleep(0.01)


@pytest.fixture
def dialogs(monkeypatch):
    seen = []
    mb = editor_update.messagebox
    for name in ('showinfo', 'showerror'):
        monkeypatch.setattr(mb, name,
                            lambda *a, _n=name, **k: seen.append((_n, a)))
    seen_answers = []
    monkeypatch.setattr(mb, 'askyesnocancel',
                        lambda *a, **k: seen_answers.pop(0))
    opened = []
    import webbrowser
    monkeypatch.setattr(webbrowser, 'open', lambda url: opened.append(url))
    return types.SimpleNamespace(shown=seen, answers=seen_answers,
                                 opened=opened)


def test_offline_check_is_silent_at_startup_but_reported_on_demand(
        monkeypatch, dialogs):
    monkeypatch.setattr(update, 'check_for_update', lambda: None)
    ed = _Ed()
    ed._check_for_updates(silent=True)
    ed.drain()
    assert dialogs.shown == []                       # startup: say nothing
    ed._check_for_updates(silent=False)
    ed.drain()
    assert [n for n, _ in dialogs.shown] == ['showinfo']
    assert "Couldn't reach GitHub" in dialogs.shown[0][1][1]


def test_up_to_date_sets_status_and_only_speaks_when_asked(monkeypatch,
                                                           dialogs):
    res = {'current': '2.6.1', 'latest': '2.6.1', 'available': False,
           'url': 'u'}
    monkeypatch.setattr(update, 'check_for_update', lambda: res)
    ed = _Ed()
    ed._check_for_updates(silent=True)
    ed.drain()
    assert ed.status[-1] == 'OpenFlo 2.6.1 is up to date.'
    assert dialogs.shown == []
    ed._check_for_updates(silent=False)
    ed.drain()
    assert len(dialogs.shown) == 1


@pytest.mark.parametrize('answer, expect', [
    (None, 'nothing'), (False, 'browser'), (True, 'update')])
def test_offer_update_routes_each_answer(monkeypatch, dialogs, answer, expect):
    res = {'current': '2.6.0', 'latest': '2.6.1', 'available': True,
           'url': 'https://example/rel'}
    monkeypatch.setattr(update, 'check_for_update', lambda: res)
    monkeypatch.setattr(update, 'detect_install_kind', lambda: 'pip')
    ran = []
    monkeypatch.setattr(update, 'run_update',
                        lambda kind=None: (ran.append(kind), (True, 'l1\nl2'))[1])
    dialogs.answers.append(answer)
    ed = _Ed()
    ed._check_for_updates(silent=True)
    ed.drain()
    assert any(s.startswith('Update available: OpenFlo 2.6.1') for s in ed.status)
    if expect == 'nothing':
        assert dialogs.opened == [] and ran == []
    elif expect == 'browser':
        assert dialogs.opened == ['https://example/rel'] and ran == []
    else:
        ed.drain(until=lambda: bool(ran) and 'restart' in ed.status[-1])
        assert ran == ['pip'] and dialogs.opened == []
        assert ed.status[-1] == 'Update installed — restart OpenFlo to use it.'
        assert dialogs.shown[-1][0] == 'showinfo'
        assert 'l2' in dialogs.shown[-1][1][1]


def test_failed_update_shows_error_with_log_tail(monkeypatch, dialogs):
    monkeypatch.setattr(update, 'run_update',
                        lambda kind=None: (False, '\n'.join(f'line{i}'
                                                            for i in range(30))))
    ed = _Ed()
    ed._run_update('pip')
    ed.drain()
    kind, args = dialogs.shown[-1]
    assert kind == 'showerror' and ed.status[-1].startswith('Update failed')
    assert 'line29' in args[1] and 'line17' not in args[1]   # last 12 lines


# ── openflo.update pieces the existing tests do not run ───────────────────

def test_run_update_reports_command_exit_status(monkeypatch):
    for code, ok in ((0, True), (4, False)):
        monkeypatch.setattr(update, 'update_command', lambda kind=None, repo=None,
                            _c=code: [sys.executable, '-c',
                                      f'print("out"); raise SystemExit({_c})'])
        got_ok, log = update.run_update(kind='pip')
        assert got_ok is ok and 'out' in log


def _git_repo(path, pyproject):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    (path / 'pyproject.toml').write_text(pyproject, encoding='utf-8')
    pkg = path / 'pkgdir'
    pkg.mkdir()
    return pkg


def test_run_update_without_upstream_explains_and_does_not_pull(tmp_path,
                                                                monkeypatch):
    _git_repo(tmp_path / 'co', '[project]\nname = "openflo"\nversion = "1.0"\n')
    monkeypatch.setattr(update, '_package_git_root',
                        lambda: str(tmp_path / 'co'))
    calls = []
    real = subprocess.run
    monkeypatch.setattr(update.subprocess, 'run',
                        lambda cmd, *a, **k: (calls.append(cmd), real(cmd, *a, **k))[1])
    ok, msg = update.run_update(kind='git')
    assert ok is False
    assert 'no upstream' in msg and 'git checkout master' in msg
    assert not any('pull' in c for c in calls), calls


def test_current_version_reads_the_live_checkout_pyproject(tmp_path,
                                                           monkeypatch):
    _git_repo(tmp_path / 'co', '[project]\nname = "openflo"\nversion = "7.7.7"\n')
    monkeypatch.setattr(update, '_package_git_root',
                        lambda: str(tmp_path / 'co'))
    assert update.current_version() == '7.7.7'


def test_pip_install_inside_another_projects_repo_is_not_a_checkout(
        tmp_path, monkeypatch):
    # Any enclosing git work tree used to count as an OpenFlo checkout, so a
    # pip install in myproject/.venv reported myproject's version and Update
    # ran `git pull` on myproject.
    pkg = _git_repo(tmp_path / 'myproject',
                    '[project]\nname = "other-project"\nversion = "9.9.9"\n')
    site = pkg / '.venv' / 'Lib' / 'site-packages' / 'openflo'
    site.mkdir(parents=True)
    monkeypatch.setattr(update, '__file__', str(site / 'update.py'))
    assert update.detect_install_kind() == 'pip'
    assert update.current_version() != '9.9.9'


def test_openflo_source_checkout_is_still_a_checkout(tmp_path, monkeypatch):
    root = tmp_path / 'openflo'
    _git_repo(root, '[project]\nname = "openflo"\nversion = "7.7.7"\n')
    src = root / 'src' / 'openflo'
    src.mkdir(parents=True)
    monkeypatch.setattr(update, '__file__', str(src / 'update.py'))
    assert update.detect_install_kind() == 'git'
    assert update.current_version() == '7.7.7'
    cmd = update.update_command()
    assert cmd[:2] == ['git', '-C'] and cmd[-2:] == ['pull', '--ff-only']


def test_installed_copy_inside_the_openflo_checkout_is_a_pip_install(
        tmp_path, monkeypatch):
    """A non-editable install into a venv kept inside the checkout runs the
    installed copy, which `git pull` on the checkout would not update."""
    root = tmp_path / 'openflo'
    _git_repo(root, '[project]\nname = "openflo"\nversion = "7.7.7"\n')
    site = root / '.venv' / 'Lib' / 'site-packages' / 'openflo'
    site.mkdir(parents=True)
    monkeypatch.setattr(update, '__file__', str(site / 'update.py'))
    assert update.detect_install_kind() == 'pip'
