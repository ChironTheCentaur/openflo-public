"""The CI GUI guard (tests/conftest.py) must tell a broken Tk from a lost race.

On windows-latest a Tk root fails while other xdist workers start Tk at the
same moment ("couldn't read file .../init.tcl: No error"). gui_unavailable
retried once, saw Tk come up and skipped as TRANSIENT; then the report hook
probed AGAIN, lost the same race, and turned that proven-transient skip into
a failure. That redded master and every open PR. These tests pin both halves
of the fix without needing a display: a TRANSIENT skip is final, and a probe
retries before calling Tk broken.
"""
import sys
import types

from tests import conftest


class _TclError(Exception):
    pass


class _FlakyTk:
    """tkinter.Tk that fails its first `fails` constructions."""

    def __init__(self, fails):
        self.calls = 0
        self.fails = fails

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fails:
            raise _TclError("couldn't read file \"init.tcl\": No error")
        return self

    def destroy(self):
        pass


def _fake_tkinter(monkeypatch, tk):
    """Stand-in tkinter: no display or Tcl install needed to test the guard."""
    mod = types.ModuleType('tkinter')
    mod.Tk = tk            # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'tkinter', mod)


def test_probe_survives_a_lost_race(monkeypatch):
    tk = _FlakyTk(fails=conftest._TK_PROBES - 1)
    _fake_tkinter(monkeypatch, tk)
    assert conftest._tk_probe_error(pause=0) is None
    assert tk.calls == conftest._TK_PROBES


def test_probe_still_reports_a_broken_tk(monkeypatch):
    tk = _FlakyTk(fails=10**6)
    _fake_tkinter(monkeypatch, tk)
    assert isinstance(conftest._tk_probe_error(pause=0), _TclError)
    assert tk.calls == conftest._TK_PROBES


class _Report:
    def __init__(self, reason):
        self.when = 'setup'
        self.skipped = True
        self.outcome = 'skipped'
        self.longrepr = ('tests/x.py', 1, f'Skipped: {reason}')


class _Outcome:
    def __init__(self, report):
        self.report = report

    def get_result(self):
        return self.report


def _run_hook(report):
    gen = conftest.pytest_runtest_makereport(None, None)
    next(gen)
    try:
        gen.send(_Outcome(report))
    except StopIteration:
        pass
    return report


def test_a_transient_skip_is_never_converted(monkeypatch):
    """Even when every later probe fails, the helper already proved Tk works."""
    monkeypatch.setenv('OPENFLO_REQUIRE_GUI', '1')
    _fake_tkinter(monkeypatch, _FlakyTk(fails=10**6))
    monkeypatch.setattr(conftest, '_TK_PROBE_PAUSE_S', 0)
    reason = f"Tk cannot initialise: init.tcl — {conftest._TK_TRANSIENT}, so the environment is fine"
    assert _run_hook(_Report(reason)).outcome == 'skipped'


def test_a_tk_skip_with_tk_really_down_still_fails(monkeypatch):
    monkeypatch.setenv('OPENFLO_REQUIRE_GUI', '1')
    _fake_tkinter(monkeypatch, _FlakyTk(fails=10**6))
    monkeypatch.setattr(conftest, '_tk_really_broken',
                        lambda: conftest._tk_probe_error(pause=0) is not None)
    assert _run_hook(_Report("Tk cannot initialise: can't find a usable init.tcl")).outcome == 'failed'
