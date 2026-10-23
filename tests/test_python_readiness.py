"""Readiness for newer Pythons, and how fast OpenFlo starts.

- The editor opens when the drag-and-drop library (tkdnd) can't load, as with
  tkinterdnd2 0.4.3 under Tk 9 (uv's Python 3.14/3.15, python.org's macOS
  3.14.5+): drag-and-drop is turned off instead of the app never starting.
- The real app start (gui.main) paints the window before matplotlib loads,
  and the plot area follows.
- A Mann-Whitney test of two groups holding one identical value gives p = 1
  on every SciPy (1.18 returns NaN where earlier versions gave 1).
- ``openflo-run --help`` and a bad argument are answered without loading the
  scientific stack.

Each behaviour is checked both ways. Data is generated here: plain numbers
and synthetic sample names, no files.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tkinter as tk

import numpy as np
import pytest

from tests.conftest import gui_unavailable

os.environ.setdefault('MPLBACKEND', 'Agg')

HEAVY = ('numpy', 'pandas', 'scipy', 'matplotlib', 'sklearn', 'numba')


def _run(code, *args, home=None, timeout=240):
    """Run `code` in a fresh interpreter (with `home` as its home folder, so
    no real preferences are read)."""
    env = dict(os.environ, MPLBACKEND='Agg')
    if home is not None:
        env.update(HOME=str(home), USERPROFILE=str(home), APPDATA=str(home),
                   LOCALAPPDATA=str(home), XDG_CONFIG_HOME=str(home))
    r = subprocess.run([sys.executable, '-c', code, *args], capture_output=True,
                       text=True, timeout=timeout, env=env)
    return r


def _json_tail(r):
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _tk_or_skip():
    try:
        r = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f"Tk cannot start: {exc}")
    r.destroy()


# ── The real start-up, in a fresh process ───────────────────────────────────

# gui.main() as the openflo-gui command runs it, stopped where the event loop
# would start. It reports what was loaded when the window first painted and
# whether the plot area exists by the time the event loop would run.
# `--tkdnd-fails` makes the drag-and-drop library refuse to load, as tkdnd
# does when it doesn't match the installed Tk.
_START = r'''
import json, sys, time, tkinter as tk
seen = {}
if '--tkdnd-fails' in sys.argv:
    from tkinterdnd2 import TkinterDnD
    def refuse(_root):
        raise RuntimeError('Unable to load tkdnd library.')
    TkinterDnD._require = refuse
    if hasattr(TkinterDnD, 'require'):
        TkinterDnD.require = refuse
import openflo.gui as g
cls = g.ViewGateEditorWindow
# An Expose event reaches a window only when the event loop runs, after the
# window system has shown it: the sign that it has painted.
exposed = []
_orig_init = cls.__init__
def init(self, *a, **k):
    _orig_init(self, *a, **k)
    self.bind('<Expose>', lambda e: exposed.append(1), add='+')
cls.__init__ = init
finish = getattr(cls, '_finish_plot_canvas', None)
if finish is not None:
    def watched(self):
        seen['painted'] = bool(self.winfo_viewable()) and bool(exposed)
        seen['mpl_at_paint'] = 'matplotlib' in sys.modules
        finish(self)
    cls._finish_plot_canvas = watched
def stop(self, n=0):
    eds = [w for w in self.winfo_children() if isinstance(w, cls)]
    seen['editors'] = len(eds)
    d = eds[0].__dict__ if eds else {}
    seen['plot_ready'] = (d.get('_canvas') or d.get('canvas')) is not None
    seen['has_dnd'] = g.HAS_DND
    seen['tkdnd'] = getattr(self, 'TkdndVersion', None)
    seen['mpl_at_loop'] = 'matplotlib' in sys.modules
    self.destroy()
tk.Misc.mainloop = stop
g.main()
print(json.dumps(seen))
'''


def _start(*flags, tmp_home):
    _tk_or_skip()
    return _run(_START, *flags, home=tmp_home)


def test_the_app_opens_with_the_installed_tk(tmp_path):
    """The check that would have caught Tk 9: the whole start-up, with the
    real drag-and-drop library against whatever Tk this Python has."""
    pytest.importorskip('tkinterdnd2')
    seen = _json_tail(_start(tmp_home=tmp_path))
    assert seen['editors'] == 1
    assert seen['plot_ready']


def test_the_app_opens_without_drag_and_drop_when_tkdnd_cannot_load(tmp_path):
    pytest.importorskip('tkinterdnd2')
    r = _start('--tkdnd-fails', tmp_home=tmp_path)
    seen = _json_tail(r)
    assert seen['editors'] == 1, 'the editor did not open'
    assert seen['has_dnd'] is False, 'drag-and-drop still claimed to be on'
    assert 'drag-and-drop is off' in r.stdout


def test_drag_and_drop_stays_on_when_tkdnd_loads(tmp_path):
    """The other side: a loadable tkdnd is used, not turned off."""
    pytest.importorskip('tkinterdnd2')
    seen = _json_tail(_start(tmp_home=tmp_path))
    assert seen['has_dnd'] is True
    assert seen['tkdnd'], 'tkdnd was never loaded onto the window'


def test_the_window_paints_before_matplotlib_loads(tmp_path):
    seen = _json_tail(_start(tmp_home=tmp_path))
    assert seen.get('painted') is True, (
        'gui.main() never painted the window before building the plot area')
    assert seen['mpl_at_paint'] is False, (
        'matplotlib was already loaded when the window first painted')
    # ...and the plot area is there before the event loop takes over.
    assert seen['mpl_at_loop'] is True
    assert seen['plot_ready'] is True


# ── The plot area built on demand ───────────────────────────────────────────

@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f"Tk cannot start: {exc}")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def _editor(root, **kw):
    from openflo.gui import ViewGateEditorWindow
    ed = ViewGateEditorWindow(root, fcs_dir=None, labels_str='', on_save=None,
                              **kw)
    ed.withdraw()
    return ed


def test_a_deferred_editor_has_no_plot_area_until_asked(root):
    ed = _editor(root, defer_plot_canvas=True)
    assert ed._canvas is None
    ed._finish_plot_canvas()
    assert ed._canvas is not None
    assert ed.canvas.get_tk_widget().winfo_manager() == 'grid'
    ed.destroy()


def test_reaching_for_the_plot_before_it_is_built_builds_it(root):
    """Nothing can see a deferred editor without its plot: the first use of
    ax (or fig, or canvas) makes it on the spot."""
    ed = _editor(root, defer_plot_canvas=True)
    assert ed._canvas is None
    ax = ed.ax
    assert ax is ed.fig.axes[0]
    assert ed._canvas is not None
    # Building twice is a no-op, not a second canvas.
    canvas = ed.canvas
    ed._build_plot_canvas()
    assert ed.canvas is canvas
    canvases = [w for w in ed._plot_host.winfo_children()
                if w.winfo_class() == 'Canvas']
    assert len(canvases) == 1
    ed.destroy()


def test_an_ordinary_editor_builds_its_plot_at_once(root):
    ed = _editor(root)
    assert ed._canvas is not None
    ed.destroy()


def test_the_start_overlay_stays_above_a_late_plot_area(root):
    """The canvas is created after the start-here overlay, so it would cover
    it unless the overlay is lifted again."""
    ed = _editor(root, defer_plot_canvas=True)
    ed.deiconify()
    ed.update()
    ed._finish_plot_canvas()
    ed.update()
    ov = ed._empty_overlay
    assert ov.winfo_ismapped()
    # Ask Tk for the stacking order of the overlay and the plot inside their
    # shared parent (lowest first). Asking what is on top at a screen point
    # instead also sees other programs' windows, such as a parallel test
    # worker's, which made this fail on busy Windows runners.
    plot = ed.canvas.get_tk_widget()
    assert plot.winfo_parent() == ov.winfo_parent()
    order = ed.tk.splitlist(ed.tk.call('winfo', 'children', ov.winfo_parent()))
    assert order.index(str(ov)) > order.index(str(plot)), (
        'the plot area covers the start-here overlay')
    ed.destroy()


# ── Mann-Whitney ties on every SciPy ────────────────────────────────────────

def _scipy_says_nan_for_ties(monkeypatch):
    """Make the installed SciPy behave like 1.18 for all-identical groups,
    whatever version is installed, so both behaviours are tested here."""
    import scipy.stats as ss
    real = ss.mannwhitneyu

    def like_118(a, b, *args, **kw):
        r = real(a, b, *args, **kw)
        both = np.concatenate([np.ravel(a), np.ravel(b)]).astype(float)
        if both.size and np.all(both == both[0]):
            return type(r)(r.statistic, float('nan'))
        return r
    monkeypatch.setattr(ss, 'mannwhitneyu', like_118)


@pytest.mark.parametrize('nan_for_ties', [False, True],
                         ids=['scipy-before-1.18', 'scipy-1.18'])
def test_identical_groups_give_p_1(monkeypatch, nan_for_ties):
    from openflo.stats import mann_whitney
    if nan_for_ties:
        _scipy_says_nan_for_ties(monkeypatch)
    u, p = mann_whitney([20.0, 20.0, 20.0], [20.0, 20.0, 20.0])
    assert (u, p) == (4.5, 1.0)


@pytest.mark.parametrize('nan_for_ties', [False, True],
                         ids=['scipy-before-1.18', 'scipy-1.18'])
def test_a_real_difference_keeps_scipys_p(monkeypatch, nan_for_ties):
    """Only the all-identical case is touched."""
    from scipy.stats import mannwhitneyu

    from openflo.stats import mann_whitney
    a, b = [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]
    want = mannwhitneyu(a, b, alternative='two-sided')
    if nan_for_ties:
        _scipy_says_nan_for_ties(monkeypatch)
    assert mann_whitney(a, b) == (float(want.statistic), float(want.pvalue))


def test_groups_with_different_constant_values_are_not_called_identical(
        monkeypatch):
    """Each group constant, but not the same constant: a real difference,
    whose p must not be forced to 1."""
    from openflo.stats import mann_whitney
    _scipy_says_nan_for_ties(monkeypatch)
    _, p = mann_whitney([2.0, 2.0, 2.0], [9.0, 9.0, 9.0])
    assert p < 1.0


def test_a_nan_from_scipy_for_other_reasons_stays_nan():
    """Only identical groups are read as no difference; a NaN that SciPy
    gives for anything else (here a missing value) is kept."""
    from openflo.stats import mann_whitney
    _, p = mann_whitney([float('nan'), 1.0, 2.0], [3.0, 4.0, 5.0])
    assert np.isnan(p)


@pytest.mark.parametrize('nan_for_ties', [False, True],
                         ids=['scipy-before-1.18', 'scipy-1.18'])
def test_every_mann_whitney_caller_reads_ties_as_no_difference(
        monkeypatch, nan_for_ties):
    """differential abundance (GUI and CLI), the two-group comparison and the
    pairwise post-hoc all go through mann_whitney."""
    from openflo.diffexp import differential_test
    from openflo.stats import compare_groups, posthoc_pairwise
    if nan_for_ties:
        _scipy_says_nan_for_ties(monkeypatch)
    same = [12.5, 12.5, 12.5]
    rows = differential_test({'pop1': same}, {'pop1': list(same)},
                             parametric=False)
    assert rows[0]['p'] == 1.0 and rows[0]['p_adj'] == 1.0
    two = compare_groups({'ctrl': same, 'treat': list(same)},
                         parametric=False)
    assert two['test'] == 'Mann-Whitney U' and two['p'] == 1.0
    pairs = posthoc_pairwise({'ctrl': same, 'treat': list(same)})
    assert pairs[0]['p'] == 1.0


# ── openflo-run answers --help without the scientific stack ────────────────

_ENTRY = r'''
import json, sys
sys.argv = ['openflo-run'] + sys.argv[1:]
from openflo.cli_args import main
try:
    rc = main()
except SystemExit as e:
    rc = e.code
print(json.dumps({'rc': rc, 'heavy': sorted(m for m in HEAVY if m in sys.modules)}))
'''.replace('HEAVY', repr(HEAVY))


def test_help_does_not_load_the_scientific_stack():
    r = _run(_ENTRY, '--help')
    seen = _json_tail(r)
    assert seen['rc'] == 0
    assert '--trials' in r.stdout
    assert seen['heavy'] == [], f"--help loaded {seen['heavy']}"


def test_a_bad_argument_is_refused_without_the_scientific_stack():
    r = _run(_ENTRY, '--no-such-option')
    seen = _json_tail(r)
    assert seen['rc'] == 2
    assert 'unrecognized arguments: --no-such-option' in r.stderr
    assert seen['heavy'] == []


def test_a_real_run_still_reaches_the_pipeline(tmp_path):
    """The other side: valid arguments go on to cli.main, which loads the
    stack and runs. An empty trial folder makes it stop early, after the
    pipeline has been loaded."""
    trial = tmp_path / 'trial_synthetic'
    trial.mkdir()
    r = _run(_ENTRY, '--trials', str(trial), '--out', str(tmp_path / 'out'),
             '-q')
    seen = json.loads(r.stdout.strip().splitlines()[-1])
    assert 'pandas' in seen['heavy'] and 'numpy' in seen['heavy'], (
        'valid arguments never reached cli.main')


def test_cli_main_and_the_entry_point_take_the_same_options():
    """cli.main parses with the same parser, so an option can't be accepted
    by one and refused by the other."""
    import inspect

    from openflo import cli
    src = inspect.getsource(cli.main)
    assert 'ap = build_parser()' in src and 'ap.parse_args()' in src
    assert 'add_argument(' not in src


def test_the_console_script_points_at_the_light_entry():
    import tomllib
    from pathlib import Path
    cfg = tomllib.loads(
        (Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text(
            encoding='utf-8'))
    assert cfg['project']['scripts']['openflo-run'] == 'openflo.cli_args:main'
