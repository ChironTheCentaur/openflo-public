"""The opt-in "Fast graph" (Leiden's pruned neighbour graph) beyond the
workspace: the Cluster dialog and openflo-run --fast-graph.

The dense graph stays the default everywhere; the option only reaches
Leiden. Every check runs on generated data with synthetic sample names and
generic channels (FL1-A ...). Where a check compares labels it compares them
with a direct FlowSample.run_leiden call on the same data, so a flag that is
accepted but dropped, or sent where it should not go, fails.
"""
from __future__ import annotations

import contextlib
import io
import json
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample
from tests.conftest import gui_unavailable

# openflo-run's fixtures: a trial folder of two groups, and a stubbed run.
from tests.test_cli_run import stubbed as stubbed  # noqa: F401
from tests.test_cli_run import two_groups as two_groups  # noqa: F401

CH = ['FL1-A', 'FL2-A', 'FL3-A', 'FL4-A', 'FL5-A', 'FL6-A']
N, K = 2500, 15


def _df(seed=0):
    """Four overlapping populations: dense and pruned graphs partition them
    differently (tests/test_fast_graph.py uses the same recipe)."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(3.5, 6.5, size=(4, len(CH)))
    truth = rng.choice(4, N)
    X = centres[truth] + rng.normal(0, 0.6, (N, len(CH)))
    return pd.DataFrame(X, columns=CH)


def _sample(df, name='SampleA'):
    s = FlowSample.from_dataframe(df.copy(), name=name)
    s.fluor_channels = list(CH)
    return s


def _direct(df, fast):
    s = _sample(df)
    s.run_leiden(channels=CH, n_neighbors=K, max_events=N, random_state=42,
                 fast_graph=fast)
    return s.data['leiden'].to_numpy()


# ── the shared tip ──────────────────────────────────────────────────────────

def test_fast_graph_tip_explains_the_tradeoff():
    from openflo.tool_window import FAST_GRAPH_TIP
    for words in ('Leiden only', 'Off (default)', 'faster',
                  'not comparable', 'differ'):
        assert words in FAST_GRAPH_TIP, words


# ── openflo-run ─────────────────────────────────────────────────────────────

def test_cli_flag_is_off_by_default_and_explained():
    from openflo.cli_args import build_parser
    ap = build_parser()
    assert ap.parse_args([]).fast_graph is False
    assert ap.parse_args(['--fast-graph']).fast_graph is True
    help_text = ' '.join(ap.format_help().split())
    assert '--fast-graph' in help_text
    for words in ('Leiden only', 'not comparable', 'Off by default'):
        assert words in help_text, words


def test_cluster_settings_carry_the_flag_and_default_off():
    from openflo.shared_clustering import cluster_settings
    base = cluster_settings(30, 1, None, 1.0, 42)
    assert base['fast_graph'] is False
    assert cluster_settings(30, 1, None, 1.0, 42,
                            {'method': 'leiden'})['fast_graph'] is False
    assert cluster_settings(30, 1, None, 1.0, 42,
                            {'method': 'leiden',
                             'fast_graph': True})['fast_graph'] is True


def test_cli_worker_leiden_labels_match_a_direct_run_with_the_same_graph():
    pytest.importorskip('leidenalg')
    from openflo.shared_clustering import cluster_by_method
    df = _df()
    task = {'k': K, 'n_jobs': 1, 'random_state': 42, 'method': 'leiden',
            'max_events': N}
    got = {}
    for fast in (False, True):
        s = _sample(df)
        cluster_by_method(s, dict(task, fast_graph=fast))
        got[fast] = s.data['cluster'].to_numpy()
    plain = _sample(df)
    cluster_by_method(plain, task)            # no key at all: the default
    assert np.array_equal(plain.data['cluster'].to_numpy(), got[False])
    assert np.array_equal(got[False], _direct(df, False))
    assert np.array_equal(got[True], _direct(df, True))
    assert not np.array_equal(got[False], got[True])


def test_cli_worker_does_not_send_the_flag_to_other_methods(monkeypatch):
    from openflo.shared_clustering import cluster_by_method
    seen = {}
    monkeypatch.setattr(FlowSample, 'cluster',
                        lambda self, **kw: seen.setdefault('pg', kw))
    monkeypatch.setattr(FlowSample, 'run_flowsom',
                        lambda self, **kw: seen.setdefault('som', kw))
    s = _sample(_df())
    task = {'k': K, 'n_jobs': 1, 'random_state': 42, 'fast_graph': True}
    cluster_by_method(s, dict(task, method='phenograph'))
    cluster_by_method(s, dict(task, method='flowsom'))
    assert 'fast_graph' not in seen['pg']
    assert 'fast_graph' not in seen['som']


def _leiden_spy(monkeypatch):
    seen = []
    orig = FlowSample.run_leiden

    def spy(self, *a, **kw):
        seen.append(kw.get('fast_graph', False))
        return orig(self, *a, **kw)
    monkeypatch.setattr(FlowSample, 'run_leiden', spy)
    return seen


def test_openflo_run_fast_graph_reaches_every_leiden_call(
        two_groups, tmp_path, stubbed, monkeypatch):
    """--fast-graph reaches each sample's clustering and the shared one, and
    the run manifest records it; without it every call is the default."""
    pytest.importorskip('leidenalg')
    from tests.test_cli_run import _main
    trial, groups, _ = two_groups
    for flag, out in ((['--fast-graph'], 'fast'), ([], 'dense')):
        seen = _leiden_spy(monkeypatch)
        _main(monkeypatch, '--trials', trial, '--groups', json.dumps(groups),
              '--fmo-sets', '{}', '-q', '--cluster-method', 'leiden',
              *flag, '--out', str(tmp_path / out))
        assert len(seen) == 5                # 4 samples + 1 shared
        assert seen == [bool(flag)] * 5
        with open(tmp_path / out / 'run_manifest.json', encoding='utf-8') as f:
            assert json.load(f)['settings']['fast_graph'] is bool(flag)


def test_openflo_run_says_the_flag_is_leiden_only(
        two_groups, tmp_path, stubbed, monkeypatch, capsys):
    from tests.test_cli_run import _main
    trial, groups, _ = two_groups
    _main(monkeypatch, '--trials', trial, '--groups', json.dumps(groups),
          '--fmo-sets', '{}', '-q', '--fast-graph',
          '--out', str(tmp_path / 'pg'))
    out = capsys.readouterr().out
    assert '--fast-graph applies only to --cluster-method leiden' in out
    _main(monkeypatch, '--trials', trial, '--groups', json.dumps(groups),
          '--fmo-sets', '{}', '-q', '--out', str(tmp_path / 'pg2'))
    assert '--fast-graph applies only' not in capsys.readouterr().out
    # Leiden with the flag is the case it is for: no note.
    pytest.importorskip('leidenalg')
    _main(monkeypatch, '--trials', trial, '--groups', json.dumps(groups),
          '--fmo-sets', '{}', '-q', '--cluster-method', 'leiden',
          '--fast-graph', '--out', str(tmp_path / 'ld'))
    assert '--fast-graph applies only' not in capsys.readouterr().out


def test_python_entry_points_default_to_the_dense_graph():
    """Code that calls run() or the dialog's run directly, without the
    argument, gets the default graph."""
    import inspect

    from openflo.cli_run import run
    from openflo.editor_analysis import AnalysisMixin
    for fn in (run, AnalysisMixin._run_clustering):
        assert inspect.signature(fn).parameters['fast_graph'].default is False


# ── the Cluster dialog ─────────────────────────────────────────────────────

def _editor_or_skip():
    from tests.test_cluster_dialog import _editor_or_skip as make
    return make()


def _add(ed, df, name='SampleA'):
    from tests.test_cluster_dialog import _add as add
    return add(ed, df, name)


def _walk(w):
    stack = [w]
    while stack:
        x = stack.pop()
        yield x
        stack.extend(x.winfo_children())


def _open(ed):
    ed._open_cluster_dialog()
    return next(w for w in ed.winfo_children()
                if isinstance(w, tk.Toplevel) and w.title() == 'Cluster')


def _press(dlg, text):
    next(w for w in _walk(dlg)
         if isinstance(w, (ttk.Button, ttk.Radiobutton, ttk.Checkbutton))
         and w.cget('text') == text).invoke()


def test_dialog_fast_graph_box_is_leiden_only_off_and_tipped():
    from openflo.tool_window import FAST_GRAPH_TIP
    root, ed = _editor_or_skip()
    try:
        _add(ed, _df())
        tipped = []
        ed._tip = lambda widget, text: tipped.append((widget, text))
        got = []
        ed._run_clustering = lambda **kw: got.append(kw)
        dlg = _open(ed)
        (box,) = ed._cluster_method_rows['fast']
        assert (box, FAST_GRAPH_TIP) in tipped
        assert not box.grid_info()                       # PhenoGraph
        _press(dlg, 'FlowSOM')
        assert not box.grid_info()
        _press(dlg, 'Leiden')
        assert box.grid_info()
        assert not box.instate(['selected'])             # off by default
        _press(dlg, 'Run')
        assert got[-1]['method'] == 'leiden'
        assert got[-1]['fast_graph'] is False
        dlg = _open(ed)
        _press(dlg, 'Leiden')
        _press(dlg, 'Fast graph')                        # tick
        _press(dlg, 'Run')
        assert got[-1]['method'] == 'leiden'
        assert got[-1]['fast_graph'] is True
        # Ticked, then another method picked: not sent to it.
        dlg = _open(ed)
        _press(dlg, 'Leiden')
        _press(dlg, 'Fast graph')
        _press(dlg, 'PhenoGraph')
        _press(dlg, 'Run')
        assert got[-1]['method'] == 'phenograph'
        assert got[-1]['fast_graph'] is False
    finally:
        root.destroy()


def test_dialog_run_uses_the_fast_graph_and_audits_it():
    pytest.importorskip('leidenalg')
    root, ed = _editor_or_skip()
    try:
        df = _df()
        s = _add(ed, df)
        got = {}
        for fast in (False, True):
            with contextlib.redirect_stdout(io.StringIO()):
                ed._run_clustering(method='leiden', all_samples=False, k=K,
                                   grid=10, n_meta=10, embedding='none',
                                   resolution=1.0, fast_graph=fast)
            got[fast] = s.data['leiden'].to_numpy().copy()
        # The dialog run caps Leiden at 200,000 events; N is below it, so it
        # partitions every event, as the direct call does.
        assert np.array_equal(got[False], _direct(df, False))
        assert np.array_equal(got[True], _direct(df, True))
        assert not np.array_equal(got[False], got[True])
        entries = [d for a, d in ed._audited if a == 'cluster']
        assert len(entries) == 2
        assert 'fast_graph' not in entries[0]            # default: unchanged
        assert entries[1]['fast_graph'] is True
    finally:
        root.destroy()


# ── the workspace (Pipeline panel) ─────────────────────────────────────────

def test_workspace_fast_graph_box_uses_the_shared_tip():
    from types import SimpleNamespace
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    try:
        from openflo.tool_window import FAST_GRAPH_TIP
        from openflo.workspace import WorkspacePanel
        tips = {}
        editor = SimpleNamespace(
            _samples={}, _sample_order=[], _sample_trial={},
            _sample_colors={}, _sample_gates={}, _sample_gate_order={},
            _channel_labels={},
            _tip=lambda w, t: tips.__setitem__(str(w), t))
        panel = WorkspacePanel(root, editor=editor)
        v = panel._view
        [box] = [w for w in _walk(v) if isinstance(w, ttk.Checkbutton)
                 and w.cget('text') == 'Fast graph']
        assert tips.get(str(box)) == FAST_GRAPH_TIP
        assert v._sync_run_cfg()['fast_graph'] is False     # pre-filled off
    finally:
        root.destroy()
