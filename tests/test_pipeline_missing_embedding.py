"""Pipeline (beta) says when an embedding it was asked for can't run.

TriMap was ticked by default. Without the [embed] extra FlowSample skipped it
with only a log line, and the run ended "Done: 4 ok" with no TriMap output and
nothing saying why. Now a backend that isn't installed is unticked and
disabled in the panel, and a run asked for it anyway (a saved config) lists
it as skipped in the run's summary.
"""
import io
import sys
from tkinter import ttk

import numpy as np
import pandas as pd
import pytest

import openflo.workspace as W
from tests.conftest import gui_unavailable

MISSING = 'openflo_test_no_such_backend'


@pytest.fixture
def no_trimap(monkeypatch):
    monkeypatch.setitem(W.EXTRA_EMBEDDINGS, 'trimap', (MISSING, 'TriMap'))


def test_a_missing_backend_is_reported_as_not_installed(no_trimap):
    assert not W.embedding_available('trimap')
    assert 'not installed' in W.embedding_unavailable('trimap')
    assert W.embedding_available('umap')


def test_a_backend_that_fails_to_import_is_reported(monkeypatch, tmp_path):
    # A generated package whose own import fails: installed, can't run.
    pkg = tmp_path / 'ofl_test_broken_backend'
    pkg.mkdir()
    (pkg / '__init__.py').write_text(
        "raise ImportError('a dependency is missing')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    from openflo import capabilities
    monkeypatch.setattr(capabilities, '_IMPORT_RESULT', {})
    monkeypatch.setitem(W.EXTRA_EMBEDDINGS, 'trimap',
                        ('ofl_test_broken_backend', 'TriMap'))
    assert W.embedding_available('trimap')            # the panel's quick look
    why = W.embedding_unavailable('trimap', load=True)
    assert why and 'fails to load' in why
    assert 'a dependency is missing' in why
    assert not W.embedding_available('trimap')       # and now everywhere


def test_the_run_summary_names_the_skipped_embedding(no_trimap, tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({c: rng.normal(0, 1, 200)
                       for c in ('FL1-A', 'FL2-A', 'FL3-A')})
    prep = {'data': df, 'channels': list(df.columns), 'label': 'b1',
            'n_events': 200, 'note': '', 'color_by': None}
    cfg = W.default_run_cfg()
    cfg.update(method='phenograph', umap=False, tsne=False, phate=False,
               trimap=True, pacmap=False, max_events=0)
    _so = sys.stdout
    sys.stdout = io.StringIO()                       # silence cluster chatter
    try:
        r = W.compute_run(dict(prep), cfg, str(tmp_path))
    finally:
        sys.stdout = _so
    assert r['ok'] and not r['trimap']
    assert any('TriMap skipped: not installed' in w for w in r['warnings'])


def test_the_panel_unticks_and_disables_a_missing_backend(no_trimap):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f'Tk unavailable: {exc}')
        return
    try:
        view = W.PipelineWorkspaceView(root, W.WorkspaceModel())
        view.pack()
        root.update()
        assert view._trimap_var.get() is False
        boxes = {}
        stack = [view]
        while stack:
            w = stack.pop()
            stack.extend(w.winfo_children())
            if isinstance(w, ttk.Checkbutton):
                boxes[w.cget('text')] = w
        assert boxes['TriMap'].instate(['disabled'])
        assert not boxes['UMAP'].instate(['disabled'])
        view.model.run_cfg = dict(W.default_run_cfg(), trimap=True)
        view._load_cfg_into_widgets()
        assert view._trimap_var.get() is False
    finally:
        root.destroy()
