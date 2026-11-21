"""Shared pieces for the tests that drive a shown editor with three samples.

``editor`` is the sweep editor (tests/test_windows_nothing_hidden.py): three
generated samples alpha_1, alpha_2 and beta_1 with a gate and a child, on a
1920x1080 virtual screen. A test file uses it by importing ``editor`` together
with the three fixtures it asks for (``virtual_screen``, ``no_dialogs``,
``engines_known``). ``loaded`` is an editor that loaded three generated FCS
files (sample_a, sample_b, sample_c) the way File > Open does.

Generated data, synthetic sample names and generic channels only.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import tests.test_windows_nothing_hidden as sweep
from tests.conftest import gui_unavailable


def _s(name, **kw):
    return SimpleNamespace(name=name, metadata=dict(kw.items()))


@pytest.fixture
def editor(virtual_screen, no_dialogs, engines_known):
    virtual_screen(1920, 1080)
    root = sweep._root(1.0)
    try:
        yield root, sweep._editor(root)
    finally:
        root.destroy()


virtual_screen = sweep.virtual_screen
no_dialogs = sweep.no_dialogs
engines_known = sweep.engines_known


def _tag(ed, **per_sample):
    for name, kw in per_sample.items():
        ed._samples[name].metadata = kw


def _set(ed, name, **cols):
    import pandas as pd
    n = len(next(iter(cols.values())))
    df = pd.DataFrame({c: np.zeros(n) for c in sweep.COLS})
    for c, v in cols.items():
        df[c] = v
    ed._samples[name].data = df
    ed._data_changed(name)


def _mask_count(ed, name, gate_name):
    from openflo.gate_eval import cumulative_gate_mask
    gates = ed._sample_gates[name]
    gid = next(g for g, d in gates.items() if d['name'] == gate_name)
    return int(np.count_nonzero(cumulative_gate_mask(
        gates, gid, ed._samples[name].data)))


@pytest.fixture
def loaded(tmp_path, monkeypatch):
    import tkinter as tk

    from tests._background_work import _let_the_load_settle
    from tests.test_beta_look import _load, _write_fcs
    from tests.test_transform_records import (
        _new_editor,
        capture_warnings,
        isolate_home,
    )
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        root = tk.Tk()
    except Exception as e:                                 # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    ed = _new_editor(root)
    _load(root, ed, [('sample_a', _write_fcs(tmp_path / 'a.fcs', seed=1)),
                     ('sample_b', _write_fcs(tmp_path / 'b.fcs', seed=2)),
                     ('sample_c', _write_fcs(tmp_path / 'c.fcs', seed=3))])
    # The load's debounced settle pass writes "3 sample(s) loaded" to the
    # status bar 120 ms later; left pending, a slow runner ran it after an
    # export had written its answer there.
    _let_the_load_settle(root, ed)
    yield root, ed, tmp_path
    root.destroy()

