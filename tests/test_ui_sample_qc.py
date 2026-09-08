"""Regression tests for the Sample QC dialog's empty-selection guard.

``_markers()`` reads ``names[0]``. Both callers (``_compute`` and
``_export_h5ad``) call it BEFORE their own length check, so opening Sample QC
with nothing checked used to raise an unguarded ``IndexError`` — and because
``_compute`` is armed via ``self.after(50, ...)``, the traceback was swallowed
by the Tk callback and the window just sat blank.
"""
import importlib
import os
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


def _qc_or_skip():
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip("tkinter not available — headless environment")
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:  # noqa: BLE001 — no display
        pytest.skip(str(e))

    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(
        root, fcs_dir=None, labels_str='', on_save=None, primary=False)
    ed.withdraw()

    from openflo.ui_sample_qc import SampleQCWindow
    win = SampleQCWindow(ed)
    win.withdraw()
    return root, ed, win


def _add_sample(ed, name, cols=('FSC-A', 'CD3', 'CD4'), enabled=True):
    df = pd.DataFrame({c: np.linspace(0, 1, 50) for c in cols})
    ed._samples[name] = SimpleNamespace(
        name=name, path=rf'C:\exp\{name}.fcs', data=df,
        fluor_channels=[c for c in cols if c != 'FSC-A'],
        channel_labels={c: c for c in cols})
    ed._sample_order.append(name)
    ed._sample_plot_enabled[name] = enabled
    ed._sample_trial[name] = 'T'


def test_markers_empty_selection_returns_empty_not_indexerror():
    """Zero checked samples -> [], not IndexError on names[0]."""
    root, ed, win = _qc_or_skip()
    try:
        assert win._samples() == []
        assert win._markers() == []
    finally:
        root.destroy()


def test_compute_with_no_selection_is_a_noop():
    """_compute must return quietly rather than raising inside after()."""
    root, ed, win = _qc_or_skip()
    try:
        win._compute()          # must not raise
        assert win._D is None   # nothing computed, nothing drawn
    finally:
        root.destroy()


def test_export_h5ad_with_no_selection_is_a_noop():
    """The second _markers() caller had the same ordering bug."""
    root, ed, win = _qc_or_skip()
    try:
        called = []
        import openflo.ui_sample_qc as mod
        mod.filedialog.asksaveasfilename = lambda *a, **k: called.append(1)
        win._export_h5ad()      # must not raise
        assert called == []     # bailed before prompting for a path
    finally:
        root.destroy()


def test_markers_still_works_with_a_real_selection():
    """The guard must not break the normal path."""
    root, ed, win = _qc_or_skip()
    try:
        _add_sample(ed, 's1')
        _add_sample(ed, 's2')
        assert win._markers() == ['CD3', 'CD4']
    finally:
        root.destroy()
