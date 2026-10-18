"""Quick preview (Tools -> Quick preview…) — load, plot, quadrant %, export.

preview.annotate_quadrants is tested; the dialog's _load_sample / _plot /
_export were 3-20% covered. Ground truth: 1000 raw events built so that,
with thresholds X = 100 and Y = 100, the quadrants hold exactly
lo/lo 10%, hi/lo 20%, lo/hi 30%, hi/hi 40%.
"""
from __future__ import annotations

import os
import tkinter as tk

import numpy as np
import pandas as pd

from tests.conftest import gui_unavailable


def _root_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        root = tk.Tk()
    except Exception as e:                          # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    root.status_var = tk.StringVar(master=root, value='')
    return root


def _write(path, seed=0):
    from openflo.fcs_export import write_fcs
    rng = np.random.default_rng(seed)
    blocks = [(100, 50, 50), (200, 150, 50), (300, 50, 150), (400, 150, 150)]
    x = np.concatenate([rng.uniform(cx - 40, cx + 40, n)
                        for n, cx, _ in blocks])
    y = np.concatenate([rng.uniform(cy - 40, cy + 40, n)
                        for n, _, cy in blocks])
    write_fcs(pd.DataFrame({'FSC-A': x * 100, 'CD3-A': x, 'CD4-A': y}),
              str(path))


def _texts(dlg):
    """{'<x side>/<y side>': label} — where each % label sits on the axes."""
    ax = dlg._ax
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    out = {}
    for t in ax.texts:
        x, y = t.get_position()
        key = (('hi' if x > (x0 + x1) / 2 else 'lo') + '/'
               + ('hi' if y > (y0 + y1) / 2 else 'lo'))
        out[key] = t.get_text()
    return out


def test_quick_preview_quadrant_percentages_and_export(tmp_path):
    from openflo import ui_preview as mod
    p = tmp_path / 's.fcs'
    _write(p)
    root = _root_or_skip()
    try:
        dlg = mod.QuickPreviewDialog(root)
        dlg._load_sample(str(p))
        assert 's.fcs  (1,000 events)' == dlg._file_var.get()
        assert list(dlg._x_combo['values']) == ['FSC-A', 'CD3-A', 'CD4-A']
        dlg._x_combo.set('CD3-A')
        dlg._y_combo.set('CD4-A')
        dlg._xthr_var.set('100')
        dlg._ythr_var.set('100')
        dlg._plot()
        assert _texts(dlg) == {'lo/lo': '10.0%', 'hi/lo': '20.0%',
                               'lo/hi': '30.0%', 'hi/hi': '40.0%'}
        # raw (untransformed) values are plotted: the axis spans 10..190
        lo, hi = dlg._ax.get_xlim()
        assert lo < 15 and hi > 185
        assert root.status_var.get() == "Quick preview: plotted CD3-A vs CD4-A"
        # one threshold only -> a two-way split, not four numbers
        dlg._ythr_var.set('')
        dlg._plot()
        # X <= 100 holds blocks 1+3 (10%+30%), X > 100 blocks 2+4 (60%)
        assert _texts(dlg) == {'lo/hi': '40.0%', 'hi/hi': '60.0%'}
        out = tmp_path / 'p.png'
        saved = mod.filedialog.asksaveasfilename
        mod.filedialog.asksaveasfilename = lambda *a, **k: str(out)
        try:
            dlg._export()
        finally:
            mod.filedialog.asksaveasfilename = saved
        assert out.read_bytes()[:4] == b'\x89PNG'
    finally:
        root.destroy()


def test_quick_preview_bad_file_reports_and_keeps_state(tmp_path):
    from openflo import ui_preview as mod
    bad = tmp_path / 'bad.fcs'
    bad.write_bytes(b'not an fcs file')
    root = _root_or_skip()
    errors = []
    saved = mod.messagebox.showerror
    mod.messagebox.showerror = lambda *a, **k: errors.append(a)
    try:
        dlg = mod.QuickPreviewDialog(root)
        dlg._load_sample(str(bad))
        assert dlg._sample is None and len(errors) == 1
        assert root.status_var.get().startswith(
            "Quick preview: load failed:")
    finally:
        mod.messagebox.showerror = saved
        root.destroy()
