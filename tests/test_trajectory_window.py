"""Analyze -> Trajectory recovers a known maturation order and exports it.

ui_trajectory.TrajectoryWindow._export / _trends_frame never ran under the
suite, and _compute only on fakes. Cells on a 1-D path (t ~ U(0, 1); CD902 falls
and CD901b rises with t) split into a Day1 (t < .5) / Day3 (t >= .5) series give
a known answer: pseudotime ~ t, written back to the right sample's rows.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

from openflo.async_task import wait_for_job  # noqa: E402


def _window(monkeypatch, scale=1.0, logicle=False):
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 1, 1000)
    df = pd.DataFrame({'CD902-A': 10 * (1 - t) + rng.normal(0, .2, t.size),
                       'CD901b-A': 10 * t + rng.normal(0, .2, t.size)}) * scale
    root, ed, gui = _editor_or_skip()
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *a, **k: None)
    truth = {}
    # Day3 is loaded FIRST so a write-back that ignores sample offsets shows.
    for name, sel in (('Day3', t >= .5), ('Day1', t < .5)):
        s = FlowSample.from_dataframe(df[sel].reset_index(drop=True), name=name,
                                      labels={'CD902-A': 'CD902', 'CD901b-A': 'CD901b'})
        truth[name] = t[sel]
        if logicle:                       # as the editor's loader stores it
            s.apply_transform(['CD902-A', 'CD901b-A'])
        ed._samples[name] = s
        ed._sample_order.append(name)
        ed._sample_trial[name] = name
        ed._sample_plot_enabled[name] = True
        ed._sample_gates[name] = {}
        ed._sample_gate_order[name] = []
    ed._channels = ['CD902-A', 'CD901b-A']
    ed._channel_labels = {'CD902-A': 'CD902', 'CD901b-A': 'CD901b'}
    ed._active_sample = 'Day1'
    from openflo.ui_trajectory import TrajectoryWindow
    w = TrajectoryWindow(ed)
    w.withdraw()
    return root, ed, w, truth


def test_pseudotime_recovers_the_path_per_sample(monkeypatch):
    from scipy.stats import spearmanr
    root, ed, w, truth = _window(monkeypatch)
    try:
        assert w.root_var.get().startswith('CD902')     # stemness default
        w._compute()
        assert wait_for_job(w._busy)
        for name in ('Day1', 'Day3'):
            pt = ed._samples[name].data['pseudotime'].to_numpy()
            assert spearmanr(pt, truth[name]).statistic > 0.95
        assert (ed._samples['Day3'].data['pseudotime'].mean()
                > ed._samples['Day1'].data['pseudotime'].mean() + 0.3)
        tf = w._trends_frame()
        assert tf.shape == (20, 3)
        assert tf.iloc[0, 1] > 8 and tf.iloc[-1, 1] < 2      # CD902 falls
        assert tf.iloc[0, 2] < 2 and tf.iloc[-1, 2] > 8      # CD901b rises
        w.dir_var.set('Low')                                 # root at CD902-low
        w._compute()
        assert wait_for_job(w._busy)
        pt = ed._samples['Day1'].data['pseudotime'].to_numpy()
        assert spearmanr(pt, truth['Day1']).statistic < -0.95
    finally:
        root.destroy()


def test_trend_exports(monkeypatch, tmp_path):
    import openflo.ui_trajectory as ut
    root, _ed, w, _truth = _window(monkeypatch)
    try:
        w._compute()
        assert wait_for_job(w._busy)
        for kind in ('tidy', 'prism'):
            out = tmp_path / f'{kind}.csv'
            monkeypatch.setattr(ut.filedialog, 'asksaveasfilename',
                                lambda out=out, **k: str(out))
            w._export(kind)
            got = pd.read_csv(out)
            assert list(got.columns) == ['pseudotime',
                                         'CD902 (CD902-A) mean (linear)',
                                         'CD901b (CD901b-A) mean (linear)']
            assert np.allclose(got['pseudotime'], np.arange(20) / 20 + 0.025)
            assert np.allclose(got.iloc[:, 1:].to_numpy(), w._means_linear,
                               equal_nan=True)
    finally:
        root.destroy()


def test_exported_trends_are_linear_intensities(monkeypatch, tmp_path):
    """The editor stores fluor channels on logicle, and the trends table
    (tooltip: "marker intensity against pseudotime") exported the per-bin
    means of those coordinates: CD902 falling from ~9,860 to ~-160 (linear
    bin means) was written as 0.682 -> -0.037. The export now holds the
    linear per-bin means, named "(linear)"; the drawn curves stay on the
    display scale and say so."""
    import openflo.ui_trajectory as ut
    from openflo.pipeline import inverse_transform_values
    root, ed, w, _truth = _window(monkeypatch, scale=1000.0, logicle=True)
    try:
        w._compute()
        assert wait_for_job(w._busy)
        out = tmp_path / 'prism.csv'
        monkeypatch.setattr(ut.filedialog, 'asksaveasfilename',
                            lambda **k: str(out))
        w._export('prism')
        got = pd.read_csv(out)
        # Truth: each bin's mean of the linear values of the events whose
        # pseudotime falls in it (the graph itself runs on display values).
        days = [ed._samples[n] for n in ('Day3', 'Day1')]
        pt = np.concatenate([s.data['pseudotime'].to_numpy() for s in days])
        b = np.clip((pt * 20).astype(int), 0, 19)
        for col, ch in (('CD902 (CD902-A) mean (linear)', 'CD902-A'),
                        ('CD901b (CD901b-A) mean (linear)', 'CD901b-A')):
            lin = np.concatenate([inverse_transform_values(
                s.data[ch].to_numpy(float)) for s in days])
            want = [lin[b == k].mean() if (b == k).any() else np.nan
                    for k in range(20)]
            np.testing.assert_allclose(got[col], want, rtol=1e-6)
        cd902 = got['CD902 (CD902-A) mean (linear)']
        assert cd902.iloc[0] > 8000 and cd902.dropna().iloc[-1] < 1000
        assert np.nanmax(w._means) < 1.1               # drawn: logicle units
        assert 'display scale' in w._fig.axes[0].get_ylabel()
    finally:
        root.destroy()


def test_exports_to_a_folder_that_does_not_exist_say_so(monkeypatch,
                                                        tmp_path):
    """Trends CSV, Prism XY and Export figure to a path whose folder does
    not exist each show "could not be saved" in the error dialog; before,
    the OSError escaped to Tk's error handler as a traceback. Nothing is
    written and no export is recorded."""
    from tkinter import messagebox

    import openflo.ui_trajectory as ut
    root, _ed, w, _truth = _window(monkeypatch)
    try:
        w._compute()
        assert wait_for_job(w._busy)
        errors, recorded = [], []
        monkeypatch.setattr(messagebox, 'showerror',
                            lambda title, msg, **k: errors.append(msg))
        monkeypatch.setattr(ut, 'record_export',
                            lambda *a, **k: recorded.append(a))
        missing = tmp_path / 'no_such_folder'
        for kind, name, what in (('tidy', 'x.csv', 'table'),
                                 ('prism', 'x.csv', 'table'),
                                 ('figure', 'x.png', 'figure')):
            out = missing / name
            monkeypatch.setattr(ut.filedialog, 'asksaveasfilename',
                                lambda out=out, **k: str(out))
            errors.clear()
            w._export(kind)
            assert len(errors) == 1, (kind, errors)
            assert errors[0].startswith(f'The {what} could not be saved.')
            assert str(out) in errors[0]
        assert not missing.exists() and recorded == []
    finally:
        root.destroy()


def test_a_changed_setting_marks_the_result_as_from_the_last_compute(
        monkeypatch):
    """Root end, Root marker or Neighbors changed after Compute: the status
    line no longer reads "Done" as if the curves were for the new settings."""
    root, _ed, w, _truth = _window(monkeypatch)
    try:
        w._compute()
        assert wait_for_job(w._busy)
        done = w.status.cget('text')
        assert done.startswith('Done')
        w.dir_var.set('Low')
        stale = w.status.cget('text')
        assert stale.startswith('Settings changed')
        assert f"root {w.root_var.get()} (High), 15 neighbors" in stale
        w.dir_var.set('High')                    # back as computed
        assert w.status.cget('text') == done
        w.k_var.set('20')
        assert w.status.cget('text').startswith('Settings changed')
        other = next(v for v in w.root_combo.cget('values')
                     if v != w.root_var.get())
        w.k_var.set('15')
        w.root_var.set(other)
        assert w.status.cget('text').startswith('Settings changed')
        w._compute()
        assert wait_for_job(w._busy)
        assert w.status.cget('text').startswith('Done')
    finally:
        root.destroy()
