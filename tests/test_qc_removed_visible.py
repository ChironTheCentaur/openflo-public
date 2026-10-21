"""Events the editor's load QC removes are said, not silently gone.

The editor runs acquisition QC on every FCS it loads and keeps only the
events QC passes, before any gate. FlowJo has no such step and the .wsp
export has no QC population, so the same gates count differently in the two
(measured: 20,000 events -> 19,614; 'FL1 bright' 6,371 in FlowJo, 5,985 in
the editor, %Total 31.86 vs 30.51), with nothing saying why. QC itself is
unchanged; the load records how many events it removed, the load status says
so, and the .wsp export lists it among what FlowJo will not reproduce.
"""
from __future__ import annotations

import os

import numpy as np

import openflo.pipeline as fp
from tests.conftest import gui_unavailable


def _fcs(path, n=20_000):
    import flowio
    rng = np.random.default_rng(5)
    fl1 = np.where(rng.random(n) < 0.3, rng.normal(20000, 4000, n),
                   rng.normal(0, 150, n))
    fl1[rng.random(n) < 0.02] = 262143.0         # saturated: QC's margin check
    ev = np.column_stack([rng.uniform(2e4, 2e5, n), rng.uniform(1e4, 1.5e5, n),
                          fl1, np.arange(n) * 5.0]).astype(np.float32)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.ravel().tolist(),
                          ['FSC-A', 'SSC-A', 'FL1-A', 'Time'],
                          metadata_dict={'TIMESTEP': '0.01'})
    return str(path)


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def test_load_records_and_reports_qc_removed_events(tmp_path, monkeypatch):
    import openflo.prefs as prefs
    monkeypatch.setattr(prefs, 'load_qc_enabled', lambda: True)
    fcs = _fcs(tmp_path / 'q.fcs')
    n_file = len(fp.FlowSample(fcs).data)
    expect = fp.FlowSample(fcs).run_qc()
    removed = n_file - len(expect.data)
    assert removed > 300                           # the fixture trips QC
    assert expect.qc_removed == removed            # run_qc records it
    root, ed = _editor_or_skip()
    try:
        landed = []
        ed.after = lambda ms, fn=None, *a: landed.append(fn) or 'id'
        ed._load_worker('q', fcs)
        (land,) = landed
        ed.after = lambda ms, fn=None, *a: 'id'     # settle runs below
        land()
        s = ed._samples['q']
        assert len(s.data) == n_file - removed
        assert s.qc_removed == removed
        assert f'{removed:,}' in ed.status_var.get()
        ed._on_load_settled()
        assert f'{removed:,}' in ed.status_var.get()
        items = ed._wsp_lossy_summary()
        (qc,) = [i for i in items if 'QC' in i]
        assert f'{removed:,}' in qc and 'FlowJo' in qc
    finally:
        root.destroy()


def test_no_qc_line_when_nothing_was_removed():
    root, ed = _editor_or_skip()
    try:
        assert not [i for i in ed._wsp_lossy_summary() if 'QC' in i]
    finally:
        root.destroy()


def test_load_qc_is_off_unless_turned_on(tmp_path, monkeypatch):
    """Off by default: FlowJo has no QC step, so a load keeps every event
    and the same gates count the same as there. Turned on in Preferences it
    runs; a sample coming back from a session or another window keeps the
    choice it was loaded with, whatever the setting says now."""
    import openflo.prefs as prefs
    monkeypatch.setattr(prefs, 'read_prefs', lambda: {})
    assert prefs.load_qc_enabled() is False
    fcs = _fcs(tmp_path / 'q.fcs')
    n_file = len(fp.FlowSample(fcs).data)
    root, ed = _editor_or_skip()
    try:
        landed = []
        ed.after = lambda ms, fn=None, *a: landed.append(fn) or 'id'
        ed._load_worker('q', fcs)
        landed[-1]()
        s = ed._samples['q']
        assert len(s.data) == n_file and s.qc_removed == 0
        assert s.qc_applied is False
        assert not [i for i in ed._wsp_lossy_summary() if 'QC' in i]

        # A session sample saved with QC reloads with QC, setting off.
        ed._samples.pop('q')
        ed.stage_sample_prep(fcs, qc=True)
        ed._load_worker('q', fcs)
        landed[-1]()
        s = ed._samples['q']
        assert s.qc_applied is True and len(s.data) < n_file

        # ...and one saved without, setting on, reloads without.
        monkeypatch.setattr(prefs, 'load_qc_enabled', lambda: True)
        ed._samples.pop('q')
        ed.stage_sample_prep(fcs, qc=False)
        ed._load_worker('q', fcs)
        landed[-1]()
        assert len(ed._samples['q'].data) == n_file
    finally:
        root.destroy()


def test_a_session_records_whether_qc_ran():
    import types

    from openflo.gui import ViewGateEditorWindow as V

    class _Var:
        def __init__(self, v): self._v = v
        def get(self): return self._v
    smp = types.SimpleNamespace(path='/d/a.fcs', qc_applied=True,
                                comp_matrix=None, comp_channels=[])
    st = types.SimpleNamespace(
        _sample_order=['a'], _samples={'a': smp}, _sample_colors={},
        _sample_plot_enabled={}, _sample_trial={}, _sample_is_comp={},
        _sample_gates={}, _sample_gate_order={}, _channel_range={},
        _channel_scale={}, _channel_labels={}, _active_sample='a',
        mode_var=_Var('scatter'), x_combo=_Var(''), y_combo=_Var(''),
        color_combo=_Var(''), ds_display_var=_Var(False),
        ds_propagate_var=_Var(False), max_points_var=_Var(1000),
        show_removed_var=_Var(False), contour_scatter_var=_Var(False),
        contour_outliers_var=_Var(False), hist_y_mode=_Var('count'),
        _cluster_labels={},
        _audit_log=types.SimpleNamespace(to_list=lambda: []))
    entry = V._session_state(st)['samples'][0]
    assert entry['qc_applied'] is True
