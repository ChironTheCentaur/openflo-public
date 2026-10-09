"""Compensation editor (Tools -> Compensation…) — the action paths.

tests/test_compensation.py covers the maths (FlowSample._apply_comp,
optimize_compensation) and tests/test_comp_editor.py the cell colours; the
editor's Apply / Save / Load / Optimize actions and the GUI callback that
re-compensates a loaded sample had ~5-25% coverage.

Ground truth: true signals T and a spillover matrix M (row = source,
col = destination); measured = T @ M is written to an FCS with NO $SPILL.
Correct compensation in the editor returns logicle(T), the scale the GUI
loader leaves fluor channels on.

Tk-guarded like tests/test_gui_ux.py.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from tests.conftest import gui_unavailable

FL = ['FLA-A', 'FLB-A', 'FLC-A']
COLS = ['FSC-A', 'SSC-A'] + FL
M = np.array([[1.0, 0.20, 0.05],
              [0.10, 1.0, 0.15],
              [0.00, 0.08, 1.0]])


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
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
        on_done(work()) if on_done else work())
    return root, ed


def _logicle(x):
    from openflo.pipeline import transform_values
    x = np.asarray(x, float)
    return np.column_stack([
        np.asarray(transform_values(x[:, j].copy(), method='logicle'), float)
        for j in range(x.shape[1])])


def _write_mixed(path, n=1500, seed=11):
    from openflo.fcs_export import write_fcs
    rng = np.random.default_rng(seed)
    T = rng.uniform(100, 20000, (n, 3))
    scat = rng.uniform(20000, 200000, (n, 2))
    write_fcs(pd.DataFrame(np.hstack([scat, T @ M]), columns=pd.Index(COLS)),
              str(path))
    return T


def _load(root, ed, items):
    """Load FCS files through the editor's real loader."""
    for name, path in items:
        ed._load_worker(name, str(path))
    for _ in range(100):
        root.update()
        if all(n in ed._samples for n, _ in items):
            break
    assert all(n in ed._samples for n, _ in items)


def _comp_window(ed):
    from openflo.ui_comp import CompensationEditorWindow
    ed._open_comp_editor()
    wins = [w for w in ed.winfo_children()
            if isinstance(w, CompensationEditorWindow)]
    assert len(wins) == 1
    return wins[0]


def test_comp_editor_apply_recovers_true_signal(tmp_path):
    """Matrix entered in the editor + Apply -> the active sample's fluor
    channels equal logicle(T); scatter is untouched; the applied matrix is
    kept on the sample; Compensation QC then opens on it."""
    root, ed = _editor_or_skip()
    try:
        T = _write_mixed(tmp_path / 'a.fcs')
        _load(root, ed, [('a', tmp_path / 'a.fcs')])
        ed._set_active_sample('a')
        s = ed._samples['a']
        fsc = s.data['FSC-A'].to_numpy(float).copy()
        win = _comp_window(ed)
        win._set_matrix(FL, M)
        win._apply()
        np.testing.assert_allclose(s.data[FL].to_numpy(float), _logicle(T),
                                   atol=1e-5)
        np.testing.assert_allclose(s.data['FSC-A'].to_numpy(float), fsc)
        np.testing.assert_allclose(s.comp_matrix, M)
        assert list(s.comp_channels) == FL
        assert "Applied 3×3 matrix to 'a'" in ed.status_var.get()

        from openflo import ui_figure_window
        opened = []
        saved = ui_figure_window._FigureWindow
        ui_figure_window._FigureWindow = lambda p, fig, title: opened.append(
            title)
        try:
            ed._open_comp_qc()
        finally:
            ui_figure_window._FigureWindow = saved
        assert opened == ['Compensation QC — a']
    finally:
        root.destroy()


def test_comp_editor_reopen_shows_applied_matrix_and_apply_is_not_compounded(
        tmp_path):
    """Reopening shows the applied matrix ('currently applied'); a second
    Apply re-reads the raw FCS rather than compensating twice."""
    root, ed = _editor_or_skip()
    try:
        T = _write_mixed(tmp_path / 'a.fcs')
        _load(root, ed, [('a', tmp_path / 'a.fcs')])
        ed._set_active_sample('a')
        win = _comp_window(ed)
        win._set_matrix(FL, M)
        win._apply()
        win.destroy()
        win2 = _comp_window(ed)
        assert 'currently applied' in win2.status_var.get()
        np.testing.assert_allclose(win2._read_matrix_from_entries(), M,
                                   atol=1e-6)
        win2._apply()
        np.testing.assert_allclose(ed._samples['a'].data[FL].to_numpy(float),
                                   _logicle(T), atol=1e-5)
    finally:
        root.destroy()


def test_comp_editor_csv_save_then_load_round_trips(tmp_path):
    root, ed = _editor_or_skip()
    try:
        from tkinter import filedialog

        from openflo.ui_comp import CompensationEditorWindow
        win = CompensationEditorWindow(root, sample=None)
        win._set_matrix(FL, M)
        win.entries[(0, 1)][0].set('0.25')          # a hand edit
        out = str(tmp_path / 'comp.csv')
        saved = (filedialog.asksaveasfilename, filedialog.askopenfilename)
        filedialog.asksaveasfilename = lambda *a, **k: out
        filedialog.askopenfilename = lambda *a, **k: out
        try:
            win._save()
            win._reset_identity()
            np.testing.assert_allclose(win._read_matrix_from_entries(),
                                       np.eye(3))
            win._load()
        finally:
            filedialog.asksaveasfilename, filedialog.askopenfilename = saved
        expect = M.copy()
        expect[0, 1] = 0.25
        assert win.channels == FL
        np.testing.assert_allclose(win._read_matrix_from_entries(), expect)
    finally:
        root.destroy()


def test_comp_editor_apply_rejects_non_numeric_cell():
    root, ed = _editor_or_skip()
    try:
        from openflo.ui_comp import CompensationEditorWindow
        got = []
        win = CompensationEditorWindow(root, sample=None,
                                       on_apply=lambda c, m: got.append(m))
        win._set_matrix(FL, M)
        win.entries[(2, 0)][0].set('abc')
        win._apply()
        assert got == []
        assert win.status_var.get() == "Cell [2, 0] is not a number."
    finally:
        root.destroy()


def test_optimize_dialog_run_hands_recovered_matrix_to_editor(tmp_path):
    """'Optimize from single-stains…' -> Run: the editor receives the
    spillover estimated from single stains built from M."""
    from openflo.fcs_export import write_fcs
    root, ed = _editor_or_skip()
    try:
        from openflo.ui_comp import CompensationEditorWindow, OptimizeCompensationDialog
        rng = np.random.default_rng(5)
        n = 3000
        win = CompensationEditorWindow(root, sample=None)
        win._set_matrix(FL, np.eye(3))
        dlg = OptimizeCompensationDialog(win, FL, on_complete=win._set_matrix)
        for i, ch in enumerate(FL):
            Ti = rng.uniform(0, 50, (n, 3))                 # autofluorescence
            Ti[:, i] += np.where(rng.random(n) < 0.3,
                                 rng.uniform(5000, 50000, n),
                                 rng.uniform(0, 200, n))
            scat = rng.uniform(20000, 200000, (n, 2))
            p = tmp_path / f'ss_{i}.fcs'
            write_fcs(pd.DataFrame(np.hstack([scat, Ti @ M]),
                                   columns=pd.Index(COLS)), str(p))
            dlg.path_vars[ch].set(str(p))
        dlg._run()
        assert win.channels == FL
        np.testing.assert_allclose(win._read_matrix_from_entries(), M,
                                   atol=0.01)
    finally:
        root.destroy()


def test_headerless_sibling_csv_is_labelled_with_fluor_channels(tmp_path):
    # The first N data columns put FSC-A/SSC-A on the matrix, so Apply
    # compensated scatter and left FLB/FLC alone.
    root, ed = _editor_or_skip()
    try:
        T = _write_mixed(tmp_path / 'a.fcs')
        np.savetxt(tmp_path / 'compensation.csv', M, delimiter=',')
        _load(root, ed, [('a', tmp_path / 'a.fcs')])
        ed._set_active_sample('a')
        win = _comp_window(ed)
        assert win.channels == FL
        win._apply()
        np.testing.assert_allclose(ed._samples['a'].data[FL].to_numpy(float),
                                   _logicle(T), atol=1e-5)
    finally:
        root.destroy()


def test_headerless_csv_load_labels_fluor_channels_or_refuses(tmp_path):
    """Load… of a header-less CSV: an N x N matrix on a sample with N fluor
    channels takes their names; any other size is refused, not guessed."""
    root, ed = _editor_or_skip()
    try:
        from tkinter import filedialog
        _write_mixed(tmp_path / 'a.fcs')
        _load(root, ed, [('a', tmp_path / 'a.fcs')])
        ed._set_active_sample('a')
        win = _comp_window(ed)
        win._reset_identity()
        np.savetxt(tmp_path / 'm3.csv', M, delimiter=',')
        np.savetxt(tmp_path / 'm2.csv', M[:2, :2], delimiter=',')
        saved = filedialog.askopenfilename
        try:
            filedialog.askopenfilename = lambda *a, **k: str(tmp_path / 'm2.csv')
            win._load()
            assert win.channels == FL                      # unchanged
            np.testing.assert_allclose(win._read_matrix_from_entries(), np.eye(3))
            assert 'no header row' in win.status_var.get()
            filedialog.askopenfilename = lambda *a, **k: str(tmp_path / 'm3.csv')
            win._load()
        finally:
            filedialog.askopenfilename = saved
        assert win.channels == FL
        np.testing.assert_allclose(win._read_matrix_from_entries(), M)
    finally:
        root.destroy()


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT: the Apply tooltip says 'Apply this matrix to every loaded "
    "sample' but ToolsMixin._open_comp_editor._on_apply re-compensates only "
    "the active sample. Fix either the tooltip or the behaviour; remove "
    "this marker then."))
def test_apply_tooltip_matches_what_apply_does(tmp_path):
    import openflo.ui_comp as uc
    root, ed = _editor_or_skip()
    tips = []
    saved = uc.auto_tip
    uc.auto_tip = lambda w, text: tips.append((w, text))
    try:
        T = _write_mixed(tmp_path / 'a.fcs')
        _write_mixed(tmp_path / 'b.fcs')
        _load(root, ed, [('a', tmp_path / 'a.fcs'), ('b', tmp_path / 'b.fcs')])
        ed._set_active_sample('a')
        win = _comp_window(ed)
        apply_tip = [t for w, t in tips if str(w.cget('text')) == 'Apply']
        assert len(apply_tip) == 1
        win._set_matrix(FL, M)
        win._apply()
        if 'every loaded sample' in apply_tip[0]:
            np.testing.assert_allclose(
                ed._samples['b'].data[FL].to_numpy(float), _logicle(T),
                atol=1e-5)
    finally:
        uc.auto_tip = saved
        root.destroy()


def _autodetect(root, folder, chans):
    from tkinter import filedialog

    from openflo.ui_comp import OptimizeCompensationDialog
    saved = filedialog.askdirectory
    filedialog.askdirectory = lambda *a, **k: str(folder)
    try:
        dlg = OptimizeCompensationDialog(root, chans, on_complete=lambda c, m: None)
        dlg._autofill_from_dir()
    finally:
        filedialog.askdirectory = saved
    return ({c: os.path.basename(v.get()) for c, v in dlg.path_vars.items()},
            dlg.status_var.get())


@pytest.mark.parametrize('pattern,sep', [
    ('Compensation Controls_{} Stained Control.fcs', '-'),   # BD FACSDiva
    ('{}-A.fcs', '-'),                                       # the channel's name
    ('ss_{}.fcs', '_'),
    ('{} single stain.fcs', ''),                             # 'FLBCy7'
])
def test_autodetect_matches_single_stains_named_after_their_channel(
        tmp_path, pattern, sep):
    # Substring tokens left FLB-A unmatched and gave FLB-Cy7-A the FLC-Cy7 file.
    root, ed = _editor_or_skip()
    try:
        chans = ['FLA-A', 'FLB-A', 'FLB-Cy7-A', 'FLC-A', 'FLC-Cy7-A', 'Horizon FLD-A']
        name = {c: pattern.format(c[:-2].replace('Horizon ', '').replace('-', sep))
                for c in chans}
        for f in [*name.values(), pattern.format('Unstained')]:
            (tmp_path / f).write_bytes(b'')
        got, status = _autodetect(root, tmp_path, chans)
        assert got == name
        assert status.startswith('Matched 6/6')
    finally:
        root.destroy()


def test_autodetect_never_guesses_between_equal_candidates(tmp_path):
    root, ed = _editor_or_skip()
    try:
        for f in ('FLB-A.fcs', 'FLB-Cy7-A.fcs', 'FLC-A run1.fcs', 'FLC-A run2.fcs'):
            (tmp_path / f).write_bytes(b'')
        # No FLB-Cy7 channel: the closer name wins. Two FLC files: neither.
        got, status = _autodetect(root, tmp_path, ['FLB-A', 'FLC-A'])
        assert got == {'FLB-A': 'FLB-A.fcs', 'FLC-A': ''}
        assert 'ambiguous, left empty: FLC-A matched 2 files' in status
    finally:
        root.destroy()


_BD = 'Compensation Controls_{} Stained Control.fcs'
_FORTESSA = ['FLA', 'FLB', 'FLB-Texas Red', 'FLG-Cy5-5', 'FLB-Cy7', 'FLC',
             'Alexa Fluor 700', 'FLC-Cy7', 'FLD', 'FLE', 'FLF', 'BUV395']


@pytest.mark.parametrize('chans,files,want', [
    # BD FACSDiva, a 12-colour Fortessa panel, plus the unstained control.
    ([f'{d}-A' for d in _FORTESSA],
     [_BD.format(d) for d in _FORTESSA] + [_BD.format('Unstained')],
     {f'{d}-A': _BD.format(d) for d in _FORTESSA}),
    # Marker names in the channel must not decide: CD4 FLB-Cy7-A took FLB.fcs.
    (['CD279 PD-1 FLB-A', 'CD4 FLB-Cy7-A'], ['FLB.fcs', 'FLB-Cy7 single stain.fcs'],
     {'CD279 PD-1 FLB-A': 'FLB.fcs', 'CD4 FLB-Cy7-A': 'FLB-Cy7 single stain.fcs'}),
    # Area and height of one fluor share its control.
    (['FLA-A', 'FLA-H', 'FLB-A', 'FLB-H', 'FLC-A', 'FLC-H'],
     ['FLA.fcs', 'FLB.fcs', 'FLC.fcs'],
     {'FLA-A': 'FLA.fcs', 'FLA-H': 'FLA.fcs', 'FLB-A': 'FLB.fcs',
      'FLB-H': 'FLB.fcs', 'FLC-A': 'FLC.fcs', 'FLC-H': 'FLC.fcs'}),
    # Joined names.
    (['Alexa Fluor 647-A', 'FLA-A', 'eFluor 450-A'],
     ['AF647.fcs', 'CD3FLA.fcs', 'eF450.fcs'],
     {'Alexa Fluor 647-A': 'AF647.fcs', 'FLA-A': 'CD3FLA.fcs',
      'eFluor 450-A': 'eF450.fcs'}),
    (['FLA-A', 'FLB-A', 'FLC-A'], ['compFLA.fcs', 'compFLB.fcs', 'compFLC.fcs'],
     {'FLA-A': 'compFLA.fcs', 'FLB-A': 'compFLB.fcs', 'FLC-A': 'compFLC.fcs'}),
    (['FLA-A', 'FLB-A', 'FLC-A'], ['FLA1.fcs', 'FLB1.fcs', 'FLC1.fcs'],
     {'FLA-A': 'FLA1.fcs', 'FLB-A': 'FLB1.fcs', 'FLC-A': 'FLC1.fcs'}),
    # 'pe' inside 'Experiment' is not a match.
    (['FLA-A', 'FLB-A', 'FLB-Cy7-A', 'FLC-A'],
     ['Experiment_01_FLA_001.fcs', 'Experiment_01_FLB_002.fcs',
      'Experiment_01_FLB-Cy7_003.fcs', 'Experiment_01_FLC_004.fcs'],
     {'FLA-A': 'Experiment_01_FLA_001.fcs', 'FLB-A': 'Experiment_01_FLB_002.fcs',
      'FLB-Cy7-A': 'Experiment_01_FLB-Cy7_003.fcs', 'FLC-A': 'Experiment_01_FLC_004.fcs'}),
    (['FLA-A', 'FLB-A', 'FLB-Cy7-A', 'FLC-A', 'FLC-Cy7-A', 'FLE-A'],
     ['CD3 FLA.fcs', 'CD4 FLB.fcs', 'CD56 FLB-Cy7.fcs', 'CD19 FLC.fcs',
      'CD8 FLC-Cy7.fcs', 'Live-Dead FLE.fcs', 'unstained.fcs'],
     {'FLA-A': 'CD3 FLA.fcs', 'FLB-A': 'CD4 FLB.fcs', 'FLB-Cy7-A': 'CD56 FLB-Cy7.fcs',
      'FLC-A': 'CD19 FLC.fcs', 'FLC-Cy7-A': 'CD8 FLC-Cy7.fcs',
      'FLE-A': 'Live-Dead FLE.fcs'}),
    # No FLB-Cy7 control: FLB-Cy7-A must not take the FLC-Cy7 one.
    (['FLB-Cy7-A', 'FLC-Cy7-A'], ['FLC-Cy7.fcs'],
     {'FLB-Cy7-A': '', 'FLC-Cy7-A': 'FLC-Cy7.fcs'}),
    # A file names one dye completely and another only in part: the complete
    # one takes it (both were left empty as 'ambiguous').
    (['FLC-A', 'FLC-Cy7-A'], ['FLC.fcs'], {'FLC-A': 'FLC.fcs', 'FLC-Cy7-A': ''}),
    (['FLB-A', 'FLB-Cy5-A', 'FLB-Cy7-A', 'FLB-CF594-A'], ['FLB.fcs', 'FLA.fcs'],
     {'FLB-A': 'FLB.fcs', 'FLB-Cy5-A': '', 'FLB-Cy7-A': '', 'FLB-CF594-A': ''}),
    # One fluor on two detectors (laser in brackets) shares its control.
    (['FLB (YG)-A', 'FLB (B)-A', 'FLA-A'], ['FLB.fcs', 'FLA.fcs'],
     {'FLB (YG)-A': 'FLB.fcs', 'FLB (B)-A': 'FLB.fcs', 'FLA-A': 'FLA.fcs'}),
    # The dye in brackets after its detector: stripping the bracket as a
    # laser left only 'B530' and 'YG582', and neither channel matched.
    (['B530 (FLA)-A', 'B530 (FLA)-H', 'YG582 (FLB)-A'], ['FLA.fcs', 'FLB.fcs'],
     {'B530 (FLA)-A': 'FLA.fcs', 'B530 (FLA)-H': 'FLA.fcs',
      'YG582 (FLB)-A': 'FLB.fcs'}),
    # A marker in brackets after the dye is not the dye.
    (['FLA (CD3)-A', 'FLB (CD4)-A'], ['FLA.fcs', 'FLB.fcs', 'FMO CD3.fcs'],
     {'FLA (CD3)-A': 'FLA.fcs', 'FLB (CD4)-A': 'FLB.fcs'}),
    # Whole words only: 'pe' inside 'Experiment' is no FLB control, so with no
    # FLB file FLB-A stays empty (matching any substring gave it the unstained
    # tube).
    (['FLA-A', 'FLB-A'],
     ['Experiment_01_FLA.fcs', 'Experiment_01_Unstained.fcs'],
     {'FLA-A': 'Experiment_01_FLA.fcs', 'FLB-A': ''}),
])
def test_autodetect_layouts(tmp_path, chans, files, want):
    root, ed = _editor_or_skip()
    try:
        for f in files:
            (tmp_path / f).write_bytes(b'')
        got, status = _autodetect(root, tmp_path, chans)
        assert got == want
        n = sum(1 for v in want.values() if v)
        assert status.startswith(f'Matched {n}/{len(chans)}'), status
    finally:
        root.destroy()


def test_autodetect_leaves_a_file_two_dyes_fit_equally_unassigned(tmp_path):
    root, ed = _editor_or_skip()
    try:
        (tmp_path / 'Cy7.fcs').write_bytes(b'')
        got, status = _autodetect(root, tmp_path, ['FLB-Cy7-A', 'FLC-Cy7-A'])
        assert got == {'FLB-Cy7-A': '', 'FLC-Cy7-A': ''}
        assert ('ambiguous, left empty: Cy7.fcs fits FLB-Cy7-A and FLC-Cy7-A equally'
                in status), status
    finally:
        root.destroy()
