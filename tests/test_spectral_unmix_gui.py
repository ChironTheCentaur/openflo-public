"""Spectral unmixing — the GUI action path (Tools -> Spectral unmixing).

spectral.py is tested directly (tests/test_spectral.py); the dialog and the
editor action that wires it to loaded samples (ToolsMixin._apply_spectral_unmix)
had ~1% coverage. Ground truth: noise-free single stains over 8 detectors with
known spectra, a constant autofluorescence of 10 in every detector, and a
mixed sample Y = A_true @ S + 10, so the exact linear unmix returns A_true.

Tk-guarded like tests/test_gui_ux.py.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tests.conftest import gui_unavailable

SPECTRA = np.array([
    [1.0, 0.7, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.1, 0.4, 1.0, 0.8, 0.3, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.1, 0.3, 0.7, 1.0, 0.6],
])
FLUORS = ['F1', 'F2', 'F3']
DETS = [f'D{i + 1}-A' for i in range(8)]
AF = 10.0


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


def _arrays(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    arrs = {}
    for i, f in enumerate(FLUORS):
        amp = rng.uniform(200, 5000, (n, 1))
        arrs[f] = amp * SPECTRA[i][None, :] + AF
    arrs['unstained'] = np.full((n, len(DETS)), AF)
    A = rng.uniform(0, 5000, (n, len(FLUORS)))
    arrs['mixed'] = A @ SPECTRA + AF
    return arrs, A


def _register(ed, name, df, trial='T'):
    s = SimpleNamespace(name=name, path=None, data=df,
                        fluor_channels=list(DETS),
                        channel_labels={c: c for c in df.columns})
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = trial
    if trial not in ed._trial_order:
        ed._trial_order.append(trial)
    ed._sample_plot_enabled[name] = True
    if len(ed._samples) == 1:
        ed._channels = list(df.columns)
        ed._channel_labels = {c: c for c in df.columns}
        ed._populate_channel_combos()
    return s


def test_unmix_dialog_passes_roles_detectors_and_nonneg():
    """Build & Apply hands on_apply {sample: fluor} (blank fluor -> sample
    name), the unstained sample, the parsed detector list and nonneg."""
    root, ed = _editor_or_skip()
    try:
        from openflo.ui_spectral_unmix import SpectralUnmixDialog
        got = []
        dlg = SpectralUnmixDialog(ed, ['ssA', 'ssB', 'un', 'mix'],
                                  ['D1-A', 'D2-A', 'D3-A'],
                                  lambda *a: got.append(a))
        roles = {'ssA': 'Single-stain', 'ssB': 'Single-stain',
                 'un': 'Unstained', 'mix': 'Ignore'}
        for nm, rv, _fv in dlg.rows:
            rv.set(roles[nm])
        dlg.rows[0][2].set('  FL4 ')          # ssA -> 'FL4'; ssB left blank
        dlg.det_txt.delete('1.0', 'end')
        dlg.det_txt.insert('1.0', 'D1-A, D2-A,\nD3-A ,')
        dlg.nonneg_var.set(False)
        dlg._apply()
        assert got == [({'ssA': 'FL4', 'ssB': 'ssB'}, 'un',
                        ['D1-A', 'D2-A', 'D3-A'], False)]
    finally:
        root.destroy()


def test_unmix_dialog_refuses_without_single_stain():
    root, ed = _editor_or_skip()
    try:
        from openflo import ui_spectral_unmix as mod
        warned, got = [], []
        saved = mod.messagebox.showwarning
        mod.messagebox.showwarning = lambda *a, **k: warned.append(a)
        try:
            dlg = mod.SpectralUnmixDialog(ed, ['a', 'b'], ['D1-A', 'D2-A'],
                                          lambda *a: got.append(a))
            dlg._apply()                       # every role 'Ignore'
        finally:
            mod.messagebox.showwarning = saved
        assert got == [] and len(warned) == 1
        dlg.destroy()
    finally:
        root.destroy()


def test_apply_spectral_unmix_recovers_known_abundances_on_linear_data():
    """On linear detector data the action writes U:<fluor> columns equal to
    the true abundances, unmixes only same-group non-control samples, and
    keeps the QC report for the Spectral QC window."""
    root, ed = _editor_or_skip()
    try:
        arrs, A = _arrays()
        for nm in ('F1', 'F2', 'F3', 'unstained', 'mixed'):
            _register(ed, nm, pd.DataFrame(arrs[nm], columns=pd.Index(DETS)))
        _register(ed, 'other_group',
                  pd.DataFrame(arrs['mixed'].copy(), columns=pd.Index(DETS)),
                  trial='T2')
        ed._apply_spectral_unmix({'F1': 'F1', 'F2': 'F2', 'F3': 'F3'},
                                 'unstained', list(DETS), False)
        mixed = ed._samples['mixed'].data
        got = mixed[[f'U:{f}' for f in FLUORS]].to_numpy(float)
        np.testing.assert_allclose(got, A, atol=1e-6)
        np.testing.assert_allclose(mixed['U:Autofluorescence'], AF, atol=1e-6)
        # controls are not unmixed; another group's sample is left alone
        for nm in ('F1', 'F2', 'F3', 'unstained', 'other_group'):
            assert not any(c.startswith('U:')
                           for c in ed._samples[nm].data.columns), nm
        assert 'Unmixed 1 sample(s)' in ed.status_var.get()
        assert '1 sample(s) in other groups' in ed.status_var.get()
        qc = ed._last_unmix_qc
        assert qc is not None and qc['fluors'] == FLUORS + ['Autofluorescence']
        assert np.isfinite(qc['condition_number'])
    finally:
        root.destroy()


def test_gui_unmix_on_loader_processed_samples_preserves_abundance_order(
        tmp_path):
    """Samples loaded the way the GUI loads FCS files (the loader
    logicle-transforms every detector) must unmix to the true abundances:
    linear least squares has to run on the linear detector values."""
    from scipy.stats import spearmanr

    from openflo.fcs_export import write_fcs
    root, ed = _editor_or_skip()
    try:
        arrs, A = _arrays()
        for nm in ('F1', 'F2', 'F3', 'unstained', 'mixed'):
            p = tmp_path / f'{nm}.fcs'
            write_fcs(pd.DataFrame(arrs[nm], columns=pd.Index(DETS)), str(p))
            ed._load_worker(nm, str(p))        # the real loader
        for _ in range(50):                    # flush its after(0) callbacks
            root.update()
            if len(ed._samples) == 5:
                break
        assert len(ed._samples) == 5
        ed._apply_spectral_unmix({'F1': 'F1', 'F2': 'F2', 'F3': 'F3'},
                                 'unstained', list(DETS), False)
        from openflo.pipeline import linear_values, transform_spec
        s = ed._samples['mixed']
        mixed = s.data
        for j, f in enumerate(FLUORS):
            rho = spearmanr(mixed[f'U:{f}'].to_numpy(float), A[:, j])[0]
            assert rho > 0.99, (f, rho)
        # U: columns are abundances on the detectors' linear scale, so they
        # take the same default display as the detectors -- logicle, recorded
        # -- and their linear values are the true abundances; the residual is
        # the float32 rounding of the written FCS, not the transform.
        for f in FLUORS:
            assert s.data_transforms[f'U:{f}'] == transform_spec('logicle'), f
            assert ed._channel_transform[f'U:{f}'] == 'logicle', f
        got = np.column_stack([linear_values(s, f'U:{f}') for f in FLUORS])
        np.testing.assert_allclose(got, A, atol=0.01)     # measured 6e-4
    finally:
        root.destroy()


def test_gui_unmix_says_which_spectra_were_matched_and_which_are_too_dim():
    """A dim dye on cells whose autofluorescence varies (30% at 10x) gets its
    spectrum from matched AF, and a control too faint to define one is
    flagged -- in the status bar, the audit entry and the QC report -- rather
    than plotted like any other (tests/test_spectral_matched_af.py)."""
    root, ed = _editor_or_skip()
    try:
        rng = np.random.default_rng(11)
        x = np.arange(len(DETS))
        af = np.exp(-0.5 * ((x - 1.5) / 2.0) ** 2) * 40 + 5
        n = 3000

        def cells(amp, spec):
            k = np.where(rng.random(n) < 0.3, 10.0, 1.0)[:, None]
            sig = k * af + amp[:, None] * spec
            return sig + rng.normal(0, 1, sig.shape) * np.sqrt(
                np.clip(sig, 1, None))

        def pos(level):
            return np.where(rng.random(n) < 0.5,
                            rng.lognormal(np.log(level), 0.3, n), 0.0)

        data = {'dim': cells(pos(400), SPECTRA[2]),
                'faint': cells(pos(3), SPECTRA[1]),
                'unstained': cells(np.zeros(n), SPECTRA[0]),
                'mixed': cells(pos(400), SPECTRA[2])}
        for nm, arr in data.items():
            _register(ed, nm, pd.DataFrame(arr, columns=pd.Index(DETS)))
        audits = []
        ed._audit = lambda ev, **k: audits.append((ev, k))
        ed._apply_spectral_unmix({'dim': 'Dim', 'faint': 'Faint'},
                                 'unstained', list(DETS), False)
        status = ed.status_var.get()
        assert 'Dim: spectrum estimated against matched autofluorescence' \
            in status
        assert '[!] Too dim for a reliable spectrum: Faint' in status
        ev = dict(audits)['unmix']
        assert ev['spectra_methods']['Dim'] == 'matched'
        assert ev['dim_controls'] == ['Faint']
        rep = {d['fluor']: d for d in ed._last_unmix_qc['reference_spectra']}
        assert rep['Dim']['primary_detector'] in DETS
        assert rep['Faint']['warning']
    finally:
        root.destroy()
