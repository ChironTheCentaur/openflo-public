"""Spectral unmixing on what the detectors measured, and its spread matrix.

* The editor's Unmix read `linear_values`, which undoes the loader's logicle
  but not the $SPILL compensation it applied. Where compensation pushes a
  control's spectrum below zero, build_reference_spectra clips it: with a
  SPILL of 0.5 from D1 into D6 the median abundance error was 36-54% against
  7-9% on the measured values (noisy synthetic panel).
* The CLI's --unmix ran run_qc and then unmixed `sample.raw`, which run_qc
  did not filter: QC had no effect on anything it unmixed.
* The Spillover Spread Matrix took the median over quantile bins of
  SD(A_i)/sqrt(median A_j) with no negative-population baseline subtracted,
  so it grew with the control's negative fraction (1.0x, 1.0x, 4.3x, 5.9x the
  analytic spread at 0, 50, 80, 90% negative), and with nonneg=True (the GUI
  default) it was measured on clipped abundances, 0.58x.
* One inf event in the unstained control made every reference spectrum
  non-finite and unmix refused them.

Ground truth: spectra and abundances planted by the test; for detector noise
of variance = signal, the spread of stain j into fluor i is
sqrt(sum_d P[d, i]^2 S[j, d]), P = pinv(S).
"""
from __future__ import annotations

import os

import flowio
import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp
from openflo.spectral import (
    build_reference_spectra,
    measured_signal,
    spillover_spread_matrix,
    unmix,
)
from tests.conftest import gui_unavailable

# ── spillover spread matrix ─────────────────────────────────────────────────

D = 10
_X = np.arange(D)


def _gauss(mu, sd):
    return np.exp(-0.5 * ((_X - mu) / sd) ** 2)


S_TRUE = np.array([_gauss(2, 1.2), _gauss(4.5, 1.5), _gauss(7, 1.3)])
S_TRUE /= S_TRUE.max(axis=1, keepdims=True)
AF = 0.5 * _gauss(3, 3) + 20
FL = ['F1', 'F2', 'F3']
N = 10_000


def _events(rng, A):
    sig = A @ S_TRUE + AF[None, :]
    return (sig + rng.normal(0, 1, sig.shape) * np.sqrt(np.clip(sig, 0, None))
            + rng.normal(0, 3, sig.shape))


def _controls(negfrac, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for i, f in enumerate(FL):
        A = np.zeros((N, 3))
        k = int(N * (1 - negfrac))
        A[:k, i] = rng.lognormal(np.log(3000), 0.4, k)
        out[f] = _events(rng, A)
    return out


@pytest.fixture(scope='module')
def panel():
    rng = np.random.default_rng(42)
    un = _events(rng, np.zeros((N, 3)))
    S, fluors = build_reference_spectra(_controls(0.5, 1), unstained=un)
    P = np.linalg.pinv(S)
    truth = np.array([[0.0 if i == j else np.sqrt(np.sum(P[:, i] ** 2 * S_TRUE[j]))
                       for j in range(3)] for i in range(3)])
    return S, fluors, un, truth


_OFF = ~np.eye(3, dtype=bool)


@pytest.mark.parametrize('negfrac', [0.0, 0.5, 0.8, 0.9])
def test_ssm_is_the_analytic_spread_whatever_the_negative_fraction(
        panel, negfrac):
    """Before: 1.0x, 1.0x, 4.3x and 5.9x the analytic value."""
    S, fluors, un, truth = panel
    ssm, _ = spillover_spread_matrix(_controls(negfrac, 7), S, fluors,
                                     unstained=un)
    ratio = ssm[:3, :3][_OFF] / truth[_OFF]
    assert np.all(np.abs(ratio - 1.0) < 0.12), ratio


def test_ssm_ignores_nonneg(panel):
    """Clipped abundances lose half the spread: 0.58x before."""
    S, fluors, un, _truth = panel
    st = _controls(0.5, 7)
    a, _ = spillover_spread_matrix(st, S, fluors, unstained=un)
    b, _ = spillover_spread_matrix(st, S, fluors, nonneg=True, unstained=un)
    np.testing.assert_array_equal(a, b)


def test_ssm_without_any_negative_fits_the_spread_growth(panel):
    """No negative events and no unstained control: sigma^2 = a + SS^2 F over
    quantile bins of the primary."""
    S, fluors, _un, truth = panel
    ssm, _ = spillover_spread_matrix(_controls(0.0, 9), S, fluors)
    ratio = ssm[:3, :3][_OFF] / truth[_OFF]
    assert np.all(np.abs(ratio - 1.0) < 0.15), ratio


def test_ssm_diagonal_is_zero_and_unmeasured_columns_nan(panel):
    S, fluors, un, _truth = panel
    st = _controls(0.5, 7)
    del st['F3']
    ssm, _ = spillover_spread_matrix(st, S, fluors, unstained=un)
    assert ssm[0, 0] == 0.0 and ssm[1, 1] == 0.0
    assert np.isnan(ssm[:, 2]).all()


def test_an_inf_event_in_the_unstained_control_is_dropped(panel):
    S, _fluors, un, _truth = panel
    bad = un.copy()
    bad[5, 2] = np.inf
    bad[9, 0] = np.nan
    S_bad, _ = build_reference_spectra(_controls(0.5, 1), unstained=bad)
    assert np.isfinite(S_bad).all()
    np.testing.assert_allclose(S_bad, S, atol=1e-4)
    A = unmix(_events(np.random.default_rng(0),
                      np.full((50, 3), 1000.0)), S_bad)
    assert np.isfinite(A).all()


# ── measured signal ─────────────────────────────────────────────────────────

DETS = [f'D{i + 1}-A' for i in range(8)]
SPECTRA = np.array([
    [1.0, 0.7, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.1, 0.4, 1.0, 0.8, 0.3, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.1, 0.3, 0.7, 1.0, 0.6],
])
SPILL = np.eye(8)
SPILL[0, 5] = 0.5                      # D1 -> D6
_SPILL_TEXT = ('8,' + ','.join(DETS) + ','
               + ','.join(repr(float(v)) for v in SPILL.ravel()))


def _fcs(path, det_values, spill=True, extra=None):
    n = len(det_values)
    rng = np.random.default_rng(1)
    cols = ['FSC-A', 'SSC-A'] + DETS
    ev = np.column_stack([rng.uniform(5e4, 1e5, n), rng.uniform(1e4, 5e4, n),
                          det_values])
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).ravel().tolist(), cols,
                          metadata_dict={'SPILL': _SPILL_TEXT} if spill else None)
    return str(path)


def test_measured_signal_undoes_transform_and_compensation(tmp_path):
    rng = np.random.default_rng(3)
    Y = rng.uniform(50, 5000, (500, 8)).astype(np.float32).astype(float)
    s = fp.FlowSample(_fcs(tmp_path / 'm.fcs', Y))
    s.auto_compensate()
    s.apply_transform()
    assert s.is_compensated() and s.data_transforms
    np.testing.assert_allclose(measured_signal(s, DETS), Y, rtol=1e-6)
    # Detector order is the caller's.
    np.testing.assert_allclose(measured_signal(s, DETS[::-1]), Y[:, ::-1],
                               rtol=1e-6)


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
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


def test_editor_unmixes_the_measured_values_of_compensated_samples(tmp_path):
    """Every file carries a SPILL the loader applies. Noise-free data, so the
    unmix of the measured values is exact up to float32; on the compensated
    values the clipped F1 spectrum put the abundances far off."""
    root, ed = _editor_or_skip()
    try:
        rng = np.random.default_rng(7)
        n, af = 1500, 10.0
        arrs = {f'F{i + 1}': rng.uniform(200, 5000, (n, 1)) * SPECTRA[i] + af
                for i in range(3)}
        arrs['unstained'] = np.full((n, 8), af)
        A = rng.uniform(0, 5000, (n, 3))
        arrs['mixed'] = A @ SPECTRA + af
        for nm, Y in arrs.items():
            ed._load_worker(nm, _fcs(tmp_path / f'{nm}.fcs', Y))
        for _ in range(50):
            root.update()
            if len(ed._samples) == 5:
                break
        assert len(ed._samples) == 5
        assert ed._samples['mixed'].is_compensated()
        ed._apply_spectral_unmix({'F1': 'F1', 'F2': 'F2', 'F3': 'F3'},
                                 'unstained', list(DETS), False)
        s = ed._samples['mixed']
        got = np.column_stack([fp.linear_values(s, f'U:F{i + 1}')
                               for i in range(3)])
        np.testing.assert_allclose(got, A, atol=0.05)
    finally:
        root.destroy()


# ── QC and the rows it keeps ────────────────────────────────────────────────

def _piled(rng, Y, frac=0.05):
    """`frac` of the events piled at the ceiling of D1 (saturation)."""
    Y = Y.copy()
    k = int(len(Y) * frac)
    Y[:k, 0] = 262143.0
    return Y, k


def test_run_qc_keeps_raw_in_step_with_data(tmp_path):
    rng = np.random.default_rng(5)
    Y, k = _piled(rng, rng.uniform(50, 5000, (2000, 8)))
    s = fp.FlowSample(_fcs(tmp_path / 'q.fcs', Y, spill=False))
    s.run_qc()
    assert len(s.data) == len(s.raw) == 2000 - k
    np.testing.assert_array_equal(s.raw.to_numpy(), s.data.to_numpy())


def test_cli_unmix_writes_only_the_events_qc_kept(tmp_path):
    """Before: the unmixed CSV held every event, the QC-removed ones too."""
    import json
    import types

    from openflo.cli import run_batch_unmix
    rng = np.random.default_rng(11)
    ctrl = {}
    for i in range(3):
        Y = (rng.uniform(50, 800, (3000, 1)) * SPECTRA[i]
             + rng.normal(0, 1.0, (3000, 8)) + 5.0)
        ctrl[f'F{i + 1}'] = _fcs(tmp_path / f'ss{i}.fcs', Y, spill=False)
    in_dir = tmp_path / 'in'
    in_dir.mkdir()
    A = rng.uniform(0, 500, (3000, 3))
    Y, k = _piled(rng, A @ SPECTRA + rng.normal(0, 1.0, (3000, 8)) + 5.0)
    _fcs(in_dir / 'mixed.fcs', Y, spill=False)
    out = tmp_path / 'out'
    args = types.SimpleNamespace(
        unmix=True, out=str(out), unmix_controls=json.dumps(ctrl),
        unmix_input=str(in_dir), unmix_detectors=','.join(DETS),
        unmix_nonneg=False, fcs='', trials='')
    assert run_batch_unmix(args) == 0
    df = pd.read_csv(out / 'mixed_unmixed.csv')
    assert len(df) == 3000 - k
    r = np.corrcoef(df['U:F2'].to_numpy(), A[k:, 1])[0, 1]
    assert r > 0.99
    qc = json.loads((out / 'spectral_qc.json').read_text(encoding='utf-8'))
    assert 'into fluors[i]' in qc['ssm_orientation']
