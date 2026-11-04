"""Spectral unmixing can use several autofluorescence signatures, each cell
taking the one that fits it best; Spectral QC draws the NxN unmixed pairs.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import numpy as np
import pytest

import tests._sample_editor as shared
import tests.test_windows_nothing_hidden as sweep
from openflo.spectral_af import (
    af_signatures,
    unmix_multi_af,
    unmixed_pairs_figure,
)

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def _af_panel(seed=1, n=4000, d=8):
    rng = np.random.default_rng(seed)
    S = np.abs(rng.normal(size=(3, d)))
    af1 = np.exp(-((np.arange(d) - 1) ** 2) / 4)
    af2 = np.exp(-((np.arange(d) - 6) ** 2) / 4)
    lab = rng.integers(0, 2, n)
    amount = rng.uniform(500, 1500, n)
    U = np.where(lab[:, None] == 0, amount[:, None] * af1,
                 amount[:, None] * af2) + rng.normal(0, 5, (n, d))
    truth = rng.uniform(0, 1000, (n, 3))
    return S, U, lab, truth


def test_multiple_af_signatures_unmix_better_than_one():
    S, U, lab, truth = _af_panel()
    sigs = af_signatures(U, k=2)
    assert len(sigs) == 2
    A, ab, idx = unmix_multi_af(truth @ S + U, S, sigs)
    one, _ab, _ix = unmix_multi_af(truth @ S + U, S, U.mean(axis=0))
    assert np.median(np.abs(A - truth)) < 10
    assert np.median(np.abs(one - truth)) > 100
    agree = np.mean(idx == lab)
    assert max(agree, 1 - agree) > 0.98
    assert np.all(ab > 0)


def test_multiple_af_leaves_bad_events_nan_and_merges_duplicates():
    S, U, _lab, truth = _af_panel()
    Y = truth @ S + U
    Y[0, 0] = np.nan
    A, ab, idx = unmix_multi_af(Y, S, af_signatures(U, k=2))
    assert np.isnan(A[0]).all() and np.isnan(ab[0]) and idx[0] == -1
    assert np.isfinite(A[1:]).all() and (idx[1:] >= 0).all()   # the rest
    Y[1:5] -= 5_000.0                       # pushed below zero
    A, _ab, _idx = unmix_multi_af(Y, S, af_signatures(U, k=2), nonneg=True)
    assert np.isnan(A[0]).all() and (A[1:] >= 0).all()
    one_kind = np.outer(np.random.default_rng(2).uniform(5e2, 2e3, 800),
                        np.exp(-((np.arange(8) - 3) ** 2) / 4))
    assert len(af_signatures(one_kind, k=3)) == 1
    with pytest.raises(ValueError):
        af_signatures(U[:5], k=3)


def test_unmixed_pairs_figure_is_n_by_n():
    _S, _U, _lab, truth = _af_panel()
    fig = unmixed_pairs_figure(truth, ['F1', 'F2', 'F3'])
    assert len(fig.axes) == 9
    # A 60-bin histogram on the diagonal, of every drawn event (4,000 capped
    # at the default 3,000).
    bars = fig.axes[0].patches
    assert len(bars) == 60
    assert sum(b.get_height() for b in bars) == 3000
    assert len(fig.axes[1].collections) == 1         # a scatter off it


def _wait_status(root, ed, prefix, timeout=30.0):
    from tests._tk_wait import pump_until
    assert pump_until(root, lambda: ed.status_var.get().startswith(prefix),
                      timeout), ed.status_var.get()


def test_unmix_with_two_af_signatures_and_the_pairs_window(editor):
    import pandas as pd
    root, ed = editor
    det = ['FL1-A', 'FL2-A', 'FL3-A', 'FL4-A']
    rng = np.random.default_rng(3)
    spec = {'F1': np.array([1.0, 0.3, 0.05, 0.0]),
            'F2': np.array([0.0, 0.1, 1.0, 0.4])}
    af = [np.array([0.2, 1.0, 0.1, 0.0]), np.array([0.0, 0.1, 0.3, 1.0])]
    n = 400

    def cells(which, amount):
        kind = rng.integers(0, 2, n)
        a = rng.uniform(200, 400, n)
        out = np.where(kind[:, None] == 0, a[:, None] * af[0],
                       a[:, None] * af[1])
        for f in which:
            out = out + np.outer(amount[f], spec[f])
        return out + rng.normal(0, 2, (n, 4)), kind

    def put(name, arr):
        df = pd.DataFrame({c: np.zeros(n) for c in sweep.COLS})
        for i, c in enumerate(det):
            df[c] = arr[:, i]
        ed._samples[name].data = df

    hi = {'F1': rng.uniform(2e3, 3e3, n), 'F2': rng.uniform(2e3, 3e3, n)}
    put('alpha_1', cells({'F1': None}, hi)[0])
    put('alpha_2', cells({'F2': None}, hi)[0])
    put('beta_1', cells({}, hi)[0])
    import types
    ed._samples['gamma_1'] = types.SimpleNamespace(
        name='gamma_1', path='gamma_1.fcs', data=None,
        fluor_channels=list(det), channel_labels={c: c for c in sweep.COLS})
    ed._sample_order.append('gamma_1')
    ed._sample_trial['gamma_1'] = 'Day1'
    truth = {'F1': rng.uniform(0, 1000, n), 'F2': rng.uniform(0, 1000, n)}
    mixed, kind = cells({'F1': None, 'F2': None}, truth)
    put('gamma_1', mixed)
    ed.run_async = lambda work, on_done, on_error, busy_msg: on_done(work())
    ed._apply_spectral_unmix({'alpha_1': 'F1', 'alpha_2': 'F2'}, 'beta_1',
                             det, False, af_signatures=2)
    _wait_status(root, ed, 'Unmixed')
    from openflo.pipeline import linear_values
    g = ed._samples['gamma_1']
    idx = g.data['U:AF index'].to_numpy()
    assert set(np.unique(idx)) == {1.0, 2.0}
    agree = np.mean((idx == 1) == (kind == 0))
    assert max(agree, 1 - agree) > 0.95
    err = np.median(np.abs(linear_values(g, 'U:F1') - truth['F1']))
    assert err < 30, err
    name, A, _fluors = ed._last_unmix_pairs
    assert name == 'gamma_1' and A.shape == (n, 3)
    from openflo.ui_unmixed_pairs import UnmixedPairsWindow
    before = set(sweep._toplevels(ed))
    ed._open_unmixed_pairs()
    root.update()
    new = [w for k, w in sweep._toplevels(ed).items() if k not in before]
    pairs = [w for w in new if isinstance(w, UnmixedPairsWindow)]
    assert pairs and len(pairs[0].fig.axes) == 9
    for w in new:
        w.destroy()


def test_unmix_dialog_passes_the_af_signature_count(editor):
    from openflo.ui_spectral_unmix import SpectralUnmixDialog
    root, ed = editor
    got = []
    dlg = SpectralUnmixDialog(ed, ['alpha_1', 'alpha_2'],
                              ['FL1-A', 'FL2-A', 'FL3-A'],
                              lambda *a, **k: got.append(k))
    root.update()
    assert dlg.af_var.get() == 1
    _nm, rv, fv = dlg.rows[0]
    rv.set('Single-stain')
    fv.set('F1')
    dlg.af_var.set(3)
    dlg._apply()
    assert got == [{'af_signatures': 3}]
