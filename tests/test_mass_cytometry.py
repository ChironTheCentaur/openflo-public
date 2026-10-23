"""Mass cytometry basics: bead normalization over a run (Finck et al. 2013,
bead_normalize) and clean-up on the Gaussian discrimination parameters
(gaussian_cleanup), with their window, clean-up gate and session record
(ui_mass_cytometry).

Ground truth is built into generated CyTOF-like data: the instrument's
sensitivity falls linearly to 60 % over the run, every channel alike; 3 %
of the events are beads carrying five metals at known levels; doublets and
debris are planted at known indices on the Gaussian parameters. The
expected values are worked out in the tests from that truth, not read back
from the code under test. Channel names are synthetic (Mass101, Mass140).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from openflo import bead_normalize as bn
from openflo import gaussian_cleanup as gc

BEADS = ['Mass140', 'Mass151', 'Mass153', 'Mass165', 'Mass175']
BEAD_LEVEL = np.array([800.0, 2500.0, 2600.0, 3000.0, 2000.0])
GAUSS = ['Event_length', 'Center', 'Offset', 'Width', 'Residual']
T_END = 1000.0


def _run(seed=0, n=40000, drift=0.4, scale=1.0, bead_frac=0.03):
    """A run whose sensitivity is ``scale * (1 - drift * t / T_END)``.

    Returns ``(frame of LINEAR values, truth)``; truth holds the
    sensitivity per event, the bead indices, the undrifted Mass101 and the
    doublet / debris indices."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, T_END, n))
    d = scale * (1.0 - drift * t / T_END)
    is_bead = rng.random(n) < bead_frac
    cols = {}
    for j, c in enumerate(BEADS):
        cell = rng.gamma(0.3, 3.0, n)
        if c == 'Mass165':
            # A marker on a bead channel: a bright cell population, high on
            # this one channel only, which must not be taken for beads.
            cell = np.where(rng.random(n) < 0.3, rng.lognormal(7.5, 0.3, n),
                            cell)
        bead = BEAD_LEVEL[j] * rng.lognormal(0.0, 0.15, n)
        cols[c] = np.where(is_bead, bead, cell) * d
    true101 = rng.lognormal(5.0, 1.0, n)
    cols['Mass101'] = true101 * d
    cols['Mass102'] = rng.lognormal(3.0, 0.7, n) * d
    cols['Time'] = t
    # Gaussian parameters: tight (log-normal, 8 %) for single cells;
    # doublets are long with a large residual, debris short and dim.
    gauss = {'Event_length': 25.0, 'Center': 600.0, 'Offset': 60.0,
             'Width': 40.0, 'Residual': 90.0}
    for c, v in gauss.items():
        cols[c] = v * rng.lognormal(0.0, 0.08, n)
    cand = np.flatnonzero(~is_bead)
    pick = rng.choice(cand, 400, replace=False)
    doublets, debris = np.sort(pick[:200]), np.sort(pick[200:])
    cols['Event_length'][doublets] *= 2.0
    cols['Residual'][doublets] *= 3.0
    cols['Center'][debris] *= 0.25
    cols['Width'][debris] *= 0.5
    df = pd.DataFrame(cols)
    return df, {'d': d, 'beads': np.flatnonzero(is_bead), 'true101': true101,
                'doublets': doublets, 'debris': debris}


# ── Bead normalization: the pieces, on hand-worked numbers ─────────────────

def test_least_squares_factor_on_a_hand_worked_example():
    # f = sum(b*s) / sum(s^2): baseline (2, 4) against (1, 2) is exactly 2;
    # against (1, 1): (2 + 4) / (1 + 1) = 3; against (2, 1): (4 + 4) / 5.
    f = bn.ls_factors([2.0, 4.0], np.array([[1.0, 2.0], [1.0, 1.0],
                                            [2.0, 1.0], [0.0, 0.0]]))
    np.testing.assert_allclose(f[:3], [2.0, 3.0, 1.6])
    assert np.isnan(f[3])


def test_running_median_is_centred_and_shrinks_at_the_ends():
    # Window 3 over (1, 5, 2, 8, 3): ends see two points.
    got = bn.running_median(np.array([1.0, 5.0, 2.0, 8.0, 3.0]), 3)
    np.testing.assert_allclose(got, [3.0, 2.0, 5.0, 3.0, 5.5])


def test_knots_average_tied_times_and_keep_both_ends():
    t, f = bn._thin(np.array([1.0, 1.0, 2.0, np.nan]),
                    np.array([1.0, 3.0, 5.0, 7.0]), 10)
    np.testing.assert_allclose(t, [1.0, 2.0])
    np.testing.assert_allclose(f, [2.0, 5.0])
    t, f = bn._thin(np.arange(100.0), np.arange(100.0) * 2, 5)
    np.testing.assert_allclose(t, [0, 25, 50, 74, 99])
    # Between knots linear, outside them the end value, no Time: 1.
    np.testing.assert_allclose(
        bn.factors_at(np.array([-5.0, 0.5, 9.0, np.nan]), [0.0, 1.0],
                      [2.0, 4.0]), [2.0, 3.0, 4.0, 1.0])


def test_otsu_split_lands_in_the_middle_of_the_gap():
    v = np.r_[np.linspace(0.0, 1.0, 970), np.linspace(5.0, 6.0, 30)]
    t = bn.otsu_split(v)
    assert 2.0 < t < 4.0
    assert np.isnan(bn.otsu_split(np.ones(10)))


def test_beads_are_the_events_high_on_every_bead_channel():
    df, truth = _run(1)
    mask, thr = bn.identify_beads(df, BEADS)
    # Every planted bead and nothing else: the Mass165-bright cells are
    # high on one bead channel only.
    np.testing.assert_array_equal(np.flatnonzero(mask), truth['beads'])
    assert set(thr) == set(BEADS)
    # A threshold typed for one channel is used as given.
    m2, thr2 = bn.identify_beads(df, BEADS, {'Mass140': 50.0})
    assert thr2['Mass140'] == 50.0 and not m2.any()


def test_normalization_recovers_a_linear_decay_to_60_percent():
    # About 3,600 beads: a 500-bead window spans 14 % of the run.
    df, truth = _run(2, n=120000)
    out = bn.normalize_runs({'A': df}, BEADS)['A']
    d = truth['d']
    # Baseline = the run's mean bead level, so the true factor at time t is
    # mean(sensitivity at the beads) / sensitivity(t).
    want = d[truth['beads']].mean() / d
    f = out['factors']
    inner = (df['Time'] > 0.1 * T_END) & (df['Time'] < 0.9 * T_END)
    assert np.max(np.abs(f[inner] / want[inner] - 1)) < 0.015
    # At the ends the running median sees one side only: a little bias.
    assert np.max(np.abs(f / want - 1)) < 0.03
    # The cells: Mass101 / its undrifted value no longer falls over time.
    ratio = out['frame']['Mass101'].to_numpy() / truth['true101']
    early = np.median(ratio[df['Time'] < 0.2 * T_END])
    late = np.median(ratio[df['Time'] > 0.8 * T_END])
    assert late / early == pytest.approx(1.0, abs=0.02)
    before = (np.median(df['Mass101'][df['Time'] > 0.8 * T_END]
                        / truth['true101'][df['Time'] > 0.8 * T_END])
              / np.median(df['Mass101'][df['Time'] < 0.2 * T_END]
                          / truth['true101'][df['Time'] < 0.2 * T_END]))
    assert before < 0.7                  # it did fall before
    # The beads are flat afterwards: their running median spans < 3 %
    # where it spanned ~ 50 %.
    fit = out['fit']
    g = bn.factors_at(fit['times'], out['record']['knots_t'],
                      out['record']['knots_f'])
    _t, after = bn.smooth_beads(fit['times'], fit['raw'] * g[:, None], 500)

    def decile_levels(sm):
        # Each tenth of the beads' mean running median, each channel
        # relative to its own run mean, averaged over the channels.
        rel = sm / sm.mean(axis=0)
        return np.array([part.mean() for part in np.array_split(rel, 10)])
    assert np.ptp(decile_levels(fit['smoothed'])) > 0.3
    assert np.ptp(decile_levels(after)) < 0.02


def test_time_and_gaussian_parameters_are_never_scaled():
    df, _truth = _run(3)
    keep = df.copy()
    out = bn.normalize_runs({'A': df}, BEADS)['A']
    for c in ['Time', *GAUSS]:
        np.testing.assert_array_equal(out['frame'][c].to_numpy(),
                                      df[c].to_numpy())
        assert c not in out['record']['channels']
    assert set(out['record']['channels']) == {'Mass101', 'Mass102', *BEADS}
    assert bn.mass_channels(df.columns) == [c for c in df.columns
                                            if c not in ('Time', *GAUSS)]
    pd.testing.assert_frame_equal(df, keep)        # input not changed
    # Named outright, Time and the Gaussian parameters are still refused.
    every = bn.normalize_runs({'A': df}, BEADS, channels=list(df.columns))['A']
    for c in ['Time', *GAUSS]:
        np.testing.assert_array_equal(every['frame'][c].to_numpy(),
                                      df[c].to_numpy())


def test_several_files_share_one_baseline_or_a_reference():
    # A: no drift, full sensitivity. B: no drift, 70 % sensitivity.
    a, ta = _run(4, drift=0.0)
    b, tb = _run(5, drift=0.0, scale=0.7)
    frames = {'A': a, 'B': b}

    def level(res, name, truth):
        return float(np.median(res[name]['frame']['Mass101'].to_numpy()
                               / truth['true101']))
    each = bn.normalize_runs(frames, BEADS, baseline='each')
    assert level(each, 'B', tb) / level(each, 'A', ta) == pytest.approx(
        0.7, rel=0.02)
    both = bn.normalize_runs(frames, BEADS, baseline='all')
    assert level(both, 'B', tb) / level(both, 'A', ta) == pytest.approx(
        1.0, abs=0.02)
    # The shared level is the mean of the two runs' bead levels: 0.85.
    assert level(both, 'A', ta) == pytest.approx(0.85, rel=0.02)
    ref = bn.normalize_runs(frames, BEADS, baseline='A')
    assert level(ref, 'A', ta) == pytest.approx(1.0, abs=0.02)
    assert level(ref, 'B', tb) == pytest.approx(1.0, abs=0.02)
    with pytest.raises(ValueError):
        bn.normalize_runs(frames, BEADS, baseline='nobody')


def test_a_record_undoes_itself_and_a_population_can_name_the_beads():
    df, truth = _run(6)
    out = bn.normalize_runs({'A': df}, BEADS)['A']
    back = bn.apply_record(out['frame'], out['record'], inverse=True)
    np.testing.assert_allclose(back[BEADS + ['Mass101']].to_numpy(),
                               df[BEADS + ['Mass101']].to_numpy(),
                               rtol=1e-12)
    mask = np.zeros(len(df), dtype=bool)
    mask[truth['beads']] = True
    pop = bn.normalize_runs({'A': df}, BEADS, bead_masks={'A': mask})['A']
    np.testing.assert_allclose(pop['factors'], out['factors'], rtol=1e-9)
    with pytest.raises(ValueError, match='bead event'):
        bn.normalize_runs({'A': df}, BEADS,
                          bead_masks={'A': np.zeros(len(df), dtype=bool)})


def test_bead_removal_method_drops_the_beads_on_a_display_scale():
    df, truth = _run(7)
    shown = df.copy()
    for c in BEADS:
        shown[c] = np.arcsinh(df[c] / 5.0)
    keep = bn.bead_keep_mask(shown, {'channels': ','.join(BEADS)})
    np.testing.assert_array_equal(np.flatnonzero(~keep), truth['beads'])
    # A bead channel the sample lacks: nothing removed.
    assert bn.bead_keep_mask(shown, {'channels': 'Mass140,Mass999'}).all()


# ── Gaussian-parameter clean-up ─────────────────────────────────────────────

def test_gaussian_parameters_are_found_by_their_standard_names():
    cols = ['Time', 'event_length', 'CENTER', 'Offset', 'Width',
            'Residual', 'FSC-W', 'Mass101']
    assert gc.find_gaussian_channels(cols) == ['event_length', 'CENTER',
                                               'Offset', 'Width', 'Residual']
    assert gc.parse_channels('Width, Nope', cols) == ['Width']


def test_robust_bounds_on_hand_worked_numbers():
    # 0..20: median 10; |x - 10| has median 5, so the robust SD is
    # 5 / Phi^-1(3/4) = 7.41301...; one of them either side of 10.
    sd = 5.0 / norm.ppf(0.75)
    lo, hi = gc.robust_bounds(np.arange(21.0), n_mad=1.0, scale='linear')
    assert (lo, hi) == pytest.approx((10 - sd, 10 + sd), rel=1e-12)
    # No spread: no bounds.
    assert gc.robust_bounds(np.full(50, 3.0)) == (-np.inf, np.inf)


def test_cleanup_removes_planted_doublets_and_debris():
    df, truth = _run(8)
    keep = gc.gaussian_keep_mask(df, {'n_mad': 3.0})
    planted = np.r_[truth['doublets'], truth['debris']]
    assert not keep[planted].any()
    # Independent recomputation of the rule: median +/- 3 * MAD / Phi^-1(3/4)
    # of arcsinh(x / 5), per parameter.
    expect = np.ones(len(df), dtype=bool)
    for c in GAUSS:
        z = np.arcsinh(df[c].to_numpy() / 5.0)
        med = np.median(z)
        sd = np.median(np.abs(z - med)) / norm.ppf(0.75)
        expect &= np.abs(z - med) <= 3.0 * sd
    np.testing.assert_array_equal(keep, expect)
    # Few clean events lost: 5 parameters at 3 SDs is ~1.3 %.
    clean = np.setdiff1d(np.arange(len(df)), planted)
    assert (~keep[clean]).mean() < 0.02
    # The same through an auto-clean recipe gate.
    from openflo.pipeline import autoclean_keep_mask, autoclean_methods_signature
    gate = {'kind': 'autoclean', 'methods': [gc.default_method()]}
    np.testing.assert_array_equal(autoclean_keep_mask(gate, df), keep)
    hash(autoclean_methods_signature(gate))
    # Only Event_length: the debris (short Center / Width) stay.
    only = gc.gaussian_keep_mask(df, {'channels': 'Event_length'})
    assert not only[truth['doublets']].any()
    assert only[truth['debris']].mean() > 0.9


def test_an_unmeasured_event_is_kept_and_a_bad_scale_falls_back():
    df, truth = _run(9, n=5000)
    df.loc[truth['doublets'][0], 'Event_length'] = np.nan
    keep = gc.gaussian_keep_mask(df, {'channels': 'Event_length'})
    assert keep[truth['doublets'][0]]
    assert not keep[truth['doublets'][1:]].any()
    np.testing.assert_array_equal(
        gc.gaussian_keep_mask(df, {'scale': 'Bogus'}),
        gc.gaussian_keep_mask(df, {'scale': 'asinh'}))
    assert not np.array_equal(gc.gaussian_keep_mask(df, {'scale': 'linear'}),
                              gc.gaussian_keep_mask(df))


def test_without_gaussian_parameters_nothing_is_removed():
    df = pd.DataFrame({'Mass101': np.arange(100.0), 'Time': np.arange(100.0)})
    assert gc.gaussian_keep_mask(df).all()
    from openflo.pipeline import autoclean_method_diagnostic
    assert 'no Gaussian' in autoclean_method_diagnostic('gaussian', df, {})


def test_methods_paragraph_cites_both():
    from openflo.provenance import methods_paragraph
    text = methods_paragraph([{'action': 'bead_normalize', 'details': {}},
                              {'action': 'gaussian_cleanup', 'details': {}}])
    assert 'Finck et al., 2013' in text and 'Bagwell et al., 2020' in text
    assert 'Finck et al., 2013' not in methods_paragraph([])


# ── The windows ─────────────────────────────────────────────────────────────

def _asinh_sample(name, df):
    """A loaded-sample stand-in whose mass channels hold arcsinh(x/5), as
    the editor would after a cofactor-5 transform."""
    from openflo.pipeline import transform_spec, transform_values
    spec = transform_spec('asinh', cofactor=5.0)
    data = df.copy()
    chans = ['Mass101', 'Mass102', *BEADS]
    for c in chans:
        data[c] = transform_values(df[c].to_numpy(), **spec)
    return SimpleNamespace(name=name, path=f'{name}.fcs', data=data,
                           fluor_channels=list(chans),
                           channel_labels={c: c for c in data.columns},
                           data_transforms={c: dict(spec) for c in chans},
                           metadata={})


@pytest.fixture
def ed():
    from tests.conftest import gui_unavailable
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:                              # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    editor = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                      on_save=None, primary=False)
    editor.withdraw()
    truths = {}
    for name, seed in (('SampleA', 11), ('SampleB', 12)):
        df, truth = _run(seed, n=20000)
        truths[name] = (df, truth)
        editor._samples[name] = _asinh_sample(name, df)
        editor._sample_order.append(name)
        editor._sample_trial[name] = 'Trial'
        editor._sample_gates.setdefault(name, {})
        editor._sample_gate_order.setdefault(name, [])
    editor._set_active_sample('SampleA')
    editor._truths = truths
    yield editor
    root.destroy()


def _linear(sample, c):
    from openflo.pipeline import linear_values
    return np.asarray(linear_values(sample, c), dtype=float)


def test_bead_window_normalizes_adds_the_gate_and_undoes(ed):
    from openflo.pipeline import cumulative_gate_mask
    from openflo.ui_mass_cytometry import BeadNormalizeWindow, cleanup_gate_id
    w = BeadNormalizeWindow(ed)
    w.select_bead_channels(BEADS)
    assert w.bead_channels() == BEADS
    w.base_var.set("Each file's own beads")
    # 600 beads a sample: a window of 100 follows the drift.
    w.win_var.set('100')
    raw = {n: ed._samples[n].data.copy() for n in ed._sample_order}
    w._run(sync=True)
    assert w._res and set(w._res['results']) == {'SampleA', 'SampleB'}, \
        w.status_var.get()
    for n in ('SampleA', 'SampleB'):
        s = ed._samples[n]
        df, truth = ed._truths[n]
        # The written values, back on the linear scale, are the raw ones
        # times the factor: the late cells no longer read dimmer.
        ratio = _linear(s, 'Mass101') / truth['true101']
        t = df['Time'].to_numpy()
        late = np.median(ratio[t > 0.8 * T_END])
        early = np.median(ratio[t < 0.2 * T_END])
        assert late / early == pytest.approx(1.0, abs=0.03)
        for c in ['Time', *GAUSS]:
            np.testing.assert_array_equal(s.data[c].to_numpy(),
                                          raw[n][c].to_numpy())
        assert s.bead_norm['bead_channels'] == BEADS
        # The clean-up gate removes exactly the beads.
        gid = cleanup_gate_id(ed, n)
        assert gid is not None
        keep = cumulative_gate_mask(ed._sample_gates[n], gid, s.data)
        np.testing.assert_array_equal(np.flatnonzero(~np.asarray(keep)),
                                      truth['beads'])
    w._draw()
    tr = w.traces('SampleA', 'Mass151')
    assert tr is not None and len(tr[0]) == len(tr[1]) == len(tr[2])
    assert 'Finck et al. 2013' in w._res['methods']
    assert any(e.get('action') == 'bead_normalize'
               for e in ed._audit_log.to_list())
    # A second run skips the normalized samples rather than stacking.
    w._run(sync=True)
    assert 'already bead-normalized' in w.status_var.get()
    w._undo()
    for n in ('SampleA', 'SampleB'):
        np.testing.assert_allclose(ed._samples[n].data['Mass101'].to_numpy(),
                                   raw[n]['Mass101'].to_numpy(), rtol=1e-9)
        assert not ed._samples[n].bead_norm
    w.destroy()


def test_gaussian_window_adds_one_cleanup_gate_and_updates_it(ed):
    from openflo.pipeline import cumulative_gate_mask
    from openflo.ui_mass_cytometry import GaussianCleanupWindow, cleanup_gate_id
    w = GaussianCleanupWindow(ed)
    assert w._params == GAUSS
    assert [r['sample'] for r in w._rows] == ['SampleA', 'SampleB']
    w._add()
    for n in ('SampleA', 'SampleB'):
        gid = cleanup_gate_id(ed, n)
        df, truth = ed._truths[n]
        keep = np.asarray(cumulative_gate_mask(ed._sample_gates[n], gid,
                                               ed._samples[n].data))
        np.testing.assert_array_equal(keep, gc.gaussian_keep_mask(df))
        assert not keep[truth['doublets']].any()
        assert not keep[truth['debris']].any()
    # Only Event_length, 4 SDs: the same gate, updated.
    for c, v in w.param_vars.items():
        v.set(c == 'Event_length')
    w.n_var.set('4')
    w._add()
    gates = [g for g in ed._sample_gates['SampleA'].values()
             if g.get('kind') == 'autoclean']
    assert len(gates) == 1
    m = [x for x in gates[0]['methods'] if x['key'] == 'gaussian']
    assert m and m[0]['params'] == {'channels': 'Event_length', 'n_mad': 4.0,
                                    'scale': 'asinh'}
    w.destroy()


def test_menu_offers_the_mass_cytometry_entries(ed):
    labels = {}
    for b in ed._menubar_buttons:
        if str(b.cget('text')).strip() != 'Tools':
            continue
        m = b.nametowidget(b.cget('menu'))
        for i in range(m.index('end') + 1):
            if m.type(i) == 'cascade' and m.entrycget(i, 'label') == \
                    'Mass cytometry':
                sub = m.nametowidget(m.entrycget(i, 'menu'))
                labels = {sub.entrycget(j, 'label')
                          for j in range(sub.index('end') + 1)}
    assert labels == {"Bead normalization…", "Undo bead normalization",
                      "Gaussian parameter clean-up…"}


# ── Session round trip ──────────────────────────────────────────────────────

def _fcs(path, seed):
    import flowio
    df, truth = _run(seed, n=6000, bead_frac=0.05)
    names = list(df.columns)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, df.to_numpy(np.float32).flatten().tolist(),
                          names)
    return str(path), df, truth


@pytest.mark.parametrize('sidecar', [True, False])
def test_bead_normalization_survives_a_session_round_trip(tmp_path,
                                                          monkeypatch,
                                                          sidecar):
    """Saved as its factor record; reopened from the processed sidecar (the
    values already normalized) or, the sidecar gone, from the raw FCS (the
    factors applied again)."""
    from openflo.ui_mass_cytometry import (
        apply_bead_results,
        run_bead_normalization,
    )
    from tests.test_derived_kinetics import _loaded, _restore_session
    from tests.test_transform_records import (
        _new_editor,
        capture_warnings,
        isolate_home,
    )
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    try:
        ed = _new_editor(root)
        p, _df, _truth = _fcs(tmp_path / 'run1.fcs', 21)
        s = _loaded(p)
        ed._on_loaded('run1', s)
        before = _linear(s, 'Mass101')
        res = run_bead_normalization(ed, ['run1'], BEADS, window=200)
        assert apply_bead_results(ed, res) == ['run1'], res['skipped']
        want = _linear(s, 'Mass101')
        fac = bn.factors_at(s.data['Time'].to_numpy(),
                            s.bead_norm['knots_t'], s.bead_norm['knots_f'])
        np.testing.assert_allclose(want, before * fac, rtol=1e-6, atol=1e-6)
        path = tmp_path / 'run1.flowsession'
        if sidecar:
            # A sidecar is written for a sample with computed columns.
            s.data['cluster'] = 0
        data = ed._write_session(str(path))
        entry = data['samples'][0]
        assert entry['bead_norm']['knots_f'] == s.bead_norm['knots_f']
        assert bool(entry.get('processed_data')) == sidecar
        ed2, jobs = _restore_session(root, path)
        assert bool(jobs['csv']) == sidecar, jobs
        s2 = ed2._samples['run1']
        np.testing.assert_allclose(_linear(s2, 'Mass101'), want, rtol=1e-5,
                                   atol=1e-5)
        assert s2.bead_norm['knots_t'] == s.bead_norm['knots_t']
        # Undo still works after the reopen.
        from openflo.ui_mass_cytometry import restore_bead_normalization
        assert restore_bead_normalization(ed2) == ['run1']
        np.testing.assert_allclose(_linear(s2, 'Mass101'), before,
                                   rtol=1e-5, atol=1e-5)
    finally:
        root.destroy()
