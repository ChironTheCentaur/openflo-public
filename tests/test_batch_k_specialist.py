"""Batch K (Complaints 21, 24, 25, 28): specialist analysis.

- Cell cycle can fit the Dean-Jett-Fox model (Fox 1980): phase fractions
  are fitted areas, so S-phase cells under the peaks count as S.
- Analyze > Proliferation fits dye-dilution generations and reports the
  standard indices (Roederer 2011).
- Tools > Spreading matrix measures the spillover spread of conventional
  compensation from compensated single-stain controls.
- Spectral unmixing can use several autofluorescence signatures, each cell
  taking the one that fits it best; Spectral QC draws the NxN unmixed pairs.
- Tools > Bead tracking keeps a Levey-Jennings log judged by the Westgard
  rules; MESF calibration can fit log-log.
- Index-sort keywords give each sorted cell its plate well, shown in the
  plate view.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by
a mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from openflo import bead_tracking as bt
from openflo import index_sort as ix
from openflo.cellcycle_djf import fit_dean_jett_fox, phase_by_model
from openflo.comp_spread import (
    compensation_spread_matrix,
    primary_channel,
    worst_pairs,
)
from openflo.proliferation import fit_generations, proliferation_indices
from openflo.spectral_af import (
    af_signatures,
    unmix_multi_af,
    unmixed_pairs_figure,
)

os.environ.setdefault('MPLBACKEND', 'Agg')


# ── Dean-Jett-Fox ───────────────────────────────────────────────────────────

def _dna(n=40_000, g1=0.56, s=0.30, seed=3):
    """DNA content with known phases: G1 at 50,000 (CV 4%), G2/M at
    100,000, S spread evenly between the two true means, every cell
    measured with the same 4% broadening."""
    rng = np.random.default_rng(seed)
    code = rng.choice(3, n, p=[g1, s, 1 - g1 - s])
    true = np.where(code == 0, 50_000.0,
                    np.where(code == 2, 100_000.0,
                             rng.uniform(50_000, 100_000, n)))
    return true * rng.normal(1.0, 0.04, n), code


def test_dean_jett_fox_recovers_the_true_phase_fractions():
    from openflo.pipeline import analyze_dna
    v, code = _dna()
    truth = [100 * np.mean(code == k) for k in range(3)]
    m = analyze_dna(v)
    fit = fit_dean_jett_fox(v, m['g1_mean'], m['g2_mean'], m['g1_sd'],
                            m['g2_sd'])
    assert fit['ok']
    got = [fit['pct_g1'], fit['pct_s'], fit['pct_g2m']]
    assert np.allclose(got, truth, atol=2.0), (got, truth)
    assert abs(sum(got) - 100) < 1e-6
    assert 1.9 < fit['g2_g1_ratio'] < 2.1
    assert fit['g1_cv'] == pytest.approx(4.0, abs=0.6)
    # The boundaries method counts the S cells under the G2 peak as G2.
    bounds = [m['pct_g1'], m['pct_s'], m['pct_g2m']]
    assert abs(bounds[2] - truth[2]) > abs(got[2] - truth[2]) + 1.0


def test_dean_jett_fox_without_enough_events_reads_na():
    fit = fit_dean_jett_fox(np.full(50, 5.0), 5.0)
    assert not fit['ok'] and np.isnan(fit['pct_s'])
    assert not fit_dean_jett_fox(np.ones(1000), float('nan'))['ok']


def test_phase_by_model_follows_the_fitted_curves():
    from openflo.pipeline import analyze_dna, assign_phase
    v, code = _dna(seed=5)
    m = analyze_dna(v)
    fit = fit_dean_jett_fox(v, m['g1_mean'], m['g2_mean'], m['g1_sd'],
                            m['g2_sd'])
    cur = assign_phase(v, m)
    new = phase_by_model(v, fit, cur)
    keep = ~np.isin(cur, ['G1', 'S', 'G2M'])
    assert (new[keep] == cur[keep]).all()        # sub-G1, >G2M, NA stay
    # Cells at the G1 mean are G1, midway are S, at the G2 mean G2M.
    probe = phase_by_model(np.array([50_000.0, 75_000.0, 100_000.0]), fit,
                           np.array(['S', 'G1', 'S'], dtype=object))
    assert list(probe) == ['G1', 'S', 'G2M']
    assert list(phase_by_model(v[:3], {'ok': False}, cur[:3])) == \
        list(cur[:3])


# ── Proliferation ───────────────────────────────────────────────────────────

def test_proliferation_indices_by_hand():
    # 100 undivided, 200 in generation 1 (100 precursors), 400 in 2 (100).
    r = proliferation_indices([100, 200, 400])
    assert r['precursor_frequency'] == pytest.approx([1 / 3] * 3)
    assert r['pct_divided'] == pytest.approx(200 / 3)
    assert r['division_index'] == pytest.approx(1.0)
    assert r['proliferation_index'] == pytest.approx(1.5)
    assert r['expansion_index'] == pytest.approx(700 / 300)
    none = proliferation_indices([0, 0])
    assert np.isnan(none['pct_divided'])
    undivided = proliferation_indices([50, 0, 0])
    assert undivided['pct_divided'] == 0
    assert np.isnan(undivided['proliferation_index'])


def _generations(counts, m0=20_000.0, sd=0.22, seed=1):
    rng = np.random.default_rng(seed)
    parts = [m0 / 2 ** k * 2 ** rng.normal(0, sd, n)
             for k, n in enumerate(counts)]
    return np.concatenate(parts)


@pytest.mark.parametrize('undivided', [None, 20_000.0])
def test_fit_generations_recovers_the_counts(undivided):
    counts = [3000, 2500, 4000, 3000, 1500]
    v = _generations(counts)
    fit = fit_generations(v, undivided=undivided, max_generations=6)
    assert fit['ok']
    got = fit['counts'][:5]
    assert np.allclose(got, counts, rtol=0.06, atol=60), got
    assert sum(fit['counts'][5:]) < 100
    assert fit['spacing'] == pytest.approx(1.0, abs=0.05)
    assert fit['undivided'] == pytest.approx(20_000, rel=0.05)
    truth = proliferation_indices(counts)
    assert fit['division_index'] == pytest.approx(truth['division_index'],
                                                  abs=0.08)
    assert (fit['labels'][:3000] == 0).mean() > 0.9
    assert (fit['labels'][-1500:] == 4).mean() > 0.8
    if undivided:                    # a control's peak stays where it is
        assert fit['undivided'] == pytest.approx(undivided, rel=1e-9)


def test_fit_generations_too_few_events_reads_na():
    assert not fit_generations(np.full(20, 100.0))['ok']
    assert not fit_generations(-np.ones(500))['ok']


# ── Spreading matrix ────────────────────────────────────────────────────────

CH = ['FL1-A', 'FL2-A', 'FL3-A']


def _control(j, into, ss, n=3000, seed=0):
    """A compensated single stain for channel ``j``: half negative, half
    bright, spreading ``ss`` x sqrt(signal) into channel ``into``."""
    rng = np.random.default_rng(seed)
    neg = rng.normal(0, 30, (n, 3))
    pos = rng.normal(0, 30, (n, 3))
    pos[:, j] += rng.lognormal(9, 0.3, n)
    pos[:, into] += rng.normal(0, 1, n) * np.sqrt(pos[:, j]) * ss
    return np.vstack([neg, pos])


def test_spread_matrix_recovers_the_spread_and_leaves_missing_na():
    ssm, chans = compensation_spread_matrix(
        {'FL1-A': _control(0, 1, 0.5), 'FL3-A': _control(2, 0, 1.2, seed=1)},
        CH)
    assert chans == CH
    assert ssm[1, 0] == pytest.approx(0.5, rel=0.1)    # FL1 into FL2
    assert ssm[0, 2] == pytest.approx(1.2, rel=0.1)    # FL3 into FL1
    assert ssm[2, 0] < 0.1 and ssm[0, 0] == 0
    assert np.isnan(ssm[:, 1]).all()                   # no FL2 control
    assert worst_pairs(ssm, CH, top=1) == [('FL3-A', 'FL1-A',
                                            pytest.approx(ssm[0, 2]))]


def test_primary_channel_is_the_stain_not_its_spread():
    assert primary_channel(_control(0, 1, 2.0), CH) == 'FL1-A'
    assert primary_channel(_control(2, 1, 2.0), CH) == 'FL3-A'
    un = np.random.default_rng(4).normal(0, 30, (2000, 3))
    assert primary_channel(_control(2, 1, 2.0), CH, unstained=un) == 'FL3-A'
    assert primary_channel(np.zeros((3, 3)), CH) is None


# ── Multiple autofluorescence ───────────────────────────────────────────────

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
    S, U, _lab, truth = _af_panel()
    fig = unmixed_pairs_figure(truth, ['F1', 'F2', 'F3'])
    assert len(fig.axes) == 9
    assert len(fig.axes[0].patches) > 0              # a histogram on the diagonal
    assert len(fig.axes[1].collections) == 1         # a scatter off it


# ── Bead tracking ───────────────────────────────────────────────────────────

@pytest.mark.parametrize('values, last', [
    ([100, 131], ['1-3s', '1-2s']),
    ([100, 121, 122], ['2-2s', '1-2s']),
    ([100, 121, 78], ['R-4s', '1-2s']),
    ([100, 111, 112, 111, 113], ['4-1s']),
    ([100] + [101] * 10, ['10x']),
    ([100, 121, 99], []),
    ([100, 121, 90, 122], ['1-2s']),
    ([100, 111, 112, 95, 113], []),
    ([100] + [101] * 8 + [99, 101], []),
    ([100, 111, 112, 89, 113], []),
    ([100, 121, 165], ['1-3s', '2-2s', '1-2s']),
])
def test_westgard_rules(values, last):
    assert bt.westgard(values, 100.0, 10.0)[-1] == last


def test_westgard_without_a_baseline_judges_nothing():
    m, sd = bt.baseline([100.0])
    assert np.isnan(sd)
    assert bt.westgard([100, 500], m, sd) == [[], []]
    m, sd = bt.baseline([5, 5, 5])
    assert np.isnan(sd)
    v = [100, np.nan, 131]
    assert bt.westgard(v, 100, 10) == [[], [], ['1-3s', '1-2s']]


def test_bead_log_baseline_status_and_csv_round_trip():
    log = bt.BeadLog(baseline_runs=4)
    for i, x in enumerate([100, 102, 98, 100, 101, 140]):
        log.add(f'2026-01-0{i + 1}', f'beads_{i}', {'FL1-A': x, 'FL2-A': 50})
    log.add('2026-01-07', 'beads_6', {'FL1-A': 99})
    j = log.judge('FL1-A')
    assert j['mean'] == pytest.approx(100.0)
    assert j['sd'] == pytest.approx(np.std([100, 102, 98, 100], ddof=1))
    assert j['status'][5] == 'reject' and j['status'][4] == 'ok'
    assert log.judge('FL2-A')['status'][-1] == 'NA'
    text = log.to_csv()
    assert text.splitlines()[0] == 'date,sample,FL1-A,FL2-A'
    assert text.splitlines()[-1].endswith(',NA')
    back = bt.BeadLog.from_csv(text, baseline_runs=4)
    assert back.runs == log.runs
    with pytest.raises(ValueError):
        bt.BeadLog.from_csv("when,what\n1,2\n")


# ── Index sort ──────────────────────────────────────────────────────────────

def test_parse_locations_forms():
    assert ix.parse_locations('0,0;0,1;7,11') == [(0, 0), (0, 1), (7, 11)]
    assert ix.parse_locations('1,1;2,3') == [(0, 0), (1, 2)]
    assert ix.parse_locations('A1;B02;junk;H12') == [(0, 0), (1, 1), None,
                                                    (7, 11)]
    assert ix.parse_locations('nothing here') == []
    assert ix.parse_locations('') == []


def test_event_wells_take_each_events_location():
    import pandas as pd
    data = pd.DataFrame({'FL1-A': [10.0, 20.0, 30.0, 40.0]})
    s = SimpleNamespace(name='sort_1', data=data, metadata={
        'INDEX SORTING LOCATIONS': '0,0;0,1;2,5'})
    assert ix.has_index_sort(s)
    rows, cols = ix.event_wells(s)
    assert list(rows) == [0, 0, 2, -1] and list(cols) == [0, 1, 5, -1]
    assert ix.well_values(s, 'FL1-A') == {(0, 0): 10.0, (0, 1): 20.0,
                                          (2, 5): 30.0}
    s2 = SimpleNamespace(name='plain', data=data, metadata={'$INDEX': 'C3'})
    assert ix.well_values(s2, 'FL1-A') == {(2, 2): 10.0}
    assert not ix.has_index_sort(SimpleNamespace(name='x', data=data,
                                                 metadata={}))


# ── Log-log MESF ────────────────────────────────────────────────────────────

def test_loglog_mesf_fit_follows_a_power_law_the_linear_fit_misses():
    from openflo.calibration import apply_calibration, fit_mesf_calibration
    mfi = np.array([40.0, 400.0, 4_000.0, 40_000.0, 400_000.0])
    known = 3.0 * mfi ** 0.9                  # a slightly non-linear detector
    c = fit_mesf_calibration(mfi, known, model='loglog')
    assert c['model'] == 'loglog' and c['n'] == 5
    assert c['slope'] == pytest.approx(0.9)
    assert c['intercept'] == pytest.approx(np.log10(3.0))
    got = apply_calibration(mfi, c['slope'], c['intercept'], model='loglog')
    assert got == pytest.approx(known)
    lin = fit_mesf_calibration(mfi, known)
    assert lin['model'] == 'linear'
    low = apply_calibration(mfi[:1], lin['slope'], lin['intercept'])
    assert abs(low[0] - known[0]) / known[0] > 1.0      # the dim end is lost
    # Non-positive MFI has no log-log value: 0 when clipped, else NaN.
    assert list(apply_calibration([0.0, -5.0], 1.0, 0.0, model='loglog')) \
        == [0.0, 0.0]
    assert np.isnan(apply_calibration([0.0], 1.0, 0.0, clip=False,
                                      model='loglog')).all()
    c2 = fit_mesf_calibration([0.0, 10.0, 100.0], [5.0, 10.0, 100.0],
                              model='loglog')
    assert c2['n'] == 2                       # the zero peak is left out
    with pytest.raises(ValueError):
        fit_mesf_calibration(mfi, known, model='cubic')


# ── Cell cycle in the pipeline ──────────────────────────────────────────────

def _cc_sample(seed=3):
    import pandas as pd

    from openflo.pipeline import FlowSample
    v, code = _dna(seed=seed)
    frame = pd.DataFrame({'FSC-A': np.full(v.size, 8e4), 'FL3-A': v})
    return FlowSample.from_dataframe(frame, name='sample_cc'), code


def test_cell_cycle_model_dean_jett_fox_method():
    s, code = _cc_sample()
    truth = [100 * np.mean(code == k) for k in range(3)]
    phases_b, model_b = s.cell_cycle_model(dna_channel='FL3-A', k=2.0)
    phases, model = s.cell_cycle_model(dna_channel='FL3-A', k=2.0,
                                       method='dean-jett-fox')
    assert model_b['method'] == 'boundaries' and 'djf_params' not in model_b
    assert model['method'] == 'Dean-Jett-Fox'
    got = [model['pct_g1'], model['pct_s'], model['pct_g2m']]
    assert np.allclose(got, truth, atol=2.0)
    assert model['boundaries'] == {p: model_b[p] for p in (
        'pct_g1', 'pct_s', 'pct_g2m', 'g1_mean', 'g2_mean')}
    assert len(model['djf_params']) == 9
    assert (phases != phases_b).any()
    s.cell_cycle(dna_channel='FL3-A', k=2.0, method='dean-jett-fox')
    assert list(s.data['cell_cycle']) == list(phases)
    with pytest.raises(ValueError):
        s.cell_cycle_model(dna_channel='FL3-A', method='modfit')


# ── In the editor ──────────────────────────────────────────────────────────

import tests.test_windows_nothing_hidden as sweep  # noqa: E402

virtual_screen = sweep.virtual_screen
no_dialogs = sweep.no_dialogs
engines_known = sweep.engines_known


@pytest.fixture
def editor(virtual_screen, no_dialogs, engines_known):
    virtual_screen(1920, 1080)
    root = sweep._root(1.0)
    try:
        yield root, sweep._editor(root)
    finally:
        root.destroy()


def _set(ed, name, **cols):
    import pandas as pd
    n = len(next(iter(cols.values())))
    df = pd.DataFrame({c: np.zeros(n) for c in sweep.COLS})
    for c, v in cols.items():
        df[c] = v
    ed._samples[name].data = df
    ed._data_changed(name)


def test_menus_offer_the_new_windows(editor):
    _root, ed = editor
    labels = {label for _m, _i, label in sweep._menu_items(ed)}
    assert {"Proliferation…", "Spreading matrix…", "Bead tracking…"} <= labels


def test_proliferation_window_fits_the_active_sample(editor):
    from openflo.ui_proliferation import ALL_EVENTS, ProliferationWindow
    root, ed = editor
    counts = [1500, 1200, 2000, 1500, 800]
    _set(ed, 'alpha_1', **{'FL1-A': _generations(counts)})
    _set(ed, 'alpha_2', **{'FL1-A': _generations([3000], seed=4)})
    win = ProliferationWindow(ed)
    root.update()
    win.channel_var.set('FL1-A')
    win.population_var.set(ALL_EVENTS)
    win.gen_var.set(6)
    win.fit()
    assert win.result['ok']
    assert np.allclose(win.result['counts'][:5], counts, rtol=0.08, atol=60)
    assert len(win.table.get_children()) == 7
    assert 'Division index' in win.status_var.get()
    rows = win.rows()
    assert rows[0] == ['generation', 'cells', 'precursor_frequency']
    assert any(r[:1] == ['division_index'] for r in rows)
    # The undivided peak from an unstimulated control's median.
    win.control_var.set('alpha_2')
    win.fit()
    control = np.median(_generations([3000], seed=4))
    assert win.result['undivided'] == pytest.approx(control, rel=1e-9)
    # A population the control lacks reads as such, not as a fit.
    win.population_var.set('FL2 pos')
    win.fit()
    assert "alpha_2 has no events" in win.status_var.get() or \
        "has no population" in win.status_var.get()
    win.destroy()


def test_proliferation_window_reads_na_with_too_few_cells(editor):
    from openflo.ui_proliferation import ALL_EVENTS, ProliferationWindow
    root, ed = editor
    _set(ed, 'alpha_1', **{'FL1-A': np.r_[np.full(20, 100.0),
                                          np.zeros(380)]})
    win = ProliferationWindow(ed)
    win.channel_var.set('FL1-A')
    win.population_var.set(ALL_EVENTS)
    win.fit()
    root.update()
    assert not win.result['ok']
    assert 'indices read NA' in win.status_var.get()
    assert win.rows() == [['note', 'too few events; indices NA']]
    win.destroy()


def _comp(ed, names):
    for n in names:
        s = ed._samples[n]
        s.comp_channels = list(CH)
        s.comp_matrix = np.eye(3)


def test_spread_matrix_window_matches_controls_and_computes(editor):
    from openflo.ui_comp_spread import NONE, SpreadMatrixWindow
    root, ed = editor
    c1, c3 = _control(0, 1, 0.5), _control(2, 0, 1.2, seed=1)
    _set(ed, 'alpha_1', **{c: c1[:, i] for i, c in enumerate(CH)})
    _set(ed, 'alpha_2', **{c: c3[:, i] for i, c in enumerate(CH)})
    un = np.random.default_rng(9).normal(0, 30, (500, 3))
    _set(ed, 'beta_1', **{c: un[:, i] for i, c in enumerate(CH)})
    _comp(ed, ['alpha_1', 'alpha_2', 'beta_1'])
    win = SpreadMatrixWindow(ed)
    root.update()
    assert win.channels == CH
    assert win.controls() == {'FL1-A': 'alpha_1', 'FL3-A': 'alpha_2'}
    win.compute()
    disp = win.display()                       # stains on the rows
    assert disp[0, 1] == pytest.approx(0.5, rel=0.1)
    assert disp[2, 0] == pytest.approx(1.2, rel=0.1)
    assert np.isnan(disp[1]).all()
    assert 'FL3-A into FL1-A' in win.status_var.get()
    rows = win.rows()
    assert rows[0] == ['stain'] + CH and rows[2][1:] == ['NA'] * 3
    # A control that is not compensated is left out and named.
    ed._samples['alpha_2'].comp_matrix = None
    win.compute()
    assert 'Skipped (not compensated): alpha_2' in win.status_var.get()
    assert np.isnan(win.display()[2]).all()
    win.control_vars['FL1-A'].set(NONE)
    win.control_vars['FL3-A'].set(NONE)
    win.compute()
    assert win.status_var.get() == "Pick at least one control."
    win.destroy()


def test_spread_matrix_window_needs_compensation(editor):
    from openflo.ui_comp_spread import SpreadMatrixWindow
    root, ed = editor
    win = SpreadMatrixWindow(ed)
    root.update()
    assert win.channels == [] and 'has no compensation' in \
        win.status_var.get()
    win.destroy()


def test_bead_tracking_window_adds_runs_and_judges(editor, tmp_path,
                                                   monkeypatch):
    from openflo import ui_bead_tracking as ub
    root, ed = editor
    win = ub.BeadTrackingWindow(ed)
    root.update()
    win.population_var.set('All events')
    for i, level in enumerate([1000, 1010, 990, 1000, 1300]):
        _set(ed, 'alpha_1', **{'FL1-A': np.full(400, float(level)),
                               'FL2-A': np.full(400, 50.0 + i)})
        ed._samples['alpha_1'].metadata = {'$DATE': f'0{i + 1}-JAN-2026'}
        win.add_active()
    win.baseline_var.set(4)
    win.channel_var.set('FL1-A')
    win.draw()
    j = win.judged()
    assert j['mean'] == pytest.approx(1000.0)
    assert j['status'][-1] == 'reject'
    rows = win.table.get_children()
    assert len(rows) == 5
    assert '1-3s' in win.table.item(rows[-1])['values'][2]
    assert win.table.item(rows[0])['values'][0] == '01-JAN-2026'
    path = tmp_path / 'beads.csv'
    monkeypatch.setattr(ub.filedialog, 'asksaveasfilename',
                        lambda **k: str(path))
    win.save_log()
    back = bt.BeadLog.from_csv(path.read_text())
    assert [r['values']['FL1-A'] for r in back.runs] == \
        [1000, 1010, 990, 1000, 1300]
    monkeypatch.setattr(ub.filedialog, 'askopenfilename',
                        lambda **k: str(path))
    win2 = ub.BeadTrackingWindow(ed)
    win2.open_log()
    assert len(win2.log.runs) == 5 and win2.channel_var.get() == 'FL1-A'
    win.population_var.set('no such gate')
    win.add_active()
    assert 'has no events' in win.status_var.get()
    assert len(win.log.runs) == 5
    win.destroy()
    win2.destroy()


def test_plate_view_shows_index_sorted_cells(editor):
    from openflo.ui_plate import INDEX_PREFIX, PlateWindow
    root, ed = editor
    vals = np.arange(400, dtype=float)
    _set(ed, 'alpha_1', **{'FL1-A': vals})
    ed._samples['alpha_1'].metadata = {
        'INDEX SORTING LOCATIONS': '0,0;0,1;1,0;7,11'}
    win = PlateWindow(ed)
    root.update()
    assert win.index_samples == ['alpha_1']
    win.source_var.set(INDEX_PREFIX + 'alpha_1')
    win._source_changed()
    win.stat_var.set('FL1-A')
    win.draw()
    mat = win.matrix()
    assert mat[0, 0] == 0 and mat[0, 1] == 1 and mat[1, 0] == 2 \
        and mat[7, 11] == 3
    assert np.isfinite(mat).sum() == 4
    assert win.rows()[0] == ('A01', 'alpha_1 event 1', 0.0)
    win.source_var.set('Samples')
    win._source_changed()
    assert win.stat_var.get() == 'Events'
    win.destroy()


def test_cell_cycle_dialog_runs_dean_jett_fox_by_default(editor,
                                                         monkeypatch):
    root, ed = editor
    seen = {}
    monkeypatch.setattr(ed, '_run_cell_cycle',
                        lambda col, all_samples, **kw: seen.update(kw))
    ed._open_cell_cycle_dialog()
    root.update()
    assert ed._cell_cycle_model_var.get() == 'Dean-Jett-Fox'
    dlg = [w for w in sweep._toplevels(ed).values() if 'Cell cycle' in w.title()][-1]
    run = next(w for w in sweep._walk(dlg)
               if w.winfo_class() == 'TButton' and w.cget('text') == 'Run')
    run.invoke()
    assert seen['method'] == 'dean-jett-fox'
    ed._open_cell_cycle_dialog()
    root.update()
    ed._cell_cycle_model_var.set('Boundaries')
    dlg = [w for w in sweep._toplevels(ed).values() if 'Cell cycle' in w.title()][-1]
    next(w for w in sweep._walk(dlg) if w.winfo_class() == 'TButton'
         and w.cget('text') == 'Run').invoke()
    assert seen['method'] == 'boundaries'


def _wait_status(root, ed, prefix, timeout=30.0):
    import time
    end = time.monotonic() + timeout
    while not ed.status_var.get().startswith(prefix):
        assert time.monotonic() < end, ed.status_var.get()
        root.update()
        time.sleep(0.01)


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
    name, A, fluors = ed._last_unmix_pairs
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


def test_calibration_dialog_fits_log_log_by_default(editor):
    from openflo.ui_calibration import CalibrationDialog
    root, ed = editor
    _set(ed, 'alpha_1', **{'FL1-A': np.full(400, 400.0)})
    dlg = CalibrationDialog(ed)
    root.update()
    assert dlg.model_var.get() == 'loglog'
    mfi = [40.0, 400.0, 4_000.0, 40_000.0]
    dlg.txt.delete('1.0', 'end')
    dlg.txt.insert('1.0', '\n'.join(f"{m}\t{3.0 * m ** 0.9}" for m in mfi))
    dlg._fit()
    assert dlg._cal['model'] == 'loglog'
    assert dlg._cal['slope'] == pytest.approx(0.9)
    assert 'log value' in dlg.result.cget('text')
    dlg.apply_var.set(ed._fmt_channel('FL1-A'))
    dlg._apply()
    col = next(c for c in ed._samples['alpha_1'].data.columns
               if c.startswith('MESF:'))
    assert ed._samples['alpha_1'].data[col].iloc[0] == \
        pytest.approx(3.0 * 400 ** 0.9)
    dlg.model_var.set('linear')
    dlg._fit()
    assert dlg._cal['model'] == 'linear' and 'log' not in \
        dlg.result.cget('text')
    dlg.destroy()


def test_cell_cycle_run_passes_the_model_on(editor):
    from openflo.async_task import wait_for_job
    root, ed = editor
    seen = {}
    ed._samples['alpha_1'].cell_cycle_model = \
        lambda **kw: seen.update(kw)
    ed._run_cell_cycle('FL3-A', False, k=2.0, method='dean-jett-fox')
    assert wait_for_job(ed._job_strip)
    assert seen['method'] == 'dean-jett-fox'


def test_cell_cycle_window_draws_the_fit(editor):
    from openflo.ui_cell_cycle import CellCycleWindow
    root, ed = editor
    s, _code = _cc_sample()
    s.cell_cycle(dna_channel='FL3-A', k=2.0, method='dean-jett-fox')
    ed._samples['alpha_1'] = s
    win = CellCycleWindow(ed, 'alpha_1')
    root.update()
    labels = [ln.get_label() for ax in win.fig.axes
              for ln in ax.get_lines()]
    texts = [str(w.cget('text')) for w in sweep._walk(win)
             if w.winfo_class() == 'TLabel']
    assert 'S (fit)' in labels and 'Fit' in labels
    assert any(t.startswith('Dean-Jett-Fox:') for t in texts)
    assert any('Boundaries gave' in t for t in texts)
    win.destroy()


def test_unmix_dialog_passes_the_af_signature_count(editor):
    from openflo.ui_spectral_unmix import SpectralUnmixDialog
    root, ed = editor
    got = []
    dlg = SpectralUnmixDialog(ed, ['alpha_1', 'alpha_2'],
                              ['FL1-A', 'FL2-A', 'FL3-A'],
                              lambda *a, **k: got.append(k))
    root.update()
    assert dlg.af_var.get() == 1
    nm, rv, fv = dlg.rows[0]
    rv.set('Single-stain')
    fv.set('F1')
    dlg.af_var.set(3)
    dlg._apply()
    assert got == [{'af_signatures': 3}]
