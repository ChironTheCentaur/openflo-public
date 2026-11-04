"""Cell cycle can fit the Dean-Jett-Fox model (Fox 1980): phase fractions are
fitted areas, so S-phase cells under the peaks count as S. The pipeline
and the Cell cycle dialog use it, and the window draws the fit.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import numpy as np
import pytest

import tests._sample_editor as shared
import tests.test_windows_nothing_hidden as sweep
from openflo.cellcycle_djf import fit_dean_jett_fox, phase_by_model

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


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
    # Truth is the drawn phases, so sampling noise cancels and only the fit's
    # error remains (40,000 events); seeds 0-7 stayed within ~1 point.
    assert np.allclose(got, truth, atol=2.0), (got, truth)
    assert abs(sum(got) - 100) < 1e-6
    assert 1.9 < fit['g2_g1_ratio'] < 2.1
    # Fitted CV of a 4%-broadened G1 peak; seeds 0-7 read 3.96 to 4.03, so 0.6 is loose.
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
    v, _code = _dna(seed=5)
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
    # Same 40,000-event fit through the pipeline; seeds 0-7 within ~1 point of truth.
    assert np.allclose(got, truth, atol=2.0)
    assert model['boundaries'] == {p: model_b[p] for p in (
        'pct_g1', 'pct_s', 'pct_g2m', 'g1_mean', 'g2_mean')}
    assert len(model['djf_params']) == 9
    assert (phases != phases_b).any()
    s.cell_cycle(dna_channel='FL3-A', k=2.0, method='dean-jett-fox')
    assert list(s.data['cell_cycle']) == list(phases)
    with pytest.raises(ValueError):
        s.cell_cycle_model(dna_channel='FL3-A', method='modfit')


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


def test_cell_cycle_run_passes_the_model_on(editor):
    from openflo.async_task import wait_for_job
    _root, ed = editor
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
