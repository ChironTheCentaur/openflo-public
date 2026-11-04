"""Analyze > Proliferation fits dye-dilution generations and reports the
standard indices (Roederer 2011).
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import numpy as np
import pytest

import tests._sample_editor as shared
from openflo.proliferation import fit_generations, proliferation_indices
from tests._sample_editor import _set

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


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
    # The fit splits overlapping peaks (log2 sd 0.22, one generation apart);
    # seeds 1-6 stayed within 1.3% of each count, so these bounds are loose.
    assert np.allclose(got, counts, rtol=0.06, atol=60), got
    assert sum(fit['counts'][5:]) < 100
    # Fitted generation spacing; seeds 1-6 within 0.007 of 1.
    assert fit['spacing'] == pytest.approx(1.0, abs=0.05)
    # Undivided peak fitted from the data; seeds 1-6 within 0.6%.
    assert fit['undivided'] == pytest.approx(20_000, rel=0.05)
    truth = proliferation_indices(counts)
    # Division index from the fitted counts; seeds 1-6 within 0.005 of truth.
    assert fit['division_index'] == pytest.approx(truth['division_index'],
                                                  abs=0.08)
    assert (fit['labels'][:3000] == 0).mean() > 0.9
    assert (fit['labels'][-1500:] == 4).mean() > 0.8
    if undivided:                    # a control's peak stays where it is
        assert fit['undivided'] == pytest.approx(undivided, rel=1e-9)


def test_fit_generations_too_few_events_reads_na():
    assert not fit_generations(np.full(20, 100.0))['ok']
    assert not fit_generations(-np.ones(500))['ok']


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
    # Same peak-splitting fit on fewer events (7,000); seeds 1-6 within 1.3%.
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


def test_proliferation_population_starts_on_the_selected_gate(editor):
    # It opened on whatever gate was last in the list (an imported cell-cycle
    # phase, say), so the first fit ran on the wrong cells. Now: the gate
    # selected in the tree, else all events.
    from openflo.tree_ids import gate_iid
    from openflo.ui_proliferation import ALL_EVENTS, ProliferationWindow
    root, ed = editor
    _set(ed, 'alpha_1', **{'FL1-A': _generations([1500, 1200, 2000])})
    gates = ed._sample_gates['alpha_1']
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(())
    root.update()
    win = ProliferationWindow(ed)
    assert win.population_var.get() == ALL_EVENTS      # not 'FL2 pos' (last)
    win.destroy()
    parent = next(g for g, d in gates.items() if d['name'] == 'FL4 pos')
    ed.gate_tv.selection_set(gate_iid('alpha_1', parent))
    root.update()
    win = ProliferationWindow(ed)
    assert win.population_var.get() == 'FL4 pos'
    # And the fit uses that gate's cells: as many as the gate holds.
    assert win.values.size == shared._mask_count(ed, 'alpha_1', 'FL4 pos')
    win.destroy()
