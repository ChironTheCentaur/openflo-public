"""Pure-function tests for the bead-calibrated debris cut and the viability
(dead-cell) auto-clean method, plus viability-dye channel detection. No Tk,
no FCS files — DataFrames are built inline so the maths is exact."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo import synthetic as syn
from openflo.pipeline import (
    _autoclean_debris_mask,
    _autoclean_viability_mask,
    autoclean_keep_mask,
    autoclean_method_diagnostic,
    default_autoclean_methods,
    find_viability_channel,
    transform_values,
)

rng = np.random.default_rng(0)


# ── recipe registry ───────────────────────────────────────────────────────────

def test_default_recipe_has_viability_and_bead_debris():
    methods = {m['key']: m for m in default_autoclean_methods()}
    assert 'viability' in methods
    deb = methods['debris']['params']
    assert deb['mode'] == 'bead'
    assert deb['bead_um'] == 8.0 and deb['min_um'] == 4.0


# ── viability-dye channel detection ─────────────────────────────────────────────

def test_find_viability_by_column_name():
    assert find_viability_channel(['FSC-A', 'LiveDead-A', 'CD3-A']) == 'LiveDead-A'
    assert find_viability_channel(['FSC-A', 'Zombie-A']) == 'Zombie-A'


def test_find_viability_by_label_prefers_area():
    cols = ['FSC-A', 'BV510-H', 'BV510-A', 'CD3-A']
    labels = {'BV510-A': 'Live/Dead Aqua', 'BV510-H': 'Live/Dead Aqua'}
    assert find_viability_channel(cols, labels) == 'BV510-A'


def test_find_viability_whole_word_guards():
    # 'pi' / 'l/d' must not fire on PE / APC / unrelated detectors
    assert find_viability_channel(['FSC-A', 'PE-A', 'APC-A', 'CD3-A']) is None
    # but a real PI / L/D label is found
    assert find_viability_channel(['FSC-A', 'PerCP-A'],
                                  {'PerCP-A': 'PI'}) == 'PerCP-A'


def test_find_viability_none_when_absent():
    assert find_viability_channel(['FSC-A', 'SSC-A', 'CD3-A', 'CD4-A']) is None


# ── viability mask (dead = high signal) ─────────────────────────────────────────

def _live_dead_df(n_live=750, n_dead=250, col='LiveDead-A'):
    # Comparable peak widths, as on a real logicle viability axis (a
    # pathologically narrow live spike would hide the broad dead mode below
    # the prominence floor — not representative of transformed data).
    live = rng.normal(300, 80, n_live)
    dead = rng.normal(3000, 350, n_dead)
    return pd.DataFrame({col: np.concatenate([live, dead])})


def test_viability_drops_dead_population():
    df = _live_dead_df()
    keep = _autoclean_viability_mask(df, {})
    # ~750 live kept, ~250 dead dropped (allow detector slop)
    assert 700 <= keep.sum() <= 800
    # the dropped events are the high-signal ones
    assert df['LiveDead-A'][~keep].mean() > df['LiveDead-A'][keep].mean()


def test_viability_noop_when_all_live():
    df = pd.DataFrame({'LiveDead-A': rng.normal(300, 80, 1000)})
    assert _autoclean_viability_mask(df, {}).all()


def test_viability_explicit_channel_and_manual_ceiling():
    df = _live_dead_df(col='BV510-A')
    # auto-detect would miss a generic detector name; pin it explicitly
    keep = _autoclean_viability_mask(df, {'channel': 'BV510-A'})
    assert 700 <= keep.sum() <= 800
    # a generous manual ceiling keeps everything (dead mode ~5000 ± 400)
    assert _autoclean_viability_mask(df, {'channel': 'BV510-A',
                                          'max_signal': 8000}).all()


def test_viability_noop_without_channel():
    df = pd.DataFrame({'FSC-A': rng.normal(60000, 8000, 1000),
                       'CD3-A': rng.normal(100, 20, 1000)})
    assert _autoclean_viability_mask(df, {}).all()


# ── bead-calibrated debris cut ──────────────────────────────────────────────────

def test_debris_bead_absolute_size_threshold():
    # bead_fsc 60000 ≙ 8 µm; min_um 4 ⇒ FSC-A threshold = 4*60000/8 = 30000
    df = pd.DataFrame({'FSC-A': np.array([10000, 25000, 29999, 30001, 60000],
                                         dtype=float)})
    keep = _autoclean_debris_mask(
        df, {'mode': 'bead', 'bead_fsc': 60000.0, 'bead_um': 8.0, 'min_um': 4.0})
    assert list(keep) == [False, False, False, True, True]


def test_debris_falls_back_to_valley_without_bead_anchor():
    # bimodal: debris ~7000, cells ~60000 — valley cut should drop the low mode
    df = pd.DataFrame({'FSC-A': np.concatenate([
        rng.normal(7000, 1500, 300), rng.normal(60000, 8000, 700)])})
    keep = _autoclean_debris_mask(df, {'mode': 'bead', 'min_um': 4.0})  # no bead_fsc
    assert 600 <= keep.sum() <= 800           # ~the 700 cells survive
    assert df['FSC-A'][~keep].mean() < df['FSC-A'][keep].mean()


def test_debris_2d_scatter_gate_keeps_granular_cells():
    """The valley fallback is a 2-D FSC-A × SSC-A gate: it drops the
    bottom-left corner (low FSC AND low SSC) but KEEPS low-FSC / high-SSC
    granular cells that a 1-D FSC cut would wrongly remove."""
    big_fsc   = rng.normal(60000, 8000, 600)   # main cells
    deb_fsc   = rng.normal(7000, 1500, 300)     # debris: low FSC + low SSC
    gran_fsc  = rng.normal(9000, 1500, 200)     # granulocytes: low FSC, HIGH SSC
    big_ssc   = rng.normal(30000, 8000, 600)
    deb_ssc   = rng.normal(4000, 1200, 300)
    gran_ssc  = rng.normal(60000, 9000, 200)
    df = pd.DataFrame({
        'FSC-A': np.concatenate([big_fsc, deb_fsc, gran_fsc]),
        'SSC-A': np.concatenate([big_ssc, deb_ssc, gran_ssc])})
    keep = _autoclean_debris_mask(df, {'mode': 'valley'})   # no bead anchor
    kept_low_fsc_high_ssc = keep[600 + 300:]                # the granulocytes
    assert kept_low_fsc_high_ssc.mean() > 0.9               # granular cells kept
    assert keep[600:600 + 300].mean() < 0.1                 # debris corner dropped
    # opting out of SSC (use_ssc=False) reverts to a 1-D FSC cut → cuts both
    keep1d = _autoclean_debris_mask(df, {'mode': 'valley', 'use_ssc': False})
    assert keep1d[600 + 300:].mean() < 0.2                  # granulocytes now cut


def test_debris_manual_min_fsc_overrides():
    df = pd.DataFrame({'FSC-A': np.array([100, 500, 1000, 2000], dtype=float)})
    keep = _autoclean_debris_mask(df, {'min_fsc': 1000, 'mode': 'bead',
                                       'bead_fsc': 60000.0, 'min_um': 4.0})
    assert list(keep) == [False, False, True, True]


# ── recipe dispatch (AND of enabled methods) ────────────────────────────────────

def test_keep_mask_dispatches_viability():
    df = _live_dead_df()
    gate = {'kind': 'autoclean',
            'methods': [{'key': 'viability', 'enabled': True, 'params': {}}]}
    keep = autoclean_keep_mask(gate, df)
    assert 700 <= keep.sum() <= 800
    # disabled ⇒ no-op
    gate['methods'][0]['enabled'] = False
    assert autoclean_keep_mask(gate, df).all()


# ── freeze auto-clean cuts (copy as fixed) ──────────────────────────────────────

def test_freeze_autoclean_pins_debris_and_viability():
    from openflo.pipeline import freeze_autoclean_gate
    fsc = np.concatenate([rng.normal(7000, 1500, 300),
                          rng.normal(60000, 8000, 700)])     # bimodal
    dye = np.concatenate([rng.normal(0.2, 0.04, 800),
                          rng.normal(0.8, 0.05, 200)])       # live low / dead high
    df = pd.DataFrame({'FSC-A': fsc, 'FSC-H': fsc / 2.0, 'LiveDead-A': dye})
    methods = default_autoclean_methods()
    gate = {'kind': 'autoclean', 'methods': methods}
    frozen = freeze_autoclean_gate(gate, df)
    fp = {m['key']: m['params'] for m in frozen['methods']}
    # debris → fixed min_fsc between the two FSC modes
    assert 7000 < fp['debris']['min_fsc'] < 60000
    # viability → resolved channel + fixed ceiling between live/dead
    assert fp['viability']['channel'] == 'LiveDead-A'
    assert 0.2 < fp['viability']['max_signal'] < 0.8
    # source gate untouched (deep-copied)
    src = {m['key']: m['params'] for m in methods}
    assert 'min_fsc' not in src['debris'] and 'max_signal' not in src['viability']


def test_freeze_non_autoclean_is_noop():
    from openflo.pipeline import freeze_autoclean_gate
    g = {'kind': 'threshold', 'channel': 'APC-A', 'value': 0.5}
    assert freeze_autoclean_gate(g, pd.DataFrame({'APC-A': [0.1, 0.9]})) == g


def test_frozen_gate_applies_same_cut_across_samples():
    """A frozen debris cut drops by the FIXED threshold regardless of the
    target's own distribution (the point of freezing)."""
    from openflo.pipeline import autoclean_keep_mask, freeze_autoclean_gate
    src = pd.DataFrame({'FSC-A': np.concatenate([
        rng.normal(7000, 1500, 300), rng.normal(60000, 8000, 700)])})
    gate = {'kind': 'autoclean',
            'methods': [{'key': 'debris', 'enabled': True,
                         'params': {'mode': 'valley'}}]}
    frozen = freeze_autoclean_gate(gate, src)
    thr = frozen['methods'][0]['params']['min_fsc']
    # apply the frozen cut to a different sample: keep == (FSC >= thr)
    other = pd.DataFrame({'FSC-A': np.array([thr - 1, thr + 1, 90000.0])})
    keep = np.asarray(autoclean_keep_mask(frozen, other), bool)
    assert list(keep) == [False, True, True]


def test_frozen_valley_debris_preserves_2d_granulocyte_rescue():
    """Freezing a valley-mode debris gate pins BOTH the FSC valley and the
    SSC-granular threshold, so the frozen gate replays the 2-D cut — keeping
    low-FSC / high-SSC granulocytes — instead of a lossy 1-D floor that would
    drop them, while staying identical/deterministic across samples."""
    from openflo.pipeline import autoclean_keep_mask, freeze_autoclean_gate
    df = pd.DataFrame({
        'FSC-A': np.concatenate([rng.normal(60000, 8000, 600),   # main cells
                                 rng.normal(7000, 1500, 300),     # debris
                                 rng.normal(9000, 1500, 200)]),   # granulocytes
        'SSC-A': np.concatenate([rng.normal(30000, 8000, 600),
                                 rng.normal(4000, 1200, 300),
                                 rng.normal(60000, 9000, 200)])})
    gate = {'kind': 'autoclean',
            'methods': [{'key': 'debris', 'enabled': True,
                         'params': {'mode': 'valley'}}]}
    live = np.asarray(autoclean_keep_mask(gate, df), bool)       # per-sample 2-D
    frozen = freeze_autoclean_gate(gate, df)
    fp = frozen['methods'][0]['params']
    assert 'min_fsc' in fp and 'min_ssc_granular' in fp          # BOTH pinned
    keep = np.asarray(autoclean_keep_mask(frozen, df), bool)
    assert (keep == live).all()                                  # frozen == live cut
    assert keep[900:].mean() > 0.9                               # granulocytes KEPT
    assert keep[600:900].mean() < 0.1                            # debris still dropped
    # a min_fsc-only (1-D) freeze would instead drop the granulocytes:
    onedim = {'kind': 'autoclean',
              'methods': [{'key': 'debris', 'enabled': True,
                           'params': {'min_fsc': fp['min_fsc']}}]}
    keep1d = np.asarray(autoclean_keep_mask(onedim, df), bool)
    assert keep1d[900:].mean() < 0.2                             # dropped without pin


# ── diagnostics: explain a silent 0-drop ────────────────────────────────────────

def test_diagnostic_viability_no_dye():
    df = pd.DataFrame({'FSC-A': rng.normal(60000, 8000, 500),
                       'CD3-A': rng.normal(100, 20, 500)})
    msg = autoclean_method_diagnostic('viability', df, {})
    assert msg and 'no viability dye' in msg


def test_diagnostic_viability_unimodal_and_majority_high():
    # all-live unimodal → "no bimodal split"
    uni = pd.DataFrame({'LiveDead-A': rng.normal(300, 80, 1000)})
    assert 'no bimodal' in autoclean_method_diagnostic('viability', uni, {})
    # bimodal but the HIGH population is the majority → "majority … not dead"
    maj = pd.DataFrame({'LiveDead-A': np.concatenate([
        rng.normal(300, 80, 250), rng.normal(3000, 350, 750)])})
    msg = autoclean_method_diagnostic('viability', maj, {})
    assert msg and 'majority' in msg


def test_diagnostic_debris_unimodal_suggests_beads():
    df = pd.DataFrame({'FSC-A': rng.normal(60000, 8000, 1000)})   # unimodal
    msg = autoclean_method_diagnostic('debris', df, {'mode': 'bead',
                                                     'min_um': 4.0})
    assert msg and 'unimodal' in msg and 'beads' in msg


def test_diagnostic_none_when_method_works():
    # a clear debris/cell bimodal → debris cuts → no diagnostic
    df = pd.DataFrame({'FSC-A': np.concatenate([
        rng.normal(7000, 1500, 300), rng.normal(60000, 8000, 700)])})
    assert autoclean_method_diagnostic('debris', df, {'mode': 'valley'}) is None
    # a real bead anchor: deterministic cut, no "diagnostic" even if it's small
    assert autoclean_method_diagnostic(
        'debris', df, {'mode': 'bead', 'bead_fsc': 60000.0, 'min_um': 4.0}) is None


# ── continuity: locked reference drops on the synthetic dataset ─────────────────
#
# The synthetic generators are seeded, so the auto-clean recipe must produce the
# SAME drops run-to-run. These reference counts (seed 42, n=20 000) match the
# data's designed composition: ~7 % debris, ~8 % dead, ~5 % doublets. If a
# method's maths changes, these break first — that's the "match back" guard.
# Recompute + update the numbers ONLY with an intended behaviour change.

def _synthetic_autoclean_setup():
    """A PBMC sample (dye logicle-transformed, as the editor stores it) + an
    auto-clean recipe whose debris is anchored to the synthetic size beads and
    whose viability channel is pinned. Returns (df, methods)."""
    bead_fsc = float(np.median(syn.size_bead_sample(seed=950)['FSC-A']))
    raw = syn.immunophenotyping_sample(n=20000, seed=42, group='ctrl')
    via = find_viability_channel(list(raw.columns), syn.PBMC_LABELS)
    d = raw.copy()
    d[via] = transform_values(raw[via].to_numpy(float), method='logicle')
    methods = default_autoclean_methods()
    for m in methods:
        if m['key'] == 'debris':
            m['params']['bead_fsc'] = bead_fsc
        if m['key'] == 'viability':
            m['params']['channel'] = via
    return d, methods, bead_fsc


def _solo_drop(d, m):
    solo = {'kind': 'autoclean', 'methods': [{**m, 'enabled': True}]}
    return int((~np.asarray(autoclean_keep_mask(solo, d), bool)).sum())


def test_continuity_bead_anchor_scale():
    bead_fsc = float(np.median(syn.size_bead_sample(seed=950)['FSC-A']))
    assert bead_fsc / syn.SIZE_BEAD_UM == pytest.approx(
        syn.SIZE_BEAD_FSC_PER_UM, abs=80)


def test_continuity_debris_doublets_exact():
    """Bead-mode debris + doublets are deterministic maths (no peak-finding)
    → locked to exact reference counts (≈7 % debris, ≈5 % doublets of 20 000)."""
    d, methods, _ = _synthetic_autoclean_setup()
    debris = next(m for m in methods if m['key'] == 'debris')
    doublets = next(m for m in methods if m['key'] == 'doublets')
    assert _solo_drop(d, debris) == 1410       # 4 µm bead cut: 1400 debris + 10
    assert _solo_drop(d, doublets) == 1000     # FSC-A/FSC-H ratio band


def test_continuity_viability_and_recipe():
    """Viability (bimodal valley on the logicle dye) ≈ the designed 8 % dead;
    the full recipe drop is stable. Banded (scipy peak-finding)."""
    d, methods, _ = _synthetic_autoclean_setup()
    viab = next(m for m in methods if m['key'] == 'viability')
    assert 1500 <= _solo_drop(d, viab) <= 1700      # ref 1593 (~8 % dead)
    gate = {'kind': 'autoclean', 'methods': methods}
    union = int((~np.asarray(autoclean_keep_mask(gate, d), bool)).sum())
    assert 9800 <= union <= 10500                   # ref 10130


def test_continuity_debris_keeps_real_cells():
    """The bead cut removes only sub-cell debris — live cells (FSC ≈ 60 000,
    ~7.4 µm) are all kept."""
    d, methods, bead_fsc = _synthetic_autoclean_setup()
    debris = next(m for m in methods if m['key'] == 'debris')
    keep = np.asarray(_autoclean_debris_mask(d, debris['params']), bool)
    live = d['FSC-A'].to_numpy(float) > 50000
    assert keep[live].mean() > 0.999                # no real cells removed


def test_gui_autodetect_stamps_bead_and_viability(tmp_path):
    """End-to-end GUI wiring: with the synthetic size-bead file loaded next to a
    PBMC sample, creating an auto-clean gate auto-detects the bead anchor (into
    debris) and the viability dye channel (into viability)."""
    import types

    from openflo.gui import ViewGateEditorWindow as W
    from openflo.pipeline import FlowSample
    from openflo.synthetic import _write_fcs
    bead_fcs, _ = syn.make_size_beads(str(tmp_path), seed=950)
    pbmc_fcs = str(tmp_path / 'ctrl_sample.fcs')
    _write_fcs(pbmc_fcs, syn.immunophenotyping_sample(n=8000, seed=42),
               labels=syn.PBMC_LABELS)
    stub = types.SimpleNamespace(
        _sample_order=['ctrl_sample', 'size_beads'],
        _samples={'ctrl_sample': FlowSample(pbmc_fcs),
                  'size_beads': FlowSample(bead_fcs)})
    stub._resolve_bead_anchor = W._resolve_bead_anchor.__get__(stub)
    stub._autoclean_stamp_refs = W._autoclean_stamp_refs.__get__(stub)

    gate = {'kind': 'autoclean', 'methods': default_autoclean_methods()}
    bead_name = stub._autoclean_stamp_refs('ctrl_sample', gate)
    assert bead_name == 'size_beads'
    deb = next(m for m in gate['methods'] if m['key'] == 'debris')
    viab = next(m for m in gate['methods'] if m['key'] == 'viability')
    assert deb['params']['bead_fsc'] == pytest.approx(
        syn.SIZE_BEAD_UM * syn.SIZE_BEAD_FSC_PER_UM, abs=2000)
    assert viab['params']['channel'] == 'LiveDead-A'
