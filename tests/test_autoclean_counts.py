"""Headless tests for the auto-clean drop-count readout
(ViewGateEditorWindow._autoclean_counts / _drop_suffix). Uses a stub `self`
so no Tk display is needed."""
from __future__ import annotations

import types

import numpy as np

from openflo import synthetic
from openflo.gui import ViewGateEditorWindow as W
from openflo.pipeline import FlowSample, default_autoclean_methods


def _stub(sample, gid='g1'):
    gate = {'kind': 'autoclean', 'name': 'autocleaned sample',
            'methods': default_autoclean_methods()}
    s = types.SimpleNamespace(
        _samples={'S': sample},
        _sample_gates={'S': {gid: gate}},
        _ac_cache={}, _ac_count_cache={})
    return s, gate, gid


def test_drop_suffix_formatting():
    assert W._drop_suffix(0, 0) == ''           # unknown total
    assert W._drop_suffix(None, 100) == ''       # no count
    assert W._drop_suffix(5, 0) == ''            # zero total
    s = W._drop_suffix(150, 1000)
    assert 'drops 150' in s and '15.0%' in s


def test_autoclean_counts_shape_and_cache(synthetic_fcs):
    s = FlowSample(synthetic_fcs)
    stub, gate, gid = _stub(s)
    res = W._autoclean_counts(stub, 'S', gid)
    assert res is not None
    total, total_drop, per, reasons = res
    assert total == len(s.data)
    assert 0 <= total_drop <= total
    # one entry per recipe method
    assert set(per) == {m['key'] for m in gate['methods']}
    assert all(0 <= v <= total for v in per.values())
    # cached: identical result, single cache entry
    assert W._autoclean_counts(stub, 'S', gid) == res
    assert len(stub._ac_count_cache) == 1


def test_autoclean_counts_recomputes_on_recipe_change(synthetic_fcs):
    s = FlowSample(synthetic_fcs)
    stub, gate, gid = _stub(s)
    total, drop0, per0, _r0 = W._autoclean_counts(stub, 'S', gid)
    # Disable every method → nothing dropped (union of no methods = keep all).
    for m in gate['methods']:
        m['enabled'] = False
    total2, drop1, per1, _r1 = W._autoclean_counts(stub, 'S', gid)
    assert drop1 == 0
    # Per-method previews are independent of the enabled flag.
    assert per1 == per0


def test_autoclean_counts_none_when_not_loaded():
    gate = {'kind': 'autoclean', 'methods': default_autoclean_methods()}
    stub = types.SimpleNamespace(
        _samples={}, _sample_gates={'S': {'g1': gate}}, _ac_count_cache={})
    assert W._autoclean_counts(stub, 'S', 'g1') is None


def test_autoclean_counts_none_for_non_autoclean(synthetic_fcs):
    s = FlowSample(synthetic_fcs)
    gate = {'kind': 'threshold', 'channel': 'FL2-A', 'value': 0.5}
    stub = types.SimpleNamespace(
        _samples={'S': s}, _sample_gates={'S': {'g1': gate}},
        _ac_count_cache={})
    assert W._autoclean_counts(stub, 'S', 'g1') is None


def test_removed_events_matches_drop_count(synthetic_fcs):
    """`_removed_events` returns exactly the events the recipe drops (the
    complement of the keep mask) — the data the red overlay draws."""
    s = FlowSample(synthetic_fcs)
    s.apply_transform()
    stub, gate, gid = _stub(s)
    # Bind the collaborator methods the stub needs.
    stub._axis_alias_for_sample = W._axis_alias_for_sample.__get__(stub)
    stub._autoclean_overrides = W._autoclean_overrides.__get__(stub)
    _bind_plot_cache(stub)
    xcol, ycol = 'FL1-A', 'FL2-A'
    _total, drop, _per, _r = W._autoclean_counts(stub, 'S', gid)
    rem = W._removed_events(stub, 'S', xcol, ycol)
    if drop == 0:
        assert rem is None
    else:
        assert rem is not None
        assert len(rem) == drop
        assert xcol in rem.columns and ycol in rem.columns


def test_removed_events_none_without_autoclean(synthetic_fcs):
    s = FlowSample(synthetic_fcs)
    s.apply_transform()
    stub = types.SimpleNamespace(
        _samples={'S': s},
        _sample_gates={'S': {}},          # no auto-clean gate
        _ac_cache={}, _ac_count_cache={})
    stub._axis_alias_for_sample = W._axis_alias_for_sample.__get__(stub)
    stub._autoclean_overrides = W._autoclean_overrides.__get__(stub)
    _bind_plot_cache(stub)
    assert W._removed_events(stub, 'S', 'FL1-A', 'FL2-A') is None


def _bind_plot_cache(stub):
    """`_removed_events` reads the sample through the editor's per-sample
    caches (editor_plotcache): bind those helpers onto the stub as well."""
    from openflo.editor_plotcache import PlotCacheMixin
    for fn, obj in vars(PlotCacheMixin).items():
        if fn.startswith(('_pc_', '_warm_')):
            setattr(stub, fn, obj.__func__ if isinstance(obj, staticmethod)
                    else types.MethodType(obj, stub))


def _dirty_and_clean():
    """A generated sample with known contamination, and the same sample with
    it taken out. openflo.synthetic puts exactly 7 % doublets (FSC-A/FSC-H
    about 3.1 against about 2 for singlets) into a differentiation sample; on
    top of that 600 events are piled up at an SSC-A ceiling, as a saturated
    detector records them. The clean copy drops the doublets by their known
    ratio and has no pile-up."""
    df = synthetic.differentiation_sample(6, 'Ctrl', n=20_000, seed=1)
    dirty = df.copy()
    piled = np.random.default_rng(2).choice(len(df), 600, replace=False)
    dirty.loc[piled, 'SSC-A'] = 262143.0
    clean = df[df['FSC-A'] / df['FSC-H'] < 2.6].reset_index(drop=True)
    return (FlowSample.from_dataframe(dirty, name='dirty'),
            FlowSample.from_dataframe(clean, name='clean'))


def test_autoclean_counts_find_the_contamination_put_there():
    """Positive: every planted doublet and every piled-up event is counted,
    and together they are a meaningful share of the sample (this replaces a
    check that only ran on a real file someone had to supply)."""
    dirty, _clean = _dirty_and_clean()
    stub, _gate, gid = _stub(dirty)
    total, drop, per, _reasons = W._autoclean_counts(stub, 'S', gid)
    assert total == 20_000
    assert per['doublets'] == 1400          # 7 % of 20,000, all of them
    assert per['margin'] == 600
    assert drop / total > 0.05


def test_autoclean_counts_find_nothing_in_a_clean_sample():
    """Negative: with the doublets and the pile-up taken out, the same
    recipe has (almost) nothing left to remove."""
    _dirty, clean = _dirty_and_clean()
    stub, _gate, gid = _stub(clean)
    total, drop, per, _reasons = W._autoclean_counts(stub, 'S', gid)
    assert per['margin'] == 0
    assert per['doublets'] <= total * 0.001
    assert drop <= total * 0.001
