"""Population-statistics computation (FlowJo-style table).

Tests the pure helpers behind the editor's Statistics window —
_population_path, _population_stats, _collect_stats_rows — against a
small known DataFrame, no Tk. Same stub approach as the other editor
helper tests.
"""
# pyright: reportArgumentType=false, reportCallIssue=false
from __future__ import annotations

import os
import types

import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

try:
    import tkinter as _tk  # noqa: F401

    from openflo.gui import ViewGateEditorWindow as V
except (ImportError, RuntimeError) as e:
    pytest.skip(f"openflo.gui not importable: {e}", allow_module_level=True)


# A 10-row frame where CD11b and CD45 both run 0..9, so gate cumulative
# counts are easy to reason about.
def _df():
    return pd.DataFrame({'CD11b': list(range(10)), 'CD45': list(range(10))})


# g1: CD11b > 4.5  → rows {5,6,7,8,9}  (5 events)
# g2: child of g1, CD45 > 6.5 → cumulative rows {7,8,9} (3 events)
def _gates():
    return {
        'g1': {'kind': 'threshold', 'channel': 'CD11b', 'value': 4.5,
               'parent_id': None, 'name': 'CD11b+'},
        'g2': {'kind': 'threshold', 'channel': 'CD45', 'value': 6.5,
               'parent_id': 'g1', 'name': 'CD45+'},
    }


WANT_ALL = {'Count', '%Parent', '%Total', 'Median'}


def _rows_by_pop(rows):
    return {r['Population']: r for r in rows}


# ── _population_path ─────────────────────────────────────────────────────────

def test_population_path_builds_hierarchy():
    gates = _gates()
    assert V._population_path(gates, 'g1') == 'CD11b+'
    assert V._population_path(gates, 'g2') == 'CD11b+/CD45+'


def test_population_path_cycle_safe():
    gates = {'a': {'name': 'A', 'parent_id': 'b'},
             'b': {'name': 'B', 'parent_id': 'a'}}
    # Must terminate, not loop forever.
    path = V._population_path(gates, 'a')
    assert 'A' in path and 'B' in path


# ── _population_stats: counts + percentages ──────────────────────────────────

def test_population_stats_counts_and_percentages():
    rows = V._population_stats(
        'S1', _df(), _gates(), ['g1', 'g2'],
        {'CD11b': 'CD11b', 'CD45': 'CD45'}, ['CD11b', 'CD45'], WANT_ALL)
    by = _rows_by_pop(rows)
    g1 = by['CD11b+']
    g2 = by['CD11b+/CD45+']

    assert g1['Count'] == 5
    assert g1['%Total'] == pytest.approx(50.0)
    assert g1['%Parent'] == pytest.approx(50.0)        # parent = total (root)

    assert g2['Count'] == 3
    assert g2['%Total'] == pytest.approx(30.0)
    assert g2['%Parent'] == pytest.approx(60.0)        # 3 / 5


def test_population_stats_per_channel_median():
    rows = V._population_stats(
        'S1', _df(), _gates(), ['g1', 'g2'],
        {'CD11b': 'CD11b', 'CD45': 'CD45'}, ['CD11b', 'CD45'], WANT_ALL)
    by = _rows_by_pop(rows)
    # g1 population = rows 5..9 → median of {5,6,7,8,9} = 7
    assert by['CD11b+']['Median CD11b'] == pytest.approx(7.0)
    assert by['CD11b+']['Median CD45'] == pytest.approx(7.0)
    # g2 population = rows 7,8,9 → median 8
    assert by['CD11b+/CD45+']['Median CD11b'] == pytest.approx(8.0)


def test_population_stats_column_selection():
    """Only requested stats appear as keys."""
    rows = V._population_stats(
        'S1', _df(), _gates(), ['g1'],
        {'CD11b': 'CD11b'}, ['CD11b'], {'Count'})
    r = rows[0]
    assert 'Count' in r
    assert '%Parent' not in r
    assert '%Total' not in r
    assert not any(k.startswith('Median') for k in r)


def test_population_stats_empty_population_is_nan():
    """A gate that selects nothing → count 0, NaN per-channel stats."""
    gates = {'g': {'kind': 'threshold', 'channel': 'CD11b', 'value': 999.0,
                   'parent_id': None, 'name': 'none'}}
    rows = V._population_stats(
        'S1', _df(), gates, ['g'], {'CD11b': 'CD11b'}, ['CD11b'],
        {'Count', 'Median'})
    r = rows[0]
    assert r['Count'] == 0
    assert r['Median CD11b'] != r['Median CD11b']     # NaN


def test_population_stats_cv_mean():
    rows = V._population_stats(
        'S1', _df(), _gates(), ['g1'],
        {'CD11b': 'CD11b'}, ['CD11b'], {'Mean', 'CV'})
    r = rows[0]
    # rows 5..9 → mean 7.0
    assert r['Mean CD11b'] == pytest.approx(7.0)
    # std of {5,6,7,8,9} (population, ddof=0) = sqrt(2) ≈ 1.414; CV = /7*100
    assert r['CV CD11b'] == pytest.approx((2 ** 0.5) / 7.0 * 100.0, rel=1e-6)


# ── _collect_stats_rows: aggregation + stable columns ────────────────────────

def _editor_stub():
    df = _df()
    sample = types.SimpleNamespace(data=df, fluor_channels=['CD11b', 'CD45'])
    stub = types.SimpleNamespace(
        _sample_order=['S1'],
        _samples={'S1': sample},
        _sample_gates={'S1': _gates()},
        _sample_gate_order={'S1': ['g1', 'g2']},
        _channel_labels={'CD11b': 'CD11b', 'CD45': 'CD45'},
        STAT_POP=V.STAT_POP,
        STAT_CHAN=V.STAT_CHAN,
        # bound helpers the method calls off self
        _population_stats=V._population_stats,
        _population_path=V._population_path,
    )
    # _sample_rows is a plain instance method → bind it so self is the stub.
    stub._sample_rows = types.MethodType(V._sample_rows, stub)
    return stub


def test_collect_stats_rows_columns_stable_and_ordered():
    stub = _editor_stub()
    rows, cols = V._collect_stats_rows(stub, WANT_ALL)
    assert len(rows) == 2
    # Identity + pop-level columns lead, in canonical order.
    assert cols[:5] == ['Sample', 'Population', 'Count', '%Parent', '%Total']
    # Per-channel median columns present for both channels.
    assert 'Median CD11b' in cols and 'Median CD45' in cols
    # Every row carries every column key the table will render (via .get).
    assert all(r['Sample'] == 'S1' for r in rows)


# ── Cross-sample label tying (Chunk B) ───────────────────────────────────────

def test_stats_ties_per_channel_columns_by_label_across_fluors():
    """Two samples carry CD11b on DIFFERENT detectors (BV421-A vs FITC-A).
    The per-channel median column must be named by the antibody label
    ('Median CD11b') for BOTH, so _collect_stats_rows merges them into a
    single shared column instead of two detector-named ones."""
    import pandas as pd
    a = types.SimpleNamespace(
        data=pd.DataFrame({'BV421-A': [0, 1, 2, 3]}),
        fluor_channels=['BV421-A'],
        channel_labels={'BV421-A': 'CD11b'})
    b = types.SimpleNamespace(
        data=pd.DataFrame({'FITC-A': [10, 20, 30, 40]}),
        fluor_channels=['FITC-A'],
        channel_labels={'FITC-A': 'CD11b'})
    stub = types.SimpleNamespace(
        _sample_order=['A', 'B'],
        _samples={'A': a, 'B': b},
        _sample_gates={
            'A': {'g': {'kind': 'threshold', 'channel': 'BV421-A',
                        'value': -1, 'parent_id': None, 'name': 'all'}},
            'B': {'g': {'kind': 'threshold', 'channel': 'FITC-A',
                        'value': -1, 'parent_id': None, 'name': 'all'}},
        },
        _sample_gate_order={'A': ['g'], 'B': ['g']},
        _channel_labels={},          # editor global empty → per-sample wins
        STAT_POP=V.STAT_POP,
        STAT_CHAN=V.STAT_CHAN,
        _population_stats=V._population_stats,
        _population_path=V._population_path,
    )
    stub._sample_rows = types.MethodType(V._sample_rows, stub)
    rows, cols = V._collect_stats_rows(stub, {'Count', 'Median'})
    # Single shared label column, not two detector columns.
    assert 'Median CD11b' in cols
    assert 'Median BV421-A' not in cols
    assert 'Median FITC-A' not in cols
    by = {r['Sample']: r for r in rows}
    assert by['A']['Median CD11b'] == 1.5      # median {0,1,2,3}
    assert by['B']['Median CD11b'] == 25.0     # median {10,20,30,40}


# ── _axis_alias_for_sample: label-first plot axes (Chunk #48) ─────────────────

def _aliasable_sample(det_to_label, cols):
    """FlowSample-like stub for _axis_alias_for_sample: needs
    channel_labels ({detector: antibody}), fluor_channels (detectors),
    and a .data frame carrying `cols`."""
    s = types.SimpleNamespace()
    s.channel_labels = dict(det_to_label)
    s.fluor_channels = list(det_to_label.keys())
    s.data = pd.DataFrame({c: [0] for c in cols})
    return s


def _editor_with_global_labels(global_det_to_label):
    """Editor stub exposing _channel_labels (global det→label, from the
    first-loaded sample) and the real bound method under test."""
    return types.SimpleNamespace(
        _channel_labels=dict(global_det_to_label),
        _axis_alias_for_sample=V._axis_alias_for_sample,
    )


def test_axis_alias_maps_marker_on_other_fluor():
    """Global panel has CD11b on BV421-A; this sample carries CD11b on
    FITC-A → the chosen BV421-A axis aliases to the sample's FITC-A."""
    ed = _editor_with_global_labels({'BV421-A': 'CD11b', 'SSC-A': 'SSC-A'})
    s = _aliasable_sample({'FITC-A': 'CD11b'}, cols=['SSC-A', 'FITC-A'])
    alias = ed._axis_alias_for_sample(ed, s, ['BV421-A', 'SSC-A'])
    assert alias == {'BV421-A': 'FITC-A'}    # SSC-A present → untouched


def test_axis_alias_noop_when_sample_has_detector():
    """Sample already carries the chosen detector → no alias."""
    ed = _editor_with_global_labels({'BV421-A': 'CD11b'})
    s = _aliasable_sample({'BV421-A': 'CD11b'}, cols=['BV421-A'])
    assert ed._axis_alias_for_sample(ed, s, ['BV421-A']) == {}


def test_axis_alias_skips_marker_sample_lacks():
    """Sample lacks the marker entirely → no alias (sample is dropped
    from the overlay by the plot-path column guard, not aliased here)."""
    ed = _editor_with_global_labels({'BV421-A': 'CD11b'})
    s = _aliasable_sample({'PE-A': 'CD45'}, cols=['PE-A'])
    assert ed._axis_alias_for_sample(ed, s, ['BV421-A']) == {}


def test_axis_alias_leaves_scatter_axes_alone():
    """Non-fluor axes (FSC-A/SSC-A) present in every sample → no alias,
    and a None axis (e.g. histogram y) is ignored."""
    ed = _editor_with_global_labels({'FSC-A': 'FSC-A'})
    s = _aliasable_sample({'FITC-A': 'CD11b'}, cols=['FSC-A', 'FITC-A'])
    assert ed._axis_alias_for_sample(ed, s, ['FSC-A', None]) == {}


# ── Clusters as selectable populations (#43) ──────────────────────────────────

def _cluster_editor(samples):
    """Editor stub for the cluster helpers. Instance methods need a real
    `self`, so bind them with MethodType (the staticmethod-attribute trick
    used elsewhere won't pass self to nested calls)."""
    stub = types.SimpleNamespace(
        _samples=samples,
        _sample_order=list(samples),
        _sample_gates={},
        _sample_gate_order={},
        _sample_gate_seq={},
        _cluster_labels={},
        _active_sample=None,
        _gate_id_seq=0,
        status_var=types.SimpleNamespace(set=lambda *a, **k: None),
        _refresh_gate_list=lambda *a, **k: None,
        _checkpoint=lambda *a, **k: None,
        LABEL_COLUMNS=V.LABEL_COLUMNS,
    )
    for fn in ('_sample_cluster_ids', '_cluster_display_name',
               '_next_gate_id_for', '_import_clusters',
               '_sample_label_values', '_import_label_populations'):
        setattr(stub, fn, types.MethodType(getattr(V, fn), stub))
    return stub


def _clustered_sample(cluster_vals):
    return types.SimpleNamespace(
        data=pd.DataFrame({'FITC-A': list(range(len(cluster_vals))),
                           'cluster': cluster_vals}),
        fluor_channels=['FITC-A'],
        channel_labels={'FITC-A': 'CD11b'})


def test_sample_cluster_ids_sorted_unique_ints():
    ed = _cluster_editor({'A': _clustered_sample([2, 0, 0, 1, 2])})
    assert ed._sample_cluster_ids('A') == [0, 1, 2]


def test_sample_cluster_ids_empty_when_unclustered():
    s = types.SimpleNamespace(data=pd.DataFrame({'FITC-A': [1, 2]}))
    ed = _cluster_editor({'A': s})
    assert ed._sample_cluster_ids('A') == []


def test_cluster_display_name_falls_back():
    ed = _cluster_editor({'A': _clustered_sample([0, 1])})
    assert ed._cluster_display_name('A', 0) == 'Cluster 0'
    ed._cluster_labels['A'] = {1: 'T cells'}
    assert ed._cluster_display_name('A', 1) == 'T cells'
    # JSON-restored sessions stringify the int key — the getter handles it.
    ed._cluster_labels['A'] = {'1': 'B cells'}
    assert ed._cluster_display_name('A', 1) == 'B cells'


def test_import_clusters_creates_population_per_label():
    ed = _cluster_editor({
        'A': _clustered_sample([0, 1, 2, 2]),
        'B': types.SimpleNamespace(data=pd.DataFrame({'FITC-A': [1, 2]})),
    })
    ed._cluster_labels['A'] = {0: 'Monocytes'}
    ed._import_clusters()

    gates = ed._sample_gates['A']
    clusters = [g for g in gates.values() if g.get('kind') == 'cluster']
    assert len(clusters) == 3
    assert {g['cluster_id'] for g in clusters} == {0, 1, 2}
    assert all(g['enabled'] is False for g in clusters)        # start hidden
    assert all(g['parent_id'] is None for g in clusters)       # root pops
    # Phenotype name carried through from _cluster_labels.
    by_id = {g['cluster_id']: g for g in clusters}
    assert by_id[0]['name'] == 'Monocytes'
    assert by_id[1]['name'] == 'Cluster 1'
    # Unclustered sample gets nothing.
    assert ed._sample_gates.get('B', {}) == {}
    assert len(ed._sample_gate_order['A']) == 3


def test_import_clusters_idempotent():
    ed = _cluster_editor({'A': _clustered_sample([0, 1])})
    ed._import_clusters()
    ed._import_clusters()           # second run must not duplicate
    clusters = [g for g in ed._sample_gates['A'].values()
                if g.get('kind') == 'cluster']
    assert len(clusters) == 2


# ── Generic label-column import (FlowSOM metaclusters etc.) ───────────────────

def _flowsom_sample(meta_vals):
    return types.SimpleNamespace(
        data=pd.DataFrame({'FITC-A': list(range(len(meta_vals))),
                           'flowsom_meta': meta_vals}),
        fluor_channels=['FITC-A'],
        channel_labels={'FITC-A': 'CD11b'})


def test_sample_label_values_skips_unassigned_sentinel():
    ed = _cluster_editor({'A': _flowsom_sample([0, 1, 2, -1, -1])})
    # -1 (unassigned) is dropped.
    assert ed._sample_label_values('A', 'flowsom_meta') == [0, 1, 2]


def test_import_label_populations_creates_category_gates():
    ed = _cluster_editor({'A': _flowsom_sample([0, 0, 1, 2, -1])})
    ed._import_label_populations('flowsom_meta')
    gates = list(ed._sample_gates['A'].values())
    cats = [g for g in gates if g.get('kind') == 'category']
    assert {g['value'] for g in cats} == {0, 1, 2}       # -1 excluded
    assert all(g['channel'] == 'flowsom_meta' for g in cats)
    assert all(g['enabled'] is False for g in cats)


def test_import_label_populations_idempotent():
    ed = _cluster_editor({'A': _flowsom_sample([0, 1])})
    ed._import_label_populations('flowsom_meta')
    ed._import_label_populations('flowsom_meta')
    cats = [g for g in ed._sample_gates['A'].values()
            if g.get('kind') == 'category']
    assert len(cats) == 2
