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
        PHASE_COLORS=V.PHASE_COLORS,
    )
    for fn in ('_sample_cluster_ids', '_cluster_display_name',
               '_next_gate_id_for', '_import_clusters',
               '_sample_label_values', '_import_label_populations',
               '_import_cell_cycle'):
        setattr(stub, fn, types.MethodType(getattr(V, fn), stub))
    return stub


def _clustered_sample(cluster_vals):
    return types.SimpleNamespace(
        data=pd.DataFrame({'FITC-A': list(range(len(cluster_vals))),
                           'cluster': cluster_vals}),
        fluor_channels=['FITC-A'],
        channel_labels={'FITC-A': 'CD11b'})


class _Var:
    def __init__(self, value): self._v = value
    def get(self): return self._v


def _backgate_editor(n_rows, max_points):
    """Stub wired enough to exercise _overlay_backgate on a real Agg axes.
    One sample, one category gate selecting every event."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = pd.DataFrame({'x': range(n_rows), 'y': range(n_rows),
                       'grp': ['A'] * n_rows})
    sample = types.SimpleNamespace(data=df, fluor_channels=['x', 'y'],
                                   channel_labels={})
    fig, ax = plt.subplots()
    stub = types.SimpleNamespace(
        _samples={'A': sample},
        _sample_order=['A'],
        _sample_plot_enabled={'A': True},
        _sample_gates={'A': {'g1': {'kind': 'category', 'channel': 'grp',
                                    'value': 'A', 'parent_id': None,
                                    'name': 'all', 'enabled': False}}},
        _backgate=[('A', 'g1')],
        _gate_density_full=set(),               # empty = scaled (default)
        _backgate_hidden=set(),                 # empty = all shown
        _backgate_legend_pick={},
        _backgate_legend_anchor=(0.020, 0.965),
        _backgate_legend_collapsed=False,
        _backgate_legend_bbox=None,
        _backgate_legend_header=None,
        _backgate_legend_artists=[],
        _backgate_legend_rows=[],
        _legend_drag=None,
        _BACKGATE_COLORS=V._BACKGATE_COLORS,
        max_points_var=_Var(str(max_points)),
        ds_display_var=_Var(True),
        ds_propagate_var=_Var(False),
        ax=ax,
        canvas=types.SimpleNamespace(draw_idle=lambda: None),
        _population_path=V._population_path,        # staticmethod
        _in_box=V._in_box,                          # staticmethod
        _schedule_replot=lambda *a, **k: None,
        _refresh_gate_list=lambda *a, **k: None,
    )
    for fn in ('_overlay_backgate', '_draw_backgate_legend', '_backgate_color',
               '_reposition_backgate_legend', '_event_axes_frac',
               '_legend_press', '_sample_display_count', '_display_point_cap',
               '_smallest_loaded_sample_size', '_axis_alias_for_sample',
               '_autoclean_overrides', '_on_canvas_pick'):
        setattr(stub, fn, types.MethodType(getattr(V, fn), stub))
    return stub, ax


def _scatter_point_count(ax):
    return sum(len(c.get_offsets()) for c in ax.collections)


class _FakeTree:
    """Minimal ttk.Treeview stand-in for the expand/collapse-all logic."""
    def __init__(self, nodes, roots):
        self._n = nodes            # iid -> {'open': bool, 'children': [iid]}
        self._roots = roots
        self.heading_text = ''

    def get_children(self, parent=''):
        return self._roots if parent == '' else self._n[parent]['children']

    def item(self, iid, option=None, **kw):
        if 'open' in kw:
            self._n[iid]['open'] = bool(kw['open'])
            return None
        if option == 'open':
            return self._n[iid]['open']
        return None

    def heading(self, _col, text=None, **_kw):
        if text is not None:
            self.heading_text = text


def _expand_editor(all_open):
    nodes = {
        't': {'open': all_open, 'children': ['sg']},
        'sg': {'open': all_open, 'children': ['s']},
        's': {'open': True, 'children': ['grp']},
        'grp': {'open': all_open, 'children': []},
    }
    stub = types.SimpleNamespace(
        _sample_gates={'A': {'grp': {'kind': 'group', 'open': all_open},
                             'g1': {'kind': 'cluster', 'open': all_open}}},
        gate_tv=_FakeTree(nodes, roots=['t']),
        _refresh_gate_list=lambda *a, **k: None,
    )
    for fn in ('_toggle_expand_all', '_set_all_expanded'):
        setattr(stub, fn, types.MethodType(getattr(V, fn), stub))
    return stub


def test_set_all_expanded_writes_open_state_and_heading():
    ed = _expand_editor(all_open=True)
    ed._set_all_expanded(False)
    assert all(g['open'] is False for g in ed._sample_gates['A'].values())
    assert ed.gate_tv.heading_text.startswith('▸')
    ed._set_all_expanded(True)
    assert all(g['open'] is True for g in ed._sample_gates['A'].values())
    assert ed.gate_tv.heading_text.startswith('▾')


def test_toggle_expand_all_expands_when_anything_collapsed():
    ed = _expand_editor(all_open=False)        # group collapsed
    ed._toggle_expand_all()
    assert all(g['open'] is True for g in ed._sample_gates['A'].values())


def test_toggle_expand_all_collapses_when_all_open():
    ed = _expand_editor(all_open=True)
    ed._toggle_expand_all()
    assert all(g['open'] is False for g in ed._sample_gates['A'].values())


def test_backgate_default_matches_display_density():
    # Default (not in _gate_density_full) = scaled: cloud shows 1000 of 4000
    # (25%) → overlay thins to the same rate.
    ed, ax = _backgate_editor(4000, max_points=1000)
    ed._overlay_backgate(ed._samples, 'x', 'y')
    assert _scatter_point_count(ax) == 1000


def test_backgate_full_density_draws_all():
    # Opting a population OUT (☐ in the density column) draws it at full
    # density, under the 60k render guard.
    ed, ax = _backgate_editor(4000, max_points=1000)
    ed._gate_density_full = {('A', 'g1')}
    ed._overlay_backgate(ed._samples, 'x', 'y')
    assert _scatter_point_count(ax) == 4000


def _legend_artist(ed, action, target):
    """The pickable legend glyph for (action, target)."""
    return next(a for a, v in ed._backgate_legend_pick.items()
                if v == (action, target))


def test_backgate_legend_has_clickable_controls_per_target():
    """Each backgate gets pickable colour / on-off / density glyphs mapped to
    the right action+target."""
    ed, ax = _backgate_editor(4000, max_points=1000)
    ed._overlay_backgate(ed._samples, 'x', 'y')
    vals = set(ed._backgate_legend_pick.values())
    assert ('color', ('A', 'g1')) in vals
    assert ('toggle', ('A', 'g1')) in vals
    assert ('density', ('A', 'g1')) in vals
    assert all(t.get_picker() for t in ed._backgate_legend_pick)


def test_backgate_legend_click_toggles_density():
    """Clicking the density glyph flips that target between scaled and full."""
    ed, ax = _backgate_editor(4000, max_points=1000)
    ed._overlay_backgate(ed._samples, 'x', 'y')
    ev = types.SimpleNamespace(artist=_legend_artist(ed, 'density', ('A', 'g1')))
    ed._on_canvas_pick(ev)
    assert ('A', 'g1') in ed._gate_density_full        # scaled → full
    ed._on_canvas_pick(ev)
    assert ('A', 'g1') not in ed._gate_density_full    # full → scaled


def _evt_at_frac(ax, fx, fy, button=1):
    px, py = ax.transAxes.transform((fx, fy))
    return types.SimpleNamespace(x=px, y=py, button=button)


def test_backgate_legend_records_bbox_and_header():
    ed, ax = _backgate_editor(4000, max_points=1000)
    ed._overlay_backgate(ed._samples, 'x', 'y')
    assert ed._backgate_legend_bbox is not None
    assert ed._backgate_legend_header is not None
    x0, y0, x1, y1 = ed._backgate_legend_bbox
    assert x0 < x1 and y0 < y1


def test_legend_press_blocks_plot_click_and_starts_drag():
    ed, ax = _backgate_editor(4000, max_points=1000)
    ed._overlay_backgate(ed._samples, 'x', 'y')
    bx0, by0, bx1, by1 = ed._backgate_legend_bbox
    # A point in the legend body (below the header) blocks the plot click but
    # doesn't start a drag.
    body = _evt_at_frac(ax, (bx0 + bx1) / 2, by0 + (by1 - by0) * 0.15)
    assert ed._legend_press(body) is True
    assert ed._legend_drag is None
    # A point on the header strip starts a drag.
    hx0, hy0, hx1, hy1 = ed._backgate_legend_header
    head = _evt_at_frac(ax, (hx0 + hx1) / 2, (hy0 + hy1) / 2)
    assert ed._legend_press(head) is True
    assert ed._legend_drag is not None


def test_legend_press_ignores_clicks_outside():
    ed, ax = _backgate_editor(4000, max_points=1000)
    ed._overlay_backgate(ed._samples, 'x', 'y')
    far = _evt_at_frac(ax, 0.95, 0.05)        # bottom-right, away from legend
    assert ed._legend_press(far) is False


def test_backgate_legend_collapse_toggles():
    ed, ax = _backgate_editor(4000, max_points=1000)
    ed._overlay_backgate(ed._samples, 'x', 'y')
    col = _legend_artist(ed, 'collapse', None)
    ed._on_canvas_pick(types.SimpleNamespace(artist=col))
    assert ed._backgate_legend_collapsed is True
    # Collapsed: only the collapse glyph remains pickable (no row controls).
    assert set(ed._backgate_legend_pick.values()) == {('collapse', None)}


def test_backgate_legend_toggle_hides_overlay():
    """Clicking the on/off glyph hides the backgate: it stays in the legend
    (so it can be re-enabled) but draws no points."""
    ed, ax = _backgate_editor(4000, max_points=1000)
    ed._overlay_backgate(ed._samples, 'x', 'y')
    assert _scatter_point_count(ax) > 0
    ev = types.SimpleNamespace(artist=_legend_artist(ed, 'toggle', ('A', 'g1')))
    ed._on_canvas_pick(ev)
    assert ('A', 'g1') in ed._backgate_hidden
    ax.clear()
    ed._overlay_backgate(ed._samples, 'x', 'y')
    assert _scatter_point_count(ax) == 0               # hidden → nothing drawn
    assert ed._backgate_legend_pick                    # but still listed


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


def _cell_cycle_sample(phases):
    return types.SimpleNamespace(
        data=pd.DataFrame({'DAPI-A': list(range(len(phases))),
                           'cell_cycle': phases}),
        fluor_channels=['DAPI-A'], channel_labels={})


def test_import_cell_cycle_nests_under_collapsed_group():
    """Cell-cycle phases nest under ONE collapsed 'group' container, the
    same UX as clusters / auto-clean — not flat at the root."""
    ed = _cluster_editor({'A': _cell_cycle_sample(['G1', 'G1', 'S', 'G2M'])})
    ed._active_sample = 'A'
    ed._import_cell_cycle('A')

    gates = ed._sample_gates['A']
    grp = [gid for gid, g in gates.items()
           if g.get('kind') == 'group' and g.get('group_for') == 'cell_cycle']
    assert len(grp) == 1 and gates[grp[0]].get('open') is False
    cats = [g for g in gates.values()
            if g.get('kind') == 'category' and g.get('channel') == 'cell_cycle']
    assert cats and all(g['parent_id'] == grp[0] for g in cats)
    # Group label carries the phase count.
    assert gates[grp[0]]['name'].endswith(f'({len(cats)})')


def test_import_cell_cycle_idempotent():
    """A second import doesn't duplicate the group or the phases."""
    ed = _cluster_editor({'A': _cell_cycle_sample(['G1', 'S', 'G2M'])})
    ed._active_sample = 'A'
    ed._import_cell_cycle('A')
    ed._import_cell_cycle('A')
    gates = ed._sample_gates['A']
    grps = [g for g in gates.values()
            if g.get('kind') == 'group' and g.get('group_for') == 'cell_cycle']
    cats = [g for g in gates.values()
            if g.get('kind') == 'category' and g.get('channel') == 'cell_cycle']
    assert len(grps) == 1
    assert len(cats) == 3


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
    # Populations nest under ONE collapsed 'group' container (like auto-clean).
    grp = [gid for gid, g in gates.items()
           if g.get('kind') == 'group' and g.get('group_for') == 'cluster']
    assert len(grp) == 1 and gates[grp[0]].get('open') is False
    assert all(g['parent_id'] == grp[0] for g in clusters)     # nested, not root
    # Phenotype name carried through from _cluster_labels.
    by_id = {g['cluster_id']: g for g in clusters}
    assert by_id[0]['name'] == 'Monocytes'
    assert by_id[1]['name'] == 'Cluster 1'
    # Unclustered sample gets nothing.
    assert ed._sample_gates.get('B', {}) == {}
    assert len(ed._sample_gate_order['A']) == 4               # group + 3 clusters


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


def test_import_label_populations_nests_under_collapsed_group():
    """Cluster populations land under ONE collapsed 'group' container (not flat
    roots), so the tree stays condensed by default and expands the subgroup."""
    ed = _cluster_editor({'A': _flowsom_sample([0, 0, 1, 2, -1])})
    ed._import_label_populations('flowsom_meta')
    gates = ed._sample_gates['A']
    groups = [g for g in gates.values() if g.get('kind') == 'group']
    assert len(groups) == 1                         # one container
    grp = groups[0]
    assert grp.get('open') is False                  # collapsed by default
    assert grp.get('group_for') == 'flowsom_meta'
    grp_id = next(gid for gid, g in gates.items() if g is grp)
    cats = [g for g in gates.values() if g.get('kind') == 'category']
    assert cats and all(g['parent_id'] == grp_id for g in cats)  # all nested
    # re-import doesn't make a second group
    ed._import_label_populations('flowsom_meta')
    assert sum(1 for g in gates.values() if g.get('kind') == 'group') == 1
