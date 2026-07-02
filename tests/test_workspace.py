"""Tests for the Pipeline Workspace.

Pure-logic helpers and the WorkspaceModel (groups, comp precedence, display)
run headlessly. A guarded smoke test drives the actual view / drop routing /
pop-out only if Tk can initialise, mirroring tests/test_gui_smoke.py.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from openflo.workspace import (
    DEFAULT_RUN_CFG,
    WorkspaceModel,
    build_drop_payload,
    build_run_units,
    comp_iid,
    comp_summary,
    default_run_cfg,
    derive_trial_name,
    extract_editor_context,
    extract_one_context,
    find_run_outputs,
    gate_chain,
    gate_iid,
    gate_is_leaf,
    gate_path,
    parse_iid,
    prepare_unit,
    proper_run_channels,
    resolve_run_sample,
    run_label,
    run_workspace_item,
    sample_iid,
)

# ── Fixtures ───────────────────────────────────────────────────────────────


def _make_gates():
    """g1(threshold,root) → g2(rect) → g3(threshold,leaf);  g1 → g4(leaf)."""
    return {
        'g1': {'kind': 'threshold', 'channel': 'CD11b', 'value': 1000.0,
               'id': 'g1', 'parent_id': None, 'color': '#111111'},
        'g2': {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
               'x0': 0.0, 'x1': 1.0, 'y0': 0.0, 'y1': 1.0,
               'id': 'g2', 'parent_id': 'g1', 'color': '#222222'},
        'g3': {'kind': 'threshold', 'channel': 'CD45', 'value': 500.0,
               'id': 'g3', 'parent_id': 'g2', 'color': '#333333'},
        'g4': {'kind': 'threshold', 'channel': 'CD34', 'value': 250.0,
               'id': 'g4', 'parent_id': 'g1', 'color': '#444444'},
    }


def _make_editor():
    """Single sample s1 with a comp matrix (used by the helper tests)."""
    comp = np.eye(3)
    sample = SimpleNamespace(name='s1', path=r'C:\data\s1.fcs',
                             comp_matrix=comp, comp_channels=['BV421-A', 'PE-A', 'APC-A'])
    return SimpleNamespace(
        _samples={'s1': sample},
        _sample_order=['s1'],
        _sample_trial={'s1': 'TrialX'},
        _sample_colors={'s1': '#1f77b4'},
        _sample_gates={'s1': _make_gates()},
        _sample_gate_order={'s1': ['g1', 'g2', 'g3', 'g4']},
        _channel_labels={'BV421-A': 'CD11b', 'PE-A': 'CD45'},
    ), comp


def _make_editor2():
    """Two samples: s1 has a comp matrix, s2 has none (needs beads)."""
    s1 = SimpleNamespace(name='s1', path=r'C:\exp\TrialX\blank\s1.fcs',
                         comp_matrix=np.eye(3), comp_channels=list('ABC'))
    s2 = SimpleNamespace(name='s2', path=r'C:\exp\TrialX\blank\s2.fcs',
                         comp_matrix=None, comp_channels=[])
    return SimpleNamespace(
        _samples={'s1': s1, 's2': s2},
        _sample_order=['s1', 's2'],
        _sample_trial={'s1': 'TrialX', 's2': 'TrialX'},
        _sample_colors={'s1': '#1f77b4', 's2': '#ff7f0e'},
        _sample_gates={'s1': {}, 's2': {}},
        _sample_gate_order={'s1': [], 's2': []},
        _channel_labels={},
    )


# ── Helpers ────────────────────────────────────────────────────────────────


def test_iid_roundtrip():
    assert parse_iid(sample_iid('s1')) == ('sample', 's1')
    assert parse_iid(comp_iid('s1')) == ('comp', 's1')
    assert parse_iid(gate_iid('s1', 'g3')) == ('gate', 's1', 'g3')
    assert parse_iid('bogus') is None


def test_derive_trial_name():
    p = os.path.join('root', 'MyTrial', 'blank experiment with X', 'f.fcs')
    assert derive_trial_name(p) == 'MyTrial'
    assert derive_trial_name(None) == 'Trial'
    assert derive_trial_name('') == 'Trial'


def test_derive_trial_name_day_grouping():
    """The 'Day N' token is found at whatever ancestor depth it sits, nearest
    wins, and is normalised to 'Day N' — so a whole experiment folder groups by
    collection day instead of collapsing under a shared grandparent."""
    j = os.path.join
    # Day token in the immediate experiment folder.
    assert derive_trial_name(
        j('root', 'MyStudy', '2024-01-15 Day 3 Check', 'x.fcs')) == 'Day 3'
    # Day token a grandparent up (host is a generic 'Blank Experiment with…').
    assert derive_trial_name(
        j('root', 'MyStudy', 'batch-1', 'day 0 raw',
          'Blank Experiment with Sample Tube', 'x.fcs')) == 'Day 0'
    # Glued 'day12' and odd-prefixed 'day 15' both normalise; nearest is used.
    assert derive_trial_name(
        j('root', 'day12 batch-a', 'MyStudy', '2024-02-01 Day 12',
          'x.fcs')) == 'Day 12'
    assert derive_trial_name(
        j('root', 'MyStudy', 'set2 day 15', 'x.fcs')) == 'Day 15'
    # No day token anywhere → fall back to the grandparent heuristic.
    assert derive_trial_name(
        j('root', 'MyTrial', 'blank experiment with X', 'f.fcs')) == 'MyTrial'
    # 'day' glued inside a word must NOT match.
    assert derive_trial_name(
        j('root', 'Monday', 'host', 'f.fcs')) == 'Monday'


def test_is_comp_sample():
    from openflo.workspace import is_comp_sample
    assert is_comp_sample('Compensation Controls_APC Stained Control_008')
    assert is_comp_sample('Compensation Controls_Unstained Control_006')
    assert is_comp_sample('PE-Cy7 Stained Control')
    assert is_comp_sample('Unstained')
    assert not is_comp_sample('Sample_sample_a_003')
    # An 'unstained' file counts as a comp even with a Specimen_ prefix.
    assert is_comp_sample('Sample_unstained_001')
    assert not is_comp_sample('sample_b')
    assert not is_comp_sample('')
    assert not is_comp_sample(None)


def test_trial_day_number():
    from openflo.workspace import trial_day_number
    assert trial_day_number('Day 0') == 0
    assert trial_day_number('Day 15') == 15
    assert trial_day_number('MyStudy') is None
    assert trial_day_number('') is None
    assert trial_day_number(None) is None


def test_sample_name_for_disambiguates_cross_day_collisions(tmp_path):
    """Identical filenames in different day folders must each get a unique,
    collision-free sample name (the second would otherwise be silently skipped
    as 'already loaded'). The same path always resolves to the same name."""
    from openflo.gui import ViewGateEditorWindow as GE

    class Reg:
        _sample_name_for = GE._sample_name_for
        def __init__(self):
            self._path_to_name = {}
            self._name_to_path = {}
            self._samples = {}

    r = Reg()
    p6 = tmp_path / '02-26 Day 6 Check' / 'Comp_APC_008.fcs'
    p9 = tmp_path / '03-05 Day 9 Check' / 'Comp_APC_008.fcs'
    p6.parent.mkdir(parents=True); p9.parent.mkdir(parents=True)
    p6.touch(); p9.touch()

    n6 = r._sample_name_for(str(p6))
    n9 = r._sample_name_for(str(p9))
    assert n6 != n9                       # no silent collision
    assert n6 == 'Comp_APC_008'           # first keeps the bare name
    assert n9 == 'Comp_APC_008 [Day 9]'   # second disambiguated by its day
    # Idempotent: same path → same name, no proliferation.
    assert r._sample_name_for(str(p9)) == n9
    assert len(r._name_to_path) == 2


def test_sample_name_for_case_insensitive_key(tmp_path):
    """On a case-insensitive filesystem the same file referenced with differing
    folder-name case (e.g. a .wsp's stored path vs the dropped folder) must map
    to ONE name, not load twice."""
    import os

    from openflo.gui import ViewGateEditorWindow as GE

    class Reg:
        _sample_name_for = GE._sample_name_for
        def __init__(self):
            self._path_to_name = {}
            self._name_to_path = {}
            self._samples = {}

    r = Reg()
    d = tmp_path / 'Day6 Check'
    d.mkdir()
    f = d / 'Comp.fcs'
    f.touch()
    lower = str(f).replace('Day6 Check', 'day6 check')
    n1 = r._sample_name_for(str(f))
    n2 = r._sample_name_for(lower)
    if os.path.normcase('A') == os.path.normcase('a'):   # Windows / mac
        assert n1 == n2                                   # same file, one name
        assert len(r._name_to_path) == 1
    else:                                                 # case-sensitive POSIX
        assert len(r._name_to_path) == 2


def test_expand_dropped_paths_groups_trials(tmp_path):
    """A folder drop is walked recursively into its .fcs / .wsp files, and
    each sample's trial is the grandparent folder — so a multi-trial parent
    folder lands each sample under its own group, while a single trial folder
    collapses to one group."""
    from openflo.gui import ViewGateEditorWindow as GE

    for trial in ('Trial_A', 'Trial_B'):
        host = tmp_path / trial / 'blank experiment with comp'
        host.mkdir(parents=True)
        (host / 'sampleA.fcs').touch()
        (host / 'sampleB.fcs').touch()
        (host / 'analysis.wsp').touch()
    (tmp_path / 'notes.txt').touch()

    # Whole parent folder → both trials, both workspaces, junk ignored.
    fcs, wsp = GE._expand_dropped_paths([str(tmp_path)])
    assert len(fcs) == 4 and len(wsp) == 2
    assert sorted({derive_trial_name(f) for f in fcs}) == ['Trial_A', 'Trial_B']

    # Single trial folder → one group.
    fcs1, wsp1 = GE._expand_dropped_paths([str(tmp_path / 'Trial_A')])
    assert len(fcs1) == 2 and len(wsp1) == 1
    assert {derive_trial_name(f) for f in fcs1} == {'Trial_A'}

    # A drop with nothing importable yields empty lists.
    assert GE._expand_dropped_paths([str(tmp_path / 'notes.txt')]) == ([], [])


def test_extract_context_trial_and_matrix():
    editor, comp = _make_editor()
    c = extract_one_context(editor, 's1')
    assert c['trial'] == 'TrialX' and c['comp_matrix'] is comp
    editor._sample_trial = {}
    assert extract_one_context(editor, 's1')['trial'] == derive_trial_name(r'C:\data\s1.fcs')
    assert extract_one_context(editor, 'missing') is None


def test_extract_context_deepcopies_gates():
    editor, _ = _make_editor()
    extract_editor_context(editor)[0]['gates']['g1']['value'] = 99999.0
    assert editor._sample_gates['s1']['g1']['value'] == 1000.0


def test_gate_helpers():
    g = _make_gates()
    assert gate_is_leaf(g, 'g3') and not gate_is_leaf(g, 'g1')
    assert gate_chain(g, 'g3') == ['g1', 'g2', 'g3']
    g['g1']['parent_id'] = 'g3'                       # cycle
    assert len(gate_chain(g, 'g3')) == len(set(gate_chain(g, 'g3')))
    assert ' / ' in gate_path(_make_gates(), 'g3')


def test_payload_leaf_carries_chain_trial_comp():
    editor, comp = _make_editor()
    p = build_drop_payload(extract_editor_context(editor)[0], 'g3')
    assert p['trial'] == 'TrialX' and p['gate_chain'] == ['g1', 'g2', 'g3']
    assert p['is_leaf'] is True and p['comp_matrix'] is comp


def test_comp_summary():
    assert comp_summary({'comp_matrix': np.eye(12)}) == '12×12'
    assert comp_summary({'comp_matrix': None}) == 'none'


# ── WorkspaceModel: groups, comp precedence, display ───────────────────────


def _pl(name, matrix=None):
    return {'sample': name, 'trial': 'T', 'gate_id': None, 'color': '#000000',
            'comp_matrix': matrix, 'comp_channels': []}


def test_model_groups_and_membership():
    m = WorkspaceModel()
    a = m.add_item(_pl('s1', np.eye(2)))             # loose
    g = m.new_group('Trial 1')
    b = m.add_item(_pl('s2', np.eye(2)), gid=g)      # in group
    assert m.item_group(a) is None and m.item_group(b) == g
    m.move_item(a, g)
    assert m.item_group(a) == g and a not in m.loose
    # removing a group keeps its items (back to loose)
    m.remove_group(g, keep_items=True)
    assert m.item_group(a) is None and m.item_group(b) is None


def test_model_group_selected():
    m = WorkspaceModel()
    a = m.add_item(_pl('s1', np.eye(2)))
    b = m.add_item(_pl('s2', np.eye(2)))
    g = m.group_selected([a, b], name='Both')
    assert set(m.groups[g]['items']) == {a, b} and not m.loose


def test_model_comp_precedence_and_validation():
    m = WorkspaceModel()
    a = m.add_item(_pl('with', np.eye(4)))           # has metadata matrix
    b = m.add_item(_pl('without', None))             # no matrix → needs beads
    assert m.comp_ready(a) and m.effective_comp(a)['label'] == '4×4'
    assert not m.comp_ready(b) and m.effective_comp(b)['kind'] == 'none'
    assert m.unready_items() == [b]
    # group beads cascade to the member that lacked a matrix
    g = m.new_group()
    m.move_item(b, g)
    m.set_group_comp(g, {'kind': 'beads', 'files': ['x.fcs', 'y.fcs']})
    assert m.comp_ready(b) and m.effective_comp(b)['label'] == 'beads:2'
    assert m.unready_items() == []
    # item-level beads OVERRIDE a present metadata matrix
    m.set_item_comp(a, {'kind': 'beads', 'files': ['z.fcs']})
    assert m.effective_comp(a)['kind'] == 'beads' and m.effective_comp(a)['label'] == 'beads:1'


def test_model_fmo_precedence():
    m = WorkspaceModel()
    g = m.new_group()
    a = m.add_item(_pl('s1', np.eye(2)), gid=g)
    assert m.effective_fmo(a) is None
    m.set_group_fmo(g, {'files': ['f1.fcs']})
    assert m.effective_fmo(a) == {'files': ['f1.fcs']}
    m.set_item_fmo(a, {'files': ['own.fcs']})        # item overrides group
    assert m.effective_fmo(a) == {'files': ['own.fcs']}


def test_model_display_group_and_item():
    m = WorkspaceModel()
    g = m.new_group()
    a = m.add_item(_pl('s1', np.eye(2)), gid=g)
    m.add_item(_pl('s2', np.eye(2)))                 # loose
    assert len(m.displayed_items()) == 2
    m.toggle_item_display(a)
    assert {it['sample'] for it in m.displayed_items()} == {'s2'}
    m.toggle_item_display(a)
    m.toggle_group_display(g)                         # group off hides its items
    assert {it['sample'] for it in m.displayed_items()} == {'s2'}
    m.set_all_display(False)
    assert m.displayed_items() == []


# ── Run engine (M2): pure helpers + per-item error isolation ───────────────


def test_run_label():
    assert run_label({'trial': 'T1', 'sample': 's1'}) == 'T1_s1'
    assert run_label({'trial': 'T 1', 'sample': 's/1', 'gate_id': 'g3'}) == 'T_1_s_1_g3'
    assert run_label({}) == 'Trial_sample'


def test_default_run_cfg_is_independent():
    cfg = default_run_cfg()
    assert set(cfg) == {'method', 'resolution', 'n_metaclusters', 'k',
                        'max_events', 'seed', 'reproducible', 'channels',
                        'umap', 'tsne', 'phate', 'trimap', 'pacmap',
                        'concatenate'}
    cfg['k'] = 999
    assert DEFAULT_RUN_CFG['k'] != 999          # returned a copy, not the original


def test_resolve_run_sample_reuses_live_without_mutating():
    import pandas as pd

    from openflo.pipeline import FlowSample
    df = pd.DataFrame({'FSC-A': [1.0, 2.0, 3.0], 'BV421-A': [10.0, 20.0, 30.0]})
    s = FlowSample.from_dataframe(df, name='s1')
    s.clusters = 'STALE'                         # must be reset by resolve
    editor = SimpleNamespace(_samples={'s1': s})
    item = {'sample': 's1', 'path': None, 'gate_id': None, 'gates': {}}

    sample, note = resolve_run_sample(editor, item)
    assert 'live' in note
    assert sample.clusters is None               # editor analysis state not carried
    sample.data.iloc[0, 0] = 999.0               # mutate the run copy …
    assert s.data.iloc[0, 0] == 1.0              # … editor sample is untouched


def test_resolve_run_sample_raises_when_unresolvable():
    editor = SimpleNamespace(_samples={})
    with pytest.raises(RuntimeError):
        resolve_run_sample(editor, {'sample': 'missing', 'path': None})


def test_run_item_isolates_errors(tmp_path):
    # Unresolvable sample → ok=False with an error string, never raises.
    editor = SimpleNamespace(_samples={})
    res = run_workspace_item(editor, {'sample': 'missing', 'path': None},
                             default_run_cfg(), str(tmp_path))
    assert res['ok'] is False and res['error'] and res['n_clusters'] == 0


def test_run_item_too_few_events(tmp_path):
    import pandas as pd

    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(pd.DataFrame({'BV421-A': [1.0, 2.0]}), name='s1')
    editor = SimpleNamespace(_samples={'s1': s})
    res = run_workspace_item(editor, {'sample': 's1', 'path': None, 'gate_id': None,
                                      'gates': {}}, default_run_cfg(), str(tmp_path))
    assert res['ok'] is False and 'too few events' in (res['error'] or '')


def test_proper_run_channels_drops_height_width():
    s = SimpleNamespace(fluor_channels=['CD3-A', 'CD3-H', 'CD3-W', 'CD4-A',
                                        'BV421 H', 'PE-W'])
    assert proper_run_channels(s) == ['CD3-A', 'CD4-A']
    # fallback: if dropping -H/-W would empty it, keep all fluor channels
    assert proper_run_channels(SimpleNamespace(fluor_channels=['M-H', 'M-W'])) == ['M-H', 'M-W']
    assert proper_run_channels(SimpleNamespace(fluor_channels=[])) == []


def test_run_uses_proper_channels_and_subsamples(tmp_path):
    import numpy as np
    import pandas as pd

    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    cols = {}
    for m in ('CD3', 'CD4', 'CD8'):
        base = rng.normal(size=300)
        cols[f'{m}-A'] = base + rng.normal(scale=0.1, size=300)
        cols[f'{m}-H'] = base * 0.9          # collinear height version (must be dropped)
        cols[f'{m}-W'] = rng.normal(size=300)
    s = FlowSample.from_dataframe(pd.DataFrame(cols), name='s1')
    editor = SimpleNamespace(_samples={'s1': s})
    cfg = {'k': 10, 'max_events': 100, 'seed': 42, 'umap': False, 'trimap': False}
    res = run_workspace_item(editor, {'sample': 's1', 'path': None, 'gate_id': None,
                                      'gates': {}}, cfg, str(tmp_path))
    assert res['ok'], res['error']
    assert res['channels'] == ['CD3-A', 'CD4-A', 'CD8-A']     # one column per marker
    assert res['n_events'] == 100                             # subsampled up front
    assert res['n_clusters'] >= 1


def test_run_item_honors_cancel(tmp_path):
    import pandas as pd

    from openflo.pipeline import FlowSample
    df = pd.DataFrame({f'M{i}-A': list(range(20)) for i in range(3)})
    s = FlowSample.from_dataframe(df, name='s1')
    editor = SimpleNamespace(_samples={'s1': s})
    res = run_workspace_item(
        editor, {'sample': 's1', 'path': None, 'gate_id': None, 'gates': {}},
        {'k': 5, 'max_events': 0, 'seed': 42, 'umap': False, 'trimap': False},
        str(tmp_path), should_cancel=lambda: True)
    assert res['cancelled'] is True and res['ok'] is False and res['n_clusters'] == 0


def test_run_item_flowsom_too_few_events_is_error(tmp_path):
    """A flowsom run below its ~100-event floor writes no label column; that
    must surface as ok=False, not a spurious 0-cluster 'success'."""
    import pandas as pd

    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    cols = {f'{m}-A': rng.normal(size=50) for m in ('CD3', 'CD4', 'CD8')}
    s = FlowSample.from_dataframe(pd.DataFrame(cols), name='s1')
    editor = SimpleNamespace(_samples={'s1': s})
    cfg = {'method': 'flowsom', 'k': 10, 'max_events': 100, 'seed': 42,
           'umap': False, 'trimap': False}
    res = run_workspace_item(editor, {'sample': 's1', 'path': None,
                                      'gate_id': None, 'gates': {}},
                             cfg, str(tmp_path))
    assert res['ok'] is False
    assert 'no labels' in (res['error'] or '')


def test_run_item_reproducible_phenograph_is_deterministic(tmp_path):
    """cfg['reproducible']=True threads through compute_run to cluster()'s seeded
    Leiden backend, so two batch runs give identical cluster labels."""
    import inspect

    import pandas as pd
    ph = pytest.importorskip('phenograph')
    if not {'clustering_algo', 'seed'} <= set(
            inspect.signature(ph.cluster).parameters):
        pytest.skip('phenograph build lacks seeded Leiden')
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 0.3, (150, 3)) for m in (0.0, 6.0, 12.0)])
    df = pd.DataFrame(X, columns=['CD3-A', 'CD4-A', 'CD8-A'])
    cfg = {'method': 'phenograph', 'k': 12, 'max_events': 0, 'seed': 42,
           'reproducible': True, 'umap': False, 'trimap': False}

    def run(sub):
        s = FlowSample.from_dataframe(df.copy(), name='s1')
        editor = SimpleNamespace(_samples={'s1': s})
        out = tmp_path / sub
        out.mkdir()
        res = run_workspace_item(editor, {'sample': 's1', 'path': None,
                                          'gate_id': None, 'gates': {}},
                                 cfg, str(out))
        assert res['ok'], res.get('error')
        return pd.read_csv(res['events'])['cluster'].to_numpy()

    a, b = run('a'), run('b')
    assert np.array_equal(a, b)              # reproducible across runs
    assert len(set(a.tolist())) >= 2        # actually recovered structure


def test_build_run_units_dedupes_colliding_labels():
    """Two groups whose names sanitise to the same label must get DISTINCT unit
    labels — otherwise their {label}_*.csv/.png silently overwrite each other in
    the shared run dir (and both are still counted 'ok')."""
    from openflo.workspace import build_run_units
    it = {'sample': 's', 'trial': 'T'}
    model = SimpleNamespace(
        groups={'g1': {'gid': 'g1', 'name': 'Day 1', 'items': {'m1': it}},
                'g2': {'gid': 'g2', 'name': 'Day/1', 'items': {'m2': it}}},
        loose={})
    labels = [u['label'] for u in build_run_units(model, default_run_cfg())]
    assert len(labels) == len(set(labels))                    # no collision
    assert 'Day_1' in labels                                  # both → Day_1 …
    assert any(lbl.startswith('Day_1__') for lbl in labels)   # … one suffixed


def test_build_run_units_grouping_and_concat():
    m = WorkspaceModel()
    g1 = m.new_group('G1')
    m.add_item(_pl('a'), gid=g1)
    m.add_item(_pl('b'), gid=g1)
    g2 = m.new_group('G2')
    m.add_item(_pl('c'), gid=g2)
    m.add_item(_pl('loose1'))                         # loose / ungrouped

    # per-group (default): G1 (2 samples) + G2 (1) + loose1 (1) = 3 units
    units = build_run_units(m, {'concatenate': False})
    assert len(units) == 3
    by_label = {u['label']: len(u['members']) for u in units}
    assert by_label.get('G1') == 2 and by_label.get('G2') == 1

    # concatenate: ONE unit holding all four samples
    cu = build_run_units(m, {'concatenate': True})
    assert len(cu) == 1 and len(cu[0]['members']) == 4


def test_prepare_unit_concatenates_and_tags():
    import pandas as pd

    from openflo.pipeline import FlowSample
    s1 = FlowSample.from_dataframe(pd.DataFrame({'CD3-A': [1., 2., 3.], 'CD4-A': [1., 2., 3.]}), name='s1')
    s2 = FlowSample.from_dataframe(pd.DataFrame({'CD3-A': [4., 5.], 'CD4-A': [4., 5.]}), name='s2')
    editor = SimpleNamespace(_samples={'s1': s1, 's2': s2})

    def _m(name, group):
        return {'item': {'sample': name, 'path': None, 'gate_id': None, 'gates': {}},
                'group': group, 'sample': name}

    prep = prepare_unit(editor, [_m('s1', 'G1'), _m('s2', 'G1')], 'G1',
                        {'max_events': 0, 'seed': 42})
    assert prep['n_events'] == 5                       # 3 + 2 concatenated
    assert '__group__' in prep['data'].columns and '__sample__' in prep['data'].columns
    assert prep['channels'] == ['CD3-A', 'CD4-A']      # source cols excluded
    assert prep['color_by'] == '__sample__'            # one group, two samples

    prep2 = prepare_unit(editor, [_m('s1', 'G1'), _m('s2', 'G2')], 'all',
                         {'max_events': 0, 'seed': 42})
    assert prep2['color_by'] == '__group__'            # spans two groups


def test_prepare_unit_honors_channel_selection():
    import pandas as pd

    from openflo.pipeline import FlowSample
    s1 = FlowSample.from_dataframe(
        pd.DataFrame({'CD3-A': [1., 2., 3.], 'CD4-A': [1., 2., 3.],
                      'CD8-A': [1., 2., 3.]}), name='s1')
    editor = SimpleNamespace(_samples={'s1': s1})
    members = [{'item': {'sample': 's1', 'path': None, 'gate_id': None,
                         'gates': {}}, 'group': None, 'sample': 's1'}]

    # explicit channel selection is honoured (intersected with the sample)
    prep = prepare_unit(editor, members, 's1',
                        {'channels': ['CD3-A', 'CD8-A'], 'max_events': 0,
                         'seed': 1})
    assert prep['channels'] == ['CD3-A', 'CD8-A']

    # a selection naming an absent channel falls back to proper markers
    prep2 = prepare_unit(editor, members, 's1',
                         {'channels': ['NOPE-A'], 'max_events': 0, 'seed': 1})
    assert prep2['channels'] == proper_run_channels(s1)


def test_merge_unit_cfg_group_override():
    from openflo.workspace import merge_unit_cfg
    base = {'method': 'phenograph', 'k': 30, 'resolution': 1.0}
    # group override replaces only known keys; base is not mutated
    merged = merge_unit_cfg(base, {'method': 'leiden', 'resolution': 2.5})
    assert merged == {'method': 'leiden', 'k': 30, 'resolution': 2.5}
    assert base['method'] == 'phenograph'                  # untouched
    # unknown keys are ignored; empty/None inherits the base
    assert merge_unit_cfg(base, {'bogus': 9})['k'] == 30
    assert merge_unit_cfg(base, None) == base
    assert merge_unit_cfg(base, {}) == base


def test_group_units_carry_gid_and_cfg_roundtrips():
    """Per-group overrides hinge on (a) units knowing their gid and (b) the
    group cfg surviving save/load — verify both."""
    import json

    m = WorkspaceModel()
    g1 = m.new_group('G1')
    m.add_item(_pl('a'), gid=g1)
    m.groups[g1]['cfg'] = {'method': 'leiden', 'resolution': 3.0}

    units = build_run_units(m, {'concatenate': False})
    gu = next(u for u in units if u['label'] == 'G1')
    assert gu['gid'] == g1                                  # unit knows its group

    m2 = WorkspaceModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert m2.groups[g1]['cfg'] == {'method': 'leiden', 'resolution': 3.0}


def test_to_dict_stamps_version():
    from openflo.workspace import WORKSPACE_FORMAT, WORKSPACE_SCHEMA
    d = WorkspaceModel('WS').to_dict()
    assert d['format'] == WORKSPACE_FORMAT
    assert d['schema'] == WORKSPACE_SCHEMA
    assert isinstance(d['app_version'], str) and d['app_version']


def test_newer_version_note(monkeypatch):
    from openflo import workspace as ws
    monkeypatch.setattr('openflo.update.current_version', lambda: '1.4.1')

    # same / older / absent → no alert
    assert ws.newer_version_note({'app_version': '1.4.1'}) is None
    assert ws.newer_version_note({'app_version': '1.0.0'}) is None
    assert ws.newer_version_note({}) is None

    # newer producing app version → suggests updating
    note = ws.newer_version_note({'app_version': '2.0.0'})
    assert note and '2.0.0' in note and 'updating OpenFlo' in note

    # newer on-disk schema → format warning
    note = ws.newer_version_note({'schema': ws.WORKSPACE_SCHEMA + 1})
    assert note and 'newer OpenFlo format' in note

    # recipe kind is reflected in the wording
    note = ws.newer_version_note({'app_version': '9.9.9'}, kind='recipe')
    assert note and note.startswith('This recipe')


def test_find_run_outputs(tmp_path):
    assert find_run_outputs(str(tmp_path / 'nope')) == []
    (tmp_path / 'T1_s1_clusters.csv').write_text('cluster,count,percent\n0,5,50\n1,5,50\n')
    (tmp_path / 'T1_s1_umap.png').write_bytes(b'x')
    (tmp_path / 'T1_s1_trimap.png').write_bytes(b'x')
    (tmp_path / 'T1_s2_umap.png').write_bytes(b'x')        # umap only, no csv
    recs = find_run_outputs(str(tmp_path))
    assert [r['label'] for r in recs] == ['T1_s1', 'T1_s2']
    r1 = recs[0]
    assert r1['csv'] and r1['umap'] and r1['trimap'] and r1['clusters'] == 2
    r2 = recs[1]
    assert r2['umap'] and r2['csv'] is None and r2['clusters'] is None


def test_model_persistence_roundtrip():
    import json

    m = WorkspaceModel('WS')
    m.run_cfg['k'] = 7
    a = m.add_item(_pl('s1', np.eye(3)))               # loose, has comp matrix
    g = m.new_group('Trial 1')
    m.add_item(_pl('s2', None), gid=g)
    m.set_group_comp(g, {'kind': 'beads', 'files': ['x.fcs']})
    m.toggle_item_display(a)                           # display -> False

    blob = json.dumps(m.to_dict())                     # must be JSON-serialisable
    m2 = WorkspaceModel.from_dict(json.loads(blob))

    assert m2.run_cfg['k'] == 7
    assert set(m2.groups) == set(m.groups)
    assert m2.groups[g]['comp'] == {'kind': 'beads', 'files': ['x.fcs']}
    ia = m2.item(a)
    assert ia is not None and ia['comp_matrix'].shape == (3, 3) and ia['display'] is False
    # id counters restored → a fresh add can't collide with loaded ids
    c = m2.add_item(_pl('s3', None))
    assert c != a and c not in m.loose


# ── Guarded Tk smoke ───────────────────────────────────────────────────────


def test_panel_smoke():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:
        pytest.skip(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    try:
        from openflo.workspace import WorkspacePanel
        editor = _make_editor2()
        panel = WorkspacePanel(root, editor=editor)
        v = panel._view
        assert v is not None

        # Add as populations (c1 / tree column).
        assert v._route_drop(editor, [('sample', 's1')], 'tree', None) is True
        [mid1] = list(v.model.loose)
        assert v.model.comp_ready(mid1)                       # s1 has a matrix

        # New group, drop s2 into it — s2 lacks a matrix → flagged unready.
        gid = v.model.new_group('Trial 1'); v._render()
        assert v._route_drop(editor, [('sample', 's2')], 'tree', ('group', gid)) is True
        mid2 = next(iter(v.model.groups[gid]['items']))
        assert not v.model.comp_ready(mid2)

        # Drop comp beads on its Comp column → overrides → ready.
        assert v._route_drop(editor, [('sample', 's1')], 'comp', ('item', mid2)) is True
        assert v.model.comp_ready(mid2)
        assert v.model.effective_comp(mid2)['kind'] == 'beads'

        # Drop FMOs on the group's FMO column → cascades to the member.
        assert v._route_drop(editor, [('sample', 's1')], 'fmo', ('group', gid)) is True
        assert v.model.effective_fmo(mid2) is not None

        # Display toggle through to displayed_items.
        before = len(v.model.displayed_items())
        v.model.toggle_group_display(gid)
        assert len(v.model.displayed_items()) < before

        # Panel-level drop delegates to a live view (coords route within it).
        seen = {}
        orig = v.drop_at
        v.drop_at = lambda ed, nodes, xr, yr: seen.setdefault('hit', True) or orig(ed, nodes, xr, yr)
        panel.drop_at(editor, [('sample', 's2')], 0, 0)
        assert seen.get('hit') is True
        v.drop_at = orig

        # Pop-out / re-dock preserves the model.
        total = len(list(v.model.all_items()))
        panel._toggle_popout()
        assert panel.popped_count() == 1 and panel._view is None
        panel._toggle_popout()
        assert panel.popped_count() == 0 and panel._view is not None
        assert len(list(panel.model.all_items())) == total

        # Results: no run dir → friendly status, no crash.
        panel.last_run_dir = None
        panel._open_results()
        assert 'No results' in panel.status_var.get()
        # Populated dir → opens a ResultsViewer over the found records.
        import tempfile
        rd = tempfile.mkdtemp(prefix='wsout_')
        with open(os.path.join(rd, 'T_s1_clusters.csv'), 'w') as fh:
            fh.write('cluster,count\n0,1\n')
        panel.last_run_dir = rd
        panel._open_results()
        assert len(find_run_outputs(rd)) == 1

        # Undo wiring: a workspace mutation fires the editor checkpoint hook,
        # and restore_model reverts to a snapshot (and rebuilds the view).
        fired = []
        panel._on_before_change = lambda: fired.append(1)
        before = panel.model.to_dict()
        n0 = len(panel.model.groups)
        panel._view._new_group()
        assert fired and len(panel.model.groups) == n0 + 1
        panel.restore_model(before)
        assert len(panel.model.groups) == n0 and panel._view is not None
        root.destroy()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_no_window_creationflags():
    """CREATE_NO_WINDOW is added unless an explicit new/detached console is
    requested; idempotent and order-independent."""
    from openflo.workspace import _no_window_creationflags as f
    CNW, NEW_CONSOLE, DETACHED = 0x08000000, 0x00000010, 0x00000008
    assert f(0) == CNW                       # default → window-less
    assert f(CNW) == CNW                     # idempotent
    assert f(0x40) == 0x40 | CNW             # unrelated flag preserved + CNW
    assert f(NEW_CONSOLE) == NEW_CONSOLE     # caller wants a console → untouched
    assert f(DETACHED) == DETACHED           # detached → untouched
    assert f(NEW_CONSOLE | 0x40) == NEW_CONSOLE | 0x40


def test_suppress_child_windows_idempotent():
    """Patches subprocess.Popen on Windows (once), no-op elsewhere."""
    import subprocess
    import sys

    from openflo.workspace import _suppress_child_windows
    before = subprocess.Popen
    try:
        _suppress_child_windows()
        if sys.platform == 'win32':
            assert getattr(subprocess.Popen, '_openflo_nowindow', False)
            patched = subprocess.Popen
            _suppress_child_windows()                 # second call must not re-wrap
            assert subprocess.Popen is patched
            assert subprocess.Popen.__base__ is before
        else:
            assert subprocess.Popen is before          # untouched off Windows
    finally:
        subprocess.Popen = before


def test_view_drag_move_between_groups_and_stats_drop():
    """The intra-tree move-drag reparents items between groups / to loose, and
    a drag that ends over a Statistics window hands the sample over."""
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:
        pytest.skip(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    try:
        from openflo.workspace import WorkspacePanel
        editor = _make_editor2()
        panel = WorkspacePanel(root, editor=editor)
        v = panel._view

        # Two loose items + an empty group.
        assert v._route_drop(editor, [('sample', 's1')], 'tree', None) is True
        assert v._route_drop(editor, [('sample', 's2')], 'tree', None) is True
        mids = list(v.model.loose)
        assert len(mids) == 2
        mid = mids[0]                          # the s1 item
        gid = v.model.new_group('G'); v._render()

        # Drag onto the group row → moves in.
        v.tree.selection_set(mid); v._press_row = mid
        v._move_drag_to(mid, gid)
        assert v.model.item_group(mid) == gid

        # Drag onto empty space ('') → back to loose.
        v.tree.selection_set(mid); v._press_row = mid
        v._move_drag_to(mid, '')
        assert v.model.item_group(mid) is None and mid in v.model.loose

        # Drag onto another item that lives in the group → joins that group.
        other = mids[1]
        v.model.move_item(other, gid); v._render()
        v.tree.selection_set(mid); v._press_row = mid
        v._move_drag_to(mid, other)
        assert v.model.item_group(mid) == gid

        # Drop onto an open Statistics window. A bare-sample item carries no
        # gate → consumed, but nothing handed over (stats is population-based).
        captured = {}
        editor._stats_window_under = lambda xr, yr: SimpleNamespace(
            add_targets=lambda targets, source: captured.update(
                targets=list(targets), source=source))
        v.tree.selection_set(mid); v._press_row = mid
        assert v._drop_to_stats(SimpleNamespace(x_root=0, y_root=0)) is True
        assert 'targets' not in captured

        # Give the item a gate → now it's a population stats accepts.
        v.model.item(mid)['gate_id'] = 'g1'
        v.tree.selection_set(mid); v._press_row = mid
        assert v._drop_to_stats(SimpleNamespace(x_root=0, y_root=0)) is True
        assert captured.get('source') == 'workspace'
        assert (v.model.item(mid)['sample'], 'g1') in captured.get('targets', [])
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_recipe_run_cfg_round_trip(tmp_path):
    """A saved recipe's run_cfg merges back onto the defaults (forward/back
    compatible), preserving method + embedding choices."""
    import json

    from openflo.workspace import default_run_cfg
    cfg = default_run_cfg()
    cfg.update(method='leiden', resolution=2.0, phate=True, pacmap=True, k=15)
    p = tmp_path / 'recipe.json'
    p.write_text(json.dumps({'kind': 'openflo-workspace-recipe',
                             'run_cfg': cfg}), encoding='utf-8')
    rc = json.loads(p.read_text(encoding='utf-8'))['run_cfg']
    merged = default_run_cfg()
    merged.update({k: v for k, v in rc.items() if k in merged})
    assert merged['method'] == 'leiden' and merged['resolution'] == 2.0
    assert merged['phate'] and merged['pacmap'] and merged['k'] == 15


def test_compute_run_writes_importable_events_csv(tmp_path):
    """A run writes a per-event CSV (markers + cluster + embedding coords) that
    find_run_outputs surfaces — the basis of 'Load in editor' and batch-folder."""
    import io
    import sys

    import pandas as pd

    from openflo.workspace import compute_run, default_run_cfg, find_run_outputs
    rng = np.random.RandomState(0)
    df = pd.DataFrame({'CD3': rng.normal(0, 1, 200),
                       'CD4': rng.normal(0, 1, 200),
                       'CD8': rng.normal(0, 1, 200)})
    prep = {'data': df, 'channels': ['CD3', 'CD4', 'CD8'], 'label': 'b1',
            'n_events': 200, 'note': '', 'color_by': None}
    cfg = default_run_cfg()
    cfg.update(method='phenograph', umap=True, tsne=False, phate=False,
               trimap=False, pacmap=False, max_events=0)
    _so = sys.stdout
    sys.stdout = io.StringIO()                       # silence cluster chatter
    try:
        r = compute_run(dict(prep), cfg, str(tmp_path))
    finally:
        sys.stdout = _so
    assert r['ok'] and r.get('events')
    ev = pd.read_csv(r['events'])
    assert {'cluster', 'UMAP1', 'UMAP2', 'CD3'} <= set(ev.columns)
    recs = find_run_outputs(str(tmp_path))
    assert recs and recs[0].get('events')
