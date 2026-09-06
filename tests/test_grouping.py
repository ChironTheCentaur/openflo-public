"""Run-spec grouping / per-sample FMO normalization (cli._normalise_groups).

Chunk A: per-sample FMO assignment. A `samples` entry may be a plain
string (inherits the group's default fmo_set) or a dict
{'name', 'fmo_set'} to override just that sample. Compensation +
antibody labels are resolved per sample from each FCS ($SPILL / $PnS)
in the pipeline, so they're not part of the group spec.
"""
import os

import openflo.cli as cli


def test_plain_string_samples_inherit_group_fmo():
    g = cli._normalise_groups(
        [{'name': 'Day1', 'samples': ['m1', 'm2'], 'fmo_set': 'Early'}])
    assert len(g) == 1
    assert g[0]['samples'] == ['m1', 'm2']
    assert g[0]['fmo_set'] == 'Early'
    assert g[0]['sample_fmo'] == {'m1': 'Early', 'm2': 'Early'}


def test_per_sample_override():
    g = cli._normalise_groups([{
        'name': 'D', 'fmo_set': 'Early',
        'samples': ['m1', {'name': 'm2', 'fmo_set': 'Late'}],
    }])
    # m1 inherits the group default; m2 overrides to Late.
    assert g[0]['samples'] == ['m1', 'm2']
    assert g[0]['sample_fmo'] == {'m1': 'Early', 'm2': 'Late'}


def test_dict_sample_without_fmo_inherits_group():
    g = cli._normalise_groups([{
        'name': 'D', 'fmo_set': 'Standard',
        'samples': [{'name': 'm1'}, {'name': 'm2', 'fmo_set': ''}],
    }])
    # An empty/absent per-sample fmo_set falls back to the group default.
    assert g[0]['sample_fmo'] == {'m1': 'Standard', 'm2': 'Standard'}


def test_csv_string_samples():
    g = cli._normalise_groups([{'name': 'D', 'samples': 'a, b ,c', 'fmo_set': ''}])
    assert g[0]['samples'] == ['a', 'b', 'c']
    assert g[0]['sample_fmo'] == {'a': '', 'b': '', 'c': ''}


def test_empty_group_dropped():
    assert cli._normalise_groups([{'name': '', 'samples': ['x']}]) == []
    assert cli._normalise_groups([{'name': 'D', 'samples': []}]) == []
    assert cli._normalise_groups([]) == []


def test_non_dict_entries_skipped():
    g = cli._normalise_groups(['nonsense', {'name': 'D', 'samples': ['x']}])
    assert len(g) == 1
    assert g[0]['name'] == 'D'


def test_no_fmo_set_defaults_empty_string():
    g = cli._normalise_groups([{'name': 'D', 'samples': ['x']}])
    assert g[0]['fmo_set'] == ''
    assert g[0]['sample_fmo'] == {'x': ''}


# ── By-day auto-grouping (cli._discover_day_folders / _auto_groups_by_day) ───
# Discovery only checks the .fcs extension, so empty placeholder files
# are enough to exercise it.

def _touch(path):
    with open(path, 'w') as f:
        f.write('')


def test_discover_day_folders_direct(tmp_path):
    d = tmp_path / 'day0'
    d.mkdir()
    _touch(d / 'a.fcs')
    _touch(d / 'b.fcs')
    found = cli._discover_day_folders([str(tmp_path)])
    assert str(d) in [os.path.normpath(p) for p in found]


def test_discover_day_folders_parent_with_subfolders(tmp_path):
    """Point at a PARENT; each sub-folder with FCS is discovered."""
    for sub in ('day0', 'day3', 'day7'):
        s = tmp_path / sub
        s.mkdir()
        _touch(s / 'm1.fcs')
    found = cli._discover_day_folders([str(tmp_path)])
    bases = sorted(os.path.basename(p) for p in found)
    assert bases == ['day0', 'day3', 'day7']


def test_discover_day_folders_ignores_empty_dirs(tmp_path):
    (tmp_path / 'empty').mkdir()
    withfcs = tmp_path / 'has'
    withfcs.mkdir()
    _touch(withfcs / 'x.fcs')
    found = [os.path.basename(p) for p in cli._discover_day_folders([str(tmp_path)])]
    assert 'has' in found
    assert 'empty' not in found


def test_auto_groups_by_day_names_and_samples(tmp_path):
    for sub, files in [('Day 0 baseline', ['sample_1.fcs', 'sample_2.fcs']),
                       ('batch2 Day 3 recheck', ['x.fcs'])]:
        s = tmp_path / sub
        s.mkdir()
        for fn in files:
            _touch(s / fn)
    groups = cli._auto_groups_by_day([str(tmp_path)])
    by_name = {g['name']: g for g in groups}
    # Day token parsed into a tidy 'Day N' name.
    assert 'Day 0' in by_name
    assert 'Day 3' in by_name
    # Samples are the file stems; each group tagged with its folder.
    assert sorted(by_name['Day 0']['samples']) == ['sample_1', 'sample_2']
    assert by_name['Day 0']['trial_dir'].endswith('Day 0 baseline')
    assert by_name['Day 0']['fmo_set'] == ''     # auto / ungated default


def test_auto_groups_by_day_disambiguates_duplicate_day_names(tmp_path):
    # Two different parents each with a 'day3' folder → unique group names.
    for parent in ('exptA', 'exptB'):
        s = tmp_path / parent / 'day3'
        s.mkdir(parents=True)
        _touch(s / 'm1.fcs')
    groups = cli._auto_groups_by_day([str(tmp_path)])
    names = [g['name'] for g in groups]
    assert len(names) == 2
    assert len(set(names)) == 2, f"names must be unique, got {names}"


def test_auto_groups_by_day_empty_when_no_fcs(tmp_path):
    (tmp_path / 'nothing').mkdir()
    assert cli._auto_groups_by_day([str(tmp_path)]) == []


def test_normalise_groups_preserves_trial_dir():
    g = cli._normalise_groups(
        [{'name': 'Day 0', 'samples': ['m1'], 'fmo_set': '',
          'trial_dir': '/data/day0'}])
    assert g[0]['trial_dir'] == '/data/day0'


def test_build_export_gate_list_child_before_parent():
    # A child gate listed BEFORE its parent must still nest under the parent's
    # remapped id (two-pass allocation), not get silently re-rooted at None.
    gates = [
        {'kind': 'polygon', 'id': 'child', 'parent_id': 'parent', 'channel': 'x'},
        {'kind': 'rect', 'id': 'parent', 'parent_id': None, 'channel': 'y'},
    ]
    out = cli._build_export_gate_list(gates, {})
    child = next(g for g in out if g['kind'] == 'polygon')
    parent = next(g for g in out if g['kind'] == 'rect')
    assert child['parent_id'] == parent['id']       # nested under the parent
    assert parent['parent_id'] is None


def test_parse_labels():
    assert cli.parse_labels(' BV421-A = CD11b ; APC-A=CD34 ; junk ') == {
        'BV421-A': 'CD11b', 'APC-A': 'CD34'}
    assert cli.parse_labels('') == {}
    assert cli.parse_labels('X=Y') == {'X': 'Y'}


def test_parse_gates_arg():
    overrides, regions = cli._parse_gates_arg(
        '[{"kind":"threshold","channel":"CD4","value":"3.5"},'
        '{"kind":"rect","x_channel":"a","y_channel":"b",'
        '"x_min":0,"x_max":1,"y_min":0,"y_max":1},'
        '{"kind":"bogus"}]')
    assert overrides == {'CD4': 3.5}                     # threshold → override
    assert [g['kind'] for g in regions] == ['rect']      # rect → region list
    assert cli._parse_gates_arg('') == ({}, [])
    assert cli._parse_gates_arg('not json') == ({}, [])
    assert cli._parse_gates_arg('{"kind":"threshold"}') == ({}, [])   # dict, not list


def test_calibrate_bead_micron(tmp_path):
    import numpy as np
    import pandas as pd

    from openflo.fcs_export import write_fcs
    from openflo.pipeline import FlowSample
    df = pd.DataFrame({'FSC-A': np.full(400, 40000.0),
                       'SSC-A': np.full(400, 20000.0),
                       'CD3-A': np.linspace(1.0, 1000.0, 400)})
    write_fcs(df, str(tmp_path / 'bead.fcs'))
    s = FlowSample(str(tmp_path / 'bead.fcs'))
    fsc = next(c for c in s.data.columns
               if c.upper().startswith('FSC') and c.upper().endswith('-A'))
    med = float(np.median(s.data[fsc]))
    factor, path = cli._calibrate_bead_micron(
        str(tmp_path), {'S': {'b': 'bead'}}, bead_um=8.0)
    assert path is not None and abs(factor - 8.0 / med) < 1e-6
    # a <100-event control yields no calibration
    sub = tmp_path / 'sub'
    sub.mkdir()
    write_fcs(df.iloc[:50], str(sub / 'few.fcs'))
    f2, p2 = cli._calibrate_bead_micron(str(sub), {'S': {'b': 'few'}}, bead_um=8.0)
    assert f2 is None and p2 is None


def test_gates_thresholds_reach_ungated_groups():
    # --gates threshold overrides must apply to samples with an EMPTY fmo_set
    # (the norm in by-day grouping) — not just named FMO sets. Ungated groups do
    # no FCS reads, so this needs no real data.
    g = [{'name': 'G', 'fmo_set': '', 'samples': ['m1'], 'sample_fmo': {}}]
    fmo = cli._build_fmo_thresholds_for_groups(
        '/nonexistent', g, fmo_sets={}, fmo_percentile=99.0,
        gate_overrides={'CD4': 3.5}, report=lambda *a, **k: None)
    assert fmo.get('') == {'CD4': 3.5}       # ungated samples still get the gate
    # No overrides → no phantom '' entry (ungated stays ungated).
    fmo2 = cli._build_fmo_thresholds_for_groups(
        '/nonexistent', g, fmo_sets={}, fmo_percentile=99.0,
        gate_overrides={}, report=lambda *a, **k: None)
    assert '' not in fmo2
