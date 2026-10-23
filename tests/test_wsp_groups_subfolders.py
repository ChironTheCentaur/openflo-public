"""A .wsp's sample groups become OpenFlo groups, and its FCS files are found
in subfolders (Complaints item 20).

Synthetic workspaces and FCS files only; the expected groups and paths are
fixed by how they were written.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import openflo.pipeline as fp
from openflo.compare import FcsLocator, _compare_one_sample, _per_sample_inventory

CH = ['FSC-A', 'SSC-A', 'FL1-A']


def _fcs(path, n=2000, seed=0, bright=0.3):
    import flowio
    rng = np.random.default_rng(seed)
    fl1 = np.where(rng.random(n) < bright, rng.normal(20000, 2000, n),
                   rng.normal(100, 50, n))
    ev = np.column_stack([rng.uniform(2e4, 2e5, n), rng.uniform(1e4, 1e5, n),
                          fl1]).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.ravel().tolist(), CH)
    return str(path)


def _gate():
    return [{'id': 'g1', 'parent_id': None, 'kind': 'threshold',
             'channel': 'FL1-A', 'value': 5000.0, 'name': 'FL1+'}]


def _wsp(path, samples):
    """`samples`: (name, fcs path, group or None). The workspace's uris point
    at a folder that does not exist (it was written on another computer);
    the files themselves sit under `path`'s folder."""
    w = fp.WspWriter()
    for name, fcs, group in samples:
        w.add_sample(name=name, fcs_path=fcs, channels=CH, gates=_gate(),
                     group=group)
    w.write(str(path))
    text = path.read_text(encoding='utf-8')
    for _name, fcs, _g in samples:
        moved = fp._wsp_dataset_uri(fcs)
        gone = moved.replace(os.path.dirname(fcs).replace(os.sep, '/'),
                             '/elsewhere/' + os.path.basename(
                                 os.path.dirname(fcs)))
        text = text.replace(moved, gone)
    path.write_text(text, encoding='utf-8')
    return str(path)


def test_writer_puts_groups_in_the_workspace_and_reader_maps_them(tmp_path):
    fcs = [_fcs(tmp_path / f's{i}.fcs', seed=i) for i in range(3)]
    wsp = _wsp(tmp_path / 'g.wsp', [('s0', fcs[0], 'Stim'),
                                    ('s1', fcs[1], 'Control'),
                                    ('s2', fcs[2], None)])
    groups = fp.WspReader(wsp).sample_groups()
    assert groups == {'1': ['Stim'], '2': ['Control']}


def test_flowjo_default_groups_are_not_user_groups(tmp_path):
    fcs = _fcs(tmp_path / 's.fcs')
    wsp = tmp_path / 'd.wsp'
    _wsp(wsp, [('s', fcs, 'Compensation')])
    text = wsp.read_text(encoding='utf-8')
    assert 'name="Compensation"' not in text       # not written either
    # FlowJo's own 'Compensation' group, hand-written, is not taken either.
    text = text.replace(
        '</Groups>',
        '<GroupNode name="Compensation"><Group name="Compensation">'
        '<SampleRefs><SampleRef sampleID="1"/></SampleRefs></Group>'
        '</GroupNode><GroupNode name="Day 2"><Group name="Day 2"><SampleRefs>'
        '<SampleRef sampleID="1"/></SampleRefs></Group></GroupNode></Groups>')
    wsp.write_text(text, encoding='utf-8')
    assert fp.WspReader(str(wsp)).sample_groups() == {'1': ['Day 2']}


def test_locator_finds_files_in_subfolders_and_never_guesses(tmp_path):
    a = _fcs(tmp_path / 'plate1' / 'well.fcs', seed=1)
    b = _fcs(tmp_path / 'plate2' / 'well.fcs', seed=2)
    c = _fcs(tmp_path / 'day3' / 'deep' / 'only.fcs', seed=3)
    loc = FcsLocator(str(tmp_path / 'x.wsp'))
    assert loc.find('file:/elsewhere/day3/deep/only.fcs') == c
    # Two files named well.fcs: the uri's folder name decides.
    assert loc.find('file:/elsewhere/plate2/well.fcs') == b
    assert loc.find('file:/elsewhere/plate1/well.fcs') == a
    # Nothing decides: not located, and named.
    assert loc.find('file:/elsewhere/other/well.fcs') is None
    assert loc.ambiguous == [('well.fcs', 2)]
    # The uri as written still wins when it exists.
    assert loc.find(fp._wsp_dataset_uri(a)) == a


def test_openflo_compare_counts_a_workspace_whose_files_moved_to_subfolders(
        tmp_path):
    fcs = _fcs(tmp_path / 'day1' / 's.fcs', seed=4, bright=0.25)
    wsp = _wsp(tmp_path / 's.wsp', [('s', fcs, None)])
    loc = FcsLocator(wsp)
    ((name, uri, pops),) = _per_sample_inventory(fp.WspReader(wsp),
                                                 locate=loc.find)
    found = loc.find(uri)
    assert found == fcs
    (row,) = _compare_one_sample(name, found, pops, wsp_path=wsp)
    want = int((fp.FlowSample(fcs).data['FL1-A'] >= 5000).sum())
    assert row['openflo_count'] == want


def test_editor_import_puts_samples_in_the_workspace_groups(tmp_path):
    """The editor's .wsp import finds the files in subfolders of the
    workspace's folder and puts each sample in its workspace group; a
    sample in no group keeps its folder's group."""
    sys.path.insert(0, os.path.dirname(__file__))
    from test_gui_smoke import _editor_or_skip
    root, ed, _gui = _editor_or_skip()
    try:
        fcs = [_fcs(tmp_path / 'run' / f'day{i}' / f's{i}.fcs', seed=i)
               for i in range(3)]
        wsp = _wsp(tmp_path / 'run' / 'g.wsp', [('s0', fcs[0], 'Stim'),
                                               ('s1', fcs[1], 'Stim'),
                                               ('s2', fcs[2], None)])
        queued = []
        ed._queue_fcs_loads = lambda paths, *a, **k: queued.extend(paths)
        ed._ingest_wsp(wsp)
        assert sorted(queued) == sorted(fcs)
        for path in queued:
            s = fp.FlowSample(path)
            s.apply_transform()
            ed._on_loaded(ed._sample_name_for(path), s)
        trials = {os.path.basename(ed._name_to_path[n]): ed._trial_for(n)
                  for n in ed._samples}
        from openflo.workspace import derive_trial_name
        assert trials == {'s0.fcs': 'Stim', 's1.fcs': 'Stim',
                          's2.fcs': derive_trial_name(fcs[2])}
        assert not ed._pending_wsp_trial
    finally:
        root.destroy()


@pytest.mark.parametrize('group', ['Stim', None])
def test_gui_export_writes_the_sample_group(tmp_path, group):
    fcs = _fcs(tmp_path / 's.fcs')
    w = fp.WspWriter()
    w.add_sample(name='s', fcs_path=fcs, channels=CH, gates=_gate(),
                 group=group)
    w.write(str(tmp_path / 'e.wsp'))
    got = fp.WspReader(str(tmp_path / 'e.wsp')).sample_groups()
    assert got == ({'1': [group]} if group else {})
