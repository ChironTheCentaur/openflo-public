"""The FlowJo .wsp export writes real event counts, never a fake 0.

The sample node carries the FCS file's event total ($TOT, what FlowJo
records and what a re-import checks the file against; the sample's own count
when the file gives none), and each population the count Statistics shows.
A population that cannot be computed (a gate on a channel the sample lacks)
carries no count at all rather than a 0 nobody measured, and so does every
population of a sample whose loaded events are only part of its file (QC,
downsampling): FlowJo counts the whole file.

Generated data, synthetic sample names and generic channels only.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from tests.test_gui_ux import _editor_or_skip, _load_sample


def _nodes(path):
    """{sample name: (SampleNode count, {population name: count})}."""
    out = {}
    for sample in ET.parse(path).getroot().iter('Sample'):
        sn = sample.find('SampleNode')
        pops = {p.get('name'): p.get('count') for p in sn.iter()
                if p.tag in ('Population', 'AndNode', 'OrNode', 'NotNode')}
        out[sn.get('name')] = (sn.get('count'), pops)
    return out


def test_gui_wsp_export_writes_the_editors_counts_and_no_fake_zero(tmp_path):
    import tkinter.filedialog as fd
    root, ed = _editor_or_skip()
    saved = fd.asksaveasfilename
    try:
        out = tmp_path / 'counts.wsp'
        fd.asksaveasfilename = lambda *a, **k: str(out)
        ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
            on_done(work()) if on_done else work())
        s = _load_sample(ed, 'sample_a', cols=('FSC-A', 'SSC-A', 'FL1-A',
                                               'FL2-A'), n=200)
        rng = np.random.default_rng(4)
        s.data['FL1-A'] = rng.uniform(0, 1, 200)
        s.data['FL2-A'] = rng.uniform(0, 1, 200)
        ed._sample_gates['sample_a'] = {
            'g1': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.4,
                   'op': '>=', 'parent_id': None, 'enabled': True,
                   'name': 'FL1 pos'},
            'g2': {'kind': 'threshold', 'channel': 'FL2-A', 'value': 0.5,
                   'op': '>=', 'parent_id': 'g1', 'enabled': True,
                   'name': 'FL2 pos'},
            'g3': {'kind': 'threshold', 'channel': 'FL9-A', 'value': 0.5,
                   'op': '>=', 'parent_id': None, 'enabled': True,
                   'name': 'Missing chan'},
            'g4': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.1,
                   'op': '>=', 'parent_id': 'g3', 'enabled': True,
                   'name': 'Under missing'},
        }
        ed._sample_gate_order['sample_a'] = ['g1', 'g2', 'g3', 'g4']
        ed._set_active_sample('sample_a')
        ed._export_flowjo_wsp()
        total, pops = _nodes(out)['sample_a']
        fl1 = s.data['FL1-A'].to_numpy() >= 0.4
        fl2 = fl1 & (s.data['FL2-A'].to_numpy() >= 0.5)
        assert 0 < fl2.sum() < fl1.sum() < 200
        # No FCS behind this sample, so its own event count.
        assert total == '200'
        assert pops['FL1 pos'] == str(int(fl1.sum()))
        assert pops['FL2 pos'] == str(int(fl2.sum()))
        # Uncomputable: no count attribute, never '0'.
        assert pops['Missing chan'] is None
        assert pops['Under missing'] is None
    finally:
        fd.asksaveasfilename = saved
        root.destroy()


def test_gui_wsp_export_of_a_subset_of_the_file_writes_no_population_counts(
        tmp_path):
    """300 events in the FCS, 120 kept in the editor: the SampleNode keeps
    the file's 300, and no population gets the count of the 120 (the gate
    holds a different number of the file's 300 events)."""
    import tkinter.filedialog as fd

    from openflo.fcs_export import write_fcs
    root, ed = _editor_or_skip()
    saved = fd.asksaveasfilename
    try:
        rng = np.random.default_rng(7)
        full = pd.DataFrame(rng.uniform(0, 1000, (300, 2)),
                            columns=pd.Index(['FSC-A', 'FL1-A']))
        fcs = tmp_path / 'sample_e.fcs'
        write_fcs(full, str(fcs))
        out = tmp_path / 'subset.wsp'
        fd.asksaveasfilename = lambda *a, **k: str(out)
        ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
            on_done(work()) if on_done else work())
        s = _load_sample(ed, 'sample_e', cols=('FSC-A', 'FL1-A'), n=120)
        s.path = str(fcs)
        s.data = full.iloc[:120].reset_index(drop=True).copy()
        ed._sample_gates['sample_e'] = {
            'g1': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 500.0,
                   'op': '>=', 'parent_id': None, 'enabled': True,
                   'name': 'Hi'}}
        ed._sample_gate_order['sample_e'] = ['g1']
        ed._set_active_sample('sample_e')
        ed._export_flowjo_wsp()
        in_file = int((full['FL1-A'].to_numpy() >= 500).sum())
        in_kept = int((s.data['FL1-A'].to_numpy() >= 500).sum())
        assert in_kept != in_file
        total, pops = _nodes(out)['sample_e']
        assert total == '300'
        assert pops == {'Hi': None}
    finally:
        fd.asksaveasfilename = saved
        root.destroy()


def test_writer_sample_count_is_the_files_total_and_counts_are_optional(
        tmp_path):
    """The library: the <SampleNode> count is the FCS's $TOT even when the
    caller counted fewer events (QC, downsampling), so a re-import still
    recognises the file, and then the caller's counts (of fewer events) are
    left out; `events` is used only when there is no file; a population
    with no count given carries none."""
    from openflo.fcs_export import write_fcs
    from openflo.wsp_reader import fcs_event_total, wsp_sample_events
    from openflo.wsp_writer import WspWriter
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.uniform(0, 1000, (300, 2)),
                      columns=pd.Index(['FSC-A', 'FL1-A']))
    fcs = tmp_path / 'sample_b.fcs'
    write_fcs(df, str(fcs))
    assert fcs_event_total(fcs) == 300
    gates = [{'id': 'a', 'kind': 'threshold', 'channel': 'FL1-A',
              'value': 500.0, 'op': '>=', 'parent_id': None, 'name': 'Pos'},
             {'id': 'b', 'kind': 'threshold', 'channel': 'FSC-A',
              'value': 500.0, 'op': '>=', 'parent_id': 'a', 'name': 'Big'}]
    w = WspWriter()
    w.add_sample('sample_b', str(fcs), ['FSC-A', 'FL1-A'], gates,
                 counts={'a': 140, 'b': float('nan')}, events=300)
    w.add_sample('sample_f', str(fcs), ['FSC-A', 'FL1-A'], gates,
                 counts={'a': 55, 'b': 20}, events=120)
    w.add_sample('sample_c', '', ['FSC-A', 'FL1-A'], gates, events=250)
    w.add_sample('sample_d', '', ['FSC-A', 'FL1-A'], gates)
    out = tmp_path / 'lib.wsp'
    w.write(str(out))
    got = _nodes(out)
    assert got['sample_b'] == ('300', {'Pos': '140', 'Big': None})
    assert got['sample_f'] == ('300', {'Pos': None, 'Big': None})
    assert got['sample_c'] == ('250', {'Pos': None, 'Big': None})
    assert got['sample_d'] == (None, {'Pos': None, 'Big': None})
    root = ET.parse(out).getroot()
    sample_b = next(s for s in root.iter('Sample')
                    if s.find('SampleNode').get('name') == 'sample_b')
    assert wsp_sample_events(sample_b) == 300
