"""WspWriter: every <Sample> with a readable FCS carries a <Keywords> block.

FlowJo writes the FCS TEXT segment into each sample of a workspace, keys in
their original case ('$P3N', not 'p3n'), between the DataSet and the
SampleNode. The writer emitted only DataSet + SampleNode.

Display <Transformations> are deliberately not written: FlowJo stores
gate coordinates in raw units and fills in a default transform for any
parameter the workspace leaves out.
"""
import xml.etree.ElementTree as ET

import flowio
import numpy as np

import openflo.pipeline as fp

CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A']


def _write_fcs(path, extra=None):
    rng = np.random.default_rng(0)
    events = np.column_stack([
        rng.uniform(0, 2e5, 200),
        rng.uniform(0, 2e5, 200),
        rng.normal(500, 300, 200),
    ]).astype(np.float32)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, events.ravel().tolist(), CHANNELS,
                          opt_channel_names=['', '', 'MarkerX'],
                          metadata_dict=extra or {})
    return str(path)


def _export(tmp_path, fcs):
    w = fp.WspWriter(cytometer='Synthetic')
    w.add_sample('tube 1', fcs, CHANNELS, [
        {'kind': 'threshold', 'channel': 'FL1-A', 'value': 100.0,
         'id': 'g1', 'parent_id': None, 'name': 'pos'}])
    out = tmp_path / 'export.wsp'
    w.write(str(out))
    return ET.parse(out).getroot().find('SampleList/Sample')


def test_sample_carries_fcs_text_keywords(tmp_path):
    fcs = _write_fcs(tmp_path / 'tube 1.fcs',
                     {'NOTE A': 'x/y', 'NOTE B': 'a&b <c> "q"'})
    sample = _export(tmp_path, fcs)
    tags = [c.tag for c in sample]
    assert 'Keywords' in tags, f'<Sample> children {tags}: no <Keywords> block'
    assert (tags.index('DataSet') < tags.index('Keywords')
            < tags.index('SampleNode'))

    pairs = [(k.get('name'), k.get('value')) for k in sample.find('Keywords')]
    # FlowJo's own first keyword, then the FCS TEXT (the form FlowJo 10.10.2
    # computed when measured).
    assert pairs[0] == ('FJ_FCS_VERSION', '3')
    pairs = pairs[1:]
    got = dict(pairs)
    assert len(got) == len(pairs), 'duplicate keyword names'
    # Original case with the '$' kept: the form FlowJo writes.
    assert got['$PAR'] == '3'
    assert [got[f'$P{i}N'] for i in (1, 2, 3)] == CHANNELS
    assert got['$P3S'] == 'MarkerX'
    # An escaped delimiter and XML metacharacters survive verbatim.
    assert got['NOTE A'] == 'x/y'
    assert got['NOTE B'] == 'a&b <c> "q"'
    # Every TEXT keyword is present, with the value FlowIO reads for it.
    text = flowio.FlowData(fcs).text
    assert {k.lstrip('$').lower(): v for k, v in pairs} == dict(text)


def test_control_character_in_keyword_is_dropped_not_written(tmp_path):
    path = tmp_path / 'tube 1.fcs'
    fcs = _write_fcs(path, {'NOTE C': 'aZb'})
    raw = path.read_bytes()
    assert raw.count(b'aZb') == 1
    path.write_bytes(raw.replace(b'aZb', b'a\x01b'))   # same length
    # ET.parse inside _export raises if the raw \x01 reached the XML.
    sample = _export(tmp_path, fcs)
    kw = sample.find('Keywords')
    assert kw is not None, 'no <Keywords> block'
    got = {k.get('name'): k.get('value') for k in kw}
    assert got['NOTE C'] == 'ab'


def test_unreadable_fcs_exports_without_keywords(tmp_path):
    sample = _export(tmp_path, str(tmp_path / 'absent.fcs'))
    tags = [c.tag for c in sample]
    assert 'DataSet' in tags and 'SampleNode' in tags
    assert 'Keywords' not in tags
