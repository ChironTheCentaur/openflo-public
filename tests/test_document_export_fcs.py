"""Document.export_population_fcs: a gated population written as a new
FCS file, as the editor's Export population as FCS writes it (the phone's
Export as FCS). Generated data only."""
import os

import flowio
import numpy as np
import pytest

from openflo.document import Document, DocumentError
from openflo.fcs_export import population_export_frame
from openflo.fcs_text import fcs_text_pairs
from tests._fcs_writer import write_fcs
from tests.test_document import CELLS, gated_doc


def _read(path):
    f = flowio.FlowData(str(path))
    names = [f.channels[k]['pnn'] for k in sorted(f.channels, key=int)]
    ev = np.reshape(np.asarray(f.events, dtype=float), (-1, f.channel_count))
    return names, ev, dict(fcs_text_pairs(str(path)))


def test_the_file_holds_the_population_s_compensated_linear_values(tmp_path):
    doc, name, _, pos, _ = gated_doc(tmp_path)
    out = tmp_path / 'out' / 'pos.fcs'
    r = doc.export_population_fcs(name, pos, str(out))
    count = doc.stats(name, gates=[pos], channels=[], stats=[], all_events=False)[0]['count']
    assert r == {'path': str(out), 'events': count, 'dropped': []}
    names, ev, text = _read(out)
    rec = doc._samples[name]
    mask, _ = doc._population(rec, pos)
    want, ranges, _ = population_export_frame(rec.data, mask)
    assert names == list(want.columns)
    np.testing.assert_allclose(ev, want.to_numpy(dtype=float), rtol=1e-6)
    # The values are compensated, so no spillover goes with them; the
    # matrix applied is said in OpenFlo's own keywords.
    assert not any('SPILL' in k.upper() for k in text)
    assert text['OPENFLO_APPLIED_MATRIX'].startswith('3,FL1-A,FL2-A,FL3-A,')
    assert text['$TOT'] == str(count)
    for i, c in enumerate(names, 1):
        assert text[f'$P{i}R'] == str(ranges[c])


def test_source_keywords_voltages_and_the_marker_set_here_go_with_it(tmp_path):
    rng = np.random.default_rng(5)
    os.makedirs(tmp_path / 'raw')
    src = write_fcs(str(tmp_path / 'raw' / 'k.fcs'), rng.uniform(0, 9e4, (600, 3)),
                    ['FSC-A', 'SSC-A', 'FL1-A'], datatype='F', pnr=[1048576, 1048576, 131072],
                    extra={'$CYT': 'Bench', '$P3S': 'FL1 file marker', '$P3V': '450',
                           '$P2V': '300'})
    doc = Document()
    name = doc.open_fcs(src)
    g = doc.add_gate(name, CELLS)['id']
    out = tmp_path / 'out' / 'cells.fcs'
    doc.export_population_fcs(name, g, str(out))
    names, _, text = _read(out)
    i3 = names.index('FL1-A') + 1
    assert text['$CYT'] == 'Bench'
    # Each channel's range is the source file's, not the 262144 default.
    assert [text[f'$P{i}R'] for i in range(1, 4)] == [
        {'FSC-A': '1048576', 'SSC-A': '1048576', 'FL1-A': '131072'}[c] for c in names]
    assert text[f'$P{i3}S'] == 'FL1 file marker'
    assert text[f'$P{i3}V'] == '450' and text[f'$P{names.index("SSC-A") + 1}V'] == '300'
    # A marker named on the phone wins over the file's.
    doc.set_channel_label('FL1-A', 'FL1 named here')
    doc.export_population_fcs(name, g, str(out))
    assert _read(out)[2][f'$P{i3}S'] == 'FL1 named here'


def test_an_empty_or_unknown_population_and_a_source_path_are_refused(tmp_path):
    doc, name, cells, pos, _ = gated_doc(tmp_path)
    empty = doc.add_gate(name, {'kind': 'threshold', 'channel': 'FL3-A', 'value': 1e12,
                                'name': 'none'}, cells)['id']
    with pytest.raises(DocumentError, match='empty'):
        doc.export_population_fcs(name, empty, str(tmp_path / 'out' / 'e.fcs'))
    assert not (tmp_path / 'out' / 'e.fcs').exists()
    both = doc.add_gate(name, {'kind': 'boolean', 'op': 'and', 'operands': [pos, empty],
                               'name': 'Both'})['id']
    doc.delete_gate(name, empty)
    with pytest.raises(DocumentError, match='cannot be computed'):
        doc.export_population_fcs(name, both, str(tmp_path / 'out' / 'b.fcs'))
    src = doc.samples()[0]['path']
    with pytest.raises(DocumentError, match='a file this document read'):
        doc.export_population_fcs(name, pos, src)
    with pytest.raises(DocumentError, match='no gate'):
        doc.export_population_fcs(name, 'nope', str(tmp_path / 'out' / 'x.fcs'))


def test_a_column_an_fcs_cannot_carry_is_left_out_and_said(tmp_path):
    doc, name, _, pos, _ = gated_doc(tmp_path)
    # A text column (a cluster's name, say) beside the channels.
    data = doc._samples[name].data.data
    data['cluster'] = np.where(np.arange(len(data)) % 2, 'a', 'b')
    r = doc.export_population_fcs(name, pos, str(tmp_path / 'out' / 'pos.fcs'))
    assert r['dropped'] == ['cluster']
    names, _ev, _text = _read(tmp_path / 'out' / 'pos.fcs')
    assert 'cluster' not in names and 'FL1-A' in names


def _pns(path, channel):
    names, _ev, text = _read(path)
    return text.get(f'$P{names.index(channel) + 1}S')


def test_a_marker_named_here_is_written_after_the_session_is_saved_and_reopened(tmp_path):
    doc, name, _, pos, _ = gated_doc(tmp_path)
    doc.set_channel_label('FL1-A', 'FL1 marker')
    doc.save_session(str(tmp_path / 'x.flowsession'))
    back = Document()
    back.open_session(str(tmp_path / 'x.flowsession'))
    back.export_population_fcs(name, pos, str(tmp_path / 'out' / 'b.fcs'))
    assert _pns(tmp_path / 'out' / 'b.fcs', 'FL1-A') == 'FL1 marker'
    # The desktop reads the same trail: the label was set, not the file's.
    import json

    from openflo.channel_label_trail import explicit_labels_from_trail
    saved = json.loads((tmp_path / 'x.flowsession').read_text(encoding='utf-8'))
    assert explicit_labels_from_trail(saved['audit'], saved['channel_labels']) == {'FL1-A': 'FL1 marker'}
    # Dropped again here: the trail says so, and the file's own label is
    # written (none on this generated file).
    back.set_channel_label('FL1-A', '')
    back.save_session(str(tmp_path / 'y.flowsession'))
    again = Document()
    again.open_session(str(tmp_path / 'y.flowsession'))
    assert again._labels == {}


def test_a_desktop_relabel_in_the_session_trail_is_written(tmp_path):
    import json
    doc, name, _, pos, _ = gated_doc(tmp_path)
    doc.save_session(str(tmp_path / 'x.flowsession'))
    data = json.loads((tmp_path / 'x.flowsession').read_text(encoding='utf-8'))
    # As the gate editor saves a relabel: the label, and its record in the
    # trail saying it was set.
    data['channel_labels'] = {'FL2-A': 'FL2 marker'}
    data['audit'] = [{'seq': 1, 'time': None, 'action': 'channel_labels.set',
                      'details': {'source': 'Label channels', 'labels': {'FL2-A': 'FL2 marker'},
                                  'explicit': {'FL2-A': 'FL2 marker'}}}]
    (tmp_path / 'x.flowsession').write_text(json.dumps(data), encoding='utf-8')
    back = Document()
    back.open_session(str(tmp_path / 'x.flowsession'))
    back.export_population_fcs(name, pos, str(tmp_path / 'out' / 'c.fcs'))
    assert _pns(tmp_path / 'out' / 'c.fcs', 'FL2-A') == 'FL2 marker'
    # Saved again after a rename here: the desktop's record kept, this
    # one after it.
    back.set_channel_label('FL1-A', 'FL1 marker')
    back.save_session(str(tmp_path / 'y.flowsession'))
    again = Document()
    again.open_session(str(tmp_path / 'y.flowsession'))
    assert again._labels == {'FL1-A': 'FL1 marker', 'FL2-A': 'FL2 marker'}
    saved = json.loads((tmp_path / 'y.flowsession').read_text(encoding='utf-8'))
    assert [e['seq'] for e in saved['audit']] == [1, 2]


def test_a_name_set_here_wins_over_one_a_session_opened_later_set(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    doc.set_channel_label('FL1-A', 'FL1 marker')
    doc.save_session(str(tmp_path / 'x.flowsession'))
    other = Document()
    other.open_fcs(str(tmp_path / 'raw' / 's1.fcs'))
    other.set_channel_label('FL1-A', 'FL1 here')
    other.open_session(str(tmp_path / 'x.flowsession'))
    assert other._labels == {'FL1-A': 'FL1 here'}
    assert {c['name']: c['label'] for c in other.channels(name)}['FL1-A'] == 'FL1 here'
