"""Boolean gates that combine boolean gates, made and changed on the phone
(openflo.document.Document, as the phone app drives it): a change to one
undone while a gate it named is gone, and the nested gates carried
through a FlowJo workspace, a Gating-ML file and a project with the counts
they had. (The session both ways with the desktop editor is in
tests/test_document_desktop_roundtrip.py, whose gates nest too.)

Generated FCS files only; channels FSC-A, SSC-A, FL1-A ...; sample s1.
"""
from __future__ import annotations

import pytest

from openflo import project_archive as pa
from openflo.document import Document, DocumentError
from tests.test_document import _gml_of, gated_doc, spill_fcs


def _nested(tmp_path):
    """s1 gated cells > FL1+, cells > FL2 mid, then Both = AND(FL1+, FL2 mid)
    under cells, Not both = NOT(Both), and Either = OR(Not both, FL1+), with
    a gate under Not both."""
    doc, name, cells, pos, mid = gated_doc(tmp_path)
    both = doc.add_gate(name, {'kind': 'boolean', 'op': 'and',
                               'operands': [pos, mid], 'name': 'Both'},
                        cells)['id']
    nb = doc.add_gate(name, {'kind': 'boolean', 'op': 'not',
                             'operands': [both], 'name': 'Not both'})['id']
    doc.add_gate(name, {'kind': 'threshold', 'channel': 'FL3-A',
                        'value': 0.4, 'name': 'FL3+'}, nb)
    doc.add_gate(name, {'kind': 'boolean', 'op': 'or',
                        'operands': [nb, pos], 'name': 'Either'})
    return doc, name


def _counts(doc, name):
    """{population as the gate list names it: count}."""
    return {r['population']: r['count'] for r in doc.stats(name)}


def _want(doc, name):
    got = _counts(doc, name)
    # All events and the seven gates, every one counting (none NA, none
    # empty, so a lost operand or a gate left out would show).
    assert all(v is not None and v > 0 for v in got.values()), got
    assert len(got) == 8
    return got


def test_a_change_to_a_lost_boolean_is_undone_with_its_lost_gate(tmp_path):
    """A boolean whose operand was deleted, changed to drop it, is put back
    as it was by Undo (``restore``): still naming the gone gate, lost. A
    plain change cannot name a gate that is gone and was not named
    already, and ``restore`` still refuses a gate resting on itself."""
    doc, name, cells, pos, mid = gated_doc(tmp_path)
    both = doc.add_gate(name, {'kind': 'boolean', 'op': 'and',
                               'operands': [pos, mid], 'name': 'Both'})['id']
    saved = doc.gate(name, both)
    doc.delete_gate(name, mid)
    assert doc.lost_operands(name)['lost'] == ['Both']
    doc.update_gate(name, both, dict(saved, op='or', operands=[pos]))
    assert doc.lost_operands(name)['lost'] == []
    with pytest.raises(DocumentError, match='no gate'):
        doc.update_gate(name, both, saved)
    doc.update_gate(name, both, saved, restore=True)
    assert doc.gate(name, both)['operands'] == [pos, mid]
    assert doc.gate(name, both)['op'] == 'and'
    assert doc.lost_operands(name)['lost'] == ['Both']
    nb = doc.add_gate(name, {'kind': 'boolean', 'op': 'not',
                             'operands': [pos], 'name': 'Not'})['id']
    doc.update_gate(name, nb, dict(doc.gate(name, nb), operands=[both]))
    with pytest.raises(DocumentError, match='rest on itself'):
        doc.update_gate(name, both, dict(saved, operands=[pos, nb]),
                        restore=True)
    # One read with no operands left (a session or template empties a lost
    # boolean's list), changed, is put back by Undo with none: still lost.
    doc._samples[name].gates[nb]['operands'] = []
    empty = doc.gate(name, nb)
    doc.update_gate(name, nb, dict(empty, operands=[pos]))
    with pytest.raises(DocumentError, match='needs'):
        doc.update_gate(name, nb, empty)
    doc.update_gate(name, nb, empty, restore=True)
    assert doc.gate(name, nb)['operands'] == []
    assert 'Not' in doc.lost_operands(name)['lost']
    # A gate that is not a boolean names no operands to keep.
    with pytest.raises(DocumentError, match='needs'):
        doc.update_gate(name, cells, {'kind': 'boolean', 'op': 'and',
                                      'operands': []}, restore=True)


def test_nested_booleans_count_the_same_from_a_flowjo_workspace(tmp_path):
    """Written as the desktop's FlowJo export writes them (AndNode, NotNode
    and OrNode over boolean populations), the workspace opens on the phone
    with the same populations counting the same events."""
    from openflo.gate_text import wsp_unexportable
    from openflo.sample_prep import stored_transforms
    from openflo.wsp_writer import WspWriter
    doc, name = _nested(tmp_path)
    want = _want(doc, name)
    rec = doc._samples[name]
    assert wsp_unexportable(rec.gates) == set()
    s = rec.data
    w = WspWriter()
    w.add_sample(name=name, fcs_path=str(tmp_path / 'raw' / 's1.fcs'),
                 channels=list(s.data.columns),
                 gates=[dict(rec.gates[i], id=i) for i in rec.order],
                 compensation=(s.comp_channels, s.comp_matrix),
                 transforms=stored_transforms(s))
    wsp = tmp_path / 'raw' / 'x.wsp'
    w.write(str(wsp))
    phone = Document()
    assert phone.import_wsp(str(wsp))['samples'] == [name]
    assert _counts(phone, name) == want


def test_nested_booleans_count_the_same_from_a_gatingml_file(tmp_path):
    """The desktop's Gating-ML export of nested booleans (BooleanGates
    referring to BooleanGates), put on the same file in a fresh phone
    document, counts the same."""
    doc, name = _nested(tmp_path)
    want = _want(doc, name)
    gml = _gml_of(doc, name, tmp_path / 'panel.xml')
    phone = Document()
    other = phone.open_fcs(spill_fcs(tmp_path / 'again' / 's1.fcs'))
    out = phone.apply_gatingml(gml, [other])
    assert len(out['ids'][other]) == 7
    assert _counts(phone, other) == want


def test_nested_booleans_count_the_same_from_a_project(tmp_path):
    """Saved and packed as a .flowproject, unpacked and opened on the
    phone: the same."""
    doc, name = _nested(tmp_path)
    want = _want(doc, name)
    (tmp_path / 'proj').mkdir()
    sess = tmp_path / 'proj' / 'run.flowsession'
    doc.save_session(str(sess))
    out = tmp_path / ('run' + pa.ARCHIVE_EXT)
    pa.write_project_archive(str(sess), str(out))
    phone = Document()
    r = phone.unpack_project(str(out), str(tmp_path / 'phone'))
    assert phone.open_session(r['session'])['samples'] == [name]
    assert _counts(phone, name) == want
