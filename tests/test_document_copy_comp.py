"""Document.copy_gates and the compensation view/switch (phone PR 7b)."""
from __future__ import annotations

import numpy as np
import pytest

from openflo.document import DocumentError
from tests.test_document import SPILL, gated_doc, spill_fcs


def _names(doc, s):
    out = []

    def walk(ns, depth):
        for n in ns:
            out.append((depth, n['name']))
            walk(n['children'], depth + 1)
    walk(doc.gate_tree(s), 0)
    return out


def _counts(doc, s):
    return {r['population']: r['count'] for r in doc.stats(s, channels=[])}


def test_copy_gives_each_target_the_same_tree_and_counts(tmp_path):
    doc, s1, cells, pos, mid = gated_doc(tmp_path)
    doc.add_gate(s1, {'kind': 'boolean', 'op': 'or', 'operands': [pos, mid],
                      'name': 'either'}, cells)
    q = doc.add_quadrant(s1, 'FL1-A', 'FL2-A', 0.4, 0.4, cells)
    s2 = doc.open_fcs(spill_fcs(tmp_path / 'b' / 's1.fcs'))
    s3 = doc.open_fcs(spill_fcs(tmp_path / 'c' / 's1.fcs'))
    junk = doc.add_gate(s3, {'kind': 'threshold', 'channel': 'FSC-A',
                             'value': 1.0, 'name': 'old'})['id']
    out = doc.copy_gates(s1)
    assert set(out) == {s2, s3}
    assert out[s3]['removed'] == 1 and out[s2]['removed'] == 0
    assert all(o['missing'] == [] for o in out.values())
    for t in (s2, s3):
        assert _names(doc, t) == _names(doc, s1)
        # The same files: the same counts, population by population.
        assert _counts(doc, t) == _counts(doc, s1)
        assert len(out[t]['ids']) == len(doc._samples[s1].order)
        either = [g for g in out[t]['ids']
                  if doc.gate(t, g)['name'] == 'either']
        ops = doc.gate(t, either[0])['operands']
        assert set(ops) <= set(out[t]['ids'])
    assert junk not in doc._samples[s3].gates
    # The copied quadrant is one set in each target, the source's kept.
    assert {doc.gate(s1, g)['quad_set'] for g in q} == {'qs1'}
    for t in (s2, s3):
        sets = {doc.gate(t, g).get('quad_set') for g in out[t]['ids']}
        assert len(sets - {None}) == 1


def test_copy_can_add_after_a_targets_own_gates(tmp_path):
    doc, s1, *_ = gated_doc(tmp_path)
    s2 = doc.open_fcs(spill_fcs(tmp_path / 'b' / 's1.fcs'))
    own = doc.add_gate(s2, {'kind': 'threshold', 'channel': 'FSC-A',
                            'value': 1.0, 'name': 'own'})['id']
    out = doc.copy_gates(s1, [s2], replace=False)
    assert out[s2]['removed'] == 0
    assert doc._samples[s2].order[0] == own
    assert len(doc._samples[s2].order) == 1 + len(doc._samples[s1].order)


def test_copy_names_channels_a_target_lacks_and_refuses_bad_targets(tmp_path):
    doc, s1, *_ = gated_doc(tmp_path)
    doc.add_gate(s1, {'kind': 'threshold', 'channel': 'FL3-A', 'value': 0.1,
                      'name': 'FL3+'})
    s2 = doc.open_fcs(spill_fcs(tmp_path / 'b' / 's1.fcs'))
    doc._samples[s2].data.data = doc._samples[s2].data.data.drop(
        columns=['FL3-A'])
    assert doc.copy_gates(s1, [s2])[s2]['missing'] == ['FL3-A']
    with pytest.raises(DocumentError, match='onto itself'):
        doc.copy_gates(s1, [s1])
    with pytest.raises(DocumentError):
        doc.copy_gates(s1, ['nope'])
    assert doc.copy_gates(s1, []) == {}


def test_compensation_shows_the_files_matrix_and_switches_off_and_on(tmp_path):
    doc, s1, *_ = gated_doc(tmp_path)
    c = doc.compensation(s1)
    assert c['on'] and c['source'] == '$SPILL'
    assert c['channels'] == ['FL1-A', 'FL2-A', 'FL3-A']
    np.testing.assert_allclose(c['matrix'], SPILL)
    np.testing.assert_allclose(c['file']['matrix'], SPILL)
    before = _counts(doc, s1)
    stored = doc._samples[s1].data.data['FL1-A'].to_numpy().copy()
    off = doc.set_compensation(s1, False)
    assert not off['on'] and off['source'] == 'off' and off['lost'] == 0
    np.testing.assert_array_equal(off['matrix'], np.eye(3))
    # Off: the measured (spilled) values, so FL1+ changes; gates are kept.
    assert not np.allclose(doc._samples[s1].data.data['FL1-A'], stored)
    assert _counts(doc, s1) != before
    assert doc.set_compensation(s1, False)['lost'] == 0   # already off
    on = doc.set_compensation(s1, True)
    assert on['on'] and on['source'] == '$SPILL'
    np.testing.assert_allclose(doc._samples[s1].data.data['FL1-A'], stored,
                               rtol=1e-9, atol=1e-9)
    assert _counts(doc, s1) == before


def test_a_file_without_spillover_cannot_be_turned_on(tmp_path):
    from openflo.document import Document
    doc = Document()
    s = doc.open_fcs(spill_fcs(tmp_path / 'n.fcs', spill=False))
    c = doc.compensation(s)
    assert not c['on'] and c['file'] is None and c['matrix'] is None
    with pytest.raises(DocumentError, match='no spillover'):
        doc.set_compensation(s, True)


def test_copy_and_compensation_are_callable_through_rpc(tmp_path):
    from openflo import document_rpc as rpc
    r = rpc.DocumentRpc()
    a = r.call('open_fcs', {'path': spill_fcs(tmp_path / 'a' / 's.fcs')})
    b = r.call('open_fcs', {'path': spill_fcs(tmp_path / 'b' / 's.fcs')})
    g = r.call('add_gate', {'sample': a, 'gate': {
        'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5, 'name': 'p'}})
    out = r.call('copy_gates', {'sample': a, 'to': [b]})
    assert list(out) == [b] and len(out[b]['ids']) == 1 and g['id']
    c = r.call('compensation', {'sample': a})
    assert c['on'] and c['matrix']['__ndarray__'] == 'float64'
    assert c['file']['matrix']['__ndarray__'] == 'float64'
    off = r.call('set_compensation', {'sample': a, 'on': False})
    assert off['on'] is False and off['lost'] == 0


def test_turning_on_keeps_a_matrix_already_applied(tmp_path):
    doc, s1, *_ = gated_doc(tmp_path)
    other = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]])
    doc._samples[s1].data.recompensate(other, ['FL1-A', 'FL2-A', 'FL3-A'],
                                       source='manual')
    r = doc.set_compensation(s1, True)
    assert r['source'] == 'manual' and r['lost'] == 0
    np.testing.assert_allclose(r['matrix'], other)
