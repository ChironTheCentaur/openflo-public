"""gate_store: adding, replacing and removing gates in a sample's
``{gid: gate}`` dict and id order, without an editor."""
from __future__ import annotations

import itertools

import pytest

from openflo import gate_store as gs
from openflo.gate_lineage import descendants, subtree


def _ids(start=1):
    """A new_id callable handing out g<start>, g<start+1>, ..."""
    return (f'g{n}' for n in itertools.count(start)).__next__


def _thr(ch='FL1-A', value=1.0, **kw):
    return {'kind': 'threshold', 'channel': ch, 'op': '>', 'value': value,
            **kw}


def _rect(**kw):
    return {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
            'x0': 0.0, 'x1': 1.0, 'y0': 0.0, 'y1': 1.0, **kw}


def _tree():
    """g1 > (g2 > g4, g3), g5 at root; ids in that order."""
    gates, order = {}, []
    new = _ids()
    g1 = gs.add_gate(gates, order, _rect(name='A'), new).id
    g2 = gs.add_gate(gates, order, _rect(name='B'), new, parent_id=g1).id
    g3 = gs.add_gate(gates, order, _rect(name='C'), new, parent_id=g1).id
    gs.add_gate(gates, order, _rect(name='D'), new, parent_id=g2)
    gs.add_gate(gates, order, _rect(name='E'), new)
    assert (g1, g2, g3) == ('g1', 'g2', 'g3')
    return gates, order


# ── ids ──────────────────────────────────────────────────────────────────────

def test_next_gate_id_counts_up_per_sample_and_never_reuses():
    seqs = {}
    assert [gs.next_gate_id(seqs, 's1') for _ in range(3)] == ['g1', 'g2', 'g3']
    assert gs.next_gate_id(seqs, 's2') == 'g1'
    assert seqs == {'s1': 3, 's2': 1}
    seqs['s1'] = 7
    assert gs.next_gate_id(seqs, 's1') == 'g8'
    assert gs.gate_id(12) == 'g12'


def test_the_wsp_export_and_template_ids_are_gate_ids():
    """Ids made outside an editor take the same g<n> form, from g1."""
    from openflo.cli_wsp import build_export_gate_list
    from openflo.gate_templates import template_ids
    out = build_export_gate_list(
        [_rect(id='a', parent_id=None), _rect(id='b', parent_id='a')],
        {'FL3-A': 2.0})
    assert [g['id'] for g in out] == ['g1', 'g2', 'g3']
    assert out[1]['parent_id'] == 'g1'
    raw = [_rect(_import_id='x', parent_id=None),
           _rect(_import_id='y', parent_id='x')]
    assert [(g['id'], g['parent_id']) for g in template_ids(raw)] == [
        ('g1', None), ('g2', 'g1')]


# ── add ──────────────────────────────────────────────────────────────────────

def test_add_gate_fills_parent_and_enabled_and_appends():
    gates, order = {}, []
    assert gs.add_gate(gates, order, _rect(), _ids(5),
                       parent_id='g9') == ('g5', False)
    assert order == ['g5']
    assert gates['g5']['parent_id'] == 'g9'
    assert gates['g5']['enabled'] is True


def test_add_gate_keeps_a_parent_and_off_state_the_gate_names():
    gates, order = {}, []
    gid = gs.add_gate(gates, order, _rect(parent_id=None, enabled=False),
                      _ids(), parent_id='g9').id
    assert gates[gid]['parent_id'] is None
    assert gates[gid]['enabled'] is False


def test_a_second_threshold_on_a_channel_replaces_the_first():
    gates, order = {}, []
    new = _ids()
    first = gs.add_gate(gates, order, _thr(value=1.0, name='Pos', color='red',
                                           enabled=False, open=True), new).id
    nudged = _thr(value=2.0, color='blue', name='other')
    assert gs.add_gate(gates, order, nudged, new) == (first, True)
    assert order == [first]
    g = gates[first]
    assert g is nudged
    assert g['value'] == 2.0
    # What the user set on the gate is kept; only the position moves.
    assert (g['color'], g['enabled'], g['name'], g['open']) == (
        'red', False, 'Pos', True)
    # No id was spent on the replacement.
    assert gs.add_gate(gates, order, _rect(), new) == ('g2', False)


@pytest.mark.parametrize('second', [
    _thr(ch='FL2-A'),                                   # other channel
    _thr(parent_id='g7'),                               # other parent
    {'kind': 'interval', 'channel': 'FL1-A', 'lo': 0, 'hi': 1,
     'parent_id': 'g7'},
])
def test_a_1d_gate_on_another_channel_or_parent_is_added(second):
    gates, order = {}, []
    new = _ids()
    gs.add_gate(gates, order, _thr(parent_id=None), new)
    assert gs.add_gate(gates, order, second, new) == ('g2', False)
    assert order == ['g1', 'g2']


def test_an_interval_replaces_a_threshold_on_its_channel():
    gates, order = {}, []
    new = _ids()
    gs.add_gate(gates, order, _thr(), new)
    iv = {'kind': 'interval', 'channel': 'FL1-A', 'lo': 0, 'hi': 1}
    assert gs.add_gate(gates, order, iv, new) == ('g1', True)
    assert gates['g1']['kind'] == 'interval'


def test_2d_gates_are_never_replaced():
    gates, order = {}, []
    new = _ids()
    gs.add_gate(gates, order, _rect(), new)
    assert gs.add_gate(gates, order, _rect(), new) == ('g2', False)


def test_replace_1d_false_adds_siblings():
    gates, order = {}, []
    new = _ids()
    gs.add_gate(gates, order, _thr(), new)
    assert gs.add_gate(gates, order, _thr(value=3.0), new,
                       replace_1d=False) == ('g2', False)
    assert len(gates) == 2


def test_replace_1d_set_replaces_only_those_ids_once():
    gates, order = {}, []
    new = _ids()
    gs.add_gate(gates, order, _thr(value=1.0), new)       # g1, pre-existing
    allowed = {'g1'}
    # Replaces g1 and spends it ...
    assert gs.add_gate(gates, order, _thr(value=2.0), new,
                       replace_1d=allowed) == ('g1', True)
    assert allowed == set()
    # ... so the next sibling of the install is a gate of its own.
    assert gs.add_gate(gates, order, _thr(value=3.0), new,
                       replace_1d=allowed) == ('g2', False)
    # And a gate outside the set is never replaced.
    assert gs.add_gate(gates, order, _thr(value=4.0), new,
                       replace_1d={'g9'}) == ('g3', False)


def test_a_new_id_that_is_already_taken_is_reported_as_added():
    """The counter decides the id: one it hands out that a gate already
    holds overwrites that gate as a new gate, and the result says so -- it
    is not a 1-D replacement, whatever the dict's size did."""
    gates, order = {}, []
    gs.add_gate(gates, order, _rect(name='old'), _ids())
    res = gs.add_gate(gates, order, _rect(name='new'), _ids())
    assert res == ('g1', False)
    assert res.replaced is False and res.id == 'g1'
    assert len(gates) == 1 and gates['g1']['name'] == 'new'
    assert order == ['g1', 'g1']


def test_one_d_twin_takes_the_first_match():
    gates = {'g1': _rect(parent_id=None),
             'g2': _thr(parent_id=None),
             'g3': _thr(parent_id=None)}
    assert gs.one_d_twin(gates, _thr(parent_id=None)) == 'g2'
    assert gs.one_d_twin(gates, _thr(parent_id=None), {'g3'}) == 'g3'
    assert gs.one_d_twin(gates, _thr(parent_id=None), False) is None


# ── update ───────────────────────────────────────────────────────────────────

def test_update_gate_in_place_keeps_the_dict_and_the_named_keys():
    old = _rect(parent_id='g1', enabled=False, name='old', open=True)
    gates = {'g2': old}
    new = _rect(x1=5.0, parent_id=None, enabled=True, name='new')
    snapshot = dict(new)
    out = gs.update_gate(gates, 'g2', new, ('parent_id', 'enabled', 'open'),
                         in_place=True)
    assert out is old and gates['g2'] is old
    assert old['x1'] == 5.0 and old['name'] == 'new'
    assert (old['parent_id'], old['enabled'], old['open']) == ('g1', False,
                                                               True)
    assert new == snapshot                      # the copy is left as it was


def test_update_gate_stores_the_new_dict_by_default():
    gates = {'g1': _thr(color='red')}
    new = _thr(value=9.0)
    assert gs.update_gate(gates, 'g1', new) is new
    assert gates['g1'] is new and new['color'] == 'red'
    assert 'name' not in new                    # only keys the old one had


# ── remove ───────────────────────────────────────────────────────────────────

def test_remove_gate_cascades_to_every_descendant():
    gates, order = _tree()
    assert gs.remove_gate(gates, order, 'g1') == ['g1', 'g2', 'g3', 'g4']
    assert list(gates) == ['g5'] and order == ['g5']


def test_remove_gate_of_a_leaf_leaves_its_siblings():
    gates, order = _tree()
    assert gs.remove_gate(gates, order, 'g2') == ['g2', 'g4']
    assert order == ['g1', 'g3', 'g5']


def test_remove_gate_without_cascade_leaves_the_children():
    gates, order = _tree()
    assert gs.remove_gate(gates, order, 'g1', cascade=False) == ['g1']
    assert order == ['g2', 'g3', 'g4', 'g5']
    assert gates['g2']['parent_id'] == 'g1'


def test_removing_a_missing_gate_changes_nothing():
    gates, order = _tree()
    order.append('g99')
    assert gs.remove_gate(gates, order, 'g99') == []
    assert order[-1] == 'g99' and len(gates) == 5


def test_remove_gate_ends_on_a_parent_cycle():
    gates = {'g1': _rect(parent_id='g2'), 'g2': _rect(parent_id='g1'),
             'g3': _rect(parent_id=None)}
    order = ['g1', 'g2', 'g3']
    assert gs.remove_gate(gates, order, 'g1') == ['g1', 'g2']
    assert order == ['g3']


def test_clear_gates():
    gates, order = _tree()
    assert gs.clear_gates(gates, order) == 5
    assert gates == {} and order == []
    # The order is emptied with the gates, even one left stale.
    stale = ['g1']
    assert gs.clear_gates({}, stale) == 0
    assert stale == []


# ── the walk the cascade uses ────────────────────────────────────────────────

def test_subtree_lists_parents_before_children_breadth_first():
    gates, _ = _tree()
    assert subtree(gates, 'g1') == ['g1', 'g2', 'g3', 'g4']
    assert subtree(gates, 'g4') == ['g4']
    assert subtree(gates, 'missing') == []
    assert descendants(gates, 'g1') == {'g2', 'g3', 'g4'}


def test_depends_on_follows_parents_and_boolean_operands():
    from openflo.gate_store import depends_on
    gates = {'g1': {'kind': 'rect', 'parent_id': None},
             'g2': {'kind': 'threshold', 'parent_id': 'g1'},
             'g3': {'kind': 'boolean', 'parent_id': None,
                    'operands': ['g2', 'gone']},
             'g4': {'kind': 'threshold', 'parent_id': None}}
    assert depends_on(gates, ['g2'], 'g1')            # through a parent
    assert depends_on(gates, ['g3'], 'g1')            # operand, then parent
    assert depends_on(gates, ['g4'], 'g4')            # itself
    assert not depends_on(gates, ['g4', None, 'gone'], 'g1')
    assert not depends_on(gates, ['g1'], 'g2')        # not downwards
    gates['g1']['parent_id'] = 'g3'                   # a cycle: still ends
    assert not depends_on(gates, ['g3'], 'g4')
