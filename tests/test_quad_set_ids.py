"""Quadrant-set ids stay apart within a sample (scope audit G3).

The four rects of an editor quadrant share a ``quad_set`` id, and dragging
the quadrant's centre moves every gate in the sample with that id. Ids came
from a window counter that started at qs1 and never went past ids a
reopened session, template, paste, copy or move brought in, so a new
quadrant on FL1-A x FL2-A could take the id of a restored one on
FSC-A x SSC-A, and dragging it moved the off-screen quadrant too: four
populations changed unseen.

Now a new set's id goes past every id the editor holds (each sample's and
each tab's gates), gates that arrive in a sample keep their id only when
no other set there has it (``gate_store.separate_quad_sets``), and a drag
or typed origin moves only the set on its own axes.

Generated data only.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import gate_coords as gc
from openflo.gate_store import (
    highest_quad_set_number,
    install_gates,
    next_gate_id,
    quad_set_ids,
    separate_quad_sets,
)
from openflo.gating import same_gate, same_gate_identity
from tests.conftest import editor_or_skip

OPEN = 1e12
LABELS = ('Q++ (x>, y>)', 'Q+- (x>, y<)', 'Q-+ (x<, y>)', 'Q-- (x<, y<)')


def _quad(x, y, qs, origin=1000.0, first=1, parent=None):
    """Four saved quadrant rects on ``x`` x ``y`` sharing ``qs``, ids
    g<first>.., centred at ``origin`` on both axes."""
    o = float(origin)
    bounds = ((o, OPEN, o, OPEN), (o, OPEN, -OPEN, o),
              (-OPEN, o, o, OPEN), (-OPEN, o, -OPEN, o))
    return [{'id': f'g{first + i}', 'kind': 'rect', 'x_channel': x,
             'y_channel': y, 'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
             'label': lab, 'quad_set': qs, 'quad_origin_x': o,
             'quad_origin_y': o, 'parent_id': parent, 'enabled': True}
            for i, (lab, (x0, x1, y0, y1)) in enumerate(zip(LABELS, bounds, strict=True))]


def _origins(gates, x):
    """The quadrant origins on x channel ``x`` in one sample's gates."""
    return {(g['quad_origin_x'], g['quad_origin_y']) for g in gates.values()
            if g.get('quad_set') and g['x_channel'] == x}


def _ids(gates, x):
    return {g['quad_set'] for g in gates.values()
            if g.get('quad_set') and g['x_channel'] == x}


# ── pure helpers (no Tk) ──────────────────────────────────────────────────

def test_highest_number_reads_only_qs_ids():
    assert highest_quad_set_number(['qs3', 'qs12', 'imp_40', 'qs', 'qs7x',
                                    'xqs99']) == 12
    assert highest_quad_set_number([]) == 0


def test_a_set_keeps_a_free_id_and_stays_one_set():
    added = _quad('FL1-A', 'FL2-A', 'qs4')
    assert separate_quad_sets(added, {'qs1'}) == 0
    assert {g['quad_set'] for g in added} == {'qs4'}


def test_a_taken_id_is_replaced_for_the_whole_set():
    added = _quad('FL1-A', 'FL2-A', 'qs1')
    assert separate_quad_sets(added, {'qs1', 'qs5', 'imp_9'}) == 1
    # One new id for all four, past every qsN there.
    assert {g['quad_set'] for g in added} == {'qs6'}


def test_two_sets_in_one_batch_sharing_an_id_are_split():
    # By channel pair (a session saved while ids collided) ...
    added = _quad('FSC-A', 'SSC-A', 'qs1') + _quad('FL1-A', 'FL2-A', 'qs1')
    assert separate_quad_sets(added, set()) == 1
    assert [g['quad_set'] for g in added] == ['qs1'] * 4 + ['qs2'] * 4
    # ... and by parent (two copies of one quadrant on the same axes).
    added = (_quad('FL1-A', 'FL2-A', 'qs1', parent='g1')
             + _quad('FL1-A', 'FL2-A', 'qs1', parent='g2'))
    assert separate_quad_sets(added, set()) == 1
    assert len({g['quad_set'] for g in added}) == 2


def test_new_ids_are_asked_for_until_one_is_free():
    asked = iter(['qs1', 'qs2', 'qs3'])
    added = _quad('FL1-A', 'FL2-A', 'qs1')
    separate_quad_sets(added, {'qs1', 'qs2'}, lambda: next(asked))
    assert {g['quad_set'] for g in added} == {'qs3'}


def test_gates_without_a_set_are_left_alone():
    g = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A'}
    assert separate_quad_sets([g], {'qs1'}) == 0
    assert 'quad_set' not in g


def test_install_keeps_ids_apart_from_the_sample_s_own():
    gates, order, seqs = {}, [], {'s': 0}
    install_gates(gates, order, _quad('FSC-A', 'SSC-A', 'qs1'),
                  lambda: next_gate_id(seqs, 's'))
    assert quad_set_ids(gates.values()) == {'qs1'}
    install_gates(gates, order, _quad('FL1-A', 'FL2-A', 'qs1'),
                  lambda: next_gate_id(seqs, 's'))
    assert _ids(gates, 'FSC-A') == {'qs1'}
    assert _ids(gates, 'FL1-A') == {'qs2'}


def test_typed_origin_moves_only_the_set_on_its_axes():
    gates = {f'{g["id"]}{k}': g for k, quad in
             enumerate((_quad('FSC-A', 'SSC-A', 'qs1'),
                        _quad('FL1-A', 'FL2-A', 'qs1'))) for g in quad}
    assert gc.set_quadrant_origin(gates, 'qs1', 5.0, 6.0,
                                  key=('qs1', 'FL1-A', 'FL2-A', None)) == 4
    assert _origins(gates, 'FL1-A') == {(5.0, 6.0)}
    assert _origins(gates, 'FSC-A') == {(1000.0, 1000.0)}
    # Without channels, every member with the id (the old behaviour).
    assert gc.set_quadrant_origin(gates, 'qs1', 7.0, 8.0) == 8


# ── the editor ────────────────────────────────────────────────────────────

@pytest.fixture
def ed():
    root, editor = editor_or_skip(accept_confirms=True)
    editor._finish_plot_canvas()
    rng = np.random.default_rng(11)
    cols = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
    for k, name in enumerate(('synthetic_a1', 'synthetic_a2')):
        df = pd.DataFrame(rng.normal(1000 + 100 * k, 50, (1500, 4)),
                          columns=cols)
        editor._samples[name] = SimpleNamespace(
            name=name, path=f'/data/{name}.fcs', data=df,
            fluor_channels=['FL1-A', 'FL2-A'],
            channel_labels={c: c for c in cols})
        editor._sample_order.append(name)
        editor._sample_trial[name] = 'Day1'
        editor._sample_plot_enabled[name] = True
        editor._sample_gates.setdefault(name, {})
        editor._sample_gate_seq.setdefault(name, 0)
        editor._sample_gate_order.setdefault(name, [])
    editor._trial_order.append('Day1')
    editor._channels = cols
    editor._channel_labels = {c: c for c in cols}
    editor._populate_channel_combos()
    editor._set_active_sample('synthetic_a1')
    editor.x_combo.set('FL1-A')
    editor.y_combo.set('FL2-A')
    editor.mode_var.set('dot')
    yield editor
    root.destroy()


def _restore(ed, name, gates):
    """Install saved gates as a reopened session does."""
    ed._install_pending_gates(name, ed._samples[name], gates)


def _draw(ed, at=1050.0):
    """Draw a quadrant on the plotted FL1-A x FL2-A; its quad_set id."""
    assert ed._add_quadrant_set_at(SimpleNamespace(xdata=at, ydata=at),
                                   'FL1-A', 'FL2-A')
    (qs,) = _ids(ed._gates, 'FL1-A')
    return qs


def _drag(ed, qs, kind='quad_origin', to=1200.0, parent=None):
    ed._drag_state = (kind, qs, parent)
    assert ed._motion_drag_quad(kind, to, to)


def test_new_quadrant_after_a_reopened_one_does_not_drag_it(ed):
    _restore(ed, 'synthetic_a1', _quad('FSC-A', 'SSC-A', 'qs1'))
    qs = _draw(ed)
    assert qs != 'qs1'
    _drag(ed, qs)
    assert _origins(ed._gates, 'FL1-A') == {(1200.0, 1200.0)}
    assert _origins(ed._gates, 'FSC-A') == {(1000.0, 1000.0)}


def test_reopened_sets_sharing_an_id_are_split(ed):
    _restore(ed, 'synthetic_a1', _quad('FSC-A', 'SSC-A', 'qs1')
             + _quad('FL1-A', 'FL2-A', 'qs1', first=5))
    assert _ids(ed._gates, 'FSC-A') == {'qs1'}
    assert _ids(ed._gates, 'FL1-A') == {'qs2'}


def test_new_id_goes_past_other_samples_ids(ed):
    # Another sample's reopened set: a new quadrant there (all shown, or a
    # copy) would otherwise share its id.
    _restore(ed, 'synthetic_a2', _quad('FSC-A', 'SSC-A', 'qs5'))
    assert _draw(ed) == 'qs6'


def test_new_id_goes_past_another_tab_s_ids(ed):
    ts = ed._tabs()
    tab = ts.tabs[ts.new_tab()]
    tab.gates = {'synthetic_a1': {g['id']: g for g in
                                  _quad('FSC-A', 'SSC-A', 'qs9')}}
    tab.gate_order = {'synthetic_a1': [f'g{i}' for i in range(1, 5)]}
    assert _draw(ed) == 'qs10'


def test_pasted_quadrant_beside_its_original_takes_a_new_id(ed):
    qs = _draw(ed)
    first = next(g for g in ed._gate_id_order
                 if ed._gates[g]['label'].startswith('Q++'))
    ed._clip_kind = 'gate_tree'
    ed._clip_payload = ed._collect_gate_subtree('synthetic_a1', first)
    ed.gate_tv.selection_set(())
    ed._paste_gate_tree()
    pasted = [g for g in ed._gates.values()
              if g['label'].startswith('Q++') and g['quad_set'] != qs]
    assert len(pasted) == 1
    _drag(ed, qs)
    assert (pasted[0]['quad_origin_x'], pasted[0]['x0']) == (1050.0, 1050.0)


def test_templated_quadrant_takes_a_new_id_beside_a_set_with_its_id(ed):
    _restore(ed, 'synthetic_a1', _quad('FSC-A', 'SSC-A', 'qs1'))
    ed._apply_template_to_sample('synthetic_a1',
                                 _quad('FL1-A', 'FL2-A', 'qs1'),
                                 overwrite=False)
    (qs,) = _ids(ed._gates, 'FL1-A')
    assert qs != 'qs1'
    _drag(ed, qs)
    assert _origins(ed._gates, 'FSC-A') == {(1000.0, 1000.0)}


def test_copied_quadrant_takes_a_new_id_on_the_target(ed):
    _restore(ed, 'synthetic_a2', _quad('FSC-A', 'SSC-A', 'qs1'))
    _restore(ed, 'synthetic_a1', _quad('FL1-A', 'FL2-A', 'qs1'))
    ed._copy_gates_to(['synthetic_a2'])
    tgt = ed._sample_gates['synthetic_a2']
    assert _ids(tgt, 'FSC-A') == {'qs1'}
    (qs,) = _ids(tgt, 'FL1-A')
    assert qs != 'qs1'


def test_moved_quadrant_member_takes_a_new_id_on_the_target(ed):
    _restore(ed, 'synthetic_a1', _quad('FSC-A', 'SSC-A', 'qs1'))
    _restore(ed, 'synthetic_a2', _quad('FL1-A', 'FL2-A', 'qs1'))
    src = ed._sample_gates['synthetic_a2']
    gid = next(g for g in src if src[g]['label'].startswith('Q++'))
    new = ed._move_gate_to_sample('synthetic_a2', gid, 'synthetic_a1', None)
    assert ed._sample_gates['synthetic_a1'][new]['quad_set'] != 'qs1'
    assert _ids(ed._sample_gates['synthetic_a1'], 'FSC-A') == {'qs1'}


@pytest.mark.parametrize('kind', ['quad_origin', 'quad_x', 'quad_y'])
def test_drag_moves_only_the_set_on_the_plotted_axes(ed, kind):
    # Two sets one id, put in directly (no path makes them any more).
    for g in (_quad('FSC-A', 'SSC-A', 'qs1')
              + _quad('FL1-A', 'FL2-A', 'qs1', first=5)):
        ed._gates[g.pop('id')] = g
    _drag(ed, 'qs1', kind)
    assert _origins(ed._gates, 'FSC-A') == {(1000.0, 1000.0)}
    assert _origins(ed._gates, 'FL1-A') != {(1000.0, 1000.0)}


def test_typed_origin_in_the_editor_moves_only_its_set(ed):
    gates = ed._gates
    for g in (_quad('FSC-A', 'SSC-A', 'qs1')
              + _quad('FL1-A', 'FL2-A', 'qs1', first=5)):
        gid = g.pop('id')
        gates[gid] = g
        ed._gate_id_order.append(gid)
    # The FSC-A x SSC-A set (linear axes) takes a typed x divider.
    typed = {k: t for k, _lab, t in gc.fields(gates['g1'], {})}
    typed['quad_origin_x'] = '1300'
    ed._apply_gate_coordinates('synthetic_a1', 'g1', typed)
    assert {x for x, _y in _origins(gates, 'FSC-A')} == {1300.0}
    assert _origins(gates, 'FL1-A') == {(1000.0, 1000.0)}


def test_an_id_is_never_handed_out_twice(ed):
    # A deleted quadrant's id is not given to the next one (the counter
    # only goes up, as gate ids do).
    qs = _draw(ed)
    for name in ed._sample_gates:        # it may be on every sample shown
        ed._sample_gates[name].clear()
        ed._sample_gate_order[name].clear()
    assert _draw(ed) != qs


def test_a_quadrant_arriving_with_a_free_id_keeps_it(ed):
    ed._apply_template_to_sample('synthetic_a1',
                                 _quad('FL1-A', 'FL2-A', 'qs3'),
                                 overwrite=False)
    assert _ids(ed._gates, 'FL1-A') == {'qs3'}


def _rect(gid, name):
    return {'id': gid, 'kind': 'rect', 'x_channel': 'FSC-A',
            'y_channel': 'SSC-A', 'x0': 0.0, 'x1': 5000.0, 'y0': 0.0,
            'y1': 5000.0, 'name': name, 'parent_id': None, 'enabled': True}


def _sets(gates, x='FL1-A'):
    """{(quad_set, parent): sorted origins x} of the sets on ``x``."""
    out = {}
    for g in gates.values():
        if g.get('quad_set') and g['x_channel'] == x:
            out.setdefault((g['quad_set'], g['parent_id']), []).append(
                g['quad_origin_x'])
    return {k: sorted(v) for k, v in out.items()}


def test_twin_tests_ignore_which_set_a_member_is_in():
    a, b = _quad('FL1-A', 'FL2-A', 'qs1')[0], _quad('FL1-A', 'FL2-A', 'qs7')[0]
    assert same_gate(a, b) and same_gate_identity(a, b)
    plain = {k: v for k, v in a.items() if k != 'quad_set'}
    assert not same_gate(a, plain) and not same_gate_identity(a, plain)


def test_copy_gates_to_twice_does_not_stack_a_quadrant(ed):
    # Review P1: the first copy took a new id beside the target's own qs1;
    # the second must find it as the same gates.
    _restore(ed, 'synthetic_a2', _quad('FSC-A', 'SSC-A', 'qs1'))
    _restore(ed, 'synthetic_a1', _quad('FL1-A', 'FL2-A', 'qs1'))
    ed._copy_gates_to(['synthetic_a2'])
    first = _sets(ed._sample_gates['synthetic_a2'])
    ed._copy_gates_to(['synthetic_a2'])
    tgt = ed._sample_gates['synthetic_a2']
    assert _sets(tgt) == first and len(_quad_rects(tgt)) == 8
    assert 'qs1' not in {k[0] for k in first}


def _quad_rects(gates):
    return [g for g in gates.values() if g.get('quad_set')]


def test_recopy_after_moving_the_source_updates_in_place(ed):
    # Review P1b: the target's copy follows the source's new centre and
    # keeps its own id.
    _restore(ed, 'synthetic_a2', _quad('FSC-A', 'SSC-A', 'qs1'))
    _restore(ed, 'synthetic_a1', _quad('FL1-A', 'FL2-A', 'qs1'))
    ed._copy_gates_to(['synthetic_a2'])
    ((qs, _p),) = _sets(ed._sample_gates['synthetic_a2'])
    _drag(ed, 'qs1')
    ed._copy_gates_to(['synthetic_a2'])
    assert _sets(ed._sample_gates['synthetic_a2']) == {(qs, None): [1200.0] * 4}


def test_recopy_joins_a_missing_member_to_its_set(ed):
    # One member of the copied quadrant was deleted on the target; copying
    # again brings it back in the same set as the three still there.
    _restore(ed, 'synthetic_a2', _quad('FSC-A', 'SSC-A', 'qs1'))
    _restore(ed, 'synthetic_a1', _quad('FL1-A', 'FL2-A', 'qs1'))
    ed._copy_gates_to(['synthetic_a2'])
    tgt = ed._sample_gates['synthetic_a2']
    ((qs, _p),) = _sets(tgt)
    gone = next(g for g in tgt if tgt[g].get('label', '').startswith('Q--')
                and tgt[g]['x_channel'] == 'FL1-A')
    del tgt[gone]
    ed._sample_gate_order['synthetic_a2'].remove(gone)
    ed._copy_gates_to(['synthetic_a2'])
    assert _sets(tgt) == {(qs, None): [1000.0] * 4}
    assert _ids(tgt, 'FSC-A') == {'qs1'}


def _two_parents_same_axes(ed):
    """Sets one id on FL1-A x FL2-A under two parents (as Copy to tab can
    bring from a session saved before the fix), put in directly."""
    gates = ed._gates
    for g in ([_rect('g1', 'P1'), _rect('g2', 'P2')]
              + _quad('FL1-A', 'FL2-A', 'qs1', first=3, parent='g1')
              + _quad('FL1-A', 'FL2-A', 'qs1', origin=2000.0, first=7,
                      parent='g2')):
        gid = g.pop('id')
        gates[gid] = g
        ed._gate_id_order.append(gid)
    return gates


@pytest.mark.parametrize('kind', ['quad_origin', 'quad_x', 'quad_y'])
def test_drag_moves_only_the_set_under_its_parent(ed, kind):
    gates = _two_parents_same_axes(ed)
    _drag(ed, 'qs1', kind, to=2500.0, parent='g2')
    sets = _sets(gates)
    assert sets[('qs1', 'g1')] == [1000.0] * 4
    if kind != 'quad_y':
        assert sets[('qs1', 'g2')] == [2500.0] * 4


def test_hit_names_the_parent_of_the_set_under_the_cursor(ed):
    _two_parents_same_axes(ed)
    h = SimpleNamespace(cx=2000.0, cy=2000.0, tol_x=5.0, tol_y=5.0,
                        x_ch='FL1-A', y_ch='FL2-A')
    assert ed._hit_quadrants(h) == ('quad_origin', 'qs1', 'g2')
    h.cx = h.cy = 1000.0
    assert ed._hit_quadrants(h) == ('quad_origin', 'qs1', 'g1')
    # A divider away from both centres.
    h.cx, h.cy = 2000.0, 5000.0
    assert ed._hit_quadrants(h) == ('quad_x', 'qs1', 'g2')
    h.cx, h.cy = 5000.0, 2000.0
    assert ed._hit_quadrants(h) == ('quad_y', 'qs1', 'g2')


def test_typed_origin_moves_only_the_set_under_its_parent(ed):
    gates = _two_parents_same_axes(ed)
    typed = {k: t for k, _lab, t in gc.fields(gates['g7'], {})}
    typed['quad_origin_x'] = '1300'
    ed._apply_gate_coordinates('synthetic_a1', 'g7', typed)
    assert _sets(gates)[('qs1', 'g1')] == [1000.0] * 4
    # FL1-A may be shown on a transformed axis: only check it moved.
    assert _sets(gates)[('qs1', 'g2')] != [2000.0] * 4


def test_copy_to_tab_then_drag_leaves_the_copied_set(ed):
    # Review P2: a session saved before the fix, the live gates and a tab
    # each holding a different qs1 on the same axes under different parents.
    p1, p2 = _rect('g1', 'P1'), _rect('g2', 'P2')
    _restore(ed, 'synthetic_a1', [dict(p1), dict(p2)]
             + _quad('FL1-A', 'FL2-A', 'qs1', first=3, parent='g1'))
    ts = ed._tabs()
    idx = ts.new_tab()
    tab = ts.tabs[idx]
    g, o, _m = ed._install_tab_gates(
        'synthetic_a1', [dict(p1), dict(p2)]
        + _quad('FL1-A', 'FL2-A', 'qs1', origin=2000.0, first=7,
                parent='g2'))
    tab.gates, tab.gate_order = {'synthetic_a1': g}, {'synthetic_a1': o}
    ed._copy_gates_to_tab(idx, [('synthetic_a1', 'g1')])
    ed._switch_tab(idx)
    own = next(gg for gg in ed._gates.values()
               if gg.get('quad_set') and gg['quad_origin_x'] == 2000.0)
    _drag(ed, own['quad_set'], to=2500.0, parent=own['parent_id'])
    origins = sorted(gg['quad_origin_x'] for gg in ed._gates.values()
                     if gg.get('quad_set'))
    assert origins == [1000.0] * 4 + [2500.0] * 4


@pytest.mark.parametrize('origin', [1000.0, 1200.0])
def test_copying_part_of_a_set_keeps_the_target_s_set_whole(ed, origin):
    # Review P5 / P6: the same quadrant on both samples (drawn on all
    # shown); the source lost two members. The copy meets the other two as
    # twins (skipped at the same place, updated in place when the source
    # moved), and the target's quadrant stays one set.
    full = _quad('FL1-A', 'FL2-A', 'qs1')
    _restore(ed, 'synthetic_a2', [dict(g) for g in full])
    _restore(ed, 'synthetic_a1',
             [dict(g) for g in _quad('FL1-A', 'FL2-A', 'qs1',
                                     origin=origin)[:2]])
    ed._copy_gates_to(['synthetic_a2'])
    tgt = ed._sample_gates['synthetic_a2']
    assert len(_quad_rects(tgt)) == 4
    assert _ids(tgt, 'FL1-A') == {'qs1'}


def test_hit_finds_a_second_set_with_the_id_on_the_axes(ed):
    # Two sets one id on the same axes under different parents: each
    # shows its own handles (one hit per set, not per id).
    _two_parents_same_axes(ed)
    h = SimpleNamespace(cx=2000.0, cy=2000.0, tol_x=5.0, tol_y=5.0,
                        x_ch='FL1-A', y_ch='FL2-A')
    first = next(iter(ed._gates.values()))
    assert first['parent_id'] is None   # the parent rects come first
    assert ed._hit_quadrants(h)[2] == 'g2'
