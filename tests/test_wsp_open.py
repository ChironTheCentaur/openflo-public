"""wsp_open: reading a FlowJo workspace into per-sample records, putting
its gates on a sample's units, and installing a gate list with fresh ids --
without a window.

Workspaces are written by OpenFlo's own writer from generated FCS files;
generic channel names only; no Tk.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from openflo import gate_store
from openflo import wsp_open as wo
from tests._open_fixtures import SPILL, make_fcs, session_gates, write_wsp


@pytest.fixture
def run(tmp_path):
    """Two samples beside a workspace: 'a' in group Stim with its own
    matrix, 'b' in no group with none; the interval gate names FL2-A
    without 'Comp-'."""
    a = make_fcs(tmp_path / 'run' / 'a.fcs', seed=1)
    b = make_fcs(tmp_path / 'run' / 'b.fcs', seed=2)
    wsp = write_wsp(tmp_path / 'run' / 'x.wsp',
                    [('a', a, 'Stim', True), ('b', b, None, False)],
                    uncompensated=True)
    return {'wsp': wsp, 'a': a, 'b': b, 'tmp': tmp_path}


def test_each_found_sample_brings_its_gates_group_and_matrix(run):
    plan = wo.plan_wsp(run['wsp'])
    a, b = plan.samples
    assert (a.node_name, a.fcs_path, a.groups) == ('a', run['a'], ['Stim'])
    assert (b.node_name, b.fcs_path, b.groups) == ('b', run['b'], [])
    assert a.comp['source'] == 'wsp:Acquisition-defined'
    assert a.comp['channels'] == ['FL1-A', 'FL2-A']
    np.testing.assert_allclose(a.comp['matrix'], SPILL)
    assert b.comp is None
    assert [g['label'] for g in a.gates] == ['Cells', 'FL1+', 'FL2 mid',
                                             'both']
    assert plan.imported == a.gates + b.gates
    assert not (plan.unresolved or plan.mismatched or plan.unread
                or plan.ambiguous)


def test_the_locator_is_the_callers(run, tmp_path):
    """A caller that keeps the files elsewhere (a phone's app storage)
    finds them itself; a file it cannot find is named, not guessed."""
    store = tmp_path / 'store'
    store.mkdir()
    moved = str(store / 'copy_of_a.fcs')
    os.replace(run['a'], moved)

    class Finder:
        def __init__(self):
            self.asked = []

        def find(self, uri):
            self.asked.append(os.path.basename(uri or ''))
            return moved if uri and uri.endswith('a.fcs') else None

    finder = Finder()
    plan = wo.plan_wsp(run['wsp'], finder)
    assert finder.asked == ['a.fcs', 'b.fcs']
    assert [s.fcs_path for s in plan.samples] == [moved]
    assert plan.unresolved == ['b']


def test_a_plain_function_can_be_the_locator(run):
    found = {'a.fcs': run['a']}

    def find(uri):
        return found.get(os.path.basename(uri or ''))

    plan = wo.plan_wsp(run['wsp'], find)
    assert [s.fcs_path for s in plan.samples] == [run['a']]
    assert plan.unresolved == ['b'] and plan.ambiguous == []


def test_time_gates_use_the_timestep_of_the_file_found(tmp_path):
    """FlowJo writes Time gates in seconds; they come back in ticks with the
    $TIMESTEP of the FCS file the locator found, not of a workspace keyword
    or of a uri that no longer exists."""
    from openflo.wsp_writer import WspWriter
    from tests._fcs_writer import write_float_fcs
    chans = ['FSC-A', 'SSC-A', 'Time']
    rng = np.random.default_rng(5)
    ev = np.column_stack([rng.uniform(2e4, 2e5, 300),
                          rng.uniform(1e4, 1e5, 300),
                          np.arange(300, dtype=float) * 10.0])
    (tmp_path / 'run').mkdir()
    fcs = write_float_fcs(tmp_path / 'run' / 't.fcs', ev, chans,
                          metadata_dict={'TIMESTEP': '0.01'})
    w = WspWriter()
    w.add_sample(name='t', fcs_path=fcs, channels=chans,
                 gates=[{'id': 'g1', 'parent_id': None, 'kind': 'interval',
                         'channel': 'Time', 'lo': 300.0, 'hi': 2500.0,
                         'name': 'steady'}])
    wsp = str(tmp_path / 'run' / 't.wsp')
    w.write(wsp)
    text = open(wsp, encoding='utf-8').read()
    assert 'name="$TIMESTEP" value="0.01"' in text
    with open(wsp, 'w', encoding='utf-8') as fh:     # a keyword FlowJo ignores
        fh.write(text.replace('name="$TIMESTEP" value="0.01"',
                              'name="$TIMESTEP" value="0.05"'))
    moved = str(tmp_path / 'store' / 'copy.fcs')
    os.makedirs(os.path.dirname(moved))
    os.replace(fcs, moved)
    (s,) = wo.plan_wsp(wsp, lambda uri: moved).samples
    (g,) = s.gates
    assert (g['lo'], g['hi']) == pytest.approx((300.0, 2500.0))


def test_a_file_with_other_events_than_the_workspace_is_named(run):
    make_fcs(run['b'], seed=9, n=450)
    plan = wo.plan_wsp(run['wsp'])
    assert [s.node_name for s in plan.samples] == ['a', 'b']
    ((name, why),) = plan.mismatched
    assert name == 'b' and why


def test_the_default_locator_reports_a_name_found_twice(run):
    os.replace(run['a'], os.path.join(os.path.dirname(run['a']), 'gone'))
    for sub in ('p1', 'p2'):
        make_fcs(os.path.join(os.path.dirname(run['wsp']), sub, 'a.fcs'))
    with open(run['wsp'], encoding='utf-8') as fh:
        text = fh.read()
    with open(run['wsp'], 'w', encoding='utf-8') as fh:
        fh.write(text.replace('/run/a.fcs', '/elsewhere/a.fcs'))
    plan = wo.plan_wsp(run['wsp'])
    assert plan.unresolved == ['a']
    assert plan.ambiguous == [('a.fcs', 2)]


def test_an_unreadable_workspace_says_why(tmp_path):
    bad = tmp_path / 'bad.wsp'
    bad.write_text('<Workspace><SampleList>', encoding='utf-8')
    with pytest.raises(wo.WspUnreadable) as err:
        wo.plan_wsp(str(bad))
    assert str(err.value)


# ── gates on a sample ───────────────────────────────────────────────────────

def _loaded(path, comp=None):
    from openflo.pipeline import FlowSample
    s = FlowSample(path)
    if comp is not None:
        s.manual_compensate(comp['matrix'], comp['channels'],
                            source=comp['source'])
    s.apply_transform()
    return s


def test_gates_go_onto_the_samples_stored_units(run):
    """FlowJo's "FL1-A > 2000" holds, on the stored (transformed) values,
    the events whose linear value is above 2000."""
    from openflo.channel_transforms import to_linear
    plan = wo.plan_wsp(run['wsp'])
    a = plan.samples[0]
    s = _loaded(run['a'], a.comp)
    stored = dict(s.data_transforms)
    assert stored.get('FL1-A') not in (None, 'linear')
    gates, _unc = wo.gates_on_sample(a.gates, stored, s)
    thr = gates[1]
    fl1 = s.data['FL1-A'].to_numpy()
    linear = np.asarray(to_linear(fl1, stored['FL1-A']))
    assert 0 < (linear > 2000).sum() < len(fl1)
    assert (fl1 > thr['value']).sum() == (linear > 2000).sum()
    assert a.gates[1]['value'] == 2000.0        # the plan is not changed
    assert gates[0]['x0'] == a.gates[0]['x0']   # linear scatter: as is


def test_a_gate_on_the_uncompensated_parameter_is_reported(run):
    plan = wo.plan_wsp(run['wsp'])
    a, b = plan.samples
    _g, unc = wo.gates_on_sample(a.gates, {}, _loaded(run['a'], a.comp))
    assert unc == [('FL2 mid', 'FL2-A')]
    # A sample holding FL2-A uncompensated counts what FlowJo counts.
    _g, unc_b = wo.gates_on_sample(b.gates, {}, _loaded(run['b']))
    assert unc_b == []


def test_a_gate_on_a_channel_of_unknown_scale_is_marked(run):
    a = wo.plan_wsp(run['wsp']).samples[0]
    gates, _ = wo.gates_on_sample(
        a.gates, {'FL1-A': {'method': 'unknown'}}, _loaded(run['a']))
    assert gates[1].get('scale_unknown') is True
    assert not gates[0].get('scale_unknown')


# ── installing a gate list ──────────────────────────────────────────────────

def _install(pending, gates=None, order=None, fill=None):
    gates = {} if gates is None else gates
    order = [] if order is None else order
    seqs = {'s': len(order)}
    id_map = gate_store.install_gates(
        gates, order, pending, lambda: gate_store.next_gate_id(seqs, 's'),
        fill=fill)
    return gates, order, id_map, seqs


def test_a_workspace_tree_gets_fresh_ids_parents_and_operands(run):
    a = wo.plan_wsp(run['wsp']).samples[0]
    gates, order, id_map, seqs = _install(a.gates)
    assert order == ['g1', 'g2', 'g3', 'g4'] and seqs['s'] == 4
    assert id_map == {'imp_0': 'g1', 'imp_1': 'g2', 'imp_2': 'g3',
                      'imp_3': 'g4'}
    assert gates['g2']['parent_id'] == 'g1'
    assert gates['g4']['operands'] == ['g2', 'g3']
    assert all(g['enabled'] is True for g in gates.values())
    assert all('_import_id' not in g for g in gates.values())
    assert '_import_id' in a.gates[0]           # the input is not changed


def test_a_saved_list_keeps_its_flags_and_remaps_shifted_ids():
    """Saved ids g2, g5, g7 come back as the next free ids; the boolean's
    operands follow them."""
    gates, order, id_map, _ = _install(
        session_gates('A'), gates={'g1': {'kind': 'rect'}}, order=['g1'])
    assert id_map == {'g2': 'g2', 'g5': 'g3', 'g7': 'g4'}
    assert order == ['g1', 'g2', 'g3', 'g4']
    assert gates['g3']['parent_id'] == 'g2'
    assert gates['g3']['enabled'] is False
    assert gates['g4']['operands'] == ['g3', 'g2']
    assert gates['g2']['color'] == '#aa0000'
    assert all('id' not in g for k, g in gates.items() if k != 'g1')


def test_sibling_one_d_gates_on_a_channel_are_all_kept():
    pending = [{'id': f'x{i}', 'parent_id': None, 'kind': 'interval',
                'channel': 'FL1-A', 'lo': i, 'hi': i + 1}
               for i in range(3)]
    gates, order, _, _ = _install(pending)
    assert len(order) == 3 and [gates[g]['lo'] for g in order] == [0, 1, 2]


def test_a_parent_after_its_child_and_a_lost_operand_are_dropped():
    pending = [{'id': 'c', 'parent_id': 'p', 'kind': 'threshold',
                'channel': 'FL1-A', 'value': 1.0},
               {'id': 'p', 'parent_id': None, 'kind': 'threshold',
                'channel': 'FL2-A', 'value': 1.0},
               {'id': 'b', 'parent_id': None, 'kind': 'boolean', 'op': 'not',
                'operands': ['gone']}]
    gates, _, _, _ = _install(pending)
    assert gates['g1']['parent_id'] is None
    assert gates['g3']['operands'] != ['gone']


def test_the_callers_defaults_fill_each_gate_before_it_is_added():
    seen = []

    def fill(g):
        seen.append(dict(g))
        g.setdefault('color', '#777777')
        g.setdefault('parent_id', None)

    pending = [{'id': 'a', 'kind': 'threshold', 'channel': 'FL1-A',
                'value': 1.0, 'color': '#111111'},
               {'id': 'b', 'kind': 'threshold', 'channel': 'FL2-A',
                'value': 1.0}]
    gates, _, _, _ = _install(pending, fill=fill)
    assert [s.get('enabled') for s in seen] == [True, True]
    assert 'id' not in seen[0]
    assert gates['g1']['color'] == '#111111'
    assert gates['g2']['color'] == '#777777'
