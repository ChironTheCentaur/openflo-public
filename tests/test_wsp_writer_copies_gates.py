"""WspWriter works on its own copies of the gates it is given (scope audit L2).

`add_sample` kept the caller's gate dicts and `write` added an ``id`` to each
one (`_emit_gate_tree`), so a library user passing `WspReader.extract_gates()`
output found their gates edited, and a change made to a gate between
`add_sample()` and `write()` went into the file. The GUI export builds fresh
dicts for each export and was not affected.
"""
import copy
import xml.etree.ElementTree as ET

import pytest

from openflo.wsp_writer import WspWriter


def _gates():
    return [
        {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
         'parent_id': None, '_import_id': 'imp1', 'name': 'SampleA_pop'},
        {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>', 'value': 0.5,
         'parent_id': 'imp1', 'name': 'child'},
    ]


def _population_names(path):
    return [p.get('name') for p in ET.parse(path).iter('Population')]


def _write(tmp_path, gates, **kw):
    w = WspWriter(cytometer='Generic')
    w.add_sample('SampleA', '', ['FL1-A', 'FL2-A'], gates, **kw)
    out = tmp_path / 'x.wsp'
    w.write(str(out))
    return w, out


def test_the_callers_gates_are_left_as_they_were(tmp_path):
    gates = _gates()
    before = copy.deepcopy(gates)
    _write(tmp_path, gates)
    assert gates == before                 # no 'id' added, nothing changed


def test_the_callers_gates_are_left_when_coordinates_are_converted(
        tmp_path):
    """With a transform the coordinates go to linear units on the way out;
    the caller's stored (transformed) values stay."""
    gates = _gates()
    before = copy.deepcopy(gates)
    _write(tmp_path, gates, transforms={'FL1-A': 'logicle',
                                        'FL2-A': 'logicle'})
    assert gates == before


def test_the_writer_holds_its_own_dicts(tmp_path):
    gates = _gates()
    w = WspWriter(cytometer='Generic')
    w.add_sample('SampleA', '', ['FL1-A', 'FL2-A'], gates)
    held = w.samples[0]['gates']
    assert all(h is not g for h, g in zip(held, gates, strict=True))
    assert held[0]['vertices'] is not gates[0]['vertices']


def _with_ids(gates):
    """The GUI and CLI style: each gate carries its own id."""
    for k, g in enumerate(gates, start=1):
        g['id'] = f'g{k}'
    gates[1]['parent_id'] = 'g1'
    return gates


@pytest.mark.parametrize('make', [_gates, lambda: _with_ids(_gates())],
                         ids=['reader-style', 'with-ids'])
def test_an_edit_after_add_sample_does_not_reach_the_file(tmp_path, make):
    gates = make()
    w = WspWriter(cytometer='Generic')
    w.add_sample('SampleA', '', ['FL1-A', 'FL2-A'], gates)
    gates[1]['name'] = 'renamed later'
    gates[0]['vertices'][0][0] = 0.7
    out = tmp_path / 'x.wsp'
    w.write(str(out))
    names = _population_names(out)
    assert 'child' in names and 'renamed later' not in names
    assert w.samples[0]['gates'][0]['vertices'][0][0] == 0.1


def test_a_reader_style_list_still_nests_its_child(tmp_path):
    """Gates that carry only the reader's _import_id still build the tree:
    the child is written inside its parent's population."""
    _w, out = _write(tmp_path, _gates())
    pops = {p.get('name'): p for p in ET.parse(out).iter('Population')}
    inner = [p.get('name') for p in pops['SampleA_pop'].iter('Population')]
    assert 'child' in inner
