"""Importing a workspace must keep the population names.

A gate's name lives on the enclosing `<Population name="...">` element, not on
the `<Gate>` inside it. `walk_population` parsed the gate and never read the
name, so every label in the file was discarded: a 40-population FlowJo
workspace imported fully unnamed, and nothing downstream put the names back.

The writer was never at fault — it emits the names correctly, including
quoting and unicode — which is why a round trip through this codebase looked
like it worked as long as nobody checked what the gates were called.
"""
import numpy as np
import pytest

import openflo.pipeline as fp

NAMED_GATES = [
    {'kind': 'threshold', 'channel': 'FSC-A', 'value': 1000.0,
     'id': 'g1', 'parent_id': None, 'name': 'Live'},
    {'kind': 'interval', 'channel': 'SSC-A', 'lo': 10.0, 'hi': 900.0,
     'id': 'g2', 'parent_id': 'g1', 'name': 'CD3+ / CD4+'},
    {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
     'x0': 1.0, 'x1': 9.0, 'y0': 1.0, 'y1': 9.0,
     'id': 'g3', 'parent_id': None, 'name': 'αβ "T" cells'},
    {'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
     'vertices': [[1.0, 1.0], [5.0, 1.0], [5.0, 5.0]],
     'id': 'g4', 'parent_id': None, 'name': 'Tregsµ'},
]


@pytest.fixture
def workspace(tmp_path):
    writer = fp.WspWriter(cytometer='name-test')
    writer.add_sample('s', fcs_path='', channels=['FSC-A', 'SSC-A'],
                      gates=NAMED_GATES)
    out = tmp_path / 'named.wsp'
    writer.write(str(out))
    return str(out)


def _names(gates):
    return {g.get('name') or g.get('label') for g in gates}


def test_the_writer_puts_the_names_in_the_file(workspace):
    """Establish that the names are there to be read — otherwise the test
    below would pass for the wrong reason."""
    import re
    raw = open(workspace, encoding='utf-8').read()
    written = set(re.findall(r'Population name="([^"]*)"', raw))
    assert 'Live' in written and 'CD3+ / CD4+' in written, written


def test_every_population_name_survives_the_import(workspace):
    back, _ = fp.read_template_gates(workspace)
    assert len(back) == len(NAMED_GATES)
    assert _names(back) == {g['name'] for g in NAMED_GATES}


def test_no_gate_comes_back_unnamed(workspace):
    back, _ = fp.read_template_gates(workspace)
    unnamed = [g for g in back if not (g.get('name') or g.get('label'))]
    assert not unnamed, (
        f'{len(unnamed)} of {len(back)} imported gates have no name: '
        f'{[g.get("kind") for g in unnamed]}')


@pytest.mark.parametrize('name', [
    pytest.param('CD3+ / CD4+', id='slash-and-plus'),
    pytest.param('αβ "T" cells', id='quotes-and-greek'),
    pytest.param('Tregsµ', id='micro-sign'),
])
def test_awkward_names_survive_intact(workspace, name):
    """Characters that need XML escaping must come back byte-for-byte."""
    back, _ = fp.read_template_gates(workspace)
    assert name in _names(back)


def test_the_hierarchy_is_still_correct(workspace):
    """Reading the name must not disturb the parent chain."""
    back, _ = fp.read_template_gates(workspace)
    by_name = {(g.get('name') or g.get('label')): g for g in back}
    child = by_name['CD3+ / CD4+']
    parent_id = child.get('parent_id')
    assert parent_id is not None, 'the nested gate lost its parent'
    # read_template_gates remaps the temporary _import_id onto 'id'.
    parents = [g for g in back if g.get('id') == parent_id]
    assert len(parents) == 1, f'parent {parent_id!r} not found among the gates'
    assert (parents[0].get('name') or parents[0].get('label')) == 'Live'


def test_the_geometry_is_unchanged(workspace):
    """The names must not have come at the cost of the numbers."""
    back, _ = fp.read_template_gates(workspace)
    by_name = {(g.get('name') or g.get('label')): g for g in back}
    poly = by_name['Tregsµ']
    np.testing.assert_allclose(np.asarray(poly['vertices'], dtype=float),
                               np.asarray(NAMED_GATES[3]['vertices'],
                                          dtype=float))
    assert by_name['Live']['value'] == pytest.approx(1000.0)
