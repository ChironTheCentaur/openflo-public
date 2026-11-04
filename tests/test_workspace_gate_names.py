"""The Workspace names a gate the way Statistics and Frequencies do.

Survey item R7: ``workspace_run.gate_path`` (the Workspace item label) said
every gate by its description, bounds and all ('R  Cells  FSC-A x SSC-A
FSC-A in [...]'), while ``gating.population_path`` used the gate's label or
name and fell back to the description only for an unnamed gate. Sky chose
names everywhere. Only the shown text changes: output files are named by
gate id, and a saved recipe keeps the label it was saved with.
"""
from __future__ import annotations

from openflo.gate_text import describe_gate
from openflo.gating import population_path
from openflo.workspace_run import (
    _item_from_json,
    _item_to_json,
    build_drop_payload,
    gate_path,
    run_label,
)


def _gates():
    """Synthetic chain: labelled root -> named rect -> unnamed threshold."""
    return {
        'g1': {'kind': 'rect', 'label': 'Lymphs', 'name': 'R1',
               'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
               'x0': 1.0, 'x1': 2.0, 'y0': 3.0, 'y1': 4.0, 'parent_id': None},
        'g2': {'kind': 'rect', 'name': 'Singlets',
               'x_channel': 'FSC-A', 'y_channel': 'FSC-H',
               'x0': 5.0, 'x1': 6.0, 'y0': 7.0, 'y1': 8.0, 'parent_id': 'g1'},
        'g3': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5,
               'parent_id': 'g2'},
    }


def test_gate_path_uses_label_then_name_then_bounds():
    g = _gates()
    unnamed = describe_gate(g['g3'])
    assert unnamed and 'FL1-A' in unnamed
    assert gate_path(g, 'g1') == 'Lymphs'               # label beats name
    assert gate_path(g, 'g2') == 'Lymphs / Singlets'
    assert gate_path(g, 'g3') == f'Lymphs / Singlets / {unnamed}'


def test_gate_path_matches_population_path_names():
    g = _gates()
    for gid in g:
        assert gate_path(g, gid).split(' / ') == population_path(g, gid).split('/')


def test_gate_path_names_an_unknown_kind_as_population_path_does():
    g = {'gx': {'kind': 'mystery', 'parent_id': None}}
    assert gate_path(g, 'gx') == population_path(g, 'gx') == describe_gate(g['gx'])


def _ctx():
    return {'name': 'S1', 'trial': 'T1', 'path': 'S1.fcs', 'gates': _gates(),
            'comp_matrix': None, 'comp_channels': []}


def test_drop_payload_label_names_gates():
    p = build_drop_payload(_ctx(), 'g3')
    assert p['gate_path'].startswith('Lymphs / Singlets / ')
    assert 'FSC-A in' not in p['gate_path']             # no bounds for named gates


def test_output_names_and_saved_recipes_unchanged():
    p = build_drop_payload(_ctx(), 'g3')
    # Output files are named by gate id, not by the shown path.
    assert run_label(p) == 'T1_S1_g3'
    # A recipe saved before this change keeps its own label on reload.
    old = dict(p, gate_path='R  Lymphs  FSC-A x SSC-A ... / T  FL1-A >= 0.5')
    assert _item_from_json(_item_to_json(old))['gate_path'] == old['gate_path']
