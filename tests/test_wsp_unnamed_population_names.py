"""An unnamed gate exports under a name that is the same in every sample.

The writer used to fall back to the editor's gate id for a gate the user had
not named. Ids are per sample: a sample that once held a gate that was later
deleted numbers its next gates one higher, so the same lymphocyte gate came
out as 'g2' in one sample and 'g3' in the next. FlowJo ties a population
across samples by name, so its tables and group gates did not line up.
"""
import re

import pytest

import openflo.pipeline as fp


def _gates(root_id, child_id):
    return [
        {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'x0': 1.0, 'x1': 9.0, 'y0': 1.0, 'y1': 9.0,
         'id': root_id, 'parent_id': None},
        {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': [[1.0, 1.0], [5.0, 1.0], [5.0, 5.0]],
         'id': child_id, 'parent_id': root_id},
    ]


def _write(tmp_path, samples):
    writer = fp.WspWriter(cytometer='name-test')
    for name, gates in samples.items():
        writer.add_sample(name, fcs_path='',
                          channels=['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A'],
                          gates=gates)
    out = tmp_path / 'unnamed.wsp'
    writer.write(str(out))
    return out.read_text(encoding='utf-8')


def _names_per_sample(raw):
    out = {}
    for m in re.finditer(r'<SampleNode name="([^"]+)".*?</SampleNode>', raw,
                         re.S):
        out[m.group(1)] = re.findall(r'<Population name="([^"]*)"',
                                     m.group(0))
    return out


def test_same_gate_same_name_across_samples(tmp_path):
    raw = _write(tmp_path, {'s1': _gates('g2', 'g3'),
                            's2': _gates('g1', 'g2')})
    names = _names_per_sample(raw)
    assert names['s1'] == names['s2'] == ['FSC-A x SSC-A', 'FL1-A x FL2-A']


def test_no_population_is_named_for_an_editor_id(tmp_path):
    raw = _write(tmp_path, {'s1': _gates('g2', 'g3')})
    written = re.findall(r'<Population name="([^"]*)"', raw)
    assert not [n for n in written if re.fullmatch(r'_?g\d+', n)], written


def test_siblings_on_the_same_axes_get_unique_names(tmp_path):
    a, b = _gates('g1', 'g2')
    twin = dict(a, id='g3', x0=2.0)
    raw = _write(tmp_path, {'s1': [a, b, twin]})
    written = re.findall(r'<Population name="([^"]*)"', raw)
    assert written.count('FSC-A x SSC-A') == 1
    assert 'FSC-A x SSC-A (2)' in written


def test_a_user_name_still_wins(tmp_path):
    a, b = _gates('g1', 'g2')
    a['name'] = 'Cells'
    raw = _write(tmp_path, {'s1': [a, b]})
    assert '<Population name="Cells"' in raw


@pytest.mark.parametrize('gate, expected', [
    ({'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.0}, 'FL1-A'),
    ({'kind': 'boolean', 'op': 'and', 'operands': []}, 'Boolean'),
])
def test_one_d_and_other_gates_have_a_readable_name(gate, expected):
    assert fp.wsp_population_name(gate) == expected


def test_the_names_come_back_on_import(tmp_path):
    _write(tmp_path, {'s1': _gates('g2', 'g3')})
    back, _ = fp.read_template_gates(str(tmp_path / 'unnamed.wsp'))
    assert {g.get('name') or g.get('label') for g in back} == {
        'FSC-A x SSC-A',
        'FL1-A x FL2-A'}
