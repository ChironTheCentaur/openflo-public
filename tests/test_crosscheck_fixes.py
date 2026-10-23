"""Bugs found by checking OpenFlo against an independent reader on public
FlowJo and Gating-ML files. Each test rebuilds the case on generated data
(channels FL1-A, FL2-A, ...) and computes what it expects with numpy alone:

- openflo-compare left out FlowJo's AND / OR / NOT populations and
  everything under them, and named populations by their own name only;
- a spillover CSV whose header numpy wrote ('# FL1-A,...') lost its first
  channel to the '#', and that channel was not compensated;
- index-sort locations split over numbered keywords (_1 .. _16) read as none;
- a Gating-ML NOT over several gates was written as NOT of the first;
- a Gating-ML boolean could name a gate left out of the file;
- a Gating-ML gate named by its matrix's fluorochrome read no channel;
- gates nested in a non-standard element were skipped without a word;
- the .wsp export said flowJoVersion "OpenFlo-export-1.0", had no
  <Transformations> and named boolean operands '../X', so other readers
  (FlowKit) refused it;
- the note for an FCS whose HEADER and TEXT disagree on where the data ENDS
  said "starts".
"""
from __future__ import annotations

import csv
import types
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp
from openflo import index_sort as ix
from openflo.compare import (
    _compare_one_sample,
    _per_sample_inventory,
    _write_csv,
)
from openflo.fcs_export import write_fcs
from openflo.gatingml import read_gatingml, write_gatingml


def _frame(n=3000, seed=7):
    rng = np.random.default_rng(seed)

    def pos(p):
        return np.where(rng.random(n) < p, rng.uniform(20, 90, n),
                        rng.uniform(0, 10, n))
    return pd.DataFrame({'FSC-A': rng.uniform(1e3, 1e5, n),
                         'FL1-A': pos(0.4), 'FL2-A': pos(0.3),
                         'FL3-A': pos(0.5)})


# ── openflo-compare: boolean populations and paths ─────────────────────────

def _rect(name, ch, lo, count=0, kids=''):
    sub = f'<Subpopulations>{kids}</Subpopulations>' if kids else ''
    return (f'<Population name="{name}" count="{count}"><Gate><RectangleGate>'
            f'<dimension min="{lo}"><fcs-dimension name="{ch}"/></dimension>'
            f'</RectangleGate></Gate>{sub}</Population>')


def _bool(tag, name, deps, kids=''):
    sub = f'<Subpopulations>{kids}</Subpopulations>' if kids else ''
    d = ''.join(f'<Dependent name="{x}"/>' for x in deps)
    return (f'<{tag} name="{name}" count="0"><Dependents>{d}</Dependents>'
            f'{sub}</{tag}>')


_COMPARE_WSP = ('<Workspace><SampleList><Sample><DataSet uri="file:///x/s.fcs"/>'
                '<SampleNode name="S1"><Subpopulations>'
                + _rect('A', 'FL1-A', 50, kids=_rect('X', 'FL3-A', 50))
                + _rect('B', 'FL2-A', 50, kids=_rect('X', 'FL3-A', 50))
                # FlowJo 10's form: each operand by its path from the sample.
                + _bool('OrNode', 'A or B', ['A', 'B'])
                # OpenFlo's earlier exports: relative to the boolean.
                + _bool('AndNode', 'A and B', ['../A', '../B'],
                        kids=_rect('hi', 'FL3-A', 50))
                + _bool('NotNode', 'not A', ['A'])
                + _bool('NotNode', 'lost', ['Nowhere'], kids=_rect('under', 'FL3-A', 50))
                + '</Subpopulations></SampleNode></Sample></SampleList></Workspace>')


def test_compare_counts_boolean_populations_by_path(tmp_path):
    df = _frame()
    fcs = tmp_path / 's.fcs'
    write_fcs(df, str(fcs))
    reader = types.SimpleNamespace(root=ET.fromstring(_COMPARE_WSP),
                                   _channel_name=fp.WspReader._channel_name)
    (_name, _uri, pops), = _per_sample_inventory(reader)
    rows = {r['path']: r for r in _compare_one_sample('S1', str(fcs), pops)}

    a, b, c = (df['FL1-A'] >= 50), (df['FL2-A'] >= 50), (df['FL3-A'] >= 50)
    want = {'A': a, 'A/X': a & c, 'B': b, 'B/X': b & c, 'A or B': a | b,
            'A and B': a & b, 'A and B/hi': a & b & c, 'not A': ~a}
    for path, mask in want.items():
        assert rows[path]['openflo_count'] == int(mask.sum()), path
        assert not rows[path]['error'], path
    # Same name under different parents: two rows, told apart by the path.
    assert rows['A/X']['population'] == rows['B/X']['population'] == 'X'
    assert rows['A/X']['openflo_count'] != rows['B/X']['openflo_count']
    # An operand the workspace does not have: no number, a reason, and the
    # population under it says so too.
    for path in ('lost', 'lost/under'):
        assert rows[path]['openflo_count'] is None, path
        assert "'Nowhere' not found" in rows[path]['error'], path
    out = tmp_path / 'r.csv'
    _write_csv(list(rows.values()), str(out))
    got = list(csv.DictReader(open(out, encoding='utf-8')))
    assert 'path' in got[0] and {r['path'] for r in got} == set(rows)


# ── Spillover CSV with numpy's '#' header ────────────────────────────────────

def test_numpy_header_keeps_the_first_channel_compensated(tmp_path):
    true = _frame()[['FL1-A', 'FL2-A', 'FL3-A']].to_numpy()
    spill = np.array([[1.0, 0.20, 0.05], [0.10, 1.0, 0.15], [0.02, 0.08, 1.0]])
    measured = true @ spill
    fcs = tmp_path / 'm.fcs'
    write_fcs(pd.DataFrame(measured, columns=['FL1-A', 'FL2-A', 'FL3-A']),
              str(fcs))
    path = tmp_path / 'spill.csv'
    np.savetxt(path, spill, delimiter=',', header='FL1-A,FL2-A,FL3-A')
    assert path.read_text().startswith('# FL1-A,')       # numpy's own form
    channels, matrix = fp.read_compensation_matrix(str(path))
    assert channels == ['FL1-A', 'FL2-A', 'FL3-A']
    s = fp.FlowSample(str(fcs)).manual_compensate(matrix, channels)
    np.testing.assert_allclose(s.data[['FL1-A', 'FL2-A', 'FL3-A']].to_numpy(),
                               true, rtol=1e-5, atol=1e-3)


def test_numpy_header_names_the_columns_of_an_events_csv(tmp_path):
    path = tmp_path / 'ev.csv'
    np.savetxt(path, np.array([[1.5, 2.5], [3.5, 4.5]]), delimiter=',',
               header='FL1-A,FL2-A')
    df, notes = fp.read_events_csv(str(path))
    assert list(df.columns) == ['FL1-A', 'FL2-A'] and len(df) == 2
    assert any("'#' line" in n for n in notes)


# ── Index sort over numbered keywords ────────────────────────────────────────

def test_index_sort_locations_split_over_numbered_keywords():
    wells = [(r, c) for r in range(4) for c in range(12)]        # 48 events
    chunks = [wells[i:i + 4] for i in range(0, len(wells), 4)]   # _1 .. _12
    meta = {f'INDEX SORTING LOCATIONS_{k}': ''.join(f'{r},{c};' for r, c in ch)
            for k, ch in enumerate(chunks, 1)}
    # Keyword order in the file is no guide: _10 sorts before _2 as text.
    meta = dict(sorted(meta.items()))
    s = types.SimpleNamespace(name='sort_1', metadata=meta,
                              data=pd.DataFrame({'FL1-A': np.arange(48.0)}))
    assert ix.locations_of(s) == wells
    rows, cols = ix.event_wells(s)
    assert list(zip(rows.tolist(), cols.tolist(), strict=True)) == wells


# ── Gating-ML export: NOT over several gates, dangling booleans ─────────────

def _gml_counts(path, df):
    gates, warnings = read_gatingml(str(path))
    by_id = {g['_import_id']: g for g in gates}
    return ({g['label']: int(np.asarray(fp.cumulative_gate_mask(
        by_id, g['_import_id'], df)).sum()) for g in gates}, warnings)


def test_gatingml_not_over_several_gates(tmp_path):
    df = _frame()
    gates = [{'id': 'a', 'kind': 'threshold', 'channel': 'FL1-A', 'value': 50.0,
              'parent_id': None, 'name': 'A'},
             {'id': 'b', 'kind': 'threshold', 'channel': 'FL2-A', 'value': 50.0,
              'parent_id': None, 'name': 'B'},
             {'id': 'n', 'kind': 'boolean', 'op': 'not', 'operands': ['a', 'b'],
              'parent_id': None, 'name': 'neither'}]
    by_id = {g['id']: g for g in gates}
    want = int((~((df['FL1-A'] >= 50) | (df['FL2-A'] >= 50))).sum())
    assert int(np.asarray(fp.cumulative_gate_mask(by_id, 'n', df)).sum()) == want
    out = tmp_path / 'not.xml'
    write_gatingml(str(out), gates)
    counts, _w = _gml_counts(out, df)
    assert counts['neither'] == want
    assert want != int((df['FL1-A'] < 50).sum())        # NOT of A alone differs


def test_gatingml_boolean_on_a_gate_left_out_is_left_out(tmp_path):
    gates = [{'id': 'a', 'kind': 'threshold', 'channel': 'FL1-A', 'value': 50.0,
              'parent_id': None, 'name': 'A'},
             {'id': 'k', 'kind': 'cluster', 'parent_id': None, 'name': 'K'},
             {'id': 'ab', 'kind': 'boolean', 'op': 'and', 'operands': ['a', 'k'],
              'parent_id': None, 'name': 'A and K'},
             {'id': 'nab', 'kind': 'boolean', 'op': 'or', 'operands': ['a', 'ab'],
              'parent_id': None, 'name': 'A or AK'},
             {'id': 'kid', 'kind': 'threshold', 'channel': 'FL2-A', 'value': 50.0,
              'parent_id': 'ab', 'name': 'under AK'}]
    out = tmp_path / 'dangling.xml'
    warnings = write_gatingml(str(out), gates)
    assert any("'A and K' left out" in w and "'K'" in w for w in warnings)
    assert any("'A or AK' left out" in w for w in warnings)
    assert any("'under AK' left out" in w for w in warnings)
    root = ET.parse(out).getroot()
    ids = {v for el in root.iter() for k, v in el.attrib.items()
           if k.endswith('}id')}
    refs = [v for el in root.iter() for k, v in el.attrib.items()
            if k.endswith('}ref') or k.endswith('}parent_id')]
    assert refs == [] or set(refs) <= ids
    counts, _w = _gml_counts(out, _frame())
    assert set(counts) == {'A'}


# ── Gating-ML import: fluorochrome names, nested gates ───────────────────────

_GML_HEAD = ('<gating:Gating-ML '
             'xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" '
             'xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/transformations" '
             'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">')


def test_gatingml_gate_named_by_its_matrix_fluorochrome(tmp_path):
    def dims(names):
        return ''.join(f'<data-type:fcs-dimension data-type:name="{n}"/>'
                       for n in names)
    xml = (_GML_HEAD
           + '<transforms:spectrumMatrix transforms:id="M1">'
           + f'<transforms:fluorochromes>{dims(["Fluor1", "Fluor2"])}</transforms:fluorochromes>'
           + f'<transforms:detectors>{dims(["FL1-A", "FL2-A"])}</transforms:detectors>'
           + ''.join('<transforms:spectrum>' + ''.join(
               f'<transforms:coefficient transforms:value="{v}"/>' for v in row)
               + '</transforms:spectrum>' for row in ((1, 0.1), (0.2, 1)))
           + '</transforms:spectrumMatrix>'
           + '<gating:RectangleGate gating:id="R">'
           + '<gating:dimension gating:compensation-ref="M1" gating:min="50">'
           + '<data-type:fcs-dimension data-type:name="Fluor2"/></gating:dimension>'
           + '</gating:RectangleGate></gating:Gating-ML>')
    path = tmp_path / 'm.xml'
    path.write_text(xml, encoding='utf-8')
    (g,), _w = read_gatingml(str(path))
    assert g['channel'] == 'FL2-A'
    assert g['wsp_comp_channels'] == ['FL2-A']
    df = _frame()
    assert int(np.asarray(fp.gate_to_mask(g, df)).sum()) == int((df['FL2-A'] >= 50).sum())


def test_gatingml_gates_in_a_non_standard_element_are_named(tmp_path):
    xml = (_GML_HEAD + '<group><gating:RectangleGate gating:id="R">'
           + '<gating:dimension gating:min="50">'
           + '<data-type:fcs-dimension data-type:name="FL1-A"/></gating:dimension>'
           + '</gating:RectangleGate></group></gating:Gating-ML>')
    path = tmp_path / 'nested.xml'
    path.write_text(xml, encoding='utf-8')
    gates, warnings = read_gatingml(str(path))
    assert gates == []
    assert any('1 gate(s) inside <group> not read' in w for w in warnings)


# ── .wsp export other readers accept ─────────────────────────────────────────

_WSP_GATES = [
    {'id': 'p', 'kind': 'threshold', 'channel': 'FSC-A', 'value': 10.0,
     'parent_id': None, 'name': 'Cells'},
    {'id': 'a', 'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
     'vertices': [[40, 0.5], [95, 0.5], [95, 95], [40, 60]],
     'parent_id': 'p', 'name': 'A'},
    {'id': 'b', 'kind': 'threshold', 'channel': 'FL3-A', 'value': 50.0,
     'parent_id': 'p', 'name': 'B'},
    {'id': 'ab', 'kind': 'boolean', 'op': 'and', 'operands': ['a', 'b'],
     'parent_id': 'p', 'name': 'A and B'},
    {'id': 'na', 'kind': 'boolean', 'op': 'not', 'operands': ['a'],
     'parent_id': 'p', 'name': 'not A'},
]


def _poly_truth(df):
    from matplotlib.path import Path
    x, y = df['FL1-A'].to_numpy(), df['FL2-A'].to_numpy()
    inside = Path(np.array(_WSP_GATES[1]['vertices'], float)).contains_points(
        np.column_stack([x, y]))
    cells = df['FSC-A'].to_numpy() >= 10.0
    a, b = cells & inside, cells & (df['FL3-A'].to_numpy() >= 50)
    return {'Cells': cells, 'A': a, 'B': b, 'A and B': a & b, 'not A': cells & ~a}


def _wsp_counts(path, df):
    gates = fp.WspReader(str(path)).extract_gates()
    by_id = {g['_import_id']: g for g in gates}
    return {g['label']: int(np.asarray(fp.cumulative_gate_mask(
        by_id, g['_import_id'], df)).sum()) for g in gates}


def test_wsp_export_in_flowjo_form_reads_back_as_written(tmp_path):
    # FSC-A from 1,000: no event lies on a polygon edge or the threshold.
    df = _frame()
    fcs = tmp_path / 'w.fcs'
    write_fcs(df, str(fcs))
    w = fp.WspWriter()
    w.add_sample('synthetic_1', str(fcs), list(df.columns),
                 [dict(g) for g in _WSP_GATES])
    out = tmp_path / 'out.wsp'
    w.write(str(out))
    root = ET.parse(out).getroot()
    # flowJoVersion a number other readers parse; OpenFlo named apart.
    assert int(root.get('flowJoVersion').split('.')[0]) >= 10
    assert root.get(fp.WSP_OPENFLO_ATTR)
    sample = root.find('SampleList/Sample')
    t = sample.find('Transformations')
    assert t is not None
    axes = {p.get('{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name'):
            el.tag.rsplit('}', 1)[-1] for el in t for p in el}
    # The polygon's axes are linear: its edges are straight in data units,
    # as OpenFlo counts them. Scatter is linear in FlowJo anyway.
    assert axes == {'FSC-A': 'linear', 'FL1-A': 'linear', 'FL2-A': 'linear'}
    deps = {n.get('name'): [d.get('name') for d in n.iter('Dependent')]
            for n in root.iter() if n.tag in ('AndNode', 'NotNode')}
    assert deps == {'A and B': ['Cells/A', 'Cells/B'], 'not A': ['Cells/A']}
    # Read back: every count as written. The threshold at 10 on FSC-A sits
    # where FlowJo opens an edge on a linear 0..262144 axis; OpenFlo counted
    # it literally, and its own export keeps it so.
    want = {k: int(v.sum()) for k, v in _poly_truth(df).items()}
    assert _wsp_counts(out, df) == want
    small = df.assign(**{'FSC-A': np.where(np.arange(len(df)) % 3 == 0, 5.0,
                                           df['FSC-A'])})
    assert _wsp_counts(out, small)['Cells'] == int((small['FSC-A'] >= 10).sum())


def test_wsp_export_of_an_earlier_openflo_still_reads(tmp_path):
    # A guard, passing before and after: the earlier form ('../X' operands,
    # no <Transformations>, flowJoVersion "OpenFlo-export-1.0") still reads.
    df = _frame()
    w = fp.WspWriter()
    w.add_sample('synthetic_1', '', list(df.columns), [dict(g) for g in _WSP_GATES])
    out = tmp_path / 'old.wsp'
    w.write(str(out))
    tree = ET.parse(out)
    root = tree.getroot()
    root.set('flowJoVersion', 'OpenFlo-export-1.0')
    root.attrib.pop('openfloVersion', None)
    for s in root.iter('Sample'):
        for t in s.findall('Transformations'):
            s.remove(t)
    for d in root.iter('Dependent'):
        d.set('name', '../' + d.get('name').split('/')[-1])
    tree.write(out, encoding='utf-8', xml_declaration=True)
    want = {k: int(v.sum()) for k, v in _poly_truth(df).items()}
    assert _wsp_counts(out, df) == want


# ── FCS whose HEADER and TEXT disagree on where the data ends ───────────────

@pytest.mark.parametrize('where', ['start', 'end'])
def test_offset_note_names_the_end_that_disagrees(where):
    DataOffsetDiscrepancyError = type('DataOffsetDiscrepancyError', (Exception,), {})
    exc = DataOffsetDiscrepancyError(
        f'x.fcs has a discrepancy in the DATA {where} byte location: '
        '100 (HEADER) vs 200 (TEXT)')
    told = 'starts' if where == 'start' else 'ends'
    assert f'where the data {told}' in fp.plain_fcs_problem(exc)
    note = fp._quirk_note(dict((q[0], q[2]) for q in fp._FCS_QUIRKS)['discrepancy'],
                          exc)
    assert f'where the data {told}' in note
