"""openflo-compare counts a FlowJo workspace's gates the way FlowJo does.

Two ways it disagreed with FlowJo, each reproduced first on a synthetic FCS
(20,000 events, FL1-A/FL2-A with a 10% / 2% $SPILL):

* A population whose gate it could not read (a QuadrantGate, a gate type it
  does not know) got no row, and each child was re-rooted onto the sample
  root: counted against every event the unread gate had excluded, a
  plausible number with no error beside it. Such populations and everything
  under them are now error rows with no OpenFlo count.
* WspReader strips FlowJo's 'Comp-' prefix, so a gate FlowJo drew on the
  UNcompensated parameter (the bare 'FL2-A' of a compensated sample) was
  evaluated on compensated values: 5,216 events where FlowJo counts 8,413.
  The parsed gate now records which of its channels FlowJo named
  compensated, and compare evaluates the others on the uncompensated data.
"""
from __future__ import annotations

import numpy as np

import openflo.pipeline as fp
from openflo.compare import _compare_one_sample, _per_sample_inventory

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'Time']
NS = ('xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" '
      'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"')


def _fcs(path, n=20_000, seed=1):
    import flowio
    rng = np.random.default_rng(seed)
    k = rng.choice(3, n, p=[0.5, 0.3, 0.2])
    fl1 = np.where(k == 0, rng.normal(0, 150, n),
                   np.where(k == 1, rng.normal(1500, 300, n),
                            rng.normal(20000, 4000, n)))
    fl2 = rng.normal(2000, 800, n)
    meas = np.column_stack([fl1, fl2]) @ np.array([[1, 0.10], [0.02, 1]])
    ev = np.column_stack([rng.uniform(2e4, 2e5, n), rng.uniform(1e4, 1.5e5, n),
                          meas[:, 0], meas[:, 1],
                          np.arange(n) * 5.0]).astype(np.float32)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.ravel().tolist(), CH,
                          metadata_dict={'TIMESTEP': '0.01',
                                         'SPILL': '2,FL1-A,FL2-A,1,0.10,0.02,1'})
    return str(path)


def _dim(ch, lo=None, hi=None):
    a = (f' gating:min="{lo}"' if lo is not None else '') + \
        (f' gating:max="{hi}"' if hi is not None else '')
    return (f'<gating:dimension{a}><data-type:fcs-dimension '
            f'data-type:name="{ch}"/></gating:dimension>')


def _pop(name, gate_xml, children=''):
    sub = f'<Subpopulations>{children}</Subpopulations>' if children else ''
    return (f'<Population name="{name}" count="0"><Gate>{gate_xml}</Gate>'
            f'{sub}</Population>')


def _rect(*dims):
    return '<gating:RectangleGate>' + ''.join(dims) + '</gating:RectangleGate>'


def _wsp(path, fcs, pops):
    path.write_text(
        f'<?xml version="1.0"?>\n<Workspace {NS}><SampleList><Sample>'
        f'<DataSet uri="file:{fcs}" sampleID="1"/>'
        f'<SampleNode name="s" sampleID="1"><Subpopulations>{pops}'
        '</Subpopulations></SampleNode></Sample></SampleList></Workspace>',
        encoding='utf-8')
    return str(path)


def _rows(wsp, fcs, tmp_path):
    r = fp.WspReader(wsp)
    (_, _, pops), = _per_sample_inventory(r, str(tmp_path))
    return {row['population']: row
            for row in _compare_one_sample('s', fcs, pops, wsp_path=wsp)}


def test_bare_name_gate_is_counted_on_uncompensated_values(tmp_path):
    """FlowJo evaluates 'FL2-A' on the uncompensated parameter and
    'Comp-FL2-A' on the compensated one. Both used to read 5,216."""
    fcs = _fcs(tmp_path / 's.fcs')
    wsp = _wsp(tmp_path / 's.wsp', fcs,
               _pop('raw', _rect(_dim('FL2-A', 2500, 262144)))
               + _pop('comp', _rect(_dim('Comp-FL2-A', 2500, 262144))))
    s = fp.FlowSample(fcs)
    raw = s.raw['FL2-A'].to_numpy(dtype=float)
    s.auto_compensate()
    comp = s.data['FL2-A'].to_numpy(dtype=float)
    want_raw = int(((raw >= 2500) & (raw < 262144)).sum())
    want_comp = int(((comp >= 2500) & (comp < 262144)).sum())
    assert want_raw != want_comp                  # the fixture separates them
    rows = _rows(wsp, fcs, tmp_path)
    assert rows['raw']['openflo_count'] == want_raw
    assert rows['comp']['openflo_count'] == want_comp


def test_bare_name_axis_of_a_2d_gate_uses_uncompensated_values(tmp_path):
    """Per axis: a polygon on Comp-FL1-A x FL2-A reads FL1 compensated and
    FL2 uncompensated."""
    fcs = _fcs(tmp_path / 's.fcs')
    poly = ('<gating:PolygonGate>' + _dim('Comp-FL1-A') + _dim('FL2-A')
            + ''.join(f'<gating:vertex><gating:coordinate data-type:value="{x}"/>'
                      f'<gating:coordinate data-type:value="{y}"/></gating:vertex>'
                      for x, y in [(-1000, 2500), (5000, 2500), (5000, 9000),
                                   (-1000, 9000)])
            + '</gating:PolygonGate>')
    wsp = _wsp(tmp_path / 's.wsp', fcs, _pop('mixed', poly))
    s = fp.FlowSample(fcs)
    fl2_raw = s.raw['FL2-A'].to_numpy(dtype=float)
    s.auto_compensate()
    fl1 = s.data['FL1-A'].to_numpy(dtype=float)
    want = int(((fl1 >= -1000) & (fl1 < 5000)
                & (fl2_raw >= 2500) & (fl2_raw < 9000)).sum())
    assert _rows(wsp, fcs, tmp_path)['mixed']['openflo_count'] == want


def test_reader_records_which_channels_flowjo_named_compensated(tmp_path):
    """The editor import cannot evaluate a bare name on uncompensated data
    (it holds only compensated values), so the parsed gate says which names
    carried FlowJo's compensated marker, for it to warn on the others."""
    fcs = _fcs(tmp_path / 's.fcs')
    wsp = _wsp(tmp_path / 's.wsp', fcs,
               _pop('raw', _rect(_dim('FL2-A', 2500, 262144)))
               + _pop('comp', _rect(_dim('Comp-FL1-A', 0, 10),
                                    _dim('&lt;FL2-A&gt;', 0, 10))))
    g = {x['label']: x for x in fp.WspReader(wsp).extract_gates()}
    assert g['raw'].get('wsp_comp_channels', []) == []
    assert g['comp']['wsp_comp_channels'] == ['FL1-A', 'FL2-A']


def _unreadable_parent(tmp_path, gate_xml):
    fcs = _fcs(tmp_path / 's.fcs')
    child = _pop('kid', _rect(_dim('FL1-A', 5000, 262144)))
    wsp = _wsp(tmp_path / 's.wsp', fcs,
               _pop('Cells', _rect(_dim('FSC-A', 3e4, 1.9e5)),
                    _pop('odd', gate_xml, child)))
    return fcs, wsp


def test_population_under_an_unread_gate_is_an_error_row_not_rerooted(tmp_path):
    """A gate type compare cannot read: the population and its child were
    missing / counted from the sample root (4,003 bright events, where the
    unread gate's own count is unknown). Both are now error rows."""
    fcs, wsp = _unreadable_parent(
        tmp_path, '<gating:CurlyQuad gating:id="c"/>')
    rows = _rows(wsp, fcs, tmp_path)
    assert rows['Cells']['openflo_count'] is not None
    for name in ('odd', 'kid'):
        assert rows[name]['openflo_count'] is None, rows[name]
        assert rows[name]['delta'] is None
        assert 'odd' in rows[name]['error'] or 'not read' in rows[name]['error']
    assert 'CurlyQuad' in rows['odd']['error']


def test_quadrant_gate_population_is_an_error_row(tmp_path):
    """A Gating-ML QuadrantGate inside one Population cannot say which
    quadrant the population is; its child must not be counted from Cells."""
    quad = ('<gating:QuadrantGate gating:id="q">'
            '<gating:divider gating:id="d1"><data-type:fcs-dimension '
            'data-type:name="FL1-A"/><gating:value>1000</gating:value>'
            '</gating:divider><gating:divider gating:id="d2">'
            '<data-type:fcs-dimension data-type:name="FL2-A"/>'
            '<gating:value>1000</gating:value></gating:divider>'
            '</gating:QuadrantGate>')
    fcs, wsp = _unreadable_parent(tmp_path, quad)
    rows = _rows(wsp, fcs, tmp_path)
    assert rows['odd']['openflo_count'] is None
    assert rows['kid']['openflo_count'] is None


def test_wsp_reader_does_not_lift_children_of_an_unread_gate(tmp_path):
    """The editor's import attached 'kid' to Cells, where it counted every
    bright Cells event instead of the unread gate's share. It is left out
    (with a warning) instead of re-rooted."""
    fcs, wsp = _unreadable_parent(
        tmp_path, '<gating:CurlyQuad gating:id="c"/>')
    gates = fp.WspReader(wsp).extract_gates()
    assert [g['label'] for g in gates] == ['Cells']


def test_editor_import_says_which_gates_are_on_uncompensated_values(tmp_path):
    """The editor holds a compensated sample's channels compensated only, so
    it cannot count FlowJo's bare-name gate as FlowJo does; the load status
    names the gate and the channel instead of showing 5,216 as if it were
    FlowJo's 8,413. Populations under an unread gate are named at ingest."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_gui_smoke import _editor_or_skip
    root, ed, _gui = _editor_or_skip()
    try:
        fcs = _fcs(tmp_path / 's.fcs')
        wsp = _wsp(tmp_path / 's.wsp', fcs,
                   _pop('FL2 raw', _rect(_dim('FL2-A', 2500, 262144)))
                   + _pop('FL2 comp', _rect(_dim('Comp-FL2-A', 2500, 262144)))
                   + _pop('odd', '<gating:CurlyQuad gating:id="c"/>',
                          _pop('kid', _rect(_dim('FL1-A', 0, 10)))))
        queued = []
        ed._queue_fcs_loads = lambda paths, *a, **k: queued.extend(paths)
        ed._ingest_wsp(wsp)
        ingest = ed.status_var.get()
        assert "'odd'" in ingest and 'not imported' in ingest and '1 under' in ingest
        (name,) = ed._pending_sample_gates
        s = fp.FlowSample(queued[0])               # as the loader prepares it
        s.auto_compensate()
        s.apply_transform()
        ed._on_loaded(name, s)
        assert sorted(g.get('label') for g in ed._sample_gates[name].values()) \
            == ['FL2 comp', 'FL2 raw']
        ed._on_load_settled()
        msg = ed.status_var.get()
        assert "'FL2 raw'" in msg and 'FL2-A' in msg and 'UNcompensated' in msg
        assert "'FL2 comp'" not in msg
    finally:
        root.destroy()
