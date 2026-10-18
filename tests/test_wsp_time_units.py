"""Time gates cross the .wsp boundary in FlowJo's unit: seconds.

OpenFlo keeps the Time channel as the FCS stores it, in raw ticks. FlowJo
writes Time gate coordinates in seconds (ticks x $TIMESTEP) and records
$TIMESTEP per sample under <Sample><Keywords>. A Time gate therefore has to
be multiplied by $TIMESTEP on export and divided by it on import; passing the
number through unchanged moves the gate by a factor of 1/$TIMESTEP (100x at
the common 0.01), so a narrow Time window selects a different set of events
on the other side.

Synthetic data: 2000 events with Time = 0, 5, 10, ... ticks and
$TIMESTEP = 0.01, i.e. 0 .. 99.995 s. The window [20 s, 40 s) holds exactly
400 events.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from openflo import pipeline as fp
from openflo.compare import _per_sample_inventory

GT = '{http://www.isac-net.org/std/Gating-ML/v2.0/gating}'
TIMESTEP = 0.01
N = 2000
IN_WINDOW = 400          # events with 20 s <= Time < 40 s


@pytest.fixture
def timed_fcs(tmp_path):
    import flowio
    rng = np.random.default_rng(0)
    ticks = np.arange(N, dtype=float) * 5.0
    ev = np.column_stack([rng.normal(5e4, 5e3, N), rng.normal(2e4, 2e3, N),
                          ticks]).astype(np.float32)
    path = tmp_path / 'timed.fcs'
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), ['FSC-A', 'SSC-A', 'Time'],
                          metadata_dict={'TIMESTEP': str(TIMESTEP)})
    return str(path)


def _count(gate, df):
    g = dict(gate, id='g', parent_id=None)
    return int(fp.cumulative_gate_mask({'g': g}, 'g', df).sum())


def _flowjo_style_wsp(path, fcs, lo_s, hi_s, timestep=TIMESTEP):
    """A workspace laid out the way FlowJo 10 writes one: $TIMESTEP under
    <Sample><Keywords>, a 1-D Time RectangleGate in seconds, plus a 2-D
    scatter gate below it that must come through unchanged."""
    kws = (f'<Keywords><Keyword name="$TIMESTEP" value="{timestep}"/></Keywords>'
           if timestep is not None else '')
    path.write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Workspace version="20.0"
  xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating"
  xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">
 <SampleList><Sample>
  <DataSet uri="{fp._wsp_dataset_uri(fcs)}" sampleID="1"/>
  {kws}
  <SampleNode name="s1" sampleID="1" count="{N}"><Subpopulations>
   <Population name="Time" count="{IN_WINDOW}">
    <Gate gating:id="ID1"><gating:RectangleGate>
     <gating:dimension gating:min="{lo_s}" gating:max="{hi_s}">
      <data-type:fcs-dimension data-type:name="Time"/></gating:dimension>
    </gating:RectangleGate></Gate>
    <Subpopulations><Population name="Cells" count="0">
     <Gate gating:id="ID2"><gating:RectangleGate>
      <gating:dimension gating:min="100.0" gating:max="90000.0">
       <data-type:fcs-dimension data-type:name="FSC-A"/></gating:dimension>
      <gating:dimension gating:min="200.0" gating:max="80000.0">
       <data-type:fcs-dimension data-type:name="SSC-A"/></gating:dimension>
     </gating:RectangleGate></Gate>
    </Population></Subpopulations>
   </Population>
  </Subpopulations></SampleNode>
 </Sample></SampleList>
</Workspace>''', encoding='utf-8')
    return str(path)


def test_writer_emits_time_gate_in_seconds(timed_fcs, tmp_path):
    s = fp.FlowSample(timed_fcs)
    gate = {'id': 't', 'parent_id': None, 'kind': 'interval',
            'channel': 'Time', 'lo': 2000.0, 'hi': 4000.0, 'label': 'Time'}
    assert _count(gate, s.data) == IN_WINDOW          # OpenFlo: ticks

    w = fp.WspWriter()
    w.add_sample('s1', timed_fcs, list(s.data.columns), [gate])
    out = tmp_path / 'out.wsp'
    w.write(str(out))

    dim = next(ET.parse(out).getroot().iter(f'{GT}dimension'))
    lo, hi = float(dim.get(f'{GT}min')), float(dim.get(f'{GT}max'))
    assert (lo, hi) == pytest.approx((20.0, 40.0)), (
        f"Time gate written as [{lo}, {hi}] -- FlowJo reads Time gates in "
        f"seconds (ticks x $TIMESTEP), so it must be [20, 40]")
    secs = s.data['Time'].to_numpy(dtype=float) * TIMESTEP
    assert int(((secs >= lo) & (secs < hi)).sum()) == IN_WINDOW
    assert gate['lo'] == 2000.0                       # caller's dict untouched


def test_reader_returns_flowjo_time_gate_in_ticks(timed_fcs, tmp_path):
    wsp = _flowjo_style_wsp(tmp_path / 'fj.wsp', timed_fcs, 20.0, 40.0)
    gates = fp.WspReader(wsp).extract_gates()
    tg = next(g for g in gates if g.get('channel') == 'Time')
    assert (tg['lo'], tg['hi']) == pytest.approx((2000.0, 4000.0)), (
        f"FlowJo's [20 s, 40 s) came back as [{tg['lo']}, {tg['hi']}] "
        f"-- OpenFlo's Time is in ticks, so it must be [2000, 4000)")
    s = fp.FlowSample(timed_fcs)
    assert _count(tg, s.data) == IN_WINDOW
    rect = next(g for g in gates if g.get('kind') == 'rect')
    assert (rect['x0'], rect['x1'], rect['y0'], rect['y1']) == (
        100.0, 90000.0, 200.0, 80000.0)               # non-Time axes untouched


def test_reader_per_sample_node_converts_time(timed_fcs, tmp_path):
    wsp = _flowjo_style_wsp(tmp_path / 'fj.wsp', timed_fcs, 20.0, 40.0)
    reader = fp.WspReader(wsp)
    sn = next(reader.root.iter('SampleNode'))
    tg = next(g for g in reader.extract_gates(sample_node=sn)
              if g.get('channel') == 'Time')
    assert (tg['lo'], tg['hi']) == pytest.approx((2000.0, 4000.0))


def test_compare_inventory_evaluates_time_gate_in_ticks(timed_fcs, tmp_path):
    wsp = _flowjo_style_wsp(tmp_path / 'fj.wsp', timed_fcs, 20.0, 40.0)
    (_name, _uri, pops), = _per_sample_inventory(fp.WspReader(wsp))
    tg = next(g for _u, _n, _c, g, _p in pops if g.get('channel') == 'Time')
    s = fp.FlowSample(timed_fcs)
    assert _count(tg, s.data) == IN_WINDOW, (
        "compare would report FlowJo's Time population against the wrong "
        "events: the gate is in seconds, the data in ticks")


def test_openflo_round_trip_keeps_ticks(timed_fcs, tmp_path):
    s = fp.FlowSample(timed_fcs)
    gates = [
        {'id': 't', 'parent_id': None, 'kind': 'interval', 'channel': 'Time',
         'lo': 2000.0, 'hi': 4000.0, 'label': 'Time'},
        {'id': 'th', 'parent_id': 't', 'kind': 'threshold', 'channel': 'Time',
         'value': 3000.0, 'label': 'Late'},
        {'id': 'p', 'parent_id': 't', 'kind': 'polygon', 'x_channel': 'Time',
         'y_channel': 'FSC-A', 'label': 'Poly',
         'vertices': [[2000.0, 0.0], [4000.0, 0.0], [4000.0, 9e4]]},
    ]
    w = fp.WspWriter()
    w.add_sample('s1', timed_fcs, list(s.data.columns), gates)
    out = tmp_path / 'rt.wsp'
    w.write(str(out))
    back = fp.WspReader(str(out)).extract_gates()
    by_label = {g['label']: g for g in back}
    assert (by_label['Time']['lo'], by_label['Time']['hi']) == pytest.approx((2000.0, 4000.0))
    assert by_label['Late']['value'] == pytest.approx(3000.0)
    assert np.allclose(by_label['Poly']['vertices'],
                       [[2000.0, 0.0], [4000.0, 0.0], [4000.0, 9e4]])


def test_import_scales_with_the_fcs_not_the_workspace_keyword(timed_fcs, tmp_path):
    # FlowJo scaled Time with the FCS's own $TIMESTEP when the workspace
    # <Keywords> said otherwise (measured: keyword 0.01, file 0.05, FlowJo
    # used 0.05). The import must divide by what FlowJo multiplied by.
    wsp = _flowjo_style_wsp(tmp_path / 'fj.wsp', timed_fcs, 20.0, 40.0, timestep=0.05)
    tg = next(g for g in fp.WspReader(wsp).extract_gates() if g.get('channel') == 'Time')
    assert (tg['lo'], tg['hi']) == pytest.approx((2000.0, 4000.0))      # 1/0.01


def test_import_without_a_keyword_still_scales_with_the_fcs(timed_fcs, tmp_path):
    # FlowJo's save of a file has no $TIMESTEP keyword when the file has none,
    # and the workspace may carry no <Keywords> at all: the FCS decides.
    wsp = _flowjo_style_wsp(tmp_path / 'fj.wsp', timed_fcs, 20.0, 40.0, timestep=None)
    tg = next(g for g in fp.WspReader(wsp).extract_gates() if g.get('channel') == 'Time')
    assert (tg['lo'], tg['hi']) == pytest.approx((2000.0, 4000.0))


def test_keyword_stands_in_only_when_the_fcs_is_missing(timed_fcs, tmp_path):
    wsp = _flowjo_style_wsp(tmp_path / 'fj.wsp', timed_fcs, 20.0, 40.0, timestep=0.05)
    import os
    os.remove(timed_fcs)
    tg = next(g for g in fp.WspReader(wsp).extract_gates() if g.get('channel') == 'Time')
    assert (tg['lo'], tg['hi']) == pytest.approx((400.0, 800.0))        # 1/0.05
    wsp2 = _flowjo_style_wsp(tmp_path / 'fj2.wsp', timed_fcs, 20.0, 40.0, timestep=None)
    tg2 = next(g for g in fp.WspReader(wsp2).extract_gates() if g.get('channel') == 'Time')
    assert (tg2['lo'], tg2['hi']) == (20.0, 40.0)                        # nothing to scale by


def _fcs_without_timestep(tmp_path, btim, etim):
    """Same events as `timed_fcs` (ticks 0..9995), no $TIMESTEP, clock span
    given by $BTIM/$ETIM."""
    import flowio
    rng = np.random.default_rng(0)
    ticks = np.arange(N, dtype=float) * 5.0
    ev = np.column_stack([rng.normal(5e4, 5e3, N), rng.normal(2e4, 2e3, N),
                          ticks]).astype(np.float32)
    path = tmp_path / 'nots.fcs'
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), ['FSC-A', 'SSC-A', 'Time'],
                          metadata_dict={'BTIM': btim, 'ETIM': etim})
    assert b'$TIMESTEP' not in path.read_bytes()
    return str(path)


@pytest.mark.parametrize('btim, etim', [
    ('10:00:00', '10:03:19.90'),          # FCS 3.1 hh:mm:ss.cc
    ('10:00:00', '10:03:19:54'),          # FCS 3.0 hh:mm:ss:tt, tt in 1/60 s
    ('23:59:00', '00:02:19.90'),          # acquisition over midnight
], ids=['fcs31', 'fcs30', 'midnight'])
def test_no_timestep_uses_flowjos_clock_span_over_last_tick(tmp_path, btim, etim):
    # FlowJo 10.10.2 on a file with no $TIMESTEP: seconds per tick =
    # ($ETIM - $BTIM) / last tick (measured). Here 199.9 s / 9995.
    fcs = _fcs_without_timestep(tmp_path, btim, etim)
    f = 199.9 / 9995.0
    assert fp._fcs_timestep(fcs) == pytest.approx(f)
    gate = {'id': 't', 'parent_id': None, 'kind': 'interval',
            'channel': 'Time', 'lo': 2000.0, 'hi': 4000.0, 'label': 'Time'}
    w = fp.WspWriter()
    w.add_sample('s1', fcs, ['FSC-A', 'SSC-A', 'Time'], [gate])
    out = tmp_path / 'out.wsp'
    w.write(str(out))
    root = ET.parse(out).getroot()
    dim = next(root.iter(f'{GT}dimension'))
    assert (float(dim.get(f'{GT}min')), float(dim.get(f'{GT}max'))) == pytest.approx(
        (2000.0 * f, 4000.0 * f))
    # FlowJo reads $TIMESTEP from the FCS, so the writer must not invent one
    # beside a TEXT that lacks it.
    names = [k.get('name') for k in root.iter('Keyword')]
    assert '$BTIM' in names and '$TIMESTEP' not in names


def test_no_timestep_and_no_clock_means_no_conversion_on_export(tmp_path):
    import flowio
    path = tmp_path / 'bare.fcs'
    ev = np.column_stack([np.ones(10), np.ones(10), np.arange(10.0)]).astype(np.float32)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), ['FSC-A', 'SSC-A', 'Time'])
    assert fp._fcs_timestep(str(path)) is None
    gate = {'id': 't', 'parent_id': None, 'kind': 'interval',
            'channel': 'Time', 'lo': 2.0, 'hi': 4.0, 'label': 'Time'}
    w = fp.WspWriter()
    w.add_sample('s1', str(path), ['FSC-A', 'SSC-A', 'Time'], [gate])
    out = tmp_path / 'out.wsp'
    w.write(str(out))
    dim = next(ET.parse(out).getroot().iter(f'{GT}dimension'))
    assert (float(dim.get(f'{GT}min')), float(dim.get(f'{GT}max'))) == (2.0, 4.0)


def test_round_trip_of_a_file_without_timestep_keeps_ticks(tmp_path):
    # The writer scales by ($ETIM - $BTIM) / last tick for such a file, so the
    # reader must divide by the same factor, read from the same file.
    fcs = _fcs_without_timestep(tmp_path, '10:00:00', '10:03:19.90')
    gate = {'id': 't', 'parent_id': None, 'kind': 'interval',
            'channel': 'Time', 'lo': 1000.0, 'hi': 2000.0, 'label': 'Time'}
    w = fp.WspWriter()
    w.add_sample('s1', fcs, ['FSC-A', 'SSC-A', 'Time'], [gate])
    out = tmp_path / 'rt.wsp'
    w.write(str(out))
    tg = next(g for g in fp.WspReader(str(out)).extract_gates() if g.get('channel') == 'Time')
    assert (tg['lo'], tg['hi']) == pytest.approx((1000.0, 2000.0))


def test_given_timestep_never_overrides_the_files(timed_fcs, tmp_path):
    # FlowJo scales with the file's $TIMESTEP, so a different value from the
    # caller would put the gate somewhere else in FlowJo; the file's wins.
    gate = {'id': 't', 'parent_id': None, 'kind': 'interval',
            'channel': 'Time', 'lo': 2000.0, 'hi': 4000.0, 'label': 'Time'}
    w = fp.WspWriter()
    w.add_sample('s1', timed_fcs, ['FSC-A', 'SSC-A', 'Time'], [gate], timestep=0.05)
    out = tmp_path / 'o.wsp'
    w.write(str(out))
    dim = next(ET.parse(out).getroot().iter(f'{GT}dimension'))
    assert (float(dim.get(f'{GT}min')), float(dim.get(f'{GT}max'))) == pytest.approx((20.0, 40.0))


def test_time_threshold_is_written_in_seconds(timed_fcs, tmp_path):
    gate = {'id': 't', 'parent_id': None, 'kind': 'threshold', 'channel': 'Time',
            'value': 3000.0, 'label': 'Late'}
    w = fp.WspWriter()
    w.add_sample('s1', timed_fcs, ['FSC-A', 'SSC-A', 'Time'], [gate])
    out = tmp_path / 'th.wsp'
    w.write(str(out))
    dim = next(ET.parse(out).getroot().iter(f'{GT}dimension'))
    assert float(dim.get(f'{GT}min')) == pytest.approx(30.0)


def test_each_sample_of_a_document_uses_its_own_timestep(tmp_path):
    # Two samples, two $TIMESTEPs: reading the whole document must convert
    # each sample's Time gate with its own factor, once.
    import flowio
    paths = []
    for name, ts in (('a', '0.01'), ('b', '0.05')):
        ev = np.column_stack([np.ones(10), np.ones(10), np.arange(10.0)]).astype(np.float32)
        p = tmp_path / f'{name}.fcs'
        with open(p, 'wb') as fh:
            flowio.create_fcs(fh, ev.flatten().tolist(), ['FSC-A', 'SSC-A', 'Time'],
                              metadata_dict={'TIMESTEP': ts})
        paths.append(str(p))
    gate = {'id': 't', 'parent_id': None, 'kind': 'interval', 'channel': 'Time',
            'lo': 2000.0, 'hi': 4000.0, 'label': 'Time'}
    w = fp.WspWriter()
    for name, p in zip(('a', 'b'), paths, strict=True):
        w.add_sample(name, p, ['FSC-A', 'SSC-A', 'Time'], [dict(gate)])
    out = tmp_path / 'two.wsp'
    w.write(str(out))
    secs = [(float(d.get(f'{GT}min')), float(d.get(f'{GT}max')))
            for d in ET.parse(out).getroot().iter(f'{GT}dimension')]
    assert secs == [pytest.approx((20.0, 40.0)), pytest.approx((100.0, 200.0))]
    back = [g for g in fp.WspReader(str(out)).extract_gates() if g.get('channel') == 'Time']
    assert [(g['lo'], g['hi']) for g in back] == [pytest.approx((2000.0, 4000.0))] * 2


def _bare_fcs(path, meta=None, time=None):
    import flowio
    t = np.arange(10.0) if time is None else np.asarray(time, dtype=float)
    ev = np.column_stack([np.ones(len(t)), np.ones(len(t)), t]).astype(np.float32)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), ['FSC-A', 'SSC-A', 'Time'],
                          metadata_dict=meta or {})
    return str(path)


def test_given_timestep_is_used_when_the_fcs_cannot_be_found(timed_fcs, tmp_path):
    # A caller that located the FCS elsewhere hands over its timestep.
    wsp = _flowjo_style_wsp(tmp_path / 'fj.wsp', 'Z:/nowhere/timed.fcs', 20.0, 40.0,
                            timestep=None)
    r = fp.WspReader(wsp)
    sn = r.root.find('SampleList/Sample/SampleNode')
    tg = next(g for g in r.extract_gates(sample_node=sn, timestep=0.01)
              if g.get('channel') == 'Time')
    assert (tg['lo'], tg['hi']) == pytest.approx((2000.0, 4000.0))


def test_keyword_does_not_stand_in_for_a_readable_fcs_without_a_factor(tmp_path):
    # The file is there but gives FlowJo no factor: a workspace keyword must
    # not invent one (FlowJo ignores workspace keywords for Time).
    fcs = _bare_fcs(tmp_path / 'bare.fcs')
    wsp = _flowjo_style_wsp(tmp_path / 'fj.wsp', fcs, 20.0, 40.0, timestep=0.05)
    tg = next(g for g in fp.WspReader(wsp).extract_gates() if g.get('channel') == 'Time')
    assert (tg['lo'], tg['hi']) == (20.0, 40.0)


@pytest.mark.parametrize('uri, want', [
    ('file:/C:/a%20b,c/x.fcs', 'C:/a b,c/x.fcs'),
    ('file:///C:/a%20b/x.fcs', 'C:/a b/x.fcs'),
    ('file:C:/x.fcs', 'C:/x.fcs'),
    ('file:/data/x%20y.fcs', '/data/x y.fcs'),
    ('file:///data/x.fcs', '/data/x.fcs'),
    ('C:\\plain\\x.fcs', 'C:/plain/x.fcs'),
])
def test_dataset_uri_forms_resolve_to_a_local_path(uri, want):
    assert fp._wsp_uri_path(uri) == want


def test_time_import_through_a_percent_encoded_path(tmp_path):
    # FlowJo writes 'tube 1.fcs' as 'tube%201.fcs'; the FCS must still be found.
    d = tmp_path / 'run a'
    d.mkdir()
    fcs = _bare_fcs(d / 'tube 1.fcs', {'TIMESTEP': '0.01'}, np.arange(0, 6000, 50.0))
    gate = {'id': 't', 'parent_id': None, 'kind': 'interval', 'channel': 'Time',
            'lo': 2000.0, 'hi': 4000.0, 'label': 'Time'}
    w = fp.WspWriter()
    w.add_sample('s', fcs, ['FSC-A', 'SSC-A', 'Time'], [gate])
    out = tmp_path / 'enc.wsp'
    w.write(str(out))
    assert '%20' in out.read_text(encoding='utf-8')
    # Make the workspace keyword wrong, so only the FCS can give 0.01.
    xml = out.read_text(encoding='utf-8')
    kw = 'name="$TIMESTEP" value="0.01"'
    assert xml.count(kw) == 1
    out.write_text(xml.replace(kw, 'name="$TIMESTEP" value="0.05"'), encoding='utf-8')
    tg = next(g for g in fp.WspReader(str(out)).extract_gates() if g.get('channel') == 'Time')
    assert (tg['lo'], tg['hi']) == pytest.approx((2000.0, 4000.0))


@pytest.mark.parametrize('meta, time', [
    ({'BTIM': '10:00:00'}, None),                        # no $ETIM
    ({'BTIM': '10:00:00', 'ETIM': '10:01:00'}, np.zeros(10)),   # last tick 0
])
def test_unusable_clock_gives_no_factor_and_does_not_crash(tmp_path, meta, time):
    fcs = _bare_fcs(tmp_path / 'c.fcs', meta, time)
    assert fp._fcs_timestep(fcs) is None
    w = fp.WspWriter()
    w.add_sample('s', fcs, ['FSC-A', 'SSC-A', 'Time'], [
        {'id': 't', 'parent_id': None, 'kind': 'interval', 'channel': 'Time',
         'lo': 2.0, 'hi': 4.0, 'label': 'Time'}])
    w.write(str(tmp_path / 'c.wsp'))


def test_unconvertible_time_warning_covers_the_y_axis(tmp_path):
    g = {'id': 'r', 'parent_id': None, 'kind': 'rect', 'x_channel': 'FSC-A',
         'y_channel': 'Time', 'x0': 1e3, 'x1': 1e5, 'y0': 100.0, 'y1': 500.0, 'name': 'R'}
    w = fp.WspWriter()
    w.add_sample('s', '', ['FSC-A', 'Time'], [g])
    w.write(str(tmp_path / 'y.wsp'))
    assert any("'R' is on Time" in m for m in w.warnings), w.warnings


def test_compare_finds_the_fcs_through_fcs_dir_for_time(tmp_path):
    # openflo-compare's --fcs-dir: the uri does not resolve, the FCS is in
    # fcs_dir, and its own $TIMESTEP converts the Time gate.
    d = tmp_path / 'fcs'
    d.mkdir()
    fcs = _bare_fcs(d / 'timed.fcs', {'TIMESTEP': '0.01'})
    wsp = _flowjo_style_wsp(tmp_path / 'fj.wsp', 'Z:/nowhere/timed.fcs', 20.0, 40.0,
                            timestep=0.05)
    (_n, _u, pops), = _per_sample_inventory(fp.WspReader(wsp), str(d))
    tg = next(g for _, name, _, g, _ in pops if name == 'Time')
    assert (tg['lo'], tg['hi']) == pytest.approx((2000.0, 4000.0))
    assert fcs.endswith('timed.fcs')
