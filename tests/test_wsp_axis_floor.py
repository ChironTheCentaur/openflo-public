"""FlowJo opens a rectangle's lower edge at the bottom of an axis.

Measured on FlowJo 10.10.2 (synthetic probe files, power-of-two event
multiplicities so every count decodes to an exact set of events):

- A rectangle whose lower edge is at or below a floor F also holds every
  event below the axis, however far below and whatever the upper edge is.
  The data is not clamped: only the edge opens.
- On a LINEAR axis F follows the axis minRange FlowJo stores for the sample:
  minRange 0 -> edges <= 20.125 open, >= 64 kept; minRange 3 -> 10 open, 64
  kept; minRange 500 -> 500 open, 1000 kept; minRange -2000 -> -2000 open,
  -1500 kept. All fit F = minRange + c, c in [20.125, 61).
- On FlowJo's default biex (width -100) an edge at -100 is kept, and F lies
  in [-2000, -1000).
- Time is a linear axis like any other: a Time gate starting at 10 s
  counted everything from 0 s on real files.

So WspReader (and compare's own parser) must open such an edge on import, or
an imported FlowJo gate undercounts by exactly the events below the axis.
OpenFlo's own exports carry no <Transformations>, and keep literal edges.
"""
import logging
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp
from openflo.compare import _per_sample_inventory

OPEN = -1e12
BIEX = ('<transforms:biex transforms:length="256" transforms:maxRange="262144" '
        'transforms:neg="0" transforms:width="-100" transforms:pos="4.418539922">'
        '<data-type:parameter data-type:name="{p}"/></transforms:biex>')
LIN = ('<transforms:linear transforms:minRange="{m}" transforms:maxRange="262144" gain="1">'
       '<data-type:parameter data-type:name="{p}"/></transforms:linear>')


def _wsp(tmp_path, gates_xml, axes=None, keywords=''):
    """One FlowJo-style sample. `axes` = {param: minRange or 'biex'}; None
    writes no <Transformations> at all (OpenFlo's own export form)."""
    trans = ''
    if axes is not None:
        trans = '<Transformations>' + ''.join(
            BIEX.format(p=p) if m == 'biex' else LIN.format(p=p, m=m)
            for p, m in axes.items()) + '</Transformations>'
    pops = ''.join(
        f'<Population name="{name}" count="0"><Gate gating:id="g{i}">'
        f'<gating:RectangleGate>{dims}</gating:RectangleGate></Gate></Population>'
        for i, (name, dims) in enumerate(gates_xml))
    out = tmp_path / 'fj.wsp'
    out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<Workspace version="20.0" '
        'xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" '
        'xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/transformations" '
        'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">'
        f'<SampleList><Sample><DataSet uri="file:/x.fcs" sampleID="1"/>{trans}{keywords}'
        f'<SampleNode name="s" sampleID="1" count="0"><Subpopulations>{pops}'
        '</Subpopulations></SampleNode></Sample></SampleList></Workspace>',
        encoding='utf-8')
    return str(out)


def _dim(ch, lo=None, hi=None):
    a = (f' gating:min="{lo}"' if lo is not None else '') + (
        f' gating:max="{hi}"' if hi is not None else '')
    return (f'<gating:dimension{a}><data-type:fcs-dimension data-type:name="{ch}"/>'
            '</gating:dimension>')


def _lo(gates, name):
    g = next(g for g in gates if (g.get('label') or g.get('name')) == name)
    return g['value'] if g['kind'] == 'threshold' else (
        g['lo'] if g['kind'] == 'interval' else (g['x0'], g['y0']))


def _by_name(path):
    gates = fp.WspReader(path).extract_gates()
    names = [(g.get('label') or g.get('name')) for g in gates]
    return gates, names


@pytest.mark.parametrize('minrange, lo, opened', [
    (0, 0.0, True), (0, 20.0, True), (0, -5000.0, True),
    (0, 64.0, False), (0, 1000.0, False),
    (3, 10.0, True), (3, 64.0, False),
    (500, 0.0, True), (500, 500.0, True), (500, 1000.0, False),
    # The triage's rule ("open when min <= 0") opened this one; FlowJo keeps it.
    (-2000, -100.0, False), (-2000, -1500.0, False), (-2000, -2000.0, True),
])
def test_linear_lower_edge_opens_at_the_floor(tmp_path, minrange, lo, opened):
    path = _wsp(tmp_path, [('G', _dim('SSC-A', lo, 250000))], axes={'SSC-A': minrange})
    gates, _ = _by_name(path)
    assert _lo(gates, 'G') == (OPEN if opened else lo)


def test_upper_edge_below_the_floor_still_opens_the_lower_edge(tmp_path):
    # FlowJo kept the off-scale events in [-100000, -50): the edge opens, the
    # data is not clamped up to the floor (that would have emptied the gate).
    path = _wsp(tmp_path, [('G', _dim('SSC-A', -100000, -50))], axes={'SSC-A': 0})
    gates, _ = _by_name(path)
    g = gates[0]
    assert (g['lo'], g['hi']) == (OPEN, -50.0)


def test_between_measured_bounds_stays_literal_and_is_logged(tmp_path, caplog):
    path = _wsp(tmp_path, [('G', _dim('SSC-A', 40, 250000))], axes={'SSC-A': 0})
    with caplog.at_level(logging.INFO, logger=fp.log.name):
        gates, _ = _by_name(path)
    assert _lo(gates, 'G') == 40.0
    assert any('not measured' in r.getMessage() for r in caplog.records)


def test_biex_floor_and_two_d_rect(tmp_path):
    path = _wsp(tmp_path, [
        ('kept', _dim('Comp-FL1-A', -100, 5000) + _dim('SSC-A', 0, 250000)),
        ('deep', _dim('Comp-FL1-A', -5000, 5000) + _dim('SSC-A', 100, 250000)),
    ], axes={'Comp-FL1-A': 'biex', 'SSC-A': 0})
    gates, _ = _by_name(path)
    assert _lo(gates, 'kept') == (-100.0, OPEN)     # biex -100 literal; SSC 0 open
    assert _lo(gates, 'deep') == (OPEN, 100.0)


@pytest.mark.parametrize('minrange, lo, want', [
    # Exactly on the measured "opened" value: inclusive.
    (0, 20.125, OPEN), (0, 20.2, 20.2),
    (3, 10.0, OPEN), (3, 10.1, 10.1),
    (500, 500.0, OPEN), (500, 500.1, 500.1),
])
def test_linear_floor_is_inclusive_exactly_at_the_measured_value(tmp_path, minrange, lo, want):
    path = _wsp(tmp_path, [('G', _dim('SSC-A', lo, 250000))], axes={'SSC-A': minrange})
    assert _lo(_by_name(path)[0], 'G') == want


@pytest.mark.parametrize('lo, want', [
    (-2000.0, OPEN), (-3000.0, OPEN),        # at / below the floor's measured bottom
    (-1999.0, -1999.0), (-1500.0, -1500.0),  # unmeasured band: literal
    (-1000.0, -1000.0), (-500.0, -500.0),    # kept
])
def test_biex_floor_bounds(tmp_path, lo, want):
    path = _wsp(tmp_path, [('G', _dim('Comp-FL1-A', lo, 5000))], axes={'Comp-FL1-A': 'biex'})
    assert _lo(_by_name(path)[0], 'G') == want


@pytest.mark.parametrize('change', [
    ('transforms:maxRange="262144"', 'transforms:maxRange="1024"'),
    ('transforms:neg="0"', 'transforms:neg="1"'),
    ('transforms:pos="4.418539922"', 'transforms:pos="4.0"'),
    ('transforms:width="-100"', 'transforms:width="-10"'),
])
def test_biex_floor_only_on_the_measured_parameters(tmp_path, change):
    # Only FlowJo's default biex was measured; any other parameter set keeps
    # the edge literal.
    path = _wsp(tmp_path, [('G', _dim('Comp-FL1-A', -5000, 5000))], axes={'Comp-FL1-A': 'biex'})
    txt = open(path, encoding='utf-8').read()
    assert change[0] in txt
    open(path, 'w', encoding='utf-8').write(txt.replace(change[0], change[1]))
    assert _lo(_by_name(path)[0], 'G') == -5000.0

def test_other_axis_kinds_and_no_transformations_stay_literal(tmp_path):
    # OpenFlo's own export writes no <Transformations>: literal on re-import.
    path = _wsp(tmp_path, [('G', _dim('SSC-A', 0, 250000))], axes=None)
    assert _lo(_by_name(path)[0], 'G') == 0.0
    # A biex with another width was not measured: literal.
    other = BIEX.replace('width="-100"', 'width="-10"')
    p2 = _wsp(tmp_path, [('G', _dim('FL1-A', -5000, 5000))], axes={})
    txt = open(p2, encoding='utf-8').read().replace(
        '<Transformations></Transformations>',
        '<Transformations>' + other.format(p='FL1-A') + '</Transformations>')
    open(p2, 'w', encoding='utf-8').write(txt)
    assert _lo(_by_name(p2)[0], 'G') == -5000.0


def test_time_gate_floor_applies_in_seconds_before_conversion_to_ticks(tmp_path):
    # A Time gate 10-50 s: FlowJo counted everything from 0 s (the lower edge
    # is below the Time axis floor). In ticks at $TIMESTEP 0.01 the upper
    # edge is 5000; the open edge must stay open, not become -1e14.
    kw = '<Keywords><Keyword name="$TIMESTEP" value="0.01"/></Keywords>'
    path = _wsp(tmp_path, [('T', _dim('Time', 10, 50))], axes={'Time': 0}, keywords=kw)
    g = fp.WspReader(path).extract_gates()[0]
    assert g['lo'] == OPEN and g['hi'] == pytest.approx(5000.0)
    ticks = pd.DataFrame({'Time': np.arange(0, 6000, 10, dtype=float)})
    got = int(fp.gate_to_mask(dict(g, id='t', parent_id=None), ticks).sum())
    assert got == 500                     # every tick below 5000, from 0


def test_editor_path_with_its_own_parse_uses_sample_id(tmp_path):
    path = _wsp(tmp_path, [('G', _dim('SSC-A', 0, 250000))], axes={'SSC-A': 0})
    other_parse = ET.parse(path).getroot()
    for el in other_parse.iter():
        el.tag = el.tag.split('}', 1)[-1]
        for k in list(el.attrib):
            el.attrib[k.split('}', 1)[-1]] = el.attrib.pop(k)
    sn = other_parse.find('SampleList/Sample/SampleNode')
    gates = fp.WspReader(path).extract_gates(sample_node=sn)
    assert _lo(gates, 'G') == OPEN


def test_compare_inventory_applies_the_same_floor(tmp_path):
    path = _wsp(tmp_path, [('G', _dim('SSC-A', 0, 250000)),
                           ('H', _dim('SSC-A', 1000.0, 250000))], axes={'SSC-A': 0})
    (_, _, pops), = _per_sample_inventory(fp.WspReader(path))
    got = {name: g['lo'] for _, name, _, g, _ in pops}
    assert got == {'G': OPEN, 'H': 1000.0}


def test_polygons_stay_exact(tmp_path):
    out = tmp_path / 'poly.wsp'
    out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<Workspace '
        'xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" '
        'xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/transformations" '
        'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">'
        '<SampleList><Sample><Transformations>' + LIN.format(p='FSC-A', m=0)
        + LIN.format(p='SSC-A', m=0) + '</Transformations>'
        '<SampleNode name="s" sampleID="1"><Subpopulations><Population name="P">'
        '<Gate gating:id="p"><gating:PolygonGate>'
        + ''.join(f'<gating:dimension><data-type:fcs-dimension data-type:name="{c}"/>'
                  '</gating:dimension>' for c in ('FSC-A', 'SSC-A'))
        + ''.join(f'<gating:vertex><gating:coordinate data-type:value="{x}"/>'
                  f'<gating:coordinate data-type:value="{y}"/></gating:vertex>'
                  for x, y in ((0, 0), (1000, 0), (1000, 1000)))
        + '</gating:PolygonGate></Gate></Population></Subpopulations></SampleNode>'
        '</Sample></SampleList></Workspace>', encoding='utf-8')
    (g,) = fp.WspReader(str(out)).extract_gates()
    assert np.allclose(g['vertices'], [[0, 0], [1000, 0], [1000, 1000]])


@pytest.mark.parametrize('axis, lo, want', [
    # A minRange FlowJo was not measured on: only an edge AT minRange opens.
    ({'minRange': '100'}, 100.0, OPEN),
    ({'minRange': '100'}, 110.0, 110.0),
    # Another maxRange or gain was never measured: literal.
    ({'minRange': '0', 'maxRange': '1024'}, 0.0, 0.0),
    ({'minRange': '0', 'gain': '2'}, 0.0, 0.0),
])
def test_unmeasured_linear_axes_open_only_what_was_measured(tmp_path, axis, lo, want):
    attrs = {'minRange': '0', 'maxRange': '262144', 'gain': '1', **axis}
    lin = ('<transforms:linear transforms:minRange="{minRange}" '
           'transforms:maxRange="{maxRange}" gain="{gain}">'
           '<data-type:parameter data-type:name="SSC-A"/></transforms:linear>').format(**attrs)
    path = _wsp(tmp_path, [('G', _dim('SSC-A', lo, 250000))], axes={})
    txt = open(path, encoding='utf-8').read().replace(
        '<Transformations></Transformations>', f'<Transformations>{lin}</Transformations>')
    open(path, 'w', encoding='utf-8').write(txt)
    assert _lo(_by_name(path)[0], 'G') == want


def test_flowjo_one_sided_quadrant_sectors_are_imported(tmp_path):
    # FlowJo writes each quadrant sector with only its divider bound per
    # dimension. The reader used to skip such a rectangle, dropping the whole
    # quadrant and lifting its children out of their sector.
    path = _wsp(tmp_path, [
        ('UR', _dim('Comp-FL1-A', lo=900) + _dim('Comp-FL2-A', lo=1200)),
        ('LL', _dim('Comp-FL1-A', hi=900) + _dim('Comp-FL2-A', hi=1200)),
    ], axes=None)
    gates, _ = _by_name(path)
    got = {g['label'] if g.get('label') else g['name']: (g['x0'], g['x1'], g['y0'], g['y1'])
           for g in gates}
    assert got == {'UR': (900.0, 1e12, 1200.0, 1e12), 'LL': (OPEN, 900.0, OPEN, 1200.0)}
    (_, _, pops), = _per_sample_inventory(fp.WspReader(path))
    assert {n: (g['x0'], g['x1'], g['y0'], g['y1']) for _, n, _, g, _ in pops} == got


def test_compare_reads_a_max_only_range(tmp_path):
    path = _wsp(tmp_path, [('M', _dim('SSC-A', hi=5000))], axes=None)
    (_, _, pops), = _per_sample_inventory(fp.WspReader(path))
    (g,) = [g for _, _, _, g, _ in pops]
    assert (g['kind'], g['lo'], g['hi']) == ('interval', OPEN, 5000.0)


# ── Export: the writer cannot change FlowJo, so it says where FlowJo differs ──

def _export_warnings(tmp_path, gates, timestep=None):
    w = fp.WspWriter()
    w.add_sample('s', '', ['FSC-A', 'SSC-A', 'Comp-FL1-A', 'Time'], gates, timestep=timestep)
    w.write(str(tmp_path / 'out.wsp'))
    return w.warnings


OPENS, MAY = 'FlowJo treats it as open', 'may treat it as open'


@pytest.mark.parametrize('gate, expect', [
    ({'kind': 'interval', 'channel': 'SSC-A', 'lo': 0.0, 'hi': 1e5}, OPENS),
    ({'kind': 'interval', 'channel': 'SSC-A', 'lo': 20.0, 'hi': 1e5}, OPENS),
    ({'kind': 'interval', 'channel': 'SSC-A', 'lo': 50.0, 'hi': 1e5}, MAY),
    ({'kind': 'interval', 'channel': 'SSC-A', 'lo': 64.0, 'hi': 1e5}, None),
    ({'kind': 'interval', 'channel': 'SSC-A', 'lo': -1e12, 'hi': 1e5}, None),
    ({'kind': 'threshold', 'channel': 'FSC-A', 'value': 10.0}, OPENS),
    ({'kind': 'interval', 'channel': 'Comp-FL1-A', 'lo': -2500.0, 'hi': 1e5}, OPENS),
    ({'kind': 'interval', 'channel': 'Comp-FL1-A', 'lo': -1500.0, 'hi': 1e5}, MAY),
    ({'kind': 'interval', 'channel': 'Comp-FL1-A', 'lo': -500.0, 'hi': 1e5}, None),
    ({'kind': 'rect', 'x_channel': 'Comp-FL1-A', 'y_channel': 'SSC-A',
      'x0': 100.0, 'x1': 1e5, 'y0': 0.0, 'y1': 1e5}, OPENS),
    # A polygon opens from higher up than a rectangle (its bottom at 1000
    # was opened on an axis from 500 where a rectangle edge at 1000 was not):
    # a bottom at or below 1 was opened (measured), and any other bottom
    # inside the first 1024-unit cell is flagged as possibly opened.
    ({'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
      'vertices': [[2000, -2000], [5e4, -2000], [5e4, 5e4]]}, OPENS),
    ({'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
      'vertices': [[2000, 900], [5e4, 900], [5e4, 5e4]]}, MAY),
    ({'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
      'vertices': [[2000, 2000], [5e4, 2000], [5e4, 5e4]]}, None),
    # Nothing lies below 0 s: a Time gate from 0 is not flagged.
    ({'kind': 'interval', 'channel': 'Time', 'lo': 0.0, 'hi': 1e5}, None),
])
def test_export_warns_where_flowjo_opens_the_lower_edge(tmp_path, gate, expect):
    msgs = _export_warnings(tmp_path, [dict(gate, id='g', parent_id=None, name='G')],
                            timestep=0.01)
    hits = [m for m in msgs if 'lower edge' in m]
    if expect is None:
        assert hits == [], msgs
    else:
        assert hits and all(expect in m for m in hits), msgs


def test_export_warns_for_a_time_gate_in_seconds(tmp_path):
    # 1000 ticks x $TIMESTEP 0.01 = 10 s: opened by FlowJo. 10000 ticks =
    # 100 s: kept. The check runs on FlowJo's units, after conversion.
    g = {'kind': 'interval', 'channel': 'Time', 'lo': 1000.0, 'hi': 5000.0,
         'id': 't', 'parent_id': None, 'name': 'T'}
    msgs = _export_warnings(tmp_path, [g], timestep=0.01)
    assert any('lower edge 10 on Time' in m and OPENS in m for m in msgs), msgs
    assert not _export_warnings(tmp_path, [dict(g, lo=10000.0, hi=20000.0)], timestep=0.01)


def test_export_warns_when_time_cannot_be_converted(tmp_path):
    g = {'kind': 'interval', 'channel': 'Time', 'lo': 1000.0, 'hi': 5000.0,
         'id': 't', 'parent_id': None, 'name': 'T'}
    msgs = _export_warnings(tmp_path, [g])          # no FCS, no timestep
    assert any("on Time but the FCS gives no $TIMESTEP" in m for m in msgs), msgs


def test_export_names_compensated_channels_comp(tmp_path):
    # OpenFlo gates compensated values under the bare channel name; in FlowJo
    # the bare name is the RAW parameter and 'Comp-<name>' the compensated
    # one. Only channels in this sample's own matrix are compensated.
    gates = [
        {'id': 'a', 'parent_id': None, 'kind': 'interval', 'channel': 'FL1-A',
         'lo': 100.0, 'hi': 1e5, 'name': 'A'},
        {'id': 'b', 'parent_id': 'a', 'kind': 'rect', 'x_channel': 'FL2-A',
         'y_channel': 'SSC-A', 'x0': 100.0, 'x1': 1e5, 'y0': 100.0, 'y1': 1e5, 'name': 'B'},
        {'id': 'c', 'parent_id': 'a', 'kind': 'interval', 'channel': 'FL3-A',
         'lo': 100.0, 'hi': 1e5, 'name': 'C'},
    ]
    w = fp.WspWriter()
    w.add_sample('comp', '', ['FL1-A', 'FL2-A', 'FL3-A', 'SSC-A'], gates,
                 compensation=(['FL1-A', 'FL2-A'], np.eye(2)))
    w.add_sample('raw', '', ['FL1-A', 'FL2-A', 'FL3-A', 'SSC-A'], gates)
    out = tmp_path / 'c.wsp'
    w.write(str(out))
    D = '{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}'
    comp_s, raw_s = ET.parse(out).getroot().findall('SampleList/Sample')
    names = [[d.get(f'{D}name') for d in s.iter(f'{D}fcs-dimension')] for s in (comp_s, raw_s)]
    assert names[0] == ['Comp-FL1-A', 'Comp-FL2-A', 'SSC-A', 'FL3-A']
    assert names[1] == ['FL1-A', 'FL2-A', 'SSC-A', 'FL3-A']
    assert gates[0]['channel'] == 'FL1-A'                 # caller's dicts untouched
    # OpenFlo's reader strips the prefix back: the round trip is unchanged.
    r = fp.WspReader(str(out))
    back = r.extract_gates(sample_node=r.root.find('SampleList/Sample/SampleNode'))
    assert [g.get('channel') or g.get('x_channel') for g in back] == ['FL1-A', 'FL2-A', 'FL3-A']


def test_open_bounds_are_left_out_and_read_back(tmp_path):
    big = 1e12
    gates = [{'id': 'q', 'parent_id': None, 'kind': 'rect', 'label': 'UR',
              'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
              'x0': 900.0, 'x1': big, 'y0': 1200.0, 'y1': big},
             {'id': 'i', 'parent_id': None, 'kind': 'interval', 'name': 'below',
              'channel': 'SSC-A', 'lo': -big, 'hi': 5000.0},
             {'id': 'both', 'parent_id': None, 'kind': 'interval', 'name': 'all',
              'channel': 'SSC-A', 'lo': -big, 'hi': big}]
    w = fp.WspWriter()
    w.add_sample('s', '', ['FL1-A', 'FL2-A', 'SSC-A'], gates)
    out = tmp_path / 'o.wsp'
    w.write(str(out))
    xml = out.read_text(encoding='utf-8')
    assert '1000000000000' in xml                        # only the all-open one
    assert xml.count('1000000000000') == 2
    back = {(g.get('label') or g.get('name')): g for g in fp.WspReader(str(out)).extract_gates()}
    assert (back['UR']['x0'], back['UR']['x1'], back['UR']['y0'], back['UR']['y1']) == (
        900.0, big, 1200.0, big)
    assert (back['below']['lo'], back['below']['hi']) == (-big, 5000.0)
    assert (back['all']['lo'], back['all']['hi']) == (-big, big)


@pytest.mark.parametrize('display, warned', [('LIN', True), ('LOG', False), (None, True)])
def test_export_warning_follows_the_fcs_display_keyword(tmp_path, display, warned):
    # With no <Transformations> FlowJo puts a parameter on the axis its FCS
    # P<n>DISPLAY keyword names: scatter marked LOG got a biex axis (measured),
    # where an SSC lower edge at 0 is far above the biex floor and is kept.
    import flowio
    rng = np.random.default_rng(1)
    ev = rng.uniform(1e3, 1e5, size=(50, 2)).astype(np.float32)
    meta = {'P2DISPLAY': display} if display else {}
    path = tmp_path / 'd.fcs'
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.ravel().tolist(), ['FSC-A', 'SSC-A'], metadata_dict=meta)
    gate = {'id': 'g', 'parent_id': None, 'kind': 'interval', 'channel': 'SSC-A',
            'lo': 0.0, 'hi': 1e5, 'name': 'G'}
    w = fp.WspWriter()
    w.add_sample('s', str(path), ['FSC-A', 'SSC-A'], [gate])
    w.write(str(tmp_path / 'o.wsp'))
    assert any('lower edge 0 on SSC-A' in m for m in w.warnings) is warned, w.warnings


def test_each_sample_of_a_document_uses_its_own_axes(tmp_path):
    # Two samples whose SSC-A axes differ (minRange 0 vs -2000): an edge at
    # -100 is open on the first and kept on the second. Reading the whole
    # document must not carry one sample's floors over to the next.
    def sample(i, minrange):
        return (f'<Sample><DataSet uri="file:/x{i}.fcs" sampleID="{i}"/><Transformations>'
                + LIN.format(p='SSC-A', m=minrange) + '</Transformations>'
                f'<SampleNode name="s{i}" sampleID="{i}"><Subpopulations>'
                f'<Population name="G{i}"><Gate gating:id="g{i}"><gating:RectangleGate>'
                + _dim('SSC-A', -100, 250000)
                + '</gating:RectangleGate></Gate></Population></Subpopulations></SampleNode></Sample>')
    out = tmp_path / 'two.wsp'
    out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<Workspace '
        'xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" '
        'xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/transformations" '
        'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">'
        f'<SampleList>{sample(1, 0)}{sample(2, -2000)}</SampleList></Workspace>',
        encoding='utf-8')
    gates, _ = _by_name(str(out))
    assert (_lo(gates, 'G1'), _lo(gates, 'G2')) == (OPEN, -100.0)
    (_, _, p1), (_, _, p2) = _per_sample_inventory(fp.WspReader(str(out)))
    assert (p1[0][3]['lo'], p2[0][3]['lo']) == (OPEN, -100.0)


@pytest.mark.parametrize('ymin, warned', [(-1500.0, True), (-500.0, False)])
def test_export_flags_a_biex_polygon_near_the_floor_as_may_open(tmp_path, ymin, warned):
    # FlowJo's polygon floor on biex was not measured: a polygon reaching
    # below the kept-from bound gets the "may" warning, never "treats".
    g = {'id': 'p', 'parent_id': None, 'kind': 'polygon', 'name': 'P',
         'x_channel': 'Comp-FL1-A', 'y_channel': 'Comp-FL2-A',
         'vertices': [[100, ymin], [5e4, ymin], [5e4, 5e4]]}
    msgs = [m for m in _export_warnings(tmp_path, [g]) if 'lower edge' in m]
    assert bool(msgs) is warned, msgs
    assert all(MAY in m for m in msgs), msgs
