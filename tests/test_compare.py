"""compare.py had NO test file — the workspace<->FlowJo QC tool was unexecuted.
This pins the FlowJo-XML -> OpenFlo-gate translation (_gate_from_population),
which is a coordinate-transpose risk in the spirit of the compensation bug:
x/y channel order, rect min/max order, and polygon vertex (x,y) order.
"""
import os
import types
import xml.etree.ElementTree as ET

import pandas as pd

import openflo.pipeline as fp
from openflo.compare import (
    _compare_one_sample,
    _gate_from_population,
    _per_sample_inventory,
    _resolve_fcs_uri,
)

_WSP = """<Workspace><SampleList><Sample>
<DataSet uri="file:///data/s1.fcs"/>
<SampleNode name="S1"><Subpopulations>
<Population name="Cells" count="100"><Gate><RectangleGate>
<dimension min="0" max="9"><fcs-dimension name="FSC-A"/></dimension>
<dimension min="0" max="9"><fcs-dimension name="SSC-A"/></dimension>
</RectangleGate></Gate><Subpopulations>
<Population name="Live" count="40"><Gate><RectangleGate>
<dimension min="1" max="8"><fcs-dimension name="FSC-A"/></dimension>
<dimension min="1" max="8"><fcs-dimension name="SSC-A"/></dimension>
</RectangleGate></Gate></Population>
</Subpopulations></Population>
</Subpopulations></SampleNode></Sample></SampleList></Workspace>"""


def test_per_sample_inventory_walks_nested_populations():
    """_per_sample_inventory parses each Population's count, recurses through
    nested <Subpopulations>, and threads per-occurrence uid + parent_uid.
    Untested (compare.py had no test file)."""
    reader = types.SimpleNamespace(root=ET.fromstring(_WSP),
                                   _channel_name=fp.WspReader._channel_name)
    inv = _per_sample_inventory(reader)
    assert len(inv) == 1
    name, uri, pops = inv[0]
    assert name == 'S1' and uri == 'file:///data/s1.fcs'
    by_name = {p[1]: p for p in pops}                 # (uid, name, count, gate, parent)
    assert by_name['Cells'][2] == 100 and by_name['Live'][2] == 40
    assert by_name['Cells'][4] is None                # root has no parent
    assert by_name['Live'][4] == by_name['Cells'][0]  # Live nested under Cells
    assert by_name['Cells'][0] != by_name['Live'][0]  # distinct per-occurrence uids


def test_resolve_fcs_uri_file_uri_and_basename_fallback(tmp_path):
    """file: URI unquoting + Windows leading-slash-drive fix + fcs_dir basename
    fallback — the path mapping session reattach relies on. Untested."""
    f = tmp_path / 'my sample.fcs'                 # space → must be url-unquoted
    f.write_bytes(b'x')
    r = _resolve_fcs_uri(f.as_uri())               # file:///.../my%20sample.fcs
    assert r and os.path.samefile(r, str(f))
    stale = 'file:///Z:/gone/my%20sample.fcs'      # wrong dir, same basename
    r2 = _resolve_fcs_uri(stale, fcs_dir=str(tmp_path))
    assert r2 and os.path.samefile(r2, str(f))
    assert _resolve_fcs_uri(stale) is None         # no fcs_dir → unresolved
    assert _resolve_fcs_uri('') is None


def test_compare_one_sample_count_and_delta_arithmetic(tmp_path):
    """The headline QC numbers: openflo_count = cumulative-gate mask sum,
    delta = openflo - flowjo, rel_delta = delta/flowjo. compare.py had NO tests,
    so a sign-flip or wrong-denominator would have shipped silently."""
    from openflo.fcs_export import write_fcs
    fcs = tmp_path / 's.fcs'
    write_fcs(pd.DataFrame({'X': [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]}), str(fcs))
    # gate X > 5 → 5 events (6..10); declared FlowJo count 7 (deliberately off).
    pops = [(1, 'P', 7, {'kind': 'threshold', 'channel': 'X', 'value': 5.0}, None)]
    r = _compare_one_sample('s', str(fcs), pops)[0]
    assert r['openflo_count'] == 5
    assert r['flowjo_count'] == 7
    assert r['delta'] == 5 - 7                 # openflo - flowjo = -2
    assert r['rel_delta'] == (5 - 7) / 7


def test_gate_from_population_rect_channels_and_bounds():
    pop = ET.fromstring(
        '<Population><Gate><RectangleGate>'
        '<dimension min="10" max="20"><fcs-dimension name="FSC-A"/></dimension>'
        '<dimension min="30" max="40"><fcs-dimension name="SSC-A"/></dimension>'
        '</RectangleGate></Gate></Population>')
    g = _gate_from_population(pop, fp.WspReader)
    assert g == {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
                 'x0': 10.0, 'x1': 20.0, 'y0': 30.0, 'y1': 40.0}


def test_gate_from_population_polygon_vertex_order():
    pop = ET.fromstring(
        '<Population><Gate><PolygonGate>'
        '<dimension><fcs-dimension name="APC-A"/></dimension>'
        '<dimension><fcs-dimension name="PE-A"/></dimension>'
        '<vertex><coordinate value="1"/><coordinate value="2"/></vertex>'
        '<vertex><coordinate value="3"/><coordinate value="4"/></vertex>'
        '<vertex><coordinate value="5"/><coordinate value="6"/></vertex>'
        '</PolygonGate></Gate></Population>')
    g = _gate_from_population(pop, fp.WspReader)
    assert g['kind'] == 'polygon'
    assert g['x_channel'] == 'APC-A' and g['y_channel'] == 'PE-A'
    assert g['vertices'] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_gate_from_population_1d_interval_and_threshold():
    interval = ET.fromstring(
        '<Population><Gate><RectangleGate>'
        '<dimension min="2" max="5"><fcs-dimension name="CD3-A"/></dimension>'
        '</RectangleGate></Gate></Population>')
    assert _gate_from_population(interval, fp.WspReader) == {
        'kind': 'interval', 'channel': 'CD3-A', 'lo': 2.0, 'hi': 5.0}
    thr = ET.fromstring(
        '<Population><Gate><RectangleGate>'
        '<dimension min="7"><fcs-dimension name="CD3-A"/></dimension>'
        '</RectangleGate></Gate></Population>')
    assert _gate_from_population(thr, fp.WspReader) == {
        'kind': 'threshold', 'channel': 'CD3-A', 'value': 7.0}
