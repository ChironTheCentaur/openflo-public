"""WspReader.extract_gates' EllipsoidGate parser (parse_ellipsoid).

The full-suite coverage run executed 1 of its 47 statements: no test fed an
EllipsoidGate to the reader, although tests/test_ellipsoid_quadrant.py's
docstring says one does (it was dropped when the writer started emitting
ellipses as polygons).

Two forms exist:
  * Gating-ML 2.0: <mean>, <covarianceMatrix>, <distanceSquare>. Parsed.
  * FlowJo-native: <foci> + 4 <edge> vertices in FlowJo's 256x256 display
    space, no mean/covariance (FlowKit's WSPEllipsoidGate documents and
    parses this form). It used to be dropped, and its children re-parented
    to the grandparent, so they silently widened. It is now converted with
    the sample's axis transforms: exactly on linear axes, as a polygon on
    logicle axes. On any other axis it becomes a placeholder that admits no
    events, so its children stay under it and stay empty.

Hand-built XML in the namespaced shape WspWriter emits; synthetic events only.
The truth for a FlowJo-native ellipse is computed in FlowJo's display space
(display = fraction of the axis's transformed range x 256), independently of
the reader's conversion.
"""
import logging

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp

_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Workspace xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes" '
    'xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" version="20.0">'
    '<SampleList><Sample><SampleNode name="s1" sampleID="1"><Subpopulations>'
    '<Population name="Cells"><Gate gating:id="p"><gating:RectangleGate>'
    '<gating:dimension gating:min="-1000000.0" gating:max="1000000.0">'
    '<data-type:fcs-dimension data-type:name="FSC-A" /></gating:dimension>'
    '</gating:RectangleGate></Gate><Subpopulations>'
    '<Population name="Blob"><Gate gating:id="e">')
_TAIL = (
    '</Gate><Subpopulations><Population name="Child"><Gate gating:id="c">'
    '<gating:RectangleGate>'
    '<gating:dimension gating:min="0.0" gating:max="5.0">'
    '<data-type:fcs-dimension data-type:name="FL1-A" /></gating:dimension>'
    '<gating:dimension gating:min="0.0" gating:max="5.0">'
    '<data-type:fcs-dimension data-type:name="FL2-A" /></gating:dimension>'
    '</gating:RectangleGate></Gate></Population></Subpopulations></Population>'
    '</Subpopulations></Population></Subpopulations></SampleNode></Sample>'
    '</SampleList></Workspace>')


def _dims():
    return ('<gating:dimension><data-type:fcs-dimension data-type:name="FL1-A" />'
            '</gating:dimension><gating:dimension>'
            '<data-type:fcs-dimension data-type:name="FL2-A" /></gating:dimension>')


def _coord(v):
    return f'<gating:coordinate data-type:value="{v!r}" />'


def _gml_ellipse(mean, cov, dsq):
    rows = ''.join('<gating:row>' + ''.join(
        f'<gating:entry data-type:value="{e!r}" />' for e in r) + '</gating:row>'
        for r in cov)
    return ('<gating:EllipsoidGate>' + _dims()
            + '<gating:mean>' + _coord(mean[0]) + _coord(mean[1]) + '</gating:mean>'
            + '<gating:covarianceMatrix>' + rows + '</gating:covarianceMatrix>'
            + f'<gating:distanceSquare data-type:value="{dsq!r}" />'
            + '</gating:EllipsoidGate>')


def _flowjo_ellipse():
    # Centre (5,5), semi-axes 3 (x) / 2 (y) in data units, display 0..10 -> 0..256.
    s = 25.6
    a, b = 3 * s, 2 * s
    c = float(np.sqrt(a * a - b * b))

    def v(x, y):
        return '<gating:vertex>' + _coord(x) + _coord(y) + '</gating:vertex>'
    return ('<gating:EllipsoidGate eventsInside="1">' + _dims()
            + '<gating:foci>' + v(128 - c, 128) + v(128 + c, 128) + '</gating:foci>'
            + '<gating:edge>' + v(128 - a, 128) + v(128 + a, 128)
            + v(128, 128 - b) + v(128, 128 + b) + '</gating:edge>'
            + '</gating:EllipsoidGate>')


def _write(tmp_path, ellipse_xml):
    p = tmp_path / 'ell.wsp'
    p.write_text(_HEAD + ellipse_xml + _TAIL, encoding='utf-8')
    return str(p)


def _events(seed=7, n=40_000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FSC-A': rng.uniform(1, 1e5, n),
                         'FL1-A': rng.uniform(0, 10, n),
                         'FL2-A': rng.uniform(0, 10, n)})


_MEAN = [5.0, 4.0]
_COV = [[9.0, 2.4], [2.4, 4.0]]          # correlated, so x/y order matters
_DSQ = 1.5


def test_gatingml_ellipsoid_parses_mean_cov_distance(tmp_path):
    gates = fp.WspReader(_write(tmp_path, _gml_ellipse(_MEAN, _COV, _DSQ))).extract_gates()
    by_label = {g.get('label'): g for g in gates}
    ell = by_label['Blob']
    assert ell['kind'] == 'ellipsoid'
    assert (ell['x_channel'], ell['y_channel']) == ('FL1-A', 'FL2-A')
    assert ell['mean'] == _MEAN
    assert ell['cov'] == _COV
    assert ell['distance_sq'] == _DSQ
    # Hierarchy: Cells > Blob > Child.
    assert ell['parent_id'] == by_label['Cells']['_import_id']
    assert by_label['Child']['parent_id'] == ell['_import_id']


def test_gatingml_ellipsoid_selects_the_mahalanobis_set(tmp_path):
    gates, _ = fp.read_template_gates(_write(tmp_path, _gml_ellipse(_MEAN, _COV, _DSQ)))
    by_id = {g['id']: g for g in gates}
    blob = next(g for g in gates if g.get('label') == 'Blob')
    child = next(g for g in gates if g.get('label') == 'Child')
    df = _events()
    d = df[['FL1-A', 'FL2-A']].values - np.asarray(_MEAN)
    in_ell = np.einsum('ij,jk,ik->i', d, np.linalg.inv(_COV), d) <= _DSQ
    assert (fp.gate_to_mask(blob, df) == in_ell).all()
    in_rect = (df['FL1-A'].values < 5) & (df['FL2-A'].values < 5)
    got = fp.cumulative_gate_mask(by_id, child['id'], df)
    assert (np.asarray(got) == (in_ell & in_rect)).all()


def test_ellipsoid_per_sample_walk_matches_full_walk(tmp_path):
    # The editor's Add-FCS/Workspace path calls extract_gates(sample_node=...).
    r = fp.WspReader(_write(tmp_path, _gml_ellipse(_MEAN, _COV, _DSQ)))
    sn = next(r.root.iter('SampleNode'))
    per = r.extract_gates(sample_node=sn, timestep=None)
    full = r.extract_gates()
    strip = [{k: v for k, v in g.items() if k != '_import_id'} for g in full]
    assert [g['kind'] for g in per] == ['interval', 'ellipsoid', 'rect']
    assert [{k: v for k, v in g.items() if k not in ('_import_id', 'parent_id')}
            for g in per] == [{k: v for k, v in g.items() if k != 'parent_id'}
                              for g in strip]


def test_flowjo_native_ellipse_does_not_widen_its_child(tmp_path):
    # No <Transformations> here, so the ellipse cannot be converted. Dropping
    # it attached Child to Cells: 12,358 events where 2,405 were right.
    gates, _ = fp.read_template_gates(_write(tmp_path, _flowjo_ellipse()))
    by_id = {g['id']: g for g in gates}
    child = next(g for g in gates if g.get('label') == 'Child')
    df = _events()
    in_ell = ((df['FL1-A'] - 5) / 3) ** 2 + ((df['FL2-A'] - 5) / 2) ** 2 <= 1
    in_rect = (df['FL1-A'] < 5) & (df['FL2-A'] < 5)
    got = int(np.asarray(fp.cumulative_gate_mask(by_id, child['id'], df)).sum())
    truth = int((in_ell & in_rect).sum())
    # Either import the ellipse (count matches) or refuse/flag the subtree --
    # but never hand back a superset under the same name.
    assert got <= truth * 1.02


def _unreadable(kind):
    """The FlowJo-form ellipse above with points no reader can use."""
    x = _flowjo_ellipse()
    if kind == 'three_coordinates':
        return x.replace('<gating:edge><gating:vertex>',
                         '<gating:edge><gating:vertex>' + _coord(1.0), 1)
    if kind == 'not_a_number':
        return x.replace('<gating:edge><gating:vertex><gating:coordinate data-type:value="',
                         '<gating:edge><gating:vertex><gating:coordinate data-type:value="x', 1)
    start, end = x.index('<gating:edge>'), x.index('</gating:edge>') + len('</gating:edge>')
    return x[:start] + x[end:]                                     # no <edge>


@pytest.mark.parametrize('kind', ['three_coordinates', 'not_a_number', 'no_edge'])
def test_unreadable_flowjo_ellipse_is_a_placeholder_not_dropped(tmp_path, kind):
    """Dropping an unreadable FlowJo-form ellipse attached Child to Cells:
    every event of 40,000 inside Child's rectangle, in the reader and in the
    compare tool. It is an empty placeholder instead, keeping the element as
    FlowJo wrote it for the export."""
    import xml.etree.ElementTree as ET

    import flowio

    from openflo.compare import WspReader, _compare_one_sample, _per_sample_inventory
    path = _write(tmp_path, _unreadable(kind))
    gates, _ = fp.read_template_gates(path)
    blob, child = _blob_and_child(gates)
    assert blob['kind'] == 'flowjo_ellipse'
    assert child['parent_id'] == blob['id']
    by_id = {g['id']: g for g in gates}
    assert not np.asarray(fp.cumulative_gate_mask(by_id, child['id'], _events())).any()
    # The compare tool: Child stays under the ellipse, and the rows say why.
    ev = _events(n=2000)
    with open(tmp_path / 'u.fcs', 'wb') as fh:
        flowio.create_fcs(fh, ev.to_numpy(np.float32).flatten().tolist(), list(ev.columns))
    (sample, _uri, pops), = _per_sample_inventory(WspReader(path))
    rows = {r['population']: r for r in _compare_one_sample(sample, str(tmp_path / 'u.fcs'), pops)}
    for pop in ('Blob', 'Child'):
        assert rows[pop]['openflo_count'] == 0
        assert 'FlowJo ellipse not converted' in rows[pop]['error']
    # The export writes the element back as it was read, Child under it.
    w = fp.WspWriter()
    w.add_sample('s1', '', ['FSC-A', 'FL1-A', 'FL2-A'], gates)
    w.write(str(tmp_path / 'out.wsp'))
    assert _population_paths(str(tmp_path / 'out.wsp'))['Child'] == 'Cells>Blob>Child'

    def shape(el):
        return (el.tag.split('}')[-1],
                sorted((k.split('}')[-1], v) for k, v in el.attrib.items()),
                [shape(c) for c in el])
    src = fp.WspReader(path).root.find('.//EllipsoidGate')
    out = next(ET.parse(tmp_path / 'out.wsp').getroot().iter(f'{_GATING}EllipsoidGate'))
    assert shape(out) == shape(src)


def test_parsed_ellipse_is_not_reported_as_skipped(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=fp.log.name):
        gates = fp.WspReader(_write(tmp_path, _gml_ellipse(_MEAN, _COV, _DSQ))).extract_gates()
    assert any(g['kind'] == 'ellipsoid' for g in gates)
    assert not any('EllipsoidGate' in r.getMessage() for r in caplog.records)


# ── FlowJo-native ellipses on declared axes ──────────────────────────────────

_TX_NS = 'xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/transformations" '
_LOGICLE = dict(t=262144.0, w=1.0, m=4.418539922, a=0.0)


def _transforms(kind, min_range=3.0):
    def one(name):
        p = f'<data-type:parameter data-type:name="{name}" />'
        if kind == 'linear':
            return (f'<transforms:linear transforms:minRange="{min_range!r}" '
                    f'transforms:maxRange="60000" gain="1">{p}</transforms:linear>')
        if kind == 'logicle':
            return ('<transforms:logicle transforms:length="256" transforms:T="262144" '
                    'transforms:A="0" transforms:W="1" transforms:M="4.418539922">'
                    f'{p}</transforms:logicle>')
        if kind == 'log':
            # An axis OpenFlo does not convert.
            return ('<transforms:log transforms:offset="1" transforms:decades="5">'
                    f'{p}</transforms:log>')
        return ('<transforms:biex transforms:length="256" transforms:maxRange="262144" '
                'transforms:neg="0" transforms:width="-100" transforms:pos="4.418539922">'
                f'{p}</transforms:biex>')
    return '<Transformations>' + one('FL1-A') + one('FL2-A') + '</Transformations>'


def _display_ellipse(centre, a, b, angle_deg):
    """FlowJo's form: foci, then edge points (major-axis ends first), all in
    the 256x256 display space."""
    th = np.radians(angle_deg)
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-u[1], u[0]])
    c = np.asarray(centre, dtype=float)
    f = np.sqrt(a * a - b * b)

    def vx(p):
        return ('<gating:vertex>' + _coord(float(p[0])) + _coord(float(p[1]))
                + '</gating:vertex>')
    return ('<gating:EllipsoidGate eventsInside="1">' + _dims()
            + '<gating:foci>' + vx(c - f * u) + vx(c + f * u) + '</gating:foci>'
            + '<gating:edge>' + vx(c - a * u) + vx(c + a * u)
            + vx(c - b * v) + vx(c + b * v) + '</gating:edge>'
            + '</gating:EllipsoidGate>')


def _in_display_ellipse(disp, centre, a, b, angle_deg):
    th = np.radians(angle_deg)
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-u[1], u[0]])
    d = disp - np.asarray(centre, dtype=float)
    return ((d @ u) / a) ** 2 + ((d @ v) / b) ** 2 <= 1


def _write_native(tmp_path, ellipse_xml, kind, child_hi, min_range=3.0):
    head = (_HEAD.replace('<Workspace ', '<Workspace ' + _TX_NS, 1)
            .replace('<Sample>', '<Sample>' + _transforms(kind, min_range), 1))
    tail = _TAIL.replace('gating:min="0.0" gating:max="5.0"',
                         f'gating:min="-1000000.0" gating:max="{child_hi!r}"')
    p = tmp_path / f'native_{kind}.wsp'
    p.write_text(head + ellipse_xml + tail, encoding='utf-8')
    return str(p)


def _blob_and_child(gates):
    by_label = {g.get('label'): g for g in gates}
    return by_label['Blob'], by_label['Child']


_ELL = ((100.0, 140.0), 40.0, 15.0, 30.0)     # centre, a, b, angle (display units)


def test_flowjo_native_ellipse_on_linear_axes_is_exact(tmp_path):
    # Linear 3..60000: display -> data is affine, so the ellipse is exact.
    path = _write_native(tmp_path, _display_ellipse(*_ELL), 'linear', 20000.0)
    gates, _ = fp.read_template_gates(path)
    blob, child = _blob_and_child(gates)
    assert blob['kind'] == 'ellipsoid'
    assert child['parent_id'] == blob['id']
    rng = np.random.default_rng(11)
    n = 100_000
    df = pd.DataFrame({'FSC-A': rng.uniform(1, 1e5, n),
                       'FL1-A': rng.uniform(3, 60000, n),
                       'FL2-A': rng.uniform(3, 60000, n)})
    disp = (df[['FL1-A', 'FL2-A']].to_numpy() - 3.0) / (60000.0 - 3.0) * 256.0
    in_ell = _in_display_ellipse(disp, *_ELL)
    assert in_ell.sum() > 1000, 'fixture no longer covers the ellipse'
    assert (np.asarray(fp.gate_to_mask(blob, df)) == in_ell).all()
    in_rect = ((df['FL1-A'] < 20000) & (df['FL2-A'] < 20000)).to_numpy()
    got = np.asarray(fp.cumulative_gate_mask({g['id']: g for g in gates}, child['id'], df))
    assert (got == (in_ell & in_rect)).all()


def test_flowjo_native_ellipse_on_logicle_axes_is_a_polygon(tmp_path, caplog):
    centre, a, b, ang = (80.0, 70.0), 33.0, 20.0, -35.0
    x_mid = float(fp.inverse_transform_values([centre[0] / 256], 'logicle', **_LOGICLE)[0])
    path = _write_native(tmp_path, _display_ellipse(centre, a, b, ang), 'logicle', x_mid)
    with caplog.at_level(logging.WARNING, logger=fp.log.name):
        gates, _ = fp.read_template_gates(path)
    blob, child = _blob_and_child(gates)
    assert blob['kind'] == 'polygon'
    assert child['parent_id'] == blob['id']
    # The approximation is said out loud, without claiming one fixed size for
    # it: the two real FlowJo logicle ellipses measured differ by 2.2% and 3.2%.
    msgs = [r.getMessage() for r in caplog.records if "'Blob'" in r.getMessage()]
    assert any("FlowJo's own count can differ" in m for m in msgs)
    assert not any('%' in m for m in msgs)
    # Events drawn uniformly over the display, mapped to data with OpenFlo's
    # own logicle inverse; truth is the ellipse in display space.
    rng = np.random.default_rng(5)
    n = 100_000
    disp = rng.uniform(0, 256, size=(n, 2))
    data = fp.inverse_transform_values(disp.ravel() / 256, 'logicle',
                                       **_LOGICLE).reshape(n, 2)
    df = pd.DataFrame({'FSC-A': rng.uniform(1, 1e5, n),
                       'FL1-A': data[:, 0], 'FL2-A': data[:, 1]})
    in_ell = _in_display_ellipse(disp, centre, a, b, ang)
    got = np.asarray(fp.gate_to_mask(blob, df))
    # A 64-gon in display space, mapped vertex by vertex: only events within
    # a hair of the outline can differ.
    assert in_ell.sum() > 1000
    assert (got != in_ell).sum() <= 0.01 * in_ell.sum()
    assert abs(int(got.sum()) - int(in_ell.sum())) <= 0.005 * in_ell.sum()
    xy = df[['FL1-A', 'FL2-A']].to_numpy()
    in_rect = ((xy >= -1e6) & (xy < x_mid)).all(axis=1)
    kid = np.asarray(fp.cumulative_gate_mask({g['id']: g for g in gates}, child['id'], df))
    assert (kid == (got & in_rect)).all()


def test_linear_axis_display_zero_is_min_range_not_flowkits_offset(tmp_path):
    """Pins which linear form is used when minRange != 0 (here 5000):
    display = (x - minRange) / (maxRange - minRange) * 256, where FlowKit uses
    (x + minRange) / (maxRange + minRange) * 256. FlowJo 10.10.2 placed the
    bottom of a linear axis it saved with minRange 500 or -2000 AT minRange
    (pipeline._LINEAR_FLOORS_MEASURED), which FlowKit's form contradicts; both
    forms agree at minRange 0, the only case measured with an ellipse."""
    lo, hi = 5000.0, 60000.0
    path = _write_native(tmp_path, _display_ellipse(*_ELL), 'linear', 1e9, min_range=lo)
    gates, _ = fp.read_template_gates(path)
    blob, _child = _blob_and_child(gates)
    rng = np.random.default_rng(13)
    n = 100_000
    xy = rng.uniform(-10000, 60000, size=(n, 2))
    df = pd.DataFrame({'FSC-A': rng.uniform(1, 1e5, n), 'FL1-A': xy[:, 0], 'FL2-A': xy[:, 1]})
    openflo_form = _in_display_ellipse((xy - lo) / (hi - lo) * 256.0, *_ELL)
    flowkit_form = _in_display_ellipse((xy + lo) / (hi + lo) * 256.0, *_ELL)
    got = np.asarray(fp.gate_to_mask(blob, df))
    assert openflo_form.sum() > 1000 and flowkit_form.sum() > 1000
    assert (openflo_form != flowkit_form).sum() > 1000, 'the forms must disagree here'
    assert (got == openflo_form).all()


# FlowJo's own biex lookup table for width -7.943282, neg 1, pos 4.418540,
# maxRange 262144.000029, as FlowJo exported it (FlowKit's test data,
# data/flowjo_xforms; printed to 6 significant digits): channel -> data value.
_FLOWJO_BIEX_TABLE = {0: -865.899, 512: -211.214, 1024: -19.4534, 1536: 139.301,
                      2048: 578.468, 3072: 11648.9, 4095: 260651.0}


def test_biex_table_matches_the_one_flowjo_exports():
    data, chan = fp._flowjo_biex_lut(1.0, -7.943282, 4.418540, 262144.000029)
    assert len(chan) == 4097 and chan[0] == 0 and chan[-1] == 4096
    for i, v in _FLOWJO_BIEX_TABLE.items():
        assert abs(data[i] - v) <= 5e-6 * abs(v) + 5e-4, (i, data[i], v)
    # Not a biex axis FlowJo can draw: no table, so the axis is not converted.
    assert fp._flowjo_biex_lut(0.0, 10.0, 4.42, 262144.0) is None


def test_flowjo_native_ellipse_on_biex_axes_is_a_polygon(tmp_path):
    # FlowJo's default fluorescence axis. Events drawn uniformly over the
    # display, mapped to data through FlowJo's table (pinned above); truth is
    # the ellipse in display space.
    centre, a, b, ang = (150.0, 120.0), 45.0, 18.0, 20.0
    data_lut, chan = fp._flowjo_biex_lut(0.0, -100.0, 4.418539922, 262144.0)

    def to_data(d):
        return np.interp(np.asarray(d) * 16.0, chan, data_lut)
    x_mid = float(to_data(centre[0]))
    path = _write_native(tmp_path, _display_ellipse(centre, a, b, ang), 'biex', x_mid)
    gates, _ = fp.read_template_gates(path)
    blob, child = _blob_and_child(gates)
    assert blob['kind'] == 'polygon'
    assert child['parent_id'] == blob['id']
    rng = np.random.default_rng(17)
    n = 100_000
    disp = rng.uniform(0, 256, size=(n, 2))
    data = to_data(disp)
    df = pd.DataFrame({'FSC-A': rng.uniform(1, 1e5, n),
                       'FL1-A': data[:, 0], 'FL2-A': data[:, 1]})
    in_ell = _in_display_ellipse(disp, centre, a, b, ang)
    got = np.asarray(fp.gate_to_mask(blob, df))
    assert in_ell.sum() > 1000
    assert (got != in_ell).sum() <= 0.01 * in_ell.sum()
    assert abs(int(got.sum()) - int(in_ell.sum())) <= 0.005 * in_ell.sum()
    kid = np.asarray(fp.cumulative_gate_mask({g['id']: g for g in gates}, child['id'], df))
    assert (kid == (got & (df['FL1-A'] < x_mid).to_numpy())).all()


def test_biex_ellipse_past_the_axis_bottom_takes_the_events_below_the_axis(tmp_path):
    """An event below a biex axis sits at its bottom (display 0), as FlowKit
    clamps it and FlowJo draws it. Mapping the outline's off-axis vertices to
    the table's bottom value left those events out (352 where FlowKit counts
    359 on the real ICS ellipse moved down 60 bins on a width -10 axis)."""
    centre, a, b, ang = (100.0, 10.0), 40.0, 20.0, 0.0
    width = -10.0
    path = _write_native(tmp_path, _display_ellipse(centre, a, b, ang), 'biex', 1e9)
    with open(path, encoding='utf-8') as fh:
        text = fh.read().replace('transforms:width="-100"', f'transforms:width="{width!r}"')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    gates, _ = fp.read_template_gates(path)
    blob, _child = _blob_and_child(gates)
    assert blob['kind'] == 'polygon'
    lut_data, chan = fp._flowjo_biex_lut(0.0, width, 4.418539922, 262144.0)
    rng = np.random.default_rng(29)
    n = 60_000
    disp = rng.uniform(0, 256, size=(n, 2))
    data = np.interp(disp * 16.0, chan, lut_data)
    # A third of the events lie below the axis bottom on y.
    data[: n // 3, 1] = rng.uniform(-20_000, lut_data[0], n // 3)
    df = pd.DataFrame({'FSC-A': rng.uniform(1, 1e5, n), 'FL1-A': data[:, 0], 'FL2-A': data[:, 1]})
    clamped = np.interp(data, lut_data, chan) / 16.0       # FlowKit's forward, clamped
    truth = _in_display_ellipse(clamped, centre, a, b, ang)
    off_axis_in = truth[: n // 3].sum()
    assert off_axis_in > 1000, 'fixture no longer puts events under the ellipse'
    got = np.asarray(fp.gate_to_mask(blob, df))
    assert got[: n // 3].sum() >= 0.99 * off_axis_in
    assert (got != truth).sum() <= 0.01 * truth.sum()


def test_flowjo_ellipse_on_an_unconverted_axis_is_an_empty_placeholder(tmp_path, caplog):
    # A log axis is not converted. The editor's path: extract_gates(sample_node=...).
    r = fp.WspReader(_write_native(tmp_path, _display_ellipse(*_ELL), 'log', 5.0))
    sn = next(r.root.iter('SampleNode'))
    with caplog.at_level(logging.WARNING, logger=fp.log.name):
        gates = r.extract_gates(sample_node=sn, timestep=None)
    blob, child = _blob_and_child(gates)
    assert blob['kind'] == 'flowjo_ellipse'
    assert child['parent_id'] == blob['_import_id']
    # FlowJo's own points and axes are kept for the export.
    fj = blob['flowjo']
    assert fj['parameters'] == ['FL1-A', 'FL2-A']
    assert len(fj['foci']) == 2 and len(fj['edge']) == 4
    assert [ax['kind'] for ax in fj['axes']] == ['log', 'log']
    assert fj['axes'][0]['attrs'] == {'offset': '1', 'decades': '5'}
    by_id = {g['_import_id']: g for g in gates}
    df = _events()
    assert not np.asarray(fp.cumulative_gate_mask(by_id, blob['_import_id'], df)).any()
    assert not np.asarray(fp.cumulative_gate_mask(by_id, child['_import_id'], df)).any()
    assert 'not converted' in fp.describe_gate(blob)
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("'Blob'" in m and 'log axis' in m and 'no events' in m for m in msgs)
    # Imported as a placeholder, so not called "skipped".
    assert any('imported as empty placeholders' in m and 'Blob' in m for m in msgs)
    assert not any('skipped' in m for m in msgs)


def _native_workspace_with_fcs(tmp_path, kind='log'):
    """A FlowJo-drawn ellipse workspace whose <DataSet> points at a moved FCS
    in tmp_path (500 events)."""
    import flowio
    rng = np.random.default_rng(2)
    ev = np.column_stack([rng.uniform(1, 1e5, 500), rng.uniform(0, 10, 500),
                          rng.uniform(0, 10, 500)]).astype(np.float32)
    with open(tmp_path / 'ell.fcs', 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), ['FSC-A', 'FL1-A', 'FL2-A'])
    path = _write_native(tmp_path, _display_ellipse(*_ELL), kind, 5.0)
    with open(path, encoding='utf-8') as fh:
        text = fh.read().replace(
            '<Sample>', '<Sample><DataSet uri="file:/Z:/gone/ell.fcs" sampleID="1"/>', 1)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    return path


def _editor_with_import(tmp_path, path):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_gui_smoke import _editor_or_skip
    root, ed, _gui = _editor_or_skip()
    queued = []
    ed._queue_fcs_loads = lambda paths, *a, **k: queued.extend(paths)
    ed._ingest_wsp(path)
    ed.ingest_status = ed.status_var.get()      # before the load replaces it
    assert len(queued) == 1
    (name,) = ed._pending_sample_gates
    ed._on_loaded(name, fp.FlowSample(queued[0]))
    return root, ed, name


def test_editor_import_keeps_the_placeholder_and_its_empty_child(tmp_path):
    """The editor's Add FCS / Workspace path end to end: the placeholder and
    its child land in the sample's gate tree, both count 0, the gate list and
    plot redraw with them, and the status bar says so once the load settles
    (the log alone went unseen)."""
    from openflo.gating import gate_counts
    root, ed, name = _editor_with_import(tmp_path, _native_workspace_with_fcs(tmp_path))
    try:
        # Named at import too, not only after the load settles.
        assert 'queued 1 sample(s), 3 gate(s)' in ed.ingest_status
        assert '1 FlowJo ellipse(s) not converted (Blob)' in ed.ingest_status
        gates = ed._sample_gates[name]
        by_label = {g.get('label'): (gid, g) for gid, g in gates.items()}
        (blob_id, blob), (child_id, child) = by_label['Blob'], by_label['Child']
        assert blob['kind'] == 'flowjo_ellipse' and child['parent_id'] == blob_id
        data = ed._samples[name].data
        assert gate_counts(gates, blob_id, data, None)[0] == 0
        assert gate_counts(gates, child_id, data, None)[0] == 0
        ed._on_load_settled()
        status = ed.status_var.get()
        assert '1 FlowJo ellipse(s) not converted (Blob)' in status, status
        assert 'count no events' in status
        ed._refresh_gate_list()
        ed._replot()
        assert 'not converted' in ed.gate_tv.item(ed._gate_iid(name, blob_id), 'text')
    finally:
        root.destroy()


def test_session_save_and_restore_keep_the_placeholder(tmp_path):
    """A .flowsession carries the placeholder and FlowJo's points as plain
    JSON (no schema change: still v1 keys), and a restore puts it back with
    its child under it, still empty."""
    import json
    import os

    from openflo.gating import gate_counts
    from openflo.session_format import SESSION_KEYS, SESSION_VERSION
    root, ed, name = _editor_with_import(tmp_path, _native_workspace_with_fcs(tmp_path))
    try:
        state = ed._session_state()
        assert set(state) == set(SESSION_KEYS) and state['version'] == SESSION_VERSION
        saved = json.loads(json.dumps(state, allow_nan=False))
        sample_path = ed._samples[name].path
    finally:
        root.destroy()
    blob = next(g for g in saved['sample_gates'][name] if g.get('label') == 'Blob')
    assert blob['kind'] == 'flowjo_ellipse' and len(blob['flowjo']['edge']) == 4
    from test_gui_smoke import _editor_or_skip  # on sys.path via _editor_with_import
    root, ed, _gui = _editor_or_skip()
    try:
        pkey = os.path.normcase(os.path.abspath(sample_path))
        ed._pending_sample_meta[pkey] = {'gates': saved['sample_gates'][name]}
        ed._on_loaded(name, fp.FlowSample(sample_path))
        gates = ed._sample_gates[name]
        by_label = {g.get('label'): (gid, g) for gid, g in gates.items()}
        (blob_id, blob2), (child_id, child) = by_label['Blob'], by_label['Child']
        assert blob2['kind'] == 'flowjo_ellipse' and child['parent_id'] == blob_id
        assert blob2['flowjo'] == blob['flowjo']
        data = ed._samples[name].data
        assert gate_counts(gates, child_id, data, None)[0] == 0
    finally:
        root.destroy()


def test_a_restored_boolean_on_the_placeholder_is_empty_and_named(tmp_path):
    """A session whose boolean combines a gate under the placeholder (only a
    hand-made or older file can hold one; the dialog offers none): restored,
    NOT(Child) selects nothing, not every event, and the status bar names it
    with the placeholder once the load settles."""
    import json
    import os

    from openflo.gating import gate_counts
    root, ed, name = _editor_with_import(tmp_path, _native_workspace_with_fcs(tmp_path))
    try:
        saved = json.loads(json.dumps(ed._session_state(), allow_nan=False))
        sample_path = ed._samples[name].path
    finally:
        root.destroy()
    gates = saved['sample_gates'][name]
    child = next(g for g in gates if g.get('label') == 'Child')
    gates.append({'id': 'g99', 'kind': 'boolean', 'op': 'not',
                  'operands': [child['id']], 'parent_id': None,
                  'name': 'NOT Child', 'enabled': True})
    from test_gui_smoke import _editor_or_skip  # on sys.path via _editor_with_import
    root, ed, _gui = _editor_or_skip()
    try:
        pkey = os.path.normcase(os.path.abspath(sample_path))
        ed._pending_sample_meta[pkey] = {'gates': gates}
        ed._on_loaded(name, fp.FlowSample(sample_path))
        restored = ed._sample_gates[name]
        not_id = next(gid for gid, g in restored.items()
                      if g.get('name') == 'NOT Child')
        assert gate_counts(restored, not_id, ed._samples[name].data, None)[0] == 0
        ed._on_load_settled()
        status = ed.status_var.get()
        assert '1 FlowJo ellipse(s) not converted (Blob)' in status, status
        assert "boolean gate(s) built on them: 'NOT Child'" in status, status
    finally:
        root.destroy()


def test_boolean_dialog_offers_neither_the_placeholder_nor_its_descendants(tmp_path):
    """Under a placeholder every population is empty, so NOT of any of them
    reads as every event (NOT(Child) was 500 of 500): none is offered."""
    from tkinter import ttk
    root, ed, name = _editor_with_import(tmp_path, _native_workspace_with_fcs(tmp_path))
    try:
        ed._set_active_sample(name)
        ed._open_boolean_dialog()
        dlg = next(w for w in ed.winfo_children() if str(w.winfo_class()) == 'Toplevel'
                   and w.title().startswith('Boolean gate'))

        def checkbuttons(w):
            out = [w.cget('text')] if isinstance(w, ttk.Checkbutton) else []
            for c in w.winfo_children():
                out += checkbuttons(c)
            return out
        offered = checkbuttons(dlg)
        dlg.destroy()
    finally:
        root.destroy()
    assert len(offered) == 1 and offered[0].startswith('I  FSC-A'), offered


def _cli_fcs(tmp_path):
    import flowio
    ev = _events(n=2000)
    path = tmp_path / 'cli.fcs'
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.to_numpy(np.float32).flatten().tolist(), list(ev.columns))
    return str(path)


def test_cli_gates_keep_an_ellipse_and_restrict_its_child(tmp_path):
    """--gates skipped 'ellipsoid' (a FlowJo ellipse on linear axes imports as
    one), so its child applied unrestricted and --export-wsp wrote the child
    as a root."""
    import json

    from openflo.cli import _build_export_gate_list, _parse_gates_arg
    ell = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
           'mean': [5.0, 5.0], 'cov': [[4.0, 0.0], [0.0, 1.0]], 'distance_sq': 1.0,
           'id': 'e', 'parent_id': None, 'label': 'Blob'}
    child = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
             'x0': -1e6, 'x1': 5.0, 'y0': -1e6, 'y1': 5.0,
             'id': 'c', 'parent_id': 'e', 'label': 'Child'}
    overrides, region = _parse_gates_arg(json.dumps([ell, child]))
    assert [g['label'] for g in region] == ['Blob', 'Child']
    s = fp.FlowSample(_cli_fcs(tmp_path))
    x, y = s.data['FL1-A'].to_numpy(float), s.data['FL2-A'].to_numpy(float)
    truth = (((x - 5) ** 2 / 4 + (y - 5) ** 2) <= 1) & (x < 5) & (y < 5)
    assert 10 < truth.sum() < ((x < 5) & (y < 5)).sum() * 0.5
    s.apply_region_gates(region)
    assert len(s.data) == int(truth.sum())
    w = fp.WspWriter()
    w.add_sample('s1', '', [], _build_export_gate_list(region, overrides))
    w.write(str(tmp_path / 'cli.wsp'))
    assert _population_paths(str(tmp_path / 'cli.wsp'))['Child'] == 'Blob>Child'


def test_cli_gates_keep_an_unconverted_flowjo_ellipse_and_admit_nothing(tmp_path, capsys):
    """--gates keeps the placeholder (skipping it would admit every event it
    excluded), says it admits nothing, and the run's region gates keep none."""
    import json

    from openflo.cli import _parse_gates_arg
    gates = fp.WspReader(_write_native(tmp_path, _display_ellipse(*_ELL), 'log', 5.0)
                         ).extract_gates()
    blob, _child = _blob_and_child(gates)
    _overrides, region = _parse_gates_arg(json.dumps([blob]))
    assert [g['kind'] for g in region] == ['flowjo_ellipse']
    assert "FlowJo ellipse 'Blob' was not converted: it admits no events" in capsys.readouterr().out
    s = fp.FlowSample(_cli_fcs(tmp_path))
    s.apply_region_gates(region)
    assert len(s.data) == 0


def test_export_keeps_flowjos_gating_distance_attribute_namespaced(tmp_path):
    """FlowJo writes gating:distance on its EllipsoidGate; the export must put
    it back in the gating namespace, not as a plain attribute."""
    import xml.etree.ElementTree as ET
    xml = _display_ellipse(*_ELL).replace(
        '<gating:EllipsoidGate eventsInside="1">',
        '<gating:EllipsoidGate eventsInside="1" gating:distance="70.9149179906794">')
    gates = fp.WspReader(_write_native(tmp_path, xml, 'log', 5.0)).extract_gates()
    w = fp.WspWriter()
    w.add_sample('s1', '', [], gates)
    w.write(str(tmp_path / 'd.wsp'))
    ell = next(ET.parse(tmp_path / 'd.wsp').getroot().iter(f'{_GATING}EllipsoidGate'))
    assert ell.get(f'{_GATING}distance') == '70.9149179906794'
    assert ell.get('distance') is None and ell.get('eventsInside') == '1'


_GATING = '{http://www.isac-net.org/std/Gating-ML/v2.0/gating}'
_DTYPE = '{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}'
_TXF = '{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}'


def _population_paths(wsp_path):
    """{population name: path of names from the sample root} in a .wsp."""
    import xml.etree.ElementTree as ET
    out = {}

    def walk(pop, trail):
        here = (*trail, pop.get('name'))
        out[pop.get('name')] = '>'.join(here)
        for sub in pop.findall('Subpopulations'):
            for p in sub.findall('Population'):
                walk(p, here)
    for sn in ET.parse(wsp_path).getroot().iter('SampleNode'):
        for sub in sn.findall('Subpopulations'):
            for p in sub.findall('Population'):
                walk(p, ())
    return out


def test_editor_export_writes_the_flowjo_ellipse_back_and_keeps_its_child(
        tmp_path, monkeypatch):
    """Export to FlowJo used to drop the placeholder (a 'boolean') and move
    Child under Cells, so FlowJo and a re-import counted Child unrestricted
    (118 of 500 events here). The ellipse goes back as FlowJo drew it, on the
    axes it was drawn on, with Child under it."""
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb
    import xml.etree.ElementTree as ET
    src = _native_workspace_with_fcs(tmp_path)
    root, ed, name = _editor_with_import(tmp_path, src)
    out = tmp_path / 'exported.wsp'
    monkeypatch.setattr(fd, 'asksaveasfilename', lambda *a, **k: str(out))
    # Imported gates start disabled, which the lossy-export prompt lists.
    monkeypatch.setattr(mb, 'askyesnocancel', lambda *a, **k: True)
    try:
        ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
            on_done(work()) if on_done else work())
        ed._set_active_sample(name)
        ed._export_flowjo_wsp()
        status = ed.status_var.get()
    finally:
        root.destroy()
    assert _population_paths(str(out))['Child'] == 'Cells>Blob>Child'
    assert 'Exported 1 sample(s) / 3 gate(s)' in status
    assert '1 place(s) FlowJo will count differently' in status
    # FlowJo's own form: the same points, attributes and axes as the source.
    src_ell = fp.WspReader(src).root.find('.//EllipsoidGate')
    ell = next(ET.parse(out).getroot().iter(f'{_GATING}EllipsoidGate'))
    assert ell.get('eventsInside') == '1'
    for tag in ('foci', 'edge'):
        got = [[float(c.get(f'{_DTYPE}value')) for c in v]
               for v in ell.find(f'{_GATING}{tag}')]
        want = [[float(c.get('value')) for c in v] for v in src_ell.find(tag)]
        assert got == want
    sample = next(ET.parse(out).getroot().iter('Sample'))
    axes = {t.find(f'{_DTYPE}parameter').get(f'{_DTYPE}name'):
            (t.tag, dict(t.attrib)) for t in sample.find('Transformations')}
    assert axes == {ch: (f'{_TXF}log', {f'{_TXF}offset': '1', f'{_TXF}decades': '5'})
                    for ch in ('FL1-A', 'FL2-A')}
    # Re-read: the placeholder again (OpenFlo still cannot convert a log
    # axis), Child under it, both empty.
    gates = fp.WspReader(str(out)).extract_gates()
    blob, child = _blob_and_child(gates)
    assert blob['kind'] == 'flowjo_ellipse'
    assert child['parent_id'] == blob['_import_id']
    by_id = {g['_import_id']: g for g in gates}
    assert not np.asarray(fp.cumulative_gate_mask(by_id, child['_import_id'], _events())).any()


def test_writer_round_trips_the_flowjo_ellipse_with_compensated_names(tmp_path):
    """Without the GUI: reader -> WspWriter -> reader. On a sample the export
    compensates, the dimensions AND the axes are named 'Comp-<ch>', as FlowJo
    names a compensated parameter."""
    import xml.etree.ElementTree as ET
    gates = fp.WspReader(_write_native(tmp_path, _display_ellipse(*_ELL), 'log', 5.0)
                         ).extract_gates()
    w = fp.WspWriter()
    w.add_sample('s1', '', ['FSC-A', 'FL1-A', 'FL2-A'], gates,
                 compensation=(['FL1-A', 'FL2-A'], np.eye(2)))
    out = tmp_path / 'rt.wsp'
    w.write(str(out))
    assert any("FlowJo ellipse 'Blob' written back" in m for m in w.warnings)
    ell = next(ET.parse(out).getroot().iter(f'{_GATING}EllipsoidGate'))
    assert [d.get(f'{_DTYPE}name') for d in ell.iter(f'{_DTYPE}fcs-dimension')] == \
        ['Comp-FL1-A', 'Comp-FL2-A']
    sample = next(ET.parse(out).getroot().iter('Sample'))
    assert [t.find(f'{_DTYPE}parameter').get(f'{_DTYPE}name')
            for t in sample.find('Transformations')] == ['Comp-FL1-A', 'Comp-FL2-A']
    again = fp.WspReader(str(out)).extract_gates()
    blob, child = _blob_and_child(again)
    orig_blob, _ = _blob_and_child(gates)
    assert blob['kind'] == 'flowjo_ellipse'
    assert (blob['flowjo']['foci'], blob['flowjo']['edge'], blob['flowjo']['axes']) == \
        (orig_blob['flowjo']['foci'], orig_blob['flowjo']['edge'], orig_blob['flowjo']['axes'])
    assert child['parent_id'] == blob['_import_id']


# ── compare: the ellipse is the parent, not skipped ─────────────────────────

def _compare_counts(tmp_path, kind, centre, a, b, ang, to_data, child_hi, to_disp=None):
    """Compare-tool rows for Blob/Child of a FlowJo-drawn ellipse on `kind`
    axes, on events drawn uniformly over the display and mapped to data with
    `to_data`; also the display-space truth (from the float32 values the FCS
    holds when `to_disp` maps them back exactly)."""
    import flowio

    from openflo.compare import WspReader, _compare_one_sample, _per_sample_inventory
    rng = np.random.default_rng(23)
    n = 20_000
    disp = rng.uniform(0, 256, size=(n, 2))
    data = to_data(disp)
    ev = np.column_stack([rng.uniform(1, 1e5, n), data]).astype(np.float32)
    with open(tmp_path / 'cmp.fcs', 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), ['FSC-A', 'FL1-A', 'FL2-A'])
    path = _write_native(tmp_path, _display_ellipse(centre, a, b, ang), kind, child_hi)
    (sample, _uri, pops), = _per_sample_inventory(WspReader(path))
    rows = {r['population']: r
            for r in _compare_one_sample(sample, str(tmp_path / 'cmp.fcs'), pops)}
    held = ev[:, 1:].astype(float)
    in_ell = _in_display_ellipse(disp if to_disp is None else to_disp(held),
                                 centre, a, b, ang)
    in_child = in_ell & (held[:, 0] < child_hi) & (held[:, 1] < child_hi)
    return rows, int(in_ell.sum()), int(in_child.sum())


def test_compare_counts_a_linear_flowjo_ellipse_as_the_childs_parent(tmp_path):
    rows, n_ell, n_child = _compare_counts(
        tmp_path, 'linear', *_ELL,
        to_data=lambda d: 3.0 + d / 256.0 * (60000.0 - 3.0), child_hi=35000.0,
        to_disp=lambda x: (x - 3.0) / (60000.0 - 3.0) * 256.0)
    assert n_child > 200 and n_ell > n_child
    assert rows['Blob']['openflo_count'] == n_ell
    # Unparsed, the ellipse made Child a root: every event below 35000 on both.
    assert rows['Child']['openflo_count'] == n_child
    assert not rows['Child']['error']


def test_compare_counts_a_logicle_flowjo_ellipse_as_the_childs_parent(tmp_path):
    centre, a, b, ang = (80.0, 70.0), 33.0, 20.0, -35.0

    def to_data(d):
        return fp.inverse_transform_values(np.ravel(d) / 256, 'logicle',
                                           **_LOGICLE).reshape(np.shape(d))
    x_mid = float(to_data(np.array([[centre[0], 0.0]]))[0, 0])
    rows, n_ell, n_child = _compare_counts(tmp_path, 'logicle', centre, a, b, ang,
                                           to_data, child_hi=x_mid)
    assert n_child > 200 and n_ell > n_child
    # A 64-gon mapped vertex by vertex: within a hair of the display truth.
    assert abs(rows['Blob']['openflo_count'] - n_ell) <= 0.005 * n_ell
    assert abs(rows['Child']['openflo_count'] - n_child) <= 0.01 * n_child + 2
    assert rows['Child']['openflo_count'] <= rows['Blob']['openflo_count']


def test_compare_keeps_an_unconverted_ellipse_as_an_empty_flagged_parent(tmp_path):
    rows, _n_ell, _n_child = _compare_counts(
        tmp_path, 'log', *_ELL, to_data=lambda d: 1.0 + d, child_hi=5.0)
    for pop in ('Blob', 'Child'):
        assert rows[pop]['openflo_count'] == 0
        assert 'FlowJo ellipse not converted' in rows[pop]['error']
        assert 'log axis' in rows[pop]['error']
    assert rows['Cells']['openflo_count'] == 20_000 and not rows['Cells']['error']


def _quadrant_wsp(tmp_path, second_divider=True):
    div_y = ('<gating:divider gating:id="dy">'
             '<data-type:fcs-dimension data-type:name="FL2-A"/>'
             '<gating:value>5.0</gating:value></gating:divider>') if second_divider else ''
    p = tmp_path / 'quad.wsp'
    p.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<Workspace '
        'xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" '
        'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">'
        '<SampleList><Sample><SampleNode name="S" sampleID="1"><Subpopulations>'
        '<Population name="Quadrants"><Gate gating:id="q"><gating:QuadrantGate>'
        '<gating:divider gating:id="dx">'
        '<data-type:fcs-dimension data-type:name="FL1-A"/>'
        '<gating:value>5.0</gating:value></gating:divider>' + div_y
        + '</gating:QuadrantGate></Gate></Population>'
        '</Subpopulations></SampleNode></Sample></SampleList></Workspace>',
        encoding='utf-8')
    return str(p)


def test_quadrant_is_reported_as_skipped_only_when_not_imported(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=fp.log.name):
        gates = fp.WspReader(_quadrant_wsp(tmp_path)).extract_gates()
    assert len([g for g in gates if g.get('quad_set')]) == 4
    assert not any('QuadrantGate' in r.getMessage() for r in caplog.records)
    caplog.clear()
    # One divider is not a quadrant: nothing is imported, and both paths say so.
    r = fp.WspReader(_quadrant_wsp(tmp_path, second_divider=False))
    with caplog.at_level(logging.WARNING, logger=fp.log.name):
        assert r.extract_gates() == []
        assert r.extract_gates(sample_node=next(r.root.iter('SampleNode'))) == []
    assert sum('QuadrantGate' in rec.getMessage() for rec in caplog.records) == 2
