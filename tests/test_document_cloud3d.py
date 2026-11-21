"""Document.cloud3d (the phone's 3D view) and the SciPy-free cube density
it shares with the desktop's 3D plot (openflo.cloud3d)."""
from __future__ import annotations

import numpy as np
import pytest

from openflo import cloud3d
from openflo.document import DocumentError
from openflo.document_rpc import DocumentRpc
from tests.test_document import CELLS, gated_doc


def test_the_numpy_smoothing_is_scipys_gaussian_filter():
    from scipy.ndimage import gaussian_filter
    grid = np.random.default_rng(3).poisson(2.0, (17, 11, 13)).astype(float)
    for sigma in (1.0, 1.6):
        want = gaussian_filter(grid, sigma=sigma, mode='constant')
        np.testing.assert_allclose(cloud3d.smooth3d(grid, sigma), want,
                                   rtol=1e-9, atol=1e-12)


def test_the_kernel_reaches_four_sigmas_and_sums_to_one():
    w = cloud3d.gaussian_kernel(1.0)
    assert len(w) == 9 and w.sum() == pytest.approx(1.0)
    assert w[4] == w.max() and w[0] == pytest.approx(w[8])


def test_the_cube_density_is_the_scipy_one_read_per_point():
    from scipy.ndimage import gaussian_filter
    p = np.clip(np.random.default_rng(4).normal(0, 0.4, (5000, 3)), -1, 1)
    b = 16
    idx = np.clip(np.floor((p + 1) / 2 * b), 0, b - 1).astype(int)
    hist = np.zeros((b, b, b))
    np.add.at(hist, tuple(idx.T), 1)
    want = gaussian_filter(hist, sigma=1.0, mode='constant')[tuple(idx.T)]
    np.testing.assert_allclose(cloud3d.density_values(p, b), want, rtol=1e-9)


def test_the_default_grid_is_64_bins_a_side():
    """The phone's cube density is the desktop's: 64 bins a side."""
    from scipy.ndimage import gaussian_filter
    p = np.clip(np.random.default_rng(6).normal(0, 0.3, (4000, 3)), -1, 1)
    idx = np.clip(np.floor((p + 1) / 2 * 64), 0, 63).astype(int)
    hist = np.zeros((64, 64, 64))
    np.add.at(hist, tuple(idx.T), 1)
    want = gaussian_filter(hist, sigma=1.0, mode='constant')[tuple(idx.T)]
    np.testing.assert_allclose(cloud3d.density_values(p), want, rtol=1e-9)


def test_the_shade_is_the_2d_plots_gamma():
    piles = np.r_[np.full((2000, 3), -0.5), np.full((1000, 3), 0.5)]
    c = cloud3d.density_shade(cloud3d.density_values(piles))
    assert c[0] == pytest.approx(1.0)
    assert c[-1] == pytest.approx(0.5 ** 0.4)
    assert cloud3d.density_shade(np.zeros(3)).tolist() == [0, 0, 0]
    # Below zero is no density: shaded as none, never NaN.
    assert cloud3d.density_shade(np.array([-1.0, 0.0, 4.0])).tolist() == [0, 0, 1]


def _front(doc, name, **kw):
    """The cloud's X and Y next to the 2D plot's dots, both on -1..1 over
    the 2D plot's own ranges, each sorted the same way."""
    d3 = doc.cloud3d(name, 'SSC-A', 'FL1-A', 'FL2-A', x_scale='logicle',
                     y_scale='linear', **kw)
    d2 = doc.density_events(name, 'SSC-A', 'FL1-A', x_scale='logicle',
                            y_scale='linear', max_events=10 ** 6,
                            gate=kw.get('gate'))
    flat = np.column_stack([
        cloud3d.normalize(d2['x'].astype(float), d2['x_range']),
        cloud3d.normalize(d2['y'].astype(float), d2['y_range'])])
    got = d3['points'][:, :2].astype(float)
    order = lambda a: a[np.lexsort((a[:, 1], a[:, 0]))]  # noqa: E731
    return d3, d2, order(got), order(flat)


def test_the_front_view_is_the_2d_plot(tmp_path):
    """SSC-A is stored linear and shown on logicle, FL1-A stored logicle
    and shown linear: both must sit where the 2D plot puts them."""
    doc, name, cells, *_ = gated_doc(tmp_path)
    for gate in (None, cells):
        d3, d2, got, flat = _front(doc, name, gate=gate)
        assert d3['x_range'] == d2['x_range'] and d3['y_range'] == d2['y_range']
        assert d3['shown'] == d2['shown'] == d3['events'] == d2['events']
        np.testing.assert_allclose(got, flat, atol=2e-6)


def test_points_are_the_channels_on_their_scales(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    d = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL3-A', z_scale='log')
    df = doc._samples[name].data.data
    zv = doc.to_view(name, 'FL3-A', df['FL3-A'].to_numpy(), 'log')
    want = np.sort(cloud3d.normalize(zv, d['z_range']))
    np.testing.assert_allclose(np.sort(d['points'][:, 2]), want, atol=2e-6)
    assert d['z_range'] == [float(zv.min()), float(zv.max())]
    assert d['points'].dtype == np.float32 and d['points'].shape == (3000, 3)
    assert np.abs(d['points']).max() <= 1.0


def test_a_population_is_its_events(tmp_path):
    doc, name, cells, *_ = gated_doc(tmp_path)
    d = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A', gate=cells)
    count = {r['id']: r['count'] for r in doc.stats(name, channels=[])}[cells]
    assert d['events'] == d['shown'] == count < 3000
    lo, hi = d['x_range']
    fsc = (d['points'][:, 0] + 1) / 2 * (hi - lo) + lo
    assert fsc.min() >= CELLS['x0'] - 1 and fsc.max() <= CELLS['x1'] + 1


def test_a_large_population_is_a_seeded_subset(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    a = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A', max_events=500, seed=7)
    b = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A', max_events=500, seed=7)
    c = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A', max_events=500, seed=8)
    assert (a['shown'], a['events'], a['seed']) == (500, 3000, 7)
    np.testing.assert_array_equal(a['points'], b['points'])
    assert not np.array_equal(a['points'], c['points'])
    whole = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A')
    pick = cloud3d.seeded_subset(3000, 500, 7)
    np.testing.assert_array_equal(a['points'], whole['points'][pick])
    assert a['x_range'] == whole['x_range']


def test_a_gated_subset_is_picked_from_the_population(tmp_path):
    doc, name, cells, *_ = gated_doc(tmp_path)
    whole = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A', gate=cells)
    n = whole['events']
    a = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A', gate=cells,
                    max_events=n // 2, seed=3)
    assert a['shown'] == n // 2
    np.testing.assert_array_equal(
        a['points'], whole['points'][cloud3d.seeded_subset(n, n // 2, 3)])


def test_events_with_no_value_are_left_out(tmp_path):
    """An event with no value (NaN) on an axis has no place: left out,
    neither drawn nor counted as off the axes."""
    doc, name, *_ = gated_doc(tmp_path)
    df = doc._samples[name].data.data
    gone = np.zeros(len(df), dtype=bool)
    gone[::7] = True
    df.loc[gone, 'SSC-A'] = np.nan
    d = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A')
    assert d['shown'] == int((~gone).sum()) and d['off_axis'] == 0
    assert d['events'] == len(df)
    assert np.isfinite(d['points']).all()


def test_a_given_range_leaves_the_rest_out_and_counts_it(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    df = doc._samples[name].data.data
    mid = float(np.median(df['FSC-A']))
    d = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A',
                    x_range=[float(df['FSC-A'].min()), mid])
    assert d['off_axis'] == int((df['FSC-A'] > mid).sum()) > 0
    assert d['shown'] + d['off_axis'] == 3000
    assert d['points'][:, 0].max() <= 1.0 and d['x_range'][1] == mid


def test_dots_are_colored_by_how_crowded_they_are(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    d = doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A')
    want = cloud3d.density_shade(cloud3d.density_values(
        d['points'].astype(float)))
    np.testing.assert_allclose(d['c'], want, rtol=1e-5)
    assert d['c'].max() == pytest.approx(1.0) and d['c'].dtype == np.float32
    centre = np.argmin(np.abs(d['points']).sum(axis=1))
    edge = np.argmax(np.abs(d['points']).sum(axis=1))
    assert d['c'][centre] > d['c'][edge]


def test_cloud3d_refusals(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    with pytest.raises(DocumentError, match='no channel'):
        doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL9-A')
    with pytest.raises(DocumentError, match='max_events'):
        doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A', max_events=0)
    with pytest.raises(DocumentError, match='seed'):
        doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A', seed=-1)
    assert doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A', max_events=1)['shown'] == 1
    assert doc.cloud3d(name, 'FSC-A', 'SSC-A', 'FL1-A', seed=0)['seed'] == 0


def test_cloud3d_through_rpc(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    rpc = DocumentRpc()
    rpc.doc = doc
    r = rpc.handle({'id': 3, 'method': 'cloud3d',
                    'args': {'sample': name, 'x': 'FSC-A', 'y': 'SSC-A',
                             'z': 'FL1-A', 'max_events': 100}})
    assert r['ok'] and r['result']['shown'] == 100
    pts = r['result']['points']
    assert pts['__ndarray__'] == 'float32' and pts['shape'] == [100, 3]
