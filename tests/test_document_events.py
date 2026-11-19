"""Document.density_events and the SciPy-free density behind it (the phone
app's pseudocolor scatter)."""
from __future__ import annotations

import numpy as np
import pytest

from openflo import density
from openflo.document import DocumentError
from tests.test_document import CELLS, gated_doc, spill_fcs


def _cloud(seed=1):
    rng = np.random.default_rng(seed)
    xs = np.r_[rng.normal(0, 1, 20000), rng.normal(4, 0.5, 5000)]
    ys = np.r_[rng.normal(0, 1, 20000), rng.normal(3, 0.7, 5000)]
    return xs, ys, np.linspace(-4, 6, 257), np.linspace(-4, 6, 257)


def test_the_numpy_grid_is_scipys_gaussian_filter():
    from scipy.ndimage import gaussian_filter
    xs, ys, xe, ye = _cloud()
    grid, gx, _gy = density.smoothed_density_grid(xs, ys, xe, ye)
    hist = np.histogram2d(xs, ys, bins=[xe, ye])[0]
    n = len(xs) / (256 * 256)
    sigma = float(np.clip(np.sqrt(1.0 / n) * 1.2, 1.8, 6.0))
    pad = int(np.ceil(3.0 * sigma)) + 1
    want = gaussian_filter(np.pad(hist, pad), sigma=sigma)
    np.testing.assert_allclose(grid, want, rtol=1e-9, atol=1e-12)
    assert len(gx) == len(xe) + 2 * pad and gx[pad] == xe[0]


def test_the_numpy_density_is_close_to_the_desktops():
    xs, ys, xe, ye = _cloud()
    z = density.event_density(xs, ys, xe, ye)
    zn = density.event_density_numpy(xs, ys, xe, ye)
    assert np.max(np.abs(z - zn)) / z.max() < 0.02
    assert (zn >= 0).all() and zn.shape == xs.shape


def test_events_are_the_populations_dots_densest_last(tmp_path):
    doc, name, cells, *_ = gated_doc(tmp_path)
    d = doc.density_events(name, 'FSC-A', 'SSC-A', gate=cells)
    stats = {r['id']: r['count'] for r in doc.stats(name, channels=[])}
    assert d['events'] == stats[cells] == d['shown'] == len(d['x'])
    assert d['x'].dtype == np.float32 and d['c'].dtype == np.float32
    assert np.all(np.diff(d['c']) >= 0) and d['c'][-1] == pytest.approx(1.0)
    assert d['c'].min() >= 0
    # The dots are the population's events, on the display axes.
    df = doc._samples[name].data.data
    m = doc._population(doc._samples[name], cells)[0]
    np.testing.assert_array_equal(np.sort(d['x']),
                                  np.sort(df['FSC-A'][m].to_numpy(np.float32)))
    inside = ((d['x'] >= CELLS['x0']) & (d['x'] < CELLS['x1'])
              & (d['y'] >= CELLS['y0']) & (d['y'] < CELLS['y1']))
    assert inside.all()


def test_a_display_scale_moves_the_dots(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    d = doc.density_events(name, 'FL1-A', 'FL2-A', x_scale='linear')
    want = doc.to_view(name, 'FL1-A', doc._samples[name].data.data['FL1-A']
                       .to_numpy(), 'linear')
    np.testing.assert_allclose(np.sort(d['x']),
                               np.sort(want[np.isfinite(want)]).astype(
                                   np.float32), rtol=1e-6)
    assert d['x_range'] == doc.density(name, 'FL1-A', 'FL2-A',
                                       x_scale='linear')['x_range']


def test_a_large_population_is_a_fixed_subsample(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    a = doc.density_events(name, 'FSC-A', 'SSC-A', max_events=500)
    b = doc.density_events(name, 'FSC-A', 'SSC-A', max_events=500)
    assert a['shown'] == 500 and a['events'] == 3000
    np.testing.assert_array_equal(a['x'], b['x'])
    np.testing.assert_array_equal(a['c'], b['c'])


def test_every_event_is_drawn_past_the_default_cap(tmp_path):
    # The phone's "All events" setting asks for Number.MAX_SAFE_INTEGER dots.
    from openflo.document import Document
    doc = Document()
    name = doc.open_fcs(spill_fcs(tmp_path / 'big.fcs', n=60_000))
    every = doc.density_events(name, 'FSC-A', 'SSC-A', max_events=2**53 - 1)
    assert every['events'] == every['shown'] == every['x'].size == 60_000
    assert doc.density_events(name, 'FSC-A', 'SSC-A')['shown'] == 50_000


def test_density_events_refusals(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    with pytest.raises(DocumentError, match='no channel'):
        doc.density_events(name, 'FSC-A', 'FL9-A')
    with pytest.raises(DocumentError, match='max_events'):
        doc.density_events(name, 'FSC-A', 'SSC-A', max_events=0)
    with pytest.raises(DocumentError, match='no gate'):
        doc.density_events(name, 'FSC-A', 'SSC-A', gate='nope')


def test_density_events_through_rpc(tmp_path):
    from openflo import document_rpc as rpc
    from tests.test_document import spill_fcs
    r = rpc.DocumentRpc()
    s = r.call('open_fcs', {'path': spill_fcs(tmp_path / 'a' / 's.fcs')})
    d = r.call('density_events', {'sample': s, 'x': 'FSC-A', 'y': 'SSC-A',
                                  'max_events': 100})
    assert d['x']['__ndarray__'] == 'float32' and d['shown'] == 100


def test_a_highlighted_population_is_drawn_wherever_its_events_fall(tmp_path):
    doc, name, _cells, pos, _mid = gated_doc(tmp_path)
    stats = {r['id']: r['count'] for r in doc.stats(name, channels=[])}
    # FL1+ (under cells) on the all-events FSC/SSC plot: every one of its
    # events, at its place on these axes.
    d = doc.density_events(name, 'FSC-A', 'SSC-A', highlight=pos)
    assert d['h_events'] == stats[pos] == len(d['hx'])
    df = doc._samples[name].data.data
    m = doc._population(doc._samples[name], pos)[0]
    np.testing.assert_array_equal(np.sort(d['hx']),
                                  np.sort(df['FSC-A'][m].to_numpy(np.float32)))
    assert 'hx' not in doc.density_events(name, 'FSC-A', 'SSC-A')
    # Subsampled at the dots' own rate.
    s = doc.density_events(name, 'FSC-A', 'SSC-A', highlight=pos,
                           max_events=300)
    assert len(s['hx']) == round(stats[pos] * 300 / 3000)
    with pytest.raises(DocumentError, match='no gate'):
        doc.density_events(name, 'FSC-A', 'SSC-A', highlight='nope')


def test_a_histogram_counts_a_highlighted_population_in_its_bins(tmp_path):
    doc, name, _cells, _pos, mid = gated_doc(tmp_path)
    h = doc.histogram(name, 'FL1-A', bins=32, highlight=mid)
    alone = doc.histogram(name, 'FL1-A', gate=mid, bins=32)
    np.testing.assert_array_equal(h['h_counts'], alone['counts'])
    assert h['h_events'] == alone['events']
    assert 'h_counts' not in doc.histogram(name, 'FL1-A', bins=32)


def test_dot_colours_are_the_desktops_density_with_gamma_04(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    d = doc.density_events(name, 'FSC-A', 'SSC-A')
    df = doc._samples[name].data.data
    xs = df['FSC-A'].to_numpy(float)
    ys = df['SSC-A'].to_numpy(float)
    z = density.event_density_numpy(xs, ys, np.linspace(*d['x_range'], 257),
                                    np.linspace(*d['y_range'], 257))
    np.testing.assert_allclose(d['c'], np.sort((z / z.max()) ** 0.4),
                               rtol=1e-5, atol=1e-6)


def test_a_histograms_highlight_need_not_lie_in_its_population(tmp_path):
    doc, name, _cells, pos, mid = gated_doc(tmp_path)
    h = doc.histogram(name, 'FL1-A', gate=pos, bins=32, highlight=mid)
    m = doc._population(doc._samples[name], mid)[0]
    raw = doc._samples[name].data.data['FL1-A'].to_numpy()
    view = np.asarray(doc.to_view(name, 'FL1-A', raw[m], None))
    want = np.histogram(view[np.isfinite(view)], bins=h['edges'])[0]
    np.testing.assert_array_equal(h['h_counts'], want)
    # Some of FL2 mid lies outside FL1+, and still counts.
    pm = doc._population(doc._samples[name], pos)[0]
    assert (m & ~pm).any() and h['h_counts'].sum() > np.histogram(
        np.asarray(doc.to_view(name, 'FL1-A', raw[m & pm], None)),
        bins=h['edges'])[0].sum()
