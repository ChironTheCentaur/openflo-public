"""The two per-series steps of PlotMixin._plot_histogram: a series' counts
and hidden-event count from the plot cache or worked out and cached
(_hist_series_counts), and its smoothed curve on the y mode's scale
(_hist_curve).

A stand-in editor and cache record what is fetched and stored; values
are generated. No Tk.
"""
import numpy as np
import pytest

from openflo.editor_plot import PlotMixin


class _Store:
    def __init__(self, **have):
        self.d = dict(have)
        self.puts = []

    def get(self, key, default=None):
        return self.d.get(key, default)

    def put(self, key, value, nbytes=None):
        self.puts.append((key, nbytes))
        self.d[key] = value


class _Ed(PlotMixin):
    def __init__(self, values):
        self.values = values
        self.fetched = 0

    def _plotcache_hist_key(self, name, hs, kind, extra=None):
        return kind

    def _plotcache_hist_values(self, name, hs):
        self.fetched += 1
        return self.values


EDGES = np.array([-1.0, 0.0, 1.0, 2.0])
VALUES = np.array([-0.5, 0.5, 0.5, 1.5, -3.0])


def _counts(store, arr=None, scale='log'):
    ed = _Ed(VALUES)
    out = ed._hist_series_counts(store, 's1', {}, arr, EDGES, 'e', scale)
    return ed, out


def test_both_cached_fetches_and_stores_nothing():
    store = _Store(hcnt=[9, 9, 9], hhid=4)
    ed, (counts, hid) = _counts(store)
    assert (counts, hid) == ([9, 9, 9], 4)
    assert ed.fetched == 0 and store.puts == []


def test_nothing_cached_counts_and_hides_then_stores_both():
    store = _Store()
    ed, (counts, hid) = _counts(store)
    assert list(counts) == [1, 2, 1]          # -3.0 falls outside the edges
    assert hid == 2                           # two non-positive on a log axis
    assert ed.fetched == 1
    assert store.puts == [('hcnt', None), ('hhid', 64)]


def test_a_given_array_is_used_without_fetching():
    store = _Store()
    ed, (counts, hid) = _counts(store, arr=np.array([0.5, 1.5]))
    assert ed.fetched == 0
    assert list(counts) == [0, 1, 1] and hid == 0


def test_only_the_missing_one_is_worked_out():
    store = _Store(hcnt=[7, 7, 7])
    ed, (counts, hid) = _counts(store)
    assert counts == [7, 7, 7] and hid == 2
    assert store.puts == [('hhid', 64)] and ed.fetched == 1
    store = _Store(hhid=5)
    ed, (counts, hid) = _counts(store)
    assert list(counts) == [1, 2, 1] and hid == 5
    assert store.puts == [('hcnt', None)]


def test_linear_scale_hides_nothing():
    _ed, (_c, hid) = _counts(_Store(), scale='linear')
    assert hid == 0


class _Smooth:
    """Stands in for gaussian_filter1d: records its sigma and mode."""

    def __init__(self):
        self.calls = []

    def __call__(self, counts, sigma, mode):
        self.calls.append((list(counts), sigma, mode))
        return np.asarray(counts) * 2.0


@pytest.mark.parametrize('ymode, want', [
    ('Count', [2.0, 8.0, 4.0]),
    ('% of Max', [25.0, 100.0, 50.0]),
    ('Fraction', [0.2, 0.8, 0.4]),
    ('anything else', [0.2, 0.8, 0.4]),
])
def test_curve_on_each_y_scale(ymode, want):
    sm = _Smooth()
    y = PlotMixin._hist_curve([1, 4, 2], 10, 256, ymode, sm)
    assert list(y) == pytest.approx(want)
    assert sm.calls[0][0] == [1.0, 4.0, 2.0] and sm.calls[0][2] == 'constant'


def test_flat_zero_and_empty_series_are_left_as_they_are():
    assert list(PlotMixin._hist_curve([0, 0], 5, 256, '% of Max',
                                      _Smooth())) == [0.0, 0.0]
    assert list(PlotMixin._hist_curve([], 5, 256, '% of Max',
                                      _Smooth())) == []
    assert list(PlotMixin._hist_curve([3], 0, 256, 'Fraction',
                                      _Smooth())) == [6.0]


@pytest.mark.parametrize('size, sigma', [
    (256 * 100, 1.0),                    # dense: the narrowest kernel
    (256, 1.5),                          # one event per bin
    (64, 3.0),                           # 1/4 per bin
    (8, 4.0),                            # sparse: the widest kernel
    (0, 4.0),
])
def test_sparser_series_get_a_wider_kernel(size, sigma):
    sm = _Smooth()
    PlotMixin._hist_curve([1, 2], size, 256, 'Count', sm)
    assert sm.calls[0][1] == pytest.approx(sigma)


def test_a_peak_below_one_is_still_scaled_to_100():
    """Counts smoothed to a peak under 1 (a sparse series) still reach 100."""
    def halve(counts, sigma, mode):
        return np.asarray(counts) / 4.0
    y = PlotMixin._hist_curve([1, 2], 10, 256, '% of Max', halve)
    assert list(y) == pytest.approx([50.0, 100.0])
