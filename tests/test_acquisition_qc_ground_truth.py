"""Acquisition QC (FlowSample.run_qc -> AcquisitionQC.run) against constructed
defects, at the defaults the GUI loader uses on EVERY loaded FCS
(editor_loadpool._load_worker: n_bins=200, threshold=5).

tests/test_qc.py checks each detector in isolation and a clean frame on one
seed (n_bins=50). Here: one constructed defect per sample with its exact
event set known, and the clean-data claim ("A clean acquisition trips none
of them") measured over seeds rather than one.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import AcquisitionQC


def _clean(n=50_000, n_fluor=8, seed=123):
    rng = np.random.default_rng(seed)
    d = {'Time': np.sort(rng.uniform(0, 1000, n)),
         'FSC-A': rng.normal(100_000, 10_000, n),
         'FSC-H': rng.normal(80_000, 8_000, n),
         'SSC-A': rng.normal(50_000, 8_000, n)}
    for i in range(n_fluor):
        d[f'FL{i + 1}-A'] = rng.normal(1000, 200, n)
    return pd.DataFrame(d)


def _qc(df, **kw):
    logging.disable(logging.INFO)
    try:
        qc = AcquisitionQC(df)
        keep = qc.run(**kw)
    finally:
        logging.disable(logging.NOTSET)
    removed = np.ones(len(df), dtype=bool)
    removed[np.asarray(keep)] = False
    return removed, qc.report


def test_drift_segment_is_removed_exactly():
    df = _clean()
    t = df['Time'].to_numpy()
    seg = (t >= 400) & (t < 500)
    fl = [c for c in df.columns if c.startswith('FL')]
    df.loc[seg, fl] *= 1.3                       # +30% in every fluor
    removed, rep = _qc(df)
    np.testing.assert_array_equal(removed, seg)
    assert rep['drift'] == int(seg.sum()) and rep['total'] == int(seg.sum())


def test_events_with_infinite_time_are_kept_and_the_rest_still_judged():
    """pd.cut raises on +-inf, so one infinite Time crashed QC, and the GUI
    load that runs it on every file. Those events now get no time bin and
    are kept, as NaN Time already was; the drift segment is still removed
    exactly."""
    df = _clean()
    t = df['Time'].to_numpy().copy()
    seg = (t >= 400) & (t < 500)
    fl = [c for c in df.columns if c.startswith('FL')]
    df.loc[seg, fl] *= 1.3
    idx = np.random.default_rng(3).choice(len(df), 300, replace=False)
    t[idx[:100]], t[idx[100:200]], t[idx[200:]] = np.inf, -np.inf, np.nan
    df['Time'] = t
    removed, rep = _qc(df)
    timed = np.isfinite(t)
    assert not removed[~timed].any()
    np.testing.assert_array_equal(removed[timed], seg[timed])
    removed, rep = _qc(df.assign(Time=np.inf))      # no usable Time at all
    assert not removed.any()


def test_a_shift_of_seven_standard_errors_in_one_channel_is_caught():
    """The +30% case above is caught at any sane threshold. This one is
    calibrated: one fluor shifted by 7 SEs of a bin's median over 10% of
    the run moves the bins' mean rank by ~8.6 SDs, past the 5-SD cut with
    room, but short of 10. With the cut doubled this fails."""
    for seed in range(10):
        df = _clean(seed=seed)
        t = df['Time'].to_numpy()
        seg = (t >= 400) & (t < 500)
        se = 1.2533 * 200 / np.sqrt(len(df) / 200)
        df.loc[seg, 'FL1-A'] += 7 * se
        removed, rep = _qc(df)
        assert removed[seg].mean() > 0.95, (seed, rep)
        assert removed[~seg].mean() < 0.002, (seed, rep)


@pytest.mark.parametrize('n', [3000, 5000])
def test_a_long_fault_does_not_hide_itself(n):
    """A bin's mean rank can move by at most ~0.5, however large the shift.
    A fault over 40% of a short run inflated the MAD across bins until the
    5-SD band passed that limit: one channel shifted by 1000 SD lost 29% of
    its faulty events at 5000 events (none on one seed) and 6% at 3000. The
    centre and SD now come from the bins within 3 SDs, so the faulty bins
    do not set the band meant to catch them."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, 1000, n))
        df = pd.DataFrame({'Time': t, 'FSC-A': rng.normal(1e5, 1e4, n),
                           'SSC-A': rng.normal(5e4, 8e3, n),
                           'FL0-A': rng.normal(1000, 200, n)})
        seg = (t >= 400) & (t < 800)
        df.loc[seg, 'FL0-A'] += 1000 * 200
        removed, rep = _qc(df)
        assert removed[seg].mean() > 0.9, (seed, rep)
        assert removed[~seg].mean() < 0.01, (seed, rep)


def test_burst_and_partial_clog_are_flow_rate_flagged():
    base = _clean()
    t = base['Time'].to_numpy()
    rng = np.random.default_rng(7)
    burst = pd.concat([base, _clean(2500, 8, 99).assign(
        Time=rng.uniform(300, 310, 2500))]).sort_values('Time')
    burst = burst.reset_index(drop=True)
    removed, rep = _qc(burst)
    in_burst = (burst['Time'] >= 300) & (burst['Time'] < 310)
    assert rep['flow_rate'] > 0
    assert removed[in_burst.to_numpy()].mean() > 0.95
    assert removed[~in_burst.to_numpy()].mean() < 0.02

    keep = ~((t >= 600) & (t < 650)) | (rng.random(len(t)) < 0.1)
    clog = base.loc[keep].reset_index(drop=True)
    removed, rep = _qc(clog)
    in_clog = ((clog['Time'] >= 600) & (clog['Time'] < 650)).to_numpy()
    assert rep['flow_rate'] > 0
    assert removed[in_clog].mean() > 0.9


def test_margin_pileup_removes_exactly_the_piled_events():
    df = _clean()
    idx = np.random.default_rng(5).choice(len(df), 1500, replace=False)
    df.loc[idx, 'FSC-A'] = 262143.0
    removed, rep = _qc(df)
    want = np.zeros(len(df), dtype=bool)
    want[idx] = True
    np.testing.assert_array_equal(removed, want)
    assert rep['margin'] == 1500


def test_a_bin_thinned_by_a_clog_is_not_drift():
    """A partial clog keeps a random 10% of the events, so its bins hold the
    same distribution, only fewer events: their mean rank is noisier, not
    shifted. Judged on the full bins' scale, 194 of the 238 clog events were
    also reported as drift. Each bin is now rescaled to the median bin's
    event count."""
    base = _clean()
    t = base['Time'].to_numpy()
    rng = np.random.default_rng(7)
    rng.uniform(300, 310, 2500)          # the draw the burst case makes first
    keep = ~((t >= 600) & (t < 650)) | (rng.random(len(t)) < 0.1)
    removed, rep = _qc(base.loc[keep].reset_index(drop=True))
    assert rep['flow_rate'] > 0
    assert rep['drift'] == 0, rep


@pytest.mark.parametrize('n,n_fluor', [(10_000, 4), (50_000, 16)])
def test_clean_stationary_acquisition_trips_nothing_at_loader_defaults(
        n, n_fluor):
    """Every (bin, channel) pair is a test, so the drift cut must hold for
    the family. At 5 raw MADs (3.4 SD) clean samples lost events in 32/40
    seeds at 10k x 4 fluor and 40/40 at 50k x 16 fluor (mean 2.0%)."""
    hits = []
    for seed in range(20):
        removed, rep = _qc(_clean(n, n_fluor, seed))
        if removed.any():
            hits.append((seed, rep))
    assert not hits, hits


def _lognormal(n, n_fluor, seed):
    """Skewed, linear-scale data: gamma scatter, log-normal fluorescence."""
    rng = np.random.default_rng(seed)
    d = {'Time': np.sort(rng.uniform(0, 600, n)),
         'FSC-A': rng.gamma(9, 10_000, n),
         'SSC-A': rng.lognormal(10.5, 0.4, n),
         'FSC-H': rng.gamma(9, 8_000, n)}
    for i in range(n_fluor):
        d[f'X{i}-A'] = rng.lognormal(6 + 0.2 * i, 1.0, n)
    return pd.DataFrame(d)


def test_clean_skewed_acquisitions_trip_nothing():
    """The MAD of 200 bins is itself off by ~8% (1 SD). Where it fell low a
    clean bin passed 5 SD, so the scale is floored at the sampling SD of a
    bin's mean rank; without the floor 3 of these 30 samples lost a bin.
    The raw median removed events from all 30, a mean of 2.2%."""
    hits = []
    for seed in range(1000, 1030):
        removed, rep = _qc(_lognormal(100_000, 10, seed))
        if removed.any():
            hits.append((seed, rep))
    assert not hits, hits


def test_bin_to_bin_gain_jitter_is_spread_not_drift():
    """Each bin's gain jitters by 2% (laser and fluidics noise), more than the
    sampling noise of a bin's mean rank, so the MAD across bins sets the
    scale. 5 of THOSE SDs is the cut: 5 unscaled MADs (3.4 SD) flagged
    bins in 8 of these 10 samples, 4.7 a sample."""
    hits = []
    for seed in range(10):
        df = _clean(50_000, 8, seed)
        rng = np.random.default_rng(1000 + seed)
        b = np.asarray(pd.cut(df['Time'].to_numpy(), bins=200, labels=False))
        gain = rng.normal(1.0, 0.02, (200, 8))
        for i in range(8):
            df[f'FL{i + 1}-A'] *= gain[b, i]
        removed, rep = _qc(df)
        if removed.any():
            hits.append((seed, rep))
    assert not hits, hits


def test_sparse_tied_channels_are_not_drift():
    """Mass-cytometry shaped: 40 channels, each exactly 0 but for 2% of
    events. A bin's mean rank then counts its positives, ~Poisson(5), which
    is skewed: 5 SDs above is a ~4e-5 tail, not 3e-7, and over 200 bins x
    40 channels 5 of these 20 clean samples lost a bin. The band is widened
    on the long side by the Cornish-Fisher skew term."""
    hits = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = 50_000
        d = {'Time': np.sort(rng.uniform(0, 600, n))}
        for i in range(40):
            x = np.zeros(n)
            on = rng.random(n) < 0.02
            x[on] = rng.lognormal(4, 1, int(on.sum()))
            d[f'M{i}'] = x
        removed, rep = _qc(pd.DataFrame(d))
        if removed.any():
            hits.append((seed, rep))
    assert not hits, hits


def _tied(rng, n, n_ch, p, both=False):
    """``n_ch`` channels exactly 0 but for a fraction ``p`` of events above,
    and with ``both`` another ``p`` below (a compensated zero-inflated
    channel)."""
    d = {}
    for i in range(n_ch):
        x = np.zeros(n)
        if both:
            u = rng.random(n)
            lo, hi = u < p, u > 1 - p
            x[lo] = -rng.lognormal(2, 1, int(lo.sum()))
        else:
            hi = rng.random(n) < p
        x[hi] = rng.lognormal(4, 1, int(hi.sum()))
        d[f'Z{i}'] = x
    return d


@pytest.mark.parametrize('n,q,tail_rate', [(50_000, 0.001, 1.0),
                                           (100_000, 0.005, 0.05)])
def test_tied_channels_with_rare_events_both_ways_are_not_drift(
        n, q, tail_rate):
    """A fraction q of events below a tied 0 and q above: no skew to widen
    the band for, but a bin's mean rank is a difference of two Poisson
    counts, heavier-tailed than a normal. At q=0.1% (Poisson(0.25) a side)
    10 of the first 20 clean samples lost events, up to 1.05%; at q=0.5%
    with the last 30% of the run at 5% of the rate, 1 of the second 20 lost
    a bin (2 in an unshipped version with every part but the kurtosis
    term). The band is now also widened by the Cornish-Fisher kurtosis
    term."""
    hits = []
    for seed in range(20):
        if tail_rate == 1.0:
            rng = np.random.default_rng(30_000 + seed)
            d = {'Time': np.sort(rng.uniform(0, 600, n))}
            for i in range(8):
                d[f'FL{i}'] = rng.lognormal(6, 1, n)
        else:
            rng = np.random.default_rng(50_000 + seed)
            k = int(n * 0.3 * tail_rate / (0.7 + 0.3 * tail_rate))
            d = {'Time': np.sort(np.r_[rng.uniform(0, 420, n - k),
                                       rng.uniform(420, 600, k)])}
        d.update(_tied(rng, n, 40, q, both=True))
        removed, rep = _qc(pd.DataFrame(d), flow_rate=False, margins=False)
        if removed.any():
            hits.append((seed, rep))
    assert not hits, hits


@pytest.mark.parametrize('profile,p', [('tail', 0.01), ('decay30', 0.02)])
def test_sparse_channels_where_the_rate_falls_are_not_drift(profile, p):
    """A bin's skew and kurtosis grow as its event count falls. Taken from
    the median bin, the sparse bins of sparse tied channels were drift: the
    tube running dry (the last 30% of the time at 5% of the rate) cost 12 of
    20 clean samples a bin, a 30-fold decay of the rate 1 of 20. Both are
    now per bin, and the centre is the kept events' mean rank: on a skewed
    channel the median bin sits below it."""
    hits = []
    for seed in range(20):
        rng = np.random.default_rng(20_000 + seed)
        n = 50_000
        if profile == 'tail':
            k = int(n * 0.3 * 0.05 / (0.7 + 0.3 * 0.05))
            t = np.sort(np.r_[rng.uniform(0, 420, n - k),
                              rng.uniform(420, 600, k)])
        else:
            a = np.log(30.0) / 600
            u = rng.uniform(0, 1, n)
            t = np.sort(-np.log(1 - u * (1 - np.exp(-a * 600))) / a)
        df = pd.DataFrame({'Time': t, **_tied(rng, n, 40, p)})
        removed, rep = _qc(df, flow_rate=False, margins=False)
        if removed.any():
            hits.append((seed, rep))
    assert not hits, hits


@pytest.mark.parametrize('p0,p1,floor', [(0.02, 0.10, 0.65),
                                         (0.10, 0.00, 0.35)])
def test_a_sparse_channel_whose_positive_share_moves_is_drift(p0, p1, floor):
    """The skew and kurtosis terms only WIDEN the band, so they cost
    sensitivity; this pins how much. One tied channel's positive share moves
    from p0 to p1 over 10% of a 50k-event run (10 seeds): 71% of the faulty
    events are removed going up from 2% to 10%, 57% going down from 10% to
    none. Doubling the skew term took the first to 40%, counting all the
    kurtosis rather than what the skew does not explain took it to 60%, and
    widening both sides of the band, not just the long one, took the second
    to 3%."""
    caught = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = 50_000
        t = np.sort(rng.uniform(0, 1000, n))
        df = pd.DataFrame({'Time': t, 'FSC-A': rng.normal(1e5, 1e4, n),
                           'SSC-A': rng.normal(5e4, 8e3, n)})
        for i in range(5):
            df[f'FL{i}-A'] = rng.normal(1000, 200, n)
        seg = (t >= 400) & (t < 500)
        x = np.zeros(n)
        on = rng.random(n) < np.where(seg, p1, p0)
        x[on] = rng.lognormal(4, 1, int(on.sum()))
        df['FL0-A'] = x
        removed, rep = _qc(df, flow_rate=False, margins=False)
        assert removed[~seg].mean() < 0.002, (seed, rep)
        caught.append(removed[seg].mean())
    assert np.mean(caught) > floor, caught


@pytest.mark.parametrize('kind,q,q1,length,floor', [
    ('two', 0.01, 0.08, 100, 0.60), ('int', 0.005, 0.05, 20, 0.45)])
def test_a_two_sided_tied_channel_whose_rare_events_surge_is_drift(
        kind, q, q1, length, floor):
    """The kurtosis term widens both sides of the band, so it costs
    sensitivity exactly where it was needed; this pins how much. A channel
    tied at 0 with q of events below and q above ('two'), or an unused
    integer detector at 37 with q of -1 and q of +1 glitches ('int'), has
    the upper ones surge to q1 over part of a 50k run (10 seeds): 66% and
    55% of the faulty events are removed. With the kurtosis term doubled,
    54% and 30%."""
    caught = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = 50_000
        t = np.sort(rng.uniform(0, 1000, n))
        df = pd.DataFrame({'Time': t, 'FSC-A': rng.normal(1e5, 1e4, n),
                           'SSC-A': rng.normal(5e4, 8e3, n)})
        for i in range(5):
            df[f'FL{i}-A'] = rng.normal(1000, 200, n)
        seg = (t >= 400) & (t < 400 + length)
        u = rng.random(n)
        lo = u < q
        hi = u > 1 - np.where(seg, q1, q)
        if kind == 'two':
            x = np.zeros(n)
            x[lo] = -rng.lognormal(2, 1, int(lo.sum()))
            x[hi] = rng.lognormal(4, 1, int(hi.sum()))
        else:
            x = np.full(n, 37.0)
            x[lo] = 36.0
            x[hi] = 38.0
        df['FL0-A'] = x
        removed, rep = _qc(df, flow_rate=False, margins=False)
        assert removed[~seg].mean() < 0.002, (seed, rep)
        caught.append(removed[seg].mean())
    assert np.mean(caught) > floor, caught


def test_clean_data_in_many_short_bins_trips_nothing():
    """At 2000 bins of 25 events a bin's mean rank is platykurtic (uniform
    ranks have excess kurtosis -1.2). The kurtosis term only widens: let it
    narrow the band for negative kurtosis and 1 of these 20 clean samples
    lost a bin."""
    hits = []
    for seed in range(20):
        rng = np.random.default_rng(20_000 + seed)
        n = 50_000
        d = {'Time': np.sort(rng.uniform(0, 600, n)),
             'FSC-A': rng.normal(1e5, 1e4, n),
             'SSC-A': rng.lognormal(10.5, .4, n)}
        for i in range(8):
            d[f'X{i}'] = rng.lognormal(6, 1, n)
        removed, rep = _qc(pd.DataFrame(d), n_bins=2000, flow_rate=False,
                           margins=False)
        if removed.any():
            hits.append((seed, rep))
    assert not hits, hits


def test_a_channel_of_rare_spikes_is_not_drift():
    """A channel that is 0 but for 0.1% of events: most bins hold no spike,
    so the MAD is ~0 and the sampling SD sets the scale. A bin's spike count
    is ~Poisson(0.25), whose upper tail is far heavier than a normal's:
    without the skew term 4 of these 20 clean samples lost a bin, and 4
    without the sampling-SD floor."""
    hits = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = 50_000
        spk = np.zeros(n)
        on = rng.random(n) < 0.001
        spk[on] = rng.uniform(500, 5000, int(on.sum()))
        df = pd.DataFrame({'Time': np.sort(rng.uniform(0, 600, n)),
                           'FSC-A': rng.normal(100_000, 10_000, n),
                           'Spk-A': spk})
        removed, rep = _qc(df, flow_rate=False, margins=False)
        if removed.any():
            hits.append((seed, rep))
    assert not hits, hits


def _bimodal(n=20_000, pos=0.48, seed=0):
    """A stationary sample whose marker is bimodal with its median at the
    edge of the gap between the modes, as CD3 is in total PBMC."""
    rng = np.random.default_rng(seed)
    k = int(n * pos)
    marker = np.r_[rng.normal(5000, 800, k), rng.normal(150, 50, n - k)]
    rng.shuffle(marker)
    return pd.DataFrame({'Time': np.sort(rng.uniform(0, 300, n)),
                         'FSC-A': rng.normal(60_000, 8_000, n),
                         'CD3': marker})


def test_bimodal_marker_on_a_stationary_sample_is_not_drift():
    """A bin's raw median jumps across the empty gap whenever chance puts
    its positive fraction past one half. Compared as raw values those jumps
    were "drift": the shuffled synthetic PBMC sample lost 37.8% of its
    events. Compared as ranks they are small steps."""
    for seed in range(5):
        removed, rep = _qc(_bimodal(seed=seed))
        assert rep['drift'] == 0, (seed, rep)


def test_real_drift_on_a_bimodal_marker_is_still_caught():
    """Ranks do not blind the detector to a bimodal channel: a 1.5x gain
    drift is caught when the median sits inside a mode (here the positive
    one, 70% of events)."""
    df = _bimodal(pos=0.7)
    t = df['Time'].to_numpy()
    seg = (t >= 120) & (t < 150)
    df.loc[seg, 'CD3'] *= 1.5                    # the whole channel shifts
    removed, rep = _qc(df)
    assert removed[seg].mean() > 0.9, rep
    assert removed[~seg].mean() < 0.01, rep
