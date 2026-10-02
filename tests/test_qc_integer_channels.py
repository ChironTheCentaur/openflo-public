"""Acquisition QC must not punish integer-valued channels.

`_mad_outliers` used `mad = median(|v - med|) + 1e-10`. On an integer channel
(`$DATATYPE I`, very common in real FCS files) more than half the per-bin
medians tie with the overall median, so the MAD is exactly 0 — and the epsilon
turned that into a band of width ~1e-10, making any bin differing by a single
quantisation step a "drift outlier".

Measured before the fix: a clean, drift-free integer channel lost **38% of its
events**, reported as an acquisition problem, while the identical data stored
as float lost none.

A zero MAD carries no scale, so one has to come from outside the deviations.
It must not be derived from them: the standard deviation is inflated by the
very outliers being hunted (112 against a real 400-unit excursion — a band of
560 that hid the fault completely), and the median of the non-zero deviations
lets a LONE outlier set its own band (a 350-event burst among uniform 50s made
its band 1750 and escaped). Each caller now supplies a floor from the physics
of its own data: the quantisation step for a channel, sqrt(N) Poisson noise for
an event count.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import AcquisitionQC, FlowSample

N = 40_000
TIME = np.linspace(0.0, 300.0, N)


def _qc(values):
    df = pd.DataFrame({'Time': TIME, 'FSC-A': values, 'CD3': values})
    s = FlowSample.from_dataframe(df, name='qc')
    before = len(s.data)
    logging.disable(logging.CRITICAL)
    try:
        s.run_qc()
    finally:
        logging.disable(logging.NOTSET)
    return before, len(s.data)


@pytest.fixture
def clean():
    return np.random.default_rng(0).normal(1000.0, 6.0, N)


@pytest.fixture
def clogged(clean):
    v = clean.copy()
    v[18_000:21_500] += 400.0          # an excursion, the fault QC exists for
    return v


def test_a_clean_integer_channel_keeps_every_event(clean):
    before, after = _qc(np.round(clean))
    assert after == before, (
        f'{before - after} of {before} events were removed from a drift-free '
        'integer channel — the quantisation was read as an acquisition fault')


def test_a_clean_float_channel_keeps_every_event(clean):
    before, after = _qc(clean)
    assert after == before


def test_integer_and_float_agree_on_clean_data(clean):
    assert _qc(np.round(clean)) == _qc(clean), (
        'storage dtype changed the QC verdict on identical data')


def test_a_real_excursion_is_still_caught_on_an_integer_channel(clogged):
    before, after = _qc(np.round(clogged))
    assert after < before, (
        'the guard made QC blind — a 400-unit clog went undetected')
    assert after < before * 0.99


def test_integer_and_float_agree_on_a_faulty_sample(clogged):
    """The strongest statement of the invariant: rounding the data must not
    change what QC decides, in either direction."""
    assert _qc(np.round(clogged)) == _qc(clogged)


def test_a_perfectly_constant_channel_flags_nothing():
    """No spread at all means no outlier claim — not infinite sensitivity."""
    before, after = _qc(np.full(N, 1000.0))
    assert after == before


def test_one_dead_detector_does_not_delete_the_whole_sample():
    """The margin filter is OR-ed across the panel, and a channel that is
    constant across the file is trivially "all at its max". A single unused or
    disabled detector — routine in real FCS files — therefore removed 100% of
    an otherwise healthy sample. A channel with no range has no ceiling."""
    rng = np.random.default_rng(0)
    n = 20_000
    df = pd.DataFrame({
        'Time': np.linspace(0.0, 300.0, n),
        'FSC-A': rng.normal(60_000, 8_000, n),
        'CD3': rng.normal(1_000, 200, n),
        'Unused-A': np.zeros(n)})            # a dead detector
    s = FlowSample.from_dataframe(df.copy(), name='dead-channel')
    logging.disable(logging.CRITICAL)
    try:
        s.run_qc()
        alive = FlowSample.from_dataframe(df.drop(columns=['Unused-A']),
                                          name='no-dead-channel')
        alive.run_qc()
    finally:
        logging.disable(logging.NOTSET)
    assert len(s.data) > 0, (
        'one unused detector emptied the entire sample')
    assert len(s.data) == len(alive.data), (
        'the presence of a dead channel changed the QC verdict for every '
        'other channel')


def test_real_saturation_is_still_removed():
    """The guard must not disable margin detection for a channel that really
    is piled up at its ceiling."""
    rng = np.random.default_rng(3)
    n = 10_000
    col = rng.normal(50_000, 5_000, n)
    col[:2_000] = 262_143.0                  # pegged at the top of scale
    df = pd.DataFrame({'Time': np.linspace(0.0, 300.0, n), 'FSC-A': col,
                       'CD3': rng.normal(1_000, 200, n)})
    logging.disable(logging.CRITICAL)
    try:
        keep = AcquisitionQC(df).run(drift=False, flow_rate=False)
    finally:
        logging.disable(logging.NOTSET)
    removed = n - len(df.loc[keep])
    assert removed >= 2_000, (
        f'{removed} removed — the saturated pile-up was not caught')


def test_a_lone_count_burst_is_still_flagged():
    """The other half of the invariant, and the case a deviation-derived
    fallback got wrong: bins of exactly 50 with one bin of 400. Every
    deviation but one is 0, so the MAD is 0 — but a 7x surge is real."""
    qc = AcquisitionQC(pd.DataFrame({'x': range(600)}))
    bins = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 400 + [4] * 50)
    assert 3 in qc._flowrate_bad_bins(bins, n_bins=5, threshold=5.0)


def test_uniform_counts_are_clean():
    qc = AcquisitionQC(pd.DataFrame({'x': range(500)}))
    bins = np.repeat(np.arange(10), 50)
    assert qc._flowrate_bad_bins(bins, n_bins=10, threshold=5.0) == set()


def test_count_noise_alone_is_not_a_flow_fault():
    """Poisson scatter around a steady rate must not read as a fault."""
    rng = np.random.default_rng(7)
    counts = rng.poisson(50, 20)
    bins = np.repeat(np.arange(20), counts)
    qc = AcquisitionQC(pd.DataFrame({'x': range(len(bins))}))
    assert qc._flowrate_bad_bins(bins, n_bins=20, threshold=5.0) == set(), (
        'ordinary counting noise was reported as an acquisition fault')


def test_mad_outliers_needs_a_scale_before_it_will_accuse():
    """The unit contract. More than half of these values tie, so the MAD is
    exactly 0 and the deviations carry no scale. With a floor from the data's
    own quantisation the three far values are caught; with no scale at all the
    honest answer is to accuse nobody, not to accuse everybody — which is what
    the `+ 1e-10` epsilon did."""
    vals = np.array([10.0] * 50 + [11.0] * 40 + [900.0] * 3)
    assert float(np.median(np.abs(vals - np.median(vals)))) == 0.0

    step = AcquisitionQC._quantisation_step(vals)
    assert step == 1.0
    mask = AcquisitionQC._mad_outliers(vals, threshold=5, min_scale=step)
    assert mask.sum() == 3, f'expected the 3 far values, flagged {mask.sum()}'
    assert mask[-3:].all()
    assert not mask[:90].any(), 'the 1-step ties were called outliers'

    assert not AcquisitionQC._mad_outliers(vals, threshold=5).any()


def test_mad_outliers_on_a_fully_constant_array():
    vals = np.full(20, 7.0)
    assert not AcquisitionQC._mad_outliers(vals, threshold=5).any()


def test_a_supplied_floor_never_shrinks_a_real_mad():
    """min_scale is a floor, not a replacement — a measurable MAD still wins,
    so passing a floor cannot make the detector MORE sensitive."""
    rng = np.random.default_rng(11)
    vals = np.append(rng.normal(100.0, 10.0, 200), 900.0)
    strict = AcquisitionQC._mad_outliers(vals, threshold=5)
    floored = AcquisitionQC._mad_outliers(vals, threshold=5, min_scale=1e-9)
    assert (strict == floored).all()
    assert strict[-1]


def test_quantisation_step():
    assert AcquisitionQC._quantisation_step(np.array([5.0, 6.0, 9.0])) == 1.0
    assert AcquisitionQC._quantisation_step(np.full(4, 3.0)) == 0.0
    assert AcquisitionQC._quantisation_step(np.array([1.0])) == 0.0
