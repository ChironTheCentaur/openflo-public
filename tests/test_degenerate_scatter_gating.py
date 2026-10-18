"""Degenerate scatter must not silently delete the sample.

`filter_doublets` computed the acceptance window from the median FSC-A/FSC-H
ratio, substituting `0.0` when no event had a usable ratio. That makes the
window `[0, 0]`, which matches nothing — so a sample whose FSC-H is
non-positive throughout was reduced to ZERO events, silently. The whole sample
disappeared with no error and no warning.

The cell-cycle singlet gate had the identical fallback, where the effect is
every cell scored 'NA' and an empty cell-cycle result.

Doublets cannot be identified without the ratio, so the honest action is to
filter nothing and say so — which is what the missing-channel branch directly
above `filter_doublets` already did.
"""
import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample


def _singlets_and_doublets(n_singlet=450, n_doublet=50, seed=0):
    rng = np.random.default_rng(seed)
    a1 = rng.normal(60_000, 8_000, n_singlet)
    a2 = rng.normal(118_000, 12_000, n_doublet)
    return pd.DataFrame({
        'FSC-A': np.concatenate([a1, a2]),
        'FSC-H': np.concatenate([a1 / 2.0 * rng.normal(1, 0.02, n_singlet),
                                 a2 / 3.1 * rng.normal(1, 0.04, n_doublet)]),
        'CD3': rng.random(n_singlet + n_doublet)})


@pytest.mark.parametrize('bad_h', [
    pytest.param(0.0, id='all-zero-FSC-H'),
    pytest.param(-1.0, id='all-negative-FSC-H'),
])
def test_a_degenerate_ratio_keeps_every_event(bad_h):
    df = _singlets_and_doublets()
    df['FSC-H'] = bad_h
    n = len(df)
    s = FlowSample.from_dataframe(df, name='broken')
    s.filter_doublets()
    assert len(s.data) == n, (
        f'{len(s.data)} of {n} events survived — a sample with unusable '
        'scatter was silently deleted rather than left unfiltered')


def test_a_normal_sample_still_loses_its_doublets():
    """The guard must not turn the filter into a no-op for real data."""
    df = _singlets_and_doublets()
    s = FlowSample.from_dataframe(df, name='good')
    s.filter_doublets()
    assert 400 <= len(s.data) < 500, (
        f'expected the ~50 doublets to be removed, got {len(s.data)}')


def test_the_skip_is_reported(caplog):
    df = _singlets_and_doublets()
    df['FSC-H'] = 0.0
    s = FlowSample.from_dataframe(df, name='broken')
    with caplog.at_level('WARNING'):
        s.filter_doublets()
    assert any('doublet filter skipped' in r.message.lower()
               or 'doublet filter skipped' in r.getMessage().lower()
               for r in caplog.records), (
        'the sample was left unfiltered without telling anyone')


def test_a_partially_usable_ratio_still_filters():
    """Only a TOTAL absence of usable ratios triggers the skip."""
    df = _singlets_and_doublets()
    df.loc[:99, 'FSC-H'] = 0.0            # 100 unusable, the rest fine
    s = FlowSample.from_dataframe(df, name='mixed')
    s.filter_doublets()
    assert 0 < len(s.data) < len(df), (
        'a sample with SOME usable events should still be filtered')


def test_cell_cycle_scores_events_when_the_width_ratio_is_unusable():
    """The same fallback in the cell-cycle singlet gate emptied the result."""
    rng = np.random.default_rng(1)
    n = 600
    dna = np.concatenate([rng.normal(50_000, 3_000, 400),
                          rng.normal(100_000, 5_000, 200)])
    df = pd.DataFrame({'FSC-A': rng.normal(60_000, 8_000, n),
                       'FSC-H': rng.normal(30_000, 4_000, n),
                       'DAPI-A': dna,
                       'DAPI-W': np.zeros(n)})     # unusable width
    s = FlowSample.from_dataframe(df, name='cc')
    s.cell_cycle(dna_channel='DAPI-A', singlet_channel='DAPI-W')
    phases = s.data['cell_cycle']
    assert (phases != 'NA').any(), (
        'every cell was scored NA — the unusable width ratio emptied the '
        'cell-cycle result instead of being skipped')
