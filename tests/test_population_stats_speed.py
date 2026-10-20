"""The faster Statistics table (speed audit item 5) must give the same
numbers, in the same columns, as the code it replaced.

``_reference_rows`` is that code, copied verbatim from master before the
change. The new code runs on one thread and split across threads, over
channels stored linear, on logicle, of unknown scale and with a guessed
transform, with non-finite values, float32 columns, an empty population and
one resting on an unknown gate.
"""
import math

import numpy as np
import pandas as pd
import pytest

import openflo.gating as G
from openflo.gating import (
    GUESSED_TAG,
    population_keys,
    population_path,
    population_stats,
)
from openflo.pipeline import (
    UNKNOWN_SPEC,
    cumulative_gate_mask,
    guessed_spec,
    transform_spec,
    transform_values,
)

STAT_CHAN = ('Median', 'Mean', 'GeoMean', 'CV', 'rCV')
ALL = {'Count', '%Parent', '%Total', *STAT_CHAN}


def _reference_rows(sample_name, df, gates, order, channel_labels, channels,
                     want, stat_chan, select=None, transforms=None):
    # population_stats as it was on master before batch C, verbatim.
    import numpy as np

    from openflo.pipeline import (
        cumulative_gate_mask,
        is_guessed_spec,
        is_unknown_spec,
        to_linear,
        unknown_population_reason,
    )
    total = len(df)
    order = order or list(gates)
    counts, masks = {}, {}
    # One cache for this whole collection pass: `gates`/`df` are fixed here, so
    # a child reuses its parent's cumulative mask instead of re-walking (and
    # re-evaluating) the entire ancestor chain. Masks are read-only below —
    # only boolean-indexed and summed.
    mask_cache = {}
    for gid in order:
        if gid not in gates:
            continue
        m = cumulative_gate_mask(gates, gid, df, cache=mask_cache)
        masks[gid] = m
        counts[gid] = int(np.asarray(m).sum())

    emit = select if select is not None else order
    keys = population_keys(gates, channel_labels)
    rows = []
    row_gids = []
    for gid in emit:
        if gid not in gates or gid not in counts:
            continue
        g = gates[gid]
        cnt = counts[gid]
        parent = g.get('parent_id')
        parent_cnt = counts.get(parent, total) if parent else total
        row = {'Sample': sample_name,
               # Displayed and exported: an unnamed gate's bounds as the
               # intensities the gate list shows, not stored coordinates.
               'Population': population_path(gates, gid,
                                             transforms=transforms),
               '__gid__': gid,
               # Identity across samples (population_keys); the path above is
               # for display and changes with a gate's coordinates.
               '__key__': keys.get(gid, str(gid))}
        # Its gates select nothing, but that 0 was never measured: a
        # missing population, as one a sample does not have.
        why = unknown_population_reason(gates, gid, columns=df.columns)
        if why:
            row['__unknown__'] = why
        if 'Count' in want:
            row['Count'] = float('nan') if why else cnt
        if '%Parent' in want:
            # An empty parent has no fraction to give: NaN, not a 0 % that
            # reads as "measured, none of them".
            row['%Parent'] = ((cnt / parent_cnt * 100.0)
                              if parent_cnt and not why else float('nan'))
        if '%Total' in want:
            row['%Total'] = ((cnt / total * 100.0) if total and not why
                             else float('nan'))
        rows.append(row)
        row_gids.append(gid)

    need_chan = want & set(stat_chan)
    if not (need_chan and channels):
        return rows
    # Channel by channel, so only ONE linearised column is alive at a time
    # (holding them all would be 8 MB per channel per million events), and
    # each column is inverted once rather than once per population. Every
    # row still receives its channels in `channels` order, so the column
    # order the callers derive from first-seen keys is unchanged.
    transforms = transforms or {}
    for ch in channels:
        if ch not in df.columns:
            continue
        lbl = channel_labels.get(ch, ch)
        # A column whose display transform is unknown (an old OpenFlo CSV)
        # has no linear values: its statistics are left out (NaN, to_linear)
        # and every row names it, so the window can say why they are blank.
        unknown = is_unknown_spec(transforms.get(ch))
        # A column whose transform was GUESSED (no record, values in the
        # logicle range) gets its own keys, so its numbers never sit under
        # the certain '(linear)' heading: right for a default-logicle old
        # session, ~20x off for a dim channel the old editor had on asinh.
        if is_guessed_spec(transforms.get(ch)):
            lbl = f'{lbl} {GUESSED_TAG}'
        lin = None
        for row, gid in zip(rows, row_gids, strict=True):
            if unknown:
                row.setdefault('__scale_unknown__', []).append(lbl)
            if not counts[gid]:
                med = mean = cv = gmean = rcv = float('nan')
            else:
                if lin is None:
                    lin = to_linear(df[ch].values, transforms.get(ch))
                vals = lin[np.asarray(masks[gid], dtype=bool)]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    med = mean = cv = gmean = rcv = float('nan')
                else:
                    med = float(np.median(vals))
                    mean = float(np.mean(vals))
                    # Sample SD (n - 1); one event has none.
                    sd = (float(np.std(vals, ddof=1)) if vals.size > 1
                          else float('nan'))
                    cv = (sd / mean * 100.0) if mean else float('nan')
                    # Geometric mean, as FlowJo defines it: the mean in the
                    # channel's display (transform) space, back-transformed.
                    # That is what makes it defined for the negative values
                    # compensation produces. The classic exp(mean(log x))
                    # had to drop them, which biased a dim population
                    # upward (measured: median 151, GeoMean 211.7 from 69%
                    # of its events, where FlowJo's reads 83.5). A channel
                    # stored linear has no display space here, so it keeps
                    # the classic form over its positive values.
                    spec = transforms.get(ch)
                    if spec and spec.get('method', 'linear') != 'linear':
                        stored = np.asarray(df[ch].values, dtype=float)[
                            np.asarray(masks[gid], dtype=bool)]
                        stored = stored[np.isfinite(stored)]
                        gmean = (float(to_linear(
                            np.array([np.mean(stored)]), spec)[0])
                            if stored.size else float('nan'))
                    else:
                        pos = vals[vals > 0]
                        gmean = (float(np.exp(np.mean(np.log(pos))))
                                 if pos.size else float('nan'))
                    # Robust CV, FlowJo's definition: 100 * 1/2 (P84.13 -
                    # P15.87) / median. The MAD form used before agrees only
                    # for a symmetric population (log-normal sigma 0.8: 74%
                    # where FlowJo reports 89%), under the same column name.
                    p_lo, p_hi = np.percentile(vals, [15.87, 84.13])
                    rcv = (50.0 * float(p_hi - p_lo) / med
                           if med else float('nan'))
            if 'Median' in want:
                row[f'Median {lbl}'] = med
            if 'Mean' in want:
                row[f'Mean {lbl}'] = mean
            if 'GeoMean' in want:
                row[f'GeoMean {lbl}'] = gmean
            if 'CV' in want:
                row[f'CV {lbl}'] = cv
            if 'rCV' in want:
                row[f'rCV {lbl}'] = rcv
    return rows

def _data(seed=0, n=40_000):
    rng = np.random.default_rng(seed)
    cols = [f'FL{i}-A' for i in range(1, 7)]
    raw = rng.lognormal(6, 1.2, size=(n, 6)) - 150
    df = pd.DataFrame(raw, columns=cols)
    logicle = transform_spec('logicle')
    transforms = {'FL1-A': transform_spec('linear')}
    for ch in ('FL2-A', 'FL3-A'):
        df[ch] = transform_values(df[ch].to_numpy(), **logicle)
        transforms[ch] = logicle
    transforms['FL4-A'] = dict(UNKNOWN_SPEC)
    df['FL5-A'] = transform_values(df['FL5-A'].to_numpy(), **logicle)
    transforms['FL5-A'] = guessed_spec(logicle)
    df['FL6-A'] = df['FL6-A'].astype('float32')
    df.iloc[::53, 1] = np.nan
    df.iloc[::71, 5] = np.inf
    gates, prev = {}, None
    for k in range(8):
        x, y = ('FL1-A', 'FL6-A') if k % 2 else ('FL6-A', 'FL1-A')
        r = 600 + 80 * k
        ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        gates[f'g{k}'] = {
            'id': f'g{k}', 'kind': 'polygon', 'x_channel': x, 'y_channel': y,
            'vertices': np.c_[r * np.cos(ang) + 300,
                              r * np.sin(ang) + 300].tolist(),
            'parent_id': prev if k % 3 else None, 'name': f'Pop{k}',
            'enabled': True}
        prev = f'g{k}'
    gates['empty'] = {'id': 'empty', 'kind': 'threshold', 'channel': 'FL1-A',
                      'op': '>', 'value': 1e12, 'parent_id': None,
                      'name': 'None', 'enabled': True}
    gates['pos'] = {'id': 'pos', 'kind': 'threshold', 'channel': 'FL2-A',
                    'op': '>', 'value': 0.5, 'parent_id': 'g0',
                    'name': 'Pos', 'enabled': True}
    gates['lost'] = {'id': 'lost', 'kind': 'threshold', 'channel': 'FL9-A',
                     'op': '>', 'value': 0.0, 'parent_id': None,
                     'name': 'Lost', 'enabled': True}
    return df, gates, cols + ['FL9-A'], transforms   # FL9-A is not in df


def _same(a, b):
    assert len(a) == len(b)
    for ra, rb in zip(a, b, strict=True):
        assert list(ra) == list(rb)          # same columns, same order
        for k in ra:
            va, vb = ra[k], rb[k]
            if isinstance(va, float) and math.isnan(va):
                assert isinstance(vb, float) and math.isnan(vb), k
            else:
                assert va == vb, (k, va, vb)


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('want', [
    ALL, {'Median'}, {'rCV'}, {'Count', 'GeoMean', 'CV'}, {'Count', '%Parent'}])
def test_statistics_identical_to_the_old_table(monkeypatch, threads, want):
    monkeypatch.setenv('OMP_NUM_THREADS', str(threads))
    monkeypatch.setattr(G, '_PARALLEL_MIN_VALUES', 0)
    df, gates, chans, tr = _data()
    order = list(gates)
    labels = {'FL1-A': 'Marker 1'}
    new = population_stats('S1', df, gates, order, labels, chans, want,
                           STAT_CHAN, transforms=tr)
    old = _reference_rows('S1', df, gates, order, labels, chans, want,
                          STAT_CHAN, transforms=tr)
    _same(new, old)


def test_select_restricts_rows_but_keeps_parent_counts(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '4')
    monkeypatch.setattr(G, '_PARALLEL_MIN_VALUES', 0)
    df, gates, chans, tr = _data(1)
    order = list(gates)
    pick = ['g2', 'pos', 'lost', 'missing']
    new = population_stats('S1', df, gates, order, {}, chans, ALL, STAT_CHAN,
                           select=pick, transforms=tr)
    old = _reference_rows('S1', df, gates, order, {}, chans, ALL, STAT_CHAN,
                          select=pick, transforms=tr)
    _same(new, old)


def test_the_data_reaches_every_branch():
    """The comparison covers what it claims: populations of each kind and
    the special channels produce what the reference says they should."""
    df, gates, chans, tr = _data()
    rows = {r['__gid__']: r for r in _reference_rows(
        'S1', df, gates, list(gates), {}, chans, ALL, STAT_CHAN,
        transforms=tr)}
    assert rows['empty']['Count'] == 0
    assert '__unknown__' in rows['lost']
    assert rows['g3']['Count'] > 100
    assert rows['g3']['__scale_unknown__'] == ['FL4-A']
    assert f'Median FL5-A {GUESSED_TAG}' in rows['g3']
    assert not math.isnan(rows['g3']['GeoMean FL2-A'])


def test_comparison_catches_a_one_event_change():
    """Negative partner: nudging one value inside a population changes the
    table, so ``_same`` is a real check and not vacuously true."""
    df, gates, chans, tr = _data()
    order = list(gates)
    before = population_stats('S1', df, gates, order, {}, chans, ALL,
                              STAT_CHAN, transforms=tr)
    inside = np.flatnonzero(cumulative_gate_mask(gates, 'g0', df))[0]
    df2 = df.copy()
    df2.iloc[inside, 2] += 1e-3
    after = _reference_rows('S1', df2, gates, order, {}, chans, ALL,
                            STAT_CHAN, transforms=tr)
    with pytest.raises(AssertionError):
        _same(before, after)


def test_large_tables_run_on_several_threads(monkeypatch):
    """Above the size threshold the populations are split across threads
    (the point of the change); below it, they stay on one."""
    import concurrent.futures as cf
    made = []
    real = cf.ThreadPoolExecutor

    def spy(*a, **k):
        made.append(k.get('max_workers'))
        return real(*a, **k)

    monkeypatch.setattr(cf, 'ThreadPoolExecutor', spy)
    monkeypatch.setenv('OMP_NUM_THREADS', '4')
    df, gates, chans, tr = _data()
    population_stats('S1', df, gates, list(gates), {}, chans, ALL, STAT_CHAN,
                     transforms=tr)
    assert made == []                       # 40k events: too small to split
    monkeypatch.setattr(G, '_PARALLEL_MIN_VALUES', 1000)
    population_stats('S1', df, gates, list(gates), {}, chans, ALL, STAT_CHAN,
                     transforms=tr)
    assert made == [4]
