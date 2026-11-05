"""Characterization of population_stats, pinned before it was split into
helpers: the rows it returns must not change.

Two kinds of check. A few small tables are worked out by hand (count,
%Parent, %Grandparent, %Total, mean, median, the n - 1 SD, CV, MAD, a
percentile, the textbook GeoMean) and asserted directly. Then a golden
record: every branch of population_stats (each gate kind, booleans, nested
parents, empty and one-event populations, gates that read NA, NaN and inf
values, every GeoMean and rCV form, transforms that are logicle, guessed or
unknown, a FlowJo axis, the all-events row, select, the threaded path) run
on generated data, with each row's keys in order and each float as its hex
form, compared against tests/golden/population_stats_split.json: every key,
string, type, NaN and infinity exactly, every finite float to within
FLOAT_REL_TOL. The last bits of a float are not portable: the same pinned
numpy sums and takes logs with different SIMD widths and libm on each CPU and
OS (measured on CI: 1 ulp on Linux, 5 on Windows, 37 on macOS, all below
5e-15 relative), so a bit-exact record fails on every machine but the one
that wrote it.

Synthetic data and channel names only. Regenerate the golden record (only
when a change in the numbers is intended) with
``python tests/test_population_stats_split.py --write``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openflo.channel_transforms import (
    UNKNOWN_SPEC,
    guessed_spec,
    transform_spec,
    transform_values,
)
from openflo.gating import CHANNEL_STAT_ORDER, MFI_RATIO, population_stats

GOLDEN = Path(__file__).parent / 'golden' / 'population_stats_split.json'
# About 200 times the largest platform drift seen on CI (4.7e-15), and a
# thousand times below a planted 1e-9 relative error, which fails.
FLOAT_REL_TOL = 1e-12
CH = ['FL1-A', 'FL2-A', 'FSC-A', 'SSC-A']
NAN = float('nan')

ALL_WANT = frozenset({'Count', '%Parent', '%Grandparent', '%Total',
                      'Median', 'Mean', 'GeoMean', 'SD', 'rSD', 'CV', 'rCV',
                      'MAD', 'P5', 'P50', 'P2.5', 'P95', MFI_RATIO})
FORMS = [None,
         {'geomean': 'display', 'rcv': 'percentile'},
         {'geomean': 'log', 'rcv': 'mad'},
         {'geomean': 'display', 'rcv': 'mad'},
         {'geomean': 'log', 'rcv': 'percentile'}]
WSP_AXIS = {'kind': 'logicle', 'T': 262144.0, 'W': 1.0, 'M': 4.4185,
            'A': 0.0}


# ── hand-worked tables ──────────────────────────────────────────────────────

def _small():
    """Ten events: 'lo' holds the eight in [0, 10), 'mid' the four of those
    at or above 5, 'top' the single 9, 'none' nothing."""
    vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0, 100.0, 200.0]
    df = pd.DataFrame({'FL1-A': vals, 'FSC-A': np.arange(10.0)})
    gates = {
        'lo': {'kind': 'interval', 'channel': 'FL1-A', 'lo': 0.0, 'hi': 10.0,
               'parent_id': None, 'name': 'lo'},
        'mid': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 5.0,
                'parent_id': 'lo', 'name': 'mid'},
        'top': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 9.0,
                'parent_id': 'mid', 'name': 'top'},
        'none': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 1e9,
                 'parent_id': 'mid', 'name': 'none'},
        'kid': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.0,
                'parent_id': 'none', 'name': 'kid'},
        'gone': {'kind': 'threshold', 'channel': 'FL9-A', 'value': 0.0,
                 'parent_id': 'lo', 'name': 'gone'},
    }
    return df, gates


def _small_rows(forms=None, all_events=True):
    df, gates = _small()
    rows = population_stats('sample_a', df, gates, None, {}, ['FL1-A'],
                            set(ALL_WANT), CHANNEL_STAT_ORDER,
                            stat_forms=forms, all_events=all_events)
    return {r['__gid__']: r for r in rows}, rows


def test_counts_and_frequencies_by_hand():
    by, rows = _small_rows()
    assert [r['Population'] for r in rows] == [
        'All events', 'lo', 'lo/mid', 'lo/mid/top', 'lo/mid/none',
        'lo/mid/none/kid', 'lo/gone']
    ae = by[None]
    assert ae['Count'] == 10 and ae['%Total'] == 100.0
    assert math.isnan(ae['%Parent']) and math.isnan(ae['%Grandparent'])
    lo, mid, top = by['lo'], by['mid'], by['top']
    assert lo['Count'] == 8 and lo['%Parent'] == 80.0 and lo['%Total'] == 80.0
    # A root gate has no grandparent.
    assert math.isnan(lo['%Grandparent'])
    assert mid['Count'] == 4 and mid['%Parent'] == 50.0
    assert mid['%Grandparent'] == 40.0 and mid['%Total'] == 40.0
    assert top['Count'] == 1 and top['%Parent'] == 25.0
    assert top['%Grandparent'] == 12.5 and top['%Total'] == 10.0


def test_channel_statistics_by_hand():
    by, _ = _small_rows(forms={'geomean': 'log', 'rcv': 'percentile'})
    lo, mid = by['lo'], by['mid']
    # {2, 4, 4, 4, 5, 5, 7, 9}: mean 5, squared deviations sum to 32.
    assert lo['Mean FL1-A'] == 5.0 and lo['Median FL1-A'] == 4.5
    assert lo['SD FL1-A'] == pytest.approx(math.sqrt(32 / 7), rel=1e-15)
    assert lo['CV FL1-A'] == pytest.approx(math.sqrt(32 / 7) / 5 * 100,
                                           rel=1e-15)
    # |x - 4.5| sorted: 0.5 x5, 2.5, 2.5, 4.5 -> MAD 0.5.
    assert lo['MAD FL1-A'] == 0.5
    assert lo['P50 FL1-A'] == 4.5
    logs = [math.log(v) for v in (2, 4, 4, 4, 5, 5, 7, 9)]
    assert lo['GeoMean FL1-A'] == pytest.approx(math.exp(sum(logs) / 8),
                                                rel=1e-14)
    # {5, 5, 7, 9}: mean 6.5, squared deviations 2.25+2.25+0.25+6.25 = 11.
    assert mid['Mean FL1-A'] == 6.5 and mid['Median FL1-A'] == 6.0
    assert mid['SD FL1-A'] == pytest.approx(math.sqrt(11 / 3), rel=1e-15)
    assert mid['__median__'] == {'FL1-A': 6.0}


def test_one_empty_and_unknown_populations_by_hand():
    by, _ = _small_rows()
    top, none, kid, gone = by['top'], by['none'], by['kid'], by['gone']
    # One event: a mean and median, but no SD or CV.
    assert top['Mean FL1-A'] == 9.0 and top['Median FL1-A'] == 9.0
    assert math.isnan(top['SD FL1-A']) and math.isnan(top['CV FL1-A'])
    # Empty: 0 events, 0 %, NaN statistics; its child has no %Parent.
    assert none['Count'] == 0 and none['%Parent'] == 0.0
    assert math.isnan(none['Mean FL1-A'])
    assert kid['Count'] == 0 and math.isnan(kid['%Parent'])
    assert kid['%Grandparent'] == 0.0
    # A gate on a channel the sample lacks is NA, not 0.
    assert 'FL9-A' in gone['__unknown__']
    for k in ('Count', '%Parent', '%Grandparent', '%Total', 'Mean FL1-A'):
        assert math.isnan(gone[k]), k


def test_all_events_row_only_when_asked():
    _, rows = _small_rows(all_events=False)
    assert rows[0]['__gid__'] == 'lo' and len(rows) == 6


# ── golden record ───────────────────────────────────────────────────────────

def _events(n, seed):
    rng = np.random.default_rng(seed)
    fl1 = np.concatenate([rng.lognormal(7.0, 0.8, n - n // 4),
                          rng.normal(0.0, 150.0, n // 4)])
    rng.shuffle(fl1)
    fl2 = rng.lognormal(6.0, 1.1, n) - 80.0
    fsc = rng.uniform(0.0, 250000.0, n)
    ssc = rng.uniform(0.0, 250000.0, n)
    if n >= 20:
        fl1[[3, 11]] = np.nan
        fl1[17] = np.inf
        fl2[5] = -np.inf
    return pd.DataFrame({'FL1-A': fl1, 'FL2-A': fl2, 'FSC-A': fsc,
                         'SSC-A': ssc})


def _gates(df, fl_cut, fl2_band):
    """Every kind of gate, nested, plus empty, one-event and NA ones.
    `fl_cut` / `fl2_band` are thresholds in df's stored FL1 / FL2 units."""
    one_at = float(np.nanmax(df['FSC-A'].values)) if len(df) else 0.0
    q = {'x_channel': 'FL1-A', 'y_channel': 'FL2-A', 'quad_set': 'qs1',
         'parent_id': 'ell'}
    lo, hi = fl2_band
    big = 1e12
    return {
        'cells': {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
                  'x0': 20000.0, 'x1': 240000.0, 'y0': 10000.0,
                  'y1': 245000.0, 'parent_id': None, 'name': 'cells'},
        'sing': {'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
                 'vertices': [[25000.0, 15000.0], [235000.0, 20000.0],
                              [230000.0, 240000.0], [30000.0, 200000.0]],
                 'parent_id': 'cells'},
        'ell': {'kind': 'ellipsoid', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
                'mean': [130000.0, 125000.0],
                'cov': [[2.5e9, 3e8], [3e8, 2.0e9]], 'distance_sq': 4.0,
                'parent_id': 'sing', 'name': 'ell'},
        'pos': {'kind': 'threshold', 'channel': 'FL1-A', 'value': fl_cut,
                'parent_id': 'sing'},
        'band': {'kind': 'interval', 'channel': 'FL2-A', 'lo': lo, 'hi': hi,
                 'parent_id': 'sing'},
        'q_pp': dict(q, kind='rect', x0=fl_cut, x1=big, y0=lo, y1=big),
        'q_pn': dict(q, kind='rect', x0=fl_cut, x1=big, y0=-big, y1=lo),
        'q_np': dict(q, kind='rect', x0=-big, x1=fl_cut, y0=lo, y1=big),
        'q_nn': dict(q, kind='rect', x0=-big, x1=fl_cut, y0=-big, y1=lo),
        'and': {'kind': 'boolean', 'op': 'and', 'operands': ['pos', 'band'],
                'parent_id': 'sing', 'name': 'pos and band'},
        'or': {'kind': 'boolean', 'op': 'or', 'operands': ['pos', 'band'],
               'parent_id': 'sing'},
        'not': {'kind': 'boolean', 'op': 'not', 'operands': ['pos'],
                'parent_id': 'sing'},
        'deep': {'kind': 'threshold', 'channel': 'SSC-A', 'value': 100000.0,
                 'parent_id': 'q_pp', 'name': 'deep'},
        # An interval, not a threshold: the inf event is >= any cut.
        'empty': {'kind': 'interval', 'channel': 'FL1-A', 'lo': big,
                  'hi': 2 * big, 'parent_id': 'pos', 'name': 'empty'},
        'under_empty': {'kind': 'threshold', 'channel': 'SSC-A', 'value': 0.0,
                        'parent_id': 'empty'},
        'one': {'kind': 'threshold', 'channel': 'FSC-A', 'value': one_at,
                'parent_id': None, 'name': 'one'},
        'missing': {'kind': 'threshold', 'channel': 'FL9-A', 'value': 1.0,
                    'parent_id': 'cells', 'name': 'missing'},
        'under_missing': {'kind': 'threshold', 'channel': 'FSC-A',
                          'value': 0.0, 'parent_id': 'missing'},
        'unk_scale': {'kind': 'rect', 'x_channel': 'FL1-A',
                      'y_channel': 'FL2-A', 'x0': 0.0, 'x1': 1.0, 'y0': 0.0,
                      'y1': 1.0, 'scale_unknown': True, 'parent_id': None},
        'bad_ell': {'kind': 'ellipsoid', 'x_channel': 'FSC-A',
                    'y_channel': 'SSC-A', 'mean': [1.0, 1.0],
                    'cov': [[1.0, 1.0], [1.0, 1.0]], 'parent_id': 'cells'},
        'lost_op': {'kind': 'boolean', 'op': 'and',
                    'operands': ['pos', 'no_such_gate'], 'parent_id': 'sing'},
        'fj_ell': {'kind': 'flowjo_ellipse', 'x_channel': 'FSC-A',
                   'y_channel': 'SSC-A', 'parent_id': None},
    }


def _logicle_df(df, chans):
    out = df.copy()
    spec = transform_spec('logicle')
    params = {k: v for k, v in spec.items() if k != 'method'}
    for ch in chans:
        out[ch] = transform_values(df[ch].values, 'logicle', **params)
    return out, spec


def scenarios():
    """``[(name, kwargs, threaded)]`` covering every branch."""
    out = []
    lin = _events(1500, 11)
    g_lin = _gates(lin, 1500.0, (200.0, 3000.0))
    # FSC-A / SSC-A statistics in two scenarios only: the fluorescence
    # channels carry every branch, and the record stays small.
    base = {'sample_name': 'sample_a', 'df': lin, 'gates': g_lin,
            'order': None,
            'channel_labels': {'FL1-A': 'FL1-A', 'FL2-A': 'FL2-A'},
            'channels': ['FL1-A', 'FL9-A', 'FL2-A'], 'want': set(ALL_WANT),
            'stat_chan': CHANNEL_STAT_ORDER}
    out.append(('linear_all_channels',
                dict(base, channels=CH + ['FL9-A'], all_events=True), False))
    for i, forms in enumerate(FORMS):
        ae = bool(i % 2)
        out.append((f'linear_forms{i}_ae{int(ae)}',
                    dict(base, stat_forms=forms, all_events=ae), False))

    # Stored logicle on both fluorescence channels; gates in display units.
    disp, spec = _logicle_df(lin, ['FL1-A', 'FL2-A'])
    g_disp = _gates(disp, 0.55, (0.35, 0.7))
    tr = {'FL1-A': spec, 'FL2-A': spec}
    for i, forms in enumerate(FORMS):
        ae = bool(i % 2)
        out.append((f'logicle_forms{i}_ae{int(ae)}',
                    dict(base, df=disp, gates=g_disp, transforms=tr,
                         stat_forms=forms, all_events=ae), False))
        out.append((f'logicle_axis_forms{i}_ae{int(not ae)}',
                    dict(base, df=disp, gates=g_disp, transforms=tr,
                         stat_axes={'FL1-A': WSP_AXIS}, stat_forms=forms,
                         all_events=not ae), False))
    out.append(('guessed_and_unknown',
                dict(base, df=disp, gates=g_disp,
                     transforms={'FL1-A': guessed_spec(spec),
                                 'FL2-A': dict(UNKNOWN_SPEC)},
                     channel_labels={'FL1-A': 'FL1-A', 'FL2-A': 'FL2-A'},
                     all_events=True), False))
    out.append(('guessed_and_unknown_log',
                dict(base, df=disp, gates=g_disp,
                     transforms={'FL1-A': guessed_spec(spec),
                                 'FL2-A': dict(UNKNOWN_SPEC)},
                     stat_forms=FORMS[2], all_events=True), False))
    out.append(('axis_unreadable',
                dict(base, df=disp, gates=g_disp, transforms=tr,
                     stat_axes={'FL1-A': {'kind': 'nonsense'},
                                'FL2-A': {'kind': 'log', 'offset': 1.0,
                                          'decades': 4.5}}), False))

    # Which rows and which statistics.
    out.append(('select_subset',
                dict(base, select=['ell', 'no_such_gate', 'missing', 'pos'],
                     all_events=True), False))
    out.append(('explicit_order',
                dict(base, order=['pos', 'sing', 'cells', 'ghost', 'band']),
                False))
    out.append(('counts_only',
                dict(base, want={'Count', '%Parent', '%Total'},
                     all_events=True), False))
    out.append(('no_counts_some_stats',
                dict(base, want={'Mean', 'SD', 'P95'},
                     stat_chan=('Mean', 'Median'), all_events=True), False))
    out.append(('ratio_only',
                dict(base, want={MFI_RATIO, 'Count'}), False))
    out.append(('pct_bad_names',
                dict(base, want={'P101', 'Px', 'P0', 'P100', 'Median'}), False))
    out.append(('no_channels', dict(base, channels=[]), False))
    out.append(('labels_renamed',
                dict(base, channel_labels={'FL1-A': 'marker one',
                                           'FSC-A': 'size'}), False))
    out.append(('gp_count_only',
                dict(base, want={'%Grandparent'}, all_events=True), False))

    # Population sizes.
    zero = _events(0, 3)
    out.append(('zero_events',
                dict(base, df=zero, gates=_gates(zero, 1500.0, (200.0, 3000.0)),
                     all_events=True), False))
    out.append(('zero_events_counts',
                dict(base, df=zero, gates=_gates(zero, 1500.0, (200.0, 3000.0)),
                     want={'Count', '%Total', '%Parent'}, all_events=True),
                False))
    single = _events(1, 5)
    out.append(('one_event',
                dict(base, df=single,
                     gates=_gates(single, 1500.0, (200.0, 3000.0)),
                     all_events=True), False))

    # The threaded path (the same numbers, by construction).
    out.append(('threaded', dict(base, channels=CH, all_events=True), True))
    # Every row populated, so a pool that lost or reordered a population's
    # result shows.
    out.append(('threaded_select',
                dict(base, select=['q_nn', 'one', 'and', 'cells', 'deep'],
                     all_events=True), True))
    out.append(('threaded_logicle',
                dict(base, df=disp, gates=g_disp, transforms=tr,
                     stat_axes={'FL1-A': WSP_AXIS}, stat_forms=FORMS[3],
                     all_events=True), True))
    return out


def run(fn, name, kwargs, threaded):
    """Call population_stats-like `fn` on one scenario; threaded forces the
    pool by lowering the size cut-off on `fn`'s module."""
    mod = sys.modules[fn.__module__]
    old = mod._PARALLEL_MIN_VALUES
    if threaded:
        mod._PARALLEL_MIN_VALUES = 0
    try:
        return fn(**kwargs)
    finally:
        mod._PARALLEL_MIN_VALUES = old


def encode(v):
    """`v` as JSON that keeps every bit: a float as its bare hex form
    (``float.hex``), a string, other numbers and containers tagged with
    their type, dict key order kept."""
    if v is None:
        return v
    if type(v) is float:
        return v.hex()
    if type(v) is str:
        return {'s': v}
    if isinstance(v, (bool, np.bool_)):
        return {'b': bool(v), 't': type(v).__name__}
    if isinstance(v, (int, np.integer)):
        return {'i': int(v), 't': type(v).__name__}
    if isinstance(v, np.floating):
        return {'x': float(v).hex(), 't': type(v).__name__}
    if isinstance(v, dict):
        return {'d': [[encode(k), encode(x)] for k, x in v.items()]}
    if isinstance(v, (list, tuple)):
        return {'l': [encode(x) for x in v], 't': type(v).__name__}
    raise TypeError(f'cannot encode {type(v).__name__}')


def encode_rows(rows):
    """The rows as ``{'keys': [...], 'rows': [[[key index, value], ...]]}``:
    each row's keys in order, named once per table."""
    keys: list = []
    out = []
    for r in rows:
        assert type(r) is dict
        cells = []
        for k, v in r.items():
            if k not in keys:
                keys.append(k)
            cells.append([keys.index(k), encode(v)])
        out.append(cells)
    return {'keys': keys, 'rows': out}


def frame_shape(rows):
    """The DataFrame a caller builds from the rows: its columns, dtypes and
    index, and each cell's NaN mask."""
    df = pd.DataFrame(rows)
    return {'columns': [str(c) for c in df.columns],
            'dtypes': [str(t) for t in df.dtypes],
            'index': [int(i) for i in df.index],
            'na': [''.join('1' if x else '0' for x in col)
                   for col in df.isna().to_numpy().T.tolist()]}


def record(fn=population_stats):
    out = {}
    for name, kwargs, threaded in scenarios():
        rows = run(fn, name, kwargs, threaded)
        out[name] = {'rows': encode_rows(rows), 'frame': frame_shape(rows)}
    return out


def _floats_close(a, b):
    """Two hex floats are the same number, to FLOAT_REL_TOL if finite and
    non-zero; NaN, the infinities and signed zeros exactly."""
    x, y = float.fromhex(a), float.fromhex(b)
    if math.isfinite(x) and math.isfinite(y) and x and y:
        return math.isclose(x, y, rel_tol=FLOAT_REL_TOL, abs_tol=0.0)
    return a == b


def _snap(got, want):
    if type(got) is str and type(want) is str:
        return want if _floats_close(got, want) else got
    if type(got) is dict and type(want) is dict and got.keys() == want.keys():
        # A string value {'s': ...} and a type name {'t': ...}: exactly.
        return {k: got[k] if k in ('s', 't') else _snap(got[k], want[k])
                for k in got}
    if type(got) is list and type(want) is list and len(got) == len(want):
        return [_snap(g, w) for g, w in zip(got, want, strict=True)]
    return got


def snap_to_golden(got, want):
    """`got` (``encode_rows`` output) with each float that is within
    FLOAT_REL_TOL of the golden one at the same place replaced by the golden
    text, so ``snap_to_golden(got, want) == want`` holds exactly when
    nothing but last-bit float drift differs, and a failure's diff shows only
    the real differences. Keys are compared exactly; in the rows a bare
    string is always a float's hex form (``encode`` tags every other
    string)."""
    return dict(got, rows=_snap(got['rows'], want['rows']))


@pytest.fixture(scope='module')
def golden():
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize('name', [s[0] for s in scenarios()])
def test_matches_golden_record(golden, name):
    kwargs, threaded = next((k, t) for n, k, t in scenarios() if n == name)
    rows = run(population_stats, name, kwargs, threaded)
    want = golden[name]['rows']
    assert snap_to_golden(encode_rows(rows), want) == want
    assert frame_shape(rows) == golden[name]['frame']


def test_golden_record_covers_every_scenario(golden):
    assert sorted(golden) == sorted(s[0] for s in scenarios())


def test_threaded_and_serial_agree():
    name, kwargs, _ = next(s for s in scenarios() if s[0] == 'threaded')
    assert encode(run(population_stats, name, kwargs, True)) == encode(
        run(population_stats, name, kwargs, False))


def test_scenarios_reach_every_branch():
    """The golden record would pin nothing if the data missed a branch."""
    rows = run(population_stats, 'x', dict(scenarios()[0][1]), False)
    by = {r['__gid__']: r for r in rows}
    assert by['empty']['Count'] == 0 and math.isnan(by['under_empty']['%Parent'])
    assert by['one']['Count'] == 1
    for gid in ('missing', 'under_missing', 'unk_scale', 'bad_ell',
                'lost_op', 'fj_ell'):
        assert by[gid].get('__unknown__'), gid
    for gid in ('cells', 'sing', 'ell', 'pos', 'band', 'q_pp', 'q_pn',
                'q_np', 'q_nn', 'and', 'or', 'not', 'deep'):
        assert by[gid]['Count'] > 0, gid
    assert by['and']['Count'] < by['pos']['Count'] < by['or']['Count']


if __name__ == '__main__' and sys.argv[1:] == ['--write']:
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(record(), separators=(',', ':')) + '\n')
    print(f'wrote {GOLDEN.name}')
