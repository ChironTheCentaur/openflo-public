#!/usr/bin/env python
"""Write tests/reference/independent_reference.json: numbers OpenFlo's tests
check against, computed WITHOUT OpenFlo.

Most expected values in the suite were produced by running OpenFlo, so a
wrong formula and its test can agree. The values here come from elsewhere:

- compensation and gate membership from FlowKit (an independent flow
  cytometry library), on generated events stored in the file itself;
- the gates as a Gating-ML 2.0 document FlowKit writes, so OpenFlo's
  Gating-ML reader is checked against another program's file;
- logicle values by solving the defining equation of Moore & Parks (2012,
  Cytometry A 81:273) numerically, not with flowutils (which OpenFlo and
  FlowKit both use for logicle);
- Gating-ML 2.0 fasinh from the formula in the specification (section 6.3).

Run it in a separate environment that has FlowKit (OpenFlo does not depend
on it; CI only reads the JSON):

    python -m venv fk && fk/bin/pip install flowkit
    fk/bin/python scripts/make_reference_values.py

Inputs are drawn from a seeded generator and written into the JSON, so the
tests never regenerate them.
"""
from __future__ import annotations

import io
import json
import math
import pathlib
import sys

import numpy as np

OUT = (pathlib.Path(__file__).resolve().parents[1] / 'tests' / 'reference'
       / 'independent_reference.json')
CH = ['FL1-A', 'FL2-A', 'FL3-A']


def _events(rng, n=2000):
    """Three overlapping populations with a negative tail, rounded to whole
    numbers: FlowKit holds events as float32, which stores integers of this
    size exactly, so both programs see the same inputs."""
    a = rng.multivariate_normal([800, 300, 200], np.diag([300, 150, 80]) ** 2,
                                n // 2)
    b = rng.multivariate_normal([4000, 2500, 600],
                                [[900 ** 2, 0.6 * 900 * 700, 0],
                                 [0.6 * 900 * 700, 700 ** 2, 0],
                                 [0, 0, 250 ** 2]], n // 4)
    c = rng.multivariate_normal([-50, 9000, 3000], np.diag([200, 2000, 900]) ** 2,
                                n - n // 2 - n // 4)
    ev = np.vstack([a, b, c])
    rng.shuffle(ev)
    return np.round(ev)


# An asymmetric spillover matrix (rows: fluorochrome, columns: detector), so
# applying it transposed gives different numbers.
SPILL = [[1.0, 0.25, 0.04],
         [0.02, 1.0, 0.15],
         [0.10, 0.01, 1.0]]


def _gates(fk):
    from flowkit import gates as G
    dim = fk.Dimension
    out = [
        G.RectangleGate('rect', [dim('FL1-A', range_min=1500, range_max=6000),
                                 dim('FL2-A', range_min=1000, range_max=5000)]),
        # One-sided range: only a lower bound.
        G.RectangleGate('fl3_high', [dim('FL3-A', range_min=1000)]),
        # A concave polygon (an L shape), so a convex-hull bug shows.
        G.PolygonGate('poly', [dim('FL1-A'), dim('FL2-A')],
                      [[0, 0], [3000, 0], [3000, 1500], [1200, 1500],
                       [1200, 6000], [0, 6000]]),
        G.EllipsoidGate('ell', [dim('FL1-A'), dim('FL2-A')],
                        coordinates=[4000, 2500],
                        covariance_matrix=[[900 ** 2, 0.6 * 900 * 700],
                                           [0.6 * 900 * 700, 700 ** 2]],
                        distance_square=4.0),
        G.BooleanGate('rect_and_not_ell', 'and',
                      [{'ref': 'rect', 'path': ('root',), 'complement': False},
                       {'ref': 'ell', 'path': ('root',), 'complement': True}]),
        G.BooleanGate('poly_or_fl3', 'or',
                      [{'ref': 'poly', 'path': ('root',), 'complement': False},
                       {'ref': 'fl3_high', 'path': ('root',),
                        'complement': False}]),
    ]
    return out


def _gating(fk, events):
    import pandas as pd
    sample = fk.Sample(pd.DataFrame(events, columns=CH), sample_id='synthetic_1')
    gs = fk.GatingStrategy()
    for g in _gates(fk):
        gs.add_gate(g, gate_path=('root',))
    res = gs.gate_sample(sample)
    members = {g.gate_name: np.flatnonzero(
        res.get_gate_membership(g.gate_name)).tolist() for g in _gates(fk)}
    buf = io.BytesIO()
    fk.export_gatingml(gs, buf)
    return members, buf.getvalue().decode('utf-8')


def _compensated(fk, events):
    import pandas as pd
    sample = fk.Sample(pd.DataFrame(events, columns=CH), sample_id='synthetic_1')
    m = fk.Matrix(np.array(SPILL), CH, fluorochromes=CH)
    comp = m.apply(sample)
    return np.asarray(comp, dtype=float)


# ── logicle from its definition (Moore & Parks 2012, eqs. 5-11) ─────────────

def _logicle_params(T, W, M, A):
    w = W / (M + A)
    x2 = A / (M + A)
    x1 = x2 + w
    x0 = x2 + 2 * w
    b = (M + A) * math.log(10)
    # d solves 2 (ln d - ln b) + w (b + d) = 0 (eq. 9); bisection on (0, b).
    lo, hi = 1e-12, b
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 2 * (math.log(mid) - math.log(b)) + w * (b + mid) > 0:
            hi = mid
        else:
            lo = mid
    d = 0.5 * (lo + hi)
    c_a = math.exp(x0 * (b + d))
    mf_a = math.exp(b * x1) - c_a / math.exp(d * x1)
    a = T / ((math.exp(b) - mf_a) - c_a / math.exp(d))
    return dict(a=a, b=b, c=c_a * a, d=d, f=-mf_a * a, x1=x1)


def _logicle(x, T, W, M, A):
    """Display value y (y = 1 at T) with S(y) = x, by bisection on S,
    the biexponential a e^(by) - c e^(-dy) + f (f = -mf_a a), made odd
    about x1."""
    p = _logicle_params(T, W, M, A)

    def S(y):
        if y >= p['x1']:
            return (p['a'] * math.exp(p['b'] * y) + p['f']
                    - p['c'] * math.exp(-p['d'] * y))
        return -S(2 * p['x1'] - y)

    lo, hi = -2.0, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if S(mid) < x:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _fasinh(x, T, M, A):
    """Gating-ML 2.0 section 6.3 (fasinh)."""
    return ((math.asinh(x * math.sinh(M * math.log(10)) / T) + A * math.log(10))
            / ((M + A) * math.log(10)))


LOGICLE_X = [-5000.0, -1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0,
             10000.0, 100000.0, 262144.0]
LOGICLE_PARAMS = [dict(T=262144.0, W=0.5, M=4.5, A=0.0),
                  dict(T=262144.0, W=1.0, M=4.5, A=1.0),
                  dict(T=10000.0, W=0.8, M=4.0, A=0.5)]
FASINH_PARAMS = [dict(T=262144.0, M=4.5, A=0.0), dict(T=10000.0, M=4.0, A=1.0)]


def _stats(events, members):
    """Per-population statistics from scipy's own estimators (SD n - 1,
    median absolute deviation) and the FlowJo rSD definition, on the raw
    linear events of each FlowKit gate."""
    from scipy import stats as st
    out = {}
    for name in ('rect', 'poly', 'ell'):
        rows = events[members[name]]
        out[name] = {'count': len(rows),
                     'pct_of_total': 100.0 * len(rows) / len(events)}
        for j, ch in enumerate(CH):
            v = rows[:, j]
            lo, hi = np.percentile(v, [15.87, 84.13])
            out[name][ch] = {
                'median': float(np.median(v)), 'mean': float(st.tmean(v)),
                'sd': float(st.tstd(v)),
                'cv': float(100 * st.variation(v, ddof=1)),
                'mad': float(st.median_abs_deviation(v)),
                'rsd_percentile': float(0.5 * (hi - lo)),
                'rsd_mad': float(st.median_abs_deviation(v, scale='normal')),
            }
    return out


def main():
    import flowkit as fk  # pyright: ignore[reportMissingImports]
    rng = np.random.default_rng(20261020)
    events = _events(rng)
    members, gml = _gating(fk, events)
    comp = _compensated(fk, events)
    # The logicle anchors: S(x1) = 0 and S(1) = T must hold for any correct
    # parameter solution; check before trusting the solver.
    for prm in LOGICLE_PARAMS:
        assert abs(_logicle(0.0, **prm) - _logicle_params(**prm)['x1']) < 1e-12
        assert abs(_logicle(prm['T'], **prm) - 1.0) < 1e-12
    out = {
        '_about': ('Reference values computed without OpenFlo by '
                   'scripts/make_reference_values.py. Do not edit by hand; '
                   'do not regenerate with OpenFlo.'),
        'sources': {
            'flowkit': fk.__version__,
            'numpy': np.__version__,
            'scipy': __import__('scipy').__version__,
            'stats': 'scipy.stats (tstd, variation, median_abs_deviation)',
            'python': sys.version.split()[0],
            'logicle': 'Moore & Parks 2012 eqs. 5-11, solved by bisection',
            'fasinh': 'Gating-ML 2.0 specification, section 6.3',
        },
        'channels': CH,
        'events': events.tolist(),
        'spillover': SPILL,
        'compensated': np.round(comp, 9).tolist(),
        'gatingml': gml,
        'membership': members,
        'stats': _stats(events, members),
        'logicle': [dict(prm, x=LOGICLE_X,
                         y=[_logicle(x, **prm) for x in LOGICLE_X])
                    for prm in LOGICLE_PARAMS],
        'fasinh': [dict(prm, x=LOGICLE_X,
                        y=[_fasinh(x, **prm) for x in LOGICLE_X])
                   for prm in FASINH_PARAMS],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out) + '\n', encoding='utf-8')
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB); gate counts:",
          {k: len(v) for k, v in members.items()})


if __name__ == '__main__':
    main()
