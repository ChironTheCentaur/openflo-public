"""openflo-voltage end to end: analyze() over real FCS files and main()'s CSV.

The pure metric layer is pinned in tests/test_voltage.py; this covers the
glue that file says "isn't re-tested here": resolving an antibody LABEL to
its detector, reading that detector's $PnV, pooling replicate files at one
voltage, and the console entry's CSV / exit codes. The expected Stain Index
is computed from the TRUE population labels with the documented formula,
so the test does not trust the tool's own GMM split to define the answer.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from openflo.voltage import VoltageTitration, main

CH = ['FSC-A', 'SSC-A', 'FL5-A', 'FL2-A', 'Time']
MK = ['', '', 'CD4', 'CD8', '']
N_NEG, N_POS = 2000, 1000
# voltage -> (mu_neg, sd_neg, mu_pos) for FL5-A; SI rises then flattens.
PLAN = {300: (200, 100, 2200), 400: (300, 80, 3500), 500: (400, 70, 4400),
        600: (500, 70, 5500), 700: (600, 75, 6400)}


def _si(pos, neg):
    mad = np.median(np.abs(neg - np.median(neg)))
    return (np.median(pos) - np.median(neg)) / (2 * 1.4826 * mad)


@pytest.fixture(scope='module')
def series(tmp_path_factory):
    import flowio
    root = tmp_path_factory.mktemp('volt')
    files, truth = [], {}
    # 500 V appears twice: the two files must be POOLED into one point.
    for i, v in enumerate([300, 400, 500, 500, 600, 700]):
        rng = np.random.default_rng(100 + i)
        mn, sd, mp = PLAN[v]
        neg = rng.normal(mn, sd, N_NEG)
        pos = rng.normal(mp, 2 * sd, N_POS)
        n = N_NEG + N_POS
        cols = {'FSC-A': rng.normal(5e4, 3e3, n), 'SSC-A': rng.normal(2e4, 2e3, n),
                'FL5-A': np.concatenate([neg, pos]),
                'FL2-A': rng.normal(300, 50, n),        # unimodal, no titration
                'Time': np.arange(n, dtype=float)}
        ev = np.column_stack([cols[c] for c in CH]).astype(np.float32)
        p = root / f'tit_{v}_{i}.fcs'
        with open(p, 'wb') as fh:
            flowio.create_fcs(fh, ev.flatten().tolist(), CH,
                              opt_channel_names=MK,
                              metadata_dict={f'$P{j + 1}V': str(v)
                                             for j in range(len(CH))})
        files.append(str(p))
        t = truth.setdefault(v, ([], []))
        t[0].append(neg.astype(np.float32).astype(float))
        t[1].append(pos.astype(np.float32).astype(float))
    exp = {v: _si(np.concatenate(p), np.concatenate(n))
           for v, (n, p) in truth.items()}
    mx = max(exp.values())
    rec = min(v for v, s in exp.items() if s >= 0.95 * mx)
    return files, exp, rec, root


def test_analyze_by_label_reads_detector_voltage_and_pools(series):
    files, exp, rec, _ = series
    res = VoltageTitration.analyze(files, channels=['CD4'])
    assert res['order'] == ['CD4']
    out = res['results']['CD4']
    by_v = {r['voltage']: r for r in out['rows']}
    # Every file's $PnV was found (no None bucket) and 500 V pooled 2 files.
    assert sorted(by_v) == [300.0, 400.0, 500.0, 600.0, 700.0]
    assert by_v[500.0]['n_files'] == 2
    assert by_v[500.0]['n_events'] == 2 * (N_NEG + N_POS)
    for v, s in exp.items():
        assert by_v[float(v)]['si'] == pytest.approx(s, rel=0.02), v
    assert out['recommended_voltage'] == rec


def test_main_writes_csv_and_exit_codes(series, tmp_path):
    files, exp, rec, root = series
    out_csv = tmp_path / 'si.csv'
    # glob expansion is part of the CLI contract ("globs expanded")
    assert main([str(root / 'tit_*.fcs'), '--channel', 'FL5-A',
                 '-o', str(out_csv)]) == 0
    rows = list(csv.DictReader(open(out_csv, encoding='utf-8')))
    assert {r['channel'] for r in rows} == {'FL5-A'}
    got = {float(r['voltage']): float(r['si']) for r in rows}
    assert sorted(got) == [300.0, 400.0, 500.0, 600.0, 700.0]
    for v, s in exp.items():
        assert got[float(v)] == pytest.approx(s, rel=0.02)
    # A channel present in no file is reported, not silently written.
    assert main(files + ['--channel', 'NOPE']) == 1
    with pytest.raises(SystemExit) as ei:
        main(files)                                   # neither flag
    assert ei.value.code == 2


def test_all_channels_is_fluor_only(series):
    files, *_ = series
    res = VoltageTitration.analyze(files, all_channels=True)
    assert set(res['order']) == {'FL5-A', 'FL2-A'}      # not FSC/SSC/Time
