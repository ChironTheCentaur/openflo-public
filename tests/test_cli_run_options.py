"""openflo-run options that only run inside run(), which the default suite
never executes: --filter-debris-um/--bead-um (bead-calibrated debris cut),
--doublet-tol, --panel auto, --batch-correct.

Same in-process approach as test_cli_run.py: the expensive, separately
tested parts (clustering, UMAP, plots) are stubbed; every expected value is
fixed by how the synthetic files were built.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

import openflo.cli as cli
from openflo.pipeline import FlowSample

CH6 = ['FSC-A', 'FSC-H', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
MK6 = ['', '', '', 'CD901b', 'CD902', 'CD903']


def _write(path, ev):
    import flowio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).flatten().tolist(), CH6,
                          opt_channel_names=MK6)


def _fluor(rng, n, gain=1.0, sigma=0.5, frac_bv=.6):
    n1 = int(n * frac_bv)
    ln = rng.lognormal
    bv = np.r_[ln(np.log(20000), sigma, n1), ln(np.log(150), sigma, n - n1)]
    fl2 = np.r_[ln(np.log(150), sigma, n1), ln(np.log(20000), sigma, n - n1)]
    return np.column_stack([bv, fl2, ln(np.log(800), sigma, n)]) * gain


def _scatter(rng, n):
    a = rng.normal(1e5, 5e3, n)
    return np.column_stack([a, a / 1.2, rng.normal(9e3, 500, n)])


@pytest.fixture
def stubbed(monkeypatch):
    def fake_cluster(self, **kw):
        self.data['cluster'] = np.where(
            (self.data[self._resolve('FL1-A')]
             > self.data[self._resolve('FL2-A')]).to_numpy(), 0, 1)
        return self
    monkeypatch.setattr(FlowSample, 'cluster', fake_cluster)
    monkeypatch.setattr(cli, 'save_plots',
                        lambda s, out_dir: os.makedirs(out_dir, exist_ok=True))
    monkeypatch.setattr(cli, 'run_group_umap', lambda *a, **k: None)
    monkeypatch.setattr(cli, 'save_group_pair_scatters', lambda *a, **k: None)
    monkeypatch.setattr(cli, '_gpu_present', lambda: False)


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['openflo-run', *argv])
    return cli.main()


def test_debris_cut_is_bead_calibrated_and_doublets_use_fsc_ratio(
        tmp_path, stubbed, monkeypatch):
    """Beads (the FMO control) sit at FSC-A 80 000 = 8 um, so --filter-debris-um
    4 cuts at 40 000 and removes exactly the 200 debris events; --doublet-tol
    0.25 then removes exactly the 100 events whose FSC-A/FSC-H is 2.2 (cells
    are 1.2)."""
    rng = np.random.default_rng(0)
    trial = tmp_path / 'trial'
    bead = rng.normal(80000, 2000, 2000)
    _write(str(trial / 'expt_fmo_bv.fcs'),
           np.column_stack([bead, bead / 1.2, rng.normal(9e3, 500, 2000),
                            _fluor(rng, 2000)]))
    fa = np.r_[rng.normal(1e5, 5e3, 1000), rng.uniform(5e3, 3e4, 200),
               rng.normal(1.8e5, 5e3, 100)]
    fh = np.r_[fa[:1000] / 1.2, fa[1000:1200] / 1.2, fa[1200:] / 2.2]
    _write(str(trial / 'expt_s1.fcs'),
           np.column_stack([fa, fh, rng.normal(9e3, 500, 1300),
                            _fluor(rng, 1300)]))
    grp = json.dumps([{'name': 'G', 'samples': ['s1'], 'fmo_set': 'std'}])
    fmo = json.dumps({'std': {'FL1-A': 'fmo_bv'}})
    kept = {}
    for label, extra in (('none', []),
                         ('debris', ['--filter-debris-um', '4']),
                         ('both', ['--filter-debris-um', '4',
                                   '--doublet-tol', '0.25'])):
        out = tmp_path / label
        _main(monkeypatch, '--trials', str(trial), '--out', str(out),
              '--groups', grp, '--fmo-sets', fmo, '--bead-um', '8', '-q',
              *extra)
        kept[label] = int(pd.read_csv(out / 'G' / 's1_stats.csv')['count'].sum())
    assert kept == {'none': 1300, 'debris': 1100, 'both': 1000}


def test_panel_auto_labels_reach_the_stats(tmp_path, stubbed, monkeypatch):
    rng = np.random.default_rng(1)
    day = tmp_path / 'study' / 'day1'
    _write(str(day / 'expt_p1.fcs'),
           np.column_stack([_scatter(rng, 800), _fluor(rng, 800)]))
    pd.DataFrame([['Marker', 'Fluor'], ['CD99', 'FL1'], ['CD7', 'FL2']]) \
        .to_excel(tmp_path / 'study' / 'staining panel.xlsx', header=False,
                  index=False)
    out = tmp_path / 'out'
    _main(monkeypatch, '--trials', str(day), '--out', str(out), '--groups',
          json.dumps([{'name': 'G', 'samples': ['p1'], 'fmo_set': ''}]),
          '--fmo-sets', '{}', '--panel', 'auto', '--labels', 'FL3-A=CD45RA',
          '-q')
    cols = list(pd.read_csv(out / 'G' / 'p1_stats.csv').columns)
    # panel overrides $PnS; --labels overrides both
    assert [c for c in cols if c.startswith('median_')] == \
        ['median_CD99', 'median_CD7', 'median_CD45RA']


def test_batch_correct_pulls_two_days_together(tmp_path, stubbed, monkeypatch):
    """Day 3 = Day 0 with a 1.5x gain on every marker. Without correction the
    FL1+ medians differ by ~0.04 (logicle units); CytoNorm must remove most
    of that gap."""
    par = tmp_path / 'expt'
    for d, gain, names in (('day0', 1.0, ['a1', 'a2']),
                           ('day3', 1.5, ['b1', 'b2'])):
        for i, nm in enumerate(names):
            rng = np.random.default_rng(100 * int(d[-1]) + i)
            _write(str(par / d / f'{nm}.fcs'),
                   np.column_stack([_scatter(rng, 3000),
                                    _fluor(rng, 3000, gain)]))
    gap = {}
    for label, extra in (('off', []), ('on', ['--batch-correct'])):
        out = tmp_path / label
        _main(monkeypatch, '--trials', str(par), '--out', str(out),
              '--fmo-sets', '{}', '-q', *extra)
        med = {}
        for d, names in (('Day_0', ['a1', 'a2']), ('Day_3', ['b1', 'b2'])):
            med[d] = np.mean([
                pd.read_csv(out / d / f'{nm}_stats.csv')
                .query('cluster == 0')['median_CD901b'].iloc[0] for nm in names])
        gap[label] = med['Day_3'] - med['Day_0']
    assert gap['off'] > 0.03
    assert abs(gap['on']) < 0.35 * gap['off']
