"""openflo-run (cli.main -> run -> _run_independent / _run_concatenated) on a
KNOWN answer, in-process and fast.

The default suite never executes run(): the only full runs are the
OPENFLO_RUN_SLOW_TESTS subprocess smokes, and they assert that files exist,
not what is in them. Here the expensive, separately-tested parts are stubbed
so the WIRING of a run is checked against exact numbers:

  * FlowSample.cluster -> an ORACLE labelling (cluster 0 = FL1+, 1 = FL2+),
    consistent across samples, so every expected percentage is known.
    `size_ranked` mimics PhenoGraph instead (it numbers communities by size).
  * run_group_umap / save_group_pair_scatters / save_plots -> recorders
    (save_plots still creates the folder export_stats writes into).
  * _gpu_present -> False (skips a 6 s nvidia-smi call).

Every sample holds two well-separated populations; the FL1+ fraction is
set per file, so each <name>_stats.csv and every compare_*.csv row has an
exact expected value.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

import openflo.cli as cli
from openflo.pipeline import FlowSample

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
MK = ['', '', 'CD901b', 'CD902', 'CD903']


def _write(path, n, frac_bv, seed, mk=MK, drop=()):
    """`frac_bv` of the events are FL1-high / FL2-low, the rest the mirror
    image. `mk` is the file's own $PnS panel; `drop` leaves channels out."""
    import flowio
    rng = np.random.default_rng(seed)
    n1 = int(round(n * frac_bv))
    n2 = n - n1
    ev = np.column_stack([
        rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
        np.r_[rng.normal(20000, 1500, n1), rng.normal(150, 40, n2)],
        np.r_[rng.normal(150, 40, n1), rng.normal(20000, 1500, n2)],
        rng.normal(800, 100, n)]).astype(np.float32)
    keep = [i for i, c in enumerate(CH) if c not in drop]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev[:, keep].flatten().tolist(),
                          [CH[i] for i in keep],
                          opt_channel_names=[mk[i] for i in keep])


@pytest.fixture
def stubbed(monkeypatch):
    calls: dict = {'umap': [], 'gated': []}
    mode = {'size_ranked': False}

    def fake_cluster(self, **kw):
        is_bv = (self.data[self._resolve('FL1-A')]
                 > self.data[self._resolve('FL2-A')]).to_numpy()
        if mode['size_ranked']:
            lab = np.where(is_bv == (is_bv.mean() >= .5), 0, 1)
        else:
            lab = np.where(is_bv, 0, 1)
        self.data['cluster'] = lab
        return self

    def fake_save_plots(s, out_dir):
        os.makedirs(out_dir, exist_ok=True)

    real_gate = FlowSample.apply_threshold_gates

    def spy_gate(self, thresholds):
        real_gate(self, thresholds)
        pos = {c: float(self.data[c].mean()) for c in self.data.columns
               if c.endswith('_pos')}
        calls['gated'].append((self.name, dict(thresholds), pos))
        return self

    monkeypatch.setattr(FlowSample, 'cluster', fake_cluster)
    monkeypatch.setattr(FlowSample, 'apply_threshold_gates', spy_gate)
    monkeypatch.setattr(cli, 'save_plots', fake_save_plots)
    monkeypatch.setattr(cli, 'run_group_umap',
                        lambda samples, label, out_dir, random_state=42:
                        calls['umap'].append((label, [s.name for s in samples])))
    monkeypatch.setattr(cli, 'save_group_pair_scatters',
                        lambda *a, **k: None)
    monkeypatch.setattr(cli, '_gpu_present', lambda: False)
    calls['mode'] = mode
    return calls


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['openflo-run', *argv])
    return cli.main()


def _bv(stats_csv):
    df = pd.read_csv(stats_csv)
    return float(df.loc[df['cluster'] == 0, 'pct_total'].sum())


@pytest.fixture
def two_groups(tmp_path):
    trial = tmp_path / 'trial'
    fr = {'c1': .7, 'c2': .6, 't1': .3, 't2': .2}
    for i, (nm, f) in enumerate(fr.items()):
        _write(str(trial / f'expt_{nm}.fcs'), 1200, f, 20 + i)
    groups = [{'name': 'Ctrl', 'samples': ['c1', 'c2'], 'fmo_set': ''},
              {'name': 'Treat', 'samples': ['t1', 't2'], 'fmo_set': ''}]
    return str(trial), groups, fr


# ── working paths: must pass now ──────────────────────────────────────────

def test_run_stats_and_compare_match_truth(two_groups, tmp_path, stubbed,
                                           monkeypatch):
    trial, groups, fr = two_groups
    out = tmp_path / 'out'
    assert _main(monkeypatch, '--trials', trial, '--out', str(out),
                 '--groups', json.dumps(groups), '--fmo-sets', '{}', '-q') == 0
    for g, names in (('Ctrl', ['c1', 'c2']), ('Treat', ['t1', 't2'])):
        for nm in names:
            assert _bv(out / g / f'{nm}_stats.csv') == pytest.approx(
                100 * fr[nm], abs=0.1)
    cmp = pd.read_csv(out / 'compare_Ctrl_vs_Treat.csv')
    row = cmp[(cmp.cluster == 0)].set_index('condition')
    assert row.loc['Ctrl', 'mean_pct'] == pytest.approx(65, abs=0.1)
    assert row.loc['Treat', 'mean_pct'] == pytest.approx(25, abs=0.1)
    assert set(row['n_samples']) == {2}
    assert sorted(stubbed['umap']) == [('Ctrl_trial', ['c1', 'c2']),
                                       ('Treat_trial', ['t1', 't2'])]


def test_by_day_same_filename_each_day_keeps_its_own_file(tmp_path, stubbed,
                                                          monkeypatch):
    par = tmp_path / 'expt'
    _write(str(par / 'day0' / 'sample_1.fcs'), 1200, .7, 1)
    _write(str(par / 'day3' / 'sample_1.fcs'), 1200, .3, 2)
    out = tmp_path / 'out'
    assert _main(monkeypatch, '--trials', str(par), '--out', str(out),
                 '--fmo-sets', '{}', '-q') == 0
    assert _bv(out / 'Day_0' / 'sample_1_stats.csv') == pytest.approx(70, abs=.1)
    assert _bv(out / 'Day_3' / 'sample_1_stats.csv') == pytest.approx(30, abs=.1)


def test_fmo_threshold_percentile_and_unstained_fallback(tmp_path, stubbed,
                                                         monkeypatch):
    """FL1 has a real FMO; FL2 has none, so it falls back to the unstained
    control. With --fmo-percentile 90, 10 % of each control's negatives
    sit above the cut, so the 70 % FL1+ sample is 70 + 0.1*30 = 73 %
    FL1_pos and 30 + 0.1*70 = 37 % FL2_pos (99.5 would give ~70 / ~30)."""
    trial = tmp_path / 'trial'
    _write(str(trial / 'expt_s1.fcs'), 2000, .7, 5)
    _write(str(trial / 'expt_fmo_bv.fcs'), 4000, 0.0, 6)     # all FL1-neg
    _write(str(trial / 'expt_unstained.fcs'), 4000, 1.0, 7)  # all FL2-neg
    groups = [{'name': 'G', 'samples': ['s1'], 'fmo_set': 'std'}]
    fmo = {'std': {'FL1-A': 'fmo_bv', 'FL2-A': 'fmo_fl2_missing'}}
    assert _main(monkeypatch, '--trials', str(trial), '--out',
                 str(tmp_path / 'o'), '--groups', json.dumps(groups),
                 '--fmo-sets', json.dumps(fmo), '--fmo-percentile', '90',
                 '-q') == 0
    (name, thr, pos), = stubbed['gated']
    assert set(thr) == {'FL1-A', 'FL2-A'}          # fallback supplied FL2
    assert pos['FL1-A_pos'] == pytest.approx(0.73, abs=0.012)
    assert pos['FL2-A_pos'] == pytest.approx(0.37, abs=0.012)


# ── formerly confirmed defects, now fixed ─────────────────────────────────

def test_by_day_same_filename_compare_has_both_days(tmp_path, stubbed,
                                                    monkeypatch):
    """_compare_group_pair keyed samples by name, so two day folders'
    identically named files collapsed to one: the CSV had only 'Day 0',
    holding Day 3's data."""
    par = tmp_path / 'expt'
    _write(str(par / 'day0' / 'sample_1.fcs'), 1200, .7, 1)
    _write(str(par / 'day3' / 'sample_1.fcs'), 1200, .3, 2)
    out = tmp_path / 'out'
    _main(monkeypatch, '--trials', str(par), '--out', str(out),
          '--fmo-sets', '{}', '-q')
    cmp = pd.read_csv(out / 'compare_Day_0_vs_Day_3.csv')
    row = cmp[cmp.cluster == 0].set_index('condition')['mean_pct']
    assert row.get('Day 0') == pytest.approx(70, abs=.1)
    assert row.get('Day 3') == pytest.approx(30, abs=.1)


def test_by_day_concatenate_reads_each_days_file(tmp_path, stubbed,
                                                 monkeypatch):
    """_run_concatenated ignored each by-day group's trial_dir and resolved
    names against the collapsed parent, so every day read day0's file."""
    par = tmp_path / 'expt'
    _write(str(par / 'day0' / 'sample_1.fcs'), 1200, .7, 1)
    _write(str(par / 'day3' / 'sample_1.fcs'), 1200, .3, 2)
    out = tmp_path / 'out'
    _main(monkeypatch, '--trials', str(par), '--out', str(out),
          '--fmo-sets', '{}', '--batch-mode', 'concatenate', '-q')
    assert _bv(out / 'Day_0' / 'sample_1_stats.csv') == pytest.approx(70, abs=.1)
    assert _bv(out / 'Day_3' / 'sample_1_stats.csv') == pytest.approx(30, abs=.1)


def test_skip_existing_keeps_group_compare_complete(two_groups, tmp_path,
                                                    stubbed, monkeypatch):
    """--skip-existing dropped the done samples from `results`, then rewrote
    the compare CSV and the UMAPs from the re-run samples alone."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    os.remove(out / 'Ctrl' / 'c2_stats.csv')       # lost in a crash
    os.remove(out / 'Treat' / 't2_stats.csv')
    _main(monkeypatch, *args, '--skip-existing')
    cmp = pd.read_csv(out / 'compare_Ctrl_vs_Treat.csv')
    assert set(cmp['n_samples']) == {2}


def test_skip_existing_resume_matches_the_full_run(two_groups, tmp_path,
                                                   stubbed, monkeypatch):
    """A resumed run reloads the kept samples' events, so its comparison is
    the full run's, value for value. A group's UMAP needs each sample's own
    clusters, which a kept sample no longer has, so it is left as it is."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    full = pd.read_csv(out / 'compare_Ctrl_vs_Treat.csv')
    shared = pd.read_csv(out / 'compare_shared_clusters.csv')
    os.remove(out / 'Ctrl' / 'c2_stats.csv')       # lost in a crash
    stubbed['umap'].clear()
    _main(monkeypatch, *args, '--skip-existing')
    pd.testing.assert_frame_equal(
        pd.read_csv(out / 'compare_Ctrl_vs_Treat.csv'), full)
    pd.testing.assert_frame_equal(
        pd.read_csv(out / 'compare_shared_clusters.csv'), shared)
    assert stubbed['umap'] == []    # Ctrl keeps c1; Treat was not re-run


def test_compare_detects_population_swap_under_size_ranked_ids(
        tmp_path, stubbed, monkeypatch):
    """Samples were clustered independently and compared by bare cluster id;
    PhenoGraph numbers clusters by size, so a 70% -> 30% population swap
    compared 70 vs 70 and was invisible. The comparison now uses one
    clustering shared by every sample."""
    stubbed['mode']['size_ranked'] = True
    trial = tmp_path / 'trial'
    _write(str(trial / 'expt_a.fcs'), 1200, .7, 1)
    _write(str(trial / 'expt_b.fcs'), 1200, .3, 2)
    groups = [{'name': 'A', 'samples': ['a'], 'fmo_set': ''},
              {'name': 'B', 'samples': ['b'], 'fmo_set': ''}]
    out = tmp_path / 'out'
    _main(monkeypatch, '--trials', str(trial), '--out', str(out),
          '--groups', json.dumps(groups), '--fmo-sets', '{}', '-q')
    cmp = pd.read_csv(out / 'compare_A_vs_B.csv')
    w = cmp.pivot(index='cluster', columns='condition', values='mean_pct')
    assert (w['A'] - w['B']).abs().max() >= 30     # truth: 40-point change


def test_shared_cluster_table_gives_each_samples_truth(two_groups, tmp_path,
                                                       stubbed, monkeypatch):
    """compare_shared_clusters.csv says what each shared id is: every
    sample's own share of it, plus its marker medians."""
    trial, groups, fr = two_groups
    out = tmp_path / 'out'
    _main(monkeypatch, '--trials', trial, '--out', str(out), '--groups',
          json.dumps(groups), '--fmo-sets', '{}', '-q')
    t = pd.read_csv(out / 'compare_shared_clusters.csv')
    assert set(zip(t['group'], t['sample'], strict=True)) == {
        ('Ctrl', 'c1'), ('Ctrl', 'c2'), ('Treat', 't1'), ('Treat', 't2')}
    bv = t[t['cluster'] == 0].set_index('sample')
    for nm, f in fr.items():
        assert bv.loc[nm, 'pct_total'] == pytest.approx(100 * f, abs=0.1)
    assert (bv['median_CD901b'] > bv['median_CD902']).all()  # 0 is FL1+


def test_marker_medians_are_written_on_the_linear_scale(two_groups, tmp_path,
                                                        stubbed, monkeypatch):
    """A run logicle-transforms every fluor channel before clustering, and
    <name>_stats.csv / compare_shared_clusters.csv took their median_<marker>
    columns from those coordinates: the FL1+ cluster's CD901b, at linear
    ~20,000, was written as 0.75 and its FL1- side (~150) as 0.25. Both
    files now hold the compensated linear medians the Statistics window
    reports. Truth: FL1 ~ N(20000, 1500) in cluster 0, N(150, 40) in 1."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    _main(monkeypatch, '--trials', trial, '--out', str(out), '--groups',
          json.dumps(groups), '--fmo-sets', '{}', '-q')
    tables = [pd.read_csv(out / 'Ctrl' / 'c1_stats.csv'),
              pd.read_csv(out / 'compare_shared_clusters.csv')]
    for t in tables:
        by = t.groupby('cluster')
        np.testing.assert_allclose(by['median_CD901b'].median().loc[[0, 1]],
                                   [20000, 150], rtol=0.05)
        np.testing.assert_allclose(by['median_CD902'].median().loc[[0, 1]],
                                   [150, 20000], rtol=0.05)
        np.testing.assert_allclose(t['median_CD903'], 800, rtol=0.05)


@pytest.fixture
def three_groups(tmp_path):
    """The pooled FL1+ share is 60 % with Treat and 30 % without it, so
    the size-ranked stub numbers the shared clusters the other way round
    when Treat is left out of the shared clustering."""
    trial = tmp_path / 'trial'
    for i, (nm, f) in enumerate({'c1': .3, 't1': .9, 't2': .9,
                                 'h1': .3}.items()):
        _write(str(trial / f'expt_{nm}.fcs'), 1200, f, 40 + i)
    groups = [{'name': 'Ctrl', 'samples': ['c1'], 'fmo_set': ''},
              {'name': 'Treat', 'samples': ['t1', 't2'], 'fmo_set': ''},
              {'name': 'Third', 'samples': ['h1'], 'fmo_set': ''}]
    return trial, groups


def test_partial_resume_rewrites_no_comparison(three_groups, tmp_path,
                                              stubbed, monkeypatch, capsys):
    """A resume that could not reload one kept sample (t1's file is gone)
    left Treat out, then rewrote compare_shared_clusters.csv and
    compare_Ctrl_vs_Third.csv from a shared clustering without Treat, and
    kept the old compare_Ctrl_vs_Treat.csv: two clusterings in one folder,
    with Ctrl's cluster 0 at 30 % in one file and 70 % in the other."""
    stubbed['mode']['size_ranked'] = True
    trial, groups = three_groups
    out = tmp_path / 'out'
    args = ['--trials', str(trial), '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    before = {p.name: p.read_bytes() for p in out.glob('compare_*')}
    assert len(before) == 7          # 3 pairs x (csv, png) + the table
    os.remove(out / 'Third' / 'h1_stats.csv')      # h1 is re-run ...
    os.remove(trial / 'expt_t1.fcs')               # ... t1 cannot be reloaded
    capsys.readouterr()
    _main(monkeypatch, *args, '--skip-existing')
    assert {p.name: p.read_bytes() for p in out.glob('compare_*')} == before
    said = capsys.readouterr().out
    assert 'No group comparisons were written' in said
    assert ('Treat/t1: kept from an earlier run, but its input is missing'
            in said)


def test_resume_with_nothing_new_recomputes_the_same_comparisons(
        two_groups, tmp_path, stubbed, monkeypatch, capsys):
    """Every sample kept: each pair once printed '<A> or <B> has no
    processed samples'. The comparisons are now recomputed on every run,
    and with nothing changed they come out the same."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    before = {p.name: pd.read_csv(p) for p in out.glob('compare_*.csv')}
    capsys.readouterr()
    _main(monkeypatch, *args, '--skip-existing')
    said = capsys.readouterr().out
    assert 'has no processed samples' not in said
    assert said.count('Reloading ') == 4
    after = {p.name: pd.read_csv(p) for p in out.glob('compare_*.csv')}
    assert sorted(after) == sorted(before)
    for name, frame in before.items():
        pd.testing.assert_frame_equal(after[name], frame)


def test_resume_recomputes_what_a_changed_openflo_would(
        two_groups, tmp_path, stubbed, monkeypatch):
    """A resume with nothing new kept the earlier set whenever its record
    (groups, files, settings) matched, so a set made by older code (here:
    PhenoGraph-style size-ranked ids) survived an upgrade."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    assert _compare_value(out, 'Ctrl_vs_Treat', 'Ctrl') == pytest.approx(65)
    stubbed['mode']['size_ranked'] = True   # pooled FL1+ 45 %: ids flip
    _main(monkeypatch, *args, '--skip-existing')
    assert _compare_value(out, 'Ctrl_vs_Treat', 'Ctrl') == pytest.approx(35)


def test_publishing_leaves_compare_files_openflo_run_did_not_write(
        two_groups, tmp_path, stubbed, monkeypatch):
    """compare_report.csv (the README's own openflo-compare --csv example)
    in the output folder was deleted with the old comparison set."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    out.mkdir()
    mine = {'compare_report.csv': b'flowjo,openflo\n',
            'compare_report.html': b'<html></html>',
            'compare_notes.json': b'{}'}
    for name, data in mine.items():
        (out / name).write_bytes(data)
    (out / 'compare_Old_vs_Treat.csv').write_bytes(b'stale pair\n')
    _main(monkeypatch, '--trials', trial, '--out', str(out), '--groups',
          json.dumps(groups), '--fmo-sets', '{}', '-q')
    assert {n: (out / n).read_bytes() for n in mine} == mine
    assert not (out / 'compare_Old_vs_Treat.csv').exists()


def test_a_case_only_rename_keeps_the_new_files(tmp_path, stubbed,
                                                monkeypatch):
    """Group 'ctrl' renamed 'Ctrl': the old set's compare_ctrl_vs_Treat.*
    counted as stale (case-sensitive match), and removing them deleted the
    just-written compare_Ctrl_vs_Treat.* on Windows / macOS."""
    trial = tmp_path / 'trial'
    _write(str(trial / 'expt_c1.fcs'), 1200, .7, 1)
    _write(str(trial / 'expt_t1.fcs'), 1200, .3, 2)
    out = tmp_path / 'out'
    for name in ('ctrl', 'Ctrl'):
        groups = [{'name': name, 'samples': ['c1']},
                  {'name': 'Treat', 'samples': ['t1']}]
        _main(monkeypatch, '--trials', str(trial), '--out', str(out),
              '--groups', json.dumps(groups), '--fmo-sets', '{}', '-q')
    have = {p.name.casefold() for p in out.glob('compare_*')}
    assert {'compare_ctrl_vs_treat.csv', 'compare_ctrl_vs_treat.png',
            'compare_shared_clusters.csv'} <= have


def _locked(monkeypatch, *paths):
    """Make moving these files fail as a file open in Excel does on Windows
    (PermissionError), for the rest of the test."""
    real = os.replace
    held = {os.path.normcase(os.path.abspath(p)) for p in paths}

    def replace(src, dst):
        if os.path.normcase(os.path.abspath(src)) in held:
            raise PermissionError(13, 'The process cannot access the file '
                                  'because it is being used by another '
                                  'process', src)
        return real(src, dst)
    monkeypatch.setattr(os, 'replace', replace)


def test_a_locked_compare_file_leaves_the_old_set_and_says_where_the_new_is(
        two_groups, tmp_path, stubbed, monkeypatch, capsys):
    """A compare file open in Excel on a re-run: PermissionError escaped
    run(), with the table already new, the pair old, and the rest of the
    new set stuck in the staging folder. The locked file is the one moved
    LAST (compare_shared_clusters.csv), so the pair's .csv and .png are
    already moved aside when the move fails: without the rollback putting
    them back, they are missing from the folder and this test fails."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    assert _main(monkeypatch, *args) == 0
    before = {p.name: p.read_bytes() for p in out.glob('compare_*')}
    assert sorted(before)[-1] == 'compare_shared_clusters.csv'  # moved last
    _write(os.path.join(trial, 'expt_t1.fcs'), 1200, .9, 99)    # new answer
    _locked(monkeypatch, out / 'compare_shared_clusters.csv')
    capsys.readouterr()
    assert _main(monkeypatch, *args) == 1
    assert {p.name: p.read_bytes() for p in out.glob('compare_*')} == before
    said = capsys.readouterr().out
    where = re.search(r'This run\'s comparisons are in (.+?)\. ', said)
    assert where and 'compare_shared_clusters.csv cannot be moved' in said
    new = pd.read_csv(os.path.join(where.group(1), 'compare_Ctrl_vs_Treat.csv'))
    assert float(new.set_index(['condition', 'cluster']).loc[
        ('Treat', 0), 'mean_pct']) == pytest.approx(55, abs=.1)


def test_staging_left_by_a_locked_run_is_cleared_and_never_shown(
        two_groups, tmp_path, stubbed, monkeypatch):
    """A locked run keeps its new set in a .compare-staging-* folder. Such
    folders were never removed, and --show-plots (rglob) re-opened their
    stale compare PNGs as if they were the run's."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    real = os.replace
    _locked(monkeypatch, out / 'compare_Ctrl_vs_Treat.csv')
    assert _main(monkeypatch, *args) == 1
    left = list(out.glob('.compare-staging-*'))
    assert len(left) == 1
    shown = []
    monkeypatch.setattr(cli.plt, 'imread', lambda p: shown.append(p) or
                        np.zeros((2, 2, 3)))
    cli._show_saved_group_plots(str(out))
    assert shown and not [p for p in shown if '.compare-staging-' in p]
    monkeypatch.setattr(os, 'replace', real)               # unlocked
    group_like = out / '.compare-staging-abcdefgh'          # not ours
    group_like.mkdir()
    (group_like / 'keep.csv').write_text('x')
    assert _main(monkeypatch, *args) == 0
    assert [p.name for p in out.glob('.compare-staging-*')] == [
        '.compare-staging-abcdefgh']


@pytest.mark.skipif(sys.platform != 'win32',
                    reason='an open file blocks a rename only on Windows')
def test_a_really_open_compare_file_is_reported_not_a_crash(
        two_groups, tmp_path, stubbed, monkeypatch, capsys):
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    before = (out / 'compare_shared_clusters.csv').read_bytes()
    with open(out / 'compare_shared_clusters.csv', 'rb'):
        assert _main(monkeypatch, *args) == 1
    assert (out / 'compare_shared_clusters.csv').read_bytes() == before
    assert 'compare_shared_clusters.csv cannot be moved' in \
        capsys.readouterr().out


def test_a_locked_file_in_one_trial_does_not_stop_the_next(
        tmp_path, stubbed, monkeypatch):
    """Independent mode, two trial folders: the first trial's locked file
    crashed the run before the second trial was analysed."""
    for tr in ('trA', 'trB'):
        for i, (nm, f) in enumerate({'c1': .7, 't1': .3}.items()):
            _write(str(tmp_path / tr / f'expt_{nm}.fcs'), 1200, f, 50 + i)
    groups = [{'name': 'Ctrl', 'samples': ['c1']},
              {'name': 'Treat', 'samples': ['t1']}]
    out = tmp_path / 'out'
    args = ['--trials', f"{tmp_path / 'trA'},{tmp_path / 'trB'}", '--out',
            str(out), '--groups', json.dumps(groups), '--fmo-sets', '{}',
            '-q']
    _main(monkeypatch, *args)
    os.remove(out / 'trB' / 'compare_Ctrl_vs_Treat.csv')
    _locked(monkeypatch, out / 'trA' / 'compare_Ctrl_vs_Treat.png')
    assert _main(monkeypatch, *args) == 1
    assert (out / 'trB' / 'compare_Ctrl_vs_Treat.csv').exists()


def test_a_group_named_like_the_staging_folder_keeps_its_outputs(
        tmp_path, stubbed, monkeypatch):
    """A group named '.compare_staging' had its output folder emptied by
    the publish step, which rmtree'd the staging folder of that name."""
    trial = tmp_path / 'trial'
    _write(str(trial / 'expt_c1.fcs'), 1200, .7, 1)
    _write(str(trial / 'expt_t1.fcs'), 1200, .3, 2)
    groups = [{'name': '.compare_staging', 'samples': ['c1']},
              {'name': 'Treat', 'samples': ['t1']}]
    out = tmp_path / 'out'
    _main(monkeypatch, '--trials', str(trial), '--out', str(out), '--groups',
          json.dumps(groups), '--fmo-sets', '{}', '-q')
    assert (out / '.compare_staging' / 'c1_stats.csv').exists()
    assert (out / 'compare_.compare_staging_vs_Treat.csv').exists()


def test_a_run_killed_while_publishing_leaves_no_stale_pair_behind(
        three_groups, tmp_path, stubbed, monkeypatch):
    """Third leaves the groups; the run is killed while moving its set in,
    and run again. The old manifest, the only list of what to remove, was
    already gone, so compare_Ctrl_vs_Third.* and compare_Treat_vs_Third.*
    from the old clustering stayed beside the new set for good."""
    trial, groups = three_groups
    out = tmp_path / 'out'
    base = ['--trials', str(trial), '--out', str(out), '--fmo-sets', '{}',
            '-q']
    _main(monkeypatch, *base, '--groups', json.dumps(groups))
    two = json.dumps(groups[:2])
    real = os.replace

    def killed_once(src, dst):
        if os.path.basename(dst).startswith('compare_'):
            monkeypatch.setattr(os, 'replace', real)
            raise _Killed
        return real(src, dst)
    monkeypatch.setattr(os, 'replace', killed_once)
    with pytest.raises(_Killed):
        _main(monkeypatch, *base, '--groups', two)
    _main(monkeypatch, *base, '--groups', two)
    assert sorted(p.name for p in out.glob('compare_*')) == [
        'compare_Ctrl_vs_Treat.csv', 'compare_Ctrl_vs_Treat.png',
        'compare_shared_clusters.csv']


class _Killed(BaseException):
    """Stands in for the process being killed (watchdog, Ctrl-C, power)."""


def _compare_value(out, pair, condition, cluster=0):
    cmp = pd.read_csv(out / f'compare_{pair}.csv').set_index(
        ['condition', 'cluster'])
    return float(cmp.loc[(condition, cluster), 'mean_pct'])


def test_resume_redoes_comparisons_whose_input_changed(two_groups, tmp_path,
                                                       stubbed, monkeypatch):
    """t1.fcs is replaced (30 % -> 90 % FL1+) and a full re-run is killed
    after the per-sample outputs, before the comparisons. The resume found
    every sample done, nothing new, compare files present, and kept the
    comparison from the OLD t1: Treat 25 % beside a t1_stats.csv at 90 %."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    assert _compare_value(out, 'Ctrl_vs_Treat', 'Treat') == pytest.approx(25)
    _write(os.path.join(trial, 'expt_t1.fcs'), 1200, .9, 99)
    real = cli._write_group_outputs

    def killed(*a, **k):
        raise _Killed
    monkeypatch.setattr(cli, '_write_group_outputs', killed)
    with pytest.raises(_Killed):
        _main(monkeypatch, *args)
    assert _bv(out / 'Treat' / 't1_stats.csv') == pytest.approx(90, abs=.1)
    monkeypatch.setattr(cli, '_write_group_outputs', real)
    _main(monkeypatch, *args, '--skip-existing')
    assert _compare_value(out, 'Ctrl_vs_Treat', 'Treat') == pytest.approx(
        55, abs=.1)                                      # (90 + 20) / 2


def test_a_kill_inside_the_pair_loop_leaves_no_mixed_set(
        three_groups, tmp_path, stubbed, monkeypatch):
    """Killed after the first pair of a re-run: compare_shared_clusters.csv
    and compare_Ctrl_vs_Treat.csv came from the new shared clustering and
    the other pairs from the old one, and every later resume kept that mix
    (all files existed). Now nothing in place changes until the whole set
    is written; the resume sees the inputs changed and redoes it."""
    stubbed['mode']['size_ranked'] = True
    trial, groups = three_groups
    out = tmp_path / 'out'
    args = ['--trials', str(trial), '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    before = {p.name: p.read_bytes() for p in out.glob('compare_*')}
    for i, nm in enumerate(('t1', 't2')):          # pooled FL1+ 60 -> 20 %
        _write(str(trial / f'expt_{nm}.fcs'), 1200, .1, 70 + i)
    real = cli._compare_group_pair
    calls = []

    def dies_on_the_second(*a, **k):
        calls.append(a[:2])
        if len(calls) == 2:
            raise _Killed
        return real(*a, **k)
    monkeypatch.setattr(cli, '_compare_group_pair', dies_on_the_second)
    with pytest.raises(_Killed):
        _main(monkeypatch, *args)
    assert {p.name: p.read_bytes() for p in out.glob('compare_*')} == before
    monkeypatch.setattr(cli, '_compare_group_pair', real)
    _main(monkeypatch, *args, '--skip-existing')
    ctrl = {_compare_value(out, p, 'Ctrl')
            for p in ('Ctrl_vs_Treat', 'Ctrl_vs_Third')}
    assert len(ctrl) == 1                          # one clustering everywhere
    t = pd.read_csv(out / 'compare_shared_clusters.csv').set_index(
        ['sample', 'cluster'])
    assert ctrl.pop() == pytest.approx(float(t.loc[('c1', 0), 'pct_total']))


def test_resume_after_a_sample_left_its_group_redoes_the_comparisons(
        two_groups, tmp_path, stubbed, monkeypatch):
    """c2 is taken out of Ctrl and the run resumed: every listed sample was
    kept, so the earlier comparisons were left, still counting c2."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    base = ['--trials', trial, '--out', str(out), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *base, '--groups', json.dumps(groups))
    fewer = [dict(groups[0], samples=['c1']), groups[1]]
    _main(monkeypatch, *base, '--groups', json.dumps(fewer),
          '--skip-existing')
    t = pd.read_csv(out / 'compare_shared_clusters.csv')
    assert set(t['sample']) == {'c1', 't1', 't2'}
    cmp = pd.read_csv(out / 'compare_Ctrl_vs_Treat.csv')
    ctrl = cmp[(cmp.cluster == 0) & (cmp.condition == 'Ctrl')].iloc[0]
    assert (ctrl['mean_pct'], ctrl['n_samples']) == (pytest.approx(70, abs=.1),
                                                     1)        # c1 alone


def test_resume_writes_the_comparisons_an_earlier_run_did_not(
        two_groups, tmp_path, stubbed, monkeypatch, capsys):
    """A run stopped after its per-sample outputs, before the comparisons,
    was never completed by --skip-existing: with nothing new to process it
    wrote no group output. It now reloads the kept samples, one progress
    line each, and writes the comparisons the full run wrote."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    full = pd.read_csv(out / 'compare_Ctrl_vs_Treat.csv')
    for p in out.glob('compare_*'):
        os.remove(p)
    capsys.readouterr()
    _main(monkeypatch, *args, '--skip-existing')
    pd.testing.assert_frame_equal(
        pd.read_csv(out / 'compare_Ctrl_vs_Treat.csv'), full)
    said = capsys.readouterr().out
    for nm in ('Ctrl/c1', 'Ctrl/c2', 'Treat/t1', 'Treat/t2'):
        assert f'Reloading {nm} (kept from an earlier run)' in said


@pytest.mark.parametrize('how', ['fails', 'has no rows'])
def test_a_pair_with_no_new_file_leaves_no_old_one(three_groups, tmp_path,
                                                   stubbed, monkeypatch,
                                                   capsys, how):
    """Ctrl vs Third fails (or has no rows) on a re-run: the other pairs
    and compare_shared_clusters.csv were rewritten from the new clustering,
    and the old compare_Ctrl_vs_Third.csv / .png stayed beside them."""
    from openflo.pipeline import FlowExperiment
    trial, groups = three_groups
    out = tmp_path / 'out'
    args = ['--trials', str(trial), '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    assert (out / 'compare_Ctrl_vs_Third.csv').exists()
    real = FlowExperiment.compare_conditions

    def flaky(self, groupA, groupB, label_a='A', label_b='B', plot=True):
        if (label_a, label_b) == ('Ctrl', 'Third'):
            if how == 'fails':
                raise RuntimeError('boom')
            return pd.DataFrame()
        return real(self, groupA, groupB, label_a=label_a, label_b=label_b,
                    plot=plot)
    monkeypatch.setattr(FlowExperiment, 'compare_conditions', flaky)
    capsys.readouterr()
    _main(monkeypatch, *args)
    assert not list(out.glob('compare_Ctrl_vs_Third.*'))
    assert (out / 'compare_Ctrl_vs_Treat.csv').exists()
    assert re.search(r'compare_Ctrl_vs_Third\.csv.*removed',
                     capsys.readouterr().out)


def test_a_failed_sample_means_no_comparison(two_groups, tmp_path, stubbed,
                                             monkeypatch, capsys):
    """Also in a fresh run: Treat's comparison came from t1 alone when t2
    failed, with n_samples 1 the only sign. No comparison is written, the
    run names the missing sample, and it exits 1: it did not produce what
    was asked (it exited 0)."""
    trial, groups, _ = two_groups
    with open(os.path.join(trial, 'expt_t2.fcs'), 'wb') as fh:
        fh.write(b'not an FCS file')
    out = tmp_path / 'out'
    assert _main(monkeypatch, '--trials', trial, '--out', str(out),
                 '--groups', json.dumps(groups), '--fmo-sets', '{}',
                 '-q') == 1
    assert (out / 'Treat' / 't1_stats.csv').exists()
    assert not list(out.glob('compare_*'))
    said = capsys.readouterr().out
    assert re.search(r'No group comparisons were written.*\n'
                     r'.*Treat/t2: failed in this run \(FcsParseError', said)


def test_a_missing_file_is_named_as_not_found(two_groups, tmp_path, stubbed,
                                              monkeypatch, capsys):
    """The user's rule: any failed or missing sample means no comparison,
    and the run says plainly which sample and why. 'not processed in this
    run: it failed or its file was not found' did not say which."""
    trial, groups, _ = two_groups
    groups = [dict(groups[0], samples=['c1', 'c2', 'c9']), groups[1]]
    out = tmp_path / 'out'
    _main(monkeypatch, '--trials', trial, '--out', str(out), '--groups',
          json.dumps(groups), '--fmo-sets', '{}', '-q')
    assert not list(out.glob('compare_*'))
    said = capsys.readouterr().out
    assert 'No group comparisons were written' in said
    assert 'Ctrl/c9: its FCS file was not found' in said


@pytest.mark.parametrize('spec, said', [
    # two groups named Ctrl: their samples merged into one side of every
    # comparison, the second's missing c9 went unreported, and the run also
    # wrote compare_Ctrl_vs_Ctrl / compare_Treat_vs_Ctrl
    ([{'name': 'Ctrl', 'samples': ['c1', 'c2']},
      {'name': 'Treat', 'samples': ['t1', 't2']},
      {'name': 'Ctrl', 'samples': ['c9']}],
     "2 groups are named 'Ctrl'"),
    # 'Ctrl' and 'ctrl' share one output folder on Windows / macOS
    ([{'name': 'Ctrl', 'samples': ['c1']}, {'name': 'ctrl', 'samples': ['c2']}],
     "'Ctrl' and 'ctrl' would share the output folder"),
    # c1 twice: processed twice, counted twice in the comparison
    ([{'name': 'Ctrl', 'samples': ['c1', 'c1', 'c2']},
      {'name': 'Treat', 'samples': ['t1']}],
     "'Ctrl' lists c1 more than once"),
    # 'c1' and 'C1' (or 'c1' and 'expt_c1') are one file: the name check
    # passed, and c1 was processed and counted twice (Ctrl 66.7 %, not 65)
    ([{'name': 'Ctrl', 'samples': ['c1', 'C1', 'c2']},
      {'name': 'Treat', 'samples': ['t1']}],
     "'Ctrl' lists c1 and C1, which are one file"),
    ([{'name': 'Ctrl', 'samples': ['c1', 'expt_c1', 'c2']},
      {'name': 'Treat', 'samples': ['t1']}],
     "'Ctrl' lists c1 and expt_c1, which are one file"),
    # A vs 'B vs C' and 'A vs B' vs C both make compare_A_vs_B_vs_C: one
    # pair's file overwrote the other's (and, staged, crashed the move)
    ([{'name': 'A', 'samples': ['c1']}, {'name': 'B vs C', 'samples': ['c2']},
      {'name': 'A vs B', 'samples': ['t1']}, {'name': 'C', 'samples': ['t2']}],
     "'A' vs 'B vs C' and 'A vs B' vs 'C' would both write "
     "compare_A_vs_B_vs_C.csv"),
])
def test_ambiguous_groups_are_an_error(two_groups, tmp_path, stubbed,
                                       monkeypatch, capsys, spec, said):
    trial, _, _ = two_groups
    out = tmp_path / 'out'
    assert _main(monkeypatch, '--trials', trial, '--out', str(out),
                 '--groups', json.dumps(spec), '--fmo-sets', '{}', '-q') == 2
    assert said in capsys.readouterr().out
    assert not out.exists() or not any(out.iterdir())


def test_a_missing_sample_stops_the_comparisons_before_any_reload(
        two_groups, tmp_path, stubbed, monkeypatch, capsys):
    """t2 fails on resume, so no comparison can be written, yet every kept
    sample was reloaded first: on two dozen large files, minutes of work
    for nothing."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    os.remove(out / 'Treat' / 't2_stats.csv')
    with open(os.path.join(trial, 'expt_t2.fcs'), 'wb') as fh:
        fh.write(b'not an FCS file')
    loaded = []
    real = cli._load_task_sample

    def spy(task):
        loaded.append(task['name'])
        return real(task)
    monkeypatch.setattr(cli, '_load_task_sample', spy)
    capsys.readouterr()
    _main(monkeypatch, *args, '--skip-existing')
    assert loaded == ['t2']            # its own (failing) processing only
    assert 'Treat/t2: failed in this run' in capsys.readouterr().out


@pytest.fixture
def two_trials(tmp_path):
    """Concatenate mode: every sample in trA and trB; t1 is 30 % FL1+ in
    trA and 90 % in trB."""
    fr = {'trA': {'c1': .7, 't1': .3}, 'trB': {'c1': .7, 't1': .9}}
    for tr, samples in fr.items():
        for i, (nm, f) in enumerate(samples.items()):
            _write(str(tmp_path / tr / f'expt_{nm}.fcs'), 1200, f, 60 + i)
    groups = [{'name': 'Ctrl', 'samples': ['c1']},
              {'name': 'Treat', 'samples': ['t1']}]
    return ['--trials', f"{tmp_path / 'trA'},{tmp_path / 'trB'}",
            '--batch-mode', 'concatenate', '--groups', json.dumps(groups),
            '--fmo-sets', '{}', '-q']


def test_a_kept_sample_missing_a_trials_file_means_no_comparison(
        two_trials, tmp_path, stubbed, monkeypatch, capsys):
    """trB/expt_t1.fcs is deleted and the run resumed: t1 was reloaded from
    trA alone, so Treat fell from 60 % to 30 % while t1_stats.csv still
    held both trials, and nothing named the missing file."""
    out = tmp_path / 'out'
    _main(monkeypatch, *two_trials, '--out', str(out))
    before = {p.name: p.read_bytes() for p in out.glob('compare_*')}
    os.remove(tmp_path / 'trB' / 'expt_t1.fcs')
    capsys.readouterr()
    _main(monkeypatch, *two_trials, '--out', str(out), '--skip-existing')
    assert {p.name: p.read_bytes() for p in out.glob('compare_*')} == before
    said = capsys.readouterr().out
    assert 'No group comparisons were written' in said
    assert re.search(r'Treat/t1: kept from an earlier run, but its input is '
                     r'missing \(.*trB', said)


def test_a_sample_missing_from_one_trial_means_no_comparison(
        two_trials, tmp_path, stubbed, monkeypatch, capsys):
    """The same data on a fresh run: t1 is processed from trA alone (as
    before, with a warning), but a comparison from it is not the one the
    groups describe, so none is written."""
    os.remove(tmp_path / 'trB' / 'expt_t1.fcs')
    out = tmp_path / 'out'
    _main(monkeypatch, *two_trials, '--out', str(out))
    assert (out / 'Treat' / 't1_stats.csv').exists()
    assert not list(out.glob('compare_*'))
    assert re.search(r'Treat/t1: its input is missing \(.*trB',
                     capsys.readouterr().out)


def test_a_kept_sample_whose_file_is_gone_is_named_before_any_reload(
        two_groups, tmp_path, stubbed, monkeypatch, capsys):
    """expt_t2.fcs deleted, t2_stats.csv kept: c1, c2 and t1 were reloaded
    before t2 was found unloadable and nothing was written."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    args = ['--trials', trial, '--out', str(out), '--groups',
            json.dumps(groups), '--fmo-sets', '{}', '-q']
    _main(monkeypatch, *args)
    os.remove(os.path.join(trial, 'expt_t2.fcs'))
    loaded = []
    real = cli._load_task_sample

    def spy(task):
        loaded.append(task['name'])
        return real(task)
    monkeypatch.setattr(cli, '_load_task_sample', spy)
    capsys.readouterr()
    _main(monkeypatch, *args, '--skip-existing')
    assert loaded == []
    assert 'Treat/t2: kept from an earlier run, but its input is missing' in \
        capsys.readouterr().out


def test_by_day_folders_with_same_named_parents_get_distinct_groups(
        tmp_path, stubbed, monkeypatch):
    """X/exp/day3 and Y/exp/day3 both became 'Day 3 (exp)', and the run
    then refused the clash as if the user had typed it in --groups."""
    _write(str(tmp_path / 'X' / 'exp' / 'day3' / 's1.fcs'), 1200, .7, 1)
    _write(str(tmp_path / 'Y' / 'exp' / 'day3' / 's1.fcs'), 1200, .3, 2)
    out = tmp_path / 'out'
    assert _main(monkeypatch, '--trials',
                 f"{tmp_path / 'X' / 'exp'},{tmp_path / 'Y' / 'exp'}",
                 '--out', str(out), '--fmo-sets', '{}', '-q') == 0
    assert _bv(out / 'Day_3_(X_exp)' / 's1_stats.csv') == pytest.approx(70)
    assert _bv(out / 'Day_3_(Y_exp)' / 's1_stats.csv') == pytest.approx(30)


@pytest.mark.parametrize('folders, made', [
    # 'ctrl' and 'Ctrl' share one output folder on Windows / macOS; the
    # names differed only in case, so they were not disambiguated, and the
    # run refused them
    (('A/ctrl', 'B/Ctrl'), ('ctrl_(A)', 'Ctrl_(B)')),
    # one parent: walking up the parents never told day3 from day_3 (and
    # day_3, which already reads 'Day 3' as a folder, keeps that name)
    (('exp/day3', 'exp/day_3'), ('Day_3_(day3)', 'Day_3')),
])
def test_by_day_names_differ_by_case_and_by_the_folders_own_name(
        tmp_path, stubbed, monkeypatch, folders, made):
    for i, f in enumerate(folders):
        _write(str(tmp_path / 'root' / f / 's1.fcs'), 1200, (.7, .3)[i], i)
    out = tmp_path / 'out'
    assert _main(monkeypatch, '--trials', str(tmp_path / 'root'), '--out',
                 str(out), '--fmo-sets', '{}', '-q') == 0
    for folder, frac in zip(made, (70, 30), strict=True):
        assert _bv(out / folder / 's1_stats.csv') == pytest.approx(frac)


def test_a_bad_group_spec_is_refused_before_any_fcs_is_read(
        tmp_path, stubbed, monkeypatch, capsys):
    """--panel read a sample's FCS before the group check ran, and the
    message blamed --groups for a --samples list."""
    trial = tmp_path / 'trial'
    _write(str(trial / 'expt_m1.fcs'), 1200, .7, 1)
    opened = []
    real = FlowSample.__init__

    def spy(self, *a, **k):
        opened.append(a)
        real(self, *a, **k)
    monkeypatch.setattr(FlowSample, '__init__', spy)
    assert _main(monkeypatch, '--trials', str(trial), '--out',
                 str(tmp_path / 'out'), '--samples', 'm1,m1', '--panel',
                 str(tmp_path / 'panel.xlsx'), '-q') == 2
    assert opened == []
    assert "[!] --samples: group 'Group A' lists m1 more than once" in \
        capsys.readouterr().out


def test_run_refuses_ambiguous_groups(two_groups, tmp_path, stubbed):
    """The programmatic entry point (cli.run) raises rather than return 2."""
    trial, groups, _ = two_groups
    with pytest.raises(ValueError, match="2 groups are named 'Ctrl'"):
        cli.run(trial, str(tmp_path / 'o'), groups + [groups[0]],
                fmo_sets={})
    assert not (tmp_path / 'o').exists()


def test_failed_shared_clustering_writes_no_comparison(
        two_groups, tmp_path, stubbed, monkeypatch, capsys):
    """Falling back to each sample's own cluster ids would compare unrelated
    populations, so a failed shared clustering writes no comparison. The
    run did not produce what was asked, so it exits 1 (it exited 0), and it
    says any compare files present are an earlier run's."""
    trial, groups, _ = two_groups
    per_sample = FlowSample.cluster            # the fixture's oracle stub

    def fail_shared(self, **kw):
        if self.name == 'shared clustering':
            raise RuntimeError('boom')
        return per_sample(self, **kw)
    monkeypatch.setattr(FlowSample, 'cluster', fail_shared)
    out = tmp_path / 'out'
    assert _main(monkeypatch, '--trials', trial, '--out', str(out),
                 '--groups', json.dumps(groups), '--fmo-sets', '{}',
                 '-q') == 1
    assert (out / 'Ctrl' / 'c1_stats.csv').exists()
    assert not list(out.glob('compare_*'))
    said = ' '.join(capsys.readouterr().out.split())
    assert 'Shared clustering failed (RuntimeError: boom)' in said
    assert f'Any compare_* files in {out} are from an earlier run' in said


def test_shared_clustering_matches_channels_by_antibody_label(
        tmp_path, stubbed, monkeypatch, capsys):
    """Treat's panel moved CD901b to FL2 (and CD902 to FL1). The shared
    clustering matched channels by detector, so one of its columns held
    CD901b for Ctrl and CD902 for Treat: Treat's 20 % CD901b+ events were
    counted in the CD902+ cluster and its CD901b+ cluster read 80 %."""
    trial = tmp_path / 'trial'
    _write(str(trial / 'expt_c1.fcs'), 1200, .7, 1)       # 70 % CD901b+
    # FL1 now carries CD902: 80 % FL1-high = 20 % CD901b+ (FL2-high)
    _write(str(trial / 'expt_t1.fcs'), 1200, .8, 2,
           mk=['', '', 'CD902', 'CD901b', 'CD903'])
    groups = [{'name': 'Ctrl', 'samples': ['c1'], 'fmo_set': ''},
              {'name': 'Treat', 'samples': ['t1'], 'fmo_set': ''}]
    out = tmp_path / 'out'
    _main(monkeypatch, '--trials', str(trial), '--out', str(out),
          '--groups', json.dumps(groups), '--fmo-sets', '{}', '-q')
    t = pd.read_csv(out / 'compare_shared_clusters.csv')
    cd901b = t[t['cluster'] == 0].set_index('sample')  # stub: 0 = CD901b > CD902
    assert cd901b.loc['c1', 'pct_total'] == pytest.approx(70, abs=0.1)
    assert cd901b.loc['t1', 'pct_total'] == pytest.approx(20, abs=0.1)
    assert (cd901b['median_CD901b'] > cd901b['median_CD902']).all()
    said = capsys.readouterr().out
    assert re.search(r'CD901b: FL1-A in Ctrl/c1; FL2-A in Treat/t1', said)


def test_shared_clustering_names_every_channel_it_leaves_out(
        tmp_path, stubbed, monkeypatch, capsys):
    """A channel missing from one sample was dropped from the shared
    clustering without a word."""
    trial = tmp_path / 'trial'
    _write(str(trial / 'expt_c1.fcs'), 1200, .7, 1)
    _write(str(trial / 'expt_t1.fcs'), 1200, .3, 2, drop=('FL3-A',))
    groups = [{'name': 'Ctrl', 'samples': ['c1'], 'fmo_set': ''},
              {'name': 'Treat', 'samples': ['t1'], 'fmo_set': ''}]
    _main(monkeypatch, '--trials', str(trial), '--out', str(tmp_path / 'o'),
          '--groups', json.dumps(groups), '--fmo-sets', '{}', '-q')
    assert re.search(r'CD903 \(FL3-A\): no matching channel in Treat/t1',
                     capsys.readouterr().out)


def test_threshold_gate_changes_some_output(two_groups, tmp_path, stubbed,
                                            monkeypatch):
    """FMO / --gates threshold gates only added in-memory <ch>_pos columns;
    no file openflo-run wrote changed."""
    trial, groups, _ = two_groups
    outs = []
    for i, gates in enumerate(([], [{'kind': 'threshold',
                                     'channel': 'FL1-A', 'value': 3.0}])):
        o = tmp_path / f'o{i}'
        _main(monkeypatch, '--trials', trial, '--out', str(o), '--groups',
              json.dumps(groups), '--fmo-sets', '{}', '-q',
              *(['--gates', json.dumps(gates)] if gates else []))
        outs.append({p.relative_to(o).as_posix(): p.read_bytes()
                     for p in o.rglob('*.csv')})
    assert outs[0] != outs[1]


def test_fmo_thresholds_reach_the_stats_as_percent_positive(tmp_path, stubbed,
                                                            monkeypatch):
    """Same controls as the percentile test above: at --fmo-percentile 90,
    10 % of each control's negatives sit above its cut. Cluster 0 (FL1+,
    FL2-neg) is therefore ~100 % CD901b+ and ~10 % CD902+; cluster 1 the
    mirror image."""
    trial = tmp_path / 'trial'
    _write(str(trial / 'expt_s1.fcs'), 2000, .7, 5)
    _write(str(trial / 'expt_fmo_bv.fcs'), 4000, 0.0, 6)     # all FL1-neg
    _write(str(trial / 'expt_unstained.fcs'), 4000, 1.0, 7)  # all FL2-neg
    groups = [{'name': 'G', 'samples': ['s1'], 'fmo_set': 'std'}]
    fmo = {'std': {'FL1-A': 'fmo_bv', 'FL2-A': 'fmo_fl2_missing'}}
    out = tmp_path / 'o'
    _main(monkeypatch, '--trials', str(trial), '--out', str(out),
          '--groups', json.dumps(groups), '--fmo-sets', json.dumps(fmo),
          '--fmo-percentile', '90', '-q')
    st = pd.read_csv(out / 'G' / 's1_stats.csv').set_index('cluster')
    # binomial sd of a 10 % rate: 0.8 points over 1400 events, 1.2 over 600
    assert st.loc[0, 'pct_pos_CD901b'] == pytest.approx(100, abs=0.5)
    assert st.loc[0, 'pct_pos_CD902'] == pytest.approx(10, abs=3)
    assert st.loc[1, 'pct_pos_CD902'] == pytest.approx(100, abs=0.5)
    assert st.loc[1, 'pct_pos_CD901b'] == pytest.approx(10, abs=4)


def test_ungated_run_stats_have_no_positivity_columns(two_groups, tmp_path,
                                                      stubbed, monkeypatch):
    trial, groups, _ = two_groups
    out = tmp_path / 'o'
    _main(monkeypatch, '--trials', trial, '--out', str(out), '--groups',
          json.dumps(groups), '--fmo-sets', '{}', '-q')
    cols = list(pd.read_csv(out / 'Ctrl' / 'c1_stats.csv').columns)
    assert not [c for c in cols if c.startswith('pct_pos_')]


def test_step_counter_never_exceeds_total(two_groups, tmp_path, stubbed,
                                          monkeypatch, capsys):
    """The step total counted a UMAP only for groups with >= 2 samples, but
    every non-empty group runs one, so [STEP 4/3] was printed."""
    trial, _, _ = two_groups
    groups = [{'name': 'G1', 'samples': ['c1'], 'fmo_set': ''},
              {'name': 'G2', 'samples': ['t1'], 'fmo_set': ''}]
    _main(monkeypatch, '--trials', trial, '--out', str(tmp_path / 'o'),
          '--groups', json.dumps(groups), '--fmo-sets', '{}', '-q')
    for n, tot in re.findall(r'\[STEP (\d+)/(\d+)\]', capsys.readouterr().out):
        assert int(n) <= int(tot)


def test_cluster_receives_the_cli_flags(two_groups, tmp_path, stubbed,
                                        monkeypatch):
    """--k / --seed / --max-events / --no-reproducible reach FlowSample.cluster
    (the run() path; test_cli_cluster.py only calls the worker directly)."""
    trial, groups, _ = two_groups
    seen = []
    orig = FlowSample.cluster

    def spy(self, **kw):
        seen.append(kw)
        return orig(self, **kw)
    monkeypatch.setattr(FlowSample, 'cluster', spy)
    base = ['--trials', trial, '--groups', json.dumps(groups),
            '--fmo-sets', '{}', '-q', '--k', '17', '--seed', '7']
    _main(monkeypatch, *base, '--out', str(tmp_path / 'a'))
    _main(monkeypatch, *base, '--out', str(tmp_path / 'b'),
          '--max-events', '900', '--no-reproducible')
    # Per run: 4 samples, then 1 shared clustering for the group comparison,
    # which takes the same flags.
    assert len(seen) == 10
    first, shared_a, second = seen[0], seen[4], seen[-1]
    assert (first['k'], first['random_state'], first['max_events'],
            first['reproducible']) == (17, 7, None, True)
    assert (shared_a['k'], shared_a['random_state'],
            shared_a['reproducible']) == (17, 7, True)
    assert (second['max_events'], second['reproducible']) == (900, False)


@pytest.mark.parametrize('free_gb, shared_n_jobs', [(64.0, -1), (1.0, 1)])
def test_shared_phenograph_uses_every_core_when_ram_allows(
        two_groups, tmp_path, stubbed, monkeypatch, free_gb, shared_n_jobs):
    """--workers > 1 gives each per-sample PhenoGraph n_jobs=1, since they
    run side by side, and the shared clustering inherited it although it
    runs alone after them: 356.6 s on a 200k pool against 87.5 s with every
    core (identical labels). Every core costs RAM (peak 2.0 -> 5.7 GB), so
    it keeps n_jobs=1 when free RAM is short."""
    import types

    import psutil
    trial, groups, _ = two_groups
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: types.SimpleNamespace(
        available=free_gb * 1024 ** 3, total=64 * 1024 ** 3, percent=0.0))
    serial = cli._run_sample_tasks       # in-process, so the stubs apply
    monkeypatch.setattr(cli, '_run_sample_tasks',
                        lambda tasks, workers, adm, report, *rest:
                        serial(tasks, 1, adm, report, *rest))
    seen = []
    orig = FlowSample.cluster

    def spy(self, **kw):
        seen.append((self.name, kw['n_jobs']))
        return orig(self, **kw)
    monkeypatch.setattr(FlowSample, 'cluster', spy)
    _main(monkeypatch, '--trials', trial, '--out', str(tmp_path / 'o'),
          '--groups', json.dumps(groups), '--fmo-sets', '{}', '-q',
          '--workers', '2')
    assert [nj for nm, nj in seen if nm != 'shared clustering'] == [1] * 4
    assert [nj for nm, nj in seen if nm == 'shared clustering'] == [
        shared_n_jobs]


def _help(monkeypatch, capsys, flag):
    """The --help text of one option, whitespace collapsed."""
    with pytest.raises(SystemExit):
        _main(monkeypatch, '--help')
    lines = capsys.readouterr().out.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.lstrip().startswith(flag) and ln.startswith('  -'))
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith('  -')), len(lines))
    return ' '.join(' '.join(lines[start:end]).split())


def test_help_says_what_max_events_and_skip_existing_do(monkeypatch, capsys):
    """--max-events said 'per sample', but the group comparison's shared
    clustering uses it as the cap for the whole trial; --skip-existing had
    no help at all."""
    mev = _help(monkeypatch, capsys, '--max-events')
    assert 'per sample' in mev and 'IN TOTAL' in mev
    skip = _help(monkeypatch, capsys, '--skip-existing')
    assert 'reloaded' in skip and 'every sample of every group' in skip


def test_compare_csv_says_its_ids_are_the_shared_clusterings(
        two_groups, tmp_path, stubbed, monkeypatch):
    """compare_<A>_vs_<B>.csv's cluster ids are the shared clustering's,
    not those of the <name>_stats.csv beside it, and nothing in the file
    said so."""
    trial, groups, _ = two_groups
    out = tmp_path / 'out'
    _main(monkeypatch, '--trials', trial, '--out', str(out), '--groups',
          json.dumps(groups), '--fmo-sets', '{}', '-q')
    cmp = pd.read_csv(out / 'compare_Ctrl_vs_Treat.csv')
    assert list(cmp.columns[:6]) == ['condition', 'cluster', 'population',
                                     'mean_pct', 'sd_pct', 'n_samples']
    note, = set(cmp['cluster_ids'])
    assert 'compare_shared_clusters.csv' in note
    assert 'NOT the <name>_stats.csv' in note


def test_readme_documents_the_run_outputs():
    readme = open(os.path.join(os.path.dirname(__file__), '..', 'README.md'),
                  encoding='utf-8').read()
    text = ' '.join(readme.split())
    for must in ('compare_shared_clusters.csv', 'cluster_ids',
                 'pct_pos_<marker>', 'IN TOTAL', "*max ev*",
                 # a kill while the files move leaves a partial set; say so
                 'killed while the files are moved'):
        assert must in text, must
    # The README describes what the run does, not what changed (that is
    # the CHANGELOG's job): no "now ...", "Before, ...".
    section = text.split('#### What `openflo-run` writes', 1)[1]
    section = section.split('### ', 1)[0]
    assert not re.findall(r'\b(?:now|Before|used to|no longer)\b', section)


def test_legacy_samples_flag_splits_on_late_prefix(tmp_path, stubbed,
                                                   monkeypatch):
    trial = tmp_path / 'trial'
    _write(str(trial / 'expt_m1.fcs'), 1200, .7, 1)
    _write(str(trial / 'expt_late1.fcs'), 1200, .3, 2)
    out = tmp_path / 'out'
    assert _main(monkeypatch, '--trials', str(trial), '--out', str(out),
                 '--samples', 'm1,late1', '-q') == 0
    assert _bv(out / 'Group_A' / 'm1_stats.csv') == pytest.approx(70, abs=.1)
    assert _bv(out / 'Group_B' / 'late1_stats.csv') == pytest.approx(30, abs=.1)
