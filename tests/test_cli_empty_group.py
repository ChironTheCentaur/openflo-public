"""A batch run whose gates leave a group (almost) no events ends with a
plain message, not umap-learn's error.

Before, a group gated down to 0 events stopped the whole run in
write_group_outputs with sklearn's "Found array with 0 sample(s)", and 2
or 3 events with a scipy error from UMAP's layout. A gate on a channel a
sample lacks now keeps no events (#447), so this is easy to reach.
Generated samples only.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp
from openflo import cli_compare
from openflo.cli_plots import UMAP_MIN_EVENTS


def _sample(name, n, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'FSC-A': rng.random(n) * 1e5,
                       'SSC-A': rng.random(n) * 1e5,
                       'FL1-A': rng.random(n) * 1e4,
                       'FL2-A': rng.random(n) * 1e4})
    return fp.FlowSample.from_dataframe(df, name=name)


def _run(tmp_path, sizes, tl=''):
    """write_group_outputs over one group per entry of `sizes` (events per
    sample of that group), with the real FlowSOM shared clustering."""
    groups, results, tasks = [], {}, []
    for i, ns in enumerate(sizes):
        names = [f's{i}{j}' for j in range(len(ns))]
        groups.append({'name': f'G{i}', 'samples': names})
        results[f'G{i}'] = [_sample(nm, n, seed=j)
                            for j, (nm, n) in enumerate(zip(names, ns, strict=True))]
        tasks += [{'name': nm, 'group': f'G{i}'} for nm in names]
    return cli_compare.write_group_outputs(
        groups, results, {}, str(tmp_path), lambda *_: None,
        {'method': 'flowsom', 'k': 3}, tasks=tasks, tl=tl)


@pytest.mark.parametrize('sizes', [[0], [0, 0], [2], [3], [1, 2]],
                         ids=str)
def test_a_group_with_too_few_events_is_not_drawn_and_says_why(
        tmp_path, capsys, sizes):
    """The real UMAP: on master 0, 2 and 3 events raised from umap-learn."""
    assert _run(tmp_path, [sizes]) is True     # one group: nothing to compare
    out = capsys.readouterr().out
    n = sum(sizes)
    assert (f"G0: {n:,} event(s) left after gating in its {len(sizes)} "
            "sample(s), too few to draw") in out
    assert 'UMAP and pair scatters were not drawn' in out
    assert not [f for f in os.listdir(tmp_path) if f.endswith('.png')]


@pytest.mark.parametrize('n,drawn', [(UMAP_MIN_EVENTS - 1, False),
                                     (UMAP_MIN_EVENTS, True), (500, True)])
def test_the_umap_is_drawn_from_the_minimum_up(tmp_path, capsys,
                                               monkeypatch, n, drawn):
    calls = []
    monkeypatch.setattr(cli_compare, 'run_group_umap',
                        lambda samples, *a, **k: calls.append(len(samples)))
    monkeypatch.setattr(cli_compare, 'save_group_pair_scatters',
                        lambda *a, **k: None)
    _run(tmp_path, [[n]])
    assert bool(calls) is drawn
    assert ('too few to draw' in capsys.readouterr().out) is not drawn


def test_a_group_of_four_events_gets_its_umap(tmp_path, capsys):
    """The real UMAP draws from 4 events (3 raise in umap-learn's spectral
    layout), so UMAP_MIN_EVENTS is not set higher than it must be."""
    assert _run(tmp_path, [[4]]) is True
    assert 'too few to draw' not in capsys.readouterr().out
    assert 'G0_umap.png' in os.listdir(tmp_path)


def _no_figures(monkeypatch):
    monkeypatch.setattr(cli_compare, 'run_group_umap', lambda *a, **k: None)
    monkeypatch.setattr(cli_compare, 'save_group_pair_scatters',
                        lambda *a, **k: None)


@pytest.mark.parametrize('sizes', [[[500], [0]], [[500, 500], [0]],
                                   [[500, 500], [0, 0]], [[0], [500]]],
                         ids=str)
def test_a_group_with_no_events_gets_no_comparison(tmp_path, capsys,
                                                    monkeypatch, sizes):
    """Before, compare_G0_vs_G1 was written with one side's rows only and
    the run reported success."""
    _no_figures(monkeypatch)
    assert _run(tmp_path, sizes) is False
    out = capsys.readouterr().out
    gone = 'G1' if sum(sizes[0]) else 'G0'
    assert (f"{gone} had no events after gating -- G0 vs G1 comparison "
            "skipped") in out
    files = os.listdir(tmp_path)
    assert 'compare_G0_vs_G1.csv' not in files
    assert 'compare_G0_vs_G1.png' not in files


def test_an_earlier_comparison_of_that_pair_is_removed(tmp_path, capsys,
                                                       monkeypatch):
    _no_figures(monkeypatch)
    assert _run(tmp_path, [[500], [500]]) is True
    assert 'compare_G0_vs_G1.csv' in os.listdir(tmp_path)
    assert _run(tmp_path, [[500], [0]]) is False
    assert 'compare_G0_vs_G1.csv' not in os.listdir(tmp_path)


def test_an_empty_sample_is_left_out_of_its_groups_n_and_says_so(
        tmp_path, capsys, monkeypatch):
    _no_figures(monkeypatch)
    assert _run(tmp_path, [[500, 0, 3], [500, 500]]) is True
    out = capsys.readouterr().out
    assert ("No events left after gating in G0/s01: it is left out of the "
            "group means, SD, p-values and sample counts (n) in the group "
            "comparisons.") in out
    assert 's02' not in out.split('No events left')[1].split('\n')[0]
    table = pd.read_csv(tmp_path / 'compare_G0_vs_G1.csv')
    by = table.groupby('condition')['n_samples'].max()
    assert by['G0'] == 2 and by['G1'] == 2    # 3 samples, one empty


def test_two_empty_samples_are_named_together(tmp_path, capsys,
                                              monkeypatch):
    _no_figures(monkeypatch)
    _run(tmp_path, [[500, 0], [500, 0]], tl='T1')
    out = capsys.readouterr().out
    assert ("[T1] No events left after gating in G0/s01, G1/s11: they are "
            "left out of") in out


def test_the_trial_is_named_in_the_messages(tmp_path, capsys, monkeypatch):
    _no_figures(monkeypatch)
    _run(tmp_path, [[500], [0]], tl='T1')
    out = capsys.readouterr().out
    assert '[T1] G1: 0 event(s) left after gating' in out
    assert '[T1] G1 had no events after gating' in out


@pytest.mark.parametrize('n', [0, 5, 30])
def test_phenograph_skips_a_sample_with_k_or_fewer_events(caplog, n):
    """Before, the sample failed with sklearn's 'Found array with 0
    sample(s)' / 'Expected n_neighbors <= n_samples_fit'."""
    from openflo.shared_clustering import cluster_by_method
    s = _sample('few', n)
    with caplog.at_level('WARNING'):
        cluster_by_method(s, {'method': 'phenograph', 'k': 30, 'n_jobs': 1})
    assert (s.data['cluster'] == -1).all() and len(s.data) == n
    stats = s.cluster_stats()
    assert int(stats['count'].sum()) == n and 'count' in stats.columns
    assert any(f'few: {n:,} event(s) left after gating, too few to cluster '
               'with k=30 (it needs more than 30)' in r.getMessage()
               for r in caplog.records)


def test_phenograph_clusters_from_k_plus_one_events(caplog):
    from openflo.shared_clustering import cluster_by_method
    s = _sample('enough', 31)
    cluster_by_method(s, {'method': 'phenograph', 'k': 30, 'n_jobs': 1})
    assert (s.data['cluster'] >= 0).all()


def test_too_few_events_for_the_shared_phenograph_clustering_is_said(
        tmp_path, capsys, monkeypatch):
    """Every sample together under k events: the shared clustering gives
    no labels, and the comparisons are not written."""
    _no_figures(monkeypatch)
    groups = [{'name': 'G0', 'samples': ['a']}, {'name': 'G1', 'samples': ['b']}]
    results = {'G0': [_sample('a', 10)], 'G1': [_sample('b', 10, seed=1)]}
    tasks = [{'name': 'a', 'group': 'G0'}, {'name': 'b', 'group': 'G1'}]
    ok = cli_compare.write_group_outputs(
        groups, results, {}, str(tmp_path), lambda *_: None,
        {'method': 'phenograph', 'k': 30, 'n_jobs': 1}, tasks=tasks)
    assert ok is False
    out = capsys.readouterr().out
    assert 'phenograph gave no cluster labels for' in out
    assert 'compare_G0_vs_G1.csv' not in os.listdir(tmp_path)
