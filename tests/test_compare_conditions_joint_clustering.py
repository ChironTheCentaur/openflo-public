"""FlowExperiment.compare_conditions compares cluster ids ACROSS samples, so
they must come from one clustering of all of them.

FlowExperiment.run_all clusters every sample on its own, and PhenoGraph,
Leiden and FlowSOM number clusters arbitrarily (PhenoGraph by size). With CD4
at 80 % of group A and 20 % of group B, every sample's largest cluster was id
0, CD4 in A and CD8 in B, and the comparison table read 80 % vs 80 % for
"cluster 0": no difference, where CD4 fell four-fold. openflo-run's group
comparisons already cluster the samples together (cli._shared_cluster_
labels); the library call now does the same unless it is told the labels are
already shared.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowExperiment, FlowSample


def _sample(rng, name, f_cd4, n=3000):
    """CD4-like (FL5 high) and CD8-like (FL4 high) cells, plus each sample's
    OWN cluster ids numbered by size, as a per-sample PhenoGraph run does."""
    n4 = int(n * f_cd4)
    n8 = n - n4
    df = pd.DataFrame({
        'FSC-A': rng.normal(5e4, 5e3, n),
        'FL5-A': np.r_[rng.normal(0.8, 0.03, n4), rng.normal(0.3, 0.03, n8)],
        'FL4-A': np.r_[rng.normal(0.3, 0.03, n4), rng.normal(0.8, 0.03, n8)]})
    s = FlowSample.from_dataframe(df, name=name)
    s.fluor_channels = ['FL5-A', 'FL4-A']
    cd4 = np.r_[np.ones(n4, bool), np.zeros(n8, bool)]
    s.data['cluster'] = np.where(cd4 == (n4 >= n8), 0, 1)
    s.data['truth_cd4'] = cd4
    return s


@pytest.fixture
def experiment():
    logging.disable(logging.INFO)
    rng = np.random.default_rng(11)
    exp = FlowExperiment.__new__(FlowExperiment)
    exp.samples = {nm: _sample(rng, nm, f) for nm, f in
                   (('A1', .8), ('A2', .8), ('B1', .2), ('B2', .2))}
    # The setup's point: each sample's own id 0 is its largest population.
    assert exp.samples['A1'].data.loc[exp.samples['A1'].data.cluster == 0,
                                      'truth_cd4'].all()
    assert not exp.samples['B1'].data.loc[exp.samples['B1'].data.cluster == 0,
                                          'truth_cd4'].any()
    yield exp
    logging.disable(logging.NOTSET)


def _cd4_pct(exp, summary, condition):
    """% of events in clusters whose events are CD4 (by the planted truth),
    for one condition; and checks every joint cluster is one phenotype."""
    col = exp.joint_clustering['label_col']
    pure_cd4 = {}
    for s in exp.samples.values():
        for c, grp in s.data.groupby(col)['truth_cd4']:
            frac = float(grp.mean())
            assert frac > 0.99 or frac < 0.01, (s.name, c, frac)
            pure_cd4.setdefault(c, frac > 0.5)
    rows = summary[summary.condition == condition]
    return float(rows[rows.cluster.map(pure_cd4)].mean_pct.sum())


def test_conditions_are_compared_on_one_joint_clustering(experiment):
    """Default (PhenoGraph, k=30): CD4 80 % vs 20 %, as planted; the old
    per-sample-id comparison reported 80 % vs 80 % for cluster 0."""
    summary = experiment.compare_conditions(['A1', 'A2'], ['B1', 'B2'],
                                            plot=False)
    assert experiment.joint_clustering['samples'] == ('A1', 'A2', 'B1', 'B2')
    assert _cd4_pct(experiment, summary, 'A') == pytest.approx(80.0, abs=1.0)
    assert _cd4_pct(experiment, summary, 'B') == pytest.approx(20.0, abs=1.0)
    # Each sample's own ids are left as they were.
    assert set(experiment.samples['B1'].data['cluster']) == {0, 1}


def test_method_and_settings_pass_through(experiment):
    summary = experiment.compare_conditions(
        ['A1', 'A2'], ['B1', 'B2'], plot=False, method='flowsom',
        n_metaclusters=2)
    assert experiment.joint_clustering['method'] == 'flowsom'
    assert summary.cluster.nunique() == 2
    assert _cd4_pct(experiment, summary, 'A') == pytest.approx(80.0, abs=1.0)
    assert _cd4_pct(experiment, summary, 'B') == pytest.approx(20.0, abs=1.0)


def test_a_joint_clustering_already_made_is_reused(experiment, monkeypatch):
    """cluster_jointly() once on every sample, then any pair compares on the
    same ids without clustering again."""
    experiment.cluster_jointly(method='flowsom', n_metaclusters=2)
    calls = []
    monkeypatch.setattr(FlowExperiment, 'cluster_jointly',
                        lambda self, *a, **k: calls.append(a))
    summary = experiment.compare_conditions(['A1'], ['B2'], plot=False)
    assert not calls
    assert _cd4_pct(experiment, summary, 'A') == pytest.approx(80.0, abs=1.0)
    assert _cd4_pct(experiment, summary, 'B') == pytest.approx(20.0, abs=1.0)
