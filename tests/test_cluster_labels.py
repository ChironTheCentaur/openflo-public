"""The -1 noise/unclustered bucket is KEPT (not silently dropped) and clearly
labelled 'Unclustered (noise)' in both the shared helper and the stats export."""
from __future__ import annotations

import numpy as np

from openflo.pipeline import NOISE_CLUSTER_LABEL, FlowSample, cluster_label


def test_cluster_label_names_noise_and_real():
    assert cluster_label(-1) == NOISE_CLUSTER_LABEL == 'Unclustered (noise)'
    assert cluster_label(0) == 'Cluster 0'
    assert cluster_label(7) == 'Cluster 7'
    assert cluster_label('3') == 'Cluster 3'          # coerces stringified ids


def test_cluster_frequencies_keeps_and_labels_noise(synthetic_fcs):
    s = FlowSample(synthetic_fcs)
    n = len(s.data)
    lab = np.zeros(n, dtype=int)
    lab[: n // 5] = -1                                 # a slice → noise bucket
    lab[n // 5: n // 2] = 1
    s.data['cluster'] = lab

    freq = s.cluster_frequencies()
    # The noise row is PRESENT (not dropped) and unmistakably labelled.
    assert (freq['cluster'] == -1).any(), "noise cluster was dropped from export"
    noise = freq[freq['cluster'] == -1].iloc[0]
    assert noise['population'] == 'Unclustered (noise)'
    # Real clusters get 'Cluster N'; counts still sum to ALL events.
    real = freq[freq['cluster'] >= 0]
    assert set(real['population']) <= {'Cluster 0', 'Cluster 1'}
    assert int(freq['count'].sum()) == n


def test_cluster_frequencies_accepts_any_label_column(synthetic_fcs):
    # The same noise-labelled frequency export must work for a non-'cluster'
    # label column (e.g. 'leiden'), so FlowSOM/Leiden aren't stuck with a bare -1.
    s = FlowSample(synthetic_fcs)
    n = len(s.data)
    lab = np.ones(n, dtype=int)
    lab[: n // 4] = -1
    s.data['leiden'] = lab
    assert s.cluster_frequencies(label_col='leiden').empty is False
    freq = s.cluster_frequencies(label_col='leiden')
    assert (freq['cluster'] == -1).any()
    assert freq[freq['cluster'] == -1].iloc[0]['population'] == 'Unclustered (noise)'
    # A column that doesn't exist yields an empty frame (no crash).
    assert s.cluster_frequencies(label_col='nope').empty
