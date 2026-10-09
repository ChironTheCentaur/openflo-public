"""FlowSample.cluster_heatmap — the per-cluster MEDIAN expression figure that
``cli.save_plots`` writes as ``<sample>_heatmap.png``.

It was 6% covered: only the "no cluster column" early return ran. These check
the numbers it draws, not just that a figure appears: every cell must equal
that cluster's median of that channel (a skewed fixture makes mean != median,
so a mean would be caught), rows must be named with the shared cluster names
(including the explicit noise row), columns with the antibody labels.
"""
import os

import numpy as np
import pandas as pd

os.environ.setdefault('MPLBACKEND', 'Agg')

from openflo.pipeline import NOISE_CLUSTER_LABEL, FlowSample  # noqa: E402


def _fixture():
    rng = np.random.default_rng(0)
    parts, labs = [], []
    for cid, (a, b) in {0: (1.0, 5.0), 1: (4.0, -2.0), 2: (-3.0, 0.5)}.items():
        # exponential tails: mean sits well away from the median
        parts.append(np.column_stack([a + rng.exponential(2.0, 101),
                                      b - rng.exponential(1.0, 101)]))
        labs += [cid] * 101
    parts.append(rng.normal(9.0, 0.1, (11, 2)))
    labs += [-1] * 11
    df = pd.DataFrame(np.vstack(parts), columns=['CD3-A', 'CD19-A'])
    df['cluster'] = labs
    return df


def _cells(ax, n_rows, n_cols):
    return np.asarray(ax.collections[0].get_array()).reshape(n_rows, n_cols)


def test_heatmap_cells_are_cluster_medians_with_named_rows():
    import matplotlib.pyplot as plt
    df = _fixture()
    s = FlowSample.from_dataframe(df, labels={'CD3-A': 'CD3', 'CD19-A': 'CD19'})
    assert s.fluor_channels == ['CD3-A', 'CD19-A']
    ax = s.cluster_heatmap()
    try:
        expected = df.groupby('cluster')[['CD3-A', 'CD19-A']].median()
        assert not np.allclose(
            expected, df.groupby('cluster')[['CD3-A', 'CD19-A']].mean())
        assert np.allclose(_cells(ax, 4, 2), expected.to_numpy())
        assert [t.get_text() for t in ax.get_yticklabels()] == [
            NOISE_CLUSTER_LABEL, 'Cluster 0', 'Cluster 1', 'Cluster 2']
        assert [t.get_text() for t in ax.get_xticklabels()] == ['CD3', 'CD19']
        assert 'Cluster Median Expression' in ax.get_title()
    finally:
        plt.close('all')


def test_heatmap_other_label_column_and_channel_subset():
    import matplotlib.pyplot as plt
    df = _fixture().rename(columns={'cluster': 'leiden'})
    s = FlowSample.from_dataframe(df)
    ax = s.cluster_heatmap(channels=['CD19-A'], label_col='leiden')
    try:
        expected = df.groupby('leiden')['CD19-A'].median().to_numpy()
        assert np.allclose(_cells(ax, 4, 1).ravel(), expected)
    finally:
        plt.close('all')


def test_heatmap_without_labels_returns_none():
    s = FlowSample.from_dataframe(_fixture().drop(columns='cluster'))
    assert s.cluster_heatmap() is None
