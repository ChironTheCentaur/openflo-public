"""`.h5ad` export, round-tripped through a real file.

`to_anndata` is tested in memory, but `write_h5ad` — the path a user takes to
move an analysis into scanpy — had no coverage at all. An in-memory object that
serialises wrongly is indistinguishable from one that serialises correctly
until someone opens the file somewhere else, which is exactly when it is too
late to notice.

These write a real file and read it back with anndata, checking the matrix,
the observation annotations, the marker names and the recorded
`dropped_markers` all survive.
"""
import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip('anndata', reason='anndata not installed')

from openflo.interop import to_anndata, write_h5ad  # noqa: E402

MARKERS = ['CD3', 'CD4', 'CD8']


def _samples(n=50):
    rng = np.random.default_rng(0)
    return {
        's1': pd.DataFrame({**{m: rng.random(n) for m in MARKERS},
                            'cluster': rng.integers(-1, 4, n)}),
        's2': pd.DataFrame({**{m: rng.random(n) for m in MARKERS},
                            'cluster': rng.integers(-1, 4, n)}),
    }


def test_file_round_trip_preserves_the_matrix(tmp_path):
    samples = _samples()
    path = tmp_path / 'out.h5ad'
    n = write_h5ad(str(path), samples, MARKERS)

    assert path.exists() and path.stat().st_size > 0
    assert n == sum(len(df) for df in samples.values())

    back = ad.read_h5ad(str(path))
    assert back.n_obs == n
    assert list(back.var_names) == MARKERS, 'marker names did not survive'

    expected = np.vstack([df[MARKERS].to_numpy(float)
                          for df in samples.values()])
    np.testing.assert_allclose(np.asarray(back.X, dtype=float), expected,
                               rtol=0, atol=1e-12)


def test_sample_labels_survive_the_file(tmp_path):
    samples = _samples()
    path = tmp_path / 'out.h5ad'
    write_h5ad(str(path), samples, MARKERS)
    back = ad.read_h5ad(str(path))
    counts = back.obs['sample'].value_counts().to_dict()
    assert counts == {k: len(v) for k, v in samples.items()}


def test_obs_columns_survive_the_file(tmp_path):
    """Cluster ids are what a scanpy user joins on; losing them silently would
    make the export useless in a way nothing here would catch."""
    samples = _samples()
    path = tmp_path / 'out.h5ad'
    write_h5ad(str(path), samples, MARKERS, obs_cols=['cluster'])
    back = ad.read_h5ad(str(path))
    assert 'cluster' in back.obs.columns
    expected = np.concatenate([df['cluster'].to_numpy()
                               for df in samples.values()])
    got = pd.to_numeric(back.obs['cluster'], errors='coerce').to_numpy()
    np.testing.assert_array_equal(got, expected)


def test_dropped_markers_are_recorded_in_the_file(tmp_path):
    """A marker missing from ONE sample is omitted from every sample's block.
    That fact must reach the file, not just the in-memory object — otherwise
    the scanpy user cannot tell a 3-marker panel from a 4-marker one that lost
    a channel."""
    samples = _samples()
    samples['s2'] = samples['s2'].drop(columns=['CD8'])
    path = tmp_path / 'out.h5ad'
    write_h5ad(str(path), samples, MARKERS)

    back = ad.read_h5ad(str(path))
    assert list(back.var_names) == ['CD3', 'CD4'], 'shared markers only'
    dropped = list(back.uns.get('dropped_markers', []))
    assert 'CD8' in dropped, (
        'the omitted marker was not recorded in the written file')


def test_max_events_subsamples_and_reports_the_written_count(tmp_path):
    samples = _samples(n=200)
    path = tmp_path / 'out.h5ad'
    n = write_h5ad(str(path), samples, MARKERS, max_events=25)
    assert n == 50, 'expected 25 events from each of two samples'
    back = ad.read_h5ad(str(path))
    assert back.n_obs == n
    assert set(back.obs['sample']) == set(samples)


def test_written_count_matches_the_in_memory_object(tmp_path):
    """write_h5ad's return value is the number a caller reports to the user."""
    samples = _samples()
    path = tmp_path / 'out.h5ad'
    n = write_h5ad(str(path), samples, MARKERS)
    assert n == to_anndata(samples, MARKERS).n_obs
