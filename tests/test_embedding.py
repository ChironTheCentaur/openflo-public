"""Shared dimensionality-reduction plumbing (UMAP / TriMap / PaCMAP).

The backends themselves (umap-learn / trimap / pacmap) are heavy and
partly non-deterministic, so these cover the pure helpers that all three
share — channel/row selection (_embedding_input) and writing the result
back (_store_embedding) — without importing any embedding library.
"""
import types

import numpy as np
import pandas as pd

import openflo.pipeline as fp


def _sample(df, fluor):
    """Minimal FlowSample-like stub carrying just what the embedding
    helpers read: a .data frame and a fluor_channels list."""
    s = types.SimpleNamespace()
    s.data = df
    s.fluor_channels = list(fluor)
    return s


# ── _embedding_input ─────────────────────────────────────────────────────────

def test_embedding_input_defaults_to_fluor_channels():
    df = pd.DataFrame({'FSC-A': [1.0, 2.0], 'CD11b': [3.0, 4.0],
                       'CD45': [5.0, 6.0]})
    s = _sample(df, ['CD11b', 'CD45'])
    X, sub_mask, avail = fp.FlowSample._embedding_input(
        s, None, sample_n=0, random_state=42)
    assert avail == ['CD11b', 'CD45']          # scatter excluded
    assert X.shape == (2, 2)
    assert sub_mask.all()


def test_embedding_input_drops_nonfinite_rows():
    df = pd.DataFrame({'CD11b': [1.0, np.nan, 3.0],
                       'CD45': [4.0, 5.0, np.inf]})
    s = _sample(df, ['CD11b', 'CD45'])
    _X, sub_mask, _avail = fp.FlowSample._embedding_input(
        s, None, sample_n=0, random_state=42)
    assert list(sub_mask) == [True, False, False]


def test_embedding_input_subsamples_deterministically():
    df = pd.DataFrame({'CD11b': np.arange(100.0), 'CD45': np.arange(100.0)})
    s = _sample(df, ['CD11b', 'CD45'])
    _X, m1, _ = fp.FlowSample._embedding_input(s, None, 10, random_state=7)
    _X, m2, _ = fp.FlowSample._embedding_input(s, None, 10, random_state=7)
    assert m1.sum() == 10
    assert list(m1) == list(m2)               # same seed → same rows


def test_embedding_input_honours_explicit_channels():
    df = pd.DataFrame({'CD11b': [1.0], 'CD45': [2.0], 'NOPE': [3.0]})
    s = _sample(df, ['CD11b', 'CD45'])
    _X, _m, avail = fp.FlowSample._embedding_input(
        s, ['CD45', 'NOPE', 'GONE'], 0, 42)
    assert avail == ['CD45', 'NOPE']          # only columns present survive


# ── _store_embedding ─────────────────────────────────────────────────────────

def test_store_embedding_writes_prefixed_columns_with_nan_gaps():
    df = pd.DataFrame({'CD11b': [1.0, 2.0, 3.0]})
    s = _sample(df, ['CD11b'])
    sub_mask = np.array([True, False, True])
    emb = np.array([[10.0, 20.0], [30.0, 40.0]])   # one row per embedded event
    out = fp.FlowSample._store_embedding(s, emb, sub_mask, 'TRIMAP')
    x = s.data['TRIMAP1'].to_numpy()
    y = s.data['TRIMAP2'].to_numpy()
    # Embedded rows carry the coords in order; the gap row is NaN on both axes.
    assert x[0] == 10.0 and x[2] == 30.0 and np.isnan(x[1])
    assert y[0] == 20.0 and y[2] == 40.0 and np.isnan(y[1])
    assert np.array_equal(out, emb)


# ── FlowSample.from_dataframe (processed-CSV ingest) ─────────────────────────

def test_from_dataframe_classifies_and_excludes_derived():
    df = pd.DataFrame({
        'FSC-A': [1.0, 2.0], 'BV421-A': [3.0, 4.0],
        'cluster': [0, 1], 'UMAP1': [0.1, 0.2], 'UMAP2': [0.3, 0.4],
        'flowsom_meta': [0, 1],
    })
    s = fp.FlowSample.from_dataframe(df, name='proc',
                                     labels={'BV421-A': 'CD11b'})
    assert s.name == 'proc'
    assert list(s.data.columns) == list(df.columns)
    # Derived columns are NOT markers; the real fluor is.
    assert 'BV421-A' in s.fluor_channels
    for derived in ('cluster', 'UMAP1', 'UMAP2', 'flowsom_meta'):
        assert derived not in s.fluor_channels
    assert 'FSC-A' in s.scatter_channels
    assert s.channel_labels['BV421-A'] == 'CD11b'
    # raw mirrors data; analysis-result slots initialised.
    assert s.raw.equals(s.data)
    assert s.cell_cycle_result is None and s.flowsom_result is None
