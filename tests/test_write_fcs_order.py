"""Both public ``write_fcs`` functions take ``(df, path)``.

``openflo.write_fcs`` (``openflo.pipeline.write_fcs``) used to take
``(path, df)`` while ``openflo.fcs_export.write_fcs`` took ``(df, path)``.
Both now take ``(df, path)`` and write through one writer; the old order
still works with a DeprecationWarning, and anything else is a TypeError.
All data here are generated in the test."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import openflo
from openflo import fcs_export
from openflo.pipeline import FlowSample

LABELS = {'FL1-A': 'CD901b', 'FL2-A': 'CD902a'}
WRITERS = {'openflo.write_fcs': openflo.write_fcs,
           'fcs_export.write_fcs': fcs_export.write_fcs}
DEPRECATION = r'write_fcs\(path, df\) is deprecated; pass write_fcs\(df, path\)'


def _events(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FSC-A': rng.uniform(1e4, 9e4, n),
                         'FL1-A': rng.normal(2000, 300, n),
                         'FL2-A': rng.lognormal(6, 1, n)})


def _assert_holds(path, df):
    """The file at ``path`` re-opens with ``df``'s channels, values and labels."""
    s = FlowSample(str(path))
    assert list(s.channel_names) == list(df.columns)
    assert len(s.raw) == len(df)
    np.testing.assert_allclose(s.raw[list(df.columns)].to_numpy(),
                               df.to_numpy(dtype=np.float32), rtol=1e-6)
    assert s.channel_labels.get('FL1-A') == 'CD901b'


def _write_strict(fn, *args, **kw):
    """Call ``fn`` with every warning raised as an error."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return fn(*args, **kw)


def test_both_writers_write_identical_bytes_for_df_path(tmp_path):
    df = _events()
    top, export = tmp_path / 'top.fcs', tmp_path / 'export.fcs'
    assert _write_strict(openflo.write_fcs, df, str(top),
                         channel_labels=LABELS) == len(df)
    assert _write_strict(fcs_export.write_fcs, df, str(export),
                         channel_labels=LABELS) == len(df)
    assert top.read_bytes() == export.read_bytes()
    _assert_holds(top, df)


@pytest.mark.parametrize('name', list(WRITERS))
def test_df_path_order_does_not_warn(tmp_path, name):
    df = _events(seed=1)
    out = tmp_path / 'new.fcs'
    _write_strict(WRITERS[name], df, out, channel_labels=LABELS)   # PathLike
    _assert_holds(out, df)


@pytest.mark.parametrize('name', list(WRITERS))
def test_legacy_path_df_order_warns_and_writes_the_same_file(tmp_path, name):
    fn = WRITERS[name]
    df = _events(seed=2)
    new, old = tmp_path / 'new.fcs', tmp_path / 'old.fcs'
    _write_strict(fn, df, str(new), channel_labels=LABELS)
    with pytest.warns(DeprecationWarning, match=DEPRECATION) as rec:
        n = fn(str(old), df, channel_labels=LABELS)
    assert n == len(df)
    # The warning names the caller's line, not a line inside openflo.
    assert [os.path.basename(w.filename) for w in rec] == \
        [os.path.basename(__file__)]
    assert old.read_bytes() == new.read_bytes()
    _assert_holds(old, df)


@pytest.mark.parametrize('name', list(WRITERS))
@pytest.mark.parametrize('args', [
    ('a.fcs', 'b.fcs'),                     # two paths
    ('df', 'df'),                           # two DataFrames
    ('df', 3),                              # not a path
    (np.zeros((4, 2)), 'out.fcs'),          # an array, not a DataFrame
], ids=['two-paths', 'two-frames', 'int-path', 'ndarray'])
def test_wrong_types_raise_type_error(tmp_path, name, args):
    df = _events(10)
    real = [a if not isinstance(a, str) else df if a == 'df' else str(tmp_path / a)
            for a in args]
    with pytest.raises(TypeError, match=r'write_fcs\(df, path\)'):
        _write_strict(WRITERS[name], *real)
    assert list(tmp_path.iterdir()) == []      # nothing written


def test_fcs_export_channels_selects_and_orders(tmp_path):
    df = _events(seed=3)
    out = tmp_path / 'subset.fcs'
    n = _write_strict(fcs_export.write_fcs, df, str(out),
                      channel_labels=LABELS, channels=['FL2-A', 'FL1-A'])
    assert n == len(df)
    s = FlowSample(str(out))
    assert list(s.channel_names) == ['FL2-A', 'FL1-A']
    np.testing.assert_allclose(s.raw[['FL2-A', 'FL1-A']].to_numpy(),
                               df[['FL2-A', 'FL1-A']].to_numpy(dtype=np.float32),
                               rtol=1e-6)
    assert s.channel_labels.get('FL2-A') == 'CD902a'
    assert s.channel_labels.get('FL1-A') == 'CD901b'
    # The top-level function's channels= gives the same file.
    top = tmp_path / 'top.fcs'
    _write_strict(openflo.write_fcs, df, str(top), channels=['FL2-A', 'FL1-A'],
                  channel_labels=LABELS)
    assert top.read_bytes() == out.read_bytes()
