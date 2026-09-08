"""Tests for the headless FCS export module — DataFrames written to FCS 3.1
must round-trip event/channel counts via FlowIO, and export_populations must
write one sanitised file per population."""
from __future__ import annotations

import flowio
import numpy as np
import pandas as pd

from openflo.fcs_export import export_populations, safe_filename, write_fcs


def _make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'FSC-A': rng.uniform(1e4, 9e4, n),
        'SSC-A': rng.uniform(1e4, 9e4, n),
        'BV421-A': rng.normal(2000, 300, n),
    })


def test_write_fcs_roundtrips_counts(tmp_path):
    df = _make_df(n=300)
    out = tmp_path / 'pop.fcs'
    n = write_fcs(df, str(out), channel_labels={'BV421-A': 'CD11b'})
    assert n == 300
    assert out.is_file()

    fd = flowio.FlowData(str(out))
    assert fd.event_count == 300
    assert fd.channel_count == 3
    # Channel names ($PnN) round-trip.
    names = [fd.channels[i]['pnn'] for i in range(1, fd.channel_count + 1)]
    assert names == ['FSC-A', 'SSC-A', 'BV421-A']
    # The antibody label rides along as $P3S.
    assert fd.channels[3].get('pns') == 'CD11b'

    # Event values survive the round-trip.
    events = np.reshape(fd.events, (fd.event_count, fd.channel_count))
    assert np.allclose(np.sort(events[:, 0]),
                       np.sort(df['FSC-A'].to_numpy()), rtol=1e-4)


def test_flowsample_reads_external_fcs_empty_pns_falls_back_to_name(tmp_path):
    """Reading an externally-produced FCS (not our own writer): a channel with an
    empty $PnS falls back to the detector name for its label; a real $PnS is kept.
    The read path was only ever exercised through write_fcs before."""
    import openflo.pipeline as fp
    out = tmp_path / 'ext.fcs'
    ev = np.array([1.0, 10.0, 2.0, 20.0], dtype=np.float32)   # 2 events × 2 chans
    with open(out, 'wb') as fh:
        flowio.create_fcs(fh, ev, channel_names=['FSC-A', 'PE-A'],
                          opt_channel_names=['', 'CD3'])
    s = fp.FlowSample(str(out))
    assert s.channel_labels['FSC-A'] == 'FSC-A'    # empty $PnS → detector name
    assert s.channel_labels['PE-A'] == 'CD3'       # real $PnS kept


def test_write_fcs_preserves_per_cell_row_pairing(tmp_path):
    """Each event's cross-channel values must round-trip TOGETHER (row alignment
    + column order) — not just each column's sorted multiset. Guards the coverage
    gap where sort-per-channel checks would pass a row-shuffle / column-swap /
    single-channel-only corruption."""
    df = pd.DataFrame({'FSC-A': [1.0, 4.0, 7.0],
                       'SSC-A': [2.0, 5.0, 8.0],
                       'BV421-A': [3.0, 6.0, 9.0]})
    out = tmp_path / 'rows.fcs'
    write_fcs(df, str(out))
    fd = flowio.FlowData(str(out))
    events = np.reshape(fd.events, (fd.event_count, fd.channel_count))
    np.testing.assert_allclose(events, df.to_numpy(), rtol=1e-6)


def test_write_fcs_zeros_nonfinite(tmp_path):
    df = pd.DataFrame({'A': [1.0, np.nan, np.inf, -np.inf]})
    out = tmp_path / 'nf.fcs'
    write_fcs(df, str(out))
    fd = flowio.FlowData(str(out))
    vals = list(fd.events)
    assert vals[0] == 1.0
    assert all(np.isfinite(v) for v in vals)
    # non-finite cells become EXACTLY 0.0 (not merely finite), finite kept as-is.
    np.testing.assert_allclose(np.asarray(vals), [1.0, 0.0, 0.0, 0.0])


def test_export_populations_writes_n_files(tmp_path):
    pops = {
        'CD4 T': _make_df(n=100, seed=1),
        'CD8 T': _make_df(n=120, seed=2),
        'B cell': _make_df(n=80, seed=3),
    }
    paths = export_populations(pops, str(tmp_path / 'out'),
                               channel_labels={'BV421-A': 'CD11b'})
    assert len(paths) == 3
    names = sorted(p.rsplit('\\', 1)[-1].rsplit('/', 1)[-1] for p in paths)
    assert names == ['B_cell.fcs', 'CD4_T.fcs', 'CD8_T.fcs']
    for p in paths:
        assert flowio.FlowData(p).channel_count == 3


def test_export_populations_sanitises_and_disambiguates(tmp_path):
    pops = {
        'CD4/CD8 ratio*': _make_df(n=10, seed=4),
        'CD4_CD8 ratio': _make_df(n=11, seed=5),   # collides after sanitising
    }
    paths = export_populations(pops, str(tmp_path / 'out'))
    assert len(paths) == 2
    assert len({p.lower() for p in paths}) == 2     # no silent overwrite
    for p in paths:
        assert p.endswith('.fcs')


def test_safe_filename():
    assert safe_filename('CD4 T') == 'CD4_T'
    # Path separators become underscores; other unsafe chars are stripped.
    assert safe_filename('a/b\\c:d*?') == 'a_b_cd'
    assert safe_filename('   ') == 'population'
    assert safe_filename('trailing.') == 'trailing'
