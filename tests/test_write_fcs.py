"""Round-trip test for write_fcs — a gated population written to FCS and
re-read with FlowSample must preserve events, channels and labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openflo.pipeline import FlowSample, write_fcs


def test_write_fcs_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'FSC-A': rng.uniform(1e4, 9e4, 500),
        'SSC-A': rng.uniform(1e4, 9e4, 500),
        'BV421-A': rng.normal(2000, 300, 500),
    })
    labels = {'BV421-A': 'CD11b'}
    out = tmp_path / 'pop.fcs'
    n = write_fcs(str(out), df, channel_labels=labels)
    assert n == 500
    assert out.is_file()

    s = FlowSample(str(out))
    s.run_qc()
    assert len(s.raw) == 500
    assert set(['FSC-A', 'SSC-A', 'BV421-A']) <= set(s.channel_names)
    # Raw values survive the round-trip (FCS stores finite floats).
    assert np.allclose(np.sort(s.raw['FSC-A'].to_numpy()),
                       np.sort(df['FSC-A'].to_numpy()), rtol=1e-4)
    # The antibody label rides along as $PnS.
    assert s.channel_labels.get('BV421-A') == 'CD11b'


def test_write_fcs_subset_and_order(tmp_path):
    df = pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0], 'C': [5.0, 6.0]})
    out = tmp_path / 'subset.fcs'
    write_fcs(str(out), df, channels=['C', 'A'])     # subset + reorder
    s = FlowSample(str(out))
    s.run_qc()
    assert list(s.channel_names) == ['C', 'A']
    assert s.raw['C'].tolist() == [5.0, 6.0]


def test_write_fcs_zeros_nonfinite(tmp_path):
    df = pd.DataFrame({'A': [1.0, np.nan, np.inf, -np.inf]})
    out = tmp_path / 'nf.fcs'
    write_fcs(str(out), df)
    s = FlowSample(str(out))
    s.run_qc()
    vals = s.raw['A'].tolist()
    assert vals[0] == 1.0
    assert all(np.isfinite(v) for v in vals)         # NaN/inf → 0
