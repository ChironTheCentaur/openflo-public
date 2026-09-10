"""Loading must not copy the events twice — and `raw` must stay pristine.

`FlowSample._load` used to materialise the event array into pandas and then
take a second full copy for `data`. Neither was needed, and at the workload
that matters — 20-25 multimillion-event files — they dominated: measured on
8 x 2M events x 18 channels, 2.25s / +2304 MB became 0.75s / +1152 MB;
projected to 25 files, 7.0s / 7.2 GB -> 2.3s / 3.6 GB. The memory halving is
the bigger half, because 7 GB resident is where a machine starts paging and
paging is what makes loading *feel* slow.

The optimisation is only safe because of one property, which is what these
tests pin: `data` and `raw` are INDEPENDENT despite sharing memory until
written. `raw` holds the pristine detector values the .fcs export reads, and
`self.data` is written in place in a dozen places (compensation and transforms
among them). If someone later "simplifies" the shallow copy into an alias,
compensation would overwrite `raw` and the export would silently emit
transformed values as if they were raw.
"""
import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample, write_fcs


@pytest.fixture(scope='module')
def sample_path(tmp_path_factory):
    rng = np.random.RandomState(0)
    n = 20_000
    df = pd.DataFrame({
        'FSC-A': rng.normal(50000, 12000, n).astype(np.float32),
        'SSC-A': rng.lognormal(6, 1, n).astype(np.float32),
        'CD3-A': rng.lognormal(5, 1, n).astype(np.float32),
        'CD4-A': rng.lognormal(5, 1, n).astype(np.float32),
    })
    path = tmp_path_factory.mktemp('load') / 'sample.fcs'
    write_fcs(str(path), df)
    return str(path)


def test_data_and_raw_start_equal(sample_path):
    s = FlowSample(sample_path)
    assert s.data.equals(s.raw)
    assert list(s.data.columns) == list(s.raw.columns)


def test_writing_data_leaves_raw_pristine(sample_path):
    """THE property. `raw` is what the .fcs export writes out, so a leak here
    means exported files silently contain compensated/transformed values
    presented as raw detector readings."""
    s = FlowSample(sample_path)
    ch = s.data.columns[2]
    original = s.raw[ch].copy()

    s.data[ch] = 12345.0                       # what _apply_comp does
    assert s.raw[ch].equals(original), (
        'a column write to data leaked into raw — data must not alias raw')

    s.data.loc[s.data.index[:10], ch] = -1.0   # what transforms/masks do
    assert s.raw[ch].equals(original), (
        'a .loc write to data leaked into raw')


def test_writing_raw_does_not_disturb_data(sample_path):
    """The converse: the two are independent in both directions."""
    s = FlowSample(sample_path)
    ch = s.data.columns[1]
    original = s.data[ch].copy()
    s.raw[ch] = 999.0
    assert s.data[ch].equals(original)


def test_dtypes_are_not_upcast_on_load(sample_path):
    """float32 events must stay float32. Silently doubling to float64 would
    undo the memory saving this change exists for."""
    s = FlowSample(sample_path)
    assert set(map(str, s.raw.dtypes)) == {'float32'}
    assert set(map(str, s.data.dtypes)) == {'float32'}


def test_the_events_survive_the_source_buffer_going_away(sample_path):
    """`raw` shares memory with flowio's buffer rather than copying it, so the
    frame must keep that buffer alive on its own."""
    import gc
    s = FlowSample(sample_path)
    expected = float(s.raw.iloc[0, 0])
    gc.collect()                                # drop any transient references
    assert float(s.raw.iloc[0, 0]) == expected
    assert len(s.raw) == 20_000
