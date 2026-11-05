"""An integer FCS channel stored log-amplified ($PnE f1,f2) or with a gain
($PnG) loads as intensities, as FCS 3.1 defines them.

The loader used to hand flowio's stored integers straight to the analysis,
so a 4-decade log channel's 1023 read 1023 instead of 9,910 and a gain-2
channel read twice its intensity, with no warning. Every case here writes
its own synthetic file and checks the loaded numbers against the formula,
and the negative cases check what must NOT change: linear integer channels,
float data (whose $PnG is not a storage scale) and the Time channel.
"""
from __future__ import annotations

import numpy as np
import pytest

from openflo import fcs_export
from openflo.pipeline import FlowSample
from tests._fcs_writer import write_fcs

_NAMES = ['FSC-A', 'FL1-A', 'FL2-A', 'Time']
_STORED = np.array([[100, 0, 100, 1],
                    [200, 256, 200, 2],
                    [300, 512, 300, 3],
                    [400, 768, 400, 4],
                    [500, 1023, 500, 5]])


def _load(tmp_path, **kw):
    path = write_fcs(tmp_path / 'synthetic_1.fcs', _STORED, _NAMES, **kw)
    return FlowSample(str(path))


def test_log_amplified_channel_loads_as_intensity(tmp_path):
    s = _load(tmp_path, pne=['0,0', '4,1', '0,0', '0,0'])
    expected = 10.0 ** (4 * _STORED[:, 1] / 1024)
    np.testing.assert_allclose(s.raw['FL1-A'], expected, rtol=1e-12)
    np.testing.assert_allclose(s.data['FL1-A'], expected, rtol=1e-12)
    assert s.raw['FL1-A'].iloc[-1] == pytest.approx(9910.0, rel=1e-3)
    assert s.fcs_decoding == {'FL1-A': 'log, 4 decades'}


def test_log_zero_value_follows_f2(tmp_path):
    # f2 is the value of a stored 0; a 0 there means 1 (FCS 3.1).
    s = _load(tmp_path, pne=['0,0', '3,0.5', '2,0', '0,0'])
    np.testing.assert_allclose(
        s.raw['FL1-A'], 0.5 * 10.0 ** (3 * _STORED[:, 1] / 1024))
    np.testing.assert_allclose(
        s.raw['FL2-A'], 10.0 ** (2 * _STORED[:, 2] / 1024))


def test_gain_divides_linear_integer_channel(tmp_path):
    s = _load(tmp_path, png=[1, 1, 2, 1])
    np.testing.assert_allclose(s.raw['FL2-A'], _STORED[:, 2] / 2)
    assert s.fcs_decoding == {'FL2-A': 'gain 2'}


def test_export_range_covers_decoded_values(tmp_path):
    s = _load(tmp_path, pne=['0,0', '4,1', '0,0', '0,0'])
    frame, ranges, _ = fcs_export.population_export_frame(
        s, np.ones(len(s.data), dtype=bool))
    assert float(ranges['FL1-A']) >= frame['FL1-A'].max()
    assert ranges['FL1-A'] == '10000'


# ── negative cases: nothing to decode, nothing changes ────────────────────


def test_linear_integer_file_is_unchanged(tmp_path):
    s = _load(tmp_path)
    np.testing.assert_array_equal(s.raw.to_numpy(), _STORED)
    assert s.fcs_decoding == {}


def test_time_channel_is_never_rescaled(tmp_path):
    s = _load(tmp_path, pne=['0,0', '0,0', '0,0', '4,1'], png=[1, 1, 1, 5])
    np.testing.assert_array_equal(s.raw['Time'], _STORED[:, 3])


def test_float_data_keeps_values_despite_gain(tmp_path):
    s = _load(tmp_path, datatype='F', png=[1, 1, 2, 1])
    np.testing.assert_array_equal(s.raw.to_numpy(), _STORED)
    assert s.fcs_decoding == {}
