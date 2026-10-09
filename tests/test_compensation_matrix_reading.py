"""Every way a spillover matrix arrives must compensate to the TRUE signal.

Four readers accepted a file and then compensated wrongly, without a word:

* a SPILL / $SPILLOVER that names its channels by 1-based PARAMETER NUMBER
  (``3,3,4,5,...``, as some acquisition software writes it) matched no column,
  so auto_compensate logged "No matching channels" and left the sample
  uncompensated (median error 2%, 21%, 9% on the three channels here), while
  read_compensation_matrix handed the editor channels named '3', '4', '5';
* a matrix written in PERCENT (diagonal 100) was applied as fractions: every
  compensated channel came out at 0.01x its true value;
* a CSV whose rows are listed in another order than its header had its row
  labels ignored, so the FL2 row was applied as the FL4 row;
* FlowJo's CSV export (a header of channel names, no corner cell, unlabelled
  rows) failed as "not square (shape (3, 2))", and a .wsp with no matrix
  raised RuntimeError where (None, None) is documented.

Each test checks the recovered signal (or matrix) against the planted truth.
"""
from __future__ import annotations

import logging

import flowio
import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp

CHANNELS = ['FSC-A', 'SSC-A', 'FL4-A', 'FL5-A', 'FL2-A', 'Time']
FLUOR = ['FL4-A', 'FL5-A', 'FL2-A']
# Asymmetric on purpose: a transposed or row-shuffled matrix cannot pass.
M = np.array([[1.0, 0.21, 0.01],
              [0.02, 1.0, 0.07],
              [0.0, 0.005, 1.0]])
N = 2000


def _truth(seed=0):
    return np.exp(np.random.default_rng(seed).normal(7, 1, (N, 3)))


TRUE = _truth()


def _fcs(path, spill_text):
    rng = np.random.default_rng(1)
    ev = np.column_stack([rng.uniform(1e4, 1e5, N), rng.uniform(1e4, 1e5, N),
                          TRUE @ M, np.arange(N, dtype=float)])
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.ravel().tolist(), CHANNELS,
                          metadata_dict={'$SPILLOVER': spill_text})
    return str(path)


def _values(matrix):
    return ','.join(repr(float(v)) for v in np.asarray(matrix).ravel())


def _rel_err(sample):
    got = sample.data[FLUOR].to_numpy(float)
    return float(np.max(np.abs(got - TRUE) / TRUE))


# ── SPILL naming its channels by parameter number ───────────────────────────

def test_spill_by_parameter_number_compensates(tmp_path):
    """FL4/FL5/FL2 are parameters 3, 4 and 5. Before: comp_channels [] and
    the data left as measured."""
    path = _fcs(tmp_path / 'idx.fcs', f'3,3,4,5,{_values(M)}')
    s = fp.FlowSample(path)
    s.auto_compensate()
    assert s.comp_channels == FLUOR
    assert _rel_err(s) < 1e-5


def test_both_readers_agree_on_a_spill_by_parameter_number(tmp_path):
    """The editor and the .wsp export read the file with
    read_compensation_matrix; it must name the same $PnN channels and hold the
    same matrix auto_compensate applied (it returned ['3', '4', '5'])."""
    path = _fcs(tmp_path / 'idx.fcs', f'3,3,4,5,{_values(M)}')
    s = fp.FlowSample(path)
    s.auto_compensate()
    channels, matrix = fp.read_compensation_matrix(path)
    assert channels == FLUOR == s.comp_channels
    np.testing.assert_array_equal(matrix, s.comp_matrix)
    np.testing.assert_array_equal(matrix, M)


def test_spill_by_name_is_not_remapped(tmp_path):
    """Control: the ordinary form still reads as names."""
    path = _fcs(tmp_path / 'names.fcs', f'3,{",".join(FLUOR)},{_values(M)}')
    s = fp.FlowSample(path)
    s.auto_compensate()
    assert s.comp_channels == FLUOR
    assert _rel_err(s) < 1e-5


def test_numbers_outside_the_parameter_range_are_not_guessed(tmp_path):
    """Only a reference to an existing parameter is a parameter number:
    7, 8, 9 on a 6-parameter file stay unmatched (nothing is compensated, as
    before), rather than wrapping onto some other channel."""
    path = _fcs(tmp_path / 'bad.fcs', f'3,7,8,9,{_values(M)}')
    channels, _matrix = fp.read_compensation_matrix(path)
    assert channels == ['7', '8', '9']
    s = fp.FlowSample(path)
    s.auto_compensate()
    assert not s.is_compensated()


# ── percent matrices ────────────────────────────────────────────────────────

def _plain_sample():
    df = pd.DataFrame(TRUE @ M, columns=FLUOR)
    return fp.FlowSample.from_dataframe(df, name='s')


def test_percent_csv_is_read_as_fractions(tmp_path, caplog):
    """Before: diagonal 100 read back as is, and the compensated data came
    out at 0.01x the truth."""
    path = str(tmp_path / 'percent.csv')
    fp.write_compensation_matrix(path, M * 100, FLUOR)
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        channels, matrix = fp.read_compensation_matrix(path)
    np.testing.assert_allclose(matrix, M, rtol=1e-12)
    assert 'PERCENT' in caplog.text
    s = _plain_sample()
    s.manual_compensate(matrix, channels)
    assert _rel_err(s) < 1e-9


def test_percent_matrix_given_directly_is_converted(caplog):
    """manual_compensate / a .wsp sample matrix reach _apply_comp without a
    reader: the same conversion applies there."""
    s = _plain_sample()
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        s.manual_compensate(M * 100, FLUOR)
    assert 'PERCENT' in caplog.text
    assert _rel_err(s) < 1e-9
    np.testing.assert_allclose(s.comp_matrix, M, rtol=1e-12)


def test_percent_spill_keyword_is_converted_in_both_readers(tmp_path):
    path = _fcs(tmp_path / 'pct.fcs', f'3,{",".join(FLUOR)},{_values(M * 100)}')
    s = fp.FlowSample(path)
    s.auto_compensate()
    assert _rel_err(s) < 1e-5
    _channels, matrix = fp.read_compensation_matrix(path)
    np.testing.assert_allclose(matrix, M, rtol=1e-12)


def test_a_diagonal_neither_1_nor_100_is_kept_but_warned_about(tmp_path, caplog):
    odd = M.copy()
    np.fill_diagonal(odd, 10.0)
    path = str(tmp_path / 'odd.csv')
    fp.write_compensation_matrix(path, odd, FLUOR)
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        _channels, matrix = fp.read_compensation_matrix(path)
    np.testing.assert_array_equal(matrix, odd)
    assert 'neither 1' in caplog.text


def test_a_fraction_matrix_is_left_alone(tmp_path, caplog):
    """Negative control: the ordinary file raises no unit warning."""
    path = str(tmp_path / 'ok.csv')
    fp.write_compensation_matrix(path, M, FLUOR)
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        _channels, matrix = fp.read_compensation_matrix(path)
    np.testing.assert_array_equal(matrix, M)
    assert 'PERCENT' not in caplog.text and 'neither 1' not in caplog.text


# ── CSV layouts ─────────────────────────────────────────────────────────────

def _write(path, lines):
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(path)


def _row(v):
    return ','.join(repr(float(x)) for x in v)


def test_rows_in_another_order_are_put_in_header_order(tmp_path):
    """Before: the FL2-A row (0, 0.005, 1) was taken as FL4-A's."""
    path = _write(tmp_path / 'reordered.csv', [
        ',FL4-A,FL5-A,FL2-A',
        f'FL2-A,{_row(M[2])}',
        f'FL4-A,{_row(M[0])}',
        f'FL5-A,{_row(M[1])}'])
    channels, matrix = fp.read_compensation_matrix(path)
    assert channels == FLUOR
    np.testing.assert_array_equal(matrix, M)


def test_row_labels_that_are_not_the_header_channels_are_refused(tmp_path):
    path = _write(tmp_path / 'mismatch.csv', [
        ',FL4-A,FL5-A,FL2-A',
        f'FL4-A,{_row(M[0])}',
        f'FL5-A,{_row(M[1])}',
        f'FL1-A,{_row(M[2])}'])
    with pytest.raises(fp.CompensationError, match='row labels'):
        fp.read_compensation_matrix(path)


def test_flowjo_style_header_without_corner_cell(tmp_path):
    """Before: CompensationError 'not square (shape (3, 2))'."""
    path = _write(tmp_path / 'flowjo.csv',
                  ['FL4-A,FL5-A,FL2-A'] + [_row(r) for r in M])
    channels, matrix = fp.read_compensation_matrix(path)
    assert channels == FLUOR
    np.testing.assert_array_equal(matrix, M)


def test_blank_corner_over_unlabelled_rows(tmp_path):
    path = _write(tmp_path / 'corner.csv',
                  [',FL4-A,FL5-A,FL2-A'] + [_row(r) for r in M])
    channels, matrix = fp.read_compensation_matrix(path)
    assert channels == FLUOR
    np.testing.assert_array_equal(matrix, M)


def test_a_short_row_is_reported_as_such(tmp_path):
    """A row missing a value used to be skipped, and the error then blamed
    the matrix shape."""
    path = _write(tmp_path / 'short.csv', [
        ',FL4-A,FL5-A,FL2-A',
        f'FL4-A,{_row(M[0])}',
        f'FL5-A,{_row(M[1][:2])}',
        f'FL2-A,{_row(M[2])}'])
    with pytest.raises(fp.CompensationError, match='row 3 has 2 value'):
        fp.read_compensation_matrix(path)


def test_headerless_csv_still_returns_no_channels(tmp_path):
    path = _write(tmp_path / 'bare.csv', [_row(r) for r in M])
    channels, matrix = fp.read_compensation_matrix(path)
    assert channels is None
    np.testing.assert_array_equal(matrix, M)


# ── .wsp ────────────────────────────────────────────────────────────────────

def test_wsp_without_a_matrix_reads_as_none(tmp_path):
    """Documented: (None, None) for a legitimate file with no spillover.
    Before: RuntimeError('No matrices available.')."""
    path = str(tmp_path / 'nomatrix.wsp')
    fp.WspWriter().write(path)
    assert fp.read_compensation_matrix(path) == (None, None)


def test_wsp_matrix_round_trips(tmp_path):
    path = str(tmp_path / 'm.wsp')
    fp.write_compensation_matrix(path, M, FLUOR)
    channels, matrix = fp.read_compensation_matrix(path)
    assert channels == FLUOR
    np.testing.assert_array_equal(matrix, M)
