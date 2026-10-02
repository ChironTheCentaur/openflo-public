"""A spillover matrix that covers only some detectors must say so.

`WspReader._extract_matrix` builds a Gating-ML v2 matrix row by row from
`<spillover parameter="src">` elements, starting from the identity. Rows for
channels the workspace never mentions therefore stay identity, which means
"this detector has no spillover" — so spillover from every other channel leaks
into it uncorrected, producing populations that are not there.

The acceptance guard was `seen >= n`, and `seen` counts COEFFICIENTS rather
than channels. Measured: a four-channel matrix in which only two channels
declared spillover contributed eight coefficients, cleared the bar, and was
accepted with the remaining two detectors left uncompensated and nothing said.

A partial matrix is still accepted — the declared rows are real and better
than no compensation at all — but the uncovered channels are now named in a
warning. The two ends of the range are unchanged: a complete matrix is silent,
and a workspace with no spillover values at all is still rejected outright.
"""
import logging
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from openflo.pipeline import WspReader

CHANNELS = ['FITC-A', 'PE-A', 'APC-A', 'PerCP-A']


def _matrix_node(spillover_xml, channels=CHANNELS):
    params = ''.join(f'<parameter name="{c}"/>' for c in channels)
    return ET.fromstring(f'<matrix name="M">{params}{spillover_xml}</matrix>')


def _spillover_for(sources, channels=CHANNELS, leak=0.08):
    return ''.join(
        f'<spillover parameter="{src}">' + ''.join(
            f'<coefficient parameter="{dst}" '
            f'value="{1.0 if src == dst else leak}"/>'
            for dst in channels) + '</spillover>'
        for src in sources)


def _read(spillover_xml, channels=CHANNELS):
    reader = WspReader.__new__(WspReader)
    reader.matrices = {}
    reader._extract_matrix(_matrix_node(spillover_xml, channels))
    return reader.matrices.get('M')


def _identity_rows(matrix, channels=CHANNELS):
    eye = np.eye(len(channels))
    return [c for i, c in enumerate(channels)
            if np.allclose(matrix[i], eye[i])]


def test_a_partial_matrix_names_the_uncompensated_channels(caplog):
    with caplog.at_level(logging.WARNING):
        got = _read(_spillover_for(CHANNELS[:2]))
    assert got is not None, 'the usable rows were discarded'
    assert _identity_rows(got['matrix']) == ['APC-A', 'PerCP-A']

    warned = ' '.join(r.getMessage() for r in caplog.records)
    assert 'APC-A' in warned and 'PerCP-A' in warned, (
        'two detectors were left uncompensated without being named: '
        f'{warned!r}')


def test_a_complete_matrix_warns_about_nothing(caplog):
    """The warning must not fire for the ordinary case, or it is noise."""
    with caplog.at_level(logging.WARNING):
        got = _read(_spillover_for(CHANNELS))
    assert got is not None
    assert _identity_rows(got['matrix']) == []
    assert not [r for r in caplog.records if 'UNCOMPENSATED' in r.getMessage()]


def test_no_spillover_at_all_is_still_rejected():
    """An empty matrix must not become a silent identity 'compensation'."""
    assert _read('') is None


def test_the_declared_rows_are_the_declared_values():
    """The warning is not a substitute for reading the file correctly."""
    got = _read(_spillover_for(CHANNELS[:2], leak=0.13))
    matrix = got['matrix']
    off_diagonal = matrix[0][1:]
    assert np.allclose(off_diagonal, 0.13), (
        f'declared coefficients were not parsed: {off_diagonal}')
    assert matrix[0][0] == pytest.approx(1.0)
