"""read_compensation_matrix must find the spillover an FCS file carries.

Acquisition software writes the spillover matrix into the FCS TEXT segment as
``N,ch1,...,chN,v11,v12,...,vNN``. BD FACSDiva uses the bare, non-standard
keyword ``SPILL`` (no ``$``); FCS 3.1 defines ``$SPILLOVER``. FlowIO strips
every ``$`` and lower-cases every keyword, so both arrive as ``spill`` /
``spillover``.

read_compensation_matrix looked the keyword up only as ``SPILL``,
``SPILLOVER``, ``$SPILL`` and ``$SPILLOVER`` -- spellings FlowIO never
produces -- so it returned ``(None, None)`` for every FCS file, while
FlowSample.auto_compensate (which also tries the lower-case keys) applied the
same file's matrix. ``--export-wsp`` then wrote a workspace with no spillover
matrix, and the compensation editor skipped the embedded matrix.

The fixture writes the keyword the way the acquisition software does and
checks the raw TEXT bytes, so the test cannot pass on a fixture that drifted
to a different spelling.
"""
from __future__ import annotations

import flowio
import numpy as np
import pytest

import openflo.pipeline as fp

_DETECTORS = ['FL1-A', 'FL2-A', 'FL3-A']
_CHANNELS = ['FSC-A', 'SSC-A'] + _DETECTORS

# Row = source detector, column = destination detector. The text form mixes
# bare integers with full-precision decimals, as acquisition software writes
# it: no spaces, no trailing comma.
_MATRIX = np.array([
    [1.0, 0.0, 0.0012345678901234567],
    [0.0, 1.0, 0.0048219377620193840],
    [0.0, 0.0219730041561829370, 1.0],
])
_SPILL_TEXT = ('3,FL1-A,FL2-A,FL3-A,'
               '1,0,0.0012345678901234567,'
               '0,1,0.004821937762019384,'
               '0,0.021973004156182937,1')


def _write_fcs(path, keyword=None, diva_framing=False):
    rng = np.random.default_rng(0)
    events = rng.uniform(100.0, 5000.0, size=(200, len(_CHANNELS)))
    meta = {keyword: _SPILL_TEXT} if keyword else None
    with open(path, 'wb') as f:
        flowio.create_fcs(f, events.astype(np.float32).ravel().tolist(),
                          _CHANNELS, metadata_dict=meta)
    if diva_framing:
        # Frame the TEXT segment as BD FACSDiva does: an FCS3.0 header and a
        # form-feed delimiter. Same byte length, so every offset stays valid.
        with open(path, 'r+b') as f:
            header = f.read(58)
            start, stop = int(header[10:18]), int(header[18:26])
            f.seek(start)
            text = f.read(stop - start + 1)
            assert text[:1] == b'/' and b'//' not in text and b'\x0c' not in text
            f.seek(0)
            f.write(b'FCS3.0')
            f.seek(start)
            f.write(text.replace(b'/', b'\x0c'))
    return str(path)


def _raw_text_keywords(path):
    """Keyword -> value from the TEXT segment, as bytes on disk (no FlowIO)."""
    with open(path, 'rb') as f:
        header = f.read(58)
        start, stop = int(header[10:18]), int(header[18:26])
        f.seek(start)
        raw = f.read(stop - start + 1)
    delim = raw[:1]
    tokens = raw[1:].rstrip(delim).split(delim)
    return dict(zip(tokens[0::2], tokens[1::2], strict=True))


# keyword passed to the writer, keyword expected on disk, Diva framing
_FORMS = [
    pytest.param('SPILL', b'SPILL', True, id='diva-SPILL'),
    pytest.param('SPILL', b'SPILL', False, id='SPILL'),
    pytest.param('SPILLOVER', b'$SPILLOVER', False, id='fcs31-$SPILLOVER'),
]


@pytest.mark.parametrize('keyword, on_disk, diva', _FORMS)
def test_fixture_writes_the_keyword_as_acquisition_software_does(
        tmp_path, keyword, on_disk, diva):
    path = _write_fcs(tmp_path / 'spill.fcs', keyword, diva)
    kw = _raw_text_keywords(path)
    assert kw.get(on_disk) == _SPILL_TEXT.encode()
    with open(path, 'rb') as f:
        head = f.read(6)
    assert head == (b'FCS3.0' if diva else b'FCS3.1')
    assert flowio.FlowData(path).event_count == 200
    others = [k for k in kw if b'SPILL' in k.upper() and k != on_disk]
    assert others == [], f'fixture wrote extra spillover keywords: {others}'


@pytest.mark.parametrize('keyword, on_disk, diva', _FORMS)
def test_read_compensation_matrix_reads_fcs_spillover(
        tmp_path, keyword, on_disk, diva):
    path = _write_fcs(tmp_path / 'spill.fcs', keyword, diva)
    channels, matrix = fp.read_compensation_matrix(path)
    assert matrix is not None, (
        f'read_compensation_matrix returned (None, None) for an FCS whose '
        f'TEXT carries {on_disk.decode()}')
    assert channels == _DETECTORS
    np.testing.assert_array_equal(matrix, _MATRIX)


@pytest.mark.parametrize('keyword, on_disk, diva', _FORMS)
def test_read_compensation_matrix_agrees_with_auto_compensate(
        tmp_path, keyword, on_disk, diva):
    path = _write_fcs(tmp_path / 'spill.fcs', keyword, diva)
    sample = fp.FlowSample(path)
    sample.auto_compensate()
    # Response check on the reference side: auto_compensate did apply it.
    assert sample.comp_channels == _DETECTORS
    channels, matrix = fp.read_compensation_matrix(path)
    assert matrix is not None, (
        'read_compensation_matrix found no spillover in a file '
        'auto_compensate compensated')
    assert channels == sample.comp_channels
    np.testing.assert_array_equal(matrix, sample.comp_matrix)


def test_fcs_without_spillover_still_reads_as_none(tmp_path):
    # Negative control: the fix must not invent a matrix.
    path = _write_fcs(tmp_path / 'plain.fcs')
    assert not [k for k in _raw_text_keywords(path) if b'SPILL' in k.upper()]
    assert fp.read_compensation_matrix(path) == (None, None)


# A second, different matrix for $SPILLOVER, with the same text length as
# _SPILL_TEXT so the two keyword/value pairs can swap places in TEXT.
_OTHER_TEXT = _SPILL_TEXT.replace('0.0012345678901234567', '0.0300000000000000000')
assert len(_OTHER_TEXT) == len(_SPILL_TEXT) and _OTHER_TEXT != _SPILL_TEXT


def _write_both(path, spill_first):
    """An FCS carrying BOTH bare SPILL (_SPILL_TEXT) and $SPILLOVER
    (_OTHER_TEXT). flowio always writes $SPILLOVER first; ``spill_first``
    swaps the two adjacent pairs in place (same byte length, offsets valid)."""
    rng = np.random.default_rng(0)
    events = rng.uniform(100.0, 5000.0, size=(200, len(_CHANNELS)))
    with open(path, 'wb') as f:
        flowio.create_fcs(f, events.astype(np.float32).ravel().tolist(), _CHANNELS,
                          metadata_dict={'SPILL': _SPILL_TEXT, 'SPILLOVER': _OTHER_TEXT})
    raw = bytearray(open(path, 'rb').read())
    a = b'$SPILLOVER/' + _OTHER_TEXT.encode() + b'/'
    b = b'SPILL/' + _SPILL_TEXT.encode() + b'/'
    at = raw.find(a + b)
    assert at > 0, 'flowio no longer writes the two keywords adjacent, $SPILLOVER first'
    if spill_first:
        raw[at:at + len(a + b)] = b + a
        open(path, 'wb').write(bytes(raw))
    return str(path)


@pytest.mark.parametrize('spill_first', [False, True], ids=['SPILLOVER-first', 'SPILL-first'])
def test_spill_wins_over_spillover_whatever_the_text_order(tmp_path, spill_first):
    # FlowJo 10.10.2 adopts SPILL by NAME when a file carries both, with either
    # TEXT order (measured on synthetic files). OpenFlo must read the same matrix, in both
    # readers, or its export disagrees with what FlowJo applies.
    path = _write_both(tmp_path / 'both.fcs', spill_first)
    kw = _raw_text_keywords(path)
    assert kw[b'SPILL'] == _SPILL_TEXT.encode() and kw[b'$SPILLOVER'] == _OTHER_TEXT.encode()
    order = [k for k in kw if b'SPILL' in k.upper()]
    assert order == ([b'SPILL', b'$SPILLOVER'] if spill_first else [b'$SPILLOVER', b'SPILL'])
    channels, matrix = fp.read_compensation_matrix(path)
    assert channels == _DETECTORS
    np.testing.assert_array_equal(matrix, _MATRIX)
    sample = fp.FlowSample(path)
    sample.auto_compensate()
    np.testing.assert_array_equal(sample.comp_matrix, _MATRIX)
