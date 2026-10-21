"""FCS writer quirks that other readers accept now load, with a note.

Three quirks real instruments and exporters write, each built here from a
clean synthetic file (generic channels FL1-A to FL3-A): an $ENDDATA one byte
past the data, a HEADER that disagrees with TEXT on where the data starts,
and a second data set chained by $NEXTDATA. A damaged file still fails, with
a message that says so.
"""
import re

import numpy as np
import pytest

from openflo.pipeline import FcsParseError, FlowSample, plain_fcs_problem
from tests._fcs_writer import write_fcs

NAMES = ['FL1-A', 'FL2-A', 'FL3-A']


def _data(seed=0, n=200):
    return np.random.default_rng(seed).integers(1, 1000, (n, 3))


def _clean(tmp_path, name='s1', seed=0, extra=None):
    data = _data(seed)
    p = write_fcs(tmp_path / f'{name}.fcs', data, NAMES, extra=extra)
    return p, data


def _header_field(raw, i):
    return int(raw[10 + 8 * i:18 + 8 * i])


def _set_header_field(raw, i, value):
    return raw[:10 + 8 * i] + str(value).rjust(8).encode() + raw[18 + 8 * i:]


def _off_by_one(tmp_path):
    p, data = _clean(tmp_path)
    raw = p.read_bytes()
    end = _header_field(raw, 3)
    old = f'$ENDDATA/{end}/'.encode()
    new = f'$ENDDATA/{end + 1}/'.encode()
    assert len(old) == len(new)            # keeps every offset in place
    raw = _set_header_field(raw.replace(old, new), 3, end + 1)
    p.write_bytes(raw)
    return p, data


def _discrepant(tmp_path):
    p, data = _clean(tmp_path)
    raw = p.read_bytes()
    raw = _set_header_field(raw, 2, _header_field(raw, 2) + 4)
    raw = _set_header_field(raw, 3, _header_field(raw, 3) + 4)
    p.write_bytes(raw)
    return p, data


def _two_data_sets(tmp_path):
    first, data = _clean(tmp_path, 'a', 0, extra={'$NEXTDATA': '0' * 10})
    second, _ = _clean(tmp_path, 'b', 1)
    raw = first.read_bytes()
    raw = raw.replace(b'$NEXTDATA/' + b'0' * 10 + b'/',
                      b'$NEXTDATA/' + str(len(raw)).rjust(10, '0').encode() + b'/')
    first.write_bytes(raw + second.read_bytes())
    return first, data


def _events(s):
    return s.data[NAMES].to_numpy()


def test_a_clean_file_loads_with_no_quirk_note(tmp_path):
    p, data = _clean(tmp_path)
    s = FlowSample(str(p))
    assert np.array_equal(_events(s), data)
    assert not [w for w in s.load_warnings if 'data' in w.lower()]


@pytest.mark.parametrize('make, note', [
    (_off_by_one, 'one byte off'),
    (_discrepant, 'HEADER and TEXT disagree'),
    (_two_data_sets, 'more than one data set'),
])
def test_a_quirky_file_loads_its_true_events_and_says_why(tmp_path, make, note):
    p, data = make(tmp_path)
    s = FlowSample(str(p))
    assert np.array_equal(_events(s), data)
    assert any(note in w for w in s.load_warnings), s.load_warnings


def test_the_quirks_are_real_without_the_tolerant_reader(tmp_path):
    # Each built file must be refused by a plain reader, or the test above
    # would pass on files that were never quirky.
    from openflo import fcs_read
    for make, err in ((_off_by_one, 'FCSParsingError'),
                      (_discrepant, 'DataOffsetDiscrepancyError'),
                      (_two_data_sets, 'MultipleDataSetsError')):
        d = tmp_path / make.__name__
        d.mkdir()
        p, _ = make(d)
        with pytest.raises(Exception) as ei:
            fcs_read.FlowData(str(p))
        assert type(ei.value).__name__ == err


def test_a_truncated_file_still_fails_and_says_cut_short(tmp_path):
    p, _ = _clean(tmp_path)
    raw = p.read_bytes()
    p.write_bytes(raw[:len(raw) - 100])
    with pytest.raises(FcsParseError, match='cut short'):
        FlowSample(str(p))


def test_quirk_messages_name_the_quirk_not_damage():
    from flowio.exceptions import (
        DataOffsetDiscrepancyError,
        FCSParsingError,
        MultipleDataSetsError,
    )
    assert 'one byte off' in plain_fcs_problem(
        FCSParsingError('x.fcs reports a data offset that is off by 1.'))
    assert 'disagree' in plain_fcs_problem(DataOffsetDiscrepancyError('x'))
    assert 'more than one data set' in plain_fcs_problem(
        MultipleDataSetsError('x'))
    assert not re.search('one byte|disagree|data set',
                         plain_fcs_problem(ValueError('bad')))
