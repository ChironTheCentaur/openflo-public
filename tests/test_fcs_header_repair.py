"""HEADER and TEXT faults that made OpenFlo refuse an FCS file other
programs open now load, with a note saying what was repaired
(openflo.fcs_repair, through fcs_quirks.read_fcs_tolerant).

Each fault is byte-patched into a clean synthetic file (generic channels
FL1-A to FL3-A). Real instrument files differ in detail; these are the
faults as the FCS 3.1 / 3.2 standard and NIST fireflow's list of common
issues describe them. A clean file gets no note, and a file whose fault
cannot be confirmed from its own sizes is still refused.
"""
import io
import os
import re
from contextlib import redirect_stdout

import flowio
import numpy as np
import pytest

from openflo import fcs_read
from openflo.fcs_quirks import LMD_FCS3, lmd_data_set, read_fcs_tolerant
from openflo.fcs_repair import repair_fcs
from openflo.pipeline import FcsParseError, FlowSample
from tests._fcs_writer import write_fcs, write_lmd

NAMES = ['FL1-A', 'FL2-A', 'FL3-A']
# FlowIO warns as it gives up on a $PnB of 10; the repair reads the file.
pytestmark = pytest.mark.filterwarnings(
    'ignore:Non-standard bit width:UserWarning')


def _data(seed=0, n=200):
    return np.random.default_rng(seed).integers(1, 1000, (n, 3))


def _clean(tmp_path, data=None, **kw):
    data = _data() if data is None else data
    return write_fcs(tmp_path / 's1.fcs', data, NAMES, **kw), data


def _field(raw, i):
    return int(raw[10 + 8 * i:18 + 8 * i])


def _set_field(raw, i, text):
    return raw[:10 + 8 * i] + str(text).rjust(8).encode() + raw[18 + 8 * i:]


def _grow_text(raw, old, new):
    """`old` -> `new` inside TEXT, the extra bytes taken from the padding
    write_fcs leaves between TEXT and DATA, so DATA stays in place."""
    end, grow = _field(raw, 1), len(new) - len(old)
    assert grow >= 0 and raw.count(old) == 1
    raw = raw.replace(old, new)
    raw = raw[:end + 1 + grow] + raw[end + 1 + 2 * grow:]
    return _set_field(raw, 1, end + grow)


def _patched(tmp_path, fix, **kw):
    p, data = _clean(tmp_path, **kw)
    p.write_bytes(fix(p.read_bytes()))
    return p, data


def _blank_header(raw):
    return _set_field(_set_field(raw, 2, ''), 3, '')


def _no_text_offsets(raw):
    for key in (b'$BEGINDATA/', b'$ENDDATA/'):
        raw = raw.replace(key, b'X' + key[1:])      # same length: not $...
    return raw


def _pseudo_empty_analysis(raw):
    return _grow_text(raw, b'$ENDANALYSIS/0/', b'$ENDANALYSIS/-1/')


def _blank_analysis(raw):
    return raw.replace(b'$BEGINANALYSIS/0/', b'$BEGINANALYSIS/ /')


def _ten_bit_widths(raw):
    assert raw.count(b'B/16/') == 3
    return raw.replace(b'B/16/', b'B/10/')


def _no_tot(raw):
    return raw.replace(b'$TOT/200/', b'XTOT/200/')


def _high_bits(data):
    """`data` (10-bit values) with junk in bits 10-15, which the standard
    says a reader must ignore when $PnR is 1024."""
    junk = np.random.default_rng(9).integers(0, 64, data.shape) << 10
    return data | junk


FAULTS = {
    'blank HEADER data offsets': (
        _blank_header, "HEADER does not say where the data is"),
    'no $BEGINDATA/$ENDDATA': (
        _no_text_offsets, "TEXT does not say where the data is"),
    'ANALYSIS 0,-1': (
        _pseudo_empty_analysis, "no usable position for the ANALYSIS"),
    'ANALYSIS blank': (
        _blank_analysis, "no usable position for the ANALYSIS"),
    '$PnB 10 in 16-bit words': (
        _ten_bit_widths, "10 bits per value ($PnB)"),
    'no $TOT': (_no_tot, "($TOT) is missing; 200 events were counted"),
}


def _make(tmp_path, name):
    fix, _ = FAULTS[name]
    if fix is _ten_bit_widths:
        data = _data()
        p, _ = _patched(tmp_path, fix, data=_high_bits(data))
        return p, data
    return _patched(tmp_path, fix)


def _events(s):
    return s.raw[NAMES].to_numpy()


def test_a_clean_file_loads_with_no_note_and_needs_no_repair(tmp_path):
    p, data = _clean(tmp_path)
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert s.load_warnings == []
    assert repair_fcs(str(p)) is None


@pytest.mark.parametrize('name', sorted(FAULTS))
def test_each_fault_is_refused_by_flowio_alone(tmp_path, name):
    # Without the repair the file does not load: FlowIO raises, or (for a
    # $PnB that is not 8, 16 or 32) returns no events at all.
    p, _ = _make(tmp_path, name)
    try:
        fcs = fcs_read.FlowData(str(p))
    except Exception:  # noqa: BLE001 - any refusal will do
        return
    assert fcs.events is None


@pytest.mark.parametrize('name', sorted(FAULTS))
def test_each_fault_loads_its_true_events_with_one_plain_note(tmp_path, name):
    p, data = _make(tmp_path, name)
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert len(s.load_warnings) == 1, s.load_warnings
    note = s.load_warnings[0]
    assert note.startswith('s1.fcs: ') and FAULTS[name][1] in note, note


@pytest.mark.parametrize('name', sorted(FAULTS))
def test_the_file_inspectors_read_it_too(tmp_path, name):
    # inspect_fcs and the FCS inspector read with plain flowio.FlowData.
    from openflo import inspect_fcs
    from openflo.ui_inspect import read_fcs_metadata
    p, _ = _make(tmp_path, name)
    notes = []
    fcs = read_fcs_tolerant(str(p), flowio.FlowData, notes)
    assert fcs.event_count == 200 and len(notes) == 1
    assert read_fcs_metadata(str(p))['text']['p1n'] == 'FL1-A'
    out = io.StringIO()
    with redirect_stdout(out):
        inspect_fcs.main([str(p)])
    note = next(ln for ln in out.getvalue().splitlines()
                if ln.startswith('Note: s1.fcs: '))
    assert FAULTS[name][1] in note


def test_the_file_on_disk_is_not_changed(tmp_path):
    p, _ = _make(tmp_path, 'no $TOT')
    before = p.read_bytes()
    FlowSample(str(p))
    assert p.read_bytes() == before


def test_a_large_file_header_gets_zeros_as_the_standard_writes_them(tmp_path):
    # A DATA end past byte 99,999,999 cannot fit the 8-digit HEADER: the
    # standard writes 0 there and the TEXT value is used.
    def fix(raw):
        end = re.search(rb'\$ENDDATA/\d+/', raw).group(0)
        return _blank_header(_grow_text(raw, end, b'$ENDDATA/123456789/'))
    p, _ = _patched(tmp_path, fix)
    src, notes = repair_fcs(str(p))
    src.seek(26)
    assert src.read(16) == b'       0       0'
    assert 'HEADER does not say where the data is' in notes[0]


def test_the_last_value_keeps_its_end_when_the_final_delimiter_is_missing(
        tmp_path):
    # TEXT rewritten for a missing $TOT keeps an escaped delimiter in a
    # value, and closes a last value whose delimiter the writer left out.
    def fix(raw):
        raw = _no_tot(raw)
        end = _field(raw, 1)
        assert raw[end:end + 1] == b'/'
        return _set_field(raw[:end] + b' ' + raw[end + 1:], 1, end - 1)
    p, data = _patched(tmp_path, fix, extra={'$COM': 'a//b'})
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert s.metadata['com'] == 'a/b'
    assert s.metadata['endstext'] == '0'
    assert s.metadata['tot'] == '200'


def test_a_bit_width_the_data_size_does_not_confirm_is_still_refused(
        tmp_path):
    # $TOT says 199 events: 10-bit values in 16-bit words would need 200,
    # so the widths are not changed and the file is not read.
    p, _ = _patched(tmp_path, lambda raw: _ten_bit_widths(raw).replace(
        b'$TOT/200/', b'$TOT/199/'))
    assert repair_fcs(str(p)) is None
    with pytest.raises(FcsParseError, match=re.escape(
            "s1.fcs: its bits per value $P1B is 10, and the data size does "
            "not say how the values are stored")):
        FlowSample(str(p))


def test_the_first_error_is_reported_when_the_repair_does_not_help(tmp_path):
    # No $TOT, and the data cut short: the count can be repaired but the
    # file still cannot be read, so the message is about the first fault.
    p, _ = _make(tmp_path, 'no $TOT')
    raw = p.read_bytes()
    p.write_bytes(raw[:-61])
    with pytest.raises(FcsParseError, match=re.escape('$TOT')):
        FlowSample(str(p))


def test_a_repaired_file_without_a_range_gets_its_range_filled(tmp_path):
    # A missing $PnR is filled from $PnB (fcs_range_fill) after the repair.
    p, data = _patched(tmp_path, lambda raw: _blank_header(
        raw.replace(b'$P2R/1024/', b'XP2R/1024/')))
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert any('HEADER does not say' in w for w in s.load_warnings)
    # said once, though TEXT was parsed again for the repair
    assert [w for w in s.load_warnings if '$P2R' in w] == [
        "$P2R (range of FL2-A) is missing: taken as 2^16 = 65,536 from its "
        "$P2B"]


def test_a_file_too_big_to_point_at_new_text_is_left_as_it_is(tmp_path):
    p, _ = _make(tmp_path, 'no $TOT')
    with open(p, 'r+b') as fh:                 # sparse: no disk is used
        fh.truncate(100_000_000)
    assert repair_fcs(str(p)) is None


def test_the_second_data_set_of_an_lmd_file_is_repaired_at_its_offset(
        tmp_path):
    first, second = _data(1), _data(2).astype(float)
    p = write_lmd(tmp_path / 's1.lmd', first, second, NAMES,
                  second_kw={'drop': ('$TOT',)})
    notes = []
    fcs = read_fcs_tolerant(str(p), fcs_read.FlowData, notes)
    assert lmd_data_set(fcs) == LMD_FCS3
    assert fcs.event_count == 200
    assert notes == [f"{os.path.basename(str(p))}: its event count ($TOT) "
                     "is missing; 200 events were counted from the size of "
                     "its data section"]


def test_a_repaired_file_is_still_read_past_a_writer_quirk(tmp_path):
    # No $TOT and an $ENDDATA one byte past the data: the count is taken
    # from the whole events, and the off-by-one note is kept too.
    def fix(raw):
        end = _field(raw, 3)
        old = f'$ENDDATA/{end}/'.encode()
        raw = _no_tot(raw).replace(old, f'$ENDDATA/{end + 1}/'.encode())
        return _set_field(raw, 3, end + 1) + b'\0'
    p, data = _patched(tmp_path, fix)
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert len(s.load_warnings) == 2, s.load_warnings
    assert s.load_warnings[0].startswith('s1.fcs: its event count ($TOT)')
    assert s.load_warnings[1].startswith('s1.fcs: its data end is recorded')


def test_the_text_reader_keeps_an_escaped_delimiter_inside_a_value():
    # A doubled delimiter is one literal delimiter in the value (FCS 3.1
    # 3.2.9), not the end of the value.
    from openflo.fcs_repair import _Text
    text = _Text(b'/$COM/a//b/$TOT/5/')
    assert text.get('COM') == 'a/b'
    assert text.int('TOT') == 5


# ── TEXT's last value with no closing delimiter ──────────────────────────────
# FlowIO drops the last byte of TEXT, which is meant to be the closing
# delimiter. When the HEADER ends TEXT too early, or the writer left the
# delimiter out, that byte is the last value's last character: the file
# loads with a wrong value and no warning. Here the last keyword is $P3N.

def _p3n_last(raw, pair=b'$P3N/FL3-A/'):
    """The clean file with `pair` ($P3N) moved to the end of TEXT (same
    length)."""
    t0, t1 = _field(raw, 0), _field(raw, 1)
    body = raw[t0:t1 + 1]
    assert body.count(pair) == 1
    return raw[:t0] + body.replace(pair, b'') + pair + raw[t1 + 1:]


def _text_end_short(by):
    def fix(raw):
        raw = _p3n_last(raw)
        return _set_field(raw, 1, _field(raw, 1) - by)
    return fix


def _no_closing_delimiter(raw):
    raw = _p3n_last(raw)
    end = _field(raw, 1)
    return _set_field(raw[:end] + b' ' + raw[end + 1:], 1, end - 1)


def _names(s):
    return list(s.channel_names)


@pytest.mark.parametrize('by', [1, 3])
def test_a_text_end_short_of_its_delimiter_is_read_up_to_it(tmp_path, by):
    p, data = _patched(tmp_path, _text_end_short(by))
    plain = fcs_read.FlowData(str(p))
    assert plain.channels[3]['pnn'] != 'FL3-A'    # what FlowIO alone reads
    s = FlowSample(str(p))
    assert _names(s) == NAMES
    np.testing.assert_array_equal(_events(s), data)
    assert s.load_warnings == [
        f"s1.fcs: its HEADER ends TEXT {by} byte(s) too early, before its "
        "closing delimiter; TEXT was read up to the delimiter, so the last "
        "keyword ($P3N) keeps its full value"]


def test_a_missing_closing_delimiter_keeps_the_value_as_written(tmp_path):
    p, data = _patched(tmp_path, _no_closing_delimiter)
    assert fcs_read.FlowData(str(p)).channels[3]['pnn'] == 'FL3-'
    s = FlowSample(str(p))
    assert _names(s) == NAMES
    np.testing.assert_array_equal(_events(s), data)
    assert s.load_warnings == [
        "s1.fcs: its TEXT has no closing delimiter; the last keyword "
        "($P3N) was kept as written"]


def test_a_delimiter_inside_the_data_is_not_taken_as_the_end_of_text(
        tmp_path):
    # TEXT runs into the padding with no delimiter, and the first DATA
    # byte is a '/': it is data, so the value is kept as written.
    data = _data()
    data[0, 0] = ord('/')

    def fix(raw):
        raw = _p3n_last(raw)
        end, start = _field(raw, 1), _field(raw, 2)
        raw = raw[:end] + b'Z' * (start - end) + raw[start:]
        return _set_field(raw, 1, start - 2)
    p, _ = _patched(tmp_path, fix, data=data)
    s = FlowSample(str(p))
    assert s.load_warnings == [
        "s1.fcs: its TEXT has no closing delimiter; the last keyword "
        "($P3N) was kept as written"]
    assert s.channel_names[2].startswith('FL3-AZZ')


def test_text_padded_after_its_delimiter_gets_no_note(tmp_path):
    p, data = _patched(tmp_path, lambda raw: _set_field(
        _p3n_last(raw), 1, _field(raw, 2) - 1))
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert _names(s) == NAMES and s.load_warnings == []
    assert repair_fcs(str(p), text_end_only=True) is None


def test_a_text_only_read_gets_the_full_last_value(tmp_path):
    import functools
    p, _ = _patched(tmp_path, _text_end_short(1))
    notes = []
    fcs = read_fcs_tolerant(
        str(p), functools.partial(flowio.FlowData, only_text=True), notes)
    assert fcs.text['p3n'] == 'FL3-A' and len(notes) == 1


def test_a_file_too_big_for_the_closed_text_says_the_value_may_be_cut(
        tmp_path):
    p, _ = _patched(tmp_path, _no_closing_delimiter)
    with open(p, 'r+b') as fh:                 # sparse: no disk is used
        fh.truncate(100_000_000)
    notes = []
    read_fcs_tolerant(str(p), fcs_read.FlowData, notes)
    assert notes == ["s1.fcs: its TEXT has no closing delimiter, so its last "
                     "keyword may be missing its last character"]


def test_the_text_end_check_alone_mends_nothing_else(tmp_path):
    # Every read checks TEXT's end; the other repairs wait for a failure.
    p, _ = _make(tmp_path, 'no $TOT')
    assert repair_fcs(str(p), text_end_only=True) is None
    assert repair_fcs(str(p)) is not None


# ── Round 2: review findings ─────────────────────────────────────────────────

def _packed_ten_bit(tmp_path, keep_tot):
    """10-bit values stored PACKED (no padding: 750 bytes for 200 x 3) with
    $PnB 10, as some writers do; $TOT kept or not."""
    data = _data()
    p, _ = _clean(tmp_path, data=data)
    raw = p.read_bytes()
    d0 = _field(raw, 2)
    bits = ''.join(format(int(v), '010b') for v in data.ravel())
    packed = int(bits, 2).to_bytes(len(bits) // 8, 'big')
    end = re.search(rb'\$ENDDATA/(\d+)/', raw)
    new_end = str(d0 + len(packed) - 1).encode()
    assert len(new_end) == len(end.group(1))
    raw = _ten_bit_widths(raw).replace(
        end.group(0), b'$ENDDATA/' + new_end + b'/')
    if not keep_tot:
        raw = _no_tot(raw)
    p.write_bytes(_set_field(raw[:d0] + packed, 3, int(new_end)))
    return p


@pytest.mark.parametrize('keep_tot', [False, True])
def test_packed_ten_bit_values_are_refused_not_read_as_words(
        tmp_path, keep_tot):
    # Without $TOT, 750 bytes fit 125 events of three 16-bit words as well
    # as 200 packed ones: nothing confirms the word, so nothing is changed.
    p = _packed_ten_bit(tmp_path, keep_tot)
    assert repair_fcs(str(p)) is None
    with pytest.raises(FcsParseError) as ei:
        FlowSample(str(p))
    assert ('$TOT' if not keep_tot else 'bits per value $P1B is 10') in str(
        ei.value)


@pytest.mark.parametrize('field', [2, 3])
def test_one_blank_header_data_field_is_repaired_too(tmp_path, field):
    p, data = _patched(tmp_path, lambda raw: _set_field(raw, field, ''))
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert len(s.load_warnings) == 1
    assert 'HEADER does not say where the data is' in s.load_warnings[0]


def test_only_enddata_missing_is_taken_from_the_header(tmp_path):
    p, data = _patched(tmp_path, lambda raw: raw.replace(
        b'$ENDDATA/', b'XENDDATA/'))
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert len(s.load_warnings) == 1
    assert 'TEXT does not say where the data is' in s.load_warnings[0]


def test_no_text_offsets_and_header_zeros_is_still_refused(tmp_path):
    # HEADER 0, 0 means "see TEXT"; with nothing in TEXT there is no data
    # position, so it is not read from byte 0.
    p, _ = _patched(tmp_path, lambda raw: _set_field(_set_field(
        _no_text_offsets(raw), 2, 0), 3, 0))
    assert repair_fcs(str(p)) is None
    with pytest.raises(FcsParseError):
        FlowSample(str(p))


def test_an_fcs2_file_gets_no_text_offsets_added(tmp_path):
    # FCS 2.0 keeps the data position in the HEADER only.
    p, data = _patched(tmp_path, lambda raw: _no_text_offsets(_no_tot(raw)),
                       version='2.0')
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert len(s.load_warnings) == 1 and '($TOT)' in s.load_warnings[0]


def test_an_fcs2_file_counts_events_from_the_header_offsets(tmp_path):
    # FlowIO reads an FCS 2.0 file where its HEADER says; a stray TEXT
    # $ENDDATA must not change the count.
    def fix(raw):
        end = re.search(rb'\$ENDDATA/(\d+)/', raw)
        wrong = str(int(end.group(1)) - 60).encode()
        assert len(wrong) == len(end.group(1))
        return _no_tot(raw).replace(end.group(0), b'$ENDDATA/' + wrong + b'/')
    p, data = _patched(tmp_path, fix, version='2.0')
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert s.load_warnings == ["s1.fcs: its event count ($TOT) is missing; "
                               "200 events were counted from the size of its "
                               "data section"]


def test_a_float_file_keeps_its_bit_counts(tmp_path):
    # $PnB does not describe float values; a float file with an odd $PnB
    # (and a blank HEADER to repair) gets no $PnB change.
    data = _data().astype(float)
    p, _ = _clean(tmp_path, data=data, datatype='F')
    p.write_bytes(_blank_header(p.read_bytes().replace(b'B/32/', b'B/20/')))
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert len(s.load_warnings) == 1
    assert 'HEADER does not say' in s.load_warnings[0]


@pytest.mark.parametrize('bits', [b'0', b'40'])
def test_a_bit_count_no_word_holds_is_refused_plainly(tmp_path, bits):
    # 0 bits, or more than the 32-bit word FlowIO reads: no repair.
    data = _data() % 256
    p, _ = _clean(tmp_path, data=data, bits=8, pnr=[256] * 3)
    p.write_bytes(p.read_bytes().replace(b'B/8/', b'B/' + bits + b'/'))
    with pytest.raises(FcsParseError, match=re.escape(
            f"its bits per value $P1B is {bits.decode()}, and the data size")):
        FlowSample(str(p))


def test_lowercase_keywords_are_repaired_too(tmp_path):
    import re as _re

    def fix(raw):
        raw = _no_tot(raw)
        return _re.sub(rb'/(\$[A-Z0-9]+)/',
                       lambda m: b'/' + m.group(1).lower() + b'/', raw)
    p, data = _patched(tmp_path, fix)
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert len(s.load_warnings) == 1 and '($TOT)' in s.load_warnings[0]


def test_the_text_reader_reads_as_flowio_does():
    from openflo.fcs_repair import _Text, _token_spans
    # A later duplicate keyword wins, as in FlowIO's dict.
    assert _Text(b'/$TOT/5/$TOT/7/').int('TOT') == 7
    # NULs after a value missing its delimiter are padding, not the value.
    assert _token_spans(b'/$A/1/$B/2\0\0\0')[-1] == (9, 10)


@pytest.mark.parametrize('by, kept', [(8, 'ABCDEFGHIJ'), (9, 'AB')])
def test_the_closing_delimiter_is_looked_for_8_bytes_on(tmp_path, by, kept):
    pair = b'$COM/ABCDEFGHIJ/'

    def fix(raw):
        raw = _p3n_last(raw, pair)
        return _set_field(raw, 1, _field(raw, 1) - by)
    p, _ = _patched(tmp_path, fix, extra={'$COM': 'ABCDEFGHIJ'})
    s = FlowSample(str(p))
    assert s.metadata['com'] == kept
    assert len(s.load_warnings) == 1
    assert ('8 byte(s) too early' if by == 8 else 'kept as written') in \
        s.load_warnings[0]


def test_an_escaped_delimiter_after_the_cut_is_part_of_the_value(tmp_path):
    pair = b'$P3N/FL3//A/'

    def fix(raw):
        raw = _p3n_last(raw, pair)
        return _set_field(raw, 1, _field(raw, 1) - 4)     # cut after 'FL3'
    names = ['FL1-A', 'FL2-A', 'FL3//A']
    q = write_fcs(tmp_path / 's1.fcs', _data(), names)
    q.write_bytes(fix(q.read_bytes()))
    s = FlowSample(str(q))
    assert s.channel_names[2] == 'FL3/A'
    assert '4 byte(s) too early' in s.load_warnings[0]


def test_junk_before_a_delimiter_is_not_joined_to_the_value(tmp_path):
    def fix(raw):
        raw = _p3n_last(raw)
        t1, d0 = _field(raw, 1), _field(raw, 2)
        mid = b'\x01\x02\x03/'
        raw = raw[:t1] + mid + b' ' * (d0 - t1 - len(mid)) + raw[d0:]
        return _set_field(raw, 1, t1 - 1)
    p, data = _patched(tmp_path, fix)
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert _names(s) == NAMES
    assert s.load_warnings == [
        "s1.fcs: its TEXT has no closing delimiter; the last keyword "
        "($P3N) was kept as written"]


def test_a_delimiter_in_data_the_header_places_is_not_the_end_of_text(
        tmp_path):
    # As the DATA test above, with no $BEGINDATA in TEXT: the HEADER's
    # data start is what tells the '/' is data.
    data = _data()
    data[0, 0] = ord('/')

    def fix(raw):
        raw = _p3n_last(raw)
        end, start = _field(raw, 1), _field(raw, 2)
        raw = raw[:end] + b'Z' * (start - end) + raw[start:]
        return _no_text_offsets(_set_field(raw, 1, start - 2))
    p, _ = _patched(tmp_path, fix, data=data)
    s = FlowSample(str(p))
    assert s.channel_names[2].startswith('FL3-AZZ')
    assert any('kept as written' in w for w in s.load_warnings)


def test_a_read_that_works_keeps_its_notes_when_the_mend_cannot_be_read(
        tmp_path):
    # The plain read worked (past an off-by-one data end); reading the
    # mended view fails: the first result stands, with both notes.
    def fix(raw):
        raw = _p3n_last(raw)
        end = _field(raw, 3)
        old = f'$ENDDATA/{end}/'.encode()
        raw = raw.replace(old, f'$ENDDATA/{end + 1}/'.encode())
        raw = _set_field(raw, 3, end + 1) + b'\0'
        return _set_field(raw, 1, _field(raw, 1) - 1)

    def path_only(src, **kw):
        if not isinstance(src, str):
            raise RuntimeError('this reader takes paths only')
        return fcs_read.FlowData(src, **kw)
    p, _ = _patched(tmp_path, fix)
    notes = []
    fcs = read_fcs_tolerant(str(p), path_only, notes)
    assert fcs.event_count == 200
    assert len(notes) == 2
    assert 'one byte off' in notes[0]
    assert notes[1] == ("s1.fcs: its TEXT has no closing delimiter, so its "
                        "last keyword may be missing its last character")


def test_a_filled_range_is_noted_once_after_a_quirk_retry(tmp_path):
    # TEXT is parsed again when the read is retried past an off-by-one
    # data end; the "$P2R missing" note is still said once.
    def fix(raw):
        raw = raw.replace(b'$P2R/1024/', b'XP2R/1024/')
        end = _field(raw, 3)
        raw = raw.replace(f'$ENDDATA/{end}/'.encode(),
                          f'$ENDDATA/{end + 1}/'.encode())
        return _set_field(raw, 3, end + 1) + b'\0'
    p, data = _patched(tmp_path, fix)
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), data)
    assert [w for w in s.load_warnings if '$P2R' in w] == [
        "$P2R (range of FL2-A) is missing: taken as 2^16 = 65,536 from its "
        "$P2B"]


def test_a_text_only_read_of_a_cut_short_file_is_still_refused(tmp_path):
    # FlowIO checks DATA against the file's size even for a TEXT-only
    # read; the repaired view reports the real size, so a file cut short
    # is refused as before, not read because the repaired TEXT made the
    # view longer.
    import functools
    p, _ = _make(tmp_path, 'no $TOT')
    p.write_bytes(p.read_bytes()[:-61])
    with pytest.raises(KeyError, match='tot'):          # the first error
        read_fcs_tolerant(str(p), functools.partial(
            fcs_read.FlowData, only_text=True), [])
