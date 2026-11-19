"""BD Accuri C6 files (.c6, .ci, .cfl): a zip archive of FCS files, one
per well, opened one sample per well (fcs_containers).

Every archive here is generated: synthetic FCS files zipped in the layout
public sources describe (members ``Sample A01.fcs``, a well index
``collectManager.txt``). No file from a real instrument was available, so
these tests pin the reader to that described layout only."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
import zipfile

import numpy as np
import pytest

from openflo import fcs_containers as fc
from openflo.fcs_containers import ContainerError, list_members
from openflo.fcs_data_sets import (
    MissingDataSetError,
    data_set_location,
    data_set_names,
    sample_data_sets,
)
from openflo.pipeline import FcsParseError, FlowSample
from tests._fcs_writer import write_fcs, write_float_fcs
from tests.test_multi_data_set_fcs import _queue, _run, desk  # noqa: F401

NAMES = ['FL1-A', 'FL2-A', 'FL3-A']


@pytest.fixture(autouse=True)
def _home(monkeypatch, tmp_path):
    """Copies go to ~/.openflo/cache: point ~ at a temp folder, and forget
    the listings an earlier test read."""
    home = tmp_path / 'cache-home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    fc._LISTINGS.clear()
    yield
    fc._LISTINGS.clear()


def _events(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(1.0, 1000.0, (n, 3)).astype(np.float32)


def _index(wells):
    """A well index as the C6 software is described writing it."""
    return '\r\n'.join(f'$SAMPLEIDENTIFIER/{w}/$SAMPLENAME/{nm}/'
                       for w, nm in wells.items())


def _c6(tmp_path, members, name='plate.c6', index=None, extra=(),
        index_name='collectManager.txt'):
    """Zip `members` ({archive name: events}) into `name`, members in the
    order given; `index` ({'A1': sample name}) adds the well index (as
    `index_name`) and `extra` more (name, bytes) members. Returns the
    path."""
    path = tmp_path / name
    parts = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, (arc, ev) in enumerate(members.items()):
                src = write_float_fcs(os.path.join(parts, f'{i}.fcs'), ev,
                                      NAMES)
                zf.write(src, arc)
            if index is not None:
                zf.writestr(index_name, _index(index))
            for arc, raw in extra:
                zf.writestr(arc, raw)
    finally:
        shutil.rmtree(parts, ignore_errors=True)
    return str(path)


def _values(s):
    return s.raw[NAMES].to_numpy()


A01, B01 = _events(50, 1), _events(70, 2)
TWO = {'Sample B01.fcs': B01, 'Sample A01.fcs': A01}   # not in well order
WELLS = {'A1': 'Tube 1', 'B1': 'Tube 2'}


# ── listing and reading ─────────────────────────────────────────────────────

def test_two_wells_are_listed_in_well_order_and_named(tmp_path):
    p = _c6(tmp_path, TWO, index=WELLS)
    members, note = list_members(p)
    assert note == ''
    assert [(m.number, m.member, m.label) for m in members] == [
        (1, 'Sample A01.fcs', 'A01 Tube 1'), (2, 'Sample B01.fcs', 'B01 Tube 2')]
    sets, _ = sample_data_sets(p)
    assert data_set_names(sets, p) == ['A01 Tube 1', 'B01 Tube 2']


@pytest.mark.parametrize('ext', ['.c6', '.ci', '.cfl', '.C6'])
def test_each_well_reads_its_own_events(tmp_path, ext):
    p = _c6(tmp_path, TWO, name=f'plate{ext}', index=WELLS)
    np.testing.assert_array_equal(_values(FlowSample(p, set_number=1)), A01)
    two = FlowSample(p, set_number=2)
    np.testing.assert_array_equal(_values(two), B01)
    assert two.set_number == 2 and not two.load_warnings


def test_without_an_index_a_well_is_named_alone(tmp_path):
    p = _c6(tmp_path, {'Sample C01.fcs': A01, 'run/extra tube.fcs': B01})
    assert [m.label for m in list_members(p)[0]] == ['C01', 'extra tube']


def test_a_one_well_file_is_still_named_by_its_well(tmp_path):
    p = _c6(tmp_path, {'Sample A01.fcs': A01}, index={'A1': 'Tube 1'})
    sets, _ = sample_data_sets(p)
    assert [s.number for s in sets] == [1]
    assert data_set_names(sets, p) == ['A01 Tube 1']


def test_wells_sort_by_row_then_column_not_by_name(tmp_path):
    p = _c6(tmp_path, {'B1.fcs': A01, 'A10.fcs': B01, 'A2.fcs': _events(9, 7),
                       'notes.fcs': _events(9, 8)})
    assert [m.label for m in list_members(p)[0]] == ['A02', 'A10', 'B01',
                                                     'notes']


def test_a_listing_is_read_once_per_version_of_the_file(tmp_path,
                                                        monkeypatch):
    p = _c6(tmp_path, TWO)
    calls = []
    real = fc.list_members_uncached
    monkeypatch.setattr(fc, 'list_members_uncached',
                        lambda path: calls.append(path) or real(path))
    list_members(p)
    list_members(p)
    assert len(calls) == 1
    _c6(tmp_path, {'Sample A01.fcs': B01})
    assert len(list_members(p)[0]) == 1
    assert len(calls) == 2


def test_only_fcs_members_are_samples(tmp_path):
    p = _c6(tmp_path, TWO, index=WELLS,
            extra=[('plots.xml', b'<plots/>'), ('Sample H12.fcs', b'nope' * 30),
                   ('Sample H11.fcs', b'FCS3')])          # shorter than a HEADER
    assert [m.member for m in list_members(p)[0]] == ['Sample A01.fcs',
                                                     'Sample B01.fcs']


def test_without_a_well_number_the_first_is_read_with_a_note(tmp_path):
    p = _c6(tmp_path, TWO, index=WELLS)
    s = FlowSample(p)
    np.testing.assert_array_equal(_values(s), A01)
    assert any('holds 2 samples, one per well' in w and 'A01 Tube 1' in w
               for w in s.load_warnings), s.load_warnings


def test_a_one_well_file_reads_without_a_note(tmp_path):
    p = _c6(tmp_path, {'Sample A01.fcs': A01})
    assert not FlowSample(p).load_warnings


def test_a_well_the_file_no_longer_holds_says_so(tmp_path):
    p = _c6(tmp_path, TWO)
    with pytest.raises(FcsParseError, match='sample 3 was not found: the '
                       'file now holds 2 samples'):
        FlowSample(p, set_number=3)
    with pytest.raises(MissingDataSetError):
        data_set_location(p, 3)


def test_keywords_and_event_total_come_from_the_well(tmp_path):
    from openflo.fcs_text import fcs_text_pairs
    from openflo.wsp_reader import fcs_event_total
    p = _c6(tmp_path, TWO)
    f, off = data_set_location(p, 2)
    assert off == 0
    assert dict(fcs_text_pairs(f, off))['$TOT'].strip() == '70'
    assert fcs_event_total(p, 2) == 70
    assert fcs_event_total(p, 1) == 50


# ── laid out like the public C6 files (Zenodo 4972042, 15351644) ───────────

# What those files hold besides the wells: settings and layout members
# (made-up contents here), a well index whose unused rows name the well
# '?2147483647', and one member stored twice under the same name.
SETTINGS = [('CFlowMasterFileHeader.txt', b'$FILETYPE/CFlowWorkspace24/'),
            ('PlotSpecs.txt', b'$PlotID0/1,Density,0,0/'),
            ('CFlowInfo.txt', b'$FLOWRATE/14/'),
            ('PlateTypeFile.txt', b'<Plate Type>Generic</Plate Type>'),
            ('CustomNames.txt', b'<XML>\r\n</XML>'),
            ('analyzeManager.txt', b'$SAMPLEIDENTIFIER/A1,B1/$SAMPLENAME//'),
            ('cumulativeData.txt', b'$CUMULATIVEDATA/A1,50,10,1/'),
            ('cumulativeData.txt', b'$CUMULATIVEDATA/A1,50,10,1/')]


def _real_layout(tmp_path, wells, empty=()):
    """A plate the way the public C6 files lay one out: settings members
    first, the index with a placeholder row and unnamed wells, every member
    stored (not compressed), wells not in well order, and a member of no
    events ($TOT 0) for each well in `empty`."""
    path = tmp_path / 'plate.c6'
    index = ('$SAMPLEIDENTIFIER/?2147483647/$SAMPLENAME//$PLOTID/Density 1-0/'
             + ''.join(f'$SAMPLEIDENTIFIER/{w}/$SAMPLENAME/{nm}/$PLOTID//'
                       for w, nm in WELLS.items())
             + '$SAMPLEIDENTIFIER/H12/$SAMPLENAME//$PLOTID//')
    parts = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as zf, \
                warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)    # the name twice
            for arc, raw in SETTINGS:
                zf.writestr(arc, raw)
            zf.writestr('collectManager.txt', index)
            for i, (arc, ev) in enumerate(wells.items()):
                zf.write(write_float_fcs(os.path.join(parts, f'{i}.fcs'), ev,
                                         NAMES), arc)
            for i, arc in enumerate(empty):
                zf.write(write_fcs(os.path.join(parts, f'e{i}.fcs'),
                                   np.zeros((0, 3)), NAMES, bits=32,
                                   pnr=[16777216] * 3), arc)
    finally:
        shutil.rmtree(parts, ignore_errors=True)
    return str(path)


def test_a_plate_laid_out_like_a_real_c6_file_opens_each_well(tmp_path):
    p = _real_layout(tmp_path, TWO)
    members, note = list_members(p)
    assert note == ''
    assert [(m.member, m.label) for m in members] == [
        ('Sample A01.fcs', 'A01 Tube 1'), ('Sample B01.fcs', 'B01 Tube 2')]
    np.testing.assert_array_equal(_values(FlowSample(p, set_number=1)), A01)
    np.testing.assert_array_equal(_values(FlowSample(p, set_number=2)), B01)


def test_an_empty_well_is_left_out_with_a_note(tmp_path):
    p = _real_layout(tmp_path, TWO, empty=['Sample A02.fcs', 'Sample C01.fcs'])
    members, note = list_members(p)
    assert [m.label for m in members] == ['A01 Tube 1', 'B01 Tube 2']
    assert note == 'plate.c6: 2 empty well(s) with no events were left out'
    sets, said = sample_data_sets(p)
    assert len(sets) == 2 and said == note
    np.testing.assert_array_equal(_values(FlowSample(p, set_number=2)), B01)


def test_a_member_whose_event_total_is_unreadable_is_kept(tmp_path):
    p = _c6(tmp_path, TWO)
    with zipfile.ZipFile(p, 'a') as zf:
        zf.write(write_fcs(tmp_path / 'odd.fcs', np.zeros((0, 3)), NAMES,
                           extra={'$TOT': 'none'}), 'Sample C01.fcs')
    assert [m.label for m in list_members(p)[0]] == ['A01', 'B01', 'C01']


def test_a_file_of_only_empty_wells_says_its_wells_are_empty(tmp_path):
    p = _real_layout(tmp_path, {}, empty=['Sample A01.fcs'])
    assert list_members(p) == (
        [], 'plate.c6: 1 empty well(s) with no events were left out')
    with pytest.raises(FcsParseError) as err:
        FlowSample(p)
    assert 'plate.c6 holds no events: every well in it is empty.' in str(
        err.value)


# ── values above the range (public C6 files: area channels past $PnR) ──────

TOP = 16777216                      # $PnR of every channel of a C6 well
# FL1-A holds two values past the top of the range: 18,042,191, which the
# standard's 24-bit mask wraps to 1,264,975, and $PnR itself, which it wraps
# to 0. Everything else is in range, the top value included.
OVER = np.array([[18042191, 5, 7], [TOP - 1, 6, 8], [TOP, 9, 10],
                 [1234, 11, 12]], dtype=np.uint32)


def _int_well(path, data, cyt='BD Accuri C6', names=NAMES):
    """A 32-bit integer well with $PnR 16777216, as a C6 well is stored;
    `cyt` None leaves $CYT out."""
    extra = {} if cyt is None else {'$CYT': cyt}
    return write_fcs(path, data, names, bits=32, pnr=[TOP] * 3, extra=extra)


def _pinned(data):
    return np.minimum(data, TOP - 1)


def _over_plate(tmp_path, names=NAMES):
    """A plate laid out like the public C6 files whose A01 holds OVER."""
    p = _real_layout(tmp_path, {'Sample B01.fcs': B01})
    with zipfile.ZipFile(p, 'a', zipfile.ZIP_STORED) as zf:
        zf.write(_int_well(tmp_path / 'a01.fcs', OVER, names=names),
                 'Sample A01.fcs')
    return p


def test_an_accuri_well_pins_values_above_the_range_with_a_note(tmp_path):
    p = _over_plate(tmp_path)
    s = FlowSample(p, set_number=1)
    np.testing.assert_array_equal(_values(s), _pinned(OVER))
    said = [w for w in s.load_warnings if 'above the channel' in w]
    assert said == ["A01 Tube 1.fcs: 2 value(s) above the channel's range "
                    "(FL1-A 2) were set to the top of the scale instead of "
                    "wrapping round to small values"], s.load_warnings
    two = FlowSample(p, set_number=2)          # a well with none: no note
    assert not any('above the channel' in w for w in two.load_warnings)


def test_an_accuri_fcs_export_is_pinned_too(tmp_path):
    f = _int_well(tmp_path / 'A01 Tube 1.fcs', OVER, cyt='Accuri C6')
    s = FlowSample(f)
    np.testing.assert_array_equal(_values(s), _pinned(OVER))
    assert any('2 value(s) above' in w for w in s.load_warnings)


@pytest.mark.parametrize('cyt', [None, 'FACSCanto II'])
def test_other_instruments_keep_the_standard_mask(tmp_path, cyt):
    f = _int_well(tmp_path / 'other.fcs', OVER, cyt=cyt)
    s = FlowSample(f)
    np.testing.assert_array_equal(_values(s), OVER & (TOP - 1))
    assert not any('above the channel' in w for w in s.load_warnings)


def test_every_reader_of_a_well_gets_the_pinned_values(tmp_path):
    import functools

    from openflo import fcs_read
    from openflo.fcs_quirks import read_fcs_tolerant
    from openflo.fcs_range_fill import RangeFillingFlowData
    from openflo.wsp_writer import WspWriter
    names = ['FL1-A', 'FL2-A', 'Time']
    p = _over_plate(tmp_path, names=names)
    want = _pinned(OVER).astype(float)
    for reader in (fcs_read.FlowData,
                   functools.partial(RangeFillingFlowData, filled=[])):
        notes: list = []
        fd = read_fcs_tolerant(p, reader, notes, 1)
        got = np.asarray(fd.events, dtype=float).reshape(-1, 3)
        np.testing.assert_array_equal(got, want)
        assert any('2 value(s) above' in n for n in notes), notes
    # The FlowJo export reads the Time channel itself.
    swapped = OVER[:, [1, 2, 0]]
    with zipfile.ZipFile(p, 'a') as zf:
        zf.write(_int_well(tmp_path / 'c01.fcs', swapped, names=names),
                 'Sample C01.fcs')
    fc._LISTINGS.clear()
    ticks = WspWriter._time_ticks({'fcs_path': p, 'set_number': 3})
    np.testing.assert_array_equal(ticks, _pinned(swapped)[:, 2])


# ── what is refused ─────────────────────────────────────────────────────────

def test_an_empty_archive_says_it_holds_no_fcs_data(tmp_path):
    p = _c6(tmp_path, {}, index=WELLS)
    assert list_members(p) == ([], '')
    assert sample_data_sets(p) == ([], '')
    with pytest.raises(FcsParseError) as err:
        FlowSample(p)
    msg = str(err.value)
    assert 'plate.c6 holds no FCS data' in msg
    assert 'Export ALL Samples as FCS' in msg
    assert 'ContainerError' not in msg          # said once, in plain words


def test_a_file_that_is_not_an_archive_is_refused_plainly(tmp_path):
    p = tmp_path / 'plate.c6'
    p.write_bytes(b'FCS3.0 not really a zip')
    with pytest.raises(ContainerError, match='is not a zip archive'):
        list_members(str(p))
    assert sample_data_sets(str(p)) == ([], '')
    with pytest.raises(FcsParseError, match='plate.c6 is not a BD Accuri '
                       'file OpenFlo can read'):
        FlowSample(str(p))


def test_a_damaged_archive_is_refused(tmp_path):
    p = _c6(tmp_path, TWO)
    raw = open(p, 'rb').read()
    cut = tmp_path / 'cut.c6'
    cut.write_bytes(raw[:len(raw) // 2])
    with pytest.raises(FcsParseError):
        FlowSample(str(cut))


# ── caps (zip-bomb safety) ──────────────────────────────────────────────────

def test_too_many_members_are_refused(tmp_path, monkeypatch):
    p = _c6(tmp_path, TWO, index=WELLS)
    monkeypatch.setattr(fc, 'MAX_MEMBERS', 3)
    assert len(list_members(p)[0]) == 2
    monkeypatch.setattr(fc, 'MAX_MEMBERS', 2)
    fc._LISTINGS.clear()
    with pytest.raises(ContainerError, match='holds 3 files, more than the 2'):
        list_members(p)


def test_a_member_over_the_size_cap_is_left_out_with_a_note(tmp_path,
                                                            monkeypatch):
    p = _c6(tmp_path, {'Sample A01.fcs': _events(5, 3),
                       'Sample B01.fcs': _events(500, 4)})
    small = os.path.getsize(write_float_fcs(tmp_path / 's.fcs',
                                            _events(5, 3), NAMES))
    monkeypatch.setattr(fc, 'MAX_MEMBER_BYTES', small)
    members, note = list_members(p)
    assert [m.member for m in members] == ['Sample A01.fcs']
    assert '1 file(s) larger than' in note


def test_a_total_over_the_cap_is_refused(tmp_path, monkeypatch):
    p = _c6(tmp_path, TWO)
    monkeypatch.setattr(fc, 'MAX_TOTAL_BYTES', 100)
    with pytest.raises(ContainerError, match='which OpenFlo does not unpack'):
        list_members(p)


def test_an_index_over_its_cap_is_not_read(tmp_path, monkeypatch):
    p = _c6(tmp_path, TWO, index=WELLS)
    # A cap the first entry fits in: an index over the cap is not read at
    # all, not read in part.
    first = len(_index({'A1': WELLS['A1']})) + 2
    monkeypatch.setattr(fc, 'MAX_INDEX_BYTES', first)
    assert [m.label for m in list_members(p)[0]] == ['A01', 'B01']


def test_a_member_longer_than_declared_is_not_copied(tmp_path):
    p = _c6(tmp_path, TWO)
    real = list_members(p)[0][0]
    lie = fc.Member(number=1, offset=0, version='', member=real.member,
                    label=real.label, size=real.size - 10)
    out = tmp_path / 'out.fcs'
    with pytest.raises(ContainerError, match='larger than the archive says'):
        fc._copy_member(p, lie, str(out))
    assert not out.exists()
    assert not [f for f in os.listdir(tmp_path) if f.endswith('.part')]


def test_a_member_shorter_than_declared_is_not_copied(tmp_path):
    p = _c6(tmp_path, TWO)
    real = list_members(p)[0][0]
    lie = fc.Member(number=1, offset=0, version='', member=real.member,
                    label=real.label, size=real.size + 10)
    out = tmp_path / 'out.fcs'
    with pytest.raises(ContainerError, match='is cut short'):
        fc._copy_member(p, lie, str(out))
    assert not out.exists()


# ── where copies go ─────────────────────────────────────────────────────────

def test_nothing_is_written_next_to_the_users_file(tmp_path):
    folder = tmp_path / 'data'
    folder.mkdir()
    q = _c6(folder, TWO)
    FlowSample(q, set_number=2)
    FlowSample(q)
    assert os.listdir(folder) == ['plate.c6']
    copy, member, count = fc.member_file(q, 2)
    assert copy.startswith(fc.cache_root())
    assert (member.member, count) == ('Sample B01.fcs', 2)


def test_a_copy_is_reused_and_redone_when_the_file_changes(tmp_path):
    p = _c6(tmp_path, TWO)
    first, _m, _n = fc.member_file(p, 1)
    os.utime(first, (1, 1))
    again, _m, _n = fc.member_file(p, 1)
    assert again == first and os.path.getmtime(first) == 1      # reused
    _c6(tmp_path, {'Sample A01.fcs': B01})                       # rewritten
    fresh, _m, _n = fc.member_file(p, 1)
    assert fresh != first
    np.testing.assert_array_equal(_values(FlowSample(p, set_number=1)), B01)


def test_old_copies_are_pruned_when_a_file_is_copied(tmp_path):
    old = os.path.join(fc.cache_root(), 'old')
    os.makedirs(old)
    os.utime(old, (1, 1))
    p = _c6(tmp_path, TWO)
    copy, _m, _n = fc.member_file(p, 1)
    assert not os.path.exists(old)
    # The folder being used is kept, however old.
    mine = os.path.dirname(os.path.dirname(copy))
    os.utime(mine, (1, 1))
    fc.prune_cache(keep=mine)
    assert os.path.isdir(mine)


def test_reading_a_copy_again_keeps_its_folder_from_pruning(tmp_path):
    p = _c6(tmp_path, TWO)
    copy, _m, _n = fc.member_file(p, 1)
    mine = os.path.dirname(os.path.dirname(copy))
    os.utime(mine, (1, 1))
    assert fc.member_file(p, 1)[0] == copy
    fc.prune_cache()
    assert os.path.isfile(copy)


# ── file-type hooks ─────────────────────────────────────────────────────────

def test_the_file_type_helpers(tmp_path):
    from openflo.paths import (
        expand_dropped_paths,
        fcs_stem,
        is_fcs_container,
        is_fcs_file,
        is_single_fcs_file,
        sample_key,
    )
    for name in ('a.c6', 'a.CI', 'a.cfl'):
        assert is_fcs_container(name) and is_fcs_file(name)
        assert not is_single_fcs_file(name)
    assert is_single_fcs_file('a.lmd') and not is_fcs_container('a.fcs')
    assert fcs_stem('plate.c6') == 'plate'
    p = _c6(tmp_path, TWO)
    assert sample_key(p, 2).endswith('plate.c6#2')
    assert expand_dropped_paths([str(tmp_path)])[0] == [p]


def test_the_command_line_scans_leave_accuri_files_out(tmp_path):
    from openflo.cli_groups import auto_groups_by_day, discover_day_folders
    from openflo.cli_samples import find_unstained, scan_for_token
    from openflo.compare import FcsLocator
    from openflo.flow_experiment import FlowExperiment
    _c6(tmp_path, TWO, name='x_tube_1.c6')
    _c6(tmp_path, TWO, name='x_unstained.c6')
    _c6(tmp_path, TWO, name='unstained x.ci')
    assert scan_for_token(str(tmp_path), 'tube', False) is None
    assert discover_day_folders([str(tmp_path)]) == []
    assert find_unstained(str(tmp_path)) is None
    assert FlowExperiment._resolve(str(tmp_path)) == []
    # A workspace can name an Accuri file, so its locator finds one.
    assert sorted(FcsLocator(fcs_dir=str(tmp_path))._build_index()) == [
        'unstained x.ci', 'x_tube_1.c6', 'x_unstained.c6']
    day = tmp_path / 'day 1'
    day.mkdir()
    write_float_fcs(day / 'a.fcs', A01, NAMES)
    _c6(day, TWO, name='b.c6')
    (group,) = auto_groups_by_day([str(tmp_path)])
    assert group['samples'] == ['a'], group


def test_a_pasted_line_of_paths_splits_after_an_accuri_file():
    import re

    from openflo.editor_clipboard import _PATH_END
    assert re.split(_PATH_END, 'a b.c6 c.fcs d.CFL e f.ci') == [
        'a b.c6', 'c.fcs', 'd.CFL', 'e f.ci']


def test_the_flowjo_export_reads_the_well_and_says_to_export_it(tmp_path):
    from openflo.wsp_writer import WspWriter
    p = _c6(tmp_path, TWO, index=WELLS)
    w = WspWriter(cytometer='T')
    w.add_sample('B01 Tube 2', p, NAMES, [], set_number=2)
    w.write(str(tmp_path / 'out.wsp'))
    (one,) = w.samples
    assert one['events'] == 70
    assert dict(one['keywords'])['$TOT'].strip() == '70'
    said = [x for x in w.warnings if 'well 2 of the BD Accuri file' in x]
    assert len(said) == 1 and 'Export FCS File' in said[0], w.warnings


# ── sessions (headless) ─────────────────────────────────────────────────────

def _session(tmp_path, entry):
    sess = tmp_path / 's.flowsession'
    sess.write_text(json.dumps({'format': 'openflo-session', 'version': 3,
                                'samples': [entry], 'sample_gates': {}}),
                    'utf-8')
    return str(sess)


def test_a_session_saves_the_wells_member_and_reopens_it(tmp_path):
    from openflo.document import Document
    p = _c6(tmp_path, TWO, index=WELLS)
    doc = Document()
    doc.open_session(_session(tmp_path, {'name': 'B01 Tube 2', 'path': p,
                                         'set_number': 2}))
    s = doc._samples['B01 Tube 2'].data
    np.testing.assert_array_equal(_values(s), B01)
    out = tmp_path / 'b.flowsession'
    doc.save_session(str(out))
    entry = json.loads(out.read_text('utf-8'))['samples'][0]
    assert (entry['set_number'], entry['container_member']) == (
        2, 'Sample B01.fcs')
    kw = {r[0]: r[1] for r in doc.keywords('B01 Tube 2')['keywords']}
    assert kw.get('$TOT', '').strip() == '70'


def test_a_session_reopens_its_well_after_the_file_gained_one(tmp_path):
    from openflo.session_open import plan_sample, planned_set_number
    p = _c6(tmp_path, TWO, index=WELLS)
    entry = {'name': 'B01 Tube 2', 'path': p, 'set_number': 2,
             'container_member': 'Sample B01.fcs'}
    assert planned_set_number(entry, p) == 2
    # The instrument ran A02 since: B01 is now the third well.
    _c6(tmp_path, {**TWO, 'Sample A02.fcs': _events(30, 5)}, index=WELLS)
    assert planned_set_number(entry, p) == 3
    plan = plan_sample(entry, {}, str(tmp_path))
    assert plan.set_number == 3 and plan.key.endswith('#3')
    assert plan.prep['set_number'] == 3
    np.testing.assert_array_equal(
        _values(FlowSample(p, set_number=plan.set_number)), B01)
    # Without the saved name, the saved number is kept.
    assert planned_set_number({'set_number': 2}, p) == 2


def test_a_session_well_gone_from_the_file_loads_as_missing(tmp_path):
    from openflo.session_open import planned_set_number
    p = _c6(tmp_path, TWO)
    entry = {'set_number': 1, 'container_member': 'Sample G07.fcs'}
    n = planned_set_number(entry, p)
    assert n == 3
    with pytest.raises(FcsParseError, match='was not found'):
        FlowSample(p, set_number=n)


# ── the desktop editor ──────────────────────────────────────────────────────

def test_opening_a_file_adds_a_sample_per_well(desk, tmp_path):  # noqa: F811
    root, ed = desk
    p = _c6(tmp_path, TWO, index=WELLS)
    jobs = _queue(ed, [p])
    assert jobs == [('A01 Tube 1', p, 'fcs', 1), ('B01 Tube 2', p, 'fcs', 2)]
    assert 'plate.c6 holds 2 samples, one per well' in ed.status_var.get()
    _run(root, ed, jobs)
    np.testing.assert_array_equal(_values(ed._samples['A01 Tube 1']), A01)
    np.testing.assert_array_equal(_values(ed._samples['B01 Tube 2']), B01)
    assert _queue(ed, [p]) == []                       # not opened twice


def test_names_in_use_and_repeated_names_are_made_unique(desk,  # noqa: F811
                                                         tmp_path):
    _root, ed = desk
    p = _c6(tmp_path, {'a/x.fcs': A01, 'b/x.fcs': B01})
    q = _c6(tmp_path, TWO, name='second.c6', index=WELLS)
    ed._name_to_path['A01 Tube 1'] = 'elsewhere'
    names = [j[0] for j in _queue(ed, [p, q])]
    assert len(names) == 4 and len(set(names)) == 4, names
    assert names[0] == 'x' and names[1].startswith('x ')
    assert names[2].startswith('A01 Tube 1 ') and names[3] == 'B01 Tube 2'


def test_an_empty_or_foreign_file_is_one_failing_job(desk, tmp_path):  # noqa: F811
    _root, ed = desk
    bad = tmp_path / 'bad.c6'
    bad.write_bytes(b'not a zip')
    empty = _c6(tmp_path, {}, name='empty.c6')
    jobs = _queue(ed, [str(bad), empty])
    assert jobs == [('bad', str(bad)), ('empty', empty)]
    assert 'one per well' not in ed.status_var.get()


def test_a_saved_session_reopens_each_well(desk, tmp_path):  # noqa: F811
    from tests.test_transform_records import _new_editor
    root, ed = desk
    p = _c6(tmp_path, TWO, index=WELLS)
    _run(root, ed, _queue(ed, [p]))
    path = tmp_path / 'desk.flowsession'
    data = ed._write_session(str(path))
    assert {e['name']: (e.get('set_number'), e.get('container_member'))
            for e in data['samples']} == {
        'A01 Tube 1': (1, 'Sample A01.fcs'),
        'B01 Tube 2': (2, 'Sample B01.fcs')}
    # A well was run since; the session still finds each by its member.
    _c6(tmp_path, {**TWO, 'Sample A02.fcs': _events(30, 5)}, index=WELLS)
    ed2 = _new_editor(root)
    jobs = []
    ed2._enqueue_load = lambda job, priority=1: jobs.append(job)
    ed2._load_session_path(str(path))
    assert sorted((j[0], j[3]) for j in jobs) == [('A01 Tube 1', 1),
                                                  ('B01 Tube 2', 3)]
    _run(root, ed2, jobs)
    np.testing.assert_array_equal(_values(ed2._samples['B01 Tube 2']), B01)


def test_the_add_dialog_lists_accuri_files(desk, monkeypatch):  # noqa: F811
    _root, ed = desk
    seen = {}

    def ask(**kw):
        seen.update(kw)
        return ()
    monkeypatch.setattr('openflo.editor_load.filedialog.askopenfilenames',
                        ask)
    ed._add_samples()
    types = dict(seen['filetypes'])
    assert '*.c6 *.ci *.cfl' in types['FCS & workspaces']
    assert types['BD Accuri files'] == '*.c6 *.ci *.cfl'


def test_the_watch_folder_leaves_accuri_files_alone(desk, tmp_path,  # noqa: F811
                                                    monkeypatch):
    _root, ed = desk
    _c6(tmp_path, TWO, name='old.c6')
    monkeypatch.setattr('openflo.editor_load.filedialog.askdirectory',
                        lambda **kw: str(tmp_path))
    ed._watch_after = None
    monkeypatch.setattr(ed, '_poll_watch', lambda: None)
    ed._toggle_watch_folder()
    assert ed._watch_seen == set()


def test_an_error_report_hides_accuri_file_names():
    from openflo.editor_lifecycle import _tokenise_for_report
    for name in ('Run7.c6', 'Run7.ci', 'Run7.cfl'):
        out = _tokenise_for_report(f"could not read {name}: damaged")
        assert 'Run7' not in out, out


def test_the_watch_poll_leaves_accuri_files_alone(desk, tmp_path,  # noqa: F811
                                                  monkeypatch):
    _root, ed = desk
    queued = []
    monkeypatch.setattr(ed, '_queue_fcs_loads', queued.extend)
    monkeypatch.setattr(ed, 'after', lambda ms, fn: None)
    ed._watch_dir = str(tmp_path)
    ed._watch_seen = set()
    _c6(tmp_path, TWO, name='new.c6')
    write_float_fcs(tmp_path / 'new.fcs', A01, NAMES)
    ed._poll_watch()
    assert [os.path.basename(q) for q in queued] == ['new.fcs']


def test_one_or_several_samples_are_counted_in_words():
    from openflo.editor_load import _samples_count
    assert (_samples_count(1), _samples_count(3)) == ('1 sample', '3 samples')


# ── workspaces: drops onto the workspace window, and FlowJo round trips ────

def test_a_member_lists_its_own_keywords(tmp_path):
    p = _c6(tmp_path, TWO)
    assert [m.keywords.get('$TOT', '').strip()
            for m in list_members(p)[0]] == ['50', '70']


def test_a_member_text_past_the_cap_is_not_read(tmp_path, monkeypatch):
    p = _c6(tmp_path, TWO)
    monkeypatch.setattr(fc, 'MAX_TEXT_BYTES', 64)
    assert [m.keywords for m in list_members(p)[0]] == [{}, {}]


def test_a_workspace_item_reloads_its_well_after_the_file_gained_one(
        tmp_path):
    from types import SimpleNamespace

    from openflo.workspace_run import (
        build_drop_payload,
        extract_one_context,
        member_channels,
        resolve_run_sample,
    )
    p = _c6(tmp_path, TWO, index=WELLS)
    ed = SimpleNamespace(_samples={'B': FlowSample(p, set_number=2)})
    item = build_drop_payload(extract_one_context(ed, 'B'), None)
    assert (item['set_number'], item['container_member']) == (
        2, 'Sample B01.fcs')
    _c6(tmp_path, {**TWO, 'Sample A02.fcs': _events(30, 5)}, index=WELLS)
    closed = SimpleNamespace(_samples={})
    s, _note = resolve_run_sample(closed, dict(item, qc_applied=False))
    np.testing.assert_array_equal(_values(s), B01)
    assert fc.item_set_number(item) == 3
    assert 'FL1-A' in member_channels(closed, dict(item, comp_channels=[]))
    # An item saved before the member name was kept reloads its number.
    assert fc.item_set_number(dict(item, container_member='')) == 2


def _wells_wsp(tmp_path, p, numbers=(1, 2)):
    from openflo.wsp_writer import WspWriter
    w = WspWriter(cytometer='T')
    for n in numbers:
        w.add_sample(f'well {n}', p, NAMES, [{
            'id': 'g1', 'parent_id': None, 'kind': 'threshold',
            'channel': 'FL1-A', 'value': 100.0 * n, 'name': f'cut {n}'}],
            set_number=n)
    out = tmp_path / 'out.wsp'
    w.write(str(out))
    return str(out)


def test_a_flowjo_round_trip_puts_each_sample_back_on_its_well(tmp_path):
    from openflo.document import Document
    from openflo.wsp_open import plan_wsp
    p = _c6(tmp_path, TWO, index=WELLS)
    wsp = _wells_wsp(tmp_path, p)
    plan = plan_wsp(wsp)
    assert [(r.node_name, r.set_number, r.set_name) for r in plan.samples] == [
        ('well 1', 1, 'A01 Tube 1'), ('well 2', 2, 'B01 Tube 2')]
    assert plan.notes == []
    doc = Document()
    res = doc.import_wsp(wsp)
    assert res['samples'] == ['A01 Tube 1', 'B01 Tube 2']
    np.testing.assert_array_equal(_values(doc._samples['B01 Tube 2'].data),
                                  B01)
    assert [g['label'] for g in doc._samples['B01 Tube 2'].gates.values()
            ] == ['cut 2']


def test_a_round_trip_that_cannot_tell_wells_apart_says_so(tmp_path):
    from openflo.wsp_open import plan_wsp
    same = {'Sample A01.fcs': _events(60, 1), 'Sample B01.fcs': _events(60, 2)}
    p = _c6(tmp_path, same)
    plan = plan_wsp(_wells_wsp(tmp_path, p, numbers=(2,)))
    assert [r.set_number for r in plan.samples] == [None]
    said = ' | '.join(plan.notes)
    assert ("plate.c6 holds 2 wells and several have this sample's event "
            "count" in said), said
    assert 'on its first well' in said


def test_a_bead_matrix_reads_the_channels_of_the_items_well(tmp_path):
    from types import SimpleNamespace

    from openflo.workspace_run import member_channels
    p = _c6(tmp_path, TWO)
    item = {'path': p, 'set_number': 2, 'container_member': 'Sample B01.fcs',
            'comp_channels': []}
    # The instrument ran A02 since, with a fourth detector: it is now
    # number 2, and B01 number 3.
    four = write_float_fcs(tmp_path / 'four.fcs', _events(20, 9)[:, [0, 1, 2, 0]],
                           NAMES + ['FL4-A'])
    with zipfile.ZipFile(p, 'a') as zf:
        zf.write(four, 'Sample A02.fcs')
    fc._LISTINGS.clear()
    closed = SimpleNamespace(_samples={})
    assert 'FL4-A' not in member_channels(closed, item)
    assert 'FL4-A' in member_channels(closed, dict(item, container_member=''))


# ── review round 1: time gates, archive guards, concurrent copies ──────────

def test_a_time_gate_comes_back_on_its_wells_timestep(tmp_path):
    from openflo.document import Document
    from openflo.fcs_text import fcs_timestep
    from openflo.wsp_writer import WspWriter
    names = ['FL1-A', 'FL2-A', 'Time']
    ev = _events(80, 3)
    ev[:, 2] = np.arange(80, dtype=np.float32)
    path = tmp_path / 'plate.c6'
    with zipfile.ZipFile(path, 'w') as zf:
        for arc, n in (('Sample A01.fcs', 80), ('Sample B01.fcs', 60)):
            src = write_float_fcs(tmp_path / 'part.fcs', ev[:n], names,
                                  metadata_dict={'TIMESTEP': '0.25'})
            zf.write(src, arc)
    p = str(path)
    assert fcs_timestep(p, set_number=2) == pytest.approx(0.25)
    w = WspWriter(cytometer='T')
    w.add_sample('s', p, names, [{
        'id': 'g1', 'parent_id': None, 'kind': 'threshold',
        'channel': 'Time', 'value': 40.0, 'name': 'late'}], set_number=2)
    assert w.samples[0]['timestep'] == pytest.approx(0.25)
    out = str(tmp_path / 't.wsp')
    w.write(out)
    doc = Document()
    (nm,) = doc.import_wsp(out)['samples']
    (gate,) = [g for g in doc._samples[nm].gates.values()
               if g.get('channel') == 'Time']
    assert gate['value'] == pytest.approx(40.0)


def _fcs_bytes(tmp_path, events):
    return open(write_float_fcs(tmp_path / 'raw.fcs', events, NAMES),
                'rb').read()


def test_a_member_whose_text_starts_inside_its_header_is_not_read(
        tmp_path, monkeypatch):
    raw = bytearray(_fcs_bytes(tmp_path, A01))
    raw[10:18] = b'10'.rjust(8)                 # TEXT "starts" at byte 10
    p = _c6(tmp_path, {'Sample B01.fcs': B01},
            extra=[('Sample A01.fcs', bytes(raw))])
    sizes = []
    real = zipfile.ZipExtFile.read

    def read(self, n=-1):
        sizes.append(n)
        return real(self, n)
    monkeypatch.setattr(zipfile.ZipExtFile, 'read', read)
    members = list_members(p)[0]
    assert [m.keywords for m in members][0] == {}
    assert all(n is not None and n >= 0 for n in sizes), sizes


def test_the_total_cap_is_on_all_members_together(tmp_path, monkeypatch):
    p = _c6(tmp_path, TWO)
    biggest = max(m.size for m in list_members(p)[0])
    fc._LISTINGS.clear()
    monkeypatch.setattr(fc, 'MAX_TOTAL_BYTES', biggest + 1)
    with pytest.raises(ContainerError, match='which OpenFlo does not unpack'):
        list_members(p)


def test_index_names_are_capped_and_lose_control_characters(tmp_path):
    long = 'x' * 200
    p = _c6(tmp_path, TWO, index={'A1': long, 'B1': 'Tube\x07\x072'})
    a, b = (m.label for m in list_members(p)[0])
    assert a == 'A01 ' + 'x' * 80
    assert b == 'B01 Tube 2'


def test_an_index_with_two_digit_columns_names_its_wells(tmp_path):
    p = _c6(tmp_path, {'Sample A10.fcs': A01, 'Sample H12.fcs': B01},
            index={'A10': 'ten', 'H12': 'last'})
    assert [m.label for m in list_members(p)[0]] == ['A10 ten', 'H12 last']


def test_an_index_inside_a_folder_is_read(tmp_path):
    p = _c6(tmp_path, TWO, index=WELLS, index_name='run 1/collectManager.txt')
    assert [m.label for m in list_members(p)[0]] == ['A01 Tube 1',
                                                     'B01 Tube 2']


def test_a_sidecar_sample_saves_its_wells_member(tmp_path):
    from types import SimpleNamespace

    from openflo.session_format import container_member
    p = _c6(tmp_path, TWO)
    side = tmp_path / 'B01.npz'
    side.write_bytes(b'')
    s = SimpleNamespace(path=str(side), source_fcs=p, set_number=2)
    assert container_member(s, 2) == 'Sample B01.fcs'
    assert container_member(SimpleNamespace(path=p), 1) == 'Sample A01.fcs'


def test_a_cached_copy_of_the_wrong_size_is_copied_again(tmp_path):
    p = _c6(tmp_path, TWO)
    copy, member, _n = fc.member_file(p, 2)
    with open(copy, 'r+b') as fh:
        fh.truncate(member.size // 2)
    again, _m, _n = fc.member_file(p, 2)
    assert again == copy and os.path.getsize(copy) == member.size
    np.testing.assert_array_equal(_values(FlowSample(p, set_number=2)), B01)


def test_the_remembered_listings_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, '_MAX_LISTINGS', 2)
    for k in range(5):
        list_members(_c6(tmp_path, TWO, name=f'p{k}.c6'))
        assert len(fc._LISTINGS) <= 2


def test_two_threads_copying_one_well_both_succeed(tmp_path):
    import threading
    p = _c6(tmp_path, {'Sample A01.fcs': _events(5000, 4)})
    member = list_members(p)[0][0]
    out = str(tmp_path / 'cache' / 'A01.fcs')
    os.makedirs(os.path.dirname(out))
    errors = []

    def copy():
        for _ in range(15):
            try:
                fc._copy_member(p, member, out)
            except Exception as exc:  # noqa: BLE001 - collected for the assert
                errors.append(exc)
    threads = [threading.Thread(target=copy) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert os.path.getsize(out) == member.size
    assert os.listdir(os.path.dirname(out)) == ['A01.fcs']


def test_a_copy_windows_will_not_replace_leaves_the_whole_one(tmp_path,
                                                             monkeypatch):
    """Windows refuses to replace a file that another thread has open or is
    replacing (PermissionError: Access is denied), which failed loader threads
    copying one well at once. Simulated here: a whole copy already in place
    stays; a partial one is not taken for whole, so the refusal is raised."""
    p = _c6(tmp_path, {'Sample A01.fcs': _events(500, 4)})
    member = list_members(p)[0][0]
    out = str(tmp_path / 'cache' / 'A01.fcs')
    os.makedirs(os.path.dirname(out))
    fc._copy_member(p, member, out)
    real = os.replace

    def windows_replace(src, dst):
        if os.path.exists(dst):
            raise PermissionError(13, 'Access is denied')
        return real(src, dst)
    monkeypatch.setattr(os, 'replace', windows_replace)
    fc._copy_member(p, member, out)
    assert os.path.getsize(out) == member.size
    assert os.listdir(os.path.dirname(out)) == ['A01.fcs']
    with open(out, 'r+b') as fh:
        fh.truncate(member.size // 2)
    with pytest.raises(PermissionError):
        fc._copy_member(p, member, out)
    assert os.listdir(os.path.dirname(out)) == ['A01.fcs']


# ── review round 1: the reviewer's remaining plants ─────────────────────────

def test_an_index_in_lower_case_names_its_wells(tmp_path):
    p = _c6(tmp_path, TWO)
    with zipfile.ZipFile(p, 'a') as zf:
        zf.writestr('collectManager.txt',
                    '$sampleidentifier/a1/$samplename/Tube 1/\r\n'
                    '$sampleidentifier/b1/$samplename/Tube 2/')
    members, _ = list_members(p)
    assert [m.label for m in members] == ['A01 Tube 1', 'B01 Tube 2']


def test_a_line_break_inside_an_index_entry_is_dropped(tmp_path):
    p = _c6(tmp_path, TWO)
    with zipfile.ZipFile(p, 'a') as zf:
        zf.writestr('collectManager.txt',
                    '$SAMPLEIDENTIFIER/A1/$SAMPLENAME/Tu\r\nbe 1/\r\n'
                    '$SAMPLEIDENTIFIER/B1/$SAMPLENAME/Tube 2/')
    members, _ = list_members(p)
    assert [m.label for m in members] == ['A01 Tube 1', 'B01 Tube 2']


def test_folders_in_the_archive_do_not_count_toward_the_member_cap(
        tmp_path, monkeypatch):
    p = _c6(tmp_path, TWO)
    with zipfile.ZipFile(p, 'a') as zf:
        zf.writestr(zipfile.ZipInfo('wells/'), b'')
    monkeypatch.setattr(fc, 'MAX_MEMBERS', 2)
    members, _ = list_members(p)
    assert [m.member for m in members] == ['Sample A01.fcs', 'Sample B01.fcs']


def test_a_well_must_stand_apart_at_the_end_of_a_member_name(tmp_path):
    # 'DATA01' ends in letters-then-digits but names no well: it sorts
    # after the wells, not as A01 before them.
    p = _c6(tmp_path, {'DATA01.fcs': A01, 'Sample B01.fcs': B01})
    members, _ = list_members(p)
    assert [m.member for m in members] == ['Sample B01.fcs', 'DATA01.fcs']


def test_a_saved_number_whose_member_still_matches_is_kept(tmp_path):
    # Two members under one name (a zip allows it): the saved number
    # still names its own member, so it stays, not the first of that name.
    other = _c6(tmp_path, {'Sample A01.fcs': B01}, name='other.c6')
    with zipfile.ZipFile(other) as zf:
        raw = zf.read('Sample A01.fcs')
    p = _c6(tmp_path, {'Sample A01.fcs': A01})
    with zipfile.ZipFile(p, 'a') as zf, pytest.warns(UserWarning):
        zf.writestr('Sample A01.fcs', raw)
    assert len(list_members(p)[0]) == 2
    assert fc.resolve_member(p, 2, 'Sample A01.fcs') == 2


def test_a_plain_fcs_item_is_never_opened_as_an_archive(tmp_path,
                                                        monkeypatch):
    plain = tmp_path / 'tube.fcs'
    write_float_fcs(str(plain), A01, NAMES)

    def boom(*a, **k):
        raise AssertionError('a plain FCS file was opened as an archive')
    monkeypatch.setattr(fc, 'list_members', boom)
    item = {'path': str(plain), 'set_number': 2,
            'container_member': 'Sample A01.fcs'}
    assert fc.item_set_number(item) == 2


def test_a_missing_accuri_file_saves_no_member(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openflo.session_format import container_member

    def boom(*a, **k):
        raise AssertionError('a missing file was opened')
    monkeypatch.setattr(fc, 'member_name', boom)
    s = SimpleNamespace(path=str(tmp_path / 'gone.c6'), source_fcs=None)
    assert container_member(s, 1) == ''


def test_labels_name_the_sets_only_when_every_set_has_one(tmp_path):
    sets = [fc.Member(number=1, offset=0, version='FCS3.0',
                      keywords={'$FIL': 'one.fcs'}, label='A01 Tube 1'),
            fc.Member(number=2, offset=0, version='FCS3.0',
                      keywords={'$FIL': 'two.fcs'}, label='')]
    assert data_set_names(sets, str(tmp_path / 'plate.c6')) == ['one', 'two']


def test_the_comp_optimizer_folder_fill_skips_accuri_files(
        desk, tmp_path, monkeypatch):  # noqa: F811
    # One control per file: an Accuri file holds a plate of wells, so it
    # is not taken as the control for the channel its name matches.
    from openflo import ui_comp
    d = tmp_path / 'controls'
    d.mkdir()
    write_float_fcs(str(d / 'FL1.fcs'), A01, NAMES)
    _c6(d, TWO, name='FL2.c6')
    dlg = ui_comp.OptimizeCompensationDialog(desk[0], ['FL1-A', 'FL2-A'],
                                             on_complete=lambda c, m: None)
    monkeypatch.setattr(ui_comp.filedialog, 'askdirectory',
                        lambda **k: str(d))
    dlg._autofill_from_dir()
    got = {c: os.path.basename(v.get()) for c, v in dlg.path_vars.items()}
    assert got == {'FL1-A': 'FL1.fcs', 'FL2-A': ''}, got
