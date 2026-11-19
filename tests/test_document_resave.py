"""openflo.document.Document.save_session over an earlier save, and with
sources that changed after they were read: no stale transform record is
left beside a new sidecar copy, a move that fails partway never pairs a
sidecar with another one's record, a sidecar changed on disk is refused,
and an interrupted save leaves nothing behind.

Generated FCS files and sessions only (FSC-A, SSC-A, FL1-A ...; sample
s1). No Tk.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from openflo import document
from openflo.document import Document, DocumentError
from tests.test_document import gated_doc
from tests.test_document_save_safety import _digest, _tree, sidecar_doc

ASINH = {'method': 'asinh', 'cofactor': 150.0}


def _sessions(tmp_path):
    """``(make, s)``: ``make(stem, frame, record)`` writes session
    ``<stem>.flowsession`` whose sample s1 loads from sidecar
    ``<stem>_data/s1.npz`` holding `frame`, with a transform record when
    `record` (FL1-A recorded as asinh) and none otherwise, and returns its
    path; `s` is the generated sample."""
    from openflo.session_sidecar import write_frame
    from openflo.transform_sidecar import write_transforms_sidecar
    doc, name, *_ = gated_doc(tmp_path)
    s = doc._samples[name].data
    doc.save_session(str(tmp_path / 'base.flowsession'))

    def make(stem, frame, record):
        side = tmp_path / f'{stem}_data' / 's1.npz'
        side.parent.mkdir()
        assert write_frame(frame, str(side))
        if record:
            tr = dict(s.data_transforms, **{'FL1-A': ASINH})
            write_transforms_sidecar(str(side), tr, frame.columns,
                                     len(frame))
        d = json.loads((tmp_path / 'base.flowsession').read_text('utf-8'))
        d['samples'][0]['processed_data'] = f'{stem}_data/s1.npz'
        out = tmp_path / f'{stem}.flowsession'
        out.write_text(json.dumps(d), 'utf-8')
        return str(out)
    return make, s


def _reopened(path):
    doc = Document()
    notes = doc.open_session(path)['notes']
    return notes, doc.stats('s1', channels=['FL1-A'], stats=('Median',))


def _medians(rows):
    """FL1-A median of each population (its column may carry the tag of a
    transform assumed for it)."""
    return [[st['median'] for c, st in r['channels'].items()
             if c.startswith('FL1-A')] for r in rows]


# ── H1: no stale transform record ───────────────────────────────────────────

def test_a_resave_without_a_record_drops_the_earlier_saves_record(tmp_path):
    """The reviewer's p_stale case: session B (FL1-A recorded as asinh) is
    saved to out, then session A (same sample, no record) is saved over
    it. B's record must not stay beside A's copy: out reopens with the
    numbers of A, and of A saved into a fresh folder."""
    make, s = _sessions(tmp_path)
    b = make('B', s.data, True)
    a = make('A', s.data.copy(), False)
    out = tmp_path / 'out.flowsession'
    doc_b = Document()
    doc_b.open_session(b)
    doc_b.save_session(str(out))
    assert (tmp_path / 'out_data' / 's1_transforms.json').is_file()
    doc_a = Document()
    doc_a.open_session(a)
    doc_a.save_session(str(out))
    doc_a.save_session(str(tmp_path / 'fresh.flowsession'))
    assert not (tmp_path / 'out_data' / 's1_transforms.json').exists()
    notes_a, stats_a = _reopened(a)
    notes_o, stats_o = _reopened(str(out))
    _notes_f, stats_f = _reopened(str(tmp_path / 'fresh.flowsession'))
    assert _medians(stats_o) == _medians(stats_f)
    assert _medians(stats_o) != _medians(_reopened(b)[1])
    assert json.dumps(stats_o, sort_keys=True) == json.dumps(
        stats_a, sort_keys=True)
    # Reopened as A is: on a transform assumed for it, with A's note.
    assert notes_o == [n.replace('A_data', 'out_data') for n in notes_a]
    assert notes_o


@pytest.mark.parametrize('fails_on', ['s1.npz', 's1_transforms.json'])
def test_a_move_failing_partway_never_pairs_a_copy_with_an_old_record(
        tmp_path, monkeypatch, fails_on):
    """Saving C (its own sidecar and record) over an earlier save of B
    (other data, other record), with the move of the sidecar or of its
    record failing: the record left beside the sidecar, if any, is the
    one written with it, never B's with C's sidecar or C's with B's."""
    make, s = _sessions(tmp_path)
    b = make('B', s.data, True)
    frame = s.data.copy()
    frame['FL3-A'] = frame['FL3-A'] * 2.0
    c = make('C', frame, True)
    rec = tmp_path / 'C_data' / 's1_transforms.json'
    data = json.loads(rec.read_text('utf-8'))
    data['transforms']['FL1-A'] = dict(ASINH, cofactor=5.0)
    rec.write_text(json.dumps(data), 'utf-8')
    out = tmp_path / 'out.flowsession'
    doc_b = Document()
    doc_b.open_session(b)
    doc_b.save_session(str(out))
    side_d = tmp_path / 'out_data' / 's1.npz'
    rec_d = tmp_path / 'out_data' / 's1_transforms.json'
    pairs = {(_digest(tmp_path / f'{k}_data' / 's1.npz'),
              _digest(tmp_path / f'{k}_data' / 's1_transforms.json'))
             for k in 'BC'}
    assert len({p[0] for p in pairs}) == len({p[1] for p in pairs}) == 2
    doc_c = Document()
    doc_c.open_session(c)
    real = os.replace

    def replace(src, dst):
        if os.path.basename(dst) == fails_on:
            raise OSError('cannot move (planted)')
        return real(src, dst)
    monkeypatch.setattr(document.os, 'replace', replace)
    with pytest.raises(OSError, match='planted'):
        doc_c.save_session(str(out))
    monkeypatch.undo()
    # B's record was removed first; the sidecar left is B's when its move
    # failed, C's when the move of C's record did, and no record either way.
    kept = 'B' if fails_on == 's1.npz' else 'C'
    assert _digest(side_d) == _digest(tmp_path / f'{kept}_data' / 's1.npz')
    assert not rec_d.exists()
    assert sorted(os.listdir(tmp_path / 'out_data')) == ['s1.npz']


def test_a_record_to_drop_that_is_a_file_read_is_refused(tmp_path):
    """The record a save would remove beside its copy is checked like
    every file it writes: a hard link there to a file the document read
    is refused before anything changes (nothing read is ever removed)."""
    make, s = _sessions(tmp_path)
    a = make('A', s.data.copy(), False)
    doc = Document()
    doc.open_session(a)
    (tmp_path / 'out_data').mkdir()
    os.link(tmp_path / 'raw' / 's1.fcs',
            tmp_path / 'out_data' / 's1_transforms.json')
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match='s1_transforms.json is a file'):
        doc.save_session(str(tmp_path / 'out.flowsession'))
    assert _tree(tmp_path) == before


# ── L3: a sidecar changed on disk is not copied ─────────────────────────────

def _rewrite_sidecar(tmp_path):
    from openflo.session_sidecar import read_frame, write_frame
    side = tmp_path / 'run_data' / 's1.npz'
    frame = read_frame(str(side))
    frame['FL3-A'] = np.zeros(len(frame))
    assert write_frame(frame, str(side))


def _rewrite_record(tmp_path):
    rec = tmp_path / 'run_data' / 's1_transforms.json'
    data = json.loads(rec.read_text('utf-8'))
    data['transforms']['FL1-A'] = ASINH
    rec.write_text(json.dumps(data), 'utf-8')


def _remove_record(tmp_path):
    os.remove(tmp_path / 'run_data' / 's1_transforms.json')


def _touch_sidecar(tmp_path):
    side = tmp_path / 'run_data' / 's1.npz'
    st = os.stat(side)
    os.utime(side, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def _grow_sidecar_keeping_time(tmp_path):
    """Written in place (same inode) with its time put back: only the
    size tells."""
    side = tmp_path / 'run_data' / 's1.npz'
    st = os.stat(side)
    with open(side, 'ab') as fh:
        fh.write(b'\0')
    os.utime(side, ns=(st.st_atime_ns, st.st_mtime_ns))


@pytest.mark.parametrize('change', [_rewrite_sidecar, _rewrite_record,
                                    _remove_record, _touch_sidecar,
                                    _grow_sidecar_keeping_time])
def test_a_sidecar_changed_since_it_was_read_is_refused(tmp_path, change):
    """The sidecar or its record rewritten, removed or touched after the
    document read it (a desktop re-save, a sync): the save is refused,
    naming the file, before anything is written. Its gates were drawn on
    the data read, not on what is there now."""
    doc, _name = sidecar_doc(tmp_path)
    change(tmp_path)
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match='s1.* has changed since'):
        doc.save_session(str(tmp_path / 'new.flowsession'))
    assert _tree(tmp_path) == before


def test_a_record_that_appeared_since_it_was_read_is_refused(tmp_path):
    """A sidecar opened with no record, which has one by save time."""
    from openflo.transform_sidecar import write_transforms_sidecar
    make, s = _sessions(tmp_path)
    a = make('A', s.data.copy(), False)
    doc = Document()
    doc.open_session(a)
    write_transforms_sidecar(str(tmp_path / 'A_data' / 's1.npz'),
                             s.data_transforms, s.data.columns, len(s.data))
    before = _tree(tmp_path)
    with pytest.raises(DocumentError,
                       match='s1_transforms.json has changed since'):
        doc.save_session(str(tmp_path / 'out.flowsession'))
    assert _tree(tmp_path) == before


def test_a_sidecar_changed_while_it_is_copied_is_refused(tmp_path,
                                                         monkeypatch):
    """The sidecar changes while its copy is written: the save is refused
    and nothing is moved into place."""
    doc, _name = sidecar_doc(tmp_path)
    before = _tree(tmp_path)
    real = document._copy_file

    def copy_then_touch(f, src):
        real(f, src)
        if src.endswith('s1.npz'):
            st = os.stat(src)
            os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    monkeypatch.setattr(document, '_copy_file', copy_then_touch)
    with pytest.raises(DocumentError, match='s1.npz has changed since'):
        doc.save_session(str(tmp_path / 'new.flowsession'))
    assert _tree(tmp_path) == before


# ── an interrupted save ─────────────────────────────────────────────────────

def _interrupt(*_a, **_k):
    raise KeyboardInterrupt


@pytest.mark.parametrize('broken', ['_write_json', '_copy_file'])
def test_an_interrupted_save_leaves_nothing_behind(tmp_path, monkeypatch,
                                                   broken):
    """Ctrl+C while a file is written: the temporary files and the new data
    folder are removed, as for an error, and the interrupt goes on."""
    doc, _name = sidecar_doc(tmp_path)
    before = _tree(tmp_path)
    monkeypatch.setattr(document, broken, _interrupt)
    with pytest.raises(KeyboardInterrupt):
        doc.save_session(str(tmp_path / 'new.flowsession'))
    assert _tree(tmp_path) == before


def test_a_failed_save_keeps_an_empty_data_folder_already_there(
        tmp_path, monkeypatch):
    """Only a data folder the save made is removed when it fails: an empty
    one already there stays."""
    doc, _name = sidecar_doc(tmp_path)
    (tmp_path / 'new_data').mkdir()
    before = _tree(tmp_path)
    monkeypatch.setattr(document, '_write_json', _interrupt)
    with pytest.raises(KeyboardInterrupt):
        doc.save_session(str(tmp_path / 'new.flowsession'))
    assert _tree(tmp_path) == before


@pytest.mark.skipif(not os.path.isdir('/proc/self/fd'),
                    reason='needs /proc/self/fd to count open files')
def test_an_interrupt_as_a_temporary_file_opens_leaks_nothing(tmp_path,
                                                              monkeypatch):
    """Interrupted between making the temporary file and opening it as a
    file object: the file is removed and its descriptor closed."""
    doc, _name = sidecar_doc(tmp_path)
    before = _tree(tmp_path)
    fds = len(os.listdir('/proc/self/fd'))
    monkeypatch.setattr(document.os, 'fdopen', _interrupt)
    with pytest.raises(KeyboardInterrupt):
        doc.save_session(str(tmp_path / 'new.flowsession'))
    monkeypatch.undo()
    assert _tree(tmp_path) == before
    assert len(os.listdir('/proc/self/fd')) == fds


def test_a_temporary_name_is_short(tmp_path):
    """A temporary file's name is ``<name>.<6 hex>.tmp``: 7 characters
    longer than the editor's ``<name>.tmp`` (Windows path length)."""
    staged = []
    with document._new_temp(str(tmp_path / 'x.flowsession'), staged) as f:
        f.write(b'')
    tmp, dst = staged[0]
    assert dst == str(tmp_path / 'x.flowsession')
    assert len(tmp) - len(dst + '.tmp') == 7
    assert os.path.isfile(tmp)


# ── L4: another spelling of a file read ─────────────────────────────────────

def test_a_spelling_that_differs_in_case_is_refused_where_case_folds(
        tmp_path, monkeypatch):
    """Where os.path.normcase folds case (Windows), the session read
    spelled in other case is the same file: refused. Simulated on any
    system by standing in Windows's folding for the module's normcase,
    from before the session is opened."""
    monkeypatch.setattr(document, '_normcase', str.lower)
    doc, _name = sidecar_doc(tmp_path)
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match='opened from'):
        doc.save_session(str(tmp_path / 'RUN.flowsession'))
    assert _tree(tmp_path) == before



def test_an_earlier_saves_records_stay_when_a_copy_changed_is_refused(
        tmp_path, monkeypatch):
    """Saving A over an earlier save of B, A's sidecar changing while it
    is copied: the save is refused before B's record (beside the sidecar
    A's copy would replace) is removed, so the earlier save is untouched."""
    make, s = _sessions(tmp_path)
    b = make('B', s.data, True)
    a = make('A', s.data.copy(), False)
    out = tmp_path / 'out.flowsession'
    doc_b = Document()
    doc_b.open_session(b)
    doc_b.save_session(str(out))
    doc_a = Document()
    doc_a.open_session(a)
    before = _tree(tmp_path)
    real = document._copy_file

    def copy_then_touch(f, src):
        real(f, src)
        st = os.stat(src)
        os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    monkeypatch.setattr(document, '_copy_file', copy_then_touch)
    with pytest.raises(DocumentError, match='s1.npz has changed since'):
        doc_a.save_session(str(out))
    monkeypatch.undo()
    after = _tree(tmp_path)
    assert after.pop('A_data/s1.npz') == before.pop('A_data/s1.npz')
    assert after == before
    assert (tmp_path / 'out_data' / 's1_transforms.json').is_file()


def test_a_record_that_cannot_be_removed_fails_the_save(tmp_path,
                                                        monkeypatch):
    """The earlier save's record cannot be removed (held open by another
    program on Windows, say): the save fails with that error, and A's
    sidecar is never moved in beside B's record."""
    make, s = _sessions(tmp_path)
    b = make('B', s.data, True)
    a = make('A', s.data.copy(), False)
    out = tmp_path / 'out.flowsession'
    doc_b = Document()
    doc_b.open_session(b)
    doc_b.save_session(str(out))
    doc_a = Document()
    doc_a.open_session(a)
    before = _tree(tmp_path)
    real = os.remove

    def remove(p, *a, **k):
        if os.path.basename(p) == 's1_transforms.json':
            raise PermissionError('in use by another program (planted)')
        return real(p, *a, **k)
    monkeypatch.setattr(document.os, 'remove', remove)
    with pytest.raises(PermissionError, match='planted'):
        doc_a.save_session(str(out))
    monkeypatch.undo()
    assert _tree(tmp_path) == before


def test_a_sidecar_replaced_with_its_size_and_time_kept_is_refused(
        tmp_path):
    """The sidecar replaced by another file of the same size, its
    modification time put back (a sync tool): the new inode tells it."""
    doc, _name = sidecar_doc(tmp_path)
    side = tmp_path / 'run_data' / 's1.npz'
    st = os.stat(side)
    os.link(side, tmp_path / 'old.npz')       # the old inode stays in use
    data = bytearray(side.read_bytes())
    data[-1] ^= 0xFF
    new = tmp_path / 'run_data' / 'new.part'
    new.write_bytes(bytes(data))
    os.replace(new, side)
    os.utime(side, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert (os.stat(side).st_size, os.stat(side).st_mtime_ns) == (
        st.st_size, st.st_mtime_ns)
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match='s1.npz has changed since'):
        doc.save_session(str(tmp_path / 'new.flowsession'))
    assert _tree(tmp_path) == before


def test_a_sidecar_changed_as_it_is_read_is_refused(tmp_path, monkeypatch):
    """The stamps are taken before the sidecar is read: one that changes
    while the document reads it is refused at save."""
    from openflo import sample_prep
    real = sample_prep.read_sidecar_checked

    def touch_then_read(path):
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        return real(path)
    make, s = _sessions(tmp_path)
    a = make('A', s.data.copy(), True)
    monkeypatch.setattr(sample_prep, 'read_sidecar_checked', touch_then_read)
    doc = Document()
    doc.open_session(a)
    monkeypatch.undo()
    with pytest.raises(DocumentError, match='s1.npz has changed since'):
        doc.save_session(str(tmp_path / 'out.flowsession'))


# ── round 3 H1: no earlier save's sidecar found by name ─────────────────────

def test_a_raw_sample_saved_over_an_earlier_sidecar_save(tmp_path):
    """The reviewer's case 7: B (s1 from its sidecar, FL1-A recorded as
    asinh) saved to out, then a document holding s1 from its raw FCS saved
    to out. The entry has no sidecar pointer, so the opener looks for s1
    by name in out_data: B's files must be gone, and out reopens with the
    numbers of the raw document saved into a fresh folder."""
    make, s = _sessions(tmp_path)
    b = make('B', s.data, True)
    out = tmp_path / 'out.flowsession'
    doc_b = Document()
    doc_b.open_session(b)
    doc_b.save_session(str(out))
    raw = Document()
    name = raw.open_fcs(str(tmp_path / 'raw' / 's1.fcs'))
    assert name == 's1'
    want = raw.stats(name, channels=['FL1-A'], stats=('Median',))
    raw.save_session(str(tmp_path / 'fresh.flowsession'))
    raw.save_session(str(out))
    assert os.listdir(tmp_path / 'out_data') == []
    reopened = Document()
    assert reopened.open_session(str(out))['samples'] == ['s1']
    assert reopened._samples['s1'].sidecar == ''
    _n, stats_o = _reopened(str(out))
    _n, stats_f = _reopened(str(tmp_path / 'fresh.flowsession'))
    assert _medians(stats_o) == _medians(stats_f) == _medians(want)
    assert _medians(stats_o) != _medians(_reopened(b)[1])


def test_an_offline_sample_saved_over_an_earlier_sidecar_save(tmp_path):
    """A sample whose files were not found is written back with no
    sidecar pointer: an earlier save's s1.npz (and the CSV an older one
    wrote) in out_data must not be found for it by name. It reopens
    offline again."""
    make, s = _sessions(tmp_path)
    b = make('B', s.data, True)
    out = tmp_path / 'out.flowsession'
    doc_b = Document()
    doc_b.open_session(b)
    doc_b.save_session(str(out))
    (tmp_path / 'out_data' / 's1.csv').write_text('FL1-A\n1.0\n', 'utf-8')
    gone = tmp_path / 'away'
    gone.mkdir()
    d = json.loads((tmp_path / 'base.flowsession').read_text('utf-8'))
    for k in ('rel_path', 'basename', 'processed_data'):
        d['samples'][0].pop(k, None)
    d['samples'][0]['path'] = str(tmp_path / 'gone' / 's1.fcs')
    sess = gone / 'off.flowsession'
    sess.write_text(json.dumps(d), 'utf-8')
    doc = Document()
    assert doc.open_session(str(sess))['missing'] == ['s1']
    doc.save_session(str(out))
    assert os.listdir(tmp_path / 'out_data') == []
    again = Document()
    assert again.open_session(str(out))['missing'] == ['s1']


def test_a_raw_sample_whose_legacy_name_is_another_copy_is_refused(
        tmp_path):
    """Sample 'a b' (raw) would be looked for by name under the legacy
    stem a_b, which is sample a_b's sidecar copy: refused, naming it,
    before anything is written (it would reopen holding a_b's events)."""
    from openflo.session_sidecar import write_frame
    from tests.test_document import spill_fcs
    doc, name, *_ = gated_doc(tmp_path)
    frame = doc._samples[name].data.data
    side = tmp_path / 'run_data' / 'a_b.npz'
    side.parent.mkdir()
    assert write_frame(frame, str(side))
    doc.save_session(str(tmp_path / 'base.flowsession'))
    d = json.loads((tmp_path / 'base.flowsession').read_text('utf-8'))
    d['samples'][0]['name'] = 'a_b'
    d['samples'][0]['processed_data'] = 'run_data/a_b.npz'
    d['sample_gates'] = {'a_b': d['sample_gates'][name]}
    (tmp_path / 'run.flowsession').write_text(json.dumps(d), 'utf-8')
    both = Document()
    both.open_session(str(tmp_path / 'run.flowsession'))
    assert both._samples['a_b'].sidecar == str(side)
    assert both.open_fcs(spill_fcs(tmp_path / 'raw2' / 'a b.fcs')) == 'a b'
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match="sample a b would reopen from "
                                            "a_b.npz, another sample's saved "
                                            "data; remove one of the two "
                                            "samples"):
        both.save_session(str(tmp_path / 'out.flowsession'))
    assert _tree(tmp_path) == before


def test_a_by_name_file_to_remove_that_is_a_file_read_is_refused(tmp_path):
    """The by-name files a save removes for a raw sample are checked like
    every file it writes: a hard link there to the FCS the document read
    is refused before anything changes."""
    doc, _name, *_ = gated_doc(tmp_path)
    (tmp_path / 'out_data').mkdir()
    os.link(tmp_path / 'raw' / 's1.fcs', tmp_path / 'out_data' / 's1.npz')
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match='s1.npz is a file'):
        doc.save_session(str(tmp_path / 'out.flowsession'))
    assert _tree(tmp_path) == before


# ── round 4 review: the other format, shared folders, case, order ───────────

def _csv_session(tmp_path, s, stem, frame, record):
    """Session ``<stem>.flowsession`` whose sample s1 loads from the CSV
    sidecar ``<stem>_data/s1.csv`` holding `frame` (a CSV is still written
    for a frame the binary format cannot carry), with a transform record
    (FL1-A recorded as asinh) when `record`; returns its path."""
    from openflo.transform_sidecar import write_transforms_sidecar
    side = tmp_path / f'{stem}_data' / 's1.csv'
    side.parent.mkdir()
    frame.to_csv(side, index=False)
    if record:
        tr = dict(s.data_transforms, **{'FL1-A': ASINH})
        write_transforms_sidecar(str(side), tr, frame.columns, len(frame))
    d = json.loads((tmp_path / 'base.flowsession').read_text('utf-8'))
    d['samples'][0]['processed_data'] = f'{stem}_data/s1.csv'
    out = tmp_path / f'{stem}.flowsession'
    out.write_text(json.dumps(d), 'utf-8')
    return str(out)


def test_a_csv_sidecar_sample_is_saved_as_a_csv_copy(tmp_path):
    """A sample loaded from a CSV sidecar is saved with a copy of it and
    its record, and reopens with the numbers it was opened with."""
    _make, s = _sessions(tmp_path)
    frame = s.data.copy()
    frame['FL1-A'] = frame['FL1-A'] * 0.25
    c = _csv_session(tmp_path, s, 'C', frame, True)
    doc = Document()
    doc.open_session(c)
    want = doc.stats('s1', channels=['FL1-A'], stats=('Median',))
    out = tmp_path / 'out.flowsession'
    res = doc.save_session(str(out))
    assert res['sidecars'] == [str(tmp_path / 'out_data' / 's1.csv')]
    assert sorted(os.listdir(tmp_path / 'out_data')) == [
        's1.csv', 's1_transforms.json']
    again = Document()
    assert again.open_session(str(out))['notes'] == []
    assert again._samples['s1'].sidecar == str(tmp_path / 'out_data'
                                               / 's1.csv')
    got = again.stats('s1', channels=['FL1-A'], stats=('Median',))
    assert _medians(got) == _medians(want)


def test_a_csv_copy_removes_an_earlier_saves_npz_of_its_name(tmp_path):
    """The reviewer's p4c: out_data is a link out of the session folder,
    so every pointer is refused and s1 is found by name, .npz first. B
    (s1.npz) saved to out, then C (s1.csv, other data, with a record)
    saved over it: B's s1.npz must go, as the desktop's _write_sidecar
    removes the other format's file, or out reopens B's events with C's
    record."""
    make, s = _sessions(tmp_path)
    b = make('B', s.data, True)
    frame = s.data.copy()
    frame['FL1-A'] = frame['FL1-A'] * 0.25
    c = _csv_session(tmp_path, s, 'C', frame, True)
    proj = tmp_path / 'proj'
    proj.mkdir()
    ext = tmp_path / 'elsewhere'
    ext.mkdir()
    os.symlink(ext, proj / 'out_data')
    out = proj / 'out.flowsession'
    doc_b = Document()
    doc_b.open_session(b)
    doc_b.save_session(str(out))
    assert sorted(os.listdir(ext)) == ['s1.npz', 's1_transforms.json']
    doc_c = Document()
    doc_c.open_session(c)
    want = doc_c.stats('s1', channels=['FL1-A'], stats=('Median',))
    doc_c.save_session(str(out))
    assert sorted(os.listdir(ext)) == ['s1.csv', 's1_transforms.json']
    again = Document()
    assert again.open_session(str(out))['notes'] == []
    assert os.path.basename(again._samples['s1'].sidecar) == 's1.csv'
    got = again.stats('s1', channels=['FL1-A'], stats=('Median',))
    assert _medians(got) == _medians(want)
    assert _medians(got) != _medians(_reopened(b)[1])


def test_an_npz_copy_removes_an_earlier_saves_csv_of_its_name(tmp_path):
    """The other direction: C (s1.csv) saved to out, then B (s1.npz) saved
    over it: C's s1.csv goes, so no stale table is left beside the copy."""
    make, s = _sessions(tmp_path)
    b = make('B', s.data, True)
    c = _csv_session(tmp_path, s, 'C', s.data.copy(), False)
    out = tmp_path / 'out.flowsession'
    doc_c = Document()
    doc_c.open_session(c)
    doc_c.save_session(str(out))
    assert os.listdir(tmp_path / 'out_data') == ['s1.csv']
    doc_b = Document()
    doc_b.open_session(b)
    doc_b.save_session(str(out))
    assert sorted(os.listdir(tmp_path / 'out_data')) == [
        's1.npz', 's1_transforms.json']


def test_a_new_session_beside_another_sharing_its_data_folder_is_refused(
        tmp_path):
    """The reviewer's p4 case B: run.json (s1 from its sidecar, other
    data, with a record) owns run_data/s1.npz. A raw document saved to a
    NEW run.flowsession would remove those files (s1 is looked for by
    name there): refused, naming them, and run.json still reopens with its
    own numbers."""
    make, s = _sessions(tmp_path)
    frame = s.data.copy()
    frame['FL1-A'] = frame['FL1-A'] * 0.25
    b = make('B', frame, True)
    doc_b = Document()
    doc_b.open_session(b)
    other = tmp_path / 'run.json'
    doc_b.save_session(str(other))
    want = _reopened(str(other))[1]
    raw = Document()
    raw.open_fcs(str(tmp_path / 'raw' / 's1.fcs'))
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match=r'run_data already holds '
                                            r's1\.npz, s1_transforms\.json'):
        raw.save_session(str(tmp_path / 'run.flowsession'))
    assert _tree(tmp_path) == before
    assert _medians(_reopened(str(other))[1]) == _medians(want)


def test_a_new_session_whose_copy_would_replace_anothers_files_is_refused(
        tmp_path):
    """The round-5 reviewer's p5e: run.json owns run_data/s1.csv with a
    record. A sidecar document (s1.npz) saved to a NEW run.flowsession
    would remove that s1.csv and its record (the other format's file of
    its copy) and leave run.json reopening another session's events:
    refused, naming them, with nothing changed."""
    make, s = _sessions(tmp_path)
    frame = s.data.copy()
    frame['FL1-A'] = frame['FL1-A'] * 0.25
    c = _csv_session(tmp_path, s, 'C', frame, True)
    doc_c = Document()
    doc_c.open_session(c)
    other = tmp_path / 'run.json'
    doc_c.save_session(str(other))
    want = _reopened(str(other))[1]
    b = make('B', s.data, False)
    doc_b = Document()
    doc_b.open_session(b)
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match=r'run_data already holds '
                                            r's1\.csv, s1_transforms\.json'):
        doc_b.save_session(str(tmp_path / 'run.flowsession'))
    assert _tree(tmp_path) == before
    assert _medians(_reopened(str(other))[1]) == _medians(want)


def test_a_raw_sample_and_a_copy_differing_only_in_case_are_refused(
        tmp_path):
    """The reviewer's p4 case A: raw s1 beside sidecar sample S1 (other
    data). Where case folds (Windows, macOS, Android shared storage) s1
    would be found by name as S1.npz and reopen with S1's events, and a
    session written where case is told apart may be opened there: refused
    on every OS, naming the file, before anything is written."""
    make, s = _sessions(tmp_path)
    frame = s.data.copy()
    frame['FL1-A'] = frame['FL1-A'] * 0.25
    b = make('B', frame, False)
    d = json.loads(open(b, encoding='utf-8').read())
    d['samples'][0]['name'] = 'S1'
    d['sample_gates'] = {'S1': d['sample_gates']['s1']}
    d['active_sample'] = 'S1'
    with open(b, 'w', encoding='utf-8') as f:
        f.write(json.dumps(d))
    doc = Document()
    doc.open_session(b)
    assert doc.open_fcs(str(tmp_path / 'raw' / 's1.fcs')) == 's1'
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match=r'sample s1 would reopen from '
                                            r's1\.npz'):
        # A capital in the folder: the whole path is compared folded.
        doc.save_session(str(tmp_path / 'Out.flowsession'))
    assert _tree(tmp_path) == before


def test_a_failed_write_keeps_the_earlier_saves_by_name_sidecar(
        tmp_path, monkeypatch):
    """A raw document saved over an earlier sidecar save fails while the
    session is written (a full disk, say): the by-name files are removed
    only once everything is written, so the earlier save keeps its
    sidecar and record."""
    make, s = _sessions(tmp_path)
    b = make('B', s.data, True)
    out = tmp_path / 'out.flowsession'
    doc_b = Document()
    doc_b.open_session(b)
    doc_b.save_session(str(out))
    raw = Document()
    raw.open_fcs(str(tmp_path / 'raw' / 's1.fcs'))
    before = _tree(tmp_path)

    def full(*_a, **_k):
        raise OSError(28, 'No space left on device (planted)')
    monkeypatch.setattr(document, '_write_json', full)
    with pytest.raises(OSError, match='planted'):
        raw.save_session(str(out))
    monkeypatch.undo()
    assert _tree(tmp_path) == before


def test_the_opener_tries_npz_before_csv(tmp_path):
    """A pointer-less sample with both s1.npz and s1.csv in its data
    folder is read from s1.npz (the format the writer prefers), as
    sidecar_name_candidates states."""
    from openflo.session_open import resolve_processed_path, sidecar_name_candidates
    data_dir = tmp_path / 'run_data'
    data_dir.mkdir()
    (data_dir / 's1.csv').write_text('FL1-A\n1.0\n', 'utf-8')
    (data_dir / 's1.npz').write_bytes(b'')
    assert [os.path.basename(p) for p in
            sidecar_name_candidates('s1', str(data_dir))] == [
        's1.npz', 's1.csv']
    got = resolve_processed_path({'processed_data': None,
                                  'path': '/raw/s1.fcs'},
                                 's1', str(tmp_path), str(data_dir))
    assert os.path.basename(got) == 's1.npz'


def test_a_file_where_the_data_folder_goes_is_refused(tmp_path):
    """A regular file at out_data: the save is refused with a DocumentError
    naming it, not an OS error, and nothing changes."""
    doc, _name, *_ = gated_doc(tmp_path)
    (tmp_path / 'out_data').write_text('not a folder', 'utf-8')
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match='out_data is a file'):
        doc.save_session(str(tmp_path / 'out.flowsession'))
    assert _tree(tmp_path) == before
