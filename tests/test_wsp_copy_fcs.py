"""File > Export > FlowJo workspace copies each sample's FCS file beside the
.wsp (on by default; File > Export > Copy FCS files with FlowJo workspace),
and the workspace names each file relative to itself, so the folder opens
in FlowJo on any computer. Originals are only read; a file already in that
folder is never overwritten; a sample whose FCS file is gone is warned of.

Generated data, synthetic sample names and generic channels only."""
from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from openflo import wsp_export
from openflo.wsp_export import copy_fcs_beside
from tests._fcs_writer import write_float_fcs


def _fcs(path, seed=0, n=300):
    """A generated two-channel FCS file at `path`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ev = np.random.default_rng(seed).uniform(0, 1000, (n, 2))
    return write_float_fcs(path, ev, ['FL1-A', 'FL2-A'])


def _fingerprint(path):
    st = os.stat(path)
    with open(path, 'rb') as f:
        return f.read(), st.st_mtime_ns


def _bytes(path):
    with open(path, 'rb') as f:
        return f.read()


# ── copy_fcs_beside ──────────────────────────────────────────────────────────

def test_copies_each_file_beside_the_workspace_and_leaves_the_original(tmp_path):
    a = _fcs(str(tmp_path / 'raw' / 'a.fcs'), seed=1)
    b = _fcs(str(tmp_path / 'raw' / 'b c.fcs'), seed=2)
    before = {p: _fingerprint(p) for p in (a, b)}
    out = tmp_path / 'out'
    out.mkdir()
    paths, missing = copy_fcs_beside({'s1': a, 's2': b}, str(out / 'w.wsp'))
    assert missing == []
    assert paths == {'s1': str(out / 'a.fcs'), 's2': str(out / 'b c.fcs')}
    for name, src in (('s1', a), ('s2', b)):
        assert _bytes(paths[name]) == _bytes(src)
        # The copy keeps the file's dates, as a file manager's copy does.
        assert os.stat(paths[name]).st_mtime_ns == os.stat(src).st_mtime_ns
    assert {p: _fingerprint(p) for p in (a, b)} == before
    assert sorted(os.listdir(out)) == ['a.fcs', 'b c.fcs']      # no .part left


def test_a_different_file_of_that_name_there_is_never_overwritten(tmp_path):
    src = _fcs(str(tmp_path / 'raw' / 'a.fcs'), seed=1)
    out = tmp_path / 'out'
    other = _fcs(str(out / 'a.fcs'), seed=9)
    kept = _bytes(other)
    paths, _ = copy_fcs_beside({'s1': src}, str(out / 'w.wsp'))
    assert _bytes(other) == kept
    assert paths['s1'] == str(out / 'a (2).fcs')
    assert _bytes(paths['s1']) == _bytes(src)


def test_a_file_already_beside_the_workspace_is_used_not_copied(tmp_path):
    out = tmp_path / 'out'
    here = _fcs(str(out / 'a.fcs'), seed=1)
    paths, _ = copy_fcs_beside({'s1': here}, str(out / 'w.wsp'))
    assert paths == {'s1': here} and os.listdir(out) == ['a.fcs']
    # The same bytes under that name elsewhere: also used, not copied again.
    twin = str(tmp_path / 'raw' / 'a.fcs')
    os.makedirs(os.path.dirname(twin))
    shutil.copyfile(here, twin)
    paths, _ = copy_fcs_beside({'s1': twin}, str(out / 'w.wsp'))
    assert paths == {'s1': here} and os.listdir(out) == ['a.fcs']


@pytest.mark.parametrize('first, second', [('x.fcs', 'x.fcs'),
                                           ('X.fcs', 'x.fcs'),
                                           ('x.fcs', 'X.fcs')])
def test_different_files_of_one_name_each_get_their_own_copy(
        tmp_path, first, second):
    # Names differing only in case are one name on Windows and macOS.
    one = _fcs(str(tmp_path / 'one' / first), seed=1)
    two = _fcs(str(tmp_path / 'two' / second), seed=2)
    out = tmp_path / 'out'
    out.mkdir()
    paths, _ = copy_fcs_beside({'s1': one, 's2': two}, str(out / 'w.wsp'))
    stem = os.path.splitext(second)[0]
    assert paths == {'s1': str(out / first), 's2': str(out / f'{stem} (2).fcs')}
    assert _bytes(paths['s1']) == _bytes(one)
    assert _bytes(paths['s2']) == _bytes(two)


def test_samples_of_one_file_share_one_copy(tmp_path):
    src = _fcs(str(tmp_path / 'raw' / 'plate.fcs'))
    out = tmp_path / 'out'
    out.mkdir()
    paths, _ = copy_fcs_beside({'w1': src, 'w2': src}, str(out / 'w.wsp'))
    assert paths == {'w1': str(out / 'plate.fcs'), 'w2': str(out / 'plate.fcs')}
    assert os.listdir(out) == ['plate.fcs']


def test_wells_of_one_file_read_it_once(tmp_path, monkeypatch):
    # 96 wells of one plate file: the copy is not read back once per well.
    src = _fcs(str(tmp_path / 'raw' / 'plate.fcs'))
    out = tmp_path / 'out'
    out.mkdir()
    compared = []
    real = wsp_export._same_bytes
    monkeypatch.setattr(wsp_export, '_same_bytes',
                        lambda a, b: compared.append(a) or real(a, b))
    wells = {f'w{i}': src for i in range(96)}
    paths, _ = copy_fcs_beside(wells, str(out / 'w.wsp'))
    assert set(paths.values()) == {str(out / 'plate.fcs')}
    assert compared == []


def test_a_missing_file_is_reported_and_keeps_its_path(tmp_path):
    src = _fcs(str(tmp_path / 'raw' / 'a.fcs'))
    gone = str(tmp_path / 'raw' / 'gone.fcs')
    out = tmp_path / 'out'
    out.mkdir()
    paths, missing = copy_fcs_beside({'s1': src, 's2': gone, 's3': ''},
                                     str(out / 'w.wsp'))
    assert missing == ['s2', 's3']
    assert paths['s2'] == gone and paths['s3'] == ''
    assert os.listdir(out) == ['a.fcs']


def test_a_failed_copy_leaves_no_half_file(tmp_path, monkeypatch):
    src = _fcs(str(tmp_path / 'raw' / 'a.fcs'))
    out = tmp_path / 'out'
    out.mkdir()

    written = []

    def half(s, d):
        written.append(os.path.basename(d))
        with open(d, 'wb') as f:
            f.write(b'half')
        raise OSError('disk full')
    monkeypatch.setattr(shutil, 'copyfile', half)
    with pytest.raises(OSError):
        copy_fcs_beside({'s1': src}, str(out / 'w.wsp'))
    assert os.listdir(out) == []
    # Written under a temporary name: a copy cut short (the program
    # killed) is never mistaken for the FCS file.
    assert len(written) == 1
    assert written[0].startswith('.a.fcs.') and written[0].endswith('.part')


def test_a_users_own_part_file_is_left_alone(tmp_path):
    src = _fcs(str(tmp_path / 'raw' / 'c.fcs'))
    out = tmp_path / 'out'
    out.mkdir()
    for mine in ('c.fcs.part', '.c.fcs.part'):
        (out / mine).write_bytes(b'mine')
    paths, _ = copy_fcs_beside({'s1': src}, str(out / 'w.wsp'))
    assert _bytes(paths['s1']) == _bytes(src)
    assert sorted(os.listdir(out)) == ['.c.fcs.part', 'c.fcs', 'c.fcs.part']
    assert _bytes(out / 'c.fcs.part') == _bytes(out / '.c.fcs.part') == b'mine'


def test_same_size_and_date_but_other_bytes_is_not_reused(tmp_path):
    # Fixed event counts give same-sized files; copies keep dates. Only the
    # bytes tell two such files apart.
    src = _fcs(str(tmp_path / 'raw' / 'a.fcs'), seed=1)
    out = tmp_path / 'out'
    other = _fcs(str(out / 'a.fcs'), seed=2)
    assert os.path.getsize(other) == os.path.getsize(src)
    st = os.stat(src)
    os.utime(other, ns=(st.st_atime_ns, st.st_mtime_ns))
    kept = _bytes(other)
    paths, _ = copy_fcs_beside({'s1': src}, str(out / 'w.wsp'))
    assert paths['s1'] == str(out / 'a (2).fcs')
    assert _bytes(paths['s1']) == _bytes(src) and _bytes(other) == kept


def test_a_reused_files_name_is_taken_for_the_next(tmp_path):
    # a.fcs is already there (same bytes); another file named A.fcs must
    # not land beside it as A.fcs: one name on Windows and macOS.
    out = tmp_path / 'out'
    here = _fcs(str(out / 'a.fcs'), seed=1)
    other = _fcs(str(tmp_path / 'raw' / 'A.fcs'), seed=2)
    paths, _ = copy_fcs_beside({'s1': here, 's2': other}, str(out / 'w.wsp'))
    assert paths == {'s1': here, 's2': str(out / 'A (2).fcs')}


def test_the_option_is_on_unless_turned_off(monkeypatch):
    ee = pytest.importorskip('openflo.editor_export')    # needs Tk
    monkeypatch.setattr(ee, 'read_prefs', dict)
    assert ee.wsp_copy_fcs_enabled() is True
    monkeypatch.setattr(ee, 'read_prefs',
                        lambda: {ee.WSP_COPY_FCS_PREF: False})
    assert ee.wsp_copy_fcs_enabled() is False


# ── the editor's File > Export > FlowJo workspace ───────────────────────────

def _uris(path):
    return [d.get('uri') for d in ET.parse(path).getroot().iter('DataSet')]


def _export(tmp_path, monkeypatch, copy, samples):
    """Run the editor's .wsp export to tmp_path/out/w.wsp with the copy
    option `copy` on `samples` ({name: FCS path}). Returns (wsp path,
    warning titles and texts shown)."""
    from tests.test_gui_ux import _editor_or_skip, _load_sample
    root, ed = _editor_or_skip()
    import openflo.editor_export as ee
    out = tmp_path / 'out'
    out.mkdir(exist_ok=True)
    wsp = out / 'w.wsp'
    shown = []
    monkeypatch.setattr(ee.filedialog, 'asksaveasfilename',
                        lambda *a, **k: str(wsp))
    monkeypatch.setattr(ee.messagebox, 'showwarning',
                        lambda title, msg, **k: shown.append((title, msg)))
    monkeypatch.setattr(ee, 'wsp_copy_fcs_enabled', lambda: copy)
    try:
        ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
            on_done(work()) if on_done else work())
        for name, path in samples.items():
            s = _load_sample(ed, name, cols=('FL1-A', 'FL2-A'), n=50)
            s.path = path
        ed._export_flowjo_wsp()
        status = ed.status_var.get()
    finally:
        root.destroy()
    return wsp, shown, status


def test_export_copies_the_files_and_names_them_relative(tmp_path, monkeypatch):
    a = _fcs(str(tmp_path / 'raw' / 'a.fcs'), seed=1)
    b = _fcs(str(tmp_path / 'raw' / 'b c.fcs'), seed=2)
    before = {p: _fingerprint(p) for p in (a, b)}
    wsp, shown, status = _export(tmp_path, monkeypatch, True,
                                 {'s1': a, 's2': b})
    assert _uris(wsp) == ['file:a.fcs', 'file:b%20c.fcs']
    assert _bytes(wsp.parent / 'a.fcs') == _bytes(a)
    assert _bytes(wsp.parent / 'b c.fcs') == _bytes(b)
    assert {p: _fingerprint(p) for p in (a, b)} == before
    assert shown == []
    assert '+ 2 FCS file(s) beside it' in status


def test_a_renamed_copy_is_the_one_the_workspace_names(tmp_path, monkeypatch):
    a = _fcs(str(tmp_path / 'raw' / 'a.fcs'), seed=1)
    other = _fcs(str(tmp_path / 'out' / 'a.fcs'), seed=9)
    kept = _bytes(other)
    wsp, shown, _ = _export(tmp_path, monkeypatch, True, {'s1': a})
    assert _uris(wsp) == ['file:a%20%282%29.fcs']      # 'a (2).fcs'
    assert _bytes(other) == kept
    assert _bytes(wsp.parent / 'a (2).fcs') == _bytes(a)
    assert shown == []


def test_the_folder_opens_with_its_files_after_moving(tmp_path, monkeypatch):
    from openflo.document import Document
    a = _fcs(str(tmp_path / 'raw' / 'a.fcs'), seed=1)
    wsp, *_ = _export(tmp_path, monkeypatch, True, {'s1': a})
    moved = tmp_path / 'moved'
    shutil.move(str(wsp.parent), str(moved))
    os.remove(a)                     # the original is gone on this computer
    doc = Document()
    r = doc.import_wsp(str(moved / 'w.wsp'))
    assert r['unresolved'] == []
    assert doc.samples()[0]['path'] == str(moved / 'a.fcs')


def test_export_with_copying_off_keeps_absolute_paths(tmp_path, monkeypatch):
    from openflo.wsp_writer import _wsp_dataset_uri
    a = _fcs(str(tmp_path / 'raw' / 'a.fcs'), seed=1)
    wsp, shown, status = _export(tmp_path, monkeypatch, False, {'s1': a})
    assert _uris(wsp) == [_wsp_dataset_uri(a)]
    assert os.listdir(wsp.parent) == ['w.wsp']
    assert shown == [] and 'FCS file(s)' not in status


def test_export_warns_of_a_missing_fcs_file(tmp_path, monkeypatch):
    a = _fcs(str(tmp_path / 'raw' / 'a.fcs'), seed=1)
    gone = str(tmp_path / 'raw' / 'gone.fcs')
    wsp, shown, status = _export(tmp_path, monkeypatch, True,
                                 {'s1': a, 's2': gone})
    from openflo.wsp_writer import _wsp_dataset_uri
    assert [t for t, _ in shown] == ['FCS files not found']
    msg = shown[0][1]
    assert '• s2' in msg and '• s1' not in msg
    # Its gates are still written, at its own absolute path: FlowJo asks
    # for the file.
    assert _uris(wsp) == ['file:a.fcs', _wsp_dataset_uri(gone)]
    assert '+ 1 FCS file(s) beside it' in status


def test_a_missing_file_never_takes_a_same_named_file_there(tmp_path,
                                                            monkeypatch):
    from openflo.wsp_writer import _wsp_dataset_uri
    # P1: s2's a.fcs is gone; s1's a.fcs (another file) is copied there.
    a = _fcs(str(tmp_path / 'x' / 'a.fcs'), seed=1)
    gone_a = str(tmp_path / 'y' / 'a.fcs')
    # P2: s3's b.fcs is gone; an unrelated b.fcs is in the export folder.
    _fcs(str(tmp_path / 'out' / 'b.fcs'), seed=5)
    gone_b = str(tmp_path / 'gone' / 'b.fcs')
    wsp, shown, _ = _export(tmp_path, monkeypatch, True,
                            {'s1': a, 's2': gone_a, 's3': gone_b})
    assert _uris(wsp) == ['file:a.fcs', _wsp_dataset_uri(gone_a),
                          _wsp_dataset_uri(gone_b)]
    assert '• s2' in shown[0][1] and '• s3' in shown[0][1]


def test_the_warning_lists_twelve_and_counts_the_rest(tmp_path, monkeypatch):
    gone = {f's{i:02d}': str(tmp_path / 'gone' / f'f{i}.fcs')
            for i in range(14)}
    _, shown, status = _export(tmp_path, monkeypatch, True, gone)
    msg = shown[0][1]
    assert msg.startswith('14 sample(s)')
    assert '• s11' in msg and '• s12' not in msg
    assert msg.endswith('… and 2 more')
    assert '+ 0 FCS file(s) beside it' in status


def test_wells_of_one_file_count_as_one_copied_file(tmp_path, monkeypatch):
    plate = _fcs(str(tmp_path / 'raw' / 'plate.fcs'))
    _, _, status = _export(tmp_path, monkeypatch, True,
                           {'w1': plate, 'w2': plate, 'w3': plate})
    assert '+ 1 FCS file(s) beside it' in status


@pytest.mark.parametrize('saved', [True, False])
def test_the_menu_tick_shows_the_saved_choice(monkeypatch, saved):
    from tests.conftest import editor_or_skip
    ee = pytest.importorskip('openflo.editor_export')
    monkeypatch.setattr(ee, 'read_prefs',
                        lambda: {ee.WSP_COPY_FCS_PREF: saved})
    root, ed = editor_or_skip()
    try:
        for b in ed._menubar_buttons:
            b.destroy()
        ed._build_menubar()
        assert ed._wsp_copy_fcs_var.get() is saved
    finally:
        root.destroy()
