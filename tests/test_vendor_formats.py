"""Instrument files OpenFlo cannot read say how to export FCS from them.

MACSQuant .mqd, NovoExpress .ncf, Sony .sraw and Attune CytPix .acs files
are each instrument's own undocumented format. Opening, dropping or pasting
one shows a short message naming the program and how to get FCS out of it
(openflo.vendor_formats), never a parse error or a silent skip. A file is
recognised by its extension only.

Every file here is generated: the instrument files are a few junk bytes
under the extension (real ones differ, and are never parsed), and the FCS
files come from tests/_fcs_writer with generic channels FL1-A and FL2-A.
"""
from __future__ import annotations

import os
import tkinter.messagebox
from types import SimpleNamespace

import numpy as np
import pytest

from openflo.exceptions import CompensationError, FcsParseError
from openflo.vendor_formats import (
    OTHER_PROGRAM,
    VENDOR_FORMATS,
    vendor_format,
    vendor_message,
    vendor_status,
    vendor_summary,
)
from tests._fcs_writer import write_fcs

NAMES = ['FL1-A', 'FL2-A']
PROGRAMS = {'.mqd': ('MACSQuant', 'MACSQuantify', "FCS 3.1 (not 'FCS 3.1 "
                     "compatible')"),
            '.ncf': ('NovoExpress',
                     'Sample > Import and Export > Export FCS File'),
            '.sraw': ('Sony', 'same name', 'or export FCS from the Sony '
                      'software'),
            '.acs': ('Attune', 'right-click the experiment and choose '
                     'Export FCS Files')}
CANT = "OpenFlo can't read it directly."


def _junk(path):
    """A stand-in instrument file: bytes no reader takes for FCS."""
    path.write_bytes(b'\x00\x01not an FCS file\x00' * 8)
    return path


def _fcs(path, n=200):
    rng = np.random.default_rng(0)
    return write_fcs(path, rng.integers(1, 1000, (n, 2)), NAMES)


# ── the table ───────────────────────────────────────────────────────────────

def test_the_table_holds_exactly_the_confirmed_formats():
    assert set(VENDOR_FORMATS) == set(PROGRAMS)


@pytest.mark.parametrize('ext', sorted(PROGRAMS))
def test_each_format_names_its_program_and_how_to_export(ext):
    msg = vendor_message(f'/x/tube1{ext.upper()}')
    assert msg is not None
    assert CANT in msg
    for word in PROGRAMS[ext]:
        assert word in msg, (ext, msg)
    assert vendor_message(f'tube1{ext}') == msg   # any case


@pytest.mark.parametrize('ext', ['.mqd', '.ncf', '.sraw'])
def test_an_extension_only_one_program_uses_says_what_it_is(ext):
    msg = vendor_message(f'tube1{ext}')
    assert msg.startswith('This is ') and OTHER_PROGRAM not in msg


def test_acs_only_looks_like_attune_and_covers_other_programs():
    # Chemometec image cytometers write .acs too.
    assert vendor_message('/x/tube1.ACS') == (
        "This looks like an Attune CytPix file (.acs). OpenFlo can't read "
        "it directly. In the Attune software, right-click the experiment "
        "and choose Export FCS Files, then open those files. If it came "
        "from another instrument, export FCS from that program.")
    text = vendor_summary(['/d/a.acs', '/d/b.acs'])
    assert text.startswith('2 Attune CytPix file(s) not opened: a.acs, b.acs.')
    assert text.endswith(OTHER_PROGRAM)
    assert OTHER_PROGRAM not in vendor_summary(['/d/a.mqd', '/d/b.mqd'])


def test_the_mqd_message_is_the_plain_one():
    assert vendor_message('a.mqd') == (
        "This is a MACSQuant file. OpenFlo can't read it directly. In "
        "MACSQuantify, export the data as FCS 3.1 (not 'FCS 3.1 "
        "compatible'), then open the FCS file.")


@pytest.mark.parametrize('name', [
    'a.fcs', 'a.lmd', 'a.wsp', 'a.xml', 'a.csv',
    # Accuri files get a real reader of their own, not this message.
    'a.c6', 'a.cfl', 'a.ci',
    # Not confirmed by a public page: what they hold, or how to export.
    'a.nxt', 'a.tmo',
    'mqd', 'a.mqd.fcs', 'a.mqdx', ''])
def test_other_names_are_not_instrument_files(name):
    assert vendor_format(name) is None and vendor_message(name) is None


def test_a_summary_groups_several_files_by_format():
    paths = ['/d/b.mqd', '/d/a.mqd', '/d/c.sraw', '/d/x.fcs']
    text = vendor_summary(paths)
    paras = text.split('\n\n')
    assert len(paras) == 2, text
    assert paras[0].startswith('2 MACSQuant file(s) not opened: b.mqd, a.mqd.')
    assert 'MACSQuantify' in paras[0] and 'x.fcs' not in text
    assert paras[1].startswith('1 Sony spectral raw file(s) not opened: c.sraw.')
    assert vendor_summary(['/d/x.fcs']) == ''


def test_a_summary_of_one_file_is_its_message():
    assert vendor_summary(['/d/a.ncf']) == f"a.ncf: {vendor_message('a.ncf')}"
    assert vendor_status(['/d/a.ncf']) == vendor_summary(['/d/a.ncf'])


def test_a_long_list_is_cut_with_a_count():
    paths = [f'/d/t{i:02d}.mqd' for i in range(11)]
    for text in (vendor_summary(paths, cap=8), vendor_summary(paths)):
        assert 't07.mqd and 3 more.' in text and 't08.mqd' not in text
    assert vendor_status(paths) == (
        "Not opened: 11 instrument file(s) OpenFlo can't read directly. "
        "Export them as FCS first.")


def test_the_dialog_pattern_lists_every_table_extension():
    from openflo.vendor_formats import VENDOR_FILE_PATTERNS
    assert sorted(VENDOR_FILE_PATTERNS.split()) == sorted(
        f'*{ext}' for ext in VENDOR_FORMATS)


# ── the readers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('ext', sorted(PROGRAMS))
def test_loading_one_raises_its_message_alone(tmp_path, ext):
    from openflo.flow_sample import FlowSample
    p = _junk(tmp_path / f'tube1{ext}')
    with pytest.raises(FcsParseError) as got:
        FlowSample(str(p))
    assert str(got.value) == vendor_message(p)


def test_the_decision_is_by_extension_only(tmp_path):
    # A real FCS file named oddly still opens; the same bytes named .mqd
    # get the message and are not parsed.
    from openflo.flow_sample import FlowSample
    for name in ('tube1.dat', 'tube1.FCS', 'tube1'):
        s = FlowSample(str(_fcs(tmp_path / name)))
        assert len(s.data) == 200, name
    q = _fcs(tmp_path / 'tube2.mqd')
    with pytest.raises(FcsParseError, match='This is a MACSQuant file'):
        FlowSample(str(q))


def test_a_real_fcs_still_opens(tmp_path):
    from openflo.flow_sample import FlowSample
    s = FlowSample(str(_fcs(tmp_path / 'tube1.fcs')))
    assert s.channel_names[:2] == NAMES and len(s.data) == 200


def test_the_fcs_inspector_says_it_too(tmp_path):
    from openflo.ui_inspect import read_fcs_metadata
    p = _junk(tmp_path / 'tube1.sraw')
    with pytest.raises(FcsParseError, match='same name'):
        read_fcs_metadata(str(p))


def test_a_compensation_load_says_it_too(tmp_path):
    from openflo.compensation_io import read_compensation_matrix
    p = _junk(tmp_path / 'ctrl.acs')
    with pytest.raises(CompensationError) as got:
        read_compensation_matrix(str(p))
    assert str(got.value) == f"ctrl.acs: {vendor_message(p)}"


# ── drops and pastes, without Tk ────────────────────────────────────────────

def _mixed_folder(tmp_path):
    d = tmp_path / 'run'
    (d / 'sub').mkdir(parents=True)
    _fcs(d / 'a.fcs')
    _fcs(d / 'sub' / 'b.fcs')
    _junk(d / 'c.mqd')
    _junk(d / 'sub' / 'd.NCF')
    _junk(d / 'sub' / 'e.dat')                 # not in the table: ignored
    (d / 'notes.txt').write_text('x')
    return d


def test_a_dropped_folder_hands_back_its_instrument_files(tmp_path):
    from openflo.paths import expand_dropped_paths
    d = _mixed_folder(tmp_path)
    vendor = []
    fcs, wsp = expand_dropped_paths([str(d)], vendor)
    assert [os.path.basename(p) for p in fcs] == ['a.fcs', 'b.fcs']
    assert wsp == []
    assert [os.path.basename(p) for p in vendor] == ['c.mqd', 'd.NCF']
    # Without the list the result is as before.
    assert expand_dropped_paths([str(d)]) == (fcs, wsp)


def test_pasted_paths_split_after_an_instrument_extension(tmp_path):
    from openflo.editor_clipboard import _PATH_END
    a = tmp_path / 'my run' / 'x.SRAW'
    assert _PATH_END.split(f'{a} {tmp_path}/y.fcs {tmp_path}/z.acs') == [
        str(a), f'{tmp_path}/y.fcs', f'{tmp_path}/z.acs']


# ── the command line ────────────────────────────────────────────────────────

def test_openflo_run_notes_instrument_files_in_its_folders(tmp_path, capsys):
    from openflo.cli_main import _resolve_groups
    d = _mixed_folder(tmp_path)
    args = SimpleNamespace(trials=str(d), fcs='', groups='', samples='',
                           fmo_sets='')
    _resolve_groups(args)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith('[input]')]
    assert len(lines) == 1, out
    assert 'c.mqd' in out and 'd.NCF' in out and 'MACSQuantify' in out


def test_openflo_run_says_nothing_without_instrument_files(tmp_path, capsys):
    from openflo.cli_groups import vendor_files_note
    _fcs(tmp_path / 'a.fcs')
    assert vendor_files_note([str(tmp_path)]) == ''


def test_unmix_skips_an_instrument_file_with_the_note(tmp_path, capsys):
    from openflo.cli_unmix import _resolve_inputs
    f = _fcs(tmp_path / 'a.fcs')
    m = _junk(tmp_path / 'b.mqd')
    args = SimpleNamespace(unmix_input=f'{f},{m}', fcs=None, trials=None)
    assert _resolve_inputs(args, {}, None) == [str(f)]
    out = capsys.readouterr().out
    assert out.startswith('[unmix] b.mqd: This is a MACSQuant file.'), out


# ── the desktop app ─────────────────────────────────────────────────────────

@pytest.fixture
def editor(monkeypatch, tmp_path):
    from tests.conftest import editor_or_skip
    from tests.test_transform_records import isolate_home
    isolate_home(monkeypatch, tmp_path)
    root, ed = editor_or_skip()
    queued = []
    monkeypatch.setattr(ed, '_enqueue_load',
                        lambda job, priority=1: queued.append(job))
    shown = []
    monkeypatch.setattr(tkinter.messagebox, 'showinfo',
                        lambda title, msg, **k: shown.append((title, msg)))
    yield ed, queued, shown
    root.destroy()


def _pick(monkeypatch, paths):
    from openflo import editor_load
    monkeypatch.setattr(editor_load.filedialog, 'askopenfilenames',
                        lambda **k: tuple(str(p) for p in paths))


@pytest.mark.parametrize('ext', sorted(PROGRAMS))
def test_add_fcs_shows_the_message_for_one_file(editor, tmp_path,
                                                monkeypatch, ext):
    ed, queued, shown = editor
    p = _junk(tmp_path / f'tube1{ext}')
    _pick(monkeypatch, [p])
    ed._add_samples()
    assert queued == []
    want = f"tube1{ext}: {vendor_message(p)}"
    assert shown == [("Export as FCS first", want)]
    assert ed.status_var.get() == want


def test_the_add_fcs_dialog_shows_instrument_files(editor, monkeypatch):
    # Its first filter, the one a user browsing a folder sees, lists every
    # table extension; so does an entry of their own.
    from openflo import editor_load
    ed, _queued, _shown = editor
    seen = {}

    def pick(**k):
        seen.update(k)
        return ()
    monkeypatch.setattr(editor_load.filedialog, 'askopenfilenames', pick)
    ed._add_samples()
    first_name, first = seen['filetypes'][0]
    assert first_name == 'FCS & workspaces'
    for ext in VENDOR_FORMATS:
        assert f'*{ext}' in first.split(), (ext, first)
    assert {'*.fcs', '*.lmd', '*.wsp', '*.xml'} <= set(first.split())
    own = dict(seen['filetypes'])['Instrument files (export as FCS first)']
    assert sorted(own.split()) == sorted(f'*{e}' for e in VENDOR_FORMATS)


def test_a_skipped_file_and_an_instrument_file_keep_both_notes(
        editor, tmp_path, monkeypatch):
    # Nothing is queued, but the skipped note stays on the status line
    # ahead of the instrument-file note.
    ed, queued, shown = editor
    gone = tmp_path / 'gone.fcs'                 # never written: missing
    m = _junk(tmp_path / 'tube2.mqd')
    _pick(monkeypatch, [gone, m])
    ed._add_samples()
    assert queued == []
    assert len(shown) == 1 and 'tube2.mqd' in shown[0][1]
    status = ed.status_var.get()
    assert status.startswith('Queued 0.  Skipped: gone.fcs(missing)'), status
    assert status.endswith(vendor_status([m])), status


def test_add_fcs_with_a_real_file_still_queues_it(editor, tmp_path,
                                                  monkeypatch):
    ed, queued, shown = editor
    f = _fcs(tmp_path / 'tube1.fcs')
    m = _junk(tmp_path / 'tube2.mqd')
    _pick(monkeypatch, [f, m])
    ed._add_samples()
    assert queued == [('tube1', str(f))]
    assert len(shown) == 1 and 'tube2.mqd' in shown[0][1]
    status = ed.status_var.get()
    assert status.startswith('Queued 1.') and 'MACSQuant' in status, status


def test_a_real_fcs_alone_shows_no_message(editor, tmp_path, monkeypatch):
    ed, queued, shown = editor
    f = _fcs(tmp_path / 'tube1.fcs')
    _pick(monkeypatch, [f])
    ed._add_samples()
    assert queued == [('tube1', str(f))] and shown == []


def test_a_mixed_folder_drop_lists_every_instrument_file_once(
        editor, tmp_path):
    ed, queued, shown = editor
    d = _mixed_folder(tmp_path)
    event = SimpleNamespace(data='{' + str(d) + '}', action='copy')
    assert ed._on_dnd_drop(event) == 'copy'
    assert sorted(n for n, _p in queued) == ['a', 'b']
    assert len(shown) == 1, shown
    text = shown[0][1]
    assert 'c.mqd' in text and 'd.NCF' in text and 'e.dat' not in text
    assert 'MACSQuantify' in text and 'NovoExpress' in text


def test_dropping_only_instrument_files_names_them(editor, tmp_path):
    ed, queued, shown = editor
    a = _junk(tmp_path / 'a.acs')
    ed._import_dropped_paths([str(a)])
    assert queued == []
    assert shown == [("Export as FCS first", f"a.acs: {vendor_message(a)}")]
    assert 'no .fcs' not in ed.status_var.get()


def test_a_pasted_instrument_path_shows_the_message(editor, tmp_path,
                                                    monkeypatch):
    ed, queued, shown = editor
    m = _junk(tmp_path / 'x.sraw')
    monkeypatch.setattr(ed, 'clipboard_get', lambda: str(m))
    ed._on_paste()
    assert queued == []
    assert shown == [("Export as FCS first", f"x.sraw: {vendor_message(m)}")]


# ── the workspace window's batch folder ─────────────────────────────────────

class _BatchView:
    def __init__(self):
        self.model = SimpleNamespace(run_cfg={'method': 'flowsom'})
        self._editor = None

    def _sync_run_cfg(self):
        return {}


def _batch_folder(tmp_path, monkeypatch, folder):
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb

    from openflo import workspace as W
    answers = [str(folder), str(tmp_path)]
    said = []
    monkeypatch.setattr(fd, 'askdirectory', lambda **k: answers.pop(0))
    monkeypatch.setattr(mb, 'askyesno', lambda t, m, **k: said.append(m))
    monkeypatch.setattr(mb, 'showinfo', lambda t, m, **k: said.append(m))
    W.PipelineWorkspaceView._batch_folder(_BatchView())
    return said


def test_a_batch_folder_of_instrument_files_says_how_to_export(
        tmp_path, monkeypatch):
    d = tmp_path / 'in'
    d.mkdir()
    m = _junk(d / 'a.mqd')
    said = _batch_folder(tmp_path, monkeypatch, d)
    assert said == [f"a.mqd: {vendor_message(m)}"], said


def test_a_mixed_batch_folder_names_the_files_it_leaves_out(
        tmp_path, monkeypatch):
    d = tmp_path / 'in'
    d.mkdir()
    _fcs(d / 'a.fcs')
    _junk(d / 'b.ncf')
    said = _batch_folder(tmp_path, monkeypatch, d)
    assert len(said) == 1 and ' on 1 file(s)?' in said[0], said
    assert 'b.ncf: This is a NovoExpress file.' in said[0]
