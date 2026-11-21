"""Beckman Coulter list-mode (.lmd) files open like FCS files.

An .lmd file holds an FCS 2.0 data set (10-bit integers, already
compensated by the instrument) followed by an FCS 3.0 data set at full
resolution (raw values, with the instrument's matrix when there is one),
chained by $NEXTDATA. OpenFlo reads the FCS 3.0 one because it holds the
full-resolution values, and falls back to the first, with a note, when the
second cannot be read; that copy is never compensated again. The same bytes
named .fcs still load their first data set, as before. Every file here is
generated (tests/_fcs_writer.write_lmd), with generic channels FL1-A to FL3-A.
"""
from __future__ import annotations

import ast
import os
import re
import warnings

import numpy as np
import pytest

from openflo.exceptions import CompensationError
from openflo.pipeline import FcsParseError, FlowSample
from tests._fcs_writer import write_fcs, write_lmd

NAMES = ['FL1-A', 'FL2-A', 'FL3-A']
SPILL = '2,FL1-A,FL2-A,1,0.2,0,1'
NOT_COMP = 'its values are not compensated'


def _sets(n_first=150, n_second=200, seed=0):
    rng = np.random.default_rng(seed)
    first = rng.integers(1, 1000, (n_first, 3))
    second = rng.uniform(1.0, 2.0e5, (n_second, 3)).astype(np.float32)
    return first, second


def _lmd(tmp_path, name='tube1.lmd', first_kw=None, second_kw=None, **kw):
    first, second = _sets(**kw)
    return (write_lmd(tmp_path / name, first, second, NAMES,
                      first_kw=first_kw, second_kw=second_kw),
            first, second)


def _cut_second(p):
    """Cut the FCS 3.0 data short, so only the FCS 2.0 set can be read."""
    raw = p.read_bytes()
    p.write_bytes(raw[:len(raw) - 400])


def _events(s):
    return s.data[NAMES].to_numpy()


# ── reading ─────────────────────────────────────────────────────────────────

def test_an_lmd_file_loads_its_full_resolution_fcs3_data_set(tmp_path):
    p, _first, second = _lmd(tmp_path)
    s = FlowSample(str(p))
    assert s.name == 'tube1'
    assert s.channel_names[:3] == NAMES
    assert len(s.data) == len(second) == 200
    np.testing.assert_array_equal(_events(s), second)
    assert s.metadata.get('nextdata', '0').strip('0') == ''  # the 2nd set
    assert not [w for w in s.load_warnings if 'data set' in w], s.load_warnings


def test_the_generated_lmd_is_really_two_data_sets(tmp_path):
    # The file must be the two-set layout an .lmd has, or the tests above
    # would pass on a plain file: FCS 2.0 integers first, FCS 3.0 floats next.
    import flowio
    p, first, second = _lmd(tmp_path)
    sets = flowio.read_multiple_data_sets(str(p))
    assert [d.version for d in sets] == ['2.0', '3.0']
    assert [d.text['datatype'] for d in sets] == ['I', 'F']
    assert [d.event_count for d in sets] == [len(first), len(second)]


def test_the_same_bytes_named_fcs_still_load_the_first_data_set(tmp_path):
    p, first, _second = _lmd(tmp_path)
    q = p.with_suffix('.fcs')
    os.replace(p, q)
    s = FlowSample(str(q))
    np.testing.assert_array_equal(_events(s), first)
    assert any('only the first was loaded' in w for w in s.load_warnings)
    assert not s.instrument_compensated
    assert not [w for w in s.load_warnings if NOT_COMP in w]


def test_an_unreadable_second_data_set_falls_back_to_the_first(tmp_path):
    p, first, _second = _lmd(tmp_path)
    _cut_second(p)                               # the FCS 3.0 data cut short
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), first)
    notes = [w for w in s.load_warnings if 'data set' in w]
    assert len(notes) == 1, s.load_warnings
    assert re.search(r'full-resolution \(FCS 3\.0\) data set could not be '
                     r'read, so its first \(FCS 2\.0\) data set was loaded',
                     notes[0]), notes
    assert 'already compensated by the instrument' in notes[0], notes
    assert s.instrument_compensated
    assert not [w for w in s.load_warnings if NOT_COMP in w]


def test_a_nextdata_past_the_end_falls_back_to_the_first(tmp_path):
    p, first, _second = _lmd(tmp_path)
    raw = p.read_bytes()
    m = re.search(rb'\$NEXTDATA/(\d{10})/', raw)
    bad = str(len(raw) + 1000).rjust(10, '0').encode()
    p.write_bytes(raw[:m.start(1)] + bad + raw[m.end(1):])
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), first)
    assert any('FCS 2.0' in w for w in s.load_warnings), s.load_warnings


def test_a_single_data_set_lmd_loads_as_it_is(tmp_path):
    first, _ = _sets()
    p = write_fcs(tmp_path / 'one.lmd', first, NAMES)
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), first)
    assert not [w for w in s.load_warnings if 'data set' in w]
    # Which copy this is cannot be told, so nothing is said about it.
    assert not [w for w in s.load_warnings if NOT_COMP in w]
    assert not s.instrument_compensated


def test_an_lmd_with_both_data_sets_broken_still_fails(tmp_path):
    p, _first, _second = _lmd(tmp_path)
    raw = p.read_bytes()
    p.write_bytes(raw[:300])
    with pytest.raises(FcsParseError):
        FlowSample(str(p))


def test_the_lmd_second_read_keeps_reading_past_its_own_quirks(tmp_path):
    # A HEADER/TEXT disagreement inside the FCS 3.0 set is read past (at the
    # TEXT position, with a note), not taken for a broken second set.
    p, _first, second = _lmd(tmp_path)
    raw = bytearray(p.read_bytes())
    start = raw.index(b'FCS3.0')
    fld = start + 10 + 8 * 2                     # its HEADER data start
    raw[fld:fld + 8] = str(int(raw[fld:fld + 8]) + 4).rjust(8).encode()
    p.write_bytes(bytes(raw))
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), second)
    assert any('HEADER and TEXT disagree' in w for w in s.load_warnings)
    assert not any('FCS 2.0' in w for w in s.load_warnings)


def test_an_off_by_one_fcs3_set_is_read_with_a_note_not_a_warning(tmp_path):
    # flowio warns as it reads past the quirk; the user gets the note
    # instead, as for a plain .fcs.
    p, _first, second = _lmd(tmp_path)
    raw = p.read_bytes()
    start = raw.index(b'FCS3.0')
    fld = start + 10 + 8 * 3                     # its HEADER data end
    end = int(raw[fld:fld + 8])
    old = f'$ENDDATA/{end}/'.encode()
    new = f'$ENDDATA/{end + 1}/'.encode()
    assert len(old) == len(new) and raw.count(old) == 1, (old, new)
    raw = raw.replace(old, new)
    raw = raw[:fld] + str(end + 1).rjust(8).encode() + raw[fld + 8:]
    p.write_bytes(raw)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), second)
    assert any('one byte off' in w for w in s.load_warnings), s.load_warnings
    assert not any('FCS 2.0' in w for w in s.load_warnings)
    assert not [w for w in seen if 'incorrect data offset' in str(w.message)]


def test_an_upper_case_lmd_name_loads_the_fcs3_data_set(tmp_path):
    # Instruments write 'TUBE1.LMD'.
    p, _first, second = _lmd(tmp_path, 'TUBE1.LMD')
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), second)
    assert not [w for w in s.load_warnings if 'data set' in w]


def test_a_missing_range_in_the_fcs3_set_is_filled_not_fallen_back(tmp_path):
    # As for a plain .fcs: the range is taken from $PnB and the FCS 3.0
    # data still loads, not the FCS 2.0 copy.
    p, _first, second = _lmd(tmp_path, second_kw={'drop': ('$P1R',)})
    s = FlowSample(str(p))
    np.testing.assert_array_equal(_events(s), second)
    assert any('$P1R (range of FL1-A) is missing' in w
               for w in s.load_warnings), s.load_warnings
    assert not any('FCS 2.0' in w for w in s.load_warnings), s.load_warnings
    assert not s.instrument_compensated


def test_other_readers_of_the_file_get_the_same_data_set(tmp_path):
    # The keyword and channel-name readers go through the same reader.
    from openflo import run_history, run_template
    p, _first, second = _lmd(tmp_path)
    assert run_template.file_channels(str(p))[:3] == NAMES
    assert run_history.read_fcs_keywords(str(p))['TOT'] == str(len(second))


# ── compensation ────────────────────────────────────────────────────────────

def test_an_fcs3_copy_with_no_matrix_says_it_is_not_compensated(tmp_path):
    p, _first, _second = _lmd(tmp_path)
    s = FlowSample(str(p))
    said = [w for w in s.load_warnings if NOT_COMP in w]
    assert len(said) == 1, s.load_warnings
    assert said[0].startswith('tube1.lmd: ')
    assert '$SPILL or $SPILLOVER' in said[0]
    assert not s.instrument_compensated


def test_an_fcs3_copy_with_a_matrix_is_compensated_with_it(tmp_path):
    p, _first, second = _lmd(tmp_path,
                             second_kw={'extra': {'$SPILLOVER': SPILL}})
    s = FlowSample(str(p))
    assert not [w for w in s.load_warnings if NOT_COMP in w], s.load_warnings
    s.auto_compensate()
    assert s.comp_channels == ['FL1-A', 'FL2-A']
    want = second[:, 1] - 0.2 * second[:, 0]
    np.testing.assert_allclose(s.data['FL2-A'].to_numpy(), want, rtol=1e-4,
                               atol=1e-2)


def test_the_fcs2_fallback_is_never_compensated_again(tmp_path):
    # The instrument compensated the FCS 2.0 copy: neither its own matrix,
    # nor one the user or a workspace applies, may un-mix it again.
    from openflo.workspace_run import _recompensate_live
    p, first, _second = _lmd(tmp_path,
                             first_kw={'extra': {'$SPILLOVER': SPILL}})
    _cut_second(p)
    s = FlowSample(str(p))
    assert s.instrument_compensated
    s.auto_compensate()
    assert not s.is_compensated()
    np.testing.assert_array_equal(_events(s), first)
    m, ch = np.array([[1.0, 0.1], [0.0, 1.0]]), ['FL1-A', 'FL2-A']
    with pytest.raises(CompensationError, match='already compensated'):
        s.manual_compensate(m, ch)
    with pytest.raises(CompensationError, match='already compensated'):
        s.recompensate(m, ch)
    with pytest.raises(RuntimeError, match='already compensated'):
        _recompensate_live(s, {'matrix': m, 'channels': ch}, 'tube1')
    np.testing.assert_array_equal(_events(s), first)
    assert not s.is_compensated()


def test_the_editor_comp_apply_refuses_the_fcs2_fallback(tmp_path):
    # The compensation editor's Apply goes through recompensate.
    from openflo.editor_tools import ToolsMixin
    p, first, _second = _lmd(tmp_path)
    _cut_second(p)
    s = FlowSample(str(p))

    class _Ed:
        _samples = {'tube1': s}
    ok, note = ToolsMixin._comp_apply_sample(
        _Ed(), 'tube1', ['FL1-A', 'FL2-A'],
        np.array([[1.0, 0.1], [0.0, 1.0]]), report=False)
    assert not ok and 'already compensated' in note, note
    np.testing.assert_array_equal(_events(s), first)


def test_the_matrix_in_an_lmd_files_fcs3_copy_can_be_loaded(tmp_path):
    from openflo.compensation_io import read_compensation_matrix
    p, _first, _second = _lmd(tmp_path, 'CTRL.LMD',
                              second_kw={'extra': {'$SPILLOVER': SPILL}})
    ch, m = read_compensation_matrix(str(p))
    assert ch == ['FL1-A', 'FL2-A']
    np.testing.assert_allclose(m, [[1.0, 0.2], [0.0, 1.0]])


def _control(tmp_path, name):
    """A single-stain FL1-A control as an .lmd: 10 % of FL1-A in FL2-A in
    both copies (the FCS 2.0 copy scaled into its 10-bit range)."""
    rng = np.random.default_rng(3)
    n = 2000
    fl1 = np.r_[rng.normal(100, 10, n // 2), rng.normal(50000, 2000, n // 2)]
    second = np.column_stack([fl1, 100 + 0.1 * fl1,
                              rng.normal(100, 10, n)]).astype(np.float32)
    first = np.clip(second / 60.0, 0, 1023).astype(int)
    return write_lmd(tmp_path / name, first, second, NAMES)


def test_the_comp_optimizer_reads_an_lmd_control(tmp_path):
    from openflo.compensation_optimizer import optimize_compensation
    p = _control(tmp_path, 'fl1.lmd')
    _ch, m = optimize_compensation(['FL1-A', 'FL2-A'], {'FL1-A': str(p)})
    assert m[0, 1] == pytest.approx(0.1, abs=0.01)


def test_the_comp_optimizer_skips_an_instrument_compensated_control(
        tmp_path, caplog):
    from openflo.compensation_optimizer import optimize_compensation
    p = _control(tmp_path, 'fl1.lmd')
    _cut_second(p)
    with caplog.at_level('WARNING'):
        _ch, m = optimize_compensation(['FL1-A', 'FL2-A'], {'FL1-A': str(p)})
    np.testing.assert_array_equal(m, np.eye(2))
    assert 'instrument already compensated' in caplog.text


# ── other readers ───────────────────────────────────────────────────────────

def test_the_wsp_export_reads_time_from_the_fcs3_copy(tmp_path):
    from openflo.wsp_writer import WspWriter
    names = ['FL1-A', 'Time']
    rng = np.random.default_rng(1)
    first = rng.integers(1, 1000, (50, 2))
    second = np.column_stack([rng.uniform(1, 1e5, 80),
                              np.arange(80) * 2.0]).astype(np.float32)
    p = write_lmd(tmp_path / 'tube1.lmd', first, second, names)
    ticks = WspWriter._time_ticks({'fcs_path': str(p)})
    np.testing.assert_array_equal(ticks, second[:, 1])


def test_inspect_fcs_reads_an_lmd_file(tmp_path, capsys):
    from openflo import inspect_fcs
    p, _first, _second = _lmd(tmp_path)
    assert inspect_fcs.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert 'Data set read: FCS 3.0 copy' in out
    assert "'FL1-A'" in out


def test_a_paired_test_names_an_lmd_sample_by_its_donor():
    from openflo.group_test_choice import pair_key
    assert pair_key('D1_Stim.lmd', 'Stim') == pair_key('D1_Ctrl', 'Ctrl')
    assert pair_key('D1_Stim.LMD', 'Stim') == 'd1'


# ── accepted where .fcs is ──────────────────────────────────────────────────

def test_the_extension_helpers_take_lmd_in_any_case():
    from openflo.paths import FCS_FILE_PATTERNS, fcs_stem, is_fcs_file
    assert is_fcs_file('a.lmd') and is_fcs_file('A.LMD')
    assert is_fcs_file('a.fcs') and not is_fcs_file('a.csv')
    assert '*.lmd' in FCS_FILE_PATTERNS.split()
    assert fcs_stem('/x/tube1.LMD') == 'tube1'
    assert fcs_stem('/x/notes.txt') == 'notes.txt'


def test_a_dropped_folder_surfaces_its_lmd_files(tmp_path):
    from openflo.paths import expand_dropped_paths
    d = tmp_path / 'run'
    (d / 'sub').mkdir(parents=True)
    _lmd(d / 'sub', 'b.lmd')
    (d / 'notes.txt').write_text('x')
    fcs, wsp = expand_dropped_paths([str(d)])
    assert [os.path.basename(f) for f in fcs] == ['b.lmd'] and wsp == []


def test_cli_folder_scans_find_lmd_files(tmp_path):
    from openflo.cli_groups import auto_groups_by_day, discover_day_folders
    from openflo.cli_samples import find_unstained, scan_for_token
    from openflo.flow_experiment import FlowExperiment
    d = tmp_path / 'Day 1'
    d.mkdir()
    _lmd(d, 'expt_m1.lmd')
    _lmd(d, 'expt_unstained.lmd')
    assert discover_day_folders([str(tmp_path)]) == [str(d)]
    assert auto_groups_by_day([str(tmp_path)])
    assert scan_for_token(str(d), 'expt_m1', recurse=False).endswith(
        'expt_m1.lmd')
    assert scan_for_token(str(d), 'm1', recurse=False).endswith('m1.lmd')
    assert find_unstained(str(d), 'early').endswith('expt_unstained.lmd')
    late = tmp_path / 'late'
    late.mkdir()
    _lmd(late, 'late_unstained.lmd')          # named by its tail only
    assert find_unstained(str(late), 'early').endswith('late_unstained.lmd')
    assert len(FlowExperiment._resolve(str(d))) == 2


def test_cli_unmix_takes_lmd_files_from_a_folder(tmp_path):
    from types import SimpleNamespace

    from openflo.cli_unmix import _resolve_inputs
    _lmd(tmp_path, 'a.lmd')
    args = SimpleNamespace(unmix_input=str(tmp_path), fcs=None, trials=None)
    got = _resolve_inputs(args, {}, None)
    assert [os.path.basename(p) for p in got] == ['a.lmd']


def test_a_condition_table_names_an_lmd_sample_by_its_file_name():
    from openflo.sample_meta import read_condition_table
    out = read_condition_table('sample,condition\ntube1.lmd,ctrl\n',
                               ['tube1'])
    assert out['groups'] == {'tube1': 'ctrl'}


def test_a_run_template_matches_an_lmd_file_by_its_stem():
    from openflo.run_template import _stem
    assert _stem('/x/tube1.lmd') == 'tube1'


def test_a_raw_lmd_sample_counts_as_raw_fcs_for_compensation(tmp_path):
    # A sample read from an .lmd is raw data, so a workspace matrix may be
    # put on it; one built from a data table with no record may not.
    from openflo.workspace_run import _recompensate_live
    p, _first, _second = _lmd(tmp_path)
    s = FlowSample(str(p))
    s.comp_channels = []
    comp = {'matrix': np.eye(2), 'channels': ['FL1-A', 'FL2-A']}
    try:
        _recompensate_live(s, comp, 'tube1')
    except RuntimeError as e:
        assert 'does not record the compensation' not in str(e), e


def _fcs_literals(tree):
    """(line, text) of each string in `tree`, docstrings left out, that is a
    bare '.fcs' extension or pattern, or ends in one ('_x.fcs' in an
    f-string): what any spelling of an .fcs-only name test is made of."""
    docs = set()
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)):
            docs.add(id(body[0].value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docs):
            v = node.value.strip()
            if re.search(r'(?:^|\*)\.fcs$', v, re.I) or (
                    len(v) < 40 and v.lower().endswith('.fcs')):
                yield node.lineno, v


def test_no_fcs_only_filter_is_left_where_files_are_opened():
    """Every place that picks event-data files by name takes .lmd too: no
    code (docstrings aside) names the '.fcs' extension by itself, however
    the test is spelled, except in the modules listed."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'
    allowed = {
        'paths.py',                # defines FCS_EXTENSIONS itself
        'editor_populations.py',   # Save as FCS writes .fcs
        'synthetic.py', 'ui_synth.py',  # the synthetic data it writes
        'batch_normalize.py', 'fcs_export.py',  # files OpenFlo writes
        'diva_reader.py',          # a FACSDiva export's own .fcs files
        'tutorial_sample.py',      # writes its own generated .fcs, opens none
        # Still .fcs only (CHANGELOG says so): voltage titration's folder,
        # the FMO folder and the preview script's FCS lookup.
        'ui_voltage.py', 'fmo_gater.py', 'preview.py',
        # Nearby send's 'fcs' kind is .fcs by its protocol (docs/nearby-send.md),
        # the same on every platform: a .lmd sent as it would be saved as .fcs.
        'ui_nearby.py',
    }
    bad = []
    for f in sorted(src.glob('*.py')):
        if f.name in allowed:
            continue
        tree = ast.parse(f.read_text(encoding='utf-8'))
        bad += [f'{f.name}:{i}: {v!r}' for i, v in _fcs_literals(tree)]
    assert not bad, bad


def test_the_source_scan_sees_other_spellings():
    # The scan above must see an .fcs-only test however it is written.
    for code in ("p.lower().endswith(('.fcs',))",
                 "p.lower()[-4:] in ('.fcs',)",
                 "b.endswith(f'_{t}.fcs')",
                 "glob.glob(os.path.join(d, '*.FCS'))"):
        assert list(_fcs_literals(ast.parse(code))), code
    assert not list(_fcs_literals(ast.parse(
        'def f():\n    """Open an .fcs"""\n    return "No .fcs files here."')))


class _BatchView:
    """Just what PipelineWorkspaceView._batch_folder uses before it runs."""

    def __init__(self):
        import types
        self.model = types.SimpleNamespace(run_cfg={'method': 'flowsom'})
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


def test_the_batch_folder_takes_lmd_files_in_any_case(tmp_path, monkeypatch):
    d = tmp_path / 'in'
    d.mkdir()
    _lmd(d, 'a.lmd')
    _lmd(d, 'B.LMD')
    (d / 'notes.txt').write_text('x')
    said = _batch_folder(tmp_path, monkeypatch, d)
    assert len(said) == 1 and ' on 2 file(s)?' in said[0], said


def test_an_empty_batch_folder_names_both_extensions(tmp_path, monkeypatch):
    d = tmp_path / 'in'
    d.mkdir()
    said = _batch_folder(tmp_path, monkeypatch, d)
    assert said == ['No .fcs or .lmd files in that folder.'], said


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
    yield ed, queued
    root.destroy()


def test_the_open_dialog_offers_and_queues_lmd_files(editor, tmp_path,
                                                     monkeypatch):
    from openflo import editor_load
    ed, queued = editor
    p, _first, _second = _lmd(tmp_path)
    seen = {}

    def pick(**k):
        seen.update(k)
        return (str(p),)
    monkeypatch.setattr(editor_load.filedialog, 'askopenfilenames', pick)
    ed._add_samples()
    pats = dict(seen['filetypes'])
    assert '*.lmd' in pats['FCS files'].split()
    assert '*.lmd' in pats['FCS & workspaces'].split()
    assert queued == [('tube1', str(p))]
    assert 'Skipped' not in ed.status_var.get()


def test_a_watched_folder_loads_new_lmd_files(editor, tmp_path, monkeypatch):
    from openflo import editor_load
    ed, queued = editor
    d = tmp_path / 'incoming'
    d.mkdir()
    monkeypatch.setattr(editor_load.filedialog, 'askdirectory',
                        lambda **k: str(d))
    monkeypatch.setattr(ed, 'after', lambda *a, **k: 'after#1')
    ed._toggle_watch_folder()
    assert '.lmd' in ed.status_var.get()
    p, _first, _second = _lmd(d)
    ed._poll_watch()
    assert queued == [('tube1', str(p))]


def test_lmd_paths_pasted_from_the_clipboard_are_taken(editor, tmp_path,
                                                       monkeypatch):
    ed, _queued = editor
    a, _f, _s = _lmd(tmp_path, 'a.lmd')
    b, _f, _s = _lmd(tmp_path, 'b.lmd')
    monkeypatch.setattr(ed, 'clipboard_get', lambda: f'{a} {b}')
    assert ed._paste_fcs_from_clipboard() == [str(a), str(b)]


@pytest.fixture
def tk_root(monkeypatch, tmp_path):
    from tests.conftest import editor_or_skip
    from tests.test_transform_records import isolate_home
    isolate_home(monkeypatch, tmp_path)
    root, _ed = editor_or_skip()
    yield root
    root.destroy()


def test_the_comp_editor_loads_a_matrix_from_an_lmd_file(tk_root, tmp_path,
                                                         monkeypatch):
    from openflo import ui_comp
    p, _first, _second = _lmd(tmp_path, second_kw={
        'extra': {'$SPILLOVER': SPILL}})
    seen = {}

    def pick(**k):
        seen.update(k)
        return str(p)
    monkeypatch.setattr(ui_comp.filedialog, 'askopenfilename', pick)
    win = ui_comp.CompensationEditorWindow(tk_root, sample=None)
    win._load()
    pats = dict(seen['filetypes'])
    assert '*.lmd' in pats['Compensation'].split()
    assert '*.lmd' in pats['FCS (embedded $SPILL)'].split()
    assert win.channels == ['FL1-A', 'FL2-A']
    np.testing.assert_allclose(win._read_matrix_from_entries(),
                               [[1.0, 0.2], [0.0, 1.0]])


def test_the_comp_optimizer_picks_lmd_controls(tk_root, tmp_path,
                                               monkeypatch):
    from openflo import ui_comp
    d = tmp_path / 'controls'
    d.mkdir()
    _lmd(d, 'FL1.lmd')
    _lmd(d, 'FL2.LMD')
    dlg = ui_comp.OptimizeCompensationDialog(tk_root, ['FL1-A', 'FL2-A'],
                                             on_complete=lambda c, m: None)
    monkeypatch.setattr(ui_comp.filedialog, 'askdirectory',
                        lambda **k: str(d))
    dlg._autofill_from_dir()
    got = {c: os.path.basename(v.get()) for c, v in dlg.path_vars.items()}
    assert got == {'FL1-A': 'FL1.lmd', 'FL2-A': 'FL2.LMD'}, got
    seen = {}

    def pick(**k):
        seen.update(k)
        return str(d / 'FL1.lmd')
    monkeypatch.setattr(ui_comp.filedialog, 'askopenfilename', pick)
    dlg._browse('FL2-A')
    assert '*.lmd' in dict(seen['filetypes'])['FCS'].split()
    assert dlg.path_vars['FL2-A'].get() == str(d / 'FL1.lmd')
