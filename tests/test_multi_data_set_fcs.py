"""An FCS file holding several data sets ($NEXTDATA) opens as one sample per
data set, and each sample stays on its own data set wherever it goes.

Guava / easyCyte plate runs, Kaluza and concatenated
files put several data sets in one FCS file, each chained to the next by
$NEXTDATA. Before, only the first loaded (with a note). Now the editor adds
each as its own sample, named from $FIL, $SMNO or the tube / sample name when
every data set has a different one, else '<file> #n', and a session, a
transfer, the .wsp export and the Compensation editor read the same data set
again. A chain that loops or points past the end of the file stops there, with
a note. A Beckman Coulter .lmd file (two copies of the same events) is not
split. Every file here is generated (tests/_fcs_writer.write_multi_fcs), with
generic channels FL1-A to FL3-A.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from openflo.fcs_data_sets import (
    MissingDataSetError,
    data_set_names,
    data_set_offset,
    list_data_sets,
    sample_data_sets,
)
from openflo.pipeline import FcsParseError, FlowSample
from tests._fcs_writer import set_nextdata, write_lmd, write_multi_fcs

NAMES = ['FL1-A', 'FL2-A', 'FL3-A']
FIRST_ONLY = 'only the first was loaded'
WELLS = [{'$FIL': f'A0{i}.fcs'} for i in (1, 2, 3)]


def _sets(counts=(50, 70, 90), seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(1.0, 1000.0, (n, 3)).astype(np.float32)
            for n in counts]


def _multi(tmp_path, counts=(50, 70, 90), name='plate.fcs', kws=None):
    sets = _sets(counts)
    p = write_multi_fcs(tmp_path / name, sets, NAMES, kws=kws)
    return str(p), sets


def _values(s):
    """The detector values a sample was read with (before any transform)."""
    return s.raw[NAMES].to_numpy()


def _join(path, parts):
    """Write the data sets `parts` (``(data, names, write_fcs options)``
    each) one after the other to `path`, chained by $NEXTDATA counted from
    each data set's start. Unlike write_multi_fcs, each may have its own
    channels and FCS version, so their TEXT and DATA differ in size."""
    from tests._fcs_writer import write_fcs
    blobs = []
    for i, (data, names, opts) in enumerate(parts):
        part = path.with_name(f'{path.name}.part{i}')
        opts = dict(opts)
        opts['extra'] = {**opts.get('extra', {}), '$NEXTDATA': '0' * 10}
        write_fcs(part, data, names, **opts)
        raw = part.read_bytes()
        part.unlink()
        nxt = len(raw) if i < len(parts) - 1 else 0
        blobs.append(raw.replace(
            b'$NEXTDATA/' + b'0' * 10 + b'/',
            b'$NEXTDATA/' + str(nxt).rjust(10, '0').encode() + b'/'))
    path.write_bytes(b''.join(blobs))
    return str(path)


def _mixed(tmp_path):
    """A file of two data sets that differ in their channels (the second
    adds FL4-A) and so in the size of their TEXT and DATA."""
    rng = np.random.default_rng(2)
    sets = [rng.uniform(1, 1e4, (40, 3)).astype(np.float32),
            rng.uniform(1, 1e4, (60, 4)).astype(np.float32)]
    p = _join(tmp_path / 'mixed.fcs', [
        (sets[0], NAMES, {'datatype': 'F', 'extra': {'$FIL': 'one.fcs'}}),
        (sets[1], NAMES + ['FL4-A'],
         {'datatype': 'F', 'extra': {'$FIL': 'two.fcs',
                                     '$COM': 'a longer TEXT ' * 20}})])
    return p, sets


# ── reading ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('counts', [(50, 70), (50, 70, 90)])
def test_each_data_set_reads_as_its_own_sample(tmp_path, counts):
    import flowio
    p, sets = _multi(tmp_path, counts)
    found, note = list_data_sets(p)
    assert note == ''
    assert [d.number for d in found] == list(range(1, len(counts) + 1))
    # The same data sets flowio's own multi-set reader finds.
    assert [d.event_count for d in flowio.read_multiple_data_sets(p)] == \
        list(counts)
    for k, want in enumerate(sets, 1):
        s = FlowSample(p, set_number=k)
        assert s.set_number == k
        np.testing.assert_array_equal(_values(s), want)
        assert not [w for w in s.load_warnings if 'data set' in w]


def test_without_a_data_set_the_first_loads_with_a_note_as_before(tmp_path):
    p, sets = _multi(tmp_path)
    s = FlowSample(p)
    assert s.set_number is None
    np.testing.assert_array_equal(_values(s), sets[0])
    assert any(FIRST_ONLY in w for w in s.load_warnings), s.load_warnings


def test_a_data_set_the_file_no_longer_holds_says_so(tmp_path):
    p, _sets3 = _multi(tmp_path, (50, 70))
    with pytest.raises(MissingDataSetError, match='holds 2 data sets'):
        data_set_offset(p, 3)
    with pytest.raises(FcsParseError,
                       match=r'could not read plate\.fcs: its data set 3 was '
                             'not found'):
        FlowSample(p, set_number=3)


def test_a_data_set_without_its_ranges_is_read_with_them_filled(tmp_path):
    sets = _sets((50, 70))
    p = str(write_multi_fcs(tmp_path / 'norange.fcs', sets, NAMES,
                            drop=('$P1R',)))
    s = FlowSample(p, set_number=2)
    np.testing.assert_array_equal(_values(s), sets[1])


def test_a_single_data_set_file_is_one_sample(tmp_path):
    p, _sets1 = _multi(tmp_path, (40,))
    assert sample_data_sets(p) == ([], '')
    assert len(list_data_sets(p)[0]) == 1


# ── a broken $NEXTDATA chain ────────────────────────────────────────────────

def test_a_chain_that_loops_back_stops_with_a_note(tmp_path):
    p, sets = _multi(tmp_path)
    offsets = [d.offset for d in list_data_sets(p)[0]]
    set_nextdata(p, 3, -offsets[2])            # data set 3 -> data set 1
    found, note = list_data_sets(p)
    assert len(found) == 3
    assert 'of data set 3 points back to an earlier data set' in note, note
    assert 'only the first 3 data sets were opened' in note, note
    np.testing.assert_array_equal(_values(FlowSample(p, set_number=3)),
                                  sets[2])


def test_a_chain_pointing_back_to_the_start_stops(tmp_path):
    p, _sets2 = _multi(tmp_path, (50, 70))
    offsets = [d.offset for d in list_data_sets(p)[0]]
    set_nextdata(p, 2, -offsets[1])            # back to byte 0
    found, note = list_data_sets(p)
    assert len(found) == 2 and 'points back' in note


def test_a_chain_pointing_past_the_end_stops_with_a_note(tmp_path):
    p, sets = _multi(tmp_path)
    set_nextdata(p, 2, 10 ** 9)
    found, note = list_data_sets(p)
    assert len(found) == 2
    assert 'of data set 2 points past the end of the file' in note, note
    assert 'only the first 2 data sets were opened' in note, note
    assert sample_data_sets(p) == (found, note)
    np.testing.assert_array_equal(_values(FlowSample(p, set_number=2)),
                                  sets[1])


def test_a_data_set_whose_header_cannot_be_read_stops_the_chain(tmp_path):
    p, _sets3 = _multi(tmp_path)
    second = list_data_sets(p)[0][1].offset
    raw = bytearray(open(p, 'rb').read())
    raw[second + 10:second + 18] = b'    xxxx'      # TEXT start not a number
    open(p, 'wb').write(bytes(raw))
    found, note = list_data_sets(p)
    assert len(found) == 1
    assert 'data set 2 could not be read' in note, note


def test_a_chain_longer_than_the_cap_stops(tmp_path, monkeypatch):
    import openflo.fcs_data_sets as fds
    p, _sets3 = _multi(tmp_path)
    monkeypatch.setattr(fds, 'MAX_DATA_SETS', 2)
    found, note = list_data_sets(p)
    assert len(found) == 2 and 'more than 2 data sets' in note, note


def test_a_chain_pointing_into_the_middle_of_a_data_set_stops(tmp_path):
    p, _sets3 = _multi(tmp_path)
    set_nextdata(p, 1, 100)
    found, note = list_data_sets(p)
    assert len(found) == 1 and 'does not point to a data set' in note


def test_an_offset_counted_from_the_file_start_is_followed(tmp_path):
    # Some writers count $NEXTDATA from the start of the file. For the third
    # data set that differs from the FCS 2.0 rule, and is still found.
    p, sets = _multi(tmp_path)
    offsets = [d.offset for d in list_data_sets(p)[0]]
    set_nextdata(p, 2, offsets[2])
    found, note = list_data_sets(p)
    assert note == '' and [d.offset for d in found] == offsets
    np.testing.assert_array_equal(_values(FlowSample(p, set_number=3)),
                                  sets[2])


# ── names ───────────────────────────────────────────────────────────────────

def test_data_sets_are_named_from_fil_when_each_has_its_own(tmp_path):
    p, _sets3 = _multi(tmp_path, kws=WELLS)
    assert data_set_names(sample_data_sets(p)[0], p) == ['A01', 'A02', 'A03']


def test_names_fall_back_to_the_next_keyword_then_the_file_name(tmp_path):
    same_fil = [{'$FIL': 'plate.fcs', '$SMNO': f'S{i}'} for i in (1, 2, 3)]
    p, _s = _multi(tmp_path, kws=same_fil)
    assert data_set_names(list_data_sets(p)[0], p) == ['S1', 'S2', 'S3']
    clash = [{'$SMNO': 'S1'}, {'$SMNO': 'S1'}, {'$SMNO': 'S3'}]
    q, _s = _multi(tmp_path, name='run.fcs', kws=clash)
    assert data_set_names(list_data_sets(q)[0], q) == [
        'run #1', 'run #2', 'run #3']
    r, _s = _multi(tmp_path, name='bare.fcs')
    assert data_set_names(list_data_sets(r)[0], r) == [
        'bare #1', 'bare #2', 'bare #3']


# ── what is not split ───────────────────────────────────────────────────────

def test_an_lmd_file_is_not_split_and_reads_as_before(tmp_path):
    rng = np.random.default_rng(1)
    first = rng.integers(1, 1000, (150, 3))
    second = rng.uniform(1.0, 2.0e5, (120, 3)).astype(np.float32)
    p = str(write_lmd(tmp_path / 'tube.lmd', first, second, NAMES))
    assert len(list_data_sets(p)[0]) == 2
    assert sample_data_sets(p) == ([], '')
    for number in (None, 1, 2):         # a data set number changes nothing
        s = FlowSample(p, set_number=number)
        np.testing.assert_array_equal(_values(s), second)
        assert s.lmd_data_set == 'fcs3'


def test_lmd_copies_named_fcs_stay_one_sample(tmp_path):
    rng = np.random.default_rng(1)
    first = rng.integers(1, 1000, (120, 3))
    second = rng.uniform(1.0, 2.0e5, (120, 3)).astype(np.float32)
    p = str(write_lmd(tmp_path / 'tube.fcs', first, second, NAMES))
    assert len(list_data_sets(p)[0]) == 2
    assert sample_data_sets(p) == ([], '')


# ── sessions (headless) ─────────────────────────────────────────────────────

def _session(tmp_path, entry):
    sess = tmp_path / 's.flowsession'
    sess.write_text(json.dumps({'format': 'openflo-session', 'version': 3,
                                'samples': [entry],
                                'sample_gates': {}}), 'utf-8')
    return str(sess)


def test_a_session_entry_reopens_its_data_set_from_the_raw_file(tmp_path):
    from openflo.document import Document
    p, sets = _multi(tmp_path, kws=WELLS)
    doc = Document()
    doc.open_session(_session(tmp_path, {'name': 'A02', 'path': p,
                                         'set_number': 2}))
    s = doc._samples['A02'].data
    assert s.set_number == 2
    np.testing.assert_array_equal(_values(s), sets[1])
    # A save writes it back.
    out = tmp_path / 'b.flowsession'
    doc.save_session(str(out))
    entry = json.loads(out.read_text('utf-8'))['samples'][0]
    assert entry['set_number'] == 2


def test_a_session_entry_without_a_data_set_reads_the_first(tmp_path):
    from openflo.document import Document
    p, sets = _multi(tmp_path)
    doc = Document()
    doc.open_session(_session(tmp_path, {'name': 'old', 'path': p}))
    s = doc._samples['old'].data
    assert s.set_number is None
    np.testing.assert_array_equal(_values(s), sets[0])


def test_a_sidecar_sample_reads_its_data_sets_keywords(tmp_path):
    from openflo.document import Document
    from openflo.session_sidecar import write_frame
    p, _sets3 = _multi(tmp_path, kws=WELLS)
    s3 = FlowSample(p, set_number=3)
    side = tmp_path / 's_data' / 'A03.npz'
    side.parent.mkdir()
    assert write_frame(s3.data, str(side))
    doc = Document()
    doc.open_session(_session(tmp_path, {
        'name': 'A03', 'path': p, 'set_number': 3,
        'processed_data': 's_data/A03.npz'}))
    rec = doc._samples['A03']
    assert rec.sidecar
    assert rec.data.set_number == 3
    assert str(rec.data.metadata['tot']).strip() == '90'
    assert rec.data.metadata['fil'] == 'A03.fcs'
    kw = {r[0]: r[1] for r in doc.keywords('A03')['keywords']}
    assert kw.get('$FIL') == 'A03.fcs', kw


def test_plan_keys_a_raw_data_set_apart_from_its_file(tmp_path):
    from openflo.session_open import plan_sample, raw_loads, saved_set_number
    p, _s = _multi(tmp_path)
    plan = plan_sample({'name': 'x', 'path': p, 'set_number': 2}, {},
                       str(tmp_path))
    assert plan.key.endswith('#2') and plan.prep['set_number'] == 2
    assert raw_loads([plan])[0] == [(p, 2)]
    for bad in (0, -1, True, '2', 1.5):
        assert saved_set_number({'set_number': bad}) is None
    # A sidecar is keyed by its own path alone.
    (tmp_path / 'x.npz').write_bytes(b'')
    side = plan_sample({'name': 'x', 'path': p, 'set_number': 2,
                        'processed_data': 'x.npz'}, {}, str(tmp_path))
    assert side.processed_path and not side.key.endswith('#2')
    assert side.prep['set_number'] == 2


def test_a_document_wsp_export_warns_about_the_data_set(tmp_path):
    from openflo.document import Document
    p, _s = _multi(tmp_path, kws=WELLS)
    doc = Document()
    doc.open_session(_session(tmp_path, {'name': 'A02', 'path': p,
                                         'set_number': 2}))
    res = doc.export_wsp(str(tmp_path / 'out.wsp'))
    assert any('A02 is data set 2 of plate.fcs' in w
               for w in res['warnings']), res['warnings']


def test_flowjo_values_are_read_from_the_data_set(tmp_path):
    from openflo.flowjo_events import fcs_values_as_flowjo_gates_them
    p, sets = _multi(tmp_path)
    got = fcs_values_as_flowjo_gates_them(p, set_number=2)
    np.testing.assert_array_equal(got[NAMES].to_numpy(), sets[1])


def test_a_workspace_reload_reads_the_items_data_set(tmp_path):
    from types import SimpleNamespace

    from openflo.workspace_run import resolve_run_sample
    p, sets = _multi(tmp_path)
    item = {'sample': 'A02', 'path': p, 'set_number': 2,
            'qc_applied': False, 'gate_id': None}
    s, _note = resolve_run_sample(SimpleNamespace(_samples={}), item)
    assert len(s.data) == len(sets[1])


# ── exports that name the file ──────────────────────────────────────────────

def test_the_wsp_export_uses_the_data_sets_keywords_and_warns(tmp_path):
    from openflo.wsp_writer import WspWriter
    p, _s = _multi(tmp_path, kws=WELLS)
    w = WspWriter(cytometer='T')
    w.add_sample('A02', p, NAMES, [], set_number=2)
    w.add_sample('plate', p, NAMES, [])
    w.write(str(tmp_path / 'out.wsp'))
    two, one = w.samples
    assert two['events'] == 70 and one['events'] == 50
    assert dict(two['keywords'])['$FIL'] == 'A02.fcs'
    assert dict(one['keywords'])['$FIL'] == 'A01.fcs'
    said = [x for x in w.warnings if 'data set 2' in x]
    assert len(said) == 1 and 'FlowJo may open' in said[0], w.warnings


def test_the_samples_own_spill_is_read_from_its_data_set(tmp_path, caplog):
    from openflo.compensation_io import read_compensation_matrix
    kws = [{'$SPILLOVER': f'2,FL1-A,FL2-A,1,0.{i},0,1'} for i in (1, 2)]
    p, _s = _multi(tmp_path, (60, 60), kws=kws)
    _ch, m2 = read_compensation_matrix(p, 2)
    assert m2[0, 1] == pytest.approx(0.2)
    caplog.set_level('WARNING')
    _ch, m1 = read_compensation_matrix(p)
    assert m1[0, 1] == pytest.approx(0.1)
    assert FIRST_ONLY in caplog.text


def test_an_lmd_control_reports_as_before(tmp_path, caplog):
    from openflo.compensation_optimizer import optimize_compensation
    from tests.test_lmd_files import _cut_second, _lmd
    p, _first, _second = _lmd(tmp_path)
    _cut_second(p)
    caplog.set_level('WARNING')
    optimize_compensation(['FL1-A', 'FL2-A'], {'FL1-A': str(p)})
    assert 'only the FCS 2.0 copy' in caplog.text
    assert 'control tube1.lmd:' not in caplog.text


def test_a_comp_control_holding_several_data_sets_says_the_first_is_used(
        tmp_path, caplog):
    from openflo.compensation_optimizer import optimize_compensation
    p, _s = _multi(tmp_path, (300, 300))
    caplog.set_level('WARNING')
    optimize_compensation(['FL1-A', 'FL2-A'], {'FL1-A': p})
    assert f'control plate.fcs: it holds more than one data set; {FIRST_ONLY}' \
        in caplog.text


def test_a_packed_archive_keeps_every_data_set(tmp_path):
    from openflo.fcs_pack import pack_fcs, unpack_fcs
    p, sets = _multi(tmp_path)
    buf = io.BytesIO()
    pack_fcs(p, buf)
    buf.seek(0)
    out = tmp_path / 'back.fcs'
    with open(out, 'wb') as fh:
        unpack_fcs(buf, fh)
    np.testing.assert_array_equal(
        _values(FlowSample(str(out), set_number=3)), sets[2])


def test_a_workspace_item_keeps_its_data_set(tmp_path):
    # An item is made as the workspace makes it (drag, Add, the menu):
    # build_drop_payload(extract_one_context()). Once the sample is closed
    # in the editor, a run and a bead matrix reload its own data set.
    from types import SimpleNamespace

    from openflo.workspace_run import (
        build_drop_payload,
        extract_one_context,
        member_channels,
        resolve_run_sample,
    )
    p, sets = _mixed(tmp_path)
    ed = SimpleNamespace(_samples={'B': FlowSample(p, set_number=2)})
    item = build_drop_payload(extract_one_context(ed, 'B'), None)
    assert item['set_number'] == 2
    closed = SimpleNamespace(_samples={})
    s, _note = resolve_run_sample(closed, dict(item, qc_applied=False))
    np.testing.assert_array_equal(s.raw[NAMES + ['FL4-A']].to_numpy(),
                                  sets[1])
    assert 'FL4-A' in member_channels(closed, dict(item, comp_channels=[]))
    first = dict(item, comp_channels=[], set_number=None)
    assert 'FL4-A' not in member_channels(closed, first)


# ── the desktop editor ──────────────────────────────────────────────────────

@pytest.fixture
def desk(monkeypatch, tmp_path):
    from tests.conftest import editor_or_skip
    from tests.test_transform_records import capture_warnings, isolate_home
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    root, ed = editor_or_skip()
    yield root, ed
    root.destroy()


def _queue(ed, items):
    jobs = []
    ed._enqueue_load = lambda job, priority=1: jobs.append(job)
    ed._queue_fcs_loads(items)
    return jobs


def _run(root, ed, jobs):
    """Run the queued raw-FCS jobs on this thread and land them."""
    from tests._tk_wait import pump_until
    for name, path, *rest in jobs:
        if len(rest) > 1:
            ed._load_worker(name, path, rest[1])
        else:
            ed._load_worker(name, path)
    assert pump_until(root, lambda: all(j[0] in ed._samples for j in jobs)), \
        ed.status_var.get()


def test_opening_a_file_adds_a_sample_per_data_set(desk, tmp_path):
    root, ed = desk
    p, sets = _multi(tmp_path, kws=WELLS)
    jobs = _queue(ed, [p])
    assert jobs == [('A01', p, 'fcs', 1), ('A02', p, 'fcs', 2),
                    ('A03', p, 'fcs', 3)]
    assert 'plate.fcs holds 3 data sets' in ed.status_var.get()
    _run(root, ed, jobs)
    for k, name in enumerate(('A01', 'A02', 'A03')):
        np.testing.assert_array_equal(_values(ed._samples[name]), sets[k])
        assert ed._samples[name].set_number == k + 1
    # Each data set is registered apart; the file itself is not a sample.
    from openflo.paths import sample_key
    assert ed._path_to_name.get(sample_key(p, 2)) == 'A02'
    assert sample_key(p) not in ed._path_to_name
    assert ed._loaded_name_for_path(p, 2) == 'A02'
    assert ed._loaded_name_for_path(p) is None
    # Opening it again adds nothing.
    assert _queue(ed, [p]) == []


def test_a_name_in_use_is_made_unique(desk, tmp_path):
    _root, ed = desk
    p, _sets3 = _multi(tmp_path, kws=WELLS)
    ed._name_to_path['A02'] = 'elsewhere'
    names = [j[0] for j in _queue(ed, [p])]
    assert names[0] == 'A01' and names[2] == 'A03'
    assert names[1].startswith('A02 [') and names[1] not in ('A01', 'A03')
    assert len(set(names)) == 3


def test_a_broken_chain_is_told_when_loading(desk, tmp_path):
    _root, ed = desk
    p, _s = _multi(tmp_path)
    set_nextdata(p, 2, 10 ** 9)
    assert [j[3] for j in _queue(ed, [p])] == [1, 2]
    assert any('points past the end' in n for n in ed._load_notes)


def test_a_workspace_staged_file_and_an_lmd_load_as_one_sample(desk,
                                                               tmp_path):
    _root, ed = desk
    p, _s = _multi(tmp_path)
    ed._sample_name_for(p)                # what a .wsp import stages
    assert _queue(ed, [p]) == [('plate', p)]
    q, _s = _multi(tmp_path, name='moved.fcs')
    from openflo.paths import sample_key
    ed._pending_sample_meta[sample_key(q)] = {'gates': []}  # a transfer's
    assert _queue(ed, [q]) == [('moved', q)]
    rng = np.random.default_rng(1)
    q = str(write_lmd(tmp_path / 'tube.lmd', rng.integers(1, 1000, (90, 3)),
                      rng.uniform(1, 2e5, (90, 3)).astype(np.float32),
                      NAMES))
    assert _queue(ed, [q]) == [('tube', q)]


def test_a_saved_session_reopens_each_sample_on_its_data_set(desk, tmp_path):
    from tests.test_transform_records import _new_editor
    root, ed = desk
    p, sets = _multi(tmp_path, kws=WELLS)
    _run(root, ed, _queue(ed, [p]))
    ed._sample_trial['A02'] = 'Run 7'
    path = tmp_path / 'desk.flowsession'
    data = ed._write_session(str(path))
    assert {e['name']: e.get('set_number') for e in data['samples']} == {
        'A01': 1, 'A02': 2, 'A03': 3}
    ed2 = _new_editor(root)
    jobs = []
    ed2._enqueue_load = lambda job, priority=1: jobs.append(job)
    ed2._load_session_path(str(path))
    assert sorted(j[3] for j in jobs) == [1, 2, 3]
    _run(root, ed2, jobs)
    for k, name in enumerate(('A01', 'A02', 'A03')):
        np.testing.assert_array_equal(_values(ed2._samples[name]), sets[k])
        assert ed2._samples[name].set_number == k + 1
    # What the session kept about the sample came back with it.
    assert ed2._sample_trial.get('A02') == 'Run 7'


def test_a_data_set_with_saved_numbers_is_hashed_on_load(desk, tmp_path):
    from openflo.paths import sample_key
    from tests._tk_wait import pump_until
    root, ed = desk
    p, _sets3 = _multi(tmp_path)
    ed._pending_sample_meta[sample_key(p, 2)] = {'gates': [],
                                                 'saved_stats': {'v': 1}}
    ed._load_worker('A02', p, 2)
    assert pump_until(root, lambda: 'A02' in ed._samples)
    assert getattr(ed._samples['A02'], 'loaded_digest', None) is not None


def test_a_session_of_one_data_set_reopens_only_that_one(desk, tmp_path):
    from tests.test_transform_records import _new_editor
    root, ed = desk
    p, _sets3 = _multi(tmp_path)
    jobs = _queue(ed, [(p, 2)])
    assert jobs == [('plate #2', p, 'fcs', 2)]
    _run(root, ed, jobs)
    path = tmp_path / 'one.flowsession'
    ed._write_session(str(path))
    ed2 = _new_editor(root)
    jobs2 = []
    ed2._enqueue_load = lambda job, priority=1: jobs2.append(job)
    ed2._load_session_path(str(path))
    assert jobs2 == [('plate #2', p, 'fcs', 2)]


def test_staged_records_are_kept_per_data_set(desk, tmp_path):
    _root, ed = desk
    p, _s = _multi(tmp_path)
    ed.stage_sample_prep(p, qc=True, set_number=2)
    ed.stage_sample_prep(p, set_number=3)
    assert ed._take_pending_prep(p) is None
    assert ed._take_pending_prep(p, 2)['qc'] is True
    assert ed._take_pending_prep(p, 3)['set_number'] == 3


def test_the_load_pool_reads_a_data_set_job(desk, tmp_path):
    from tests._tk_wait import pump_until
    root, ed = desk
    p, sets = _multi(tmp_path)
    ed._load_queue.put((1, 1, ('A02', p, 'fcs', 2)))
    ed._load_queue.put((2, 2, None))
    from openflo.editor_loadpool import _JOB
    try:
        ed._load_pool_worker()          # on this thread, as a pool thread
    finally:
        # The worker marks its thread's job; this thread is the Tk one.
        _JOB.__dict__.pop('epoch', None)
    assert pump_until(root, lambda: 'A02' in ed._samples)
    np.testing.assert_array_equal(_values(ed._samples['A02']), sets[1])


def test_an_unreadable_sidecar_rebuilds_from_its_data_set(desk, tmp_path):
    from tests._tk_wait import pump_until
    root, ed = desk
    p, sets = _multi(tmp_path)
    side = tmp_path / 'A02.npz'
    side.write_bytes(b'not a sidecar')
    from openflo.paths import sample_key
    ed.stage_sample_prep(str(side), source_fcs=p, qc=False, set_number=2)
    ed._pending_sample_meta[sample_key(str(side))] = {'gates': [],
                                                      'trial': 'Run 9'}
    ed._load_csv_worker('A02', str(side), {})
    assert pump_until(root, lambda: 'A02' in ed._samples)
    np.testing.assert_array_equal(_values(ed._samples['A02']), sets[1])
    assert ed._sample_trial['A02'] == 'Run 9'


def test_reloads_from_the_raw_file_keep_the_data_set(desk, tmp_path):
    from openflo.sample_prep import sidecar_sample
    _root, ed = desk
    p, sets = _multi(tmp_path)
    two = FlowSample(p, set_number=2)
    s, _note = sidecar_sample(two.data.copy(), str(tmp_path / 'A02.npz'),
                              {'source_fcs': p, 'set_number': 2}, name='A02')
    ed._samples['A02'] = s
    ok, why = ed._comp_apply_by_reload('A02', s, ['FL1-A', 'FL2-A'],
                                       np.eye(2))
    assert ok, why
    assert len(s.data) == len(sets[1])
    full = ed._prepared_fcs_sample(p, {'qc': False, 'set_number': 2})
    frame, why = ed._ds_rebuild_full(full, {'from': len(sets[1]),
                                            'steps': [], 'seed': 0})
    assert frame is not None, why


def test_a_transfer_carries_the_data_set(desk, tmp_path):
    root, ed = desk
    p, _sets3 = _multi(tmp_path, kws=WELLS)
    _run(root, ed, _queue(ed, [p]))
    bundle = ed._sample_transfer_bundle(['A02'])
    (entry,) = bundle['samples']
    assert entry['set_number'] == 2
    ed2_root = root
    from tests.test_transform_records import _new_editor
    ed2 = _new_editor(ed2_root)
    jobs = []
    ed2._enqueue_load = lambda job, priority=1: jobs.append(job)
    ed2._read_transfer_bundle = lambda: bundle
    ed2._paste_samples_transfer()
    assert jobs == [('plate #2', p, 'fcs', 2)]
    # Pasted where it is open already, it is not loaded again.
    again = []
    ed._enqueue_load = lambda job, priority=1: again.append(job)
    ed._read_transfer_bundle = lambda: bundle
    ed._paste_samples_transfer()
    assert again == [] and 'Already open here' in ed.status_var.get()


# ── the chain, at its edges ─────────────────────────────────────────────────

def test_data_sets_with_their_own_channels_and_text_sizes(tmp_path):
    from openflo.fcs_text import fcs_text_pairs
    p, sets = _mixed(tmp_path)
    found, note = list_data_sets(p)
    assert note == '' and len(found) == 2
    # The TEXT of the second, read from its own HEADER (another layout).
    two = {k.upper(): v for k, v in fcs_text_pairs(p, found[1].offset)}
    assert two == found[1].keywords
    assert two['$PAR'] == '4' and two['$TOT'] == '60'
    assert two['$COM'].strip().endswith('a longer TEXT')
    s2 = FlowSample(p, set_number=2)
    np.testing.assert_array_equal(s2.raw[NAMES + ['FL4-A']].to_numpy(),
                                  sets[1])
    assert 'FL4-A' not in FlowSample(p, set_number=1).data.columns


def test_an_offset_naming_its_own_data_set_stops(tmp_path):
    # Counted from the file start, data set 2's $NEXTDATA names data set 2
    # itself: followed, the chain would never end.
    p, sets = _multi(tmp_path)
    offsets = [d.offset for d in list_data_sets(p)[0]]
    set_nextdata(p, 2, offsets[1])
    found, note = list_data_sets(p)
    assert [d.offset for d in found] == offsets[:2]
    assert 'of data set 2 names its own data set' in note, note
    np.testing.assert_array_equal(_values(FlowSample(p, set_number=2)),
                                  sets[1])


def test_an_offset_inside_the_file_from_its_start_is_not_past_the_end(
        tmp_path):
    # Counted from data set 2 it is past the end; from the file start it is
    # inside the file, at no data set.
    import os
    p, _s = _multi(tmp_path)
    set_nextdata(p, 2, os.path.getsize(p) - 200)
    found, note = list_data_sets(p)
    assert len(found) == 2
    assert 'does not point to a data set' in note, note


def test_a_header_cut_off_by_the_end_of_the_file_is_not_read(tmp_path):
    # The last bytes hold 'FCS', offsets and a TEXT, but not a whole
    # HEADER: not a data set.
    import os
    p, _s = _multi(tmp_path, (50, 70))
    o2 = list_data_sets(p)[0][1].offset
    end = os.path.getsize(p)
    set_nextdata(p, 2, end - o2)
    with open(p, 'ab') as fh:
        fh.write(b'FCS3.0    ' + b'26'.rjust(8) + b'37'.rjust(8)
                 + b'/$TOT/1/X/Y/')
    found, note = list_data_sets(p)
    assert len(found) == 2 and note, note


def test_stray_fcs_bytes_do_not_take_the_place_of_a_data_set(tmp_path):
    # The offset is counted from the file start; counted from data set 2 it
    # lands on bytes 'FCS' inside DATA, which are no HEADER.
    p, _sets = _multi(tmp_path)
    offsets = [d.offset for d in list_data_sets(p)[0]]
    set_nextdata(p, 2, offsets[2])
    raw = bytearray(open(p, 'rb').read())
    stray = offsets[1] + offsets[2]
    assert stray + 58 <= len(raw)
    raw[stray:stray + 3] = b'FCS'
    open(p, 'wb').write(bytes(raw))
    found, note = list_data_sets(p)
    assert note == '' and [d.offset for d in found] == offsets


def test_a_next_data_offset_that_is_no_number_ends_the_chain(tmp_path):
    p, _s = _multi(tmp_path)
    raw = open(p, 'rb').read()
    first = raw.index(b'$NEXTDATA/') + len(b'$NEXTDATA/')
    open(p, 'wb').write(raw[:first] + b'ninetynine' + raw[first + 10:])
    found, note = list_data_sets(p)
    assert len(found) == 1 and note == ''
    assert sample_data_sets(p) == ([], '')


def test_data_set_numbers_count_from_one(tmp_path):
    p, _s = _multi(tmp_path)
    assert data_set_offset(p, 1) == 0
    for bad in (0, -1, 4):
        with pytest.raises(MissingDataSetError):
            data_set_offset(p, bad)


def test_text_pairs_keep_an_escaped_delimiter_and_a_last_value():
    from openflo.fcs_data_sets import parse_text
    assert parse_text(b'/$FIL/a//b.fcs/$TOT/12/') == [
        ('$FIL', 'a/b.fcs'), ('$TOT', '12')]
    # A TEXT that does not end with its delimiter keeps its last value.
    assert parse_text(b'/$FIL/a.fcs/$TOT/12') == [
        ('$FIL', 'a.fcs'), ('$TOT', '12')]


# ── what is split: copies of the same events or not ─────────────────────────

def _pair(tmp_path, first, second, names2=NAMES, v2='3.0', name='p.fcs'):
    return _join(tmp_path / name, [
        (first, NAMES, {'version': '2.0'}),
        (second, names2, {'datatype': 'F', 'version': v2})])


def test_an_fcs2_and_fcs3_pair_of_other_samples_is_split(tmp_path):
    rng = np.random.default_rng(5)
    ints = rng.integers(1, 1000, (120, 3))
    floats = rng.uniform(1, 1e4, (120, 3)).astype(np.float32)
    # The same channels and events: copies (an .lmd file named .fcs).
    assert sample_data_sets(_pair(tmp_path, ints, floats)) == ([], '')
    # Another event count: another sample.
    other_tot = _pair(tmp_path, ints, floats[:90], name='tot.fcs')
    assert len(sample_data_sets(other_tot)[0]) == 2
    # Other channels: another sample.
    four = rng.uniform(1, 1e4, (120, 4)).astype(np.float32)
    other_par = _pair(tmp_path, ints, four, NAMES + ['FL4-A'],
                      name='par.fcs')
    assert len(sample_data_sets(other_par)[0]) == 2


def test_two_fcs3_data_sets_of_one_size_are_split(tmp_path):
    p, _s = _multi(tmp_path, (60, 60))
    assert len(sample_data_sets(p)[0]) == 2


# ── names, at their edges ───────────────────────────────────────────────────

def _named(*values, key='$SMNO'):
    from openflo.fcs_data_sets import DataSet
    return [DataSet(n, 0, '3.1', {} if v is None else {key: v})
            for n, v in enumerate(values, 1)]


def test_a_keyword_only_some_data_sets_have_does_not_name_them():
    assert data_set_names(_named('S1', None, 'S3'), '/d/run.fcs') == [
        'run #1', 'run #2', 'run #3']


def test_names_drop_control_characters_and_are_capped():
    assert data_set_names(_named('S\x001', 'S\t\n2'), 'r.fcs') == [
        'S 1', 'S 2']
    long = data_set_names(_named('a' * 100 + '1', 'b' * 100), 'r.fcs')
    assert long == ['a' * 80, 'b' * 80]


# ── matching a workspace sample to its data set ─────────────────────────────

def test_a_workspace_sample_is_matched_by_its_count_and_name():
    from openflo.fcs_data_sets import DataSet, match_data_set
    sets = [DataSet(n, 0, '3.1', {'$TOT': tot, '$FIL': fil})
            for n, (tot, fil) in enumerate(
                [('50', 'A01.fcs'), ('70', 'A02.fcs'), ('70', 'A03.fcs')], 1)]
    assert match_data_set(sets, {'$TOT': '70', '$FIL': 'A03.fcs'}) == 3
    assert match_data_set(sets, {'$tot': ' 50 '}) == 1
    assert match_data_set(sets, {'$FIL': 'A02.fcs'}) == 2
    assert match_data_set(sets, {'$TOT': '70'}) is None      # two agree
    assert match_data_set(sets, {'$TOT': '50', '$FIL': 'A02.fcs'}) is None
    assert match_data_set(sets, {'$CYT': 'x'}) is None       # nothing to go by


def _two_set_wsp(tmp_path, p):
    """A workspace of data sets 2 and 3 of `p`, each with its own gate and
    its own matrix."""
    from openflo.wsp_writer import WspWriter
    w = WspWriter(cytometer='T')
    for n, cut in ((2, 300.0), (3, 700.0)):
        w.add_sample(f'A0{n}', p, NAMES, [{
            'id': 'g1', 'parent_id': None, 'kind': 'threshold',
            'channel': 'FL1-A', 'value': cut, 'name': f'cut {n}'}],
            compensation=(['FL1-A', 'FL2-A'],
                          np.array([[1.0, 0.1 * n], [0.0, 1.0]])),
            set_number=n)
    out = tmp_path / 'out.wsp'
    w.write(str(out))
    return str(out)


def test_a_workspace_of_data_sets_reopens_each_on_its_own(tmp_path):
    from openflo.wsp_open import plan_wsp
    p, _s = _multi(tmp_path, kws=WELLS)
    plan = plan_wsp(_two_set_wsp(tmp_path, p))
    assert [(r.node_name, r.set_number, r.set_name)
            for r in plan.samples] == [('A02', 2, 'A02'), ('A03', 3, 'A03')]
    assert plan.mismatched == [] and plan.notes == []


def test_unmatched_workspace_samples_do_not_overwrite_each_other(tmp_path):
    from openflo.wsp_open import plan_wsp
    p, _s = _multi(tmp_path, kws=WELLS)
    wsp = _two_set_wsp(tmp_path, p)
    # The file is replaced by another run: no data set matches any more.
    write_multi_fcs(p, _sets((55, 75, 95)), NAMES)
    plan = plan_wsp(wsp)
    assert [(r.node_name, r.set_number) for r in plan.samples] == [
        ('A02', None)]
    said = ' | '.join(plan.notes)
    assert "'A02': plate.fcs holds 3 data sets and none" in said, said
    assert "'A03' was not imported" in said, said


def test_a_document_imports_each_data_set_of_a_workspace(tmp_path):
    from openflo.document import Document
    p, sets = _multi(tmp_path, kws=WELLS)
    doc = Document()
    res = doc.import_wsp(_two_set_wsp(tmp_path, p))
    assert res['samples'] == ['A02', 'A03']
    for k, name in ((1, 'A02'), (2, 'A03')):
        rec = doc._samples[name]
        np.testing.assert_array_equal(_values(rec.data), sets[k])
        assert [g['label'] for g in rec.gates.values()] == [f'cut {k + 1}']


# ── what reads the data set when exporting or reloading ─────────────────────

def test_the_wsp_writer_reads_time_from_the_data_set(tmp_path):
    from openflo.wsp_writer import WspWriter
    rng = np.random.default_rng(6)
    names = ['FL1-A', 'Time']
    sets = [np.column_stack([rng.uniform(1, 1e3, n),
                             np.arange(n, dtype=float)]).astype(np.float32)
            for n in (50, 70)]
    p = str(write_multi_fcs(tmp_path / 't.fcs', sets, names))
    ticks = WspWriter._time_ticks({'fcs_path': p, 'set_number': 2})
    assert ticks is not None and len(ticks) == 70


def test_unknown_scales_are_matched_on_the_data_set(tmp_path, monkeypatch):
    import openflo.flowjo_events as fe
    from openflo.wsp_export import match_unknown_scales
    p, _sets = _multi(tmp_path)
    s = FlowSample(p, set_number=2)
    seen = []
    real = fe.fcs_values_as_flowjo_gates_them

    def spy(*a, **k):
        seen.append(k.get('set_number'))
        return real(*a, **k)

    monkeypatch.setattr(fe, 'fcs_values_as_flowjo_gates_them', spy)
    gate = {'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.0}
    match_unknown_scales(s, {'FL1-A': {'method': 'unknown'}}, [gate],
                         None, None)
    assert seen == [2]


def test_the_compensation_editor_shows_the_data_sets_own_spill(tmp_path):
    from types import SimpleNamespace

    from openflo.ui_comp import CompensationEditorWindow
    kws = [{'$SPILLOVER': f'2,FL1-A,FL2-A,1,0.{i},0,1'} for i in (1, 2)]
    p, _s = _multi(tmp_path, (60, 60), kws=kws)
    got = []
    me = SimpleNamespace(
        sample=SimpleNamespace(path=p, set_number=2, comp_matrix=None,
                               comp_channels=None),
        _set_matrix=lambda ch, m, source_label='': got.append(m))
    CompensationEditorWindow._auto_import(me)
    assert got and got[0][0, 1] == pytest.approx(0.2)


def test_the_editors_wsp_export_and_import_keep_each_data_set(desk,
                                                              tmp_path):
    import tkinter.filedialog as fd

    from openflo.wsp_open import plan_wsp
    root, ed = desk
    p, _sets = _multi(tmp_path, kws=WELLS)
    _run(root, ed, _queue(ed, [p]))
    out = tmp_path / 'ed.wsp'
    saved = fd.asksaveasfilename
    fd.asksaveasfilename = lambda *a, **k: str(out)
    try:
        ed.run_async = (lambda work, on_done=None, on_error=None,
                        busy_msg=None: on_done(work()))
        ed._export_flowjo_wsp()
    finally:
        fd.asksaveasfilename = saved
    assert [(r.node_name, r.set_number) for r in plan_wsp(str(out)).samples
            ] == [('A01', 1), ('A02', 2), ('A03', 3)]


def test_the_editor_imports_a_workspace_of_data_sets(desk, tmp_path):
    root, ed = desk
    p, sets = _multi(tmp_path, kws=WELLS)
    wsp = _two_set_wsp(tmp_path, p)
    jobs = []
    ed._enqueue_load = lambda job, priority=1: jobs.append(job)
    ed._ingest_wsp(wsp)
    assert jobs == [('A02', p, 'fcs', 2), ('A03', p, 'fcs', 3)]
    _run(root, ed, jobs)
    for k, name in ((1, 'A02'), (2, 'A03')):
        np.testing.assert_array_equal(_values(ed._samples[name]), sets[k])
        assert [g['label'] for g in ed._sample_gates[name].values()] == [
            f'cut {k + 1}']
        s = ed._samples[name]
        assert s.compensation_source.startswith('wsp:')
        assert s.comp_matrix[0, 1] == pytest.approx(0.1 * (k + 1))


# ── Time on a data set's own timestep ───────────────────────────────────────

def _timed(tmp_path, kws):
    """Two data sets with a Time channel (ticks 0, 1, ...; 50 and 100
    events), each with the keywords `kws`."""
    rng = np.random.default_rng(7)
    names = ['FL1-A', 'Time']
    sets = [np.column_stack([rng.uniform(1, 1e3, n),
                             np.arange(n, dtype=float)]).astype(np.float32)
            for n in (50, 100)]
    return str(write_multi_fcs(tmp_path / 't.fcs', sets, names, kws=kws))


def test_the_timestep_is_the_data_sets_own(tmp_path):
    from xml.etree import ElementTree as ET

    from openflo.fcs_text import fcs_timestep
    from openflo.wsp_reader import WspReader
    p = _timed(tmp_path, [{'$TIMESTEP': '0.01'}, {'$TIMESTEP': '0.1'}])
    assert fcs_timestep(p) == pytest.approx(0.01)
    assert fcs_timestep(p, set_number=2) == pytest.approx(0.1)
    elem = ET.fromstring('<Sample/>')
    assert WspReader.sample_timestep(elem, p, 2) == pytest.approx(0.1)
    # Without $TIMESTEP: the run's length over its own last Time tick.
    clock = {'$BTIM': '10:00:00', '$ETIM': '10:00:10'}
    q = str(write_multi_fcs(tmp_path / 'clock.fcs', [
        np.column_stack([np.ones(n), np.arange(n, dtype=float)]).astype(
            np.float32) for n in (51, 101)], ['FL1-A', 'Time'],
        kws=[clock, clock]))
    assert fcs_timestep(q, set_number=2) == pytest.approx(0.1)
    assert fcs_timestep(q) == pytest.approx(0.2)


def test_a_time_gate_comes_back_on_its_data_sets_timestep(tmp_path):
    from openflo.wsp_open import plan_wsp
    from openflo.wsp_writer import WspWriter
    clock = {'$BTIM': '10:00:00', '$ETIM': '10:00:10'}
    p = str(write_multi_fcs(tmp_path / 'clock.fcs', [
        np.column_stack([np.ones(n), np.arange(n, dtype=float)]).astype(
            np.float32) for n in (51, 101)], ['FL1-A', 'Time'],
        kws=[dict(clock, **{'$FIL': 'a.fcs'}),
             dict(clock, **{'$FIL': 'b.fcs'})]))
    w = WspWriter(cytometer='T')
    w.add_sample('b', p, ['FL1-A', 'Time'], [{
        'id': 'g1', 'parent_id': None, 'kind': 'threshold',
        'channel': 'Time', 'value': 30.0, 'name': 'late'}], set_number=2)
    assert w.samples[0]['timestep'] == pytest.approx(0.1)
    out = str(tmp_path / 't.wsp')
    w.write(out)
    (rec,) = plan_wsp(out).samples
    assert rec.set_number == 2
    assert rec.gates[0]['value'] == pytest.approx(30.0)


def test_a_name_keyword_the_same_in_every_data_set_is_passed_over():
    from openflo.fcs_data_sets import DataSet, data_set_matches, match_data_set
    sets = [DataSet(n, 0, '3.1', {'$TOT': '70', '$FIL': 'plate.fcs',
                                  '$SMNO': f'S{n}'}) for n in (1, 2, 3)]
    # $FIL names the whole file in each: $SMNO tells them apart.
    assert match_data_set(sets, {'$TOT': '70', '$FIL': 'plate.fcs',
                                 '$SMNO': 'S2'}) == 2
    assert data_set_matches(sets, {'$TOT': '70', '$FIL': 'plate.fcs'}) == [
        1, 2, 3]
    assert data_set_matches(sets, {'$TOT': '71'}) == []


def test_the_unmatched_note_says_when_several_data_sets_match(tmp_path):
    from openflo.wsp_open import plan_wsp
    p, _s = _multi(tmp_path, counts=(70, 70, 90))
    plan = plan_wsp(_two_set_wsp(tmp_path, p))
    said = ' | '.join(plan.notes)
    assert "'A02': plate.fcs holds 3 data sets and several have" in said, said
    assert [(r.node_name, r.set_number) for r in plan.samples][0] == (
        'A02', None)
