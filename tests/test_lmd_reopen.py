"""A Beckman Coulter .lmd sample keeps what its copy says about compensation
after a session is saved and reopened, and wherever else it travels.

An .lmd file holds an FCS 2.0 data set (already compensated by the
instrument) and an FCS 3.0 copy (raw values). When the FCS 3.0 copy cannot
be read, the FCS 2.0 one is loaded and no matrix may be applied on top of
it. A session saves which copy was loaded (``lmd_data_set``) and the
instrument's compensation (``instrument_compensated``); a sample reopened
from its sidecar reads that copy's keywords and still refuses a matrix. A
session saved before those records existed reads the .lmd file again to find
the copy. The batch run, the FlowJo comparison and Concatenate treat such a
sample the same way. Every file here is generated (tests/_fcs_writer), with
generic channels FL1-A to FL3-A.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import openflo.pipeline as fp
from openflo.document import Document
from openflo.exceptions import CompensationError
from openflo.session_open import plan_sample
from openflo.session_sidecar import write_frame
from tests.test_lmd_files import NAMES, _cut_second, _lmd

MATRIX = np.array([[1.0, 0.1], [0.0, 1.0]])
PAIR = ['FL1-A', 'FL2-A']
NEW_KEYS = ('lmd_data_set', 'instrument_compensated')


def _fallback(tmp_path, name='tube1.lmd'):
    """An .lmd file whose FCS 3.0 data is cut short (150 events in its
    FCS 2.0 set, 200 in its FCS 3.0 set), and its first set."""
    p, first, _second = _lmd(tmp_path, name)
    _cut_second(p)
    return p, first


def _saved(tmp_path, p, *, drop=()):
    """``(session path, loaded sample)``: a session holding the sample
    loaded from `p`, reopening from a sidecar of its data; the keys in
    `drop` are taken out of its entry (a session an older OpenFlo saved)."""
    doc = Document()
    name = doc.open_fcs(str(p))
    loaded = doc._samples[name].data
    sess = tmp_path / 'a.flowsession'
    doc.save_session(str(sess))
    side = tmp_path / 'a_data' / f'{name}.npz'
    side.parent.mkdir()
    assert write_frame(loaded.data, str(side))
    d = json.loads(sess.read_text('utf-8'))
    entry = d['samples'][0]
    entry['processed_data'] = f'a_data/{name}.npz'
    for k in drop:
        entry.pop(k, None)
    sess.write_text(json.dumps(d), 'utf-8')
    return str(sess), loaded


def _reopened(sess):
    doc = Document()
    doc.open_session(sess)
    rec = next(iter(doc._samples.values()))
    assert rec.sidecar, "the sample must come back from its sidecar"
    return doc, rec.data


def _refuses_a_matrix(s):
    with pytest.raises(CompensationError, match='already compensated'):
        s.manual_compensate(MATRIX, PAIR)


# ── the session records ─────────────────────────────────────────────────────

def test_a_session_saves_which_lmd_copy_was_loaded(tmp_path):
    p, _first = _fallback(tmp_path)
    sess, _s = _saved(tmp_path, p)
    entry = json.loads(open(sess, encoding='utf-8').read())['samples'][0]
    assert entry['lmd_data_set'] == 'fcs2'
    assert entry['instrument_compensated'] is True
    ok = tmp_path / 'ok'
    ok.mkdir()
    q, _f, _s2 = _lmd(ok, 'tube2.lmd')
    sess2, _s3 = _saved(ok, q)
    entry2 = json.loads(open(sess2, encoding='utf-8').read())['samples'][0]
    assert entry2['lmd_data_set'] == 'fcs3'
    assert 'instrument_compensated' not in entry2


def test_a_reopened_fcs2_fallback_is_still_compensated_by_the_instrument(
        tmp_path):
    p, _first = _fallback(tmp_path)
    sess, loaded = _saved(tmp_path, p)
    doc, s = _reopened(sess)
    assert s.instrument_compensated
    assert s.lmd_data_set == 'fcs2'
    # The keywords and decoding are the FCS 2.0 copy's, as loaded: not the
    # FCS 3.0 copy's (datatype F, $TOT 200, linear).
    assert s.metadata['datatype'] == loaded.metadata['datatype'] == 'I'
    assert str(s.metadata['tot']).strip() == '150'
    assert s.fcs_decoding == loaded.fcs_decoding
    assert s.linear_ranges == loaded.linear_ranges
    assert any('FCS 2.0 data set' in w for w in s.load_warnings)
    before = s.data[NAMES].to_numpy().copy()
    _refuses_a_matrix(s)
    np.testing.assert_array_equal(s.data[NAMES].to_numpy(), before)
    # ... and a second save keeps the records.
    doc.save_session(str(tmp_path / 'b.flowsession'))
    entry = json.loads((tmp_path / 'b.flowsession').read_text('utf-8'))
    assert {k: entry['samples'][0].get(k) for k in NEW_KEYS} == {
        'lmd_data_set': 'fcs2', 'instrument_compensated': True}


def test_an_older_session_without_the_records_reads_the_lmd_again(tmp_path):
    p, _first = _fallback(tmp_path)
    sess, _loaded = _saved(tmp_path, p, drop=NEW_KEYS)
    _doc, s = _reopened(sess)
    assert s.instrument_compensated
    assert s.metadata['datatype'] == 'I'
    _refuses_a_matrix(s)


def test_an_older_session_of_a_readable_lmd_reads_its_fcs3_keywords(
        tmp_path):
    p, _first, _second = _lmd(tmp_path)
    sess, _loaded = _saved(tmp_path, p, drop=NEW_KEYS)
    _doc, s = _reopened(sess)
    assert not s.instrument_compensated
    assert s.lmd_data_set == 'fcs3'
    assert s.metadata['datatype'] == 'F'


def test_the_record_names_the_copy_whose_keywords_are_read(tmp_path):
    # Both copies readable: the record, not the file, says which copy the
    # saved values are.
    from openflo.sample_prep import sidecar_sample
    p, _first, _second = _lmd(tmp_path)
    df = fp.FlowSample(str(p)).data
    for which, datatype, tot, flagged in (('fcs2', 'I', '150', True),
                                          ('fcs3', 'F', '200', False)):
        s, _note = sidecar_sample(
            df, str(tmp_path / 'x.csv'),
            {'source_fcs': str(p), 'lmd_data_set': which}, name='tube1')
        assert s.metadata['datatype'] == datatype, which
        assert str(s.metadata['tot']).strip() == tot, which
        assert s.instrument_compensated is flagged, which


def test_a_fallback_whose_lmd_is_gone_still_refuses_a_matrix(tmp_path):
    p, _first = _fallback(tmp_path)
    sess, _loaded = _saved(tmp_path, p)
    p.unlink()
    doc, s = _reopened(sess)
    assert s.instrument_compensated
    _refuses_a_matrix(s)
    # ... and saving it again keeps both records.
    doc.save_session(str(tmp_path / 'b.flowsession'))
    entry = json.loads((tmp_path / 'b.flowsession').read_text('utf-8'))
    assert {k: entry['samples'][0].get(k) for k in NEW_KEYS} == {
        'lmd_data_set': 'fcs2', 'instrument_compensated': True}


def test_a_repaired_lmd_drops_the_saved_records_on_the_next_save(tmp_path):
    # Saved as the fallback, reopened from the whole file again (no
    # sidecar): the next save says what the sample is now, not then.
    p, _first = _fallback(tmp_path)
    (tmp_path / 'w').mkdir()
    whole, _f, _s = _lmd(tmp_path / 'w', 'tube1.lmd')
    doc = Document()
    doc.open_fcs(str(p))
    doc.save_session(str(tmp_path / 'a.flowsession'))
    p.write_bytes(whole.read_bytes())
    doc = Document()
    doc.open_session(str(tmp_path / 'a.flowsession'))
    s = next(iter(doc._samples.values())).data
    assert not s.instrument_compensated
    doc.save_session(str(tmp_path / 'b.flowsession'))
    entry = json.loads((tmp_path / 'b.flowsession').read_text('utf-8'))
    assert entry['samples'][0].get('lmd_data_set') == 'fcs3'
    assert 'instrument_compensated' not in entry['samples'][0]


def test_a_plain_fcs_source_keeps_the_saved_flag(tmp_path):
    # Only an .lmd file's copy decides the flag on reopen; any other raw
    # file leaves the session's record as it is.
    from openflo.sample_prep import sidecar_sample
    from tests._fcs_writer import write_fcs
    first = np.random.default_rng(0).integers(1, 1000, (50, 3))
    p = write_fcs(tmp_path / 'one.fcs', first, NAMES)
    df = fp.FlowSample(str(p)).data
    s, _note = sidecar_sample(
        df, str(tmp_path / 'x.csv'),
        {'source_fcs': str(p), 'instrument_compensated': True}, name='one')
    assert s.instrument_compensated
    assert s.lmd_data_set is None


def test_the_editor_stages_the_records_for_its_sidecar_load(tmp_path):
    # The editor's reopen stages plan.prep through stage_sample_prep.
    from openflo.editor_loadpool import LoadPoolMixin
    entry = {'name': 'tube1', 'path': str(tmp_path / 'tube1.lmd'),
             'processed_data': 'tube1.npz', 'lmd_data_set': 'fcs2',
             'instrument_compensated': True}
    (tmp_path / 'tube1.npz').write_bytes(b'')
    plan = plan_sample(entry, {}, str(tmp_path))
    assert plan.prep['lmd_data_set'] == 'fcs2'
    assert plan.prep['instrument_compensated'] is True

    class _Ed(LoadPoolMixin):
        _pending_sample_prep = None
    ed = _Ed()
    ed.stage_sample_prep(plan.load_path, **plan.prep)
    rec = ed._take_pending_prep(plan.load_path)
    assert rec['lmd_data_set'] == 'fcs2'
    assert rec['instrument_compensated'] is True
    # A concatenation's sidecar (no raw file) is staged for its flag alone.
    ed.stage_sample_prep(str(tmp_path / 'c.npz'), instrument_compensated=True)
    assert ed._take_pending_prep(str(tmp_path / 'c.npz'))[
        'instrument_compensated'] is True


def test_an_unknown_copy_record_is_ignored(tmp_path):
    entry = {'name': 'tube1', 'path': str(tmp_path / 'tube1.lmd'),
             'processed_data': 'tube1.npz', 'lmd_data_set': 'other'}
    (tmp_path / 'tube1.npz').write_bytes(b'')
    assert plan_sample(entry, {}, str(tmp_path)).prep['lmd_data_set'] is None


# ── other places a fallback sample goes ─────────────────────────────────────

def _wsp(tmp_path, fcs):
    w = fp.WspWriter(cytometer='T')
    w.add_sample('s0', str(fcs), NAMES, [], compensation=(PAIR, MATRIX))
    out = tmp_path / 'x.wsp'
    w.write(str(out))
    return str(out)


def test_a_batch_run_with_a_workspace_skips_a_fallback_sample(tmp_path):
    p, first = _fallback(tmp_path)
    exp = fp.FlowExperiment([str(p)])
    exp.run_all(compensate=True, wsp_path=_wsp(tmp_path, p),
                transform=False, cluster=False)
    s = exp.samples['tube1']
    assert not s.is_compensated()
    np.testing.assert_array_equal(s.data[NAMES].to_numpy(), first)
    assert any('workspace matrix was not applied' in w
               for w in s.load_warnings), s.load_warnings


def test_the_comparison_names_an_instrument_compensated_sample(tmp_path):
    from openflo.compare import INSTRUMENT_COMPENSATED, _compensate_for_compare
    p, _first = _fallback(tmp_path)
    s = fp.FlowSample(str(p))
    assert (_compensate_for_compare(s, _wsp(tmp_path, p))
            == INSTRUMENT_COMPENSATED)
    assert not s.is_compensated()


def _samples(tmp_path, flags):
    out = []
    for i, flag in enumerate(flags):
        d = tmp_path / f'd{i}'
        d.mkdir()
        p, _first = (_fallback(d, f's{i}.lmd') if flag
                     else _lmd(d, f's{i}.lmd')[:2])
        s = fp.FlowSample(str(p))
        s.data = s.data[NAMES].astype(float)
        out.append(s)
    return out


@pytest.mark.parametrize('flags, marked, noted', [
    ((True, True), True, False),
    ((True, False), True, True),
    ((False, False), False, False),
])
def test_concatenating_keeps_the_instruments_compensation(
        tmp_path, flags, marked, noted):
    from openflo.ui_concatenate import concatenate_samples
    out = concatenate_samples(_samples(tmp_path, flags), 'joined')
    assert out.instrument_compensated is marked
    assert out.lmd_data_set is None     # it is not one copy of one file
    assert any('no matrix can be applied' in w
               for w in out.load_warnings) is noted
    if marked:
        _refuses_a_matrix(out)


def test_a_resave_drops_a_copy_record_the_sample_no_longer_has(tmp_path):
    # An entry naming an .lmd copy for a sample that now loads from a plain
    # FCS file (its records were stale): the next save does not repeat it.
    from tests._fcs_writer import write_fcs
    first = np.random.default_rng(0).integers(1, 1000, (50, 3))
    p = write_fcs(tmp_path / 'one.fcs', first, NAMES)
    doc = Document()
    doc.open_fcs(str(p))
    sess = tmp_path / 'a.flowsession'
    doc.save_session(str(sess))
    d = json.loads(sess.read_text('utf-8'))
    d['samples'][0]['lmd_data_set'] = 'fcs3'
    sess.write_text(json.dumps(d), 'utf-8')
    doc = Document()
    doc.open_session(str(sess))
    doc.save_session(str(tmp_path / 'b.flowsession'))
    entry = json.loads((tmp_path / 'b.flowsession').read_text('utf-8'))
    assert 'lmd_data_set' not in entry['samples'][0]
