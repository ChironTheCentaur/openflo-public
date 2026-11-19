"""openflo.document.Document: what it keeps of a sample, a gate and a
session (forms, axes, derived channels, QC, links, labels, groups, the
active sample) and the gates it refuses.

Generated FCS files, workspaces and sessions only; channel names FSC-A,
SSC-A, FL1-A ... and sample names s1, s2, a, b. No Tk.
"""
from __future__ import annotations

import json
import os

import pytest

from openflo import document, gating
from openflo.document import Document, DocumentError
from tests._open_fixtures import make_fcs, write_wsp
from tests.test_document import gated_doc, spill_fcs
from tests.test_document_save_safety import sidecar_doc

T = 262144.0
# The axis FlowJo declares for Comp-FL1-A in the workspace below, as
# wsp_gate_coords.wsp_stat_axes reads it.
LOGICLE_XML = (
    '<Transformations><transforms:logicle transforms:length="256" '
    f'transforms:T="{T:g}" transforms:W="1.0" transforms:M="4.4185" '
    'transforms:A="0"><data-type:parameter data-type:name="Comp-FL1-A"/>'
    '</transforms:logicle></Transformations>')


def _resave(path, edit):
    data = json.loads(open(path, encoding='utf-8').read())
    edit(data)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)


def _entry(path, i=0):
    return json.loads(open(path, encoding='utf-8').read())['samples'][i]


def _two_sample_session(tmp_path, active):
    doc = Document()
    doc.open_fcs(spill_fcs(tmp_path / 'raw' / 's1.fcs', seed=1))
    doc.open_fcs(spill_fcs(tmp_path / 'raw' / 's2.fcs', seed=2))
    out = tmp_path / 'two.flowsession'
    doc.save_session(str(out))
    _resave(out, lambda d: d.update(active_sample=active))
    return out


# ── statistics: forms and workspace axes ────────────────────────────────────

def test_the_stat_forms_chosen_are_used(tmp_path):
    """`stat_forms` reaches population_stats: a log GeoMean and a
    percentile rCV differ from the defaults and match a direct call."""
    p = spill_fcs(tmp_path / 's1.fcs')
    forms = {'geomean': 'log', 'rcv': 'percentile'}
    plain, chosen = Document(), Document(stat_forms=forms)
    name = plain.open_fcs(p)
    chosen.open_fcs(p)
    want = ('GeoMean', 'rCV')
    a = plain.stats(name, stats=want)[0]['channels']['FL1-A']
    b = chosen.stats(name, stats=want)[0]['channels']['FL1-A']
    assert a['geomean'] != pytest.approx(b['geomean'], rel=1e-6)
    assert a['rcv'] != pytest.approx(b['rcv'], rel=1e-6)
    s = chosen._samples[name].data
    (ref,) = gating.population_stats(
        name, s.data, {}, [], {'FL1-A': 'FL1-A'}, ['FL1-A'], set(want),
        gating.CHANNEL_STAT_ORDER, transforms=s.data_transforms,
        stat_forms=forms, all_events=True)
    assert b == {'geomean': ref['GeoMean FL1-A'], 'rcv': ref['rCV FL1-A']}


def _axes_wsp(tmp_path):
    a = make_fcs(tmp_path / 'run' / 'a.fcs', seed=1)
    wsp = write_wsp(tmp_path / 'run' / 'x.wsp', [('a', a, 'Stim', True)])
    text = open(wsp, encoding='utf-8').read()
    assert text.count('<SampleNode') == 1
    text = text.replace('<SampleNode', LOGICLE_XML + '<SampleNode')
    with open(wsp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    return wsp


def test_a_workspaces_axes_are_kept_used_saved_and_restored(tmp_path):
    """A workspace declaring a FlowJo axis for FL1-A: the sample keeps it,
    its GeoMean is taken on it (not on the stored logicle), a save writes
    it and a reopen puts it back, with the same numbers."""
    from openflo.wsp_gate_coords import wsp_stat_axes
    from openflo.wsp_reader import WspReader
    wsp = _axes_wsp(tmp_path)
    axes = wsp_stat_axes(next(WspReader(wsp).root.iter('Sample')))
    assert set(axes) == {'FL1-A'}
    doc = Document()
    doc.import_wsp(wsp)
    s = doc._samples['a'].data
    assert s.stat_axes == axes
    rows = doc.stats('a', channels=['FL1-A'], stats=['GeoMean'])
    kw = {'transforms': s.data_transforms, 'all_events': True}
    ref = gating.population_stats(
        'a', s.data, doc._samples['a'].gates, doc._samples['a'].order,
        {'FL1-A': 'FL1-A'}, ['FL1-A'], {'GeoMean'},
        gating.CHANNEL_STAT_ORDER, stat_axes=axes, **kw)
    off = gating.population_stats(
        'a', s.data, doc._samples['a'].gates, doc._samples['a'].order,
        {'FL1-A': 'FL1-A'}, ['FL1-A'], {'GeoMean'},
        gating.CHANNEL_STAT_ORDER, **kw)
    got = [r['channels']['FL1-A']['geomean'] for r in rows]
    assert got == [r['GeoMean FL1-A'] for r in ref]
    assert got[0] != pytest.approx(off[0]['GeoMean FL1-A'], rel=1e-6)
    out = tmp_path / 'a.flowsession'
    doc.save_session(str(out))
    assert _entry(out)['stat_axes'] == axes
    doc2 = Document()
    doc2.open_session(str(out))
    assert doc2._samples['a'].data.stat_axes == axes
    assert doc2.stats('a', channels=['FL1-A'], stats=['GeoMean']) == rows


# ── what a session entry carries ────────────────────────────────────────────

def test_a_saved_derived_channel_is_restored_and_kept(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    out = tmp_path / 'd.flowsession'
    doc.save_session(str(out))
    _resave(out, lambda d: d['samples'][0].update(
        derived={'D:Ratio': '[FL1-A] / [FL2-A]'}))
    doc2 = Document()
    doc2.open_session(str(out))
    s = doc2._samples[name].data
    assert 'D:Ratio' in s.data.columns
    assert s.derived == {'D:Ratio': '[FL1-A] / [FL2-A]'}
    doc2.save_session(str(tmp_path / 'e.flowsession'))
    assert _entry(tmp_path / 'e.flowsession')['derived'] == {
        'D:Ratio': '[FL1-A] / [FL2-A]'}


def test_the_qc_setting_runs_qc_and_the_save_records_it(tmp_path):
    p = spill_fcs(tmp_path / 's1.fcs')
    off = Document()
    off.open_fcs(p)
    assert off._samples['s1'].data.qc_applied is False
    doc = Document(qc=True)
    doc.open_fcs(p)
    assert doc._samples['s1'].data.qc_applied is True
    doc.save_session(str(tmp_path / 'q.flowsession'))
    assert _entry(tmp_path / 'q.flowsession')['qc_applied'] is True
    off.save_session(str(tmp_path / 'r.flowsession'))
    assert _entry(tmp_path / 'r.flowsession')['qc_applied'] is False


def test_a_workspace_group_is_saved_as_the_samples_trial(tmp_path):
    a = make_fcs(tmp_path / 'run' / 'a.fcs', seed=1)
    wsp = write_wsp(tmp_path / 'run' / 'x.wsp', [('a', a, 'Stim', False)])
    doc = Document()
    doc.import_wsp(wsp)
    out = tmp_path / 't.flowsession'
    doc.save_session(str(out))
    assert _entry(out)['trial'] == 'Stim'
    doc2 = Document()
    doc2.open_session(str(out))
    assert doc2._samples['a'].trial == 'Stim'


def test_a_missing_samples_link_is_carried_to_a_new_folder(tmp_path):
    """Its saved path is another computer's: the link it was saved with
    (``raw/s1.fcs`` from the session folder) is what finds it from a
    session saved elsewhere."""
    doc, name, *_ = gated_doc(tmp_path)
    one = tmp_path / 'one.flowsession'
    doc.save_session(str(one))
    _resave(one, lambda d: d['samples'][0].update(
        path='/other/computer/s1.fcs', rel_path='raw/s1.fcs'))
    os.rename(tmp_path / 'raw' / 's1.fcs', tmp_path / 'raw' / 'gone')
    doc2 = Document()
    assert doc2.open_session(str(one))['missing'] == [name]
    os.makedirs(tmp_path / 'sub')
    doc2.save_session(str(tmp_path / 'sub' / 'two.flowsession'))
    assert _entry(tmp_path / 'sub' / 'two.flowsession')['rel_path'] == \
        '../raw/s1.fcs'
    os.rename(tmp_path / 'raw' / 'gone', tmp_path / 'raw' / 's1.fcs')
    doc3 = Document()
    assert doc3.open_session(str(tmp_path / 'sub' / 'two.flowsession'))[
        'samples'] == [name]


def test_a_sidecar_samples_link_is_carried_to_a_new_folder(tmp_path):
    """A sample loaded from its sidecar whose raw FCS is not found keeps
    the link it was saved with, carried over to the new folder."""
    _doc, name = sidecar_doc(tmp_path)
    sess = tmp_path / 'run.flowsession'
    _resave(sess, lambda d: d['samples'][0].update(
        path='/other/computer/s1.fcs', rel_path='raw/s1.fcs'))
    os.rename(tmp_path / 'raw' / 's1.fcs', tmp_path / 'raw' / 'gone')
    doc2 = Document()
    assert doc2.open_session(str(sess))['samples'] == [name]
    assert doc2._samples[name].sidecar
    assert not os.path.isfile(doc2._samples[name].data.source_fcs)
    os.makedirs(tmp_path / 'sub')
    out = tmp_path / 'sub' / 'two.flowsession'
    doc2.save_session(str(out))
    assert _entry(out)['rel_path'] == '../raw/s1.fcs'


def test_the_active_sample_is_the_sessions_and_moves_on_removal(tmp_path):
    out = _two_sample_session(tmp_path, 's2')
    doc = Document()
    doc.open_session(str(out))
    assert doc.active == 's2'
    doc.save_session(str(tmp_path / 'again.flowsession'))
    again = json.loads((tmp_path / 'again.flowsession').read_text('utf-8'))
    assert again['active_sample'] == 's2'
    doc.remove_sample('s2')
    assert doc.active == 's1'
    doc.remove_sample('s1')
    assert doc.active is None


# ── labels ──────────────────────────────────────────────────────────────────

def _labels(doc, name):
    return {c['name']: c['label'] for c in doc.channels(name)}


def test_the_labels_setting_names_raw_and_sidecar_samples(tmp_path):
    doc = Document(labels='FL1-A=Lab one')
    name = doc.open_fcs(spill_fcs(tmp_path / 'raw' / 's1.fcs'))
    assert _labels(doc, name)['FL1-A'] == 'Lab one'
    sidecar_doc(tmp_path)                       # run.flowsession, sidecar
    doc2 = Document(labels='FL1-A=Lab one')
    doc2.open_session(str(tmp_path / 'run.flowsession'))
    assert doc2._samples['s1'].sidecar
    assert _labels(doc2, 's1')['FL1-A'] == 'Lab one'


def test_a_later_sessions_labels_add_to_the_firsts(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    for stem, labels in (('a', {'FL2-A': 'First'}),
                         ('b', {'FL2-A': 'Second', 'FL3-A': 'Three'})):
        doc.save_session(str(tmp_path / f'{stem}.flowsession'))
        _resave(tmp_path / f'{stem}.flowsession',
                lambda d, lb=labels: d.update(channel_labels=lb))
    both = Document()
    both.open_session(str(tmp_path / 'a.flowsession'))
    both.open_session(str(tmp_path / 'b.flowsession'))
    got = _labels(both, name)
    assert (got['FL2-A'], got['FL3-A']) == ('First', 'Three')


def test_a_marker_named_on_the_phone_wins_and_is_saved(tmp_path):
    """set_channel_label names a channel for every sample, over the
    session's own label, and a saved session reopens with it."""
    doc, name, *_ = gated_doc(tmp_path)
    doc.save_session(str(tmp_path / 'a.flowsession'))
    _resave(tmp_path / 'a.flowsession',
            lambda d: d.update(channel_labels={'FL2-A': 'Old', 'FL3-A': 'Three'}))
    spill_fcs(tmp_path / 'raw' / 's2.fcs', seed=2)
    both = Document()
    both.open_session(str(tmp_path / 'a.flowsession'))
    other = both.open_fcs(str(tmp_path / 'raw' / 's2.fcs'))
    both.set_channel_label('FL2-A', '  New \t\n two ')
    both.set_channel_label('FL1-A', 'One')
    for s in (name, other):
        got = _labels(both, s)
        assert (got['FL1-A'], got['FL2-A']) == ('One', 'New two')
    assert _labels(both, name)['FL3-A'] == 'Three'
    both.save_session(str(tmp_path / 'b.flowsession'))
    saved = json.loads((tmp_path / 'b.flowsession').read_text('utf-8'))
    assert saved['channel_labels'] == {'FL1-A': 'One', 'FL2-A': 'New two',
                                       'FL3-A': 'Three'}
    again = Document()
    again.open_session(str(tmp_path / 'b.flowsession'))
    assert _labels(again, name)['FL2-A'] == 'New two'


def test_a_cleared_marker_puts_back_the_name_it_was_opened_with(tmp_path):
    """An empty label drops the name set here: the session's (or file's)
    label shows and is saved again, and a channel neither named shows its
    detector."""
    doc, name, *_ = gated_doc(tmp_path)
    doc.save_session(str(tmp_path / 'a.flowsession'))
    _resave(tmp_path / 'a.flowsession',
            lambda d: d.update(channel_labels={'FL2-A': 'Old'}))
    again = Document()
    again.open_session(str(tmp_path / 'a.flowsession'))
    again.set_channel_label('FL2-A', 'New')
    again.set_channel_label('FL3-A', 'Three')
    assert _labels(again, name)['FL2-A'] == 'New'
    again.set_channel_label('FL2-A', '   ')
    again.set_channel_label('FL3-A', '')
    assert (_labels(again, name)['FL2-A'], _labels(again, name)['FL3-A']) == ('Old', 'FL3-A')
    again.save_session(str(tmp_path / 'b.flowsession'))
    saved = json.loads((tmp_path / 'b.flowsession').read_text('utf-8'))
    assert saved['channel_labels'] == {'FL2-A': 'Old'}


def test_a_marker_on_a_document_with_no_session_keeps_the_next_sessions_view(
        tmp_path):
    """Naming a marker before any session is opened leaves the first
    session's own view settings (its scales) to be read, as without it."""
    doc, *_ = gated_doc(tmp_path)
    doc.save_session(str(tmp_path / 'a.flowsession'))
    _resave(tmp_path / 'a.flowsession',
            lambda d: d.update(channel_scale={'FL1-A': 'asinh:c=40'}))
    spill_fcs(tmp_path / 'raw' / 's2.fcs', seed=2)
    fresh = Document()
    fresh.open_fcs(str(tmp_path / 'raw' / 's2.fcs'))
    fresh.set_channel_label('FL1-A', 'One')
    fresh.open_session(str(tmp_path / 'a.flowsession'))
    fresh.save_session(str(tmp_path / 'b.flowsession'))
    saved = json.loads((tmp_path / 'b.flowsession').read_text('utf-8'))
    assert saved['channel_scale'] == {'FL1-A': 'asinh:c=40'}
    assert saved['channel_labels']['FL1-A'] == 'One'


@pytest.mark.parametrize('channel, label', [
    ('FL9-A', 'Nine'), ('FL1-A', 'a\x01b'), ('FL1-A', 'a\x7f'), ('FL1-A', '\x01a'),
    ('FL1-A', 'a\x1fb'), ('FL1-A', 'a\x85b'), ('FL1-A', 'a\x9f'), ('FL1-A', 'x' * 65),
    ('FL1-A', 7), (None, 'One')])
def test_a_marker_label_that_cannot_be_set_changes_nothing(
        tmp_path, channel, label):
    doc, name, *_ = gated_doc(tmp_path)
    before = _labels(doc, name)
    with pytest.raises(DocumentError):
        doc.set_channel_label(channel, label)
    assert _labels(doc, name) == before
    doc.set_channel_label('FL1-A', 'x' * 64)
    assert _labels(doc, name)['FL1-A'] == 'x' * 64


def test_a_relative_path_is_kept_absolute(tmp_path, monkeypatch):
    spill_fcs(tmp_path / 'raw' / 's1.fcs')
    monkeypatch.chdir(tmp_path)
    doc = Document()
    name = doc.open_fcs(os.path.join('raw', 's1.fcs'))
    want = str(tmp_path / 'raw' / 's1.fcs')
    assert doc.samples()[0]['path'] == want
    doc.save_session(str(tmp_path / 'p.flowsession'))
    assert _entry(tmp_path / 'p.flowsession')['path'] == want
    assert name == 's1'


# ── gates ───────────────────────────────────────────────────────────────────

def test_an_update_keeps_a_gate_switched_off(tmp_path):
    """And its expand state ('open') in the tree, as the editor keeps it."""
    doc, name, _cells, _pos, mid = gated_doc(tmp_path)
    doc._samples[name].gates[mid]['enabled'] = False
    doc._samples[name].gates[mid]['open'] = False
    g = doc.update_gate(name, mid, {'kind': 'interval', 'channel': 'FL2-A',
                                    'lo': 0.2, 'hi': 0.7})
    assert g['enabled'] is False
    assert doc._samples[name].gates[mid]['open'] is False
    assert doc.gate_tree(name)[0]['children'][1]['enabled'] is False
    g = doc.update_gate(name, mid, {'kind': 'interval', 'channel': 'FL2-A',
                                    'lo': 0.2, 'hi': 0.7, 'enabled': True})
    assert g['enabled'] is True


def _boolean(*ops):
    return {'kind': 'boolean', 'op': 'and', 'operands': list(ops),
            'name': 'both'}


def test_a_gate_cannot_rest_on_itself_through_a_boolean(tmp_path):
    """L1: an operand that is the gate, or rests on it, and a parent that
    rests on it through an operand, are refused before anything changes;
    so are missing operands."""
    doc, name, cells, pos, mid = gated_doc(tmp_path)
    both = doc.add_gate(name, _boolean(pos, mid), cells)['id']
    under = doc.add_gate(name, {'kind': 'threshold', 'channel': 'FL3-A',
                                'value': 0.4, 'name': 'FL3+'}, pos)['id']
    gates = doc._samples[name].gates
    before = json.dumps(gates, sort_keys=True)
    for gid, gate in ((pos, _boolean(both)),            # through `both`
                      (pos, _boolean(pos)),             # itself
                      (both, _boolean(both, mid)),
                      (pos, _boolean(under)),           # under pos
                      (pos, {'kind': 'threshold', 'channel': 'FL1-A',
                             'value': 0.5, 'parent_id': both})):
        with pytest.raises(DocumentError, match='rest on itself'):
            doc.update_gate(name, gid, gate)
    with pytest.raises(DocumentError, match="no gate 'g99' to combine"):
        doc.add_gate(name, _boolean(pos, 'g99'), cells)
    with pytest.raises(DocumentError, match='needs the gates it combines'):
        doc.add_gate(name, _boolean(), cells)
    with pytest.raises(DocumentError, match='needs the gates it combines'):
        doc.update_gate(name, both, {'kind': 'boolean', 'op': 'and'})
    assert json.dumps(gates, sort_keys=True) == before
    # A boolean that rests on nothing of its own is fine to update.
    doc.update_gate(name, both, _boolean(mid))
    assert doc.gate(name, both)['operands'] == [mid]


def test_density_says_why_a_population_is_unknown(tmp_path):
    doc, name, cells, pos, _mid = gated_doc(tmp_path)
    assert doc.density(name, 'FL1-A', 'FL2-A', gate=pos)['unknown'] is None
    assert doc.density(name, 'FL1-A', 'FL2-A')['unknown'] is None
    gid = doc.add_gate(name, {'kind': 'threshold', 'channel': 'FL9-A',
                              'value': 0.5}, cells)['id']
    d = doc.density(name, 'FL1-A', 'FL2-A', gate=gid)
    assert d['events'] == 0 and 'FL9-A' in d['unknown']


# ── the session schema ──────────────────────────────────────────────────────

def test_the_view_defaults_and_written_keys_are_the_session_keys(tmp_path):
    """L5: a key added to SESSION_KEYS must reach _VIEW_DEFAULTS (or the
    keys _session_data writes itself), or the Document writes sessions
    without it."""
    from openflo.session_format import SESSION_KEYS
    own = {'format', 'version', 'app_version', 'created', 'active_sample',
           'samples', 'sample_gates'}
    assert set(document._VIEW_DEFAULTS) | own == set(SESSION_KEYS)
    assert not set(document._VIEW_DEFAULTS) & own
    doc, *_ = gated_doc(tmp_path)
    doc.save_session(str(tmp_path / 'k.flowsession'))
    data = json.loads((tmp_path / 'k.flowsession').read_text('utf-8'))
    assert set(data) == set(SESSION_KEYS)

