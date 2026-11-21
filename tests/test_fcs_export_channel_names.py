"""FCS exports write the session's channel names by default; Preferences >
Export, "FCS export channel names", keeps each file's original names.

Sky 11-05: "Channel names do default common names, but preference for
original." The rule: each channel gets the label set explicitly in the
session (a relabel such as Label channels from vial labels, or a template's
label, also after a save and reopen), else the file's own label. Labels the
view copied from a file's $PnS never reach another file, even once that
file is removed.

Every check writes real FCS files from generated data (synthetic sample
names, generic markers CD9xx and channels FL1-A ...), reads their $PnS back,
and is proved to fail on the old behaviour by the mutation check saved with
this change.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import tests.test_session_reopen_losses as reopen
from openflo import prefs
from openflo.fcs_export import export_channel_labels

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
# Two files with different panels on the same detectors.
OWN_A = {'FL1-A': 'CD901b', 'FL2-A': 'CD902'}
OWN_B = {'FL1-A': 'CD905', 'FL2-A': 'CD906'}


def _pns(path):
    """``{$PnN: $PnS}`` of an FCS file, read back from disk ('' for none)."""
    import flowio
    f = flowio.FlowData(str(path))
    return {c['pnn']: c.get('pns', '') or '' for c in f.channels.values()}


def _set_pref(monkeypatch, tmp_path, value):
    p = tmp_path / 'prefs.json'
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(p))
    if value is not None:
        p.write_text(json.dumps({'fcs_export_channel_names': value}))
    return p


# ── the rule, without Tk ────────────────────────────────────────────────────

def test_original_returns_the_files_own_labels_unchanged():
    own = {'FL1-A': 'CD905'}
    assert export_channel_labels(own, 'original', file_own=OWN_B,
                                 explicit={'FL1-A': 'CD907'}) is own
    assert export_channel_labels(None, 'original',
                                 explicit={'FL1-A': 'CD907'}) is None


def test_session_names_are_the_explicit_label_else_the_files_own():
    # FL2-A: set explicitly. FL1-A: the file's own, whatever ``own`` held
    # (a single export starts from the view's labels: file A's names).
    got = export_channel_labels(dict(OWN_A), 'session', file_own=OWN_B,
                                explicit={'FL2-A': 'CD907'})
    assert got == {'FL1-A': 'CD905', 'FL2-A': 'CD907'}
    # An explicit label wins even where it is some file's own name.
    assert export_channel_labels(dict(OWN_A), 'session',
                                 explicit={'FL1-A': 'CD905'}) == {
        'FL1-A': 'CD905', 'FL2-A': 'CD902'}
    assert export_channel_labels(None, 'session') == {}


def test_the_trail_rebuilds_the_explicit_labels():
    """explicit_labels_from_trail replays channel_labels.set records: their
    ``explicit`` field when present, else (a trail saved before it) every
    label but a revert's; a label the saved session no longer shows does
    not count."""
    from openflo.channel_label_trail import explicit_labels_from_trail

    def rec(labels, source='vial labels', **kw):
        return {'action': 'channel_labels.set',
                'details': dict(labels=labels, source=source, **kw)}
    trail = [{'action': 'sample.load', 'details': {}},
             rec({'FL1-A': 'CD905', 'FL2-A': 'CD907'},
                 explicit={'FL1-A': 'CD905', 'FL2-A': 'CD907'}),
             rec({'FL2-A': 'CD902'}, source='Undo or Redo', explicit={})]
    saved = {'FL1-A': 'CD905', 'FL2-A': 'CD902'}
    assert explicit_labels_from_trail(trail, saved) == {'FL1-A': 'CD905'}
    # An old trail: no ``explicit``; the revert takes FL2-A out.
    old = [rec({'FL1-A': 'CD905', 'FL2-A': 'CD907'}),
           rec({'FL2-A': 'CD902'}, source="the file's labels")]
    assert explicit_labels_from_trail(old, saved) == {'FL1-A': 'CD905'}
    # A relabel later undone in a run that recorded no Undo: the saved
    # label differs, so it is not taken for an explicit one.
    assert explicit_labels_from_trail(
        [rec({'FL1-A': 'CD909'})], saved) == {}
    assert explicit_labels_from_trail(None, None) == {}


# ── the preference ──────────────────────────────────────────────────────────

def test_preference_defaults_to_session_names_and_round_trips(tmp_path,
                                                              monkeypatch):
    p = _set_pref(monkeypatch, tmp_path, None)
    assert prefs.load_fcs_export_channel_names() == 'session'
    p.write_text(json.dumps({'fcs_export_channel_names': 'original'}))
    assert prefs.load_fcs_export_channel_names() == 'original'
    p.write_text(json.dumps({'fcs_export_channel_names': 'bogus'}))
    assert prefs.load_fcs_export_channel_names() == 'session'
    prefs.write_pref('fcs_export_channel_names', 'original')
    assert prefs.load_fcs_export_channel_names() == 'original'


# ── GUI: two loaded files with different panels ────────────────────────────

def _write(path, labels, seed):
    from openflo.fcs_export import write_fcs
    rng = np.random.default_rng(seed)
    n = 2000
    df = pd.DataFrame(np.column_stack([
        rng.normal(100_000, 20_000, n), rng.normal(50_000, 10_000, n),
        rng.normal(5_000, 1_000, n), rng.normal(800, 100, n)]),
        columns=pd.Index(CH))
    write_fcs(df, str(path), channel_labels=labels)
    return str(path)


@pytest.fixture
def two(tmp_path, monkeypatch):
    """An editor holding sample_a (loaded first, active) and sample_b, whose
    files label FL1-A / FL2-A differently, each with a gate 'Big'."""
    import tkinter as tk

    from tests._background_work import _let_the_load_settle
    from tests.conftest import gui_unavailable
    from tests.test_beta_look import _load
    from tests.test_transform_records import (
        _new_editor,
        capture_warnings,
        isolate_home,
    )
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    ed = _new_editor(root)
    _load(root, ed, [('sample_a', _write(tmp_path / 'a.fcs', OWN_A, 1))])
    _load(root, ed, [('sample_b', _write(tmp_path / 'b.fcs', OWN_B, 2))])
    _let_the_load_settle(root, ed)
    gids = {}
    for name in ('sample_b', 'sample_a'):
        ed._set_active_sample(name)
        gids[name] = ed._add_gate(
            {'kind': 'threshold', 'channel': 'FSC-A', 'op': '>',
             'value': 60_000.0, 'name': 'Big'}, audit=False)
    assert ed._channel_labels['FL1-A'] == 'CD901b'      # seeded from file A
    ed._clear_undo_pending()        # a label change is its own Undo step
    yield root, ed, tmp_path, gids
    root.destroy()


def _export_all(ed, monkeypatch, out):
    from openflo import editor_populations
    from openflo.async_task import wait_for_job
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs_all()
    assert wait_for_job(ed._job_strip)
    return {name: _pns(out / name / 'Big.fcs')
            for name in ('sample_a', 'sample_b')}


def _export_one(ed, monkeypatch, name, gid, path):
    from openflo import editor_populations
    from openflo.async_task import wait_for_job
    monkeypatch.setattr(editor_populations.filedialog, 'asksaveasfilename',
                        lambda **k: str(path))
    ed._export_population_fcs(name, gid)
    assert wait_for_job(ed._job_strip)
    return _pns(path)


def _fl(pns):
    return {k: pns[k] for k in ('FL1-A', 'FL2-A')}


@pytest.mark.parametrize('pref', [None, 'session', 'original'])
def test_population_exports_follow_the_preference(two, monkeypatch, pref):
    """File > Export populations (all samples) and a single population's
    export: a relabel reaches every file under session names; file B never
    takes file A's name for FL1-A; 'original' writes each file's own."""
    _root, ed, tmp, gids = two
    _set_pref(monkeypatch, tmp, pref)
    ed._set_channel_labels({'FL2-A': 'CD907'}, source='test')
    bulk = _export_all(ed, monkeypatch, tmp / 'bulk')
    one = {n: _export_one(ed, monkeypatch, n, gids[n], tmp / f'{n}.fcs')
           for n in ('sample_a', 'sample_b')}
    if pref == 'original':
        want = {'sample_a': OWN_A, 'sample_b': OWN_B}
    else:
        want = {'sample_a': {'FL1-A': 'CD901b', 'FL2-A': 'CD907'},
                'sample_b': {'FL1-A': 'CD905', 'FL2-A': 'CD907'}}
    for got in (bulk, one):
        assert {n: _fl(p) for n, p in got.items()} == want
        assert got['sample_b']['FSC-A'] == ''          # no marker, no $PnS


def test_undoing_the_relabel_takes_it_out_of_the_export(two, monkeypatch):
    _root, ed, tmp, gids = two
    _set_pref(monkeypatch, tmp, None)
    ed._set_channel_labels({'FL2-A': 'CD907'}, source='test')
    ed._undo()
    got = _export_one(ed, monkeypatch, 'sample_b', gids['sample_b'],
                      tmp / 'out_b.fcs')
    assert _fl(got) == OWN_B


def _gid(ed, name, gate='Big'):
    return next(g for g, d in ed._sample_gates[name].items()
                if d.get('name') == gate)


def _save_and_reopen(root, ed, tmp):
    """Write ``ed``'s session to disk and reopen it in a fresh editor."""
    path = tmp / 's.flowsession'
    ed._write_session(str(path))
    ed2, _jobs = reopen._reopen(root, path)
    return ed2


def test_removing_the_first_file_leaks_none_of_its_names(two, monkeypatch):
    """F1: the view's labels are seeded from sample_a's $PnS. Once sample_a
    is removed, sample_b must still export its own names, nobody having
    relabelled anything."""
    _root, ed, tmp, gids = two
    _set_pref(monkeypatch, tmp, None)
    ed._remove_samples(['sample_a'])
    assert 'sample_a' not in ed._samples
    assert ed._channel_labels['FL1-A'] == 'CD901b'      # still in the view
    got = _export_one(ed, monkeypatch, 'sample_b', gids['sample_b'],
                      tmp / 'out_b.fcs')
    assert _fl(got) == OWN_B
    bulk = tmp / 'bulk'
    bulk.mkdir()
    from openflo import editor_populations
    from openflo.async_task import wait_for_job
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(bulk))
    ed._export_populations_fcs_all()
    assert wait_for_job(ed._job_strip)
    assert _fl(_pns(bulk / 'sample_b' / 'Big.fcs')) == OWN_B
    # A channel sample_b holds no label for takes none from the view (whose
    # FL1-A is sample_a's); "original" writes what it always did.
    del ed._samples['sample_b'].channel_labels['FL1-A']
    got = _export_one(ed, monkeypatch, 'sample_b', gids['sample_b'],
                      tmp / 'b1.fcs')
    assert _fl(got) == {'FL1-A': '', 'FL2-A': 'CD906'}
    _set_pref(monkeypatch, tmp, 'original')
    got = _export_one(ed, monkeypatch, 'sample_b', gids['sample_b'],
                      tmp / 'b2.fcs')
    assert _fl(got) == {'FL1-A': 'CD901b', 'FL2-A': 'CD906'}


def test_a_relabel_to_another_files_name_survives_save_and_reopen(
        two, monkeypatch):
    """F2: relabel sample_a's FL1-A to sample_b's own name, save, reopen:
    sample_a still exports it; a label nobody set (sample_a's own FL2-A,
    saved because the view showed it) stays sample_a's alone."""
    root, ed, tmp, _gids = two
    _set_pref(monkeypatch, tmp, None)
    ed._set_channel_labels({'FL1-A': 'CD905'}, source='test')
    ed2 = _save_and_reopen(root, ed, tmp)
    try:
        got = {n: _fl(_export_one(ed2, monkeypatch, n, _gid(ed2, n),
                                  tmp / f'{n}_2.fcs'))
               for n in ('sample_a', 'sample_b')}
        assert got == {'sample_a': {'FL1-A': 'CD905', 'FL2-A': 'CD902'},
                       'sample_b': OWN_B}
        assert ed2._explicit_channel_labels == {'FL1-A': 'CD905'}
    finally:
        ed2.destroy()


def test_an_undone_relabel_stays_undone_after_save_and_reopen(two,
                                                             monkeypatch):
    """Undo of an explicit relabel to the view's own name changes no label
    the session saves; the trail's Undo record keeps it out on reopen."""
    root, ed, tmp, _gids = two
    _set_pref(monkeypatch, tmp, None)
    ed._set_channel_labels({'FL1-A': 'CD901b'}, source='test')
    ed._undo()
    ed2 = _save_and_reopen(root, ed, tmp)
    try:
        got = _export_one(ed2, monkeypatch, 'sample_b',
                          _gid(ed2, 'sample_b'), tmp / 'b2.fcs')
        assert _fl(got) == OWN_B
        assert ed2._explicit_channel_labels == {}
    finally:
        ed2.destroy()


def test_a_relabel_to_the_views_own_name_reaches_every_file(two,
                                                            monkeypatch):
    """F3: naming FL1-A as the view already shows it (sample_a's CD901b) is
    an explicit choice: it is recorded and sample_b's export takes it. Undo
    takes it back out; applying it twice is one change."""
    _root, ed, tmp, gids = two
    _set_pref(monkeypatch, tmp, None)
    changed = ed._set_channel_labels({'FL1-A': 'CD901b'}, source='test')
    assert changed == {'FL1-A': 'CD901b'}
    assert ed._set_channel_labels({'FL1-A': 'CD901b'}, source='test') == {}
    got = _export_one(ed, monkeypatch, 'sample_b', gids['sample_b'],
                      tmp / 'out_b.fcs')
    assert _fl(got) == {'FL1-A': 'CD901b', 'FL2-A': 'CD906'}
    ed._undo()
    assert ed._channel_labels['FL1-A'] == 'CD901b'
    got = _export_one(ed, monkeypatch, 'sample_b', gids['sample_b'],
                      tmp / 'b1.fcs')
    assert _fl(got) == OWN_B
    ed._redo()
    got = _export_one(ed, monkeypatch, 'sample_b', gids['sample_b'],
                      tmp / 'b2.fcs')
    assert _fl(got)['FL1-A'] == 'CD901b'


def test_two_label_changes_in_one_event_are_one_undo_step(two):
    """A second change in the same Tk event joins the first's Undo step,
    which then puts back the explicit labels from before both."""
    _root, ed, _tmp, _gids = two
    ed._set_channel_labels({'FL1-A': 'CD905'}, source='test')
    ed._set_channel_labels({'FL2-A': 'CD907'}, source='test')
    assert ed._explicit_channel_labels == {'FL1-A': 'CD905',
                                           'FL2-A': 'CD907'}
    ed._undo()
    assert ed._explicit_channel_labels == {}
    assert _fl(ed._channel_labels) == OWN_A


def test_each_trail_record_says_which_labels_are_explicit(two):
    """channel_labels.set records carry ``explicit``: the record's channels
    that are explicit after it, whatever its source says; Undo and Redo of
    a label change are recorded the same way."""
    root, ed, _tmp, _gids = two
    ed._set_channel_labels({'FL1-A': 'CD905'}, source='test')
    root.update()                       # the next change: its own Undo step
    ed._set_channel_labels({'FL1-A': 'CD901b'}, source='test',
                           explicit=False)
    ed._undo()
    recs = [e['details'] for e in ed._audit_log.to_list()
            if e['action'] == 'channel_labels.set']
    assert [(r['source'], r['labels'], r['explicit']) for r in recs] == [
        ('test', {'FL1-A': 'CD905'}, {'FL1-A': 'CD905'}),
        ('test', {'FL1-A': 'CD901b'}, {}),
        ('Undo or Redo', {'FL1-A': 'CD905'}, {'FL1-A': 'CD905'})]


def test_revert_to_the_files_labels_clears_the_explicit_ones(two,
                                                             monkeypatch):
    """Label channels from vial labels, "Revert to the file's labels": the
    reverted channels are no longer explicit, even one whose label the
    revert leaves as it is; Undo makes them explicit again."""
    from openflo.ui_vial_label import VialLabelWindow
    root, ed, tmp, gids = two
    _set_pref(monkeypatch, tmp, None)
    ed._set_active_sample('sample_a')
    ed._set_channel_labels({'FL1-A': 'CD901b', 'FL2-A': 'CD907'},
                           source='test')
    root.update()                       # the next change: its own Undo step
    win = VialLabelWindow(ed)
    try:
        win.revert()
    finally:
        win.destroy()
    assert ed._channel_labels['FL2-A'] == 'CD902'
    got = _export_one(ed, monkeypatch, 'sample_b', gids['sample_b'],
                      tmp / 'out_b.fcs')
    assert _fl(got) == OWN_B
    assert ed._explicit_channel_labels == {}
    ed._undo()
    assert ed._explicit_channel_labels == {'FL1-A': 'CD901b',
                                           'FL2-A': 'CD907'}


def test_a_template_label_reaches_every_file_and_a_reopen(two, monkeypatch):
    """A template's labels are explicit session names, even one that is a
    loaded file's own name; a reopened session keeps them."""
    root, ed, tmp, gids = two
    _set_pref(monkeypatch, tmp, None)
    tpl = tmp / 'tpl.json'
    tpl.write_text(json.dumps({'gates': [], 'labels': {'FL2-A': 'CD911',
                                                       'FL1-A': 'CD901b'}}))
    # The labels go in on Apply (to sample_a; the template has no gates).
    monkeypatch.setattr(ed, '_ask_template_apply',
                        lambda *a, **k: (['sample_a'], False))
    ed._apply_template_path(str(tpl))
    assert ed._channel_labels['FL2-A'] == 'CD911'
    got = _export_one(ed, monkeypatch, 'sample_b', gids['sample_b'],
                      tmp / 'out_b.fcs')
    assert _fl(got) == {'FL1-A': 'CD901b', 'FL2-A': 'CD911'}
    ed2 = _save_and_reopen(root, ed, tmp)
    try:
        got = _export_one(ed2, monkeypatch, 'sample_b',
                          _gid(ed2, 'sample_b'), tmp / 'b2.fcs')
        assert _fl(got) == {'FL1-A': 'CD901b', 'FL2-A': 'CD911'}
    finally:
        ed2.destroy()


# ── Preferences > Export ────────────────────────────────────────────────────

def test_preferences_box_stores_the_choice_and_has_a_tip(two, monkeypatch):
    from openflo.ui_preferences import PreferencesDialog
    root, ed, tmp, _gids = two
    p = _set_pref(monkeypatch, tmp, None)
    dlg = PreferencesDialog(ed)
    root.update()
    combo = dlg.fcs_names_combo
    assert list(combo.cget('values')) == [
        'Session names (default)', "Each file's original names"]
    assert combo.get() == 'Session names (default)'        # safe default
    combo.set("Each file's original names")
    combo.event_generate('<<ComboboxSelected>>')
    assert json.loads(p.read_text())['fcs_export_channel_names'] == 'original'
    assert prefs.load_fcs_export_channel_names() == 'original'
    combo.set('Session names (default)')
    combo.event_generate('<<ComboboxSelected>>')
    assert prefs.load_fcs_export_channel_names() == 'session'
    text = ed._tip_texts[str(combo)]
    assert 'exported FCS file' in text and 'original names' in text
    dlg.destroy()
    dlg2 = PreferencesDialog(ed)
    root.update()
    assert dlg2.fcs_names_combo.get() == 'Session names (default)'
    prefs.write_pref('fcs_export_channel_names', 'original')
    dlg2.destroy()
    dlg3 = PreferencesDialog(ed)
    root.update()
    assert dlg3.fcs_names_combo.get() == "Each file's original names"
    dlg3.destroy()


# ── Debarcode: Save each as FCS ─────────────────────────────────────────────

@pytest.mark.parametrize('pref', ['session', 'original'])
def test_debarcode_save_each_follows_the_preference(tmp_path, monkeypatch,
                                                    pref):
    import tkinter as tk
    from tkinter import filedialog

    from openflo.async_task import wait_for_job
    from tests._tk_wait import pump_until
    from tests.conftest import gui_unavailable
    from tests.test_debarcode import BC, SAMPLES, _pooled_editor
    from tests.test_transform_records import (
        _new_editor,
        capture_warnings,
        isolate_home,
    )
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    try:
        ed = _new_editor(root)
        env = (root, ed, tmp_path)
        w, _s, _parent, _truth = _pooled_editor(env, monkeypatch)
        _set_pref(monkeypatch, tmp_path, pref)
        ed._set_channel_labels({BC[1]: 'CD907'}, source='test')
        for var, ch in zip(w._map_vars, BC, strict=True):
            var.set(ed._fmt_channel(ch))         # the key's channels, renamed
        out = tmp_path / 'split'
        out.mkdir()
        monkeypatch.setattr(filedialog, 'askdirectory', lambda **k: str(out))
        w._save_fcs()
        assert wait_for_job(ed._job_strip)
        assert pump_until(root, lambda: 'Saved' in w.status_var.get()), (
            w.status_var.get(), ed.status_var.get())
        got = _pns(out / f'{SAMPLES[0]}.fcs')
        assert got[BC[0]] == 'BC1'                     # the file's own
        assert got[BC[1]] == ('CD907' if pref == 'session' else '')
    finally:
        root.destroy()


# ── Normalize batches: written copies ───────────────────────────────────────

@pytest.mark.parametrize('pref', ['session', 'original'])
def test_normalized_copies_follow_the_preference(tmp_path, monkeypatch, pref):
    from openflo.ui_batch_normalize import BatchNormalizeWindow
    from tests.conftest import editor_or_skip
    from tests.test_batch_normalize_dialog import _sample
    from tests.test_transform_records import isolate_home
    isolate_home(monkeypatch, tmp_path)
    root, ed = editor_or_skip(accept_confirms=False)
    try:
        rng = np.random.default_rng(11)
        for name, scale, date in (('d1_ref', 1.0, '01-JAN-2026'),
                                  ('d1_study', 1.0, '01-JAN-2026'),
                                  ('d2_ref', 1.3, '08-JAN-2026'),
                                  ('d2_study', 1.3, '08-JAN-2026')):
            s = _sample(name, scale, 0.5, date, rng)
            own = OWN_A if name.startswith('d1') else OWN_B
            s.channel_labels = {c: own.get(c, c) for c in s.data.columns}
            ed._samples[name] = s
            ed._sample_order.append(name)
            ed._sample_trial[name] = 'Run' + name[1]
            ed._sample_gates.setdefault(name, {})
            ed._sample_gate_order.setdefault(name, [])
        ed._channel_labels = dict(ed._samples['d1_ref'].channel_labels)
        _set_pref(monkeypatch, tmp_path, pref)
        ed._set_channel_labels({'FL2-A': 'CD907'}, source='test')
        w = BatchNormalizeWindow(ed)
        w.mode_var.set('goal')
        w.meta_var.set(1)
        w.apply_var.set(False)
        w.copies_var.set(True)
        w.dir_var.set(str(tmp_path / 'out'))
        w._run(sync=True)
        assert w._result is not None
        got = {n: _fl(_pns(tmp_path / 'out' / f'{n}_norm.fcs'))
               for n in ('d1_ref', 'd2_ref')}
        if pref == 'original':
            assert got == {'d1_ref': OWN_A, 'd2_ref': OWN_B}
        else:
            assert got == {'d1_ref': {'FL1-A': 'CD901b', 'FL2-A': 'CD907'},
                           'd2_ref': {'FL1-A': 'CD905', 'FL2-A': 'CD907'}}
        w.destroy()
    finally:
        root.destroy()
