"""Tools > Panel & instrument QC > Label channels from vial labels
(ui_vial_label.VialLabelWindow): paste vial labels, check the preview, set
the session's channel labels, undo them, revert to the file's own, and keep
them through a session save and reopen.

Every sample is generated here: generic detector names (FL1-A ...), invented
laser and filter keywords, generic markers (CD901b, CD902 ...). The fluor
names come from fluor_catalog.FLUOR_SPECTRA at run time, chosen by where the
spectral model reads them on the generated detectors, so no fluor is named
in this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tests.test_session_reopen_losses as reopen
import tests.test_windows_nothing_hidden as sweep
from openflo import vial_label as vl
from openflo.fluor_catalog import FLUOR_SPECTRA
from tests._tk_wait import pump_until
from tests.conftest import editor_or_skip

virtual_screen = sweep.virtual_screen
no_dialogs = sweep.no_dialogs
engines_known = sweep.engines_known
env = reopen.env

COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL1-H', 'FL2-A', 'FL3-A', 'FL4-A',
        'Time']
# Invented optics: (laser nm, filter) per detector; FL4-A has none, so the
# model cannot place it and the user picks it by hand.
OPTICS = {'FL1-A': (488, '530/30'), 'FL1-H': (488, '530/30'),
          'FL2-A': (561, '586/15'), 'FL3-A': (640, '670/30')}
MODELLED = [(d, float(laser), float(f.split('/')[0]), float(f.split('/')[1]))
            for d, (laser, f) in OPTICS.items()]


def _meta(cols=COLS, optics=OPTICS, labels=None):
    """FCS keywords for `cols`: $PnN, and $PnL / $PnF from `optics`, $PnS
    from `labels`."""
    meta = {'$PAR': str(len(cols))}
    for i, c in enumerate(cols, 1):
        meta[f'$P{i}N'] = c
        if c in optics:
            meta[f'$P{i}L'] = f'{optics[c][0]} nm'
            meta[f'$P{i}F'] = optics[c][1]
        if labels and c in labels:
            meta[f'$P{i}S'] = labels[c]
    return meta


def _fluors_on(detector):
    """Catalogue fluors the model reads best on `detector` of MODELLED."""
    return [f for f in FLUOR_SPECTRA
            if vl.primary_detector(f, MODELLED).primary == detector]


FL1 = _fluors_on('FL1-A')
FL2 = _fluors_on('FL2-A')
UNSEEN = _fluors_on(None)       # no laser here excites them


def test_the_catalogue_has_fluors_for_each_case():
    assert FL1 and len(FL2) >= 2 and UNSEEN


def _sample(name='gen', cols=COLS, meta=None, labels=None):
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.random((300, len(cols))) * 1000, columns=cols)
    return FlowSample.from_dataframe(
        df, name=name, labels=labels,
        metadata=_meta(cols, labels=labels) if meta is None else meta)


def _register(ed, s):
    name = s.name
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._sample_gates[name] = {}
    ed._sample_gate_order[name] = []
    ed._sample_gate_seq[name] = 0
    ed._channels = list(s.data.columns)
    ed._channel_labels = dict(s.channel_labels)
    ed._populate_channel_combos()
    ed._active_sample = name
    ed._gates = ed._sample_gates[name]
    ed._gate_id_order, ed._gate_id_seq = [], 0
    return name


@pytest.fixture
def editor():
    root, ed = editor_or_skip()
    try:
        yield root, ed
    finally:
        root.destroy()


def _window(ed, sample=None, text=None):
    from openflo.ui_vial_label import VialLabelWindow
    _register(ed, sample if sample is not None else _sample())
    win = VialLabelWindow(ed)
    if text is not None:
        win.text.insert('1.0', text)
        win.refresh()
    return win


def _rows(win):
    return [(r.marker, r.label.fluor, r.detector, r.status)
            for r in win.rows]


# ── Splitting the pasted text ───────────────────────────────────────────────

def test_one_vial_per_line_or_per_blank_line_separated_block():
    from openflo.ui_vial_label import split_vials
    assert split_vials('a 1\n\n') == ['a 1']
    assert split_vials('a 1\nb 2\n  \n') == ['a 1', 'b 2']
    assert split_vials('\na 1\nlot 9\n\n\nb 2\nclone x\n') == [
        'a 1\nlot 9', 'b 2\nclone x']
    assert split_vials('  \n') == []


# ── The menu ────────────────────────────────────────────────────────────────

def test_the_tools_menu_opens_it_and_it_fits_a_small_screen(
        virtual_screen, no_dialogs, engines_known):
    from openflo.ui_vial_label import VialLabelWindow
    virtual_screen(1024, 768)
    r = sweep._root(1.0)
    try:
        ed = sweep._editor(r)
        items = [(m, i) for m, i, label in sweep._menu_items(ed)
                 if label == 'Label channels from vial labels…']
        assert len(items) == 1
        m, i = items[0]
        m.invoke(i)
        sweep._settle(r)
        wins = [w for w in sweep._toplevels(r).values()
                if isinstance(w, VialLabelWindow)]
        assert len(wins) == 1
        win = wins[0]
        assert win.sample is not None and win.channels == sweep.FLUOR
        assert sweep.problems(win) == []
        assert win.winfo_width() <= 1024 and win.winfo_height() <= 768
    finally:
        r.destroy()


def test_with_no_sample_it_says_one_is_needed(editor):
    from openflo.ui_vial_label import VialLabelWindow
    _root, ed = editor
    win = VialLabelWindow(ed)
    assert win.sample is None and not hasattr(win, 'text')
    assert 'No sample is loaded' in win.message_var.get()
    win._draw_rows()                        # the dialog's actions are safe
    win._on_text_modified()
    assert win.rows == []


# ── The preview ─────────────────────────────────────────────────────────────

def test_pasting_fills_the_preview_after_typing_pauses(editor):
    root, ed = editor
    win = _window(ed)
    win.text.insert('1.0', f"{FL1[0]} anti-human CD901b\n"
                           f"CD902 {FL2[0]}\nClone: XY-1\n")
    assert win.rows == []                    # not on every keystroke
    assert pump_until(root, lambda: len(win.rows) == 3, timeout=5)
    assert _rows(win)[:2] == [
        ('CD901b', FL1[0], 'FL1-A', vl.PLACED),
        ('CD902', FL2[0], 'FL2-A', vl.PLACED)]
    assert win.rows[2].status == vl.NO_MARKER
    assert win.rows[2].widgets['vial'].cget('text') == 'Clone: XY-1'
    status = win.rows[0].widgets['status'].cget('text')
    assert status == 'placed'
    assert str(win.rows[2].widgets['status'].cget('text')).startswith('⚠')
    assert win.planned_labels() == {'FL1-A': 'CD901b', 'FL1-H': 'CD901b',
                                    'FL2-A': 'CD902'}


def test_closing_while_typing_paused_drops_the_re_read(editor,
                                                       monkeypatch):
    from openflo import ui_vial_label
    root, ed = editor
    monkeypatch.setattr(ui_vial_label, 'REFRESH_MS', 60_000)
    win = _window(ed)
    root.update()
    win.text.insert('1.0', f"{FL1[0]} anti-human CD901b\n")
    root.update()                           # the re-read is now waiting
    job = win._pending
    assert job is not None and job in root.tk.call('after', 'info')
    win.destroy()
    assert job not in root.tk.call('after', 'info')


def test_two_vials_on_one_detector_are_a_marked_conflict(editor):
    _root, ed = editor
    win = _window(ed, text=f"CD902 {FL2[0]}\nCD903.1 {FL2[1]}\n"
                           f"{FL1[0]} CD901b\n")
    assert [r.status for r in win.rows] == [vl.CONFLICT, vl.CONFLICT,
                                            vl.PLACED]
    first = win.rows[0]
    assert first.detector == 'FL2-A'
    assert first.status_text.startswith('conflict: shares FL2-A with vial 2')
    runner = first.pick.runner_up
    assert runner and f'try {runner}' in first.status_text
    assert str(first.widgets['status'].cget('text')).startswith('⚠')
    assert win.planned_labels() == {'FL1-A': 'CD901b', 'FL1-H': 'CD901b'}
    # Moving one vial off the shared detector settles both.
    win.rows[1].detector_var.set('FL3-A')
    win.recheck()
    assert [r.status for r in win.rows] == [vl.PLACED] * 3
    assert win.planned_labels()['FL3-A'] == 'CD903.1'


def test_an_edited_marker_is_checked_again(editor):
    _root, ed = editor
    win = _window(ed, text=f"{FL1[0]} anti-human CD901b\n")
    row = win.rows[0]
    row.marker_var.set('  ')
    assert row.status == vl.NO_MARKER
    row.marker_var.set('CD999')
    assert row.status == vl.PLACED
    assert win.planned_labels()['FL1-A'] == 'CD999'


def test_a_channel_the_file_does_not_describe_is_picked_by_hand(editor):
    _root, ed = editor
    win = _window(ed, text=f"{UNSEEN[0]} CD904\nCD905 no dye named\n")
    assert win.unmatched == ('FL4-A',)
    assert [r.status for r in win.rows] == [vl.UNREADABLE, vl.NO_FLUOR]
    assert [r.detector for r in win.rows] == ['', '']
    box = win.rows[0].widgets['detector']
    assert 'FL4-A' in box.cget('values')
    win.rows[0].detector_var.set('FL4-A')
    win.rows[1].detector_var.set('FL3-A')
    win.recheck()
    assert [r.status for r in win.rows] == [vl.PLACED, vl.PLACED]
    assert win.planned_labels() == {'FL4-A': 'CD904', 'FL3-A': 'CD905'}


def test_with_no_described_detector_a_generic_instrument_only_suggests(
        editor):
    from openflo import panel_check as pc
    _root, ed = editor
    cols = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
    win = _window(ed, sample=_sample(cols=cols, meta={}),
                  text=f"{FL1[0]} anti-human CD901b\n")
    assert win.detectors == ()
    # A conventional instrument is the safe first guess for a file whose
    # channels are named FL1-A...
    assert win.generic_var.get() in pc.DETECTOR_SETS
    assert 'conventional' in win.generic_var.get()
    row = win.rows[0]
    assert (row.detector, row.status) == ('', 'pick')
    dets = pc.DETECTOR_SETS[win.generic_var.get()]['detectors']
    best = next(d for d in dets
                if d[0] == vl.primary_detector(FL1[0], dets).primary)
    assert f"{best[2]:g}/{best[3]:g} on the {best[1]:g} nm laser" in \
        row.status_text
    assert list(row.widgets['detector'].cget('values'))[1:] == [
        'FL1-A', 'FL2-A', 'FL3-A']
    row.detector_var.set('FL2-A')
    win.recheck()
    assert win.planned_labels() == {'FL2-A': 'CD901b'}


# ── Apply, undo, revert ─────────────────────────────────────────────────────

def test_apply_labels_the_channels_wherever_they_are_named(editor):
    from openflo.ui_logic import format_channel
    _root, ed = editor
    win = _window(ed, text=f"{FL1[0]} anti-human CD901b\n")
    ed.x_combo.set('FL1-A')
    win.apply()
    assert ed._channel_labels['FL1-A'] == 'CD901b'
    assert ed._channel_labels['FL1-H'] == 'CD901b'
    assert ed._session_channel_labels == {'FL1-A': 'CD901b',
                                          'FL1-H': 'CD901b'}
    assert ed._fmt_channel('FL1-A') == 'CD901b (FL1-A)'
    assert format_channel('FL1-A', ed._channel_labels) == 'CD901b (FL1-A)'
    assert 'CD901b (FL1-A)' in ed.x_combo['values']
    assert ed.x_combo.get() == 'CD901b (FL1-A)'     # the chosen axis kept
    assert ed.y_combo.get() == 'SSC-A'
    assert 'Labeled 2 channel(s)' in win.message_var.get()
    win.apply()
    assert win.message_var.get().startswith('Nothing to change')


def test_one_undo_reverses_apply_and_redo_brings_it_back(editor):
    _root, ed = editor
    win = _window(ed, text=f"{FL1[0]} anti-human CD901b\n")
    ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A', 'op': '>',
                  'value': 100.0, 'name': 'pos'}, audit=False)
    ed._clear_undo_pending()
    win.apply()
    ed._undo()
    assert ed._channel_labels['FL1-A'] == 'FL1-A'
    assert ed._session_channel_labels == {}
    assert len(ed._gates) == 1                  # the gate edit stays
    assert 'FL1-A' in ed.x_combo['values']
    ed._redo()
    assert ed._channel_labels['FL1-A'] == 'CD901b'
    assert ed._session_channel_labels['FL1-H'] == 'CD901b'
    assert len(ed._gates) == 1
    ed._undo()
    ed._undo()                                  # now the gate
    assert ed._gates == {} and ed._channel_labels['FL1-A'] == 'FL1-A'


def test_revert_puts_back_the_files_own_labels(editor):
    root, ed = editor
    s = _sample(labels={'FL2-A': 'CD906'})
    win = _window(ed, sample=s, text=f"CD902 {FL2[0]}\n")
    win.apply()
    root.update()                       # two clicks: two Undo steps
    assert ed._channel_labels['FL2-A'] == 'CD902'
    win.revert()
    assert ed._channel_labels['FL2-A'] == 'CD906'
    assert 'Reverted 1 channel(s)' in win.message_var.get()
    ed._undo()
    assert ed._channel_labels['FL2-A'] == 'CD902'


def test_revert_reads_the_files_labels_not_the_samples(editor):
    """A sample rebuilt from a session's data carries the session's labels;
    the file's own are its $PnS keywords."""
    from openflo.ui_vial_label import file_labels
    s = _sample(meta=_meta(labels={'FL2-A': 'CD906'}),
                labels={'FL2-A': 'CD907'})
    assert s.channel_labels['FL2-A'] == 'CD907'
    assert file_labels(s)['FL2-A'] == 'CD906'
    assert file_labels(_sample(meta={}, labels={'FL2-A': 'CD907'}))[
        'FL2-A'] == 'CD907'


def test_applied_labels_survive_a_session_save_and_reopen(env):
    root, ed, tmp = env
    from openflo.ui_vial_label import VialLabelWindow
    optics = {'FL1-A': (488, '530/30'), 'FL2-A': (561, '586/15')}
    meta = {k.lstrip('$'): v for k, v in
            _meta(reopen.CH, optics).items() if k[2:].endswith(('L', 'F'))}
    p = reopen._fcs(tmp / 'A.fcs', 1, meta=meta)
    ed._on_loaded('A', reopen._loaded(p))
    ed._active_sample = 'A'
    win = VialLabelWindow(ed)
    assert {d[0] for d in win.detectors} == {'FL1-A', 'FL2-A'}
    win.text.insert('1.0', f"{FL1[0]} anti-human CD901b\nCD902 {FL2[0]}\n")
    win.refresh()
    win.apply()
    win.destroy()
    path = tmp / 'v.flowsession'
    data = ed._write_session(str(path))
    assert data['channel_labels']['FL1-A'] == 'CD901b'
    ed2, _jobs = reopen._reopen(root, path)
    assert ed2._channel_labels['FL1-A'] == 'CD901b'
    assert ed2._channel_labels['FL2-A'] == 'CD902'
    assert 'CD902 (FL2-A)' in ed2.x_combo['values']


def test_two_applies_in_one_event_are_one_undo_step(editor):
    """A second label change in the same Tk event joins the first one's
    Undo step, which then restores the labels from before both."""
    _root, ed = editor
    win = _window(ed, text=f"{FL1[0]} anti-human CD901b\n")
    win.apply()
    win.text.delete('1.0', 'end')
    win.text.insert('1.0', f"CD902 {FL2[0]}\n")
    win.refresh()
    win.apply()
    assert ed._channel_labels['FL2-A'] == 'CD902'
    ed._undo()
    assert [ed._channel_labels[d] for d in ('FL1-A', 'FL1-H', 'FL2-A')] == [
        'FL1-A', 'FL1-H', 'FL2-A']
    assert ed._session_channel_labels == {}
    assert ed._undo_stack == []


def test_a_long_vial_line_is_cut_in_the_vial_column(editor):
    from openflo.ui_vial_label import VIAL_CHARS
    _root, ed = editor
    long = f"{FL1[0]} anti-human CD901b Antibody, catalog 900-123"
    win = _window(ed, text=long + "\n")
    shown = str(win.rows[0].widgets['vial'].cget('text'))
    assert len(shown) == VIAL_CHARS and shown.endswith('…')
    assert long.startswith(shown[:-1])


def test_vials_on_twin_channels_are_a_conflict(editor):
    """FL1-H reads what FL1-A reads: two vials there share a channel."""
    _root, ed = editor
    win = _window(ed, text=f"{FL1[0]} CD901b\nCD902 no dye named\n")
    win.rows[1].detector_var.set('FL1-H')
    win.recheck()
    assert [r.status for r in win.rows] == [vl.CONFLICT, vl.CONFLICT]
    assert 'with vial 1' in win.rows[1].status_text


def test_edits_are_kept_while_the_text_changes_around_them(editor):
    _root, ed = editor
    win = _window(ed, text=f"{FL1[0]} CD901b\n")
    win.rows[0].marker_var.set('CD908')
    win.rows[0].detector_var.set('FL4-A')
    win.text.insert('end', f"CD902 {FL2[0]}\n")
    win.refresh()
    assert [(r.marker, r.detector) for r in win.rows] == [
        ('CD908', 'FL4-A'), ('CD902', 'FL2-A')]


def test_parameter_names_are_every_pnn_in_order_or_none():
    from openflo.ui_vial_label import parameter_names
    assert parameter_names(_meta()) == COLS
    lower = {k.lstrip('$').lower(): v for k, v in _meta().items()}
    assert parameter_names(lower) == COLS
    gap = _meta()
    del gap[f'$P{len(COLS)}N']
    assert parameter_names(gap) == []
    assert parameter_names({'$PAR': 'x'}) == []
    assert parameter_names({}) == []


def test_a_repeated_detector_name_keeps_the_first_parameters_label():
    """The loader keeps a repeated $PnN for the first parameter and renames
    the second ('FL1-A (2)'), so Revert must give that name the first
    parameter's $PnS, not the second's."""
    from types import SimpleNamespace

    from openflo.ui_vial_label import file_labels
    meta = _meta(['FL1-A', 'FL2-A', 'FL1-A'], optics={})
    meta.update({'$P1S': 'CD901b', '$P2S': 'CD902', '$P3S': 'CD903'})
    got = file_labels(SimpleNamespace(
        metadata=meta, channel_labels={'FL1-A (2)': 'CD903'}))
    assert got == {'FL1-A': 'CD901b', 'FL2-A': 'CD902',
                   'FL1-A (2)': 'CD903'}


def test_fluorescence_channels_without_the_samples_own_list():
    from types import SimpleNamespace

    from openflo.ui_vial_label import fluorescence_channels
    df = pd.DataFrame(np.zeros((2, len(COLS) + 1)),
                      columns=COLS + ['cluster'])
    got = fluorescence_channels(SimpleNamespace(data=df, fluor_channels=[]))
    assert got == ['FL1-A', 'FL1-H', 'FL2-A', 'FL3-A', 'FL4-A']
    assert fluorescence_channels(SimpleNamespace(
        data=df, fluor_channels=['FL2-A'])) == ['FL2-A']


def test_undo_takes_off_a_label_the_channel_did_not_have(editor):
    _root, ed = editor
    _register(ed, _sample())
    ed._channel_labels.pop('FL4-A')
    assert ed._set_channel_labels({'FL4-A': 'CD909'}, 'test') == {
        'FL4-A': 'CD909'}
    ed._undo()
    assert 'FL4-A' not in ed._channel_labels
    ed._redo()
    assert ed._channel_labels['FL4-A'] == 'CD909'
