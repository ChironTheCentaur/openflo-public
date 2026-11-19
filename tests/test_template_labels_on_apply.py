"""Load template: its channel labels change only on Apply, inside the
template's one undo step (scope audit W2).

Before, ``_apply_template_path`` wrote the template's labels into the
session before the "Apply template to…" dialog, so Cancel (or Apply with no
sample ticked) still relabelled every sample, Undo said "Nothing to undo",
and the X / Y pickers jumped back to FSC / SSC.

Generated data only: two synthetic samples with generic channels and a
template naming FL1-A 'MarkerX' and FL2-A 'MarkerY'.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import editor_or_skip

_COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
_NAMES = ('synthetic_a1', 'synthetic_a2')
_LABELS = {'FL1-A': 'MarkerX', 'FL2-A': 'MarkerY'}


@pytest.fixture
def ed():
    root, editor = editor_or_skip()
    editor._finish_plot_canvas()
    rng = np.random.default_rng(11)
    for k, name in enumerate(_NAMES):
        df = pd.DataFrame(rng.normal(1000 + 100 * k, 50, (500, 4)),
                          columns=_COLS)
        editor._samples[name] = SimpleNamespace(
            name=name, path=f'/data/{name}.fcs', data=df,
            fluor_channels=['FL1-A', 'FL2-A'],
            channel_labels={c: c for c in _COLS})
        editor._sample_order.append(name)
        editor._sample_plot_enabled[name] = True
        editor._sample_gates.setdefault(name, {})
        editor._sample_gate_seq.setdefault(name, 0)
        editor._sample_gate_order.setdefault(name, [])
    editor._channels = list(_COLS)
    editor._channel_labels = {c: c for c in _COLS}
    editor._populate_channel_combos()
    editor._set_active_sample(_NAMES[0])
    editor.x_combo.set('FL1-A')
    editor.y_combo.set('FL2-A')
    editor.update_idletasks()       # settle the undo-coalescing flag
    yield editor
    root.destroy()


def _template(tmp_path, labels=None):
    path = tmp_path / 'tpl.json'
    path.write_text(json.dumps({
        'version': 2,
        'gates': [{'id': 't1', 'kind': 'threshold', 'channel': 'FL1-A',
                   'value': 1000.0, 'parent_id': None, 'name': 'PopT'}],
        'labels': dict(_LABELS if labels is None else labels)}))
    return str(path)


def _apply(ed, path, choice):
    """Apply the template at ``path``, the dialog answering ``choice``;
    returns what the dialog was asked to name as relabelled."""
    asked = {}

    def dialog(relabel=None):
        asked['relabel'] = relabel
        return choice
    ed._ask_template_apply = dialog
    ed._apply_template_path(path)
    return asked.get('relabel')


def _gate_counts(ed):
    return {n: len(ed._sample_gates.get(n) or {}) for n in _NAMES}


def _shown_labels(ed):
    return {d: ed._channel_labels.get(d) for d in _LABELS}


def test_cancel_leaves_labels_session_labels_and_undo_alone(ed, tmp_path):
    _apply(ed, _template(tmp_path), None)                   # Cancel
    assert _shown_labels(ed) == {'FL1-A': 'FL1-A', 'FL2-A': 'FL2-A'}
    assert not (getattr(ed, '_session_channel_labels', None) or {})
    assert ed._undo_stack == []
    assert _gate_counts(ed) == dict.fromkeys(_NAMES, 0)
    assert 'MarkerX' not in ' '.join(ed.x_combo['values'])


def test_apply_with_no_sample_ticked_leaves_labels_alone(ed, tmp_path):
    _apply(ed, _template(tmp_path), ([], True))
    assert _shown_labels(ed) == {'FL1-A': 'FL1-A', 'FL2-A': 'FL2-A'}
    assert ed._undo_stack == []
    assert ed.status_var.get() == "No target samples selected."


def test_apply_relabels_and_keeps_them_for_later_loads(ed, tmp_path):
    _apply(ed, _template(tmp_path), ([_NAMES[0]], True))
    assert _shown_labels(ed) == _LABELS
    # The session's own labels: what a later first sample (or a reopen)
    # puts back over the files' names (editor_load._seed_first_sample_channels).
    assert {d: ed._session_channel_labels.get(d) for d in _LABELS} == _LABELS
    assert any('MarkerX' in v for v in ed.x_combo['values'])


def test_one_undo_reverts_labels_and_every_targets_gates(ed, tmp_path):
    _apply(ed, _template(tmp_path), (list(_NAMES), True))
    assert _gate_counts(ed) == dict.fromkeys(_NAMES, 1)
    assert len(ed._undo_stack) == 1         # labels + both samples: one step
    ed._undo()
    assert _shown_labels(ed) == {'FL1-A': 'FL1-A', 'FL2-A': 'FL2-A'}
    assert {d: ed._session_channel_labels.get(d) for d in _LABELS} \
        == {'FL1-A': None, 'FL2-A': None}
    assert _gate_counts(ed) == dict.fromkeys(_NAMES, 0)
    assert not any('MarkerX' in v for v in ed.x_combo['values'])
    ed._redo()
    assert _shown_labels(ed) == _LABELS
    assert _gate_counts(ed) == dict.fromkeys(_NAMES, 1)


def test_apply_keeps_the_chosen_x_and_y_channels(ed, tmp_path):
    from openflo.ui_logic import resolve_channel
    _apply(ed, _template(tmp_path), ([_NAMES[0]], True))
    assert resolve_channel(ed.x_combo.get()) == 'FL1-A'
    assert resolve_channel(ed.y_combo.get()) == 'FL2-A'
    assert 'MarkerX' in ed.x_combo.get()


def test_status_names_the_relabelled_channels(ed, tmp_path):
    _apply(ed, _template(tmp_path), ([_NAMES[0]], True))
    status = ed.status_var.get()
    assert status.endswith(
        "Relabeled FL1-A as MarkerX, FL2-A as MarkerY for the whole "
        "session."), status


def test_labels_already_shown_add_no_note_but_become_session_names(ed,
                                                                    tmp_path):
    """Labels the view already shows change nothing on screen, so the note
    names none; they still become explicit session names (an FCS export
    writes them in every file), in the same single undo step, and Undo
    takes them out of that set again."""
    ed._channel_labels.update(_LABELS)
    _apply(ed, _template(tmp_path), ([_NAMES[0]], True))
    assert 'Relabeled' not in ed.status_var.get()
    assert len(ed._undo_stack) == 1
    assert ed._explicit_channel_labels == _LABELS
    assert ed._undo_stack[-1]['labels']['explicit'] == {}
    ed._undo()
    assert ed._explicit_channel_labels == {}
    assert {d: ed._channel_labels[d] for d in _LABELS} == _LABELS


def test_relabel_note_caps_the_list():
    from openflo.template_apply import relabel_note as _relabel_note
    assert _relabel_note({}) == ''
    assert _relabel_note(None) == ''
    many = {f'FL{i}-A': f'M{i}' for i in range(1, 6)}
    assert _relabel_note(many) == (
        " Relabeled FL1-A as M1, FL2-A as M2, FL3-A as M3 and 2 more for "
        "the whole session.")


def test_dialog_is_told_only_the_labels_that_change(ed, tmp_path):
    ed._channel_labels['FL2-A'] = 'MarkerY'
    asked = _apply(ed, _template(tmp_path), None)
    assert asked == {'FL1-A': 'MarkerX'}


def test_template_without_labels_tells_the_dialog_nothing(ed, tmp_path):
    asked = _apply(ed, _template(tmp_path, labels={}), None)
    assert asked == {}


def _dialog_texts(ed, relabel):
    import tkinter as tk
    before = set(ed.winfo_children())
    ed.wait_window = lambda *_a: None          # return at once
    try:
        assert ed._ask_template_apply(relabel=relabel) is None
        (dlg,) = [w for w in ed.winfo_children()
                  if w not in before and isinstance(w, tk.Toplevel)]
        texts, todo = [], [dlg]
        while todo:
            w = todo.pop()
            todo.extend(w.winfo_children())
            try:
                texts.append(str(w.cget('text')))
            except tk.TclError:
                pass
        dlg.destroy()
        return texts
    finally:
        del ed.wait_window


def test_dialog_names_the_session_wide_relabel(ed):
    texts = _dialog_texts(ed, {'FL1-A': 'MarkerX'})
    assert ("Apply also relabels FL1-A as MarkerX, for every sample in the "
            "session (Undo puts the old labels back).") in texts
    assert not any('relabels' in t for t in _dialog_texts(ed, {}))


def test_relabel_preview_caps_the_list():
    from openflo.editor_template import _relabel_preview
    many = {f'FL{i}-A': f'M{i}' for i in range(1, 9)}
    assert _relabel_preview(many).startswith(
        "Apply also relabels FL1-A as M1, FL2-A as M2, FL3-A as M3, FL4-A as "
        "M4, FL5-A as M5, FL6-A as M6 and 2 more, for every sample")
