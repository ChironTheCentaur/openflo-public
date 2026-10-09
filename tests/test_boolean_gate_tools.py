"""Boolean gates in the editor: the AND/OR/NOT dialog, and what happens to a
boolean gate's operand ids when the tree is copied, templated or edited.

Coverage before: _open_boolean_dialog ran 1/57 statements and its do_create
0/11; _open_copy_gates_dialog 1/54. gate_to_mask's boolean branch itself is
tested (tests/test_ellipsoid_quadrant.py, tests/test_boolean_gate_fails_closed.py).

A boolean gate stores operands as gate ids of the SAME sample. Nothing that
re-ids gates remapped them (copy to samples, template apply, move, paste,
session restore), and gate_to_mask dropped an operand that no longer existed,
so a boolean gate silently changed meaning. Every re-id path now remaps them
through gating.remap_operands; a move or paste that would separate a boolean
from its operands is refused; a missing operand fails closed.

Ground truth: CD3, CD4 ~ N(0,1) independent; g1 = CD3 > 0, g2 = CD4 > 0.
"""
import os
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tests.conftest import gui_unavailable


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _frame(seed, n=20000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FSC-A': rng.uniform(1e4, 1e5, n),
                         'SSC-A': rng.uniform(1e4, 1e5, n),
                         'CD3': rng.normal(0, 1, n), 'CD4': rng.normal(0, 1, n)})


def _load(ed, name, df):
    cols = list(df.columns)
    ed._samples[name] = SimpleNamespace(
        name=name, path=rf'C:\exp\{name}.fcs', data=df,
        fluor_channels=['CD3', 'CD4'], channel_labels={c: c for c in cols})
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = cols
    ed._channel_labels = {c: c for c in cols}
    ed._populate_channel_combos()


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _toplevel(ed, prefix):
    (dlg,) = [w for w in _walk(ed.master)
              if isinstance(w, tk.Toplevel) and w.title().startswith(prefix)]
    return dlg


def _press(dlg, text):
    (b,) = [w for w in _walk(dlg) if isinstance(w, ttk.Button) and w.cget('text') == text]
    b.invoke()


def _make_boolean(ed, op_text, operand_labels, name):
    ed._open_boolean_dialog()
    dlg = _toplevel(ed, 'Boolean gate')
    for w in _walk(dlg):
        if isinstance(w, ttk.Radiobutton) and w.cget('text') == op_text:
            w.invoke()
        elif isinstance(w, ttk.Checkbutton) and w.cget('text') in operand_labels:
            w.invoke()
        elif isinstance(w, ttk.Entry):
            w.insert(0, name)
    before = set(ed._gates)
    _press(dlg, 'Create')
    (gid,) = [g for g in ed._gates if g not in before]
    return gid


def _mask(ed, sample, gid):
    import openflo.pipeline as fp
    return np.asarray(fp.cumulative_gate_mask(
        ed._sample_gates[sample], gid, ed._samples[sample].data))


def _with_two_thresholds(ed):
    import openflo.pipeline as fp
    _load(ed, 's1', _frame(5))
    ed._set_active_sample('s1')
    g1 = ed._add_gate({'kind': 'threshold', 'channel': 'CD3', 'value': 0.0,
                       'name': 'CD3+', 'parent_id': None})
    g2 = ed._add_gate({'kind': 'threshold', 'channel': 'CD4', 'value': 0.0,
                       'name': 'CD4+', 'parent_id': None})
    return g1, g2, fp.describe_gate(ed._gates[g1]), fp.describe_gate(ed._gates[g2])


def test_boolean_dialog_builds_and_or_not_populations():
    root, ed = _editor_or_skip()
    try:
        g1, g2, t1, t2 = _with_two_thresholds(ed)
        df = ed._samples['s1'].data
        x, y = df.CD3.values > 0, df.CD4.values > 0
        for op, labels, want in (('AND', [t1, t2], x & y), ('OR', [t1, t2], x | y),
                                 ('NOT', [t1], ~x)):
            gid = _make_boolean(ed, op, labels, f'{op} pop')
            g = ed._gates[gid]
            assert g['kind'] == 'boolean' and g['op'] == op.lower()
            assert g['operands'] == ([g1, g2] if op != 'NOT' else [g1])
            assert g['name'] == f'{op} pop' and g['parent_id'] is None
            assert (_mask(ed, 's1', gid) == want).all()
        # The dialog lists only non-boolean gates as operands.
        ed._open_boolean_dialog()
        dlg = _toplevel(ed, 'Boolean gate')
        listed = [w.cget('text') for w in _walk(dlg) if isinstance(w, ttk.Checkbutton)]
        assert listed == [t1, t2]
        dlg.destroy()
    finally:
        root.destroy()


def test_copied_boolean_gate_still_combines_its_copied_operands():
    root, ed = _editor_or_skip()
    try:
        _g1, _g2, t1, t2 = _with_two_thresholds(ed)
        and_id = _make_boolean(ed, 'AND', [t1, t2], 'AND pop')
        _load(ed, 's2', _frame(6))
        ed._set_active_sample('s2')
        ed._add_gate({'kind': 'threshold', 'channel': 'CD3', 'value': 1.5,
                      'name': 'CD3 bright', 'parent_id': None})
        ed._add_gate({'kind': 'threshold', 'channel': 'CD4', 'value': -1.5,
                      'name': 'CD4 not-dim', 'parent_id': None})
        ed._set_active_sample('s1')
        assert and_id in ed._gates
        ed._open_copy_gates_dialog()                 # the dialog's own path
        dlg = _toplevel(ed, 'Copy gates')
        for w in _walk(dlg):
            if isinstance(w, ttk.Checkbutton) and w.cget('text') == 's2':
                w.invoke()
        _press(dlg, 'Copy')
        s2 = ed._sample_gates['s2']
        (copied,) = [k for k, g in s2.items() if g.get('name') == 'AND pop']
        df2 = ed._samples['s2'].data
        want = (df2.CD3.values > 0) & (df2.CD4.values > 0)
        assert (_mask(ed, 's2', copied) == want).all()
    finally:
        root.destroy()


def test_template_boolean_gate_survives_add_to_existing(tmp_path):
    import tkinter.filedialog as fdlg

    import openflo.editor_template as et
    root, ed = _editor_or_skip()
    saved = (fdlg.asksaveasfilename, et.messagebox.showwarning)
    try:
        _g1, _g2, t1, t2 = _with_two_thresholds(ed)
        _make_boolean(ed, 'AND', [t1, t2], 'AND pop')
        out = tmp_path / 'tmpl.json'
        fdlg.asksaveasfilename = lambda *a, **k: str(out)
        et.messagebox.showwarning = lambda *a, **k: None
        ed._save_template()
        _load(ed, 's3', _frame(7))
        ed._set_active_sample('s3')
        ed._add_gate({'kind': 'rect', 'x_channel': 'CD3', 'y_channel': 'CD4',
                      'x0': 1.5, 'x1': 9, 'y0': -9, 'y1': 9,
                      'name': 'pre-existing A', 'parent_id': None})
        ed._add_gate({'kind': 'rect', 'x_channel': 'CD3', 'y_channel': 'CD4',
                      'x0': -9, 'x1': 9, 'y0': 1.5, 'y1': 9,
                      'name': 'pre-existing B', 'parent_id': None})
        ed._set_active_sample('s1')
        ed._ask_template_apply = lambda: (['s3'], False)       # Add to existing
        ed._apply_template_path(str(out))
        s3 = ed._sample_gates['s3']
        (b,) = [k for k, g in s3.items() if g.get('name') == 'AND pop']
        df3 = ed._samples['s3'].data
        want = (df3.CD3.values > 0) & (df3.CD4.values > 0)
        assert (_mask(ed, 's3', b) == want).all()
    finally:
        fdlg.asksaveasfilename, et.messagebox.showwarning = saved
        root.destroy()


def test_deleting_an_operand_does_not_silently_widen_the_boolean_gate():
    root, ed = _editor_or_skip()
    try:
        _g1, g2, t1, t2 = _with_two_thresholds(ed)
        and_id = _make_boolean(ed, 'AND', [t1, t2], 'AND pop')
        before = int(_mask(ed, 's1', and_id).sum())
        ed._remove_gate_cascade_in('s1', g2)
        # Acceptable: the boolean is removed with its operand, or it stops
        # admitting events (fail-closed). Not acceptable: a wider population.
        if and_id in ed._gates:
            assert int(_mask(ed, 's1', and_id).sum()) <= before
    finally:
        root.destroy()


def test_clearing_an_operand_says_the_boolean_now_selects_nothing():
    """The Delete/Clear path keeps the boolean (Ctrl+Z restores both), it
    fails closed, and the status line names it."""
    root, ed = _editor_or_skip()
    try:
        _g1, g2, t1, t2 = _with_two_thresholds(ed)
        and_id = _make_boolean(ed, 'AND', [t1, t2], 'AND pop')
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('s1', g2))
        ed._clear_selected_gate()
        assert and_id in ed._gates and g2 not in ed._gates
        assert not _mask(ed, 's1', and_id).any()
        assert "'AND pop' lost an operand" in ed.status_var.get()
    finally:
        root.destroy()


def _tree_with_boolean_under(ed, name):
    """Cells > {CD3+, CD4+, DP = AND(CD3+, CD4+)} on sample `name`: a subtree
    that carries a boolean together with its operands."""
    ed._set_active_sample(name)
    p = ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A', 'value': 3e4,
                      'name': 'Cells', 'parent_id': None})
    a = ed._add_gate({'kind': 'threshold', 'channel': 'CD3', 'value': 0.0,
                      'name': 'CD3+', 'parent_id': p})
    b = ed._add_gate({'kind': 'threshold', 'channel': 'CD4', 'value': 0.0,
                      'name': 'CD4+', 'parent_id': p})
    dp = ed._add_gate({'kind': 'boolean', 'op': 'and', 'operands': [a, b],
                       'name': 'DP', 'parent_id': p})
    return p, dp


def _target_with_own_gates(ed, name, seed=6):
    """A second sample whose own gates hold the ids g1..g3, so a copied
    operand id that was NOT remapped names one of them."""
    _load(ed, name, _frame(seed))
    ed._set_active_sample(name)
    for ch, v, nm in (('CD3', 1.5, 'own A'), ('CD4', 1.5, 'own B'),
                      ('FSC-A', 5e4, 'own C')):
        ed._add_gate({'kind': 'threshold', 'channel': ch, 'value': v,
                      'name': nm, 'parent_id': None})


def _dp_truth(df):
    return ((df['FSC-A'].values > 3e4) & (df.CD3.values > 0)
            & (df.CD4.values > 0))


def test_moving_a_boolean_with_its_operands_remaps_them():
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(5))
        _target_with_own_gates(ed, 's2')
        p, _dp = _tree_with_boolean_under(ed, 's1')
        assert ed._move_gate_to_sample('s1', p, 's2', None) is not None
        s2 = ed._sample_gates['s2']
        (dp,) = [k for k, g in s2.items() if g.get('name') == 'DP']
        assert {s2[o]['name'] for o in s2[dp]['operands']} == {'CD3+', 'CD4+'}
        assert (_mask(ed, 's2', dp) == _dp_truth(ed._samples['s2'].data)).all()
    finally:
        root.destroy()


def test_move_refuses_to_split_a_boolean_from_its_operands():
    """Operand ids mean nothing in another sample, so neither half moves:
    not an operand away from its boolean, not a boolean away from its
    operands. Copy gates to... carries both."""
    import copy
    root, ed = _editor_or_skip()
    try:
        _g1, g2, t1, t2 = _with_two_thresholds(ed)
        and_id = _make_boolean(ed, 'AND', [t1, t2], 'AND pop')
        _target_with_own_gates(ed, 's2')
        ed._set_active_sample('s1')
        before = _mask(ed, 's1', and_id)
        snapshot = copy.deepcopy(ed._sample_gates)
        for gid in (g2, and_id):
            assert ed._move_gate_to_sample('s1', gid, 's2', None) is None
            assert ed.status_var.get().startswith('Move refused')
            assert ed._sample_gates == snapshot
        assert (_mask(ed, 's1', and_id) == before).all()
    finally:
        root.destroy()


def test_paste_remaps_operands_and_refuses_to_strand_them():
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(5))
        _target_with_own_gates(ed, 's2')
        p, dp = _tree_with_boolean_under(ed, 's1')
        ed._refresh_gate_list()

        # The whole subtree, pasted onto s2: DP combines the PASTED operands.
        ed.gate_tv.selection_set(ed._gate_iid('s1', p))
        ed._on_copy()
        ed._set_active_sample('s2')
        ed.gate_tv.selection_set(())
        ed._on_paste()
        s2 = ed._sample_gates['s2']
        (new_dp,) = [k for k, g in s2.items() if g.get('name') == 'DP']
        assert {s2[o]['name'] for o in s2[new_dp]['operands']} == {'CD3+', 'CD4+'}
        assert (_mask(ed, 's2', new_dp) == _dp_truth(ed._samples['s2'].data)).all()

        # The boolean alone, back into its own sample: operands still valid.
        ed._set_active_sample('s1')
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('s1', dp))
        ed._on_copy()
        ed.gate_tv.selection_set(())
        ed._on_paste()
        s1 = ed._sample_gates['s1']
        dps = [k for k, g in s1.items() if g.get('name') == 'DP']
        assert len(dps) == 2
        assert s1[dps[0]]['operands'] == s1[dps[1]]['operands']

        # The boolean alone onto s2: its operands are not there -> refused.
        n2 = len(ed._sample_gates['s2'])
        ed._set_active_sample('s2')
        ed.gate_tv.selection_set(())
        ed._on_paste()
        assert len(ed._sample_gates['s2']) == n2
        assert ed.status_var.get().startswith('Paste refused')
    finally:
        root.destroy()


def test_session_restore_keeps_boolean_operands_after_a_deletion():
    """A session saves each gate's editor id; the restore hands out fresh
    sequential ids. With a deleted gate ahead of the operands, the saved
    operand ids shifted onto other gates."""
    import json
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(5))
        ed._set_active_sample('s1')
        ids = [ed._add_gate({'kind': 'threshold', 'channel': ch, 'value': v,
                             'name': nm, 'parent_id': None})
               for ch, v, nm in (('FSC-A', 5e4, 'temp'), ('CD3', 0.0, 'CD3+'),
                                 ('CD4', 0.0, 'CD4+'), ('SSC-A', 5e4, 'SSC hi'))]
        ed._add_gate({'kind': 'boolean', 'op': 'and', 'operands': ids[1:3],
                      'name': 'DP', 'parent_id': None})
        ed._remove_gate_cascade_in('s1', ids[0])
        saved = json.loads(json.dumps(ed._session_state()))['sample_gates']['s1']
        sample = ed._samples['s1']
    finally:
        root.destroy()
    root, ed = _editor_or_skip()
    try:
        ed._pending_sample_gates['s1'] = saved
        ed._on_loaded('s1', sample)
        gates = ed._sample_gates['s1']
        (dp,) = [k for k, g in gates.items() if g.get('name') == 'DP']
        assert [gates[o]['name'] for o in gates[dp]['operands']] == ['CD3+', 'CD4+']
        df = sample.data
        assert (_mask(ed, 's1', dp)
                == ((df.CD3.values > 0) & (df.CD4.values > 0))).all()
    finally:
        root.destroy()


def test_a_refused_paste_leaves_the_undo_history_and_active_sample_alone():
    """The paste took its undo checkpoint, and made a selected sample row the
    active sample, before checking whether it could go ahead: a refused paste
    left an undo step that undid nothing, cleared the redo history and
    switched the sample under the user."""
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(5))
        _target_with_own_gates(ed, 's2')
        _p, dp = _tree_with_boolean_under(ed, 's1')
        ed.update_idletasks()             # each gesture its own undo step
        ed._add_gate({'kind': 'threshold', 'channel': 'SSC-A', 'value': 5e4,
                      'name': 'undone', 'parent_id': None})
        ed.update_idletasks()
        ed._undo()                        # leaves one step to redo
        assert dp in ed._gates
        undo, redo = list(ed._undo_stack), list(ed._redo_stack)
        assert redo
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('s1', dp))
        ed._on_copy()
        # DP alone onto s2: its operands are not there, so it is refused.
        ed.gate_tv.selection_set(ed._sample_iid('s2'))
        ed._on_paste()
        assert ed.status_var.get().startswith('Paste refused')
        assert ed._active_sample == 's1'
        assert [id(s) for s in ed._undo_stack] == [id(s) for s in undo]
        assert [id(s) for s in ed._redo_stack] == [id(s) for s in redo]
    finally:
        root.destroy()


def test_cut_and_paste_of_an_operand_in_its_own_sample_keeps_the_boolean():
    """Cut + paste within one sample is a move. The pasted gate took a fresh
    id, so a boolean gate that combined it stayed empty (fail-closed) for
    good. Pasted back into the sample it was cut from, the gate keeps its id
    and the boolean follows it, as it follows a drag; a second paste of the
    same clip is a copy and gets a fresh id."""
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(5))
        p, dp = _tree_with_boolean_under(ed, 's1')
        (b,) = [k for k, g in ed._gates.items() if g['name'] == 'CD4+']
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('s1', b))
        ed._on_cut()
        assert b not in ed._gates and not _mask(ed, 's1', dp).any()
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('s1', p))
        ed._on_paste()
        gates = ed._sample_gates['s1']
        assert {gates[o]['name'] for o in gates[dp]['operands']} == {'CD3+', 'CD4+'}
        assert gates[b]['parent_id'] == p
        assert (_mask(ed, 's1', dp) == _dp_truth(ed._samples['s1'].data)).all()
        # Again: a copy now, with a fresh id that no gate has had.
        ed._on_paste()
        (b2,) = [k for k, g in gates.items() if g['name'] == 'CD4+' and k != b]
        assert int(b2[1:]) > max(int(k[1:]) for k in gates if k != b2)
        assert len(gates) == 5
    finally:
        root.destroy()


def test_a_gate_pasted_under_its_own_id_is_never_given_out_again():
    """An undo winds the id counter back. A gate copied, then undone, then
    pasted back keeps its id; the next new gate must not be handed that id
    too (it would overwrite the pasted gate)."""
    root, ed = _editor_or_skip()
    try:
        _g1, _g2, _t1, _t2 = _with_two_thresholds(ed)
        ed.update_idletasks()
        x = ed._add_gate({'kind': 'threshold', 'channel': 'SSC-A',
                          'value': 5e4, 'name': 'X', 'parent_id': None})
        ed.update_idletasks()
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('s1', x))
        ed._on_copy()
        ed._undo()
        assert x not in ed._gates
        ed.gate_tv.selection_set(())
        ed._on_paste()
        y = ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A',
                          'value': 5e4, 'name': 'Y', 'parent_id': None})
        assert y != x
        assert {ed._gates[x]['name'], ed._gates[y]['name']} == {'X', 'Y'}
    finally:
        root.destroy()


def test_session_restore_says_when_a_boolean_lost_an_operand():
    """A session saved after an operand was deleted restores the boolean with
    no operands (it selects nothing). The status line says so, as Clear and
    Cut do, once the load has settled."""
    import json
    root, ed = _editor_or_skip()
    try:
        _g1, g2, t1, t2 = _with_two_thresholds(ed)
        _make_boolean(ed, 'AND', [t1, t2], 'AND pop')
        ed._remove_gate_cascade_in('s1', g2)
        saved = json.loads(json.dumps(ed._session_state()))['sample_gates']['s1']
        sample = ed._samples['s1']
    finally:
        root.destroy()
    root, ed = _editor_or_skip()
    try:
        ed._pending_sample_gates['s1'] = saved
        ed._on_loaded('s1', sample)
        ed._on_load_settled()
        msg = ed.status_var.get()
        assert msg.startswith('1 sample(s) loaded')
        assert "boolean gate 'AND pop' lost an operand and now selects nothing" in msg
        ed._on_load_settled()                 # said once, not on every settle
        assert 'lost an operand' not in ed.status_var.get()
    finally:
        root.destroy()


def test_template_apply_says_when_a_boolean_lost_an_operand(tmp_path):
    import tkinter.filedialog as fdlg

    import openflo.editor_template as et
    root, ed = _editor_or_skip()
    saved = (fdlg.asksaveasfilename, et.messagebox.showwarning)
    try:
        _g1, g2, t1, t2 = _with_two_thresholds(ed)
        _make_boolean(ed, 'AND', [t1, t2], 'AND pop')
        ed._remove_gate_cascade_in('s1', g2)
        out = tmp_path / 'tmpl.json'
        fdlg.asksaveasfilename = lambda *a, **k: str(out)
        et.messagebox.showwarning = lambda *a, **k: None
        ed._save_template()
        _load(ed, 's3', _frame(7))
        _load(ed, 's4', _frame(8))
        ed._ask_template_apply = lambda: (['s3', 's4'], False)
        ed._apply_template_path(str(out))
        msg = ed.status_var.get()
        assert msg.startswith('Applied 2 gate(s)')
        assert msg.endswith("; boolean gate 'AND pop' lost an operand and now "
                            "selects nothing.")
    finally:
        fdlg.asksaveasfilename, et.messagebox.showwarning = saved
        root.destroy()


def test_template_apply_names_lost_operands_and_unconverted_ellipses(
        tmp_path, monkeypatch):
    """One template carrying both a boolean whose operand cannot be resolved
    and an unconverted FlowJo ellipse: the status line names both, since
    each selects nothing."""
    import json

    import openflo.editor_template as et
    monkeypatch.setattr(et.messagebox, 'showwarning', lambda *a, **k: None)
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(1))
        ed._set_active_sample('s1')
        tmpl = tmp_path / 'both.json'
        tmpl.write_text(json.dumps({'gates': [
            {'id': 't1', 'kind': 'threshold', 'channel': 'CD3', 'value': 0.0,
             'parent_id': None, 'name': 'CD3+'},
            {'id': 'e1', 'kind': 'flowjo_ellipse', 'x_channel': 'CD3',
             'y_channel': 'CD4', 'label': 'Blob', 'name': 'Blob',
             'parent_id': None},
            {'id': 'b1', 'kind': 'boolean', 'op': 'and',
             'operands': ['t1', 'gone'], 'parent_id': None,
             'name': 'AND pop'}]}), encoding='utf-8')
        ed._ask_template_apply = lambda: (['s1'], False)
        ed._apply_template_path(str(tmpl))
        msg = ed.status_var.get()
        assert msg.startswith('Applied 3 gate(s)'), msg
        assert ("; boolean gate 'AND pop' lost an operand and now selects "
                "nothing." in msg), msg
        assert '1 FlowJo ellipse(s) not converted (Blob)' in msg, msg
    finally:
        root.destroy()


def test_template_boolean_on_an_unconverted_ellipse_is_empty_and_named(
        tmp_path, monkeypatch):
    """A template whose boolean combines a gate under an unconverted FlowJo
    ellipse: applied, NOT(kid) selects nothing (its region is unknown, so is
    its complement), and the status line names it with the ellipse."""
    import json

    import openflo.editor_template as et
    from openflo.pipeline import cumulative_gate_mask
    monkeypatch.setattr(et.messagebox, 'showwarning', lambda *a, **k: None)
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(1))
        ed._set_active_sample('s1')
        tmpl = tmp_path / 'ph.json'
        tmpl.write_text(json.dumps({'gates': [
            {'id': 'e1', 'kind': 'flowjo_ellipse', 'x_channel': 'CD3',
             'y_channel': 'CD4', 'label': 'Blob', 'name': 'Blob',
             'parent_id': None},
            {'id': 'k1', 'kind': 'threshold', 'channel': 'CD3', 'value': 0.0,
             'parent_id': 'e1', 'name': 'kid'},
            {'id': 'b1', 'kind': 'boolean', 'op': 'not', 'operands': ['k1'],
             'parent_id': None, 'name': 'NOT kid'}]}), encoding='utf-8')
        ed._ask_template_apply = lambda: (['s1'], False)
        ed._apply_template_path(str(tmpl))
        msg = ed.status_var.get()
        assert '1 FlowJo ellipse(s) not converted (Blob)' in msg, msg
        assert "boolean gate(s) built on them: 'NOT kid'" in msg, msg
        gates = ed._sample_gates['s1']
        bid = next(gid for gid, g in gates.items() if g.get('name') == 'NOT kid')
        mask = np.asarray(cumulative_gate_mask(gates, bid, ed._samples['s1'].data))
        assert not mask.any(), f'NOT(kid) admitted {int(mask.sum())} events'
    finally:
        root.destroy()


def _paste_into(ed, name):
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(ed._sample_iid(name))
    ed._on_paste()


def _dp_over(ed, names):
    """CD3+, CD4+ and DP = AND(CD3+, CD4+) on the active sample."""
    g = [ed._add_gate({'kind': 'threshold', 'channel': ch, 'value': 0.0,
                       'name': nm, 'parent_id': None})
         for ch, nm in names]
    dp = ed._add_gate({'kind': 'boolean', 'op': 'and', 'operands': g,
                       'name': 'DP', 'parent_id': None})
    return g, dp


def test_after_an_undo_a_paste_never_relinks_a_boolean_to_another_gate():
    """Undo wound the sample's id counter back, so the next gate took the
    undone gate's id. Pasting a clip of the undone gate, which keeps its id
    as a Cut does, then made it the operand of a boolean that had used the
    newer gate: DP = AND(CD3+, A), where DP had lost CD4+ and selects
    nothing."""
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(5))
        ed._set_active_sample('s1')
        cd3 = ed._add_gate({'kind': 'threshold', 'channel': 'CD3',
                            'value': 0.0, 'name': 'CD3+', 'parent_id': None})
        ed.update_idletasks()
        a = ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A',
                          'value': 5e4, 'name': 'A', 'parent_id': None})
        ed.update_idletasks()
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('s1', a))
        ed._on_copy()
        ed._undo()
        ed.update_idletasks()
        cd4 = ed._add_gate({'kind': 'threshold', 'channel': 'CD4',
                            'value': 0.0, 'name': 'CD4+', 'parent_id': None})
        dp = ed._add_gate({'kind': 'boolean', 'op': 'and',
                           'operands': [cd3, cd4], 'name': 'DP',
                           'parent_id': None})
        ed._remove_gate_cascade_in('s1', cd4)
        _paste_into(ed, 's1')
        assert not _mask(ed, 's1', dp).any()
        assert cd4 != a and ed._sample_gates['s1'][a]['name'] == 'A'
    finally:
        root.destroy()


def test_after_a_template_overwrite_a_paste_never_relinks_a_boolean(tmp_path):
    """'Overwrite' restarted the sample's ids at g1, so the template's gates
    took the ids of the gates it replaced, and a clip of one of those, pasted
    back under its id, became the operand of the template's boolean."""
    import tkinter.filedialog as fdlg
    root, ed = _editor_or_skip()
    saved = fdlg.asksaveasfilename
    try:
        _load(ed, 's0', _frame(4))
        ed._set_active_sample('s0')
        _dp_over(ed, [('CD3', 'CD3+'), ('CD4', 'CD4+')])
        out = tmp_path / 'tmpl.json'
        fdlg.asksaveasfilename = lambda *a, **k: str(out)
        ed._save_template()
        _load(ed, 's1', _frame(5))
        ed._set_active_sample('s1')
        a = ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A',
                          'value': 5e4, 'name': 'A', 'parent_id': None})
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('s1', a))
        ed._on_copy()
        ed._ask_template_apply = lambda: (['s1'], True)        # Overwrite
        ed._apply_template_path(str(out))
        gates = ed._sample_gates['s1']
        (cd3,) = [k for k, g in gates.items() if g['name'] == 'CD3+']
        (dp,) = [k for k, g in gates.items() if g['name'] == 'DP']
        ed._remove_gate_cascade_in('s1', cd3)
        _paste_into(ed, 's1')
        assert not _mask(ed, 's1', dp).any()
    finally:
        fdlg.asksaveasfilename = saved
        root.destroy()


def test_a_clip_from_a_removed_sample_is_not_pasted_under_its_old_ids():
    """A sample removed and loaded again under the same name starts its ids
    at g1 again, so a clip copied from it before named the new tree's ids."""
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(5))
        ed._set_active_sample('s1')
        a = ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A',
                          'value': 5e4, 'name': 'A', 'parent_id': None})
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('s1', a))
        ed._on_copy()
        ed._remove_samples(['s1'])
        _load(ed, 's1', _frame(5))
        ed._set_active_sample('s1')
        (cd3, _cd4), dp = _dp_over(ed, [('CD3', 'CD3+'), ('CD4', 'CD4+')])
        assert cd3 == a                          # the same id, another gate
        ed._remove_gate_cascade_in('s1', cd3)
        _paste_into(ed, 's1')
        assert not _mask(ed, 's1', dp).any()
    finally:
        root.destroy()
