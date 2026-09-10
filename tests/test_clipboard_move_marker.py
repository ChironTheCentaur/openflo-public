"""A cross-instance move must not delete what the destination never took.

`_paste_samples_transfer` used to write the `<move_id>.done` marker the moment
the FCS loads were *queued*. The source's `_watch_pending_move` treats that
marker as authority to delete its copies — so any path the destination skipped
or failed to load still cost the source its sample AND its gate tree, while the
destination applied nothing. The most common trigger was benign: sending a
sample to a window that already had that .fcs open, which `_queue_fcs_loads`
skips as "already loaded".

The marker is now written only once every accepted path has landed, and lists
the PATHS actually taken (names are per-instance — collision disambiguation
renames them per window). The source removes only those.
"""
import json
import os
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tests.conftest import gui_unavailable

COLS = ('FSC-A', 'SSC-A', 'CD3')


def _editor_or_skip(tmp_path):
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable('tkinter not available')
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:                        # noqa: BLE001
        pytest.skip(str(e))
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    ed._transfer_dir = lambda _d=str(tmp_path): _d      # isolate the marker dir
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    return root, ed


def _fcs(tmp_path, name):
    p = tmp_path / f'{name}.fcs'
    p.write_bytes(b'not-a-real-fcs')                    # only its path matters
    return str(p)


def _add(ed, name, path):
    df = pd.DataFrame({c: np.linspace(0, 1, 50) for c in COLS})
    ed._samples[name] = SimpleNamespace(
        name=name, path=path, data=df, fluor_channels=['CD3'],
        channel_labels={c: c for c in COLS})
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._sample_gates.setdefault(name, {})
    ed._sample_gate_order.setdefault(name, [])
    ed._sample_gate_seq.setdefault(name, 0)
    ed._name_to_path[name] = path
    ed._path_to_name[os.path.normcase(os.path.abspath(path))] = name


def _bundle(paths, move_id='mv1', pid=-1):
    return {'move_id': move_id, 'move_pid': pid,
            'samples': [{'name': os.path.basename(p).rsplit('.', 1)[0],
                         'path': p, 'gates': [], 'trial': 'T'}
                        for p in paths]}


def _marker(tmp_path, move_id='mv1'):
    f = tmp_path / f'{move_id}.done'
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else None


def test_no_marker_when_the_destination_already_has_the_sample(tmp_path):
    """The bug: the source deleted a sample the destination skipped."""
    root, ed = _editor_or_skip(tmp_path)
    try:
        path = _fcs(tmp_path, 'S1')
        _add(ed, 'S1', path)                    # destination ALREADY has it
        queued = []
        ed._queue_fcs_loads = lambda ps: queued.extend(ps)
        ed._read_transfer_bundle = lambda: _bundle([path])

        ed._paste_samples_transfer()

        assert queued == [], 'should not queue an already-loaded sample'
        assert _marker(tmp_path) is None, (
            'wrote a completion marker for a sample it did not take — the '
            'source would delete its copy and its gate tree')
        assert 'Already open here' in ed.status_var.get()
    finally:
        root.destroy()


def test_marker_waits_for_the_load_to_land(tmp_path):
    root, ed = _editor_or_skip(tmp_path)
    try:
        path = _fcs(tmp_path, 'S2')
        ed._queue_fcs_loads = lambda ps: None    # never completes
        ed._read_transfer_bundle = lambda: _bundle([path])

        ed._paste_samples_transfer()
        assert _marker(tmp_path) is None, 'marker written before the load landed'

        pkey = os.path.normcase(os.path.abspath(path))
        ed._note_inbound_landed(pkey, ok=True)
        assert _marker(tmp_path) == {'taken': [pkey]}
    finally:
        root.destroy()


def test_failed_load_is_not_reported_as_taken(tmp_path):
    root, ed = _editor_or_skip(tmp_path)
    try:
        p1, p2 = _fcs(tmp_path, 'A'), _fcs(tmp_path, 'B')
        ed._queue_fcs_loads = lambda ps: None
        ed._read_transfer_bundle = lambda: _bundle([p1, p2])
        ed._paste_samples_transfer()

        k1 = os.path.normcase(os.path.abspath(p1))
        k2 = os.path.normcase(os.path.abspath(p2))
        ed._note_inbound_landed(k1, ok=True)
        assert _marker(tmp_path) is None, 'marker written before all resolved'
        ed._note_inbound_landed(k2, ok=False)          # B failed to load
        assert _marker(tmp_path) == {'taken': [k1]}, 'B must not be claimed'
    finally:
        root.destroy()


def test_source_removes_only_confirmed_samples(tmp_path):
    """Two staged, one taken: the other must survive."""
    root, ed = _editor_or_skip(tmp_path)
    try:
        pa, pb = _fcs(tmp_path, 'A'), _fcs(tmp_path, 'B')
        _add(ed, 'A', pa)
        _add(ed, 'B', pb)
        ka = os.path.normcase(os.path.abspath(pa))
        ed._pending_move = {'id': 'mv1', 'names': ['A', 'B'], 'clip': 'x'}
        (tmp_path / 'mv1.done').write_text(json.dumps({'taken': [ka]}),
                                           encoding='utf-8')

        ed._watch_pending_move()

        assert 'A' not in ed._samples, 'the taken sample should be removed'
        assert 'B' in ed._samples, (
            'B was never taken by the destination but was deleted anyway')
        assert 'kept here' in ed.status_var.get()
    finally:
        root.destroy()


def test_unreadable_marker_keeps_everything(tmp_path):
    """Deleting on ambiguous evidence is what caused the data loss."""
    root, ed = _editor_or_skip(tmp_path)
    try:
        pa = _fcs(tmp_path, 'A')
        _add(ed, 'A', pa)
        ed._pending_move = {'id': 'mv2', 'names': ['A'], 'clip': 'x'}
        (tmp_path / 'mv2.done').write_text('not json', encoding='utf-8')

        ed._watch_pending_move()

        assert 'A' in ed._samples, 'removed a sample on an unreadable marker'
    finally:
        root.destroy()


def test_second_send_supersedes_the_first(tmp_path):
    """A second Send must clear the first batch's ✄ flags and stop its poll.

    `_mark_pending_move` only ADDS to `_pending_move_names`, and both clearing
    paths clear `pm['names']` — which by then is the second batch. So the first
    batch's rows kept the pending-move indicator for the rest of the session,
    with two 700 ms poll loops running in parallel.
    """
    root, ed = _editor_or_skip(tmp_path)
    try:
        pa, pb = _fcs(tmp_path, 'A'), _fcs(tmp_path, 'B')
        _add(ed, 'A', pa)
        _add(ed, 'B', pb)

        ed._send_samples_transfer(['A'])
        assert 'A' in ed._pending_move_names
        first_poll = ed._pending_move_poll
        assert first_poll is not None

        ed._send_samples_transfer(['B'])
        assert 'B' in ed._pending_move_names
        assert 'A' not in ed._pending_move_names, (
            "the first batch's pending-move flag was stranded — its rows keep "
            'the ✄ indicator with no way to clear it')
        assert ed._pending_move['names'] == ['B']
        assert ed._pending_move_poll != first_poll, 'old poll loop still armed'
    finally:
        root.destroy()
