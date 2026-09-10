"""A replicated gate must measure the SAME population on every sample.

`_add_gate_multi` fans a new gate out to every displayed sample (the "→ all
shown" checkbox, default ON). It used to force `parent_id = None` on every
replica while the active sample's copy kept the selected parent — so with a
parent gate selected, sample A got `Singlets` nested under `Lymphocytes` and
samples B…N got `Singlets` at root. One gate name then measured a different
population per sample, which silently invalidates cross-sample frequency
comparison — the thing the fan-out exists to enable.

Replicas are now re-parented by population PATH, and a target that genuinely
lacks that population is reported instead of being quietly rooted.
"""
import os
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from openflo.gating import population_path  # noqa: E402
from tests.conftest import gui_unavailable

COLS = ('FSC-A', 'SSC-A', 'CD3', 'CD4')


def _editor_or_skip():
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
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    return root, ed


def _add(ed, name):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.random(300) for c in COLS})
    ed._samples[name] = SimpleNamespace(
        name=name, path=rf'C:\e\{name}.fcs', data=df,
        fluor_channels=['CD3', 'CD4'],
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


def _seed_parent(ed, name, gid, label='Lymphocytes'):
    """Give a sample a root parent gate with a known label."""
    ed._sample_gates[name][gid] = {
        'id': gid, 'kind': 'threshold', 'parent_id': None, 'channel': 'CD3',
        'op': '>', 'value': 0.5, 'name': label, 'enabled': True,
        'color': '#ff0000'}
    ed._sample_gate_order[name].append(gid)
    ed._sample_gate_seq[name] = max(ed._sample_gate_seq[name],
                                    int(gid.lstrip('g') or 0))


def _activate(ed, name):
    """Make `name` active and sync the active-sample shortcuts.

    The editor keeps per-sample stores (_sample_gates/_order/_seq) alongside
    active-sample shortcuts (_gates/_gate_id_order/_gate_id_seq). Setting only
    the former leaves _next_gate_id() re-issuing ids that already exist, which
    silently overwrites a gate and can make it its own parent.
    """
    ed._active_sample = name
    ed._gates = ed._sample_gates[name]
    ed._gate_id_order = list(ed._sample_gate_order[name])
    ed._gate_id_seq = ed._sample_gate_seq[name]


def test_replica_is_nested_under_the_same_population():
    """The whole point: one gate name, one population, every sample."""
    root, ed = _editor_or_skip()
    try:
        for nm in ('a', 'b', 'c'):
            _add(ed, nm)
            _seed_parent(ed, nm, 'g1')
        _activate(ed, 'a')
        ed._gate_all_var.set(True)

        gid = ed._add_gate_multi(
            {'kind': 'threshold', 'channel': 'CD4', 'op': '>', 'value': 0.5,
             'name': 'Singlets'}, parent_id='g1', audit=False)

        origin = population_path(ed._sample_gates['a'], gid)
        assert origin.endswith('Lymphocytes/Singlets'), origin
        for nm in ('b', 'c'):
            new = [g for g in ed._sample_gates[nm].values()
                   if g.get('name') == 'Singlets']
            assert new, f'{nm} did not receive the replica'
            path = population_path(ed._sample_gates[nm], new[0]['id'])
            assert path == origin, (
                f'{nm} measures {path!r} while the original measures '
                f'{origin!r} — same name, different population')
    finally:
        root.destroy()


def test_missing_parent_is_reported_not_silent():
    """A target genuinely lacking the population still gets a root gate, but
    the user is told — the old behaviour was to do it silently."""
    root, ed = _editor_or_skip()
    try:
        _add(ed, 'a')
        _seed_parent(ed, 'a', 'g1')
        _add(ed, 'b')                      # b has NO Lymphocytes gate
        _activate(ed, 'a')
        ed._gate_all_var.set(True)

        ed._add_gate_multi(
            {'kind': 'threshold', 'channel': 'CD4', 'op': '>', 'value': 0.5,
             'name': 'Singlets'}, parent_id='g1', audit=False)

        new = [g for g in ed._sample_gates['b'].values()
               if g.get('name') == 'Singlets']
        assert new and new[0]['parent_id'] is None, 'expected a root fallback'
        status = ed.status_var.get()
        assert 'Lymphocytes' in status and 'different population' in status, (
            f'the rooted fallback was not reported: {status!r}')
    finally:
        root.destroy()


def test_root_level_gate_still_replicates_to_root():
    """No parent selected → nothing to match, and no warning."""
    root, ed = _editor_or_skip()
    try:
        for nm in ('a', 'b'):
            _add(ed, nm)
        _activate(ed, 'a')
        ed._gate_all_var.set(True)

        ed._add_gate_multi(
            {'kind': 'threshold', 'channel': 'CD4', 'op': '>', 'value': 0.5,
             'name': 'Live'}, audit=False)

        for nm in ('a', 'b'):
            g = [x for x in ed._sample_gates[nm].values()
                 if x.get('name') == 'Live']
            assert g and g[0]['parent_id'] is None
        assert 'different population' not in ed.status_var.get()
    finally:
        root.destroy()
