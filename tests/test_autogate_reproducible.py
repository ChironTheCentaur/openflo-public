"""Auto-gate proposals must not depend on the display subsample.

`_get_df` defaults to `downsample=True`, which caps the frame at the SMALLEST
loaded sample's size and draws it with
``random_state = hash((name, x, y, cap)) & 0xFFFFFFFF``. Python randomises
`str` hashing per process, so fitting a gate on that frame made the proposal
depend on (a) what else happened to be loaded and (b) which process you were
in — the same file gave a different gate on every app restart.

These tests pin the property (same data in, same gate out; a small unrelated
sample must not move it), not merely the call signature.
"""
import os
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tests.conftest import gui_unavailable

COLS = ('FSC-A', 'FSC-H', 'SSC-A', 'CD3')


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


def _add(ed, name, n, seed=0):
    """A sample with a real singlet diagonal plus doublets."""
    rng = np.random.default_rng(seed)
    n_dbl = int(n * 0.1)
    a = np.concatenate([rng.normal(60000, 8000, n - n_dbl),
                        rng.normal(118000, 12000, n_dbl)])
    h = np.concatenate([
        rng.normal(60000, 8000, n - n_dbl) / 2.0,
        rng.normal(118000, 12000, n_dbl) / 3.1])
    df = pd.DataFrame({'FSC-A': a, 'FSC-H': h,
                       'SSC-A': rng.normal(50000, 9000, n),
                       'CD3': rng.normal(500, 100, n)})
    ed._samples[name] = SimpleNamespace(
        name=name, path=rf'C:\e\{name}.fcs', data=df,
        fluor_channels=['CD3'], channel_labels={c: c for c in COLS})
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._sample_gates.setdefault(name, {})
    return df


def test_autogate_frame_is_the_full_sample():
    """The fit must see every event, not the display draw."""
    root, ed = _editor_or_skip()
    try:
        big = _add(ed, 'big', 40000)
        ed.ds_display_var.set(True)              # the default
        shown = ed._get_df('big', 'FSC-A', 'FSC-H')
        full = ed._get_df('big', 'FSC-A', 'FSC-H', downsample=False)
        assert len(full) == len(big), 'downsample=False must return everything'
        assert len(shown) <= len(full)
    finally:
        root.destroy()


def test_a_small_unrelated_sample_does_not_change_the_fit_frame():
    """Loading a 2k comp control must not shrink a 40k sample's fit to 2k."""
    root, ed = _editor_or_skip()
    try:
        big = _add(ed, 'big', 40000)
        before = len(ed._get_df('big', 'FSC-A', 'FSC-H', downsample=False))
        _add(ed, 'tiny_comp', 2000, seed=9)      # a small control alongside
        after = len(ed._get_df('big', 'FSC-A', 'FSC-H', downsample=False))
        assert before == after == len(big), (
            'the fit frame shrank because another sample was loaded — '
            'auto-gate would fit the big sample on the control\'s event count')
    finally:
        root.destroy()


def test_singlet_gate_is_identical_across_repeated_runs():
    """Same data in, same gate out — within a process."""
    root, ed = _editor_or_skip()
    try:
        _add(ed, 'big', 20000)
        ed._active_sample = 'big'
        ed._gates = ed._sample_gates['big']
        opts = {'area': 'FSC-A', 'height': 'FSC-H', 'k': 3.0}
        verts = []
        for _ in range(3):
            ed._sample_gates['big'].clear()
            ed._auto_gate_singlet('big', opts)
            got = [g for g in ed._sample_gates['big'].values()
                   if g.get('kind') == 'polygon']
            assert got, 'no singlet gate proposed'
            verts.append(np.asarray(got[0]['vertices'], dtype=float))
        assert np.allclose(verts[0], verts[1]) and np.allclose(verts[0],
                                                               verts[2]), \
            'repeated auto-gate runs proposed different gates'
    finally:
        root.destroy()


def test_autogate_calls_do_not_use_the_display_subsample():
    """Guard the call sites themselves: a reintroduced default would restore
    the per-process-random fit that this module exists to prevent."""
    import inspect

    from openflo import editor_autogate
    for fn in (editor_autogate.AutoGateMixin._auto_gate_singlet,
               editor_autogate.AutoGateMixin._auto_gate_gmm,
               editor_autogate.AutoGateMixin._auto_gate_threshold):
        src = inspect.getsource(fn)
        assert '_get_df(' in src, f'{fn.__name__} no longer fetches a frame'
        call = src.split('_get_df(', 1)[1].split(')', 1)[0]
        assert 'downsample=False' in call, (
            f'{fn.__name__} fits on the display-downsampled frame again; that '
            'frame is capped by the smallest loaded sample and seeded with a '
            'per-process-random hash')


def _thr_gate(ed, name, gid, ch='CD3', val=500.0, label='Lymphocytes'):
    ed._sample_gates[name][gid] = {
        'id': gid, 'kind': 'threshold', 'parent_id': None, 'channel': ch,
        'op': '>', 'value': val, 'name': label, 'enabled': True,
        'color': '#ff0000'}
    ed._sample_gate_order[name].append(gid)
    ed._sample_gate_seq[name] = max(ed._sample_gate_seq[name],
                                    int(gid.lstrip('g') or 0))


def test_fit_domain_equals_applied_domain_in_filter_mode():
    """A gate proposed inside a filtered population must be ATTACHED to that
    population — otherwise it was fit on lymphocytes and applied to everything.

    Filter mode sets `apply_gates_var`, so `_get_df('auto')` returns the union
    of every ENABLED gate's chain while `_add_gate` parents on the SELECTED
    row. Auto-gate now resolves the parent once and fits on exactly it.
    """
    root, ed = _editor_or_skip()
    try:
        _add(ed, 'big', 6000)
        ed._sample_gate_order.setdefault('big', [])
        ed._sample_gate_seq.setdefault('big', 0)
        _thr_gate(ed, 'big', 'g1')   # CD3 > 500 ~ the median: a real split
        ed._active_sample = 'big'
        ed._gates = ed._sample_gates['big']
        ed._gate_id_order = list(ed._sample_gate_order['big'])
        ed._gate_id_seq = ed._sample_gate_seq['big']

        # The explicit population must be the gate's chain, not the enabled
        # union and not the whole sample.
        ed.apply_gates_var.set(True)
        whole = ed._get_df('big', 'FSC-A', 'FSC-H',
                           downsample=False, gate_parent=None)
        inside = ed._get_df('big', 'FSC-A', 'FSC-H',
                            downsample=False, gate_parent='g1')
        assert len(inside) < len(whole), 'gate_parent did not restrict the frame'

        # And the proposal lands under that same parent.
        ed._gate_all_var.set(False)
        ed._selected_gate_id = lambda: 'g1'
        ed._run_auto_gate({'method': 'singlet', 'area': 'FSC-A',
                           'height': 'FSC-H', 'k': 3.0})
        new = [g for g in ed._sample_gates['big'].values()
               if g.get('name') == 'Singlets']
        assert new, 'no singlet gate proposed'
        assert new[0]['parent_id'] == 'g1', (
            'the gate was fit inside g1 but attached elsewhere — it now '
            'selects events the fit never saw')
    finally:
        root.destroy()
