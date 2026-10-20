"""Gate templates: Save template -> Apply template, through the editor.

Coverage before: TemplateMixin._apply_template_path ran 1/42 statements,
_apply_template_to_sample 1/22, ExportMixin._save_template 1/35.

Ground truth: the source sample's gate tree. Applied to another sample it
must reproduce the same kinds, geometry and parent structure, so each
population on the target equals the source definition evaluated on the
target's events. 'Overwrite' replaces the target's tree; 'Add to existing'
keeps it.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tests.conftest import gui_unavailable


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable("tkinter not available — headless environment")
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


def _load(ed, name, seed, cols=('FSC-A', 'SSC-A', 'CD3', 'CD4')):
    rng = np.random.default_rng(seed)
    n = 5000
    df = pd.DataFrame({'FSC-A': rng.uniform(0, 1e5, n), 'SSC-A': rng.uniform(0, 1e5, n),
                       'CD3': rng.normal(0, 1, n), 'CD4': rng.normal(0, 1, n)})[list(cols)]
    ed._samples[name] = SimpleNamespace(
        name=name, path=rf'C:\exp\{name}.fcs', data=df,
        fluor_channels=[c for c in cols if c.startswith('CD')],
        channel_labels={c: c for c in cols})
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(cols)
    ed._channel_labels = {c: c for c in cols}
    ed._populate_channel_combos()


_TREE = [  # (name, gate, parent name)
    ('Cells', {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
               'x0': 1e4, 'x1': 9e4, 'y0': 1e4, 'y1': 9e4}, None),
    ('Blob', {'kind': 'ellipsoid', 'x_channel': 'CD3', 'y_channel': 'CD4',
              'mean': [0.5, 0.2], 'cov': [[1.0, 0.3], [0.3, 0.6]],
              'distance_sq': 1.5}, 'Cells'),
    ('Tri', {'kind': 'polygon', 'x_channel': 'CD3', 'y_channel': 'CD4',
             'vertices': [[-1, -1], [2, -0.5], [0, 2]]}, 'Blob'),
    ('CD3hi', {'kind': 'interval', 'channel': 'CD3', 'lo': 0.0, 'hi': 3.0}, None),
]


def _build_source(ed):
    ed._set_active_sample('s1')
    ids = {}
    for name, g, parent in _TREE:
        ids[name] = ed._add_gate(dict(g, name=name, parent_id=ids.get(parent)))
    return ids


def _by_name(gates):
    return {g.get('name'): (gid, g) for gid, g in gates.items()}


def _save(ed, path):
    import tkinter.filedialog as fdlg
    saved = fdlg.asksaveasfilename
    try:
        fdlg.asksaveasfilename = lambda *a, **k: str(path)
        ed._save_template()
    finally:
        fdlg.asksaveasfilename = saved


def _check_tree(ed, target, n_expected):
    import openflo.pipeline as fp
    tg = ed._sample_gates[target]
    assert len(tg) == n_expected
    named = _by_name(tg)
    src_named = _by_name(ed._sample_gates['s1'])
    df = ed._samples[target].data
    for name, g, parent in _TREE:
        gid, got = named[name]
        want_parent = named[parent][0] if parent else None
        assert got['parent_id'] == want_parent, name
        assert got['kind'] == g['kind']
        # The population equals the SOURCE definition evaluated on the target.
        src_gid = src_named[name][0]
        want = np.asarray(fp.cumulative_gate_mask(ed._sample_gates['s1'], src_gid, df))
        assert (np.asarray(fp.cumulative_gate_mask(tg, gid, df)) == want).all(), name
        assert 0 < want.sum() < len(df)


def test_template_overwrite_reproduces_the_tree(tmp_path):
    root, ed = _editor_or_skip()
    try:
        for i, nm in enumerate(('s1', 's2')):
            _load(ed, nm, seed=i)
        _build_source(ed)
        out = tmp_path / 't.json'
        _save(ed, out)
        assert out.is_file()
        ed._set_active_sample('s2')
        ed._add_gate({'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
                      'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'name': 'old', 'parent_id': None})
        ed._set_active_sample('s1')
        ed._ask_template_apply = lambda: (['s2'], True)            # Overwrite
        ed._apply_template_path(str(out))
        _check_tree(ed, 's2', len(_TREE))
        assert 'old' not in _by_name(ed._sample_gates['s2'])
        assert ed._active_sample == 's1'                          # restored
        assert 'overwrote 1 sample' in ed.status_var.get()
    finally:
        root.destroy()


def test_template_add_to_existing_keeps_target_gates_and_rewires_parents(tmp_path):
    root, ed = _editor_or_skip()
    try:
        for i, nm in enumerate(('s1', 's3')):
            _load(ed, nm, seed=i)
        _build_source(ed)
        out = tmp_path / 't.json'
        _save(ed, out)
        ed._set_active_sample('s3')
        for k in range(3):        # shifts the ids the template's gates receive
            ed._add_gate({'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
                          'x0': k, 'x1': k + 1, 'y0': 0, 'y1': 1,
                          'name': f'keep{k}', 'parent_id': None})
        ed._set_active_sample('s1')
        ed._ask_template_apply = lambda: (['s3'], False)           # Add to existing
        ed._apply_template_path(str(out))
        _check_tree(ed, 's3', 3 + len(_TREE))
        assert {'keep0', 'keep1', 'keep2'} <= set(_by_name(ed._sample_gates['s3']))
    finally:
        root.destroy()


def test_template_warns_when_target_lacks_a_channel(tmp_path):
    import openflo.editor_template as et
    root, ed = _editor_or_skip()
    shown = []
    saved = et.messagebox.showwarning
    try:
        _load(ed, 's1', seed=0)
        _load(ed, 's4', seed=4, cols=('FSC-A', 'SSC-A', 'CD3'))   # no CD4
        _build_source(ed)
        out = tmp_path / 't.json'
        _save(ed, out)
        et.messagebox.showwarning = lambda *a, **k: shown.append(a)
        ed._ask_template_apply = lambda: (['s4'], True)
        ed._apply_template_path(str(out))
        assert len(shown) == 1
        assert 's4: 2 gate(s)' in shown[0][1]                    # Blob + Tri use CD4
    finally:
        et.messagebox.showwarning = saved
        root.destroy()
