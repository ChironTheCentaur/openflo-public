"""A reopened session gives each sample back its cell-cycle and FlowSOM models.

Both analyses write label columns (``cell_cycle``; ``flowsom`` /
``flowsom_meta``) and keep a model on the sample (``cell_cycle_result``,
``flowsom_result``). The session saved the columns, in the processed sidecar,
but not the models: after a reopen the phases and metaclusters were still
there, yet the Cell cycle window said "No cell-cycle result for this sample"
and Analyze > SOM tree said "Run FlowSOM first". The models now go into the
sample's session entry (session_models) and come back equal to the ones
saved, checked against the columns they label.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tests.test_session_restore_transforms import _restore
from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
    settle,
)

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'Time']
SPILL = '3,FL1-A,FL2-A,FL3-A,1,0.1,0.02,0.05,1,0.1,0.01,0.03,1'


def _fcs(path, seed, n=2000, dna=False, spill=True):
    """Two bright/dim populations on FL1-A and FL2-A; with ``dna``, FL3-A is
    a DNA-content profile (G1 peak, G2/M at twice it, S between)."""
    import flowio
    rng = np.random.default_rng(seed)
    fl1 = rng.choice([150., 20000.], n) + rng.normal(0, 40, n)
    fl2 = rng.choice([150., 20000.], n) + rng.normal(0, 40, n)
    if dna:
        ph = rng.choice([0, 1, 2], n, p=[.6, .15, .25])
        fl3 = np.where(ph == 0, rng.normal(50000, 2000, n),
                       np.where(ph == 2, rng.normal(100000, 3500, n),
                                rng.uniform(55000, 95000, n)))
    else:
        fl3 = rng.normal(800, 100, n)
    ev = np.column_stack([rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
                          fl1, fl2, fl3, np.sort(rng.uniform(0, 1000, n))])
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).flatten().tolist(), CH,
                          metadata_dict={'SPILL': SPILL} if spill else {})
    return str(path)


def _loaded(path):
    from openflo.pipeline import FlowSample
    s = FlowSample(path)            # what editor_loadpool._load_worker does
    s.auto_compensate()
    s.apply_transform()
    return s


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)       # never a real (blocking) modal
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, _new_editor(root), tmp_path
    root.destroy()


def _reopen(root, path):
    ed2 = _restore(root, path, raw_first=True)
    settle(root, 1.0)               # let the load settle / restore run
    return ed2


def _same(a, b, where='model'):
    """Exact equality, type for type: arrays by dtype, shape and value, a NaN
    only to a NaN."""
    assert type(a) is type(b), (where, type(a), type(b))
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, where
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'), where
    elif isinstance(a, dict):
        assert list(a) == list(b), (where, list(a), list(b))
        for k in a:
            _same(a[k], b[k], f'{where}.{k}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f'{where}[{i}]')
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b), where
    else:
        assert a == b, (where, a, b)


def _label_texts(widget):
    from tkinter import ttk
    out = []
    for w in widget.winfo_children():
        if isinstance(w, ttk.Label):
            out.append(str(w.cget('text')))
        out.extend(_label_texts(w))
    return out


def _cell_cycle(ed, name, method='boundaries'):
    s = ed._samples[name]
    found = s.cell_cycle_model(dna_channel='FL3-A', method=method)
    ed._finish_cell_cycle('FL3-A', {name: s.data}, {name: found}, [])
    return s


def _entry(data, name):
    return next(e for e in data['samples'] if e['name'] == name)


@pytest.mark.parametrize('method', ['boundaries', 'dean-jett-fox'])
def test_cell_cycle_model_comes_back_equal(env, method):
    """The model came back None, so the Cell cycle window had nothing to
    draw; now it is the saved model, and the window shows the same numbers."""
    from openflo.ui_cell_cycle import CellCycleWindow
    root, ed, tmp = env
    ed._on_loaded('dna', _loaded(_fcs(tmp / 'dna.fcs', 3, n=4000, dna=True)))
    s = _cell_cycle(ed, 'dna', method)
    assert s.cell_cycle_result and s.cell_cycle_result.get('ok')
    if method == 'dean-jett-fox':
        assert s.cell_cycle_result.get('djf_params')
    before = _label_texts(CellCycleWindow(ed, 'dna'))
    path = tmp / 'cc.flowsession'
    data = ed._write_session(str(path))
    assert _entry(data, 'dna').get('processed_data')
    ed2 = _reopen(root, path)
    s2 = ed2._samples['dna']
    assert 'cell_cycle' in s2.data.columns
    _same(s.cell_cycle_result, s2.cell_cycle_result)
    after = _label_texts(CellCycleWindow(ed2, 'dna'))
    assert after == before
    assert not any('No cell-cycle result' in t for t in after)


def test_flowsom_model_comes_back_equal(env):
    """flowsom_result came back None, so Analyze > SOM tree refused to open;
    now the tree opens on the saved model and draws the same nodes, counts
    and metaclusters."""
    from openflo.ui_flowsom_tree import FlowSOMTreeWindow
    root, ed, tmp = env
    ed._on_loaded('fs', _loaded(_fcs(tmp / 'fs.fcs', 4)))
    s = ed._samples['fs']
    s.run_flowsom(channels=['FL1-A', 'FL2-A'], grid=(4, 4), n_metaclusters=3)
    assert s.flowsom_result
    live = FlowSOMTreeWindow(ed, 'fs')
    path = tmp / 'fs.flowsession'
    data = ed._write_session(str(path))
    rec = _entry(data, 'fs')['flowsom_model']
    # The per-event node / metacluster ids are in the sidecar's columns, not
    # written again in the record: it holds the 16 x 2 prototypes and map.
    assert len(json.dumps(rec)) < 4000
    ed2 = _reopen(root, path)
    s2 = ed2._samples['fs']
    assert 'flowsom_meta' in s2.data.columns
    _same(s.flowsom_result, s2.flowsom_result)
    ed2._set_active_sample('fs')
    ed2._open_flowsom_tree()            # the menu command, as the user opens it
    trees = [w for w in ed2.winfo_children()
             if isinstance(w, FlowSOMTreeWindow)]
    assert len(trees) == 1, ed2.status_var.get()
    tree = trees[0]
    assert np.array_equal(tree._W, live._W)
    assert tree._channels == live._channels and tree._n_meta == live._n_meta
    assert np.array_equal(tree._counts, live._counts)
    assert np.array_equal(tree._node_meta, live._node_meta)


def test_a_cleared_or_orphaned_model_is_not_saved(env):
    """A save writes a model only while its label columns are in the data,
    and never brings back one the user cleared: a reopened sample whose
    re-run found no DNA peaks stays without a model through the next save."""
    root, ed, tmp = env
    ed._on_loaded('dna', _loaded(_fcs(tmp / 'dna.fcs', 3, n=4000, dna=True)))
    _cell_cycle(ed, 'dna')
    s = ed._samples['dna']
    s.run_flowsom(channels=['FL1-A', 'FL2-A'], grid=(4, 4), n_metaclusters=3)
    path = tmp / 'a.flowsession'
    ed._write_session(str(path))
    ed2 = _reopen(root, path)
    s2 = ed2._samples['dna']
    assert s2.cell_cycle_result and s2.flowsom_result
    # The re-run finds nothing: the model is cleared (its old column stays).
    ed2._finish_cell_cycle('FL3-A', {'dna': s2.data}, {'dna': None}, [])
    assert s2.cell_cycle_result is None
    # The FlowSOM columns go (as a rebuild from the FCS drops them).
    s2.data = s2.data.drop(columns=['flowsom', 'flowsom_meta'])
    path2 = tmp / 'b.flowsession'
    entry = _entry(ed2._write_session(str(path2)), 'dna')
    assert 'cell_cycle_model' not in entry
    assert 'flowsom_model' not in entry
    ed3 = _reopen(root, path2)
    s3 = ed3._samples['dna']
    assert s3.cell_cycle_result is None and s3.flowsom_result is None


def test_a_rebuild_from_the_fcs_drops_the_models_with_their_columns(env):
    """Compensating a sidecar-restored sample rebuilds it from its FCS, which
    has no label columns; the models went on pointing at columns that were
    gone (the SOM tree then failed on a missing 'flowsom'). They go too, so
    the next save has nothing stale to write."""
    root, ed, tmp = env
    ed._on_loaded('fs', _loaded(_fcs(tmp / 'fs.fcs', 4, dna=True)))
    _cell_cycle(ed, 'fs')
    ed._samples['fs'].run_flowsom(channels=['FL1-A', 'FL2-A'], grid=(4, 4),
                                  n_metaclusters=3)
    path = tmp / 'r.flowsession'
    ed._write_session(str(path))
    ed2 = _reopen(root, path)
    s2 = ed2._samples['fs']
    assert s2.flowsom_result and s2.cell_cycle_result
    assert getattr(s2, 'source_fcs', None)
    applied, why = ed2._comp_apply_by_reload(
        'fs', s2, list(s2.comp_channels), np.asarray(s2.comp_matrix))
    assert applied, why
    assert 'flowsom' not in s2.data.columns
    assert s2.flowsom_result is None and s2.cell_cycle_result is None
    ed2._set_active_sample('fs')
    ed2._open_flowsom_tree()
    assert 'Run FlowSOM first' in ed2.status_var.get()
    entry = _entry(ed2._write_session(str(tmp / 'r2.flowsession')), 'fs')
    assert 'flowsom_model' not in entry and 'cell_cycle_model' not in entry


@pytest.mark.parametrize('damage', [
    'other_version', 'not_a_record', 'wrong_shape', 'missing_keys',
    'unknown_dtype', 'missing_channel'])
def test_a_malformed_record_is_ignored_with_a_log_line(env, capsys, damage):
    """An old or damaged model record is left out, with a log line, and the
    rest of the sample (its columns) restores as before."""
    root, ed, tmp = env
    ed._on_loaded('dna', _loaded(_fcs(tmp / 'dna.fcs', 3, n=4000, dna=True)))
    _cell_cycle(ed, 'dna')
    ed._samples['dna'].run_flowsom(channels=['FL1-A', 'FL2-A'], grid=(4, 4),
                                   n_metaclusters=3)
    path = tmp / 'm.flowsession'
    ed._write_session(str(path))
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    e = _entry(data, 'dna')
    cc, fs = e['cell_cycle_model'], e['flowsom_model']
    if damage == 'other_version':
        cc['version'] = fs['version'] = 99
    elif damage == 'not_a_record':
        e['cell_cycle_model'] = e['flowsom_model'] = 'G1 50%'
    elif damage == 'wrong_shape':
        cc['model']['g1_mean'] = 'high'
        fs['model']['weights']['$ndarray'] = [[1.0, 2.0]]
    elif damage == 'missing_keys':
        del cc['model']['pct_s']
        del fs['model']['meta_of_node']
    elif damage == 'unknown_dtype':
        cc['model'] = {'$ndarray': [1], 'dtype': '|O'}
        fs['model']['weights']['dtype'] = 'not-a-dtype'
    elif damage == 'missing_channel':
        cc['model']['channel'] = 'FL9-A'
        fs['model']['grid'] = {'$tuple': [2, 2]}
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)
    capsys.readouterr()
    ed2 = _reopen(root, path)
    s2 = ed2._samples['dna']
    assert {'cell_cycle', 'flowsom', 'flowsom_meta'} <= set(s2.data.columns)
    assert s2.cell_cycle_result is None and s2.flowsom_result is None
    log = capsys.readouterr().out
    assert 'cell_cycle_model' in log and 'flowsom_model' in log, log
