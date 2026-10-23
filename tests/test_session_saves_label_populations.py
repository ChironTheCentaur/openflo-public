"""A session with Leiden or FlowSOM populations can be saved.

The importer took each population's value straight from ``df[col].unique()``,
a numpy int64, and the session writer json.dumps the gate dicts as they are:
"Object of type int64 is not JSON serializable". Saving failed for any session
with Leiden or FlowSOM populations, and so did the close-time autosave (a
dialog) and the periodic one (silently). Present in 2.6.1 too.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.test_compensation_editor_apply import _load, _write_mixed
from tests.test_session_restore_transforms import _restore
from tests.test_transform_records import _new_editor, capture_warnings, isolate_home


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


def _counts(ed, name, col):
    from openflo.pipeline import cumulative_gate_mask
    s, gates = ed._samples[name], ed._sample_gates[name]
    return {g['value']: int(np.asarray(
                cumulative_gate_mask(gates, gid, s.data), dtype=bool).sum())
            for gid, g in gates.items()
            if g.get('kind') == 'category' and g.get('channel') == col}


@pytest.mark.parametrize('method,col', [('leiden', 'leiden'),
                                        ('flowsom', 'flowsom_meta')])
def test_a_session_with_label_populations_saves_and_restores(env, method, col):
    root, ed, tmp = env
    _write_mixed(tmp / 'a.fcs')
    _load(root, ed, [('a', tmp / 'a.fcs')])
    ed._set_active_sample('a')
    ed._run_clustering(method=method, all_samples=False, k=15, grid=4,
                       n_meta=3, embedding='none', resolution=0.3)
    counts = _counts(ed, 'a', col)
    assert len(counts) >= 2
    assert all(type(v) is int for v in counts)      # Python, not numpy
    path = tmp / f'{method}.flowsession'
    ed._write_session(str(path))
    saved = json.loads(path.read_text(encoding='utf-8'))
    values = [g['value'] for g in saved['sample_gates']['a']
              if g.get('kind') == 'category']
    assert sorted(values) == sorted(counts)
    ed2 = _restore(root, path, raw_first=True)
    assert _counts(ed2, 'a', col) == counts


def test_the_session_writer_takes_numpy_scalars_in_gates(env):
    """Belt and braces for any other numpy scalar a gate may hold."""
    root, ed, tmp = env
    _write_mixed(tmp / 'a.fcs')
    _load(root, ed, [('a', tmp / 'a.fcs')])
    ed._sample_gates['a']['g9'] = {
        'kind': 'threshold', 'channel': 'FLA-A', 'value': np.float64(0.5),
        'parent_id': None, 'name': 'T', 'enabled': np.bool_(True),
        'order_hint': np.int64(3)}
    ed._sample_gate_order['a'].append('g9')
    path = tmp / 'np.flowsession'
    ed._write_session(str(path))
    g = next(g for g in json.loads(path.read_text(encoding='utf-8'))
             ['sample_gates']['a'] if g['id'] == 'g9')
    assert (g['value'], g['enabled'], g['order_hint']) == (0.5, True, 3)
