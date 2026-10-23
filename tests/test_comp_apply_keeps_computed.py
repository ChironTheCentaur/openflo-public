"""The compensation editor's Apply keeps what was computed on the sample.

Apply used to replace the sample's data with a fresh load of its FCS. Every
computed column went with it, measured on one sample: leiden, TSNE1, TSNE2,
U:Dye and FLA-A_pos. The six Leiden populations (461, 260, 237, 182, 181,
179 events) then selected 0; a gate on TSNE1 and one on U:Dye selected all
1500 events, because a geometric gate on a missing channel admits every
event. A downsampled sample came back at its full 1500 events. The status
line said only "Applied 3x3 matrix".

Apply now re-compensates the sample's own events in place, so every event and
every other column stays. What was computed from the old compensation is
named as stale rather than recomputed.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests.test_compensation_editor_apply import (
    FL,
    M,
    _comp_window,
    _load,
    _logicle,
    _write_mixed,
)
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


def _computed(root, ed, tmp, name='a'):
    """Load an FCS without $SPILL and give it what analysis adds: Leiden
    clusters imported as populations, a t-SNE, an unmixed U: column, an FMO
    '_pos' flag, and gates on the embedding and on the U: column."""
    T = _write_mixed(tmp / f'{name}.fcs')
    _load(root, ed, [(name, tmp / f'{name}.fcs')])
    ed._set_active_sample(name)
    s = ed._samples[name]
    s.run_leiden(n_neighbors=15, resolution=0.3)
    ed._import_populations('leiden')
    n = len(s.data)
    s._store_embedding(np.column_stack([np.linspace(-1.0, 1.0, n),
                                        np.zeros(n)]),
                       np.ones(n, dtype=bool), 'TSNE')
    s.data['U:Dye'] = np.linspace(0.0, 1.0, n)
    s.apply_threshold_gates({'FLA-A': float(np.median(s.data['FLA-A']))})
    gates = ed._sample_gates[name]
    gates['tsne'] = {'kind': 'rect', 'x_channel': 'TSNE1',
                     'y_channel': 'TSNE2', 'x0': -9.0, 'x1': 0.0,
                     'y0': -9.0, 'y1': 9.0, 'parent_id': None,
                     'name': 'TSNE1 < 0', 'enabled': True}
    gates['unmixed'] = {'kind': 'threshold', 'channel': 'U:Dye', 'value': 0.5,
                        'parent_id': None, 'name': 'U:Dye > 0.5',
                        'enabled': True}
    ed._sample_gate_order[name] += ['tsne', 'unmixed']
    return s, T


def _counts(ed, name):
    from openflo.pipeline import cumulative_gate_mask
    s, gates = ed._samples[name], ed._sample_gates[name]
    return {g['name']: int(np.asarray(
                cumulative_gate_mask(gates, gid, s.data), dtype=bool).sum())
            for gid, g in gates.items() if g.get('kind') != 'group'}


def _apply(ed, matrix=M):
    win = _comp_window(ed)
    win._set_matrix(FL, matrix)
    win._apply()
    win.destroy()


def test_apply_keeps_computed_columns_and_their_populations(env):
    root, ed, tmp = env
    s, T = _computed(root, ed, tmp)
    cols, counts = list(s.data.columns), _counts(ed, 'a')
    assert sum(1 for k in counts if k.startswith('leiden')) >= 2
    _apply(ed)
    assert list(s.data.columns) == cols
    assert _counts(ed, 'a') == counts
    # ... and the compensation itself is right.
    np.testing.assert_allclose(s.data[FL].to_numpy(float), _logicle(T),
                               atol=1e-5)
    np.testing.assert_allclose(s.comp_matrix, M)
    st = ed.status_var.get()
    assert "Applied 3×3 matrix to 'a'" in st
    assert 'computed before this compensation change' in st
    for stale in ('leiden', 'TSNE', 'U:Dye', 'FLA-A_pos'):
        assert stale in st, (stale, st)
    assert 're-run' in st.lower()


def test_apply_after_a_downsample_keeps_the_events(env):
    """A fresh load is the whole file: Apply on a downsampled sample put it
    back to every event, out of step with its gates and columns."""
    root, ed, tmp = env
    s, T = _computed(root, ed, tmp)
    s.data['row'] = np.arange(len(s.data))
    s.data = s.data.sample(1000, random_state=42).reset_index(drop=True)
    counts = _counts(ed, 'a')
    _apply(ed)
    assert len(s.data) == 1000
    assert _counts(ed, 'a') == counts
    rows = s.data['row'].to_numpy()
    np.testing.assert_allclose(s.data[FL].to_numpy(float), _logicle(T)[rows],
                               atol=1e-5)


def test_a_session_saved_after_apply_keeps_the_computed_columns(env):
    root, ed, tmp = env
    s, _T = _computed(root, ed, tmp)
    _apply(ed)
    counts = _counts(ed, 'a')
    path = tmp / 'after_apply.flowsession'
    data = ed._write_session(str(path))
    assert data['samples'][0].get('processed_data'), data['samples'][0]
    ed2 = _restore(root, path, raw_first=True)
    s2 = ed2._samples['a']
    for col in ('leiden', 'TSNE1', 'TSNE2', 'U:Dye', 'FLA-A_pos'):
        assert col in s2.data.columns, col
    assert _counts(ed2, 'a') == counts


def test_a_restored_session_sample_is_recompensated_in_place(env):
    """A sample rebuilt from a session's CSV carries the matrix it was saved
    with (comp_matrix / comp_channels), so Apply replaces that compensation
    in place, computed columns and all, rather than rebuilding it from the
    FCS."""
    root, ed, tmp = env
    _computed(root, ed, tmp)
    _apply(ed, np.eye(3))                  # saved with the identity applied
    path = tmp / 'restore_me.flowsession'
    ed._write_session(str(path))
    ed2 = _restore(root, path, raw_first=True)
    ed2._set_active_sample('a')
    s2 = ed2._samples['a']
    assert s2.path.endswith('.npz') and s2.comp_channels == FL
    cols, counts = list(s2.data.columns), _counts(ed2, 'a')
    for col in ('leiden', 'TSNE1', 'U:Dye', 'FLA-A_pos'):
        assert col in cols, col
    T = _write_mixed(tmp / 'again.fcs')    # same seed: the same true signal
    _apply(ed2)
    st = ed2.status_var.get()
    assert st.startswith("Applied 3×3 matrix to 'a'."), st
    assert list(s2.data.columns) == cols
    assert _counts(ed2, 'a') == counts
    np.testing.assert_allclose(s2.data[FL].to_numpy(float), _logicle(T),
                               atol=1e-5)
    np.testing.assert_allclose(s2.comp_matrix, M)


def _csv_without_record(root, ed, tmp):
    _computed(root, ed, tmp)
    _apply(ed)
    path = tmp / 'restore_me.flowsession'
    ed._write_session(str(path))
    ed2 = _restore(root, path, raw_first=True)
    ed2._set_active_sample('a')
    s2 = ed2._samples['a']
    assert s2.path.endswith('.npz')
    # As written before sessions recorded the matrix: none on the sample.
    s2.comp_matrix, s2.comp_channels = None, []
    return ed2, s2


def test_apply_refuses_a_csv_sample_whose_compensation_is_unknown(env):
    """A sample built from a CSV that records neither the compensation in
    its values nor its raw FCS: a new matrix would compound the unknown one.
    Apply used to fail on it with "could not parse FCS"; it now says why and
    changes nothing."""
    root, ed, tmp = env
    ed2, s2 = _csv_without_record(root, ed, tmp)
    s2.source_fcs = None
    before = s2.data.copy()
    _apply(ed2)
    st = ed2.status_var.get()
    assert st.startswith("Cannot re-compensate 'a'"), st
    assert 'does not record the compensation' in st
    assert s2.data.equals(before)


def test_apply_rebuilds_from_the_fcs_a_csv_sample_without_a_record(env):
    """With no compensation record but its raw FCS known (source_fcs), Apply
    rebuilds the sample from that FCS, as master did for every sample."""
    root, ed, tmp = env
    ed2, s2 = _csv_without_record(root, ed, tmp)
    assert str(s2.source_fcs).endswith('a.fcs')
    T = _write_mixed(tmp / 'again.fcs')
    _apply(ed2)
    st = ed2.status_var.get()
    assert st.startswith("Applied 3×3 matrix to 'a' (rebuilt from a.fcs)"), st
    np.testing.assert_allclose(s2.data[FL].to_numpy(float), _logicle(T),
                               atol=1e-5)
    np.testing.assert_allclose(s2.comp_matrix, M)


def test_recompensate_spoils_only_values_that_depend_on_an_unknown_one():
    """On a log channel a non-positive value has no linear value (NaN). Only
    the outputs that value feeds become unknown: a plain matrix product
    spread NaN through zero coefficients too, as NaN * 0 is NaN."""
    import pandas as pd

    from openflo.pipeline import FlowSample, transform_spec, transform_values
    rng = np.random.default_rng(3)
    n = 400
    raw = np.column_stack([rng.uniform(-50, 5000, n),
                           rng.uniform(100, 20000, n),
                           rng.uniform(100, 20000, n)])
    cols = ['A-A', 'B-A', 'C-A']
    s = FlowSample.from_dataframe(pd.DataFrame(raw, columns=cols),
                                  path='x.fcs')
    s.data['A-A'] = transform_values(raw[:, 0], method='log')
    s.data['B-A'] = transform_values(raw[:, 1], method='logicle')
    s.data['C-A'] = transform_values(raw[:, 2], method='logicle')
    s.data_transforms = {'A-A': transform_spec('log'),
                         'B-A': transform_spec('logicle'),
                         'C-A': transform_spec('logicle')}
    unknown = raw[:, 0] <= 0
    assert unknown.any()
    S = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 0.1, 1.0]])
    n_unknown = s.recompensate(S, cols)
    want = raw[:, 1:] @ np.linalg.inv(S[1:, 1:])
    np.testing.assert_allclose(
        s.data[['B-A', 'C-A']].to_numpy(float),
        np.column_stack([transform_values(want[:, j], method='logicle')
                         for j in range(2)]), atol=1e-9)
    assert s.data['A-A'].isna().to_numpy().tolist() == unknown.tolist()
    assert n_unknown == 0                  # nothing new became unknown
    # Where A does spill into B, B is unknown on exactly those events.
    S2 = S.copy()
    S2[0, 1] = 0.1
    n_unknown = s.recompensate(S2, cols)
    assert s.data['B-A'].isna().to_numpy().tolist() == unknown.tolist()
    assert n_unknown == int(unknown.sum())
    assert not s.data['C-A'].isna().to_numpy()[~unknown].any()
