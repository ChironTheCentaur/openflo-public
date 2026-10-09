"""Tools menu (editor_tools.ToolsMixin, 6 % covered): every launcher opens its
window on a real editor with real loaded samples, and the compensation
editor's Apply callback produces the right data.

Known answers for Apply: re-applying a sample's OWN $SPILL through the editor
must reproduce exactly what the loader produced (QC -> auto_compensate ->
logicle), and applying it twice must not compound.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
MK = ['', '', 'CD901b', 'CD902', 'CD903']
SPILL = '3,FL1-A,FL2-A,FL3-A,1,0.1,0,0,1,0.05,0,0,1'
FL = ['FL1-A', 'FL2-A', 'FL3-A']
SPILL_M = np.array([[1, .1, 0], [0, 1, .05], [0, 0, 1]], float)


def _fcs(path, seed):
    import flowio
    rng = np.random.default_rng(seed)
    n = 800
    true = np.column_stack([rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
                            rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
                            rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
                            rng.normal(800, 100, n)])
    obs = true.copy()
    obs[:, 2:5] = true[:, 2:5] @ SPILL_M                 # add spillover
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, obs.astype(np.float32).flatten().tolist(), CH,
                          opt_channel_names=MK, metadata_dict={'SPILL': SPILL})
    return str(path)


def _loaded(path):
    from openflo.pipeline import FlowSample
    s = FlowSample(path)            # exactly what editor_loadpool does
    s.run_qc()
    s.auto_compensate()
    s.apply_transform()
    return s


@pytest.fixture
def ed_env(tmp_path):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    # synchronous background work, so results are observable immediately
    ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
        on_done(work()) if on_done else work())
    paths = [_fcs(tmp_path / f'expt_s{i}.fcs', i) for i in (1, 2)]
    for p in paths:
        ed._on_loaded(os.path.splitext(os.path.basename(p))[0], _loaded(p))
    yield root, ed, paths
    root.destroy()


def _toplevels(w):
    import tkinter as tk
    out = []
    for c in w.winfo_children():
        if isinstance(c, tk.Toplevel):
            out.append(c)
        out.extend(_toplevels(c))
    return out


LAUNCHERS = ['_open_voltage_dialog', '_open_compare_wsp',
             '_open_synthetic_dialog', '_open_quick_preview',
             '_open_fcs_inspector', '_open_calibration_dialog',
             '_open_comp_qc', '_open_gate_tree', '_open_figure_layout',
             '_open_transform_editor', '_open_comp_editor',
             '_open_spectral_unmix']


@pytest.mark.parametrize('launcher', LAUNCHERS)
def test_each_tools_launcher_opens_a_window(ed_env, launcher, monkeypatch):
    root, ed, _ = ed_env
    from openflo import editor_tools
    shown = []
    monkeypatch.setattr(editor_tools.messagebox, 'showinfo',
                        lambda *a, **k: shown.append(a))
    if launcher == '_open_gate_tree':
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5,
                      'op': '>=', 'parent_id': None, 'name': 'CD901b+'})
    before = set(_toplevels(root))
    getattr(ed, launcher)()
    new = [w for w in _toplevels(root) if w not in before]
    try:
        assert shown == [], f'{launcher} refused: {shown}'
        assert new, (f'{launcher} opened no window; status: '
                     f'{ed.status_var.get()!r}')
    finally:
        for w in new:
            try:
                w.grab_release()
                w.destroy()
            except Exception:                   # noqa: BLE001
                pass


def _capture_apply(monkeypatch, ed):
    import openflo.ui_comp as ui_comp
    got = {}

    class _Stub:
        def __init__(self, parent, sample=None, on_apply=None):
            got['sample'], got['on_apply'] = sample, on_apply
    monkeypatch.setattr(ui_comp, 'CompensationEditorWindow', _Stub)
    ed._open_comp_editor()
    return got


def test_comp_editor_apply_own_spill_reproduces_loader_and_is_idempotent(
        ed_env, monkeypatch):
    root, ed, paths = ed_env
    name = ed._active_sample
    s = ed._samples[name]
    expected = _loaded(s.path).data[FL].to_numpy()
    np.testing.assert_allclose(s.data[FL].to_numpy(), expected, rtol=1e-9)
    got = _capture_apply(monkeypatch, ed)
    assert got['sample'] is s
    got['on_apply'](FL, SPILL_M)
    assert ed.status_var.get().startswith('Applied 3×3 matrix')
    np.testing.assert_allclose(s.data[FL].to_numpy(), expected, rtol=1e-9)
    np.testing.assert_allclose(s.comp_matrix, SPILL_M)
    got['on_apply'](FL, SPILL_M)                     # second Apply
    np.testing.assert_allclose(s.data[FL].to_numpy(), expected, rtol=1e-9)


def test_comp_editor_apply_identity_removes_compensation(ed_env, monkeypatch):
    """Identity matrix = no compensation: values must equal the raw-file
    values pushed through the same QC + transform, with no unmixing."""
    from openflo.pipeline import FlowSample
    root, ed, paths = ed_env
    s = ed._samples[ed._active_sample]
    got = _capture_apply(monkeypatch, ed)
    got['on_apply'](FL, np.eye(3))
    raw = FlowSample(s.path)
    raw.run_qc()
    raw.apply_transform()
    np.testing.assert_allclose(s.data[FL].to_numpy(),
                               raw.data[FL].to_numpy(), rtol=1e-9)


def test_comp_apply_keeps_a_user_changed_channel_transform(ed_env,
                                                           monkeypatch):
    """Apply re-runs the loader (logicle); a channel the Transform editor put
    on asinh must come back on asinh, data and record alike."""
    from openflo.pipeline import inverse_transform_values, transform_values
    root, ed, paths = ed_env
    s = ed._samples[ed._active_sample]
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    assert ed._channel_transform['FL1-A'] == 'asinh'
    want = transform_values(inverse_transform_values(
        _loaded(s.path).data['FL1-A'].to_numpy(float), method='logicle'),
        method='asinh')
    np.testing.assert_allclose(s.data['FL1-A'].to_numpy(float), want,
                               rtol=1e-6, atol=1e-6)    # transform editor ok
    got = _capture_apply(monkeypatch, ed)
    got['on_apply'](FL, SPILL_M)
    np.testing.assert_allclose(s.data['FL1-A'].to_numpy(float), want,
                               rtol=1e-6, atol=1e-6)
    assert s.data_transforms['FL1-A']['method'] == 'asinh'
    assert s.data_transforms['FL2-A']['method'] == 'logicle'   # untouched


def _asinh_of_loader(path, ch='FL1-A'):
    from openflo.pipeline import inverse_transform_values, transform_values
    return transform_values(inverse_transform_values(
        _loaded(path).data[ch].to_numpy(float), method='logicle'),
        method='asinh')


def test_transform_editor_apply_button_retransforms_every_sample(ed_env):
    """The dialog's own action path: pick 'asinh' for FL1-A, press Apply."""
    from tkinter import ttk
    root, ed, paths = ed_env
    before = set(_toplevels(root))
    ed._open_transform_editor()
    dlg, = [w for w in _toplevels(root) if w not in before]

    def walk(w):
        for c in w.winfo_children():
            yield c
            yield from walk(c)
    combos = [w for w in walk(dlg) if isinstance(w, ttk.Combobox)]
    assert len(combos) == len(ed._channels)
    combos[ed._channels.index('FL1-A')].set('asinh')
    apply_btn, = [w for w in walk(dlg) if isinstance(w, ttk.Button)
                  and w.cget('text') == 'Apply']
    apply_btn.invoke()
    assert ed._channel_transform['FL1-A'] == 'asinh'
    for nm, p in zip(ed._sample_order, paths, strict=True):
        np.testing.assert_allclose(ed._samples[nm].data['FL1-A'].to_numpy(float),
                                   _asinh_of_loader(p), rtol=1e-6, atol=1e-6)


def test_sample_loaded_after_transform_change_follows_it(ed_env, tmp_path):
    """The loader always applies logicle; a sample loaded after the Transform
    editor changed a channel must follow the transform in effect."""
    root, ed, paths = ed_env
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    p3 = _fcs(tmp_path / 'expt_s3.fcs', 3)
    ed._on_loaded('expt_s3', _loaded(p3))
    ed._on_load_settled()                    # the deferred post-load pass
    np.testing.assert_allclose(ed._samples['expt_s3'].data['FL1-A']
                               .to_numpy(float), _asinh_of_loader(p3),
                               rtol=1e-6, atol=1e-6)
    assert ed._samples['expt_s3'].data_transforms['FL1-A']['method'] == 'asinh'
    # ...so a further change inverts the right method for every sample.
    ed._apply_channel_transforms({'FL1-A': 'logicle'})
    for nm, p in (('expt_s1', paths[0]), ('expt_s3', p3)):
        np.testing.assert_allclose(
            ed._samples[nm].data['FL1-A'].to_numpy(float),
            _loaded(p).data['FL1-A'].to_numpy(float), rtol=1e-6, atol=1e-6)


def test_sample_loaded_during_a_retransform_follows_it(ed_env, tmp_path):
    """A sample that lands while the background re-transform runs is not in
    its snapshot; it must still end on the new transform."""
    root, ed, paths = ed_env
    p3 = _fcs(tmp_path / 'expt_s3.fcs', 3)
    pending = {}
    ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
        pending.update(work=work, on_done=on_done))
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    out = pending['work']()                  # runs on the pre-load snapshot
    ed._on_loaded('expt_s3', _loaded(p3))    # arrives before on_done
    pending['on_done'](out)
    np.testing.assert_allclose(ed._samples['expt_s3'].data['FL1-A']
                               .to_numpy(float), _asinh_of_loader(p3),
                               rtol=1e-6, atol=1e-6)


def test_later_sample_seeds_a_channel_the_first_lacked(ed_env, tmp_path):
    """A column the first sample did not have takes the transform the new
    sample's data actually carries, not the 'linear' fallback."""
    import flowio
    root, ed, paths = ed_env
    rng = np.random.default_rng(9)
    n = 400
    ev = np.column_stack([rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
                          rng.lognormal(7, .5, n)]).astype(np.float32)
    p = tmp_path / 'extra.fcs'
    with open(p, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(),
                          ['FSC-A', 'SSC-A', 'FL4-A'],
                          opt_channel_names=['', '', 'CD4'])
    assert 'FL4-A' not in ed._channel_transform
    ed._on_loaded('extra', _loaded(str(p)))
    assert ed._channel_transform['FL4-A'] == 'logicle'
    assert ed._channel_transform['FL1-A'] == 'logicle'      # unchanged


def test_processed_csv_records_logicle_detectors_only(ed_env, tmp_path,
                                                      monkeypatch):
    """A processed CSV holds the editor's transformed data. Loading it must
    record its detectors as logicle (the editor's long-standing assumption,
    so they are not transformed twice and tools can invert them) and leave
    the linear columns OpenFlo derives -- U: abundances, cluster ids -- alone."""
    from openflo.editor_loadpool import messagebox
    from openflo.pipeline import linear_values
    shown = []                 # it has no record: the load warns (modal)
    monkeypatch.setattr(messagebox, 'showwarning',
                        lambda *a, **k: shown.append(a))
    root, ed, paths = ed_env
    s1 = ed._samples['expt_s1']
    df = s1.data.copy()
    df['U:F1'] = np.linspace(0.0, 5000.0, len(df))
    df['cluster'] = np.arange(len(df)) % 3
    csv = tmp_path / 'expt_s1_processed.csv'
    df.to_csv(csv, index=False)
    name = ed._import_processed_csv(str(csv))
    s = ed._samples[name]
    for ch in FL:
        assert s.data_transforms[ch]['method'] == 'logicle', ch
        np.testing.assert_allclose(s.data[ch].to_numpy(float),
                                   s1.data[ch].to_numpy(float), rtol=1e-9)
        np.testing.assert_allclose(linear_values(s, ch), linear_values(s1, ch),
                                   rtol=1e-6, atol=1e-6)
    assert 'U:F1' not in s.data_transforms and 'cluster' not in s.data_transforms
    assert ed._channel_transform['U:F1'] == 'linear'
    np.testing.assert_allclose(s.data['U:F1'], df['U:F1'])


def _transform_dialog(root, ed, channel, method):
    """Open the Transform editor, pick `method` for `channel`; returns the
    dialog and its Apply button."""
    from tkinter import ttk
    before = set(_toplevels(root))
    ed._open_transform_editor()
    dlg, = [w for w in _toplevels(root) if w not in before]

    def walk(w):
        for c in w.winfo_children():
            yield c
            yield from walk(c)
    combos = [w for w in walk(dlg) if isinstance(w, ttk.Combobox)]
    combos[ed._channels.index(channel)].set(method)
    apply_btn, = [w for w in walk(dlg) if isinstance(w, ttk.Button)
                  and w.cget('text') == 'Apply']
    return dlg, apply_btn


def test_log_asks_before_events_at_or_below_zero_lose_their_value(
        ed_env, tmp_path, monkeypatch):
    """'log' gives an event at or below zero no value (NaN), and nothing
    restores it: logicle -> log -> logicle left 7,525 of 30,000 events of a
    compensated channel as NaN, out of every gate, and the session sidecar
    saved them so. The Transform editor now names the count and asks first
    ('No' changes nothing), and the status says how many have no value, both
    going onto log and coming off it."""
    import tkinter.messagebox as mb

    import flowio

    from openflo.pipeline import linear_values
    root, ed, paths = ed_env
    rng = np.random.default_rng(5)
    n = 600
    ev = np.column_stack([rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
                          np.where(rng.random(n) < .5,
                                   rng.lognormal(8, .5, n),
                                   rng.normal(0, 150, n)),
                          rng.normal(20000, 500, n), rng.normal(800, 100, n)])
    p = tmp_path / 'neg.fcs'
    with open(p, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).flatten().tolist(), CH,
                          opt_channel_names=MK)
    ed._on_loaded('neg', _loaded(str(p)))
    n_bad = sum(int((linear_values(s, 'FL1-A') <= 0).sum())
                for s in ed._samples.values())
    n_all = sum(len(s.data) for s in ed._samples.values())
    assert n_bad > 100
    before = {nm: s.data['FL1-A'].to_numpy(float).copy()
              for nm, s in ed._samples.items()}

    asked = []
    monkeypatch.setattr(mb, 'askyesno',
                        lambda title, msg, **k: asked.append(msg) or False)
    dlg, apply_btn = _transform_dialog(root, ed, 'FL1-A', 'log')
    apply_btn.invoke()
    assert len(asked) == 1 and f"{n_bad:,} of {n_all:,} events" in asked[0]
    assert ed._channel_transform['FL1-A'] == 'logicle'      # 'No': unchanged
    for nm, s in ed._samples.items():
        np.testing.assert_array_equal(s.data['FL1-A'].to_numpy(float),
                                      before[nm])
    assert dlg.winfo_exists()                       # still open to pick again

    monkeypatch.setattr(mb, 'askyesno', lambda *a, **k: True)
    apply_btn.invoke()
    assert ed._channel_transform['FL1-A'] == 'log'
    assert (f"No log value (≤ 0, out of every gate): FL1-A {n_bad:,} "
            "events") in ed.status_var.get()
    ed._apply_channel_transforms({'FL1-A': 'logicle'})
    assert (f"Still without a value since log: FL1-A {n_bad:,} events"
            in ed.status_var.get())

    # A channel with nothing at or below zero switches without a question.
    asked.clear()
    monkeypatch.setattr(mb, 'askyesno',
                        lambda title, msg, **k: asked.append(msg) or False)
    _dlg, apply_btn = _transform_dialog(root, ed, 'FL3-A', 'log')
    apply_btn.invoke()
    assert asked == [] and ed._channel_transform['FL3-A'] == 'log'
