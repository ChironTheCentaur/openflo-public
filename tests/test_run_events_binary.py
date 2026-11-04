"""A workspace run writes its per-event table as a binary file.

Formatting every value of the per-event table as CSV text cost ~10 s a
sample where the binary session-sidecar format takes ~0.2 s, on every run.
So a run writes ``<label>_events.npz`` (session_sidecar.write_frame) with
its transform record beside it, and writes ``<label>_events.csv`` only when
the run setting 'Also write events as CSV' is on, or when the table holds a
column the binary format can't carry. 'Load in editor', the results listing
and File > Load processed data read both, so the CSVs of earlier runs load as
they always did.

All data is generated here: two well-separated populations on FL1-A / FL2-A,
so the run's two clusters are known.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import workspace_run as W
from openflo.session_sidecar import SIDECAR_EXT, read_frame
from openflo.transform_sidecar import (
    read_transforms_sidecar,
    transforms_sidecar_path,
    write_transforms_sidecar,
)
from tests.conftest import gui_unavailable

N = 600
CFG = {'method': 'flowsom', 'n_metaclusters': 2, 'umap': False,
       'trimap': False, 'tsne': False, 'phate': False, 'pacmap': False,
       'max_events': 0}


def _linear_frame(seed=0):
    """Linear intensities and the TRUE population of each event (60 % bright
    on FL1-A, 40 % bright on FL2-A)."""
    rng = np.random.default_rng(seed)
    pop = np.arange(N) < int(N * 0.6)
    rng.shuffle(pop)
    df = pd.DataFrame({
        'FSC-A': rng.normal(6e4, 5e3, N),
        'SSC-A': rng.normal(3e4, 4e3, N),
        'FL1-A': np.where(pop, rng.normal(20000, 1500, N),
                          rng.normal(150, 40, N)),
        'FL2-A': np.where(pop, rng.normal(150, 40, N),
                          rng.normal(15000, 1500, N)),
        'FL3-A': rng.normal(800, 100, N),
    })
    return df, pop


def _display_sample(name='synth_a', seed=0):
    """A sample as the editor holds one: on the loader's logicle."""
    from openflo.pipeline import FlowSample
    df, pop = _linear_frame(seed)
    s = FlowSample.from_dataframe(df, name=name)
    s.apply_transform()
    return s, pop


def _run(editor, name, out_dir, **over):
    cfg = dict(W.default_run_cfg(), **CFG, **over)
    prep = W.prepare_run(editor, {'sample': name, 'trial': 'T'}, cfg)
    res = W.compute_run(prep, cfg, str(out_dir))
    assert res['ok'], res
    return res, prep


@pytest.fixture
def spy_written(monkeypatch):
    """Every frame session_sidecar.write_frame was handed, by path."""
    from openflo import session_sidecar
    seen = {}
    real = session_sidecar.write_frame

    def spy(df, path):
        seen[path] = df.copy()
        return real(df, path)
    monkeypatch.setattr(session_sidecar, 'write_frame', spy)
    return seen


def _read_csv_exact(path):
    """A CSV read back to the exact float64 it was written from (pandas'
    default parser can be one unit in the last place off)."""
    return pd.read_csv(path, float_precision='round_trip')


def _assert_events_match_source(ev, sample, pop):
    """The events table holds the run's input values unchanged, its source
    tags, and a cluster per planted population."""
    for c in ('FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A'):
        np.testing.assert_array_equal(ev[c].to_numpy(),
                                      sample.data[c].to_numpy(), err_msg=c)
    assert set(ev['__sample__'].astype(str)) == {sample.name}
    assert set(ev['__group__'].astype(str)) == {'(ungrouped)'}
    ct = pd.crosstab(ev['cluster'].to_numpy(), pop)
    # Each cluster is one population, each population one cluster.
    assert ct.shape == (2, 2)
    assert sorted(ct.to_numpy().max(axis=1).tolist()) == [int((~pop).sum()),
                                                          int(pop.sum())]


# ── core: what a run writes ──────────────────────────────────────────────

def test_a_run_writes_binary_events_and_no_csv(tmp_path, spy_written):
    s, pop = _display_sample()
    out = tmp_path / 'run'
    res, _prep = _run(SimpleNamespace(_samples={s.name: s}), s.name, out)
    ev_path = str(out / f'T_{s.name}_events{SIDECAR_EXT}')
    assert res['events'] == ev_path and os.path.isfile(ev_path)
    assert ev_path in res['files']
    assert not (out / f'T_{s.name}_events.csv').exists()
    assert not any(f.endswith('_events.csv') for f in res['files'])
    ev = read_frame(ev_path)
    # Exactly what was written: every value and dtype.
    pd.testing.assert_frame_equal(ev, spy_written[ev_path], check_exact=True)
    assert ev['FL1-A'].dtype == np.float64
    _assert_events_match_source(ev, s, pop)
    # The transform record is beside it, named from the full stem.
    rec = read_transforms_sidecar(ev_path, list(ev.columns), len(ev))
    assert rec is not None
    assert {rec[c]['method'] for c in ('FL1-A', 'FL2-A', 'FL3-A')} == {'logicle'}
    assert transforms_sidecar_path(ev_path).endswith(
        f'T_{s.name}_events_transforms.json')


def test_also_write_csv_writes_both_and_they_agree(tmp_path):
    s, pop = _display_sample()
    out = tmp_path / 'run'
    res, _prep = _run(SimpleNamespace(_samples={s.name: s}), s.name, out,
                      events_csv=True)
    npz = str(out / f'T_{s.name}_events{SIDECAR_EXT}')
    csv = str(out / f'T_{s.name}_events.csv')
    assert res['events'] == npz                  # the editor reads the binary
    assert {npz, csv} <= set(res['files'])
    ev, from_csv = read_frame(npz), _read_csv_exact(csv)
    assert list(from_csv.columns) == list(ev.columns)
    # Every value is in the CSV exactly; the tags read back as the same text.
    pd.testing.assert_frame_equal(from_csv, ev, check_dtype=False,
                                  check_exact=True)
    _assert_events_match_source(from_csv, s, pop)
    # One record serves both files.
    for p in (npz, csv):
        rec = read_transforms_sidecar(p, list(ev.columns), len(ev))
        assert rec['FL1-A']['method'] == 'logicle', p


def test_a_column_the_binary_file_cannot_carry_falls_back_to_csv(tmp_path):
    """A nullable integer column (missing values) has no binary form: the
    run writes the CSV, as runs did before, and still records transforms."""
    s, pop = _display_sample()
    flag = pd.array([1 if i % 3 else None for i in range(N)], dtype='Int64')
    s.data['FL3-A_flag'] = flag
    out = tmp_path / 'run'
    # An earlier run's binary file in the same folder must not outlive it.
    out.mkdir()
    stale = out / f'T_{s.name}_events{SIDECAR_EXT}'
    stale.write_bytes(b'old')
    res, _prep = _run(SimpleNamespace(_samples={s.name: s}), s.name, out)
    csv = str(out / f'T_{s.name}_events.csv')
    assert res['events'] == csv and res['files'].count(csv) == 1
    assert not stale.exists()
    ev = _read_csv_exact(csv)
    _assert_events_match_source(ev, s, pop)
    assert ev['FL3-A_flag'].isna().sum() == int(pd.isna(flag).sum())
    rec = read_transforms_sidecar(csv, list(ev.columns), len(ev))
    assert rec['FL1-A']['method'] == 'logicle'
    # The listing finds it.
    [r] = W.find_run_outputs(str(out))
    assert r['events'] == csv


def test_a_rerun_without_csv_removes_the_earlier_csv(tmp_path):
    """A run with the setting on, then one with it off, into one folder:
    the earlier CSV would sit beside newer data under the same record."""
    s, _pop = _display_sample()
    ed = SimpleNamespace(_samples={s.name: s})
    out = tmp_path / 'run'
    _run(ed, s.name, out, events_csv=True)
    csv = out / f'T_{s.name}_events.csv'
    assert csv.exists()
    _run(ed, s.name, out)
    assert not csv.exists()
    assert (out / f'T_{s.name}_events{SIDECAR_EXT}').exists()


def test_grouped_run_tags_go_in_the_binary_file(tmp_path, spy_written):
    """A pooled run's __group__ / __sample__ tags are categoricals; they are
    written as text (what a CSV of them read back as), not left to force
    every run back to CSV."""
    a, _pop_a = _display_sample('synth_a', seed=1)
    b, _pop_b = _display_sample('synth_b', seed=2)
    ed = SimpleNamespace(_samples={'synth_a': a, 'synth_b': b})
    members = [{'item': {'sample': n}, 'group': 'G1', 'sample': n}
               for n in ('synth_a', 'synth_b')]
    cfg = dict(W.default_run_cfg(), **CFG)
    prep = W.prepare_unit(ed, members, 'G1', cfg)
    res = W.compute_run(prep, cfg, str(tmp_path))
    assert res['ok'], res
    assert res['events'].endswith('G1_events' + SIDECAR_EXT)
    ev = read_frame(res['events'])
    pd.testing.assert_frame_equal(ev, spy_written[res['events']],
                                  check_exact=True)
    assert ev['__sample__'].tolist() == (['G1 / synth_a'] * N
                                         + ['G1 / synth_b'] * N)
    assert set(ev['__group__']) == {'G1'}
    np.testing.assert_array_equal(
        ev['FL1-A'].to_numpy(),
        np.r_[a.data['FL1-A'].to_numpy(), b.data['FL1-A'].to_numpy()])
    assert not os.path.exists(os.path.join(str(tmp_path), 'G1_events.csv'))


def test_results_listing_takes_both_formats(tmp_path):
    new, old, both = (tmp_path / d for d in ('new', 'old', 'both'))
    for d in (new, old, both):
        d.mkdir()
    (new / f'T_a_events{SIDECAR_EXT}').write_bytes(b'')
    (old / 'T_a_events.csv').write_text('FL1-A\n1\n')
    (both / 'T_a_events.csv').write_text('FL1-A\n1\n')
    (both / f'T_a_events{SIDECAR_EXT}').write_bytes(b'')
    (both / 'T_b_events.csv').write_text('FL1-A\n1\n')
    assert [r['events'] for r in W.find_run_outputs(str(new))] == [
        str(new / f'T_a_events{SIDECAR_EXT}')]
    assert [r['events'] for r in W.find_run_outputs(str(old))] == [
        str(old / 'T_a_events.csv')]
    got = {r['label']: r['events'] for r in W.find_run_outputs(str(both))}
    assert got == {'T_a': str(both / f'T_a_events{SIDECAR_EXT}'),
                   'T_b': str(both / 'T_b_events.csv')}


def test_the_setting_is_off_by_default_and_in_older_files():
    assert W.default_run_cfg()['events_csv'] is False
    m = W.WorkspaceModel('WS')
    d = m.to_dict()
    del d['run_cfg']['events_csv']                     # a file from before it
    assert W.WorkspaceModel.from_dict(json.loads(json.dumps(d))
                                      ).run_cfg['events_csv'] is False
    d['run_cfg']['events_csv'] = True
    assert W.WorkspaceModel.from_dict(d).run_cfg['events_csv'] is True
    # A group override can't turn it into anything else.
    assert W.merge_unit_cfg(W.default_run_cfg(), {})['events_csv'] is False


# ── GUI: 'Load in editor', File > Load processed data, the run bar ───────

@pytest.fixture
def editor(tmp_path, monkeypatch):
    """A real editor with one generated sample, FL1-A moved to asinh so the
    transform record has something to carry. No dialog is ever shown."""
    from tests.test_transform_records import (
        _new_editor,
        capture_warnings,
        isolate_home,
    )
    isolate_home(monkeypatch, tmp_path)
    shown = capture_warnings(monkeypatch)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    ed = _new_editor(root)
    s, pop = _display_sample()
    ed._on_loaded(s.name, s)
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    yield SimpleNamespace(root=root, ed=ed, name=s.name, pop=pop, shown=shown)
    root.destroy()


def _assert_loaded_like_source(loaded, src, rtol=0.0):
    """Values as the source holds them (exactly, or within `rtol` for a
    CSV, which the editor parses with pandas' default float parser), and
    the source's transform per channel."""
    from openflo.pipeline import linear_values
    for c in ('FL1-A', 'FL2-A', 'FL3-A'):
        np.testing.assert_allclose(loaded.data[c].to_numpy(float),
                                   src.data[c].to_numpy(float), rtol=rtol,
                                   atol=0, err_msg=c)
        np.testing.assert_allclose(linear_values(loaded, c),
                                   linear_values(src, c), rtol=1e-9,
                                   err_msg=c)
    assert loaded.data_transforms['FL1-A']['method'] == 'asinh'
    assert loaded.data_transforms['FL2-A']['method'] == 'logicle'


def test_load_in_editor_reads_the_binary_file_with_its_transforms(editor,
                                                                   tmp_path):
    from tests._tk_wait import settle
    ed = editor.ed
    src = ed._samples[editor.name]
    res, _prep = _run(ed, editor.name, tmp_path / 'run')
    assert res['events'].endswith(SIDECAR_EXT)
    name = ed._import_processed_csv(res['events'])
    assert name == f'T_{editor.name}'
    loaded = ed._samples[name]
    _assert_loaded_like_source(loaded, src)
    pd.testing.assert_frame_equal(loaded.data[list(read_frame(res['events']))],
                                  read_frame(res['events']), check_exact=True)
    settle(editor.root, 0.3)
    assert editor.shown == []           # a record leaves nothing to assume


def test_an_earlier_runs_events_csv_still_loads(editor, tmp_path):
    """An `_events.csv` and its record as runs wrote them before the binary
    file: written here the way that code did (to_csv, then the record)."""
    from tests._tk_wait import settle
    ed = editor.ed
    src = ed._samples[editor.name]
    res, _prep = _run(ed, editor.name, tmp_path / 'new')
    frame = read_frame(res['events'])
    old = tmp_path / 'old'
    old.mkdir()
    csv = str(old / f'T_{editor.name}_events.csv')
    frame.to_csv(csv, index=False)
    write_transforms_sidecar(csv, src.data_transforms, frame.columns,
                             len(frame))
    [rec] = W.find_run_outputs(str(old))
    assert rec['events'] == csv
    loaded = ed._samples[ed._import_processed_csv(rec['events'])]
    _assert_loaded_like_source(loaded, src, rtol=1e-15)
    np.testing.assert_array_equal(loaded.data['cluster'].to_numpy(),
                                  frame['cluster'].to_numpy())
    settle(editor.root, 0.3)
    assert editor.shown == []


def test_file_load_processed_data_reads_the_binary_file(editor, tmp_path,
                                                        monkeypatch):
    from openflo import editor_load
    ed = editor.ed
    src = ed._samples[editor.name]
    res, _prep = _run(ed, editor.name, tmp_path / 'run')
    asked = {}

    def pick(**k):
        asked.update(k)
        return (res['events'],)
    monkeypatch.setattr(editor_load.filedialog, 'askopenfilenames', pick)
    ed._load_processed_data()
    assert any('*.npz' in pat for _t, pat in asked['filetypes'])
    loaded = ed._samples[f'T_{editor.name}']
    _assert_loaded_like_source(loaded, src)


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_run_bar_setting_has_a_tip_and_reaches_the_run(tmp_path, monkeypatch):
    try:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    try:
        from openflo.workspace import EVENTS_CSV_TIP, WorkspacePanel
        tips = {}
        editor = SimpleNamespace(
            _samples={}, _sample_order=[], _sample_trial={},
            _sample_colors={}, _sample_gates={}, _sample_gate_order={},
            _channel_labels={},
            _tip=lambda w, t: tips.__setitem__(str(w), t))
        panel = WorkspacePanel(root, editor=editor)
        v = panel._view
        [box] = [w for w in _walk(v) if isinstance(w, ttk.Checkbutton)
                 and w.cget('text') == 'Also write events as CSV']
        assert tips.get(str(box)) == EVENTS_CSV_TIP
        assert '_events.csv' in EVENTS_CSV_TIP and 'Excel' in EVENTS_CSV_TIP
        assert v._sync_run_cfg()['events_csv'] is False     # pre-filled off
        box.invoke()
        assert v._sync_run_cfg()['events_csv'] is True
        assert panel.model.run_cfg['events_csv'] is True
        # A loaded workspace / recipe puts the box back.
        v.model.run_cfg['events_csv'] = False
        v._load_cfg_into_widgets()
        assert v._sync_run_cfg()['events_csv'] is False
    finally:
        root.destroy()


def test_results_window_loads_the_binary_file(tmp_path):
    """Results… > 'Load in editor' hands the editor the binary file."""
    try:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    try:
        from openflo.workspace import ResultsViewer
        s, _pop = _display_sample()
        out = tmp_path / 'run'
        res, _prep = _run(SimpleNamespace(_samples={s.name: s}), s.name, out,
                          events_csv=True)
        got = []
        ed = SimpleNamespace(_import_processed_csv=got.append)
        win = ResultsViewer(root, str(out), editor=ed)
        [btn] = [w for w in _walk(win) if isinstance(w, ttk.Button)
                 and w.cget('text') == 'Load in editor']
        btn.invoke()
        assert got == [res['events']] and got[0].endswith(SIDECAR_EXT)
        win.destroy()
    finally:
        root.destroy()


def test_a_failed_events_write_is_reported_as_a_warning(tmp_path,
                                                        monkeypatch):
    """A run whose per-event table can't be written still keeps its clusters,
    but the failure travels in the result's warnings, which the run's result
    line and final 'Done' line show, instead of being said once on a progress
    line the next job replaces."""
    s, _pop = _display_sample()

    def refuse(*_a, **_k):
        raise OSError('disk full (synthetic)')

    monkeypatch.setattr(W, 'write_run_events', refuse)
    out = tmp_path / 'run'
    res, _prep = _run(SimpleNamespace(_samples={s.name: s}), s.name, out)
    assert res['n_clusters'] == 2
    assert res['events'] is None
    assert not any('_events' in os.path.basename(f) for f in res['files'])
    assert res['warnings'] == [
        'events file not written: OSError: disk full (synthetic)']


def test_the_events_warning_reaches_the_run_result_line(tmp_path,
                                                        monkeypatch):
    """The run panel reads a finished job's result and shows its warnings on
    the job's line and keeps them for the final line. Driven with the real
    result of a run whose events write failed."""
    import pickle

    from openflo.workspace import WorkspacePanel

    s, _pop = _display_sample()

    def refuse(*_a, **_k):
        raise OSError('disk full (synthetic)')

    monkeypatch.setattr(W, 'write_run_events', refuse)
    res, _prep = _run(SimpleNamespace(_samples={s.name: s}), s.name,
                      tmp_path / 'run')
    jobdir = tmp_path / 'job'
    jobdir.mkdir()
    result_path = jobdir / 'result.pkl'
    result_path.write_bytes(pickle.dumps(res))
    shown = []
    panel = SimpleNamespace(
        _cur={'job': {'label': 'synth_a', 'attempt': 1},
              'proc': SimpleNamespace(poll=lambda: 0),
              'result_path': str(result_path), 'jobdir': str(jobdir),
              'n': 1},
        _absorb_stdout=lambda _cur: None,
        status_var=SimpleNamespace(set=shown.append),
        _total=1, _ok=0, _err=0, _warnings=[])
    WorkspacePanel._finalize_current(panel)
    assert panel._ok == 1 and panel._err == 0
    assert shown[-1].endswith(
        '[!] events file not written: OSError: disk full (synthetic)'), shown
    assert panel._warnings == [
        'synth_a: events file not written: OSError: disk full (synthetic)']
