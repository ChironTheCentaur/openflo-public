"""Every CSV OpenFlo writes says which transform each column carries.

A processed CSV holds whatever the source sample held: logicle by default,
another method on a channel the Transform editor changed, plain linear values
for a CSV someone else wrote. Treating every float detector as logicle on load
was measured wrong three ways: an asinh channel round-tripped through a
workspace run came back transformed twice (median 2.51 -> 23.8), a linear
FlowJo export read back as 5.2e44 (93 % inf), and a session sidecar is the
editor's own data, with the same risk. The record written beside each CSV
(`<stem>_transforms.json`) removes the guess; without one, only a column whose
every value lies inside the loader's logicle range is taken as logicle.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from tests.test_editor_tools import _fcs, _loaded


def isolate_home(monkeypatch, tmp_path):
    """Point ~ at an empty temp dir. Every editor schedules
    _maybe_resume_session, and an earlier test module may have patched
    askyesno to answer yes: pumping Tk here would otherwise resume the user's
    real ~/.openflo autosave (and prune old ones), and prefs would be written
    there too."""
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('USERPROFILE', str(home))
    monkeypatch.setenv('HOME', str(home))


def editor_with_samples(tmp_path):
    """test_editor_tools' setting: an editor holding two loader-processed
    samples (QC -> $SPILL compensation -> logicle), background work run
    synchronously. Call isolate_home first."""
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    ed = _new_editor(root)
    paths = [_fcs(tmp_path / f'expt_s{i}.fcs', i) for i in (1, 2)]
    for p in paths:
        ed._on_loaded(os.path.splitext(os.path.basename(p))[0], _loaded(p))
    yield root, ed, paths
    root.destroy()


def capture_warnings(monkeypatch):
    """Every warning dialog raised, as (title, message). None is displayed:
    a real modal would block the test."""
    from tkinter import messagebox
    out = []
    monkeypatch.setattr(messagebox, 'showwarning',
                        lambda title=None, message=None, **k: out.append(
                            (title, message)))
    return out


@pytest.fixture
def shown(monkeypatch):
    return capture_warnings(monkeypatch)


def settle(root, seconds=1.0):
    """Pump the event loop for a while, so deferred callbacks (load settle,
    status rewrites) have run before anything is asserted -- a note that only
    lived in the status bar is gone by then."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        root.update()
        time.sleep(0.01)


def the_dialog(shown):
    """The one transform-assumption dialog the batch raised; its text."""
    titles = [t for t, _m in shown]
    assert titles == ['Scale assumed for CSV data'], titles
    return shown[0][1]


@pytest.fixture
def ed_env(tmp_path, monkeypatch, shown):
    isolate_home(monkeypatch, tmp_path)
    yield from editor_with_samples(tmp_path)


def _new_editor(root):
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
        on_done(work()) if on_done else work())
    return ed


def _pump(root, done, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not done():
        root.update()
        time.sleep(0.01)
    return done()


# ── BLOCKER (b): a CSV with no record ───────────────────────────────────────

def test_a_linear_csv_without_a_record_is_read_as_linear(ed_env, tmp_path,
                                                         monkeypatch, shown):
    """File > Load CSV of linear values (e.g. a FlowJo scale-values export):
    nothing in it is logicle, so nothing may be inverted as logicle."""
    from openflo import editor_load
    from openflo.pipeline import linear_values
    root, ed, _ = ed_env
    rng = np.random.default_rng(3)
    n = 2000
    fl2 = rng.lognormal(7.0, 0.5, n)
    df = pd.DataFrame({'FSC-A': rng.normal(8e4, 8e3, n),
                       'SSC-A': rng.normal(4e4, 5e3, n),
                       'FL2-A': fl2, 'FL4-A': rng.lognormal(5.0, 0.5, n)})
    p = tmp_path / 'flowjo_export.csv'
    df.to_csv(p, index=False)
    monkeypatch.setattr(editor_load.filedialog, 'askopenfilenames',
                        lambda **k: (str(p),))
    ed._load_processed_data()
    s = ed._samples['flowjo_export']
    for ch, true in (('FL2-A', fl2), ('FL4-A', df['FL4-A'].to_numpy())):
        lin = linear_values(s, ch)
        assert np.isfinite(lin).all(), ch
        np.testing.assert_allclose(lin, true, rtol=1e-6)
    # FL4-A is new to the editor, so it stays as the CSV gave it; FL2-A is
    # put on the editor's logicle, and its record says so.
    assert 'FL4-A' not in s.data_transforms
    np.testing.assert_allclose(s.data['FL4-A'], df['FL4-A'])
    assert s.data_transforms['FL2-A']['method'] == 'logicle'
    settle(root)
    msg = the_dialog(shown)
    assert 'no transform record' in msg and 'linear' in msg, msg


def test_a_logicle_csv_without_a_record_is_still_recognised(ed_env, tmp_path,
                                                            shown):
    """An older OpenFlo wrote no record; its logicle columns lie inside the
    loader's logicle range and are recognised as logicle."""
    from openflo.pipeline import linear_values, transforms_sidecar_path
    root, ed, _ = ed_env
    s1 = ed._samples['expt_s1']
    csv = tmp_path / 'old_session_sidecar.csv'
    s1.data.to_csv(csv, index=False)
    assert not os.path.exists(transforms_sidecar_path(str(csv)))
    name = ed._import_processed_csv(str(csv))
    s = ed._samples[name]
    for ch in ('FL1-A', 'FL2-A', 'FL3-A'):
        assert s.data_transforms[ch]['method'] == 'logicle', ch
        np.testing.assert_allclose(linear_values(s, ch), linear_values(s1, ch),
                                   rtol=1e-6, atol=1e-6)
    assert 'FSC-A' not in s.data_transforms             # scatter stays linear
    # A guess, so the user is told it was one, column by column.
    settle(root)
    msg = the_dialog(shown)
    assert 'GUESSED logicle' in msg and 'FL1-A' in msg, msg


def _load_csv(ed, monkeypatch, path):
    """File > Load CSV… of `path` (a CSV of unknown origin)."""
    from openflo import editor_load
    monkeypatch.setattr(editor_load.filedialog, 'askopenfilenames',
                        lambda **k: (str(path),))
    ed._load_processed_data()
    return ed._samples[os.path.basename(str(path)).rsplit('.', 1)[0]]


def test_a_guessed_logicle_column_is_announced(ed_env, tmp_path, monkeypatch,
                                               shown):
    """Values cannot tell [0, 1] proportions (or min-max-scaled data) from
    logicle coordinates: both lie in the logicle range. The fallback takes
    them as logicle -- a median of 0.5 comes back near 1,560 -- so it must
    say it guessed, naming the column, not stay silent."""
    root, ed, _ = ed_env
    rng = np.random.default_rng(5)
    n = 500
    df = pd.DataFrame({'FSC-A': rng.normal(8e4, 8e3, n),
                       'Frac-A': rng.uniform(0.0, 1.0, n)})
    csv = tmp_path / 'proportions.csv'
    df.to_csv(csv, index=False)
    s = _load_csv(ed, monkeypatch, csv)
    assert s.data_transforms['Frac-A']['method'] == 'logicle'
    settle(root)
    msg = the_dialog(shown)
    assert 'no transform record' in msg, msg
    assert 'GUESSED logicle' in msg and 'Frac-A' in msg, msg


def test_one_load_csv_batch_raises_one_dialog_naming_every_file(
        ed_env, tmp_path, monkeypatch, shown):
    """File > Load CSV of two record-less files: ONE warning, listing both
    and what was assumed for each, still there after the event loop has run
    (the status-bar note was rewritten on the next turn)."""
    from openflo import editor_load
    root, ed, _ = ed_env
    rng = np.random.default_rng(8)
    paths = []
    for nm, col in (('first', rng.lognormal(7, .5, 300)),
                    ('second', rng.uniform(0, 1, 300))):
        p = tmp_path / f'{nm}.csv'
        pd.DataFrame({'FSC-A': rng.normal(8e4, 8e3, 300),
                      'FL2-A': col}).to_csv(p, index=False)
        paths.append(str(p))
    monkeypatch.setattr(editor_load.filedialog, 'askopenfilenames',
                        lambda **k: tuple(paths))
    ed._load_processed_data()
    settle(root)
    msg = the_dialog(shown)
    assert 'first.csv' in msg and 'second.csv' in msg, msg
    assert 'taken as linear: FL2-A' in msg and 'GUESSED logicle' in msg, msg


def test_a_constant_column_without_a_record_stays_linear(ed_env, tmp_path,
                                                         monkeypatch, shown):
    """File > Load CSV: an empty channel (all zeros) or any constant column
    lies inside the logicle range, but nothing logicle-transformed is
    constant, and inverting it as logicle turns zeros into -110.87 at the
    default parameters."""
    from openflo.pipeline import inverse_transform_values, linear_values
    assert inverse_transform_values(np.zeros(1), method='logicle')[0] == (
        pytest.approx(-110.8746, abs=1e-3))
    root, ed, _ = ed_env
    n = 300
    df = pd.DataFrame({'FSC-A': np.linspace(1e4, 9e4, n),
                       'Empty-A': np.zeros(n), 'Flat-A': np.full(n, 0.5),
                       'Dim-A': np.linspace(0.05, 0.9, n)})
    csv = tmp_path / 'empty_channel.csv'
    df.to_csv(csv, index=False)
    s = _load_csv(ed, monkeypatch, csv)
    for ch in ('Empty-A', 'Flat-A'):
        assert ch not in s.data_transforms, ch
        np.testing.assert_array_equal(linear_values(s, ch), df[ch].to_numpy())
    assert s.data_transforms['Dim-A']['method'] == 'logicle'   # varies: guessed
    settle(root)
    msg = the_dialog(shown)
    assert 'Empty-A' in msg and 'Flat-A' in msg, msg


def test_an_openflo_csv_with_one_event_is_still_logicle(ed_env, tmp_path,
                                                        shown):
    """Provenance, not variation, decides for OpenFlo's own CSVs: a session
    sidecar or _events.csv holding a single in-range event is display-
    transformed by definition, so logicle. The constancy rule (an empty
    channel inverts to -110.87) is for File > Load CSV only; before, every
    column of a one-event sidecar was taken as unknown or linear."""
    from openflo.pipeline import linear_values
    root, ed, _ = ed_env
    s1 = ed._samples['expt_s1']
    csv = tmp_path / 'one_events.csv'
    s1.data.iloc[[0]].to_csv(csv, index=False)
    s = ed._samples[ed._import_processed_csv(str(csv))]
    for ch in ('FL1-A', 'FL2-A', 'FL3-A'):
        assert s.data_transforms[ch]['method'] == 'logicle', ch
        np.testing.assert_allclose(linear_values(s, ch),
                                   linear_values(s1, ch)[:1], rtol=1e-6)


def test_logicle_range_is_the_transforms_own(ed_env):
    """The bounds come from the logicle the loader applies, not a constant:
    its image of +-100 x T (the scale plus two decades of headroom)."""
    from openflo.pipeline import logicle_display_bounds, transform_values
    lo, hi = logicle_display_bounds()
    want = transform_values(np.array([-262144e2, 262144e2]), method='logicle')
    assert (lo, hi) == pytest.approx(tuple(want), abs=1e-12)
    assert lo < -1.0 and 1.4 < hi < 1.5
    assert logicle_display_bounds(t=4194304)[1] == pytest.approx(
        float(transform_values(np.array([4194304e2]), method='logicle',
                               t=4194304)[0]))


def test_a_record_for_other_columns_is_not_trusted(ed_env, tmp_path, shown):
    """A record whose column list does not match the CSV (a stale or copied
    file) is ignored with a note; the range rule decides instead."""
    from openflo.pipeline import transforms_sidecar_path, write_transforms_sidecar
    root, ed, _ = ed_env
    s1 = ed._samples['expt_s1']
    csv = tmp_path / 'edited.csv'
    s1.data.drop(columns=['SSC-A']).to_csv(csv, index=False)
    write_transforms_sidecar(str(csv), {'FL2-A': {'method': 'asinh'}},
                             list(s1.data.columns),       # still lists SSC-A
                             len(s1.data))
    assert os.path.exists(transforms_sidecar_path(str(csv)))
    name = ed._import_processed_csv(str(csv))
    s = ed._samples[name]
    assert s.data_transforms['FL2-A']['method'] == 'logicle'
    settle(root)
    assert 'ignored' in the_dialog(shown)


# ── BLOCKER (a): an asinh channel through a workspace run ───────────────────

def test_asinh_events_csv_round_trips_unchanged(ed_env, tmp_path, shown):
    """asinh in the editor -> workspace run -> 'Load in editor' of its
    _events.csv: the values come back as written (not transformed again) and
    their linear values are the source's."""
    from openflo import workspace as W
    from openflo.pipeline import linear_values
    root, ed, _ = ed_env
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    name = 'expt_s1'
    src = ed._samples[name]
    cfg = dict(W.default_run_cfg(), method='flowsom', n_metaclusters=2,
               umap=False, trimap=False, max_events=0)
    prep = W.prepare_run(ed, {'sample': name, 'trial': 'T'}, cfg)
    res = W.compute_run(prep, cfg, str(tmp_path / 'run'))
    assert res['ok'], res
    ev = res['events']
    loaded = ed._samples[ed._import_processed_csv(ev)]
    np.testing.assert_allclose(loaded.data['FL1-A'].to_numpy(float),
                               src.data['FL1-A'].to_numpy(float),
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.median(linear_values(loaded, 'FL1-A')),
                               np.median(linear_values(src, 'FL1-A')),
                               rtol=1e-9)
    assert loaded.data_transforms['FL1-A']['method'] == 'asinh'
    from openflo.pipeline import transforms_sidecar_path
    rec = json.loads(open(transforms_sidecar_path(ev), encoding='utf-8').read())
    assert rec['transforms']['FL1-A']['method'] == 'asinh'
    assert rec['transforms']['FL2-A']['method'] == 'logicle'
    settle(root)
    assert shown == []                 # a record leaves nothing to assume


def test_a_pooled_run_puts_every_member_on_one_transform(ed_env, tmp_path):
    """A unit pools members; one reloaded from file comes back logicle while
    the editor's samples are asinh. The pooled column must be one transform,
    and the record must say which."""
    from openflo import workspace as W
    from openflo.pipeline import (
        FlowSample,
        inverse_transform_values,
        transform_spec,
    )
    root, ed, paths = ed_env
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    p3 = _fcs(tmp_path / 'expt_s3.fcs', 3)
    members = [{'item': {'sample': 'expt_s1'}, 'group': 'G', 'sample': 'expt_s1'},
               {'item': {'sample': 'expt_s3', 'path': p3}, 'group': 'G',
                'sample': 'expt_s3'}]
    cfg = dict(W.default_run_cfg(), max_events=0)
    prep = W.prepare_unit(ed, members, 'G', cfg)
    n1 = len(ed._samples['expt_s1'].data)
    tail = prep['data']['FL1-A'].to_numpy(float)[n1:]
    reload = FlowSample(p3)              # what resolve_run_sample does for it
    reload.apply_transform()
    want = reload.data['FL1-A'].to_numpy(float)
    np.testing.assert_allclose(
        inverse_transform_values(tail, method='asinh'),
        inverse_transform_values(want, method='logicle'), rtol=1e-6, atol=1e-6)
    assert prep['data_transforms']['FL1-A'] == transform_spec('asinh')


@pytest.mark.parametrize('unknown_first', [False, True])
def test_a_pool_mixing_an_unknown_scale_leaves_that_column_out(
        ed_env, tmp_path, unknown_first):
    """One member on asinh, another of unknown scale (an older OpenFlo
    sidecar holding its asinh values): one column, two scales that cannot be
    reconciled. The pooled record says unknown, the column is left out of
    clustering and embedding (it used to cluster on the mixed coordinates
    and report a clean run), and the run says so."""
    from openflo import workspace as W
    from openflo.pipeline import UNKNOWN_SPEC
    root, ed, _ = ed_env
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    old = ed._samples['expt_s2']                  # as restored from 2.6.1
    old.data_transforms = {**old.data_transforms,
                           'FL1-A': dict(UNKNOWN_SPEC)}
    names = (['expt_s2', 'expt_s1'] if unknown_first
             else ['expt_s1', 'expt_s2'])
    members = [{'item': {'sample': n}, 'group': 'G', 'sample': n}
               for n in names]
    cfg = dict(W.default_run_cfg(), method='flowsom', n_metaclusters=2,
               umap=False, trimap=False, max_events=0)
    prep = W.prepare_unit(ed, members, 'G', cfg)
    assert prep['data_transforms']['FL1-A'] == {'method': 'unknown'}
    assert 'FL1-A' not in prep['channels']
    assert {'FL2-A', 'FL3-A'} <= set(prep['channels'])
    res = W.compute_run(prep, cfg, str(tmp_path / 'run'))
    assert res['ok'], res
    assert 'FL1-A' not in res['channels']
    assert any('FL1-A' in w for w in res['warnings']), res['warnings']
    assert 'FL1-A' in res['note']


@pytest.mark.parametrize('how, kept', [
    ('one session folder', True),
    ('two session folders', False),
    ('two _events.csv, one folder', False),
])
def test_two_unknown_members_pool_only_from_one_old_session(
        ed_env, tmp_path, how, kept):
    """A column of unknown scale in every member is one scale only if the
    members came from ONE older session's data folder (one 2.6.1 editor,
    one transform per channel). Two old _events.csv -- even in one folder --
    or the sidecars of two sessions may differ: measured, an asinh member
    (median 1.197) beside a linear one (median 225.6) went into FlowSOM, and
    it split them by scale. Those are left out, with the warning."""
    from openflo import workspace as W
    from openflo.pipeline import inverse_transform_values
    root, ed, _ = ed_env
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    asinh = ed._samples['expt_s1'].data.copy()
    other = asinh.copy()
    if not kept:                       # really on another scale: linear
        other['FL1-A'] = inverse_transform_values(
            other['FL1-A'].to_numpy(float), method='asinh')
    names = []
    for i, df in enumerate((asinh, other)):
        if how == 'two _events.csv, one folder':
            folder = tmp_path / 'run'
            csv = folder / f'Old{i}_events.csv'
        else:
            folder = tmp_path / ('s_data' if how == 'one session folder'
                                 else f's{i}_data')
            csv = folder / f'Old{i}.csv'
        folder.mkdir(exist_ok=True)
        df.to_csv(csv, index=False)                   # no record: older
        if how.endswith('one folder'):
            names.append(ed._import_processed_csv(str(csv)))
        else:
            ed._load_csv_worker(f'Old{i}', str(csv), {})    # a restore
            settle(root, 0.3)
            names.append(f'Old{i}')
    for n in names:
        assert ed._samples[n].data_transforms['FL1-A']['method'] == 'unknown'
    members = [{'item': {'sample': n}, 'group': 'G', 'sample': n}
               for n in names]
    cfg = dict(W.default_run_cfg(), method='flowsom', n_metaclusters=2,
               umap=False, trimap=False, max_events=0)
    prep = W.prepare_unit(ed, members, 'G', cfg)
    assert ('FL1-A' in prep['channels']) is kept, prep['note']
    assert any('FL1-A' in w for w in prep['warnings']) is not kept


def test_a_pool_clusters_on_the_channels_every_member_has(ed_env, tmp_path):
    """A pooled unit took its channels from the FIRST member. A later one
    lacking a column got NaN there, and every one of its events went
    unclustered while the run said 'ok' (2.6.1 too). The unit clusters on
    the channels common to every member and names the one left out."""
    import pandas as pd

    from openflo import workspace as W
    root, ed, _ = ed_env
    b = ed._samples['expt_s2']
    b.data = b.data.drop(columns=['FL3-A'])        # another panel
    members = [{'item': {'sample': n}, 'group': 'G', 'sample': n}
               for n in ('expt_s1', 'expt_s2')]
    cfg = dict(W.default_run_cfg(), method='flowsom', n_metaclusters=2,
               umap=False, trimap=False, max_events=0)
    prep = W.prepare_unit(ed, members, 'G', cfg)
    res = W.compute_run(prep, cfg, str(tmp_path / 'run'))
    assert res['ok'], res
    ev = pd.read_csv(res['events'])
    from_b = ev['__sample__'].astype(str).str.endswith('expt_s2')
    clustered = pd.to_numeric(ev.loc[from_b, 'cluster'], errors='coerce')
    n_b = int(from_b.sum())
    assert int((clustered >= 0).sum()) == n_b, (
        f"{int((clustered >= 0).sum())} of {n_b} events of expt_s2 clustered")
    assert 'FL3-A' not in prep['channels']
    assert {'FL1-A', 'FL2-A'} <= set(prep['channels'])
    assert any('FL3-A' in w for w in res['warnings']), res['warnings']


def test_the_run_panel_reports_a_left_out_column(tmp_path):
    """The run's note never reached the panel: its per-job line, and the
    final 'Done' line (the per-job one is replaced by the next job's), must
    name the left-out column."""
    import collections
    import pickle
    import queue
    import types

    from openflo import workspace as W
    res = dict(W._blank_result('G', 800), ok=True, n_clusters=2,
               warnings=['left out of clustering, its members are on '
                         'different scales: FL1-A'])
    result = tmp_path / 'result.pkl'
    with open(result, 'wb') as fh:
        pickle.dump(res, fh)
    (tmp_path / 'job').mkdir()
    lines, finished = [], []
    view = types.SimpleNamespace(
        status_var=types.SimpleNamespace(set=lines.append), _total=1, _ok=0,
        _err=0, _jobs=collections.deque(), _warnings=[], _prep=None,
        _cancel_requested=False, _out_dir=str(tmp_path),
        _absorb_stdout=lambda cur: None, _finish=finished.append,
        _cur={'job': {'label': 'G', 'attempt': 1, 'cfg': {}},
              'proc': types.SimpleNamespace(poll=lambda: 0),
              'result_path': str(result), 'jobdir': str(tmp_path / 'job'),
              'n': 1, 'msgq': queue.Queue()})
    W.WorkspacePanel._finalize_current(view)
    assert '✓' in lines[-1] and 'FL1-A' in lines[-1], lines
    W.WorkspacePanel._pump_jobs(view)
    assert finished and 'Done' in finished[-1] and 'FL1-A' in finished[-1]


# ── BLOCKER (c): a session sidecar ──────────────────────────────────────────

def test_session_sidecar_round_trips_its_transforms(ed_env, tmp_path, shown):
    """Save a session with an asinh channel, reopen it in a fresh editor: the
    sidecar's values and record come back exactly, and the session file's
    schema is untouched (the record lives beside the CSV)."""
    from openflo.pipeline import linear_values
    from openflo.session_format import SESSION_KEYS
    root, ed, _ = ed_env
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    path = tmp_path / 'sess.flowsession'
    data = ed._write_session(str(path))
    assert set(data) == set(SESSION_KEYS)
    csvs = {e['name']: tmp_path / e['processed_csv'] for e in data['samples']}
    assert set(csvs) == {'expt_s1', 'expt_s2'}

    ed2 = _new_editor(root)
    assert ed2._find_resumable_session() is None     # ~ is the test's own
    # The restore's own reader, run on this thread: Tk refuses calls from
    # the pool's threads while a test pumps update() instead of mainloop().
    ed2._queue_processed_loads = lambda items, front_names=(): [
        ed2._load_csv_worker(nm, p, {}) for nm, p in items]
    ed2._load_session_path(str(path))
    assert _pump(root, lambda: len(ed2._samples) == 2), ed2.status_var.get()
    for nm in ('expt_s1', 'expt_s2'):
        s, src = ed2._samples[nm], ed._samples[nm]
        np.testing.assert_allclose(s.data['FL1-A'].to_numpy(float),
                                   src.data['FL1-A'].to_numpy(float),
                                   rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(linear_values(s, 'FL1-A'),
                                   linear_values(src, 'FL1-A'),
                                   rtol=1e-6, atol=1e-6)
        assert s.data_transforms['FL1-A']['method'] == 'asinh', nm
        assert s.data_transforms['FL2-A']['method'] == 'logicle', nm
    assert ed2._channel_transform['FL1-A'] == 'asinh'
    from openflo.pipeline import transforms_sidecar_path
    for csv in csvs.values():
        assert os.path.isfile(transforms_sidecar_path(str(csv)))
    settle(root)
    assert shown == []                 # records: nothing assumed, no dialog


def test_pipeline_export_csv_writes_its_record(tmp_path):
    """FlowSample.export_csv (the CLI's <name>_processed.csv) writes the
    record beside it, so the CSV re-opens on the right scale."""
    from openflo.pipeline import transforms_sidecar_path
    p = _fcs(tmp_path / 'cli.fcs', 4)
    s = _loaded(p)
    out = tmp_path / 'cli_processed.csv'
    s.export_csv(str(out))
    rec = json.loads(open(transforms_sidecar_path(str(out)),
                          encoding='utf-8').read())
    assert rec['columns'] == list(s.data.columns)
    assert rec['rows'] == len(s.data)
    assert rec['transforms'] == s.data_transforms


# ── MINOR (5): apply_transform does not stack ─────────────────────────────

def test_apply_transform_refuses_to_stack(tmp_path):
    """A second apply_transform would transform the transformed values and
    record only the last (measured: linear_values 0.446 where 924 is true).
    It raises instead and leaves data and record as they were;
    retransform is the call that changes a channel's method."""
    from openflo.pipeline import linear_values
    s = _loaded(_fcs(tmp_path / 'twice.fcs', 5))
    before = s.data['FL1-A'].to_numpy(float).copy()
    rec = dict(s.data_transforms)
    with pytest.raises(ValueError, match='retransform'):
        s.apply_transform()
    with pytest.raises(ValueError, match='retransform'):
        s.apply_transform(channels=['FL2-A'], method='linear')
    np.testing.assert_array_equal(s.data['FL1-A'].to_numpy(float), before)
    assert s.data_transforms == rec
    lin = linear_values(s, 'FL1-A')
    s.retransform(['FL1-A'], method='asinh')
    np.testing.assert_allclose(linear_values(s, 'FL1-A'), lin, rtol=1e-6,
                               atol=1e-6)
    s.apply_transform(channels=['FSC-A'], method='asinh')     # untransformed: ok
    assert s.data_transforms['FSC-A']['method'] == 'asinh'
