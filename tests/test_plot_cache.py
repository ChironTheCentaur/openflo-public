"""Per-sample display caches and the background warm-up (editor_plotcache).

Switching the displayed/active sample used to redo work over every event on
every replot. The caches that replace it must (1) give exactly the numbers the
old full recomputation gave -- gate counts, filtered rows, histogram values --
and (2) never be stale: every way a sample's data or gates change has to miss
the cache. The second half of this file exercises each kind of invalidation.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

CHANS = ['FSC-A', 'FSC-H', 'SSC-A', 'CD3-A', 'CD4-A', 'Time']


def _frame(n, seed, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    fsc = rng.normal(100.0, 30.0, n)
    df = pd.DataFrame({
        'FSC-A': fsc,
        'FSC-H': fsc * 0.8 + rng.normal(0, 5, n),
        'SSC-A': rng.normal(50.0, 20.0, n),
        'CD3-A': rng.normal(0.5, 0.2, n),
        'CD4-A': rng.normal(0.4, 0.2, n),
        'Time': np.arange(n) / 100.0,
    })
    if nan_frac:
        df.loc[rng.random(n) < nan_frac, 'SSC-A'] = np.nan
        df.loc[rng.random(n) < nan_frac, 'CD3-A'] = np.nan
    return df


def _gates():
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    poly = [[100 + 45 * np.cos(t), 80 + 40 * np.sin(t)] for t in th]
    return {
        'g1': {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
               'x0': 40.0, 'x1': 170.0, 'y0': 10.0, 'y1': 95.0,
               'parent_id': None, 'name': 'Cells', 'enabled': True},
        'g2': {'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'FSC-H',
               'vertices': poly, 'parent_id': 'g1', 'name': 'Singlets',
               'enabled': True},
        'g3': {'kind': 'threshold', 'channel': 'CD3-A', 'value': 0.45,
               'parent_id': 'g2', 'name': 'CD3+', 'enabled': True},
        'g4': {'kind': 'threshold', 'channel': 'CD4-A', 'value': 0.4,
               'parent_id': 'g3', 'name': 'CD4+', 'enabled': True},
        'g5': {'kind': 'interval', 'channel': 'CD4-A', 'lo': 0.0, 'hi': 0.3,
               'parent_id': 'g3', 'name': 'CD4lo', 'enabled': True},
    }


def _add(ed, name, df, shown=False):
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(df, name=name)
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = shown
    ed._sample_gates[name] = _gates()
    ed._sample_gate_order[name] = ['g1', 'g2', 'g3', 'g4', 'g5']
    ed._sample_gate_seq[name] = 5
    return s


@pytest.fixture
def ed(monkeypatch):
    import openflo.async_task as _at

    def _inline(widget, work, on_done=None, on_error=None, on_finally=None):
        try:
            res = work()
        except Exception as exc:                  # noqa: BLE001
            if on_error:
                on_error(exc)
        else:
            if on_done:
                on_done(res)
        finally:
            if on_finally:
                on_finally()
    monkeypatch.setattr(_at, 'run_async', _inline)
    root, editor, _gui = _editor_or_skip()
    editor._pc_warm_enabled = False          # tests opt in to the warm-up
    _add(editor, 'A', _frame(6000, 1, nan_frac=0.02), shown=True)
    _add(editor, 'B', _frame(4000, 2))
    _add(editor, 'C', _frame(5000, 3, nan_frac=0.05))
    editor._channels = list(CHANS)
    editor._channel_labels = {c: c for c in CHANS}
    editor._populate_channel_combos()
    editor._set_active_sample('A')
    editor._refresh_gate_list()
    try:
        yield editor
    finally:
        try:
            editor.destroy()
        except Exception:                         # noqa: BLE001
            pass
        root.destroy()


# ── reference: the pre-cache computations, verbatim in spirit ────────────────

def _legacy_frame(ed, name, x, y, apply_gates):
    from openflo.pipeline import cumulative_gate_mask
    s = ed._samples[name]
    df = s.data.dropna(subset=[c for c in (x, y) if c])
    gates = ed._sample_gates[name]
    if apply_gates and gates:
        ov = ed._autoclean_overrides(name, df)
        mask = np.zeros(len(df), dtype=bool)
        anyen = False
        for gid, g in gates.items():
            if g.get('enabled', True):
                mask |= cumulative_gate_mask(gates, gid, df, overrides=ov)
                anyen = True
        if anyen:
            df = df[mask]
    return df


def _legacy_counts(ed, name, gid):
    from openflo.gating import gate_counts
    s = ed._samples[name]
    return gate_counts(ed._sample_gates[name], gid, s.data,
                       ed._autoclean_overrides(name, s.data))


def _new_counts(ed, name, gid):
    g = ed._sample_gates[name][gid]
    pid = g.get('parent_id')
    if pid and pid in ed._sample_gates[name]:
        c = ed._pc_counts(name, [gid, pid])
        return c[gid], c[pid], 'parent'
    return ed._pc_counts(name, [gid])[gid], len(ed._samples[name].data), 'all'


def _assert_counts(ed, name):
    for gid in ed._sample_gates[name]:
        assert _new_counts(ed, name, gid) == _legacy_counts(ed, name, gid), gid


def _set_view(ed, x='FSC-A', y='SSC-A', mode='dot', display='all',
              ds='Display only', maxpts='60000'):
    ed.x_combo.set(ed._fmt_channel(x))
    ed.y_combo.set(ed._fmt_channel(y))
    ed.mode_var.set(mode)
    ed.gate_display_var.set(display)
    ed.apply_gates_var.set(display == 'filter')
    ed._ds_mode_var.set(ds)
    ed.ds_display_var.set(ds != 'Off')
    ed.ds_propagate_var.set(False)
    ed.max_points_var.set(maxpts)


def _pump(ed, until, timeout=20.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        ed.update()
        if until():
            return
        time.sleep(0.005)
    raise AssertionError('timed out pumping the Tk loop')


def _warm_idle(ed):
    return (getattr(ed, '_pc_inflight', None) is None
            and not getattr(ed, '_pc_warm_names', None)
            and getattr(ed, '_pc_sched_after', None) is None)


# ── same numbers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('display', ['all', 'filter'])
def test_uncapped_frame_is_the_legacy_frame(ed, display):
    """downsample=False: exactly the rows (and order, index, values) the old
    dropna + gate-union path returned."""
    _set_view(ed, display=display)
    for name in ('A', 'B', 'C'):
        new = ed._get_df(name, 'FSC-A', 'SSC-A', downsample=False)
        old = _legacy_frame(ed, name, 'FSC-A', 'SSC-A', display == 'filter')
        pd.testing.assert_frame_equal(new, old)


def test_explicit_population_frame_is_the_legacy_frame(ed):
    from openflo.pipeline import cumulative_gate_mask
    for gid in ('g2', 'g4'):
        new = ed._get_df('C', 'CD3-A', None, downsample=False, gate_parent=gid)
        df = ed._samples['C'].data.dropna(subset=['CD3-A'])
        m = cumulative_gate_mask(ed._sample_gates['C'], gid, df)
        pd.testing.assert_frame_equal(new, df[m])
    whole = ed._get_df('C', 'CD3-A', None, downsample=False, gate_parent=None)
    pd.testing.assert_frame_equal(
        whole, ed._samples['C'].data.dropna(subset=['CD3-A']))


@pytest.mark.parametrize('display', ['all', 'filter'])
def test_capped_subsample_is_deterministic_and_drawn_from_the_population(ed, display):
    _set_view(ed, display=display, maxpts='1000')
    full = _legacy_frame(ed, 'A', 'FSC-A', 'SSC-A', display == 'filter')
    a = ed._get_df('A', 'FSC-A', 'SSC-A')
    assert len(a) == min(1000, len(full))
    assert a.index.is_monotonic_increasing
    assert set(a.index) <= set(full.index)
    pd.testing.assert_frame_equal(a, full.loc[a.index])
    ed._pc_store().purge()                          # recomputed from scratch
    b = ed._get_df('A', 'FSC-A', 'SSC-A')
    pd.testing.assert_frame_equal(a, b)             # no jitter between replots
    c = ed._get_df('A', 'FSC-A', 'CD3-A')
    assert not a.index.equals(c.index)              # seeded per axes


def test_gate_counts_match_gate_counts(ed):
    for name in ('A', 'B', 'C'):
        _assert_counts(ed, name)


def test_gate_row_status_text_is_unchanged(ed):
    from openflo.gating import format_gate_count
    ed._show_gate_count('C', 'g4')
    n, p, of = _legacy_counts(ed, 'C', 'g4')
    assert ed.status_var.get() == format_gate_count('CD4+', n, p, of)


def test_histogram_values_unchanged(ed):
    """The drawn histogram (Count mode) equals np.histogram over every finite
    event on the legacy edges, cold and from the cache."""
    _set_view(ed, x='CD3-A', mode='histogram', display='filter')
    ed.hist_y_mode.set('Count')
    from scipy.ndimage import gaussian_filter1d
    df = _legacy_frame(ed, 'A', 'CD3-A', None, True)
    arr = np.asarray(df['CD3-A'].values, dtype=float)
    arr = arr[np.isfinite(arr)]
    a, b = np.percentile(arr, (0.1, 99.9))
    pad = (b - a) * 0.02
    lo, hi = float(a) - pad, float(b) + pad
    edges = np.asarray(ed._screen_uniform_edges('CD3-A', lo, hi, 256,
                                                data_sample=arr), dtype=float)
    counts = np.histogram(arr, bins=edges)[0].astype(float)
    sigma = float(np.clip(np.sqrt(1.0 / max(arr.size / 256.0, 1e-6)) * 1.5,
                          1.0, 4.0))
    want = gaussian_filter1d(counts, sigma=sigma, mode='constant')
    for _ in range(2):                              # cold, then cached
        ed._replot()
        lines = [ln for ln in ed.ax.get_lines() if ln.get_label() == 'A']
        assert lines
        np.testing.assert_array_equal(lines[0].get_ydata(), want)
        assert ed.ax.get_xlim() == (lo, hi)


def test_highlight_asks_for_the_frame_once(ed, monkeypatch):
    _set_view(ed, display='highlight')
    calls = []
    real = ed._pc_rows
    monkeypatch.setattr(ed, '_pc_rows',
                        lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    ed._replot()
    assert calls.count('A') == 1


def test_scan_in_slices_equals_whole_sample():
    """The chunked pass (bounded GIL time in the worker) gives the masks and
    counts of one whole-sample evaluation, whatever the slice size."""
    from openflo import plotcache as pcm
    from openflo.pipeline import cumulative_gate_mask
    df = _frame(5003, 7, nan_frac=0.03)
    gates = _gates()
    whole = {g: np.asarray(cumulative_gate_mask(gates, g, df), dtype=bool)
             for g in gates}
    for chunk in (8, 64, 1000, pcm.CHUNK):
        res = pcm.scan(df, gates=gates, count_gids=list(gates),
                       keep_gids=['g2', 'g3'], union_gids=['g4', 'g5'],
                       valid_cols=['SSC-A', 'CD3-A'], chunk=chunk)
        assert res['counts'] == {g: int(m.sum()) for g, m in whole.items()}
        for g in ('g2', 'g3'):
            np.testing.assert_array_equal(
                pcm.unpack(res['masks'][g], len(df)), whole[g])
        u = whole['g4'] | whole['g5']
        np.testing.assert_array_equal(pcm.unpack(res['union'][0], len(df)), u)
        assert res['union'][1] == int(u.sum())
        v = df[['SSC-A', 'CD3-A']].notna().all(axis=1).to_numpy()
        np.testing.assert_array_equal(pcm.unpack(res['valid'][0], len(df)), v)
        elig = pcm.pack(u & v)
        a = pcm.pick_rows(len(df), elig, int((u & v).sum()), 300, 11, chunk=chunk)
        b = pcm.pick_rows(len(df), elig, int((u & v).sum()), 300, 11)
        np.testing.assert_array_equal(a, b)
        assert np.all((u & v)[a]) and np.all(np.diff(a) > 0) and a.size == 300


def test_boolean_over_autoclean_is_evaluated_whole(ed):
    """Not per-event (auto-clean is computed over the whole sample, and a
    boolean evaluates its operands without the override): no slicing, and
    the filter keeps the old evaluate-on-valid-rows semantics."""
    from openflo.pipeline import default_autoclean_methods
    g = ed._sample_gates['C']
    g['ac'] = {'kind': 'autoclean', 'methods': default_autoclean_methods(),
               'parent_id': None, 'name': 'clean', 'enabled': True}
    g['g1']['parent_id'] = 'ac'
    g['b1'] = {'kind': 'boolean', 'op': 'not', 'operands': ['ac'],
               'parent_id': None, 'name': 'dirty', 'enabled': True}
    _assert_counts(ed, 'C')
    _set_view(ed, display='filter')
    pd.testing.assert_frame_equal(
        ed._get_df('C', 'FSC-A', 'SSC-A', downsample=False),
        _legacy_frame(ed, 'C', 'FSC-A', 'SSC-A', True))


# ── never stale ──────────────────────────────────────────────────────────────

def _scan_counter(monkeypatch):
    import openflo.plotcache as pcm
    calls = []
    real = pcm.scan
    monkeypatch.setattr(pcm, 'scan',
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def test_gate_edits_invalidate(ed):
    _assert_counts(ed, 'A')
    g = ed._sample_gates['A']
    g['g3']['value'] = 0.6                                  # move
    _assert_counts(ed, 'A')
    g['g2']['vertices'][0] = [20.0, 20.0]                   # vertex drag
    _assert_counts(ed, 'A')
    g['g6'] = {'kind': 'threshold', 'channel': 'SSC-A', 'value': 40.0,
               'parent_id': 'g1', 'name': 'SSChi', 'enabled': True}  # add
    _assert_counts(ed, 'A')
    g['g4']['parent_id'] = 'g1'                             # reparent
    _assert_counts(ed, 'A')
    del g['g1']                                             # delete a root
    _assert_counts(ed, 'A')


def test_rename_and_colour_reuse_the_cache(ed, monkeypatch):
    _assert_counts(ed, 'A')
    calls = _scan_counter(monkeypatch)
    g = ed._sample_gates['A']
    g['g3']['name'] = 'T cells'
    g['g3']['color'] = '#123456'
    g['g3']['open'] = False
    _assert_counts(ed, 'A')
    assert calls == []


def test_enable_toggle_changes_the_filter_not_the_counts(ed, monkeypatch):
    _set_view(ed, display='filter')
    before = ed._get_df('A', 'FSC-A', 'SSC-A', downsample=False)
    _assert_counts(ed, 'A')
    for gid in ('g1', 'g2', 'g3', 'g4', 'g5'):
        ed._sample_gates['A'][gid]['enabled'] = gid in ('g4',)
    after = ed._get_df('A', 'FSC-A', 'SSC-A', downsample=False)
    pd.testing.assert_frame_equal(
        after, _legacy_frame(ed, 'A', 'FSC-A', 'SSC-A', True))
    assert len(after) < len(before)
    calls = _scan_counter(monkeypatch)
    _assert_counts(ed, 'A')
    assert calls == []                  # counts don't depend on the toggle


def test_undo_restores_and_counts_follow(ed):
    _assert_counts(ed, 'A')
    ed._checkpoint()
    ed._sample_gates['A']['g3']['value'] = 0.9
    ed._undo_pending = False
    _assert_counts(ed, 'A')
    ed._undo()
    assert ed._sample_gates['A']['g3']['value'] == 0.45
    _assert_counts(ed, 'A')


def test_paste_gate_tree_to_another_sample(ed):
    _assert_counts(ed, 'B')
    ed._sample_gates['B']['g3']['value'] = 0.7
    ed._sample_gates['B']['g9'] = dict(ed._sample_gates['A']['g4'], parent_id='g2')
    _assert_counts(ed, 'B')


def test_data_replaced_invalidates_without_a_hook(ed):
    _assert_counts(ed, 'B')
    rows0 = ed._get_df('B', 'FSC-A', 'SSC-A', downsample=False)
    s = ed._samples['B']
    s.data = s.data.iloc[::2].reset_index(drop=True)        # a QC-style trim
    _assert_counts(ed, 'B')
    rows1 = ed._get_df('B', 'FSC-A', 'SSC-A', downsample=False)
    assert len(rows1) == len(s.data) != len(rows0)


def test_column_rewritten_invalidates_without_a_hook(ed):
    _assert_counts(ed, 'B')
    s = ed._samples['B']
    s.data['CD3-A'] = s.data['CD3-A'] + 0.25                # re-transform-like
    _assert_counts(ed, 'B')


def test_in_place_write_with_the_hook(ed):
    """A partial in-place write keeps the column's buffer; the editor paths
    that do that call _data_changed."""
    _assert_counts(ed, 'B')
    s = ed._samples['B']
    s.data.loc[s.data.index[:500], 'CD4-A'] = 5.0
    ed._data_changed('B')
    _assert_counts(ed, 'B')


def test_transform_editor_apply_invalidates(ed):
    _assert_counts(ed, 'A')
    _set_view(ed, display='filter')
    ed._get_df('A', 'CD3-A', 'CD4-A', downsample=False)
    ed._apply_channel_transforms({'CD3-A': 'asinh'})
    _assert_counts(ed, 'A')
    pd.testing.assert_frame_equal(
        ed._get_df('A', 'CD3-A', 'CD4-A', downsample=False),
        _legacy_frame(ed, 'A', 'CD3-A', 'CD4-A', True))


def test_remove_sample_leaves_nothing(ed):
    _assert_counts(ed, 'B')
    ed._get_df('B', 'FSC-A', 'SSC-A')
    assert 'B' in ed._pc_store().names()
    ed._remove_samples(['B'])
    assert 'B' not in ed._pc_store().names()
    assert 'B' not in ed._pc_explicit and 'B' not in ed._pc_tokens


def test_reload_under_the_same_name_starts_clean(ed):
    _assert_counts(ed, 'B')
    from openflo.pipeline import FlowSample
    ed._on_loaded('B', FlowSample.from_dataframe(_frame(3000, 9), name='B'))
    assert 'B' not in ed._pc_store().names()
    ed._sample_gates['B'] = _gates()
    _assert_counts(ed, 'B')


def test_tree_rebuild_echo_does_not_recount(ed, monkeypatch):
    ed.gate_tv.selection_set(ed._gate_iid('A', 'g4'))
    _pump(ed, lambda: getattr(ed, '_tree_select_echo', None) is None, 5)
    seen = []
    monkeypatch.setattr(ed, '_show_gate_count', lambda *a: seen.append(a))
    ed._refresh_gate_list()
    _pump(ed, lambda: getattr(ed, '_tree_select_echo', None) is None, 5)
    assert seen == []
    ed.gate_tv.selection_set(ed._gate_iid('A', 'g3'))     # a real click
    _pump(ed, lambda: bool(seen), 5)
    assert seen == [('A', 'g3')]


# ── background warm-up ───────────────────────────────────────────────────────

def _warm(ed):
    ed._pc_warm_enabled = True
    ed._pc_closed = False
    ed._warm_schedule(0)
    _pump(ed, lambda: _warm_idle(ed))


def test_warm_up_fills_the_samples_not_shown(ed, monkeypatch):
    _set_view(ed, display='filter')
    _warm(ed)
    assert ed._pc_worker is not None and ed._pc_worker.thread.is_alive()
    calls = _scan_counter(monkeypatch)
    for name in ('B', 'C'):
        _assert_counts(ed, name)               # served from the cache
        ed._get_df(name, 'FSC-A', 'SSC-A')     # rows warmed too
    assert calls == []
    # The warm results equal the foreground ones.
    for name in ('B', 'C'):
        pd.testing.assert_frame_equal(
            ed._get_df(name, 'FSC-A', 'SSC-A', downsample=False),
            _legacy_frame(ed, name, 'FSC-A', 'SSC-A', True))


def test_warm_up_histogram_counts(ed, monkeypatch):
    _set_view(ed, x='CD3-A', mode='histogram')
    _warm(ed)
    seen = []
    real = np.histogram
    monkeypatch.setattr(np, 'histogram',
                        lambda *a, **k: seen.append(1) or real(*a, **k))
    ed._sample_plot_enabled['A'] = False
    ed._sample_plot_enabled['C'] = True
    ed._replot()                                 # C shown alone: all cached
    assert seen == []


def test_stale_background_result_is_discarded(ed, monkeypatch):
    """Data changes while the worker computes: its result must not be kept."""
    import openflo.plotcache as pcm
    gate = threading.Event()
    started = threading.Event()
    real = pcm.scan

    def slow(*a, **k):
        started.set()
        gate.wait(5)
        return real(*a, **k)
    monkeypatch.setattr(pcm, 'scan', slow)
    ed._pc_warm_enabled = True
    ed._warm_schedule(0)
    _pump(ed, started.is_set)
    name = ed._pc_inflight['name']
    s = ed._samples[name]
    s.data = s.data.assign(**{'CD3-A': s.data['CD3-A'] * 2.0})
    gate.set()
    monkeypatch.setattr(pcm, 'scan', real)
    _pump(ed, lambda: _warm_idle(ed))
    _assert_counts(ed, name)


def test_newer_invalidation_cancels_the_running_task(ed, monkeypatch):
    import openflo.plotcache as pcm
    started = threading.Event()
    real = pcm.scan

    def slow(*a, **k):
        cp = k.get('checkpoint')
        started.set()
        for _ in range(400):
            if cp is not None:
                cp()
            time.sleep(0.01)
        return real(*a, **k)
    monkeypatch.setattr(pcm, 'scan', slow)
    ed._pc_warm_enabled = True
    ed._warm_schedule(0)
    _pump(ed, started.is_set)
    first = ed._pc_inflight
    ed._sample_gates[first['name']]['g3']['value'] = 0.8   # its gates changed
    monkeypatch.setattr(pcm, 'scan', real)
    t0 = time.time()
    ed._warm_replan()
    _pump(ed, lambda: (ed._pc_inflight or {}).get('tid') != first['tid'])
    assert time.time() - t0 < 2.0                   # cancelled, not waited out
    _pump(ed, lambda: _warm_idle(ed))
    _assert_counts(ed, first['name'])


def test_replan_keeps_wanted_work_running(ed, monkeypatch):
    import openflo.plotcache as pcm
    started, gate = threading.Event(), threading.Event()
    real = pcm.scan

    def slow(*a, **k):
        started.set()
        gate.wait(5)
        return real(*a, **k)
    monkeypatch.setattr(pcm, 'scan', slow)
    ed._pc_warm_enabled = True
    ed._warm_schedule(0)
    _pump(ed, started.is_set)
    tid = ed._pc_inflight['tid']
    for _ in range(3):
        ed._warm_replan()                         # nothing changed
    assert ed._pc_inflight['tid'] == tid          # not restarted / duplicated
    gate.set()
    monkeypatch.setattr(pcm, 'scan', real)
    _pump(ed, lambda: _warm_idle(ed))


def test_destroy_stops_the_worker_and_drops_the_cache(monkeypatch):
    root, editor, _gui = _editor_or_skip()
    try:
        _add(editor, 'A', _frame(3000, 1), shown=True)
        _add(editor, 'B', _frame(3000, 2))
        editor._channels = list(CHANS)
        editor._channel_labels = {c: c for c in CHANS}
        editor._populate_channel_combos()
        editor._set_active_sample('A')
        editor._warm_schedule(0)
        _pump(editor, lambda: _warm_idle(editor))
        w = editor._pc_worker
        assert w is not None and w.thread.is_alive()
        assert len(editor._pc_store())
        editor.destroy()
        root.update()
        assert not w.thread.is_alive()
        assert len(editor._pc_store()) == 0
    finally:
        root.destroy()


# ── every data write is hooked ───────────────────────────────────────────────

# FlowSample methods that rewrite a sample's data in place.
_MUTATORS = {'retransform', 'apply_transform', 'run_qc', '_apply_comp',
             'manual_compensate', 'cluster', 'run_leiden', 'run_louvain',
             'run_flowsom', 'run_umap', 'run_tsne', 'run_trimap', 'run_pacmap',
             'cell_cycle', 'apply_threshold_gates', 'apply_region_gates'}

# Functions that write sample data WITHOUT invalidating in the same function,
# and why that is right.
_EXEMPT = {
    # Off-thread clustering: _finish_clustering / _clustering_error
    # invalidate on the Tk thread when the run ends.
    ('editor_analysis.py', 'work'),
    # Load workers build a FRESH sample; _on_loaded drops the name's caches.
    ('editor_loadpool.py', '_load_worker'),
    # A staticmethod _on_loaded calls on the sample that just arrived, after
    # dropping the name's caches and before anything reads it.
    ('editor_compute.py', '_restore_sample_transforms'),
}


def test_every_editor_data_write_invalidates_the_plot_cache():
    """Static guard: in the editor and its dialogs, a function that writes
    ``<sample>.data`` (assign, item-assign, or a FlowSample mutator) also calls
    ``_data_changed`` -- or is listed above with the reason it need not."""
    import ast
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'
    offenders = []
    for path in sorted(src.glob('*.py')):
        if not (path.name.startswith(('editor_', 'ui_')) or path.name == 'gui.py'):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            own = [n for n in ast.walk(fn)]
            # Nodes of nested defs belong to them, not to fn.
            nested = {id(x) for d in ast.walk(fn)
                      if d is not fn and isinstance(d, (ast.FunctionDef,
                                                        ast.AsyncFunctionDef))
                      for x in ast.walk(d)}
            own = [n for n in own if id(n) not in nested]
            writes = False
            for n in own:
                targets = (n.targets if isinstance(n, ast.Assign) else
                           [n.target] if isinstance(n, ast.AugAssign) else [])
                for t in targets:
                    base = t.value if isinstance(t, ast.Subscript) else t
                    if isinstance(base, ast.Attribute) and base.attr == 'data':
                        writes = True
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr in _MUTATORS):
                    writes = True
            if not writes:
                continue
            hooked = any(isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute)
                         and n.func.attr == '_data_changed' for n in own)
            if not hooked and (path.name, fn.name) not in _EXEMPT:
                offenders.append(f'{path.name}:{fn.lineno} {fn.name}')
    assert not offenders, ('writes sample data without _data_changed:\n  '
                           + '\n  '.join(offenders))


def test_compensation_editor_apply_invalidates(tmp_path):
    """Tools -> Compensation -> Apply replaces the active sample's data: the
    gate counts afterwards are the new data's."""
    from test_compensation_editor_apply import (
        FL,
        M,
        _comp_window,
        _load,
        _write_mixed,
    )
    from test_compensation_editor_apply import _editor_or_skip as _ed
    root, ed = _ed()
    try:
        _write_mixed(tmp_path / 'a.fcs')
        _load(root, ed, [('a', tmp_path / 'a.fcs')])
        ed._set_active_sample('a')
        ed._sample_gates['a'] = {
            'g1': {'kind': 'threshold', 'channel': 'FLB-A', 'value': 0.6,
                   'parent_id': None, 'name': 'FLB+', 'enabled': True},
            'g2': {'kind': 'threshold', 'channel': 'FLA-A', 'value': 0.6,
                   'parent_id': 'g1', 'name': 'FLA+', 'enabled': True}}
        ed._gates = ed._sample_gates['a']
        before = {g: _legacy_counts(ed, 'a', g) for g in ('g1', 'g2')}
        assert {g: _new_counts(ed, 'a', g) for g in ('g1', 'g2')} == before
        win = _comp_window(ed)
        win._set_matrix(FL, M)
        win._apply()
        after = {g: _legacy_counts(ed, 'a', g) for g in ('g1', 'g2')}
        assert after != before                  # compensation moved events
        assert {g: _new_counts(ed, 'a', g) for g in ('g1', 'g2')} == after
    finally:
        root.destroy()
