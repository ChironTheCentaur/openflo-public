"""Population export writes in the background and stops on Cancel, and it
reuses the plot cache's auto-clean masks: each sample is cleaned once,
and no mask is kept for data that was replaced.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import threading

import numpy as np

import tests._background_work as shared
from openflo.async_task import wait_for_job
from tests._background_work import _count, _gated, _on_main, _quiet
from tests.test_beta_look import _load, _write_fcs

env = shared.env


def test_population_export_writes_off_the_window_thread(env, monkeypatch):
    import openflo.fcs_export as fe
    from openflo import editor_populations
    root, ed, tmp, _ = env
    gid = _gated(root, ed, tmp)
    threads = []
    real = fe.population_export_frame
    monkeypatch.setattr(fe, 'population_export_frame', lambda *a, **k: (
        threads.append(_on_main()), real(*a, **k))[1])
    path = tmp / 'one.fcs'
    monkeypatch.setattr(editor_populations.filedialog, 'asksaveasfilename',
                        lambda **k: str(path))
    ed._export_population_fcs('sample_a', gid)
    assert wait_for_job(ed._job_strip)
    assert threads == [False] and path.exists()
    assert f"Exported {_count(ed, 'sample_a', gid):,} events" in (
        ed.status_var.get())
    out = tmp / 'bulk'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs()
    assert wait_for_job(ed._job_strip)
    assert threads == [False, False]
    assert [p.name for p in out.glob('*.fcs')] == ['Cells.fcs']


def test_population_export_cancel_writes_nothing(env, monkeypatch):
    import openflo.fcs_export as fe
    from openflo import editor_populations
    root, ed, tmp, _ = env
    _gated(root, ed, tmp)
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                  'value': 5_000.0, 'name': 'FL1 pos', 'parent_id': None})
    hold = threading.Event()
    real = fe.population_export_frame
    monkeypatch.setattr(fe, 'population_export_frame', lambda *a, **k: (
        hold.wait(10), real(*a, **k))[1])
    out = tmp / 'bulk'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs()
    root.update()
    ed._job_strip.cancel()
    hold.set()
    assert wait_for_job(ed._job_strip)
    assert list(out.iterdir()) == []
    assert 'cancelled' in ed.status_var.get()


_AC_POPS = ('Cells', 'FL1pos', 'FL2hi')


def _autoclean_tree(root, ed, tmp, names=('sample_a',)):
    """Each of ``names`` loaded with an auto-clean root (a manual FSC floor,
    so it really removes events) and three populations under it: Cells (rect)
    and FL2hi (threshold) on the root, FL1pos (threshold) inside Cells.
    Returns ``{name: {'ac': gid, pop: gid}}``, with the background warm-up
    done and nothing left scheduled."""
    from openflo.autoclean import default_autoclean_methods
    _load(root, ed, [(n, _write_fcs(tmp / f'{n}.fcs', seed=5 + i))
                     for i, n in enumerate(names)])
    ids = {}
    for n in names:
        ed._set_active_sample(n)
        methods = default_autoclean_methods()
        for m in methods:
            if m['key'] == 'debris':
                m['params'] = {'min_fsc': 50_000.0}
        ac = ed._add_gate({'kind': 'autoclean', 'name': 'autocleaned sample',
                           'parent_id': None, 'open': False,
                           'methods': methods})
        cells = ed._add_gate({
            'kind': 'rect', 'name': 'Cells', 'x_channel': 'FSC-A',
            'y_channel': 'SSC-A', 'x0': 20_000.0, 'x1': 180_000.0,
            'y0': 5_000.0, 'y1': 100_000.0, 'parent_id': ac})
        # the editor holds fluorescence on its display scale: cut at quantiles
        d = ed._samples[n].data
        pos = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                            'value': float(np.quantile(d['FL1-A'], 0.7)),
                            'name': 'FL1pos',
                            'parent_id': cells})
        hi = ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A',
                           'value': float(np.quantile(d['FL2-A'], 0.5)),
                           'name': 'FL2hi',
                           'parent_id': ac})
        ids[n] = {'ac': ac, 'Cells': cells, 'FL1pos': pos, 'FL2hi': hi}
    ed._set_active_sample(names[0])
    _quiet(root, ed)
    return ids


def _autoclean_spy(monkeypatch):
    """Count ``autoclean.autoclean_keep_mask`` calls made on behalf of the
    population export (any call with an editor_populations frame on its
    stack: the export calling it directly, or through gate_to_mask /
    cumulative_gate_mask), apart from those of a replot or the plot cache's
    warm-up. The export imports it inside the function, so patching the
    autoclean attribute reaches it, and patching gate_eval's reaches
    gate_to_mask. Returns the list of the export's calls (each the number of
    events cleaned)."""
    import sys

    from openflo import autoclean, gate_eval
    real = autoclean.autoclean_keep_mask
    export_calls = []

    def spy(gate, df):
        f = sys._getframe(1)
        while f is not None:
            if f.f_code.co_filename.endswith('editor_populations.py'):
                export_calls.append(len(df))
                break
            f = f.f_back
        return real(gate, df)
    monkeypatch.setattr(autoclean, 'autoclean_keep_mask', spy)
    monkeypatch.setattr(gate_eval, 'autoclean_keep_mask', spy)
    return export_calls


class _WriterLog(dict):
    """An ``_autoclean_cache`` that remembers which functions wrote each entry, so
    a test can tell the export's own caching from a replot or the plot
    cache's warm-up filling the same entry meanwhile."""

    def __init__(self, *a):
        super().__init__(*a)
        self.writers = {}

    def __setitem__(self, key, value):
        import sys
        self.writers.setdefault(key, set()).add(sys._getframe(1).f_code.co_name)
        super().__setitem__(key, value)


def _log_cache_writers(ed):
    ed._autoclean_cache = _WriterLog()
    return ed._autoclean_cache


def _old_count(ed, name, gid):
    """The population's size the way the export used to make it: the whole
    chain evaluated afresh, auto-clean included, with no cached masks."""
    from openflo.gate_eval import cumulative_gate_mask
    gates = ed._sample_gates[name]
    assert any(g.get('kind') == 'autoclean' for g in gates.values())
    return int(np.asarray(cumulative_gate_mask(
        gates, gid, ed._samples[name].data), dtype=bool).sum())


def _fcs_events(path):
    from openflo.pipeline import FlowSample
    return len(FlowSample(str(path)).data)


def _warm(ed, name):
    """Fill ``_autoclean_cache`` for ``name`` on the Tk thread, as a plot does."""
    ed._autoclean_cache.clear()
    ed._autoclean_overrides(name, ed._samples[name].data)
    assert any(k[0] == name for k in ed._autoclean_cache)


def _export_all(ed, monkeypatch, out):
    from openflo import editor_populations
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs(all_samples=True)
    assert wait_for_job(ed._job_strip)


def _written(out, name):
    return {p.stem: _fcs_events(p) for p in (out / name).glob('*.fcs')}


def test_population_export_reuses_the_cached_auto_clean_masks(env,
                                                              monkeypatch):
    root, ed, tmp, _ = env
    names = ('sample_a', 'sample_b')
    ids = _autoclean_tree(root, ed, tmp, names)
    want = {n: {p: _old_count(ed, n, ids[n][p]) for p in _AC_POPS}
            for n in names}
    assert all(0 < c for w in want.values() for c in w.values())
    # the floor really cleans, so a population that skipped it would differ
    from openflo.autoclean import autoclean_keep_mask
    raw = ed._samples['sample_a'].data
    keep = np.asarray(autoclean_keep_mask(
        ed._sample_gates['sample_a'][ids['sample_a']['ac']], raw), dtype=bool)
    assert 0 < keep.sum() < len(raw)
    ed._autoclean_cache.clear()
    for n in names:                            # as plotting each one does
        ed._autoclean_overrides(n, ed._samples[n].data)
    assert {k[0] for k in ed._autoclean_cache} == set(names)
    export_calls = _autoclean_spy(monkeypatch)
    out = tmp / 'bulk'
    _export_all(ed, monkeypatch, out)
    assert export_calls == []                  # the cleaning is not re-run
    for n in names:
        assert _written(out, n) == want[n]     # the same events as before
    assert 'Exported 6 population FCS file(s)' in ed.status_var.get()


def test_population_export_cleans_once_per_sample_and_caches_it(env,
                                                                monkeypatch):
    root, ed, tmp, _ = env
    names = ('sample_a', 'sample_b')
    ids = _autoclean_tree(root, ed, tmp, names)
    want = {n: {p: _old_count(ed, n, ids[n][p]) for p in _AC_POPS}
            for n in names}
    cache = _log_cache_writers(ed)             # starts empty: cold
    export_calls = _autoclean_spy(monkeypatch)
    out = tmp / 'cold'
    _export_all(ed, monkeypatch, out)
    # once per sample -- not once per population on the auto-clean root
    assert export_calls == [len(ed._samples[n].data) for n in names]
    for n in names:
        assert _written(out, n) == want[n]
        ac = ids[n]['ac']
        assert '_autoclean_store_masks' in cache.writers.get((n, ac), ())
        ent = ed._autoclean_cache[(n, ac)]
        assert ent[0] == id(ed._samples[n].data)
        np.testing.assert_array_equal(
            ent[2].to_numpy(), np.asarray(ed._autoclean_overrides(
                n, ed._samples[n].data)[ac], dtype=bool))
    # the masks it made are reused by the next export
    del export_calls[:]
    out2 = tmp / 'again'
    _export_all(ed, monkeypatch, out2)
    assert export_calls == []
    for n in names:
        assert _written(out2, n) == want[n]


def test_population_export_keeps_no_mask_of_replaced_data(env, monkeypatch):
    """A sample whose data changed while the export ran gets no cached mask
    from it (the mask belongs to the old data)."""
    root, ed, tmp, _ = env
    ids = _autoclean_tree(root, ed, tmp)
    ed._autoclean_cache.clear()
    s = ed._samples['sample_a']
    gates = ed._sample_gates['sample_a']
    ac = ids['sample_a']['ac']
    old = s.data
    m = np.ones(len(old), dtype=bool)
    s.data = old.copy()
    try:
        ed._autoclean_store_masks('sample_a', old, gates, {ac: m})
        assert ed._autoclean_cache == {}
    finally:
        s.data = old
    ed._autoclean_store_masks('sample_a', old, gates, {ac: m})
    assert set(ed._autoclean_cache) == {('sample_a', ac)}
    assert set(ed._autoclean_cached_masks('sample_a')) == {ac}
    # an edited recipe no longer matches what was stored
    gates[ac]['methods'][0]['enabled'] = not gates[ac]['methods'][0].get(
        'enabled', True)
    assert ed._autoclean_cached_masks('sample_a') == {}


def test_single_population_export_reuses_the_cached_auto_clean_mask(
        env, monkeypatch):
    from openflo import editor_populations
    root, ed, tmp, _ = env
    ids = _autoclean_tree(root, ed, tmp)
    gid = ids['sample_a']['FL1pos']
    ac = ids['sample_a']['ac']
    want = _old_count(ed, 'sample_a', gid)
    paths = iter([tmp / 'warm.fcs', tmp / 'cold.fcs', tmp / 'after.fcs'])
    monkeypatch.setattr(editor_populations.filedialog, 'asksaveasfilename',
                        lambda **k: str(next(paths)))
    export_calls = _autoclean_spy(monkeypatch)
    # warm: the plot's cached mask is used, the cleaning is not re-run
    _warm(ed, 'sample_a')
    ed._export_population_fcs('sample_a', gid)
    assert wait_for_job(ed._job_strip)
    assert export_calls == []
    assert _fcs_events(tmp / 'warm.fcs') == want
    assert f"Exported {want:,} events" in ed.status_var.get()
    # cold: computed once, and the cache keeps it
    cache = _log_cache_writers(ed)
    ed._export_population_fcs('sample_a', gid)
    assert wait_for_job(ed._job_strip)
    assert export_calls == [len(ed._samples['sample_a'].data)]
    assert _fcs_events(tmp / 'cold.fcs') == want
    assert '_autoclean_store_masks' in cache.writers.get(
        ('sample_a', ac), ())
    # ... so the next export of the population runs no cleaning
    ed._export_population_fcs('sample_a', gid)
    assert wait_for_job(ed._job_strip)
    assert export_calls == [len(ed._samples['sample_a'].data)]
    assert _fcs_events(tmp / 'after.fcs') == want
