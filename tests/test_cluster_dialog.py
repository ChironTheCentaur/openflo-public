"""The editor's Cluster… dialog and the run it starts:
editor_analysis._open_cluster_dialog -> _run_clustering -> (worker) ->
editor_compute._finish_clustering / _clustering_error.

All of it was 1-6% covered. The dialog is driven through its own widgets and
Run button (so a wrong keyword or signature between the dialog and
_run_clustering fails here), and the run itself goes through the real sample
methods with run_async stubbed inline, as tests/test_gui_ux.py does.
"""
import contextlib
import io
import os
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from tests.conftest import gui_unavailable

CH = ['CD1', 'CD2', 'CD3']


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
        _inline(work, on_done, on_error))
    ed._audited = []
    ed._audit = lambda action, **d: ed._audited.append((action, d))
    return root, ed


def _inline(work, on_done, on_error):
    try:
        r = work()
    except Exception as e:                       # noqa: BLE001
        if on_error:
            on_error(e)
        return
    if on_done:
        on_done(r)


def _blobs(sizes=(200, 120, 80), seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(8.0 * i, 0.4, (n, 3)) for i, n in enumerate(sizes)])
    return pd.DataFrame(X, columns=CH), np.repeat(np.arange(len(sizes)), sizes)


def _add(ed, df, name='s1'):
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(df, name=name, path=rf'C:\exp\{name}.fcs')
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(df.columns)
    ed._channel_labels = {c: c for c in df.columns}
    ed._populate_channel_combos()
    ed._set_active_sample(name)
    return s


def _walk(w):
    stack = [w]
    while stack:
        x = stack.pop()
        yield x
        stack.extend(x.winfo_children())


def _dialog(ed, title):
    return next(w for w in ed.winfo_children()
                if isinstance(w, tk.Toplevel) and w.title().startswith(title))


def _set(widget, value):
    widget.delete(0, 'end')
    widget.insert(0, str(value))


def test_dialog_run_forwards_what_the_user_chose():
    root, ed = _editor_or_skip()
    try:
        df, _ = _blobs()
        _add(ed, df)
        got = {}
        ed._run_clustering = lambda **kw: got.update(kw)
        ed._open_cluster_dialog()
        dlg = _dialog(ed, 'Cluster')
        ws = list(_walk(dlg))
        next(w for w in ws if isinstance(w, ttk.Radiobutton)
             and w.cget('text') == 'Leiden').invoke()
        spin = {(int(w.grid_info()['row']), int(w.grid_info()['column'])): w
                for w in ws if isinstance(w, ttk.Spinbox) and w.grid_info()}
        _set(spin[(1, 1)], 12)                        # k
        _set(spin[(1, 3)], 0.7)                       # Leiden resolution
        _set(spin[(2, 1)], 6)                         # FlowSOM grid
        _set(spin[(3, 1)], 4)                         # FlowSOM metaclusters
        lb = next(w for w in ws if isinstance(w, tk.Listbox))
        assert lb.get(0, 'end') == tuple(CH)
        lb.selection_clear(2)                         # drop CD3
        # The cap applies only with 'Custom…' selected: typing in the entry
        # fires no <Key> binding from a test, so choose it explicitly.
        (ds,) = [w for w in ws if isinstance(w, ttk.Combobox)
                 and 'Custom…' in tuple(w.cget('values'))]
        ds.set('Custom…')
        _set(next(w for w in ws if isinstance(w, ttk.Entry)
                  and not isinstance(w, (ttk.Spinbox, ttk.Combobox))), 150)
        next(w for w in ws if isinstance(w, ttk.Button)
             and w.cget('text') == 'Run').invoke()
        assert got['method'] == 'leiden'
        assert got['k'] == 12 and got['resolution'] == 0.7
        assert got['grid'] == 6 and got['n_meta'] == 4
        assert got['channels'] == ['CD1', 'CD2']
        assert (got['downsample'], got['custom_n']) == ('Custom…', '150')
        assert not dlg.winfo_exists()
    finally:
        root.destroy()


def test_dialog_with_all_markers_selected_passes_none():
    root, ed = _editor_or_skip()
    try:
        _add(ed, _blobs()[0])
        got = {}
        ed._run_clustering = lambda **kw: got.update(kw)
        ed._open_cluster_dialog()
        dlg = _dialog(ed, 'Cluster')
        next(w for w in _walk(dlg) if isinstance(w, ttk.Button)
             and w.cget('text') == 'Run').invoke()
        assert got['channels'] is None and got['method'] == 'phenograph'
        assert got['k'] == 30 and got['downsample'] == 'Full'
    finally:
        root.destroy()


def test_leiden_run_imports_leiden_populations():
    root, ed = _editor_or_skip()
    try:
        df, truth = _blobs()
        s = _add(ed, df)
        ed._run_clustering(method='leiden', all_samples=False, k=15, grid=10,
                           n_meta=10, embedding='none', resolution=0.3)
        lab = s.data['leiden'].to_numpy()
        for c in np.unique(lab):
            assert np.bincount(truth[lab == c]).max() / (lab == c).sum() > 0.98
        pops = {g['value'] for g in ed._sample_gates['s1'].values()
                if g.get('kind') == 'category' and g.get('channel') == 'leiden'}
        assert pops == set(np.unique(lab).tolist())
        assert ed._clustering_busy is False
        assert ed._audited[-1][0] == 'cluster'
        assert ed._audited[-1][1]['method'] == 'leiden'
        assert ed._audited[-1][1]['column'] == 'leiden'
        assert "populations imported from 'leiden'" in ed.status_var.get()
    finally:
        root.destroy()


def test_flowsom_run_uses_dialog_grid_and_metaclusters():
    root, ed = _editor_or_skip()
    try:
        df, truth = _blobs(sizes=(300, 300, 300))
        s = _add(ed, df)
        ed._run_clustering(method='flowsom', all_samples=False, k=30, grid=4,
                           n_meta=3, embedding='none')
        assert s.flowsom_result['grid'] == (4, 4)
        assert s.flowsom_result['n_metaclusters'] == 3
        meta = s.data['flowsom_meta'].to_numpy()
        from sklearn.metrics import adjusted_rand_score
        assert adjusted_rand_score(truth, meta) == 1.0
        pops = {g['value'] for g in ed._sample_gates['s1'].values()
                if g.get('kind') == 'category'
                and g.get('channel') == 'flowsom_meta'}
        assert pops == {0, 1, 2}
    finally:
        root.destroy()


def test_phenograph_run_imports_cluster_gates_and_honours_the_cap():
    root, ed = _editor_or_skip()
    try:
        df, truth = _blobs()
        s = _add(ed, df)
        seen = {}
        real = type(s).cluster

        def spy(self, **kw):
            seen.update(kw)
            return real(self, **kw)
        s.cluster = spy.__get__(s)
        with contextlib.redirect_stdout(io.StringIO()):
            ed._run_clustering(method='phenograph', all_samples=False, k=15,
                               grid=10, n_meta=10, embedding='none',
                               reproducible=True, downsample='Custom…',
                               custom_n='300')
        assert seen['k'] == 15 and seen['max_events'] == 300
        assert seen['reproducible'] is True
        lab = s.data['cluster'].to_numpy()
        assert (lab >= 0).all()
        ids = {g['cluster_id'] for g in ed._sample_gates['s1'].values()
               if g.get('kind') == 'cluster'}
        assert ids == set(np.unique(lab).tolist())
        assert ed._audited[-1][1]['column'] == 'cluster'
    finally:
        root.destroy()


def test_flowsom_skipping_one_sample_reports_and_audits_only_the_other():
    root, ed = _editor_or_skip()
    try:
        big = _add(ed, _blobs(sizes=(300, 300, 300))[0], name='big')
        small = _add(ed, _blobs(sizes=(30, 20, 10))[0], name='small')
        ed._run_clustering(method='flowsom', all_samples=True, k=30, grid=10,
                           n_meta=3, embedding='none')
        assert 'flowsom_meta' in big.data.columns
        assert 'flowsom_meta' not in small.data.columns      # 60 < 10x10
        action, d = ed._audited[-1]
        assert action == 'cluster' and d['samples'] == ['big']
        assert d['n_samples'] == 1
        st = ed.status_var.get()
        assert st.startswith('flowsom done on 1 sample(s)')
        assert 'No labels on small' in st
    finally:
        root.destroy()


def test_flowsom_skip_over_an_older_column_is_not_reported_as_done():
    """A skipped re-run leaves the previous run's column in place, so column
    presence alone would still claim the run worked."""
    root, ed = _editor_or_skip()
    try:
        s = _add(ed, _blobs(sizes=(30, 20, 10))[0])
        ed._run_clustering(method='flowsom', all_samples=False, k=30, grid=4,
                           n_meta=3, embedding='none')
        assert 'flowsom_meta' in s.data.columns          # 60 >= 4x4: ran
        first = s.data['flowsom_meta'].to_numpy().copy()
        n_audits = len(ed._audited)
        ed._run_clustering(method='flowsom', all_samples=False, k=30, grid=10,
                           n_meta=3, embedding='none')
        assert np.array_equal(s.data['flowsom_meta'].to_numpy(), first)
        assert len(ed._audited) == n_audits               # nothing new audited
        assert 'done' not in ed.status_var.get()
        assert 'produced no labels' in ed.status_var.get()
    finally:
        root.destroy()


def test_no_embedding_runs_on_a_sample_whose_clustering_was_skipped():
    """The embedding belongs to the clustering run: on a skipped sample it
    wrote TSNE1/TSNE2 that no audit or status line accounted for."""
    root, ed = _editor_or_skip()
    try:
        big = _add(ed, _blobs(sizes=(300, 300, 300))[0], name='big')
        small = _add(ed, _blobs(sizes=(30, 20, 10))[0], name='small')
        embedded = []

        def fake_tsne(self, **kw):
            embedded.append(self.name)
            self.data['TSNE1'] = 0.0
            self.data['TSNE2'] = 0.0
        for s in (big, small):
            s.run_tsne = fake_tsne.__get__(s)
        ed._run_clustering(method='flowsom', all_samples=True, k=30, grid=10,
                           n_meta=3, embedding='t-SNE')
        assert embedded == ['big']
        assert 'TSNE1' not in small.data.columns
        assert ed._audited[-1][1]['samples'] == ['big']
        assert ed._audited[-1][1]['embedding'] == 'TSNE'
        # Only the skipped sample: nothing is embedded either.
        ed._run_clustering(method='flowsom', all_samples=False, k=30, grid=10,
                           n_meta=3, embedding='t-SNE')      # active = small
        assert embedded == ['big']
        assert 'TSNE1' not in small.data.columns
        assert 'Nothing was imported' in ed.status_var.get()
    finally:
        root.destroy()


def test_a_skipped_samples_old_embedding_does_not_switch_the_axes():
    root, ed = _editor_or_skip()
    try:
        big = _add(ed, _blobs(sizes=(300, 300, 300))[0], name='big')
        small = _add(ed, _blobs(sizes=(30, 20, 10))[0], name='small')
        small.data['TSNE1'] = 0.0                    # from an earlier run
        small.data['TSNE2'] = 0.0
        big.run_tsne = lambda **kw: None             # backend writes nothing
        x_before = ed.x_combo.get()
        ed._run_clustering(method='flowsom', all_samples=True, k=30, grid=10,
                           n_meta=3, embedding='t-SNE')
        assert ed._audited[-1][1]['samples'] == ['big']
        assert ed.x_combo.get() == x_before
    finally:
        root.destroy()


def test_an_earlier_runs_embedding_on_a_clustered_sample_is_not_this_runs():
    """Presence of TSNE1 cannot tell this run's embedding from an earlier
    one: with a backend that wrote nothing, the axes switched to TSNE1/TSNE2
    and the audit recorded embedding='TSNE'."""
    root, ed = _editor_or_skip()
    try:
        s = _add(ed, _blobs(sizes=(300, 300, 300))[0])
        s.data['TSNE1'] = 0.0                        # from an earlier run
        s.data['TSNE2'] = 0.0
        s.run_tsne = lambda **kw: None               # backend writes nothing
        x_before = ed.x_combo.get()

        def run():
            ed._run_clustering(method='flowsom', all_samples=False, k=30,
                               grid=4, n_meta=3, embedding='t-SNE')
            return ed._audited[-1][1]

        audit = run()
        assert audit['samples'] == ['s1'] and audit['embedding'] == 'none'
        assert ed.x_combo.get() == x_before

        # A real write over those columns (the path every embedding takes)
        # does count.
        def store(self, **kw):
            n = len(self.data)
            self._store_embedding(np.ones((n, 2)), np.ones(n, dtype=bool),
                                  'TSNE')
        s.run_tsne = store.__get__(s)
        assert run()['embedding'] == 'TSNE'
        assert ed.x_combo.get() == ed._fmt_channel('TSNE1')
    finally:
        root.destroy()


def _planted_2d():
    rng = np.random.default_rng(0)
    # (CD1, CD2) groups: A=(0,0) 300, B=(0,8) 120, C=(8,0) 200
    spec = [((0, 0), 300), ((0, 8), 120), ((8, 0), 200)]
    X = np.vstack([rng.normal(c, 0.4, (n, 2)) for c, n in spec])
    return pd.DataFrame(X, columns=['CD1', 'CD2'])


def _name_cluster(ed, sample, cid, text):
    """What the Annotate clusters dialog's Apply stores."""
    ed._cluster_labels.setdefault(sample, {})[cid] = text
    for g in ed._sample_gates[sample].values():
        if g.get('kind') == 'cluster' and g.get('cluster_id') == cid:
            g['name'] = text


def _cluster_names(ed, sample):
    return {g['cluster_id']: g['name'] for g in ed._sample_gates[sample].values()
            if g.get('kind') == 'cluster'}


def test_reclustering_resets_names_given_to_the_previous_partition():
    """Cluster ids are size ranks of one run: re-clustering on other markers
    gives 'Cluster 0' different cells, so the old name must not stay on it."""
    root, ed = _editor_or_skip()
    try:
        s = _add(ed, _planted_2d())

        def run(channels):
            with contextlib.redirect_stdout(io.StringIO()):
                ed._run_clustering(method='phenograph', all_samples=False,
                                   k=30, grid=10, n_meta=10, embedding='none',
                                   channels=channels, reproducible=True)
            return s.data['cluster'].to_numpy().copy()

        lab1 = run(None)
        _name_cluster(ed, 's1', 0, 'T cells')
        lab2 = run(['CD2'])
        assert not np.array_equal(lab1, lab2)       # precondition: ids moved
        assert _cluster_names(ed, 's1')[0] == 'Cluster 0'
        assert ed._cluster_display_name('s1', 0) == 'Cluster 0'
        assert "population names were reset" in ed.status_var.get()
        # A re-run that reproduces the same partition keeps the names.
        _name_cluster(ed, 's1', 0, 'B cells')
        lab3 = run(['CD2'])
        assert np.array_equal(lab2, lab3)
        assert _cluster_names(ed, 's1')[0] == 'B cells'
        assert ed._cluster_display_name('s1', 0) == 'B cells'
        assert "reset" not in ed.status_var.get()
    finally:
        root.destroy()


def test_reclustering_resets_leiden_population_names():
    root, ed = _editor_or_skip()
    try:
        s = _add(ed, _planted_2d())
        ed._run_clustering(method='leiden', all_samples=False, k=15, grid=10,
                           n_meta=10, embedding='none', resolution=0.3)
        lab1 = s.data['leiden'].to_numpy().copy()
        for g in ed._sample_gates['s1'].values():
            if g.get('kind') == 'category' and g.get('value') == 0:
                g['name'] = 'T cells'
        ed._run_clustering(method='leiden', all_samples=False, k=15, grid=10,
                           n_meta=10, embedding='none', resolution=2.0,
                           channels=['CD2'])
        assert not np.array_equal(lab1, s.data['leiden'].to_numpy())
        names = {g['value']: g['name'] for g in ed._sample_gates['s1'].values()
                 if g.get('kind') == 'category' and g.get('channel') == 'leiden'}
        assert names[0] == ed._population_label('leiden', 0)
    finally:
        root.destroy()


def test_reclustering_with_fewer_clusters_removes_their_stale_gates():
    """12 clusters, then 8: 'Cluster 8'-'Cluster 11' used to stay in the tree
    selecting nothing. A gate the user drew is never removed, and neither is
    a stale population holding one (that would orphan the user's gate)."""
    root, ed = _editor_or_skip()
    try:
        s = _add(ed, _planted_2d())

        def run(channels):
            with contextlib.redirect_stdout(io.StringIO()):
                ed._run_clustering(method='phenograph', all_samples=False,
                                   k=30, grid=10, n_meta=10, embedding='none',
                                   channels=channels, reproducible=True)
            return set(np.unique(s.data['cluster']).tolist())

        assert run(['CD2']) == set(range(12))
        gates = ed._sample_gates['s1']
        g11 = next(gid for gid, g in gates.items()
                   if g.get('kind') == 'cluster' and g.get('cluster_id') == 11)
        gates['mine'] = {'kind': 'threshold', 'channel': 'CD1', 'value': 4.0,
                         'parent_id': None, 'name': 'Mine', 'enabled': True}
        gates['inside'] = {'kind': 'threshold', 'channel': 'CD2', 'value': 4.0,
                           'parent_id': g11, 'name': 'Inside', 'enabled': True}
        ed._sample_gate_order['s1'] += ['mine', 'inside']
        assert run(None) == set(range(8))
        ids = {g['cluster_id'] for g in gates.values()
               if g.get('kind') == 'cluster'}
        assert ids == set(range(8)) | {11}
        assert gates['mine']['name'] == 'Mine'
        assert gates['inside']['parent_id'] == g11 and g11 in gates
        assert set(ed._sample_gate_order['s1']) == set(gates)
        grp = next(g for g in gates.values() if g.get('group_for') == 'cluster')
        assert grp['name'].endswith('(9)')
        st = ed.status_var.get()
        assert 'Removed 3 cluster population(s)' in st
        assert 'Kept 1' in st
        assert "'Cluster 11' in s1, parent of 'Inside'" in st
    finally:
        root.destroy()


def test_reclustering_keeps_a_stale_population_a_boolean_gate_uses():
    """A boolean gate names its operands by id, not as children. Pruning
    'Cluster 11' silently changed 'Cluster 11 AND CD1+': with the missing
    operand skipped it became all of CD1+ (12 events to 200), and where a
    missing operand fails closed it becomes empty. Kept, the operand selects
    nothing, so the AND is honestly empty and the status line says why."""
    from openflo.pipeline import cumulative_gate_mask
    root, ed = _editor_or_skip()
    try:
        s = _add(ed, _planted_2d())

        def run(channels):
            with contextlib.redirect_stdout(io.StringIO()):
                ed._run_clustering(method='phenograph', all_samples=False,
                                   k=30, grid=10, n_meta=10, embedding='none',
                                   channels=channels, reproducible=True)

        run(['CD2'])
        gates = ed._sample_gates['s1']
        g11 = next(gid for gid, g in gates.items()
                   if g.get('kind') == 'cluster' and g.get('cluster_id') == 11)
        gates['cd1'] = {'kind': 'threshold', 'channel': 'CD1', 'value': 4.0,
                        'parent_id': None, 'name': 'CD1+', 'enabled': True}
        gates['both'] = {'kind': 'boolean', 'op': 'and',
                         'operands': [g11, 'cd1'], 'name': 'C11 and CD1+',
                         'parent_id': None, 'enabled': True}
        ed._sample_gate_order['s1'] += ['cd1', 'both']
        run(None)
        assert 11 not in set(np.unique(s.data['cluster']).tolist())
        assert g11 in gates
        n = int(np.asarray(cumulative_gate_mask(gates, 'both', s.data)).sum())
        assert n == 0                 # cluster 11 is gone; not all of CD1+
        ids = {g['cluster_id'] for g in gates.values()
               if g.get('kind') == 'cluster'}
        assert ids == set(range(8)) | {11}
        st = ed.status_var.get()
        assert 'Removed 3 cluster population(s)' in st
        assert "'Cluster 11' in s1, used by boolean 'C11 and CD1+'" in st
    finally:
        root.destroy()


def test_undo_after_a_rerun_brings_back_gates_but_not_old_names():
    """Undo restores gates, not the label column. Ctrl+Z after a re-run
    (12 clusters, then 8) must bring 'Cluster 8'-'Cluster 11' back, which the
    import's late checkpoint never did, but must not put the old partition's
    names back on the new one: 'Named population' (61 cells of run 1) then
    selected 79 cells of run 2, none of the original 61."""
    root, ed = _editor_or_skip()
    try:
        _add(ed, _planted_2d())

        def run(channels):
            with contextlib.redirect_stdout(io.StringIO()):
                ed._run_clustering(method='phenograph', all_samples=False,
                                   k=30, grid=10, n_meta=10, embedding='none',
                                   channels=channels, reproducible=True)
            ed.update_idletasks()             # end of this Tk event

        run(['CD2'])
        _name_cluster(ed, 's1', 3, 'Named population')
        run(None)
        assert len(_cluster_names(ed, 's1')) == 8          # the run changed
        ed._undo()
        assert _cluster_names(ed, 's1') == {
            i: f'Cluster {i}' for i in range(12)}
        assert 'Named population' not in str(ed._cluster_labels)
        ed._redo()
        assert _cluster_names(ed, 's1') == {
            i: f'Cluster {i}' for i in range(8)}
    finally:
        root.destroy()


def test_no_undo_or_redo_state_puts_old_names_on_a_rerun():
    """Each naming step saves an undo state holding the names given so far.
    After a re-run, a second Ctrl+Z reached the state from before the second
    name and put the first name back on the new partition's cluster 3. Only
    the re-run sample's populations of the rewritten column are reset in the
    saved states: another sample, a user gate and another column keep their
    names."""
    root, ed = _editor_or_skip()
    try:
        _add(ed, _planted_2d(), name='s2')
        _add(ed, _planted_2d())                         # s1, now active

        def run(channels, all_samples):
            with contextlib.redirect_stdout(io.StringIO()):
                ed._run_clustering(method='phenograph', all_samples=all_samples,
                                   k=30, grid=10, n_meta=10, embedding='none',
                                   channels=channels, reproducible=True)
            ed.update_idletasks()             # end of this Tk event

        def name_step(sample, cid, text):     # what the Annotate dialog does
            ed._checkpoint()
            _name_cluster(ed, sample, cid, text)
            ed.update_idletasks()

        run(['CD2'], all_samples=True)
        ed._sample_gates['s1']['mine'] = {
            'kind': 'threshold', 'channel': 'CD1', 'value': 4.0,
            'parent_id': None, 'name': 'Mine', 'enabled': True}
        ed._sample_gates['s1']['meta'] = {
            'kind': 'category', 'channel': 'flowsom_meta', 'value': 0,
            'parent_id': None, 'name': 'Meta name', 'enabled': False}
        ed._sample_gate_order['s1'] += ['mine', 'meta']
        name_step('s2', 0, 'Other sample name')
        name_step('s1', 3, 'First name')
        name_step('s1', 5, 'Second name')
        run(None, all_samples=False)                    # re-run s1 only
        assert len(_cluster_names(ed, 's1')) == 8

        def check(n_clusters):
            assert _cluster_names(ed, 's1') == {
                i: f'Cluster {i}' for i in range(n_clusters)}
            assert 's1' not in ed._cluster_labels
            assert _cluster_names(ed, 's2')[0] == 'Other sample name'
            assert ed._cluster_labels['s2'][0] == 'Other sample name'
            assert ed._sample_gates['s1']['mine']['name'] == 'Mine'
            assert ed._sample_gates['s1']['meta']['name'] == 'Meta name'

        ed._undo()
        check(12)
        ed._undo()
        check(12)
        ed._redo()
        check(12)
        ed._redo()
        check(8)
    finally:
        root.destroy()


def test_reclustering_flowsom_removes_stale_metacluster_gates():
    root, ed = _editor_or_skip()
    try:
        s = _add(ed, _blobs(sizes=(300, 300, 300))[0])

        def run(n_meta):
            ed._run_clustering(method='flowsom', all_samples=False, k=30,
                               grid=4, n_meta=n_meta, embedding='none')
            return set(np.unique(s.data['flowsom_meta']).tolist())

        first = run(6)
        second = run(3)
        assert first - second                      # precondition: ids vanished
        vals = {g['value'] for g in ed._sample_gates['s1'].values()
                if g.get('kind') == 'category'
                and g.get('channel') == 'flowsom_meta'}
        assert vals == second
    finally:
        root.destroy()


def test_a_failing_run_reports_and_clears_busy():
    from openflo.pipeline import ClusteringError
    root, ed = _editor_or_skip()
    try:
        s = _add(ed, _blobs()[0])

        def boom(**kw):
            raise ClusteringError('phenograph is required for CPU clustering')
        s.cluster = boom
        ed._run_clustering(method='phenograph', all_samples=False, k=15,
                           grid=10, n_meta=10, embedding='none')
        assert ed._clustering_busy is False
        assert ed.status_var.get().startswith('Clustering failed:')
        assert 'phenograph is required' in ed.status_var.get()
        assert ed._audited == []                  # nothing claimed as done
    finally:
        root.destroy()
