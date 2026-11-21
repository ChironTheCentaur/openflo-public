"""3-component UMAP: FlowSample.run_umap(n_components=3), the Cluster
dialog's Components box, openflo-run --umap-components, exports and
sessions. Generated data only.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from openflo import view3d as v3

CH = ['FL1-A', 'FL2-A', 'FL3-A', 'FL4-A']


def _blobs(n=(150, 120, 90), seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(6.0 * i, 0.5, (k, len(CH)))
                   for i, k in enumerate(n)])
    return pd.DataFrame(X, columns=CH)


def _sample(name='s1', seed=0):
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(_blobs(seed=seed), name=name)
    s.fluor_channels = list(CH)
    return s


# ── the core ─────────────────────────────────────────────────────────────

def test_three_components_are_stored_and_reproducible_from_the_seed():
    a, b, c = _sample(), _sample(), _sample()
    for s, seed in ((a, 7), (b, 7), (c, 8)):
        s.run_umap(channels=CH, n_neighbors=10, random_state=seed,
                   n_components=3)
    cols = ['UMAP1', 'UMAP2', 'UMAP3']
    for s in (a, b, c):
        assert all(col in s.data.columns for col in cols)
        assert np.isfinite(s.data[cols].to_numpy()).all()
        assert s.umap_coords.shape == (len(s.data), 3)
    np.testing.assert_array_equal(a.data[cols].to_numpy(), b.data[cols].to_numpy())
    assert not np.allclose(a.data[cols].to_numpy(), c.data[cols].to_numpy())
    # The third axis carries structure of its own, not a copy of another.
    assert np.std(a.data['UMAP3']) > 0
    assert not np.allclose(a.data['UMAP3'], a.data['UMAP1'])


def test_a_2d_run_removes_an_earlier_third_component():
    s = _sample()
    s.run_umap(channels=CH, n_neighbors=10, n_components=3)
    s.run_umap(channels=CH, n_neighbors=10)
    assert 'UMAP3' not in s.data.columns
    assert {'UMAP1', 'UMAP2'} <= set(s.data.columns)
    assert s.umap_coords.shape[1] == 2


@pytest.mark.parametrize('bad', [1, 4, 2.5, True, '3'])
def test_only_two_or_three_components(bad):
    s = _sample()
    with pytest.raises((ValueError, TypeError)):
        s.run_umap(channels=CH, n_components=bad)
    assert 'UMAP1' not in s.data.columns


def test_the_subsample_leaves_the_rest_nan_on_all_three():
    s = _sample()
    s.run_umap(channels=CH, n_neighbors=10, sample_n=200, n_components=3)
    emb = s.data[['UMAP1', 'UMAP2', 'UMAP3']].to_numpy()
    done = np.isfinite(emb).all(axis=1)
    assert done.sum() == 200
    assert np.isnan(emb[~done]).all()


def test_exports_carry_the_third_component(tmp_path):
    from openflo.fcs_export import population_export_frame, write_fcs
    from openflo.pipeline import FlowSample
    s = _sample()
    s.run_umap(channels=CH, n_neighbors=10, n_components=3)
    want = s.data['UMAP3'].to_numpy()
    # CSV
    csv = tmp_path / 's1.csv'
    s.export_csv(str(csv))
    got = pd.read_csv(csv)
    np.testing.assert_allclose(got['UMAP3'].to_numpy(), want)
    # FCS of a population, read back
    mask = np.arange(len(s.data)) % 2 == 0
    frame, ranges, dropped = population_export_frame(s, mask)
    assert 'UMAP3' in frame.columns and 'UMAP3' not in dropped
    fcs = tmp_path / 'pop.fcs'
    write_fcs(frame, str(fcs), ranges=ranges)
    back = FlowSample(str(fcs))
    np.testing.assert_allclose(back.data['UMAP3'].to_numpy(), want[mask],
                               rtol=1e-6, atol=1e-6)


def test_the_third_component_is_never_a_marker():
    from openflo.channel_roles import is_excluded
    assert is_excluded('UMAP3')


def test_embedding_prefix_reads_all_three_components():
    from openflo.channel_roles import embedding_prefix
    assert embedding_prefix('UMAP3') == 'UMAP'
    assert embedding_prefix('tsne2') == 'TSNE'
    assert embedding_prefix('UMAP4') is None
    assert embedding_prefix('FL3-A') is None


# ── turntable of an embedding ───────────────────────────────────────────

def test_embedding_turntable_colors_each_cluster_its_own(tmp_path):
    from openflo.cluster_labels import GATE_PALETTE
    rng = np.random.default_rng(2)
    coords = np.vstack([rng.normal(c, 0.2, (300, 3)) for c in (0, 5)])
    labels = np.repeat([0, 1], 300)
    path = tmp_path / 't.gif'
    n = v3.save_embedding_turntable(coords, labels, str(path), n_frames=36,
                                    size=201)
    assert n == 600
    with Image.open(path) as im:
        assert im.n_frames == 36 and im.size == (200, 200)
        assert im.info.get('loop') == 0
        rgb = np.asarray(im.convert('RGB')).reshape(-1, 3)
    seen = {tuple(c) for c in rgb}
    for cid in (0, 1):
        want = v3.hex_rgb(GATE_PALETTE[cid])
        assert any(np.abs(np.array(c) - want).max() <= 8 for c in seen), cid
    assert dict(v3.cluster_palette(np.array([-1, 0])))[-1] != GATE_PALETTE[0]


def test_embedding_turntable_caps_its_events(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, 'MAX_EVENTS', 1000)
    coords = np.random.default_rng(0).normal(0, 1, (5000, 3))
    coords[:10] = np.nan
    n = v3.save_embedding_turntable(coords, None, str(tmp_path / 'c.gif'),
                                    n_frames=36, size=80)
    assert n == 1000


# ── openflo-run --umap-components ───────────────────────────────────────

def test_the_cli_flag_reaches_the_group_umap(monkeypatch):
    from openflo import cli_args, cli_main
    ap = cli_args.build_parser()
    args = ap.parse_args(['--trials', 'x', '--umap-components', '3'])
    assert args.umap_components == 3
    assert ap.parse_args(['--trials', 'x']).umap_components == 2
    with pytest.raises(SystemExit):
        ap.parse_args(['--trials', 'x', '--umap-components', '4'])
    seen = {}
    monkeypatch.setattr(cli_main, 'run', lambda **kw: seen.update(kw) or 0)
    cli_main._run_analysis(args, [], [], {}, {}, {}, {}, None)
    assert seen['umap_components'] == 3


def test_the_run_settings_carry_the_components_to_the_group_plots(monkeypatch):
    from openflo import cli_compare
    from openflo.shared_clustering import cluster_settings
    st = cluster_settings(30, 1, None, 1.0, 42, {'umap_components': 3})
    assert st['umap_components'] == 3
    assert cluster_settings(30, 1, None, 1.0, 42)['umap_components'] == 2
    got = []
    monkeypatch.setattr(cli_compare, 'group_plots_job',
                        lambda *a, **k: got.append(a))
    cli_compare.draw_group_plots([('G', [object()], 'G', 'G')], 'out', 42,
                                 None, 1, lambda m: None, umap_components=3)
    assert got[0][-1] == 3


def test_group_umap_with_three_components_writes_gif_and_csv(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from openflo.cli_plots import run_group_umap
    samples = []
    for i in range(2):
        s = _sample(name=f's{i}', seed=i)
        s.data['cluster'] = np.repeat([0, 1, 2], (150, 120, 90))
        samples.append(s)
    seeds = []
    real = v3.save_embedding_turntable
    monkeypatch.setattr(v3, 'save_embedding_turntable',
                        lambda *a, **k: seeds.append(k.get('seed')) or real(*a, **k))
    ref = run_group_umap(samples, 'G', str(tmp_path), random_state=3,
                         dpi=60, n_components=3)
    assert seeds == [3]
    assert 'UMAP3' in ref.data.columns
    assert (tmp_path / 'G_umap.png').is_file()
    table = pd.read_csv(tmp_path / 'G_umap3d.csv')
    assert list(table.columns) == ['UMAP1', 'UMAP2', 'UMAP3', 'cluster',
                                   'sample_origin']
    assert len(table) == 720
    assert sorted(set(table['sample_origin'])) == ['s0', 's1']
    with Image.open(tmp_path / 'G_umap3d.gif') as im:
        assert im.n_frames == v3.DEFAULT_FRAMES
    # Two components: no 3D files.
    out2 = tmp_path / 'two'
    run_group_umap(samples, 'G', str(out2), random_state=3, dpi=60)
    assert not (out2 / 'G_umap3d.gif').exists()
    assert not (out2 / 'G_umap3d.csv').exists()
    # Two components into the folder of the 3-component run: its 3D files
    # go, so they are never taken for this run's.
    run_group_umap(samples, 'G', str(tmp_path), random_state=3, dpi=60)
    assert not (tmp_path / 'G_umap3d.gif').exists()
    assert not (tmp_path / 'G_umap3d.csv').exists()
    assert (tmp_path / 'G_umap.png').is_file()


# ── the Cluster dialog ──────────────────────────────────────────────────

def _cluster_editor():
    from tests.test_cluster_dialog import _add
    from tests.test_cluster_dialog import _editor_or_skip as cl_editor
    root, ed = cl_editor()
    s = _add(ed, _blobs())
    s.fluor_channels = list(CH)
    return root, ed, s


def _walk(w):
    stack = [w]
    while stack:
        x = stack.pop()
        yield x
        stack.extend(x.winfo_children())


def test_components_box_is_for_umap_and_reaches_the_run():
    root, ed, _s = _cluster_editor()
    try:
        got = {}
        ed._run_clustering = lambda **kw: got.update(kw)
        ed._open_cluster_dialog()
        dlg = next(w for w in ed.winfo_children()
                   if isinstance(w, tk.Toplevel) and w.title() == 'Cluster')
        dims = ed._cluster_dims_combo
        emb = next(w for w in _walk(dlg) if isinstance(w, ttk.Combobox)
                   and 'none' in tuple(w.cget('values')))
        assert emb.get() == 'UMAP' and dims.get() == '2'
        assert 'disabled' not in dims.state()
        emb.set('none')
        root.update()
        assert 'disabled' in dims.state()
        emb.set('UMAP')
        root.update()
        assert 'disabled' not in dims.state()
        dims.set('3')
        next(w for w in _walk(dlg) if isinstance(w, ttk.Button)
             and w.cget('text') == 'Run').invoke()
        assert got['embedding'] == 'UMAP' and got['embed_dims'] == 3
    finally:
        root.destroy()


def test_a_non_umap_embedding_stays_two_components():
    root, ed, _s = _cluster_editor()
    try:
        got = {}
        ed._run_clustering = lambda **kw: got.update(kw)
        ed._open_cluster_dialog()
        dlg = next(w for w in ed.winfo_children()
                   if isinstance(w, tk.Toplevel) and w.title() == 'Cluster')
        ed._cluster_dims_combo.set('3')
        emb = next(w for w in _walk(dlg) if isinstance(w, ttk.Combobox)
                   and 'none' in tuple(w.cget('values')))
        emb.set('t-SNE')
        next(w for w in _walk(dlg) if isinstance(w, ttk.Button)
             and w.cget('text') == 'Run').invoke()
        assert got['embedding'] == 't-SNE' and got['embed_dims'] == 2
    finally:
        root.destroy()


def test_a_3d_run_writes_umap3_audits_it_and_opens_the_3d_plot():
    root, ed, s = _cluster_editor()
    try:
        ed._run_clustering(method='flowsom', all_samples=False, k=15, grid=4,
                           n_meta=3, embedding='UMAP', embed_dims=3)
        root.update()
        assert {'UMAP1', 'UMAP2', 'UMAP3'} <= set(s.data.columns)
        audit = ed._audited[-1][1]
        assert audit['embedding'] == 'UMAP'
        assert audit['embedding_components'] == 3
        stamps = getattr(s, 'result_stamps', None) or {}
        assert 'UMAP3' in stamps and 'UMAP1' in stamps
        assert ed._channel_scale.get('UMAP3') == 'linear'
        win = ed._tool_windows.get('view3d')
        assert win is not None and win.winfo_exists()
        assert win._axes() == ['UMAP1', 'UMAP2', 'UMAP3']
        assert win.color_mode_var.get() == 'Cluster'
        assert win.color_target_var.get() == 'flowsom_meta'
        assert win._cloud is not None and win._cloud.shown == len(s.data)
        # A 2-component run after it: UMAP3 is gone, no new window.
        win.destroy()
        ed._run_clustering(method='flowsom', all_samples=False, k=15, grid=4,
                           n_meta=3, embedding='UMAP')
        root.update()
        assert 'UMAP3' not in s.data.columns
        assert ed._audited[-1][1]['embedding_components'] == 2
        w2 = ed._tool_windows.get('view3d')
        assert w2 is None or not w2.winfo_exists()
    finally:
        root.destroy()


def test_computed_results_name_the_3d_embedding_once():
    root, ed, s = _cluster_editor()
    try:
        s.data['UMAP1'] = 0.0
        s.data['UMAP2'] = 0.0
        s.data['UMAP3'] = 0.0
        s.channel_names = list(CH)
        items = ed._computed_results(s)
        assert items == ['UMAP embedding (UMAP1/UMAP2/UMAP3)']
        s.data.drop(columns=['UMAP3'], inplace=True)
        assert ed._computed_results(s) == ['UMAP embedding (UMAP1/UMAP2)']
    finally:
        root.destroy()


# ── sessions ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('dims', [2, 3])
def test_a_session_keeps_its_embedding(tmp_path, monkeypatch, dims):
    """A 2-D embedding (what every earlier version saved) reopens as it
    was, with no third axis, and the 3D plot then starts on channels; a 3-D
    one reopens with UMAP3 and the 3D plot starts on it."""
    from tests.test_derived_kinetics import _fcs, _loaded, _restore_session
    from tests.test_transform_records import _new_editor, isolate_home
    isolate_home(monkeypatch, tmp_path)
    try:
        root = tk.Tk()
    except tk.TclError as e:
        from tests.conftest import gui_unavailable
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    try:
        ed = _new_editor(root)
        s = _loaded(_fcs(tmp_path / 'emb.fcs', 4))
        ed._on_loaded('emb', s)
        s.run_umap(n_neighbors=10, n_components=dims)
        want = s.data[[f'UMAP{i}' for i in range(1, dims + 1)]].to_numpy(copy=True)
        path = tmp_path / 'emb.flowsession'
        ed._write_session(str(path))
        ed2, _jobs = _restore_session(root, path)
        s2 = ed2._samples['emb']
        got = s2.data[[f'UMAP{i}' for i in range(1, dims + 1)]].to_numpy()
        np.testing.assert_allclose(got, want, rtol=1e-6)
        assert ('UMAP3' in s2.data.columns) == (dims == 3)
        ed2._set_active_sample('emb')
        win = ed2._open_view3d()
        root.update()
        axes = win._axes()
        if dims == 3:
            assert axes == ['UMAP1', 'UMAP2', 'UMAP3']
        else:
            assert 'UMAP3' not in axes and all(axes)
        assert win._cloud is not None and win._cloud.shown > 0
    finally:
        root.destroy()


def test_has_computed_columns_sees_a_lone_third_component():
    import types

    from openflo.gui import ViewGateEditorWindow as V
    has = V._has_computed_columns
    assert has(types.SimpleNamespace(
        data=pd.DataFrame({'FSC-A': [1.0], 'UMAP3': [0.1]}),
        comp_channels=[], path='/data/x.fcs'))
    assert not has(types.SimpleNamespace(
        data=pd.DataFrame({'FSC-A': [1.0], 'UMAP4': [0.1]}),
        comp_channels=[], path='/data/x.fcs'))


def test_the_cli_turntable_needs_no_tk(tmp_path):
    """The CLI writes a 3D embedding's GIF headless: tkinter missing."""
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent(f"""
        import sys
        sys.modules['tkinter'] = None
        import numpy as np
        from openflo import view3d as v3
        xyz = np.random.default_rng(2).normal(size=(300, 3))
        assert v3.save_embedding_turntable(xyz, None, {str(tmp_path / 'a.gif')!r}, n_frames=36, size=64) == 300
        assert 'openflo.theme' not in sys.modules
        print('ok')
    """)
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                       env={**__import__('os').environ, 'MPLBACKEND': 'Agg'}, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr


def test_the_3d_plot_opens_on_the_active_sample_else_the_first():
    from tests.test_cluster_dialog import _add
    root, ed, s1 = _cluster_editor()
    try:
        b = _blobs(seed=3)
        s2 = _add(ed, b, name='s2')
        s3 = _add(ed, _blobs(seed=4), name='s3')
        for s in (s1, s2):
            for i in (1, 2, 3):
                s.data[f'UMAP{i}'] = np.arange(len(s.data), dtype=float) * i
        opened = []
        ed._open_view3d_on = lambda name, axes, color_column=None: opened.append(
            (name, tuple(axes), color_column))
        ed._set_active_sample('s2')
        ed._show_embedding_axes('UMAP', 'cluster', ['s1', 's2', 's3'])
        assert opened[-1] == ('s2', ('UMAP1', 'UMAP2', 'UMAP3'), 'cluster')
        # The active sample has no third component: the first one that does.
        assert 'UMAP3' not in s3.data.columns
        ed._set_active_sample('s3')
        ed._show_embedding_axes('UMAP', 'cluster', ['s3', 's2', 's1'])
        assert opened[-1][0] == 's2'
    finally:
        root.destroy()


def test_a_non_umap_run_is_never_asked_for_three_components(monkeypatch):
    """Components is UMAP's alone: a t-SNE run with the box at 3 is not
    handed n_components and records none."""
    from openflo.pipeline import FlowSample
    root, ed, _s = _cluster_editor()
    try:
        seen = []

        def spy(self, *a, **kw):
            seen.append(kw)
            return self
        monkeypatch.setattr(FlowSample, 'run_tsne', spy)
        ed._run_clustering(method='flowsom', all_samples=False, k=15, grid=4,
                           n_meta=3, embedding='t-SNE', embed_dims=3)
        root.update()
        assert len(seen) == 1 and 'n_components' not in seen[0]
        assert 'embedding_components' not in ed._audited[-1][1]
    finally:
        root.destroy()


def test_a_third_component_defaults_to_linear_even_if_stored_transformed():
    """An embedding axis is drawn linear by default, whatever transform a
    file recorded for a column of that name."""
    root, ed, _s = _cluster_editor()
    try:
        for c in ('UMAP3', 'TSNE2', 'FL1-A'):
            ed._channel_transform[c] = 'logicle'
        assert ed._default_scale_for('UMAP3') == 'linear'
        assert ed._default_scale_for('TSNE2') == 'linear'
        assert ed._default_scale_for('FL1-A') == ed._default_channel_scale != 'linear'
    finally:
        root.destroy()


def test_the_3d_plot_shows_the_sample_it_is_asked_for():
    from tests.test_cluster_dialog import _add
    root, ed, _s1 = _cluster_editor()
    try:
        s2 = _add(ed, _blobs(seed=5), name='s2')
        for i in (1, 2, 3):
            s2.data[f'UMAP{i}'] = np.arange(len(s2.data), dtype=float) * i
        ed._set_active_sample('s1')
        win = ed._open_view3d_on('s2', ['UMAP1', 'UMAP2', 'UMAP3'])
        root.update()
        assert win.sample_var.get() == 's2'
        assert win._axes() == ['UMAP1', 'UMAP2', 'UMAP3']
    finally:
        root.destroy()


def test_a_run_hands_the_components_to_its_mode(tmp_path, monkeypatch):
    from openflo import cli_run
    seen = []
    for mode in ('run_concatenated', 'run_independent'):
        monkeypatch.setattr(cli_run, mode,
                            lambda *a, **k: seen.append(k['cluster_cfg']) or True)
    monkeypatch.setattr(cli_run, 'gpu_present', lambda: False)
    trial = tmp_path / 'trial_1'
    trial.mkdir()
    (trial / 'sample_1.fcs').write_bytes(b'')
    for mode in ('concatenate', 'independent'):
        cli_run.run([str(trial)], str(tmp_path / 'out'),
                    [{'name': 'G1', 'fmo_set': '', 'samples': ['sample_1']}],
                    fmo_sets={}, batch_mode=mode, umap_components=3)
    assert [c['umap_components'] for c in seen] == [3, 3]


def test_the_group_outputs_draw_the_umap_in_three_components(tmp_path, monkeypatch):
    from openflo import cli_compare
    got = []
    monkeypatch.setattr(cli_compare, 'too_few_to_draw', lambda *a: False)
    monkeypatch.setattr(cli_compare, 'draw_group_plots',
                        lambda *a, **k: got.append(k['umap_components']))
    cli_compare.write_group_outputs(
        [{'name': 'G1'}], {'G1': [object()]}, {}, str(tmp_path),
        lambda m: None, {'umap_components': 3})
    assert got == [3]


def _pooled(n=400, seed=0):
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(0, 1, (n, 3)), columns=['UMAP1', 'UMAP2', 'UMAP3'])
    df['cluster'] = np.arange(n) % 4
    df['sample_origin'] = 's1'
    df.loc[::5, ['UMAP1', 'UMAP2', 'UMAP3']] = np.nan
    return SimpleNamespace(data=df)


def test_the_group_3d_files_hold_the_embedded_events_by_cluster(tmp_path, monkeypatch):
    """A UMAP run on a subsample leaves the other events with no place: the
    CSV leaves them out; the GIF is colored by cluster with the run's seed."""
    from openflo.cli_plots import save_group_umap3d
    ref = _pooled()
    calls = []
    real = v3.save_embedding_turntable
    monkeypatch.setattr(v3, 'save_embedding_turntable',
                        lambda xyz, labels, path, **kw: calls.append((labels, kw))
                        or real(xyz, labels, path, n_frames=36, size=60))
    save_group_umap3d(ref, 'G', str(tmp_path), random_state=11)
    table = pd.read_csv(tmp_path / 'G_umap3d.csv')
    assert len(table) == 320 and table[['UMAP1', 'UMAP2', 'UMAP3']].notna().all().all()
    labels, kw = calls[0]
    np.testing.assert_array_equal(labels, ref.data['cluster'].to_numpy())
    assert kw['seed'] == 11


def test_cluster_colors_follow_the_ids_and_noise_is_grey():
    from openflo.cluster_labels import GATE_PALETTE, NOISE_CLUSTER_COLOR
    pal = dict(v3.cluster_palette(np.array([7, -1, 3, 7])))
    assert pal == {-1: NOISE_CLUSTER_COLOR, 3: GATE_PALETTE[3 % len(GATE_PALETTE)],
                   7: GATE_PALETTE[7 % len(GATE_PALETTE)]}


def test_an_embedding_turntable_leaves_out_events_with_no_place(tmp_path):
    coords = np.random.default_rng(1).normal(0, 1, (500, 3))
    coords[::10, 2] = np.nan
    n = v3.save_embedding_turntable(coords, None, str(tmp_path / 'n.gif'),
                                    n_frames=36, size=60)
    assert n == 450


def test_parallel_group_plots_carry_the_components(monkeypatch):
    """--workers > 1 draws the groups in a process pool: each job is
    still handed the component count."""
    import concurrent.futures

    from openflo import cli_compare
    submitted = []

    class Pool:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *a):
            submitted.append((fn, a))
            return _Done()

    class _Done:
        def result(self):
            return None
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', Pool)
    jobs = [('A', [object()], 'A', 'A'), ('B', [object()], 'B', 'B')]
    cli_compare.draw_group_plots(jobs, 'out', 42, None, 2, lambda m: None,
                                 umap_components=3)
    assert [fn for fn, _a in submitted] == [cli_compare.group_plots_job] * 2
    assert [a[-1] for _fn, a in submitted] == [3, 3]


def _scene_of(monkeypatch):
    got = []
    real = v3.turntable_frames
    monkeypatch.setattr(v3, 'turntable_frames',
                        lambda scene, *a, **k: got.append(scene) or real(scene, *a, **k))
    return got


def test_the_turntable_subset_follows_its_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, 'MAX_EVENTS', 300)
    scenes = _scene_of(monkeypatch)
    coords = np.random.default_rng(5).normal(0, 1, (2000, 3))
    for seed in (1, 1, 2):
        v3.save_embedding_turntable(coords, None, str(tmp_path / 's.gif'),
                                    n_frames=36, size=40, seed=seed)
    a, b, c = (sc.points for sc in scenes)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_each_drawn_event_has_its_own_cluster_color(tmp_path, monkeypatch):
    """Events left out (no value) or not picked must not shift the colors:
    each drawn dot takes its own event's cluster."""
    from openflo.cluster_labels import GATE_PALETTE
    monkeypatch.setattr(v3, 'MAX_EVENTS', 500)
    scenes = _scene_of(monkeypatch)
    rng = np.random.default_rng(8)
    labels = rng.integers(0, 2, 1500)
    coords = rng.normal(0, 0.3, (1500, 3))
    coords[:, 0] += np.where(labels == 1, 5.0, -5.0)   # cluster 1 on the right
    coords[::4, 1] = np.nan
    v3.save_embedding_turntable(coords, labels, str(tmp_path / 'c.gif'),
                                n_frames=36, size=40)
    sc = scenes[0]
    assert len(sc.points) == 500
    right = sc.points[:, 0] > 0
    want = np.where(right[:, None], v3.hex_rgb(GATE_PALETTE[1]),
                    v3.hex_rgb(GATE_PALETTE[0]))
    np.testing.assert_allclose(np.asarray(sc.colors)[:, :3], want, atol=1e-6)


def test_the_3d_plot_keeps_density_for_a_column_the_sample_lacks():
    root, ed, s = _cluster_editor()
    try:
        for i in (1, 2, 3):
            s.data[f'UMAP{i}'] = np.arange(len(s.data), dtype=float) * i
        win = ed._open_view3d_on('s1', ['UMAP1', 'UMAP2', 'UMAP3'],
                                 color_column='no_such_column')
        root.update()
        assert win.color_mode_var.get() != 'Cluster'
    finally:
        root.destroy()


def test_an_open_3d_plot_follows_a_later_2_component_run():
    """Left open after a 3-component run, the 3D plot must not keep the
    UMAP3 that a 2-component run removes, nor fail on its next reload."""
    root, ed, s = _cluster_editor()
    try:
        ed._run_clustering(method='flowsom', all_samples=False, k=15, grid=4,
                           n_meta=3, embedding='UMAP', embed_dims=3)
        root.update()
        win = ed._tool_windows.get('view3d')
        assert win._axes() == ['UMAP1', 'UMAP2', 'UMAP3']
        stale = win._cloud
        errors = []
        root.report_callback_exception = lambda *a: errors.append(a[1])
        ed._run_clustering(method='flowsom', all_samples=False, k=15, grid=4,
                           n_meta=3, embedding='UMAP')
        root.update()
        assert 'UMAP3' not in s.data.columns
        assert win.winfo_exists() and 'UMAP3' not in win._axes()
        assert all(win._axes()) and win._cloud is not None
        assert win._cloud is not stale            # the old picture is gone
        win.axis_combos[0].event_generate('<<ComboboxSelected>>')
        root.update()
        assert errors == []
    finally:
        root.destroy()


def test_a_3d_plot_whose_column_is_gone_picks_its_axes_again():
    root, ed, s = _cluster_editor()
    try:
        for i in (1, 2, 3):
            s.data[f'UMAP{i}'] = np.arange(len(s.data), dtype=float) * i
        win = ed._open_view3d_on('s1', ['UMAP1', 'UMAP2', 'UMAP3'])
        root.update()
        s.data.drop(columns=['UMAP3'], inplace=True)
        win.refresh()
        assert 'UMAP3' not in win._axes() and all(win._axes())
        assert win._cloud is not None and win._cloud.shown == len(s.data)
    finally:
        root.destroy()


def test_a_3d_plot_whose_sample_is_gone_picks_one_again_after_a_run():
    root, ed, s = _cluster_editor()
    try:
        win = ed._open_view3d_on('s1', ['FL3-A', 'FL1-A', 'FL2-A'])
        root.update()
        win.sample_var.set('a sample since removed')
        win.columns_changed()
        assert win.sample_var.get() == 's1'
        assert all(win._axes()) and win._cloud is not None
        assert win._cloud.shown == len(s.data)
    finally:
        root.destroy()


def test_an_open_3d_plot_keeps_what_a_run_did_not_remove():
    from openflo.ui_view3d import ALL_EVENTS
    root, ed, _s = _cluster_editor()
    try:
        win = ed._open_view3d_on('s1', ['FL3-A', 'FL1-A', 'FL2-A'])
        win.pop_var.set(ALL_EVENTS)
        root.update()
        ed._run_clustering(method='flowsom', all_samples=False, k=15, grid=4,
                           n_meta=3, embedding='UMAP')
        root.update()
        assert win._axes() == ['FL3-A', 'FL1-A', 'FL2-A']
        assert win.pop_var.get() == ALL_EVENTS
    finally:
        root.destroy()


def test_a_bad_component_count_stops_the_run_before_it_starts(tmp_path, monkeypatch):
    from openflo import cli_run
    monkeypatch.setattr(cli_run, 'run_independent',
                        lambda *a, **k: pytest.fail('the run started'))
    with pytest.raises(ValueError, match='2 or 3 components'):
        cli_run.run([str(tmp_path)], str(tmp_path / 'out'),
                    [{'name': 'G1', 'fmo_set': '', 'samples': ['x']}],
                    fmo_sets={}, umap_components=4)


@pytest.mark.parametrize('held,freed', [('csv', 'gif'), ('gif', 'csv')])
def test_an_old_3d_file_that_cannot_go_is_named_and_the_run_goes_on(
        tmp_path, monkeypatch, capsys, held, freed):
    """On Windows a file open elsewhere (a CSV in a spreadsheet, a GIF in a
    viewer) cannot be removed: the 2-D run says which old file stayed and
    why, still removes the other, and finishes."""
    import matplotlib
    matplotlib.use('Agg')
    from openflo import cli_plots
    samples = []
    for i in range(2):
        s = _sample(name=f's{i}', seed=i)
        s.data['cluster'] = np.repeat([0, 1, 2], (150, 120, 90))
        samples.append(s)
    for ext in ('gif', 'csv'):
        (tmp_path / f'G_umap3d.{ext}').write_text('old')
    real = cli_plots.os.remove

    def locked(path):
        if str(path).endswith(f'.{held}'):
            raise PermissionError(13, 'in use by another program')
        real(path)
    monkeypatch.setattr(cli_plots.os, 'remove', locked)
    ref = cli_plots.run_group_umap(samples, 'G', str(tmp_path),
                                   random_state=3, dpi=60)
    assert ref is not None and (tmp_path / 'G_umap.png').is_file()
    assert not (tmp_path / f'G_umap3d.{freed}').exists()
    assert (tmp_path / f'G_umap3d.{held}').exists()
    out = capsys.readouterr().out
    assert f'G_umap3d.{held}' in out and 'in use by another program' in out
    assert f'G_umap3d.{freed}' not in out
