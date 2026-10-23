"""The beta look: what the Samples & Gates tree, the cluster colouring, the
folder groups and the Add FCS dialog show.

Every test builds its own data set (a generated FCS file or frame with
synthetic sample names and generic channels) and checks both sides: the
thing that should show, and the thing that should not.

- Gate rows showed their bounds ('R  FSC-A x SSC-A  FSC-A in [11.4K, ...')
  cut off at the panel width, never the count. They now show the gate's
  name and its count and frequency, filled in as counts are computed.
- Clusters were coloured on a continuous scale, so neighbouring ids looked
  alike; each population now has its own colour, the one its tree row has.
- A cluster population of a sample that came back without its clustering
  (a session whose processed data was not found) read 0 events; it reads NA.
- Groups were named after the grandparent folder ('.openflo' for the
  example data); now the file's own folder.
- The Add FCS dialog opened in the installed package's folder.
- Samples were listed in the order the loader finished them.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
    settle,
)

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']


def _write_fcs(path, n=3000, seed=5):
    """Two scatter clouds and two fluorescence populations, linear values."""
    from openflo.fcs_export import write_fcs
    rng = np.random.default_rng(seed)
    big = rng.random(n) < 0.7
    fsc = np.where(big, rng.normal(120_000, 15_000, n),
                   rng.normal(30_000, 6_000, n))
    ssc = np.where(big, rng.normal(60_000, 10_000, n),
                   rng.normal(15_000, 4_000, n))
    pos = rng.random(n) < 0.4
    fl1 = np.where(pos, rng.normal(20_000, 3_000, n), rng.normal(300, 80, n))
    fl2 = rng.normal(500, 100, n)
    write_fcs(pd.DataFrame(np.column_stack([fsc, ssc, fl1, fl2]),
                           columns=pd.Index(CH)), str(path))
    return str(path)


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    notes = capture_warnings(monkeypatch)       # never a real (blocking) modal
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    ed = _new_editor(root)
    yield root, ed, tmp_path, notes
    root.destroy()


def _load(root, ed, items):
    for name, path in items:
        ed._load_worker(name, str(path))
    for _ in range(200):
        root.update()
        if all(n in ed._samples for n, _ in items):
            break
    assert all(n in ed._samples for n, _ in items)


def _count(ed, name, gid):
    from openflo.pipeline import cumulative_gate_mask
    return int(np.asarray(cumulative_gate_mask(
        ed._sample_gates[name], gid, ed._samples[name].data),
        dtype=bool).sum())


def _row(ed, name, gid):
    return ed.gate_tv.item(ed._gate_iid(name, gid), 'text')


# ── gate rows: name and count, not bounds ───────────────────────────────────

def _gated(root, ed, tmp):
    _load(root, ed, [('sample_a', _write_fcs(tmp / 'a.fcs'))])
    ed._set_active_sample('sample_a')
    data = ed._samples['sample_a'].data
    fl1 = data['FL1-A']
    root_id = ed._add_gate({
        'kind': 'rect', 'name': 'Cells', 'x_channel': 'FSC-A',
        'y_channel': 'SSC-A', 'x0': 60_000.0, 'x1': 250_000.0,
        'y0': 20_000.0, 'y1': 150_000.0, 'parent_id': None})
    child = ed._add_gate({
        'kind': 'threshold', 'channel': 'FL1-A',
        'value': float(np.quantile(fl1, 0.5)), 'parent_id': root_id})
    unnamed = ed._add_gate({
        'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
        'x0': 0.0, 'x1': 60_000.0, 'y0': 0.0, 'y1': 40_000.0,
        'parent_id': None})
    return root_id, child, unnamed


def test_gate_rows_show_name_count_and_percent(env):
    root, ed, tmp, _notes = env
    root_id, child, unnamed = _gated(root, ed, tmp)
    ed._pc_counts('sample_a', [root_id, child, unnamed])     # as a replot does
    ed._refresh_gate_list()
    n_all = len(ed._samples['sample_a'].data)
    n_root, n_child = _count(ed, 'sample_a', root_id), _count(ed, 'sample_a', child)
    n_un = _count(ed, 'sample_a', unnamed)
    assert 0 < n_child < n_root < n_all and n_un > 0

    assert _row(ed, 'sample_a', root_id) == (
        f"Cells  {n_root:,} · {100 * n_root / n_all:.1f}%")
    # A child's frequency is of its parent; an unnamed 1-D gate is its cut.
    assert _row(ed, 'sample_a', child).endswith(
        f"  {n_child:,} · {100 * n_child / n_root:.1f}%")
    assert _row(ed, 'sample_a', child).startswith('FL1-A >= ')
    # An unnamed 2-D gate is named by its channels.
    assert _row(ed, 'sample_a', unnamed) == (
        f"FSC-A x SSC-A  {n_un:,} · {100 * n_un / n_all:.1f}%")
    # ...and no row carries the rectangle's bounds or kind letter any more.
    for gid in (root_id, unnamed):
        text = _row(ed, 'sample_a', gid)
        assert '[' not in text and not text.startswith('R  ')


def test_gate_rows_fill_in_counts_when_they_arrive(env):
    """The tree never computes a count (it rebuilds on every zoom); a row
    waits for one and is updated in place once it is cached."""
    root, ed, tmp, _notes = env
    root_id, _child, _unnamed = _gated(root, ed, tmp)
    ed._pc_store().purge('sample_a')
    ed._refresh_gate_list()
    assert _row(ed, 'sample_a', root_id) == 'Cells'           # not yet counted
    iid = ed._gate_iid('sample_a', root_id)
    ed._pc_counts('sample_a', [root_id])                      # the count lands
    root.update()                                             # idle pass
    assert ed.gate_tv.exists(iid)
    n = _count(ed, 'sample_a', root_id)
    assert _row(ed, 'sample_a', root_id).startswith(f"Cells  {n:,} · ")


def test_gate_rows_are_not_cut_off(env):
    """The tree column is as wide as its longest row (it scrolls sideways
    when that is wider than the panel)."""
    import tkinter.font as tkfont
    root, ed, tmp, _notes = env
    root_id, _child, _unnamed = _gated(root, ed, tmp)
    long_name = 'Viable single lymphocytes of the left scatter cloud ' * 2
    ed._sample_gates['sample_a'][root_id]['name'] = long_name.strip()
    ed._refresh_gate_list()
    text = _row(ed, 'sample_a', root_id)
    need = tkfont.nametofont('TkDefaultFont').measure(text)
    width = int(ed.gate_tv.column('#0', 'width'))
    assert width >= need > 212                    # wider than the default
    # A short tree keeps the default width.
    ed._sample_gates['sample_a'][root_id]['name'] = 'Cells'
    ed._refresh_gate_list()
    assert int(ed.gate_tv.column('#0', 'width')) < need


# ── clusters: a colour each, NA when the clustering is gone ─────────────────

def _clustered(root, ed, tmp):
    """A sample with a 'cluster' column of 4 ids (and the -1 noise bucket),
    imported as populations."""
    _load(root, ed, [('sample_a', _write_fcs(tmp / 'sample_a.fcs'))])
    ed._set_active_sample('sample_a')
    s = ed._samples['sample_a']
    rng = np.random.default_rng(3)
    s.data['cluster'] = rng.choice([-1, 0, 1, 2, 3], len(s.data))
    ed._data_changed('sample_a')
    ed._import_clusters()
    return s


def test_cluster_colouring_gives_each_population_its_colour(env):
    from matplotlib.colors import to_hex

    from openflo.pipeline import GATE_PALETTE, NOISE_CLUSTER_COLOR
    root, ed, tmp, _notes = env
    s = _clustered(root, ed, tmp)
    gates = ed._sample_gates['sample_a']
    # A population the user recoloured keeps its colour on the plot.
    c2 = next(g for g in gates.values() if g.get('cluster_id') == 2)
    c2['color'] = '#123456'
    ed.ax.clear()
    ed._plot_dot(['sample_a'], 'FSC-A', 'SSC-A', 'cluster')
    got = {}
    for coll in ed.ax.collections:
        fc = coll.get_facecolors()
        assert len({to_hex(c) for c in fc}) == 1    # one colour per scatter
        got[coll.get_label()] = (to_hex(fc[0]), len(coll.get_offsets()))
    want = {'Cluster 0': GATE_PALETTE[0], 'Cluster 1': GATE_PALETTE[1],
            'Cluster 2': '#123456', 'Cluster 3': GATE_PALETTE[3],
            'Unclustered (noise)': NOISE_CLUSTER_COLOR}
    assert {k: v[0] for k, v in got.items()} == want
    for cid, label in ((0, 'Cluster 0'), (-1, 'Unclustered (noise)')):
        assert got[label][1] == int((s.data['cluster'] == cid).sum())
    assert ed._cbar is None                         # a legend, no colour bar
    assert ed.ax.get_legend() is not None


def test_numeric_colouring_keeps_its_colour_scale(env):
    root, ed, tmp, _notes = env
    _clustered(root, ed, tmp)
    ed.ax.clear()
    ed._cbar = None
    ed._plot_dot(['sample_a'], 'FSC-A', 'SSC-A', 'FL1-A')
    assert len(ed.ax.collections) == 1
    assert ed.ax.collections[0].get_cmap().name == 'viridis'
    assert ed._cbar is not None


def test_cluster_population_without_its_column_reads_na():
    from openflo.pipeline import unknown_population_reason
    gates = {'g1': {'kind': 'group', 'name': 'clusters (2)', 'parent_id': None},
             'g2': {'kind': 'cluster', 'channel': 'cluster', 'cluster_id': 0,
                    'parent_id': 'g1'},
             'g3': {'kind': 'category', 'channel': 'flowsom_meta', 'value': 1,
                    'parent_id': None}}
    with_cols = ['FSC-A', 'FL1-A', 'cluster', 'flowsom_meta']
    without = ['FSC-A', 'FL1-A']
    assert unknown_population_reason(gates, 'g2', columns=with_cols) is None
    assert unknown_population_reason(gates, 'g3', columns=with_cols) is None
    assert "'cluster'" in unknown_population_reason(gates, 'g2', columns=without)
    assert "'flowsom_meta'" in unknown_population_reason(
        gates, 'g3', columns=without)
    # The container is not a population of the column.
    assert unknown_population_reason(gates, 'g1', columns=without) is None


def test_reopened_session_without_its_clustering_says_na(env):
    """The smoke test's 'clusters (13)' rows read 0 under a group of 20,000
    after an older autosave reopened: that sample's processed data was not
    there, so it came back from the bare FCS with no cluster column."""
    import glob

    from tests.test_session_restore_transforms import _restore
    root, ed, tmp, notes = env
    s = _clustered(root, ed, tmp)
    counts = {g['cluster_id']: _count(ed, 'sample_a', gid)
              for gid, g in ed._sample_gates['sample_a'].items()
              if g.get('kind') == 'cluster'}
    path = tmp / 'run.flowsession'
    ed._write_session(str(path))

    # With its processed data the session restores every count.
    ed2 = _restore(root, path, raw_first=True)
    gates2 = ed2._sample_gates['sample_a']
    assert {g['cluster_id']: _count(ed2, 'sample_a', gid)
            for gid, g in gates2.items() if g.get('kind') == 'cluster'} == counts
    ed2._pc_counts('sample_a', list(gates2))
    ed2._refresh_gate_list()
    c0 = next(gid for gid, g in gates2.items() if g.get('cluster_id') == 0)
    assert _row(ed2, 'sample_a', c0).endswith(
        f"{counts[0]:,} · {100 * counts[0] / len(s.data):.1f}%")

    # Without it: NA, and the load says why.
    for f in glob.glob(str(tmp / 'run_data' / '*')):
        os.remove(f)
    del notes[:]
    ed3 = _restore(root, path, raw_first=True)
    assert 'cluster' not in ed3._samples['sample_a'].data.columns
    gates3 = ed3._sample_gates['sample_a']
    ed3._pc_counts('sample_a', list(gates3))
    ed3._refresh_gate_list()
    for gid, g in gates3.items():
        if g.get('kind') == 'cluster':
            assert _row(ed3, 'sample_a', gid).endswith('  NA'), gid
    settle(root, 0.5)
    ed3._flush_load_notes()
    shown = '\n'.join(m for _t, m in notes)
    assert 'sample_a: the cluster results' in shown
    assert 'read NA' in shown


# ── groups, dialogs, order ──────────────────────────────────────────────────

def test_group_is_the_files_own_folder():
    from openflo.workspace import derive_trial_name
    j = os.path.join
    # Two folders under one parent are two groups, not the parent twice.
    assert derive_trial_name(j('root', 'synth', 'pbmc', 'a.fcs')) == 'pbmc'
    assert derive_trial_name(
        j('root', 'synth', 'compensation', 'a.fcs')) == 'compensation'
    # The example data's folder, never the hidden one above it.
    assert derive_trial_name(
        j('home', '.openflo', 'example_data', 'a.fcs')) == 'example_data'
    assert derive_trial_name(j('home', 'data', '.cache', 'a.fcs')) == 'data'
    # A generic acquisition host still defers to the folder above it.
    assert derive_trial_name(
        j('root', 'Run_7', 'Blank Experiment with Tubes', 'a.fcs')) == 'Run_7'
    assert derive_trial_name(
        j('root', 'Run_7', 'Experiment 2', 'a.fcs')) == 'Experiment 2'


def test_file_dialog_never_opens_in_the_package(tmp_path, monkeypatch):
    import openflo
    from openflo.paths import dialog_start_dir
    pkg = os.path.dirname(os.path.abspath(openflo.__file__))
    home = tmp_path / 'home'
    home.mkdir()
    data = tmp_path / 'data'
    data.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    # The last folder used wins while it exists.
    assert dialog_start_dir(str(data), pkg) == str(data)
    # Otherwise the current folder...
    monkeypatch.chdir(data)
    assert dialog_start_dir(str(tmp_path / 'gone'), pkg) == str(data)
    assert dialog_start_dir(None, pkg) == str(data)
    # ...but never the package's own folder or one inside it: home instead.
    monkeypatch.chdir(pkg)
    assert dialog_start_dir(None, pkg) == str(home)
    monkeypatch.chdir(os.path.join(pkg, 'template_library'))
    assert dialog_start_dir(None, pkg) == str(home)


def test_add_fcs_dialog_starts_outside_the_package(env, monkeypatch):
    from tkinter import filedialog
    root, ed, tmp, _notes = env
    seen = []
    monkeypatch.setattr(filedialog, 'askopenfilenames',
                        lambda **k: seen.append(k['initialdir']) or ())
    import openflo
    pkg = os.path.dirname(os.path.abspath(openflo.__file__))
    monkeypatch.chdir(pkg)                      # started from the package
    ed._add_samples()
    assert seen[-1] == os.path.expanduser('~') and seen[-1] != pkg
    # It then remembers the folder files were picked from.
    data = tmp / 'run1'
    data.mkdir()
    picked = _write_fcs(data / 'b.fcs')
    monkeypatch.setattr(filedialog, 'askopenfilenames',
                        lambda **k: seen.append(k['initialdir']) or (picked,))
    ed._queue_fcs_loads = lambda paths, front_names=(): None
    ed._add_samples()
    ed._add_samples()
    assert seen[-1] == str(data)


def test_samples_are_listed_by_name(env):
    root, ed, tmp, _notes = env
    names = ['treat_d1', 'ctrl_d10', 'ctrl_d2', 'ctrl_d1']
    _load(root, ed, [(n, _write_fcs(tmp / f'{n}.fcs', n=400, seed=i))
                     for i, n in enumerate(names)])
    ed._on_load_settled()
    assert ed._sample_order == ['ctrl_d1', 'ctrl_d2', 'ctrl_d10', 'treat_d1']
    trial = ed._trial_iid(ed._sample_trial['ctrl_d1'])
    rows = [ed._parse_iid(i)[1] for i in ed.gate_tv.get_children(trial)]
    assert rows == ['ctrl_d1', 'ctrl_d2', 'ctrl_d10', 'treat_d1']
