"""Analyze -> Annotate populations names clusters by phenotype, and only the
clusters it was asked about.

annotate.py's pure functions are unit-tested, but the window that drives them
(ui_annotation.PopulationAnnotationWindow: detector->antibody relabel, MEM,
reference matching, write-back through editor._apply_population_names) never
ran under the suite. Four synthetic blobs with known phenotypes give a known
answer under the window's default reference table.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

MARKERS = ['CD3-A', 'CD4-A', 'CD8-A', 'CD19-A']
PHENO = {0: (1, 1, 0, 0), 1: (1, 0, 1, 0), 2: (0, 0, 0, 1), 3: (0, 0, 0, 0)}
EXPECT = {0: 'CD4 T', 1: 'CD8 T', 2: 'B cell', 3: 'unknown'}


def _sample(with_cluster_col):
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    blocks, pops = [], []
    for p, ph in PHENO.items():
        blocks.append(np.column_stack([rng.normal(8 if on else 1, 0.5, 300)
                                       for on in ph]))
        pops += [p] * 300
    df = pd.DataFrame(np.vstack(blocks), columns=MARKERS)
    df['leiden'] = np.array(pops)
    if with_cluster_col:
        # A PhenoGraph run on the same sample numbers the same cells differently.
        df['cluster'] = (np.array(pops) + 1) % 4
    return FlowSample.from_dataframe(
        df, name='pbmc', labels={m: m.split('-')[0] for m in MARKERS})


def _editor_with(s):
    root, ed, _gui = _editor_or_skip()
    name = s.name
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._sample_gates[name] = {}
    ed._sample_gate_order[name] = []
    ed._sample_gate_seq[name] = 0
    ed._channels = list(s.data.columns)
    ed._channel_labels = dict(s.channel_labels)
    ed._active_sample = name
    return root, ed


def _annotate_leiden(ed):
    from openflo.ui_annotation import PopulationAnnotationWindow
    w = PopulationAnnotationWindow(ed, 'pbmc')
    w.withdraw()
    w.col_var.set('leiden')
    w._compute()
    w._apply()
    return w


def test_window_names_each_cluster_by_its_phenotype():
    root, ed = _editor_with(_sample(with_cluster_col=False))
    try:
        ed._import_populations('leiden')
        w = _annotate_leiden(ed)
        # MEM is computed on antibody names, not detector names.
        assert list(w._mem.columns) == ['CD3', 'CD4', 'CD8', 'CD19']
        assigned = {int(i): w._tv.set(i, 'name') for i in w._tv.get_children()}
        assert float(w.state_thr_var.get()) == 3.0   # at the default
        assert assigned == EXPECT
        leiden = {int(g['value']): g['name']
                  for g in ed._sample_gates['pbmc'].values()
                  if g.get('kind') == 'category' and g.get('channel') == 'leiden'}
        assert leiden[0] == 'CD4 T' and leiden[1] == 'CD8 T'
        assert leiden[2] == 'B cell'
        assert leiden[3] == 'leiden 3'          # 'unknown' is not written back
    finally:
        root.destroy()


def _named_with_states(ed, mem, state_thr=None):
    """Run Assign names on a planted MEM table (pops x markers)."""
    from openflo.ui_annotation import PopulationAnnotationWindow
    w = PopulationAnnotationWindow(ed, 'pbmc')
    w.withdraw()
    w.col_var.set('leiden')
    w._compute()
    w._mem = mem
    if state_thr is not None:
        w.state_thr_var.set(state_thr)
    w._apply()
    return w, {int(i): w._tv.set(i, 'name') for i in w._tv.get_children()}


def _weak_cd19_mem(cd19):
    # Population 2 is clearly CD3- with a weak CD19 score; the others carry
    # no markers. Under the default table that is 'B cell' only if CD19 counts.
    mem = pd.DataFrame(0.0, index=[0, 1, 2, 3],
                       columns=['CD3', 'CD4', 'CD8', 'CD19'])
    mem.loc[2, 'CD3'] = -8.0
    mem.loc[2, 'CD19'] = cd19
    return mem


def test_state_threshold_defaults_to_population_states_and_is_separate():
    """Reference matching uses its own State threshold, 3 like
    annotate.population_states, not the MEM label threshold (2): a CD19 score
    of 2.5 does not make a CD3- population a B cell."""
    import inspect

    from openflo.annotate import population_states
    root, ed = _editor_with(_sample(with_cluster_col=False))
    try:
        ed._import_populations('leiden')
        w, named = _named_with_states(ed, _weak_cd19_mem(2.5))
        dflt = inspect.signature(population_states).parameters['threshold']
        assert float(w.state_thr_var.get()) == dflt.default == 3.0
        assert float(w.thr_var.get()) == 2.0     # MEM label field unchanged
        assert named[2] == 'unknown'
        # The added field keeps the window within 1024x768.
        w.update_idletasks()
        assert w.winfo_reqwidth() <= 1024 and w.winfo_reqheight() <= 768
        # The same population with CD19 above 3 is a B cell.
        _w, named = _named_with_states(ed, _weak_cd19_mem(4.0))
        assert named[2] == 'B cell'
    finally:
        root.destroy()


def test_state_threshold_below_two_is_used_as_given():
    """No floor at 2: a State threshold of 1.5 lets a CD19 score of 1.5
    count, so the population is named B cell."""
    root, ed = _editor_with(_sample(with_cluster_col=False))
    try:
        ed._import_populations('leiden')
        _w, named = _named_with_states(ed, _weak_cd19_mem(1.5), '1.5')
        assert named[2] == 'B cell'
    finally:
        root.destroy()


def test_annotating_leiden_leaves_phenograph_clusters_alone():
    # Naming Leiden populations must not relabel the PhenoGraph clusters: the
    # two id spaces partition the same cells differently.
    root, ed = _editor_with(_sample(with_cluster_col=True))
    try:
        ed._import_populations('cluster')
        ed._import_populations('leiden')
        _annotate_leiden(ed)
        cluster_names = {g['cluster_id']: g['name']
                         for g in ed._sample_gates['pbmc'].values()
                         if g.get('kind') == 'cluster'}
        assert cluster_names == {c: f'Cluster {c}' for c in range(4)}, (
            f'PhenoGraph clusters renamed by a Leiden annotation: {cluster_names}')
        assert ed._cluster_display_name('pbmc', 0) == 'Cluster 0'
    finally:
        root.destroy()


def _generic_sample():
    """Two planted populations on generic channels: Leiden 0 is FL1-A+ FL2-A-,
    Leiden 1 is FL1-A- FL2-A+."""
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(1)
    hi_lo = np.column_stack([rng.normal(8, 0.5, 300), rng.normal(1, 0.5, 300)])
    lo_hi = np.column_stack([rng.normal(1, 0.5, 300), rng.normal(8, 0.5, 300)])
    df = pd.DataFrame(np.vstack([hi_lo, lo_hi]), columns=['FL1-A', 'FL2-A'])
    df['leiden'] = np.repeat([0, 1], 300)
    return FlowSample.from_dataframe(df, name='sample_a')


def test_assign_names_is_one_undo_step_that_undoes_only_the_naming():
    """Assign names says it can be undone. It adds exactly one undo
    step, and Ctrl+Z restores the old names while the imported populations
    stay (before, the stack did not grow and Undo removed the import)."""
    from openflo.ui_annotation import PopulationAnnotationWindow
    root, ed = _editor_with(_generic_sample())
    try:
        ed._import_populations('leiden')
        root.update()                    # end the import's undo coalescing

        def leiden_names():
            return {int(g['value']): g['name']
                    for g in ed._sample_gates['sample_a'].values()
                    if g.get('kind') == 'category'
                    and g.get('channel') == 'leiden'}
        assert leiden_names() == {0: 'leiden 0', 1: 'leiden 1'}
        n_gates, n_undo = len(ed._sample_gates['sample_a']), len(ed._undo_stack)
        w = PopulationAnnotationWindow(ed, 'sample_a')
        w.withdraw()
        w.col_var.set('leiden')
        w._compute()
        w.ref_txt.delete('1.0', 'end')
        w.ref_txt.insert('1.0', "TypeB: FL2-A+ FL1-A-\n")
        w._apply()
        assert leiden_names() == {0: 'leiden 0', 1: 'TypeB'}
        assert len(ed._undo_stack) == n_undo + 1
        root.update()
        ed._undo()
        assert leiden_names() == {0: 'leiden 0', 1: 'leiden 1'}
        assert len(ed._sample_gates['sample_a']) == n_gates
        w.destroy()
    finally:
        root.destroy()


def test_mem_export_to_a_folder_that_does_not_exist_says_so(monkeypatch,
                                                            tmp_path):
    """Export MEM CSV… to a path whose folder does not exist shows "could
    not be saved" instead of a traceback; nothing is written or recorded."""
    from tkinter import messagebox

    import openflo.ui_annotation as ua
    root, ed = _editor_with(_sample(with_cluster_col=False))
    try:
        ed._import_populations('leiden')
        w = _annotate_leiden(ed)
        assert w._mem is not None
        errors, recorded = [], []
        monkeypatch.setattr(messagebox, 'showerror',
                            lambda title, msg, **k: errors.append(msg))
        monkeypatch.setattr(ua, 'record_export',
                            lambda *a, **k: recorded.append(a))
        out = tmp_path / 'no_such_folder' / 'mem.csv'
        monkeypatch.setattr(ua.filedialog, 'asksaveasfilename',
                            lambda **k: str(out))
        w._export_csv()
        assert len(errors) == 1, errors
        assert errors[0].startswith('The table could not be saved.')
        assert str(out) in errors[0]
        assert not out.parent.exists() and recorded == []
    finally:
        root.destroy()


def _three_way_sample():
    """Three planted populations on generic channels in 'leiden' (300, 200
    and 100 events) and a 'cluster' column that splits the same events in
    two (FL1-A+ 300, the rest 300)."""
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(2)
    blocks = [np.column_stack([rng.normal(a, 0.5, n), rng.normal(b, 0.5, n)])
              for a, b, n in ((8, 1, 300), (1, 8, 200), (1, 1, 100))]
    df = pd.DataFrame(np.vstack(blocks), columns=['FL1-A', 'FL2-A'])
    df['leiden'] = np.repeat([0, 1, 2], [300, 200, 100])
    df['cluster'] = np.repeat([0, 1], 300)
    return FlowSample.from_dataframe(df, name='sample_a')


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_cluster_column_and_threshold_refresh_the_table(monkeypatch):
    """Choosing another Cluster column (or MEM threshold) recomputes the
    table, and Assign names that column's populations, not the old one's."""
    from tkinter import messagebox

    from openflo.annotate import mem_label
    from openflo.ui_annotation import PopulationAnnotationWindow
    root, ed = _editor_with(_three_way_sample())
    try:
        w = PopulationAnnotationWindow(ed, 'sample_a')
        w.withdraw()

        def rows():
            return {int(i): int(w._tv.set(i, 'n'))
                    for i in w._tv.get_children()}
        assert w.col_var.get() == 'leiden' and rows() == {0: 300, 1: 200,
                                                          2: 100}
        combo = next(c for c in _walk(w) if c.winfo_class() == 'TCombobox'
                     and str(c.cget('textvariable')) == str(w.col_var))
        w.col_var.set('cluster')
        combo.event_generate('<<ComboboxSelected>>')
        assert rows() == {0: 300, 1: 300}
        named = []
        monkeypatch.setattr(type(ed), '_apply_population_names',
                            lambda self, s, col, names: named.append(col))
        w.ref_txt.delete('1.0', 'end')
        w.ref_txt.insert('1.0', "TypeA: FL1-A+\n")
        w._apply()
        assert named == ['cluster']
        # Set from code, with no selection event: Assign still names the
        # column the box shows.
        w.col_var.set('leiden')
        w._apply()
        assert named == ['cluster', 'leiden']
        assert rows() == {0: 300, 1: 200, 2: 100}
        # The threshold relabels at once, as MEM labels at that threshold.
        spin = next(c for c in _walk(w) if c.winfo_class() == 'TSpinbox'
                    and str(c.cget('textvariable')) == str(w.thr_var))
        seen = []
        for thr in ('0', '9'):
            w.thr_var.set(thr)
            spin.event_generate('<FocusOut>')
            want = {int(str(p)): mem_label(r, threshold=float(thr))
                    for p, r in w._mem.iterrows()}
            assert {int(i): w._tv.set(i, 'mem')
                    for i in w._tv.get_children()} == want
            seen.append(want)
        assert seen[0] != seen[1]               # the two thresholds differ
        # Text that isn't a number: said in the window, no dialog.
        monkeypatch.setattr(messagebox, 'showerror',
                            lambda *a, **k: pytest.fail('a dialog'))
        w.thr_var.set('x')
        spin.event_generate('<FocusOut>')
        assert 'must be a number' in w.status.cget('text')
        w.destroy()
    finally:
        root.destroy()
