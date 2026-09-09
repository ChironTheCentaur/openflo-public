"""The differential-abundance table must not present an unsupported p-value
as if it were evidence.

`differential_abundance` fits `count ~ group` with a dispersion shared across
clusters, so it still returns a p-value when there is ONE sample per group —
and the window rendered that as a significance marker with no per-group `n`
anywhere on screen. A 1-vs-1 comparison therefore looked exactly like a
replicated one, in a table users export straight into a figure.

The numbers are unchanged; what changed is that `n` is shown and a design too
small to support the p-values says so.
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from openflo.diffexp import differential_abundance  # noqa: E402


def _rows(counts, groups):
    return differential_abundance(pd.DataFrame(counts), groups,
                                  cluster_names=[f'P{i}'
                                                 for i in range(len(counts))])


def test_the_result_carries_the_per_group_sample_count():
    rows = _rows([[100, 110, 60, 58], [50, 52, 90, 95]],
                 ['a', 'a', 'b', 'b'])
    assert rows and rows[0]['n_a'] == 2 and rows[0]['n_b'] == 2


def test_one_sample_per_group_still_produces_a_significant_p_value():
    """Pinning the motivation, not endorsing it: the model DOES return a small
    p here, which is why the count has to be visible."""
    rows = _rows([[100, 60], [50, 90]], ['a', 'b'])
    assert min(r['p_adj'] for r in rows) < 0.05
    assert all(r['n_a'] == 1 and r['n_b'] == 1 for r in rows)


def _window_or_skip(rows, groups):
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip('tkinter not available')
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:                            # noqa: BLE001
        pytest.skip(str(e))
    from openflo.ui_diff import DiffAbundanceWindow
    w = DiffAbundanceWindow(root, rows, groups)
    w.withdraw()
    return root, w


def _labels(widget):
    """Every piece of user-visible text: Label/Button captions AND the
    Treeview's column headings, which `cget('text')` does not reach."""
    import tkinter.ttk as ttk
    out = []
    for child in widget.winfo_children():
        try:
            out.append(str(child.cget('text')))
        except Exception:                             # noqa: BLE001
            pass
        if isinstance(child, ttk.Treeview):
            for col in child.cget('columns'):
                try:
                    out.append(str(child.heading(col, 'text')))
                except Exception:                     # noqa: BLE001
                    pass
        out.extend(_labels(child))
    return out


def test_the_table_shows_n_per_group():
    rows = _rows([[100, 110, 60, 58], [50, 52, 90, 95]],
                 ['a', 'a', 'b', 'b'])
    root, w = _window_or_skip(rows, ('a', 'b'))
    try:
        headings = _labels(w)
        assert 'na' in headings and 'nb' in headings, (
            'no per-group n column in the table — a 1-vs-1 comparison would '
            f'look identical to a replicated one: {headings}')
    finally:
        root.destroy()


def test_a_one_versus_one_design_is_called_out():
    rows = _rows([[100, 60], [50, 90]], ['a', 'b'])
    root, w = _window_or_skip(rows, ('a', 'b'))
    try:
        text = ' '.join(_labels(w))
        assert 'exploratory' in text, (
            'a 1-vs-1 comparison rendered significance stars with no caution '
            f'anywhere on screen: {text[:250]}')
    finally:
        root.destroy()


def test_a_replicated_design_is_not_nagged():
    """The caution must not fire on a design that does support its p-values."""
    rows = _rows([[100, 110, 105, 60, 58, 62], [50, 52, 48, 90, 95, 92]],
                 ['a', 'a', 'a', 'b', 'b', 'b'])
    root, w = _window_or_skip(rows, ('a', 'b'))
    try:
        assert 'exploratory' not in ' '.join(_labels(w))
    finally:
        root.destroy()
