"""The differential-abundance window piece by piece: heading, toolbar and
its tip, the table's columns, the two bottom notes (and which sits lowest)
and each row's formatted values.

Result rows are written by hand (no model fit), so each case reaches one
branch: a paired heading, NaN fold change / proportion / p, a missing n,
a population missing from some samples or from a whole group. Generated
group and population names only.
"""
import tkinter as tk
from tkinter import ttk

import pytest

from tests.conftest import gui_unavailable

NAN = float('nan')


def _row(name, **kw):
    r = {'cluster': name, 'log2fc': 1.0, 'prop_a': 0.1, 'prop_b': 0.2,
         'n_a': 4, 'n_b': 4, 'p': 0.01, 'p_adj': 0.02}
    r.update(kw)
    return r


class _Parent(tk.Frame):
    """Records the tips auto_tip attaches (it looks for ``_tip``)."""

    def __init__(self, root):
        super().__init__(root)
        self.tips = []

    def _tip(self, widget, text):
        self.tips.append((widget, text))


@pytest.fixture
def make():
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError as e:
        gui_unavailable(str(e))
    from openflo.ui_diff import DiffAbundanceWindow

    def build(rows, groups=('G1', 'G2'), samples=None):
        parent = _Parent(root)
        w = DiffAbundanceWindow(parent, rows, list(groups), samples=samples)
        w.withdraw()
        return parent, w
    yield build
    root.destroy()


def _kids(w, cls):
    return [c for c in w.winfo_children() if isinstance(c, cls)]


def _heading(w):
    return _kids(w, ttk.Label)[0].cget('text')


def _pad(widget):
    return ' '.join(map(str, widget.cget('padding')))


def _note_labels(w):
    return [c for c in w.pack_slaves()
            if isinstance(c, ttk.Label) and c.pack_info()['side'] == 'bottom']


def _notes(w):
    """The bottom-packed labels' texts in packing order (first = lowest);
    each is a muted, left-justified, wrapped note filling the width."""
    labels = _note_labels(w)
    for c in labels:
        assert (str(c.cget('style')), int(c.cget('wraplength')),
                str(c.cget('justify')), _pad(c),
                c.pack_info()['fill']) == ('Muted.TLabel', 760, 'left',
                                           '8 4', 'x')
    return [str(c.cget('text')) for c in labels]


def test_heading_names_the_groups_and_population_count(make):
    _p, w = make([_row('P1'), _row('P2')])
    text = _heading(w)
    assert text.startswith('G2 vs G1 — negative-binomial GLM')
    assert 'log2FC = log2(G2 / G1). 2 populations.' in text
    assert 'Paired' not in text
    lab = _kids(w, ttk.Label)[0]
    assert (str(lab.cget('font')), lab.pack_info()['side']) == (
        'OpenFloHeading', 'top')
    assert (int(lab.cget('wraplength')), str(lab.cget('justify')),
            _pad(lab)) == (760, 'left', '6')
    assert lab.pack_info()['fill'] == 'x'
    assert w.title() == 'Differential abundance (NB-GLM)'


def test_window_asks_for_its_size(make, monkeypatch):
    from openflo.ui_diff import DiffAbundanceWindow
    asked = []
    monkeypatch.setattr(DiffAbundanceWindow, 'geometry',
                        lambda self, *a: asked.append(a))
    make([_row('P1')])
    assert asked == [('780x460',)]


def test_heading_counts_donors_for_a_paired_design(make):
    _p, w = make([_row('P1', n_blocks=3)])
    assert _heading(w).endswith(
        '1 populations. Paired: the donor is a term in the model (3 donors).')


def test_export_button_and_its_tip(make):
    parent, w = make([_row('P1')], samples=['s1', 's2'])
    (bar,) = _kids(w, ttk.Frame)
    (btn,) = bar.winfo_children()
    assert btn.cget('text') == 'Export CSV…'
    assert btn.pack_info()['side'] == 'right'
    assert (btn.pack_info()['padx'], btn.pack_info()['pady']) == (6, 4)
    assert bar.pack_info()['fill'] == 'x'
    assert [(t[0], t[1]) for t in parent.tips] == [
        (btn, 'Save this comparison table, including the per-group n that '
              'the significance markers depend on.')]
    assert w._sample_names == ['s1', 's2']


def test_table_columns_name_the_groups(make):
    rows = [_row('P1')]
    _p, w = make(rows, groups=('ctl', 'trt'))
    (tv,) = _kids(w, ttk.Treeview)
    cols = tv.cget('columns')
    assert [tv.heading(c, 'text') for c in cols] == [
        'Population', 'log2FC', '%ctl', '%trt', 'nctl', 'ntrt', 'missing',
        'p', 'p(adj)', '']
    assert [int(tv.column(c, 'width')) for c in cols] == [
        280, 70, 70, 70, 45, 45, 55, 80, 80, 40]
    assert [bool(tv.column(c, 'stretch')) for c in cols] == (
        [True] + [False] * 9)
    assert {str(tv.column(c, 'anchor')) for c in cols} == {'w'}
    assert ' '.join(map(str, tv.cget('show'))) == 'headings'
    assert (tv.pack_info()['fill'], tv.pack_info()['expand']) == ('both', 1)
    assert w._rows is rows


def test_rows_format_each_value(make):
    rows = [_row('P1', log2fc=-0.5, prop_a=0.123, prop_b=NAN, n_b=None,
                 n_missing=2, p=0.0123, p_adj=0.0004),
            _row('P2', log2fc=NAN, p=NAN, p_adj=NAN, n_missing=0),
            {'cluster': 7, 'log2fc': 2.0, 'prop_a': 0.0, 'prop_b': 1.0,
             'p': 0.5, 'p_adj': 0.6}]
    _p, w = make(rows)
    (tv,) = _kids(w, ttk.Treeview)
    got = [tuple(str(v) for v in tv.item(i, 'values'))
           for i in tv.get_children('')]
    assert got == [
        ('P1', '-0.50', '12.30', 'missing', '4', 'None', '2', '0.012',
         '0.0004', '***'),
        ('P2', 'n/a', '10.00', '20.00', '4', '4', '', 'n/a', 'n/a', ''),
        ('7', '+2.00', '0.00', '100.00', '', '', '', '0.5', '0.6', 'ns'),
    ]


def test_one_sample_per_group_says_there_are_no_p_values(make):
    _p, w = make([_row('P1', n_a=1, n_b=1, p=NAN, p_adj=NAN)])
    (note,) = _notes(w)
    assert note.startswith('⚠ One sample per group')


def test_a_small_group_is_called_out_with_its_size(make):
    _p, w = make([_row('P1', n_a=2, n_b=5), _row('P2', n_a=6, n_b=6)])
    (note,) = _notes(w)
    assert note.startswith('⚠ Only 2 sample(s) in the smaller group.')


def test_a_population_absent_from_a_group_does_not_set_the_design(make):
    # P2 has no samples in G2: it says nothing about the design (n 4 vs 4).
    _p, w = make([_row('P1'), _row('P2', n_b=0, n_missing=4)])
    assert _notes(w) == [
        '1 population(s) are missing from some samples (gate not drawn '
        'there). Each was fitted on the samples that have it — a missing '
        'population is not 0 events. Not testable, absent from a whole '
        'group: P2.']


def test_both_notes_with_the_design_note_lowest(make):
    _p, w = make([_row('P1', n_a=2, n_b=2, n_missing=1),
                  _row('P2', n_a=2, n_b=2)])
    notes = _notes(w)
    assert len(notes) == 2
    assert notes[0].startswith('⚠ Only 2 sample(s)')
    assert notes[1] == (
        '1 population(s) are missing from some samples (gate not drawn '
        'there). Each was fitted on the samples that have it — a missing '
        'population is not 0 events.')


def test_no_rows_no_notes(make):
    _p, w = make([])
    assert _notes(w) == []
    assert _heading(w).endswith('0 populations.')
    (tv,) = _kids(w, ttk.Treeview)
    assert tv.get_children('') == ()


def test_replicated_rows_with_one_nan_p_say_nothing_about_p_values(make):
    # n 4 vs 4 everywhere: one population without a p is not the
    # one-sample-per-group design, so no note.
    _p, w = make([_row('P1', p=NAN, p_adj=NAN), _row('P2'), _row('P3')])
    assert _notes(w) == []


def test_parts_are_packed_heading_bar_table_then_notes(make):
    _p, w = make([_row('P1', n_a=2, n_b=2, n_missing=1)])
    got = [(type(c).__name__, c.pack_info()['side']) for c in w.pack_slaves()]
    assert got == [('Label', 'top'), ('Frame', 'top'), ('Treeview', 'top'),
                   ('Label', 'bottom'), ('Label', 'bottom')]
    assert w.pack_slaves()[0] is _kids(w, ttk.Label)[0]


def test_donor_count_is_read_from_the_first_row(make):
    _p, w = make([_row('P1', n_blocks=3), _row('P2', n_blocks=5)])
    assert _heading(w).endswith('(3 donors).')
