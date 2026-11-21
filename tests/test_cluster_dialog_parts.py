"""The Cluster dialog after its build was cut into row builders (R6 C3).

AnalysisMixin._open_cluster_dialog now calls six builders that return the
Tk variables and widgets they make. These checks drive the dialog through
its widgets and the Tcl variables they are bound to, so a builder whose
variable is not kept (tkinter unsets a tk.Variable when its last Python
reference goes), a row missing from the method rows, or Run reading another
object than the box shows, fails here.

Generated data, sample names s1/s2 and generic channels FL1-A ... only.
"""
from __future__ import annotations

import gc
import tkinter as tk
import tkinter.messagebox

import numpy as np
import pandas as pd
import pytest

from tests.conftest import editor_or_skip

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL1-H', 'FL2-A', 'FL3-H', 'FL4-W']
DEFAULT = ['FL1-A', 'FL2-A', 'FL3-H']        # one per marker, never -W
METHODS = {'PhenoGraph': 'phenograph', 'FlowSOM': 'flowsom',
           'Leiden': 'leiden'}
SHOWN = {'phenograph': {'k', 'resolution', 'repro'},
         'flowsom': {'grid', 'meta'},
         'leiden': {'k', 'resolution', 'fast'}}
REPRO = 'Reproducible (seeded PhenoGraph)'
ALL = 'Run on all loaded samples'


def _add(ed, name, n):
    from openflo.flow_sample import FlowSample
    df = pd.DataFrame(np.random.default_rng(n).normal(100, 10, (n, len(CH))),
                      columns=CH)
    ed._samples[name] = FlowSample.from_dataframe(df, name=name,
                                                  path=f'/data/{name}.fcs')
    ed._sample_order.append(name)
    ed._active_sample = name


@pytest.fixture
def ed():
    root, editor = editor_or_skip(accept_confirms=False)
    editor._available_embeddings = lambda: (['UMAP', 't-SNE'], ['PHATE'])
    _add(editor, 's1', 400)
    _add(editor, 's2', 250)
    try:
        yield editor
    finally:
        root.destroy()


def _walk(w):
    out = [w]
    for c in w.winfo_children():
        out += _walk(c)
    return out


class Dialog:
    """The open Cluster dialog, found by its widgets."""

    def __init__(self, ed):
        before = set(map(str, ed.winfo_children()))
        ed._open_cluster_dialog()
        gc.collect()                   # a variable only a helper held dies
        ed.update()
        (self.dlg,) = [w for w in ed.winfo_children()
                       if isinstance(w, tk.Toplevel) and str(w) not in before]
        self.ed = ed
        ws = _walk(self.dlg)
        self.radios = {w.cget('text'): w for w in ws
                       if w.winfo_class() == 'TRadiobutton'}
        self.checks = {w.cget('text'): w for w in ws
                       if w.winfo_class() == 'TCheckbutton'}
        self.spins = {(float(w.cget('from')), float(w.cget('to'))): w
                      for w in ws if w.winfo_class() == 'TSpinbox'}
        combos = [w for w in ws if w.winfo_class() == 'TCombobox']
        (self.events,) = [w for w in combos
                          if 'Custom…' in tuple(w.cget('values'))]
        (self.dims,) = [w for w in combos
                        if tuple(map(str, w.cget('values'))) == ('2', '3')]
        (self.embed,) = [w for w in combos
                         if w is not self.events and w is not self.dims]
        (self.number,) = [w for w in ws if w.winfo_class() == 'TEntry']
        (self.markers,) = [w for w in ws if w.winfo_class() == 'Listbox']
        (self.run,) = [w for w in ws if w.winfo_class() == 'TButton'
                       and w.cget('text') == 'Run']

    def var(self, widget, option='variable'):
        return str(widget.cget(option))

    def setvar(self, widget, value, option='variable'):
        """Set the Tcl variable the widget shows, as the widget itself
        would on a click or a keystroke."""
        self.ed.setvar(self.var(widget, option), value)
        self.ed.update()

    def shown(self):
        return {k for k, ws in self.ed._cluster_method_rows.items()
                if all(w.winfo_manager() == 'grid' for w in ws)}

    def press_run(self):
        got = []
        self.ed._run_clustering = lambda **kw: got.append(kw)
        self.run.invoke()
        return got


def test_every_box_has_a_live_variable_after_the_build(ed):
    d = Dialog(ed)
    boxes = ([(w, 'variable') for w in d.radios.values()]
             + [(w, 'variable') for w in d.checks.values()]
             + [(w, 'textvariable') for w in d.spins.values()]
             + [(d.events, 'textvariable'), (d.embed, 'textvariable'),
                (d.dims, 'textvariable'), (d.number, 'textvariable')])
    names = {d.var(w, opt) for w, opt in boxes}
    # method, k, resolution, grid, metaclusters, fast graph, embedding,
    # components, events, number, all samples, reproducible
    assert len(names) == 12
    for name in names:
        assert ed.tk.call('info', 'exists', name), name
    assert d.events.get() == 'Default'
    assert d.embed.get() == 'UMAP'
    assert ed.getvar(d.var(d.checks[REPRO])) in (1, True, '1')


def test_rows_follow_the_method_and_reproducible_through_their_traces(ed):
    d = Dialog(ed)
    assert set(ed._cluster_method_rows) == {'k', 'resolution', 'grid',
                                            'meta', 'fast', 'repro'}
    assert d.shown() == SHOWN['phenograph']
    for value in ('flowsom', 'leiden', 'phenograph'):
        # Through the Tcl variable, as the radio button writes it: the
        # trace must still be there.
        d.setvar(d.radios['FlowSOM'], value)
        assert d.shown() == SHOWN[value]
    res = d.spins[(0.1, 5.0)]
    assert str(res.cget('state')) == 'normal'
    assert not ed._cluster_resolution_note.winfo_manager()
    d.setvar(d.checks[REPRO], 0)
    assert str(res.cget('state')) == 'disabled'
    assert ed._cluster_resolution_note.winfo_manager() == 'grid'
    d.setvar(d.radios['Leiden'], 'leiden')
    assert str(res.cget('state')) == 'normal'
    assert not ed._cluster_resolution_note.winfo_manager()


def test_the_metacluster_note_shows_only_with_flowsom(ed):
    d = Dialog(ed)
    (note,) = [w for w in _walk(d.dlg) if w.winfo_class() == 'TLabel'
               and 'populations you expect' in str(w.cget('text'))]
    assert note in ed._cluster_method_rows['meta']
    assert ed._cluster_marker_list is d.markers
    for label, value in METHODS.items():
        d.setvar(d.radios[label], value)
        assert (note.winfo_manager() == 'grid') == (value == 'flowsom')


def test_the_reproducible_box_is_a_method_row_on_the_first_pass(ed,
                                                                 monkeypatch):
    # rows['repro'] is added before _show_method_settings first runs, so
    # that pass places it (a bare .grid() after its own placement).
    calls = []
    orig = tk.Grid.grid_configure

    def grid(w, cnf=None, **kw):
        if w.winfo_class() == 'TCheckbutton' and w.cget('text') == REPRO:
            calls.append(dict(kw))
        return orig(w, cnf or {}, **kw)
    monkeypatch.setattr(tk.Grid, 'grid', grid)
    Dialog(ed)
    assert len(calls) == 2 and calls[0] and calls[1] == {}


def _ident(w):
    if w.winfo_class() == 'TSpinbox':
        return float(w.cget('from'))
    return str(w.cget('text')) if 'text' in w.keys() else ''


def test_rows_are_made_in_order_so_tab_moves_down_the_dialog(ed):
    # Keyboard focus follows the order the widgets were made in: the rows
    # must still be built top to bottom.
    d = Dialog(ed)
    assert [(w.winfo_class(), _ident(w)) for w in d.dlg.winfo_children()] == [
        ('TFrame', ''),
        ('TLabel', 'Neighbors (k):'), ('TSpinbox', 5.0),
        ('TLabel', 'Resolution:'), ('TSpinbox', 0.1),
        ('TLabel', 'FlowSOM grid (N×N):'), ('TSpinbox', 4.0),
        ('TLabel', 'FlowSOM metaclusters:'), ('TSpinbox', 2.0),
        ('TLabel', '(PhenoGraph: needs Reproducible)'),
        ('TCheckbutton', 'Fast graph'),
        ('TLabel', '(set above the populations you expect)'),
        ('TLabel', 'Embedding (for visualization):'), ('TFrame', ''),
        ('TLabel', '(PHATE need pip install openflo[embed])'),
        ('TLabel', 'Events / sample:'), ('TFrame', ''),
        ('TLabel', 'Markers:'), ('TFrame', ''),
        ('TCheckbutton', ALL), ('TCheckbutton', REPRO),
        ('TFrame', '')]
    assert [_ident(w) for w in d.radios.values()] == sorted(
        METHODS, key=['PhenoGraph', 'FlowSOM', 'Leiden'].index)


def test_typing_in_the_number_box_switches_to_custom(ed):
    d = Dialog(ed)
    d.number.event_generate('<Button-1>', when='now')
    ed.update()
    assert d.events.get() == 'Custom…'


@pytest.mark.parametrize('label', sorted(METHODS))
@pytest.mark.parametrize('mode,number,cap', [
    ('Default', '', None), ('All events', '', None),
    ('Smallest in group', '', None),
    ('Custom…', '1,500', 1500)])
def test_run_queues_the_job_for_each_method_and_events_mode(
        ed, label, mode, number, cap):
    method = METHODS[label]
    d = Dialog(ed)
    d.setvar(d.radios[label], method)
    d.setvar(d.events, mode, 'textvariable')
    d.setvar(d.number, number, 'textvariable')
    d.setvar(d.checks['Fast graph'], 1)
    d.setvar(d.checks[ALL], 1)
    d.setvar(d.embed, 't-SNE', 'textvariable')
    for (lo, hi), v in (((5.0, 200.0), 12), ((0.1, 5.0), 0.7),
                        ((4.0, 20.0), 6), ((2.0, 40.0), 8)):
        d.setvar(d.spins[(lo, hi)], v, 'textvariable')
    d.markers.selection_clear(0, 'end')
    d.markers.selection_set(0, 1)
    (got,) = d.press_run()
    assert got == {
        'method': method, 'all_samples': True, 'k': 12, 'grid': 6,
        'n_meta': 8, 'embedding': 't-SNE',
        'resolution': 1.0 if method == 'flowsom' else 0.7,
        'channels': ['FL1-A', 'FL1-H'], 'reproducible': True,
        'fast_graph': method == 'leiden', 'downsample': mode,
        'custom_n': cap, 'embed_dims': 2}
    assert not d.dlg.winfo_exists()


def test_run_defaults_and_the_pre_ticked_markers(ed):
    d = Dialog(ed)
    assert [d.markers.get(i) for i in d.markers.curselection()] == DEFAULT
    (got,) = d.press_run()
    assert got == {
        'method': 'phenograph', 'all_samples': False, 'k': 30, 'grid': 10,
        'n_meta': 20, 'embedding': 'UMAP', 'resolution': 1.0,
        'channels': None, 'reproducible': True, 'fast_graph': False,
        'downsample': 'Default', 'custom_n': None, 'embed_dims': 2}


def test_phenograph_without_reproducible_sends_no_resolution(ed):
    d = Dialog(ed)
    d.setvar(d.checks[REPRO], 0)
    d.setvar(d.spins[(0.1, 5.0)], 'not a number', 'textvariable')
    (got,) = d.press_run()
    assert (got['reproducible'], got['resolution']) == (False, 1.0)


@pytest.mark.parametrize('number', ['', 'abc', '99'])
def test_a_bad_custom_number_keeps_the_dialog_open(ed, monkeypatch, number):
    errors = []
    monkeypatch.setattr(tkinter.messagebox, 'showerror',
                        lambda *a, **k: errors.append(a))
    d = Dialog(ed)
    d.setvar(d.events, 'Custom…', 'textvariable')
    d.setvar(d.number, number, 'textvariable')
    assert d.press_run() == []
    assert d.dlg.winfo_exists() and len(errors) == 1
