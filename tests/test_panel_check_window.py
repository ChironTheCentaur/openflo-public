"""Tools > Panel check window (ui_panel_check.PanelCheckWindow): choose a
detector set or load spectra, build a panel, read the predicted similarity,
complexity index and flagged pairs, and export them.

File pickers are answered by the test. The numbers shown are checked against
numpy on synthetic signatures (invented dyes, an invented catalogue).
"""
from __future__ import annotations

import csv
import math
import os
import tkinter as tk

import numpy as np
import pytest

from openflo import fluor_catalog
from openflo import panel_check as pc
from tests.conftest import gui_unavailable

S_CSV = np.array([[1.0, 0.5, 0.1, 0.0],
                  [0.9, 0.55, 0.1, 0.0],
                  [0.0, 0.1, 0.4, 1.0]])
FL_CSV = ['DyeA', 'DyeB', 'DyeC']
DET_CSV = ['Det1', 'Det2', 'Det3', 'Det4']


def _cosine(S):
    U = S / np.linalg.norm(S, axis=1, keepdims=True)
    return U @ U.T


@pytest.fixture
def root():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        r = tk.Tk()
    except Exception as e:                          # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    r.withdraw()
    try:
        yield r
    finally:
        r.destroy()


@pytest.fixture
def dyes(monkeypatch):
    cat = {'DyeA': (((488, 1.0),), 40, 520, 40, ()),
           'DyeB': (((488, 1.0),), 40, 525, 40, ()),
           'DyeC': (((640, 1.0),), 40, 670, 35, ()),
           'DyeD': (((405, 1.0),), 40, 450, 40, ())}
    monkeypatch.setattr(fluor_catalog, 'FLUOR_SPECTRA', cat)
    return cat


def _csv(tmp_path, text, name='spectra.csv'):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return str(p)


def _read(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.reader(f))


def test_model_panel_scores_and_flags(root, dyes):
    from openflo.ui_panel_check import MODEL_NOTE, PanelCheckWindow
    win = PanelCheckWindow(root)
    root.update()
    assert win.winfo_width() <= 1024 and win.winfo_height() <= 768
    assert win.source_var.get() == pc.DEFAULT_SET
    assert win.thr_var.get() == '0.98'
    assert list(win.avail.get(0, 'end')) == ['DyeA', 'DyeB', 'DyeC', 'DyeD']
    # Search, then add the one match through the list.
    win.search_var.set('dyeb')
    assert list(win.avail.get(0, 'end')) == ['DyeB']
    win.avail.selection_set(0)
    win._add()
    win.search_var.set('')
    win.add_fluors(['DyeA', 'DyeC', 'DyeZ'])
    assert list(win.chosen.get(0, 'end')) == ['DyeB', 'DyeA', 'DyeC']
    dets = pc.DETECTOR_SETS[pc.DEFAULT_SET]['detectors']
    S = np.vstack([pc.model_signature(f, dets)
                   for f in ('DyeB', 'DyeA', 'DyeC')])
    r = win.report
    np.testing.assert_allclose(r['similarity'], _cosine(S), atol=1e-12)
    sv = np.linalg.svd(S, compute_uv=False)
    assert r['condition_number'] == pytest.approx(sv.max() / sv.min())
    # Two dyes 5 nm apart on the same laser: flagged.
    assert [(d['fluor_a'], d['fluor_b']) for d in r['similar_pairs']] == \
        [('DyeB', 'DyeA')]
    text = win.summary_text()
    assert text.startswith(MODEL_NOTE)
    assert f"{r['condition_number']:.2f}" in win.header_var.get()
    assert '1 similar pair(s)' in win.header_var.get()
    assert 'DyeB ~ DyeA' in text
    assert 'without it' in text                   # a swap that helps
    # Spectral: one heatmap (+ its colour bar), no spillover export.
    assert len(win.fig.axes) == 2
    assert win.spill_btn.instate(['disabled'])
    # Remove one from the panel.
    win.chosen.selection_set(0)
    win._remove()
    assert win.report['fluors'] == ['DyeA', 'DyeC']
    assert win.report['similar_pairs'] == []


def test_conventional_set_shows_spillover_and_threshold(root, dyes):
    from openflo.ui_panel_check import PanelCheckWindow
    win = PanelCheckWindow(root)
    name = '3-laser conventional (405/488/640 nm)'
    win.source_var.set(name)
    win._source_changed()
    assert win.thr_var.get() == '0.9'
    win.add_fluors(['DyeA', 'DyeC', 'DyeD'])
    root.update()
    assert len(win.fig.axes) == 4                 # two maps, two bars
    assert win.spill_btn.instate(['!disabled'])
    dets = pc.DETECTOR_SETS[name]['detectors']
    S = pc.signature_matrix(['DyeA', 'DyeC', 'DyeD'], dets)
    prim = S.argmax(axis=1)
    want = S[:, prim] / S.max(axis=1, keepdims=True)
    np.testing.assert_allclose(win.report['spillover'], want, atol=1e-12)
    assert win.report['primary_detector'] == [dets[k][0] for k in prim]
    assert 'Primary detectors:' in win.summary_text()
    # A low threshold flags more; a nonsense one falls back to the default.
    win.thr_var.set('0.0001')
    win._refresh()
    C = _cosine(S)
    assert len(win.report['similar_pairs']) == int(
        (C[np.triu_indices(3, 1)] >= 0.0001).sum())
    win.thr_var.set('abc')
    win._refresh()
    assert win.thr_var.get() == '0.9' and win.report['threshold'] == 0.9


def test_load_csv_check_and_export(root, monkeypatch, tmp_path):
    from openflo import tool_window
    from openflo import ui_panel_check as mod
    rows = pc.spectra_csv_rows(S_CSV, FL_CSV, DET_CSV)
    src = _csv(tmp_path, '\n'.join(','.join(r) for r in rows) + '\n')
    monkeypatch.setattr(mod.filedialog, 'askopenfilename',
                        lambda *a, **k: src)
    win = mod.PanelCheckWindow(root)
    win._load_csv()
    root.update()
    assert win.source_var.get() == 'Spectra CSV: spectra.csv'
    assert win.report['fluors'] == FL_CSV
    # Four detectors: scored as conventional.
    assert win.thr_var.get() == '0.9'
    C = _cosine(S_CSV)
    np.testing.assert_allclose(win.report['similarity'], C, atol=1e-6)
    assert [(d['fluor_a'], d['fluor_b'])
            for d in win.report['similar_pairs']] == [('DyeA', 'DyeB')]
    assert not win.summary_text().startswith('Approximate')
    # Without the middle fluor the rows left are scored, not the first two.
    win.chosen.selection_set(1)
    win._remove()
    assert win.report['fluors'] == ['DyeA', 'DyeC']
    assert win.report['similarity'][0, 1] == pytest.approx(C[0, 2],
                                                           abs=1e-6)
    win.add_fluors(['DyeB'])
    assert win.report['fluors'] == ['DyeA', 'DyeC', 'DyeB']
    win._panel = list(FL_CSV)
    win._fill_lists()
    win._refresh()
    # Export the similarity matrix and the signatures.
    out = {'path': ''}
    monkeypatch.setattr(tool_window.filedialog, 'asksaveasfilename',
                        lambda *a, **k: out['path'])
    out['path'] = str(tmp_path / 'sim.csv')
    assert win._export_matrix('similarity') == out['path']
    got = _read(out['path'])
    assert got[0] == ['', *FL_CSV]
    np.testing.assert_allclose(
        np.array([[float(v) for v in r[1:]] for r in got[1:]]), C,
        atol=1e-5)
    out['path'] = str(tmp_path / 'spill.csv')
    win._export_matrix('spillover')
    got = _read(out['path'])
    assert got[2][0] == 'DyeB' and float(got[2][1]) == pytest.approx(1.0)
    out['path'] = str(tmp_path / 'sig.csv')
    win._save_signatures()
    S2, fl2, d2 = pc.load_spectra_csv(out['path'])
    assert fl2 == FL_CSV and d2 == DET_CSV
    np.testing.assert_allclose(S2, S_CSV / S_CSV.max(axis=1, keepdims=True),
                               atol=1e-6)
    out['path'] = str(tmp_path / 'fig.png')
    assert win._export_figure() == out['path']
    assert os.path.getsize(out['path']) > 0
    # Cancelled: nothing written.
    out['path'] = ''
    assert win._export_matrix('similarity') is None
    # Back to a model set: the CSV dyes are not in the catalogue.
    win.source_var.set(pc.DEFAULT_SET)
    win._source_changed()
    assert win.report is None and win._panel == []


def test_bad_csv_is_reported_and_changes_nothing(root, monkeypatch,
                                                 tmp_path):
    from tkinter import messagebox

    from openflo import ui_panel_check as mod
    src = _csv(tmp_path, 'Fluor,Det1,Det2\nDyeA,1,-2\n')
    monkeypatch.setattr(mod.filedialog, 'askopenfilename',
                        lambda *a, **k: src)
    errors = []
    monkeypatch.setattr(messagebox, 'showerror',
                        lambda title, msg, **k: errors.append((title, msg)))
    win = mod.PanelCheckWindow(root)
    win._load_csv()
    assert errors and errors[0][0] == 'Panel check'
    assert win.source_var.get() == pc.DEFAULT_SET and win._loaded is None
    # A cancelled picker does nothing either.
    monkeypatch.setattr(mod.filedialog, 'askopenfilename',
                        lambda *a, **k: '')
    win._load_csv()
    assert len(errors) == 1 and win._loaded is None


def test_last_unmix_spectra(root):
    from openflo.ui_panel_check import PanelCheckWindow
    win = PanelCheckWindow(root)
    assert win.unmix_btn.instate(['disabled'])
    win.destroy()
    root._last_unmix_spectra = (S_CSV, FL_CSV)    # type: ignore[attr-defined]
    root._last_unmix_detectors = DET_CSV          # type: ignore[attr-defined]
    win = PanelCheckWindow(root)
    assert win.unmix_btn.instate(['!disabled'])
    win.unmix_btn.invoke()
    assert win.source_var.get() == 'Last unmix (this session)'
    assert win.report['detectors'] == DET_CSV
    np.testing.assert_allclose(win.report['similarity'], _cosine(S_CSV),
                               atol=1e-12)
    # Detector names that do not fit the spectra: numbered instead.
    root._last_unmix_detectors = ['Det1']         # type: ignore[attr-defined]
    win._use_last_unmix()
    assert win.report['detectors'] == ['D1', 'D2', 'D3', 'D4']
    assert math.isfinite(win.report['condition_number'])


# ── In the editor ──────────────────────────────────────────────────────────

import tests.test_windows_nothing_hidden as sweep  # noqa: E402

virtual_screen = sweep.virtual_screen
no_dialogs = sweep.no_dialogs
engines_known = sweep.engines_known


def test_tools_menu_opens_the_panel_check(virtual_screen, no_dialogs,
                                          engines_known):
    from openflo.ui_panel_check import PanelCheckWindow
    virtual_screen(1920, 1080)
    r = sweep._root(1.0)
    try:
        ed = sweep._editor(r)
        items = [(m, i) for m, i, label in sweep._menu_items(ed)
                 if label == 'Panel check…']
        assert len(items) == 1
        m, i = items[0]
        m.invoke(i)
        r.update()
        wins = [w for w in sweep._toplevels(r).values()
                if isinstance(w, PanelCheckWindow)]
        assert len(wins) == 1
        assert sweep.problems(wins[0]) == []
    finally:
        r.destroy()


def test_use_last_unmix_after_a_real_unmix():
    """Unmix noise-free synthetic controls in the editor, then check the
    measured spectra: the panel check sees them on the unmix's detectors,
    with the similarity of the true spectra."""
    import pandas as pd

    from openflo.ui_panel_check import PanelCheckWindow
    from tests import test_spectral_unmix_gui as un
    r, ed = un._editor_or_skip()
    try:
        arrs, _A = un._arrays()
        for nm in ('F1', 'F2', 'F3', 'unstained', 'mixed'):
            un._register(ed, nm, pd.DataFrame(arrs[nm],
                                              columns=pd.Index(un.DETS)))
        ed._apply_spectral_unmix({'F1': 'F1', 'F2': 'F2', 'F3': 'F3'},
                                 'unstained', list(un.DETS), False)
        win = PanelCheckWindow(ed)
        win.unmix_btn.invoke()
        rep = win.report
        assert rep['detectors'] == un.DETS
        assert rep['fluors'][:3] == un.FLUORS
        np.testing.assert_allclose(rep['similarity'][:3, :3],
                                   _cosine(un.SPECTRA), atol=1e-6)
    finally:
        r.destroy()
