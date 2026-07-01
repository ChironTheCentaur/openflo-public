"""Compensation-matrix display: theme-aware, legible cell colours.

Guards the Midnight-theme contrast fix — used cells (non-zero, incl. the
diagonal) carry the foreground colour; zero cells are muted — so the matrix
isn't near-black-on-dark. Tk-guarded.
"""
from __future__ import annotations

import importlib
import os

import pytest


def _root_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:                          # noqa: BLE001
        pytest.skip(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    return root


def test_comp_matrix_colours_track_theme():
    import numpy as np
    root = _root_or_skip()
    gui = importlib.import_module('openflo.gui')
    try:
        mid = gui.apply_theme(root, 'midnight')
        from openflo.ui_comp import CompensationEditorWindow
        win = CompensationEditorWindow(root, sample=None)
        win.withdraw()
        win.channels = ['CD3-A', 'CD4-A']
        win.matrix = np.array([[1.0, 0.0], [0.12, 1.0]])
        win._render_matrix()
        fg, muted = mid['fg'], mid['muted']

        def _fg(cell):
            # cget returns a Tk color object on some builds — normalise to str.
            return str(win.entries[cell][1].cget('foreground'))

        # diagonal 1.0 → used (fg); off-diagonal 0.0 → muted; 0.12 → used (fg)
        assert _fg((0, 0)) == fg
        assert _fg((0, 1)) == muted
        assert _fg((1, 0)) == fg
        # the used colour must differ from the muted one (legible contrast)
        assert fg != muted
    finally:
        gui.apply_theme(root, 'light')              # don't leak palette globally
        root.destroy()
