"""Headless tests for the plot display-cap parsing and density colour norm
(ViewGateEditorWindow._display_point_cap / _density_norm). Stub `self` so no
Tk display is needed."""
from __future__ import annotations

import types

import numpy as np

from openflo.gui import ViewGateEditorWindow as W


def _cap(value):
    stub = types.SimpleNamespace(
        max_points_var=types.SimpleNamespace(get=lambda: value))
    return W._display_point_cap(stub)


def test_cap_presets_and_default():
    assert _cap('60000') == 60_000
    assert _cap('20000') == 20_000
    assert _cap('250000') == 250_000


def test_cap_shorthand():
    assert _cap('250k') == 250_000
    assert _cap('1.5m') == 1_500_000
    assert _cap('100,000') == 100_000


def test_cap_uncapped_tokens():
    big = 1 << 62
    assert _cap('All') == big
    assert _cap('all') == big
    assert _cap('') == big
    assert _cap('0') == big


def test_cap_garbage_falls_back():
    assert _cap('banana') == 60_000


def test_cap_floor():
    # Tiny values are floored so a typo can't blank the plot.
    assert _cap('5') == 1000


def test_cap_missing_var_defaults():
    assert W._display_point_cap(types.SimpleNamespace()) == 60_000


def test_density_norm_spreads_range():
    z = np.array([0, 1, 2, 50, 1000], dtype=float)
    norm = W._density_norm(z)
    # PowerNorm gamma<1: low values map to a higher fraction than linear would.
    assert norm.gamma < 1.0
    assert norm.vmin == 0.0
    assert norm.vmax == 1000.0
    # A mid value sits well above its linear position (0.05) after gamma.
    assert norm(50.0) > 0.05


def test_density_norm_empty():
    norm = W._density_norm(np.array([]))
    assert norm.vmax > 0       # no div-by-zero / degenerate range
