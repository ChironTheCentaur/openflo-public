"""Headless tests for the plot display-cap parsing and density colour norm
(ViewGateEditorWindow._display_point_cap / _density_norm). Stub `self` so no
Tk display is needed."""
from __future__ import annotations

import types

import numpy as np
import pytest

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


@pytest.mark.parametrize('text', ['banana', 'nan', 'inf', '1e400', '-5',
                                  '0', '999', '0.5k', '1000.5'])
def test_cap_refuses_text_that_is_not_a_cap(text):
    """MAIN-2/3: words, nan, inf (1e400 overflowed int() into a traceback),
    negatives, 0 and values under 1,000 are not a cap. parse_point_cap says
    so (None) rather than turning them into 60,000, 1,000 or no cap."""
    from openflo.editor_plot import parse_point_cap
    assert parse_point_cap(text) is None


def test_cap_reader_keeps_the_last_valid_value_for_text_being_typed():
    """Half-typed text that is not a cap is never drawn with: the plot keeps
    the cap the box last held validly (not a fixed 60,000)."""
    stub = types.SimpleNamespace(
        max_points_var=types.SimpleNamespace(get=lambda: 'abc'),
        _mp_last_good='20000')
    assert W._display_point_cap(stub) == 20_000
    stub._mp_last_good = 'All'
    assert W._display_point_cap(stub) == 1 << 62
    stub.max_points_var = types.SimpleNamespace(get=lambda: '1e400')
    assert W._display_point_cap(stub) == 1 << 62


def test_cap_shorthand_is_rounded_to_whole_events():
    from openflo.editor_plot import parse_point_cap
    assert parse_point_cap('1.2k') == 1200
    assert parse_point_cap('1.005k') == 1005     # 1.005*1000 = 1004.99...
    assert parse_point_cap('1,000') == 1000
    assert parse_point_cap('2e5') == 200_000


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
