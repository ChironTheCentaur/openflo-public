"""The plate view and the spectral QC SSM heatmap draw empty / NaN cells in
grey. They set that colour with the supported ``Colormap.with_extremes``,
which returns a new colormap, not the in-place ``set_bad``, which matplotlib
deprecates. These checks pin the grey on the drawn image, and fail if a
window goes back to an in-place setter or changes the shared registry entry.

Generated data only, synthetic sample names and generic channels.
"""
from __future__ import annotations

import pathlib
import re
import sys

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pytest

import tests._sample_editor as shared
from tests.test_spectral_qc_window import _root_or_skip

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'
PLATE_GREY = mcolors.to_rgba('#d9d9d9')
SSM_GREY = mcolors.to_rgba('lightgrey')


@pytest.fixture
def no_inplace_extremes(monkeypatch):
    """Make the in-place setters fail loudly when OpenFlo code calls them,
    as a future matplotlib will (matplotlib's own with_extremes still uses
    them on its fresh copy, so only OpenFlo callers are refused), and
    remember the shared viridis bad colour to check it is untouched."""
    for name in ('set_bad', 'set_under', 'set_over', 'set_extremes'):
        orig = getattr(mcolors.Colormap, name)

        def guarded(self, *a, _orig=orig, _name=name, **k):
            caller = sys._getframe(1).f_globals.get('__name__', '')
            if caller.split('.')[0] == 'openflo':
                raise AssertionError(f'{caller} called Colormap.{_name}')
            return _orig(self, *a, **k)
        monkeypatch.setattr(mcolors.Colormap, name, guarded)
    before = matplotlib.colormaps['viridis'].get_bad().tolist()
    yield
    assert matplotlib.colormaps['viridis'].get_bad().tolist() == before


def _bad(image):
    return tuple(float(x) for x in image.cmap.get_bad())


def test_plate_view_empty_wells_are_grey(editor, no_inplace_extremes):
    from openflo.ui_plate import PlateWindow
    root, ed = editor
    shared._tag(ed, alpha_1={'WELL ID': 'B03'})
    win = PlateWindow(ed)
    root.update()
    win.draw()
    im = win.fig.axes[0].get_images()[0]
    assert np.isnan(win.matrix()).sum() > 0          # some wells are empty
    assert _bad(im) == pytest.approx(PLATE_GREY)
    # the empty wells really are masked, so they take the bad colour
    rgba = im.to_rgba(im.get_array())
    assert tuple(rgba[0, 0]) == pytest.approx(PLATE_GREY)
    win.destroy()


def test_ssm_heatmap_nan_cells_are_grey(no_inplace_extremes):
    from openflo.ui_spectral_qc import SpectralQCWindow
    root = _root_or_skip()
    try:
        ssm = np.array([[np.nan, 0.9, 0.1],
                        [0.2, np.nan, 0.3],
                        [0.4, 0.5, np.nan]])
        qc = {'fluors': ['A', 'B', 'C'], 'similarity': np.eye(3), 'ssm': ssm,
              'condition_number': 1.0, 'similar_pairs': [],
              'worst_spread': [], 'degenerate_fluors': []}
        win = SpectralQCWindow(root, qc)
        im = win._fig.axes[2].get_images()[0]   # similarity, colorbar, SSM
        assert _bad(im) == pytest.approx(SSM_GREY)
        rgba = im.to_rgba(im.get_array())
        assert tuple(rgba[1, 1]) == pytest.approx(SSM_GREY)
        assert tuple(rgba[1, 0]) != pytest.approx(SSM_GREY)
    finally:
        root.destroy()


def test_no_inplace_colormap_extreme_setters_in_source():
    """A static net for windows no GUI test draws."""
    pat = re.compile(r'\.set_(bad|under|over|extremes)\(')
    hits = [f"{p.name}:{i}" for p in sorted(SRC.rglob('*.py'))
            for i, line in enumerate(p.read_text(encoding='utf-8')
                                     .splitlines(), 1)
            if pat.search(line)]
    assert hits == []
