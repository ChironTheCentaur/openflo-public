"""Legend placement preference.

matplotlib's ``loc='best'`` tests candidate corners against every plotted
point, so it costs ~1.1 s on 8 samples x 50k events — on every replot. The
default 'cached' mode keeps that automatic placement but solves it once per
plot shape and reuses the answer, which measured 808 ms -> 435 ms per replot.
"""
import os
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

COLS = ('FSC-A', 'SSC-A', 'CD3', 'CD4')


def _editor_or_skip(n_samples=2):
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip('tkinter not available')
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:                       # noqa: BLE001
        pytest.skip(str(e))
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    rng = np.random.default_rng(0)
    for i in range(n_samples):
        nm = f's{i}'
        df = pd.DataFrame({c: rng.random(500) for c in COLS})
        ed._samples[nm] = SimpleNamespace(
            name=nm, path=rf'C:\e\{nm}.fcs', data=df,
            fluor_channels=list(COLS[2:]),
            channel_labels={c: c for c in COLS})
        ed._sample_order.append(nm)
        ed._sample_colors[nm] = '#1f77b4'
        ed._sample_trial[nm] = 'T'
        ed._sample_plot_enabled[nm] = True
        ed._sample_gates[nm] = {}
    ed._trial_order.append('T')
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    ed._populate_channel_combos()
    ed.x_combo.set('FSC-A')
    ed.y_combo.set('SSC-A')
    return root, ed


def test_default_is_cached():
    root, ed = _editor_or_skip()
    try:
        assert ed._legend_placement.get() == 'cached'
        assert ed._legend_mode() == 'cached'
    finally:
        root.destroy()


def test_fixed_mode_uses_a_corner_not_best():
    root, ed = _editor_or_skip()
    try:
        ed._legend_placement.set('fixed')
        ed.ax.plot([0, 1], [0, 1], label='a')
        leg = ed._apply_legend(fontsize=8)
        assert leg is not None
        # 'best' is loc code 0; a fixed corner is anything else.
        assert leg._loc != 0
        assert not ed._legend_pos_cache
    finally:
        root.destroy()


def test_cached_mode_solves_once_then_reuses():
    root, ed = _editor_or_skip()
    try:
        ed._legend_placement.set('cached')
        ed.ax.plot([0, 1], [0, 1], label='a')
        leg = ed._apply_legend(fontsize=8)
        assert leg._loc == 0, 'first draw should solve best'
        assert ed._legend_pending is not None
        ed.fig.canvas.draw()
        ed._capture_legend_pos()
        assert ed._legend_pending is None
        assert len(ed._legend_pos_cache) == 1, 'position not remembered'
        key, pos = next(iter(ed._legend_pos_cache.items()))
        assert isinstance(pos, tuple) and len(pos) == 2
        assert all(np.isfinite(pos))
        # Second call must reuse it, not re-solve.
        leg2 = ed._apply_legend(fontsize=8)
        assert leg2._loc != 0, 'second draw re-solved best instead of reusing'
    finally:
        root.destroy()


def test_cache_key_changes_with_axes_and_samples():
    """A remembered position must not leak across different plots."""
    root, ed = _editor_or_skip()
    try:
        k1 = ed._legend_cache_key()
        ed.y_combo.set('CD3')
        assert ed._legend_cache_key() != k1, 'y-channel must invalidate'
        ed.y_combo.set('SSC-A')
        assert ed._legend_cache_key() == k1
        ed._sample_plot_enabled['s1'] = False
        assert ed._legend_cache_key() != k1, 'sample set must invalidate'
    finally:
        root.destroy()


def test_best_mode_always_resolves_best():
    root, ed = _editor_or_skip()
    try:
        ed._legend_placement.set('best')
        ed.ax.plot([0, 1], [0, 1], label='a')
        for _ in range(2):
            assert ed._apply_legend(fontsize=8)._loc == 0
        assert not ed._legend_pos_cache, 'best mode must not populate a cache'
    finally:
        root.destroy()


def test_unknown_preference_value_falls_back_to_cached():
    root, ed = _editor_or_skip()
    try:
        ed._legend_placement.set('nonsense')
        assert ed._legend_mode() == 'cached'
    finally:
        root.destroy()
