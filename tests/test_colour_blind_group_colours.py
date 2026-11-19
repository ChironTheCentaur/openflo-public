"""Colour-blind-safe group colours (item 14): samples and groups default to
the Okabe-Ito palette, which stays apart for a red-green deficiency; a
palette switch keeps colours picked by hand; View > Group colors.

All data are generated here (synthetic sample names, generic FL channels).
"""
from __future__ import annotations

import numpy as np

import tests._four_sample_editor as shared
from openflo import theme

ed = shared.ed


def _srgb_to_lin(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_lab(rgb):
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = rgb @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz),
                 (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


# Deuteranopia (the commonest red-green deficiency), Machado et al. 2009,
# severity 1.0, applied to linear RGB.
_DEUTAN = np.array([[0.367322, 0.860646, -0.227968],
                    [0.280085, 0.672501, 0.047413],
                    [-0.011820, 0.042940, 0.968881]])


def _min_deutan_distance(colors):
    from matplotlib.colors import to_rgb
    lin = _srgb_to_lin([to_rgb(c) for c in colors])
    lab = _lin_to_lab(np.clip(lin @ _DEUTAN.T, 0, 1))
    d = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1)
    return d[np.triu_indices(len(colors), 1)].min()


def test_default_colours_stay_apart_for_a_red_green_deficiency():
    # Okabe-Ito's eight stay well apart as a deuteranope sees them; tab10's
    # ten include pairs that close up, its green and red among them.
    # (Delta E, CIELAB.)
    assert _min_deutan_distance(theme.OKABE_ITO) > 12
    assert _min_deutan_distance(theme.TAB10) < 8
    assert _min_deutan_distance(['#2ca02c', '#d62728']) < 8


def test_group_colours_default_to_okabe_ito(monkeypatch):
    monkeypatch.setattr(theme, 'read_prefs', dict)
    assert theme.group_palette_name() == 'okabe_ito'
    assert theme.categorical_colors(3) == list(theme.OKABE_ITO[:3])
    monkeypatch.setattr(theme, 'read_prefs',
                        lambda: {'group_palette': 'classic'})
    assert theme.categorical_colors(3) == list(theme.TAB10[:3])


def test_many_groups_still_get_distinct_colours():
    for n in (8, 9, 14, 15, 20, 25):
        cols = theme.categorical_colors(n, 'okabe_ito')
        assert len(cols) == n and len(set(cols)) == n


def test_pipeline_auto_palette_is_okabe_ito_in_order():
    from matplotlib.colors import to_hex

    from openflo.pipeline import categorical_cmap
    cm = categorical_cmap('auto', 3)
    # With vmin=0, vmax=n-1 group i gets colour i (not first, middle, last).
    got = [to_hex(cm(i / 2)) for i in range(3)]
    assert got == [c.lower() for c in theme.OKABE_ITO[:3]]
    tab = categorical_cmap('tab10', 2)
    assert [to_hex(tab(i)) for i in (0.0, 1.0)] == list(theme.TAB10[:2])
    # Any other name is still a matplotlib map.
    assert categorical_cmap('viridis', 4).name == 'viridis'


def test_samples_take_okabe_ito_and_a_switch_keeps_hand_picked(ed, monkeypatch):
    prefs = {}
    monkeypatch.setattr(theme, 'read_prefs', lambda: dict(prefs))
    import openflo.prefs as prefs_mod
    monkeypatch.setattr(prefs_mod, 'write_pref',
                        lambda k, v: prefs.__setitem__(k, v))
    saved = dict(ed._sample_colors)
    try:
        ed._sample_colors.clear()
        got = [ed._color_for(n) for n in ed._sample_order[:3]]
        assert got == list(theme.OKABE_ITO[:3])
        ed._sample_colors['synthetic_b2'] = '#123456'      # picked by hand
        ed._group_palette_var.set('classic')
        ed._set_group_palette()
        assert prefs['group_palette'] == 'classic'
        assert [ed._sample_colors[n] for n in ed._sample_order[:3]] == \
            list(theme.TAB10[:3])
        assert ed._sample_colors['synthetic_b2'] == '#123456'
        ed._group_palette_var.set('okabe_ito')
        ed._set_group_palette()
        assert [ed._sample_colors[n] for n in ed._sample_order[:3]] == \
            list(theme.OKABE_ITO[:3])
    finally:
        ed._sample_colors.clear()
        ed._sample_colors.update(saved)
        ed._group_palette_var.set('okabe_ito')


def test_view_menu_offers_group_colours(ed):
    btn = next(b for b in ed._menubar_buttons if b.cget('text') == 'View')
    viewm = ed.nametowidget(btn.cget('menu'))
    labels = [viewm.entrycget(j, 'label')
              for j in range(viewm.index('end') + 1)
              if viewm.type(j) != 'separator']
    assert 'Group colors' in labels and 'Density colors' in labels
