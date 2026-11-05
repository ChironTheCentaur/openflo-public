"""Group (categorical) colour palettes: the data and the pure colour logic.

Tk-free and preference-free, so the library can colour groups without
loading the GUI: every function takes the palette name explicitly. The GUI
reads the user's choice and passes it in (theme.group_colors /
theme.categorical_colors).
"""
from __future__ import annotations

# ── Group (categorical) colours ─────────────────────────────────────────────
# Samples, conditions and groups are told apart by colour. matplotlib's tab10
# puts a red and a green side by side, the pair the commonest colour-vision
# deficiency (red-green, about 1 man in 12) confuses. Okabe and Ito's
# colour-universal palette (2002) stays distinct under every common
# deficiency; its black is swapped for a dark grey so it shows on the dark
# theme as well as on white (a lighter grey would sit close to its pink for
# a deuteranope). Chosen in View → Group colors, persisted as the
# 'group_palette' preference.
OKABE_ITO = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9',
             '#D55E00', '#F0E442', '#666666')
TAB10 = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
GROUP_PALETTES = (('okabe_ito', 'Color-blind safe (Okabe-Ito)'),
                  ('classic', 'Classic (tab10)'))
DEFAULT_GROUP_PALETTE = 'okabe_ito'


def palette_colors(name: str) -> tuple[str, ...]:
    """The cycle of group colours of palette ``name``, safest first.

    'classic' is tab10; any other name gives Okabe-Ito. Only eight colours
    stay distinct under colour-vision deficiency, so the Okabe-Ito cycle
    continues with tab10's colours that it lacks (brown, olive, cyan, then
    the rest) for a ninth group onwards rather than repeating a colour."""
    if name == 'classic':
        return TAB10
    extra = tuple(c for c in (TAB10[5], TAB10[8], TAB10[9], TAB10[3],
                              TAB10[4], TAB10[6]) if c not in OKABE_ITO)
    return OKABE_ITO + extra


def n_group_colors(n: int, name: str) -> list[str]:
    """``n`` distinct colours for ``n`` groups, in order, from palette ``name``.

    Up to the palette's own length the colours are its first ``n`` (so two
    groups get its two most distinct colours, not its first and last);
    beyond that tab20, then evenly spaced hues."""
    n = max(0, int(n))
    base = palette_colors(name)
    if n <= len(base):
        return list(base[:n])
    import matplotlib
    from matplotlib.colors import to_hex
    cmap = matplotlib.colormaps['tab20' if n <= 20 else 'gist_ncar']
    if n <= 20:
        return [to_hex(cmap(i)) for i in range(n)]
    return [to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
