# openflo — UI design notes

Design notes for the Tkinter desktop GUI. **Verified against the current source**
(2026-07-02, three code-reading passes over `gui.py` + the `editor_*.py` mixins +
`theme.py` + the `ui_*.py` dialogs). The earlier version of this file was derived from
session history, not code, and its central claim turned out to be stale — see below.

## How theming actually works (verified)

Theming is **not** confined to the main window. `apply_theme(root, mode)` (`gui.py:2090`)
works through two interpreter-global mechanisms, run once at startup (`gui.py:2351`) and
re-run on every theme switch (`editor_chrome._set_theme`, `editor_chrome.py:111`):

1. **`ttk.Style(root)`** — reconfigures every ttk widget *class* (Frame, Label, Button,
   Entry, Combobox field, Treeview, Notebook, Scrollbar, …). Because a ttk style DB is
   per-interpreter, this restyles every ttk widget in **all** windows (incl. open dialogs)
   live, with no per-window calls and no `winfo_children()` walk.
2. **Tk option database** (`gui.py:2196`) — seeds `*Toplevel.background`, `*Menu.*`,
   `*Listbox.*`, `*Text.*`, `*Canvas.*`, so every **raw-tk** widget created *afterward*
   opens already themed.

Plus: OS **title bars** via a class-level `<Map>` bind on Toplevel (`gui.py:1201` →
`editor_chrome._apply_titlebar_to`, Windows DWM dark caption); **menus** via `_theme_menu`
(`editor_tree.py`) and `_theme_menubar` (`editor_menu.py`); **matplotlib** figures via
`theme._theme_figure_dark` / `tool_window.figure_panel` / `editor_plot._apply_plot_theme`,
keyed on `_dialog_dark_on` (dark only under **Midnight** or the "dark figs" toggle — by
design; plain Dark chrome keeps white plots).

**Consequence:** the old claim — *"theme applied to main window but not to child windows /
menus / dropdowns / dialogs / figures"* — is **false for the current code**. All four
historically-buggy windows (frequency, voltage-opt, trajectory, comp-optimize) are themed
(frequency even has an alpha-0 anti-flash reveal). The ~6 historical "white in midnight"
reports were already fixed by the option-DB + figure-helper work.

## Genuine residual gaps (the real, small work)

1. **ttk.Combobox drop-down *list*** — only generic `*Listbox.*` is seeded; there is no
   `*TCombobox*Listbox.*` popdown targeting, so the dropdown list can still show
   native/white in Midnight on some Tk builds. App-wide.
2. **Populations context menu** (`editor_populations.py:95`) relies on the option DB but,
   unlike every other context menu, does not call `_theme_menu`.
3. **Voltage dialog** (`ui_voltage.py:47`) has no explicit `configure(bg)` / anti-flash
   guard → possible brief white flash on open.
4. **Live theme *switch*** does not recolor raw-tk widgets in *already-open* dialogs (the
   option DB is creation-time only). This is the one place a recursive re-theme walk is
   genuinely useful — for the switch, not for initial theming.
5. Minor hardcoded greys ignoring the palette (`ui_trajectory.py:61`, `ui_comp.py:334,359`,
   `editor_analysis` help labels).

## Look & space work done (2026-07-02, branch `feature/gui-theme-svttk`)

- **Adopted the `sleek` theme (2026-07-05): sv-ttk REMOVED.** sv-ttk was trialled first, but
  the app now ships a hand-styled, dependency-free look called *sleek* — the built-in `clam`
  theme styled to a refined near-black / off-white palette (midnight bg `#0f1013` / panel
  `#17181c` / cool accent `#6ea8fe`; light bg `#f6f7f9` / white panels / `#2f6fed`), flat
  surfaces, hairline borders tied to the palette, generous padding, muted flat headings/tabs,
  a cool accent on focus, PIL-drawn flat check/radio indicators. Single apply path
  `gui._apply_sleek`; `theme.THEMES` carries the palette; the `sv-ttk` dependency,
  `_apply_svttk`, `SVTTK_THEMES`, and `svttk_available` are all gone. The retained OpenFlo
  machinery (option DB, titlebar hook, `_theme_menu`/`_theme_menubar`, figure theming) runs on
  top so raw-tk widgets + figures stay coherent. (Branch name `feature/gui-theme-svttk` is now
  historical.)
- Closed the residual gaps: combobox popdown-list option-DB entries; `_retheme_open_tk`
  recolours raw-tk widgets in already-open dialogs on a *live* switch (greyscale-only, so
  gate swatches survive); populations context menu → `_theme_menu`; voltage dialog paints
  its bg up-front; hardcoded greys → palette `muted`.
- **The audit found a real bug the code-reading missed:** clam's `TProgressbar` trough was
  `#dcdad5` (near-white in dark) — now palette-tied in `_apply_sleek`.
- Space: `ui_frequency` summary got a scrollbar (was truncating), `ui_comp` matrix now fills
  the canvas width. The main window (already correctly grid-weighted), the optimize-dialog
  form (top-aligned fields — intentional), and the log console (already scrollbar'd,
  collapsed by default) were left as-is.
- Verified: `scripts/theme_audit.py` clean in midnight (regression test in
  `tests/test_theme.py`); full suite + selftest + ruff + pyright green; no functional change.

## Audit pass 2 — static sweep of every dialog/mixin (fable subagents)

A five-way parallel static audit (grep + read) of ~40 `ui_*` dialogs and `editor_*` mixins
for hardcoded colours that leak light in Midnight. The dominant pattern was `ttk.Label(...,
foreground='grey'|'#555'|'#666')` — an explicit foreground that overrides the themed default
and either sits off-palette or (dark greys) goes invisible on the dark panel. Fixes:

- **`Muted.TLabel` token** (defined once in `apply_theme`, both engines): all ~25 hint/
  status/caption labels now use `style='Muted.TLabel'` instead of a hand-set grey, so they
  track the palette and recolour on a *live* switch. (Matches "tokens once, not per-screen".)
- **`_theme_figure_dark` blind spot** (systemic): it recolours spines/ticks/labels but NOT
  free `ax.text` / `Line2D` artists, so significance brackets, empty-state text and FlowSOM
  star-tree spokes/labels stayed black/grey and vanished on a dark canvas. Added
  `theme.plot_ink(widget)` — returns the midnight-vs-light plot foreground by the same
  `_dialog_dark_on` decision — and routed those annotations through it (`ui_expression`,
  `ui_frequency`, `ui_flowsom_tree`, `ui_diff`, `editor_plot` empty-state).
- **Treeview flag rows**: `ui_compare` `bad`/`warn` and `workspace` `drop_target` hardcoded a
  light pink/yellow row bg (unreadable light-on-light in Midnight). Added
  `theme.flag_tints()` (deep tints + light ink on dark, light tints + dark ink on light).
- Left by design: `editor_export`/`savefig` `'White'` (publication export chrome),
  `editor_plot` histogram/`_render_into` placeholders (`#888`/`grey` — theme-agnostic on an
  un-themed early-return canvas), `editor_autoclean` `#b00` (semantic error red), data-viz
  colours (series/colormaps/gate overlays).
- Verified: audit clean in midnight, `test_theme.py` + selftest 7/7 + dialog test subset +
  ruff green; no functional change.

## Verify (the check that was missed before)

`scripts/theme_audit.py` builds every window/dialog under **Midnight** on a withdrawn root,
walks each widget, resolves its effective background (tk `cget` / `ttk.Style.lookup` /
matplotlib facecolor) and **fails on any white/near-white surface** — kept as a regression
test in `tests/test_theme.py` so new windows can't regress. `--screens` also saves a PNG of
each window for eyeballing. Always confirm the audit is empty AND skim the screenshots after
any chrome change — do not sign off on the main window alone.

## Web fork

openflo also has a web fork (branch `web`, checked out in a separate worktree); if that's the "web
gui" in a complaint, run `/design-pass` on it separately — it is not covered here.
