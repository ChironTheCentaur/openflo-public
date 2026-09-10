"""Theme audit — prove no white/near-white chrome survives in a dark theme.

The recurring "white background in midnight theme" class of bug was historically
re-reported against many surfaces. This tool builds the real editor window (and,
best-effort, the secondary windows) under a chosen theme on a *withdrawn* root,
walks every widget, resolves its effective background colour (tk ``cget`` /
``ttk.Style.lookup`` / combobox popdown / matplotlib facecolor) and flags anything
whose luminance is near-white. ``tests/test_theme.py`` calls :func:`audit_offenders`
as a regression guard so new windows can't reintroduce the bug.

Usage::

    python scripts/theme_audit.py                 # audit midnight, print offenders
    python scripts/theme_audit.py --mode dark
    python scripts/theme_audit.py --screens        # also save a PNG per window

Screenshots require a real display (windows are mapped briefly); the plain audit
runs withdrawn and needs only a Tk that can initialise.
"""
from __future__ import annotations

import argparse
import os
import sys

# Near-white cutoff on 0..1 relative luminance. Midnight chrome sits ~0.02; a
# genuine white/#fafafa surface is ~0.95+. 0.80 leaves a wide safety margin.
WHITE_LUM = 0.80


def _lum(widget, color):
    """Relative luminance 0..1 of a Tk colour spec (name or #rrggbb), or None if
    it can't be resolved (empty string / inherited / bad spec)."""
    if not color:
        return None
    try:
        r, g, b = widget.winfo_rgb(color)          # each 0..65535
    except Exception:
        return None
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 65535.0


def _effective_bg(widget, style):
    """The background colour a widget actually paints with. Raw-tk widgets expose
    ``-background``; ttk widgets don't, so fall back to the style DB for their
    style name (or widget class)."""
    try:
        return widget.cget('background')            # tk widgets (Menu, Canvas, …)
    except Exception:
        pass
    try:
        st = widget.cget('style') or widget.winfo_class()
        return (style.lookup(st, 'background')
                or style.lookup(st, 'fieldbackground') or None)
    except Exception:
        return None


def _walk(widget, style, offenders, seen):
    key = str(widget)
    if key in seen:
        return
    seen.add(key)
    bg = _effective_bg(widget, style)
    lum = _lum(widget, bg)
    if lum is not None and lum >= WHITE_LUM:
        offenders.append((key, widget.winfo_class(), str(bg), round(lum, 3)))
    try:
        children = widget.winfo_children()
    except Exception:
        children = []
    for c in children:
        _walk(c, style, offenders, seen)


def _check_combobox_popdowns(root, style, offenders):
    """The ttk.Combobox drop-down list is a separate popdown toplevel that the
    normal child walk never reaches. Force each one to exist and check its
    listbox background."""
    from tkinter import ttk

    def visit(w):
        if isinstance(w, ttk.Combobox):
            try:
                pop = w.tk.eval(f'ttk::combobox::PopdownWindow {w}')
                lb = f'{pop}.f.l'
                bg = w.tk.call(lb, 'cget', '-background')
                lum = _lum(w, bg)
                if lum is not None and lum >= WHITE_LUM:
                    offenders.append((lb, 'ComboboxPopdown', str(bg),
                                      round(lum, 3)))
            except Exception:
                pass
        for c in w.winfo_children():
            visit(c)

    visit(root)


def _check_figures(objs, offenders):
    """matplotlib facecolor for any window object exposing a `.fig`/`.figure`."""
    for obj in objs:
        for attr in ('fig', 'figure', '_fig'):
            fig = getattr(obj, attr, None)
            if fig is None or not hasattr(fig, 'get_facecolor'):
                continue
            try:
                import matplotlib.colors as mcolors
                r, g, b, _a = mcolors.to_rgba(fig.get_facecolor())
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                if lum >= WHITE_LUM:
                    offenders.append((f'{type(obj).__name__}.{attr}', 'Figure',
                                      str(fig.get_facecolor()), round(lum, 3)))
            except Exception:
                pass


def _load_fake(ed, name='S1', trial='Trial1'):
    """A lightweight fake sample so the tree/plot/combos populate (mirrors the
    gui_smoke test helper)."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    df = pd.DataFrame({'CD3': np.linspace(0, 1, 200), 'CD4': np.linspace(1, 0, 200)})
    s = SimpleNamespace(name=name, path=rf'C:\exp\{name}.fcs', data=df,
                        fluor_channels=['CD3', 'CD4'],
                        channel_labels={'CD3': 'CD3', 'CD4': 'CD4'})
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = trial
    if trial not in ed._trial_order:
        ed._trial_order.append(trial)
    ed._sample_gates[name] = {'g1': {'kind': 'threshold', 'channel': 'CD3',
                                     'value': 0.0, 'parent_id': None,
                                     'color': '#111', 'enabled': True, 'id': 'g1'}}
    ed._sample_gate_order[name] = ['g1']
    ed._sample_gate_seq[name] = 1
    ed._sample_plot_enabled[name] = True


def build_editor(mode='midnight'):
    """Construct a hidden editor with `mode` applied and one fake sample loaded.
    Returns (root, editor, gui_module) or raises if Tk can't initialise."""
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import importlib
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    gui.apply_theme(root, mode)                    # seed option DB before widgets
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    try:
        ed._theme_var.set(mode)
        ed._set_theme()                            # exercise the live-switch path
    except Exception:
        pass
    _load_fake(ed)
    try:
        ed._set_active_sample('S1')
        ed._refresh_gate_list()
    except Exception:
        pass
    root.update_idletasks()
    return root, ed, gui


def audit_offenders(mode='midnight'):
    """Build the editor under `mode` and return the list of near-white surfaces
    (path, class, colour, luminance). Empty list == clean."""
    from tkinter import ttk
    root, ed, _gui = build_editor(mode)
    try:
        style = ttk.Style(root)
        offenders, seen = [], set()
        _walk(root, style, offenders, seen)
        _check_combobox_popdowns(root, style, offenders)
        _check_figures([ed], offenders)
        # The root is permanently withdrawn (the editor Toplevel is the UI), so
        # its own default bg is never visible — don't flag it.
        return [o for o in offenders if o[1] != 'Tk']
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _screenshots(mode, outdir):
    """Map the editor (and best-effort sub-windows) and save a PNG of each.
    Requires a real display + Pillow's ImageGrab (Windows/mac)."""
    from PIL import ImageGrab
    os.makedirs(outdir, exist_ok=True)
    root, ed, _gui = build_editor(mode)
    shots = []

    def grab(win, label):
        try:
            win.deiconify()
            win.update()
            win.update_idletasks()
            x, y = win.winfo_rootx(), win.winfo_rooty()
            w, h = win.winfo_width(), win.winfo_height()
            if w < 5 or h < 5:
                return
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            path = os.path.join(outdir, f'{mode}_{label}.png')
            img.save(path)
            shots.append(path)
            print(f'  saved {path}')
        except Exception as exc:
            print(f'  [skip {label}] {exc}')

    try:
        ed.geometry('1200x800')
        grab(ed, 'editor')
        # Best-effort: open a few of the dialogs the white-bug was filed against
        # so the screenshots cover secondary surfaces too. Each is guarded.
        for label, opener in (
            ('voltage', lambda: __import__('openflo.ui_voltage',
                fromlist=['VoltageDialog']).VoltageDialog(ed)),
            ('preferences', lambda: __import__('openflo.ui_preferences',
                fromlist=['PreferencesDialog']).PreferencesDialog(ed)),
        ):
            try:
                win = opener()
                win.update_idletasks()
                grab(win, label)
            except Exception as exc:
                print(f'  [skip {label}] {exc}')
        # Any other already-open Toplevels (dialogs the editor spawned).
        for child in root.winfo_children():
            if child is ed:
                continue
            if child.winfo_class() == 'Toplevel':
                grab(child, f'top_{child.winfo_name()}')
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    return shots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', default='midnight',
                    help='theme to audit (default: midnight)')
    ap.add_argument('--screens', action='store_true',
                    help='also save a PNG of each window (needs a display)')
    ap.add_argument('--outdir', default=None, help='screenshot output dir')
    args = ap.parse_args(argv)

    try:
        offenders = audit_offenders(args.mode)
    except Exception as exc:                        # noqa: BLE001
        print(f'audit could not run: {exc}')
        return 2
    if offenders:
        print(f'FAIL — {len(offenders)} near-white surface(s) in {args.mode!r}:')
        for path, cls, color, lum in offenders:
            print(f'  [{cls}] {path}  bg={color}  lum={lum}')
    else:
        print(f'OK — no near-white chrome in {args.mode!r} theme.')

    if args.screens:
        outdir = args.outdir or os.path.join('scratchpad', 'theme_shots')
        print(f'screenshots → {outdir}')
        _screenshots(args.mode, outdir)

    return 1 if offenders else 0


if __name__ == '__main__':
    sys.exit(main())
