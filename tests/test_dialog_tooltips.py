"""Every dialog control carries hover help, and the workspace opens unclipped.

Two user-reported faults, pinned so they cannot come back quietly.

**Tooltips.** The editor had hover tips from early on; the thirty ``ui_*``
dialogs never did. An audit found ZERO tooltip calls against ~105 meaningful
interactive widgets, so Preferences, Compensation, Frequencies and the rest
offered no hover help at all. That is backwards: the main window's controls are
seen constantly and learned by repetition, while a dialog's options are met
once a month and forgotten in between. The gap was invisible because the
machinery worked perfectly — it was simply never called.

**Clipping.** Revealing the docked Pipeline Workspace opened it at a guessed
320 px floor. The toolbar needed 336, so ``Group v`` lost its arrow, and the
plot column was squeezed to 512 px against control rows needing 754, clipping
the axis combos, ``Auto-gate`` and ``Show cleaned-out events``.

Both tests assert a PROPERTY rather than a number: "nothing an interactive
widget needs is missing" and "nothing is narrower than it requires". They stay
true as controls are added, which a pinned pixel count would not.
"""
import importlib
import inspect
import os

import pytest

from tests.conftest import gui_unavailable

# Widget classes a user can act on, and therefore might hover for help.
INTERACTIVE = {'TButton', 'TMenubutton', 'TEntry', 'TCombobox', 'TCheckbutton',
               'TRadiobutton', 'TScale', 'TSpinbox', 'Button', 'Entry',
               'Checkbutton', 'Menubutton', 'Spinbox'}

# Buttons whose label IS the explanation. A tooltip reading "Close: closes the
# window" is noise, and demanding one would push the suite toward filler.
SELF_EVIDENT = {'OK', 'Cancel', 'Close', 'Done', ''}


def _walk(widget):
    yield widget
    try:
        for child in widget.winfo_children():
            yield from _walk(child)
    except Exception:                                    # noqa: BLE001
        pass


def _label(widget):
    try:
        return str(widget.cget('text'))
    except Exception:                                    # noqa: BLE001
        return ''


def _editor_or_skip():
    """A real editor with Tk up, or skip where there is no display."""
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable('tkinter not available — headless environment')
    try:
        root = tk.Tk()
    except Exception as exc:                             # noqa: BLE001
        gui_unavailable(f'Tk cannot initialise without a display: {exc}')
    root.withdraw()
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _dialog_modules():
    here = os.path.dirname(importlib.import_module('openflo').__file__)
    return sorted(f[:-3] for f in os.listdir(here)
                  if f.startswith('ui_') and f.endswith('.py')
                  and f != 'ui_tips.py')


@pytest.mark.parametrize('modname', _dialog_modules())
def test_dialog_controls_have_hover_help(modname):
    """Open each dialog and require a tooltip on every actionable control.

    Dialogs needing constructor arguments we cannot synthesise are skipped —
    the ones that open are enough to catch a whole module losing its tooltips.
    """
    import tkinter as tk

    root, ed = _editor_or_skip()
    try:
        mod = importlib.import_module(f'openflo.{modname}')
        cls = next((obj for obj in vars(mod).values()
                    if inspect.isclass(obj) and issubclass(obj, tk.Toplevel)
                    and obj.__module__ == mod.__name__), None)
        if cls is None:
            pytest.skip(f'{modname} defines no Toplevel')
        try:
            win = cls(ed)
        except Exception as exc:                         # noqa: BLE001
            pytest.skip(f'{modname} needs state we do not have: '
                        f'{type(exc).__name__}')
        win.update_idletasks()

        missing = []
        for w in _walk(win):
            if w.winfo_class() not in INTERACTIVE:
                continue
            if w.winfo_class() == 'TButton' and _label(w) in SELF_EVIDENT:
                continue
            try:
                bound = w.bind() or ()
            except Exception:                            # noqa: BLE001
                continue
            if '<Enter>' not in bound:
                missing.append(f'{w.winfo_class()} {_label(w)!r}')

        assert not missing, (
            f'{modname}: {len(missing)} control(s) have no hover tooltip. '
            f'Attach one with openflo.ui_tips.tip / auto_tip: {missing[:6]}')
    finally:
        try:
            root.destroy()
        except Exception:                                # noqa: BLE001
            pass


def test_revealing_the_workspace_does_not_clip_its_controls():
    """The docked workspace and the plot's control rows must both fit.

    A Tk container reports `winfo_reqwidth()` (what its children need) against
    `winfo_width()` (what it got); required > actual means content is cut off
    with no way to reach it. Previously the panel opened 17 px short of its own
    toolbar.
    """
    root, ed = _editor_or_skip()
    try:
        ed.deiconify()
        # Narrow on purpose: the proportional 38% share is generous on a wide
        # window and the floor never binds there, so a broken floor would slip
        # through. At laptop width the floor is what keeps the toolbar whole.
        ed.geometry('1100x840')
        ed.update_idletasks()
        ed.update()

        # Pretend the screen is exactly the window's width, so revealing the
        # workspace CANNOT widen the window and has to share what it has.
        # Without this the growth path papers over a bad floor and the
        # assertion below can never fail — verified by reverting the fix.
        ed.winfo_screenwidth = lambda: ed.winfo_width()

        ed._open_pipeline_workspace()
        ed.update_idletasks()
        ed.update()

        panel = getattr(ed, '_workspace_panel', None)
        if panel is None:
            pytest.skip('workspace panel unavailable in this environment')

        got, need = panel.winfo_width(), panel.winfo_reqwidth()

        # The reveal caps the panel at 60% of the pane so it cannot eat the
        # plot. When the panel's own requirement exceeds that cap, the cap
        # wins BY DESIGN and some clipping is unavoidable — something has to
        # give in a window too narrow for both. Asserting "never clipped"
        # here demands the impossible, and did: this failed on Linux/py3.11
        # (needs 343px, capped at 328) purely because font metrics make the
        # toolbar wider there than on Windows, where it needs 336.
        #
        # So assert what the floor is actually responsible for: the panel gets
        # everything the cap allows. A regressed floor shows up as the panel
        # being narrower than BOTH its requirement and the cap.
        pane = ed._editor_paned.winfo_width()
        cap = int(pane * 0.6)
        entitled = min(need, cap)
        assert got + 1 >= entitled, (
            f'workspace opened at {got}px but was entitled to {entitled}px '
            f'(needs {need}, 60% cap of the {pane}px pane is {cap}) — the '
            f'width floor has regressed')
    finally:
        try:
            root.destroy()
        except Exception:                                # noqa: BLE001
            pass


def test_docking_the_workspace_does_not_squeeze_the_plot_controls():
    """With room on the screen, revealing the panel must WIDEN the window
    rather than take the width out of the plot's control rows.

    Docking used to steal the panel's width from the plot column: at the
    default 1500px window that left ~512px against control rows needing ~754,
    clipping the axis combos, `Auto-gate` and `Show cleaned-out events` —
    controls the user never asked to give up.

    Note this is the ROOMY case, deliberately. When the screen genuinely has no
    room, something has to give and the panel's own controls are what get
    protected (the test above). Asserting both at once would demand the
    impossible.
    """
    root, ed = _editor_or_skip()
    try:
        ed.deiconify()
        ed.geometry('1500x840')
        ed.update_idletasks()
        ed.update()
        if ed.winfo_screenwidth() < 1900:
            pytest.skip('screen too small to grow into — nothing to assert')

        ed._open_pipeline_workspace()
        ed.update_idletasks()
        ed.update()

        bar = getattr(ed, '_ctrl_bar', None)
        if bar is None or not bar.winfo_ismapped():
            pytest.skip('control bar unavailable in this environment')
        assert bar.winfo_width() + 1 >= bar.winfo_reqwidth(), (
            f'the docked workspace squeezed the plot controls: '
            f'{bar.winfo_width()}px wide but needs {bar.winfo_reqwidth()}px — '
            f'the window should have grown instead')
    finally:
        try:
            root.destroy()
        except Exception:                                # noqa: BLE001
            pass


def test_a_status_message_cannot_widen_the_workspace():
    """The status line must wrap, not dictate the panel's width.

    This is what actually caused the clipping: an unwrapped Label reports its
    whole single-line text as its required width. The workspace hint asked for
    456 px and, being the widest child, made that the panel's requirement.
    """
    root, ed = _editor_or_skip()
    try:
        ed.deiconify()
        ed.geometry('1500x840')
        ed.update_idletasks()
        ed.winfo_screenwidth = lambda: ed.winfo_width()   # no room to grow
        ed._open_pipeline_workspace()
        ed.update_idletasks()
        panel = getattr(ed, '_workspace_panel', None)
        if panel is None:
            pytest.skip('workspace panel unavailable in this environment')

        panel.status_var.set(
            'A deliberately very long status message, far wider than the '
            'docked panel, of the kind a run failure or a long output path '
            'produces in normal use.')
        panel.update_idletasks()

        assert panel.winfo_reqwidth() < 900, (
            'a long status message widened the workspace to '
            f'{panel.winfo_reqwidth()}px — the status Label is not wrapping')

        # And the label itself must not be demanding that width: this is the
        # actual mechanism, and checking only the panel lets a wide window
        # satisfy the test with the bug still present.
        labels = [w for w in _walk(panel)
                  if w.winfo_class() == 'TLabel'
                  and 'deliberately very long' in _label(w)]
        assert labels, 'status label not found'
        assert labels[0].winfo_reqwidth() <= panel.winfo_width() + 4, (
            f'the status label demands {labels[0].winfo_reqwidth()}px inside a '
            f'{panel.winfo_width()}px panel — it is not wrapping, so it will '
            f'dictate the panel width again')
    finally:
        try:
            root.destroy()
        except Exception:                                # noqa: BLE001
            pass


# The 13 dialogs the tooltip test above cannot construct with the state a test
# has. It skips them politely — which means a dialog that genuinely CRASHES on
# construction is indistinguishable from one that merely needs a real sample,
# and drops out of the check with no complaint. This ratchet closes that: the
# set may SHRINK freely as dialogs are made constructible, but it must not grow.
_UNCONSTRUCTIBLE = {
    'ui_annotation', 'ui_audit', 'ui_autogate', 'ui_axis_config',
    'ui_cell_cycle', 'ui_diff', 'ui_embedding', 'ui_figure_layout',
    'ui_figure_window', 'ui_flowsom_tree', 'ui_logic', 'ui_spectral_qc',
    'ui_spectral_unmix',
}


def test_no_new_dialog_becomes_unconstructible():
    """A construction crash must not hide behind a skip.

    ``test_dialog_controls_have_hover_help`` wraps ``cls(ed)`` in
    ``except Exception: pytest.skip(...)``. That is reasonable — many dialogs
    need a loaded sample — but it means breaking a dialog's ``__init__``
    removes it from the tooltip check silently, and the suite stays green
    while a window no longer opens.

    So: 18 of 31 ui_* dialogs construct today and 13 do not. This asserts the
    13 have not become 14. It never blocks a fix — making a dialog
    constructible only shrinks the set, which passes.
    """
    import tkinter as tk

    root, ed = _editor_or_skip()
    try:
        failed = set()
        # Reuse the same discovery the tooltip test uses, so the two cannot
        # disagree about what a dialog is (ui_tips is a helper, not a window).
        names = _dialog_modules()
        assert names, 'no ui_* modules found — this test is testing nothing'

        for modname in names:
            mod = importlib.import_module(f'openflo.{modname}')
            cls = next((obj for obj in vars(mod).values()
                        if inspect.isclass(obj)
                        and issubclass(obj, tk.Toplevel)
                        and obj.__module__ == mod.__name__), None)
            if cls is None:
                failed.add(modname)          # no Toplevel = nothing to check
                continue
            try:
                win = cls(ed)
            except Exception:                # noqa: BLE001
                failed.add(modname)
            else:
                try:
                    win.destroy()
                except Exception:            # noqa: BLE001
                    pass

        new = failed - _UNCONSTRUCTIBLE
        assert not new, (
            f'{sorted(new)} can no longer be constructed. Either a dialog is '
            f'broken — which would otherwise vanish from the tooltip check as '
            f'a skip — or it needs new state; if the latter, add it to '
            f'_UNCONSTRUCTIBLE deliberately.')
    finally:
        try:
            root.destroy()
        except Exception:                    # noqa: BLE001
            pass
