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
import ast
import importlib
import inspect
import os

import pytest

from tests.conftest import editor_or_skip

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
    except Exception:                                    # noqa: BLE001 - a widget destroyed mid-walk has no children to visit
        pass


def _label(widget):
    try:
        return str(widget.cget('text'))
    except Exception:                                    # noqa: BLE001 - skip widgets that have no text option
        return ''


def _describe(widget):
    """How a failure names a control: its label, or for an entry, combobox,
    scale or spinbox (which have none; ``cget('text')`` would answer with
    the name of its variable) its current value."""
    cls = widget.winfo_class()
    if cls.lstrip('T') in ('Button', 'Checkbutton', 'Radiobutton',
                           'Menubutton'):
        return f'{cls} {_label(widget)!r}'
    try:
        value = widget.get()
    except Exception:                                    # noqa: BLE001 - not every widget has get(); the description then shows no value
        value = ''
    if isinstance(value, float):
        value = f'{value:g}'
    return f'{cls} value={str(value)!r}'


def _editor_or_skip():
    """A real editor with Tk up, or skip where there is no display."""
    return editor_or_skip()


def _dialog_modules():
    """Every ui_* module that defines a window (ui_tips and ui_logic hold
    helpers only; ui_style holds the ttk theming; ui_frequency holds the
    Frequencies by group panel, checked inside Statistics through its
    _OPENERS entry)."""
    here = os.path.dirname(importlib.import_module('openflo').__file__)
    return sorted(f[:-3] for f in os.listdir(here)
                  if f.startswith('ui_') and f.endswith('.py')
                  and f not in ('ui_tips.py', 'ui_logic.py', 'ui_notice.py',
                                'ui_style.py', 'ui_frequency.py'))


def _window_classes():
    """``(module, class)`` for EVERY window class a ui_* module defines.

    Read from the source rather than by importing, so collection stays cheap
    and Tk-free. A module often holds more than one window (an optimizer next
    to its editor, a results window next to its launcher); checking only the
    first let four of them go unchecked. ``test_no_new_dialog_becomes_
    unconstructible`` checks this list against the real Toplevel subclasses,
    so a window this reading misses fails there instead of going unchecked.
    """
    here = os.path.dirname(importlib.import_module('openflo').__file__)
    out = []
    for modname in _dialog_modules():
        with open(os.path.join(here, f'{modname}.py'), encoding='utf-8') as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and any(
                    ast.unparse(b).endswith(('Toplevel', 'ToolWindow'))
                    for b in node.bases):
                out.append((modname, node.name))
    return out


def _untipped(top, skip_toplevels=False):
    """Every actionable control under ``top`` with no hover help, as
    'Class 'label''. The test reads the ``<Enter>`` binding a tooltip makes;
    it never moves the pointer or waits for a tip to show."""
    import tkinter as tk
    missing = []
    stack = [top]
    while stack:
        w = stack.pop()
        try:
            kids = w.winfo_children()
        except Exception:                                # noqa: BLE001 - a widget destroyed mid-walk has no children to visit
            kids = []
        for k in kids:
            if skip_toplevels and isinstance(k, (tk.Toplevel, tk.Menu)):
                continue                                 # another window
            stack.append(k)
        if w.winfo_class() not in INTERACTIVE:
            continue
        if w.winfo_class() == 'TButton' and _label(w) in SELF_EVIDENT:
            continue
        try:
            bound = w.bind() or ()
        except Exception:                                # noqa: BLE001 - skip widgets whose bindings cannot be read
            continue
        if '<Enter>' not in bound:
            missing.append(_describe(w))
    return sorted(missing)


# ── Opening each dialog on generated data ────────────────────────────────────
#
# Seventeen windows take more than the editor to open (_UNCONSTRUCTIBLE).
# They used to be skipped ("needs state we do not have"), so their tooltips
# were never checked at all. Each gets what it needs here, built from
# generated data: synthetic channels (FL1-A, ...) and sample names only.

def _register(ed, s):
    name = s.name
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._sample_gates[name] = {}
    ed._sample_gate_order[name] = []
    ed._sample_gate_seq[name] = 0
    ed._channels = list(s.data.columns)
    ed._channel_labels = dict(s.channel_labels)
    ed._active_sample = name
    return name


def _blobs_sample(name='gen'):
    import numpy as np
    import pandas as pd

    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.5, (300, 3)) for c in (0.0, 6.0, 12.0)])
    df = pd.DataFrame(X, columns=['FL1-A', 'FL2-A', 'FL3-A'])
    df['leiden'] = np.repeat([0, 1, 2], 300)
    return FlowSample.from_dataframe(df, name=name)


def _cell_cycle_sample():
    from openflo import synthetic
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(
        synthetic.cell_cycle_sample(n=4000, seed=0), name='cc')
    s.cell_cycle(dna_channel='DAPI-A')
    assert s.cell_cycle_result and s.cell_cycle_result.get('ok')
    return s


def _flowsom_sample():
    s = _blobs_sample('fs')
    s.fluor_channels = ['FL1-A', 'FL2-A', 'FL3-A']
    s.run_flowsom(grid=(4, 4), n_metaclusters=3, iters=4, seed=0)
    return s


def _compare_results():
    from openflo.stats import compare_all_features
    return compare_all_features({'up': {'A': [1, 2, 3], 'B': [7, 8, 9]},
                                 'flat': {'A': [5, 5, 6], 'B': [5, 6, 5]}})


def _spectral_qc():
    import numpy as np

    from openflo.spectral import unmixing_qc
    S = np.array([[0.8, 0.5, 0.1, 0.0], [0.3, 0.9, 0.2, 0.05],
                  [0.1, 0.2, 0.9, 0.3]])
    return unmixing_qc({}, S, ['FL1', 'FL2', 'FL3'])


def _figure():
    from matplotlib.figure import Figure
    fig = Figure(figsize=(3, 2))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    return fig


def _audit_log():
    from openflo.audit import AuditLog
    log = AuditLog()
    log.record('load', sample='gen', events=900)
    return log


def _noop(*_a, **_k):
    return None


def _layout_items(cls, ed):
    """The layout editor's item dialog, once per kind of item: each kind
    builds a different set of controls. Its parent is the layout editor,
    which lends it the editor (``.ed``) and the page's sample."""
    import tkinter as tk
    name = _register(ed, _blobs_sample())
    host = tk.Toplevel(ed)
    host.withdraw()
    host.ed = ed
    host._sample = lambda: name
    box = (0.0, 0.0, 0.3, 0.3)
    return [cls(host, {'kind': kind, 'box': box, 'x': 'FL1-A', 'y': 'FL2-A',
                       'channel': 'FL1-A', 'xlim': (0.0, 5.0)},
                [(None, 'All events')], _noop)
            for kind in ('plot', 'stats', 'chart', 'text')]


def _diff_rows():
    """Two populations of a two-group differential-abundance result."""
    return [{'cluster': f'Pop{i}', 'log2fc': 0.5 * i, 'prop_a': 0.1,
             'prop_b': 0.2, 'n_a': 3, 'n_b': 3, 'n_missing': 0, 'p': 0.04,
             'p_adj': 0.08} for i in (1, 2)]


def _gaussian_cleanup(cls, ed):
    """A sample with Gaussian parameters, so the per-parameter boxes show."""
    import numpy as np
    import pandas as pd

    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.lognormal(3, 0.3, (500, 4)),
                      columns=['FL1-A', 'Event_length', 'Center', 'Width'])
    _register(ed, FlowSample.from_dataframe(df, name='gen'))
    return cls(ed)


# Keyed 'module.Class'. A window missing here is opened as ``cls(editor)``.
def _statistics_both_tabs(cls, ed):
    win = cls(ed)
    win.show_compare()
    return win


_OPENERS = {
    'ui_annotation.PopulationAnnotationWindow':
        lambda cls, ed: cls(ed, _register(ed, _blobs_sample())),
    'ui_audit.AuditWindow': lambda cls, ed: cls(ed, _audit_log()),
    'ui_autogate.AutoGateDialog':
        lambda cls, ed: cls(ed, True, 'FSC-A', 'FSC-H', 'FL1-A', 'FL2-A',
                            _noop),
    'ui_axis_config.AxisConfigDialog':
        lambda cls, ed: cls(ed, 'FL1-A', 'log', (1.0, 1e5), _noop,
                            show_link=True),
    'ui_cell_cycle.CellCycleWindow':
        lambda cls, ed: cls(ed, _register(ed, _cell_cycle_sample())),
    'ui_comp.OptimizeCompensationDialog':
        lambda cls, ed: cls(ed, ['FL1-A', 'FL2-A'], _noop),
    'ui_diff.CompareAllWindow':
        lambda cls, ed: cls(ed, _compare_results(), ['A', 'B'], '%Parent'),
    'ui_diff.DiffAbundanceWindow':
        lambda cls, ed: cls(ed, _diff_rows(), ['A', 'B'], ['s1', 's2']),
    'ui_embedding.EmbeddingDialog': lambda cls, ed: cls(
        ed, 'gen', 900, ['umap', 'tsne'], _blobs_sample().data,
        ['FL1-A', 'FL2-A', 'FL3-A']),
    'ui_figure_layout.FigureLayoutDialog':
        lambda cls, ed: cls(ed, 2, 'overlay', '', _noop),
    'ui_figure_window.FigureWindow':
        lambda cls, ed: cls(ed, _figure(), 'Figure'),
    'ui_flowsom_tree.FlowSOMTreeWindow':
        lambda cls, ed: cls(ed, _register(ed, _flowsom_sample())),
    'ui_layout_dialogs.ItemDialog': _layout_items,
    'ui_layout_dialogs.LayerColorsDialog': lambda cls, ed: cls(
        ed, [('L1', 'gen · All events', '#1f77b4')], {'L1': '#d62728'},
        _noop),
    'ui_mass_cytometry.GaussianCleanupWindow': _gaussian_cleanup,
    'ui_spectral_qc.SpectralQCWindow': lambda cls, ed: cls(ed, _spectral_qc()),
    # Both tabs: the Frequencies by group panel is built when first shown.
    'ui_statistics.StatisticsWindow': _statistics_both_tabs,
    'ui_spectral_unmix.SpectralUnmixDialog':
        lambda cls, ed: cls(ed, ['a', 'b'], ['D1-A', 'D2-A'], _noop),
    'ui_unmixed_pairs.UnmixedPairsWindow': lambda cls, ed: cls(
        ed, _blobs_sample().data[['FL1-A', 'FL2-A', 'FL3-A']].to_numpy(),
        ['FL1-A', 'FL2-A', 'FL3-A'], 'gen'),
}


@pytest.mark.parametrize('modname,clsname', _window_classes(),
                         ids=lambda v: v)
def test_dialog_controls_have_hover_help(modname, clsname):
    """Open each window and require a tooltip on every actionable control.

    EVERY window class of every ui_* module, not only the first one a module
    defines: windows that need more than the editor get it from _OPENERS,
    built on generated data, so none is skipped.
    """
    import tkinter as tk

    root, ed = _editor_or_skip()
    try:
        cls = getattr(importlib.import_module(f'openflo.{modname}'), clsname)
        assert issubclass(cls, tk.Toplevel), f'{clsname} is not a window'
        opened = _OPENERS.get(f'{modname}.{clsname}', lambda c, e: c(e))(
            cls, ed)
        missing = []
        for win in (opened if isinstance(opened, list) else [opened]):
            win.update_idletasks()
            missing += _untipped(win)

        assert not missing, (
            f'{modname}.{clsname}: {len(missing)} control(s) have no hover '
            f'tooltip. Attach one with openflo.ui_tips.tip / auto_tip: '
            f'{missing[:6]}')
    finally:
        try:
            root.destroy()
        except Exception:                                # noqa: BLE001 - teardown; the root may already be destroyed, and a teardown error must not hide the test's own result
            pass


def test_main_window_controls_have_hover_help():
    """The editor's own controls carry hover help too: the side panel, the
    plot's control rows, the log pane and the histogram sliders. Dialogs and
    menus it owns are other windows and are checked above.
    """
    root, ed = _editor_or_skip()
    try:
        ed.update_idletasks()
        missing = set(_untipped(ed, skip_toplevels=True))
        assert not missing, (
            f'main window: {len(missing)} control(s) have no hover tooltip. '
            f'Attach one with self._tip: {sorted(missing)[:6]}')
    finally:
        try:
            root.destroy()
        except Exception:                                # noqa: BLE001 - teardown; the root may already be destroyed, and a teardown error must not hide the test's own result
            pass


@pytest.mark.parametrize('side_px', [None, 760])
def test_revealing_the_workspace_does_not_clip_its_controls(side_px):
    """The docked workspace and the plot's control rows must both fit.

    A Tk container reports `winfo_reqwidth()` (what its children need) against
    `winfo_width()` (what it got); required > actual means content is cut off
    with no way to reach it. Previously the panel opened 17 px short of its own
    toolbar.

    With ``side_px`` the side panel is that wide, as a wide one was on macOS,
    which leaves the pane so little that the 60% cap binds: the panel then
    opened short of the cap by the sash's own width.
    """
    root, ed = _editor_or_skip()
    try:
        if side_px:
            import tkinter as tk
            tk.Frame(ed._left_host, width=side_px, height=1).pack(side='top')
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
        assert panel is not None, (
            'the Pipeline Workspace panel was not built: _open_pipeline_workspace '
            'left _workspace_panel unset, so the checks below cannot run')

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
        if side_px:
            assert cap < need, (
                f'the cap did not bind (needs {need}, cap {cap}): this case '
                f'no longer tests the capped share')
    finally:
        try:
            root.destroy()
        except Exception:                                # noqa: BLE001 - teardown; the root may already be destroyed, and a teardown error must not hide the test's own result
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
        # A roomy screen, whatever the real one is (a CI display can be
        # small): this test is about the case where there IS room to grow.
        ed.winfo_screenwidth = lambda: 2560

        ed._open_pipeline_workspace()
        ed.update_idletasks()
        ed.update()

        bar = getattr(ed, '_ctrl_bar', None)
        assert bar is not None, 'the editor built no plot control bar'
        if not bar.winfo_ismapped():
            pytest.skip('control bar not mapped in this environment')
        real = root.winfo_screenwidth()
        if (bar.winfo_width() + 1 < bar.winfo_reqwidth()
                and ed.winfo_width() >= real - 16):
            # The window did try to grow, but this window manager will not
            # make it wider than the real screen (Windows on a small CI
            # display). X lets it grow, so Linux CI still runs the check.
            pytest.skip(f'the real screen ({real}px) cannot hold the grown '
                        f'window ({ed.winfo_width()}px wide)')
        assert bar.winfo_width() + 1 >= bar.winfo_reqwidth(), (
            f'the docked workspace squeezed the plot controls: '
            f'{bar.winfo_width()}px wide but needs {bar.winfo_reqwidth()}px — '
            f'the window should have grown instead')
    finally:
        try:
            root.destroy()
        except Exception:                                # noqa: BLE001 - teardown; the root may already be destroyed, and a teardown error must not hide the test's own result
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
        assert panel is not None, (
            'the Pipeline Workspace panel was not built: _open_pipeline_workspace '
            'left _workspace_panel unset, so the checks below cannot run')

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
        except Exception:                                # noqa: BLE001 - teardown; the root may already be destroyed, and a teardown error must not hide the test's own result
            pass


# The windows that cannot be opened as ``cls(editor)``: each needs data (a
# sample, a result, an item) that the tooltip test builds in _OPENERS. A
# window that starts failing ``cls(editor)`` for a new reason is either broken
# or newly needs state, so this set may SHRINK freely but must not grow, and
# each one must have an opener so the tooltip test still opens it.
_UNCONSTRUCTIBLE = {
    'ui_annotation.PopulationAnnotationWindow', 'ui_audit.AuditWindow',
    'ui_autogate.AutoGateDialog', 'ui_axis_config.AxisConfigDialog',
    'ui_cell_cycle.CellCycleWindow', 'ui_diff.CompareAllWindow',
    'ui_embedding.EmbeddingDialog', 'ui_figure_layout.FigureLayoutDialog',
    'ui_figure_window.FigureWindow', 'ui_flowsom_tree.FlowSOMTreeWindow',
    'ui_spectral_qc.SpectralQCWindow', 'ui_spectral_unmix.SpectralUnmixDialog',
    'ui_unmixed_pairs.UnmixedPairsWindow',   # needs the arrays of an unmix
    'ui_comp.OptimizeCompensationDialog',    # the channels to optimize
    'ui_diff.DiffAbundanceWindow',           # the rows of a result
    'ui_layout_dialogs.ItemDialog',          # the layout editor, an item
    'ui_layout_dialogs.LayerColorsDialog',   # an item's layers
}


def test_no_new_dialog_becomes_unconstructible():
    """A construction crash must not hide, and no window may go unchecked.

    The tooltip test opens each window with ``cls(editor)`` unless _OPENERS
    says how. So: every window that cannot open that way is listed in
    _UNCONSTRUCTIBLE, deliberately, and has an opener; a new failure means a
    broken ``__init__`` or new state it needs, and either way someone looks.
    Making a window constructible only shrinks the set, which passes.

    It also checks the tooltip test's window list (read from the source) is
    the real one: every Toplevel subclass each ui_* module defines.
    """
    import tkinter as tk

    root, ed = _editor_or_skip()
    try:
        listed = _window_classes()
        assert listed, 'no ui_* windows found — this test is testing nothing'
        real = []
        for modname in _dialog_modules():
            mod = importlib.import_module(f'openflo.{modname}')
            found = [(modname, name) for name, obj in vars(mod).items()
                     if inspect.isclass(obj) and issubclass(obj, tk.Toplevel)
                     and obj.__module__ == mod.__name__]
            assert found, f'{modname} defines no window'
            real += found
        assert sorted(real) == sorted(listed), (
            f'the tooltip test misses {sorted(set(real) - set(listed))} or '
            f'checks a non-window {sorted(set(listed) - set(real))}: '
            f'update _window_classes')

        failed = set()
        for modname, clsname in listed:
            cls = getattr(importlib.import_module(f'openflo.{modname}'),
                          clsname)
            try:
                win = cls(ed)
            except Exception:                # noqa: BLE001 - any construction error is collected and compared with _UNCONSTRUCTIBLE below
                failed.add(f'{modname}.{clsname}')
            else:
                try:
                    win.destroy()
                except Exception:            # noqa: BLE001 - teardown; the window may already be gone, and a teardown error must not hide the test's own result
                    pass

        new = failed - _UNCONSTRUCTIBLE
        assert not new, (
            f'{sorted(new)} can no longer be constructed. Either a window is '
            f'broken, or it needs new state; if the latter, give it an opener '
            f'in _OPENERS and add it to _UNCONSTRUCTIBLE deliberately.')
        unopened = _UNCONSTRUCTIBLE - set(_OPENERS)
        assert not unopened, (
            f'{sorted(unopened)} need state to open but have no opener in '
            f'_OPENERS, so the tooltip test cannot check them')
    finally:
        try:
            root.destroy()
        except Exception:                    # noqa: BLE001 - teardown; the root may already be destroyed, and a teardown error must not hide the test's own result
            pass
