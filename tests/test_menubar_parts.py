"""The menubar's parts: each menu's entries in order (type, label,
accelerator, Tk variable and value, the handler a click reaches), what a
failure part-way through the build leaves, and a rebuild.

MenuMixin._build_menubar builds the six buttons and then File, Edit, View,
Analyze, Tools and Help, flattens and binds every menu and submenu, and
themes the buttons, all inside one try. A failure in one menu leaves the
menus before it attached, the rest unbuilt, nothing flattened or themed,
and one note. tests/test_menus.py rebuilds the menubar, so the build must
work again after the buttons are destroyed.

The editor holds no samples here; nothing reads a file. The accelerators are
written as on Windows and Linux and shown through ui_logic.accel.
"""
from __future__ import annotations

import tkinter as tk

import pytest

import openflo.editor_menu as editor_menu
from openflo import errors
from openflo.theme import current_palette
from openflo.ui_logic import accel
from tests.conftest import editor_or_skip

# path of cascade labels: [(type, label, accelerator[, variable attribute
# [, value]], handler)]; handler is the editor method a click calls, or
# "lambda:" and the names the lambda reads. Populations and Templates are
# filled when they open; Open recent is filled at build time too and is only
# checked for being there.
MENUS = {
    ('File',): [
        ('command', 'Add FCS…', 'Ctrl+Shift+A', '_add_samples'),
        ('command', 'Load CSV…', '', '_load_processed_data'),
        ('command', 'Load example dataset', '', '_load_example_data'),
        ('command', 'Generate dataset…', '', '_open_synthetic_dialog'),
        ('command', 'Quick preview…', '', '_open_quick_preview'),
        ('command', 'Concatenate samples…', '', '_open_concatenate'),
        ('separator',),
        ('command', 'New window', 'Ctrl+N', '_open_new_window'),
        ('command', 'Open session…', 'Ctrl+O', '_load_session'),
        ('cascade', 'Open recent', '', ''),
        ('command', 'Save session', 'Ctrl+S', '_save_session'),
        ('command', 'Save session as…', '', '_save_session_as'),
        ('command', 'Close session', '', '_close_session'),
        ('command', 'Export project archive…', '', '_export_project_archive'),
        ('command', 'Expand project archive…', '', '_expand_project_archive'),
        ('command', 'Nearby send…', '', '_open_nearby_send'),
        ('command', 'Upgrade saved session…', '', '_upgrade_session_file'),
        ('command', 'Compare with saved session…', '', '_compare_saved_session'),
        ('separator',),
        ('cascade', 'Export', '', ''),
        ('command', 'Copy plot to clipboard', '', '_copy_plot_image'),
        ('separator',),
        ('command', 'Close window', 'Ctrl+W', '_on_close'),
    ],
    ('File', 'Open recent'): [
        ('command', '(no recent sessions)', '', ''),
    ],
    ('File', 'Export'): [
        ('command', 'FlowJo workspace…', 'Ctrl+E', '_export_flowjo_wsp'),
        ('checkbutton', 'Copy FCS files with FlowJo workspace', '', '_wsp_copy_fcs_var', 'lambda:write_pref,_wsp_copy_fcs_var'),
        ('command', 'Gating-ML 2.0 gates…', '', '_export_gatingml'),
        ('command', 'Gate list (CSV)…', '', '_export_gate_list'),
        ('command', 'Plot image…', 'Ctrl+Shift+S', '_save_plot_image'),
        ('command', 'Populations (FCS)…', '', '_export_populations_fcs'),
        ('command', 'Populations, all samples (FCS)…', '', '_export_populations_fcs_all'),
        ('command', 'Analysis report (HTML)…', '', '_export_report'),
    ],
    ('Edit',): [
        ('command', 'Undo', 'Ctrl+Z', '_undo'),
        ('command', 'Redo', 'Ctrl+Y', '_redo'),
        ('separator',),
        ('command', 'Find sample or gate', 'Ctrl+F', '_focus_find'),
        ('separator',),
        ('command', 'Delete gate', 'Delete', '_clear_selected_gate'),
        ('command', 'Delete all gates', '', '_clear_all'),
        ('command', 'Copy gates to…', '', '_open_copy_gates_dialog'),
        ('command', 'Group samples…', '', '_open_group_samples'),
        ('cascade', 'Populations', '', ''),
        ('separator',),
        ('cascade', 'Gates', '', ''),
        ('separator',),
        ('command', 'Preferences…', 'Ctrl+,', '_open_preferences'),
    ],
    ('Edit', 'Populations'): [
    ],
    ('Edit', 'Gates'): [
        ('command', 'Add singlet gate', '', '_add_singlet_gate'),
        ('command', 'FMO gating…', '', '_open_fmo_gating'),
        ('command', 'Auto-clean gate', '', '_create_autoclean_gate'),
        ('command', 'Create boolean gate…', '', '_open_boolean_dialog'),
        ('command', 'Combination gates…', '', '_open_combination_dialog'),
        ('command', 'Edit gate coordinates…', '', '_edit_gate_coordinates'),
        ('command', 'Re-place gate on each sample…', '', '_refit_gate_per_sample'),
    ],
    ('View',): [
        ('cascade', 'Display', '', ''),
        ('cascade', 'Density colors', '', ''),
        ('cascade', 'Group colors', '', ''),
        ('command', 'Reset plot view', 'Ctrl+0', '_reset_plot_view'),
        ('command', 'Zoom in', 'Ctrl++', 'lambda:_zoom_step'),
        ('command', 'Zoom out', 'Ctrl+-', 'lambda:_zoom_step'),
        ('command', 'Redraw plot', 'F5', 'lambda:_schedule_replot'),
        ('separator',),
        ('command', 'Pipeline workspace', 'F9', '_open_pipeline_workspace'),
        ('checkbutton', 'Show log / console', 'Ctrl+`', '_show_log_var', '_toggle_log'),
        ('checkbutton', 'Show hover tips', '', '_tooltips_enabled', 'lambda:write_pref,_tooltips_enabled'),
        ('checkbutton', 'Gate labels on plots', '', '_gate_labels_var', '_toggle_gate_labels'),
        ('checkbutton', 'FMO on histograms', '', '_fmo_hist_var', '_toggle_fmo_histograms'),
        ('cascade', 'New windows open at', '', ''),
        ('command', 'Dock all panels', '', '_dock_all_panels'),
        ('separator',),
        ('cascade', 'Theme', '', ''),
        ('cascade', 'Plot background', '', ''),
    ],
    ('View', 'Display'): [
        ('radiobutton', 'All events', 'Ctrl+1', 'gate_display_var', 'all', '_apply_display_mode'),
        ('radiobutton', 'Highlight gated', 'Ctrl+2', 'gate_display_var', 'highlight', '_apply_display_mode'),
        ('radiobutton', 'Filter to gated', 'Ctrl+3', 'gate_display_var', 'filter', '_apply_display_mode'),
    ],
    ('View', 'Density colors'): [
        ('radiobutton', 'Turbo', '', '_density_cmap_var', 'turbo', '_set_density_cmap'),
        ('radiobutton', 'Classic rainbow', '', '_density_cmap_var', 'jet', '_set_density_cmap'),
        ('radiobutton', 'Viridis', '', '_density_cmap_var', 'viridis', '_set_density_cmap'),
    ],
    ('View', 'Group colors'): [
        ('radiobutton', 'Color-blind safe (Okabe-Ito)', '', '_group_palette_var', 'okabe_ito', '_set_group_palette'),
        ('radiobutton', 'Classic (tab10)', '', '_group_palette_var', 'classic', '_set_group_palette'),
    ],
    ('View', 'New windows open at'): [
        ('radiobutton', 'OS default', '', '_spawn_corner', 'off', 'lambda:write_pref,_spawn_corner'),
        ('radiobutton', 'Top-left of main window', '', '_spawn_corner', 'top-left', 'lambda:write_pref,_spawn_corner'),
        ('radiobutton', 'Top-right of main window', '', '_spawn_corner', 'top-right', 'lambda:write_pref,_spawn_corner'),
    ],
    ('View', 'Theme'): [
        ('radiobutton', 'Light', '', '_chrome_var', 'light', '_set_theme_parts'),
        ('radiobutton', 'Dark', '', '_chrome_var', 'dark', '_set_theme_parts'),
    ],
    ('View', 'Plot background'): [
        ('radiobutton', 'White', '', '_plot_bg_var', 'light', '_set_theme_parts'),
        ('radiobutton', 'Black', '', '_plot_bg_var', 'dark', '_set_theme_parts'),
    ],
    ('Analyze',): [
        ('command', 'Statistics…', 'Ctrl+T', '_open_stats_window'),
        ('command', 'Frequencies…', '', '_open_frequency_window'),
        ('command', 'Expression…', '', '_open_expression_window'),
        ('command', 'Group comparison…', '', '_open_group_stats'),
        ('command', 'Sample QC…', '', '_open_sample_qc_window'),
        ('command', 'Saturation report…', '', '_open_saturation_report'),
        ('command', 'Kinetics…', '', '_open_kinetics_window'),
        ('command', 'Plate view…', '', '_open_plate_view'),
        ('command', 'Absolute counts…', '', '_open_abs_counts'),
        ('separator',),
        ('command', 'Methods & provenance…', '', '_open_methods_report'),
        ('command', 'History / audit…', '', '_show_audit_window'),
        ('separator',),
        ('command', 'Cluster…', '', '_open_cluster_dialog'),
        ('command', 'Compare embeddings…', '', '_open_dr_compare'),
        ('command', 'Cell cycle…', '', '_open_cell_cycle_dialog'),
        ('command', 'Proliferation…', '', '_open_proliferation'),
        ('command', 'Trajectory…', '', '_open_trajectory_window'),
        ('command', 'Annotate…', '', '_open_annotation_window'),
        ('command', 'SOM tree…', '', '_open_flowsom_tree'),
    ],
    ('Tools',): [
        ('cascade', 'Compensation & spectral', '', ''),
        ('cascade', 'Channels', '', ''),
        ('cascade', 'Normalize & clean', '', ''),
        ('cascade', 'Figures', '', ''),
        ('cascade', 'Panel & instrument QC', '', ''),
        ('cascade', 'Files & checks', '', ''),
        ('separator',),
        ('cascade', 'Templates', '', ''),
        ('command', 'Save template…', '', '_save_template'),
    ],
    ('Tools', 'Compensation & spectral'): [
        ('command', 'Compensation…', '', '_open_comp_editor'),
        ('command', 'Compensation QC…', '', '_open_comp_qc'),
        ('command', 'Spectral unmix…', '', '_open_spectral_unmix'),
        ('command', 'Spectral QC…', '', '_show_spectral_qc'),
    ],
    ('Tools', 'Channels'): [
        ('command', 'Transforms…', '', '_open_transform_editor'),
        ('command', 'Calibration…', '', '_open_calibration_dialog'),
        ('command', 'Derived channels…', '', '_open_derived_channels'),
    ],
    ('Tools', 'Normalize & clean'): [
        ('command', 'Normalize batches (CytoNorm)…', '', '_batch_correct_cytonorm'),
        ('command', 'Undo batch normalization', '', '_restore_batch_norm'),
        ('command', 'Debarcode…', '', '_open_debarcode_window'),
        ('cascade', 'Mass cytometry', '', ''),
    ],
    ('Tools', 'Normalize & clean', 'Mass cytometry'): [
        ('command', 'Bead normalization…', '', '_open_bead_normalize'),
        ('command', 'Undo bead normalization', '', '_restore_bead_norm'),
        ('command', 'Gaussian parameter clean-up…', '', '_open_gaussian_cleanup'),
    ],
    ('Tools', 'Figures'): [
        ('command', 'Layout editor (beta)…', '', '_open_layout_editor'),
        ('command', 'Quick figure grid…', '', '_open_figure_layout'),
        ('command', 'Gating tree diagram…', '', '_open_gate_tree'),
    ],
    ('Tools', 'Panel & instrument QC'): [
        ('command', 'Panel check…', '', '_open_panel_check'),
        ('command', 'Label channels from vial labels…', '', '_open_vial_labels'),
        ('command', 'Spreading matrix…', '', '_open_spread_matrix'),
        ('command', 'Bead tracking…', '', '_open_bead_tracking'),
        ('command', 'Voltage optimization…', '', '_open_voltage_dialog'),
    ],
    ('Tools', 'Files & checks'): [
        ('command', 'Compare FlowJo workspace…', '', '_open_compare_wsp'),
        ('command', 'FCS inspector…', '', '_open_fcs_inspector'),
        ('command', 'Watch folder…', '', '_toggle_watch_folder'),
        ('command', 'Review fixes for this session…', '', '_review_fixes'),
    ],
    ('Tools', 'Templates'): [
    ],
    ('Help',): [
        ('command', 'Check for updates…', '', '_check_for_updates'),
        ('command', "What's new", '', '_show_whats_new'),
        ('command', 'Run diagnostics…', '', '_run_diagnostics'),
        ('command', 'Environment…', '', '_show_environment'),
        ('command', 'Report a problem…', '', '_report_a_problem'),
        ('separator',),
        ('command', 'User guide', 'F1', '_open_guide'),
        ('command', 'Getting started tour', '', '_start_tour'),
        ('command', 'Open the sample data', '', '_open_sample_data'),
        ('command', 'Online documentation', '', '_open_documentation'),
        ('command', 'Keyboard shortcuts', '', '_show_shortcuts'),
        ('command', 'About OpenFlo', '', '_show_about'),
    ],
}


@pytest.fixture
def ed():
    root, editor = editor_or_skip()
    try:
        yield editor
    finally:
        root.destroy()


def _tops(ed):
    """{label: button} of the menubar strip, in strip order."""
    return {b.cget('text'): b for b in ed._menubar_frame.winfo_children()}


def _attached(ed):
    """The labels of the buttons that carry a menu."""
    return [t for t, b in _tops(ed).items() if str(b.cget('menu'))]


def _walk(menu, path, out):
    """``out`` with ``menu`` and each submenu under it, by label path."""
    out[path] = menu
    end = menu.index('end')
    for i in range(0 if end is None else end + 1):
        if menu.type(i) == 'cascade':
            _walk(menu.nametowidget(menu.entrycget(i, 'menu')),
                  path + (menu.entrycget(i, 'label'),), out)
    return out


def _menus(ed):
    """{path: tk.Menu} of every menu under every attached button."""
    out = {}
    for label, b in _tops(ed).items():
        if str(b.cget('menu')):
            _walk(b.nametowidget(b.cget('menu')), (label,), out)
    return out


def _all_menu_widgets(ed):
    """Every tk.Menu under the menubar strip, attached or not."""
    out, todo = [], list(ed._menubar_frame.winfo_children())
    while todo:
        w = todo.pop(0)
        if isinstance(w, tk.Menu):
            out.append(w)
        todo += w.winfo_children()
    return out


def _rows(ed, menu):
    """The golden form of ``menu``'s entries (handler left out)."""
    names = {str(v): k for k, v in vars(ed).items()
             if isinstance(v, tk.Variable)}
    end = menu.index('end')
    rows = []
    for i in range(0 if end is None else end + 1):
        t = menu.type(i)
        if t == 'separator':
            rows.append(('separator',))
            continue
        row = [t, menu.entrycget(i, 'label'),
               menu.entrycget(i, 'accelerator')]
        if t in ('radiobutton', 'checkbutton'):
            row.append(names.get(str(menu.entrycget(i, 'variable')), '?'))
            if t == 'radiobutton':
                row.append(str(menu.entrycget(i, 'value')))
        rows.append(tuple(row))
    return rows


def _want(rows):
    """Golden rows without the handler, accelerators as this system shows
    them."""
    out = []
    for r in rows:
        if r[0] == 'separator':
            out.append(r)
        else:
            out.append((r[0], r[1], accel(r[2]), *r[3:-1]))
    return out


def _rebuild(ed):
    """Destroy the buttons and build the menubar again, as test_menus does."""
    for b in ed._menubar_buttons:
        b.destroy()
    ed._build_menubar()
    ed.update_idletasks()


def test_every_menu_holds_its_entries_in_order(ed):
    menus = _menus(ed)
    assert list(menus) == list(MENUS)
    for path, rows in MENUS.items():
        if path == ('File', 'Open recent'):
            assert _rows(ed, menus[path]), path     # filled at build time
            continue
        assert _rows(ed, menus[path]) == _want(rows), path
    assert [b.cget('text') for b in ed._menubar_buttons] == [
        'File', 'Edit', 'View', 'Analyze', 'Tools', 'Help']
    assert ed._display_menu is menus[('View', 'Display')]
    assert ed._watch_menu is menus[('Tools', 'Files & checks')]


def test_main_window_file_menu_ends_in_quit(ed):
    ed._primary = True
    _rebuild(ed)
    last = _rows(ed, _menus(ed)[('File',)])[-1]
    assert last == ('command', 'Quit OpenFlo', accel('Ctrl+Q'))


def test_each_entry_reaches_its_handler(ed, monkeypatch):
    """Every handler is replaced by a recorder before a rebuild, then each
    entry is invoked: it must reach the handler the golden list names."""
    calls = []
    handlers = set()
    for rows in MENUS.values():
        for r in rows:
            if r[0] != 'separator' and r[-1]:
                h = r[-1]
                handlers.update(h[len('lambda:'):].split(',')
                                if h.startswith('lambda:') else [h])
    for h in sorted(handlers):
        if h == 'write_pref':
            monkeypatch.setattr(editor_menu, 'write_pref',
                                lambda *a, **k: calls.append('write_pref'))
        elif callable(getattr(type(ed), h, None)):
            monkeypatch.setattr(
                type(ed), h,
                lambda self, *a, _h=h, **k: calls.append(_h))
    _rebuild(ed)
    menus = _menus(ed)
    seen = 0
    for path, rows in MENUS.items():
        if path == ('File', 'Open recent'):
            continue
        for i, r in enumerate(rows):
            if r[0] in ('separator', 'cascade') or not r[-1]:
                continue
            del calls[:]
            menus[path].entryconfigure(i, state='normal')  # e.g. Highlight
            menus[path].invoke(i)
            h = r[-1]
            want = (h[len('lambda:'):].split(',')[0]
                    if h.startswith('lambda:') else h)
            assert want in calls, (path, r[1], calls)
            seen += 1
    assert seen >= 100


def _flattened(m):
    """True if ``m`` was flattened and bound for status-bar help."""
    binds = m.bind()
    return ('<<MenuSelect>>' in binds and '<Unmap>' in binds
            and str(m.cget('relief')) == 'flat'
            and str(m.cget('activeborderwidth')) == '0')


def test_every_menu_is_flattened_bound_and_the_buttons_themed(ed):
    menus = _menus(ed)
    assert len(menus) == 24
    assert all(_flattened(m) for m in menus.values()), [
        p for p, m in menus.items() if not _flattened(m)]
    pal = current_palette()
    for b in ed._menubar_buttons:
        assert (b.cget('bg'), b.cget('fg')) == (pal['bg'], pal['fg'])
    # One hover tip per button.
    tips = getattr(ed, '_tip_texts', {})
    assert all(tips.get(str(b)) for b in ed._menubar_buttons)


# A call in each part of the build made to raise, and the buttons that
# carry their menu afterwards.
FAILS = [
    ('buttons', None, []),
    ('File', 'Export project archive…', []),
    ('Edit', 'FMO gating…', ['File']),
    ('View', 'Zoom out', ['File', 'Edit']),
    ('Analyze', 'Kinetics…', ['File', 'Edit', 'View']),
    ('Tools', 'Debarcode…', ['File', 'Edit', 'View', 'Analyze']),
    ('Help', 'Keyboard shortcuts',
     ['File', 'Edit', 'View', 'Analyze', 'Tools']),
]


@pytest.mark.parametrize('where, label, attached', FAILS)
def test_a_failure_part_way_leaves_the_earlier_menus(ed, monkeypatch, where,
                                                     label, attached):
    themed = []
    monkeypatch.setattr(type(ed), '_theme_menubar',
                        lambda self, pal=None: themed.append(1))
    if label is None:                 # the tip of the fourth button fails
        real_tip = ed._tip

        def tip(w, text):
            if w.cget('text') == 'Analyze':
                raise ValueError('injected')
            return real_tip(w, text)
        monkeypatch.setattr(ed, '_tip', tip, raising=False)
    else:
        real = tk.Menu.add_command

        def add_command(m, cnf=None, **kw):
            if kw.get('label') == label:
                raise ValueError('injected')
            return real(m, cnf or {}, **kw)
        monkeypatch.setattr(tk.Menu, 'add_command', add_command)
    before_attrs = set(vars(ed))
    seen = []
    monkeypatch.setattr(editor_menu, 'note',
                        lambda what, exc=None: seen.append((what, repr(exc))))
    for b in ed._menubar_buttons:
        b.destroy()
    ed._build_menubar()
    assert seen == [('menu bar not fully built', "ValueError('injected')")]
    assert themed == []
    assert _attached(ed) == attached
    tops = _tops(ed)
    if label is None:
        assert list(tops) == ['File', 'Edit', 'View', 'Analyze']
        assert _all_menu_widgets(ed) == []
    else:
        assert list(tops) == ['File', 'Edit', 'View', 'Analyze', 'Tools',
                              'Help']
        # The failing menu was started and stops at the failing entry;
        # nothing after it was built.
        order = list(tops)
        for t in order[order.index(where) + 1:]:
            assert tops[t].winfo_children() == [], t
        started = [w for w in tops[where].winfo_children()
                   if isinstance(w, tk.Menu)]
        assert len(started) == 1
        labels = [r[1] for m in _all_menu_widgets(ed)
                  if str(m).startswith(str(tops[where]))
                  for r in _rows(ed, m) if r[0] != 'separator']
        assert labels and label not in labels
    # Nothing flattened or bound.
    assert not [m for m in _all_menu_widgets(ed)
                if '<<MenuSelect>>' in m.bind()]
    assert set(vars(ed)) == before_attrs


def test_a_failure_after_the_menus_leaves_them_built(ed, monkeypatch):
    def boom(self, pal=None):
        raise ValueError('injected')
    monkeypatch.setattr(type(ed), '_theme_menubar', boom)
    seen = []
    monkeypatch.setattr(editor_menu, 'note',
                        lambda what, exc=None: seen.append((what, repr(exc))))
    _rebuild(ed)
    assert _attached(ed) == ['File', 'Edit', 'View', 'Analyze', 'Tools',
                             'Help']
    assert all(_flattened(m) for m in _menus(ed).values())
    assert seen == [('menu bar not fully built', "ValueError('injected')")]


def test_the_failure_is_noted_once(ed, monkeypatch):
    """Through errors.note itself: one recorded line."""
    real = tk.Menu.add_command

    def add_command(m, cnf=None, **kw):
        if kw.get('label') == 'Zoom out':
            raise ValueError('injected')
        return real(m, cnf or {}, **kw)
    monkeypatch.setattr(tk.Menu, 'add_command', add_command)
    marker = '[handled] marker before the rebuild'
    errors.note('marker before the rebuild', None)
    _rebuild(ed)
    lines = errors.noted()
    assert lines[lines.index(marker) + 1:] == [
        '[handled] menu bar not fully built: ValueError: injected']


def test_a_rebuild_builds_the_same_menubar_again(ed):
    first = {p: _rows(ed, m) for p, m in _menus(ed).items()
             if p != ('File', 'Open recent')}
    attrs = set(vars(ed))
    for _ in range(2):
        _rebuild(ed)
        menus = _menus(ed)
        assert {p: _rows(ed, m) for p, m in menus.items()
                if p != ('File', 'Open recent')} == first
        assert len(ed._menubar_frame.winfo_children()) == 6
        assert len(ed._menubar_buttons) == 6
        assert all(b.winfo_exists() for b in ed._menubar_buttons)
        assert all(_flattened(m) for m in menus.values())
        for m in menus.values():                 # bound once, not twice
            assert m.bind('<<MenuSelect>>').count('_on_menu_select') == 1
        assert ed._display_menu is menus[('View', 'Display')]
        assert ed._watch_menu is menus[('Tools', 'Files & checks')]
        assert set(vars(ed)) == attrs
