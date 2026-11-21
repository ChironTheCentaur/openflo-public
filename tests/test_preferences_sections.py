"""Preferences dialog, section by section (R6 slice C1).

PreferencesDialog.__init__ was one 500-line builder; it is now one builder
per heading. These tests pin what each section shows and does, on a small
stand-in editor and a temporary prefs file, so a section dropped, moved or
built from the wrong values fails here:

* the order of every heading and control in the dialog body;
* each control writes its preference (and calls the editor where it should);
* every Tk variable outlives its builder: still set in Tcl after a garbage
  collection, still shown by its widget, and its traces still fire (a
  tk.Variable held only by a helper's local is unset when the helper returns,
  silently);
* the <Destroy> handler removes the theme trace it added, only when the
  dialog itself goes;
* the GPU rows use the one probe made while building.
"""
from __future__ import annotations

import gc
import json
import tkinter as tk

import pytest

from tests.conftest import gui_unavailable

# ── stand-ins ────────────────────────────────────────────────────────────────


class _Editor(tk.Toplevel):
    """Just what PreferencesDialog reads from the editor, recording calls."""

    def __init__(self, master, theme='light', legend='cached'):
        super().__init__(master)
        self.withdraw()
        self.calls = []
        self.tips = {}
        self._theme_var = tk.StringVar(self, value=theme)
        self._tooltips_enabled = tk.BooleanVar(self, value=True)
        self._legend_placement = tk.StringVar(self, value=legend)
        self._legend_pos_cache = {'stale': 1}

    def _set_theme(self):
        self.calls.append(('set_theme', self._theme_var.get()))

    def _replot(self):
        self.calls.append(('replot',))

    def _notify_changed(self):
        self.calls.append(('notify',))

    def _tip(self, widget, text):
        self.tips[str(widget)] = text


class _Gpu:
    """A deterministic GPU probe: ``devices`` maps backend -> device name
    (None: that backend has no usable device); ``fail`` makes it raise."""

    def __init__(self, devices, fail=False):
        self.devices = devices
        self.fail = fail
        self.backend = None
        self.calls = []

    def install(self, monkeypatch):
        from openflo import gpu_accel
        for name in ('set_backend', 'gpu_available', 'device_name',
                     'backend_name', 'set_enabled'):
            monkeypatch.setattr(gpu_accel, name, getattr(self, name))

    def set_backend(self, name):
        self.calls.append(('set_backend', name))
        self.backend = name

    def gpu_available(self):
        self.calls.append(('gpu_available',))
        if self.fail:
            raise RuntimeError('probe failed')
        return self.devices.get(self.backend) is not None

    def device_name(self):
        self.calls.append(('device_name',))
        return self.devices.get(self.backend) or None

    def backend_name(self):
        return f'bk-{self.backend}'

    def set_enabled(self, on):
        self.calls.append(('set_enabled', on))
        return bool(on) and self.devices.get(self.backend) is not None


@pytest.fixture
def env(monkeypatch, tmp_path):
    """(root, prefs_file, errors); Tk callback errors are collected."""
    from openflo import prefs
    path = tmp_path / 'prefs.json'
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(path))
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f'Tk cannot start: {exc}')
    root.withdraw()
    errors = []
    root.report_callback_exception = (
        lambda et, ev, tb: errors.append(f'{et.__name__}: {ev}'))
    yield root, path, errors
    try:
        root.destroy()
    except tk.TclError:
        pass


def _build(env, monkeypatch, stored=None, gpu=None, **editor_kw):
    root, path, _errors = env
    path.write_text(json.dumps(stored or {}), encoding='utf-8')
    (gpu or _Gpu({'auto': 'Dev-1', 'torch': 'Dev-2'})).install(monkeypatch)
    from openflo.ui_preferences import PreferencesDialog
    ed = _Editor(root, **editor_kw)
    dlg = PreferencesDialog(ed)
    root.update()
    return ed, dlg


def _prefs(env):
    return json.loads(env[1].read_text(encoding='utf-8'))


def _body(dlg):
    """The scrolled frame the sections are packed into."""
    return dlg.density_binned_cb.master


def _describe(w):
    c = w.winfo_class()
    if c == 'TLabel':
        t = str(w.cget('text'))
        if str(w.cget('font')) == 'OpenFloHeading':
            return ('heading', t)
        return ('label', t[:30])
    if c == 'TFrame':
        return ('row', [_describe(x) for x in w.pack_slaves()])
    if c == 'TCheckbutton':
        return ('check', str(w.cget('text')))
    if c == 'TButton':
        return ('button', str(w.cget('text')))
    return (c,)


def _rows(dlg):
    """{label text: the combobox in that row} for every label + combo row."""
    out = {}
    for w in _body(dlg).pack_slaves():
        if w.winfo_class() == 'TFrame':
            kids = w.pack_slaves()
            for a, b in zip(kids, kids[1:], strict=False):
                if a.winfo_class() == 'TLabel' and b.winfo_class() == 'TCombobox':
                    out[str(a.cget('text'))] = b
    return out


def _checks(dlg):
    return {str(w.cget('text')): w for w in _body(dlg).pack_slaves()
            if w.winfo_class() == 'TCheckbutton'}


def _select(root, combo, value):
    combo.set(value)
    combo.event_generate('<<ComboboxSelected>>')
    root.update()


# ── the layout, in order ────────────────────────────────────────────────────

EXPECTED_BODY = [
    ('heading', 'Appearance'),
    ('row', [('label', 'Windows:'), ('TCombobox',), ('label', 'Plot background:'),
             ('TCombobox',)]),
    ('row', [('label', 'Text size:'), ('TCombobox',)]),
    ('check', 'Color-blind safe colors (gates, backgating, density)'),
    ('check', 'High contrast (stronger text, borders and gate lines)'),
    ('check', 'Show hover tips & menu help'),
    ('row', [('label', 'Legend position:'), ('TCombobox',)]),
    ('label', 'Automatic placement avoids cov'),
    ('heading', 'Export'),
    ('check', 'Stamp OpenFlo version on exported figures'),
    ('check', 'Keep text editable in PDF and SVG'),
    ('row', [('label', 'CSV tables:'), ('TCombobox',)]),
    ('row', [('label', 'FCS export channel names:'), ('TCombobox',)]),
    ('row', [('label', 'Export resolution:'), ('TCombobox',)]),
    ('row', [('label', 'Figure font:'), ('TCombobox',)]),
    ('check', 'Layout: drop plots as the current graph is shown'),
    ('heading', 'Statistics'),
    ('row', [('label', 'GeoMean:'), ('TCombobox',)]),
    ('row', [('label', 'rCV and rSD:'), ('TCombobox',)]),
    ('check', 'Westgard R-4s: literature form only (within run)'),
    ('heading', 'Pipeline workspace'),
    ('row', [('label', 'Run outputs:'), ('TEntry',), ('button', 'Choose…'),
             ('button', 'Default')]),
    ('heading', 'Loading'),
    ('check', 'Run acquisition QC on load (drift, flow rate, margin events)'),
    ('heading', 'Updates'),
    ('check', 'Check for updates when OpenFlo starts'),
    ('heading', 'Performance'),
    ('row', [('label', 'Concurrent file loaders:'), ('TCombobox',)]),
    ('label', 'How many files load in paralle'),
    ('check', 'Faster Gaussian KDE density (binned, approximate)'),
    ('check', 'Use GPU acceleration'),
    ('row', [('label', 'GPU backend:'), ('TCombobox',)]),
    ('label', 'resolved: bk-auto — Dev-1'),
    ('label', 'Offloads compensation + arcsin'),
    ('label', 'Window size/position and recen'),
]


def test_every_section_in_order(env, monkeypatch):
    _ed, dlg = _build(env, monkeypatch)
    assert [_describe(w) for w in _body(dlg).pack_slaves()] == EXPECTED_BODY
    assert dlg.title() == 'Preferences'
    # The Close row is packed before the body, at the bottom.
    first = dlg.pack_slaves()[0]
    assert [str(b.cget('text')) for b in first.pack_slaves()] == ['Close']
    assert not env[2]


# ── Appearance ──────────────────────────────────────────────────────────────

def test_appearance_theme_legend_and_tips(env, monkeypatch):
    root = env[0]
    ed, dlg = _build(env, monkeypatch, theme='dark', legend='fixed')
    box, pbox = dlg.theme_combos
    assert (box.get(), pbox.get()) == ('Dark', 'White')
    _select(root, pbox, 'Black')
    assert ed._theme_var.get() == 'midnight'
    assert ed.calls[-1] == ('set_theme', 'midnight')
    rows = _rows(dlg)
    legend = rows['Legend position:']
    assert legend.get() == 'Always top-right'
    _select(root, legend, 'Automatic (recompute each time)')
    assert ed._legend_placement.get() == 'best'
    assert ed._legend_pos_cache == {}
    assert ed.calls[-1] == ('replot',)
    assert _prefs(env)['legend_placement'] == 'best'
    _checks(dlg)['Show hover tips & menu help'].invoke()
    assert _prefs(env)['tooltips'] is False
    assert ed.tips[str(box)].startswith('Color of the windows')
    assert not env[2]


def test_theme_follow_trace_fires_after_build_and_gc(env, monkeypatch):
    root = env[0]
    ed, dlg = _build(env, monkeypatch)
    gc.collect()
    ed._theme_var.set('midnight')        # picked from the View menu
    root.update()
    box, pbox = dlg.theme_combos
    assert (box.get(), pbox.get()) == ('Dark', 'Black')
    assert not env[2]


def test_destroy_handler_removes_only_its_trace(env, monkeypatch):
    root = env[0]
    ed, dlg = _build(env, monkeypatch)
    name = str(ed._theme_var)

    def traces():
        return root.tk.splitlist(root.tk.call('trace', 'info', 'variable', name))
    assert len(traces()) == 1
    # A child going does not remove it: only the dialog itself.
    dlg.theme_combos[0].destroy()
    root.update()
    assert len(traces()) == 1
    dlg.destroy()
    root.update()
    assert traces() == ()
    ed._theme_var.set('dark')            # no trace into a dead dialog
    root.update()
    assert not env[2]


# ── Export ──────────────────────────────────────────────────────────────────

def test_export_controls_write_their_preferences(env, monkeypatch):
    root = env[0]
    ed, dlg = _build(env, monkeypatch, stored={'export_dpi': 600,
                                               'figure_font': 'No-Such-Font'})
    checks = _checks(dlg)
    checks['Stamp OpenFlo version on exported figures'].invoke()
    checks['Keep text editable in PDF and SVG'].invoke()
    checks['Layout: drop plots as the current graph is shown'].invoke()
    rows = _rows(dlg)
    assert rows['Export resolution:'].get() == '600 dpi'
    assert rows['Figure font:'].get() == 'No-Such-Font'
    _select(root, rows['CSV tables:'], rows['CSV tables:'].cget('values')[1])
    _select(root, rows['FCS export channel names:'], "Each file's original names")
    _select(root, rows['Export resolution:'], '1200 dpi')
    _select(root, rows['Figure font:'], 'Default (DejaVu Sans)')
    p = _prefs(env)
    assert p['export_provenance'] is False
    assert p['export_editable_text'] is False
    assert p['layout_drop_view'] == 'current'
    assert p['csv_style'] == 'system'
    assert p['fcs_export_channel_names'] == 'original'
    assert p['export_dpi'] == 1200
    assert p['figure_font'] == ''
    assert ed.calls[-1] == ('replot',)
    assert dlg.fcs_names_combo is rows['FCS export channel names:']
    assert not env[2]


# ── Statistics ──────────────────────────────────────────────────────────────

def test_statistics_forms_and_westgard(env, monkeypatch):
    root = env[0]
    ed, dlg = _build(env, monkeypatch, stored={'stat_rcv': 'percentile'})
    stat = sorted(k for k in vars(dlg) if k.startswith('_stat_'))
    assert stat == ['_stat_geomean_var', '_stat_rcv_var']
    rows = _rows(dlg)
    geo, rcv = rows['GeoMean:'], rows['rCV and rSD:']
    assert str(geo.cget('textvariable')) == str(dlg._stat_geomean_var)
    assert str(rcv.cget('textvariable')) == str(dlg._stat_rcv_var)
    assert rcv.get().startswith('Half the 15.87')
    _select(root, geo, geo.cget('values')[1])
    assert _prefs(env)['stat_geomean'] == 'log'
    assert ed.calls[-1] == ('notify',)
    _select(root, rcv, rcv.cget('values')[0])
    assert _prefs(env)['stat_rcv'] == 'mad'
    assert ed.calls.count(('notify',)) == 2
    dlg.r4s_literature_box.invoke()
    assert _prefs(env)['westgard_r4s_literature'] is True
    assert not env[2]


# ── Pipeline workspace, Loading, Performance ────────────────────────────────

def test_pipeline_loading_and_performance(env, monkeypatch, tmp_path):
    root = env[0]
    from tkinter import filedialog
    pick = tmp_path / 'runs'
    pick.mkdir()
    monkeypatch.setattr(filedialog, 'askdirectory', lambda **k: str(pick))
    ed, dlg = _build(env, monkeypatch, stored={'load_workers': 3})
    box, choose, reset = dlg.pipeline_output_controls
    choose.invoke()
    assert _prefs(env)['pipeline_output_dir'] == str(pick)
    assert box.get() == str(pick)
    reset.invoke()
    assert box.get() == dlg.PIPELINE_OUTPUT_DEFAULT
    _checks(dlg)['Run acquisition QC on load (drift, flow rate, margin '
                 'events)'].invoke()
    assert _prefs(env)['load_qc'] is True
    upd = _checks(dlg)['Check for updates when OpenFlo starts']
    assert 'selected' in upd.state()                # on by default
    upd.invoke()
    assert _prefs(env)['check_updates_on_start'] is False
    workers = _rows(dlg)['Concurrent file loaders:']
    assert workers.get() == '3'
    _select(root, workers, workers.cget('values')[0])      # Auto (n)
    assert _prefs(env)['load_workers'] is None
    _select(root, workers, '5')
    assert _prefs(env)['load_workers'] == 5
    dlg.density_binned_cb.invoke()
    assert _prefs(env)['density_binned'] is True
    assert ed.calls[-1] == ('replot',)
    assert not env[2]


# ── every variable outlives its builder ─────────────────────────────────────

def _bound_widgets(dlg):
    """{Tcl variable name: widget} for every widget showing a variable."""
    out = {}
    stack = [dlg]
    while stack:
        w = stack.pop()
        stack.extend(w.winfo_children())
        for opt in ('textvariable', 'variable'):
            try:
                name = str(w.cget(opt))
            except tk.TclError:
                continue
            if name:
                out[name] = w
    return out


def test_updates_box_shows_a_saved_off(env, monkeypatch):
    _ed, dlg = _build(env, monkeypatch,
                     stored={'check_updates_on_start': False})
    upd = _checks(dlg)['Check for updates when OpenFlo starts']
    assert 'selected' not in upd.state()
    assert dlg._upd_var.get() is False
    upd.invoke()
    assert 'selected' in upd.state()
    assert _prefs(env)['check_updates_on_start'] is True


def test_every_variable_is_kept_and_shown(env, monkeypatch):
    root = env[0]
    ed, dlg = _build(env, monkeypatch)
    gc.collect()
    tkvars = {k: v for k, v in vars(dlg).items() if isinstance(v, tk.Variable)}
    assert sorted(tkvars) == sorted([
        '_chrome_name', '_plot_name', '_text_size_var',
        '_safe_colors_var', '_high_contrast_var', '_legend_var', '_prov_var',
        '_editable_var', '_csv_var', '_fcs_names_var', '_dpi_var',
        '_font_var', '_layout_view_var', '_stat_geomean_var',
        '_stat_rcv_var', '_r4s_literature_var', '_pipeline_out_var',
        '_qc_var', '_workers_var', '_density_binned_var', '_gpu_var',
        '_backend_var', '_upd_var'])
    bound = _bound_widgets(dlg)
    for key, var in tkvars.items():
        name = str(var)
        assert root.tk.call('info', 'exists', name), key
        assert name in bound, key
        w = bound[name]
        if isinstance(var, tk.BooleanVar):
            var.set(True)
            assert 'selected' in w.state(), key
            var.set(False)
            assert 'selected' not in w.state(), key
        else:
            var.set('shown?')
            assert w.get() == 'shown?', key
    # Every variable a widget shows (but the editor's own) belongs to the
    # dialog, so none is held by a builder's local alone.
    own = {str(v) for v in tkvars.values()}
    editor = {str(ed._tooltips_enabled)}
    assert set(bound) - own == editor
    assert not env[2]


# ── GPU: the rows use the probe made while building ─────────────────────────

def _gpu_rows(dlg):
    body = _body(dlg).pack_slaves()
    i = body.index(_checks(dlg)['Use GPU acceleration'])
    cb, brow, status, note = body[i:i + 4]
    return cb, brow.pack_slaves()[1], status, note


def test_gpu_with_a_device(env, monkeypatch):
    root = env[0]
    gpu = _Gpu({'auto': 'Dev-1', 'torch': 'Dev-2'})
    ed, dlg = _build(env, monkeypatch, stored={'use_gpu': True}, gpu=gpu)
    cb, backend, status, note = _gpu_rows(dlg)
    assert gpu.calls[:3] == [('set_backend', 'auto'), ('gpu_available',),
                             ('device_name',)]
    assert gpu.calls.count(('device_name',)) == 1      # one probe
    assert 'disabled' not in cb.state() and dlg._gpu_var.get() is True
    assert status.cget('text') == 'resolved: bk-auto — Dev-1'
    assert 'Detected: Dev-1.' in note.cget('text')
    assert ed.tips[str(cb)]().endswith('Detected: Dev-1.')
    # A backend without a device turns the box off and greys it ...
    _select(root, backend, 'CuPy (NVIDIA)')
    assert status.cget('text') == 'no usable device for that backend'
    assert 'disabled' in cb.state() and dlg._gpu_var.get() is False
    assert _prefs(env)['gpu_backend'] == 'cupy'
    # ... and one with a device enables it again.
    _select(root, backend, 'PyTorch (NVIDIA/AMD/Intel/Apple)')
    assert status.cget('text') == 'resolved: bk-torch — Dev-2'
    assert 'disabled' not in cb.state()
    cb.invoke()
    assert _prefs(env)['use_gpu'] is True
    assert ('set_enabled', True) in gpu.calls
    assert not env[2]


def test_gpu_probe_ok_without_a_device_name(env, monkeypatch):
    gpu = _Gpu({'auto': ''})
    gpu.gpu_available = lambda: True                    # usable, no label
    _ed, dlg = _build(env, monkeypatch, gpu=gpu)
    cb, _backend, status, note = _gpu_rows(dlg)
    assert 'disabled' not in cb.state()
    assert status.cget('text') == 'resolved: bk-auto — None'
    assert note.cget('text').startswith(
        'Offloads compensation + arcsinh to the GPU (~10x on large samples). GPU')
    assert not env[2]


@pytest.mark.parametrize('fail', [False, True])
def test_gpu_without_a_device(env, monkeypatch, fail):
    from openflo import ui_preferences
    notes = []
    monkeypatch.setattr(ui_preferences, 'note', lambda *a: notes.append(a[0]))
    gpu = _Gpu({}, fail=fail)
    ed, dlg = _build(env, monkeypatch, stored={'use_gpu': True,
                                               'gpu_backend': 'torch'}, gpu=gpu)
    cb, backend, status, note = _gpu_rows(dlg)
    assert gpu.calls[0] == ('set_backend', 'torch')
    assert notes == (['detecting a GPU backend'] if fail else [])
    assert 'disabled' in cb.state() and dlg._gpu_var.get() is False
    assert status.cget('text') == ''
    assert note.cget('text').startswith('No GPU backend detected.')
    assert ed.tips[str(cb)]().endswith('no usable GPU device was detected.')
    assert backend.get() == 'PyTorch (NVIDIA/AMD/Intel/Apple)'
    assert not env[2]


def test_every_control_has_its_tip(env, monkeypatch):
    ed, dlg = _build(env, monkeypatch)
    controls = []
    stack = [_body(dlg)]
    while stack:
        w = stack.pop(0)
        stack.extend(w.winfo_children())
        if w.winfo_class() in ('TCombobox', 'TCheckbutton', 'TEntry', 'TButton'):
            controls.append(w)
    assert len(controls) == 26
    assert [str(w) for w in controls if str(w) not in ed.tips] == []
    assert len(ed.tips) == 26                 # Close has none
