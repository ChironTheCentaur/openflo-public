"""The steps of PlotMixin._replot_draw on a recording stand-in editor: the
early stops (nothing loaded, nothing selected, no x channel), the mode
dispatch and its 'Plot error', the two overlays that may fail on their
own, the axis labels and scale anchors (the histogram's binned array when
there is one) and the order of the finishing steps.

The stand-in records each call; no Tk and no real data. Generated sample
and channel names only.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import openflo.editor_plot as ep

STEPS = ['_hide_empty_overlay', '_sync_display_mode_availability',
         '_sync_color_combo', '_render_placeholder', '_apply_plot_theme',
         '_plot_histogram', '_plot_dot', '_plot_pseudocolor', '_plot_contour',
         '_overlay_removed_events', '_overlay_backgate', '_apply_axis_to_ax',
         '_draw_highlight_overlays', '_draw_gates', '_refresh_gate_list',
         '_apply_kept_zoom', '_draw_gate_labels', '_capture_legend_pos',
         '_activate_gate_tool', '_sync_slider_panel']


class _Recorder:
    """Any attribute is a recording call (the axes, figure and canvas)."""

    def __init__(self, log, name, returns=None):
        self._log, self._name, self._ret = log, name, returns or {}

    def __getattr__(self, attr):
        def call(*a, **k):
            self._log.append((f'{self._name}.{attr}', a, k))
            return self._ret.get(attr)
        return call


class _Var:
    def __init__(self, v):
        self._v = v

    def get(self):
        return self._v


def _editor(samples=None, selected=None, mode='dot', x='FL1-A', y='FL2-A',
            raises=(), cbar=None, color='', **attrs):
    log = []

    def step(name):
        def call(self, *a, **k):
            log.append((name, a, k))
            if name in raises:
                raise ValueError(f'{name} broke')
            return ['label'] if name == '_draw_gate_labels' else None
        return call

    ns = {n: step(n) for n in STEPS}
    ns['_selected_samples'] = lambda self: list(selected or [])
    ns['_fmt_channel'] = lambda self, c: f'<{c}>'
    cls = type('Ed', (ep.PlotMixin,), ns)
    ed = cls.__new__(cls)
    ed.log = log
    ed._samples = samples or {}
    ed._cbar = cbar
    ed._vlines, ed._hlines = {'g': 1}, {'g': 2}
    ed.ax = _Recorder(log, 'ax', {'get_xlim': (0, 1), 'get_ylim': (2, 3)})
    ed.fig = _Recorder(log, 'fig')
    ed.canvas = _Recorder(log, 'canvas')
    ed.mode_var, ed.x_combo = _Var(mode), _Var(x)
    ed.y_combo, ed.color_combo = _Var(y), _Var(color)
    for k, v in attrs.items():
        setattr(ed, k, v)
    return ed


class _Frame(dict):
    """Enough of a DataFrame: ``columns`` and ``frame[c].values``."""

    @property
    def columns(self):
        return list(self)


def _loaded(**cols):
    return SimpleNamespace(data=_Frame(
        {k: SimpleNamespace(values=np.asarray(v)) for k, v in cols.items()}))


def _names(ed):
    return [e[0] for e in ed.log]


START = ['_hide_empty_overlay', '_sync_display_mode_availability',
         '_sync_color_combo']
FINISH = ['_draw_highlight_overlays', '_draw_gates', '_refresh_gate_list',
          'fig.tight_layout', 'ax.get_xlim', 'ax.get_ylim',
          '_apply_kept_zoom', '_draw_gate_labels', '_apply_plot_theme',
          'canvas.draw_idle', '_capture_legend_pos', '_activate_gate_tool',
          '_sync_slider_panel']


def test_a_full_replot_runs_every_step_in_order():
    ed = _editor({'s1': _loaded(**{'FL1-A': [1.0], 'FL2-A': [2.0]})},
                 ['s1'], cbar=SimpleNamespace(remove=lambda: None))
    ed._replot_draw()
    assert _names(ed) == START + ['ax.clear', '_plot_dot',
                                  '_overlay_removed_events', '_overlay_backgate',
                                  'ax.set_xlabel', 'ax.set_ylabel',
                                  '_apply_axis_to_ax', '_apply_axis_to_ax',
                                  *FINISH]
    assert ed._cbar is None and ed._vlines == {} and ed._hlines == {}
    assert ed._home_view == ((0, 1), (2, 3))
    assert ed._gate_label_artists == ['label']
    gates = [e for e in ed.log if e[0] == '_draw_gates'][0]
    assert gates[1] == ('FL1-A', 'FL2-A')


def test_a_colorbar_that_will_not_go_is_still_forgotten():
    def broken():
        raise RuntimeError('gone')
    ed = _editor({}, [], cbar=SimpleNamespace(remove=broken))
    ed._replot_draw()
    assert ed._cbar is None


def test_nothing_loaded_draws_the_placeholder_only():
    ed = _editor({}, [])
    ed._replot_draw()
    assert _names(ed) == START + ['ax.clear', '_render_placeholder']


def test_nothing_selected_says_so():
    ed = _editor({'s1': _loaded()}, [])
    ed._replot_draw()
    assert _names(ed) == START + ['ax.clear', 'ax.text', 'ax.set_xticks',
                                  'ax.set_yticks', '_apply_plot_theme',
                                  'canvas.draw_idle']
    text = [e for e in ed.log if e[0] == 'ax.text'][0]
    assert text[1][2] == 'Select one or more samples on the left'


def test_no_x_channel_stops_after_theming():
    ed = _editor({'s1': _loaded()}, ['s1'], x='')
    ed._replot_draw()
    assert _names(ed) == START + ['ax.clear', '_apply_plot_theme',
                                  'canvas.draw_idle']


@pytest.mark.parametrize('mode, call, args', [
    ('histogram', '_plot_histogram', (['s1'], 'FL1-A')),
    ('dot', '_plot_dot', (['s1'], 'FL1-A', 'FL2-A', 'FL3-A')),
    ('pseudocolor', '_plot_pseudocolor', (['s1'], 'FL1-A', 'FL2-A')),
    ('contour', '_plot_contour', (['s1'], 'FL1-A', 'FL2-A')),
    ('other', None, None),
])
def test_each_mode_draws_its_own_plot(mode, call, args):
    ed = _editor({'s1': _loaded()}, ['s1'], mode=mode, color='FL3-A')
    ed._replot_draw()
    plots = [e for e in ed.log if e[0].startswith('_plot_')]
    assert plots == ([] if call is None else [(call, args, {})])


def test_a_failing_plot_shows_plot_error_and_the_replot_goes_on():
    ed = _editor({'s1': _loaded()}, ['s1'], mode='contour',
                 raises={'_plot_contour'})
    ed._replot_draw()
    err = [e for e in ed.log if e[0] == 'ax.text'][0]
    assert err[1][2] == 'Plot error:\n_plot_contour broke'
    assert err[2]['color'] == 'red'
    assert _names(ed)[-len(FINISH):] == FINISH


def test_a_failing_overlay_is_noted_and_the_other_still_drawn(monkeypatch):
    noted = []
    monkeypatch.setattr(ep, 'note', lambda what, exc: noted.append(what))
    ed = _editor({'s1': _loaded()}, ['s1'],
                 raises={'_overlay_removed_events', '_overlay_backgate'})
    ed._replot_draw()
    assert noted == ['cleaned-out overlay not drawn',
                     'backgate overlay not drawn']
    assert _names(ed)[-len(FINISH):] == FINISH


def test_scales_anchor_on_the_first_samples_data():
    ed = _editor({'s1': _loaded(**{'FL1-A': [1.0, 2.0], 'FL2-A': [5.0]})},
                 ['s1', 's2'])
    ed._replot_draw()
    axes = [e[1] for e in ed.log if e[0] == '_apply_axis_to_ax']
    assert [(a[0], a[1], list(a[2])) for a in axes] == [
        ('FL1-A', 'x', [1.0, 2.0]), ('FL2-A', 'y', [5.0])]


def test_a_missing_channel_or_unloaded_first_sample_gives_no_anchor():
    ed = _editor({'s1': _loaded(**{'FL2-A': [5.0]})}, ['s1'])
    ed._replot_draw()
    axes = [e[1] for e in ed.log if e[0] == '_apply_axis_to_ax']
    assert axes[0] == ('FL1-A', 'x', None)
    ed = _editor({'s1': _loaded(**{'FL1-A': [1.0], 'FL2-A': [5.0]})},
                 ['gone', 's1'])
    ed._replot_draw()
    axes = [e[1] for e in ed.log if e[0] == '_apply_axis_to_ax']
    assert axes == [('FL1-A', 'x', None), ('FL2-A', 'y', None)]


def test_a_histogram_anchors_x_on_the_array_it_binned():
    binned = np.array([7.0, 8.0])
    ed = _editor({'s1': _loaded(**{'FL1-A': [1.0], 'FL2-A': [5.0]})},
                 ['s1'], mode='histogram', _hist_x_anchor=binned)
    ed._replot_draw()
    assert _names(ed) == START + ['ax.clear', '_plot_histogram',
                                  '_overlay_removed_events', '_overlay_backgate',
                                  'ax.set_xlabel', '_apply_axis_to_ax',
                                  *FINISH]
    (axis,) = [e[1] for e in ed.log if e[0] == '_apply_axis_to_ax']
    assert axis[0:2] == ('FL1-A', 'x') and axis[2] is binned
    # A histogram has no y channel: the overlays, gates and labels get None.
    for name in ('_draw_highlight_overlays', '_draw_gates', '_draw_gate_labels'):
        (call,) = [e for e in ed.log if e[0] == name]
        assert call[1][-1] is None


def test_a_leftover_histogram_array_does_not_anchor_a_dot_plot():
    ed = _editor({'s1': _loaded(**{'FL1-A': [1.0], 'FL2-A': [5.0]})},
                 ['s1'], mode='dot', _hist_x_anchor=np.array([7.0, 8.0]))
    ed._replot_draw()
    axes = [e[1] for e in ed.log if e[0] == '_apply_axis_to_ax']
    assert [list(a[2]) for a in axes] == [[1.0], [5.0]]


def test_a_histogram_without_its_array_anchors_on_the_sample():
    ed = _editor({'s1': _loaded(**{'FL1-A': [1.0]})}, ['s1'],
                 mode='histogram', _hist_x_anchor=None)
    ed._replot_draw()
    (axis,) = [e[1] for e in ed.log if e[0] == '_apply_axis_to_ax']
    assert list(axis[2]) == [1.0]


def test_no_y_channel_sets_no_y_label_or_scale():
    ed = _editor({'s1': _loaded(**{'FL1-A': [1.0]})}, ['s1'], y='')
    ed._replot_draw()
    assert 'ax.set_ylabel' not in _names(ed)
    assert [e[1][1] for e in ed.log if e[0] == '_apply_axis_to_ax'] == ['x']
    labels = [e for e in ed.log if e[0] == 'ax.set_xlabel'][0]
    assert labels[1] == ('<FL1-A>',) and labels[2] == {'fontsize': 9}


def test_overlays_get_the_samples_both_channels_and_the_mode():
    ed = _editor({'s1': _loaded()}, ['s1'], mode='pseudocolor')
    ed._replot_draw()
    (removed,) = [e for e in ed.log if e[0] == '_overlay_removed_events']
    (backgate,) = [e for e in ed.log if e[0] == '_overlay_backgate']
    assert removed[1] == (['s1'], 'FL1-A', 'FL2-A', 'pseudocolor')
    assert backgate[1] == (['s1'], 'FL1-A', 'FL2-A')


def test_labelled_channel_choices_reach_every_step_as_detector_names():
    ed = _editor({'s1': _loaded(**{'FL1-A': [1.0], 'FL2-A': [5.0]})},
                 ['s1'], x='Label one (FL1-A)', y='Label two (FL2-A)')
    ed._replot_draw()
    (plot,) = [e for e in ed.log if e[0] == '_plot_dot']
    assert plot[1][1:3] == ('FL1-A', 'FL2-A')
    (removed,) = [e for e in ed.log if e[0] == '_overlay_removed_events']
    assert removed[1][1:3] == ('FL1-A', 'FL2-A')
    axes = [e[1][:2] for e in ed.log if e[0] == '_apply_axis_to_ax']
    assert axes == [('FL1-A', 'x'), ('FL2-A', 'y')]
    (gates,) = [e for e in ed.log if e[0] == '_draw_gates']
    assert gates[1] == ('FL1-A', 'FL2-A')
    (ylabel,) = [e for e in ed.log if e[0] == 'ax.set_ylabel']
    assert ylabel[1] == ('<FL2-A>',) and ylabel[2] == {'fontsize': 9}


AXES_FRAME = object()     # stands in for ax.transAxes


def test_the_select_note_is_centred_in_the_plot_colour(monkeypatch):
    monkeypatch.setattr(ep, 'current_palette',
                        lambda: {'plot_fg': '#010203', 'fg': '#040506'})
    ed = _editor({'s1': _loaded()}, [])
    ed.ax.transAxes = AXES_FRAME
    ed._replot_draw()
    (text,) = [e for e in ed.log if e[0] == 'ax.text']
    assert text[1][:2] == (0.5, 0.5)
    assert text[2] == {'ha': 'center', 'va': 'center',
                       'transform': AXES_FRAME, 'fontsize': 11,
                       'color': '#010203'}


def test_a_plot_error_is_centred_in_red():
    ed = _editor({'s1': _loaded()}, ['s1'], mode='dot',
                 raises={'_plot_dot'})
    ed.ax.transAxes = AXES_FRAME
    ed._replot_draw()
    (text,) = [e for e in ed.log if e[0] == 'ax.text']
    assert text[1][:2] == (0.5, 0.5)
    assert text[2] == {'ha': 'center', 'va': 'center',
                       'transform': AXES_FRAME, 'fontsize': 10,
                       'color': 'red'}


def test_a_y_channel_missing_from_the_first_sample_gives_no_y_anchor():
    ed = _editor({'s1': _loaded(**{'FL1-A': [1.0]})}, ['s1'])
    ed._replot_draw()
    axes = [e[1] for e in ed.log if e[0] == '_apply_axis_to_ax']
    assert [(a[0], a[1]) for a in axes] == [('FL1-A', 'x'), ('FL2-A', 'y')]
    assert list(axes[0][2]) == [1.0] and axes[1][2] is None
