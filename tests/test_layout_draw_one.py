"""One layout plot panel (``layout_render._draw_one``) on stand-ins.

The layout tests that drive the real editor check the picture; these pin
what the panel itself decides, on generated events and generic channels:
the note when nothing can be drawn, which layers and gates go to the
editor's renderer and with which colour, the title, each gate's
name-and-%Parent label, and the outside-the-zoom fraction kept for the
Zoom to fit offer. No Tk: a real matplotlib figure, a stand-in editor that
records what it is asked to draw, and stand-in sample contexts."""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from openflo import layout_render as lr  # noqa: E402

X, Y = 'FL1-A', 'FL2-A'
BOX = (0.0, 0.0, 1.0, 1.0)


class Ctx:
    """A loaded sample: its gates, population names, unknown reasons and
    stats rows. ``frames`` maps a population to its events."""

    def __init__(self, sample, gates=None, names=None, unknown=None,
                 rows=None, frames=None):
        self.sample = sample
        self.gates = gates or {}
        self._names = names or {}
        self._unknown = unknown or {}
        self._rows = rows or {}
        self.frames = frames or {}
        self.rows_asked = []

    def unknown(self, gid):
        return self._unknown.get(gid, '')

    def name(self, gid):
        return self._names.get(gid, f'pop {gid}')

    def rows(self, stats):
        self.rows_asked.append(set(stats))
        return self._rows


class Ed:
    """Records each ``_render_into`` call; ``_get_df`` serves the frame of
    the population ``_render_population`` points at."""

    def __init__(self, texts=0):
        self._channel_scale: dict = {}
        self._channel_range: dict = {}
        self.calls: list = []
        self.df_calls: list = []
        self.texts = texts
        self.ax = None
        self.ctxs: dict = {}                    # sample -> Ctx (set by Rc)

    def _get_df(self, sample, x, y, for_hist=False):
        self.df_calls.append((sample, self._render_population, for_hist))
        return self.ctxs[sample].frames.get(self._render_population)

    def _render_into(self, ax, samples, x, y, mode, color, **kw):
        self.calls.append(dict(samples=samples, x=x, y=y, mode=mode,
                               color=color, **kw))
        ax.set_xlim(0, 100)                     # as a plot of the events
        ax.set_ylim(0, 100)
        for i in range(self.texts):             # the editor's own notes
            ax.text(0.5, 0.5, f'note {i}')
        if kw.get('painter') is not None:
            self.ax = ax
            kw['painter']()


class Rc:
    def __init__(self, ed, layers, missing=0, page_sample='S1'):
        self.ed = ed
        self.page_sample = page_sample
        self.info: dict = {}
        self._layers = layers
        self._missing = missing
        ed.ctxs.update({c.sample: c for c, _g in layers})

    def layers(self, item):
        return list(self._layers), self._missing

    def layer_colors(self, item, layers):
        return ['#d62728', '#1f77b4', '#2ca02c'][:len(layers)]


def _frame(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({X: rng.uniform(lo, hi, n), Y: rng.uniform(lo, hi, n)})


def _draw(rc, item, iid=None):
    fig = Figure(figsize=(4, 4))
    return lr._draw_one(rc, fig, BOX, item, iid)


def _texts(ax):
    return [t.get_text() for t in ax.texts]


# ── nothing to draw ─────────────────────────────────────────────────────

def test_a_sample_without_the_population_gets_a_note_not_a_plot():
    ed = Ed()
    ax, meta = _draw(Rc(ed, [], missing=2), {'x': X, 'y': Y})
    assert meta is None and not ed.calls
    assert _texts(ax) == ['This sample has no such population']
    assert not ax.axison


def test_a_sample_that_is_not_loaded_is_named_in_the_note():
    ed = Ed()
    ax, meta = _draw(Rc(ed, [], missing=0, page_sample='tube 7'), {'x': X})
    assert meta is None and not ed.calls
    assert _texts(ax) == ['Sample not loaded:\ntube 7']


def test_a_population_that_cannot_be_computed_reads_na_with_the_first_reason():
    a = Ctx('A', unknown={'p': 'needs FL3-A'})
    b = Ctx('B', unknown={'p': 'a ratio gate'})
    ed = Ed()
    ax, meta = _draw(Rc(ed, [(a, 'p'), (b, 'p')]), {'x': X, 'y': Y})
    assert meta is None and not ed.calls
    assert _texts(ax) == ['NA: this population cannot be computed\n(needs FL3-A)']


def test_only_the_computable_layers_are_drawn():
    a = Ctx('A', unknown={'p': 'needs FL3-A'})
    b = Ctx('B')                                    # unknown() is '' here
    ed = Ed()
    _ax, meta = _draw(Rc(ed, [(a, 'p'), (b, 'p')]), {'x': X, 'y': Y})
    assert meta['layers'] == [(b, 'p')]
    assert ed.calls[0]['samples'] == ['B']


# ── what goes to the editor's renderer ──────────────────────────────────

GATES = {'kid': {'kind': 'rect', 'parent_id': 'p', 'x_channel': X,
                 'y_channel': Y, 'x0': 10.0, 'x1': 30.0, 'y0': 60.0,
                 'y1': 80.0, 'color': '#123456'},
         'kid2': {'kind': 'rect', 'parent_id': 'p', 'x_channel': X,
                  'y_channel': Y, 'x0': 50.0, 'x1': 70.0, 'y0': 0.0,
                  'y1': 20.0},
         'top': {'kind': 'rect', 'parent_id': None, 'x_channel': X,
                 'y_channel': Y, 'x0': 0.0, 'x1': 90.0, 'y0': 0.0,
                 'y1': 90.0}}


def test_one_population_draws_its_child_gates_in_its_own_colour_mode():
    ctx = Ctx('A', gates=dict(GATES))
    ed = Ed()
    _draw(Rc(ed, [(ctx, 'p')]), {'x': X, 'y': Y, 'mode': 'contour',
                                 'color': 'Red'})
    call, = ed.calls
    assert call == {'samples': ['A'], 'x': X, 'y': Y, 'mode': 'contour',
                    'color': 'Red', 'draw_gates': True, 'draw_overlays': False,
                    'population': 'p', 'painter': None,
                    'gates': {'kid': GATES['kid'], 'kid2': GATES['kid2']}}


def test_the_root_draws_its_top_gates_and_draw_gates_off_draws_none():
    ctx = Ctx('A', gates=dict(GATES))
    ed = Ed()
    _draw(Rc(ed, [(ctx, None)]), {'x': X, 'y': Y, 'color': None})
    assert ed.calls[-1]['gates'] == {'top': GATES['top']}
    assert ed.calls[-1]['color'] == 'By density'
    assert ed.calls[-1]['mode'] == 'dot'
    _draw(Rc(ed, [(ctx, 'p')]), {'x': X, 'y': Y, 'draw_gates': False})
    assert ed.calls[-1]['gates'] == {}
    assert ed.calls[-1]['draw_gates'] is False


def test_an_exploded_gate_draws_only_its_own_outline_and_skips_stale_ids():
    ctx = Ctx('A', gates=dict(GATES))
    ed = Ed()
    _draw(Rc(ed, [(ctx, 'p')]), {'x': X, 'y': Y, '_outline': ['top', 'gone']})
    assert ed.calls[-1]['gates'] == {'top': GATES['top']}
    # An empty or missing outline falls back to the child gates.
    _draw(Rc(ed, [(ctx, 'p')]), {'x': X, 'y': Y, '_outline': None})
    assert set(ed.calls[-1]['gates']) == {'kid', 'kid2'}


def test_two_layers_or_a_category_colour_go_through_the_overlay_painter():
    a = Ctx('A', frames={'p': _frame(50, 0, 100, 1)})
    b = Ctx('B', frames={'p': _frame(50, 0, 100, 2)})
    ed = Ed()
    _draw(Rc(ed, [(a, 'p'), (b, 'p')]), {'x': X, 'y': Y, 'color': 'Red'})
    assert ed.calls[-1]['painter'] is not None
    assert ed.calls[-1]['color'] == 'By density'
    assert ed.calls[-1]['samples'] == ['A']
    a.frames['p']['cluster'] = np.arange(50) % 3
    _draw(Rc(ed, [(a, 'p')]), {'x': X, 'y': Y, 'color': 'cat:cluster'})
    assert ed.calls[-1]['painter'] is not None
    assert ed.calls[-1]['color'] == 'By density'


def test_the_editors_own_notes_are_clipped_to_the_small_panel():
    ed = Ed(texts=3)
    ax, _meta = _draw(Rc(ed, [(Ctx('A'), 'p')]), {'x': X, 'y': Y})
    assert len(ax.texts) == 3
    assert all(t.get_clip_on() for t in ax.texts)


# ── title ───────────────────────────────────────────────────────────────

def test_the_title_is_the_items_own_else_the_population_names():
    a = Ctx('A', names={'p': 'Lymphocytes'})
    b = Ctx('B', names={'p': 'Lymphocytes', 'q': 'Q' * 70})
    ed = Ed()
    ax, _ = _draw(Rc(ed, [(a, 'p')]), {'x': X, 'title': 'Mine'})
    assert ax.get_title() == 'Mine'
    assert ax.title.get_fontsize() == 8
    ax, _ = _draw(Rc(ed, [(a, 'p')]), {'x': X, 'title': ''})
    assert ax.get_title() == 'Lymphocytes'
    # Two samples of one population: the name once, not twice.
    ax, _ = _draw(Rc(ed, [(a, 'p'), (b, 'p')]), {'x': X})
    assert ax.get_title() == 'Lymphocytes'
    ax, _ = _draw(Rc(ed, [(a, 'p'), (b, 'q')]), {'x': X})
    assert ax.get_title() == ('Lymphocytes, ' + 'Q' * 70)[:60]
    assert len(ax.get_title()) == 60


# ── gate labels ─────────────────────────────────────────────────────────

def _labelled(rows, x=X, y=Y, layers=1, mode='dot'):
    ctx = Ctx('A', gates=dict(GATES), names={'kid': 'Kid', 'kid2': 'Kid two'},
              rows=rows)
    ed = Ed()
    ax, _ = _draw(Rc(ed, [(ctx, 'p')] * layers),
                  {'x': x, 'y': y, 'mode': mode})
    return ax, ctx


def test_each_drawn_gate_is_labelled_with_its_name_and_percent_parent():
    ax, ctx = _labelled({'kid': {'Count': 7, '%Parent': 12.345},
                         'kid2': {'Count': 3, '%Parent': 4.0,
                                  '__unknown__': False}})
    assert ctx.rows_asked == [{'Count', '%Parent'}]
    lab = {t.get_text(): t for t in ax.texts}
    assert set(lab) == {'Kid\n12.35%', 'Kid two\n4.00%'}
    t = lab['Kid\n12.35%']
    assert t.get_position() == pytest.approx((20.0, 70.0))    # rect centre
    assert t.get_transform() is ax.transData
    assert matplotlib.colors.to_hex(t.get_color()) == '#123456'
    assert matplotlib.colors.to_hex(lab['Kid two\n4.00%'].get_color()) == '#000000'
    assert t.get_clip_on()
    assert (t.get_ha(), t.get_va(), t.get_fontsize()) == ('center', 'center', 6.5)
    halo, = t.get_path_effects()
    assert halo._gc == {'linewidth': 2,
                        'foreground': ax.get_facecolor()}


def test_an_unknown_row_reads_na_and_a_gate_without_a_row_is_not_labelled():
    ax, _ = _labelled({'kid': {'Count': 0, '%Parent': 0.0, '__unknown__': True}})
    assert _texts(ax) == ['Kid\nNA']


def test_a_gate_not_on_these_axes_is_not_labelled():
    ax, _ = _labelled({'kid': {'%Parent': 1.0}, 'kid2': {'%Parent': 2.0}},
                      y='SSC-A')
    assert _texts(ax) == []


def test_no_labels_without_an_x_channel_or_with_several_layers():
    rows = {'kid': {'%Parent': 1.0}, 'kid2': {'%Parent': 2.0}}
    ax, ctx = _labelled(rows, x=None)
    assert _texts(ax) == [] and ctx.rows_asked == []
    ax, ctx = _labelled(rows, layers=2)
    assert _texts(ax) == [] and ctx.rows_asked == []


# ── zoom and the outside fraction ───────────────────────────────────────

def _zoomed(item, iid='i1', info=None):
    ctx = Ctx('A', frames={'p': _frame(400, 0, 100, 3)})
    ed = Ed()
    rc = Rc(ed, [(ctx, 'p')])
    rc.info.update(info or {})
    ax, meta = _draw(rc, dict({'x': X, 'y': Y}, **item), iid)
    return ax, meta, rc, ed, ctx


def test_a_zoom_reports_the_fraction_of_events_outside_it():
    df = _frame(400, 0, 100, 3)
    ax, meta, rc, ed, _ = _zoomed({'xlim': (0, 50)})
    want = float(((df[X] < 0) | (df[X] > 50)).mean())
    assert 0.3 < want < 0.7
    assert meta['outside'] == pytest.approx(want)
    assert rc.info['i1'] == {'ax': ax, 'outside': meta['outside']}
    assert ed.df_calls == [('A', 'p', False)]
    _, meta, rc, _, _ = _zoomed({'ylim': (0, 25)})
    want = float(((df[Y] < 0) | (df[Y] > 25)).mean())
    assert meta['outside'] == pytest.approx(want) and want > 0.5


def test_no_zoom_means_nothing_outside_and_no_frames_read():
    _, meta, rc, ed, _ = _zoomed({})
    assert meta['outside'] == 0.0 and ed.df_calls == []
    assert set(rc.info['i1']) == {'ax'}


def test_the_info_keeps_what_was_there_and_any_id_but_none_is_kept():
    ax, meta, rc, _, _ = _zoomed({'xlim': (0, 50)}, iid='',
                                 info={'': {'exploded': 'zoom'}})
    assert rc.info[''] == {'exploded': 'zoom', 'ax': ax,
                           'outside': meta['outside']}
    _, _, rc, _, _ = _zoomed({'xlim': (0, 50)}, iid=None)
    assert rc.info == {}


def test_a_zoomed_histogram_reads_histogram_frames_and_has_no_y():
    _, meta, _, ed, ctx = _zoomed({'mode': 'histogram', 'xlim': (0, 50)})
    assert meta['y'] is None and meta['x'] == X
    assert ed.calls[0]['y'] is None and ed.calls[0]['mode'] == 'histogram'
    assert ed.df_calls == [('A', 'p', True)]
    assert meta['layers'] == [(ctx, 'p')]


def test_an_overlay_reuses_the_frames_it_painted_for_the_outside_fraction():
    a = Ctx('A', frames={'p': _frame(300, 0, 100, 4)})
    b = Ctx('B', frames={'p': _frame(300, 0, 100, 5)})
    ed = Ed()
    rc = Rc(ed, [(a, 'p'), (b, 'p')])
    _, meta = _draw(rc, {'x': X, 'y': Y, 'xlim': (0, 50)}, 'i1')
    # Each layer's events read once, by the painter, not again afterwards.
    assert ed.df_calls == [('A', 'p', False), ('B', 'p', False)]
    both = pd.concat([a.frames['p'], b.frames['p']])
    assert meta['outside'] == pytest.approx(float((both[X] > 50).mean()))
