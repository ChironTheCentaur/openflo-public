"""The layout editor turns its items into pixels off the window thread, so the
window keeps answering the mouse while a page draws. The pixels shown are the
ones the PNG route showed; exports are not touched. Synthetic samples and
generic channels only."""
from __future__ import annotations

import io
import threading
import time
import tkinter as tk

import numpy as np
import pytest

from openflo import layout_page as lp
from tests.test_layout_editor import _editor_or_skip, _load


def _run_until(root, done, timeout=30.0):
    """Run Tk's main loop until ``done()``. The main loop, not update():
    a background thread hands its result to the window thread through
    ``after``, which tkinter delivers only while the main loop runs."""
    end = time.monotonic() + timeout
    error = []

    def poll():
        try:
            finished = done() or time.monotonic() > end
        except Exception as exc:                 # noqa: BLE001 - re-raised below, once the main loop has stopped
            error.append(exc)
            finished = True
        if finished:
            root.quit()
        else:
            root.after(5, poll)
    root.after(0, poll)
    root.mainloop()
    if error:
        raise error[0]
    assert done(), 'timed out'


def _drain(root, win):
    """Run the window until it has no item left to draw, no drawing thread
    and no redraw waiting to start."""
    _run_until(root, lambda: (win._pending == 0 and not win._raster_busy
                              and win._redraw_after is None))
    root.update()


def _stack(win, tag):
    """Stacking positions (bottom 0) of the canvas items tagged ``tag``."""
    order = win.canvas.find_all()
    return [order.index(i) for i in win.canvas.find_withtag(tag)]


@pytest.fixture
def page(monkeypatch):
    """A layout editor (kept inside 1024x768, the Windows runner's screen)
    with a sample and three plots on its page."""
    root, ed = _editor_or_skip()
    import openflo.ui_layout_editor as ule
    from openflo.layout_render import SampleContext
    monkeypatch.setattr(ule.LayoutEditorWindow, 'geometry',
                        lambda self, *a, **k: None)
    cells, _pos = _load(ed, 's1', 1, 3000, 300.0)
    keys = SampleContext(ed, 's1').keys
    win = ule.LayoutEditorWindow(ed)
    win.wm_geometry('900x640')
    win.update()
    for mode, box in (('dot', (0.0, 0.0, 0.3, 0.3)),
                      ('histogram', (0.35, 0.0, 0.3, 0.3)),
                      ('density', (0.0, 0.4, 0.3, 0.3))):
        win.layout.add(lp.plot_item(
            population=keys[cells], x='FL1-A',
            y=None if mode == 'histogram' else 'FL2-A', mode=mode, box=box))
    win._redraw()
    _drain(root, win)
    # Window-size redraws are not under test here, and one arriving late
    # under a loaded machine (xdist) would start a new redraw mid-test.
    win.canvas.unbind('<Configure>')
    if win._redraw_after is not None:
        win.after_cancel(win._redraw_after)
        win._redraw_after = None
    try:
        yield root, ed, win, ule
    finally:
        root.destroy()


def _ppm_pixels(ppm):
    """``(h, w, 3)`` uint8 pixels of binary PPM data."""
    magic, w, h, maxval, body = ppm.split(maxsplit=4)
    assert (magic, maxval) == (b'P6', b'255')
    return np.frombuffer(body, np.uint8).reshape(int(h), int(w), 3)


def _shown_pixels(img):
    """Every pixel of a Tk image, as Tk holds it (``PhotoImage.data`` is
    Python 3.13+; this is the Tcl command it wraps)."""
    return str(img.tk.call(str(img), 'data'))


def _png_pixels(fig):
    """``(h, w, 3)`` uint8 pixels of ``fig`` saved as the old preview did."""
    import matplotlib.image as mpimg
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, facecolor='white')
    buf.seek(0)
    rgba = mpimg.imread(buf, format='png')
    return np.round(rgba[..., :3] * 255).astype(np.uint8)


def test_the_preview_pixels_are_the_png_routes_pixels(page):
    """The PPM pixels equal the PNG preview's, pixel for pixel, for every
    kind of plot on the page, and Tk shows them the same."""
    _root, ed, win, ule = page
    from openflo.layout_render import RenderContext, render_item
    rc = RenderContext(ed, 's1')
    for iid in win.layout.order:
        fig = render_item(ed, win.layout, iid, 's1', 60, rc=rc)
        ppm = ule._raster(fig)
        want = _png_pixels(fig)
        got = _ppm_pixels(ppm)
        assert got.shape == want.shape
        assert np.array_equal(got, want)
        # Not a blank page: the plot is in there.
        assert len(np.unique(got.reshape(-1, 3), axis=0)) > 10
        new = ule._photo_from(ppm, fig, win)
        old = ule._photo(fig, win)
        assert (new.width(), new.height()) == (old.width(), old.height())
        assert _shown_pixels(new) == _shown_pixels(old)


def test_redraw_returns_before_the_pixels_and_the_window_keeps_answering(
        page, monkeypatch):
    root, _ed, win, ule = page
    release = threading.Event()
    calls = []
    real = ule._raster

    def slow(fig):
        calls.append(threading.current_thread() is threading.main_thread())
        release.wait(5.0)
        return real(fig)
    monkeypatch.setattr(ule, '_raster', slow)
    iid = win.layout.order[0]
    win.layout.items[iid]['box'] = [0.0, 0.0, 0.4, 0.35]   # needs new pixels
    # Zoomed in, not exploded: it gets a Zoom to fit badge. Selected.
    win.layout.items[iid].update(xlim=(400.0, 450.0), explode=False)
    win._sel = iid
    t0 = time.monotonic()
    win._redraw()
    assert time.monotonic() - t0 < 2.0       # did not wait for the pixels
    assert win._pending == 1
    # Its old image stands in meanwhile.
    stand_in = win._shown[iid]
    assert win.canvas.type(win.canvas.find_withtag(f'img:{iid}')[0]) == 'image'
    # The window still runs its event loop and answers.
    answered = []
    win.after(0, lambda: answered.append(True))
    _run_until(root, lambda: bool(answered), timeout=5.0)
    assert not release.is_set() and win._pending == 1
    release.set()
    _drain(root, win)
    assert calls == [False]                  # drawn off the window thread
    assert win._shown[iid] is not stand_in
    (img_id,) = win.canvas.find_withtag(f'img:{iid}')
    assert win.canvas.itemcget(img_id, 'image') == str(win._shown[iid])
    # The new image goes back into the page order (it overlaps the item
    # after it), under the ids, the badge and the selection.
    top = max(_stack(win, f'img:{iid}'))
    for tag in ('uid', 'fit', 'sel'):
        assert _stack(win, tag) and min(_stack(win, tag)) > top, tag
    positions = [min(_stack(win, f'img:{i}')) for i in win.layout.order]
    assert positions == sorted(positions)


def test_a_new_item_shows_a_placeholder_until_it_is_drawn(page, monkeypatch):
    root, _ed, win, ule = page
    release = threading.Event()
    real = ule._raster
    monkeypatch.setattr(ule, '_raster',
                        lambda fig: (release.wait(5.0), real(fig))[1])
    iid = win.layout.add(lp.text_item('Run {sample}',
                                      box=(0.5, 0.5, 0.3, 0.1)))
    win._redraw()
    kinds = {win.canvas.type(i) for i in win.canvas.find_withtag(f'img:{iid}')}
    assert kinds == {'rectangle', 'text'}
    texts = [win.canvas.itemcget(i, 'text')
             for i in win.canvas.find_withtag(f'img:{iid}')
             if win.canvas.type(i) == 'text']
    assert texts == ['text\ndrawing…']
    release.set()
    _drain(root, win)
    kinds = [win.canvas.type(i) for i in win.canvas.find_withtag(f'img:{iid}')]
    assert kinds == ['image']


def test_a_newer_redraw_drops_the_older_pixels(page, monkeypatch):
    """Pixels of a superseded redraw are neither shown nor cached, and the
    older redraw stops drawing its remaining items."""
    root, _ed, win, ule = page
    release = threading.Event()
    drawn = []
    real = ule._raster

    pending_then, stale_then = [], []

    def slow(fig):
        drawn.append(fig)
        if len(drawn) == 1:
            release.wait(5.0)
        if len(drawn) == 2:     # the newer redraw's first: none placed yet
            pending_then.append(win._pending)
            # The older redraw's pixels have arrived by now, and were dropped.
            stale_then.append(any('"first"' in k[3] for k in list(win._images)))
        return real(fig)
    monkeypatch.setattr(ule, '_raster', slow)
    a = win.layout.order[0]
    for iid in win.layout.order:                 # every item needs pixels
        win.layout.items[iid]['title'] = 'first'
    win._redraw()
    end = time.monotonic() + 5.0
    while not drawn and time.monotonic() < end:
        time.sleep(0.005)
    assert len(drawn) == 1                       # the first is in the works
    for iid in win.layout.order:
        win.layout.items[iid]['title'] = 'second'
    win._redraw()
    release.set()
    _drain(root, win)
    # 1 of the first redraw (cut short) and 3 of the second.
    assert len(drawn) == 4
    assert pending_then == [3]       # the older one did not touch its count
    assert stale_then == [False]     # nor cached its pixels
    keys = [k for k in win._images if k[1] == a]
    assert len(keys) == 1 and '"second"' in keys[0][3]
    assert not any('"first"' in k[3] for k in win._images)
    (img_id,) = win.canvas.find_withtag(f'img:{a}')
    assert win.canvas.itemcget(img_id, 'image') == str(win._images[keys[0]][0])


def test_unchanged_items_are_not_drawn_again(page, monkeypatch):
    root, _ed, win, ule = page
    calls = []
    real = ule._raster
    monkeypatch.setattr(ule, '_raster',
                        lambda fig: (calls.append(1), real(fig))[1])
    shown = dict(win._shown)
    win._redraw()
    assert win._pending == 0
    _drain(root, win)
    assert calls == []
    assert win._shown == shown
    # Cached items keep their axes, for zooming and panning them.
    assert set(win._axes) == set(win.layout.order)
    for key, (_img, _outside, ax) in win._images.items():
        assert ax is not None and win._axes[key[1]] is ax
    win.layout.items[win.layout.order[1]]['title'] = 'changed'
    win._redraw()
    _drain(root, win)
    assert len(calls) == 1                       # only the changed item


def test_a_tk_that_cannot_read_ppm_gets_png(page, monkeypatch):
    _root, ed, win, ule = page
    from openflo.layout_render import RenderContext, render_item
    real = ule.tk.PhotoImage
    tried = []

    def no_ppm(*args, **kwargs):
        if kwargs.get('format') == 'ppm':
            tried.append(1)
            raise tk.TclError("couldn't recognize image data")
        return real(*args, **kwargs)
    monkeypatch.setattr(ule.tk, 'PhotoImage', no_ppm)
    monkeypatch.setattr(ule, '_PPM_OK', [True])
    fig = render_item(ed, win.layout, win.layout.order[0], 's1', 60,
                      rc=RenderContext(ed, 's1'))
    ppm = ule._raster(fig)
    img = ule._photo_from(ppm, fig, win)
    assert _shown_pixels(img) == _shown_pixels(ule._photo(fig, win))
    assert ule._PPM_OK == [False]
    ule._photo_from(ppm, fig, win)
    assert tried == [1]                          # not asked again


def test_a_drawing_error_is_shown_and_the_page_carries_on(page, monkeypatch):
    root, _ed, win, ule = page
    real = ule._raster
    n = []

    def first_fails(fig):
        n.append(1)
        if len(n) == 1:
            raise RuntimeError('synthetic draw failure')
        return real(fig)
    monkeypatch.setattr(ule, '_raster', first_fails)
    for iid in win.layout.order:
        win.layout.items[iid]['title'] = 'again'
    win._redraw()
    _drain(root, win)
    assert win.status.get() == 'Could not draw an item: synthetic draw failure'
    assert len(n) == 3                           # the others still drew
    first, *rest = win.layout.order
    assert all('"again"' in k[3] for k in win._images if k[1] in rest)
    assert not any('"again"' in k[3] for k in win._images if k[1] == first)


def test_a_new_item_that_cannot_be_drawn_says_so(page, monkeypatch):
    root, _ed, win, ule = page

    def fails(fig):
        raise RuntimeError('synthetic draw failure')
    monkeypatch.setattr(ule, '_raster', fails)
    iid = win.layout.add(lp.text_item('Run {sample}',
                                      box=(0.5, 0.5, 0.3, 0.1)))
    win._redraw()
    _drain(root, win)
    texts = [win.canvas.itemcget(i, 'text')
             for i in win.canvas.find_withtag(f'img:{iid}')
             if win.canvas.type(i) == 'text']
    assert texts == ['text\ncould not draw']


def test_a_superseded_redraws_error_is_not_shown(page, monkeypatch):
    root, _ed, win, ule = page
    release = threading.Event()
    real = ule._raster
    n = []

    def first_fails_late(fig):
        n.append(1)
        if len(n) == 1:
            release.wait(5.0)
            raise RuntimeError('synthetic draw failure')
        return real(fig)
    monkeypatch.setattr(ule, '_raster', first_fails_late)
    iid = win.layout.order[0]
    win.layout.items[iid]['title'] = 'first'
    win._redraw()
    win.layout.items[iid]['title'] = 'second'
    win._redraw()
    release.set()
    _drain(root, win)
    assert len(n) == 2
    assert 'Could not draw' not in win.status.get()
    assert any('"second"' in k[3] for k in win._images if k[1] == iid)


def test_an_item_deleted_while_it_draws_is_dropped(page, monkeypatch):
    root, _ed, win, ule = page
    from openflo import errors
    reported = []
    monkeypatch.setattr(errors, 'report', lambda *a, **k: reported.append(a))
    release = threading.Event()
    real = ule._raster
    monkeypatch.setattr(ule, '_raster',
                        lambda fig: (release.wait(5.0), real(fig))[1])
    iid = win.layout.order[0]
    win.layout.items[iid]['title'] = 'going'
    win._redraw()
    win.layout.remove(iid)
    release.set()
    _drain(root, win)
    assert reported == []
    assert not any('"going"' in k[3] for k in win._images)


def test_one_drawing_thread_at_a_time(page, monkeypatch):
    """Redraws in quick succession (a pan redraws every 80 ms) never have
    two items drawing at once."""
    root, _ed, win, ule = page
    real = ule._raster
    lock = threading.Lock()
    now, most = [0], [0]

    def counted(fig):
        with lock:
            now[0] += 1
            most[0] = max(most[0], now[0])
        try:
            time.sleep(0.05)
            return real(fig)
        finally:
            with lock:
                now[0] -= 1
    monkeypatch.setattr(ule, '_raster', counted)
    for k in range(5):
        for iid in win.layout.order:
            win.layout.items[iid]['title'] = f'pan {k}'
        win._redraw()
        _run_until(root, lambda: now[0] > 0, timeout=5.0)
    _drain(root, win)
    assert most[0] == 1
    assert all('"pan 4"' in k[3] for k in win._images)


def test_the_editor_window_installs_the_mathtext_lock(page, monkeypatch):
    _root, ed, _win, ule = page
    from matplotlib import mathtext
    parse = mathtext.MathTextParser.parse
    monkeypatch.setattr(mathtext.MathTextParser, 'parse',
                        getattr(parse, '__wrapped__', parse))
    assert not getattr(mathtext.MathTextParser.parse, '_openflo_locked', False)
    other = ule.LayoutEditorWindow(ed)
    try:
        assert getattr(mathtext.MathTextParser.parse, '_openflo_locked', False)
    finally:
        other.destroy()


def test_the_newest_of_several_waiting_redraws_is_the_one_drawn(
        page, monkeypatch):
    """Three redraws while one item draws (a pan does this): the third is
    drawn, not the second, and nothing is left pending."""
    root, _ed, win, ule = page
    release = threading.Event()
    real = ule._raster
    n = []

    def first_held(fig):
        n.append(1)
        if len(n) == 1:
            release.wait(5.0)
        return real(fig)
    monkeypatch.setattr(ule, '_raster', first_held)
    for title in ('one', 'two', 'three'):
        for iid in win.layout.order:
            win.layout.items[iid]['title'] = title
        win._redraw()
    release.set()
    _drain(root, win)
    assert win._pending == 0
    for iid in win.layout.order:
        (key,) = [k for k in win._images if k[1] == iid]
        assert '"three"' in key[3]
        (img_id,) = win.canvas.find_withtag(f'img:{iid}')
        assert win.canvas.itemcget(img_id, 'image') == str(win._images[key][0])


def test_mathtext_parses_are_one_at_a_time():
    """matplotlib's mathtext parser holds per-parse state in one shared
    object: two threads parsing tick labels at once corrupt each other. The
    layout editor's lock makes them take turns; installing it twice wraps
    once."""
    import sys

    from matplotlib import mathtext
    from matplotlib.font_manager import FontProperties

    import openflo.ui_layout_editor as ule
    ule._serialise_mathtext()
    ule._serialise_mathtext()
    parse = mathtext.MathTextParser.parse
    assert getattr(parse, '_openflo_locked', False)
    assert not getattr(parse.__wrapped__, '_openflo_locked', False)
    parser = mathtext.MathTextParser('agg')
    exprs = ['$10^{' + str(i - 200) + '}$' for i in range(400)]
    # The answers, parsed one at a time by a parser of their own (its own
    # cache), at the dpi each thread will use.
    alone = mathtext.MathTextParser('agg')
    want = {i: alone.parse(e, dpi=101 + i % 2, prop=FontProperties(size=10))
            for i, e in enumerate(exprs)}
    errors, got = [], {}

    def run(k):
        for i in range(k, len(exprs), 2):
            try:
                got[i] = parser.parse(exprs[i], dpi=101 + k,
                                      prop=FontProperties(size=10))
            except Exception as exc:             # noqa: BLE001 - every failure is collected and asserted on below
                errors.append(exc)
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)                  # threads swap constantly
    try:
        threads = [threading.Thread(target=run, args=(k,)) for k in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60.0)
    finally:
        sys.setswitchinterval(old)
    assert errors == []
    assert sorted(got) == list(range(len(exprs)))
    for i, res in got.items():
        assert (res.width, res.height, res.depth) == (
            want[i].width, want[i].height, want[i].depth)
        assert np.array_equal(np.asarray(res.image), np.asarray(want[i].image))


def test_drawing_opens_no_window_and_closing_mid_draw_is_quiet(
        page, monkeypatch):
    """Windows-only hang risks: the background drawing opens no window or
    dialog, and closing the editor while an item draws neither raises nor
    reports an error when the pixels arrive."""
    root, ed, win, ule = page
    from openflo import errors
    threads = []
    real_run = ule.run_async
    monkeypatch.setattr(ule, 'run_async', lambda *a, **k: (
        threads.append(real_run(*a, **k)), threads[-1])[1])
    reported = []
    monkeypatch.setattr(errors, 'report', lambda *a, **k: reported.append(a))
    before = set(root.winfo_children()) | set(ed.winfo_children())
    release = threading.Event()
    real = ule._raster
    started = threading.Event()

    def slow(fig):
        started.set()
        release.wait(5.0)
        return real(fig)
    monkeypatch.setattr(ule, '_raster', slow)
    for iid in win.layout.order:
        win.layout.items[iid]['title'] = 'closing'
    win._redraw()
    assert started.wait(5.0)
    root.update()
    after = set(root.winfo_children()) | set(ed.winfo_children())
    assert after == before                       # no new window
    win.destroy()
    release.set()
    assert len(threads) == 1
    _run_until(root, lambda: not threads[0].is_alive())
    root.update()                                # its callbacks ran, quietly
    assert reported == []
    assert len(threads) == 1                     # and start nothing more
