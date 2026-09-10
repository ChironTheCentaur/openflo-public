"""Propagate-downsample must not depend on which sample loads first.

`ds_propagate_var` trims every loaded sample to the smallest ENABLED sample's
size. The toggle handler does exactly that, in one pass. But `_on_loaded` also
trimmed each arriving sample against the floor of whatever had loaded SO FAR —
and that order comes from a thread pool, so it is not reproducible.

Measured on four samples of 50k/40k/30k/20k events, all enabled:

    arrival a,b,c,d -> 50000, 40000, 30000, 20000   (nothing trimmed at all)
    arrival d,c,b,a -> 20000, 20000, 20000, 20000   (all trimmed)
    arrival b,d,a,c -> 20000, 40000, 20000, 20000   (neither)

Only a descending arrival order ever trimmed anything. The same session
resumed twice could therefore hold different event counts — and every
frequency, gate count and statistic computed from them would differ, for a
reason having nothing to do with the data.

The trim now happens once, over the whole set, after the load burst settles.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

from tests.test_gui_smoke import _editor_or_skip  # noqa: E402

SIZES = {'a': 5_000, 'b': 4_000, 'c': 3_000, 'd': 2_000}
SMALLEST = min(SIZES.values())


def _sample(name):
    rng = np.random.default_rng(len(name) + SIZES[name])
    return SimpleNamespace(
        name=name, path=f'C:/d/{name}.fcs',
        data=pd.DataFrame({'FSC-A': rng.normal(1000.0, 200.0, SIZES[name]),
                           'CD3': rng.normal(500.0, 100.0, SIZES[name])}),
        channel_labels={'FSC-A': 'FSC-A', 'CD3': 'CD3'},
        fluor_channels=['CD3'], scatter_channels=['FSC-A'])


def _resume(order, propagate=True, enabled=None, settle=True):
    """Drive the real arrival + settle path in a given arrival order."""
    root, editor, _gui = _editor_or_skip()
    try:
        editor.ds_propagate_var.set(propagate)
        for name in order:
            on = True if enabled is None else enabled.get(name, False)
            editor._sample_plot_enabled[name] = on
            editor._on_loaded(name, _sample(name))
            editor._sample_plot_enabled[name] = on
        if settle:
            editor._on_load_settled()
        return {n: len(editor._samples[n].data) for n in SIZES}
    finally:
        try:
            root.destroy()
        except Exception:
            pass


ORDERS = [
    pytest.param(('a', 'b', 'c', 'd'), id='largest-first'),
    pytest.param(('d', 'c', 'b', 'a'), id='smallest-first'),
    pytest.param(('b', 'd', 'a', 'c'), id='interleaved'),
]


@pytest.mark.parametrize('order', ORDERS)
def test_every_arrival_order_gives_the_same_result(order):
    counts = _resume(order)
    assert set(counts.values()) == {SMALLEST}, (
        f'arrival order {order} produced {counts}; every enabled sample should '
        f'be trimmed to the smallest ({SMALLEST})')


def test_the_orders_agree_with_each_other():
    """Stated directly, because this is the property that was broken."""
    results = [_resume(order.values[0]) for order in ORDERS]
    assert results[0] == results[1] == results[2], results


def test_it_matches_what_the_toggle_does():
    """The toggle handler is the reference semantics — one pass, floor over
    enabled samples. Loading must converge to the same place."""
    loaded = _resume(('a', 'b', 'c', 'd'))

    root, editor, _gui = _editor_or_skip()
    try:
        editor.ds_propagate_var.set(False)
        for name in ('a', 'b', 'c', 'd'):
            editor._sample_plot_enabled[name] = True
            editor._on_loaded(name, _sample(name))
            editor._sample_plot_enabled[name] = True
        editor.ds_propagate_var.set(True)
        editor._apply_propagate_downsample()
        toggled = {n: len(editor._samples[n].data) for n in SIZES}
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    assert loaded == toggled, f'load path {loaded} vs toggle path {toggled}'


def test_nothing_is_trimmed_when_propagate_is_off():
    counts = _resume(('a', 'b', 'c', 'd'), propagate=False)
    assert counts == SIZES, 'samples were trimmed with propagate off'


def test_the_floor_ignores_samples_that_are_not_plotted():
    """A loaded-but-unchecked smaller sample must not drag everyone down — the
    documented behaviour of `_smallest_loaded_sample_size`."""
    enabled = {'a': True, 'b': True, 'c': True, 'd': False}
    counts = _resume(('a', 'b', 'c', 'd'), enabled=enabled)
    assert counts['d'] == SIZES['d'], 'the unchecked sample was itself resized'
    for name in ('a', 'b', 'c'):
        assert counts[name] == SIZES['c'], (
            f'{name} was trimmed to {counts[name]}; the smallest ENABLED '
            f"sample is 'c' at {SIZES['c']}")


def test_applying_it_twice_changes_nothing():
    """It runs from both the load settle and the end of session restore, so it
    has to be safe to repeat."""
    root, editor, _gui = _editor_or_skip()
    try:
        editor.ds_propagate_var.set(True)
        for name in ('a', 'b', 'c', 'd'):
            editor._sample_plot_enabled[name] = True
            editor._on_loaded(name, _sample(name))
            editor._sample_plot_enabled[name] = True
        editor._apply_propagate_downsample()
        once = {n: len(editor._samples[n].data) for n in SIZES}
        trimmed, _floor = editor._apply_propagate_downsample()
        twice = {n: len(editor._samples[n].data) for n in SIZES}
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    assert once == twice
    assert trimmed == 0, 'a second pass re-sampled already-trimmed data'
