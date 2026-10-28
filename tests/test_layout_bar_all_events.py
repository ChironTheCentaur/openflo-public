"""A layout bar chart of the whole sample shows the statistic it names.

A bar on the ungated layer ('All events') used to take the Median or else
the plain Mean of the channel, so a GeoMean bar showed the Mean. It now
comes from gating.population_stats like every gated layer, in the GeoMean
form Preferences > Statistics picks. The expected
values below are worked out here with numpy from the generated events, not
by OpenFlo.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

from matplotlib.figure import Figure  # noqa: E402

from openflo import prefs  # noqa: E402
from openflo.layout_page import ALL_EVENTS, chart_item  # noqa: E402
from openflo.layout_render import RenderContext, draw_chart  # noqa: E402

COFACTOR = 150.0
LIN = np.random.default_rng(7).lognormal(6.0, 0.9, 4000)


def _editor():
    """One synthetic sample: FL1-A stored on asinh (cofactor 150), as the
    editor stores a transformed channel, and no gates."""
    stored = np.arcsinh(LIN / COFACTOR)
    sample = SimpleNamespace(
        data=pd.DataFrame({'FL1-A': stored}), channel_labels={},
        data_transforms={'FL1-A': {'method': 'asinh',
                                   'cofactor': COFACTOR}},
        stat_axes=None)
    return SimpleNamespace(_samples={'synthetic-1': sample},
                           _sample_gates={'synthetic-1': {}},
                           _sample_gate_order={'synthetic-1': []},
                           _channel_labels={})


def _bar(value, monkeypatch, geomean='display'):
    monkeypatch.setattr(prefs, 'read_prefs',
                        lambda: {'stat_geomean': geomean})
    rc = RenderContext(_editor(), 'synthetic-1')
    item = chart_item('bar', populations=[ALL_EVENTS], channel='FL1-A',
                      value=value)
    ax = draw_chart(rc, Figure(), (0.0, 0.0, 1.0, 1.0), item)
    (bar,) = ax.patches
    return bar.get_height()


EXPECTED = {
    # FlowJo's GeoMean (the default): the mean on the display axis,
    # back-transformed.
    ('GeoMean', 'display'):
        COFACTOR * np.sinh(np.mean(np.arcsinh(LIN / COFACTOR))),
    # The textbook form Preferences can pick instead.
    ('GeoMean', 'log'): np.exp(np.mean(np.log(LIN))),
    ('Mean', 'display'): np.mean(LIN),
    ('Median', 'display'): np.median(LIN),
}


def test_the_generated_channel_round_trips_to_its_linear_values():
    from openflo.pipeline import to_linear
    back = to_linear(np.arcsinh(LIN / COFACTOR),
                     {'method': 'asinh', 'cofactor': COFACTOR})
    assert np.allclose(back, LIN, rtol=1e-9)


@pytest.mark.parametrize('value,geomean', list(EXPECTED))
def test_an_all_events_bar_shows_the_chosen_statistic(value, geomean,
                                                      monkeypatch):
    got = _bar(value, monkeypatch, geomean)
    assert got == pytest.approx(EXPECTED[value, geomean], rel=1e-9)


def test_both_geomean_forms_differ_from_the_mean(monkeypatch):
    mean = np.mean(LIN)
    for form in ('display', 'log'):
        assert (_bar('GeoMean', monkeypatch, form)
                != pytest.approx(mean, rel=0.05))


def test_count_and_percent_total_of_all_events(monkeypatch):
    assert _bar('Count', monkeypatch) == len(LIN)
    assert _bar('%Total', monkeypatch) == 100.0
