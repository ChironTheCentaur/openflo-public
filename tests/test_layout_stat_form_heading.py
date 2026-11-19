"""A layout's statistics table and bar chart name a non-default form.

Layouts compute rCV and GeoMean in the forms Preferences > Statistics picks
and leave the app as PDF, PNG and PowerPoint, so their headings say which
formula made the numbers, as the Statistics window and its CSV do
(gating.stat_form_note). Generated data and gates only (FL1-A).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from openflo import layout_render, prefs
from openflo.layout_page import chart_item
from openflo.layout_render import RenderContext, draw_chart, draw_stats

LIN = np.random.default_rng(7).lognormal(6.0, 0.9, 4000)
GATE = {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>', 'value': 300.0,
        'parent_id': None, 'name': 'Bright', 'enabled': True}
# How a layout item names that population (gating.population_keys).
KEY = 'n:bright'


def _editor():
    """One synthetic sample stored linear, with one threshold gate."""
    sample = SimpleNamespace(data=pd.DataFrame({'FL1-A': LIN}),
                             channel_labels={}, data_transforms={},
                             stat_axes=None)
    return SimpleNamespace(_samples={'synthetic-1': sample},
                           _sample_gates={'synthetic-1': {'g1': dict(GATE)}},
                           _sample_gate_order={'synthetic-1': ['g1']},
                           _channel_labels={},
                           # As the editor does: Preferences' forms.
                           _stat_forms=prefs.load_stat_forms)


def _table_headings(monkeypatch, stored):
    monkeypatch.setattr(prefs, 'read_prefs', lambda: stored)
    seen = {}
    real = layout_render._stats_table
    monkeypatch.setattr(layout_render, '_stats_table', lambda ax, body, cols: (
        seen.__setitem__('cols', cols), real(ax, body, cols)))
    item = {'kind': 'stats', 'sample': 'synthetic-1', 'populations': [KEY],
            'channel': 'FL1-A', 'stats': ['GeoMean', 'rCV', 'Median']}
    draw_stats(RenderContext(_editor(), 'synthetic-1'), Figure(),
               (0.0, 0.0, 1.0, 1.0), item)
    return seen['cols']


@pytest.mark.parametrize('stored,want', [
    ({}, ['Population', 'GeoMean FL1-A', 'rCV FL1-A', 'Median FL1-A']),
    ({'stat_rcv': 'percentile'},
     ['Population', 'GeoMean FL1-A', 'rCV FL1-A (FlowJo percentile form)',
      'Median FL1-A']),
    ({'stat_geomean': 'log'},
     ['Population', 'GeoMean FL1-A (textbook form)', 'rCV FL1-A',
      'Median FL1-A']),
], ids=['defaults', 'flowjo rcv', 'textbook geomean'])
def test_a_layout_table_names_a_non_default_form(monkeypatch, stored, want):
    assert _table_headings(monkeypatch, stored) == want


@pytest.mark.parametrize('stored,want', [
    ({}, 'GeoMean FL1-A'),
    ({'stat_geomean': 'log'}, 'GeoMean FL1-A (textbook form)'),
], ids=['default', 'textbook'])
def test_a_layout_bar_label_names_a_non_default_form(monkeypatch, stored,
                                                      want):
    monkeypatch.setattr(prefs, 'read_prefs', lambda: stored)
    item = chart_item('bar', populations=[KEY], channel='FL1-A',
                      value='GeoMean')
    ax = draw_chart(RenderContext(_editor(), 'synthetic-1'), Figure(),
                    (0.0, 0.0, 1.0, 1.0), item)
    assert ax.get_ylabel() == want


def test_only_a_geomean_heading_names_the_workspace_axis():
    # The axis note belongs to GeoMean alone; the form note follows it.
    rows = [{'__stat_forms__': {'geomean': 'display', 'rcv': 'percentile'}},
            {'__stat_forms__': {'geomean': 'log', 'rcv': 'percentile'}}]
    notes = layout_render._heading_notes
    assert notes('rCV', 'workspace axis', rows) == ' (FlowJo percentile form)'
    assert notes('Median', 'workspace axis', rows) == ''
    assert (notes('GeoMean', 'workspace axis', rows)
            == ' (workspace axis, form differs by sample)')
