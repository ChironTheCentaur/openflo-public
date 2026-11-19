"""A statistic computed in a non-default form says so in its heading.

Preferences can switch rCV/rSD to FlowJo's percentile form and GeoMean to
the textbook form on the desktop; the phone always uses the defaults
(gating.DEFAULT_STAT_FORMS). Under one bare "rCV" heading, the two apps
exported different numbers for the same population. Every heading now names
a non-default form (gating.stat_form_note). Generated data only.
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from openflo.document import Document
from openflo.gating import (
    CHANNEL_STAT_ORDER,
    DEFAULT_STAT_FORMS,
    FLOWJO_STAT_FORMS,
    population_stats,
    stat_column_heading,
)
from tests.test_document import spill_fcs

RNG = np.random.default_rng(5)
DF = pd.DataFrame({'FL1-A': RNG.lognormal(6, 0.8, 500)})


def _rows(forms):
    return population_stats('s', DF, {}, [], {'FL1-A': 'FL1-A'}, ['FL1-A'],
                            {'rCV', 'rSD', 'GeoMean', 'Median'},
                            CHANNEL_STAT_ORDER, stat_forms=forms,
                            all_events=True)


def _heading(col, rows):
    return stat_column_heading(col, rows, CHANNEL_STAT_ORDER)


def test_the_default_forms_keep_the_plain_heading():
    rows = _rows(None)
    for col in ('rCV FL1-A', 'rSD FL1-A', 'GeoMean FL1-A', 'Median FL1-A'):
        assert _heading(col, rows) == f'{col} (linear)'


def test_flowjo_percentile_rcv_is_named():
    rows = _rows(FLOWJO_STAT_FORMS)
    assert _heading('rCV FL1-A', rows) == 'rCV FL1-A (linear, FlowJo percentile form)'
    assert _heading('rSD FL1-A', rows) == 'rSD FL1-A (linear, FlowJo percentile form)'
    # GeoMean's form is FlowJo's display form in both, so no note there.
    assert _heading('GeoMean FL1-A', rows) == 'GeoMean FL1-A (linear)'
    assert _heading('Median FL1-A', rows) == 'Median FL1-A (linear)'


def test_textbook_geomean_is_named():
    rows = _rows({'geomean': 'log', 'rcv': 'mad'})
    assert _heading('GeoMean FL1-A', rows) == 'GeoMean FL1-A (linear, textbook form)'
    assert _heading('rCV FL1-A', rows) == 'rCV FL1-A (linear)'


def test_rows_in_different_forms_say_the_form_differs():
    rows = _rows(None) + _rows(FLOWJO_STAT_FORMS)
    assert _heading('rCV FL1-A', rows) == 'rCV FL1-A (linear, form differs by sample)'


def test_the_two_forms_really_give_different_numbers_here():
    # Why the heading matters: the same events, two rCVs.
    a = _rows(DEFAULT_STAT_FORMS)[0]['rCV FL1-A']
    b = _rows(FLOWJO_STAT_FORMS)[0]['rCV FL1-A']
    assert abs(a - b) / a > 0.05


@pytest.mark.parametrize('forms,want', [
    (None, 'rCV FL1-A (linear)'),
    (FLOWJO_STAT_FORMS, 'rCV FL1-A (linear, FlowJo percentile form)')])
def test_the_exported_csv_heading_names_the_form(tmp_path, forms, want):
    doc = Document(stat_forms=forms)
    doc.open_fcs(spill_fcs(tmp_path / 'raw' / 'a.fcs'))
    out = tmp_path / 'stats.csv'
    doc.export_stats_csv(str(out), channels=['FL1-A'], stats=['rCV'])
    with open(out, newline='', encoding='utf-8-sig') as f:
        head = next(csv.reader(f))
    assert want in head


def test_the_phone_stats_say_the_form_too(tmp_path):
    doc = Document(stat_forms=FLOWJO_STAT_FORMS)
    s = doc.open_fcs(spill_fcs(tmp_path / 'raw' / 'a.fcs'))
    res = doc.stats(s, channels=['FL1-A'], stats=['rCV'])
    assert 'FlowJo percentile form' in str(res)


def test_a_guessed_transform_is_named_first_then_the_form():
    from openflo.gating import GUESSED_TAG
    rows = _rows(FLOWJO_STAT_FORMS)
    assert (_heading(f'rCV FL1-A {GUESSED_TAG}', rows)
            == 'rCV FL1-A (linear, guessed logicle, FlowJo percentile form)')
    assert (_heading(f'Median FL1-A {GUESSED_TAG}', rows)
            == 'Median FL1-A (linear, guessed logicle)')


def test_a_heading_without_rows_keeps_its_plain_name():
    # The Statistics window asks for headings before any rows are computed.
    for col in ('rCV FL1-A', 'GeoMean FL1-A'):
        assert _heading(col, None) == f'{col} (linear)'


def test_an_axis_and_a_mixed_form_are_both_named():
    # A GeoMean column mixing FlowJo's form (on a workspace axis) with the
    # textbook form names the axis and that the form differs.
    rows = [{'__geomean_axis__': {'FL1-A': 'workspace axis'},
             '__stat_forms__': {'geomean': 'display', 'rcv': 'mad'}},
            {'__stat_forms__': {'geomean': 'log', 'rcv': 'mad'}}]
    assert (_heading('GeoMean FL1-A', rows)
            == 'GeoMean FL1-A (linear, workspace axis, form differs by sample)')
