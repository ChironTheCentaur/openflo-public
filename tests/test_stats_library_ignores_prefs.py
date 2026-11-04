"""The library's statistics never read the app's Preferences file.

gating.population_stats used to take the GeoMean and rCV forms from
Preferences when it was given none, so the same script on the same data gave
different numbers on two machines. It now uses FlowJo's forms unless told
otherwise, and the app passes the user's Preferences in itself: the
Statistics window (also when its rows are computed off the Tk thread) and
layouts keep following Preferences > Statistics.

Every expected value is worked out here with numpy from generated events,
not by OpenFlo.
"""
# pyright: reportArgumentType=false, reportCallIssue=false
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from openflo import prefs  # noqa: E402
from openflo.gating import (  # noqa: E402
    CHANNEL_STAT_ORDER,
    DEFAULT_STAT_FORMS,
    FLOWJO_STAT_FORMS,
    population_stats,
)
from openflo.robust import MAD_TO_SD  # noqa: E402

COFACTOR = 150.0
LIN = np.random.default_rng(11).lognormal(6.0, 0.9, 3000)
SPEC = {'method': 'asinh', 'cofactor': COFACTOR}
WANT = {'Count', 'GeoMean', 'rCV'}
GATES = {'g1': {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                'value': -np.inf, 'parent_id': None, 'name': 'All',
                'enabled': True}}
# Both forms differ from the defaults (display GeoMean, MAD rCV).
CHOSEN = {'stat_geomean': 'log', 'stat_rcv': 'percentile'}


def _expected(geomean, rcv):
    """GeoMean and rCV of LIN in the named forms, by hand."""
    if geomean == 'display':
        g = COFACTOR * np.sinh(np.mean(np.arcsinh(LIN / COFACTOR)))
    else:
        g = np.exp(np.mean(np.log(LIN)))
    med = np.median(LIN)
    if rcv == 'mad':
        rsd = MAD_TO_SD * np.median(np.abs(LIN - med))
    else:
        lo, hi = np.percentile(LIN, [15.87, 84.13])
        rsd = 0.5 * (hi - lo)
    return g, rsd / med * 100.0


def _df():
    """FL1-A stored on asinh, as the editor stores a transformed channel."""
    return pd.DataFrame({'FL1-A': np.arcsinh(LIN / COFACTOR)})


def _values(row):
    return row['GeoMean FL1-A'], row['rCV FL1-A']


def _prefs(monkeypatch, values):
    monkeypatch.setattr(prefs, 'read_prefs', lambda: dict(values))


def test_the_two_forms_differ_on_this_data():
    # Otherwise the tests below could not tell which form was used.
    a, b = _expected('display', 'mad'), _expected('log', 'percentile')
    assert abs(a[0] - b[0]) / b[0] > 0.05
    assert abs(a[1] - b[1]) / b[1] > 0.05


def test_library_uses_its_defaults_whatever_preferences_say(monkeypatch):
    _prefs(monkeypatch, CHOSEN)
    row = population_stats('synthetic_1', _df(), GATES, ['g1'], {},
                           ['FL1-A'], WANT, CHANNEL_STAT_ORDER,
                           transforms={'FL1-A': SPEC})[0]
    assert _values(row) == pytest.approx(
        _expected('display', 'mad'), rel=1e-9)


def test_library_never_opens_preferences(monkeypatch):
    def boom():
        raise AssertionError('population_stats read Preferences')
    monkeypatch.setattr(prefs, 'read_prefs', boom)
    row = population_stats('synthetic_1', _df(), GATES, ['g1'], {},
                           ['FL1-A'], WANT, CHANNEL_STAT_ORDER,
                           transforms={'FL1-A': SPEC})[0]
    assert row['Count'] == len(LIN)


def test_library_follows_the_forms_it_is_given(monkeypatch):
    _prefs(monkeypatch, {})
    row = population_stats('synthetic_1', _df(), GATES, ['g1'], {},
                           ['FL1-A'], WANT, CHANNEL_STAT_ORDER,
                           transforms={'FL1-A': SPEC},
                           stat_forms={'geomean': 'log', 'rcv': 'mad'})[0]
    assert _values(row) == pytest.approx(_expected('log', 'mad'), rel=1e-9)


def test_library_defaults_match_preferences_defaults(monkeypatch):
    _prefs(monkeypatch, {})
    assert prefs.load_stat_forms() == DEFAULT_STAT_FORMS
    # rCV defaults to the robust-statistics (MAD) form; FlowJo's percentile
    # form is the choice for matching FlowJo.
    assert DEFAULT_STAT_FORMS == {'geomean': 'display', 'rcv': 'mad'}
    assert FLOWJO_STAT_FORMS == {'geomean': 'display', 'rcv': 'percentile'}


# ── The app keeps following Preferences ──────────────────────────────────────

def _gui():
    try:
        import tkinter as _tk  # noqa: F401

        from openflo.gui import ViewGateEditorWindow
    except (ImportError, RuntimeError) as e:
        pytest.skip(f'openflo.gui not importable: {e}')
    return ViewGateEditorWindow


def test_app_statistics_follow_preferences(monkeypatch):
    V = _gui()
    _prefs(monkeypatch, CHOSEN)
    row = V._population_stats('synthetic_1', _df(), GATES, ['g1'], {},
                              ['FL1-A'], WANT,
                              transforms={'FL1-A': SPEC})[0]
    assert _values(row) == pytest.approx(_expected('log', 'percentile'), rel=1e-9)


def _editor(V):
    sample = types.SimpleNamespace(
        data=_df(), fluor_channels=['FL1-A'], channel_labels={},
        data_transforms={'FL1-A': SPEC}, stat_axes=None)
    ed = types.SimpleNamespace(
        _samples={'synthetic_1': sample},
        _sample_gates={'synthetic_1': GATES},
        _sample_gate_order={'synthetic_1': ['g1']},
        _sample_order=['synthetic_1'], _channel_labels={},
        _population_stats=V._population_stats, STAT_POP=V.STAT_POP,
        STAT_CHAN=V.STAT_CHAN)
    ed._stats_rows_for = lambda args: V._stats_rows_for(ed, args)
    return ed


def test_off_thread_rows_use_the_forms_read_with_the_snapshot(monkeypatch):
    # The snapshot (Tk thread) reads Preferences along with the cache key's
    # settings; the rows computed later off-thread must use those forms,
    # not whatever Preferences hold by then.
    V = _gui()
    ed = _editor(V)
    _prefs(monkeypatch, CHOSEN)
    snap = V._stats_snapshot(ed, WANT)
    _prefs(monkeypatch, {})
    rows, _cols = V._stats_rows_from_snapshot(ed, snap, WANT)
    assert _values(rows[0]) == pytest.approx(_expected('log', 'percentile'),
                                             rel=1e-9)
