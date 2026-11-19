"""Shared pieces for the tests that drive the main editor with its background
jobs run inline (``editor``), register generated samples and gates in it,
and read figure preferences from a dict (``prefs``).

Generated data, synthetic sample names and generic channels only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from openflo import theme

sys.path.insert(0, os.path.dirname(__file__))


def _inline(widget, work, on_done=None, on_error=None, on_finally=None):
    try:
        r = work()
    except Exception as exc:                      # noqa: BLE001 - stands in for async_task.run_async, which hands any error from the work to on_error
        if on_error:
            on_error(exc)
    else:
        if on_done:
            on_done(r)
    finally:
        if on_finally:
            on_finally()


def _register(ed, s, gates):
    from test_statistics_export import _register as reg
    reg(ed, s, gates, list(gates))
    ed._trial_order[:] = list(dict.fromkeys(ed._trial_order))  # one trial


@pytest.fixture
def editor(monkeypatch):
    from test_gui_smoke import _editor_or_skip

    import openflo.async_task as _at
    monkeypatch.setattr(_at, 'run_async', _inline)
    root, ed, _gui = _editor_or_skip()
    yield ed
    try:
        ed.destroy()
        root.destroy()
    except Exception:                             # noqa: BLE001 - teardown; the root may already be destroyed, and a teardown error must not hide the test's own result
        pass


def _sample(name, vals):
    from openflo.pipeline import FlowSample
    return FlowSample.from_dataframe(
        pd.DataFrame({'FSC-A': np.linspace(1.0, 10.0, len(vals)),
                      'FL1-A': np.asarray(vals, dtype=float)}), name=name)


@pytest.fixture
def prefs(monkeypatch):
    """theme reads preferences from this dict instead of the user's file."""
    store = {'export_provenance': False}
    monkeypatch.setattr(theme, 'read_prefs', lambda: dict(store))
    return store


def _figure():
    from matplotlib.figure import Figure
    fig = Figure(figsize=(3, 2), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.set_title('GATELABEL')
    return fig
