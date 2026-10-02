"""The histogram says how many events it could not place.

A log axis cannot place a non-positive value at all, and the curve looks
exactly the same whether it omitted none of them or a third of the sample. On a
compensated channel the non-positive population routinely IS a third.

The count is drawn on the axes rather than the status line, so it survives into
an exported panel — and it reports only what the SCALE cannot represent. The
view range is a robust percentile, so "outside the plotted range" would fire on
every histogram at a fraction of a percent and mean nothing.

Driven through the real `_plot_histogram`, not a copy of it.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

from tests.test_gui_smoke import _editor_or_skip  # noqa: E402

N = 20_000
NOTE = 'cannot be shown on a log axis'


def _load(editor, name, values, channel='CD3'):
    from types import SimpleNamespace
    df = pd.DataFrame({channel: values})
    s = SimpleNamespace(name=name, path=f'C:/x/{name}.fcs', data=df,
                        fluor_channels=[channel], channel_labels={})
    editor._samples[name] = s
    editor._sample_order.append(name)
    editor._sample_colors[name] = '#1f77b4'
    editor._sample_trial[name] = 'T'
    editor._sample_plot_enabled[name] = True
    editor._sample_gates[name] = {}
    editor._sample_gate_order[name] = []
    editor._sample_gate_seq[name] = 0


def _notes(editor):
    return [t.get_text() for t in editor.ax.texts if NOTE in t.get_text()]


def _draw(values, scale):
    root, ed, _gui = _editor_or_skip()
    try:
        _load(ed, 'a', values)
        ed._channels = ['CD3']
        ed.apply_gates_var.set(False)
        ed._channel_scale['CD3'] = scale
        ed.ax.clear()
        ed._plot_histogram(['a'], 'CD3')
        return _notes(ed)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


@pytest.fixture
def compensated():
    """A third of the events are non-positive — a log axis cannot place them."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(-300.0, 600.0, N // 3),
                           rng.normal(500.0, 900.0, N // 3),
                           rng.normal(9_000.0, 3_000.0, N - 2 * (N // 3))])


def test_a_log_axis_reports_the_events_it_cannot_place(compensated):
    notes = _draw(compensated, 'log')
    assert len(notes) == 1, (
        f'expected one off-scale note, got {notes}')
    assert 'events' in notes[0] and '%' in notes[0]


def test_a_linear_axis_says_nothing(compensated):
    """On a linear axis every finite value is displayable, so there is nothing
    to report and the note must not appear."""
    assert _draw(compensated, 'linear') == []


def test_an_all_positive_channel_on_a_log_axis_says_nothing():
    """The note must not fire on ordinary data, or it is decoration. The
    percentile view still clips a fraction of a percent at each tail — that is
    by design and is NOT what this reports."""
    rng = np.random.default_rng(3)
    assert _draw(rng.lognormal(6.0, 1.0, N), 'log') == []


def test_the_count_is_exactly_the_non_positive_population(compensated):
    note = _draw(compensated, 'log')[0]
    reported = int(note.split(' of ')[0].replace(',', ''))
    assert reported == int((compensated <= 0).sum())
