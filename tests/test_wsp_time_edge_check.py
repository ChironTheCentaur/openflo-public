"""The FlowJo export warns about a Time gate whose edge was given in seconds.

OpenFlo keeps Time in ticks and multiplies by $TIMESTEP for FlowJo. A script
that converts to seconds first gets its edge written 1/$TIMESTEP times too
small. Generated FCS: 2000 events at Time 0, 5, ..., 9995 ticks with
$TIMESTEP 0.01 (a 99.95 s run)."""
from __future__ import annotations

import numpy as np
import pytest

from openflo import pipeline as fp

N = 2000


@pytest.fixture
def timed_fcs(tmp_path):
    import flowio
    rng = np.random.default_rng(1)
    ev = np.column_stack([rng.normal(5e4, 5e3, N), rng.normal(2e4, 2e3, N),
                          np.arange(N, dtype=float) * 5.0]).astype(np.float32)
    path = tmp_path / 'timed.fcs'
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(),
                          ['FSC-A', 'SSC-A', 'Time'],
                          metadata_dict={'TIMESTEP': '0.01'})
    return str(path)


def _warnings(fcs, tmp_path, gate):
    w = fp.WspWriter()
    w.add_sample('s1', fcs, ['FSC-A', 'SSC-A', 'Time'],
                 [dict(gate, id='t', parent_id=None, name='Time')])
    w.write(str(tmp_path / 'out.wsp'))
    return [m for m in w.warnings if 'Time gate' in m]


def test_edge_in_seconds_warns_with_the_events_it_cuts(timed_fcs, tmp_path):
    # The whole run meant, as seconds: [0, 100) s.
    msgs = _warnings(timed_fcs, tmp_path,
                     {'kind': 'interval', 'channel': 'Time', 'lo': 0,
                      'hi': 100})
    assert len(msgs) == 1
    # Ticks 100 .. 9995 are cut: 1980 of 2000 events.
    assert "cuts 1,980 of 2,000 events (99.0%)" in msgs[0]
    assert "ends at 100 ticks" in msgs[0]


@pytest.mark.parametrize('gate', [
    {'kind': 'interval', 'channel': 'Time', 'lo': 0, 'hi': 9000},  # trim end
    {'kind': 'interval', 'channel': 'Time', 'lo': 500, 'hi': 1e12},  # open
    {'kind': 'interval', 'channel': 'Time', 'lo': 0, 'hi': 1e6},  # past end
    {'kind': 'threshold', 'channel': 'Time', 'value': 50},  # lower edge only
    {'kind': 'rect', 'x_channel': 'Time', 'y_channel': 'FSC-A',
     'x0': 0, 'x1': 9500, 'y0': 0, 'y1': 1e5},
])
def test_ordinary_time_gates_do_not_warn(timed_fcs, tmp_path, gate):
    assert _warnings(timed_fcs, tmp_path, gate) == []


def test_two_dimensional_edge_in_seconds_warns(timed_fcs, tmp_path):
    msgs = _warnings(timed_fcs, tmp_path,
                     {'kind': 'polygon', 'x_channel': 'Time',
                      'y_channel': 'FSC-A',
                      'vertices': [[0, 0], [99, 0], [99, 1e5], [0, 1e5]]})
    assert len(msgs) == 1


def test_upper_edge_reading():
    assert fp._time_upper_edge({'channel': 'Time', 'lo': 1, 'hi': 7}) == 7
    assert fp._time_upper_edge({'channel': 'FSC-A', 'hi': 7}) is None
    assert fp._time_upper_edge({'channel': 'Time', 'lo': 1,
                                'hi': 1e12}) is None
    assert fp._time_upper_edge({'x_channel': 'FSC-A', 'y_channel': 'time',
                                'x0': 0, 'x1': 9, 'y0': 2, 'y1': 5}) == 5
