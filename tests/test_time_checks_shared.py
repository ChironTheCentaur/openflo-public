"""The remaining time-channel checks: which ask `is_time_channel`, and which
keep an exact 'Time' on purpose.

Asking the shared rule (so 'Time (s)', 'HDR-T Time', 'TIMESTAMP' count):
  - the auto-clean "Viability channel" menu leaves time columns out;
  - the FlowJo lower-edge check skips an edge at or below 0 on a time axis
    (nothing lies below 0 s);
  - the FCS loader never decodes a time channel's $PnE / $PnG.

Exact 'Time' (only a channel named Time, in any case):
  - `scale_time_coords` ($TIMESTEP scaling: a file with TIME_MSW and
    TIME_LSW would scale both), and the export checks that must agree with
    it (`_touches_time`, `_time_upper_edge`);
  - the FlowJo axis rule (`_flowjo_axis_is_linear`, the linear axes
    `_emit_axes` writes): measured on a channel named Time only.

Every file and gate here is generated; channel names are synthetic.
"""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from openflo import pipeline as fp
from openflo.channel_names import is_time_channel
from openflo.fcs_text import scale_time_coords
from openflo.pipeline import FlowSample
from openflo.wsp_writer import (
    _edges_flowjo_opens,
    _flowjo_axis_is_linear,
    _time_upper_edge,
    _touches_time,
)
from tests._fcs_writer import write_fcs

OTHER_TIME_NAMES = ['Time (s)', 'HDR-T Time', 'TIMESTAMP', 'TIME_MSW', 'Time-A']
EXACT_TIME_NAMES = ['Time', 'TIME', 'time']


# --- switched: the auto-clean viability-channel menu ----------------------

@pytest.mark.parametrize('name', EXACT_TIME_NAMES + OTHER_TIME_NAMES)
def test_viability_menu_leaves_out_every_time_name(name):
    pytest.importorskip('tkinter')
    from openflo.editor_tree import _viability_menu_skips
    assert _viability_menu_skips(name)


@pytest.mark.parametrize('name', ['FL1-A', 'FL2-H', 'LIFETIME', 'FL3TIME'])
def test_viability_menu_keeps_fluorescence(name):
    pytest.importorskip('tkinter')
    from openflo.editor_tree import _viability_menu_skips
    assert not _viability_menu_skips(name)


# --- switched: the FlowJo lower-edge check --------------------------------

@pytest.mark.parametrize('name', EXACT_TIME_NAMES + OTHER_TIME_NAMES)
def test_lower_edge_at_zero_on_a_time_axis_is_not_flagged(name):
    # Marked linear, a lower edge of 0 sits below FlowJo's linear floor; on a
    # time axis nothing lies below it, so there is nothing to report.
    g = {'kind': 'interval', 'channel': name, 'lo': 0.0, 'hi': 500.0}
    assert _edges_flowjo_opens(g, {name: 'LIN'}) == []


def test_lower_edge_at_zero_on_fluorescence_is_flagged():
    g = {'kind': 'interval', 'channel': 'FL1-A', 'lo': 0.0, 'hi': 500.0}
    assert _edges_flowjo_opens(g, {'FL1-A': 'LIN'}) == [('FL1-A', 0.0, True)]


def test_lower_edge_above_zero_on_a_time_axis_is_still_checked():
    g = {'kind': 'interval', 'channel': 'Time (s)', 'lo': 10.0, 'hi': 500.0}
    assert _edges_flowjo_opens(g, {'Time (s)': 'LIN'}) == [
        ('Time (s)', 10.0, True)]


# --- switched: the FCS $PnE / $PnG decode ---------------------------------

_STORED = np.array([[100, 0, 100, 1],
                    [200, 256, 200, 2],
                    [300, 512, 300, 3],
                    [400, 768, 400, 4],
                    [500, 1023, 500, 5]])


@pytest.mark.parametrize('time_name', ['Time', 'Time (s)', 'HDR-T Time'])
def test_time_channel_is_never_decoded(tmp_path, time_name):
    # FL1-A: 4 decades; the time column: 2 decades and gain 2, which a
    # loader that decoded it would turn into 10**(2*t/1024) / ... values.
    names = ['FSC-A', 'FL1-A', 'FL2-A', time_name]
    path = write_fcs(tmp_path / 'synthetic_1.fcs', _STORED, names,
                     pne=['0,0', '4,1', '0,0', '2,1'], png=[1, 1, 1, 2])
    s = FlowSample(str(path))
    np.testing.assert_array_equal(s.raw[time_name], _STORED[:, 3])
    np.testing.assert_allclose(s.raw['FL1-A'],
                               10.0 ** (4 * _STORED[:, 1] / 1024), rtol=1e-12)
    assert s.fcs_decoding == {'FL1-A': 'log, 4 decades'}


# --- kept exact: $TIMESTEP scaling and the checks paired with it ----------

def _gates(name):
    return [
        {'kind': 'interval', 'channel': name, 'lo': 100.0, 'hi': 900.0},
        {'kind': 'threshold', 'channel': name, 'value': 300.0},
        {'kind': 'rect', 'x_channel': name, 'y_channel': 'FL1-A',
         'x0': 100.0, 'x1': 900.0, 'y0': 1.0, 'y1': 5.0},
        {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': name,
         'vertices': [[1.0, 100.0], [5.0, 100.0], [5.0, 900.0]]},
    ]


@pytest.mark.parametrize('name', EXACT_TIME_NAMES)
def test_timestep_scales_a_gate_on_time(name):
    iv, th, rect, poly = (scale_time_coords(g, 0.01) for g in _gates(name))
    assert (iv['lo'], iv['hi']) == pytest.approx((1.0, 9.0))
    assert th['value'] == pytest.approx(3.0)
    assert (rect['x0'], rect['x1'], rect['y0'], rect['y1']) == pytest.approx(
        (1.0, 9.0, 1.0, 5.0))
    assert np.allclose(poly['vertices'], [[1.0, 1.0], [5.0, 1.0], [5.0, 9.0]])


@pytest.mark.parametrize('name', OTHER_TIME_NAMES)
def test_timestep_leaves_other_time_names_alone(name):
    for g in _gates(name):
        assert scale_time_coords(g, 0.01) is g


@pytest.mark.parametrize('name', OTHER_TIME_NAMES)
def test_time_edge_checks_skip_other_time_names(name):
    for g in _gates(name):
        assert not _touches_time(g)
        assert _time_upper_edge(g) is None


@pytest.mark.parametrize('name', EXACT_TIME_NAMES)
def test_time_edge_checks_read_a_gate_on_time(name):
    for g in _gates(name):
        assert _touches_time(g)
    assert [_time_upper_edge(g) for g in _gates(name)] == [
        900.0, None, 900.0, 900.0]


def test_touches_time_agrees_with_timestep_scaling_on_generated_gates():
    """`_touches_time(g)` is True exactly when scale_time_coords changes g:
    the edge check and the no-$TIMESTEP warning talk about the conversion."""
    rng = random.Random(5)
    names = EXACT_TIME_NAMES + OTHER_TIME_NAMES + ['FL1-A', 'FL2-A', 'SSC-A']
    for _ in range(300):
        g = rng.choice(_gates(rng.choice(names)))
        if 'y_channel' in g and rng.random() < 0.5:
            g = dict(g, y_channel=rng.choice(names))
        assert _touches_time(g) == (scale_time_coords(g, 0.01) is not g), g


# --- kept exact: the FlowJo axis rule -------------------------------------

@pytest.mark.parametrize('name', EXACT_TIME_NAMES)
def test_flowjo_shows_time_linear_whatever_its_display(name):
    assert _flowjo_axis_is_linear(name)
    assert _flowjo_axis_is_linear(name, {name: 'LOG'})


@pytest.mark.parametrize('name', OTHER_TIME_NAMES)
def test_flowjo_axis_of_other_time_names_follows_display(name):
    assert not _flowjo_axis_is_linear(name)
    assert not _flowjo_axis_is_linear(name, {name: 'LOG'})
    assert _flowjo_axis_is_linear(name, {name: 'LIN'})


def _linear_axes(tmp_path, time_name):
    """Channels the exported workspace declares a linear axis for, with one
    polygon on (time_name, FL1-A)."""
    gate = {'kind': 'polygon', 'x_channel': time_name, 'y_channel': 'FL1-A',
            'vertices': [[10.0, 1.0], [90.0, 1.0], [90.0, 50.0]],
            'id': 'p', 'parent_id': None, 'name': 'P'}
    w = fp.WspWriter()
    w.add_sample('s', '', ['FSC-A', 'FL1-A', time_name], [gate])
    out = tmp_path / 'out.wsp'
    w.write(str(out))
    block = next(el for el in ET.parse(out).getroot().iter()
                 if el.tag == 'Transformations')
    return {v for lin in block if lin.tag.endswith('}linear')
            for p in lin.iter() if p.tag.endswith('}parameter')
            for k, v in p.attrib.items() if k.endswith('}name')}


def test_export_leaves_the_time_axis_to_flowjo(tmp_path):
    assert _linear_axes(tmp_path, 'Time') == {'FL1-A'}


@pytest.mark.parametrize('name', ['Time (s)', 'HDR-T Time'])
def test_export_writes_a_linear_axis_for_other_time_names(tmp_path, name):
    assert _linear_axes(tmp_path, name) == {'FL1-A', name}


def test_names_used_here_are_all_time_channels():
    for name in EXACT_TIME_NAMES + OTHER_TIME_NAMES:
        assert is_time_channel(name)
