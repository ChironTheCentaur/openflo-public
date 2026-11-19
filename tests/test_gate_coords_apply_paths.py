"""Direct path tests for gate_coords.apply and its per-kind helpers (R6 slice
A2): the edges the dialog tests do not reach. Generated gates on generic
channels; FL2-A is stored as logicle, FL1-A is linear."""
import numpy as np
import pytest

from openflo import gate_coords as gc
from openflo.pipeline import transform_spec, transform_values

LOG = transform_spec('logicle')
TR = {'FL2-A': LOG, 'FL3-A': LOG}
INTERVAL = {'kind': 'interval', 'channel': 'FL2-A', 'lo': 0.2, 'hi': 0.8}
RECT = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
        'x0': 10.0, 'x1': 500.0, 'y0': 0.1, 'y1': 0.9}
QUAD = dict(RECT, quad_set='qs1', quad_origin_x=300.0, quad_origin_y=0.4)
POLY = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
        'vertices': [[10.0, 0.2], [500.0, 0.3], [250.0, 0.9]]}
ELLIPSE = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
           'mean': [100.0, 0.5], 'cov': [[400.0, 1.0], [1.0, 0.01]]}


def _logicle(v):
    return float(np.asarray(transform_values(np.array([float(v)]),
                                             'logicle')).ravel()[0])


def test_interval_with_from_equal_to_below_is_refused():
    """'from' must be strictly below 'below': equal bounds are refused."""
    with pytest.raises(ValueError, match="'from' must be below 'below'"):
        gc.apply(INTERVAL, {'lo': '2K', 'hi': '2K'}, TR)


def test_rect_empty_below_opens_upward():
    """An empty x 'below' is the open upper side, not the lower one."""
    g = gc.apply(RECT, {'x0': '100', 'x1': '', 'y0': '', 'y1': '1e3'}, TR)
    assert g['x1'] == 1e12 and g['y0'] == -1e12
    assert g['x0'] == 100.0


def test_quadrant_y_divider_uses_the_y_channels_scale():
    """The y divider is converted on the y channel's (logicle) scale, the x
    divider on the x channel's (linear) one; the corners are left alone."""
    g = gc.apply(QUAD, {'quad_origin_x': '250', 'quad_origin_y': '1000'}, TR)
    assert g['quad_origin_x'] == 250.0
    assert g['quad_origin_y'] == pytest.approx(_logicle(1000))
    assert (g['x0'], g['y0']) == (QUAD['x0'], QUAD['y0'])


@pytest.mark.parametrize('text, message', [
    ('0, 0, 0\n1, 1\n2, 2', "line 1: give 'x, y'"),
    ('0, 0\nopen, 1\n2, 2', "line 2: 'open' is not a number"),
], ids=['three-numbers', 'open'])
def test_polygon_line_errors(text, message):
    """A vertex line needs exactly two numbers; 'open' is not one."""
    with pytest.raises(ValueError) as err:
        gc.apply(POLY, {'vertices': text}, TR)
    assert str(err.value) == message


def test_ellipse_shape_defaults_to_distance_sq_4():
    """A gate without distance_sq is reshaped on the 4.0 boundary."""
    g = gc.apply(ELLIPSE, {'a': '10', 'b': '0.05', 'angle': '0'}, TR)
    assert np.allclose(g['cov'], [[25.0, 0.0], [0.0, 0.000625]])
    g = gc.apply(dict(ELLIPSE, distance_sq=1.0),
                 {'a': '10', 'b': '0.05', 'angle': '0'}, TR)
    assert np.allclose(g['cov'], [[100.0, 0.0], [0.0, 0.0025]])


def test_ellipse_angle_turns_the_shape():
    """A typed angle of 90 swaps the axes of the covariance."""
    g = gc.apply(ELLIPSE, {'a': '10', 'b': '0.05', 'angle': '90'}, TR)
    assert np.allclose(g['cov'], [[0.000625, 0.0], [0.0, 25.0]])


def test_interval_from_is_converted_on_the_channels_scale():
    """A typed 'from' on a logicle channel is stored as its logicle value."""
    g = gc.apply(INTERVAL, {'lo': '1000', 'hi': ''}, TR)
    assert g['lo'] == pytest.approx(_logicle(1000))
    assert g['hi'] == 1e12


def test_rect_y_from_is_converted_on_the_y_channels_scale():
    """y 'from' uses the y channel's (logicle) scale, not the x channel's."""
    g = gc.apply(RECT, {'y0': '1000', 'y1': ''}, TR)
    assert g['y0'] == pytest.approx(_logicle(1000))


def test_rect_empty_from_opens_downward():
    """An empty x 'from' is the open lower side."""
    g = gc.apply(RECT, {'x0': ''}, TR)
    assert g['x0'] == -1e12 and g['x1'] == RECT['x1']


def test_rect_equal_y_bounds_are_refused():
    """y 'from' must be strictly below y 'below'."""
    with pytest.raises(ValueError, match="each 'from' must be below"):
        gc.apply(RECT, {'y0': '5', 'y1': '5'}, TR)


def test_polygon_whitespace_only_line_is_skipped():
    """A line of spaces is skipped like an empty one, and does not shift the
    line numbers in error messages."""
    g = gc.apply(POLY, {'vertices': '0, 0\n   \n100, 1\n50, 2'}, TR)
    assert len(g['vertices']) == 3
    with pytest.raises(ValueError, match="line 4: give 'x, y'"):
        gc.apply(POLY, {'vertices': '0, 0\n   \n100, 1\n50'}, TR)


def test_polygon_open_in_capitals_names_the_line():
    """'OPEN' is refused like 'open', with the line number."""
    with pytest.raises(ValueError) as err:
        gc.apply(POLY, {'vertices': '0, 0\nOPEN, 1\n2, 2'}, TR)
    assert str(err.value) == "line 2: 'OPEN' is not a number"


def test_ellipse_empty_centre_names_the_centre():
    """An empty center field says which value is missing."""
    with pytest.raises(ValueError, match='The center needs a value'):
        gc.apply(ELLIPSE, {'cy': ''}, TR)


@pytest.mark.parametrize('typed', [
    {'a': '10', 'b': '0', 'angle': '0'},
    {'a': '10', 'b': '0.05', 'angle': ''},
], ids=['zero-half-height', 'empty-angle'])
def test_ellipse_bad_shape_text_is_refused(typed):
    """A zero half-height or an empty angle is a ValueError, not a shape."""
    with pytest.raises(ValueError, match='half-width and half-height must be'):
        gc.apply(ELLIPSE, typed, TR)


def test_ellipse_negative_angle_and_text_distance_sq():
    """A negative angle is a valid orientation; distance_sq stored as text
    is read as a number."""
    g = gc.apply(ELLIPSE, {'a': '10', 'b': '0.05', 'angle': '-45'}, TR)
    assert np.allclose(g['cov'], gc.ellipse_cov(10.0, 0.05, -45.0, 4.0))
    g = gc.apply(dict(ELLIPSE, distance_sq='1'),
                 {'a': '10', 'b': '0.05', 'angle': '0'}, TR)
    assert np.allclose(g['cov'], [[100.0, 0.0], [0.0, 0.0025]])


class _Recording(dict):
    """A transforms mapping that records every channel looked up."""

    def __init__(self, *a):
        super().__init__(*a)
        self.looked = []

    def get(self, key, default=None):
        """Record ``key``, then look it up."""
        self.looked.append(key)
        return super().get(key, default)


@pytest.mark.parametrize('gate, typed', [
    ({'kind': 'threshold', 'channel': 'FL2-A', 'value': 0.5}, {'value': '1K'}),
    (INTERVAL, {'lo': '', 'hi': '2K'}),
], ids=['threshold', 'interval'])
def test_one_d_gates_look_up_only_their_channel(gate, typed):
    """The 2-D axis lookups (x_channel/y_channel) run only past the 1-D
    returns: a threshold or interval asks the transforms for its own
    channel only."""
    rec = _Recording(TR)
    gc.apply(gate, typed, rec)
    assert rec.looked and set(rec.looked) == {'FL2-A'}
