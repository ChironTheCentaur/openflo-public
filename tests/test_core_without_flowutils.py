"""The transform and gating core runs, with the same numbers, where FlowUtils
cannot be imported (its compiled extension does not build in a browser
runtime).

FlowUtils is hidden by putting None in ``sys.modules`` for it, which makes
``import flowutils`` raise ImportError, and fresh copies of
``channel_transforms``, ``transform_sidecar`` and ``gate_eval`` are imported
under that; the copies loaded normally, with FlowUtils, are the reference.
Everything is restored afterwards.

A FlowUtils that is installed but broken (a compiled extension that fails to
load) is a different case: it must not be quietly replaced by the ports. The
transforms fail to import, as they always did, and a gate still evaluates on
the numpy test but says, at warning level, that the install is broken.
"""
import importlib
import importlib.abc
import logging
import sys

import numpy as np
import pandas as pd
import pytest

import openflo
from openflo import biexp_np, channel_transforms, gate_eval, transform_sidecar
from tests.conftest import import_flowutils_unless_browser

# The reference the copies are held to: a broken install fails here rather
# than skipping the file; only a browser runtime has none.
import_flowutils_unless_browser()

_HIDDEN = ('flowutils', 'flowutils.transforms', 'flowutils.gating',
           'flowutils.gating_c', 'flowutils.logicle_c')
_MODULES = ('channel_transforms', 'transform_sidecar', 'gate_eval')


@pytest.fixture
def no_flowutils(monkeypatch, caplog):
    """Fresh imports of the three modules with FlowUtils unimportable."""
    # The reference gate_eval resolves its backend on first use: have it do
    # so while FlowUtils is still importable.
    assert gate_eval.points_in_polygon([[0, 0], [2, 0], [0, 2]],
                                       [[0.5, 0.5]]).all()
    for name in _HIDDEN:
        monkeypatch.setitem(sys.modules, name, None)
    for mod in _MODULES:
        full = f'openflo.{mod}'
        monkeypatch.setitem(sys.modules, full, sys.modules[full])
        monkeypatch.setattr(openflo, mod, getattr(openflo, mod))
        del sys.modules[full]
    caplog.set_level(logging.INFO, logger='openflo.pipeline')
    return {mod: importlib.import_module(f'openflo.{mod}') for mod in _MODULES}


def test_flowutils_is_really_hidden(no_flowutils):
    with pytest.raises(ImportError):
        importlib.import_module('flowutils')
    assert no_flowutils['channel_transforms'].transforms is biexp_np
    assert no_flowutils['transform_sidecar'].transforms is biexp_np
    assert no_flowutils['gate_eval'] is not gate_eval
    assert channel_transforms.transforms.__name__ == 'flowutils.transforms'


def _values():
    rng = np.random.default_rng(7)
    return np.concatenate([rng.normal(0, 3000, 50_000),
                           rng.lognormal(8, 2, 50_000),
                           [0.0, -1.0, 1.0, 262144.0, np.nan, np.inf]])


@pytest.mark.parametrize('method', ['logicle', 'hyperlog'])
@pytest.mark.parametrize('params', [{}, {'a': 0.7, 'w': 0.8}])
def test_transform_values_unchanged(no_flowutils, method, params):
    fresh = no_flowutils['channel_transforms']
    v = _values()
    want = channel_transforms.transform_values(v, method, **params)
    got = fresh.transform_values(v, method, **params)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    assert np.nanmax(np.abs(got - want)) <= 1e-12
    s = np.linspace(-0.5, 1.2, 20_001)
    want_inv = channel_transforms.inverse_transform_values(s, method, **params)
    got_inv = fresh.inverse_transform_values(s, method, **params)
    np.testing.assert_allclose(got_inv, want_inv, rtol=1e-12, atol=1e-12)


def test_logicle_display_bounds_unchanged(no_flowutils):
    fresh = no_flowutils['transform_sidecar']
    want = transform_sidecar.logicle_display_bounds()
    got = fresh.logicle_display_bounds()
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_points_in_polygon_unchanged(no_flowutils, caplog):
    fresh = no_flowutils['gate_eval']
    rng = np.random.default_rng(11)
    pts = np.round(rng.uniform(-1, 11, size=(100_000, 2)))
    for verts in ([[0, 0], [10, 0], [10, 10], [0, 10]],
                  [[5, 0], [10, 5], [5, 10], [0, 5]],
                  [[0, 0], [10, 10], [10, 0], [0, 10]]):
        want = gate_eval.points_in_polygon(verts, pts)
        got = fresh.points_in_polygon(verts, pts)
        assert want.any()
        np.testing.assert_array_equal(got, want)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    notes = [r for r in caplog.records if 'flowutils is not installed' in
             r.getMessage()]
    assert len(notes) == 1 and notes[0].levelno == logging.INFO


def test_ellipsoid_gate_unchanged(no_flowutils, caplog):
    fresh = no_flowutils['gate_eval']
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 1, size=(50_000, 2)),
                      columns=['FL1-A', 'FL2-A'])
    gate = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
            'mean': [0.2, -0.1], 'cov': [[1.0, 0.3], [0.3, 0.5]],
            'distance_sq': 2.0}
    want = gate_eval.gate_to_mask(gate, df)
    got = fresh.gate_to_mask(gate, df)
    assert 0 < want.sum() < len(df)
    np.testing.assert_array_equal(got, want)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_exact_boundary_points_are_inside(no_flowutils):
    # Points with d^2 exactly equal to distance_sq (all arithmetic exact):
    # the ellipsoid is closed, with FlowUtils and without it.
    fresh = no_flowutils['gate_eval']
    df = pd.DataFrame({'FL1-A': [2.0, 0.0, -2.0, 0.0, 2.0, 0.0],
                       'FL2-A': [0.0, 1.0, 0.0, -1.0, 1.0, 1.5]})
    gate = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
            'mean': [0.0, 0.0], 'cov': [[4.0, 0.0], [0.0, 1.0]],
            'distance_sq': 1.0}
    want = [True, True, True, True, False, False]
    np.testing.assert_array_equal(gate_eval.gate_to_mask(gate, df), want)
    np.testing.assert_array_equal(fresh.gate_to_mask(gate, df), want)


class _BrokenExtension(importlib.abc.MetaPathFinder):
    """Makes FlowUtils' compiled extensions fail to load, as a missing or
    ABI-broken .so does, while the ``flowutils`` package itself is found."""

    def __init__(self, error):
        self.error = error

    def find_spec(self, name, path=None, target=None):
        if name in ('flowutils.logicle_c', 'flowutils.gating_c'):
            if self.error == 'missing':
                raise ModuleNotFoundError(f'No module named {name!r}',
                                          name=name)
            if self.error == 'abi':
                raise ImportError('undefined symbol: PyFloat_Type', name=name)
            if self.error == 'dependency':
                raise ModuleNotFoundError("No module named 'fu_dependency'",
                                          name='fu_dependency')
            raise RuntimeError('extension failed to initialise')
        return


@pytest.fixture(params=['missing', 'abi', 'dependency', 'runtime'])
def broken_flowutils(request, monkeypatch, caplog):
    """FlowUtils installed but broken; fresh imports of the three modules
    are left to the test (they may raise)."""
    for name in [k for k in sys.modules if k == 'flowutils'
                 or k.startswith('flowutils.')]:
        monkeypatch.delitem(sys.modules, name)
    for mod in _MODULES:
        full = f'openflo.{mod}'
        monkeypatch.setitem(sys.modules, full, sys.modules[full])
        monkeypatch.setattr(openflo, mod, getattr(openflo, mod))
    monkeypatch.setattr(sys, 'meta_path',
                        [_BrokenExtension(request.param), *sys.meta_path])
    caplog.set_level(logging.INFO, logger='openflo.pipeline')
    return request.param


@pytest.mark.parametrize('mod', ['channel_transforms', 'transform_sidecar'])
def test_broken_flowutils_is_not_replaced_by_the_port(broken_flowutils, mod):
    for name in ('openflo.channel_transforms', 'openflo.transform_sidecar'):
        del sys.modules[name]
    error = RuntimeError if broken_flowutils == 'runtime' else ImportError
    with pytest.raises(error):
        importlib.import_module(f'openflo.{mod}')


def test_broken_flowutils_gate_warns_and_still_evaluates(broken_flowutils,
                                                         caplog):
    del sys.modules['openflo.gate_eval']
    fresh = importlib.import_module('openflo.gate_eval')
    got = fresh.points_in_polygon([[0, 0], [2, 0], [0, 2]],
                                  [[0.5, 0.5], [3.0, 3.0]])
    np.testing.assert_array_equal(got, [True, False])
    df = pd.DataFrame({'FL1-A': [0.0, 5.0], 'FL2-A': [0.0, 5.0]})
    gate = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
            'mean': [0.0, 0.0], 'cov': [[1.0, 0.0], [0.0, 1.0]],
            'distance_sq': 1.0}
    np.testing.assert_array_equal(fresh.gate_to_mask(gate, df), [True, False])
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all('install broken' in w for w in warnings)
    assert not [r for r in caplog.records
                if 'flowutils is not installed' in r.getMessage()]
