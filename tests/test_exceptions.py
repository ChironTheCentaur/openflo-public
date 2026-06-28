"""The OpenFlo exception hierarchy must be intact and behave as a
hierarchy (every domain error is also catchable as OpenFloError).
"""
import pytest

import openflo.pipeline as fp


@pytest.mark.parametrize('cls', [
    fp.FcsParseError,
    fp.CompensationError,
    fp.WspParseError,
    fp.GateError,
    fp.ClusteringError,
])
def test_subclass_of_openflo_error(cls):
    assert issubclass(cls, fp.OpenFloError)
    assert issubclass(cls, Exception)


def test_can_catch_specific_via_base():
    """A subclass must be catchable through its base — the hierarchy is
    the whole reason for OpenFloError's existence."""
    with pytest.raises(fp.OpenFloError, match='boom'):
        raise fp.CompensationError("boom")
