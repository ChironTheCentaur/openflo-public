"""No test forks, except inside PhenoGraph.

A fork copies the process with only the calling thread, so a lock another
thread held at that instant (logging, numba, BLAS, Tk) stays held in the
child and it can hang. OpenFlo's own pools spawn, and since its PhenoGraph
calls run phenograph's pools spawned too (cluster_graph), nothing in OpenFlo
forks. A test that calls phenograph directly still gets phenograph's own
forked pools, so a fork whose call stack runs through phenograph is let
through and every other one fails the test.

Python 3.12+ warns about such forks, but CPython drops that warning without
a trace when a filter makes it an error, as pyproject's filters do, so the
warning cannot be what catches a new forking pool. This guard replaces
``os.fork`` for every test (tests/conftest.py) and works on every version.
"""
from __future__ import annotations

import sys


class ForkOutsidePhenoGraph(AssertionError):
    """A test forked from code other than phenograph's."""


def runs_through_phenograph(frame) -> bool:
    """Whether any frame from ``frame`` outwards is phenograph's code."""
    while frame is not None:
        name = frame.f_globals.get('__name__') or ''
        if name == 'phenograph' or name.startswith('phenograph.'):
            return True
        frame = frame.f_back
    return False


def guarded(real_fork):
    """``os.fork`` that forks only when called from within phenograph."""
    def fork():
        caller = sys._getframe(1)
        if not runs_through_phenograph(caller):
            raise ForkOutsidePhenoGraph(
                'os.fork() called outside phenograph: start processes with '
                "multiprocessing.get_context('spawn'), as OpenFlo's pools do.")
        return real_fork()
    fork.__wrapped__ = real_fork  # type: ignore[attr-defined]
    return fork
