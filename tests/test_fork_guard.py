"""The suite's fork guard (tests/_fork_guard.py): a fork fails the test
unless phenograph made it. Probes are generated modules forking real pools
from a process with a second thread running, as in the GUI."""
import multiprocessing
import os
import threading

import pytest

from tests._fork_guard import ForkOutsidePhenoGraph, runs_through_phenograph

pytestmark = pytest.mark.skipif(not hasattr(os, 'fork'), reason='no fork here')


@pytest.fixture
def busy_thread():
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, daemon=True)
    t.start()
    yield
    stop.set()
    t.join()


def _pool_from(module_name):
    """A function defined in a module named ``module_name`` that opens a
    forked Pool and uses it."""
    ns = {'__name__': module_name, 'multiprocessing': multiprocessing}
    exec('def run():\n'
         '    with multiprocessing.get_context("fork").Pool(1) as p:\n'
         '        return p.map(abs, [-3])\n', ns)
    return ns['run']


def test_the_guard_is_on():
    assert getattr(os.fork, '__wrapped__', None) is not None


def test_an_openflo_fork_pool_fails_the_test(busy_thread):
    with pytest.raises(ForkOutsidePhenoGraph):
        _pool_from('openflo._fork_probe')()


def test_a_fork_from_a_test_fails_the_test(busy_thread):
    with pytest.raises(ForkOutsidePhenoGraph):
        _pool_from(__name__)()


def test_phenographs_own_fork_pool_is_let_through(busy_thread):
    assert _pool_from('phenograph._fork_probe')() == [3]


def test_only_phenograph_itself_counts():
    def frame_in(name):
        ns = {'__name__': name, 'sys': __import__('sys')}
        exec('def here():\n    return sys._getframe()\n', ns)
        return ns['here']()
    assert runs_through_phenograph(frame_in('phenograph'))
    assert runs_through_phenograph(frame_in('phenograph.cluster'))
    assert not runs_through_phenograph(frame_in('phenographx.cluster'))
    assert not runs_through_phenograph(frame_in('openflo.phenograph'))
