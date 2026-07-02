"""CLI clustering-parity dispatch.

The run-engine method dispatch is covered elsewhere; this pins the CLI
worker's own branch (`_process_sample_task`) so the --cluster-method wiring
can't silently regress. The clustering backends are stubbed (no leidenalg /
igraph / flowsom deps required) — we're testing the dispatch, not the maths.
"""
from __future__ import annotations

import pytest

from openflo.cli import _process_sample_task
from openflo.pipeline import FlowSample


def _task(synthetic_fcs, **over):
    t = {'name': 's', 'key': 's', 'paths': [('o', synthetic_fcs)],
         'labels': {}, 'thresholds': {}, 'k': 15, 'n_jobs': 1,
         'max_events': 0, 'random_state': 42}
    t.update(over)
    return t


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch):
    """Replace the three clustering paths with markers so the dispatch is
    exercised without the heavy/optional clustering dependencies."""
    def fake_leiden(self, **kw):
        self.data['leiden'] = 0
    def fake_flowsom(self, **kw):
        self.data['flowsom_meta'] = 1
    def fake_cluster(self, **kw):
        self.data['cluster'] = 9
    monkeypatch.setattr(FlowSample, 'run_leiden', fake_leiden)
    monkeypatch.setattr(FlowSample, 'run_flowsom', fake_flowsom)
    monkeypatch.setattr(FlowSample, 'cluster', fake_cluster)


def test_dispatch_leiden(synthetic_fcs):
    key, s, err = _process_sample_task(_task(synthetic_fcs, method='leiden',
                                             resolution=2.0))
    assert err is None and s is not None
    assert 'leiden' in s.data.columns
    assert (s.data['cluster'] == s.data['leiden']).all()      # leiden → cluster


def test_dispatch_flowsom(synthetic_fcs):
    key, s, err = _process_sample_task(_task(synthetic_fcs, method='flowsom',
                                             n_metaclusters=8))
    assert err is None and s is not None
    assert 'flowsom_meta' in s.data.columns
    assert (s.data['cluster'] == s.data['flowsom_meta']).all()  # flowsom → cluster


def test_dispatch_phenograph_default(synthetic_fcs):
    key, s, err = _process_sample_task(_task(synthetic_fcs))   # no method = default
    assert err is None and s is not None
    assert (s.data['cluster'] == 9).all()                      # phenograph path
