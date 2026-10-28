"""openflo-selftest paths tests/test_selftest.py does not run (selftest.main
58 %, compute_metrics 50 %): the --update gate that refuses to re-pin while a
response control fails, --update / --force rewriting values but keeping each
tolerance and label, and a crashing metric group being reported as a failed
row instead of aborting the run.

The golden file is redirected to a tmp copy, so the tracked _golden.json is
never written.
"""
from __future__ import annotations

import json
import shutil

import pytest

from openflo import selftest, selftest_controls


@pytest.fixture
def golden(tmp_path, monkeypatch):
    p = tmp_path / '_golden.json'
    shutil.copy(selftest._GOLDEN, p)
    monkeypatch.setattr(selftest, '_GOLDEN', str(p))
    return p


def _shifted():
    m = {k: v['value'] for k, v in selftest.load_golden().items()}
    return {k: v + 1.5 for k, v in m.items()}


def test_update_refuses_while_a_response_control_fails(golden, monkeypatch,
                                                       capsys):
    before = golden.read_bytes()
    monkeypatch.setattr(selftest_controls, 'run_controls',
                        lambda: [('ctl.x', False, 1.0, 2.0, 'a control')])
    monkeypatch.setattr(selftest, 'compute_metrics', _shifted)
    assert selftest.main(['--update']) == 1
    assert 'REFUSING to update' in capsys.readouterr().out
    assert golden.read_bytes() == before


@pytest.mark.parametrize('argv, controls_ok', [
    (['--update'], True), (['--update', '--force'], False)])
def test_update_repins_values_and_keeps_tolerances(golden, monkeypatch,
                                                   argv, controls_ok):
    old = json.loads(golden.read_text(encoding='utf-8'))
    monkeypatch.setattr(selftest_controls, 'run_controls',
                        lambda: [('ctl.x', controls_ok, 1.0, 1.0, 'c')])
    monkeypatch.setattr(selftest, 'compute_metrics', _shifted)
    assert selftest.main(argv) == 0
    new = json.loads(golden.read_text(encoding='utf-8'))
    assert new['_comment'] == old['_comment']
    for k, spec in old['metrics'].items():
        assert new['metrics'][k]['value'] == pytest.approx(spec['value'] + 1.5)
        assert new['metrics'][k]['tol'] == spec['tol']
        assert new['metrics'][k]['label'] == spec['label']


def test_a_crashing_metric_group_fails_its_rows_only(golden, monkeypatch,
                                                     capsys):
    def boom():
        raise RuntimeError('backend missing')
    boom.__name__ = '_cluster_metric'
    groups = tuple(boom if g.__name__ == '_cluster_metric' else g
                   for g in selftest._METRIC_GROUPS)
    monkeypatch.setattr(selftest, '_METRIC_GROUPS', groups)
    m = selftest.compute_metrics()
    assert m['cluster.leiden_n'] is None
    assert m['_error._cluster_metric'] == 'backend missing'
    assert m['calibration.slope'] is not None          # other groups ran
    assert selftest.main([]) == 1
    out = capsys.readouterr().out
    assert '7/8 passed' in out and 'BEHAVIOR CHANGED' in out
