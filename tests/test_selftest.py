"""The end-user self-test (`openflo-selftest`) is also the suite's behavior
baseline: it runs the seeded synthetic data through the core feature paths and
checks each metric against the committed golden file. Here we assert the build
matches that baseline, and that the golden file and metric set stay in sync —
so the golden JSON is the single source of truth for both the CLI and pytest."""
from __future__ import annotations

from openflo import selftest


def test_selftest_all_metrics_pass():
    """Every golden metric reproduces within tolerance on this build. A failure
    here means a feature's behavior changed (or a backend is missing)."""
    results, ok = selftest.run_selftest()
    failed = [r['key'] for r in results if not r['passed']]
    assert ok, f"behavior drifted from golden baseline: {failed}\n" + \
        "\n".join(f"  {r['key']}: got {r['got']} exp {r['expected']}±{r['tol']}"
                  for r in results if not r['passed'])
    assert len(results) >= 7                       # all baseline metrics present


def test_golden_covers_every_computed_metric():
    """No computed metric is missing from the golden file, and no golden key is
    orphaned — they must match exactly so neither drifts unnoticed."""
    golden = set(selftest.load_golden())
    computed = {k for k in selftest.compute_metrics() if not k.startswith('_')}
    assert computed == golden, (
        f"golden vs computed mismatch — "
        f"only in golden: {golden - computed}; "
        f"only computed: {computed - golden}")


def test_selftest_main_exits_zero(capsys):
    rc = selftest.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert 'passed' in out and 'baseline' in out


def test_selftest_json_mode(capsys):
    import json
    rc = selftest.main(['--json'])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data['autoclean.debris_pct'] is not None
