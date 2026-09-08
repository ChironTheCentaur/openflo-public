"""`openflo-doctor`'s behavioural gate.

`run_diagnostics(include_selftest=True)` is the DEFAULT and what the doctor
command runs, but every existing test passed `include_selftest=False` — so the
branch that decides whether a broken install is reported as broken had never
executed. A doctor that reports "healthy" while the seeded behaviour is wrong
is worse than no doctor: it converts a detectable problem into false
assurance.
"""
import openflo.diagnostics as dx


def _result(key='m', label='Metric', passed=True, got=1.0, expected=1.0,
            tol=0.0, unit=''):
    """A result dict with the REAL key set produced by selftest.run_selftest
    (key/label/unit/expected/tol/got/passed). Built deliberately rather than
    ad hoc: a fake that omits a key the consumer reads tests nothing, and this
    file exists because an untested consumer gave false assurance."""
    return {'key': key, 'label': label, 'unit': unit, 'expected': expected,
            'tol': tol, 'got': got, 'passed': passed}


def test_the_fake_matches_the_real_result_shape():
    """Guard the guard: if run_selftest's contract changes, these fakes must
    change with it rather than silently drifting out of sync."""
    from openflo.selftest import run_selftest
    real, _ok = run_selftest()
    assert real, 'run_selftest produced no results'
    assert set(real[0]) == set(_result()), (
        f'fake shape {sorted(_result())} no longer matches real '
        f'{sorted(real[0])}')


def _fake_selftest(monkeypatch, ok, results=None):
    import openflo.selftest as st
    monkeypatch.setattr(
        st, 'run_selftest',
        lambda: (results if results is not None else [], ok))


def test_a_failing_selftest_makes_the_install_unhealthy(monkeypatch):
    """The gate: behaviour drift must flip `ok`, not just add a note."""
    _fake_selftest(monkeypatch, ok=False,
                   results=[_result(key='x', label='X', passed=False,
                                    got=1, expected=2)])
    report = dx.run_diagnostics(include_selftest=True)
    assert report['selftest'] is not None, 'the self-test did not run'
    assert report['selftest']['ok'] is False
    assert report['ok'] is False, (
        'diagnostics reported a healthy install while the seeded behaviour '
        'test was failing — false assurance')


def test_a_passing_selftest_leaves_the_verdict_to_the_dependencies(monkeypatch):
    _fake_selftest(monkeypatch, ok=True)
    report = dx.run_diagnostics(include_selftest=True)
    assert report['selftest']['ok'] is True
    missing = [d for d in report['dependencies'] if d['status'] == 'missing']
    assert report['ok'] == (not missing)


def test_a_crashing_selftest_is_reported_not_raised(monkeypatch):
    """A broken native backend must produce a diagnosis, not a traceback —
    diagnostics is the tool you reach for when things are already broken."""
    import openflo.selftest as st

    def boom():
        raise RuntimeError('logicle backend missing')
    monkeypatch.setattr(st, 'run_selftest', boom)

    report = dx.run_diagnostics(include_selftest=True)
    assert report['selftest']['available'] is False
    assert 'logicle backend missing' in report['selftest']['error']
    assert report['ok'] is False, 'a crashing self-test must not read healthy'


def test_skipping_the_selftest_does_not_run_it(monkeypatch):
    """--quick must actually skip the expensive part."""
    import openflo.selftest as st
    calls = []
    monkeypatch.setattr(st, 'run_selftest',
                        lambda: calls.append(1) or ([], True))
    report = dx.run_diagnostics(include_selftest=False)
    assert calls == [], 'the self-test ran despite being skipped'
    assert report['selftest'] is None


def test_the_report_renders_a_failing_selftest(monkeypatch):
    """format_report must not hide the failure it was handed."""
    _fake_selftest(monkeypatch, ok=False,
                   results=[_result(key='autoclean.debris_pct',
                                    label='Auto-clean debris removed',
                                    passed=False, got=9.9, expected=7.05,
                                    tol=1e-9, unit='%')])
    text = dx.format_report(dx.run_diagnostics(include_selftest=True))
    assert 'Auto-clean debris removed' in text, (
        'the failing metric is absent from the rendered report')


def test_the_real_selftest_passes_through_diagnostics():
    """End to end, unmocked: the shipped build must be healthy by its own
    gate — the assurance `openflo-doctor` actually gives a user."""
    report = dx.run_diagnostics(include_selftest=True)
    assert report['selftest']['available'] is True
    assert report['selftest']['ok'] is True, (
        f"the shipped build fails its own behavioural gate: "
        f"{[r for r in report['selftest']['results'] if not r['passed']]}")
