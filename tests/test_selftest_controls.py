"""The response controls, under pytest — plus proof that they catch what the
golden baseline cannot.

``test_golden_is_blind_to_broken_compensation`` is the load-bearing test here.
It sabotages compensation two ways and asserts the golden stays GREEN while the
controls go RED. If someone ever "fixes" the controls by pinning expected
values, that test starts failing — which is the point.
"""
import numpy as np
import pytest

from openflo.selftest_controls import (
    _compensation_controls,
    _doublet_controls,
    run_controls,
)


def test_all_response_controls_pass():
    results = run_controls()
    assert results, 'no controls ran'
    failed = [(k, got, exp) for k, ok, got, exp, _lab in results if not ok]
    assert not failed, f'response controls failed: {failed}'


def test_doublet_removal_tracks_the_planted_fraction():
    """Isolated so a failure names the doublet path specifically."""
    (_key, ok, got, exp, _lab), = _doublet_controls()
    assert ok, f'doublet titration did not track: got {got}, expected {exp}'


def test_compensation_negative_control_and_titration():
    results = _compensation_controls()
    assert len(results) == 2
    for key, ok, got, exp, _lab in results:
        assert ok, f'{key}: got {got}, expected {exp}'


@pytest.mark.parametrize('mode', ['transposed', 'hardcoded'])
def test_golden_is_blind_to_broken_compensation(monkeypatch, mode):
    """The whole reason this module exists.

    ``transposed`` recreates the v2.2.1 bug class (right magnitudes, wrong
    orientation — it corrupted every real compensated analysis). ``hardcoded``
    is the degenerate case: ignore the input, return the expected answer.

    The golden baseline cannot see either, because its compensation metric
    round-trips the generator's own planted matrix through the CSV
    writer/reader and never invokes a compensation routine. The response
    controls must catch both.
    """
    import openflo.pipeline as P
    from openflo.synthetic import DEFAULT_SPILL_LEAKS
    real = P.optimize_compensation

    def transposed(channels, paths, **kw):
        chans, mat = real(channels, paths, **kw)
        return chans, np.asarray(mat, float).T

    def hardcoded(channels, paths, **kw):
        chans = list(channels)
        m = np.eye(len(chans))
        for (s, d), v in DEFAULT_SPILL_LEAKS.items():
            if s in chans and d in chans:
                m[chans.index(s), chans.index(d)] = v
        return chans, m

    monkeypatch.setattr(P, 'optimize_compensation',
                        transposed if mode == 'transposed' else hardcoded)

    # The golden baseline notices nothing.
    from openflo.selftest import run_selftest
    _results, golden_ok = run_selftest()
    assert golden_ok, ('precondition changed: the golden baseline now catches '
                       'broken compensation — if that is intentional, this '
                       'test should be updated, not deleted')

    # The response controls do.
    failed = [k for k, ok, *_ in _compensation_controls() if not ok]
    assert failed, (f'{mode} compensation went undetected by BOTH the golden '
                    'baseline and the response controls')


def test_clustering_controls_pass():
    """Purity on real structure, and collapse under permutation."""
    from openflo.selftest_controls import _clustering_controls
    for key, ok, got, exp, _lab in _clustering_controls():
        assert ok, f'{key}: got {got}, expected {exp}'


def test_ground_truth_labels_do_not_perturb_the_seeded_stream():
    """`return_labels` rides through the shuffle as an extra COLUMN, which a
    row-wise Fisher-Yates cannot notice. If someone rewrites that as an index
    permutation the RNG stream shifts and every 1e-9 golden metric silently
    re-baselines — this test is the tripwire."""
    from openflo.synthetic import immunophenotyping_sample
    for kwargs in ({'seed': 42}, {'seed': 3, 'batch_gain': 1.4},
                   {'seed': 3, 'fmo': 'CD4'}, {'seed': 7, 'group': 'treat'}):
        plain = immunophenotyping_sample(n=4000, **kwargs)
        withlab, labels = immunophenotyping_sample(n=4000, return_labels=True,
                                                   **kwargs)
        assert np.array_equal(plain.to_numpy(float), withlab.to_numpy(float)), \
            f'return_labels changed the data for {kwargs}'
        assert len(labels) == len(plain)


def test_ground_truth_labels_match_the_data():
    """A label is only useful if it names what the event actually is."""
    from openflo.synthetic import (
        PBMC_MARKER_DET,
        PBMC_POPS,
        immunophenotyping_sample,
    )
    df, truth = immunophenotyping_sample(n=8000, seed=42, return_labels=True)
    live = ~truth.isin(['dead', 'debris', 'doublet'])
    # Planted artefact fractions come back exactly.
    assert abs(float((truth == 'dead').mean()) - 0.08) < 1e-9
    assert abs(float((truth == 'debris').mean()) - 0.07) < 1e-9
    assert abs(float((truth == 'doublet').mean()) - 0.05) < 1e-9
    # Each population is bright for its own defining marker and dim for it
    # everywhere else.
    for name, (_frac, profile, _ssc) in PBMC_POPS.items():
        if not profile:
            continue
        det = PBMC_MARKER_DET[list(profile)[-1]]
        inside = df.loc[(truth == name).to_numpy(), det].median()
        outside = df.loc[(truth != name).to_numpy() & live.to_numpy(),
                         det].median()
        assert inside > 10 * outside, f'{name}: {inside} vs {outside}'


def test_compensation_applier_controls_pass():
    """The APPLIER controls (manual_compensate), distinct from the estimator."""
    from openflo.selftest_controls import _compensation_apply_controls
    for key, ok, got, exp, _lab in _compensation_apply_controls():
        assert ok, f'{key}: got {got}, expected {exp}'


def test_applier_controls_catch_the_transpose_bug(monkeypatch):
    """The 2.2.1 bug lived in the applier, which the estimator controls cannot
    see — that gap is exactly why these exist."""
    import openflo.pipeline as P
    from openflo.selftest_controls import _compensation_apply_controls
    real = P.FlowSample.manual_compensate
    monkeypatch.setattr(
        P.FlowSample, 'manual_compensate',
        lambda self, m, c: real(self, np.asarray(m, float).T, c))
    failed = [k for k, ok, *_ in _compensation_apply_controls() if not ok]
    assert failed, 'a transposed applier went undetected'


def test_golden_metric_now_catches_broken_compensation(monkeypatch):
    """The golden's compensation metric used to round-trip the generator's own
    matrix through a CSV and run no compensation code. It must now fail when
    compensation is broken."""
    import openflo.pipeline as P
    from openflo.selftest import run_selftest
    real = P.FlowSample.manual_compensate
    monkeypatch.setattr(
        P.FlowSample, 'manual_compensate',
        lambda self, m, c: real(self, np.asarray(m, float).T, c))
    _results, ok = run_selftest()
    assert not ok, ('the golden baseline is blind to a transposed compensation '
                    'matrix again — the metric has regressed to a round-trip')


def test_update_refuses_to_rebaseline_while_a_control_fails(tmp_path,
                                                            monkeypatch):
    """--update rewrites every pinned value from the current run, so a broken
    pipeline could bless itself. It must refuse while a relational control is
    failing."""
    import shutil

    import openflo.pipeline as P
    import openflo.selftest as S

    scratch = tmp_path / '_golden.json'
    shutil.copy(S._GOLDEN, scratch)
    before = scratch.read_text(encoding='utf-8')
    monkeypatch.setattr(S, '_GOLDEN', str(scratch))

    real = P.FlowSample.manual_compensate
    monkeypatch.setattr(
        P.FlowSample, 'manual_compensate',
        lambda self, m, c: real(self, np.asarray(m, float).T, c))

    rc = S.main(['--update'])
    assert rc == 1, '--update should exit non-zero while a control fails'
    assert scratch.read_text(encoding='utf-8') == before, \
        '--update rewrote the baseline despite a failing control'
