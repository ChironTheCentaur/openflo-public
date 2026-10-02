"""Cosine similarity against a zero vector is undefined, not zero.

`spectral_similarity_matrix` normalises each reference spectrum by its norm and
substituted 1.0 when that norm was 0. A degenerate spectrum then came back as
0.0 against every other fluor — which on this scale means "maximally distinct,
trivially unmixable", the most reassuring answer available, for a fluor that
cannot be unmixed at all.

It is reachable from ordinary data: `build_reference_spectra` subtracts
autofluorescence and clips at zero, so a single stain dimmer than the unstained
control in every detector becomes an all-zero row.

The consequence in the report a user reads: `unmixing_qc` lists `similar_pairs`
where similarity is at or above the threshold, and a fluor whose similarities
are all 0.0 can never appear — so the panel is declared free of problematic
pairs precisely because one of its fluors has no usable spectrum.

`spectral_condition_number` was already honest about the same matrix (it
returns inf), which is why this shipped: the report carried one true signal
beside the false one.
"""
import numpy as np
import pytest

from openflo.spectral import (
    spectral_condition_number,
    spectral_similarity_matrix,
    unmixing_qc,
)

FLUORS = ['FITC', 'PE', 'APC']

USABLE = np.array([[0.8, 0.5, 0.1, 0.00],
                   [0.3, 0.9, 0.2, 0.05],
                   [0.1, 0.2, 0.9, 0.30]])

DEGENERATE = np.array([[0.8, 0.5, 0.1, 0.00],
                       [0.3, 0.9, 0.2, 0.05],
                       [0.0, 0.0, 0.0, 0.00]])     # clipped to nothing


def test_a_zero_norm_spectrum_has_no_similarity():
    sim = spectral_similarity_matrix(DEGENERATE)
    assert np.isnan(sim[2]).all(), (
        f'the degenerate fluor reported similarities {sim[2]}')
    assert np.isnan(sim[:, 2]).all()


def test_it_is_not_reported_as_maximally_distinct():
    """0.0 is the *best* value on this scale — the specific wrong answer."""
    sim = spectral_similarity_matrix(DEGENERATE)
    assert not (sim[2, 0] == 0.0), (
        'an unusable spectrum was called perfectly distinguishable'
    )


def test_usable_spectra_are_unaffected():
    """The guard must not disturb a normal panel."""
    good = spectral_similarity_matrix(USABLE)
    assert np.isfinite(good).all()
    assert np.allclose(np.diag(good), 1.0)
    assert good[0, 1] == pytest.approx(good[1, 0])
    # and the usable rows of a mixed panel keep exactly the same values
    mixed = spectral_similarity_matrix(DEGENERATE)
    assert mixed[0, 1] == pytest.approx(good[0, 1], abs=1e-12)


def test_the_diagonal_is_one_only_where_there_is_a_spectrum():
    sim = spectral_similarity_matrix(DEGENERATE)
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[1, 1] == pytest.approx(1.0)
    assert np.isnan(sim[2, 2]), (
        'a fluor with no spectrum was reported as perfectly similar to itself'
    )


def test_the_condition_number_still_says_the_matrix_is_singular():
    """The signal that was already honest must stay honest."""
    assert not np.isfinite(spectral_condition_number(DEGENERATE))


def test_the_qc_report_names_the_unusable_fluor():
    """A fluor that can never appear in similar_pairs must not be silently
    absent from the report."""
    single_stains = {f: np.zeros((10, 4)) for f in FLUORS}
    report = unmixing_qc(single_stains, DEGENERATE, FLUORS)
    assert report['degenerate_fluors'] == ['APC'], report['degenerate_fluors']


def test_a_healthy_panel_names_no_degenerate_fluor():
    single_stains = {f: np.zeros((10, 4)) for f in FLUORS}
    report = unmixing_qc(single_stains, USABLE, FLUORS)
    assert report['degenerate_fluors'] == []
