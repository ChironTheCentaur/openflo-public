"""Reference spectra of dim dyes on cells whose autofluorescence varies.

`build_reference_spectra` kept each single-stain control's brightest 10% of
events by TOTAL signal and subtracted the unstained control's MEAN
autofluorescence (AF). For a dim dye on cells with heterogeneous AF
(lymphocytes beside myeloid cells with 10x their AF) the high-AF cells won
that selection, and subtracting the average AF left most of theirs in the
dye's spectrum: on the audit's 8-detector case (dye amplitude 400, 30% of
cells at 10x AF) the spectrum's cosine to the truth was 0.856, and the
unmixing built on it put the AF into the dye's channel.

'auto' (the default) now also selects positives by the dye's OWN excess in
its primary detector and subtracts each one's OWN AF -- the mean of the
unstained events matched to it in detectors where AF dominates the dye --
and uses that estimate only when it disagrees with the total-signal one
(cosine < 0.99) or the dye sits inside the AF band (separation index < 20).
Otherwise the total-signal spectrum is returned bit for bit, so nothing
that worked changes.

Ground truth: simulated controls with known dye and AF spectra, Poisson-like
plus electronic noise, and a mixed sample unmixed against the estimates.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest

from openflo.spectral import (
    build_reference_spectra,
    reference_spectra_markdown,
    unmix,
    unmixing_qc,
)

# ── the audit's 8-detector case ─────────────────────────────────────────────

_X8 = np.arange(8)
_DYE8 = np.exp(-0.5 * ((_X8 - 5) / 1.0) ** 2)
_AF8 = np.exp(-0.5 * ((_X8 - 1.5) / 2.0) ** 2) * 40 + 5


def _cells8(rng, amp, hi_frac=0.3, fold=10.0):
    n = len(amp)
    k = np.where(rng.random(n) < hi_frac, fold, 1.0)[:, None]
    sig = k * _AF8[None, :] + amp[:, None] * _DYE8[None, :]
    return sig + rng.normal(0, 1, sig.shape) * np.sqrt(np.clip(sig, 1, None))


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_dim_dye_on_heterogeneous_af_recovers_its_spectrum():
    """Amplitude 400, 30% of cells at 10x AF: the total-signal estimate is
    still the AF-contaminated 0.85; 'auto' recovers the dye (measured
    0.9993-1.0000 over 15 seeds and sizes)."""
    rng = np.random.default_rng(0)
    n = 5000
    amp = np.where(rng.random(n) < 0.5,
                   rng.lognormal(np.log(400), 0.3, n), 0.0)
    stain, un = _cells8(rng, amp), _cells8(rng, np.zeros(n))
    S_old, _ = build_reference_spectra({'Dim': stain}, unstained=un,
                                       method='total')
    diag = []
    S, fluors = build_reference_spectra(
        {'Dim': stain}, unstained=un, diagnostics=diag,
        detectors=[f'V{i + 1}' for i in range(8)])
    assert _cos(S_old[0], _DYE8) < 0.9            # the defect, still there
    assert _cos(S[0], _DYE8) > 0.98, S[0].round(3)
    # The AF endmember is unchanged: still the unstained control's mean.
    np.testing.assert_array_equal(S[-1], S_old[-1])
    assert fluors == ['Dim', 'Autofluorescence']
    d = diag[0]
    assert d['method'] == 'matched' and d['requested_method'] == 'auto'
    assert d['primary_detector'] == 'V6'           # the dye's peak, index 5
    assert d['agreement'] < 0.99 and 'disagrees' in d['reason']
    assert d['n_positives'] > 100 and d['warning'] is None


# ── a 16-detector panel for the grid ────────────────────────────────────────

D = 16
_X = np.arange(D)


def _gauss(mu, sd):
    g = np.exp(-0.5 * ((_X - mu) / sd) ** 2)
    return g / g.max()


AF_SHAPE = _gauss(2.0, 3.5) + 0.08
AF_SHAPE /= AF_SHAPE.max()
DYES = {'far': _gauss(8.5, 1.0),        # cosine to AF 0.21
        'under': _gauss(4.5, 2.7)}      # cosine to AF 0.90


def _af(rng, n, hi_frac, fold):
    k = rng.lognormal(0.0, 0.3, n) * 50.0
    k = np.where(rng.random(n) < hi_frac, k * fold, k)
    return k[:, None] * AF_SHAPE[None, :]


def _noisy(rng, sig):
    return (sig + rng.normal(0, 1, sig.shape) * np.sqrt(2 * np.clip(sig, 0, None))
            + rng.normal(0, 3, sig.shape))


def _control(rng, n, spec, bright, hi_frac, fold):
    amp = np.where(rng.random(n) < 0.5,
                   rng.lognormal(np.log(bright), 0.3, n), 0.0)
    return _noisy(rng, _af(rng, n, hi_frac, fold) + amp[:, None] * spec)


def _unmix_error(rng, S, spec, bright, hi_frac, fold, n=4000):
    """RMS error of the dye's unmixed abundance / its brightness, on a mixed
    sample with the same AF heterogeneity."""
    a = np.where(rng.random(n) < 0.5,
                 rng.lognormal(np.log(bright), 0.3, n), 0.0)
    Y = _noisy(rng, _af(rng, n, hi_frac, fold) + a[:, None] * spec)
    return float(np.sqrt(np.mean((unmix(Y, S)[:, 0] - a) ** 2)) / bright)


_GRID = [(b, f, fo, s) for b in (200, 400, 2000, 50000)
         for f, fo in ((0.0, 1), (0.3, 10), (0.1, 20), (0.5, 3))
         for s in ('far', 'under')]


def test_compact_grid_auto_is_never_worse_than_total():
    """A seeded slice of the validation grid (brightness x AF heterogeneity
    x dye/AF similarity; the full grid is in the commit message). In every
    cell 'auto' is no worse than 'total' -- cosine within 0.002, unmixing
    error within 2% -- and where 'total' is AF-contaminated (dim dye, 10x+
    AF) 'auto' is above 0.98."""
    worst, fixed = [], 0
    for i, (bright, frac, fold, sk) in enumerate(_GRID):
        rng = np.random.default_rng(1000 + i)
        spec = DYES[sk]
        stain = _control(rng, 3000, spec, bright, frac, fold)
        un = _noisy(rng, _af(rng, 3000, frac, fold))
        S_old, _ = build_reference_spectra({'F': stain}, unstained=un,
                                           method='total')
        S_new, _ = build_reference_spectra({'F': stain}, unstained=un)
        c_old, c_new = _cos(S_old[0], spec), _cos(S_new[0], spec)
        e_old = _unmix_error(np.random.default_rng(i), S_old, spec, bright,
                             frac, fold)
        e_new = _unmix_error(np.random.default_rng(i), S_new, spec, bright,
                             frac, fold)
        worst.append((c_new - c_old, e_new / e_old))
        assert c_new >= c_old - 0.002, (bright, frac, fold, sk, c_old, c_new)
        assert e_new <= 1.02 * e_old, (bright, frac, fold, sk, e_old, e_new)
        if bright <= 400 and fold >= 10 and c_old < 0.98:
            assert c_new > 0.98, (bright, frac, fold, sk, c_old, c_new)
            fixed += 1
    assert fixed >= 4, worst


def test_agreeing_estimates_inside_the_af_band_still_use_matched_af():
    """A dye of 2,000 under a 20x AF band (cosine 0.9 to AF): the two
    estimates agree to 0.996, yet the AF the total one keeps quintuples the
    unmixing error (grid: 0.130 against 0.025, floor 0.022). Below
    separation 20 'auto' takes the matched spectrum."""
    rng = np.random.default_rng(12)
    spec = DYES['under']
    stain = _control(rng, 3000, spec, 2000, 0.3, 20)
    un = _noisy(rng, _af(rng, 3000, 0.3, 20))
    S_old, _ = build_reference_spectra({'F': stain}, unstained=un,
                                       method='total')
    diag = []
    S, _ = build_reference_spectra({'F': stain}, unstained=un,
                                   diagnostics=diag)
    d = diag[0]
    assert d['agreement'] >= 0.99 and d['separation_index'] < 20
    assert d['method'] == 'matched' and 'AF spreads clear' in d['reason']
    e_old = _unmix_error(np.random.default_rng(1), S_old, spec, 2000, 0.3, 20)
    e_new = _unmix_error(np.random.default_rng(1), S, spec, 2000, 0.3, 20)
    assert e_new < 0.5 * e_old, (e_old, e_new)


@pytest.mark.parametrize('frac,fold', [(0.3, 10), (0.0, 1)])
def test_bright_dye_spectrum_is_identical_to_the_total_method(frac, fold):
    """The guardrail: where the two estimates agree (bright dye, or AF that
    does not vary), 'auto' returns the total-signal spectrum bit for bit."""
    rng = np.random.default_rng(3)
    stains = {s: _control(rng, 4000, spec, 20000, frac, fold)
              for s, spec in DYES.items()}
    un = _noisy(rng, _af(rng, 4000, frac, fold))
    S_old, f_old = build_reference_spectra(stains, unstained=un,
                                           method='total')
    diag = []
    S, fluors = build_reference_spectra(stains, unstained=un,
                                        diagnostics=diag)
    np.testing.assert_array_equal(S, S_old)
    assert fluors == f_old
    assert [d['method'] for d in diag] == ['total', 'total']
    assert all(('agree' in d['reason']) or ('varies too little' in d['reason'])
               for d in diag), [d['reason'] for d in diag]
    assert all(d['warning'] is None for d in diag)


def test_diagnostics_change_no_numbers():
    rng = np.random.default_rng(4)
    stain = _control(rng, 3000, DYES['far'], 400, 0.3, 10)
    un = _noisy(rng, _af(rng, 3000, 0.3, 10))
    a, _ = build_reference_spectra({'F': stain}, unstained=un)
    diag = []
    b, _ = build_reference_spectra({'F': stain}, unstained=un,
                                   diagnostics=diag)
    np.testing.assert_array_equal(a, b)
    d = diag[0]
    assert d['fluor'] == 'F' and d['n_events'] == 3000
    assert d['primary_detector'] in (8, 9) or d['primary_detector'] == 7
    assert d['separation_index'] > 3
    # The spectrum it returned looks like the dye, not like AF (cosine to
    # AF 0.21 for the true dye).
    assert d['af_cosine'] < 0.35


def test_too_dim_control_is_flagged_not_silently_returned(caplog):
    """A dye at 20 counts under a 20x AF band cannot define a spectrum; the
    report says so and recommends beads or a brighter fluor."""
    rng = np.random.default_rng(5)
    stain = _control(rng, 3000, DYES['under'], 20, 0.3, 20)
    un = _noisy(rng, _af(rng, 3000, 0.3, 20))
    good = _control(rng, 3000, DYES['far'], 20000, 0.3, 20)
    diag = []
    with caplog.at_level(logging.WARNING, logger='openflo.spectral'):
        build_reference_spectra({'Dim': stain, 'Bright': good}, unstained=un,
                                diagnostics=diag)
    by = {d['fluor']: d for d in diag}
    assert by['Dim']['warning'], by['Dim']
    assert by['Bright']['warning'] is None
    assert any('Dim' in r.getMessage() and 'beads' in r.getMessage()
               for r in caplog.records)
    md = '\n'.join(reference_spectra_markdown(
        unmixing_qc({}, np.eye(2), ['Dim', 'Bright'],
                    reference_spectra=diag)['reference_spectra']))
    assert 'too dim' in md and '**Dim**' in md and 'beads' in md


def test_method_total_beads_and_no_unstained_keep_the_total_estimate():
    rng = np.random.default_rng(6)
    stain = _control(rng, 3000, DYES['far'], 400, 0.3, 10)
    un = _noisy(rng, _af(rng, 3000, 0.3, 10))
    S_old, _ = build_reference_spectra({'F': stain}, unstained=un,
                                       method='total')
    for beads, method in ((True, 'auto'), (['F'], 'auto'), (False, 'total')):
        diag = []
        S, _ = build_reference_spectra({'F': stain}, unstained=un,
                                       diagnostics=diag, beads=beads,
                                       method=method)
        np.testing.assert_array_equal(S, S_old)
        assert diag[0]['method'] == 'total'
        assert ('bead' in diag[0]['reason']) == (beads is not False)
    # 'matched' forces the new estimate ...
    diag = []
    S_m, _ = build_reference_spectra({'F': stain}, unstained=un,
                                     method='matched', diagnostics=diag)
    assert diag[0]['method'] == 'matched'
    assert _cos(S_m[0], DYES['far']) > 0.98
    # ... and without an unstained control has nothing to match: it falls
    # back to the total-signal estimate, and says so.
    diag = []
    S_nu, _ = build_reference_spectra({'F': stain}, method='matched',
                                      diagnostics=diag)
    S_nu_old, _ = build_reference_spectra({'F': stain}, method='total')
    np.testing.assert_array_equal(S_nu, S_nu_old)
    assert diag[0]['method'] == 'total' and 'unstained' in diag[0]['reason']
    with pytest.raises(ValueError):
        build_reference_spectra({'F': stain}, method='brightest')


def test_matched_estimate_is_seeded_and_order_independent():
    rng = np.random.default_rng(7)
    a = _control(rng, 3000, DYES['far'], 400, 0.3, 10)
    b = _control(rng, 3000, DYES['under'], 400, 0.3, 10)
    un = _noisy(rng, _af(rng, 3000, 0.3, 10))
    S1, f1 = build_reference_spectra({'A': a, 'B': b}, unstained=un)
    S2, f2 = build_reference_spectra({'B': b, 'A': a}, unstained=un)
    np.testing.assert_array_equal(S1[0], S2[1])
    np.testing.assert_array_equal(S1[1], S2[0])


def test_unmixing_qc_report_is_json_safe():
    rng = np.random.default_rng(8)
    stain = _control(rng, 3000, DYES['far'], 400, 0.3, 10)
    un = _noisy(rng, _af(rng, 3000, 0.3, 10))
    diag = []
    S, fl = build_reference_spectra({'F': stain}, unstained=un,
                                    diagnostics=diag,
                                    detectors=[f'D{i}' for i in range(D)])
    qc = unmixing_qc({'F': stain}, S, fl, unstained=un,
                     reference_spectra=diag)
    rep = json.loads(json.dumps(qc['reference_spectra']))
    assert rep[0]['fluor'] == 'F' and rep[0]['method'] == 'matched'
    assert rep[0]['primary_detector'].startswith('D')
    assert unmixing_qc({}, S, fl)['reference_spectra'] == []


def test_cli_report_carries_the_reference_spectra(tmp_path):
    """--unmix writes how each spectrum was estimated into spectral_qc.json
    and .md, and names a control too dim to trust."""
    import types

    import flowio

    from openflo.cli import run_batch_unmix
    rng = np.random.default_rng(9)
    chans = [f'D{i + 1}' for i in range(D)]

    def write(name, arr):
        p = os.path.join(str(tmp_path), name)
        with open(p, 'wb') as f:
            flowio.create_fcs(f, np.asarray(arr, np.float32).flatten().tolist(),
                              chans)
        return p

    ctrls = {'Far': write('far.fcs', _control(rng, 3000, DYES['far'], 400,
                                              0.3, 10)),
             'Faint': write('faint.fcs', _control(rng, 3000, DYES['far'],
                                                  10, 0.3, 10)),
             'unstained': write('un.fcs', _noisy(rng, _af(rng, 3000, 0.3, 10)))}
    out = os.path.join(str(tmp_path), 'out')
    args = types.SimpleNamespace(
        unmix=True, out=out, unmix_controls=json.dumps(ctrls),
        unmix_input='', unmix_detectors=','.join(chans), unmix_nonneg=False,
        fcs='', trials='')
    assert run_batch_unmix(args) == 0
    with open(os.path.join(out, 'spectral_qc.json'), encoding='utf-8') as f:
        rep = json.load(f)
    by = {d['fluor']: d for d in rep['reference_spectra']}
    assert by['Far']['method'] == 'matched'
    assert by['Far']['primary_detector'] in chans
    assert 'Faint' in rep['dim_controls'] and 'Far' not in rep['dim_controls']
    md = open(os.path.join(out, 'spectral_qc.md'), encoding='utf-8').read()
    assert '## Reference spectra' in md and '**Faint**' in md
