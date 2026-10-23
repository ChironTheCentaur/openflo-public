"""Panel check (openflo.panel_check): the predicted similarity matrix,
complexity index, flagged pairs and spillover of a planned panel.

Ground truth is computed here independently with numpy on synthetic
signatures: Gaussian emission bands on an invented detector set, exact
duplicates (similarity 1, a flagged pair, an infinite condition number) and
non-overlapping bands (the identity). The model's band integrals are checked
against the closed form of a split Gaussian (erf). The catalogue tests read
the fluor names from fluor_catalog at run time.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from openflo import fluor_catalog
from openflo import panel_check as pc
from openflo.spectral import (
    spectral_condition_number,
    spectral_similarity_matrix,
    unmixing_qc,
)

CENTRES = np.arange(400.0, 800.0, 20.0)                  # 20 detectors
DETS = [f"Det{k + 1}" for k in range(len(CENTRES))]


def _gauss(peak, sd):
    return np.exp(-0.5 * ((CENTRES - peak) / sd) ** 2)


def _cosine(S):
    U = S / np.linalg.norm(S, axis=1, keepdims=True)
    return U @ U.T


def _cond(S):
    sv = np.linalg.svd(S, compute_uv=False)
    return sv.max() / sv.min()


# ── similarity and complexity ───────────────────────────────────────────────

def test_gaussian_signatures_match_independent_numpy():
    S = np.vstack([_gauss(470, 25), _gauss(490, 28), _gauss(560, 25),
                   _gauss(650, 40)])
    fl = ['DyeA', 'DyeB', 'DyeC', 'DyeD']
    r = pc.check_panel(S, fl, DETS, threshold=0.8)
    np.testing.assert_allclose(r['similarity'], _cosine(S), atol=1e-12)
    assert r['condition_number'] == pytest.approx(_cond(S), rel=1e-10)
    assert r['condition_number'] == pytest.approx(np.linalg.cond(S),
                                                  rel=1e-10)
    # Flagged exactly where numpy's cosine exceeds the threshold.
    C = _cosine(S)
    want = sorted(((fl[i], fl[j]) for i in range(4) for j in range(i + 1, 4)
                   if C[i, j] >= 0.8), key=lambda p: -C[fl.index(p[0]),
                                                       fl.index(p[1])])
    got = [(d['fluor_a'], d['fluor_b']) for d in r['similar_pairs']]
    assert got == want and got            # DyeA/DyeB overlap that much
    for d in r['similar_pairs']:
        i, j = fl.index(d['fluor_a']), fl.index(d['fluor_b'])
        assert d['similarity'] == pytest.approx(C[i, j], abs=1e-12)


def test_numbers_agree_with_spectral_qc_on_the_same_matrix():
    rng = np.random.default_rng(3)
    S = rng.uniform(0.0, 1.0, (6, len(DETS)))
    fl = [f"Dye{c}" for c in 'ABCDEF']
    r = pc.check_panel(S, fl, DETS, threshold=0.8)
    np.testing.assert_array_equal(r['similarity'],
                                  spectral_similarity_matrix(S))
    assert r['condition_number'] == spectral_condition_number(S)
    assert r['similar_pairs'] == unmixing_qc({}, S, fl,
                                             sim_threshold=0.8)[
        'similar_pairs']


def test_identical_fluors_are_similarity_one_and_flagged():
    a = _gauss(520, 30)
    S = np.vstack([a, a, _gauss(700, 20)])
    r = pc.check_panel(S, ['DyeA', 'DyeB', 'DyeC'], DETS)
    assert r['similarity'][0, 1] == pytest.approx(1.0, abs=1e-12)
    assert [(d['fluor_a'], d['fluor_b']) for d in r['similar_pairs']] == \
        [('DyeA', 'DyeB')]
    assert r['condition_number'] == math.inf
    # Leaving either duplicate out makes the panel solvable again; leaving
    # the distinct one out does not.
    swap = {d['fluor']: d['condition_number'] for d in r['swap']}
    assert swap['DyeC'] == math.inf
    for f, keep in (('DyeA', [1, 2]), ('DyeB', [0, 2])):
        assert swap[f] == pytest.approx(_cond(S[keep]), rel=1e-10)
    assert r['swap'][-1]['fluor'] == 'DyeC'


def test_well_separated_fluors_give_the_identity():
    S = np.zeros((4, len(DETS)))
    for i in range(4):
        S[i, 5 * i:5 * i + 3] = [0.5, 1.0, 0.5]
    r = pc.check_panel(S, ['DyeA', 'DyeB', 'DyeC', 'DyeD'], DETS)
    np.testing.assert_allclose(r['similarity'], np.eye(4), atol=1e-12)
    assert r['condition_number'] == pytest.approx(1.0, rel=1e-12)
    assert r['similar_pairs'] == [] and r['shared_primary'] == []
    assert r['primary_detector'] == ['Det2', 'Det7', 'Det12', 'Det17']
    np.testing.assert_allclose(r['spillover'], np.eye(4), atol=1e-12)


def test_threshold_boundary():
    # cos(angle) = 0.95 exactly.
    t = math.acos(0.95)
    S = np.array([[1.0, 0.0, 0.0], [math.cos(t), math.sin(t), 0.0],
                  [0.0, 0.0, 1.0]])
    fl, dets = ['DyeA', 'DyeB', 'DyeC'], ['Det1', 'Det2', 'Det3']
    assert len(pc.check_panel(S, fl, dets, 0.94)['similar_pairs']) == 1
    assert pc.check_panel(S, fl, dets, 0.96)['similar_pairs'] == []


def test_swap_ranking_matches_numpy_deletions():
    S = np.vstack([_gauss(470, 25), _gauss(490, 25), _gauss(600, 25),
                   _gauss(700, 25)])
    fl = ['DyeA', 'DyeB', 'DyeC', 'DyeD']
    got = pc.swap_ranking(S, fl)
    want = sorted(((f, _cond(np.delete(S, i, axis=0)))
                   for i, f in enumerate(fl)), key=lambda p: p[1])
    assert [d['fluor'] for d in got] == [f for f, _ in want]
    for d, (_f, c) in zip(got, want, strict=True):
        assert d['condition_number'] == pytest.approx(c, rel=1e-10)
    assert got[0]['fluor'] in ('DyeA', 'DyeB')   # one of the close pair
    assert pc.swap_ranking(S[:2], fl[:2]) == []


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError):
        pc.check_panel(np.ones((2, 3)), ['DyeA'], ['Det1', 'Det2', 'Det3'])


# ── predicted spillover ─────────────────────────────────────────────────────

def test_predicted_spillover_by_hand():
    S = np.array([[2.0, 0.4, 0.0, 0.2],
                  [0.1, 1.0, 0.3, 0.0],
                  [0.0, 0.5, 0.2, 1.0],
                  [0.0, 0.0, 0.0, 0.0]])
    spill, primary = pc.predicted_spillover(S)
    assert list(primary) == [0, 1, 3, -1]
    want = np.array([[1.0, 0.2, 0.1, np.nan],
                     [0.1, 1.0, 0.0, np.nan],
                     [0.0, 0.5, 1.0, np.nan],
                     [np.nan] * 4])
    np.testing.assert_allclose(spill, want, atol=1e-12)


def test_shared_primary_detector_is_named():
    S = np.array([[1.0, 0.3, 0.0], [0.8, 0.5, 0.0], [0.0, 0.2, 1.0]])
    r = pc.check_panel(S, ['DyeA', 'DyeB', 'DyeC'], ['D1', 'D2', 'D3'])
    assert r['shared_primary'] == [('DyeA', 'DyeB', 'D1')]
    assert r['primary_detector'] == ['D1', 'D1', 'D3']


def test_matrix_csv_rows():
    S = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 0.0]])
    r = pc.check_panel(S, ['DyeA', 'DyeB', 'DyeC'], ['D1', 'D2'])
    rows = pc.matrix_csv_rows(r, 'similarity')
    assert rows[0] == ['', 'DyeA', 'DyeB', 'DyeC']
    assert rows[1] == ['DyeA', '1', '0.6', 'NA']
    assert rows[2] == ['DyeB', '0.6', '1', 'NA']
    assert rows[3] == ['DyeC', 'NA', 'NA', 'NA']
    assert r['degenerate_fluors'] == ['DyeC']
    sp = pc.matrix_csv_rows(r, 'spillover')
    # DyeB's primary is D2 (0.8): it puts 0.6/0.8 into DyeA's D1.
    assert sp[2] == ['DyeB', '0.75', '1', 'NA']


# ── spectra CSV ─────────────────────────────────────────────────────────────

def test_spectra_csv_round_trip(tmp_path):
    S = np.array([[0.2, 1.0, 0.5], [1.0, 0.1, 0.0]])
    path = tmp_path / 'sig.csv'
    pc.write_spectra_csv(path, S, ['DyeA', 'DyeB'], ['D1', 'D2', 'D3'])
    S2, fl, dets = pc.load_spectra_csv(path)
    assert fl == ['DyeA', 'DyeB'] and dets == ['D1', 'D2', 'D3']
    np.testing.assert_allclose(S2, S, atol=1e-6)


def test_spectra_csv_rows_are_max_normalised_and_any_delimiter():
    text = '﻿Fluor;D1;D2;D3\nDyeA;2;4;1\n\nDyeB;0;0.5;0.25\n'
    S, fl, dets = pc.parse_spectra_csv(text)
    assert fl == ['DyeA', 'DyeB'] and dets == ['D1', 'D2', 'D3']
    np.testing.assert_allclose(S, [[0.5, 1.0, 0.25], [0.0, 1.0, 0.5]])
    S, _fl, _d = pc.parse_spectra_csv('Fluor\tD1\tD2\nDyeA\t3\t1\n')
    np.testing.assert_allclose(S, [[1.0, 1 / 3]])


@pytest.mark.parametrize('text, says', [
    ('', 'empty'),
    ('Fluor,D1\nDyeA,1\n', 'at least two detector'),
    ('Fluor,D1,\nDyeA,1,2\n', 'no name'),
    ('Fluor,D1,D1\nDyeA,1,2\n', 'detector named twice: D1'),
    ('Fluor,D1,D2\n', 'no fluor rows'),
    ('Fluor,D1,D2\nDyeA,1\n', 'Line 2: 2 cells, but the header has 3'),
    ('Fluor,D1,D2\nDyeA,1,2\n,1,2\n', 'Line 3: the fluor has no name'),
    ('Fluor,D1,D2\nDyeA,1,2\nDyeA,2,1\n', 'Line 3: DyeA is listed twice'),
    ('Fluor,D1,D2\nDyeA,1,abc\n', "Line 2, D2: 'abc' is not a number"),
    ('Fluor,D1,D2\nDyeA,1,nan\n', 'Line 2, D2: not a finite'),
    ('Fluor,D1,D2\nDyeA,1,inf\n', 'Line 2, D2: not a finite'),
    ('Fluor,D1,D2\nDyeA,-0.5,1\n', 'Line 2, D1: negative value -0.5'),
    ('Fluor,D1,D2\nDyeA,0,0\n', 'Line 2: DyeA is zero in every detector'),
])
def test_spectra_csv_errors(text, says):
    with pytest.raises(pc.SpectraCSVError) as e:
        pc.parse_spectra_csv(text)
    assert says in str(e.value)
    assert isinstance(e.value, ValueError)


# ── the approximate model ──────────────────────────────────────────────────

def test_skewed_gaussian_peak_and_half_widths():
    peak, fwhm, skew = 600.0, 50.0, 1.5
    blue, red = fwhm / (1 + skew), fwhm * skew / (1 + skew)
    v = pc.skewed_gaussian(np.array([peak, peak - blue, peak + red,
                                     peak - 2 * blue]), peak, fwhm, skew)
    np.testing.assert_allclose(v, [1.0, 0.5, 0.5, 2.0 ** -4], rtol=1e-12)


def _split_band(lo, hi, peak, sd_b, sd_r):
    """Closed-form integral of a split Gaussian over [lo, hi]."""
    def part(a, b, sd):
        return sd * math.sqrt(math.pi / 2) * (
            math.erf((b - peak) / (sd * math.sqrt(2)))
            - math.erf((a - peak) / (sd * math.sqrt(2))))
    out = 0.0
    if lo < peak:
        out += part(lo, min(hi, peak), sd_b)
    if hi > peak:
        out += part(max(lo, peak), hi, sd_r)
    return out


@pytest.fixture
def dyes(monkeypatch):
    k = 2 * math.sqrt(2 * math.log(2))
    cat = {'DyeA': (((480, 1.0),), 40, 530, 50, ()),
           'DyeB': (((560, 1.0),), 40, 600, 40, ()),
           'DyeC': (((700, 1.0),), 30, 720, 30, ())}
    monkeypatch.setattr(fluor_catalog, 'FLUOR_SPECTRA', cat)
    return cat, k


def test_model_signature_matches_closed_form(dyes):
    cat, k = dyes
    dets = [('L1-a', 488.0, 520.0, 20.0), ('L1-b', 488.0, 560.0, 40.0),
            ('L2-a', 561.0, 610.0, 20.0), ('L1-c', 488.0, 610.0, 20.0)]
    for name in ('DyeA', 'DyeB'):
        (ex,), exw, em, emw, _x = cat[name]
        # Emission: red half-width 1.5x the blue; excitation: blue
        # half-width 1.8x the red (half-widths -> SD via 2/k).
        e_b, e_r = emw / 2.5 * 2 / k, emw * 1.5 / 2.5 * 2 / k
        x_b, x_r = exw * 1.8 / 2.8 * 2 / k, exw / 2.8 * 2 / k
        total = math.sqrt(2 * math.pi) * (e_b + e_r) / 2
        raw = []
        for _n, laser, c, w in dets:
            sd = x_b if laser < ex[0] else x_r
            eff = math.exp(-0.5 * ((laser - ex[0]) / sd) ** 2)
            raw.append(eff * _split_band(c - w / 2, c + w / 2, em, e_b, e_r)
                       / total)
        want = np.array(raw) / max(raw)
        got = pc.model_signature(name, dets)
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=1e-6)


def test_unexcited_fluor_is_a_zero_signature_and_named(dyes):
    dets = [('L1-a', 488.0, 520.0, 20.0), ('L1-b', 488.0, 600.0, 40.0)]
    S = pc.signature_matrix(['DyeA', 'DyeC'], dets)
    assert not S[1].any() and S[0].max() == 1.0
    r = pc.check_panel(S, ['DyeA', 'DyeC'], ['L1-a', 'L1-b'])
    assert r['degenerate_fluors'] == ['DyeC']
    assert r['condition_number'] == math.inf
    with pytest.raises(KeyError):
        pc.model_signature('DyeZ', dets)
    assert pc.catalog_fluors() == ['DyeA', 'DyeB', 'DyeC']


# ── the catalogue and the detector sets ─────────────────────────────────────

def test_every_catalogued_fluor_is_well_formed_and_detected():
    dets = pc.DETECTOR_SETS[pc.DEFAULT_SET]['detectors']
    names = list(fluor_catalog.FLUOR_SPECTRA)
    assert len(names) >= 40
    for name, (ex, exw, em, emw, extra) in \
            fluor_catalog.FLUOR_SPECTRA.items():
        assert ex and all(300 <= p <= 750 and 0 < h <= 1 for p, h in ex)
        assert max(h for _p, h in ex) == 1.0, name
        assert em > min(p for p, _h in ex), name          # Stokes shift
        assert 10 <= exw <= 150 and 10 <= emw <= 100, name
        assert all(0 < h < 1 and w > 0 for _p, h, w in extra), name
        sig = pc.model_signature(name, dets)
        assert sig.max() == pytest.approx(1.0), name
    # The full catalogue on the spectral set: no two entries are the same
    # signature unless their catalogue entries are the same numbers.
    S = pc.signature_matrix(names, dets)
    C = spectral_similarity_matrix(S)
    spec = fluor_catalog.FLUOR_SPECTRA
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            same = spec[names[i]] == spec[names[j]]
            assert (C[i, j] > 1 - 1e-12) == same, (names[i], names[j])


def test_detector_sets():
    assert pc.DEFAULT_SET in pc.DETECTOR_SETS
    kinds = {v['kind'] for v in pc.DETECTOR_SETS.values()}
    assert kinds == {'spectral', 'conventional'}
    assert set(pc.DEFAULT_THRESHOLD) == kinds
    assert len(pc.DETECTOR_SETS[pc.DEFAULT_SET]['detectors']) == 64
    for cfg in pc.DETECTOR_SETS.values():
        dets = cfg['detectors']
        assert len({d[0] for d in dets}) == len(dets)
        for _n, laser, c, w in dets:
            assert w > 0 and c - w / 2 > laser          # behind its laser
    # Contiguous spectral bands per laser.
    sp = [d for d in pc.DETECTOR_SETS[pc.DEFAULT_SET]['detectors']
          if d[1] == 405.0]
    assert len(sp) == 16
    for a, b in zip(sp, sp[1:], strict=False):
        assert a[2] + a[3] / 2 == pytest.approx(b[2] - b[3] / 2, abs=0.11)
