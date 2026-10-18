"""Batch normalization helpers around CytoNorm (openflo.batch_normalize).

Synthetic data only: generic FL1-A / FL2-A channels, samples named
batch1_ref, batch2_sampleA and so on."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from openflo import batch_normalize as bn

CHS = ['FL1-A', 'FL2-A']


def _events(rng, n=4000, shift=(0.0, 0.0), scale=(1.0, 1.0), extra=0.0):
    """Bimodal events on a transformed-like scale, with a batch shift and
    stretch per channel; `extra` adds a biological brightening."""
    cols = {}
    for j, ch in enumerate(CHS):
        base = np.concatenate([rng.normal(1.0, 0.15, n // 2),
                               rng.normal(2.5, 0.25, n - n // 2)]) + extra
        cols[ch] = base * scale[j] + shift[j]
    cols['FSC-A'] = rng.normal(50000, 5000, n)
    return pd.DataFrame(cols)


B2 = dict(shift=(0.6, 0.3), scale=(1.0, 1.2))


@pytest.fixture
def batches():
    rng = np.random.default_rng(3)
    frames = {
        'batch1_ref': _events(rng),
        'batch1_sampleA': _events(rng),
        'batch2_ref': _events(rng, **B2),
        # The same batch effect, plus a real biological difference that a
        # reference-anchored normalization must keep.
        'batch2_sampleA': _events(rng, extra=0.8, **B2),
    }
    batch_of = {n: n.split('_')[0] for n in frames}
    refs = {'batch1': 'batch1_ref', 'batch2': 'batch2_ref'}
    return frames, batch_of, refs


def _fit(frames, batch_of, refs=None, mode='controls', **kw):
    kw.setdefault('n_metaclusters', 1)
    kw.setdefault('grid', (3, 3))
    return bn.fit_batch_model(frames, batch_of, CHS, mode=mode,
                              references=refs, **kw)


def _med(df, ch):
    return float(np.median(df[ch]))


# ── grouping ────────────────────────────────────────────────────────────────

class _S:
    def __init__(self, md):
        self.metadata = md


def test_group_by_keyword_reads_the_loaders_keys_in_any_spelling():
    samples = {'batch1_ref': _S({'date': '01-JAN-2026'}),
               'batch1_sampleA': _S({'$DATE': '01-JAN-2026 '}),
               'batch2_ref': _S({'DATE': '02-JAN-2026'}),
               'batch3_sampleA': _S({'src': 'x'})}
    groups = bn.group_by_keyword(samples, '$DATE')
    assert groups == {'01-JAN-2026': ['batch1_ref', 'batch1_sampleA'],
                      '02-JAN-2026': ['batch2_ref'],
                      bn.MISSING_BATCH: ['batch3_sampleA']}
    assert bn.keyword_value({'date': ''}, 'DATE') is None


def test_batch_of_samples_inverts_and_refuses_a_sample_in_two_batches():
    assert bn.batch_of_samples({'b1': ['x', 'y'], 'b2': ['z']}) == {
        'x': 'b1', 'y': 'b1', 'z': 'b2'}
    with pytest.raises(ValueError, match='batch'):
        bn.batch_of_samples({'b1': ['x'], 'b2': ['x']})


# ── fit / apply ─────────────────────────────────────────────────────────────

def test_reference_mode_removes_a_known_batch_shift(batches):
    frames, batch_of, refs = batches
    model = _fit(frames, batch_of, refs)
    out = bn.apply_batch_model(model, frames, batch_of)
    for ch in CHS:
        before = abs(_med(frames['batch1_ref'], ch) - _med(frames['batch2_ref'], ch))
        after = abs(_med(out['batch1_ref'], ch) - _med(out['batch2_ref'], ch))
        assert before > 0.25 and after < 0.05, (ch, before, after)
        # The robust spread of the references agrees too (the stretch).
        s1, s2 = (bn.robust_spread(out[n][ch]) for n in ('batch1_ref', 'batch2_ref'))
        assert abs(s1 - s2) < 0.05 * s1
    # A non-reference sample in batch 2 goes through the SAME monotone
    # mapping as its reference: read the mapping off the reference (sorted
    # before -> sorted after) and it predicts the sample's corrected values.
    for ch in CHS:
        x = np.sort(frames['batch2_ref'][ch].to_numpy())
        y = np.sort(out['batch2_ref'][ch].to_numpy())
        v = frames['batch2_sampleA'][ch].to_numpy()
        inside = (v > x[40]) & (v < x[-40])
        assert inside.sum() > 1000
        pred = np.interp(v[inside], x, y)
        np.testing.assert_allclose(out['batch2_sampleA'][ch].to_numpy()[inside],
                                   pred, atol=0.03)
    # ... and its real brightening is kept, not normalized away.
    d1 = _med(out['batch2_sampleA'], 'FL1-A') - _med(out['batch2_ref'], 'FL1-A')
    assert d1 > 0.5


def test_reference_mode_fits_on_the_references_only(batches):
    """The batch-2 sample is brighter for a biological reason; fitting on it
    (goal mode) would pull batch 2's reference down, fitting on the
    references keeps them together."""
    frames, batch_of, refs = batches
    goal = bn.apply_batch_model(_fit(frames, batch_of, mode='goal'),
                                frames, batch_of)
    ctrl = bn.apply_batch_model(_fit(frames, batch_of, refs), frames, batch_of)
    gap = {k: abs(_med(o['batch1_ref'], 'FL1-A') - _med(o['batch2_ref'], 'FL1-A'))
           for k, o in (('goal', goal), ('ctrl', ctrl))}
    assert gap['ctrl'] < 0.05 < 0.15 < gap['goal']


def test_goal_mode_pools_every_sample_and_aligns_batches():
    rng = np.random.default_rng(5)
    frames = {'batch1_sampleA': _events(rng), 'batch1_sampleB': _events(rng),
              'batch2_sampleA': _events(rng, **B2),
              'batch2_sampleB': _events(rng, **B2)}
    batch_of = {n: n.split('_')[0] for n in frames}
    model = _fit(frames, batch_of, mode='goal')
    assert model.mode == 'goal'
    out = bn.apply_batch_model(model, frames, batch_of)
    for ch in CHS:
        assert abs(_med(out['batch1_sampleB'], ch)
                   - _med(out['batch2_sampleB'], ch)) < 0.05


def test_inputs_are_not_changed_and_other_columns_pass_through(batches):
    frames, batch_of, refs = batches
    kept = {n: f.copy(deep=True) for n, f in frames.items()}
    model = _fit(frames, batch_of, refs)
    out = bn.apply_batch_model(model, frames, batch_of)
    for n in frames:
        pd.testing.assert_frame_equal(frames[n], kept[n])
        assert out[n] is not frames[n]
        np.testing.assert_array_equal(out[n]['FSC-A'], frames[n]['FSC-A'])
        assert list(out[n].columns) == list(frames[n].columns)


def test_fit_is_reproducible_with_a_cap(batches):
    frames, batch_of, refs = batches
    a = _fit(frames, batch_of, refs, max_events_per_batch=1000).to_dict()
    b = _fit(frames, batch_of, refs, max_events_per_batch=1000).to_dict()
    assert a == b


def test_apply_refuses_an_unknown_batch_or_a_missing_channel(batches):
    frames, batch_of, refs = batches
    model = _fit(frames, batch_of, refs)
    with pytest.raises(KeyError, match='batch3'):
        bn.apply_batch_model(model, {'batch3_sampleA': frames['batch1_ref']},
                             {'batch3_sampleA': 'batch3'})
    with pytest.raises(ValueError, match='FL2-A'):
        bn.apply_batch_model(model, {'batch1_ref': frames['batch1_ref'][['FL1-A']]},
                             batch_of)


def test_fit_refuses_what_it_cannot_anchor(batches):
    frames, batch_of, refs = batches
    with pytest.raises(ValueError, match='no reference'):
        _fit(frames, batch_of, {'batch1': 'batch1_ref'})
    with pytest.raises(ValueError, match='not in batch'):
        _fit(frames, batch_of, {'batch1': 'batch1_ref', 'batch2': 'batch1_sampleA'})
    one = {n: f for n, f in frames.items() if n.startswith('batch1')}
    with pytest.raises(ValueError, match='at least 2'):
        _fit(one, batch_of, mode='goal')
    with pytest.raises(ValueError, match='lacks'):
        bn.fit_batch_model(frames, batch_of, ['FL1-A', 'FL9-A'], mode='goal')


# ── summary ─────────────────────────────────────────────────────────────────

def test_robust_spread_is_half_the_central_68_percent():
    v = np.r_[np.arange(101.0), np.nan]
    lo, hi = np.percentile(np.arange(101.0), [15.865, 84.135])
    assert bn.robust_spread(v) == pytest.approx(0.5 * (hi - lo))
    assert np.isnan(bn.robust_spread(np.array([np.nan])))


def test_summary_describes_each_batch_by_its_reference(batches):
    frames, batch_of, refs = batches
    after = {n: f + 1.0 for n, f in frames.items()}      # a known 'after'
    tab = bn.normalization_summary(frames, after, batch_of, CHS, refs)
    assert len(tab) == 4 and set(tab['basis']) == {'reference'}
    row = tab[(tab.channel == 'FL2-A') & (tab.batch == 'batch2')].iloc[0]
    ref = frames['batch2_ref']['FL2-A']
    assert row.median_before == pytest.approx(np.median(ref))
    assert row.median_after == pytest.approx(np.median(ref) + 1.0)
    assert row.spread_before == pytest.approx(bn.robust_spread(ref))
    assert row.spread_after == pytest.approx(bn.robust_spread(ref))
    assert row.n_events == len(ref)
    pooled = bn.normalization_summary(frames, after, batch_of, CHS)
    row = pooled[(pooled.channel == 'FL1-A') & (pooled.batch == 'batch2')].iloc[0]
    both = np.r_[frames['batch2_ref']['FL1-A'], frames['batch2_sampleA']['FL1-A']]
    assert row.basis == 'pooled' and row.median_before == pytest.approx(np.median(both))


def test_methods_text_states_what_was_done_and_the_gap(batches):
    frames, batch_of, refs = batches
    model = _fit(frames, batch_of, refs)
    out = bn.apply_batch_model(model, frames, batch_of)
    tab = bn.normalization_summary(frames, out, batch_of, CHS, refs)
    text = bn.methods_text(model, tab, refs)
    sd = tab.groupby('channel')[['median_before', 'median_after']].std(ddof=0)
    assert 'CytoNorm' in text and '2 batches' in text and '2 channel' in text
    assert 'batch1_ref' in text and 'single per-channel' in text
    assert f"from {sd['median_before'].mean():.3g} to " \
           f"{sd['median_after'].mean():.3g}" in text


# ── FCS copies ──────────────────────────────────────────────────────────────

def _write(path, rng, factor=1.0, date='01-JAN-2026', spill=None):
    from openflo.fcs_export import write_fcs
    n = 3000
    fl = np.exp(np.concatenate([rng.normal(6.5, 0.3, (n // 2, 2)),
                                rng.normal(9.0, 0.3, (n - n // 2, 2))]))
    df = pd.DataFrame({'FSC-A': rng.normal(60000, 4000, n),
                       'SSC-A': rng.normal(30000, 3000, n),
                       'FL1-A': fl[:, 0] * factor, 'FL2-A': fl[:, 1] * factor})
    kw = {'DATE': date}
    if spill:
        kw['SPILLOVER'] = spill
    write_fcs(df, str(path), keywords=kw)
    return str(path)


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


@pytest.fixture
def fcs_set(tmp_path):
    rng = np.random.default_rng(11)
    src = tmp_path / 'in'
    src.mkdir()
    paths = [_write(src / 'batch1_ref.fcs', rng),
             _write(src / 'batch1_sampleA.fcs', rng),
             _write(src / 'batch2_ref.fcs', rng, 1.6, '02-JAN-2026'),
             _write(src / 'batch2_sampleA.fcs', rng, 1.6, '02-JAN-2026')]
    return paths, tmp_path


def test_written_copies_hold_the_normalized_values_and_provenance(fcs_set):
    from openflo.pipeline import FlowSample, linear_values
    paths, tmp = fcs_set
    hashes = {p: _sha(p) for p in paths}
    samples = bn.load_samples(paths)
    assert samples['batch1_ref'].data_transforms['FL1-A']['method'] == 'logicle'
    batch_of = bn.batch_of_samples(bn.group_by_keyword(samples, '$DATE'))
    assert batch_of['batch2_sampleA'] == '02-JAN-2026'
    refs = {'01-JAN-2026': 'batch1_ref', '02-JAN-2026': 'batch2_ref'}
    frames = {n: s.data for n, s in samples.items()}
    model = bn.fit_batch_model(frames, batch_of, CHS, mode='controls',
                               references=refs, n_metaclusters=1, grid=(3, 3))
    out = bn.apply_batch_model(model, frames, batch_of)
    written = bn.write_normalized_copies(samples, out, batch_of, model,
                                         str(tmp / 'out'), refs)
    assert [os.path.basename(p) for p in written] == [
        'batch1_ref_norm.fcs', 'batch1_sampleA_norm.fcs',
        'batch2_ref_norm.fcs', 'batch2_sampleA_norm.fcs']
    assert all(_sha(p) == h for p, h in hashes.items())     # originals intact
    back = {os.path.basename(p): FlowSample(p) for p in written}
    b2 = back['batch2_sampleA_norm.fcs']
    import copy
    view = copy.copy(samples['batch2_sampleA'])
    view.data = out['batch2_sampleA']
    for ch in CHS + ['FSC-A']:
        np.testing.assert_allclose(b2.data[ch], linear_values(view, ch),
                                   rtol=1e-5, atol=1e-3)
    # Normalized linear medians of the two references agree; the files'
    # did not (a 1.6-fold batch effect).
    r1, r2 = back['batch1_ref_norm.fcs'], back['batch2_ref_norm.fcs']
    ratio = np.median(r2.data['FL1-A']) / np.median(r1.data['FL1-A'])
    assert abs(ratio - 1.0) < 0.05
    md = {k.lower(): v for k, v in b2.metadata.items()}
    assert md['openflo_normalization'] == 'CytoNorm'
    assert md['openflo_norm_mode'] == 'controls'
    assert md['openflo_norm_batch'] == '02-JAN-2026'
    assert md['openflo_norm_reference'] == 'batch2_ref'
    assert md['openflo_norm_channels'] == 'FL1-A,FL2-A'
    assert 'openflo_version' in md
    assert md['date'] == '02-JAN-2026'                        # source keywords


def test_copies_are_compensated_and_never_overwrite_a_source(tmp_path):
    rng = np.random.default_rng(2)
    spill = '2,FL1-A,FL2-A,1,0.2,0,1'
    paths = [_write(tmp_path / 'batch1_ref.fcs', rng, spill=spill),
             _write(tmp_path / 'batch2_ref.fcs', rng, 1.3, '02-JAN-2026', spill)]
    samples = bn.load_samples(paths)
    assert all(s.compensation_source == '$SPILL' for s in samples.values())
    batch_of = bn.batch_of_samples(bn.group_by_keyword(samples, 'date'))
    frames = {n: s.data for n, s in samples.items()}
    model = bn.fit_batch_model(frames, batch_of, CHS, mode='goal',
                               n_metaclusters=1, grid=(3, 3))
    out = bn.apply_batch_model(model, frames, batch_of)
    with pytest.raises(ValueError, match='overwrite'):
        bn.write_normalized_copies(samples, out, batch_of, model,
                                   str(tmp_path), suffix='')
    from openflo.pipeline import FlowSample
    p = bn.write_normalized_copies(samples, out, batch_of, model,
                                   str(tmp_path / 'out'))[0]
    md = {k.lower(): v for k, v in FlowSample(p).metadata.items()}
    assert 'openflo_applied_matrix' in md
    assert not any('spill' in k for k in md)
    assert md['openflo_norm_reference'] == 'none'
