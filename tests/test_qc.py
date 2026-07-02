"""Acquisition QC anomaly detectors.

Covers the three independent detectors in AcquisitionQC — signal drift,
flow-rate anomalies (clogs/bubbles), and margin/saturation events — plus
the combined run(). Pure helpers are exercised directly; run() is checked
on small synthetic frames with injected anomalies.
"""
import numpy as np
import pandas as pd

from openflo.pipeline import AcquisitionQC

# ── _margin_events ───────────────────────────────────────────────────────────

def test_margin_events_flags_ceiling_pileup():
    # 100 events; 5 piled at the ceiling 262143 on CH → above 1% frac.
    col = np.concatenate([np.linspace(0, 1000, 95), np.full(5, 262143.0)])
    df = pd.DataFrame({'CH': col})
    bad = AcquisitionQC._margin_events(df, ['CH'], frac=0.01)
    assert bad.sum() == 5
    assert bad[-5:].all()


def test_margin_events_ignores_single_offscale_max():
    # Continuous data: the lone max is not a pile-up → not flagged.
    df = pd.DataFrame({'CH': np.linspace(0, 1000, 100)})
    bad = AcquisitionQC._margin_events(df, ['CH'], frac=0.01)
    assert bad.sum() == 0


def test_margin_events_per_channel_union():
    a = np.concatenate([np.zeros(90), np.full(10, 5.0)])      # pile at 5
    b = np.concatenate([np.full(10, 9.0), np.linspace(0, 1, 90)])  # pile at 9
    df = pd.DataFrame({'A': a, 'B': b})
    bad = AcquisitionQC._margin_events(df, ['A', 'B'], frac=0.05)
    assert bad.sum() == 20      # 10 from each channel, disjoint rows


# ── _flowrate_bad_bins ───────────────────────────────────────────────────────

def test_flowrate_flags_empty_interior_bin():
    # bins 0..4 dense, bin 2 empty (a gap / bubble).
    bins = np.array([0]*50 + [1]*50 + [3]*50 + [4]*50)  # no events in bin 2
    qc = AcquisitionQC(pd.DataFrame({'x': range(len(bins))}))
    bad = qc._flowrate_bad_bins(bins, n_bins=5, threshold=5.0)
    assert 2 in bad


def test_flowrate_flags_count_outlier():
    # bin 3 has a huge burst (bubble surge) vs uniform ~50.
    bins = np.array([0]*50 + [1]*50 + [2]*50 + [3]*400 + [4]*50)
    qc = AcquisitionQC(pd.DataFrame({'x': range(len(bins))}))
    bad = qc._flowrate_bad_bins(bins, n_bins=5, threshold=5.0)
    assert 3 in bad


def test_flowrate_uniform_is_clean():
    bins = np.repeat(np.arange(10), 50)
    qc = AcquisitionQC(pd.DataFrame({'x': range(len(bins))}))
    assert qc._flowrate_bad_bins(bins, n_bins=10, threshold=5.0) == set()


# ── run(): integration ───────────────────────────────────────────────────────

def _timed_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'Time': np.linspace(0, 100, n),
        'FSC-A': rng.normal(1000, 50, n),
        'CD11b': rng.normal(500, 30, n),
    })


def test_run_clean_data_drops_nothing():
    df = _timed_frame()
    qc = AcquisitionQC(df)
    clean = qc.run(n_bins=50)
    assert len(clean) == len(df)
    assert qc.report['total'] == 0


def test_run_combines_detectors_and_reports():
    df = _timed_frame(n=2000)
    # Inject a saturation pile-up on CD11b in the last 50 rows.
    df.loc[df.index[-50:], 'CD11b'] = 2_000_000.0
    qc = AcquisitionQC(df)
    qc.run(n_bins=50, margin_frac=0.01)
    assert qc.report['margin'] == 50
    assert qc.report['total'] >= 50
    assert set(qc.report) >= {'drift', 'flow_rate', 'margin', 'total'}


def test_run_no_time_channel_still_does_margins():
    df = pd.DataFrame({'CD11b': np.concatenate(
        [np.linspace(0, 1, 950), np.full(50, 9e6)])})
    qc = AcquisitionQC(df)
    qc.run()
    assert qc.time_channel is None
    assert qc.report['margin'] == 50
    assert qc.report['flow_rate'] == 0


def test_run_drift_can_be_disabled():
    """The new drift= kwarg lets the autoclean gate toggle drift independently
    of flow-rate/margin (default True preserves existing callers)."""
    df = _timed_frame(n=2000)
    # Inject a drifted block in one time region on CD11b.
    df.loc[df.index[500:560], 'CD11b'] = 50_000.0
    on = AcquisitionQC(df).run(n_bins=50, flow_rate=False, margins=False)
    off = AcquisitionQC(df).run(n_bins=50, drift=False,
                                flow_rate=False, margins=False)
    assert len(on) < len(df)        # drift detector removed the block
    assert len(off) == len(df)      # disabled → nothing removed


# ── autoclean_keep_mask: the recipe gate ──────────────────────────────────────

def _clean_frame(seed=0):
    """A frame with a low-FSC debris mode, doublets, a Time channel, and a
    ceiling pile-up on APC-A."""
    r = np.random.default_rng(seed)
    n = 4000
    fsc = np.concatenate([r.normal(2e4, 3e3, n // 5),
                          r.normal(1.2e5, 1.5e4, 4 * n // 5)])
    fsch = fsc / 2 + r.normal(0, 2e3, fsc.size)
    fsch[:200] = fsc[:200] / 4                      # doublets (high A/H ratio)
    time = np.sort(r.uniform(0, 100, fsc.size))
    apc = r.normal(1000, 100, fsc.size)
    apc[:40] = apc.max()                            # ceiling pile-up
    df = pd.DataFrame({'FSC-A': fsc, 'FSC-H': fsch, 'Time': time, 'APC-A': apc})
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def test_autoclean_and_of_methods_and_dispatch():
    from openflo.pipeline import autoclean_keep_mask, default_autoclean_methods, gate_to_mask
    df = _clean_frame()
    gate = {'kind': 'autoclean', 'name': 'autocleaned sample',
            'methods': default_autoclean_methods()}
    keep = autoclean_keep_mask(gate, df)
    assert 0 < keep.sum() < len(df)                 # removes some, keeps most
    # gate_to_mask dispatches to the same result.
    assert np.array_equal(keep, gate_to_mask(gate, df))


def test_autoclean_individual_methods():
    from openflo.pipeline import autoclean_keep_mask, default_autoclean_methods
    df = _clean_frame()

    def only(key):
        ms = default_autoclean_methods()
        for m in ms:
            m['enabled'] = (m['key'] == key)
        return autoclean_keep_mask({'kind': 'autoclean', 'methods': ms}, df)

    assert (~only('debris')).sum() == 800           # the low-FSC mode (n//5)
    assert (~only('margin')).sum() >= 40            # ceiling pile-up
    assert (~only('doublets')).sum() > 0
    # AND is at least as strict as any single method.
    full = autoclean_keep_mask(
        {'kind': 'autoclean', 'methods': default_autoclean_methods()}, df)
    assert full.sum() <= only('debris').sum()


def test_autoclean_no_enabled_methods_is_noop():
    from openflo.pipeline import autoclean_keep_mask, default_autoclean_methods
    df = _clean_frame()
    ms = default_autoclean_methods()
    for m in ms:
        m['enabled'] = False
    assert autoclean_keep_mask({'kind': 'autoclean', 'methods': ms}, df).all()


def test_autoclean_recomputes_per_sample():
    """The SAME recipe yields different masks on different samples — the core
    'apply the calculations, not the gating' guarantee."""
    from openflo.pipeline import autoclean_keep_mask, default_autoclean_methods
    gate = {'kind': 'autoclean', 'methods': default_autoclean_methods()}
    a = autoclean_keep_mask(gate, _clean_frame(seed=1))
    b = autoclean_keep_mask(gate, _clean_frame(seed=2).iloc[:2500])
    assert a.shape != b.shape                        # sized to each sample
    assert a.sum() != b.sum()


def test_autoclean_debris_manual_min_fsc():
    """A manual min_fsc param overrides the auto cutoff (used by the QC dialog)
    and still recomputes per sample."""
    from openflo.pipeline import autoclean_keep_mask
    df = _clean_frame()
    ms = [{'key': 'debris', 'enabled': True, 'params': {'min_fsc': 50_000.0}}]
    keep = autoclean_keep_mask({'kind': 'autoclean', 'methods': ms}, df)
    fsc = df['FSC-A'].to_numpy()
    assert keep.sum() == int((fsc >= 50_000.0).sum())


def test_autoclean_methods_signature():
    from openflo.pipeline import autoclean_methods_signature, default_autoclean_methods
    g1 = {'kind': 'autoclean', 'methods': default_autoclean_methods()}
    g2 = {'kind': 'autoclean', 'methods': default_autoclean_methods()}
    assert autoclean_methods_signature(g1) == autoclean_methods_signature(g2)
    # Toggling a method changes the signature.
    g2['methods'][0]['enabled'] = False
    assert autoclean_methods_signature(g1) != autoclean_methods_signature(g2)
    # Changing a param changes the signature.
    g3 = {'kind': 'autoclean', 'methods': default_autoclean_methods()}
    g3['methods'][1]['params']['tol'] = 0.1
    assert autoclean_methods_signature(g1) != autoclean_methods_signature(g3)
    # It's hashable (usable as a cache key).
    assert hash(autoclean_methods_signature(g1)) is not None


def test_autoclean_empty_and_missing_channels_safe():
    from openflo.pipeline import autoclean_keep_mask, default_autoclean_methods
    gate = {'kind': 'autoclean', 'methods': default_autoclean_methods()}
    # Empty df.
    assert autoclean_keep_mask(gate, _clean_frame().iloc[:0]).shape == (0,)
    # No Time / FSC columns: time-based + scatter methods no-op, margin applies.
    df = pd.DataFrame({'APC-A': np.concatenate(
        [np.linspace(0, 1, 960), np.full(40, 9e6)])})
    keep = autoclean_keep_mask(gate, df)
    assert (~keep).sum() == 40                       # only margin fired
