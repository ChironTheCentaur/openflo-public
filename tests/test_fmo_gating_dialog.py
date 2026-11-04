"""FMO gating dialog (Gate tools -> FMO gating…) — the Apply action.

gating_helpers.fmo_threshold(_gate) is tested directly; FMOGatingDialog._apply
was 44% covered. Ground truth: an FMO control whose CD3 distribution is known,
so the gate value must be exactly np.percentile(FMO CD3, pct).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo.async_task import wait_for_job
from openflo.channel_transforms import unknown_spec
from tests.conftest import editor_or_skip


def _editor_or_skip():
    return editor_or_skip()


def _register(ed, name, df):
    s = SimpleNamespace(name=name, path=None, data=df,
                        fluor_channels=['CD3', 'CD4'],
                        channel_labels={c: c for c in df.columns})
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    if len(ed._samples) == 1:
        ed._channels = list(df.columns)
        ed._channel_labels = {c: c for c in df.columns}
        ed._populate_channel_combos()


def _data(seed=4):
    rng = np.random.default_rng(seed)
    fmo = pd.DataFrame({
        'FSC-A': np.r_[rng.normal(0.6, 0.05, 9000),
                       rng.normal(0.1, 0.03, 1000)],
        'CD3': np.r_[rng.normal(0.2, 0.05, 9000),
                     rng.normal(0.6, 0.05, 1000)],         # bright debris
        'CD4': rng.normal(0.2, 0.05, 10000)})
    stained = pd.DataFrame({
        'FSC-A': rng.normal(0.6, 0.05, 10000),
        'CD3': np.r_[rng.normal(0.2, 0.05, 5000),
                     rng.normal(0.65, 0.03, 5000)],
        'CD4': rng.normal(0.2, 0.05, 10000)})
    return fmo, stained


def _setup(ed):
    fmo, stained = _data()
    _register(ed, 'stained', stained)
    _register(ed, 'fmo_cd3', fmo)
    ed._set_active_sample('stained')
    return fmo, stained


def test_fmo_dialog_places_gate_at_fmo_percentile():
    root, ed = _editor_or_skip()
    try:
        fmo, _ = _setup(ed)
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._pct.set('99.5')
        dlg._map['CD3'].set('fmo_cd3')              # CD4 left '(none)'
        dlg._apply()
        assert wait_for_job(dlg._busy)
        gates = [g for g in ed._gates.values() if g.get('kind') == 'threshold']
        assert len(gates) == 1
        assert gates[0]['channel'] == 'CD3'
        assert gates[0]['value'] == pytest.approx(
            float(np.percentile(fmo['CD3'], 99.5)), abs=1e-12)
        assert ed.status_var.get() == (
            "Added 1 FMO threshold gate(s) at the 99.5th percentile to "
            "stained.")
    finally:
        root.destroy()


def test_fmo_dialog_rejects_out_of_range_percentile():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._pct.set('150')
        dlg._map['CD3'].set('fmo_cd3')
        dlg._apply()
        assert wait_for_job(dlg._busy)
        assert not ed._gates
        assert ed.status_var.get() == "FMO percentile must be between 0 and 100."
    finally:
        root.destroy()


def test_fmo_dialog_without_mapping_adds_nothing():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._apply()
        assert wait_for_job(dlg._busy)
        assert not ed._gates
        assert ed.status_var.get() == "No FMO mappings chosen — nothing added."
    finally:
        root.destroy()


def test_fmo_gate_follows_selected_parent_gate():
    """With a parent gate selected (here 'cells', excluding bright debris),
    the FMO gate goes under it and its cutoff is the FMO percentile INSIDE
    that parent. Taken over every FMO event, the debris set the cutoff at
    0.662 instead of 0.317 and called 16.8% of a 50%-positive sample CD3+."""
    root, ed = _editor_or_skip()
    try:
        fmo, _ = _setup(ed)
        pid = ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A',
                            'value': 0.3, 'op': '>=', 'parent_id': None,
                            'name': 'cells'})
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('stained', pid))
        root.update()
        assert ed._selected_gate_id() == pid
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._map['CD3'].set('fmo_cd3')
        dlg._apply()
        assert wait_for_job(dlg._busy)
        g = [g for g in ed._gates.values() if g.get('channel') == 'CD3'][0]
        assert g['parent_id'] == pid
        inside = fmo['FSC-A'] >= 0.3
        assert g['value'] == pytest.approx(
            float(np.percentile(fmo.loc[inside, 'CD3'], 99)), abs=1e-9)
        assert ed.status_var.get() == (
            "Added 1 FMO threshold gate(s) at the 99th percentile to "
            "stained under cells.")
    finally:
        root.destroy()


# ── An FMO on another display transform than the stained sample ────────────

_N = 10000


def _linear_truth(seed=1):
    """FMO background ~N(100, 60); stained 50% at that background, 50% bright
    (~5000). The true positive fraction above the FMO's 99th percentile."""
    rng = np.random.default_rng(seed)
    fmo = rng.normal(100, 60, _N)
    stained = np.r_[rng.normal(100, 60, _N // 2),
                    rng.lognormal(np.log(5000), 0.3, _N // 2)]
    cut = float(np.percentile(fmo, 99))
    return rng, fmo, stained, float((stained > cut).mean() * 100)


def _flow(ed, name, rng, cd3_display, record, **extra):
    import openflo.pipeline as fp
    df = pd.DataFrame({'FSC-A': rng.normal(8e4, 8e3, _N), 'CD3-A': cd3_display,
                       **extra})
    s = fp.FlowSample.from_dataframe(df, name=name, labels={'CD3-A': 'CD3'})
    s.data_transforms = record
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    if len(ed._samples) == 1:
        ed._channels = list(df.columns)
        ed._channel_labels = {'CD3-A': 'CD3'}
        ed._populate_channel_combos()
    return s


def _positive(ed, stained):
    from openflo.gate_eval import cumulative_gate_mask
    gid, = [k for k, g in ed._gates.items() if g.get('kind') == 'threshold']
    return float(np.mean(cumulative_gate_mask(ed._gates, gid,
                                              stained.data)) * 100)


def test_an_fmo_on_another_cofactor_gates_on_the_linear_cutoff():
    """FMO stored on asinh cofactor 5 beside a stained sample on cofactor
    150: the percentile was taken in the FMO's own display coordinates and
    used as the stained sample's, calling 5.36% positive where 50.52% is
    right. The cutoff is now taken on the FMO's linear values and mapped
    through the stained sample's record."""
    import openflo.pipeline as fp
    root, ed = _editor_or_skip()
    try:
        rng, fmo, stained, truth = _linear_truth()
        st = _flow(ed, 'stained', rng,
                   fp.transform_values(stained, method='asinh'),
                   {'CD3-A': fp.transform_spec('asinh')})
        _flow(ed, 'FMO_CD3', rng,
              fp.transform_values(fmo, method='asinh', cofactor=5.0),
              {'CD3-A': fp.transform_spec('asinh', cofactor=5.0)})
        ed._set_active_sample('stained')
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._pct.set('99')
        dlg._map['CD3-A'].set('FMO_CD3')
        dlg._apply()
        assert wait_for_job(dlg._busy)
        # percent of 10,000 events (one event = 0.01 points): allows a few events at the
        # cutoff to flip in the display-to-linear mapping; the old bug was off by ~45 points
        assert _positive(ed, st) == pytest.approx(truth, abs=0.05)
        g, = [g for g in ed._gates.values() if g.get('kind') == 'threshold']
        assert g['value'] == pytest.approx(float(fp.transform_values(
            np.array([np.percentile(fmo, 99)]), method='asinh')[0]), abs=1e-9)
    finally:
        root.destroy()


def test_the_parent_gate_reads_the_fmo_in_the_stained_coordinates():
    """The parent gate lives in the stained sample's coordinates. Read on an
    FMO whose CD4 is on another cofactor, it kept the wrong FMO events. The
    FMO is read through it in the stained coordinates: the cutoff is the
    FMO's linear CD3 percentile among its CD4-dim (linear < 500) events."""
    import openflo.pipeline as fp
    root, ed = _editor_or_skip()
    try:
        rng = np.random.default_rng(7)
        cd4 = np.r_[rng.normal(100, 30, _N // 2), rng.normal(5000, 300, _N // 2)]
        fmo_cd3 = np.r_[rng.normal(100, 60, _N // 2),       # CD4-dim cells
                        rng.normal(3000, 300, _N // 2)]     # CD4-bright debris
        st_cd3 = rng.normal(100, 60, _N)
        rec = {'CD3-A': fp.transform_spec('asinh'),
               'CD4-A': fp.transform_spec('asinh')}
        _flow(ed, 'stained', rng, fp.transform_values(st_cd3, method='asinh'),
              rec, **{'CD4-A': fp.transform_values(cd4, method='asinh')})
        five = fp.transform_spec('asinh', cofactor=5.0)
        _flow(ed, 'FMO_CD3', rng, fp.transform_values(fmo_cd3, **five),
              {'CD3-A': five, 'CD4-A': five},
              **{'CD4-A': fp.transform_values(cd4, **five)})
        ed._set_active_sample('stained')
        hi = float(np.arcsinh(500 / 150))                 # stained coords
        pid = ed._add_gate({'kind': 'interval', 'channel': 'CD4-A',
                            'lo': -10.0, 'hi': hi, 'parent_id': None,
                            'name': 'CD4 dim'})
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('stained', pid))
        root.update()
        assert ed._selected_gate_id() == pid
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._pct.set('99')
        dlg._map['CD3-A'].set('FMO_CD3')
        dlg._apply()
        assert wait_for_job(dlg._busy)
        g, = [g for g in ed._gates.values() if g.get('kind') == 'threshold']
        want = np.percentile(fmo_cd3[cd4 < 500], 99)
        assert g['value'] == pytest.approx(float(np.arcsinh(want / 150)),
                                           abs=1e-6)
    finally:
        root.destroy()


def test_an_fmo_of_unknown_scale_is_refused_by_name():
    """An FMO tube read from an older OpenFlo CSV, its CD3 on an unknown
    scale: its percentile is in coordinates nothing can map, and the gate
    it placed called 0.0% positive where 50.52% is right, while the status
    said 'Added 1 FMO threshold gate'. No gate, and the status names it."""
    import openflo.pipeline as fp
    root, ed = _editor_or_skip()
    try:
        rng, fmo, stained, _truth = _linear_truth()
        _flow(ed, 'stained', rng, fp.transform_values(stained, method='logicle'),
              {'CD3-A': fp.transform_spec('logicle')})
        _flow(ed, 'FMO_CD3', rng, fp.transform_values(fmo, method='asinh'),
              {'CD3-A': unknown_spec('session:old')})
        ed._set_active_sample('stained')
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._pct.set('99')
        dlg._map['CD3-A'].set('FMO_CD3')
        dlg._apply()
        assert wait_for_job(dlg._busy)
        assert not [g for g in ed._gates.values()
                    if g.get('kind') == 'threshold']
        msg = ed.status_var.get()
        assert 'FMO_CD3' in msg and 'CD3-A' in msg and 'unknown' in msg, msg
    finally:
        root.destroy()


# ── A channel whose cutoff cannot be computed is named, not dropped ────────
# Before, any failure other than an unknown scale went only to the console:
# the channel got no gate and the status said nothing about it, or even
# "No FMO mappings chosen" when it was the only channel mapped.

def _register_fl(ed, name, df):
    s = SimpleNamespace(name=name, path=None, data=df,
                        fluor_channels=['FL1-A', 'FL2-A'],
                        channel_labels={c: c for c in df.columns})
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    if len(ed._samples) == 1:
        ed._channels = list(df.columns)
        ed._channel_labels = {c: c for c in df.columns}
        ed._populate_channel_combos()


def _fl_frame(rng, fsc, n=4000):
    return pd.DataFrame({'FSC-A': rng.normal(fsc, 0.03, n),
                         'FL1-A': rng.normal(0.2, 0.05, n),
                         'FL2-A': rng.normal(0.3, 0.05, n)})


def test_an_fmo_with_no_events_in_the_parent_is_named_in_the_status():
    """The parent gate (FSC-A >= 0.9) holds every stained event and no FMO
    event: there is no background to take a percentile of. No gate, and
    the status names the channel, the FMO and why."""
    rng = np.random.default_rng(11)
    root, ed = _editor_or_skip()
    try:
        _register_fl(ed, 'sample_stained', _fl_frame(rng, 1.0))
        _register_fl(ed, 'sample_fmo_fl1', _fl_frame(rng, 0.6))
        ed._set_active_sample('sample_stained')
        pid = ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A',
                            'value': 0.9, 'op': '>=', 'parent_id': None,
                            'name': 'cells'})
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('sample_stained', pid))
        root.update()
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._map['FL1-A'].set('sample_fmo_fl1')
        dlg._apply()
        assert wait_for_job(dlg._busy)
        assert [g['channel'] for g in ed._gates.values()
                if g.get('kind') == 'threshold'] == ['FSC-A']
        assert ed.status_var.get() == (
            "No FMO gate added.  [!] Not gated, no cutoff: FL1-A from "
            "sample_fmo_fl1 (no FMO events inside the parent gate).")
    finally:
        root.destroy()


def test_the_gated_channels_say_which_channel_got_no_gate():
    """FL1-A's FMO is fine; FL2-A's FMO holds no finite FL2-A value. FL1-A
    gets its gate at the FMO percentile and the status names FL2-A."""
    rng = np.random.default_rng(12)
    root, ed = _editor_or_skip()
    try:
        _register_fl(ed, 'sample_stained', _fl_frame(rng, 0.6))
        fmo1 = _fl_frame(rng, 0.6)
        fmo2 = _fl_frame(rng, 0.6)
        fmo2['FL2-A'] = np.nan
        _register_fl(ed, 'sample_fmo_fl1', fmo1)
        _register_fl(ed, 'sample_fmo_fl2', fmo2)
        ed._set_active_sample('sample_stained')
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._map['FL1-A'].set('sample_fmo_fl1')
        dlg._map['FL2-A'].set('sample_fmo_fl2')
        dlg._apply()
        assert wait_for_job(dlg._busy)
        gates = [g for g in ed._gates.values() if g.get('kind') == 'threshold']
        assert [g['channel'] for g in gates] == ['FL1-A']
        assert gates[0]['value'] == pytest.approx(
            float(np.percentile(fmo1['FL1-A'], 99)), abs=1e-12)
        assert ed.status_var.get() == (
            "Added 1 FMO threshold gate(s) at the 99th percentile to "
            "sample_stained.  [!] Not gated, no cutoff: FL2-A from "
            "sample_fmo_fl2 (no finite values).")
    finally:
        root.destroy()


def test_an_unexpected_cutoff_error_is_named_and_logged(monkeypatch):
    """Any other error while reading one FMO is reported for that channel
    (and noted in the log) while the next channel still gets its cutoff."""
    import openflo.gating_helpers as gh
    from openflo import errors
    from openflo.ui_fmo import FMOGatingDialog
    rng = np.random.default_rng(13)
    fmo = SimpleNamespace(name='sample_fmo', data=_fl_frame(rng, 0.6),
                          data_transforms={})
    real = gh.fmo_threshold

    def flaky(values, percentile=99.0):
        if flaky.calls == 0:
            flaky.calls += 1
            raise RuntimeError("synthetic failure")
        return real(values, percentile=percentile)
    flaky.calls = 0
    monkeypatch.setattr(gh, 'fmo_threshold', flaky)
    job = SimpleNamespace(progress=lambda *a, **k: None)
    stained = SimpleNamespace(name='sample_stained', data_transforms={})
    cuts, refused, failed = FMOGatingDialog._cutoffs(
        job, [('FL1-A', 'sample_fmo', fmo), ('FL2-A', 'sample_fmo', fmo)],
        None, {}, set(), stained, {}, 99.0)
    assert [c for c, _ in cuts] == ['FL2-A']
    assert cuts[0][1] == pytest.approx(
        float(np.percentile(fmo.data['FL2-A'], 99)), abs=1e-12)
    assert refused == []
    assert failed == ["FL1-A from sample_fmo (RuntimeError: synthetic "
                      "failure)"]
    assert errors.noted()[-1] == (
        "[handled] FMO cutoff for FL1-A from sample_fmo: RuntimeError: "
        "synthetic failure")
