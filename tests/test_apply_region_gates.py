"""End-to-end gate application, on generated samples only."""
import json

import numpy as np
import pytest

import openflo.pipeline as fp


def test_flowsample_loads_synthetic(synthetic_fcs, synthetic_channels):
    s = fp.FlowSample(synthetic_fcs)
    assert len(s.data) == 1000
    for ch in synthetic_channels:
        assert ch in s.channel_names, f"missing channel {ch}"


def test_apply_threshold_gate_synthetic(synthetic_fcs):
    """A threshold on the bimodal channel should split ~50/50."""
    s = fp.FlowSample(synthetic_fcs)
    s.apply_threshold_gates({'FL1-A': 1000.0})
    assert 'FL1-A_pos' in s.data.columns
    pos_frac = float(np.asarray(s.data['FL1-A_pos']).mean())
    # Channel is bimodal centred at 100 and 5000 — should be ~50% positive.
    assert 0.3 < pos_frac < 0.7, f"expected ~50% positive, got {pos_frac:.0%}"


def test_apply_rect_gate_synthetic(synthetic_fcs):
    """A rectangle on FSC/SSC should select a non-empty subset."""
    s = fp.FlowSample(synthetic_fcs)
    initial = len(s.data)
    rect = [{
        'kind': 'rect',
        'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
        'x0': 0, 'x1': 1e6, 'y0': 0, 'y1': 1e6,
        'id': 'g1', 'parent_id': None,
    }]
    s.apply_region_gates(rect)
    assert 0 < len(s.data) <= initial, (
        f"region gate produced empty or impossible result: {len(s.data)}")


def test_apply_region_gates_two_gate_AND_exact_subset():
    """Two region gates compose as AND, the second evaluated only on the rows the
    first kept and mapped back to the right positions. Assert the EXACT survivors
    equal (gateA & gateB) — single-gate / keep-all / keep-none tests never expose
    the partial-subset writeback."""
    import pandas as pd
    df = pd.DataFrame({'FSC-A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'SSC-A': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}, dtype=float)
    s = fp.FlowSample.from_dataframe(df.copy(), name='x')
    gates = [
        {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'x0': 2.5, 'x1': 8.5, 'y0': 0, 'y1': 100, 'id': 'a', 'parent_id': None},
        {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'x0': 0, 'x1': 100, 'y0': 3.5, 'y1': 6.5, 'id': 'b', 'parent_id': None}]
    s.apply_region_gates(gates)
    # a: FSC in (2.5,8.5) → FSC 3..8;  b: SSC in (3.5,6.5) → SSC 4,5,6 (FSC 7,6,5)
    #   AND → FSC {5,6,7}.
    assert sorted(s.data['FSC-A'].tolist()) == [5.0, 6.0, 7.0]


# ── A template applied end to end, on a generated FCS file ───────────────────
#
# Was an opt-in test that ran only on real FCS files someone had to supply, so
# it never ran in CI and only checked "no events were gained". Now the sample
# is generated (openflo.synthetic), written to a real FCS file, read back, and
# every gate's result is checked against an independent computation.

TEMPLATES = ["templates/testtemplate.json",
             "src/openflo/template_library/example_panel.json"]


def _load_template(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    region_gates, overrides = [], {}
    for g in data.get('gates', []):
        if g.get('kind') == 'threshold':
            overrides[g['channel']] = float(g['value'])
        elif g.get('kind') in ('interval', 'rect', 'polygon'):
            region_gates.append(g)
    return region_gates, overrides


def _generated_sample(tmp_path):
    from openflo import synthetic
    df = synthetic.differentiation_sample(9, 'Stim', n=20_000, seed=3)
    path = synthetic._write_fcs(str(tmp_path / 'generated.fcs'), df)
    s = fp.FlowSample(path)
    s.run_qc()
    s.auto_compensate()
    s.apply_transform()
    return s


def _inside(gates, data):
    """Which events lie inside every polygon, by matplotlib's own test."""
    from matplotlib.path import Path
    keep = np.ones(len(data), dtype=bool)
    for g in gates:
        xy = data[[g['x_channel'], g['y_channel']]].to_numpy(dtype=float)
        keep &= Path(np.asarray(g['vertices'], dtype=float)).contains_points(xy)
    return keep


@pytest.mark.parametrize('tpl', TEMPLATES)
def test_template_applies_exactly_on_a_generated_sample(tmp_path, tpl):
    """Positive: each threshold marks exactly the events above it, and the
    region gates keep exactly the events inside every polygon."""
    region_gates, overrides = _load_template(tpl)
    s = _generated_sample(tmp_path)
    before = s.data.copy()
    if overrides:
        s.apply_threshold_gates(overrides)
        for ch, v in overrides.items():
            expected = before[ch].to_numpy() > v
            assert np.array_equal(np.asarray(s.data[ch + '_pos'], dtype=bool),
                                  expected), ch
    if region_gates:
        keep = _inside(region_gates, before)
        assert 0 < keep.sum() < len(before)    # the gate really cuts
        s.apply_region_gates(region_gates)
        assert len(s.data) == int(keep.sum())
        assert np.array_equal(np.sort(s.data['FSC-A'].to_numpy()),
                              np.sort(before['FSC-A'].to_numpy()[keep]))


@pytest.mark.parametrize('tpl', TEMPLATES)
def test_template_moved_off_the_data_keeps_nothing(tmp_path, tpl):
    """Negative: the same gates placed where no event is keep no event, and
    thresholds above every event mark none positive."""
    region_gates, overrides = _load_template(tpl)
    s = _generated_sample(tmp_path)
    if overrides:
        top = {ch: float(s.data[ch].max()) + 1.0 for ch in overrides}
        s.apply_threshold_gates(top)
        for ch in overrides:
            assert not np.asarray(s.data[ch + '_pos'], dtype=bool).any()
    if region_gates:
        far = [dict(g, vertices=[[x + 1e7, y + 1e7] for x, y in g['vertices']])
               for g in region_gates]
        s.apply_region_gates(far)
        assert len(s.data) == 0


# The batch / CLI path has its own lean evaluator for these four kinds. A
# record it cannot read admits nothing, as gate_to_mask does: before, a
# polygon of two points kept every event and the others raised KeyError /
# ValueError (PR #446 review).
_UNREADABLE = {
    'polygon of two points': {'kind': 'polygon', 'x_channel': 'FSC-A',
                              'y_channel': 'SSC-A',
                              'vertices': [[0, 0], [1e6, 1e6]]},
    'polygon vertices text': {'kind': 'polygon', 'x_channel': 'FSC-A',
                              'y_channel': 'SSC-A', 'vertices': 'abc'},
    'threshold without value': {'kind': 'threshold', 'channel': 'FL1-A'},
    'threshold value text': {'kind': 'threshold', 'channel': 'FL1-A',
                             'value': 'abc'},
    'interval without hi': {'kind': 'interval', 'channel': 'FL1-A',
                            'lo': 0.0},
    'rect without y1': {'kind': 'rect', 'x_channel': 'FSC-A',
                        'y_channel': 'SSC-A', 'x0': -1e12, 'x1': 1e12,
                        'y0': -1e12},
    'unknown kind': {'kind': 'spline', 'channel': 'FL1-A'},
}


@pytest.mark.parametrize('label', sorted(_UNREADABLE))
def test_a_gate_that_cannot_be_read_keeps_no_events(synthetic_fcs, label,
                                                     caplog):
    s = fp.FlowSample(synthetic_fcs)
    assert len(s.data) == 1000
    g = dict(_UNREADABLE[label], id='g1', parent_id=None, name='Odd')
    with caplog.at_level('WARNING', logger='openflo'):
        s.apply_region_gates([g])
    assert len(s.data) == 0
    assert any('Odd' in r.getMessage() and 'fail-closed' in r.getMessage()
               for r in caplog.records)


# A gate on a channel the sample lacks: the app reads the population NA
# (gate_to_mask fails closed), so the batch run keeps no events too. Before,
# the batch run's own evaluator skipped the gate and kept every event, so a
# sample whose FL1-A is named FL1-H passed the whole gate.
_ON_A_MISSING_CHANNEL = {
    'threshold': {'kind': 'threshold', 'channel': 'FL9-A', 'value': 0.0},
    'interval': {'kind': 'interval', 'channel': 'FL9-A', 'lo': -1e12,
                 'hi': 1e12},
    'rect x': {'kind': 'rect', 'x_channel': 'FL9-A', 'y_channel': 'SSC-A',
               'x0': -1e12, 'x1': 1e12, 'y0': -1e12, 'y1': 1e12},
    'rect y': {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'FL9-A',
               'x0': -1e12, 'x1': 1e12, 'y0': -1e12, 'y1': 1e12},
    'polygon x': {'kind': 'polygon', 'x_channel': 'FL9-A',
                  'y_channel': 'SSC-A',
                  'vertices': [[-1e9, -1e9], [1e9, -1e9], [0, 1e9]]},
    'polygon y': {'kind': 'polygon', 'x_channel': 'FSC-A',
                  'y_channel': 'FL9-A',
                  'vertices': [[-1e9, -1e9], [1e9, -1e9], [0, 1e9]]},
}


@pytest.mark.parametrize('label', sorted(_ON_A_MISSING_CHANNEL))
def test_a_gate_on_a_missing_channel_keeps_no_events(synthetic_fcs, label,
                                                     caplog):
    s = fp.FlowSample(synthetic_fcs)
    assert 'FL9-A' not in s.data.columns and len(s.data) == 1000
    g = dict(_ON_A_MISSING_CHANNEL[label], id='g1', parent_id=None,
             name='Lost')
    # The app's evaluator agrees: nothing, so the population reads NA.
    assert not fp.gate_to_mask(g, s.data).any()
    caplog.clear()
    with caplog.at_level('WARNING', logger='openflo'):
        s.apply_region_gates([g])
    assert len(s.data) == 0
    (rec,) = [r for r in caplog.records if 'Lost' in r.getMessage()]
    assert rec.levelname == 'WARNING'
    assert "'FL9-A'" in rec.getMessage()
    assert 'fail-closed' in rec.getMessage()


def test_a_gate_whose_channels_are_present_still_keeps_its_events(
        synthetic_fcs):
    """The same gates on channels the sample has keep every event."""
    for g in _ON_A_MISSING_CHANNEL.values():
        s = fp.FlowSample(synthetic_fcs)
        g = {k: ('FL1-A' if v == 'FL9-A' else v) for k, v in g.items()}
        s.apply_region_gates([dict(g, id='g1', parent_id=None)])
        assert len(s.data) == 1000, g
