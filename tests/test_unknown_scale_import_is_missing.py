"""A FlowJo gate imported onto a channel of unknown scale reads NA.

.wsp gates are in FlowJo's linear units and are converted to the space the
sample stores its fluorescence in. A channel whose display scale is unknown
(an older OpenFlo CSV with no transform record) cannot be converted, and the
gate used to be kept as is with a log warning: a cut at 1,000 compared
against values stored on some unknown scale, naming another region. It is
now kept, selects nothing and its population reads NA with the reason.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

from openflo.gating import population_stats  # noqa: E402
from openflo.pipeline import (  # noqa: E402
    cumulative_gate_mask,
    mark_unknown_scale,
    unknown_population_reason,
    unknown_spec,
)

CUT = {'kind': 'threshold', 'channel': 'FL1-A', 'value': 1000.0,
       'parent_id': None, 'name': 'FL1pos'}


def test_only_a_gate_on_an_unknown_channel_is_flagged():
    tf = {'FL1-A': unknown_spec(), 'FL2-A': {'method': 'asinh',
                                             'cofactor': 150.0}}
    assert mark_unknown_scale(dict(CUT), tf)['scale_unknown'] is True
    on_known = dict(CUT, channel='FL2-A')
    assert 'scale_unknown' not in mark_unknown_scale(on_known, tf)
    rect = {'kind': 'rect', 'x_channel': 'FL2-A', 'y_channel': 'FL1-A',
            'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1}
    assert mark_unknown_scale(rect, tf)['scale_unknown'] is True
    boolean = {'kind': 'boolean', 'op': 'not', 'operands': ['a']}
    assert mark_unknown_scale(boolean, tf) is boolean
    assert 'scale_unknown' not in mark_unknown_scale(dict(CUT), {})


def test_a_flagged_gate_and_what_rests_on_it_read_na():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'FL1-A': rng.random(100) * 2000,
                       'FL2-A': rng.random(100)})
    gates = {'cut': dict(CUT, scale_unknown=True),
             'child': {'kind': 'threshold', 'channel': 'FL2-A', 'value': 0.5,
                       'parent_id': 'cut', 'name': 'FL2pos'},
             'not': {'kind': 'boolean', 'op': 'not', 'operands': ['cut'],
                     'parent_id': None, 'name': 'NotCut'}}
    for gid in gates:
        assert 'unknown scale' in unknown_population_reason(gates, gid)
        assert not cumulative_gate_mask(gates, gid, df).any(), gid
    rows = population_stats('s', df, gates, list(gates), {}, [],
                            {'Count', '%Total'}, ())
    assert all(r['__unknown__'] and math.isnan(r['Count']) for r in rows)


def test_a_wsp_gate_queued_for_an_unknown_scale_sample_is_flagged():
    root, ed, _gui = _editor_or_skip()
    try:
        from openflo.pipeline import FlowSample
        rng = np.random.default_rng(1)
        s = FlowSample.from_dataframe(
            pd.DataFrame({'FL1-A': rng.random(50), 'FL2-A': rng.random(50)}),
            name='old_csv')
        s.data_transforms = {'FL1-A': unknown_spec(),
                             'FL2-A': {'method': 'linear'}}
        ed._pending_sample_gates['old_csv'] = [
            dict(CUT, id='w1'),
            {'kind': 'threshold', 'channel': 'FL2-A', 'value': 0.5,
             'parent_id': None, 'name': 'FL2pos', 'id': 'w2'}]
        ed._on_loaded('old_csv', s)
        by_name = {g.get('name'): g for g in
                   ed._sample_gates['old_csv'].values()}
        assert by_name['FL1pos'].get('scale_unknown') is True
        assert not by_name['FL2pos'].get('scale_unknown')
    finally:
        root.destroy()
