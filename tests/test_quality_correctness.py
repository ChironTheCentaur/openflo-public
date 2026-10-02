"""Tranche-3 correctness-adjacent fixes from the Fable review:

- WspReader._channel_name strips FlowJo's native-compensated '<PE-A>' bracket
  form (and legacy 'Comp-' prefix) so gates on compensated channels match data.
- FlowSample._evaluate_gate_on delegates gate kinds it has no memory-lean path
  for (ellipsoid/cluster/category/boolean) to gate_to_mask, instead of silently
  passing every event through.
"""
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import openflo.pipeline as fp


def test_channel_name_strips_native_compensated_brackets():
    dim = ET.fromstring(
        '<dimension><fcs-dimension name="&lt;PE-A&gt;"/></dimension>')
    assert fp.WspReader._channel_name(dim) == 'PE-A'
    dim2 = ET.fromstring(
        '<dimension><fcs-dimension name="Comp-APC-A"/></dimension>')
    assert fp.WspReader._channel_name(dim2) == 'APC-A'


def test_evaluate_gate_on_delegates_unsupported_kind():
    """A 'category' gate (no memory-lean branch in _evaluate_gate_on) is now
    evaluated via gate_to_mask, not silently passed through as all-True."""
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame({'phase': ['G1', 'S', 'G1', 'G2M']}), name='x')
    gate = {'kind': 'category', 'channel': 'phase', 'value': 'G1'}
    mask = s._evaluate_gate_on(gate, np.arange(4))
    assert list(mask) == [True, False, True, False]


def test_evaluate_gate_on_partial_out_of_order_active_idx():
    """The returned mask is aligned to active_idx (its length + order), not the
    full frame — the only existing test uses the full arange(n), so a partial,
    reordered subset (the memory-lean path) is never exercised."""
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame({'X': [1.0, 2.0, 3.0, 4.0, 5.0]}), name='x')
    gate = {'kind': 'threshold', 'channel': 'X', 'value': 2.5}
    mask = s._evaluate_gate_on(gate, np.array([4, 1, 3]))   # rows X=5,2,4 in order
    assert list(mask) == [True, False, True]                # 5>2.5, 2>2.5, 4>2.5
