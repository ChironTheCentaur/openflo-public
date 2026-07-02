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
