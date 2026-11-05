"""A 'threshold' gate is x >= value, as Gating-ML and FlowJo count it.

The 'threshold' kind is what a min-only Gating-ML RectangleGate becomes on
import, and what WspWriter writes back as `gating:min`. Gating-ML's rectangle
is min <= x < max, and FlowJo counts a min-only gate that way; OpenFlo
evaluated the same gate as x > value. On integer events 995..1005 with a
minimum of 1000, FlowJo counts 6 and OpenFlo counted 5 -- on integer
($DATATYPE I) data, every event sitting exactly on the cut moved out of the
population. Interval and rect stay half-open [lo, hi), so the threshold now
agrees with an interval [value, open) of the same gate.

Every evaluator of the kind must agree: gate_to_mask (editor, gating stats),
FlowSample._evaluate_gate_on (apply_region_gates, the CLI), the CLI's
threshold flags (apply_threshold_gates, which a --gates threshold feeds and
--export-wsp writes back as a min-only gate) and openflo-compare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import openflo.pipeline as fp
from openflo.compare import _compare_one_sample, _per_sample_inventory

# Integer events 995..1005, ten of each: 60 at or above 1000.
VALUES = np.repeat(np.arange(995, 1006, dtype=float), 10)
TRUTH = int((VALUES >= 1000).sum())
GATE = {'id': 't', 'parent_id': None, 'kind': 'threshold', 'channel': 'X',
        'value': 1000.0}


def _df():
    return pd.DataFrame({'X': VALUES, 'Y': np.ones_like(VALUES)})


def test_truth_is_sixty():
    """The fixture itself: 10 events on the cut, 50 above it."""
    assert TRUTH == 60 and int((VALUES == 1000).sum()) == 10


def test_gate_to_mask_counts_events_on_the_cut():
    """Was 50 (x > 1000), so the 10 events AT 1000 were left out."""
    assert int(fp.gate_to_mask(GATE, _df()).sum()) == TRUTH


def test_threshold_agrees_with_the_open_interval_of_the_same_gate():
    """A min-only rectangle read as 'threshold' or as an interval [min, open)
    is one region; the two used to differ by every event on the bound."""
    interval = {'kind': 'interval', 'channel': 'X', 'lo': 1000.0, 'hi': 1e12}
    df = _df()
    assert (fp.gate_to_mask(GATE, df) == fp.gate_to_mask(interval, df)).all()


def test_apply_region_gates_path_counts_events_on_the_cut():
    """The memory-lean evaluator behind apply_region_gates (and the CLI)
    has its own threshold branch; it said x > value too."""
    s = fp.FlowSample.from_dataframe(_df())
    s.apply_region_gates([dict(GATE)])
    assert len(s.data) == TRUTH


def test_cli_threshold_flags_count_events_on_the_cut():
    """A --gates threshold becomes a `<channel>_pos` flag and is exported to
    FlowJo as a min-only gate; the flag must count what FlowJo counts."""
    s = fp.FlowSample.from_dataframe(_df())
    s.apply_threshold_gates({'X': 1000.0})
    assert int(s.data['X_pos'].sum()) == TRUTH


def test_flowjo_min_only_gate_round_trip_and_compare(tmp_path):
    """A min-only RectangleGate written by WspWriter (as `gating:min`), read
    back by WspReader and counted by openflo-compare: FlowJo's count is 60
    and each OpenFlo path must say 60, where all said 50."""
    from openflo.fcs_export import write_fcs
    fcs = tmp_path / 's.fcs'
    write_fcs(_df(), str(fcs))
    w = fp.WspWriter()
    w.add_sample('s', str(fcs), ['X', 'Y'], [dict(GATE)])
    wsp = tmp_path / 's.wsp'
    w.write(str(wsp))
    xml = wsp.read_text(encoding='utf-8')
    assert 'gating:min="1000.0"' in xml and 'gating:max' not in xml

    back = fp.WspReader(str(wsp)).extract_gates()
    assert [g['kind'] for g in back] == ['threshold']
    assert int(fp.gate_to_mask(back[0], _df()).sum()) == TRUTH

    (_, _, pops), = _per_sample_inventory(fp.WspReader(str(wsp)))
    (row,) = _compare_one_sample('s', str(fcs), pops)
    assert row['openflo_count'] == TRUTH


def test_describe_gate_says_at_or_above():
    """The population label shown in the gate tree and written to the
    statistics said '>' -- now the rule it counts by."""
    assert fp.describe_gate(GATE) == 'T  X >= 1e+03'
