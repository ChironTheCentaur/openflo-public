"""Cross-sample label alignment (Chunk B).

The same antibody can sit on a different fluorophore across samples, so
cross-sample comparison must align by antibody LABEL, not detector.
These cover the pure helpers in openflo.pipeline:
align_fluor_labels, common_fluor_warning, concatenate_by_label.
"""
import types

import pandas as pd

import openflo.pipeline as fp


def _sample(name, det_to_label, data=None):
    """Minimal FlowSample-like stub: name + fluor_channels (detectors) +
    channel_labels ({detector: antibody}) + optional data frame."""
    s = types.SimpleNamespace()
    s.name = name
    s.channel_labels = dict(det_to_label)
    s.fluor_channels = list(det_to_label.keys())
    s.data = data
    return s


# ── align_fluor_labels ───────────────────────────────────────────────────────

def test_align_same_markers_different_fluors():
    """CD901b on FL1 in A, CD901b on FL4 in B → one common label 'CD901b'
    mapping to each sample's own detector."""
    a = _sample('A', {'FL1-A': 'CD901b', 'FL3-A': 'CD903'})
    b = _sample('B', {'FL4-A': 'CD901b', 'FL5-A': 'CD903'})
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['CD901b', 'CD903']
    assert info['per_sample']['A'] == {'CD901b': 'FL1-A', 'CD903': 'FL3-A'}
    assert info['per_sample']['B'] == {'CD901b': 'FL4-A', 'CD903': 'FL5-A'}
    assert info['missing'] == {}


def test_align_flags_non_common_labels():
    a = _sample('A', {'FL1-A': 'CD901b', 'FL2-A': 'CD902', 'FL3-A': 'CD903'})
    b = _sample('B', {'FL4-A': 'CD901b', 'FL5-A': 'CD903'})    # no CD902
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['CD901b', 'CD903']
    assert info['missing'] == {'CD902': ['B']}
    # union preserves first-seen order from sample A
    assert info['all_labels'] == ['CD901b', 'CD902', 'CD903']


def test_align_falls_back_to_detector_when_unlabelled():
    """No antibody label → the detector name IS the label."""
    a = _sample('A', {'FL1-A': 'FL1-A'})   # unlabelled
    b = _sample('B', {'FL1-A': 'FL1-A'})
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['FL1-A']


# ── common_fluor_warning ─────────────────────────────────────────────────────

def test_warning_empty_when_consistent():
    a = _sample('A', {'FL1-A': 'CD901b', 'FL5-A': 'CD903'})
    b = _sample('B', {'FL4-A': 'CD901b', 'FL2-A': 'CD903'})
    assert fp.common_fluor_warning([a, b]) == ''


def test_warning_lists_missing_labels():
    a = _sample('A', {'FL1-A': 'CD901b', 'FL2-A': 'CD902', 'FL3-A': 'CD903'})
    b = _sample('B', {'FL4-A': 'CD901b', 'FL5-A': 'CD903'})
    msg = fp.common_fluor_warning([a, b])
    assert 'CD902' in msg
    assert 'B' in msg
    assert 'CD901b' in msg and 'CD903' in msg   # common set named


# ── concatenate_by_label ─────────────────────────────────────────────────────

def test_concatenate_by_label_ties_across_fluors():
    a = _sample('A', {'FL1-A': 'CD901b', 'FL3-A': 'CD903'},
                data=pd.DataFrame({'FSC-A': [1, 2], 'FL1-A': [10, 20],
                                   'FL3-A': [30, 40]}))
    b = _sample('B', {'FL4-A': 'CD901b', 'FL5-A': 'CD903'},
                data=pd.DataFrame({'FSC-A': [3, 4], 'FL4-A': [50, 60],
                                   'FL5-A': [70, 80]}))
    merged, common = fp.concatenate_by_label([a, b])
    assert common == ['CD901b', 'CD903']
    # Columns are the common LABELS + the origin tag; detectors gone.
    assert set(merged.columns) == {'CD901b', 'CD903', 'sample_origin'}
    assert len(merged) == 4
    # A's CD901b came from FL1-A (10,20); B's from FL4-A (50,60).
    a_rows = merged[merged['sample_origin'] == 'A']
    b_rows = merged[merged['sample_origin'] == 'B']
    assert list(a_rows['CD901b']) == [10, 20]
    assert list(b_rows['CD901b']) == [50, 60]


def test_concatenate_by_label_drops_non_common():
    """CD902 (only in A) must not appear in the merged label frame."""
    a = _sample('A', {'FL1-A': 'CD901b', 'FL2-A': 'CD902'},
                data=pd.DataFrame({'FL1-A': [1], 'FL2-A': [2]}))
    b = _sample('B', {'FL4-A': 'CD901b'},
                data=pd.DataFrame({'FL4-A': [3]}))
    merged, common = fp.concatenate_by_label([a, b])
    assert common == ['CD901b']
    assert 'CD902' not in merged.columns
    assert set(merged.columns) == {'CD901b', 'sample_origin'}


# ── relabel_gate_for_sample (label-first gate retargeting) ───────────────────

def test_relabel_threshold_gate_by_label():
    # Template gate authored on FL1-A (CD901b); target sample has CD901b
    # on FL4-A → channel retargets to FL4-A.
    gate = {'kind': 'threshold', 'channel': 'FL1-A', 'label': 'CD901b',
            'value': 0.5}
    out = fp.relabel_gate_for_sample(gate, {'CD901b': 'FL4-A', 'CD903': 'FL5-A'})
    assert out['channel'] == 'FL4-A'
    assert gate['channel'] == 'FL1-A'          # original untouched (copy)


def test_relabel_rect_gate_both_axes():
    gate = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
            'x_label': 'CD901b', 'y_label': 'CD902',
            'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1}
    out = fp.relabel_gate_for_sample(
        gate, {'CD901b': 'FL4-A', 'CD902': 'FL5-A'})
    assert out['x_channel'] == 'FL4-A'
    assert out['y_channel'] == 'FL5-A'


def test_relabel_leaves_unlabelled_channels():
    """No label stamped → channel left as-is (gate reads its detector)."""
    gate = {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5}
    out = fp.relabel_gate_for_sample(gate, {'CD901b': 'FL4-A'})
    assert out['channel'] == 'FL1-A'


def test_relabel_label_absent_in_sample_keeps_detector():
    """Label stamped but the target sample lacks it → channel unchanged."""
    gate = {'kind': 'threshold', 'channel': 'FL1-A', 'label': 'CD901b',
            'value': 0.5}
    out = fp.relabel_gate_for_sample(gate, {'CD903': 'FL5-A'})   # no CD901b
    assert out['channel'] == 'FL1-A'


def test_relabel_empty_map_is_passthrough():
    gate = {'kind': 'threshold', 'channel': 'FL1-A', 'label': 'CD901b'}
    out = fp.relabel_gate_for_sample(gate, {})
    assert out == gate and out is not gate       # copy, unchanged
