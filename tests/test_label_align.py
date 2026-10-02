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
    """CD11b on BV421 in A, CD11b on FITC in B → one common label 'CD11b'
    mapping to each sample's own detector."""
    a = _sample('A', {'BV421-A': 'CD11b', 'PE-Cy7-A': 'CD45'})
    b = _sample('B', {'FITC-A': 'CD11b', 'PE-A': 'CD45'})
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['CD11b', 'CD45']
    assert info['per_sample']['A'] == {'CD11b': 'BV421-A', 'CD45': 'PE-Cy7-A'}
    assert info['per_sample']['B'] == {'CD11b': 'FITC-A', 'CD45': 'PE-A'}
    assert info['missing'] == {}


def test_align_flags_non_common_labels():
    a = _sample('A', {'BV421-A': 'CD11b', 'APC-A': 'CD34', 'PE-Cy7-A': 'CD45'})
    b = _sample('B', {'FITC-A': 'CD11b', 'PE-A': 'CD45'})    # no CD34
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['CD11b', 'CD45']
    assert info['missing'] == {'CD34': ['B']}
    # union preserves first-seen order from sample A
    assert info['all_labels'] == ['CD11b', 'CD34', 'CD45']


def test_align_falls_back_to_detector_when_unlabelled():
    """No antibody label → the detector name IS the label."""
    a = _sample('A', {'BV421-A': 'BV421-A'})   # unlabelled
    b = _sample('B', {'BV421-A': 'BV421-A'})
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['BV421-A']


# ── common_fluor_warning ─────────────────────────────────────────────────────

def test_warning_empty_when_consistent():
    a = _sample('A', {'BV421-A': 'CD11b', 'PE-A': 'CD45'})
    b = _sample('B', {'FITC-A': 'CD11b', 'APC-A': 'CD45'})
    assert fp.common_fluor_warning([a, b]) == ''


def test_warning_lists_missing_labels():
    a = _sample('A', {'BV421-A': 'CD11b', 'APC-A': 'CD34', 'PE-Cy7-A': 'CD45'})
    b = _sample('B', {'FITC-A': 'CD11b', 'PE-A': 'CD45'})
    msg = fp.common_fluor_warning([a, b])
    assert 'CD34' in msg
    assert 'B' in msg
    assert 'CD11b' in msg and 'CD45' in msg   # common set named


# ── concatenate_by_label ─────────────────────────────────────────────────────

def test_concatenate_by_label_ties_across_fluors():
    a = _sample('A', {'BV421-A': 'CD11b', 'PE-Cy7-A': 'CD45'},
                data=pd.DataFrame({'FSC-A': [1, 2], 'BV421-A': [10, 20],
                                   'PE-Cy7-A': [30, 40]}))
    b = _sample('B', {'FITC-A': 'CD11b', 'PE-A': 'CD45'},
                data=pd.DataFrame({'FSC-A': [3, 4], 'FITC-A': [50, 60],
                                   'PE-A': [70, 80]}))
    merged, common = fp.concatenate_by_label([a, b])
    assert common == ['CD11b', 'CD45']
    # Columns are the common LABELS + the origin tag; detectors gone.
    assert set(merged.columns) == {'CD11b', 'CD45', 'sample_origin'}
    assert len(merged) == 4
    # A's CD11b came from BV421-A (10,20); B's from FITC-A (50,60).
    a_rows = merged[merged['sample_origin'] == 'A']
    b_rows = merged[merged['sample_origin'] == 'B']
    assert list(a_rows['CD11b']) == [10, 20]
    assert list(b_rows['CD11b']) == [50, 60]


def test_concatenate_by_label_drops_non_common():
    """CD34 (only in A) must not appear in the merged label frame."""
    a = _sample('A', {'BV421-A': 'CD11b', 'APC-A': 'CD34'},
                data=pd.DataFrame({'BV421-A': [1], 'APC-A': [2]}))
    b = _sample('B', {'FITC-A': 'CD11b'},
                data=pd.DataFrame({'FITC-A': [3]}))
    merged, common = fp.concatenate_by_label([a, b])
    assert common == ['CD11b']
    assert 'CD34' not in merged.columns
    assert set(merged.columns) == {'CD11b', 'sample_origin'}


# ── relabel_gate_for_sample (label-first gate retargeting) ───────────────────

def test_relabel_threshold_gate_by_label():
    # Template gate authored on BV421-A (CD11b); target sample has CD11b
    # on FITC-A → channel retargets to FITC-A.
    gate = {'kind': 'threshold', 'channel': 'BV421-A', 'label': 'CD11b',
            'value': 0.5}
    out = fp.relabel_gate_for_sample(gate, {'CD11b': 'FITC-A', 'CD45': 'PE-A'})
    assert out['channel'] == 'FITC-A'
    assert gate['channel'] == 'BV421-A'          # original untouched (copy)


def test_relabel_rect_gate_both_axes():
    gate = {'kind': 'rect', 'x_channel': 'BV421-A', 'y_channel': 'APC-A',
            'x_label': 'CD11b', 'y_label': 'CD34',
            'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1}
    out = fp.relabel_gate_for_sample(
        gate, {'CD11b': 'FITC-A', 'CD34': 'PE-A'})
    assert out['x_channel'] == 'FITC-A'
    assert out['y_channel'] == 'PE-A'


def test_relabel_leaves_unlabelled_channels():
    """No label stamped → channel left as-is (gate reads its detector)."""
    gate = {'kind': 'threshold', 'channel': 'BV421-A', 'value': 0.5}
    out = fp.relabel_gate_for_sample(gate, {'CD11b': 'FITC-A'})
    assert out['channel'] == 'BV421-A'


def test_relabel_label_absent_in_sample_keeps_detector():
    """Label stamped but the target sample lacks it → channel unchanged."""
    gate = {'kind': 'threshold', 'channel': 'BV421-A', 'label': 'CD11b',
            'value': 0.5}
    out = fp.relabel_gate_for_sample(gate, {'CD45': 'PE-A'})   # no CD11b
    assert out['channel'] == 'BV421-A'


def test_relabel_empty_map_is_passthrough():
    gate = {'kind': 'threshold', 'channel': 'BV421-A', 'label': 'CD11b'}
    out = fp.relabel_gate_for_sample(gate, {})
    assert out == gate and out is not gate       # copy, unchanged
