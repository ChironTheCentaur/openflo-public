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


# ── one matcher: the library pairs channels as the CLI does ─────────────────
#
# align_fluor_labels pairs through cli.align_shared_channels: spelling and
# fluor words fold away, an unlabelled channel pairs by detector, and
# -A/-H/-W stay apart.

def _frame(**cols):
    return pd.DataFrame(cols)


def test_case_and_punctuation_pair_into_one_column():
    """'CD3' in A and 'cd-3' in B are one marker: one column, each sample's
    values from its own detector, named by the first sample's label."""
    a = _sample('A', {'FL1-A': 'CD3', 'FL2-A': 'CD902'},
                data=_frame(**{'FL1-A': [1.0, 2.0], 'FL2-A': [3.0, 4.0]}))
    b = _sample('B', {'FL4-A': 'cd-3', 'FL2-A': 'CD902'},
                data=_frame(**{'FL4-A': [5.0, 6.0], 'FL2-A': [7.0, 8.0]}))
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['CD3', 'CD902']
    assert info['missing'] == {}
    assert info['per_sample']['B']['CD3'] == 'FL4-A'
    merged, common = fp.concatenate_by_label([a, b])
    assert common == ['CD3', 'CD902']
    assert list(merged.loc[merged['sample_origin'] == 'A', 'CD3']) == [1, 2]
    assert list(merged.loc[merged['sample_origin'] == 'B', 'CD3']) == [5, 6]


def test_trailing_fluor_word_pairs_with_bare_label():
    """'CD901b FL1' names the marker CD901b on its fluor: it pairs with a
    bare 'CD901b' on another detector."""
    a = _sample('A', {'FL1-A': 'CD901b FL1', 'FL3-A': 'CD903'})
    b = _sample('B', {'FL2-A': 'CD901b', 'FL3-A': 'CD903'})
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['CD901b FL1', 'CD903']
    assert info['per_sample']['B']['CD901b FL1'] == 'FL2-A'
    assert info['missing'] == {}


def test_two_labels_on_one_detector_are_not_pooled():
    """CD902 on FL2-A in A and CD903 on FL2-A in B are different markers:
    no column holds both, and each is reported missing from the other.
    (A guard: the exact-string matcher got this right too; pairing by
    detector must not break it.)"""
    a = _sample('A', {'FL1-A': 'CD901b', 'FL2-A': 'CD902'},
                data=_frame(**{'FL1-A': [1.0], 'FL2-A': [2.0]}))
    b = _sample('B', {'FL1-A': 'CD901b', 'FL2-A': 'CD903'},
                data=_frame(**{'FL1-A': [3.0], 'FL2-A': [4.0]}))
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['CD901b']
    assert info['missing'] == {'CD902': ['B'], 'CD903': ['A']}
    merged, common = fp.concatenate_by_label([a, b])
    assert set(merged.columns) == {'CD901b', 'sample_origin'}


def test_unlabelled_channel_pairs_by_detector():
    """FL2-A carries no label in A and CD902 in B: the same detector with
    one marker on it is one column, named by the label."""
    a = _sample('A', {'FL1-A': 'CD901b', 'FL2-A': 'FL2-A'},
                data=_frame(**{'FL1-A': [1.0], 'FL2-A': [2.0]}))
    b = _sample('B', {'FL1-A': 'CD901b', 'FL2-A': 'CD902'},
                data=_frame(**{'FL1-A': [3.0], 'FL2-A': [4.0]}))
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['CD901b', 'CD902']
    assert info['missing'] == {}
    merged, _common = fp.concatenate_by_label([a, b])
    assert list(merged['CD902']) == [2.0, 4.0]


def test_warning_empty_when_panels_differ_only_by_spelling():
    a = _sample('A', {'FL1-A': 'CD901b', 'FL2-A': 'CD3'})
    b = _sample('B', {'FL4-A': 'cd901B', 'FL5-A': 'CD 3'})
    assert fp.common_fluor_warning([a, b]) == ''


def test_label_on_area_never_pairs_with_label_on_height():
    """CD3 on FL1-A in A and CD3 on FL1-H in B are different measurements
    (area vs height): not one column, each reported missing."""
    a = _sample('A', {'FL1-A': 'CD3', 'FL2-A': 'CD902'})
    b = _sample('B', {'FL1-H': 'CD3', 'FL2-A': 'CD902'})
    info = fp.align_fluor_labels([a, b])
    assert info['common'] == ['CD902']
    assert info['missing'] == {'CD3 (FL1-A)': ['B'], 'CD3 (FL1-H)': ['A']}
    assert 'CD3 (FL1-A): missing from B' in fp.common_fluor_warning([a, b])
