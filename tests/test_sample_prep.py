"""sample_prep.prepare_fcs_sample: QC, compensation, transform and labels,
as every loaded sample gets them, from plain data (no editor, no prefs)."""
from __future__ import annotations

import numpy as np
import pytest

from openflo import prefs
from openflo.flow_sample import FlowSample
from openflo.sample_prep import prepare_fcs_sample
from tests._fcs_writer import write_float_fcs

CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'Time']
FL = ['FL1-A', 'FL2-A', 'FL3-A']
SPILL = np.array([[1.0, 0.12, 0.03],
                  [0.08, 1.0, 0.20],
                  [0.01, 0.05, 1.0]])
# A staged matrix that differs from the file's $SPILL in every off-diagonal.
STAGED = np.array([[1.0, 0.30, 0.00],
                   [0.02, 1.0, 0.10],
                   [0.04, 0.00, 1.0]])
N_EVENTS = 3000


@pytest.fixture
def fcs(tmp_path):
    """A synthetic FCS with three spilled fluorescence channels and $SPILL."""
    rng = np.random.default_rng(11)
    true = rng.uniform(100, 60_000, (N_EVENTS, 3))
    ev = np.column_stack([rng.uniform(5e4, 1e5, N_EVENTS),
                          rng.uniform(1e4, 5e4, N_EVENTS),
                          true @ SPILL,
                          np.arange(N_EVENTS) / 10.0]).astype(np.float32)
    meta = {'SPILL': '3,FL1-A,FL2-A,FL3-A,'
                     + ','.join(map(str, SPILL.ravel()))}
    return write_float_fcs(tmp_path / 'sample_1.fcs', ev, CHANNELS,
                           metadata_dict=meta)


@pytest.fixture
def no_prefs(monkeypatch):
    """Any read of the preferences file fails the test."""
    def refuse():
        raise AssertionError('prepare_fcs_sample read the preferences')
    monkeypatch.setattr(prefs, 'read_prefs', refuse)


def _reference(path, matrix=None):
    """The same preparation by hand: the file's $SPILL (or `matrix`), then
    the default transform."""
    s = FlowSample(path)
    if matrix is None:
        s.auto_compensate()
    else:
        s.manual_compensate(matrix, FL)
    s.apply_transform()
    return s


def test_without_a_staged_matrix_the_files_spill_is_applied(fcs, no_prefs):
    s = prepare_fcs_sample(fcs)
    assert s.compensation_source == '$SPILL'
    np.testing.assert_array_equal(s.comp_matrix, SPILL)
    assert s.qc_applied is False
    ref = _reference(fcs)
    np.testing.assert_array_equal(s.data[FL].to_numpy(),
                                  ref.data[FL].to_numpy())


def test_a_staged_matrix_wins_over_the_files_spill(fcs, no_prefs):
    prep = {'comp': {'matrix': STAGED.tolist(), 'channels': FL,
                     'source': 'wsp:Custom'}}
    s = prepare_fcs_sample(fcs, prep)
    assert s.compensation_source == 'wsp:Custom'
    np.testing.assert_array_equal(s.comp_matrix, STAGED)
    ref = _reference(fcs, STAGED)
    np.testing.assert_array_equal(s.data[FL].to_numpy(),
                                  ref.data[FL].to_numpy())
    # ... and it is not the $SPILL result.
    assert not np.array_equal(s.data[FL].to_numpy(),
                              _reference(fcs).data[FL].to_numpy())


def test_a_staged_matrix_without_a_source_is_recorded_as_manual(fcs):
    s = prepare_fcs_sample(fcs, {'comp': {'matrix': STAGED, 'channels': FL}})
    assert s.compensation_source == 'manual'


def test_a_non_finite_staged_matrix_leaves_the_sample_uncompensated(fcs):
    bad = STAGED.copy()
    bad[1, 2] = np.nan
    s = prepare_fcs_sample(fcs, {'comp': {'matrix': bad, 'channels': FL}})
    assert s.compensation_source is None
    assert s.comp_matrix is None
    assert len(s.load_warnings) == 1
    assert 'the saved spillover matrix was ignored' in s.load_warnings[0]
    assert 'FL2-A' in s.load_warnings[0]
    plain = FlowSample(fcs)
    plain.apply_transform()
    np.testing.assert_array_equal(s.data[FL].to_numpy(),
                                  plain.data[FL].to_numpy())


@pytest.mark.parametrize('staged, default, ran', [
    (None, False, False),
    (None, True, True),       # nothing staged: the caller's default
    (False, True, False),     # a staged choice wins over the default
    (True, False, True),
])
def test_qc_follows_the_staged_choice_else_the_default(fcs, no_prefs, staged,
                                                       default, ran):
    s = prepare_fcs_sample(fcs, {'qc': staged}, qc_default=default)
    assert s.qc_applied is ran


def test_labels_are_applied(fcs, no_prefs):
    s = prepare_fcs_sample(fcs, labels='FL1-A=M1; FL3-A = M3')
    assert s.channel_labels['FL1-A'] == 'M1'
    assert s.channel_labels['FL3-A'] == 'M3'
    assert s.channel_labels.get('FL2-A') != 'M1'


@pytest.mark.parametrize('prep', [
    {'qc': True},
    {'qc': True, 'comp': {'matrix': STAGED, 'channels': FL}},
])
def test_qc_runs_on_the_raw_events_before_compensation(fcs, no_prefs,
                                                       monkeypatch, prep):
    """QC flags events by the acquisition values, so it runs before any
    compensation; the transform and labels come after both."""
    calls = []
    for meth in ('run_qc', 'auto_compensate', 'manual_compensate',
                 'apply_transform', 'set_labels'):
        orig = getattr(FlowSample, meth)

        def spy(self, *a, _orig=orig, _meth=meth, **k):
            calls.append(_meth)
            return _orig(self, *a, **k)
        monkeypatch.setattr(FlowSample, meth, spy)
    prepare_fcs_sample(fcs, prep, labels='FL1-A=M1')
    comp = 'manual_compensate' if 'comp' in prep else 'auto_compensate'
    assert calls == ['run_qc', comp, 'apply_transform', 'set_labels']


def test_stored_transforms_names_the_channels_stored_off_linear(fcs):
    """The space a sample's gates are in: its logicle channels, not one
    recorded as linear (the editor's _stored_transforms and the
    Document's workspace import both read it)."""
    from openflo.sample_prep import stored_transforms
    s = prepare_fcs_sample(fcs)
    # A record may name a channel as linear (a library export's record).
    s.data_transforms = {**s.data_transforms, 'FL3-A': {'method': 'linear'}}
    got = stored_transforms(s)
    assert sorted(got) == ['FL1-A', 'FL2-A']
    assert all(spec['method'] == 'logicle' for spec in got.values())
    assert got['FL1-A'] == s.data_transforms['FL1-A']
    assert got['FL1-A'] is not s.data_transforms['FL1-A']      # a copy


def test_the_editors_sidecar_loader_passes_its_window_labels(fcs, tmp_path):
    """The editor's pool worker for a processed-data sidecar hands the
    window's label snapshot to ``sidecar_sample``, as the raw loader applies
    its labels string (a stand-in window, no Tk)."""
    from openflo.editor_loadpool import LoadPoolMixin
    from openflo.session_sidecar import write_frame
    s = prepare_fcs_sample(fcs)
    side = str(tmp_path / 'sample_1.npz')
    assert write_frame(s.data, side)

    class Window:
        def __init__(self):
            self.loaded = {}

        def _take_pending_prep(self, _path):
            return {}

        def after(self, _ms, fn):
            fn()

        def _on_loaded(self, name, sample):
            self.loaded[name] = sample

        def _note_transform_assumption(self, _note):
            pass

        def _on_load_error(self, _name, exc):
            raise exc

        _post_load_result = LoadPoolMixin._post_load_result

    win = Window()
    LoadPoolMixin._load_csv_worker(win, 'sample_1', side,
                                   {'FL1-A': 'Lab one'})
    got = win.loaded['sample_1'].channel_labels
    assert got['FL1-A'] == 'Lab one' and got['FL2-A'] == 'FL2-A'
