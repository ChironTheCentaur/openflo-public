"""Compensation is applied exactly once, with the right matrix, on every path.

Two failure modes were reported against 2.6.1 -- compensation "either ignored
or applied twice" -- and both were real:

* applied twice: `FlowSample._apply_comp` multiplied whatever `data` held by
  inv(M), so a second call (auto_compensate() then compensate_from_wsp(), the
  same call twice, FMOGater.prepare() twice) compensated compensated values;
* ignored / wrong matrix: compensate_from_wsp() took the workspace's FIRST
  matrix for every file, a workspace without matrices raised instead of using
  each file's $SPILL as FlowJo does, the editor's .wsp import never applied
  the workspace's per-sample matrix, and a session reload or a transfer to
  another window swapped the applied matrix for the FCS's own $SPILL.

Each test drives a known spillover through the real code and checks that the
TRUE signal comes back -- not merely that a matrix was stored.
"""
from __future__ import annotations

import logging
import types

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp
from openflo.session_format import comp_entry, comp_from_entry

CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'Time']
FLUOR = ['FL1-A', 'FL2-A']
MA = np.array([[1.0, 0.30], [0.05, 1.0]])
MB = np.array([[1.0, 0.10], [0.20, 1.0]])
N = 600


def _true_signal(seed=0):
    rng = np.random.default_rng(seed)
    t = np.zeros((N, 2))
    t[: N // 2, 0] = rng.uniform(1_000, 50_000, N // 2)
    t[N // 2:, 1] = rng.uniform(1_000, 50_000, N - N // 2)
    return t


TRUE = _true_signal()


def _fcs(path, spill, *, embed_spill=True):
    """FCS whose fluorescence is TRUE measured through `spill`, carrying
    `spill` as its $SPILL unless `embed_spill` is False."""
    import flowio
    rng = np.random.default_rng(1)
    measured = TRUE @ spill
    ev = np.column_stack([rng.uniform(5e4, 1e5, N), rng.uniform(1e4, 5e4, N),
                          measured, np.arange(N) / 10.0]).astype(np.float32)
    meta = ({'SPILL': '2,FL1-A,FL2-A,' + ','.join(map(str, spill.ravel()))}
            if embed_spill else {})
    with open(path, 'wb') as f:
        flowio.create_fcs(f, ev.ravel().tolist(), CHANNELS, metadata_dict=meta)
    return str(path)


def _err(s):
    return float(np.max(np.abs(s.data[FLUOR].to_numpy(float) - TRUE)))


def _wsp(tmp_path, samples, name='x.wsp'):
    """`samples` = [(fcs_path, matrix-or-None)]."""
    w = fp.WspWriter(cytometer='T')
    for k, (p, m) in enumerate(samples):
        w.add_sample(f's{k}', p, CHANNELS, [],
                     compensation=None if m is None else (FLUOR, m))
    out = tmp_path / name
    w.write(str(out))
    return str(out)


# ── applied twice ───────────────────────────────────────────────────────────

def test_compensating_twice_is_the_same_as_once(tmp_path):
    s = fp.FlowSample(_fcs(tmp_path / 'a.fcs', MA))
    s.auto_compensate()
    once = s.data[FLUOR].to_numpy(float).copy()
    s.auto_compensate()
    np.testing.assert_allclose(s.data[FLUOR].to_numpy(float), once,
                               rtol=1e-9, atol=1e-6)
    assert _err(s) < 0.05


def test_a_second_matrix_replaces_the_first(tmp_path):
    """auto_compensate() with the wrong matrix, then the right one: the result
    is the right one alone, not the product of both."""
    s = fp.FlowSample(_fcs(tmp_path / 'a.fcs', MA, embed_spill=False))
    s.manual_compensate(MB, FLUOR)                  # wrong matrix first
    assert _err(s) > 1_000
    s.manual_compensate(MA, FLUOR)                  # then the right one
    assert _err(s) < 0.05
    np.testing.assert_array_equal(s.comp_matrix, MA)


def test_fmo_prepare_twice_does_not_compound(tmp_path):
    g = fp.FMOGater()
    g.add_fmo('FL1-A', _fcs(tmp_path / 'fmo.fcs', MA))
    g.prepare()
    first = g.fmos['FL1-A'].data[FLUOR].to_numpy(float).copy()
    g.prepare()
    np.testing.assert_allclose(g.fmos['FL1-A'].data[FLUOR].to_numpy(float),
                               first, rtol=1e-7, atol=1e-7)


def test_compensation_after_the_transform_is_refused(tmp_path):
    s = fp.FlowSample(_fcs(tmp_path / 'a.fcs', MA))
    s.auto_compensate().apply_transform()
    before = s.data[FLUOR].to_numpy(float).copy()
    with pytest.raises(fp.CompensationError, match='transform'):
        s.manual_compensate(MB, FLUOR)
    np.testing.assert_array_equal(s.data[FLUOR].to_numpy(float), before)


def test_a_misshapen_matrix_is_an_error_not_a_guess():
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame(TRUE @ MA, columns=FLUOR), name='x')
    with pytest.raises(fp.CompensationError):
        s.manual_compensate(np.eye(3), FLUOR)


def test_singular_matrix_keeps_the_one_already_applied():
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame(TRUE @ MA, columns=FLUOR), name='x')
    s.manual_compensate(MA, FLUOR)
    s.manual_compensate(np.ones((2, 2)), FLUOR)       # singular: skipped
    assert _err(s) < 1e-6
    np.testing.assert_array_equal(s.comp_matrix, MA)


def test_a_partial_channel_match_is_named(caplog):
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame(TRUE @ MA, columns=FLUOR), name='x')
    m = np.eye(3)
    m[:2, :2] = MA
    with caplog.at_level(logging.WARNING):
        s.manual_compensate(m, FLUOR + ['FL9-A'])
    assert 'FL9-A' in caplog.text
    assert _err(s) < 1e-6


def test_compensation_source_is_recorded(tmp_path):
    s = fp.FlowSample(_fcs(tmp_path / 'a.fcs', MA))
    assert s.compensation_source is None and not s.is_compensated()
    s.auto_compensate()
    assert s.compensation_source == '$SPILL' and s.is_compensated()


# ── ignored / wrong matrix: .wsp ────────────────────────────────────────────

def test_each_sample_gets_its_own_wsp_matrix(tmp_path):
    """The FCS files carry NO $SPILL, so the only correct source is the
    sample's own matrix in the workspace."""
    a = _fcs(tmp_path / 'a.fcs', MA, embed_spill=False)
    b = _fcs(tmp_path / 'b.fcs', MB, embed_spill=False)
    wsp = _wsp(tmp_path, [(a, MA), (b, MB)])
    for path, m in ((a, MA), (b, MB)):
        s = fp.FlowSample(path)
        s.compensate_from_wsp(wsp)
        assert _err(s) < 0.05, f'{path} was compensated with another matrix'
        np.testing.assert_array_equal(s.comp_matrix, m)


def test_a_sample_without_its_own_matrix_uses_its_spill(tmp_path):
    """FlowJo's Acquisition-defined matrix: the file's own $SPILL. A workspace
    with no matrix at all used to raise 'No matrices available.'"""
    a = _fcs(tmp_path / 'a.fcs', MA)
    wsp = _wsp(tmp_path, [(a, None)])
    s = fp.FlowSample(a)
    s.compensate_from_wsp(wsp)
    assert s.compensation_source == '$SPILL'
    assert _err(s) < 0.05


def test_compare_compensates_each_sample_with_its_own_matrix(tmp_path):
    from openflo.compare import _compare_one_sample
    b = _fcs(tmp_path / 'b.fcs', MB, embed_spill=False)
    a = _fcs(tmp_path / 'a.fcs', MA, embed_spill=False)
    wsp = _wsp(tmp_path, [(a, MA), (b, MB)])
    # FL1 > 500 selects exactly the first half once compensation is right;
    # with A's 0.30 spillover left in FL2 (or B's 0.20 in FL1) it does not.
    pops = [(1, 'FL1+', N // 2,
             {'kind': 'interval', 'channel': 'FL1-A', 'lo': 500.0, 'hi': 1e9},
             None)]
    for path in (a, b):
        rows = _compare_one_sample('s', path, pops, wsp_path=wsp)
        assert rows[0]['openflo_count'] == N // 2, rows[0]
        assert not rows[0].get('error')


def test_same_named_matrices_are_all_kept(tmp_path):
    """FlowJo names every file's own SPILL 'Acquisition-defined'. Keyed by
    bare name, only the last survived."""
    import xml.etree.ElementTree as ET
    a = _fcs(tmp_path / 'a.fcs', MA)
    b = _fcs(tmp_path / 'b.fcs', MB)
    wsp = _wsp(tmp_path, [(a, MA), (b, MB)])
    tree = ET.parse(wsp)
    for el in tree.iter():
        if el.tag.endswith('spilloverMatrix'):
            el.set('name', 'Acquisition-defined')
    tree.write(wsp)
    r = fp.WspReader(wsp)
    got = sorted(tuple(np.round(m['matrix'].ravel(), 6))
                 for m in r.matrices.values())
    assert got == sorted([tuple(MA.ravel()), tuple(MB.ravel())])


# ── ignored: session / transfer / editor load ───────────────────────────────

def test_session_comp_entry_round_trips():
    s = types.SimpleNamespace(comp_matrix=MA, comp_channels=FLUOR,
                              compensation_source='wsp:Beads')
    d = comp_entry(s)
    import json
    back = comp_from_entry(json.loads(json.dumps(d)))
    assert back is not None
    assert back['channels'] == FLUOR and back['source'] == 'wsp:Beads'
    np.testing.assert_array_equal(back['matrix'], MA)
    assert comp_entry(types.SimpleNamespace(comp_matrix=None,
                                            comp_channels=[])) is None


@pytest.mark.parametrize('bad', [
    None, 'x', {'channels': FLUOR}, {'channels': FLUOR, 'matrix': [[1.0]]},
    {'channels': FLUOR, 'matrix': [[1.0, float('nan')], [0.0, 1.0]]},
    {'channels': [], 'matrix': []},
])
def test_a_damaged_session_matrix_is_ignored_not_fatal(bad):
    assert comp_from_entry(bad) is None


def _loader_stub():
    pytest.importorskip('tkinter')
    from openflo.editor_loadpool import LoadPoolMixin

    class _Stub(LoadPoolMixin):
        labels_str = ''

        def __init__(self):
            self._pending_sample_prep = {}
            self.loaded = {}
            self.errors = {}

        def after(self, _ms, fn):
            fn()

        def _on_loaded(self, name, s):
            self.loaded[name] = s

        def _on_load_error(self, name, exc):
            self.errors[name] = exc
    return _Stub()


def test_editor_load_applies_a_staged_matrix_instead_of_spill(tmp_path):
    """The FCS's $SPILL is the WRONG matrix (MB); the staged one (from a .wsp,
    a saved session or another window) is right (MA)."""
    p = _fcs(tmp_path / 'a.fcs', MA)
    import flowio
    fd = flowio.FlowData(p)       # rewrite with a misleading $SPILL
    ev = np.reshape(np.asarray(fd.events), (-1, fd.channel_count))
    with open(p, 'wb') as f:
        flowio.create_fcs(f, ev.ravel().tolist(), CHANNELS, metadata_dict={
            'SPILL': '2,FL1-A,FL2-A,' + ','.join(map(str, MB.ravel()))})
    ed = _loader_stub()
    ed.stage_sample_prep(p, comp={'channels': FLUOR, 'matrix': MA,
                                  'source': 'wsp:Beads'})
    ed._load_worker('a', p)
    s = ed.loaded['a']
    np.testing.assert_array_equal(s.comp_matrix, MA)
    assert s.compensation_source == 'wsp:Beads'
    assert not ed._pending_sample_prep, 'staged prep must be consumed once'
    # Without a staged matrix the loader keeps using $SPILL.
    ed._load_worker('b', p)
    np.testing.assert_array_equal(ed.loaded['b'].comp_matrix, MB)


def test_editor_csv_load_reattaches_the_session_matrix(tmp_path):
    csv = tmp_path / 'a.csv'
    pd.DataFrame(TRUE, columns=FLUOR).to_csv(csv, index=False)
    ed = _loader_stub()
    ed.stage_sample_prep(str(csv), comp={'channels': FLUOR, 'matrix': MA,
                                         'source': '$SPILL'},
                         source_fcs='/data/a.fcs')
    ed._load_csv_worker('a', str(csv), {})
    s = ed.loaded['a']
    np.testing.assert_array_equal(s.comp_matrix, MA)
    assert s.comp_channels == FLUOR and s.source_fcs == '/data/a.fcs'
    # the CSV values are already compensated: nothing was re-applied
    np.testing.assert_allclose(s.data[FLUOR].to_numpy(float), TRUE)


def test_editor_wsp_import_stages_each_samples_own_matrix(tmp_path):
    pytest.importorskip('tkinter')
    from openflo.editor_load import LoadMixin
    a = _fcs(tmp_path / 'a.fcs', MA, embed_spill=False)
    b = _fcs(tmp_path / 'b.fcs', MB)
    wsp = _wsp(tmp_path, [(a, MA), (b, None)])
    ed = _loader_stub()
    ed.fcs_dir = ''
    ed._samples = {}
    ed._pending_sample_gates = {}
    ed.status_var = types.SimpleNamespace(set=lambda _m: None)
    ed._sample_name_for = lambda p: p.rsplit('/', 1)[-1][:-4]
    queued = []
    ed._queue_fcs_loads = lambda paths: queued.extend(paths)
    LoadMixin._ingest_wsp(ed, wsp)
    assert sorted(queued) == sorted([a, b])
    for p in queued:
        ed._load_worker(p, p)
    np.testing.assert_array_equal(ed.loaded[a].comp_matrix, MA)
    assert ed.loaded[a].compensation_source.startswith('wsp:')
    # b has no matrix of its own in the workspace: its $SPILL applies
    assert ed.loaded[b].compensation_source == '$SPILL'


def test_session_state_records_the_applied_matrix():
    pytest.importorskip('tkinter')
    try:
        from openflo.gui import ViewGateEditorWindow as V
    except (ImportError, RuntimeError) as e:
        pytest.skip(f'openflo.gui not importable: {e}')

    class _V:
        def __init__(self, v): self._v = v
        def get(self): return self._v
    smp = types.SimpleNamespace(path='/data/a.csv', source_fcs='/data/a.fcs',
                                comp_matrix=MA, comp_channels=FLUOR,
                                compensation_source='manual')
    st = types.SimpleNamespace(
        _sample_order=['a'], _samples={'a': smp}, _sample_colors={},
        _sample_plot_enabled={}, _sample_trial={}, _sample_is_comp={},
        _sample_gates={}, _sample_gate_order={}, _channel_range={},
        _channel_scale={}, _channel_labels={}, _active_sample='a',
        mode_var=_V('scatter'), x_combo=_V(''), y_combo=_V(''),
        color_combo=_V(''), ds_display_var=_V(False),
        ds_propagate_var=_V(False), max_points_var=_V(1000),
        show_removed_var=_V(False), contour_scatter_var=_V(False),
        contour_outliers_var=_V(False), hist_y_mode=_V('count'),
        _cluster_labels={},
        _audit_log=types.SimpleNamespace(to_list=lambda: []))
    entry = V._session_state(st)['samples'][0]
    assert entry['path'] == '/data/a.fcs', 'the raw FCS link must survive'
    back = comp_from_entry(entry['comp'])
    assert back is not None
    np.testing.assert_array_equal(back['matrix'], MA)
