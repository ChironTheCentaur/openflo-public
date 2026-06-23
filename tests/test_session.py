"""Editor session (.flowsession) serialization contract.

Tests `ViewGateEditorWindow._session_state` against a SimpleNamespace
stub (no Tk) — same duck-typing approach as test_gate_editor_helpers.
The serialised dict must JSON round-trip and preserve everything the
session is supposed to carry that a .wsp / template can't: per-sample
gates at full fidelity (incl. ellipsoid), per-channel scale + range,
plot mode, channel labels, downsample toggles, and the reserved
cluster-labels slot.

The async restore path (_apply_session → threaded FCS reload) needs a
live Tk + real FCS, so it's exercised manually / by the GUI smoke
test rather than here — this locks down the data contract.
"""
# pyright: reportArgumentType=false, reportCallIssue=false
from __future__ import annotations

import json
import os
import types

import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

try:
    import tkinter as _tk  # noqa: F401

    from openflo.gui import ViewGateEditorWindow as V
except (ImportError, RuntimeError) as e:
    pytest.skip(f"openflo.gui not importable: {e}", allow_module_level=True)


class _Combo:
    def __init__(self, value): self._v = value
    def get(self): return self._v


class _Var:
    def __init__(self, value): self._v = value
    def get(self): return self._v


class _Sample:
    def __init__(self, path): self.path = path


def _session_stub():
    s = types.SimpleNamespace()
    s._sample_order = ['A', 'B']
    s._samples = {'A': _Sample('/data/a.fcs'), 'B': _Sample('/data/b.fcs')}
    s._sample_colors = {'A': '#1f77b4', 'B': '#ff7f0e'}
    s._sample_plot_enabled = {'A': True, 'B': False}
    s._sample_trial = {'A': 'Day 3', 'B': 'Day 6'}
    s._sample_is_comp = {'B': True}        # B manually moved to Comps
    s._sample_gates = {
        'A': {
            'g1': {'kind': 'threshold', 'channel': 'BV421-A',
                   'value': 1000.0, 'parent_id': None, 'color': '#e6194b',
                   'enabled': True},
            'g2': {'kind': 'ellipsoid', 'x_channel': 'APC-A',
                   'y_channel': 'PE-Cy7-A', 'mean': [1.0, 2.0],
                   'cov': [[4.0, 0.0], [0.0, 9.0]], 'distance_sq': 4.0,
                   'parent_id': 'g1', 'color': '#3cb44b', 'enabled': True},
        },
        'B': {},
    }
    s._sample_gate_order = {'A': ['g1', 'g2'], 'B': []}
    s._channel_scale = {'BV421-A': 'symlog', 'APC-A': 'log'}
    s._channel_range = {'BV421-A': (0.0, 5.0)}
    s._channel_labels = {'BV421-A': 'CD11b'}
    s._active_sample = 'A'
    s._cluster_labels = {}
    from openflo.audit import AuditLog
    s._audit_log = AuditLog()
    s.mode_var = _Var('pseudocolor')
    s.x_combo = _Combo('APC-A (CD34)')
    s.y_combo = _Combo('PE-Cy7-A (CD45)')
    s.color_combo = _Combo('density')
    s.ds_display_var = _Var(True)
    s.ds_propagate_var = _Var(False)
    s.max_points_var = _Var('60000')
    s.show_removed_var = _Var(False)
    s.contour_scatter_var = _Var(True)
    s.contour_outliers_var = _Var(True)
    s.hist_y_mode = _Var('Count')
    return s


def test_session_state_structure_and_json_round_trip():
    stub = _session_stub()
    state = V._session_state(stub)

    # Must be JSON-serialisable.
    blob = json.dumps(state)
    restored = json.loads(blob)
    assert restored == state, "session dict must survive a JSON round-trip"

    assert state['format'] == 'openflo-session'
    assert state['version'] == 1
    assert state['active_sample'] == 'A'


def test_session_state_samples_carry_path_color_and_enabled():
    state = V._session_state(_session_stub())
    by_name = {s['name']: s for s in state['samples']}
    assert by_name['A']['path'] == '/data/a.fcs'
    assert by_name['A']['color'] == '#1f77b4'
    assert by_name['A']['plot_enabled'] is True
    assert by_name['B']['plot_enabled'] is False
    # Grouping persists: trial for both; Comps override only where set.
    assert by_name['A']['trial'] == 'Day 3'
    assert by_name['B']['trial'] == 'Day 6'
    assert 'is_comp' not in by_name['A']          # untouched → name-based on load
    assert by_name['B']['is_comp'] is True        # manual move persisted


def test_session_state_preserves_gate_fidelity():
    """The ellipsoid gate — which a template keeps but a .wsp-only
    workflow would have to translate — must serialise with its mean,
    cov, distance_sq, colour, enabled flag, and parent link intact."""
    state = V._session_state(_session_stub())
    gates_a = state['sample_gates']['A']
    assert [g['id'] for g in gates_a] == ['g1', 'g2'], "order preserved"
    ell = next(g for g in gates_a if g['kind'] == 'ellipsoid')
    assert ell['mean'] == [1.0, 2.0]
    assert ell['cov'] == [[4.0, 0.0], [0.0, 9.0]]
    assert ell['distance_sq'] == 4.0
    assert ell['color'] == '#3cb44b'
    assert ell['parent_id'] == 'g1'      # hierarchy link survives


def test_session_state_display_config():
    state = V._session_state(_session_stub())
    assert state['channel_scale'] == {'BV421-A': 'symlog', 'APC-A': 'log'}
    # tuple range → JSON list
    assert state['channel_range'] == {'BV421-A': [0.0, 5.0]}
    assert state['channel_labels'] == {'BV421-A': 'CD11b'}
    assert state['plot_mode'] == 'pseudocolor'
    assert state['downsample_display'] is True
    assert state['downsample_propagate'] is False
    assert state['hist_y_mode'] == 'Count'


def test_session_state_reserves_cluster_labels_slot():
    """Cluster labels are reserved (empty here — the editor doesn't
    cluster) but the key MUST exist so the schema is forward-compatible."""
    state = V._session_state(_session_stub())
    assert 'cluster_labels' in state
    assert state['cluster_labels'] == {}


def test_session_state_embeds_audit_trail():
    """The provenance trail round-trips inside the session as a JSON list."""
    stub = _session_stub()
    stub._audit_log.record('sample.load', time='t1', name='A', n_events=10)
    state = V._session_state(stub)
    assert 'audit' in state
    assert isinstance(state['audit'], list)
    assert state['audit'][0]['action'] == 'sample.load'
    # The whole state must stay JSON-serialisable.
    import json
    json.loads(json.dumps(state))


def test_session_autosave_path_under_home():
    stub = types.SimpleNamespace()
    stub.SESSION_EXT = V.SESSION_EXT     # class attr the method reads off self
    p = V._session_autosave_path(stub)
    assert p.endswith('.flowsession')
    assert '.openflo' in p
    # Directory should have been created.
    assert os.path.isdir(os.path.dirname(p))


# ── _wsp_lossy_summary (Phase 4: warn before a lossy .wsp export) ────────────

def _lossy_stub(scales=None, ranges=None, gates=None, cluster_labels=None):
    s = types.SimpleNamespace()
    s._channel_scale = dict(scales or {})
    s._channel_range = dict(ranges or {})
    s._sample_gates = gates or {}
    s._cluster_labels = dict(cluster_labels or {})
    s._default_channel_scale = 'symlog'
    return s


def test_lossy_summary_empty_for_plain_state():
    """Default scales, no custom ranges, all gates enabled, no cluster
    labels → nothing surprising is lost, so no warning."""
    stub = _lossy_stub(
        scales={'BV421-A': 'symlog'},   # == default
        gates={'A': {'g1': {'kind': 'rect', 'enabled': True}}},
    )
    assert V._wsp_lossy_summary(stub) == []


def test_lossy_summary_flags_custom_scale():
    stub = _lossy_stub(scales={'BV421-A': 'log'})   # != default symlog
    summary = V._wsp_lossy_summary(stub)
    assert any('axis scale' in s for s in summary)


def test_lossy_summary_flags_custom_range():
    stub = _lossy_stub(ranges={'APC-A': (0.0, 5.0)})
    summary = V._wsp_lossy_summary(stub)
    assert any('display range' in s for s in summary)


def test_lossy_summary_flags_disabled_gates():
    stub = _lossy_stub(gates={
        'A': {'g1': {'kind': 'rect', 'enabled': True},
              'g2': {'kind': 'rect', 'enabled': False}},
    })
    summary = V._wsp_lossy_summary(stub)
    assert any('disabled gate' in s for s in summary)


def test_lossy_summary_flags_cluster_labels():
    stub = _lossy_stub(cluster_labels={'A': {0: 'MkP'}})
    summary = V._wsp_lossy_summary(stub)
    assert any('cluster' in s for s in summary)


def test_lossy_summary_combines_multiple():
    stub = _lossy_stub(
        scales={'X': 'log'},
        ranges={'Y': (1.0, 2.0)},
        gates={'A': {'g': {'kind': 'rect', 'enabled': False}}},
        cluster_labels={'A': {1: 'HSC'}},
    )
    summary = V._wsp_lossy_summary(stub)
    assert len(summary) == 4
