"""A replot (every zoom or axis change) used to rebuild the whole Samples &
Gates tree; the tree is now rebuilt only when what it shows changed.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import tests._background_work as shared
from tests._background_work import _gated, _quiet, _spy
from tests._tk_wait import settle

env = shared.env


def test_a_zoom_replot_leaves_the_tree_alone(env, monkeypatch):
    root, ed, tmp, _ = env
    _gated(root, ed, tmp)
    _quiet(root, ed)
    inserts = _spy(monkeypatch, ed.gate_tv, 'insert')
    ed.ax.set_xlim(70_000, 160_000)
    ed._remember_zoom()
    ed._replot()
    ed.mode_var.set('contour')
    ed._replot()
    settle(root, 0.3)
    assert inserts == [], f"tree rebuilt by a replot: {len(inserts)} rows"


def test_a_new_gate_still_rebuilds_the_tree(env, monkeypatch):
    root, ed, tmp, _ = env
    _gated(root, ed, tmp)
    inserts = _spy(monkeypatch, ed.gate_tv, 'insert')
    gid = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                        'value': 5_000.0, 'name': 'FL1 pos',
                        'parent_id': None})
    ed._replot()
    assert inserts, "a new gate did not reach the tree"
    assert ed.gate_tv.exists(ed._gate_iid('sample_a', gid))
    assert 'FL1 pos' in ed.gate_tv.item(ed._gate_iid('sample_a', gid), 'text')


def test_a_row_put_in_by_hand_is_cleared_by_the_next_refresh(env):
    """The loader adds and removes ⏳ rows itself; the next refresh must not
    take the tree for unchanged and leave them standing."""
    root, ed, tmp, _ = env
    _gated(root, ed, tmp)
    ed._sample_lb_insert_loading('sample_late')      # a load that then failed
    assert ed.gate_tv.exists(ed._sample_iid('sample_late'))
    ed._refresh_gate_list()
    assert not ed.gate_tv.exists(ed._sample_iid('sample_late'))
    assert ed.gate_tv.exists(ed._sample_iid('sample_a'))
