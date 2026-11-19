"""Dragging a gate redrew every plotted event on each mouse move and, with
gates applied, recounted every event on the window thread; it now redraws
the outline alone over a saved picture of the plot, and the counts are
made after the release by the plot cache's background worker.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import pytest

import tests._background_work as shared
from tests._background_work import _count, _gated, _on_main, _spy
from tests._tk_wait import pump_until, settle

env = shared.env


def _ev(ed, x, y, button=1):
    from types import SimpleNamespace
    px, py = ed.ax.transData.transform((x, y))
    return SimpleNamespace(inaxes=ed.ax, xdata=x, ydata=y, x=px, y=py,
                           button=button, key=None, dblclick=False)


def _scan_spy(monkeypatch):
    """(on the window thread?, the gates counted) for each plot-cache pass."""
    import openflo.editor_plotcache as epc
    calls = []
    real = epc._pcm.scan

    def scan(*a, **k):
        calls.append((_on_main(), tuple(k.get('count_gids') or ())))
        return real(*a, **k)
    monkeypatch.setattr(epc._pcm, 'scan', scan)
    return calls


@pytest.mark.parametrize('gated_view', [False, True])
def test_dragging_a_corner_redraws_the_outline_alone(env, monkeypatch,
                                                     gated_view):
    root, ed, tmp, _ = env
    gid = _gated(root, ed, tmp)
    ed.gate_display_var.set('filter' if gated_view else 'all')
    ed._replot()
    settle(root, 0.4)
    before = _count(ed, 'sample_a', gid)
    draws = _spy(monkeypatch, ed.canvas, 'draw_idle')
    blits = _spy(monkeypatch, ed.canvas, 'blit')
    replots = _spy(monkeypatch, ed, '_schedule_replot')
    scans = _scan_spy(monkeypatch)

    ed._on_press(_ev(ed, 180_000.0, 100_000.0))          # the top-right corner
    assert ed._drag_state is not None and ed._drag_state[0] == 'rect_corner'
    for step in range(1, 21):
        ed._on_motion(_ev(ed, 180_000.0 - step * 3_500.0,
                          100_000.0 - step * 2_250.0))
    assert draws == [], f"{len(draws)} full redraws during the drag"
    assert len(blits) >= 20, "the moving outline was not drawn"
    assert replots == [] and scans == [], "the drag recounted while moving"

    ed._on_release(_ev(ed, 110_000.0, 55_000.0))
    pump_until(root, lambda: not getattr(ed, '_replot_when_warm', None), 20)
    settle(root, 0.4)
    after = _count(ed, 'sample_a', gid)
    assert after < before                             # the gate really shrank
    counted = [main for main, gids in scans if gid in gids]
    assert counted and not any(counted), (
        "the moved gate was counted on the window thread", scans)
    row = ed.gate_tv.item(ed._gate_iid('sample_a', gid), 'text')
    assert f"{after:,}" in row and f"{before:,}" not in row, row
    assert ed._gate_label_artists, "labels did not come back after the drop"
    assert not any(a.get_animated() for a in ed.ax.get_children())


def test_without_the_background_worker_the_release_replots_itself(env,
                                                                  monkeypatch):
    root, ed, tmp, _ = env
    gid = _gated(root, ed, tmp)
    ed._plotcache_warm_enabled = False
    ed._on_press(_ev(ed, 180_000.0, 100_000.0))
    ed._on_motion(_ev(ed, 200_000.0, 120_000.0))
    replots = _spy(monkeypatch, ed, '_schedule_replot')
    ed._on_release(_ev(ed, 200_000.0, 120_000.0))
    assert replots == [(0,)]
    settle(root, 0.4)
    after = _count(ed, 'sample_a', gid)
    assert f"{after:,}" in ed.gate_tv.item(ed._gate_iid('sample_a', gid),
                                           'text')
