"""Files written by OpenFlo 2.6.1 (no transform records) still restore as
2.6.1 restored them.

tests/golden/v261_asinh_session/ was written by the v2.6.1 code itself
(session, its processed sidecar session_data/A.csv, and a workspace
run/T_A_events.csv), with FL1-A on asinh: sample A is compensated (so 2.6.1
gave it a sidecar holding the asinh values as displayed) and carries a U:X
column in 2.6.1's on-disk shape (linear, unmixed from displayed values);
sample B is not (it reloads from its raw FCS). Only the machine-specific
absolute FCS paths were rewritten, so the restore relinks through the
session's own rel_path. The FCS files are regenerated here byte for byte
(*.fcs is not committed), and expected_v261.json holds the counts 2.6.1
showed before saving and after its own restore in both load orders.

Before provenance, a sidecar column outside the logicle range was taken as
LINEAR and conformed: A's gated counts went 215 -> 0 when B loaded first, and
B's went 191 -> 400 when A loaded first (the editor was seeded 'linear').
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
    settle,
    the_dialog,
)

GOLD = Path(__file__).parent / 'golden' / 'v261_asinh_session'
CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
MK = ['', '', 'CD901b', 'CD902', 'CD903']
SPILL = '3,FL1-A,FL2-A,FL3-A,1,0.1,0,0,1,0.05,0,0,1'
SPILL_M = np.array([[1, .1, 0], [0, 1, .05], [0, 0, 1]], float)


def _fcs(path, seed, spill, n=400):
    """The generator that wrote the fixture's FCS (same seeds -> same bytes)."""
    import flowio
    rng = np.random.default_rng(seed)
    true = np.column_stack([
        rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
        rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
        rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
        rng.normal(800, 100, n)])
    obs = true.copy()
    md = {}
    if spill:
        obs[:, 2:5] = true[:, 2:5] @ SPILL_M
        md = {'SPILL': SPILL}
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, obs.astype(np.float32).flatten().tolist(), CH,
                          opt_channel_names=MK, metadata_dict=md)
    return str(path)


@pytest.fixture
def shown(monkeypatch):
    return capture_warnings(monkeypatch)


@pytest.fixture
def v261(tmp_path, monkeypatch, shown):
    """The 2.6.1 project unpacked in a temp dir, a Tk root, and the expected
    numbers. Warning dialogs are captured (`shown`), never displayed."""
    isolate_home(monkeypatch, tmp_path)
    proj = tmp_path / 'project'
    shutil.copytree(GOLD, proj)
    (proj / 'session.flowsession.json').rename(proj / 'session.flowsession')
    (proj / 'fcs').mkdir()
    _fcs(proj / 'fcs' / 'A.fcs', 1, True)
    _fcs(proj / 'fcs' / 'B.fcs', 2, False)
    expected = json.loads((proj / 'expected_v261.json').read_text('utf-8'))
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, proj, expected
    root.destroy()


_settle = settle


def _restore(root, session, raw_first):
    """Reopen `session` in a fresh editor, running the restore's own loaders
    on this thread, raw-FCS samples first or last."""
    ed = _new_editor(root)
    jobs = {'fcs': [], 'csv': []}
    ed._queue_fcs_loads = lambda paths, front_names=(): jobs['fcs'].extend(
        paths)
    ed._queue_processed_loads = lambda items, front_names=(): jobs[
        'csv'].extend(items)
    ed._load_session_path(str(session))
    labels = dict(ed._channel_labels)          # what the restore snapshots
    for kind in (['fcs', 'csv'] if raw_first else ['csv', 'fcs']):
        for job in jobs[kind]:
            if kind == 'fcs':
                ed._load_worker(ed._sample_name_for(job), job)
            else:
                ed._load_csv_worker(job[0], job[1], labels)
            root.update()
    _settle(root)
    assert len(ed._samples) == 2, ed.status_var.get()
    return ed


def _counts(ed):
    rows, _ = ed._collect_stats_rows({'Count'})
    return {f"{r['Sample']}|{r['Population']}": int(r['Count']) for r in rows}


def test_the_fixture_fcs_are_the_ones_v261_loaded(v261):
    """A guard on the fixture itself: what 2.6.1 counted before saving is what
    these regenerated files give a fresh load, so B's numbers below are not an
    artefact of different data."""
    root, proj, expected = v261
    assert expected['openflo'] == '2.6.1'
    before = expected['counts_before_save']
    # 2.6.1 got A right in both orders and B wrong (its raw FCS came back on
    # logicle under an asinh gate): the cases the branch must match.
    for order in ('raw_first', 'csv_first'):
        got = expected['v261_restore'][order]['counts']
        assert {k: v for k, v in got.items() if k.startswith('A|')} == {
            k: v for k, v in before.items() if k.startswith('A|')}
        assert got['B|CD901b+'] == 0 != before['B|CD901b+']


@pytest.mark.parametrize('raw_first', [True, False])
def test_a_v261_asinh_session_restores_with_v261s_counts(v261, raw_first):
    root, proj, expected = v261
    order = 'raw_first' if raw_first else 'csv_first'
    ed = _restore(root, proj / 'session.flowsession', raw_first)
    want = expected['v261_restore'][order]['counts']
    got = _counts(ed)
    assert {k: v for k, v in got.items() if k.startswith('A|')} == {
        k: v for k, v in want.items() if k.startswith('A|')}
    assert got['B|CD901b+'] == want['B|CD901b+']          # not inflated to 400
    a, b = ed._samples['A'], ed._samples['B']
    # A's asinh column: display-transformed by provenance, scale unknown,
    # kept exactly as stored, never conformed; its origin is the old
    # session's data folder.
    assert a.data_transforms['FL1-A']['method'] == 'unknown'
    assert a.data_transforms['FL1-A']['origin'].startswith('session:')
    for ch, med in expected['stored_medians']['A'].items():
        assert float(np.median(a.data[ch])) == pytest.approx(med, rel=1e-12), ch
    assert a.data_transforms['FL2-A']['method'] == 'logicle'
    assert 'U:X' not in a.data_transforms               # 2.6.1 wrote it linear
    # The editor is seeded from the default (logicle), as 2.6.1 seeded it,
    # never from an unknown column; B stays on the loader's logicle.
    assert ed._channel_transform['FL1-A'] == 'logicle'
    assert b.data_transforms['FL1-A']['method'] == 'logicle'


def test_a_v261_events_csv_loads_as_stored(v261, shown):
    """'Load in editor' of a 2.6.1 _events.csv whose FL1-A was on asinh:
    values as stored, scale unknown, beside a sample already on logicle --
    and one warning says so, still there after the event loop ran."""
    from openflo.pipeline import FlowSample
    root, proj, expected = v261
    ed = _new_editor(root)
    b = FlowSample(str(proj / 'fcs' / 'B.fcs'))
    b.run_qc()
    b.auto_compensate()
    b.apply_transform()
    ed._on_loaded('B', b)
    name = ed._import_processed_csv(str(proj / expected['events_csv']))
    _settle(root)
    s = ed._samples[name]
    assert s.data_transforms['FL1-A']['method'] == 'unknown'
    assert s.data_transforms['FL1-A']['origin'].startswith('file:')
    assert float(np.median(s.data['FL1-A'])) == pytest.approx(
        expected['stored_medians']['events']['FL1-A'], rel=1e-12)
    assert 'U:X' not in s.data_transforms
    msg = the_dialog(shown)
    assert 'T_A_events.csv' in msg and 'UNKNOWN' in msg and 'FL1-A' in msg


@pytest.mark.parametrize('raw_first', [True, False])
def test_a_v261_restore_shows_one_dialog_after_the_batch(v261, shown,
                                                         raw_first):
    """A session restore's notes used to reach only the console. One dialog,
    after the whole batch, naming the sidecar and what was assumed."""
    root, proj, _expected = v261
    _restore(root, proj / 'session.flowsession', raw_first)   # pumps 1 s
    msg = the_dialog(shown)
    assert 'A.csv' in msg and 'older OpenFlo' in msg, msg
    assert 'UNKNOWN' in msg and 'FL1-A' in msg, msg
    assert 'taken as linear: U:X' in msg, msg


def test_unknown_scale_is_omitted_with_a_note_everywhere(v261, monkeypatch):
    """No window reports a linear number for a column whose scale is unknown:
    linear_values refuses, Statistics leaves it blank and says why, and the
    Expression and Group windows leave that sample out and say so."""
    import openflo.async_task as _at
    from openflo.pipeline import UnknownScaleError, linear_values
    root, proj, _expected = v261
    ed = _restore(root, proj / 'session.flowsession', raw_first=True)
    a = ed._samples['A']
    with pytest.raises(UnknownScaleError):
        linear_values(a, 'FL1-A')
    rows, _cols = ed._collect_stats_rows({'Median'})
    ra = next(r for r in rows if r['Sample'] == 'A')
    assert np.isnan(ra['Median CD901b'])
    assert ra['__scale_unknown__'] == ['CD901b']
    assert np.isfinite(ra['Median CD902'])

    def _inline(widget, work, on_done=None, on_error=None, on_finally=None):
        on_done(work())
    monkeypatch.setattr(_at, 'run_async', _inline)
    from openflo.ui_statistics import StatisticsWindow
    w = StatisticsWindow(ed)
    w.withdraw()
    _settle(root, 0.3)
    msg = w.status_var.get()
    assert 'scale unknown' in msg and 'A: CD901b' in msg, msg

    from openflo.ui_group_stats import GroupStatsWindow
    g = GroupStatsWindow(ed)
    g.withdraw()
    g._ch.set('FL1-A')
    g._run()
    txt = g._txt.get('1.0', 'end')
    assert 'scale unknown' in txt and 'A' in txt.split('scale unknown', 1)[1]

    from openflo.ui_expression import MarkerExpressionWindow
    ed._sample_plot_enabled.update({'A': True, 'B': True})
    ed._selected_samples = lambda: ['A', 'B']
    x = MarkerExpressionWindow(ed)
    x.withdraw()
    x.marker_var.set(ed._fmt_channel('FL1-A'))
    x._rebuild()
    assert sum(len(v) for v in x._medians.values()) == 1      # B only
    assert 'scale unknown' in x._summary.get('1.0', 'end')


def test_resaving_a_v261_session_keeps_the_unknown_scale(v261, tmp_path):
    """Saved again by this build, the unknown column is recorded as unknown
    (not as logicle or linear) with the ORIGIN it had -- the old session's
    data folder, not the new one -- so the next restore is the same."""
    root, proj, expected = v261
    ed = _restore(root, proj / 'session.flowsession', raw_first=False)
    first = ed._samples['A'].data_transforms['FL1-A']
    again = proj / 'resaved.flowsession'
    ed._write_session(str(again))
    for raw_first in (True, False):
        ed2 = _restore(root, again, raw_first)
        assert ed2._samples['A'].data_transforms['FL1-A'] == first
        assert _counts(ed2) == _counts(ed)


def test_the_transform_editor_leaves_an_unknown_column_as_stored(v261):
    from openflo.pipeline import transform_values
    root, proj, expected = v261
    ed = _restore(root, proj / 'session.flowsession', raw_first=True)
    a_before = ed._samples['A'].data['FL1-A'].to_numpy(float).copy()
    b_before = ed._samples['B'].data['FL1-A'].to_numpy(float).copy()
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    np.testing.assert_array_equal(
        ed._samples['A'].data['FL1-A'].to_numpy(float), a_before)
    assert ed._samples['A'].data_transforms['FL1-A']['method'] == 'unknown'
    from openflo.pipeline import inverse_transform_values
    np.testing.assert_allclose(
        ed._samples['B'].data['FL1-A'].to_numpy(float),
        transform_values(inverse_transform_values(b_before, method='logicle'),
                         method='asinh'), rtol=1e-6, atol=1e-6)
    assert 'scale unknown' in ed.status_var.get()
