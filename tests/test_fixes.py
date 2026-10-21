"""Fixes that reach saved sessions (fixes.py): the registry's scan by step
and version range, applied once and never forced, declined fixes not asked
again, shift fixes only with the fixes they build on, default changes
flagged only, originals kept through a save and reopen, exports naming the
applied fixes, hand-edited gates losing their stamp, follow-up re-runs of
what was built on a corrected result, and ``openflo-run check-sessions``.

All data is synthetic: samples s1, s2, ...; channels FL1-A, FL2-A, ...
"""
from __future__ import annotations

import copy
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from openflo import fixes
from openflo import result_stamp as rs
from openflo.fixes import Fix
from openflo.session_format import SESSION_FORMAT, SESSION_VERSION
from tests.test_result_stamps import _add, _editor, _restore


@pytest.fixture
def running(monkeypatch):
    """Pin the running version; returns a setter for a later 'upgrade'."""
    state = {'v': '9.9.9'}
    monkeypatch.setattr(rs, 'app_version', lambda: state['v'])

    def set_to(v):
        state['v'] = v
    return set_to


def _stamp(step, version, inputs=()):
    return {'step': step, 'version': version, 'inputs': list(inputs)}


def _bump(model, target):
    """A recompute fix for the tests: the threshold moves up by 1."""
    g = copy.deepcopy(target['current'])
    g['value'] = float(g['value']) + 1.0
    return {'new': g, 'why': None}


def _double(model, target):
    """A shift fix for the tests: the threshold doubles."""
    g = copy.deepcopy(target['current'])
    g['value'] = float(g['value']) * 2.0
    return {'new': g, 'why': None}


A = Fix('a', 'Fix A', 'gate.refit', '2.0', redo=_bump)
B = Fix('b', 'Fix B', 'gate.auto.', '2.6', redo=_bump)
C = Fix('c', 'Fix C', 'gate.refit', '2.7', redo=_bump)
S = Fix('s', 'Shift S', 'gate.refit', '2.7', mode='shift', requires=('a',),
        redo=_double)
D = Fix('d', 'Default D', 'cell_cycle', '2.5', kind='default-change')


class Model:
    """The parts of an editor the fixes use: gates, samples and _get_df."""

    def __init__(self, frames, gates):
        from types import SimpleNamespace
        self._samples = {n: SimpleNamespace(name=n, data=df)
                         for n, df in frames.items()}
        self._sample_order = list(frames)
        self._sample_gates = gates
        self._sample_gate_order = {n: list(g) for n, g in gates.items()}

    def _get_df(self, name, x, y=None, downsample=False, gate_parent=None):
        from openflo.pipeline import gate_to_mask
        df = self._samples[name].data
        gid = gate_parent
        while gid is not None:
            g = self._sample_gates[name][gid]
            df = df[np.asarray(gate_to_mask(g, df), dtype=bool)]
            gid = g.get('parent_id')
        return df


def _model(*stamps, value=1.0):
    """One sample, one threshold gate per stamp (g1, g2, ...)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'FL1-A': rng.normal(2, 1, 500),
                       'FL2-A': rng.normal(0, 1, 500)})
    gates = {}
    for i, st in enumerate(stamps, 1):
        g = {'kind': 'threshold', 'channel': 'FL1-A', 'value': value,
             'parent_id': None, 'name': f'G{i}',
             'stamp': copy.deepcopy(st)}
        gates[f'g{i}'] = rs.seal_gate_stamp(g)
    return Model({'s1': df}, {'s1': gates})


def _ids(targets):
    return [(t['key'], t['fix']) for t in targets]


# ── Scan: step and version range ────────────────────────────────────────────

def test_scan_matches_the_step_and_every_fix_after_the_result(running):
    m = _model(_stamp('gate.refit', '1.1'), _stamp('gate.auto.gmm', '1.1'),
               _stamp('gate.refit', '2.7'), _stamp('cell_cycle', '1.1'))
    got = fixes.scan(m, [C, A, B])
    # Oldest fixed_in first on one result; a prefix step matches.
    assert _ids(got) == [('g1', 'a'), ('g1', 'c'), ('g2', 'b')]
    assert got[0]['version'] == '1.1' and not got[0]['unknown']


def test_skipping_versions_matches_by_range_and_never_offers_twice(running):
    """1.1 -> 2.6 -> 2.7: on the first jump the fixes up to 2.6 apply, the
    stamp moves to 2.6, and the jump to 2.7 offers only C."""
    m = _model(_stamp('gate.refit', '1.1'), _stamp('gate.auto.gmm', '1.1'))
    reg = [A, B, C]
    running('2.6')
    first = fixes.scan(m, reg)
    assert _ids(first) == [('g1', 'a'), ('g2', 'b')]     # C not in 2.6
    done = fixes.apply_rows(m, fixes.plan(m, first, reg))
    assert len(done) == 2
    assert m._sample_gates['s1']['g1']['stamp']['version'] == '2.6'
    running('2.7')
    assert _ids(fixes.scan(m, reg)) == [('g1', 'c')]


def test_a_fix_this_build_does_not_have_is_not_offered(running):
    running('1.9')
    m = _model(_stamp('gate.refit', '1.1'))
    assert fixes.scan(m, [A]) == []
    running('2.0')
    assert _ids(fixes.scan(m, [A])) == [('g1', 'a')]


def test_an_unknown_version_is_flagged_for_every_fix_of_its_step(running):
    m = _model(_stamp('gate.refit', None))
    got = fixes.scan(m, [A, C])
    assert _ids(got) == [('g1', 'a'), ('g1', 'c')]
    assert all(t['unknown'] for t in got)
    assert 'unknown version' in fixes.describe_target(got[0])


# ── Applied once ────────────────────────────────────────────────────────────

def test_applied_once_the_second_scan_finds_nothing(running):
    m = _model(_stamp('gate.refit', '1.1', ['FL1-A']))
    rows = fixes.plan(m, fixes.scan(m, [A]), [A])
    assert rows[0]['changed'] and rows[0]['after'] != rows[0]['before']
    fixes.apply_rows(m, rows)
    fixes.record(m, applied=['a'])
    g = m._sample_gates['s1']['g1']
    assert g['value'] == 2.0
    assert rs.stamp_of(g)['version'] == '9.9.9'          # moved to now
    assert rs.stamp_of(g)['inputs'] == ['FL1-A']          # same step, inputs
    assert fixes.scan(m, [A]) == []
    # Even with the session's record gone, the moved stamp no longer matches.
    m._applied_fixes = []
    assert fixes.scan(m, [A]) == []


def test_the_refit_redo_recomputes_so_running_it_again_changes_nothing(
        running):
    from tests.test_gate_refit import _bimodal
    v = _bimodal(3, 0.7, sd=0.4)
    m = Model({'s1': pd.DataFrame({'FL2-A': v})},
              {'s1': {'g1': rs.seal_gate_stamp(
                  {'kind': 'threshold', 'channel': 'FL2-A', 'value': 1.5,
                   'parent_id': None,
                   'stamp': _stamp('gate.refit', '2.6.0')})}})
    (fix,) = [f for f in fixes.REGISTRY if f.id == 'refit-valley']
    assert fix.fixed_in == '2.7.0b2' and fix.mode == 'recompute'
    rows = fixes.plan(m, fixes.scan(m), None)
    assert [r['fixes'] for r in rows] == [['refit-valley']]
    fixes.apply_rows(m, rows)
    once = m._sample_gates['s1']['g1']['value']
    assert once == pytest.approx(0.7 + 1.5, abs=0.15)
    again = fix.redo(m, {'sample': 's1', 'current': copy.deepcopy(
        m._sample_gates['s1']['g1'])})
    assert again['why'] is None and again['new']['value'] == once


def test_a_declined_fix_is_not_asked_again_but_can_be_reviewed(running):
    m = _model(_stamp('gate.refit', '1.1'))
    fixes.record(m, declined=['a'])
    assert fixes.scan(m, [A, C]) == [dict(fixes.scan(m, [C])[0])]
    allx = fixes.scan(m, [A, C], include_settled=True)
    assert [(t['fix'], t['status']) for t in allx] == [('a', 'declined'),
                                                       ('c', 'open')]
    fixes.record(m, applied=['a'])                    # applied later after all
    assert m._declined_fixes == [] and m._applied_fixes == ['a']


# ── Shift fixes and default changes ─────────────────────────────────────────

def test_a_shift_fix_comes_only_with_the_fixes_it_requires(running):
    old = _model(_stamp('gate.refit', '1.1'))
    assert _ids(fixes.scan(old, [S, A])) == [('g1', 'a'), ('g1', 's')]
    # Applied in order, the shift on the recomputed result: (1 + 1) * 2.
    (row,) = fixes.plan(old, fixes.scan(old, [S, A]), [S, A])
    assert row['fixes'] == ['a', 's'] and row['new']['value'] == 4.0
    # A declined: S would stand on nothing, so it is not offered.
    fixes.record(old, declined=['a'])
    assert fixes.scan(old, [S, A]) == []
    # A result made after A was fixed gets S alone.
    after_a = _model(_stamp('gate.refit', '2.1'))
    assert _ids(fixes.scan(after_a, [S, A])) == [('g1', 's')]
    # A requirement the registry does not know never holds.
    lost = Fix('s2', 'Shift', 'gate.refit', '2.7', mode='shift',
               requires=('gone',), redo=_double)
    assert fixes.scan(after_a, [lost]) == []


def test_a_default_change_is_flagged_only(running):
    m = _model()
    m._samples['s1'].data['cell_cycle'] = 1
    m._samples['s1'].result_stamps = {'cell_cycle': _stamp('cell_cycle',
                                                           '2.0')}
    (t,) = fixes.scan(m, [D])
    assert t['informational'] and t['result'] == 'column'
    (row,) = fixes.plan(m, [t], [D])
    assert row['informational'] and row['new'] is None
    assert fixes.apply_rows(m, [row]) == []
    assert '[default change, flagged only]' in fixes.describe_target(t)


# ── Hand edits ──────────────────────────────────────────────────────────────

def test_a_gate_moved_by_hand_loses_its_stamp(running):
    m = _model(_stamp('gate.refit', '1.1'), _stamp('gate.refit', '1.1'))
    gates = m._sample_gates['s1']
    gates['g1']['value'] = 1.3                          # dragged
    gates['g2']['name'] = 'Renamed'                     # not a geometry edit
    assert rs.live_gate_stamp(gates['g1']) is None
    assert _ids(fixes.scan(m, [A])) == [('g2', 'a')]
    assert rs.settle_gate_stamps(gates) == ['g1']
    assert 'stamp' not in gates['g1'] and 'stamp' in gates['g2']
    # A stamp read without a digest is sealed on the geometry it has then.
    g = {'kind': 'threshold', 'channel': 'FL1-A', 'value': 2.0,
         'stamp': _stamp('gate.auto.threshold', '1.1')}
    assert rs.settle_gate_stamps({'g': g}) == []
    assert rs.GEOM_KEY in g['stamp']
    # Saved and reopened, an open bound is the 1e12 sentinel: still live.
    open_g = rs.stamp_gate({'kind': 'interval', 'channel': 'FL1-A',
                            'lo': 0.5, 'hi': float('inf')}, 'gate.refit')
    from openflo.editor_session import _json_safe
    assert rs.live_gate_stamp(json.loads(json.dumps(_json_safe(open_g))))


# ── Follow-ups ──────────────────────────────────────────────────────────────

def test_results_built_on_a_corrected_one_are_follow_ups(running):
    m = _model(_stamp('gate.refit', '1.1'), _stamp('gate.refit', '1.1'),
               _stamp('gate.refit', '1.1'))
    gates = m._sample_gates['s1']
    gates['g2']['parent_id'] = 'g1'
    gates['g3']['parent_id'] = 'g2'
    for g in gates.values():
        g['stamp'].pop(rs.GEOM_KEY)
        rs.seal_gate_stamp(g)
    m._samples['s1'].data['UMAP1'] = 0.0
    m._samples['s1'].result_stamps = {
        'UMAP1': _stamp('embed.umap', '9.9.9', ['g2', 'FL1-A'])}
    fol = fixes.downstream(m, [('s1', 'gate', 'g1')])
    assert [(f['key'], f['because'], f['rerun'], f['slow']) for f in fol] == [
        ('g2', 'g1', True, False), ('g3', 'g2', True, False),
        ('UMAP1', 'g2', False, True)]
    assert fol[2]['where'].startswith('Analyze > Cluster')


# ── Sessions on disk ────────────────────────────────────────────────────────

def _session_file(path, stamps, applied=()):
    gates = [rs.seal_gate_stamp({'id': f'g{i}', 'kind': 'threshold',
                                 'channel': 'FL2-A', 'value': 1.5,
                                 'parent_id': None, 'name': f'Pos{i}',
                                 'stamp': st})
             for i, st in enumerate(stamps, 1)]
    data = {'format': SESSION_FORMAT, 'version': SESSION_VERSION,
            'samples': [{'name': 's1', 'path': 's1.fcs'}],
            'sample_gates': {'s1': gates}, 'fixes_applied': list(applied)}
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def test_check_sessions_lists_the_fixes_and_changes_nothing(tmp_path,
                                                            running):
    from openflo.cli import run_check_sessions
    sub = tmp_path / 'day2'
    sub.mkdir()
    old = _session_file(tmp_path / 'old.flowsession',
                        [_stamp('gate.refit', '2.6.0'),
                         _stamp('gate.auto.gmm', '2.6.0')])
    _session_file(sub / 'done.flowsession', [_stamp('gate.refit', '2.6.0')],
                  applied=['refit-valley'])
    _session_file(sub / 'new.flowsession', [_stamp('gate.refit', '9.9.9')])
    _session_file(tmp_path / 'old.backup.flowsession',
                  [_stamp('gate.refit', '2.6.0')])
    (tmp_path / 'bad.flowsession').write_text('{', encoding='utf-8')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.flowsession')}
    out = io.StringIO()
    assert run_check_sessions(str(tmp_path), out) == 0
    text = out.getvalue().splitlines()
    assert text[0].endswith('4 session(s) under ' + tmp_path.name)
    assert text[1].startswith("bad.flowsession: can't read")
    assert text[2:5] == [
        os.path.join('day2', 'done.flowsession') + ': nothing to fix',
        os.path.join('day2', 'new.flowsession') + ': nothing to fix',
        'old.flowsession: 1 result(s), fix(es) refit-valley']
    assert text[5] == ("  s1: gate 'Pos1' (g1) made by OpenFlo 2.6.0 -- "
                       "refit-valley: Re-fit per sample missed the valley "
                       "before a bright population")
    assert text[-1] == '1 result(s) to review; nothing was changed.'
    assert {p: p.read_bytes() for p in tmp_path.rglob('*.flowsession')} == \
        before
    assert old.read_bytes() == before[old]
    assert run_check_sessions(str(tmp_path / 'nope'), io.StringIO()) == 2


def test_check_sessions_is_an_openflo_run_command(tmp_path, running, capsys):
    from openflo.cli_args import main
    _session_file(tmp_path / 'a.flowsession', [_stamp('gate.refit', None)])
    assert main(['check-sessions', str(tmp_path)]) == 0
    assert 'a.flowsession: 1 result(s), fix(es) refit-valley' in \
        capsys.readouterr().out


def test_a_v2_session_has_no_fixes_recorded():
    from openflo.session_format import migrate_session
    data, _ = migrate_session({'format': SESSION_FORMAT, 'version': 2,
                               'samples': [{'name': 's1'}]})
    assert data['fixes_applied'] == [] and data['fixes_declined'] == []
    assert data['samples'][0]['result_originals'] == {}


# ── In the editor ────────────────────────────────────────────────────────────

@pytest.fixture
def tk_root(tmp_path, monkeypatch):
    from tests.test_transform_records import capture_warnings, isolate_home
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        from tests.conftest import gui_unavailable
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    yield root
    root.destroy()


def _two_pop(seed=3, m=0.7, n=4000):
    from tests.test_gate_refit import _bimodal
    rng = np.random.default_rng(seed)
    v = _bimodal(seed, m, n=n, sd=0.4)
    return pd.DataFrame({'FL1-A': rng.uniform(1, 2, v.size), 'FL2-A': v,
                         'leiden': rng.integers(0, 3, v.size)})


def _stamped_editor(root, tmp_path, version='2.6.0'):
    """An editor with one sample: All (FL1-A) > Pos (FL2-A, re-fitted by
    ``version``) > Child (FL1-A, re-fitted by ``version``)."""
    ed = _editor(root)
    _add(ed, _two_pop(), path=str(tmp_path / 's1.fcs'))
    p = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.0,
                      'name': 'All', 'parent_id': None})
    pos = ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A',
                        'value': 1.5, 'name': 'Pos', 'parent_id': p,
                        'stamp': _stamp('gate.refit', version,
                                        [p, 'FL2-A'])})
    child = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                          'value': 1.5, 'name': 'Child', 'parent_id': pos,
                          'stamp': _stamp('gate.refit', version,
                                          [pos, 'FL1-A'])})
    ed._clear_undo_pending()
    ed._refresh_gate_list()
    return ed, pos, child


def _review_windows(ed):
    from openflo.ui_fixes import FixReviewWindow
    return [w for w in ed.winfo_children() if isinstance(w, FixReviewWindow)]


def test_editor_apply_is_one_undo_step_and_keeps_the_original(
        tk_root, tmp_path, running):
    ed, pos, child = _stamped_editor(tk_root, tmp_path)
    before = copy.deepcopy(ed._sample_gates['s1'][pos])
    win = ed._check_fixes_on_open()
    assert win is not None and [r['key'] for r in win.rows] == [pos, child]
    row = win.rows[0]
    n0, pct0 = row['before']
    n1, pct1 = row['after']
    assert n0 != n1 and 0 < pct1 < 100
    assert win.tv.set('0', 'before') == f"{n0:,} ({pct0:.1f}%)"
    assert win.tv.set('0', 'use') == '☑'
    # Child is on a channel with no valley: kept as it was, unticked.
    assert win.tv.set('1', 'use') == '☐'
    assert win.tv.set('1', 'note').startswith('kept: ')
    win.apply()
    g = ed._sample_gates['s1'][pos]
    assert g['value'] == pytest.approx(0.7 + 1.5, abs=0.15)
    assert g['original']['value'] == 1.5
    assert g['original']['stamp'] == _stamp('gate.refit', '2.6.0',
                                            [before['parent_id'], 'FL2-A'])
    assert rs.stamp_of(g)['version'] == '9.9.9'
    assert ed._applied_fixes == ['refit-valley']
    assert ed._audit_log.to_list()[-1]['action'] == 'fix.apply'
    # The window now offers the gate under it as a follow-up re-run.
    assert win.winfo_exists() and win.mode == 'follow'
    assert [f['key'] for f in win.follows] == [child]
    assert win.tv.set('0', 'use') == '☑'
    # Stats and plots ignore the kept original.
    from openflo.plotcache import chain_sigs
    sigs = chain_sigs(ed._sample_gates['s1'])
    g['original']['value'] = 99.0
    assert chain_sigs(ed._sample_gates['s1']) == sigs
    ed._clear_undo_pending()                 # the re-run is another click
    win.rerun()
    assert ed._sample_gates['s1'][child]['stamp']['version'] == '9.9.9'
    ed._undo()                                       # the re-run
    ed._undo()                                       # the fixes
    assert ed._sample_gates['s1'][pos]['value'] == 1.5
    assert 'original' not in ed._sample_gates['s1'][pos]
    assert ed._applied_fixes == []
    assert _ids(fixes.scan(ed)) == [(pos, 'refit-valley'),
                                    (child, 'refit-valley')]


def test_editor_nothing_changes_unless_applied_and_declined_stays_declined(
        tk_root, tmp_path, running):
    ed, pos, child = _stamped_editor(tk_root, tmp_path)
    before = copy.deepcopy(ed._sample_gates)
    win = ed._check_fixes_on_open()
    win.destroy()                                     # "Not now"
    assert ed._sample_gates == before
    assert fixes.scan(ed)                             # asked again next time
    win = ed._check_fixes_on_open()
    for iid, on in list(win.ticked.items()):
        if on:
            win.toggle(iid)
    win.apply()                                       # nothing ticked
    assert ed._sample_gates == before
    assert ed._declined_fixes == ['refit-valley']
    assert ed._check_fixes_on_open() is None
    menu = ed._review_fixes()                         # Tools menu: listed
    assert [r['status'] for r in menu.rows] == ['declined', 'declined']
    assert menu.tv.set('0', 'note') == 'declined earlier'
    assert menu.tv.set('0', 'use') == '☐'
    menu.destroy()


def test_session_round_trip_keeps_originals_and_applied_fixes(
        tk_root, tmp_path, running):
    from openflo import export_stamp
    ed, pos, _child = _stamped_editor(tk_root, tmp_path)
    s = ed._samples['s1']
    s.result_stamps = {'leiden': _stamp('cluster.leiden', '2.6.0',
                                        ['FL1-A'])}
    col_fix = Fix('relabel', 'Relabel', 'cluster.leiden', '2.7',
                  redo=lambda m, t: {'new': np.asarray(t['current']) + 10,
                                     'why': None})
    reg = list(fixes.REGISTRY) + [col_fix]
    rows = fixes.plan(ed, [t for t in fixes.scan(ed, reg)
                           if t['key'] in (pos, 'leiden')], reg)
    assert [r['key'] for r in rows] == [pos, 'leiden']
    assert '0: ' in rows[1]['before']                # a label summary
    ed._apply_fix_rows(rows, rows)
    kept = rs.original_column('leiden', '2.6.0')
    assert kept == 'leiden [original 2.6.0]'
    assert set(s.data['leiden']) == {10, 11, 12}
    assert set(s.data[kept]) == {0, 1, 2}
    # A kept original is no channel: not on the axes, not in the stats.
    ed._refresh_channel_choices()
    assert kept not in ed._channels
    assert s._derived_column(kept)
    # Exports name the applied fixes.
    assert export_stamp.corrections_for(ed) == ['refit-valley', 'relabel']
    assert 'Corrections' in export_stamp.savefig_metadata('x.png', None,
                                                          corrections=
                                                          export_stamp.
                                                          corrections_for(ed))

    path = tmp_path / 'one.flowsession'
    data = ed._write_session(str(path))
    assert data['fixes_applied'] == ['refit-valley', 'relabel']
    assert data['fixes_declined'] == []
    assert data['samples'][0]['result_originals'] == {'leiden': {
        'stamp': _stamp('cluster.leiden', '2.6.0', ['FL1-A']),
        'column': kept}}
    ed2 = _restore(tk_root, path)
    s2 = ed2._samples['s1']
    assert ed2._applied_fixes == ['refit-valley', 'relabel']
    assert s2.result_originals == data['samples'][0]['result_originals']
    assert np.array_equal(s2.data[kept].to_numpy(), s.data[kept].to_numpy())
    g2 = next(g for g in ed2._sample_gates['s1'].values()
              if g.get('name') == 'Pos')
    assert g2['original']['value'] == 1.5
    assert rs.live_gate_stamp(g2)['version'] == '9.9.9'
    assert export_stamp.corrections_for(ed2) == ['refit-valley', 'relabel']
    # Reopened, nothing is offered again (the child was left as it was, but
    # its fix is recorded as applied to this session).
    assert fixes.scan(ed2, reg) == []
    again = ed2._write_session(str(tmp_path / 'two.flowsession'))
    assert again['samples'][0]['result_originals'] == \
        data['samples'][0]['result_originals']


def test_reopening_a_session_offers_its_fixes(tk_root, tmp_path, running):
    ed, pos, child = _stamped_editor(tk_root, tmp_path)
    path = tmp_path / 'old.flowsession'
    ed._write_session(str(path))
    ed2 = _restore(tk_root, path)
    if ed2._session_restore:
        ed2._apply_session_restore()
    (win,) = _review_windows(ed2)
    assert [r['name'] for r in win.rows] == ['Pos', 'Child']
    win.destroy()


def test_hand_edits_take_the_stamp_off_on_every_edit_path(tk_root, tmp_path,
                                                         running):
    ed, pos, child = _stamped_editor(tk_root, tmp_path)
    gates = ed._sample_gates['s1']
    # A drag (motion edits the gate in place; the release refreshes).
    gates[pos]['value'] = 1.6
    ed._on_release(type('E', (), {'inaxes': None})())
    assert 'stamp' not in gates[pos] and 'stamp' in gates[child]
    # Typed coordinates.
    ed._apply_gate_coordinates('s1', child, {'value': '1.4'})
    assert 'stamp' not in gates[child]
    # A paste carries no stamp, and neither does a copy to another sample.
    ed._sample_gates['s1'][pos]['stamp'] = _stamp('gate.refit', '2.6.0')
    ed._refresh_gate_list()
    ed._clip_kind = 'gate_tree'
    ed._clip_payload = ed._collect_gate_subtree('s1', pos)
    ed._paste_gate_tree()
    pasted = [g for gid, g in gates.items() if gid not in (pos, child)
              and g.get('name') == 'Pos']
    assert pasted and all('stamp' not in g for g in pasted)
    assert 'stamp' in gates[pos]
    _add(ed, _two_pop(seed=4), name='s2', path=str(tmp_path / 's2.fcs'))
    ed._set_active_sample('s1')
    ed._copy_gates_to(['s2'])
    copied = [g for g in ed._sample_gates['s2'].values()
              if g.get('name') == 'Pos']
    assert copied and all('stamp' not in g for g in copied)
    # A save never writes a stale stamp.
    gates[pos]['value'] = 1.7
    data = ed._session_state()
    saved = next(g for g in data['sample_gates']['s1'] if g['id'] == pos)
    assert 'stamp' not in saved


# ── After Apply: exports made before the fix, and Open… ─────────────────────

def test_stale_exports_are_files_made_before_the_fix_from_a_corrected_sample(
        tmp_path):
    from openflo.audit import AuditLog
    there = tmp_path / 'report.html'
    there.write_text('x', encoding='utf-8')
    gone = str(tmp_path / 'gone' / 'fig.png')
    page = [str(tmp_path / f'page_s1_{i}.png') for i in (1, 2)]
    log = AuditLog(version='2.6.0')
    log.record('report.export', time='t1', path=str(there),
               samples=['s1', 's2'])
    log.record('export.gate_list', time='t2', path='only_s2.csv',
               samples=['s2'])                        # no corrected sample
    log.record('figure.export', time='t3', path=gone)  # names no sample
    log.record('layout.export', time='t4', path='layout.png',
               samples=['s1'], files=page)
    log.record('spectral.qc.export', time='t5', path='qc.md')  # not results
    log.record('population.export_fcs', time='t6', path='pop.fcs',
               sample='s3')
    log.version = '2.7.0'
    log.record('report.export', time='t7', path=str(there),
               samples=['s1'])                        # the same file again
    log.record('fix.apply', fixes=['refit-valley'],
               results=[{'sample': 's1', 'result': 'gate', 'key': 'g1'}])
    log.record('report.export', time='t9', path='after.html',
               samples=['s1'])                        # made after the fix
    rows = fixes.stale_exports(log.entries())
    assert [(r['path'], r['time'], r['version']) for r in rows] == [
        (str(there), 't7', '2.7.0'), (page[0], 't4', '2.6.0'),
        (page[1], 't4', '2.6.0'), (gone, 't3', '2.6.0')]
    assert [r['exists'] for r in rows] == [True, False, False, False]
    assert rows[3]['samples'] is None
    # An Apply that corrected nothing lists nothing.
    log.record('fix.apply', fixes=[], declined=['refit-valley'], results=[])
    assert fixes.stale_exports(log.entries()) == []


def test_exports_record_their_file_and_samples(tk_root, tmp_path, running,
                                               monkeypatch):
    import webbrowser

    from openflo import editor_export
    ed, _pos, _child = _stamped_editor(tk_root, tmp_path)
    _add(ed, _two_pop(seed=5), name='s2', path=str(tmp_path / 's2.fcs'))
    out = iter([str(tmp_path / 'gates.csv'), str(tmp_path / 'report.html')])
    monkeypatch.setattr(editor_export.filedialog, 'asksaveasfilename',
                        lambda **kw: next(out))
    monkeypatch.setattr(webbrowser, 'open', lambda *a, **k: True)
    ed._export_gate_list()
    ed._export_report()
    got = [(e['action'], e['details'].get('path'), e['details'].get('samples'))
           for e in ed._audit_log.entries()
           if e['action'] in fixes.EXPORT_ACTIONS]
    assert got == [('export.gate_list', str(tmp_path / 'gates.csv'), ['s1']),
                   ('report.export', str(tmp_path / 'report.html'),
                    ['s1', 's2'])]


def test_after_apply_the_window_lists_earlier_exports_and_shows_their_folder(
        tk_root, tmp_path, running, monkeypatch):
    from openflo import ui_fixes
    ed, _pos, _child = _stamped_editor(tk_root, tmp_path)
    kept = tmp_path / 'stats.html'
    kept.write_text('x', encoding='utf-8')
    gone = tmp_path / 'deleted.png'
    ed._audit('report.export', path=str(kept), samples=['s1'])
    ed._audit('figure.export', path=str(gone), samples=['s1'])
    ed._audit('export.gate_list', path=str(tmp_path / 'other.csv'),
              samples=['s9'])
    opened = []
    monkeypatch.setattr(ui_fixes, 'open_folder',
                        lambda p: opened.append(p) or True)
    win = ed._check_fixes_on_open()
    win.apply()
    ed._audit('report.export', path=str(tmp_path / 'later.html'),
              samples=['s1'])
    assert win.mode == 'follow' and win.export_tv is not None
    tv = win.export_tv
    assert [tv.set(i, 'file') for i in tv.get_children()] == [
        str(gone), str(kept)]
    assert tv.set('0', 'state') == 'missing'
    assert tv.set('1', 'state') == 'there'
    assert tv.set('1', 'by') == '9.9.9'
    assert tv.set('0', 'show') == ui_fixes.SHOW
    assert win.show_export('1') and opened == [str(kept)]
    assert win.show_export('0') and opened[-1] == str(tmp_path)
    # Listed only: nothing was written, the missing file is still missing.
    assert not gone.exists() and kept.read_text(encoding='utf-8') == 'x'
    assert [r['path'] for r in ed._stale_exports()] == [str(gone), str(kept)]
    win.destroy()


def test_open_opens_the_dialog_for_a_follow_up_and_starts_nothing(
        tk_root, tmp_path, running, monkeypatch):
    ed, pos, child = _stamped_editor(tk_root, tmp_path)
    s = ed._samples['s1']
    s.result_stamps = {'leiden': _stamp('cluster.leiden', '9.9.9',
                                        [pos, 'FL1-A'])}
    win = ed._check_fixes_on_open()
    win.apply()
    keys = [f['key'] for f in win.follows]
    assert keys == [child, 'leiden']
    assert win.follows[1]['opens'] == '_open_cluster_dialog'
    assert win.tv.set('1', 'use') == 'Open…'
    assert win.tv.set('0', 'use') == '☑'               # a re-fit: ticked
    # Another sample is active; Open… selects the one the result is in.
    _add(ed, _two_pop(seed=5), name='s2', path=str(tmp_path / 's2.fcs'))
    ed._refresh_gate_list()
    assert ed._active_sample == 's2'
    calls = []
    for m in ('_open_cluster_dialog', '_open_cell_cycle_dialog',
              '_open_calibration_dialog', '_open_trajectory_window',
              '_auto_gate', '_rerun_follow_ups', '_checkpoint'):
        monkeypatch.setattr(ed, m, lambda *a, _m=m: calls.append(
            (_m, ed._active_sample)))
    monkeypatch.setattr(fixes, 'apply_rows',
                        lambda *a: calls.append(('apply_rows', None)) or [])
    gates = copy.deepcopy(ed._sample_gates)
    labels = s.data['leiden'].to_numpy(copy=True)
    assert win.open_follow_up('1')
    assert calls == [('_open_cluster_dialog', 's1')]
    assert ed._sample_gates == gates
    assert np.array_equal(s.data['leiden'].to_numpy(), labels)
    assert win.winfo_exists() and win.mode == 'follow'
    # A re-fit row has no Open…: it re-runs from the window when ticked.
    assert not win.open_follow_up('0')
    assert calls == [('_open_cluster_dialog', 's1')]
    win.destroy()


def test_a_follow_up_opens_the_dialog_for_its_step(running):
    m = _model(_stamp('gate.refit', '1.1'), _stamp('gate.auto.gmm', '1.1'))
    m._sample_gates['s1']['g2']['parent_id'] = 'g1'
    for g in m._sample_gates['s1'].values():
        g['stamp'].pop(rs.GEOM_KEY)
        rs.seal_gate_stamp(g)
    for col, step in (('cc', 'cell_cycle'), ('cal', 'calibration'),
                      ('pt', 'trajectory'), ('U1', 'embed.umap')):
        m._samples['s1'].data[col] = 0.0
        m._samples['s1'].result_stamps = dict(
            getattr(m._samples['s1'], 'result_stamps', {}),
            **{col: _stamp(step, '9.9.9', ['g1'])})
    fol = {f['key']: f['opens']
           for f in fixes.downstream(m, [('s1', 'gate', 'g1')])}
    assert fol == {'g2': '_auto_gate', 'cc': '_open_cell_cycle_dialog',
                   'cal': '_open_calibration_dialog',
                   'pt': '_open_trajectory_window',
                   'U1': '_open_cluster_dialog'}
    # Each is the method its menu item (or the Auto-gate button) calls.
    from openflo.gui import ViewGateEditorWindow
    assert all(callable(getattr(ViewGateEditorWindow, n, None))
               for n in fol.values())
