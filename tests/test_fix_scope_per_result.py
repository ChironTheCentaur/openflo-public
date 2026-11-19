"""A fix applied to some results is applied to those results only (scope
audit L1).

The review window lists one row per result a fix would change. Applying
the ticked rows corrects those results; each unticked row is declined for
its own result. The other results of the same fix stay open (offered on
the next open, listed as open, not "applied"), a result that joins later is
offered, and an export says a correction was applied only when the samples
it was made from carry a corrected result.

All data is synthetic: samples s1, s2, ...; channels FL1-A, FL2-A.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import tests.test_debarcode as _debarcode_tests
import tests.test_export_audit as _audit_tests
from openflo import fixes
from openflo import result_stamp as rs
from openflo.fixes import Fix
from tests.test_result_stamps import _add, _editor

tk_root = _audit_tests.tk_root
ed = _audit_tests.ed
env = _debarcode_tests.env

REFIT = 'refit-valley'


@pytest.fixture
def running(monkeypatch):
    """Pin the running version (later than every fix in the registry)."""
    monkeypatch.setattr(rs, 'app_version', lambda: '9.9.9')


def _stamp(version, inputs=()):
    return {'step': 'gate.refit', 'version': version, 'inputs': list(inputs)}


def _bump(model, target):
    """A recompute fix for the tests: the threshold moves up by 1."""
    g = copy.deepcopy(target['current'])
    g['value'] = float(g['value']) + 1.0
    return {'new': g, 'why': None}


A = Fix('a', 'Fix A', 'gate.refit', '2.0', redo=_bump)


class Model:
    """The parts of an editor the fixes use (as tests/test_fixes.Model)."""

    def __init__(self, names):
        from types import SimpleNamespace
        rng = np.random.default_rng(0)
        self._samples = {n: SimpleNamespace(name=n, data=pd.DataFrame(
            {'FL1-A': rng.normal(2, 1, 300)})) for n in names}
        self._sample_order = list(names)
        self._sample_gates = {n: {} for n in names}
        self._sample_gate_order = {n: [] for n in names}

    def add_gate(self, name, gid, version='1.1'):
        g = {'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.0,
             'parent_id': None, 'name': gid, 'stamp': _stamp(version)}
        self._sample_gates[name][gid] = rs.seal_gate_stamp(g)
        self._sample_gate_order[name].append(gid)

    def _get_df(self, name, x, y=None, downsample=False, gate_parent=None):
        return self._samples[name].data


def _model():
    m = Model(['s1', 's2'])
    m.add_gate('s1', 'g1')
    m.add_gate('s2', 'g1')
    return m


def _where(targets):
    return [(t['sample'], t['key'], t['status']) for t in targets]


# ── Library: settled per result ─────────────────────────────────────────────

def test_applying_one_result_leaves_the_other_results_of_the_fix_open(
        running):
    m = _model()
    rows = fixes.plan(m, fixes.scan(m, [A]), [A])
    s1 = [r for r in rows if r['sample'] == 's1']
    fixes.apply_rows(m, s1)
    assert fixes.record(m, applied=[fixes.decline_of(
        'a', 's1', 'gate', 'g1')]) == []
    assert m._sample_gates['s2']['g1']['value'] == 1.0     # untouched
    # s2 is still offered, and listed as open, never as "applied".
    assert _where(fixes.scan(m, [A])) == [('s2', 'g1', 'open')]
    assert _where(fixes.scan(m, [A], include_settled=True)) == [
        ('s2', 'g1', 'open')]
    assert fixes.applied_results(m, [A]) == [
        {'fix': 'a', 'sample': 's1', 'result': 'gate', 'key': 'g1'}]
    assert fixes.applied_ids(m, [A], samples=['s2']) == []
    assert fixes.applied_ids(m, [A], samples=['s1']) == ['a']


def test_a_decline_holds_for_its_result_and_a_later_result_is_offered(
        running):
    m = _model()
    fixes.record(m, declined=[fixes.decline_of('a', 's2', 'gate', 'g1')])
    assert _where(fixes.scan(m, [A])) == [('s1', 'g1', 'open')]
    assert _where(fixes.scan(m, [A], include_settled=True)) == [
        ('s1', 'g1', 'open'), ('s2', 'g1', 'declined')]
    # A result that joins later (an old gate copied in, say) is offered.
    m.add_gate('s2', 'g2')
    assert _where(fixes.scan(m, [A])) == [('s1', 'g1', 'open'),
                                          ('s2', 'g2', 'open')]
    # Declining twice records once.
    assert fixes.record(m, declined=[fixes.decline_of(
        'a', 's2', 'gate', 'g1')]) == []
    assert len(m._declined_fixes) == 1


def test_a_session_saved_with_a_fix_id_applied_still_offers_its_results(
        running):
    """Sessions saved before this listed a fix id in ``fixes_applied`` for
    results that were never corrected (the unticked rows): they are offered,
    since the result itself shows it was not corrected."""
    data = {'sample_gates': {'s1': [rs.seal_gate_stamp(
        {'id': 'g1', 'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.0,
         'parent_id': None, 'stamp': _stamp('1.1')})]},
        'fixes_applied': ['a'], 'fixes_declined': []}
    assert _where(fixes.scan(data, [A])) == [('s1', 'g1', 'open')]
    assert fixes.applied_ids(data, [A]) == []
    # A bare declined id keeps its old meaning in a session file.
    data['fixes_declined'] = ['a']
    assert fixes.scan(data, [A]) == []


def test_a_session_file_reads_applied_from_its_kept_originals(running):
    """``fixes_applied`` in a file is not trusted: a gate or column counts
    as corrected when its kept original's stamp the fix affects and its own
    does not. A default change is never "applied"."""
    D = Fix('d', 'Default D', 'gate.refit', '2.0', kind='default-change')
    col = Fix('c', 'Col C', 'cluster.leiden', '2.0', redo=_bump)
    fixed = rs.seal_gate_stamp(
        {'id': 'g1', 'kind': 'threshold', 'channel': 'FL1-A', 'value': 2.0,
         'parent_id': None, 'stamp': _stamp('9.9.9'),
         'original': {'value': 1.0, 'stamp': _stamp('1.1')}})
    data = {'fixes_applied': [],
            'sample_gates': {'s1': [fixed]},
            'samples': [{'name': 's2', 'result_stamps': {
                'leiden': {'step': 'cluster.leiden', 'version': '9.9.9',
                           'inputs': []}},
                'result_originals': {'leiden': {
                    'stamp': {'step': 'cluster.leiden', 'version': '1.0',
                              'inputs': []},
                    'column': 'leiden [original 1.0]'}}}]}
    assert fixes.applied_results(data, [A, D, col]) == [
        {'fix': 'a', 'sample': 's1', 'result': 'gate', 'key': 'g1'},
        {'fix': 'c', 'sample': 's2', 'result': 'column', 'key': 'leiden'}]
    assert fixes.applied_ids(data, [A, D, col], samples=['s2']) == ['c']


def test_legacy_declined_ids_expand_per_result_once_every_sample_is_in(
        running, monkeypatch):
    from openflo.editor_fixes import FixesMixin
    monkeypatch.setattr(fixes, 'REGISTRY', [A])
    m = Model(['s1'])
    m.add_gate('s1', 'g1')
    m._declined_fixes = ['a']
    m._legacy_decline_samples = ['s1', 's2']        # s2 not located yet
    FixesMixin._settle_legacy_declines(m)
    assert m._declined_fixes == ['a']
    assert m._legacy_decline_samples == ['s1', 's2']
    # s2 located: every result affected now is declined for itself.
    m._samples['s2'] = m._samples['s1']
    m._sample_order.append('s2')
    m._sample_gates['s2'], m._sample_gate_order['s2'] = {}, []
    m.add_gate('s2', 'g1')
    FixesMixin._settle_legacy_declines(m)
    assert m._declined_fixes == [fixes.decline_of('a', 's1', 'gate', 'g1'),
                                 fixes.decline_of('a', 's2', 'gate', 'g1')]
    assert m._legacy_decline_samples is None
    assert fixes.scan(m) == []
    # A result that joins after that is offered.
    m.add_gate('s1', 'g2')
    assert _where(fixes.scan(m)) == [('s1', 'g2', 'open')]


def test_a_fix_newer_than_the_correction_is_not_counted_applied(running,
                                                               monkeypatch):
    """Corrected by A under 2.6; C (fixed in 2.7) came after: the result is
    still affected by C, so only A counts as applied (and C is offered)."""
    C = Fix('c', 'Fix C', 'gate.refit', '2.7', redo=_bump)
    monkeypatch.setattr(rs, 'app_version', lambda: '2.6')
    m = _model()
    fixes.apply_rows(m, fixes.plan(m, fixes.scan(m, [A, C]), [A, C]))
    monkeypatch.setattr(rs, 'app_version', lambda: '2.7')
    assert fixes.applied_ids(m, [A, C]) == ['a']
    assert [t['fix'] for t in fixes.scan(m, [A, C])] == ['c', 'c']


def test_a_re_run_follow_up_counts_as_corrected_by_the_fixes_it_predates(
        running, monkeypatch):
    m = _model()
    rows = [dict(r, fixes=[]) for r in fixes.plan(m, fixes.scan(m, [A]), [A])
            if r['sample'] == 's1']                 # as rerun_row builds them
    fixes.apply_rows(m, rows)
    assert fixes.applied_ids(m, [A], samples=['s1']) == ['a']
    assert fixes.applied_ids(m, [A], samples=['s2']) == []


def test_the_status_line_comes_from_what_was_applied_and_declined():
    one = fixes.decline_of('a', 's2', 'gate', 'g1')
    two = fixes.decline_of('b', 's2', 'gate', 'g1')
    assert fixes.apply_status(['a'], [('s1', 'gate', 'g1')], []) == (
        "Applied a to 1 result(s); the originals are kept in the session.")
    assert fixes.apply_status([], [], [one, two]).startswith(
        "1 result(s) left as they were; their fixes will not be offered for "
        "them again")
    assert fixes.apply_status([], [], []) == ''


# ── In the editor ───────────────────────────────────────────────────────────

def _two_samples(root, tmp_path):
    """s1 and s2, each with one gate Pos re-fitted by 2.6.0 (before
    refit-valley was fixed)."""
    from tests.test_fixes import _two_pop
    ed = _editor(root)
    for i, name in enumerate(('s1', 's2'), 3):
        _add(ed, _two_pop(seed=i), name=name, path=str(tmp_path / f'{name}.fcs'))
    for name in ('s1', 's2'):
        ed._set_active_sample(name)
        ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A', 'value': 1.5,
                      'name': 'Pos', 'parent_id': None,
                      'stamp': _stamp('2.6.0', ['FL2-A'])})
    ed._clear_undo_pending()
    ed._refresh_gate_list()
    return ed


def test_editor_apply_ticked_corrects_only_the_ticked_result(
        tk_root, tmp_path, running):
    ed = _two_samples(tk_root, tmp_path)
    win = ed._check_fixes_on_open()
    assert [r['sample'] for r in win.rows] == ['s1', 's2']
    win.toggle('1')                                 # untick s2
    win.apply()
    pos2 = next(iter(ed._sample_gates['s2']))
    assert ed._sample_gates['s2'][pos2]['value'] == 1.5
    assert 'original' not in ed._sample_gates['s2'][pos2]
    assert ed._declined_fixes == [fixes.decline_of(REFIT, 's2', 'gate', pos2)]
    assert ed.status_var.get() == (
        f"Applied {REFIT} to 1 result(s); the originals are kept in the "
        "session. 1 result(s) left as they were; their fixes will not be "
        "offered for them again (Tools > Files & checks > Review fixes for "
        "this session…).")
    win.destroy()
    # Review fixes lists s2 as declined for itself, not "applied".
    menu = ed._review_fixes()
    assert [(r['sample'], r['status']) for r in menu.rows] == [
        ('s2', 'declined')]
    assert menu.tv.set('0', 'note') == 'declined earlier for this result'
    menu.destroy()
    # Exports: only s1 carries the correction.
    from openflo.export_stamp import corrections_for
    assert corrections_for(ed, ['s1']) == [REFIT]
    assert corrections_for(ed, ['s2']) is None
    assert corrections_for(ed) == [REFIT]
    # Saved: applied read from the results, the decline per result.
    data = ed._write_session(str(tmp_path / 'one.flowsession'))
    assert data['fixes_applied'] == [REFIT]
    assert data['fixes_declined'] == [
        fixes.decline_of(REFIT, 's2', 'gate', pos2)]


def test_editor_a_result_not_shown_is_neither_applied_nor_declined(
        tk_root, tmp_path, running):
    ed = _two_samples(tk_root, tmp_path)
    rows = fixes.plan(ed, fixes.scan(ed))
    s1 = [r for r in rows if r['sample'] == 's1']
    ed._apply_fix_rows(s1, s1)                      # s2 was never listed
    assert ed._declined_fixes == []
    assert [t['sample'] for t in fixes.scan(ed)] == ['s2']


def test_editor_a_row_that_could_not_be_redone_is_not_declined(
        tk_root, tmp_path, running):
    """A row with nothing to apply (not loaded, the redo failed) was never a
    choice: it is not declined, so it is offered again."""
    ed = _two_samples(tk_root, tmp_path)
    rows = fixes.plan(ed, fixes.scan(ed))
    rows[1] = dict(rows[1], new=None, why='not loaded (test)')
    ed._apply_fix_rows([rows[0]], rows)
    assert ed._declined_fixes == []
    assert [t['sample'] for t in fixes.scan(ed)] == ['s2']


def _gate_names(ed):
    return {gid: g['name'] for gid, g in ed._sample_gates['s1'].items()}


def test_editor_a_decline_follows_its_gate_when_a_reopen_renumbers_ids(
        tk_root, tmp_path, running):
    """A restore gives gates fresh ids (gate_store.install_gates): after a
    gate was deleted before the save, the ids shift, and each per-result
    decline must follow its gate, not the old id."""
    from tests.test_fixes import _restore, _two_pop
    ed = _editor(tk_root)
    _add(ed, _two_pop(), path=str(tmp_path / 's1.fcs'))
    junk = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                         'value': 1.0, 'name': 'Junk', 'parent_id': None})
    pos = ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A', 'value': 1.5,
                        'name': 'Pos', 'parent_id': None,
                        'stamp': _stamp('2.6.0', ['FL2-A'])})
    kid = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.7,
                        'name': 'Pos2', 'parent_id': pos,
                        'stamp': _stamp('2.6.0', ['FL1-A'])})
    assert (junk, pos, kid) == ('g1', 'g2', 'g3')
    del ed._sample_gates['s1'][junk]
    ed._sample_gate_order['s1'].remove(junk)
    ed._clear_undo_pending()
    ed._refresh_gate_list()
    rows = fixes.plan(ed, fixes.scan(ed))
    ed._apply_fix_rows([], [r for r in rows if r['key'] == pos])
    # A decline of a gate the session no longer holds is dropped on reopen.
    ed._declined_fixes.append(fixes.decline_of(REFIT, 's1', 'gate', 'g9'))
    path = tmp_path / 'one.flowsession'
    ed._write_session(str(path))
    ed2 = _restore(tk_root, path)
    names = _gate_names(ed2)
    assert sorted(names.values()) == ['Pos', 'Pos2']
    assert pos not in names or names[pos] != 'Pos'          # ids did shift
    assert [(names[t['key']], t['status']) for t in fixes.scan(
        ed2, include_settled=True)] == [('Pos', 'declined'), ('Pos2', 'open')]
    new_pos = next(gid for gid, n in names.items() if n == 'Pos')
    assert ed2._declined_fixes == [
        fixes.decline_of(REFIT, 's1', 'gate', new_pos)]


def test_editor_a_decline_of_a_gate_only_another_tab_holds_survives_reopen(
        tk_root, tmp_path, running):
    """Tabs own their gates: a gate drawn only in tab B is restored with
    tab B's own id map. Its decline follows it (remapped through every
    tab's map), not dropped because the active tab does not hold it."""
    from tests.test_fixes import _restore, _two_pop
    ed = _editor(tk_root)
    _add(ed, _two_pop(), path=str(tmp_path / 's1.fcs'))
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.0,
                  'name': 'Base', 'parent_id': None})
    ed._clear_undo_pending()
    ed._refresh_gate_list()
    ts = ed._tabs()
    idx = ts.new_tab()
    ts.tabs[idx].gates, ts.tabs[idx].gate_order = ed._live_gate_copy()
    ed._activate_tab(idx)
    only_b = ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A',
                           'value': 1.5, 'name': 'OnlyB', 'parent_id': None,
                           'stamp': _stamp('2.6.0', ['FL2-A'])})
    ed._refresh_gate_list()
    rows = fixes.plan(ed, fixes.scan(ed))
    assert [r['key'] for r in rows] == [only_b]
    ed._apply_fix_rows([], rows)
    ed._activate_tab(0)
    assert fixes.scan(ed) == []
    path = tmp_path / 'tabs.flowsession'
    ed._write_session(str(path))
    ed2 = _restore(tk_root, path)
    assert len(ed2._tabs().tabs) == 2
    ed2._activate_tab(1)
    names = _gate_names(ed2)
    assert [(names[t['key']], t['status']) for t in fixes.scan(
        ed2, include_settled=True)] == [('OnlyB', 'declined')]


def _legacy_session(root, tmp_path):
    """A session file as a build before per-result declines wrote it: one
    old re-fitted gate Pos on s1, refit-valley declined by bare id."""
    import json

    from tests.test_fixes import _two_pop
    ed = _editor(root)
    _add(ed, _two_pop(), path=str(tmp_path / 's1.fcs'))
    ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A', 'value': 1.5,
                  'name': 'Pos', 'parent_id': None,
                  'stamp': _stamp('2.6.0', ['FL2-A'])})
    ed._clear_undo_pending()
    ed._refresh_gate_list()
    path = tmp_path / 'old.flowsession'
    ed._write_session(str(path))
    data = json.loads(path.read_text(encoding='utf-8'))
    data['fixes_applied'], data['fixes_declined'] = [], [REFIT]
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def test_reopening_a_legacy_session_expands_its_bare_decline(
        tk_root, tmp_path, running):
    from tests.test_fixes import _restore, _review_windows
    ed2 = _restore(tk_root, _legacy_session(tk_root, tmp_path))
    if ed2._session_restore:
        ed2._apply_session_restore()
    assert _review_windows(ed2) == []               # still declined
    pos = next(iter(ed2._sample_gates['s1']))
    assert ed2._declined_fixes == [fixes.decline_of(REFIT, 's1', 'gate', pos)]
    assert ed2._legacy_decline_samples is None
    # A result added afterwards (an old gate copied in, say) is offered.
    ed2._sample_gates['s1']['g7'] = rs.seal_gate_stamp(
        {'kind': 'threshold', 'channel': 'FL2-A', 'value': 1.2,
         'name': 'Later', 'parent_id': None,
         'stamp': _stamp('2.6.0', ['FL2-A'])})
    ed2._sample_gate_order['s1'].append('g7')
    assert [t['name'] for t in fixes.scan(ed2)] == ['Later']


def test_review_fixes_expands_a_bare_decline_too(tk_root, tmp_path, running):
    from tests.test_fixes import _restore
    ed2 = _restore(tk_root, _legacy_session(tk_root, tmp_path))
    ed2._declined_fixes = [REFIT]                  # as restored, not expanded
    ed2._legacy_decline_samples = ['s1']
    menu = ed2._review_fixes()
    pos = next(iter(ed2._sample_gates['s1']))
    assert ed2._declined_fixes == [fixes.decline_of(REFIT, 's1', 'gate', pos)]
    assert [r['status'] for r in menu.rows] == ['declined']
    menu.destroy()


def test_editor_undo_of_apply_offers_the_declined_result_again(
        tk_root, tmp_path, running):
    ed = _two_samples(tk_root, tmp_path)
    rows = fixes.plan(ed, fixes.scan(ed))
    ed._apply_fix_rows([rows[0]], rows)
    assert [t['sample'] for t in fixes.scan(ed)] == []
    ed._undo()
    assert ed._declined_fixes == []
    assert [t['sample'] for t in fixes.scan(ed)] == ['s1', 's2']
    ed._redo()
    assert [d['sample'] for d in ed._declined_fixes] == ['s2']


# ── Exports say the corrections of the samples they were made from ─────────

@pytest.fixture
def s4_corrected(ed, running):
    """The export fixture's editor (s1..s4, s4 not plotted, s2 active) with
    each Pos gate re-fitted by 2.6.0, and refit-valley applied to s4's
    alone through the review window's Apply."""
    for name in ed._sample_order:
        for g in ed._sample_gates[name].values():
            g['stamp'] = _stamp('2.6.0', ['FL1-A'])
            rs.seal_gate_stamp(g)
    rows = fixes.plan(ed, [t for t in fixes.scan(ed) if t['sample'] == 's4'])
    assert len(rows) == 1 and rows[0]['new'] is not None
    ed._apply_fix_rows(rows, rows)
    return ed


def _png_corrections(path):
    from PIL import Image
    with Image.open(path) as im:
        return dict(im.text).get('Corrections')


def test_plot_image_of_uncorrected_samples_claims_no_correction(
        s4_corrected, monkeypatch, tmp_path):
    _action, path, samples = _audit_tests._plot(s4_corrected, monkeypatch,
                                                tmp_path)
    assert samples == ['s1', 's2', 's3']           # s4 is not plotted
    assert _png_corrections(path) is None
    s4_corrected._sample_plot_enabled['s4'] = True
    path.unlink()
    _audit_tests._plot(s4_corrected, monkeypatch, tmp_path)
    assert _png_corrections(path) == REFIT


def test_gatingml_says_the_active_samples_corrections(
        s4_corrected, monkeypatch, tmp_path):
    _action, path, _s = _audit_tests._gatingml(s4_corrected, monkeypatch,
                                               tmp_path)
    assert 'corrections' not in path.read_text(encoding='utf-8')
    s4_corrected._set_active_sample('s4')
    _audit_tests._gatingml(s4_corrected, monkeypatch, tmp_path)
    assert f'corrections: {REFIT}' in path.read_text(encoding='utf-8')


def test_wsp_of_every_sample_says_the_correction(s4_corrected, monkeypatch,
                                                 tmp_path):
    _action, path, _s = _audit_tests._wsp(s4_corrected, monkeypatch,
                                          tmp_path)
    assert f'corrections: {REFIT}' in path.read_text(encoding='utf-8')
    # Without the corrected sample loaded, the workspace claims none.
    s4_corrected._samples.pop('s4')
    path.unlink()
    _audit_tests._wsp(s4_corrected, monkeypatch, tmp_path)
    assert 'corrections' not in path.read_text(encoding='utf-8')


def _fcs_corrections(path):
    from openflo import pipeline as fp
    return fp.FlowSample(str(path)).metadata.get('openflo_corrections')


@pytest.mark.parametrize('all_samples', [False, True], ids=['active', 'all'])
def test_population_files_say_their_own_samples_corrections(
        s4_corrected, monkeypatch, tmp_path, all_samples):
    from openflo import editor_populations
    from openflo.async_task import wait_for_job
    out = tmp_path / 'pops'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    s4_corrected._export_populations_fcs(all_samples=all_samples)
    assert wait_for_job(s4_corrected._job_strip)
    got = {p.parent.name if all_samples else 's2': _fcs_corrections(p)
           for p in out.rglob('*.fcs')}
    assert got == ({'s1': None, 's2': None, 's3': None, 's4': REFIT}
                   if all_samples else {'s2': None})


@pytest.mark.parametrize('name', ['s2', 's4'])
def test_one_population_file_says_its_samples_corrections(
        s4_corrected, monkeypatch, tmp_path, name):
    from openflo import editor_populations
    from openflo.async_task import wait_for_job
    path = tmp_path / f'{name}.fcs'
    monkeypatch.setattr(editor_populations.filedialog, 'asksaveasfilename',
                        lambda **k: str(path))
    gid = next(iter(s4_corrected._sample_gates[name]))
    s4_corrected._export_population_fcs(name, gid)
    assert wait_for_job(s4_corrected._job_strip)
    assert _fcs_corrections(path) == (REFIT if name == 's4' else None)


def _spy_corrections(monkeypatch):
    """Record the samples each export_stamp.corrections_for call names
    (the real function still answers)."""
    from openflo import export_stamp
    calls = []
    real = export_stamp.corrections_for

    def spy(owner, samples=None):
        calls.append(None if samples is None else list(samples))
        return real(owner, samples)
    monkeypatch.setattr(export_stamp, 'corrections_for', spy)
    return calls


def test_layout_exports_name_the_page_and_overlaid_samples(tmp_path,
                                                           monkeypatch):
    from openflo.async_task import wait_for_job
    from tests.test_layout_editor import _editor_or_skip, _load
    root, ed = _editor_or_skip()
    try:
        import openflo.ui_layout_editor as ule
        monkeypatch.setattr(ule.LayoutEditorWindow, 'geometry',
                            lambda self, *a, **k: None)
        _load(ed, 's1', 1, 300, 300.0)
        _load(ed, 's2', 2, 300, 500.0)
        _load(ed, 's3', 3, 300, 500.0)
        win = ule.LayoutEditorWindow(ed)
        win.withdraw()
        win.update()
        win._add_text()
        page = win._sample()
        other = next(n for n in ('s1', 's2', 's3') if n != page)
        # The text item draws for another sample than the page's.
        win.layout.items[next(iter(win.layout.items))]['samples'] = [other]
        calls = _spy_corrections(monkeypatch)

        def export(path, batch):
            calls.clear()
            monkeypatch.setattr(ule.filedialog, 'asksaveasfilename',
                                lambda *a, **k: str(path))
            if batch:
                win._export_batch()
                assert wait_for_job(win._busy)
            else:
                win._export_page()
            return {tuple(c) for c in calls}

        assert export(tmp_path / 'page.png', False) == {(page, other)}
        assert export(tmp_path / 'page.pptx', False) == {(page, other)}
        # The batch runs over every loaded sample: s1, s2, s3; each file
        # (PNG, slide) names its sample and the overlaid one, a file of all
        # samples (PDF, deck) names them all.
        each = {tuple(dict.fromkeys((nm, other))) for nm in ('s1', 's2', 's3')}
        assert export(tmp_path / 'b.pdf', True) == {('s1', 's2', 's3')}
        assert export(tmp_path / 'b.png', True) == each
        assert export(tmp_path / 'b.pptx', True) == each | {('s1', 's2', 's3')}
        # A second page with no overlay: its slides name their sample alone,
        # the first page's slides still name the overlaid sample too.
        win._new_page()
        win._add_text()
        alone = {(nm,) for nm in ('s1', 's2', 's3')}
        assert export(tmp_path / 'c.pptx', True) == (
            each | alone | {('s1', 's2', 's3')})
        win.destroy()
    finally:
        root.destroy()


def test_debarcode_files_say_the_pooled_samples_corrections(env,
                                                            monkeypatch):
    from tkinter import filedialog

    from openflo.async_task import wait_for_job
    _root, ed, tmp = env
    w, _s, _parent, _truth = _debarcode_tests._pooled_editor(env, monkeypatch)
    out = tmp / 'split'
    out.mkdir()
    monkeypatch.setattr(filedialog, 'askdirectory', lambda **k: str(out))
    calls = _spy_corrections(monkeypatch)
    w._save_fcs()
    assert wait_for_job(ed._job_strip)
    assert calls == [['pool']]
