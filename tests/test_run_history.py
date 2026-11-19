"""A run template's runs are kept as a history: each run folder records
what it ran (run_info.json) and history.csv joins every finished run's
population tables, the newest run of each experiment date active
(``openflo.run_history``).

Generated data, generic channel and sample names only.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

from openflo import run_history as H
from openflo import run_template as T
from openflo.workspace_run import WorkspaceModel
from tests._fcs_writer import write_fcs
from tests.conftest import gui_unavailable

GATE = {'g1': {'kind': 'threshold', 'name': 'Big', 'channel': 'FSC-A',
               'value': 100.0, 'id': 'g1', 'parent_id': None}}


def _template():
    m = WorkspaceModel('Course')
    gid = m.new_group('Ctrl')
    for rep in ('1', '2'):
        m.add_item({'sample': f'd1_ctrl_{rep}', 'trial': 'Day 1',
                    'gate_id': 'g1', 'gate_path': 'Big', 'gate_chain': ['g1'],
                    'gates': dict(GATE), 'comp_matrix': None,
                    'comp_channels': [], 'qc_applied': False}, gid)
    return T.template_from_model(m)


# ── versions, folders, keywords ────────────────────────────────────────────

def test_template_version_follows_what_the_template_does(monkeypatch):
    tpl = _template()
    v = H.template_version(tpl)
    assert len(v) == 12
    # The OpenFlo version that saved it does not count.
    import openflo.update as U
    monkeypatch.setattr(U, 'current_version', lambda: '99.0.0')
    assert H.template_version(tpl) == v
    assert H.template_version(T.template_from_json(
        json.loads(json.dumps(T.template_to_json(tpl))))) == v
    changed = dict(tpl, slots=[dict(tpl['slots'][0], pattern='other')])
    assert H.template_version(changed) != v


def test_history_folder_is_beside_the_template_or_in_the_chosen_one(
        tmp_path, monkeypatch):
    t = str(tmp_path / 'course.json')
    assert H.history_dir(t) == str(tmp_path / 'course_history')
    assert H.history_dir(t, '  ') == str(tmp_path / 'course_history')
    out = str(tmp_path / 'out')
    chosen = H.history_dir(t, out)
    assert os.path.dirname(chosen) == out
    assert re.fullmatch(r'course_[0-9a-f]{8}_history', os.path.basename(chosen))
    # One template, one folder, however its path is written.
    monkeypatch.chdir(tmp_path)
    assert H.history_dir('course.json', out) == chosen


def test_two_templates_with_the_same_file_name_do_not_share_a_history(tmp_path):
    chosen = str(tmp_path / 'runs')
    a = H.history_dir(str(tmp_path / 'exp_a' / 'course.json'), chosen)
    b = H.history_dir(str(tmp_path / 'exp_b' / 'course.json'), chosen)
    assert a != b
    _run(a, 'run_20261111_090000', '2026-11-11', count=10)
    _run(b, 'run_20261111_100000', '2026-11-11', count=90)
    H.rebuild_history(a)
    assert [r['run_id'] for r in H.mark_active(H.load_runs(a))
            if r['active']] == ['run_20261111_090000']


def test_a_history_folder_with_glob_characters_is_read(tmp_path):
    h = str(tmp_path / 'course [day]_history')
    _run(h, 'run_1', '2026-11-10')
    assert [r['run_id'] for r in H.load_runs(h)] == ['run_1']


def test_the_version_ignores_the_order_of_keys():
    a = _template()
    b = dict(reversed(list(a.items())))
    b['run_cfg'] = dict(reversed(list(a['run_cfg'].items())))
    assert list(b['run_cfg']) != list(a['run_cfg'])
    assert H.template_version(a) == H.template_version(b)


@pytest.mark.parametrize('text,iso', [
    ('11-NOV-2026', '2026-11-11'), ('2026-11-12', '2026-11-12'),
    ('03-feb-26', '2026-02-03'), ('11/13/2026', '2026-11-13'),
    ('25/11/2026', '2026-11-25'), ('2026/11/14', '2026-11-14'),
    ('14.11.2026', '2026-11-14'), ('', None), (None, None), ('soon', None),
    ('03/04/2026', '2026-03-04'),          # both readings parse: month first
])
def test_fcs_dates(text, iso):
    assert H.parse_fcs_date(text) == iso


def test_voltages_by_detector_name():
    kw = {'PAR': '4', 'P1N': 'FSC-A', 'P1V': '350', 'P2N': 'FL1-A',
          'P2V': '480.5', 'P3N': 'FL2-A', 'P3V': 'n/a', 'P4N': 'Time'}
    assert H.voltages(kw) == {'FSC-A': 350.0, 'FL1-A': 480.5}
    assert H.voltages({'PAR': 'x'}) == {}


def test_fcs_keywords_read_the_text_segment(tmp_path):
    path = write_fcs(str(tmp_path / 'a.fcs'), np.ones((5, 2)),
                     ['FSC-A', 'FL1-A'],
                     extra={'$DATE': '11-NOV-2026', '$P2V': '500'})
    kw = H.read_fcs_keywords(path)
    assert kw['DATE'] == '11-NOV-2026'
    assert H.voltages(kw) == {'FL1-A': 500.0}


# ── the run record ─────────────────────────────────────────────────────────

def test_run_files_lists_each_file_once_and_whether_it_ran():
    m = T.instantiate(_template(), {'/d/x_ctrl_1.fcs': 0, '/d/x_ctrl_2.fcs': 0,
                                    '/d/x_ctrl_3.fcs': 0})
    gid = next(iter(m.groups))
    # file 1: its first row unticked, a second row ticked; file 3 unticked.
    m.add_item({'sample': 'extra', 'path': '/d/x_ctrl_1.fcs'}, gid)
    for _mid, it, _g in m.all_items():
        if it.get('sample') in ('x_ctrl_1', 'x_ctrl_3'):
            it['display'] = False
    assert H.run_files(m) == [
        {'path': '/d/x_ctrl_1.fcs', 'sample': 'x_ctrl_1', 'slot': 'Ctrl',
         'ran': True},
        {'path': '/d/x_ctrl_2.fcs', 'sample': 'x_ctrl_2', 'slot': 'Ctrl',
         'ran': True},
        {'path': '/d/x_ctrl_3.fcs', 'sample': 'x_ctrl_3', 'slot': 'Ctrl',
         'ran': False}]
    # Ticked anywhere means it ran, whichever of its rows comes last.
    for _mid, it, _g in m.all_items():
        if it.get('path') == '/d/x_ctrl_1.fcs':
            it['display'] = it.get('sample') == 'x_ctrl_1'
    assert H.run_files(m)[0]['ran'] is True
    m.groups[gid]['display'] = False
    assert not any(f['ran'] for f in H.run_files(m))


def test_run_info_dates_from_the_files_else_the_run_day():
    kws = {'/a': {'DATE': '12-NOV-2026', 'PAR': '1', 'P1N': 'FL1-A', 'P1V': '400'},
           '/b': {'DATE': '11-NOV-2026'}}

    def read(p):
        if p == '/bad':
            raise OSError('gone')
        return kws.get(p, {})
    files = [{'path': p, 'sample': p, 'slot': 'Ctrl'} for p in ('/a', '/b', '/bad')]
    tpl = {'path': '/t.json', 'name': 't', 'version': 'abc'}
    info = H.build_run_info('run_1', tpl, files, read,
                            today=lambda: datetime.date(2030, 1, 1),
                            stat_forms={'geomean': 'log'})
    # One file each: a tie takes the later date, and says the dates differ.
    assert (info['experiment_date'], info['date_from']) == ('2026-11-12', 'files')
    assert info['status'] == H.STATUS_STARTED
    assert info['template'] == tpl
    assert info['stat_forms'] == {'geomean': 'log'}
    assert info['files'][0]['voltages'] == {'FL1-A': 400.0}
    assert info['notes'] == [
        'bad: keywords not read (OSError: gone)',
        "the files' $DATE differ (2026-11-11: 1 file(s), 2026-11-12: 1 "
        "file(s)): the run is dated 2026-11-12, the latest of them"]
    none = H.build_run_info('run_2', tpl, files[2:], read,
                            today=lambda: datetime.date(2030, 1, 1))
    assert (none['experiment_date'], none['date_from']) == ('2030-01-01', 'run day')


def test_a_carried_over_old_file_does_not_redate_the_run():
    dates = {'new1': '12-NOV-2026', 'new2': '12-NOV-2026', 'old': '10-NOV-2026',
             'off': '01-NOV-2026'}
    files = [{'path': p, 'sample': p, 'slot': 'A', 'ran': p != 'off'}
             for p in dates]
    info = H.build_run_info('run_x', {}, files,
                            read_keywords=lambda p: {'DATE': dates[p]})
    assert info['experiment_date'] == '2026-11-12'
    assert info['notes'] == [
        "the files' $DATE differ (2026-11-10: 1 file(s), 2026-11-12: 2 "
        "file(s)): the run is dated 2026-11-12, the latest of them"]
    # The unticked file's date is no vote, even alone.
    only_off = H.build_run_info('run_y', {}, files[3:],
                                read_keywords=lambda p: {'DATE': dates[p]},
                                today=lambda: datetime.date(2030, 1, 1))
    assert only_off['experiment_date'] == '2030-01-01'


def test_a_majority_of_old_files_does_not_move_the_run_back():
    assert H.experiment_date(['2026-11-10', '2026-11-10', '2026-11-12']) == (
        '2026-11-12', "the files' $DATE differ (2026-11-10: 2 file(s), "
        "2026-11-12: 1 file(s)): the run is dated 2026-11-12, the latest of them")
    assert H.experiment_date(['2026-11-12', '2026-11-10']) == (
        '2026-11-12', "the files' $DATE differ (2026-11-10: 1 file(s), "
        "2026-11-12: 1 file(s)): the run is dated 2026-11-12, the latest of them")


def test_files_that_agree_say_nothing():
    assert H.experiment_date(['2026-11-12'] * 3) == ('2026-11-12', None)
    assert H.experiment_date([]) == (None, None)


def test_a_record_without_ran_counts_its_date():
    info = H.build_run_info('run_z', {}, [{'path': 'a', 'sample': 'a',
                                           'slot': 'A'}],
                            read_keywords=lambda p: {'DATE': '12-NOV-2026'})
    assert (info['experiment_date'], info['notes']) == ('2026-11-12', [])


def test_detector_names_are_stripped():
    kw = {'PAR': '2', 'P1N': ' FL1-A ', 'P1V': '400', 'P2N': 'FL2-A',
          'P2V': '500'}
    assert H.voltages(kw) == {'FL1-A': 400.0, 'FL2-A': 500.0}


# ── the history ────────────────────────────────────────────────────────────

def _run(hdir, run_id, date, status=H.STATUS_DONE, count=10, version='v1',
         volts=None):
    d = os.path.join(hdir, run_id)
    os.makedirs(d)
    H.write_run_info(d, {'run_id': run_id, 'status': status,
                         'experiment_date': date,
                         'template': {'version': version},
                         'files': [{'voltages': volts or {'FL1-A': 400.0}}]})
    with open(os.path.join(d, 'Ctrl_population_stats.csv'), 'w',
              encoding='utf-8') as fh:
        fh.write('Group,Trial,Sample,Population,Count\n'
                 f'Ctrl,Day,s,Big,{count}\n')
    return d


def test_the_newest_finished_run_of_a_date_is_active(tmp_path):
    h = str(tmp_path)
    _run(h, 'run_1', '2026-11-11')
    _run(h, 'run_2', '2026-11-11')                     # the same day again
    _run(h, 'run_3', '2026-11-12')
    _run(h, 'run_4', '2026-11-12', status=H.STATUS_CANCELLED)
    _run(h, 'run_5', '2026-11-13', status=H.STATUS_STARTED)
    os.makedirs(os.path.join(h, 'run_6'))               # no record
    runs = H.mark_active(H.load_runs(h))
    assert [(r['run_id'], r['active']) for r in runs] == [
        ('run_1', False), ('run_2', True), ('run_3', True),
        ('run_4', False), ('run_5', False)]


def _rows(path):
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def test_history_joins_finished_runs_and_marks_the_superseded(tmp_path):
    h = str(tmp_path)
    _run(h, 'run_1', '2026-11-11', count=10)
    _run(h, 'run_2', '2026-11-11', count=11)
    _run(h, 'run_3', '2026-11-12', count=12, version='v2')
    _run(h, 'run_4', '2026-11-13', status=H.STATUS_CANCELLED, count=99)
    path, n = H.rebuild_history(h)
    assert (path, n) == (os.path.join(h, 'history.csv'), 3)
    rows = _rows(path)
    assert list(rows[0])[:5] == ['Run', 'Experiment date', 'Active',
                                 'Template version', 'Group']
    assert [(r['Run'], r['Active'], r['Count'], r['Template version'])
            for r in rows] == [('run_1', 'False', '10', 'v1'),
                               ('run_2', 'True', '11', 'v1'),
                               ('run_3', 'True', '12', 'v2')]


def test_no_finished_run_removes_an_old_history(tmp_path):
    h = str(tmp_path)
    (tmp_path / 'history.csv').write_text('old', encoding='utf-8')
    _run(h, 'run_1', '2026-11-11', status=H.STATUS_CANCELLED)
    assert H.rebuild_history(h) == (None, 0)
    assert not (tmp_path / 'history.csv').exists()


def test_set_run_status(tmp_path):
    d = _run(str(tmp_path), 'run_1', '2026-11-11', status=H.STATUS_STARTED)
    H.set_run_status(d, H.STATUS_DONE)
    assert H.load_runs(str(tmp_path))[0]['status'] == H.STATUS_DONE


def test_changes_since_the_previous_days_run(tmp_path):
    h = str(tmp_path)
    _run(h, 'run_1', '2026-11-11', volts={'FL1-A': 400.0, 'FL2-A': 500.0})
    _run(h, 'run_2', '2026-11-12', version='v2',
         volts={'FL1-A': 410.0, 'FL2-A': 500.0})
    _run(h, 'run_3', '2026-11-12', version='v2',
         volts={'FL1-A': 410.0, 'FL2-A': 500.0})        # the day again
    runs = H.mark_active(H.load_runs(h))
    assert H.changes_since_last(runs, 'run_1') == []
    got = H.changes_since_last(runs, 'run_3')
    assert len(got) == 2
    assert got[0].startswith('the template changed since the 2026-11-11 run')
    assert got[1].startswith('detector voltages changed since the 2026-11-11 '
                             'run (FL1-A)')
    assert H.changes_since_last(runs, 'run_9') == []


def test_a_cancelled_run_between_days_is_not_compared(tmp_path):
    h = str(tmp_path)
    _run(h, 'run_1', '2026-11-11')
    _run(h, 'run_2', '2026-11-12', status=H.STATUS_CANCELLED,
         volts={'FL1-A': 999.0}, version='vX')
    _run(h, 'run_3', '2026-11-13')
    runs = H.mark_active(H.load_runs(h))
    assert H.changes_since_last(runs, 'run_3') == []


def _record(hdir, run_id, date, files, stat_forms=None):
    d = os.path.join(hdir, run_id)
    os.makedirs(d)
    rec = {'run_id': run_id, 'status': H.STATUS_DONE, 'experiment_date': date,
           'template': {'version': 'v1'}, 'files': files}
    if stat_forms is not None:
        rec['stat_forms'] = stat_forms
    H.write_run_info(d, rec)
    # A finished run has written a table (H.finished_status).
    with open(os.path.join(d, 'Ctrl_population_stats.csv'), 'w',
              encoding='utf-8') as fh:
        fh.write('Group,Sample,Population,Count\nCtrl,s,Big,1\n')


def test_a_voltage_that_moved_in_any_file_is_said(tmp_path):
    h = str(tmp_path)
    _record(h, 'run_1', '2026-11-11', [{'voltages': {'FL1-A': 400.0}},
                                       {'voltages': {'FL1-A': 400.0}}])
    # The second file moved; the last file read did not.
    _record(h, 'run_2', '2026-11-12', [{'voltages': {'FL1-A': 450.0}},
                                       {'voltages': {'FL1-A': 400.0}}])
    runs = H.mark_active(H.load_runs(h))
    [got] = H.changes_since_last(runs, 'run_2')
    assert got.startswith('detector voltages changed since the 2026-11-11 '
                          'run (FL1-A)')


def test_a_changed_geomean_form_is_said(tmp_path):
    h = str(tmp_path)
    _record(h, 'run_1', '2026-11-11', [], {'geomean': 'log'})
    _record(h, 'run_2', '2026-11-12', [], {'geomean': 'log'})
    _record(h, 'run_3', '2026-11-13', [], {'geomean': 'asinh'})
    _record(h, 'run_4', '2026-11-14', [])           # an older record: no forms
    _record(h, 'run_5', '2026-11-15', [], {'geomean': 'log'})
    runs = H.mark_active(H.load_runs(h))
    assert H.changes_since_last(runs, 'run_2') == []
    assert H.changes_since_last(runs, 'run_4') == []
    assert H.changes_since_last(runs, 'run_5') == []
    [got] = H.changes_since_last(runs, 'run_3')
    assert got == ("the statistic forms (Preferences: GeoMean) changed since "
                   "the 2026-11-12 run: GeoMean is not comparable across "
                   "that change")


def test_unchanged_days_say_nothing(tmp_path):
    h = str(tmp_path)
    _run(h, 'run_1', '2026-11-11')
    _run(h, 'run_2', '2026-11-12')
    runs = H.mark_active(H.load_runs(h))
    assert H.changes_since_last(runs, 'run_2') == []


def test_the_workspace_keeps_its_template_through_save_and_undo():
    m = WorkspaceModel()
    m.template = {'path': '/t/course.json', 'name': 'course', 'version': 'v'}
    back = WorkspaceModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert back.template == m.template
    assert WorkspaceModel.from_dict(WorkspaceModel().to_dict()).template is None


# ── the panel ──────────────────────────────────────────────────────────────

def test_a_template_run_goes_in_the_history(tmp_path, monkeypatch):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        from openflo import prefs
        from openflo.workspace import WorkspacePanel
        monkeypatch.setattr(prefs, 'load_pipeline_output_dir', lambda: None)
        f = write_fcs(str(tmp_path / 'd2_ctrl_1.fcs'), np.ones((5, 2)),
                      ['FSC-A', 'FL1-A'], extra={'$DATE': '12-NOV-2026'})
        tpath = str(tmp_path / 'course.json')
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        plain = panel._run_root()
        assert plain == panel._output_root()
        panel._start_history(str(tmp_path), 'run_0')     # not a template run
        assert not (tmp_path / H.RUN_INFO).exists()

        panel._load_from_template(_template(), tpath, {f: 0}, run=False)
        assert panel.model.template['version'] == H.template_version(_template())
        hdir = str(tmp_path / 'course_history')
        assert panel._run_root() == hdir
        out = os.path.join(hdir, 'run_1')
        os.makedirs(out)
        panel._start_history(out, 'run_1')
        info = H.load_runs(hdir)[0]
        assert info['experiment_date'] == '2026-11-12'
        assert info['template']['path'] == tpath
        with open(os.path.join(out, 'Ctrl_population_stats.csv'), 'w',
                  encoding='utf-8') as fh:
            fh.write('Sample,Population,Count\nd2_ctrl_1,Big,3\n')
        panel._finish('Done: 1 ok', done=True)
        assert panel.status_var.get() == \
            'Done: 1 ok  History: 1 row(s) in history.csv.'
        assert H.load_runs(hdir)[0]['status'] == H.STATUS_DONE

        out2 = os.path.join(hdir, 'run_2')
        os.makedirs(out2)
        # A cancelled run is compared with nothing, even one that changed.
        panel.model.template['version'] = 'changed'
        panel._start_history(out2, 'run_2')
        panel._finish('Cancelled.', done=False)
        assert H.load_runs(hdir)[1]['status'] == H.STATUS_CANCELLED
        assert panel.status_var.get() == \
            'Cancelled.  History: 1 row(s) in history.csv.'
        # A run that ends with no history open adds nothing.
        panel._finish('Done: 0 ok')
        assert panel.status_var.get() == 'Done: 0 ok'
    finally:
        root.destroy()


def test_a_record_that_cannot_be_written_is_said(tmp_path, monkeypatch):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        from openflo.workspace import WorkspacePanel
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        panel.model.template = {'path': str(tmp_path / 't.json'),
                                'name': 't', 'version': 'v'}
        panel._start_history(str(tmp_path / 'missing' / 'run_1'), 'run_1')
        panel._finish('Done: 1 ok')
        msg = panel.status_var.get()
        assert msg.startswith('Done: 1 ok  [!] not added to the run history '
                              '(FileNotFoundError')
        # The next run that is recorded does not repeat that note.
        panel._start_history(str(tmp_path / 'missing' / 'run_2'), 'run_2')
        out = tmp_path / 'run_3'
        out.mkdir()
        panel._start_history(str(out), 'run_3')
        panel._finish('Done: 1 ok')
        # run_3 wrote no table: it is marked failed and says so.
        assert panel.status_var.get() == (
            'Done: 1 ok  [!] no population table with a row was written: this run is '
            'not in the history and replaces no earlier run.')
        assert H.load_runs(str(tmp_path))[0]['status'] == H.STATUS_FAILED
    finally:
        root.destroy()


def test_files_of_two_dates_are_said_on_the_final_line(tmp_path):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        from openflo.workspace import WorkspacePanel
        files = {}
        for i, date in enumerate(('12-NOV-2026', '12-NOV-2026', '10-NOV-2026')):
            files[write_fcs(str(tmp_path / f'd2_ctrl_{i}.fcs'), np.ones((5, 2)),
                            ['FSC-A', 'FL1-A'], extra={'$DATE': date})] = 0
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        panel.model = T.instantiate(_template(), files)
        panel.model.template = {'path': str(tmp_path / 'course.json'),
                                'name': 'course', 'version': 'v'}
        out = tmp_path / 'course_history' / 'run_1'
        out.mkdir(parents=True)
        panel._start_history(str(out), 'run_1', {'geomean': 'log'})
        info = H.load_runs(str(out.parent))[0]
        assert info['experiment_date'] == '2026-11-12'
        assert info['stat_forms'] == {'geomean': 'log'}
        panel._finish('Done: 1 ok')
        assert panel.status_var.get() == (
            "Done: 1 ok  [!] the files' $DATE differ (2026-11-10: 1 file(s), "
            "2026-11-12: 2 file(s)): the run is dated 2026-11-12, the latest "
            "of them  [!] no population table with a row was written: this run is not "
            "in the history and replaces no earlier run.")
    finally:
        root.destroy()


def test_run_starts_a_template_run_in_the_history(tmp_path, monkeypatch):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        import openflo.workspace as W
        from openflo import prefs
        monkeypatch.setattr(prefs, 'load_pipeline_output_dir', lambda: None)
        monkeypatch.setattr(prefs, 'load_stat_forms',
                            lambda: {'geomean': 'asinh'})
        monkeypatch.setattr(W, 'build_run_units', lambda model, cfg: [
            {'label': 'Ctrl', 'members': [], 'gid': None}])
        panel = W.WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        panel._load_from_template(_template(), str(tmp_path / 'course.json'),
                                  {str(tmp_path / 'd2_ctrl_1.fcs'): 0},
                                  run=False)
        monkeypatch.setattr(panel, 'after', lambda *a, **k: None)
        panel._run()
        hdir = tmp_path / 'course_history'
        [run] = H.load_runs(str(hdir))
        assert run['dir'] == panel._out_dir
        assert run['status'] == H.STATUS_STARTED
        assert run['stat_forms'] == {'geomean': 'asinh'}
        assert panel._history_run == panel._out_dir
    finally:
        root.destroy()


def test_the_history_follows_the_chosen_run_folder(tmp_path, monkeypatch):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        from openflo import prefs
        from openflo.workspace import WorkspacePanel
        chosen = str(tmp_path / 'runs')
        monkeypatch.setattr(prefs, 'load_pipeline_output_dir', lambda: chosen)
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        panel.model.template = {'path': str(tmp_path / 'course.json'),
                                'name': 'course', 'version': 'v'}
        assert panel._run_root() == H.history_dir(str(tmp_path / 'course.json'),
                                                  chosen)
        assert os.path.dirname(panel._run_root()) == chosen
    finally:
        root.destroy()


# ── which rows count: failed and partial re-runs (audit H1) ────────────────

def _day(hdir, run_id, date, rows, status=H.STATUS_DONE, volts=None,
         version='v1'):
    """A run folder with one population table per group in `rows`
    (``(group, sample, count)``)."""
    d = os.path.join(hdir, run_id)
    os.makedirs(d)
    H.write_run_info(d, {'run_id': run_id, 'status': status,
                         'experiment_date': date,
                         'template': {'version': version},
                         'files': [{'voltages': volts or {'FL1-A': 400.0}}]})
    by: dict[str, list[str]] = {}
    for g, s, n in rows:
        by.setdefault(g, []).append(f'{g},Day,{s},Big,{n}\n')
    for g, lines in by.items():
        with open(os.path.join(d, f'{g}_population_stats.csv'), 'w',
                  encoding='utf-8') as fh:
            fh.write('Group,Trial,Sample,Population,Count\n' + ''.join(lines))
    return d


def _active(hdir):
    return sorted((r['Run'], r['Group'], r['Sample'], r['Count'])
                  for r in _rows(os.path.join(hdir, 'history.csv'))
                  if r['Active'] == 'True')


def test_a_run_that_wrote_no_table_is_failed_and_replaces_nothing(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 'a1', 10), ('B', 'b1', 20)])
    # The day again, every unit failing before its table.
    d2 = _day(h, 'run_2', '2026-11-11', [], status=H.STATUS_STARTED)
    assert H.finished_status(d2, done=True) == H.STATUS_FAILED
    H.set_run_status(d2, H.finished_status(d2, done=True))
    H.rebuild_history(h)
    assert _active(h) == [('run_1', 'A', 'a1', '10'), ('run_1', 'B', 'b1', '20')]
    assert [r['run_id'] for r in H.mark_active(H.load_runs(h))
            if r['active']] == ['run_1']


def test_a_header_only_table_is_no_table(tmp_path):
    d = _day(str(tmp_path), 'run_1', '2026-11-11', [])
    assert H.finished_status(d, done=True) == H.STATUS_FAILED
    (tmp_path / 'run_1' / 'A_population_stats.csv').write_text(
        'Group,Trial,Sample,Population,Count\n', encoding='utf-8')
    assert H.finished_status(d, done=True) == H.STATUS_FAILED
    (tmp_path / 'run_1' / 'C_population_stats.csv').write_text(
        '', encoding='utf-8')                    # empty: unreadable
    assert H.finished_status(d, done=True) == H.STATUS_FAILED
    (tmp_path / 'run_1' / 'B_population_stats.csv').write_text(
        'Group,Trial,Sample,Population,Count\nB,Day,b1,Big,3\n',
        encoding='utf-8')
    assert H.finished_status(d, done=True) == H.STATUS_DONE
    assert H.finished_status(d, done=False) == H.STATUS_CANCELLED


def test_a_rerun_of_one_group_replaces_only_that_group(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 'a1', 10), ('B', 'b1', 20)])
    _day(h, 'run_2', '2026-11-11', [('A', 'a1', 11)])     # A again only
    _day(h, 'run_3', '2026-11-12', [('A', 'a1', 30), ('B', 'b1', 40)])
    H.rebuild_history(h)
    assert _active(h) == [('run_1', 'B', 'b1', '20'), ('run_2', 'A', 'a1', '11'),
                          ('run_3', 'A', 'a1', '30'), ('run_3', 'B', 'b1', '40')]
    runs = H.mark_active(H.load_runs(h))
    assert [r['active'] for r in runs] == [True, True, True]
    # The comparison sees one value per group and day, each day once.
    from openflo import run_compare as C
    table = C.per_run_means(C.load_history(h), 'Big', 'Count')
    assert table.to_dict() == {'A': {'2026-11-11': 11.0, '2026-11-12': 30.0},
                               'B': {'2026-11-11': 20.0, '2026-11-12': 40.0}}


def test_a_rerun_of_every_group_supersedes_the_whole_day(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 'a1', 10), ('B', 'b1', 20)])
    _day(h, 'run_2', '2026-11-11', [('A', 'a1', 11), ('B', 'b1', 21)])
    H.rebuild_history(h)
    assert _active(h) == [('run_2', 'A', 'a1', '11'), ('run_2', 'B', 'b1', '21')]
    assert [r['active'] for r in H.mark_active(H.load_runs(h))] == [False, True]


def test_a_sample_moved_to_another_group_is_not_counted_twice(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 'a1', 10), ('B', 'x', 20)])
    _day(h, 'run_2', '2026-11-11', [('C', 'x', 21)])      # x moved to C
    H.rebuild_history(h)
    assert _active(h) == [('run_1', 'A', 'a1', '10'), ('run_2', 'C', 'x', '21')]


def test_tables_without_a_sample_column_are_kept_by_group(tmp_path):
    h = str(tmp_path)
    for rid, groups in (('run_1', 'AB'), ('run_2', 'A')):
        d = _day(h, rid, '2026-11-11', [])
        for g in groups:
            with open(os.path.join(d, f'{g}_population_stats.csv'), 'w',
                      encoding='utf-8') as fh:
                fh.write(f'Group,Population,Count\n{g},Big,1\n')
    H.rebuild_history(h)
    assert sorted((r['Run'], r['Group']) for r in
                  _rows(os.path.join(h, 'history.csv'))
                  if r['Active'] == 'True') == [('run_1', 'B'), ('run_2', 'A')]


def test_rows_without_a_sample_are_not_moved_by_another_group(tmp_path):
    h = str(tmp_path)
    for rid, groups in (('run_1', 'AB'), ('run_2', 'C')):
        d = _day(h, rid, '2026-11-11', [])
        for g in groups:
            with open(os.path.join(d, f'{g}_population_stats.csv'), 'w',
                      encoding='utf-8') as fh:
                fh.write(f'Group,Population,Count\n{g},Big,1\n')
    H.rebuild_history(h)
    assert sorted((r['Run'], r['Group']) for r in
                  _rows(os.path.join(h, 'history.csv'))
                  if r['Active'] == 'True') == [('run_1', 'A'), ('run_1', 'B'),
                                                ('run_2', 'C')]


def test_group_names_are_matched_without_surrounding_spaces(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 'a1', 10)])
    _day(h, 'run_2', '2026-11-11', [(' A ', 'a2', 11)])
    H.rebuild_history(h)
    assert [r[0] for r in _active(h)] == ['run_2']


def test_a_partial_rerun_is_compared_with_the_day_before(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 'a1', 1)], volts={'FL1-A': 450.0})
    _day(h, 'run_2', '2026-11-12', [('A', 'a1', 1), ('B', 'b1', 1)],
         volts={'FL1-A': 400.0}, version='v0')
    _day(h, 'run_3', '2026-11-12', [('A', 'a1', 2)], volts={'FL1-A': 450.0})
    runs = H.mark_active(H.load_runs(h))
    # run_2 is still active (group B), but it is the same day: run_3 is
    # compared with 2026-11-11's run, which had the same voltage and version.
    assert runs[1]['active']
    assert H.changes_since_last(runs, 'run_3') == []


# ── the tables are read back whatever the CSV style (audit H2) ─────────────

@pytest.mark.parametrize('style', ['standard', 'system', 'semicolon'])
def test_population_tables_and_history_ignore_the_csv_style(tmp_path,
                                                           monkeypatch, style):
    import openflo.table_export as TE
    from openflo import run_compare as C
    from openflo.workspace_run import compute_run
    monkeypatch.setattr(TE, 'csv_style', lambda: style)
    # Decimal commas, as a comma-decimal locale's system style writes.
    monkeypatch.setattr(TE, 'system_csv_format', lambda: (';', ','))
    d = os.path.join(str(tmp_path), 'run_1')
    os.makedirs(d)
    H.write_run_info(d, {'run_id': 'run_1', 'status': H.STATUS_DONE,
                         'experiment_date': '2026-11-11'})
    rng = np.random.default_rng(0)
    import pandas as pd
    data = pd.DataFrame(rng.normal(5, 1, (40, 2)), columns=['FL1-A', 'FL2-A'])
    prep = {'data': data, 'channels': ['FL1-A', 'FL2-A'], 'label': 'A',
            'pop_rows': [{'Group': 'A', 'Trial': 'Day', 'Sample': '007',
                          'Population': 'Big', 'Count': 40, '%Parent': 12.5}]}
    compute_run(prep, {'method': 'nope'}, d)   # the table comes first
    with open(os.path.join(d, 'A_population_stats.csv'),
              encoding='utf-8') as fh:
        assert fh.readline().rstrip('\n') == \
            'Group,Trial,Sample,Population,Count,%Parent'
    path, n = H.rebuild_history(str(tmp_path))
    assert n == 1
    with open(path, encoding='utf-8') as fh:
        assert fh.readline().startswith('Run,Experiment date,Active,')
    df = C.load_history(str(tmp_path))
    assert C.populations(df) == ['Big']
    assert df['Sample'].tolist() == ['007']
    assert df['%Parent'].tolist() == [12.5]


def test_tables_written_in_a_semicolon_style_are_still_read(tmp_path):
    from openflo.table_export import write_rows_csv
    d = _day(str(tmp_path), 'run_1', '2026-11-11', [])
    write_rows_csv(os.path.join(d, 'A_population_stats.csv'),
                   ['Group', 'Sample', 'Population', 'Count', '%Parent'],
                   [['A', '007', 'Big', 40, 12.5]], style='semicolon')
    _path, n = H.rebuild_history(str(tmp_path))
    from openflo import run_compare as C
    df = C.load_history(str(tmp_path))
    assert (n, df['%Parent'].tolist(), df['Sample'].tolist()) == \
        (1, [12.5], ['007'])


# ── one bad file does not stop the history (audit M5) ──────────────────────

def test_an_unreadable_table_is_left_out_and_said(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 'a1', 10)])
    d2 = _day(h, 'run_2', '2026-11-12', [('A', 'a1', 11)])
    open(os.path.join(d2, 'B_population_stats.csv'), 'w').close()  # empty
    notes: list[str] = []
    path, n = H.rebuild_history(h, notes)
    assert n == 2
    [note] = notes
    assert note.startswith('run_2/B_population_stats.csv not read (EmptyDataError')
    # Without a list to say it in, it is still only left out.
    assert H.rebuild_history(h) == (path, 2)


def test_the_run_record_is_written_whole_or_not_at_all(tmp_path):
    d = str(tmp_path)
    H.write_run_info(d, {'run_id': 'run_1', 'status': H.STATUS_STARTED})
    with pytest.raises(TypeError):
        H.write_run_info(d, {'run_id': 'run_1', 'bad': object()})
    with open(os.path.join(d, H.RUN_INFO), encoding='utf-8') as fh:
        assert json.load(fh) == {'run_id': 'run_1', 'status': H.STATUS_STARTED}
    assert os.listdir(d) == [H.RUN_INFO]


def test_a_history_table_is_written_whole_or_not_at_all(tmp_path, monkeypatch):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 'a1', 10)])
    path, _n = H.rebuild_history(h)
    before = open(path, encoding='utf-8').read()
    _day(h, 'run_2', '2026-11-12', [('A', 'a1', 11)])
    import openflo.table_export as TE

    def half(p, header, rows, **kw):
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write('Run,Exp')
        raise OSError('disk full')
    monkeypatch.setattr(TE, 'write_rows_csv', half)
    with pytest.raises(OSError):
        H.rebuild_history(h)
    assert open(path, encoding='utf-8').read() == before
    assert not [n for n in os.listdir(h) if n.endswith('.tmp')]


def test_an_unreadable_table_is_said_on_the_final_line(tmp_path):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        from openflo.workspace import WorkspacePanel
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        panel.model.template = {'path': str(tmp_path / 'course.json'),
                                'name': 'course', 'version': 'v'}
        hdir = tmp_path / 'course_history'
        d1 = _day(str(hdir), 'run_1', '2026-11-11', [('A', 'a1', 10)])
        open(os.path.join(d1, 'B_population_stats.csv'), 'w').close()
        out = hdir / 'run_2'
        out.mkdir()
        panel._start_history(str(out), 'run_2')
        (out / 'A_population_stats.csv').write_text(
            'Group,Sample,Population,Count\nA,a1,Big,3\n', encoding='utf-8')
        panel._finish('Done: 1 ok')
        assert panel.status_var.get().startswith(
            'Done: 1 ok  History: 2 row(s) in history.csv.  [!] '
            'run_1/B_population_stats.csv not read (EmptyDataError')
        assert H.load_runs(str(hdir))[1]['status'] == H.STATUS_DONE
    finally:
        root.destroy()


def test_a_failed_run_is_compared_with_nothing(tmp_path):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        from openflo.workspace import WorkspacePanel
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        panel.model.template = {'path': str(tmp_path / 'course.json'),
                                'name': 'course', 'version': 'changed'}
        hdir = tmp_path / 'course_history'
        _day(str(hdir), 'run_1', '2000-01-01', [('A', 'a1', 10)])
        out = hdir / 'run_2'
        out.mkdir()
        panel._start_history(str(out), 'run_2')     # dated the run day
        panel._finish('Done: 0 ok')
        assert panel.status_var.get() == (
            'Done: 0 ok  [!] no population table with a row was written: this run is '
            'not in the history and replaces no earlier run.  History: 1 '
            'row(s) in history.csv.')
    finally:
        root.destroy()


# ── #404 review follow-ups ─────────────────────────────────────────────────

def test_a_rerun_keeps_a_same_named_sample_of_another_group(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 's1', 10), ('B', 's1', 20),
                                    ('B', 's2', 30)])
    _day(h, 'run_2', '2026-11-11', [('A', 's1', 11)])     # A again only
    H.rebuild_history(h)
    # s1 is in A just as it was: no move, so B's s1 (another file) stays.
    assert _active(h) == [('run_1', 'B', 's1', '20'), ('run_1', 'B', 's2', '30'),
                          ('run_2', 'A', 's1', '11')]


def test_a_sample_moved_out_of_an_ungrouped_row_is_counted_once(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 's1', 10), ('', 'x', 20)])
    _day(h, 'run_2', '2026-11-11', [('A', 's1', 11), ('A', 'x', 21)])
    H.rebuild_history(h)
    assert _active(h) == [('run_2', 'A', 's1', '11'), ('run_2', 'A', 'x', '21')]


def test_a_rerun_of_one_ungrouped_sample_replaces_only_it(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('', 'x1', 10), ('', 'x2', 20)])
    _day(h, 'run_2', '2026-11-11', [('', 'x1', 11)])
    H.rebuild_history(h)
    assert _active(h) == [('run_1', '', 'x2', '20'), ('run_2', '', 'x1', '11')]


def test_an_unreadable_table_of_a_failed_run_is_named(tmp_path):
    d = _day(str(tmp_path), 'run_1', '2026-11-11', [])
    (tmp_path / 'run_1' / 'A_population_stats.csv').write_text(
        '', encoding='utf-8')
    notes: list[str] = []
    assert H.finished_status(d, done=True, notes=notes) == H.STATUS_FAILED
    assert len(notes) == 1 and notes[0].startswith(
        'run_1/A_population_stats.csv not read (EmptyDataError')
    assert H.failed_line(notes).startswith(
        'no population table of this run could be read: ')
    assert H.failed_line([]).startswith('no population table with a row was written: ')
    # A finished run's unreadable tables are said by the history instead.
    (tmp_path / 'run_1' / 'B_population_stats.csv').write_text(
        'Group,Sample,Population,Count\nB,b1,Big,3\n', encoding='utf-8')
    done: list[str] = []
    assert H.finished_status(d, done=True, notes=done) == H.STATUS_DONE
    assert done == []


def test_tables_written_in_a_tab_style_are_still_read(tmp_path):
    p = tmp_path / 'A_population_stats.csv'
    p.write_text('Group\tTrial\tSample\tPopulation\tCount\t%Parent\n'
                 'A\tDay\t007\tBig\t10\t12,5\n', encoding='utf-8-sig')
    df = H.read_history_table(str(p))
    assert list(df.columns) == ['Group', 'Trial', 'Sample', 'Population',
                                'Count', '%Parent']
    assert df.iloc[0]['Sample'] == '007' and df.iloc[0]['%Parent'] == 12.5
    p.write_text('Group\tSample\tPopulation\t%Parent\nA\ts\tBig\t12.5\n',
                 encoding='utf-8')
    assert H.read_history_table(str(p)).iloc[0]['%Parent'] == 12.5


def test_update_history_reads_each_table_once(tmp_path, monkeypatch):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 'a1', 10), ('B', 'b1', 20)])
    _day(h, 'run_2', '2026-11-11', [('A', 'a1', 11), ('B', 'b1', 21)])
    _day(h, 'run_3', '2026-11-12', [('A', 'a1', 12)])
    read = []
    real = H.read_history_table

    def counted(p):
        read.append(os.path.basename(os.path.dirname(p)))
        return real(p)
    monkeypatch.setattr(H, 'read_history_table', counted)
    path, n, runs = H.update_history(h)
    assert sorted(read) == ['run_1', 'run_1', 'run_2', 'run_2', 'run_3']
    assert (os.path.basename(path), n) == ('history.csv', 5)
    assert [r['active'] for r in runs] == [False, True, True]
    monkeypatch.setattr(H, 'read_history_table', real)
    assert [r['active'] for r in H.mark_active(H.load_runs(h))] == \
        [False, True, True]
    # No finished run with a table: no history, every run still marked.
    f = str(tmp_path / 'failed_only')
    _day(f, 'run_1', '2026-11-11', [], status=H.STATUS_FAILED)
    assert H.update_history(f)[:2] == (None, 0)
    assert [r['active'] for r in H.update_history(f)[2]] == [False]


def test_the_final_line_names_the_unreadable_table_of_a_failed_run(
        tmp_path, monkeypatch):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        from openflo.workspace import WorkspacePanel
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        panel.model.template = {'path': str(tmp_path / 'course.json'),
                                'name': 'course', 'version': 'v'}
        hdir = tmp_path / 'course_history'
        _day(str(hdir), 'run_1', '2000-01-01', [('A', 'a1', 10)])
        out = hdir / 'run_2'
        out.mkdir()
        panel._start_history(str(out), 'run_2')
        (out / 'A_population_stats.csv').write_text('', encoding='utf-8')
        read = []
        real = H.read_history_table
        monkeypatch.setattr(H, 'read_history_table',
                            lambda p: read.append(p) or real(p))
        panel._finish('Done: 1 ok')
        assert panel.status_var.get().startswith(
            'Done: 1 ok  [!] no population table of this run could be read: '
            'this run is not in the history and replaces no earlier run.  '
            'History: 1 row(s) in history.csv.  [!] '
            'run_2/A_population_stats.csv not read (EmptyDataError')
        # Its own table once (its status), run_1's once (the history).
        assert len(read) == 2
    finally:
        root.destroy()


# ── #409 review follow-ups ─────────────────────────────────────────────────

def test_a_sample_moved_into_an_ungrouped_row_is_counted_once(tmp_path):
    h = str(tmp_path)
    _day(h, 'run_1', '2026-11-11', [('A', 's1', 10), ('A', 'x', 20)])
    _day(h, 'run_2', '2026-11-11', [('', 'x', 21)])
    H.rebuild_history(h)
    assert _active(h) == [('run_1', 'A', 's1', '10'), ('run_2', '', 'x', '21')]


@pytest.mark.parametrize('sep', [';', '\t'])
def test_a_comma_in_a_name_is_no_decimal_comma(tmp_path, sep):
    p = tmp_path / 'A_population_stats.csv'
    rows = [['Group', 'Trial', 'Sample', 'Population', 'Count', '%Parent'],
            ['A', '1,2', 'Tube 1,2', 'Q1,2', '10', '12.5']]
    p.write_text(''.join(sep.join(r) + '\n' for r in rows), encoding='utf-8')
    df = H.read_history_table(str(p))
    assert (df.iloc[0]['Trial'], df.iloc[0]['Population']) == ('1,2', 'Q1,2')
    assert df.iloc[0]['%Parent'] == 12.5
    # A value written with a decimal comma still reads as a decimal.
    rows[1][5] = '12,5'
    p.write_text(''.join(sep.join(r) + '\n' for r in rows), encoding='utf-8')
    assert H.read_history_table(str(p)).iloc[0]['%Parent'] == 12.5


@pytest.mark.parametrize('sep', [';', '\t'])
def test_a_decimal_comma_in_the_last_row_only_is_read(tmp_path, sep):
    p = tmp_path / 'A_population_stats.csv'
    rows = [['Group', 'Sample', 'Population', '%Parent'],
            ['A', 'a1', 'Big', '12'], ['A', 'a2', 'Big', '12,5']]
    p.write_text(''.join(sep.join(r) + '\n' for r in rows), encoding='utf-8')
    assert list(H.read_history_table(str(p))['%Parent']) == [12.0, 12.5]

