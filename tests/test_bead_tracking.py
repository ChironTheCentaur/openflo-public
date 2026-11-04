"""Tools > Bead tracking keeps a Levey-Jennings log judged by the Westgard
rules.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import numpy as np
import pytest

import tests._sample_editor as shared
from openflo import bead_tracking as bt
from tests._sample_editor import _set

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


@pytest.mark.parametrize('values, last', [
    ([100, 131], ['1-3s', '1-2s']),
    ([100, 121, 122], ['2-2s', '1-2s']),
    ([100, 121, 78], ['R-4s', '1-2s']),
    ([100, 111, 112, 111, 113], ['4-1s']),
    ([100] + [101] * 10, ['10x']),
    ([100, 121, 99], []),
    ([100, 121, 90, 122], ['1-2s']),
    ([100, 111, 112, 95, 113], []),
    ([100] + [101] * 8 + [99, 101], []),
    ([100, 111, 112, 89, 113], []),
    ([100, 121, 165], ['1-3s', '2-2s', '1-2s']),
])
def test_westgard_rules(values, last):
    assert bt.westgard(values, 100.0, 10.0)[-1] == last


def test_westgard_without_a_baseline_judges_nothing():
    m, sd = bt.baseline([100.0])
    assert np.isnan(sd)
    assert bt.westgard([100, 500], m, sd) == [[], []]
    m, sd = bt.baseline([5, 5, 5])
    assert np.isnan(sd)
    v = [100, np.nan, 131]
    assert bt.westgard(v, 100, 10) == [[], [], ['1-3s', '1-2s']]


def test_bead_log_baseline_status_and_csv_round_trip():
    log = bt.BeadLog(baseline_runs=4)
    for i, x in enumerate([100, 102, 98, 100, 101, 140]):
        log.add(f'2026-01-0{i + 1}', f'beads_{i}', {'FL1-A': x, 'FL2-A': 50})
    log.add('2026-01-07', 'beads_6', {'FL1-A': 99})
    j = log.judge('FL1-A')
    assert j['mean'] == pytest.approx(100.0)
    assert j['sd'] == pytest.approx(np.std([100, 102, 98, 100], ddof=1))
    assert j['status'][5] == 'reject' and j['status'][4] == 'ok'
    assert log.judge('FL2-A')['status'][-1] == 'NA'
    text = log.to_csv()
    assert text.splitlines()[0] == 'date,sample,FL1-A,FL2-A'
    assert text.splitlines()[-1].endswith(',NA')
    back = bt.BeadLog.from_csv(text, baseline_runs=4)
    assert back.runs == log.runs
    with pytest.raises(ValueError):
        bt.BeadLog.from_csv("when,what\n1,2\n")


def test_bead_tracking_window_adds_runs_and_judges(editor, tmp_path,
                                                   monkeypatch):
    from openflo import ui_bead_tracking as ub
    root, ed = editor
    win = ub.BeadTrackingWindow(ed)
    root.update()
    win.set_populations(['All events'])
    for i, level in enumerate([1000, 1010, 990, 1000, 1300]):
        _set(ed, 'alpha_1', **{'FL1-A': np.full(400, float(level)),
                               'FL2-A': np.full(400, 50.0 + i)})
        ed._samples['alpha_1'].metadata = {'$DATE': f'0{i + 1}-JAN-2026'}
        win.add_active()
    win.baseline_var.set(4)
    win.channel_var.set('FL1-A')
    win.draw()
    j = win.judged()
    assert j['mean'] == pytest.approx(1000.0)
    assert j['status'][-1] == 'reject'
    rows = win.table.get_children()
    assert len(rows) == 5
    assert '1-3s' in win.table.item(rows[-1])['values'][2]
    assert win.table.item(rows[0])['values'][0] == '01-JAN-2026'
    path = tmp_path / 'beads.csv'
    monkeypatch.setattr(ub.filedialog, 'asksaveasfilename',
                        lambda **k: str(path))
    win.save_log()
    back = bt.BeadLog.from_csv(path.read_text())
    assert [r['values']['FL1-A'] for r in back.runs] == \
        [1000, 1010, 990, 1000, 1300]
    monkeypatch.setattr(ub.filedialog, 'askopenfilename',
                        lambda **k: str(path))
    win2 = ub.BeadTrackingWindow(ed)
    win2.open_log()
    assert len(win2.log.runs) == 5 and win2.channel_var.get() == 'FL1-A'
    win.set_populations(['no such gate'])
    win.add_active()
    assert 'Pick the beads' in win.status_var.get()
    assert len(win.log.runs) == 5
    win.destroy()
    win2.destroy()
