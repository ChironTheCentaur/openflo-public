"""Statistics window: "Send to Statistics app…" saves the table as a Pocket
Lab Table, offered only when stats_plt says so (off until the Statistics
app is released), and offers the app's page afterwards.

Generated data: two samples on generic channels in two groups, one gate.
"""
from __future__ import annotations

import tkinter.filedialog
import tkinter.messagebox
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import stats_plt
from openflo.pocketlab_table import read_plt
from tests.conftest import editor_or_skip

COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']


def _window(monkeypatch, enabled, trials=('Day1', 'Day2')):
    monkeypatch.setattr(stats_plt, 'RELEASED', False)
    if enabled:
        monkeypatch.setenv(stats_plt.ENV_FLAG, '1')
    else:
        monkeypatch.delenv(stats_plt.ENV_FLAG, raising=False)
    root, ed = editor_or_skip(accept_confirms=False)
    rng = np.random.default_rng(5)
    for nm, trial in zip(('sample_a', 'sample_b'), trials, strict=True):
        df = pd.DataFrame({c: rng.random(400) * 50000 for c in COLS})
        ed._samples[nm] = SimpleNamespace(
            name=nm, path=f'{nm}.fcs', data=df, fluor_channels=COLS[2:],
            raw=df.copy(), channel_labels={c: c for c in COLS})
        ed._sample_order.append(nm)
        ed._sample_trial[nm] = trial
        ed._sample_plot_enabled[nm] = True
        ed._sample_gates[nm] = {}
        ed._sample_gate_order[nm] = []
        ed._sample_gate_seq[nm] = 0
        ed._active_sample = nm
        ed._gates = ed._sample_gates[nm]
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                      'value': 25000.0, 'name': 'FL1 pos'}, audit=False)
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    from openflo.ui_statistics import StatisticsWindow
    win = StatisticsWindow(ed)
    win.withdraw()
    want = win._selected_stats()
    rows, cols = ed._stats_rows_from_snapshot(ed._stats_snapshot(want), want)
    win._populate(rows, cols, curated=False)
    return root, win


def _buttons(widget):
    out = []
    for w in widget.winfo_children():
        try:
            out.append(str(w.cget('text')))
        except Exception:  # noqa: BLE001, S112 - a widget without text
            pass
        out += _buttons(w)
    return out


def test_hidden_until_released(monkeypatch):
    root, win = _window(monkeypatch, enabled=False)
    try:
        assert "Send to Statistics app…" not in _buttons(win)
        assert "Export CSV…" in _buttons(win)
    finally:
        root.destroy()


def test_saves_a_table_the_statistics_app_reads_then_offers_its_page(monkeypatch, tmp_path):
    root, win = _window(monkeypatch, enabled=True)
    try:
        assert "Send to Statistics app…" in _buttons(win)
        out = tmp_path / 'population_stats.plt.csv'
        asked = {}

        def save(**kw):
            asked.update(kw)
            return str(out)
        monkeypatch.setattr(tkinter.filedialog, 'asksaveasfilename', save)
        answers = [True]
        monkeypatch.setattr(tkinter.messagebox, 'askyesno', lambda *a, **k: answers.pop())
        opened = []
        import webbrowser
        monkeypatch.setattr(webbrowser, 'open', opened.append)
        audits = []
        monkeypatch.setattr(win, '_audit_export', lambda path, kind: audits.append((path, kind)))
        win._send_to_statistics()
        assert audits == [(str(out), 'plt')]
        assert asked['initialfile'] == 'population_stats.plt.csv'
        assert asked['defaultextension'] == '.plt.csv'
        table, notices, _extra = read_plt(out.read_bytes())
        assert notices == []
        names = [c['name'] for c in table['columns']]
        assert names[:3] == ['sample', 'group', 'population']
        k = names.index('group')
        assert {r[0]: r[k] for r in table['data']} == {'sample_a': 'Day1', 'sample_b': 'Day2'}
        # The same rows and headings as Export CSV.
        lines = win._table_lines()
        assert len(table['data']) == len(lines) - 1
        labels = [c['label'] for c in table['columns'] if c['name'] not in ('sample', 'group', 'population')]
        assert labels == [h for h in lines[0] if h not in ('Sample', 'Population')]
        assert opened == [stats_plt.STATISTICS_APP_URL]
        assert 'Statistics app' in win.status_var.get()

        # "No" opens nothing; a cancelled save writes nothing.
        out.unlink()
        answers.append(False)
        win._send_to_statistics()
        assert out.exists() and opened == [stats_plt.STATISTICS_APP_URL]
        out.unlink()
        monkeypatch.setattr(tkinter.filedialog, 'asksaveasfilename', lambda **kw: '')
        win._send_to_statistics()
        assert not out.exists()
    finally:
        root.destroy()


def test_one_group_only_has_no_group_column(monkeypatch, tmp_path):
    root, win = _window(monkeypatch, enabled=True, trials=('Day1', 'Day1'))
    try:
        out = tmp_path / 's.plt.csv'
        monkeypatch.setattr(tkinter.filedialog, 'asksaveasfilename', lambda **kw: str(out))
        monkeypatch.setattr(tkinter.messagebox, 'askyesno', lambda *a, **k: False)
        win._send_to_statistics()
        assert 'group' not in [c['name'] for c in read_plt(out.read_bytes())[0]['columns']]
    finally:
        root.destroy()


def test_nothing_to_send(monkeypatch):
    root, win = _window(monkeypatch, enabled=True)
    try:
        win._rows = []
        monkeypatch.setattr(tkinter.filedialog, 'asksaveasfilename',
                            lambda **kw: pytest.fail('asked for a file with nothing to send'))
        win._send_to_statistics()
        assert win.status_var.get().startswith('Nothing to send')
    finally:
        root.destroy()
