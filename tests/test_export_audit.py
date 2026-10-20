"""Every export that writes results records it in the audit log: the action,
the file and the samples it draws on, once the file is written, so that a
fix applied later can list it (fixes.stale_exports, the follow-up section of
the fix window).

All data is synthetic: samples s1, s2, ...; channels FL1-A, FL2-A, Time.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from openflo import fixes
from tests.test_result_stamps import _add, _editor

N = 2000


def _inline(widget, work, on_done=None, on_error=None, on_finally=None):
    try:
        res = work()
    except Exception as exc:                      # noqa: BLE001
        if on_error:
            on_error(exc)
    else:
        if on_done:
            on_done(res)
    finally:
        if on_finally:
            on_finally()


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


def _frame(seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FL1-A': rng.uniform(0, 4, N),
                         'FL2-A': rng.normal(2, 1, N),
                         'Time': np.sort(rng.uniform(0, 1000, N))})


@pytest.fixture
def ed(tk_root, tmp_path, monkeypatch):
    """Four gated samples s1..s4 in trials T1 / T2; s4 is not plotted."""
    import openflo.async_task as _at
    monkeypatch.setattr(_at, 'run_async', _inline)
    from openflo import gui
    for fn in ('showinfo', 'showwarning', 'showerror'):
        monkeypatch.setattr(gui.messagebox, fn, lambda *a, **k: None)
    ed = _editor(tk_root)
    for i, name in enumerate(('s1', 's2', 's3', 's4'), 1):
        _add(ed, _frame(i), name=name, path=str(tmp_path / f'{name}.fcs'))
        ed._sample_trial[name] = 'T1' if i % 2 else 'T2'
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.0,
                      'name': 'Pos', 'parent_id': None})
    for t in ('T1', 'T2'):
        if t not in ed._trial_order:
            ed._trial_order.append(t)
    ed._sample_plot_enabled['s4'] = False
    ed._set_active_sample('s2')
    ed._refresh_gate_list()
    yield ed
    ed.destroy()


def _exports(ed):
    return [(e['action'], e['details'].get('path'),
             e['details'].get('samples'))
            for e in ed._audit_log.entries()
            if e['action'] in fixes.EXPORT_ACTIONS]


def _save_as(monkeypatch, module, path):
    monkeypatch.setattr(module.filedialog, 'asksaveasfilename',
                        lambda *a, **k: str(path))


ALL = ['s1', 's2', 's3', 's4']


# ── Editor exports ──────────────────────────────────────────────────────────

def _plot(ed, monkeypatch, tmp_path):
    from openflo import editor_export
    path = tmp_path / 'plot.png'
    _save_as(monkeypatch, editor_export, path)
    ed._save_plot_image()
    return 'plot.export', path, ['s1', 's2', 's3']   # s4 is not plotted


def _gatingml(ed, monkeypatch, tmp_path):
    from openflo import editor_export
    path = tmp_path / 'gates.xml'
    _save_as(monkeypatch, editor_export, path)
    ed._export_gatingml()
    return 'gatingml.export', path, ['s2']           # the active sample


def _wsp(ed, monkeypatch, tmp_path):
    from openflo import editor_export
    path = tmp_path / 'workspace.wsp'
    _save_as(monkeypatch, editor_export, path)
    monkeypatch.setattr(editor_export.messagebox, 'askyesnocancel',
                        lambda *a, **k: True)
    ed._export_flowjo_wsp()
    return 'wsp.export', path, ALL


@pytest.mark.parametrize('export', [_plot, _gatingml, _wsp],
                         ids=['plot', 'gatingml', 'wsp'])
def test_editor_export_records_file_and_samples(ed, monkeypatch, tmp_path,
                                                export):
    action, path, samples = export(ed, monkeypatch, tmp_path)
    assert path.exists()
    assert _exports(ed) == [(action, str(path), samples)]


@pytest.mark.parametrize('all_samples', [False, True], ids=['active', 'all'])
def test_population_fcs_export_records_every_file(ed, monkeypatch, tmp_path,
                                                  all_samples):
    from openflo import editor_populations
    from openflo.async_task import wait_for_job
    out = tmp_path / 'pops'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs(all_samples=all_samples)
    assert wait_for_job(ed._job_strip)
    written = sorted(str(p) for p in out.rglob('*.fcs'))
    assert len(written) == (4 if all_samples else 1)
    [entry] = [e for e in ed._audit_log.entries()
               if e['action'] in fixes.EXPORT_ACTIONS]
    assert entry['action'] == 'population.export_fcs'
    assert entry['details']['path'] == str(out)
    assert sorted(entry['details']['files']) == written
    assert entry['details']['samples'] == (ALL if all_samples else ['s2'])


def test_a_failed_write_records_nothing(ed, monkeypatch, tmp_path):
    from openflo import editor_export
    _save_as(monkeypatch, editor_export, tmp_path / 'no' / 'such' / 'p.png')
    monkeypatch.setattr(editor_export, 'show_error', lambda *a, **k: None)
    ed._save_plot_image()
    assert _exports(ed) == []


# ── Window exports ──────────────────────────────────────────────────────────

def _statistics(kind):
    def run(ed, monkeypatch, tmp_path):
        from openflo import ui_statistics
        w = ui_statistics.StatisticsWindow(ed)
        w.withdraw()
        w._refresh()
        path = tmp_path / f'stats.{kind}'
        _save_as(monkeypatch, ui_statistics, path)
        (w._export_csv if kind == 'csv' else w._export_xlsx)()
        return 'statistics.export', path, ALL
    return run


def _frequency_window(ed):
    from openflo.ui_frequency import FrequencyComparisonWindow
    w = FrequencyComparisonWindow(ed)
    w.withdraw()
    w._collect(w._after_collect)
    w.pop_var.set('Pos')
    w.metric_var.set('%Parent')
    w._rebuild()
    return w


def _frequencies(ed, monkeypatch, tmp_path):
    from openflo import ui_frequency
    w = _frequency_window(ed)
    path = tmp_path / 'tidy.csv'
    _save_as(monkeypatch, ui_frequency, path)
    w._export_tidy()
    return 'frequencies.export', path, ALL


def _diff_abundance(ed, monkeypatch, tmp_path):
    from openflo import ui_diff
    w = _frequency_window(ed)
    w._diff_abundance()
    [win] = [c for c in w.winfo_children()
             if isinstance(c, ui_diff.DiffAbundanceWindow)]
    path = tmp_path / 'diff.csv'
    _save_as(monkeypatch, ui_diff, path)
    win._export()
    return 'diff_abundance.export', path, ALL


def _compare_all(ed, monkeypatch, tmp_path):
    from openflo import ui_diff
    w = _frequency_window(ed)
    w._compare_all()
    [win] = [c for c in w.winfo_children()
             if isinstance(c, ui_diff.CompareAllWindow)]
    path = tmp_path / 'compare_all.csv'
    _save_as(monkeypatch, ui_diff, path)
    win._export_csv()
    return 'compare_all.export', path, ALL


def _kinetics(kind):
    def run(ed, monkeypatch, tmp_path):
        from openflo import tool_window
        from openflo.ui_kinetics import KineticsWindow
        w = KineticsWindow(ed)
        w.withdraw()
        w.chan_var.set(ed._fmt_channel('FL2-A'))
        w._refresh()
        path = tmp_path / f'kinetics_{kind}.csv'
        _save_as(monkeypatch, tool_window, path)
        (w._export_csv if kind == 'table' else w._export_traces)()
        return 'kinetics.export', path, ALL
    return run


def _compare_wsp(ed, monkeypatch, tmp_path):
    from openflo import ui_compare
    d = ui_compare.CompareWspDialog(ed)
    d.withdraw()
    # The rows a comparison run gives (one per sample and population).
    d._rows = [{'sample': s, 'population': 'Pos', 'flowjo_count': 10,
                'openflo_count': 11, 'delta': 1, 'rel_delta': 0.1,
                'total_events': N, 'error': None} for s in ('s1', 's3', 's1')]
    path = tmp_path / 'compare.csv'
    _save_as(monkeypatch, ui_compare, path)
    d._export()
    return 'compare_wsp.export', path, ['s1', 's3']


WINDOW_EXPORTS = {
    'statistics-csv': _statistics('csv'),
    'statistics-xlsx': _statistics('xlsx'),
    'frequencies': _frequencies,
    'diff-abundance': _diff_abundance,
    'compare-all': _compare_all,
    'kinetics-table': _kinetics('table'),
    'kinetics-traces': _kinetics('traces'),
    'compare-wsp': _compare_wsp,
}


@pytest.mark.parametrize('export', list(WINDOW_EXPORTS.values()),
                         ids=list(WINDOW_EXPORTS))
def test_window_export_records_file_and_samples(ed, monkeypatch, tmp_path,
                                                export):
    action, path, samples = export(ed, monkeypatch, tmp_path)
    assert path.exists() and path.stat().st_size
    assert _exports(ed) == [(action, str(path), samples)]


def test_a_statistics_export_made_before_a_fix_is_listed(ed, monkeypatch,
                                                         tmp_path):
    action, path, _ = _statistics('csv')(ed, monkeypatch, tmp_path)
    ed._audit('fix.apply', fixes=['refit-valley'],
              results=[{'sample': 's3', 'result': 'gate', 'key': 'g1'}])
    rows = fixes.stale_exports(ed._audit_log.entries())
    assert [(r['path'], r['action'], r['exists']) for r in rows] == [
        (str(path), action, True)]
    assert rows[0]['samples'] == ALL
