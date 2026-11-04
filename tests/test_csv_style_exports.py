"""Every window's Export CSV follows the CSV style preference.

In the standard style (the default) each export writes exactly the bytes it
always wrote: comma, decimal point, UTF-8 without a byte-order mark, the
window's own missing-value text and line endings. In the system style it
writes this computer's separator and decimal mark (pinned here to ';' and ',',
a continental-European Excel) with a UTF-8 byte-order mark, the decimal mark
changed in numbers only, never in text.

All data is synthetic: samples sample_1, sample_2, ...; channels FL1-A,
FL2-A, ...; fluors Fluor_1, Fluor_2, ...
"""
from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd
import pytest

from openflo import table_export as te

BOM = b'\xef\xbb\xbf'


@pytest.fixture
def system_style(monkeypatch):
    """The system style on a computer whose Excel wants ';' and ','."""
    monkeypatch.setattr(te, 'system_csv_format', lambda: (';', ','))

    def use(style):
        monkeypatch.setattr(te, 'csv_style', lambda: style)
    return use


# A table with everything a window writes: floats (python and numpy), ints,
# NaN, None, booleans, numbers already formatted, and text holding both
# separators and a point.
ROWS = [
    ['Sample', 'FL1-A mean', 'FL2-A, %', 'note; text', 'count'],
    ['sample_1', 1.5, np.float64(0.125), 'a,b;c', 3],
    ['sample_2', float('nan'), np.float32(2.5), 'v1.2', np.int64(7)],
    ['sample_3', None, -1e-05, '', True],
    ['sample_4', te.NumberText('12.75'), float('inf'), 'NA', 0],
]


def _plain(rows, **kw):
    buf = io.StringIO()
    csv.writer(buf, **kw).writerows(rows)
    return buf.getvalue()


def _styled(rows, style, **kw):
    buf = io.StringIO()
    te.StyledCsvWriter(buf, style, **kw).writerows(rows)
    return buf.getvalue()


# ── The helpers ─────────────────────────────────────────────────────────────

def test_standard_writer_matches_csv_writer_byte_for_byte(system_style):
    system_style('standard')
    assert _styled(ROWS, 'standard') == _plain(ROWS)
    assert _styled(ROWS, None) == _plain(ROWS)               # the preference
    assert _styled(ROWS, 'standard', lineterminator='\n') == \
        _plain(ROWS, lineterminator='\n')


def test_standard_dict_writer_matches_dictwriter(system_style):
    system_style('standard')
    fields = ROWS[0]
    dicts = [dict(zip(fields, r, strict=True)) for r in ROWS[1:]]
    want = io.StringIO()
    w = csv.DictWriter(want, fieldnames=fields)
    w.writeheader()
    w.writerows(dicts)
    got = io.StringIO()
    s = te.StyledCsvWriter(got, None, fieldnames=fields)
    s.writeheader()
    s.writerows(dicts)
    assert got.getvalue() == want.getvalue()


def test_system_writer_changes_separator_and_numbers_only(system_style):
    out = _styled(ROWS, 'system')
    # A cell holding ';' is quoted, one holding ',' no longer needs to be.
    assert out.splitlines()[:2] == [
        'Sample;FL1-A mean;FL2-A, %;"note; text";count',
        'sample_1;1,5;0,125;"a,b;c";3']
    back = list(csv.reader(io.StringIO(out), delimiter=';'))
    assert back == [
        ['Sample', 'FL1-A mean', 'FL2-A, %', 'note; text', 'count'],
        ['sample_1', '1,5', '0,125', 'a,b;c', '3'],
        # NaN as csv.writer writes it; text with a point untouched.
        ['sample_2', 'nan', '2,5', 'v1.2', '7'],
        ['sample_3', '', '-1e-05', '', 'True'],
        ['sample_4', '12,75', 'inf', 'NA', '0']]
    assert out.endswith('\r\n')


def test_system_dict_writer(system_style):
    buf = io.StringIO()
    w = te.StyledCsvWriter(buf, 'system', fieldnames=['Sample', 'x'])
    w.writeheader()
    w.writerow({'Sample': 'sample_1;b', 'x': 0.5})
    assert buf.getvalue() == 'Sample;x\r\n"sample_1;b";0,5\r\n'


def test_open_and_write_styled_rows_encoding(system_style, tmp_path):
    p = tmp_path / 'std.csv'
    te.write_styled_rows(str(p), ROWS, style='standard')
    assert p.read_bytes() == _plain(ROWS).encode('utf-8')
    p = tmp_path / 'sys.csv'
    te.write_styled_rows(str(p), ROWS, style='system')
    raw = p.read_bytes()
    assert raw.startswith(BOM)
    assert raw[len(BOM):].decode('utf-8') == _styled(ROWS, 'system')
    assert te.csv_encoding('standard') == 'utf-8'
    assert te.csv_encoding('system') == 'utf-8-sig'


def _frame():
    index = pd.Index(['r1', 'r2', 'r3', 'r4'], name='id')
    # A float column with text in it: pandas keeps it as objects.
    mixed = pd.Series([0.25, None, 'n/a', 1.5], dtype=object, index=index)
    return pd.DataFrame({'Sample': ['sample_1', 'a,b', 'c;d', 'v1.2'],
                         'FL1-A': [1.5, np.nan, 2.25, -0.5],
                         'count': [1, 2, 3, 4],
                         'mixed': mixed}, index=index)


@pytest.mark.parametrize('index', [False, True])
def test_standard_frame_matches_to_csv(system_style, tmp_path, index):
    system_style('standard')
    df = _frame()
    want, got = tmp_path / 'want.csv', tmp_path / 'got.csv'
    df.to_csv(want, index=index)
    te.frame_to_csv(df, str(got), index=index)
    assert got.read_bytes() == want.read_bytes()


def test_system_frame(system_style, tmp_path):
    df = _frame()
    p = tmp_path / 'sys.csv'
    te.frame_to_csv(df, str(p), style='system', index=False)
    raw = p.read_bytes()
    assert raw.startswith(BOM)
    back = list(csv.reader(io.StringIO(raw[len(BOM):].decode('utf-8')),
                           delimiter=';'))
    assert back == [['Sample', 'FL1-A', 'count', 'mixed'],
                    ['sample_1', '1,5', '1', '0,25'],
                    ['a,b', '', '2', ''],
                    ['c;d', '2,25', '3', 'n/a'],
                    ['v1.2', '-0,5', '4', '1,5']]
    assert df['mixed'].iloc[0] == 0.25                  # the caller's frame kept


def test_restyle_csv_text(system_style):
    text = _plain([['seq', 'details'], [1, 'gain=1.5; k=v']])
    assert te.restyle_csv_text(text, 'standard') is text
    assert te.restyle_csv_text(text, 'system') == \
        'seq;details\r\n1;"gain=1.5; k=v"\r\n'


# ── The windows ─────────────────────────────────────────────────────────────

@pytest.fixture
def tk_root(tmp_path, monkeypatch):
    from tests.test_transform_records import capture_warnings, isolate_home
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        from tests.conftest import gui_unavailable
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def ed(tk_root, tmp_path, monkeypatch):
    """An editor holding two samples, sample_1 and sample_2."""
    import tkinter.filedialog
    import tkinter.messagebox

    from tests.test_result_stamps import _add, _editor
    for fn in ('showinfo', 'showwarning', 'showerror'):
        monkeypatch.setattr(tkinter.messagebox, fn, lambda *a, **k: None)
    # A save dialog that is not answered would wait for a person.
    monkeypatch.setattr(tkinter.filedialog, 'asksaveasfilename',
                        lambda *a, **k: pytest.fail('unanswered save dialog'))
    ed = _editor(tk_root)
    rng = np.random.default_rng(0)
    for i in (1, 2):
        df = pd.DataFrame({'FL1-A': rng.uniform(0, 4, 300),
                           'FL2-A': rng.normal(2, 1, 300),
                           'cluster': np.arange(300) % 3})
        _add(ed, df, name=f'sample_{i}',
             path=str(tmp_path / f'sample_{i}.fcs'))
    yield ed
    ed.destroy()


def _withdrawn(win):
    win.withdraw()
    return win


def _table(ed):
    from openflo.ui_table import TableWindow
    win = _withdrawn(TableWindow(
        ed, columns=['Sample', 'FL1-A', 'flag'],
        rows=[{'Sample': 'sample_1', 'FL1-A': 1.25, 'flag': True},
              {'Sample': 'sample_2', 'FL1-A': float('nan'), 'flag': None},
              {'Sample': 'a;b', 'FL1-A': 1000.5, 'flag': False}]))
    return win._export_csv, lambda: _plain(win.table_lines())


def _saturation(ed):
    from openflo.ui_saturation import SaturationReportWindow
    win = _withdrawn(SaturationReportWindow(ed))
    win._cols = ['Sample', 'FL1-A', 'FL2-A']
    win._rows = [{'Sample': 'sample_1', 'FL1-A': 0.5, 'FL2-A': 12},
                 {'Sample': 'sample_2', 'FL1-A': float('nan'),
                  'FL2-A': 2.75}]
    return win._export_csv, lambda: _plain(win._table_lines())


def _kinetics_rows():
    return [{'Sample': f'sample_{i}', 'baseline': 1.5 * i, 'peak': 4.25,
             'time_to_peak': 30.5, 'auc': float('nan'), 'fold': 2.0,
             'n_events': 300, 'centres': np.array([0.5, 1.5]),
             'trace': np.array([1.25, np.nan]), 'counts': np.array([3, 4]),
             'stimulus': 10.0} for i in (1, 2)]


def _kinetics(kind):
    def build(ed):
        from openflo.ui_kinetics import KineticsWindow
        win = _withdrawn(KineticsWindow(ed))
        win._results = _kinetics_rows()
        win._unit = 's'
        if kind == 'table':
            return win._export_csv, lambda: _plain(win._table_lines())

        def legacy():
            rows = [['Sample', 'Time (s)', win.stat_var.get(), 'Events']]
            for r in win._results:
                for t, v, k in zip(r['centres'], r['trace'], r['counts'],
                                   strict=True):
                    rows.append([r['Sample'], float(t),
                                 float(v) if np.isfinite(v) else 'NA',
                                 int(k)])
            return _plain(rows)
        return win._export_traces, legacy
    return build


def _comp_spread(ed):
    from openflo.ui_comp_spread import SpreadMatrixWindow
    win = _withdrawn(SpreadMatrixWindow(ed))
    win.channels = ['FL1-A', 'FL2-A']
    win.ssm = np.zeros((2, 2))
    mat = np.array([[0.0, 0.4375], [np.nan, 1.5]])
    win.display = lambda: mat

    def legacy():
        rows = [['stain', 'FL1-A', 'FL2-A']]
        for i, ch in enumerate(win.channels):
            rows.append([ch] + ['NA' if not np.isfinite(v) else f"{v:.6g}"
                                for v in mat[i]])
        return _plain(rows)
    return win._export_csv, legacy


def _proliferation(ed):
    from openflo.ui_proliferation import ProliferationWindow
    win = _withdrawn(ProliferationWindow(ed))
    win.sample = 'sample_1'
    win.result = {'ok': True, 'counts': [1500, 750],
                  'precursor_frequency': [0.6667, 0.1667],
                  'pct_divided': 33.25, 'division_index': 0.5,
                  'proliferation_index': 1.5, 'expansion_index': 1.125,
                  'undivided': 2048.0, 'spacing': None}

    def legacy():
        res = win.result
        rows = [['generation', 'cells', 'precursor_frequency']]
        for k, (c, p) in enumerate(zip(res['counts'],
                                       res['precursor_frequency'],
                                       strict=True)):
            rows.append([k, c, f"{p:.6g}"])
        rows.append([])
        for key in ('pct_divided', 'division_index', 'proliferation_index',
                    'expansion_index', 'undivided', 'spacing'):
            v = res.get(key)
            rows.append([key, 'NA' if v is None or not np.isfinite(v)
                         else f"{v:.6g}"])
        return _plain(rows)
    return win._export_csv, legacy


def _plate(ed):
    from openflo.ui_plate import PlateWindow
    win = _withdrawn(PlateWindow(ed))
    cells = [((0, 0), 'sample_1', 1.25), ((0, 1), 'sample_2', None),
             ((1, 0), 'a;b', 300.0)]
    win.cells = lambda: cells

    def legacy():
        rows = [['well', 'sample', win.stat_var.get()]]
        rows += [[w, n, 'NA' if v is None else f"{v:g}"]
                 for w, n, v in win.rows()]
        return _plain(rows)
    return win._export_csv, legacy


def _compare_wsp(ed):
    from openflo.ui_compare import CompareWspDialog
    win = _withdrawn(CompareWspDialog(ed))
    fields = ["sample", "population", "path", "flowjo_count",
              "openflo_count", "delta", "rel_delta", "total_events", "error"]
    win._rows = [{'sample': f'sample_{i}', 'population': 'Pos',
                  'path': 'Pos', 'flowjo_count': 10, 'openflo_count': 11,
                  'delta': 1, 'rel_delta': 0.1 * i, 'total_events': 300,
                  'error': None if i == 1 else 'x, y'} for i in (1, 2)]

    def legacy():
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fields)
        w.writeheader()
        for r in win._rows:
            w.writerow({k: "" if r.get(k) is None else r.get(k)
                        for k in fields})
        return buf.getvalue()
    return win._export_csv, legacy


def _panel_matrix(ed):
    from openflo import panel_check as pc
    from openflo.ui_panel_check import PanelCheckWindow
    win = _withdrawn(PanelCheckWindow(ed))
    S = np.array([[1.0, 0.5, 0.1], [0.75, 1.0, 0.0]])
    U = S / np.linalg.norm(S, axis=1, keepdims=True)
    win.report = {'fluors': ['Fluor_1', 'Fluor_2'], 'similarity': U @ U.T,
                  'spillover': np.array([[1.0, 0.5], [np.nan, 1.0]]),
                  'signatures': S, 'detectors': ['Det_1', 'Det_2', 'Det_3']}

    def legacy():
        rows = pc.matrix_csv_rows(win.report, 'similarity')
        return _plain([[str(c) for c in r] for r in rows])
    return lambda: win._export_matrix('similarity'), legacy


def _polyfunction(ed):
    from openflo.ui_polyfunction import PolyfunctionWindow
    win = _withdrawn(PolyfunctionWindow(ed))
    win.rows = [{'Sample': f'sample_{i}', 'Group': 'grp_1',
                 'Combination': 'FL1-A+ FL2-A-', 'Positives': 1,
                 'Count': 30 * i, 'Parent count': 300,
                 '% of parent': 10.0 * i + 0.125} for i in (1, 2)]
    win.rows.append({'Sample': 'sample_3', 'Group': 'grp_1',
                     'Combination': 'FL1-A- FL2-A-', 'Positives': 0,
                     'Count': float('nan'), 'Parent count': float('nan'),
                     '% of parent': float('nan')})

    def legacy():
        cols = ['Sample', 'Group', 'Combination', 'Positives', 'Count',
                'Parent count', '% of parent']
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        for r in win.rows:
            out = dict(r)
            for k in ('Count', 'Parent count', '% of parent'):
                v = out[k]
                if isinstance(v, float) and v != v:
                    out[k] = 'NA'
                elif k == '% of parent':
                    out[k] = f'{v:.4f}'
            w.writerow(out)
        return buf.getvalue()
    return win.export_table, legacy


def _frame_legacy(make, **kw):
    def legacy():
        buf = io.StringIO()
        make().to_csv(buf, **kw)
        return buf.getvalue()
    return legacy


def _diff_compare_all(ed):
    from openflo.stats import compare_all_features
    from openflo.ui_diff import CompareAllWindow
    res = compare_all_features(
        {'Pos': {'grp_1': [1.5, 2.25, 3.0], 'grp_2': [6.5, 8.25, 9.0]},
         'Neg': {'grp_1': [5.0, 5.5, 4.75], 'grp_2': [5.25, 5.0, 5.5]}})
    win = _withdrawn(CompareAllWindow(ed, res, ['grp_1', 'grp_2'],
                                      '%Parent'))
    return win._export_csv, _frame_legacy(
        lambda: pd.DataFrame(win._flat_rows()), index=False)


def _diff_abundance(ed):
    from openflo.ui_diff import DiffAbundanceWindow
    rows = [{'cluster': f'pop_{i}', 'log2fc': 0.5 * i, 'prop_a': 0.25,
             'prop_b': 0.125 * i, 'n_a': 2, 'n_b': 2, 'n_missing': 0,
             'p': 0.03125, 'p_adj': float('nan')} for i in (1, 2)]
    win = _withdrawn(DiffAbundanceWindow(ed, rows, ['grp_1', 'grp_2']))
    return win._export_csv, _frame_legacy(lambda: pd.DataFrame(rows),
                                      index=False)


def _sample_qc(ed):
    from openflo.ui_sample_qc import SampleQCWindow
    win = _withdrawn(SampleQCWindow(ed))
    win._names = ['sample_1', 'sample_2']
    win._D = np.array([[0.0, 0.375], [0.375, 0.0]])
    return win._export_csv, _frame_legacy(lambda: pd.DataFrame(
        win._D, index=pd.Index(win._names), columns=pd.Index(win._names)))


def _trajectory(ed):
    from openflo.ui_trajectory import TrajectoryWindow
    win = _withdrawn(TrajectoryWindow(ed))
    win._means = win._means_linear = np.array([[1.5, np.nan], [2.25, 0.5]])
    win._centers = np.array([0.25, 0.75])
    win._channels = ['FL1-A', 'FL2-A']
    return (lambda: win._export('tidy'),
            _frame_legacy(win._trends_frame, index=False))


def _expression(ed):
    from openflo.stats import to_prism_column
    from openflo.ui_expression import MarkerExpressionWindow
    win = _withdrawn(MarkerExpressionWindow(ed))
    win._medians = {'grp_1': [1.5, 2.25], 'grp_2': [3.125]}
    return win._export_prism, _frame_legacy(
        lambda: to_prism_column(win._medians), index=False)


def _annotation(ed):
    from openflo.ui_annotation import PopulationAnnotationWindow
    win = _withdrawn(PopulationAnnotationWindow(ed, 'sample_1'))
    win._mem = pd.DataFrame({'FL1-A': [1.5, -0.25], 'FL2-A': [np.nan, 3.0]},
                            index=pd.Index(['0', '1'], name='population'))
    return win._export_csv, _frame_legacy(lambda: win._mem)


def _audit(ed):
    from openflo.ui_audit import AuditWindow
    ed._audit('test.step', sample='sample_1', gain=1.5, note='a, b')
    win = _withdrawn(AuditWindow(ed, ed._audit_log))
    return lambda: win._export('csv'), win._log.to_csv


EXPORTS = {
    'table': _table,
    'saturation': _saturation,
    'kinetics-table': _kinetics('table'),
    'kinetics-traces': _kinetics('traces'),
    'comp-spread': _comp_spread,
    'proliferation': _proliferation,
    'plate': _plate,
    'compare-wsp': _compare_wsp,
    'panel-check-matrix': _panel_matrix,
    'polyfunction': _polyfunction,
    'diff-compare-all': _diff_compare_all,
    'diff-abundance': _diff_abundance,
    'sample-qc': _sample_qc,
    'trajectory': _trajectory,
    'expression-prism': _expression,
    'annotation': _annotation,
    'audit': _audit,
}


def _save_to(monkeypatch, path):
    import tkinter.filedialog
    monkeypatch.setattr(tkinter.filedialog, 'asksaveasfilename',
                        lambda *a, **k: str(path))


def _is_decimal(cell):
    try:
        float(cell)
    except ValueError:
        return False
    return '.' in cell


@pytest.mark.parametrize('build', list(EXPORTS.values()), ids=list(EXPORTS))
def test_window_export_follows_the_csv_style(ed, monkeypatch, tmp_path,
                                             system_style, build):
    export, legacy = build(ed)
    system_style('standard')
    std = tmp_path / 'standard.csv'
    _save_to(monkeypatch, std)
    export()
    raw_std = std.read_bytes()
    # The standard style: what the window wrote before the preference.
    assert raw_std == legacy().encode('utf-8')
    assert not raw_std.startswith(BOM)

    system_style('system')
    sys_ = tmp_path / 'system.csv'
    _save_to(monkeypatch, sys_)
    export()
    raw_sys = sys_.read_bytes()
    assert raw_sys.startswith(BOM)
    want = list(csv.reader(io.StringIO(raw_std.decode('utf-8'))))
    got = list(csv.reader(io.StringIO(raw_sys[len(BOM):].decode('utf-8')),
                          delimiter=';'))
    # The same cells, numbers with a decimal comma, text as it was.
    assert got == [[c.replace('.', ',') if _is_decimal(c) else c for c in r]
                   for r in want]
    if build is not _audit:              # its cells are all text or integers
        assert any(_is_decimal(c) for r in want for c in r)
    assert raw_sys.count(b';') >= len(want)


def test_panel_signatures_stay_readable_in_the_system_style(
        ed, monkeypatch, tmp_path, system_style):
    """Save signatures writes the file Load spectra CSV reads back, so it
    stays comma and point whatever the style."""
    from openflo import panel_check as pc
    _export, _legacy = _panel_matrix(ed)
    [win] = [w for w in ed.winfo_children()
             if type(w).__name__ == 'PanelCheckWindow']
    system_style('system')
    path = tmp_path / 'spectra.csv'
    _save_to(monkeypatch, path)
    win._save_signatures()
    r = win.report
    assert path.read_bytes() == _plain(pc.spectra_csv_rows(
        r['signatures'], r['fluors'], r['detectors'])).encode('utf-8')
    S, fluors, dets = pc.load_spectra_csv(str(path))
    assert fluors == ['Fluor_1', 'Fluor_2'] and dets == r['detectors']
    np.testing.assert_allclose(S, r['signatures'], atol=1e-6)


def test_audit_csv_keeps_single_line_endings_on_windows(
        ed, monkeypatch, tmp_path, system_style):
    """Windows text mode turns each '\\n' into '\\r\\n'; the audit CSV text
    already ends its lines with '\\r\\n', so without newline='' every line
    ended '\\r\\r\\n' and Excel showed a blank row between entries."""
    import openflo.ui_audit as ui_audit

    def windows_open(file, mode='r', *args, newline=None, **kw):
        # Text mode as on Windows: newline=None writes '\n' as '\r\n'.
        return open(file, mode, *args,
                    newline='\r\n' if newline is None else newline, **kw)
    monkeypatch.setattr(ui_audit, 'open', windows_open, raising=False)
    export, legacy = _audit(ed)
    for style in ('standard', 'system'):
        system_style(style)
        path = tmp_path / f'{style}.csv'
        _save_to(monkeypatch, path)
        export()
        raw = path.read_bytes()
        assert b'\r\r\n' not in raw
        assert raw.count(b'\r\n') == legacy().count('\r\n')


def test_system_frame_styles_numpy_floats_in_object_columns(
        tmp_path, system_style):
    """A float32 in a mixed column gets the decimal comma, as StyledCsvWriter
    gives it."""
    from openflo.table_export import frame_to_csv
    system_style('system')
    df = pd.DataFrame({'mixed': pd.Series([np.float32(1.5), 'x', None],
                                          dtype=object)})
    path = tmp_path / 'f.csv'
    frame_to_csv(df, str(path), index=False)
    rows = list(csv.reader(io.StringIO(
        path.read_bytes()[len(BOM):].decode('utf-8')), delimiter=';'))
    assert rows == [['mixed'], ['1,5'], ['x'], ['']]
