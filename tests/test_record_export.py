"""Results windows record a written export through one helper,
``export_audit.record_export`` (best-practices survey R8).

Nine results windows (panel check, Statistics, kinetics, the two
comparison windows, workspace comparison, AnnData export, FlowSOM tree,
marker expression, trajectory) each had their own copy of the same few
lines; the copies had drifted in how they found the editor and what they
did when it had no audit trail. These tests pin the helper's rule and what
the windows record (tests/test_export_audit.py drives the others end to
end), and an AST scan stops a private copy coming back.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from openflo import errors
from openflo.export_audit import record_export

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src' / 'openflo'


class _Editor:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def _audit(self, action, **details):
        if self.fail:
            raise RuntimeError('trail is broken')
        self.calls.append((action, details))


def test_record_export_appends_action_path_and_details():
    ed = _Editor()
    record_export(ed, 'x.export', '/out/a.csv', kind='csv', samples=['s1'])
    assert ed.calls == [('x.export', {'path': '/out/a.csv', 'kind': 'csv',
                                      'samples': ['s1']})]


@pytest.mark.parametrize('editor', [None, SimpleNamespace()])
def test_record_export_without_an_audit_trail_does_nothing(editor):
    before = errors.noted()
    record_export(editor, 'x.export', '/out/a.csv', kind='csv')
    assert errors.noted() == before


def test_a_failing_audit_trail_is_noted_and_never_raises():
    record_export(_Editor(fail=True), 'x.export', '/out/a.csv')
    assert errors.noted()[-1] == ("[handled] recording the export in the "
                                  "audit trail: RuntimeError: trail is broken")


def test_panel_check_records_its_fluors():
    from openflo.ui_panel_check import PanelCheckWindow
    ed = _Editor()
    win = SimpleNamespace(editor=ed, report={'fluors': ('FL1-A', 'FL2-A')})
    PanelCheckWindow._audit_export(win, '/out/p.csv', 'similarity')
    assert ed.calls == [('panel_check.export',
                         {'path': '/out/p.csv', 'kind': 'similarity',
                          'fluors': ['FL1-A', 'FL2-A']})]
    win.report = None
    PanelCheckWindow._audit_export(win, '/out/p.png', 'figure')
    assert ed.calls[-1][1]['fluors'] == []


def test_statistics_records_rows_and_samples_once_each_in_order():
    from openflo.ui_statistics import StatisticsWindow
    ed = _Editor()
    rows = [{'Sample': 'S2'}, {'Sample': 'S1'}, {'Sample': 'S2'},
            {'Sample': None}, {}]
    win = SimpleNamespace(editor=ed, _rows=rows)
    StatisticsWindow._audit_export(win, '/out/s.xlsx', 'xlsx')
    assert ed.calls == [('statistics.export',
                         {'path': '/out/s.xlsx', 'kind': 'xlsx',
                          'n_rows': 5, 'samples': ['S2', 'S1']})]


def test_kinetics_records_samples_once_each_in_order():
    from openflo.ui_kinetics import KineticsWindow
    ed = _Editor()
    win = SimpleNamespace(editor=ed, _results=[{'Sample': 'B'},
                                               {'Sample': 'A'},
                                               {'Sample': 'B'}])
    KineticsWindow._audit_export(win, '/out/k.csv', 'table')
    assert ed.calls == [('kinetics.export',
                         {'path': '/out/k.csv', 'kind': 'table',
                          'samples': ['B', 'A']})]


def test_comparison_windows_reach_the_editor_through_their_parent():
    from openflo.ui_diff import _audit_export
    ed = _Editor()
    win = SimpleNamespace(master=SimpleNamespace(editor=ed))
    _audit_export(win, 'compare_all.export', '/out/c.csv', ('S1', 'S2'),
                  kind='table')
    _audit_export(win, 'diff_abundance.export', '/out/d.csv', None)
    assert ed.calls == [
        ('compare_all.export', {'path': '/out/c.csv', 'samples': ['S1', 'S2'],
                                'kind': 'table'}),
        ('diff_abundance.export', {'path': '/out/d.csv', 'samples': []})]
    before = errors.noted()
    _audit_export(SimpleNamespace(master=None), 'compare_all.export',
                  '/out/c.csv', ['S1'])
    assert errors.noted() == before


def test_expression_records_marker_and_every_loaded_sample():
    from openflo.ui_expression import MarkerExpressionWindow
    ed = _Editor()
    ed._sample_order = ['S1', 'S2', 'S3']
    ed._samples = {'S1': object(), 'S3': object()}
    win = SimpleNamespace(editor=ed,
                          marker_var=SimpleNamespace(get=lambda: 'FL2-A'))
    win._samples = lambda: MarkerExpressionWindow._samples(win)
    MarkerExpressionWindow._done(win, '/out/e.png', 'figure')
    assert ed.calls == [('expression.export',
                         {'path': '/out/e.png', 'kind': 'figure',
                          'marker': 'FL2-A', 'samples': ['S1', 'S3']})]


def test_trajectory_records_the_samples_it_ran_on():
    from openflo.ui_trajectory import TrajectoryWindow
    ed = _Editor()
    win = SimpleNamespace(editor=ed, _ran_on=['S4', 'S2'])
    TrajectoryWindow._done(win, '/out/t.csv', 'trends')
    assert ed.calls == [('trajectory.export',
                         {'path': '/out/t.csv', 'kind': 'trends',
                          'samples': ['S4', 'S2']})]


def test_flowsom_tree_records_its_sample(monkeypatch):
    from openflo import ui_flowsom_tree
    saved = []
    monkeypatch.setattr(ui_flowsom_tree.filedialog, 'asksaveasfilename',
                        lambda **kw: '/out/tree.png')
    monkeypatch.setattr(ui_flowsom_tree, 'savefig_background',
                        lambda fig, path, background: saved.append(path))
    monkeypatch.setattr(ui_flowsom_tree, 'report_saved',
                        lambda win, path, verb: None)
    ed = _Editor()
    win = SimpleNamespace(editor=ed, sample='S5', _fig=None,
                          bg_var=SimpleNamespace(get=lambda: 'white'))
    ui_flowsom_tree.FlowSOMTreeWindow._export_figure(win)
    assert saved == ['/out/tree.png']
    assert ed.calls == [('flowsom_tree.export',
                         {'path': '/out/tree.png', 'sample': 'S5'})]


def test_no_window_keeps_its_own_copy_of_the_recording():
    """Only export_audit catches a failed ``_audit`` call: any other
    ``try`` whose body calls ``._audit(...)`` and notes the failure is a
    private copy of record_export."""
    copies = []
    for path in sorted(SRC.glob('*.py')):
        if path.name == 'export_audit.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            calls_audit = any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == '_audit'
                for stmt in node.body for n in ast.walk(stmt))
            says_export = any(
                isinstance(n, ast.Constant)
                and n.value == "recording the export in the audit trail"
                for h in node.handlers for n in ast.walk(h))
            if calls_audit and says_export:
                copies.append(f"{path.name}:{node.lineno}")
    assert copies == []


def test_export_audit_is_tk_free():
    tree = ast.parse((SRC / 'export_audit.py').read_text(encoding='utf-8'))
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or '' for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom)}
    assert not {m for m in imported if 'tkinter' in m or m == 'theme'}


# ── The five calls that had their own wording or no guard (R8c) ────────────
#
# Frequencies (_done), the layout editor, population annotation, debarcode
# and spectral QC each recorded their export their own way: two with their
# own note wording, three with no guard, so a broken audit trail raised out
# of the export after the file was written and the window never said it was
# saved. They now go through record_export too. The layout editor and
# debarcode are driven end to end in their own test files.

_EXPORT_NOTE = ("[handled] recording the export in the audit trail: "
                "RuntimeError: trail is broken")


def _reports(monkeypatch, module):
    seen = []
    monkeypatch.setattr(module, 'report_saved',
                        lambda win, path, verb: seen.append((path, verb)))
    return seen


def _frequency_window(ed, samples):
    import pandas as pd
    tidy = (None if samples is None
            else pd.DataFrame({'Sample': samples,
                               'Population': ['P'] * len(samples)}))
    return SimpleNamespace(editor=ed, _tidy=tidy)


def test_frequencies_record_their_samples_once_each_in_order(monkeypatch):
    from openflo import ui_frequency
    seen = _reports(monkeypatch, ui_frequency)
    ed = _Editor()
    win = _frequency_window(ed, ['S2', 'S1', 'S2'])
    ui_frequency.FrequencyPanel._done(
        win, 'frequencies.export', '/out/f.png', kind='figure',
        background='white')
    assert ed.calls == [('frequencies.export',
                         {'path': '/out/f.png', 'kind': 'figure',
                          'background': 'white', 'samples': ['S2', 'S1']})]
    assert seen == [('/out/f.png', 'Exported')]
    ui_frequency.FrequencyPanel._done(
        _frequency_window(ed, None), 'frequencies.export', '/out/g.csv',
        kind='table')
    assert ed.calls[-1] == ('frequencies.export',
                            {'path': '/out/g.csv', 'kind': 'table'})


def test_a_broken_trail_still_reports_a_frequencies_export(monkeypatch):
    from openflo import ui_frequency
    seen = _reports(monkeypatch, ui_frequency)
    win = _frequency_window(_Editor(fail=True), ['S1'])
    ui_frequency.FrequencyPanel._done(
        win, 'frequencies.export', '/out/f.csv', kind='table')
    assert errors.noted()[-1] == _EXPORT_NOTE
    assert seen == [('/out/f.csv', 'Exported')]


def _annotation_window(ed):
    import pandas as pd
    mem = pd.DataFrame({'FL1-A': [1.5, -0.5], 'FL2-A': [0.25, 2.0]},
                       index=pd.Index([0, 1], name='population'))
    return SimpleNamespace(editor=ed, sample='S7', _mem=mem)


@pytest.mark.parametrize('fail', [False, True])
def test_annotation_export_records_its_sample_and_survives_a_broken_trail(
        monkeypatch, tmp_path, fail):
    from openflo import ui_annotation
    seen = _reports(monkeypatch, ui_annotation)
    out = tmp_path / 'mem.csv'
    monkeypatch.setattr(ui_annotation.filedialog, 'asksaveasfilename',
                        lambda **kw: str(out))
    ed = _Editor(fail=fail)
    ui_annotation.PopulationAnnotationWindow._export_csv(
        _annotation_window(ed))
    assert out.read_text(encoding='utf-8').splitlines()[0].endswith(
        'FL1-A,FL2-A')
    assert seen == [(str(out), 'Exported')]
    if fail:
        assert errors.noted()[-1] == _EXPORT_NOTE
    else:
        assert ed.calls == [('annotate.export',
                             {'path': str(out), 'sample': 'S7'})]


def _spectral_window(audit):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    fig = Figure(figsize=(2, 2))
    fig.add_subplot().plot([0, 1], [1, 0])
    return SimpleNamespace(_audit=audit, _markdown=lambda: '# Spectral QC\n',
                           _fig=fig, _bg_var=SimpleNamespace(get=lambda: 'white'))


@pytest.mark.parametrize('trail', ['working', 'broken', 'none'])
def test_spectral_qc_exports_record_through_the_helper(monkeypatch, tmp_path,
                                                        trail):
    from openflo import ui_spectral_qc
    seen = _reports(monkeypatch, ui_spectral_qc)
    ed = _Editor(fail=trail == 'broken')
    win = _spectral_window(None if trail == 'none' else ed._audit)
    md, png = tmp_path / 'qc.md', tmp_path / 'qc.png'
    for path, export in ((md, ui_spectral_qc.SpectralQCWindow._export_md),
                         (png, ui_spectral_qc.SpectralQCWindow._export_figure)):
        monkeypatch.setattr(ui_spectral_qc.filedialog, 'asksaveasfilename',
                            lambda p=path, **kw: str(p))
        before = len(errors.noted())
        export(win)
        assert errors.noted()[before:] == (
            [_EXPORT_NOTE] if trail == 'broken' else [])
    assert md.read_text(encoding='utf-8') == '# Spectral QC\n'
    assert png.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    assert seen == [(str(md), 'Exported'), (str(png), 'Exported')]
    assert ed.calls == ([] if trail != 'working' else [
        ('spectral.qc.export', {'path': str(md)}),
        ('spectral.qc.export', {'path': str(png), 'background': 'white'})])


def test_no_window_records_an_export_without_the_helper():
    """A results window (ui_*.py) never calls ``_audit`` itself with an
    ``...export...`` action, or with a ``path=`` and an action it works out
    at run time (Frequencies' ``_done``): every export goes through
    record_export, so a broken trail is noted the same way everywhere and
    never stops the window saying the file was saved. A named action that is
    not an export (the compliance signature) keeps its own entry."""
    direct = []
    for path in sorted(SRC.glob('ui_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == '_audit'):
                continue
            has_path = any(k.arg == 'path' for k in node.keywords)
            named = bool(node.args) and isinstance(node.args[0], ast.Constant)
            if ((named and 'export' in str(node.args[0].value))
                    or (has_path and not named)):
                direct.append(f"{path.name}:{node.lineno}")
    assert direct == []
