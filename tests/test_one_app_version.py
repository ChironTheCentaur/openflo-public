"""Every place OpenFlo shows or records its own version reports the same one
(B13): ``result_stamp.app_version()``, which is ``update.current_version()``.

In an editable install the package metadata is frozen at ``pip install -e``
time, so after a version bump ``importlib.metadata`` and ``openflo.__version__``
lag the source. Diagnostics, the About window, the status bar, methods text,
manifests and export stamps each used to read one of those on their own, so
they could disagree with the version a saved session records. Each test below
makes the running version differ from the metadata and checks the reader
reports the running one.
"""
from __future__ import annotations

import json
import tkinter as tk
from types import SimpleNamespace

import pandas as pd
import pytest

import tests.test_windows_nothing_hidden as sweep
from openflo import result_stamp

RUNNING = '9.8.7.dev42'   # what current_version() says; not the metadata's


@pytest.fixture
def running_version(monkeypatch):
    """The running version differs from the installed metadata."""
    from openflo import __version__
    assert __version__ != RUNNING
    monkeypatch.setattr('openflo.update.current_version', lambda: RUNNING)
    result_stamp.app_version.cache_clear()
    yield RUNNING
    result_stamp.app_version.cache_clear()


def test_diagnostics_and_figure_stamp(running_version):
    from openflo import capabilities
    from openflo.diagnostics import run_diagnostics
    assert capabilities.openflo_version() == RUNNING
    assert capabilities.build_provenance().startswith(f"OpenFlo {RUNNING}")
    assert run_diagnostics(include_selftest=False)['openflo'] == RUNNING


def test_export_stamp(running_version):
    from openflo import export_stamp
    assert export_stamp.version() == RUNNING


def test_provenance_methods_and_manifest(running_version):
    from openflo import provenance
    assert provenance.OPENFLO_VERSION == RUNNING
    assert f"v{RUNNING}" in provenance.methods_paragraph([])
    assert provenance.run_manifest()['openflo_version'] == RUNNING


def test_normalization_methods_text(running_version):
    from openflo import batch_normalize, bead_normalize
    model = SimpleNamespace(batches=['b1', 'b2'], channels=['FL1-A'],
                            mode='goal', n_metaclusters=1, n_quantiles=11)
    assert f"OpenFlo {RUNNING}." in batch_normalize.methods_text(
        model, pd.DataFrame())
    records = {'s1': {'bead_channels': ['FL1-A'], 'n_beads': 100,
                      'knots_f': [0.0, 1.0], 'window': 5}}
    assert f"OpenFlo {RUNNING}:" in bead_normalize.methods_text(records)


def test_cli_run_manifest(running_version, tmp_path):
    from openflo import cli_manifest
    args = SimpleNamespace(out=str(tmp_path), gates_from=None, panel=None)
    path = cli_manifest.write_run_manifest(
        args, [], None, None, None, 0,
        pd.Timestamp.now(tz='UTC').to_pydatetime())
    with open(path, encoding='utf-8') as f:
        assert json.load(f)['openflo'] == RUNNING


# ── GUI: About window, status bar, audit-trail export and signed record ────

def _all_text(w, out):
    try:
        out.append(str(w.cget('text')))
    except tk.TclError:
        pass
    for c in w.winfo_children():
        _all_text(c, out)
    return out


no_dialogs = sweep.no_dialogs
engines_known = sweep.engines_known


@pytest.fixture
def editor(running_version, no_dialogs, engines_known):
    root = sweep._root(1.0)
    try:
        yield root, sweep._editor(root)
    finally:
        root.destroy()


def test_about_window_and_status_bar(editor):
    from openflo.editor_help import AboutWindow
    root, ed = editor
    assert f"v{RUNNING}" in _all_text(ed, [])
    win = AboutWindow(ed)
    root.update()
    assert win.version_label.cget('text') == f"Version {RUNNING}"
    win.destroy()


def test_audit_export_and_signed_record(editor, monkeypatch, tmp_path):
    from openflo.ui_audit import AuditWindow
    root, ed = editor
    win = AuditWindow(ed, ed._audit_log)
    root.update()
    assert win._meta()['openflo_version'] == RUNNING
    answers = iter(['Synthetic Signer', 'Reviewed'])
    monkeypatch.setattr('tkinter.simpledialog.askstring',
                        lambda *a, **k: next(answers))
    out = tmp_path / 'record.json'
    monkeypatch.setattr('tkinter.filedialog.asksaveasfilename',
                        lambda *a, **k: str(out))
    monkeypatch.setattr('openflo.ui_audit.report_saved', lambda *a, **k: None)
    win._sign_record()
    with open(out, encoding='utf-8') as f:
        assert json.load(f)['software_version'] == RUNNING
    win.destroy()
