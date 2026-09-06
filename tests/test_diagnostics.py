"""Tests for the install health check (openflo.diagnostics) + its GUI hook."""
import os

import pytest

from openflo import diagnostics as dx


def test_parse_requirement_pin_and_extra():
    assert dx._parse_requirement('numpy==2.4.6') == ('numpy', '2.4.6', False)
    assert dx._parse_requirement('scikit-learn==1.8.0') == \
        ('scikit-learn', '1.8.0', False)
    # a non-'==' specifier has no pinned version
    name, pinned, extra = dx._parse_requirement('anndata>=0.10')
    assert name == 'anndata' and pinned is None and extra is False
    # extra-gated requirement is flagged
    name, pinned, extra = dx._parse_requirement("trimap>=1.1.4; extra == 'embed'")
    assert name == 'trimap' and extra is True


def test_check_dependencies_reports_core_pins():
    rows = dx.check_dependencies()
    assert rows, "no core dependencies parsed from openflo metadata"
    names = {r['name'] for r in rows}
    # core pins that must be present in a working install
    assert {'numpy', 'pandas', 'scipy', 'matplotlib'} <= names
    for r in rows:
        assert r['status'] in ('ok', 'drift', 'missing')
    # extras (gui/embed/interop) are NOT treated as core deps
    assert 'trimap' not in names and 'anndata' not in names


def test_run_diagnostics_quick_is_healthy_on_this_install():
    """The dev environment is a known-good install — --quick (no self-test)
    must report healthy with the expected report shape."""
    report = dx.run_diagnostics(include_selftest=False)
    assert report['selftest'] is None
    assert set(report) >= {'python', 'executable', 'openflo', 'dependencies',
                           'engines', 'warnings', 'ok'}
    missing = [d for d in report['dependencies'] if d['status'] == 'missing']
    assert not missing, f"core deps missing in test env: {missing}"
    assert report['ok'] is True


def test_format_report_mentions_deps_and_verdict():
    report = dx.run_diagnostics(include_selftest=False)
    text = dx.format_report(report)
    assert 'OpenFlo diagnostics' in text
    assert 'numpy' in text
    assert ('HEALTHY' in text) or ('ISSUES FOUND' in text)


def test_main_quick_returns_zero_on_healthy_install(capsys):
    rc = dx.main(['--quick'])
    out = capsys.readouterr().out
    assert rc == 0
    assert 'Core dependencies' in out


def test_main_json_quick_emits_parseable_report(capsys):
    import json
    rc = dx.main(['--json', '--quick'])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert data['ok'] is True and data['selftest'] is None


def test_gui_diagnostics_dialog_renders():
    """Help → Run diagnostics shows its report in a dialog. We render the
    result dialog directly with canned text (no subprocess) so the test is fast
    and display-only."""
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip("tkinter not available")
    try:
        root = tk.Tk()
    except Exception as e:                          # noqa: BLE001
        pytest.skip(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    try:
        # Both healthy and unhealthy branches must build without error.
        ed._show_diagnostics_result("OpenFlo diagnostics\nall good\nHEALTHY", 0)
        ed._show_diagnostics_result("OpenFlo diagnostics\n[FAIL] numpy", 1)
        assert callable(ed._run_diagnostics)
    finally:
        try:
            root.destroy()
        except Exception:
            pass
