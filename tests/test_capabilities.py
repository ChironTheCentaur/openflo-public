"""Tests for the optional-engine probe + export provenance stamp."""
from __future__ import annotations

from openflo import capabilities as cap


def test_probe_returns_all_engines_with_shape():
    caps = cap.probe_capabilities()
    keys = {c['key'] for c in caps}
    # core + optional engines are all represented
    assert {'flowio', 'umap', 'leidenalg', 'trimap', 'pacmap', 'phate',
            'anndata', 'tkinterdnd2'} <= keys
    for c in caps:
        assert set(c) == {'key', 'label', 'powers', 'extra', 'available',
                          'version', 'problem', 'checked'}
        assert isinstance(c['available'], bool)
        assert c['checked']                       # load=True: final answers
        # Usable engines report a version and no problem; an unusable one
        # says why (it may still have a version: installed, fails to load).
        if c['available']:
            assert c['version'] != '' and c['problem'] == ''
        else:
            assert c['problem'] != ''


def test_quick_probe_does_not_import_heavy_engines():
    """Regression: the quick probe Help → Environment opens with must not
    import the engines — importing umap/phate on the Tk thread froze it. The
    import check runs afterwards, in another process. Checked in a clean
    subprocess so earlier test imports don't mask it."""
    import subprocess
    import sys
    code = (
        "import sys, openflo.capabilities as c; c.probe_capabilities("
        "load=False);"
        "heavy=[m for m in ('umap','phate','trimap','pacmap') "
        "if m in sys.modules];"
        "assert not heavy, heavy")
    r = subprocess.run([sys.executable, '-c', code],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_full_probe_imports_in_another_process():
    """The import check itself never imports an engine into this process."""
    import subprocess
    import sys
    code = (
        "import sys, openflo.capabilities as c; c.probe_capabilities();"
        "heavy=[m for m in ('umap','phate','trimap','pacmap') "
        "if m in sys.modules];"
        "assert not heavy, heavy")
    r = subprocess.run([sys.executable, '-c', code],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr


def test_install_hint():
    assert cap.install_hint('embed') == "pip install 'openflo[embed]'"
    assert cap.install_hint('interop') == "pip install 'openflo[interop]'"
    assert 'openflo' in cap.install_hint('')          # core → plain reinstall


def test_build_provenance_includes_version():
    stamp = cap.build_provenance()
    assert stamp.startswith('OpenFlo ')
    assert cap.openflo_version() in stamp
    # extra is appended when given
    assert 'run42' in cap.build_provenance(extra='run42')


def test_known_core_engine_resolves_or_reports_missing():
    """flowio is a core dep; whether or not it's installed in this env, the
    probe must classify it consistently (available <=> has a version)."""
    flowio = next(c for c in cap.probe_capabilities() if c['key'] == 'flowio')
    assert flowio['extra'] == ''                       # core, not an extra
    assert flowio['available'] == (flowio['problem'] == '')
