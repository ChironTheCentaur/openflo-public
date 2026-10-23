"""One answer to "can this engine run here?", from an import, everywhere.

Help → Environment and the Pipeline looked for the package (find_spec) and
Compare embeddings imported it, so for a package installed without its own
dependencies (TriMap without annoy, in the audit) Environment ticked it, the
Pipeline offered it, Compare embeddings said it was not installed, and a
Pipeline run with it ticked made nothing. Every one of them now asks
``capabilities.engine_unavailable``, which imports the package in a separate
Python process and remembers the answer.

Generated engines: three tiny packages written for each test, one that
imports, one that fails to (a dependency of its own missing), and one name
that is not installed at all.
"""
from __future__ import annotations

import sys
import textwrap
import tkinter as tk

import pytest

from openflo import capabilities as cap
from tests.conftest import gui_unavailable

OK, BROKEN, MISSING = ('ofl_test_engine_ok', 'ofl_test_engine_broken',
                       'ofl_test_engine_absent')


@pytest.fixture
def engines(tmp_path, monkeypatch):
    """Write the generated engines, put them on the path of this process and
    of the import-check process, and start with nothing remembered."""
    (tmp_path / OK).mkdir()
    (tmp_path / OK / '__init__.py').write_text('VALUE = 1\n')
    (tmp_path / BROKEN).mkdir()
    (tmp_path / BROKEN / '__init__.py').write_text(textwrap.dedent('''
        import ofl_test_dependency_not_installed  # noqa: F401
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    monkeypatch.setattr(cap, '_IMPORT_RESULT', {}, raising=False)
    for m in (OK, BROKEN):
        sys.modules.pop(m, None)
    yield
    for m in (OK, BROKEN):
        sys.modules.pop(m, None)


def test_the_check_tells_the_three_apart(engines):
    assert cap.engine_unavailable(OK) is None
    assert cap.engine_unavailable(MISSING) == cap.NOT_INSTALLED
    why = cap.engine_unavailable(BROKEN)
    assert why.startswith('fails to load (ModuleNotFoundError')
    assert 'ofl_test_dependency_not_installed' in why


def test_the_check_imports_in_another_process(engines):
    """Imports in a thread of the GUI process set off garbage collection
    there, which finalised Tk objects off the Tk thread."""
    assert cap.check_engines([OK, BROKEN]) == {
        OK: None, BROKEN: cap.engine_unavailable(BROKEN)}
    assert OK not in sys.modules and BROKEN not in sys.modules


def test_a_quick_look_uses_presence_until_the_import_has_run(engines):
    assert cap.engine_unavailable(BROKEN, load=False) is None   # present
    assert not cap.engine_checked(BROKEN)
    cap.check_engines([BROKEN])
    assert cap.engine_checked(BROKEN)
    assert cap.engine_unavailable(BROKEN, load=False) is not None
    # Not installed needs no import to know.
    assert cap.engine_unavailable(MISSING, load=False) == cap.NOT_INSTALLED


def test_the_answer_is_remembered(engines, monkeypatch):
    cap.check_engines([OK, BROKEN])
    monkeypatch.setattr(cap, '_import_check',
                        lambda mods: pytest.fail(f"checked again: {mods}"))
    assert cap.check_engines([OK, BROKEN])[OK] is None
    assert cap.engine_unavailable(BROKEN) is not None


def test_without_the_check_process_it_imports_here(engines, monkeypatch):
    import subprocess

    def no_process(*a, **k):
        raise OSError('no interpreter')
    monkeypatch.setattr(subprocess, 'run', no_process)
    assert cap._import_check([OK, BROKEN])[OK] is None
    assert cap._import_check([BROKEN])[BROKEN].startswith('fails to load')


def test_environment_lists_a_broken_engine_as_failing(engines, monkeypatch):
    monkeypatch.setattr(cap, '_ENGINES', [
        (OK, OK, 'Fine engine', 'An engine that works', 'embed'),
        (BROKEN, BROKEN, 'Broken engine', 'An engine that fails', 'embed'),
        (MISSING, MISSING, 'Absent engine', 'An engine not there', 'embed'),
    ])
    rows = {r['label']: r for r in cap.probe_capabilities()}
    assert rows['Fine engine']['available']
    assert not rows['Broken engine']['available']
    assert rows['Broken engine']['problem'].startswith('fails to load')
    assert rows['Absent engine']['problem'] == cap.NOT_INSTALLED
    assert all(r['checked'] for r in rows.values())


def test_every_place_gives_the_same_answer(engines, monkeypatch):
    """Environment, Compare embeddings, the Cluster dialog and the Pipeline
    agree, for an engine that runs and one that is installed but doesn't."""
    from openflo import dr_compare
    from openflo import workspace as W
    from openflo.gui import ViewGateEditorWindow as Editor
    monkeypatch.setattr(dr_compare, '_BACKEND_MODULE',
                        {**dr_compare._BACKEND_MODULE,
                         'trimap': BROKEN, 'pacmap': OK})
    monkeypatch.setitem(W.EXTRA_EMBEDDINGS, 'trimap', (BROKEN, 'TriMap'))
    monkeypatch.setitem(W.EXTRA_EMBEDDINGS, 'pacmap', (OK, 'PaCMAP'))
    monkeypatch.setattr(Editor, '_EMBED_BACKEND',
                        {**Editor._EMBED_BACKEND, 'TriMap': BROKEN,
                         'PaCMAP': OK})
    monkeypatch.setattr(cap, '_ENGINES', [
        (OK, OK, 'Fine', '', 'embed'), (BROKEN, BROKEN, 'Broken', '', 'embed')])
    if hasattr(cap, 'check_engines'):
        cap.check_engines([OK, BROKEN])             # as the editor does
    env = {r['key']: r['available'] for r in cap.probe_capabilities()}
    compare = {OK: dr_compare._backend_importable('pacmap'),
               BROKEN: dr_compare._backend_importable('trimap')}
    pipeline = {OK: W.embedding_available('pacmap'),
                BROKEN: W.embedding_available('trimap')}
    avail, _missing = Editor._available_embeddings()
    cluster = {OK: 'PaCMAP' in avail, BROKEN: 'TriMap' in avail}
    assert env == compare == pipeline == cluster == {OK: True, BROKEN: False}
    assert 'installed but fails to load' in W.embedding_unavailable('trimap')


def test_the_pipeline_panel_disables_a_broken_engine(engines, monkeypatch):
    from openflo import workspace as W
    monkeypatch.setitem(W.EXTRA_EMBEDDINGS, 'trimap', (BROKEN, 'TriMap'))
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f'Tk unavailable: {exc}')
    try:
        view = W.PipelineWorkspaceView(root, W.WorkspaceModel())
        root.update()
        box, var = view._embed_boxes['trimap']
        # Before the import has run: offered (it is installed), as before.
        assert not box.instate(['disabled'])
        cap.check_engines([BROKEN])         # what _check_embed_engines runs
        view._sync_embed_boxes()
        assert box.instate(['disabled']) and var.get() is False
        assert 'fails to load' in view._embed_tip('trimap', 'TriMap')
    finally:
        root.destroy()
