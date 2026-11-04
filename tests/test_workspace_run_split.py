"""The Pipeline Workspace's run logic lives in workspace_run, a module that
does not load Tk, and every name workspace.py offered before the split still
resolves there to the very same object."""
import subprocess
import sys
import warnings

import pytest

from openflo import workspace, workspace_run

# Every top-level name the run section defined before it moved, with the name
# it has now (six private names the window shares became public).
MOVED = {
    '_PROG': 'PROGRESS_MARKER', 'sample_iid': 'sample_iid', 'gate_iid': 'gate_iid',
    'comp_iid': 'comp_iid', 'parse_iid': 'parse_iid', '_DAY_TOKEN_RE': '_DAY_TOKEN_RE',
    '_GENERIC_HOST_RE': '_GENERIC_HOST_RE', 'derive_trial_name': 'derive_trial_name',
    'natural_sort_key': 'natural_sort_key', '_COMP_NAME_RE': '_COMP_NAME_RE',
    'is_comp_sample': 'is_comp_sample', 'trial_day_number': 'trial_day_number',
    'extract_one_context': 'extract_one_context',
    '_display_transforms': '_display_transforms',
    'extract_editor_context': 'extract_editor_context', 'gate_is_leaf': 'gate_is_leaf',
    'gate_chain': 'gate_chain', 'gate_path': 'gate_path',
    'build_drop_payload': 'build_drop_payload', 'comp_summary': 'comp_summary',
    'DEFAULT_RUN_CFG': 'DEFAULT_RUN_CFG', 'default_run_cfg': 'default_run_cfg',
    'EXTRA_EMBEDDINGS': 'EXTRA_EMBEDDINGS', 'embedding_unavailable': 'embedding_unavailable',
    'embedding_available': 'embedding_available', 'run_label': 'run_label',
    'proper_run_channels': 'proper_run_channels', 'resolve_run_sample': 'resolve_run_sample',
    '_plot_embedding': '_plot_embedding', 'run_events_text': 'run_events_text',
    '_blank_result': '_blank_result', 'WORKSPACE_FORMAT': 'WORKSPACE_FORMAT',
    'WORKSPACE_SCHEMA': 'WORKSPACE_SCHEMA', 'newer_version_note': 'newer_version_note',
    'merge_unit_cfg': 'merge_unit_cfg', 'build_run_units': 'build_run_units',
    '_source_tags': '_source_tags', '_cap_events': 'cap_events',
    '_BATCH_FAILURES_SHOWN': '_BATCH_FAILURES_SHOWN',
    '_batch_failure_text': 'batch_failure_text', 'prepare_unit': 'prepare_unit',
    'prepare_run': 'prepare_run', 'compute_run': 'compute_run',
    'run_workspace_item': 'run_workspace_item', '_CREATE_NO_WINDOW': '_CREATE_NO_WINDOW',
    '_CREATE_NEW_CONSOLE': '_CREATE_NEW_CONSOLE', '_DETACHED_PROCESS': '_DETACHED_PROCESS',
    '_no_window_creationflags': '_no_window_creationflags',
    '_suppress_child_windows': '_suppress_child_windows',
    '_own_process_group': 'own_process_group',
    '_signal_process_group': 'signal_process_group',
    '_subprocess_main': 'subprocess_main', 'find_run_outputs': 'find_run_outputs',
    '_item_to_json': '_item_to_json', '_item_from_json': '_item_from_json',
    'WorkspaceModel': 'WorkspaceModel',
}


def test_the_run_logic_imports_with_tkinter_blocked():
    code = ("import sys\n"
            "sys.modules['tkinter'] = None\n"
            "import openflo.workspace_run as r\n"
            "assert r.WorkspaceModel and r.compute_run\n"
            "assert not [m for m in sys.modules if m.startswith('tkinter') and sys.modules[m]]\n"
            "print('ok')\n")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'ok'


@pytest.mark.parametrize('old, new', sorted(MOVED.items()))
def test_every_old_workspace_name_is_the_moved_object(old, new):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        assert getattr(workspace, old) is getattr(workspace_run, new)


def test_names_the_window_does_not_use_warn_where_they_moved():
    with pytest.warns(DeprecationWarning, match=r'workspace_run\.derive_trial_name'):
        _ = workspace.derive_trial_name
    with pytest.warns(DeprecationWarning, match=r'workspace_run\.PROGRESS_MARKER'):
        _ = workspace._PROG


def test_names_the_window_imports_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        assert workspace.WorkspaceModel is workspace_run.WorkspaceModel
        assert workspace.compute_run is workspace_run.compute_run


def test_an_unknown_name_is_still_an_attribute_error():
    with pytest.raises(AttributeError):
        _ = workspace.no_such_name


def test_the_run_section_left_the_window_module():
    # The window module defines only the Tk side now.
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(workspace))
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert defined == {'_tip', '__getattr__', 'PipelineWorkspaceView',
                       'ResultsViewer', 'WorkspacePanel'}
