"""The editor's title is "session name — OpenFlo", with * while there are
unsaved changes.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import tests._sample_editor as shared

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def test_title_names_the_session_and_marks_unsaved_changes(
        editor, tmp_path, monkeypatch):
    root, ed = editor
    import openflo.editor_session as es
    # The sweep's editor starts with gates added: open from a clean state.
    ed._note_saved()
    assert ed.title() == "Untitled session — OpenFlo"
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                  'value': 100.0, 'name': 'FL1 pos'})
    root.update()
    assert ed.title() == "*Untitled session — OpenFlo"
    path = tmp_path / 'synthetic_run.flowsession'
    monkeypatch.setattr(ed, '_write_session', lambda p: {'samples': []})
    monkeypatch.setattr(es.filedialog, 'asksaveasfilename',
                        lambda **k: str(path))
    ed._save_session(save_as=True)
    assert ed.title() == "synthetic_run.flowsession — OpenFlo"
    ed._undo()
    assert ed.title() == "*synthetic_run.flowsession — OpenFlo"
    ed._note_saved()
    ed._remove_samples(['beta_1'])
    assert ed.title() == "*synthetic_run.flowsession — OpenFlo"
