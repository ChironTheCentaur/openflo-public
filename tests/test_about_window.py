"""Help > About OpenFlo is a proper window with the version, licence,
citation and links to the documentation.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import tests._sample_editor as shared
import tests.test_windows_nothing_hidden as sweep
from tests._editor_ui import _texts

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def test_about_has_version_licence_citation_and_links(editor, monkeypatch):
    from openflo.editor_help import CITATION, AboutWindow
    from openflo.result_stamp import app_version
    from openflo.update import DOCS_URL
    root, ed = editor
    opened = []
    monkeypatch.setattr('webbrowser.open', opened.append)
    win = AboutWindow(ed)
    root.update()
    texts = []
    _texts(win, texts, 'about')
    flat = ' '.join(t for _w, _k, t in texts)
    assert f"Version {app_version()}" in flat and 'MIT' in flat
    assert 'please cite' in flat
    buttons = {t: w for w in sweep._walk(win) if w.winfo_class() == 'TButton'
               for t in [str(w.cget('text'))]}
    buttons['Copy citation'].invoke()
    assert root.clipboard_get() == CITATION
    buttons['Documentation'].invoke()
    assert opened == [DOCS_URL]
