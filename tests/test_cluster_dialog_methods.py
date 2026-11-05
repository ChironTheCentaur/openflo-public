"""The Cluster dialog shows only the picked method's settings and lists
markers as "marker (channel)" while running on the channels.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import tests._editor_ui as ui
import tests._sample_editor as shared
import tests.test_windows_nothing_hidden as sweep

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known

root = ui.root


def _cluster_dialog(root, ed):
    before = set(sweep._toplevels(root))
    ed._open_cluster_dialog()
    root.update()
    (dlg,) = [w for k, w in sweep._toplevels(root).items()
              if k not in before]
    return dlg


def _shown(dlg):
    return {key for key, widgets in dlg.master._cluster_method_rows.items()
            if all(w.winfo_manager() == 'grid' for w in widgets)}


def test_cluster_dialog_shows_only_the_picked_methods_settings(editor):
    root, ed = editor
    dlg = _cluster_dialog(root, ed)
    assert _shown(dlg) == {'k', 'resolution', 'repro'}   # PhenoGraph
    radios = {w.cget('text'): w for w in sweep._walk(dlg)
              if w.winfo_class() == 'TRadiobutton'}
    radios['FlowSOM'].invoke()
    root.update()
    assert _shown(dlg) == {'grid', 'meta'}
    radios['Leiden'].invoke()
    root.update()
    assert _shown(dlg) == {'k', 'resolution', 'fast'}
    dlg.destroy()


def test_cluster_dialog_lists_markers_by_name_and_runs_on_channels(
        editor, monkeypatch):
    root, ed = editor
    ed._channel_labels = {c: c for c in ed._channels}
    ed._channel_labels['FL2-A'] = 'Marker2'
    ran = []
    monkeypatch.setattr(type(ed), '_run_clustering',
                        lambda self, **kw: ran.append(kw))
    dlg = _cluster_dialog(root, ed)
    shown = list(ed._cluster_marker_list.get(0, 'end'))
    assert shown == ['FL1-A', 'Marker2 (FL2-A)', 'FL3-A', 'FL4-A']
    ed._cluster_marker_list.selection_clear(0, 'end')
    ed._cluster_marker_list.selection_set(1)
    run = next(w for w in sweep._walk(dlg)
               if w.winfo_class() == 'TButton' and w.cget('text') == 'Run')
    run.invoke()
    assert ran and ran[0]['channels'] == ['FL2-A']
