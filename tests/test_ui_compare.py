"""Headless smoke test for the FlowJo-workspace comparison dialog.

Constructs the gate-editor window and the CompareWspDialog without any file,
then destroys them. Skips cleanly when no display / Tk is available.
"""

from tests.conftest import editor_or_skip


def test_compare_dialog_constructs_and_destroys():
    root, ed = editor_or_skip()

    from openflo.ui_compare import CompareWspDialog
    d = CompareWspDialog(ed)
    # No file picked: the table is empty and export is disabled.
    assert d._tree.get_children() == ()
    assert str(d._export_btn["state"]) == "disabled"
    d.destroy()
    ed.destroy()
    root.destroy()
