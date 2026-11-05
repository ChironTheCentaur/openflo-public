"""Headless smoke test for the Quick preview dialog.

Constructs the editor + the dialog WITHOUT requiring an FCS file, then tears
both down. Skips cleanly on machines without a usable Tk display.
"""

from tests.conftest import editor_or_skip


def test_quick_preview_dialog_constructs_and_destroys():
    root, ed = editor_or_skip()
    try:
        from openflo.ui_preview import QuickPreviewDialog
        d = QuickPreviewDialog(ed)
        # Threshold parsing helper is pure — exercise it without a file.
        assert QuickPreviewDialog._parse_thr('') is None
        assert QuickPreviewDialog._parse_thr('abc') is None
        assert QuickPreviewDialog._parse_thr('1.5') == 1.5
        d.destroy()
    finally:
        root.destroy()
