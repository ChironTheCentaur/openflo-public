"""Methods & provenance report window.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog, ttk

from .clipboard_export import clipboard_put
from .tool_window import set_readonly_text
from .ui_errors import show_error
from .ui_notice import report_saved
from .ui_tips import auto_tip


class MethodsWindow(tk.Toplevel):
    """Paper-ready Methods paragraph (from the audit trail + citations) plus a
    reproducibility run manifest (versions / params / samples)."""

    def __init__(self, editor):
        super().__init__(editor)
        self.title("Methods & provenance")
        self.geometry("740x620")
        self._editor = editor

        self._para, self._manifest = self._build()

        # The button row is packed first, at the bottom, so a short window
        # shrinks the manifest box and never hides the buttons.
        bar = ttk.Frame(self)
        bar.pack(side='bottom', fill='x', pady=6)
        _b_copy = ttk.Button(bar, text="Copy methods", command=self._copy)
        _b_copy.pack(side='left', padx=8)
        auto_tip(_b_copy,
                 "Copy the methods paragraph to the clipboard — the "
                 "compensation, transforms, gating and clustering settings "
                 "actually used, written out for a paper's methods section.")
        _b_save = ttk.Button(bar, text="Save manifest…", command=self._save)
        _b_save.pack(side='left')
        auto_tip(_b_save,
                 "Write the full machine-readable manifest: every parameter, "
                 "version and seed behind this analysis, so it can be "
                 "reproduced or audited later.")
        ttk.Button(bar, text="Close", command=self.destroy).pack(
            side='right', padx=8)
        ttk.Label(self, text="Methods paragraph (paper-ready):",
                  font='OpenFloHeading').pack(
            anchor='w', padx=10, pady=(10, 2))
        t1 = tk.Text(self, wrap='word', height=9)
        t1.pack(fill='x', padx=10)
        ttk.Label(self, text="Run manifest:",
                  font='OpenFloHeading').pack(
            anchor='w', padx=10, pady=(10, 2))
        t2 = tk.Text(self, wrap='none', height=14)
        t2.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        self._texts = (t1, t2)
        self._show()
        # The paragraph follows the analysis by itself (editor_watch).
        watch = getattr(editor, '_watch_changes', None)
        if watch is not None:
            watch(self, self._on_editor_changed, ('audit', 'samples', 'data'))

    def _build(self):
        """The methods paragraph and the manifest text, from the editor
        as it is now."""
        from .provenance import methods_paragraph, run_manifest
        editor = self._editor
        audit = getattr(editor, '_audit_log', None)
        samples = list(editor._samples.values())
        try:
            para = methods_paragraph(audit, samples=samples)
        except Exception as exc:  # noqa: BLE001 - the window shows the error in place of the methods paragraph
            para = f"(methods paragraph unavailable: {exc})"
        # Samples set aside from the comparisons are part of the method: a
        # reader needs to know which were dropped, and why.
        aside = editor._set_aside_items()
        if aside:
            para += (" Samples set aside from group comparisons, with "
                     "the reason: "
                     + "; ".join(f"{n} ({r})" for n, r in aside) + ".")
        try:
            manifest = json.dumps(run_manifest(samples=samples), indent=2)
        except Exception as exc:  # noqa: BLE001 - the window shows the error in place of the manifest
            manifest = f"(manifest unavailable: {exc})"
        return para, manifest

    def _show(self):
        for t, text in zip(self._texts, (self._para, self._manifest),
                           strict=True):
            set_readonly_text(t, text)

    def _on_editor_changed(self):
        """An analysis step was recorded, or the samples changed
        (editor_watch): write the paragraph and manifest again."""
        self._para, self._manifest = self._build()
        self._show()

    def _copy(self):
        try:
            clipboard_put(self, self._para)
            self._editor.status_var.set("Methods paragraph copied to clipboard.")
        except (tk.TclError, AttributeError):
            pass

    def _save(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Save run manifest", defaultextension='.json',
            initialfile='openflo_manifest.json',
            filetypes=[('JSON', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self._manifest)
        except OSError as exc:  # a failed save is shown in an error dialog
            show_error(self, "Save failed", "The manifest could not be saved.",
                       exc, path)
            return
        report_saved(self, path, "Saved")
