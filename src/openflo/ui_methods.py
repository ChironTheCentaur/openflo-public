"""Methods & provenance report window.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog, ttk

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

        from .provenance import methods_paragraph, run_manifest
        audit = getattr(editor, '_audit_log', None)
        samples = list(editor._samples.values())
        try:
            self._para = methods_paragraph(audit, samples=samples)
        except Exception as exc:
            self._para = f"(methods paragraph unavailable: {exc})"
        # Samples set aside from the comparisons are part of the method: a
        # reader needs to know which were dropped, and why.
        aside = editor._set_aside_items()
        if aside:
            self._para += (" Samples set aside from group comparisons, with "
                           "the reason: "
                           + "; ".join(f"{n} ({r})" for n, r in aside) + ".")
        try:
            self._manifest = json.dumps(run_manifest(samples=samples), indent=2)
        except Exception as exc:
            self._manifest = f"(manifest unavailable: {exc})"

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
        t1.insert('1.0', self._para)
        t1.configure(state='disabled')
        ttk.Label(self, text="Run manifest:",
                  font='OpenFloHeading').pack(
            anchor='w', padx=10, pady=(10, 2))
        t2 = tk.Text(self, wrap='none', height=14)
        t2.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        t2.insert('1.0', self._manifest)
        t2.configure(state='disabled')

    def _copy(self):
        try:
            self.clipboard_clear()
            self.clipboard_append(self._para)
            self._editor.status_var.set("Methods paragraph copied to clipboard.")
        except Exception:
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
            report_saved(self, path, "Saved")
        except Exception as exc:
            show_error(self, "Save failed", "The manifest could not be saved.",
                       exc, path)
