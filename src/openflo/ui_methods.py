"""Methods & provenance report window.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


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
        try:
            self._manifest = json.dumps(run_manifest(samples=samples), indent=2)
        except Exception as exc:
            self._manifest = f"(manifest unavailable: {exc})"

        ttk.Label(self, text="Methods paragraph (paper-ready):",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            anchor='w', padx=10, pady=(10, 2))
        t1 = tk.Text(self, wrap='word', height=9)
        t1.pack(fill='x', padx=10)
        t1.insert('1.0', self._para)
        t1.configure(state='disabled')
        ttk.Label(self, text="Run manifest:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            anchor='w', padx=10, pady=(10, 2))
        t2 = tk.Text(self, wrap='none')
        t2.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        t2.insert('1.0', self._manifest)
        t2.configure(state='disabled')
        bar = ttk.Frame(self)
        bar.pack(fill='x', pady=6)
        ttk.Button(bar, text="Copy methods", command=self._copy).pack(
            side='left', padx=8)
        ttk.Button(bar, text="Save manifest…",
                   command=self._save).pack(side='left')
        ttk.Button(bar, text="Close", command=self.destroy).pack(
            side='right', padx=8)

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
            messagebox.showinfo("Saved", f"Saved:\n{path}", parent=self)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
