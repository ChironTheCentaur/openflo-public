"""Audit-trail / provenance viewer.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class AuditWindow(tk.Toplevel):
    """Read-only viewer for the analysis audit trail, with Markdown / CSV /
    JSON export. Non-modal and live: ``refresh()`` is called by the editor's
    ``_audit`` whenever a new operation is recorded while this is open."""

    def __init__(self, parent, audit_log):
        super().__init__(parent)
        self.editor = parent
        self.title("Analysis history (audit trail)")
        self.geometry("820x520")
        self._log = audit_log

        bar = ttk.Frame(self)
        bar.pack(fill='x', side='top')
        ttk.Label(bar, text="Provenance — operations in order:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='left', padx=6, pady=4)
        ttk.Button(bar, text="Export Markdown…",
                   command=lambda: self._export('md')).pack(
            side='right', padx=(0, 6), pady=4)
        ttk.Button(bar, text="Export CSV…",
                   command=lambda: self._export('csv')).pack(
            side='right', padx=(0, 4), pady=4)
        ttk.Button(bar, text="Export JSON…",
                   command=lambda: self._export('json')).pack(
            side='right', padx=(0, 4), pady=4)

        bar2 = ttk.Frame(self)
        bar2.pack(fill='x', side='top')
        ttk.Label(bar2, text="Compliance:", foreground='#555').pack(
            side='left', padx=6)
        ttk.Button(bar2, text="Sign & export record…",
                   command=self._sign_record).pack(side='left', padx=(0, 4),
                                                   pady=2)
        ttk.Button(bar2, text="Verify record…",
                   command=self._verify_record).pack(side='left', pady=2)

        cols = ('seq', 'time', 'action', 'details')
        widths = (40, 150, 130, 460)
        tv = ttk.Treeview(self, columns=cols, show='headings',
                          selectmode='browse')
        for c, w in zip(cols, widths, strict=True):
            tv.heading(c, text=c.capitalize())
            tv.column(c, width=w, anchor='w',
                      stretch=(c == 'details'))
        sb = ttk.Scrollbar(self, orient='vertical', command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        tv.pack(side='left', fill='both', expand=True)
        self._tv = tv
        self.refresh()

    def refresh(self):
        from .audit import _short
        tv = self._tv
        tv.delete(*tv.get_children())
        entries = self._log.entries()
        for e in entries:
            det = ", ".join(f"{k}={_short(v)}"
                            for k, v in e['details'].items())
            tv.insert('', 'end',
                      values=(e['seq'], e.get('time') or '',
                              e['action'], det))
        if entries:
            tv.see(tv.get_children()[-1])

    def _meta(self):
        from datetime import datetime

        from . import __version__
        return {'openflo_version': __version__,
                'exported': datetime.now().isoformat(timespec='seconds'),
                'operations': len(self._log)}

    def _export(self, fmt):
        if not len(self._log):
            messagebox.showinfo("History", "Nothing to export yet.",
                                parent=self)
            return
        ext = {'md': '.md', 'csv': '.csv', 'json': '.json'}[fmt]
        ftypes = {'md': [('Markdown', '*.md')],
                  'csv': [('CSV', '*.csv')],
                  'json': [('JSON', '*.json')]}[fmt]
        path = filedialog.asksaveasfilename(
            parent=self, title="Export audit trail", defaultextension=ext,
            initialfile='audit_trail' + ext,
            filetypes=ftypes + [('All files', '*.*')])
        if not path:
            return
        try:
            if fmt == 'md':
                text = self._log.to_markdown(meta=self._meta())
            elif fmt == 'csv':
                text = self._log.to_csv()
            else:
                text = json.dumps(
                    {'format': 'openflo-audit', 'version': 1,
                     'meta': self._meta(), 'entries': self._log.to_list()},
                    indent=2, ensure_ascii=False)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as exc:
            messagebox.showerror(
                "History", f"Export failed:\n{type(exc).__name__}: {exc}",
                parent=self)
            return
        messagebox.showinfo("History", f"Exported:\n{path}", parent=self)

    # ── Compliance / sign-off ────────────────────────────────────────────
    def _sign_record(self):
        """Build an integrity manifest (data-file hashes + audit-trail hash +
        version), attach an electronic signature, and write a signed JSON
        record + a Markdown copy."""
        from datetime import datetime
        from tkinter import simpledialog

        from . import __version__
        from .compliance import (
            build_manifest,
            record_to_markdown,
            sign_manifest,
        )
        ed = self.editor
        files = {n: getattr(ed._samples[n], 'path', '') or ''
                 for n in getattr(ed, '_sample_order', [])
                 if n in ed._samples}
        signer = simpledialog.askstring(
            "Electronic signature", "Signer (name / ID):", parent=self)
        if not signer:
            return
        meaning = simpledialog.askstring(
            "Electronic signature", "Meaning of signature:",
            initialvalue="Reviewed and approved", parent=self) or "Signed"
        now = datetime.now().isoformat(timespec='seconds')
        manifest = build_manifest(files, self._log.to_list(), __version__,
                                  created=now)
        record = sign_manifest(manifest, signer, meaning, now)
        path = filedialog.asksaveasfilename(
            parent=self, title="Save signed compliance record",
            defaultextension='.json', initialfile='compliance_record.json',
            filetypes=[('Signed record (JSON)', '*.json')])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            md = os.path.splitext(path)[0] + '.md'
            with open(md, 'w', encoding='utf-8') as f:
                f.write(record_to_markdown(record))
        except Exception as exc:
            messagebox.showerror("Compliance", f"Write failed:\n{exc}",
                                 parent=self)
            return
        try:
            ed._audit('compliance.sign', signer=signer, meaning=meaning,
                      path=path)
        except Exception:
            pass
        messagebox.showinfo(
            "Compliance",
            f"Signed by {signer} — {len(files)} files hashed.\n\n{path}",
            parent=self)

    def _verify_record(self):
        """Load a signed record and re-check the manifest + file hashes,
        reporting whether the signatures are still valid (tamper check)."""
        from .compliance import verify_record
        path = filedialog.askopenfilename(
            parent=self, title="Verify signed compliance record",
            filetypes=[('Signed record (JSON)', '*.json'),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                record = json.load(f)
        except Exception as exc:
            messagebox.showerror("Compliance", f"Could not read:\n{exc}",
                                 parent=self)
            return
        v = verify_record(record)
        lines = [f"Overall: {'VALID' if v['all_valid'] else 'INVALID / TAMPERED'}",
                 ""]
        for s in v['signatures']:
            lines.append(f"  {'✓' if s['valid'] else '✗'} {s['signer']} — "
                         f"{s['meaning']} ({s['time']})")
        bad = [n for n, ok in v['files_ok'].items() if not ok]
        if bad:
            lines.append("")
            lines.append("Changed/missing data files: " + ", ".join(bad))
        (messagebox.showinfo if v['all_valid'] else messagebox.showwarning)(
            "Verify compliance record", "\n".join(lines), parent=self)
