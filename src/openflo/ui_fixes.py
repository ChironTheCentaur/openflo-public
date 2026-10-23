"""The review window for fixes that reach this session's saved results: one
row per result a fix would change, with its numbers before and after and a
tick to apply it. Nothing changes until Apply; what was built on a
corrected result is then offered as a follow-up re-run in the same window
(Open… opens the dialog for a step it cannot re-run itself), with the files
exported from the corrected samples before the fix (listed, never
re-exported).

The rows come from fixes.plan; `row_values` formats one without Tk.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

from .tool_window import ToolWindow
from .ui_notice import open_folder
from .ui_tips import tip

ON, OFF, NA = '☑', '☐', '—'
OPEN, SHOW = 'Open…', 'Show in folder'


def _gate_numbers(v):
    n, pct = v if isinstance(v, tuple) else (None, None)
    if n is None:
        return 'NA'
    return f"{n:,} ({'NA' if pct != pct else f'{pct:.1f}%'})"


def _numbers(row, key):
    v = row.get(key)
    if v is None:
        return ''
    return _gate_numbers(v) if row['result'] == 'gate' else str(v)


def _what(row):
    if row['result'] == 'gate':
        return f"gate {row['name']}" if row.get('name') else f"gate {row['key']}"
    return f"column {row['key']}"


def row_values(row):
    """``(sample, result, fix, made by, before, after, note)`` for a row."""
    made = row['stamp'].get('version') or 'unknown'
    if row.get('informational'):
        note = "default changed: flagged only, nothing to apply"
    elif row.get('status') == 'declined':
        note = "declined earlier"
    elif row.get('status') == 'applied':
        note = "fix applied to other results earlier"
    elif row.get('why'):
        note = f"kept: {row['why']}"
    elif not row.get('changed'):
        note = "no change (the result already matches)"
    else:
        note = ''
    fix = ' + '.join(row.get('fixes') or ()) or 're-run'
    return (row['sample'], _what(row), fix, made, _numbers(row, 'before'),
            _numbers(row, 'after'), note)


def export_values(e):
    """``(file, made, made by, state)`` for a stale export (fixes.
    stale_exports)."""
    return (e['path'], (e.get('time') or '').replace('T', ' '),
            e.get('version') or 'unknown',
            'there' if e['exists'] else 'missing')


def _folder_of(e):
    """What Show in folder opens: the file itself while it exists (the file
    manager selects it), else the folder it was in."""
    return e['path'] if e['exists'] else os.path.dirname(e['path'])


def ticked_by_default(row) -> bool:
    """A row starts ticked when the fix changes it cleanly and nobody said no
    to it before."""
    return (not row.get('informational') and row.get('new') is not None
            and row.get('status', 'open') == 'open' and not row.get('why'))


class FixReviewWindow(ToolWindow):
    """Lists the fixes for this session; Apply applies the ticked rows."""

    COLUMNS = (('use', 'Apply', 60), ('sample', 'Sample', 120),
               ('result', 'Result', 130), ('fix', 'Fix', 110),
               ('made', 'Made by', 80), ('before', 'Before', 120),
               ('after', 'After', 120), ('note', 'Note', 220))

    EXPORT_COLUMNS = (('show', '', 110), ('file', 'File', 420),
                      ('made', 'Made', 140), ('by', 'Made by', 80),
                      ('state', 'Now', 70))

    def __init__(self, editor, rows=(), on_apply=None, on_rerun=None,
                 on_open=None, stale_exports=None):
        super().__init__(editor, "Fixes for this session",
                         geometry='900x420', editor=editor)
        self.minsize(560, 300)
        self.rows = list(rows)
        self.on_apply = on_apply
        self.on_rerun = on_rerun
        self.on_open = on_open
        self.stale_exports = stale_exports
        self.follows: list = []
        self.exports: list = []
        self.export_tv: ttk.Treeview | None = None
        self.ticked: dict = {}
        self.mode = 'fixes'

        self.btns = ttk.Frame(self, padding=(10, 0, 10, 8))
        self.btns.pack(side='bottom', fill='x')
        self.close_btn = ttk.Button(self.btns, text="Not now",
                                    command=self.destroy)
        self.close_btn.pack(side='right')
        tip(editor, self.close_btn, "Close without changing anything; the "
            "fixes are offered again the next time the session opens.")
        self.apply_btn = ttk.Button(self.btns, text="Apply ticked",
                                    default='active', command=self.apply)
        self.apply_btn.pack(side='right', padx=(0, 6))
        tip(editor, self.apply_btn, "Correct the ticked results (one undo "
            "step). Each result's original is kept in the session. Unticked "
            "fixes are recorded as declined and not offered again.")

        self.intro = tk.StringVar(master=self)
        ttk.Label(self, textvariable=self.intro, padding=(10, 8, 10, 4),
                  wraplength=860, justify='left').pack(side='top', fill='x')
        body = ttk.Frame(self, padding=(10, 0, 10, 6))
        body.pack(side='top', fill='both', expand=True)
        self.tv = ttk.Treeview(body, show='headings', height=10,
                               columns=[c for c, _t, _w in self.COLUMNS])
        for c, t, w in self.COLUMNS:
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor='center' if c == 'use' else 'w',
                           stretch=c == 'note')
        sb = ttk.Scrollbar(body, orient='vertical', command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.tv.pack(side='left', fill='both', expand=True)
        self.tv.bind('<Button-1>', self._on_click)
        self.tv.bind('<space>', self._on_space)
        self._show_fixes()

    # ── the two lists ───────────────────────────────────────────────────────

    def _fill(self, items):
        self.tv.delete(*self.tv.get_children())
        for iid, mark, values in items:
            self.tv.insert('', 'end', iid=iid, values=(mark,) + tuple(values))

    def _show_fixes(self):
        self.mode = 'fixes'
        if not self.rows:
            self.intro.set("No registered fix applies to the results in "
                           "this session.")
            self.apply_btn.configure(state='disabled')
            self.close_btn.configure(text="Close")
        else:
            self.intro.set(
                "These results were made by an OpenFlo version with a bug "
                "since fixed (or a default since changed). Tick the ones to "
                "correct: each is recomputed with this version, its original "
                "is kept in the session, and nothing changes until Apply.")
        self.ticked = {str(i): ticked_by_default(r)
                       for i, r in enumerate(self.rows)
                       if not r.get('informational')
                       and r.get('new') is not None}
        self._fill([(str(i), (ON if self.ticked[str(i)] else OFF)
                     if str(i) in self.ticked else NA, row_values(r))
                    for i, r in enumerate(self.rows)])

    def _show_follow_ups(self, follows, exports=()):
        self.mode = 'follow'
        self.follows = list(follows)
        self.intro.set(
            "Applied. These results were built on a corrected one. Tick the "
            "ones to re-run now with this version; slow steps start "
            "unticked. Open… opens the tool that re-runs the rest; nothing "
            "starts until you run it there." if self.follows else "Applied.")
        self.ticked = {str(i): not f['slow'] for i, f in
                       enumerate(self.follows) if f['rerun']}
        items = []
        for i, f in enumerate(self.follows):
            row = {'sample': f['sample'], 'result': f['result'],
                   'key': f['key'], 'name': f['name'], 'stamp': f['stamp'],
                   'fixes': [], 'changed': True, 'status': 'open'}
            vals = list(row_values(row))
            vals[2] = f['step']
            vals[6] = (f"built on {f['because']}" + (
                '' if f['rerun'] else f"; re-run from {f['where']}"
                if f['where'] else '; re-run it from its tool'))
            mark = ((ON if self.ticked[str(i)] else OFF)
                    if str(i) in self.ticked
                    else OPEN if f.get('opens') and self.on_open else NA)
            items.append((str(i), mark, vals))
        self._fill(items)
        self.apply_btn.configure(text="Re-run ticked", command=self.rerun,
                                 state='normal' if self.ticked else 'disabled')
        self.close_btn.configure(text="Close")
        self._show_exports(exports)

    def _show_exports(self, exports):
        """List the files exported from the corrected samples before the
        fix, each with Show in folder. Nothing is re-exported: that needs the
        export's own settings, which are not recorded."""
        self.exports = list(exports)
        if not self.exports:
            return
        box = ttk.LabelFrame(self, text="Exported before this fix",
                             padding=(8, 4, 8, 6))
        # Packed before the list, so the list gives way when space is short.
        box.pack(side='bottom', fill='x', padx=10, pady=(0, 6),
                 before=self.tv.master)
        ttk.Label(box, wraplength=860, justify='left', text=(
            "These files still hold the uncorrected numbers; their metadata "
            "says which OpenFlo version made them. Export them again from "
            "where they were made to replace them.")).pack(side='top',
                                                          fill='x')
        tv = ttk.Treeview(box, show='headings',
                          height=min(5, len(self.exports)),
                          columns=[c for c, _t, _w in self.EXPORT_COLUMNS])
        for c, t, w in self.EXPORT_COLUMNS:
            tv.heading(c, text=t)
            tv.column(c, width=w, anchor='center' if c == 'show' else 'w',
                      stretch=c == 'file')
        for i, e in enumerate(self.exports):
            tv.insert('', 'end', iid=str(i), values=(
                SHOW if os.path.exists(_folder_of(e)) else NA,)
                + export_values(e))
        tv.pack(side='top', fill='x', pady=(4, 0))
        tv.bind('<Button-1>', self._on_export_click)
        self.export_tv = tv
        self.geometry('900x560')

    # ── ticks ───────────────────────────────────────────────────────────────

    def toggle(self, iid):
        if iid in self.ticked:
            self.ticked[iid] = not self.ticked[iid]
            self.tv.set(iid, 'use', ON if self.ticked[iid] else OFF)

    def _on_click(self, event):
        if self.tv.identify_column(event.x) != '#1':
            return None
        iid = self.tv.identify_row(event.y)
        if iid and iid in self.ticked:
            self.toggle(iid)
        elif iid and self.mode == 'follow':
            self.open_follow_up(iid)
        return 'break'

    def _on_export_click(self, event):
        tv = self.export_tv
        if tv is None or tv.identify_column(event.x) != '#1':
            return None
        iid = tv.identify_row(event.y)
        if iid:
            self.show_export(iid)
        return 'break'

    def _on_space(self, _event):
        for iid in self.tv.selection():
            self.toggle(iid)
        return 'break'

    # ── actions ─────────────────────────────────────────────────────────────

    def apply(self):
        chosen = [self.rows[int(i)] for i, on in self.ticked.items() if on]
        follows, exports = [], []
        if self.on_apply is not None:
            follows = self.on_apply(chosen, self.rows) or []
        if chosen and self.stale_exports is not None:
            exports = self.stale_exports() or []
        if (follows or exports) and self.winfo_exists():
            self._show_follow_ups(follows, exports)
        else:
            self.destroy()

    def open_follow_up(self, iid):
        """Open… on a follow-up row: open the dialog that re-runs its step.
        Starts nothing; the window stays for the other rows."""
        f = self.follows[int(iid)]
        if (iid in self.ticked or not f.get('opens')
                or self.on_open is None):
            return False
        return bool(self.on_open(f))

    def show_export(self, iid):
        """Show in folder on a listed export: its folder in the file
        manager, the file selected where it still exists."""
        folder = _folder_of(self.exports[int(iid)])
        if not os.path.exists(folder):
            return False
        return open_folder(folder)

    def rerun(self):
        chosen = [self.follows[int(i)] for i, on in self.ticked.items() if on]
        self.destroy()
        if self.on_rerun is not None and chosen:
            self.on_rerun(chosen)
