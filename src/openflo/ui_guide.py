"""Help, User guide: the bundled guide, offline (Complaints 27).

Topics on the left, with a search box; the chosen topic on the right,
rendered from its Markdown (user_guide.blocks). The getting-started topic
offers to load the example dataset it walks through.

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ui_tips import auto_tip
from .user_guide import blocks, search, topics


class GuideWindow(tk.Toplevel):
    """[Search] / topics | topic text / [Load example dataset] [Online
    documentation] [Close]."""

    def __init__(self, editor, topic=None):
        super().__init__(editor)
        self.editor = editor
        self.title("OpenFlo user guide")
        self.minsize(760, 480)
        self.items = topics()
        self.shown = [slug for slug, _t, _x in self.items]

        bar = ttk.Frame(self, padding=(12, 4, 12, 10))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Close", command=self.destroy).pack(side='right')
        if hasattr(editor, '_open_documentation'):
            w = ttk.Button(bar, text="Online documentation",
                           command=editor._open_documentation)
            w.pack(side='right', padx=(0, 6))
            auto_tip(w, "Open the documentation on the web (needs a "
                        "connection; this guide does not).")
        self.example_btn = ttk.Button(bar, text="Load example dataset",
                                      command=self._load_example)
        auto_tip(self.example_btn, "Load the synthetic dataset the "
                                   "tutorial uses.")

        side = ttk.Frame(self, padding=(12, 10, 6, 0))
        side.pack(side='left', fill='y')
        self.query_var = tk.StringVar()
        entry = ttk.Entry(side, textvariable=self.query_var, width=28)
        self.search_entry = entry
        entry.pack(fill='x')
        auto_tip(entry, "Search every topic; Enter shows the first match.")
        self.query_var.trace_add('write', lambda *_a: self._filter())
        entry.bind('<Return>', lambda _e: self._open_first())
        self.topic_list = tk.Listbox(side, width=30, activestyle='none',
                                     exportselection=False)
        self.topic_list.pack(fill='y', expand=True, pady=(6, 0))
        self.topic_list.bind('<<ListboxSelect>>', lambda _e: self._picked())
        self.match_label = ttk.Label(side, style='Muted.TLabel',
                                     wraplength=220, justify='left')
        self.match_label.pack(fill='x', pady=(4, 0))

        body = ttk.Frame(self, padding=(6, 10, 12, 0))
        body.pack(side='left', fill='both', expand=True)
        self.text = tk.Text(body, wrap='word', padx=12, pady=8,
                            relief='flat', cursor='arrow')
        sb = ttk.Scrollbar(body, orient='vertical', command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.text.pack(side='left', fill='both', expand=True)
        self._style_text()
        self.current = None
        self._filter()
        self.show(topic or self.items[0][0])
        entry.focus_set()

    def _style_text(self):
        from .theme import current_palette
        pal = current_palette()
        self.text.configure(background=pal['panel'], foreground=pal['fg'],
                            insertbackground=pal['fg'], font='TkTextFont')
        self.text.tag_configure('h1', font='OpenFloTitle',
                                spacing1=4, spacing3=8)
        self.text.tag_configure('h2', font='OpenFloHeading',
                                spacing1=10, spacing3=4)
        self.text.tag_configure('bold', font='OpenFloHeading')
        self.text.tag_configure('code', font='OpenFloMono')
        self.text.tag_configure('bullet', lmargin1=12, lmargin2=26)
        self.text.tag_configure('step', lmargin1=12, lmargin2=26)
        self.text.tag_configure('para', spacing3=6)

    def _filter(self):
        q = self.query_var.get().strip()
        if q:
            hits = search(q, self.items)
            self.shown = [slug for slug, _t, _s in hits]
            self.match_label.configure(
                text=(hits[0][2] if hits else "No topic matches."))
        else:
            self.shown = [slug for slug, _t, _x in self.items]
            self.match_label.configure(text='')
        titles = {slug: title for slug, title, _x in self.items}
        self.topic_list.delete(0, 'end')
        for slug in self.shown:
            self.topic_list.insert('end', titles[slug])
        if self.current in self.shown:
            i = self.shown.index(self.current)
            self.topic_list.selection_set(i)
            self.topic_list.see(i)

    def _open_first(self):
        if self.shown:
            self.show(self.shown[0])

    def _picked(self):
        sel = self.topic_list.curselection()
        if sel:
            self.show(self.shown[sel[0]])

    def show(self, slug):
        """Render topic ``slug``."""
        text = next((x for s, _t, x in self.items if s == slug), None)
        if text is None:
            return
        self.current = slug
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        for kind, runs in blocks(text):
            if kind == 'bullet':
                self.text.insert('end', '•  ', (kind,))
            for run_kind, run in runs:
                tags = (kind,) if run_kind == 'text' else (kind, run_kind)
                self.text.insert('end', run, tags)
            self.text.insert('end', '\n', (kind,))
        self.text.configure(state='disabled')
        self.text.yview_moveto(0)
        if slug.endswith('getting-started') and \
                hasattr(self.editor, '_load_example_data'):
            self.example_btn.pack(side='left')
        else:
            self.example_btn.pack_forget()
        self.topic_list.selection_clear(0, 'end')
        if slug in self.shown:
            self.topic_list.selection_set(self.shown.index(slug))

    def rendered(self):
        """The topic as shown, for tests."""
        return self.text.get('1.0', 'end-1c')

    def _load_example(self):
        self.editor._load_example_data()
