"""Freeze reference clusters: choose the FlowSOM map a run template labels
every later run with, or take it away (``run_reference`` does the work).
"""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence
from tkinter import ttk
from typing import Any

from .ui_tips import auto_tip

_INTRO = ("Freezing makes one run's FlowSOM clusters the template's "
          "reference: every later run gives each event the cluster of its "
          "nearest reference node instead of clustering again, so a cluster "
          "means the same population on every day and Run history can "
          "follow it. Each run warns when many of its events lie far from "
          "the reference. The run you freeze clustered on its own, so it has "
          "no reference-cluster rows: run that day again to add it to the "
          "cluster lines in Run history.")
_NONE = ("No FlowSOM run in this template's history yet. Set the run "
         "method to FlowSOM, run a day, then freeze that run here.")


class ReferenceDialog(tk.Toplevel):
    """Pick one of `maps` (``(title, path)``, newest first) to freeze, or
    unfreeze the reference named `current`. ``on_choose`` gets the chosen
    map's path, or None to unfreeze."""

    def __init__(self, editor: Any, maps: Sequence[tuple[str, str]] = (),
                 current: str | None = None,
                 on_choose: Callable[[str | None], object] | None = None,
                 parent: tk.Misc | None = None):
        super().__init__(parent or editor)
        self.title("Freeze reference clusters")
        self._paths = dict(maps)
        self._on_choose = on_choose
        ttk.Label(self, text=_INTRO if maps else _NONE, wraplength=520,
                  justify='left', padding=(10, 10, 10, 4)).pack(anchor='w')
        self.now = tk.StringVar(value=(
            f"Frozen now: {current}" if current
            else "Nothing frozen: each run clusters on its own."))
        ttk.Label(self, textvariable=self.now, style='Muted.TLabel',
                  wraplength=520, padding=(10, 0)).pack(anchor='w')

        row = ttk.Frame(self, padding=(10, 6))
        row.pack(fill='x')
        ttk.Label(row, text="Run:").pack(side='left')
        self.map_var = tk.StringVar(value=maps[0][0] if maps else '')
        self.map_cb = ttk.Combobox(row, textvariable=self.map_var, width=58,
                                   values=[t for t, _ in maps],
                                   state='readonly')
        self.map_cb.pack(side='left', padx=6)
        auto_tip(self.map_cb, "The finished FlowSOM run (and run unit) whose "
                 "clusters become the reference, newest first. Pick a day "
                 "that holds every population you expect.")

        btns = ttk.Frame(self, padding=(10, 8))
        btns.pack(fill='x')
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')
        self.freeze_btn = ttk.Button(btns, text="Freeze", command=self.freeze,
                                     state='normal' if maps else 'disabled')
        self.freeze_btn.pack(side='right', padx=6)
        auto_tip(self.freeze_btn, "Save this run's clusters into the "
                 "template as its reference. The template's version changes, "
                 "so Run history marks the day it took effect.")
        self.clear_btn = ttk.Button(btns, text="Unfreeze",
                                    command=self.unfreeze,
                                    state='normal' if current else 'disabled')
        self.clear_btn.pack(side='right')
        auto_tip(self.clear_btn, "Take the reference out of the template: "
                 "each run clusters on its own again.")

    def _choose(self, path: str | None) -> None:
        if self._on_choose is not None:
            self._on_choose(path)
        self.destroy()

    def freeze(self) -> None:
        """Freeze the chosen run's map."""
        path = self._paths.get(self.map_var.get())
        if path:
            self._choose(path)

    def unfreeze(self) -> None:
        """Take the reference away."""
        self._choose(None)
