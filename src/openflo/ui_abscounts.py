"""Absolute-count (counting-bead) dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ui_tips import auto_tip

# Per-field help. The arithmetic is on-screen already; what is not obvious is
# where each number should come from, which is where this calculation goes
# wrong in practice.
_ABS_HELP = {
    "Cell events:": "Events in the population you are counting, taken from "
                    "the gate you care about (not the whole sample).",
    "Bead events:": "Events in the counting-bead gate for the SAME tube. "
                    "Beads from another acquisition invalidate the ratio.",
    "Beads per tube:":
        "For a bead PELLET tube (e.g. Trucount): the bead count printed on "
        "the pouch for this lot. Leave blank for liquid beads and fill in "
        "concentration and volume instead.",
    "Bead concentration (beads/µL):":
        "Liquid beads: beads per microlitre for this bead LOT, from the vial "
        "insert. It differs between lots, so check rather than reusing a "
        "remembered number.",
    "Bead volume added (µL):":
        "Liquid beads: how much bead suspension went into the tube. "
        "Concentration × this volume is the number of beads in the tube.",
    "Sample volume (µL):":
        "How much sample (e.g. whole blood) went into the tube. The result "
        "is per µL of THIS volume, so it is required with a bead count per "
        "tube; for liquid beads, blank here and in bead volume means the "
        "two volumes are equal.",
    "Sample dilution factor:":
        "If the sample was diluted BEFORE it went into the tube, the factor "
        "(10 for 1 part in 10), so the result is per µL of the undiluted "
        "sample. 1 if it was not.",
}

_FIELDS = ("Cell events:", "Bead events:", "Beads per tube:",
           "Bead concentration (beads/µL):", "Bead volume added (µL):",
           "Sample volume (µL):", "Sample dilution factor:")


class AbsCountsDialog(tk.Toplevel):
    """Counting-bead absolute counts: cells/µL of the original sample from
    cell vs bead event counts, the beads in the tube (per-tube count, or
    concentration × volume added), the sample volume and its dilution."""

    def __init__(self, editor):
        super().__init__(editor)
        self.title("Absolute counts")
        self.geometry("520x430")
        self._editor = editor
        self._vars = {lbl: tk.StringVar() for lbl in _FIELDS}
        self._vars["Sample dilution factor:"].set("1")
        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=12, pady=10)
        ttk.Label(frm, justify='left', wraplength=490,
                  text="Counting-bead absolute count:\n"
                       "cells/µL = (cell events / bead events) × beads in "
                       "tube / sample volume (µL) × dilution factor.\n"
                       "Beads in tube = beads per tube (pellet), or bead "
                       "concentration × bead volume added (liquid beads)."
                  ).pack(anchor='w', pady=(0, 8))
        for lbl in _FIELDS:
            row = ttk.Frame(frm)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=lbl, width=30).pack(side='left')
            _e = ttk.Entry(row, textvariable=self._vars[lbl], width=14)
            _e.pack(side='left')
            auto_tip(_e, _ABS_HELP.get(lbl, ''))
        self._result = ttk.Label(frm, text="", font=('TkDefaultFont', 11, 'bold'))
        self._result.pack(anchor='w', pady=(10, 2))
        self._detail = ttk.Label(frm, text="", justify='left', wraplength=490)
        self._detail.pack(anchor='w')
        bar = ttk.Frame(frm)
        bar.pack(fill='x', side='bottom')
        _b = ttk.Button(bar, text="Compute", command=self._compute)
        _b.pack(side='left')
        auto_tip(_b,
                 "Work out cells/uL from the numbers above. The event counts "
                 "must come from the SAME acquisition — a bead count from a "
                 "different tube gives a confidently wrong answer.")
        ttk.Button(bar, text="Close", command=self.destroy).pack(side='right')

    def _num(self, lbl):
        """The field's number, or None when it is blank."""
        text = self._vars[lbl].get().strip().replace(',', '')
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{lbl.rstrip(':')} is not a number: "
                             f"{text!r}") from None

    def _compute(self):
        from .calibration import (
            absolute_count_from_known_beads,
            absolute_count_per_uL,
        )
        try:
            cells = self._num("Cell events:")
            beads = self._num("Bead events:")
            per_tube = self._num("Beads per tube:")
            conc = self._num("Bead concentration (beads/µL):")
            bead_vol = self._num("Bead volume added (µL):")
            sample_vol = self._num("Sample volume (µL):")
            dil = self._num("Sample dilution factor:")
            dil = 1.0 if dil is None else dil
            if cells is None or beads is None:
                raise ValueError("enter the cell and bead event counts")
            if per_tube is not None:
                if conc is not None or bead_vol is not None:
                    raise ValueError("give beads per tube OR bead "
                                     "concentration and volume, not both")
                if sample_vol is None:
                    raise ValueError("a bead count per tube needs the "
                                     "sample volume")
                val = absolute_count_from_known_beads(
                    cells, beads, per_tube, sample_vol, dil)
                how = (f"({cells:,.0f} / {beads:,.0f}) × {per_tube:,.0f} "
                       f"beads / {sample_vol:g} µL")
            else:
                if conc is None:
                    raise ValueError("enter beads per tube, or the bead "
                                     "concentration")
                val = absolute_count_per_uL(cells, beads, conc, bead_vol,
                                            sample_vol, dil)
                how = (f"({cells:,.0f} / {beads:,.0f}) × {conc:g} beads/µL"
                       + (f" × {bead_vol:g} µL / {sample_vol:g} µL"
                          if bead_vol is not None else
                          "  — bead and sample volumes not given, so taken "
                          "as EQUAL; if they differ, enter both"))
            if dil != 1.0:
                how += f" × dilution {dil:g}"
            self._result.configure(text=f"= {val:,.1f} cells/µL")
            self._detail.configure(text=how)
        except Exception as exc:
            self._result.configure(text=f"— {exc}")
            self._detail.configure(text="")
