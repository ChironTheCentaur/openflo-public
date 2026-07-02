"""ui_compare.py — GUI dialog for OpenFlo vs FlowJo workspace comparison.

A thin Tk front-end over the comparison backend in ``compare.py``. It lets
the user pick a FlowJo ``.wsp`` and an FCS directory, re-applies every
Population gate to the matching FCS data, and shows OpenFlo's per-population
event counts side-by-side with the counts FlowJo wrote into the workspace.

The heavy lifting (parsing, gating, counting) is NOT reimplemented here — we
import and call the helpers from :mod:`openflo.compare`:

* ``WspReader(wsp_path)``           — parse the workspace XML.
* ``_per_sample_inventory(reader)`` — ``[(sample_name, uri, pops), ...]``.
* ``_resolve_fcs_uri(uri, fcs_dir)``— map a DataSet uri onto a local path.
* ``_compare_one_sample(...)``      — list of row dicts with keys
  ``sample, population, flowjo_count, openflo_count, delta, rel_delta,
  total_events, error``.

``CompareWspDialog`` is a ``tk.Toplevel`` child of the gate-editor window.
The editor supplies ``status_var`` (tk.StringVar) and ``_begin_busy(msg)`` /
``_end_busy()``; the comparison runs on a background thread and marshals
results back via ``self.after(0, ...)``.
"""
import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .async_task import run_async

# Columns shown in the table, in order: (tree-column id, heading, width, anchor)
_COLUMNS = (
    ("sample", "Sample", 150, "w"),
    ("population", "Population", 170, "w"),
    ("flowjo_count", "FlowJo count", 100, "e"),
    ("openflo_count", "OpenFlo count", 100, "e"),
    ("delta", "Difference", 90, "e"),
    ("rel_delta", "Diff %", 80, "e"),
    ("total_events", "Total events", 100, "e"),
    ("error", "Error", 160, "w"),
)

# Relative-delta thresholds for flagging a row.
_WARN_REL = 0.01   # > 1 %  → warn (yellow)
_BAD_REL = 0.05    # > 5 %  → bad  (red)


class CompareWspDialog(tk.Toplevel):
    """Run + display a FlowJo ``.wsp`` vs OpenFlo gate-count comparison."""

    def __init__(self, editor):
        super().__init__(editor)
        self._editor = editor
        self.title("Compare FlowJo workspace")
        self.geometry("960x560")
        self.transient(editor)

        self._wsp_var = tk.StringVar()
        self._dir_var = tk.StringVar()
        self._rows = []          # last computed list of row dicts (for export)
        self._running = False

        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────────────
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=4)

        # Workspace picker row.
        wrow = ttk.Frame(top)
        wrow.pack(fill="x", pady=2)
        ttk.Label(wrow, text="FlowJo workspace (.wsp):", width=24).pack(side="left")
        ttk.Entry(wrow, textvariable=self._wsp_var).pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Button(wrow, text="Browse…", command=self._pick_wsp).pack(side="left")

        # FCS directory picker row.
        drow = ttk.Frame(top)
        drow.pack(fill="x", pady=2)
        ttk.Label(drow, text="FCS directory (optional):", width=24).pack(side="left")
        ttk.Entry(drow, textvariable=self._dir_var).pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Button(drow, text="Browse…", command=self._pick_dir).pack(side="left")

        ttk.Label(
            top, foreground="grey", justify="left", wraplength=900,
            text=("Re-applies every Population gate from the workspace to its "
                  "matching FCS and compares OpenFlo's event counts against the "
                  "counts FlowJo recorded. The FCS directory is used as a "
                  "fallback when a workspace's stored path does not resolve. "
                  "Rows over 5% apart are flagged red, over 1% yellow.")
        ).pack(anchor="w", pady=(6, 2))

        # Action bar.
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=4)
        self._run_btn = ttk.Button(bar, text="Run", command=self._run)
        self._run_btn.pack(side="left")
        self._export_btn = ttk.Button(
            bar, text="Export CSV…", command=self._export, state="disabled")
        self._export_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right")

        # Results table.
        tblfrm = ttk.Frame(self)
        tblfrm.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        cols = [c[0] for c in _COLUMNS]
        self._tree = ttk.Treeview(tblfrm, columns=cols, show="headings")
        for cid, heading, width, anchor in _COLUMNS:
            self._tree.heading(cid, text=heading)
            self._tree.column(cid, width=width, anchor=anchor,  # type: ignore[arg-type]
                              stretch=True)
        self._tree.tag_configure("bad", background="#ffe9e6")
        self._tree.tag_configure("warn", background="#fff7d9")
        ysb = ttk.Scrollbar(tblfrm, orient="vertical", command=self._tree.yview)
        xsb = ttk.Scrollbar(tblfrm, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        tblfrm.rowconfigure(0, weight=1)
        tblfrm.columnconfigure(0, weight=1)

        self._summary = ttk.Label(self, text="", foreground="grey")
        self._summary.pack(anchor="w", padx=10, pady=(0, 8))

    # ── File pickers ─────────────────────────────────────────────────────────
    def _pick_wsp(self):
        path = filedialog.askopenfilename(
            parent=self, title="Select a FlowJo workspace",
            filetypes=[("FlowJo workspace", "*.wsp"), ("All files", "*.*")])
        if path:
            self._wsp_var.set(path)
            # Default the FCS dir to the workspace's folder if unset.
            if not self._dir_var.get():
                self._dir_var.set(os.path.dirname(path))

    def _pick_dir(self):
        path = filedialog.askdirectory(
            parent=self, title="Select the FCS directory")
        if path:
            self._dir_var.set(path)

    # ── Run (background) ───────────────────────────────────────────────────────
    def _run(self):
        if self._running:
            return
        wsp = self._wsp_var.get().strip()
        fcs_dir = self._dir_var.get().strip()
        if not wsp:
            messagebox.showwarning(
                "Compare workspace", "Pick a FlowJo .wsp file first.", parent=self)
            return
        if not os.path.isfile(wsp):
            messagebox.showerror(
                "Compare workspace", f"Workspace not found:\n{wsp}", parent=self)
            return
        if fcs_dir and not os.path.isdir(fcs_dir):
            messagebox.showerror(
                "Compare workspace",
                f"FCS directory not found:\n{fcs_dir}", parent=self)
            return

        self._running = True
        self._run_btn.configure(state="disabled")
        self._export_btn.configure(state="disabled")
        for iid in self._tree.get_children():
            self._tree.delete(iid)
        self._summary.configure(text="")
        try:
            self._editor._begin_busy("Comparing workspace gates…")
        except Exception:
            pass

        run_async(self, lambda: self._worker(wsp, fcs_dir or None),
                  on_done=lambda res: self._on_done(*res),
                  on_error=self._on_error)

    def _worker(self, wsp, fcs_dir):
        """Background: parse the workspace and compare every sample. Returns
        ``(rows, skipped)``; a per-sample failure becomes an error row (the
        whole-run failure path is handled by run_async's on_error)."""
        from .compare import (
            WspReader,
            _compare_one_sample,
            _per_sample_inventory,
            _resolve_fcs_uri,
        )
        reader = WspReader(wsp)
        samples = _per_sample_inventory(reader)
        rows = []
        skipped = []
        for sample_name, uri, pops in samples:
            fcs = _resolve_fcs_uri(uri, fcs_dir)
            if fcs is None:
                skipped.append(sample_name)
                continue
            try:
                rows.extend(_compare_one_sample(
                    sample_name, fcs, pops, wsp_path=wsp))
            except Exception as exc:  # noqa: BLE001 — surface per-sample failure
                rows.append({
                    "sample": sample_name, "population": "(sample error)",
                    "flowjo_count": None, "openflo_count": None,
                    "delta": None, "rel_delta": None, "total_events": None,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return rows, skipped

    # ── Result handlers (Tk thread) ───────────────────────────────────────────
    def _on_done(self, rows, skipped):
        self._running = False
        self._run_btn.configure(state="normal")
        try:
            self._editor._end_busy()
        except Exception:
            pass

        self._rows = rows
        flagged = 0
        for r in rows:
            tag = self._row_tag(r)
            if tag:
                flagged += 1
            self._tree.insert("", "end", values=self._row_values(r),
                              tags=(tag,) if tag else ())

        self._export_btn.configure(state="normal" if rows else "disabled")
        parts = [f"{len(rows)} population row(s)"]
        if flagged:
            parts.append(f"{flagged} flagged")
        if skipped:
            parts.append(f"{len(skipped)} sample(s) skipped (FCS not found)")
        msg = "Comparison complete — " + ", ".join(parts) + "."
        self._summary.configure(text=msg)
        if self._safe_widget(self._editor, "status_var"):
            self._editor.status_var.set(msg)
        if not rows:
            messagebox.showinfo(
                "Compare workspace",
                "Nothing compared — no FCS resolved or no populations carried "
                "event counts. Try setting the FCS directory.", parent=self)

    def _on_error(self, exc):
        self._running = False
        self._run_btn.configure(state="normal")
        try:
            self._editor._end_busy()
        except Exception:
            pass
        msg = f"{type(exc).__name__}: {exc}"
        self._summary.configure(text=f"Failed: {msg}")
        messagebox.showerror("Compare workspace", msg, parent=self)

    # ── Row formatting ─────────────────────────────────────────────────────────
    @staticmethod
    def _row_tag(r):
        if r.get("error"):
            return "bad"
        rd = r.get("rel_delta")
        if rd is None:
            return ""
        ard = abs(rd)
        if ard > _BAD_REL:
            return "bad"
        if ard > _WARN_REL:
            return "warn"
        return ""

    @staticmethod
    def _row_values(r):
        fc = r.get("flowjo_count")
        oc = r.get("openflo_count")
        dl = r.get("delta")
        rl = r.get("rel_delta")
        te = r.get("total_events")
        return (
            r.get("sample", ""),
            r.get("population", ""),
            "" if fc is None else f"{fc:,}",
            "" if oc is None else f"{oc:,}",
            "" if dl is None else f"{dl:+,}",
            "" if rl is None else f"{rl * 100:+.2f}%",
            "" if te is None else f"{te:,}",
            r.get("error", "") or "",
        )

    @staticmethod
    def _safe_widget(obj, attr):
        try:
            return getattr(obj, attr, None) is not None
        except Exception:
            return False

    # ── CSV export ─────────────────────────────────────────────────────────────
    def _export(self):
        if not self._rows:
            messagebox.showinfo(
                "Export CSV", "Run a comparison first.", parent=self)
            return
        default = ""
        wsp = self._wsp_var.get().strip()
        if wsp:
            default = os.path.splitext(os.path.basename(wsp))[0] + "_compare.csv"
        path = filedialog.asksaveasfilename(
            parent=self, title="Export comparison as CSV",
            defaultextension=".csv", initialfile=default,
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        fields = ["sample", "population", "flowjo_count", "openflo_count",
                  "delta", "rel_delta", "total_events", "error"]
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in self._rows:
                    w.writerow({k: "" if r.get(k) is None else r.get(k)
                                for k in fields})
        except Exception as exc:  # noqa: BLE001 — report write failure to the UI
            messagebox.showerror("Export CSV", f"{type(exc).__name__}: {exc}",
                                 parent=self)
            return
        if self._safe_widget(self._editor, "status_var"):
            self._editor.status_var.set(f"Wrote {path}")
        messagebox.showinfo("Export CSV", f"Wrote:\n{path}", parent=self)
