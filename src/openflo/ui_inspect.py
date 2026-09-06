"""Raw-FCS metadata inspector dialog.

A lightweight, theme-aware ``tk.Toplevel`` that lets the user pick an FCS file
and view its metadata only — channel table ($PnN / $PnS / $PnV / gain), the raw
TEXT keywords (including any spillover / $SPILL matrix), and a summary line.

The heavy event matrix is never materialised: :class:`flowio.FlowData` parses
the TEXT segment eagerly but exposes the event array lazily, so reading
``.channels`` / ``.text`` / ``.event_count`` is cheap. This mirrors what the
``openflo.inspect_fcs`` script prints, factored into a reusable helper.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

import flowio


def read_fcs_metadata(path: str) -> dict[str, Any]:
    """Read an FCS file's metadata without loading the full event matrix.

    Returns a dict with:
      ``filename``     basename of the file,
      ``event_count``  number of events (int) or ``None`` if unavailable,
      ``channel_count`` number of channels (int),
      ``channels``     list of per-channel dicts with keys ``index`` (1-based),
                       ``pnn`` (detector / $PnN), ``pns`` (label / $PnS),
                       ``pnv`` (voltage / $PnV or ''), ``gain`` ($PnG or ''),
      ``text``         dict of raw FCS TEXT keywords (str -> str),
      ``spillover``    the spillover/$SPILL/$COMP keyword value (str) or ''.

    Mirrors the logic in ``openflo.inspect_fcs``: ``flowio.FlowData`` exposes
    ``.channels`` keyed by 1-based int with lowercase ``pnn``/``pns`` fields,
    ``.text`` (TEXT keywords) and ``.event_count``.
    """
    f = flowio.FlowData(path)

    text: dict[str, str] = {str(k): str(v) for k, v in f.text.items()}

    # Per-channel $PnV (voltage) / $PnG (gain) come from TEXT, keyed by the
    # 1-based channel number; FlowData's .channels carries pnn/pns.
    channels: list[dict[str, Any]] = []
    for key, ch in sorted(f.channels.items(), key=lambda kv: int(kv[0])):
        idx = int(key)

        def _text(field: str, _idx: int = idx) -> str:
            # FCS keywords may appear with or without the leading '$'.
            for cand in (f"$P{_idx}{field}", f"P{_idx}{field}"):
                if cand in text:
                    return text[cand]
            return ""

        channels.append(
            {
                "index": idx,
                "pnn": str(ch.get("pnn", "") if isinstance(ch, dict) else ch),
                "pns": str(ch.get("pns", "")) if isinstance(ch, dict) else "",
                "pnv": _text("V"),
                "gain": _text("G"),
            }
        )

    spillover = ""
    for k in text:
        kl = k.lower().lstrip("$")
        if kl in ("spillover", "spill", "comp"):
            spillover = text[k]
            break

    event_count = getattr(f, "event_count", None)
    try:
        event_count = int(event_count) if event_count is not None else None
    except (TypeError, ValueError):
        event_count = None

    return {
        "filename": os.path.basename(path),
        "event_count": event_count,
        "channel_count": len(channels),
        "channels": channels,
        "text": text,
        "spillover": spillover,
    }


class FcsInspectorDialog(tk.Toplevel):
    """Pick an FCS file and inspect its raw metadata (channels + TEXT keywords).

    Construct with the editor window so we inherit the app theme (the option DB
    already themes tk Text/Canvas) and can post to ``editor.status_var``.
    """

    def __init__(self, editor: tk.Misc) -> None:
        super().__init__(editor)
        self.editor = editor
        self.title("FCS metadata inspector")
        self.minsize(720, 480)
        try:
            self.geometry("860x620")
        except Exception:
            pass

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────
    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(side="top", fill="x")
        ttk.Button(top, text="Pick FCS…", command=self._pick).pack(
            side="left"
        )
        self.summary_var = tk.StringVar(value="No file loaded.")
        ttk.Label(top, textvariable=self.summary_var).pack(
            side="left", padx=(10, 0)
        )

        body = ttk.Panedwindow(self, orient="vertical")
        body.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

        # Channel table.
        chan_frame = ttk.Labelframe(body, text="Channels", padding=4)
        cols = ("index", "pnn", "pns", "pnv", "gain")
        headings = {
            "index": "#",
            "pnn": "$PnN (detector)",
            "pns": "$PnS (label)",
            "pnv": "$PnV (voltage)",
            "gain": "$PnG (gain)",
        }
        self.chan_tv = ttk.Treeview(
            chan_frame, columns=cols, show="headings", height=10
        )
        for c in cols:
            self.chan_tv.heading(c, text=headings[c])
            self.chan_tv.column(
                c, width=60 if c == "index" else 150, anchor="w", stretch=True
            )
        chan_sb = ttk.Scrollbar(
            chan_frame, orient="vertical", command=self.chan_tv.yview
        )
        self.chan_tv.configure(yscrollcommand=chan_sb.set)
        self.chan_tv.pack(side="left", fill="both", expand=True)
        chan_sb.pack(side="right", fill="y")
        body.add(chan_frame, weight=1)

        # Raw TEXT keywords table.
        kw_frame = ttk.Labelframe(
            body, text="TEXT keywords (incl. spillover/$SPILL)", padding=4
        )
        self.kw_tv = ttk.Treeview(
            kw_frame, columns=("key", "value"), show="headings", height=10
        )
        self.kw_tv.heading("key", text="Key")
        self.kw_tv.heading("value", text="Value")
        self.kw_tv.column("key", width=200, anchor="w", stretch=False)
        self.kw_tv.column("value", width=560, anchor="w", stretch=True)
        kw_sb = ttk.Scrollbar(
            kw_frame, orient="vertical", command=self.kw_tv.yview
        )
        kw_sbx = ttk.Scrollbar(
            kw_frame, orient="horizontal", command=self.kw_tv.xview
        )
        self.kw_tv.configure(
            yscrollcommand=kw_sb.set, xscrollcommand=kw_sbx.set
        )
        self.kw_tv.grid(row=0, column=0, sticky="nsew")
        kw_sb.grid(row=0, column=1, sticky="ns")
        kw_sbx.grid(row=1, column=0, sticky="ew")
        kw_frame.rowconfigure(0, weight=1)
        kw_frame.columnconfigure(0, weight=1)
        body.add(kw_frame, weight=1)

        btnbar = ttk.Frame(self, padding=(8, 0, 8, 8))
        btnbar.pack(side="bottom", fill="x")
        ttk.Button(btnbar, text="Close", command=self.destroy).pack(
            side="right"
        )

    # ── status helper ────────────────────────────────────────────────────
    def _set_status(self, msg: str) -> None:
        sv = getattr(self.editor, "status_var", None)
        if sv is not None:
            try:
                sv.set(msg)
            except Exception:
                pass

    # ── actions ──────────────────────────────────────────────────────────
    def _pick(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Pick an FCS file",
            filetypes=[("FCS files", "*.fcs"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load(path)

    def _load(self, path: str) -> None:
        try:
            meta = read_fcs_metadata(path)
        except Exception as exc:
            self.summary_var.set(f"Could not read {os.path.basename(path)}: {exc}")
            self._clear_tables()
            self._set_status(f"FCS inspect failed: {exc}")
            return

        ev = meta["event_count"]
        ev_txt = f"{ev:,}" if isinstance(ev, int) else "?"
        self.summary_var.set(
            f"{meta['filename']}  •  {ev_txt} events  •  "
            f"{meta['channel_count']} channels"
        )

        self._clear_tables()
        for ch in meta["channels"]:
            self.chan_tv.insert(
                "",
                "end",
                values=(
                    ch["index"],
                    ch["pnn"],
                    ch["pns"],
                    ch["pnv"],
                    ch["gain"],
                ),
            )
        for k, v in sorted(meta["text"].items()):
            self.kw_tv.insert("", "end", values=(k, v))

        self._set_status(
            f"Inspected {meta['filename']}: {meta['channel_count']} channels, "
            f"{ev_txt} events."
        )

    def _clear_tables(self) -> None:
        for tv in (self.chan_tv, self.kw_tv):
            for iid in tv.get_children():
                tv.delete(iid)
