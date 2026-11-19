"""Label channels from vial labels: paste the printed text of the antibody
vials of a panel, check which channel each one labels, and set those labels
for the session (openflo.vial_label reads the text and picks the detector).

Self-contained Tk window (see ui_*.py convention). The labels go through the
editor's ``_set_channel_labels``, so one Undo in the main window reverses an
Apply or a Revert, and they are saved with the session.
"""
from __future__ import annotations

import re
import tkinter as tk
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from tkinter import ttk

from . import panel_check as pc
from . import vial_label as vl
from .channel_roles import is_excluded, is_scatter
from .fcs_text import keyword_value
from .theme import current_palette
from .tool_window import scrolled_body
from .ui_tips import tip

# How the pasted text splits into vials, as the window says it.
SPLIT_HINT = ("One vial per line, or one block per vial with a blank line "
              "between vials (for labels that run over several lines).")

# The detector box's entry for "no detector".
NO_DETECTOR = '(none)'

# A vial whose fluor is fine but whose detector the user must pick: no
# detector of this file says which laser and filter it reads.
PICK = 'pick'

# Wait this long after the last keystroke before reading the text again.
REFRESH_MS = 350

# The vial column shows this much of a label's first line.
VIAL_CHARS = 24

# A pulse measurement at the end of a detector name: 'FL1-A', 'FL1 H'.
_PULSE = re.compile(r'^(.*?\S)[\s_-]+(A|H|W)$', re.IGNORECASE)


def split_vials(text: str) -> list[str]:
    """The vial labels in pasted `text`: blocks separated by blank lines
    when a blank line sits between two filled ones (a label printed over
    several lines), else one per non-empty line. Each label is stripped
    line by line."""
    lines = [ln.strip() for ln in str(text).splitlines()]
    filled = [i for i, ln in enumerate(lines) if ln]
    if not filled:
        return []
    if '' not in lines[filled[0]:filled[-1]]:
        return [lines[i] for i in filled]
    blocks: list[str] = []
    cur: list[str] = []
    for ln in lines[filled[0]:filled[-1] + 1] + ['']:
        if ln:
            cur.append(ln)
        elif cur:
            blocks.append('\n'.join(cur))
            cur = []
    return blocks


def parameter_names(metadata: Mapping[str, object]) -> list[str]:
    """The file's detector names ($PnN) in parameter order, or [] when its
    keywords do not list them."""
    try:
        n = int(keyword_value(metadata, '$PAR') or 0)
    except ValueError:
        return []
    names = [keyword_value(metadata, f'$P{i}N') for i in range(1, n + 1)]
    return [str(x) for x in names] if all(names) else []


def file_labels(sample) -> dict[str, str]:
    """``{detector: label}`` as the sample's own file has them: its $PnS,
    else (no keywords, a sample rebuilt from data) the labels it holds.
    When two parameters share one $PnN the first one's $PnS is kept: the
    loader keeps that name for the first and renames the second ('FL1-A
    (2)', flow_sample), whose label the sample itself holds."""
    own = dict(getattr(sample, 'channel_labels', None) or {})
    meta = getattr(sample, 'metadata', None) or {}
    seen: set[str] = set()
    for i, name in enumerate(parameter_names(meta), 1):
        if name not in seen:
            seen.add(name)
            own[name] = keyword_value(meta, f'$P{i}S') or name
    return own


def fluorescence_channels(sample) -> list[str]:
    """The sample's fluorescence channels: its own list, else every column
    that is not scatter, time or a derived result."""
    chans = list(getattr(sample, 'fluor_channels', None) or [])
    if chans:
        return [str(c) for c in chans]
    data = getattr(sample, 'data', None)
    cols = [] if data is None else [str(c) for c in data.columns]
    return [c for c in cols if not is_scatter(c) and not is_excluded(c)]


def sample_detectors(sample, channels: Sequence[str]) -> vl.DetectorNames:
    """The detectors of `channels` whose laser and filter are known, from
    their names or the file's $PnL / $PnF (vial_label.fcs_optics)."""
    meta = getattr(sample, 'metadata', None) or {}
    names = parameter_names(meta)
    lasers, filters = vl.fcs_optics(meta, names) if names else ({}, {})
    return vl.detectors_from_names(channels, lasers, filters)


def twins(det: str, channels: Sequence[str],
          detectors: Sequence[pc.Detector]) -> list[str]:
    """`det` and the channels that read the same light: the same laser and
    filter in `detectors`, or the same name with another pulse measurement
    ('FL1-A', 'FL1-H')."""
    optics = next((d[1:] for d in detectors if d[0] == det), None)
    m = _PULSE.match(det)
    base = m.group(1).upper() if m else None
    out = []
    for c in channels:
        cm = _PULSE.match(c)
        same_name = base is not None and cm is not None and \
            cm.group(1).upper() == base
        same_optics = optics is not None and any(
            d[0] == c and d[1:] == optics for d in detectors)
        if c == det or same_name or same_optics:
            out.append(c)
    return out


def generic_hint(fluor: str, set_name: str) -> str | None:
    """Where `fluor` is read on the generic instrument `set_name`
    (panel_check.DETECTOR_SETS): 'centre/width on the N nm laser', or None
    when nothing there reads it."""
    dets = pc.DETECTOR_SETS[set_name]['detectors']
    pick = vl.primary_detector(fluor, dets)
    found = next((d for d in dets if d[0] == pick.primary), None)
    if found is None:
        return None
    _n, laser, centre, width = found
    return f"{centre:g}/{width:g} on the {laser:g} nm laser"


@dataclass
class VialRow:
    """One vial of the preview: its text, what it says, the detector the
    model picks, and the user's marker and detector boxes."""
    text: str
    label: vl.VialLabel
    pick: vl.DetectorPick | None
    marker_var: tk.StringVar
    detector_var: tk.StringVar
    status: str = ''
    status_text: str = ''
    group: str = ''
    widgets: dict = field(default_factory=dict)

    @property
    def detector(self) -> str:
        """The detector chosen, '' for none."""
        d = self.detector_var.get()
        return '' if d == NO_DETECTOR else d

    @property
    def marker(self) -> str:
        """The marker typed or read, stripped."""
        return self.marker_var.get().strip()


def first_status(row: VialRow, modelled: bool) -> str:
    """A row's status before two rows on one detector are counted:
    vial_label's PLACED, NO_MARKER, NO_FLUOR, UNKNOWN_FLUOR, UNREADABLE, or
    PICK. `modelled` says whether any detector of the file is modelled."""
    if not row.marker:
        return vl.NO_MARKER
    if row.detector:
        return vl.PLACED
    if row.label.fluor is None:
        return vl.UNKNOWN_FLUOR if row.label.fluor_text else vl.NO_FLUOR
    if modelled and (row.pick is None or row.pick.primary is None):
        return vl.UNREADABLE
    return PICK


def check_rows(rows: Sequence[VialRow], channels: Sequence[str],
               detectors: Sequence[pc.Detector], generic_set: str | None
               ) -> None:
    """Set each row's ``status`` and ``status_text``: two placed rows on
    one detector (or on its twins) are both a CONFLICT, each offered its
    runner-up."""
    modelled = bool(detectors)
    for r in rows:
        r.status = first_status(r, modelled)
        r.group = (min(twins(r.detector, channels, detectors))
                   if r.detector else '')
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r.status == vl.PLACED:
            groups.setdefault(r.group, []).append(i)
    for ix in groups.values():
        if len(ix) > 1:
            for i in ix:
                rows[i].status = vl.CONFLICT
    for i, r in enumerate(rows):
        r.status_text = _status_text(i, r, rows, generic_set)


def _status_text(i: int, row: VialRow, rows: Sequence[VialRow],
                 generic_set: str | None) -> str:
    """What the status column says for `row` (index `i`)."""
    s = row.status
    if s == vl.PLACED:
        return "placed"
    if s == vl.CONFLICT:
        others = [str(j + 1) for j, r in enumerate(rows)
                  if j != i and r.status == vl.CONFLICT
                  and r.group == row.group]
        alt = row.pick.runner_up if row.pick is not None else None
        tail = (f"try {alt}" if alt and alt != row.detector
                else "pick another detector")
        with_ = f" with vial {', '.join(others)}" if others else ""
        return f"conflict: shares {row.detector}{with_}; {tail}"
    if s == vl.NO_MARKER:
        return "no marker: type it in"
    if s == vl.NO_FLUOR:
        return "no fluor found: pick the detector by hand"
    if s == vl.UNKNOWN_FLUOR:
        return "fluor not in catalog: pick the detector by hand"
    if s == vl.UNREADABLE:
        return "not readable on this instrument: pick a detector by hand"
    hint = (generic_hint(row.label.fluor, generic_set)
            if generic_set and row.label.fluor else None)
    return (f"pick a detector (a generic instrument reads it at {hint})"
            if hint else "pick a detector")


def _short(text: str) -> str:
    """A vial's first line, cut to VIAL_CHARS."""
    first = text.splitlines()[0] if text else ''
    return first if len(first) <= VIAL_CHARS else \
        first[:VIAL_CHARS - 1] + '…'


class VialLabelWindow(tk.Toplevel):
    """Paste vial labels, check the preview (marker, fluor, detector,
    status per vial; marker and detector can be changed), then Apply the
    labels to the session's channels, or Revert to the file's own."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Label channels from vial labels")
        self.geometry("860x560")
        self.minsize(640, 360)
        self.rows: list[VialRow] = []
        # The user's own marker / detector per vial text, kept while the
        # text is edited around it.
        self._kept: dict[str, tuple[str, str]] = {}
        self._pending = None
        name = getattr(editor, '_active_sample', None)
        self.sample = (editor._samples.get(name)
                       if name is not None else None)
        self.sample_name = name
        self.message_var = tk.StringVar(master=self, value='')
        if self.sample is None:
            self._build_no_sample()
            return
        self.channels = fluorescence_channels(self.sample)
        found = sample_detectors(self.sample, self.channels)
        self.detectors = found.detectors
        self.unmatched = found.unmatched
        self._build()

    # ── Layout ──────────────────────────────────────────────────────────

    def _build_no_sample(self):
        """No sample to label: say what to do, and nothing else."""
        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        self.message_var.set(
            "No sample is loaded. Channel labels are set on a sample's "
            "channels: add an FCS file, select it, then open this window "
            "again.")
        ttk.Label(body, textvariable=self.message_var, wraplength=420,
                  justify='left').pack(side='top', anchor='w')
        ttk.Button(body, text="Close", command=self.destroy).pack(
            side='bottom', anchor='e', pady=(12, 0))
        self.geometry("460x160")

    def _build(self):
        """The text box, the instrument line, the preview table and the
        button row."""
        ed = self.editor
        top = ttk.Frame(self, padding=(8, 8, 8, 2))
        top.pack(side='top', fill='x')
        ttk.Label(top, text="Paste the text printed on each vial. "
                  + SPLIT_HINT, wraplength=700, justify='left').pack(
            side='top', anchor='w')
        self.text = tk.Text(top, height=6, width=60, wrap='word', undo=True)
        self.text.pack(side='top', fill='x', pady=(4, 2))
        tip(ed, self.text,
            "The vial's printed name, marker and fluor (catalog, lot and "
            "clone lines are fine). Each line is one vial; leave a blank "
            "line between vials when a label runs over several lines.")
        self.text.bind('<<Modified>>', self._on_text_modified)

        inst = ttk.Frame(self, padding=(8, 2, 8, 2))
        inst.pack(side='top', fill='x')
        ttk.Label(inst, text=self._detector_summary(), wraplength=700,
                  justify='left', style='Muted.TLabel').pack(
            side='top', anchor='w')
        self.generic_var = tk.StringVar(master=self, value='')
        if not self.detectors:
            row = ttk.Frame(inst)
            row.pack(side='top', fill='x', pady=(2, 0))
            ttk.Label(row, text="Suggest from a generic instrument:").pack(
                side='left')
            default = next((k for k in pc.DETECTOR_SETS if '4-laser' in k),
                           pc.DEFAULT_SET)
            self.generic_var.set(default)
            box = ttk.Combobox(row, textvariable=self.generic_var,
                               state='readonly', width=56,
                               values=list(pc.DETECTOR_SETS))
            box.pack(side='left', padx=(4, 0))
            box.bind('<<ComboboxSelected>>', lambda _e: self.recheck())
            tip(ed, box,
                "Only a hint: which laser and filter a fluor needs on a "
                "typical instrument, shown in the status column. The "
                "channel names of this file do not say, so pick each "
                "vial's real channel in its Detector box.")

        btns = ttk.Frame(self, padding=(8, 4, 8, 8))
        btns.pack(side='bottom', fill='x')
        self.revert_btn = ttk.Button(btns, text="Revert to the file's labels",
                                     command=self.revert)
        self.revert_btn.pack(side='left')
        tip(ed, self.revert_btn,
            "Put back the labels this sample's file gives its channels "
            "($PnS), dropping labels set here or by a template. Undo in "
            "the main window reverses it.")
        ttk.Button(btns, text="Close", command=self.destroy).pack(
            side='right')
        self.apply_btn = ttk.Button(btns, text="Apply", default='active',
                                    command=self.apply)
        self.apply_btn.pack(side='right', padx=(0, 6))
        tip(ed, self.apply_btn,
            "Label each placed vial's channel (and its -A / -H / -W twins) "
            "with its marker, for this session: plots, axis and gate names "
            "and statistics use it, and it is saved with the session. "
            "Vials marked with a problem are left out. Undo in the main window "
            "reverses it.")
        ttk.Label(self, textvariable=self.message_var, wraplength=700,
                  justify='left', padding=(8, 0, 8, 2)).pack(
            side='bottom', fill='x')

        self.table = scrolled_body(self, padding=(8, 2, 8, 2))
        self._draw_rows()

    def _detector_summary(self) -> str:
        """One line on the detectors the vials are matched to."""
        n, k = len(self.channels), len(self.detectors)
        head = (f"Sample '{self.sample_name}': {n} fluorescence "
                f"channel{'s' if n != 1 else ''}")
        if not n:
            return head + ". There is nothing to label."
        if not k:
            return (head + "; their names and the file's keywords do not "
                    "say which laser and filter each reads, so pick each "
                    "vial's channel by hand.")
        if self.unmatched:
            return (head + f"; the detector is picked for you on the {k} "
                    "whose laser and filter are known. Others: "
                    + ', '.join(self.unmatched) + " (pick by hand).")
        return head + "; the detector is picked from each one's laser and " \
                      "filter."

    # ── Reading the text ────────────────────────────────────────────────

    def _on_text_modified(self, _event=None):
        """Re-read the text once typing pauses."""
        if self.sample is None:             # no text box without a sample
            return
        try:
            if not self.text.edit_modified():
                return
            self.text.edit_modified(False)
            if self._pending is not None:
                self.after_cancel(self._pending)
            self._pending = self.after(REFRESH_MS, self.refresh)
        except tk.TclError:
            pass

    def refresh(self):
        """Read the pasted text again and rebuild the preview, keeping the
        marker and detector the user changed on vials still there."""
        self._pending = None
        for r in self.rows:
            self._remember(r)
        texts = split_vials(self.text.get('1.0', 'end'))
        self.rows = [self._make_row(t) for t in texts]
        self._draw_rows()

    def _make_row(self, text: str) -> VialRow:
        """A preview row for one vial's text, its detector pre-filled with
        the model's pick."""
        label = vl.parse_vial_label(text)
        pick = (vl.primary_detector(label.fluor, self.detectors)
                if label.fluor and self.detectors else None)
        det = pick.primary if pick is not None and pick.primary else ''
        marker = label.marker or ''
        if text in self._kept:
            marker, det = self._kept[text]
        return VialRow(text, label, pick,
                       tk.StringVar(master=self, value=marker),
                       tk.StringVar(master=self, value=det or NO_DETECTOR))

    def destroy(self):
        """Close, dropping a re-read still waiting for typing to pause."""
        if self._pending is not None:
            try:
                self.after_cancel(self._pending)
            except tk.TclError:
                pass
            self._pending = None
        super().destroy()

    def _remember(self, row: VialRow):
        """Keep a row's marker and detector across a re-read."""
        self._kept[row.text] = (row.marker_var.get(), row.detector)

    # ── The preview table ───────────────────────────────────────────────

    def _draw_rows(self):
        """Rebuild the table: a header, then one line of widgets per vial."""
        if self.sample is None:             # no table without a sample
            return
        for w in self.table.winfo_children():
            w.destroy()
        heads = ("#", "Vial", "Marker", "Fluor", "Detector", "Status")
        for c, h in enumerate(heads):
            ttk.Label(self.table, text=h, style='Muted.TLabel').grid(
                row=0, column=c, sticky='w', padx=(0, 6))
        if not self.rows:
            ttk.Label(self.table, text="Paste vial labels above to see "
                      "which channel each one labels.",
                      style='Muted.TLabel').grid(row=1, column=0,
                                                 columnspan=6, sticky='w')
        values = [NO_DETECTOR] + list(self.channels)
        for i, r in enumerate(self.rows, 1):
            ttk.Label(self.table, text=str(i)).grid(row=i, column=0,
                                                   sticky='w')
            vial = ttk.Label(self.table, text=_short(r.text), width=VIAL_CHARS)
            vial.grid(row=i, column=1, sticky='w', padx=(0, 6))
            marker = ttk.Entry(self.table, textvariable=r.marker_var, width=12)
            marker.grid(row=i, column=2, sticky='w', padx=(0, 6))
            tip(self.editor, marker,
                "The marker this vial's channel is labeled with. Correct "
                "it here if the label was read wrong.")
            r.marker_var.trace_add('write', lambda *_a: self.recheck())
            fluor = r.label.fluor or (f"? {r.label.fluor_text}"
                                      if r.label.fluor_text else "—")
            ttk.Label(self.table, text=fluor, width=-12).grid(
                row=i, column=3, sticky='w', padx=(0, 6))
            width = max([len(v) for v in values] + [10]) + 2
            box = ttk.Combobox(self.table, textvariable=r.detector_var,
                               values=values, state='readonly', width=width)
            box.grid(row=i, column=4, sticky='w', padx=(0, 6))
            box.bind('<<ComboboxSelected>>', lambda _e: self.recheck())
            tip(self.editor, box,
                "The channel this vial is read on, picked from the fluor's "
                "laser and filter where the file says them. Choose another "
                "if it is wrong, or (none) to leave this vial out.")
            status = ttk.Label(self.table, text="", wraplength=210,
                               justify='left')
            status.grid(row=i, column=5, sticky='w')
            r.widgets = {'vial': vial, 'marker': marker, 'detector': box,
                         'status': status}
        self.recheck()

    def recheck(self):
        """Check every row again (after an edit) and show its status; a
        row with a problem is marked and coloured."""
        check_rows(self.rows, self.channels, self.detectors,
                   self.generic_var.get() or None)
        pal = current_palette()
        for r in self.rows:
            w = r.widgets.get('status')
            if w is None:
                continue
            ok = r.status == vl.PLACED
            w.configure(text=r.status_text if ok else f"⚠ {r.status_text}",
                        foreground=pal['success'] if ok else pal['warning'])
        n_ok = sum(r.status == vl.PLACED for r in self.rows)
        n_bad = len(self.rows) - n_ok
        if self.rows:
            self.message_var.set(
                f"{n_ok} of {len(self.rows)} vial(s) ready to apply"
                + (f"; {n_bad} marked ⚠ need you first and are left out."
                   if n_bad else "."))

    # ── Apply / revert ──────────────────────────────────────────────────

    def planned_labels(self) -> dict[str, str]:
        """``{channel: marker}`` the placed vials give, each vial's channel
        and its twins."""
        out: dict[str, str] = {}
        for r in self.rows:
            if r.status != vl.PLACED:
                continue
            for d in twins(r.detector, self.channels, self.detectors):
                out[d] = r.marker
        return out

    def apply(self):
        """Set the placed vials' labels for the session (one Undo step)."""
        self.recheck()
        changed = self.editor._set_channel_labels(self.planned_labels(),
                                                  'vial labels')
        self._report(changed, "Labeled")

    def revert(self):
        """Put the file's own labels back on this sample's channels (one
        Undo step) and take them out of the session's explicit labels, so
        an FCS export gives every file its own name there again."""
        from .channel_label_trail import FILE_LABELS_SOURCE
        own = file_labels(self.sample)
        changed = self.editor._set_channel_labels(
            {d: own.get(d, d) for d in own}, FILE_LABELS_SOURCE,
            explicit=False)
        self._report(changed, "Reverted")

    def _report(self, changed: Mapping[str, str], verb: str):
        """Say what Apply or Revert changed, here and in the status bar."""
        if not changed:
            msg = "Nothing to change: the channels already have these labels."
        else:
            msg = (f"{verb} {len(changed)} channel(s): "
                   + ', '.join(f"{d} = {v}" for d, v in changed.items())
                   + ". Undo in the main window reverses it.")
        self.message_var.set(msg)
        try:
            self.editor.status_var.set(msg)
        except (AttributeError, tk.TclError):
            pass
