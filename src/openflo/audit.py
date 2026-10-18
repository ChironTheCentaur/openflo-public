"""Provenance / audit trail for an analysis session.

An :class:`AuditLog` is an append-only, chronological record of the
meaningful operations performed on a set of samples — load, compensation,
transform, cleaning, gating, clustering, batch normalization, unmixing,
export. It is *complementary* to the session snapshot: the snapshot is the
current state, the audit trail is HOW that state was reached, in order, with
the parameters and key results needed to reproduce or document the analysis
(e.g. a methods section).

Pure / dependency-free (stdlib only) so it is fully unit-testable without Tk.
Each entry is a JSON-able dict::

    {'seq': int, 'time': iso-str|None, 'action': str, 'details': {...}}

``seq`` is a monotonic 1-based counter (stable ordering even if two events
share a timestamp); ``details`` holds action-specific parameters.
"""
from __future__ import annotations

import csv
import io


def _short(value, maxlen=80):
    """Compact one-line repr of a detail value for tables / text export."""
    if isinstance(value, float):
        s = f"{value:.4g}"
    elif isinstance(value, bool):
        s = "yes" if value else "no"
    elif isinstance(value, (list, tuple)):
        inner = ", ".join(_short(v, 24) for v in value[:8])
        if len(value) > 8:
            inner += f", +{len(value) - 8} more"
        s = f"[{inner}]"
    elif isinstance(value, dict):
        s = ", ".join(f"{k}={_short(v, 24)}" for k, v in list(value.items())[:8])
    else:
        s = str(value)
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) > maxlen:
        s = s[:maxlen - 1] + "…"
    return s


class AuditLog:
    """Append-only log of analysis operations. See module docstring."""

    def __init__(self, entries=None):
        self._entries = []
        self._seq = 0
        if entries:
            for e in entries:
                entry = {
                    'seq': int(e.get('seq', 0)) or (len(self._entries) + 1),
                    'time': e.get('time'),
                    'action': str(e.get('action', '')),
                    'details': dict(e.get('details') or {}),
                }
                self._entries.append(entry)
                self._seq = max(self._seq, entry['seq'])

    # ── recording ────────────────────────────────────────────────────────
    def record(self, action, time=None, details=None, **kw):
        """Append an entry. Extra keyword args merge into ``details`` (so
        ``log.record('gate.add', kind='polygon', sample='s1')`` works).
        Returns the stored entry dict."""
        self._seq += 1
        d = dict(details or {})
        d.update(kw)
        entry = {'seq': self._seq, 'time': time,
                 'action': str(action), 'details': d}
        self._entries.append(entry)
        return entry

    # ── access ───────────────────────────────────────────────────────────
    def entries(self):
        return [dict(e, details=dict(e['details'])) for e in self._entries]

    def __len__(self):
        return len(self._entries)

    def __bool__(self):
        return bool(self._entries)

    def clear(self):
        self._entries = []
        self._seq = 0

    # ── (de)serialization ────────────────────────────────────────────────
    def to_list(self):
        """JSON-able list of entries (for embedding in a session file)."""
        return [dict(e, details=dict(e['details'])) for e in self._entries]

    @classmethod
    def from_list(cls, data):
        return cls(entries=data or [])

    # ── export formats ───────────────────────────────────────────────────
    def to_text(self):
        """Plain-text log, one line per entry (for the in-app viewer)."""
        lines = []
        for e in self._entries:
            t = e.get('time') or '—'
            head = f"[{e['seq']:>3}] {t}  {e['action']}"
            det = self._format_details(e['details'])
            lines.append(f"{head}    {det}" if det else head)
        return "\n".join(lines)

    def to_markdown(self, title="OpenFlo analysis audit trail", meta=None):
        """Markdown report: a metadata header block + a chronological table.
        ``meta`` (dict) is rendered as a bullet list above the table — pass
        e.g. ``{'openflo_version': ..., 'exported': ...}``."""
        out = [f"# {title}", ""]
        if meta:
            for k, v in meta.items():
                out.append(f"- **{k}**: {_short(v, 200)}")
            out.append("")
        out.append(f"{len(self._entries)} recorded operation(s).")
        out.append("")
        out.append("| # | Time | Action | Details |")
        out.append("|---|------|--------|---------|")
        for e in self._entries:
            t = e.get('time') or ''
            det = self._format_details(e['details']).replace("|", "\\|")
            out.append(f"| {e['seq']} | {t} | `{e['action']}` | {det} |")
        out.append("")
        return "\n".join(out)

    def to_csv(self):
        """CSV with one row per entry; details flattened to ``k=v; k=v``."""
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['seq', 'time', 'action', 'details'])
        for e in self._entries:
            w.writerow([e['seq'], e.get('time') or '', e['action'],
                        self._format_details(e['details'], sep='; ')])
        return buf.getvalue()

    @staticmethod
    def _format_details(details, sep=", "):
        if not details:
            return ""
        return sep.join(f"{k}={_short(v)}" for k, v in details.items())
