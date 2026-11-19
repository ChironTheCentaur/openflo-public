"""Which channel labels a saved session set explicitly (a relabel, a
template's labels), rebuilt from its provenance trail. No GUI: the gate
editor and ``Document`` (the phone) both read sessions with it, so an FCS
export writes the same $PnS either way."""
from __future__ import annotations

# The source "Revert to the file's labels" (ui_vial_label) records: in a
# trail written before records said which channels are explicit, it is the
# one label change that takes channels out of the explicit set.
FILE_LABELS_SOURCE = "the file's labels"


def explicit_labels_from_trail(entries, saved):
    """``{channel: label}`` set explicitly in a saved session, rebuilt from
    its provenance trail (``entries``, AuditLog.to_list form) and the
    labels it saved (``saved``, the session's ``channel_labels``).

    The ``channel_labels.set`` records are replayed in order: a record's
    ``explicit`` says which of its channels are explicit after it (a
    relabel, a template's labels, an Undo or Redo); a record without one (a
    trail saved before it existed) makes its labels explicit, unless its
    source is "the file's labels", which takes them out. A channel counts
    only where the saved session still shows that label, so a change undone
    in a run that recorded no Undo is not taken for one."""
    out: dict[str, str] = {}
    for e in entries or ():
        if not isinstance(e, dict) or e.get('action') != 'channel_labels.set':
            continue
        det = e.get('details') or {}
        labels = det.get('labels') or {}
        if not isinstance(labels, dict):
            continue
        chosen = det.get('explicit')
        if not isinstance(chosen, dict):
            chosen = ({} if det.get('source') == FILE_LABELS_SOURCE
                      else labels)
        for ch in labels:
            if ch in chosen and chosen[ch] is not None:
                out[str(ch)] = str(chosen[ch])
            else:
                out.pop(str(ch), None)
    saved = saved or {}
    return {ch: lab for ch, lab in out.items()
            if str(saved.get(ch, '')) == lab}
