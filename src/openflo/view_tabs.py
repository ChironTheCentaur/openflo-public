"""Tabs of the Samples & Gates tray (beta) -- Tk-free, headless-testable.

A tab is a saved view: which samples (and, optionally, which of their gates)
it shows, which of them are switched on, and how the plot looked -- axes,
plot type, colouring, axis types and zoom. Selecting a tab shows only its
members and restores its plot. The first tab, "All", always holds every
sample; other tabs hold what was copied into them.

Each tab owns its gates: a gate drawn, edited or deleted in one tab does not
touch any other tab. The editor works on the active tab's gates (its live
``_sample_gates``); every other tab keeps its own store here (``gates``,
``gate_order``), handed back when it is selected again. Copying rows to a tab
copies their gates into its store. A session saves each tab's gates; one
saved before tabs kept their own gives every tab a copy of its shared gates.
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .gate_lineage import descendants, lineage

ALL_TAB = 'All'


def gate_lists(gates_by_sample: Mapping[str, Mapping],
               order_by_sample: Mapping[str, list]) -> dict:
    """Gates as a session saves them: ``{sample: [gate carrying its 'id',
    ...]}`` in tree order (each carries its ``parent_id``, so the hierarchy
    restores)."""
    out = {}
    for name, gates in gates_by_sample.items():
        order = order_by_sample.get(name, list(gates))
        out[name] = [dict(g, id=gid) for gid in order
                     if (g := gates.get(gid)) is not None]
    return out


class ViewTab:
    """One tray tab: its member samples and gates, their on/off states and its saved view.

    ``members`` None makes it the All tab, which shows every sample."""
    def __init__(self, name, members=None, plot_enabled=None,
                 gate_enabled=None, view=None, selected=None, order=None,
                 presort=None):
        self.name = str(name)
        # None: every sample (the All tab). Else {sample: None | [gid, ...]},
        # None meaning every gate of that sample.
        self.members: dict | None = (
            None if members is None
            else {str(s): (None if g is None else list(g))
                  for s, g in members.items()})
        self.plot_enabled: dict = dict(plot_enabled or {})
        self.gate_enabled: dict = {s: dict(v) for s, v in
                                   (gate_enabled or {}).items()}
        self.view: dict = dict(view or {})
        self.selected = selected
        # This tab's own sample order (empty: the editor's order), and the
        # rows dropped on it that wait in its Presort zone to be placed.
        self.order: list = [str(s) for s in order or ()]
        self.presort: list = [(str(s), g) for s, g in presort or ()]
        # This tab's own gates while another tab is active: {sample: {gid:
        # gate}} and {sample: [gid, ...]}. None while it is the active tab,
        # whose gates are the editor's live ones.
        self.gates: dict | None = {}
        self.gate_order: dict | None = {}
        # Gates read from a session for samples that have not arrived yet:
        # {sample: [gate dict carrying its saved 'id', ...]}.
        self.saved_gates: dict = {}

    @property
    def is_all(self):
        """True for the All tab, which holds every sample."""
        return self.members is None

    def has_sample(self, sample):
        """Whether the tab shows ``sample`` (always True for the All tab)."""
        return self.is_all or sample in (self.members or {})

    def samples(self, order: Iterable[str]):
        """This tab's samples, in its own order first, then the editor's."""
        order = list(order)
        mine = [s for s in self.order if s in order and self.has_sample(s)]
        return mine + [s for s in order
                       if self.has_sample(s) and s not in mine]

    def is_grouped(self, trial_of, order: Iterable[str]):
        """Whether the tab is arranged: its samples span groups (trials),
        or it has its own order. Rows dropped on an arranged tab wait in its
        Presort zone; otherwise they go to the top of its stack."""
        names = self.samples(order)
        return bool(self.order) or len({trial_of(s) for s in names}) > 1

    def drop(self, rows, grouped):
        """Rows ``[(sample, gid|None), ...]`` dropped on this tab: into
        the Presort zone when ``grouped``, else added and moved to the top
        of the stack. Returns 'presort' or 'top'."""
        if grouped:
            for r in rows:
                r = (str(r[0]), r[1])
                if r not in self.presort:
                    self.presort.append(r)
            return 'presort'
        self.place(rows, before=None, top=True)
        return 'top'

    def place(self, rows, before=None, top=False):
        """Make rows members, taking them out of the Presort zone. Their
        samples go before sample ``before`` in this tab's order, or to the
        top when ``top`` (else they keep their place)."""
        new = []
        for s, g in rows:
            s = str(s)
            self.add(s, g)
            if (s, g) in self.presort:
                self.presort.remove((s, g))
            if s not in new:
                new.append(s)
        if top or before is not None:
            rest = [s for s in self.order if s not in new]
            if top:
                self.order = new + rest
            else:
                i = rest.index(before) if before in rest else len(rest)
                self.order = rest[:i] + new + rest[i:]

    def shown_gates(self, sample, gates: Mapping[str, Mapping]):
        """Gate ids shown for ``sample``: every gate, or the chosen ones plus
        their ancestors (a gate can't be shown without its parents) and
        their descendants (a copied gate brings its subtree)."""
        if not self.has_sample(sample):
            return set()
        chosen = None if self.is_all else (self.members or {}).get(sample)
        if chosen is None:
            return set(gates)
        keep = set()
        for gid in chosen:
            keep.update(lineage(gates, gid))
            keep.update(descendants(gates, gid))
        return keep

    def add(self, sample, gid=None):
        """Put a sample (``gid`` None: with every gate) or one of its gates
        in this tab. The All tab already holds everything."""
        if self.is_all:
            return
        members = self.members if self.members is not None else {}
        self.members = members
        if gid is None:
            members[sample] = None
        else:
            cur = members.get(sample, [])
            if cur is None:
                return                      # already every gate
            if gid not in cur:
                cur.append(gid)
            members[sample] = cur

    def remove(self, sample, gid=None):
        """Remove ``sample`` from the tab, or only its gate ``gid`` when given.

        Removing a sample's last listed gate removes the sample. No effect on the All tab."""
        if self.is_all or self.members is None:
            return
        if gid is None:
            self.members.pop(sample, None)
            self.plot_enabled.pop(sample, None)
            self.gate_enabled.pop(sample, None)
            return
        cur = self.members.get(sample)
        if cur is not None and gid in cur:
            cur.remove(gid)
            if not cur:
                self.members.pop(sample, None)

    def forget_sample(self, sample):
        """A sample was removed from the session."""
        self.order = [s for s in self.order if s != sample]
        self.presort = [r for r in self.presort if r[0] != sample]
        if self.members is not None:
            self.members.pop(sample, None)
        self.plot_enabled.pop(sample, None)
        self.gate_enabled.pop(sample, None)
        for store in (self.gates, self.gate_order, self.saved_gates):
            if store is not None:
                store.pop(sample, None)

    def remap_gate_ids(self, sample: str, id_map: Mapping[str, str]) -> None:
        """``sample``'s gates were given new ids (a session restore hands
        out fresh ones): point this tab's members, on/off states and Presort
        rows at them. An id not in ``id_map`` is left as it is."""
        if not id_map:
            return
        if self.members is not None and self.members.get(sample):
            self.members[sample] = [id_map.get(g, g)
                                    for g in self.members[sample]]
        if sample in self.gate_enabled:
            self.gate_enabled[sample] = {
                id_map.get(g, g): on
                for g, on in self.gate_enabled[sample].items()}
        self.presort = [(s, id_map.get(g, g) if s == sample else g)
                        for s, g in self.presort]

    def gate_lists(self) -> dict | None:
        """This tab's own gates as a session saves them, ``{sample: [gate
        carrying its 'id', ...]}`` in tree order, plus the session gates of
        samples that never arrived. None when it has none (the active tab's
        are the session's own; an empty tab's samples take a copy of those).
        """
        out = (gate_lists(self.gates, self.gate_order or {})
               if self.gates is not None else {})
        for name, gl in self.saved_gates.items():
            out.setdefault(name, [dict(g) for g in gl])
        return out or None

    def to_dict(self):
        """A JSON-able copy of the tab's state, for saving in a session."""
        return {'name': self.name,
                'members': copy.deepcopy(self.members),
                'plot_enabled': dict(self.plot_enabled),
                'gate_enabled': copy.deepcopy(self.gate_enabled),
                'view': copy.deepcopy(self.view),
                'selected': self.selected,
                'order': list(self.order),
                'presort': [list(r) for r in self.presort],
                'sample_gates': copy.deepcopy(self.gate_lists())}

    @classmethod
    def from_dict(cls, d):
        """Rebuild a tab from ``to_dict`` output (a missing name becomes 'Tab').

        Its saved gates wait in ``saved_gates`` for their samples to load.
        A tab saved before tabs kept their own gates has none, and each of
        its samples takes a copy of the session's gates when it arrives."""
        t = cls(d.get('name') or 'Tab', members=d.get('members'),
                plot_enabled=d.get('plot_enabled'),
                gate_enabled=d.get('gate_enabled'),
                view=d.get('view'), selected=d.get('selected'),
                order=d.get('order'), presort=d.get('presort'))
        sg = d.get('sample_gates')
        if isinstance(sg, Mapping):
            t.saved_gates = {str(n): [dict(g) for g in gl
                                      if isinstance(g, Mapping)]
                             for n, gl in sg.items()
                             if isinstance(gl, (list, tuple))}
        return t


class TabSet:
    """The tray's tabs; ``tabs[0]`` is always the All tab."""

    def __init__(self, tabs=None, current=0):
        tabs = list(tabs or [])
        if not tabs or not tabs[0].is_all:
            tabs.insert(0, ViewTab(ALL_TAB))
        self.tabs: list[ViewTab] = tabs
        self.current = min(max(int(current), 0), len(tabs) - 1)
        # The active tab's gates are the editor's live ones.
        self.active.gates = None
        self.active.gate_order = None

    @property
    def active(self) -> ViewTab:
        """The currently selected tab."""
        return self.tabs[self.current]

    def names(self):
        """Every tab's name, in tab order."""
        return [t.name for t in self.tabs]

    def unique_name(self, base):
        """``base`` if no tab has that name, else ``base`` with the first free number appended."""
        names = set(self.names())
        if base not in names:
            return base
        i = 2
        while f'{base} {i}' in names:
            i += 1
        return f'{base} {i}'

    def new_tab(self, name=None, members=None):
        """Append an empty tab (name made unique) and return its index."""
        t = ViewTab(self.unique_name(name or f'Tab {len(self.tabs)}'),
                    members=members if members is not None else {})
        self.tabs.append(t)
        return len(self.tabs) - 1

    def duplicate(self, idx: int, samples: Iterable[str] = (),
                  live: tuple[dict, dict] | None = None) -> int:
        """Copy a tab after itself, with a copy of its gates (``live``,
        ``(gates, order)``, are the active tab's). A copy of All is a normal
        tab holding ``samples`` (the samples loaded now)."""
        src = self.tabs[idx]
        d = src.to_dict()
        d['name'] = self.unique_name(f'{src.name} copy')
        if d['members'] is None:
            d['members'] = dict.fromkeys(samples)
        dup = ViewTab.from_dict(d)
        gates, order = ((src.gates, src.gate_order) if src.gates is not None
                        else (live or ({}, {})))
        dup.gates = copy.deepcopy(gates)
        dup.gate_order = copy.deepcopy(order or {})
        dup.saved_gates = copy.deepcopy(src.saved_gates)
        if self.current > idx:
            self.current += 1
        self.tabs.insert(idx + 1, dup)
        return idx + 1

    def remove(self, idx):
        """Delete a tab; the All tab stays."""
        if idx <= 0 or idx >= len(self.tabs):
            return
        self.tabs.pop(idx)
        if self.current >= len(self.tabs):
            self.current = len(self.tabs) - 1
        elif self.current > idx:
            self.current -= 1

    def rename(self, idx, name):
        """Rename tab ``idx`` to a unique form of ``name``; ignores the All tab and blank names."""
        name = str(name).strip()
        if idx <= 0 or not name or name == self.tabs[idx].name:
            return
        self.tabs[idx].name = ''              # its own name doesn't clash
        self.tabs[idx].name = self.unique_name(name)

    def forget_sample(self, sample):
        """Drop a sample removed from the session from every tab."""
        for t in self.tabs:
            t.forget_sample(sample)

    def to_dict(self):
        """A JSON-able form of every tab and the selected index, for saving in a session."""
        return {'current': self.current,
                'tabs': [t.to_dict() for t in self.tabs]}

    @classmethod
    def from_dict(cls, d):
        """Rebuild a tab set from ``to_dict`` output; anything not a mapping gives a fresh set."""
        if not isinstance(d, Mapping):
            return cls()
        tabs = [ViewTab.from_dict(t) for t in d.get('tabs') or ()
                if isinstance(t, Mapping)]
        return cls(tabs, current=d.get('current', 0) or 0)


def gate_closure(gates: Mapping[str, Mapping], gid) -> set:
    """What a copied gate needs where it lands: the gate, its ancestors, its
    subtree, and the gates a boolean among them combines (with theirs)."""
    keep: set = set()
    todo = [gid]
    while todo:
        g = todo.pop()
        if g not in gates or g in keep:
            continue
        for a in lineage(gates, g):
            if a not in keep:
                keep.add(a)
                todo.extend(gates[a].get('operands') or ())
        todo.extend(descendants(gates, g) - keep)
    return keep


def copy_gates(rows: Iterable[tuple[str, str | None]],
               src_gates: Mapping[str, Mapping[str, Any]],
               src_order: Mapping[str, list],
               dst_gates: dict[str, dict], dst_order: dict[str, list],
               shown: Callable[[str, Mapping], Iterable[str]] | None = None,
               ) -> int:
    """Copy the rows' gates -- ``[(sample, gid|None), ...]``, None meaning
    every gate of that sample -- from one tab's gates into another's.

    A gate the target shows (``shown(sample, gates)`` -> its shown ids;
    every gate it holds when None) is the target's own and is left as it
    is there: ids are never reused within a sample, so it is the same gate,
    perhaps edited in that tab. One it holds but does not show is replaced
    by the copy. Returns how many gates were copied."""
    n = 0
    for sample, gid in rows:
        sg: Mapping[str, Any] = src_gates.get(sample) or {}
        want = set(sg) if gid is None else gate_closure(sg, gid)
        dg = dst_gates.setdefault(sample, {})
        do = dst_order.setdefault(sample, [])
        keep = set(dg) if shown is None else set(shown(sample, dg)) & set(dg)
        order = list(src_order.get(sample) or ())
        order += [g for g in sg if g not in order]
        for g in order:
            if g in want and g not in keep:
                dg[g] = copy.deepcopy(sg[g])
                if g not in do:
                    do.append(g)
                n += 1
    return n


def rename_gate_ids(gates: dict, order: list, rename: Mapping) -> dict:
    """Give ``gates`` (one sample's, ``{gid: gate}``, in place) the ids in
    ``rename`` (old -> new), with their parents, boolean operands and
    ``order`` following. Returns the new ``{gid: gate}``; an id not in
    ``rename`` keeps its own. The new ids must not clash with kept ones."""
    if not rename:
        return gates
    out = {}
    for gid, g in gates.items():
        pid = g.get('parent_id')
        if pid in rename:
            g['parent_id'] = rename[pid]
        ops = g.get('operands')
        if ops:
            g['operands'] = [rename.get(o, o) for o in ops]
        out[rename.get(gid, gid)] = g
    order[:] = [rename.get(g, g) for g in order]
    return out


def apply_within(tab: ViewTab, rows, targets, plot_enabled, gate_enabled,
                 key_of, gid_for):
    """Apply the rows' on/off state to chosen samples ``targets`` of one
    ``tab`` only (not to every sample there). A sample row sets each
    target's plot on/off; a gate row finds the same population in each
    target (``key_of(sample, gid)`` -> population key, ``gid_for(sample,
    key)`` -> that sample's gate id or None) and sets it on/off, adding it to
    the tab when the tab shows only some of that sample's gates. Returns how
    many (target, row) pairs were applied."""
    n = 0
    for sample, gid in rows:
        if gid is None:
            if sample not in plot_enabled:
                continue
            for t in targets:
                if tab.has_sample(t):
                    tab.plot_enabled[t] = bool(plot_enabled[sample])
                    n += 1
            continue
        if (sample, gid) not in gate_enabled:
            continue
        key = key_of(sample, gid)
        for t in targets:
            if not tab.has_sample(t):
                continue
            tgid = gid_for(t, key)
            if tgid is None:
                continue
            chosen = None if tab.is_all else (tab.members or {}).get(t)
            if chosen is not None and tgid not in chosen:
                tab.add(t, tgid)
            tab.gate_enabled.setdefault(t, {})[tgid] = bool(
                gate_enabled[(sample, gid)])
            n += 1
    return n


def apply_rows(tabs: Iterable[ViewTab], rows, plot_enabled, gate_enabled,
               add=True):
    """Push the given rows -- ``[(sample, gid|None), ...]`` -- with their
    current on/off state into each tab in ``tabs``: added to the tab when
    ``add`` (and missing), and switched on/off as they are now.
    ``plot_enabled`` is ``{sample: bool}``; ``gate_enabled`` is
    ``{(sample, gid): bool}``."""
    for t in tabs:
        for sample, gid in rows:
            if add and not t.has_sample(sample):
                t.add(sample, gid)
            elif add and gid is not None and not t.is_all:
                chosen = (t.members or {}).get(sample)
                if chosen is not None and gid not in chosen:
                    t.add(sample, gid)
            if gid is None:
                if sample in plot_enabled:
                    t.plot_enabled[sample] = bool(plot_enabled[sample])
            elif (sample, gid) in gate_enabled:
                t.gate_enabled.setdefault(sample, {})[gid] = bool(
                    gate_enabled[(sample, gid)])
