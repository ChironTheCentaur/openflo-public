"""Tabs of the Samples & Gates tray (beta) -- Tk-free, headless-testable.

A tab is a saved view: which samples (and, optionally, which of their gates)
it shows, which of them are switched on, and how the plot looked -- axes,
plot type, colouring, axis types and zoom. Selecting a tab shows only its
members and restores its plot. The first tab, "All", always holds every
sample; other tabs hold what was copied into them.

Gates themselves are shared: a gate edited in one tab is the same gate in
every tab that shows its sample. What a tab owns is membership, the on/off
states and the view.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping

ALL_TAB = 'All'


class ViewTab:
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

    @property
    def is_all(self):
        return self.members is None

    def has_sample(self, sample):
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
            cur, seen = gid, set()
            while cur and cur in gates and cur not in seen:
                seen.add(cur)
                keep.add(cur)
                cur = gates[cur].get('parent_id')
        frontier = [g for g in chosen if g in gates]
        while frontier:
            pid = frontier.pop()
            for gid, g in gates.items():
                if g.get('parent_id') == pid and gid not in keep:
                    keep.add(gid)
                    frontier.append(gid)
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

    def to_dict(self):
        return {'name': self.name,
                'members': copy.deepcopy(self.members),
                'plot_enabled': dict(self.plot_enabled),
                'gate_enabled': copy.deepcopy(self.gate_enabled),
                'view': copy.deepcopy(self.view),
                'selected': self.selected,
                'order': list(self.order),
                'presort': [list(r) for r in self.presort]}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get('name') or 'Tab', members=d.get('members'),
                   plot_enabled=d.get('plot_enabled'),
                   gate_enabled=d.get('gate_enabled'),
                   view=d.get('view'), selected=d.get('selected'),
                   order=d.get('order'), presort=d.get('presort'))


class TabSet:
    """The tray's tabs; ``tabs[0]`` is always the All tab."""

    def __init__(self, tabs=None, current=0):
        tabs = list(tabs or [])
        if not tabs or not tabs[0].is_all:
            tabs.insert(0, ViewTab(ALL_TAB))
        self.tabs: list[ViewTab] = tabs
        self.current = min(max(int(current), 0), len(tabs) - 1)

    @property
    def active(self) -> ViewTab:
        return self.tabs[self.current]

    def names(self):
        return [t.name for t in self.tabs]

    def unique_name(self, base):
        names = set(self.names())
        if base not in names:
            return base
        i = 2
        while f'{base} {i}' in names:
            i += 1
        return f'{base} {i}'

    def new_tab(self, name=None, members=None):
        t = ViewTab(self.unique_name(name or f'Tab {len(self.tabs)}'),
                    members=members if members is not None else {})
        self.tabs.append(t)
        return len(self.tabs) - 1

    def duplicate(self, idx, samples=()):
        """Copy a tab after itself. A copy of All is a normal tab holding
        ``samples`` (the samples loaded now)."""
        src = self.tabs[idx]
        d = src.to_dict()
        d['name'] = self.unique_name(f'{src.name} copy')
        if d['members'] is None:
            d['members'] = {s: None for s in samples}
        dup = ViewTab.from_dict(d)
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
        name = str(name).strip()
        if idx <= 0 or not name or name == self.tabs[idx].name:
            return
        self.tabs[idx].name = ''              # its own name doesn't clash
        self.tabs[idx].name = self.unique_name(name)

    def forget_sample(self, sample):
        for t in self.tabs:
            t.forget_sample(sample)

    def to_dict(self):
        return {'current': self.current,
                'tabs': [t.to_dict() for t in self.tabs]}

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, Mapping):
            return cls()
        tabs = [ViewTab.from_dict(t) for t in d.get('tabs') or ()
                if isinstance(t, Mapping)]
        return cls(tabs, current=d.get('current', 0) or 0)


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
