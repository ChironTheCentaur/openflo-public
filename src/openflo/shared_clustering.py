"""The clustering every sample of a comparison shares, and the matching of
marker channels across samples by antibody label that it runs on: the
clustering settings, one run per --cluster-method, the pooling of an equal
share of every sample's events, and the label folding that decides which
channels hold one marker. Tk-free and display-free, so the library
(FlowExperiment.cluster_jointly, align_fluor_labels) and openflo-run use the
same code.
"""
from __future__ import annotations

import copy
import logging
import re

import numpy as np

from .fluor_catalog import COMMON_FLUORS
from .som_kernel import FLOWSOM_DEFAULT_METACLUSTERS

log = logging.getLogger(__name__)


def cluster_settings(k, n_jobs, max_events, vram_admission_gb, random_state,
                     cluster_cfg=None):
    """The clustering settings every task of a run shares; the comparison's
    shared clustering uses the same ones."""
    cc = cluster_cfg or {}
    return {'k': k, 'n_jobs': n_jobs,
            'method': cc.get('method', 'phenograph'),
            'resolution': cc.get('resolution', 1.0),
            'n_metaclusters': cc.get('n_metaclusters',
                                     FLOWSOM_DEFAULT_METACLUSTERS),
            'reproducible': cc.get('reproducible', True),
            'fast_graph': bool(cc.get('fast_graph', False)),
            'max_events': max_events,
            'vram_admission_gb': vram_admission_gb,
            'random_state': random_state}


# The events a method clusters when --max-events is not given: Leiden
# partitions at most this many and assigns the rest to their nearest
# partitioned event; FlowSOM trains its SOM on at most this many.
# PhenoGraph has no cap of its own.
_METHOD_EVENT_CAP = {'leiden': 200_000, 'flowsom': 50_000}


def cluster_by_method(s, task):
    """Cluster `s` in place with the task's --cluster-method settings and
    alias the method's label column to 'cluster'."""
    # The seed is held constant across every sample in a trial so the
    # subsampling cuts (when max_events truncates) and any subsequent
    # UMAP are reproducible. Workers can run in parallel safely —
    # each spawns its own RNG from the same seed value.
    method = task.get('method', 'phenograph')
    mev = task.get('max_events')
    seed = task.get('random_state', 42)
    if method == 'leiden':
        s.run_leiden(n_neighbors=task['k'],
                     resolution=task.get('resolution', 1.0),
                     max_events=mev or _METHOD_EVENT_CAP['leiden'],
                     random_state=seed,
                     fast_graph=bool(task.get('fast_graph', False)))
        if 'leiden' in s.data.columns:
            s.data['cluster'] = s.data['leiden']
    elif method == 'flowsom':
        s.run_flowsom(n_metaclusters=task.get('n_metaclusters',
                                             FLOWSOM_DEFAULT_METACLUSTERS),
                      max_events=mev or _METHOD_EVENT_CAP['flowsom'],
                      seed=seed)
        if 'flowsom_meta' in s.data.columns:
            s.data['cluster'] = s.data['flowsom_meta']
    else:
        s.cluster(k=task['k'], n_jobs=task['n_jobs'],
                  max_events=mev,
                  vram_admission_gb=task.get('vram_admission_gb', 1.0),
                  random_state=seed,
                  reproducible=bool(task.get('reproducible', True)),
                  resolution=task.get('resolution', 1.0))
    return s


def equal_quota(sizes, cap):
    """How many events each sample gives the shared clustering's pool of at
    most `cap`: an equal share each. A sample smaller than its share gives
    every event it has, and what it leaves is shared among the others; an
    indivisible remainder goes to the first samples.

    It is not one uniform draw over all events, which would give each
    sample its share of the events: a population held only by a small
    group would barely reach the pool (Ctrl 2 x 400k against Treat 2 x 20k
    would give Treat 5 % of it), while an equal share per sample, the usual
    practice when pooling samples for clustering, gives every sample the
    same weight in defining the clusters."""
    quota = [0] * len(sizes)
    left = min(int(cap), sum(sizes))
    active = [i for i, n in enumerate(sizes) if n > 0]
    while left > 0 and active:
        share = left // len(active)
        if share == 0:
            for i in active[:left]:
                quota[i] += 1
            break
        for i in active:
            take = min(share, sizes[i] - quota[i])
            quota[i] += take
            left -= take
        active = [i for i in active if quota[i] < sizes[i]]
    return quota


_DETECTOR_SUFFIXES = ('-A', '-H', '-W', ' A', ' H', ' W')

# Greek letters in marker names, as their Latin spellings: 'CD8α' is
# 'CD8a', 'TCRγδ' is 'TCRgd' (and so never 'TCRb').
_GREEK = str.maketrans({'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'κ': 'k',
                        'λ': 'l'})

# Common fluor words (fluor_catalog), dropped from a label where one
# stands apart as a word; fluor_tokens adds each panel's detector fluors.
_COMMON_FLUORS = COMMON_FLUORS


def fold_label(text):
    """A marker label as compared: case, spacing and punctuation dropped,
    Greek letters as their Latin spellings, other letters of any script
    kept ('Ly-6G' is 'Ly6G'; 'CD8α' is 'CD8a', never 'CD8β')."""
    return re.sub(r'[\W_]+', '', str(text).casefold().translate(_GREEK))


def _detector_suffix(det):
    """The detector's -A / -H / -W (or ' A' ...), matched without case, as
    canonical upper case; '' when it has none."""
    return next((x for x in _DETECTOR_SUFFIXES if det.upper().endswith(x)),
                '')


def _detector_fluor(det):
    """The detector's fluor, folded: FL1-A -> 'fl1'."""
    return fold_label(det[:len(det) - len(_detector_suffix(det))])


def fluor_tokens(detectors):
    """The fluor words a label may carry: _COMMON_FLUORS plus each
    detector's own fluor, two characters or more (a detector named 'B'
    would make every trailing 'b' a fluor)."""
    return _COMMON_FLUORS | {f for f in map(_detector_fluor, detectors)
                             if len(f) >= 2}


def label_key(lbl, det, fluors):
    """`lbl` on detector `det` as compared (fold_label), without a leading
    or trailing fluor word: 'CD901b FL1', 'CD3 FL5' on FL3-A, 'FL1-CD901b'
    and 'FL1 CD3 FL5' are CD901b / CD3. A fluor is dropped only as words of
    its own ('CD3 FL3', 'CD3 Brand Dye 488'), never as a bare suffix, so a
    marker's own letters stay, and the LONGEST fluor run goes: with FLB and
    its tandem FLB-Cy7 both fluors, 'FLB-Cy7-F4/80' dropping just 'FLB'
    read 'Cy7F4/80'. None for no label."""
    if not lbl:
        return None

    def gaps(text):                           # a Greek letter is no gap
        return list(re.finditer(r'[\W_]+', text))

    for sep in reversed(gaps(lbl)):           # longest leading fluor run
        head, tail = lbl[:sep.start()], lbl[sep.end():]
        if fold_label(head) in fluors and fold_label(tail):
            lbl = tail
            break
    for sep in gaps(lbl):                     # longest trailing fluor run
        head, tail = lbl[:sep.start()], lbl[sep.end():]
        if fold_label(tail) in fluors and fold_label(head):
            lbl = head
            break
    return fold_label(lbl)


def align_shared_channels(samples, names):
    """Match the marker channels of `samples` by antibody label, so each
    column of the shared clustering holds ONE marker in every sample.
    Returns (channels, columns, notes, display): `channels` are the first
    sample's detectors for the matched markers (the pooled frame's column
    names), `columns[i]` sample i's detectors in the same order, `notes`
    one line per channel matched across different detectors, matched by
    detector although its label differs, or left out, naming the samples
    by `names`, and `display` one distinct name per column.

    Matching by detector name put two antibodies in one column when a day's
    panel moved a marker to another fluor (CD901b on FL1 one day, on FL2
    the next), and dropped a channel missing from one sample without a word.
    A label is the channel's $PnS / --labels / --panel name, compared as
    label_key does: case, spacing and punctuation ignored, Greek letters
    as Latin ('CD8α' is 'CD8a', not 'CD8β'), and a known fluor word at
    either end dropped ('CD901b FL1', 'CD3 FL5' on FL3-A); a label that
    only names a fluor ('FL1') is no label. Detector names and their
    -A/-H/-W suffix compare without case ('FL6-A' is 'fl6-a'); the
    suffix is part of the match, so CD901b-A never pairs with CD901b-H. A
    label a sample puts on more than one detector ('Dump') is matched by
    label AND detector.

    The rules look at every sample at once, so the result does not depend
    on their order (it did: the first sample's channels drove the match):

    1. A labelled marker is one column when EVERY sample has it.
    2. Otherwise a detector every sample has, and no rule-1 column uses, is
       one column by detector name when it carries at most ONE label across
       the samples, and no sample that leaves it unlabelled has that label
       on another detector (there, the unlabelled channel is something
       else). Two different labels on one detector are never pooled.
    3. Every other channel is left out, and named."""
    match = _ChannelMatch(samples, names)
    match.by_label()
    match.by_detector()
    match.note_left_out()
    return match.result()


def _show(lbl, det):
    return f'{lbl} ({det})' if lbl else det


class _ChannelMatch:
    """The state align_shared_channels' three rules share: per sample its
    detectors, their labels and marker keys, the detectors a column already
    uses, and, per first-sample detector, the matched row and its name."""

    def __init__(self, samples, names):
        self.names = names
        self.n = n = len(samples)
        # A sample without data (a library stub) has all its fluor channels.
        self.dets = [[d for d in s.fluor_channels
                      if s.data is None or d in s.data.columns]
                     for s in samples]
        self.fluors = fluor_tokens(d for ds in self.dets for d in ds)
        self.labs = [{d: self.label(s, d) for d in ds}
                     for s, ds in zip(samples, self.dets, strict=True)]
        # A marker's key: its label (as norm() folds it) + -A/-H/-W. A
        # label a sample puts on two detectors ('Dump' on FL1-A and FL6-A)
        # is told apart by detector, so the same panel pairs detector with
        # detector; one key per label kept the first detector and paired a's
        # FL1-A with b's FL6-A when b listed its channels in another order.
        self.held = []                  # per sample: {(label, suffix)}
        self.keyed = []                 # per sample: {marker key: detector}
        for i in range(n):
            held, keyed = self._marker_keys(self.dets[i], self.labs[i])
            self.held.append(held)
            self.keyed.append(keyed)
        self.used = [set() for _ in samples]
        self.by_first = {}              # first sample's detector -> row
        self.display = {}               # first sample's detector -> name
        self.notes = []
        self.reported = set()           # (sample, detector) already named

    def label(self, s, det):
        """The channel's marker label, or None when it has none: no $PnS,
        or one that only repeats the detector ('FL1-A') or names a fluor
        ('FL1', or 'FL5' on any detector), which names no marker."""
        lbl = (getattr(s, 'channel_labels', None) or {}).get(det) or det
        tok = fold_label(lbl)
        return (None if tok in ('', fold_label(det)) or tok in self.fluors
                else lbl)

    def norm(self, lbl, det):
        """A label as compared (label_key): folding only case and spacing
        dropped a marker over punctuation in 207 of 4000 fuzzed panels, and
        dropping only the detector's own fluor word lost 'CD3 FL5' on
        FL3-A."""
        return label_key(lbl, det, self.fluors)

    def where(self, values):
        """'v1 in s0, s2; v2 in s1' over the samples."""
        by = {}
        for i, v in enumerate(values):
            by.setdefault(v, []).append(self.names[i])
        return '; '.join(f"{v} in {', '.join(ns)}" for v, ns in by.items())

    def _marker_keys(self, dets, labs):
        """One sample's {(label, suffix)} and {marker key: detector}."""
        suffix = _detector_suffix
        seen = [(self.norm(labs[d], d), suffix(d)) for d in dets
                if labs[d] is not None]
        keyed = {}
        for d in dets:
            if labs[d] is not None:
                ls = (self.norm(labs[d], d), suffix(d))
                keyed[ls + ((None,) if seen.count(ls) == 1
                            else (d.casefold(),))] = d
        return set(seen), keyed

    def _take(self, d, row, name):
        """Make `row` (one detector per sample) the column of first-sample
        detector `d`, named `name`."""
        for i, r in enumerate(row):
            self.used[i].add(r)
        self.by_first[d] = row
        self.display[d] = name

    def by_label(self):
        """Rule 1: markers every sample labels (walked in the first
        sample's channel order only so the notes come out in a stable
        order)."""
        keyed, labs, first = self.keyed, self.labs, self.dets[0]
        common = set(keyed[0]).intersection(*keyed[1:])
        for key in sorted(common, key=lambda k: first.index(keyed[0][k])):
            row = [keyed[i][key] for i in range(self.n)]
            name = labs[0][row[0]]
            self._take(row[0], row, _show(name, row[0]) if key[2] else name)
            if len({d.casefold() for d in row}) > 1:
                self.notes.append(f'{name}: {self.where(row)}')

    def by_detector(self):
        """Rule 2: detectors every sample has, by name (without case: 'FL6-A'
        is 'fl6-a'), when that pools ONE marker."""
        by_name = [{d.casefold(): d for d in ds} for ds in self.dets]
        for d in self.dets[0]:
            row = [by_name[i].get(d.casefold()) for i in range(self.n)]
            if not any(r is None or r in self.used[i]
                       for i, r in enumerate(row)):
                self._pool_detector(d, row)

    def _pool_detector(self, d, row):
        """Rule 2 for one detector `d` every sample has, unused: `row`."""
        rl = [self.labs[i][r] for i, r in enumerate(row)]
        found = {self.norm(lb, r)
                 for lb, r in zip(rl, row, strict=True)} - {None}
        if len(found) > 1:
            self.notes.append(f'{d}: different markers, not pooled; '
                              + self.where([lb or 'no label' for lb in rl]))
            self.reported.update(enumerate(row))
            return
        if found:
            lbl, = found
            if any(lb is None and (lbl, _detector_suffix(r)) in self.held[i]
                   for i, (lb, r) in enumerate(zip(rl, row, strict=True))):
                return
            if any(lb is None for lb in rl):
                self.notes.append(f'{d}: matched by detector; ' + self.where(
                    [lb or 'no label' for lb in rl]))
        self._take(d, row, next((lb for lb in rl if lb), None) or d)

    def note_left_out(self):
        """Rule 3: the rest is left out, by name."""
        left = {}
        for i in range(self.n):
            for d in self.dets[i]:
                if d not in self.used[i] and (i, d) not in self.reported:
                    left.setdefault(_show(self.labs[i][d], d), set()).add(i)
        for key, have in left.items():
            self.notes.append(
                f'{key}: no matching channel in '
                + ', '.join(self.names[j] for j in range(self.n)
                            if j not in have))

    def result(self):
        """(channels, columns, notes, display) of align_shared_channels."""
        channels = [d for d in self.dets[0] if d in self.by_first]
        columns = [[self.by_first[d][i] for d in channels]
                   for i in range(self.n)]
        # One name per column: two columns sharing one ('Dump' twice) would
        # give the table two median_Dump columns.
        shown = self.display
        taken = [shown[d] for d in channels]
        display = {d: (_show(shown[d], d) if taken.count(shown[d]) > 1
                       and shown[d] != d else shown[d]) for d in channels}
        return channels, columns, self.notes, display


def shared_cluster_labels(samples, cluster_task, names=None):
    """Cluster `samples` TOGETHER, once. Returns (labels, channels,
    columns, display): one int array per sample (-1 = unclustered) from a
    single clustering that every sample shares, the marker channels it used
    (the first sample's detector names), per sample that sample's detectors
    for them, and a name per channel (see align_shared_channels; `names`
    name the samples in its messages). Raises ValueError when no shared
    clustering can be made.

    Each sample is also clustered on its own (its <name>_stats.csv), but
    PhenoGraph, Leiden and FlowSOM number clusters arbitrarily (PhenoGraph by
    size), so one id names different populations in different samples.
    Comparing groups by that id compared unrelated populations: with one
    population falling from 70% to 30% between two groups, the largest
    difference in the table was 2.2 points, and 17 of its 23 ids held
    different populations in different samples. A comparison needs ids that
    mean the same population in every sample, so the samples are pooled and
    clustered as one (the usual concatenate-then-cluster design), and each
    sample's events are then counted per shared cluster.

    The pooled run clusters a subsample (the pool) of at most --max-events
    events IN TOTAL, an equal share drawn at random from each sample (see
    equal_quota). With no cap, the pool holds as many events as the
    largest sample, or the method's own cap when that is smaller (Leiden
    200,000, FlowSOM 50,000; _METHOD_EVENT_CAP), so the clustering itself
    costs no more than the largest per-sample run. Every event left out of
    the pool then takes the label of its nearest pooled event (1-NN), as
    --max-events already does inside one sample. FlowSOM instead labels
    every event through the SOM the pool trained (best-matching node ->
    its metacluster), which is FlowSOM's own rule and costs one pass.

    That 1-NN step is NOT free, and it is extra work no per-sample run does:
    it labels every unpooled event of the whole trial, in the main process.
    It runs on a scipy KDTree queried on every core. Measured on 14
    channels: 3 x 300k events with a 60k pool, 840k queries, took 144 s on
    one core with the scikit-learn KD-tree it replaced and 10 s here, with
    identical labels. A query costs more as the pool grows."""
    import pandas as pd
    from scipy.spatial import KDTree

    first = samples[0]
    names = names or [s.name for s in samples]
    channels, columns, notes, display = align_shared_channels(samples, names)
    if notes:
        log.warning("  [!] Shared clustering: channels are matched across "
                    "samples by antibody label (by detector when one has no "
                    "label). Left out, or matched across different "
                    "detectors:%s", ''.join(f'\n      {note}' for note in notes))
    sizes = [len(s.data) for s in samples]
    total = sum(sizes)
    if not channels:
        raise ValueError('no marker channel common to every sample')
    if not total:
        raise ValueError('the samples hold no events')
    method = cluster_task.get('method', 'phenograph')
    cap = shared_pool_cap(sizes, cluster_task)
    rng = np.random.default_rng(cluster_task.get('random_state', 42))
    picked = [np.sort(rng.choice(n, q, replace=False)) if q < n
              else np.arange(n)
              for n, q in zip(sizes, equal_quota(sizes, cap), strict=True)]
    pooled = copy.copy(first)
    pooled.data = pd.concat(
        [s.data[cols].iloc[ix].set_axis(channels, axis=1)
         for s, cols, ix in zip(samples, columns, picked, strict=True)],
        ignore_index=True)
    pooled.fluor_channels = list(channels)
    pooled.name = 'shared clustering'
    pooled.flowsom_result = None        # not the first sample's own SOM
    cluster_by_method(pooled, cluster_task)
    if 'cluster' not in pooled.data.columns:
        raise ValueError(f"{cluster_task.get('method', 'phenograph')} gave "
                         f"no cluster labels for {len(pooled.data):,} events")
    som = (getattr(pooled, 'flowsom_result', None) if method == 'flowsom'
           else None)               # set by run_flowsom, inside the call
    if som and 'meta_of_node' in som:
        # FlowSOM's own label for an event is the metacluster of its
        # best-matching SOM node, so every event, pooled or not, gets that
        # from the SOM the pool trained. Its nearest POOLED event's label
        # differed for 996 of 9,000 events on overlapping test data.
        from .flowsom import som_assign
        som_cols = [channels.index(c) for c in som['channels']]
        labels = []
        for s, cols in zip(samples, columns, strict=True):
            X = s.data[[cols[i] for i in som_cols]].to_numpy(dtype=float)
            ok = np.all(np.isfinite(X), axis=1)
            lab = np.full(len(s.data), -1, dtype=int)
            lab[ok] = som['meta_of_node'][som_assign(X[ok], som['weights'])]
            labels.append(lab)
        return labels, channels, columns, display
    plab = pooled.data['cluster'].to_numpy(dtype=int)
    ref = plab >= 0
    tree = None
    if ref.any() and (total > cap or not ref.all()):
        tree = KDTree(pooled.data[channels].to_numpy(dtype=float)[ref])
    ref_lab = plab[ref]
    labels, at = [], 0
    for s, cols, ix in zip(samples, columns, picked, strict=True):
        lab = np.full(len(s.data), -1, dtype=int)
        lab[ix] = plab[at:at + len(ix)]
        at += len(ix)
        if tree is not None:
            X = s.data[cols].to_numpy(dtype=float)
            # One rule for every finite event, pooled or not: the label of
            # its nearest labelled pooled event. PhenoGraph leaves the
            # events of tiny clusters at -1; kept for pooled events only,
            # a sample's noise share scaled with its pooled fraction (5.0 %
            # for a 100-event sample pooled whole against 0.75 % for a
            # 2000-event one pooled 5 %, from one pool). Only rows with a
            # non-finite value stay -1.
            rest = np.all(np.isfinite(X), axis=1)
            rest[ix] &= lab[ix] < 0
            if rest.any():
                _, nbr = tree.query(X[rest], k=1, workers=-1)
                lab[rest] = ref_lab[nbr]
        labels.append(lab)
    return labels, channels, columns, display


def shared_pool_cap(sizes, cluster_task):
    """At most how many events the shared clustering pools: --max-events,
    or with no cap the largest sample's size, never more than the method's
    own cap (_METHOD_EVENT_CAP)."""
    biggest = max(sizes)
    own = _METHOD_EVENT_CAP.get(cluster_task.get('method', 'phenograph'))
    return int(cluster_task.get('max_events')
               or (min(biggest, own) if own else biggest))
