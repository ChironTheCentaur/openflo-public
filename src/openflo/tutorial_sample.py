"""The tutorial sample: synthetic blood-like control and treated samples
with a gate tree on them, packed as one OpenFlo project archive.

The phone tutorial (``mobile/scripts/make_sample.py``) and the desktop
first-run guide (``scripts/make_desktop_sample.py``) both build their
bundled sample with ``write_sample``; only the number of events differs.

Every number here is synthetic; nothing comes from a real study. Channels are
FSC-A, FSC-H, SSC-A, FL1-A ... FL8-A and Time. Each FL channel carries a
textbook marker as its label ($PnS), so the tutorial's gates read as they
would in a lab; no dye is named. Spillover is baked in and recorded in
$SPILLOVER, so the app compensates the files as it opens them.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

FL = [f'FL{i}-A' for i in range(1, 9)]
# FL1 viability, FL2 CD3 (T cells), FL3 CD4 / FL4 CD8 (T-cell subsets), FL5
# CD19 (B cells), FL6 CD56 (NK), FL7 CD14 (monocytes), FL8 CD69, an early
# activation marker the treatment switches on.
MARKERS = dict(zip(FL, ['Live/Dead', 'CD3', 'CD4', 'CD8', 'CD19', 'CD56', 'CD14', 'CD69'], strict=True))
CHANNELS = ['FSC-A', 'FSC-H', 'SSC-A', *FL, 'Time']

# Spillover: row = the dye's own detector, column = where it leaks.
SPILL = np.eye(8)
for (src, dst), v in {(1, 2): 0.06, (2, 3): 0.12, (3, 4): 0.18, (4, 3): 0.04,
                      (5, 6): 0.09, (6, 7): 0.14, (7, 8): 0.07, (8, 7): 0.03,
                      (2, 8): 0.02}.items():
    SPILL[src - 1, dst - 1] = v

# Autofluorescence relative to FL1 per detector, and photon noise (signal
# units per photoelectron).
AF_SCALE = np.array([1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.3, 0.25])
PHOTON_GAIN = 25.0

# name: (fraction ctrl, fraction treated, SSC mean, {FL index: positive mean})
POPS = {
    'T CD4':     (0.30, 0.22, 28000, {2: 6000, 3: 3500}),
    'T CD8':     (0.15, 0.22, 27000, {2: 5500, 4: 6000}),
    'T DP':      (0.015, 0.02, 28000, {2: 5800, 3: 2000, 4: 2600}),
    'T DN':      (0.03, 0.03, 28000, {2: 4800}),
    'B':         (0.10, 0.08, 30000, {5: 4000}),
    'NK dim':    (0.07, 0.10, 34000, {6: 1600, 4: 700}),
    'NK bright': (0.015, 0.025, 33000, {6: 7000}),
    'Mono':      (0.16, 0.13, 62000, {7: 13000, 3: 1000}),
    'Other':     (0.03, 0.03, 30000, {}),
}
# Fraction of each population activated (FL8 high) under each condition.
ACTIVATED = {'T CD4': (0.08, 0.45), 'T CD8': (0.10, 0.60),
             'NK dim': (0.05, 0.35)}


def _population(rng, true, ssc, m, name, treated):
    """Fill one population's rows `m` of `true` (FL) and `ssc`."""
    c = int(m.sum())
    _, _, sscm, prof = POPS[name]
    ssc[m] = rng.lognormal(np.log(sscm), 0.24, c)
    for ch, mean in prof.items():
        level = rng.lognormal(np.log(mean), 0.5, c)
        # Some cells part-way: a smear between negative and positive, as
        # real stains show, not islands.
        dim = rng.random(c) < 0.2
        level[dim] *= rng.uniform(0.08, 0.7, int(dim.sum()))
        true[m, ch - 1] = level
    if name in ACTIVATED:
        share = ACTIVATED[name][1 if treated else 0] * rng.normal(1.0, 0.15)
        act = rng.random(c) < share
        col = true[m, 7]
        # Switched on by degrees: a smear up from the negatives.
        col[act] = col[act] * np.exp(rng.uniform(0.3, 1.0, int(act.sum())) ** 0.7 * np.log(150))
        true[m, 7] = col


def _live(rng, n, treated):
    """Live single cells: scatter and true (unspilled) fluorescence."""
    names = list(POPS)
    f = np.array([POPS[k][1 if treated else 0] for k in names])
    f = f * rng.normal(1.0, 0.08, len(f))
    pop = rng.choice(len(names), n, p=f / f.sum())
    # Cell size: bigger cells scatter more light and carry more of
    # everything, so every channel leans with it.
    size = rng.lognormal(0, 0.16, n)
    # Autofluorescence: one per cell, shared across detectors (strongest in
    # the blue end, little in the far red), plus a little per detector.
    af = rng.lognormal(np.log(160), 0.4, n)[:, None] * AF_SCALE
    true = af * rng.lognormal(0, 0.2, (n, 8))
    ssc = np.empty(n)
    for i, name in enumerate(names):
        _population(rng, true, ssc, pop == i, name, treated)
    fsc = rng.normal(62000, 6000, n) * size
    fsc[pop == names.index('Mono')] += 18000
    true *= size[:, None] ** 0.8
    ssc *= size ** 0.6
    return np.column_stack([fsc, fsc * rng.normal(0.5, 0.012, n), ssc, true])


def _block(rng, k, fsc, ratio, ssc, fl):
    """A blob of `k` non-cell events: (mean, sd) scatter, lognormal FL."""
    fs = np.abs(rng.normal(*fsc, k)) + 500
    return np.column_stack([fs, fs * rng.normal(ratio, 0.03, k),
                            np.abs(rng.normal(*ssc, k)) + 300,
                            rng.lognormal(np.log(fl[0]), fl[1], (k, 8))])


def sample(n, seed, treated):
    """One sample of `n` events as the cytometer would write it."""
    rng = np.random.default_rng(seed)
    n_dead, n_debris, n_dbl = int(n * 0.07), int(n * 0.06), int(n * 0.04)
    dead = _block(rng, n_dead, (50000, 9000), 0.5, (40000, 12000), (250, 0.9))
    dead[:, 3] = rng.lognormal(np.log(12000), 0.4, n_dead)       # FL1 bright
    ev = np.vstack([_live(rng, n - n_dead - n_debris - n_dbl, treated), dead,
                    _block(rng, n_debris, (9000, 5000), 0.5, (6000, 4000), (150, 0.8)),
                    _block(rng, n_dbl, (122000, 14000), 0.32, (60000, 15000), (900, 1.3))])
    rng.shuffle(ev)
    fl = ev[:, 3:] @ SPILL
    # Photon counting: the noise grows with the light, so after
    # compensation the negatives fan out where a bright dye spills in.
    fl += rng.normal(0, 1, fl.shape) * np.sqrt(np.clip(fl, 0, None) * PHOTON_GAIN)
    fl += rng.normal(0, 60, fl.shape)                        # electronic noise
    ev[:, 3:] = np.minimum(fl, 262143)                       # the detector's top
    ev[:, :3] = np.minimum(ev[:, :3], 262143)
    t = np.sort(rng.uniform(0, 240, len(ev)))
    return pd.DataFrame(np.column_stack([ev, t]), columns=pd.Index(CHANNELS))


def gate_tree(doc, s):
    """The tutorial's gates on sample `s`, in stored coordinates: linear
    scatter, fluorescence on the 0..1 logicle scale."""
    def add(gate, parent=None):
        return doc.add_gate(s, gate, parent)['id']

    cells = add({'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
                 'name': 'Cells', 'vertices': [
                     [24000, 9000], [45000, 4000], [104000, 9000],
                     [108000, 96000], [62000, 98000], [30000, 46000]]})
    singlets = add({'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'FSC-H',
                    'name': 'Singlets', 'vertices': [
                        [20000, 8200], [112000, 48500], [112000, 62000],
                        [20000, 12200]]}, cells)
    live = add({'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'SSC-A',
                'name': 'Live', 'x0': 0.0, 'x1': 0.45, 'y0': 0, 'y1': 120000},
               singlets)
    mono = add({'kind': 'ellipsoid', 'x_channel': 'FL7-A', 'y_channel': 'SSC-A',
                'name': 'Monocytes', 'mean': [0.71, 62000],
                'cov': [[0.0016, 0.0], [0.0, 1.6e8]], 'distance_sq': 6.0}, live)
    lymph = add({'kind': 'boolean', 'op': 'not', 'operands': [mono],
                 'name': 'Not monocytes'}, live)
    t = add({'kind': 'polygon', 'x_channel': 'FL2-A', 'y_channel': 'SSC-A',
             'name': 'T cells', 'vertices': [
                 [0.54, 9000], [0.80, 9000], [0.80, 52000], [0.54, 52000]]}, lymph)
    q = doc.add_quadrant(s, 'FL3-A', 'FL4-A', 0.43, 0.47, t)
    for gid, nm in zip(q, ['CD4+ CD8+', 'CD4+ T', 'CD8+ T', 'CD4- CD8-'], strict=True):
        doc.update_gate(s, gid, {**doc.gate(s, gid), 'name': nm})
    add({'kind': 'boolean', 'op': 'or', 'operands': [q[1], q[2]],
         'name': 'Single positive'}, t)
    add({'kind': 'threshold', 'channel': 'FL8-A', 'value': 0.5, 'name': 'CD69+'}, t)
    non_t = add({'kind': 'rect', 'x_channel': 'FL2-A', 'y_channel': 'SSC-A',
                 'name': 'Non-T', 'x0': 0.0, 'x1': 0.45, 'y0': 0, 'y1': 120000}, lymph)
    add({'kind': 'polygon', 'x_channel': 'FL5-A', 'y_channel': 'FL6-A',
         'name': 'B cells', 'vertices': [
             [0.52, 0.05], [0.75, 0.05], [0.75, 0.38], [0.52, 0.38]]}, non_t)
    add({'kind': 'polygon', 'x_channel': 'FL5-A', 'y_channel': 'FL6-A',
         'name': 'NK cells', 'vertices': [
             [0.08, 0.42], [0.42, 0.42], [0.42, 0.82], [0.08, 0.82]]}, non_t)


# The two samples, in order, and the seed of the first (the next adds 1).
CONDITIONS = ('Control', 'Treated')
SEED = 2027
# Every gate gate_tree draws, by name.
GATES = frozenset({
    'Cells', 'Singlets', 'Live', 'Monocytes', 'Not monocytes', 'T cells',
    'CD4+ CD8+', 'CD4+ T', 'CD8+ T', 'CD4- CD8-', 'Single positive',
    'CD69+', 'Non-T', 'B cells', 'NK cells'})


def _plot_on_open(session, names):
    """Tick samples `names` for the plot in the saved `session`, so the
    desktop editor draws them as soon as the session opens."""
    import json
    with open(session, encoding='utf-8') as f:
        data = json.load(f)
    for entry in data.get('samples', []):
        if entry.get('name') in names:
            entry['plot_enabled'] = True
    with open(session, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def write_sample(out, events, plotted=()):
    """Write the project archive `out`: one sample per condition of
    `events` events each, gated with gate_tree. The samples named in
    `plotted` open ticked for the plot. Returns `out`."""
    from .document import Document
    from .fcs_export import write_fcs
    from .project_archive import write_project_archive
    spill = '8,' + ','.join(FL) + ',' + ','.join(f'{v:g}' for v in SPILL.ravel())
    with tempfile.TemporaryDirectory() as tmp:
        doc = Document()
        names = []
        for i, cond in enumerate(CONDITIONS):
            p = os.path.join(tmp, f'{cond}.fcs')
            write_fcs(sample(events, SEED + i, cond == 'Treated'), p,
                      channel_labels=MARKERS,
                      keywords={'$SPILLOVER': spill, '$CYT': 'Synthetic', '$TIMESTEP': '0.01'})
            names.append(doc.open_fcs(p))
        gate_tree(doc, names[0])
        doc.copy_gates(names[0], names[1:])
        session = os.path.join(tmp, 'sample.flowsession')
        doc.save_session(session)
        if plotted:
            _plot_on_open(session, set(plotted))
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        write_project_archive(session, out)
    return out
