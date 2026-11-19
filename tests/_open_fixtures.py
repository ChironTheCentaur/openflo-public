"""Generated files for the session-open and workspace-import tests:
synthetic FCS files, FlowJo workspaces written by OpenFlo's own writer, and
session dicts. Generic channel names (FSC-A, SSC-A, FL1-A, FL2-A) only."""
from __future__ import annotations

import os

import numpy as np

from tests._fcs_writer import write_float_fcs

CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
# A 2x2 spillover for FL1-A / FL2-A: a workspace sample's own matrix.
SPILL = np.array([[1.0, 0.12], [0.04, 1.0]])


def make_fcs(path, seed=0, n=600):
    """A synthetic list-mode FCS at `path` (folders made); returns the
    path as a string."""
    rng = np.random.default_rng(seed)
    ev = np.column_stack([
        rng.uniform(2e4, 2e5, n), rng.uniform(1e4, 1e5, n),
        np.where(rng.random(n) < 0.4, rng.normal(8000, 900, n),
                 rng.normal(150, 60, n)),
        rng.normal(900, 250, n)]).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return write_float_fcs(str(path), ev, CHANNELS)


def wsp_gates():
    """A gate tree in OpenFlo's schema: a scatter rectangle, a threshold and
    an interval under it, and a boolean AND of those two."""
    return [
        {'id': 'g1', 'parent_id': None, 'kind': 'rect', 'name': 'Cells',
         'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'x0': 3e4, 'x1': 1.8e5, 'y0': 1.5e4, 'y1': 9e4},
        {'id': 'g2', 'parent_id': 'g1', 'kind': 'threshold',
         'channel': 'FL1-A', 'value': 2000.0, 'name': 'FL1+'},
        {'id': 'g3', 'parent_id': 'g1', 'kind': 'interval',
         'channel': 'FL2-A', 'lo': 600.0, 'hi': 1200.0, 'name': 'FL2 mid'},
        {'id': 'g4', 'parent_id': 'g1', 'kind': 'boolean', 'op': 'and',
         'operands': ['g2', 'g3'], 'name': 'both'},
    ]


def write_wsp(path, samples, uncompensated=False):
    """A workspace at `path` from `samples`, each ``(name, fcs path, group
    or None, own matrix?)``. With `uncompensated`, the interval gate names
    FL2-A without FlowJo's 'Comp-' prefix (a gate on the measured values)."""
    from openflo.pipeline import WspWriter
    w = WspWriter()
    for name, fcs, group, own in samples:
        w.add_sample(name=name, fcs_path=fcs, channels=CHANNELS,
                     gates=wsp_gates(), group=group,
                     compensation=(['FL1-A', 'FL2-A'], SPILL) if own
                     else None)
    w.write(str(path))
    if uncompensated:
        text = open(path, encoding='utf-8').read()
        text = text.replace(
            '<data-type:fcs-dimension data-type:name="Comp-FL2-A" />',
            '<data-type:fcs-dimension data-type:name="FL2-A" />')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text)
    return str(path)


def session_gates(prefix):
    """A saved gate list whose ids are not 1..n (a gate was deleted before
    the save), with a child and a boolean naming saved ids."""
    return [
        {'id': 'g2', 'parent_id': None, 'kind': 'threshold',
         'channel': 'FL1-A', 'value': 0.5, 'name': f'{prefix} pos',
         'enabled': True, 'color': '#aa0000'},
        {'id': 'g5', 'parent_id': 'g2', 'kind': 'interval',
         'channel': 'FL2-A', 'lo': 0.1, 'hi': 0.9, 'name': f'{prefix} mid',
         'enabled': False, 'color': '#00aa00'},
        {'id': 'g7', 'parent_id': None, 'kind': 'boolean', 'op': 'or',
         'operands': ['g5', 'g2'], 'name': f'{prefix} either',
         'enabled': True, 'color': '#0000aa'},
    ]


def session_dict(entries, gates):
    """A current-format session dict holding sample `entries` and their
    `gates` (``{name: [gate, ...]}``)."""
    from openflo.session_format import SESSION_FORMAT, SESSION_VERSION
    return {'format': SESSION_FORMAT, 'version': SESSION_VERSION,
            'active_sample': entries[0]['name'] if entries else None,
            'samples': entries, 'sample_gates': gates}
