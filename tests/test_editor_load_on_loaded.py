"""Characterization of ``LoadMixin._on_loaded`` (editor_load).

``_on_loaded`` is what every finished sample load runs on the Tk thread:
first load, a reload under a name already held, a sample a session restore
brought back (with its saved transforms, normalizations, derived channels,
models, trim, gates and stats), and a sample a FlowJo workspace import staged
gates, axes and a group for. These tests pin what it does, call by call.

The editor here is a recording stand-in: ``LoadMixin`` itself (the code under
test, with its own ``_add_load_note``) over stubs for every other editor
method it calls, and recording fakes for the module functions it imports
(pipeline, gating, session_models, ...), whose return values each scenario
chooses so that every branch is reached. Samples are real ``FlowSample`` loads
of synthetic FCS files. A run yields an ordered event log (each editor call,
each module call, each ``print``) plus the editor and sample state after each
arrival; ``tests/golden/editor_load_on_loaded.json`` holds those records as
the code produced them before ``_on_loaded`` was split into helpers.

Regenerate the golden file (only on purpose, from code known to be right):
    python -m tests.test_editor_load_on_loaded --regen
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import random
import sys

import numpy as np
import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden',
                      'editor_load_on_loaded.json')
CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
CELL_CYCLE_KEY = 'cell_cycle_model'
FLOWSOM_KEY = 'flowsom_model'
UNMIX_KEY = 'unmix_model'


# ── recording ───────────────────────────────────────────────────────────────

def _plain(v, root):
    """`v` as JSON-able data, with the scenario's temp folder as <TMP> and
    the working folder as <CWD>."""
    if isinstance(v, str):
        # A path-less sample registers abspath('') -- the working folder.
        v = v.replace(root, '<TMP>') if root else v
        return v.replace(os.getcwd(), '<CWD>')
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    if isinstance(v, dict):
        return {str(_plain(k, root)): _plain(x, root) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x, root) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(_plain(x, root) for x in v)
    return f'<{type(v).__name__}>'


class _Status:
    def __init__(self, ed):
        self._ed, self.value = ed, None

    def set(self, v):
        self._ed._ev('status_var.set', v)
        self.value = v

    def get(self):
        return self.value


class _Stubs:
    """Every editor method ``_on_loaded`` calls that ``LoadMixin`` does not
    define, recording each call (and ``_schedule_load_settle``, whose real
    body arms a Tk timer)."""

    def _ev(self, *a):
        self._log.append(_plain(list(a), self._root))

    def _plotcache_forget(self, name):
        self._ev('_plotcache_forget', name)

    def _note_unsaved(self):
        self._ev('_note_unsaved')

    def _restore_sample_transforms(self, sample, saved):
        self._ev('_restore_sample_transforms', saved)

    def _adopt_sample_transforms(self, sample, first):
        self._ev('_adopt_sample_transforms', first)
        if self._fx.get('adopt_changes'):
            dt = dict(getattr(sample, 'data_transforms', None) or {})
            dt['FL1-A'] = 'adopted'
            sample.data_transforms = dt

    def _ds_restore_trim(self, name, sample, rec):
        self._ev('_ds_restore_trim', name, rec)
        return self._fx.get('trim_why')

    def _stored_transforms(self, name, sample):
        self._ev('_stored_transforms', name)
        return {'FL1-A': 'stored'}

    def _set_active_sample(self, name):
        self._ev('_set_active_sample', name, self._suspend_undo)
        self._active_sample = name
        self._gates = self._sample_gates.setdefault(name, {})

    def _add_gate(self, g, replace_1d=True):
        self._ev('_add_gate', dict(g), replace_1d, self._suspend_undo,
                 self._active_sample)
        if g.get('boom'):
            raise RuntimeError(f"cannot add {g.get('name')}")
        act = self._active_sample
        seq = self._sample_gate_seq.get(act, 0) + 1
        self._sample_gate_seq[act] = seq
        gid = f'{act}#{seq}'
        self._gates[gid] = g
        self._sample_gate_order.setdefault(act, []).append(gid)
        return gid

    def _populate_channel_combos(self):
        self._ev('_populate_channel_combos', self._channels,
                 self._channel_labels)

    def _audit(self, kind, **kw):
        self._ev('_audit', kind, kw)

    def _saved_stats_restore(self, name, entry):
        self._ev('_saved_stats_restore', name, entry)

    def _schedule_load_settle(self, ms=120):
        self._ev('_schedule_load_settle')


def make_editor(mixin_cls, root):
    """A recording editor over `mixin_cls` (a ``LoadMixin``)."""
    cls = type('RecordingEditor', (_Stubs, mixin_cls), {})
    ed = cls.__new__(cls)
    ed._log, ed._root, ed._fx = [], root, {}
    ed._loading = set()
    ed._samples, ed._sample_order = {}, []
    ed._pending_sample_meta = {}
    ed._path_to_name, ed._name_to_path = {}, {}
    ed._pending_wsp_trial, ed._pending_wsp_axes = {}, {}
    ed._pending_sample_gates = {}
    ed._sample_is_comp, ed._sample_set_aside, ed._sample_notes = {}, {}, {}
    ed._sample_trial, ed._trial_order = {}, []
    ed._sample_gates, ed._sample_gate_seq, ed._sample_gate_order = {}, {}, {}
    ed._active_sample, ed._gates, ed._suspend_undo = None, {}, False
    ed._restore_lost_operands, ed._restore_built_on = [], []
    ed._sample_plot_enabled = {}
    ed._channels, ed._channel_labels = [], {}
    ed.status_var = _Status(ed)
    return ed


@contextlib.contextmanager
def recording_collaborators(ed, modules):
    """Swap the module functions ``_on_loaded`` imports for recording fakes
    driven by ``ed._fx``, and `print` in each of `modules` (editor_load, old
    or new) for a recorder; everything is put back on exit."""
    import openflo.derived
    import openflo.gate_eval
    import openflo.gate_text
    import openflo.gating
    import openflo.pipeline
    import openflo.result_stamp
    import openflo.session_models
    import openflo.ui_mass_cytometry
    import openflo.workspace_run
    ev, fx = ed._ev, (lambda: ed._fx)

    def restore_on_load(sample, rec):
        ev('restore_on_load', rec)
        return fx().get('bead_why')

    def restore_derived(sample, rec):
        ev('derived.restore', rec)
        return [tuple(x) for x in fx().get('derived', [])]

    def restore_cell_cycle(sample, rec):
        ev('restore_cell_cycle', rec)
        return fx().get('cc_why')

    def restore_flowsom(sample, rec):
        ev('restore_flowsom', rec)
        return fx().get('fs_why')

    def restore_unmix(editor, name, sample, rec):
        ev('restore_unmix', editor is ed, name, rec)
        return fx().get('unmix_why')

    def gate_rescale_fns(drawn_on, now):
        ev('gate_rescale_fns', sorted(drawn_on), sorted(now or {}),
           dict(drawn_on) == dict(now or {}))
        return {'FL1-A': 'rescale'} if fx().get('rescale') else {}

    def map_gate_coords(g, fns):
        ev('map_gate_coords', g.get('name'), fns)
        return dict(g, mapped=sorted(fns) if isinstance(fns, dict) else fns)

    def carry_gate_stamp(old, new):
        ev('carry_gate_stamp', old.get('name'))
        return dict(new, carried=True)

    def gate_scale_fns(stored, direction):
        ev('gate_scale_fns', stored, direction)
        return 'from_linear_fns'

    def mark_unknown_scale(g, stored):
        ev('mark_unknown_scale', g.get('name'), stored)
        return dict(g, marked=True)

    def wsp_uncompensated_channels(g, comp):
        ev('wsp_uncompensated_channels', g.get('name'), list(comp))
        return [c for c in g.get('uncomp', ()) if c not in comp]

    def broken_booleans(restored):
        ev('broken_booleans', sorted(restored))
        return ([i for i, g in restored.items() if g.get('lost')],
                [i for i, g in restored.items() if g.get('built')])

    def describe_gate(g):
        ev('describe_gate', g.get('kind'))
        return f"<{g.get('kind')}>"

    def placeholder_booleans(pending):
        ev('placeholder_booleans', [g.get('name') for g in pending])
        return [g.get('name') for g in pending if g.get('placeholder')]

    def remap_operands(g, old_to_new):
        ev('remap_operands', g.get('name'), old_to_new)
        if 'operands' in g:
            g['operands'] = [old_to_new.get(o, o) for o in g['operands']]

    def derive_trial_name(path):
        ev('derive_trial_name', path)
        return ('trial_' + os.path.basename(os.path.dirname(path))
                if path else 'trial_none')

    def rec_print(*a, **k):
        ev('print', ' '.join(str(x) for x in a), sorted(k))

    swaps = [
        (openflo.ui_mass_cytometry, 'restore_on_load', restore_on_load),
        (openflo.derived, 'restore', restore_derived),
        (openflo.session_models, 'restore_cell_cycle', restore_cell_cycle),
        (openflo.session_models, 'restore_flowsom', restore_flowsom),
        (openflo.session_models, 'restore_unmix', restore_unmix),
        (openflo.wsp_gate_coords, 'gate_rescale_fns', gate_rescale_fns),
        (openflo.wsp_gate_coords, 'map_gate_coords', map_gate_coords),
        (openflo.wsp_gate_coords, 'gate_scale_fns', gate_scale_fns),
        (openflo.wsp_gate_coords, 'mark_unknown_scale', mark_unknown_scale),
        (openflo.wsp_gate_coords, 'wsp_uncompensated_channels',
         wsp_uncompensated_channels),
        (openflo.gate_eval, 'broken_booleans', broken_booleans),
        (openflo.gate_text, 'describe_gate', describe_gate),
        (openflo.pipeline, 'placeholder_booleans', placeholder_booleans),
        (openflo.result_stamp, 'carry_gate_stamp', carry_gate_stamp),
        (openflo.gating, 'remap_operands', remap_operands),
        (openflo.workspace_run, 'derive_trial_name', derive_trial_name),
    ] + [(m, 'print', rec_print) for m in modules]
    missing = object()
    saved = [(m, a, getattr(m, a, missing)) for m, a, _ in swaps]
    try:
        for m, a, f in swaps:
            setattr(m, a, f)
        yield
    finally:
        for m, a, old in reversed(saved):
            if old is missing:
                delattr(m, a)
            else:
                setattr(m, a, old)


# ── synthetic samples ───────────────────────────────────────────────────────

_SAMPLE_CACHE = {}


def _fcs_sample(root, rel, seed):
    """A loaded FlowSample of a synthetic FCS at <root>/<rel>.fcs (written
    with the shared tests/_fcs_writer helper), cached, returned as a copy."""
    from openflo.pipeline import FlowSample
    from tests._fcs_writer import write_fcs
    path = os.path.join(root, rel + '.fcs')
    if path not in _SAMPLE_CACHE:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rng = np.random.default_rng(seed)
        n = 200 + 25 * (seed % 5)
        data = np.column_stack([
            rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
            rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
            rng.exponential(200, n) + 50, rng.exponential(500, n) + 100])
        write_fcs(path, data, CHANNELS, datatype='F')
        s = FlowSample(path)
        s.auto_compensate()
        s.apply_transform()
        _SAMPLE_CACHE[path] = s
    return copy.deepcopy(_SAMPLE_CACHE[path])


def _sample_for(root, arr):
    s = _fcs_sample(root, arr.get('file', arr['name']), arr.get('seed', 1))
    if arr.get('path') == 'none':
        s.path = ''
    elif arr.get('path') == 'csv':
        s.path = s.path[:-4] + '.csv'
    for col in arr.get('extra_cols', ()):
        s.data[col] = np.arange(len(s.data)) % 3
    for k, v in (arr.get('attrs') or {}).items():
        setattr(s, k, copy.deepcopy(v))
    return s


def _pkey(path):
    return os.path.normcase(os.path.abspath(path))


# ── running a scenario ──────────────────────────────────────────────────────

def _summary(ed, root):
    keep = ('_loading', '_sample_order', '_pending_sample_meta',
            '_path_to_name', '_name_to_path', '_pending_wsp_trial',
            '_pending_wsp_axes', '_pending_sample_gates', '_sample_is_comp',
            '_sample_set_aside', '_sample_notes', '_sample_trial',
            '_trial_order', '_sample_gates', '_sample_gate_seq',
            '_sample_gate_order', '_active_sample', '_suspend_undo',
            '_restore_lost_operands', '_restore_built_on',
            '_sample_plot_enabled', '_channels', '_channel_labels',
            '_last_loaded', '_qc_removed_batch', '_uncompensated_imports',
            '_unconverted_imports', '_unconverted_booleans', '_load_notes')
    out = {k: _plain(getattr(ed, k, '<unset>'), root) for k in keep}
    out['_gates_is'] = next((n for n, g in ed._sample_gates.items()
                             if g is ed._gates), None)
    out['status'] = ed.status_var.value
    out['samples'] = {
        n: _plain({'id': s.__dict__.get('_scenario_id'),
                   'n_cols': len(s.data.columns), 'n': len(s.data),
                   'transforms': sorted(
                       (getattr(s, 'data_transforms', None) or {}).items(),
                       key=str),
                   **{a: getattr(s, a, '<unset>') for a in (
                       'batch_norm', 'stat_axes', 'result_stamps',
                       'result_originals')}}, root)
        for n, s in ed._samples.items()}
    return out


def run_scenario(mixin_cls, modules, sc, root):
    """Run scenario `sc` on a recording editor over `mixin_cls`; returns the
    record (per arrival: the event log, any exception, the state after)."""
    _SAMPLE_CACHE.clear()
    ed = make_editor(mixin_cls, root)
    init = sc.get('init') or {}
    for k, v in init.items():
        if k == 'inbound':
            ed._note_inbound_landed = (
                lambda pkey, ok: ed._ev('_note_inbound_landed', pkey, ok))
        else:
            setattr(ed, k, copy.deepcopy(v))
    record = []
    with recording_collaborators(ed, modules):
        for i, arr in enumerate(sc['arrivals']):
            name = arr['name']
            sample = _sample_for(root, arr)
            sample._scenario_id = i
            path = getattr(sample, 'path', '') or ''
            ed._loading.add(name)
            if arr.get('meta') is not None:
                key = _pkey(path) if path else _pkey(
                    os.path.join(root, 'elsewhere.fcs'))
                ed._pending_sample_meta[key] = copy.deepcopy(arr['meta'])
            wsp = arr.get('wsp') or {}
            if 'gates' in wsp:
                ed._pending_sample_gates[name] = copy.deepcopy(wsp['gates'])
            if 'axes' in wsp:
                ed._pending_wsp_axes[name] = copy.deepcopy(wsp['axes'])
            if 'trial' in wsp:
                ed._pending_wsp_trial[name] = wsp['trial']
            ed._fx = arr.get('fx') or {}
            ed._log = []
            err = None
            try:
                ed._on_loaded(name, sample)
            except Exception as e:  # noqa: BLE001 - recorded and compared
                err = [type(e).__name__, str(e)]
            record.append({'events': ed._log, 'error': err,
                           'state': _summary(ed, root)})
    return record


# ── scenarios ───────────────────────────────────────────────────────────────

def _g(name, kind='rect', **kw):
    return {'name': name, 'kind': kind, 'channel': 'FL1-A', **kw}


FULL_META = {
    'transforms': {'FL1-A': 'logicle'},
    'bead_norm': {'factors': [1, 2]},
    'batch_norm': {'ref': 'b1'},
    'stat_axes': {'FL1-A': {'lin': 1}, 'FL9-A': {'lin': 2}},
    'derived': {'R1': 'FL1-A / FL2-A'},
    'result_stamps': {'cluster': 'st1'},
    'result_originals': {'FL1-A [original 1]': 'st0'},
    'downsample_trim': {'keep': 100},
    CELL_CYCLE_KEY: {'m': 1}, FLOWSOM_KEY: {'k': 4}, UNMIX_KEY: {'u': 1},
    'trial': 'Day 2', 'is_comp': 1, 'set_aside': 'bad stain',
    'note': 'note a', 'saved_stats': {'G1': 10},
}

FIXED = [
    {'name': 'first_load_plain', 'arrivals': [
        {'name': 'sample_a', 'file': 'day1/sample_a', 'seed': 1}]},
    {'name': 'two_loads_then_reload', 'arrivals': [
        {'name': 'sample_a', 'file': 'day1/sample_a', 'seed': 1,
         'attrs': {'load_warnings': ['w1', 'w2'], 'qc_removed': 7,
                   'compensation_source': '$SPILL', 'qc_applied': True,
                   'fcs_decoding': {'bits': 32}}},
        {'name': 'sample_b', 'file': 'day2/sample_b', 'seed': 2,
         'attrs': {'qc_removed': 3}},
        {'name': 'sample_a', 'file': 'day1/sample_a', 'seed': 1}],
     'init': {'_qc_removed_batch': [1]}},
    {'name': 'no_path_sample', 'arrivals': [
        {'name': 'sample_a', 'path': 'none',
         'attrs': {'load_warnings': ('only',)}},
        {'name': 'sample_b', 'path': 'none', 'meta': {'trial': 'x'}}],
     'init': {'_active_sample': 'held', 'inbound': True}},
    {'name': 'session_restore_full_fcs', 'init': {
        'inbound': True, '_session_channel_labels': {'FL1-A': 'label one'}},
     'arrivals': [
        {'name': 'sample_a', 'file': 'day1/sample_a', 'seed': 3,
         'extra_cols': ['cluster', 'FL1-A [original 1]'],
         'meta': dict(FULL_META, gates=[
             _g('G1', id='a'), _g('G2', id='b', parent_id='a',
                                  enabled=False),
             _g('B', 'boolean', id='c', operands=['a', 'b', 'zz'],
                lost=True),
             _g(None, 'boolean', id='d', built=True),
             _g('C', 'cluster', id='e', channel='cluster'),
             _g('K', 'category', id='f', channel='kmeans'),
             _g('E', 'flowjo_ellipse', id='g', placeholder=True)]),
         'fx': {'bead_why': 'no beads', 'derived': [['R1', 'bad']],
                'cc_why': 'shape', 'fs_why': None, 'unmix_why': 'gone',
                'trim_why': 'trim moved', 'rescale': True,
                'adopt_changes': True}}]},
    {'name': 'session_restore_csv_no_whys', 'arrivals': [
        {'name': 'sample_a', 'file': 'day1/sample_a', 'seed': 4,
         'meta': {'batch_norm': {'ref': 'b2'}, CELL_CYCLE_KEY: None,
                  FLOWSOM_KEY: {'k': 2}, 'is_comp': 0, 'trial': '',
                  'gates': [_g('G1', id='a')]}},
        {'name': 'sample_b', 'file': 'day1/sample_b', 'seed': 5,
         'path': 'csv',
         'meta': dict(FULL_META, gates=[_g('G1', _import_id='q'),
                                        _g('G2', parent_id='q')]),
         'fx': {'rescale': False}}]},
    {'name': 'empty_meta_dict', 'arrivals': [
        {'name': 'sample_a', 'file': 'd/sample_a', 'seed': 6, 'meta': {},
         'wsp': {'gates': [_g('W1')], 'axes': {'FL1-A': {'x': 1}},
                 'trial': 'wsp group'}}]},
    {'name': 'wsp_import', 'init': {'_uncompensated_imports': [['old', 'X']],
                                    '_unconverted_imports': [{'name': 'o'}]},
     'arrivals': [
        {'name': 'sample_a', 'file': 'day1/sample_a', 'seed': 7,
         'attrs': {'comp_channels': ['FL2-A']},
         'wsp': {'gates': [
             _g('W1', id='1', uncomp=['FL1-A', 'FL2-A']),
             _g(None, id='2', parent_id='1', label='lab', uncomp=['FL3-A']),
             _g(None, 'flowjo_ellipse', id='3', parent_id='9'),
             _g('C', 'cluster', channel=None)],
             'axes': {'FL1-A': {'lin': 1}, 'FL7-A': {'lin': 3}},
             'trial': 'group 1'}},
        {'name': 'sample_b', 'file': 'day1/sample_b', 'seed': 8,
         'wsp': {'gates': [], 'axes': {}, 'trial': ''}},
        {'name': 'sample_c', 'file': 'day3/sample_c', 'seed': 9,
         'wsp': {'gates': [_g('Z')]}}]},
    {'name': 'gate_add_raises', 'init': {'_suspend_undo': 'prev'},
     'arrivals': [
        {'name': 'sample_a', 'file': 'day1/sample_a', 'seed': 1},
        {'name': 'sample_b', 'file': 'day1/sample_b', 'seed': 2,
         'meta': {'gates': [_g('ok', id='a'), _g('bad', id='b', boom=1)]}},
        {'name': 'sample_c', 'file': 'day1/sample_c', 'seed': 3}]},
    {'name': 'active_is_the_arrival', 'init': {'_active_sample': 'sample_a'},
     'arrivals': [
        {'name': 'sample_a', 'file': 'day1/sample_a', 'seed': 1,
         'wsp': {'gates': [_g('W1')]}},
        {'name': 'sample_a', 'file': 'day1/sample_a', 'seed': 1,
         'meta': {'gates': [_g('W2')], 'transforms': None}}]},
    {'name': 'meta_under_other_path', 'init': {'_trial_order': ['trial_day1'],
                                               '_path_to_name': {}},
     'arrivals': [
        {'name': 'sample_a', 'file': 'day1/sample_a', 'seed': 1,
         'meta': {'trial': 'never'}, 'path': 'none'},
        {'name': 'sample_a2', 'file': 'day1/sample_a', 'seed': 1}]},
]


def _rand_gates(rng, n_max=5):
    gates, ids = [], []
    for i in range(rng.randint(0, n_max)):
        g = {'kind': rng.choice(['rect', 'boolean', 'cluster', 'category',
                                 'flowjo_ellipse', 'polygon'])}
        if rng.random() < .8:
            g['name'] = f'G{i}'
        if rng.random() < .3:
            g['label'] = f'L{i}'
        g['channel'] = rng.choice(['FL1-A', 'cluster', 'kmeans', None])
        idk = rng.choice(['id', '_import_id', None])
        if idk:
            g[idk] = f's{i}'
            ids.append(f's{i}')
        if ids and rng.random() < .5:
            g['parent_id'] = rng.choice(ids + ['missing'])
        if rng.random() < .3:
            g['enabled'] = rng.random() < .5
        if g['kind'] == 'boolean' and rng.random() < .7:
            g['operands'] = rng.sample(ids + ['missing'],
                                       min(2, len(ids) + 1))
        for flag in ('lost', 'built', 'placeholder'):
            if rng.random() < .2:
                g[flag] = True
        if rng.random() < .3:
            g['uncomp'] = rng.sample(['FL1-A', 'FL2-A', 'FL3-A'], 2)
        if rng.random() < .04:
            g['boom'] = 1
        gates.append(g)
    return gates


def random_scenario(seed):
    """A seeded random scenario: 1-6 arrivals over a handful of names, each
    with a random mix of session meta, staged .wsp data, sample attributes
    and collaborator answers."""
    rng = random.Random(seed)
    names = ['sample_a', 'sample_b', 'sample_c', 'sample_d']
    init = {}
    if rng.random() < .3:
        init['inbound'] = True
    if rng.random() < .3:
        init['_session_channel_labels'] = {'FL2-A': 'saved two'}
    if rng.random() < .2:
        init['_active_sample'] = rng.choice(names + ['held'])
    if rng.random() < .3:
        init['_suspend_undo'] = rng.choice([True, 'x'])
    for k in ('_uncompensated_imports', '_unconverted_imports',
              '_unconverted_booleans', '_qc_removed_batch'):
        if rng.random() < .2:
            init[k] = [k[1:4]]
    if rng.random() < .2:
        init['_trial_order'] = ['trial_day1']
    arrivals = []
    for _ in range(rng.randint(1, 6)):
        name = rng.choice(names)
        arr = {'name': name, 'seed': rng.randint(0, 9),
               'file': f"day{rng.randint(1, 2)}/{name}"}
        arr['path'] = rng.choice(['fcs', 'fcs', 'fcs', 'csv', 'none'])
        attrs = {}
        if rng.random() < .3:
            attrs['load_warnings'] = [f'w{j}' for j in range(rng.randint(
                0, 2))]
        if rng.random() < .3:
            attrs['qc_removed'] = rng.choice([0, 5, None])
        if rng.random() < .2:
            attrs['compensation_source'] = rng.choice(['$SPILL', None])
        if rng.random() < .3:
            attrs['comp_channels'] = rng.sample(['FL1-A', 'FL2-A'], 1)
        arr['attrs'] = attrs
        if rng.random() < .3:
            arr['extra_cols'] = rng.sample(['cluster', 'kmeans',
                                            'FL1-A [original 1]'], 2)
        if rng.random() < .5:
            meta = {}
            for k, v in FULL_META.items():
                if rng.random() < .4:
                    meta[k] = rng.choice([v, v, None, {}]) if not isinstance(
                        v, str) else rng.choice([v, ''])
            if 'is_comp' in FULL_META and rng.random() < .3:
                meta['is_comp'] = rng.choice([0, 1])
            if rng.random() < .6:
                meta['gates'] = _rand_gates(rng)
            arr['meta'] = meta
        if rng.random() < .5:
            wsp = {}
            if rng.random() < .7:
                wsp['gates'] = _rand_gates(rng)
            if rng.random() < .5:
                wsp['axes'] = {'FL1-A': {'a': 1}, 'FL8-A': {'a': 2}}
            if rng.random() < .5:
                wsp['trial'] = rng.choice(['wsp group', ''])
            arr['wsp'] = wsp
        arr['fx'] = {
            'bead_why': rng.choice([None, 'why']),
            'derived': rng.choice([[], [['R1', 'bad']]]),
            'cc_why': rng.choice([None, 'cc']),
            'fs_why': rng.choice([None, 'fs']),
            'unmix_why': rng.choice([None, 'um']),
            'trim_why': rng.choice([None, 'trim']),
            'rescale': rng.random() < .5,
            'adopt_changes': rng.random() < .5}
        arrivals.append(arr)
    return {'name': f'random_{seed}', 'init': init, 'arrivals': arrivals}


GOLDEN_RANDOM_SEEDS = range(24)


def all_golden_scenarios():
    return FIXED + [random_scenario(s) for s in GOLDEN_RANDOM_SEEDS]


def _new_mixin():
    import openflo.editor_load as mod
    return mod.LoadMixin, [mod]


def record_all(mixin_cls, modules, scenarios, tmp_root):
    out = {}
    for sc in scenarios:
        root = os.path.join(tmp_root, sc['name'])
        os.makedirs(root, exist_ok=True)
        out[sc['name']] = run_scenario(mixin_cls, modules, sc, root)
    return out


# ── tests ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def records(tmp_path_factory):
    cls, mods = _new_mixin()
    return record_all(cls, mods, all_golden_scenarios(),
                      str(tmp_path_factory.mktemp('on_loaded')))


@pytest.fixture(scope='module')
def golden():
    with open(GOLDEN, encoding='utf-8') as fh:
        return json.load(fh)


@pytest.mark.parametrize('sc', [s['name'] for s in all_golden_scenarios()])
def test_matches_the_recorded_behaviour(records, golden, sc):
    """Every editor call, module call and print, in order, and the state
    after each arrival, are what the recorded code produced."""
    want, got = golden[sc], records[sc]
    assert len(got) == len(want)
    for i, (w, g) in enumerate(zip(want, got, strict=True)):
        assert g['error'] == w['error'], f'arrival {i}'
        assert g['events'] == w['events'], f'arrival {i}'
        assert g['state'] == w['state'], f'arrival {i}'


def _calls(arrival, name):
    return [e for e in arrival['events'] if e[0] == name]


def _order(arrival):
    return [e[0] for e in arrival['events']]


def test_first_load_seeds_the_editor_and_later_ones_do_not(records):
    a, b, a2 = records['two_loads_then_reload']
    assert _calls(a, '_adopt_sample_transforms') == [
        ['_adopt_sample_transforms', True]]
    assert _calls(b, '_adopt_sample_transforms') == [
        ['_adopt_sample_transforms', False]]
    assert a['state']['_sample_plot_enabled'] == {'sample_a': True}
    assert b['state']['_sample_plot_enabled'] == {'sample_a': True,
                                                  'sample_b': False}
    assert _calls(a, '_populate_channel_combos')
    assert not _calls(b, '_populate_channel_combos')
    assert a['state']['_channels'] == CHANNELS
    assert a['state']['_active_sample'] == 'sample_a'
    # A reload keeps its place in the order and is not seeded again.
    assert a2['state']['_sample_order'] == ['sample_a', 'sample_b']
    assert not _calls(a2, '_populate_channel_combos')
    # Warnings become load notes; added (not restored) = unsaved.
    assert a['state']['_load_notes'] == ['sample_a: w1', 'sample_a: w2']
    assert _calls(a, '_note_unsaved') and _calls(b, '_note_unsaved')
    assert b['state']['_qc_removed_batch'] == [1, 7, 3]
    assert b['state']['status'] == ('2 sample(s) loaded (latest: sample_b, '
                                    '250 events after QC removed 3)…')
    audit = [e for e in _calls(a, '_audit') if e[1] == 'sample.load'][0][2]
    assert audit['compensation'] == '$SPILL' and audit['qc'] is True
    assert audit['trial'] == 'trial_day1'
    # The phases run in this order.
    seq = [e for e in _order(a) if e != '_audit']
    assert seq == ['_plotcache_forget', '_note_unsaved',
                   '_adopt_sample_transforms', 'gate_rescale_fns',
                   'derive_trial_name', '_populate_channel_combos',
                   '_set_active_sample', 'status_var.set',
                   '_schedule_load_settle']


def test_session_restore_puts_everything_back_in_order(records):
    (a,) = records['session_restore_full_fcs']
    order = _order(a)
    first = [order.index(k) for k in (
        '_restore_sample_transforms', '_adopt_sample_transforms',
        'restore_on_load', 'derived.restore', '_ds_restore_trim',
        'restore_cell_cycle', 'restore_flowsom', 'restore_unmix',
        'gate_rescale_fns', 'derive_trial_name', 'map_gate_coords',
        'carry_gate_stamp', '_set_active_sample', '_add_gate',
        'remap_operands', 'broken_booleans', 'placeholder_booleans',
        '_populate_channel_combos', '_note_inbound_landed',
        '_saved_stats_restore', 'status_var.set', '_schedule_load_settle')]
    assert first == sorted(first)
    assert '_note_unsaved' not in order
    st = a['state']
    assert st['_sample_trial'] == {'sample_a': 'Day 2'}
    assert st['_sample_is_comp'] == {'sample_a': True}
    assert st['_sample_set_aside'] == {'sample_a': 'bad stain'}
    assert st['_sample_notes'] == {'sample_a': 'note a'}
    smp = st['samples']['sample_a']
    assert smp['stat_axes'] == {'FL1-A': {'lin': 1}}
    assert smp['batch_norm'] == '<unset>'      # a raw FCS: said, not set
    assert smp['result_stamps'] == {'cluster': 'st1'}
    assert st['_channel_labels']['FL1-A'] == 'label one'
    assert 'FL1-A [original 1]' not in st['_channels']
    assert st['_restore_lost_operands'] == ['B']
    assert st['_restore_built_on'] == ['<boolean>']
    assert st['_unconverted_booleans'] == ['E']
    assert [g['name'] for g in st['_unconverted_imports']] == ['E']
    notes = st['_load_notes']
    assert any('bead normalization not restored (no beads)' in n
               for n in notes)
    assert any('batch normalization not restored' in n for n in notes)
    assert any('derived channel R1 not restored (bad)' in n for n in notes)
    assert 'trim moved' in notes
    assert any('the kmeans results' in n for n in notes)
    prints = [e[1] for e in _calls(a, 'print')]
    assert prints == [
        '[session] sample_a: cell_cycle_model not restored (shape)',
        '[session] sample_a: unmix_model not restored (gone)']
    adds = _calls(a, '_add_gate')
    assert [e[1].get('parent_id') for e in adds][:2] == [None, 'sample_a#1']
    assert all(e[2] is False and e[3] is True for e in adds)
    assert [e[1]['enabled'] for e in adds][:2] == [True, False]
    assert all(e[1].get('carried') for e in adds)
    gates = st['_sample_gates']['sample_a']
    assert gates['sample_a#3']['operands'] == ['sample_a#1', 'sample_a#2',
                                               'zz']
    assert st['_suspend_undo'] is False


def test_csv_restore_keeps_batch_norm_and_skips_none_models(records):
    a, b = records['session_restore_csv_no_whys']
    assert [e[0] for e in a['events']].count('restore_cell_cycle') == 0
    assert _calls(a, 'restore_flowsom')
    assert a['state']['_sample_trial'] == {'sample_a': 'trial_day1'}
    assert a['state']['_sample_is_comp'] == {'sample_a': False}
    assert b['state']['samples']['sample_b']['batch_norm'] == {'ref': 'b1'}
    # Rescale off: the saved gates go in unmapped.
    assert not _calls(b, 'map_gate_coords')
    adds = _calls(b, '_add_gate')
    assert adds[1][1]['parent_id'] == 'sample_b#1'
    # Another sample was active: it is active again afterwards.
    assert _calls(b, '_set_active_sample')[-1][1] == 'sample_a'
    assert b['state']['_gates_is'] == 'sample_a'


def test_empty_meta_is_a_restore_without_gates(records):
    (a,) = records['empty_meta_dict']
    assert '_note_unsaved' not in _order(a)
    assert not _calls(a, '_add_gate')
    st = a['state']
    assert st['_pending_sample_gates'] == {
        'sample_a': [{'name': 'W1', 'kind': 'rect', 'channel': 'FL1-A'}]}
    assert st['_sample_trial'] == {'sample_a': 'wsp group'}
    assert st['_pending_wsp_trial'] == {}


def test_wsp_gates_are_converted_and_flagged(records):
    a, b, c = records['wsp_import']
    order = _order(a)
    assert order.index('_stored_transforms') < order.index('gate_scale_fns')
    st = a['state']
    assert st['samples']['sample_a']['stat_axes'] == {'FL1-A': {'lin': 1}}
    assert st['_uncompensated_imports'] == [
        ['old', 'X'], ['W1', 'FL1-A'], ['lab', 'FL3-A']]
    assert [g['name'] for g in st['_unconverted_imports']] == ['o', None]
    assert st['_sample_trial'] == {'sample_a': 'group 1'}
    adds = _calls(a, '_add_gate')
    assert [e[1].get('parent_id') for e in adds] == [
        None, 'sample_a#1', None, None]
    assert all(e[1]['marked'] for e in adds)
    assert any('the cluster results' in n for n in st['_load_notes'])
    assert b['state']['_sample_trial']['sample_b'] == 'trial_day1'
    assert not _calls(b, '_add_gate')
    assert c['state']['_trial_order'] == ['group 1', 'trial_day1',
                                          'trial_day3']


def test_a_failing_gate_add_restores_undo_and_stops_the_load(records):
    _a, b, c = records['gate_add_raises']
    assert b['error'] == ['RuntimeError', 'cannot add bad']
    assert b['state']['_suspend_undo'] == 'prev'
    assert '_schedule_load_settle' not in _order(b)
    assert 'sample_b' in b['state']['_sample_order']
    assert c['error'] is None and '_schedule_load_settle' in _order(c)


def test_no_path_and_held_active_sample(records):
    a, b = records['no_path_sample']
    assert '_note_inbound_landed' not in _order(a)
    # abspath('') is the working folder: registered like a path.
    assert a['state']['_path_to_name'] == {'<CWD>': 'sample_a'}
    assert a['state']['_name_to_path'] == {'sample_a': '<CWD>'}
    assert a['state']['_active_sample'] == 'held'
    assert _calls(a, '_note_unsaved')
    # Meta filed under another path never reaches a path-less sample.
    assert b['state']['_sample_trial']['sample_b'] == 'trial_none'
    assert len(b['state']['_pending_sample_meta']) == 1


def test_the_arrival_already_active_stays_active(records):
    a, a2 = records['active_is_the_arrival']
    assert [e[1] for e in _calls(a, '_set_active_sample')] == ['sample_a']
    assert '_restore_sample_transforms' not in _order(a2)


def test_names_and_paths_registry(records):
    _a, a2 = records['meta_under_other_path']
    assert a2['state']['_path_to_name'] == {
        '<CWD>': 'sample_a', '<TMP>/day1/sample_a.fcs': 'sample_a2'}
    assert a2['state']['_trial_order'] == ['trial_day1', 'trial_none']


if __name__ == '__main__':
    if '--regen' in sys.argv:
        import tempfile
        cls, mods = _new_mixin()
        with tempfile.TemporaryDirectory() as td:
            rec = record_all(cls, mods, all_golden_scenarios(), td)
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=0, ensure_ascii=False)
            fh.write('\n')
        print(f'wrote {len(rec)} scenarios')
