"""Generated BD FACSDiva experiment XML for the FACSDiva import tests.

``diva_xml`` writes a ``<bdfacs>`` file laid out as FACSDiva's "Export >
Experiment" writes one (experiment > specimen > tube, each tube with its
<data_filename>, <instrument_settings> and <gates>). Everything is made up
here: generic channel names (FSC-A, SSC-A, FL1-A, FL2-A), generic specimen
and tube names, synthetic FCS files. No real export is used.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from tests._fcs_writer import write_float_fcs

CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
FLUOR = ['FL1-A', 'FL2-A']
# FACSDiva's compensation coefficients, parameter by parameter: FL1-A's
# compensated value is 1.02*FL1 - 0.25*FL2, FL2-A's is -0.08*FL1 + 1.0*FL2.
# Not symmetric, so a transposed reading gives other numbers.
COEFFS = {'FL1-A': [1.02, -0.25], 'FL2-A': [-0.08, 1.0]}


def make_events(seed=0, n=4000):
    """Synthetic events (n x 4) for CHANNELS: scatter spread over the whole
    linear range, fluorescence from below zero to past the top."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0, 262143, n), rng.uniform(0, 262143, n),
        np.concatenate([rng.normal(-200, 150, n // 4),
                        10 ** rng.uniform(1, 5.5, n - n // 4)]),
        np.concatenate([rng.normal(50, 300, n // 4),
                        10 ** rng.uniform(1, 5.5, n - n // 4)]),
    ]).astype(np.float32)


def make_fcs(path, seed=0, n=4000, keywords=None):
    """A synthetic FCS file at `path` (folders made), with extra TEXT
    `keywords` such as FACSDiva writes ($FIL, $SRC, TUBE NAME, ...)."""
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    return write_float_fcs(str(path), make_events(seed, n), CHANNELS,
                           metadata_dict=dict(keywords or {}))


def _text(parent, tag, text):
    el = ET.SubElement(parent, tag)
    el.text = str(text)
    return el


def _settings(parent, comp_on=True, coeffs=None):
    s = ET.SubElement(parent, 'instrument_settings')
    _text(s, 'compensation_enabled', 'true' if comp_on else 'false')
    _text(s, 'use_auto_biexp_scale', 'true')
    coeffs = COEFFS if coeffs is None else coeffs
    for ch in CHANNELS:
        p = ET.SubElement(s, 'parameter', name=ch)
        fl = ch in FLUOR
        _text(p, 'is_log', 'true' if fl else 'false')
        _text(p, 'can_be_compensated', 'true' if fl else 'false')
        if fl and ch in coeffs:
            c = ET.SubElement(p, 'compensation')
            for v in coeffs[ch]:
                _text(c, 'compensation_coefficient', v)
    return s


def region_gate(name, parent, xparm, points, *, yparm=None,
                rtype='POLYGON_REGION', gtype='Region_Classifier',
                x=(False, False, 0), y=(False, False, 0), inputs=None):
    """A <gate> dict for diva_xml: `x` / `y` are (scaled, log, scale value)
    for the axis; `points` the region's (x, y) points as stored."""
    return {'name': name, 'parent': parent, 'type': gtype, 'rtype': rtype,
            'xparm': xparm, 'yparm': yparm, 'points': points, 'x': x,
            'y': y, 'inputs': inputs}


def boolean_gate(name, parent, gtype, inputs):
    return {'name': name, 'parent': parent, 'type': gtype, 'inputs': inputs}


def _gate(gates_el, g, events):
    full = (g['parent'] + '\\' + g['name']) if g['parent'] else g['name']
    el = ET.SubElement(gates_el, 'gate', fullname=full, type=g['type'])
    _text(el, 'name', g.get('title', g['name']))
    _text(el, 'num_events', events if not g['parent'] else 0)
    if g['parent']:
        _text(el, 'parent', g['parent'])
    if 'rtype' in g:
        attrs = {'name': '', 'xparm': g['xparm'], 'type': g['rtype']}
        if g['yparm']:
            attrs['yparm'] = g['yparm']
        reg = ET.SubElement(el, 'region', attrs)
        pts = ET.SubElement(reg, 'points')
        for px, py in g['points']:
            ET.SubElement(pts, 'point', x=repr(float(px)), y=repr(float(py)))
        for ax in ('x', 'y'):
            scaled, log, r = g[ax]
            _text(el, f'is_{ax}_parameter_scaled', str(scaled).lower())
            _text(el, f'is_{ax}_parameter_log', str(log).lower())
            _text(el, f'{ax}_parameter_scale_value', r)
    inputs = g.get('inputs') or ([g['parent']] if g['parent'] else [])
    for i in inputs:
        _text(el, 'input', i)
    return el


def diva_xml(path, specimens, *, log_decades=5, experiment='Exp_001',
             shared_settings=False):
    """Write a FACSDiva experiment XML at `path`. `specimens` is a list of
    (specimen name, [tube dict]); a tube dict has 'name', 'file' (its
    <data_filename>), 'gates' (gate dicts after an 'All Events' root),
    'events' (the root's num_events), 'comp_on' and 'coeffs' (None: the
    default COEFFS), and 'settings' False to leave them out ('empty': a
    settings block with no parameters). `log_decades` None leaves the
    experiment's out; a gate dict's 'title' is its <name> when that
    differs from the last part of its path.
    `shared_settings` also writes the experiment's own settings, which a
    tube without any falls back to (compensation on, COEFFS)."""
    root = ET.Element('bdfacs', version='Version 0.0')
    exp = ET.SubElement(root, 'experiment', name=experiment)
    if log_decades is not None:
        _text(exp, 'log_decades', log_decades)
    if shared_settings:
        _settings(exp)
    for sname, tubes in specimens:
        sp = ET.SubElement(exp, 'specimen', name=sname)
        for t in tubes:
            tube = ET.SubElement(sp, 'tube', name=t['name'])
            _text(tube, 'data_filename', t['file'])
            if t.get('settings', True) == 'empty':
                _text(ET.SubElement(tube, 'instrument_settings'),
                      'compensation_enabled', 'true')
            elif t.get('settings', True):
                _settings(tube, t.get('comp_on', True), t.get('coeffs'))
            gates = ET.SubElement(tube, 'gates')
            _gate(gates, {'name': 'All Events', 'parent': None,
                          'type': 'EventSource_Classifier'},
                  t.get('events', 4000))
            for g in t.get('gates', []):
                _gate(gates, g, 0)
    ET.ElementTree(root).write(str(path), encoding='UTF-8',
                               xml_declaration=True)
    return str(path)
