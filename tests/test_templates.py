"""The bundled gating-template library stays valid and applies.

Every `templates/*.json` must parse through `read_template_gates`, and the
panel-agnostic cleanup recipes must actually remove events when applied to a
sample (so a shipped 'Standard cleanup' template isn't silently a no-op)."""
from __future__ import annotations

import glob
import os

import numpy as np

from openflo.pipeline import autoclean_keep_mask, read_template_gates

_TEMPLATES = os.path.join(os.path.dirname(__file__), '..', 'src', 'openflo',
                          'template_library')


def _template_paths():
    return sorted(glob.glob(os.path.join(_TEMPLATES, '*.json')))


def test_every_bundled_template_parses():
    paths = _template_paths()
    assert paths, "no templates found"
    for p in paths:
        gates, _labels = read_template_gates(p)
        assert gates, f"{os.path.basename(p)} produced no gates"
        assert all(g.get('kind') for g in gates), os.path.basename(p)


def test_cleanup_standard_is_full_autoclean_recipe():
    gates, _ = read_template_gates(
        os.path.join(_TEMPLATES, 'cleanup_standard.json'))
    assert len(gates) == 1 and gates[0]['kind'] == 'autoclean'
    keys = {m['key'] for m in gates[0]['methods']}
    assert keys == {'debris', 'viability', 'doublets', 'margin',
                    'flow_rate', 'drift'}


def test_cleanup_recipe_removes_events_on_synthetic_data():
    from openflo import synthetic as syn
    df = syn.immunophenotyping_sample(n=8000, seed=1)   # has debris + doublets
    gates, _ = read_template_gates(
        os.path.join(_TEMPLATES, 'cleanup_minimal.json'))
    keep = np.asarray(autoclean_keep_mask(gates[0], df), bool)
    assert 0 < (~keep).sum() < len(df)                  # cleans, doesn't nuke


def test_editor_lists_shipped_library():
    """The editor's Templates ▾ menu source surfaces the bundled cleanup
    recipes (so the curated library is actually reachable, not just on disk)."""
    from openflo.gui import ViewGateEditorWindow as W
    names = {n for n, _d, _p in W._bundled_templates()}
    assert 'Standard cleanup' in names
    assert any('cleanup' in os.path.basename(p).lower()
               for _n, _d, p in W._bundled_templates())


def test_acquisition_qc_template_has_no_biology():
    gates, _ = read_template_gates(
        os.path.join(_TEMPLATES, 'cleanup_acquisition_qc.json'))
    keys = {m['key'] for m in gates[0]['methods']}
    assert keys == {'margin', 'flow_rate', 'drift'}     # time/instrument only
