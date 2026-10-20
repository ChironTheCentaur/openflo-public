"""OpenFlo's numbers against values computed WITHOUT OpenFlo.

Most expected values in the suite came from running OpenFlo, so a wrong
formula and its test can agree. tests/reference/independent_reference.json
was written by scripts/make_reference_values.py from other sources: FlowKit
for compensation, gate membership and the Gating-ML file; the defining
equations of logicle (Moore & Parks 2012) and Gating-ML 2.0 fasinh, solved
independently; scipy.stats for the population statistics. The events are
generated and stored in the file.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from openflo.gating import CHANNEL_STAT_ORDER, population_stats
from openflo.gatingml import _fasinh, read_gatingml
from openflo.pipeline import FlowSample, cumulative_gate_mask, transform_values

REF = json.loads((pathlib.Path(__file__).parent / 'reference'
                  / 'independent_reference.json').read_text(encoding='utf-8'))
CH = REF['channels']
# FlowKit reads "rect AND NOT ell"; OpenFlo's Gating-ML import leaves a
# complemented boolean operand out (with a warning), so it has no count here.
NOT_IMPORTED = {'rect_and_not_ell'}


def _events():
    return pd.DataFrame(np.array(REF['events'], dtype=float), columns=CH)


@pytest.fixture(scope='module')
def imported(tmp_path_factory):
    path = tmp_path_factory.mktemp('gml') / 'flowkit_gates.xml'
    path.write_text(REF['gatingml'], encoding='utf-8')
    gates, warnings = read_gatingml(str(path))
    by_id = {g['_import_id']: dict(g, id=g['_import_id']) for g in gates}
    return by_id, warnings


@pytest.mark.parametrize('name', sorted(set(REF['membership']) - NOT_IMPORTED))
def test_gate_membership_matches_flowkit(imported, name):
    by_id, _warnings = imported
    gid = next(k for k, g in by_id.items() if g['label'] == name)
    mask = np.asarray(cumulative_gate_mask(by_id, gid, _events()), dtype=bool)
    assert np.flatnonzero(mask).tolist() == REF['membership'][name]


def test_every_flowkit_gate_is_imported_or_reported(imported):
    by_id, warnings = imported
    got = {g['label'] for g in by_id.values()}
    assert got == set(REF['membership']) - NOT_IMPORTED
    for name in NOT_IMPORTED:
        assert any(name in w for w in warnings)


def test_compensation_matches_flowkit():
    s = FlowSample.from_dataframe(_events(), name='synthetic_1')
    s.manual_compensate(np.array(REF['spillover'], dtype=float), CH)
    np.testing.assert_allclose(s.data[CH].values,
                               np.array(REF['compensated']),
                               rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize('case', REF['logicle'],
                         ids=lambda c: f"T{c['T']:g}-W{c['W']:g}-M{c['M']:g}-A{c['A']:g}")
def test_logicle_matches_its_defining_equation(case):
    y = transform_values(np.array(case['x']), 'logicle', t=case['T'],
                         m=case['M'], w=case['W'], a=case['A'])
    np.testing.assert_allclose(y, case['y'], rtol=0, atol=1e-10)


@pytest.mark.parametrize('case', REF['fasinh'],
                         ids=lambda c: f"T{c['T']:g}-M{c['M']:g}-A{c['A']:g}")
def test_gatingml_fasinh_matches_the_specification(case):
    y = _fasinh(np.array(case['x']), case['T'], case['M'], case['A'])
    np.testing.assert_allclose(y, case['y'], rtol=0, atol=1e-12)


WANT = {'Count', '%Total', 'Median', 'Mean', 'SD', 'CV', 'MAD', 'rSD'}


@pytest.mark.parametrize('form', ['percentile', 'mad'])
@pytest.mark.parametrize('name', sorted(REF['stats']))
def test_population_statistics_match_scipy(imported, name, form):
    by_id, _warnings = imported
    gid = next(k for k, g in by_id.items() if g['label'] == name)
    row = population_stats('synthetic_1', _events(), {gid: by_id[gid]},
                           [gid], {}, CH, WANT, CHANNEL_STAT_ORDER,
                           stat_forms={'rcv': form})[0]
    ref = REF['stats'][name]
    assert row['Count'] == ref['count']
    assert row['%Total'] == pytest.approx(ref['pct_of_total'], rel=1e-12)
    for ch in CH:
        r = ref[ch]
        for stat, key in (('Median', 'median'), ('Mean', 'mean'),
                          ('SD', 'sd'), ('CV', 'cv'), ('MAD', 'mad')):
            assert row[f'{stat} {ch}'] == pytest.approx(r[key], rel=1e-9), (
                stat, ch)
        if form == 'percentile':
            assert row[f'rSD {ch}'] == pytest.approx(r['rsd_percentile'],
                                                     rel=1e-9)
        else:
            # OpenFlo scales the MAD by the customary 1.4826; scipy uses the
            # exact 1 / Phi^-1(3/4) = 1.4826022..., 1.5e-6 apart.
            assert row[f'rSD {ch}'] == pytest.approx(r['rsd_mad'], rel=2e-6)
            assert row[f'rSD {ch}'] == pytest.approx(1.4826 * r['mad'],
                                                     rel=1e-12)
