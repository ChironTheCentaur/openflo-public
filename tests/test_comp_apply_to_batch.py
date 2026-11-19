"""Compensation "Apply to: this batch" -- the samples acquired together.

Apply reached only the active sample, so a set of files needed one Apply
each. Grouping by folder or export time would also be wrong: files exported
together can come from different days and voltages, where one matrix does
not hold. The batch is read from each file's acquisition keywords instead
(instrument, $DATE, $PnV, stored spillover).
"""
import numpy as np
import pandas as pd
import pytest

from openflo.acquisition import acquisition, batch_of, mismatch
from tests.test_compensation_editor_apply import (
    COLS,
    FL,
    M,
    _comp_window,
    _editor_or_skip,
    _load,
)

RUN = {'CYT': 'Synthetic Analyzer', 'CYTSN': 'SN-0001', 'DATE': '09-OCT-2026'}
VOLTS = {'FLA-A': 450, 'FLB-A': 500, 'FLC-A': 520}


def _meta(keywords=None, volts=None):
    """An FCS TEXT dict as FlowIO returns it (no '$', lower case)."""
    md = {str(k).lower(): v for k, v in (keywords or {}).items()}
    names = ['FSC-A', 'SSC-A'] + FL
    for i, c in enumerate(names, 1):
        md[f'p{i}n'] = c
        if volts and c in volts:
            md[f'p{i}v'] = str(volts[c])
    return md


def test_same_instrument_date_and_voltages_is_one_batch():
    meta = {'a': _meta(RUN, VOLTS), 'b': _meta(RUN, VOLTS),
            'c': _meta(RUN, VOLTS)}
    members, left_out = batch_of('b', meta, FL)
    assert members == ['b', 'a', 'c'] and left_out == {}


@pytest.mark.parametrize('change, why', [
    ({'DATE': '10-OCT-2026'}, 'acquired on 10-OCT-2026'),
    ({'CYTSN': 'SN-0002'}, 'another instrument'),
    ({'SPILL': '3,FLA-A,FLB-A,FLC-A,1,0.3,0,0,1,0,0,0,1'}, None),
])
def test_a_different_acquisition_is_left_out(change, why):
    meta = {'a': _meta(RUN, VOLTS), 'b': _meta(dict(RUN, **change), VOLTS)}
    members, left_out = batch_of('a', meta, FL)
    if why is None:                       # a spillover only one file stores
        assert members == ['a', 'b']
    else:
        assert members == ['a'] and why in left_out['b']


def test_other_voltages_or_stored_matrices_are_left_out():
    spill = '3,FLA-A,FLB-A,FLC-A,1,0.2,0,0,1,0,0,0,1'
    a = acquisition(_meta(dict(RUN, SPILL=spill), VOLTS))
    hot = acquisition(_meta(dict(RUN, SPILL=spill), dict(VOLTS, **{'FLB-A': 610})))
    other = acquisition(_meta(dict(RUN, SPILL=spill.replace('0.2', '0.3')), VOLTS))
    assert 'other voltages (FLB-A)' == mismatch(a, hot, FL)
    assert 'another stored spillover matrix' == mismatch(a, other, FL)
    # A voltage on a channel the matrix doesn't touch doesn't matter.
    assert mismatch(a, acquisition(_meta(dict(RUN, SPILL=spill),
                                         dict(VOLTS, **{'FSC-A': 300}))),
                    FL) is None


def test_files_without_acquisition_keywords_are_not_guessed_into_a_batch():
    meta = {'a': _meta(), 'b': _meta(), 'c': _meta(RUN, VOLTS)}
    assert batch_of('a', meta, FL)[0] == ['a']
    members, left_out = batch_of('c', meta, FL)
    assert members == ['c']
    assert left_out == {'a': 'no acquisition keywords',
                        'b': 'no acquisition keywords'}


def test_dollar_spelled_keywords_are_read_too():
    md = {'$CYTSN': 'SN-0001', '$DATE': '09-oct-2026', '$P3N': 'FLA-A',
          '$P3V': '450'}
    acq = acquisition(md)
    assert acq.instrument == 'SN SN-0001' and acq.date == '09-OCT-2026'
    assert acq.voltages == {'FLA-A': 450.0}


# ── In the editor ────────────────────────────────────────────────────────

def _write(path, T, keywords, volts, seed):
    from openflo.fcs_export import write_fcs
    rng = np.random.default_rng(seed)
    scat = rng.uniform(20000, 200000, (len(T), 2))
    write_fcs(pd.DataFrame(np.hstack([scat, T @ M]), columns=pd.Index(COLS)),
              str(path), keywords=keywords, voltages=volts)


def test_apply_to_batch_compensates_the_batch_and_nothing_else(tmp_path):
    root, ed = _editor_or_skip()
    try:
        rng = np.random.default_rng(5)
        T = rng.uniform(100, 20000, (1200, 3))
        _write(tmp_path / 'a.fcs', T, RUN, VOLTS, 1)
        _write(tmp_path / 'b.fcs', T, RUN, VOLTS, 2)
        _write(tmp_path / 'c.fcs', T, dict(RUN, DATE='10-OCT-2026'), VOLTS, 3)
        _load(root, ed, [(n, tmp_path / f'{n}.fcs') for n in 'abc'])
        before_c = ed._samples['c'].data[FL].to_numpy(float).copy()
        ed._set_active_sample('a')
        win = _comp_window(ed)
        win._set_matrix(FL, M)
        assert win._members == ['a', 'b']
        assert win.scope_var.get() == 'batch'
        assert '2 samples: a, b' in win._batch_var.get()
        assert 'c: acquired on 10-OCT-2026' in win._batch_tip_text
        win._apply()
        a = ed._samples['a'].data[FL].to_numpy(float)
        b = ed._samples['b'].data[FL].to_numpy(float)
        np.testing.assert_allclose(a, b, atol=1e-6)    # same T, same matrix
        assert ed._samples['b'].comp_channels == FL
        np.testing.assert_array_equal(
            ed._samples['c'].data[FL].to_numpy(float), before_c)
        st = ed.status_var.get()
        assert st.startswith("Applied 3×3 matrix to 2 sample(s) in this "
                             "batch (a, b)."), st
    finally:
        root.destroy()


def test_this_sample_still_applies_to_one(tmp_path):
    root, ed = _editor_or_skip()
    try:
        rng = np.random.default_rng(6)
        T = rng.uniform(100, 20000, (800, 3))
        _write(tmp_path / 'a.fcs', T, RUN, VOLTS, 1)
        _write(tmp_path / 'b.fcs', T, RUN, VOLTS, 2)
        _load(root, ed, [(n, tmp_path / f'{n}.fcs') for n in 'ab'])
        before_b = ed._samples['b'].data[FL].to_numpy(float).copy()
        ed._set_active_sample('a')
        win = _comp_window(ed)
        win._set_matrix(FL, M)
        win.scope_var.set('sample')
        win._apply()
        assert ed._samples['a'].comp_channels == FL
        np.testing.assert_array_equal(
            ed._samples['b'].data[FL].to_numpy(float), before_b)
        assert ed.status_var.get().startswith("Applied 3×3 matrix to 'a'")
    finally:
        root.destroy()
