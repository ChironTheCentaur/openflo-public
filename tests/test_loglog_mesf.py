"""MESF calibration can fit log-log, which follows a power law the linear
fit misses; the Calibration dialog fits log-log by default.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import numpy as np
import pytest

import tests._sample_editor as shared
from tests._sample_editor import _set

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def test_loglog_mesf_fit_follows_a_power_law_the_linear_fit_misses():
    from openflo.calibration import apply_calibration, fit_mesf_calibration
    mfi = np.array([40.0, 400.0, 4_000.0, 40_000.0, 400_000.0])
    known = 3.0 * mfi ** 0.9                  # a slightly non-linear detector
    c = fit_mesf_calibration(mfi, known, model='loglog')
    assert c['model'] == 'loglog' and c['n'] == 5
    assert c['slope'] == pytest.approx(0.9)
    assert c['intercept'] == pytest.approx(np.log10(3.0))
    got = apply_calibration(mfi, c['slope'], c['intercept'], model='loglog')
    assert got == pytest.approx(known)
    lin = fit_mesf_calibration(mfi, known)
    assert lin['model'] == 'linear'
    low = apply_calibration(mfi[:1], lin['slope'], lin['intercept'])
    assert abs(low[0] - known[0]) / known[0] > 1.0      # the dim end is lost
    # Non-positive MFI has no log-log value: 0 when clipped, else NaN.
    assert list(apply_calibration([0.0, -5.0], 1.0, 0.0, model='loglog')) \
        == [0.0, 0.0]
    assert np.isnan(apply_calibration([0.0], 1.0, 0.0, clip=False,
                                      model='loglog')).all()
    c2 = fit_mesf_calibration([0.0, 10.0, 100.0], [5.0, 10.0, 100.0],
                              model='loglog')
    assert c2['n'] == 2                       # the zero peak is left out
    with pytest.raises(ValueError):
        fit_mesf_calibration(mfi, known, model='cubic')


def test_calibration_dialog_fits_log_log_by_default(editor):
    from openflo.ui_calibration import CalibrationDialog
    root, ed = editor
    _set(ed, 'alpha_1', **{'FL1-A': np.full(400, 400.0)})
    dlg = CalibrationDialog(ed)
    root.update()
    assert dlg.model_var.get() == 'loglog'
    mfi = [40.0, 400.0, 4_000.0, 40_000.0]
    dlg.txt.delete('1.0', 'end')
    dlg.txt.insert('1.0', '\n'.join(f"{m}\t{3.0 * m ** 0.9}" for m in mfi))
    dlg._fit()
    assert dlg._cal['model'] == 'loglog'
    assert dlg._cal['slope'] == pytest.approx(0.9)
    assert 'log value' in dlg.result.cget('text')
    dlg.apply_var.set(ed._fmt_channel('FL1-A'))
    dlg._apply()
    col = next(c for c in ed._samples['alpha_1'].data.columns
               if c.startswith('MESF:'))
    assert ed._samples['alpha_1'].data[col].iloc[0] == \
        pytest.approx(3.0 * 400 ** 0.9)
    dlg.model_var.set('linear')
    dlg._fit()
    assert dlg._cal['model'] == 'linear' and 'log' not in \
        dlg.result.cget('text')
    dlg.destroy()
