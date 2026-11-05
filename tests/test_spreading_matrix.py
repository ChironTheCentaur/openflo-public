"""Tools > Spreading matrix measures the spillover spread of conventional
compensation from compensated single-stain controls.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import numpy as np
import pytest

import tests._sample_editor as shared
from openflo.comp_spread import (
    compensation_spread_matrix,
    primary_channel,
    worst_pairs,
)
from tests._sample_editor import _set

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


CH = ['FL1-A', 'FL2-A', 'FL3-A']


def _control(j, into, ss, n=3000, seed=0):
    """A compensated single stain for channel ``j``: half negative, half
    bright, spreading ``ss`` x sqrt(signal) into channel ``into``."""
    rng = np.random.default_rng(seed)
    neg = rng.normal(0, 30, (n, 3))
    pos = rng.normal(0, 30, (n, 3))
    pos[:, j] += rng.lognormal(9, 0.3, n)
    pos[:, into] += rng.normal(0, 1, n) * np.sqrt(pos[:, j]) * ss
    return np.vstack([neg, pos])


def test_spread_matrix_recovers_the_spread_and_leaves_missing_na():
    ssm, chans = compensation_spread_matrix(
        {'FL1-A': _control(0, 1, 0.5), 'FL3-A': _control(2, 0, 1.2, seed=1)},
        CH)
    assert chans == CH
    assert ssm[1, 0] == pytest.approx(0.5, rel=0.1)    # FL1 into FL2
    assert ssm[0, 2] == pytest.approx(1.2, rel=0.1)    # FL3 into FL1
    assert ssm[2, 0] < 0.1 and ssm[0, 0] == 0
    assert np.isnan(ssm[:, 1]).all()                   # no FL2 control
    assert worst_pairs(ssm, CH, top=1) == [('FL3-A', 'FL1-A',
                                            pytest.approx(ssm[0, 2]))]


def test_primary_channel_is_the_stain_not_its_spread():
    assert primary_channel(_control(0, 1, 2.0), CH) == 'FL1-A'
    assert primary_channel(_control(2, 1, 2.0), CH) == 'FL3-A'
    un = np.random.default_rng(4).normal(0, 30, (2000, 3))
    assert primary_channel(_control(2, 1, 2.0), CH, unstained=un) == 'FL3-A'
    assert primary_channel(np.zeros((3, 3)), CH) is None


def _comp(ed, names):
    for n in names:
        s = ed._samples[n]
        s.comp_channels = list(CH)
        s.comp_matrix = np.eye(3)


def test_spread_matrix_window_matches_controls_and_computes(editor):
    from openflo.ui_comp_spread import NONE, SpreadMatrixWindow
    root, ed = editor
    c1, c3 = _control(0, 1, 0.5), _control(2, 0, 1.2, seed=1)
    _set(ed, 'alpha_1', **{c: c1[:, i] for i, c in enumerate(CH)})
    _set(ed, 'alpha_2', **{c: c3[:, i] for i, c in enumerate(CH)})
    un = np.random.default_rng(9).normal(0, 30, (500, 3))
    _set(ed, 'beta_1', **{c: un[:, i] for i, c in enumerate(CH)})
    _comp(ed, ['alpha_1', 'alpha_2', 'beta_1'])
    win = SpreadMatrixWindow(ed)
    root.update()
    assert win.channels == CH
    assert win.controls() == {'FL1-A': 'alpha_1', 'FL3-A': 'alpha_2'}
    win.compute()
    disp = win.display()                       # stains on the rows
    assert disp[0, 1] == pytest.approx(0.5, rel=0.1)
    # spread from 3,000 bright events: across seeds the estimates vary by ~5%
    assert disp[2, 0] == pytest.approx(1.2, rel=0.1)
    assert np.isnan(disp[1]).all()
    assert 'FL3-A into FL1-A' in win.status_var.get()
    rows = win.rows()
    assert rows[0] == ['stain'] + CH and rows[2][1:] == ['NA'] * 3
    # A control that is not compensated is left out and named.
    ed._samples['alpha_2'].comp_matrix = None
    win.compute()
    assert 'Skipped (not compensated): alpha_2' in win.status_var.get()
    assert np.isnan(win.display()[2]).all()
    win.control_vars['FL1-A'].set(NONE)
    win.control_vars['FL3-A'].set(NONE)
    win.compute()
    assert win.status_var.get() == "Pick at least one control."
    win.destroy()


def test_spread_matrix_window_needs_compensation(editor):
    from openflo.ui_comp_spread import SpreadMatrixWindow
    root, ed = editor
    win = SpreadMatrixWindow(ed)
    root.update()
    assert win.channels == [] and 'has no compensation' in \
        win.status_var.get()
    win.destroy()
