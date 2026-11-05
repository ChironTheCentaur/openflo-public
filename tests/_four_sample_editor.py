"""Shared module-scoped editor for the tests of set-aside samples, group
colours, arrow-key gate moves and statistics export: four generated
samples (synthetic_a1, synthetic_a2 on Day1; synthetic_b1, synthetic_b2
on Day2) with generic channels, FL1-A x FL2-A plotted.

Each test file that binds ``ed`` gets its own editor, and every test puts
back what it changed.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import editor_or_skip


@pytest.fixture(scope='module')
def ed():
    root, editor = editor_or_skip(accept_confirms=False)
    editor._finish_plot_canvas()
    rng = np.random.default_rng(7)
    cols = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
    for k, (name, trial) in enumerate((('synthetic_a1', 'Day1'),
                                       ('synthetic_a2', 'Day1'),
                                       ('synthetic_b1', 'Day2'),
                                       ('synthetic_b2', 'Day2'))):
        df = pd.DataFrame(rng.normal(1000 + 100 * k, 50, (2000, 4)),
                          columns=cols)
        editor._samples[name] = SimpleNamespace(
            name=name, path=f'/data/{name}.fcs', data=df,
            fluor_channels=['FL1-A', 'FL2-A'],
            channel_labels={c: c for c in cols})
        editor._sample_order.append(name)
        editor._sample_trial[name] = trial
        if trial not in editor._trial_order:
            editor._trial_order.append(trial)
        editor._sample_plot_enabled[name] = True
        editor._sample_gates.setdefault(name, {})
        editor._sample_gate_seq.setdefault(name, 0)
        editor._sample_gate_order.setdefault(name, [])
    editor._channels = cols
    editor._channel_labels = {c: c for c in cols}
    editor._populate_channel_combos()
    editor._set_active_sample('synthetic_a1')
    editor.x_combo.set('FL1-A')
    editor.y_combo.set('FL2-A')
    editor.mode_var.set('dot')
    editor._replot()
    yield editor
    root.destroy()
