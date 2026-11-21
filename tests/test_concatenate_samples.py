"""File > Concatenate samples joins several samples into one, each event
tagged with its source.

Generated data, synthetic sample names and generic channels (FL1-A ...);
proved to fail on the old behaviour by a mutation check (see
docs/test-assurance/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tests._sample_editor as shared

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def _frame(seed, n):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FSC-A': rng.random(n) * 1e5,
                         'FL1-A': rng.random(n) * 1e4,
                         'FL2-A': rng.random(n) * 1e4})


def test_concatenate_samples_tags_each_event():
    from openflo.pipeline import FlowSample
    from openflo.ui_concatenate import SOURCE_COLUMN, concatenate_samples
    a = FlowSample.from_dataframe(_frame(1, 30), name='s_a')
    b = FlowSample.from_dataframe(_frame(2, 20), name='s_b')
    out = concatenate_samples([a, b], 'joined')
    assert len(out.data) == 50
    assert out.data[SOURCE_COLUMN].tolist() == [1] * 30 + [2] * 20
    assert out.data['FL1-A'].tolist() == \
        a.data['FL1-A'].tolist() + b.data['FL1-A'].tolist()
    assert SOURCE_COLUMN not in out.fluor_channels
    assert out.metadata['OPENFLO_CONCAT_SOURCES'] == '1=s_a; 2=s_b'
    c = FlowSample.from_dataframe(_frame(3, 10).drop(columns='FL2-A'),
                                  name='s_c')
    with pytest.raises(ValueError, match='same channels'):
        concatenate_samples([a, c], 'bad')
    d = FlowSample.from_dataframe(_frame(4, 10), name='s_d')
    d.data_transforms = {'FL1-A': {'method': 'logicle'}}
    with pytest.raises(ValueError, match='same scales'):
        concatenate_samples([a, d], 'bad')
    with pytest.raises(ValueError):
        concatenate_samples([a], 'bad')


def test_concatenate_from_the_file_menu(editor, monkeypatch):
    from openflo.pipeline import FlowSample
    from openflo.ui_concatenate import ConcatenateDialog
    root, ed = editor
    for i, n in enumerate(('alpha_1', 'alpha_2', 'beta_1')):
        s = FlowSample.from_dataframe(_frame(i, 100 + i), name=n)
        ed._samples[n] = s
    monkeypatch.setattr(type(ed), '_refresh_gate_list', lambda self: None)
    dlg = ConcatenateDialog(ed)
    dlg.listbox.selection_clear(0, 'end')
    dlg.listbox.selection_set(0)
    dlg.listbox.selection_set(2)
    dlg.apply()
    root.update()
    assert 'Concatenated' in ed._samples
    assert len(ed._samples['Concatenated'].data) == 100 + 102
    assert ed._trial_for('Concatenated') == ed._trial_for('alpha_1')
    assert ed.title().startswith('*')
    dlg = ConcatenateDialog(ed)
    assert dlg.name_var.get() == 'Concatenated 2'
    dlg.listbox.selection_clear(0, 'end')
    dlg.listbox.selection_set(0)
    dlg.apply()
    assert 'two or more' in dlg.note_var.get()
    dlg.destroy()
