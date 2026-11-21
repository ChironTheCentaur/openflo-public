"""Downsampling the data (Display + data) can be undone: setting it back
restores every event, unless the data changed since.

Generated FCS files with synthetic sample names; proved to fail on the old
behaviour by a mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import pandas as pd

import tests._sample_editor as shared

loaded = shared.loaded


def test_downsampling_the_data_can_be_undone(loaded):
    from tests._tk_wait import settle
    root, ed, _tmp = loaded
    for n in ed._samples:
        ed._sample_plot_enabled[n] = True
    small = ed._samples['sample_c']
    small.data = small.data.iloc[:1000].reset_index(drop=True)
    full = {n: ed._samples[n].data.copy() for n in ('sample_a', 'sample_b')}
    ed._ds_mode_var.set('Display + data')
    ed._on_ds_mode_changed()
    settle(root, 0.2)
    assert [len(ed._samples[n].data) for n in ('sample_a', 'sample_b')] \
        == [1000, 1000]
    ed._ds_mode_var.set('Off')
    ed._on_ds_mode_changed()
    settle(root, 0.2)
    for n in ('sample_a', 'sample_b'):
        pd.testing.assert_frame_equal(ed._samples[n].data, full[n])
    assert 'Restored 2 sample(s)' in ed.status_var.get()
    # Data changed after the trim stays as it is: sample_b changed in place
    # (as clustering adds a column), sample_a replaced.
    ed._ds_mode_var.set('Display + data')
    ed._on_ds_mode_changed()
    b = ed._samples['sample_b']
    b.data['FL2-A'] = b.data['FL2-A'] * 2
    ed._data_changed('sample_b')
    a = ed._samples['sample_a']
    a.data = a.data.copy()
    ed._ds_mode_var.set('Off')
    ed._on_ds_mode_changed()
    assert len(ed._samples['sample_a'].data) == 1000
    assert len(ed._samples['sample_b'].data) == 1000
    assert 'sample_a, sample_b changed after the trim' in \
        ed.status_var.get()
