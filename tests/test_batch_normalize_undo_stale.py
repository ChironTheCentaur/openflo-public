"""Undo batch normalization never puts back values that no longer fit the
sample (scope audit D4).

The values from before a normalization are kept for Tools > Undo batch
normalization. When the sample's data changes afterwards (a channel
transform, compensation, a new column, a trim), those kept values sit on
the old scale while the sample's records describe the new one; putting them
back would make plots, gates and statistics silently wrong. Such a sample
stays as it is and the status names it and says why.

Generated data with synthetic sample names (the CytoNorm dialog fixture);
proved to fail on the old behaviour by a mutation check.
"""
from __future__ import annotations

import threading

import numpy as np

from tests import test_batch_normalize_dialog as dlg

ed = dlg.ed


def batch_norm_restore_status(*a):
    # Imported per call so the other tests run (and fail) on code without it.
    from openflo.editor_tools import batch_norm_restore_status as status
    return status(*a)


def _normalize(ed):
    """Open the window as Tools does (so the editor knows it) and run it."""
    ed._batch_correct_cytonorm()
    w = ed._batch_norm_win
    w.mode_var.set('goal')
    w.meta_var.set(1)
    w._run(sync=True)
    return w


def _rewrite(ed, name, factor=2.0):
    """What every data writer does (compensation, calibration, fixes...):
    rewrite the frame in place, then tell the editor."""
    s = ed._samples[name]
    s.data['FL1-A'] = s.data['FL1-A'] * factor
    ed._data_changed(name)


def test_a_transform_change_after_normalizing_is_not_undone(ed):
    """The audit's path: normalize, switch a channel's transform, Undo."""
    from openflo.channel_transforms import transform_spec
    for s in ed._samples.values():
        s.data_transforms = {c: transform_spec('linear') for c in dlg.CH}
    ed._channel_transform = dict.fromkeys(dlg.CH, 'linear')
    w = _normalize(ed)
    # The transform's work runs on the Tk thread here (as the transform
    # tests do), so its on_done has run when the call returns.
    ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
        on_done(work()) if on_done else work())
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    assert ed._samples['d1_ref'].data_transforms['FL1-A']['method'] == 'asinh'
    after = {n: ed._samples[n].data for n in ed._sample_order}
    # Nothing kept to go back to any more: the window's Restore is grey.
    assert str(w.restore_btn.cget('state')) == 'disabled'
    assert ed._restore_batch_norm() == 0
    for n, df in after.items():
        assert ed._samples[n].data is df            # left as it is
        assert ed._samples[n].batch_norm            # still says normalized
    msg = ed.status_var.get()
    assert msg.startswith('Not undone for d1_ref, d1_study, d2_ref, '
                          'd2_study: their data changed after normalizing')
    assert 'Reload the FCS files' in msg
    w.destroy()


def test_only_the_changed_sample_stays_and_is_named(ed):
    raw = {n: ed._samples[n].data for n in ed._sample_order}
    w = _normalize(ed)
    _rewrite(ed, 'd2_study')
    changed = ed._samples['d2_study'].data
    assert ed._restore_batch_norm() == 3
    for n in ('d1_ref', 'd1_study', 'd2_ref'):
        assert ed._samples[n].data is raw[n]
        assert ed._samples[n].batch_norm is None
    assert ed._samples['d2_study'].data is changed
    assert ed._samples['d2_study'].batch_norm
    assert ed.status_var.get() == batch_norm_restore_status(
        3, ['d2_study'], True)
    assert ed._audit_log.entries()[-1]['details']['samples'] == [
        'd1_ref', 'd1_study', 'd2_ref']
    # A second Undo says the same about d2_study, not "session closed".
    assert ed._restore_batch_norm() == 0
    assert ed.status_var.get() == batch_norm_restore_status(
        0, ['d2_study'], True)
    w.destroy()


def test_a_frame_replaced_without_notice_is_not_undone(ed):
    """A writer that swaps the frame and forgets _data_changed: the kept
    values still belong to the frame the normalization made, not this one."""
    raw = {n: ed._samples[n].data for n in ed._sample_order}
    w = _normalize(ed)
    swapped = ed._samples['d1_study'].data.copy()
    ed._samples['d1_study'].data = swapped
    assert ed._restore_batch_norm() == 3
    assert ed._samples['d1_study'].data is swapped
    assert ed._samples['d1_ref'].data is raw['d1_ref']
    assert ed.status_var.get() == batch_norm_restore_status(
        3, ['d1_study'], True)
    w.destroy()


def test_normalizing_again_after_a_change_undoes_to_the_changed_values(ed):
    """The second normalization starts from the changed frame, so Undo goes
    back to it (not to the stale values from before the change), and the
    sample is no longer reported as lost."""
    w = _normalize(ed)
    _rewrite(ed, 'd1_ref')
    before_second = {n: ed._samples[n].data for n in ed._sample_order}
    w._run(sync=True)
    assert ed._restore_batch_norm() == 4
    assert ed._samples['d1_ref'].data is before_second['d1_ref']
    assert ed.status_var.get() == batch_norm_restore_status(4, [], False)
    w.destroy()


def test_a_second_run_with_no_change_still_undoes_both(ed):
    raw = {n: ed._samples[n].data for n in ed._sample_order}
    w = _normalize(ed)
    w._run(sync=True)
    assert ed._restore_batch_norm() == 4
    for n, df in raw.items():
        assert ed._samples[n].data is df
    w.destroy()


def test_a_sample_loaded_again_drops_what_was_kept(ed):
    """A sample arriving under the name (reload, removal) starts clean:
    nothing to put back, and not reported as lost."""
    w = _normalize(ed)
    _rewrite(ed, 'd1_ref')
    ed._plotcache_forget('d1_ref')
    ed._plotcache_forget('d2_ref')
    assert 'd1_ref' not in ed._batch_norm_lost
    assert set(ed._batch_norm_originals) == {'d1_study', 'd2_study'}
    assert ed._restore_batch_norm() == 2
    assert 'Not undone' not in ed.status_var.get()
    w.destroy()


def test_the_window_restore_says_what_the_editor_says(ed):
    w = _normalize(ed)
    _rewrite(ed, 'd2_ref')
    assert str(w.restore_btn.cget('state')) == 'normal'   # 3 still kept
    w._restore()
    assert w.status_var.get() == ed.status_var.get()
    assert 'Not undone for d2_ref:' in w.status_var.get()
    w.destroy()


def test_a_change_off_the_tk_thread_still_drops_the_kept_values(ed):
    """_data_changed may run on a worker thread: the kept values are still
    dropped there, and the window is not touched from that thread."""
    w = _normalize(ed)
    synced = []
    ed._batch_norm_sync_window = lambda: synced.append(
        threading.current_thread())
    t = threading.Thread(target=ed._batch_norm_invalidate, args=(['d1_ref'],))
    t.start()
    t.join()
    assert 'd1_ref' not in ed._batch_norm_originals
    assert 'd1_ref' in ed._batch_norm_lost
    assert synced == []
    ed._batch_norm_invalidate(['d1_study'])
    assert synced == [threading.main_thread()]
    del ed._batch_norm_sync_window
    w.destroy()


def test_normalizing_again_after_a_silent_swap_undoes_to_the_swapped_frame(
        ed):
    """A frame swapped without _data_changed, then normalized again: the
    first kept values belong to the old frame, so Undo goes back to the
    swapped one."""
    w = _normalize(ed)
    swapped = ed._samples['d2_ref'].data.copy()
    ed._samples['d2_ref'].data = swapped
    w._run(sync=True)
    assert ed._restore_batch_norm() == 4
    assert ed._samples['d2_ref'].data is swapped
    w.destroy()


def test_the_status_lines():
    names = [f'sample_{i}' for i in range(7)]
    assert batch_norm_restore_status(2, [], True) == (
        "Batch normalization undone: 2 sample(s) back on their original "
        "values.")
    both = batch_norm_restore_status(1, names, True)
    assert both.startswith("Batch normalization undone: 1 sample(s) back")
    assert ("Not undone for sample_0, sample_1, sample_2, sample_3, "
            "sample_4 and 2 more: their data changed") in both
    assert batch_norm_restore_status(0, names[:1], True).startswith(
        "Not undone for sample_0: ")
    assert batch_norm_restore_status(0, names[:5], True).startswith(
        "Not undone for sample_0, sample_1, sample_2, sample_3, sample_4: ")
    assert batch_norm_restore_status(0, [], True).startswith(
        "Nothing to undo: the values from before batch normalization are "
        "not kept once the session is closed.")
    assert batch_norm_restore_status(0, [], False) == \
        "No batch normalization to undo."


def test_normalized_values_differ_from_the_kept_ones(ed):
    """Guard for the tests above: the normalization really changes values,
    so 'data is raw' checks tell restored from left-alone."""
    raw = float(np.median(ed._samples['d2_ref'].data['FL1-A']))
    w = _normalize(ed)
    assert abs(dlg._median(ed, 'd2_ref') - raw) > 1e-6
    w.destroy()


# ── callers that touch data without changing it keep the undo ─────────────


def test_a_failed_cell_cycle_model_keeps_the_undo(ed):
    """A sample whose cell cycle model failed got nothing written: Undo
    still puts it back (review of #323, B1)."""
    raw = {n: ed._samples[n].data for n in ed._sample_order}
    w = _normalize(ed)
    frames = {'d1_ref': ed._samples['d1_ref'].data}
    ed._finish_cell_cycle('FL1-A', frames, {}, [])
    assert ed._restore_batch_norm() == 4
    assert ed._samples['d1_ref'].data is raw['d1_ref']
    assert 'Not undone' not in ed.status_var.get()
    w.destroy()


def _fake_flowsom(write=True, fail=False):
    """A sample's run_flowsom: writes both label columns, writes nothing
    (a skipped sample) or fails after writing (a run stopping part way)."""
    def run(sample, **_kw):
        if write:
            sample.data['flowsom'] = 0
            sample.data['flowsom_meta'] = 0
        if fail:
            raise RuntimeError('planted clustering failure')
    return run


def _cluster(ed, behaviour):
    """Run FlowSOM on every sample through _run_clustering, each sample's
    run given by ``behaviour[name]`` (write, fail)."""
    import functools
    for name, (write, fail) in behaviour.items():
        s = ed._samples[name]
        s.run_flowsom = functools.partial(_fake_flowsom(write, fail), s)
    ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
        _call(work, on_done, on_error))
    ed._run_clustering('flowsom', True, 30, 10, 5, embedding='None')


def _call(work, on_done, on_error):
    try:
        res = work()
    except RuntimeError as exc:       # the planted failure, to on_error
        on_error(exc)
        return
    on_done(res)


def test_a_failed_clustering_run_drops_only_the_samples_it_reached(ed):
    """The run stops at the second sample: the first two may hold written
    columns and are not undone; the two it never reached are."""
    raw = {n: ed._samples[n].data for n in ed._sample_order}
    w = _normalize(ed)
    _cluster(ed, {'d1_ref': (True, False), 'd1_study': (True, True),
                  'd2_ref': (False, False), 'd2_study': (False, False)})
    assert ed.status_var.get().startswith('Clustering failed')
    assert ed._restore_batch_norm() == 2
    for n in ('d2_ref', 'd2_study'):
        assert ed._samples[n].data is raw[n]
    assert ed.status_var.get() == batch_norm_restore_status(
        2, ['d1_ref', 'd1_study'], True)
    w.destroy()


def test_a_clustering_run_drops_only_the_samples_it_wrote_into(ed):
    """A finished run: a sample it skipped (no labels written) keeps its
    undo; one it labelled does not."""
    raw = {n: ed._samples[n].data for n in ed._sample_order}
    w = _normalize(ed)
    _cluster(ed, {'d1_ref': (True, False), 'd1_study': (False, False),
                  'd2_ref': (False, False), 'd2_study': (False, False)})
    assert ed._restore_batch_norm() == 3
    for n in ('d1_study', 'd2_ref', 'd2_study'):
        assert ed._samples[n].data is raw[n]
    assert ed.status_var.get() == batch_norm_restore_status(
        3, ['d1_ref'], True)
    w.destroy()


def test_a_clustering_error_with_no_sample_list_drops_every_undo(ed):
    w = _normalize(ed)
    ed._clustering_error(RuntimeError('x'))
    assert ed._restore_batch_norm() == 0
    w.destroy()


def test_a_downsample_trim_and_its_restore_keep_the_undo(ed):
    """Display + data trims the normalized frame, Off puts the same frame
    back: Undo then works (refused while trimmed, see the identity guard)."""
    from tests._tk_wait import settle
    raw = {n: ed._samples[n].data for n in ed._sample_order}
    w = _normalize(ed)
    for n in ed._sample_order:
        ed._sample_plot_enabled[n] = True
    small = ed._samples['d1_ref']
    small.data = small.data.iloc[:1000].reset_index(drop=True)
    ed._data_changed('d1_ref')
    ed._ds_mode_var.set('Display + data')
    ed._on_ds_mode_changed()
    settle(ed, 0.2)
    assert len(ed._samples['d2_ref'].data) == 1000
    assert set(ed._batch_norm_originals) == {'d1_study', 'd2_ref',
                                             'd2_study'}
    ed._ds_mode_var.set('Off')
    ed._on_ds_mode_changed()
    settle(ed, 0.2)
    assert ed._restore_batch_norm() == 3
    for n in ('d1_study', 'd2_ref', 'd2_study'):
        assert ed._samples[n].data is raw[n]
    w.destroy()
