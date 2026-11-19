"""The steps of ComputeMixin._finish_clustering on a recording stand-in
editor: their order (the name reset, the one undo step, the prune and the
import must run in that order), which samples lose their cached work, how
samples split into done and skipped, the embedding switch and every part
of the status line.

The stand-in records each editor call with its arguments; no Tk, no data.
Generated sample names only.
"""
from types import SimpleNamespace

import pytest

from openflo.editor_compute import ComputeMixin

LABEL = 'flowsom_cluster'


class _Var:
    def __init__(self, log, name):
        self._log, self._name = log, name

    def set(self, v):
        self._log.append((self._name, v))


class _Editor(ComputeMixin):
    """Records every call _finish_clustering makes on the editor."""

    def __init__(self, samples, reset=(), pruned=0, kept=()):
        self.log = []
        self._samples = {n: SimpleNamespace(data=SimpleNamespace(columns=c))
                         for n, c in samples.items()}
        self._channel_scale = {}
        self._clustering_busy = True
        self._reset, self._pruned, self._kept = list(reset), pruned, list(kept)
        for n in ('status_var', 'mode_var', 'x_combo', 'y_combo',
                  'color_combo'):
            setattr(self, n, _Var(self.log, n))

    def _end_busy(self):
        self.log.append(('_end_busy', self._clustering_busy))

    def _data_changed(self, names, keep_batch_norm=False):
        self.log.append(('_data_changed', names, keep_batch_norm))

    def _batch_norm_invalidate(self, names):
        self.log.append(('_batch_norm_invalidate', names))

    def _refresh_channel_choices(self):
        self.log.append(('_refresh_channel_choices',))

    def _cluster_label_column(self, method):
        return LABEL                  # one column name, whatever the method

    def _schedule_replot(self, delay):
        self.log.append(('_schedule_replot', delay))

    def _reset_population_names(self, col, changed):
        self.log.append(('_reset_population_names', col, changed))
        return self._reset

    def _checkpoint(self):
        self.log.append(('_checkpoint',))

    def _prune_stale_populations(self, col, changed):
        self.log.append(('_prune_stale_populations', col, changed))
        return self._pruned, self._kept

    def _import_populations(self, col):
        self.log.append(('_import_populations', col))

    def _stamp_clustering(self, method, emb_prefix, done, embedded,
                          channels=None):
        self.log.append(('_stamp_clustering', method, emb_prefix, done,
                         embedded, channels))

    def _audit(self, action, **kw):
        self.log.append(('_audit', action, kw))

    def _fmt_channel(self, c):
        return f'<{c}>'

    def calls(self):
        return [e[0] for e in self.log]

    def entry(self, name):
        (hit,) = [e for e in self.log if e[0] == name]
        return hit

    def status(self):
        return [e[1] for e in self.log if e[0] == 'status_var'][-1]


def _with(*extra):
    return ['FL1-A', LABEL, *extra]


def test_steps_run_in_order_with_one_undo_step_after_the_reset():
    ed = _Editor({'s1': _with('UMAP1', 'UMAP2')})
    ed._finish_clustering('flowsom', 'UMAP', ['s1'],
                          outcome={'s1': 'changed'}, embedded=['s1'],
                          channels=['FL1-A'], settings={'grid': 10})
    assert ed.calls() == [
        '_end_busy', '_data_changed', '_batch_norm_invalidate',
        '_refresh_channel_choices',
        '_reset_population_names', '_checkpoint',
        '_prune_stale_populations', '_import_populations',
        '_stamp_clustering',
        'mode_var', 'x_combo', 'y_combo', 'color_combo',
        '_schedule_replot', '_audit', 'status_var']
    assert ed.entry('_end_busy') == ('_end_busy', False)   # cleared first
    assert ed.entry('_stamp_clustering') == (
        '_stamp_clustering', 'flowsom', 'UMAP', ['s1'], ['s1'], ['FL1-A'])
    assert ed.entry('_audit') == ('_audit', 'cluster', {
        'method': 'flowsom', 'column': LABEL, 'n_samples': 1,
        'samples': ['s1'], 'embedding': 'UMAP', 'grid': 10})


def test_cached_work_is_dropped_where_the_run_wrote():
    samples = {n: _with() for n in ('s1', 's2', 's3')}
    ed = _Editor(samples)
    ed._finish_clustering('flowsom', None, ['s1', 's2', 's3', 'gone'],
                          outcome={'s1': 'done', 's2': 'skipped',
                                   's3': 'skipped'},
                          embedded=['s3'])
    assert ed.entry('_data_changed') == (
        '_data_changed', ['s1', 's2', 's3'], True)
    # s2 was skipped and got no embedding: it keeps its batch-norm undo.
    assert ed.entry('_batch_norm_invalidate') == (
        '_batch_norm_invalidate', ['s1', 's3'])


def test_without_an_outcome_every_loaded_target_is_invalidated():
    ed = _Editor({'s1': _with(), 's2': ['FL1-A']})
    ed._finish_clustering('leiden', None, ['s2', 's1', 'gone'])
    assert ed.entry('_batch_norm_invalidate') == (
        '_batch_norm_invalidate', ['s2', 's1'])


def test_done_needs_the_label_column_and_no_skip():
    ed = _Editor({'s1': _with(), 's2': ['FL1-A'], 's3': _with()})
    ed._finish_clustering('flowsom', None, ['s1', 's2', 's3', 'gone'],
                          outcome={'s1': 'changed', 's3': 'skipped'})
    assert ed.entry('_reset_population_names') == (
        '_reset_population_names', LABEL, ['s1'])
    assert ed.status().startswith(
        "flowsom done on 1 sample(s) — populations imported from "
        f"'{LABEL}'. Toggle them in the tree. No labels on s2, s3: too few "
        "finite events or no usable channels (FlowSOM needs at least "
        "grid × grid events).")


def test_only_changed_samples_get_reset_and_pruned():
    ed = _Editor({'s1': _with(), 's2': _with()})
    ed._finish_clustering('flowsom', None, ['s1', 's2'],
                          outcome={'s1': 'done', 's2': 'changed'})
    assert ed.entry('_reset_population_names')[2] == ['s2']
    assert ed.entry('_prune_stale_populations')[2] == ['s2']


def test_nothing_clustered_imports_and_audits_nothing():
    ed = _Editor({'s1': ['FL1-A'], 's2': _with()})
    ed._finish_clustering('leiden', 'UMAP', ['s1', 's2'],
                          outcome={'s2': 'skipped'})
    assert ed.calls() == [
        '_end_busy', '_data_changed', '_batch_norm_invalidate',
        '_refresh_channel_choices', '_schedule_replot', 'status_var']
    assert ed.status() == (
        "leiden produced no labels on s1, s2: too few finite events or no "
        "usable channels. Nothing was imported.")


def test_the_axes_switch_to_this_runs_embedding_on_a_linear_scale():
    ed = _Editor({'s1': _with('UMAP1', 'UMAP2')})
    ed._finish_clustering('leiden', 'UMAP', ['s1'])
    sets = [e for e in ed.log if e[0] in ('mode_var', 'x_combo', 'y_combo',
                                          'color_combo')]
    assert sets == [('mode_var', 'dot'), ('x_combo', '<UMAP1>'),
                    ('y_combo', '<UMAP2>'),
                    ('color_combo', f'<{LABEL}>')]
    assert ed._channel_scale == {'UMAP1': 'linear', 'UMAP2': 'linear'}
    assert ed.entry('_audit')[2]['embedding'] == 'UMAP'


@pytest.mark.parametrize('emb, embedded, cols, stamped', [
    ('UMAP', ['s9'], _with('UMAP1', 'UMAP2'), []),  # this run embedded others
    ('UMAP', None, _with(), []),                    # no embedding column
    (None, ['s1'], _with('UMAP1', 'UMAP2'), ['s1']),  # no prefix to show
])
def test_no_axes_switch_where_this_run_wrote_no_embedding(emb, embedded,
                                                          cols, stamped):
    ed = _Editor({'s1': cols})
    ed._finish_clustering('leiden', emb, ['s1'], embedded=embedded)
    assert 'mode_var' not in ed.calls()
    assert ed._channel_scale == {}
    assert ed.entry('_audit')[2]['embedding'] == 'none'
    assert ed.entry('_stamp_clustering')[4] == stamped


def test_status_line_names_reset_removed_and_kept_populations():
    ed = _Editor({'s1': _with(), 's2': _with()}, reset=['s1'], pruned=2,
                 kept=[('s2', 'pop 3', 'used by AND1')])
    ed._finish_clustering('phenograph', None, ['s1', 's2'],
                          outcome={'s1': 'changed', 's2': 'changed'})
    col = LABEL
    assert ed.status() == (
        f"phenograph done on 2 sample(s) — populations imported from "
        f"'{col}'. Toggle them in the tree."
        f" The {col} ids changed on s1, so the previous run's population "
        "names were reset."
        f" Removed 2 {col} population(s) whose id no longer occurs."
        f" Kept 1 empty {col} population(s) that other gates use: "
        "'pop 3' in s2, used by AND1.")


def test_a_plain_run_says_only_what_was_imported():
    ed = _Editor({'s1': _with()})
    ed._finish_clustering('leiden', None, ['s1'], outcome={'s1': 'done'})
    assert ed.status() == (
        f"leiden done on 1 sample(s) — populations imported from '{LABEL}'. "
        "Toggle them in the tree.")


def test_done_keeps_the_target_order():
    ed = _Editor({'s1': _with(), 's2': _with(), 's3': _with()})
    ed._finish_clustering('leiden', None, ['s3', 's1', 's2'])
    assert ed.entry('_stamp_clustering')[3] == ['s3', 's1', 's2']
    assert ed.entry('_audit')[2]['samples'] == ['s3', 's1', 's2']


def test_every_target_skipped_keeps_every_batch_norm_undo():
    ed = _Editor({'s1': _with(), 's2': _with()})
    ed._finish_clustering('leiden', None, ['s1', 's2'],
                          outcome={'s1': 'skipped', 's2': 'skipped'})
    assert ed.entry('_batch_norm_invalidate') == (
        '_batch_norm_invalidate', [])


def test_status_lists_several_skipped_reset_and_kept_in_order():
    col = LABEL
    ed = _Editor({n: _with() for n in ('s1', 's2', 's3', 's4')},
                 reset=['s1', 's2'],
                 kept=[('s1', 'pop 3', 'used by AND1'),
                       ('s2', 'pop 5', 'used by OR2')])
    ed._finish_clustering('leiden', None, ['s1', 's2', 's3', 's4'],
                          outcome={'s1': 'changed', 's2': 'changed',
                                   's3': 'skipped', 's4': 'skipped'})
    assert ed.status() == (
        f"leiden done on 2 sample(s) — populations imported from '{col}'. "
        "Toggle them in the tree."
        " No labels on s3, s4: too few finite events or no usable channels."
        f" The {col} ids changed on s1, s2, so the previous run's population "
        "names were reset."
        f" Kept 2 empty {col} population(s) that other gates use: "
        "'pop 3' in s1, used by AND1; 'pop 5' in s2, used by OR2.")
