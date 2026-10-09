"""Background clustering / embedding orchestration and busy indicator.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import numpy as np

from .editor_base import EditorMixin


def _log_note(head, counts, tail):
    """Status text naming how many events per channel have no value."""
    if not counts:
        return ''
    return head + ', '.join(f"{ch} {n:,} events"
                            for ch, n in counts.items()) + tail


def label_run_outcome(before, df, col):
    """How one clustering call left label column ``col`` of ``df``.

    ``before`` is ``df[col].to_numpy()`` taken just before the call, or None
    when the column did not exist. Returns 'skipped' when the call wrote
    nothing, 'new' for a first labelling, 'same' when it rewrote identical
    labels (a seeded re-run), else 'changed'.

    run_flowsom and run_leiden return without writing when there are too few
    finite events or no usable channels; FlowSOM needs grid x grid events, so
    a 60-event sample on the default 10x10 grid wrote nothing yet was reported
    as done. A skip also leaves an older column in place, so presence alone
    cannot tell a skip from a run. The methods assign a fresh array to the
    column; a skipped call leaves the array we read before untouched, so
    memory shared with ``before`` means nothing was written."""
    if col not in df.columns:
        return 'skipped'
    after = df[col].to_numpy()
    if before is None:
        return 'new'
    if np.may_share_memory(before, after):
        return 'skipped'
    return 'same' if np.array_equal(before, after) else 'changed'


class ComputeMixin(EditorMixin):
    """Launch and finalise clustering / embedding jobs off the UI thread, with a modal busy indicator and channel-transform application."""

    # Label column each clustering method writes.
    _CLUSTER_LABEL_COLUMN = {'phenograph': 'cluster', 'leiden': 'leiden',
                             'flowsom': 'flowsom_meta'}

    def _cluster_label_column(self, method):
        return self._CLUSTER_LABEL_COLUMN.get(method, 'flowsom_meta')

    def _reset_population_names(self, col, samples):
        """Reset the populations of label column ``col`` on ``samples`` to
        their default names, and drop the stored phenotype names for 'cluster'.
        Returns the samples where a name was actually reset.

        Cluster ids are size ranks of one run, not identities: after a re-run
        on other markers, PhenoGraph's 'Cluster 0' went from all 120 cells of
        one planted group to 41 + 28 cells of the two others, yet kept the
        user's name for the old cells. Names given to the previous partition
        therefore cannot carry over to a different one.

        The same reset goes into every saved undo and redo state. Undo
        restores gates, not the label column, so a state saved before the run
        put the old names back on the new partition: two naming steps, a
        re-run, and the second Ctrl+Z restored the first name."""
        reset = [name for name in samples
                 if self._reset_names_in(self._sample_gates,
                                         self._cluster_labels, col, name)]
        for snap in [*self._undo_stack, *self._redo_stack]:
            for name in samples:
                self._reset_names_in(snap['gates'], snap['cluster_labels'],
                                     col, name)
        return reset

    def _reset_names_in(self, sample_gates, cluster_labels, col, name):
        """Reset sample ``name``'s populations of ``col`` to their default
        names in one gate state: the live one or a saved undo / redo state.
        True when a name changed."""
        from .pipeline import cluster_label
        hit = False
        if col == 'cluster':
            stored = cluster_labels.pop(name, None) or {}
            hit = any(str(v) != cluster_label(k) for k, v in stored.items())
        for g in sample_gates.get(name, {}).values():
            if col == 'cluster' and g.get('kind') == 'cluster':
                default = cluster_label(g.get('cluster_id'))
            elif g.get('kind') == 'category' and g.get('channel') == col:
                default = self._population_label(col, g.get('value'))
            else:
                continue
            if g.get('name') != default:
                g['name'] = default
                hit = True
        return hit

    def _prune_stale_populations(self, col, samples):
        """Remove the populations of label column ``col`` on ``samples`` whose
        id no longer occurs in that column. Returns (removed, kept): the count
        removed, and for each stale population left in place because other
        gates refer to it, ``(sample, population name, users)`` where users
        says which gates ('parent of …', 'used by boolean …').

        A re-run with fewer clusters (12, then 8) left 'Cluster 8' to
        'Cluster 11' in the tree, selecting nothing. Only the importer's own
        gates are candidates: 'cluster' gates for 'cluster', else 'category'
        gates on ``col``. One that another gate refers to stays: as the parent
        of a child gate (removing it would orphan the child) or as a boolean
        gate's operand (a missing operand silently changes the boolean:
        skipped, 'Cluster 11 AND CD1+' became all of CD1+, 12 events to 200;
        failed closed, AND, OR and NOT all became empty)."""
        def ours(g):
            if col == 'cluster':
                return g.get('kind') == 'cluster'
            return g.get('kind') == 'category' and g.get('channel') == col

        key = 'cluster_id' if col == 'cluster' else 'value'
        removed, kept = 0, []
        for name in samples:
            live = set(self._sample_cluster_ids(name) if col == 'cluster'
                       else self._sample_label_values(name, col,
                                                      include_sentinel=True))
            gates = self._sample_gates.get(name, {})
            stale = [gid for gid, g in gates.items()
                     if ours(g) and g.get(key) not in live]
            for gid in stale:
                users = (
                    [f"parent of '{g.get('name', o)}'"
                     for o, g in gates.items() if g.get('parent_id') == gid]
                    + [f"used by {g.get('kind', 'gate')} '{g.get('name', o)}'"
                       for o, g in gates.items()
                       if gid in (g.get('operands') or [])])
                if users:
                    kept.append((name, gates[gid].get('name', gid),
                                 ', '.join(users)))
                else:
                    removed += self._purge_gate_subtree(name, gid)
        return removed, kept

    @staticmethod
    def _recorded_method(rec, ch):
        """The transform method a sample's ``data_transforms`` records for
        `ch` ('linear' when it records none; 'unknown' for a column whose
        transform cannot be known, see pipeline.UNKNOWN_SPEC)."""
        return (rec.get(ch) or {}).get('method', 'linear')

    @staticmethod
    def _default_method(sample, ch):
        """The loader's convention for `ch`: logicle on fluor channels,
        linear elsewhere -- what the editor shows for a column whose own
        transform is unknown (as 2.6.1 showed it)."""
        fluor = getattr(sample, 'fluor_channels', []) or []
        return 'logicle' if ch in fluor else 'linear'

    def _seed_method(self, sample, rec, ch):
        """What a column of `sample` seeds the editor with: its recorded
        method, or the default when that is unknown -- an unknown column must
        never put the editor (and every other sample) on 'unknown'."""
        m = self._recorded_method(rec, ch)
        return self._default_method(sample, ch) if m == 'unknown' else m

    def _adopt_sample_transforms(self, sample, first):
        """Make a newly loaded sample and ``_channel_transform`` agree.

        ``_channel_transform`` is what the axes invert and what the Transform
        editor shows; the sample's ``data_transforms`` is what its values
        actually carry. The FIRST sample seeds the editor from its record. A
        LATER one is put on the transforms currently in effect — the loader
        always applies logicle, so a sample loaded after a Transform-editor
        change used to stay logicle while the editor displayed (and later
        inverted) it as the new method. Columns the editor has not seen yet
        take the sample's own transform. A sample without a record (not a
        FlowSample) keeps the loader's convention: logicle on fluor channels.
        A column whose transform is unknown seeds the default and is never
        re-transformed (see _conform_sample_transforms)."""
        rec = getattr(sample, 'data_transforms', None)
        cols = list(sample.data.columns)
        if rec is None:
            if first:
                self._channel_transform = {
                    c: self._default_method(sample, c) for c in cols}
            return
        if first:
            self._channel_transform = {
                c: self._seed_method(sample, rec, c) for c in cols}
            return
        for c in cols:
            self._channel_transform.setdefault(
                c, self._seed_method(sample, rec, c))
        self._conform_sample_transforms(sample)

    def _conform_sample_transforms(self, s):
        """Put every channel of sample `s` on the method ``_channel_transform``
        names, undoing the transform its data carries first (so its record and
        the editor agree). No-op for a sample without a ``data_transforms``
        record, and for a column whose transform is unknown: it cannot be
        undone, so it stays as stored. Returns the channels it changed."""
        rec = getattr(s, 'data_transforms', None)
        if rec is None:
            return []
        by_method = {}
        for ch in s.data.columns:
            have = self._recorded_method(rec, ch)
            want = self._channel_transform.get(ch)
            if want is not None and have != 'unknown' and want != have:
                by_method.setdefault(want, []).append(ch)
        for method, chans in by_method.items():
            s.retransform(chans, method=method)
        live = [n for n, smp in self._samples.items() if smp is s]
        if by_method and live:
            self._data_changed(live)
        return [c for chans in by_method.values() for c in chans]

    @staticmethod
    def _restore_sample_transforms(s, saved):
        """Put sample `s` back on the transforms a session saved for it
        (`saved`, a ``data_transforms`` dict: no entry = linear), undoing what
        its data carries first. A sidecar sample already matches; a sample
        reloaded from its raw FCS arrives on the loader's logicle. No-op for a
        sample without a record, and for a column whose transform is unknown
        on either side (nothing can be converted to or from it). Returns the
        channels it changed."""
        from .pipeline import is_unknown_spec
        rec = getattr(s, 'data_transforms', None)
        if rec is None:
            return []
        changed = [ch for ch in s.data.columns
                   if saved.get(ch) != rec.get(ch)
                   and not is_unknown_spec(saved.get(ch))
                   and not is_unknown_spec(rec.get(ch))]
        for ch in changed:
            s.retransform([ch], **(saved.get(ch) or {'method': 'linear'}))
        return changed

    def _log_would_drop(self, new_methods):
        """``{channel: (n_events, n_total)}`` for each channel `new_methods`
        moves onto 'log' whose loaded samples hold events at or below zero
        (compensated linear value). 'log' gives those events no value (NaN),
        and nothing brings them back: switching the channel away again
        inverts NaN to NaN, and the session sidecar saves it. Measured on a
        compensated channel with half its events negative: logicle -> log ->
        logicle left 7,525 of 30,000 events as NaN, out of every gate."""
        from .pipeline import (
            UnknownScaleError,
            inverse_transform_values,
            linear_values,
        )
        out = {}
        for ch, m in new_methods.items():
            if m != 'log' or self._channel_transform.get(ch) == 'log':
                continue
            bad = total = 0
            for s in self._samples.values():
                if ch not in s.data.columns:
                    continue
                if getattr(s, 'data_transforms', None) is None:
                    # No record: the editor's map says what it carries, as
                    # in _apply_channel_transforms.
                    lin = inverse_transform_values(
                        np.asarray(s.data[ch].values, dtype=float),
                        method=self._channel_transform.get(ch, 'linear'))
                else:
                    try:
                        lin = linear_values(s, ch)
                    except UnknownScaleError:
                        continue          # left as stored, never re-transformed
                bad += int((lin <= 0).sum())
                total += int(lin.size)
            if bad:
                out[ch] = (bad, total)
        return out

    def _apply_channel_transforms(self, new_methods):
        """Re-transform channels across ALL loaded samples by inverting each
        channel's current transform and applying the new one (so no
        re-compensation is needed). The (pure) transforms run off the Tk thread
        and only the resulting column arrays are written to ``s.data`` on the Tk
        thread in on_done — no freeze, no cross-thread write race. Sets its own
        status (was: returned a count to the caller)."""
        from .pipeline import (
            inverse_transform_values,
            is_unknown_spec,
            to_linear,
            transform_spec,
            transform_values,
        )
        changed = {c: m for c, m in new_methods.items()
                   if m != self._channel_transform.get(c, 'linear')}
        if not changed:
            self.status_var.set("No transform changes.")
            return
        samples = list(self._samples.items())     # (name, sample) refs
        # A column whose transform is unknown cannot be undone: it stays as
        # stored, and the status says which.
        as_stored = sorted({f"{name}: {ch}" for name, s in samples
                            for ch in changed
                            if is_unknown_spec((getattr(
                                s, 'data_transforms', None) or {}).get(ch))})

        def _work():
            out = {}
            for name, s in samples:
                cols = set(s.data.columns)
                rec = getattr(s, 'data_transforms', None)
                for ch, new_m in changed.items():
                    if ch not in cols or is_unknown_spec((rec or {}).get(ch)):
                        continue
                    # The sample's own record says what its values carry,
                    # parameters included (an asinh cofactor-5 channel undone
                    # with the default 150 is wrong); the editor's map stands
                    # in for a sample without one.
                    vals = np.asarray(s.data[ch].values, dtype=float)
                    if rec is not None:
                        lin = to_linear(vals, rec.get(ch))
                    else:
                        lin = inverse_transform_values(
                            vals,
                            method=self._channel_transform.get(ch, 'linear'))
                    out[(name, ch)] = transform_values(lin, method=new_m)
            return out

        def _done(out):
            # Events 'log' leaves without a value (NaN, from a value <= 0),
            # counted going onto log and coming off it: inverting NaN gives
            # NaN, so the new method cannot bring them back either.
            to_log, from_log = {}, {}
            for (name, ch), arr in out.items():
                s = self._samples.get(name)
                if s is not None and ch in s.data.columns:
                    n_nan = int(np.isnan(arr).sum())
                    tally = (to_log if changed[ch] == 'log' else from_log
                             if self._channel_transform.get(ch) == 'log'
                             else None)
                    if n_nan and tally is not None:
                        tally[ch] = tally.get(ch, 0) + n_nan
                    s.data[ch] = arr
                    rec = getattr(s, 'data_transforms', None)
                    if rec is not None:
                        new_m = changed[ch]
                        rec = {k: v for k, v in rec.items() if k != ch}
                        if new_m != 'linear':
                            rec[ch] = transform_spec(new_m)
                        s.data_transforms = rec
            self._channel_transform.update(changed)
            # A sample that finished loading while this ran was not in the
            # snapshot; bring it onto the new transforms too.
            snap = {name for name, _s in samples}
            for name, s in list(self._samples.items()):
                if name not in snap:
                    self._conform_sample_transforms(s)
            # Columns were rewritten in place: nothing cached may survive.
            self._data_changed({name for name, _ch in out} | (
                set(self._samples) - snap))
            self._audit('transform', n_channels=len(changed),
                        changes={ch: m for ch, m in changed.items()})
            self.status_var.set(
                f"Re-transformed {len(changed)} channel(s). Gates on those "
                "channels may need re-checking."
                + (f"  Left as stored (scale unknown): {', '.join(as_stored)}."
                   if as_stored else '')
                + _log_note("  No log value (≤ 0, out of every gate): ",
                            to_log, '.')
                + _log_note("  Still without a value since log: ", from_log,
                            " (reload those samples to recover them)."))
            self._schedule_replot(0)

        self.run_async(_work, on_done=_done,
                       busy_msg=f"Re-transforming {len(changed)} channel(s)…")

    def _refresh_channel_choices(self):
        """Rebuild the axis/colour combo value lists from the union of
        columns across all loaded samples (so freshly-added cluster / UMAP /
        flowsom columns become selectable), preserving current selections."""
        cols = list(self._channels)
        seen = set(cols)
        for s in self._samples.values():
            df = getattr(s, 'data', None)
            if df is None:
                continue
            for c in df.columns:
                if c not in seen:
                    seen.add(c)
                    cols.append(c)
        self._channels = cols
        disp = [self._fmt_channel(c) for c in cols]
        self._xy_choices = disp
        self._color_choices = ['By sample', 'By density'] + disp
        self.x_combo['values'] = disp
        self.y_combo['values'] = disp
        self.color_combo['values'] = self._color_choices

    def _begin_busy(self, msg=None):
        """Show the animated 'working' bar in the status bar (+ optional
        message). Call from the Tk thread when a long job starts."""
        if msg:
            self.status_var.set(msg)
        try:
            self._busy_bar.grid()
            self._busy_bar.start(12)
        except Exception:
            pass

    def _busy(self, msg):
        """Thread-safe phase update: marshal a status message onto the Tk
        thread (the animated bar keeps moving meanwhile)."""
        try:
            self.after(0, lambda m=msg: self.status_var.set(m))
        except Exception:
            pass

    def _end_busy(self):
        """Stop + hide the working bar (call from the Tk thread)."""
        try:
            self._busy_bar.stop()
            self._busy_bar.grid_remove()
        except Exception:
            pass

    def run_async(self, work, on_done=None, on_error=None, busy_msg=None):
        """Run ``work()`` off the Tk thread with the busy bar showing, then
        deliver its result to ``on_done`` (or the exception to ``on_error``) on
        the Tk thread and hide the bar. The one editor-wide way to keep a heavy
        op from freezing the UI — see :mod:`openflo.async_task`. ``on_error``
        defaults to a status-bar message."""
        from .async_task import run_async as _run_async
        if busy_msg is not None:
            self._begin_busy(busy_msg)

        def _err(exc):
            if on_error is not None:
                on_error(exc)
            else:
                try:
                    self.status_var.set(f"Error: {exc}")
                except Exception:
                    pass

        return _run_async(self, work, on_done=on_done, on_error=_err,
                          on_finally=(self._end_busy
                                      if busy_msg is not None else None))

    def _finish_clustering(self, method, emb_prefix, targets, outcome=None,
                           embedded=None):
        """Import, audit and report a finished run. ``outcome`` maps each
        sample to its :func:`label_run_outcome`; without it a sample counts as
        clustered when it carries the label column. ``embedded`` lists the
        samples whose embedding this run wrote; without it a clustered sample
        counts as embedded when it carries the embedding column."""
        self._clustering_busy = False
        self._end_busy()
        # The run wrote label/embedding columns into these samples off the Tk
        # thread: their cached display work is stale.
        self._data_changed([n for n in targets if n in self._samples])
        self._refresh_channel_choices()
        col = self._cluster_label_column(method)
        outcome = outcome or {}
        done = [n for n in targets
                if n in self._samples and col in self._samples[n].data.columns
                and outcome.get(n) != 'skipped']
        skipped = [n for n in targets if n in self._samples and n not in done]
        why = ("too few finite events or no usable channels"
               + (" (FlowSOM needs at least grid × grid events)"
                  if method == 'flowsom' else ''))
        if not done:
            # Nothing was clustered: no import, no audit (so nothing reaches
            # the Methods paragraph), and no claim that the run is done.
            self._schedule_replot(0)
            self.status_var.set(
                f"{method} produced no labels on {', '.join(skipped)}: "
                f"{why}. Nothing was imported.")
            return
        changed = [n for n in done if outcome.get(n) == 'changed']
        reset = self._reset_population_names(col, changed)
        # One undo step for the prune and the import (the import's own
        # checkpoint came after the prune, so Ctrl+Z never brought pruned
        # populations back). It is taken AFTER the name reset: undo restores
        # gates, not the label column, so restoring the names would put the
        # old partition's names on the new partition's cells.
        self._checkpoint()
        pruned, kept = self._prune_stale_populations(col, changed)
        self._import_populations(col)
        # Switch to the embedding axes, and audit the embedding, only where
        # this run wrote it (an uninstalled optional backend silently writes
        # nothing, and an earlier run's columns may still be there).
        if embedded is None:
            embedded = [n for n in done if emb_prefix and
                        f'{emb_prefix}1' in self._samples[n].data.columns]
        embedded = [n for n in done if n in embedded]
        if emb_prefix and embedded:
            self.mode_var.set('dot')
            # Embedding coordinates are abstract → force a LINEAR axis scale
            # (the global default is log, tuned for fluorescence intensity).
            self._channel_scale[f'{emb_prefix}1'] = 'linear'
            self._channel_scale[f'{emb_prefix}2'] = 'linear'
            self.x_combo.set(self._fmt_channel(f'{emb_prefix}1'))
            self.y_combo.set(self._fmt_channel(f'{emb_prefix}2'))
            self.color_combo.set(self._fmt_channel(col))
        self._schedule_replot(0)
        self._audit('cluster', method=method, column=col,
                    n_samples=len(done), samples=list(done),
                    embedding=(emb_prefix if emb_prefix and embedded
                               else 'none'))
        msg = (f"{method} done on {len(done)} sample(s) — "
               f"populations imported from '{col}'. Toggle them in the tree.")
        if skipped:
            msg += f" No labels on {', '.join(skipped)}: {why}."
        if reset:
            msg += (f" The {col} ids changed on {', '.join(reset)}, so the "
                    "previous run's population names were reset.")
        if pruned:
            msg += (f" Removed {pruned} {col} population(s) whose id no "
                    "longer occurs.")
        if kept:
            msg += (f" Kept {len(kept)} empty {col} population(s) that other "
                    "gates use: "
                    + '; '.join(f"'{gname}' in {n}, {users}"
                                for n, gname, users in kept) + ".")
        self.status_var.set(msg)

    def _clustering_error(self, exc):
        self._clustering_busy = False
        self._end_busy()
        # A failed run may have written some columns before it stopped.
        self._data_changed()
        self.status_var.set(f"Clustering failed: {exc}")

    def _start_embedding(self, name, df, chans, methods, cap):
        """Background-run the chosen embeddings and show the result grid. The
        array extraction happens in the worker, not on dialog open."""
        if getattr(self, '_dr_running', False):
            return
        self._dr_running = True

        def _work():
            from .dr_compare import run_embeddings
            X = df[chans].to_numpy(dtype=float)
            color = (df['cluster'].to_numpy()
                     if 'cluster' in df.columns else None)
            res = run_embeddings(X, methods=tuple(methods), seed=0,
                                 max_points=cap)
            res['_color'] = color
            return res

        def _done(out):
            self._dr_running = False
            coords = out.get('coords', {})
            idx = out.get('index')
            color = out.get('_color')
            if not coords:
                self.status_var.set("Embedding comparison produced no result.")
                return
            col = color[idx] if (color is not None and idx is not None) else None
            from matplotlib.figure import Figure
            ncol = len(coords)
            fig = Figure(figsize=(5 * ncol, 5), dpi=100)
            for i, (m, xy) in enumerate(coords.items(), 1):
                ax = fig.add_subplot(1, ncol, i)
                ax.scatter(xy[:, 0], xy[:, 1], s=3, c=col,
                           cmap='tab10' if col is not None else None,
                           alpha=0.6, linewidths=0)
                ax.set_title(m)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"Embedding comparison — {name}")
            fig.tight_layout()
            from .ui_figure_window import _FigureWindow
            _FigureWindow(self, fig, f"Embedding comparison — {name}")
            skipped = ', '.join(m for m, _ in out.get('skipped', []))
            self.status_var.set(
                f"Embedding comparison: {', '.join(coords)}"
                + (f"  (skipped: {skipped})" if skipped else "") + ".")

        def _err(exc):
            self._dr_running = False
            self.status_var.set(
                f"Embedding comparison failed: {type(exc).__name__}: {exc}")

        self.run_async(
            _work, on_done=_done, on_error=_err,
            busy_msg=f"Embedding {name} — {', '.join(methods)}… (background)")
