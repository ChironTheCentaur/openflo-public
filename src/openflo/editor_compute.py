"""Background clustering / embedding orchestration and busy indicator.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import numpy as np

from .editor_base import EditorMixin


class ComputeMixin(EditorMixin):
    """Launch and finalise clustering / embedding jobs off the UI thread, with a modal busy indicator and channel-transform application."""

    def _apply_channel_transforms(self, new_methods):
        """Re-transform channels across ALL loaded samples by inverting each
        channel's current transform and applying the new one (so no
        re-compensation is needed). The (pure) transforms run off the Tk thread
        and only the resulting column arrays are written to ``s.data`` on the Tk
        thread in on_done — no freeze, no cross-thread write race. Sets its own
        status (was: returned a count to the caller)."""
        from .pipeline import inverse_transform_values, transform_values
        changed = {c: m for c, m in new_methods.items()
                   if m != self._channel_transform.get(c, 'linear')}
        if not changed:
            self.status_var.set("No transform changes.")
            return
        samples = list(self._samples.items())     # (name, sample) refs

        def _work():
            out = {}
            for name, s in samples:
                cols = set(s.data.columns)
                for ch, new_m in changed.items():
                    if ch not in cols:
                        continue
                    old_m = self._channel_transform.get(ch, 'linear')
                    lin = inverse_transform_values(
                        np.asarray(s.data[ch].values, dtype=float),
                        method=old_m)
                    out[(name, ch)] = transform_values(lin, method=new_m)
            return out

        def _done(out):
            for (name, ch), arr in out.items():
                s = self._samples.get(name)
                if s is not None and ch in s.data.columns:
                    s.data[ch] = arr
            self._channel_transform.update(changed)
            self._audit('transform', n_channels=len(changed),
                        changes={ch: m for ch, m in changed.items()})
            self.status_var.set(
                f"Re-transformed {len(changed)} channel(s). Gates on those "
                "channels may need re-checking.")
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

    def _finish_clustering(self, method, emb_prefix, targets):
        self._clustering_busy = False
        self._end_busy()
        self._refresh_channel_choices()
        col = {'phenograph': 'cluster', 'leiden': 'leiden'}.get(
            method, 'flowsom_meta')
        self._import_populations(col)
        # Switch to the embedding axes only if it actually produced columns
        # (an uninstalled optional backend silently writes nothing).
        if emb_prefix and any(
                f'{emb_prefix}1' in self._samples[n].data.columns
                for n in targets if n in self._samples):
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
                    n_samples=len(targets), samples=list(targets),
                    embedding=emb_prefix or 'none')
        self.status_var.set(
            f"{method} done on {len(targets)} sample(s) — "
            f"populations imported from '{col}'. Toggle them in the tree.")

    def _clustering_error(self, exc):
        self._clustering_busy = False
        self._end_busy()
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
