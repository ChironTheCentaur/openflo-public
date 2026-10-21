"""Re-place on each sample… -- re-fit a gate's placement on every sample's
own data, reviewed before it is accepted.

The fitting is in gate_refit (Tk-free); this gathers each sample's parent
population, builds the proposals and installs the accepted ones as one undo
step.
"""
from __future__ import annotations

from .editor_base import EditorMixin


class RefitMixin(EditorMixin):
    """A gate's right-click > Re-place on each sample…"""

    def _refit_proposals(self, name, gid):
        """One proposal per loaded sample holding the same population as
        gate ``gid`` of ``name``: that sample's gate placed where ``gid`` is
        and re-fitted on its own parent population."""
        from . import gate_refit as gr
        from .gating import population_keys
        ref = self._sample_gates[name][gid]
        key = population_keys(self._sample_gates[name],
                              self._channel_labels)[gid]
        chans = [c for c in (ref.get('channel'), ref.get('x_channel'),
                             ref.get('y_channel')) if c]
        out = []
        for n in [name] + [s for s in self._sample_order if s != name]:
            if n not in self._samples or n not in self._sample_gates:
                continue
            sg = self._sample_gates[n]
            tid = gid if n == name else next(
                (g for g, k in population_keys(
                    sg, self._channel_labels).items() if k == key), None)
            if tid is None:
                continue
            target = sg[tid]
            df = self._get_df(n, chans[0], chans[1] if len(chans) > 1 else None,
                              downsample=False,
                              gate_parent=target.get('parent_id'))
            if any(c not in df.columns for c in chans):
                continue
            data = {c: df[c].to_numpy(dtype=float) for c in chans}
            new, why = gr.refit(gr.with_geometry(target, ref), data)
            out.append({'sample': n, 'gid': tid, 'old': target, 'new': new,
                        'why': why, 'data': data,
                        'before': gr.fraction_in(target, df),
                        'after': gr.fraction_in(new, df)})
        return out

    def _refit_gate_per_sample(self, name=None, gid=None, review=True):
        """Re-fit gate ``gid`` (default: the selected gate) on every sample
        and show the review grid; Accept installs it. ``review=False``
        installs every proposal that found its landmark (tests, scripting).
        """
        from .gate_refit import refittable
        from .ui_refit import RefitReviewWindow
        name = name or self._active_sample
        gid = gid or self._selected_gate_id()
        gate = self._sample_gates.get(name, {}).get(gid)
        if gate is None:
            self.status_var.set("Select a gate first.")
            return None
        if not refittable(gate):
            self.status_var.set(
                "Only threshold and interval gates, and gates from Auto-gate "
                "(singlet or Gaussian mixture), can be re-placed per sample.")
            return None
        props = self._refit_proposals(name, gid)
        if not props:
            self.status_var.set("No sample has this gate's channels.")
            return None
        if not review:
            return self._install_refits([p for p in props if not p['why']])
        from .pipeline import describe_gate
        return RefitReviewWindow(
            self, props, self._install_refits,
            f"Re-place on each sample — "
            f"{describe_gate(gate, transforms=self._display_transforms(name))}")

    def _install_refits(self, proposals):
        """Put each accepted proposal's placement on its sample's gate (its
        name, colour and children stay), stamped as re-fitted by this version
        on its parent population's channels. One undo step."""
        from .result_stamp import stamp_gate
        if not proposals:
            self.status_var.set("Nothing re-placed.")
            return 0
        self._checkpoint()
        for p in proposals:
            old = self._sample_gates[p['sample']][p['gid']]
            old.clear()
            old.update(p['new'])
            stamp_gate(old, 'gate.refit', [old.get('parent_id')]
                       + [old.get(k) for k in ('channel', 'x_channel',
                                               'y_channel')])
        self._audit('gate.refit', samples=[p['sample'] for p in proposals],
                    kind=proposals[0]['new'].get('kind'))
        self._refresh_gate_list()
        self._schedule_replot(0)
        self.status_var.set(f"Gate re-placed on {len(proposals)} sample(s).")
        return len(proposals)
