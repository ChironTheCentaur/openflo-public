"""Exports + gate templates — editor mixin.

FlowJo .wsp export, HTML analysis report, plot-image save, and the gate
template library (bundled + user templates). See editor_base.EditorMixin.
"""
from __future__ import annotations

import json
import os
from tkinter import filedialog, messagebox

from .editor_base import EditorMixin
from .theme import savefig_background

# Package directory (same value as gui.BASE) — for the bundled template library.
BASE = os.path.dirname(os.path.abspath(__file__))


class ExportMixin(EditorMixin):
    """FlowJo .wsp / HTML report / plot-image export and gate templates."""

    @staticmethod
    def _templates_dir():
        d = os.path.join(BASE, 'templates')
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _library_dir():
        """The shipped, read-only curated template library (package data)."""
        return os.path.join(BASE, 'template_library')

    @classmethod
    def _bundled_templates(cls):
        """``[(name, description, path)]`` for every ``*.json`` in the shipped
        library AND the user's saved-template dir, sorted with the
        panel-agnostic ``cleanup_*`` recipes first. ``name``/``description``
        come from each template's metadata (filename stem as a fallback);
        duplicate basenames in the user dir shadow the shipped copy."""
        import glob
        import json as _json
        seen, out = set(), []
        for d in (cls._templates_dir(), cls._library_dir()):
            for p in sorted(glob.glob(os.path.join(d, '*.json'))):
                base = os.path.basename(p)
                if base in seen:
                    continue
                seen.add(base)
                name = os.path.splitext(base)[0]
                desc = ''
                try:
                    with open(p, encoding='utf-8') as f:
                        data = _json.load(f)
                    if isinstance(data, dict):
                        name = str(data.get('name') or name)
                        desc = str(data.get('description') or '')
                except Exception:
                    pass
                out.append((name, desc, p))
        # cleanup recipes first (the everyday, panel-agnostic ones)
        out.sort(key=lambda t: (0 if 'cleanup' in os.path.basename(t[2]).lower()
                                 else 1, t[0].lower()))
        return out

    def _fill_template_menu(self, menu):
        """(Re)build the Templates ▾ menu on open: one entry per bundled
        template (friendly name, description as a tooltip-ish accelerator),
        then 'From file…'. Rebuilt each time so a newly-saved template shows."""
        menu.delete(0, 'end')
        bundled = self._bundled_templates()
        if bundled:
            # List templates directly (a disabled "Apply a template:" header
            # rendered poorly/dim, especially in dark mode — the submenu name
            # already says it).
            for name, _desc, path in bundled:
                menu.add_command(
                    label=name,
                    command=lambda p=path: self._apply_template_path(p))
            menu.add_separator()
        menu.add_command(label="From file… (.json / FlowJo .wsp)",
                         command=self._load_template)

    def _load_template(self):
        """Load gates from a .json (native OpenFlo template) or .wsp
        (FlowJo workspace — polygon, rect & threshold gates supported)."""
        path = filedialog.askopenfilename(
            initialdir=self._templates_dir(),
            title="Load gating template",
            filetypes=[('Gating templates', '*.json *.wsp'),
                       ('JSON template',     '*.json'),
                       ('FlowJo workspace',  '*.wsp'),
                       ('All files',         '*.*')])
        if path:
            self._apply_template_path(path)

    def _save_template(self):
        """Write the current gates + channel labels to a v2 JSON template.

        On write-path failures we use messagebox.showerror — the status
        bar alone is too easy to miss after a Save action and silent
        data loss is worse than the alert pop-up.
        """
        if not self._gates:
            self.status_var.set("No gates to save — set at least one first.")
            return
        init = self._templates_dir()
        path = filedialog.asksaveasfilename(
            initialdir=init,
            title="Save gating template",
            defaultextension='.json',
            initialfile='my_template.json',
            filetypes=[('JSON template', '*.json')])
        if not path:
            return
        try:
            from datetime import datetime
            # Embed each gate's editor id so the load path can rewire
            # parent_id references after fresh gate_ids are assigned.
            gate_list = []
            for gid in self._ordered_gate_ids():
                g = dict(self._gates[gid])
                g['id'] = gid
                # Stamp the antibody label for each channel field so the
                # template can retarget by label when applied to a sample
                # whose marker sits on a different detector (label-first
                # tying). Only added when a label differs from the
                # detector name.
                for chan_field, label_field in (('channel', 'label'),
                                                 ('x_channel', 'x_label'),
                                                 ('y_channel', 'y_label')):
                    det = g.get(chan_field)
                    if det:
                        lbl = self._channel_labels.get(det, det)
                        if lbl and lbl != det:
                            g[label_field] = lbl
                gate_list.append(g)
            chans = set()
            for g in gate_list:
                if 'channel' in g:
                    chans.add(g['channel'])
                for k in ('x_channel', 'y_channel'):
                    if k in g:
                        chans.add(g[k])
            template = {
                'name':        os.path.splitext(os.path.basename(path))[0],
                'description': '',
                'version':     2,
                'created':     datetime.now().isoformat(timespec='seconds'),
                'gates':       gate_list,
                'labels':      {ch: self._channel_labels.get(ch, ch)
                                for ch in chans},
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            self.status_var.set(
                f"Saved {len(gate_list)} gate(s) → "
                f"{os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Save failed: {exc}")
            messagebox.showerror(
                "Save template failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}",
                parent=self)

    def _save_plot_image(self):
        """Save exactly what's on the main plot to PNG / SVG / PDF, with a
        publication-friendly white background (even under the dark themes)."""
        if not self._samples:
            self.status_var.set("Nothing to save — load a sample and plot first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save plot as image",
            defaultextension='.png',
            initialfile='openflo_plot.png',
            filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'),
                       ('PDF', '*.pdf'), ('All files', '*.*')])
        if not path:
            return
        try:
            savefig_background(self.fig, path, background='White', dpi=300)
            self.status_var.set(f"Saved plot → {os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Save plot failed: {exc}")
            messagebox.showerror("Save plot failed",
                                 f"{type(exc).__name__}: {exc}", parent=self)

    def _export_report(self):
        """Bundle the current analysis into one self-contained HTML report."""
        if not self._samples:
            self.status_var.set("Load samples first to build a report.")
            return
        from datetime import datetime

        import pandas as pd

        from . import __version__
        from .report import build_html_report, df_to_html_table, figure_html

        path = filedialog.asksaveasfilename(
            parent=self, title="Save analysis report",
            defaultextension='.html', initialfile='openflo_report.html',
            filetypes=[('HTML', '*.html'), ('All files', '*.*')])
        if not path:
            return

        meta = {
            'OpenFlo version': __version__,
            'Generated': datetime.now().isoformat(timespec='seconds'),
            'Samples': len(self._samples),
            'Channels': len(self._channels),
            'Active sample': self._active_sample or '—',
        }
        # Snapshot the LIVE plot to HTML on the Tk thread — self.fig is owned by
        # the canvas, so it must not be rendered from the worker thread. Building
        # the (heavy) stats + heatmap sections and writing the file then happen
        # off-thread so the UI doesn't freeze for the seconds-to-tens-of-seconds
        # a big report takes.
        try:
            plot_html = figure_html(self.fig, alt='current plot')
        except Exception as exc:
            print(f"[report] plot embed: {exc}", flush=True)
            plot_html = ('<p class="note">(plot could not be embedded)</p>')

        def _build():
            sections = []
            srows = []
            for n in self._sample_order:
                s = self._samples.get(n)
                if s is None:
                    continue
                srows.append({
                    'Sample': n,
                    'Trial': self._sample_trial.get(n, ''),
                    'Events': len(s.data),
                    'Gates': len(self._sample_gates.get(n, {})),
                    'Plotted': 'yes' if self._sample_plot_enabled.get(n) else ''})
            sections.append({'heading': 'Samples & gates',
                             'html': df_to_html_table(pd.DataFrame(srows))})
            sections.append({'heading': 'Current plot', 'html': plot_html})
            try:
                rows, cols = self._collect_stats_rows(
                    {'Count', '%Parent', '%Total'})
                if rows:
                    disp = [c for c in cols if not c.startswith('__')]
                    df = pd.DataFrame(
                        [{c: r.get(c) for c in disp} for r in rows])
                    sections.append({'heading': 'Population statistics',
                                     'html': df_to_html_table(df, max_rows=500)})
            except Exception as exc:
                print(f"[report] stats: {exc}", flush=True)
            try:
                hm = self._report_heatmap_html()
                if hm:
                    sections.append({'heading': 'Cluster heatmap', 'html': hm})
            except Exception as exc:
                print(f"[report] heatmap: {exc}", flush=True)
            try:
                from .audit import _short
                entries = self._audit_log.entries()
                if entries:
                    arows = [{'#': e['seq'], 'Time': e.get('time') or '',
                              'Action': e['action'],
                              'Details': ', '.join(
                                  f"{k}={_short(v)}"
                                  for k, v in e['details'].items())}
                             for e in entries]
                    sections.append({'heading': 'Provenance (audit trail)',
                                     'html': df_to_html_table(
                                         pd.DataFrame(arows))})
            except Exception as exc:
                print(f"[report] audit: {exc}", flush=True)
            doc = build_html_report('OpenFlo analysis report', meta=meta,
                                    sections=sections)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(doc)
            return len(sections)

        def _done(n_sections):
            self._audit('report.export', path=path, sections=n_sections)
            self.status_var.set(f"Report → {os.path.basename(path)}")
            try:
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(path))
            except Exception:
                pass

        def _err(exc):
            messagebox.showerror("Report", f"Could not write report:\n{exc}",
                                 parent=self)

        self.run_async(_build, on_done=_done, on_error=_err,
                       busy_msg="Building report…")

    def _export_flowjo_wsp(self):
        """Build a FlowJo-compatible .wsp from every loaded sample's gate
        tree. Each FlowSample becomes a SampleNode whose Subpopulations
        carry that sample's gate hierarchy. The shared `WspWriter` is the
        same one the pipeline export uses.

        Before writing, warn about any OpenFlo-only state that the .wsp
        format can't represent (offer to save a full session instead)."""
        if not self._samples:
            self.status_var.set(
                "No samples loaded — load FCS files first, then export.")
            return

        lossy = self._wsp_lossy_summary()
        if lossy:
            bullets = '\n'.join(f"  • {item}" for item in lossy)
            # Yes = export anyway, No = save a .flowsession instead,
            # Cancel = abort.
            choice = messagebox.askyesnocancel(
                "Some state won't fit in a FlowJo .wsp",
                "A FlowJo workspace can't store the following — it will be "
                "lost on export (gates + compensation are preserved):\n\n"
                f"{bullets}\n\n"
                "Export to .wsp anyway?\n\n"
                "Yes = export (lose the above)\n"
                "No = save a full .flowsession instead\n"
                "Cancel = don't export",
                parent=self)
            if choice is None:           # Cancel
                return
            if choice is False:          # No → save session instead
                self._save_session()
                return
            # Yes → fall through to the .wsp export.

        path = filedialog.asksaveasfilename(
            title="Export workspace to FlowJo .wsp",
            defaultextension='.wsp',
            initialfile='openflo_export.wsp',
            filetypes=[('FlowJo workspace', '*.wsp')])
        if not path:
            return

        def _build():
            from .pipeline import WspWriter
            w = WspWriter(cytometer='OpenFlo')
            total = 0
            # Compensation: WspWriter stores one workspace-wide matrix.
            # In normal use every sample in a single trial shares the
            # same spillover (auto_compensate pulls it from $SPILL which
            # is panel-specific, not sample-specific). Pick the first
            # loaded sample that has a matrix and register it.
            comp_set = False
            for name in self._sample_order:
                if name not in self._samples:
                    continue
                sample = self._samples[name]
                gates  = self._sample_gates.get(name, {})
                if (not comp_set
                        and getattr(sample, 'comp_matrix', None) is not None
                        and getattr(sample, 'comp_channels', None)):
                    w.set_compensation(
                        sample.comp_channels, sample.comp_matrix)
                    comp_set = True
                # Build the gate list with ids/parent_ids that the writer
                # expects (the in-memory store already keys by id). Gates with
                # no FlowJo geometry are dropped; any surviving child of a
                # dropped gate (e.g. real gates built UNDER an auto-clean group)
                # is re-rooted onto its nearest exportable ancestor so it isn't
                # orphaned.
                skipped = {gid for gid, gg in gates.items()
                           if gg.get('kind') in ('cluster', 'category',
                                                 'boolean', 'autoclean',
                                                 'group')}
                gate_list = []
                for gid in self._sample_gate_order.get(name, []):
                    if gid in skipped:
                        continue          # no FlowJo geometry — see lossy note
                    g = dict(gates[gid])
                    pid, seen = g.get('parent_id'), set()
                    while pid in skipped and pid not in seen:
                        seen.add(pid)
                        pid = gates.get(pid, {}).get('parent_id')
                    g['parent_id'] = pid
                    g['id'] = gid
                    gate_list.append(g)
                data = getattr(sample, 'data', None)
                channels = list(data.columns) if data is not None else []
                w.add_sample(
                    name=name,
                    fcs_path=getattr(sample, 'path', '') or '',
                    channels=channels,
                    gates=gate_list)
                total += len(gate_list)
            w.write(path)
            return total, comp_set

        def _done(res):
            total, comp_set = res
            comp_note = ' + spillover' if comp_set else ''
            self.status_var.set(
                f"Exported {len(self._samples)} sample(s) / {total} gate(s)"
                f"{comp_note} → {os.path.basename(path)}")

        def _err(exc):
            self.status_var.set(f"Export failed: {exc}")
            # Status-bar message alone is too easy to miss after a Save
            # dialog — surface the failure visibly.
            messagebox.showerror(
                "Export to FlowJo .wsp failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}",
                parent=self)

        self.run_async(_build, on_done=_done, on_error=_err,
                       busy_msg="Exporting FlowJo .wsp…")
