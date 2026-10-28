"""Exports + gate templates — editor mixin.

FlowJo .wsp export, HTML analysis report, plot-image save, and the gate
template library (bundled + user templates). See editor_base.EditorMixin.
"""
from __future__ import annotations

import json
import logging
import os
import re
from tkinter import filedialog, messagebox

from .editor_base import EditorMixin
from .export_stamp import corrections_for, export_metadata
from .theme import savefig_background
from .ui_errors import show_error

log = logging.getLogger(__name__)

# Package directory (same value as gui.BASE) — for the bundled template library.
BASE = os.path.dirname(os.path.abspath(__file__))


class ExportMixin(EditorMixin):
    """FlowJo .wsp / HTML report / plot-image export and gate templates."""

    @staticmethod
    def _templates_dir():
        """Where saved templates go: ``~/.openflo/templates``. It used to be a
        ``templates`` folder inside the package, which a pip install puts in
        site-packages: read-only on a system Python, and wiped by every
        upgrade or reinstall."""
        d = os.path.join(os.path.expanduser('~'), '.openflo', 'templates')
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
        return d

    @staticmethod
    def _legacy_templates_dir():
        """The old in-package folder, still read (never created) so templates
        saved by an older OpenFlo stay in the menu."""
        return os.path.join(BASE, 'templates')

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
        for d in (cls._templates_dir(), cls._legacy_templates_dir(),
                  cls._library_dir()):
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
                except (OSError, ValueError):
                    pass
                out.append((name, desc, p))
        # cleanup recipes first (the everyday, panel-agnostic ones)
        out.sort(key=lambda t: (0 if 'cleanup' in os.path.basename(t[2]).lower()
                                 else 1, t[0].lower()))
        return out

    def _fill_template_menu(self, menu):
        """(Re)build the Templates ▾ menu on open: one entry per bundled
        template (labelled by friendly name), then 'From file…'. Rebuilt each
        time so a newly-saved template shows."""
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
        menu.add_command(label="From file… (.json / FlowJo .wsp / Gating-ML)",
                         command=self._load_template)

    def _load_template(self):
        """Load gates from a .json (native OpenFlo template) or .wsp
        (FlowJo workspace — polygon, rect & threshold gates supported)."""
        path = filedialog.askopenfilename(
            initialdir=self._templates_dir(),
            title="Load gating template",
            filetypes=[('Gating templates', '*.json *.wsp *.xml'),
                       ('JSON template',     '*.json'),
                       ('FlowJo workspace',  '*.wsp'),
                       ('Gating-ML 2.0',     '*.xml'),
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
            show_error(self, "Save template failed",
                       "The gate template could not be saved.", exc, path)

    def _save_plot_image(self):
        """Save exactly what's on the main plot to PNG / SVG / PDF, with a
        publication-friendly white background (even under the dark themes)."""
        if not self._samples:
            self.status_var.set("Nothing to save — load a sample and plot first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export plot image",
            defaultextension='.png',
            initialfile='openflo_plot.png',
            filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'),
                       ('PDF', '*.pdf'), ('All files', '*.*')])
        if not path:
            return
        try:
            savefig_background(self.fig, path, background='White',
                               corrections=corrections_for(self))
        except Exception as exc:
            self.status_var.set(f"Save plot failed: {exc}")
            show_error(self, "Save plot failed",
                       "The plot image could not be saved.", exc, path)
            return
        self._audit('plot.export', path=path,
                    samples=[n for n in self._sample_order
                             if n in self._samples
                             and self._sample_plot_enabled.get(n, True)])
        self.status_var.set(f"Saved plot → {os.path.basename(path)}")

    def _copy_plot_image(self):
        """Copy the main plot to the clipboard as an image, rendered as
        Save plot as image would save it."""
        if not self._samples:
            self.status_var.set("Nothing to copy — load a sample and plot "
                                "first.")
            return
        from .clipboard_export import copy_figure
        ok, msg = copy_figure(self.fig)
        self.status_var.set(msg)
        if not ok:
            messagebox.showerror("Copy plot failed", msg, parent=self)

    def _export_report(self):
        """Bundle the current analysis into one self-contained HTML report."""
        if not self._samples:
            self.status_var.set("Load samples first to build a report.")
            return
        from datetime import datetime

        import pandas as pd

        from .report import build_html_report, df_to_html_table, figure_html

        path = filedialog.asksaveasfilename(
            parent=self, title="Save analysis report",
            defaultextension='.html', initialfile='openflo_report.html',
            filetypes=[('HTML', '*.html'), ('All files', '*.*')])
        if not path:
            return

        fixes = corrections_for(self)
        stamp = export_metadata(fixes)
        meta = {
            'OpenFlo version': stamp['OpenFlo version'],
            **({'Corrections': stamp['Corrections']}
               if 'Corrections' in stamp else {}),
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
                    # A population that cannot be computed reads NA.
                    df = pd.DataFrame(
                        [{c: ('NA' if r.get('__unknown__') and c not in
                              ('Sample', 'Population') else r.get(c))
                          for c in disp} for r in rows])
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
                                    sections=sections, corrections=fixes)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(doc)
            return len(sections)

        def _done(n_sections):
            self._audit('report.export', path=path, sections=n_sections,
                        samples=[n for n in self._sample_order
                                 if n in self._samples])
            self.status_var.set(f"Report → {os.path.basename(path)}")
            try:
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(path))
            except Exception:  # noqa: BLE001 - opening the browser is a nicety
                pass

        def _err(exc):
            show_error(self, "Report", "The report could not be written.", exc)

        self.run_async(_build, on_done=_done, on_error=_err,
                       busy_msg="Building report…")

    def _match_unknown_scales(self, sample, stored, gates, comp_channels,
                              comp_matrix):
        """``{channel: f}`` (see pipeline.rank_matched_fn) for each channel of
        unknown scale that one of `gates` is on, matched to the same channel
        of the sample's raw FCS as FlowJo will gate it: compensated with the
        matrix written for the sample, after QC when the sample had it. A
        channel without such an FCS, or one that cannot be matched, is left
        out, and the writer exports its gates as stored (with a warning)."""
        from .gating import gate_channels
        from .pipeline import (
            fcs_values_as_flowjo_gates_them,
            is_unknown_spec,
            rank_matched_fn,
        )
        unknown = {ch for ch, sp in (stored or {}).items()
                   if is_unknown_spec(sp)}
        used = sorted({ch for g in gates if not g.get('scale_unknown')
                       for ch in gate_channels(g)} & unknown)
        fcs = str(getattr(sample, 'source_fcs', '')
                  or getattr(sample, 'path', '') or '')
        data = getattr(sample, 'data', None)
        if (not used or data is None or not fcs.lower().endswith('.fcs')
                or not os.path.isfile(fcs)):
            return {}
        try:
            raw = fcs_values_as_flowjo_gates_them(
                fcs, qc=bool(getattr(sample, 'qc_applied', False)),
                compensation=((comp_channels, comp_matrix)
                              if comp_matrix is not None and comp_channels
                              else None))
        except Exception as exc:                  # unreadable: export as is
            log.warning("[wsp export] %s: could not read %s to match its "
                        "unknown-scale channels (%s)", sample.name,
                        os.path.basename(fcs), exc)
            return {}
        out = {}
        for ch in used:
            if ch in data.columns and ch in raw.columns:
                f = rank_matched_fn(data[ch].to_numpy(),
                                    raw[ch].to_numpy())
                if f is not None:
                    out[ch] = f
        return out

    def _export_gate_list(self):
        """Every sample's gates as one CSV: population path, kind, channels,
        and coordinates as stored and in linear units (see gate_table)."""
        from .gate_table import gate_rows, write_csv
        names = [n for n in self._sample_order
                 if self._sample_gates.get(n)]
        if not names:
            self.status_var.set(
                "No gates to list — draw gates on a sample first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export gate list", defaultextension='.csv',
            initialfile='gate_list.csv', filetypes=[('CSV', '*.csv')])
        if not path:
            return
        rows = []
        for n in names:
            rows += gate_rows(n, self._sample_gates[n],
                              self._sample_gate_order.get(n),
                              transforms=self._display_transforms(n))
        try:
            write_csv(path, rows)
        except OSError as exc:
            messagebox.showerror("Export gate list",
                                 f"Could not write the file:\n{exc}",
                                 parent=self)
            return
        self._audit('export.gate_list', path=path, n_samples=len(names),
                    n_gates=len(rows), samples=names)
        self.status_var.set(f"Exported {len(rows)} gate(s) of {len(names)} "
                            f"sample(s) → {os.path.basename(path)}")

    def _export_gatingml(self):
        """Write the active sample's gate tree as a standalone Gating-ML 2.0
        file, the ISAC standard other analysis tools read. Coordinates stay
        on each channel's own scale (see gatingml.write_gatingml), with the
        sample's compensation matrix; a gate Gating-ML cannot hold is left
        out and named."""
        name = self._active_sample
        sample = self._samples.get(name) if name else None
        gates = self._sample_gates.get(name, {}) if name else {}
        if sample is None or not gates:
            self.status_var.set(
                "No gates to export — draw gates on a sample first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export gates as Gating-ML 2.0",
            defaultextension='.xml',
            initialfile=f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}_gates.xml",
            filetypes=[('Gating-ML 2.0', '*.xml')])
        if not path:
            return
        from .gatingml import write_gatingml
        gate_list = []
        for gid in self._sample_gate_order.get(name, []):
            g = dict(gates[gid])
            g['id'] = gid
            gate_list.append(g)
        own_m = getattr(sample, 'comp_matrix', None)
        own_c = getattr(sample, 'comp_channels', None)
        comp = (own_c, own_m) if own_m is not None and own_c else None
        stored = self._stored_transforms(name, sample)
        try:
            warns = write_gatingml(
                path, gate_list, stored, compensation=comp,
                unknown_fns=self._match_unknown_scales(
                    sample, stored, gate_list, own_c, own_m),
                corrections=corrections_for(self))
        except Exception as exc:
            self.status_var.set(f"Gating-ML export failed: {exc}")
            show_error(self, "Gating-ML export failed",
                       "The Gating-ML file could not be written.", exc, path)
            return
        self._audit('gatingml.export', path=path, n_gates=len(gate_list),
                    samples=[name])
        self.status_var.set(
            f"Exported {len(gate_list)} gate(s) of {name} → "
            f"{os.path.basename(path)}"
            + (f" — {len(warns)} note(s)" if warns else ''))
        if warns:
            shown = warns[:12]
            more = len(warns) - len(shown)
            messagebox.showwarning(
                "Gating-ML export notes",
                '\n'.join(f"  • {w}" for w in shown)
                + (f"\n  … and {more} more (see log)" if more else ''),
                parent=self)

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
            w.corrections = corrections_for(self)
            total = 0
            # Compensation: each sample carries the matrix OpenFlo applied to
            # it (FlowJo applies a per-sample matrix as given), and WspWriter
            # lists each distinct one in <Matrices>. Nothing is registered
            # workspace-wide: whether FlowJo would apply such a matrix to a
            # sample that has none of its own is not known.
            comp_set = False
            for name in self._sample_order:
                if name not in self._samples:
                    continue
                sample = self._samples[name]
                gates  = self._sample_gates.get(name, {})
                if (getattr(sample, 'comp_matrix', None) is not None
                        and getattr(sample, 'comp_channels', None)):
                    comp_set = True
                # Build the gate list with ids/parent_ids that the writer
                # expects (the in-memory store already keys by id). Gates with
                # no FlowJo geometry are dropped; any surviving child of a
                # dropped gate (e.g. real gates built UNDER an auto-clean group)
                # is re-rooted onto its nearest exportable ancestor so it isn't
                # orphaned.
                # Booleans go out as FlowJo AndNode / OrNode / NotNode
                # unless FlowJo cannot express them (wsp_unexportable).
                from .pipeline import wsp_unexportable
                skipped = wsp_unexportable(gates)
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
                own_m = getattr(sample, 'comp_matrix', None)
                own_c = getattr(sample, 'comp_channels', None)
                # Gates are kept in the sample's stored (e.g. logicle)
                # coordinates; FlowJo wants linear scale units.
                stored = self._stored_transforms(name, sample)
                unknown_fns = self._match_unknown_scales(
                    sample, stored, gate_list, own_c, own_m)
                w.add_sample(
                    name=name,
                    # The raw FCS, also for a sample restored from a
                    # processed CSV (FlowJo cannot open the CSV).
                    fcs_path=(getattr(sample, 'source_fcs', '')
                              or getattr(sample, 'path', '') or ''),
                    channels=channels,
                    gates=gate_list,
                    compensation=((own_c, own_m) if own_m is not None and own_c
                                  else None),
                    transforms=stored, unknown_fns=unknown_fns,
                    group=self._trial_for(name))
                total += len(gate_list)
            w.write(path)
            return total, comp_set, len(w.warnings), list(w.scale_notes)

        def _done(res):
            total, comp_set, n_warn, scale_notes = res
            self._audit('wsp.export', path=path, n_gates=total,
                        samples=[n for n in self._sample_order
                                 if n in self._samples])
            comp_note = ' + spillover' if comp_set else ''
            # Each warning (an ellipse sent as a polygon, a lower edge FlowJo
            # treats as open) is logged by WspWriter; the status bar counts them.
            warn_note = (f" — {n_warn} place(s) FlowJo will count differently "
                         "(see log)" if n_warn else '')
            self.status_var.set(
                f"Exported {len(self._samples)} sample(s) / {total} gate(s)"
                f"{comp_note} → {os.path.basename(path)}{warn_note}")
            if scale_notes:
                shown = scale_notes[:12]
                more = len(scale_notes) - len(shown)
                messagebox.showwarning(
                    "Gates on channels of unknown scale",
                    "These gates are on a channel whose display scale this "
                    "OpenFlo cannot know (an older OpenFlo's CSV). They were "
                    "exported anyway; check them in FlowJo:\n\n"
                    + '\n'.join(f"  • {n}" for n in shown)
                    + (f"\n  … and {more} more (see log)" if more else ''),
                    parent=self)

        def _err(exc):
            self.status_var.set(f"Export failed: {exc}")
            # Status-bar message alone is too easy to miss after a Save
            # dialog — surface the failure visibly.
            show_error(self, "FlowJo workspace export failed",
                       "The FlowJo workspace could not be written.", exc, path)

        self.run_async(_build, on_done=_done, on_error=_err,
                       busy_msg="Exporting FlowJo .wsp…")
