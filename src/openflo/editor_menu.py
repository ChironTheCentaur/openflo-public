"""Window menubar construction + theming — editor mixin.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import tkinter as tk

from .editor_base import EditorMixin
from .prefs import write_pref
from .theme import current_palette


class MenuMixin(EditorMixin):
    """Builds the File/Edit/View/Analyze/Tools/Help menubar and re-themes it."""

    def _theme_menubar(self, pal=None):
        """Colour the tk.Menubutton menubar from the palette (tk widgets don't
        follow ttk styles)."""
        pal = pal or current_palette()
        for btn in getattr(self, '_menubar_buttons', []):
            try:
                btn.configure(bg=pal['bg'], fg=pal['fg'],
                              activebackground=pal['active'],
                              activeforeground=pal['fg'])
            except Exception:
                pass

    def _build_menubar(self):
        """Window menubar: File / Edit / View / Analyze / Tools / Help. Every
        item calls an existing handler; built once at the end of __init__ so it
        can bind the Display/log vars. Best-effort — a failure leaves the
        toolbar + side panel fully functional."""
        try:
            # tk.Menubutton requires its dropdown menu to be a CHILD of the
            # button (ttk.Menubutton is lenient; tk.Menubutton is not), so the
            # buttons are created first and each menu is parented to its button.
            labels = ("File", "Edit", "View", "Analyze", "Tools", "Help")
            _menu_tips = {
                "File": "Add FCS / Load CSV, open & save sessions, export a "
                        "FlowJo .wsp or an HTML report.",
                "Edit": "Undo / redo, clear or copy gates, boolean & auto-clean "
                        "gates, import populations.",
                "View": "Display mode, log/console, hover tips, light/dark "
                        "theme, and dock-all-panels.",
                "Analyze": "Statistics, frequencies, expression, sample QC; "
                           "clustering, cell cycle, trajectory, annotation, "
                           "SOM tree.",
                "Tools": "Compensation, transforms, calibration; batch-norm, "
                         "spectral unmix, figure layout; templates; history.",
                "Help": "Check for updates and About OpenFlo.",
            }
            self._menubar_buttons = []
            btns = {}
            for _label in labels:
                _btn = tk.Menubutton(self._menubar_frame, text=_label,
                                     relief='flat', bd=0, padx=9, pady=2)
                _btn.pack(side='left')
                self._menubar_buttons.append(_btn)
                btns[_label] = _btn
                self._tip(_btn, _menu_tips.get(_label, ''))

            filem = tk.Menu(btns["File"], tearoff=0)
            filem.add_command(label="Add FCS…", command=self._add_samples,
                              accelerator="Ctrl+Shift+A")
            filem.add_command(label="Load CSV…",
                              command=self._load_processed_data)
            filem.add_command(label="Load example dataset",
                              command=self._load_example_data)
            filem.add_command(label="Generate dataset…",
                              command=self._open_synthetic_dialog)
            filem.add_command(label="Quick preview…",
                              command=self._open_quick_preview)
            filem.add_separator()
            filem.add_command(label="Open session…", command=self._load_session,
                              accelerator="Ctrl+O")
            recentm = tk.Menu(filem, tearoff=0)
            recentm.configure(postcommand=lambda m=recentm:
                              self._fill_recent_menu(m))
            self._fill_recent_menu(recentm)
            filem.add_cascade(label="Open Recent", menu=recentm)
            filem.add_command(label="Save session…", command=self._save_session,
                              accelerator="Ctrl+S")
            filem.add_command(label="Upgrade saved session…",
                              command=self._upgrade_session_file)
            filem.add_separator()
            filem.add_command(label="Export → FlowJo .wsp…",
                              command=self._export_flowjo_wsp,
                              accelerator="Ctrl+E")
            filem.add_command(label="Save plot as image…",
                              command=self._save_plot_image,
                              accelerator="Ctrl+Shift+S")
            filem.add_command(label="Analysis report (HTML)…",
                              command=self._export_report)
            filem.add_separator()
            filem.add_command(label="Close", command=self._on_close,
                              accelerator="Ctrl+W")
            btns["File"]['menu'] = filem

            editm = tk.Menu(btns["Edit"], tearoff=0)
            editm.add_command(label="Undo", command=self._undo)
            editm.add_command(label="Redo", command=self._redo)
            editm.add_separator()
            editm.add_command(label="Clear gate",
                              command=self._clear_selected_gate)
            editm.add_command(label="Clear all gates", command=self._clear_all)
            editm.add_command(label="Copy gates to…",
                              command=self._open_copy_gates_dialog)
            popm = tk.Menu(editm, tearoff=0)
            popm.configure(postcommand=lambda m=popm:
                           self._fill_populations_menu(m))
            editm.add_cascade(label="Populations", menu=popm)
            editm.add_separator()
            editm.add_command(label="Add singlet gate",
                              command=self._add_singlet_gate)
            editm.add_command(label="FMO gating…",
                              command=self._open_fmo_gating)
            editm.add_command(label="Auto-clean gate",
                              command=self._create_autoclean_gate)
            editm.add_separator()
            editm.add_command(label="Preferences…",
                              command=self._open_preferences,
                              accelerator="Ctrl+,")
            btns["Edit"]['menu'] = editm

            viewm = tk.Menu(btns["View"], tearoff=0)
            dispm = tk.Menu(viewm, tearoff=0)

            for val, lbl, acc in (('all', 'All events', 'Ctrl+1'),
                                  ('highlight', 'Highlight gated', 'Ctrl+2'),
                                  ('filter', 'Filter to gated', 'Ctrl+3')):
                dispm.add_radiobutton(label=lbl, value=val,
                                      variable=self.gate_display_var,
                                      command=self._apply_display_mode,
                                      accelerator=acc)
            viewm.add_cascade(label="Display", menu=dispm)
            viewm.add_command(label="Reset plot view",
                              command=self._reset_plot_view,
                              accelerator="Ctrl+0")
            viewm.add_separator()
            viewm.add_command(label="Pipeline Workspace",
                              command=self._open_pipeline_workspace,
                              accelerator="F9")
            viewm.add_checkbutton(label="Show log / console",
                                  variable=self._show_log_var,
                                  command=self._toggle_log,
                                  accelerator="Ctrl+`")
            viewm.add_checkbutton(
                label="Show hover tips", variable=self._tooltips_enabled,
                command=lambda: write_pref('tooltips',
                                           bool(self._tooltips_enabled.get())))
            viewm.add_checkbutton(
                label="Dark figures in pop-ups", variable=self._dark_figs,
                command=lambda: write_pref('dark_figures',
                                           bool(self._dark_figs.get())))
            cornm = tk.Menu(viewm, tearoff=0)
            for val, lbl in (('off', 'OS default'),
                             ('top-left', 'Top-left of main window'),
                             ('top-right', 'Top-right of main window')):
                cornm.add_radiobutton(
                    label=lbl, value=val, variable=self._spawn_corner,
                    command=lambda: write_pref('spawn_corner',
                                               self._spawn_corner.get()))
            viewm.add_cascade(label="New windows open at", menu=cornm)
            viewm.add_command(label="Dock all panels",
                              command=self._dock_all_panels)
            viewm.add_separator()
            thememenu = tk.Menu(viewm, tearoff=0)
            for val, lbl in (('light', 'Light'), ('dark', 'Dark'),
                             ('midnight', 'Midnight (dark plot)')):
                thememenu.add_radiobutton(label=lbl, value=val,
                                          variable=self._theme_var,
                                          command=self._set_theme)
            viewm.add_cascade(label="Theme", menu=thememenu)
            btns["View"]['menu'] = viewm

            anam = tk.Menu(btns["Analyze"], tearoff=0)
            for lbl, cmd, acc in (
                    ("Statistics…", self._open_stats_window, "Ctrl+T"),
                    ("Frequencies…", self._open_frequency_window, ""),
                    ("Expression…", self._open_expression_window, ""),
                    ("Group comparison…", self._open_group_stats, ""),
                    ("Sample QC…", self._open_sample_qc_window, ""),
                    ("Methods & provenance…", self._open_methods_report, "")):
                anam.add_command(label=lbl, command=cmd, accelerator=acc)
            anam.add_separator()
            for lbl, cmd in (("Cluster…", self._open_cluster_dialog),
                             ("Compare embeddings…", self._open_dr_compare),
                             ("Cell cycle…", self._open_cell_cycle_dialog),
                             ("Trajectory…", self._open_trajectory_window),
                             ("Annotate…", self._open_annotation_window),
                             ("SOM tree…", self._open_flowsom_tree)):
                anam.add_command(label=lbl, command=cmd)
            btns["Analyze"]['menu'] = anam

            toolm = tk.Menu(btns["Tools"], tearoff=0)
            for lbl, cmd in (("Compensation…", self._open_comp_editor),
                             ("Compensation QC…", self._open_comp_qc),
                             ("Transforms…", self._open_transform_editor),
                             ("Calibration…", self._open_calibration_dialog)):
                toolm.add_command(label=lbl, command=cmd)
            toolm.add_separator()
            for lbl, cmd in (("Batch-norm (CytoNorm)",
                              self._batch_correct_cytonorm),
                             ("Spectral unmix…", self._open_spectral_unmix),
                             ("Figure layout…", self._open_figure_layout),
                             ("Gating tree diagram…", self._open_gate_tree)):
                toolm.add_command(label=lbl, command=cmd)
            toolm.add_separator()
            tmplm = tk.Menu(toolm, tearoff=0)
            tmplm.configure(
                postcommand=lambda m=tmplm: self._fill_template_menu(m))
            toolm.add_cascade(label="Templates", menu=tmplm)
            toolm.add_command(label="Save template…",
                              command=self._save_template)
            toolm.add_separator()
            toolm.add_command(label="Absolute counts…",
                              command=self._open_abs_counts)
            toolm.add_command(label="Export populations (FCS)…",
                              command=self._export_populations_fcs)
            toolm.add_separator()
            toolm.add_command(label="Voltage optimization…",
                              command=self._open_voltage_dialog)
            toolm.add_command(label="Compare FlowJo workspace…",
                              command=self._open_compare_wsp)
            toolm.add_command(label="FCS inspector…",
                              command=self._open_fcs_inspector)
            toolm.add_command(label="Watch folder…",
                              command=self._toggle_watch_folder)
            toolm.add_separator()
            toolm.add_command(label="History / audit…",
                              command=self._show_audit_window)
            btns["Tools"]['menu'] = toolm

            helpm = tk.Menu(btns["Help"], tearoff=0)
            helpm.add_command(label="Check for updates…",
                              command=self._check_for_updates)
            helpm.add_command(label="Run diagnostics…",
                              command=self._run_diagnostics)
            helpm.add_command(label="Environment…",
                              command=self._show_environment)
            helpm.add_command(label="Report a problem…",
                              command=self._report_a_problem)
            helpm.add_separator()
            helpm.add_command(label="Documentation",
                              command=self._open_documentation)
            helpm.add_command(label="Keyboard shortcuts",
                              command=self._show_shortcuts)
            helpm.add_command(label="About OpenFlo", command=self._show_about,
                              accelerator="F1")
            btns["Help"]['menu'] = helpm

            # Flatten the dropdown borders directly (belt-and-suspenders over
            # the option DB). Any hairline that survives is the OS popup frame,
            # which Tk can't recolour on Windows.
            for _m in (filem, editm, viewm, dispm, thememenu, anam, toolm,
                       tmplm, helpm):
                try:
                    _m.configure(bd=0, relief='flat', activeborderwidth=0)
                    # Per-entry help in the status bar as you navigate.
                    _m.bind('<<MenuSelect>>', self._on_menu_select, add='+')
                    _m.bind('<Unmap>', self._on_menu_unmap, add='+')
                except Exception:
                    pass

            self._theme_menubar(current_palette())
        except Exception as exc:
            print(f"[menubar] {exc}", flush=True)
