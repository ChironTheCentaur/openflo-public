"""
openflo.gui — OpenFlo pipeline GUI
Run with:  openflo-gui     (or:  python -m openflo.gui)
"""

# Surface C-level crashes (e.g. native Tk / tkdnd faults) with a Python
# stack so we can diagnose them instead of just seeing exit-139.
import faulthandler
import json
import os
import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# Cap BLAS thread pools BEFORE numpy is imported. OpenBLAS otherwise spins up
# one worker thread — each with its own memory arena — per core at import
# time. On a many-core box under memory pressure that allocation can fail
# ("OpenBLAS error: Memory allocation still failed after 10 retries, giving
# up"), and because the GUI launches via console-less pythonw the process
# just aborts with no window and no visible error. A modest cap bounds
# startup memory; the editor isn't BLAS-throughput-bound. Mirrors the same
# guard in cli.py. Env vars are read by these libraries only at import time.
def _cap_blas_threads(n=4):
    try:
        cores = os.cpu_count() or 2
        cap = str(max(1, min(n, cores)))
        for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ.setdefault(var, cap)
    except Exception:
        pass


_cap_blas_threads()

import numpy as np  # noqa: E402

# flowio is imported lazily inside the single call site that needs it
# (`_inspect_channels_for_labels` — see below). Module-level
# deferral isn't worth a proxy class for one use.

faulthandler.enable()

# Optional OS-level drag-and-drop support (drop FCS files from File
# Explorer onto the gate editor). When tkinterdnd2 is installed, the
# root Tk class is replaced so any descendant widget can register as a
# drop target. Without the package, the GUI still works — just no DnD.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore[import-not-found]
    _APP_BASE = TkinterDnD.Tk
    HAS_DND   = True
except ImportError:
    _APP_BASE = tk.Tk
    HAS_DND   = False
    DND_FILES = '*'        # sentinel; never used when HAS_DND is False

BASE    = os.path.dirname(os.path.abspath(__file__))

# Max FCS files loaded concurrently. Loading is memory-bound — FlowSample holds
# the full event matrix (raw + transformed) plus QC histograms / compensation /
# logicle transients, commonly 200–800 MB per large file — so a fixed pool of 2
# keeps peak RAM sane (~1.5–2 GB) while still overlapping I/O and CPU. Dropping a
# whole folder no longer spawns one thread per file (which exhausted memory and
# crashed the app). Bump this on a high-RAM machine.
_LOAD_POOL_SIZE = 2


# ── In-app log capture ─────────────────────────────────────────────────────────
#
# The GUI prints diagnostics to stdout/stderr (e.g. "[DnD] …", traceback text).
# A _StreamTee wraps the real stream once per process and fans every write out
# to the real stream PLUS any registered sink queues, so an editor's collapsible
# "log" pane can mirror that output. Writes only enqueue (thread-safe); the Tk
# Text is touched solely on the main thread by _drain_log.

class _StreamTee:
    def __init__(self, real):
        self._real = real
        self._sinks = []

    def write(self, s):
        try:
            self._real.write(s)
        except Exception:
            pass
        if s:
            for q in list(self._sinks):
                try:
                    q.put(s)
                except Exception:
                    pass
        return len(s) if s else 0

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass

    def add_sink(self, q):
        self._sinks.append(q)

    def remove_sink(self, q):
        try:
            self._sinks.remove(q)
        except ValueError:
            pass

    def __getattr__(self, name):
        # Delegate isatty/encoding/fileno/etc. to the wrapped stream.
        return getattr(self._real, name)


_STDOUT_TEE = None
_STDERR_TEE = None


def _install_log_tees():
    """Wrap sys.stdout/stderr in _StreamTee once; return (out_tee, err_tee)."""
    global _STDOUT_TEE, _STDERR_TEE
    if _STDOUT_TEE is None and sys.stdout is not None:
        _STDOUT_TEE = _StreamTee(sys.stdout)
        sys.stdout = _STDOUT_TEE
    if _STDERR_TEE is None and sys.stderr is not None:
        _STDERR_TEE = _StreamTee(sys.stderr)
        sys.stderr = _STDERR_TEE
    return _STDOUT_TEE, _STDERR_TEE


# ── View & Gate Editor ────────────────────────────────────────────────────────

class ViewGateEditorWindow(tk.Toplevel):
    """A FlowJo-style multi-sample viewer with editable threshold gates.

    Features
    --------
    Sample management
      • Load one or many FCS files; each is QC'd / compensated / logicle-
        transformed once and kept in memory.
      • Multi-select the sample list to overlay (or pick a single sample
        for pseudocolor / channel-heatmap views).

    Plot controls
      • X / Y channel pickers populated from the first loaded sample.
      • Plot mode:  dot  /  pseudocolor (KDE-shaded scatter)  /  contour
                    /  histogram (1-D, X channel only).
      • Color source:  By sample  /  By density  /  any channel.
        ("any channel" = FSC-A, SSC-A, or any fluor) gives FlowJo's
        "FSC-A vs SSC-A heatmapped by CD11b expression"-style view.

    Gates
      • Draggable threshold lines per channel (red dashed). One value per
        channel — if a channel is on both axes of different plots the
        line moves consistently when you drag.
      • "Apply gates" toggle filters events with `value > threshold` for
        any channel that has a gate set, before plotting.
      • "Apply Gates" button hands the gate dict to the parent GUI for
        use in the next pipeline run (same callback as the old editor).
    """

    PLOT_MODES = ('dot', 'pseudocolor', 'contour', 'histogram')
    SAMPLE_PALETTE = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')

    def __init__(self, parent, fcs_dir=None, labels_str='', on_save=None,
                 primary=False):
        super().__init__(parent)
        self.title("View & Gate Editor")
        self.geometry("1500x840")
        self.minsize(1180, 620)

        self.fcs_dir    = fcs_dir
        self.labels_str = labels_str
        self.on_save    = on_save
        # `primary` = this editor is the app's main window (launched via
        # main()); the pipeline-config App is hosted behind it. Closing a
        # primary editor tears the whole app down; a "Run Pipeline…"
        # button surfaces the hidden config window.
        self._primary   = primary
        self._app       = parent       # the App (pipeline-config root)

        # State
        self._samples         = {}    # name -> FlowSample
        self._sample_order    = []    # ordered insertion list (for stable legend)
        self._sample_colors   = {}    # name -> hex color
        # Trial grouping: each sample belongs to a trial (derived from its FCS
        # path — the grandparent folder; see workspace.derive_trial_name). The
        # Samples & Gates tree nests samples under their trial's header row.
        self._sample_trial    = {}    # name -> trial name
        self._trial_order     = []    # trial names in first-seen order
        # Manual Comps/Samples override (name -> bool). When set, it wins over
        # the name-based `is_comp_sample` guess — lets the user drag a sample
        # between the Comps and Samples subgroups to fix an import.
        self._sample_is_comp  = {}    # name -> bool (overrides is_comp_sample)
        self._channels        = []    # all channel names (from first sample)
        self._channel_labels  = {}    # detector -> antibody label
        self._channel_transform = {}  # detector -> transform method (display)
        # Per-sample gate stores. Each loaded sample gets its own gating
        # tree: gate id → gate dict (schema: flow_pipeline.gate_to_mask).
        # `self._gates` (and friends) are direct references to the active
        # sample's containers — switching active sample rebinds them via
        # `_set_active_sample`. Before any sample is loaded they're empty
        # placeholders so existing read paths don't blow up.
        self._sample_gates      = {}   # name → {gate_id: gate_dict}
        self._sample_gate_seq   = {}   # name → int (next id counter)
        self._sample_gate_order = {}   # name → list[gate_id] (insertion order)
        # Pending gates from a .wsp drop. Keyed by sample name; consumed
        # in `_on_loaded` once the FCS finishes parsing. Each value is a
        # flat list of gate dicts already in the editor's schema, with
        # `_import_id` + `parent_id` pointers among themselves.
        self._pending_sample_gates = {}   # name → list[gate_dict]
        # Per-sample session-restore bundle (grouping + gates) staged by a
        # session load and consumed in `_on_loaded`. Keyed by FILE PATH (not
        # name) so collision-disambiguated names can't mismatch on reload; holds
        # {'trial':…, 'is_comp':…, 'gates':[…]}.
        self._pending_sample_meta = {}    # normcase(abspath) → bundle dict
        # Auto-clean keep-mask cache: (name, gid) -> (id(sample.data),
        # recipe_signature, full-data pd.Series mask). Avoids recomputing the
        # (expensive) cleaning on every replot; invalidated automatically when
        # the sample's data object or the recipe signature changes.
        self._ac_cache = {}
        # Auto-clean drop-count cache: (name, gid) -> (id(sample.data),
        # recipe_signature, total, total_drop, {method_key: method_drop}).
        # Powers the per-gate / per-method "drops N (X%)" readout in the tree.
        self._ac_count_cache = {}
        # Auto-clean per-method removed-mask cache: (name, gid) -> (data_id,
        # signature, {method_key: full-data boolean removed-mask}). Powers the
        # per-method colouring of the "cleaned-out" overlay.
        self._ac_method_cache = {}
        # Names with a background FCS load in flight. Guards against double-
        # queuing the same sample (e.g. a folder drop where a .wsp and the
        # raw .fcs both reference the same file). Discarded in
        # `_on_loaded` / `_on_load_error`.
        self._loading = set()             # sample names currently loading
        # Path ⇄ name registry. Samples are keyed app-wide by name, but
        # day-organised drops reuse filenames across days (the same
        # 'Compensation Controls_…_008.fcs' lives under Day 6 AND Day 9), so a
        # bare basename would make the second silently collide with the first.
        # `_sample_name_for` disambiguates colliding names by their day/trial
        # and records the mapping here so every caller agrees on the name.
        self._path_to_name = {}           # abspath -> unique sample name
        self._name_to_path = {}           # unique sample name -> abspath
        # Bounded background loader. A fixed pool of `_LOAD_POOL_SIZE` daemon
        # threads drains `_load_queue`; the pool size IS the concurrency cap, so
        # no semaphore is needed. Replaces the old one-thread-per-file spawn
        # that crashed on big folder drops. Progress is two plain ints —
        # `_load_total` (bumped on enqueue, main thread) and `_load_done`
        # (bumped in each worker's finally, success OR error). int += is
        # GIL-atomic and both are read only on the Tk thread (in
        # `_update_progress_bar`), so no lock is required.
        self._load_queue = queue.Queue()  # (name, path) jobs; None = shutdown
        self._load_pool = []              # the daemon worker Threads
        self._load_pool_started = False   # one-shot lazy-spawn guard
        self._load_stop = threading.Event()  # set on close so workers exit
        self._load_total = 0              # FCS enqueued this run
        self._load_done = 0               # FCS finished this run (ok + failed)

        # Per-channel axis display config (keyed by channel name, not
        # by axis letter — switching the X combo to a different channel
        # picks up that channel's preferred scale).
        #   _channel_scale[ch] in {'linear', 'symlog', 'log'}; default log.
        #   _channel_range[ch] is (lo, hi) tuple or None for auto-range.
        # Gates live in data coordinates regardless — set_xscale/yscale
        # only changes the axis transform, so gate positions move with
        # the scale automatically.
        # NOTE: 'symlog' is still fully supported by the backend (and used by
        # the composite-logicle FuncScale internals), but it is no longer the
        # default or offered in the scale picker — its screen-uniform density
        # binning is prone to visible artefacts on some scatter views. Log is
        # the default on open; linear/log are the user-facing choices.
        self._channel_scale: dict[str, str] = {}
        self._channel_range: dict[str, tuple[float, float] | None] = {}
        # Default scale for any channel the user hasn't customised yet.
        self._default_channel_scale = 'log'
        self._active_sample     = None
        self._gates           = {}
        self._gate_id_seq     = 0
        self._gate_id_order   = []
        self._vlines          = {}    # gate_id -> matplotlib Line2D (vertical)
        self._hlines          = {}    # gate_id -> matplotlib Line2D (horizontal)
        self._shape_artists   = {}    # gate_id -> Patch (rect/polygon)
        # Drag state — extended to handle shape parts as well as 1D lines:
        #   ('v'|'h', key)                        line drag (existing)
        #   ('rect_corner', gid, 'bl|br|tl|tr')   rect corner
        #   ('rect_edge',   gid, 'top|bottom|left|right')   rect edge
        #   ('poly_vertex', gid, vertex_idx)      polygon vertex
        #   ('quad_origin', quad_set_id)          quadrant centre (both axes)
        #   ('quad_x',      quad_set_id)          quadrant x divider only
        #   ('quad_y',      quad_set_id)          quadrant y divider only
        self._drag_state      = None
        # Counter for the per-quad-set id so the 4 rect gates that come out
        # of a quadrant click can be treated together at hit-test / drag time.
        self._quad_set_seq    = 0
        # Translate-drag scratch: anchor point + a deep copy of the gate at
        # press time, so motion can apply (cur - anchor) to the original
        # geometry instead of compounding rounding errors. Set by _on_press
        # for ('rect_translate', ...) / ('poly_translate', ...) drags;
        # cleared on release.
        self._drag_translate_ctx = None
        self._replot_after_id = None  # for debouncing
        self._cbar            = None  # active colorbar (if any)
        # Provenance: append-only audit trail of analysis operations (load,
        # compensate, transform, clean, gate, cluster, batch-norm, unmix,
        # export). Persisted in the session; viewable / exportable. See
        # `_audit` and the History window.
        from .audit import AuditLog
        self._audit_log = AuditLog()
        self._audit_window = None
        # Last spectral-unmixing QC report (similarity + spillover-spread) and
        # the reference spectra it came from, for the Spectral-QC viewer.
        self._last_unmix_qc = None
        self._last_unmix_spectra = None
        # Active matplotlib region selector (set when tool != 'quadrant').
        self._selector        = None
        # Reserved slot for cluster phenotype labels — sample name →
        # {cluster_id: name}. The gate editor doesn't run clustering, so
        # this stays empty here; it's persisted/restored by the session
        # so a future labelling UI (or a loaded clustered sample) can
        # populate it without a schema change.
        self._cluster_labels  = {}

        # Backgating: populations to project (highlight) onto the current plot,
        # as a list of (sample_name, gate_id). Each is drawn in its own colour
        # on top of whatever's plotted, so you can see WHERE a downstream
        # population / cluster sits on any axes. Transient view state.
        self._backgate: list[tuple[str, str]] = []

        # Undo/redo — snapshot stacks of the gate-related state. Mutations
        # call _checkpoint() before changing anything; multiple mutations in
        # one Tk event coalesce into a single undo step (see _checkpoint).
        # Programmatic bulk loads set _suspend_undo so they don't pollute
        # the history.
        self._undo_stack    = []
        self._redo_stack    = []
        self._undo_pending  = False
        self._suspend_undo  = False
        self._UNDO_MAX      = 100

        self._build()
        # Undo / redo keyboard shortcuts (text widgets keep their own via
        # the focus guard in _undo/_redo).
        self.bind_all('<Control-z>', self._undo)
        self.bind_all('<Control-Z>', self._redo)   # Ctrl+Shift+Z
        self.bind_all('<Control-y>', self._redo)
        # Save the session on window close, and offer to resume the last
        # one shortly after the window paints.
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self.after(200, self._maybe_resume_session)

    # ── UI ───────────────────────────────────────────────────────────────

    def _build(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)    # the only content row now

        # ── Left column: merged samples-and-gates tree + buttons ─────────
        # The samples are root nodes; their gates are nested children.
        # This puts each sample's gates visually "attached" to it,
        # matching the FlowJo gating-hierarchy layout. No separate
        # bottom strip — Add/Remove sample buttons live here too.
        left = ttk.Frame(self, padding=6)
        left.grid(row=0, column=0, sticky='ns')
        left.rowconfigure(2, weight=1)   # tree expands

        ttk.Label(left, text="Samples & Gates",
                  font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky='nw')

        sb_row = ttk.Frame(left)
        sb_row.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(2, 4))
        ttk.Button(sb_row, text="Add FCS…", width=11,
                   command=self._add_samples).pack(side='left')
        ttk.Button(sb_row, text="Load CSV…", width=10,
                   command=self._load_processed_data).pack(
            side='left', padx=(4, 0))
        ttk.Button(sb_row, text="Remove",   width=8,
                   command=self._remove_selected).pack(side='left', padx=(4, 0))
        ttk.Button(sb_row, text="Batch-norm", width=11,
                   command=self._batch_correct_cytonorm).pack(
            side='left', padx=(4, 0))
        ttk.Button(sb_row, text="Unmix", width=7,
                   command=self._open_spectral_unmix).pack(
            side='left', padx=(4, 0))
        ttk.Button(sb_row, text="Figure…", width=8,
                   command=self._open_figure_layout).pack(
            side='left', padx=(4, 0))

        gf = ttk.Frame(left)
        gf.grid(row=2, column=0, columnspan=2, sticky='nsew', pady=(2, 4))
        gf.columnconfigure(0, weight=1)
        gf.rowconfigure(0, weight=1)
        # Treeview as a tree-with-one-trailing-column. The gate description
        # lives IN the tree column (#0) so it indents under its parent and
        # the disclosure triangle has room — that's how the hierarchy
        # becomes visible (FlowJo gating-tree look). The trailing column
        # is just the on/off marker. Click on the marker toggles enabled;
        # row click is the default Treeview selection (and becomes the
        # parent for the next gate the user creates).
        self.gate_tv = ttk.Treeview(
            gf, columns=('on',),
            show='tree headings', height=18, selectmode='extended')
        self.gate_tv.heading('#0', text='Gate')
        # Clicking the ✓ header toggles plot-inclusion for every sample
        # at once — checks all when any are off, otherwise unchecks all.
        self.gate_tv.heading('on', text='✓',
                             command=self._toggle_all_sample_plots)
        self.gate_tv.column('#0', width=220, anchor='w', stretch=True)
        self.gate_tv.column('on', width=26,  anchor='center', stretch=False)
        self.gate_tv.grid(row=0, column=0, sticky='nsew')
        gate_sb = ttk.Scrollbar(gf, orient='vertical',
                                command=self.gate_tv.yview)
        gate_sb.grid(row=0, column=1, sticky='ns')
        self.gate_tv.configure(yscrollcommand=gate_sb.set)
        # Press / motion / release lets us tell a click apart from a drag.
        # A bare click toggles the ☑/☐ column; a drag past the threshold
        # initiates a gate-reparent gesture (drop on another gate/sample).
        self.gate_tv.bind('<Button-1>',         self._on_tv_press)
        self.gate_tv.bind('<B1-Motion>',        self._on_tv_motion)
        self.gate_tv.bind('<ButtonRelease-1>',  self._on_tv_release)
        self.gate_tv.bind('<Double-Button-1>',  self._on_tv_double_click)
        self.gate_tv.bind('<<TreeviewSelect>>', self._on_tree_select)
        # Drag-state scratch.
        self._press_iid    = None
        self._press_col    = None
        self._press_x      = 0
        self._press_y      = 0
        self._drag_active  = False
        self._press_selection = ()     # multi-selection captured at press
        self._drag_threshold = 5     # pixels before a press counts as a drag
        # Display tags.
        self.gate_tv.tag_configure('off',     foreground='grey')
        self.gate_tv.tag_configure('loading', foreground='grey')
        self.gate_tv.tag_configure('sample',
                                   font=('TkDefaultFont', 9, 'bold'))
        self.gate_tv.tag_configure('drop_target',
                                   background='#fff2a8')   # transient hover

        # Bottom action buttons — uniform size (equal grid columns), compact,
        # so the left column has room for the collapsible log pane below.
        # ↶/↷ stay as narrow icon buttons outside the weighted columns.
        gb_row = ttk.Frame(left)
        gb_row.grid(row=3, column=0, columnspan=2, sticky='ew')
        for c in range(5):
            gb_row.columnconfigure(c, weight=1, uniform='gb')
        ttk.Button(gb_row, text="Clear", command=self._clear_selected_gate).grid(
            row=0, column=0, sticky='ew')
        ttk.Button(gb_row, text="Clear all", command=self._clear_all).grid(
            row=0, column=1, sticky='ew', padx=(3, 0))
        ttk.Button(gb_row, text="Auto-clean",
                   command=self._create_autoclean_gate).grid(
            row=0, column=2, sticky='ew', padx=(3, 0))
        ttk.Button(gb_row, text="Copy", command=self._open_copy_gates_dialog).grid(
            row=0, column=3, sticky='ew', padx=(3, 0))
        ttk.Button(gb_row, text="Pops", command=self._open_clusters_menu).grid(
            row=0, column=4, sticky='ew', padx=(3, 0))
        ttk.Button(gb_row, text="↶", width=2, command=self._undo).grid(
            row=0, column=5, padx=(3, 0))
        ttk.Button(gb_row, text="↷", width=2, command=self._redo).grid(
            row=0, column=6, padx=(2, 0))

        # Discoverability hint (one line — "Clear"=gates of the selected gate/
        # sample/trial, "Clear all"=all gates (samples kept), "Pops"=
        # populations/clusters menu; double-click the plot to add a gate).
        ttk.Label(left,
                  text="Tip: double-click the plot to add a gate · click a "
                       "sample to make it active.",
                  font=('TkDefaultFont', 8), foreground='grey',
                  wraplength=300, justify='left').grid(
            row=4, column=0, columnspan=2, sticky='w', pady=(3, 0))

        # Display mode (3-way): how should gates affect the rendered cloud?
        #   all       — ignore gates entirely (just show all events).
        #   highlight — base population in grey + each enabled LEAF gate's
        #               cumulative-chain events overlaid in that leaf's
        #               colour. Shows forks side-by-side.
        #   filter    — show only events satisfying every enabled gate.
        # The new radio takes the slot the old Apply-gates checkbox used.
        # apply_gates_var stays as a back-compat shim some code paths read.
        self.gate_display_var = tk.StringVar(value='all')
        self.apply_gates_var = tk.BooleanVar(value=False)

        def _on_display_changed(*_):
            self.apply_gates_var.set(self.gate_display_var.get() == 'filter')
            self._schedule_replot(0)

        disp_frame = ttk.Frame(left)
        disp_frame.grid(row=5, column=0, columnspan=2, sticky='w', pady=(6, 0))
        ttk.Label(disp_frame, text="Display:",
                  foreground='grey').grid(row=0, column=0, sticky='w')
        for i, (val, lbl) in enumerate([
                ('all',       'All events'),
                ('highlight', 'Highlight gated'),
                ('filter',    'Filter to gated')]):
            ttk.Radiobutton(disp_frame, text=lbl, value=val,
                            variable=self.gate_display_var,
                            command=_on_display_changed).grid(
                row=0, column=1 + i, sticky='w', padx=(6, 0))

        ttk.Separator(left, orient='horizontal').grid(
            row=6, column=0, columnspan=2, sticky='ew', pady=(8, 4))

        # Templates (gates only) + full sessions (samples + gates + display
        # state) share one compact row of four equal-width buttons. A session
        # captures more than a template or a FlowJo-representable .wsp.
        ts_row = ttk.Frame(left)
        ts_row.grid(row=7, column=0, columnspan=2, sticky='ew', pady=(0, 3))
        for c in range(4):
            ts_row.columnconfigure(c, weight=1, uniform='ts')
        tmpl_mb = ttk.Menubutton(ts_row, text="Templates ▾")
        tmpl_menu = tk.Menu(tmpl_mb, tearoff=0)
        tmpl_mb['menu'] = tmpl_menu
        tmpl_menu.configure(postcommand=lambda m=tmpl_menu: self._fill_template_menu(m))
        tmpl_mb.grid(row=0, column=0, sticky='ew')
        ttk.Button(ts_row, text="Save Tmpl",
                   command=self._save_template).grid(
            row=0, column=1, sticky='ew', padx=(3, 0))
        ttk.Button(ts_row, text="Load Sess",
                   command=self._load_session).grid(
            row=0, column=2, sticky='ew', padx=(3, 0))
        ttk.Button(ts_row, text="Save Sess",
                   command=self._save_session).grid(
            row=0, column=3, sticky='ew', padx=(3, 0))

        # Export every loaded sample's gate tree to a single .wsp that
        # FlowJo can open. Re-uses the same XML writer the pipeline does.
        ttk.Button(left, text="Export → FlowJo .wsp",
                   command=self._export_flowjo_wsp).grid(
            row=9, column=0, columnspan=2, sticky='ew', pady=(0, 3))

        # Compensation matrix editor + per-channel transform editor share
        # a row. Compensation auto-imports from the active sample's $SPILL /
        # a sibling .wsp / compensation.csv; Transforms re-maps each
        # channel's display transform (logicle/hyperlog/asinh/log/linear).
        comp_row = ttk.Frame(left)
        comp_row.grid(row=10, column=0, columnspan=2, sticky='ew', pady=(0, 3))
        ttk.Button(comp_row, text="Compensation…",
                   command=self._open_comp_editor).pack(
            side='left', fill='x', expand=True)
        ttk.Button(comp_row, text="Transforms…",
                   command=self._open_transform_editor).pack(
            side='left', fill='x', expand=True, padx=(3, 0))
        ttk.Button(comp_row, text="Calibration…",
                   command=self._open_calibration_dialog).pack(
            side='left', fill='x', expand=True, padx=(3, 0))

        # Population statistics table (FlowJo-style), the frequency /
        # group-comparison plots, and the provenance / audit trail viewer
        # share a row — all review / reporting tools.
        rep_row = ttk.Frame(left)
        rep_row.grid(row=11, column=0, columnspan=2, sticky='ew', pady=(0, 3))
        ttk.Button(rep_row, text="Statistics…",
                   command=self._open_stats_window).pack(
            side='left', fill='x', expand=True)
        ttk.Button(rep_row, text="Frequencies…",
                   command=self._open_frequency_window).pack(
            side='left', fill='x', expand=True, padx=(3, 0))
        ttk.Button(rep_row, text="Expression…",
                   command=self._open_expression_window).pack(
            side='left', fill='x', expand=True, padx=(3, 0))
        ttk.Button(rep_row, text="Sample QC…",
                   command=self._open_sample_qc_window).pack(
            side='left', fill='x', expand=True, padx=(3, 0))
        ttk.Button(rep_row, text="History…",
                   command=self._show_audit_window).pack(
            side='left', fill='x', expand=True, padx=(3, 0))

        # Clustering (Phenograph / FlowSOM + UMAP) and cell-cycle share a
        # row. Clustering runs on loaded samples and auto-imports the
        # resulting populations; cell-cycle models G1/S/G2M.
        clu_row = ttk.Frame(left)
        clu_row.grid(row=12, column=0, columnspan=2, sticky='ew', pady=(0, 3))
        ttk.Button(clu_row, text="Cluster…",
                   command=self._open_cluster_dialog).pack(
            side='left', fill='x', expand=True)
        ttk.Button(clu_row, text="Cell cycle…",
                   command=self._open_cell_cycle_dialog).pack(
            side='left', fill='x', expand=True, padx=(3, 0))
        ttk.Button(clu_row, text="Trajectory…",
                   command=self._open_trajectory_window).pack(
            side='left', fill='x', expand=True, padx=(3, 0))
        ttk.Button(clu_row, text="Annotate…",
                   command=self._open_annotation_window).pack(
            side='left', fill='x', expand=True, padx=(3, 0))
        ttk.Button(clu_row, text="SOM tree…",
                   command=self._open_flowsom_tree).pack(
            side='left', fill='x', expand=True, padx=(3, 0))

        ttk.Button(left, text="Pipeline Workspace…",
                   command=self._open_pipeline_workspace).grid(
            row=13, column=0, columnspan=2, sticky='ew', pady=(0, 3))

        # One-click HTML analysis report bundling metadata, sample/gate
        # summary, the current plot, population statistics, a cluster heatmap
        # (when clustered) and the provenance trail into one portable file.
        ttk.Button(left, text="Analysis report (HTML)…",
                   command=self._export_report).grid(
            row=14, column=0, columnspan=2, sticky='ew', pady=(0, 3))

        # ── Log + interactive Python console (mirrors stdout/stderr) ─────
        # Shown by default; the prompt runs Python in-process against the live
        # editor (`editor`/`self`, `samples`, `np`, `pd` are pre-bound).
        logbar = ttk.Frame(left)
        logbar.grid(row=14, column=0, columnspan=2, sticky='ew', pady=(4, 0))
        self._show_log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(logbar, text="Show log / console",
                        variable=self._show_log_var,
                        command=self._toggle_log).pack(side='left')
        ttk.Button(logbar, text="Clear log", width=9,
                   command=self._clear_log).pack(side='right')

        self._log_frame = ttk.Frame(left)
        self._log_frame.grid(row=15, column=0, columnspan=2, sticky='ew',
                             pady=(2, 0))
        self._log_frame.columnconfigure(0, weight=1)
        self._log_text = tk.Text(self._log_frame, height=7, wrap='none',
                                 font=('Consolas', 8), background='#1e1e1e',
                                 foreground='#d4d4d4',
                                 insertbackground='#d4d4d4',
                                 relief='flat', state='disabled')
        _log_sb = ttk.Scrollbar(self._log_frame, orient='vertical',
                                command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=_log_sb.set)
        self._log_text.grid(row=0, column=0, sticky='ew')
        _log_sb.grid(row=0, column=1, sticky='ns')

        # Interactive prompt. Enter runs the line through a persistent
        # code.InteractiveConsole; its output/repr/tracebacks flow back through
        # the stdout/stderr tee into the Text above. Up/Down = history.
        prompt_row = ttk.Frame(self._log_frame)
        prompt_row.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(2, 0))
        prompt_row.columnconfigure(1, weight=1)
        self._console_prompt = tk.StringVar(value='>>>')
        ttk.Label(prompt_row, textvariable=self._console_prompt,
                  font=('Consolas', 8)).grid(row=0, column=0, padx=(0, 4))
        self._console_entry = ttk.Entry(prompt_row, font=('Consolas', 8))
        self._console_entry.grid(row=0, column=1, sticky='ew')
        self._console_entry.bind('<Return>', self._console_run)
        self._console_entry.bind('<Up>', self._console_history_prev)
        self._console_entry.bind('<Down>', self._console_history_next)
        self._console_history = []
        self._console_hist_idx = 0
        self._console = None          # built lazily on first command

        # Mirror process stdout/stderr into the pane (thread-safe via a queue;
        # the Text is only ever touched on the main thread, by _drain_log).
        self._log_queue = queue.Queue()
        try:
            self._log_tees = [t for t in _install_log_tees() if t is not None]
            for tee in self._log_tees:
                tee.add_sink(self._log_queue)
        except Exception:
            self._log_tees = []
        self.after(400, self._drain_log)
        self._toggle_log()            # apply default visibility (shown)

        ttk.Button(left, text="Close",
                   command=self._on_close).grid(
            row=16, column=0, columnspan=2, sticky='ew')

        # ── Right column: a horizontal sash splitting the plot (left) from
        # the docked Pipeline Workspace (right). The workspace pane is hidden
        # by default — "Pipeline Workspace…" reveals it, which shrinks the
        # plot to share the space (drag the sash to taste).
        self._editor_paned = ttk.PanedWindow(self, orient='horizontal')
        self._editor_paned.grid(row=0, column=1, sticky='nsew')
        right = ttk.Frame(self._editor_paned, padding=(0, 6, 6, 6))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self._editor_paned.add(right, weight=4)
        from .workspace import WorkspacePanel
        self._workspace_panel = WorkspacePanel(self._editor_paned, editor=self,
                                               on_before_change=self._checkpoint)
        self._workspace_shown = False

        # Top controls. Each logical group is its own packed sub-row, so the
        # bar WRAPS downward instead of running off the right edge on a narrow
        # window (e.g. with the Pipeline Workspace docked) — previously Mode,
        # Color and the downsample toggles were clipped. Packing per sub-row
        # also avoids grid column-width bleed between unrelated rows.
        ctrl = ttk.Frame(right)
        ctrl.grid(row=0, column=0, sticky='ew', pady=(0, 4))

        # Row A — axes + colour.
        row_a = ttk.Frame(ctrl)
        row_a.pack(fill='x', anchor='w')
        ttk.Label(row_a, text="X:").pack(side='left', padx=(0, 4))
        xframe = ttk.Frame(row_a)
        xframe.pack(side='left', padx=(0, 12))
        self.x_combo = ttk.Combobox(xframe, width=20, state='readonly')
        self.x_combo.pack(side='left')
        self.x_combo.bind('<<ComboboxSelected>>',
                          lambda *_: self._schedule_replot(0))
        ttk.Button(xframe, text='⚙', width=2,
                   command=lambda: self._open_axis_dialog('x')).pack(
            side='left', padx=(2, 0))

        ttk.Label(row_a, text="Y:").pack(side='left', padx=(0, 4))
        yframe = ttk.Frame(row_a)
        yframe.pack(side='left', padx=(0, 12))
        self.y_combo = ttk.Combobox(yframe, width=20, state='readonly')
        self.y_combo.pack(side='left')
        self.y_combo.bind('<<ComboboxSelected>>',
                          lambda *_: self._schedule_replot(0))
        ttk.Button(yframe, text='⚙', width=2,
                   command=lambda: self._open_axis_dialog('y')).pack(
            side='left', padx=(2, 0))

        ttk.Label(row_a, text="Color:").pack(side='left', padx=(0, 4))
        self.color_combo = ttk.Combobox(row_a, width=18, state='readonly')
        self.color_combo.pack(side='left')
        self.color_combo.bind('<<ComboboxSelected>>',
                              lambda *_: self._schedule_replot(0))

        # Row B — plot mode + KDE toggle.
        row_b = ttk.Frame(ctrl)
        row_b.pack(fill='x', anchor='w', pady=(6, 0))
        ttk.Label(row_b, text="Mode:").pack(side='left', padx=(0, 4))
        self.mode_var = tk.StringVar(value='dot')
        for m in self.PLOT_MODES:
            ttk.Radiobutton(row_b, text=m, value=m, variable=self.mode_var,
                            command=self._on_mode_changed).pack(
                side='left', padx=(0, 8))
        self.true_kde_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row_b, text="True Gaussian KDE (slow)",
                        variable=self.true_kde_var,
                        command=lambda: self._schedule_replot(0)).pack(
            side='left', padx=(12, 0))
        # Overlay the events the auto-clean recipe removes, in red, on top of
        # whatever's plotted — so cleaning artefacts stay visible against the
        # full sample even at a tiny error rate. Bypasses the display cap.
        self.show_removed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row_b, text="⚠ Show cleaned-out events",
                        variable=self.show_removed_var,
                        command=lambda: self._schedule_replot(0)).pack(
            side='left', padx=(12, 0))
        # Contour mode draws a faint per-event scatter under the contour
        # lines. 'Contour scatter' is the master switch (off = clean
        # contour-only plot). 'Outliers' (only meaningful when scatter is on)
        # toggles the sparse low-density points OUTSIDE the contoured
        # population: off = show just the within-population points.
        self.contour_scatter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_b, text="Contour scatter",
                        variable=self.contour_scatter_var,
                        command=lambda: self._schedule_replot(0)).pack(
            side='left', padx=(12, 0))
        self.contour_outliers_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_b, text="Outliers",
                        variable=self.contour_outliers_var,
                        command=lambda: self._schedule_replot(0)).pack(
            side='left', padx=(8, 0))
        # Histogram Y-axis mode. Only affects 'histogram' mode (disabled
        # otherwise). Fraction (default) = events per bin ÷ sample total;
        # Count = raw events per bin (respects the auto-downsample toggle so
        # overlays stay comparable); % of Max = each curve's peak set to 100.
        ttk.Label(row_b, text="Hist Y:").pack(side='left', padx=(12, 2))
        self.hist_y_mode = tk.StringVar(value='Fraction')
        self.hist_y_combo = ttk.Combobox(
            row_b, textvariable=self.hist_y_mode, state='disabled', width=10,
            values=['Fraction', 'Count', '% of Max'])
        self.hist_y_combo.pack(side='left')
        self.hist_y_combo.bind('<<ComboboxSelected>>',
                               lambda *_: self._schedule_replot(0))

        # Row C — gate-shape tool + auto-gate. Auto-gate now offers
        # well-posed, reviewable proposals (singlet ratio-band, BIC-selected
        # GMM ellipses, 1-D valley/Otsu threshold) each with a quality score,
        # rather than the old single-contour heuristic.
        row_c = ttk.Frame(ctrl)
        row_c.pack(fill='x', anchor='w', pady=(6, 0))
        ttk.Label(row_c, text="Tool:").pack(side='left', padx=(0, 4))
        self.gate_tool_var = tk.StringVar(value='quadrant')
        tools = [('Quadrant',  'quadrant'),
                 ('Rectangle', 'rectangle'),
                 ('Polygon',   'polygon'),
                 ('Ellipse',   'ellipse'),
                 ('Lasso',     'lasso'),
                 ('Edit',      'edit')]
        for lbl, val in tools:
            ttk.Radiobutton(row_c, text=lbl, value=val,
                            variable=self.gate_tool_var,
                            command=self._activate_gate_tool).pack(
                side='left', padx=(0, 8))
        ttk.Button(row_c, text="Auto-gate", width=10,
                   command=self._auto_gate).pack(
            side='left', padx=(4, 0))

        # Row D — downsample toggles. Display-only is ON by default so a
        # multi-sample overlay draws the same count from each (FlowJo-like);
        # propagate is OFF (it trims FlowSample.data, shrinking clustering /
        # stats input too).
        row_d = ttk.Frame(ctrl)
        row_d.pack(fill='x', anchor='w', pady=(4, 0))
        self.ds_display_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row_d, text="Auto-downsample display to smallest sample",
            variable=self.ds_display_var,
            command=self._on_downsample_display_toggled).pack(
            side='left', padx=(0, 16))
        self.ds_propagate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row_d, text="…and propagate to data (affects clustering / stats)",
            variable=self.ds_propagate_var,
            command=self._on_downsample_propagate_toggled).pack(side='left')
        # Max points actually drawn per sample in scatter / pseudocolor /
        # contour (histograms are uncapped). Editable — pick a preset or type
        # a number; 'All' removes the cap (slow on millions of events).
        ttk.Label(row_d, text="   Max points:").pack(side='left', padx=(16, 2))
        self.max_points_var = tk.StringVar(value='60000')
        mp_combo = ttk.Combobox(
            row_d, textvariable=self.max_points_var, width=8,
            values=['20000', '60000', '100000', '250000', '500000', 'All'])
        mp_combo.pack(side='left')
        mp_combo.bind('<<ComboboxSelected>>',
                      lambda *_: self._on_max_points_changed())
        mp_combo.bind('<Return>', lambda *_: self._on_max_points_changed())

        # Row E — tool gesture hint (updates with the active tool).
        self.tool_hint_var = tk.StringVar(value='')
        ttk.Label(ctrl, textvariable=self.tool_hint_var,
                  foreground='#666', font=('TkDefaultFont', 8, 'italic')
                  ).pack(fill='x', anchor='w', padx=(0, 4), pady=(2, 4))

        # Plot canvas
        cf = ttk.Frame(right)
        cf.grid(row=1, column=0, sticky='nsew')
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=1)
        self.fig    = Figure(figsize=(9, 6), dpi=100)
        self.ax     = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=cf)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event',  self._on_motion)

        # ── Histogram slider panel (visible only in histogram mode) ──────
        self.slider_panel = ttk.Frame(right)
        self.slider_panel.columnconfigure(3, weight=1)
        self.slider_kind_var = tk.StringVar(value='threshold')
        ttk.Label(self.slider_panel,
                  text="Histogram gate:").grid(row=0, column=0, sticky='w',
                                                padx=(0, 8))
        ttk.Radiobutton(self.slider_panel, text='Threshold (bisect)',
                        value='threshold', variable=self.slider_kind_var,
                        command=self._on_slider_kind_changed).grid(
            row=0, column=1, padx=(0, 6), sticky='w')
        ttk.Radiobutton(self.slider_panel, text='Interval (interior)',
                        value='interval', variable=self.slider_kind_var,
                        command=self._on_slider_kind_changed).grid(
            row=0, column=2, padx=(0, 6), sticky='w')
        self._slider_axis_label = ttk.Label(self.slider_panel,
                                            text="", foreground='grey')
        self._slider_axis_label.grid(row=0, column=3, sticky='e')

        # Lo slider: always present (threshold value OR interval lo).
        self._slider_updating = False     # guard to suppress feedback loops
        self.slider_lo = ttk.Scale(self.slider_panel, orient='horizontal',
                                   from_=0.0, to=1.0,
                                   command=self._on_slider_lo)
        self.slider_lo.grid(row=1, column=0, columnspan=3,
                            sticky='ew', padx=(0, 6), pady=(4, 0))
        self.slider_lo_lbl = ttk.Label(self.slider_panel, text='—', width=12)
        self.slider_lo_lbl.grid(row=1, column=3, sticky='e', pady=(4, 0))

        # Hi slider: shown only when slider_kind_var == 'interval'.
        self.slider_hi = ttk.Scale(self.slider_panel, orient='horizontal',
                                   from_=0.0, to=1.0,
                                   command=self._on_slider_hi)
        self.slider_hi_lbl = ttk.Label(self.slider_panel, text='—', width=12)
        # (slider_hi widgets are grid()/grid_remove()'d by _rebuild_slider_gate)

        # Track the gate-id currently being driven by the sliders, so the
        # user dragging doesn't keep creating new gates.
        self._slider_gate_id = None
        self._slider_channel = None       # last channel we configured for

        # Background-load progress: a determinate bar + "N/M loaded" label in
        # their own frame at row 3 (the slider panel owns row 2). Hidden at idle
        # via grid_remove(); shown/extended by `_update_progress_bar` while the
        # bounded loader runs. The status label moves to row 4.
        self._load_progress_var = tk.DoubleVar(value=0)
        self._load_progress_lbl_var = tk.StringVar(value='')
        self._load_progress_frame = ttk.Frame(right)
        self._load_progress_frame.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(
            self._load_progress_frame, mode='determinate',
            variable=self._load_progress_var)
        self.progress_bar.grid(row=0, column=0, sticky='ew')
        ttk.Label(self._load_progress_frame,
                  textvariable=self._load_progress_lbl_var,
                  foreground='grey').grid(row=0, column=1, sticky='e',
                                          padx=(8, 0))
        self._load_progress_frame.grid(row=3, column=0, sticky='ew',
                                       pady=(4, 0))
        self._load_progress_frame.grid_remove()

        self.status_var = tk.StringVar(value="Add one or more FCS files to begin.")
        ttk.Label(right, textvariable=self.status_var,
                  foreground='grey').grid(row=4, column=0, sticky='w', pady=(4, 0))

        right.rowconfigure(2, weight=0)   # slider panel is fixed-height

        # Plot-inclusion state per sample (drives which samples render in
        # the canvas — controlled by the checkbox in the merged tree).
        self._sample_plot_enabled = {}    # name -> bool

        # Internal clipboard for gate cut/copy/paste. `kind` is
        # 'gate_tree' (payload = list of deep-copied gate dicts with
        # a temporary '_clip_id' on each) or 'sample_paths'
        # (payload = list of file paths). The OS clipboard is also
        # consulted on paste — see `_paste_fcs_from_clipboard`.
        self._clip_kind    = None
        self._clip_payload = None

        # Tree-level keyboard shortcuts + right-click context menu.
        self.gate_tv.bind('<Control-c>',     self._on_copy)
        self.gate_tv.bind('<Control-x>',     self._on_cut)
        self.gate_tv.bind('<Control-v>',     self._on_paste)
        self.gate_tv.bind('<Control-C>',     self._on_copy)
        self.gate_tv.bind('<Control-X>',     self._on_cut)
        self.gate_tv.bind('<Control-V>',     self._on_paste)
        self.gate_tv.bind('<Delete>',        self._on_delete_key)
        self.gate_tv.bind('<Button-3>',      self._on_right_click)

        # Register the editor window itself as the OS file-drop target.
        # We use the Toplevel rather than the Treeview because tkdnd builds
        # under some Tk versions are unstable when binding to ttk widgets.
        # Drops anywhere in the window route to the same handler.
        if HAS_DND:
            try:
                self.drop_target_register(DND_FILES)   # type: ignore[attr-defined]
                self.dnd_bind('<<Drop>>',              # type: ignore[attr-defined]
                              self._on_dnd_drop)
            except Exception as exc:
                print(f"[DnD] register failed: {exc}", flush=True)

        self._render_placeholder()

    # ── Sample loading ───────────────────────────────────────────────────

    def _add_samples(self):
        """Add-FCS button: dialog picker, then queue.

        Accepts both ``.fcs`` and ``.wsp``. A workspace is "exploded":
        every ``<Sample>`` it references is queued for load, and the
        gate trees attached to each ``<SampleNode>`` are merged into
        that sample's per-sample gate storage as it finishes loading.
        """
        init = self.fcs_dir if self.fcs_dir and os.path.isdir(self.fcs_dir) else BASE
        paths = filedialog.askopenfilenames(
            initialdir=init, title="Select FCS file(s) or FlowJo workspace",
            filetypes=[
                ('FCS & FlowJo workspace', '*.fcs *.wsp'),
                ('FCS files',              '*.fcs'),
                ('FlowJo workspace',       '*.wsp'),
                ('All files',              '*.*')])
        if not paths:
            return
        fcs_paths, wsp_paths = [], []
        for p in paths:
            (wsp_paths if p.lower().endswith('.wsp') else fcs_paths).append(p)
        # Process workspaces first so their pending-gates map is populated
        # before any FCS load completes.
        for wsp in wsp_paths:
            self._ingest_wsp(wsp)
        if fcs_paths:
            self._queue_fcs_loads(fcs_paths)

    def _load_processed_data(self):
        """Load a pipeline-processed CSV (already compensated + transformed,
        carrying cluster / UMAP / flowsom columns) as a sample, via
        FlowSample.from_dataframe. No QC / compensation / transform is
        re-applied. A sibling ``<name>_labels.json`` ({detector: label}) is
        picked up if present."""
        init = self.fcs_dir if self.fcs_dir and os.path.isdir(self.fcs_dir) else BASE
        paths = filedialog.askopenfilenames(
            initialdir=init, title="Load processed data (CSV)",
            filetypes=[('Processed CSV', '*.csv'), ('All files', '*.*')])
        if not paths:
            return
        import json

        import pandas as pd

        from .pipeline import FlowSample
        added = 0
        for p in paths:
            name = os.path.basename(p).rsplit('.', 1)[0]
            if name.endswith('_processed'):
                name = name[:-len('_processed')]
            if name in self._samples:
                self.status_var.set(f"{name} already loaded — skipped.")
                continue
            try:
                df = pd.read_csv(p)
            except Exception as exc:
                self.status_var.set(f"Load CSV failed: {exc}")
                continue
            labels = None
            sidecar = os.path.join(os.path.dirname(p), f'{name}_labels.json')
            if os.path.isfile(sidecar):
                try:
                    with open(sidecar, encoding='utf-8') as fh:
                        labels = json.load(fh)
                except Exception:
                    labels = None
            s = FlowSample.from_dataframe(df, name=name, labels=labels, path=p)
            self._on_loaded(name, s)
            added += 1
        if added:
            self.status_var.set(
                f"Loaded {added} processed sample(s). Use Populations… to "
                "import cluster/FlowSOM columns; plot UMAP1/UMAP2 to view.")

    def _ingest_wsp(self, wsp_path):
        """Parse a FlowJo workspace, discover its referenced FCS files,
        and stage each sample's gate tree for application once the FCS
        finishes loading.

        We DON'T apply gates immediately — they go into
        ``self._pending_sample_gates`` keyed by sample name. The
        ``_on_loaded`` hook drains that map per-sample.

        Sample-name resolution:
          - ``<SampleNode name="...">`` provides the display name
            (matches what FlowJo shows).
          - ``<DataSet uri="...">`` gives the FCS path. We try the path
            as-is, then the WSP's own directory, then the user-set
            ``self.fcs_dir``.
        """
        from xml.etree import ElementTree as ET

        from .compare import _resolve_fcs_uri
        from .pipeline import WspReader

        try:
            reader = WspReader(wsp_path)
        except Exception as exc:
            self.status_var.set(f"[WSP] {os.path.basename(wsp_path)}: {exc}")
            return

        # Re-parse the file ourselves to walk per-sample. WspReader's
        # extract_gates() flattens; we need samples + their own gate
        # subtrees so this editor can attach the right tree to the right
        # FCS.
        try:
            ns_re = re.compile(r'\{.*?\}')
            tree = ET.parse(wsp_path)
            root = tree.getroot()
            for elem in root.iter():
                elem.tag = ns_re.sub('', elem.tag)
                if elem.attrib:
                    elem.attrib = {ns_re.sub('', k): v
                                   for k, v in elem.attrib.items()}
        except Exception as exc:
            self.status_var.set(f"[WSP] {os.path.basename(wsp_path)}: {exc}")
            return

        wsp_dir = os.path.dirname(os.path.abspath(wsp_path))
        fcs_dir_hint = (self.fcs_dir
                       if self.fcs_dir and os.path.isdir(self.fcs_dir)
                       else None)

        resolved, unresolved = [], []
        for sample_elem in root.iter('Sample'):
            ds = sample_elem.find('DataSet')
            uri = ds.get('uri') if ds is not None else None
            sn  = sample_elem.find('SampleNode')
            if sn is None:
                continue
            # Try the uri as-is first, then the WSP's own folder, then
            # the editor's fcs_dir hint. _resolve_fcs_uri already covers
            # the first + fcs_dir; we add the WSP-folder fallback here.
            fcs_path = _resolve_fcs_uri(uri, fcs_dir_hint)
            if fcs_path is None and uri:
                from urllib.parse import unquote, urlparse
                raw = unquote(urlparse(uri).path) if uri.startswith('file:') else uri
                cand = os.path.join(wsp_dir, os.path.basename(raw))
                if os.path.isfile(cand):
                    fcs_path = cand
            if fcs_path is None:
                unresolved.append(sn.get('name') or '(unnamed)')
                continue

            # Per-sample gate extraction — scope the reader's walker to
            # just this <SampleNode>. Returns the same gate-dict format
            # as a full-document extract_gates() call.
            gates = reader.extract_gates(sample_node=sn)
            # Same name the FCS queue will assign (collision-safe per day), so
            # `_on_loaded` drains these gates onto the right sample.
            sample_name = self._sample_name_for(fcs_path)
            self._pending_sample_gates[sample_name] = gates
            resolved.append((sample_name, fcs_path, len(gates)))

        if not resolved:
            self.status_var.set(
                f"[WSP] {os.path.basename(wsp_path)}: no samples resolved "
                f"({len(unresolved)} unresolved)")
            return

        summary = (f"[WSP] {os.path.basename(wsp_path)}: queued "
                   f"{len(resolved)} sample(s), "
                   f"{sum(n for _, _, n in resolved)} gate(s)")
        if unresolved:
            summary += f" — couldn't locate FCS for: {', '.join(unresolved[:3])}"
            if len(unresolved) > 3:
                summary += f" (+{len(unresolved) - 3})"
        self.status_var.set(summary)
        self._queue_fcs_loads([p for _, p, _ in resolved])

    @staticmethod
    def _expand_dropped_paths(paths):
        """Resolve a list of dropped paths (files and/or folders) into the
        flat ``.fcs`` and ``.wsp`` files they contain.

        Folders are walked recursively, so dropping a single trial folder
        — or a parent folder holding several trial folders — surfaces every
        sample inside it. Trial grouping itself is handled downstream by
        ``workspace.derive_trial_name`` (the FCS's grandparent folder), so a
        multi-trial drop naturally lands each sample under its own group.

        Returns ``(fcs_paths, wsp_paths)``, each de-duplicated and sorted for
        a deterministic load order.
        """
        fcs, wsp = set(), set()

        def _add_file(fp):
            low = fp.lower()
            if low.endswith('.fcs'):
                fcs.add(fp)
            elif low.endswith('.wsp'):
                wsp.add(fp)

        for p in paths:
            p = (p or '').strip().strip('"').strip("'")
            if not p:
                continue
            if os.path.isdir(p):
                for dirpath, _dirs, files in os.walk(p):
                    for fn in files:
                        _add_file(os.path.join(dirpath, fn))
            elif os.path.isfile(p):
                _add_file(p)
        return sorted(fcs), sorted(wsp)

    def _import_dropped_paths(self, paths):
        """Import a drop of files and/or folders. Folders are expanded to
        the ``.fcs`` / ``.wsp`` files within (see ``_expand_dropped_paths``).

        Workspaces are ingested first so their gate trees are staged before
        any FCS load completes; ``_ingest_wsp`` also queues the FCS each
        workspace references. Remaining loose FCS are then queued — the
        ``_loading`` guard means a sample referenced by both a dropped .wsp
        and a dropped .fcs is loaded only once (the workspace wins, so its
        gates ride along)."""
        fcs_paths, wsp_paths = self._expand_dropped_paths(paths)
        if not fcs_paths and not wsp_paths:
            self.status_var.set(
                "Drop contained no .fcs or .wsp files.")
            return
        for wsp in wsp_paths:
            self._ingest_wsp(wsp)
        if fcs_paths:
            self._queue_fcs_loads(fcs_paths)

    def _sample_name_for(self, path):
        """Stable, collision-free key/display name for an FCS ``path``.

        Samples are keyed app-wide by name (``self._samples``, the per-sample
        gate stores, tree iids, the workspace, statistics…). Day-organised
        drops reuse filenames across days — e.g.
        ``Compensation Controls_…_008.fcs`` appears under Day 6 *and* Day 9 —
        so a bare basename would make the second file collide with the first
        and be silently skipped as 'already loaded'. We disambiguate a
        colliding basename with its day/trial (then a numeric counter as a last
        resort) and remember the path→name mapping, so repeat calls — and
        different callers like the ``.wsp`` ingest and the FCS queue — always
        resolve the same file to the same name."""
        # normcase so the registry key is case-insensitive on Windows — a .wsp
        # whose stored path case differs from the on-disk folder (day6\ vs
        # Day6\) must map to the SAME file, not load it twice.
        ap = os.path.normcase(os.path.abspath(path))
        cached = self._path_to_name.get(ap)
        if cached is not None:
            return cached
        base = os.path.basename(path).rsplit('.', 1)[0]
        name = base
        if name in self._name_to_path or name in self._samples:
            from .workspace import derive_trial_name
            trial = derive_trial_name(path)
            name = f'{base} [{trial}]'
            n = 2
            while name in self._name_to_path or name in self._samples:
                name = f'{base} [{trial}] ({n})'
                n += 1
        self._path_to_name[ap] = name
        self._name_to_path[name] = ap
        return name

    def _ensure_load_pool(self):
        """Spawn the fixed pool of FCS-load worker threads, once, lazily on the
        first enqueue. Daemon threads so they never block process exit. Touched
        only on the main thread, so the one-shot guard is race-free."""
        if self._load_pool_started:
            return
        for i in range(_LOAD_POOL_SIZE):
            t = threading.Thread(target=self._load_pool_worker,
                                 name=f'fcs-load-{i}', daemon=True)
            t.start()
            self._load_pool.append(t)
        self._load_pool_started = True

    def _load_pool_worker(self):
        """Pool worker: block on the queue, load one FCS at a time. ``None`` is
        the shutdown sentinel. ``_load_worker`` already runs the whole pipeline
        off-thread and posts ``_on_loaded`` / ``_on_load_error`` to the Tk
        thread via ``self.after``; here the ``finally`` posts a completion tick
        so progress advances even when a load raises."""
        while True:
            try:
                job = self._load_queue.get()
            except Exception:
                break
            if job is None or self._load_stop.is_set():
                break
            name, path = job
            try:
                self._load_worker(name, path)
            except Exception as exc:
                # _load_worker catches its own pipeline errors; this is a
                # backstop so an unexpected throw can't permanently kill a pool
                # thread (which would shrink the pool and stall the queue).
                print(f"[load] pool worker error for {name}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
            finally:
                # Tally on the Tk thread, not here: two pool workers writing
                # `self._load_done += 1` concurrently would race (load-add-store
                # spans several bytecodes; the GIL can interleave them and drop
                # an increment, so the bar would never reach N/N). Posting the
                # tick keeps `_load_total`/`_load_done` single-writer.
                try:
                    self.after(0, self._mark_one_done)
                except Exception:
                    # Window/interpreter gone — nothing left to update.
                    break

    def _mark_one_done(self):
        """Tk-thread: record that one load finished (success or error) and
        refresh the bar. Counter writes live only here + on enqueue, both on the
        main thread, so no lock is needed."""
        self._load_done += 1
        self._update_progress_bar()

    def _update_progress_bar(self):
        """Reflect the load counters in the bar. Runs on the Tk thread (always
        reached via ``self.after``). Hidden when nothing is queued; shown and
        sized to ``_load_total`` otherwise; schedules a brief auto-hide once the
        run drains."""
        try:
            total, done = self._load_total, self._load_done
            if total <= 0:
                self._load_progress_frame.grid_remove()
                return
            self._load_progress_frame.grid()
            self.progress_bar.configure(maximum=total)
            self._load_progress_var.set(done)
            self._load_progress_lbl_var.set(f'{done}/{total} loaded')
            if done >= total:
                # Linger briefly at N/N, then hide+reset (re-checked in
                # _finish_progress so a mid-delay drop keeps the bar alive).
                self.after(800, self._finish_progress)
        except Exception:
            pass

    def _finish_progress(self):
        """Hide + reset the progress bar, but only if the run is still complete
        — files dropped during the 800 ms linger extend ``_load_total``, in
        which case we leave the bar running."""
        try:
            if self._load_total > 0 and self._load_done >= self._load_total:
                self._load_total = 0
                self._load_done = 0
                self._load_progress_var.set(0)
                self._load_progress_lbl_var.set('')
                self._load_progress_frame.grid_remove()
        except Exception:
            pass

    def _queue_fcs_loads(self, paths):
        """Queue a list of FCS paths for background loading. Shared by
        the Add-FCS button, the OS-clipboard paste, and the file-drop
        target. Skips non-existent / non-.fcs / already-loaded entries
        with a brief status note. Loads run through a bounded worker pool
        (`_LOAD_POOL_SIZE` at a time) so a big folder drop can't exhaust
        memory; a progress bar tracks completion."""
        self._ensure_load_pool()
        added = 0
        skipped = []
        for p in paths:
            p = (p or '').strip().strip('"').strip("'")
            if not p:
                continue
            if not os.path.isfile(p):
                skipped.append(f'{os.path.basename(p)}(missing)')
                continue
            if not p.lower().endswith('.fcs'):
                skipped.append(f'{os.path.basename(p)}(not .fcs)')
                continue
            name = self._sample_name_for(p)
            if name in self._samples:
                skipped.append(f'{name}(already loaded)')
                continue
            if name in self._loading:
                # Already queued (e.g. a .wsp ingest queued it just now, or
                # the same file appears twice in a folder drop). Don't queue
                # a second job for the same sample.
                continue
            self._loading.add(name)
            self._sample_lb_insert_loading(name)
            # Hand off to the bounded pool instead of spawning a thread per
            # file. _load_total only ever grows here (never reset mid-run), so
            # files dropped while a run is in flight extend the bar (e.g.
            # 2/5 → 2/8) rather than restarting it.
            self._load_queue.put((name, p))
            self._load_total += 1
            added += 1
        if added:
            self._update_progress_bar()
        if added or skipped:
            note = f"Queued {added}."
            if skipped:
                note += f"  Skipped: {', '.join(skipped[:4])}"
                if len(skipped) > 4:
                    note += f" (+{len(skipped) - 4} more)"
            self.status_var.set(note)

    def _sample_lb_insert_loading(self, name):
        # Insert a placeholder sample row into the merged tree; it'll
        # be replaced with the proper '■ <name>' row once _on_loaded
        # fires (or removed by _on_load_error on failure).
        try:
            self.gate_tv.insert(
                '', 'end', iid=self._sample_iid(name),
                text=f'⏳ {name}', values=('',),
                tags=('loading',))
        except Exception:
            pass

    def _load_worker(self, name, path):
        try:
            sys.path.insert(0, BASE)
            from .cli import parse_labels
            from .pipeline import FlowSample
            s = FlowSample(path)
            s.run_qc()
            s.auto_compensate()
            s.apply_transform()
            if self.labels_str:
                lbl = parse_labels(self.labels_str)
                if lbl:
                    s.set_labels(lbl)
            self.after(0, lambda: self._on_loaded(name, s))
        except Exception as exc:
            # Bind exc as a default arg — `except … as exc` deletes the name
            # at block exit, so the bare lambda would NameError when fired.
            self.after(0, lambda e=exc: self._on_load_error(name, e))

    def _on_loaded(self, name, sample):
        self._loading.discard(name)
        # Propagate downsample to this freshly-loaded sample BEFORE we
        # publish it to self._samples so the first replot already sees
        # the trimmed size.
        if (getattr(self, 'ds_propagate_var', None) is not None
                and self.ds_propagate_var.get()
                and self._samples):
            floor = self._smallest_loaded_sample_size()
            if floor is not None and floor > 0 and len(sample.data) > floor:
                sample.data = sample.data.sample(
                    floor, random_state=42).reset_index(drop=True)
        self._samples[name] = sample
        if name not in self._sample_order:
            self._sample_order.append(name)
        # Keep the path⇄name registry in sync for entry points that bypass
        # `_sample_name_for` (e.g. processed-CSV load), so later loads still
        # see this name as taken and disambiguate around it.
        ap = os.path.normcase(os.path.abspath(getattr(sample, 'path', '') or ''))
        if ap:
            self._path_to_name.setdefault(ap, name)
        self._name_to_path.setdefault(name, ap or name)
        # Color is assigned lazily — only when a sample is actually displayed
        # (see `_color_for`). Until then the tree row stays neutral so loading
        # many trials doesn't paint a rainbow of undisplayed samples.

        # Record the sample's trial (grandparent folder of its FCS path) so the
        # tree can group it and the workspace can label its origin.
        from .workspace import derive_trial_name
        trial = derive_trial_name(getattr(sample, 'path', None))
        # A restored session may pin this sample to a manually-regrouped day /
        # Comps-Samples side (and carries its gates). It's keyed by FILE PATH,
        # not name, so collision-disambiguated names can't mismatch on reload.
        spath = getattr(sample, 'path', '') or ''
        pkey = os.path.normcase(os.path.abspath(spath)) if spath else None
        meta = self._pending_sample_meta.pop(pkey, None) if pkey else None
        if meta:
            if meta.get('trial'):
                trial = meta['trial']
            if 'is_comp' in meta:
                self._sample_is_comp[name] = bool(meta['is_comp'])
        self._sample_trial[name] = trial
        if trial not in self._trial_order:
            self._trial_order.append(trial)

        # Initialise per-sample gate state.
        self._sample_gates.setdefault(name, {})
        self._sample_gate_seq.setdefault(name, 0)
        self._sample_gate_order.setdefault(name, [])

        # Drain pending gates: a restored session bundles them in `meta` (keyed
        # by path, above); a .wsp ingest stages them in `_pending_sample_gates`
        # by name. We rebind `_gates` / `_gate_id_order` to this sample's
        # storage via `_set_active_sample`, populate, then leave it active iff
        # it was the first sample loaded.
        if meta is not None:
            pending = meta.get('gates') or None
        else:
            pending = self._pending_sample_gates.pop(name, None)
        if pending:
            saved_active = self._active_sample
            self._set_active_sample(name)
            old_to_new = {}
            prev_suspend = self._suspend_undo
            self._suspend_undo = True       # bulk load isn't an undo step
            try:
                for raw in pending:
                    g = dict(raw)
                    src_id = g.pop('_import_id', None) or g.pop('id', None)
                    parent = g.get('parent_id')
                    if parent is not None:
                        g['parent_id'] = old_to_new.get(parent)
                    # Imported gates start DISABLED so a freshly-loaded sample
                    # (only the first is displayed) isn't a wall of active
                    # toggles. WSP gates carry no 'enabled' → default off; a
                    # restored session's gates carry their saved flag → kept.
                    g.setdefault('enabled', False)
                    gid = self._add_gate(g)
                    if src_id is not None:
                        old_to_new[src_id] = gid
            finally:
                self._suspend_undo = prev_suspend
            if saved_active is not None and saved_active != name:
                # Restore the previously-active sample; this sample's
                # gates are now persisted in `_sample_gates[name]`.
                self._set_active_sample(saved_active)
        # Plot inclusion: enable ONLY the very first sample loaded — the
        # user gets an immediate render to confirm the load worked.
        # Subsequent loads start unchecked so opening a session with many
        # samples doesn't cascade-render N overlays on every Add.
        was_first = (len(self._samples) == 1)
        self._sample_plot_enabled.setdefault(name, was_first)

        # First sample populates the channel choices and becomes active.
        if len(self._samples) == 1:
            self._channels       = list(sample.data.columns)
            self._channel_labels = dict(sample.channel_labels)
            # Loader applies logicle to fluor channels; everything else is
            # left linear. Seeds the per-channel transform editor.
            fluor = set(getattr(sample, 'fluor_channels', []) or [])
            self._channel_transform = {
                c: ('logicle' if c in fluor else 'linear')
                for c in self._channels}
            self._populate_channel_combos()
        if self._active_sample is None:
            self._set_active_sample(name)

        # Rebuild the tree to swap the ⏳ placeholder for the real row.
        self._refresh_gate_list()
        try:
            self.gate_tv.selection_set(self._sample_iid(name))
            self.gate_tv.see(self._sample_iid(name))
        except Exception:
            pass

        base_msg = (
            f"{len(self._samples)} sample(s) loaded "
            f"(latest: {name}, {len(sample.data):,} events). "
            f"Double-click the plot to add a gate.")
        # Flag a non-common fluor panel across loaded samples — cross-
        # sample stats tie by antibody label, so a mismatched panel
        # just means some labels won't be shared.
        if self._fluor_panel_warning():
            base_msg += "  [!] samples differ in fluor panel — see Statistics."
        self.status_var.set(base_msg)
        # Provenance: record the load (QC + auto-compensation + transform ran
        # in the loader). Capture the data identity for reproducibility.
        comp = getattr(sample, 'compensation_source', None) or 'auto/$SPILL'
        self._audit('sample.load', sample=name,
                    path=getattr(sample, 'path', '') or '',
                    n_events=int(len(sample.data)),
                    channels=int(sample.data.shape[1]),
                    trial=self._sample_trial.get(name, ''),
                    compensation=comp)
        self._schedule_replot(0)

    # ── Active sample / per-sample gate switching ────────────────────────

    def _set_active_sample(self, name):
        """Switch the gate tree to `name`'s gate set. Re-binds the
        self._gates / _gate_id_seq / _gate_id_order shortcuts so existing
        code paths (which read/write those directly) stay simple. The
        bottom strip is gone — selection lives in the merged tree
        (`_on_tree_select`)."""
        if name not in self._samples:
            self._active_sample   = None
            self._gates           = {}
            self._gate_id_seq     = 0
            self._gate_id_order   = []
            return
        # Persist the counter of the sample we're about to leave.
        if self._active_sample is not None:
            self._sample_gate_seq[self._active_sample] = self._gate_id_seq
        self._active_sample = name
        self._gates         = self._sample_gates.setdefault(name, {})
        self._gate_id_seq   = self._sample_gate_seq.setdefault(name, 0)
        self._gate_id_order = self._sample_gate_order.setdefault(name, [])

    # ── Undo / redo ──────────────────────────────────────────────────────
    #
    # The undo history snapshots the gate-related state (per-sample gates +
    # order + id sequences + cluster phenotype labels + the quadrant-set
    # counter). Every mutating gesture calls _checkpoint() *before* changing
    # anything; calls within the same Tk event coalesce into one undo step.

    def _gate_state_snapshot(self):
        import copy
        seq = dict(self._sample_gate_seq)
        active = self._active_sample
        if active is not None:
            seq[active] = max(seq.get(active, 0), self._gate_id_seq)
        ws = getattr(self, '_workspace_panel', None)
        return {
            'gates':          copy.deepcopy(self._sample_gates),
            'order':          copy.deepcopy(self._sample_gate_order),
            'seq':            seq,
            'cluster_labels': copy.deepcopy(self._cluster_labels),
            'quad_seq':       getattr(self, '_quad_set_seq', 0),
            'workspace':      ws.model.to_dict() if ws is not None else None,
        }

    def _restore_gate_state(self, snap):
        import copy
        self._sample_gates      = copy.deepcopy(snap['gates'])
        self._sample_gate_order = copy.deepcopy(snap['order'])
        self._sample_gate_seq   = dict(snap['seq'])
        self._cluster_labels    = copy.deepcopy(snap['cluster_labels'])
        self._quad_set_seq      = snap.get('quad_seq', getattr(
            self, '_quad_set_seq', 0))
        # Rebind the active-sample shortcuts to the restored containers.
        active = self._active_sample
        if active in self._sample_gates:
            self._gates         = self._sample_gates[active]
            self._gate_id_order = self._sample_gate_order.setdefault(active, [])
            self._gate_id_seq   = self._sample_gate_seq.get(active, 0)
        else:
            self._gates = {}
            self._gate_id_order = []
            self._gate_id_seq = 0
        self._refresh_gate_list()
        self._schedule_replot(0)
        # Widen undo to the Pipeline Workspace: restore its model too (same
        # Undo button reverts workspace add/remove/group/comp/fmo/clear).
        ws = getattr(self, '_workspace_panel', None)
        ws_snap = snap.get('workspace')
        if ws is not None and ws_snap is not None:
            try:
                ws.restore_model(ws_snap)
            except Exception:
                pass

    def _checkpoint(self):
        """Record a pre-mutation undo checkpoint. Call BEFORE mutating gate
        state. No-op while suspended (bulk loads) or when one was already
        taken this Tk event (coalesces a multi-step gesture into one undo)."""
        if self._suspend_undo or self._undo_pending:
            return
        self._undo_pending = True
        self._undo_stack.append(self._gate_state_snapshot())
        if len(self._undo_stack) > self._UNDO_MAX:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        try:
            self.after_idle(self._clear_undo_pending)
        except Exception:
            self._undo_pending = False

    def _clear_undo_pending(self):
        self._undo_pending = False

    # ── Provenance / audit trail ─────────────────────────────────────────
    def _audit(self, action, **details):
        """Append an operation to the session's audit trail (stamped with the
        wall-clock time) and live-refresh the History window if it's open.
        Cheap and best-effort — a logging failure must never break the
        operation being logged."""
        try:
            from datetime import datetime
            ts = datetime.now().isoformat(timespec='seconds')
            self._audit_log.record(action, time=ts, details=details)
            win = getattr(self, '_audit_window', None)
            if win is not None and win.winfo_exists():
                win.refresh()
        except Exception as exc:
            print(f"[audit] {type(exc).__name__}: {exc}", flush=True)

    def _show_audit_window(self):
        """Open (or focus) the provenance / audit-trail viewer."""
        win = getattr(self, '_audit_window', None)
        if win is not None and win.winfo_exists():
            win.refresh()
            win.lift()
            win.focus_set()
            return
        self._audit_window = AuditWindow(self, self._audit_log)

    def _focus_in_text(self):
        """True when a text-entry widget has focus, so Ctrl+Z/Y should edit
        the text rather than the gate history."""
        try:
            w = self.focus_get()
            return bool(w) and w.winfo_class() in ('TEntry', 'Entry',
                                                   'TCombobox', 'Text')
        except Exception:
            return False

    def _undo(self, event=None):
        if event is not None and self._focus_in_text():
            return
        if not self._undo_stack:
            self.status_var.set("Nothing to undo.")
            return
        self._redo_stack.append(self._gate_state_snapshot())
        self._restore_gate_state(self._undo_stack.pop())
        self.status_var.set(
            f"Undo. ({len(self._undo_stack)} more, {len(self._redo_stack)} redo)")

    def _redo(self, event=None):
        if event is not None and self._focus_in_text():
            return
        if not self._redo_stack:
            self.status_var.set("Nothing to redo.")
            return
        self._undo_stack.append(self._gate_state_snapshot())
        self._restore_gate_state(self._redo_stack.pop())
        self.status_var.set(
            f"Redo. ({len(self._redo_stack)} more)")

    # ── Clusters as selectable populations (#43) ─────────────────────────
    #
    # The pipeline writes an integer 'cluster' column into FlowSample.data
    # (pipeline.FlowSample.cluster). The editor doesn't run clustering, but
    # when a clustered sample is loaded we can surface each cluster as a
    # population: a root gate of kind 'cluster' whose mask is
    # df['cluster'] == cluster_id. From there the existing machinery —
    # tree toggle, highlight overlay, filter, and the stats table — treats
    # it like any other population. Phenotype names live in
    # self._cluster_labels[sample][cluster_id] and double as the gate name.

    def _next_gate_id_for(self, name):
        """Allocate a fresh gate id for `name`'s gate set, keeping the
        per-sample sequence (and the active-sample shortcut) in sync."""
        seq = self._sample_gate_seq.get(name, 0) + 1
        self._sample_gate_seq[name] = seq
        if name == self._active_sample:
            self._gate_id_seq = seq
        return f'g{seq}'

    def _sample_cluster_ids(self, name):
        """Sorted unique cluster ids in `name`'s data, or [] when the
        sample isn't clustered."""
        s = self._samples.get(name)
        if s is None or getattr(s, 'data', None) is None:
            return []
        df = s.data
        if 'cluster' not in df.columns:
            return []
        vals = df['cluster'].dropna().unique()
        out = []
        for v in vals:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return sorted(set(out))

    def _cluster_display_name(self, name, cid):
        """Phenotype label for one cluster, falling back to 'Cluster N'."""
        lbls = self._cluster_labels.get(name) or {}
        nm = lbls.get(cid)
        if nm is None:
            nm = lbls.get(str(cid))
        return nm or f'Cluster {cid}'

    # Label columns that can be imported as populations, with a friendly
    # name and the sentinel value that means "unassigned" (skipped).
    LABEL_COLUMNS = {
        'cluster':      ('clusters', -1),
        'flowsom_meta': ('FlowSOM metaclusters', -1),
        'cell_cycle':   ('cell-cycle phases', 'NA'),
    }

    def _label_columns_present(self):
        """Known label columns that at least one loaded sample carries."""
        present = []
        for col in self.LABEL_COLUMNS:
            for s in self._samples.values():
                df = getattr(s, 'data', None)
                if df is not None and col in df.columns:
                    present.append(col)
                    break
        return present

    def _sample_label_values(self, name, col):
        """Sorted distinct values of `col` in `name`'s data, minus the
        unassigned sentinel. [] when the column is absent."""
        s = self._samples.get(name)
        if s is None or getattr(s, 'data', None) is None:
            return []
        df = s.data
        if col not in df.columns:
            return []
        skip = self.LABEL_COLUMNS.get(col, (None, None))[1]
        vals = [v for v in df[col].dropna().unique() if v != skip]
        try:
            return sorted(vals)
        except TypeError:
            return sorted(vals, key=str)

    def _open_clusters_menu(self):
        """Popup at the pointer: import / annotate any label column present
        (clusters, FlowSOM metaclusters, cell-cycle phases) as populations."""
        menu = tk.Menu(self, tearoff=0)
        present = self._label_columns_present()
        if not present:
            menu.add_command(
                label="No cluster / FlowSOM columns loaded", state='disabled')
        for col in present:
            disp = self.LABEL_COLUMNS[col][0]
            menu.add_command(
                label=f"Import {disp} as populations",
                command=lambda c=col: self._import_populations(c))
            menu.add_command(
                label=f"Annotate {disp}…",
                command=lambda c=col: self._annotate_populations(c))
            menu.add_separator()
        try:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            menu.grab_release()

    def _import_populations(self, col):
        """Import one label column as selectable populations. 'cluster' uses
        the legacy cluster-gate path (kept for session back-compat); every
        other column becomes 'category' gates via the generic importer."""
        if col == 'cluster':
            self._import_clusters()
        else:
            self._import_label_populations(col)

    def _import_label_populations(self, col):
        """Generic: create a 'category' population per distinct value of
        `col` across every loaded sample that carries it. Idempotent;
        populations start disabled. Names live on the gate."""
        from .pipeline import GATE_PALETTE
        self._checkpoint()
        total_new = 0
        n_samples = 0
        for name in self._sample_order:
            vals = self._sample_label_values(name, col)
            if not vals:
                continue
            n_samples += 1
            gates = self._sample_gates.setdefault(name, {})
            order = self._sample_gate_order.setdefault(name, [])
            existing = {g.get('value') for g in gates.values()
                        if g.get('kind') == 'category'
                        and g.get('channel') == col}
            for i, v in enumerate(vals):
                if v in existing:
                    continue
                gid = self._next_gate_id_for(name)
                gates[gid] = {
                    'kind': 'category', 'channel': col, 'value': v,
                    'name': f'{col} {v}', 'parent_id': None,
                    'color': GATE_PALETTE[i % len(GATE_PALETTE)],
                    'enabled': False,
                }
                order.append(gid)
                total_new += 1
        disp = self.LABEL_COLUMNS.get(col, (col, None))[0]
        if n_samples == 0:
            self.status_var.set(f"No samples carry a '{col}' column.")
            return
        self._refresh_gate_list()
        self.status_var.set(
            f"Imported {total_new} new {disp} population(s) across "
            f"{n_samples} sample(s). Toggle them in the tree.")

    def _annotate_populations(self, col):
        """Generic rename dialog for one label column's populations on the
        active sample. 'cluster' delegates to the legacy annotator; others
        edit the matching 'category' gates' names in place."""
        if col == 'cluster':
            self._annotate_clusters()
            return
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        vals = self._sample_label_values(name, col)
        if not vals:
            self.status_var.set(f"'{name}' has no '{col}' column to annotate.")
            return
        gates = self._sample_gates.get(name, {})

        def _gate_for(v):
            for g in gates.values():
                if (g.get('kind') == 'category' and g.get('channel') == col
                        and g.get('value') == v):
                    return g
            return None

        dlg = tk.Toplevel(self)
        disp = self.LABEL_COLUMNS.get(col, (col, None))[0]
        dlg.title(f"Annotate {disp} — {name}")
        dlg.transient(self)
        dlg.grab_set()
        dlg.geometry("360x440")

        ttk.Label(dlg, text=f"Names for '{name}':",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 6))
        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=(0, 6))
        cv = tk.Canvas(holder, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))

        entries = {}
        for v in vals:
            row = ttk.Frame(inner)
            row.pack(side='top', fill='x', pady=1)
            ttk.Label(row, text=f"{col} {v}", width=14).pack(side='left')
            g = _gate_for(v)
            cur = (g.get('name') if g else None) or f'{col} {v}'
            var = tk.StringVar(value=cur)
            ttk.Entry(row, textvariable=var, width=22).pack(
                side='left', fill='x', expand=True)
            entries[v] = var

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)

        def do_apply():
            self._checkpoint()
            for v, var in entries.items():
                g = _gate_for(v)
                if g is not None:
                    g['name'] = var.get().strip() or f'{col} {v}'
            dlg.destroy()
            self._refresh_gate_list()
            self._schedule_replot(0)
            self.status_var.set(f"Updated {disp} names for '{name}'.")

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Apply", command=do_apply).pack(
            side='right', padx=(0, 6))

    def _import_clusters(self):
        """Create a root cluster-gate per clustering label, for every
        loaded sample that carries a 'cluster' column. Existing cluster
        gates are kept (idempotent — re-running only adds new ids). Imported
        gates start disabled so the plot isn't flooded; the user toggles
        them like any population."""
        from .pipeline import GATE_PALETTE
        self._checkpoint()
        total_new = 0
        clustered = 0
        for name in self._sample_order:
            ids = self._sample_cluster_ids(name)
            if not ids:
                continue
            clustered += 1
            gates = self._sample_gates.setdefault(name, {})
            order = self._sample_gate_order.setdefault(name, [])
            existing = {g.get('cluster_id') for g in gates.values()
                        if g.get('kind') == 'cluster'}
            for cid in ids:
                if cid in existing:
                    continue
                gid = self._next_gate_id_for(name)
                gates[gid] = {
                    'kind': 'cluster',
                    'channel': 'cluster',
                    'cluster_id': cid,
                    'parent_id': None,
                    'name': self._cluster_display_name(name, cid),
                    'color': GATE_PALETTE[cid % len(GATE_PALETTE)],
                    'enabled': False,
                }
                order.append(gid)
                total_new += 1
        if clustered == 0:
            self.status_var.set(
                "No clustered samples loaded. Run the pipeline with "
                "clustering, or load a session that has a 'cluster' column.")
            return
        self._refresh_gate_list()
        self.status_var.set(
            f"Imported {total_new} new cluster population(s) across "
            f"{clustered} clustered sample(s). Toggle them in the tree.")

    def _annotate_clusters(self):
        """Dialog to name the active sample's clusters. Pre-fills existing
        phenotype names; on Apply, stores them in self._cluster_labels and
        renames any matching cluster gates, then refreshes the tree/plot."""
        name = self._active_sample
        if name is None:
            self.status_var.set("Select a sample first.")
            return
        ids = self._sample_cluster_ids(name)
        if not ids:
            self.status_var.set(
                f"'{name}' has no 'cluster' column to annotate.")
            return

        dlg = tk.Toplevel(self)
        dlg.title(f"Annotate clusters — {name}")
        dlg.transient(self)
        dlg.grab_set()
        dlg.geometry("360x440")
        dlg.minsize(300, 240)

        ttk.Label(dlg, text=f"Phenotype names for '{name}':",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 6))

        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=(0, 6))
        cv = tk.Canvas(holder, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))

        entries = {}
        for cid in ids:
            row = ttk.Frame(inner)
            row.pack(side='top', fill='x', pady=1)
            ttk.Label(row, text=f"Cluster {cid}", width=12).pack(side='left')
            var = tk.StringVar(value=self._cluster_display_name(name, cid))
            ttk.Entry(row, textvariable=var, width=24).pack(
                side='left', fill='x', expand=True)
            entries[cid] = var

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)

        def do_apply():
            self._checkpoint()
            lbls = self._cluster_labels.setdefault(name, {})
            gates = self._sample_gates.get(name, {})
            for cid, var in entries.items():
                txt = var.get().strip()
                if txt:
                    lbls[cid] = txt
                else:
                    lbls.pop(cid, None)
                for g in gates.values():
                    if (g.get('kind') == 'cluster'
                            and g.get('cluster_id') == cid):
                        g['name'] = txt or f'Cluster {cid}'
            dlg.destroy()
            self._refresh_gate_list()
            self._schedule_replot(0)
            self.status_var.set(f"Updated cluster names for '{name}'.")

        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Apply",
                   command=do_apply).pack(side='right', padx=(0, 6))

    # ── Cell cycle (#cell-cycle) ─────────────────────────────────────────
    #
    # Runs FlowSample.cell_cycle (DNA-content G1/S/G2M model) on the active
    # (or all) sample(s), then surfaces each phase as a selectable
    # population via the 'category' gate kind — the same machinery clusters
    # use. A result window shows the DNA histogram + phase percentages.

    PHASE_COLORS = {'sub-G1': '#9a6324', 'G1': '#4363d8', 'S': '#3cb44b',
                    'G2M': '#e6194b', '>G2M': '#911eb4'}

    def _open_cell_cycle_dialog(self):
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Load and select a sample first.")
            return
        from .pipeline import find_dna_channel
        s = self._samples[name]
        default = find_dna_channel(s)

        dlg = tk.Toplevel(self)
        dlg.title(f"Cell cycle — {name}")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        ttk.Label(dlg, text="DNA-content channel:").grid(
            row=0, column=0, sticky='w', padx=10, pady=(12, 4))
        combo = ttk.Combobox(dlg, width=28, state='readonly',
                             values=[self._fmt_channel(c) for c in self._channels])
        combo.grid(row=0, column=1, padx=10, pady=(12, 4))
        if default:
            combo.set(self._fmt_channel(default))
        elif self._channels:
            combo.set(self._fmt_channel(self._channels[0]))
        if not default:
            ttk.Label(dlg, text="(no DNA dye auto-detected — pick one)",
                      foreground='grey').grid(
                row=1, column=0, columnspan=2, sticky='w', padx=10)

        all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg, text="Run on all loaded samples",
                        variable=all_var).grid(
            row=2, column=0, columnspan=2, sticky='w', padx=10, pady=(6, 4))

        btns = ttk.Frame(dlg)
        btns.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10,
                  pady=(6, 10))

        def do_run():
            col = self._resolve_channel(combo.get())
            dlg.destroy()
            if col:
                self._run_cell_cycle(col, all_var.get())

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Run", command=do_run).pack(
            side='right', padx=(0, 6))

    def _run_cell_cycle(self, dna_channel, all_samples):
        targets = (list(self._sample_order) if all_samples
                   else [self._active_sample])
        done = []
        for name in targets:
            s = self._samples.get(name)
            if s is None:
                continue
            try:
                s.cell_cycle(dna_channel=dna_channel)
            except Exception as exc:
                self.status_var.set(f"Cell cycle failed for {name}: {exc}")
                continue
            res = getattr(s, 'cell_cycle_result', None)
            if res and res.get('ok'):
                self._import_cell_cycle(name)
                done.append(name)
        self._refresh_gate_list()
        if not done:
            self.status_var.set(
                "Cell cycle: no usable DNA peaks found "
                f"on '{dna_channel}'.")
            return
        self.status_var.set(
            f"Cell cycle done for {len(done)} sample(s) on '{dna_channel}'. "
            "Phases added as populations; toggle them in the tree.")
        active = self._active_sample
        if active in done:
            try:
                CellCycleWindow(self, active)
            except Exception as exc:
                self.status_var.set(f"Cell-cycle plot failed: {exc}")

    def _import_cell_cycle(self, name):
        """Create a category population per cell-cycle phase present in
        `name`'s data. Idempotent (re-running only adds new phases).
        Populations start disabled, like imported clusters."""
        from .pipeline import CELL_CYCLE_PHASES, GATE_PALETTE
        s = self._samples.get(name)
        if s is None or 'cell_cycle' not in s.data.columns:
            return
        self._checkpoint()
        present = set(s.data['cell_cycle'].unique())
        phases = [p for p in CELL_CYCLE_PHASES if p in present]
        gates = self._sample_gates.setdefault(name, {})
        order = self._sample_gate_order.setdefault(name, [])
        existing = {g.get('value') for g in gates.values()
                    if g.get('kind') == 'category'
                    and g.get('channel') == 'cell_cycle'}
        for i, ph in enumerate(phases):
            if ph in existing:
                continue
            gid = self._next_gate_id_for(name)
            gates[gid] = {
                'kind': 'category',
                'channel': 'cell_cycle',
                'value': ph,
                'name': ph,
                'parent_id': None,
                'color': self.PHASE_COLORS.get(
                    ph, GATE_PALETTE[i % len(GATE_PALETTE)]),
                'enabled': False,
            }
            order.append(gid)

    def _open_copy_gates_dialog(self):
        """Pop a modal dialog: pick which loaded samples should receive a
        copy of the active sample's gate tree. Includes Select all /
        Deselect all helpers. Copies APPEND (don't overwrite) so the
        target sample's existing gates are preserved."""
        if self._active_sample is None or not self._gates:
            self.status_var.set("Active sample has no gates to copy.")
            return
        others = [n for n in self._sample_order
                  if n in self._samples and n != self._active_sample]
        if not others:
            self.status_var.set("No other samples loaded to copy into.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Copy gates to samples")
        dlg.transient(self)
        dlg.grab_set()
        dlg.geometry("420x440")
        dlg.minsize(360, 280)

        ttk.Label(dlg,
                  text=f"Copy {len(self._gates)} gate(s) from "
                       f"'{self._active_sample}' to:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 6))

        # Scrollable list of checkboxes (one per other sample).
        canvas_holder = ttk.Frame(dlg)
        canvas_holder.pack(side='top', fill='both', expand=True,
                           padx=10, pady=(0, 6))
        cv = tk.Canvas(canvas_holder, highlightthickness=0)
        sb = ttk.Scrollbar(canvas_holder, orient='vertical',
                           command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')

        def _on_inner_configure(_):
            cv.configure(scrollregion=cv.bbox('all'))
        inner.bind('<Configure>', _on_inner_configure)

        cb_vars = {}
        for name in others:
            var = tk.BooleanVar(value=False)
            cb_vars[name] = var
            ttk.Checkbutton(inner, text=name, variable=var).pack(
                side='top', anchor='w', padx=2, pady=1)

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)

        def select_all():
            for v in cb_vars.values():
                v.set(True)

        def select_none():
            for v in cb_vars.values():
                v.set(False)

        ttk.Button(btns, text="Select all",
                   command=select_all).pack(side='left')
        ttk.Button(btns, text="Deselect all",
                   command=select_none).pack(side='left', padx=(4, 0))

        def do_copy():
            targets = [n for n, v in cb_vars.items() if v.get()]
            dlg.destroy()
            if targets:
                self._copy_gates_to(targets, append=True)

        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Copy",
                   command=do_copy).pack(side='right', padx=(0, 4))

        dlg.bind('<Escape>', lambda *_: dlg.destroy())

    def _copy_gates_to(self, target_names, append=True):
        """Copy the active sample's gates into each target sample.

        When `append=True` (default), target's existing gates are kept
        and the copied tree is added on top (each copied gate gets a
        fresh id; parent_id pointers within the copied subset are
        remapped). When False, the target's gates are replaced.
        """
        import copy as _copy
        src = self._active_sample
        if src is None or src not in self._sample_gates:
            return
        src_gates = self._sample_gates[src]
        src_order = list(self._sample_gate_order.get(src, []))
        if not src_order:
            return
        self._checkpoint()

        copied_to = 0
        for target in target_names:
            if target == src or target not in self._samples:
                continue
            if not append:
                self._sample_gates[target]     = {}
                self._sample_gate_order[target] = []
                self._sample_gate_seq[target]   = 0
            # Ensure containers exist (target may have been gate-less).
            self._sample_gates.setdefault(target, {})
            self._sample_gate_order.setdefault(target, [])
            self._sample_gate_seq.setdefault(target, 0)

            # If the target IS the currently-active sample, mutate the
            # already-bound containers (so self._gates etc. stay in sync).
            if target == self._active_sample:
                target_gates = self._gates
                target_order = self._gate_id_order
            else:
                target_gates = self._sample_gates[target]
                target_order = self._sample_gate_order[target]

            old_to_new = {}
            for old_id in src_order:
                g = _copy.deepcopy(src_gates[old_id])
                self._sample_gate_seq[target] += 1
                new_id = f'g{self._sample_gate_seq[target]}'
                if target == self._active_sample:
                    self._gate_id_seq = self._sample_gate_seq[target]
                pid = g.get('parent_id')
                g['parent_id'] = old_to_new.get(pid) if pid else None
                target_gates[new_id] = g
                target_order.append(new_id)
                old_to_new[old_id] = new_id
            copied_to += 1

        self.status_var.set(
            f"Copied {len(src_order)} gate(s) from '{src}' to "
            f"{copied_to} sample(s) (appended; existing gates preserved).")
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _on_load_error(self, name, exc):
        self._loading.discard(name)
        try:
            self.gate_tv.delete(self._sample_iid(name))
        except Exception:
            pass
        self.status_var.set(f"Failed to load {name}: {exc}")

    def _remove_selected(self):
        """Remove the selected SAMPLE(s), or — if any TRIAL row is selected —
        every sample (and its gates) in those trials. Gate-row selections are
        ignored (use Clear gate for those)."""
        sel = self.gate_tv.selection()
        if not sel:
            return
        parsed = [self._parse_iid(s) for s in sel]
        trials  = [p[1] for p in parsed if p and p[0] == 'trial']
        samples = [p[1] for p in parsed if p and p[0] == 'sample']
        if trials:
            self._remove_trials(trials)
            return
        if not samples:
            self.status_var.set("Select a sample or trial row to remove "
                                "(use Clear gate for gates).")
            return
        n = self._remove_samples(samples)
        self.status_var.set(f"Removed {n} sample(s).")

    def _remove_trials(self, trials):
        """Remove every sample (+ gates) belonging to ``trials``. Confirmed,
        because — like single-sample Remove — it isn't on the undo stack."""
        members = []
        for t in trials:
            members.extend(n for n in self._trial_members(t) if n not in members)
        if not members:
            # Empty trial header(s) — just forget them.
            for t in trials:
                if t in self._trial_order:
                    self._trial_order.remove(t)
            self._refresh_gate_list()
            return
        label = (f"trial '{trials[0]}'" if len(trials) == 1
                 else f"{len(trials)} trials")
        if not messagebox.askyesno(
                "Remove trial",
                f"Remove {label} — {len(members)} sample(s) and all their "
                f"gates?\nThis can't be undone.",
                parent=self):
            return
        self._remove_samples(members)
        self.status_var.set(f"Removed {label} ({len(members)} sample(s)).")

    def _remove_samples(self, names):
        """Tear down a list of samples completely: FlowSample, gate tree,
        colours, plot/display state, cluster labels, trial membership. Rebinds
        the active sample if it was removed, drops now-empty trials, and
        refreshes. Returns the count removed. (Not undoable — samples hold
        large frames; matches the historic single-sample Remove.)"""
        removed = 0
        for name in list(names):
            if name not in self._samples:
                continue
            del self._samples[name]
            self._sample_colors.pop(name, None)
            if name in self._sample_order:
                self._sample_order.remove(name)
            self._sample_gates.pop(name, None)
            self._sample_gate_seq.pop(name, None)
            self._sample_gate_order.pop(name, None)
            self._sample_plot_enabled.pop(name, None)
            self._cluster_labels.pop(name, None)
            self._sample_trial.pop(name, None)
            self._sample_is_comp.pop(name, None)
            ap = self._name_to_path.pop(name, None)
            if ap is not None:
                self._path_to_name.pop(ap, None)
            for ckey in [k for k in self._ac_cache if k[0] == name]:
                self._ac_cache.pop(ckey, None)
            for ckey in [k for k in self._ac_count_cache if k[0] == name]:
                self._ac_count_cache.pop(ckey, None)
            for ckey in [k for k in self._ac_method_cache if k[0] == name]:
                self._ac_method_cache.pop(ckey, None)
            removed += 1
        if not removed:
            return 0
        # Keep only trials that still have a loaded sample, preserving order.
        self._trial_order = [t for t in self._trial_order
                             if any(self._sample_trial.get(n) == t
                                    for n in self._samples)]
        if self._active_sample not in self._samples:
            self._set_active_sample(next(iter(self._samples), None))
        self._refresh_gate_list()
        self._schedule_replot(0)
        return removed

    # ── Channel pickers ──────────────────────────────────────────────────

    def _fmt_channel(self, det):
        lbl = self._channel_labels.get(det, det)
        return f'{lbl} ({det})' if lbl and lbl != det else det

    def _resolve_channel(self, display):
        if not display:
            return None
        m = re.match(r'.*\(([^)]+)\)\s*$', display)
        return m.group(1) if m else display

    def _populate_channel_combos(self):
        display = [self._fmt_channel(c) for c in self._channels]
        self.x_combo['values']     = display
        self.y_combo['values']     = display
        self.color_combo['values'] = ['By sample', 'By density'] + display

        # Sensible defaults — FSC-A on X, SSC-A on Y if available
        fsc = next((c for c in self._channels if 'FSC' in c.upper()
                    and '-A' in c.upper()), None)
        if not fsc:
            fsc = next((c for c in self._channels if 'FSC' in c.upper()),
                       self._channels[0])
        ssc = next((c for c in self._channels if 'SSC' in c.upper()
                    and '-A' in c.upper()), None)
        if not ssc:
            ssc = next((c for c in self._channels if 'SSC' in c.upper()),
                       self._channels[min(1, len(self._channels) - 1)])
        self.x_combo.set(self._fmt_channel(fsc))
        self.y_combo.set(self._fmt_channel(ssc))
        self.color_combo.set('By sample')

    # ── Plotting ─────────────────────────────────────────────────────────

    def _selected_samples(self):
        """Samples currently checked for plot inclusion (☑ in the tree).
        Preserves the original load order."""
        return [n for n in self._sample_order
                if n in self._samples
                and self._sample_plot_enabled.get(n, True)]

    # ── Gate model bookkeeping ────────────────────────────────────────────
    #
    # Storage is `self._gates: dict[str, dict]` keyed by an auto id.
    # Schema is shared with flow_pipeline.gate_to_mask (see that module).

    def _next_gate_id(self):
        self._gate_id_seq += 1
        # Counter is per-sample; the int rebind doesn't propagate via the
        # shared-reference trick we use for dicts, so mirror it explicitly.
        if self._active_sample is not None:
            self._sample_gate_seq[self._active_sample] = self._gate_id_seq
        return f'g{self._gate_id_seq}'

    def _next_color(self):
        """Pick the next colour from flow_pipeline.GATE_PALETTE, cycling."""
        from .pipeline import GATE_PALETTE
        return GATE_PALETTE[self._gate_id_seq % len(GATE_PALETTE)]

    def _add_gate(self, gate, parent_id=None, audit=True):
        """Register `gate` and return its id.

        Fills in defaults for `parent_id` (the currently-selected gate, or
        None for a root) and `color` (next palette entry) if the caller
        didn't supply them.

        Threshold/interval gates are *unique per (channel, parent)*: if a
        matching one already exists, this call replaces it in place — that's
        how a histogram slider live-updates a single gate, and how repeated
        double-clicks on the plot move the same threshold instead of
        stacking new ones.

        ``audit`` records a ``gate.add`` provenance entry for genuinely-new
        gates (suppressed during bulk import / session restore via
        ``_suspend_undo``, and by callers that log their own summary — e.g.
        auto-gate). Threshold/interval *replacements* are never logged (they'd
        flood from slider drags).
        """
        self._checkpoint()
        if 'parent_id' not in gate:
            gate['parent_id'] = (parent_id
                                 if parent_id is not None
                                 else self._selected_gate_id())
        if 'color' not in gate:
            gate['color'] = self._next_color()
        if 'enabled' not in gate:
            gate['enabled'] = True

        if gate.get('kind') in ('threshold', 'interval'):
            ch = gate['channel']
            pid = gate['parent_id']
            for gid in list(self._gates):
                g = self._gates[gid]
                if (g.get('kind') in ('threshold', 'interval')
                        and g.get('channel') == ch
                        and g.get('parent_id') == pid):
                    # Keep the existing colour so the user's visual cue
                    # doesn't change when they nudge a threshold.
                    gate['color'] = g.get('color', gate['color'])
                    self._gates[gid] = gate
                    return gid

        gid = self._next_gate_id()
        self._gates[gid] = gate
        self._gate_id_order.append(gid)
        if audit and not self._suspend_undo:
            from .pipeline import describe_gate
            self._audit('gate.add', sample=self._active_sample, id=gid,
                        kind=gate.get('kind'),
                        gate=describe_gate(gate))
        return gid

    def _create_autoclean_gate(self):
        """Auto-clean button: add an 'autocleaned sample' gate to the active
        sample. It's a recipe gate (a group of toggleable cleaning methods —
        debris / doublets / margin / flow-rate / drift), not fixed geometry:
        each method recomputes from the sample's own data, so copying it to
        other samples (via Copy) re-runs the calculations there. Rendered as a
        collapsed group; build downstream gates under it to gate on cleaned
        events. One group per sample."""
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Load and select a sample first.")
            return
        if any(g.get('kind') == 'autoclean' for g in self._gates.values()):
            self.status_var.set(
                "Auto-clean is already on this sample (toggle its methods "
                "under it, or Copy it to other samples).")
            return
        from .pipeline import default_autoclean_methods
        gate = {
            'kind':      'autoclean',
            'name':      'autocleaned sample',
            'parent_id': None,
            'color':     '#808080',     # never drawn (no geometry)
            'open':      False,         # collapsed by default
            'methods':   default_autoclean_methods(),
        }
        bead_name = self._autoclean_stamp_refs(name, gate)
        self._add_gate(gate, parent_id=None)
        self._refresh_gate_list()
        self._schedule_replot(0)
        beadmsg = (f"Debris cut calibrated to beads ‘{bead_name}’." if bead_name
                   else "No bead file found — debris uses the auto-valley cut.")
        self.status_var.set(
            "Added 'autocleaned sample'. " + beadmsg + " Set display to "
            "'filter' to apply it; right-click its Debris/Dead-cells methods "
            "to switch mode or set bead size. Use Copy to recompute on other "
            "samples.")

    def _resolve_bead_anchor(self):
        """Median FSC-A of a size-calibration bead sample among the loaded
        samples — the absolute-size anchor for the debris cut. Scans sample
        names for a bead / rainbow / calibration token; returns
        ``(median_fsc, sample_name)`` or ``(None, None)``. FSC-A is linear in
        the editor's data (only fluorescence channels are transformed), so the
        median is a valid linear size ruler."""
        from .pipeline import _autoclean_find_scatter
        tokens = ('bead', 'rainbow', 'calib')
        for nm in self._sample_order:
            if not any(tok in nm.lower() for tok in tokens):
                continue
            sd = getattr(self._samples.get(nm), 'data', None)
            if sd is None or len(sd) == 0:
                continue
            fsc = _autoclean_find_scatter(sd, 'FSC', '-A')
            if fsc is None:
                continue
            vals = np.asarray(sd[fsc].values, dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) < 100:
                continue
            return float(np.median(vals)), nm
        return None, None

    def _autoclean_stamp_refs(self, name, gate):
        """Stamp environment-derived references into an auto-clean recipe in
        place: the debris method's bead anchor (``bead_fsc``) and the viability
        method's dye ``channel``. Missing references are left unset, so the
        pure masks degrade gracefully (debris → auto-valley, viability →
        token auto-detect or no-op). Returns the bead sample's name, or None."""
        from .pipeline import find_viability_channel
        sd = getattr(self._samples.get(name), 'data', None)
        labels = getattr(self._samples.get(name), 'channel_labels', {}) or {}
        bead_fsc, bead_name = self._resolve_bead_anchor()
        for m in gate.get('methods') or []:
            key = m.get('key')
            mp = m.setdefault('params', {})
            if key == 'debris':
                if bead_fsc:
                    mp['bead_fsc'] = bead_fsc
                else:
                    mp.pop('bead_fsc', None)
            elif key == 'viability' and sd is not None and not mp.get('channel'):
                ch = find_viability_channel(list(sd.columns), labels)
                if ch:
                    mp['channel'] = ch
        return bead_name

    # ── Auto-clean method quick-edit (right-click menu) ─────────────────────
    def _autoclean_method(self, name, gid, key):
        """The (gate, method-dict) pair for method ``key`` under auto-clean
        gate ``gid`` on ``name``; ``(None, None)`` if absent."""
        g = self._sample_gates.get(name, {}).get(gid)
        if g is None or g.get('kind') != 'autoclean':
            return None, None
        for m in g.get('methods') or []:
            if m.get('key') == key:
                return g, m
        return g, None

    def _autoclean_invalidate(self, name, gid):
        """Drop the cached masks/counts for one auto-clean gate and replot."""
        self._ac_cache.pop((name, gid), None)
        self._ac_count_cache.pop((name, gid), None)
        self._ac_method_cache.pop((name, gid), None)
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _autoclean_set_param(self, name, gid, key, **params):
        """Set (or, when a value is None, clear) params on one auto-clean
        method, with an undo checkpoint + cache invalidation."""
        _g, m = self._autoclean_method(name, gid, key)
        if m is None:
            return
        self._checkpoint()
        mp = m.setdefault('params', {})
        for k, v in params.items():
            if v is None:
                mp.pop(k, None)
            else:
                mp[k] = v
        self._autoclean_invalidate(name, gid)

    def _autoclean_set_debris_mode(self, name, gid, mode):
        """Switch the debris method between 'bead' (absolute size) and
        'valley'. Selecting 'bead' re-resolves the bead anchor from the
        loaded samples; if none is found it stays in bead mode but the mask
        falls back to the valley cut until a bead file is added."""
        _g, m = self._autoclean_method(name, gid, 'debris')
        if m is None:
            return
        self._checkpoint()
        mp = m.setdefault('params', {})
        mp['mode'] = mode
        if mode == 'bead':
            bead_fsc, bead_name = self._resolve_bead_anchor()
            if bead_fsc:
                mp['bead_fsc'] = bead_fsc
                self.status_var.set(f"Debris → beads ‘{bead_name}’.")
            else:
                mp.pop('bead_fsc', None)
                self.status_var.set(
                    "Debris → beads, but no bead file is loaded — falls back "
                    "to the auto-valley cut until one is added.")
        else:
            self.status_var.set("Debris → auto-valley (density) cut.")
        self._autoclean_invalidate(name, gid)

    def _autoclean_prompt_float(self, name, gid, key, param, title, prompt,
                                default, minval=0.0):
        from tkinter import simpledialog
        _g, m = self._autoclean_method(name, gid, key)
        if m is None:
            return
        cur = float((m.get('params') or {}).get(param, default))
        val = simpledialog.askfloat(title, prompt, initialvalue=cur,
                                    minvalue=minval, parent=self)
        if val is not None:
            self._autoclean_set_param(name, gid, key, **{param: float(val)})

    def _autoclean_set_viability_channel(self, name, gid, channel):
        """Pin the viability dye channel (``channel=None`` ⇒ auto-detect)."""
        self._autoclean_set_param(name, gid, 'viability',
                                  channel=(channel or None))

    def _batch_correct_cytonorm(self):
        """Batch-correct loaded samples with CytoNorm 2.0 (control-free 'goal'
        mode). Samples are batched by their trial/day; per-marker intensities
        are quantile-normalized within FlowSOM metaclusters onto the pooled
        goal. Modifies each sample's data in place (re-add samples to revert).
        The classic controls-anchored mode is CLI-only."""
        import pandas as pd

        from .pipeline import CytoNorm
        from .workspace import proper_run_channels

        by_batch: dict[str, list] = {}
        for name in self._sample_order:
            s = self._samples.get(name)
            if s is not None:
                by_batch.setdefault(
                    self._sample_trial.get(name, 'Trial'), []).append((name, s))
        if len(by_batch) < 2:
            self.status_var.set("Batch-norm needs ≥2 batches — load samples "
                                "from multiple trials/days first.")
            return
        # Shared marker channels present in every sample.
        all_samples = [s for lst in by_batch.values() for _, s in lst]
        shared = None
        for s in all_samples:
            cs = set(proper_run_channels(s))
            shared = cs if shared is None else (shared & cs)
        channels = [c for c in proper_run_channels(all_samples[0])
                    if c in (shared or set()) and c in all_samples[0].data.columns]
        if len(channels) < 2:
            self.status_var.set("Batch-norm: <2 shared marker channels across "
                                "the loaded samples.")
            return

        nsamp = len(all_samples)
        if not messagebox.askyesno(
                "Batch-normalize (CytoNorm 2.0)",
                f"Normalize {nsamp} sample(s) across {len(by_batch)} batches "
                f"on {len(channels)} markers?\n\n"
                f"This modifies the loaded sample data in place (gating, "
                f"clustering and stats will use the corrected values). "
                f"Re-add the samples to revert — it is not undoable.",
                parent=self):
            return

        events_by_batch = {}
        for batch, lst in by_batch.items():
            frames = [s.data[channels] for _, s in lst
                      if all(c in s.data.columns for c in channels)]
            if frames:
                events_by_batch[batch] = pd.concat(frames, ignore_index=True)
        if len(events_by_batch) < 2:
            self.status_var.set("Batch-norm: <2 usable batches.")
            return

        self.status_var.set(
            f"CytoNorm: fitting across {len(events_by_batch)} batches on "
            f"{len(channels)} markers …")
        self.update_idletasks()
        try:
            cn = CytoNorm(channels, n_metaclusters=10, mode='goal').fit(
                events_by_batch)
            qc = cn.qc(events_by_batch)
            for batch, lst in by_batch.items():
                for name, s in lst:
                    s.data = cn.apply(s.data, batch)
                    # data object changed → drop its cached masks.
                    for c in (self._ac_cache, self._ac_count_cache,
                              self._ac_method_cache):
                        for ck in [k for k in c if k[0] == name]:
                            c.pop(ck, None)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.status_var.set(
                f"CytoNorm failed: {type(exc).__name__}: {exc}")
            return
        before = float(np.mean([d['before'] for d in qc.values()]))
        after = float(np.mean([d['after'] for d in qc.values()]))
        if before > 0:
            msg = (f"CytoNorm applied: {len(channels)} markers · "
                   f"{len(events_by_batch)} batches · mean batch→goal distance "
                   f"{before:.3f} → {after:.3f} "
                   f"({100 * (1 - after / before):.0f}% lower).")
        else:
            msg = "CytoNorm applied."
        self._refresh_gate_list()
        self._schedule_replot(0)
        self._audit('cytonorm', mode='goal', n_metaclusters=10,
                    n_samples=nsamp, n_batches=len(events_by_batch),
                    batches=sorted(events_by_batch),
                    n_markers=len(channels),
                    dist_before=round(before, 4), dist_after=round(after, 4))
        self.status_var.set(msg)

    def _open_spectral_unmix(self):
        """Open the spectral-unmixing dialog. Designate loaded single-stain
        controls (→ fluorophore) and an unstained control; every other loaded
        sample gets unmixed into per-fluor ``U:`` abundance channels."""
        if len(self._samples) < 2:
            self.status_var.set(
                "Load the single-stain controls (+ unstained) and your "
                "sample(s) first, then Unmix.")
            return
        names = [n for n in self._sample_order if n in self._samples]
        ref = self._samples.get(self._active_sample) or self._samples[names[0]]
        detectors = list(getattr(ref, 'fluor_channels', None) or [])
        SpectralUnmixDialog(self, names, detectors, self._apply_spectral_unmix)

    def _apply_spectral_unmix(self, singles, unstained, detectors, nonneg):
        """Build reference spectra from the assigned controls and unmix every
        non-control loaded sample, adding ``U:<fluor>`` abundance channels."""
        from .spectral import apply_unmixing, build_reference_spectra
        try:
            stains = {}
            for nm, fluor in singles.items():
                s = self._samples[nm]
                cols = [d for d in detectors if d in s.data.columns]
                stains[fluor] = s.data[cols].to_numpy(dtype=float)
            un = None
            if unstained and unstained in self._samples:
                s = self._samples[unstained]
                un = s.data[[d for d in detectors
                             if d in s.data.columns]].to_numpy(dtype=float)
            spectra, fluors = build_reference_spectra(stains, unstained=un)
        except Exception as exc:
            self.status_var.set(
                f"Spectral build failed: {type(exc).__name__}: {exc}")
            return
        control_names = set(singles) | ({unstained} if unstained else set())
        applied = 0
        for nm in list(self._samples):
            if nm in control_names:
                continue
            s = self._samples[nm]
            cols = [d for d in detectors if d in s.data.columns]
            if len(cols) != spectra.shape[1]:
                continue
            try:
                apply_unmixing(s, spectra, fluors, cols, nonneg=nonneg)
                applied += 1
            except Exception as exc:
                print(f"[unmix] {nm}: {type(exc).__name__}: {exc}", flush=True)
        self._refresh_channel_choices()
        self._plot_reference_spectra(spectra, fluors)
        # Spectral QC: similarity + spillover-spread diagnostics. Stash so the
        # Spectral-QC view can (re)render; include the unstained array as the
        # 'Autofluorescence' stain so its SSM column is defined.
        qc_stains = dict(stains)
        if un is not None and 'Autofluorescence' in fluors:
            qc_stains['Autofluorescence'] = un
        try:
            from .spectral import unmixing_qc
            qc = unmixing_qc(qc_stains, spectra, fluors, nonneg=nonneg)
            self._last_unmix_qc = qc
            self._last_unmix_spectra = (spectra, fluors)
        except Exception as exc:
            print(f"[spectral-qc] {type(exc).__name__}: {exc}", flush=True)
            qc = None
        self._audit('unmix', n_samples=applied, n_fluors=len(fluors),
                    n_detectors=int(spectra.shape[1]),
                    fluors=list(fluors), nonneg=bool(nonneg),
                    unstained=unstained or None,
                    condition_number=(round(qc['condition_number'], 1)
                                      if qc else None),
                    similar_pairs=(len(qc['similar_pairs']) if qc else None))
        sim_note = ""
        if qc and qc['similar_pairs']:
            sim_note = (f"  [!] {len(qc['similar_pairs'])} spectrally-similar "
                        f"pair(s) — see Spectral QC.")
        self.status_var.set(
            f"Unmixed {applied} sample(s) → {len(fluors)} U: channels "
            f"({len(fluors)} fluors × {spectra.shape[1]} detectors). "
            f"Select a 'U:' channel to plot.{sim_note}")
        self._refresh_gate_list()
        if qc is not None:
            self._show_spectral_qc(qc)

    def _show_spectral_qc(self, qc=None):
        """Open the Spectral-QC window for the given (or last) unmixing QC
        report: similarity + spillover-spread heatmaps, condition number, and
        the flagged similar/spread pairs, with export."""
        qc = qc or self._last_unmix_qc
        if qc is None:
            self.status_var.set("Run Unmix first — no spectral QC yet.")
            return
        SpectralQCWindow(self, qc, audit=self._audit)

    def _plot_reference_spectra(self, spectra, fluors):
        """Signature plot of the reference spectra (one line per fluor)."""
        self.ax.clear()
        x = range(spectra.shape[1])
        for i, f in enumerate(fluors):
            self.ax.plot(x, spectra[i], marker='o', ms=2, linewidth=1.2,
                         label=f)
        self.ax.set_xlabel('detector')
        self.ax.set_ylabel('normalized signal')
        self.ax.set_title('Reference spectra (single-stain signatures)')
        self.ax.legend(fontsize=7, loc='best', framealpha=0.85)
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_tv_double_click(self, event):
        """Double-clicking an auto-clean gate row (or one of its method rows)
        opens the parameter dialog. Other rows are unaffected."""
        try:
            row = self.gate_tv.identify_row(event.y)
        except Exception:
            return
        p = self._parse_iid(row) if row else None
        if p is None:
            return
        if p[0] == 'method':
            self._edit_autoclean_params(p[1], p[2])
            return 'break'
        if p[0] == 'gate':
            g = self._sample_gates.get(p[1], {}).get(p[2])
            if g is not None and g.get('kind') == 'autoclean':
                self._edit_autoclean_params(p[1], p[2])
                return 'break'

    def _edit_autoclean_params(self, name=None, gid=None):
        """Modal dialog to tune an auto-clean gate's per-method parameters and
        enabled flags. Defaults to the active sample's first auto-clean gate.
        Applies write back the recipe, invalidate the mask cache, checkpoint,
        and replot. Parameters are sample-agnostic — they recompute per sample.
        """
        if name is None:
            name = self._active_sample
        gates = self._sample_gates.get(name, {})
        if gid is None:
            gid = next((k for k, g in gates.items()
                        if g.get('kind') == 'autoclean'), None)
        g = gates.get(gid) if gid else None
        if g is None or g.get('kind') != 'autoclean':
            self.status_var.set("No auto-clean gate to edit.")
            return
        methods = g.get('methods') or []

        dlg = tk.Toplevel(self)
        dlg.title("Auto-clean parameters")
        dlg.transient(self)
        dlg.resizable(False, False)
        ttk.Label(
            dlg, padding=(12, 10, 12, 4), justify='left',
            text=("Tune the cleaning recipe. Values recompute per sample "
                  "(no fixed coordinates).")).pack(anchor='w')
        body = ttk.Frame(dlg, padding=(12, 0))
        body.pack(fill='both', expand=True)

        int_keys = {'n_bins'}
        str_keys = {'mode', 'channel'}     # parsed as text, not numbers
        auto_keys = {'min_fsc', 'channel', 'max_signal'}  # blank ⇒ auto (pop)
        rows = []   # (method, enabled_var, {param_key: (StringVar, is_int)})
        for m in methods:
            key = m.get('key', '')
            sec = ttk.LabelFrame(body, text=m.get('label', key), padding=6)
            sec.pack(fill='x', pady=4)
            en = tk.BooleanVar(value=bool(m.get('enabled', True)))
            ttk.Checkbutton(sec, text="enabled", variable=en).grid(
                row=0, column=0, columnspan=2, sticky='w')
            params = dict(m.get('params') or {})
            if key == 'debris':            # surface the optional manual override
                params.setdefault('min_fsc', None)
            if key == 'viability':         # surface the optional dye channel
                params.setdefault('channel', None)
            pentries = {}
            r = 1
            for pk, pv in params.items():
                ttk.Label(sec, text=f'{pk}:').grid(
                    row=r, column=0, sticky='e', padx=(0, 6), pady=1)
                sv = tk.StringVar(value=('' if pv is None else str(pv)))
                ttk.Entry(sec, textvariable=sv, width=14).grid(
                    row=r, column=1, sticky='w', pady=1)
                pentries[pk] = (sv, pk in int_keys)
                r += 1
            hint = {'debris':    "(mode bead→valley · blank min_fsc/min_um/bead_um = auto)",
                    'viability': "(blank channel = auto-detect viability dye)"}.get(key)
            if hint:
                ttk.Label(sec, text=hint, foreground='grey',
                          font=('TkDefaultFont', 8)).grid(
                    row=r, column=0, columnspan=2, sticky='w')
            rows.append((m, en, pentries))

        err = tk.StringVar(value='')
        ttk.Label(dlg, textvariable=err, foreground='#b00',
                  padding=(12, 0)).pack(anchor='w')

        def _apply():
            staged = []
            for m, en, pentries in rows:
                params = {}
                for pk, (sv, is_int) in pentries.items():
                    raw = sv.get().strip()
                    if raw == '':
                        continue        # blank: leave unchanged (auto)
                    if pk in str_keys:
                        params[pk] = raw
                        continue
                    try:
                        params[pk] = int(float(raw)) if is_int else float(raw)
                    except ValueError:
                        err.set(f"{m.get('label')}: '{pk}' must be a number.")
                        return
                staged.append((m, bool(en.get()), params, pentries))
            self._checkpoint()
            for m, enabled, params, pentries in staged:
                m['enabled'] = enabled
                mp = m.setdefault('params', {})
                # A cleared auto field (min_fsc / channel / max_signal) reverts
                # that method to its automatic detection.
                for ak in auto_keys:
                    if (ak in pentries
                            and pentries[ak][0].get().strip() == ''):
                        mp.pop(ak, None)
                mp.update(params)
            self._ac_cache.pop((name, gid), None)   # recipe changed → recompute
            self._ac_count_cache.pop((name, gid), None)
            self._ac_method_cache.pop((name, gid), None)
            dlg.destroy()
            self._refresh_gate_list()
            self._schedule_replot(0)
            self.status_var.set("Auto-clean parameters updated.")

        btns = ttk.Frame(dlg, padding=12)
        btns.pack(anchor='e')
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Apply", command=_apply).pack(
            side='right', padx=(0, 6))
        dlg.bind('<Escape>', lambda _e: dlg.destroy())
        try:
            dlg.grab_set()
            self.wait_window(dlg)
        except Exception:
            pass

    # ── Treeview iid encoding (samples + gates share one tree) ──────────
    #
    # Sample rows:  'S:<sample_name>'
    # Gate rows:    'G:<sample_name>/<gate_id>'
    # Sample names usually have no ':' or '/' — FCS filenames don't — so
    # rsplit on '/' for the gate split keeps things robust.

    @staticmethod
    def _sample_iid(name):
        return f'S:{name}'

    @staticmethod
    def _gate_iid(sample_name, gid):
        return f'G:{sample_name}/{gid}'

    @staticmethod
    def _trial_iid(trial):
        return f'T:{trial}'

    @staticmethod
    def _method_iid(sample_name, gid, key):
        # Synthetic row for one auto-clean method under its 'autoclean' gate.
        return f'M:{sample_name}/{gid}/{key}'

    @staticmethod
    def _subgroup_iid(kind, trial):
        # Comps/Samples sub-header under a trial. kind ∈ {'comp', 'samp'}.
        return f'SG:{kind}:{trial}'

    @staticmethod
    def _parse_iid(iid):
        """('sample', name) | ('gate', sample_name, gid) |
        ('method', sample_name, gid, key) | ('subgroup', kind, trial) |
        ('trial', trial) | None."""
        if iid.startswith('S:'):
            return ('sample', iid[2:])
        if iid.startswith('SG:'):
            parts = iid.split(':', 2)
            if len(parts) == 3:
                return ('subgroup', parts[1], parts[2])
            return None
        if iid.startswith('T:'):
            return ('trial', iid[2:])
        if iid.startswith('M:'):
            parts = iid[2:].rsplit('/', 2)
            if len(parts) == 3:
                return ('method', parts[0], parts[1], parts[2])
            return None
        if iid.startswith('G:'):
            rest = iid[2:]
            if '/' not in rest:
                return None
            name, gid = rest.rsplit('/', 1)
            return ('gate', name, gid)
        return None

    def _selected_gate_id(self):
        """Gate_id of the currently-selected tree row IF it belongs to the
        active sample; otherwise None. Used by _add_gate to decide what
        parent a new shape should attach to."""
        if not hasattr(self, 'gate_tv'):
            return None
        sel = self.gate_tv.selection()
        if not sel:
            return None
        parsed = self._parse_iid(sel[0])
        if parsed is None or parsed[0] != 'gate':
            return None
        sample_name, gid = parsed[1], parsed[2]
        if sample_name != self._active_sample:
            return None
        return gid if gid in self._gates else None

    def _on_tree_select(self, *_):
        """Row selection switched: update active sample (= owner of the
        selected row, sample row or gate row alike) and replot if it
        actually changed."""
        sel = self.gate_tv.selection()
        if not sel:
            return
        parsed = self._parse_iid(sel[0])
        if parsed is None:
            return
        if parsed[0] in ('trial', 'subgroup'):
            return  # group header — expand/collapse + display toggle only
        owning_sample = parsed[1]
        if owning_sample and owning_sample != self._active_sample:
            self._set_active_sample(owning_sample)
            # Don't re-refresh the tree (would clobber selection); just
            # repaint the canvas.
            self._schedule_replot(50)

    def _ordered_gate_ids(self):
        """All gate ids in insertion order, with any dangling ids
        (added via _add_gate's 1D replacement path) sorted at the end."""
        seen = set(self._gate_id_order)
        return list(self._gate_id_order) + [g for g in self._gates if g not in seen]

    def _axis_alias_for_sample(self, s, dets):
        """Label-first axis resolution (#48).

        A chosen axis is a detector from the global panel (the first
        sample's columns). When THIS sample lacks that exact detector but
        carries the same antibody label on a different fluor, expose its
        own detector under the chosen name so the plot overlays on a
        common label axis instead of dropping the sample. Detectors the
        sample already has, and non-fluor axes (FSC-A/SSC-A), are left
        untouched.

        Returns {chosen_detector: own_detector} — an *alias* map. We alias
        (copy) rather than rename so the sample's own detector column stays
        present for gate masks, which read each sample's retargeted
        detectors (see #47).
        """
        from .pipeline import _sample_fluor_labels
        cols = set(s.data.columns)
        l2d = _sample_fluor_labels(s)        # {label: this sample's detector}
        alias = {}
        for det in dets:
            if not det or det in cols:
                continue                     # sample already carries it
            label = self._channel_labels.get(det, det)
            own = l2d.get(label)
            if own and own in cols:
                alias[det] = own
        return alias

    def _get_df(self, name, x, y=None, for_hist=False):
        s  = self._samples[name]
        df = s.data
        alias = self._axis_alias_for_sample(s, [x, y])
        if alias:
            # Add chosen-name columns from this sample's own detectors;
            # copy leaves s.data and the original detector columns intact.
            df = df.assign(**{chosen: df[own] for chosen, own in alias.items()})
        cols = [c for c in (x, y) if c]
        df = df.dropna(subset=[c for c in cols if c in df.columns])

        # Each sample applies its OWN gate tree.
        # Filter mode keeps events that are inside the cumulative chain of
        # ANY enabled gate (union of populations). So with `P enabled` and
        # `C enabled` you see events in P OR events in C, not the empty
        # intersection of two disjoint forks. With just `C enabled` you see
        # events in (root...P AND C) — ancestors always filter, regardless
        # of their toggle (the toggle is visibility, not chain membership).
        sample_gates = self._sample_gates.get(name, {})
        if self.apply_gates_var.get() and sample_gates:
            from .pipeline import cumulative_gate_mask
            overrides = self._autoclean_overrides(name, df)
            mask = np.zeros(len(df), dtype=bool)
            any_enabled = False
            for gid, g in sample_gates.items():
                if g.get('enabled', True):
                    mask |= cumulative_gate_mask(sample_gates, gid, df,
                                                 overrides=overrides)
                    any_enabled = True
            if any_enabled:
                df = df[mask]

        # Display-only auto-downsample: when enabled, every plotted
        # sample renders the same number of events as the smallest
        # loaded sample. Underlying FlowSample.data is untouched.
        #
        # The 60k ceiling is a *scatter-rendering* guard (drawing 200k points
        # is slow); a histogram bins cheaply, so for_hist skips that ceiling
        # and keeps raw counts truthful — while STILL honouring the
        # downsample-to-smallest toggle so overlaid counts stay comparable.
        cap = None if for_hist else self._display_point_cap()
        if (getattr(self, 'ds_display_var', None) is not None
                and self.ds_display_var.get()):
            floor = self._smallest_loaded_sample_size()
            if floor is not None and floor > 0:
                cap = floor if cap is None else min(cap, floor)
        if cap is not None and len(df) > cap:
            # Seed by name + cap so the same subsample is picked across
            # replots — keeps the plot stable while the user pans gates.
            seed = (hash((name, x, y, cap)) & 0xFFFF_FFFF)
            df = df.sample(cap, random_state=seed)
        return df

    def _display_point_cap(self):
        """Max events drawn per sample in scatter / pseudocolor / contour
        modes, from the 'Max points' control. 'All' (or blank / 0) removes the
        cap. Accepts plain integers or '250k'-style shorthand. Defaults to
        60 000 so large samples stay responsive; histograms ignore this."""
        v = getattr(self, 'max_points_var', None)
        if v is None:
            return 60_000
        raw = str(v.get()).strip().lower().replace(',', '')
        if raw in ('', 'all', '0', 'none'):
            return 1 << 62                      # effectively uncapped
        try:
            if raw.endswith('k'):
                return max(1000, int(float(raw[:-1]) * 1000))
            if raw.endswith('m'):
                return max(1000, int(float(raw[:-1]) * 1_000_000))
            return max(1000, int(float(raw)))
        except ValueError:
            return 60_000

    def _on_max_points_changed(self):
        """Max-points control edited: refresh the per-sample event counts in
        the tree (they show shown/total) and replot with the new cap."""
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _smallest_loaded_sample_size(self):
        """Smallest in-memory FlowSample.data length across all loaded
        samples that are currently enabled for plotting. None when no
        samples qualify."""
        sizes = []
        for n in self._sample_order:
            if n not in self._samples:
                continue
            if not self._sample_plot_enabled.get(n, False):
                continue
            try:
                sizes.append(len(self._samples[n].data))
            except Exception:
                continue
        if not sizes:
            return None
        return min(sizes)

    def _sample_display_count(self, name):
        """``(shown, total)`` events for ``name``: the full FlowSample size and
        how many are actually drawn after the display caps (the 60k scatter
        guard and the auto-downsample-to-smallest toggle). ``shown == total``
        when nothing is scaled down."""
        s = self._samples.get(name)
        data = getattr(s, 'data', None) if s is not None else None
        if data is None:
            return (0, 0)
        total = len(data)
        cap = self._display_point_cap()
        if (getattr(self, 'ds_display_var', None) is not None
                and self.ds_display_var.get()):
            floor = self._smallest_loaded_sample_size()
            if floor is not None and floor > 0:
                cap = min(cap, floor)
        return (min(total, cap), total)

    def _on_downsample_display_toggled(self):
        """Display auto-downsample toggled: replot AND refresh the tree so the
        per-sample event counts reflect the new scaled-down numbers."""
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _autoclean_overrides(self, name, df):
        """``{gid: df-aligned keep-mask}`` for each auto-clean gate on ``name``,
        or ``None`` when there are none. Each mask is computed once on the FULL
        sample data and cached by (data identity, recipe signature), then
        reindexed to ``df``'s rows — so a chain that nests populations under an
        auto-clean root reuses the cached cleaning instead of recomputing it per
        node and per replot."""
        gates = self._sample_gates.get(name, {})
        ac_gids = [gid for gid, g in gates.items()
                   if g.get('kind') == 'autoclean']
        if not ac_gids:
            return None
        import pandas as pd

        from .pipeline import autoclean_keep_mask, autoclean_methods_signature
        sd = getattr(self._samples.get(name), 'data', None)
        if sd is None:
            return None
        data_id = id(sd)
        out = {}
        for gid in ac_gids:
            g = gates[gid]
            sig = autoclean_methods_signature(g)
            ent = self._ac_cache.get((name, gid))
            if ent is not None and ent[0] == data_id and ent[1] == sig:
                full = ent[2]
            else:
                full = pd.Series(autoclean_keep_mask(g, sd), index=sd.index)
                self._ac_cache[(name, gid)] = (data_id, sig, full)
            # df ⊆ sd (assign/dropna preserve the index); align by label.
            out[gid] = full.reindex(df.index, fill_value=True).to_numpy()
        return out

    # Distinct colour per cleaning method, so each removed "section" reads as
    # its own population in the cleaned-out overlay.
    _METHOD_COLORS = {
        'debris':    '#8c564b',   # brown
        'viability': '#1f77b4',   # blue
        'doublets':  '#e8000b',   # red
        'margin':    '#9467bd',   # purple
        'flow_rate': '#17becf',   # cyan
        'drift':     '#2ca02c',   # green
    }

    def _autoclean_method_masks(self, name):
        """``{method_key: full-data boolean removed-mask}`` for every ENABLED
        cleaning method across the sample's auto-clean gate(s), cached by
        (data identity, recipe signature). Each mask marks the events that
        method removes on its own. ``{}`` when there's no auto-clean gate."""
        gates = self._sample_gates.get(name, {})
        ac_gids = [gid for gid, g in gates.items()
                   if g.get('kind') == 'autoclean']
        if not ac_gids:
            return {}
        import pandas as pd

        from .pipeline import autoclean_keep_mask, autoclean_methods_signature
        sd = getattr(self._samples.get(name), 'data', None)
        if sd is None or len(sd) == 0:
            return {}
        data_id = id(sd)
        out = {}
        for gid in ac_gids:
            g = gates[gid]
            sig = autoclean_methods_signature(g)
            ent = self._ac_method_cache.get((name, gid))
            if ent is not None and ent[0] == data_id and ent[1] == sig:
                masks = ent[2]
            else:
                masks = {}
                for m in g.get('methods', []):
                    if not m.get('enabled', True):
                        continue
                    solo = {'kind': 'autoclean',
                            'methods': [{**m, 'enabled': True}]}
                    rm = ~np.asarray(autoclean_keep_mask(solo, sd), dtype=bool)
                    masks[m.get('key', '')] = pd.Series(rm, index=sd.index)
                self._ac_method_cache[(name, gid)] = (data_id, sig, masks)
            # First enabled method (recipe order) wins an event it removes.
            for key, ser in masks.items():
                out.setdefault(key, ser)
        return out

    def _autoclean_counts(self, name, gid):
        """``(total, total_drop, {key: drop}, {key: reason|None})`` for the
        auto-clean gate ``gid`` on sample ``name`` — computed on the FULL sample
        data and cached by (data identity, recipe signature). ``total_drop`` is
        how many events the enabled recipe removes (union); each ``method_drop``
        is how many that single method removes on its own; ``reasons`` explains
        any method that removed nothing (e.g. "no viability dye detected") so a
        0-drop isn't a silent mystery. ``None`` when the sample isn't loaded or
        the gate isn't auto-clean."""
        g = self._sample_gates.get(name, {}).get(gid)
        if g is None or g.get('kind') != 'autoclean':
            return None
        sd = getattr(self._samples.get(name), 'data', None)
        if sd is None or len(sd) == 0:
            return None
        from .pipeline import (
            autoclean_keep_mask,
            autoclean_method_diagnostic,
            autoclean_methods_signature,
        )
        labels = getattr(self._samples.get(name), 'channel_labels', {}) or {}
        data_id = id(sd)
        sig = autoclean_methods_signature(g)
        ent = self._ac_count_cache.get((name, gid))
        if ent is not None and ent[0] == data_id and ent[1] == sig:
            return (ent[2], ent[3], ent[4], ent[5])
        total = len(sd)
        total_drop = int((~np.asarray(autoclean_keep_mask(g, sd))).sum())
        per_method = {}
        reasons = {}
        for m in g.get('methods', []):
            mkey = m.get('key', '')
            solo = {'kind': 'autoclean',
                    'methods': [{**m, 'enabled': True}]}
            drop = int((~np.asarray(autoclean_keep_mask(solo, sd))).sum())
            per_method[mkey] = drop
            reasons[mkey] = (autoclean_method_diagnostic(
                mkey, sd, m.get('params') or {}, labels) if drop == 0 else None)
        self._ac_count_cache[(name, gid)] = (
            data_id, sig, total, total_drop, per_method, reasons)
        return (total, total_drop, per_method, reasons)

    @staticmethod
    def _drop_suffix(drop, total):
        """' — drops N (X%)' for a tree row, or '' when nothing is dropped /
        the total is unknown."""
        if not total or drop is None:
            return ''
        return f'  —  drops {drop:,} ({100.0 * drop / total:.1f}%)'

    def _on_downsample_propagate_toggled(self):
        """Propagate toggle handler.

        Turning ON: trims every loaded FlowSample.data to the smallest
        loaded sample's size (seeded random subsample). NOT reversible
        from the GUI — the user must re-add the samples to restore the
        full event count. Surfaces a confirmation in the status bar.

        Turning OFF: no immediate effect on already-trimmed samples
        (we can't restore lost rows), but newly-added samples won't be
        trimmed going forward.
        """
        if not self.ds_propagate_var.get():
            self.status_var.set(
                "Propagate OFF — new samples load full-size. "
                "Already-trimmed samples are not restored (re-add to undo).")
            return
        floor = self._smallest_loaded_sample_size()
        if floor is None or floor <= 0:
            self.status_var.set(
                "Propagate ON — no samples loaded yet; will trim on add.")
            return
        trimmed = 0
        for n in list(self._sample_order):
            s = self._samples.get(n)
            if s is None:
                continue
            if len(s.data) > floor:
                s.data = s.data.sample(floor, random_state=42).reset_index(
                    drop=True)
                trimmed += 1
        self.status_var.set(
            f"Propagate ON — trimmed {trimmed} sample(s) to {floor:,} events.")
        self._schedule_replot(0)

    def _schedule_replot(self, delay_ms=100):
        if self._replot_after_id:
            try:
                self.after_cancel(self._replot_after_id)
            except Exception:
                pass
        self._replot_after_id = self.after(delay_ms, self._replot)

    def _render_placeholder(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Add FCS files and pick channels',
                     ha='center', va='center',
                     transform=self.ax.transAxes, fontsize=11, color='grey')
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas.draw_idle()

    def _replot(self):
        self._replot_after_id = None
        # Remove any prior colorbar
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        self.ax.clear()
        # Forget previously drawn gate Line2D objects (they were on the
        # old axes that we just cleared).
        self._vlines = {}
        self._hlines = {}

        samples = self._selected_samples()
        if not self._samples:
            self._render_placeholder()
            return
        if not samples:
            self.ax.text(0.5, 0.5, 'Select one or more samples on the left',
                         ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11, color='grey')
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.canvas.draw_idle()
            return

        mode  = self.mode_var.get()
        x     = self._resolve_channel(self.x_combo.get())
        y     = self._resolve_channel(self.y_combo.get())
        color = self.color_combo.get()

        if not x:
            self.canvas.draw_idle()
            return

        try:
            if mode == 'histogram':
                self._plot_histogram(samples, x)
            elif mode == 'dot':
                self._plot_dot(samples, x, y, color)
            elif mode == 'pseudocolor':
                self._plot_pseudocolor(samples, x, y)
            elif mode == 'contour':
                self._plot_contour(samples, x, y)
        except Exception as exc:
            self.ax.text(0.5, 0.5, f'Plot error:\n{exc}',
                         ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=10, color='red')

        # Overlay auto-clean-removed events (red, on top) when requested.
        try:
            self._overlay_removed_events(samples, x, y, mode)
        except Exception as exc:
            print(f"[cleaned-out overlay] {type(exc).__name__}: {exc}",
                  flush=True)
        # Backgating: project selected populations onto the current plot.
        try:
            self._overlay_backgate(samples, x, y)
        except Exception as exc:
            print(f"[backgate overlay] {type(exc).__name__}: {exc}",
                  flush=True)

        self.ax.set_xlabel(self._fmt_channel(x), fontsize=9)
        if mode != 'histogram' and y:
            self.ax.set_ylabel(self._fmt_channel(y), fontsize=9)

        # Apply per-channel scale + range. Sample data from the FIRST
        # plotted sample (when one exists) gives the symlog linthresh a
        # data-driven anchor. Done AFTER plotting so the underlying
        # density / scatter has been drawn into linear coords; the
        # scale change is purely a display transform.
        first = samples[0] if samples else None
        sample_data = None
        if first and x and first in self._samples:
            sdf = self._samples[first].data
            if x in sdf.columns:
                sample_data = sdf[x].values
        self._apply_axis_to_ax(x, 'x', sample_data)
        if mode != 'histogram' and y:
            ydata = None
            if first and y and first in self._samples:
                sdf = self._samples[first].data
                if y in sdf.columns:
                    ydata = sdf[y].values
            self._apply_axis_to_ax(y, 'y', ydata)

        # Highlight overlays sit on top of the base population. No-op
        # unless the user has switched to 'Highlight gated'.
        self._draw_highlight_overlays(
            samples, x, y if mode != 'histogram' else None)

        # Draw gates (shapes + threshold/interval lines) on top of the
        # overlays so they remain visible.
        self._draw_gates(x, y if mode != 'histogram' else None)
        self._refresh_gate_list()

        self.fig.tight_layout()
        self.canvas.draw_idle()

        # ax.clear() blew away any matplotlib Selector — reattach.
        self._activate_gate_tool()
        # Show/hide the histogram slider panel and resync ranges.
        self._sync_slider_panel()

    def _plot_dot(self, samples, x, y, color):
        if not y:
            return
        if color == 'By sample':
            for name in samples:
                df = self._get_df(name, x, y)
                if df.empty or x not in df.columns or y not in df.columns:
                    continue
                self.ax.scatter(df[x].values, df[y].values,
                                c=self._color_for(name),
                                s=2, alpha=0.35, linewidths=0, label=name)
            if len(samples) > 1:
                self.ax.legend(fontsize=8, markerscale=4, framealpha=0.85,
                               loc='best')
        elif color == 'By density':
            xs, ys = [], []
            for name in samples:
                df = self._get_df(name, x, y)
                if x not in df.columns or y not in df.columns:
                    continue
                xs.append(df[x].values); ys.append(df[y].values)
            if not xs:
                return
            xs = np.concatenate(xs); ys = np.concatenate(ys)
            self._density_scatter(xs, ys, x, y)
        else:
            cch = self._resolve_channel(color)
            xs, ys, cs = [], [], []
            for name in samples:
                df = self._get_df(name, x, y)
                if (cch not in df.columns or x not in df.columns
                        or y not in df.columns):
                    continue
                xs.append(df[x].values); ys.append(df[y].values)
                cs.append(df[cch].values)
            if xs:
                xs = np.concatenate(xs); ys = np.concatenate(ys)
                cs = np.concatenate(cs)
                sc = self.ax.scatter(xs, ys, c=cs, cmap='viridis',
                                     s=2, alpha=0.55, linewidths=0)
                # Suppress the colorbar when rendering small-multiple panels
                # for the figure exporter — it steals axes space and clutters
                # a grid. The live plot (flag unset) keeps it.
                if not getattr(self, '_suppress_panel_cbar', False):
                    self._cbar = self.fig.colorbar(
                        sc, ax=self.ax, label=self._fmt_channel(cch))

    def _plot_pseudocolor(self, samples, x, y):
        if not y:
            return
        xs, ys = [], []
        for name in samples:
            df = self._get_df(name, x, y)
            if x not in df.columns or y not in df.columns:
                continue
            xs.append(df[x].values); ys.append(df[y].values)
        if not xs:
            return
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        self._density_scatter(xs, ys, x, y)

    def _axis_view_funcs(self, channel, data_sample=None):
        """``(forward, inverse)`` callables mapping a channel's STORED data
        coordinate to screen position for its chosen display scale — or
        ``None`` when the channel's data is already linear (the caller then
        uses matplotlib's native linear/symlog/log scale, which has nicer
        tick locators).

        Fluor data is baked into a nonlinear transform space
        (``_channel_transform``, e.g. logicle). The underlying *linear
        intensity* is the canonical master; it is recovered with
        ``inverse_transform_values``. The chosen scale (linear / symlog /
        log) is then a pure VIEW of that intensity, composed as::

            forward(d) = view_forward(inverse_baked(d))
            inverse(p) = forward_baked(view_inverse(p))

        So every scale is an independent, equation-derived view of the same
        compensated intensity — no double-transform — and gates (kept in
        stored data coords) auto-follow the axis transform for free. symlog
        uses an arcsinh view whose cofactor is anchored on the data.
        """
        from .pipeline import inverse_transform_values, transform_values
        tm = self._channel_transform.get(channel, 'linear')
        if tm == 'linear':
            return None
        scale = self._channel_scale.get(channel, self._default_channel_scale)

        def inv_baked(a):
            return inverse_transform_values(
                np.asarray(a, dtype=float), method=tm)

        def fwd_baked(a):
            return transform_values(np.asarray(a, dtype=float), method=tm)

        if scale == 'log':
            def forward(d):  # pyright: ignore[reportRedeclaration]  # conditional def
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.log10(np.clip(inv_baked(d), 1e-6, None))

            def inverse(p):
                return fwd_baked(np.power(10.0, np.asarray(p, dtype=float)))
        elif scale == 'symlog':
            cof = 150.0
            if data_sample is not None:
                lin = inv_baked(np.asarray(data_sample, dtype=float))
                lin = lin[np.isfinite(lin)]
                nz = np.abs(lin[lin != 0])
                if nz.size > 50:
                    cof = max(float(np.percentile(nz, 5)), 1e-3)

            def forward(d):  # pyright: ignore[reportRedeclaration]  # conditional def
                return np.arcsinh(inv_baked(d) / cof)

            def inverse(p):
                return fwd_baked(np.sinh(np.asarray(p, dtype=float)) * cof)
        else:  # 'linear' view of nonlinear-baked data → stretch to intensity
            def forward(d):
                return inv_baked(d)

            def inverse(p):
                return fwd_baked(p)

        def _finite(fn):
            # matplotlib's FuncScale requires shape-preserving callables, but
            # FlowKit's (inverse_)logicle flattens to 1-D — so reshape back to
            # the input shape (e.g. a (1, 1) tick query) and scrub NaNs.
            def wrapped(a):
                arr = np.asarray(a, dtype=float)
                out = np.nan_to_num(np.asarray(fn(arr), dtype=float), nan=0.0)
                return out.reshape(arr.shape)
            return wrapped

        return _finite(forward), _finite(inverse)

    @staticmethod
    def _symlog_linthresh(data_sample):
        """Linear-region half-width for a native symlog axis: the 5th
        percentile of |nonzero data|, floored. The SAME value is used for the
        display axis and the density binning so the two stay aligned."""
        linthresh = 1.0
        if data_sample is not None:
            arr = np.asarray(data_sample, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 50:
                nz = np.abs(arr[arr != 0])
                if nz.size > 0:
                    linthresh = max(float(np.percentile(nz, 5)), 1e-6)
        return linthresh

    def _symlog_edges(self, lo, hi, n_bins, data_sample):
        """Bin edges uniform in matplotlib's symlog SCREEN transform (linear
        within ``linthresh``, log beyond), matching the native symlog display
        axis. Without this the density uses linear bins, which are far too
        coarse in the log decade (boxy artefacts ~10^3–10^4)."""
        from matplotlib.scale import SymmetricalLogTransform
        lt = self._symlog_linthresh(data_sample)
        t = SymmetricalLogTransform(10, lt, 1)
        slo = float(np.asarray(t.transform(np.array([float(lo)]))).ravel()[0])
        shi = float(np.asarray(t.transform(np.array([float(hi)]))).ravel()[0])
        if not (np.isfinite(slo) and np.isfinite(shi)) or shi <= slo:
            return self._hist_bin_edges(lo, hi, 'linear', n_bins)
        screen = np.linspace(slo, shi, int(n_bins) + 1)
        edges = np.unique(
            np.asarray(t.inverted().transform(screen), dtype=float).ravel())
        edges = edges[np.isfinite(edges)]
        if edges.size < 2:
            return self._hist_bin_edges(lo, hi, 'linear', n_bins)
        return edges.tolist()

    def _screen_uniform_edges(self, channel, lo, hi, n_bins, data_sample=None):
        """``n_bins + 1`` bin edges between data-coords ``lo`` and ``hi``,
        spaced uniformly in SCREEN space for the channel's display scale, so
        density bins aren't banded on a log / symlog / composite axis."""
        funcs = self._axis_view_funcs(channel, data_sample) if channel else None
        if funcs is None:
            scale = (self._channel_scale.get(channel, self._default_channel_scale)
                     if channel else 'linear')
            if scale == 'symlog':
                return self._symlog_edges(lo, hi, n_bins, data_sample)
            return self._hist_bin_edges(lo, hi, scale, n_bins)
        fwd, inv = funcs
        slo = float(np.asarray(fwd(np.array([lo], dtype=float)))[0])
        shi = float(np.asarray(fwd(np.array([hi], dtype=float)))[0])
        if not (np.isfinite(slo) and np.isfinite(shi)) or shi <= slo:
            return self._hist_bin_edges(lo, hi, 'linear', n_bins)
        screen = np.linspace(slo, shi, int(n_bins) + 1)
        edges = np.unique(np.asarray(inv(screen), dtype=float))
        edges = edges[np.isfinite(edges)]
        if edges.size < 2:
            return self._hist_bin_edges(lo, hi, 'linear', n_bins)
        return edges.tolist()

    def _axis_bin_edges(self, vals, channel, n_bins):
        """Bin edges for `vals` in the channel's *display* space, over the
        effective view range (explicit per-channel range if set, else a
        robust 0.5–99.5 percentile). So density bins are visually uniform
        on log/symlog and track the zoom instead of the full data extent."""
        rng = self._channel_range.get(channel) if channel else None
        if rng is not None:
            lo, hi = float(rng[0]), float(rng[1])
        else:
            finite = vals[np.isfinite(vals)]
            if finite.size:
                lo, hi = (float(v) for v in np.percentile(finite, [0.5, 99.5]))
            else:
                lo, hi = 0.0, 1.0
        if hi <= lo:
            lo = float(np.min(vals)) if vals.size else 0.0
            hi = float(np.max(vals)) if vals.size else 1.0
            if hi <= lo:
                hi = lo + 1.0
        return np.asarray(
            self._screen_uniform_edges(channel, lo, hi, n_bins, data_sample=vals),
            dtype=float)

    def _density_scatter(self, xs, ys, xch=None, ych=None):
        """Density-coloured scatter.

        Two modes, controlled by the 'True Gaussian KDE' checkbox:
          • Off (default, FlowJo-style): O(n) 2D histogram + smoothing
            + per-event lookup. Handles tens of millions of events in
            sub-second on CPU.
          • On: scipy.stats.gaussian_kde — mathematically smoother but
            O(n^2). Subsamples aggressively and warns the user.

        `xch`/`ych` (channel names) let the histogram bin in the axis's own
        space (log/symlog/linear) so density isn't banded on a log view.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        finite = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[finite]; ys = ys[finite]
        if xs.size == 0:
            return

        true_kde = (hasattr(self, 'true_kde_var')
                    and self.true_kde_var.get())

        if true_kde:
            self._density_scatter_truekde(xs, ys)
        else:
            self._density_scatter_histogram(xs, ys, xch, ych)

    def _density_scatter_histogram(self, xs, ys, xch=None, ych=None):
        from scipy.ndimage import gaussian_filter, map_coordinates
        BINS         = 256
        MAX_DISPLAY  = self._display_point_cap()
        try:
            # Bin in each axis's display space over the effective view
            # range — uniform linear bins would band on a log/symlog axis.
            x_edges = self._axis_bin_edges(xs, xch, BINS)
            y_edges = self._axis_bin_edges(ys, ych, BINS)
            hist, x_edges, y_edges = np.histogram2d(
                xs, ys, bins=[x_edges, y_edges])
            # Adaptive smoothing with a floor of ~1.8 bins. Sparse data needs
            # a wide kernel or the field is speckled; but even a CLEAN dense
            # histogram shows the bin lattice as faint boxes when sampled
            # per-event, so the floor blurs across ~2 bins to erase the grid
            # without losing population-scale structure.
            per_bin = xs.size / float(BINS * BINS)
            sigma = float(np.clip(np.sqrt(1.0 / max(per_bin, 1e-6)) * 1.2,
                                  1.8, 6.0))
            # Zero-pad the raw histogram, THEN smooth, so the population decays
            # smoothly into a zero border. Without this the bin range cuts the
            # population at a hard rectangular edge (bright box + sharp
            # corners), because tail events otherwise clamp to the bright edge
            # bin. Pad wide enough for the kernel to fall off fully inside it.
            pad = int(np.ceil(3.0 * sigma)) + 1
            hist = np.pad(hist, pad, mode='constant', constant_values=0.0)
            dx = x_edges[1] - x_edges[0]
            dy = y_edges[1] - y_edges[0]
            x_edges = np.concatenate([x_edges[0] + dx * np.arange(-pad, 0),
                                      x_edges,
                                      x_edges[-1] + dx * np.arange(1, pad + 1)])
            y_edges = np.concatenate([y_edges[0] + dy * np.arange(-pad, 0),
                                      y_edges,
                                      y_edges[-1] + dy * np.arange(1, pad + 1)])
            hist = gaussian_filter(hist, sigma=sigma)
            nbx, nby = len(x_edges) - 1, len(y_edges) - 1
            # Per-event density by CUBIC interpolation of the smoothed field —
            # not the bin's flat value. order=3 is C2-continuous, so there are
            # no facet/box edges at bin boundaries (which order=1 bilinear
            # leaves visible on a sharp colormap like jet). Coordinates are
            # each event's fractional position between neighbouring bin centres;
            # events past the padded border interpolate toward zero (mode
            # 'constant'), so the density support has no hard edge.
            ix = np.clip(np.searchsorted(x_edges, xs, side='right') - 1,
                         0, nbx - 1)
            iy = np.clip(np.searchsorted(y_edges, ys, side='right') - 1,
                         0, nby - 1)
            wx = x_edges[ix + 1] - x_edges[ix]
            wy = y_edges[iy + 1] - y_edges[iy]
            fx = ix + np.where(wx > 0, (xs - x_edges[ix]) / wx, 0.5) - 0.5
            fy = iy + np.where(wy > 0, (ys - y_edges[iy]) / wy, 0.5) - 0.5
            z = map_coordinates(hist, np.vstack([fx, fy]),
                                order=3, mode='constant', cval=0.0)
            # Cubic interpolation can overshoot slightly negative near sharp
            # gradients; clamp so the colour norm sees only valid densities.
            np.clip(z, 0.0, None, out=z)
            if xs.size > MAX_DISPLAY:
                rng = np.random.default_rng(42)
                sel = rng.choice(xs.size, MAX_DISPLAY, replace=False)
                xs_d, ys_d, z_d = xs[sel], ys[sel], z[sel]
            else:
                xs_d, ys_d, z_d = xs, ys, z
            order = z_d.argsort()
            self.ax.scatter(xs_d[order], ys_d[order], c=z_d[order],
                            cmap='jet', s=2, alpha=0.85, linewidths=0,
                            norm=self._density_norm(z_d), rasterized=True)
        except Exception as exc:
            print(f"[pseudocolor] density failed "
                  f"({type(exc).__name__}: {exc}); flat scatter fallback",
                  flush=True)
            self.ax.scatter(xs, ys, s=2, alpha=0.4,
                            color='steelblue', linewidths=0,
                            rasterized=True)

    @staticmethod
    def _density_norm(z):
        """A PowerNorm that spreads the colour map across the *populated*
        density range. Linear scaling lets the dense core saturate one end
        of the map and washes everything else to a single colour (the
        'even / flat' look on large samples); gamma<1 expands the low-density
        majority so population structure stays visible."""
        from matplotlib.colors import PowerNorm
        zmax = float(np.max(z)) if len(z) else 1.0
        return PowerNorm(gamma=0.4, vmin=0.0, vmax=max(zmax, 1e-9))

    def _density_scatter_truekde(self, xs, ys):
        """True scipy.stats.gaussian_kde path. O(n_src * n_query); we
        subsample both sides to stay tractable, and post a status warning
        so the user knows the trade-off."""
        from scipy.stats import gaussian_kde
        MAX_KDE_SRC = 15_000
        MAX_DISPLAY = 40_000
        rng = np.random.default_rng(42)
        if xs.size > MAX_KDE_SRC:
            src = rng.choice(xs.size, MAX_KDE_SRC, replace=False)
            xs_src, ys_src = xs[src], ys[src]
        else:
            xs_src, ys_src = xs, ys
        if xs.size > MAX_DISPLAY:
            disp = rng.choice(xs.size, MAX_DISPLAY, replace=False)
            xs_d, ys_d = xs[disp], ys[disp]
        else:
            xs_d, ys_d = xs, ys
        if xs.size > MAX_KDE_SRC:
            try:
                self.status_var.set(
                    f"True KDE on {xs_src.size:,} source / {xs_d.size:,} "
                    f"display events (subsampled from {xs.size:,}).")
            except Exception:
                pass
        try:
            kernel = gaussian_kde(np.vstack([xs_src, ys_src]))
            z      = kernel(np.vstack([xs_d, ys_d]))
            order  = z.argsort()
            self.ax.scatter(xs_d[order], ys_d[order], c=z[order],
                            cmap='jet', s=2, alpha=0.7, linewidths=0,
                            norm=self._density_norm(z), rasterized=True)
        except Exception as exc:
            print(f"[pseudocolor/KDE] failed "
                  f"({type(exc).__name__}: {exc}); flat scatter fallback",
                  flush=True)
            self.ax.scatter(xs_d, ys_d, s=2, alpha=0.4,
                            color='steelblue', linewidths=0,
                            rasterized=True)

    def _plot_contour(self, samples, x, y):
        """Smoothed density contours with outlier scatter underneath.

        Same O(n) histogram-based density as the pseudocolor path —
        gaussian_kde is overkill for flow data and doesn't scale.

        Each sample contributes:
          • A 128×128 2D histogram density, smoothed with gaussian_filter,
            rendered as 8 contour levels from 5% → 95% of peak so the
            outer line traces the population edge, not just the dense core.
          • A faint per-event scatter beneath the lines (master 'Contour
            scatter' toggle). Events below the lowest contour level are
            'outliers'; the 'Outliers' sub-toggle gates just those, so the
            scatter can show the within-population points only.
        """
        if not y:
            return
        from scipy.ndimage import gaussian_filter

        GRID            = 128
        SMOOTH_SIGMA    = 1.5
        LEVELS_FROM     = 0.05
        LEVELS_TO       = 0.95
        N_LEVELS        = 8
        MAX_OUTLIER_PTS = 30_000

        rng = np.random.default_rng(42)

        for name in samples:
            df = self._get_df(name, x, y)
            if df.empty or x not in df.columns or y not in df.columns:
                continue
            xv = np.asarray(df[x].values, dtype=float)
            yv = np.asarray(df[y].values, dtype=float)
            finite = np.isfinite(xv) & np.isfinite(yv)
            xv = xv[finite]; yv = yv[finite]
            if xv.size < 10:
                print(f"[contour] {name}: only {xv.size} finite points — skipped",
                      flush=True)
                continue
            try:
                xmin, xmax = float(xv.min()), float(xv.max())
                ymin, ymax = float(yv.min()), float(yv.max())
                if xmin == xmax or ymin == ymax:
                    print(f"[contour] {name}: degenerate range — skipped",
                          flush=True)
                    continue

                # 2% padding so outliers don't sit on the axis edge.
                xpad = (xmax - xmin) * 0.02
                ypad = (ymax - ymin) * 0.02
                xmin -= xpad; xmax += xpad
                ymin -= ypad; ymax += ypad

                color = self._color_for(name)

                # 1) Histogram-based density on the FULL population
                #    (O(n) — no subsampling needed). Bin in each axis's
                #    display space so the grid (and the contour lines) are
                #    even on a log/symlog axis.
                x_edges = self._axis_bin_edges(xv, x, GRID)
                y_edges = self._axis_bin_edges(yv, y, GRID)
                hist, x_edges, y_edges = np.histogram2d(
                    xv, yv, bins=[x_edges, y_edges])
                hist = gaussian_filter(hist, sigma=SMOOTH_SIGMA)
                fmax = float(hist.max())
                if fmax <= 0:
                    print(f"[contour] {name}: zero-density grid — skipped",
                          flush=True)
                    continue

                # 2) Scatter beneath the contours (master 'Contour scatter'
                #    toggle). Each event's density classifies it as inside the
                #    contoured population (>= the lowest contour level) or an
                #    outlier below it; the 'Outliers' sub-toggle gates only the
                #    latter, so it can show just the within-population points.
                show_scatter = (getattr(self, 'contour_scatter_var', None)
                                is None or self.contour_scatter_var.get())
                show_outliers = (getattr(self, 'contour_outliers_var', None)
                                 is None or self.contour_outliers_var.get())
                if show_scatter:
                    nbx, nby = len(x_edges) - 1, len(y_edges) - 1
                    ex = np.clip(np.searchsorted(x_edges, xv, side='right') - 1,
                                 0, nbx - 1)
                    ey = np.clip(np.searchsorted(y_edges, yv, side='right') - 1,
                                 0, nby - 1)
                    zev = hist[ex, ey]
                    keep = (np.ones(xv.size, dtype=bool) if show_outliers
                            else (zev >= fmax * LEVELS_FROM))
                    sx, sy = xv[keep], yv[keep]
                    if sx.size > MAX_OUTLIER_PTS:
                        out_idx = rng.choice(sx.size, MAX_OUTLIER_PTS,
                                             replace=False)
                        sx, sy = sx[out_idx], sy[out_idx]
                    if sx.size:
                        self.ax.scatter(sx, sy, s=1.5, alpha=0.18,
                                        color=color, linewidths=0,
                                        rasterized=True)

                # 3) Convert edges to centres for matplotlib.contour, then
                #    surround the density with a ring of zeros so every level
                #    forms a CLOSED loop (a population running to the binning
                #    edge would otherwise produce open contours).
                xc = 0.5 * (x_edges[:-1] + x_edges[1:])
                yc = 0.5 * (y_edges[:-1] + y_edges[1:])
                hist = np.pad(hist, 1, mode='constant', constant_values=0.0)
                xc = np.concatenate([[xc[0] - (xc[1] - xc[0])], xc,
                                     [xc[-1] + (xc[-1] - xc[-2])]])
                yc = np.concatenate([[yc[0] - (yc[1] - yc[0])], yc,
                                     [yc[-1] + (yc[-1] - yc[-2])]])
                xx, yy = np.meshgrid(xc, yc, indexing='ij')
                levels = np.linspace(fmax * LEVELS_FROM,
                                     fmax * LEVELS_TO,
                                     N_LEVELS)
                self.ax.contour(xx, yy, hist, levels=levels,
                                colors=[color], linewidths=1.1, alpha=0.9)

                # Legend stub.
                self.ax.plot([], [], color=color, label=name)
            except Exception as exc:
                import traceback
                print(f"[contour] {name}: {type(exc).__name__}: {exc}",
                      flush=True)
                traceback.print_exc()
                raise
        if len(samples) > 1:
            self.ax.legend(fontsize=8, loc='best')

    def _removed_events(self, name, x, y):
        """The events the auto-clean recipe REMOVES for ``name``, as a
        DataFrame carrying the (aliased) plot columns. Computed on the FULL
        sample — uncapped and ungated — so a small error rate isn't
        subsampled away before it can be shown. ``None`` when the sample has
        no auto-clean gate or nothing is removed."""
        s = self._samples.get(name)
        if s is None:
            return None
        df = s.data
        alias = self._axis_alias_for_sample(s, [x, y])
        if alias:
            df = df.assign(**{chosen: df[own] for chosen, own in alias.items()})
        cols = [c for c in (x, y) if c and c in df.columns]
        if not cols:
            return None
        df = df.dropna(subset=cols)
        overrides = self._autoclean_overrides(name, df)
        if not overrides:
            return None
        keep = np.ones(len(df), dtype=bool)
        for m in overrides.values():
            keep &= np.asarray(m, dtype=bool)
        removed = df[~keep]
        return removed if not removed.empty else None

    def _overlay_removed_events(self, samples, x, y, mode):
        """Draw the auto-clean-removed events on TOP of the current plot in
        red, so cleaning artefacts stay visible against the full sample even
        at a tiny error rate. Bypasses the display cap (surfacing the few
        dropped events is the whole point). Toggled by ``show_removed_var``.

        Scatter modes overlay the removed events as red dots; histogram mode
        overlays their channel distribution as a red curve scaled to the axis
        height (location, not magnitude — labelled as such)."""
        if not (getattr(self, 'show_removed_var', None)
                and self.show_removed_var.get()):
            return
        RED = '#e8000b'

        if mode == 'histogram':
            xs = []
            for name in samples:
                rem = self._removed_events(name, x, None)
                if rem is not None and x in rem.columns:
                    xs.append(np.asarray(rem[x].values, dtype=float))
            xs = np.concatenate(xs) if xs else np.array([])
            xs = xs[np.isfinite(xs)]
            if xs.size == 0:
                return
            from scipy.ndimage import gaussian_filter1d
            xlo, xhi = self.ax.get_xlim()
            _, ytop = self.ax.get_ylim()
            NBINS = 256
            edges = np.asarray(self._screen_uniform_edges(
                x, min(xlo, xhi), max(xlo, xhi), NBINS, data_sample=xs),
                dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            counts = np.histogram(xs, bins=edges)[0].astype(float)
            sigma = float(np.clip(np.sqrt(NBINS / max(xs.size, 1e-6)) * 1.5,
                                  1.0, 4.0))
            sm = gaussian_filter1d(counts, sigma=sigma, mode='constant')
            peak = float(sm.max())
            if peak <= 0:
                return
            # Scale so the removed-event profile peaks at ~85% of the axis —
            # visible no matter how few were removed (shows WHERE, not height).
            y_ov = sm * (0.85 * ytop / peak)
            self.ax.fill_between(centers, y_ov, color=RED, alpha=0.22,
                                 linewidth=0, zorder=5)
            self.ax.plot(centers, y_ov, color=RED, linewidth=1.5, zorder=6,
                         label=f'cleaned-out (n={xs.size:,}, location)')
            self.ax.legend(fontsize=8, loc='best')
            return

        # Scatter modes (dot / pseudocolor / contour).
        # Each removed event is coloured by the cleaning method that dropped
        # it (each method = a distinct "section" / pullable population), and
        # the overlay is SUBSAMPLED to the same fraction the main plot shows
        # (shown/total) so the red layer's density stays proportionate to the
        # visible sample instead of over-dominating it.
        import matplotlib.patches as mpatches
        rng = np.random.default_rng(42)
        order = list(self._METHOD_COLORS.keys())
        groups: dict = {}          # method_key -> [xs_arrays], [ys_arrays]
        full_counts: dict = {}     # method_key -> total removed (full sample)
        for name in samples:
            rem = self._removed_events(name, x, y)
            if rem is None or x not in rem.columns or not y \
                    or y not in rem.columns:
                continue
            # Per-event method attribution (first enabled method, recipe order).
            method_masks = self._autoclean_method_masks(name)
            label = np.full(len(rem), '', dtype=object)
            for key in order + [k for k in method_masks if k not in order]:
                ser = method_masks.get(key)
                if ser is None:
                    continue
                m = ser.reindex(rem.index, fill_value=False).to_numpy()
                take = m & (label == '')
                label[take] = key
            rx = np.asarray(rem[x].values, dtype=float)
            ry = np.asarray(rem[y].values, dtype=float)
            fin = np.isfinite(rx) & np.isfinite(ry)
            rx, ry, lab = rx[fin], ry[fin], label[fin]
            for key in np.unique(lab):
                full_counts[key] = full_counts.get(key, 0) + int((lab == key).sum())
            # Proportional subsample to the displayed fraction of this sample.
            shown, total = self._sample_display_count(name)
            frac = (shown / total) if total else 1.0
            nrem = rx.size
            k = int(round(frac * nrem))
            if nrem and k == 0:
                k = min(nrem, 25)          # keep a real error rate visible
            if 0 < k < nrem:
                sel = rng.choice(nrem, k, replace=False)
                rx, ry, lab = rx[sel], ry[sel], lab[sel]
            for key in np.unique(lab):
                gx, gy = groups.setdefault(key, ([], []))
                mk = lab == key
                gx.append(rx[mk]); gy.append(ry[mk])
        if not groups:
            return
        handles = []
        for key in order + [k for k in groups if k not in order]:
            if key not in groups:
                continue
            gx = np.concatenate(groups[key][0])
            gy = np.concatenate(groups[key][1])
            if gx.size == 0:
                continue
            color = self._METHOD_COLORS.get(key, RED)
            self.ax.scatter(gx, gy, s=7, c=color, alpha=0.85, linewidths=0,
                            marker='o', zorder=5, rasterized=True)
            lbl = key or 'removed'
            handles.append(mpatches.Patch(
                color=color, label=f'{lbl} (n={full_counts.get(key, gx.size):,})'))
        if handles:
            self.ax.legend(handles=handles, fontsize=8, loc='best',
                           framealpha=0.85, title='cleaned-out')

    # ── Backgating ──────────────────────────────────────────────────────────
    _BACKGATE_COLORS = ['#e8000b', '#1ac938', '#023eff', '#ff7c00',
                        '#8b2be2', '#f14cc1', '#00d7ff', '#ffb000']

    def _backgate_selected(self):
        """Set the backgate targets from the selected gate row(s): their
        populations get projected onto the current plot. Multi-select adds
        several, each its own colour."""
        targets = []
        for iid in self.gate_tv.selection():
            p = self._parse_iid(iid)
            if p and p[0] == 'gate':
                targets.append((p[1], p[2]))
        if not targets:
            self.status_var.set("Select a gate/population to backgate.")
            return
        self._backgate = targets
        self.status_var.set(
            f"Backgating {len(targets)} population(s) — shown in colour on the "
            f"plot. Right-click → Clear backgating to remove.")
        self._schedule_replot(0)

    def _clear_backgate(self):
        self._backgate = []
        self.status_var.set("Backgating cleared.")
        self._schedule_replot(0)

    def _overlay_backgate(self, samples, x, y):
        """Project each backgate target population onto the current plot in its
        own colour, on top. The population's cumulative gate mask is computed
        on its sample's full data, then those events are drawn at the current
        x/y — so you can see where a downstream population sits on any axes."""
        targets = getattr(self, '_backgate', None)
        if not targets:
            return
        import matplotlib.patches as mpatches

        from .pipeline import cumulative_gate_mask
        rng = np.random.default_rng(42)
        CAP = 60_000
        handles = []
        for i, (sname, gid) in enumerate(targets):
            s = self._samples.get(sname)
            if s is None:
                continue
            sample_gates = self._sample_gates.get(sname, {})
            if gid not in sample_gates:
                continue
            df = s.data
            alias = self._axis_alias_for_sample(s, [x, y])
            if alias:
                df = df.assign(**{ch: df[own] for ch, own in alias.items()})
            cols = [c for c in (x, y) if c and c in df.columns]
            if not cols:
                continue
            try:
                overrides = self._autoclean_overrides(sname, df)
                mask = np.asarray(cumulative_gate_mask(
                    sample_gates, gid, df, overrides=overrides), dtype=bool)
            except Exception as exc:
                print(f"[backgate] {sname}/{gid}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            sub = df[mask].dropna(subset=cols)
            ntot = len(sub)
            if ntot == 0:
                continue
            if ntot > CAP:
                sub = sub.sample(CAP, random_state=42)
            color = self._BACKGATE_COLORS[i % len(self._BACKGATE_COLORS)]
            label = self._population_path(sample_gates, gid)
            if len(samples) > 1:
                label = f'{sname} › {label}'
            if y:
                self.ax.scatter(sub[x].to_numpy(dtype=float),
                                sub[y].to_numpy(dtype=float),
                                s=8, c=color, alpha=0.9, linewidths=0,
                                marker='o', zorder=6, rasterized=True)
            else:
                # Histogram mode: rug ticks at the population's x-values.
                xv = sub[x].to_numpy(dtype=float)
                xv = xv[np.isfinite(xv)]
                if xv.size > CAP:
                    xv = rng.choice(xv, CAP, replace=False)
                ybot, ytop = self.ax.get_ylim()
                self.ax.vlines(xv, ybot, ybot + (ytop - ybot) * 0.04,
                               color=color, alpha=0.5, linewidth=0.5,
                               zorder=6)
            handles.append(mpatches.Patch(
                color=color, label=f'{label} (n={int(mask.sum()):,})'))
        if handles:
            self.ax.legend(handles=handles, fontsize=8, loc='best',
                           framealpha=0.85, markerscale=2, title='backgate')

    def _plot_histogram(self, samples, x):
        """Overlay per-sample density histograms of channel ``x``.

        Two failure modes the naïve ``ax.hist(df[x].values, bins=200,
        density=True)`` hits and that this implementation works around:

        1. **Non-finite values.** If any sample's column contains NaN /
           ±inf, matplotlib's histogram silently skips the offending
           bin or renders empty. Filter them up-front.
        2. **Vastly different ranges across samples / channels.** When
           sample A has data on logicle scale (~0–1) and sample B has
           raw scale (0–262144), matplotlib auto-ranges to the union →
           sample A collapses into a single bin at zero, sample B's
           bars become invisibly short. Use a robust per-sample percentile
           clip (0.1–99.9) unioned across samples, then pin every
           sample's bins to the same edges so the overlay is comparable.
        """
        clean_series = []
        for name in samples:
            df = self._get_df(name, x, None, for_hist=True)
            if x not in df.columns or df.empty:
                continue
            arr = np.asarray(df[x].values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            clean_series.append((name, arr))

        if not clean_series:
            self.ax.text(0.5, 0.5,
                         f'No finite data for "{x}" — nothing to plot.',
                         ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=10, color='#888')
            return

        # Union of robust per-sample [p0.1, p99.9] ranges, then a small
        # symmetric pad so the tails are visible. Falls back to (min,max)
        # for very small samples.
        lo, hi = np.inf, -np.inf
        for _name, arr in clean_series:
            if arr.size >= 20:
                a, b = np.percentile(arr, (0.1, 99.9))
            else:
                a, b = float(arr.min()), float(arr.max())
            if a < lo: lo = float(a)
            if b > hi: hi = float(b)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # Degenerate (constant) — fall back to ±1 around the value.
            lo, hi = lo - 1.0, lo + 1.0
        else:
            pad = (hi - lo) * 0.02
            lo, hi = lo - pad, hi + pad

        # Bin edges spaced uniformly in SCREEN space for the channel's
        # display scale (composite FuncScale view for nonlinear-baked
        # channels, else log/linear), so bins look even on the axis.
        NBINS = 256
        bin_edges = np.asarray(self._screen_uniform_edges(
            x, lo, hi, NBINS, data_sample=clean_series[0][1]), dtype=float)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Y-axis mode (user-selectable):
        #   Fraction (default) — events per bin ÷ sample total (sums to 1).
        #   Count              — raw events per bin.
        #   % of Max           — each curve scaled so its tallest bin = 100.
        # Counts (not density=True): density divides by the bar's DATA-space
        # width, but bins are uniform in SCREEN space, so on a log/symlog/
        # composite axis their data widths vary enormously and density would
        # crush the bright tail while spiking the dim peak. Raw counts keep
        # the shape true and overlaid samples comparable.
        #
        # Each profile is rendered as a kernel-SMOOTHED filled curve rather
        # than raw step bars — the bars read as chunky/jagged, the smoothed
        # curve matches the FlowJo look (and the now-smooth pseudocolor).
        # Smoothing is adaptive: sparse populations get a wider kernel.
        from scipy.ndimage import gaussian_filter1d
        ymode = (self.hist_y_mode.get()
                 if getattr(self, 'hist_y_mode', None) is not None
                 else 'Fraction')
        for name, arr in clean_series:
            counts, _ = np.histogram(arr, bins=bin_edges)
            counts = counts.astype(float)
            per_bin = arr.size / float(NBINS)
            sigma = float(np.clip(np.sqrt(1.0 / max(per_bin, 1e-6)) * 1.5,
                                  1.0, 4.0))
            sm = gaussian_filter1d(counts, sigma=sigma, mode='constant')
            if ymode == 'Count':
                y = sm
            elif ymode == '% of Max':
                peak = float(sm.max()) if sm.size else 0.0
                y = sm * (100.0 / peak) if peak > 0 else sm
            else:   # Fraction
                y = sm / arr.size if arr.size else sm
            color = self._color_for(name)
            self.ax.fill_between(centers, y, color=color, alpha=0.30,
                                 linewidth=0)
            self.ax.plot(centers, y, color=color, linewidth=1.4, label=name)

        self.ax.set_ylabel(
            {'Count': 'count', '% of Max': '% of max'}.get(ymode, 'fraction'))
        self.ax.set_xlim(lo, hi)
        if len(clean_series) > 1:
            self.ax.legend(fontsize=8, loc='best')

    @staticmethod
    def _hist_bin_edges(lo, hi, scale, n_bins=200):
        """Return ``n_bins + 1`` bin edges between ``lo`` and ``hi``,
        spaced linearly or logarithmically depending on the axis scale.

        Returns a Python list (matplotlib's hist stubs declare bins as
        ``Sequence[float]``, not ndarray, even though both work at runtime).

        Scale handling:
          - ``'linear'`` / ``'symlog'``: linear spacing. Symlog uses a
            linear bin grid because its display transform is linear-
            near-zero and only compresses the tails — true log-spaced
            bins would over-narrow the centre.
          - ``'log'``: log-spaced edges. ``lo`` is clamped to a small
            positive value when non-positive, so a channel with a few
            negative outliers still produces a usable histogram.
            Falls back to linear spacing when the clamped range is
            degenerate (``hi <= lo``).
        """
        lo = float(lo)
        hi = float(hi)
        n_bins = int(n_bins)
        if scale == 'log':
            # Floor for non-positive lo. Anything below this is folded
            # into the leftmost bin.
            lo_pos = max(lo, max(hi * 1e-6, 1e-12))
            if hi <= lo_pos:
                return np.linspace(lo, hi, n_bins + 1).tolist()
            edges = np.logspace(np.log10(lo_pos), np.log10(hi), n_bins + 1)
            return edges.tolist()
        return np.linspace(lo, hi, n_bins + 1).tolist()

    # ── Gates (draggable threshold lines + shape overlays) ───────────────

    def _draw_gates(self, x, y):
        """Render every gate that intersects the current axes, in the gate's
        own colour. Lines for 1D gates (vertical when channel == x;
        horizontal when == y); a matplotlib Patch for rect/polygon."""
        import matplotlib.patches as mpatches

        self._vlines, self._hlines, self._shape_artists = {}, {}, {}

        for gid, g in self._gates.items():
            color = g.get('color', 'red')
            k = g.get('kind')
            if k == 'threshold':
                ch = g['channel']
                if ch == x:
                    self._vlines[gid] = self.ax.axvline(
                        float(g['value']), color=color, lw=1.3,
                        ls='--', alpha=0.85)
                elif ch == y:
                    self._hlines[gid] = self.ax.axhline(
                        float(g['value']), color=color, lw=1.3,
                        ls='--', alpha=0.85)
            elif k == 'interval':
                ch = g['channel']
                lo, hi = float(g['lo']), float(g['hi'])
                if ch == x:
                    self._vlines[f'{gid}:lo'] = self.ax.axvline(
                        lo, color=color, lw=1.2, ls='--', alpha=0.85)
                    self._vlines[f'{gid}:hi'] = self.ax.axvline(
                        hi, color=color, lw=1.2, ls='--', alpha=0.85)
                elif ch == y:
                    self._hlines[f'{gid}:lo'] = self.ax.axhline(
                        lo, color=color, lw=1.2, ls='--', alpha=0.85)
                    self._hlines[f'{gid}:hi'] = self.ax.axhline(
                        hi, color=color, lw=1.2, ls='--', alpha=0.85)
            elif k == 'rect':
                if g.get('x_channel') == x and g.get('y_channel') == y:
                    x0, x1 = sorted([float(g['x0']), float(g['x1'])])
                    y0, y1 = sorted([float(g['y0']), float(g['y1'])])
                    patch = mpatches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        fill=False, edgecolor=color, linewidth=1.3,
                        linestyle='--', alpha=0.9)
                    self.ax.add_patch(patch)
                    self._shape_artists[gid] = patch
            elif k == 'polygon':
                if g.get('x_channel') == x and g.get('y_channel') == y:
                    verts = np.asarray(g['vertices'], dtype=float)
                    if verts.ndim == 2 and verts.shape[1] == 2 and len(verts) >= 3:
                        patch = mpatches.Polygon(
                            verts, closed=True, fill=False,
                            edgecolor=color, linewidth=1.3,
                            linestyle='--', alpha=0.9)
                        self.ax.add_patch(patch)
                        self._shape_artists[gid] = patch
            elif k == 'ellipsoid':
                if g.get('x_channel') == x and g.get('y_channel') == y:
                    patch = self._ellipse_patch(g, color)
                    if patch is not None:
                        self.ax.add_patch(patch)
                        self._shape_artists[gid] = patch
                        # Rotation-grip marker just beyond the rim, so the
                        # Edit tool's rotate handle is discoverable.
                        geom = self._ellipse_geom(g)
                        if geom is not None:
                            _c, _inv, _r0, (hx, hy) = geom
                            (handle,) = self.ax.plot(
                                [hx], [hy], marker='o', markersize=5,
                                markerfacecolor=color, markeredgecolor='white',
                                markeredgewidth=0.6, linestyle='None',
                                alpha=0.9, zorder=6)
                            self._shape_artists[f'{gid}:rot'] = handle

    @staticmethod
    def _ellipse_params(gate):
        """Derive (cx, cy, width, height, angle_deg) for matplotlib's
        Ellipse from an ellipsoid gate's (mean, cov, distance_sq).

        The gate boundary is the level set
        (p-µ)ᵀ Σ⁻¹ (p-µ) = distance_sq. Eigendecomposing Σ gives the
        principal axis directions (eigenvectors) and the squared
        semi-axis scale (eigenvalues); the on-screen semi-axis length
        along principal axis i is sqrt(eigval_i · distance_sq).
        Returns None if the gate is malformed / degenerate.
        """
        try:
            mean = np.asarray(gate['mean'], dtype=float)
            cov  = np.asarray(gate['cov'], dtype=float)
            dist_sq = float(gate.get('distance_sq', 4.0))
            if mean.shape != (2,) or cov.shape != (2, 2):
                return None
            # eigh: symmetric matrix → real eigenpairs, ascending eigvals.
            eigvals, eigvecs = np.linalg.eigh(cov)
            if np.any(eigvals <= 0) or dist_sq <= 0:
                return None
            # Full axis lengths (diameter) = 2 · sqrt(eigval · dist).
            semis = np.sqrt(eigvals * dist_sq)
            width  = 2.0 * float(semis[0])
            height = 2.0 * float(semis[1])
            # Angle of the FIRST eigenvector (matches width's axis).
            v = eigvecs[:, 0]
            angle = float(np.degrees(np.arctan2(v[1], v[0])))
            return float(mean[0]), float(mean[1]), width, height, angle
        except Exception:
            return None

    def _ellipse_patch(self, gate, color):
        """Build a dashed matplotlib Ellipse patch for an ellipsoid gate,
        or None if the geometry is degenerate."""
        import matplotlib.patches as mpatches
        params = self._ellipse_params(gate)
        if params is None:
            return None
        cx, cy, width, height, angle = params
        return mpatches.Ellipse(
            (cx, cy), width, height, angle=angle,
            fill=False, edgecolor=color, linewidth=1.3,
            linestyle='--', alpha=0.9)

    # ── Highlight overlay (3-way display mode == 'highlight') ────────────

    def _gates_topological_for(self, gates_dict):
        """Generic topo-sort over an arbitrary per-sample gates dict."""
        seen = set()
        out = []

        def visit(gid):
            if gid in seen or gid not in gates_dict:
                return
            seen.add(gid)
            parent = gates_dict[gid].get('parent_id')
            if parent and parent in gates_dict and parent not in seen:
                visit(parent)
            out.append(gid)

        for gid in gates_dict:
            visit(gid)
        return out

    def _draw_highlight_overlays(self, samples, x, y):
        """For EVERY enabled gate (root, intermediate, or leaf) in each
        sample's own gate tree, overlay its cumulative-chain events on
        the base plot in the gate's colour. Parents drawn first so
        children's smaller, more-specific populations sit on top —
        FlowJo-style nested-population rendering. No-op outside
        'highlight' mode."""
        if (not hasattr(self, 'gate_display_var')
                or self.gate_display_var.get() != 'highlight'):
            return
        from .pipeline import cumulative_gate_mask

        is_hist = (y is None)

        for name in samples:
            sample_gates = self._sample_gates.get(name, {})
            if not sample_gates:
                continue
            order = [gid for gid in self._gates_topological_for(sample_gates)
                     if sample_gates[gid].get('enabled', True)]
            if not order:
                continue
            df = self._get_df(name, x, y)
            if df.empty or x not in df.columns or (y and y not in df.columns):
                continue
            # Use the same full-data auto-clean masks as filter mode so a
            # cleaning gate flags the SAME events here (otherwise the time-
            # binned methods would recompute on this downsampled / dropna'd
            # subset and disagree with the filtered view).
            overrides = self._autoclean_overrides(name, df)
            for gid in order:
                mask = cumulative_gate_mask(sample_gates, gid, df,
                                            overrides=overrides)
                if not mask.any():
                    continue
                color = sample_gates[gid].get('color', '#e6194b')
                lbl = f'{name}:{gid}' if len(samples) > 1 else f'gate {gid}'
                if is_hist:
                    # Strip non-finite values + reuse the base axes' x-range
                    # so the highlight overlays line up with the underlying
                    # histogram bins. Skip if no finite values remain
                    # (rare but possible after a tight gate).
                    arr = np.asarray(df[x].values[mask], dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        continue
                    xlo, xhi = self.ax.get_xlim()
                    # Reuse the same scale-aware spacing + kernel smoothing as
                    # the base histogram so the highlight overlay lines up and
                    # reads as a smooth curve, not chunky step bars.
                    NBINS = 256
                    bin_edges = np.asarray(self._screen_uniform_edges(
                        x, xlo, xhi, NBINS, data_sample=arr), dtype=float)
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    from scipy.ndimage import gaussian_filter1d
                    counts, _ = np.histogram(arr, bins=bin_edges)
                    counts = counts.astype(float)
                    per_bin = arr.size / float(NBINS)
                    sigma = float(np.clip(
                        np.sqrt(1.0 / max(per_bin, 1e-6)) * 1.5, 1.0, 4.0))
                    sm = gaussian_filter1d(counts, sigma=sigma, mode='constant')
                    # Fraction per bin (bins are screen-uniform; density would
                    # crush the bright tail on a log/symlog/composite axis).
                    y = sm / arr.size if arr.size else sm
                    self.ax.fill_between(centers, y, color=color, alpha=0.40,
                                         linewidth=0)
                    self.ax.plot(centers, y, color=color, linewidth=1.3,
                                 label=lbl)
                else:
                    xv = np.asarray(df[x].values[mask])
                    yv = np.asarray(df[y].values[mask])
                    self.ax.scatter(xv, yv, s=4, alpha=0.85,
                                    color=color, linewidths=0,
                                    rasterized=True, label=lbl)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(fontsize=8, loc='best', framealpha=0.85)

    def _color_for(self, name):
        """Palette color for a sample, assigned lazily on first display so
        undisplayed samples stay neutral (and don't burn palette slots)."""
        c = self._sample_colors.get(name)
        if c is None:
            idx = len(self._sample_colors) % len(self.SAMPLE_PALETTE)
            c = self.SAMPLE_PALETTE[idx]
            self._sample_colors[name] = c
        return c

    def _trial_members(self, trial):
        """Loaded samples belonging to ``trial``, in display order."""
        return [n for n in self._sample_order
                if n in self._samples and self._trial_for(n) == trial]

    def _is_comp(self, name):
        """Whether ``name`` belongs in the Comps subgroup — a manual drag
        override if present, else the name-based guess."""
        from .workspace import is_comp_sample
        if name in self._sample_is_comp:
            return bool(self._sample_is_comp[name])
        return is_comp_sample(name)

    def _subgroup_members(self, kind, trial):
        """Loaded samples in ``trial`` of the given subgroup ``kind`` —
        'comp' (compensation controls) or 'samp' (everything else)."""
        want_comp = (kind == 'comp')
        return [n for n in self._trial_members(trial)
                if self._is_comp(n) == want_comp]

    def _trial_for(self, name):
        return self._sample_trial.get(name, 'Trial')

    def _ordered_trials(self):
        """Trials that currently have at least one loaded sample, in first-seen
        order (with any stragglers not yet in _trial_order appended)."""
        loaded = [n for n in self._sample_order if n in self._samples]
        trials, seen = [], set()
        for t in self._trial_order:
            if any(self._trial_for(n) == t for n in loaded):
                trials.append(t)
                seen.add(t)
        for n in loaded:
            t = self._trial_for(n)
            if t not in seen:
                trials.append(t)
                seen.add(t)
        # Day-organised groups sort numerically (Day 0 < Day 3 < … < Day 15);
        # any non-day trials keep first-seen order, after the day groups.
        from .workspace import trial_day_number
        idx = {t: i for i, t in enumerate(trials)}
        return sorted(
            trials,
            key=lambda t: (0, trial_day_number(t), 0)
            if trial_day_number(t) is not None else (1, 0, idx[t]))

    def _refresh_gate_list(self):
        """Rebuild the samples-and-gates tree, grouped by trial:
        trial row → its samples → each sample's gate hierarchy. The trial row's
        ☑ column toggles plot-display for every sample in that trial; clicking
        the disclosure triangle collapses/expands the trial. iid encoding lives
        with `_trial_iid` / `_sample_iid` / `_gate_iid`."""

        sel_iid = None
        cur = self.gate_tv.selection()
        if cur:
            sel_iid = cur[0]

        # Preserve which trials the user had collapsed across the rebuild,
        # plus each Comps/Samples subgroup's open state (default: Samples open,
        # Comps collapsed).
        collapsed = set()
        sub_open = {}        # (kind, trial) -> bool
        for iid in self.gate_tv.get_children(''):
            p = self._parse_iid(iid)
            if p and p[0] == 'trial':
                if not self.gate_tv.item(iid, 'open'):
                    collapsed.add(iid)
                for sg_iid in self.gate_tv.get_children(iid):
                    sp = self._parse_iid(sg_iid)
                    if sp and sp[0] == 'subgroup':
                        sub_open[(sp[1], sp[2])] = bool(
                            self.gate_tv.item(sg_iid, 'open'))
        # Persist each auto-clean group's expand/collapse choice back into its
        # gate dict so a rebuild keeps the user's state (default: collapsed).
        for nm, gates in self._sample_gates.items():
            for gid, g in gates.items():
                if g.get('kind') == 'autoclean':
                    giid = self._gate_iid(nm, gid)
                    if self.gate_tv.exists(giid):
                        g['open'] = bool(self.gate_tv.item(giid, 'open'))
        for iid in self.gate_tv.get_children(''):
            self.gate_tv.delete(iid)

        self.gate_tv.tag_configure('trial_row', font=('TkDefaultFont', 9, 'bold'))
        self.gate_tv.tag_configure('subgroup_row', foreground='#555')

        def _agg_mark(names):
            en = [self._sample_plot_enabled.get(n, True) for n in names]
            return '☑' if all(en) else ('☐' if not any(en) else '▣')

        for trial in self._ordered_trials():
            members = [n for n in self._sample_order
                       if n in self._samples and self._trial_for(n) == trial]
            if not members:
                continue
            t_iid = self._trial_iid(trial)
            self.gate_tv.insert(
                '', 'end', iid=t_iid,
                text=f'▦ {trial}  ({len(members)})',
                values=(_agg_mark(members),),
                open=(t_iid not in collapsed), tags=('trial_row',))

            # Split into Comps vs Samples. Only introduce the subgroup headers
            # when comps are actually present (otherwise list samples directly
            # under the trial, as before). Samples first (expanded), Comps
            # second (collapsed by default).
            comps = [n for n in members if self._is_comp(n)]
            if comps:
                samps = [n for n in members if not self._is_comp(n)]
                for kind, sub, default_open, label in (
                        ('samp', samps, True, 'Samples'),
                        ('comp', comps, False, 'Comps')):
                    if not sub:
                        continue
                    sg_iid = self._subgroup_iid(kind, trial)
                    self.gate_tv.insert(
                        t_iid, 'end', iid=sg_iid,
                        text=f'{label}  ({len(sub)})',
                        values=(_agg_mark(sub),),
                        open=sub_open.get((kind, trial), default_open),
                        tags=('subgroup_row',))
                    for name in sub:
                        self._insert_sample_subtree(name, sg_iid)
            else:
                for name in members:
                    self._insert_sample_subtree(name, t_iid)

        if sel_iid:
            try:
                self.gate_tv.selection_set(sel_iid)
            except Exception:
                pass

    def _insert_sample_subtree(self, name, parent_iid):
        """Insert one sample row (with its event count) plus its full gate
        hierarchy under ``parent_iid`` (a trial or a Comps/Samples subgroup)."""
        from .pipeline import describe_gate
        sample_iid = self._sample_iid(name)
        plot_on = self._sample_plot_enabled.get(name, True)
        # Neutral until displayed; the swatch colour matches the plot overlay
        # only while the sample is actually shown.
        fg = self._color_for(name) if plot_on else '#000000'
        sample_tag = f'sample_col_{name}'
        self.gate_tv.tag_configure(
            sample_tag, foreground=fg, font=('TkDefaultFont', 9, 'bold'))
        shown, total = self._sample_display_count(name)
        cnt = (f'  ({shown:,}/{total:,})' if shown < total else f'  ({total:,})')
        self.gate_tv.insert(
            parent_iid, 'end', iid=sample_iid,
            text=f'■ {name}{cnt}',
            values=('☑' if plot_on else '☐',),
            open=True, tags=(sample_tag,))

        sample_gates = self._sample_gates.get(name, {})
        order        = self._sample_gate_order.get(name, [])
        inserted_gates = set()

        def insert_gate(gid, _name=name, _sg=sample_gates,
                        _siid=sample_iid, _ins=inserted_gates):
            # _ins default-binds the per-iteration set so the closure is fully
            # self-contained (silences B023).
            if gid in _ins:
                return
            g = _sg.get(gid)
            if g is None:
                return
            parent_gid = g.get('parent_id')
            if (parent_gid and parent_gid in _sg
                    and parent_gid not in _ins):
                insert_gate(parent_gid)
            if parent_gid and parent_gid in _ins:
                parent_tree_iid = self._gate_iid(_name, parent_gid)
            else:
                parent_tree_iid = _siid
            on = g.get('enabled', True)
            gate_color = g.get('color', '#000000')
            color_tag = f'col_{_name}_{gid}'
            self.gate_tv.tag_configure(color_tag, foreground=gate_color)
            tags = (color_tag,) if on else ('off',)
            # Auto-clean gates carry a "drops N (X%)" readout so the cleaning's
            # effect is visible without switching to Filter mode.
            ac_counts = (self._autoclean_counts(_name, gid)
                         if g.get('kind') == 'autoclean' else None)
            gate_text = describe_gate(g)
            if ac_counts is not None:
                gate_text += self._drop_suffix(ac_counts[1], ac_counts[0])
            self.gate_tv.insert(
                parent_tree_iid, 'end',
                iid=self._gate_iid(_name, gid),
                text=gate_text,
                values=('☑' if on else '☐',),
                open=g.get('open', True), tags=tags)
            _ins.add(gid)
            # An auto-clean gate is a GROUP: render one synthetic child row per
            # cleaning method (each toggleable). 'open' default False = collapsed.
            if g.get('kind') == 'autoclean':
                g_iid = self._gate_iid(_name, gid)
                total = ac_counts[0] if ac_counts else None
                per_method = ac_counts[2] if ac_counts else {}
                reasons = ac_counts[3] if ac_counts and len(ac_counts) > 3 else {}
                for m in g.get('methods', []):
                    mkey = m.get('key', '')
                    mon = m.get('enabled', True)
                    mtext = '   ' + (m.get('label') or mkey)
                    if per_method.get(mkey) is not None:
                        mtext += self._drop_suffix(per_method[mkey], total)
                        # Explain a silent 0-drop (only for enabled methods).
                        if mon and per_method[mkey] == 0 and reasons.get(mkey):
                            mtext += f'  ·  {reasons[mkey]}'
                    self.gate_tv.insert(
                        g_iid, 'end',
                        iid=self._method_iid(_name, gid, mkey),
                        text=mtext,
                        values=('☑' if mon else '☐',),
                        open=True, tags=(() if mon else ('off',)))

        for gid in order:
            insert_gate(gid)
        for gid in list(sample_gates):
            insert_gate(gid)

    def _hit_test(self, event):
        """Find a draggable handle near the cursor. Priority order:
          1. 1D axis lines (threshold / interval)
          2. Quadrant origin / x-axis / y-axis (so the central cross of a
             4-rect quadrant set wins over each rect's individual corner)
          3. Rect corners, then edges (non-quadrant rects)
          4. Polygon vertices, then edges (edge hits will INSERT a new
             vertex at the click point in _on_press).
        Returns a drag-state tuple or None. Tolerance is axis-fraction
        (2.5% of the current view span on each axis)."""
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return None
        xl, xh = self.ax.get_xlim()
        yl, yh = self.ax.get_ylim()
        tol = 0.025
        span_x = max(xh - xl, 1e-9)
        span_y = max(yh - yl, 1e-9)
        tol_x = span_x * tol
        tol_y = span_y * tol
        cx, cy = float(event.xdata), float(event.ydata)

        # 1) 1D axis lines (existing behaviour)
        for key, line in self._vlines.items():
            val = line.get_xdata()[0]
            if abs(cx - val) < tol_x:
                return ('v', key)
        for key, line in self._hlines.items():
            val = line.get_ydata()[0]
            if abs(cy - val) < tol_y:
                return ('h', key)

        x_ch = self._resolve_channel(self.x_combo.get())
        y_ch = self._resolve_channel(self.y_combo.get())

        # 2) Quadrant-set handles. Drag the centre to move both axes;
        #    drag a divider line to move only that axis.
        seen_qs = set()
        for _gid, g in self._gates.items():
            qs = g.get('quad_set')
            if not qs or qs in seen_qs:
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            seen_qs.add(qs)
            x_o = g.get('quad_origin_x')
            y_o = g.get('quad_origin_y')
            if x_o is None or y_o is None:
                continue
            x_o, y_o = float(x_o), float(y_o)
            if abs(cx - x_o) < tol_x and abs(cy - y_o) < tol_y:
                return ('quad_origin', qs)
            if abs(cx - x_o) < tol_x:
                return ('quad_x', qs)
            if abs(cy - y_o) < tol_y:
                return ('quad_y', qs)

        # 3) Non-quadrant rect corners / edges.
        for gid, g in self._gates.items():
            if g.get('kind') != 'rect' or g.get('quad_set'):
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            x0, x1 = sorted([float(g['x0']), float(g['x1'])])
            y0, y1 = sorted([float(g['y0']), float(g['y1'])])
            for cn, hx, hy in (('bl', x0, y0), ('br', x1, y0),
                                ('tl', x0, y1), ('tr', x1, y1)):
                if abs(cx - hx) < tol_x and abs(cy - hy) < tol_y:
                    return ('rect_corner', gid, cn)
            if abs(cy - y0) < tol_y and x0 <= cx <= x1:
                return ('rect_edge', gid, 'bottom')
            if abs(cy - y1) < tol_y and x0 <= cx <= x1:
                return ('rect_edge', gid, 'top')
            if abs(cx - x0) < tol_x and y0 <= cy <= y1:
                return ('rect_edge', gid, 'left')
            if abs(cx - x1) < tol_x and y0 <= cy <= y1:
                return ('rect_edge', gid, 'right')

        # 4) Polygon vertices, then edges.
        for gid, g in self._gates.items():
            if g.get('kind') != 'polygon':
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            verts = g.get('vertices') or []
            for i, (vx, vy) in enumerate(verts):
                if abs(cx - float(vx)) < tol_x and abs(cy - float(vy)) < tol_y:
                    return ('poly_vertex', gid, i)
            n = len(verts)
            for i in range(n):
                ax, ay = float(verts[i][0]),       float(verts[i][1])
                bx, by = float(verts[(i+1) % n][0]), float(verts[(i+1) % n][1])
                if self._point_segment_dist(cx, cy, ax, ay, bx, by,
                                            span_x, span_y) < tol:
                    return ('poly_edge', gid, i)

        # 4b) Ellipsoid: rotation handle, then rim (resize), then interior
        #     (translate). Mahalanobis radius md = sqrt((p-µ)ᵀ Σ⁻¹ (p-µ));
        #     the rim is md == sqrt(distance_sq).
        for gid, g in self._gates.items():
            if g.get('kind') != 'ellipsoid':
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            geom = self._ellipse_geom(g)
            if geom is None:
                continue
            (mx, my), inv, r0, (hx, hy) = geom
            # Rotation handle (small marker offset beyond the rim).
            if abs(cx - hx) < tol_x and abs(cy - hy) < tol_y:
                return ('ellipse_rotate', gid)
            d = np.array([cx - mx, cy - my])
            md = float(np.sqrt(max(d @ inv @ d, 0.0)))
            if 0.6 * r0 <= md <= 1.5 * r0:
                return ('ellipse_rim', gid)
            # interior handled in section 5

        # 5) Interior translate. Lowest priority so corners/edges/vertices
        #    above still win when the cursor is near a handle. Quadrant
        #    rects are excluded — the quad-origin / quad-x / quad-y
        #    handles already cover translation for those.
        for gid, g in self._gates.items():
            if g.get('quad_set'):
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            kind = g.get('kind')
            if kind == 'rect':
                x0, x1 = sorted([float(g['x0']), float(g['x1'])])
                y0, y1 = sorted([float(g['y0']), float(g['y1'])])
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    return ('rect_translate', gid)
            elif kind == 'polygon':
                verts = g.get('vertices') or []
                if len(verts) >= 3:
                    try:
                        from matplotlib.path import Path as _MplPath
                        if _MplPath(np.asarray(verts, dtype=float)
                                    ).contains_point((cx, cy)):
                            return ('poly_translate', gid)
                    except Exception:
                        pass
            elif kind == 'ellipsoid':
                geom = self._ellipse_geom(g)
                if geom is not None:
                    (mx, my), inv, r0, _h = geom
                    d = np.array([cx - mx, cy - my])
                    md = float(np.sqrt(max(d @ inv @ d, 0.0)))
                    if md < 0.6 * r0:
                        return ('ellipse_translate', gid)
        return None

    @staticmethod
    def _ellipse_geom(g):
        """Geometry an ellipsoid gate needs for hit-testing / editing:
          ((mean_x, mean_y), Σ⁻¹, r0, (handle_x, handle_y))
        where r0 = sqrt(distance_sq) is the Mahalanobis rim radius and
        the handle sits just beyond the rim along the +height axis (the
        rotation grip). Returns None if the gate is degenerate."""
        try:
            mean = np.asarray(g['mean'], dtype=float)
            cov  = np.asarray(g['cov'], dtype=float)
            dist_sq = float(g.get('distance_sq', 4.0))
            if mean.shape != (2,) or cov.shape != (2, 2) or dist_sq <= 0:
                return None
            inv = np.linalg.inv(cov)
            r0 = float(np.sqrt(dist_sq))
            # Handle direction: the 2nd eigenvector (the 'height' axis),
            # placed at 1.18× the rim so it clears the dashed outline.
            eigvals, eigvecs = np.linalg.eigh(cov)
            if np.any(eigvals <= 0):
                return None
            v = eigvecs[:, 1]
            semi_h = float(np.sqrt(eigvals[1] * dist_sq))
            hx = float(mean[0] + v[0] * semi_h * 1.18)
            hy = float(mean[1] + v[1] * semi_h * 1.18)
            return (float(mean[0]), float(mean[1])), inv, r0, (hx, hy)
        except Exception:
            return None

    @staticmethod
    def _point_segment_dist(px, py, ax, ay, bx, by, span_x, span_y):
        """Axis-fraction distance from point (px,py) to the segment
        (ax,ay)-(bx,by). Both axes are normalised by their view span so
        the distance is dimensionless and directly comparable to the
        2.5% tolerance used by _hit_test."""
        sx, sy = max(span_x, 1e-9), max(span_y, 1e-9)
        pxn, pyn = px / sx, py / sy
        axn, ayn = ax / sx, ay / sy
        bxn, byn = bx / sx, by / sy
        dx, dy = bxn - axn, byn - ayn
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-18:
            ex, ey = pxn - axn, pyn - ayn
            return (ex * ex + ey * ey) ** 0.5
        t = ((pxn - axn) * dx + (pyn - ayn) * dy) / seg2
        t = max(0.0, min(1.0, t))
        qx, qy = axn + t * dx, ayn + t * dy
        ex, ey = pxn - qx, pyn - qy
        return (ex * ex + ey * ey) ** 0.5

    @staticmethod
    def _gid_from_hit(hit):
        """Best-effort extraction of the gate id encoded in a hit tuple.
        Returns None when the hit isn't gate-bound (or for malformed
        tuples). Threshold/interval lines pack the id as ``'gid'`` or
        ``'gid:lo' / 'gid:hi'``; all other shapes use the bare id."""
        if not hit or len(hit) < 2:
            return None
        second = hit[1]
        if not isinstance(second, str):
            return None
        return second.split(':', 1)[0] if ':' in second else second

    def _delete_polygon_vertex(self, gid, vi):
        """Delete vertex ``vi`` from polygon ``gid``. Refuses to drop
        the polygon below 3 vertices (degenerate)."""
        g = self._gates.get(gid)
        if g is None or g.get('kind') != 'polygon':
            return
        verts = list(g.get('vertices') or [])
        if len(verts) <= 3:
            return  # would degenerate
        if not (0 <= vi < len(verts)):
            return
        del verts[vi]
        g['vertices'] = verts
        self._redraw_only_gates()
        self._refresh_gate_list()

    def _insert_polygon_vertex(self, gid, idx, x, y):
        """Insert a vertex (x, y) at position ``idx`` in polygon ``gid``."""
        g = self._gates.get(gid)
        if g is None or g.get('kind') != 'polygon':
            return
        if x is None or y is None:
            return
        verts = list(g.get('vertices') or [])
        idx = max(0, min(int(idx), len(verts)))
        verts.insert(idx, [float(x), float(y)])
        g['vertices'] = verts
        self._redraw_only_gates()
        self._refresh_gate_list()

    def _polygon_under_point(self, x, y):
        """Return gate_id of the polygon containing (x, y), or — if none
        contain it — the nearest polygon by vertex distance. None if no
        polygon gates exist in this view's channels."""
        if x is None or y is None:
            return None
        from matplotlib.path import Path as _MplPath
        x_ch = self._resolve_channel(self.x_combo.get())
        y_ch = self._resolve_channel(self.y_combo.get())
        candidates = []
        for gid, g in self._gates.items():
            if g.get('kind') != 'polygon':
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            verts = g.get('vertices') or []
            if len(verts) < 3:
                continue
            try:
                arr = np.asarray(verts, dtype=float)
                if _MplPath(arr).contains_point((x, y)):
                    return gid
                # distance to nearest vertex (axis-fraction units)
                dx = arr[:, 0] - x
                dy = arr[:, 1] - y
                d = float(np.sqrt((dx * dx + dy * dy).min()))
                candidates.append((d, gid))
            except Exception:
                continue
        if candidates:
            candidates.sort()
            return candidates[0][1]
        return None

    def _on_press(self, event):
        btn = getattr(event, 'button', 1)
        key = (getattr(event, 'key', None) or '').lower()
        is_shift = 'shift' in key
        is_alt   = 'alt'   in key

        # ── Right-click gestures (work in any tool) ───────────────────────
        # Smart "±vertex" — on a polygon vertex it deletes, on a polygon
        # edge it inserts. Anywhere else the right-click is a no-op so it
        # doesn't accidentally pan / open a menu / etc.
        if btn == 3:
            hit = self._hit_test(event)
            if hit:
                kind = hit[0]
                if kind == 'poly_vertex':
                    _, gid, vi = hit                       # type: ignore[misc]
                    self._checkpoint()
                    self._delete_polygon_vertex(gid, int(vi))
                    return
                if kind == 'poly_edge':
                    _, gid, edge_idx = hit                 # type: ignore[misc]
                    self._checkpoint()
                    self._insert_polygon_vertex(
                        gid, int(edge_idx) + 1,
                        event.xdata, event.ydata)
                    return
            return

        # ── Alt+left-click: drop a new vertex in the polygon under the
        #    cursor (or nearest polygon when no polygon contains it). ─────
        if btn == 1 and is_alt and event.inaxes is self.ax:
            gid = self._polygon_under_point(event.xdata, event.ydata)
            if gid is not None:
                self._checkpoint()
                g = self._gates[gid]
                verts = list(g.get('vertices') or [])
                verts.append([float(event.xdata), float(event.ydata)])
                g['vertices'] = verts
                self._redraw_only_gates()
                self._refresh_gate_list()
                return

        # 1) Hit-test for any draggable gate handle (line, corner, edge,
        #    vertex, quadrant centre/axis). If something hits, start a
        #    drag and pick a cursor that hints what the gesture does.
        hit = self._hit_test(event)

        # ── Shift+left-drag: force translate-the-whole-gate even if the
        #    click landed on a vertex / edge / corner. Only meaningful
        #    for gates that have a translate mode (rect, polygon). ──────
        if btn == 1 and is_shift and hit:
            gid = self._gid_from_hit(hit)
            if gid and gid in self._gates:
                g = self._gates[gid]
                k = g.get('kind')
                if k == 'rect':
                    hit = ('rect_translate', gid)
                elif k == 'polygon':
                    hit = ('poly_translate', gid)
                elif k == 'ellipsoid':
                    hit = ('ellipse_translate', gid)
                # threshold/interval/quadrant have no translate kind —
                # fall through to existing behaviour.

        if hit:
            kind = hit[0]
            # poly_edge is special: insert a new vertex at the click
            # position and become a poly_vertex drag immediately. That
            # matches FlowJo's "click an edge to add a point and shape it".
            if kind == 'poly_edge':
                _, gid, edge_idx = hit   # type: ignore[misc]
                g = self._gates.get(gid)
                if g is not None and event.xdata is not None and event.ydata is not None:
                    self._checkpoint()
                    verts = list(g.get('vertices') or [])
                    new_i = int(edge_idx) + 1
                    verts.insert(new_i, [float(event.xdata),
                                         float(event.ydata)])
                    g['vertices'] = verts
                    self._drag_state = ('poly_vertex', gid, new_i)
                    self._redraw_only_gates()
                    self.canvas.get_tk_widget().config(cursor='fleur')
                return
            # A handle was grabbed — snapshot the pre-drag geometry so the
            # whole drag is one undo step (motion mutates in place).
            self._checkpoint()
            self._drag_state = hit
            # Translate / resize / rotate drags cache the press anchor +
            # a deep copy of the gate at start, so motion applies the
            # delta to the *original* geometry — no rounding drift over
            # the drag.
            if kind in ('rect_translate', 'poly_translate',
                        'ellipse_translate', 'ellipse_rim', 'ellipse_rotate'):
                import copy as _copy
                _, gid_t = hit                      # type: ignore[misc]
                g_t = self._gates.get(gid_t)
                if g_t is None or event.xdata is None or event.ydata is None:
                    self._drag_state = None
                    return
                ctx = {
                    'anchor_x': float(event.xdata),
                    'anchor_y': float(event.ydata),
                    'orig':     _copy.deepcopy(g_t),
                }
                # Rotation needs the press direction from the centre so
                # motion can measure the swept angle.
                if kind == 'ellipse_rotate':
                    om = g_t.get('mean', [0.0, 0.0])
                    ctx['rot_anchor_angle'] = float(np.arctan2(
                        float(event.ydata) - float(om[1]),
                        float(event.xdata) - float(om[0])))
                self._drag_translate_ctx = ctx
            # Cursor hint per drag kind.
            if kind in ('v', 'quad_x'):
                cur = 'sb_h_double_arrow'
            elif kind in ('h', 'quad_y'):
                cur = 'sb_v_double_arrow'
            elif kind == 'ellipse_rotate':
                cur = 'exchange'
            elif kind == 'rect_edge':
                cur = ('sb_v_double_arrow' if hit[2] in ('top', 'bottom')  # type: ignore[misc]
                       else 'sb_h_double_arrow')
            elif kind in ('rect_translate', 'poly_translate'):
                cur = 'fleur'
            else:                         # rect_corner / poly_vertex / quad_origin
                cur = 'fleur'
            self.canvas.get_tk_widget().config(cursor=cur)
            return

        if event.inaxes is not self.ax:
            return

        # 2) Region-shape tools (rect/polygon/lasso) are handled by their
        #    matplotlib Selector — we don't intercept those clicks here.
        tool = (self.gate_tool_var.get() if hasattr(self, 'gate_tool_var')
                else 'quadrant')
        if tool != 'quadrant':
            return

        # 3) Quadrant tool: double-click or shift-click → add gate(s).
        is_add = (event.dblclick
                  or (getattr(event, 'key', None) or '') == 'shift')
        if not is_add:
            return
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        mode = self.mode_var.get()
        if mode == 'histogram':
            # 1D: single threshold gate on the click x position.
            if x and event.xdata is not None:
                self._add_gate({'kind': 'threshold', 'channel': x,
                                'value': float(event.xdata)})
        else:
            # 2D: emit 4 quadrant rect gates centred on the click. Each
            # quadrant is a separate rect so the user can toggle / colour
            # them independently. Bounds clip to the current axis viewport
            # at the moment of the click.
            if (not x or not y
                    or event.xdata is None or event.ydata is None):
                return
            xc, yc = float(event.xdata), float(event.ydata)
            try:
                xl, xh = self.ax.get_xlim()
                yl, yh = self.ax.get_ylim()
            except Exception:
                xl, xh = xc - 1.0, xc + 1.0
                yl, yh = yc - 1.0, yc + 1.0
            parent = self._selected_gate_id()
            # Shared identifier so the 4 quadrant rects can be moved
            # together (drag the origin or one of the dividing axes) at
            # hit-test / motion time.
            self._quad_set_seq += 1
            qs_id = f'qs{self._quad_set_seq}'
            for label, x0, x1, y0, y1 in [
                    ('Q++ (x>, y>)', xc, xh, yc, yh),
                    ('Q+- (x>, y<)', xc, xh, yl, yc),
                    ('Q-+ (x<, y>)', xl, xc, yc, yh),
                    ('Q-- (x<, y<)', xl, xc, yl, yc)]:
                self._add_gate({'kind': 'rect',
                                'x_channel': x, 'y_channel': y,
                                'x0': float(x0), 'x1': float(x1),
                                'y0': float(y0), 'y1': float(y1),
                                'label': label,
                                'quad_set':       qs_id,
                                'quad_origin_x':  float(xc),
                                'quad_origin_y':  float(yc)},
                               parent_id=parent)
        self._schedule_replot(0)

    def _on_release(self, event):
        was_dragging = self._drag_state is not None
        self._drag_state = None
        self._drag_translate_ctx = None    # always clear; cheap if unset
        self.canvas.get_tk_widget().config(cursor='')
        # Refresh the gate tree once at end-of-drag so the description
        # text (rect bounds, polygon vert count, etc.) reflects the new
        # geometry. We skipped this during motion for performance.
        if was_dragging or event is None or event.inaxes is None:
            self._refresh_gate_list()

    def _on_motion(self, event):
        if self._drag_state is None or event.inaxes is not self.ax:
            return
        kind = self._drag_state[0]
        cx = event.xdata
        cy = event.ydata
        if cx is None and cy is None:
            return

        # ── 1D axis lines (existing) ─────────────────────────────────────
        if kind in ('v', 'h'):
            _, key = self._drag_state   # type: ignore[misc]
            if ':' in key:
                gid, side = key.split(':', 1)
            else:
                gid, side = key, None
            g = self._gates.get(gid)
            if g is None:
                return
            if kind == 'v' and cx is not None:
                new_val = float(cx)
                if g['kind'] == 'threshold':
                    g['value'] = new_val
                elif g['kind'] == 'interval':
                    g['lo' if side == 'lo' else 'hi'] = new_val
                    if g['lo'] > g['hi']:
                        g['lo'], g['hi'] = g['hi'], g['lo']
                line = self._vlines.get(key)
                if line is not None:
                    line.set_xdata([new_val, new_val])
            elif kind == 'h' and cy is not None:
                new_val = float(cy)
                if g['kind'] == 'threshold':
                    g['value'] = new_val
                elif g['kind'] == 'interval':
                    g['lo' if side == 'lo' else 'hi'] = new_val
                    if g['lo'] > g['hi']:
                        g['lo'], g['hi'] = g['hi'], g['lo']
                line = self._hlines.get(key)
                if line is not None:
                    line.set_ydata([new_val, new_val])
            self._refresh_gate_list()
            self.canvas.draw_idle()
            if self.apply_gates_var.get():
                self._schedule_replot(120)
            return

        # ── Rect corner / edge ───────────────────────────────────────────
        if kind in ('rect_corner', 'rect_edge'):
            _, gid, which = self._drag_state   # type: ignore[misc]
            g = self._gates.get(gid)
            if g is None or g.get('kind') != 'rect':
                return
            x0, x1 = sorted([float(g['x0']), float(g['x1'])])
            y0, y1 = sorted([float(g['y0']), float(g['y1'])])
            if kind == 'rect_corner' and cx is not None and cy is not None:
                if which == 'bl':   x0, y0 = float(cx), float(cy)
                elif which == 'br': x1, y0 = float(cx), float(cy)
                elif which == 'tl': x0, y1 = float(cx), float(cy)
                elif which == 'tr': x1, y1 = float(cx), float(cy)
            elif kind == 'rect_edge':
                if which == 'top'    and cy is not None: y1 = float(cy)
                elif which == 'bottom' and cy is not None: y0 = float(cy)
                elif which == 'left'   and cx is not None: x0 = float(cx)
                elif which == 'right'  and cx is not None: x1 = float(cx)
            if x0 > x1: x0, x1 = x1, x0
            if y0 > y1: y0, y1 = y1, y0
            g['x0'], g['x1'], g['y0'], g['y1'] = x0, x1, y0, y1

        # ── Polygon vertex ───────────────────────────────────────────────
        elif kind == 'poly_vertex' and cx is not None and cy is not None:
            _, gid, vi = self._drag_state   # type: ignore[misc]
            g = self._gates.get(gid)
            if g is None or g.get('kind') != 'polygon':
                return
            verts = list(g.get('vertices') or [])
            vi = int(vi)
            if 0 <= vi < len(verts):
                verts[vi] = [float(cx), float(cy)]
                g['vertices'] = verts

        # ── Translate the whole shape (rect / polygon / ellipsoid) ───────
        elif kind in ('rect_translate', 'poly_translate', 'ellipse_translate'):
            if cx is None or cy is None:
                return
            _, gid = self._drag_state   # type: ignore[misc]
            g = self._gates.get(gid)
            ctx = self._drag_translate_ctx or {}
            orig = ctx.get('orig')
            if g is None or orig is None:
                return
            dx = float(cx) - float(ctx['anchor_x'])
            dy = float(cy) - float(ctx['anchor_y'])
            if kind == 'rect_translate' and g.get('kind') == 'rect':
                g['x0'] = float(orig['x0']) + dx
                g['x1'] = float(orig['x1']) + dx
                g['y0'] = float(orig['y0']) + dy
                g['y1'] = float(orig['y1']) + dy
            elif kind == 'poly_translate' and g.get('kind') == 'polygon':
                g['vertices'] = [[float(v[0]) + dx, float(v[1]) + dy]
                                 for v in orig.get('vertices', [])]
            elif kind == 'ellipse_translate' and g.get('kind') == 'ellipsoid':
                om = orig.get('mean', [0.0, 0.0])
                g['mean'] = [float(om[0]) + dx, float(om[1]) + dy]

        # ── Ellipsoid resize (drag the rim) ──────────────────────────────
        # Uniformly scale the covariance so the rim passes through the
        # cursor: if the cursor sits at Mahalanobis radius md (under the
        # ORIGINAL Σ), scaling Σ by (md/r0)² moves the rim onto it.
        elif kind == 'ellipse_rim' and cx is not None and cy is not None:
            _, gid = self._drag_state   # type: ignore[misc]
            g = self._gates.get(gid)
            ctx = self._drag_translate_ctx or {}
            orig = ctx.get('orig')
            if g is None or orig is None or g.get('kind') != 'ellipsoid':
                return
            mean = np.asarray(orig['mean'], dtype=float)
            cov0 = np.asarray(orig['cov'], dtype=float)
            dist_sq = float(orig.get('distance_sq', 4.0))
            try:
                inv0 = np.linalg.inv(cov0)
            except np.linalg.LinAlgError:
                return
            d = np.array([float(cx) - mean[0], float(cy) - mean[1]])
            md_sq = float(d @ inv0 @ d)
            r0_sq = max(dist_sq, 1e-12)
            scale = md_sq / r0_sq                 # = (md/r0)²
            scale = max(scale, 1e-6)              # never collapse to zero
            g['cov'] = (cov0 * scale).tolist()

        # ── Ellipsoid rotate (drag the top handle) ───────────────────────
        # Rotate Σ by the angle swept between the handle's original
        # direction and the cursor direction, both measured from centre.
        elif kind == 'ellipse_rotate' and cx is not None and cy is not None:
            _, gid = self._drag_state   # type: ignore[misc]
            g = self._gates.get(gid)
            ctx = self._drag_translate_ctx or {}
            orig = ctx.get('orig')
            if g is None or orig is None or g.get('kind') != 'ellipsoid':
                return
            mean = np.asarray(orig['mean'], dtype=float)
            cov0 = np.asarray(orig['cov'], dtype=float)
            a0 = float(ctx.get('rot_anchor_angle', 0.0))
            a1 = float(np.arctan2(float(cy) - mean[1], float(cx) - mean[0]))
            dtheta = a1 - a0
            c, s = np.cos(dtheta), np.sin(dtheta)
            rot = np.array([[c, -s], [s, c]])
            g['cov'] = (rot @ cov0 @ rot.T).tolist()

        # ── Quadrant centre / dividers ───────────────────────────────────
        elif kind == 'quad_origin' and cx is not None and cy is not None:
            self._update_quad_set(self._drag_state[1],
                                  new_x=float(cx), new_y=float(cy))
        elif kind == 'quad_x' and cx is not None:
            self._update_quad_set(self._drag_state[1],
                                  new_x=float(cx), new_y=None)
        elif kind == 'quad_y' and cy is not None:
            self._update_quad_set(self._drag_state[1],
                                  new_x=None, new_y=float(cy))
        else:
            return

        # All non-line drags need a full overlay rebuild (the Patch
        # geometry has to be redrawn from the mutated gate dict). Tree
        # text changes for rect/polygon are unlikely to be interesting
        # mid-drag, so skip the heavier _refresh_gate_list — release
        # handler does that.
        self._redraw_only_gates()
        if self.apply_gates_var.get():
            self._schedule_replot(120)

    def _update_quad_set(self, qs_id, new_x=None, new_y=None):
        """Move the shared origin of a 4-rect quadrant set. Each member's
        origin-corner is identified by its `label` (Q++ / Q+- / Q-+ / Q--);
        the corresponding x and/or y bound is rewritten to the new value
        while the outer extent is left alone. Pass new_x and/or new_y;
        the unspecified axis keeps its current origin coord."""
        members = [g for g in self._gates.values()
                   if g.get('quad_set') == qs_id]
        if not members:
            return
        cur_x = float(members[0].get('quad_origin_x', 0.0))
        cur_y = float(members[0].get('quad_origin_y', 0.0))
        nx = cur_x if new_x is None else float(new_x)
        ny = cur_y if new_y is None else float(new_y)
        for g in members:
            label = g.get('label', '') or ''
            # Map quadrant label → which (x, y) corner of the rect is the
            # SHARED origin (the others stay put as the outer extents).
            if   'Q++' in label: g['x0'], g['y0'] = nx, ny
            elif 'Q+-' in label: g['x0'], g['y1'] = nx, ny
            elif 'Q-+' in label: g['x1'], g['y0'] = nx, ny
            elif 'Q--' in label: g['x1'], g['y1'] = nx, ny
            else: continue
            g['quad_origin_x'] = nx
            g['quad_origin_y'] = ny

    # ── Tree press / motion / release (click vs. drag-reparent) ──────────

    def _on_tv_press(self, event):
        """Record where the press started so we can disambiguate a click
        from a drag in the matching release/motion handlers. Doesn't
        suppress Treeview's default selection."""
        self._press_iid = self.gate_tv.identify_row(event.y)
        self._press_col = self.gate_tv.identify_column(event.x)
        self._press_x   = event.x
        self._press_y   = event.y
        self._drag_active = False
        # Snapshot the multi-selection BEFORE Tk's class binding collapses it
        # to the clicked row, so bulk display-toggle and multi-drag see it.
        self._press_selection = tuple(self.gate_tv.selection())

    def _on_tv_motion(self, event):
        """Once the cursor moves past the threshold on a gate or sample row, the
        gesture becomes a drag (cursor changes to fleur): a gate reparents, a
        sample regroups (move to another day, or between the Comps/Samples
        subgroups). The checkbox column and empty space aren't draggable."""
        if self._press_iid is None or self._drag_active:
            return
        if (abs(event.x - self._press_x) <= self._drag_threshold
                and abs(event.y - self._press_y) <= self._drag_threshold):
            return
        parsed = self._parse_iid(self._press_iid)
        if parsed is None:
            return
        if parsed[0] == 'trial':
            # Trial rows become draggable only to ferry their members to an open
            # Pipeline Workspace (no in-editor reparenting of a whole trial).
            if not self._workspace_open():
                return
        elif parsed[0] not in ('sample', 'gate'):
            return
        # Don't initiate drag from the toggle column (avoids accidental
        # drags during checkbox clicks).
        if self._press_col == '#1':
            return
        self._drag_active = True
        try:
            self.gate_tv.config(cursor='fleur')
        except Exception:
            pass

    def _on_tv_release(self, event):
        """Either:
          • the user just clicked (no drag): toggle ☑/☐ if they pressed
            the toggle column; otherwise let default selection stand.
          • the user dragged: reparent the source gate onto the row under
            the cursor (sample row → make it a root; gate row → make it
            that gate's child). Cycles and cross-sample drops are refused.
        """
        drag = self._drag_active
        press_iid = self._press_iid
        press_col = self._press_col
        self._press_iid = None
        self._press_col = None
        self._drag_active = False
        try:
            self.gate_tv.config(cursor='')
        except Exception:
            pass

        if drag:
            # Cross-window: a drag that ends over the Pipeline Workspace hands
            # that node to the workspace (with its cumulative gate chain + comp
            # matrix) instead of reparenting inside the editor.
            if self._maybe_drop_to_workspace(press_iid, event):
                return
            # …or over an open Statistics window → add its sample(s) there.
            if self._maybe_drop_to_stats(press_iid, event):
                return
            target_iid = self.gate_tv.identify_row(event.y)
            self._handle_drag_drop(press_iid, target_iid)
            return

        # Plain click. Toggle on press_col '#1' (the ☑ column).
        if press_iid and press_col == '#1':
            self._handle_checkbox_click(press_iid)

    def _display_toggle_target_state(self, parsed):
        """The new on/off state implied by clicking ``parsed``'s checkbox.
        Trial: off if any member is currently on, else on. Sample/gate: flip."""
        if parsed[0] == 'trial':
            members = self._trial_members(parsed[1])
            return not any(self._sample_plot_enabled.get(n, True) for n in members)
        if parsed[0] == 'subgroup':
            members = self._subgroup_members(parsed[1], parsed[2])
            return not any(self._sample_plot_enabled.get(n, True) for n in members)
        if parsed[0] == 'sample':
            return not self._sample_plot_enabled.get(parsed[1], True)
        if parsed[0] == 'gate':
            g = self._sample_gates.get(parsed[1], {}).get(parsed[2])
            return not (g.get('enabled', True) if g else True)
        if parsed[0] == 'method':
            g = self._sample_gates.get(parsed[1], {}).get(parsed[2])
            if g and g.get('kind') == 'autoclean':
                for m in g.get('methods', []):
                    if m.get('key') == parsed[3]:
                        return not m.get('enabled', True)
            return True
        return True

    def _handle_checkbox_click(self, row_id):
        clicked = self._parse_iid(row_id)
        if clicked is None:
            return
        # New state comes from the clicked row; with a live multi-selection
        # (captured pre-click) the same state is applied to every selected row.
        new_state = self._display_toggle_target_state(clicked)
        sel = self._press_selection or ()
        rows = sel if (row_id in sel and len(sel) > 1) else (row_id,)

        changed_plot = False
        changed_gate = False
        for r in rows:
            p = self._parse_iid(r)
            if p is None:
                continue
            if p[0] == 'trial':
                for n in self._trial_members(p[1]):
                    self._sample_plot_enabled[n] = new_state
                    changed_plot = True
            elif p[0] == 'subgroup':
                for n in self._subgroup_members(p[1], p[2]):
                    self._sample_plot_enabled[n] = new_state
                    changed_plot = True
            elif p[0] == 'sample':
                self._sample_plot_enabled[p[1]] = new_state
                changed_plot = True
            elif p[0] == 'gate':
                g = self._sample_gates.get(p[1], {}).get(p[2])
                if g is not None:
                    g['enabled'] = new_state
                    changed_gate = True
            elif p[0] == 'method' and len(p) == 4:
                g = self._sample_gates.get(p[1], {}).get(p[2])
                if g is not None and g.get('kind') == 'autoclean':
                    for m in g.get('methods', []):
                        if m.get('key') == p[3]:
                            m['enabled'] = new_state
                            changed_gate = True
                            break

        self._refresh_gate_list()
        if changed_plot:
            self._schedule_replot(0)
        elif changed_gate:
            mode = getattr(self, 'gate_display_var', None)
            if mode is not None and mode.get() in ('filter', 'highlight'):
                self._schedule_replot(0)
            else:
                self._redraw_only_gates()

    def _workspace_open(self):
        """True if there's a live drop target: the docked pane is shown, or a
        workspace tab has been popped out into its own window."""
        panel = getattr(self, '_workspace_panel', None)
        if panel is None:
            return False
        try:
            return bool(getattr(self, '_workspace_shown', False)) or panel.popped_count() > 0
        except Exception:
            return False

    def _maybe_drop_to_workspace(self, src_iid, event):
        """If a tree drag ended over a Pipeline Workspace tree (the docked pane's
        active tab, or a popped-out window), hand the dragged sample/leaf to that
        workspace instead of reparenting in the editor. The workspace snapshots
        the node's gate chain + linked comp matrix itself. Returns True iff the
        drop was consumed."""
        panel = getattr(self, '_workspace_panel', None)
        if panel is None:
            return False
        # Ferry the whole multi-selection if the dragged row is part of it,
        # else just the dragged row. Trials expand to their member samples.
        sel = self._press_selection or ()
        srcs = list(sel) if (src_iid in sel and len(sel) > 1) else [src_iid]
        nodes = []
        for s in srcs:
            p = self._parse_iid(s)
            if p is None:
                continue
            if p[0] in ('sample', 'gate'):
                nodes.append(p)
            elif p[0] == 'trial':
                nodes.extend(('sample', n) for n in self._trial_members(p[1]))
        if not nodes:
            return False
        try:
            under = self.winfo_containing(event.x_root, event.y_root)
            if not panel.is_drop_target(under):
                return False
            # The workspace routes by the column under the pointer (drop on the
            # Comp/FMO column to assign beads/FMOs; elsewhere adds populations).
            return bool(panel.drop_at(self, nodes, event.x_root, event.y_root))
        except Exception:
            return False

    def _stats_window_under(self, x_root, y_root):
        """The open StatisticsWindow whose Toplevel contains the screen point,
        or None. Used as a cross-window drop target (editor tree + workspace)."""
        try:
            w = self.winfo_containing(x_root, y_root)
        except Exception:
            return None
        while w is not None:
            if isinstance(w, StatisticsWindow):
                return w
            w = getattr(w, 'master', None)
        return None

    def _dragged_gate_targets(self, src_iid):
        """(sample, gid) population targets implied by a tree drag: the dragged
        row plus the rest of the live multi-selection if the dragged row is
        part of it. ONLY gate rows qualify — whole-sample and trial rows are
        ignored (statistics is population-based). De-duplicated, order-keeping;
        only gates of loaded samples are returned."""
        sel = self._press_selection or ()
        srcs = list(sel) if (src_iid in sel and len(sel) > 1) else [src_iid]
        targets = []
        for s in srcs:
            p = self._parse_iid(s) if s else None
            if p and p[0] == 'gate':
                nm, gid = p[1], p[2]
                if nm in self._samples and (nm, gid) not in targets:
                    targets.append((nm, gid))
        return targets

    def _maybe_drop_to_stats(self, src_iid, event):
        """If a tree drag ends over an open Statistics window, add the dragged
        population(s) to it. Statistics accepts gate rows only — dropping a
        whole sample/trial is consumed with an explanatory status, never
        falls through to a reparent. Returns True iff over a stats window."""
        try:
            win = self._stats_window_under(event.x_root, event.y_root)
        except Exception:
            win = None
        if win is None:
            return False
        targets = self._dragged_gate_targets(src_iid)
        if not targets:
            self.status_var.set("Statistics accepts gate/population rows only "
                                "(not whole samples) — drag a gate.")
            return True
        try:
            win.add_targets(targets, 'editor')
            self.status_var.set(f"Added {len(targets)} population(s) to statistics.")
        except Exception:
            pass
        return True

    def _handle_drag_drop(self, src_iid, target_iid):
        """Reparent (within sample) or MOVE (across samples) the dragged
        gate, including its entire subtree, onto the drop target.

        Within-sample: just rewrites the dragged gate's parent_id.
        Cross-sample: copies the whole subtree to the destination with
        fresh gate ids + remapped parent_id pointers, then removes the
        originals from the source.

        Cycles (drop onto own descendant within the same sample) are
        refused. Same-parent drops are no-ops.
        """
        if not src_iid or not target_iid or src_iid == target_iid:
            return
        src = self._parse_iid(src_iid)
        tgt = self._parse_iid(target_iid)
        if not src or not tgt:
            return
        # A dragged SAMPLE regroups (move day / switch Comps<->Samples) rather
        # than reparenting gates.
        if src[0] == 'sample':
            self._regroup_dragged_samples(src_iid, target_iid)
            return
        if src[0] != 'gate':
            return
        # Only a sample or gate row is a valid reparent target — dropping onto a
        # trial / Comps-Samples subgroup / method header is ambiguous (no owning
        # sample), so ignore it.
        if tgt[0] not in ('sample', 'gate'):
            return
        src_sample, src_gid = src[1], src[2]
        tgt_sample = tgt[1]
        self._checkpoint()

        # Compute new parent in the *destination* sample.
        if tgt[0] == 'sample':
            new_parent = None
        else:  # gate tuple is ('gate', sample, gid)
            new_parent = tgt[2] if len(tgt) > 2 else None

        # ── Same-sample reparent ─────────────────────────────────────────
        if tgt_sample == src_sample:
            if new_parent == src_gid:
                return
            if (new_parent is not None
                    and self._is_descendant_of(src_sample, src_gid, new_parent)):
                self.status_var.set(
                    "Drop refused: would create a cycle in the gate tree.")
                return
            sgates = self._sample_gates.get(src_sample, {})
            g = sgates.get(src_gid)
            if g is None:
                return
            if g.get('parent_id') == new_parent:
                return  # no-op
            g['parent_id'] = new_parent
            self._refresh_gate_list()
            try:
                self.gate_tv.selection_set(self._gate_iid(src_sample, src_gid))
                self.gate_tv.see(self._gate_iid(src_sample, src_gid))
            except Exception:
                pass
            if self.gate_display_var.get() in ('filter', 'highlight'):
                self._schedule_replot(0)
            from .pipeline import describe_gate
            self.status_var.set(
                f"Reparented {describe_gate(g)} → "
                f"{'root of ' + src_sample if new_parent is None else 'under ' + new_parent}.")
            return

        # ── Cross-sample move ────────────────────────────────────────────
        moved_root_new_gid = self._move_gate_to_sample(
            src_sample, src_gid, tgt_sample, new_parent)
        if moved_root_new_gid is None:
            return
        self._refresh_gate_list()
        try:
            self.gate_tv.selection_set(
                self._gate_iid(tgt_sample, moved_root_new_gid))
            self.gate_tv.see(
                self._gate_iid(tgt_sample, moved_root_new_gid))
        except Exception:
            pass
        if self.gate_display_var.get() in ('filter', 'highlight'):
            self._schedule_replot(0)
        else:
            self._redraw_only_gates()
        self.status_var.set(
            f"Moved gate from '{src_sample}' to '{tgt_sample}' "
            f"(now as {'root' if new_parent is None else 'child of ' + new_parent}).")

    def _regroup_target(self, tgt):
        """Resolve a drop target row into (new_trial, new_comp) for sample
        regrouping. Either may be None ('leave as-is'):
          • trial row     → (that trial, None)
          • subgroup row   → (its trial, True/False for Comps/Samples)
          • sample/gate/method row → that row's owning sample's (trial, comp)."""
        if tgt[0] == 'trial':
            return tgt[1], None
        if tgt[0] == 'subgroup':
            return tgt[2], (tgt[1] == 'comp')
        if tgt[0] in ('sample', 'gate', 'method'):
            ref = tgt[1]
            if ref in self._samples:
                return self._trial_for(ref), self._is_comp(ref)
        return None, None

    def _regroup_dragged_samples(self, src_iid, target_iid):
        """Move the dragged sample(s) to the drop target's day and/or
        Comps↔Samples subgroup. Honors a live multi-selection. Reassigning the
        Comps/Samples side sets a manual override (`_sample_is_comp`)."""
        tgt = self._parse_iid(target_iid)
        if tgt is None:
            return
        new_trial, new_comp = self._regroup_target(tgt)
        if new_trial is None and new_comp is None:
            return
        sel = self._press_selection or ()
        srcs = list(sel) if (src_iid in sel and len(sel) > 1) else [src_iid]
        names = []
        for s in srcs:
            p = self._parse_iid(s) if s else None
            if p and p[0] == 'sample' and p[1] in self._samples \
                    and p[1] not in names:
                names.append(p[1])
        if not names:
            return
        moved = 0
        for n in names:
            changed = False
            if new_trial is not None and self._trial_for(n) != new_trial:
                self._sample_trial[n] = new_trial
                if new_trial not in self._trial_order:
                    self._trial_order.append(new_trial)
                changed = True
            if new_comp is not None and self._is_comp(n) != new_comp:
                self._sample_is_comp[n] = new_comp
                changed = True
            if changed:
                moved += 1
        if not moved:
            return
        # Drop now-empty trials from the order list.
        self._trial_order = [t for t in self._trial_order
                             if any(self._trial_for(n) == t
                                    for n in self._samples)]
        self._refresh_gate_list()
        self._schedule_replot(0)
        dest = new_trial if new_trial is not None else self._trial_for(names[0])
        side = ('' if new_comp is None
                else ' / Comps' if new_comp else ' / Samples')
        self.status_var.set(f"Moved {moved} sample(s) → {dest}{side}.")

    def _move_gate_to_sample(self, src_sample, src_gid,
                             dst_sample, new_parent_in_dst):
        """Move a gate AND every descendant from src_sample to dst_sample.

        Fresh gate ids are assigned in the destination; parent_id
        pointers within the moved subtree are remapped to those new ids.
        The dragged gate's root becomes a child of `new_parent_in_dst`
        (None = root of the destination sample).

        Returns the new gate id of the dragged gate's root in dst, or
        None on failure.
        """
        import copy as _copy
        src_gates = self._sample_gates.get(src_sample, {})
        src_order = self._sample_gate_order.get(src_sample, [])
        if src_gid not in src_gates:
            return None

        # Collect the subtree rooted at src_gid (BFS, preserves any
        # ordering invariants children had relative to each other).
        subtree_ids = []
        queue       = [src_gid]
        seen        = set()
        while queue:
            cur = queue.pop(0)
            if cur in seen or cur not in src_gates:
                continue
            seen.add(cur)
            subtree_ids.append(cur)
            for other_gid, og in src_gates.items():
                if og.get('parent_id') == cur:
                    queue.append(other_gid)

        # Ensure destination containers exist.
        self._sample_gates.setdefault(dst_sample, {})
        self._sample_gate_order.setdefault(dst_sample, [])
        self._sample_gate_seq.setdefault(dst_sample, 0)
        dst_gates = self._sample_gates[dst_sample]
        dst_order = self._sample_gate_order[dst_sample]

        # Assign fresh ids in dst for every subtree member.
        old_to_new = {}
        for old_id in subtree_ids:
            self._sample_gate_seq[dst_sample] += 1
            old_to_new[old_id] = f'g{self._sample_gate_seq[dst_sample]}'
        if dst_sample == self._active_sample:
            self._gate_id_seq = self._sample_gate_seq[dst_sample]

        # Transplant each gate (deep-copied to be safe), remapping parents.
        for old_id in subtree_ids:
            g = _copy.deepcopy(src_gates[old_id])
            new_id = old_to_new[old_id]
            if old_id == src_gid:
                g['parent_id'] = new_parent_in_dst
            else:
                old_parent = g.get('parent_id')
                g['parent_id'] = old_to_new.get(old_parent, new_parent_in_dst)
            dst_gates[new_id] = g
            dst_order.append(new_id)

        # Remove the subtree from src.
        for old_id in subtree_ids:
            src_gates.pop(old_id, None)
            if old_id in src_order:
                src_order.remove(old_id)

        return old_to_new[src_gid]

    # ── Clipboard / context-menu / OS drag-drop ──────────────────────────

    def _collect_gate_subtree(self, sample_name, root_gid):
        """Deep-copy the subtree rooted at `root_gid` from `sample_name`.
        Each returned dict carries a temporary `_clip_id` (the original
        gate_id, so paste can rewire `parent_id` references). BFS order
        so paste can iterate parents-before-children safely."""
        import copy as _copy
        sgates = self._sample_gates.get(sample_name, {})
        if root_gid not in sgates:
            return []
        collected = []
        queue = [root_gid]
        seen  = set()
        while queue:
            cur = queue.pop(0)
            if cur in seen or cur not in sgates:
                continue
            seen.add(cur)
            g = _copy.deepcopy(sgates[cur])
            g['_clip_id'] = cur
            collected.append(g)
            for other_gid, og in sgates.items():
                if og.get('parent_id') == cur:
                    queue.append(other_gid)
        return collected

    def _paste_fcs_from_clipboard(self):
        """Read the OS clipboard and return any .fcs paths found.
        Tolerates Explorer's 'Copy as path' quoted form and multi-line
        / whitespace-separated entries. Defensive: any clipboard format
        we can't interpret as text is silently ignored."""
        try:
            text = self.clipboard_get()
        except Exception:
            return []
        if not isinstance(text, str):
            return []
        candidates = []
        for tok in re.split(r'[\r\n]+', text):
            for piece in re.split(r'(?<=\.fcs)\s+', tok, flags=re.I):
                p = piece.strip().strip('"').strip("'").strip()
                if p:
                    candidates.append(p)
        out = []
        for p in candidates:
            try:
                if p.lower().endswith('.fcs') and os.path.isfile(p):
                    out.append(p)
            except Exception:
                continue
        return out

    def _paste_gate_tree(self):
        """Paste the clipboard subtree into the active sample. The
        subtree's root attaches under the currently-selected gate (or
        as a root in the active sample if a sample row / nothing is
        selected). Multiple pastes don't consume the clipboard."""
        import copy as _copy
        if self._active_sample is None or not self._clip_payload:
            return
        subtree = _copy.deepcopy(self._clip_payload)
        if not subtree:
            return
        self._checkpoint()
        root_clip_id = subtree[0].get('_clip_id')

        # Resolve paste parent from current selection.
        paste_parent = None
        sel = self.gate_tv.selection()
        if sel:
            parsed = self._parse_iid(sel[0])
            if parsed:
                if parsed[0] == 'sample' and parsed[1] != self._active_sample:
                    self._set_active_sample(parsed[1])
                if parsed[0] == 'gate' and parsed[1] == self._active_sample:
                    paste_parent = parsed[2]

        # Assign fresh ids in the active sample.
        old_to_new = {}
        for g in subtree:
            self._gate_id_seq += 1
            if self._active_sample is not None:
                self._sample_gate_seq[self._active_sample] = self._gate_id_seq
            old_to_new[g['_clip_id']] = f'g{self._gate_id_seq}'

        for g in subtree:
            clip_id = g.pop('_clip_id', None)
            new_id  = old_to_new[clip_id]
            if clip_id == root_clip_id:
                g['parent_id'] = paste_parent
            else:
                g['parent_id'] = old_to_new.get(
                    g.get('parent_id'), paste_parent)
            self._gates[new_id] = g
            self._gate_id_order.append(new_id)

        self.status_var.set(
            f"Pasted {len(subtree)} gate(s) into '{self._active_sample}'.")
        self._refresh_gate_list()
        if self.gate_display_var.get() in ('filter', 'highlight'):
            self._schedule_replot(0)

    def _on_copy(self, event=None):
        sel = self.gate_tv.selection()
        if not sel:
            return 'break'
        parsed = self._parse_iid(sel[0])
        if parsed is None:
            return 'break'
        if parsed[0] == 'gate':
            subtree = self._collect_gate_subtree(parsed[1], parsed[2])
            if subtree:
                self._clip_kind    = 'gate_tree'
                self._clip_payload = subtree
                self.status_var.set(
                    f"Copied {len(subtree)} gate(s) "
                    f"(paste under a gate to nest, or a sample row for root).")
        elif parsed[0] == 'sample':
            name = parsed[1]
            sample = self._samples.get(name)
            path = getattr(sample, 'path', None) if sample else None
            if path:
                try:
                    self.clipboard_clear()
                    self.clipboard_append(path)
                except Exception:
                    pass
                self._clip_kind    = 'sample_paths'
                self._clip_payload = [path]
                self.status_var.set(f"Copied path of '{name}' to clipboard.")
        return 'break'

    def _on_cut(self, event=None):
        sel = self.gate_tv.selection()
        if not sel:
            return 'break'
        parsed = self._parse_iid(sel[0])
        if parsed is None:
            return 'break'
        if parsed[0] != 'gate':
            # Cutting a sample is destructive; we just Copy and don't
            # auto-remove the sample. User uses Remove for that.
            return self._on_copy(event)
        # Copy first, then delete the subtree from its source sample.
        self._on_copy(event)
        sample_name, gid = parsed[1], parsed[2]
        self._remove_gate_cascade_in(sample_name, gid)
        self._refresh_gate_list()
        if self.gate_display_var.get() in ('filter', 'highlight'):
            self._schedule_replot(0)
        return 'break'

    def _on_paste(self, event=None):
        """Order of precedence:
          1. Internal gate-tree clipboard → paste into active sample.
          2. OS clipboard FCS paths       → queue them as new samples.
        Reports a brief no-op message if neither applies."""
        if self._clip_kind == 'gate_tree' and self._clip_payload:
            self._paste_gate_tree()
            return 'break'
        fcs = self._paste_fcs_from_clipboard()
        if fcs:
            self._queue_fcs_loads(fcs)
            return 'break'
        self.status_var.set(
            "Nothing to paste (no copied gates, no .fcs paths on clipboard).")
        return 'break'

    def _on_delete_key(self, event=None):
        """Delete key: clear the selected gate (cascade), or — on a TRIAL row —
        remove that trial's samples + gates (confirmed). Plain sample rows are
        ignored to avoid accidental keyboard deletion; use the Remove button."""
        sel = self.gate_tv.selection()
        if not sel:
            return 'break'
        parsed = self._parse_iid(sel[0])
        if parsed and parsed[0] == 'gate':
            self._clear_selected_gate()
        elif parsed and parsed[0] == 'trial':
            self._remove_selected()
        return 'break'

    def _on_right_click(self, event):
        """Pop a context menu appropriate to whatever was right-clicked
        (gate row / sample row / empty space). Defensive — any failure
        building the menu just prints and bails (no crash)."""
        try:
            row_id = self.gate_tv.identify_row(event.y)
        except Exception:
            return 'break'
        if row_id:
            try:
                self.gate_tv.selection_set(row_id)
            except Exception:
                pass
        parsed = self._parse_iid(row_id) if row_id else None

        menu = tk.Menu(self.gate_tv, tearoff=0)
        try:
            paste_avail = bool(self._clip_payload) or bool(
                self._paste_fcs_from_clipboard())
        except Exception:
            paste_avail = False
        paste_state = 'normal' if paste_avail else 'disabled'

        if parsed and parsed[0] == 'gate':
            g = self._sample_gates.get(parsed[1], {}).get(parsed[2])
            if g is not None and g.get('kind') == 'autoclean':
                menu.add_command(
                    label="Edit auto-clean parameters…",
                    command=lambda n=parsed[1], gd=parsed[2]:
                        self._edit_autoclean_params(n, gd))
                menu.add_separator()
            menu.add_command(label="Copy gate (Ctrl+C)",
                             command=self._on_copy)
            menu.add_command(label="Cut gate (Ctrl+X)",
                             command=self._on_cut)
            menu.add_command(label="Paste (Ctrl+V)",
                             command=self._on_paste, state=paste_state)
            menu.add_separator()
            menu.add_command(label="Create boolean gate…",
                             command=self._open_boolean_dialog)
            menu.add_separator()
            menu.add_command(label="Backgate (show on plot)",
                             command=self._backgate_selected)
            if self._backgate:
                menu.add_command(label="Clear backgating",
                                 command=self._clear_backgate)
            menu.add_separator()
            menu.add_command(
                label="Export population as FCS…",
                command=lambda n=parsed[1], gd=parsed[2]:
                    self._export_population_fcs(n, gd))
            menu.add_separator()
            menu.add_command(label="Delete gate (cascade)",
                             command=self._clear_selected_gate)
        elif parsed and parsed[0] == 'method':
            n, gd, mkey = parsed[1], parsed[2], parsed[3]
            if mkey == 'debris':
                _g, m = self._autoclean_method(n, gd, 'debris')
                mp = (m.get('params') if m else {}) or {}
                mode = mp.get('mode', 'bead')
                sub = tk.Menu(menu, tearoff=0)
                sub.add_command(
                    label=("• " if mode == 'bead' else "    ")
                          + "Beads (absolute size)",
                    command=lambda: self._autoclean_set_debris_mode(n, gd, 'bead'))
                sub.add_command(
                    label=("• " if mode == 'valley' else "    ")
                          + "Auto valley (density)",
                    command=lambda: self._autoclean_set_debris_mode(n, gd, 'valley'))
                menu.add_cascade(label="Debris method", menu=sub)
                menu.add_command(
                    label="Bead size (µm)…",
                    command=lambda: self._autoclean_prompt_float(
                        n, gd, 'debris', 'bead_um', "Bead size",
                        "Calibration bead diameter (µm):", 8.0, 0.1))
                menu.add_command(
                    label="Min cell size (µm)…",
                    command=lambda: self._autoclean_prompt_float(
                        n, gd, 'debris', 'min_um', "Min cell size",
                        "Smallest size to keep (µm):", 4.0, 0.0))
                menu.add_command(
                    label="Re-detect bead reference",
                    command=lambda: self._autoclean_set_debris_mode(n, gd, 'bead'))
                menu.add_separator()
            elif mkey == 'viability':
                _g, m = self._autoclean_method(n, gd, 'viability')
                cur = ((m.get('params') if m else {}) or {}).get('channel')
                sd = getattr(self._samples.get(n), 'data', None)
                sub = tk.Menu(menu, tearoff=0)
                sub.add_command(
                    label=("• " if not cur else "    ") + "Auto-detect",
                    command=lambda: self._autoclean_set_viability_channel(n, gd, None))
                if sd is not None:
                    for col in list(sd.columns):
                        cl = str(col)
                        if (cl.lower() == 'time' or cl.endswith('_pos')
                                or cl.upper().startswith(('FSC', 'SSC'))):
                            continue
                        sub.add_command(
                            label=("• " if col == cur else "    ") + cl,
                            command=lambda c=col:
                                self._autoclean_set_viability_channel(n, gd, c))
                menu.add_cascade(label="Viability channel", menu=sub)
                menu.add_separator()
            menu.add_command(
                label="Edit auto-clean parameters…",
                command=lambda: self._edit_autoclean_params(n, gd))
        elif parsed and parsed[0] == 'sample':
            menu.add_command(label="Create boolean gate…",
                             command=self._open_boolean_dialog)
            menu.add_separator()
            menu.add_command(label="Copy gates (Ctrl+C)",
                             command=self._on_copy)
            menu.add_command(label="Paste (Ctrl+V)",
                             command=self._on_paste, state=paste_state)
            menu.add_separator()
            menu.add_command(label="Copy sample file path",
                             command=lambda: self._copy_sample_path(parsed[1]))
            menu.add_command(label="Copy gates to other samples…",
                             command=self._open_copy_gates_dialog)
            menu.add_separator()
            menu.add_command(label="Clear sample's gates",
                             command=self._clear_selected_gate)
            menu.add_command(label="Remove sample",
                             command=self._remove_selected)
        elif parsed and parsed[0] == 'trial':
            menu.add_command(label="Clear trial's gates (samples kept)",
                             command=self._clear_selected_gate)
            menu.add_command(label="Remove trial (samples + gates)",
                             command=self._remove_selected)
        else:
            menu.add_command(label="Add FCS…",
                             command=self._add_samples)
            menu.add_command(label="Paste files (Ctrl+V)",
                             command=self._on_paste, state=paste_state)

        try:
            menu.tk_popup(event.x_root, event.y_root)
        except Exception as exc:
            print(f"[context menu] popup failed: {exc}", flush=True)
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass
        return 'break'

    def _copy_sample_path(self, name):
        sample = self._samples.get(name)
        path = getattr(sample, 'path', None) if sample else None
        if not path:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(path)
        except Exception:
            pass
        self.status_var.set(f"Copied path of '{name}' to clipboard.")

    def _on_dnd_drop(self, event):
        """OS file-drop onto the gate tree. Parses tkdnd's spaces-and-
        braces filelist format using Tk's own splitlist, then imports any
        .fcs / .wsp entries. Dropped folders are walked recursively, so a
        whole trial folder (or a parent of several trial folders) imports
        in one gesture, each sample grouped under its own trial. Everything
        is wrapped in defensive try/except so a malformed drop can't crash
        the GUI."""
        try:
            raw = getattr(event, 'data', '') or ''
            print(f"[DnD] drop event raw={raw!r}", flush=True)
            try:
                paths = list(self.tk.splitlist(raw))
            except Exception:
                paths = raw.split()
            if paths:
                self._import_dropped_paths(paths)
        except Exception as exc:
            print(f"[DnD] drop handler failed: {type(exc).__name__}: {exc}",
                  flush=True)
        try:
            return event.action
        except Exception:
            return None

    def _is_descendant_of(self, sample_name, ancestor_gid, candidate_gid):
        """True if `candidate_gid` is `ancestor_gid` itself or anywhere
        below it in `sample_name`'s tree. Cycle-safe."""
        sgates = self._sample_gates.get(sample_name, {})
        cur = candidate_gid
        seen = set()
        while cur is not None and cur not in seen:
            if cur == ancestor_gid:
                return True
            seen.add(cur)
            g = sgates.get(cur)
            if g is None:
                return False
            cur = g.get('parent_id')
        return False

    def _toggle_all_sample_plots(self):
        """Header ✓ click: if ANY loaded sample is plot-on, turn them all
        off; otherwise turn them all on. Gates are untouched."""
        if not self._samples:
            return
        any_on = any(self._sample_plot_enabled.get(n, True)
                     for n in self._samples)
        new_state = not any_on
        for name in self._samples:
            self._sample_plot_enabled[name] = new_state
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _clear_selected_gate(self):
        """Clear button. Removes gates based on the current selection — the
        samples themselves are never removed (use Remove for that):
          • gate row   → that gate AND all its descendants (cascade)
          • sample row → every gate on that sample (the sample stays)
          • trial row  → every gate on all samples in that trial
        Handles a multi-row selection as a single undo step."""
        sel = self.gate_tv.selection()
        if not sel:
            self.status_var.set("Select a gate, sample, or trial to clear.")
            return
        gate_targets = []        # (sample_name, gid) → cascade
        sample_targets = []      # sample_name → all gates
        for iid in sel:
            p = self._parse_iid(iid)
            if p is None:
                continue
            if p[0] == 'gate':
                gate_targets.append((p[1], p[2]))
            elif p[0] == 'sample':
                sample_targets.append(p[1])
            elif p[0] == 'trial':
                sample_targets.extend(self._trial_members(p[1]))
        if not gate_targets and not sample_targets:
            self.status_var.set("Select a gate, sample, or trial to clear.")
            return
        self._checkpoint()
        removed = 0
        # Whole-sample clears first; a gate caught by both (its sample is also
        # selected) is then a no-op rather than a double-count.
        for name in dict.fromkeys(sample_targets):
            removed += self._purge_all_gates(name)
        for name, gid in gate_targets:
            removed += self._purge_gate_subtree(name, gid)
        self._refresh_gate_list()
        self._schedule_replot(0)
        if removed:
            self._audit('gate.remove', n_gates=removed,
                        samples=sorted({n for n in sample_targets}
                                       | {n for n, _ in gate_targets}))
        self.status_var.set(
            f"Cleared {removed} gate(s)." if removed else "No gates to clear.")

    def _remove_gate_cascade_in(self, sample_name, gid):
        """Cascade-delete a gate + its descendants from any sample's tree
        (not just the active one). Undoable."""
        self._checkpoint()
        self._purge_gate_subtree(sample_name, gid)

    def _purge_gate_subtree(self, sample_name, gid):
        """Remove ``gid`` and every descendant from ``sample_name``'s gate tree,
        in place (the caller owns the undo checkpoint). Mutating the existing
        dict/list — not rebinding them — keeps the active sample's ``self._gates``
        / ``_gate_id_order`` references valid. Returns the number removed."""
        target_gates = self._sample_gates.get(sample_name, {})
        order = self._sample_gate_order.get(sample_name, [])
        to_remove = [gid]
        seen = set()
        while to_remove:
            cur = to_remove.pop()
            if cur in seen or cur not in target_gates:
                continue
            seen.add(cur)
            for other_gid, og in target_gates.items():
                if og.get('parent_id') == cur:
                    to_remove.append(other_gid)
        for victim in seen:
            target_gates.pop(victim, None)
            if victim in order:
                order.remove(victim)
        return len(seen)

    def _purge_all_gates(self, sample_name):
        """Remove every gate from ``sample_name`` in place (caller owns the undo
        checkpoint), leaving the sample itself loaded. Clears the dict/list in
        place so the active sample's ``self._gates`` / ``_gate_id_order``
        references stay valid. Returns the number removed."""
        gates = self._sample_gates.get(sample_name)
        if not gates:
            return 0
        n = len(gates)
        gates.clear()
        order = self._sample_gate_order.get(sample_name)
        if order is not None:
            order.clear()
        return n

    def _find_area_height_channels(self):
        """Best-guess (area, height) scatter channel pair for a singlet gate.
        Prefers FSC-A/FSC-H, then SSC-A/SSC-H. Returns (area, height) or
        (None, None)."""
        chans = list(self._channels)
        up = {c: c.upper() for c in chans}
        for stem in ('FSC', 'SSC'):
            area = next((c for c in chans
                         if stem in up[c] and '-A' in up[c]), None)
            height = next((c for c in chans
                           if stem in up[c] and '-H' in up[c]), None)
            if area and height:
                return area, height
        return None, None

    def _auto_gate(self):
        """Open the auto-gate dialog: well-posed, reviewable gate proposals
        (singlet ratio-band, BIC-selected GMM ellipses, or 1-D valley/Otsu
        threshold), each reported with a quality score. Proposals are added
        as ordinary undoable gates for the user to accept / tweak / delete."""
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        area, height = self._find_area_height_channels()
        AutoGateDialog(self, has_singlet=bool(area and height),
                       area=area, height=height,
                       cur_x=self._fmt_channel(x) if x else '',
                       cur_y=self._fmt_channel(y) if y else '',
                       on_apply=self._run_auto_gate)

    def _run_auto_gate(self, opts):
        """Execute the chosen auto-gate method against the active sample and
        add the proposal(s). ``opts`` comes from AutoGateDialog."""
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        method = opts.get('method')
        if method == 'singlet':
            self._auto_gate_singlet(name, opts)
        elif method == 'gmm':
            self._auto_gate_gmm(name, opts)
        elif method == 'threshold':
            self._auto_gate_threshold(name)
        self._schedule_replot(0)

    def _auto_gate_singlet(self, name, opts):
        from .pipeline import auto_singlet_gate
        area, height = opts.get('area'), opts.get('height')
        if not (area and height):
            self.status_var.set("No FSC-A/FSC-H pair found for a singlet gate.")
            return
        df = self._get_df(name, area, height)
        if area not in df.columns or height not in df.columns:
            self.status_var.set("Active sample lacks the FSC-A/FSC-H channels.")
            return
        verts, q = auto_singlet_gate(
            np.asarray(df[area].values, dtype=float),
            np.asarray(df[height].values, dtype=float),
            k=float(opts.get('k', 3.0)))
        if not verts or q is None:
            self.status_var.set(
                "Singlet gate: ratio band undefined (too little spread/data).")
            return
        self._add_gate({'kind': 'polygon', 'x_channel': area,
                        'y_channel': height, 'vertices': verts,
                        'name': 'Singlets'}, audit=False)
        # Switch the view so the user sees what was proposed.
        self.x_combo.set(self._fmt_channel(area))
        self.y_combo.set(self._fmt_channel(height))
        if self.mode_var.get() == 'histogram':
            self.mode_var.set('pseudocolor')
        trust = ('clean' if q['frac_kept'] > 0.8 and q['ratio_cv'] < 0.12
                 else 'REVIEW')
        self._audit('autogate.singlet', sample=name, area=area, height=height,
                    k=float(opts.get('k', 3.0)),
                    frac_kept=round(q['frac_kept'], 4),
                    ratio_cv=round(q['ratio_cv'], 4), trust=trust)
        self.status_var.set(
            f"Singlet gate [{trust}]: keeps {q['frac_kept'] * 100:.1f}% "
            f"(ratio CV {q['ratio_cv']:.3f}). Drag vertices to adjust.")

    def _auto_gate_gmm(self, name, opts):
        from .pipeline import gmm_ellipse_gates
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        if not x or not y:
            self.status_var.set("Pick X and Y channels for a 2-D auto-gate.")
            return
        df = self._get_df(name, x, y)
        if x not in df.columns or y not in df.columns:
            self.status_var.set("Active sample lacks those channels.")
            return
        proposals = gmm_ellipse_gates(
            np.asarray(df[x].values, dtype=float),
            np.asarray(df[y].values, dtype=float),
            max_components=int(opts.get('max_components', 6)),
            coverage=float(opts.get('coverage', 0.90)),
            min_weight=float(opts.get('min_weight', 0.02)))
        if not proposals:
            self.status_var.set("Auto-gate: no populations found (too little "
                                "data or no structure).")
            return
        weak = 0
        for i, (gate, info) in enumerate(proposals, 1):
            gate = dict(gate, x_channel=x, y_channel=y, name=f'Pop {i}')
            self._add_gate(gate, audit=False)
            if info.get('separation') is not None and info['separation'] < 2.0:
                weak += 1
        k = proposals[0][1]['n_components']
        note = (f" — {weak} overlap heavily (separation < 2); review those"
                if weak else "")
        self._audit('autogate.gmm', sample=name, x=x, y=y,
                    n_populations=len(proposals), k_bic=k,
                    coverage=float(opts.get('coverage', 0.90)),
                    weak_overlap=weak)
        self.status_var.set(
            f"GMM found {len(proposals)} population(s) of k={k} (BIC). "
            f"Added as ellipse gates{note}.")

    def _auto_gate_threshold(self, name):
        from .pipeline import auto_threshold
        x = self._resolve_channel(self.x_combo.get())
        if not x:
            self.status_var.set("Pick an X channel first.")
            return
        df = self._get_df(name, x)
        if x not in df.columns:
            self.status_var.set("Active sample lacks that channel.")
            return
        thr = auto_threshold(np.asarray(df[x].values, dtype=float))
        if thr is None:
            self.status_var.set("Auto-gate: not enough data to split.")
            return
        self._add_gate({'kind': 'threshold', 'channel': x,
                        'value': float(thr)}, audit=False)
        self._audit('autogate.threshold', sample=name, channel=x,
                    value=float(thr))
        self.status_var.set(
            f"Auto threshold on {self._fmt_channel(x)} = {thr:.3g}.")

    def _apply_channel_transforms(self, new_methods):
        """Re-transform channels across ALL loaded samples by inverting each
        channel's current transform and applying the new one (so no
        re-compensation is needed). Returns the number of channels changed."""
        from .pipeline import inverse_transform_values, transform_values
        changed = {c: m for c, m in new_methods.items()
                   if m != self._channel_transform.get(c, 'linear')}
        if not changed:
            return 0
        for s in self._samples.values():
            cols = set(s.data.columns)
            for ch, new_m in changed.items():
                if ch not in cols:
                    continue
                old_m = self._channel_transform.get(ch, 'linear')
                lin = inverse_transform_values(
                    np.asarray(s.data[ch].values, dtype=float), method=old_m)
                s.data[ch] = transform_values(lin, method=new_m)
        self._channel_transform.update(changed)
        self._audit('transform', n_channels=len(changed),
                    changes={ch: m for ch, m in changed.items()})
        return len(changed)

    def _open_transform_editor(self):
        """Per-channel display-transform editor. Re-maps each channel's
        transform across every loaded sample. Gates already drawn on a
        re-transformed channel keep their old coordinates, so the user is
        warned to re-check them."""
        from .pipeline import TRANSFORM_METHODS
        if not self._samples:
            self.status_var.set("Load a sample first.")
            return
        dlg = tk.Toplevel(self)
        dlg.title("Channel transforms")
        dlg.transient(self)
        dlg.grab_set()
        dlg.geometry("380x460")

        ttk.Label(dlg, text="Transform per channel:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 4))

        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=6)
        cv = tk.Canvas(holder, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))

        combos = {}
        for ch in self._channels:
            row = ttk.Frame(inner)
            row.pack(side='top', fill='x', pady=1)
            ttk.Label(row, text=self._fmt_channel(ch), width=22).pack(
                side='left')
            var = tk.StringVar(
                value=self._channel_transform.get(ch, 'linear'))
            ttk.Combobox(row, textvariable=var, width=10, state='readonly',
                         values=list(TRANSFORM_METHODS)).pack(side='left')
            combos[ch] = var

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)

        def do_apply():
            new = {ch: var.get() for ch, var in combos.items()}
            dlg.destroy()
            n = self._apply_channel_transforms(new)
            if n:
                self.status_var.set(
                    f"Re-transformed {n} channel(s). Gates on those channels "
                    "may need re-checking.")
                self._schedule_replot(0)
            else:
                self.status_var.set("No transform changes.")

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Apply", command=do_apply).pack(
            side='right', padx=(0, 6))

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
        self.x_combo['values'] = disp
        self.y_combo['values'] = disp
        self.color_combo['values'] = ['By sample', 'By density'] + disp

    def _open_cluster_dialog(self):
        """Run unsupervised clustering (+ optional UMAP) on loaded samples,
        in a worker thread, then auto-import the resulting populations."""
        if not self._samples:
            self.status_var.set("Load a sample first.")
            return
        if getattr(self, '_clustering_busy', False):
            self.status_var.set("Clustering already running…")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Cluster")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        method_var = tk.StringVar(value='phenograph')
        mrow = ttk.Frame(dlg)
        mrow.grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=4)
        ttk.Label(mrow, text="Method:").pack(side='left')
        ttk.Radiobutton(mrow, text="Phenograph", value='phenograph',
                        variable=method_var).pack(side='left', padx=(6, 0))
        ttk.Radiobutton(mrow, text="FlowSOM", value='flowsom',
                        variable=method_var).pack(side='left', padx=(6, 0))
        ttk.Radiobutton(mrow, text="Leiden", value='leiden',
                        variable=method_var).pack(side='left', padx=(6, 0))

        ttk.Label(dlg, text="Phenograph/Leiden k:").grid(
            row=1, column=0, sticky='w', padx=10, pady=4)
        k_var = tk.IntVar(value=30)
        ttk.Spinbox(dlg, from_=5, to=200, textvariable=k_var, width=8).grid(
            row=1, column=1, sticky='w', padx=10, pady=4)

        ttk.Label(dlg, text="Leiden resolution:").grid(
            row=1, column=2, sticky='w', padx=10, pady=4)
        res_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(dlg, from_=0.1, to=5.0, increment=0.1, textvariable=res_var,
                    width=6).grid(row=1, column=3, sticky='w', padx=(0, 10),
                                  pady=4)

        ttk.Label(dlg, text="FlowSOM grid (NxN):").grid(row=2, column=0,
                                                        sticky='w', padx=10, pady=4)
        grid_var = tk.IntVar(value=10)
        ttk.Spinbox(dlg, from_=4, to=20, textvariable=grid_var, width=8).grid(
            row=2, column=1, sticky='w', padx=10, pady=4)

        ttk.Label(dlg, text="FlowSOM metaclusters:").grid(row=3, column=0,
                                                          sticky='w', padx=10, pady=4)
        meta_var = tk.IntVar(value=10)
        ttk.Spinbox(dlg, from_=2, to=40, textvariable=meta_var, width=8).grid(
            row=3, column=1, sticky='w', padx=10, pady=4)

        ttk.Label(dlg, text="Embedding (for visualisation):").grid(
            row=4, column=0, sticky='w', padx=10, pady=4)
        embed_var = tk.StringVar(value='UMAP')
        ttk.Combobox(dlg, textvariable=embed_var, width=10, state='readonly',
                     values=['UMAP', 't-SNE', 'TriMap', 'PaCMAP', 'PHATE',
                             'none']).grid(row=4, column=1, sticky='w',
                                           padx=10, pady=4)
        all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg, text="Run on all loaded samples",
                        variable=all_var).grid(row=5, column=0, columnspan=2,
                                               sticky='w', padx=10, pady=4)

        btns = ttk.Frame(dlg)
        btns.grid(row=6, column=0, columnspan=2, sticky='ew', padx=10, pady=4)

        def do_run():
            params: dict = dict(
                method=method_var.get(), all_samples=all_var.get(),
                k=int(k_var.get()), grid=int(grid_var.get()),
                n_meta=int(meta_var.get()), embedding=embed_var.get(),
                resolution=float(res_var.get()))
            dlg.destroy()
            self._run_clustering(**params)

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Run", command=do_run).pack(
            side='right', padx=(0, 6))

    # Embedding picker → (FlowSample method, axis-column prefix).
    _EMBEDDINGS = {
        'UMAP':   ('run_umap', 'UMAP'),
        't-SNE':  ('run_tsne', 'TSNE'),
        'TriMap': ('run_trimap', 'TRIMAP'),
        'PaCMAP': ('run_pacmap', 'PACMAP'),
        'PHATE':  ('run_phate', 'PHATE'),
    }

    def _run_clustering(self, method, all_samples, k, grid, n_meta,
                        embedding='UMAP', resolution=1.0):
        targets = [n for n in (self._sample_order if all_samples
                               else [self._active_sample])
                   if n in self._samples]
        if not targets:
            self.status_var.set("No sample selected.")
            return
        self._clustering_busy = True
        self.status_var.set(
            f"{method} clustering {len(targets)} sample(s)… "
            "(this can take a while)")
        emb_method, emb_prefix = self._EMBEDDINGS.get(embedding, (None, None))

        def work():
            try:
                for name in targets:
                    s = self._samples.get(name)
                    if s is None:
                        continue
                    if method == 'phenograph':
                        s.cluster(k=k)
                    elif method == 'leiden':
                        s.run_leiden(n_neighbors=k, resolution=resolution)
                    else:
                        s.run_flowsom(grid=(grid, grid), n_metaclusters=n_meta)
                    if emb_method:
                        getattr(s, emb_method)()
                self.after(0, lambda: self._finish_clustering(
                    method, emb_prefix, targets))
            except Exception as exc:
                self.after(0, lambda e=exc: self._clustering_error(e))

        threading.Thread(target=work, daemon=True).start()

    def _finish_clustering(self, method, emb_prefix, targets):
        self._clustering_busy = False
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
        self.status_var.set(f"Clustering failed: {exc}")

    def _open_boolean_dialog(self):
        """Build a boolean population (AND / OR / NOT) from the active
        sample's existing gates. The result is a root 'boolean' gate whose
        operands are the chosen gates (evaluated as their cumulative
        masks)."""
        from .pipeline import describe_gate
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        gates = self._sample_gates.get(name, {})
        order = [g for g in self._sample_gate_order.get(name, [])
                 if g in gates and gates[g].get('kind') != 'boolean']
        if len(order) < 1:
            self.status_var.set("Need at least one gate to combine.")
            return

        dlg = tk.Toplevel(self)
        dlg.title(f"Boolean gate — {name}")
        dlg.transient(self)
        dlg.grab_set()
        dlg.geometry("360x420")

        ttk.Label(dlg, text="Combine these gates:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 4))

        op_var = tk.StringVar(value='and')
        oprow = ttk.Frame(dlg)
        oprow.pack(side='top', fill='x', padx=10)
        for lbl, val in [('AND', 'and'), ('OR', 'or'), ('NOT', 'not')]:
            ttk.Radiobutton(oprow, text=lbl, value=val,
                            variable=op_var).pack(side='left', padx=(0, 8))

        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=6)
        cv = tk.Canvas(holder, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))

        cb_vars = {}
        for gid in order:
            var = tk.BooleanVar(value=False)
            cb_vars[gid] = var
            ttk.Checkbutton(inner, text=describe_gate(gates[gid]),
                            variable=var).pack(side='top', anchor='w')

        namerow = ttk.Frame(dlg)
        namerow.pack(side='top', fill='x', padx=10, pady=(4, 0))
        ttk.Label(namerow, text="Name:").pack(side='left')
        name_var = tk.StringVar(value='')
        ttk.Entry(namerow, textvariable=name_var).pack(
            side='left', fill='x', expand=True)

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)

        def do_create():
            operands = [g for g, v in cb_vars.items() if v.get()]
            if not operands:
                self.status_var.set("Select at least one gate.")
                return
            op = op_var.get()
            nm = name_var.get().strip() or (
                f"{op.upper()} ({len(operands)})")
            dlg.destroy()
            self._add_gate({'kind': 'boolean', 'op': op,
                            'operands': operands, 'name': nm,
                            'parent_id': None})
            self.status_var.set(f"Boolean gate '{nm}' created.")
            self._schedule_replot(0)

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Create", command=do_create).pack(
            side='right', padx=(0, 6))

    def _clear_all(self):
        """Clear gates from EVERY loaded sample, leaving the samples themselves
        intact. Auto-clean gates are PRESERVED by default (they're the cleaning
        foundation) — a checkbox in the confirm dialog opts into clearing them
        too. Confirmed (bulk wipe) but undoable. To remove samples, use Remove.
        """
        total = sum(len(g) for g in self._sample_gates.values())
        if total == 0:
            self.status_var.set("No gates to clear.")
            return
        n = sum(1 for g in self._sample_gates.values() if g)
        n_ac = sum(1 for gates in self._sample_gates.values()
                   for g in gates.values() if g.get('kind') == 'autoclean')
        include_ac = self._ask_clear_all(total, n, n_ac)
        if include_ac is None:                 # cancelled
            return
        self._checkpoint()
        removed = 0
        for name in list(self._samples):
            gates = self._sample_gates.get(name, {})
            order = self._sample_gate_order.get(name, [])
            victims = [gid for gid, g in gates.items()
                       if include_ac or g.get('kind') != 'autoclean']
            for gid in victims:
                gates.pop(gid, None)
                if gid in order:
                    order.remove(gid)
                removed += 1
        self._refresh_gate_list()
        self._schedule_replot(0)
        kept = (" (auto-clean kept)" if n_ac and not include_ac else "")
        self.status_var.set(
            f"Cleared {removed} gate(s) from all samples{kept}; samples kept.")

    def _ask_clear_all(self, total, n_samples, n_autoclean):
        """Confirm Clear-all. Returns True (also clear auto-clean), False (keep
        auto-clean), or None (cancelled). With no auto-clean gates present it's
        a plain yes/no (returns False/None)."""
        if n_autoclean == 0:
            ok = messagebox.askyesno(
                "Clear all gates",
                f"Remove all {total} gate(s) from {n_samples} sample(s)?\n"
                "Samples stay loaded. (Undoable.)",
                parent=self)
            return False if ok else None

        dlg = tk.Toplevel(self)
        dlg.title("Clear all gates")
        dlg.transient(self)
        dlg.resizable(False, False)
        ttk.Label(
            dlg, justify='left', padding=(12, 12, 12, 6),
            text=(f"Remove gates from {n_samples} sample(s)?\n"
                  "Samples stay loaded. (Undoable.)")).pack(anchor='w')
        inc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            dlg, variable=inc_var, padding=(12, 0),
            text=(f"Also clear the {n_autoclean} auto-clean gate(s) "
                  "(off = keep cleaning)")).pack(anchor='w')
        result: dict[str, object] = {'val': None}

        def _ok():
            result['val'] = bool(inc_var.get())
            dlg.destroy()

        btns = ttk.Frame(dlg, padding=12)
        btns.pack(anchor='e')
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(
            side='right')
        ttk.Button(btns, text="Clear", command=_ok).pack(
            side='right', padx=(0, 6))
        dlg.bind('<Escape>', lambda _e: dlg.destroy())
        dlg.bind('<Return>', lambda _e: _ok())
        try:
            dlg.grab_set()
            self.wait_window(dlg)
        except Exception:
            pass
        return result['val']

    # ── Mode / tool / selector lifecycle ─────────────────────────────────

    def _on_mode_changed(self):
        """Switching plot mode rebuilds the canvas and toggles the
        histogram slider panel — replot first, then sync the slider gate."""
        self._sync_hist_y_combo()
        self._schedule_replot(0)

    def _sync_hist_y_combo(self):
        """Enable the histogram Y-axis selector only in histogram mode."""
        combo = getattr(self, 'hist_y_combo', None)
        if combo is None:
            return
        try:
            combo.configure(
                state=('readonly' if self.mode_var.get() == 'histogram'
                       else 'disabled'))
        except Exception:
            pass

    # Per-tool gesture hint. Edit mode has the densest gesture map so
    # we always surface it; the others get a short reminder.
    _TOOL_HINTS = {
        'quadrant':  "Quadrant: double-click (or shift-click) the plot to drop 4 quadrant gates.",
        'rectangle': "Rectangle: click-drag on the plot to draw a rectangle gate.",
        'polygon':   "Polygon: click to drop vertices, double-click (or close near the first vertex) to finish.",
        'ellipse':   "Ellipse: click-drag to draw an axis-aligned ellipse. Switch to Edit to move / resize / rotate it.",
        'lasso':     "Lasso: click-drag a free-form outline; releases close the polygon.",
        'edit':      "Edit: left-drag moves vertex/line/ellipse  •  shift+left-drag moves whole gate  "
                     "•  drag an ellipse's rim handle to resize, its top handle to rotate  "
                     "•  right-click on vertex = delete, on edge = add  •  alt+left-click adds vertex",
    }

    def _activate_gate_tool(self, *_):
        """Wire up the matplotlib region selector matching the current
        tool. Called when the user switches tool, when _replot rebuilds
        the axes, or when the plot mode changes."""
        # Tear down any prior selector.
        if self._selector is not None:
            try:
                self._selector.set_active(False)
                self._selector.disconnect_events()
            except Exception:
                pass
            self._selector = None

        tool = self.gate_tool_var.get()
        mode = self.mode_var.get()
        # Update the gesture hint label next to the tool radios.
        if hasattr(self, 'tool_hint_var'):
            self.tool_hint_var.set(self._TOOL_HINTS.get(tool, ''))
        # Region tools don't make sense in histogram mode.
        # Edit mode also has no matplotlib Selector — its gestures are
        # handled by `_on_press` against the existing hit-test results.
        if tool in ('quadrant', 'edit') or mode == 'histogram':
            return
        # Need an X+Y channel for any region tool.
        if not (self._resolve_channel(self.x_combo.get())
                and self._resolve_channel(self.y_combo.get())):
            return

        from matplotlib.widgets import (
            EllipseSelector,
            LassoSelector,
            PolygonSelector,
            RectangleSelector,
        )
        props = dict(color='red', linestyle='--', linewidth=1.3, alpha=0.9)
        try:
            if tool == 'rectangle':
                self._selector = RectangleSelector(
                    self.ax, self._on_rect_select,
                    useblit=True,
                    minspanx=3, minspany=3, spancoords='pixels',
                    interactive=False,
                    props=props)
            elif tool == 'ellipse':
                self._selector = EllipseSelector(
                    self.ax, self._on_ellipse_select,
                    useblit=True,
                    minspanx=3, minspany=3, spancoords='pixels',
                    interactive=False,
                    props=props)
            elif tool == 'polygon':
                # matplotlib's runtime PolygonSelector passes onselect(verts)
                # (list of (x,y)) — its stubs incorrectly declare (x, y) pair.
                self._selector = PolygonSelector(
                    self.ax,
                    self._on_poly_select,  # type: ignore[arg-type]
                    useblit=True, props=props)
            elif tool == 'lasso':
                self._selector = LassoSelector(
                    self.ax,
                    self._on_lasso_select,  # type: ignore[arg-type]
                    useblit=True, props=props)
        except Exception as exc:
            # Older matplotlibs may not accept `props=`; fall back without it.
            print(f"[gate-tool] selector init failed ({exc}); retrying "
                  f"without props", flush=True)
            try:
                if tool == 'rectangle':
                    self._selector = RectangleSelector(
                        self.ax, self._on_rect_select,
                        useblit=True,
                        minspanx=3, minspany=3, spancoords='pixels',
                        interactive=False)
                elif tool == 'ellipse':
                    self._selector = EllipseSelector(
                        self.ax, self._on_ellipse_select,
                        useblit=True,
                        minspanx=3, minspany=3, spancoords='pixels',
                        interactive=False)
                elif tool == 'polygon':
                    self._selector = PolygonSelector(
                        self.ax,
                        self._on_poly_select,  # type: ignore[arg-type]
                        useblit=True)
                elif tool == 'lasso':
                    self._selector = LassoSelector(
                        self.ax,
                        self._on_lasso_select,  # type: ignore[arg-type]
                        useblit=True)
            except Exception as exc2:
                print(f"[gate-tool] selector init still failed: {exc2}",
                      flush=True)
                self._selector = None

    def _on_rect_select(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        if not x or not y or self.mode_var.get() == 'histogram':
            return
        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        self._add_gate({'kind': 'rect', 'x_channel': x, 'y_channel': y,
                        'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1})
        self._schedule_replot(0)

    def _on_ellipse_select(self, eclick, erelease):
        """EllipseSelector gives the press + release corners of the
        bounding box. Build an axis-aligned ellipsoid gate inscribed in
        that box: mean = box centre, and a diagonal covariance whose
        semi-axes equal the box half-extents at distance_sq = 1
        (so (x-µ)²/a² + (y-µ)²/b² ≤ 1 is exactly the inscribed ellipse).
        """
        if eclick.xdata is None or erelease.xdata is None:
            return
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        if not x or not y or self.mode_var.get() == 'histogram':
            return
        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        a = (x1 - x0) / 2.0    # semi-axis along x
        b = (y1 - y0) / 2.0    # semi-axis along y
        if a <= 0 or b <= 0:
            return
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        self._add_gate({
            'kind': 'ellipsoid', 'x_channel': x, 'y_channel': y,
            'mean': [cx, cy],
            'cov':  [[a * a, 0.0], [0.0, b * b]],
            'distance_sq': 1.0,
        })
        self._schedule_replot(0)

    def _on_poly_select(self, verts):
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        if not x or not y or len(verts) < 3:
            return
        self._add_gate({'kind': 'polygon', 'x_channel': x, 'y_channel': y,
                        'vertices': [[float(vx), float(vy)] for vx, vy in verts]})
        self._schedule_replot(0)

    def _on_lasso_select(self, verts):
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        if not x or not y or len(verts) < 3:
            return
        self._add_gate({'kind': 'polygon', 'x_channel': x, 'y_channel': y,
                        'vertices': [[float(vx), float(vy)] for vx, vy in verts]})
        self._schedule_replot(0)

    # ── Histogram slider gate ────────────────────────────────────────────

    def _show_slider_panel(self, show):
        if show:
            self.slider_panel.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        else:
            self.slider_panel.grid_remove()

    def _sync_slider_panel(self):
        """Visibility + range for the slider panel, based on current mode
        and X channel. Called from _replot at the end."""
        self._sync_hist_y_combo()
        mode = self.mode_var.get()
        if mode != 'histogram':
            self._show_slider_panel(False)
            return
        x = self._resolve_channel(self.x_combo.get())
        if not x:
            self._show_slider_panel(False)
            return
        self._show_slider_panel(True)
        # Range: derived from the currently-displayed data's x range.
        try:
            xl, xh = self.ax.get_xlim()
        except Exception:
            xl, xh = 0.0, 1.0
        if not np.isfinite(xl) or not np.isfinite(xh) or xh <= xl:
            xl, xh = 0.0, 1.0
        self.slider_lo.configure(from_=float(xl), to=float(xh))
        self.slider_hi.configure(from_=float(xl), to=float(xh))
        self._slider_axis_label.configure(
            text=f"channel: {self._fmt_channel(x)}")
        # If the X channel changed, seed the slider value(s) from the
        # current gate-on-this-channel (or the axis midpoint).
        if x != self._slider_channel:
            self._slider_channel = x
            self._slider_gate_id = self._find_1d_gate_id_for(x)
            self._seed_sliders_from_gate(xl, xh)
        # UI-only refresh: show/hide the hi slider, update labels. We
        # explicitly DO NOT call _commit_slider_to_gate here — the
        # slider panel should not create gates on its own when the user
        # merely switches mode or channel. Only a user-driven slider
        # drag commits.
        self._update_slider_ui()

    def _find_1d_gate_id_for(self, ch):
        for gid, g in self._gates.items():
            if g.get('channel') == ch and g.get('kind') in ('threshold', 'interval'):
                return gid
        return None

    def _seed_sliders_from_gate(self, xl, xh):
        """Set slider positions from the existing gate, falling back to
        axis-midpoint quartiles when no gate exists yet."""
        self._slider_updating = True
        try:
            g = (self._gates.get(self._slider_gate_id)
                 if self._slider_gate_id else None)
            if g is None:
                mid = (xl + xh) * 0.5
                span = (xh - xl) * 0.25
                self.slider_lo.set(mid - span)
                self.slider_hi.set(mid + span)
            elif g['kind'] == 'threshold':
                self.slider_lo.set(float(g['value']))
                self.slider_hi.set(float(g['value']))
                self.slider_kind_var.set('threshold')
            elif g['kind'] == 'interval':
                self.slider_lo.set(float(g['lo']))
                self.slider_hi.set(float(g['hi']))
                self.slider_kind_var.set('interval')
        finally:
            self._slider_updating = False

    def _update_slider_ui(self):
        """Refresh the slider panel UI (hi slider visibility + labels).
        Touches NO gate state — safe to call when entering histogram mode
        or switching the Threshold/Interval radio. The user has to drag a
        slider to commit a gate; that path runs _commit_slider_to_gate."""
        kind = self.slider_kind_var.get()
        if kind == 'interval':
            self.slider_hi.grid(row=2, column=0, columnspan=3,
                                sticky='ew', padx=(0, 6), pady=(2, 4))
            self.slider_hi_lbl.grid(row=2, column=3, sticky='e', pady=(2, 4))
        else:
            self.slider_hi.grid_remove()
            self.slider_hi_lbl.grid_remove()
        lo = float(self.slider_lo.get())
        hi = float(self.slider_hi.get())
        if kind == 'threshold':
            self.slider_lo_lbl.configure(text=f"{lo:.3g}")
            self.slider_hi_lbl.configure(text='—')
        else:
            if lo > hi:
                lo, hi = hi, lo
            self.slider_lo_lbl.configure(text=f"lo {lo:.3g}")
            self.slider_hi_lbl.configure(text=f"hi {hi:.3g}")

    def _commit_slider_to_gate(self):
        """Build (or update) the 1D gate that matches the current slider
        state. Only the user-drag handlers and the explicit kind-switch
        path (when an existing gate would be silently mis-interpreted)
        should call this — entering histogram mode does NOT."""
        self._update_slider_ui()
        ch = self._slider_channel
        if not ch:
            return
        kind = self.slider_kind_var.get()
        lo = float(self.slider_lo.get())
        hi = float(self.slider_hi.get())
        if kind == 'threshold':
            new_gate = {'kind': 'threshold', 'channel': ch, 'value': lo}
        else:
            if lo > hi:
                lo, hi = hi, lo
            new_gate = {'kind': 'interval', 'channel': ch, 'lo': lo, 'hi': hi}
        # _add_gate replaces the existing 1D gate on this (channel, parent).
        self._slider_gate_id = self._add_gate(new_gate)
        self._refresh_gate_list()
        self._redraw_only_gates()

    def _on_slider_kind_changed(self):
        """Threshold ↔ Interval radio toggle. Updates the slider UI; only
        re-commits to a gate if one already exists for this channel (so
        the user's intent of 'change kind of my gate' takes effect)."""
        self._update_slider_ui()
        if self._slider_gate_id and self._slider_gate_id in self._gates:
            self._commit_slider_to_gate()

    def _on_slider_lo(self, *_):
        if self._slider_updating:
            return
        self._commit_slider_to_gate()
        if self.apply_gates_var.get():
            self._schedule_replot(150)

    def _on_slider_hi(self, *_):
        if self._slider_updating:
            return
        self._commit_slider_to_gate()
        if self.apply_gates_var.get():
            self._schedule_replot(150)

    # ── Per-channel axis scale + range ────────────────────────────────────

    def _open_axis_dialog(self, axis_letter):
        """Open the AxisConfigDialog for the channel currently bound to
        the X or Y combo. Updates per-channel state + replots on OK."""
        combo = self.x_combo if axis_letter == 'x' else self.y_combo
        ch = self._resolve_channel(combo.get())
        if not ch:
            self.status_var.set(
                f"Pick a {axis_letter.upper()} channel before configuring its axis.")
            return
        AxisConfigDialog(
            self,
            channel=ch,
            scale=self._channel_scale.get(ch, self._default_channel_scale),
            rng=self._channel_range.get(ch),
            on_apply=lambda s, r: self._set_axis_config(ch, s, r))

    def _set_axis_config(self, channel, scale, rng):
        """Persist a channel's scale + range and trigger a replot."""
        self._channel_scale[channel] = scale
        if rng is None:
            self._channel_range.pop(channel, None)
        else:
            self._channel_range[channel] = (float(rng[0]), float(rng[1]))
        self._schedule_replot(0)

    def _apply_axis_to_ax(self, channel, axis_letter, data_sample=None):
        """Apply this channel's display scale + range to the matplotlib axes.

        Called at the end of ``_replot`` for both X and Y (when present).

        For a channel whose data is baked into a nonlinear transform
        (logicle/hyperlog/asinh/log), linear/symlog/log are rendered as
        composite FuncScale VIEWS of the underlying linear intensity (see
        ``_axis_view_funcs``) — proper, independent views with no double-
        transform, and gates auto-follow. Linear-data channels (scatter)
        use matplotlib's native linear/symlog/log scale; for symlog we pick
        a ``linthresh`` from the data (5th percentile of |data|), else 1.0.
        """
        scale = self._channel_scale.get(channel, self._default_channel_scale)
        set_scale = (self.ax.set_xscale if axis_letter == 'x'
                     else self.ax.set_yscale)
        set_lim   = (self.ax.set_xlim if axis_letter == 'x'
                     else self.ax.set_ylim)
        funcs = self._axis_view_funcs(channel, data_sample)
        try:
            if funcs is not None:
                set_scale('function', functions=funcs)
            elif scale == 'log':
                set_scale('log')
            elif scale == 'symlog':
                # Same linthresh the density binning uses → bins align with
                # the axis (no boxy artefacts in the log decade).
                set_scale('symlog',
                          linthresh=self._symlog_linthresh(data_sample))
            else:
                set_scale('linear')
        except Exception:
            # E.g. log scale with non-positive data — fall back silently
            # to linear rather than crashing the plot.
            try:
                set_scale('linear')
            except Exception:
                pass
        rng = self._channel_range.get(channel)
        if rng is not None:
            try:
                set_lim(rng[0], rng[1])
            except Exception:
                pass

    # ── Figure layout / multi-panel export ───────────────────────────────

    def _render_into(self, ax, samples, x, y, mode, color,
                     draw_gates=True, draw_overlays=True):
        """Render one plot panel into an arbitrary matplotlib ``Axes``,
        reusing the live plotting pipeline.

        The ``_plot_*`` / ``_overlay_*`` / ``_draw_gates`` / ``_apply_axis``
        helpers all draw into ``self.ax`` on ``self.fig``. Rather than
        duplicate their logic, this temporarily points those attributes
        (plus the gate-artist registries and the colorbar handle) at the
        supplied ``ax`` and its parent figure, renders, then restores live
        state in a ``finally`` so the on-screen plot is never disturbed.
        Colorbars are suppressed in panels (see ``_suppress_panel_cbar``).
        """
        saved = (self.ax, self.fig, self._cbar,
                 self._vlines, self._hlines,
                 getattr(self, '_shape_artists', {}))
        suppress_prev = getattr(self, '_suppress_panel_cbar', False)
        self.ax = ax
        self.fig = ax.figure
        self._cbar = None
        self._vlines, self._hlines, self._shape_artists = {}, {}, {}
        self._suppress_panel_cbar = True
        try:
            if not samples or not x:
                ax.text(0.5, 0.5, '(nothing to plot)', ha='center',
                        va='center', transform=ax.transAxes,
                        fontsize=9, color='grey')
                return
            try:
                if mode == 'histogram':
                    self._plot_histogram(samples, x)
                elif mode == 'dot':
                    self._plot_dot(samples, x, y, color)
                elif mode == 'pseudocolor':
                    self._plot_pseudocolor(samples, x, y)
                elif mode == 'contour':
                    self._plot_contour(samples, x, y)
            except Exception as exc:
                ax.text(0.5, 0.5, f'Plot error:\n{exc}', ha='center',
                        va='center', transform=ax.transAxes,
                        fontsize=8, color='red')
            if draw_overlays:
                try:
                    self._overlay_removed_events(samples, x, y, mode)
                except Exception:
                    pass
                try:
                    self._overlay_backgate(samples, x, y)
                except Exception:
                    pass
            ax.set_xlabel(self._fmt_channel(x), fontsize=8)
            if mode != 'histogram' and y:
                ax.set_ylabel(self._fmt_channel(y), fontsize=8)
            first = samples[0] if samples else None
            sdata = None
            if first and x and first in self._samples:
                sdf = self._samples[first].data
                if x in sdf.columns:
                    sdata = sdf[x].values
            self._apply_axis_to_ax(x, 'x', sdata)
            if mode != 'histogram' and y:
                ydata = None
                if first and y and first in self._samples:
                    sdf = self._samples[first].data
                    if y in sdf.columns:
                        ydata = sdf[y].values
                self._apply_axis_to_ax(y, 'y', ydata)
            if draw_overlays:
                try:
                    self._draw_highlight_overlays(
                        samples, x, y if mode != 'histogram' else None)
                except Exception:
                    pass
            if draw_gates:
                try:
                    self._draw_gates(x, y if mode != 'histogram' else None)
                except Exception:
                    pass
            ax.tick_params(labelsize=7)
        finally:
            (self.ax, self.fig, self._cbar,
             self._vlines, self._hlines, self._shape_artists) = saved
            self._suppress_panel_cbar = suppress_prev

    def _build_layout_figure(self, panels, ncols, draw_gates=True,
                             panel_size=(3.2, 2.6), dpi=120, suptitle=None):
        """Assemble a multi-panel matplotlib ``Figure`` from a list of panel
        specs (each: ``{samples, x, y, mode, color, title}``). Returns the
        Figure, or ``None`` when there are no panels."""
        from matplotlib.figure import Figure
        n = len(panels)
        if n == 0:
            return None
        ncols = max(1, min(int(ncols), n))
        nrows = int(np.ceil(n / ncols))
        fig = Figure(figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                     dpi=dpi)
        for i, spec in enumerate(panels):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            self._render_into(ax, spec.get('samples') or [], spec.get('x'),
                              spec.get('y'), spec.get('mode', 'dot'),
                              spec.get('color', 'By density'),
                              draw_gates=draw_gates)
            title = spec.get('title')
            if title:
                ax.set_title(title, fontsize=8)
        if suptitle:
            fig.suptitle(suptitle, fontsize=11)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return fig

    @staticmethod
    def _short_sample(name, width=24):
        """Trim a sample name for a panel title."""
        name = str(name)
        return name if len(name) <= width else name[:width - 1] + '…'

    def _resolve_token_to_channel(self, tok):
        """Map a user token (channel name, ``Label (DET)`` form, or a marker
        label like ``CD34``) to a real channel name, or ``None``."""
        tok = (tok or '').strip()
        if not tok:
            return None
        if tok in self._channels:
            return tok
        ch = self._resolve_channel(tok)
        if ch in self._channels:
            return ch
        low = tok.lower()
        for det, lbl in self._channel_labels.items():
            if lbl and lbl.lower() == low and det in self._channels:
                return det
        for det, lbl in self._channel_labels.items():
            if lbl and low in lbl.lower() and det in self._channels:
                return det
        return None

    def _parse_pairs_str(self, text):
        """Parse ``"CD34/CD11b, CD11b/CD45"`` into resolved ``(x, y)`` channel
        tuples. Unresolvable tokens are skipped."""
        pairs = []
        for chunk in re.split(r'[,;\n]', text or ''):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = re.split(r'\s*[/xX×]\s*|\s+vs\.?\s+', chunk, maxsplit=1)
            if len(parts) != 2:
                continue
            xc = self._resolve_token_to_channel(parts[0])
            yc = self._resolve_token_to_channel(parts[1])
            if xc and yc:
                pairs.append((xc, yc))
        return pairs

    def _open_figure_layout(self):
        """Open the figure-layout dialog (small-multiple publication figure
        builder) seeded from the current plot selection."""
        samples = self._selected_samples()
        if not samples:
            messagebox.showinfo(
                "Figure layout",
                "Enable one or more samples (☑ in the tree) first.",
                parent=self)
            return
        cur_x = self.x_combo.get()
        cur_y = self.y_combo.get()
        xc = self._resolve_channel(cur_x)
        yc = self._resolve_channel(cur_y)
        default_pairs = ''
        if xc and yc:
            default_pairs = f"{self._fmt_channel(xc)} / {self._fmt_channel(yc)}"
        FigureLayoutDialog(self, len(samples), self.mode_var.get(),
                           default_pairs, self._build_and_preview_figure)

    def _build_and_preview_figure(self, opts):
        """Build the multi-panel figure from the dialog's options and show it
        in a preview window with a Save control."""
        samples = self._selected_samples()
        if not samples:
            return
        mode = self.mode_var.get()
        color = self.color_combo.get()
        cur_x = self._resolve_channel(self.x_combo.get())
        cur_y = self._resolve_channel(self.y_combo.get())
        layout = opts.get('layout', 'per_sample')
        pairs = self._parse_pairs_str(opts.get('pairs', ''))
        ncols = opts.get('ncols', 3)
        draw_gates = opts.get('gates', True)

        panels = []
        if layout == 'single':
            panels.append(dict(samples=samples, x=cur_x, y=cur_y,
                               mode=mode, color=color, title=None))
        elif layout == 'per_sample':
            for nm in samples:
                panels.append(dict(samples=[nm], x=cur_x, y=cur_y,
                                   mode=mode, color=color,
                                   title=self._short_sample(nm)))
        elif layout == 'per_pair':
            if not pairs and cur_x and cur_y:
                pairs = [(cur_x, cur_y)]
            for (px, py) in pairs:
                ttl = f"{self._fmt_channel(px)} / {self._fmt_channel(py)}"
                panels.append(dict(samples=samples, x=px, y=py, mode=mode,
                                   color=color, title=ttl))
        elif layout == 'grid':
            if not pairs and cur_x and cur_y:
                pairs = [(cur_x, cur_y)]
            ncols = max(1, len(pairs))
            for nm in samples:
                for (px, py) in pairs:
                    ttl = (f"{self._short_sample(nm, 16)} · "
                           f"{self._fmt_channel(px)}/{self._fmt_channel(py)}")
                    panels.append(dict(samples=[nm], x=px, y=py, mode=mode,
                                       color=color, title=ttl))

        if not panels:
            messagebox.showwarning(
                "Figure layout",
                "Nothing to plot — check the layout and channel pairs.",
                parent=self)
            return

        fig = self._build_layout_figure(panels, ncols, draw_gates=draw_gates)
        if fig is None:
            return
        self._show_figure_preview(fig)

    def _show_figure_preview(self, fig):
        """Pop a Toplevel embedding ``fig`` with Save (PNG/PDF/SVG) / Close."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        win = tk.Toplevel(self)
        win.title("Figure preview")
        win.geometry("1000x720")

        bar = ttk.Frame(win)
        bar.pack(fill='x', side='top')

        # Export background: White (opaque, default), Transparent (fully — for
        # placing on a coloured page / poster), or Translucent (50% white).
        # PNG / PDF / SVG carry alpha; TIFF may flatten it depending on viewer.
        bg_var = tk.StringVar(value='White')

        def _save():
            path = filedialog.asksaveasfilename(
                parent=win, title="Save figure",
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'),
                           ('PDF document', '*.pdf'),
                           ('SVG vector', '*.svg'),
                           ('TIFF image', '*.tif *.tiff')])
            if not path:
                return
            bg = bg_var.get()
            try:
                savefig_background(fig, path, background=bg, dpi=300)
            except Exception as exc:
                messagebox.showerror(
                    "Figure layout", f"Could not save figure:\n{exc}",
                    parent=win)
                return
            self._audit('figure.export', path=path,
                        n_panels=len(fig.axes), background=bg)
            messagebox.showinfo("Figure layout", f"Saved:\n{path}",
                                parent=win)

        ttk.Button(bar, text="Save…", command=_save).pack(
            side='left', padx=4, pady=4)
        ttk.Button(bar, text="Close", command=win.destroy).pack(
            side='left', padx=(0, 4), pady=4)
        ttk.Label(bar, text="Background:").pack(side='left', padx=(12, 2))
        ttk.Combobox(bar, textvariable=bg_var, width=12, state='readonly',
                     values=['White', 'Transparent', 'Translucent']).pack(
            side='left', pady=4)

        cf = ttk.Frame(win)
        cf.pack(fill='both', expand=True)
        canvas = FigureCanvasTkAgg(fig, master=cf)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        # A pan/zoom toolbar would be nice, but NavigationToolbar2Tk isn't in
        # matplotlib's type stubs (pyright flags the import); the Save control
        # plus matplotlib's own keymap is enough for a preview.
        canvas.draw()

    def _redraw_only_gates(self):
        """Cheap refresh of the gate overlays without redoing the density
        plot. Used while dragging sliders."""
        # Remove existing artists.
        for line in list(self._vlines.values()) + list(self._hlines.values()):
            try:
                line.remove()
            except Exception:
                pass
        for patch in list(self._shape_artists.values()):
            try:
                patch.remove()
            except Exception:
                pass
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        self._draw_gates(x, y if self.mode_var.get() != 'histogram' else None)
        self.canvas.draw_idle()

    # ── Templates (save / load) ──────────────────────────────────────────

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
            menu.add_command(label="Apply a template:", state='disabled')
            for name, _desc, path in bundled:
                menu.add_command(
                    label=f"  {name}",
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

    def _apply_template_path(self, path):
        """Read a template/.wsp at ``path`` and apply it to chosen samples.
        Shared by the file-picker and the built-in library menu.

        Native JSON schema: {"gates": [gate_dict, …], "labels": {ch: lbl, …}}.
        """
        # Read the file first (separate from applying it, so a parse
        # failure reports cleanly before we pop the target dialog).
        try:
            sys.path.insert(0, BASE)
            from .pipeline import read_template_gates
            gate_dicts, labels = read_template_gates(path)
        except Exception as exc:
            self.status_var.set(f"Failed to read template: {exc}")
            messagebox.showerror(
                "Load template failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}", parent=self)
            return

        kind_label = 'FlowJo .wsp' if path.lower().endswith('.wsp') \
                     else 'JSON template'
        source = f'{kind_label} ({os.path.basename(path)})'

        if self._active_sample is None or not self._samples:
            self.status_var.set("Load a sample first, then load a template.")
            return

        if labels:
            self._channel_labels.update(
                {k: str(v) for k, v in labels.items()})
            if self._channels:
                self._populate_channel_combos()

        # Ask which loaded samples to apply to + whether to overwrite or
        # add to each target's existing gates.
        choice = self._ask_template_apply()
        if choice is None:
            return                              # cancelled
        targets, overwrite = choice
        if not targets:
            self.status_var.set("No target samples selected.")
            return

        saved_active = self._active_sample
        mismatches = {}
        for name in targets:
            self._apply_template_to_sample(name, gate_dicts, overwrite)
            miss = self._count_channel_mismatches(name, gate_dicts)
            if miss:
                mismatches[name] = miss
        # Restore whatever sample was active before the batch apply.
        if saved_active in self._samples:
            self._set_active_sample(saved_active)
        self._refresh_gate_list()
        self._schedule_replot(0)

        kinds = ', '.join(sorted({g.get('kind', '?') for g in gate_dicts})) \
                or 'none'
        verb = 'overwrote' if overwrite else 'added to'
        self.status_var.set(
            f"Applied {len(gate_dicts)} gate(s) [{kinds}] from {source}: "
            f"{verb} {len(targets)} sample(s).")

        # Channel-mismatch warning — gates referencing channels a target
        # sample doesn't have will sit inert (gate_to_mask no-ops them).
        if mismatches:
            lines = '\n'.join(f"  • {n}: {c} gate(s)"
                              for n, c in mismatches.items())
            messagebox.showwarning(
                "Template channels missing in some samples",
                "Some gates reference channels that aren't present in "
                "these samples — they'll be inactive (no-op) there until "
                "the channels exist:\n\n" + lines, parent=self)

    @staticmethod
    def _gate_channels(g):
        """Set of FCS channel names a gate dict references."""
        chs = set()
        for k in ('channel', 'x_channel', 'y_channel'):
            v = g.get(k)
            if v:
                chs.add(v)
        return chs

    def _count_channel_mismatches(self, name, gate_dicts):
        """How many of `gate_dicts` reference a channel absent from
        sample `name`'s data."""
        s = self._samples.get(name)
        if s is None:
            return 0
        try:
            cols = set(s.data.columns)
        except Exception:
            return 0
        return sum(1 for g in gate_dicts
                   if self._gate_channels(g) - cols)

    def _apply_template_to_sample(self, name, gate_dicts, overwrite):
        """Install a template's gates into sample `name`.

        Uses the set-active → install → (caller restores) pattern, the
        same as the .wsp-ingest path. Each sample gets independent gate
        ids, so the source ids are rewired per sample.

        overwrite=True  → replace that sample's gate tree
        overwrite=False → append (keeps existing gates; threshold/
                          interval gates with a matching (channel, parent)
                          are replaced in place by _add_gate, as usual).
        """
        if name not in self._samples:
            return
        self._checkpoint()      # applying a template is one undoable step
        self._set_active_sample(name)
        if overwrite:
            self._gates.clear()
            del self._gate_id_order[:]
            self._gate_id_seq = 0
            self._sample_gate_seq[name] = 0
        # Label-first retargeting: a template gate stamped with an
        # antibody label retargets to THIS sample's detector for that
        # label, so a CD11b gate applies wherever CD11b sits in each
        # sample (different fluors across panels). Built from the
        # target sample's own detector↔label map.
        from .pipeline import _sample_fluor_labels, relabel_gate_for_sample
        label_to_det = _sample_fluor_labels(self._samples[name])
        # Rewrite each source parent_id (.wsp `_import_id` or template
        # `id`) to the fresh editor id _add_gate returns, in parent-first
        # order.
        old_to_new = {}
        for raw in gate_dicts:
            g = relabel_gate_for_sample(raw, label_to_det)
            src_id = g.pop('_import_id', None) or g.pop('id', None)
            parent = g.get('parent_id')
            if parent is not None:
                g['parent_id'] = old_to_new.get(parent)
            gid = self._add_gate(g)
            if src_id is not None:
                old_to_new[src_id] = gid

    def _ask_template_apply(self):
        """Modal dialog: choose target samples (multiselect) + apply mode
        (overwrite vs add-to). Returns (targets:list[str], overwrite:bool)
        or None if cancelled. Blocks until dismissed."""
        loaded = [n for n in self._sample_order if n in self._samples]
        result: dict[str, tuple[list[str], bool] | None] = {'value': None}

        dlg = tk.Toplevel(self)
        dlg.title("Apply template to…")
        dlg.transient(self)
        dlg.grab_set()
        dlg.geometry("420x460")
        dlg.minsize(360, 320)

        ttk.Label(dlg, text="Apply the template to these samples:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 6))

        # Scrollable checkbox list.
        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=(0, 6))
        cv = tk.Canvas(holder, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))

        cb_vars = {}
        for name in loaded:
            # Default: every loaded sample checked (batch intent).
            var = tk.BooleanVar(value=True)
            cb_vars[name] = var
            label = name + ('  (active)' if name == self._active_sample else '')
            ttk.Checkbutton(inner, text=label, variable=var).pack(
                side='top', anchor='w', padx=2, pady=1)

        # Mode radio.
        mode_frame = ttk.Frame(dlg)
        mode_frame.pack(side='top', fill='x', padx=10, pady=(0, 6))
        mode_var = tk.StringVar(value='overwrite')
        ttk.Label(mode_frame, text="Mode:").pack(side='left')
        ttk.Radiobutton(mode_frame, text="Overwrite gates",
                        value='overwrite', variable=mode_var).pack(
            side='left', padx=(6, 0))
        ttk.Radiobutton(mode_frame, text="Add to existing",
                        value='append', variable=mode_var).pack(
            side='left', padx=(6, 0))

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)
        ttk.Button(btns, text="Select all",
                   command=lambda: [v.set(True) for v in cb_vars.values()]
                   ).pack(side='left')
        ttk.Button(btns, text="Deselect all",
                   command=lambda: [v.set(False) for v in cb_vars.values()]
                   ).pack(side='left', padx=(4, 0))

        def do_apply():
            result['value'] = (
                [n for n, v in cb_vars.items() if v.get()],
                mode_var.get() == 'overwrite',
            )
            dlg.destroy()

        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Apply",
                   command=do_apply).pack(side='right', padx=(0, 4))
        dlg.bind('<Escape>', lambda *_: dlg.destroy())

        self.wait_window(dlg)
        return result['value']

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

    # ── Session save / load (full editor state) ──────────────────────────
    #
    # A session captures EVERYTHING the editor holds, unlike the two
    # narrower formats:
    #   .wsp       — FlowJo-representable gates + compensation only
    #   .json tpl  — gate trees only (no samples, no display state)
    #   .flowsession (here) — samples (by path) + per-sample gates (full
    #               fidelity incl. ellipsoid/quadrant/colour/enabled) +
    #               per-channel scale/range + plot mode + channel labels +
    #               downsample toggles + a reserved cluster-labels slot.

    SESSION_EXT = '.flowsession'

    def _session_autosave_path(self):
        """Well-known path for the auto-saved 'last session', under the
        user's home so it's found regardless of CWD."""
        d = os.path.join(os.path.expanduser('~'), '.openflo')
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, 'last_session' + self.SESSION_EXT)

    def _session_state(self):
        """Serialise the full editor state to a JSON-able dict."""
        from datetime import datetime
        samples = []
        for name in self._sample_order:
            s = self._samples.get(name)
            if s is None:
                continue
            entry = {
                'name': name,
                'path': getattr(s, 'path', '') or '',
                'color': self._sample_colors.get(name, '#1f77b4'),
                'plot_enabled': bool(self._sample_plot_enabled.get(name, False)),
                'trial': self._sample_trial.get(name, 'Trial'),
            }
            # Persist a manual Comps/Samples override only when set (so
            # name-based detection still applies to untouched samples on load).
            if name in self._sample_is_comp:
                entry['is_comp'] = bool(self._sample_is_comp[name])
            samples.append(entry)
        # Per-sample gates as ordered lists carrying their editor id +
        # parent_id (so the hierarchy restores).
        sample_gates = {}
        for name, gates in self._sample_gates.items():
            order = self._sample_gate_order.get(name, list(gates))
            out = []
            for gid in order:
                g = gates.get(gid)
                if g is None:
                    continue
                gd = dict(g)
                gd['id'] = gid
                out.append(gd)
            sample_gates[name] = out
        # _channel_range values are tuples → JSON lists. Skip any None
        # (auto-range) entries — the type allows None even though we
        # pop rather than store it.
        ranges = {ch: [float(rng[0]), float(rng[1])]
                  for ch, rng in self._channel_range.items()
                  if rng is not None}
        return {
            'format': 'openflo-session',
            'version': 1,
            'created': datetime.now().isoformat(timespec='seconds'),
            'active_sample': self._active_sample,
            'samples': samples,
            'sample_gates': sample_gates,
            'channel_scale': dict(self._channel_scale),
            'channel_range': ranges,
            'channel_labels': dict(self._channel_labels),
            'plot_mode': self.mode_var.get(),
            'x_channel': self.x_combo.get(),
            'y_channel': self.y_combo.get(),
            'color_channel': self.color_combo.get(),
            'downsample_display': bool(self.ds_display_var.get()),
            'downsample_propagate': bool(self.ds_propagate_var.get()),
            'max_points': self.max_points_var.get(),
            'show_removed': bool(self.show_removed_var.get()),
            'contour_scatter': bool(self.contour_scatter_var.get()),
            'contour_outliers': bool(self.contour_outliers_var.get()),
            'hist_y_mode': self.hist_y_mode.get(),
            'cluster_labels': dict(self._cluster_labels),   # reserved slot
            'audit': self._audit_log.to_list(),
        }

    def _write_session(self, path):
        """Core writer — shared by Save Session… and autosave."""
        data = self._session_state()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return data

    def _save_session(self):
        if not self._samples:
            self.status_var.set("Load at least one sample before saving a session.")
            return
        path = filedialog.asksaveasfilename(
            title="Save editor session",
            defaultextension=self.SESSION_EXT,
            initialfile='session' + self.SESSION_EXT,
            filetypes=[('OpenFlo session', '*' + self.SESSION_EXT),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            data = self._write_session(path)
            self.status_var.set(
                f"Saved session: {len(data['samples'])} sample(s) → "
                f"{os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Save session failed: {exc}")
            messagebox.showerror(
                "Save session failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}",
                parent=self)

    def _load_session(self):
        path = filedialog.askopenfilename(
            title="Load editor session",
            filetypes=[('OpenFlo session', '*' + self.SESSION_EXT),
                       ('All files', '*.*')])
        if not path:
            return
        self._load_session_path(path)

    def _load_session_path(self, path):
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            self.status_var.set(f"Load session failed: {exc}")
            messagebox.showerror(
                "Load session failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}",
                parent=self)
            return
        if data.get('format') != 'openflo-session':
            messagebox.showerror(
                "Not a session file",
                f"{os.path.basename(path)} isn't an OpenFlo session "
                "(missing format marker).", parent=self)
            return
        # A fresh session starts a fresh history — undo shouldn't cross
        # back into the previous session's gates.
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._apply_session(data)

    def _apply_session(self, data):
        """Restore editor state from a parsed session dict.

        Display state is restored immediately. Samples are loaded
        asynchronously (same threaded path as Add-FCS); their gates are
        staged in `_pending_sample_gates` and applied by `_on_loaded`
        as each FCS finishes parsing — reusing the WSP-ingest mechanism.
        """
        # Restore global display config up front (independent of samples).
        self._channel_scale.update(
            {k: str(v) for k, v in (data.get('channel_scale') or {}).items()})
        self._channel_range.update(
            {k: (float(v[0]), float(v[1]))
             for k, v in (data.get('channel_range') or {}).items()
             if isinstance(v, (list, tuple)) and len(v) == 2})
        self._channel_labels.update(
            {k: str(v) for k, v in (data.get('channel_labels') or {}).items()})
        # Restore the provenance trail, then log the load itself so the
        # reopened session records that it was reopened (and from where).
        from .audit import AuditLog
        self._audit_log = AuditLog.from_list(data.get('audit') or [])
        self._audit('session.load',
                    created=data.get('created', ''),
                    n_samples=len(data.get('samples', [])))
        win = getattr(self, '_audit_window', None)
        if win is not None and win.winfo_exists():
            win.refresh()
        # cluster_labels round-trips through JSON, which stringifies the
        # inner int cluster-id keys. Coerce them back to int so lookups by
        # the numeric id (from the data column) hit.
        for sname, lbls in (data.get('cluster_labels') or {}).items():
            if not isinstance(lbls, dict):
                continue
            coerced = {}
            for cid, nm in lbls.items():
                try:
                    coerced[int(cid)] = nm
                except (TypeError, ValueError):
                    coerced[cid] = nm
            self._cluster_labels[sname] = coerced
        try:
            self.ds_display_var.set(bool(data.get('downsample_display', True)))
            self.ds_propagate_var.set(bool(data.get('downsample_propagate', False)))
            if data.get('max_points'):
                self.max_points_var.set(str(data['max_points']))
            self.show_removed_var.set(bool(data.get('show_removed', False)))
            self.contour_scatter_var.set(
                bool(data.get('contour_scatter', True)))
            self.contour_outliers_var.set(
                bool(data.get('contour_outliers', True)))
            if data.get('hist_y_mode') in ('Fraction', 'Count', '% of Max'):
                self.hist_y_mode.set(data['hist_y_mode'])
            if data.get('plot_mode') in self.PLOT_MODES:
                self.mode_var.set(data['plot_mode'])
            self._sync_hist_y_combo()
        except Exception:
            pass

        # Stage each sample's restore bundle — grouping (trial + Comps/Samples
        # override) AND its gates — keyed by FILE PATH so it survives name
        # disambiguation across reloads. `_on_loaded` drains it by the loaded
        # sample's path. Cleared first so a prior session's missing-file entries
        # can't leak onto a later load.
        self._pending_sample_meta.clear()
        sample_gates = data.get('sample_gates') or {}
        for s in data.get('samples', []):
            nm = s.get('name')
            path = s.get('path') or ''
            if not nm or not path:
                continue
            pkey = os.path.normcase(os.path.abspath(path))
            m: dict[str, object] = {'gates': list(sample_gates.get(nm, []))}
            if s.get('trial'):
                m['trial'] = s['trial']
            if 'is_comp' in s:
                m['is_comp'] = bool(s['is_comp'])
            self._pending_sample_meta[pkey] = m

        # Remember the combo selections + active sample to restore once
        # at least one sample has loaded (combos populate from sample 1).
        self._session_restore = {
            'x': data.get('x_channel'),
            'y': data.get('y_channel'),
            'color': data.get('color_channel'),
            'active': data.get('active_sample'),
            'plot_enabled': {s['name']: s.get('plot_enabled', False)
                             for s in data.get('samples', [])},
        }

        # Queue the FCS loads. Missing files are reported but don't abort.
        paths, missing = [], []
        for s in data.get('samples', []):
            p = s.get('path') or ''
            if p and os.path.isfile(p):
                paths.append(p)
            else:
                missing.append(s.get('name') or os.path.basename(p) or '?')
        if paths:
            self._queue_fcs_loads(paths)
        msg = f"Loading session: {len(paths)} sample(s)"
        if missing:
            msg += f" — missing FCS for: {', '.join(missing[:4])}"
            if len(missing) > 4:
                msg += f" (+{len(missing) - 4})"
        self.status_var.set(msg)
        # Apply the deferred combo/active restore after the load queue
        # has had a chance to populate channels.
        self.after(600, self._apply_session_restore)

    def _apply_session_restore(self):
        """Second half of session restore: combo selections, plot-enabled
        toggles, active sample. Deferred so the first sample's channels
        have populated the combos."""
        info = getattr(self, '_session_restore', None)
        if not info:
            return
        for name, on in info.get('plot_enabled', {}).items():
            if name in self._samples:
                self._sample_plot_enabled[name] = bool(on)
        for combo, key in ((self.x_combo, 'x'), (self.y_combo, 'y'),
                           (self.color_combo, 'color')):
            val = info.get(key)
            if val and val in combo['values']:
                combo.set(val)
        active = info.get('active')
        if active and active in self._samples:
            self._set_active_sample(active)
        self._session_restore = None
        self._refresh_gate_list()
        self._schedule_replot(0)

    # ── In-app log pane + Python console ──────────────────────────────────
    def _toggle_log(self):
        """Show/hide the log + console at the bottom of the left column."""
        if self._show_log_var.get():
            self._log_frame.grid()
        else:
            self._log_frame.grid_remove()

    def _make_console(self):
        """Build the persistent interpreter, pre-binding handy live objects."""
        import code
        ns = {'__name__': '__console__', 'editor': self, 'self': self,
              'samples': self._samples, 'np': np}
        try:
            import pandas as _pd
            ns['pd'] = _pd
        except Exception:
            pass
        return code.InteractiveConsole(locals=ns)

    def _console_run(self, event=None):
        """Run the entered line through the interpreter. Output / the repr of
        expressions / tracebacks all surface in the log via the stdout/stderr
        tee. A continuation (e.g. an open `def`) flips the prompt to `...`."""
        line = self._console_entry.get()
        self._console_entry.delete(0, 'end')
        if line.strip():
            self._console_history.append(line)
        self._console_hist_idx = len(self._console_history)
        self._append_log(f"{self._console_prompt.get()} {line}\n")
        if self._console is None:
            self._console = self._make_console()
        try:
            more = self._console.push(line)
        except SystemExit:
            more = False
        except BaseException:           # noqa: BLE001 — console must never crash the GUI
            more = False
        self._console_prompt.set('...' if more else '>>>')
        self._drain_log()               # flush output/repr immediately
        return 'break'

    def _console_history_prev(self, event=None):
        if not self._console_history:
            return 'break'
        self._console_hist_idx = max(0, self._console_hist_idx - 1)
        self._console_entry.delete(0, 'end')
        self._console_entry.insert(0, self._console_history[self._console_hist_idx])
        return 'break'

    def _console_history_next(self, event=None):
        if not self._console_history:
            return 'break'
        self._console_hist_idx = min(len(self._console_history),
                                     self._console_hist_idx + 1)
        self._console_entry.delete(0, 'end')
        if self._console_hist_idx < len(self._console_history):
            self._console_entry.insert(
                0, self._console_history[self._console_hist_idx])
        return 'break'

    def _clear_log(self):
        t = getattr(self, '_log_text', None)
        if t is None:
            return
        t.config(state='normal')
        t.delete('1.0', 'end')
        t.config(state='disabled')

    def _drain_log(self):
        """Append any queued stdout/stderr lines to the pane (main thread).
        Reschedules itself; cheap when the queue is empty."""
        try:
            chunks = []
            while True:
                try:
                    chunks.append(self._log_queue.get_nowait())
                except queue.Empty:
                    break
            if chunks:
                self._append_log(''.join(chunks))
        except Exception:
            pass
        finally:
            try:
                self.after(300, self._drain_log)
            except Exception:
                pass

    def _append_log(self, text):
        t = getattr(self, '_log_text', None)
        if t is None or not text:
            return
        try:
            t.config(state='normal')
            t.insert('end', text)
            # Cap the buffer so a long session doesn't grow unbounded.
            last = int(t.index('end-1c').split('.')[0])
            if last > 500:
                t.delete('1.0', f'{last - 500}.0')
            t.see('end')
            t.config(state='disabled')
        except Exception:
            pass

    def _on_close(self):
        """Autosave the current session (if there's anything worth
        saving) then close. When this editor is the app's primary window,
        closing it tears the whole app down (kills any running pipeline
        subprocess via App.shutdown, which destroys the root + this
        editor); otherwise it just closes this Toplevel."""
        # Wake any blocked load workers so they exit instead of holding the
        # queue forever (daemon=True is the backstop). Best-effort, non-blocking
        # — we don't join, so a slow in-flight FlowSample can't hang the close.
        try:
            self._load_stop.set()
            for _ in range(_LOAD_POOL_SIZE):
                self._load_queue.put_nowait(None)
        except Exception:
            pass
        # Stop mirroring stdout/stderr into this (closing) editor's pane.
        for tee in getattr(self, '_log_tees', []):
            try:
                tee.remove_sink(self._log_queue)
            except Exception:
                pass
        try:
            if self._samples:
                self._write_session(self._session_autosave_path())
        except Exception as exc:
            print(f"[session] autosave failed: {exc}", flush=True)
        if self._primary and self._app is not None:
            try:
                self._app.destroy()      # destroy the Tk root → exits mainloop
                return
            except Exception:
                pass
        self.destroy()

    def _open_pipeline_workspace(self):
        """Toggle the docked Pipeline Workspace pane. Showing it splits the
        plot area via the sash; hiding it gives the plot the full width."""
        panel = getattr(self, '_workspace_panel', None)
        if panel is None:
            return
        if getattr(self, '_workspace_shown', False):
            try:
                self._editor_paned.forget(panel)
            except Exception:
                pass
            self._workspace_shown = False
            self.status_var.set("Pipeline workspace hidden.")
            return
        try:
            self._editor_paned.add(panel, weight=3)
            self._workspace_shown = True
            self.update_idletasks()
            try:
                total = self._editor_paned.winfo_width()
                if total > 100:
                    self._editor_paned.sashpos(0, int(total * 0.62))
            except Exception:
                pass
            self.status_var.set(
                "Pipeline workspace shown. Drag samples / gate leaves in; each tab is a separate query.")
        except Exception as exc:
            self.status_var.set(f"Couldn't show pipeline workspace: {exc}")

    # ── Population statistics (FlowJo-style table) ───────────────────────
    #
    # Per sample × population: Count, %Parent, %Total, and per-channel
    # Median / Mean / CV. "Population" = a gate node; its events are the
    # cumulative AND of the gate chain from root to that node (same
    # cumulative_gate_mask the highlight/filter paths use). Modular —
    # the StatisticsWindow lets the user toggle which columns appear.

    # Available statistic columns. Population-level ones are scalar;
    # per-channel ones expand to one column per fluor channel.
    STAT_POP   = ('Count', '%Parent', '%Total')
    STAT_CHAN  = ('Median', 'Mean', 'CV')

    @staticmethod
    def _population_path(gates, gid):
        """Human-readable population path, e.g. 'Cells/Singlets/CD11b+',
        built by walking parent_id to the root. Cycle-safe."""
        from .pipeline import describe_gate
        names, seen, cur = [], set(), gid
        while cur and cur in gates and cur not in seen:
            seen.add(cur)
            g = gates[cur]
            names.append(g.get('label') or g.get('name')
                         or describe_gate(g) or cur)
            cur = g.get('parent_id')
        return '/'.join(reversed(names)) if names else str(gid)

    @classmethod
    def _population_stats(cls, sample_name, df, gates, order,
                          channel_labels, channels, want, select=None):
        """Compute statistic rows for ONE sample's populations.

        Pure (no Tk) so it's unit-testable.

        df             : the sample's DataFrame (full, not downsampled)
        gates          : {gid: gate_dict}
        order          : [gid] insertion order (falls back to dict order)
        channel_labels : {detector: antibody label} for column naming
        channels       : list of channels to compute per-channel stats on
        want           : set of selected stat names (subset of
                         STAT_POP + STAT_CHAN)
        select         : optional [gid] — emit rows ONLY for these gates (in
                         this order). Counts are still computed over the full
                         tree so %Parent stays correct. None = emit every gate.

        Each row carries a hidden ``__gid__`` (the source gate id) for callers
        that key on it; ``_collect_stats_rows`` keeps ``__``-prefixed keys out
        of the displayed column set.

        Returns a list of ordered row dicts. Empty populations yield NaN
        for per-channel stats and 0 for counts.
        """
        from .pipeline import cumulative_gate_mask
        total = len(df)
        order = order or list(gates)
        # Cumulative mask + count per gate (parent counts feed %Parent) — over
        # the FULL tree regardless of `select`, so parent counts are present.
        counts, masks = {}, {}
        for gid in order:
            if gid not in gates:
                continue
            m = cumulative_gate_mask(gates, gid, df)
            masks[gid] = m
            counts[gid] = int(np.asarray(m).sum())

        emit = select if select is not None else order
        rows = []
        for gid in emit:
            if gid not in gates or gid not in counts:
                continue
            g = gates[gid]
            cnt = counts[gid]
            parent = g.get('parent_id')
            parent_cnt = counts.get(parent, total) if parent else total
            row = {
                'Sample': sample_name,
                'Population': cls._population_path(gates, gid),
                '__gid__': gid,
            }
            if 'Count' in want:
                row['Count'] = cnt
            if '%Parent' in want:
                row['%Parent'] = (cnt / parent_cnt * 100.0) if parent_cnt else 0.0
            if '%Total' in want:
                row['%Total'] = (cnt / total * 100.0) if total else 0.0

            need_chan = want & set(cls.STAT_CHAN)
            if need_chan and channels:
                sub = df[masks[gid]] if cnt else None
                for ch in channels:
                    lbl = channel_labels.get(ch, ch)
                    if ch not in df.columns:
                        continue
                    if sub is None or len(sub) == 0:
                        med = mean = cv = float('nan')
                    else:
                        vals = np.asarray(sub[ch].values, dtype=float)
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            med = mean = cv = float('nan')
                        else:
                            med = float(np.median(vals))
                            mean = float(np.mean(vals))
                            sd = float(np.std(vals))
                            cv = (sd / mean * 100.0) if mean else float('nan')
                    if 'Median' in want:
                        row[f'Median {lbl}'] = med
                    if 'Mean' in want:
                        row[f'Mean {lbl}'] = mean
                    if 'CV' in want:
                        row[f'CV {lbl}'] = cv
            rows.append(row)
        return rows

    def _sample_rows(self, name, want, select=None):
        """Population rows for ONE loaded sample. `select` (a list of gids)
        restricts the emitted populations; None emits all of the sample's
        gates. Returns [] if the sample isn't loaded."""
        s = self._samples.get(name)
        if s is None:
            return []
        gates = self._sample_gates.get(name, {})
        order = self._sample_gate_order.get(name, list(gates))
        channels = [c for c in getattr(s, 'fluor_channels', [])
                    if c in s.data.columns]
        # Use THIS sample's own antibody labels (so a marker on a different
        # fluor still names its column by label and ties across samples); the
        # editor's global labels are a fallback.
        labels = dict(self._channel_labels)
        labels.update(getattr(s, 'channel_labels', {}) or {})
        sel = None if select is None else [g for g in select if g in gates]
        return self._population_stats(
            name, s.data, gates, order, labels, channels, want, select=sel)

    def _collect_stats_rows(self, want, samples=None, gate_targets=None):
        """Aggregate population rows. Three modes:
          • gate_targets : list of (sample, gid) → emit exactly those
            populations (grouped by sample, first-seen order). This is the
            curated, gate-only mode used by the stats window.
          • samples       : restrict to these sample names (all their pops).
          • neither       : every population of every loaded sample.
        `want` is the selected stat-name set. Returns (rows, columns); the
        column set excludes internal ``__``-prefixed keys (e.g. __gid__)."""
        all_rows = []
        if gate_targets is not None:
            by_sample = {}
            for nm, gid in gate_targets:
                by_sample.setdefault(nm, [])
                if gid not in by_sample[nm]:
                    by_sample[nm].append(gid)
            for name, gids in by_sample.items():
                all_rows.extend(self._sample_rows(name, want, select=gids))
        else:
            names = samples if samples is not None else [
                n for n in self._sample_order if n in self._samples]
            for name in names:
                all_rows.extend(self._sample_rows(name, want))
        # Stable column order: identity cols first, then pop-level, then
        # per-channel in first-seen order. Internal __keys never display.
        cols = ['Sample', 'Population']
        for stat in self.STAT_POP:
            if stat in want:
                cols.append(stat)
        seen = set(cols)
        for r in all_rows:
            for k in r:
                if k not in seen and not k.startswith('__'):
                    seen.add(k)
                    cols.append(k)
        return all_rows, cols

    def _loaded_samples(self):
        """FlowSample objects for every loaded sample, in load order."""
        return [self._samples[n] for n in self._sample_order
                if n in self._samples]

    def _fluor_panel_warning(self):
        """'' when all loaded samples share a fluor panel (by antibody
        label), else a message listing the non-common labels. Cross-
        sample stats/comparison tie by label, so a sample missing a
        marker just won't contribute to that label's column."""
        samples = self._loaded_samples()
        if len(samples) < 2:
            return ''
        from .pipeline import common_fluor_warning
        return common_fluor_warning(samples)

    def _open_stats_window(self):
        if not self._samples:
            self.status_var.set("Load a sample first to compute statistics.")
            return
        StatisticsWindow(self)

    def _open_frequency_window(self):
        if not self._samples:
            self.status_var.set("Load samples first to compare frequencies.")
            return
        FrequencyComparisonWindow(self)

    def _open_trajectory_window(self):
        if not self._samples:
            self.status_var.set("Load samples first to infer a trajectory.")
            return
        TrajectoryWindow(self)

    def _open_flowsom_tree(self):
        name = self._active_sample
        s = self._samples.get(name) if name else None
        if s is None or not getattr(s, 'flowsom_result', None):
            self.status_var.set(
                "Run FlowSOM first (Cluster… → FlowSOM), then SOM tree.")
            return
        FlowSOMTreeWindow(self, name)

    def _open_annotation_window(self):
        name = self._active_sample
        s = self._samples.get(name) if name else None
        if s is None:
            self.status_var.set("Select a clustered sample to annotate.")
            return
        if not any(c in s.data.columns
                   for c in ('leiden', 'cluster', 'flowsom_meta')):
            self.status_var.set(
                "Cluster the sample first (Cluster… → Phenograph/FlowSOM/"
                "Leiden), then Annotate.")
            return
        PopulationAnnotationWindow(self, name)

    def _apply_population_names(self, sample, label_col, names):
        """Write annotation names onto a sample's populations: into
        ``_cluster_labels`` (for the cluster path) and onto any existing
        cluster/category gate for that label value, then refresh the tree."""
        store = self._cluster_labels.setdefault(sample, {})
        for cid, nm in names.items():
            store[cid] = nm
        gates = self._sample_gates.get(sample, {})
        for g in gates.values():
            if g.get('kind') == 'cluster' and g.get('cluster_id') in names:
                g['name'] = names[g['cluster_id']]
            elif (g.get('kind') == 'category'
                  and g.get('channel') == label_col
                  and g.get('value') in names):
                g['name'] = names[g['value']]
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _report_heatmap_html(self):
        """A cluster × marker median-expression heatmap for the active (or
        first) sample carrying a label column. Returns an ``<img>`` or None."""
        name = self._active_sample or (self._sample_order[0]
                                       if self._sample_order else None)
        s = self._samples.get(name) if name else None
        if s is None:
            return None
        col = next((c for c in ('leiden', 'cluster', 'flowsom_meta')
                    if c in s.data.columns), None)
        if col is None:
            return None
        chans = [c for c in getattr(s, 'fluor_channels', [])
                 if c in s.data.columns]
        if not chans:
            return None
        df = s.data[s.data[col] >= 0]
        if df.empty:
            return None
        med = df.groupby(col)[chans].median()
        if med.empty:
            return None
        from matplotlib.figure import Figure

        from .report import figure_html
        fig = Figure(figsize=(min(1 + 0.5 * len(chans), 10),
                              min(1 + 0.3 * len(med), 9)), dpi=120)
        ax = fig.add_subplot(111)
        # Column-z-score so markers on different scales are comparable.
        arr = med.to_numpy(dtype=float)
        mu, sd = arr.mean(0), arr.std(0)
        sd[sd == 0] = 1.0
        im = ax.imshow((arr - mu) / sd, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(chans)))
        ax.set_xticklabels([self._fmt_channel(c) for c in chans],
                           rotation=90, fontsize=7)
        ax.set_yticks(range(len(med)))
        ax.set_yticklabels([str(i) for i in med.index], fontsize=7)
        ax.set_xlabel('marker'); ax.set_ylabel(col)
        ax.set_title(f"{name}: median expression per {col} (column z-score)",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return figure_html(fig, alt='cluster heatmap')

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
        self.status_var.set("Building report…")
        self.update_idletasks()

        meta = {
            'OpenFlo version': __version__,
            'Generated': datetime.now().isoformat(timespec='seconds'),
            'Samples': len(self._samples),
            'Channels': len(self._channels),
            'Active sample': self._active_sample or '—',
        }
        sections = []
        # Sample / gate summary.
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
        # Current plot.
        try:
            sections.append({'heading': 'Current plot',
                             'html': figure_html(self.fig, alt='current plot')})
        except Exception as exc:
            print(f"[report] plot embed: {exc}", flush=True)
            sections.append({'heading': 'Current plot',
                             'html': '<p class="note">(plot could not be '
                                     'embedded)</p>'})
        # Population statistics.
        try:
            rows, cols = self._collect_stats_rows(
                {'Count', '%Parent', '%Total'})
            if rows:
                disp = [c for c in cols if not c.startswith('__')]
                df = pd.DataFrame([{c: r.get(c) for c in disp} for r in rows])
                sections.append({'heading': 'Population statistics',
                                 'html': df_to_html_table(df, max_rows=500)})
        except Exception as exc:
            print(f"[report] stats: {exc}", flush=True)
        # Cluster heatmap (optional).
        try:
            hm = self._report_heatmap_html()
            if hm:
                sections.append({'heading': 'Cluster heatmap', 'html': hm})
        except Exception as exc:
            print(f"[report] heatmap: {exc}", flush=True)
        # Provenance / audit trail.
        try:
            from .audit import _short
            entries = self._audit_log.entries()
            if entries:
                arows = [{'#': e['seq'], 'Time': e.get('time') or '',
                          'Action': e['action'],
                          'Details': ', '.join(f"{k}={_short(v)}"
                                               for k, v in e['details'].items())}
                         for e in entries]
                sections.append({'heading': 'Provenance (audit trail)',
                                 'html': df_to_html_table(pd.DataFrame(arows))})
        except Exception as exc:
            print(f"[report] audit: {exc}", flush=True)

        try:
            doc = build_html_report('OpenFlo analysis report', meta=meta,
                                    sections=sections)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(doc)
        except Exception as exc:
            messagebox.showerror("Report", f"Could not write report:\n{exc}",
                                 parent=self)
            return
        self._audit('report.export', path=path, sections=len(sections))
        self.status_var.set(f"Report → {os.path.basename(path)}")
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(path))
        except Exception:
            pass

    def _open_expression_window(self):
        if not self._samples:
            self.status_var.set("Load samples first to compare expression.")
            return
        MarkerExpressionWindow(self)

    def _open_sample_qc_window(self):
        if len(self._selected_samples()) < 2:
            self.status_var.set(
                "Enable ≥2 samples (☑) to compare them.")
            return
        SampleQCWindow(self)

    def _open_calibration_dialog(self):
        if not self._samples:
            self.status_var.set("Load a bead sample to calibrate.")
            return
        CalibrationDialog(self)

    def _marker_column_for(self, sample, channel):
        """Resolve a chosen marker ``channel`` to the column it lives on in
        ``sample`` — the channel itself if present, else a detector carrying the
        same antibody label (cross-fluor tying). None if absent."""
        df = sample.data
        if channel in df.columns:
            return channel
        label = self._channel_labels.get(channel, channel)
        for det, lab in (getattr(sample, 'channel_labels', {}) or {}).items():
            if lab == label and det in df.columns:
                return det
        return None

    def _export_population_fcs(self, name, gid):
        """Write the events inside a population (the gate's cumulative mask) to
        a standalone .fcs, re-importable in FlowJo / FCS Express. Exports the
        sample's RAW detector values when they align with the gated rows (so
        the file isn't in transformed coordinates), else the processed data."""
        from .pipeline import cumulative_gate_mask, write_fcs
        s = self._samples.get(name)
        gates = self._sample_gates.get(name, {})
        if s is None or gid not in gates:
            self.status_var.set("Select a gated population to export.")
            return
        mask = np.asarray(cumulative_gate_mask(gates, gid, s.data), dtype=bool)
        n = int(mask.sum())
        if n == 0:
            self.status_var.set("That population is empty — nothing to export.")
            return
        # Prefer raw detector values (untransformed) when row-aligned with data.
        raw = getattr(s, 'raw', None)
        if raw is not None and len(raw) == len(s.data) and not raw.empty:
            export_df = raw.iloc[mask]
            labels = getattr(s, 'channel_labels', {}) or {}
        else:
            export_df = s.data[mask]
            labels = dict(self._channel_labels)
            labels.update(getattr(s, 'channel_labels', {}) or {})

        pop = self._population_path(gates, gid)
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', f"{name}_{pop}").strip('_')
        path = filedialog.asksaveasfilename(
            parent=self, title="Export population as FCS",
            defaultextension='.fcs', initialfile=f"{safe}.fcs",
            filetypes=[('FCS', '*.fcs'), ('All files', '*.*')])
        if not path:
            return
        try:
            written = write_fcs(path, export_df, channel_labels=labels)
        except Exception as exc:
            messagebox.showerror(
                "Export FCS", f"Could not write FCS:\n{type(exc).__name__}: "
                f"{exc}", parent=self)
            return
        self._audit('population.export_fcs', sample=name, population=pop,
                    n_events=written, path=path)
        self.status_var.set(
            f"Exported {written:,} events of '{pop}' → "
            f"{os.path.basename(path)}")

    def _sample_group_label(self, name, factor, tokens=None):
        """Assign a sample to a comparison group by the chosen ``factor``:

          • ``'Trial / day'``  → its trial/day (``_sample_trial``)
          • ``'Comp vs Samples'`` → 'Comps' / 'Samples' (``_sample_is_comp``)
          • ``'Name token'``   → the first token in ``tokens`` the sample name
            contains (case-insensitive), else 'Other'
        """
        if factor == 'Comp vs Samples':
            return 'Comps' if self._sample_is_comp.get(name) else 'Samples'
        if factor == 'Name token':
            low = name.lower()
            # Most-specific (longest) matching token wins, so 'Ctrl' beats
            # 'Stim' (a substring) regardless of the order they were typed.
            matches = [t.strip() for t in (tokens or [])
                       if t.strip() and t.strip().lower() in low]
            return max(matches, key=len) if matches else 'Other'
        return self._sample_trial.get(name, 'Trial')   # Trial / day (default)

    def _maybe_resume_session(self):
        """On open, if a non-empty autosaved session exists, offer to
        resume it. Only prompts when the editor opened empty (don't
        clobber samples the caller passed in)."""
        if self._samples:
            return
        path = self._session_autosave_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return
        n = len(data.get('samples', []))
        if n == 0:
            return
        when = data.get('created', 'unknown time')
        if messagebox.askyesno(
                "Resume last session?",
                f"Found an auto-saved session from {when} with "
                f"{n} sample(s).\n\nResume it?",
                parent=self):
            self._apply_session(data)

    def _open_comp_editor(self):
        """Pop the compensation matrix editor against the active sample
        (so the editor can auto-import from $SPILL / a sibling .wsp / a
        sibling compensation.csv). When the user clicks Apply, the
        active sample's data is re-compensated in place; subsequent
        gate evaluations and plots use the corrected values."""
        if self._active_sample is None or self._active_sample not in self._samples:
            self.status_var.set(
                "Pick a sample first — the editor uses it to find a "
                "matrix and to apply the result.")
            return
        sample = self._samples[self._active_sample]

        def _on_apply(channels, matrix):
            try:
                # Re-load the raw FCS so a second Apply doesn't compound
                # compensation on top of an already-compensated copy, then
                # run the SAME pipeline the loader does — QC, compensate,
                # and (critically) the logicle transform. Without the
                # transform the data would be left on the raw linear scale
                # while the plots / gates / axis scales all expect logicle
                # space, which looks like gross over-compensation (every
                # channel slammed negative).
                from .pipeline import FlowSample
                fresh = FlowSample(sample.path)
                fresh.run_qc()
                fresh.manual_compensate(matrix, list(channels))
                fresh.apply_transform()
                # Preserve antibody labels set on the original sample.
                if getattr(sample, 'channel_labels', None):
                    fresh.set_labels(dict(sample.channel_labels))
                # Drop the new data into the existing FlowSample so every
                # downstream reference (self._samples[name].data, etc.)
                # sees the recompensated values.
                sample.data = fresh.data
                # Persist the applied matrix so it survives a reopen of the
                # editor and rides along in the .wsp export.
                sample.comp_matrix   = getattr(fresh, 'comp_matrix', None)
                sample.comp_channels = list(getattr(fresh, 'comp_channels', []))
                self.status_var.set(
                    f"Applied {len(channels)}×{len(channels)} matrix to "
                    f"'{self._active_sample}'.")
                self._schedule_replot(0)
            except Exception as exc:
                self.status_var.set(f"Compensation failed: {exc}")

        CompensationEditorWindow(self, sample=sample, on_apply=_on_apply)

    def _wsp_lossy_summary(self):
        """List the OpenFlo-only state that a FlowJo .wsp export can't
        carry, given the CURRENT editor state. Empty list → a clean
        export with nothing surprising lost.

        Gate geometry (incl. ellipsoid / quadrant) and the compensation
        matrix DO survive — those aren't reported. We only flag state
        that has no slot in the FlowJo schema AND is actually present:
          - custom per-channel axis scales / ranges (set via the ⚙ dialog)
          - disabled gates (a .wsp would write them as live populations,
            silently changing the analysis)
          - cluster phenotype labels
        Gate colours are mentioned too, but on their own don't trigger
        the warning (FlowJo reassigns its own colours; not surprising).
        """
        items = []
        # Custom axis scales (anything the user changed off the default).
        custom_scales = [ch for ch, sc in self._channel_scale.items()
                         if sc != self._default_channel_scale]
        if custom_scales:
            items.append(
                f"per-channel axis scale for {len(custom_scales)} channel(s) "
                f"({', '.join(custom_scales[:3])}"
                f"{'…' if len(custom_scales) > 3 else ''})")
        if self._channel_range:
            items.append(
                f"custom display range for {len(self._channel_range)} channel(s)")
        # Disabled gates across every sample (cluster/category populations
        # are reported on their own lines below, so exclude them here).
        n_disabled = sum(
            1 for gates in self._sample_gates.values()
            for g in gates.values()
            if not g.get('enabled', True)
            and g.get('kind') not in ('cluster', 'category', 'boolean',
                                      'autoclean'))
        if n_disabled:
            items.append(
                f"{n_disabled} disabled gate(s) — FlowJo would treat them as "
                "active populations")
        # Cluster populations have no FlowJo geometry — they're dropped on
        # export (the phenotype names go with them).
        n_cluster = sum(
            1 for gates in self._sample_gates.values()
            for g in gates.values()
            if g.get('kind') == 'cluster')
        if n_cluster:
            items.append(
                f"{n_cluster} cluster population(s) — no FlowJo equivalent")
        elif self._cluster_labels:
            items.append("cluster phenotype labels")
        # Category populations (e.g. cell-cycle phases) — no FlowJo geometry.
        n_category = sum(
            1 for gates in self._sample_gates.values()
            for g in gates.values()
            if g.get('kind') == 'category')
        if n_category:
            items.append(
                f"{n_category} category population(s) (e.g. cell-cycle) — "
                "no FlowJo equivalent")
        n_boolean = sum(
            1 for gates in self._sample_gates.values()
            for g in gates.values()
            if g.get('kind') == 'boolean')
        if n_boolean:
            items.append(
                f"{n_boolean} boolean gate(s) (AND/OR/NOT) — not exported")
        n_autoclean = sum(
            1 for gates in self._sample_gates.values()
            for g in gates.values()
            if g.get('kind') == 'autoclean')
        if n_autoclean:
            items.append(
                f"{n_autoclean} auto-clean gate(s) — recomputed per sample, "
                "no FlowJo equivalent")
        return items

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
        try:
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
                                                 'boolean', 'autoclean')}
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
            comp_note = ' + spillover' if comp_set else ''
            self.status_var.set(
                f"Exported {len(self._samples)} sample(s) / {total} gate(s)"
                f"{comp_note} → {os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Export failed: {exc}")
            # Status-bar message alone is too easy to miss after a Save
            # dialog — surface the failure visibly.
            messagebox.showerror(
                "Export to FlowJo .wsp failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}",
                parent=self)


# ══════════════════════════════════════════════════════════════════════════════
# CELL-CYCLE RESULT WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class CellCycleWindow(tk.Toplevel):
    """DNA-content histogram with G1/S/G2M boundaries + phase percentages
    for one sample's cell-cycle result."""

    def __init__(self, editor, sample_name):
        super().__init__(editor)
        self.title(f"Cell cycle — {sample_name}")
        self.geometry("720x500")
        self.minsize(480, 320)

        s   = editor._samples[sample_name]
        res = getattr(s, 'cell_cycle_result', None)
        if not res or not res.get('ok'):
            ttk.Label(self, text="No cell-cycle result for this sample.").pack(
                padx=20, pady=20)
            return

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        col   = res['channel']
        phase = np.asarray(s.data['cell_cycle'].values)
        vals  = np.asarray(s.data[col].values, dtype=float)
        keep  = (phase != 'NA') & np.isfinite(vals)
        v     = vals[keep]

        fig = Figure(figsize=(7, 4))
        ax  = fig.add_subplot(111)
        if v.size:
            lo, hi = np.percentile(v, [0.5, 99.5])
            ax.hist(v, bins=200, range=(float(lo), float(hi)),
                    color='#999999', alpha=0.65)
        # Phase means (solid) + G1|S and S|G2M boundaries (dashed).
        ax.axvline(res['g1_mean'], color='#4363d8', lw=1.4, label='G1')
        ax.axvline(res['g2_mean'], color='#e6194b', lw=1.4, label='G2/M')
        ax.axvline(res['g1_hi'], color='#3cb44b', ls='--', lw=1)
        ax.axvline(res['g2_lo'], color='#3cb44b', ls='--', lw=1)
        ax.set_xlabel(editor._fmt_channel(col))
        ax.set_ylabel('events')
        ax.set_title(f"Cell cycle — {sample_name}")
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        summary = (
            f"G1 {res['pct_g1']:.1f}%      "
            f"S {res['pct_s']:.1f}%      "
            f"G2/M {res['pct_g2m']:.1f}%        "
            f"({res['n_cycling']:,} cycling of {res['n_singlet']:,} singlets)")
        ttk.Label(self, text=summary,
                  font=('TkDefaultFont', 10, 'bold')).pack(pady=(4, 4))
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 8))


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS TABLE (FlowJo-style population statistics)
# ══════════════════════════════════════════════════════════════════════════════

class StatisticsWindow(tk.Toplevel):
    """Population-statistics table over the editor's loaded samples.

    Rows = sample × population (gate node, evaluated as the cumulative
    gate chain). Columns are modular — toggle Count / %Parent / %Total
    and per-channel Median / Mean / CV. Export the current table to CSV.
    Recomputes on toggle or Refresh; uses the full sample data (not the
    plot's downsample).
    """

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Population statistics")
        self.geometry("1000x560")
        self.minsize(640, 360)

        self._cols = []     # current Treeview column ids
        self._rows = []     # current row dicts
        # Curated target set: an ordered list of (sample, gid) POPULATIONS —
        # statistics is population-based, so only gate rows are accepted (never
        # whole samples). Drag from either tree APPENDS; the Import buttons
        # OVERRIDE; Clear empties it. ``_gate_sources`` maps (sample, gid) →
        # {'editor','workspace'} for the Source column.
        #
        # ``_user_curated`` separates "fresh window → show every population as a
        # convenience" (False) from "the user has touched the set" (True). Once
        # True, the table shows EXACTLY ``_gate_targets`` — so Clear → empty
        # table that STAYS empty instead of auto-repopulating with all pops.
        self._gate_targets = []     # [(sample, gid)] in display order
        self._gate_sources = {}     # (sample, gid) -> {source}
        self._user_curated = False

        # ── Column-selection checkboxes ──────────────────────────────────
        opts = ttk.Frame(self, padding=(10, 8, 10, 4))
        opts.pack(side='top', fill='x')
        ttk.Label(opts, text="Columns:",
                  font=('TkDefaultFont', 9, 'bold')).pack(side='left',
                                                          padx=(0, 8))
        self._stat_vars = {}
        defaults = {'Count', '%Parent', '%Total', 'Median'}
        for stat in (*editor.STAT_POP, *editor.STAT_CHAN):
            var = tk.BooleanVar(value=(stat in defaults))
            self._stat_vars[stat] = var
            ttk.Checkbutton(opts, text=stat, variable=var,
                            command=self._refresh).pack(side='left',
                                                        padx=(0, 6))

        # ── Table ────────────────────────────────────────────────────────
        table_holder = ttk.Frame(self, padding=(10, 0, 10, 6))
        table_holder.pack(side='top', fill='both', expand=True)
        self.tv = ttk.Treeview(table_holder, show='headings', height=18)
        ysb = ttk.Scrollbar(table_holder, orient='vertical',
                            command=self.tv.yview)
        xsb = ttk.Scrollbar(table_holder, orient='horizontal',
                            command=self.tv.xview)
        self.tv.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tv.grid(row=0, column=0, sticky='nsew')
        ysb.grid(row=0, column=1, sticky='ns')
        xsb.grid(row=1, column=0, sticky='ew')
        table_holder.rowconfigure(0, weight=1)
        table_holder.columnconfigure(0, weight=1)

        # ── Buttons + status ─────────────────────────────────────────────
        self.status_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.status_var, foreground='grey',
                  padding=(10, 0, 10, 2)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side='bottom', fill='x')
        ttk.Button(btns, text="Refresh", command=self._refresh).pack(side='left')
        ttk.Button(btns, text="Export CSV…",
                   command=self._export_csv).pack(side='left', padx=(6, 0))
        # Per-side bulk import — each REPLACES the curated set (override).
        # Dragging a gate row from either tree APPENDS instead.
        ttk.Button(btns, text="Import S&G gates",
                   command=self._import_all_editor).pack(side='left', padx=(12, 0))
        ttk.Button(btns, text="Import workspace",
                   command=self._import_all_workspace).pack(side='left', padx=(6, 0))
        ttk.Button(btns, text="Clear",
                   command=self._clear_targets).pack(side='left', padx=(6, 0))
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')

        self.status_var.set("Showing all populations. Drag a gate from the "
                            "panel or a gated workspace item here to add it; "
                            "the Import buttons replace the set.")
        self._refresh()

    # ── Curated population set (drag = append, import = override) ─────────
    def add_targets(self, targets, source):
        """APPEND (sample, gid) populations to the curated set, tagging each
        with the side it came from ('editor' | 'workspace'). Switches the table
        out of the default 'all populations' mode. Called by the editor /
        workspace gate-drag handlers."""
        added = 0
        for t in targets:
            if not (isinstance(t, tuple) and len(t) == 2 and all(t)):
                continue
            if t not in self._gate_targets:
                self._gate_targets.append(t)
            self._gate_sources.setdefault(t, set()).add(source)
            added += 1
        if added:
            self._user_curated = True
            self._refresh()
            try:
                self.lift()
                self.focus_set()
            except Exception:
                pass

    def _set_targets(self, targets, source):
        """REPLACE the curated set with `targets` (override). Used by Import."""
        self._gate_targets = []
        self._gate_sources = {}
        for t in targets:
            if t not in self._gate_targets:
                self._gate_targets.append(t)
            self._gate_sources.setdefault(t, set()).add(source)
        self._user_curated = True
        self._refresh()
        try:
            self.lift()
            self.focus_set()
        except Exception:
            pass

    def _clear_targets(self):
        """Empty the table and keep it empty (does NOT revert to showing every
        population). Use Refresh or an Import button to repopulate."""
        self._gate_targets = []
        self._gate_sources = {}
        self._user_curated = True
        self._refresh()

    def _import_all_editor(self):
        """Override the set with every gate of every loaded editor sample."""
        targets = []
        for name in self.editor._sample_order:
            if name not in self.editor._samples:
                continue
            order = (self.editor._sample_gate_order.get(name)
                     or list(self.editor._sample_gates.get(name, {})))
            for gid in order:
                targets.append((name, gid))
        if not targets:
            self.status_var.set("No gates in the loaded samples to import.")
            return
        self._set_targets(targets, 'editor')

    def _import_all_workspace(self):
        """Override the set with every gated population in the workspace."""
        panel = getattr(self.editor, '_workspace_panel', None)
        model = getattr(panel, 'model', None) if panel is not None else None
        if model is None:
            self.status_var.set("No Pipeline Workspace is open.")
            return
        targets = []
        for _mid, it, _gid in model.all_items():
            nm, gid = it.get('sample'), it.get('gate_id')
            if nm and gid and (nm, gid) not in targets:
                targets.append((nm, gid))
        if not targets:
            self.status_var.set("The workspace has no gated populations to import.")
            return
        self._set_targets(targets, 'workspace')

    def _selected_stats(self):
        return {s for s, v in self._stat_vars.items() if v.get()}

    def _refresh(self):
        want = self._selected_stats()
        # Curated mode (the user has dragged/imported/cleared): show EXACTLY the
        # targeted (sample, gid) populations — an empty set stays empty. A fresh
        # window (not yet curated) shows every population as a convenience.
        curated = self._user_curated
        try:
            if curated:
                rows, cols = self.editor._collect_stats_rows(
                    want, gate_targets=self._gate_targets)
            else:
                rows, cols = self.editor._collect_stats_rows(want)
        except Exception as exc:
            self.status_var.set(f"Stats failed: {exc}")
            return
        # Annotate each row with the side(s) it came from (keyed on the gate).
        if curated:
            cols = list(cols)
            if 'Source' not in cols:
                cols.insert(2, 'Source')   # after Sample, Population
            for r in rows:
                srcs = self._gate_sources.get((r.get('Sample'), r.get('__gid__')))
                r['Source'] = '+'.join(sorted(srcs)) if srcs else ''
        self._rows, self._cols = rows, cols

        self.tv.delete(*self.tv.get_children())
        self.tv['columns'] = cols
        for c in cols:
            self.tv.heading(c, text=c)
            anchor = 'w' if c in ('Sample', 'Population', 'Source') else 'e'
            width = (220 if c == 'Population'
                     else 120 if c == 'Sample'
                     else 80 if c == 'Source' else 90)
            self.tv.column(c, anchor=anchor, width=width, stretch=False)

        for r in rows:
            values = [self._fmt(c, r.get(c, '')) for c in cols]
            self.tv.insert('', 'end', values=values)
        msg = (f"{len(rows)} population row(s) across "
               f"{len({r['Sample'] for r in rows})} sample(s)"
               + (" (curated populations)." if curated else "."))
        # Per-channel columns are named by antibody label, so a marker on
        # different fluors across samples already merges into one column.
        # Flag any label that isn't shared by every sample.
        if self.editor._fluor_panel_warning():
            msg += ("  [!] samples differ in fluor panel — non-common "
                    "labels are blank where absent.")
        if curated:
            missing = {nm for nm, _gid in self._gate_targets
                       if nm not in self.editor._samples}
            if missing:
                msg += (f"  [!] {len(missing)} sample(s) not loaded in the "
                        "editor — Add FCS to include their populations.")
        self.status_var.set(msg)

    @staticmethod
    def _fmt(col, val):
        if val == '' or val is None:
            return ''
        if col == 'Count':
            try:
                return f"{int(val):,}"
            except (TypeError, ValueError):
                return str(val)
        if isinstance(val, float):
            if val != val:        # NaN
                return ''
            return f"{val:.3g}"
        return str(val)

    def _export_csv(self):
        if not self._rows:
            self.status_var.set("Nothing to export — no populations.")
            return
        path = filedialog.asksaveasfilename(
            title="Export statistics to CSV",
            defaultextension='.csv',
            initialfile='population_stats.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if not path:
            return
        try:
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=self._cols, extrasaction='ignore')
                w.writeheader()
                for r in self._rows:
                    # Write raw numeric values (not the display-formatted
                    # strings) so the CSV is analysis-ready.
                    w.writerow({c: ('' if (isinstance(r.get(c), float)
                                           and r.get(c) != r.get(c))
                                    else r.get(c, '')) for c in self._cols})
            self.status_var.set(f"Exported {len(self._rows)} row(s) → "
                                f"{os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Export failed: {exc}")
            messagebox.showerror(
                "Export statistics failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}", parent=self)


# ══════════════════════════════════════════════════════════════════════════════
# AXIS CONFIG DIALOG (per-channel scale + range)
# ══════════════════════════════════════════════════════════════════════════════

class SpectralUnmixDialog(tk.Toplevel):
    """Assign loaded samples to roles for spectral unmixing — single-stain
    (→ fluorophore), unstained, or ignore — and pick the detector channels.
    Calls ``on_apply(singles {name: fluor}, unstained_name|None, detectors,
    nonneg)`` on Build & Apply."""

    def __init__(self, parent, sample_names, detectors, on_apply):
        super().__init__(parent)
        self.title("Spectral unmixing")
        self.transient(parent)
        self.resizable(False, False)
        self.on_apply = on_apply

        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Label(
            body, justify='left',
            text=("Designate the single-stain control samples (→ fluorophore) "
                  "and one unstained\ncontrol. Every other loaded sample is "
                  "unmixed into per-fluor 'U:' channels.")).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))
        ttk.Label(body, text="Sample", font=('TkDefaultFont', 9, 'bold')).grid(
            row=1, column=0, sticky='w')
        ttk.Label(body, text="Role", font=('TkDefaultFont', 9, 'bold')).grid(
            row=1, column=1, sticky='w', padx=8)
        ttk.Label(body, text="Fluorophore",
                  font=('TkDefaultFont', 9, 'bold')).grid(
            row=1, column=2, sticky='w')

        self.rows = []
        roles = ['Ignore', 'Single-stain', 'Unstained']
        for i, nm in enumerate(sample_names):
            ttk.Label(body, text=(nm[:34])).grid(row=2 + i, column=0, sticky='w')
            rv = tk.StringVar(value='Ignore')
            ttk.Combobox(body, textvariable=rv, values=roles, state='readonly',
                         width=12).grid(row=2 + i, column=1, padx=8, pady=1)
            fv = tk.StringVar(value='')
            ttk.Entry(body, textvariable=fv, width=22).grid(
                row=2 + i, column=2, sticky='w')
            self.rows.append((nm, rv, fv))

        r = 2 + len(sample_names)
        ttk.Label(body, text="Detectors:").grid(
            row=r, column=0, sticky='ne', pady=(8, 0))
        self.det_txt = tk.Text(body, height=3, width=46, wrap='word')
        self.det_txt.insert('1.0', ', '.join(detectors))
        self.det_txt.grid(row=r, column=1, columnspan=2, sticky='w', pady=(8, 0))
        self.nonneg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Non-negative abundances",
                        variable=self.nonneg_var).grid(
            row=r + 1, column=1, columnspan=2, sticky='w', pady=(6, 0))

        bb = ttk.Frame(body)
        bb.grid(row=r + 2, column=0, columnspan=3, sticky='e', pady=(12, 0))
        ttk.Button(bb, text="Cancel", command=self.destroy).pack(side='right')
        ttk.Button(bb, text="Build & Apply", command=self._apply).pack(
            side='right', padx=(0, 6))
        try:
            self.grab_set()
        except Exception:
            pass

    def _apply(self):
        singles, unstained = {}, None
        for nm, rv, fv in self.rows:
            role = rv.get()
            if role == 'Single-stain':
                singles[nm] = fv.get().strip() or nm
            elif role == 'Unstained':
                unstained = nm
        dets = [d.strip() for d in
                self.det_txt.get('1.0', 'end').replace('\n', ' ').split(',')
                if d.strip()]
        if not singles:
            messagebox.showwarning(
                "Spectral unmixing",
                "Assign at least one single-stain control to a fluorophore.",
                parent=self)
            return
        if len(dets) < 2:
            messagebox.showwarning(
                "Spectral unmixing", "Need at least 2 detector channels.",
                parent=self)
            return
        self.on_apply(singles, unstained, dets, bool(self.nonneg_var.get()))
        self.destroy()


class FigureLayoutDialog(tk.Toplevel):
    """Configure a multi-panel publication figure built from the current
    plot selection. Calls ``on_apply(opts)`` with a dict::

        {layout, ncols, pairs, gates}

    where ``layout`` is one of ``single`` / ``per_sample`` / ``per_pair`` /
    ``grid``. The plot mode, colouring and (for the single/per-sample
    layouts) the channels come from the live plot controls."""

    def __init__(self, parent, n_samples, mode, default_pairs, on_apply):
        super().__init__(parent)
        self.title("Figure layout")
        self.transient(parent)
        self.resizable(False, False)
        self.on_apply = on_apply

        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Label(
            body, justify='left',
            text=(f"{n_samples} sample(s) enabled · mode: {mode}\n"
                  "Build a multi-panel figure from the current plot. "
                  "Channel pairs apply to the\npair / grid layouts "
                  "(e.g. \"CD34/CD11b, CD11b/CD45\"; markers or "
                  "channel names).")).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))

        ttk.Label(body, text="Layout:", font=('TkDefaultFont', 9, 'bold')
                  ).grid(row=1, column=0, sticky='w')
        self.layout_var = tk.StringVar(value='per_sample')
        layouts = [
            ('One panel per sample (current channels)', 'per_sample'),
            ('One panel per channel pair (samples overlaid)', 'per_pair'),
            ('Grid: samples × channel pairs', 'grid'),
            ('Single panel (current view)', 'single'),
        ]
        lf = ttk.Frame(body)
        lf.grid(row=2, column=0, columnspan=2, sticky='w', pady=(2, 8))
        for lbl, val in layouts:
            ttk.Radiobutton(lf, text=lbl, value=val,
                            variable=self.layout_var,
                            command=self._sync_enabled).pack(anchor='w')

        ttk.Label(body, text="Channel pairs:").grid(
            row=3, column=0, sticky='nw', pady=(4, 0))
        self.pairs_txt = tk.Text(body, height=3, width=42, wrap='word')
        self.pairs_txt.insert('1.0', default_pairs)
        self.pairs_txt.grid(row=3, column=1, sticky='w', pady=(4, 0))

        ttk.Label(body, text="Columns:").grid(
            row=4, column=0, sticky='w', pady=(8, 0))
        self.ncols_var = tk.StringVar(value='3')
        self.ncols_spin = ttk.Spinbox(body, from_=1, to=12, width=6,
                                      textvariable=self.ncols_var)
        self.ncols_spin.grid(row=4, column=1, sticky='w', pady=(8, 0))

        self.gates_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Draw gates on panels",
                        variable=self.gates_var).grid(
            row=5, column=0, columnspan=2, sticky='w', pady=(8, 0))

        bb = ttk.Frame(body)
        bb.grid(row=6, column=0, columnspan=2, sticky='e', pady=(12, 0))
        ttk.Button(bb, text="Cancel", command=self.destroy).pack(side='right')
        ttk.Button(bb, text="Build", command=self._apply).pack(
            side='right', padx=(0, 6))

        self._sync_enabled()
        try:
            self.grab_set()
        except Exception:
            pass

    def _sync_enabled(self):
        layout = self.layout_var.get()
        needs_pairs = layout in ('per_pair', 'grid')
        self.pairs_txt.configure(
            state=('normal' if needs_pairs else 'disabled'))
        # Grid derives its column count from the number of pairs.
        self.ncols_spin.configure(
            state=('disabled' if layout == 'grid' else 'normal'))

    def _apply(self):
        try:
            ncols = max(1, int(self.ncols_var.get()))
        except (TypeError, ValueError):
            ncols = 3
        opts = {
            'layout': self.layout_var.get(),
            'ncols': ncols,
            'pairs': self.pairs_txt.get('1.0', 'end').strip(),
            'gates': bool(self.gates_var.get()),
        }
        self.on_apply(opts)
        self.destroy()


class AutoGateDialog(tk.Toplevel):
    """Choose an automated-gating method for the active sample. Calls
    ``on_apply(opts)`` with a dict whose ``method`` is one of:

      • ``singlet``   — FSC-A/FSC-H ratio-band polygon (+ ``k``, ``area``,
                        ``height``)
      • ``gmm``       — BIC-selected Gaussian-mixture ellipses on the current
                        X/Y plot (+ ``max_components``, ``coverage``,
                        ``min_weight``)
      • ``threshold`` — 1-D valley/Otsu split on the current X channel

    Each proposal is added as an ordinary undoable gate the user reviews."""

    def __init__(self, parent, has_singlet, area, height, cur_x, cur_y,
                 on_apply):
        super().__init__(parent)
        self.title("Auto-gate")
        self.transient(parent)
        self.resizable(False, False)
        self.on_apply = on_apply
        self._area = area
        self._height = height

        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Label(
            body, justify='left',
            text=("Propose gates from the data — each is added as an ordinary,\n"
                  "editable gate you can accept, tweak or delete. Quality is\n"
                  "reported in the status bar.")).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))

        self.method_var = tk.StringVar(
            value='singlet' if has_singlet else 'gmm')

        mf = ttk.Frame(body)
        mf.grid(row=1, column=0, columnspan=2, sticky='w')
        singlet_lbl = ("Singlet gate (FSC-A vs FSC-H ratio band)"
                       if has_singlet
                       else "Singlet gate — needs an FSC-A + FSC-H pair")
        self._singlet_rb = ttk.Radiobutton(
            mf, text=singlet_lbl, value='singlet',
            variable=self.method_var, command=self._sync)
        if not has_singlet:
            self._singlet_rb.configure(state='disabled')
        self._singlet_rb.pack(anchor='w')
        if area and height:
            ttk.Label(mf, text=f"    {area}  vs  {height}",
                      foreground='#666').pack(anchor='w')

        ttk.Radiobutton(
            mf, text=f"Find populations (GMM ellipses) on  {cur_x or '?'} × "
                     f"{cur_y or '?'}",
            value='gmm', variable=self.method_var,
            command=self._sync).pack(anchor='w', pady=(4, 0))
        ttk.Radiobutton(
            mf, text=f"1-D threshold on  {cur_x or '?'}  (valley / Otsu)",
            value='threshold', variable=self.method_var,
            command=self._sync).pack(anchor='w', pady=(4, 0))

        # ── Singlet params ──
        self.singlet_frame = ttk.LabelFrame(body, text="Singlet band",
                                            padding=8)
        self.singlet_frame.grid(row=2, column=0, columnspan=2, sticky='ew',
                                pady=(10, 0))
        ttk.Label(self.singlet_frame, text="Band width (× robust σ):").grid(
            row=0, column=0, sticky='w')
        self.k_var = tk.StringVar(value='3.0')
        k_sp = ttk.Spinbox(self.singlet_frame, from_=1.0, to=6.0,
                           increment=0.5, width=6, textvariable=self.k_var)
        k_sp.grid(row=0, column=1, sticky='w', padx=(6, 0))
        self._singlet_inputs = [k_sp]

        # ── GMM params ──
        self.gmm_frame = ttk.LabelFrame(body, text="GMM ellipses", padding=8)
        self.gmm_frame.grid(row=3, column=0, columnspan=2, sticky='ew',
                            pady=(8, 0))
        ttk.Label(self.gmm_frame, text="Max populations:").grid(
            row=0, column=0, sticky='w')
        self.kmax_var = tk.StringVar(value='6')
        kmax_sp = ttk.Spinbox(self.gmm_frame, from_=1, to=12, width=6,
                              textvariable=self.kmax_var)
        kmax_sp.grid(row=0, column=1, sticky='w', padx=(6, 12))
        ttk.Label(self.gmm_frame, text="Coverage %:").grid(
            row=0, column=2, sticky='w')
        self.cov_var = tk.StringVar(value='90')
        cov_sp = ttk.Spinbox(self.gmm_frame, from_=50, to=99, width=6,
                             textvariable=self.cov_var)
        cov_sp.grid(row=0, column=3, sticky='w', padx=(6, 0))
        ttk.Label(self.gmm_frame, text="Min population %:").grid(
            row=1, column=0, sticky='w', pady=(6, 0))
        self.minw_var = tk.StringVar(value='2')
        minw_sp = ttk.Spinbox(self.gmm_frame, from_=0, to=25, width=6,
                              textvariable=self.minw_var)
        minw_sp.grid(row=1, column=1, sticky='w', padx=(6, 0), pady=(6, 0))
        self._gmm_inputs = [kmax_sp, cov_sp, minw_sp]

        bb = ttk.Frame(body)
        bb.grid(row=4, column=0, columnspan=2, sticky='e', pady=(12, 0))
        ttk.Button(bb, text="Cancel", command=self.destroy).pack(side='right')
        ttk.Button(bb, text="Propose", command=self._apply).pack(
            side='right', padx=(0, 6))

        self._sync()
        try:
            self.grab_set()
        except Exception:
            pass

    def _sync(self):
        m = self.method_var.get()
        for sp in self._singlet_inputs:
            sp.configure(state=('normal' if m == 'singlet' else 'disabled'))
        for sp in self._gmm_inputs:
            sp.configure(state=('normal' if m == 'gmm' else 'disabled'))

    def _apply(self):
        method = self.method_var.get()
        opts: dict = {'method': method}
        if method == 'singlet':
            opts['area'] = self._area
            opts['height'] = self._height
            try:
                opts['k'] = float(self.k_var.get())
            except ValueError:
                opts['k'] = 3.0
        elif method == 'gmm':
            try:
                opts['max_components'] = max(1, int(self.kmax_var.get()))
            except ValueError:
                opts['max_components'] = 6
            try:
                opts['coverage'] = min(0.999, max(0.5,
                                   float(self.cov_var.get()) / 100.0))
            except ValueError:
                opts['coverage'] = 0.90
            try:
                opts['min_weight'] = max(0.0,
                                   float(self.minw_var.get()) / 100.0)
            except ValueError:
                opts['min_weight'] = 0.02
        self.on_apply(opts)
        self.destroy()


class AuditWindow(tk.Toplevel):
    """Read-only viewer for the analysis audit trail, with Markdown / CSV /
    JSON export. Non-modal and live: ``refresh()`` is called by the editor's
    ``_audit`` whenever a new operation is recorded while this is open."""

    def __init__(self, parent, audit_log):
        super().__init__(parent)
        self.editor = parent
        self.title("Analysis history (audit trail)")
        self.geometry("820x520")
        self._log = audit_log

        bar = ttk.Frame(self)
        bar.pack(fill='x', side='top')
        ttk.Label(bar, text="Provenance — operations in order:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='left', padx=6, pady=4)
        ttk.Button(bar, text="Export Markdown…",
                   command=lambda: self._export('md')).pack(
            side='right', padx=(0, 6), pady=4)
        ttk.Button(bar, text="Export CSV…",
                   command=lambda: self._export('csv')).pack(
            side='right', padx=(0, 4), pady=4)
        ttk.Button(bar, text="Export JSON…",
                   command=lambda: self._export('json')).pack(
            side='right', padx=(0, 4), pady=4)

        bar2 = ttk.Frame(self)
        bar2.pack(fill='x', side='top')
        ttk.Label(bar2, text="Compliance:", foreground='#555').pack(
            side='left', padx=6)
        ttk.Button(bar2, text="Sign & export record…",
                   command=self._sign_record).pack(side='left', padx=(0, 4),
                                                   pady=2)
        ttk.Button(bar2, text="Verify record…",
                   command=self._verify_record).pack(side='left', pady=2)

        cols = ('seq', 'time', 'action', 'details')
        widths = (40, 150, 130, 460)
        tv = ttk.Treeview(self, columns=cols, show='headings',
                          selectmode='browse')
        for c, w in zip(cols, widths, strict=True):
            tv.heading(c, text=c.capitalize())
            tv.column(c, width=w, anchor='w',
                      stretch=(c == 'details'))
        sb = ttk.Scrollbar(self, orient='vertical', command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        tv.pack(side='left', fill='both', expand=True)
        self._tv = tv
        self.refresh()

    def refresh(self):
        from .audit import _short
        tv = self._tv
        tv.delete(*tv.get_children())
        entries = self._log.entries()
        for e in entries:
            det = ", ".join(f"{k}={_short(v)}"
                            for k, v in e['details'].items())
            tv.insert('', 'end',
                      values=(e['seq'], e.get('time') or '',
                              e['action'], det))
        if entries:
            tv.see(tv.get_children()[-1])

    def _meta(self):
        from datetime import datetime

        from . import __version__
        return {'openflo_version': __version__,
                'exported': datetime.now().isoformat(timespec='seconds'),
                'operations': len(self._log)}

    def _export(self, fmt):
        if not len(self._log):
            messagebox.showinfo("History", "Nothing to export yet.",
                                parent=self)
            return
        ext = {'md': '.md', 'csv': '.csv', 'json': '.json'}[fmt]
        ftypes = {'md': [('Markdown', '*.md')],
                  'csv': [('CSV', '*.csv')],
                  'json': [('JSON', '*.json')]}[fmt]
        path = filedialog.asksaveasfilename(
            parent=self, title="Export audit trail", defaultextension=ext,
            initialfile='audit_trail' + ext,
            filetypes=ftypes + [('All files', '*.*')])
        if not path:
            return
        try:
            if fmt == 'md':
                text = self._log.to_markdown(meta=self._meta())
            elif fmt == 'csv':
                text = self._log.to_csv()
            else:
                text = json.dumps(
                    {'format': 'openflo-audit', 'version': 1,
                     'meta': self._meta(), 'entries': self._log.to_list()},
                    indent=2, ensure_ascii=False)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as exc:
            messagebox.showerror(
                "History", f"Export failed:\n{type(exc).__name__}: {exc}",
                parent=self)
            return
        messagebox.showinfo("History", f"Exported:\n{path}", parent=self)

    # ── Compliance / sign-off ────────────────────────────────────────────
    def _sign_record(self):
        """Build an integrity manifest (data-file hashes + audit-trail hash +
        version), attach an electronic signature, and write a signed JSON
        record + a Markdown copy."""
        from datetime import datetime
        from tkinter import simpledialog

        from . import __version__
        from .compliance import (
            build_manifest,
            record_to_markdown,
            sign_manifest,
        )
        ed = self.editor
        files = {n: getattr(ed._samples[n], 'path', '') or ''
                 for n in getattr(ed, '_sample_order', [])
                 if n in ed._samples}
        signer = simpledialog.askstring(
            "Electronic signature", "Signer (name / ID):", parent=self)
        if not signer:
            return
        meaning = simpledialog.askstring(
            "Electronic signature", "Meaning of signature:",
            initialvalue="Reviewed and approved", parent=self) or "Signed"
        now = datetime.now().isoformat(timespec='seconds')
        manifest = build_manifest(files, self._log.to_list(), __version__,
                                  created=now)
        record = sign_manifest(manifest, signer, meaning, now)
        path = filedialog.asksaveasfilename(
            parent=self, title="Save signed compliance record",
            defaultextension='.json', initialfile='compliance_record.json',
            filetypes=[('Signed record (JSON)', '*.json')])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            md = os.path.splitext(path)[0] + '.md'
            with open(md, 'w', encoding='utf-8') as f:
                f.write(record_to_markdown(record))
        except Exception as exc:
            messagebox.showerror("Compliance", f"Write failed:\n{exc}",
                                 parent=self)
            return
        try:
            ed._audit('compliance.sign', signer=signer, meaning=meaning,
                      path=path)
        except Exception:
            pass
        messagebox.showinfo(
            "Compliance",
            f"Signed by {signer} — {len(files)} files hashed.\n\n{path}",
            parent=self)

    def _verify_record(self):
        """Load a signed record and re-check the manifest + file hashes,
        reporting whether the signatures are still valid (tamper check)."""
        from .compliance import verify_record
        path = filedialog.askopenfilename(
            parent=self, title="Verify signed compliance record",
            filetypes=[('Signed record (JSON)', '*.json'),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                record = json.load(f)
        except Exception as exc:
            messagebox.showerror("Compliance", f"Could not read:\n{exc}",
                                 parent=self)
            return
        v = verify_record(record)
        lines = [f"Overall: {'VALID' if v['all_valid'] else 'INVALID / TAMPERED'}",
                 ""]
        for s in v['signatures']:
            lines.append(f"  {'✓' if s['valid'] else '✗'} {s['signer']} — "
                         f"{s['meaning']} ({s['time']})")
        bad = [n for n, ok in v['files_ok'].items() if not ok]
        if bad:
            lines.append("")
            lines.append("Changed/missing data files: " + ", ".join(bad))
        (messagebox.showinfo if v['all_valid'] else messagebox.showwarning)(
            "Verify compliance record", "\n".join(lines), parent=self)


class SpectralQCWindow(tk.Toplevel):
    """Spectral-unmixing quality view: the spectral SIMILARITY matrix and the
    Spillover Spread Matrix (SSM) as heatmaps, the condition number, and the
    flagged similar / high-spread fluor pairs — with Markdown / PNG export."""

    def __init__(self, parent, qc, audit=None):
        super().__init__(parent)
        self.title("Spectral QC (unmixing diagnostics)")
        self.geometry("960x680")
        self._qc = qc
        self._audit = audit

        bar = ttk.Frame(self)
        bar.pack(fill='x', side='top')
        cond = qc.get('condition_number', float('nan'))
        cond_txt = "∞" if cond == float('inf') else f"{cond:.1f}"
        warn = "  [ill-conditioned]" if (cond == float('inf') or cond > 100) \
            else ""
        ttk.Label(
            bar,
            text=(f"{len(qc['fluors'])} fluors · condition number {cond_txt}"
                  f"{warn} · {len(qc['similar_pairs'])} similar pair(s)"),
            font=('TkDefaultFont', 9, 'bold')).pack(side='left', padx=6, pady=4)
        self._bg_var = tk.StringVar(value='White')
        ttk.Button(bar, text="Export PNG…",
                   command=self._export_png).pack(side='right', padx=(0, 6),
                                                  pady=4)
        ttk.Combobox(bar, textvariable=self._bg_var, width=12,
                     state='readonly',
                     values=['White', 'Transparent', 'Translucent']).pack(
            side='right', padx=(0, 4), pady=4)
        ttk.Label(bar, text="PNG background:").pack(side='right', padx=(0, 2))
        ttk.Button(bar, text="Export Markdown…",
                   command=self._export_md).pack(side='right', padx=(0, 4),
                                                 pady=4)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(9, 4.2), dpi=100)
        self._draw_heatmaps(self._fig)
        cf = ttk.Frame(self)
        cf.pack(fill='both', expand=True)
        canvas = FigureCanvasTkAgg(self._fig, master=cf)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

        # Flagged pairs as plain text underneath.
        txt = tk.Text(self, height=8, wrap='word')
        txt.pack(fill='x', side='bottom')
        txt.insert('1.0', self._summary_text())
        txt.configure(state='disabled')

    def _draw_heatmaps(self, fig):
        import numpy as _np
        fluors = self._qc['fluors']
        n = len(fluors)
        sim = _np.asarray(self._qc['similarity'], dtype=float)
        ssm = _np.asarray(self._qc['ssm'], dtype=float)
        ssm_masked = _np.ma.masked_invalid(ssm)

        ax1 = fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(sim, vmin=0.0, vmax=1.0, cmap='magma',
                         aspect='auto')
        ax1.set_title('Spectral similarity', fontsize=9)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(1, 2, 2)
        cmap = plt_get_cmap('viridis')
        cmap.set_bad('lightgrey')
        im2 = ax2.imshow(ssm_masked, cmap=cmap, aspect='auto')
        ax2.set_title('Spillover spread (SSM)', fontsize=9)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        for ax in (ax1, ax2):
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(fluors, rotation=90, fontsize=7)
            ax.set_yticklabels(fluors, fontsize=7)
        try:
            fig.tight_layout()
        except Exception:
            pass

    def _summary_text(self):
        lines = []
        sp = self._qc['similar_pairs']
        if sp:
            lines.append("Spectrally-similar pairs (hard to resolve):")
            for d in sp[:10]:
                lines.append(f"  • {d['fluor_a']} ~ {d['fluor_b']}  "
                             f"(cosine {d['similarity']:.3f})")
        else:
            lines.append("No fluor pair exceeds the similarity threshold — "
                         "spectra are well separated.")
        ws = self._qc['worst_spread']
        if ws:
            lines.append("")
            lines.append("Largest spillover spread (into ← from):")
            for d in ws:
                lines.append(f"  • {d['into']} ← {d['from']}  "
                             f"({d['spread']:.3g})")
        return "\n".join(lines)

    def _markdown(self):
        from datetime import datetime

        from . import __version__
        q = self._qc
        cond = q['condition_number']
        cond_txt = "inf" if cond == float('inf') else f"{cond:.2f}"
        out = ["# Spectral unmixing QC", ""]
        out.append(f"- **openflo_version**: {__version__}")
        out.append(f"- **exported**: "
                   f"{datetime.now().isoformat(timespec='seconds')}")
        out.append(f"- **fluors**: {len(q['fluors'])}")
        out.append(f"- **condition_number**: {cond_txt}")
        out.append("")
        out.append("## Spectrally-similar pairs")
        if q['similar_pairs']:
            out.append("| Fluor A | Fluor B | Cosine similarity |")
            out.append("|---|---|---|")
            for d in q['similar_pairs']:
                out.append(f"| {d['fluor_a']} | {d['fluor_b']} | "
                           f"{d['similarity']:.4f} |")
        else:
            out.append("None above threshold.")
        out.append("")
        out.append("## Largest spillover spread")
        if q['worst_spread']:
            out.append("| Into | From | Spread |")
            out.append("|---|---|---|")
            for d in q['worst_spread']:
                out.append(f"| {d['into']} | {d['from']} | "
                           f"{d['spread']:.4g} |")
        else:
            out.append("No measured spread (single-stain controls missing).")
        out.append("")
        return "\n".join(out)

    def _export_md(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Export spectral QC", defaultextension='.md',
            initialfile='spectral_qc.md',
            filetypes=[('Markdown', '*.md'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self._markdown())
        except Exception as exc:
            messagebox.showerror("Spectral QC",
                                 f"Export failed:\n{exc}", parent=self)
            return
        if self._audit:
            self._audit('spectral.qc.export', path=path)
        messagebox.showinfo("Spectral QC", f"Exported:\n{path}", parent=self)

    def _export_png(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Export spectral QC figure",
            defaultextension='.png', initialfile='spectral_qc.png',
            filetypes=[('PNG image', '*.png'), ('PDF', '*.pdf'),
                       ('SVG', '*.svg')])
        if not path:
            return
        bg = self._bg_var.get()
        try:
            savefig_background(self._fig, path, background=bg, dpi=300)
        except Exception as exc:
            messagebox.showerror("Spectral QC",
                                 f"Export failed:\n{exc}", parent=self)
            return
        if self._audit:
            self._audit('spectral.qc.export', path=path, background=bg)
        messagebox.showinfo("Spectral QC", f"Exported:\n{path}", parent=self)


def plt_get_cmap(name):
    """matplotlib colormap copy (so per-window set_bad doesn't mutate the
    global registry entry). Uses the current ``matplotlib.colormaps`` API,
    falling back to the legacy ``cm.get_cmap`` on old matplotlib."""
    import matplotlib
    try:
        return matplotlib.colormaps[name].copy()
    except (AttributeError, KeyError):
        import matplotlib.cm as _cm
        return _cm.get_cmap(name).copy()


def savefig_background(fig, path, background='White', dpi=300):
    """Save ``fig`` with a publication-export background:

      • ``White``       — opaque white (default)
      • ``Transparent`` — full alpha (sits on a coloured page / poster / slide)
      • ``Translucent`` — 50% white wash (figure AND per-axes patches)

    The per-axes patch alpha is changed only for the duration of the save and
    restored afterwards, so an on-screen preview of ``fig`` is unaffected.
    PNG / PDF / SVG carry the alpha; TIFF may flatten it."""
    kw = {}
    axes_alpha = None
    if background == 'Transparent':
        kw['transparent'] = True
    elif background == 'Translucent':
        kw['facecolor'] = (1.0, 1.0, 1.0, 0.5)
        axes_alpha = 0.5
    else:                              # White
        kw['facecolor'] = 'white'
    restore = []
    if axes_alpha is not None:
        for ax in fig.axes:
            restore.append((ax, ax.patch.get_facecolor()))
            ax.patch.set_facecolor((1.0, 1.0, 1.0, axes_alpha))
    try:
        fig.savefig(path, dpi=dpi, bbox_inches='tight', edgecolor='none', **kw)
    finally:
        for ax, fc in restore:
            ax.patch.set_facecolor(fc)


class FrequencyComparisonWindow(tk.Toplevel):
    """Population-frequency & group-comparison view.

    Collects each loaded sample's per-population frequency (reusing the editor's
    ``_collect_stats_rows``), groups the samples by a chosen factor (trial/day,
    comp-vs-samples, or a name token like Stim), and for a selected population +
    metric draws a box/strip comparison with significance annotations plus an
    all-population overview. Exports tidy CSV, **GraphPad Prism**-ready Column
    and Grouped tables, a stats summary, and the figure."""

    METRICS = ('%Parent', '%Total', 'Count')
    FACTORS = ('Trial / day', 'Comp vs Samples', 'Name token')

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Population frequencies & group comparison")
        self.geometry("1120x760")
        self._rows = []
        self._tidy = None
        self._last_res = None

        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Population:").pack(side='left')
        self.pop_var = tk.StringVar()
        self.pop_combo = ttk.Combobox(ctl, textvariable=self.pop_var, width=26,
                                      state='readonly')
        self.pop_combo.pack(side='left', padx=(2, 8))
        ttk.Label(ctl, text="Metric:").pack(side='left')
        self.metric_var = tk.StringVar(value='%Parent')
        ttk.Combobox(ctl, textvariable=self.metric_var, width=8,
                     state='readonly', values=self.METRICS).pack(
            side='left', padx=(2, 8))
        ttk.Label(ctl, text="Group by:").pack(side='left')
        self.factor_var = tk.StringVar(value='Trial / day')
        ttk.Combobox(ctl, textvariable=self.factor_var, width=14,
                     state='readonly', values=self.FACTORS).pack(
            side='left', padx=(2, 4))
        ttk.Label(ctl, text="Tokens:").pack(side='left')
        self.tokens_var = tk.StringVar(value='Stim, Ctrl')
        ttk.Entry(ctl, textvariable=self.tokens_var, width=14).pack(
            side='left', padx=(2, 8))
        self.param_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctl, text="Parametric", variable=self.param_var).pack(
            side='left', padx=(0, 8))
        ttk.Button(ctl, text="Update", command=self._rebuild).pack(side='left')
        for w in (self.pop_combo,):
            w.bind('<<ComboboxSelected>>', lambda *_: self._rebuild())

        exp = ttk.Frame(self, padding=(6, 0))
        exp.pack(fill='x')
        ttk.Button(exp, text="Tidy CSV…",
                   command=self._export_tidy).pack(side='left')
        ttk.Button(exp, text="Prism Column…",
                   command=self._export_prism_column).pack(
            side='left', padx=(4, 0))
        ttk.Button(exp, text="Prism Grouped…",
                   command=self._export_prism_grouped).pack(
            side='left', padx=(4, 0))
        ttk.Button(exp, text="Stats summary…",
                   command=self._export_summary).pack(side='left', padx=(4, 0))
        ttk.Button(exp, text="Diff. abundance…",
                   command=self._diff_abundance).pack(side='left', padx=(4, 0))
        ttk.Button(exp, text="Compare all…",
                   command=self._compare_all).pack(side='left', padx=(4, 0))
        ttk.Button(exp, text="Figure…",
                   command=self._export_figure).pack(side='left', padx=(4, 0))
        self.bg_var = tk.StringVar(value='White')
        ttk.Combobox(exp, textvariable=self.bg_var, width=11, state='readonly',
                     values=['White', 'Transparent', 'Translucent']).pack(
            side='right')
        ttk.Label(exp, text="Fig background:").pack(side='right', padx=(0, 2))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(10.5, 4.8), dpi=100)
        cf = ttk.Frame(self)
        cf.pack(fill='both', expand=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=cf)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

        self._summary = tk.Text(self, height=6, wrap='word')
        self._summary.pack(fill='x', side='bottom')

        self._collect()
        pops = sorted({r['Population'] for r in self._rows})
        self.pop_combo['values'] = pops
        if pops:
            self.pop_var.set(pops[0])
        self._rebuild()

    # ── data ─────────────────────────────────────────────────────────────
    def _collect(self):
        try:
            rows, _ = self.editor._collect_stats_rows(
                {'Count', '%Parent', '%Total'})
        except Exception as exc:
            print(f"[frequencies] collect failed: {exc}", flush=True)
            rows = []
        self._rows = rows

    def _tokens(self):
        return [t for t in self.tokens_var.get().split(',') if t.strip()]

    def _tidy_frame(self):
        import pandas as pd
        factor = self.factor_var.get()
        tokens = self._tokens() if factor == 'Name token' else None
        recs = []
        for r in self._rows:
            nm = r['Sample']
            recs.append({
                'Sample': nm,
                'Group': self.editor._sample_group_label(nm, factor, tokens),
                'Population': r['Population'],
                'Count': r.get('Count'),
                '%Parent': r.get('%Parent'),
                '%Total': r.get('%Total')})
        return pd.DataFrame(recs)

    def _ordered_groups(self, tidy):
        """Groups in stable sample-load order (so day series stay chronological
        as loaded)."""
        order = {n: i for i, n in enumerate(self.editor._sample_order)}
        seen = {}
        for _, r in tidy.sort_values(
                'Sample', key=lambda s: s.map(lambda n: order.get(n, 1e9))
                ).iterrows():
            seen.setdefault(r['Group'], None)
        return list(seen)

    def _values_by_group(self, tidy, pop, metric):
        sub = tidy[tidy['Population'] == pop]
        vbg = {}
        for g in self._ordered_groups(tidy):
            vals = sub[sub['Group'] == g][metric].astype(float).tolist()
            vbg[g] = vals
        return vbg

    # ── rebuild + draw ───────────────────────────────────────────────────
    def _rebuild(self):
        from .stats import compare_groups
        self._tidy = self._tidy_frame()
        pop = self.pop_var.get()
        metric = self.metric_var.get()
        if not pop or self._tidy.empty:
            self._fig.clear()
            self._canvas.draw()
            return
        vbg = self._values_by_group(self._tidy, pop, metric)
        res = compare_groups(vbg, parametric=self.param_var.get())
        self._last_res = res
        self._draw(vbg, res, self._tidy, pop, metric)
        self._write_summary(res, pop, metric)

    def _draw(self, vbg, res, tidy, pop, metric):
        import numpy as _np
        fig = self._fig
        fig.clear()
        axA = fig.add_subplot(1, 2, 1)
        axB = fig.add_subplot(1, 2, 2)
        groups = list(vbg)
        data = [_np.asarray(vbg[g], float) for g in groups]
        data = [d[_np.isfinite(d)] for d in data]
        pos = list(range(1, len(groups) + 1))
        if any(len(d) for d in data):
            axA.boxplot(data, positions=pos, widths=0.55, showfliers=False)
            rng = _np.random.default_rng(0)
            for i, d in zip(pos, data, strict=True):
                if len(d):
                    jit = i + (rng.random(len(d)) - 0.5) * 0.16
                    axA.scatter(jit, d, s=16, alpha=0.75, color='#1f77b4',
                                zorder=3, linewidths=0)
        axA.set_xticks(pos)
        axA.set_xticklabels(groups, rotation=30, ha='right', fontsize=8)
        axA.set_ylabel(metric, fontsize=9)
        axA.set_title(pop, fontsize=9)
        self._draw_sig(axA, data, groups, res)

        # Panel B — all-population overview: mean metric per group (top pops).
        self._draw_overview(axB, tidy, metric, groups)
        try:
            fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw()

    def _draw_sig(self, ax, data, groups, res):
        from .stats import p_to_stars
        finite = [d for d in data if len(d)]
        if not finite:
            return
        ymax = max(float(d.max()) for d in finite)
        ymin = min(float(d.min()) for d in finite)
        span = (ymax - ymin) or (abs(ymax) or 1.0)
        h = span * 0.05
        base = ymax + span * 0.08
        idx = {g: i for i, g in enumerate(groups)}
        pairs = []
        if len(groups) == 2:
            s = p_to_stars(res.get('p'))
            if s:
                pairs = [(0, 1, s)]
        else:
            for pr in res.get('posthoc', []):
                s = p_to_stars(pr.get('p_adj'))
                if s and s != 'ns' and pr['a'] in idx and pr['b'] in idx:
                    pairs.append((idx[pr['a']], idx[pr['b']], s))
            pairs.sort(key=lambda t: abs(t[1] - t[0]))
            pairs = pairs[:6]
        for k, (i, j, s) in enumerate(pairs):
            y = base + k * h * 2.4
            x1, x2 = i + 1, j + 1
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0,
                    color='black')
            ax.text((x1 + x2) / 2.0, y + h, s, ha='center', va='bottom',
                    fontsize=9)
        if pairs:
            ax.set_ylim(top=base + len(pairs) * h * 2.4 + span * 0.12)

    def _draw_overview(self, ax, tidy, metric, groups):
        import numpy as _np
        means = (tidy.groupby(['Population', 'Group'])[metric]
                 .mean().reset_index())
        # Rank populations by overall mean; cap to keep the chart readable.
        overall = (means.groupby('Population')[metric].mean()
                   .sort_values(ascending=False))
        pops = list(overall.index[:10])
        capped = len(overall) > len(pops)
        x = _np.arange(len(pops))
        n = max(1, len(groups))
        w = 0.8 / n
        for gi, g in enumerate(groups):
            vals = []
            for p in pops:
                row = means[(means['Population'] == p) & (means['Group'] == g)]
                vals.append(float(row[metric].iloc[0]) if len(row) else 0.0)
            ax.bar(x + (gi - (n - 1) / 2.0) * w, vals, width=w, label=str(g))
        ax.set_xticks(x)
        ax.set_xticklabels([p.split('/')[-1] for p in pops], rotation=40,
                           ha='right', fontsize=7)
        ax.set_ylabel(f"mean {metric}", fontsize=9)
        ax.set_title("All populations" + (" (top 10)" if capped else ""),
                     fontsize=9)
        if len(groups) <= 8:
            ax.legend(fontsize=7, framealpha=0.85)

    def _write_summary(self, res, pop, metric):
        lines = [f"Population: {pop}    Metric: {metric}"]
        if res.get('test'):
            lines.append(f"Test: {res['test']}    p = {res.get('p'):.4g}")
        else:
            lines.append("Test: (need ≥2 non-empty groups)")
        for g, st in res.get('groups', {}).items():
            lines.append(f"  {g}: n={st['n']}  mean={st['mean']:.3g}  "
                         f"median={st['median']:.3g}  sd={st['sd']:.3g}")
        if res.get('posthoc'):
            lines.append("Pairwise (BH-adjusted):")
            from .stats import p_to_stars
            for pr in res['posthoc']:
                lines.append(f"  {pr['a']} vs {pr['b']}: "
                             f"p_adj={pr.get('p_adj'):.4g} "
                             f"{p_to_stars(pr.get('p_adj'))}")
        self._summary.configure(state='normal')
        self._summary.delete('1.0', 'end')
        self._summary.insert('1.0', "\n".join(lines))
        self._summary.configure(state='disabled')

    # ── exports ──────────────────────────────────────────────────────────
    def _ask(self, default, ftypes):
        return filedialog.asksaveasfilename(
            parent=self, defaultextension=os.path.splitext(default)[1],
            initialfile=default, filetypes=ftypes + [('All files', '*.*')])

    def _export_tidy(self):
        if self._tidy is None or self._tidy.empty:
            return
        path = self._ask('frequencies_tidy.csv', [('CSV', '*.csv')])
        if path:
            self._tidy.to_csv(path, index=False)
            self._done('frequencies.export', path, kind='tidy')

    def _export_prism_column(self):
        from .stats import to_prism_column
        pop, metric = self.pop_var.get(), self.metric_var.get()
        if self._tidy is None or not pop:
            return
        vbg = self._values_by_group(self._tidy, pop, metric)
        path = self._ask('prism_column.csv', [('CSV', '*.csv')])
        if path:
            to_prism_column(vbg).to_csv(path, index=False)
            self._done('frequencies.export', path, kind='prism_column')

    def _export_prism_grouped(self):
        from .stats import to_prism_grouped
        pop, metric = self.pop_var.get(), self.metric_var.get()
        if self._tidy is None or not pop:
            return
        tokens = self._tokens()
        if not tokens:
            messagebox.showinfo(
                "Prism Grouped",
                "Set comma-separated condition Tokens (e.g. 'Stim, Ctrl') — "
                "the Grouped table is Day (rows) × condition (columns).",
                parent=self)
            return
        sub = self._tidy[self._tidy['Population'] == pop].copy()
        sub['Day'] = [self.editor._sample_group_label(s, 'Trial / day')
                      for s in sub['Sample']]
        sub['Cond'] = [self.editor._sample_group_label(s, 'Name token', tokens)
                       for s in sub['Sample']]
        path = self._ask('prism_grouped.csv', [('CSV', '*.csv')])
        if path:
            to_prism_grouped(sub, 'Day', 'Cond', metric).to_csv(path)
            self._done('frequencies.export', path, kind='prism_grouped')

    def _export_summary(self):
        path = self._ask('frequencies_stats.md', [('Markdown', '*.md'),
                                                  ('Text', '*.txt')])
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self._summary.get('1.0', 'end'))
        self._done('frequencies.export', path, kind='summary')

    def _export_figure(self):
        path = self._ask('frequencies.png',
                         [('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if not path:
            return
        savefig_background(self._fig, path, background=self.bg_var.get())
        self._done('frequencies.export', path, kind='figure',
                   background=self.bg_var.get())

    def _done(self, action, path, **details):
        try:
            self.editor._audit(action, path=path, **details)
        except Exception:
            pass
        messagebox.showinfo("Frequencies", f"Exported:\n{path}", parent=self)

    def _diff_abundance(self):
        """Run the negative-binomial differential-abundance GLM over the
        populations between the two grouping levels, using each sample's total
        event count as the library-size offset, and show the results table."""
        from .diffexp import differential_abundance
        if self._tidy is None or self._tidy.empty:
            return
        tidy = self._tidy
        groups = self._ordered_groups(tidy)
        if len(groups) != 2:
            messagebox.showinfo(
                "Differential abundance",
                "Differential abundance needs exactly 2 groups — pick a "
                "Group-by / tokens that yield two (e.g. ctrl vs treat).",
                parent=self)
            return
        # counts: populations × samples (Count); group + library size per sample.
        wide = tidy.pivot_table(index='Population', columns='Sample',
                                values='Count', aggfunc='first', fill_value=0)
        samples = list(wide.columns)
        grp = [tidy.loc[tidy['Sample'] == s, 'Group'].iloc[0] for s in samples]
        col_sums = wide.to_numpy(dtype=float).sum(axis=0)   # per-sample totals
        lib = []
        for k, s in enumerate(samples):
            csum = int(col_sums[k])
            ev = (len(self.editor._samples[s].data)
                  if s in self.editor._samples else csum)
            lib.append(max(ev, csum, 1))
        try:
            rows = differential_abundance(wide, grp, lib_sizes=lib)
        except Exception as exc:
            messagebox.showerror("Differential abundance",
                                 f"Failed: {exc}", parent=self)
            return
        try:
            self.editor._audit('diff_abundance', n_populations=len(rows),
                               group_a=groups[0], group_b=groups[1])
        except Exception:
            pass
        DiffAbundanceWindow(self, rows, groups)

    def _compare_all(self):
        """Compare EVERY population across the current grouping in one pass
        (BH-corrected across populations) and open a results table + volcano —
        instead of stepping through populations one at a time."""
        from .stats import compare_all_features
        if self._tidy is None or self._tidy.empty:
            return
        tidy = self._tidy
        metric = self.metric_var.get()
        groups = self._ordered_groups(tidy)
        if len(groups) < 2:
            messagebox.showinfo(
                "Compare all populations",
                "Need at least 2 groups — pick a Group-by / tokens that yield "
                "two or more (e.g. Stim vs Ctrl).", parent=self)
            return
        pops = sorted({r['Population'] for r in self._rows})
        vbf = {p: self._values_by_group(tidy, p, metric) for p in pops}
        res = compare_all_features(vbf, parametric=self.param_var.get())
        try:
            self.editor._audit('compare_all_populations', metric=metric,
                               n_populations=len(res), groups=','.join(groups))
        except Exception:
            pass
        CompareAllWindow(self, res, groups, metric)


class CompareAllWindow(tk.Toplevel):
    """All-population group comparison: a sortable results table (per-group
    means, log2 fold-change, BH-adjusted p, stars) beside a volcano plot
    (log2FC vs −log10 adjusted-p, significant populations highlighted). One
    click compares every population at once; export the full table or the
    volcano figure. The volcano needs the two-group case (log2FC); with >2
    groups the table still shows the omnibus Kruskal-Wallis / ANOVA result."""

    def __init__(self, parent, results, groups, metric):
        super().__init__(parent)
        self.title("Compare all populations")
        self.geometry("1040x620")
        self._results = results
        self._groups = groups
        self._metric = metric
        self._two = len(groups) == 2
        from .stats import volcano_data
        self._volcano = volcano_data(results)

        ttk.Label(self, padding=6, font=('TkDefaultFont', 9, 'bold'),
                  text=(f"{metric} across {len(groups)} groups "
                        f"({', '.join(groups)}) — {len(results)} populations, "
                        f"BH-adjusted.")).pack(fill='x', side='top')
        bar = ttk.Frame(self)
        bar.pack(fill='x')
        ttk.Button(bar, text="Results CSV…", command=self._export_csv).pack(
            side='right', padx=6, pady=4)
        ttk.Button(bar, text="Volcano figure…",
                   command=self._export_figure).pack(side='right', pady=4)

        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True)

        # ── left: results table ──
        tbl = ttk.Frame(paned)
        paned.add(tbl, weight=1)
        if self._two:
            cols = ('pop', 'a', 'b', 'log2fc', 'p', 'padj', 'sig')
            heads = {'pop': 'Population', 'a': 'mean ' + groups[0],
                     'b': 'mean ' + groups[1], 'log2fc': 'log2FC',
                     'p': 'p', 'padj': 'p(adj)', 'sig': ''}
            widths = (210, 90, 90, 70, 70, 70, 36)
        else:
            cols = ('pop', 'p', 'padj', 'sig')
            heads = {'pop': 'Population', 'p': 'p (omnibus)',
                     'padj': 'p(adj)', 'sig': ''}
            widths = (320, 100, 90, 40)
        tv = ttk.Treeview(tbl, columns=cols, show='headings')
        for c, w in zip(cols, widths, strict=True):
            tv.heading(c, text=heads[c])
            tv.column(c, width=w, anchor='w', stretch=(c == 'pop'))
        tv.pack(fill='both', expand=True)

        def _f(x, fmt):
            return fmt.format(x) if x is not None and x == x else 'n/a'
        for r in results:
            g = r['groups']
            if self._two:
                ma = g.get(groups[0], {}).get('mean')
                mb = g.get(groups[1], {}).get('mean')
                tv.insert('', 'end', values=(
                    r['feature'], _f(ma, '{:.3g}'), _f(mb, '{:.3g}'),
                    _f(r['effect'], '{:+.2f}'), _f(r['p'], '{:.2g}'),
                    _f(r['p_adj'], '{:.2g}'), r['stars']))
            else:
                tv.insert('', 'end', values=(
                    r['feature'], _f(r['p'], '{:.2g}'),
                    _f(r['p_adj'], '{:.2g}'), r['stars']))

        # ── right: volcano ──
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self._fig = Figure(figsize=(5.2, 4.8), dpi=100)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        self._draw_volcano()

    def _draw_volcano(self):
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        if not self._two or not self._volcano:
            ax.text(0.5, 0.5, "Volcano needs exactly 2 groups\n"
                    "(log2 fold-change).", ha='center', va='center',
                    fontsize=9, color='#666', transform=ax.transAxes)
            ax.set_axis_off()
            self._canvas.draw()
            return
        import numpy as _np
        xs = _np.array([p['x'] for p in self._volcano])
        ys = _np.array([p['y'] for p in self._volcano])
        sig = _np.array([p['significant'] for p in self._volcano])
        ax.scatter(xs[~sig], ys[~sig], s=18, c='#bbb', linewidths=0,
                   label='ns')
        ax.scatter(xs[sig], ys[sig], s=22, c='#d62728', linewidths=0,
                   label='significant')
        ax.axhline(-_np.log10(0.05), color='#888', ls='--', lw=.7)
        for xc in (-1.0, 1.0):
            ax.axvline(xc, color='#888', ls=':', lw=.7)
        # label the most significant populations
        for p in sorted(self._volcano, key=lambda d: -d['y'])[:6]:
            if p['significant']:
                ax.annotate(p['feature'], (p['x'], p['y']), fontsize=7,
                            xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel(f"log2 fold-change ({self._groups[1]} / {self._groups[0]})",
                      fontsize=9)
        ax.set_ylabel("−log10 adjusted p", fontsize=9)
        ax.set_title("Volcano", fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        try:
            fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw()

    def _flat_rows(self):
        out = []
        for r in self._results:
            row = {'population': r['feature'], 'test': r['test'],
                   'log2FC': r['effect'], 'p': r['p'], 'p_adj': r['p_adj'],
                   'sig': r['stars']}
            for gname, gs in r['groups'].items():
                row[f'mean_{gname}'] = gs.get('mean')
                row[f'n_{gname}'] = gs.get('n')
            out.append(row)
        return out

    def _export_csv(self):
        import pandas as pd
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            initialfile='compare_all_populations.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            pd.DataFrame(self._flat_rows()).to_csv(path, index=False)
            messagebox.showinfo("Compare all populations",
                                f"Exported:\n{path}", parent=self)

    def _export_figure(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.png', initialfile='volcano.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg'),
                       ('All files', '*.*')])
        if path:
            self._fig.savefig(path, dpi=200, bbox_inches='tight')
            messagebox.showinfo("Compare all populations",
                                f"Saved:\n{path}", parent=self)


class DiffAbundanceWindow(tk.Toplevel):
    """Results table for the negative-binomial differential-abundance test
    (log2FC of group-B vs group-A proportion per population, with adjusted
    p-values and significance stars), with CSV export."""

    def __init__(self, parent, rows, groups):
        super().__init__(parent)
        self.title("Differential abundance (NB-GLM)")
        self.geometry("780x460")
        self._rows = rows
        from .stats import p_to_stars
        ttk.Label(self, padding=6,
                  text=(f"{groups[1]} vs {groups[0]} — negative-binomial GLM on "
                        f"counts (library-size offset). {len(rows)} populations."),
                  font=('TkDefaultFont', 9, 'bold')).pack(fill='x', side='top')
        bar = ttk.Frame(self)
        bar.pack(fill='x')
        ttk.Button(bar, text="Export CSV…", command=self._export).pack(
            side='right', padx=6, pady=4)
        cols = ('pop', 'log2fc', 'pa', 'pb', 'p', 'padj', 'sig')
        heads = {'pop': 'Population', 'log2fc': 'log2FC', 'pa': '%' + groups[0],
                 'pb': '%' + groups[1], 'p': 'p', 'padj': 'p(adj)',
                 'sig': ''}
        widths = (300, 70, 70, 70, 80, 80, 40)
        tv = ttk.Treeview(self, columns=cols, show='headings')
        for c, w in zip(cols, widths, strict=True):
            tv.heading(c, text=heads[c])
            tv.column(c, width=w, anchor='w', stretch=(c == 'pop'))
        tv.pack(fill='both', expand=True)
        for r in rows:
            tv.insert('', 'end', values=(
                r['cluster'], f"{r['log2fc']:+.2f}",
                f"{r['prop_a'] * 100:.2f}", f"{r['prop_b'] * 100:.2f}",
                f"{r['p']:.2g}" if r['p'] == r['p'] else 'n/a',
                f"{r['p_adj']:.2g}" if r['p_adj'] == r['p_adj'] else 'n/a',
                p_to_stars(r['p_adj'])))

    def _export(self):
        import pandas as pd
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            initialfile='differential_abundance.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            pd.DataFrame(self._rows).to_csv(path, index=False)
            messagebox.showinfo("Differential abundance",
                                f"Exported:\n{path}", parent=self)


class MarkerExpressionWindow(tk.Toplevel):
    """Marker-expression distributions by group — violin or ridgeline.

    Pools each enabled sample's per-cell values for a chosen marker (resolving
    the marker across fluors by antibody label), groups the samples by a factor
    (trial/day, comp-vs-samples, or a name token), and draws a violin or
    ridgeline plot per group. Significance comes from a per-SAMPLE-median
    comparison (so the test treats each sample as a replicate, not each cell),
    which is also what the **GraphPad Prism** Column export contains."""

    FACTORS = ('Trial / day', 'Comp vs Samples', 'Name token')
    PER_GROUP_CAP = 40_000

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Marker expression by group")
        self.geometry("1000x720")
        self._percell = {}
        self._medians = {}
        self._groups = []

        chans = [c for c in editor._channels]
        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Marker:").pack(side='left')
        self.marker_var = tk.StringVar()
        disp = [editor._fmt_channel(c) for c in chans]
        ttk.Combobox(ctl, textvariable=self.marker_var, width=20,
                     state='readonly', values=disp).pack(side='left', padx=(2, 8))
        fluor = next((editor._fmt_channel(c) for c in chans
                      if c in (self._first_fluor() or [])), disp[0] if disp
                     else '')
        self.marker_var.set(fluor)
        ttk.Label(ctl, text="Group by:").pack(side='left')
        self.factor_var = tk.StringVar(value='Trial / day')
        ttk.Combobox(ctl, textvariable=self.factor_var, width=14,
                     state='readonly', values=self.FACTORS).pack(
            side='left', padx=(2, 4))
        ttk.Label(ctl, text="Tokens:").pack(side='left')
        self.tokens_var = tk.StringVar(value='Stim, Ctrl')
        ttk.Entry(ctl, textvariable=self.tokens_var, width=14).pack(
            side='left', padx=(2, 8))
        ttk.Label(ctl, text="Plot:").pack(side='left')
        self.plot_var = tk.StringVar(value='Violin')
        ttk.Combobox(ctl, textvariable=self.plot_var, width=9, state='readonly',
                     values=['Violin', 'Ridgeline']).pack(side='left', padx=(2, 8))
        self.param_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctl, text="Parametric",
                        variable=self.param_var).pack(side='left', padx=(0, 8))
        ttk.Button(ctl, text="Update", command=self._rebuild).pack(side='left')

        exp = ttk.Frame(self, padding=(6, 0))
        exp.pack(fill='x')
        ttk.Button(exp, text="Prism Column (medians)…",
                   command=self._export_prism).pack(side='left')
        ttk.Button(exp, text="Stats summary…",
                   command=self._export_summary).pack(side='left', padx=(4, 0))
        ttk.Button(exp, text="Figure…",
                   command=self._export_figure).pack(side='left', padx=(4, 0))
        self.bg_var = tk.StringVar(value='White')
        ttk.Combobox(exp, textvariable=self.bg_var, width=11, state='readonly',
                     values=['White', 'Transparent', 'Translucent']).pack(
            side='right')
        ttk.Label(exp, text="Fig background:").pack(side='right', padx=(0, 2))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(9.5, 5.0), dpi=100)
        cf = ttk.Frame(self)
        cf.pack(fill='both', expand=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=cf)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        self._summary = tk.Text(self, height=5, wrap='word')
        self._summary.pack(fill='x', side='bottom')
        self._rebuild()

    def _first_fluor(self):
        for n in self.editor._sample_order:
            s = self.editor._samples.get(n)
            if s is not None:
                return getattr(s, 'fluor_channels', []) or []
        return []

    def _samples(self):
        names = self.editor._selected_samples() or (
            [self.editor._active_sample] if self.editor._active_sample else [])
        return [n for n in names if n in self.editor._samples]

    def _tokens(self):
        return [t for t in self.tokens_var.get().split(',') if t.strip()]

    def _collect(self):
        import numpy as _np
        ch = self.editor._resolve_channel(self.marker_var.get())
        factor = self.factor_var.get()
        tokens = self._tokens() if factor == 'Name token' else None
        order = {n: i for i, n in enumerate(self.editor._sample_order)}
        percell, medians, groups = {}, {}, []
        for n in sorted(self._samples(), key=lambda x: order.get(x, 1e9)):
            s = self.editor._samples[n]
            col = self.editor._marker_column_for(s, ch)
            if not col:
                continue
            vals = _np.asarray(s.data[col].values, dtype=float)
            vals = vals[_np.isfinite(vals)]
            if vals.size == 0:
                continue
            g = self.editor._sample_group_label(n, factor, tokens)
            if g not in percell:
                percell[g] = []
                medians[g] = []
                groups.append(g)
            percell[g].append(vals)
            medians[g].append(float(_np.median(vals)))
        rng = _np.random.default_rng(0)
        pooled = {}
        for g in groups:
            allv = _np.concatenate(percell[g])
            if allv.size > self.PER_GROUP_CAP:
                allv = allv[rng.choice(allv.size, self.PER_GROUP_CAP,
                                       replace=False)]
            pooled[g] = allv
        self._percell, self._medians, self._groups = pooled, medians, groups

    def _rebuild(self):
        from .stats import compare_groups
        self._collect()
        if not self._groups:
            self._fig.clear()
            self._canvas.draw()
            return
        res = compare_groups(self._medians, parametric=self.param_var.get())
        self._draw(res)
        self._write_summary(res)

    def _draw(self, res):
        import numpy as _np
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        groups = self._groups
        marker = self.marker_var.get()
        if self.plot_var.get() == 'Ridgeline':
            self._draw_ridgeline(ax, groups, marker)
        else:
            data = [self._percell[g] for g in groups]
            pos = list(range(1, len(groups) + 1))
            parts = ax.violinplot(data, positions=pos, showmedians=True,
                                  widths=0.8)
            # parts['bodies'] is a list of PolyCollection at runtime; the
            # matplotlib stub types it as a non-iterable Collection.
            bodies: list = list(parts.get('bodies') or [])  # type: ignore
            for b in bodies:
                b.set_alpha(0.6)
            ax.set_xticks(pos)
            ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=8)
            ax.set_ylabel(marker, fontsize=9)
            self._draw_sig(ax, [_np.asarray(self._medians[g]) for g in groups],
                           groups, res)
        ax.set_title(f"{marker} by {self.factor_var.get()}", fontsize=9)
        try:
            fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw()

    def _draw_ridgeline(self, ax, groups, marker):
        from .stats import group_kde
        x, dens = group_kde(self._percell)
        if x.size == 0:
            return
        peak = max((d.max() for d in dens.values() if d.size), default=1.0) or 1.0
        step = 0.8
        for i, g in enumerate(groups):
            d = dens.get(g)
            if d is None:
                continue
            base = i * step
            y = base + d / peak * step * 1.6
            ax.fill_between(x, base, y, alpha=0.7, zorder=len(groups) - i)
            ax.plot(x, y, lw=0.8, color='black', alpha=0.5)
        ax.set_yticks([i * step for i in range(len(groups))])
        ax.set_yticklabels(groups, fontsize=8)
        ax.set_xlabel(marker, fontsize=9)

    def _draw_sig(self, ax, medians, groups, res):
        from .stats import p_to_stars
        finite = [m[np.isfinite(m)] for m in medians]
        finite = [m for m in finite if len(m)]
        if not finite:
            return
        ymax = max(float(m.max()) for m in finite)
        span = (ymax - min(float(m.min()) for m in finite)) or (abs(ymax) or 1.0)
        h = span * 0.05
        base = ymax + span * 0.10
        idx = {g: i for i, g in enumerate(groups)}
        pairs = []
        if len(groups) == 2:
            s = p_to_stars(res.get('p'))
            if s:
                pairs = [(0, 1, s)]
        else:
            for pr in res.get('posthoc', []):
                s = p_to_stars(pr.get('p_adj'))
                if s and s != 'ns' and pr['a'] in idx and pr['b'] in idx:
                    pairs.append((idx[pr['a']], idx[pr['b']], s))
            pairs.sort(key=lambda t: abs(t[1] - t[0]))
            pairs = pairs[:6]
        for k, (i, j, s) in enumerate(pairs):
            y = base + k * h * 2.4
            x1, x2 = i + 1, j + 1
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0,
                    color='black')
            ax.text((x1 + x2) / 2.0, y + h, s, ha='center', va='bottom',
                    fontsize=9)

    def _write_summary(self, res):
        lines = [f"Marker: {self.marker_var.get()}    "
                 f"(significance from per-sample medians, n = samples)"]
        if res.get('test'):
            lines.append(f"Test: {res['test']}    p = {res.get('p'):.4g}")
        else:
            lines.append("Test: (need ≥2 groups with samples)")
        for g, st in res.get('groups', {}).items():
            lines.append(f"  {g}: n={st['n']}  median-of-medians="
                         f"{st['median']:.3g}")
        self._summary.configure(state='normal')
        self._summary.delete('1.0', 'end')
        self._summary.insert('1.0', "\n".join(lines))
        self._summary.configure(state='disabled')

    def _export_prism(self):
        from .stats import to_prism_column
        if not self._medians:
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            initialfile='expression_medians_prism.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            to_prism_column(self._medians).to_csv(path, index=False)
            self._done(path, 'prism_column')

    def _export_summary(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.md',
            initialfile='expression_stats.md',
            filetypes=[('Markdown', '*.md'), ('Text', '*.txt')])
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self._summary.get('1.0', 'end'))
            self._done(path, 'summary')

    def _export_figure(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.png', initialfile='expression.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if path:
            savefig_background(self._fig, path, background=self.bg_var.get())
            self._done(path, 'figure')

    def _done(self, path, kind):
        try:
            self.editor._audit('expression.export', path=path, kind=kind,
                               marker=self.marker_var.get())
        except Exception:
            pass
        messagebox.showinfo("Expression", f"Exported:\n{path}", parent=self)


class PopulationAnnotationWindow(tk.Toplevel):
    """Annotate clustered populations by phenotype.

    Computes MEM (Marker Enrichment Modeling) labels for each cluster of the
    active sample's chosen label column, and — given a reference
    ``name: CD3+ CD4+ CD8-`` table — assigns a best-matching cell-type name,
    writing it back onto the populations (and the cluster-label store). Exports
    the MEM table."""

    _DEFAULT_TABLE = (
        "# name: marker+ marker-  (one cell type per line)\n"
        "CD4 T: CD3+ CD4+ CD8-\n"
        "CD8 T: CD3+ CD8+ CD4-\n"
        "B cell: CD3- CD19+\n"
        "NK cell: CD3- CD56+\n"
        "Monocyte: CD14+ CD3-\n")

    def __init__(self, editor, sample):
        super().__init__(editor)
        self.editor = editor
        self.sample = sample
        self.title(f"Annotate populations — {sample}")
        self.geometry("860x620")
        self._mem = None
        self._label_col = None

        s = editor._samples[sample]
        cols = [c for c in ('leiden', 'cluster', 'flowsom_meta')
                if c in s.data.columns]
        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Cluster column:").pack(side='left')
        self.col_var = tk.StringVar(value=cols[0] if cols else '')
        ttk.Combobox(ctl, textvariable=self.col_var, width=14, state='readonly',
                     values=cols).pack(side='left', padx=(2, 8))
        ttk.Label(ctl, text="MEM threshold:").pack(side='left')
        self.thr_var = tk.StringVar(value='2')
        ttk.Spinbox(ctl, from_=0, to=10, width=5,
                    textvariable=self.thr_var).pack(side='left', padx=(2, 8))
        ttk.Button(ctl, text="Compute MEM", command=self._compute).pack(
            side='left')
        ttk.Button(ctl, text="Export MEM CSV…", command=self._export).pack(
            side='right')

        cols2 = ('pop', 'n', 'mem', 'name')
        tv = ttk.Treeview(self, columns=cols2, show='headings', height=12)
        for c, w in zip(cols2, (70, 80, 430, 130), strict=True):
            tv.heading(c, text={'pop': 'Cluster', 'n': 'Events',
                                'mem': 'MEM label', 'name': 'Assigned'}[c])
            tv.column(c, width=w, anchor='w', stretch=(c == 'mem'))
        tv.pack(fill='both', expand=True, padx=6)
        self._tv = tv

        ref = ttk.LabelFrame(self, text="Reference cell-type table "
                             "(name: CD3+ CD4+ CD8-)", padding=6)
        ref.pack(fill='x', padx=6, pady=(4, 6))
        self.ref_txt = tk.Text(ref, height=6, wrap='none')
        self.ref_txt.insert('1.0', self._DEFAULT_TABLE)
        self.ref_txt.pack(fill='x', side='top')
        ttk.Button(ref, text="Assign names → populations",
                   command=self._apply).pack(side='left', pady=(4, 0))
        self.status = ttk.Label(ref, text="", foreground='#555')
        self.status.pack(side='left', padx=(10, 0), pady=(4, 0))

        if cols:
            self._compute()

    def _marker_cols(self):
        s = self.editor._samples[self.sample]
        return [c for c in getattr(s, 'fluor_channels', [])
                if c in s.data.columns]

    def _label_of(self, det):
        s = self.editor._samples[self.sample]
        return (getattr(s, 'channel_labels', {}) or {}).get(
            det, self.editor._channel_labels.get(det, det))

    def _compute(self):
        from .annotate import mem_label, mem_scores
        col = self.col_var.get()
        markers = self._marker_cols()
        if not col or not markers:
            return
        s = self.editor._samples[self.sample]
        labels = s.data[col].to_numpy()
        valid = labels >= 0
        mem = mem_scores(s.data.loc[valid, markers], labels[valid], markers)
        # Relabel detector columns to antibody markers for readability + the
        # reference table (which is written in CD names).
        mem = mem.rename(columns={d: self._label_of(d) for d in markers})
        self._mem = mem
        self._label_col = col
        try:
            thr = float(self.thr_var.get())
        except ValueError:
            thr = 2.0
        uniq, cnts = np.unique(labels[valid], return_counts=True)
        counts = {int(u): int(c) for u, c in zip(uniq, cnts, strict=True)}
        self._tv.delete(*self._tv.get_children())
        for pop, row in mem.iterrows():
            pid = int(str(pop))
            self._tv.insert('', 'end', iid=str(pid),
                            values=(pid, counts.get(pid, 0),
                                    mem_label(row, threshold=thr), ''))
        self.status.configure(text=f"MEM computed for {len(mem)} clusters.")

    def _apply(self):
        from .annotate import (
            annotate_by_reference,
            parse_signature_table,
            population_states,
        )
        if self._mem is None:
            return
        table = parse_signature_table(self.ref_txt.get('1.0', 'end'))
        if not table:
            self.status.configure(text="No valid reference rows parsed.")
            return
        try:
            thr = max(2.0, float(self.thr_var.get()))
        except ValueError:
            thr = 3.0
        states = population_states(self._mem, threshold=thr)
        ann = annotate_by_reference(states, table)
        names = {int(str(p)): ann[p]['name'] for p in ann
                 if ann[p]['name'] != 'unknown'}
        for iid in self._tv.get_children():
            pop = int(iid)
            self._tv.set(iid, 'name', names.get(pop, 'unknown'))
        if names:
            self.editor._apply_population_names(self.sample, self._label_col,
                                                names)
            self.editor._audit('annotate', sample=self.sample,
                                column=self._label_col, n_named=len(names))
        self.status.configure(
            text=f"Named {len(names)} of {len(ann)} clusters.")

    def _export(self):
        if self._mem is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv', initialfile='mem_scores.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            self._mem.to_csv(path)
            self.editor._audit('annotate.export', path=path)
            messagebox.showinfo("Annotate", f"Exported:\n{path}", parent=self)


class TrajectoryWindow(tk.Toplevel):
    """Pseudotime / trajectory inference.

    Builds a geodesic pseudotime over the enabled samples' shared fluor
    channels (concatenated, so a day-series becomes one continuous trajectory),
    rooted at the extreme of a chosen marker (e.g. CD34-high = most primitive),
    writes a ``pseudotime`` column back to every sample (selectable as a plot
    colour), and draws each marker's mean expression along pseudotime — the
    CD34-down / CD11b-up maturation curve. Exports the trends as a CSV / Prism XY
    table and the figure."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Trajectory / pseudotime")
        self.geometry("960x680")
        self._centers = None
        self._means = None
        self._channels = []

        chans = self._shared_channels()
        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Root marker:").pack(side='left')
        self.root_var = tk.StringVar()
        disp = [editor._fmt_channel(c) for c in chans]
        self.root_combo = ttk.Combobox(ctl, textvariable=self.root_var,
                                       width=22, state='readonly', values=disp)
        self.root_combo.pack(side='left', padx=(2, 8))
        # Default to a stemness-ish marker if present (CD34), else first.
        default = next((editor._fmt_channel(c) for c in chans
                        if 'cd34' in (editor._channel_labels.get(c, c)).lower()),
                       disp[0] if disp else '')
        self.root_var.set(default)
        ttk.Label(ctl, text="Root end:").pack(side='left')
        self.dir_var = tk.StringVar(value='High')
        ttk.Combobox(ctl, textvariable=self.dir_var, width=6, state='readonly',
                     values=['High', 'Low']).pack(side='left', padx=(2, 8))
        ttk.Label(ctl, text="Neighbors:").pack(side='left')
        self.k_var = tk.StringVar(value='15')
        ttk.Spinbox(ctl, from_=5, to=50, width=5,
                    textvariable=self.k_var).pack(side='left', padx=(2, 8))
        ttk.Button(ctl, text="Compute", command=self._compute).pack(side='left')
        self.status = ttk.Label(ctl, text="", foreground='#555')
        self.status.pack(side='left', padx=(8, 0))

        exp = ttk.Frame(self, padding=(6, 0))
        exp.pack(fill='x')
        ttk.Button(exp, text="Trends CSV…",
                   command=lambda: self._export('tidy')).pack(side='left')
        ttk.Button(exp, text="Prism XY…",
                   command=lambda: self._export('prism')).pack(
            side='left', padx=(4, 0))
        ttk.Button(exp, text="Figure…",
                   command=lambda: self._export('figure')).pack(
            side='left', padx=(4, 0))
        self.bg_var = tk.StringVar(value='White')
        ttk.Combobox(exp, textvariable=self.bg_var, width=11, state='readonly',
                     values=['White', 'Transparent', 'Translucent']).pack(
            side='right')
        ttk.Label(exp, text="Fig background:").pack(side='right', padx=(0, 2))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(9, 4.8), dpi=100)
        cf = ttk.Frame(self)
        cf.pack(fill='both', expand=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=cf)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        if not chans:
            self.status.configure(
                text="No shared fluor channels across the enabled samples.")

    def _samples(self):
        names = self.editor._selected_samples() or (
            [self.editor._active_sample] if self.editor._active_sample else [])
        return [n for n in names if n in self.editor._samples]

    def _shared_channels(self):
        shared = None
        for n in self._samples():
            s = self.editor._samples[n]
            cols = set(getattr(s, 'fluor_channels', []) or []) & set(s.data.columns)
            shared = cols if shared is None else (shared & cols)
        # Preserve the first sample's channel order.
        if not shared:
            return []
        first = self.editor._samples[self._samples()[0]]
        return [c for c in first.fluor_channels if c in shared]

    def _compute(self):
        from .trajectory import compute_pseudotime, pseudotime_trends
        names = self._samples()
        chans = self._shared_channels()
        root = self.editor._resolve_channel(self.root_var.get())
        if not names or not chans or root not in chans:
            self.status.configure(text="Need enabled samples + a root marker.")
            return
        self.status.configure(text="Computing pseudotime…")
        self.update_idletasks()
        self.configure(cursor='watch')
        try:
            import numpy as _np
            mats, bounds, pos = [], [], 0
            for n in names:
                df = self.editor._samples[n].data
                m = df[chans].to_numpy(dtype=float)
                mats.append(m)
                bounds.append((n, pos, pos + len(m)))
                pos += len(m)
            X = _np.vstack(mats) if mats else _np.empty((0, len(chans)))
            score = X[:, chans.index(root)]
            try:
                k = int(self.k_var.get())
            except ValueError:
                k = 15
            pt, _ = compute_pseudotime(X, score, high=(self.dir_var.get() ==
                                       'High'), n_neighbors=k)
            # Write the pseudotime column back to each sample.
            for n, a, b in bounds:
                self.editor._samples[n].data['pseudotime'] = pt[a:b]
            self._centers, self._means = pseudotime_trends(pt, X, n_bins=20)
            self._channels = chans
        except Exception as exc:
            self.status.configure(text=f"Failed: {type(exc).__name__}: {exc}")
            self.configure(cursor='')
            return
        self.configure(cursor='')
        self.status.configure(
            text=f"Done — {len(X):,} cells across {len(names)} sample(s). "
                 "'pseudotime' is now a plot colour.")
        self.editor._refresh_channel_choices()
        self.editor._audit('trajectory', samples=names,
                           root=root, root_end=self.dir_var.get(),
                           n_neighbors=k, n_cells=int(len(X)))
        self._draw()

    def _draw(self):
        import numpy as _np
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        means, centers = self._means, self._centers
        if means is not None and centers is not None:
            for j, ch in enumerate(self._channels):
                col = means[:, j]
                finite = _np.isfinite(col)
                if not finite.any():
                    continue
                lo, hi = _np.nanmin(col), _np.nanmax(col)
                norm = (col - lo) / (hi - lo) if hi > lo else col * 0
                ax.plot(centers[finite], norm[finite], marker='o', ms=3,
                        lw=1.4, label=self.editor._fmt_channel(ch))
            ax.set_xlabel('pseudotime')
            ax.set_ylabel('expression (per-marker min–max normalized)')
            ax.set_title('Marker trends along pseudotime')
            if len(self._channels) <= 12:
                ax.legend(fontsize=7, framealpha=0.85, loc='best')
        try:
            fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw()

    def _trends_frame(self):
        import pandas as pd
        means = self._means
        if means is None or self._centers is None:
            return pd.DataFrame()
        data = {'pseudotime': self._centers}
        for j, ch in enumerate(self._channels):
            data[self.editor._fmt_channel(ch)] = means[:, j]
        return pd.DataFrame(data)

    def _export(self, kind):
        if self._means is None:
            messagebox.showinfo("Trajectory", "Compute a trajectory first.",
                                parent=self)
            return
        if kind == 'figure':
            path = filedialog.asksaveasfilename(
                parent=self, defaultextension='.png', initialfile='trajectory.png',
                filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
            if path:
                savefig_background(self._fig, path, background=self.bg_var.get())
                self._done(path, 'figure')
            return
        # 'tidy' and 'prism' are the same shape here (a Prism XY table: the bin
        # centre column + one mean column per marker) — both paste into Prism XY.
        name = ('prism_xy.csv' if kind == 'prism'
                else 'trajectory_trends.csv')
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv', initialfile=name,
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            self._trends_frame().to_csv(path, index=False)
            self._done(path, kind)

    def _done(self, path, kind):
        try:
            self.editor._audit('trajectory.export', path=path, kind=kind)
        except Exception:
            pass
        messagebox.showinfo("Trajectory", f"Exported:\n{path}", parent=self)


class CalibrationDialog(tk.Toplevel):
    """Fluorescence-intensity calibration to standardized units (MESF / ABC).

    Detect the bead peaks in a channel, paste each peak's assigned value from
    the bead datasheet, fit ``value = slope·MFI + intercept``, then apply it to
    a channel across all samples as a ``MESF:<marker>`` column."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Fluorescence calibration (MESF / ABC)")
        self.geometry("560x540")
        self._cal = None
        body = ttk.Frame(self, padding=10)
        body.pack(fill='both', expand=True)

        r1 = ttk.Frame(body)
        r1.pack(fill='x')
        ttk.Label(r1, text="Bead sample:").pack(side='left')
        self.bead_var = tk.StringVar(
            value=editor._active_sample or (editor._sample_order[0]
                                            if editor._sample_order else ''))
        bead_cb = ttk.Combobox(r1, textvariable=self.bead_var, width=20,
                               state='readonly', values=editor._sample_order)
        bead_cb.pack(side='left', padx=(2, 8))
        bead_cb.bind('<<ComboboxSelected>>', lambda *_: self._sync_channels())
        ttk.Label(r1, text="Channel:").pack(side='left')
        self.chan_var = tk.StringVar()
        self.chan_cb = ttk.Combobox(r1, textvariable=self.chan_var, width=16,
                                    state='readonly')
        self.chan_cb.pack(side='left', padx=(2, 8))
        ttk.Label(r1, text="Peaks:").pack(side='left')
        self.npk_var = tk.StringVar(value='6')
        ttk.Spinbox(r1, from_=2, to=12, width=4,
                    textvariable=self.npk_var).pack(side='left', padx=(2, 8))
        ttk.Button(r1, text="Detect peaks",
                   command=self._detect).pack(side='left')

        ttk.Label(body, justify='left',
                  text="Peaks — one per line as  MFI <tab/comma> assigned "
                       "value (MESF/ABC from the bead lot):").pack(
            anchor='w', pady=(8, 2))
        self.txt = tk.Text(body, height=9, wrap='none')
        self.txt.pack(fill='x')

        r2 = ttk.Frame(body)
        r2.pack(fill='x', pady=(6, 0))
        ttk.Button(r2, text="Fit", command=self._fit).pack(side='left')
        self.result = ttk.Label(r2, text="", foreground='#333')
        self.result.pack(side='left', padx=(10, 0))

        r3 = ttk.Frame(body)
        r3.pack(fill='x', pady=(10, 0))
        ttk.Label(r3, text="Apply to channel:").pack(side='left')
        self.apply_var = tk.StringVar()
        self.apply_cb = ttk.Combobox(r3, textvariable=self.apply_var, width=16,
                                     state='readonly')
        self.apply_cb.pack(side='left', padx=(2, 8))
        ttk.Button(r3, text="Apply calibration → MESF: column",
                   command=self._apply).pack(side='left')

        self._sync_channels()
        try:
            self.grab_set()
        except Exception:
            pass

    def _fluor_channels(self):
        s = self.editor._samples.get(self.bead_var.get())
        if s is None:
            return []
        return [c for c in getattr(s, 'fluor_channels', [])
                if c in s.data.columns]

    def _sync_channels(self):
        chans = self._fluor_channels()
        disp = [self.editor._fmt_channel(c) for c in chans]
        self.chan_cb['values'] = disp
        self.apply_cb['values'] = disp
        if disp:
            self.chan_var.set(disp[0])
            self.apply_var.set(disp[0])

    def _detect(self):
        from .calibration import detect_bead_peaks
        s = self.editor._samples.get(self.bead_var.get())
        ch = self.editor._resolve_channel(self.chan_var.get())
        if s is None or not ch or ch not in s.data.columns:
            self.result.configure(text="Pick a bead sample + channel.")
            return
        try:
            n = max(2, int(self.npk_var.get()))
        except ValueError:
            n = 6
        peaks = detect_bead_peaks(s.data[ch].to_numpy(dtype=float), n_peaks=n)
        self.txt.delete('1.0', 'end')
        self.txt.insert('1.0', '\n'.join(f"{p:.1f}\t" for p in peaks))
        self.result.configure(text=f"Detected {len(peaks)} peaks — enter the "
                                   "MESF/ABC value after each.")

    def _parse(self):
        pairs = []
        for line in self.txt.get('1.0', 'end').splitlines():
            toks = [t for t in re.split(r'[,\s]+', line.strip()) if t]
            if len(toks) >= 2:
                try:
                    pairs.append((float(toks[0]), float(toks[1])))
                except ValueError:
                    continue
        return pairs

    def _fit(self):
        from .calibration import fit_mesf_calibration
        pairs = self._parse()
        if len(pairs) < 2:
            self.result.configure(text="Enter ≥2 peaks as 'MFI value'.")
            return
        mfi = [p[0] for p in pairs]
        known = [p[1] for p in pairs]
        try:
            self._cal = fit_mesf_calibration(mfi, known)
        except ValueError as exc:
            self.result.configure(text=str(exc))
            return
        c = self._cal
        self.result.configure(
            text=f"value = {c['slope']:.4g}·MFI + {c['intercept']:.4g}   "
                 f"(R²={c['r2']:.4f}, n={c['n']})")

    def _apply(self):
        from .calibration import apply_calibration
        if self._cal is None:
            self.result.configure(text="Fit a calibration first.")
            return
        ch = self.editor._resolve_channel(self.apply_var.get())
        if not ch:
            return
        label = self.editor._channel_labels.get(ch, ch)
        col = f'MESF:{label}'
        n_applied = 0
        for s in self.editor._samples.values():
            if ch in s.data.columns:
                s.data[col] = apply_calibration(
                    s.data[ch].to_numpy(dtype=float),
                    self._cal['slope'], self._cal['intercept'])
                n_applied += 1
        self.editor._refresh_channel_choices()
        self.editor._audit('calibration', channel=ch, column=col,
                           slope=round(self._cal['slope'], 4),
                           r2=round(self._cal['r2'], 4), n_samples=n_applied)
        self.result.configure(
            text=f"Applied to {n_applied} sample(s) → '{col}' "
                 "(now a plottable channel).")


class SampleQCWindow(tk.Toplevel):
    """Cross-sample QC: an Earth-Mover's-distance similarity matrix between the
    enabled samples + an MDS embedding (batch effects / outlier samples show up
    as separated points). Exports the distance matrix, the figure, and an
    AnnData ``.h5ad`` for the scanpy ecosystem."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Sample QC — similarity & MDS")
        self.geometry("1000x640")
        self._names = []
        self._D = None

        bar = ttk.Frame(self)
        bar.pack(fill='x', side='top')
        ttk.Label(bar, padding=6, text="EMD sample distance + MDS",
                  font=('TkDefaultFont', 9, 'bold')).pack(side='left')
        ttk.Button(bar, text="Export AnnData (.h5ad)…",
                   command=self._export_h5ad).pack(side='right', padx=(0, 6),
                                                   pady=4)
        ttk.Button(bar, text="Distance CSV…",
                   command=self._export_csv).pack(side='right', padx=(0, 4),
                                                  pady=4)
        ttk.Button(bar, text="Figure…",
                   command=self._export_fig).pack(side='right', padx=(0, 4),
                                                  pady=4)
        self.bg_var = tk.StringVar(value='White')
        ttk.Combobox(bar, textvariable=self.bg_var, width=11, state='readonly',
                     values=['White', 'Transparent', 'Translucent']).pack(
            side='right', padx=(0, 4))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(10, 4.8), dpi=100)
        cf = ttk.Frame(self)
        cf.pack(fill='both', expand=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=cf)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        self.after(50, self._compute)

    def _samples(self):
        return self.editor._selected_samples()

    def _markers(self):
        names = self._samples()
        shared = None
        for n in names:
            s = self.editor._samples[n]
            cols = set(getattr(s, 'fluor_channels', []) or []) & set(
                s.data.columns)
            shared = cols if shared is None else (shared & cols)
        first = self.editor._samples[names[0]]
        return [c for c in first.fluor_channels if c in (shared or set())]

    def _compute(self):
        from .interop import mds_embed, sample_distance_matrix
        names = self._samples()
        markers = self._markers()
        if len(names) < 2 or not markers:
            return
        data = {n: self.editor._samples[n].data for n in names}
        self._names, self._D = sample_distance_matrix(data, markers)
        self._xy = mds_embed(self._D)
        self.editor._audit('sample_qc', n_samples=len(names),
                           n_markers=len(markers))
        self._draw()

    def _draw(self):
        import numpy as _np
        fig = self._fig
        fig.clear()
        if self._D is None:
            return
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(self._D, cmap='magma')
        ax1.set_xticks(range(len(self._names)))
        ax1.set_yticks(range(len(self._names)))
        short = [self.editor._short_sample(n, 14) for n in self._names]
        ax1.set_xticklabels(short, rotation=90, fontsize=6)
        ax1.set_yticklabels(short, fontsize=6)
        ax1.set_title('EMD sample distance', fontsize=9)
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(1, 2, 2)
        xy = self._xy
        trials = [self.editor._sample_trial.get(n, '') for n in self._names]
        uniq = list(dict.fromkeys(trials))
        cmap = plt_get_cmap('tab10')
        for k, t in enumerate(uniq):
            m = _np.array([tr == t for tr in trials])
            ax2.scatter(xy[m, 0], xy[m, 1], s=40, label=str(t) or '—',
                        color=cmap(k % 10), zorder=3)
        for i, n in enumerate(self._names):
            ax2.annotate(self.editor._short_sample(n, 12),
                         (xy[i, 0], xy[i, 1]), fontsize=6,
                         xytext=(3, 3), textcoords='offset points')
        ax2.set_title('MDS of samples', fontsize=9)
        ax2.set_xticks([]); ax2.set_yticks([])
        if len(uniq) > 1:
            ax2.legend(fontsize=7, framealpha=0.85, title='trial')
        try:
            fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw()

    def _export_csv(self):
        if self._D is None:
            return
        import pandas as pd
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            initialfile='sample_distance.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            pd.DataFrame(self._D, index=pd.Index(self._names),
                         columns=pd.Index(self._names)).to_csv(path)
            messagebox.showinfo("Sample QC", f"Exported:\n{path}", parent=self)

    def _export_fig(self):
        if self._D is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.png', initialfile='sample_qc.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if path:
            savefig_background(self._fig, path, background=self.bg_var.get())
            messagebox.showinfo("Sample QC", f"Exported:\n{path}", parent=self)

    def _export_h5ad(self):
        from .interop import write_h5ad
        names = self._samples()
        markers = self._markers()
        if len(names) < 1 or not markers:
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.h5ad',
            initialfile='openflo_export.h5ad',
            filetypes=[('AnnData', '*.h5ad'), ('All files', '*.*')])
        if not path:
            return
        data = {n: self.editor._samples[n].data for n in names}
        obs_cols = ['leiden', 'cluster', 'flowsom_meta', 'pseudotime']
        try:
            n_obs = write_h5ad(path, data, markers, obs_cols=obs_cols)
        except ImportError as exc:
            messagebox.showwarning("AnnData export", str(exc), parent=self)
            return
        except Exception as exc:
            messagebox.showerror("AnnData export", f"Failed: {exc}",
                                 parent=self)
            return
        try:
            self.editor._audit('anndata.export', path=path, n_events=n_obs,
                               n_markers=len(markers))
        except Exception:
            pass
        messagebox.showinfo("AnnData export",
                            f"Wrote {n_obs:,} events → {path}", parent=self)


class FlowSOMTreeWindow(tk.Toplevel):
    """The classic FlowSOM star-tree: SOM nodes laid out on their minimal
    spanning tree, each drawn as a star glyph of its marker profile and
    coloured by metacluster, node size ∝ event count."""

    def __init__(self, editor, sample):
        super().__init__(editor)
        self.editor = editor
        self.sample = sample
        self.title(f"FlowSOM star tree — {sample}")
        self.geometry("900x780")
        s = editor._samples[sample]
        res = s.flowsom_result
        self._W = np.asarray(res['weights'], dtype=float)
        self._channels = list(res['channels'])
        self._n_meta = int(res.get('n_metaclusters', 1))

        df = s.data
        node = df['flowsom'].to_numpy()
        meta = df['flowsom_meta'].to_numpy()
        nn = len(self._W)
        self._counts = np.bincount(node[node >= 0], minlength=nn)
        node_meta = np.full(nn, -1, dtype=int)
        for nd in range(nn):
            mm = meta[node == nd]
            mm = mm[mm >= 0]
            if mm.size:
                node_meta[nd] = int(np.bincount(mm).argmax())
        self._node_meta = node_meta

        bar = ttk.Frame(self)
        bar.pack(fill='x', side='top')
        ttk.Label(bar, padding=6,
                  text=f"{nn} SOM nodes · {self._n_meta} metaclusters · "
                       f"{len(self._channels)} markers",
                  font=('TkDefaultFont', 9, 'bold')).pack(side='left')
        ttk.Button(bar, text="Export figure…", command=self._export).pack(
            side='right', padx=6, pady=4)
        self.bg_var = tk.StringVar(value='White')
        ttk.Combobox(bar, textvariable=self.bg_var, width=11, state='readonly',
                     values=['White', 'Transparent', 'Translucent']).pack(
            side='right')

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(8.5, 7.0), dpi=100)
        cf = ttk.Frame(self)
        cf.pack(fill='both', expand=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=cf)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        self._draw()

    def _draw(self):
        import matplotlib.patches as mpatches

        from .pipeline import flowsom_layout, flowsom_mst
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        W = self._W
        edges, _ = flowsom_mst(W)
        pos = flowsom_layout(len(W), edges)
        if len(pos) == 0:
            self._canvas.draw()
            return
        # Per-channel scale of the prototypes to [0, 1] for the star spokes.
        lo = W.min(0)
        rng = W.max(0) - lo
        rng[rng == 0] = 1.0
        Ws = (W - lo) / rng
        extent = float(np.max(pos.max(0) - pos.min(0))) or 1.0
        base_r = extent * 0.045
        cmap = plt_get_cmap('tab20')

        for i, j in edges:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color='#cccccc', lw=0.8, zorder=1)

        M = len(self._channels)
        angs = np.linspace(0, 2 * np.pi, M, endpoint=False)
        cmax = float(self._counts.max()) or 1.0
        for nd in range(len(W)):
            x, y = pos[nd]
            scale = base_r * (0.45 + 1.4 * np.sqrt(self._counts[nd] / cmax))
            r = scale * (0.25 + 0.75 * Ws[nd])
            xs = x + r * np.cos(angs)
            ys = y + r * np.sin(angs)
            color = cmap((self._node_meta[nd] % 20) / 20.0) \
                if self._node_meta[nd] >= 0 else '#999999'
            ax.fill(xs, ys, color=color, alpha=0.85, zorder=3,
                    edgecolor='black', lw=0.3)

        # Reference star (marker → spoke) in the corner.
        rx, ry = pos[:, 0].min(), pos[:, 1].max()
        for k, ch in enumerate(self._channels):
            ax.plot([rx, rx + base_r * 1.5 * np.cos(angs[k])],
                    [ry, ry + base_r * 1.5 * np.sin(angs[k])],
                    color='#666', lw=0.6, zorder=2)
            ax.text(rx + base_r * 1.9 * np.cos(angs[k]),
                    ry + base_r * 1.9 * np.sin(angs[k]),
                    self.editor._fmt_channel(ch).split(' (')[0],
                    fontsize=6, ha='center', va='center', color='#444')
        # Metacluster legend.
        handles = [mpatches.Patch(color=cmap((m % 20) / 20.0), label=f"mc {m}")
                   for m in sorted(set(self._node_meta[self._node_meta >= 0]))]
        if handles:
            ax.legend(handles=handles, fontsize=7, loc='lower right',
                      framealpha=0.85, ncol=2, title='metacluster')
        ax.set_title(f"FlowSOM star tree — {self.sample}", fontsize=10)
        ax.autoscale_view()
        try:
            fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw()

    def _export(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.png', initialfile='flowsom_tree.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if not path:
            return
        savefig_background(self._fig, path, background=self.bg_var.get())
        try:
            self.editor._audit('flowsom_tree.export', path=path,
                               sample=self.sample)
        except Exception:
            pass
        messagebox.showinfo("FlowSOM tree", f"Exported:\n{path}", parent=self)


class AxisConfigDialog(tk.Toplevel):
    """Tiny modal: pick scale (linear / log) and optionally a fixed
    (min, max) display range for the given channel. Calls
    ``on_apply(scale_str, range_tuple_or_None)`` when the user hits OK.
    (Symlog is backend-only and not offered here.)
    """

    def __init__(self, parent, channel, scale, rng, on_apply):
        super().__init__(parent)
        self.title(f"Axis: {channel}")
        self.transient(parent)
        self.resizable(False, False)
        self.on_apply = on_apply

        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)

        ttk.Label(body, text=f"Channel: {channel}",
                  font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))

        # Scale radios.
        ttk.Label(body, text="Scale:").grid(row=1, column=0,
                                            sticky='e', padx=(0, 6))
        # Symlog is intentionally not offered here (backend-only — its density
        # binning is artefact-prone on some scatter views). A channel that
        # still carries a legacy 'symlog' scale shows as Log in the picker.
        self.scale_var = tk.StringVar(
            value=scale if scale in ('linear', 'log') else 'log')
        for i, (lbl, val) in enumerate([('Linear', 'linear'),
                                        ('Log',    'log')]):
            ttk.Radiobutton(body, text=lbl, value=val,
                            variable=self.scale_var).grid(
                row=1, column=1 + i, sticky='w', padx=(0, 8))

        # Range section.
        self.auto_var = tk.BooleanVar(value=(rng is None))
        ttk.Checkbutton(body, text="Auto-range",
                        variable=self.auto_var,
                        command=self._toggle_range).grid(
            row=2, column=0, columnspan=4, sticky='w', pady=(8, 2))

        ttk.Label(body, text="Min:").grid(row=3, column=0,
                                          sticky='e', padx=(0, 6))
        self.min_var = tk.StringVar(value=(f"{rng[0]:g}" if rng else ''))
        self.min_entry = ttk.Entry(body, textvariable=self.min_var, width=12)
        self.min_entry.grid(row=3, column=1, sticky='w', padx=(0, 12))

        ttk.Label(body, text="Max:").grid(row=3, column=2,
                                          sticky='e', padx=(0, 6))
        self.max_var = tk.StringVar(value=(f"{rng[1]:g}" if rng else ''))
        self.max_entry = ttk.Entry(body, textvariable=self.max_var, width=12)
        self.max_entry.grid(row=3, column=3, sticky='w')
        self._toggle_range()

        # Buttons.
        bot = ttk.Frame(self, padding=(12, 0, 12, 12))
        bot.pack(fill='x')
        ttk.Button(bot, text="Cancel",
                   command=self.destroy).pack(side='right')
        ttk.Button(bot, text="OK",
                   command=self._on_ok).pack(side='right', padx=(0, 6))

        # Status line for parse errors.
        self.err_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.err_var,
                  foreground='red', padding=(12, 0, 12, 4)).pack(
            side='bottom', fill='x')

    def _toggle_range(self):
        state = 'disabled' if self.auto_var.get() else 'normal'
        self.min_entry.configure(state=state)
        self.max_entry.configure(state=state)

    def _on_ok(self):
        rng = None
        if not self.auto_var.get():
            try:
                lo = float(self.min_var.get().strip())
                hi = float(self.max_var.get().strip())
            except ValueError:
                self.err_var.set("Min/Max must be numbers.")
                return
            if not (lo < hi):
                self.err_var.set("Min must be < Max.")
                return
            rng = (lo, hi)
        try:
            self.on_apply(self.scale_var.get(), rng)
        except Exception as exc:
            self.err_var.set(f"Apply failed: {exc}")
            return
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# COMPENSATION MATRIX EDITOR
# ══════════════════════════════════════════════════════════════════════════════

class CompensationEditorWindow(tk.Toplevel):
    """Modal window for viewing and editing a spillover matrix.

    Auto-imports on open by trying, in order:
      1. the active sample's FCS file ($SPILL keyword), and
      2. a .wsp sitting alongside the FCS file (any matrix it carries).
    Otherwise opens with whatever was loaded explicitly via Load…, or an
    identity matrix when the user clicks Reset.

    `on_apply(channels, matrix)` is called when the user clicks Apply.
    """

    def __init__(self, parent, sample=None, on_apply=None):
        super().__init__(parent)
        self.title("Compensation Matrix")
        self.transient(parent)
        self.geometry("780x520")
        self.minsize(560, 360)
        self.sample   = sample
        self.on_apply = on_apply
        self.channels = []
        self.matrix   = None
        self.entries  = {}     # (i, j) -> (StringVar, Entry)
        self._build()
        if sample is not None:
            self._auto_import()

    # ── UI ───────────────────────────────────────────────────────────────

    def _build(self):
        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(side='top', fill='x')
        ttk.Button(top, text="Load (CSV/WSP/FCS)…",
                   command=self._load).pack(side='left')
        ttk.Button(top, text="Save…",
                   command=self._save).pack(side='left', padx=(4, 0))
        ttk.Button(top, text="Reset to identity",
                   command=self._reset_identity).pack(side='left', padx=(4, 0))
        ttk.Button(top, text="Optimize from single-stains…",
                   command=self._open_optimize).pack(side='left', padx=(12, 0))

        self.status_var = tk.StringVar(
            value="Load a matrix or click Reset to start from identity.")
        ttk.Label(self, textvariable=self.status_var,
                  foreground='grey',
                  padding=(8, 0, 8, 0)).pack(side='top', fill='x')

        body = ttk.Frame(self, padding=(8, 4, 8, 4))
        body.pack(side='top', fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        # Scrollable matrix area.
        cv = tk.Canvas(body, highlightthickness=0)
        cv.grid(row=0, column=0, sticky='nsew')
        vbar = ttk.Scrollbar(body, orient='vertical', command=cv.yview)
        vbar.grid(row=0, column=1, sticky='ns')
        hbar = ttk.Scrollbar(body, orient='horizontal', command=cv.xview)
        hbar.grid(row=1, column=0, sticky='ew')
        cv.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self._cv = cv
        self.matrix_frame = ttk.Frame(cv)
        cv.create_window((0, 0), window=self.matrix_frame, anchor='nw')
        self.matrix_frame.bind(
            '<Configure>',
            lambda e: cv.configure(scrollregion=cv.bbox('all')))

        bot = ttk.Frame(self, padding=8)
        bot.pack(side='bottom', fill='x')
        ttk.Button(bot, text="Close",
                   command=self.destroy).pack(side='right')
        ttk.Button(bot, text="Apply",
                   command=self._apply).pack(side='right', padx=(0, 4))

        self._render_matrix()

    def _render_matrix(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        self.entries = {}
        if self.matrix is None or not self.channels:
            ttk.Label(self.matrix_frame,
                      text="(no matrix loaded)",
                      foreground='grey').grid(row=0, column=0, padx=20, pady=20)
            return
        n = len(self.channels)
        # Top-left corner cell hint
        ttk.Label(self.matrix_frame, text="src \\ dst",
                  foreground='grey',
                  font=('TkDefaultFont', 8, 'italic')).grid(
            row=0, column=0, padx=4, pady=2, sticky='e')
        # Destination column headers (dst across the top).
        for j, ch in enumerate(self.channels):
            ttk.Label(self.matrix_frame, text=ch,
                      font=('TkDefaultFont', 8, 'bold')).grid(
                row=0, column=j + 1, padx=2, pady=2)
        # Source row labels + entry cells.
        for i, ch in enumerate(self.channels):
            ttk.Label(self.matrix_frame, text=ch,
                      font=('TkDefaultFont', 8, 'bold')).grid(
                row=i + 1, column=0, padx=4, pady=1, sticky='e')
            for j in range(n):
                var = tk.StringVar(value=f"{float(self.matrix[i, j]):.6f}")
                e = ttk.Entry(self.matrix_frame, textvariable=var, width=10,
                              justify='right',
                              foreground=('#222' if i != j else '#888'))
                e.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                self.entries[(i, j)] = (var, e)

    # ── State helpers ────────────────────────────────────────────────────

    def _read_matrix_from_entries(self):
        if not self.entries:
            return None
        n = len(self.channels)
        m = np.zeros((n, n), dtype=float)
        for (i, j), (var, _e) in self.entries.items():
            try:
                m[i, j] = float(var.get())
            except ValueError:
                self.status_var.set(f"Cell [{i}, {j}] is not a number.")
                return None
        return m

    def _set_matrix(self, channels, matrix, source_label=''):
        self.channels = list(channels)
        self.matrix   = np.asarray(matrix, dtype=float)
        self._render_matrix()
        if source_label:
            self.status_var.set(
                f"Loaded {len(self.channels)}×{len(self.channels)} matrix "
                f"from {source_label}.")

    # ── Auto-import ──────────────────────────────────────────────────────

    def _auto_import(self):
        from .pipeline import read_compensation_matrix
        # 0) If a matrix is already applied to this sample, show that — so
        #    edits build on the active matrix and it "stays loaded" across
        #    reopens, rather than silently reverting to the FCS $SPILL.
        cm = getattr(self.sample, 'comp_matrix', None)
        cc = getattr(self.sample, 'comp_channels', None)
        if cm is not None and cc:
            self._set_matrix(list(cc), cm, source_label='currently applied')
            return
        path = getattr(self.sample, 'path', None)
        if not path:
            return
        # 1) Try embedded $SPILL in the FCS itself.
        try:
            ch, m = read_compensation_matrix(path)
            if m is not None:
                self._set_matrix(ch, m,
                                 source_label=f'FCS $SPILL ({os.path.basename(path)})')
                return
        except Exception:
            pass
        # 2) Try a sibling .wsp in the same folder.
        folder = os.path.dirname(path)
        if folder and os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith('.wsp'):
                    try:
                        ch, m = read_compensation_matrix(
                            os.path.join(folder, fn))
                        if m is not None:
                            self._set_matrix(ch, m,
                                             source_label=f'sibling .wsp ({fn})')
                            return
                    except Exception:
                        continue
        # 3) Try a sibling compensation.csv / spillover.csv.
        for name in ('compensation.csv', 'spillover.csv', 'comp.csv'):
            p = os.path.join(folder, name) if folder else name
            if os.path.isfile(p):
                try:
                    ch, m = read_compensation_matrix(p)
                    if m is None:
                        continue
                    if ch is None:        # headerless csv → use sample channels
                        data = getattr(self.sample, 'data', None)
                        if data is not None and m.shape[0] <= len(data.columns):
                            ch = list(data.columns)[:m.shape[0]]
                    if ch is not None:
                        self._set_matrix(ch, m, source_label=f'{name}')
                        return
                except Exception:
                    continue
        self.status_var.set(
            "No embedded matrix found. Load… or Reset to identity.")

    # ── Button handlers ──────────────────────────────────────────────────

    def _load(self):
        from .pipeline import read_compensation_matrix
        path = filedialog.askopenfilename(
            title="Load compensation matrix",
            filetypes=[('Compensation', '*.wsp *.csv *.tsv *.fcs'),
                       ('FlowJo workspace', '*.wsp'),
                       ('CSV', '*.csv'),
                       ('TSV', '*.tsv'),
                       ('FCS (embedded $SPILL)', '*.fcs'),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            ch, m = read_compensation_matrix(path)
        except Exception as exc:
            self.status_var.set(f"Load failed: {exc}")
            return
        if m is None:
            self.status_var.set(
                f"No matrix found in {os.path.basename(path)}")
            return
        if ch is None:
            data = getattr(self.sample, 'data', None)
            if data is not None and m.shape[0] <= len(data.columns):
                ch = list(data.columns)[:m.shape[0]]
            else:
                ch = [f'ch{i}' for i in range(m.shape[0])]
        self._set_matrix(ch, m, source_label=os.path.basename(path))

    def _save(self):
        from .pipeline import write_compensation_matrix
        m = self._read_matrix_from_entries()
        if m is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save compensation matrix",
            defaultextension='.csv',
            initialfile='compensation.csv',
            filetypes=[('CSV', '*.csv'),
                       ('TSV', '*.tsv'),
                       ('FlowJo workspace', '*.wsp')])
        if not path:
            return
        try:
            write_compensation_matrix(path, m, self.channels)
            self.status_var.set(f"Saved → {os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Save failed: {exc}")

    def _reset_identity(self):
        # If no channels yet, prefer the active sample's columns.
        if not self.channels:
            data = getattr(self.sample, 'data', None)
            if data is not None:
                # Default to the fluor channels if the sample classified
                # them; otherwise every column.
                fl = getattr(self.sample, 'fluor_channels', None) or list(data.columns)
                self.channels = list(fl)
            else:
                self.status_var.set("Load a sample before resetting to identity.")
                return
        n = len(self.channels)
        self.matrix = np.eye(n, dtype=float)
        self._render_matrix()
        self.status_var.set(f"Reset to {n}×{n} identity matrix.")

    def _open_optimize(self):
        if not self.channels:
            self.status_var.set(
                "Load a matrix or pick a sample first so the channel "
                "list is known.")
            return
        OptimizeCompensationDialog(
            self, channels=self.channels,
            on_complete=self._set_matrix)

    def _apply(self):
        m = self._read_matrix_from_entries()
        if m is None:
            return
        if self.on_apply:
            self.on_apply(self.channels, m)
        self.status_var.set("Applied.")


class OptimizeCompensationDialog(tk.Toplevel):
    """Per-fluor file picker → single-stain regression. Calls
    `on_complete(channels, matrix)` when the user runs the optimisation."""

    def __init__(self, parent, channels, on_complete):
        super().__init__(parent)
        self.title("Optimize Compensation Matrix")
        self.transient(parent)
        self.grab_set()
        self.geometry("680x420")
        self.minsize(560, 280)
        self.channels    = list(channels)
        self.on_complete = on_complete
        self.path_vars   = {}     # channel -> StringVar
        self._build()

    def _build(self):
        ttk.Label(self, text="Optimize compensation from single-stain controls",
                  font=('TkDefaultFont', 10, 'bold'),
                  padding=(10, 10, 10, 2)).pack(side='top', fill='x')
        ttk.Label(self,
                  text="For each fluor channel, point to a single-stain FCS "
                       "where ONLY that dye is bright. The regression uses "
                       "the brightest 2% of events on the source channel to "
                       "estimate spillover into every other channel.",
                  foreground='grey',
                  wraplength=620,
                  padding=(10, 0, 10, 8),
                  justify='left').pack(side='top', fill='x')

        body = ttk.Frame(self, padding=(10, 0, 10, 0))
        body.pack(side='top', fill='both', expand=True)
        body.columnconfigure(1, weight=1)
        for i, ch in enumerate(self.channels):
            var = tk.StringVar()
            self.path_vars[ch] = var
            ttk.Label(body, text=ch).grid(row=i, column=0, sticky='e', padx=4, pady=2)
            ttk.Entry(body, textvariable=var).grid(
                row=i, column=1, sticky='ew', padx=4, pady=2)
            ttk.Button(body, text="Browse…", width=10,
                       command=lambda c=ch: self._browse(c)).grid(
                row=i, column=2, padx=4, pady=2)

        # Convenience: pick a directory and try to auto-match filenames.
        ttk.Button(body, text="Auto-detect from a folder…",
                   command=self._autofill_from_dir).grid(
            row=len(self.channels), column=1, sticky='w', pady=(8, 0))

        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var,
                  foreground='grey', padding=(10, 0, 10, 4)).pack(
            side='top', fill='x')

        bot = ttk.Frame(self, padding=10)
        bot.pack(side='bottom', fill='x')
        ttk.Button(bot, text="Cancel",
                   command=self.destroy).pack(side='right')
        ttk.Button(bot, text="Run optimisation",
                   command=self._run).pack(side='right', padx=(0, 4))

    def _browse(self, ch):
        p = filedialog.askopenfilename(
            title=f"Single-stain control for '{ch}'",
            filetypes=[('FCS', '*.fcs'), ('All files', '*.*')])
        if p:
            self.path_vars[ch].set(p)

    @staticmethod
    def _channel_tokens(channel):
        """Yield candidate filename tokens for ``channel``, longest /
        most-specific first.

        Examples
        --------
        'BV421-A' -> ['bv421', 'horizonbv421']
        'PE-Cy7-A'        -> ['pecy7', 'cy7', 'pe']
        'APC-Cy7-A'       -> ['apccy7', 'cy7', 'apc']
        'PerCP-Cy5.5-A'   -> ['percpcy5.5', 'cy5.5', 'percp']
        'APC-A'           -> ['apc']
        'FITC-A'          -> ['fitc']

        The naïve ``ch.split('-')[0]`` heuristic this replaces would
        return ``'pe'`` for ``PE-Cy7-A`` — too generic; it picks up any
        FCS that happens to contain the letters ``pe``, e.g. a
        ``compensation_pe-cy7.fcs`` would be matched for the PE channel
        even though it belongs to PE-Cy7.
        """
        import re as _re
        s = channel
        # Strip vendor prefixes — most labs annotate the dye, not the brand.
        for prefix in ('Horizon ', 'BD ', 'eFluor ', 'Brilliant ',
                       'Alexa Fluor ', 'AF', 'Super Bright '):
            if s.lower().startswith(prefix.lower()):
                s = s[len(prefix):]
                break
        # Strip trailing -A / -H / -W (PMT area / height / width).
        s = _re.sub(r'-[AHW]$', '', s, flags=_re.IGNORECASE)
        parts = [p for p in s.split('-') if p]
        if not parts:
            return []
        # Most specific = the joined / collapsed form (no separators).
        full = ''.join(parts).lower()
        # Then individual parts, longest first. Skip pure-digit and
        # very short fragments (less than 3 chars) — too ambiguous.
        sub = sorted({p.lower() for p in parts
                      if len(p) >= 3 and not p.isdigit()},
                     key=len, reverse=True)
        out = []
        if len(full) >= 3 and full not in sub:
            out.append(full)
        out.extend(sub)
        # De-dup while preserving order.
        seen, uniq = set(), []
        for t in out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    def _autofill_from_dir(self):
        """Match channel names against filename tokens in the chosen
        directory. Uses :meth:`_channel_tokens` to generate ordered
        candidate tokens per channel; first token that yields exactly
        one file wins. Multiple matches surface a warning, none → skip.
        """
        folder = filedialog.askdirectory(
            title="Pick a folder with single-stain FCS files")
        if not folder:
            return
        files = [f for f in os.listdir(folder) if f.lower().endswith('.fcs')]
        flow = [f.lower() for f in files]

        results, ambiguous, missed = {}, [], []
        for ch in self.channels:
            best = None
            for token in self._channel_tokens(ch):
                hits = [files[i] for i, name in enumerate(flow)
                        if token in name]
                if len(hits) == 1:
                    best = hits[0]
                    break
                if len(hits) > 1 and best is None:
                    # Remember the first ambiguous hit-set in case no
                    # subsequent token disambiguates.
                    best = hits[0]   # heuristic: alphabetically-first match
                    ambiguous.append((ch, token, hits))
                    # Keep trying — a more specific token may still narrow.
            if best:
                results[ch] = best
            else:
                missed.append(ch)

        for ch, f in results.items():
            self.path_vars[ch].set(os.path.join(folder, f))

        msg_parts = [f"Matched {len(results)}/{len(self.channels)} channels"]
        if ambiguous:
            amb_summary = '; '.join(
                f"{ch} matched {len(hits)} files via '{tok}'"
                for ch, tok, hits in ambiguous[:3])
            msg_parts.append(f"ambiguous: {amb_summary}")
        if missed:
            msg_parts.append(f"unmatched: {', '.join(missed)}")
        self.status_var.set('  •  '.join(msg_parts))

    def _run(self):
        paths = {ch: v.get().strip()
                 for ch, v in self.path_vars.items()
                 if v.get().strip()}
        if not paths:
            self.status_var.set("Pick at least one single-stain file.")
            return
        from .pipeline import optimize_compensation
        try:
            ch, m = optimize_compensation(self.channels, paths)
        except Exception as exc:
            self.status_var.set(f"Optimization failed: {exc}")
            return
        self.on_complete(ch, m)
        self.destroy()


def main() -> int:
    """Entry point for the ``openflo-gui`` console script and
    ``python -m openflo.gui``.

    The **gate editor** is the whole UI; it owns a hidden Tk root.
    Pipelines run from the editor's docked Pipeline Workspace (one
    subprocess per item). Closing the editor destroys the root and exits.
    """
    # Under pythonw.exe (windowed launch: the gui-script .exe, the .bat, or
    # a desktop shortcut) there is no console, so sys.stdout / sys.stderr
    # are None. The editor's print() calls would then raise during startup
    # and the window would never appear ("nothing happens"). Route the
    # streams to a log file so the GUI launches and any output is
    # recoverable.
    if sys.stdout is None or sys.stderr is None:
        import tempfile
        log_path = os.path.join(tempfile.gettempdir(), 'openflo-gui.log')
        try:
            sink = open(log_path, 'a', buffering=1, encoding='utf-8')
        except Exception:
            sink = open(os.devnull, 'w')
        if sys.stdout is None:
            sys.stdout = sink
        if sys.stderr is None:
            sys.stderr = sink

    root = _APP_BASE()                   # TkinterDnD.Tk when available → OS file-drop
    root.withdraw()                      # hidden root; the editor is the UI
    editor = ViewGateEditorWindow(
        root,
        fcs_dir=None,
        labels_str='',
        on_save=None,
        primary=True,
    )
    editor.title("OpenFlo — Gate Editor")
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
