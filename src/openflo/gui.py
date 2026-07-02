"""
openflo.gui — OpenFlo pipeline GUI
Run with:  openflo-gui     (or:  python -m openflo.gui)
"""

# Surface C-level crashes (e.g. native Tk / tkdnd faults) with a Python
# stack so we can diagnose them instead of just seeing exit-139.
import faulthandler
import os
import queue
import sys
import threading
import tkinter as tk

# messagebox / filedialog are re-exported here as the conventional dialog
# monkeypatch points for the test-suite and back-compat, even though the
# editor's own dialog code now lives in the editor_* mixin modules.
from tkinter import filedialog, messagebox, ttk  # noqa: F401


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


from . import gating as _gating  # noqa: E402  (pure gating helpers)
from . import paths as _paths  # noqa: E402  (pure filesystem path helpers)
from . import plotmath as _pm  # noqa: E402  (pure plot/maths helpers)
from . import tree_ids as _tids  # noqa: E402  (pure tree-row id codec)
from .editor_analysis import AnalysisMixin  # noqa: E402  (editor mixin)
from .editor_audit import AuditMixin  # noqa: E402  (editor mixin)
from .editor_autoclean import AutoCleanMixin  # noqa: E402  (editor mixin)
from .editor_autogate import AutoGateMixin  # noqa: E402  (editor mixin)
from .editor_channels import ChannelsMixin  # noqa: E402  (editor mixin)
from .editor_chrome import ChromeMixin  # noqa: E402  (editor mixin)
from .editor_clipboard import ClipboardMixin  # noqa: E402  (editor mixin)
from .editor_compute import ComputeMixin  # noqa: E402  (editor mixin)
from .editor_console import ConsoleLogMixin  # noqa: E402  (editor mixin)
from .editor_dnd import DnDMixin  # noqa: E402  (editor mixin)
from .editor_downsample import DownsampleMixin  # noqa: E402  (editor mixin)
from .editor_export import ExportMixin  # noqa: E402  (editor mixin)
from .editor_figure import FigureMixin  # noqa: E402  (editor mixin)
from .editor_gatetools import GateToolsMixin  # noqa: E402  (editor mixin)
from .editor_gating import GatingMixin  # noqa: E402  (editor mixin)
from .editor_grouping import GroupingMixin  # noqa: E402  (editor mixin)
from .editor_help import HelpMixin  # noqa: E402  (editor mixin)
from .editor_lifecycle import LifecycleMixin  # noqa: E402  (editor mixin)
from .editor_load import LoadMixin  # noqa: E402  (editor mixin)
from .editor_loadpool import (  # noqa: E402  (editor mixin + shared constant)
    _LOAD_POOL_SIZE,  # noqa: F401  (re-exported for tests / back-compat)
    LoadPoolMixin,
)
from .editor_menu import MenuMixin  # noqa: E402  (editor mixin)
from .editor_mode import ModeMixin  # noqa: E402  (editor mixin)
from .editor_plot import PlotMixin  # noqa: E402  (editor mixin)
from .editor_populations import PopulationsMixin  # noqa: E402  (editor mixin)
from .editor_session import SessionMixin  # noqa: E402  (editor mixin)
from .editor_slider import SliderMixin  # noqa: E402  (editor mixin)
from .editor_stats import StatsMixin  # noqa: E402  (editor mixin)
from .editor_template import TemplateMixin  # noqa: E402  (editor mixin)
from .editor_tools import ToolsMixin  # noqa: E402  (editor mixin)
from .editor_tree import TreeMixin  # noqa: E402  (editor mixin)
from .editor_undo import UndoMixin  # noqa: E402  (editor mixin)
from .editor_update import UpdateMixin  # noqa: E402  (editor mixin)
from .prefs import read_prefs  # noqa: E402
from .theme import (  # noqa: E402  (shared palette / figure helpers)
    _DARK_MODES,
    THEMES,
    current_palette,
    savefig_background,  # noqa: F401  (re-exported for tests / back-compat)
    set_active_palette,
)
from .ui_logic import (  # noqa: E402  (pure, Tk-free UI logic — safe at top)
    short_label,
)

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
# (Defined in editor_loadpool and re-imported below, so the loader mixin and the
# constructor share one source of truth.)


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

class ViewGateEditorWindow(tk.Toplevel, AnalysisMixin, AuditMixin,
                           AutoCleanMixin, AutoGateMixin, ChannelsMixin,
                           ChromeMixin, ClipboardMixin, ComputeMixin,
                           ConsoleLogMixin, DnDMixin, DownsampleMixin,
                           ExportMixin, FigureMixin, GateToolsMixin, GatingMixin,
                           GroupingMixin, HelpMixin, LifecycleMixin, LoadMixin,
                           LoadPoolMixin, MenuMixin, ModeMixin, PlotMixin,
                           PopulationsMixin, SessionMixin, SliderMixin,
                           StatsMixin, TemplateMixin, ToolsMixin, TreeMixin,
                           UndoMixin, UpdateMixin):
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
        self.minsize(1180, 620)
        # Restore the last window size/position (validated on-screen), else a
        # sensible default. Only the primary window persists geometry.
        self.geometry(self._restore_geometry("1500x840") if primary
                      else "1500x840")
        # Route unhandled Tk callback errors through our handler (scrubbed
        # report + status flag + auto-reveal console). Installed on the root so
        # it covers every widget's callbacks.
        self._error_count = 0
        try:
            self._root().report_callback_exception = (  # type: ignore[attr-defined]
                self._report_callback_exception)
        except Exception:
            pass

        # Full menubar is built at the END of __init__ (it binds to display /
        # log vars that don't exist yet here). Quiet startup update check on
        # the primary window only — surfaces a note in the status bar when a
        # newer release exists; fails silently.
        if primary:
            try:
                self.after(2500, lambda: self._check_for_updates(silent=True))
            except Exception:
                pass

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
        # Priority queue of (priority, seq, payload): lower priority first, seq
        # is a FIFO tiebreak; payload is the load job or None (shutdown). The
        # active / first-rendered sample is queued at priority 0 so its plot
        # appears first; everything else at 1. See LoadPoolMixin._enqueue_load.
        self._load_queue = queue.PriorityQueue()
        self._load_seq = 0                # monotonic tiebreak (Tk thread only)
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
        # Per-population density scaling. DEFAULT is scaled (matched to the
        # cloud's display fraction) — the 'density' tree column shows ☑ then.
        # A (sample, gid) here is the OPT-OUT: drawn at FULL density instead
        # (☐ in the column). Applies to both the highlight overlay and the
        # backgate projection of that population.
        self._gate_density_full: set[tuple[str, str]] = set()
        # Backgate targets toggled OFF: still listed in the legend (so they can
        # be turned back on) but not drawn. On/off lives in the legend now.
        self._backgate_hidden: set[tuple[str, str]] = set()
        # Map of clickable legend artists → (action, target), rebuilt each time
        # the backgate legend is drawn. action ∈ {'color','toggle','density',
        # 'collapse'}; target is (sample, gid) or None for 'collapse'.
        self._backgate_legend_pick: dict = {}
        # Backgate legend placement/state. The legend can be dragged (by its
        # header) and collapsed; while expanded its box swallows plot clicks so
        # a stray click can't create/select a gate underneath it.
        self._backgate_legend_anchor = (0.020, 0.965)   # top-left, axes frac
        self._backgate_legend_collapsed = False
        self._backgate_legend_bbox = None       # (x0,y0,x1,y1) axes frac | None
        self._backgate_legend_header = None     # header-strip bbox (drag handle)
        self._backgate_legend_artists: list = []
        self._backgate_legend_rows: list = []
        self._legend_drag = None                # active drag scratch | None
        # Chrome theme (light/dark), persisted in prefs and applied app-wide.
        self._theme_var = tk.StringVar(value=read_prefs().get('theme', 'light'))
        # Render pop-up / preview figures (compensation QC, gating tree,
        # embedding comparison) on a dark background so they aren't blinding
        # under the dark themes. Defaults on when the chrome is dark.
        self._dark_figs = tk.BooleanVar(
            value=bool(read_prefs().get(
                'dark_figures', self._theme_var.get() in _DARK_MODES)))
        # Spawn child dialogs at a fixed corner of the main window instead of
        # wherever the OS places them ('off' | 'top-left' | 'top-right').
        self._spawn_corner = tk.StringVar(
            value=str(read_prefs().get('spawn_corner', 'off')))
        # Hover tooltips on controls (View → Show hover tips), persisted.
        self._tooltips_enabled = tk.BooleanVar(
            value=bool(read_prefs().get('tooltips', True)))
        try:
            self.configure(bg=current_palette()['bg'])
        except Exception:
            pass

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

        self.columnconfigure(0, weight=1)
        # row 0 = menubar · 1 = content (main split) · 2 = status
        self.rowconfigure(1, weight=1)

        # ── Menubar strip: its own compact row under the title bar (the native
        # Win32 menubar can't be dark-themed). Populated in _build_menubar.
        self._menubar_frame = ttk.Frame(self, padding=(4, 1))
        self._menubar_frame.grid(row=0, column=0, columnspan=2, sticky='ew')

        # ── Main horizontal split: [ Samples & Gates | plot + workspace ].
        # Each section is its own draggable pane (drag the sashes to resize),
        # so the left panel can be widened or narrowed just like the workspace.
        self._main_paned = ttk.PanedWindow(self, orient='horizontal')
        self._main_paned.grid(row=1, column=0, columnspan=2, sticky='nsew')

        # Left pane: a centred sample-I/O bar on top, then the samples-and-gates
        # tree (samples are root nodes, gates nested children — FlowJo-style),
        # then the gate-ops / tooling button block lower down. The pane lives in
        # a classic tk.Frame HOST so the whole panel can be popped out into its
        # own window (Tk 'wm manage' needs a tk frame, not ttk) and re-docked.
        self._left_host = tk.Frame(self._main_paned, bg=current_palette()['bg'])
        # weight=0: the left pane keeps its natural (control-fitting) width and
        # does NOT absorb a share of extra window width — all the spare space
        # flows to the plot + workspace (weight=4). The user can still drag the
        # sash freely; this only sets the default distribution.
        self._main_paned.add(self._left_host, weight=0)
        self._left_popped = False
        left = ttk.Frame(self._left_host, padding=6)
        left.pack(fill='both', expand=True)
        left.rowconfigure(1, weight=1)   # tree expands
        left.columnconfigure(0, weight=1)

        leftbar = ttk.Frame(left)
        leftbar.grid(row=0, column=0, columnspan=2, pady=(0, 4))   # centred
        _addb = ttk.Button(leftbar, text="＋ Add FCS", command=self._add_samples)
        _addb.pack(side='left')
        self._tip(_addb, "Load one or more .fcs files (or drag them onto the "
                  "window). Files group by day/trial automatically.")
        _csvb = ttk.Button(leftbar, text="Load CSV…",
                           command=self._load_processed_data)
        _csvb.pack(side='left', padx=(4, 0))
        self._tip(_csvb, "Load a processed-data CSV (e.g. an exported table "
                  "with cluster / UMAP columns) as a sample.")
        _rmb = ttk.Button(leftbar, text="Remove", command=self._remove_selected)
        _rmb.pack(side='left', padx=(4, 0))
        self._tip(_rmb, "Remove the selected sample(s) from the session.")
        self._left_popbtn = ttk.Button(leftbar, text="Pop out",
                                       command=self._toggle_left_popout)
        self._left_popbtn.pack(side='left', padx=(4, 0))
        self._tip(self._left_popbtn, "Float this Samples & Gates panel into its "
                  "own window (drag to a second monitor); click Dock or close "
                  "it to re-dock.")
        # "On top" toggle — packed only while popped out (see _toggle_left_popout).
        self._left_ontop_var = tk.BooleanVar(value=False)
        self._left_ontop_cb = ttk.Checkbutton(
            leftbar, text="On top", variable=self._left_ontop_var,
            command=self._apply_left_ontop)
        self._tip(self._left_ontop_cb,
                  "Keep this floating panel above the main window.")

        gf = ttk.Frame(left)
        gf.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=(2, 4))
        gf.columnconfigure(0, weight=1)
        gf.rowconfigure(1, weight=1)            # tree expands; find-row is fixed
        # Find box: jump to the first sample/gate whose name matches.
        find_row = ttk.Frame(gf)
        find_row.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 2))
        find_row.columnconfigure(1, weight=1)
        ttk.Label(find_row, text='Find:').grid(row=0, column=0, padx=(2, 4))
        self._find_var = tk.StringVar()
        _find_ent = ttk.Entry(find_row, textvariable=self._find_var)
        _find_ent.grid(row=0, column=1, sticky='ew')
        self._find_entry = _find_ent          # for the Ctrl+F shortcut
        self._find_var.trace_add('write', lambda *_: self._find_in_tree())
        self._tip(_find_ent,
                  "Find a sample or gate by name — jumps to the first match.")
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
        # The #0 heading names the section AND acts as expand/collapse-all
        # (▾ = mostly expanded, ▸ = collapsed); see _toggle_expand_all.
        self.gate_tv.heading('#0', text='▾ Samples & Gates',
                             command=self._toggle_expand_all)
        # Clicking the "All" header toggles plot-inclusion for every sample
        # at once — checks all when any are off, otherwise unchecks all. Using
        # a plain word reads cleanly (the lone ✓ glyph rendered poorly) and is
        # clearly distinct from the per-row ☑/☐ marks.
        self.gate_tv.heading('on', text='All',
                             command=self._toggle_all_sample_plots)
        self.gate_tv.column('#0', width=212, anchor='w', stretch=True)
        self.gate_tv.column('on', width=34,  anchor='center', stretch=False)
        self.gate_tv.grid(row=1, column=0, sticky='nsew')
        gate_sb = ttk.Scrollbar(gf, orient='vertical',
                                command=self.gate_tv.yview)
        gate_sb.grid(row=1, column=1, sticky='ns')
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
        # Display tags (theme-aware greys; recoloured by _theme_tree_tags).
        _pal = current_palette()
        self.gate_tv.tag_configure('off',     foreground=_pal['muted'])
        self.gate_tv.tag_configure('loading', foreground=_pal['muted'])
        self.gate_tv.tag_configure('sample',
                                   font=('TkDefaultFont', 9, 'bold'))
        self.gate_tv.tag_configure('drop_target',
                                   background='#fff2a8')   # transient hover
        self.gate_tv.tag_configure('pending_move',         # staged cross-instance move
                                   foreground=_pal['accent'],
                                   font=('TkDefaultFont', 9, 'italic'))

        # Bottom action buttons — uniform size (equal grid columns), compact,
        # so the left column has room for the collapsible log pane below.
        # ↶/↷ stay as narrow icon buttons outside the weighted columns.
        gb_row = ttk.Frame(left)
        gb_row.grid(row=3, column=0, columnspan=2, sticky='ew')
        for c in range(5):
            gb_row.columnconfigure(c, weight=1, uniform='gb')
        for _c, (_lbl, _cmd, _tt) in enumerate((
                ("Clear", self._clear_selected_gate,
                 "Delete the selected gate(s) and their descendants (the "
                 "sample stays). On a sample/trial row, clears its gates."),
                ("Clear all", self._clear_all,
                 "Remove every gate from all samples (samples are kept)."),
                ("Auto-clean", self._create_autoclean_gate,
                 "Add an auto-clean gate (margins / debris / dead cells) to the "
                 "active sample; toggle its methods under it in the tree."),
                ("Copy", self._open_copy_gates_dialog,
                 "Copy the active sample's gates onto other samples (as "
                 "calculations, optionally frozen to fixed coordinates)."),
                ("Pops", self._open_clusters_menu,
                 "Import label/cluster columns as populations, or manage "
                 "imported populations."))):
            _gb = ttk.Button(gb_row, text=_lbl, command=_cmd)
            _gb.grid(row=0, column=_c, sticky='ew',
                     padx=(0 if _c == 0 else 3, 0))
            self._tip(_gb, _tt)
        _ub = ttk.Button(gb_row, text="↶", width=2, command=self._undo)
        _ub.grid(row=0, column=5, padx=(3, 0))
        self._tip(_ub, "Undo the last change (gates, populations, workspace).")
        # Redo sits BELOW Undo (row 1, col 5) beside Report — keeping both rows
        # at 6 columns so the side panel is one button narrower by default.
        _rb2 = ttk.Button(gb_row, text="↷", width=2, command=self._redo)
        _rb2.grid(row=1, column=5, padx=(3, 0), pady=(3, 0))
        self._tip(_rb2, "Redo the last undone change.")
        self._zoom_mode_var = tk.BooleanVar(value=False)
        # Analysis / tooling actions stacked as a second row of 5 under the
        # gate-ops row (moved out of the top bar — they're tools, not
        # plot/workspace controls).
        for _c, (_lbl, _cmd, _tiptext) in enumerate((
                ("Cluster…", self._open_cluster_dialog,
                 "Unsupervised clustering (PhenoGraph / Leiden / FlowSOM) with "
                 "an optional embedding (UMAP / t-SNE / …); imports the result "
                 "as populations."),
                ("Frequencies…", self._open_frequency_window,
                 "Population frequencies (% of parent / of total) across "
                 "samples, with a Prism-ready export."),
                ("Statistics…", self._open_stats_window,
                 "FlowJo-style stats table: counts, frequencies, and per-channel "
                 "MFI for each gated population."),
                ("Pipeline", self._open_pipeline_workspace,
                 "Show/hide the Pipeline panel: batch co-embedded clustering "
                 "runs over groups of samples."),
                ("Report…", self._export_report,
                 "Export an HTML analysis report of the current gates, plots, "
                 "and stats."))):
            _b = ttk.Button(gb_row, text=_lbl, command=_cmd)
            _b.grid(row=1, column=_c, sticky='ew',
                    padx=(0 if _c == 0 else 3, 0), pady=(3, 0))
            self._tip(_b, _tiptext)

        # Discoverability hint (one line — "Clear"=gates of the selected gate/
        # sample/trial, "Clear all"=all gates (samples kept), "Pops"=
        # populations/clusters menu; double-click the plot to add a gate).
        ttk.Label(left,
                  text="Tip: double-click the plot to add a gate · click a "
                       "sample to make it active.",
                  font=('TkDefaultFont', 8), foreground='grey',
                  wraplength=300, justify='left').grid(
            row=4, column=0, columnspan=2, sticky='w', pady=(3, 0))

        # Display mode (3-way) state — all / highlight / filter. The radios
        # themselves live in the centre controls (row D) next to "Show
        # cleaned-out events"; apply_gates_var is a back-compat shim some code
        # paths read.
        self.gate_display_var = tk.StringVar(value='all')
        self.apply_gates_var = tk.BooleanVar(value=False)

        ttk.Separator(left, orient='horizontal').grid(
            row=6, column=0, columnspan=2, sticky='ew', pady=(8, 4))

        # Compensation/Transforms/Calibration, Statistics/Frequencies/…,
        # Cluster/Cell-cycle/…, Templates, Sessions, Export, Report and the
        # Pipeline Workspace all moved into the menubar (Analyze / Tools /
        # File / View). The side panel now keeps only the frequent gating loop.

        # ── Log + interactive Python console (mirrors stdout/stderr) ─────
        # Shown by default; the prompt runs Python in-process against the live
        # editor (`editor`/`self`, `samples`, `np`, `pd` are pre-bound).
        logbar = ttk.Frame(left)
        logbar.grid(row=14, column=0, columnspan=2, sticky='ew', pady=(4, 0))
        self._show_log_var = tk.BooleanVar(value=False)   # collapsed by default
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
        self._toggle_log()            # apply default visibility (collapsed)
        # (Close moved to the File menu.)

        # ── Right column: a horizontal sash splitting the plot (left) from
        # the docked Pipeline Workspace (right). The workspace pane is hidden
        # by default — "Pipeline Workspace…" reveals it, which shrinks the
        # plot to share the space (drag the sash to taste).
        # Second pane of the main split: a nested horizontal sash separating
        # the plot (left) from the docked Pipeline Workspace (right, hidden by
        # default — "Pipeline Workspace…" reveals it).
        self._editor_paned = ttk.PanedWindow(self._main_paned, orient='horizontal')
        right = ttk.Frame(self._editor_paned, padding=(0, 6, 6, 6))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self._editor_paned.add(right, weight=4)
        # tk.Frame host so the whole workspace (its bar + view) pops out as one
        # via 'wm manage', the same mechanism as the left panel. The
        # WorkspacePanel inside costs ~100 ms to construct, so it's built
        # lazily on first reveal (_ensure_workspace_panel) — a session that
        # never opens the Pipeline Workspace doesn't pay that at startup.
        self._ws_host = tk.Frame(self._editor_paned, bg=current_palette()['bg'])
        self._workspace_panel = None
        self._workspace_shown = False
        self._ws_popped = False
        self._main_paned.add(self._editor_paned, weight=4)

        # Top controls. Each logical group is its own packed sub-row, so the
        # bar WRAPS downward instead of running off the right edge on a narrow
        # window (e.g. with the Pipeline Workspace docked) — previously Mode,
        # Color and the downsample toggles were clipped. Packing per sub-row
        # also avoids grid column-width bleed between unrelated rows.
        ctrl = ttk.Frame(right)
        ctrl.grid(row=0, column=0, sticky='ew', pady=(0, 4))

        # Row A — axes + colour. Rows shrink to content and centre (no fill),
        # so the control groups sit centred over the plot rather than ragged-left.
        row_a = ttk.Frame(ctrl)
        row_a.pack(pady=(0, 0))
        ttk.Label(row_a, text="X:").pack(side='left', padx=(0, 4))
        xframe = ttk.Frame(row_a)
        xframe.pack(side='left', padx=(0, 12))
        self.x_combo = ttk.Combobox(xframe, width=20)
        self.x_combo.pack(side='left')
        self.x_combo.bind('<<ComboboxSelected>>',
                          lambda *_: self._schedule_replot(0))
        ttk.Button(xframe, text='⚙', width=2,
                   command=lambda: self._open_axis_dialog('x')).pack(
            side='left', padx=(2, 0))

        _swap = ttk.Button(row_a, text='⇄', width=2, command=self._swap_axes)
        _swap.pack(side='left', padx=(0, 12))
        self._tip(_swap, "Swap the X and Y channels (each axis keeps its own "
                  "scale/range).")

        ttk.Label(row_a, text="Y:").pack(side='left', padx=(0, 4))
        yframe = ttk.Frame(row_a)
        yframe.pack(side='left', padx=(0, 12))
        self.y_combo = ttk.Combobox(yframe, width=20)
        self.y_combo.pack(side='left')
        self.y_combo.bind('<<ComboboxSelected>>',
                          lambda *_: self._schedule_replot(0))
        ttk.Button(yframe, text='⚙', width=2,
                   command=lambda: self._open_axis_dialog('y')).pack(
            side='left', padx=(2, 0))

        ttk.Label(row_a, text="Color:").pack(side='left', padx=(0, 4))
        self.color_combo = ttk.Combobox(row_a, width=18)
        self.color_combo.pack(side='left')
        self.color_combo.bind('<<ComboboxSelected>>',
                              lambda *_: self._schedule_replot(0))
        # Type-to-filter the channel pickers (helps with 30+ channel panels).
        self._make_filterable(self.x_combo, '_xy_choices')
        self._make_filterable(self.y_combo, '_xy_choices')
        self._make_filterable(self.color_combo, '_color_choices')

        # Row B — plot mode + mode-specific options. Only the options that
        # apply to the current mode are shown (the rest are hidden), so the bar
        # stays uncluttered instead of cramming every toggle in at once.
        row_b = ttk.Frame(ctrl)
        row_b.pack(pady=(6, 0))
        ttk.Label(row_b, text="Mode:").pack(side='left', padx=(0, 4))
        self.mode_var = tk.StringVar(value='dot')
        mode_combo = ttk.Combobox(row_b, textvariable=self.mode_var,
                                  state='readonly', width=12,
                                  values=list(self.PLOT_MODES))
        mode_combo.pack(side='left')
        mode_combo.bind('<<ComboboxSelected>>',
                        lambda *_: self._on_mode_changed())

        # Downsample + Max points, inline with Mode. Max points only appears
        # (and is only applied) when downsampling is Display / Display+data;
        # Off draws every event uncapped.
        ttk.Label(row_b, text="Downsample:").pack(side='left', padx=(12, 4))
        self.ds_display_var = tk.BooleanVar(value=True)
        self.ds_propagate_var = tk.BooleanVar(value=False)
        self._ds_mode_var = tk.StringVar(value='Display only')
        ds_combo = ttk.Combobox(
            row_b, textvariable=self._ds_mode_var, state='readonly', width=14,
            values=['Off', 'Display only', 'Display + data'])
        ds_combo.pack(side='left')
        ds_combo.bind('<<ComboboxSelected>>',
                      lambda *_: self._on_ds_mode_changed())
        self._tip(ds_combo, "Downsample for comparable overlays. Off: draw all "
                  "(uncapped). Display only: render the smallest sample's count "
                  "from each (view only). Display + data: also TRIM the data — "
                  "affects clustering/stats, not undoable without re-adding.")
        self._mp_label = ttk.Label(row_b, text="Max points:")
        self.max_points_var = tk.StringVar(value='60000')
        self._mp_combo = ttk.Combobox(
            row_b, textvariable=self.max_points_var, width=8,
            values=['20000', '60000', '100000', '250000', '500000', 'All'])
        self._mp_combo.bind('<<ComboboxSelected>>',
                            lambda *_: self._on_max_points_changed())
        self._mp_combo.bind('<Return>',
                            lambda *_: self._on_max_points_changed())
        self._tip(self._mp_combo, "Max events drawn per sample (only while "
                  "downsampling is on). Pick a preset or type a number; 'All' "
                  "removes the cap (slow on millions of events).")

        ttk.Separator(row_b, orient='vertical').pack(
            side='left', fill='y', padx=8, pady=1)
        # Mode-specific options live in this frame, repacked per mode by
        # _update_mode_options(). Created here, shown on demand.
        self._opt_frame = ttk.Frame(row_b)
        self._opt_frame.pack(side='left', fill='x')
        _of = self._opt_frame
        self.true_kde_var = tk.BooleanVar(value=False)
        self._kde_cb = ttk.Checkbutton(
            _of, text="True Gaussian KDE (slow)", variable=self.true_kde_var,
            command=lambda: self._schedule_replot(0))
        # Contour mode: 'Contour scatter' is the master switch (off = clean
        # contour-only plot); 'Outliers' (only meaningful when scatter is on)
        # toggles the sparse low-density points outside the contoured pop.
        self.contour_scatter_var = tk.BooleanVar(value=True)
        self._cscatter_cb = ttk.Checkbutton(
            _of, text="Contour scatter", variable=self.contour_scatter_var,
            command=lambda: self._schedule_replot(0))
        self.contour_outliers_var = tk.BooleanVar(value=True)
        self._coutliers_cb = ttk.Checkbutton(
            _of, text="Outliers", variable=self.contour_outliers_var,
            command=lambda: self._schedule_replot(0))
        # Histogram Y-axis mode. Fraction = events/bin ÷ sample total; Count =
        # raw events/bin (respects auto-downsample); % of Max = peak → 100.
        self._histy_lbl = ttk.Label(_of, text="Hist Y:")
        self.hist_y_mode = tk.StringVar(value='Fraction')
        self.hist_y_combo = ttk.Combobox(
            _of, textvariable=self.hist_y_mode, state='readonly', width=10,
            values=['Fraction', 'Count', '% of Max'])
        self.hist_y_combo.bind('<<ComboboxSelected>>',
                               lambda *_: self._schedule_replot(0))
        self._update_mode_options()

        # Hover tooltips for the plot controls.
        self._tip(self.x_combo, "X axis channel. Pick a detector/marker; the "
                  "gear sets its scale (linear / log / symlog) and range.")
        self._tip(self.y_combo, "Y axis channel. Leave blank (or pick the same "
                  "as X in histogram mode) for a 1-D distribution.")
        self._tip(self.color_combo, "Colour the dots by density, by sample, or "
                  "by a channel / cluster column.")
        self._tip(mode_combo,
                  "Plot mode — dot: fast per-event scatter · pseudocolor: "
                  "density-coloured · contour: KDE contour lines · histogram: "
                  "1-D distribution. Mode-specific options appear to the right.")
        self._tip(self._kde_cb, "Use a true Gaussian kernel density estimate "
                  "for pseudocolor (smoother, but slower on large samples).")
        self._tip(self._cscatter_cb, "Draw a faint per-event scatter under the "
                  "contour lines (off = clean contour-only plot).")
        self._tip(self._coutliers_cb, "Show sparse low-density points outside "
                  "the contoured population (only when Contour scatter is on).")
        self._tip(self.hist_y_combo, "Histogram Y scaling — Fraction: events/bin "
                  "÷ total · Count: raw events/bin · % of Max: peak set to 100%.")

        # Row C — gate-shape tool + auto-gate. Auto-gate now offers
        # well-posed, reviewable proposals (singlet ratio-band, BIC-selected
        # GMM ellipses, 1-D valley/Otsu threshold) each with a quality score,
        # rather than the old single-contour heuristic.
        row_c = ttk.Frame(ctrl)
        row_c.pack(pady=(6, 0))
        ttk.Label(row_c, text="Tool:").pack(side='left', padx=(0, 4))
        self.gate_tool_var = tk.StringVar(value='quadrant')
        tools = [('Quadrant',  'quadrant'),
                 ('Rectangle', 'rectangle'),
                 ('Polygon',   'polygon'),
                 ('Ellipse',   'ellipse'),
                 ('Lasso',     'lasso'),
                 ('Edit',      'edit')]
        _tool_tips = {
            'quadrant':  "Double-click the plot to drop 4 quadrant gates at "
                         "that point.",
            'rectangle': "Click-drag a rectangle gate.",
            'polygon':   "Click to drop vertices; double-click (or click the "
                         "first vertex) to close.",
            'ellipse':   "Click-drag an axis-aligned ellipse; use Edit to move / "
                         "resize / rotate it.",
            'lasso':     "Click-drag a free-form outline; release to close it.",
            'edit':      "Edit existing gates: drag vertices/edges, shift-drag "
                         "to move the whole gate, right-click a vertex to delete.",
        }
        self._gate_tool_widgets = []
        for lbl, val in tools:
            _rb = ttk.Radiobutton(row_c, text=lbl, value=val,
                                  variable=self.gate_tool_var,
                                  command=self._activate_gate_tool)
            _rb.pack(side='left', padx=(0, 8))
            self._tip(_rb, lambda v=val:
                      self._tool_tip_text(_tool_tips.get(v, '')))
            self._gate_tool_widgets.append(_rb)
        _ag_btn = ttk.Button(row_c, text="Auto-gate", width=10,
                             command=self._auto_gate)
        self._tip(_ag_btn, lambda: self._tool_tip_text(
            "Suggest gates for the current X/Y view: a singlet ratio-band, "
            "BIC-selected GMM ellipses, or a 1-D valley/Otsu threshold — each "
            "scored. You review and keep/tweak/delete them like any gate."))
        self._gate_tool_widgets.append(_ag_btn)
        _ag_btn.pack(
            side='left', padx=(4, 0))
        # Apply a new gate (drawn or auto) to every displayed sample, not just
        # the active one. (With this off, selecting several samples in the tree
        # also fans a new gate out to all of them.)
        self._gate_all_var = tk.BooleanVar(value=True)
        _all_cb = ttk.Checkbutton(row_c, text="→ all shown",
                                  variable=self._gate_all_var)
        _all_cb.pack(side='left', padx=(8, 0))
        self._tip(_all_cb, lambda: self._tool_tip_text(
            "Apply a new gate — drawn or Auto-gate — to EVERY displayed sample, "
            "not just the active one. (When off, selecting multiple samples in "
            "the tree also applies a new gate to all of them.)"))

        # Row D — display mode (how gates affect the cloud) + the auto-clean
        # overlay, centred together (moved here from the left panel).
        row_d = ttk.Frame(ctrl)
        row_d.pack(pady=(4, 0))
        ttk.Label(row_d, text="Display:").pack(side='left', padx=(0, 4))
        self._display_radios = {}
        for _val, _lbl, _tt in (
                ('all', 'All events',
                 "Show every event; gates just outline regions."),
                ('highlight', 'Highlight gated',
                 "Grey base population + each enabled gate's events drawn in "
                 "its own colour (forks shown side by side)."),
                ('filter', 'Filter to gated',
                 "Show only events that fall inside the enabled gates.")):
            _dr = ttk.Radiobutton(row_d, text=_lbl, value=_val,
                                  variable=self.gate_display_var,
                                  command=self._on_gate_display_changed)
            _dr.pack(side='left', padx=(0, 8))
            self._tip(_dr, _tt)
            self._display_radios[_val] = _dr
        ttk.Separator(row_d, orient='vertical').pack(
            side='left', fill='y', padx=8, pady=1)
        self.show_removed_var = tk.BooleanVar(value=False)
        self._removed_cb = ttk.Checkbutton(
            row_d, text="⚠ Show cleaned-out events",
            variable=self.show_removed_var,
            command=lambda: self._schedule_replot(0))
        self._removed_cb.pack(side='left')
        self._tip(self._removed_cb, "Overlay the events auto-clean removed (in "
                  "red) on top of the plot, so cleaning artefacts stay visible "
                  "against the full sample.")
        # Reflect the initial downsample mode (hides Max points when Off).
        self._update_ds_visibility()

        # Row E — tool gesture hint (updates with the active tool).
        self.tool_hint_var = tk.StringVar(value='')
        ttk.Label(ctrl, textvariable=self.tool_hint_var,
                  foreground='#666', font=('TkDefaultFont', 8, 'italic')
                  ).pack(fill='x', anchor='w', padx=(0, 4), pady=(2, 4))

        # Plot canvas
        cf = ttk.Frame(right)
        self._plot_host = cf          # parent for the first-run empty overlay
        cf.grid(row=1, column=0, sticky='nsew')
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=1)
        self.fig    = Figure(figsize=(9, 6), dpi=100)
        self.ax     = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=cf)
        # The Tk canvas widget defaults to a white background; during a frozen
        # sash drag the newly exposed area shows that bare bg before matplotlib
        # repaints. Match it to the plot bg so the exposed strip stays dark.
        try:
            self.canvas.get_tk_widget().configure(
                bg=current_palette()['bg'], highlightthickness=0)
        except Exception:
            pass
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        # Centered zoom controls beneath the plot.
        zoombar = ttk.Frame(cf)
        zoombar.grid(row=1, column=0, pady=(3, 1))          # no sticky → centred
        _rv = ttk.Button(zoombar, text="⌂", width=2,
                         command=self._reset_plot_view)
        _rv.pack(side='left', padx=2)
        self._tip(_rv, "Reset the plot view — fit zoom/pan to the data "
                  "(same as View → Reset plot view).")
        _zb = ttk.Checkbutton(zoombar, text="⛶ Zoom", style='Toolbutton',
                              variable=self._zoom_mode_var,
                              command=self._toggle_zoom_tool)
        _zb.pack(side='left', padx=2)
        self._tip(_zb, "Zoom-to tool: drag a rectangle on the plot to zoom "
                  "into it. While active, gating tools are blocked.")
        _zin = ttk.Button(zoombar, text="+", width=2,
                          command=lambda: self._zoom_step(1 / 1.25))
        _zin.pack(side='left', padx=2)
        self._tip(_zin, "Zoom in (around the plot centre).")
        _zout = ttk.Button(zoombar, text="−", width=2,
                           command=lambda: self._zoom_step(1.25))
        _zout.pack(side='left', padx=2)
        self._tip(_zout, "Zoom out (around the plot centre).")

        self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.canvas.mpl_connect('pick_event',           self._on_canvas_pick)
        # Navigate the plot WITHOUT fighting the gating mouse: middle-drag
        # pans, the scroll wheel zooms around the cursor (left-click stays
        # gating, right-click stays vertex editing). View → Reset plot view
        # restores the data-fit view.
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self._pan_start = None
        # Zoom-to tool: when active, left-drag draws a zoom rectangle and the
        # gating tools are blocked (greyed). Mutually exclusive with gating.
        self._zoom_mode = False
        self._zoom_start = None
        self._zoom_rect_artist = None

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
        # Saved status text while a dropdown menu is showing per-entry help.
        self._status_before_menu = None
        # The status text now lives in the window-wide bottom status bar
        # (built below); the plot pane no longer carries its own copy.

        right.rowconfigure(2, weight=0)   # slider panel is fixed-height

        # ── Window-wide bottom status bar ────────────────────────────────
        statusbar = ttk.Frame(self, padding=(8, 3))
        statusbar.grid(row=2, column=0, columnspan=2, sticky='ew')
        statusbar.columnconfigure(0, weight=1)
        ttk.Label(statusbar, textvariable=self.status_var,
                  foreground='#555').grid(row=0, column=0, sticky='w')
        # Indeterminate "working" bar — shown during long background jobs
        # (clustering, UMAP/embeddings) so a slow run reads as busy, not hung.
        self._busy_bar = ttk.Progressbar(statusbar, mode='indeterminate',
                                         length=150)
        self._busy_bar.grid(row=0, column=1, padx=(10, 10))
        self._busy_bar.grid_remove()
        try:
            from . import __version__ as _ver
            ttk.Label(statusbar, text=f"v{_ver}", foreground='#8a8f98').grid(
                row=0, column=2, sticky='e')
        except Exception:
            pass

        self._build_menubar()            # full menubar (all vars now exist)
        self._bind_shortcuts()           # Ctrl+O/S/E/W, F1, Ctrl+Shift+A
        if self._primary:                # periodic crash-safe autosave
            self.after(self._AUTOSAVE_MS, self._periodic_autosave)
        # Responsive chrome: shrink the ttk control font as the window narrows
        # so the center/right control rows don't squash/clip on small screens.
        self._chrome_font_size = None
        self._chrome_resize_after = None
        self.bind('<Configure>', self._on_chrome_configure, add='+')
        self.after(250, lambda: self._apply_chrome_scale(force=True))
        # Smooth interior sash dragging: a paned-window sash drag fires
        # <B1-Motion> on the paned window itself (its panes handle their own
        # events), so freeze the heavy matplotlib redraw during the drag and do
        # one clean replot on release instead of re-rastering per pixel.
        for _pw in (self._main_paned, self._editor_paned):
            try:
                _pw.bind('<B1-Motion>', self._freeze_plot_redraw, add='+')
                _pw.bind('<ButtonRelease-1>', self._thaw_plot_redraw, add='+')
            except Exception:
                pass
        # Match the native title bar to the theme once the window is mapped
        # (Windows DWM dark caption; no-op elsewhere), and dark-theme every
        # child dialog's title bar as it opens.
        self.after(80, self._apply_titlebar_theme)
        try:
            self.bind_class('Toplevel', '<Map>', self._on_toplevel_mapped,
                            add='+')
        except Exception:
            pass

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
            # Drag SOURCE: Shift-drag a sample row OUT to another OpenFlo window
            # (cross-instance transfer). Gated on Shift in _on_tree_os_drag_init
            # so a plain drag stays the in-app regroup/workspace drag.
            try:
                self.gate_tv.drag_source_register(1, DND_FILES)  # type: ignore[attr-defined]
                self.gate_tv.dnd_bind('<<DragInitCmd>>',         # type: ignore[attr-defined]
                                      self._on_tree_os_drag_init)
            except Exception as exc:
                print(f"[DnD] drag-source register failed: {exc}", flush=True)

        self._render_placeholder()

    # ── Sample loading ───────────────────────────────────────────────────





    _expand_dropped_paths = staticmethod(_paths.expand_dropped_paths)  # → openflo.paths












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





    # ── Provenance / audit trail ─────────────────────────────────────────





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




    # Label columns that can be imported as populations, with a friendly
    # name and the sentinel value that means "unassigned" (skipped).
    LABEL_COLUMNS = {
        'cluster':      ('clusters', -1),
        'flowsom_meta': ('FlowSOM metaclusters', -1),
        'cell_cycle':   ('cell-cycle phases', 'NA'),
    }










    # ── Cell cycle (#cell-cycle) ─────────────────────────────────────────
    #
    # Runs FlowSample.cell_cycle (DNA-content G1/S/G2M model) on the active
    # (or all) sample(s), then surfaces each phase as a selectable
    # population via the 'category' gate kind — the same machinery clusters
    # use. A result window shows the DNA histogram + phase percentages.

    PHASE_COLORS = {'sub-G1': '#9a6324', 'G1': '#4363d8', 'S': '#3cb44b',
                    'G2M': '#e6194b', '>G2M': '#911eb4'}










    # ── Channel pickers ──────────────────────────────────────────────────




    # ── Axis controls (type-to-filter channel pickers) ──────────────────


    # ── Plotting ─────────────────────────────────────────────────────────


    # ── Gate model bookkeeping ────────────────────────────────────────────
    #
    # Storage is `self._gates: dict[str, dict]` keyed by an auto id.
    # Schema is shared with flow_pipeline.gate_to_mask (see that module).







    # ── Auto-clean method quick-edit (right-click menu) ─────────────────────










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
        self._apply_plot_theme()
        try:
            self.canvas.draw_idle()
        except Exception:
            pass



    # ── Treeview iid encoding (samples + gates share one tree) ──────────
    #
    # Sample rows:  'S:<sample_name>'
    # Gate rows:    'G:<sample_name>/<gate_id>'
    # Sample names usually have no ':' or '/' — FCS filenames don't — so
    # rsplit on '/' for the gate split keeps things robust.

    # Tree-row id encoders/decoders — pure logic lives in openflo.tree_ids;
    # these thin staticmethods preserve the existing self._x_iid() call sites.
    _sample_iid = staticmethod(_tids.sample_iid)
    _gate_iid = staticmethod(_tids.gate_iid)
    _trial_iid = staticmethod(_tids.trial_iid)
    _method_iid = staticmethod(_tids.method_iid)
    _subgroup_iid = staticmethod(_tids.subgroup_iid)
    _parse_iid = staticmethod(_tids.parse_iid)




















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



    _drop_suffix = staticmethod(_pm.drop_suffix)   # → openflo.plotmath










    # Embedding axis columns are abstract coordinates (can be negative, no
    # decades) — they must display on a LINEAR scale, not the global log
    # default meant for fluorescence intensity.
    _EMBED_AXIS_PREFIXES = ('UMAP', 'TSNE', 'TRIMAP', 'PACMAP', 'PHATE')



    _symlog_linthresh = staticmethod(_pm.symlog_linthresh)   # → openflo.plotmath






    _density_norm = staticmethod(_pm.density_norm)   # → openflo.plotmath





    # ── Backgating ──────────────────────────────────────────────────────────
    _BACKGATE_COLORS = ['#e8000b', '#1ac938', '#023eff', '#ff7c00',
                        '#8b2be2', '#f14cc1', '#00d7ff', '#ffb000']








    _in_box = staticmethod(_pm.in_box)   # → openflo.plotmath




    _hist_bin_edges = staticmethod(_pm.hist_bin_edges)   # → openflo.plotmath

    # ── Gates (draggable threshold lines + shape overlays) ───────────────


    _ellipse_params = staticmethod(_pm.ellipse_params)   # → openflo.plotmath


    # ── Highlight overlay (3-way display mode == 'highlight') ────────────














    _ellipse_geom = staticmethod(_pm.ellipse_geom)            # → openflo.plotmath
    _point_segment_dist = staticmethod(_pm.point_segment_dist)  # → openflo.plotmath
    _gid_from_hit = staticmethod(_pm.gid_from_hit)            # → openflo.plotmath







    # ── Display mode (All / Highlight / Filter) ─────────────────────────
    _DISPLAY_MODES = ('all', 'highlight', 'filter')
    _DISPLAY_LABELS = {'all': 'All events', 'highlight': 'Highlight gated',
                       'filter': 'Filter to gated'}



    # ── Smooth pane resizing ────────────────────────────────────────────


    # ── Responsive chrome scaling ───────────────────────────────────────


    def _tool_tip_text(self, base):
        """Tooltip text for a gating tool — swaps to a 'blocked' message while
        the Zoom-to tool is active."""
        if getattr(self, '_zoom_mode', False):
            return ("Blocked — the Zoom tool is active. Turn off Zoom (⛶) to "
                    "use the gating tools again.")
        return base







    # ── Tree press / motion / release (click vs. drag-reparent) ──────────















    # ── Clipboard / context-menu / OS drag-drop ──────────────────────────






















    def _tip(self, widget, text):
        """Attach a hover tooltip to a widget, gated by the View → Show hover
        tips toggle. No-op on failure."""
        try:
            _ToolTip(widget, text, lambda: self._tooltips_enabled.get())
        except Exception:
            pass






















    # Embedding picker → (FlowSample method, axis-column prefix).
    _EMBEDDINGS = {
        'UMAP':   ('run_umap', 'UMAP'),
        't-SNE':  ('run_tsne', 'TSNE'),
        'TriMap': ('run_trimap', 'TRIMAP'),
        'PaCMAP': ('run_pacmap', 'PACMAP'),
        'PHATE':  ('run_phate', 'PHATE'),
    }
    # Backend module each embedding needs (UMAP + t-SNE are core deps; the
    # rest are the optional `embed` extra).
    _EMBED_BACKEND = {'UMAP': 'umap', 't-SNE': 'sklearn', 'TriMap': 'trimap',
                      'PaCMAP': 'pacmap', 'PHATE': 'phate'}

    @classmethod
    def _available_embeddings(cls):
        """``(available, missing)`` embedding names by whether their backend is
        importable on this install (checked with ``find_spec`` — no heavy
        import). Used to keep the Cluster dialog's list honest so a user can't
        pick a backend that would silently produce nothing."""
        import importlib.util as _u
        avail, missing = [], []
        for name in cls._EMBEDDINGS:
            mod = cls._EMBED_BACKEND.get(name, '')
            (avail if (mod and _u.find_spec(mod) is not None)
             else missing).append(name)
        return avail, missing

    # ── Busy indicator for long background jobs ─────────────────────────









    # ── Mode / tool / selector lifecycle ─────────────────────────────────




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






    # ── Histogram slider gate ────────────────────────────────────────────










    # ── Per-channel axis scale + range ────────────────────────────────────





    # ── Figure layout / multi-panel export ───────────────────────────────



    _short_sample = staticmethod(short_label)   # → openflo.ui_logic







    # ── Templates (save / load) ──────────────────────────────────────────



    # ── Menubar ─────────────────────────────────────────────────────────

    # Per-entry help for the dropdown menus. Tk can't float a tooltip over a
    # posted menu (the menu grabs the pointer and its entries aren't separate
    # widgets), so as the highlighted entry changes we surface its description
    # in the status bar instead — the standard desktop pattern. Keyed by entry
    # label; labels are unique enough across menus to share one flat table.
    _MENU_ITEM_HELP = {
        # File
        "Add FCS…": "Load one or more .fcs files into the session.",
        "Load CSV…": "Load a processed-data CSV (e.g. cluster / UMAP columns) as a sample.",
        "Load example dataset": "Generate & load a synthetic PBMC dataset to try the app.",
        "Generate dataset…": "Generate a synthetic dataset (PBMC / differentiation / beads…) and load it.",
        "Quick preview…": "Quick density-scatter QC of a raw FCS before the full pipeline.",
        "Save plot as image…": "Save the current plot to PNG / SVG / PDF (white background).",
        "Open session…": "Open a saved .flowsession — samples, gates and view.",
        "Open Recent": "Reopen a recently used session.",
        "Save session…": "Save the current samples, gates and view to a .flowsession.",
        "Export → FlowJo .wsp…": "Write the current gates out as a FlowJo workspace (.wsp).",
        "Analysis report (HTML)…": "Export an HTML report of gates, plots and stats.",
        "Close": "Close OpenFlo (the session autosaves).",
        # Edit
        "Undo": "Undo the last change.",
        "Redo": "Redo the last undone change.",
        "Clear gate": "Delete the selected gate and its children.",
        "Clear all gates": "Remove every gate from all samples.",
        "Copy gates to…": "Copy the active sample's gates onto other samples.",
        "Populations…": "Import label / cluster columns as populations.",
        "Add singlet gate": "Add an FSC-A vs FSC-H singlet gate to the active sample.",
        "FMO gating…": "Place threshold gates from FMO control samples (per percentile).",
        "Auto-clean gate": "Add an auto-clean gate (margins / debris / dead).",
        "Preferences…": "Theme and hover-tip settings in one place.",
        # View
        "Display": "How gates affect the plot: show all, highlight, or filter to gated.",
        "Reset plot view": "Reset zoom/pan (wheel = zoom, middle-drag = pan).",
        "Pipeline Workspace": "Show or hide the batch clustering workspace panel.",
        "Show log / console": "Toggle the log / Python console pane.",
        "Show hover tips": "Toggle these hover tooltips and menu hints.",
        "Dark figures in pop-ups": "Render preview/export figures on a dark background.",
        "New windows open at": "Spawn dialogs at a fixed corner of the main window.",
        "Dock all panels": "Re-dock any popped-out panels back into the main window.",
        "Theme": "Switch between the light and dark interface theme.",
        "All events": "Show every event; gates just outline regions.",
        "Highlight gated": "Grey the base events and colour each gate's events.",
        "Filter to gated": "Show only events inside the enabled gates.",
        "Light": "Light interface theme (white plot).",
        "Dark": "Dark interface theme, white plot canvas.",
        "Midnight (dark plot)": "Dark interface with a dark plot canvas too.",
        # Analyze
        "Statistics…": "FlowJo-style stats table — counts, frequencies, MFI.",
        "Frequencies…": "Population frequencies across samples (Prism-ready export).",
        "Expression…": "Per-channel expression across samples.",
        "Group comparison…": "Kruskal-Wallis + pairwise tests across trial groups.",
        "Compare embeddings…": "Run UMAP / t-SNE / PHATE side by side on a sample.",
        "Methods & provenance…": "Paper-ready methods paragraph + reproducibility manifest.",
        "Absolute counts…": "Counting-bead cells/µL calculator.",
        "Export populations (FCS)…": "Write each gated population to its own FCS file.",
        "Voltage optimization…": "PMT voltage / stain-index titration with recommendations.",
        "Compare FlowJo workspace…": "Re-apply a .wsp and compare gate counts vs FlowJo.",
        "FCS inspector…": "View a raw FCS file's channels, keywords and spillover.",
        "Watch folder…": "Auto-load new .fcs files dropped into a folder (toggle).",
        "Sample QC…": "Sample-level quality control (EMD / MDS).",
        "Cluster…": "Unsupervised clustering with an optional embedding.",
        "Cell cycle…": "DNA-content G1 / S / G2-M cell-cycle modelling.",
        "Trajectory…": "Pseudotime / trajectory analysis.",
        "Annotate…": "Label clusters or populations.",
        "SOM tree…": "FlowSOM self-organising-map tree.",
        # Tools
        "Compensation…": "Edit or import the compensation matrix.",
        "Compensation QC…": "Spillover heatmap + metrics for the active sample's matrix.",
        "Gating tree diagram…": "View the active sample's gating hierarchy as a diagram.",
        "Transforms…": "Per-channel display transforms (logicle, biexp, log…).",
        "Calibration…": "MESF bead calibration.",
        "Batch-norm (CytoNorm)": "Batch-normalise samples across runs.",
        "Spectral unmix…": "Reference-spectra spectral unmixing (per sample group).",
        "Figure layout…": "Compose a multi-panel figure for export.",
        "Templates": "Apply a bundled or saved gating template.",
        "Save template…": "Save the current gates as a reusable template.",
        "History / audit…": "View the analysis audit trail.",
        # Help
        "Check for updates…": "Check GitHub for a newer OpenFlo release.",
        "Report a problem…": "Open a tokenised (de-identified) error report to submit.",
        "Documentation": "Open the OpenFlo README / docs in your browser.",
        "Keyboard shortcuts": "Show the keyboard-shortcut reference.",
        "About OpenFlo": "Version, license, citation and credits.",
    }



    # ── Update check (Help menu) ────────────────────────────────────────








    _gate_channels = staticmethod(_gating.gate_channels)   # → openflo.gating





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





    _sidecar_safe_name = staticmethod(_paths.safe_sidecar_name)  # → openflo.paths











    # ── In-app log pane + Python console ──────────────────────────────────
    # ── Crash handling ──────────────────────────────────────────────────





















    # ── Window geometry persistence ─────────────────────────────────────


    # ── Recent sessions ─────────────────────────────────────────────────








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

    _population_path = staticmethod(_gating.population_path)   # → openflo.gating

    @classmethod
    def _population_stats(cls, sample_name, df, gates, order,
                          channel_labels, channels, want, select=None):
        """Per-population statistic rows for one sample. Thin wrapper over the
        pure openflo.gating.population_stats (passing this class's STAT_CHAN)."""
        return _gating.population_stats(
            sample_name, df, gates, order, channel_labels, channels, want,
            cls.STAT_CHAN, select=select)
























# ══════════════════════════════════════════════════════════════════════════════
# CELL-CYCLE RESULT WINDOW
# ══════════════════════════════════════════════════════════════════════════════











_FLAT_IND = {'made': set(), 'imgs': {}}


def _install_flat_indicators(style, mode, pal):
    """Replace clam's jagged check/radio indicators with clean supersampled
    PIL shapes so the square checkbox and round radio share one crisp style:
    panel fill + muted outline, accent fill (radio dot / check tick) when on.
    Best-effort — on any failure the default indicators are left in place."""
    try:
        from PIL import Image, ImageDraw, ImageTk
    except Exception:
        return

    def _rgb(h):
        h = h.lstrip('#')
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    panel, muted, accent, bg = (_rgb(pal['panel']) + (255,),
                                _rgb(pal['muted']) + (255,),
                                _rgb(pal['accent']) + (255,),
                                _rgb(pal['bg']) + (255,))
    scale, size = 4, 15
    D = size * scale
    pad = scale

    def _photo(img):
        return ImageTk.PhotoImage(img.resize((size, size),
                                             Image.Resampling.LANCZOS))

    def _radio(selected):
        img = Image.new('RGBA', (D, D), bg)
        d = ImageDraw.Draw(img)
        d.ellipse([pad, pad, D - pad, D - pad], fill=panel, outline=muted,
                  width=scale)
        if selected:
            m = D * 0.30
            d.ellipse([m, m, D - m, D - m], fill=accent)
        return _photo(img)

    def _check(selected):
        img = Image.new('RGBA', (D, D), bg)
        d = ImageDraw.Draw(img)
        r = D * 0.22
        d.rounded_rectangle([pad, pad, D - pad, D - pad], radius=r,
                            fill=panel, outline=muted, width=scale)
        if selected:
            d.line([(D * 0.28, D * 0.52), (D * 0.44, D * 0.68),
                    (D * 0.74, D * 0.32)], fill=accent,
                   width=int(scale * 1.6), joint='curve')
        return _photo(img)

    try:
        for kind, drawer, wclass, elabel in (
                ('flatradio', _radio, 'TRadiobutton', 'Radiobutton'),
                ('flatcheck', _check, 'TCheckbutton', 'Checkbutton')):
            name = f'{kind}_{mode}'
            if name not in _FLAT_IND['made']:
                off_i, on_i = drawer(False), drawer(True)
                _FLAT_IND['imgs'][name] = (off_i, on_i)   # keep refs alive
                style.element_create(name + '.indicator', 'image', off_i,
                                     ('selected', on_i), sticky='', border=0,
                                     padding=0)
                _FLAT_IND['made'].add(name)
            style.layout(wclass, [
                (f'{elabel}.padding', {'sticky': 'nswe', 'children': [
                    (name + '.indicator', {'side': 'left', 'sticky': ''}),
                    (f'{elabel}.label',
                     {'side': 'left', 'sticky': 'nswe'})]})])
    except Exception as exc:
        print(f"[theme] flat indicators: {exc}", flush=True)


class _ToolTip:
    """Lightweight hover tooltip for a Tk widget. Shows ``text`` in a small
    themed popup after a short hover; gated by ``enabled()`` (a callable) so a
    single View toggle silences them all. Best-effort throughout."""

    def __init__(self, widget, text, enabled, delay=550):
        self.widget = widget
        self.text = text
        self.enabled = enabled
        self.delay = delay
        self._after = None
        self._tip = None
        widget.bind('<Enter>', self._schedule, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _schedule(self, _e=None):
        self._cancel()
        if not self.enabled():
            return
        try:
            self._after = self.widget.after(self.delay, self._show)
        except Exception:
            pass

    def _show(self):
        if self._tip is not None or not self.enabled():
            return
        # text may be a callable (resolved at hover time) so a control can
        # show a context-dependent message — e.g. "blocked by the Zoom tool".
        text = self.text() if callable(self.text) else self.text
        if not text:
            return
        try:
            x = self.widget.winfo_rootx() + 14
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            pal = current_palette()
            self._tip = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f'+{x}+{y}')
            tk.Label(tw, text=str(text), justify='left', bg=pal['panel'],
                     fg=pal['fg'], relief='solid', bd=1, padx=6, pady=4,
                     wraplength=320, font=('Segoe UI', 8)).pack()
        except Exception:
            self._tip = None

    def _hide(self, _e=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None

    def _cancel(self):
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None


def apply_theme(root, mode='light'):
    """Apply the chosen chrome theme ('light' or 'dark') across the whole app:
    the cross-platform ``clam`` ttk theme, one readable base font, sane
    padding, roomier Treeviews / tabs, and the palette's colours. Matplotlib
    plots and explicit gate colours are untouched; every step is best-effort.
    Returns the active palette dict; re-callable to switch themes live."""
    pal = set_active_palette(THEMES.get(mode, THEMES['light']))
    BG, PANEL, FG = pal['bg'], pal['panel'], pal['fg']
    ACC, ACCFG, BORDER = pal['accent'], pal['accfg'], pal['border']
    ACTIVE = pal['active']
    try:
        import tkinter.font as tkfont
        from tkinter import ttk
        style = ttk.Style(root)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        fam = 'Segoe UI' if 'Segoe UI' in set(tkfont.families(root)) else None
        if fam:                              # retune the named fonts in place
            for nm in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont',
                       'TkHeadingFont'):
                try:
                    tkfont.nametofont(nm).configure(family=fam, size=10)
                except Exception:
                    pass
        font = (fam or 'TkDefaultFont', 10)
        bold = (fam or 'TkDefaultFont', 10, 'bold')
        # clam draws widget bevels with light/dark edge colours that default
        # to near-white — stark in dark mode. Tie them to the palette border so
        # edges read as a neutral mid-grey instead of a bright outline.
        style.configure('.', background=BG, foreground=FG, font=font,
                        bordercolor=BORDER, lightcolor=BORDER,
                        darkcolor=BORDER, fieldbackground=PANEL)
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG)
        style.configure('TLabelframe', background=BG, bordercolor=BORDER)
        style.configure('TLabelframe.Label', background=BG, foreground=FG,
                        font=bold)
        style.configure('TButton', padding=(10, 5),
                        background=PANEL, foreground=FG,
                        bordercolor=BORDER, lightcolor=BORDER,
                        darkcolor=BORDER, focuscolor=BORDER)
        style.map('TButton',
                  background=[('active', ACTIVE)],
                  foreground=[('active', FG)],
                  bordercolor=[('active', ACTIVE), ('focus', ACTIVE)],
                  lightcolor=[('active', ACTIVE)],
                  darkcolor=[('active', ACTIVE)])
        # Check/radio indicators: flat (no 3-D bevel ring, which textures the
        # round radio edges in dark mode), filled with the panel colour and the
        # accent when selected. Both styled identically so the circle reads as
        # clean as the square.
        for _ind in ('TCheckbutton', 'TRadiobutton'):
            # bordercolor = PANEL too: the indicator's outline blends into its
            # own fill, so the round radio has no light bevel ring (the jagged
            # "contour"); selected shows the accent.
            style.configure(_ind, background=BG, foreground=FG,
                            indicatorcolor=PANEL, indicatorrelief='flat',
                            indicatormargin=2, bordercolor=PANEL,
                            lightcolor=PANEL, darkcolor=PANEL)
            style.map(_ind,
                      background=[('active', BG)],
                      indicatorcolor=[('selected', ACC), ('pressed', ACC)],
                      bordercolor=[('', PANEL)],
                      lightcolor=[('', PANEL)], darkcolor=[('', PANEL)])
        # Swap clam's jagged check/radio indicators for clean PIL-drawn shapes.
        _install_flat_indicators(style, mode, pal)
        style.configure('TMenubutton', padding=(8, 4),
                        background=PANEL, foreground=FG)
        style.configure('TEntry', padding=3, fieldbackground=PANEL,
                        foreground=FG, insertcolor=FG)
        style.configure('TCombobox', padding=3, fieldbackground=PANEL,
                        foreground=FG)
        style.map('TCombobox', fieldbackground=[('readonly', PANEL)],
                  foreground=[('readonly', FG)])
        style.configure('TNotebook', background=BG, bordercolor=BORDER)
        style.configure('TNotebook.Tab', padding=(12, 5),
                        background=BG, foreground=FG)
        style.map('TNotebook.Tab', background=[('selected', PANEL)],
                  foreground=[('selected', FG)])
        style.configure('Treeview', background=PANEL, fieldbackground=PANEL,
                        foreground=FG, rowheight=24, bordercolor=BORDER)
        style.configure('Treeview.Heading', font=bold, padding=(6, 4),
                        background=ACTIVE, foreground=FG, relief='flat')
        # Without an explicit map, clam paints its own (light) 'active' heading
        # background on hover, which clashes with FG text and reads illegible —
        # especially in dark/midnight. Pin a subtle themed hover that keeps the
        # text on FG.
        style.map('Treeview.Heading',
                  background=[('active', BORDER), ('pressed', BORDER)],
                  foreground=[('active', FG), ('pressed', FG)],
                  relief=[('active', 'flat'), ('pressed', 'flat')])
        style.map('Treeview', background=[('selected', ACC)],
                  foreground=[('selected', ACCFG)])
        style.configure('TPanedwindow', background=BG)
        style.configure('TSeparator', background=BORDER)
        # Larger surfaces (scrollbars) sit on a darker base so they read as
        # recessed, with a slightly lighter, grabbable thumb.
        TROUGH, THUMB = pal['trough'], pal['thumb']
        for _sb in ('Vertical.TScrollbar', 'Horizontal.TScrollbar'):
            style.configure(_sb, background=THUMB, troughcolor=TROUGH,
                            bordercolor=TROUGH, lightcolor=TROUGH,
                            darkcolor=TROUGH, arrowcolor=FG)
            style.map(_sb, background=[('active', BORDER)])
        # tk (non-ttk) menus read these from the option DB at creation time.
        try:
            root.option_add('*tearOff', False)
            root.option_add('*Menu.background', PANEL)
            root.option_add('*Menu.foreground', FG)
            root.option_add('*Menu.activeBackground', ACC)
            root.option_add('*Menu.activeForeground', ACCFG)
            # Without this, disabled items (e.g. a greyed-out "Paste") fall back
            # to the OS engraved-grey style, which renders garbled on a dark menu.
            root.option_add('*Menu.disabledForeground', pal['muted'])
            # Flat dropdowns with no Tk-drawn border / raised highlight (any
            # remaining hairline is the OS popup frame, not Tk-controllable).
            root.option_add('*Menu.relief', 'flat')
            root.option_add('*Menu.borderWidth', 0)
            root.option_add('*Menu.activeBorderWidth', 0)
            # Dialog Toplevels grid ttk widgets directly on themselves, so the
            # Toplevel's own (otherwise default-light) bg shows through; tie it
            # to the theme. Applies to Toplevels created after this call.
            root.option_add('*Toplevel.background', BG)
            # tk (non-ttk) widgets default to a white background that ignores
            # the ttk theme — the source of the glaring white space in dialogs
            # under the dark themes. Theme them here so Canvas areas, Text
            # panes and Listboxes follow the palette. (Applies to widgets
            # created after this runs — i.e. every dialog opened afterwards.)
            root.option_add('*Canvas.background', BG)
            root.option_add('*Text.background', PANEL)
            root.option_add('*Text.foreground', FG)
            root.option_add('*Text.insertBackground', FG)
            root.option_add('*Listbox.background', PANEL)
            root.option_add('*Listbox.foreground', FG)
            root.option_add('*Listbox.selectBackground', ACC)
            root.option_add('*Listbox.selectForeground', ACCFG)
        except Exception:
            pass
    except Exception as exc:
        print(f"[theme] {exc}", flush=True)
    return pal


def _make_app_icon():
    """Draw OpenFlo's window / taskbar icon — a flow-cytometry density scatter
    (a blue→red dot cloud) on a rounded dark-slate tile that reads on both
    light and dark taskbars. Returns a list of ``ImageTk.PhotoImage`` at a few
    sizes (largest first) for ``iconphoto``, or ``[]`` if Pillow is missing.
    Deterministic (fixed seed) so the mark never changes between launches.

    Must be called after the Tk root exists — PhotoImages bind to its interp."""
    try:
        from PIL import Image, ImageChops, ImageDraw, ImageTk
    except Exception:
        return []
    import math
    import random

    TILE = (31, 36, 48, 255)        # dark slate tile
    BORDER = (70, 80, 100, 255)
    # cold (blue) → hot (red) ramp; dense core renders hot, like a 2-D density.
    STOPS = [(0.00, (46, 102, 222)), (0.35, (46, 178, 196)),
             (0.58, (104, 200, 96)), (0.78, (240, 196, 52)),
             (1.00, (228, 64, 48))]

    def _ramp(t):
        for i in range(len(STOPS) - 1):
            t0, c0 = STOPS[i]
            t1, c1 = STOPS[i + 1]
            if t <= t1:
                f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))
        return STOPS[-1][1]

    def _draw(size):
        ss = 4                        # supersample, then LANCZOS down
        D = size * ss
        pad = D * 0.045
        radius = D * 0.22
        # Rounded tile.
        tile = Image.new('RGBA', (D, D), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        td.rounded_rectangle([pad, pad, D - pad, D - pad], radius=radius,
                             fill=TILE, outline=BORDER,
                             width=max(1, int(D * 0.012)))
        # Scatter on its own layer, then clip to the tile so nothing spills.
        dots = Image.new('RGBA', (D, D), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dots)
        rng = random.Random(7)        # fixed seed → stable icon
        cx, cy = D * 0.5, D * 0.54
        sigma = D * 0.17
        n = 60 if size >= 32 else 26
        dot = D * (0.045 if size >= 32 else 0.075)
        keep = D * 0.34               # core radius — keeps dots inside the tile
        placed = 0
        while placed < n:
            x = rng.gauss(cx, sigma)
            y = rng.gauss(cy, sigma)
            dist = math.hypot(x - cx, y - cy)
            if dist > keep:
                continue
            t = max(0.0, min(1.0, 1.0 - dist / keep))   # core = hot
            dd.ellipse([x - dot, y - dot, x + dot, y + dot],
                       fill=_ramp(t) + (255,))
            placed += 1
        # Clip the scatter to the tile's rounded silhouette.
        mask = Image.new('L', (D, D), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [pad, pad, D - pad, D - pad], radius=radius, fill=255)
        dots.putalpha(ImageChops.multiply(dots.getchannel('A'), mask))
        tile.alpha_composite(dots)
        return ImageTk.PhotoImage(
            tile.resize((size, size), Image.Resampling.LANCZOS))

    try:
        return [_draw(s) for s in (256, 48, 32, 16)]
    except Exception:
        return []


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

    # Windows taskbar: claim our own AppUserModelID BEFORE any window exists.
    # Without it the taskbar button groups under python.exe and shows Python's
    # generic feather; with it, Windows uses our iconphoto for the taskbar too.
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                'Openflo.GateEditor')
        except Exception as exc:
            print(f"[taskbar] AppUserModelID: {exc}", flush=True)

    root = _APP_BASE()                   # TkinterDnD.Tk when available → OS file-drop
    root.withdraw()                      # hidden root; the editor is the UI
    # Apply the user's saved chrome theme before widgets build.
    apply_theme(root, read_prefs().get('theme', 'light'))
    # Honour the saved GPU-acceleration preference before any sample loads
    # (no-op + auto-off when no usable GPU backend is present). The backend
    # preference (auto/cupy/torch/off) selects CuPy (NVIDIA) or the portable
    # PyTorch backend (NVIDIA/AMD/Intel/Apple) before the flag is applied.
    try:
        from . import gpu_accel
        _prefs = read_prefs()
        gpu_accel.set_backend(str(_prefs.get('gpu_backend', 'auto')))
        gpu_accel.set_enabled(bool(_prefs.get('use_gpu', False)))
    except Exception as exc:
        print(f"[gpu] {exc}", flush=True)
    # Replace Tk's default feather logo with OpenFlo's scatter mark. default=1
    # makes it the icon for the root and every Toplevel created afterwards.
    try:
        _icons = _make_app_icon()
        if _icons:
            root.iconphoto(True, *_icons)  # type: ignore[arg-type]  # PIL PhotoImage
            root._app_icons = _icons  # type: ignore[attr-defined]  # keep refs alive
    except Exception as exc:
        print(f"[icon] {exc}", flush=True)
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
