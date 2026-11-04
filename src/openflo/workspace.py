"""Pipeline Workspace — single-window grouping / comparison.

The workspace lives docked in the gate editor behind a draggable sash. Items
(samples, gate *leaves*/populations, or whole trials) are dragged in from the
editor's Samples & Gates tree. The workspace is a single tree split into three
channels:

    c1 (#0 tree)      c2 (comp)              c3 (fmo)
    group → samples   metadata / comp-beads  FMOs

  • Groups bundle items (e.g. "all of trial 1"). A group's comp/FMO cascades to
    its members; an item can override its own.
  • Compensation precedence: item override → group override → sample's own
    metadata matrix → none. Dropped comp *beads* OVERRIDE the metadata matrix
    (they become the basis for a later-derived matrix). Each item/group needs a
    matrix OR beads; rows with neither are flagged "⚠" (distinct from a group's
    neutral "—" no-override).
  • FMOs are optional (they also feed FMO-based auto-gating later).
  • Display model: an item shows in the preview when its own ☑ AND its group's
    ☑ are on (``WorkspaceModel.displayed_items``).

Beads/FMOs are supplied two ways: drag bead/FMO FCS from the editor onto a
row's comp/fmo column, or the "Set comp beads…" / "Set FMOs…" picker buttons.
Only file references travel — no event data — so many trials stay cheap.

Architecture — model / view split:
  • :mod:`openflo.workspace_run` — the run logic and ``WorkspaceModel``
                                  (plain data + logic, no Tk; unit-tested).
  • ``PipelineWorkspaceView``    — disposable Tk renderer over a model.
  • ``WorkspacePanel``           — hosts the single view + Pop-out / Import-all.
"""

from __future__ import annotations

import collections
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any

from .errors import note as note_handled
from .tool_window import MIN_CUSTOM_EVENTS, bind_right_click, scrolled_body
from .ui_errors import show_error
from .workspace_run import (
    EXTRA_EMBEDDINGS,
    PROGRESS_MARKER,
    WorkspaceModel,
    batch_failure_text,
    build_drop_payload,
    build_run_units,
    cap_events,
    compute_run,
    default_run_cfg,
    embedding_available,
    embedding_unavailable,
    extract_editor_context,
    extract_one_context,
    find_run_outputs,
    merge_unit_cfg,
    newer_version_note,
    own_process_group,
    prepare_unit,
    proper_run_channels,
    run_events_text,
    run_label,
    signal_process_group,
    subprocess_main,
)

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Callable, Sequence
    from typing import Literal

    # A tree row the view acts on: ('group', 'G1') or ('item', 'M3').
    _Target = tuple[Literal['group', 'item'], str]
    # An editor tree node dropped in: ('sample', name) or ('gate', name, gid).
    _Node = tuple[str, ...]
    # A Tk event from a binding on one of the workspace's widgets.
    _Event = tk.Event[tk.Misc]
    # A no-argument callback whose return value is ignored.
    _Hook = Callable[[], object]

EVENTS_CSV_TIP = (
    "Also save each run unit's per-event table (markers, cluster, source "
    "and embedding columns) as <label>_events.csv, to open in Excel or "
    "another program. Off: only the binary <label>_events.npz is written, "
    "which saves about 50x faster; 'Load in editor' reads either.")


def _tip(editor: Any, widget: tk.Misc, text: str | Callable[[], str]) -> None:
    """Attach a hover tooltip to a workspace control via the editor's gated
    tooltip helper (respects View → Show hover tips). No-op without an editor."""
    if editor is not None and hasattr(editor, '_tip'):
        try:
            editor._tip(widget, text)
        except Exception:  # noqa: BLE001 - hover tip is cosmetic
            pass


# The run logic moved to workspace_run; these names still resolve here, with a
# DeprecationWarning, for code that imported them from this module.
_MOVED_TO_RUN = {
    'DEFAULT_RUN_CFG': 'DEFAULT_RUN_CFG',
    'WORKSPACE_FORMAT': 'WORKSPACE_FORMAT',
    'WORKSPACE_SCHEMA': 'WORKSPACE_SCHEMA',
    '_BATCH_FAILURES_SHOWN': '_BATCH_FAILURES_SHOWN',
    '_COMP_NAME_RE': '_COMP_NAME_RE',
    '_CREATE_NEW_CONSOLE': '_CREATE_NEW_CONSOLE',
    '_CREATE_NO_WINDOW': '_CREATE_NO_WINDOW',
    '_DAY_TOKEN_RE': '_DAY_TOKEN_RE',
    '_DETACHED_PROCESS': '_DETACHED_PROCESS',
    '_GENERIC_HOST_RE': '_GENERIC_HOST_RE',
    '_PROG': 'PROGRESS_MARKER',
    '_batch_failure_text': 'batch_failure_text',
    '_blank_result': '_blank_result',
    '_cap_events': 'cap_events',
    '_display_transforms': '_display_transforms',
    '_item_from_json': '_item_from_json',
    '_item_to_json': '_item_to_json',
    '_no_window_creationflags': '_no_window_creationflags',
    '_own_process_group': 'own_process_group',
    '_plot_embedding': '_plot_embedding',
    '_signal_process_group': 'signal_process_group',
    '_source_tags': '_source_tags',
    '_subprocess_main': 'subprocess_main',
    '_suppress_child_windows': '_suppress_child_windows',
    'comp_iid': 'comp_iid',
    'comp_summary': 'comp_summary',
    'derive_trial_name': 'derive_trial_name',
    'gate_chain': 'gate_chain',
    'gate_iid': 'gate_iid',
    'gate_is_leaf': 'gate_is_leaf',
    'gate_path': 'gate_path',
    'is_comp_sample': 'is_comp_sample',
    'natural_sort_key': 'natural_sort_key',
    'parse_iid': 'parse_iid',
    'prepare_run': 'prepare_run',
    'resolve_run_sample': 'resolve_run_sample',
    'run_workspace_item': 'run_workspace_item',
    'sample_iid': 'sample_iid',
    'trial_day_number': 'trial_day_number',
}


def __getattr__(name: str) -> Any:
    new = _MOVED_TO_RUN.get(name)
    if new is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import warnings
    warnings.warn(f'workspace.{name} is deprecated and will be removed in a '
                  f'future release; use workspace_run.{new}',
                  DeprecationWarning, stacklevel=2)
    from . import workspace_run
    return getattr(workspace_run, new)


# ── View (Tk renderer over a model) ────────────────────────────────────────

_COL = {'#0': 'tree', '#1': 'on', '#2': 'comp', '#3': 'fmo'}


class PipelineWorkspaceView(ttk.Frame):
    """Renders a ``WorkspaceModel`` as a 3-column tree (pops | comp | fmo) with
    a 👁 display column. Disposable: pop-out / re-dock rebuild over the model."""

    def __init__(self, parent: tk.Misc, model: WorkspaceModel, *,
                 editor: Any = None,
                 on_status: Callable[[str], object] | None = None,
                 on_change: Callable[[PipelineWorkspaceView], object] | None = None,
                 on_run: _Hook | None = None, on_cancel: _Hook | None = None,
                 on_results: _Hook | None = None,
                 on_before_change: _Hook | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self._editor = editor
        self._on_status = on_status
        self._on_change = on_change
        self._on_run = on_run
        self._on_cancel = on_cancel
        self._on_results = on_results
        self._on_before_change = on_before_change
        self._build()
        self._render()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Condensed toolbar: one primary button + grouped ▾ menus. The ~11
        # actions as plain buttons would take two rows and clip off the right
        # edge when the panel is docked narrow (a normal monitor). Grouping the
        # secondary actions into dropdown menus keeps them all reachable in a
        # fraction of the width.
        bar = ttk.Frame(self)
        # sticky='ew', not centred: when the panel is narrower than the bar,
        # a centred frame is clipped at BOTH edges, so the first and last
        # controls are damaged at once. Anchored, overflow is predictable and
        # only the trailing control can be affected.
        bar.grid(row=0, column=0, sticky='ew', pady=(0, 3))

        _ed = self._editor

        def _menubtn(parent: tk.Misc, text: str,
                     items: Sequence[tuple[str | None, _Hook | None]],
                     tip: str, **pk: Any) -> ttk.Menubutton:
            mb = ttk.Menubutton(parent, text=text)
            menu = tk.Menu(mb, tearoff=0)
            for label, cmd in items:
                if label is None:
                    menu.add_separator()
                else:
                    # Only the separator row (label None) has no command.
                    menu.add_command(label=label, command=cmd)  # pyright: ignore[reportArgumentType]
            mb['menu'] = menu
            mb.pack(side='left', **pk)
            _tip(_ed, mb, tip)
            return mb

        _menubtn(bar, "Import / Edit", [
            ("Import all samples", self._import_all),
            ("Import selected sample(s)", self._import_selected),
            (None, None),
            ("Remove selected", self._remove_selected),
            ("Clear workspace", self._clear),
        ], "Import loaded samples (all, or just those selected in the editor's "
           "tree), or remove / clear workspace items.", padx=(0, 3))
        _gb = ttk.Button(bar, text="＋ Group", width=8, command=self._new_group)
        _gb.pack(side='left')
        _tip(_ed, _gb, "Create an empty group; drag samples / gate leaves into "
             "it. Each group is co-embedded as one run unit.")
        _menubtn(bar, "Group ▾", [
            ("Group selected", self._group_selected),
            ("Group params…", self._group_params),
            (None, None),
            ("Set comp / beads…", self._set_comp_picker),
            ("Set FMOs…", self._set_fmo_picker),
            ("Markers…", self._pick_markers),
        ], "Group the selection, override its run params, or assign "
           "compensation / FMO / marker channels.", padx=(3, 0))
        # Recipe / workspace save-load and batch-folder live in the panel's top
        # bar (Save / Load menu) so they're shared with the popped-out view.

        # Run bar — Phenograph / UMAP / TriMap modifiers + Run/Cancel (M2).
        # Split into a parameters row and an actions row so Run / Cancel /
        # Results are never clipped when the panel is docked narrow.
        runbar = ttk.Frame(self)
        runbar.grid(row=1, column=0, sticky='ew', pady=(0, 3))
        cfg = self.model.run_cfg
        self._method_var = tk.StringVar(value=str(cfg.get('method', 'phenograph')))
        self._res_var = tk.StringVar(value=str(cfg.get('resolution', 1.0)))
        self._k_var = tk.StringVar(value=str(cfg.get('k', 30)))
        self._max_var = tk.StringVar(value=str(cfg.get('max_events', 5000)))
        self._seed_var = tk.StringVar(value=str(cfg.get('seed', 42)))
        self._umap_var = tk.BooleanVar(value=bool(cfg.get('umap', True)))
        self._tsne_var = tk.BooleanVar(value=bool(cfg.get('tsne', False)))
        # An [embed] backend that isn't installed starts unticked (and its
        # box is disabled below): ticked, it ran as nothing at all.
        self._phate_var = tk.BooleanVar(value=bool(
            cfg.get('phate', False) and embedding_available('phate')))
        self._trimap_var = tk.BooleanVar(value=bool(
            cfg.get('trimap', True) and embedding_available('trimap')))
        self._pacmap_var = tk.BooleanVar(value=bool(
            cfg.get('pacmap', False) and embedding_available('pacmap')))
        self._concat_var = tk.BooleanVar(value=bool(cfg.get('concatenate', False)))
        self._reproducible_var = tk.BooleanVar(
            value=bool(cfg.get('reproducible', True)))
        self._fast_graph_var = tk.BooleanVar(
            value=bool(cfg.get('fast_graph', False)))
        self._events_csv_var = tk.BooleanVar(
            value=bool(cfg.get('events_csv', False)))

        # Method + numeric params, stacked into narrow rows so they fit a
        # docked panel on a normal monitor instead of clipping off the edge.
        # Method + k + resolution share one row, which fits the panel's
        # narrow default width.
        mrow = ttk.Frame(runbar)
        mrow.pack()
        ttk.Label(mrow, text='Method').pack(side='left')
        from .tool_window import (
            CLUSTER_METHOD_NAMES,
            CLUSTER_RESOLUTION_TIP,
            FAST_GRAPH_TIP,
            named_choice,
        )
        _mc = named_choice(mrow, self._method_var, CLUSTER_METHOD_NAMES,
                           width=11)
        _mc.pack(side='left', padx=(2, 0))
        _tip(_ed, _mc, "Clustering algorithm: PhenoGraph (Louvain), Leiden, or "
             "FlowSOM. Neighbors and resolution apply to PhenoGraph and "
             "Leiden.")
        krow = ttk.Frame(runbar)
        krow.pack()
        _kl = ttk.Label(krow, text='Neighbors')
        _kl.pack(side='left')
        _ke = ttk.Entry(krow, textvariable=self._k_var, width=4)
        _ke.pack(side='left', padx=(1, 0))
        _tip(_ed, _kl, "Neighbors (k) for PhenoGraph / Leiden / UMAP graph "
             "construction (higher = coarser clusters / smoother embedding).")
        _tip(_ed, _ke, "Neighbors (k) for the clustering / embedding graph.")
        ttk.Label(krow, text='Resolution').pack(side='left', padx=(8, 0))
        _re = ttk.Entry(krow, textvariable=self._res_var, width=4)
        _re.pack(side='left', padx=(1, 0))
        _tip(_ed, _re, CLUSTER_RESOLUTION_TIP + " Used by PhenoGraph (with "
             "Reproducible on) and Leiden; ignored by FlowSOM.")

        prow2 = ttk.Frame(runbar)
        prow2.pack()
        ttk.Label(prow2, text='Max events').pack(side='left')
        _me = ttk.Entry(prow2, textvariable=self._max_var, width=7)
        _me.pack(side='left', padx=(1, 8))
        _tip(_ed, _me, "Max events per run unit fed to clustering/embedding "
             "(subsampled if larger) — caps memory/time.")
        ttk.Label(prow2, text='Seed').pack(side='left')
        _se = ttk.Entry(prow2, textvariable=self._seed_var, width=10)
        _se.pack(side='left', padx=(1, 0))
        _tip(_ed, _se, "Random seed — fixes the subsample + embedding for "
             "reproducible results. A word or phrase works too "
             "('pilot run 3'); it is hashed to a number, so the same "
             "phrase always gives the same answer.")

        # Embeddings split across two rows so the five options + Concat don't
        # overflow a narrow panel.
        _embs = [
            ('UMAP', 'umap', self._umap_var, "Compute a UMAP embedding."),
            ('t-SNE', 'tsne', self._tsne_var, "Compute a t-SNE embedding."),
            ('PHATE', 'phate', self._phate_var,
             "Compute a PHATE embedding (needs the [embed] extra)."),
            ('TriMap', 'trimap', self._trimap_var,
             "Compute a TriMap embedding (needs the [embed] extra)."),
            ('PaCMAP', 'pacmap', self._pacmap_var,
             "Compute a PaCMAP embedding (needs the [embed] extra)."),
        ]
        erow1 = ttk.Frame(runbar)
        erow1.pack()
        ttk.Label(erow1, text='embed:').pack(side='left', padx=(0, 2))
        erow2 = ttk.Frame(runbar)
        erow2.pack()
        self._embed_boxes: dict[str, tuple[ttk.Checkbutton, tk.BooleanVar]] = {}
        for _i, (_txt, _key, _var, _tt) in enumerate(_embs):
            _cb = ttk.Checkbutton(erow1 if _i < 3 else erow2, text=_txt,
                                  variable=_var)
            _cb.pack(side='left', padx=(0, 2))
            self._embed_boxes[_key] = (_cb, _var)
            # Resolved at hover time: the import check below can change it.
            _tip(_ed, _cb, lambda k=_key, t=_tt: self._embed_tip(k, t))
        self._sync_embed_boxes()
        self._check_embed_engines()
        # Concatenate is a run-unit option, NOT an embedding method — give it its
        # own row below a divider so it doesn't read as a 6th embedding type.
        ttk.Separator(runbar, orient='horizontal').pack(
            fill='x', pady=(4, 2), padx=10)
        crow = ttk.Frame(runbar)
        crow.pack()
        _cc = ttk.Checkbutton(crow, text='Concatenate',
                              variable=self._concat_var)
        _cc.pack(side='left')
        _tip(_ed, _cc, "Concatenate a group's samples into ONE co-embedded run "
             "(off = embed each sample separately). Not an embedding method.")
        _rc = ttk.Checkbutton(crow, text='Reproducible',
                              variable=self._reproducible_var)
        _rc.pack(side='left', padx=(12, 0))
        _tip(_ed, _rc, "Seed PhenoGraph (via its Leiden backend) so a re-run "
             "gives identical clusters. Leiden / FlowSOM / embeddings are "
             "already seeded; off = PhenoGraph's default (non-reproducible) "
             "Louvain.")
        _fg = ttk.Checkbutton(crow, text='Fast graph',
                              variable=self._fast_graph_var)
        _fg.pack(side='left', padx=(12, 0))
        _tip(_ed, _fg, FAST_GRAPH_TIP)
        orow = ttk.Frame(runbar)
        orow.pack()
        _ec = ttk.Checkbutton(orow, text='Also write events as CSV',
                              variable=self._events_csv_var)
        _ec.pack(side='left')
        _tip(_ed, _ec, EVENTS_CSV_TIP)

        actions = ttk.Frame(runbar)
        actions.pack(pady=(3, 0))                # centred in the run bar
        self.btn_run = ttk.Button(actions, text='▶ Run', width=8, command=self._do_run)
        self.btn_run.pack(side='left')
        _tip(_ed, self.btn_run, "Run clustering (+ embeddings) over the "
             "workspace's run units; writes a cluster CSV + embedding PNGs.")
        self.btn_cancel = ttk.Button(actions, text='Cancel', width=8,
                                     command=self._do_cancel, state='disabled')
        self.btn_cancel.pack(side='left', padx=(3, 0))
        _tip(_ed, self.btn_cancel, "Cancel the running job (terminates the "
             "current child process).")
        _resb = ttk.Button(actions, text='Results…', width=9,
                           command=self._do_results)
        _resb.pack(side='left', padx=(3, 0))
        _tip(_ed, _resb, "Open the results viewer for the last run's output "
             "folder (cluster maps + embeddings).")

        treef = ttk.Frame(self)
        treef.grid(row=2, column=0, sticky='nsew')
        treef.columnconfigure(0, weight=1)
        treef.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(treef, columns=('on', 'comp', 'fmo'),
                                 show='tree headings', selectmode='extended')
        self.tree.heading('#0', text='Samples / populations', anchor='w')
        self.tree.heading('on', text='👁', anchor='center')
        self.tree.heading('comp', text='Comp / beads', anchor='w')
        self.tree.heading('fmo', text='FMOs', anchor='w')
        # Columns sized to fit the docked panel (~320 px): #0 stretches, the
        # rest are tight. comp/fmo show short assignment tags; hover/the full
        # width on pop-out shows the rest.
        self.tree.column('#0', anchor='w', stretch=True, width=140, minwidth=90)
        self.tree.column('on', width=26, anchor='center', stretch=False)
        self.tree.column('comp', width=76, anchor='w', stretch=False)
        self.tree.column('fmo', width=58, anchor='w', stretch=False)
        self.tree.grid(row=0, column=0, sticky='nsew')
        sb = ttk.Scrollbar(treef, orient='vertical', command=self.tree.yview)
        sb.grid(row=0, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.tag_configure('grp', font='OpenFloHeading')
        # Theme-aware row tags: muted 'off' text and a drop highlight that reads
        # on the active chrome (a deep olive on dark, soft amber on light) so the
        # drag target isn't an unreadable light band under Midnight.
        from .theme import current_palette
        _pal = current_palette()
        self.tree.tag_configure('off', foreground=_pal['muted'])
        self.tree.tag_configure('warn', foreground=_pal['error'])
        self.tree.tag_configure('drop_target', background=_pal['highlight'],
                                foreground=_pal['highlight_fg'])
        # Press / motion / release: a bare click toggles the 👁 column (old
        # _on_click); a drag past the threshold MOVES the dragged item(s) to the
        # group under the pointer (or to loose/top-level if dropped on empty
        # space) — fixes items accidentally dropped outside a group. A drag that
        # ends over an open Statistics window hands the item's sample to it.
        self.tree.bind('<Button-1>', self._on_press)
        self.tree.bind('<B1-Motion>', self._on_motion)
        self.tree.bind('<ButtonRelease-1>', self._on_release)
        self.tree.bind('<Delete>', lambda e: self._remove_selected())
        bind_right_click(self.tree, self._on_tree_right_click)
        # Drag SOURCE: Shift-drag pipeline item(s) OUT to another OpenFlo window.
        try:
            from tkinterdnd2 import DND_FILES as _DND_FILES
            self.tree.drag_source_register(1, _DND_FILES)  # type: ignore[attr-defined]
            self.tree.dnd_bind('<<DragInitCmd>>',          # type: ignore[attr-defined]
                               self._on_os_drag_init)
        except Exception as exc:  # noqa: BLE001 - tkinterdnd2/tkdnd unavailable; drag-out is optional
            print(f"[DnD] workspace drag-source: {exc}", flush=True)
        self._press_row: str | None = None
        self._press_col: str | None = None
        self._press_x = 0
        self._press_y = 0
        self._drag_active = False
        self._drag_threshold = 5
        self._drop_hi: str | None = None

    # -- drop target geometry --
    def contains(self, widget: tk.Misc | None) -> bool:
        """Whether ``widget`` is the pipeline tree or one of its descendants (drop-target test)."""
        w = widget
        while w is not None:
            if w is self.tree:
                return True
            w = getattr(w, 'master', None)
        return False

    def contains_point(self, x_root: int, y_root: int) -> bool:
        """True if the SCREEN point falls within this view's tree rectangle.
        A grab-proof alternative to winfo_containing: during an in-app button
        drag Tk holds a pointer grab on the source tree, and winfo_containing
        over a DIFFERENT toplevel (a popped-out workspace) returns None — so a
        drop onto the floating workspace wouldn't register. Geometry hit-testing
        works whether the workspace is docked or floating."""
        try:
            t = self.tree
            if not t.winfo_ismapped():
                return False
            x, y = t.winfo_rootx(), t.winfo_rooty()
            return (x <= x_root < x + t.winfo_width()
                    and y <= y_root < y + t.winfo_height())
        except tk.TclError:
            return False

    @staticmethod
    def _row_target(row: str | None) -> _Target | None:
        if not row:
            return None
        if row.startswith('G'):
            return ('group', row)
        if row.startswith('M'):
            return ('item', row)
        return None

    def drop_at(self, editor: Any, nodes: Sequence[_Node], x_root: int,
                y_root: int) -> bool:
        """Route an editor drop by the column + row under the pointer:
        comp column → set beads on that row; fmo column → set FMOs; else add
        the nodes as items (into the group row if dropped on one)."""
        try:
            x = x_root - self.tree.winfo_rootx()
            y = y_root - self.tree.winfo_rooty()
            col = self.tree.identify_column(x)
            row = self.tree.identify_row(y)
        except Exception:  # noqa: BLE001 - pointer lookup failed; drop on tree
            col, row = '#0', ''
        return self._route_drop(editor, nodes, _COL.get(col, 'tree'),
                                self._row_target(row))

    def _route_drop(self, editor: Any, nodes: Sequence[_Node], colname: str,
                    target: _Target | None) -> bool:
        if colname == 'comp':
            return self._assign_comp(editor, nodes, target)
        if colname == 'fmo':
            return self._assign_fmo(editor, nodes, target)
        gid = target[1] if (target and target[0] == 'group') else None
        self._pre_change()
        added = 0
        for p in nodes:
            ctx = extract_one_context(editor, p[1])
            if ctx is None:
                continue
            leaf = p[2] if p[0] == 'gate' else None
            self.model.add_item(build_drop_payload(ctx, leaf), gid=gid)
            added += 1
        if added:
            self._render()
            self._changed()
            self._status(f"Added {added} item(s)"
                         + (f" to '{self.model.groups[gid]['name']}'." if gid else "."))
        return added > 0

    def _node_paths(self, editor: Any, nodes: Sequence[_Node]) -> list[str]:
        paths = []
        for p in nodes:
            ctx = extract_one_context(editor, p[1])
            if ctx and ctx.get('path'):
                paths.append(ctx['path'])
        return paths

    def _assign_comp(self, editor: Any, nodes: Sequence[_Node],
                     target: _Target | None) -> bool:
        paths = self._node_paths(editor, nodes)
        if not paths or target is None:
            return False
        self._pre_change()
        comp = {'kind': 'beads', 'files': paths}
        if target[0] == 'group':
            self.model.set_group_comp(target[1], comp)
        else:
            self.model.set_item_comp(target[1], comp)
        self._render()
        self._changed()
        self._status(f"Set comp beads ({len(paths)} file(s)) — overrides matrix.")
        return True

    def _assign_fmo(self, editor: Any, nodes: Sequence[_Node],
                    target: _Target | None) -> bool:
        paths = self._node_paths(editor, nodes)
        if not paths or target is None:
            return False
        self._pre_change()
        fmo = {'files': paths}
        if target[0] == 'group':
            self.model.set_group_fmo(target[1], fmo)
        else:
            self.model.set_item_fmo(target[1], fmo)
        self._render()
        self._changed()
        self._status(f"Set {len(paths)} FMO(s).")
        return True

    # -- rendering --
    # ── Embedding availability ─────────────────────────────────────────
    def _sync_embed_boxes(self) -> None:
        """Disable, and untick, each [embed] box whose backend can't run."""
        for key, (cb, var) in getattr(self, '_embed_boxes', {}).items():
            try:
                if embedding_available(key):
                    cb.state(['!disabled'])
                else:
                    cb.state(['disabled'])
                    var.set(False)
            except tk.TclError:                       # panel closed
                return

    def _check_embed_engines(self) -> None:
        """Try importing the [embed] backends off the Tk thread, then update
        the boxes: a backend that is installed but fails to import looks
        available until this has run."""
        from .async_task import run_async
        from .capabilities import check_engines
        mods = [m for m, _t in EXTRA_EMBEDDINGS.values()]
        run_async(self, lambda: check_engines(mods),
                  on_done=lambda _r: self._sync_embed_boxes())

    def _embed_tip(self, key: str, text: str) -> str:
        why = embedding_unavailable(key)
        if why is None:
            return text
        return why.replace(' skipped:', ' can\'t run here:') + (
            ". Install it with pip install openflo[embed], then reopen the "
            "Pipeline." if 'not installed' in why else ".")

    def _render(self) -> None:
        tv = self.tree
        for iid in tv.get_children(''):
            tv.delete(iid)
        for gid, g in self.model.groups.items():
            on = g.get('display', True)
            tv.insert('', 'end', iid=gid,
                      text=f'▦ {g["name"]}  ({len(g["items"])})',
                      values=('☑' if on else '☐',
                              self._group_comp_label(g), self._group_fmo_label(g)),
                      open=True, tags=('grp',) if on else ('grp', 'off'))
            for mid, it in g['items'].items():
                self._insert_item(gid, mid, it, group_on=on)
        for mid, it in self.model.loose.items():
            self._insert_item('', mid, it, group_on=True)

    def _insert_item(self, parent: str, mid: str, it: dict[str, Any],
                     group_on: bool = True) -> None:
        disp = it.get('display', True)
        comp = self.model.effective_comp(mid)
        fmo = self.model.effective_fmo(mid)
        fmo_lbl = f"{len(fmo.get('files', []))} FMO" if fmo else '—'
        who = f"{it.get('trial', 'Trial')} / {it['sample']}"
        label = f"{who}  ›  {it['gate_path']}" if it.get('gate_id') else who
        if not comp['ready']:
            tag = 'warn'
        elif group_on and disp:
            tag = self._color_tag(mid, it.get('color'))
        else:
            tag = 'off'
        self.tree.insert(parent, 'end', iid=mid, text=f'■ {label}',
                         values=('☑' if disp else '☐', comp['label'], fmo_lbl),
                         tags=(tag,))

    def _group_comp_label(self, g: dict[str, Any]) -> str:
        c = g.get('comp')
        if not c:
            return '—'
        return (f"beads:{len(c.get('files', []))} (grp)"
                if c.get('kind') == 'beads' else 'matrix (grp)')

    def _group_fmo_label(self, g: dict[str, Any]) -> str:
        f = g.get('fmo')
        return f"{len(f.get('files', []))} FMO (grp)" if f else '—'

    def _color_tag(self, key: str, color: str | None) -> str:
        tag = f'col_{key}'
        self.tree.tag_configure(tag, foreground=color or '#000000',
                                font='OpenFloHeading')
        return tag

    # -- interactions --
    def _on_press(self, event: _Event) -> None:
        """Record the press so release can tell a click from a drag. Doesn't
        consume the event, so Treeview's own selection still happens."""
        self._press_row = self.tree.identify_row(event.y)
        self._press_col = self.tree.identify_column(event.x)
        self._press_x, self._press_y = event.x, event.y
        self._drag_active = False
        # ALT arms an OS cross-instance file-drag (see _on_os_drag_init); plain
        # drags stay the in-app move-into-group drag. Alt (not Shift/Ctrl) so it
        # never triggers tree selection. Alt mask: 0x20000 (Win) / 0x0008 (X11).
        _st = getattr(event, 'state', 0)
        self._os_drag_armed = bool(_st & 0x20000) or bool(_st & 0x0008)

    def _on_motion(self, event: _Event) -> None:
        """Past the threshold, an item-row press becomes a move-drag: the
        cursor changes and the group row under the pointer is highlighted."""
        if self._press_row is None or self._press_col == '#1':
            return
        if not self._drag_active:
            if (abs(event.x - self._press_x) <= self._drag_threshold
                    and abs(event.y - self._press_y) <= self._drag_threshold):
                return
            if not (self._press_row and self._press_row.startswith('M')):
                return   # only items are draggable; groups stay put
            self._drag_active = True
            try:
                self.tree.config(cursor='fleur')
            except tk.TclError:
                pass
        row = self.tree.identify_row(event.y)
        self._set_drop_hi(row if (row and row.startswith('G')) else None)

    def _on_release(self, event: _Event) -> None:
        """Click → toggle the 👁 column (old _on_click). Drag → move the
        dragged item(s) to the destination group (or loose), unless the drop
        landed on an open Statistics window (then add the sample there)."""
        drag = self._drag_active
        press_row = self._press_row
        press_col = self._press_col
        try:
            if not drag:
                if press_col == '#1':
                    self._toggle_display_at(press_row)
                return
            if self._drop_to_layout(event):
                return
            if self._drop_to_tray(event):
                return
            if self._drop_to_stats(event):
                return
            self._move_drag_to(press_row, self.tree.identify_row(event.y))
        finally:
            self._clear_drag()

    def _toggle_display_at(self, row: str | None) -> None:
        target = self._row_target(row)
        if target is None:
            return
        self._pre_change()
        if target[0] == 'group':
            self.model.toggle_group_display(target[1])
        else:
            self.model.toggle_item_display(target[1])
        self._render()
        self._changed()

    def _move_drag_to(self, mid: str | None, drop_row: str) -> None:
        """Move the dragged item(s) to the group implied by the row under the
        pointer: a group row → that group; an item row → that item's group;
        empty space → loose (top level)."""
        if not (mid and mid.startswith('M')):
            return
        # Carry the whole item selection if the dragged row is part of it.
        sel = [t[1] for t in self._selected_targets() if t[0] == 'item']
        mids = sel if (mid in sel and len(sel) > 1) else [mid]
        dest_gid = None
        if drop_row:
            if drop_row.startswith('G'):
                dest_gid = drop_row
            elif drop_row.startswith('M'):
                dest_gid = self.model.item_group(drop_row)
        # No-op if every dragged item already lives in the destination.
        if all(self.model.item_group(m) == dest_gid for m in mids):
            return
        self._pre_change()
        for m in mids:
            self.model.move_item(m, dest_gid)
        self._render()
        self._changed()
        where = (f"'{self.model.groups[dest_gid]['name']}'"
                 if dest_gid else 'loose items')
        self._status(f"Moved {len(mids)} item(s) to {where}.")

    def _dragged_rows(self) -> list[tuple[str, str | None]]:
        """``[(sample, gate_id|None), ...]`` for the dragged item(s)."""
        sel = [t[1] for t in self._selected_targets() if t[0] == 'item']
        mids = (sel if (self._press_row in sel and len(sel) > 1)
                else [self._press_row])
        rows = []
        for m in mids:
            it = self.model.item(m) if m else None
            if not it or not it.get('sample'):
                continue
            r = (it.get('sample'), it.get('gate_id'))
            if r not in rows:
                rows.append(r)
        return rows

    def _drop_to_tray(self, event: _Event) -> bool:
        """Dragged back onto the Samples & Gates tray (its tree or a tab on
        its tab strip, docked or popped out): the samples/gates join that
        tab. True iff consumed. Defensive -- never crashes the drag."""
        drop = getattr(self._editor, '_drop_from_workspace', None)
        if drop is None:
            return False
        try:
            return bool(drop(self._dragged_rows(), event.x_root,
                             event.y_root))
        except Exception as exc:  # noqa: BLE001 - noted; a failed drop must not crash the drag
            note_handled("hand the dragged rows to the editor", exc)
            return False

    def _drop_to_layout(self, event: _Event) -> bool:
        """If a drag ends over the layout editor's page, add the dragged
        item(s)' populations there (a bare-sample item is its whole sample).
        Returns True iff consumed. Defensive — never crashes the drag."""
        ed = self._editor
        drop = getattr(ed, '_drop_to_layout', None)
        if drop is None:
            return False
        sel = [t[1] for t in self._selected_targets() if t[0] == 'item']
        mids = (sel if (self._press_row in sel and len(sel) > 1)
                else [self._press_row])
        targets = []
        for m in mids:
            it = self.model.item(m) if m else None
            if not it or not it.get('sample'):
                continue
            t = (it.get('sample'), it.get('gate_id'))
            if t not in targets:
                targets.append(t)
        try:
            return bool(drop(targets, event.x_root, event.y_root))
        except Exception as exc:  # noqa: BLE001 - noted; a failed drop must not crash the drag
            note_handled("hand the dragged rows to the layout editor", exc)
            return False

    def _drop_to_stats(self, event: _Event) -> bool:
        """If a drag ends over an open Statistics window, add the dragged
        item(s)' gated population(s) to it. Statistics is population-based, so
        only items that carry a gate qualify — a bare-sample item is consumed
        with an explanatory status. Returns True iff over a stats window.
        Defensive — never crashes the drag."""
        ed = self._editor
        finder = getattr(ed, '_stats_window_under', None)
        if finder is None:
            return False
        try:
            win = finder(event.x_root, event.y_root)
        except Exception:  # noqa: BLE001 - drag must never crash
            win = None
        if win is None:
            return False
        sel = [t[1] for t in self._selected_targets() if t[0] == 'item']
        mids = (sel if (self._press_row in sel and len(sel) > 1)
                else [self._press_row])
        targets = []
        for m in mids:
            it = self.model.item(m) if m else None
            if not it:
                continue
            nm, gid = it.get('sample'), it.get('gate_id')
            if nm and gid and (nm, gid) not in targets:
                targets.append((nm, gid))
        if not targets:
            self._status("Statistics accepts gated populations only — "
                         "this item has no gate.")
            return True
        try:
            win.add_targets(targets, 'workspace')
            self._status(f"Added {len(targets)} population(s) to statistics.")
        except Exception as exc:  # noqa: BLE001 - noted; a failed drop must not crash the drag
            note_handled("add the dragged items to statistics", exc)
        return True

    def _set_drop_hi(self, iid: str | None) -> None:
        if self._drop_hi == iid:
            return
        if self._drop_hi is not None:
            self._strip_drop_tag(self._drop_hi)
        self._drop_hi = iid
        if iid is not None:
            try:
                tags = [t for t in self.tree.item(iid, 'tags') if t != 'drop_target']
                self.tree.item(iid, tags=tags + ['drop_target'])
            except tk.TclError:
                pass

    def _strip_drop_tag(self, iid: str) -> None:
        try:
            tags = [t for t in self.tree.item(iid, 'tags') if t != 'drop_target']
            self.tree.item(iid, tags=tags)
        except tk.TclError:
            pass

    def _clear_drag(self) -> None:
        if self._drop_hi is not None:
            self._strip_drop_tag(self._drop_hi)
        self._drop_hi = None
        self._press_row = None
        self._press_col = None
        self._drag_active = False
        try:
            self.tree.config(cursor='')
        except tk.TclError:
            pass

    def _selected_targets(self) -> list[_Target]:
        return [t for t in (self._row_target(r) for r in self.tree.selection()) if t]

    def _on_tree_right_click(self, event: _Event) -> None:
        """Context menu for cross-instance transfer of pipeline item(s).
        Delegates to the editor's clipboard bundle (FCS + gates), so a pipeline
        item can be Copied here and Pulled into another OpenFlo window."""
        try:
            row = self.tree.identify_row(event.y)
        except tk.TclError:
            return
        if row and row not in self.tree.selection():
            try:
                self.tree.selection_set(row)
            except tk.TclError:
                pass
        ed = self._editor
        m = tk.Menu(self.tree, tearoff=0)
        if ed is not None and hasattr(ed, '_theme_menu'):
            try:
                ed._theme_menu(m)
            except Exception:  # noqa: BLE001 - menu theming is cosmetic
                pass
        has_items = any(t[0] == 'item' for t in self._selected_targets())
        item_state = 'normal' if (has_items and ed is not None) else 'disabled'
        m.add_command(
            label="Copy sample (with gates)",
            command=self._copy_items_transfer, state=item_state)
        m.add_command(
            label="Move to another window",
            command=self._send_items_transfer, state=item_state)
        pull_state = 'disabled'
        if ed is not None and hasattr(ed, '_read_transfer_bundle'):
            try:
                pull_state = ('normal' if ed._read_transfer_bundle()
                              is not None else 'disabled')
            except Exception:  # noqa: BLE001 - menu entry state only
                pull_state = 'disabled'
        m.add_command(
            label="Paste sample(s) from another window",
            command=self._pull_items_transfer,
            state=pull_state)
        try:
            m.tk_popup(event.x_root, event.y_root)
        except tk.TclError:
            pass

    def _pull_items_transfer(self) -> None:
        ed = self._editor
        if ed is not None and hasattr(ed, '_paste_samples_transfer'):
            ed._paste_samples_transfer()

    # tkinterdnd2 hands its own event object (not a tk.Event), unused here.
    def _on_os_drag_init(self, event: Any) -> str | tuple[str, str, tuple[str, ...]]:
        """tkdnd drag-source for the pipeline tree: Alt-drag item(s) OUT to
        another OpenFlo window as an OS file-drag of their source FCS. Returns
        '' (abort) for a plain drag, leaving the in-app move-into-group drag."""
        if not getattr(self, '_os_drag_armed', False):
            return ''
        import os as _os
        ed = self._editor
        paths = []
        for t in self._selected_targets():
            if t[0] != 'item':
                continue
            it = self.model.item(t[1]) or {}
            nm = it.get('name')
            p = ''
            if ed is not None and nm in getattr(ed, '_samples', {}):
                p = getattr(ed._samples[nm], 'path', '') or ''
            if not p:
                p = it.get('path') or ''
            if (p and p.lower().endswith('.fcs') and _os.path.isfile(p)
                    and p not in paths):
                paths.append(p)
        if not paths:
            return ''
        try:
            from tkinterdnd2 import COPY, DND_FILES
        except ImportError:
            return ''
        return (COPY, DND_FILES, tuple(paths))

    def _copy_items_transfer(self) -> None:
        """Copy the selected pipeline item(s)' source sample(s) — with gates —
        to the clipboard for transfer to another OpenFlo instance."""
        ed = self._editor
        if ed is None or not hasattr(ed, '_copy_samples_transfer'):
            return
        names = []
        for t in self._selected_targets():
            if t[0] != 'item':
                continue
            it = self.model.item(t[1])
            nm = (it or {}).get('name')
            if nm and nm not in names:
                names.append(nm)
        if not names:
            self._status("Select pipeline item row(s) to copy.")
            return
        ed._copy_samples_transfer(names)

    def _send_items_transfer(self) -> None:
        """Send the selected pipeline item(s)' source sample(s) to another
        instance (removes them from the editor — see _send_samples_transfer)."""
        ed = self._editor
        if ed is None or not hasattr(ed, '_send_samples_transfer'):
            return
        names = []
        for t in self._selected_targets():
            if t[0] != 'item':
                continue
            it = self.model.item(t[1])
            nm = (it or {}).get('name')
            if nm and nm not in names:
                names.append(nm)
        if not names:
            self._status("Select pipeline item row(s) to send.")
            return
        ed._send_samples_transfer(names)

    def _new_group(self) -> None:
        self._pre_change()
        gid = self.model.new_group()
        self._render()
        self._changed()
        self._status(f"New group '{self.model.groups[gid]['name']}'.")

    def _group_selected(self) -> None:
        mids = [t[1] for t in self._selected_targets() if t[0] == 'item']
        if not mids:
            self._status("Select item rows to group.")
            return
        self._pre_change()
        gid = self.model.group_selected(mids)
        self._render()
        self._changed()
        self._status(f"Grouped {len(mids)} item(s) into '{self.model.groups[gid]['name']}'.")

    def _group_params(self) -> None:
        """Per-group override of method / k / resolution (empty model cfg =
        inherit the run-bar settings)."""
        gids = [t[1] for t in self._selected_targets() if t[0] == 'group']
        if not gids:
            self._status("Select a group row to set its parameters.")
            return
        gid = gids[0]
        g = self.model.groups.get(gid)
        if not g:
            return
        ov = dict(g.get('cfg') or {})
        base = self.model.run_cfg
        dlg = tk.Toplevel(self)
        dlg.title(f"Group params — {g['name']}")
        dlg.geometry("320x210")
        frm = ttk.Frame(dlg)
        frm.pack(fill='both', expand=True, padx=12, pady=10)
        ttk.Label(frm, justify='left', text=(
            f"Overrides for '{g['name']}'. Blank/unchanged inherits the "
            "run-bar settings.")).pack(anchor='w', pady=(0, 8))
        mvar = tk.StringVar(value=str(ov.get('method', base.get('method'))))
        kvar = tk.StringVar(value=str(ov.get('k', base.get('k'))))
        rvar = tk.StringVar(value=str(ov.get('resolution', base.get('resolution'))))
        grid = ttk.Frame(frm)
        grid.pack(fill='x', anchor='w')
        grid.columnconfigure(1, weight=1)
        ttk.Label(grid, text="Method:").grid(row=0, column=0, sticky='w', pady=3)
        from .tool_window import (
            CLUSTER_METHOD_NAMES,
            CLUSTER_RESOLUTION_TIP,
            named_choice,
        )
        _ed = self._editor
        _mc = named_choice(grid, mvar, CLUSTER_METHOD_NAMES, width=14)
        _mc.grid(row=0, column=1, sticky='w', padx=(8, 0), pady=3)
        _tip(_ed, _mc, "The clustering method for this group only. Other "
             "groups keep the run bar's method.")
        ttk.Label(grid, text="Neighbors (k):").grid(row=1, column=0,
                                                    sticky='w', pady=3)
        _ke = ttk.Entry(grid, textvariable=kvar, width=8)
        _ke.grid(row=1, column=1, sticky='w', padx=(8, 0), pady=3)
        _tip(_ed, _ke, "How many look-alike neighbours each cell is linked "
             "to, for this group only (at least 2). More neighbours give "
             "fewer, larger clusters. Not used by FlowSOM.")
        ttk.Label(grid, text="Resolution:").grid(
            row=2, column=0, sticky='w', pady=3)
        _re = ttk.Entry(grid, textvariable=rvar, width=8)
        _re.grid(row=2, column=1, sticky='w', padx=(8, 0), pady=3)
        _tip(_ed, _re, CLUSTER_RESOLUTION_TIP + " For this group only; "
             "not used by FlowSOM.")

        def _apply() -> None:
            new: dict = {'method': mvar.get()}
            try:
                new['k'] = max(2, int(float(kvar.get())))
            except (ValueError, tk.TclError):
                pass
            try:
                new['resolution'] = max(0.01, float(rvar.get()))
            except (ValueError, tk.TclError):
                pass
            g['cfg'] = new
            self._changed()
            self._status(f"Set params for group '{g['name']}'.")
            dlg.destroy()

        def _clear() -> None:
            g['cfg'] = {}
            self._changed()
            self._status(f"Group '{g['name']}' now inherits the run-bar settings.")
            dlg.destroy()
        bar = ttk.Frame(frm)
        bar.pack(side='bottom', fill='x', pady=(8, 0))
        _ab = ttk.Button(bar, text="Apply", default='active', command=_apply)
        _ab.pack(side='right', padx=6)
        _tip(_ed, _ab, "Use these settings for this group on the next run, "
             "instead of the run bar's. A number that cannot be read keeps "
             "the run bar's value.")
        _ib = ttk.Button(bar, text="Inherit (clear)", command=_clear)
        _ib.pack(side='right')
        _tip(_ed, _ib, "Remove this group's own settings, so it uses the "
             "run bar's method, neighbours and resolution again.")

    def _set_comp_picker(self) -> None:
        files = self._pick_files("Select compensation bead FCS files")
        if not files:
            return
        comp = {'kind': 'beads', 'files': files}
        self._apply_to_selection(lambda gid: self.model.set_group_comp(gid, comp),
                                 lambda mid: self.model.set_item_comp(mid, comp),
                                 f"comp beads ({len(files)} file(s))")

    def _set_fmo_picker(self) -> None:
        files = self._pick_files("Select FMO control FCS files")
        if not files:
            return
        fmo = {'files': files}
        self._apply_to_selection(lambda gid: self.model.set_group_fmo(gid, fmo),
                                 lambda mid: self.model.set_item_fmo(mid, fmo),
                                 f"{len(files)} FMO(s)")

    def _pick_files(self, title: str) -> list[str]:
        from tkinter import filedialog
        return list(filedialog.askopenfilenames(
            parent=self, title=title,
            filetypes=[('FCS files', '*.fcs'), ('All files', '*.*')]))

    def _apply_to_selection(self, on_group: Callable[[str], object],
                            on_item: Callable[[str], object], what: str) -> None:
        targets = self._selected_targets()
        if not targets:
            self._status(f"Select a group or item to set {what}.")
            return
        self._pre_change()
        for t in targets:
            (on_group if t[0] == 'group' else on_item)(t[1])
        self._render()
        self._changed()
        self._status(f"Set {what} on {len(targets)} row(s).")

    def _remove_selected(self) -> None:
        targets = self._selected_targets()
        if not targets:
            return
        self._pre_change()
        for t in targets:
            if t[0] == 'group':
                self.model.remove_group(t[1], keep_items=True)
            else:
                self.model.remove_item(t[1])
        self._render()
        self._changed()
        self._status("Removed.")

    def _clear(self) -> None:
        if not self.model.groups and not self.model.loose:
            return
        self._pre_change()
        self.model.clear()
        self._render()
        self._changed()
        self._status("Workspace cleared.")

    def _import_all(self) -> None:
        """Import every loaded editor sample as a loose workspace item."""
        if self._editor is None:
            return
        self._pre_change()
        n = 0
        for ctx in extract_editor_context(self._editor):
            self.model.add_item(build_drop_payload(ctx, None))
            n += 1
        self._render()
        self._changed()
        self._status(f"Imported {n} sample(s)."
                     if n else "Editor has no loaded samples.")

    def _import_selected(self) -> None:
        """Import only the sample(s) selected in the editor's tree."""
        if self._editor is None:
            return
        try:
            names = self._editor._target_samples('selected')
        except Exception:  # noqa: BLE001 - status asks to select samples
            names = []
        if not names:
            self._status("Select sample(s) in the editor's tree first, "
                         "then Import selected.")
            return
        self._pre_change()
        n = 0
        for nm in names:
            ctx = extract_one_context(self._editor, nm)
            if ctx is None:
                continue
            self.model.add_item(build_drop_payload(ctx, None))
            n += 1
        self._render()
        self._changed()
        self._status(f"Imported {n} selected sample(s)."
                     if n else "No samples imported.")

    def _status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)

    def _changed(self) -> None:
        if self._on_change:
            self._on_change(self)

    def _pre_change(self) -> None:
        """Fire the editor's undo-checkpoint hook before a model mutation."""
        if self._on_before_change:
            try:
                self._on_before_change()
            except Exception as exc:  # noqa: BLE001 - noted; the edit goes on without a checkpoint
                note_handled("take the undo checkpoint", exc)

    # -- run (M2) --
    def _sync_run_cfg(self) -> dict:
        """Push run-bar widgets into the model so settings persist across
        pop-out / re-dock. Unparseable ints keep the current model value."""
        cfg = self.model.run_cfg

        def _int(var: tk.StringVar, key: str, lo: int) -> None:
            try:
                cfg[key] = max(lo, int(float(var.get())))
            except (ValueError, tk.TclError):
                pass

        _int(self._k_var, 'k', 2)
        try:                       # max events: 0 / blank = all events
            cfg['max_events'] = max(0, int(float(self._max_var.get())))
        except (ValueError, tk.TclError):
            pass
        # Seed accepts a number OR a word/phrase (hashed — see
        # seeds.resolve_seed). Stored verbatim so the run card and the
        # methods report can show what the user actually typed; it is resolved
        # to an int at use. Blank falls back to the default rather than
        # silently keeping a stale value.
        _seed_txt = self._seed_var.get().strip()
        cfg['seed'] = _seed_txt if _seed_txt else 42
        cfg['method'] = self._method_var.get()
        try:
            cfg['resolution'] = max(0.01, float(self._res_var.get()))
        except (ValueError, tk.TclError):
            pass
        cfg['umap'] = bool(self._umap_var.get())
        cfg['tsne'] = bool(self._tsne_var.get())
        cfg['phate'] = bool(self._phate_var.get())
        cfg['trimap'] = bool(self._trimap_var.get())
        cfg['pacmap'] = bool(self._pacmap_var.get())
        cfg['concatenate'] = bool(self._concat_var.get())
        cfg['reproducible'] = bool(self._reproducible_var.get())
        cfg['fast_graph'] = bool(self._fast_graph_var.get())
        cfg['events_csv'] = bool(self._events_csv_var.get())
        return cfg

    def _load_cfg_into_widgets(self) -> None:
        """Push the model's run_cfg back into the run-bar widgets (after a
        recipe load)."""
        cfg = self.model.run_cfg
        self._method_var.set(str(cfg.get('method', 'phenograph')))
        self._res_var.set(str(cfg.get('resolution', 1.0)))
        self._k_var.set(str(cfg.get('k', 30)))
        self._max_var.set(str(cfg.get('max_events', 5000)))
        self._seed_var.set(str(cfg.get('seed', 42)))
        self._umap_var.set(bool(cfg.get('umap', True)))
        self._tsne_var.set(bool(cfg.get('tsne', False)))
        self._phate_var.set(bool(cfg.get('phate', False)
                                 and embedding_available('phate')))
        self._trimap_var.set(bool(cfg.get('trimap', True)
                                  and embedding_available('trimap')))
        self._pacmap_var.set(bool(cfg.get('pacmap', False)
                                  and embedding_available('pacmap')))
        self._concat_var.set(bool(cfg.get('concatenate', False)))
        self._reproducible_var.set(bool(cfg.get('reproducible', True)))
        self._fast_graph_var.set(bool(cfg.get('fast_graph', False)))
        self._events_csv_var.set(bool(cfg.get('events_csv', False)))

    def _save_recipe(self) -> None:
        import json
        from tkinter import filedialog
        self._sync_run_cfg()
        path = filedialog.asksaveasfilename(
            parent=self, title="Save run recipe", defaultextension='.json',
            initialfile='openflo_recipe.json',
            filetypes=[('OpenFlo recipe', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            from .update import current_version
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'kind': 'openflo-workspace-recipe',
                           'app_version': current_version(),
                           'run_cfg': dict(self.model.run_cfg)}, f, indent=2)
            self._status(f"Saved recipe → {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001 - shown in the status bar; the run setup is kept
            self._status(f"Save recipe failed: {exc}")

    def _load_recipe(self) -> None:
        import json
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(
            parent=self, title="Load run recipe",
            filetypes=[('OpenFlo recipe', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            rc = data.get('run_cfg') if isinstance(data, dict) else None
            if not isinstance(rc, dict):
                raise ValueError("not an OpenFlo recipe (no run_cfg)")
            note = newer_version_note(data, kind='recipe')
            # Merge onto the defaults so older/newer recipes stay compatible.
            merged = default_run_cfg()
            merged.update({k: v for k, v in rc.items() if k in merged})
            self.model.run_cfg = merged
            self._load_cfg_into_widgets()
            self._changed()
            if note:
                messagebox.showwarning("Recipe from a newer OpenFlo", note,
                                       parent=self)
            self._status(f"Loaded recipe: {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001 - shown in an error dialog; a bad file leaves the setup unchanged
            show_error(self, "Load recipe", "The recipe could not be loaded.",
                       exc, path)

    def _pick_markers(self) -> None:
        """Choose the marker channels for clustering/embeddings. Empty = auto
        (proper_run_channels). Stored in run_cfg['channels']."""
        ed = self._editor
        samples = list(getattr(ed, '_samples', {}).values()) if ed else []
        if not samples:
            self._status("Load samples first — markers come from loaded data.")
            return
        cand: list[str] = []
        for s in samples:
            for c in proper_run_channels(s):
                if c not in cand:
                    cand.append(c)
        if not cand:
            self._status("No marker channels found in the loaded samples.")
            return
        current = set(self.model.run_cfg.get('channels') or [])

        dlg = tk.Toplevel(self)
        dlg.title("Clustering markers")
        dlg.geometry("320x460")
        ttk.Label(dlg, justify='left', text=(
            "Markers fed to clustering + embeddings.\nLeave all checked for "
            "auto (every proper marker).")).pack(anchor='w', padx=10, pady=(10, 6))
        inner = scrolled_body(dlg, pack_kw={'padx': 10})
        vars_: dict[str, tk.BooleanVar] = {}
        for c in cand:
            v = tk.BooleanVar(value=(c in current) if current else True)
            _cb = ttk.Checkbutton(inner, text=c, variable=v)
            _cb.pack(anchor='w')
            _tip(ed, _cb, "Ticked: clustering and the embeddings compare "
                 "cells on this marker. Untick markers that should not "
                 "decide the groups, such as a viability or dump channel.")
            vars_[c] = v

        def _apply() -> None:
            chosen = [c for c in cand if vars_[c].get()]
            # All selected → store [] (auto / panel-agnostic); else the subset.
            self.model.run_cfg['channels'] = ([] if len(chosen) == len(cand)
                                              else chosen)
            self._changed()
            self._status(f"Markers: {len(chosen) or len(cand)} channel(s)"
                         + (" (auto)" if len(chosen) == len(cand) else ""))
            dlg.destroy()
        bar = ttk.Frame(dlg)
        bar.pack(side='bottom', fill='x', pady=6)
        # Main action, then Cancel, at the right (tool_window).
        ttk.Button(bar, text="Cancel", command=dlg.destroy).pack(side='right',
                                                                padx=(0, 8))
        _ab = ttk.Button(bar, text="Apply", default='active', command=_apply)
        _ab.pack(side='right', padx=(0, 6))
        _tip(ed, _ab, "Use the ticked markers for the next run. With every "
             "marker ticked, each sample uses all of its own markers.")

    def _batch_folder(self) -> None:
        """Run the current settings over every .fcs in a folder, unattended —
        each file loaded + QC'd + clustered + embedded in the background, with a
        per-event CSV written so results can be loaded back in."""
        import glob
        from tkinter import filedialog, messagebox
        self._sync_run_cfg()
        indir = filedialog.askdirectory(
            parent=self, title="Batch: folder of FCS files")
        if not indir:
            return
        fcs = sorted(glob.glob(os.path.join(indir, '*.fcs')))
        if not fcs:
            messagebox.showinfo("Batch folder", "No .fcs files in that folder.",
                                parent=self)
            return
        outdir = filedialog.askdirectory(
            parent=self, title="Batch: output folder for results")
        if not outdir:
            return
        cfg = dict(self.model.run_cfg)
        if not messagebox.askyesno(
                "Batch folder",
                f"Run {cfg.get('method')} clustering + embeddings on "
                f"{len(fcs)} file(s)?\n\nEach file is processed in the "
                f"background; outputs (clusters, embeddings, per-event CSV) go "
                f"to:\n{outdir}", parent=self):
            return
        ed = self._editor
        total = len(fcs)

        def _busy(msg: str) -> None:
            if ed is not None and hasattr(ed, '_busy'):
                try:
                    ed._busy(msg)
                except Exception:  # noqa: BLE001 - busy text is cosmetic
                    pass

        if ed is not None and hasattr(ed, '_begin_busy'):
            ed._begin_busy(f"Batch 0/{total}…")

        from .prefs import load_qc_enabled
        from .theme import group_palette_name
        qc_on = load_qc_enabled()              # read on the Tk thread
        cfg['group_palette'] = group_palette_name()

        def _work() -> tuple[int, list[tuple[str, str]]]:
            from .pipeline import FlowSample
            ok = 0
            failures: list[tuple[str, str]] = []  # (file name, why), in order
            for i, p in enumerate(fcs, 1):
                self.after(0, lambda i=i: _busy(f"Batch {i}/{total}…"))
                try:
                    s = FlowSample(p)
                    if qc_on:                    # as the editor's loader
                        s.run_qc()
                    s.auto_compensate()
                    s.apply_transform()
                    stem = os.path.basename(p).rsplit('.', 1)[0]
                    data = cap_events(s.data, cfg)
                    prep = {'data': data, 'channels': proper_run_channels(s),
                            'label': run_label({'sample': stem}),
                            'n_events': int(len(data)),
                            'note': os.path.basename(p), 'color_by': None,
                            'data_transforms': dict(s.data_transforms)}
                    r = compute_run(prep, cfg, outdir)
                    if r.get('ok'):
                        ok += 1
                    else:
                        why = r.get('error') or (
                            'cancelled' if r.get('cancelled') else 'no result')
                        failures.append((os.path.basename(p), why))
                        note_handled(f"batch folder: {os.path.basename(p)}"
                                     f" failed: {why}")
                except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
                    failures.append((os.path.basename(p),
                                     f'{type(exc).__name__}: {exc}'))
                    note_handled(f"batch folder: {os.path.basename(p)}", exc)
            return ok, failures

        def _end_busy() -> None:
            if ed is not None and hasattr(ed, '_end_busy'):
                ed._end_busy()

        def _done(ok: int, failures: list[tuple[str, str]]) -> None:
            self._status(f"Batch done: {ok} ok, {len(failures)} failed → "
                         f"{outdir}" + batch_failure_text(failures))
            try:
                ResultsViewer(self, outdir, editor=ed)
            except Exception as exc:  # noqa: BLE001 - noted; the batch outputs are already written
                note_handled("open the batch results viewer", exc)

        # Report to the editor when there is one: Load or Undo rebuilds this
        # view mid-batch, and the editor's busy bar must still stop.
        from .async_task import run_async
        run_async(ed if ed is not None else self, _work,
                  on_done=lambda r: _done(*r), on_finally=_end_busy)

    def _do_run(self) -> None:
        self._sync_run_cfg()
        if self._on_run:
            self._on_run()

    def _do_cancel(self) -> None:
        if self._on_cancel:
            self._on_cancel()

    def _do_results(self) -> None:
        if self._on_results:
            self._on_results()

    def set_running(self, running: bool) -> None:
        """Enable Cancel and disable Run while a run is going, and the reverse otherwise."""
        for btn, on in ((getattr(self, 'btn_run', None), not running),
                        (getattr(self, 'btn_cancel', None), running)):
            if btn is not None:
                try:
                    btn.configure(state='normal' if on else 'disabled')
                except tk.TclError:
                    pass


# ── Results viewer (read-only image window) ────────────────────────────────


class ResultsViewer(tk.Toplevel):
    """Read-only window showing a run's UMAP / TriMap images + cluster counts.
    Tk ≥8.6 renders PNG via PhotoImage natively (no Pillow needed)."""

    def __init__(self, parent: tk.Misc, out_dir: str, *, editor: Any = None,
                 subsample: int = 2) -> None:
        super().__init__(parent)
        self.title(f"Run results — {os.path.basename(out_dir or 'none')}")
        self.geometry('900x720')
        self._editor = editor       # enables the per-item 'Load in editor' button
        self._imgs: list[tk.PhotoImage] = []                       # keep PhotoImage refs alive
        recs = find_run_outputs(out_dir)

        top = ttk.Frame(self, padding=4)
        top.pack(fill='x')
        ttk.Label(top, text=f"{len(recs)} item(s)  ·  {out_dir}").pack(side='left')
        _ob = ttk.Button(top, text="Open folder",
                         command=lambda: self._open_folder(out_dir))
        _ob.pack(side='right')
        _tip(editor, _ob, "Show this run's output folder in your file "
             "browser: the cluster tables (CSV) and the embedding pictures "
             "(PNG).")

        body = scrolled_body(self)

        if not recs:
            ttk.Label(body, text="No results in this folder yet.",
                      style='Muted.TLabel').pack(padx=12, pady=12)
        for rec in recs:
            self._add_record(body, rec, subsample)

    def _add_record(self, parent: tk.Misc, rec: dict[str, Any],
                    subsample: int) -> None:
        head = rec['label']
        if rec['clusters'] is not None:
            head += f"   ·   {rec['clusters']} clusters"
        lf = ttk.LabelFrame(parent, text=head)
        lf.pack(fill='x', padx=6, pady=4)
        # Import the per-event result back into the editor as a sample (its
        # clusters/embeddings become populations to gate + plot).
        _ed = getattr(self, '_editor', None)
        if rec.get('events') and _ed is not None:
            act = ttk.Frame(lf)
            act.pack(fill='x', pady=(2, 0))
            _lb = ttk.Button(
                act, text="Load in editor",
                command=lambda p=rec['events'], e=_ed: (
                    e._import_processed_csv(p)))
            _lb.pack(side='left')
            _tip(_ed, _lb, "Add this result to the editor as a new sample. "
                 "Its clusters become populations, and its embedding "
                 "becomes two channels you can plot and gate.")
            ttk.Label(act, text="(adds clusters + embedding coords as a sample)",
                      style='Muted.TLabel').pack(side='left', padx=6)
        row = ttk.Frame(lf)
        row.pack(fill='x')
        for key, title in (('umap', 'UMAP'), ('tsne', 't-SNE'),
                           ('phate', 'PHATE'), ('trimap', 'TriMap'),
                           ('pacmap', 'PaCMAP')):
            if not rec.get(key):
                continue
            cell = ttk.Frame(row)
            cell.pack(side='left', padx=6, pady=4)
            ttk.Label(cell, text=title).pack()
            img = self._load_image(rec.get(key), subsample)
            if img is not None:
                ttk.Label(cell, image=img).pack()
            else:
                ttk.Label(cell, text='(not generated)',
                          style='Muted.TLabel').pack()

    def _load_image(self, path: str | None, subsample: int) -> tk.PhotoImage | None:
        if not path or not os.path.exists(path):
            return None
        try:
            img = tk.PhotoImage(file=path)
            if subsample and subsample > 1:
                img = img.subsample(subsample, subsample)
            self._imgs.append(img)
            return img
        except tk.TclError:
            return None

    @staticmethod
    def _open_folder(path: str) -> None:
        if not path or not os.path.isdir(path):
            return
        try:
            import subprocess
            import sys
            if sys.platform == 'win32':
                os.startfile(os.path.abspath(path))      # noqa: S606
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
        except OSError:
            pass


# ── Panel (hosts the single workspace) ─────────────────────────────────────


class WorkspacePanel(ttk.Frame):
    """Hosts the single Pipeline Workspace (one model + view) with Pop-out and
    Import-all. Drag routing delegates to whichever view is live."""

    def __init__(self, parent: tk.Misc, editor: Any = None,
                 on_before_change: _Hook | None = None) -> None:
        super().__init__(parent)
        self._editor = editor
        # Called BEFORE any workspace mutation so the editor can push an undo
        # checkpoint — this is what lets the editor's existing Undo button also
        # revert workspace changes.
        self._on_before_change = on_before_change
        self.model = WorkspaceModel('Pipeline')
        self._view: PipelineWorkspaceView | None = None
        self._pop: tuple[tk.Toplevel, PipelineWorkspaceView] | None = None
        self._placeholder: ttk.Label | None = None
        # Process-based job runner: a queue of items, one child process at a
        # time; cancel terminates it, a crash/OOM requeues it once.
        self._cur: dict | None = None
        self._jobs: collections.deque[dict] = collections.deque()
        self._cancel_requested = False
        self._total = 0
        self._idx = 0
        self._ok = 0
        self._err = 0
        self._out_dir: str | None = None
        self.last_run_dir: str | None = None
        self.status_var = tk.StringVar(
            value="Drag samples / leaves / whole trials in; drop beads on Comp, FMOs on FMO.")
        self._build()
        self._embed_view()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        bar = ttk.Frame(self, padding=(2, 2))
        bar.grid(row=0, column=0)                # no sticky → centred in col
        _ed = self._editor

        def _view_call(name: str) -> None:
            v = self._active_view()
            if v is not None and hasattr(v, name):
                getattr(v, name)()

        # Bar order (left→right): ✕ · Pop out · Save / Load. Import / Edit moved
        # into the view toolbar (left of ＋ Group) per layout review.
        _cl = ttk.Button(bar, text="✕", width=3, command=self._close_clicked)
        _cl.pack(side='left')
        _tip(_ed, _cl, "Hide the Pipeline Workspace (re-docks it first if "
             "floated). Reopen from the Workspace button or View menu.")

        self._popbtn = ttk.Button(bar, text="Pop out", width=8,
                                  command=self._popout_clicked)
        self._popbtn.pack(side='left', padx=(4, 0))
        _tip(_ed, self._popbtn, "Float the whole workspace into its own window "
             "(drag to a second monitor); click Dock or close it to re-dock.")
        # "On top" toggle — shown only while popped out (the editor packs/forgets
        # it via _show_ws_ontop_cb). Applies always-on-top to the float.
        self._ontop_var = tk.BooleanVar(value=False)
        self._ontop_cb = ttk.Checkbutton(bar, text="On top",
                                         variable=self._ontop_var,
                                         command=self._on_ontop_toggle)
        _tip(_ed, self._ontop_cb,
             "Keep the floating workspace above the main window.")

        # Save / Load menu — run recipes, the workspace layout, and batch-folder.
        _sl = ttk.Menubutton(bar, text="Save / Load")
        _slm = tk.Menu(_sl, tearoff=0)
        _slm.add_command(label="Save recipe…",
                         command=lambda: _view_call('_save_recipe'))
        _slm.add_command(label="Load recipe…",
                         command=lambda: _view_call('_load_recipe'))
        _slm.add_separator()
        _slm.add_command(label="Save workspace…", command=self._save)
        _slm.add_command(label="Load workspace…", command=self._load)
        _slm.add_separator()
        _slm.add_command(label="Batch folder…",
                         command=lambda: _view_call('_batch_folder'))
        _sl['menu'] = _slm
        _sl.pack(side='left', padx=(4, 0))
        _tip(_ed, _sl, "Save / load a run recipe (settings) or the whole "
             "workspace layout, or batch-run a folder unattended.")

        self._host = ttk.Frame(self)
        self._host.grid(row=1, column=0, sticky='nsew')
        self._host.columnconfigure(0, weight=1)
        self._host.rowconfigure(0, weight=1)
        status = ttk.Frame(self, padding=(4, 2))
        status.grid(row=2, column=0, sticky='ew')
        # WRAP, don't widen. An unwrapped Label reports its full single-line
        # text as its required width -- this one asked for 456 px and, being
        # the widest child, made that the whole panel's requirement. The pane
        # opened at 319 px, so the toolbar (which needs only 336) was clipped
        # and 'Group v' lost its arrow. A status message must never decide how
        # wide the workspace is; it re-wraps to whatever width the panel has.
        _status_lbl = ttk.Label(status, textvariable=self.status_var,
                                style='Muted.TLabel', wraplength=200,
                                justify='left')
        _status_lbl.pack(side='left', fill='x', expand=True)
        status.bind(
            '<Configure>',
            lambda e, _l=_status_lbl: _l.configure(
                wraplength=max(120, e.width - 12)))
        self.bind('<Destroy>', self._on_destroy)

    def _on_destroy(self, event: _Event) -> None:
        # Terminate a running (non-daemon) job so it can't keep the app alive.
        if event.widget is self:
            self._cancel_requested = True
            self._kill_current()
            self._jobs.clear()

    def _embed_view(self) -> None:
        self._view = PipelineWorkspaceView(self._host, self.model, editor=self._editor,
                                           on_status=self.status_var.set,
                                           on_run=self._run, on_cancel=self._cancel,
                                           on_results=self._open_results,
                                           on_before_change=self._before_change)
        self._view.grid(row=0, column=0, sticky='nsew')

    def _before_change(self) -> None:
        """Push an editor undo checkpoint before a workspace mutation, so the
        editor's existing Undo button also reverts workspace changes."""
        if callable(self._on_before_change):
            try:
                self._on_before_change()
            except Exception as exc:  # noqa: BLE001 - noted; the edit goes on without a checkpoint
                note_handled("take the undo checkpoint", exc)

    def _reembed(self) -> None:
        """Tear down any live / popped view and rebuild it over the current
        model (used by Load and by undo restore)."""
        if self._pop is not None:
            self._redock()
        if self._view is not None:
            self._view.destroy()
            self._view = None
        self._embed_view()

    def restore_model(self, snap: dict) -> None:
        """Replace the model from a snapshot dict (called by the editor's undo
        when reverting a step that included workspace changes)."""
        self.model = WorkspaceModel.from_dict(snap)
        self._reembed()

    # -- pop-out / re-dock --
    def _popout_clicked(self) -> None:
        """Pop-out button. When hosted by the editor, the editor floats the
        WHOLE panel (bar + view) as one window via 'wm manage'. Standalone,
        fall back to the rebuild-based pop-out below."""
        ed = self._editor
        if ed is not None and hasattr(ed, '_toggle_workspace_popout'):
            ed._toggle_workspace_popout()
        else:
            self._toggle_popout()

    def _on_ontop_toggle(self) -> None:
        """On-top checkbox — ask the editor to (un)float-above-main. Only has an
        effect while the workspace is popped out into its own window."""
        ed = self._editor
        if ed is not None and hasattr(ed, '_apply_ws_ontop'):
            ed._apply_ws_ontop(self._ontop_var.get())

    def _close_clicked(self) -> None:
        """✕ Close button — hide the workspace (the editor re-docks it first if
        it's floated). Standalone (no editor), re-dock a pop-out if any."""
        ed = self._editor
        if ed is not None and hasattr(ed, '_close_workspace'):
            ed._close_workspace()
        elif self._pop is not None:
            self._redock()

    def _toggle_popout(self) -> None:
        if self._pop is not None:
            self._redock()
            return
        if self._view is not None:
            self._view.destroy()
            self._view = None
        self._placeholder = ttk.Label(
            self._host, style='Muted.TLabel',
            text="(popped out: close that window to dock it again)")
        self._placeholder.grid(row=0, column=0)
        top = tk.Toplevel(self)
        top.independent = True  # type: ignore[attr-defined]  # a work window
        top.title(f"Workspace — {self.model.title}")
        top.geometry('560x620')
        pv = PipelineWorkspaceView(top, self.model, editor=self._editor,
                                   on_status=self.status_var.set,
                                   on_run=self._run, on_cancel=self._cancel,
                                   on_results=self._open_results,
                                   on_before_change=self._before_change)
        pv.pack(fill='both', expand=True)
        self._pop = (top, pv)
        top.protocol('WM_DELETE_WINDOW', self._redock)
        self.status_var.set("Popped out. Close the window to re-dock.")

    def _redock(self) -> None:
        if self._pop is None:
            return
        top, _pv = self._pop
        self._pop = None
        try:
            top.destroy()
        except tk.TclError:
            pass
        if self._placeholder is not None:
            try:
                self._placeholder.destroy()
            except tk.TclError:
                pass
            self._placeholder = None
        self._embed_view()
        self.status_var.set("Re-docked.")

    def popped_count(self) -> int:
        """Number of popped-out workspace windows: 1 while popped out, else 0."""
        return 1 if self._pop is not None else 0

    # -- drop routing (from the editor) --
    def _live_views(self) -> list[PipelineWorkspaceView]:
        vs: list[PipelineWorkspaceView] = []
        if self._view is not None:
            vs.append(self._view)
        if self._pop is not None:
            vs.append(self._pop[1])
        return vs

    def _active_view(self) -> PipelineWorkspaceView | None:
        vs = self._live_views()
        return vs[0] if vs else None

    def point_in_target(self, x_root: int, y_root: int) -> PipelineWorkspaceView | None:
        """The live view whose tree rectangle contains the screen point, or
        None. Geometry-based so it works for a popped-out (floating) workspace,
        where winfo_containing is unreliable during a drag's pointer grab."""
        for v in self._live_views():
            if v.contains_point(x_root, y_root):
                return v
        return None

    def drop_at(self, editor: Any, nodes: Sequence[_Node], x_root: int,
                y_root: int) -> bool:
        """Find the workspace view under the point and route the drop there
        (column-aware). Falls back to the active view (e.g. headless)."""
        v = self.point_in_target(x_root, y_root) or self._active_view()
        return bool(v and v.drop_at(editor, nodes, x_root, y_root))

    # -- run (M2): Phenograph + UMAP + TriMap, off the Tk thread --------------
    def _set_running(self, running: bool) -> None:
        for v in self._live_views():
            v.set_running(running)

    def _running(self) -> bool:
        return self._cur is not None or bool(self._jobs)

    def _run(self) -> None:
        if self._running():
            self.status_var.set("A run is already in progress.")
            return
        if self._editor is None:
            self.status_var.set("No editor attached — cannot resolve samples.")
            return
        base_cfg = dict(self.model.run_cfg)
        units = build_run_units(self.model, base_cfg)
        if not units:
            self.status_var.set("Nothing to run — add items to the workspace first.")
            return
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._out_dir = os.path.abspath(os.path.join('outputs', 'workspace', f'run_{ts}'))
        self.last_run_dir = self._out_dir
        # One job per RUN UNIT: a group's samples co-embed into one UMAP; loose
        # items are their own unit; 'concatenate' merges everything into one.
        # Each carries its own cfg copy so a retry can lower max_events alone.
        # Per-unit cfg = base, with the unit's group override merged on (only
        # in non-concat runs, where each unit maps to one group).
        def _unit_cfg(u: dict[str, Any]) -> dict:
            # A loose unit has no gid: get(None) finds no group and gives {}.
            ov = self.model.groups.get(u.get('gid'), {}).get('cfg')  # pyright: ignore[reportCallIssue, reportArgumentType]
            return merge_unit_cfg(base_cfg, ov)
        self._jobs = collections.deque(
            {'unit': u, 'label': u['label'], 'attempt': 1, 'cfg': _unit_cfg(u)}
            for u in units)
        self._total = len(self._jobs)
        self._idx = self._ok = self._err = 0
        self._warnings: list[str] = []  # per-unit notes, repeated in the final line
        self._cancel_requested = False
        self._cur = None
        self._prep: dict[str, Any] | None = None  # unit being serialised off-thread
        self._set_running(True)
        mode = 'concatenated' if base_cfg.get('concatenate') else 'per-group'
        self.status_var.set(f"Queued {self._total} run(s) [{mode}] → {self._out_dir} …")
        self.after(50, self._pump_jobs)

    def _pump_jobs(self) -> None:
        """Main-thread driver: one child process at a time, cancel terminates,
        crash/OOM requeues once with fewer events."""
        if self._cancel_requested:
            self._kill_current()
            self._jobs.clear()
            self._finish(f"Cancelled. → {self._out_dir}")
            return
        if self._cur is not None and self._service_current():
            self.after(150, self._pump_jobs)        # still running
            return
        # A unit being serialised to disk is a THIRD state: nothing is running
        # yet, but the queue must not advance and the run is not done.
        if getattr(self, '_prep', None) is not None:
            if not self._finish_prep():
                self.after(150, self._pump_jobs)    # still writing job.pkl
                return
        if self._cur is None and getattr(self, '_prep', None) is None \
                and self._jobs:
            self._launch_next()
        if self._cur is None and getattr(self, '_prep', None) is None \
                and not self._jobs:
            tail = f"{self._ok} ok" + (f", {self._err} failed" if self._err else "")
            warned = getattr(self, '_warnings', None) or []
            self._finish(f"Done: {tail} → {self._out_dir}"
                         + ''.join(f"  [!] {w}" for w in warned))
            return
        self.after(150, self._pump_jobs)

    def _launch_next(self) -> None:
        import pickle
        import tempfile
        job = self._jobs.popleft()
        self._idx += 1
        n = self._idx
        try:
            prep = prepare_unit(self._editor, job['unit']['members'],
                                job['unit']['label'], job['cfg'])      # cheap (main)
        except Exception as e:  # noqa: BLE001 - a unit that will not prepare is counted as failed with its error in the status line; the queue goes on
            self._err += 1
            self.status_var.set(f"[{n}/{self._total}] {job['label']}: ✗ {type(e).__name__}: {e}")
            return
        jobdir = tempfile.mkdtemp(prefix='wsjob_')
        job_path = os.path.join(jobdir, 'job.pkl')
        result_path = os.path.join(jobdir, 'result.pkl')

        # Serialise OFF the Tk thread. This is the dominant cost of starting a
        # unit — measured on this machine, writing job.pkl took 569 ms for a
        # 20-sample x 200k run (336 MB) and 1264 ms for 8 x 1M (672 MB), all
        # of it blocking the UI.
        #
        # It is NOT disk-bound. Measured on a 192 MB payload: 0.18s to
        # serialise into memory and 0.37s to serialise and write, so the
        # split is about 49% CPU and 51% disk. The CPU half holds the GIL,
        # which is normally the reason a thread does NOT unblock a Tk UI.
        #
        # It works anyway, measured directly rather than argued from the
        # split: heartbeating the main thread at 10 ms while a worker
        # thread wrote the same payload gave a median stall of 0.5 ms, p95
        # 4.3 ms, max 40 ms — responsive throughout. pickle releases the GIL
        # for the large buffer writes numpy arrays serialise into, so the CPU
        # half is not the GIL-monopolising kind. Verify with
        # scratchpad-style heartbeat timing if this payload shape changes.
        #
        # Only the WRITE moves. `prepare_unit` above stays on the Tk thread on
        # purpose: it reads `editor._samples`, which the user can mutate, and
        # running it concurrently would introduce a read-during-write race,
        # the one the plot path guards against. `prep` is a freshly built
        # object no one else references, so pickling it in a thread shares
        # nothing.
        #
        # The child reads no preferences, so the group palette of its by-source
        # plots travels in the job's cfg, read here from the user's choice.
        from .theme import group_palette_name
        payload = {'prep': prep,
                   'cfg': dict(job['cfg'], group_palette=group_palette_name()),
                   'out_dir': self._out_dir}
        state: dict[str, BaseException | None] = {'error': None}

        def _write() -> None:
            try:
                with open(job_path, 'wb') as fh:
                    pickle.dump(payload, fh)
            except Exception as exc:                    # noqa: BLE001 - writer thread: any pickling or write error is kept in state and reported when the thread is joined
                state['error'] = exc

        writer = threading.Thread(target=_write, daemon=True)
        writer.start()
        self._prep = {'thread': writer, 'state': state, 'job': job, 'n': n,
                      'jobdir': jobdir, 'job_path': job_path,
                      'result_path': result_path}
        self.status_var.set(
            f"[{n}/{self._total}] {job['label']}: preparing… "
            f"({prep.get('n_events', 0):,} events)")

    def _finish_prep(self) -> bool:
        """Serialisation done? If so, launch the child. Returns False while the
        write is still in flight, so the caller keeps polling."""
        import shutil
        import subprocess
        import sys
        pr = self._prep
        if pr is None:
            return True
        if pr['thread'].is_alive():
            return False
        self._prep = None
        job, n, jobdir = pr['job'], pr['n'], pr['jobdir']
        err = pr['state']['error']
        if err is not None:
            shutil.rmtree(jobdir, ignore_errors=True)
            self._err += 1
            self.status_var.set(
                f"[{n}/{self._total}] {job['label']}: ✗ could not stage the "
                f"run: {type(err).__name__}: {err}")
            return True
        try:
            # Launch a REAL subprocess running the guarded module entry point.
            # `-m openflo.workspace_run` (not `-c`) so Phenograph's own pool can
            # re-import the module cleanly on Windows spawn; that module is
            # Tk-free, so the child and its pool never load Tk. stderr→stdout; our
            # progress lines are marker-prefixed (PROGRESS_MARKER), the result comes back
            # via result.pkl. Cancel = taskkill the whole tree (see _kill_current).
            # CREATE_NO_WINDOW: when the GUI runs under pythonw (no console), the
            # child python.exe would otherwise allocate and flash its own console
            # window on every run — suppress it (and Phenograph's spawned pool,
            # which inherits the no-console state). No-op off Windows.
            # start_new_session (POSIX): the child leads its own process group,
            # which Phenograph's pool and the Louvain binaries join, so Cancel
            # can signal the whole tree (see _kill_current). No-op on Windows.
            proc = subprocess.Popen(
                [sys.executable, '-m', 'openflo.workspace_run',
                 pr['job_path'], pr['result_path']],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                start_new_session=(sys.platform != 'win32'))
        except Exception as e:  # noqa: BLE001 - a child that will not start is counted as failed in the status line; the temp dir is removed and the queue goes on
            shutil.rmtree(jobdir, ignore_errors=True)   # don't leak the temp dir
            self._err += 1
            self.status_var.set(f"[{n}/{self._total}] {job['label']}: ✗ launch failed: {e}")
            return True
        msgq: queue.Queue[str] = queue.Queue()
        reader = threading.Thread(target=self._read_stdout, args=(proc, msgq), daemon=True)
        reader.start()
        self._cur = {'proc': proc, 'msgq': msgq, 'job': job, 'n': n,
                     'result_path': pr['result_path'], 'jobdir': jobdir,
                     'lasterr': ''}
        self.status_var.set(f"[{n}/{self._total}] {job['label']}: running… (separate process)")
        return True

    @staticmethod
    def _read_stdout(proc: subprocess.Popen[str], msgq: queue.Queue[str]) -> None:
        """Reader thread: push the child's stdout lines onto a thread-safe queue.
        Touches NO Tk — the main thread drains the queue in _service_current."""
        # _finish_prep starts the child with stdout=PIPE, so stdout is set.
        try:
            for line in proc.stdout:  # pyright: ignore[reportOptionalIterable]
                msgq.put(line.rstrip('\r\n'))
        except Exception as exc:  # noqa: BLE001 - noted; the reader thread must end quietly
            note_handled("read the job's output", exc)
        finally:
            try:
                proc.stdout.close()  # pyright: ignore[reportOptionalMemberAccess]
            except Exception:  # noqa: BLE001 - output pipe may already be closed
                pass

    def _absorb_stdout(self, cur: dict) -> None:
        try:
            while True:
                line = cur['msgq'].get_nowait()
                if line.startswith(PROGRESS_MARKER):
                    self.status_var.set(f"[{cur['n']}/{self._total}] {line[len(PROGRESS_MARKER):]}")
                elif line.strip():
                    cur['lasterr'] = line.strip()       # keep last line for diagnostics
        except queue.Empty:
            pass

    def _service_current(self) -> bool:
        """Drain the child's stdout; return True while it's still running."""
        cur = self._cur
        if cur is None:
            return False
        self._absorb_stdout(cur)
        if cur['proc'].poll() is None:
            return True
        self._finalize_current()
        return False

    def _finalize_current(self) -> None:
        import pickle
        import shutil
        cur = self._cur
        if cur is None:
            return
        self._cur = None
        self._absorb_stdout(cur)                        # final lines
        job, proc = cur['job'], cur['proc']
        res = None
        try:
            if os.path.exists(cur['result_path']):
                with open(cur['result_path'], 'rb') as fh:
                    res = pickle.load(fh)
        except Exception:  # noqa: BLE001 - noted; an unreadable result reports the job failed
            note_handled("reading a workspace job result")
            res = None
        shutil.rmtree(cur['jobdir'], ignore_errors=True)
        if res is None:
            code = proc.poll()
            detail = f" — {cur['lasterr']}" if cur.get('lasterr') else ''
            # Retry ONLY a real crash/kill (nonzero exit), once, at fewer events.
            # A clean exit (0) with no result is a logic error — record it rather
            # than re-run, so a unit can never be launched a second time spuriously.
            if code not in (0, None) and job['attempt'] < 2:
                job['attempt'] += 1
                old = int(job['cfg'].get('max_events') or 0)
                job['cfg']['max_events'] = (max(MIN_CUSTOM_EVENTS, old // 2)
                                            if old else 2000)
                self._jobs.append(job)
                self.status_var.set(
                    f"[{cur['n']}/{self._total}] {job['label']}: crashed (exit {code}){detail} "
                    f"— retrying once at max_events={job['cfg']['max_events']}")
            else:
                self._err += 1
                self.status_var.set(
                    f"[{cur['n']}/{self._total}] {job['label']}: ✗ no result (exit {code}){detail}")
            return
        if res.get('ok'):
            self._ok += 1
            parts = [f"{res['n_clusters']} clusters", run_events_text(res)]
            for _k, _lbl in (('umap', 'UMAP'), ('tsne', 't-SNE'),
                             ('phate', 'PHATE'), ('trimap', 'TriMap'),
                             ('pacmap', 'PaCMAP')):
                if res.get(_k):
                    parts.append(_lbl)
            warned = res.get('warnings') or []
            if warned:
                # Kept for the run's final line too: this one is replaced
                # by the next job's.
                self._warnings = getattr(self, '_warnings', []) + [
                    f"{job['label']}: {w}" for w in warned]
            self.status_var.set(
                f"[{cur['n']}/{self._total}] {job['label']}: ✓ "
                + ", ".join(parts)
                + ''.join(f"  [!] {w}" for w in warned))
        else:
            self._err += 1
            self.status_var.set(f"[{cur['n']}/{self._total}] {job['label']}: ✗ {res.get('error')}")

    def _kill_current(self) -> None:
        import shutil
        import subprocess
        import sys
        # A unit may be mid-serialisation rather than running. The writer is a
        # daemon thread that cannot be interrupted safely, so let it finish and
        # reap its temp dir afterwards instead of leaking it (or racing it).
        pr, self._prep = getattr(self, '_prep', None), None
        if pr is not None:
            def _reap(thread: threading.Thread = pr['thread'],
                      jobdir: str = pr['jobdir']) -> None:
                try:
                    thread.join(timeout=60)
                except Exception:  # noqa: BLE001 - stale thread reap is best-effort
                    pass
                shutil.rmtree(jobdir, ignore_errors=True)
            threading.Thread(target=_reap, daemon=True).start()
        cur, self._cur = self._cur, None
        if cur is None:
            return
        proc = cur['proc']
        # The process group the child leads (start_new_session in
        # _launch_next), read while it is still alive so its id cannot have
        # been reused by the time it is signalled.
        pgid = own_process_group(proc)
        try:
            if proc.poll() is None:
                if sys.platform == 'win32':
                    # Kill the whole tree: the job process AND Phenograph's pool.
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                                   capture_output=True, timeout=10,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                else:
                    # The whole group, as taskkill /T does on Windows.
                    # terminate() alone stopped only the job process: on Linux
                    # and macOS Phenograph's pool and the Louvain binaries kept
                    # running after Cancel (test_workspace_job_driver caught a
                    # grandchild still alive).
                    signal_process_group(proc, pgid, 'SIGTERM')
                try:
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001 - any wait failure falls through to kill()
                    proc.kill()
            # Anything of the tree that outlived its leader.
            signal_process_group(None, pgid, 'SIGKILL')
        except Exception as exc:  # noqa: BLE001 - noted; Cancel must finish and clean up the job folder
            note_handled("stop the job's process tree", exc)
        shutil.rmtree(cur['jobdir'], ignore_errors=True)

    def _finish(self, msg: str) -> None:
        self._cur = None
        self._prep = None
        self._jobs.clear()
        self._set_running(False)
        self.status_var.set(msg)

    def _cancel(self) -> None:
        if self._running():
            self._cancel_requested = True
            self.status_var.set("Cancelling… (terminating the running job).")

    def _open_results(self) -> None:
        d = self.last_run_dir
        if not d or not os.path.isdir(d):
            self.status_var.set("No results yet — run the workspace first.")
            return
        try:
            ResultsViewer(self, d, editor=self._editor)
        except Exception as e:  # noqa: BLE001 - viewer must never crash the app
            self.status_var.set(f"Could not open results: {type(e).__name__}: {e}")

    # -- persistence (M6) --
    def _save(self) -> None:
        import json
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            parent=self, title="Save workspace", defaultextension='.json',
            filetypes=[('Workspace JSON', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(self.model.to_dict(), fh, indent=2)
            n = sum(1 for _ in self.model.all_items())
            self.status_var.set(f"Saved {n} item(s) to {os.path.basename(path)}.")
        except Exception as e:  # noqa: BLE001 - any write or serialise failure is shown in the status line
            self.status_var.set(f"Save failed: {type(e).__name__}: {e}")

    def _load(self) -> None:
        import json
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            parent=self, title="Load workspace",
            filetypes=[('Workspace JSON', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as fh:
                raw = json.load(fh)
            loaded = WorkspaceModel.from_dict(raw)
        except Exception as e:  # noqa: BLE001 - a file that will not read as a workspace is shown in the status line; the open one is kept
            self.status_var.set(f"Load failed: {type(e).__name__}: {e}")
            return
        note = newer_version_note(raw if isinstance(raw, dict) else {})
        if note:
            from tkinter import messagebox
            messagebox.showwarning("Workspace from a newer OpenFlo", note,
                                   parent=self)
        self._before_change()              # loading is undoable
        self.model = loaded
        self._reembed()
        n = sum(1 for _ in self.model.all_items())
        self.status_var.set(f"Loaded {n} item(s) from {os.path.basename(path)}.")


# ── Subprocess job entry point ─────────────────────────────────────────────
# The panel launches `python -m openflo.workspace_run <job.pkl> <result.pkl>`,
# which loads no Tk. `python -m openflo.workspace <job.pkl> <result.pkl>` runs
# the same job and stays supported for scripts that call it; guarded (NOT
# `python -c`) so Phenograph's own worker pool can re-import this module
# cleanly under Windows `spawn`.
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    subprocess_main()
