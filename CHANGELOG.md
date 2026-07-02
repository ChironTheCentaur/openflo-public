# Changelog

All notable changes to OpenFlo are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.2] - 2026-07-02

### Fixed
- **Compensation-QC "strong pairs"** now includes spillover exactly at the 0.10
  threshold (a strict comparison previously excluded it).
- **Spectral unmixing condition number** reports `inf` for an underdetermined
  panel (more fluorophores than detectors) instead of a misleadingly-small value.
- **Differential abundance** proportions use each sample's true total event count
  as the library size; nested/overlapping populations previously inflated it and
  understated the displayed percentages.
- **Workspace ↔ FlowJo comparison** no longer collapses populations that share a
  name (e.g. quadrant Q1–Q4, copied gates) — each is compared against its own gate.
- **CytoNorm** models whose batch label contains `|` (e.g. a POSIX path) reload
  correctly instead of failing.
- **Batch run outputs** no longer silently overwrite one another when two run
  units sanitise to the same filename (colliding labels are suffixed); a failed
  subprocess launch no longer leaks a temp directory.
- **Friedman test** no longer suppresses a perfectly-concordant (maximally
  significant) result.
- **Histogram (symlog view)**: bin spacing and the axis scale now share one
  anchor, removing uneven bins.

## [2.2.1] - 2026-07-01

### Fixed
- **Compensation was applying the transposed inverse spillover.** The internal
  apply step computed `data @ inv(M).T` instead of `data @ inv(M)`, which left
  asymmetric spillover uncorrected and corrupted otherwise-clean channels —
  affecting every compensated dataset with a non-symmetric spillover matrix (i.e.
  essentially all real data). **This is a correctness fix: compensated values,
  and everything derived from them (transforms, clustering, gating, exported
  populations), will change — for the better.** Re-run compensation on affected
  analyses; the stored spillover matrix itself is unchanged, so sessions reload
  fine and simply recompute.
- **Differential abundance could report enrichment backwards.** The results table
  labelled the two groups (and the log2FC sign) by sample-load order while the
  GLM fitted them in the count matrix's (alphabetical) column order; the table
  now matches the fitted direction.
- **Histogram highlight mode could crash** (`ValueError`) when highlighting two or
  more gates/samples at once.

## [2.2.0] - 2026-07-01

### Added
- **Reproducible clustering.** PhenoGraph's default Louvain community detection
  is not seed-reproducible (its community binary is time-seeded), so cluster
  labels could differ run-to-run. A new opt-in **Reproducible** mode routes
  PhenoGraph to its seeded Leiden backend so a re-run gives identical clusters —
  available in the Cluster dialog, the Pipeline Workspace Run bar, and the CLI
  (`--reproducible`). Off by default, so default Louvain results are unchanged;
  Leiden, FlowSOM, and all embeddings were already deterministic.

### Fixed
- **FlowJo import — max-only 1-D gates are no longer dropped.** A `RectangleGate`
  with only an upper bound (`x < hi`, no min) was silently discarded and its
  child populations re-parented to the grandparent, quietly loosening every
  descendant; it now imports as a bounded interval, keeping the constraint.
- **Quadrant gates tile the plane.** Events landing exactly on a divider (e.g. a
  value at 0 on an arcsinh axis) fell into no quadrant; the four quadrants are
  now a true partition.
- **Gate editor — invisible handles are no longer grabbable.** In histogram mode
  (or for a degenerate polygon) a 2-D gate's hidden handles could be dragged,
  silently corrupting its bounds; hit-testing is now tied to what's actually
  drawn. Quadrant shift-click-add also honors compound modifiers (e.g. Ctrl+Shift).
- **Auto-clean — freezing a valley-mode debris gate keeps granulocytes.** Freezing
  or copying it collapsed the 2-D FSC×SSC rescue into a 1-D floor, dropping
  low-FSC/high-SSC granulocytes; it now pins both thresholds and reproduces the
  full 2-D cut identically across samples.
- **Autosave no longer leaks disk.** Orphaned autosave `_data` sidecar folders
  (processed event-table CSVs) were never pruned; they are now removed alongside
  their session file.
- **Batch runs report empty clustering as a failure.** A FlowSOM/Leiden run below
  its event floor produced no labels yet was reported as a successful 0-cluster
  run; it is now surfaced as an error.
- **Loader no longer grows `sys.path`.** Each FCS load prepended a duplicate path
  entry from the worker threads; the redundant insert was removed.

## [2.1.0] - 2026-07-01

### Added
- **Portable session relink.** Saved sessions now record each sample's file
  basename and a path relative to the session file, so a project that's been
  moved, copied, or opened on another machine re-finds its raw FCS once it sits
  beside the `.flowsession` — resolving stored absolute path → session-relative
  path → basename in the session folder. Additive to the session format (older
  sessions keep opening; the fields live inside each sample entry).
- **Choose clustering markers.** The Cluster dialog now has a marker picker that
  restricts PhenoGraph / Leiden / FlowSOM to a chosen subset of fluorochrome
  channels; leaving all selected keeps the previous "all markers" behaviour.
- **Cell-cycle gating controls.** The cell-cycle dialog exposes the doublet-cut
  strength (`k`) and the singlet-tolerance window, which were previously fixed.

### Changed
- **Tighter self-test reproducibility contract.** The golden self-test
  tolerances were tightened to match each metric's real reproducibility — the
  bit-exact metrics (auto-clean debris/doublets and the compensation spill) to
  exact, and MESF calibration slope/R² to 1e-3 — so behavioural drift is caught
  far sooner. Desktop and web baselines are kept in lockstep. Platform-variable
  metrics (viability, Leiden count) intentionally stay loose.

## [2.0.1] - 2026-07-01

### Fixed
- **Compensated samples now reattach correctly on session reopen.** A sample
  that was compensated but not clustered/embedded wasn't persisted to the
  processed-data sidecar, so reopening a session reloaded the raw (uncompensated)
  FCS and any gate/population drawn in compensated space selected the wrong
  events. Compensated samples are now persisted like any other computed result.

## [2.0.0] - 2026-06-30

### Added
- **Vendor-portable GPU acceleration (PyTorch backend).** The opt-in load-math
  acceleration (compensation matmul, logicle LUT interp, arcsinh) now runs on a
  PyTorch backend alongside CuPy, reaching **AMD (ROCm), Intel (XPU), Apple
  (MPS), and any Direct3D-12 GPU on Windows via DirectML** — not just NVIDIA.
  Backend preference `auto|cupy|torch|off` (env `OPENFLO_GPU_BACKEND`);
  Preferences shows the detected device. Extras: `[gpu-torch]`, `[gpu-dml]`.
  Still OFF by default with the exact numpy fallback, so the golden baseline is
  unchanged.
- **RAPIDS 26.06 GPU-clustering image.** `docker/Dockerfile.rapids` moves to the
  RAPIDS 26.06 base, which ships numpy 2.4.6 (our exact pin); cuDF/cuML/cuGraph
  import, `leidenalg==0.11.0` keeps Leiden golden-exact, golden 7/7 in-container.
  pandas 3.0.3 remains an upstream cuDF wall (documented in `docs/RAPIDS_SHIM.md`).
- **Configurable loader concurrency + priority.** The background file-load pool
  now sizes itself from your CPU & RAM (instead of a fixed 2), with an override
  in **Edit ▸ Preferences ▸ Performance** (“Concurrent file loaders”, Auto or
  1–8, persisted). The pool is a priority queue: the active / first-rendered
  sample loads first so its plot appears soonest, and loader threads run at
  lower OS priority so a big batch doesn't starve the UI.

### Changed
- **Loading feedback now covers session resume, not just Add-FCS.** Reopening a
  session paints a muted “⏳ name” row for *every* sample up front (grouped by
  trial) before any file is read, so a large session fills the tree immediately
  instead of looking frozen. Processed samples (workspace results with
  clusters/UMAP) now load on the same background pool as raw FCS — the window
  stays responsive even with large sidecars — each row swapping to its real
  entry as it lands. Still-loading rows prefer the trial recorded in the session.

### Fixed
- **Auto-clean now applies to every selected sample**, not just the
  active/last-selected one.
- **Toggling an auto-clean gate/method refreshes the cleaned-out-events overlay**
  — previously, in “all” display mode it only redrew gate lines, leaving the red
  removed-events dots stale.
- **Right-click menu rendering on dark themes** — the context menu is now themed
  (incl. `disabledforeground`), so a greyed-out Paste no longer looks garbled.
- **Auto-clean removal is clearer** — the menu item reads “Remove auto-clean
  gate”, and the debris method row shows its effective cut
  (`[beads]` / `[valley]` / `[beads→valley: no bead file]`) so switching modes
  with no bead file in the run isn't a silent no-op.

## [1.6.0] - 2026-06-25

### Added
- **Help ▸ Run diagnostics…** — an install health check for when something
  behaves oddly or an install looks corrupted. Reports whether core
  dependencies match their pinned versions, which optional engines are present,
  and whether the seeded behavioural self-test still reproduces the golden
  baseline. It runs in a *separate* process, so a genuinely broken install (or a
  native-library crash) is reported instead of taking the editor down. Also
  available standalone for when the GUI won't start: `openflo-doctor`,
  `python -m openflo.diagnostics` (`--json` / `--quick`), or
  `scripts/diagnose.bat` / `scripts/diagnose.sh`.

### Changed
- **Faster startup.** The Pipeline Workspace panel (~100 ms to build) is now
  constructed lazily on first reveal instead of eagerly at window open, so a
  session that never opens it doesn't pay for it.
- **Loading feedback for multi-file / large loads.** Every queued FCS now shows
  as a muted “⏳ name” row in the samples tree *immediately* and stays there
  until that file finishes — previously the placeholders were wiped on the first
  file's load and samples popped in one-by-one, which made a big load look
  stalled. The first sample still renders as soon as it's ready; the rest remain
  visibly “loading” (with the existing N/N progress bar) until each lands.

### Tested
- New `tests/test_editor_mixins.py` guards the editor mixin decomposition:
  every `editor_*` module imports standalone, every `*Mixin` is actually mixed
  into `ViewGateEditorWindow`, no two mixins shadow each other's method names,
  and the `gui` back-compat re-exports stay present. New `tests/test_diagnostics.py`
  covers the health check.

## [1.5.0] - 2026-06-25

### Changed
- **Internal: `ViewGateEditorWindow` fully decomposed into mixins (no behaviour
  change).** The remaining ~190 editor methods moved out of `gui.py` into 24
  focused `editor_*` mixin modules (analysis, tools, autoclean, autogate,
  clipboard, update, figure, slider, template, drag-drop, chrome, gate-tools,
  compute, stats, populations, load-pool, downsample, mode, grouping, channels,
  lifecycle, audit; plot renderers folded into `PlotMixin`, undo checkpoints
  into `UndoMixin`). All subclass the shared `editor_base.EditorMixin`. `gui.py`
  is now ~1.8k lines (was ~7.4k): the constructor, two class-attribute
  classmethods, and tooltip glue. `messagebox`/`filedialog`/`_LOAD_POOL_SIZE`
  are re-exported from `gui` for back-compat. Pyright 0 errors, ruff clean, full
  suite green (745 passed), golden baseline 7/7.

## [1.4.5] - 2026-06-25

### Changed
- **Internal: editor decomposition continues (no behaviour change).** The
  log/console pane and the Help-menu dialogs moved out of `ViewGateEditorWindow`
  into `editor_console` / `editor_help` mixins. (Further editor mixins in
  progress.)

## [1.4.4] - 2026-06-25

### Changed
- **Internal: `gui.py` decomposed (no behaviour change).** The ~16k-line GUI
  monolith was reduced ~26% by extracting pure logic and self-contained
  windows into focused modules: `ui_logic`, `gating`, `tree_ids`, `plotmath`,
  `density`, `scales`, `paths` (pure, headless-tested), `prefs` + `theme`
  (shared palette/figure/prefs helpers), and ~24 `ui_*.py` dialog/window
  modules. Dialog modules now depend on the small shared modules rather than
  importing the whole GUI — faster imports, cleaner dependency graph. Pinned
  ruff/pyright so local and CI lint identically.

### Fixed
- Voltage-optimization dialog rendered a white plot on first open under the
  Midnight theme (now themed at build time).
- Compensation-matrix values were near-illegible on Midnight (theme-aware
  cell colours; zeros muted, used values carry the header colour).

## [1.4.3] - 2026-06-25

### Fixed
- **Help → Environment no longer freezes the UI.** The engine probe used to
  *import* each backend (umap/phate/… are slow to import) on the Tk thread; it
  now checks presence with `find_spec` and reads versions from metadata — no
  heavy import. Git SHA for the provenance stamp is cached too.
- **Pop-up figures honour the Midnight theme.** Voltage, Trajectory, and the
  other analysis dialogs rendered a white plot under the dark Midnight theme;
  dark pop-ups now follow either the "Dark figures in pop-ups" toggle *or* the
  Midnight theme.
- **Display modes greyed out without real gates.** "Highlight gated" / "Filter
  to gated" are disabled (and a stale selection falls back to "All events")
  when the active sample has no positive gates, so they can't keep drawing
  gates that were deleted. Auto-clean gates don't count as real gates.

### Added
- **Help → Environment** — lists which analysis engines (FlowIO, UMAP,
  PhenoGraph, Leiden, TriMap, PaCMAP, PHATE, AnnData, drag-and-drop, …) are
  installed, with version and a copy-paste `pip install 'openflo[extra]'` hint
  for anything missing — so a greyed-out method or skipped run is explained.
- **Provenance footer on exported figures** — every saved figure carries a
  subtle `OpenFlo <version> (<git sha>)` stamp for reproducible, paper-ready
  output. Toggle in Edit → Preferences → Export.

## [1.4.2] - 2026-06-24

### Added
- **Session format versioning + auto-migration.** `.flowsession` files now
  carry a schema version; opening an older one auto-upgrades it (with a status
  note), and one written by a newer OpenFlo is refused rather than mis-read.
  New **File → Upgrade saved session…** and a headless
  `scripts/migrate_session.py` upgrade files without opening them.
- **Save-format continuity test** (`tests/test_session_continuity.py`) locks
  the session schema (keys + version) against `openflo.session_format`, so a
  downstream-visible format change fails the suite until it's made
  intentional: bump the version, add a migration, and note it here.
- **Newer-version alerts for workspaces & recipes.** Saved workspaces and run
  recipes now record the OpenFlo version that wrote them; loading one produced
  by a newer build (newer schema or newer app version) warns that some
  features may not load and suggests updating OpenFlo (Help → Check for
  updates…).

## [1.4.1] - 2026-06-24

### Added
- **Keyboard shortcuts across the gating loop** — `Ctrl+F` find,
  `Ctrl+0` reset view, `Ctrl++`/`Ctrl+-` zoom, `F5` replot, `Esc` cancel zoom
  tool, `Ctrl+1/2/3` display mode (all / highlight / filter), `Ctrl+,`
  Preferences, `Ctrl+Shift+S` save plot image, `F9` Pipeline Workspace,
  `` Ctrl+` `` log/console, `Ctrl+T` Statistics. Menu accelerators and the
  Help → Keyboard shortcuts reference updated to match.
- **Swap X↔Y axes** — a `⇄` button between the axis pickers; each axis keeps
  its own scale/range.
- **Selected-gate readout** — selecting a gate shows its event count and
  **% of parent** (% of all events for a root gate) in the status bar;
  selecting a sample shows its total event count.
- **Type-to-filter channel pickers** — the X / Y / Color combos narrow as you
  type and snap to the matching channel on commit (helps with large panels).
- **First-run empty state** — the empty canvas offers clickable starting
  points (Add FCS / Load example / Open session) and a drag-and-drop hint.

### Fixed
- **Clear gate** now hints "Ctrl+Z to undo" in the status bar.

## [1.4.0] - 2026-06-24

### Added
- **Pipeline Workspace v2** — batch co-embedded clustering over groups of
  samples, now with:
  - **Clustering method** dropdown (PhenoGraph / Leiden / FlowSOM) with a
    **Leiden resolution** control and FlowSOM meta-cluster count.
  - **Full embedding set** — UMAP / t-SNE / PHATE / TriMap / PaCMAP, run
    together or selectively, with optional concatenation across the group.
  - **Save / Load recipe** — persist a run configuration as JSON and reload it.
  - **Import results as populations** — load a processed run's events CSV back
    into the editor as a sample.
  - **Batch over folder** — run the active recipe across every FCS in a folder.
  - **Marker picker** and **per-group parameter overrides**.
  - **Watch folder** — auto-load new FCS files as they appear.
- **CLI clustering parity** — `--cluster-method`, `--resolution`,
  `--n-metaclusters` for Leiden / FlowSOM / PhenoGraph runs.

### Changed
- **UI polish** — toolbar buttons stack into two rows (no overflow on wide
  screens), shortened the workspace tree's sample/population column, renamed
  the plot-controls **Workspace** button to **Pipeline**, and re-laid-out the
  group-parameters dialog on a clean grid.

### Fixed
- Right-click menu **Paste** label rendered garbled on dark menus (themed
  `disabledForeground`).
- Interior pane-resize lag — the matplotlib canvas no longer re-rasters on
  every pixel of a sash drag; it freezes during the drag and does one clean
  replot on release.
- Dragging a pane no longer exposes a white strip — the canvas backing matches
  the chrome background.

## [1.3.0] - 2026-06-24

### Added
- **Backend workflows surfaced in the GUI**
  - **Voltage optimization** (Tools) — PMT / stain-index titration with
    per-channel recommendations.
  - **Compare FlowJo workspace** (Tools) — re-apply a `.wsp` and compare gate
    counts vs FlowJo, with CSV export.
  - **Generate dataset** (File) — synthetic datasets (PBMC / differentiation /
    cell-cycle / spectral / beads), loaded in-app.
  - **Quick preview** (File) — raw single-sample density-scatter QC.
  - **FCS inspector** (Tools) — raw channels / keywords / spillover viewer.
- **Plot navigation** — Zoom-to tool (drag a rectangle; greys the gating
  tools while active), centered ⌂/⛶/+/- bar, middle-drag pan, wheel zoom.
- **Dark figures in pop-ups** (View) — preview + export of every analysis
  figure window on a dark background; plus a **Midnight** dark-plot theme and a
  **New windows open at** corner toggle.
- **Flow-cytometry tools**
  - **Singlet gate** (Edit → Add singlet gate) — FSC-A vs FSC-H singlet
    polygon from the robust height/area band.
  - **FMO gating** (Edit → FMO gating…) — map markers to FMO controls; places
    threshold gates at the FMO percentile.
  - **Compensation QC** (Tools → Compensation QC…) — spillover heatmap +
    metrics for the active sample's matrix.
  - **Absolute counts** (Tools → Absolute counts…) — counting-bead cells/µL.
  - **Gating-tree diagram** (Tools → Gating tree diagram…).
  - **Embedding comparison** (Analyze → Compare embeddings…) — UMAP / t-SNE /
    PHATE side by side.
- **Research / stats**
  - **Group comparison** (Analyze → Group comparison…) — Kruskal-Wallis +
    pairwise Mann-Whitney (BH) + Cliff's δ across trial groups.
  - **Methods & provenance** (Analyze → Methods & provenance…) — a paper-ready
    methods paragraph (from the audit trail + citations) and a reproducibility
    run manifest.
  - **Export populations as FCS** (Tools → Export populations (FCS)…) — each
    gated population to its own FCS 3.1 file.
- **App**
  - **Preferences** dialog (Edit → Preferences…); **Documentation** and
    **Keyboard shortcuts** in Help.
  - **Plot pan/zoom** — middle-drag pans, scroll-wheel zooms (left-click stays
    gating); View → Reset plot view.
  - **Find box** above the sample/gate tree; **periodic autosave** (5 min).

## [1.2.4] - 2026-06-24

### Added
- **File → Load example dataset** — generates and loads a small synthetic
  PBMC dataset (2 groups × 2 donors), so OpenFlo can be tried with no FCS
  files of your own.
- **File → Save plot as image…** — export the current plot directly to
  PNG / SVG / PDF (white background, 300 dpi).

## [1.2.3] - 2026-06-24

### Added
- **Global error handling.** Unhandled UI errors now flag the status bar and
  auto-reveal the log/console (instead of failing silently). A **tokenised**
  error report (Help → Report a problem…) is written for submission: file
  paths, sample names, usernames and emails are replaced with stable tokens,
  and the token→value key is kept in a separate LOCAL file that is never meant
  to be submitted.
- **Keyboard shortcuts** with menu accelerators: Ctrl+O (open session),
  Ctrl+S (save), Ctrl+E (export .wsp), Ctrl+W (close), Ctrl+Shift+A (add FCS),
  F1 (About).
- **Window size/position** is remembered across launches (validated on-screen).
- **File → Open Recent** — the last sessions you opened or saved.

## [1.2.2] - 2026-06-24

### Added
- **One-step setup scripts** (`setup.bat` / `setup.sh`) that create the `.venv`
  and install OpenFlo + all dependencies; the `openflo-gui` launchers run them
  automatically on first launch if the environment is missing.

### Fixed
- Startup session-restore no longer hard-crashes when the data dependencies
  (FlowIO, etc.) aren't installed — it reports the missing dependency clearly
  and opens an empty session instead of failing the whole window.

## [1.2.1] - 2026-06-24

### Added
- **Light / Dark / Midnight themes** (View → Theme), persisted across
  sessions. Light and Dark keep the scatter/plot light (flow-cytometry norm);
  **Midnight** darkens the plot canvas too — figure, axes, ticks, labels,
  spines, grid, legend and the backgate legend.
- **App icon** — a flow-cytometry density-scatter mark replaces Tk's default
  feather in the title bar / taskbar.
- **Dropdown-menu help** in the status bar (per entry, as you navigate).
- **Resizable, pop-out panels.** Samples & Gates | Plot | Pipeline Workspace
  are draggable panes, and both side panels float into their own window and
  re-dock.
- **Hover tooltips** on the plot controls, gate tools, and action buttons,
  toggleable via View → Show hover tips.
- **Per-population density scaling** for overlays, with a clickable backgate
  legend (on/off · density · colour) that is draggable and collapsible.
- Cell-cycle results group under a collapsed container and persist across
  session save/restore.

### Changed
- **Mode** and **Downsample** are now dropdowns; mode-specific options
  (KDE / contour scatter & outliers / Hist-Y) appear only when relevant, and
  **Max points** is shown and applied only while downsampling is enabled.
- Gate-tree heading expands/collapses all groups; control bars regrouped so
  each section aligns to its column.

### Fixed
- Session results (clusters / UMAP) recover after a dropped processed-data
  sidecar pointer; backgating clustered populations works through the new
  collapsed group containers.
- Downsampling **Off** now truly draws every event (Max points no longer
  silently caps when downsampling is off).
- Refreshed the README/limitations (Auto-gate offers reviewable scored
  proposals; it is not disabled) and added a prominent citation request for
  research use (README banner + About dialog; MIT unchanged). Renamed
  LICENSE → LICENSE.txt so it opens with a double-click.

## [1.1.0] - 2026-06-23

### Added
- **Built-in template library picker (ease-of-use).** The editor's template
  button is now a **Templates ▾** menu that lists every bundled template by its
  friendly name (the `cleanup_*` recipes first) plus your own saved templates —
  apply one in a click, no file navigation. The curated library now ships
  *inside* the package (`openflo/template_library/`, package data) so it's
  available to installed users, not just source checkouts; user-saved templates
  still live in the editor's writable dir and shadow same-named shipped ones.
- **One-click cross-group comparison + volcano plot.** The Frequencies window
  gains a **Compare all…** button that compares *every* population across the
  current grouping in a single pass (instead of stepping through populations one
  at a time), Benjamini-Hochberg-correcting across populations. Results open in a
  new window with a sortable table (per-group means, log2 fold-change, adjusted
  p, stars) beside a **volcano plot** (log2FC vs −log10 adjusted-p, significant
  populations highlighted and labelled), with full-table CSV and figure export.
  New pure `openflo.stats.compare_all_features` (runs `compare_groups` over all
  features, BH across them) and `volcano_data`, both exported. The volcano needs
  the two-group case; with >2 groups the table still shows the omnibus
  Kruskal-Wallis / ANOVA result.
- **End-user self-test + seeded data generator (regression baseline for
  everyone).** Two new console entry points let users — not just contributors —
  reproduce and regression-check behavior on data they don't have to provide:
  `openflo-synth` writes the full seeded synthetic dataset (now including the
  `beads/` size-calibration file), and **`openflo-selftest`** runs that data
  through the core feature paths (auto-clean debris/viability/doublets, Leiden
  clustering, MESF calibration, compensation) and compares each metric to a
  committed golden baseline (`openflo/_golden.json`), printing a PASS/FAIL table
  — so after pulling an update or editing code you can instantly see whether any
  feature's behavior changed (`--update` refreshes the baseline after an
  intended change; `--json` dumps raw metrics). The same golden file backs the
  pytest continuity tests, so the CLI and CI share one source of truth. Tests
  ship in the sdist (`MANIFEST.in`). New `openflo.selftest`.
- **Bead-calibrated debris removal + dead-cell (viability) auto-cleaning.** The
  auto-clean gate's **Debris** method now defaults to an *absolute-size* cut:
  when a size-calibration bead sample is loaded (name contains bead / rainbow /
  calibration), its median FSC-A anchors a µm ruler and events below
  `min_um` (default 4 µm, bead diameter default 8 µm) are dropped
  (`FSC-A ≥ min_um · bead_FSC / bead_um`) — a reproducible absolute-size ruler
  that, with a sub-cell `min_um` (≈4 µm), removes only genuine sub-cellular
  fragments and keeps small real cells (lymphocytes). With no bead file it
  falls back to a **2-D FSC-A × SSC-A scatter gate** matching the standard
  manual debris polygon (debris = low FSC AND low SSC; granular low-FSC/high-SSC
  cells are rescued when they form a separate lobe) — never a 1-D cut that would
  bisect a real population. A new **Dead cells (viability dye)** method
  finds the live/dead stain by name (`find_viability_channel`: Live/Dead,
  Zombie, Ghost, FVS/FVD, 7-AAD, PI, DAPI, …) and drops the high-signal dead
  population at a genuine bimodal valley (`_bimodal_valley`; no-op on an
  all-live, unimodal sample). Right-clicking the Debris or Dead-cells method
  rows switches mode (Beads ↔ Auto valley), sets bead / min size, re-detects
  the bead reference, or pins the viability channel. (FSC-A stays linear and
  the dye is logicle-transformed in the editor, so both cuts are
  scale-correct.) `openflo.pipeline.find_viability_channel`. The synthetic
  dataset gains a `beads/` size-calibration file (single tight 8 µm population
  matched to a real instrument's FSC scale; `openflo.synthetic.size_bead_sample`
  / `make_size_beads`) so bead-mode debris is testable headlessly, plus locked
  continuity reference drops (seeded ≈7 % debris / 8 % dead / 5 % doublets) that
  flag any future change to the cleaning maths. A method that removes **nothing**
  now explains why on its tree row (`autoclean_method_diagnostic`: "no viability
  dye detected", "FSC-A is unimodal — no low-debris mode; load size beads",
  "the high-signal population is the majority — not treated as dead") instead of
  a silent 0-drop. Both the debris valley and the viability split use the strict
  bimodal-valley detector, so a **unimodal** channel is never bisected (a clean
  single population — beads, a comp control, a pre-gated sample — correctly
  yields 0 drops rather than a spurious half-cut).
- **Compliance / sign-off layer (tamper-evident, 21 CFR Part 11-style).** On
  top of the audit trail, the **History** window gains **Sign & export record…**
  and **Verify record…**. Signing builds an integrity *manifest* — SHA-256 of
  every loaded data file plus a hash of the audit trail and the software
  version — and attaches an **electronic signature** (signer, meaning, time)
  bound to that manifest's hash; it writes a signed JSON record + a Markdown
  copy. Verifying re-hashes everything and flags any signature whose content
  changed after signing (data edited, audit altered) — so the record is
  tamper-evident. New pure `openflo.compliance` (`build_manifest`,
  `sign_manifest`, `verify_record`, `record_to_markdown`), exported. (Scope:
  tamper-evidence + attributable sign-off, not access control — it complements,
  not replaces, a controlled-access environment.)
- **Fluorescence calibration to standardized units (MESF / ABC).** A
  **Calibration…** dialog detects the bead-population peaks in a channel
  (k-means on log intensity), takes each peak's assigned MESF/ABC value from
  the bead datasheet, fits `value = slope·MFI + intercept` (with R²), and
  applies it across all samples as a plottable `MESF:<marker>` column — the
  fluorescence sibling of the existing FSC→µm bead-size calibration. New pure
  `openflo.calibration` (`detect_bead_peaks`, `fit_mesf_calibration`,
  `apply_calibration`), exported. The synthetic dataset now includes a
  `calibration/` rainbow-bead FCS + `mesf_peaks.csv`.
- **t-SNE and PHATE embeddings.** The Cluster dialog's UMAP checkbox is now an
  **Embedding** picker — UMAP / t-SNE / TriMap / PaCMAP / PHATE / none. t-SNE
  ships in core deps (scikit-learn; perplexity auto-clamped, subsampled); PHATE
  (diffusion-based, great for continuous / trajectory structure) is an optional
  `embed` extra. New `FlowSample.run_tsne` / `run_phate` write `TSNE1/2` /
  `PHATE1/2`; the view switches to the chosen embedding's axes after clustering
  (only if it produced columns, so an uninstalled backend degrades gracefully).
- **Sample QC (EMD + MDS) and AnnData interop.** A **Sample QC…** window
  computes a pairwise **Earth-Mover's-distance** matrix between the enabled
  samples (mean over markers of the 1-D Wasserstein distance, pooled-SD scaled)
  and an **MDS** embedding — batch effects and outlier samples show up as
  separated points (coloured by trial). Exports the distance matrix, the
  figure, and an **AnnData `.h5ad`** (events × markers, with `sample` + any
  `leiden`/`cluster`/`flowsom_meta`/`pseudotime` columns in `obs`) for the
  scanpy / single-cell Python ecosystem. New pure `openflo.interop`
  (`sample_distance_matrix`, `mds_embed`, `to_anndata`, `write_h5ad`); AnnData
  is an optional `interop` extra (`pip install 'openflo[interop]'`).
- **FlowSOM star-tree visualization.** A **SOM tree…** button draws the iconic
  FlowSOM plot: the SOM nodes laid out on their **minimal spanning tree**, each
  rendered as a **star glyph** of its per-marker prototype profile, coloured by
  metacluster, with node size ∝ event count — plus a reference star (marker →
  spoke) and a metacluster legend. PNG/PDF/SVG export. New pure
  `openflo.pipeline.flowsom_mst` / `flowsom_layout` (scipy MST + igraph layout,
  exported).
- **Rigorous differential abundance (negative-binomial GLM).** A **Diff.
  abundance…** button in the Frequencies window runs a diffcyt-DA-edgeR-style
  test: each population's per-sample counts are modelled with a negative-
  binomial GLM, `log(library size)` as offset (so it accounts for sequencing-
  depth / composition), a shared method-of-moments dispersion (edgeR
  common-dispersion-style, stable with few samples), a Wald test on the group
  coefficient and BH correction — replacing Mann-Whitney-on-fractions for the
  abundance question. Results table (log2FC, per-group %, p, adjusted p, stars)
  with CSV export. New pure `openflo.diffexp.differential_abundance` (scipy
  only — no statsmodels), exported.
- **Automated population annotation (MEM + reference table).** An
  **Annotate…** window turns numeric clusters into biological labels.
  **MEM** (Marker Enrichment Modeling, Diggins 2017) computes a quantitative
  per-marker enrichment score for each cluster vs the rest (capturing both the
  median shift and the IQR change), yielding labels like `CD3+5 CD4+3 CD8-6`.
  A **reference cell-type table** (`CD4 T: CD3+ CD4+ CD8-`, ACDC/Scyan style)
  then assigns each cluster a best-matching name (weighting the defining
  positive markers so a shared negative can't win), written back onto the
  populations and the cluster-label store. Exports the MEM table. New pure
  `openflo.annotate` (`mem_scores`, `mem_label`, `population_states`,
  `parse_signature_table`, `annotate_by_reference`), exported.
- **Synthetic example dataset generator** (`openflo.synthetic` +
  `scripts/make_synthetic_dataset.py`). A generic, regenerable dataset — not
  tied to any one study — that between its sub-datasets exercises every feature:
  a **PBMC immunophenotyping** set (CD3/CD4/CD8/CD19/CD56/CD14 lineages, the
  marquee generic example) for gating / clustering / Leiden / UMAP / frequencies
  / expression / heatmap / report; a **3-batch variant** with a technical gain
  shift for CytoNorm batch correction; **FMO controls**; a **cell-cycle**
  (DNA-content G1/S/G2-M) set; **conventional-compensation** single-stain
  controls with a known spillover matrix + a sibling `compensation.csv`; the
  **differentiation** time-course for trajectory; and **spectral** controls for
  unmixing/QC. Pure (numpy/pandas; FlowIO to write FCS), tested, and gitignored
  output.
- **One-click analysis report (HTML).** An **Analysis report (HTML)…** button
  bundles the whole session into a single, portable, self-contained `.html`
  file (images embedded as base64 data URIs — no sidecar files): metadata
  header, sample & gate summary, the current plot, the population-statistics
  table, a cluster × marker median-expression heatmap (column z-scored, when a
  `leiden` / `cluster` / `flowsom_meta` column exists), and the full provenance
  / audit trail. Opens in the browser on save. New pure `openflo.report`
  (`build_html_report`, `df_to_html_table`, `figure_to_data_uri`), exported.
- **Leiden clustering.** The current field-standard for high-dimensional
  spectral cytometry, alongside the existing Phenograph and FlowSOM. The
  Cluster dialog gains a **Leiden** method with a **resolution** control
  (higher → more, finer clusters); it builds a shared-nearest-neighbour
  (Jaccard) graph — the Phenograph/Seurat construction, so communities track
  real populations — and partitions it with `leidenalg` (RBConfiguration).
  Writes a ``leiden`` column imported as populations; large samples are graph-
  partitioned on a subsample and the rest assigned by nearest neighbour.
  `FlowSample.run_leiden`. (`igraph` + `leidenalg` were already declared
  dependencies.)
- **Export gated population as FCS.** Right-click any gated population →
  *Export population as FCS…* writes that population's events to a standalone,
  re-importable `.fcs` (FlowJo / FCS Express). Exports the sample's **raw**
  detector values when they're row-aligned with the gated events (so the file
  isn't in transformed coordinates), else the processed data, and carries the
  antibody labels through as `$PnS`. New pure `openflo.pipeline.write_fcs`
  (FlowIO-backed, exported) zeroes non-finite cells and supports
  channel subset/reorder.
- **Marker-expression distributions (violin / ridgeline) by group.** An
  **Expression…** window pools each enabled sample's per-cell values for a
  chosen marker (resolved across fluors by antibody label), groups samples by a
  factor (trial/day, comp-vs-samples, or a name token), and draws a **violin**
  or **ridgeline** plot per group. Significance comes from a per-SAMPLE-median
  comparison (each sample a replicate, not each cell) — also what the GraphPad
  Prism Column export contains. New pure `openflo.stats.group_kde` (KDE per
  group over a shared grid) backs the ridgeline and is exported/tested.
- **Trajectory / pseudotime (GUI + backend).** A **Trajectory…** tool orders
  cells along a differentiation trajectory: a symmetric kNN graph over the
  enabled samples' shared fluor channels (concatenated, so a day-series becomes
  one continuous trajectory), with pseudotime = geodesic distance from a root
  cell chosen at the extreme of a marker (e.g. CD34-high progenitors as t=0).
  It writes a ``pseudotime`` column to every sample (selectable as a plot
  colour) and draws each marker's mean expression along pseudotime — the
  CD34-down / CD11b-up maturation curve — with CSV / Prism XY / figure export.
  Backend (`openflo.trajectory`: `compute_pseudotime`, `robust_root`,
  `pseudotime_trends`) is pure (numpy/scipy/sklearn), subsamples large data for
  the graph and propagates by nearest neighbour, and is exported.
- **Population frequencies & group comparison (GUI + backend) with GraphPad
  Prism export.** A new **Frequencies…** window collects each sample's
  per-population frequency, groups samples by a factor (trial/day, comp-vs-
  samples, or a name token like `Stim`/`Ctrl`), and for a chosen population +
  metric (%Parent / %Total / Count) draws a box+strip comparison with
  significance annotations plus an all-population overview. Statistics pick the
  right test automatically — Mann-Whitney U / Welch t for two groups,
  Kruskal-Wallis / one-way ANOVA + BH-adjusted pairwise post-hoc for more
  (`openflo.stats.compare_groups`). Exports: tidy CSV, **Prism Column** and
  **Prism Grouped** tables (columns = groups, rows = replicates; ragged groups
  padded — paste straight into GraphPad Prism), a stats summary, and the figure
  (White/Transparent/Translucent). Backend (`compare_groups`, `to_prism_column`,
  `to_prism_grouped`, `p_to_stars`) is pure and exported.
- **Spectral unmixing QC + CLI batch-unmix.** New diagnostics for how
  trustworthy an unmix is: a spectral **similarity matrix** (cosine between
  reference spectra — flags fluorophore pairs too collinear to resolve), the
  **condition number** of the spectra matrix, and the **Spillover Spread
  Matrix** (SSM, Nguyen 2013 / Cytek — the spreading error each single-stain
  injects into every other fluor). After an Unmix the GUI opens a **Spectral
  QC** window with similarity + SSM heatmaps, the flagged similar / high-spread
  pairs, and Markdown / PNG export; the condition number and similar-pair
  count are recorded in the audit trail. New CLI mode `--unmix` builds
  reference spectra from `--unmix-controls` (a fluor→FCS JSON map, optional
  `unstained`), unmixes `--unmix-input` FCS into per-fluor CSVs, and writes
  `spectral_qc.{md,json}` + `reference_spectra.png` — so unmixing is no longer
  GUI-only. Backend: `spectral_similarity_matrix`, `spectral_condition_number`,
  `spillover_spread_matrix`, `unmixing_qc` (pure numpy, exported).
- **Provenance / audit trail (GUI + backend).** A new append-only
  `AuditLog` records the meaningful operations of an analysis session in
  order — sample load (with path, event count, compensation source),
  transforms, cleaning, gate add/remove, auto-gate proposals (with their
  quality scores), clustering, batch normalization (with before/after QC
  distance), spectral unmixing, figure export and session reload. A
  **History…** button opens a live viewer that exports the trail to
  **Markdown** (a methods-section-ready table with an OpenFlo-version
  header), **CSV**, or **JSON**. The trail is embedded in the saved session
  and restored on load, so the record of *how* a result was produced travels
  with it. Pure/stdlib backend (`openflo.audit`), fully unit-tested.
- **Trustworthy automated gating (GUI + backend).** The **Auto-gate** button
  (previously disabled — its single-contour heuristic mis-placed gates) now
  opens a dialog offering three well-posed, reviewable methods, each reported
  with a quality score in the status bar:
  - *Singlet gate* — a robust FSC-A/FSC-H ratio band (median ± k·MAD) emitted
    as a polygon; reports the fraction kept and ratio CV (`auto_singlet_gate`).
  - *Find populations (GMM ellipses)* — fits a Gaussian mixture on the current
    X/Y plot, picks the component count by BIC, and emits one **ellipsoid gate**
    per population at a chi-square coverage radius, each tagged with its weight
    and a separation score so overlapping (untrustworthy) splits are flagged
    (`gmm_ellipse_gates`).
  - *1-D threshold* — the existing valley/Otsu split.
  Every proposal is added as an ordinary undoable gate to accept / tweak /
  delete — review, not auto-apply. `describe_gate` now names polygon/rect gates
  and renders ellipsoid gates (previously shown as `? ellipsoid`).
- **Multi-panel figure layout / export (GUI).** A **Figure…** button assembles
  the current plot into a publication-style small-multiples figure: one panel
  per sample (current channels), one panel per channel pair (samples overlaid),
  a samples × pairs grid, or a single panel. Channel pairs accept marker labels
  or channel names (e.g. `CD34/CD11b, CD11b/CD45`). Each panel reuses the live
  rendering pipeline (mode, density/colour, axis scales, gates) via an
  axes-swap (`_render_into`), so panels match the on-screen plot exactly. A
  preview window saves to PNG / PDF / SVG / TIFF at 300 dpi, with a
  **background** option — White (default), Transparent, or Translucent
  (50%) — for placing publication figures on a coloured page / poster.
- **Spectral unmixing workflow (GUI).** An **Unmix** button designates loaded
  single-stain controls (→ fluorophore) + an unstained control, builds the
  reference spectra (with an autofluorescence endmember) and unmixes every
  other loaded sample into per-fluor `U:` abundance channels (OLS, optional
  non-negative) that become plottable/gateable, plus a spectrum-signature
  plot. Wraps the `spectral.py` backend.
- **Batch correction (CytoNorm).** The flow-cytometry standard for removing
  technical batch/acquisition variation: FlowSOM-metacluster the pooled data,
  then per metacluster + channel quantile-normalize each batch onto a shared
  goal distribution. One engine, two modes — `goal` (CytoNorm 2.0, control-
  free, default) and `controls` (classic, fit on per-batch controls; CLI-only).
  The fitted model serializes and applies to new samples; a QC report gives
  per-channel Wasserstein before/after. GUI: a **Batch-norm** button (2.0,
  groups by trial/day). CLI: `--batch-correct` with `--cytonorm-mode` /
  `--cytonorm-control` / `--cytonorm-metaclusters`.
- **Backgating.** Right-click a gate/population → *Backgate (show on plot)*
  projects its events, coloured, on top of the current plot — so you can see
  where a downstream population/cluster sits on any axes. Multi-select gives
  several colours + a legend; *Clear backgating* removes them.
- **Auto-clean drop-count readout.** Each auto-clean gate row in the tree now
  shows how many events the recipe removes — `autocleaned sample — drops N
  (X%)` — with a per-method breakdown under it (each method's standalone
  contribution, shown even when toggled off so you can preview it). Computed on
  the full sample and cached by data identity + recipe signature.
- **Staining-panel `.xlsx` → channel labels (CLI).** `--panel <file>` (or
  `--panel auto`, which searches the trial folders and a few ancestor levels)
  reads a CD↔fluorophore sheet and maps each fluorophore to its detector
  channel, merged with `--labels`. New `read_staining_panel` / `find_panel_xlsx`.
- **Per-group marker-pair scatters (CLI).** Every group now emits, for each
  pair in `--pairs` (default CD34/CD11b, CD11b/CD45, CD34/CD45), an *overlay*
  (all samples on one axes, coloured by sample) and a *grid* (one density panel
  per sample, shared limits). New `save_group_pair_scatters`.
- **Adjustable plot point cap.** A **Max points** control (presets + free
  entry, `250k`/`All` accepted) replaces the fixed 60 k scatter cap; drives
  scatter / pseudocolor / contour, updates the tree's shown/total counts, and
  persists in the session.
- **“Show cleaned-out events” overlay.** A plot-control toggle that draws the
  events the auto-clean recipe removes *in red, on top* of whatever's plotted —
  computed on the full sample and **bypassing the display cap**, so a small
  error rate stays visible against the full population instead of being
  subsampled away. Scatter modes overlay red dots (with a count); histogram
  mode overlays the removed events' channel distribution scaled to be visible.
  Reflects the current recipe and persists in the session.

### Changed
- **Config-driven batch runner.** `scripts/run_analyses.py` reads a JSON
  config (default the git-ignored `private/analysis_config.json`; see
  `scripts/analysis_config.example.json`) describing analyses via reusable
  `group_by` strategies — no data paths baked into tracked code. `--dry-run`
  resolves + verifies groups without clustering. Keeps the tracked tree
  generic so real experiment paths stay in `private/` (git-ignored).
- **Smoother density rendering.** Pseudocolor samples each event's colour by
  **cubic** interpolation of the smoothed density field (C2-continuous, so no
  per-cell colour blocks *or* residual bin-grid box facets), with an adaptive
  smoothing floor, and colours via `PowerNorm` so large samples no longer wash
  out to one flat hue. Histograms render as kernel-smoothed filled curves
  instead of chunky step bars. Contour density is zero-padded so every level
  closes.
- **GUI caps BLAS threads at startup** (mirroring the CLI) so OpenBLAS can't
  exhaust memory and abort the console-less launch under pressure.
- **Auto-clean gate.** A new **Auto-clean** button adds an *“autocleaned
  sample”* recipe gate (a collapsible group of toggleable cleaning methods —
  debris, doublets, margin/saturation, flow-rate bubbles/clogs, signal drift).
  It stores the *calculation*, not coordinates: its mask is the AND of the
  enabled methods, recomputed from each sample's own data, so copying it to
  other samples re-runs the cleaning per sample rather than reusing one
  sample's geometry. Build downstream gates under it to gate on cleaned events.
  Not FlowJo-representable — WSP export drops it and re-roots any children.
- **Folder drag-and-drop import.** Dropping a folder recurses into its
  `.fcs`/`.wsp` files; dropping a parent of several trial folders imports each
  independently. A bounded background load queue (fixed worker pool) replaces
  one-thread-per-file so large folder drops can't exhaust memory, with a
  determinate progress bar showing *N/M loaded*.
- **Histogram Y-axis selector** — Fraction (default) / Count / % of Max. Raw
  Count honours the auto-downsample toggle (and bypasses the scatter-only 60k
  cap) so counts are truthful.
- **Event counts in the Samples & Gates tree.** Each sample row shows its
  event count, displayed as `shown/total` when auto-downsampling scales it to
  the smallest sample (and updating when the toggle changes).
- **Auto-clean parameter dialog.** Double-click an auto-clean gate (or method
  row), or right-click → *Edit auto-clean parameters…*, to tune each method's
  enabled flag and parameters (bin counts, MAD thresholds, doublet tolerance,
  an optional manual debris FSC cutoff).
- **Auto-clean masks are cached** per (sample data, recipe) and reused across
  replots — recomputed only when the data or recipe changes — so gating on
  cleaned events stays responsive on large samples. The mask is computed on the
  full sample data (a per-acquisition property), so filter and highlight views
  flag the same events even when a plotted axis is sparse (e.g. an embedding).

### Changed
- **Imported day groups split into Comps + Samples subgroups.** When a day
  group contains compensation controls (names matching comp / control /
  (un)stained), the tree shows a *Samples* sub-header (expanded) and a *Comps*
  sub-header (collapsed by default); each subgroup's ✓ toggles its members'
  display. Days without comps list samples directly as before.
- **Imported gates load disabled.** Gates brought in with no explicit enabled
  flag (e.g. from a `.wsp`) start unchecked, so a freshly-loaded sample isn't a
  wall of active toggles. A restored session's gates keep their saved state.
- **Drag samples between groups.** A sample row can now be dragged to another
  day, or between the Comps and Samples subgroups, to fix a mis-import or for
  convenience (a manual Comps/Samples choice overrides the name-based guess).
  Multi-selection is honoured, and the regrouping persists in saved sessions.
- **“Clear all” keeps auto-clean gates by default**, with a checkbox in the
  confirm dialog to also clear them — so a bulk gate wipe doesn't discard the
  cleaning foundation.
- **Folder grouping is now by collection “Day N”.** `derive_trial_name` scans
  ancestor folders for a `Day N` token (at whatever depth it sits) and groups
  by it, falling back to the grandparent folder when absent; day groups sort
  numerically. Samples whose filenames repeat across days are disambiguated
  (e.g. `… [Day 9]`) so identical names no longer silently overwrite one
  another.
- **“Clear all” now clears all gates but keeps the samples** (undoable),
  reversing the 1.0.0 behaviour where it removed every sample. **Clear** now
  acts on the selection: a gate (cascade), a sample's gates, or a whole
  trial's gates — never removing samples (use **Remove** for that).

## [1.0.0] — 2026-05-29

First public release.

### Fixed
- **UMAP/TriMap runs no longer flash a console window** on Windows — the
  per-unit worker subprocess (and the Cancel `taskkill`) launch with
  `CREATE_NO_WINDOW`.
- **"Clear all" now actually clears the panel.** It previously only emptied
  the active sample's gates; it now removes every loaded sample and all gates
  (confirmed, since sample removal isn't undoable).
- The **Auto-gate** button is greyed out for now — its density heuristic
  mis-placed gates often enough to be untrustworthy.

### Changed
- **Repo layout consolidated.** Loose root scripts moved into folders —
  `smoke_test.py` → `scripts/`, `HANDOFF.md` → `docs/`; the top level now
  keeps only standard docs, config, and launchers.
- **Statistics is strictly population-based.** The window accepts only gate /
  population rows — dragged from the Samples & Gates panel or from a *gated*
  Pipeline Workspace item — never whole samples or trials. The two Import
  buttons (**Import S&G gates** / **Import workspace**) REPLACE the current
  set; dragging a gate APPENDS. A **Source** column tags each row
  `editor` / `workspace` / `editor+workspace`.
- **Editor bottom-left buttons unified.** Clear / Clear all / Copy / Pops are
  now equal-width and compact (↶/↷ stay as small icon buttons), making room
  for the new log pane.
- Selecting a **trial** row and pressing **Delete** (or **Remove**, or
  right-click → *Remove trial*) now clears that trial's samples and gates
  (confirmed).
- **Pipeline default grouping is now by day, not a fixed two-group split.**
  With no `--groups`/`--samples`, OpenFlo discovers every folder that
  directly holds FCS files — point it at a single PARENT and each
  sub-folder becomes its own day/group, sampled independently and
  compared across days in one analysis. Folder names are tidied to
  `Day N` when a day token is present; duplicate day names are
  disambiguated by parent. Explicit `--groups` and the legacy
  `--samples` split still work; `DEFAULT_GROUPS` remains the final
  fallback.
- **Per-sample FMO assignment.** A group's `samples` entry may be a
  plain string (inherits the group's `fmo_set`) or
  `{'name', 'fmo_set'}` to point one sample at a different FMO control
  set. FMO thresholds + both run modes resolve per sample. Compensation
  and antibody labels were already per-sample-automatic (each FCS's
  `$SPILL` / `$PnS`).
- **The gate editor is now the entire GUI.** `openflo-gui` opens straight
  into the editor (it owns a hidden Tk root); closing it exits. Pipelines
  run from the editor's docked **Pipeline Workspace** — drag samples /
  gated populations in and Run. The separate pipeline-config window was
  removed (see *Removed*).
- **Pipeline Workspace runs Phenograph + UMAP + TriMap per RUN UNIT, each
  in its own subprocess.** A *unit* is a **group's samples co-embedded into
  one UMAP** (events tagged by source sample); a **Concatenate** toggle
  merges all groups into a single UMAP so groups compare in one embedding
  (FlowJo-style); loose items run on their own. Each run writes a
  cluster-frequency CSV, a **cluster × group/sample composition CSV**, and
  embedding PNGs coloured by cluster *and* by source. Embeddings use the
  proper per-marker channels (height/width detector duplicates dropped) on
  an up-front subsample. A native crash / hang / OOM is isolated to that
  child (the GUI survives); **Cancel** terminates the job's whole process
  tree; a crashed/OOM unit is requeued once at a lower event cap, then
  skipped. The editor's Undo button also reverts workspace edits;
  workspaces save/load to JSON; a Results viewer shows the outputs.

### Removed
- **The legacy pipeline run-plan / staging window (the `App` class) and
  its in-process + subprocess run engine.** It was discontinued — most of
  its features were unreliable (crashes / restart loops). ~3,900 lines
  removed; its role is taken by the Pipeline Workspace. Also dropped the
  now-unused Windows Job-object / memory-watchdog / GPU-probe
  infrastructure and `tests/test_run_plan.py`.

### Added
- **Collapsible in-app log pane.** A "Show log" toggle at the bottom of the
  editor's left column reveals a small terminal that mirrors stdout/stderr
  (diagnostics, tracebacks) without needing a console; "Clear log" empties it.
- **Pipeline Workspace item drag.** Drag an item between groups — or onto
  empty space to pop it back to the top level — to fix a mis-drop. Dragging a
  *gated* item onto an open Statistics window adds its population.
- **In-editor clustering + the full cluster→name→use loop.** A "Cluster…"
  button runs Phenograph or FlowSOM (+ optional UMAP) on loaded samples in
  a worker thread, then auto-imports the result as populations and switches
  the plot to the UMAP coloured by the label. Population import is now
  generic — the "Populations…" menu detects any present label column
  (`cluster`, `flowsom_meta`, cell-cycle phases) and offers import + rename
  for each. Clustered/UMAP'd data can also be brought in from outside via
  **"Load CSV…"** (`FlowSample.from_dataframe` ingests a pipeline
  `*_processed.csv`, preserving cluster/UMAP/flowsom columns); derived
  columns are auto-excluded from marker lists.
- **Spectral unmixing.** New `openflo.spectral`: `build_reference_spectra`
  turns single-stain (+ unstained autofluorescence) controls into a
  reference spectra matrix; `unmix` solves per-event fluorophore abundances
  by least squares (OLS, optional non-negativity); `apply_unmixing` adds
  one abundance column per fluor to a sample. For full-spectrum cytometers
  (Cytek Aurora, BD S8) where compensation alone doesn't apply.
- **Differential abundance / expression.** New `openflo.diffexp`:
  `differential_test` (Mann-Whitney U + log2 fold-change + Benjamini-
  Hochberg FDR) over per-sample feature values, with `cluster_abundance`
  and `marker_expression` builders that turn two groups of samples into the
  per-sample feature dicts. The diffcyt/OMIQ-style comparison OpenFlo
  lacked.
- **FlowSOM clustering + metaclustering.** `FlowSample.run_flowsom()` trains
  a self-organizing map over the marker space, assigns each event to a node,
  and agglomerates nodes into metaclusters — writing `flowsom` (node) and
  `flowsom_meta` (metacluster) columns. Compact, dependency-free
  (numpy + sklearn), fast on large files.
- **More transforms + per-channel transform editor.** `transform_values` /
  `inverse_transform_values` add **arcsinh** and **hyperlog** (and a linear
  pass-through) alongside logicle/log, with FlowJo's t/m/w/a knobs (arcsinh
  uses an intuitive `cofactor`). A "Transforms…" editor in the GUI re-maps
  each channel's transform across all loaded samples by inverting the
  current one and applying the new — no re-compensation needed.
- **Boolean gates (AND / OR / NOT).** New `boolean` gate kind combining
  other gates' cumulative masks (cycle-guarded). Build one from the gate
  tree's right-click menu ("Create boolean gate…"); it toggles, highlights,
  filters, and feeds the stats table like any population. Dropped from
  `.wsp` export with a lossy-export note.
- **Automated density-based gating (auto-gate).** `auto_threshold` (valley
  between the two density modes, else Otsu) and `auto_polygon_gate` (a
  contour around the dominant 2-D density mode). An "Auto-gate" button
  proposes a threshold (histogram) or polygon (2-D) for the active sample
  to accept or tweak.
- **Undo / redo in the gate editor.** Ctrl+Z / Ctrl+Y (and ↶/↷ buttons)
  over a snapshot history of the gate state. Every structural change —
  add, delete, drag, reparent, paste, cluster/cell-cycle import,
  annotate — is one undoable step (mutations in a single gesture coalesce);
  bulk session/template loads don't pollute the history.
- **Cell-cycle recognition (DNA content).** `FlowSample.cell_cycle()`
  auto-detects a DNA-stain channel (PI / DAPI / FxCycle / 7-AAD / Hoechst /
  DRAQ5 / …; `find_dna_channel`), optionally pre-gates singlets on the
  DNA-A vs `-W`/`-H` ratio (doublet exclusion), then models the histogram
  (`analyze_dna`): locates the G1 peak and the G2/M peak at ~2× DNA,
  estimates each peak's robust spread, and assigns every event a phase
  (G1 / S / G2M / sub-G1 / >G2M) → %G1/%S/%G2M. Writes a categorical
  `cell_cycle` column. In the editor, a "Cell cycle…" button runs it on
  the active (or all) sample(s), surfaces each phase as a selectable
  population (new `category` gate kind), and shows a DNA histogram +
  phase-percentage window.
- **Acquisition QC now detects clogs, bubbles, and saturation.**
  `AcquisitionQC` gained two detectors beyond the existing signal-drift
  one: **flow-rate anomalies** (time bins whose event count is a MAD
  outlier, plus empty interior bins — clog collapses and bubble gaps/
  bursts) and **margin/saturation events** (per-event removal of pile-ups
  at a channel's ceiling). All three combine into one clean-event index;
  `qc.report` breaks down removals by category. A clean acquisition trips
  none of them.
- **TriMap and PaCMAP dimensionality reduction.** `FlowSample.run_trimap()`
  and `run_pacmap()` mirror `run_umap` (shared `_embedding_input` /
  `_store_embedding` helpers), writing `TRIMAP1/2` and `PACMAP1/2`. Both
  are optional (`pip install openflo[embed]`) and degrade gracefully when
  not installed. They preserve global structure better than UMAP on some
  panels.
- **Voltage titration / Stain Index tool.** New `openflo.voltage` module +
  `openflo-voltage` CLI: point it at a titration series (one FCS per PMT
  voltage) and a channel, and it reads `$PnV` per detector, auto-splits the
  negative/positive populations (2-component GMM), computes per-voltage
  Stain Index = (med⁺−med⁻)/(2·rSD⁻) and the robust CV of the negative,
  and recommends the lowest voltage on the SI plateau. Generalized — any
  channel, any file set; pure metric layer is independently importable as
  `VoltageTitration`.
- **Plot axes resolve by antibody label per sample.** When overlaying
  samples whose marker sits on different fluorophores, picking an axis
  (a detector from the global panel) now resolves to *each* sample's own
  detector by antibody label (`_axis_alias_for_sample`), so the samples
  overlay on a common label axis instead of being dropped. The chosen
  name is aliased onto the sample's own column (the original detector
  column stays, so per-sample gate masks are unaffected); a sample that
  lacks the marker entirely is simply skipped. Completes the label-first
  follow-up to gate-by-label retargeting.
- **Clusters as selectable, annotatable populations in the editor.** A
  "Clusters…" button imports each clustering label (the pipeline's
  `cluster` column) as a root population — a new `cluster` gate kind
  whose mask is `cluster == id` (`gate_to_mask`; a missing column selects
  nothing rather than no-op all-True). Imported populations toggle,
  highlight, filter, and feed the statistics table like any gate. An
  "Annotate clusters…" dialog names them with phenotypes, persisted in
  the session's `cluster_labels` slot and shown as the population name.
  Cluster populations have no FlowJo geometry, so the `.wsp` export
  drops them with a clear lossy-export warning.
- **Gate templates retarget by antibody label.** Saving a template now
  stamps each gate's channel with its antibody label; applying that
  template to a sample where the marker sits on a *different* detector
  retargets the gate to that sample's detector (`relabel_gate_for_sample`).
  So a CD11b gate applies wherever CD11b lives in each sample, across
  panels — compensation is unaffected (only which column the gate
  reads changes).
- **Per-sample FMO override in the config GUI.** A group's samples
  field accepts `name:FMOset` to point one sample at a different FMO
  control set (e.g. `m1, m2:Late, m3`); `_get_groups` emits the
  per-sample dict form the pipeline resolves. A hint documents the
  syntax.
- **Cross-sample label-first tying + common-fluor warning.** The same
  antibody can sit on a different fluorophore across samples/days, so
  cross-sample analysis now aligns by antibody **label**, not detector
  (compensation stays keyed on detectors — each sample compensates its
  own `$SPILL`). New `openflo.pipeline` utilities: `align_fluor_labels`
  (common labels + per-sample label→detector + missing map),
  `common_fluor_warning`, and `concatenate_by_label` (merge samples on
  the common label set, renaming each sample's fluors to labels). The
  statistics table now names per-channel columns by **each sample's own**
  label, so a marker on different fluors merges into one column
  (`Median CD11b`) across samples. The editor flags a non-common fluor
  panel on sample load and in the Statistics window; non-common labels
  are simply blank where absent.
- **Population statistics table (FlowJo-style).** A "Statistics…" window
  in the editor tabulates, per sample × population (gate node, evaluated
  as the cumulative gate chain): Count, %Parent, %Total, and per-channel
  Median / Mean / CV. Columns are modular (checkbox toggles); the table
  exports to analysis-ready CSV. Populations show a FlowJo-style path
  (`Cells/Singlets/CD11b+`). Computed on full sample data, not the plot
  downsample.
- **Full FlowJo gate parity — ellipsoid + quadrant.** `WspReader`
  parses `EllipsoidGate` (mean + covariance + distanceSquare) and
  `QuadrantGate` (two dividers → 4 linked rects); `WspWriter` emits
  both (collapsing a `quad_set` rect group back into one QuadrantGate);
  `gate_to_mask` evaluates ellipsoids via squared Mahalanobis distance.
  Round-trip is self-consistent (our writer ↔ reader); FlowJo v10's
  exact serialization still needs validation against a real file.
- **Editor: ellipsoid rendering + interactive Ellipse tool.** Ellipses
  render (rotated too, via covariance eigendecomposition); a new
  Ellipse tool draws them, and the Edit tool moves / resizes (drag rim)
  / rotates (drag grip) them.
- **`.flowsession` save/load.** Captures the full editor state —
  samples (by path + colour + plot-enabled), per-sample gates at full
  fidelity (incl. ellipsoid / quadrant / colour / enabled), per-channel
  scale + range, plot mode, channel labels, downsample toggles, and a
  reserved `cluster_labels` slot. Autosaves to
  `~/.openflo/last_session.flowsession` on editor close and offers to
  resume it on next open. Save/Load Session buttons in the editor.
- **Batch template application.** "Load Template…" now pops a dialog to
  choose which loaded samples to apply to (multiselect + select all /
  none) and whether to **overwrite** each target's gates or **add to**
  them. Previously a template loaded into the active sample only. Gates
  referencing channels a target sample lacks are reported in a
  post-apply warning (they install but sit inert).
- **Lossy-export warning.** Exporting to `.wsp` now checks for
  OpenFlo-only state the FlowJo schema can't hold (custom per-channel
  axis scales / ranges, disabled gates, cluster labels) and warns
  before writing, offering to save a full `.flowsession` instead.
  Gates + compensation always survive, so a plain gating export
  doesn't nag.
- **End-to-end CLI tests** (`tests/test_cli_e2e.py`). Two tiers:
  fast `--help`-based wiring checks (always run — catch console-
  script breakage + flag-parsing regressions); and a full-pipeline
  subprocess run against the synthetic FCS, opt-in via
  `OPENFLO_RUN_SLOW_TESTS=1` (it runs Phenograph + UMAP, ~35 s warm
  but timing-sensitive under load, so it's gated like the real-data
  fixtures rather than making the default suite flaky).
- **WSP per-sample extract tests** (`tests/test_wsp_writer.py`) —
  exercise the `extract_gates(sample_node=...)` kwarg added
  during the gate-editor WSP-ingest work. Multi-sample synthetic
  workspace, per-sample subsetting, parent_id chain preservation,
  default-walk regression.
- **OSS infrastructure** — `.github/ISSUE_TEMPLATE/` (bug + feature
  + config routing questions to Discussions), `PULL_REQUEST_TEMPLATE`
  with a "Scientific impact" section, `.pre-commit-config.yaml`
  (trailing-whitespace, EOF, large-files, ruff check + format),
  `environment.yml` (conda mirror with optional RAPIDS).
- **`docs/algorithms.md`** — ~250 lines covering compensation
  sources + optimizer heuristics, logicle T/M/W/A defaults with
  FlowJo parity notes, FMO threshold rationale, Phenograph k
  rule-of-thumb table, subsample + KD-tree-assign trick, GPU
  determinism caveats, UMAP defaults, what the pipeline is NOT
  good at. Cites Parks 2006, Levine 2015, McInnes 2018, Roederer
  2011. README links via a new `## Algorithms` section.
- **README "Common workflows" section** — three concrete examples
  (single-sample GUI exploration, multi-trial batch run with
  `--groups` + `--fmo-sets` + `--export-wsp`, `openflo-compare`
  against a FlowJo workspace).
- **Vulture dead-code config** in `pyproject.toml` `[tool.vulture]`
  with documented false-positive exclusions for PEP-562 hooks,
  public API surface, and ctypes Structure fields.

### Changed
- **`pipeline.py` lazy-imports `matplotlib.pyplot`.** Moved from
  module-top to local imports inside the 5 plot methods. Saves
  ~300 ms on `import openflo.pipeline` (1050 ms → 750 ms) —
  matters for the gate editor / compare tool / any WSP-only
  caller. PEP-562 hook now exposes `pipeline.plt` for external
  callers that still want the bare attribute.
- **Gate editor write paths surface failures visibly.**
  `_save_template`, `_export_flowjo_wsp`, and `_apply_save_gates`
  now `messagebox.showerror` on failure in addition to the
  status-bar message. Silent data loss after a Save dialog is
  worse than the alert pop-up.
- **Removed the `_LazyFlowio` proxy** from `gui.py`. Replaced
  with a function-local `import flowio` at the single call site
  in `_inspect_channels_for_labels`. Same lazy effect; 16 fewer
  lines; no `# type: ignore[assignment]` workaround.

## [0.2.0] — 2026-05-27

### Added
- **Gate editor: Edit tool** with modifier-key gestures — left-drag to
  move a vertex/line, shift+drag to translate the whole gate,
  right-click on a polygon vertex to delete (refuses below 3 verts),
  right-click on an edge to insert, alt+left-click anywhere to drop
  a vertex into the polygon under the cursor. Per-tool gesture hint
  shown below the tool selector.
- **Per-channel axis scale + range.** ⚙ buttons next to the X/Y combos
  open a dialog: Linear / Symlog / Log scale, plus optional custom
  (min, max) range. State is keyed by channel name so swapping the
  X combo to a different channel picks up that channel's saved
  preference. Symlog `linthresh` is data-driven (5th percentile of
  |nonzero|, floor 1e-6).
- **Auto-downsample toggles.** "Auto-downsample display to smallest
  sample" (default ON) caps every plotted sample at the smallest
  loaded sample's size for honest overlay comparisons; underlying
  data is untouched. "…and propagate to data" (default OFF) actually
  trims `FlowSample.data` so clustering / stats see the trimmed set.
  Seeded so the same subsample renders across replots.
- **WSP ingest in the gate editor's Add-FCS button.** Picking a `.wsp`
  walks each `<Sample>`, resolves its `<DataSet uri="...">` to a
  local FCS path (tries as-is, then the WSP's own directory, then
  the editor's `fcs_dir`), queues the FCS for load, and stages the
  sample's gate subtree to attach as the FCS finishes parsing.
- `WspReader.extract_gates(*, sample_node=...)` — opt-in per-sample
  walk that reuses the existing parsers. Default behaviour
  unchanged.
- **Log-spaced histogram bins** when a channel's axis scale is `log`
  (linear / symlog continue to use linear-spaced bins). New
  `_hist_bin_edges` helper clamps non-positive lower bounds to a
  small positive floor and falls back to linear when the clamped
  range degenerates.
- Comprehensive unit tests for the new gate-editor helpers
  (`tests/test_gate_editor_helpers.py`, 47 tests) — covers
  `_gid_from_hit`, polygon vertex add/delete/find, downsample floor,
  axis scale apply path, and log-spaced bin edges.

### Changed
- Compensation matrix actually round-trips through the workspace
  export. Both `gui._export_flowjo_wsp` and
  `cli._export_pipeline_workspace` now call `WspWriter.set_compensation`;
  `FlowSample._apply_comp` persists the matrix on
  `self.comp_matrix` / `self.comp_channels` so callers can read it
  back. Two new regression tests in `tests/test_wsp_writer.py`.
- `OptimizeCompensationDialog._autofill_from_dir` — the auto-detect
  for single-stain control files now uses an ordered candidate
  tokenizer (joined form first, then each dash-separated part
  longest-first) instead of the naïve "first dash-separated token"
  heuristic. `PE-Cy7-A` now produces tokens `['pecy7', 'cy7']`
  instead of just `['pe']`. Ambiguous and unmatched channels surface
  in the status bar.

### Fixed
- **Sample-name collision across groups.** Per-sample tasks were keyed
  by bare sample name in the dispatcher, so two groups (e.g. two day
  folders) containing identically-named FCS (`sample_1.fcs`)
  silently bucketed both results into one group and dropped the other.
  Tasks are now keyed by group+name. Surfaced + guarded by the by-day
  e2e test.
- **Histogram blank rendering on wide-range fluor data.** Non-finite
  values (NaN / ±inf) silently made matplotlib's hist skip entries;
  auto-ranging across samples with vastly different scales (one
  logicle ~0–1, one raw 0–262144) collapsed the narrow-range sample
  into a single bin at zero. Now filters non-finite up-front and
  pins all samples to a shared bin grid built from the union of
  robust per-sample percentile ranges.

## [0.1.0] — 2026-05-27
Baseline version captured for the first OSS-ready release. See git log for
the full pre-OSS feature set (compensation editor, WSP round-trip, GUI
gate editor, comparison tool, GPU clustering, seeded reproducibility).
