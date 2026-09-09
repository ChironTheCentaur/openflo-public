# Changelog

All notable changes to OpenFlo are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.4.3] - 2026-09-08

### Fixed
- **`openflo-selftest` crashed instead of reporting on a legacy Windows
  console.** Its results table prints check marks, which cp1252 cannot encode,
  so the command whose whole purpose is to say whether an install reproduces
  reference behaviour raised `UnicodeEncodeError` and reported nothing. The
  same fault was found in `openflo-doctor --help` **despite** that module
  having a guard — the guard ran after `parse_args`, and argparse prints help
  and exits from inside it — and in `openflo-compare`, `openflo-voltage` and
  `openflo-synth`, which had no guard at all. Every entry point now calls a
  shared `force_utf8_streams()` before building its parser.
- **The declared `psutil` floor could not be installed on Linux.** psutil
  5.9.0-5.9.3 publish Linux wheels only up to cp310, so Python 3.11+ had to
  build it from source, which needs a C compiler. The floor is now 5.9.4, the
  first release shipping a `cp36-abi3` wheel that covers every supported
  Python.
- **The declared `matplotlib` floor could not be resolved at all.** matplotlib
  3.8.0-3.8.3 declare `numpy<2`, which cannot be satisfied alongside this
  project's `numpy>=2.0`. The floor is now 3.8.4, the release that dropped the
  cap.

### Changed
- **The eight shared core dependencies are declared as ranges, not exact
  pins** (numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, openpyxl,
  psutil), so `pip install openflo` can coexist with an environment that
  already has a scientific stack rather than downgrading it. Each floor was
  found by testing, and the evidence is recorded per-dependency in
  `docs/PYPI_READINESS.md`. `requirements.txt` still pins the exact tested
  stack — that is what CI and `openflo-doctor` check against, and it is
  unchanged. The six leaf dependencies nobody else competes for (FlowIO,
  FlowUtils, PhenoGraph, umap-learn, igraph, leidenalg) remain exact pins,
  because their versions move OpenFlo's numbers.

### Added
- **CI installs the built wheel at both edges of every declared range** — a new
  `install` job across {ubuntu, windows} x {floor dependencies on Python 3.11,
  newest resolvable on 3.12}. It builds the wheel, installs it the way a user
  does (no editable install, no `--no-deps`), and then requires the installed
  artifact to reproduce the golden baseline and pass the response controls.
  This closes two gaps: an editable install imports from `src/`, so missing
  package data was invisible, and nothing previously installed the floor of any
  declared range. The floor pins are generated from `pyproject.toml` by
  `scripts/lowest_direct_requirements.py`, so they cannot drift from what is
  declared. All three bugs fixed above were found by this job.

## [2.4.2] - 2026-09-08

### Fixed
- **Starting a run no longer stalls on labelling events.** `prepare_unit` tags
  every event with its source group and sample — one short string repeated
  across a whole member. Assigning those as object-dtype columns materialised
  millions of Python string references and was, on a 3M-row unit, **415 ms of
  the function's 531 ms** (the concat itself was 80 ms), then paid again when
  the frame is pickled for the child process. As categoricals: 604 → 111 ms of
  UI-blocking work, 730 → 414 MB in memory, and the job file 3044 → 803 ms to
  write. The tag values are unchanged.
- **Downsample-propagate gave a different answer depending on which sample
  finished loading first.** The option trims every loaded sample to the smallest
  enabled one — but each arriving sample was compared against the floor of
  whatever had loaded *so far*, and that order comes from a thread pool.
  Measured on four samples of 50k/40k/30k/20k events: loading them
  largest-first trimmed **nothing at all**, smallest-first trimmed **everything
  to 20k**, and an interleaved order gave a mix. The same session resumed twice
  could hold different event counts, and every frequency computed from them
  would differ. The trim now happens once over the whole set, sharing its
  definition with the toggle that documents the behaviour.
- **A log histogram threw away events it could have shown.** The log floor was
  `max(lo, hi * 1e-6)`, which raised a perfectly good *positive* lower bound to
  six decades below the top of the range — on an all-positive channel spanning
  1e-3 to 1e6 that put **half the events off the bottom of the plot**, silently.
  The clamp is only needed when there is no positive lower bound to use. What a
  log axis genuinely cannot show — non-positive values, routinely a third of a
  compensated channel — is now stated on the figure itself, so an exported panel
  carries it too, and the note stays silent when nothing is hidden.

### Changed
- **Starting a pipeline run no longer freezes the window while it stages.**
  `_launch_next` built the run unit AND serialised it to `job.pkl` on the Tk
  thread — the code even called it "cheap (main)". The write dominates:
  measured here, 569 ms for a 20-sample x 200k-event run (336 MB) and 1264 ms
  for 8 x 1M (672 MB), all of it blocking the UI. The write now happens on a
  worker and the child process launches when it completes.
  **Only the write moved.** `prepare_unit` deliberately stays on the Tk thread:
  it reads the editor's loaded samples, which the user can mutate, so running
  it concurrently would introduce the same read-during-write race that was
  fixed in the plot path. Its share is the smaller one (concat + tagging, 250 of
  the 1720 ms at 8 x 1M) and removing it needs a snapshot design, not a thread.

## [2.4.1] - 2026-09-08

### Upgrading — results that legitimately change

This release corrects analyses, so re-running work done on an earlier version
can give different numbers. Nothing needs migrating: the session format is
unchanged and old sessions open normally. What follows is where to expect a
difference, and in which direction.

- **Event counts go UP where auto-clean was removing events for no reason.**
  Four separate causes, each measured on a sample with no actual fault: an
  integer-valued channel (`$DATATYPE I`) lost **38%** to reported drift; one
  dead or unused detector removed **100%**; a sample QC'd after clustering or
  FMO positivity lost **12.4%** (a cluster label) to **49.8%** (a positivity
  flag); and a coarsely quantised viability channel cost **46%** of an all-live
  sample. Those events are now kept, so any frequency computed from them
  shifts.
- **Gates on a LOG-scaled channel select FEWER events.** A non-positive value
  has no logarithm and is now `NaN` rather than `0.0`. Previously such events
  were placed at raw intensity 1.0 — above every genuinely positive event
  dimmer than 1.0 — so a negative-side gate was collecting them. They now drop
  out of gates, medians and plots. If a marker-positive frequency falls, this
  is the likely reason, and the old number was the wrong one.
- **Condition-comparison group means go DOWN for patchy populations.** A
  cluster absent from a sample now contributes 0% instead of being left out of
  the average. A population found in one of three samples at 10% was reported
  as 10%; it is now 3.3%. The table also gains an `n_samples` column, and
  `sd_pct` is now a real number where it used to be `NaN`.
- **MEM scores go UP when a very small cluster was present.** A cluster with no
  measurable spread was taking the top of the scale and compressing every other
  population; measured, three real populations recovered from 4, -5, -5 back to
  8, -10, -9. Marker labels can change as a result.
- **Doublets are removed from transformed scatter.** If FSC-A/FSC-H had been
  arcsinh- or logicle-transformed before auto-clean ran, the doublet filter was
  a no-op and kept them all.
- **A "NOT" gate no longer contains events it could not measure.** The
  complement population shrinks, and a marker's positive and negative
  populations no longer sum to the whole sample when some events have no
  reading — which is the honest arithmetic.

Sessions saved by an earlier version still load, including their processed-data
sidecars: sample names that had to be sanitised now get a different sidecar
filename, and the loader looks for both.

### Changed
- **Five more, in file formats and the auto-clean filters.** A FlowJo
  workspace import *discarded every population name* — the name lives on the
  enclosing `<Population>` element, not the `<Gate>` inside it — so a
  40-population workspace arrived fully unnamed while its geometry survived
  intact, which is why a round trip looked like it worked. A compensation
  matrix whose *channel name contained a comma* (a `$PnS` description like
  "CD4 PE-Cy7, clone SK3" is ordinary) wrote a file this same module then
  refused to read. Standards-conformant Gating-ML `-INF` bounds reached the
  session file as `-Infinity`, which *RFC 8259 forbids* — valid to Python,
  rejected by `JSON.parse` and `jq`. An *unusable reference spectrum* (a
  single stain dimmer than the unstained control clips to all-zero) was
  reported as cosine similarity 0.0 against every other fluor, i.e.
  "maximally distinct, trivially unmixable", so the panel was declared free of
  problematic pairs *because* one fluor had no spectrum. And the *auto-clean
  doublet window was a no-op on transformed scatter*: it is a fixed fraction
  of the median ratio, but that ratio's spread collapses from 14.5% to 1.7%
  after an arcsinh bake, so all 4,000 planted doublets were retained where
  linear scatter lost all 4,000.
- **Seven more places where the wrong events were selected, deleted or
  counted.** Every one was reproduced and measured before it was touched.
  *Acquisition QC read our own analysis output as a detector*: the denylist had
  fallen behind the columns the app writes, so a `leiden` column made the margin
  filter delete the whole top-numbered cluster (12.4% of a clean sample) and an
  FMO `<channel>_pos` flag deleted every marker-positive event (49.8%).
  *Two samples could be handed the same session sidecar* — `Tube 01 Rep
  A` and `Tube_01_Rep_A` sanitised to one filename — and because
  restore prefers the sidecar over the raw FCS, one sample silently came back
  holding the other's events, clusters and compensated values.
  *A cluster absent from a sample was treated as unmeasured rather than 0%*, so
  a population found in 1 of 3 samples at 10% was published as the group's 10%
  instead of 3.3%.
  *A boolean gate that could not resolve its operands admitted everything*,
  turning a "NOT X" population into the entire sample (20,000 events where
  10,004 was correct); it now fails closed, and `apply_region_gates` passes the
  gate list so operands actually resolve.
  *A NOT gate adopted every event it could not measure* — 2,000 events with no
  reading made up 20% of a CD3-negative population, while the two gates still
  summed to the sample total so the result looked self-consistent.
  *The `log` transform folded every non-positive value onto 0.0*, i.e. raw
  intensity 1.0, ranking the dimmest events above every genuine positive below
  1.0 and mis-assigning 32,620 of 100,000 events; its inverse separately
  destroyed every value at or below raw 1.0 on a round trip, which runs
  whenever a channel's scale is changed.
  *A comb was mistaken for a bimodal distribution*: an integer channel spread
  over 256 bins leaves most of them structurally empty, and a zero bin between
  two comb teeth satisfies the "is this really bimodal" test, so the auto-clean
  viability filter deleted 46% of an all-live sample — whether it fired at all
  decided by how the bins happened to line up.
- **Batch-correction QC reports what it cannot measure (hardening, not a live
  bug).** `CytoNorm.qc` returned `0.0` — the *best* score on a lower-is-better
  scale — for a channel with no usable events, and raised `Distribution can't
  be empty` when no event was finite in every channel. Both are now NaN with a
  warning. To be precise about the scope: neither was reachable from the
  shipped call paths, because both callers pass `qc()` exactly the events they
  just fitted and `fit()` rejects that input first. Measured distances for real
  data are unchanged.
- **A cluster with no spread no longer sets the scale for the whole MEM
  table.** MEM scores a population as `|median shift| + IQR_ref/IQR_pop - 1`
  and then rescales so the largest |score| maps to 10. A population with no
  measurable spread — a single event, or every event identical — has an IQR of
  exactly 0, which the epsilon guard turned into "perfectly tight", the most
  that term can reward. Since the table is divided by its own largest value,
  one such cluster compressed every real call in it: a single stray event given
  its own label pushed three well-separated populations from 10, -10, -7 down
  to 4, -5, -5. The spread term now drops out when there is no spread to
  measure, and such a population is scored on its median shift alone.
  Deliberately unchanged: a population sitting exactly at the median of a
  multimodal reference has a genuinely unstable sign under "vs all other
  cells", which is the metric's own ambiguity rather than a defect.
- **Three more places where a degenerate input was answered with a
  confident number.** The FMO
  preview substituted `-999` for a missing per-axis threshold and then drew
  four quadrant percentages from it, two of them defined by a boundary nobody
  measured — and on a compensated channel whose dim population sits near
  -2000, that sentinel fell *inside* the data and split it 67.9 / 7.0 / 23.2 /
  2.0. And a FlowJo workspace declaring spillover for only some detectors was
  accepted with the rest silently left uncompensated, because the acceptance
  check counted coefficients rather than channels. All three now report what
  they do not know, and the uncovered detectors are named.
- **Acquisition QC no longer punishes integer-valued channels, dead detectors,
  or quiet time bins.** All three came from one line: the robust outlier band
  was `median(|v - med|) + 1e-10`, and that epsilon silently redefined "no
  measurable spread" as "infinitely sensitive". On an integer channel
  (`$DATATYPE I`, very common) more than half the per-bin medians tie with the
  overall median, so the MAD is exactly 0 — and a clean, drift-free sample lost
  **38% of its events** to reported "drift", while the identical data stored as
  float lost none. The scale floor now comes from each caller's own data (a
  channel's quantisation step; sqrt(N) counting noise for an event count)
  rather than from the deviations being judged, which also stops ordinary
  Poisson scatter in low-count bins reading as a flow-rate fault. Where no
  scale exists in either place, nothing is called an outlier.
  Alongside it, the margin/saturation filter skipped its own degenerate case:
  a channel that is constant across the file has no ceiling to pile up
  against, but every event counted as "at the max" — and because margins are
  OR-ed across the panel, **one unused or disabled detector deleted 100% of an
  otherwise healthy sample**. Integer and float storage of the same data now
  produce identical QC verdicts in both directions, and real faults — a 400-unit
  clog, a saturated channel, a 7x count burst — are still caught.
- **Differential abundance now shows the sample count per group, and says so
  when the design cannot support its p-values.** The GLM borrows a dispersion
  shared across populations, so it still returns a p-value with ONE sample per
  group — and the table rendered that as significance stars with no `n`
  anywhere on screen, making a 1-vs-1 comparison look identical to a replicated
  one in a figure users export directly. The numbers are unchanged; `n` per
  group is now a column, and fewer than three samples in the smaller group
  draws an explicit note that the result is exploratory rather than evidence of
  a difference.

### Added
- **MEM annotation has a response control.** A planted marker-high population
  must score that marker positive, swapping which population carries the
  phenotype must swap the sign, a marker drawn identically for both populations
  must score near zero, and a stray one-event cluster must not compress the
  real scores.
- **The response controls are now attacked, not just run.**
  `scripts/mutate_controls.py` breaks each subsystem on purpose — compensation
  never applied, a gate ignoring its vertices, clustering collapsed to one
  cluster, QC removing a fixed 10% regardless of input — and checks that some
  control notices. All twelve attacks are caught. It paid for itself
  immediately by catching a containment check that could not fail, and the
  allow-list guard was re-keyed by enclosing function after a new fallback was
  found inheriting an unrelated entry's excuse.
- **Gating has a response control.** A polygon must recover the fraction
  planted inside it, an empty region must come back empty, a child gate must be
  the intersection with its parent, and the same events must be selected when
  the rows are shuffled or the index does not start at zero.
- **Acquisition QC now has a response control.** The golden baseline and the
  response controls between them never touched `AcquisitionQC` — the stage that
  runs on every sample, and the one that turned out to hold three live bugs.
  Four relational checks were added: a clean run is left alone, events removed
  tracks the span of a planted drift excursion, and the verdict is unchanged
  both by rounding the data and by the presence of a dead detector. The last
  two come back 45% and 100% apart against the previous release.
- **A regression guard for the "plausible number for an unknown" bug.** Six
  shipped defects in this codebase share one shape: a guard correctly notices
  the input is degenerate, then substitutes a value that reads as a successful
  measurement rather than one meaning *unknown* — r² = 1.0 for an undefined
  fit, a corrupt event at the end of a trajectory, `sd = 0` for one replicate,
  an entire sample deleted. A test now scans the analysis modules for that
  shape and fails on anything new, with a reviewed allow-list carrying a
  specific reason per site. It states its own limits (it only catches
  comparison-style degeneracy checks and literal fallbacks), checks that the
  scanner still matches the shape it was written for, and fails if an
  allow-list entry goes stale.

### Fixed
- **A trajectory is no longer rooted on an arbitrary cell.** `robust_root`
  returned index 0 when the root channel had no finite values, so pseudotime
  still came back looking like a trajectory while its origin and direction were
  meaningless. It now refuses, matching the `empty X` check beside it; the
  dialog already reports the error.
- **Unusable scatter no longer deletes the whole sample.** `filter_doublets`
  built its acceptance window from the median FSC-A/FSC-H ratio and substituted
  `0.0` when no event had a usable one — making the window `[0, 0]`, which
  matches nothing. A sample whose FSC-H is non-positive throughout was reduced
  to **zero events**, with no error and no warning. Doublets cannot be
  identified without the ratio, so the filter is now skipped and reported,
  which is what the missing-channel branch directly above it already did. The
  cell-cycle singlet gate had the identical fallback, where the effect was every
  cell scored `NA` and an empty cell-cycle result.
- **A group of one replicate no longer reports `sd = 0`.** The sample standard
  deviation divides by n-1, so it is undefined for a single value —
  `compare_groups` returned `0.0`, a confident claim of "this group had no
  variability", rendered straight into the group summary as `sd=0`. It is now
  NaN. A group that genuinely is constant still reports 0, because that is
  measured rather than assumed.
- **A corrupt event is no longer placed at the end of the trajectory, and no
  longer contaminates the trend curve.** Two halves of the same problem.
  `_geodesic_pseudotime` fills unreachable cells with the MAXIMUM distance — a
  deliberate, documented choice for a genuinely disconnected component — but an
  event with a `NaN` or `inf` coordinate is also unreachable, so it inherited
  that fill and plotted at pseudotime 1.0, indistinguishable from the most
  differentiated cell in the sample. Such events are now excluded and returned
  as NaN; a real disconnected component keeps its documented fill. Then
  `pseudotime_trends` binned them anyway: `searchsorted` sorts NaN **last**, so
  every unknown-pseudotime event was clipped into the FINAL bin, dragging the
  terminal point of the published "expression vs pseudotime" curve toward
  whatever they happened to express — measured, three such events at 1000 moved
  a last-bin mean from 1.0 to 600.4. Empty bins are still NaN rather than zero.
- **A degenerate MESF calibration no longer reports a perfect fit.** When every
  bead peak is assigned the same value — a data-entry mistake — r² is
  mathematically undefined, and `fit_mesf_calibration` returned **1.0**: the
  strongest possible "this calibration is good" signal, attached to a fit whose
  slope is ~0 and which would turn every converted value into the intercept.
  The dialog showed `R²=1.0000` and the user could apply it to every loaded
  sample. r² is now NaN in that case, and the calibration dialog refuses the
  fit and explains why. A real calibration is unchanged (the golden's
  `calibration.r2` still reads exactly 1.0), and a genuinely poor fit still
  reports r² well below 1 — the job the docstring claims for it.

## [2.4.0] - 2026-09-05

### Added
- **Response controls — `openflo-selftest --controls`.** A second synthetic
  suite that answers the question the golden baseline cannot. The golden pins
  seven numbers from one fixed dataset, so it proves *the answer did not
  change* — an implementation that ignored its input and returned those seven
  constants would pass all seven. The response controls instead assert
  *relationships between an input you vary and the output produced*: a
  spill-free negative control that must come back at zero, a spillover
  titration that must track its planted dose, a doublet titration, a planted
  composition shift that must be found with the correct sign, a ctrl-vs-ctrl
  null comparison that must find nothing, a compensation-APPLIER pair
  (identity is a no-op; the true signal is recovered at every spill level —
  the estimator checks cannot see an applier bug, which is where the 2.2.1
  transpose lived), and a clustering pair that scores
  purity against the true populations then shows it collapsing under a
  permutation control. **Nothing in the suite is a pinned number**, so there is
  no baseline to update and nothing to re-freeze — a failure is always a real
  behaviour change. `--all` runs both suites.
  Demonstrated to catch two sabotages the golden reports as 7/7 green: a
  transposed spillover matrix (the v2.2.1 bug class, which corrupted every real
  compensated analysis) and a hardcoded estimator.

### Added
- **GPU backend picker** (Preferences -> Performance). `gpu_backend` was read
  at startup and in the Preferences dialog but written by NOTHING, so the
  portable PyTorch backend (NVIDIA / AMD / Intel / Apple, incl. DirectML on
  native Windows) could only be selected through the undocumented
  `OPENFLO_GPU_BACKEND` environment variable — a shipped feature wired at one
  end and unreachable from the UI. The picker offers auto / CuPy / PyTorch /
  off, persists the choice, re-probes immediately and reports the resolved
  device. A meta-test now fails if any preference is read-but-never-written or
  written-but-never-read, so this class cannot recur.

- **Geometric mean and robust CV in the statistics table.** Fluorescence is
  approximately log-normal, so the geometric mean — not the arithmetic mean —
  is the central tendency FlowJo reports and papers cite; its absence meant
  anyone reconciling OpenFlo numbers against FlowJo was comparing different
  statistics. `GeoMean` is computed on positive values only (compensated data
  legitimately contains negatives, and clamping them to a floor would bias the
  result upward). `rCV` is the MAD-based robust CV, which unlike the ordinary
  CV survives the outliers real data carries. Both are opt-in; the default
  column set is unchanged.

### Fixed
- **The golden baseline's compensation metric never ran compensation.** It
  wrote the generator's own planted spillover matrix to a CSV, read it straight
  back, and asserted the value it had just written — a metric labelled
  "Compensation APC -> APC-Fire spill" that could not observe compensation
  being broken, and which therefore stayed green through the 2.2.1 transpose
  bug that corrupted every real compensated analysis. It now applies
  compensation to data with a known spillover baked in and reports two numbers
  that cannot both be satisfied by a degenerate answer: `residual_err` (max
  error against the true signal, relative to signal scale) and
  `signal_retained` (which a "zero everything" implementation would fail while
  flattering the residual). Verified to catch a transposed matrix, a zeroing
  implementation, and a no-op. **Baseline keys changed**: `compensation.apc_leak`
  is replaced by `compensation.residual_err` + `compensation.signal_retained`.
- **`--update` could bless a bug as the baseline.** Re-pinning rewrites every
  golden value from the current run, so running it while a bug was live
  recorded the breakage as correct — permanently. It now runs the response
  controls first and REFUSES if any fail, since a relational control failing
  means an output has stopped tracking its input and there is nothing
  legitimate to re-pin. `--force` overrides for a verified intentional change.
- **Auto-gate proposals were fit on the display subsample, so they changed on
  every app restart.** All three methods called `_get_df` with its default
  `downsample=True`, which caps the frame at the SMALLEST loaded sample's size
  and draws it with `random_state = hash((name, x, y, cap))`. Python randomises
  `str` hashing per process, so the same file yielded a different gate each
  launch — measured seeds for one identical expression across three processes:
  1618484138 / 2221425230 / 432000015 (stable only under `PYTHONHASHSEED=0`).
  Worse, loading a small compensation control alongside a 500k-event sample
  collapsed the cap to the control's size, fitting the big sample on ~1% of its
  events; `auto_singlet_gate`'s `frac_kept` / `ratio_cv`, which drive the
  clean-vs-REVIEW verdict, described that subsample rather than the sample.
  All three now fit on the full frame — `gmm_ellipse_gates` already does its
  own seeded 20k draw, and the other two are cheap O(n) statistics that are
  more accurate on complete data.
- **A cross-instance move could delete samples the destination never took.**
  Pasting a staged move wrote the `<move_id>.done` completion marker the moment
  the FCS loads were *queued*, and the source treats that marker as authority
  to delete its copies. Any path the destination skipped or failed to load
  still cost the source the sample AND its gate tree, while the destination
  applied nothing. The common trigger was benign: sending a sample to a window
  that already had that `.fcs` open, which the loader skips as "already
  loaded" — and the skip notice was itself overwritten by "Pulling N
  sample(s)", so the move looked successful. The marker is now written only
  once every accepted path has landed and lists the PATHS actually taken
  (names are per-instance; collision disambiguation renames them per window);
  the source removes only those, reports what it kept, and treats an
  unreadable marker as "took nothing" rather than deleting on ambiguous
  evidence. Samples already open at the destination are left in place and
  named in the status line.
- **A non-finite value no longer becomes a real, very negative measurement.**
  The biexponential backends map every non-finite input to the BOTTOM of the
  scale (-1.0) — `NaN`, `+inf` and `-inf` alike. That is silent corruption
  twice over: a saturated reading is rendered and gated as the DIMMEST event in
  the sample, and, worse, a `NaN` becomes a *finite* coordinate, which defeats
  the `dropna` that protects the plot and the gate masks — so the event goes on
  to count as a real measurement in every population, median and frequency.
  `asinh` always propagated `NaN` correctly, so the behaviour also differed by
  transform. `logicle` and `hyperlog` now match it. Finite values are
  bit-identical, and the golden baseline is unchanged.
- **A non-dict session file no longer crashes startup.** The resume read was
  guarded, but the `len(data.get('samples', []))` immediately after it was not
  — so a file whose JSON top level is an array, string or number parsed fine
  and then raised an uncaught `AttributeError` while the editor was still
  starting. Latent (nothing in the tree writes such a file), but
  `_find_resumable_session` appends the legacy `last_session.flowsession`
  unconditionally, which is how a foreign-format file would arrive.
- **A refused resume no longer leaves the session directory pointing at it.**
  `_session_dir` / `_session_data_dir` were assigned *before* the schema check,
  so an autosave rejected as "written by a newer OpenFlo" left both addressing
  the rejected file. They are now set only after the session is accepted.
- **`~/.openflo/transfer` is pruned.** A `<move_id>.done` marker is normally
  consumed by the source instance, but not when that instance has exited or
  cancelled the move — and nothing ever removed the leftovers, so they
  accumulated for the life of the install. Now pruned after a day, mirroring
  `_prune_autosaves`. Zero-byte files, so this is file-count hygiene rather
  than space.
- **The FMO gating dialog states its precondition instead of raising.** Opened
  without an active sample it did `editor._samples[None]` and died with a bare
  `KeyError: None` from its constructor. The launcher already refuses in that
  case, so this was defence in depth rather than a live path — found by a new
  sweep that drives every dialog in a nearly-empty editor, which is where the
  other two dialog crashers this cycle also lived.
- **One bad event no longer destroys a whole sample's spectral unmixing.**
  `unmix` solves every event in a single least-squares factorisation, so a
  single non-finite detector reading contaminated the entire solution: one
  `inf` event turned **all** events' abundances into NaN. (A `NaN` event
  happened to stay contained — an accident of how the SVD propagates, not a
  guarantee.) Non-finite events are now held out and returned as NaN, the
  remaining events solve exactly, and the count is logged. v2.2.3 hardened the
  reference spectra against non-finite values; this is the matching guard on
  the event data, which was overlooked. Non-finite reference spectra now also
  raise a clear error instead of numpy's opaque "SVD did not converge".
- **`.h5ad` export was completely broken and nothing caught it.** Writing an
  AnnData file raised `RuntimeError: allow_write_nullable_strings is False`
  before producing anything: pandas 3.x backs string columns with
  `StringArray`, and anndata >= 0.11 refuses to write those without an opt-in —
  so the export failed on the observation index alone. Every use of Sample QC →
  "Export AnnData (.h5ad)…" and of `openflo.write_h5ad` was affected. String
  columns and indices are now coerced to plain object dtype, which is the
  representation anndata has always written and does not depend on a global
  setting or an anndata version. The gap was structural: `to_anndata` was
  tested in memory while the file write had no coverage at all, so a value that
  serialised correctly and one that could not be serialised looked identical.
  There is now a real round-trip test — write a file, read it back with
  anndata, and check the matrix, sample labels, cluster columns, marker names
  and recorded `dropped_markers` all survive.
- **An unreadable audit entry no longer vanishes from the Methods paragraph.**
  `methods_paragraph` builds text meant to be pasted into a manuscript, and it
  silently skipped any audit entry that was not a record (and silently replaced
  malformed `details` with an empty one). The result was a step the user
  actually performed being absent from their published description of what they
  did, with nothing saying so. The drop still happens — a corrupt record holds
  nothing to recover — but it is now reported inside the paragraph itself,
  where it cannot be published without being noticed. Clean trails produce
  clean prose, unchanged.
- **`_normalise_groups` is now idempotent.** Normalising an already-normalised
  group silently discarded per-sample FMO overrides: the first pass flattens
  `[{'name': 'd', 'fmo_set': 'X'}, 'e']` to `['d', 'e']` and records the
  override in `sample_fmo`, and a second pass — seeing plain strings —
  overwrote that map with the group default. The sample would then be gated
  against the WRONG FMO control, silently. No shipped path normalised twice, so
  this was latent; but three call sites carry a comment telling the next
  developer to "normalise first" (the fix for a different bug in the same
  area), which actively invites a fourth. Overrides now survive a repeat pass,
  while the group default still reaches samples that lack one.
- **Auto-gate now fits on the population it attaches to.** In Display=filter
  mode `apply_gates_var` is set, so the frame `_get_df` returns is the union of
  every ENABLED gate's chain — while `_add_gate` parents the result on the
  SELECTED row, which is `None` (root) when a sample row is selected. Filtering
  down to lymphocytes and running auto-gate therefore fit ellipses on
  lymphocytes and attached them at root, where they also captured monocytes and
  debris, with nothing reporting that the fit domain differed from the applied
  domain. Auto-gate resolves the destination parent once and fits on exactly
  that population. `_get_df` gained an explicit `gate_parent` argument for
  callers that need a specific population rather than "what is on screen".
- **A failed autosave on exit is now reported instead of vanishing.** `_on_close`
  detached the in-app console's log sinks and *then* wrote the autosave,
  reporting failure with a `print()` into a sink nothing could receive, moments
  before `destroy()`. Disk full at exit meant the session was lost, the next
  launch had nothing to resume, and no message ever reached the user. The
  autosave now runs before the sinks are detached and raises a dialog on
  failure — the last moment the state still exists. The sidecar-failure flag
  that `File → Save session` already warns about is honoured here too, so an
  autosave that silently degraded to raw-FCS-only (losing clusters, embeddings
  and FMO gates on resume) now says which samples were affected.
- **A second Send no longer strands the first batch's move flags.** Staging a
  new move overwrote `_pending_move` without clearing the previous one, and
  `_mark_pending_move` only ADDS to the flagged set while both clearing paths
  clear the *new* batch's names — so the first batch's rows kept the ✄
  pending-move indicator for the rest of the session, with two 700 ms poll
  loops running in parallel. A new Send now supersedes the old one properly and
  cancels its poll.
- **A gate replicated to other samples now measures the same population.**
  With "-> all shown" on (the default), `_add_gate_multi` forced every replica
  to `parent_id = None` while the active sample's copy kept the selected
  parent. So with a parent gate selected, sample A got `Singlets` nested under
  `Lymphocytes` and samples B..N got `Singlets` at ROOT — one gate name
  measuring a different population per sample, which silently invalidates the
  cross-sample frequency comparison the fan-out exists to enable. Replicas are
  re-parented by population PATH (ids are per-sample and meaningless on a
  target); a sample that genuinely lacks that population still gets a root
  gate, but is now named in the status line instead of being quietly wrong.
- **Auto-gate provenance recorded only the active sample** while the proposal
  was applied to every displayed one; the audit entry now carries the full
  applied-to list.
- **FlowJo-comparison sweeps share one gate-mask cache.** `compare_wsp` walks
  every population in a workspace against the same sample, each re-evaluating
  its whole ancestor chain — the same redundancy already fixed in
  `population_stats` and `_get_df`.
- **Sample QC's `_xy` was assigned only in `_compute`**, after `_D`, so an
  exception between them left `_D` set and `_xy` missing while `_draw` read
  both — the same class as the `ui_voltage._chan_lookup` landmine. Initialised
  at construction and guarded in `_draw`.
- **Sample QC crashed when opened with nothing selected.** `_markers()` reads
  `names[0]` and both of its callers invoked it *before* their own length
  check, so an empty selection raised an unguarded `IndexError` inside an
  `after()` callback — swallowed by Tk, leaving the window blank with no error.
- **Voltage dialog's `_chan_lookup`** was only assigned inside
  `_populate_channels`, which returns early when the first FCS is unreadable;
  `_selected_channel()` reads it and was safe only by call order. Now
  initialised at construction.

### Changed
- **Population statistics are 5-7x faster on real gating hierarchies.**
  `cumulative_gate_mask` gained opt-in memoisation, and `population_stats` now
  shares one cache across the collection pass, so a gate reuses its parent's
  cumulative mask instead of re-walking and re-evaluating the whole ancestor
  chain. This matters because a polygon gate costs ~250x a threshold gate
  (`Path.contains_points` over every event) and real hierarchies are
  polygon-based: over 200k events a 32-gate panel drops from 1.82 s to 0.36 s
  and a 60-gate panel from 4.72 s to 0.67 s. Masks are unchanged — verified
  identical for nested trees, cycles, missing parents and injected overrides.
  Callers that omit `cache` keep the previous allocate-fresh behaviour exactly.
- **Dragging a 1-D gate no longer rebuilds the gate tree on every mousemove.**
  The line-gate drag and the histogram slider both rebuilt the whole
  samples-and-gates tree per tick. That rebuild is O(total rows) across every
  *loaded FCS file*, not just the gate being dragged — about 8 ms for a
  20-file workspace with 30 gates each (621 rows), which at drag rate is
  roughly half of wall-clock spent rebuilding a tree nobody was looking at.
  Both now refresh once at end-of-drag, which is what every other drag kind
  already did (and what the code's own comment said it did). The gate itself
  still updates every tick, so the plot tracks the cursor as before.
- **Replotting is ~23% faster on a multi-file workspace.** `_get_df` OR-ed the
  cumulative mask of every enabled gate, each re-walking and re-evaluating its
  whole ancestor chain; it now shares one call-local cache. Profiling an 8-file
  x 50k-event x 12-gate workspace, `_replot()` drops from 1034 ms to 797 ms and
  gate-mask time halves. (Profiling also showed matplotlib's `tight_layout` and
  `loc='best'` legend placement dominate what remains — see the note in
  DESIGN_NOTES rather than a code change, since legend position is a visual
  decision.)
- **New preference: legend position** (Preferences -> Appearance), and with it
  a ~1.9x faster replot by default. matplotlib's automatic (`loc='best'`)
  placement tests candidate corners against EVERY plotted point, so it costs
  ~1.1 s on 8 samples x 50k events — and it ran on every replot. Three modes:
  *Automatic (reuse position)* — the new default — solves the placement once
  per plot shape, remembers where it landed in axes coordinates and reuses it
  until the axes or sample set change; *Automatic (recompute each time)* — the
  previous behaviour; and *Always top-right* — cheapest, but can sit on top of
  data. Measured on an 8-file x 50k-event x 12-gate workspace: 808 ms per
  replot recomputing, 435 ms reusing, 440 ms fixed — so the default now costs
  what a fixed corner costs while keeping the placement that automatic
  mode chose.
- `make_compensation_controls` accepts a `leaks` mapping so the planted
  spillover can be varied (or removed entirely) rather than being fixed.
- `immunophenotyping_sample(return_labels=True)` additionally returns the TRUE
  per-event population, so a clustering result can be *scored* rather than
  merely counted. The labels ride through the existing shuffle as an extra
  column — a row-wise Fisher-Yates cannot observe a column, so the seeded RNG
  stream is untouched and every generator output stays bit-identical (asserted
  by a test, including the `batch_gain` and `fmo` paths). Reading this as a
  ground truth revealed that the golden's `leiden_n = 18` against 6 true
  populations is benign over-clustering, not error: homogeneity is 1.00, so the
  extra clusters are pure sub-clusters that never mix populations.

## [2.3.0] - 2026-09-03

### Added
- **Theme audit harness** (`scripts/theme_audit.py`). Builds the editor under a
  chosen theme on a withdrawn root, walks every widget, combobox popdown and
  figure facecolor, and flags near-white surfaces — so a light-leaking widget is
  caught mechanically rather than by eye. Wired into the test suite as a Midnight
  no-white regression.

### Changed
- **Refreshed GUI chrome ("sleek").** The `light` / `dark` / `midnight` themes
  are restyled to a hand-tuned, dependency-free look: a near-black / off-white
  palette with a cool accent, flat surfaces, hairline borders tied to the
  palette, more generous padding, muted headings and tabs, and accent-on-focus.
  Check and radio indicators are flat and understated. Theme names and the
  Preferences picker are unchanged.
- **Auto-clean now follows the standard gating order.** The doublet FSC-A/FSC-H
  ratio median was computed over *all* events, debris included. When debris
  cleaning is also enabled, the ratio window is now centred on the
  debris-removed (cell-sized) population, matching the conventional hierarchy.
  AND-of-methods semantics are unchanged. **This can shift doublet-gate results
  on runs that enable both methods.**
- **The unclustered / noise bucket is kept and named everywhere.** The `-1`
  sentinel (non-finite-channel events, PhenoGraph outliers, unassignable
  sub-sample rows) previously surfaced as a bare `-1`, as a `C-1` bar label, or
  was dropped outright depending on the path. It is now retained and labelled
  "Unclustered (noise)" — via one shared `pipeline.cluster_label()` used by both
  the pipeline stats and the GUI — across per-sample frequency exports, the
  cluster heatmap, group-vs-group comparisons and the annotate dialog. Frequency
  exports gained a leading `population` label column. Real clusters' counts and
  percentages are unchanged. The imported noise population is drawn in a neutral
  grey so it does not read as a real population.
- **FlowSOM and cell-cycle unassigned buckets** are likewise kept and labelled
  ("Unassigned (noise)"; `NA` for cell cycle) instead of being silently dropped,
  and category populations use friendly names ("FlowSOM metacluster 3") rather
  than the raw storage token ("flowsom_meta 3").
- **Workspace compensation column**: an item with no compensation is flagged
  with "⚠" alone, now clearly distinct from a group's neutral "—" (no override).
- **Prism grouped export** keeps replicates with a missing row/column factor
  under an explicit `(unassigned)` level instead of dropping them from the table.
- **Faster grouped runs.** Preparing a run unit copied every member's full event
  frame just to tag it with its group and sample — N large copies before the
  concatenation's own copy, a multi-second UI freeze on a grouped run. The tags
  are now applied once, after the concatenation. The output is identical and the
  source frames are no longer touched.

### Fixed
- **Exported population `.fcs` files could contain the wrong events.** The
  cumulative-gate walk in the population export swallowed every exception, so a
  gate that failed to evaluate was silently dropped and the exported file held
  the *un-restricted superset* — silent data corruption in a file users take
  downstream. A failed gate now skips that population, and the skip is surfaced
  in the status line.
- **A gate that errored admitted every event.** `_evaluate_gate_on` returned an
  all-True mask on failure, an inflated superset that read as success. It now
  fails closed (all-False), so the population empties visibly. Valid gates and
  the "channel missing → skip" paths are unaffected.
- **A failed per-channel transform left that channel in raw scale** among its
  logicle siblings, so gates and plots compared against raw values. Failures are
  now recorded and logged prominently, listing the affected channels.
- **Preferences could be wiped.** `write_pref` did a read-modify-write, and
  `read_prefs()` returns `{}` on *any* error — so a single transient file lock
  (antivirus, search indexer) persisted a one-key file, discarding every other
  setting. A read error now aborts the write instead of clobbering; only a
  missing or corrupt file resets. Writes are atomic.
- **Sessions are written atomically.** A mid-write failure previously truncated
  the existing good session or autosave.
- **Session sidecar failures were invisible.** A processed-data sidecar write
  failure was printed to stdout while the save reported clean success — the
  sample then silently reloaded from raw FCS with its clusters, UMAP and FMO
  gates gone. The save now warns and names the samples that lost computed
  results.
- **Session display state was lost on a slow resume.** The restore was a single
  600 ms one-shot that cleared unconditionally, so any sample finishing later
  kept its default (unchecked) plot state and lost its saved axis. It now
  re-applies while samples are still loading.
- **`PI3K` was mis-detected as the PI DNA stain.** The `pi` token boundary
  allowed an adjacent digit, so a PI3K signalling channel was auto-selected as
  the DNA channel.
- **CLI `--gates` threshold overrides were silently dropped** for any group with
  an empty FMO set — that is, every by-day auto-group — while `--export-wsp`
  still recorded them, so the two paths disagreed. Overrides now apply to every
  sample. Malformed or incomplete gate dicts are reported rather than discarded
  in silence.
- **CLI `--gates` help documented a form the parser rejects.** Copying the
  documented `{"BV421-A":0.5}` example ran the analysis **ungated**; the help now
  shows the accepted JSON-list form.
- **`--export-wsp` resolved zero samples** for a comma-string or per-sample-dict
  `samples` spec (it iterated the string or dict). It now normalises groups
  first, honours each group's `trial_dir` so by-day filename collisions pick the
  right day's FCS, and allocates gate ids in two passes so a child-before-parent
  gate nests correctly instead of re-rooting. The panel probe had the same
  un-normalised-groups bug and silently dropped the panel.
- **Compliance records verified as valid when a signed file was deleted** — the
  missing file was dropped from the checked set, and `all([])` is `True`. The
  check is now fail-closed.
- **KDE density crashed** on a single-event or constant-channel gate (it had no
  degenerate-input guard); it now falls back to flat density so the scatter still
  draws.
- **t-SNE was silently skipped for 3–5 events** — the perplexity clamp floor of
  5.0 exceeded *n*, so the call still raised.
- **Stale grouping in the frequency dialog.** The Group-by combo, Tokens entry
  and Parametric checkbox did not trigger a rebuild, so "Diff. abundance…" and
  "Compare all…" could run the GLM on the previous grouping while the UI showed a
  new one.
- **All-NaN channels produced garbage p-values** presented as valid in group
  stats; non-finite values are now filtered before the per-group median.
- **A clustering worker could be read mid-write.** The plot redraw could observe
  sample DataFrames while a clustering or embedding worker mutated them off the
  Tk thread (transient `KeyError` / torn read); the redraw now defers until the
  run completes.
- **Undo could not revert a gate toggle.** Toggling a gate or auto-clean method's
  enabled checkbox changes real gating output but took no undo snapshot.
- **A 1-D gate re-commit silently re-enabled a disabled gate** or dropped its
  name — the replace path preserved only the colour, not the enabled/name/open
  state.
- **FlowJo comparison could read as valid on uncompensated data.** A
  compensation failure left every gate evaluated against uncompensated events
  with no per-row flag; rows now carry an explicit "uncompensated" error.
- **Sample-distance MDS collapsed onto the origin.** An undefined pair (no shared
  usable markers) scored 0.0, which reads as "identical"; it is now `NaN`, and
  the embedding places those samples at the edge while still returning finite
  coordinates.
- **AnnData and distance-matrix exports** use the shared-marker intersection
  consistently (heterogeneous samples no longer raise), record which markers were
  omitted (in `adata.uns['dropped_markers']`) instead of silently shrinking the
  panel, and add a labelled sibling `obs` column alongside the raw integer id.
- **Cross-platform paths and encoding.** Session `rel_path` and `processed_csv`
  are stored forward-slash and normalised on read, so a Windows-authored
  `.flowsession` relinks its FCS on Linux and macOS; the voltage-titration CSV is
  opened as UTF-8 (it was the only text `open()` without it, giving cp1252
  mojibake on non-ASCII channel names); the watch-folder seen-set is keyed on the
  exact filename, so two case-distinct files no longer collapse on Linux.
- **A legacy workspace with deletion gaps could overwrite an existing item** —
  the `_mseq` / `_gseq` fallback resumed from the item *count* rather than the
  maximum id suffix, re-minting a live id.
- **Frequency and heatmap exports were hardcoded to the `cluster` column**, so
  Leiden and FlowSOM label columns got no noise-labelled export; both now take a
  `label_col` (the default preserves the previous behaviour).
- **Smaller dialog and export guards**: Log scale with Min ≤ 0 is rejected
  instead of producing an empty axis; the FMO percentile is validated in [0, 100]
  up front (out of range previously surfaced as a misleading "nothing added");
  quadrant counts guard an empty frame (`ZeroDivisionError`); a 0.0 annotation
  score with a non-positive threshold no longer raises `KeyError`; a zero-drop
  gate no longer shows a "drops 0" suffix; non-separable voltage SI/rCV render
  `n/a` rather than a bare `nan` in both stdout and the CSV; `write_fcs` logs a
  count of the non-finite cells it zeroed; and a group comparison that produced
  no rows now says so instead of writing nothing silently.
- **Theming completeness.** About 25 hint, status and caption labels moved from
  hardcoded greys to a palette-tracked `Muted.TLabel` token (dark greys were
  invisible on the dark panel); free matplotlib artists that the figure theming
  cannot reach — significance brackets, empty-state text, FlowSOM spokes, and the
  gating-tree diagram's connectors and node labels — now take theme-correct ink;
  Treeview bad/warn row tints are theme-matched (previously an unreadable light
  band under Midnight); a colourless gate row no longer renders black-on-dark;
  and the populations context menu, the voltage dialog's initial paint and the
  progress-bar trough no longer leak light. A live theme switch now re-themes
  already-open dialogs instead of leaving them on the old palette.
- **Space utilisation.** The frequency summary gained a vertical scrollbar — long
  multi-group / BH-pairwise output was previously truncated invisibly in a fixed
  six-line box — and the compensation spillover matrix stretches to the canvas
  width instead of jamming into the top-left corner.
- **A closed tool window retained its figure and cached DataFrame** until it was
  reopened; the registry slot is now released on destroy.
- **A long-lived log-drain timer** on a destroyed window could reschedule itself
  forever (a latent leak).
- Corrected 14 docstrings and comments that contradicted the code — including
  `gate_to_mask`'s missing-channel behaviour (all-True for geometric gates,
  all-False for cluster and category gates), `WspWriter`'s round-trip scope (only
  boolean is out of scope), the compensation editor's four-step auto-import order
  and non-modal behaviour, `multi_group_test`'s Friedman truncation, and
  `_sample_group_label`'s longest-match token rule.

### Removed
- The unused `tqdm` dependency, which was declared as a core requirement but
  never imported.

### Security
- **FlowJo `.wsp` parsing rejects a DTD or `ENTITY` declaration** before parsing,
  closing the stdlib ElementTree entity-expansion denial of service ("billion
  laughs" / quadratic blowup). No XXE file read was possible, and no new
  dependency was added.
- **Session relink is contained to the session directory.** `processed_csv` is
  always app-generated and relative, so absolute or `..`-escaping values are now
  rejected, closing an arbitrary-file-read vector. `rel_path` still permits
  legitimate cross-directory projects but warns when it resolves outside the
  session tree.

## [2.2.4] - 2026-07-02

### Changed
- **Test-coverage hardening — no runtime changes.** A ~40-test sweep from a
  "looks-tested-but-core-untested" audit now pins the *core behavior* of
  components that previously had only periphery coverage (parse / IO round-trip /
  sign / monotonicity) — the same class of gap that let the 2.2.1
  compensation-transpose bug ship. Newly pinned: FCS per-cell row pairing +
  external-read fallback, gate interval + multi-gate AND masks +
  cumulative/override semantics, density event x/y alignment + KDE-vs-SciPy +
  smoothing floor, spectral autofluorescence + bright-event selection + detector
  ordering, effect-size magnitude, workspace↔FlowJo comparison arithmetic + XML
  gate parsing + inventory walk, clustering purity + subsample assignment +
  non-finite handling, trajectory geodesic property + bin means, CytoNorm
  per-metacluster proportion preservation + QC magnitude, calibration r² + peak
  fallbacks, voltage stats (robust CV, arcsinh split, channel resolution). **No
  source changed; every closure confirmed the existing code correct (zero bugs
  found).** The compensation-transpose shape of bug would now be caught across
  the codebase.

## [2.2.3] - 2026-07-02

### Fixed
- **Defensive robustness hardening (code-review sweep).** A batch of guards that
  turn edge-case crashes / silent-wrong results into graceful handling:
  - Compensation: a singular spillover matrix is skipped with a warning instead
    of raising; compensation-QC rejects non-finite (NaN/inf) matrices; a
    malformed FCS `$SPILL` keyword is reported + ignored rather than silently
    read as "no compensation".
  - Spectral unmixing: non-finite events in a single-stain control no longer
    poison its reference spectrum; a zero-norm spectrum reads self-similarity 1.
  - CytoNorm: clear errors for `n_quantiles < 2` and for a batch missing a
    normalized channel (were cryptic downstream crashes).
  - Session restore: a corrupt per-channel display range is skipped instead of
    aborting the whole load; auto-saved sessions are schema-migrated on resume
    (same as a manual open).
  - FlowJo import: gates on natively-compensated channels (FlowJo's `<PE-A>`
    bracket form) now match the data instead of selecting everything; the
    region/CLI gate filter evaluates all gate kinds (ellipsoid / cluster /
    category / boolean), not just lines and polygons.
  - Auto-clean / cell-cycle: no more spurious divide-by-zero warnings; the
    cell-cycle width-channel lookup matches lowercase `-w`/`-h` columns.

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
