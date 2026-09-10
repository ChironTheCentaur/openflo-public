# OpenFlo algorithms — what we do and why

A short tour of the scientific choices baked into the pipeline.  If
you're evaluating whether OpenFlo is the right tool for your panel,
or sanity-checking a number that looks off, this is the page.

All defaults below match the source as of v0.2.0 — if the code drifts,
file an issue and we'll fix this doc.

---

## 1. Compensation

### Source of the matrix

In order of preference:

1. **From `$SPILL` / `$SPILLOVER` in the FCS keyword block** — what
   `auto_compensate()` does.  This is what the cytometer wrote at
   acquisition time, usually populated by Diva / Sphero / similar.
2. **From a FlowJo `.wsp`** — `compensate_from_wsp(path)` pulls the
   spillover matrix that FlowJo computed.  Useful when the operator
   re-derived spillover after the fact.
3. **Manual** — `manual_compensate(matrix, channels)`, or the GUI's
   compensation editor.
4. **Empirical via `optimize_compensation`** — single-stain control
   FCS files → NxN spillover via least-squares.  Defaults:

   | Parameter             | Default | Why                                                                              |
   |-----------------------|---------|----------------------------------------------------------------------------------|
   | `positive_percentile` | 95.0    | top 5% of the source channel's distribution counts as "positive" for regression. |
   | `min_events`          | 100     | refuse to estimate from fewer positive events than this.                          |

   The 95th-percentile / 100-event floor was tuned for typical BD
   CompBead controls (5,000–50,000 events per tube).  For sparser
   single-stain panels, drop `positive_percentile` to 90 and
   `min_events` proportionally.

### What the matrix means

The matrix is **spillover**, not "compensation".  Row *i*, column *j*
is the fraction of channel *i*'s true signal that bleeds into channel
*j*'s detector.  OpenFlo inverts the matrix and applies it to the
data — this is the standard FlowJo / FCS Express convention and is
mathematically equivalent to "subtracting the spillover".

The matrix is always **applied to the data in place** (`s.data` gets
the compensated values) and **persisted on the FlowSample** as
`s.comp_matrix` / `s.comp_channels` so the workspace exporters can
round-trip it.

---

## 2. Transform (logicle)

After compensation, fluorescence channels are run through
**Parks et al.'s logicle transform** (`flowutils.transforms.logicle`).
This compresses bright populations into a linear-near-zero region
that handles the negative values compensation routinely produces.

Defaults (`apply_transform`):

| Parameter | Default | Notes                                                                |
|-----------|---------|----------------------------------------------------------------------|
| `t`       | 262144  | Top of the data range. Matches FACSDiva's 18-bit ADC ceiling.        |
| `m`       | 4.5     | Decades of "log" portion above the linear-near-zero region.          |
| `w`       | 0.5     | Width of the linear-near-zero region (in decades). 0.5 is FlowJo's default. |
| `a`       | 0       | Additional negative decades.  0 = no negative tail beyond w.         |

These defaults match FlowJo's built-in "Logicle" choice and reproduce
its visual scale exactly for any panel where `$P{n}R` is 262144.  For
older instruments (CyAn ADP, FACSCalibur), `t` should drop to match
`$P{n}R` — pass `t=` explicitly.

Scatter channels are NOT transformed.  They stay in linear units so
the FSC/SSC gates stay interpretable.

---

## 3. Gating

### Threshold gates (1-D)

`apply_threshold_gates({'CD45-A': 0.45})` adds a `CD45-A_pos` boolean
column.  The threshold is in transformed units (so 0.45 means "logicle
position 0.45", not "raw fluorescence 0.45").  Annotation only — the
underlying data isn't filtered.

### Region gates (2-D shapes)

`apply_region_gates([gate_dict, ...])` filters `s.data` in place.
Supported `kind`s: `threshold`, `interval`, `rect`, `polygon`,
`ellipsoid`, `quadrant`.  See `gate_to_mask` for the schema.

Gates compose hierarchically via `parent_id`.  `cumulative_gate_mask`
walks the parent chain to produce the final mask — children only see
events their ancestors kept.

### FMO-derived thresholds

`FMOGater.compute(percentile=99.5)` reads the *p*-th percentile of
each FMO control's distribution in the target channel.  Default 99.5
matches Roederer 2011's recommendation for clean FMO panels (where
the FMO is genuinely fluorescence-negative).

For panels where the FMO is "low fluorescence" rather than truly
negative (e.g. CD45-PE on whole blood), drop to **98–99** — the
strict 99.5 cutoff will produce floating thresholds that drift across
acquisitions.

### Reference-based thresholds (alternative)

Where an FMO is impractical (cost, cell count), pass a fixed
threshold dict to `apply_threshold_gates({'CD45-A': value})` directly.
The value goes into the exported `.wsp` as a threshold gate so the
choice is auditable downstream.

---

## 4. Clustering

`s.cluster(channels=None, k=30, n_jobs=1, max_events=None)`.

### Algorithm — Phenograph (Levine 2015)

Phenograph builds a *k*-nearest-neighbour graph in fluorescence
space, computes Jaccard similarity between neighbour sets, then runs
Louvain community detection on the weighted graph.  We use the
upstream `phenograph` package (CPU) with an optional GPU path via
RAPIDS cuGraph (`_cluster_gpu`).

### Choosing *k*

Default 30.  Rule of thumb:

| Sample size    | Suggested *k* |
|----------------|---------------|
| < 1,000        | 10–15         |
| 1,000 – 10,000 | 20–30         |
| 10,000 – 100,000 | 30          |
| > 100,000      | 30–50         |

Larger *k* smooths the graph (fewer, broader clusters); smaller *k*
picks up rare populations but can fragment large ones.  When in
doubt, try k=15 and k=30 and compare cluster medians — the
qualitative population structure should be stable.

### Subsample + KD-tree-assign for large samples

For samples larger than `max_events`, Phenograph runs on a random
subsample of `max_events` cells.  The remaining cells are then
assigned to the nearest cluster centroid via a 1-NN KD-tree fit in
the same fluorescence space.

This is the standard scalability trick used in FlowSOM, Phenograph's
own helpers, and most production cytometry pipelines.  It loses
~1–2% of the silhouette score vs full-Phenograph but cuts wall-clock
roughly proportionally with the subsample ratio.

Recommended `max_events` settings:

- **500,000** — best quality, single sample, plenty of RAM.
- **200,000** — typical default for batch runs.
- **50,000–100,000** — for `--workers > 1` to fit several samples
  in memory simultaneously.

The subsample is seeded by the run-wide `--seed` (default 42) so
results are reproducible.

### GPU path

When RAPIDS (`cugraph`, `cudf`, `cuml`, `cupy`) is importable and
free VRAM exceeds `--admission-vram` (default 1.0 GB), clustering
runs on the GPU: cuML's kNN feeds a cuGraph weighted graph that
cuGraph's Louvain implementation processes in parallel.  Falls back
to CPU on any RAPIDS / VRAM failure.

GPU results are **not bit-identical** to CPU — cuGraph's Louvain
uses a different vertex ordering than the CPU implementation, which
shuffles cluster labels (not membership).  The cluster-to-cluster
mapping is stable; the integer label of a given cluster is not.

---

## 5. Dimensionality reduction (UMAP)

`s.run_umap(n_neighbors=30, min_dist=0.3, sample_n=100_000, random_state=42)`.

We use the upstream `umap-learn` package with these defaults:

| Parameter      | Default | Notes                                                      |
|----------------|---------|------------------------------------------------------------|
| `n_neighbors`  | 30      | Matches the Phenograph k default, so the local-structure scale is comparable. |
| `min_dist`     | 0.3     | Tighter (0.1) makes clusters more compact at the cost of inter-cluster overlap. |
| `sample_n`     | 100,000 | UMAP scales poorly above ~100k events; subsample and project the rest. |
| `random_state` | 42      | Seeded for reproducibility. Pinning this forces `n_jobs=1` internally — that's a UMAP-library limitation, not ours. |

The "project the rest" step uses UMAP's `transform()` method on the
non-subsampled events.  Visually identical results to the full UMAP
fit for samples up to ~1M events.

---

## 6. Determinism + reproducibility

The pipeline accepts `--seed` (default 42).  This is held constant
across every sample in a trial so:

- Phenograph subsample selection is reproducible per sample
- UMAP embedding is reproducible per sample
- Same `--seed` across runs → bit-identical CPU outputs

GPU clustering is **not deterministic** — cuGraph's Louvain can
return different cluster labelings on the same input on the same
hardware.  Pin the run to CPU with `--admission-vram=0` if exact
reproducibility matters more than speed.

Numpy / scipy / scikit-learn minor-version bumps occasionally shift
the float-comparison tie-breaking in kNN distance ranking, which can
re-order Louvain's initial seed.  The
`tests/test_golden_regression.py` snapshot catches drift larger than
a few percent — if your reproducibility tolerance is tighter than
that, pin the dependency versions in your environment.

---

## 7. What this pipeline is NOT good at

- **Rare-event analysis** (< 0.01% of total).  Phenograph + Louvain
  is variance-loss-prone in the long tail.  Use FlowSOM or FlowOlap
  for those.
- **Cytof / mass cytometry.**  The logicle defaults are tuned for
  18-bit fluorescence; mass cytometry channels have very different
  dynamic range.  Pass `t=`, `w=`, `m=` overrides at minimum, and
  consider `arcsinh` instead.
- **Bulk-population spectral unmixing.**  We do spillover, not
  spectral unmixing.  For Aurora / Cytek-style spectral data, pre-
  unmix in the instrument software and feed the unmixed FCS in.

---

## References

- Parks, Roederer & Moore (2006). *A new "Logicle" display method
  avoids deceptive effects of logarithmic scaling.* Cytometry A
  69:541-551.
- Levine *et al.* (2015). *Data-driven phenotypic dissection of AML
  reveals progenitor-like cells that correlate with prognosis.*
  Cell 162:184-197. (Phenograph)
- McInnes, Healy & Melville (2018). *UMAP: Uniform Manifold
  Approximation and Projection for Dimension Reduction.* arXiv
  1802.03426.
- Roederer (2011). *Compensation in flow cytometry.* Curr. Protoc.
  Cytom., Chapter 1, Unit 1.14. (FMO threshold rationale)
