# OpenFlo

A Python pipeline + Tk GUI for high-content flow cytometry analysis.
Reads FCS files, applies FlowJo-compatible compensation and gates, clusters
populations with Phenograph (CPU or RAPIDS GPU), projects with UMAP, and
exports per-sample / per-condition statistics, plots, and a FlowJo `.wsp`
that round-trips back to the desktop tool.

> Panel-agnostic — point it at any FCS set and pass `--channels`. The
> examples throughout use a generic 3-colour panel (BV421=CD11b, APC=CD34,
> PE-Cy7=CD45); substitute your own markers and controls.

---

## Features

- **FCS I/O** via FlowIO; auto-detect channels, scatter, time
- **Compensation matrix** — read from `.wsp`, `.fcs` spillover, or CSV/TSV;
  manual editor; single-stain auto-optimizer
- **Gating** — FlowJo Gating-ML v2 reader and writer (rectangle, polygon,
  ellipsoid, quadrant); FMO-based threshold mode for fluorescence channels
- **Gating** — rectangle, polygon, ellipsoid, quadrant, threshold/FMO,
  **boolean (AND/OR/NOT)**, plus density-based **auto-gating** and
  **undo/redo** in the editor
- **Transforms** — logicle, hyperlog, arcsinh, log (per-channel editor)
- **Clustering** — Phenograph (Louvain, optional cuGraph GPU) and **FlowSOM**
  (SOM + metaclustering)
- **Dimensionality reduction** — UMAP (seeded/reproducible); optional
  TriMap and PaCMAP (`pip install openflo[embed]`)
- **Differential analysis** — abundance/expression between sample groups
  (Mann-Whitney + log2 FC + BH-FDR)
- **Spectral unmixing** — reference-spectra least-squares for full-spectrum
  cytometers
- **Quality control** — time-based signal-drift detection plus flow-rate
  anomalies (clogs/bubbles) and margin/saturation event removal
- **Cell cycle** — DNA-content G1/S/G2M modelling (PI/DAPI/FxCycle/7-AAD/
  Hoechst/DRAQ5…), in the pipeline and the editor (phases as populations)
- **Voltage titration** — Stain Index voltage walk over a titration series
  (`openflo-voltage`), with a recommended plateau voltage
- **Outputs** — per-sample heatmaps, scatter, FSC/SSC, UMAP, concatenated
  condition comparison plots, FlowJo Table-style CSV, and a saved `.wsp`
- **Tk GUI** with drag-and-drop, per-sample gate trees, compensation editor,
  and an interactive post-analysis viewer
- **`compare_workspace.py`** — diff OpenFlo vs FlowJo population counts cell
  by cell, HTML + CSV report

---

## Install

Python **3.11+** required.

```bash
git clone https://github.com/ChironTheCentaur/openflo.git
cd openflo
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

This installs the `openflo-run`, `openflo-compare`, `openflo-gui`, and
`openflo-voltage` console scripts on your PATH.

Optional GPU acceleration (RAPIDS — conda only, not on PyPI):

```bash
conda install -c rapidsai -c conda-forge rapids=24.10
```

The pipeline auto-detects GPU at import time and falls back to CPU.

---

## Quickstart

### GUI
```bash
openflo-gui
```
Or use the quick-launch scripts at the repo root (they pick up the project
`.venv` automatically, and work from a fresh checkout without `pip install`):

- **Windows** — double-click `openflo-gui.bat` (or run it in a terminal)
- **Linux / macOS** — `./openflo-gui.sh` (`chmod +x openflo-gui.sh` once)

Drag FCS files into the file pane, set channels, click **Run**.

### CLI
```bash
openflo-run \
  --trials /path/to/fcs/ \
  --out outputs/ \
  --seed 42 \
  --export-wsp outputs/analysis.wsp \
  -v                                # -v for INFO, -vv for DEBUG, -q for warnings only
```

### Compare OpenFlo vs FlowJo
```bash
openflo-compare my_flowjo.wsp --html compare_report.html
```

### Programmatic
```python
import openflo

s = openflo.FlowSample("sample.fcs")
s.run_qc()
s.auto_compensate()
s.apply_transform()
s.cluster(k=30)
s.export_stats("sample_stats.csv")
```

---

## Common workflows

### Single-sample exploration in the GUI

Load one FCS, gate interactively, save the gate set as a reusable
template:

```bash
openflo-gui
```
1. **Add FCS** → pick your sample. It auto-compensates from `$SPILL` and
   logicle-transforms on load.
2. Pick X / Y channels from the combos; use **Mode → pseudocolor** for
   a density view.
3. Pick a **Tool**: Quadrant for a click-to-drop four-way split,
   Polygon for a freehand region, Edit to tweak vertices on an existing
   gate (see the hint line under the tool selector for gestures).
4. **Save Template…** writes a `.json` you can load later or hand to
   the CLI via `--gates`.

### Multi-trial batch run with FMO gating

Process every FCS under `trials/` against a single FMO control set,
producing per-sample stats CSVs, per-condition comparison plots, and
a FlowJo-compatible workspace:

```bash
openflo-run \
  --trials trials/day1,trials/day2 \
  --groups '[{"name":"Day1","samples":["m1","m2"],"fmo_set":"standard"},
             {"name":"Day2","samples":["m1","m2"],"fmo_set":"standard"}]' \
  --fmo-sets '{"standard":{"BV421-A":"bv421-fmo","APC-A":"apc-fmo","PE-Cy7-A":"pecy7-fmo"}}' \
  --k 30 \
  --workers 4 \
  --seed 42 \
  --export-wsp outputs/analysis.wsp \
  --out outputs/ \
  -v
```
The result: `outputs/Day1/`, `outputs/Day2/`, each with cluster
heatmaps + UMAP + per-sample stats; plus a single `analysis.wsp` you
can open in FlowJo to validate the gates.

### Compare OpenFlo's gating against FlowJo

`openflo-compare` reads a FlowJo workspace, re-applies every Population
gate via OpenFlo's evaluator, and emits a side-by-side diff of event
counts per population per sample:

```bash
openflo-compare my_flowjo.wsp \
  --fcs-dir test_fcs/ \
  --html compare_report.html \
  --csv compare_report.csv
```
The HTML report colour-codes |Δ| > 5% rows for review. Use it to
validate a new OpenFlo version against an established FlowJo
analysis before switching the analysis pipeline over.

---

## Synthetic data & self-test

OpenFlo ships a **seeded synthetic dataset generator** and a **self-test** so you
can try features and confirm a change hasn't altered other behavior — without
needing your own data:

```bash
openflo-synth --out synthetic_data   # write the full example dataset (PBMC,
                                      # FMO, comp, calibration, size beads, …)
openflo-selftest                      # run seeded data through the feature
                                      # paths and check against the baseline
```

`openflo-selftest` regenerates the seeded data, runs auto-clean, clustering,
calibration and compensation, and compares each metric to a committed golden
baseline (`src/openflo/_golden.json`):

```
✓ Auto-clean debris removed (bead 4 µm)            7.05%   (exp 7.05% ±0.25)
✓ Auto-clean dead cells removed (viability)       7.965%   (exp 7.97% ±0.6)
✓ Leiden clusters (PBMC, res 0.5)                     18    (exp 18 ±2)
✓ MESF calibration slope                          1.9978   (exp 2 ±0.05)
...
7/7 passed — behavior matches baseline.
```

A red row tells you exactly which feature drifted. After an *intended* change,
refresh the baseline with `openflo-selftest --update`. The same golden file
backs the pytest continuity tests, so the CLI and CI agree on one source of
truth. (The generated `.fcs` files are regenerable and stay out of git — only
the seeded generator is shipped.)

---

## Repo layout

```
src/openflo/
  __init__.py        public API surface (lazy re-exports)
  pipeline.py        core: FCS, compensation, gating, clustering, UMAP, WSP IO
  cli.py             CLI runner (entry point: openflo-run)
  gui.py             Tk GUI (entry point: openflo-gui)
  compare.py         OpenFlo vs FlowJo diff tool (entry point: openflo-compare)
  voltage.py         Voltage titration / Stain Index (entry point: openflo-voltage)
  synthetic.py       seeded example-dataset generator (entry point: openflo-synth)
  selftest.py        golden behavior self-test (entry point: openflo-selftest)
  _golden.json       locked baseline metrics (selftest + continuity tests)
  preview.py         ad-hoc gate-preview utility
  inspect_fcs.py     FCS header dump
  template_library/  shipped gating-template library (cleanup recipes + example panel)
tests/               pytest suite (runs on synthetic data in CI)
templates/           dev test fixture (testtemplate.json)
scripts/             dev utilities (manual smoke test, helper scripts)
docs/                algorithms reference + notes
```

---

## Algorithms

Curious what OpenFlo actually does to your data — defaults, parameter
choices, citations? See [`docs/algorithms.md`](docs/algorithms.md).
Covers logicle defaults, FMO threshold derivation, Phenograph k
choice, the spillover optimizer's heuristics, and what the pipeline
is *not* good at.

---

## Known limitations

- **Automatic gating is currently disabled.** The density-based "Auto-gate"
  button is greyed out — its heuristic placed gates unreliably. Gate manually
  (double-click the plot) or load a template / FMO-derived thresholds instead.
- UMAP / TriMap have a slow first run each session (numba JIT compilation,
  cached afterwards) — expected, not a hang.

Please file bugs on the GitHub issue tracker.

## Roadmap

- More automatic comparative tooling — cross-group / cross-condition
  comparison with less manual setup.
- A library of generic gating templates covering common panels and use cases
  (today ships a single example template).
- Broad ease-of-use improvements across the gate editor and pipeline workspace.

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Citing

If OpenFlo contributes to a publication, please cite it via the
`CITATION.cff` shown on the GitHub sidebar, and cite the upstream tools
we depend on:

- Levine *et al.* 2015 — Phenograph
- McInnes *et al.* 2018 — UMAP
- Parks *et al.* 2006 — logicle transform
- FlowJo, LLC — `.wsp` format

---

## Acknowledgements

OpenFlo was developed with substantial assistance from **Anthropic's Claude**
(Claude Code), which contributed to implementation, testing, and analysis as a
development tool. Claude is gratefully acknowledged here as a contributor — not
as an author of the software.

---

## License

OpenFlo's own code is MIT — see [LICENSE](LICENSE).

### Third-party licenses

OpenFlo depends on third-party packages under their own licenses; they are
installed via pip and are **not** redistributed in this repository. Most are
permissive (BSD / Apache-2.0 / MIT / PSF). Note that graph-based clustering
pulls in **copyleft** libraries through PhenoGraph:

| Dependency  | License            |
|-------------|--------------------|
| `igraph`    | GPL-2.0-or-later   |
| `leidenalg` | GPL-3.0            |

Your use of OpenFlo together with these libraries is subject to their terms. In
particular, if you redistribute a **bundled** build that includes them (e.g. a
packaged executable), the GPL obligations apply to that distribution. OpenFlo
itself does not import or vendor these libraries directly — they are runtime
dependencies of PhenoGraph. See each package's own license for details.

---

## Disclaimers

**Research use only.** OpenFlo is provided for research and educational use. It
is **not a medical device** and is **not intended for diagnostic, therapeutic,
or other clinical decision-making**. Independently validate any result before
relying on it. The software is provided "as is", without warranty of any kind
(see [LICENSE](LICENSE)).

**Trademarks.** "FlowJo" is a trademark of Becton, Dickinson and Company /
FlowJo, LLC. OpenFlo is an independent, unaffiliated project and is **not
sponsored, endorsed by, or associated with** them. The name is used only
descriptively, to indicate interoperability with the FlowJo `.wsp` file format.
All other trademarks are the property of their respective owners.
