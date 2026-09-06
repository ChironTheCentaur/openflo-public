# Gating Templates

This folder holds reusable gating templates for the OpenFlo pipeline.
Templates capture either auto-clean *recipes* (panel-agnostic, recomputed
per sample) or per-panel threshold/geometry gates, so you can apply the
same gating across experiments without re-deriving it each time.

## Bundled library

These ship with OpenFlo (the editor's **Load Template…** dialog opens here
by default). The `cleanup_*` recipes are **panel-agnostic** — they store an
auto-clean recipe with no coordinates, so they recompute on whatever sample
you apply them to and work across any panel:

| Template | What it does |
|---|---|
| `cleanup_standard.json` | Full auto-clean: debris (beads → valley), dead cells (viability dye), doublets, margin, flow-rate, signal drift. The everyday default. |
| `cleanup_minimal.json` | Debris + doublets only — fast, conservative size/singlet gating. |
| `cleanup_strict.json` | All methods with a tighter doublet window (tol 0.15) for diploid samples. |
| `cleanup_acquisition_qc.json` | Time/instrument QC only (margin + flow-rate + drift), no biology — like flowAI/PeacoQC. |
| `example_panel.json` | Example per-panel **threshold** gates (CD11b/CD34/CD45) — illustrates the threshold format; edit for your own panel. |

Drop your own curated panels alongside these (see the format below).

## Using a template

In the GUI:

1. Open **View & Gate Editor**
2. Click **Load Template…** — pick a `.json` from this folder, or a
   FlowJo `.wsp` workspace anywhere on disk
3. The threshold lines appear on the plot; adjust by dragging if needed
4. Click **Apply Gates to Pipeline** to have the next pipeline Run use
   them (overrides FMO-derived thresholds)

Save the current editor's gates as a new template by clicking
**Save Template…**.

## Native JSON template format

```json
{
  "name": "Example 3-colour panel",
  "description": "Example CD11b / CD34 / CD45 thresholds for a 3-colour panel.",
  "version": 1,
  "created": "2025-05-24T14:30:00",
  "gates": {
    "BV421-A": 0.625,
    "APC-A":           0.422,
    "PE-Cy7-A":        0.474
  },
  "labels": {
    "BV421-A": "CD11b",
    "APC-A":           "CD34",
    "PE-Cy7-A":        "CD45"
  }
}
```

`gates` is the only required field — `{detector_name: threshold_value}`.
Everything else is metadata. A bare `{ "BV421-A": 0.625, ... }`
dict also loads (legacy compatibility with the pipeline's `--gates` arg).

`labels` is optional but populates the antibody names in the editor's
channel dropdowns when the template loads.

## FlowJo `.wsp` import

The editor can also import a FlowJo workspace. It walks every
`RectangleGate` in the file and extracts the `min` value of each
dimension as a threshold for that channel. Concretely:

| Gate type in FlowJo | What we import |
|---|---|
| 1-D RectangleGate (one channel) | The `min` as a threshold on that channel |
| 2-D RectangleGate (two channels) | The `min` of each axis — two thresholds |
| PolygonGate / EllipsoidGate | Skipped (no single threshold can represent it) |
| QuadrantGate | Skipped (would need 4 threshold pairs) |
| BooleanGate | Skipped |

If the same channel is gated by several RectangleGates, the more
restrictive (higher) `min` wins. Unsupported gate types are logged so you
know what got dropped.

If your FlowJo gating relies on polygons or quadrants you'll need to
redraw those as threshold gates manually inside the View & Gate Editor.

## Committing templates

Templates are tracked in git — they're curated reference data, not
generated output. Keep a sensible filename (`example_panel.json`,
`my_panel_v2.json`) and a useful `description`.
