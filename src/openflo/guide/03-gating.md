# Gating

## Drawing gates

Pick a tool in the Tool row under the plot:

- **Rectangle** and **Ellipse**: click and drag.
- **Polygon**: click to drop corners, double-click to close.
- **Lasso**: drag a free outline and release.
- **Quadrant**: double-click to place four quadrant gates at a point.
- **Edit**: drag corners or edges, shift-drag to move a whole gate, right-click a corner to delete it.

**Auto-gate** suggests gates for the current axes (a singlet band, Gaussian-mixture ellipses or a threshold); keep, change or delete each one like a gate you drew.

## The gate tree

Gates nest: a gate drawn while another is selected becomes its child and counts only the parent's events. The tree shows each gate's count and percent of parent. Right-click a gate to set its colour, copy, cut or paste it, backgate it on the plot, or export its events as FCS. **Edit > Copy gates to…** copies the active sample's gates to others.

## Special gates

- **Edit > Add singlet gate** adds an FSC-A against FSC-H singlet gate.
- **Edit > FMO gating…** places threshold gates from fluorescence-minus-one controls at a chosen percentile.
- **Edit > Auto-clean gate** removes margin events, debris and dead cells.
- **Edit > Create boolean gate…** combines existing gates with AND, OR and NOT.
- **Edit > Combination gates…** makes every combination of 2 to 6 chosen gates as boolean populations, for polyfunctionality.
- **View > FMO on histograms** overlays the matching FMO control on each histogram.

## Exact and per-sample gates

**Edit > Edit gate coordinates…** types the selected gate's coordinates exactly. **Edit > Re-place gate on each sample…** re-fits the selected gate on each sample's own data and shows a review grid; Accept keeps the new positions.

## When a gate cannot be computed

A gate on a channel a sample does not have, or whose parent is empty, reads NA, never 0, in every table and export.

## Undo

**Edit > Undo** (Ctrl+Z) and **Edit > Redo** (Ctrl+Y) cover gate changes.
