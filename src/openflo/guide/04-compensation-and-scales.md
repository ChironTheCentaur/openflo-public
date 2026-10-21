# Compensation and scales

## Compensation

OpenFlo applies the spillover matrix stored in each FCS file when it loads. **Tools > Compensation…** shows and edits the matrix, imports one from a FlowJo workspace or a CSV, and can estimate one from single-stain controls (Optimize). "Apply to: this batch" applies a matrix to the samples acquired with the same instrument, date and voltages.

**Tools > Compensation QC…** shows the spillover as a heatmap and lists the strongest pairs.

## Spreading matrix

**Tools > Spreading matrix…** measures how much each compensated stain spreads the others, from compensated single-stain controls (Nguyen et al., 2013). OpenFlo matches each channel to the loaded control brightest in it; check the matches, then press Compute. A channel without a control reads NA. Large values mark channel pairs that should not carry co-expressed markers.

## Derived channels

**Tools > Derived channels…** adds a channel computed per event from others, such as the ratio of two detectors for a ratio dye: `[FL1-A] / [FL2-A]`. It is computed on linear values, plotted and gated like a measured channel, and kept in the session with its formula. An event where the result is not a number reads NA.

## Batch normalization

Data acquired on different days drift. **Tools > Normalize batches (CytoNorm)…** maps each batch's intensities onto a shared goal (Van Gassen et al., 2020), learnt from a reference sample run in every batch or from all of a batch's samples. Batches come from each sample's trial or day or from an FCS keyword such as $DATE. The result replaces the loaded samples, is written as normalized FCS copies, or both. **Tools > Undo batch normalization** restores the loaded samples.

## Display scales

Fluorescence channels are shown on a logicle scale by default. **Tools > Transforms…** sets each channel to logicle, hyperlog, arcsinh, log or linear. Statistics are always computed on the linear values, whatever the display.

## Calibration

**Tools > Calibration…** converts a channel to MESF or ABC units from a bead sample: detect the bead peaks, enter each peak's assigned value from the lot sheet, fit, and add a MESF column. The log-log fit is the standard for bead sets; the linear fit is kept for comparison.
