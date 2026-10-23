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

## Debarcoding

Fluorescent cell barcoding (Krutzik & Nolan, 2006) and palladium mass-tag barcoding (Zunder et al., 2015) stain each sample with its own combination of barcode reagents and acquire the samples pooled in one tube. **Tools > Debarcode…** splits the pooled sample back into its samples. Pick the population to split (live single cells, say) and load the key: a CSV with a header row, then one row per sample, its name and a 0 or 1 for each barcode channel. Each key column is matched to the data channel of the same name or marker label; change the match when the names differ.

Each barcode channel is rescaled so that its 1st to 99th percentile spans 0 to 1, on the display scale (arcsinh or logicle), so each barcode channel must be negative in a good share of the samples, as in any k-of-n scheme. In a k-of-n key, an event's k brightest barcode channels are its positives, and its separation is the gap between the k-th and the next value; in a key with mixed numbers of positives the largest gap decides. The event goes to the sample with that code when the separation reaches the cutoff (0.3 by default), and is otherwise unassigned. Raising the cutoff keeps purer samples and fewer events; lowering it keeps more events, doublets among them. **Rescale per sample** repeats the assignment after scaling each sample's channels by its own positive cells (the 95th percentile), so a dimmer sample separates as well as the others. **Preview** lists each sample's yield and plots how the yields fall as the cutoff rises. **Apply** adds each sample as a population under the chosen one, named after it ("SampleA (barcode)"), with "Unassigned (barcode)" for the rest; they are kept in the session and listed in Statistics. **Save each as FCS…** also writes each sample to its own FCS file.

## Mass cytometry

A mass cytometer's signal drifts while it acquires. **Tools > Mass cytometry > Bead normalization…** follows the normalization beads spiked into each sample over Time and scales the mass channels so the beads stay level (Finck et al., 2013). Pick the bead channels; a bead is an event high on every one of them (each channel's threshold is found per sample and can be typed in), or a population you gated on the beads. Each bead channel is smoothed by a running median over a window of beads (500 by default), and at every moment one factor is fitted across the bead channels and applied to every mass channel; Time and the Gaussian parameters are left as they are. The beads can be held at each file's own level, at one level shared by all files (so runs are comparable), or at a reference file's. The window shows a bead channel before and after, and the factor over Time. The bead events are left in the data; ticking the clean-up option adds a 'Bead events' method to the sample's mass cytometry clean-up gate, so populations gated under it leave them out. **Tools > Mass cytometry > Undo bead normalization** puts the values back. A saved session keeps the factors, so a sample reopened from its FCS file is normalized again.

**Tools > Mass cytometry > Gaussian parameter clean-up…** removes events whose Event_length, Center, Offset, Width or Residual lies far from the sample's median: doublets, debris and badly fitted ion clouds (Bagwell et al., 2020). The bounds are the median plus or minus a number of robust SDs (3 by default, on arcsinh(x/5)), worked out for every sample on its own. It becomes a method of the same clean-up gate, so it is recomputed for each sample and saved with the session; gate under it to work on the clean events.

## Display scales

Fluorescence channels are shown on a logicle scale by default. **Tools > Transforms…** sets each channel to logicle, hyperlog, arcsinh, log or linear. Statistics are always computed on the linear values, whatever the display.

## Calibration

**Tools > Calibration…** converts a channel to MESF or ABC units from a bead sample: detect the bead peaks, enter each peak's assigned value from the lot sheet, fit, and add a MESF column. The log-log fit is the standard for bead sets; the linear fit is kept for comparison.
