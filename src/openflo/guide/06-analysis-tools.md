# Analysis tools

## Clustering and embeddings

**Analyze > Cluster…** runs PhenoGraph, Leiden or FlowSOM on chosen channels, optionally with a UMAP. Clusters appear as populations you can switch on in the tree. FlowSOM starts at 20 metaclusters here, in `openflo-run --n-metaclusters` and in `FlowSample.run_flowsom`: keep it above the number of populations you expect, because asked for exactly that many it tends to merge a rare population into a neighbour; join two metaclusters that are one population with an OR gate. **Analyze > Compare embeddings…** runs several embedding methods side by side, and **Analyze > SOM tree…** draws the FlowSOM tree. **Analyze > Annotate…** labels clusters or populations.

PhenoGraph and Leiden have a **Resolution** (default 1.0, the standard result). It sets how finely cells are split: higher gives more, smaller clusters, lower merges them. PhenoGraph tends to split one population into several clusters when there are only a few markers; if that happens, lower the resolution to about 0.3–0.5 and run again. For PhenoGraph the resolution needs **Reproducible** ticked (the default): its non-reproducible Louvain method has no resolution, so the box is greyed out without it. The value used is kept in the session's history and in the Methods text. From the command line, `openflo-run --resolution 0.5` does the same for either method.

## Cell cycle

**Analyze > Cell cycle…** models DNA content. The Dean-Jett-Fox model (Fox, 1980) fits the G1 and G2/M peaks and an S phase between them, and reports their areas, so S-phase cells under the peaks count as S. The Boundaries model counts events between intensity cut-offs instead. A width channel, when present, removes doublets first. The phases become populations.

## Proliferation

**Analyze > Proliferation…** fits a dye-dilution histogram as generations one halving apart. Pick the dye channel and the population (live singlets, say), and either let OpenFlo find the undivided peak or take it from an unstimulated control. The window reports the cells in each generation and the standard indices (Roederer, 2011): percent divided, division index, proliferation index and expansion index. With fewer than 50 cells the indices read NA.

## Kinetics

**Analyze > Kinetics…** plots a channel over acquisition time for each sample, as for calcium flux: the median (or mean) in each time bin, overlaid. Each sample's baseline, peak, time to peak, area over baseline and peak over baseline are listed and export as CSV. Time is in seconds when the file gives the tick length, otherwise in ticks.

## Trajectories

**Analyze > Trajectory…** orders cells by pseudotime along a path through marker space, for differentiation series.
