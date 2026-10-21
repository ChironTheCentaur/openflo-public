# Analysis tools

## Clustering and embeddings

**Analyze > Cluster…** runs PhenoGraph, Leiden or FlowSOM on chosen channels, optionally with a UMAP. Clusters appear as populations you can switch on in the tree. **Analyze > Compare embeddings…** runs several embedding methods side by side, and **Analyze > SOM tree…** draws the FlowSOM tree. **Analyze > Annotate…** labels clusters or populations.

## Cell cycle

**Analyze > Cell cycle…** models DNA content. The Dean-Jett-Fox model (Fox, 1980) fits the G1 and G2/M peaks and an S phase between them, and reports their areas, so S-phase cells under the peaks count as S. The Boundaries model counts events between intensity cut-offs instead. A width channel, when present, removes doublets first. The phases become populations.

## Proliferation

**Analyze > Proliferation…** fits a dye-dilution histogram as generations one halving apart. Pick the dye channel and the population (live singlets, say), and either let OpenFlo find the undivided peak or take it from an unstimulated control. The window reports the cells in each generation and the standard indices (Roederer, 2011): percent divided, division index, proliferation index and expansion index. With fewer than 50 cells the indices read NA.

## Kinetics

**Analyze > Kinetics…** plots a channel over acquisition time for each sample, as for calcium flux: the median (or mean) in each time bin, overlaid. Each sample's baseline, peak, time to peak, area over baseline and peak over baseline are listed and export as CSV. Time is in seconds when the file gives the tick length, otherwise in ticks.

## Trajectories

**Analyze > Trajectory…** orders cells by pseudotime along a path through marker space, for differentiation series.
