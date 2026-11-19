# Saving and sharing

## Sessions

**File > Save session** (Ctrl+S) writes a .flowsession with the samples' file paths, gates, groups and view. **File > Open recent** reopens recent sessions. A session saved by an older OpenFlo opens and is upgraded; **File > Upgrade saved session…** rewrites one in the current format.

When a session opens, OpenFlo lists any saved result (a re-fitted gate, a cluster or cell-cycle column, ...) made by a version with a bug fixed since, with its numbers before and after. Tick the ones to correct and Apply; nothing changes otherwise, and each original stays in the session. **Tools > Files & checks > Review fixes for this session…** shows the list again, with the fixes already applied or declined. Every export records the OpenFlo version that made it, and any fixes applied, in the file's metadata.

**File > Compare with saved session…** counts every population with this session's gates beside a saved session's gates, on the events loaded now.

**File > Export project archive…** packs the session, its data and every FCS file into one .zip that opens on another computer.

## Other programs

- **File > Export > FlowJo workspace…** (Ctrl+E) writes the gates as a FlowJo .wsp.
- **File > Export > Gating-ML 2.0 gates…** writes the gates in the standard exchange format.
- **File > Export > Gate list (CSV)…** writes every sample's gates as one table: population path, kind, channels and coordinates.
- **Tools > Files & checks > Compare FlowJo workspace…** compares OpenFlo's counts with a FlowJo workspace's, population by population.
- **File > Export > Populations (FCS)…** writes each gated population of the active sample to its own FCS file; **File > Export > Populations, all samples (FCS)…** does it for every sample, a folder per sample.
- Exported FCS files (populations, barcoded samples, normalized copies) name each channel's marker with the session's names: a channel you relabeled, for example with **Tools > Panel & instrument QC > Label channels from vial labels…**, or a template named, carries that name in every file, also after you save and reopen the session. Every other channel keeps each file's own name, so a file from a different panel keeps its names for the channels you did not relabel; "Revert to the file's labels" in that window gives the channels back their files' own names. In **Edit > Preferences…**, Export, "FCS export channel names" set to "Each file's original names" writes every file's own names instead.

## Templates

**Tools > Save template…** saves the gates and settings to apply to new data from **Tools > Templates**.

## The command line

`openflo-run` runs the whole pipeline on a folder of FCS files without the window, writing statistics, plots and a run manifest; `openflo-run --help` lists its options. `openflo-compare` compares conditions, and `openflo-selftest` checks the installation.
