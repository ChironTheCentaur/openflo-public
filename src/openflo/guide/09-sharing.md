# Saving and sharing

## Sessions

**File > Save session** (Ctrl+S) writes a .flowsession with the samples' file paths, gates, groups and view. **File > Open recent** reopens recent sessions. A session saved by an older OpenFlo opens and is upgraded; **File > Upgrade saved session…** rewrites one in the current format.

**File > Compare with saved session…** counts every population with this session's gates beside a saved session's gates, on the events loaded now.

**File > Export project archive…** packs the session, its data and every FCS file into one .zip that opens on another computer.

## Other programs

- **File > Export FlowJo workspace…** (Ctrl+E) writes the gates as a FlowJo .wsp.
- **File > Export Gating-ML 2.0 gates…** writes the gates in the standard exchange format.
- **File > Export gate list (CSV)…** writes every sample's gates as one table: population path, kind, channels and coordinates.
- **Tools > Compare FlowJo workspace…** compares OpenFlo's counts with a FlowJo workspace's, population by population.
- **File > Export populations (FCS)…** writes each gated population of the active sample to its own FCS file; **File > Export populations, all samples (FCS)…** does it for every sample, a folder per sample.

## Templates

**Tools > Save template…** saves the gates and settings to apply to new data from **Tools > Templates**.

## The command line

`openflo-run` runs the whole pipeline on a folder of FCS files without the window, writing statistics, plots and a run manifest; `openflo-run --help` lists its options. `openflo-compare` compares conditions, and `openflo-selftest` checks the installation.
