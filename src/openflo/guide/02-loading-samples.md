# Loading and organising samples

## Adding files

- **File > Add FCS…** adds one or more FCS files (versions 2.0 to 3.2). You can also drag files or a folder onto the window.
- **File > Load CSV…** loads a table of events, such as an OpenFlo export with cluster or UMAP columns.
- **File > Quick preview…** shows a density plot of a raw file before loading it.
- **Tools > Watch folder…** loads new files as an instrument writes them.

Files load in the background; the status bar shows progress and has a Cancel button.

## Groups

Samples sit in groups (conditions). Drag a sample between groups in the panel, or use **Edit > Group samples…** to group them by an FCS keyword (such as a tube or condition keyword) or by a table. A table is a CSV or tab-separated file with a header: the first column names the sample or its file, and a column headed condition, group or trial gives its group.

## Joining and trimming

- **File > Concatenate samples…** joins samples with the same channels and scales into one new sample. Each event keeps a `__sample__` number saying where it came from.
- The Downsample box above the plot sets Off, Display only or Display + data. Display + data trims every sample to the size of the smallest, for clustering and statistics too; setting it back restores all the events of every sample whose data has not changed since.

## Plates

**Analyze > Plate view…** lays the samples out on a 96-well plate, reading each well from the well keyword or from the sample's name (for example Plate1_B07). Index-sorted files also show each sorted cell in the well it went to.
