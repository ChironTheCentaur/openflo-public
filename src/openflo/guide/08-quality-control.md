# Quality control

## Samples

- **Analyze > Sample QC…** compares the samples with each other (earth mover's distance, shown on an MDS map) to find one that stands out.
- **Edit > Auto-clean gate** removes events from signal drift over time, flow-rate anomalies such as clogs and bubbles, and the channel margins.
- **Analyze > Saturation report…** shows the percent of events at the top of each channel's range, per sample.
- **Tools > FCS inspector…** lists a file's keywords and parameters.

## Instruments

**Tools > Bead tracking…** keeps a Levey-Jennings chart of QC bead runs. Load the day's bead sample, pick the bead gate as the population, and press Add active sample: the medians of every fluorescence channel become the next run. The first runs (20 by default) set each channel's mean and SD, and every run is judged by the Westgard rules: 1-3s, 2-2s, R-4s, 4-1s and 10x reject a run, and 1-2s warns. Save log… keeps the log as a CSV to add to on later days.

**Tools > Voltage optimization…** walks a voltage titration and recommends the voltage where the stain index levels off.

## The installation

**Help > Run diagnostics…** checks the installation, and **Help > Environment…** lists versions for a bug report.
