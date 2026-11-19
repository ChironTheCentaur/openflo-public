# Quality control

## Samples

- **Analyze > Sample QC…** compares the samples with each other (earth mover's distance, shown on an MDS map) to find one that stands out.
- **Edit > Gates > Auto-clean gate** removes events from signal drift over time, flow-rate anomalies such as clogs and bubbles, and the channel margins.
- **Analyze > Saturation report…** shows the percent of events at the top of each channel's range, per sample.
- **Tools > Files & checks > FCS inspector…** lists a file's keywords and parameters.

## Instruments

**Tools > Panel & instrument QC > Bead tracking…** keeps a Levey-Jennings chart of QC bead runs. Load the day's bead sample, pick the bead gate as the population, and press Add active sample: the medians of every fluorescence channel become the next run. The first runs (20 by default) set each channel's mean and SD, and every run is judged by the Westgard rules: 1-3s, 2-2s, R-4s, 4-1s and 10x reject a run, and 1-2s warns. Save log… keeps the log as a CSV to add to on later days.

R-4s rejects a run whose range exceeds 4 SD, and comes in two forms. **Within a run** (Westgard's published form): pick two or more populations, one per bead peak (say dim and bright), before Add active sample, and each becomes a bead level of the run, logged as its own column (`FL1-A @ dim`) with its own mean and SD. A run where one level is beyond +2 SD and another beyond -2 SD breaks R-4s, and every level logged in that run shows it. **Across runs**: the same level beyond 2 SD on opposite sides in two runs in a row. Both forms are on by default, so a log with one bead level per run still has a range check. To use the published form only, tick **Preferences > Statistics > Westgard R-4s: literature form only (within run)**; a one-level log then never breaks R-4s. The line under the chart says which form is in use. Logs saved with one level open as before.

**Tools > Panel & instrument QC > Voltage optimization…** walks a voltage titration and recommends the voltage where the stain index levels off.

## The installation

**Help > Run diagnostics…** checks the installation, and **Help > Environment…** lists versions for a bug report.
