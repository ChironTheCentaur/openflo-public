# Getting started: a first analysis

This tutorial takes about ten minutes and needs no files of your own. It uses the synthetic example dataset, which has two groups of two donors with blood-like populations on generic channels.

## 1. Load the example data

Choose **File > Load example dataset**. Four samples appear in the panel on the left, grouped by condition. Click a sample to make it the active one; the plot shows its events.

## 2. Pick what to plot

Use the X and Y channel boxes above the plot. Start with FSC-A against SSC-A to see the cells by size and granularity. The mouse wheel zooms and a middle-drag pans; **View > Reset plot view** brings the whole plot back.

## 3. Draw your first gate

Pick **Polygon** in the Tool row under the plot, click around the main cell cloud, and double-click to close the shape. With "Apply to all shown" ticked, the gate is added to every sample on the plot.

The gate appears in the tree on the left with its count and percent of parent. Select it, change the axes to two fluorescence channels, and draw a child gate: it counts only the events inside its parent.

## 4. See the numbers

Open **Analyze > Statistics…** for counts, frequencies and medians of every gate in every sample. Each table has Copy and Export CSV buttons. **Analyze > Group comparison…** tests each population between the groups.

## 5. Save your work

**File > Save session** writes a .flowsession file with the samples, gates and view. Open it again with **File > Open session…**. To send the whole analysis to someone, use **File > Export project archive…**, which packs the session and every FCS file into one .zip.

## Where next

- Load your own files with **File > Add FCS…** or by dragging them onto the window.
- Read "Gating" and "Compensation and scales" in this guide before analysing real data.
- Press F1 at any time to open this guide, and hover over any button for a short tip.
