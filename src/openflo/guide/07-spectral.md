# Spectral cytometry

## Unmixing

**Tools > Spectral unmix…** builds reference spectra from loaded single-stain controls and an unstained control, then unmixes every other sample in the same group into U: channels, one per fluor. Give each control's fluor a name; it becomes the channel name.

## Several autofluorescence signatures

Cells of different types have different autofluorescence. Set "Autofluorescence signatures" above 1 to find that many autofluorescence spectra in the unstained control. Each cell is then unmixed with the signature that fits it best, and a U:AF index channel says which one.

## Spectral QC

After unmixing, the Spectral QC window opens; reopen it any time, also after reopening a saved session, with **Tools > Spectral QC…**. It shows the similarity of every pair of spectra, the spillover spread matrix and the condition number, and lists pairs too similar to separate. Its Unmixed pairs… button draws every pair of unmixed fluors of one sample: a pair that leans or bends instead of forming a cross points at a wrong or missing spectrum.

## Panel check

**Tools > Panel check…** predicts the same similarity matrix and complexity index for a panel you are still designing, before any reagent is bought. Pick the generic instrument closest to yours (a 3- or 4-laser conventional cytometer, or a 5-laser spectral one with 64 detectors), search the fluor list and add fluors to the panel. The heatmap outlines every pair above the similarity threshold (0.98 on a spectral instrument, 0.90 on a conventional one); on a conventional instrument a second heatmap shows the predicted spillover of each fluor into the others' primary detectors, and the text below names fluors that would share a detector. It also says which single fluor to leave out to cut the complexity index most.

The built-in signatures are approximate: each is modelled from the fluor's excitation and emission peaks and the instrument's lasers and filters. Measured spectra are better. **Load spectra CSV…** reads a header row (Fluor, then one detector per column) and one row per fluor; **Use last unmix** takes the reference spectra measured from your single-stain controls in this session, and **Save signatures…** writes them as such a CSV to check a later panel against. Export similarity… and Export spillover… save the matrices as CSV.
