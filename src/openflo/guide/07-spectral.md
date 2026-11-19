# Spectral cytometry

## Unmixing

**Tools > Compensation & spectral > Spectral unmix…** builds reference spectra from loaded single-stain controls and an unstained control, then unmixes every other sample in the same group into U: channels, one per fluor. Give each control's fluor a name; it becomes the channel name.

## Several autofluorescence signatures

Cells of different types have different autofluorescence. Set "Autofluorescence signatures" above 1 to find that many autofluorescence spectra in the unstained control. Each cell is then unmixed with the signature that fits it best, and a U:AF index channel says which one.

## Spectral QC

After unmixing, the Spectral QC window opens; reopen it any time, also after reopening a saved session, with **Tools > Compensation & spectral > Spectral QC…**. It shows the similarity of every pair of spectra, the spillover spread matrix and the condition number, and lists pairs too similar to separate. Its Unmixed pairs… button draws every pair of unmixed fluors of one sample: a pair that leans or bends instead of forming a cross points at a wrong or missing spectrum.

## Panel check

**Tools > Panel & instrument QC > Panel check…** predicts the same similarity matrix and complexity index for a panel you are still designing, before any reagent is bought. Pick the generic instrument closest to yours (a 3- or 4-laser conventional cytometer, or a 5-laser spectral one with 64 detectors), search the fluor list and add fluors to the panel. The heatmap outlines every pair above the similarity threshold (0.98 on a spectral instrument, 0.90 on a conventional one); on a conventional instrument a second heatmap shows the predicted spillover of each fluor into the others' primary detectors, and the text below names fluors that would share a detector. It also says which single fluor to leave out to cut the complexity index most.

The built-in signatures are approximate: each is modeled from the fluor's excitation and emission peaks and the instrument's lasers and filters. Measured spectra are better. **Load spectra CSV…** reads a header row (Fluor, then one detector per column) and one row per fluor; **Use last unmix** takes the reference spectra measured from your single-stain controls in this session, and **Save signatures…** writes them as such a CSV to check a later panel against. Export similarity… and Export spillover… save the matrices as CSV.

## Labeling channels from vial labels

**Tools > Panel & instrument QC > Label channels from vial labels…** sets the active sample's channel labels (the marker each channel is shown and exported as) from the text printed on your antibody vials. Paste the vials' text: one vial per line, or one block per vial with a blank line between vials when a label runs over several lines. The table reads each vial's marker and fluor and picks the channel that reads the fluor, from the laser and filter the channel's name or the file's $PnL / $PnF keywords give. Change a marker or pick another channel in the table; a channel the file does not describe (FL4-A, say) is picked by hand. Vials that need you are marked: two vials on one channel (with the next-best channel offered), a fluor no channel here reads, a fluor not in the catalog, or no marker. **Apply** labels the channels of the vials without a problem, for plots, axis and gate names and statistics, and the labels are saved with the session; Undo in the main window reverses it. **Revert to the file's labels** puts back the file's own $PnS labels.
