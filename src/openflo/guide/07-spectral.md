# Spectral cytometry

## Unmixing

**Tools > Spectral unmix…** builds reference spectra from loaded single-stain controls and an unstained control, then unmixes every other sample in the same group into U: channels, one per fluor. Give each control's fluor a name; it becomes the channel name.

## Several autofluorescence signatures

Cells of different types have different autofluorescence. Set "Autofluorescence signatures" above 1 to find that many autofluorescence spectra in the unstained control. Each cell is then unmixed with the signature that fits it best, and a U:AF index channel says which one.

## Spectral QC

After unmixing, the Spectral QC window shows the similarity of every pair of spectra, the spillover spread matrix and the condition number, and lists pairs too similar to separate. Its Unmixed pairs… button draws every pair of unmixed fluors of one sample: a pair that leans or bends instead of forming a cross points at a wrong or missing spectrum.
