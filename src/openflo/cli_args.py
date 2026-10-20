"""The ``openflo-run`` command line: its arguments, and the entry point.

Kept apart from cli.py so that ``openflo-run --help`` and a mistyped
argument are answered without loading the scientific stack: cli.py imports
numpy, pandas, matplotlib and the pipeline at the top (about 0.7 s), while
this module needs only argparse. main() checks the arguments first and loads
cli.py only to run.
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Every ``openflo-run`` option. cli.main() parses with this too."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials',      default='',
                    help='Comma-separated list of FCS trial directories')
    ap.add_argument('--fcs',         default='',
                    help='Single FCS directory (alias for --trials with one folder)')
    ap.add_argument('--out',         default='outputs', help='Output directory')
    ap.add_argument('--groups',      default='',
                    help='JSON list of groups: '
                         '\'[{"name":"Day1","samples":["d1m1","d1m2"],'
                         '"fmo_set":"standard"}, ...]\'. Replaces --samples '
                         'when given.')
    ap.add_argument('--fmo-sets',    default='', dest='fmo_sets',
                    help='JSON dict of FMO sets: '
                         '\'{"standard": {"FL1-A":"fmo_fl1", ...}}\'. '
                         'Defaults to the built-in example sets.')
    ap.add_argument('--samples',     default='',
                    help='LEGACY: comma-separated sample names; names starting '
                         'with "late" go to "Group B", others to "Group A". '
                         'Use --groups for full control.')
    ap.add_argument('--batch-mode',  default='independent',
                    choices=['independent', 'concatenate'], dest='batch_mode',
                    help='independent: one analysis per trial | '
                         'concatenate: merge trials, cells retain trial-of-origin label')
    ap.add_argument('--workers',     type=int, default=1,
                    help='Number of samples to process in parallel (default 1)')
    ap.add_argument('--cluster-method', default='phenograph',
                    choices=['phenograph', 'leiden', 'flowsom'],
                    dest='cluster_method',
                    help='Clustering algorithm (default phenograph).')
    # Reproducible is now the DEFAULT. `--reproducible` is kept so existing
    # scripts and pipelines that pass it keep working; it is simply a no-op.
    ap.add_argument('--reproducible', action='store_true',
                    help='(default since 2.4.9; kept for compatibility) Seed '
                         'PhenoGraph clustering via its Leiden backend so a '
                         're-run is bit-identical.')
    ap.add_argument('--no-reproducible', action='store_true',
                    dest='no_reproducible',
                    help="Use PhenoGraph's unseeded Louvain instead. Re-running "
                         'the same data can then give different labels, but it '
                         'reproduces prior Louvain output and re-enables the GPU '
                         'clustering path.')
    ap.add_argument('--resolution', type=float, default=1.0,
                    help='Leiden resolution — higher = more clusters '
                         '(only used with --cluster-method leiden).')
    ap.add_argument('--n-metaclusters', type=int, default=10,
                    dest='n_metaclusters',
                    help='FlowSOM metaclusters (only with '
                         '--cluster-method flowsom).')
    ap.add_argument('--max-events',  type=int, default=0, dest='max_events',
                    help='Cap clustering at this many events per sample '
                         '(0 = no cap). When exceeded, the sample is '
                         'clustered on a random sub-sample and the rest are '
                         'assigned via nearest-neighbour. Recommended: '
                         '~500000. The group comparisons cluster every '
                         'sample of an analysis together, once (once per '
                         'trial folder with --batch-mode independent), on '
                         'at most this many events IN TOTAL (an equal share '
                         'from each '
                         'sample; with 0, as many as the largest sample '
                         'holds, at most 200000 for Leiden and 50000 for '
                         'FlowSOM), and assign the rest the same way.')
    ap.add_argument('--admission-ram', type=float, default=4.0, dest='admission_gb',
                    help='Free RAM (GB) required before submitting a new '
                         'parallel worker. 0 disables admission control.')
    ap.add_argument('--admission-vram', type=float, default=1.0, dest='vram_admission_gb',
                    help='Free VRAM (GB) required for a worker to take the GPU '
                         'clustering branch (cuML+cuGraph). Below this it uses '
                         'CPU Phenograph. 0 = always try GPU when available.')
    ap.add_argument('--filter-debris-um', type=float, default=0.0,
                    dest='filter_debris_um',
                    help='Drop events whose FSC-A falls below this size in µm. '
                         '0 disables the debris filter. The conversion µm→FSC '
                         'is calibrated per trial from the comp-bead controls '
                         '(see --bead-um). Recommended: 4.0')
    ap.add_argument('--bead-um', type=float, default=8.0, dest='bead_um',
                    help='Diameter (µm) of the comp-bead control used for FSC '
                         'calibration. Default 8.0 (BD CompBeads / Invitrogen '
                         'UltraComp). Spherotech is 7.5.')
    ap.add_argument('--doublet-tol', type=float, default=0.0, dest='doublet_tol',
                    help='Doublet exclusion: keep events whose FSC-A/FSC-H is '
                         'within ±tol of the population median. 0 disables. '
                         'Recommended: 0.25 for polyploid samples, 0.15 '
                         'for typical diploid leukocytes.')
    ap.add_argument('--k',              type=int,   default=30,   help='Phenograph k')
    ap.add_argument('--fmo-percentile', type=float, default=99.5, dest='fmo_percentile')
    ap.add_argument('--labels', default='',
                    help='Channel labels: "DetA=LabelA;DetB=LabelB"')
    ap.add_argument('--panel', default='',
                    help='Path to a staining-panel .xlsx (CD↔fluorophore '
                         'rows). Resolved to {detector: CD} labels and '
                         'merged with --labels (--labels wins on conflict). '
                         'Pass "auto" to search the trial folder(s) and a '
                         'few ancestor levels for one.')
    ap.add_argument('--pairs', default='',
                    help='Marker pairs for the per-group overlay + grid '
                         'scatters, e.g. "CD902/CD901b,CD901b/CD903,CD902/CD903". '
                         'Default: those three. Labels must be assigned '
                         '(via --panel / --labels) for a pair to render.')
    ap.add_argument('--xlsx', action='store_true',
                    help='Also collect the CSV tables this run writes into '
                         'one Excel workbook, <out>/openflo_tables.xlsx, a '
                         'sheet per table (per-event exports too big for a '
                         'sheet are left out and listed).')
    ap.add_argument('--batch-correct', action='store_true', dest='batch_correct',
                    help='CytoNorm batch-normalize across batches (each '
                         'trial/day folder) before clustering.')
    ap.add_argument('--cytonorm-mode', default='goal',
                    choices=['goal', 'controls'], dest='cytonorm_mode',
                    help="'goal' (CytoNorm 2.0, control-free, default) fits on "
                         "all samples; 'controls' (classic) fits only on "
                         "per-batch control samples — see --cytonorm-control.")
    ap.add_argument('--cytonorm-control', default='', dest='cytonorm_control',
                    help="Filename token identifying the per-batch control "
                         "sample(s) for --cytonorm-mode controls (e.g. 'ref').")
    ap.add_argument('--cytonorm-metaclusters', type=int, default=10,
                    dest='cytonorm_metaclusters',
                    help='FlowSOM metaclusters for CytoNorm (default 10).')
    # ── Spectral batch-unmixing mode (full-spectrum cytometers) ──────────
    ap.add_argument('--unmix', action='store_true',
                    help='Batch spectral-unmixing mode: build reference '
                         'spectra from single-stain controls and unmix a set '
                         'of FCS into per-fluor abundances + a QC report. '
                         'Bypasses the standard analysis pipeline.')
    ap.add_argument('--unmix-controls', default='', dest='unmix_controls',
                    help='Single-stain controls for --unmix: a path to a JSON '
                         'file OR inline JSON mapping fluor→FCS path, e.g. '
                         '\'{"FL4":"ss_fl4.fcs","FL5":"ss_fl5.fcs",'
                         '"unstained":"unstained.fcs"}\'. The optional '
                         '"unstained" key adds an autofluorescence endmember.')
    ap.add_argument('--unmix-input', default='', dest='unmix_input',
                    help='FCS to unmix in --unmix mode: a directory or a '
                         'comma-separated list of files. Defaults to '
                         '--fcs / --trials.')
    ap.add_argument('--unmix-detectors', default='auto', dest='unmix_detectors',
                    help="Detector channels for --unmix: 'auto' (default — the "
                         "fluorescence channels of the first control) or a "
                         "comma-separated channel list.")
    ap.add_argument('--unmix-nonneg', action='store_true', dest='unmix_nonneg',
                    help='Clip unmixed abundances at 0 (non-negative).')
    ap.add_argument('--unmix-spectra', default='auto', dest='unmix_spectra',
                    choices=('auto', 'total', 'matched'),
                    help="How --unmix estimates each reference spectrum: "
                         "'total' (brightest events by total signal minus "
                         "the mean autofluorescence), 'matched' (brightest "
                         "by the dye's own signal, each minus the "
                         "autofluorescence of matched unstained events), or "
                         "'auto' (default: 'total' unless the two disagree "
                         "or the dye sits inside the autofluorescence band, "
                         "as for a dim dye on cells with variable "
                         "autofluorescence).")
    ap.add_argument('--unmix-beads', action='store_true', dest='unmix_beads',
                    help="The --unmix single-stain controls are beads: "
                         "estimate their spectra by total signal (no "
                         "cellular autofluorescence to match).")
    ap.add_argument('--show-plots',  action='store_true', dest='show_plots')
    ap.add_argument('--gates',       default='',
                    help='JSON list of gate dicts, e.g. '
                         '\'[{"kind":"threshold","channel":"FL1-A","value":0.5}]\'. '
                         'Fluorescence coordinates are on the logicle scale '
                         'the run applies (0.11 is zero, 1.0 is 262,144; not '
                         'intensities); '
                         'scatter and Time are linear.')
    ap.add_argument('--gate-units', default='stored', dest='gate_units',
                    choices=('stored', 'data'),
                    help="Units of --gates coordinates: 'stored' (default; "
                         "the logicle scale above) or 'data' (intensities as "
                         "in the FCS file and FlowJo, e.g. FL1-A > 1000).")
    ap.add_argument('--gates-from', default='', dest='gates_from',
                    help='Gate every sample with one population of a gate '
                         'file: a GUI session (.flowsession), a gating '
                         'template (.json), a FlowJo workspace (.wsp) or a '
                         'Gating-ML 2.0 file (.xml). The population and its '
                         'parent gates are applied, with any --gates. A '
                         "session gives its active sample's gates; a .wsp "
                         "its first gated sample's.")
    ap.add_argument('--gate-population', default='', dest='gate_population',
                    help='The population of --gates-from to gate on, by '
                         'name; needed when the file holds more than one.')
    ap.add_argument('--skip-existing', action='store_true', dest='skip_existing',
                    help='Resume an interrupted run: a sample whose '
                         '<group>/<name>_stats.csv exists is not processed '
                         'again. The group comparisons are still recomputed '
                         'on every run: a kept sample\'s events are reloaded '
                         '(not re-clustered), and the comparisons are '
                         'written only when every sample of every group is '
                         'present. A group with a kept sample keeps its '
                         'existing UMAP and pair scatters.')
    ap.add_argument('--export-wsp', default='', dest='export_wsp',
                    help='After the run, write a FlowJo-compatible .wsp '
                         'workspace at this path. Contains the gates '
                         'that were applied (FMO thresholds + --gates) '
                         'and one SampleNode per input FCS.')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed used for sample subsampling and '
                         'UMAP. Held constant across every sample in a '
                         'trial so results are reproducible across runs.')
    ap.add_argument('--palette', default='auto',
                    help='Categorical palette for per-sample / per-condition '
                         'plot colouring. Default: auto (colour-blind-safe '
                         'Okabe-Ito colours for up to 8 groups, then tab20, '
                         'then evenly spaced hues). Common alternatives: '
                         'tab10, Set1, Set2, Dark2, Paired.')
    ap.add_argument('-v', '--verbose', action='count', default=0,
                    help='Increase log verbosity (repeat for more: -v=INFO '
                         '[default], -vv=DEBUG). Mutually exclusive with -q.')
    ap.add_argument('-q', '--quiet', action='store_true',
                    help='Suppress INFO messages — only WARNING and above.')
    return ap


def build_check_sessions_parser() -> argparse.ArgumentParser:
    """``openflo-run check-sessions <folder>``."""
    ap = argparse.ArgumentParser(
        prog='openflo-run check-sessions',
        description='List, for each .flowsession under a folder, the fixes '
                    'that would apply to its saved results when it is next '
                    'opened. Read-only: no file is changed.')
    ap.add_argument('folder', help='Folder to search (sub-folders included)')
    return ap


def main(argv=None) -> int:
    """Entry point of the ``openflo-run`` console script."""
    # First, as in cli.main: a frozen Windows build's worker processes start
    # here with multiprocessing's own arguments, which argparse would reject.
    import multiprocessing
    multiprocessing.freeze_support()
    from ._console import force_utf8_streams
    force_utf8_streams()
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    # A command named first (``openflo-run check-sessions DIR``); anything
    # else is an analysis run with the options above.
    if argv and argv[0] == 'check-sessions':
        args = build_check_sessions_parser().parse_args(argv[1:])
        from .cli import run_check_sessions
        return run_check_sessions(args.folder)
    build_parser().parse_args()      # --help and bad arguments end here
    from .cli import main as run
    return run()


if __name__ == '__main__':
    raise SystemExit(main())
