"""``openflo-run --unmix``: spectral batch unmixing of FCS files against
single-stain controls, with its QC report (similarity and spillover spread,
Markdown + JSON) and reference-spectra plot.

Moved out of cli.py unchanged; the old cli names still work.
"""
from __future__ import annotations

import json
import os

from .cli_args import DEFAULT_DPI
from .export_stamp import background_kwargs, savefig_metadata


def load_unmix_controls(spec):
    """Resolve the ``--unmix-controls`` spec (a JSON file path or inline JSON)
    to a ``{fluor: fcs_path}`` dict (with an optional ``unstained`` key)."""
    spec = (spec or '').strip()
    if not spec:
        raise ValueError("--unmix needs --unmix-controls (fluor→FCS mapping).")
    if os.path.isfile(spec):
        with open(spec, encoding='utf-8') as f:
            mapping = json.load(f)
    else:
        mapping = json.loads(spec)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("--unmix-controls must be a non-empty JSON object "
                         "mapping fluor names to FCS paths.")
    return {str(k): str(v) for k, v in mapping.items()}


def run_batch_unmix(args) -> int:
    """Spectral batch-unmixing CLI mode. Builds reference spectra from the
    single-stain controls, unmixes each input FCS into per-fluor abundance
    columns (written as CSV), and writes a QC report (similarity + spillover
    spread, Markdown + JSON) plus a reference-spectra plot. Returns 0 on
    success, non-zero on a setup error."""
    from .spectral import build_reference_spectra, unmixing_qc

    out_dir = args.out or 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    try:
        controls = load_unmix_controls(args.unmix_controls)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"[unmix] bad --unmix-controls: {exc}", flush=True)
        return 2

    unstained_path = controls.pop('unstained', None)
    if not controls:
        print("[unmix] no single-stain controls given (only 'unstained').",
              flush=True)
        return 2

    single_samples = _load_single_stains(controls, args)
    if single_samples is None:
        return 2
    first = next(iter(single_samples.values()))

    detectors = _resolve_detectors(args, first)
    if len(detectors) < 2:
        print("[unmix] need ≥2 detector channels (got "
              f"{len(detectors)}). Use --unmix-detectors.", flush=True)
        return 2

    stains = _measure_stains(single_samples, detectors)
    if len(stains) < 1:
        print("[unmix] no usable single-stain controls.", flush=True)
        return 2

    un = _measure_unstained(unstained_path, detectors, args)

    spectra_diag = []
    spectra, fluors = build_reference_spectra(
        stains, unstained=un, detectors=detectors, diagnostics=spectra_diag,
        method=getattr(args, 'unmix_spectra', None) or 'auto',
        beads=bool(getattr(args, 'unmix_beads', False)))
    print(f"[unmix] built {len(fluors)} reference spectra over "
          f"{len(detectors)} detectors.", flush=True)
    _report_spectra(spectra_diag)

    inputs = _resolve_inputs(args, controls, unstained_path)

    qc_stains = dict(stains)
    if un is not None and 'Autofluorescence' in fluors:
        qc_stains['Autofluorescence'] = un
    qc = unmixing_qc(qc_stains, spectra, fluors, unstained=un,
                     reference_spectra=spectra_diag)

    # Write the QC report (Markdown + JSON) and the reference-spectra plot.
    write_unmix_qc(out_dir, qc, spectra, fluors, detectors,
                    dpi=getattr(args, 'dpi', DEFAULT_DPI))

    n_done = 0
    for path in inputs:
        try:
            n_done += _unmix_one(path, args, detectors, spectra, fluors,
                                 out_dir)
        except Exception as exc:  # noqa: BLE001 - reported; the batch goes on
            print(f"[unmix] {os.path.basename(path)}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    _print_summary(qc, n_done, out_dir, inputs)
    return 0


# Load the single-stain controls and inputs: acquisition QC (with --qc),
# then the detectors as measured (linear, never compensated -- spectral
# data is unmixed instead). This read `s.raw`, which run_qc did not
# filter, so QC had no effect on anything unmixed here.
def _load_raw(path, args):
    from .pipeline import FlowSample

    s = FlowSample(path)
    if getattr(args, 'qc', False):
        s.run_qc()
    return s


def _load_single_stains(controls, args):
    """Load each control in order; None (after saying which) when one is
    missing."""
    single_samples = {}
    for fluor, path in controls.items():
        if not os.path.isfile(path):
            print(f"[unmix] control for {fluor!r} not found: {path}",
                  flush=True)
            return None
        single_samples[fluor] = _load_raw(path, args)
    return single_samples


def _resolve_detectors(args, first):
    """Detectors: 'auto' → the first control's fluorescence channels."""
    if args.unmix_detectors and args.unmix_detectors != 'auto':
        return [d.strip() for d in args.unmix_detectors.split(',')
                if d.strip()]
    return list(getattr(first, 'fluor_channels', []) or [])


def _has_detectors(s, detectors):
    return all(d in s.data.columns for d in detectors)


def _measure_stains(single_samples, detectors):
    """Each control's measured detectors; a control missing any is skipped."""
    from .spectral import measured_signal

    stains = {}
    for fluor, s in single_samples.items():
        if not _has_detectors(s, detectors):
            print(f"[unmix] control {fluor!r} is missing some detectors — "
                  "skipped.", flush=True)
            continue
        stains[fluor] = measured_signal(s, detectors)
    return stains


def _measure_unstained(unstained_path, detectors, args):
    """The unstained control's measured detectors, or None when it is not
    given, not found, or missing a detector."""
    from .spectral import measured_signal

    if not (unstained_path and os.path.isfile(unstained_path)):
        return None
    us = _load_raw(unstained_path, args)
    if not _has_detectors(us, detectors):
        return None
    return measured_signal(us, detectors)


def _report_spectra(spectra_diag):
    for d in spectra_diag:
        if d['method'] == 'matched':
            print(f"[unmix] {d['fluor']}: spectrum estimated against matched "
                  f"autofluorescence ({d['reason']}).", flush=True)
        if d['warning']:
            print(f"[unmix] [!] {d['fluor']} is too dim to define a spectrum "
                  f"reliably: {d['warning']}. Use beads or a brighter fluor.",
                  flush=True)


def _resolve_inputs(args, controls, unstained_path):
    """The FCS files to unmix: each comma-separated token is a folder (its
    *.fcs, sorted) or a file; the control files themselves are left out."""
    import glob

    raw_in = args.unmix_input or args.fcs or args.trials or ''
    inputs = []
    for tok in raw_in.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if os.path.isdir(tok):
            inputs.extend(sorted(glob.glob(os.path.join(tok, '*.fcs'))))
        elif os.path.isfile(tok):
            inputs.append(tok)
    control_paths = {os.path.abspath(p) for p in controls.values()}
    if unstained_path:
        control_paths.add(os.path.abspath(unstained_path))
    return [p for p in inputs if os.path.abspath(p) not in control_paths]


def _unmix_one(path, args, detectors, spectra, fluors, out_dir):
    """Unmix one input to <stem>_unmixed.csv; 1 when written, 0 when it lacks
    a detector (skipped, with a note)."""
    import pandas as pd

    from .spectral import measured_signal, unmix
    from .transform_sidecar import (
        discard_transforms_sidecar,
        write_transforms_sidecar,
    )

    s = _load_raw(path, args)
    if not _has_detectors(s, detectors):
        print(f"[unmix] {os.path.basename(path)}: missing detectors — "
              "skipped.", flush=True)
        return 0
    A = unmix(measured_signal(s, detectors), spectra,
              nonneg=bool(args.unmix_nonneg))
    stem = os.path.splitext(os.path.basename(path))[0]
    out_csv = os.path.join(out_dir, f'{stem}_unmixed.csv')
    ucols = [f'U:{f}' for f in fluors]
    out = pd.DataFrame(A, columns=pd.Index(ucols))
    # Unmixed from the measured (linear) detectors, so every column is
    # linear; say so, so File > Load CSV does not have to guess (an
    # earlier record goes first, so a failed write can't leave it
    # beside new data).
    discard_transforms_sidecar(out_csv)
    out.to_csv(out_csv, index=False)
    write_transforms_sidecar(out_csv, {}, ucols, len(out))
    return 1


def _print_summary(qc, n_done, out_dir, inputs):
    cond = qc['condition_number']
    cond_txt = "inf" if cond == float('inf') else f"{cond:.1f}"
    print(f"[unmix] done: {n_done} sample(s) unmixed → {out_dir} "
          f"(condition number {cond_txt}, "
          f"{len(qc['similar_pairs'])} similar pair(s)).", flush=True)
    if not inputs:
        print("[unmix] (no input FCS to unmix — built spectra + QC only; "
              "pass --unmix-input).", flush=True)


def write_unmix_qc(out_dir, qc, spectra, fluors, detectors,
                    dpi=DEFAULT_DPI):
    """Write the spectral-QC Markdown + JSON report and a reference-spectra
    PNG into ``out_dir``."""
    import numpy as np

    cond = qc['condition_number']
    cond_txt = "inf" if cond == float('inf') else f"{cond:.2f}"
    md = ["# Spectral unmixing QC", "",
          f"- **fluors**: {len(fluors)}",
          f"- **detectors**: {len(detectors)}",
          f"- **condition_number**: {cond_txt}", ""]
    # A fluor with no usable spectrum is never among the similar pairs, so
    # the report would otherwise read as a well-separated panel.
    deg = list(qc.get('degenerate_fluors') or [])
    if deg:
        md += ["## Unusable reference spectra",
               "No usable spectrum (all zero, e.g. a stain no brighter than "
               "the unstained control), so these cannot be unmixed: "
               + ', '.join(deg), ""]
    md.append("## Spectrally-similar pairs")
    if qc['similar_pairs']:
        md += ["| Fluor A | Fluor B | Cosine similarity |", "|---|---|---|"]
        md += [f"| {d['fluor_a']} | {d['fluor_b']} | {d['similarity']:.4f} |"
               for d in qc['similar_pairs']]
    else:
        md.append("None above threshold.")
    md += ["", "## Largest spillover spread"]
    if qc['worst_spread']:
        md += ["| Into | From | Spread |", "|---|---|---|"]
        md += [f"| {d['into']} | {d['from']} | {d['spread']:.4g} |"
               for d in qc['worst_spread']]
    else:
        md.append("No measured spread (single-stain controls missing).")
    # How each spectrum was estimated, and which controls are too dim to
    # trust -- a contaminated spectrum looks like any other in the plot.
    from .spectral import reference_spectra_markdown
    ref_report = list(qc.get('reference_spectra') or [])
    if ref_report:
        md += [""] + reference_spectra_markdown(ref_report)
    with open(os.path.join(out_dir, 'spectral_qc.md'), 'w',
              encoding='utf-8') as f:
        f.write("\n".join(md) + "\n")

    payload = {
        'format': 'openflo-spectral-qc', 'version': 1,
        'fluors': list(fluors), 'detectors': list(detectors),
        'condition_number': (None if cond == float('inf') else cond),
        'similarity': np.asarray(qc['similarity']).tolist(),
        'ssm': [[None if not np.isfinite(v) else float(v) for v in row]
                for row in np.asarray(qc['ssm'])],
        # The usual SSM table has the single stain on the rows; this one is
        # the transpose, so say which index is which.
        'ssm_orientation': 'ssm[i][j] = spread into fluors[i] from the '
                           'single stain fluors[j]',
        'similar_pairs': qc['similar_pairs'],
        'worst_spread': qc['worst_spread'],
        'degenerate_fluors': deg,
        'reference_spectra': ref_report,
        'dim_controls': [d['fluor'] for d in ref_report if d['warning']],
    }
    with open(os.path.join(out_dir, 'spectral_qc.json'), 'w',
              encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, fl in enumerate(fluors):
            ax.plot(range(spectra.shape[1]), spectra[i], marker='o', ms=2,
                    lw=1.2, label=fl)
        ax.set_xlabel('detector')
        ax.set_ylabel('normalized signal')
        ax.set_title('Reference spectra')
        ax.legend(fontsize=7, loc='best')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'reference_spectra.png'), dpi=dpi,
                    metadata=savefig_metadata(fmt='png'),
                    **background_kwargs(fig, fmt='png'))
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - optional plot, reported
        print(f"[unmix] spectra plot skipped: {exc}", flush=True)
