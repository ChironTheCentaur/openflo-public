"""Panel check: predicted spectral similarity and complexity of a panel
before it is stained (Tools > Panel check, ui_panel_check.py).

Spectral QC (ui_spectral_qc.py) reports the similarity matrix and the
condition number of spectra MEASURED from single-stain controls. Designing a
panel, the same numbers are wanted from reference signatures instead:

  * from a spectra CSV (rows = fluors, columns = detectors), such as one
    saved from the reference spectra of an earlier unmix (``write_spectra_csv``
    / ``load_spectra_csv``), or
  * from an approximate model (``model_signature``): each catalogued fluor's
    excitation and emission peaks (fluor_catalog.FLUOR_SPECTRA), emission as
    a skewed Gaussian, excitation efficiency read at each laser, integrated
    over the bandpass of every detector of a generic cytometer
    (``DETECTOR_SETS``, wavelengths only).

The similarity, condition number, similar pairs and unusable signatures come
from spectral.unmixing_qc itself, so a predicted panel and a measured one are
scored by the same code. Pure: numpy only, no Tk.
"""
from __future__ import annotations

import csv
import io
import math

import numpy as np

from .spectral import _normalize, spectral_condition_number, unmixing_qc

# A detector: (name, laser nm, bandpass centre nm, bandpass width nm).
Detector = tuple[str, float, float, float]


def _spectral_bands(prefix, laser, start, stop, n):
    """``n`` contiguous bands from `start` to `stop` nm behind one laser."""
    edges = np.linspace(start, stop, n + 1)
    return tuple((f"{prefix}{k + 1}", float(laser),
                  round(float(edges[k] + edges[k + 1]) / 2, 1),
                  round(float(edges[k + 1] - edges[k]), 1))
                 for k in range(n))


def _filters(prefix, laser, bands):
    return tuple((f"{prefix}{c:g}/{w:g}", float(laser), float(c), float(w))
                 for c, w in bands)


# Generic instruments, by wavelengths only. 'kind' sets the default
# similarity threshold and whether the predicted spillover is meaningful.
DETECTOR_SETS: dict[str, dict] = {
    '3-laser conventional (405/488/640 nm)': {
        'kind': 'conventional',
        'detectors': (
            _filters('V', 405, ((450, 50), (525, 50), (610, 20), (660, 20),
                                (710, 50), (780, 60)))
            + _filters('B', 488, ((530, 30), (585, 42), (610, 20),
                                  (695, 40), (780, 60)))
            + _filters('R', 640, ((670, 30), (730, 45), (780, 60))))},
    '4-laser conventional (405/488/561/640 nm)': {
        'kind': 'conventional',
        'detectors': (
            _filters('V', 405, ((450, 50), (525, 50), (610, 20), (660, 20),
                                (710, 50), (780, 60)))
            + _filters('B', 488, ((530, 30), (695, 40)))
            + _filters('YG', 561, ((586, 15), (610, 20), (670, 30),
                                   (710, 50), (780, 60)))
            + _filters('R', 640, ((670, 30), (730, 45), (780, 60))))},
    '5-laser spectral, 64 detectors (355/405/488/561/640 nm)': {
        'kind': 'spectral',
        'detectors': (_spectral_bands('UV', 355, 370, 830, 16)
                      + _spectral_bands('V', 405, 420, 830, 16)
                      + _spectral_bands('B', 488, 500, 830, 14)
                      + _spectral_bands('YG', 561, 575, 830, 10)
                      + _spectral_bands('R', 640, 655, 830, 8))},
}
DEFAULT_SET = '5-laser spectral, 64 detectors (355/405/488/561/640 nm)'

# Above this cosine similarity two signatures are flagged as hard to tell
# apart. 0.98 is the usual spectral rule (and Spectral QC's). A conventional
# instrument sees each fluor in a few filters only, so its signatures share
# less; pairs above 0.90 there already crowd one another's detector.
DEFAULT_THRESHOLD = {'spectral': 0.98, 'conventional': 0.90}

# A modelled fluor whose best detector collects less than this fraction of
# what its own laser and an ideal filter would is not detected by the set.
MIN_SIGNAL = 0.01

# Emission is skewed to the red: its red half-width is this many times its
# blue one. Excitation is skewed to the blue the same way.
EMISSION_SKEW = 1.5
EXCITATION_SKEW = 1.0 / 1.8

_GRID = np.arange(250.0, 950.5, 0.5)            # nm
_FWHM_TO_SD = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def skewed_gaussian(wl, peak, fwhm, skew):
    """Split Gaussian with its maximum (1) at `peak`: total width at half
    height `fwhm`, the red half `skew` times as wide as the blue one."""
    wl = np.asarray(wl, dtype=float)
    blue = fwhm / (1.0 + skew)
    sd = np.where(wl < peak, blue, blue * skew) * 2.0 * _FWHM_TO_SD
    return np.exp(-0.5 * ((wl - peak) / sd) ** 2)


def _entry(name):
    from .fluor_catalog import FLUOR_SPECTRA
    try:
        return FLUOR_SPECTRA[name]
    except KeyError:
        raise KeyError(f"No spectrum model for {name!r}.") from None


def emission_curve(entry, wl=_GRID):
    """The modelled emission spectrum of a catalogue entry on `wl`."""
    _ex, _exw, em, emw, extra = entry
    out = skewed_gaussian(wl, em, emw, EMISSION_SKEW)
    for pk, h, w in extra:
        out = out + h * skewed_gaussian(wl, pk, w, EMISSION_SKEW)
    return out


def excitation_efficiency(entry, laser):
    """How well `laser` (nm) excites a catalogue entry, 0-1 (1 at the top of
    its modelled excitation spectrum)."""
    ex, exw, _em, _emw, _extra = entry

    def curve(wl):
        out = np.zeros(np.shape(wl), dtype=float)
        for pk, h in ex:
            out = out + h * skewed_gaussian(wl, pk, exw, EXCITATION_SKEW)
        return out
    top = float(np.max(curve(_GRID)))
    return float(curve(np.array([float(laser)]))[0]) / top if top > 0 else 0.0


def model_signature(name, detectors):
    """Modelled signature of the catalogued fluor `name` on `detectors`
    (``Detector`` tuples), max-normalised as spectral.build_reference_spectra
    normalises measured spectra. All zero when no detector collects
    ``MIN_SIGNAL`` of it (no laser excites it, or no filter sees it)."""
    entry = _entry(name)
    em = emission_curve(entry)
    total = float(np.trapezoid(em, _GRID))
    raw = np.zeros(len(detectors), dtype=float)
    for k, (_nm, laser, centre, width) in enumerate(detectors):
        lo, hi = centre - width / 2.0, centre + width / 2.0
        inband = (_GRID >= lo) & (_GRID <= hi)
        if inband.sum() < 2:
            continue
        frac = float(np.trapezoid(em[inband], _GRID[inband])) / total
        raw[k] = excitation_efficiency(entry, laser) * frac
    if raw.max() < MIN_SIGNAL * _best_band_fraction(em, detectors):
        return np.zeros(len(detectors), dtype=float)
    return _normalize(raw, 'max')


def _best_band_fraction(em, detectors):
    """The fraction of `em` the widest band of the set would take at its
    peak: the yardstick ``MIN_SIGNAL`` is measured against."""
    width = max((float(d[3]) for d in detectors), default=0.0)
    pk = float(_GRID[int(np.argmax(em))])
    inband = (_GRID >= pk - width / 2) & (_GRID <= pk + width / 2)
    if inband.sum() < 2:
        return 1.0
    return (float(np.trapezoid(em[inband], _GRID[inband]))
            / float(np.trapezoid(em, _GRID)))


def catalog_fluors():
    """Every fluor the model knows, in catalogue order."""
    from .fluor_catalog import FLUOR_SPECTRA
    return list(FLUOR_SPECTRA)


def signature_matrix(fluors, detectors):
    """``(n_fluors, n_detectors)`` modelled signatures of `fluors`."""
    return np.vstack([model_signature(f, detectors) for f in fluors]) \
        if fluors else np.zeros((0, len(detectors)))


# ── Spectra CSV ──────────────────────────────────────────────────────────────

class SpectraCSVError(ValueError):
    """A spectra CSV that cannot be used, with what is wrong and where."""


def parse_spectra_csv(text):
    """``(S, fluors, detectors)`` from spectra CSV `text`: a header row
    (first cell the fluor column's title, then one detector per column) and
    one row per fluor of non-negative numbers. Rows are max-normalised.
    Comma, semicolon or tab separated. Raises ``SpectraCSVError``."""
    text = text.lstrip('﻿')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise SpectraCSVError("The file is empty.")
    delim = max((',', ';', '\t'), key=lines[0].count)
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    numbered = [(i + 1, r) for i, r in enumerate(rows)
                if any(c.strip() for c in r)]
    (head_no, head), body = numbered[0], numbered[1:]
    detectors = [c.strip() for c in head[1:]]
    if len(detectors) < 2:
        raise SpectraCSVError(
            f"Line {head_no}: the header needs a fluor column and at least "
            "two detector columns.")
    if any(not d for d in detectors):
        raise SpectraCSVError(f"Line {head_no}: a detector column has no "
                              "name.")
    dup = sorted({d for d in detectors if detectors.count(d) > 1})
    if dup:
        raise SpectraCSVError(f"Line {head_no}: detector named twice: "
                              + ', '.join(dup) + ".")
    if not body:
        raise SpectraCSVError("The file has a header but no fluor rows.")
    fluors, out = [], []
    for no, r in body:
        if len(r) != len(head):
            raise SpectraCSVError(
                f"Line {no}: {len(r)} cells, but the header has "
                f"{len(head)}.")
        name = r[0].strip()
        if not name:
            raise SpectraCSVError(f"Line {no}: the fluor has no name.")
        if name in fluors:
            raise SpectraCSVError(f"Line {no}: {name} is listed twice.")
        vals = []
        for det, cell in zip(detectors, r[1:], strict=True):
            try:
                v = float(cell.strip())
            except ValueError:
                raise SpectraCSVError(
                    f"Line {no}, {det}: {cell.strip()!r} is not a "
                    "number.") from None
            if not math.isfinite(v):
                raise SpectraCSVError(f"Line {no}, {det}: not a finite "
                                      "number.")
            if v < 0:
                raise SpectraCSVError(f"Line {no}, {det}: negative value "
                                      f"{v:g}; a signature is >= 0.")
            vals.append(v)
        if not max(vals) > 0:
            raise SpectraCSVError(f"Line {no}: {name} is zero in every "
                                  "detector.")
        fluors.append(name)
        out.append(_normalize(np.array(vals), 'max'))
    return np.vstack(out), fluors, detectors


def load_spectra_csv(path):
    """``parse_spectra_csv`` on the file at `path`."""
    with open(path, encoding='utf-8-sig', newline='') as f:
        return parse_spectra_csv(f.read())


def spectra_csv_rows(S, fluors, detectors):
    """CSV rows of signatures `S` that ``parse_spectra_csv`` reads back."""
    S = np.asarray(S, dtype=float)
    return ([['Fluor', *detectors]]
            + [[f, *(f"{v:.6g}" for v in row)]
               for f, row in zip(fluors, S, strict=True)])


def write_spectra_csv(path, S, fluors, detectors):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerows(spectra_csv_rows(S, fluors, detectors))


# ── The check ────────────────────────────────────────────────────────────────

def predicted_spillover(S):
    """``(spill, primary)``: each fluor's primary detector (its brightest,
    ``-1`` for an all-zero signature) and ``spill[i, j]``, the signal of
    fluor i in fluor j's primary detector as a fraction of fluor i's signal
    in its own (a compensation matrix's rows; diagonal 1). NaN for an
    all-zero signature."""
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    top = S.max(axis=1) if S.size else np.zeros(n)
    primary = np.where(top > 0, np.argmax(S, axis=1) if S.size else 0, -1)
    spill = np.full((n, n), np.nan)
    for i in range(n):
        if top[i] <= 0:
            continue
        for j in range(n):
            if primary[j] >= 0:
                spill[i, j] = S[i, primary[j]] / top[i]
    return spill, primary


def swap_ranking(S, fluors):
    """Which one fluor to leave out to cut the complexity index most:
    ``[{'fluor', 'condition_number'}]``, the condition number of the panel
    without it, lowest first. Empty for fewer than three fluors."""
    S = np.asarray(S, dtype=float)
    if len(fluors) < 3:
        return []
    out = [{'fluor': f, 'condition_number':
            spectral_condition_number(np.delete(S, i, axis=0))}
           for i, f in enumerate(fluors)]
    out.sort(key=lambda d: d['condition_number'])
    return out


def check_panel(S, fluors, detectors, threshold=0.98):
    """Score signatures `S` (``(n_fluors, n_detectors)``) as Spectral QC
    scores measured spectra (spectral.unmixing_qc), plus the predicted
    spillover and the fluors sharing a primary detector.

    Returns ``{fluors, detectors, signatures, threshold, similarity,
    condition_number, similar_pairs, degenerate_fluors, spillover,
    primary_detector, shared_primary, swap}``."""
    S = np.asarray(S, dtype=float)
    fluors, detectors = list(fluors), list(detectors)
    if S.ndim != 2 or S.shape != (len(fluors), len(detectors)):
        raise ValueError(f"signatures are {S.shape}, expected "
                         f"{(len(fluors), len(detectors))}")
    qc = unmixing_qc({}, S, fluors, sim_threshold=float(threshold))
    spill, primary = predicted_spillover(S)
    prim_names = [detectors[k] if k >= 0 else None for k in primary]
    shared = [(fluors[i], fluors[j], prim_names[i])
              for i in range(len(fluors)) for j in range(i + 1, len(fluors))
              if primary[i] >= 0 and primary[i] == primary[j]]
    return {'fluors': fluors, 'detectors': detectors, 'signatures': S,
            'threshold': float(threshold),
            'similarity': qc['similarity'],
            'condition_number': qc['condition_number'],
            'similar_pairs': qc['similar_pairs'],
            'degenerate_fluors': qc['degenerate_fluors'],
            'spillover': spill, 'primary_detector': prim_names,
            'shared_primary': shared, 'swap': swap_ranking(S, fluors)}


def matrix_csv_rows(report, which='similarity'):
    """CSV rows of the report's ``'similarity'`` or ``'spillover'`` matrix,
    fluors on both axes (spillover: row = fluor, column = the detector of
    the column's fluor); NA for undefined cells."""
    M = np.asarray(report[which], dtype=float)
    fl = report['fluors']
    return ([['', *fl]]
            + [[f, *(f"{v:.6g}" if np.isfinite(v) else 'NA' for v in row)]
               for f, row in zip(fl, M, strict=True)])
