"""Batch normalization across acquisition days on top of :class:`CytoNorm`.

Headless helpers (no Tk) that a dialog or a script calls around the existing
``pipeline.CytoNorm`` model (Van Gassen et al. 2020, Cytometry A; CytoNorm
2.0 'goal' mode or the classic reference-sample 'controls' mode). Nothing
here re-implements the normalization itself:

* :func:`keyword_value` / :func:`group_by_keyword` -- assign samples to
  batches by an FCS TEXT keyword such as ``$DATE`` (read from the
  ``metadata`` the loader keeps, any case, with or without the ``$``).
* :func:`fit_batch_model` -- fit a CytoNorm from in-memory sample frames.
  In 'controls' mode only each batch's reference sample(s) are pooled (the
  same reference material run in every batch); in 'goal' mode every sample
  of a batch is pooled.
* :func:`apply_batch_model` -- corrected NEW frames; inputs never mutated.
* :func:`normalization_summary` / :func:`methods_text` -- per channel and
  batch, the median and robust spread before vs after, and a one-line
  methods-report sentence.
* :func:`load_samples` / :func:`write_normalized_copies` -- read FCS files
  the way ``openflo-run --batch-correct`` does and write normalized FCS
  copies with provenance keywords into an output folder.

Which values are normalized. CytoNorm works on compensated (or unmixed),
transformed intensities: the batch effect it models is a per-marker shift
and stretch of the fluorescence of a cell population, and spillover would
otherwise mix one detector's batch effect into another's. :func:`load_samples`
therefore applies the file's own spillover matrix and the logicle transform
(the ``--batch-correct`` pipeline does the same), and the frames a caller
passes in are expected to be on that scale (the editor's ``sample.data``).
The written copies hold compensated LINEAR values -- the display transform
undone -- with the applied matrix recorded in OpenFlo's own keywords and no
spillover keyword, exactly as an exported population (``fcs_export``), so
no reader compensates them a second time.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    'MISSING_BATCH', 'keyword_value', 'group_by_keyword', 'batch_of_samples',
    'fit_batch_model', 'apply_batch_model', 'robust_spread',
    'normalization_summary', 'methods_text', 'load_samples',
    'provenance_keywords', 'write_normalized_copies',
]

# The batch a sample lands in when it lacks the grouping keyword.
MISSING_BATCH = '(no value)'

# Half the central 68.27 % range: the SD of a normal distribution, but
# insensitive to the tails (FlowJo's robust SD).
_P_LO, _P_HI = 15.865, 84.135


# ── Batches from FCS keywords ────────────────────────────────────────────────

def _key(k: object) -> str:
    return str(k).strip().lstrip('$').lower()


def keyword_value(metadata: Mapping[str, Any] | None, keyword: str) -> str | None:
    """The value of FCS keyword `keyword` in `metadata`, or None.

    Matched case-insensitively and with or without a leading ``$``: the
    loader keeps FlowIO's lower-case, ``$``-less keys (``date``), while a
    user types ``$DATE``. Surrounding whitespace is stripped; an empty value
    is None."""
    want = _key(keyword)
    for k, v in (metadata or {}).items():
        if _key(k) == want:
            text = '' if v is None else str(v).strip()
            return text or None
    return None


def group_by_keyword(samples: Mapping[str, Any], keyword: str,
                     missing: str = MISSING_BATCH) -> dict[str, list[str]]:
    """``{batch: [sample name, ...]}`` grouping `samples` (``{name: sample}``,
    each with a ``metadata`` dict, e.g. a ``FlowSample``) by the value of
    `keyword`. Samples without it go to the `missing` batch. Batches and
    their samples keep the order of `samples`."""
    out: dict[str, list[str]] = {}
    for name, s in samples.items():
        md = s if isinstance(s, Mapping) else getattr(s, 'metadata', None)
        val = keyword_value(md, keyword)
        out.setdefault(val if val is not None else missing, []).append(str(name))
    return out


def batch_of_samples(groups: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """Invert ``{batch: [names]}`` into ``{name: batch}``. A name in two
    batches is refused (it could be normalized only one way)."""
    out: dict[str, str] = {}
    for batch, names in groups.items():
        for n in names:
            if n in out and out[n] != batch:
                raise ValueError(
                    f"sample {n!r} is in batch {out[n]!r} and {batch!r}")
            out[str(n)] = str(batch)
    return out


# ── Fit / apply ──────────────────────────────────────────────────────────────

def _as_list(v: str | Sequence[str] | None) -> list[str]:
    if v is None:
        return []
    return [v] if isinstance(v, str) else [str(x) for x in v]


def _check_channels(name: str, df: pd.DataFrame, channels: Sequence[str]) -> None:
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError(f"sample {name!r} lacks channel(s) {missing}")


def fit_batch_model(frames: Mapping[str, pd.DataFrame],
                    batch_of: Mapping[str, str],
                    channels: Sequence[str],
                    mode: str = 'goal',
                    references: Mapping[str, str | Sequence[str]] | None = None,
                    n_metaclusters: int = 10,
                    grid: tuple[int, int] = (10, 10),
                    n_quantiles: int = 101,
                    max_events_per_batch: int = 20_000,
                    seed: int = 42):
    """Fit a :class:`openflo.pipeline.CytoNorm` across batches.

    `frames` maps a sample name to its events (compensated, transformed;
    see the module docstring) and `batch_of` a sample name to its batch.
    'goal' mode pools every sample of a batch; 'controls' mode pools only
    the sample(s) ``references[batch]`` names, and every batch must have
    one -- a batch without its reference has nothing to anchor its mapping.
    ``n_metaclusters=1`` is the single-cluster form (one per-channel
    quantile mapping per batch); more fits one per FlowSOM metacluster.
    Each batch is subsampled to `max_events_per_batch` events (0 = all)
    with `seed`, so a re-fit is identical. Needs at least two batches."""
    from .pipeline import CytoNorm
    if mode not in ('goal', 'controls'):
        raise ValueError(f"mode must be 'goal' or 'controls', got {mode!r}")
    channels = [str(c) for c in channels]
    if not channels:
        raise ValueError("no channels to normalize")
    pick: dict[str, list[str]] = {}
    for name in frames:
        if name not in batch_of:
            raise ValueError(f"sample {name!r} has no batch")
        pick.setdefault(batch_of[name], [])
    if mode == 'controls':
        refs = references or {}
        for batch in pick:
            names = _as_list(refs.get(batch))
            if not names:
                raise ValueError(
                    f"batch {batch!r} has no reference sample; 'controls' "
                    "mode needs the reference run in every batch")
            for n in names:
                if n not in frames:
                    raise ValueError(
                        f"reference {n!r} of batch {batch!r} is not loaded")
                if batch_of.get(n) != batch:
                    raise ValueError(
                        f"reference {n!r} is not in batch {batch!r}")
            pick[batch] = names
    else:
        for name in frames:
            pick[batch_of[name]].append(name)
    if len(pick) < 2:
        raise ValueError(
            f"batch normalization needs at least 2 batches, got {len(pick)}")
    pooled: dict[str, pd.DataFrame] = {}
    for batch, names in pick.items():
        for n in names:
            _check_channels(n, frames[n], channels)
        pooled[batch] = pd.DataFrame(pd.concat(
            [frames[n][channels] for n in names], ignore_index=True))
    # CytoNorm.fit draws the per-batch subsample (cap_events_per_batch).
    return CytoNorm(channels, n_metaclusters=n_metaclusters, grid=grid,
                    n_quantiles=n_quantiles, mode=mode, seed=seed).fit(
                        pooled, max_events_per_batch=max_events_per_batch)


def apply_batch_model(model, frames: Mapping[str, pd.DataFrame],
                      batch_of: Mapping[str, str]) -> dict[str, pd.DataFrame]:
    """``{name: corrected copy}`` for every frame, each with its batch's
    mapping. The input frames are never changed; columns outside the model
    pass through. A sample whose batch the model was not fitted on, or that
    lacks one of its channels, is refused (CytoNorm itself would hand the
    frame back unchanged without saying so)."""
    known = set(model.batches)
    out: dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        batch = batch_of.get(name)
        if batch is None:
            raise ValueError(f"sample {name!r} has no batch")
        if str(batch) not in known:
            raise KeyError(
                f"the model has no batch {batch!r} (sample {name!r}); fitted "
                f"batches: {sorted(known)}")
        _check_channels(name, df, model.channels)
        out[name] = model.apply(df.copy(), str(batch))
    return out


# ── Before / after summary ───────────────────────────────────────────────────

def robust_spread(values: np.ndarray) -> float:
    """Half the 15.87th-84.13th percentile range of the finite values (the
    robust SD); NaN when there are none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float('nan')
    lo, hi = np.percentile(v, [_P_LO, _P_HI])
    return 0.5 * float(hi - lo)


def _median(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float('nan')


def normalization_summary(before: Mapping[str, pd.DataFrame],
                          after: Mapping[str, pd.DataFrame],
                          batch_of: Mapping[str, str],
                          channels: Sequence[str],
                          references: Mapping[str, str | Sequence[str]] | None = None,
                          ) -> pd.DataFrame:
    """One row per (channel, batch): the median and robust spread of the
    batch's events before and after normalization.

    With `references`, a batch is described by its reference sample(s)
    (``basis='reference'``: the same material in every batch, so after a
    good normalization its rows agree); otherwise by all its samples
    (``basis='pooled'``). Columns: channel, batch, basis, n_events,
    median_before, median_after, spread_before, spread_after."""
    refs = references or {}
    by_batch: dict[str, list[str]] = {}
    for name in before:
        by_batch.setdefault(batch_of[name], []).append(name)
    rows = []
    for ch in channels:
        for batch, names in by_batch.items():
            use = _as_list(refs.get(batch)) or names
            basis = 'reference' if refs.get(batch) else 'pooled'
            b = np.concatenate([before[n][ch].to_numpy(dtype=float) for n in use])
            a = np.concatenate([after[n][ch].to_numpy(dtype=float) for n in use])
            rows.append({
                'channel': str(ch), 'batch': str(batch), 'basis': basis,
                'n_events': int(np.isfinite(b).sum()),
                'median_before': _median(b), 'median_after': _median(a),
                'spread_before': robust_spread(b),
                'spread_after': robust_spread(a)})
    return pd.DataFrame(rows, columns=[
        'channel', 'batch', 'basis', 'n_events', 'median_before',
        'median_after', 'spread_before', 'spread_after'])


def methods_text(model, summary: pd.DataFrame,
                 references: Mapping[str, str | Sequence[str]] | None = None,
                 ) -> str:
    """A methods-report sentence: what was done, on how many batches and
    channels, and how far apart the batches' medians were (the mean over
    channels of their between-batch SD) before and after."""
    from . import __version__
    nb, nc = len(model.batches), len(model.channels)
    if model.mode == 'controls':
        what = ("a reference sample run in every batch"
                + (f" ({', '.join(sorted({r for v in (references or {}).values() for r in _as_list(v)}))})"
                   if references else ''))
    else:
        what = "all samples of each batch (CytoNorm 2.0 goal mode)"
    clusters = ("a single per-channel quantile mapping" if model.n_metaclusters <= 1
                else f"{model.n_metaclusters} FlowSOM metaclusters")
    s = (f"Batch effects across {nb} batches were normalized with CytoNorm "
         f"(Van Gassen et al. 2020) on {nc} channel(s), fitted on {what}, "
         f"with {clusters} and {model.n_quantiles} quantiles, in OpenFlo "
         f"{__version__}.")
    if len(summary):
        sd = summary.groupby('channel')[['median_before', 'median_after']].std(ddof=1)
        b, a = float(np.nanmean(sd['median_before'])), float(np.nanmean(sd['median_after']))
        if np.isfinite(b):
            s += (f" The between-batch SD of the "
                  f"{summary['basis'].iloc[0]} medians went from {b:.3g} to "
                  f"{a:.3g} (mean over channels, transformed scale).")
    return s


# ── FCS in / out ─────────────────────────────────────────────────────────────

def load_samples(paths: Iterable[str], compensate: bool = True,
                 transform: str | None = 'logicle') -> dict[str, Any]:
    """``{name: FlowSample}`` for each FCS path, compensated with its own
    spillover matrix and transformed (logicle by default; None leaves the
    values linear) as ``openflo-run --batch-correct`` prepares them. No
    events are removed (no QC), so a written copy holds every event."""
    from .pipeline import FlowSample
    out: dict[str, Any] = {}
    for p in paths:
        s = FlowSample(p)
        if compensate:
            s.auto_compensate()
        if transform:
            s.apply_transform(method=transform)
        if s.name in out:
            raise ValueError(f"two files are named {s.name!r}")
        out[s.name] = s
    return out


def provenance_keywords(model, batch: str,
                        reference: str | Sequence[str] | None = None) -> dict[str, str]:
    """The TEXT keywords a normalized copy carries to say what was done."""
    from .export_stamp import FCS_VERSION_KEY, version
    return {
        'OPENFLO_NORMALIZATION': 'CytoNorm',
        'OPENFLO_NORM_MODE': str(model.mode),
        'OPENFLO_NORM_BATCH': str(batch),
        'OPENFLO_NORM_REFERENCE': ','.join(_as_list(reference)) or 'none',
        'OPENFLO_NORM_CHANNELS': ','.join(model.channels),
        'OPENFLO_NORM_METACLUSTERS': str(model.n_metaclusters),
        'OPENFLO_NORM_QUANTILES': str(model.n_quantiles),
        FCS_VERSION_KEY: version(),
    }


def write_normalized_copies(samples: Mapping[str, Any],
                            corrected: Mapping[str, pd.DataFrame],
                            batch_of: Mapping[str, str],
                            model,
                            out_dir: str,
                            references: Mapping[str, str | Sequence[str]] | None = None,
                            suffix: str = '_norm') -> list[str]:
    """Write ``<out_dir>/<name><suffix>.fcs`` for each sample in
    `corrected`, returning the paths.

    `samples` are the loaded samples (their metadata, labels, ranges and
    transform records); `corrected` their normalized frames, on the
    samples' own scale (from :func:`apply_batch_model`). Values are written
    as compensated linear intensities (see the module docstring), with the
    source's keywords and the provenance keywords. A path that is one of
    the source files is refused, so an original is never overwritten."""
    from .fcs_export import population_export_frame, safe_filename, source_keywords, write_fcs
    refs = references or {}
    os.makedirs(out_dir, exist_ok=True)
    sources = {os.path.realpath(str(getattr(s, 'path', '') or ''))
               for s in samples.values()}
    paths, used = [], set()
    for name, df in corrected.items():
        s = samples[name]
        stem = safe_filename(f'{name}{suffix}')
        cand, i = stem, 1
        while cand.lower() in used:
            cand, i = f'{stem}_{i}', i + 1
        used.add(cand.lower())
        path = os.path.join(out_dir, f'{cand}.fcs')
        if os.path.realpath(path) in sources:
            raise ValueError(f"refusing to overwrite the source file {path}")
        view = copy.copy(s)
        view.data = df
        frame, ranges, _dropped = population_export_frame(
            view, np.ones(len(df), dtype=bool))
        keywords, voltages = source_keywords(s)
        batch = batch_of[name]
        keywords.update(provenance_keywords(model, batch, refs.get(batch)))
        write_fcs(frame, path, channel_labels=getattr(s, 'channel_labels', None),
                  ranges=ranges, keywords=keywords, voltages=voltages)
        paths.append(path)
    return paths
