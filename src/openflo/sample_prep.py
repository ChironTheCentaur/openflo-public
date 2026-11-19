"""Preparing a sample's data as every loaded sample is prepared -- Tk-free.

``prepare_fcs_sample`` reads the file and runs, in order, acquisition QC
(when asked), compensation (the matrix a workspace, a saved session or
another window staged for the file, else the file's own $SPILL), the
default transform and the channel labels. The gate editor's loader calls it
on a pool thread, and any other front end (a headless document, a phone
app) calls the same function, so a file opens with the same numbers
everywhere.

Everything it needs comes in as plain data: the staging record, the labels
string and the QC default the caller read from its own settings. It reads
no preferences file.

The same goes for the other steps a loaded sample can take before it is
gated: a session's processed-data sidecar read back as a sample
(``read_sidecar_checked``, ``sidecar_sample``), a sample put back on the
transforms a session saved for it (``restore_sample_transforms``), the
transforms its gates' coordinates are in (``stored_transforms``) and the
name it is listed under (``sample_name_for``).

Core module: no Tk.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Container, Mapping
from typing import Any

import numpy as np

from .cli_panel import parse_labels
from .compensation_io import non_finite_cells
from .errors import note
from .flow_sample import FlowSample


def prepare_fcs_sample(path: str, prep: Mapping[str, Any] | None = None, *,
                       labels: str = '', qc_default: bool = False
                       ) -> FlowSample:
    """The FlowSample for the raw FCS at `path`, prepared for gating.

    `prep` is what was staged for the file (the editor's
    ``stage_sample_prep`` record): ``'comp'`` is ``{'matrix', 'channels',
    'source'}``, the matrix the sample was compensated with where it came
    from; ``'qc'`` whether acquisition QC runs on it (None: `qc_default`);
    ``'set_number'`` which data set of a file holding several to read
    (None: the file as one sample). `labels` is a ``'DetA=LabelA;DetB=LabelB'`` string applied last.
    Safe off the main thread: it touches nothing but the new sample."""
    prep = prep or {}
    number = prep.get('set_number')
    s = FlowSample(path, set_number=number)
    # A sample arriving from a .wsp, a saved session or another window
    # brings the matrix it was compensated with there. Using the FCS's
    # own $SPILL instead would silently swap a custom (e.g. bead-derived)
    # matrix for the acquisition one, under gates drawn for the other.
    # Acquisition QC removes the events it flags, before any gate.
    # FlowJo has no such step and the .wsp has no QC population, so
    # with QC the same gates count differently there (the load status
    # and the .wsp export say so from s.qc_removed). Off unless the
    # caller's setting turns it on; a sample that comes back (session,
    # transfer) keeps the choice it was loaded with.
    qc = prep.get('qc')
    if qc is None:
        qc = qc_default
    if qc:
        s.run_qc()
    comp = prep.get('comp')
    bad = ''
    if comp:
        # Shape first: a matrix that does not fit its channel list is
        # refused by _apply_comp (CompensationError), not indexed here.
        cm = np.asarray(comp['matrix'], dtype=float)
        cch = [str(c) for c in comp['channels']]
        if cm.ndim == 2 and cm.shape == (len(cch), len(cch)):
            bad = non_finite_cells(cm, cch)
    if bad:
        # A saved matrix with a NaN/inf cell would make every value
        # unknown: the sample loads uncompensated, and says so.
        msg = (f"compensation: the saved spillover matrix was ignored, "
               f"a cell is not a finite number ({bad}); the sample is "
               "not compensated")
        s.load_warnings.append(msg)
    elif comp:
        s.manual_compensate(comp['matrix'], comp['channels'],
                            source=comp.get('source') or 'manual')
    else:
        s.auto_compensate()
    s.apply_transform()
    if labels:
        lbl = parse_labels(labels)
        if lbl:
            s.set_labels(lbl)
    return s


def mark_processed_transforms(sample, csv_path=None, provenance='foreign'):
    """Record the transform each column of a sample built from a CSV carries.
    Returns a note naming what was assumed ('' when nothing was).

    Every CSV OpenFlo writes has a record beside it (pipeline.
    write_transforms_sidecar); when it is there and matches the CSV, it is
    the answer. Without one, `provenance` decides:

    * ``'session'`` (a session's processed-data sidecar) or ``'events'`` (a
      workspace run's _events table) -- a CSV an older OpenFlo wrote from the
      editor's own data: its fluor columns are display-TRANSFORMED by
      definition, never linear. One whose finite values lie in the loader's
      logicle range is logicle (the default), however few or alike they are
      (one event, an empty channel); one outside it carries a transform that
      cannot be known (asinh, ...) and is recorded UNKNOWN: kept exactly as
      stored, never re-transformed, no linear statistics. Taking it as
      linear and re-transforming it would empty asinh gates (2.6.1 session:
      215 -> 0 events). The UNKNOWN entry names its origin -- the session's
      data folder, or this one file -- so a pooled run can tell columns that
      share one earlier editor's transform from ones that may not.
    * ``'foreign'`` -- File > Load CSV: a column is GUESSED logicle only if
      all its finite values lie in that range and are not all equal (an
      empty channel would invert to -110.87), else taken as linear (a linear
      export guessed logicle would read back as 5.2e44).

    Either way 'U:', 'MESF:' and 'D:' (derived) columns are linear (that is
    how OpenFlo writes them; 2.6.1's U: values guessed logicle would turn
    0.436 into 841), and non-float columns (cluster ids, '_pos' flags) are labels."""
    import numpy as np
    import pandas as pd

    from .channel_transforms import guessed_spec, transform_spec, unknown_spec
    from .transform_sidecar import logicle_display_bounds, read_transforms_sidecar
    ours = provenance in ('session', 'events')       # an older OpenFlo's CSV
    label = os.path.basename(csv_path) if csv_path else sample.name
    problem = ''
    if csv_path:
        try:
            rec = read_transforms_sidecar(csv_path, sample.data.columns,
                                          len(sample.data))
        except ValueError as exc:
            rec, problem = None, str(exc)
        if rec is not None:
            sample.data_transforms = rec
            return ''
    lo, hi = logicle_display_bounds()
    logicle, unknown, linear = [], [], []
    for c in sample.fluor_channels:
        if not pd.api.types.is_float_dtype(sample.data[c]):
            continue
        if str(c).startswith(('U:', 'MESF:', 'D:')):
            linear.append(c)
            continue
        v = np.asarray(sample.data[c].values, dtype=float)
        v = v[np.isfinite(v)]
        if ours:
            # Display-transformed by definition: one event, or a constant
            # (empty) channel, in range is logicle like any other.
            in_range = not v.size or (lo <= v.min() and v.max() <= hi)
        else:
            in_range = bool(v.size) and lo <= v.min() < v.max() <= hi
        if in_range:
            logicle.append(c)
        elif ours:
            unknown.append(c)
        else:
            linear.append(c)
    where = os.path.normcase(os.path.abspath(csv_path or sample.name))
    origin = (f"session:{os.path.dirname(where)}" if provenance == 'session'
              else f"file:{where}")
    # In range is only consistent with logicle, not proof of it: a dim
    # channel an earlier version had on asinh falls in range too, and read as
    # logicle its linear median comes out ~20x the truth (1793.94 vs 80.56).
    # Marking the guess keeps the common default-logicle case's statistics
    # while every window and CSV header says "(linear, guessed logicle)";
    # calling it unknown instead would cost every 2.6.1 session its numbers.
    rec = {c: guessed_spec(transform_spec('logicle')) for c in logicle}
    rec.update({c: unknown_spec(origin) for c in unknown})
    sample.data_transforms = rec
    if not (logicle or unknown or linear or problem):
        return ''

    def _names(cols):
        return ', '.join(map(str, cols[:6])) + (
            f' (+{len(cols) - 6})' if len(cols) > 6 else '')
    why = (f"its transform record was ignored ({problem})" if problem
           else "no transform record"
           + (" (written by an older OpenFlo)" if ours else ""))
    parts = []
    if logicle:
        parts.append(f"GUESSED logicle from their range: {_names(logicle)}")
    if unknown:
        parts.append(f"scale UNKNOWN, kept as stored, no linear statistics: "
                     f"{_names(unknown)}")
    if linear:
        parts.append(f"taken as linear: {_names(linear)}")
    return f"{label}: {why} — {'; '.join(parts) or 'no fluor columns'}."


def read_sidecar_checked(path: str) -> tuple[Any, str]:
    """``(frame, '')`` for the processed-data sidecar at `path`, or ``(None,
    why)`` when it cannot be used: it cannot be read, or holds fewer rows
    than its transform record says it was saved with (a write cut short by
    a crash must not load as the whole sample)."""
    from .session_sidecar import read_sidecar
    from .transform_sidecar import recorded_sidecar_rows
    try:
        df = read_sidecar(path)
        rows = recorded_sidecar_rows(path)
    except Exception as exc:              # noqa: BLE001 - any sidecar read failure makes the sample rebuild from its FCS, and the user is told
        return None, f"it could not be read ({type(exc).__name__})"
    if rows is not None and rows != len(df):
        return None, (f"it holds {len(df):,} of the {rows:,} events it was "
                      "saved with")
    return df, ''


def sidecar_sample(df: Any, path: str, prep: Mapping[str, Any] | None, *,
                   name: str, labels: Mapping[str, str] | None = None
                   ) -> tuple[FlowSample, str]:
    """``(sample, note)``: the FlowSample a session's processed-data sidecar
    `df` (read from `path`) gives, and the note naming the scale assumed for
    any column without a transform record ('' when none;
    ``mark_processed_transforms``).

    Its values are already QC'd, compensated and transformed, so none of
    that is redone. What `prep` (the staging record, as for
    ``prepare_fcs_sample``) says is put back on it: the matrix it was
    compensated with, its raw FCS (``source_fcs``; that file's keywords are
    read when it is there, from the .lmd copy ``lmd_data_set`` names, or
    the data set ``set_number`` names of a file holding several),
    whether the instrument already compensated its values
    (``instrument_compensated``) and its QC record."""
    prep = prep or {}
    s = FlowSample.from_dataframe(df, name=name, labels=labels, path=path)
    # A session's processed-data sidecar: OpenFlo's own data.
    assumed = mark_processed_transforms(s, path, provenance='session')
    # The sidecar holds values compensated with the session's matrix; put
    # the matrix (and the raw FCS) back on the sample so the .wsp export,
    # Compensation QC and the matrix editor still see them.
    comp = prep.get('comp')
    if comp:
        s.comp_matrix = np.asarray(comp['matrix'], dtype=float)
        s.comp_channels = [str(c) for c in comp['channels']]
        s.compensation_source = comp.get('source') or 'manual'
    # Values the instrument already compensated (the FCS 2.0 copy of an
    # .lmd file, or a concatenation holding one) stay protected from a
    # second matrix, even when their raw file is not found.
    s.instrument_compensated = bool(prep.get('instrument_compensated'))
    s.lmd_data_set = prep.get('lmd_data_set')
    s.set_number = prep.get('set_number')
    if prep.get('source_fcs'):
        s.source_fcs = prep['source_fcs']
        # The sidecar has no FCS keywords: read them (TEXT only) from its
        # FCS, or the instrument / date / voltages that group "Apply to:
        # this batch", CytoNorm batches and $TIMESTEP would come back empty
        # on every reopen.
        fcs_path = str(prep['source_fcs'])
        if os.path.isfile(fcs_path):
            try:
                s.restore_fcs_header(fcs_path,
                                     data_set=prep.get('lmd_data_set'),
                                     set_number=prep.get('set_number'))
            except Exception as exc:       # noqa: BLE001 - noted; the keywords only group batches, so the sample still opens without them
                note(f"session: {name}: FCS keywords not read", exc)
    if prep.get('qc') is not None:
        # The sidecar holds the events left after the QC it was saved with;
        # a rebuild from its FCS repeats that.
        s.qc_applied = bool(prep['qc'])
    if prep.get('qc_removed'):
        # ... and how many events that QC took out, which the sidecar cannot
        # say (the load status and .wsp export report it).
        s.qc_removed = int(prep['qc_removed'])
    return s, assumed


def restore_sample_transforms(s: Any, saved: Mapping[str, Any]) -> list[str]:
    """Put sample `s` back on the transforms a session saved for it
    (`saved`, a ``data_transforms`` dict: no entry = linear), undoing what
    its data carries first. A sidecar sample already matches; a sample
    reloaded from its raw FCS arrives on the loader's logicle. No-op for a
    sample without a record, and for a column whose transform is unknown
    on either side (nothing can be converted to or from it). Returns the
    channels it changed."""
    from .channel_transforms import is_guessed_spec, is_unknown_spec, transform_params
    rec = getattr(s, 'data_transforms', None)
    if rec is None:
        return []
    changed = [ch for ch in s.data.columns
               if saved.get(ch) != rec.get(ch)
               and not is_unknown_spec(saved.get(ch))
               and not is_unknown_spec(rec.get(ch))]
    for ch in changed:
        s.retransform([ch], **(transform_params(saved.get(ch))
                               or {'method': 'linear'}))
        if is_guessed_spec(saved.get(ch)):     # a guess stays one
            s.data_transforms = {**s.data_transforms,
                                 ch: dict(saved[ch])}
    return changed


def stored_transforms(sample: Any) -> dict[str, dict[str, Any]]:
    """``{channel: transform_spec}`` for the channels `sample` stores off
    the linear scale -- the space its gates' coordinates are in -- from its
    own ``data_transforms`` record."""
    rec = getattr(sample, 'data_transforms', None) or {}
    return {str(ch): dict(spec) for ch, spec in rec.items()
            if spec and spec.get('method', 'linear') != 'linear'}


def sample_name_for(path: str,
                    taken: Callable[[str], bool] | Container[str],
                    base: str | None = None) -> str:
    """The name to list the FCS at `path` under: its file name without the
    extension (or `base`, the name of one of its data sets,
    ``fcs_data_sets.data_set_names``), or, when `taken` (a set of names, or a function of a name)
    says that is in use, the name with its day or trial folder
    (``workspace_run.derive_trial_name``), then a counter. Day-organised
    drops reuse file names across days, and a bare name would make the
    second file collide with the first."""
    used = taken if callable(taken) else taken.__contains__
    if not base:
        base = os.path.basename(path).rsplit('.', 1)[0]
    name = base
    if used(name):
        from .workspace_run import derive_trial_name
        trial = derive_trial_name(path)
        name = f'{base} [{trial}]'
        n = 2
        while used(name):
            name = f'{base} [{trial}] ({n})'
            n += 1
    return name
