"""Writing a DataFrame of events to an FCS file in the ``write_fcs(df, path)``
order: ``write_fcs``, a thin wrapper over ``fcs_export.write_fcs``.

Core module: no Tk, and it does not import ``openflo.pipeline``, which imports
this name back, so ``pipeline.write_fcs`` and ``from openflo import write_fcs``
are this function.
"""

import os

import pandas as pd

from .fcs_export import df_path_args
from .fcs_export import write_fcs as _export_write_fcs


def write_fcs(df: pd.DataFrame, path: str | os.PathLike, channels: list | None = None,
              channel_labels: dict | None = None, corrections=None) -> int:
    """Write a DataFrame of events to an FCS 3.1 file (via FlowIO):
    ``write_fcs(df, path)``. The old ``write_fcs(path, df)`` order still works
    but is deprecated (DeprecationWarning). The writing is
    `fcs_export.write_fcs`'s.

    ``channels`` selects/orders the parameters written (default: every column).
    ``channel_labels`` (``{column: antibody}``) populates the per-parameter
    ``$PnS`` marker names, so the file re-opens in FlowJo / FCS Express with
    labels intact; a label that only repeats the channel name is left blank.
    Non-finite cells are zeroed (FCS stores finite floats) and counted in a
    logged warning. OPENFLO_VERSION (and OPENFLO_CORRECTIONS) record which
    OpenFlo wrote it. Returns the number of events written.

    Used to export a gated subpopulation as a standalone, re-importable FCS —
    pass the events you want and the channels to keep. For a population of a
    loaded sample, ``fcs_export.population_export_frame(sample, mask)`` gives
    them as an FCS should hold them (compensated, linear). Not
    ``sample.raw.iloc[mask]``: a mask is computed on ``sample.data``, whose
    rows the debris, doublet and region-gate filters drop without touching
    ``raw``, so the rows do not correspond."""
    df, path = df_path_args(df, path)
    return _export_write_fcs(df, path, channel_labels=channel_labels,
                             corrections=corrections, channels=channels)
