"""The single-sample analysis class, ``FlowSample``, and the categorical
palette its plots colour groups with (``categorical_cmap``,
``set_default_palette``).

Core module: no Tk, and it does not import ``openflo.pipeline``, which imports
these names back, so ``pipeline.FlowSample`` and ``from openflo import
FlowSample`` are this class.

Heavy / optional deps (phenograph, seaborn, matplotlib.pyplot, the embedding
libraries) are imported inside the methods that need them, so importing this
module stays cheap.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import re
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from . import fcs_read as _fcs_read
from . import numba_threads
from .acquisition_qc import AcquisitionQC
from .autoclean import find_scatter_channel as find_scatter_channel
from .cell_cycle import analyze_dna, assign_phase, find_dna_channel
from .channel_names import is_time_channel
from .channel_roles import default_cluster_channels
from .channel_roles import is_excluded as _is_excluded
from .channel_roles import is_scatter as _is_scatter
from .channel_transforms import (
    UnknownScaleError,
    guessed_spec,
    is_guessed_spec,
    is_unknown_spec,
    linear_values,
    to_linear,
    transform_spec,
    transform_values,
)
from .channel_transforms import inverse_transform_values as inverse_transform_values
from .cluster_graph import phenograph_prints_to_log as _phenograph_prints_to_log
from .cluster_graph import phenograph_sort_jobs as _phenograph_sort_jobs
from .cluster_graph import snn_jaccard_graph as _snn_jaccard_graph
from .cluster_labels import cluster_label
from .compensation_io import fcs_param_markers as _fcs_param_markers
from .compensation_io import fcs_param_names as _fcs_param_names
from .compensation_io import non_finite_cells as non_finite_cells
from .compensation_io import parse_spill_keyword as _parse_spill_keyword
from .compensation_io import spill_text as _spill_text
from .compensation_io import spill_units_to_fractions as _spill_units_to_fractions
from .compensation_math import matmul_known as _matmul_known
from .exceptions import ClusteringError, CompensationError, FcsParseError
from .exceptions import GateError as GateError
from .exceptions import OpenFloError as OpenFloError
from .exceptions import WspParseError as WspParseError
from .fcs_quirks import plain_fcs_problem, read_fcs_tolerant
from .fcs_range_fill import RangeFillingFlowData as _RangeFillingFlowData
from .flowsom import igraph_seeded
from .flowsom import som_assign as _som_assign
from .flowsom import som_metacluster as _som_metacluster
from .flowsom import som_train as _som_train
from .gate_eval import depends_on_flowjo_placeholder as depends_on_flowjo_placeholder
from .gate_eval import gate_to_mask as gate_to_mask
from .gate_eval import points_in_polygon as _points_in_polygon
from .gate_text import WSP_BOOLEAN_OPS as WSP_BOOLEAN_OPS
from .gate_text import (
    describe_gate,
)
from .gate_text import resolve_wsp_dependent as resolve_wsp_dependent
from .gpu_probe import CUML_UMAP as _CumlUMAP
from .gpu_probe import (
    GPU_AVAILABLE,
    GPU_CLUSTERING_AVAILABLE,
    GPU_NAME,
)
from .gpu_probe import GPU_CLUSTER_KIT as _GPU_CLUSTER_KIT
from .gpu_probe import vram_free_gb as _vram_free_gb
from .label_alignment import concatenate as concatenate
from .log_capture import collect_warnings as _collect_warnings
from .parallel import tree_kneighbors
from .scales import (
    ASINH_DEFAULT_COFACTOR,
    LOGICLE_A,
    LOGICLE_M,
    LOGICLE_T,
    LOGICLE_W,
)
from .seeds import signed32_seed
from .som_kernel import FLOWSOM_DEFAULT_METACLUSTERS
from .transform_sidecar import (
    discard_transforms_sidecar,
    write_transforms_sidecar,
)
from .wsp_reader import WspReader
from .wsp_writer import WspWriter as WspWriter

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

log = logging.getLogger('openflo.pipeline')


# Default categorical palette for per-sample / per-condition colouring on
# plots that group by `sample_origin` etc. `'auto'` picks the colour-blind
# safe Okabe-Ito colours for ≤8 groups and falls back to tab20 / evenly
# spaced hues as the group count grows. Override via `set_default_palette('Set2')` or by
# passing `--palette` / the GUI palette dropdown to the pipeline.
_DEFAULT_CATEGORICAL_PALETTE = 'auto'


def categorical_cmap(name, n_groups):
    """A colormap whose first ``n_groups`` steps are the group colours.

    'auto' and 'okabe_ito' give the colour-blind-safe Okabe-Ito colours
    (distinct under colour-vision deficiency up to 8 groups), 'classic'
    tab10's; any other name is a matplotlib
    colormap. The map is listed with exactly ``n_groups`` colours, so with
    ``vmin=0, vmax=n_groups-1`` group i gets colour i."""
    import matplotlib
    from matplotlib.colors import ListedColormap

    from .group_palettes import n_group_colors
    n = max(1, int(n_groups))
    key = {'auto': 'okabe_ito', 'okabe-ito': 'okabe_ito',
           'tab10': 'classic'}.get(str(name), str(name))
    if key in ('okabe_ito', 'classic'):
        return ListedColormap(n_group_colors(n, key), name=key)
    return matplotlib.colormaps[str(name)]


def set_default_palette(name):
    """Process-wide default for the categorical colour palette used by
    FlowSample.plot when colouring by a string column (sample_origin,
    condition, etc.). Common picks: 'auto', 'tab10', 'Set1', 'Set2',
    'Dark2', 'Paired'. Anything matplotlib's get_cmap accepts works."""
    global _DEFAULT_CATEGORICAL_PALETTE
    _DEFAULT_CATEGORICAL_PALETTE = str(name) or 'auto'


def _checked_resolution(resolution) -> float:
    """``resolution`` as a float; raise ValueError unless finite and > 0."""
    resolution = float(resolution)
    if not (np.isfinite(resolution) and resolution > 0):
        raise ValueError(f"resolution must be > 0, got {resolution}")
    return resolution


def _cluster_subsample(Xc: np.ndarray, max_events: int | None,
                       random_state: int
                       ) -> tuple[np.ndarray | None, np.ndarray]:
    """The rows ``FlowSample.cluster`` clusters: ``(sub_idx, X_cluster)``.

    When ``max_events`` is set and smaller than the row count, a sorted
    seeded random subset of that size; otherwise every row and
    ``sub_idx`` None."""
    n = len(Xc)
    if max_events and n > max_events:
        rng     = np.random.default_rng(random_state)
        sub_idx = np.sort(rng.choice(n, max_events, replace=False))
        return sub_idx, Xc[sub_idx]
    return None, Xc


def _should_try_gpu(reproducible: bool, use_gpu: bool | str,
                    vram_admission_gb: float) -> bool:
    """Whether ``FlowSample.cluster`` attempts the GPU backend first.

    Reproducible mode forces the CPU seeded-Leiden path: Louvain (CPU
    Blondel and GPU cuGraph alike) is non-deterministic per process and
    cannot be pinned from outside."""
    if reproducible or use_gpu is False or not GPU_CLUSTERING_AVAILABLE:
        return False
    free_vram = _vram_free_gb()
    if use_gpu is True or free_vram is None or free_vram >= vram_admission_gb:
        return True
    log.info(
        "  [VRAM admission] %.1f GB < %.1f GB — clustering on CPU",
        free_vram, vram_admission_gb)
    return False


def _phenograph_kwargs(cluster_fn, k: int, n_jobs: int, reproducible: bool,
                       random_state: int,
                       resolution: float) -> tuple[dict, str]:
    """Keyword arguments for ``phenograph.cluster`` and the route
    (``'leiden'`` or ``'louvain'``) they select."""
    pg_kwargs: dict = {'k': k, 'n_jobs': n_jobs}
    if not reproducible:
        return pg_kwargs, 'louvain'
    # PhenoGraph's default Louvain is non-deterministic; its Leiden
    # backend accepts an explicit seed. Feature-detect so an older
    # build degrades gracefully (a warning, not a crash).
    import inspect
    _pg = inspect.signature(cluster_fn).parameters
    if 'clustering_algo' not in _pg or 'seed' not in _pg:
        log.warning(
            "  [!] reproducible clustering needs a phenograph build "
            "with seeded Leiden; using (non-deterministic) Louvain.")
        return pg_kwargs, 'louvain'
    pg_kwargs['clustering_algo'] = 'leiden'
    # leidenalg takes a signed 32-bit seed (seeds.signed32_seed).
    pg_kwargs['seed'] = signed32_seed(random_state)
    # Only a non-default value is passed, so 1.0 sends no
    # resolution at all (phenograph's own default is 1).
    if resolution != 1.0:
        pg_kwargs['resolution_parameter'] = resolution
    return pg_kwargs, 'leiden'


def _run_phenograph_cpu(X_cluster: np.ndarray, k: int, n_jobs: int,
                        reproducible: bool, random_state: int,
                        resolution: float,
                        shape: str) -> tuple[np.ndarray, float, str]:
    """CPU PhenoGraph on ``X_cluster``: ``(communities, Q, route)``.

    Raises ClusteringError when phenograph is missing or fails."""
    log.info(f"  Phenograph: {shape} …")
    # Lazy import — drags in igraph + sklearn.community, ~1 s and
    # ~200 MB. Only relevant for callers that actually cluster.
    try:
        import phenograph
    except ImportError as e:
        raise ClusteringError(
            "phenograph is required for CPU clustering "
            "(pip install phenograph)") from e
    pg_kwargs, route = _phenograph_kwargs(
        phenograph.cluster, k, n_jobs, reproducible, random_state,
        resolution)
    try:
        with tempfile.TemporaryDirectory(
                prefix='openflo-phenograph-',
                ignore_cleanup_errors=True) as scratch, \
                contextlib.chdir(scratch), \
                _phenograph_sort_jobs(n_jobs), \
                _phenograph_prints_to_log():
            sub_comm, _, Q = phenograph.cluster(X_cluster, **pg_kwargs)
    except Exception as e:
        raise ClusteringError(
            f"Phenograph CPU clustering failed: {e}") from e
    return sub_comm, Q, route


def _phenograph_settings(k: int, route: str, backend: str,
                         random_state: int, resolution: float) -> dict:
    """``cluster_settings`` for a PhenoGraph run; warn when a non-default
    resolution could not be applied (only the Leiden route takes one)."""
    if route != 'leiden' and resolution != 1.0:
        log.warning(
            "  [!] resolution %.2f not applied: PhenoGraph's Louvain has "
            "no resolution; it needs reproducible (seeded Leiden) "
            "clustering.", resolution)
    return {
        'method': 'phenograph', 'k': int(k), 'route': route,
        'backend': backend, 'random_state': random_state,
        'resolution': resolution if route == 'leiden' else None}


def _expand_subsample_labels(Xc: np.ndarray, X_cluster: np.ndarray,
                             sub_idx: np.ndarray,
                             sub_comm: np.ndarray) -> np.ndarray:
    """Labels for every row of ``Xc`` from the labels of its sub-sample.

    Rows outside the sub-sample take the label of their nearest
    non-noise sub-sample row (KD-tree, 1-NN); when the whole sub-sample
    is noise (-1) they stay -1."""
    n = len(Xc)
    communities          = np.full(n, -1, dtype=int)
    communities[sub_idx] = sub_comm
    good = sub_comm >= 0
    if good.sum() > 0:
        from sklearn.neighbors import NearestNeighbors
        nn_clf = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn_clf.fit(X_cluster[good])
        ref_labels         = sub_comm[good]
        rest_mask          = np.ones(n, dtype=bool)
        rest_mask[sub_idx] = False
        n_rest             = int(rest_mask.sum())
        log.info(
            "  Assigning %s remaining events to nearest cluster "
            "(KD-tree, 1-NN) …", f"{n_rest:,}")
        nbr = tree_kneighbors(nn_clf, Xc[rest_mask], 1)
        communities[rest_mask] = ref_labels[nbr[:, 0]]
    else:
        log.warning(
            "  [!] All sub-sample events were marked as noise — "
            "the rest cannot be assigned.")
    return communities


class FlowSample:
    """
    Single FCS file with full analysis pipeline.

    Typical workflow
    ----------------
        s = FlowSample('file.fcs')
        s.run_qc()
        s.auto_compensate()           # OR s.compensate_from_wsp('exp.wsp')
        s.apply_transform()
        s.apply_threshold_gates(thresholds)   # optional FMO gates
        s.cluster(k=30)
        s.run_umap()
        s.plot('CD901b', 'CD903', color_by='cluster')
        s.cluster_heatmap()
        s.export_csv()
    """

    def __init__(self, fcs_path: str | Path) -> None:
        self.path            = fcs_path
        self.name            = os.path.splitext(os.path.basename(fcs_path))[0]
        self.metadata        = {}
        self.channel_names   = []
        self.channel_labels  = {}
        self.scatter_channels = []
        self.fluor_channels   = []
        self.thresholds       = {}
        # Compensation matrix actually applied to this sample (post-channel-
        # intersection). Populated by `_apply_comp`, read by the workspace
        # exporters so the .wsp round-trips the spillover.
        self.comp_matrix:   np.ndarray | None = None
        self.comp_channels: list[str] = []
        # Per channel, the transform_spec currently baked into self.data (no
        # entry = linear). Written by apply_transform / retransform /
        # mark_transformed and read by linear_values. Always REBOUND, never
        # mutated, so a copy.copy of a sample cannot change the original's.
        self.data_transforms: dict[str, dict] = {}
        # Per channel, the axis a FlowJo workspace declared for this sample
        # (wsp_stat_axes), which FlowJo's GeoMean is averaged on; empty for a
        # sample not imported from a workspace (gating.population_stats).
        self.stat_axes: dict[str, dict] = {}
        # Where comp_matrix came from ('$SPILL', 'wsp:<name>', 'manual'), or
        # None while uncompensated. Recorded in the editor's provenance log.
        self.compensation_source: str | None = None
        # The raw FCS behind a sample rebuilt from a processed CSV (whose
        # `path` is the CSV); None for a sample read from its FCS.
        self.source_fcs: str | None = None
        # Events run_qc removed from `data` (FlowJo would count them).
        self.qc_removed: int = 0
        # Whether run_qc has run on this sample: a rebuild (re-compensation,
        # session reload from the raw FCS) repeats the same choice, so the
        # event set under existing gates does not change.
        self.qc_applied: bool = False
        # Channels whose stored integers _load turned into intensities, as
        # {channel: what was done} ('log, 4 decades' / 'gain 2'), and each
        # such channel's linear $PnR equivalent ({channel: max value}).
        self.fcs_decoding: dict[str, str] = {}
        self.linear_ranges: dict[str, float] = {}
        # What the loader had to work around in the file (a duplicate
        # channel name, a wrong $TOT), for the user, not just the log.
        self.load_warnings: list[str] = []
        # data / raw are populated unconditionally by _load() below. We
        # initialise with empty DataFrames (rather than None) so static
        # analysers can see them as pd.DataFrame everywhere downstream
        # without needing per-call narrowing.
        self.data: pd.DataFrame = pd.DataFrame()
        self.raw:  pd.DataFrame = pd.DataFrame()
        self.clusters         = None
        # What cluster() last ran with (k, route, applied resolution).
        self.cluster_settings: dict | None = None
        self.umap_coords      = None
        self.trimap_coords    = None
        self.pacmap_coords    = None
        self.cell_cycle_result = None
        self.flowsom_result   = None

        self._load()
        self._classify_channels()
        self._print_summary()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = 'sample',
                       labels: Mapping[str, str] | None = None,
                       metadata: Mapping[str, Any] | None = None,
                       path: str | os.PathLike[str] = '') -> FlowSample:
        """Build a FlowSample from an in-memory DataFrame instead of an FCS
        file — e.g. to re-open a pipeline ``*_processed.csv`` (which carries
        cluster / UMAP / flowsom columns) in the editor.

        `labels` is an optional ``{column: antibody label}`` map; columns
        without one use the column name. Derived/analysis columns (cluster,
        UMAP*, flowsom*, cell_cycle, Time) are auto-excluded from the marker
        lists by ``_classify_channels``."""
        import pandas as pd
        s = cls.__new__(cls)
        s.path            = path
        s.name            = name
        s.metadata        = dict(metadata or {})
        s.channel_names   = list(df.columns)
        labels = labels or {}
        s.channel_labels  = {c: (labels.get(c) or c) for c in s.channel_names}
        s.scatter_channels = []
        s.fluor_channels   = []
        s.thresholds       = {}
        s.comp_matrix      = None
        s.comp_channels    = []
        # The frame's values are taken as given (linear). A caller holding
        # already-transformed data says so with mark_transformed().
        s.data_transforms  = {}
        s.stat_axes        = {}
        s.compensation_source = None
        s.source_fcs       = None
        s.qc_removed       = 0
        s.qc_applied       = False
        s.fcs_decoding     = {}
        s.linear_ranges    = {}
        s.load_warnings    = []
        s.data = pd.DataFrame(df).reset_index(drop=True).copy()
        s.raw  = s.data.copy()
        s.clusters         = None
        s.cluster_settings = None
        s.umap_coords      = None
        s.trimap_coords    = None
        s.pacmap_coords    = None
        s.cell_cycle_result = None
        s.flowsom_result   = None
        s._classify_channels()
        return s

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        filled: list[str] = []
        quirks: list[str] = []
        try:
            try:
                fcs = read_fcs_tolerant(self.path, _fcs_read.FlowData, quirks)
            except KeyError as e:
                # A file without a required $PnR was refused outright, where
                # FlowJo reads it with the range its $PnB allows.
                if not re.fullmatch(r"p\d+r", str(e).strip("'\"").lower()):
                    raise
                quirks.clear()
                fcs = read_fcs_tolerant(
                    self.path, functools.partial(_RangeFillingFlowData,
                                                 filled=filled), quirks)
        except FileNotFoundError as e:
            raise FcsParseError(f"FCS file not found: {self.path}") from e
        except Exception as e:
            raise FcsParseError(
                f"could not read {os.path.basename(self.path)}: "
                f"{plain_fcs_problem(e)} ({type(e).__name__}: {e})") from e
        self.metadata = dict(fcs.text)
        # FlowIO ≥1.0 uses integer keys and lowercase field names (pnn/pns).
        # A parameter with no $PnN is not in flowio's channel map at all: it
        # is named Ch<n> by the fallback, never left to escape as a bare
        # KeyError ("Failed to load X: 3").
        named, renamed = self._fcs_channel_names(fcs)
        for name, label in named:
            self.channel_names.append(name)
            self.channel_labels[name] = label if label else name
        if renamed:
            log.warning("  [%s] duplicate channel names renamed: %s",
                        self.name, ", ".join(f"{a} -> {b}" for a, b in renamed))
        if quirks:
            log.warning("  [%s] %s", self.name, '; '.join(quirks))
        self.load_warnings = quirks + filled + [
            f"duplicate channel name {a!r}: the second is {b!r}"
            for a, b in renamed]
        if filled:
            log.warning("  [%s] %s", self.name, '; '.join(filled))
        if not any(self.channel_labels[n] != n for n in self.channel_names):
            log.info("  (no PnS labels in FCS — use detector names for plotting)")
        # flowio types `events` as Optional but in practice it's always
        # populated after a successful FlowData() construction; assert so
        # static analysis follows.
        assert fcs.events is not None
        events   = np.reshape(np.asarray(fcs.events), (-1, fcs.channel_count))
        events   = self._decode_stored_values(fcs, events)
        try:
            tot = int(str(fcs.text.get('tot', '')).strip())
        except ValueError:
            tot = None
        if tot is not None and tot != len(events):
            # The events are what the data segment holds; the header's count
            # disagreeing says the file was written or copied wrongly.
            msg = (f"$TOT says {tot:,} events; the data holds "
                   f"{len(events):,}")
            log.warning("  [%s] %s", self.name, msg)
            self.load_warnings.append(msg)
        # No copy is made here that nothing needs: at the real workload --
        # 20-25 multimillion-event files -- two such copies dominate both
        # load time and memory. Measured on 8 x 2M events x 18 channels:
        # 2.25s / +2304 MB with them, 0.75s / +1152 MB without; projected to
        # 25 files, 7.0s / 7.2 GB against 2.3s / 3.6 GB. The memory halving matters
        # more than the seconds: 7 GB of resident frames is where a machine
        # starts paging, and paging is what makes loading *feel* slow.
        #
        # `copy=False`: flowio hands back an `array.array` supporting the
        # buffer protocol, so `np.asarray` is already a free view -- the
        # 175 ms/file would go on pandas materialising its own copy of a buffer
        # nothing else references.
        #
        # `copy(deep=False)`: `raw` must stay the pristine detector values
        # (the .fcs export reads it), and `self.data` is written IN PLACE in
        # a dozen places -- compensation and transforms among them -- so
        # aliasing the two would corrupt `raw`. Under pandas >= 3 copy-on-write
        # a shallow copy is a distinct object that duplicates a block only
        # when written, which gives independence without paying for it up
        # front. Verified: `data[col] = ...` and `data.loc[...] = ...` both
        # leave `raw` untouched, and no code path writes through `.values`,
        # which would bypass CoW.
        self.raw  = pd.DataFrame(events, columns=pd.Index(self.channel_names),
                                 copy=False)
        self.data = self.raw.copy(deep=False)

    @staticmethod
    def _fcs_channel_names(
            fcs: Any,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """[(column name, $PnS label)] for each parameter of the flowio
        ``fcs``, in order, and the duplicate names renamed as [(old, new)]."""
        seen: dict[str, int] = {}
        renamed = []
        named = []
        for i in range(1, fcs.channel_count + 1):
            ch    = fcs.channels.get(i) or {}
            name  = str(ch.get('pnn') or ch.get('PnN') or f'Ch{i}')
            label = (ch.get('pns') or ch.get('PnS') or '').strip()
            # Two detectors with one name loaded as two same-named columns,
            # and the second could not be reached by name: the transform
            # then failed on both. Keep both, the second as 'FL1-A (2)'.
            if name in seen:
                seen[name] += 1
                new = f'{name} ({seen[name]})'
                while new in seen:
                    seen[name] += 1
                    new = f'{name} ({seen[name]})'
                renamed.append((name, new))
                name = new
            seen.setdefault(name, 1)
            named.append((name, label))
        return named, renamed

    def restore_fcs_header(self, fcs_path: str | Path) -> None:
        """Put back what this sample's raw FCS says about it -- its TEXT
        keywords (`metadata`) and how its stored values were decoded
        (`fcs_decoding`, `linear_ranges`) -- reading the TEXT segment only.
        For a sample rebuilt from a processed CSV, which carries the events
        but none of these: without them it lost its instrument, date,
        voltages and $TIMESTEP on every reopen. Raises as FlowSample does
        when the file cannot be read."""
        quirks: list[str] = []
        fcs = read_fcs_tolerant(
            fcs_path, functools.partial(_fcs_read.FlowData, only_text=True),
            quirks)
        self.metadata = dict(fcs.text)
        names = [n for n, _ in self._fcs_channel_names(fcs)[0]]
        _plan, decoding, ranges = self._decoding_plan(fcs, names)
        self.fcs_decoding = decoding
        self.linear_ranges = ranges

    @staticmethod
    def _decoding_plan(
            fcs: Any, channel_names: Sequence[str],
    ) -> tuple[list[tuple[int, float, float, float, float]],
               dict[str, str], dict[str, float]]:
        """What _decode_stored_values does to each channel of the flowio
        ``fcs`` (its columns named `channel_names`): the plan
        [(column index, decades, log0, range, gain)], and the
        {channel: what was done} and {channel: linear $PnR} it records."""
        datatype = str(fcs.text.get('datatype', '')).strip().upper()
        plan = []                      # (column, decades, log0, range, gain)
        decoding: dict[str, str] = {}
        ranges: dict[str, float] = {}
        for i in range(1, fcs.channel_count + 1):
            ch = fcs.channels.get(i) or {}
            name = channel_names[i - 1]
            if is_time_channel(name):
                continue
            decades, log0 = ch.get('pne') or (0.0, 0.0)
            gain = float(ch.get('png') or 1.0)
            rng = float(ch.get('pnr') or 0.0)
            if datatype != 'I':
                if gain not in (0.0, 1.0):
                    log.info("  %s: $PnG %g kept as acquisition metadata "
                             "(float data is stored as intensities)",
                             name, gain)
                continue
            if decades > 0 and rng > 0:
                log0 = float(log0 or 1.0)
                plan.append((i - 1, float(decades), log0, rng, 1.0))
                decoding[name] = f"log, {float(decades):g} decades"
                ranges[name] = log0 * 10.0 ** float(decades)
            elif gain not in (0.0, 1.0):
                plan.append((i - 1, 0.0, 0.0, rng, gain))
                decoding[name] = f"gain {gain:g}"
                if rng > 0:
                    ranges[name] = rng / gain
        return plan, decoding, ranges

    def _decode_stored_values(self, fcs: Any,
                              events: np.ndarray) -> np.ndarray:
        """`events` as intensities. An integer FCS stores a log-amplified
        channel ($PnE f1,f2 with f1 > 0) as channel numbers 0..$PnR-1, whose
        intensity is f2 * 10**(f1 * x / $PnR), and an amplified linear one
        ($PnG g) as g times the intensity (FCS 3.1, section 3.3.29 / 3.3.31).
        Loaded unchanged, gates, transforms and statistics would run on
        channel numbers: a 4-decade channel's 1023 would read 1023, not 9,910.

        Only integer data is decoded: the standard requires $PnE 0,0 for
        float data, and a float file's $PnG describes acquisition, not how
        its values are stored, so it is left alone (and said so). Time is
        never rescaled. A file needing nothing keeps the zero-copy view."""
        plan, decoding, ranges = self._decoding_plan(fcs, self.channel_names)
        if not plan:
            return events
        self.fcs_decoding.update(decoding)
        self.linear_ranges.update(ranges)
        out = np.array(events, dtype=np.float64)
        for idx, decades, log0, rng, gain in plan:
            if decades > 0:
                out[:, idx] = log0 * 10.0 ** (decades * out[:, idx] / rng)
            else:
                out[:, idx] = out[:, idx] / gain
        log.warning("  [%s] stored values converted to intensities: %s",
                    self.name, ", ".join(f"{c} ({d})"
                                         for c, d in self.fcs_decoding.items()))
        return out

    def _classify_channels(self) -> None:
        for ch in self.channel_names:
            if _is_scatter(ch):
                self.scatter_channels.append(ch)
            elif not _is_excluded(ch) and not self._derived_column(ch):
                self.fluor_channels.append(ch)

    def _derived_column(self, ch: str) -> bool:
        """True for a column a sample rebuilt from a CSV carries that is not
        a measurement: a flag or label (bool / text: '<ch>_pos',
        'sample_origin') or a MESF: calibration of a detector. A live sample
        never lists these as fluorescence (they are added after loading), but
        from_dataframe did, so a restored session clustered on them by
        default: a MESF: column (linear, ~1e4-1e5) beside logicle detectors
        (0-1) is all a kNN distance sees."""
        if str(ch).startswith(('MESF:', 'D:')):
            return True
        from .result_stamp import is_original_column
        if is_original_column(ch):
            return True        # a corrected column's kept original (fixes)
        data = getattr(self, 'data', None)
        col = data.get(ch) if isinstance(data, pd.DataFrame) else None
        if isinstance(col, pd.DataFrame):          # duplicate column names
            col = col.iloc[:, 0]
        # A name with no column has no dtype to judge: its name decides.
        return col is not None and (pd.api.types.is_bool_dtype(col)
                                    or not pd.api.types.is_numeric_dtype(col))

    def _print_summary(self) -> None:
        labels = [self.channel_labels[c] for c in self.fluor_channels]
        log.info(
            "[%s]  %s events  |  fluor channels: %s",
            self.name, f"{len(self.data):,}", labels)

    def set_labels(self, mapping: Mapping[str, str]) -> FlowSample:
        """
        Assign antibody names to detector channels after loading.
        mapping: {detector_name: antibody_label}
        e.g. {'FL1-A': 'CD901b', 'FL2-A': 'CD902', 'FL3-A': 'CD903'}
        After calling this, plot('CD901b', 'CD903') and axis labels both work.
        """
        applied = {}
        for det, label in mapping.items():
            if det in self.channel_labels:
                self.channel_labels[det] = label
                applied[det] = label
        if applied:
            log.info(f"  Labels: { dict(applied.items()) }")
        return self

    # ── QC ────────────────────────────────────────────────────────────────────

    def run_qc(self, n_bins: int = 200, threshold: float = 5,
               plot: bool = False) -> FlowSample:
        """Acquisition QC: drop the events AcquisitionQC flags from `data`
        AND `raw`. Were `raw` to keep them, its rows would not match
        `data`'s, and the CLI's --unmix, which reads `raw`, would unmix every
        event QC had removed (QC would have no effect there)."""
        qc        = AcquisitionQC(self.data)
        clean_idx = qc.run(n_bins=n_bins, threshold=threshold)
        if len(clean_idx) != len(self.data) and len(self.raw) == len(self.data):
            keep = self.data.index.isin(clean_idx)
            self.raw = self.raw.loc[keep].reset_index(drop=True)
        # Recorded, not only logged: these events are gone before any gate,
        # and FlowJo, which has no such step, counts them (20,000 events ->
        # 19,614; a gate at 6,371 in FlowJo read 5,985 after QC).
        self.qc_removed = (getattr(self, 'qc_removed', 0)
                           + len(self.data) - len(clean_idx))
        self.qc_applied = True
        self.data = self.data.loc[clean_idx].reset_index(drop=True)
        if plot:
            qc.plot()
        return self

    # ── Debris + doublet filtering ────────────────────────────────────────────

    def _find_scatter_col(self, prefix: str,
                          suffix: str = '-A') -> str | None:
        """Return the column matching e.g. FSC-A or FSC-H, case-insensitive,
        preferring exact -A / -H endings. None if not found."""
        return find_scatter_channel(self.data.columns, prefix, suffix)

    def filter_debris(self, fsc_channel: str | None = None,
                      min_fsc: float | None = None) -> FlowSample:
        """Drop events whose FSC-A is below `min_fsc`. No-op if `min_fsc`
        is None or the FSC-A column can't be located."""
        if min_fsc is None:
            return self
        if fsc_channel is None:
            fsc_channel = self._find_scatter_col('FSC', '-A')
        if not fsc_channel or fsc_channel not in self.data.columns:
            log.info("  [Debris] No FSC-A channel found — debris filter skipped.")
            return self
        before = len(self.data)
        keep   = self.data[fsc_channel] >= float(min_fsc)
        self.data = cast(pd.DataFrame, self.data[keep]).reset_index(drop=True)
        kept = len(self.data)
        pct  = (kept / before * 100.0) if before else 0.0
        log.info(
            "  [Debris] Kept %s / %s events (%.1f%%) — FSC-A >= %.0f",
            f"{kept:,}", f"{before:,}", pct, float(min_fsc))
        return self

    def filter_doublets(self, fsc_a_channel: str | None = None,
                        fsc_h_channel: str | None = None,
                        tol: float | None = 0.25) -> FlowSample:
        """Drop doublets via the FSC-A / FSC-H ratio. Singlets fall along
        FSC-A ≈ k·FSC-H; doublets push FSC-A high relative to FSC-H, so
        their ratio sits well outside the population median. We keep
        events whose ratio is within ±`tol` of the median ratio.

        `tol` defaults to 0.25 (a relatively wide window) because
        polyploid cells are intrinsically more variable in
        FSC-A / FSC-H than typical leukocytes. Tighten to 0.15 for
        diploid samples.
        """
        if tol is None or tol <= 0:
            return self
        if fsc_a_channel is None:
            fsc_a_channel = self._find_scatter_col('FSC', '-A')
        if fsc_h_channel is None:
            fsc_h_channel = self._find_scatter_col('FSC', '-H')
        if (not fsc_a_channel or fsc_a_channel not in self.data.columns
                or not fsc_h_channel or fsc_h_channel not in self.data.columns):
            log.warning(
                "  [Doublets] FSC-A or FSC-H channel missing — "
                "doublet filter skipped.")
            return self
        before  = len(self.data)
        fsc_a   = self.data[fsc_a_channel].astype(float)
        fsc_h   = self.data[fsc_h_channel].astype(float)
        ratio   = np.where(fsc_h > 0, fsc_a / fsc_h, np.nan)
        valid   = np.isfinite(ratio)
        if not valid.any():
            # No usable FSC-A/FSC-H ratio anywhere. Substituting a median of
            # 0.0 would make the acceptance window [0, 0] and drop EVERY
            # event — a whole sample deleted in silence. Doublets
            # cannot be identified without the ratio, so remove nothing and say
            # so, matching the missing-channel branch above.
            log.warning(
                "  [Doublets] No usable FSC-A/FSC-H ratio (FSC-H is "
                "non-positive for every event) — doublet filter skipped, "
                "all %s events kept.", f"{before:,}")
            return self
        median  = float(np.nanmedian(ratio[valid]))
        lo, hi  = median * (1.0 - tol), median * (1.0 + tol)
        keep    = valid & (ratio >= lo) & (ratio <= hi)
        self.data = cast(pd.DataFrame, self.data[keep]).reset_index(drop=True)
        kept = len(self.data)
        pct  = (kept / before * 100.0) if before else 0.0
        log.info(
            "  [Doublets] Kept %s / %s events (%.1f%%) — "
            "FSC-A/FSC-H in [%.3f, %.3f] (median %.3f, ±%.0f%%)",
            f"{kept:,}", f"{before:,}", pct, lo, hi, median, tol * 100)
        return self

    # ── Compensation ──────────────────────────────────────────────────────────
    #
    # Compensation is a STATE of the sample, not an operation on whatever
    # `self.data` holds at the moment: every entry point below sets "the
    # matrix applied to this sample". A second call replaces the first --
    # its effect is undone before the new matrix is applied -- so it can
    # never compound. Multiplying the current data by inv(M) every time
    # would make auto_compensate() followed by compensate_from_wsp() (or the
    # same call twice, or FMOGater.prepare() run twice) compensate
    # already-compensated values. Measured that way: a second
    # auto_compensate() moved a 0.30-spillover channel by ~15,000 units.
    #
    # Compensation is linear algebra on LINEAR values, so it is refused on a
    # channel that has already been transformed (logicle etc.) rather than
    # silently mixing transformed coordinates.

    def auto_compensate(self) -> FlowSample:
        """Apply the spillover matrix embedded in the FCS metadata ($SPILL /
        $SPILLOVER) -- what FlowJo calls the "Acquisition-defined" matrix.
        Replaces any matrix applied earlier."""
        with _collect_warnings() as said:
            spill = self._parse_spillover()
            if spill is not None:
                self._apply_comp(spill['matrix'], spill['channels'],
                                 source='$SPILL')
        # What went wrong with the file's matrix (malformed, singular,
        # channels it names that the data lacks, a percent matrix) goes to
        # load_warnings too: the window never shows the log.
        warnings = getattr(self, 'load_warnings', None)
        if warnings is not None:
            warnings.extend(f"compensation: {m}" for m in said)
        if spill is None and not said:
            log.warning(f"  [!] No spillover in FCS metadata for {self.name}.")
        return self

    def compensate_from_wsp(self, wsp_path: str | os.PathLike[str],
                            matrix_name: str | None = None) -> FlowSample:
        """Apply compensation from a FlowJo .wsp file.

        With `matrix_name`, that named matrix. Without it, the matrix the
        workspace applies to THIS file, as FlowJo does: the spillover matrix
        inside the <Sample> whose DataSet is this FCS or, when that sample
        carries none, FlowJo's Acquisition-defined matrix -- this file's own
        $SPILL. The first matrix in the workspace is used only for a file the
        workspace does not list (an FMO or control analysed against it).

        Taking the workspace's FIRST matrix for every file would compensate,
        in a workspace whose samples carry different matrices, all but one
        with someone else's, and a workspace with no matrix at all would
        raise instead of using each file's $SPILL as FlowJo does."""
        reader = WspReader(wsp_path)
        if matrix_name is not None:
            m = reader.get_matrix(matrix_name)
            self._apply_comp(m['matrix'], m['channels'],
                             source=f'wsp:{matrix_name}')
            return self
        listed, m = reader.matrix_for_fcs(self.path)
        if listed:
            if m is None:
                log.info("  [%s] the workspace gives this sample no matrix of "
                         "its own -- FlowJo's Acquisition-defined ($SPILL) "
                         "applies.", self.name)
                return self.auto_compensate()
            self._apply_comp(m['matrix'], m['channels'],
                             source=f"wsp:{m.get('name', 'sample')}")
            return self
        m = reader.get_matrix()          # raises when the workspace has none
        distinct = reader.distinct_matrix_count()
        if distinct > 1:
            log.warning(
                "  [!] %s is not a sample of %s, which holds %d different "
                "matrices -- applying the first ('%s'). Pass matrix_name= to "
                "choose.", os.path.basename(str(self.path)),
                os.path.basename(str(wsp_path)), distinct,
                m.get('name', '?'))
        self._apply_comp(m['matrix'], m['channels'],
                         source=f"wsp:{m.get('name', 'first')}")
        return self

    def manual_compensate(self, matrix: ArrayLike,
                          channels: Iterable[str]) -> FlowSample:
        """Apply a user-supplied spillover `matrix` over `channels` (rows =
        source fluorochrome, columns = detector). Replaces any matrix applied
        earlier."""
        self._apply_comp(matrix, channels)
        return self

    def recompensate(self, matrix: ArrayLike, channels: Iterable[str],
                     source: str = 'manual') -> int:
        """Replace the compensation this sample's events carry with `matrix`
        (spillover, source -> destination) over `channels`, keeping every
        event and every other column. Returns the number of events that now
        have an unknown value they did not have before.

        For the channels of the old and the new matrix it undoes each
        channel's recorded display transform (``data_transforms``), puts the
        recorded spillover (``comp_matrix``) back to get the measured values,
        un-mixes those with the new matrix, and re-applies each channel's own
        transform. A fresh load of the FCS would give the same values but
        none of the computed columns, and the whole file where the sample
        was downsampled.

        A value with no linear equivalent (NaN: <= 0 on a 'log' channel)
        spoils only the outputs it feeds. Raises UnknownScaleError when a
        channel's display scale is unknown, ValueError when a cell of the
        matrix is NaN/inf, no channel of the matrix is in the data or a
        recorded comp channel is missing, and
        numpy.linalg.LinAlgError for a singular matrix, CompensationError
        when the matrix's shape does not match `channels`.

        The compensation record (``comp_matrix``, ``comp_channels``,
        ``compensation_source``) is the one _apply_comp keeps: the old
        matrix is undone the same way (``compensated @ M = measured``), here
        through each channel's transform, which _apply_comp refuses."""
        matrix   = np.asarray(matrix, dtype=float)
        channels = [str(c) for c in channels]
        n = len(channels)
        # The same checks as _apply_comp: a matrix that does not fit its
        # channel list is refused, and a percent matrix becomes fractions.
        if matrix.ndim != 2 or matrix.shape != (n, n):
            raise CompensationError(
                f"spillover matrix is {matrix.shape} but names {n} "
                f"channel(s): {channels}")
        # A NaN/inf cell inverts to an all-NaN matrix without raising, which
        # would wipe every value; and the NaN record could never be undone.
        bad = non_finite_cells(matrix, channels)
        if bad:
            raise ValueError(f"the matrix has a cell that is not a finite "
                             f"number ({bad})")
        matrix = _spill_units_to_fractions(
            matrix, f"{self.name}: the {source} spillover matrix")
        idx = [i for i, c in enumerate(channels) if c in self.data.columns]
        avail = [channels[i] for i in idx]
        if not avail:
            raise ValueError("none of the matrix's channels are in the data")
        if len(avail) < n:
            log.warning(
                "  [!] %s: %d of %d spillover channel(s) are not in the data "
                "(%s) -- compensating the other %d only.", self.name,
                n - len(avail), n,
                ', '.join(c for c in channels if c not in avail), len(avail))
        sub = matrix[np.ix_(idx, idx)]
        inv = np.linalg.inv(sub)
        old_ch = list(self.comp_channels or [])
        missing = [c for c in old_ch if c not in self.data.columns]
        if missing:
            raise ValueError(
                f"the compensation recorded for {missing} cannot be undone: "
                "the channel is no longer in the data")
        old_m = (None if self.comp_matrix is None
                 else np.asarray(self.comp_matrix, dtype=float))
        if old_ch and (old_m is None or old_m.shape != (len(old_ch),) * 2):
            raise ValueError("the compensation recorded for this sample has "
                             "no matching matrix, so it cannot be undone")
        involved = list(dict.fromkeys([*old_ch, *avail]))
        pos = {c: j for j, c in enumerate(involved)}
        lin = np.column_stack([linear_values(self, c) for c in involved])
        measured = lin.copy()
        if old_ch:
            j = [pos[c] for c in old_ch]
            measured[:, j] = _matmul_known(lin[:, j], old_m)
        comp = measured.copy()
        j = [pos[c] for c in avail]
        comp[:, j] = _matmul_known(measured[:, j], inv)
        before = np.isfinite(self.data[involved].to_numpy(dtype=float))
        recs = self.data_transforms or {}
        data = self.data.copy(deep=False)
        for c in involved:
            spec = recs.get(c)
            v = comp[:, pos[c]]
            data[c] = (transform_values(v, **spec)
                       if spec and spec.get('method', 'linear') != 'linear'
                       else v)
        after = np.isfinite(data[involved].to_numpy(dtype=float))
        # A new frame, not an in-place write: caches keyed by the data's
        # identity (auto-clean masks) must see that the values changed.
        self.data = data
        self.comp_matrix = sub.copy()
        self.comp_channels = list(avail)
        self.compensation_source = source
        log.info(f"  Re-compensated ({source}): {avail}")
        return int((before & ~after).any(axis=1).sum())

    def is_compensated(self) -> bool:
        """True when a spillover matrix is currently applied to `data`."""
        return (getattr(self, 'comp_matrix', None) is not None
                and bool(getattr(self, 'comp_channels', None)))

    def _transformed_channels(self) -> dict[str, dict[str, Any]]:
        """{channel: transform_spec} for channels whose stored values are not
        linear (``data_transforms``; unknown-scale entries included)."""
        return {ch: spec for ch, spec in
                (getattr(self, 'data_transforms', None) or {}).items()
                if spec and spec.get('method', 'linear') != 'linear'}

    def _apply_comp(self, matrix: ArrayLike, channels: Iterable[str],
                    source: str = 'manual') -> None:
        matrix   = np.asarray(matrix, dtype=float)
        channels = [str(c) for c in channels]
        n = len(channels)
        if matrix.ndim != 2 or matrix.shape != (n, n):
            raise CompensationError(
                f"spillover matrix is {matrix.shape} but names {n} "
                f"channel(s): {channels}")
        bad = non_finite_cells(matrix, channels)
        if bad:
            raise CompensationError(
                f"{self.name}: the {source} spillover matrix has a cell that "
                f"is not a finite number ({bad}); it would make every "
                "compensated value unknown")
        # Every route in (a .wsp sample's matrix, manual_compensate, a staged
        # session matrix) gets the readers' unit check: a percent matrix
        # applied as fractions leaves every channel at 0.01x.
        matrix = _spill_units_to_fractions(
            matrix, f"{self.name}: the {source} spillover matrix")
        idx   = [i for i, c in enumerate(channels) if c in self.data.columns]
        avail = [channels[i] for i in idx]
        if not avail:
            log.warning("  [!] No matching channels for compensation.")
            return
        missing = [c for c in channels if c not in self.data.columns]
        if missing:
            # Spillover FROM a missing detector cannot be removed, so the
            # remaining channels are only partly corrected. Not fatal (a
            # matrix written for a larger panel is common), but never silent.
            log.warning(
                "  [!] %s: %d of %d spillover channel(s) are not in the data "
                "(%s) -- compensating the other %d only.", self.name,
                len(missing), n, ', '.join(missing), len(avail))
        prev_chans = list(getattr(self, 'comp_channels', None) or [])
        done = self._transformed_channels()
        on_curve = sorted(c for c in set(avail) | set(prev_chans) if c in done)
        if on_curve:
            raise CompensationError(
                f"{self.name}: cannot compensate {on_curve} -- already "
                "transformed. Compensation is linear and must be applied "
                "before apply_transform().")
        sub = matrix[np.ix_(idx, idx)]
        from . import gpu_accel
        try:
            # Loader threads run this concurrently; see gpu_accel.cpu_blas_lock.
            with gpu_accel.cpu_blas_lock():
                inv = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            log.warning("  [!] Spillover matrix is singular — "
                        "compensation skipped%s.",
                        ' (the matrix applied earlier is kept)'
                        if self.is_compensated() else '')
            return
        if self.is_compensated():
            # Undo the matrix applied earlier (compensated @ M = measured)
            # so the new one starts from the measured values. Exact float64
            # on purpose: this restores data, it is not the hot load path.
            prev = [c for c in prev_chans if c in self.data.columns]
            if prev != prev_chans:
                raise CompensationError(
                    f"{self.name}: the previously compensated channel(s) "
                    f"{sorted(set(prev_chans) - set(prev))} are gone from the "
                    "data, so the earlier compensation cannot be undone.")
            prev_m = np.asarray(self.comp_matrix, dtype=float)
            with gpu_accel.cpu_blas_lock():
                undone = np.asarray(self.data[prev].values, dtype=float) @ prev_m
            self.data[prev] = undone
            log.info("  [%s] replacing the compensation applied earlier (%s).",
                     self.name, getattr(self, 'compensation_source', None)
                     or 'manual')
        # Spillover is source->dest (measured = true @ M), so un-mixing is
        # `data @ inv(M)` — NO transpose. (gpu_accel.compensate computes
        # `values @ arg`.) The historical `inv.T` here silently left asymmetric
        # spillover uncorrected AND corrupted clean channels — i.e. every real
        # compensated dataset. See test_compensation_recovers_true_signal.
        self.data[avail] = gpu_accel.compensate(self.data[avail].values, inv)
        # Persist the matrix that was ACTUALLY applied (post-channel-
        # intersection, so it matches the channels in self.data) so the
        # GUI / CLI workspace export can round-trip it.
        self.comp_matrix   = np.asarray(sub, dtype=float).copy()
        self.comp_channels = list(avail)
        self.compensation_source = source
        log.info(f"  Compensation applied ({source}): {avail}")

    def _parse_spillover(self) -> dict[str, Any] | None:
        """The file's SPILL / $SPILLOVER matrix as ``{'channels', 'matrix'}``,
        or None. The same keyword choice and parser as
        read_compensation_matrix (channels written as parameter numbers are
        mapped to $PnN; a percent matrix becomes fractions), so
        auto_compensate applies exactly what the compensation editor and the
        workspace export read from this file."""
        spill = _spill_text(self.metadata)
        if not spill:
            return None
        try:
            chans, matrix = _parse_spill_keyword(
                spill, _fcs_param_names(self.metadata, self.channel_names),
                self.name, markers=_fcs_param_markers(self.metadata))
        except CompensationError as e:
            # Includes a NaN/inf cell: the sample loads uncompensated with
            # this warning (_apply_comp refuses one handed to it explicitly).
            log.warning("  [!] Malformed $SPILL ignored (not treated as 'no "
                        "spillover'): %s", e)
            return None
        return {'channels': chans, 'matrix': matrix}

    # ── Transform ─────────────────────────────────────────────────────────────

    def apply_transform(self, channels: Iterable[str] | None = None,
                        method: str = 'logicle',
                        t: float = LOGICLE_T, m: float = LOGICLE_M,
                        w: float = LOGICLE_W, a: float = LOGICLE_A,
                        cofactor: float = ASINH_DEFAULT_COFACTOR) -> FlowSample:
        """Transform ``channels`` (default: the fluorescence channels) in place; returns self.

        Raises ValueError if any channel already carries a non-linear transform (use
        ``retransform``). A channel that fails is logged and left in raw scale."""
        if channels is None:
            channels = self.fluor_channels
        avail = [c for c in channels if c in self.data.columns]
        failed = []
        spec = transform_spec(method, t=t, m=m, w=w, a=a, cofactor=cofactor)
        done = dict(getattr(self, 'data_transforms', None) or {})
        # This transforms the values that are there. On a channel that already
        # carries a transform it would stack a second one on the first and
        # record only the second (measured: linear_values 0.446 where 924 is
        # true); 'linear' would drop the record and leave the data as it is.
        # Refuse before touching anything -- retransform undoes first.
        stacked = [str(c) for c in avail
                   if (done.get(c) or {}).get('method', 'linear') != 'linear']
        if stacked:
            raise ValueError(
                f"apply_transform: {', '.join(stacked)} already carr"
                f"{'ies' if len(stacked) == 1 else 'y'} a transform; use "
                "retransform() to change a channel's method.")
        for ch in avail:
            vals = np.asarray(self.data[ch].values, dtype=float).copy()
            try:
                self.data[ch] = transform_values(vals, **spec)
            except Exception as e:  # noqa: BLE001 - logged and listed as failed; the other channels still transform
                log.warning(f"  [!] Transform failed {ch}: {e}")
                failed.append(str(ch))
                continue
            if method == 'linear':
                done.pop(ch, None)
            else:
                done[ch] = spec
        self.data_transforms = done
        # Record the failed channels every call (empty when all succeeded, so a
        # later successful re-transform clears a prior failure) for any consumer
        # that wants to flag them.
        self._transform_failed = failed
        if failed:
            # A channel left in RAW scale among logicle-transformed siblings is a
            # silent trap: gates/plots authored in transformed space compare
            # against raw values and select the wrong events. Surface it loudly.
            log.error(
                "  [!] %d channel(s) left in RAW scale (transform failed): %s — "
                "gates/plots on them will be WRONG until re-transformed.",
                len(failed), ', '.join(failed))
        log.info(f"  {method} transform applied to {len(avail) - len(failed)} "
                 f"channel(s).")
        return self

    def retransform(self, channels: Iterable[str], method: str = 'logicle',
                    t: float = LOGICLE_T, m: float = LOGICLE_M,
                    w: float = LOGICLE_W, a: float = LOGICLE_A,
                    cofactor: float = ASINH_DEFAULT_COFACTOR) -> FlowSample:
        """Put each of `channels` on `method`, first undoing the transform its
        data already carries (``data_transforms``). apply_transform instead
        transforms whatever values are there, so calling it on a logicle
        channel would stack a second transform on the first. A channel whose
        transform is unknown cannot be undone and is left as stored."""
        spec = transform_spec(method, t=t, m=m, w=w, a=a, cofactor=cofactor)
        done = dict(getattr(self, 'data_transforms', None) or {})
        for ch in channels:
            if ch not in self.data.columns:
                continue
            if is_unknown_spec(done.get(ch)):
                log.warning(f"  [!] {ch}: display scale unknown, left as "
                            "stored (not re-transformed).")
                continue
            try:
                lin = to_linear(self.data[ch].values, done.get(ch))
                self.data[ch] = transform_values(lin, **spec)
            except Exception as e:  # noqa: BLE001 - logged; that channel stays as it was, the others re-transform
                log.warning(f"  [!] Re-transform failed {ch}: {e}")
                continue
            if is_guessed_spec(done.get(ch)):
                # Re-transformed from guessed linear values: still a guess.
                done[ch] = guessed_spec(spec)
            elif method == 'linear':
                done.pop(ch, None)
            else:
                done[ch] = spec
        self.data_transforms = done
        return self

    def mark_transformed(self, channels: Iterable[str] | None = None,
                         method: str = 'logicle',
                         t: float = LOGICLE_T, m: float = LOGICLE_M,
                         w: float = LOGICLE_W, a: float = LOGICLE_A,
                         cofactor: float = ASINH_DEFAULT_COFACTOR) -> FlowSample:
        """Record that `channels` (default: the fluor channels) ALREADY hold
        `method`-transformed values, without touching them -- for data that
        was transformed before it reached this sample, such as a processed CSV
        exported from transformed data."""
        if channels is None:
            channels = self.fluor_channels
        spec = transform_spec(method, t=t, m=m, w=w, a=a, cofactor=cofactor)
        done = dict(getattr(self, 'data_transforms', None) or {})
        for ch in channels:
            if ch not in self.data.columns:
                continue
            if method == 'linear':
                done.pop(ch, None)
            else:
                done[ch] = spec
        self.data_transforms = done
        return self

    # ── FMO gating ────────────────────────────────────────────────────────────

    def apply_threshold_gates(self, thresholds: Mapping[str, float]) -> FlowSample:
        """
        Add boolean '<channel>_pos' columns from FMO-derived thresholds.
        thresholds : {channel_name: cutoff_value}
        """
        self.thresholds = thresholds
        for ch, val in thresholds.items():
            col = self._resolve(ch)
            if col in self.data.columns:
                # At or above the cut, as a 'threshold' gate counts: a
                # --gates threshold lands here, and --export-wsp writes it
                # back to FlowJo as a min-only gate (min <= x).
                self.data[f'{col}_pos'] = self.data[col] >= val
                pct = self.data[f'{col}_pos'].mean() * 100
                log.info(f"  Gate {col} >= {val:.2f}  ->  {pct:.1f}% positive")
        return self

    def apply_region_gates(
            self, gates: Sequence[Mapping[str, Any]] | None) -> FlowSample:
        """Filter `self.data` to events satisfying every (enabled) gate in
        `gates`, combined via logical AND.

        Memory-aware: we maintain one bool `keep` mask over the original
        rows and evaluate each gate ONLY against the still-active rows
        (`np.where(keep)`). The DataFrame is sliced exactly once at the
        end, so we never pay the cost of duplicating wide intermediate
        DataFrames. Polygon gates use float32 coords so a 10 M-event
        sample's pts array is 80 MB instead of 160 MB; subsequent
        polygons after earlier gates have narrowed the data are much
        smaller still.

        Disabled gates (gate['enabled'] is False) are skipped.

        Call AFTER `apply_transform()` so gate coordinates (typically
        logicle) match the data scale.
        """
        if not gates:
            return self
        active = [g for g in gates if g.get('enabled', True)]
        if not active:
            return self

        n_total = len(self.data)
        keep = np.ones(n_total, dtype=bool)
        per_gate = []                  # (label, kept_after_this_gate)
        # The run log names each gate by the intensities it cuts at, as the
        # editor's gate list does, not by its stored logicle coordinates.
        tf = dict(getattr(self, 'data_transforms', None) or {})
        # Boolean gates reference other gates by id, and we have the whole
        # list right here — without it they cannot resolve their operands,
        # and admitting every event would turn a "NOT X" population into the
        # entire sample. Ancestors are included (not just `active`) so a
        # disabled parent still defines the population, as cumulative_gate_mask
        # documents.
        gates_by_id = {g['id']: g for g in gates if g.get('id') is not None}

        for g in active:
            n_active = int(keep.sum())
            if n_active == 0:
                per_gate.append((describe_gate(g, transforms=tf), 0))
                continue
            active_idx = np.where(keep)[0]
            sub_mask = self._evaluate_gate_on(g, active_idx, gates_by_id)
            # Write the gate's verdict back into the full-length keep
            # mask: rows the gate excluded become False; rows it kept
            # stay True (they were already True).
            keep[active_idx] = sub_mask
            per_gate.append((describe_gate(g, transforms=tf), int(keep.sum())))

        kept = int(keep.sum())
        pct  = (kept / n_total * 100.0) if n_total else 0.0
        self.data = cast(pd.DataFrame, self.data[keep]).reset_index(drop=True)
        log.info(
            "  [Gates] Kept %s / %s events (%.1f%%) after %d region gate(s):",
            f"{kept:,}", f"{n_total:,}", pct, len(active))
        for lbl, n in per_gate:
            this_pct = (n / n_total * 100.0) if n_total else 0.0
            # ASCII-only — stdout may be cp1252 in library callers.
            log.info(f"    -> {n:>10,d} ({this_pct:5.1f}%)   {lbl}")
        return self

    def _evaluate_gate_on(
            self, gate: Mapping[str, Any], active_idx: np.ndarray,
            gates_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> np.ndarray:
        """Evaluate one gate against `self.data` restricted to the rows at
        `active_idx`. Returns a bool mask of length len(active_idx).

        Allocates ONLY the columns the gate actually touches (and only the
        active rows of them), so memory scales with how much earlier
        gates have already narrowed the candidate set, not with the
        original sample size.
        """
        kind = gate.get('kind')
        n_act = len(active_idx)

        def _col(ch, dtype):
            if ch not in self.data.columns:
                return None
            # Pull only the active rows of this one column, in compact dtype.
            return np.asarray(self.data[ch].values[active_idx], dtype=dtype)

        if kind == 'threshold':
            vals = _col(gate['channel'], np.float64)
            if vals is None:
                log.warning(
                    "  [gate] threshold: channel %r not in data — skipped",
                    gate['channel'])
                return np.ones(n_act, dtype=bool)
            # At or above, matching gate_to_mask -- see there for why.
            return vals >= float(gate['value'])

        if kind == 'interval':
            vals = _col(gate['channel'], np.float64)
            if vals is None:
                log.warning(
                    "  [gate] interval: channel %r not in data — skipped",
                    gate['channel'])
                return np.ones(n_act, dtype=bool)
            # Half-open, matching gate_to_mask -- see there for why.
            return (vals >= float(gate['lo'])) & (vals < float(gate['hi']))

        if kind == 'rect':
            xs = _col(gate['x_channel'], np.float64)
            ys = _col(gate['y_channel'], np.float64)
            if xs is None or ys is None:
                log.warning(
                    "  [gate] rect: channel(s) %r / %r missing — skipped",
                    gate.get('x_channel'), gate.get('y_channel'))
                return np.ones(n_act, dtype=bool)
            # Half-open, matching gate_to_mask -- see there for why.
            return ((xs >= float(gate['x0'])) & (xs < float(gate['x1'])) &
                    (ys >= float(gate['y0'])) & (ys < float(gate['y1'])))

        if kind == 'polygon':
            xs = _col(gate['x_channel'], np.float32)
            ys = _col(gate['y_channel'], np.float32)
            if xs is None or ys is None:
                log.warning(
                    "  [gate] polygon: channel(s) %r / %r missing — skipped",
                    gate.get('x_channel'), gate.get('y_channel'))
                return np.ones(n_act, dtype=bool)
            verts = np.asarray(gate['vertices'], dtype=np.float32)
            if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
                log.warning(
                    "  [gate] polygon: malformed vertices (shape=%s) — skipped",
                    verts.shape)
                return np.ones(n_act, dtype=bool)
            pts = np.column_stack([xs, ys])
            result = _points_in_polygon(verts, pts)
            del pts, xs, ys     # release the temps before the next gate
            return result

        # Kinds without a memory-lean path here (ellipsoid / cluster / category /
        # boolean / autoclean): delegate to the full gate_to_mask on just the
        # active rows, rather than silently passing everything through.
        sub = self.data.iloc[active_idx] if n_act else self.data.iloc[:0]
        try:
            return np.asarray(gate_to_mask(gate, sub, gates_by_id),
                              dtype=bool)
        except Exception as exc:                       # noqa: BLE001 - a gate that errors fails closed (admits nothing) with a logged warning, never all-True
            # A gate that ERRORS out must NOT silently pass every event through
            # (all-True) — that inflates the population with a superset and reads
            # as success. Fail CLOSED (admit nothing) so the failure is loud (the
            # population empties) rather than silently wrong. This differs from
            # the "channel missing → skip" cases above, which are a gate that
            # legitimately doesn't apply to this sample.
            log.warning("  [gate] kind %r via gate_to_mask FAILED (%s) — admitting "
                        "no events (fail-closed)", kind, exc)
            return np.zeros(n_act, dtype=bool)

    # ── Clustering ────────────────────────────────────────────────────────────

    def cluster(self, channels: Iterable[str] | None = None, k: int = 30,
                n_jobs: int = 1, max_events: int | None = None,
                use_gpu: bool | str = 'auto', vram_admission_gb: float = 1.0,
                random_state: int = 42, reproducible: bool = True,
                resolution: float = 1.0) -> FlowSample:
        """Phenograph-style clustering.

        channels
            None (default): one column per marker, see
            ``default_cluster_channels``. ``run_leiden``, ``run_louvain``,
            ``run_flowsom`` and the embeddings use the same default.

        max_events
            If set and the sample exceeds it, cluster a random sub-sample and
            assign the remaining events to their nearest labelled neighbour
            via a KD-tree (caps peak memory regardless of sample size).

        use_gpu
            'auto' (default): use the GPU clustering path when RAPIDS is
            available *and* free VRAM ≥ `vram_admission_gb`; otherwise CPU.
            True : force GPU (fall back to CPU only on exception).
            False: never attempt GPU.

        vram_admission_gb
            Minimum free VRAM (GB) required to take the GPU branch.

        reproducible
            **On by default.** PhenoGraph's Louvain community
            detection is NOT seed-reproducible: it shells out to the Blondel
            reference binaries, which seed with
            ``srand(time(NULL) + getpid())`` and use that RNG to shuffle the
            node traversal order. The PID half is assigned by the OS, so no
            injected clock can pin it -- 20 runs on identical input inside one
            second gave 20 distinct partitions. Measured on 50k events with
            overlapping populations: two runs of the default agreed only to
            ARI 0.76, while the seeded Leiden backend was bit-identical.
            An analysis that changes when you re-run it is hard to defend in a
            methods section, so the reproducible path is the default.

            With True, clustering uses PhenoGraph's *seeded Leiden* backend on
            the CPU path — identical input + ``random_state`` → identical
            labels. It was also ~1.5x faster in that measurement (63s vs 95s).

            Two consequences to be aware of:

            * **Labels differ from a Louvain run.** Leiden is the corrected
              form of Louvain (it cannot emit internally disconnected
              communities), not a reimplementation of it: on ambiguous data the
              two agreed only to ARI 0.74, with 7 clusters against 9. Accuracy
              against planted ground truth was equivalent (0.4905 vs 0.4990).
              Set ``reproducible=False`` to reproduce prior Louvain output or
              to match published PhenoGraph results.
            * **The GPU path is skipped**, because cuGraph's Louvain is
              likewise unseeded. ``reproducible=False`` restores it.

        resolution
            Granularity of the community detection: lower merges clusters
            (fewer, larger), higher splits them (more, smaller). 1.0, the
            default, is the standard PhenoGraph result: no resolution is
            passed, so phenograph uses its own. PhenoGraph over-splits data with
            few markers: four well-separated populations in three channels
            (300 events each) came out as 8 clusters at 1.0 and as the 4
            populations at 0.5. Only the reproducible (seeded Leiden) route
            has a resolution: PhenoGraph's Louvain runs the Blondel binaries,
            which take none, so with ``reproducible=False`` (and on the GPU
            Louvain path) a value other than 1.0 is not applied and a warning
            says so. ``self.cluster_settings`` records the resolution that
            was applied (None when it was not).

        The GPU branch uses cuML NearestNeighbors + cuGraph Louvain on the
        kNN graph. Any GPU failure (OOM, CUDA error, missing kit) falls back
        to CPU Phenograph for the same call — the result is always written.
        """
        resolution = _checked_resolution(resolution)
        if channels is None:
            channels = default_cluster_channels(self.fluor_channels)
        avail = [c for c in channels if c in self.data.columns]
        X     = self.data[avail].values.astype(float)
        mask  = np.all(np.isfinite(X), axis=1)
        Xc    = X[mask]
        n     = len(Xc)

        # Pick the events that will actually be clustered (full or subsample).
        sub_idx, X_cluster = _cluster_subsample(Xc, max_events, random_state)
        size_label = (f"{max_events:,} (subsample of {n:,})"
                      if sub_idx is not None else f"{n:,}")
        shape = f"{size_label} × {len(avail)}, k={k}"

        sub_comm, Q, backend, route = None, -1.0, 'CPU', 'louvain'
        if _should_try_gpu(reproducible, use_gpu, vram_admission_gb):
            sub_comm, Q = self._try_cluster_gpu(X_cluster, k, shape)
            if sub_comm is not None:
                backend = 'GPU'
        if sub_comm is None:
            sub_comm, Q, route = _run_phenograph_cpu(
                X_cluster, k, n_jobs, reproducible, random_state,
                resolution, shape)

        self.cluster_settings = _phenograph_settings(
            k, route, backend, random_state, resolution)

        # Expand sub-sample labels back to all events (KD-tree assign for rest).
        communities = (sub_comm if sub_idx is None else
                       _expand_subsample_labels(Xc, X_cluster, sub_idx,
                                                sub_comm))

        self._store_cluster_labels(mask, communities)
        valid = communities[communities >= 0]
        log.info(
            "  → %d clusters, Q=%.3f [%s%s]",
            len(np.unique(valid)) if len(valid) else 0, Q, backend,
            f", resolution {resolution:g}" if route == 'leiden' else '')
        return self

    def _store_cluster_labels(self, mask: np.ndarray,
                              communities: np.ndarray) -> None:
        """Write ``communities`` to the rows where ``mask`` is True, -1
        elsewhere, as ``data['cluster']`` and ``self.clusters``."""
        labels       = np.full(len(self.data), -1, dtype=int)
        labels[mask] = communities
        self.data['cluster'] = labels
        self.clusters        = labels

    def _try_cluster_gpu(self, X_cluster: np.ndarray, k: int,
                         shape: str) -> tuple[np.ndarray | None, float]:
        """Run ``_cluster_gpu``; on any failure log it and return
        ``(None, -1.0)`` so the caller falls back to the CPU path."""
        try:
            log.info(f"  GPU cluster: {shape} …")
            return self._cluster_gpu(X_cluster, k)
        except Exception as e:  # noqa: BLE001 - logged; any GPU failure falls back to CPU clustering
            log.warning(
                "  [!] GPU clustering failed (%s: %s) — CPU fallback",
                type(e).__name__, e)
            return None, -1.0

    def _cluster_gpu(self, X: np.ndarray,
                     k: int) -> tuple[np.ndarray, float]:
        """GPU clustering: cuML kNN graph + cuGraph Louvain community detection.

        Returns (communities ndarray of length len(X), modularity float).
        Raises on any RAPIDS / VRAM failure — the caller is responsible
        for catching and falling back to CPU.
        """
        kit     = _GPU_CLUSTER_KIT
        # caller only invokes us when GPU_CLUSTERING_AVAILABLE is True,
        # which implies kit is non-None — assert so pyright can narrow.
        assert kit is not None
        cupy    = kit['cupy']
        cudf    = kit['cudf']
        cugraph = kit['cugraph']
        CuNN    = kit['cunn']

        n     = X.shape[0]
        X_gpu = cupy.asarray(X, dtype=cupy.float32)

        # k+1 neighbours so we can drop the self-link (column 0).
        nn = CuNN(n_neighbors=k + 1)
        nn.fit(X_gpu)
        _, indices = nn.kneighbors(X_gpu)

        if hasattr(indices, 'values'):          # cuDF DataFrame on some versions
            idx_gpu = cupy.asarray(indices.values)
        else:
            idx_gpu = cupy.asarray(indices)
        idx_gpu = idx_gpu[:, 1:]                 # drop self

        src = cupy.repeat(cupy.arange(n, dtype=cupy.int32), k)
        dst = idx_gpu.flatten().astype(cupy.int32)

        edges = cudf.DataFrame({'src': src, 'dst': dst})
        G = cugraph.Graph()
        G.from_cudf_edgelist(edges, source='src', destination='dst', renumber=False)

        parts, modularity = cugraph.louvain(G)
        parts_sorted = parts.sort_values('vertex')
        communities  = parts_sorted['partition'].to_numpy().astype(np.int64)
        return communities, float(modularity)

    # ── UMAP ──────────────────────────────────────────────────────────────────

    def _embedding_input(
            self, channels: Iterable[str] | None, sample_n: int | None,
            random_state: int | None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Shared front-end for the dimensionality-reduction backends
        (UMAP / TriMap / PaCMAP).

        Returns ``(X, sub_mask, avail)`` where ``X`` is the full float
        matrix of the available channels, ``sub_mask`` is a boolean row
        selector (finite rows, optionally sub-sampled to ``sample_n``),
        and ``avail`` is the list of channels actually used. Pure: no
        embedding is computed and nothing is written back.
        """
        if channels is None:
            channels = default_cluster_channels(self.fluor_channels)
        avail = [c for c in channels if c in self.data.columns]
        X     = self.data[avail].values.astype(float)
        mask  = np.all(np.isfinite(X), axis=1)

        if sample_n and mask.sum() > sample_n:
            idx      = np.where(mask)[0]
            chosen   = np.random.default_rng(random_state).choice(
                           idx, sample_n, replace=False)
            sub_mask = np.zeros(len(X), dtype=bool)
            sub_mask[chosen] = True
        else:
            sub_mask = mask
        return X, sub_mask, avail

    def _store_embedding(self, emb: Any, sub_mask: np.ndarray,
                         prefix: str) -> np.ndarray:
        # emb is Any: pacmap's untyped fit_transform reads as possibly None.
        """Write a 2-D embedding back as ``{prefix}1`` / ``{prefix}2``
        columns, NaN where the row wasn't embedded. Returns the embedding."""
        emb = np.asarray(emb)
        self.data[f'{prefix}1'] = np.nan
        self.data[f'{prefix}2'] = np.nan
        self.data.loc[sub_mask, f'{prefix}1'] = emb[:, 0]
        self.data.loc[sub_mask, f'{prefix}2'] = emb[:, 1]
        return emb

    def run_umap(self, channels: Iterable[str] | None = None,
                 n_neighbors: int = 30, min_dist: float = 0.3,
                 sample_n: int | None = 100_000,
                 random_state: int = 42) -> FlowSample:
        """Compute a seeded UMAP embedding (GPU when available, else CPU); returns self.

        The coordinates go into ``self.umap_coords``. Without umap-learn on the CPU path
        it logs a warning and returns self unchanged."""
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)

        log.info(f"  UMAP: {sub_mask.sum():,} events × {len(avail)} channels …")
        emb = None

        if GPU_AVAILABLE:
            try:
                import cudf  # type: ignore[import-not-found]
                assert _CumlUMAP is not None  # gated above by GPU_AVAILABLE
                X_gpu = cudf.DataFrame(X[sub_mask].astype(np.float32))
                emb   = np.array(
                    _CumlUMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                              random_state=random_state).fit_transform(X_gpu)
                )
                log.info(f"  UMAP complete [GPU — {GPU_NAME}].")
            except Exception as gpu_err:  # noqa: BLE001 - logged; any GPU failure retries UMAP on the CPU
                log.warning(f"  [!] GPU UMAP failed ({type(gpu_err).__name__}: {gpu_err})")
                log.warning("  [!] Retrying on CPU …")
                emb = None

        if emb is None:
            try:
                import umap as umap_lib
            except ImportError:
                log.warning("  [!] pip install umap-learn")
                return self
            # umap-learn forces n_jobs=1 when random_state is set (because
            # its layout optimisation is a lock-free parallel SGD: threads
            # race on the shared embedding array, and thread interleaving is
            # not an RNG, so no seed can pin it). We want the determinism —
            # same seed → same embedding across runs / samples / restarts —
            # so we take the override and silence its per-call UserWarning.
            warnings.filterwarnings(
                'ignore',
                message='.*n_jobs value.*overridden.*',
                category=UserWarning)

            # Do NOT "recover" the cores by building the kNN yourself with
            # pynndescent(n_jobs=-1) and passing precomputed_knn. It was tried
            # and reverted. pynndescent's parallel NN-descent is deterministic
            # only for a FIXED numba thread count: with random_state=42 on
            # 5000x8, thread counts 1 / 4 / 24 give three DIFFERENT neighbour
            # graphs (each perfectly reproducible at its own count). That makes
            # a seeded embedding machine-dependent — a 24-core workstation, a
            # 4-core laptop and a CI runner would produce different coordinates
            # from identical data and seed, which is exactly the guarantee this
            # function exists to provide. Measured gain was ~10% (119.3s ->
            # 108.7s at 150k events); the price was cross-machine
            # reproducibility of published figures. Bad trade.
            # tests/test_umap_embedding.py::test_embedding_does_not_depend_on
            # _thread_count pins it.
            # Serialized only where numba's threading layer needs it (macOS):
            # see openflo/numba_threads.py.
            with numba_threads.parallel_kernels():
                emb = np.asarray(
                    umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                  random_state=random_state).fit_transform(
                                      X[sub_mask]))
            log.info("  UMAP complete [CPU].")

        self.umap_coords = self._store_embedding(emb, sub_mask, 'UMAP')
        return self

    def run_trimap(self, channels: Iterable[str] | None = None,
                   sample_n: int | None = 100_000, random_state: int = 42,
                   **trimap_kwargs: Any) -> FlowSample:
        """TriMap embedding — a triplet-constraint alternative to UMAP that
        tends to preserve global structure better. CPU-only (the ``trimap``
        package has no GPU path). Writes ``TRIMAP1`` / ``TRIMAP2``.

        Install with ``pip install openflo[embed]`` (or ``pip install
        trimap``). Extra keyword args pass through to ``trimap.TRIMAP``.
        """
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        try:
            import trimap
        except ImportError:
            log.warning("  [!] TriMap not installed — pip install openflo[embed]")
            return self
        log.info(f"  TriMap: {sub_mask.sum():,} events × {len(avail)} channels …")
        with numba_threads.parallel_kernels():
            emb = trimap.TRIMAP(**trimap_kwargs).fit_transform(X[sub_mask])
        self.trimap_coords = self._store_embedding(emb, sub_mask, 'TRIMAP')
        log.info("  TriMap complete [CPU].")
        return self

    def run_pacmap(self, channels: Iterable[str] | None = None,
                   n_neighbors: int | None = None,
                   sample_n: int | None = 100_000, random_state: int = 42,
                   **pacmap_kwargs: Any) -> FlowSample:
        """PaCMAP embedding — another global-structure-preserving
        alternative to UMAP. CPU-only. Writes ``PACMAP1`` / ``PACMAP2``.

        Install with ``pip install openflo[embed]`` (or ``pip install
        pacmap``). Extra keyword args pass through to ``pacmap.PaCMAP``.
        """
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        try:
            import pacmap
        except ImportError:
            log.warning("  [!] PaCMAP not installed — pip install openflo[embed]")
            return self
        log.info(f"  PaCMAP: {sub_mask.sum():,} events × {len(avail)} channels …")
        # n_neighbors=None means "auto" to PaCMAP, but its signature types the
        # parameter as int. The ignore must sit on the ARGUMENT line, not the
        # call line, or pyright still reports it.
        with numba_threads.parallel_kernels():
            reducer = pacmap.PaCMAP(
                n_neighbors=n_neighbors,  # pyright: ignore[reportArgumentType]
                random_state=random_state, **pacmap_kwargs)
            emb = reducer.fit_transform(X[sub_mask])
        self.pacmap_coords = self._store_embedding(emb, sub_mask, 'PACMAP')
        log.info("  PaCMAP complete [CPU].")
        return self

    def run_tsne(self, channels: Iterable[str] | None = None,
                 perplexity: float = 30.0, sample_n: int | None = 50_000,
                 random_state: int = 42, **tsne_kwargs: Any) -> FlowSample:
        """t-SNE embedding (scikit-learn, a core dependency). Writes
        ``TSNE1`` / ``TSNE2``. t-SNE is O(n log n) but still heavier than UMAP,
        so it subsamples to ``sample_n`` events; ``perplexity`` is clamped below
        the sample size. Extra kwargs pass through to ``sklearn.manifold.TSNE``.
        """
        from sklearn.manifold import TSNE
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        n = int(sub_mask.sum())
        if n < 5:
            log.warning("  [t-SNE] too few events — skipped.")
            return self
        perp = float(min(perplexity, max(5.0, (n - 1) / 3.0)))
        log.info(f"  t-SNE: {n:,} events × {len(avail)} channels "
                 f"(perplexity {perp:.0f}) …")
        emb = TSNE(n_components=2, perplexity=perp,
                   random_state=random_state, init='pca',
                   **tsne_kwargs).fit_transform(X[sub_mask])
        self.tsne_coords = self._store_embedding(np.asarray(emb), sub_mask,
                                                 'TSNE')
        log.info("  t-SNE complete [CPU].")
        return self

    def run_phate(self, channels: Iterable[str] | None = None,
                  sample_n: int | None = 50_000, random_state: int = 42,
                  **phate_kwargs: Any) -> FlowSample:
        """PHATE embedding — a diffusion-based method that preserves continuous
        / trajectory structure especially well (complements the trajectory
        tool). Writes ``PHATE1`` / ``PHATE2``. Optional dependency: install with
        ``pip install openflo[embed]`` (or ``pip install phate``). Extra kwargs
        pass through to ``phate.PHATE``."""
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        try:
            import phate
        except ImportError:
            log.warning("  [!] PHATE not installed — pip install openflo[embed]")
            return self
        log.info(f"  PHATE: {sub_mask.sum():,} events × {len(avail)} channels …")
        emb = phate.PHATE(random_state=random_state,
                          verbose=False, **phate_kwargs).fit_transform(
                              X[sub_mask])
        self.phate_coords = self._store_embedding(np.asarray(emb), sub_mask,
                                                  'PHATE')
        log.info("  PHATE complete [CPU].")
        return self

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _resolve(self, name: str) -> str:
        if name in self.data.columns:
            return name
        for det, lbl in self.channel_labels.items():
            if lbl.lower() == name.lower():
                return det
        raise KeyError(f"Channel '{name}' not found.")

    def _intensity_ticker(self, channel: str) -> tuple[Any, Any] | None:
        """``(locator, formatter)`` labelling an axis of `channel`'s stored
        values in intensity (scales.intensity_ticker, 'stored'), or None for
        a channel stored linear or on an unknown scale."""
        from .scales import intensity_ticker
        spec = (getattr(self, 'data_transforms', None) or {}).get(channel)
        return intensity_ticker(spec, 'stored') if spec else None

    def plot(self, x: str, y: str, color_by: str = 'density',
             sample_n: int | None = 50_000, ax: Axes | None = None,
             title: str | None = None, s: float = 1.0,
             alpha: float = 0.5) -> Axes:
        """
        Scatter plot. x/y accept detector name, stain label, UMAP1, UMAP2.
        color_by: 'density' | 'cluster' | channel name/label
        """
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        xch = self._resolve(x)
        ych = self._resolve(y)
        df  = self.data.dropna(subset=[xch, ych])
        if sample_n and len(df) > sample_n:
            df = df.sample(sample_n, random_state=42)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        xv = np.asarray(df[xch].values)
        yv = np.asarray(df[ych].values)

        if color_by == 'cluster' and 'cluster' in df.columns:
            sc = ax.scatter(xv, yv, c=np.asarray(df['cluster'].values),
                            cmap='tab20', s=s, alpha=alpha, linewidths=0)
            plt.colorbar(sc, ax=ax, label='Cluster')
        elif color_by == 'density':
            # FlowJo-style O(n) histogram density: bin events into a
            # 256x256 grid, smooth the bin counts, look up each event's
            # density by its bin index. Replaces a gaussian_kde call
            # that was O(n^2) and took minutes on >100k events.
            try:
                from scipy.ndimage import gaussian_filter
                xv_f = np.asarray(xv, dtype=float)
                yv_f = np.asarray(yv, dtype=float)
                finite = np.isfinite(xv_f) & np.isfinite(yv_f)
                xv_f = xv_f[finite]; yv_f = yv_f[finite]
                if xv_f.size == 0:
                    raise ValueError("no finite points")
                BINS = 256
                hist, x_edges, y_edges = np.histogram2d(xv_f, yv_f, bins=BINS)
                hist = gaussian_filter(hist, sigma=1.5)
                ix = np.clip(np.searchsorted(x_edges, xv_f, side='right') - 1,
                             0, BINS - 1)
                iy = np.clip(np.searchsorted(y_edges, yv_f, side='right') - 1,
                             0, BINS - 1)
                z   = hist[ix, iy]
                idx = z.argsort()
                ax.scatter(xv_f[idx], yv_f[idx], c=z[idx],
                           cmap='turbo', s=s, alpha=alpha, linewidths=0,
                           rasterized=True)
            except Exception:  # noqa: BLE001 - density colour is cosmetic; falls back to a plain scatter
                ax.scatter(xv, yv, s=s, alpha=alpha, color='steelblue')
        else:
            try:
                cch = self._resolve(color_by)
                col = df[cch]
                # Categorical (string) column — e.g. 'sample_origin' on a
                # concatenated dataset. matplotlib can't take string values
                # for `c`; factorise to integer codes and draw a discrete
                # legend instead of a colorbar.
                if (col.dtype == object
                        or pd.api.types.is_string_dtype(col)
                        or isinstance(col.dtype, pd.CategoricalDtype)):
                    codes, uniques = pd.factorize(col, sort=True)
                    n_groups = max(1, len(uniques))
                    # 'auto' is colour-blind safe: Okabe-Ito's first n
                    # colours, then tab20, then evenly spaced hues (see
                    # group_palettes.n_group_colors). tab20 alone alternates
                    # same-hue light/dark pairs, so its first two entries
                    # are both blues.
                    palette_name = getattr(self, '_palette_name', None) \
                                    or _DEFAULT_CATEGORICAL_PALETTE
                    cmap = categorical_cmap(palette_name, n_groups)
                    sc = ax.scatter(xv, yv, c=codes, cmap=cmap,
                                    vmin=0, vmax=max(n_groups - 1, 1),
                                    s=s, alpha=alpha, linewidths=0)
                    import matplotlib.patches as mpatches
                    handles = [
                        mpatches.Patch(color=cmap((i / max(n_groups - 1, 1))
                                                  if n_groups > 1 else 0.0),
                                       label=str(u))
                        for i, u in enumerate(uniques)
                    ]
                    ax.legend(handles=handles, title=color_by, loc='best',
                              fontsize=8, framealpha=0.8)
                else:
                    sc = ax.scatter(xv, yv, c=np.asarray(col.values),
                                    cmap='viridis',
                                    s=s, alpha=alpha, linewidths=0)
                    cbar = plt.colorbar(sc, ax=ax,
                                        label=self.channel_labels.get(cch, cch))
                    ticker = self._intensity_ticker(cch)
                    if ticker is not None:
                        cbar.locator, cbar.formatter = ticker
            except KeyError:
                ax.scatter(xv, yv, s=s, alpha=alpha, color='steelblue')

        ax.set_xlabel(self.channel_labels.get(xch, xch))
        ax.set_ylabel(self.channel_labels.get(ych, ych))
        # The values drawn are STORED: a logicle axis was ticked 0.2 .. 1.0
        # where the intensities are 100 .. 10^5. Tick each transformed axis
        # at round intensities, as the editor's axes are.
        for axis, ch in ((ax.xaxis, xch), (ax.yaxis, ych)):
            ticker = self._intensity_ticker(ch)
            if ticker is not None:
                axis.set_major_locator(ticker[0])
                axis.set_major_formatter(ticker[1])
        ax.set_title(title or self.name)
        plt.tight_layout()
        return ax

    def plot_umap(self, color_by: str = 'cluster', **kwargs: Any) -> Axes:
        """Scatter UMAP1 against UMAP2, coloured by ``color_by``; returns the Axes."""
        return self.plot('UMAP1', 'UMAP2', color_by=color_by, **kwargs)

    def cluster_heatmap(self, channels: Iterable[str] | None = None,
                        label_col: str = 'cluster') -> Axes | None:
        """Heatmap of each cluster's median per channel on the display scale.

        Returns the Axes, or None when ``label_col`` is not in the data."""
        if label_col not in self.data.columns:
            log.warning("  [!] Run .cluster() first.")
            return None
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        if channels is None:
            channels = self.fluor_channels
        avail  = [c for c in channels if c in self.data.columns]
        labels = [self.channel_labels.get(c, c) for c in avail]
        # set_axis avoids the .rename(columns=Mapping) overload that pandas-stubs
        # can't resolve cleanly off a chained groupby().median() expression.
        med    = (self.data.groupby(label_col)[avail].median()
                           .set_axis(labels, axis=1))
        # Label each row by cluster name so the -1 noise/unclustered row is drawn
        # AND clearly named "Unclustered (noise)" rather than a bare -1.
        med.index = [cluster_label(c) for c in med.index]
        _, ax = plt.subplots(figsize=(max(6, len(avail)),
                                      max(4, len(med) * 0.4)))
        import seaborn as sns  # lazy: only when plotting
        # Colours are medians of the stored (display-transformed) values, as
        # cluster heatmaps conventionally are; the colour bar says so, since
        # its 0-1 ticks are not intensities (<name>_stats.csv has the linear
        # medians).
        sns.heatmap(med, cmap='vlag', center=0, ax=ax,
                    linewidths=0.3, linecolor='grey',
                    cbar_kws={'label': 'median, display scale'})
        ax.set_title(f'{self.name} — Cluster Median Expression')
        plt.tight_layout()
        return ax

    # ── Stats & export ────────────────────────────────────────────────────────

    def cluster_frequencies(self, label_col: str = 'cluster') -> pd.DataFrame:
        """One row per cluster in ``label_col``: count, % of total and linear channel medians.

        The noise cluster (-1) is kept and named in a ``population`` column. Returns an
        empty DataFrame when ``label_col`` is not in the data."""
        # ``label_col`` defaults to the canonical 'cluster' column but accepts any
        # per-event label column (e.g. 'leiden', 'flowsom_meta') so each has the
        # same noise-labelled frequency export rather than a bare -1.
        if label_col not in self.data.columns:
            return pd.DataFrame()
        total  = len(self.data)
        labels = self.data[label_col]
        counts = labels.value_counts().sort_index()
        # Medians of the compensated LINEAR values, as the Statistics window
        # reports them. The pipeline logicle-transforms fluor channels before
        # clustering, and the median was taken of those coordinates: two
        # clusters at linear ~800 and ~20,000 were written to <sample>_stats
        # .csv as 0.43 and 0.75. One channel is inverted at a time; a column
        # of unknown display scale has no linear median (NaN).
        chans = [c for c in self.fluor_channels if c in self.data.columns]
        cols = []
        for c in chans:
            try:
                lin = linear_values(self, c)
            except UnknownScaleError:
                lin = np.full(total, np.nan)
            cols.append(pd.Series(lin, index=self.data.index)
                        .groupby(labels).median().reindex(counts.index))
        meds = (pd.concat(cols, axis=1) if cols
                else pd.DataFrame(index=counts.index))
        meds.columns = [f'median_{self.channel_labels.get(c, c)}'
                        for c in chans]
        count_arr = np.asarray(counts.values)
        df = pd.DataFrame({'sample': self.name, 'cluster': counts.index,
                           'count': count_arr,
                           'pct_total': count_arr/total*100}).set_index('cluster')
        df = df.join(meds).reset_index()
        # Keep EVERY cluster (including the -1 noise/unclustered sentinel) but
        # give each an explicit 'population' name, so the noise bucket is
        # unmistakable in the exported CSV / Prism sheet rather than a bare -1 or
        # silently dropped. cluster()'s count log still excludes -1; the noise
        # row here simply carries the "Unclustered (noise)" label.
        df.insert(1, 'population',
                  np.array([cluster_label(c) for c in df['cluster']]))
        return df

    # ── FlowSOM ───────────────────────────────────────────────────────────────

    def run_flowsom(self, channels: Iterable[str] | None = None,
                    grid: tuple[int, int] = (10, 10),
                    n_metaclusters: int = FLOWSOM_DEFAULT_METACLUSTERS,
                    iters: int = 10, max_events: int = 50_000,
                    seed: int = 42) -> FlowSample:
        """FlowSOM clustering: train a SOM over the marker space, assign each
        event to a node, then agglomerate nodes into metaclusters. Writes a
        ``flowsom`` (node id) and ``flowsom_meta`` (metacluster id) column;
        non-finite rows get -1. Stores the model in ``self.flowsom_result``.

        `n_metaclusters` defaults to FLOWSOM_DEFAULT_METACLUSTERS (20), as in
        the Cluster dialog and openflo-run: set it above the number of
        populations you expect, since asked for exactly that many, FlowSOM
        tends to merge a rare population into a neighbour.

        Lighter than the R FlowSOM but the same structure — fast, CPU-only,
        good for very large files. Returns self."""
        channels = channels or default_cluster_channels(self.fluor_channels)
        avail = [c for c in channels if c in self.data.columns]
        if not avail:
            log.warning("  [FlowSOM] no usable channels — skipped.")
            return self
        X = self.data[avail].values.astype(float)
        mask = np.all(np.isfinite(X), axis=1)
        Xf = X[mask]
        if Xf.shape[0] < max(n_metaclusters, np.prod(grid)):
            log.warning("  [FlowSOM] too few events — skipped.")
            return self

        log.info("  FlowSOM: %s events × %d channels, grid %dx%d …",
                 f"{Xf.shape[0]:,}", len(avail), grid[0], grid[1])
        W, _coords = _som_train(Xf, grid=grid, iters=iters,
                                max_events=max_events, seed=seed)
        nodes = _som_assign(Xf, W)
        meta_of_node = _som_metacluster(W, n_metaclusters)
        meta = meta_of_node[nodes]

        node_col = np.full(len(self.data), -1, dtype=int)
        meta_col = np.full(len(self.data), -1, dtype=int)
        node_col[mask] = nodes
        meta_col[mask] = meta
        self.data['flowsom'] = node_col
        self.data['flowsom_meta'] = meta_col
        n_meta = len(np.unique(meta))
        # meta_of_node maps a node to its metacluster, so events this run did
        # not see can be labelled the way FlowSOM labels its own: the
        # metacluster of their best-matching node (_som_assign).
        self.flowsom_result = {
            'grid': grid, 'n_nodes': int(np.prod(grid)),
            'n_metaclusters': int(n_meta), 'channels': avail, 'weights': W,
            'meta_of_node': meta_of_node}
        log.info("  FlowSOM complete → %d nodes, %d metaclusters.",
                 int(np.prod(grid)), n_meta)
        return self

    def run_leiden(self, channels: Iterable[str] | None = None,
                   n_neighbors: int = 30, resolution: float = 1.0,
                   max_events: int | None = 200_000, random_state: int = 42,
                   fast_graph: bool = False) -> FlowSample:
        """Leiden community detection — the current standard for high-dimensional
        spectral cytometry.

        Builds a symmetric k-nearest-neighbour graph over the marker space and
        partitions it with the Leiden algorithm (RBConfiguration objective;
        ``resolution`` tunes granularity — higher gives more, smaller clusters).
        Writes a ``leiden`` column (non-finite rows → -1). Like ``cluster``,
        very large samples are graph-partitioned on a random subsample of up to
        ``max_events`` events and the rest assigned to their nearest labelled
        neighbour (KD-tree). ``channels=None`` clusters one column per marker
        (``default_cluster_channels``). ``n_neighbors`` defaults to 30, as in
        the app, the CLI and ``cluster_jointly`` (15 up to 2.7.0). Requires
        ``igraph`` + ``leidenalg`` (declared dependencies). Returns self."""
        try:
            # igraph is an availability probe here: the graph itself is built
            # by _snn_jaccard_graph, but failing early gives a better message
            # than an ImportError from inside it.
            import igraph  # noqa: F401  # pyright: ignore[reportUnusedImport]
            import leidenalg
        except ImportError as e:
            raise ClusteringError(
                "igraph + leidenalg are required for Leiden clustering "
                "(pip install igraph leidenalg)") from e
        from sklearn.neighbors import NearestNeighbors

        channels = channels or default_cluster_channels(self.fluor_channels)
        avail = [c for c in channels if c in self.data.columns]
        if not avail:
            log.warning("  [Leiden] no usable channels — skipped.")
            return self
        X = self.data[avail].values.astype(float)
        mask = np.all(np.isfinite(X), axis=1)
        Xc = X[mask]
        n = len(Xc)
        if n < 3:
            log.warning("  [Leiden] too few finite events — skipped.")
            return self

        if max_events and n > max_events:
            rng = np.random.default_rng(random_state)
            sub_idx = np.sort(rng.choice(n, max_events, replace=False))
            X_cluster = Xc[sub_idx]
            subsampled = True
        else:
            sub_idx = None
            X_cluster = Xc
            subsampled = False

        k = int(max(1, min(n_neighbors, len(X_cluster) - 1)))
        log.info("  Leiden: %s events × %d channels, k=%d, resolution=%.2f …",
                 f"{len(X_cluster):,}", len(avail), k, resolution)
        g = _snn_jaccard_graph(X_cluster, k, prune=fast_graph)
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights='weight',
            resolution_parameter=float(resolution),
            seed=signed32_seed(random_state))
        sub_comm = np.asarray(part.membership, dtype=int)

        if subsampled:
            communities = np.full(n, -1, dtype=int)
            communities[sub_idx] = sub_comm
            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(
                X_cluster)
            rest_mask = np.ones(n, dtype=bool)
            rest_mask[sub_idx] = False
            if rest_mask.any():
                nbr = tree_kneighbors(nn, Xc[rest_mask], 1)
                communities[rest_mask] = sub_comm[nbr[:, 0]]
        else:
            communities = sub_comm

        labels = np.full(len(self.data), -1, dtype=int)
        labels[mask] = communities
        self.data['leiden'] = labels
        n_clusters = len(np.unique(communities[communities >= 0]))
        log.info("  Leiden complete → %d clusters (resolution=%.2f).",
                 n_clusters, resolution)
        return self

    def run_louvain(self, channels: Iterable[str] | None = None,
                    n_neighbors: int = 30, restarts: int = 5,
                    max_events: int | None = 200_000, random_state: int = 42,
                    fast_graph: bool = False) -> FlowSample:
        """Louvain community detection — deterministic, unlike PhenoGraph's.

        Same algorithm as ``cluster(reproducible=False)``, but it cannot give a
        different answer twice. PhenoGraph shells out to the Blondel reference
        binaries, which seed themselves ``srand(time(NULL) + getpid())``; the
        PID half is assigned by the OS, so that path is unpinnable from
        outside. igraph implements Louvain in-process with a settable RNG, so
        here the seed is ours.

        ``restarts`` reproduces what makes PhenoGraph's Louvain good rather
        than merely fast. A single Louvain run lands in whatever local optimum
        its node ordering leads to, so the reference implementation re-runs it
        and keeps the highest modularity. This does the same with a
        DETERMINISTIC seed sequence (``random_state``, ``random_state + 1``,
        …), so the exploration is preserved and the answer is still identical
        every time.

        One difference worth knowing. PhenoGraph's restart count is ADAPTIVE,
        not fixed: ``phenograph.core.runlouvain(max_runs=100,
        time_limit=2000, tol=1e-3)`` keeps going until modularity has not
        improved for 20 consecutive runs, then stops — capped at 100 runs or
        2000 seconds. So it spends restarts where they help and stops where
        they do not, while ``restarts`` here is a flat count paid in full
        every time. Matching that stopping rule would be a better design than
        any fixed number.

        Re-measured 2026-09-09 on 20,000 events x 10 channels, 8 overlapping
        populations (centres uniform in [3.5, 6.5], sizes Dirichlet(0.6),
        noise sd 1.4, seed 0), k=30, on a 24-core Windows box:

            restarts=1     24.1s   ARI 0.4880   4 clusters
            restarts=5    121.0s   ARI 0.4867   4 clusters   <- default
            restarts=20   496.5s   ARI 0.4881   4 clusters
            restarts=40  1019.4s   ARI 0.4929   4 clusters

            PhenoGraph binary  41.6s   ARI 0.3784   11 clusters, NOT reproducible

        What holds: the result is identical across runs (verified), and it
        scores better than the binary against the planted truth (0.488 vs
        0.378).

        Read the sweep honestly: ARI barely moves across it. 1, 5 and 20 sit
        within 0.0014 of each other — restarts=1 actually edged restarts=5,
        which is noise, not a finding — and only 40 pulls clearly ahead, at
        forty times the cost of 1. So no restart count in this range is a
        measured optimum, and the sweep has NOT converged where the old
        default of 20 claimed it had.

        5 is therefore a compromise, chosen and labelled as one: it keeps some
        of the exploration that restarts exist for, at a quarter of what 20
        cost, on a method that is already about 12x SLOWER than the PhenoGraph
        binary it replaces (121s against 41.6s). Raise it if you want the
        extra 0.005 of ARI that 40 bought and can spend 17 minutes on 20k
        events; drop it to 1 for interactive work and lose almost nothing
        measurable.

        Modularity is deliberately not compared against the binary. Q is a
        property OF A GRAPH, and the two methods partition different graphs
        (see ``_snn_jaccard_graph``), so comparing their Q values says nothing.
        ARI against known truth is the comparable measure.

        Writes a ``louvain`` column (non-finite rows → -1). Very large samples
        are partitioned on a random subsample of up to ``max_events`` and the
        rest assigned to their nearest labelled neighbour, as ``run_leiden``
        does; ``channels`` and ``n_neighbors`` default as there (30, 15 up to
        2.7.0). Returns self.
        """
        try:
            import igraph  # noqa: F401  # pyright: ignore[reportUnusedImport]  (availability check)
        except ImportError as e:
            raise ClusteringError(
                "igraph is required for Louvain clustering "
                "(pip install igraph)") from e
        from sklearn.neighbors import NearestNeighbors

        channels = channels or default_cluster_channels(self.fluor_channels)
        avail = [c for c in channels if c in self.data.columns]
        if not avail:
            log.warning("  [Louvain] no usable channels — skipped.")
            return self
        X = self.data[avail].values.astype(float)
        mask = np.all(np.isfinite(X), axis=1)
        Xc = X[mask]
        n = len(Xc)
        if n < 3:
            log.warning("  [Louvain] too few finite events — skipped.")
            return self

        if max_events and n > max_events:
            rng = np.random.default_rng(random_state)
            sub_idx = np.sort(rng.choice(n, max_events, replace=False))
            X_cluster = Xc[sub_idx]
            subsampled = True
        else:
            sub_idx = None
            X_cluster = Xc
            subsampled = False

        k = int(max(1, min(n_neighbors, len(X_cluster) - 1)))
        n_restarts = int(max(1, restarts))
        log.info("  Louvain: %s events × %d channels, k=%d, %d restarts …",
                 f"{len(X_cluster):,}", len(avail), k, n_restarts)
        g = _snn_jaccard_graph(X_cluster, k, prune=fast_graph)

        # Deterministic restart sequence: seed s, s+1, … keep the best Q.
        best, best_q = None, -np.inf
        for step in range(n_restarts):
            # igraph_seeded hands igraph back the random module afterwards,
            # so nothing else in the process stays pinned to this seed.
            with igraph_seeded(int(random_state) + step):
                # community_multilevel returns a VertexClustering; the union
                # with list[...] in its signature only arises with
                # return_levels=True, which we never pass. cast so the
                # attribute access type-checks against the real return.
                part = cast(Any, g.community_multilevel(weights='weight'))
            if float(part.modularity) > best_q:
                best, best_q = np.asarray(part.membership, dtype=int), \
                    part.modularity
        sub_comm = best if best is not None else np.zeros(len(X_cluster), int)

        if subsampled:
            communities = np.full(n, -1, dtype=int)
            communities[sub_idx] = sub_comm
            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(
                X_cluster)
            rest_mask = np.ones(n, dtype=bool)
            rest_mask[sub_idx] = False
            if rest_mask.any():
                nbr = tree_kneighbors(nn, Xc[rest_mask], 1)
                communities[rest_mask] = sub_comm[nbr[:, 0]]
        else:
            communities = sub_comm

        labels = np.full(len(self.data), -1, dtype=int)
        labels[mask] = communities
        self.data['louvain'] = labels
        n_clusters = len(np.unique(communities[communities >= 0]))
        log.info("  Louvain complete → %d clusters (Q=%.4f, %d restarts).",
                 n_clusters, best_q, n_restarts)
        return self

    # ── Cell cycle ──────────────────────────────────────────────────────────

    def _dna_width_partner(self, dna_channel: str) -> str | None:
        """The width (`-W`) partner of an area (`-A`) DNA channel, if the
        FCS carries one (used for doublet exclusion). None otherwise."""
        cu = dna_channel.upper()
        if not cu.endswith('-A'):
            return None
        stem = dna_channel[:-2]
        cols = {str(c).upper(): c for c in self.data.columns}
        for suf in ('-W', '-H'):          # case-insensitive: match -w/-h too
            hit = cols.get((stem + suf).upper())
            if hit is not None:
                return hit
        return None

    def cell_cycle(self, dna_channel: str | None = None, k: float = 1.5,
                   singlet_channel: str | None = None,
                   singlet_tol: float = 0.25,
                   method: str = 'boundaries') -> FlowSample:
        """DNA-content cell-cycle analysis.

        Auto-detects a DNA-stain channel (PI / DAPI / FxCycle / 7-AAD /
        Hoechst / DRAQ5 / …) when `dna_channel` is None; a label is
        accepted too. Doublet exclusion matters here (a G1 doublet lands at
        the G2/M position), so when a `-W`/`-H` partner exists (or
        `singlet_channel` is given) we pre-gate singlets on the DNA-A vs
        width ratio before modelling.

        Writes a categorical ``cell_cycle`` column (G1 / S / G2M / sub-G1 /
        >G2M / NA) and stores the model in ``self.cell_cycle_result``.
        ``method`` is 'boundaries' (counts between intensity cut-offs) or
        'dean-jett-fox' (the fitted model's areas; see cell_cycle_model).
        Returns self."""
        found = self.cell_cycle_model(dna_channel=dna_channel, k=k,
                                      singlet_channel=singlet_channel,
                                      singlet_tol=singlet_tol, method=method)
        if found is None:
            self.cell_cycle_result = None
            return self
        phases, model = found
        self.data['cell_cycle'] = phases
        self.cell_cycle_result = model
        return self

    def cell_cycle_model(self, dna_channel: str | None = None, k: float = 1.5,
                         singlet_channel: str | None = None,
                         singlet_tol: float = 0.25, method: str = 'boundaries',
                         ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """What `cell_cycle` writes, computed without writing it: ``(phases,
        model)``, or None when no DNA channel is found. Reads the sample
        only, so the editor runs it off the window thread.

        With ``method='dean-jett-fox'`` the phase percents are the areas of
        the fitted Dean-Jett-Fox model (cellcycle_djf), started from the
        boundaries model's peaks, and each cycling event takes the phase
        whose fitted curve is highest at its DNA content. The boundaries
        percents stay in ``model['boundaries']``. A failed fit keeps the
        boundaries result and says so (``model['method']``)."""
        if method not in ('boundaries', 'dean-jett-fox'):
            raise ValueError(f"Unknown cell-cycle method {method!r}: use "
                             "'boundaries' or 'dean-jett-fox'.")
        col = dna_channel
        if col is None:
            col = find_dna_channel(self)
        elif col not in self.data.columns:
            try:
                col = self._resolve(col)
            except KeyError:
                col = None
        if not col or col not in self.data.columns:
            log.warning("  [cell-cycle] no DNA channel found — skipped.")
            return None

        # DNA content is modelled on the LINEAR scale (G2 = 2 x G1, the
        # A/W singlet ratio). On the editor's logicle values G2/G1 measured
        # 1.07-1.11 for any G1 from 5,000 to 100,000, outside the 1.7-2.3
        # window analyze_dna searches, so every G2/M cell was called S.
        vals = linear_values(self, col)
        singlet = np.ones(len(self.data), dtype=bool)
        wcol = singlet_channel or self._dna_width_partner(col)
        if wcol and wcol in self.data.columns:
            w = linear_values(self, wcol)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(w > 0, vals / w, np.nan)
            good  = np.isfinite(ratio)
            if good.any():
                med = float(np.nanmedian(ratio[good]))
                lo, hi = med * (1 - singlet_tol), med * (1 + singlet_tol)
                singlet = good & (ratio >= lo) & (ratio <= hi)
            else:
                # Same trap as filter_doublets: a median of 0.0 gives the
                # window [0, 0], so NO event is a singlet and every cell is
                # scored 'NA' — the cell-cycle result silently becomes empty.
                # Without a usable width ratio there is no singlet
                # discrimination to apply, so keep every event and say so.
                log.warning(
                    "  [CellCycle] No usable %s/%s ratio — singlet "
                    "discrimination skipped; all events scored.", col, wcol)

        model  = analyze_dna(vals[singlet], k=k)
        phases = np.full(len(self.data), 'NA', dtype=object)
        phases[singlet] = assign_phase(vals[singlet], model)

        model = dict(model)
        model['channel'] = col
        model['width_channel'] = wcol
        model['n_singlet'] = int(singlet.sum())
        model['method'] = 'boundaries'
        if method == 'dean-jett-fox' and model.get('ok'):
            from .cellcycle_djf import fit_dean_jett_fox, phase_by_model
            sv = vals[singlet]
            fit = fit_dean_jett_fox(sv, model['g1_mean'], model.get('g2_mean'),
                                    model.get('g1_sd'), model.get('g2_sd'))
            model['boundaries'] = {p: model.get(p) for p in (
                'pct_g1', 'pct_s', 'pct_g2m', 'g1_mean', 'g2_mean')}
            if fit['ok']:
                phases[singlet] = phase_by_model(sv, fit, phases[singlet])
                model.update(
                    method='Dean-Jett-Fox', pct_g1=float(fit['pct_g1']),
                    pct_s=float(fit['pct_s']), pct_g2m=float(fit['pct_g2m']),
                    g1_mean=fit['g1_mean'], g2_mean=fit['g2_mean'],
                    g1_cv=fit['g1_cv'], g2_g1_ratio=fit['g2_g1_ratio'],
                    rcs=fit['rcs'],
                    djf_params=[float(x) for x in fit['params']])
            else:
                model['method'] = 'boundaries (Dean-Jett-Fox fit failed)'
        log.info(
            "  [cell-cycle] %s: G1 %.1f%% / S %.1f%% / G2M %.1f%% "
            "(singlets=%s/%s)",
            col, model.get('pct_g1', float('nan')),
            model.get('pct_s', float('nan')), model.get('pct_g2m', float('nan')),
            f"{model['n_singlet']:,}", f"{len(self.data):,}")
        return phases, model

    def export_csv(self, path: str | os.PathLike[str] | None = None) -> FlowSample:
        """Write the event data to a CSV (default ``<name>_processed.csv``); returns self.

        A transforms sidecar is written beside it so reloading inverts each column exactly."""
        if path is None:
            path = f"{self.name}_processed.csv"
        discard_transforms_sidecar(path)    # never beside the new CSV
        self.data.to_csv(path, index=False)
        # Beside it, the transform each column carries, so re-opening the CSV
        # (File > Load CSV) inverts exactly that instead of guessing.
        write_transforms_sidecar(path, self.data_transforms, self.data.columns,
                                 len(self.data))
        log.info(f"  Exported → {path}")
        return self

    def cluster_stats(self) -> pd.DataFrame:
        """Per-cluster stats (cluster_frequencies), plus a
        ``pct_pos_<marker>`` column for every threshold gate applied with
        apply_threshold_gates: the % of that cluster's events at or above the
        cut. With no threshold gates it is cluster_frequencies unchanged.

        export_stats and the CLI's per-sample stats CSV both write this, so a
        library run and an openflo-run run give the same columns."""
        freq = self.cluster_frequencies()
        thresholds = getattr(self, 'thresholds', None) or {}
        if freq.empty or not thresholds or 'cluster' not in self.data.columns:
            return freq
        for ch in thresholds:
            try:
                col = self._resolve(ch)
            except KeyError:
                continue
            flag = f'{col}_pos'
            if flag not in self.data.columns:
                continue
            ids, inv = np.unique(np.asarray(self.data['cluster']),
                                 return_inverse=True)
            pos = np.bincount(inv, weights=np.asarray(self.data[flag],
                                                     dtype=float))
            pct = dict(zip(ids.tolist(),
                           (pos / np.bincount(inv) * 100).tolist(),
                           strict=True))
            freq[f'pct_pos_{self.channel_labels.get(col, col)}'] = (
                np.array([pct.get(c, np.nan) for c
                          in np.asarray(freq['cluster']).tolist()],
                         dtype=float))
        return freq

    def export_stats(self,
                     path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
        """Write ``cluster_stats()`` to CSV (default ``<name>_cluster_stats.csv``); returns it."""
        freq = self.cluster_stats()
        if path is None:
            path = f"{self.name}_cluster_stats.csv"
        freq.to_csv(path, index=False)
        log.info(f"  Stats → {path}")
        return freq
