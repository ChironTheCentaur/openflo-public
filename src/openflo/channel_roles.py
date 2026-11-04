"""Which of a sample's channels are scatter, time, pulse-width or instrument
parameters (``is_scatter``), which are derived / analysis columns never
treated as markers (``EXCLUDE_CLUSTER``, ``is_excluded``), and the one column
per marker clustering uses by default (``default_cluster_channels``).

Core module: standard library only. No Tk, and it does not import
``openflo.pipeline`` (pipeline imports it).
"""

import re

from .channel_names import TIME_PATTERN, is_time_channel

# Derived / analysis columns — never treated as markers for clustering,
# stats, channel classification or acquisition QC. Keep this in step with
# every column the app WRITES back onto sample.data: three of the five
# embeddings were listed and TSNE/PHATE were not, and `leiden` and
# `pseudotime` were missing entirely. Acquisition QC selects channels as
# "every numeric column minus this list", so an omission here is not
# cosmetic — the margin detector read a cluster label as a detector at its
# ceiling and deleted the whole top-numbered cluster (measured: 12.4% of a
# clean sample). tests/test_qc_ignores_derived_columns.py pins the list
# against what the code actually writes.
#
# '__group__' / '__sample__' are the workspace's per-event source tags,
# written into every run's _events table; counted as fluor channels they made
# the Statistics window take the median of '(ungrouped)' (ValueError).
EXCLUDE_CLUSTER  = ['Time', 'time',
                    'cluster', 'flowsom', 'flowsom_meta', 'cell_cycle',
                    'debarcode', 'leiden', 'pseudotime',
                    'UMAP1', 'UMAP2', 'TSNE1', 'TSNE2', 'TRIMAP1', 'TRIMAP2',
                    'PACMAP1', 'PACMAP2', 'PHATE1', 'PHATE2',
                    '__group__', '__sample__']


# Scatter, time, pulse-width and instrument parameters stay LINEAR and are
# never markers. A case-sensitive substring match (SCATTER_KEYWORDS) sends
# every name below to the fluorescence list, to be logicle-transformed by
# apply_transform and clustered on as a marker: Beckman's 'FS INT' / 'SS INT'
# / 'TIME', a lower-case 'fsc-a', Sony's back-scatter 'BSC-A', and CyTOF's
# Event_length. Measured: TIME 0..1000 s became 0.11..0.45 logicle units. The
# match is on the upper-cased name with token boundaries, so a scatter
# name is recognised whatever its case or separator and a fluor name is not
# caught by a stray substring.
_SCATTER_RE = re.compile(
    # FSC / SSC with up to two letters before: VSSC1-A (violet SSC),
    # BSSC-A (Attune), SSC-B-A (Cytek), 'Violet SSC-A', FSC-HLin (Guava);
    # BSC is Sony's back-scatter.
    r'(?<![A-Z0-9])[A-Z]{0,2}(?:FSC|SSC|BSC)(?![A-Z])'
    # Beckman Gallios / Navios / FC500: 'FS INT', 'SS PEAK', 'FS TOF', 'FS Lin'.
    r'|^(?:FS|SS)(?:[^A-Z0-9]|$)')
# Old private names of channel_names.TIME_PATTERN / is_time_channel.
_TIME_RE = TIME_PATTERN
# Mass-cytometry acquisition parameters (Fluidigm / Standard BioTools).
_INSTRUMENT_PARAMS = frozenset({'EVENT_LENGTH', 'EVENT LENGTH', 'CENTER',
                                'OFFSET', 'RESIDUAL'})


_is_time = is_time_channel


def is_scatter(name):
    """True for a parameter kept linear and never used as a marker: scatter,
    time, pulse width and instrument parameters (see _SCATTER_RE)."""
    u = str(name).upper()
    return (bool(_SCATTER_RE.search(u)) or is_time_channel(u) or 'WIDTH' in u
            or u in _INSTRUMENT_PARAMS)


def is_excluded(name):
    """True for a column left out of clustering: a time channel or a name containing
    an ``EXCLUDE_CLUSTER`` entry (cluster labels, embeddings and the like)."""
    return any(k in name for k in EXCLUDE_CLUSTER) or is_time_channel(name)


# A detector's pulse measurement as the last token of its name: 'FL1-A',
# 'FL1 H', 'FL1_W', 'FL3 Area', 'FL3-Height'. Group 1 is the detector, group 2
# the measurement.
_PULSE_RE = re.compile(r'^(.*?\S)[\s_-]+(A|AREA|H|HEIGHT|W|WIDTH)$',
                       re.IGNORECASE)


def default_cluster_channels(channels):
    """The channels clustering and embedding use when none are given: ONE
    column per marker, out of ``channels`` (a sample's fluorescence list).

    A marker exported as area, height and width counted about three times in
    every neighbour distance, because the three are collinear. So each
    detector keeps its area (``-A`` / ``Area``) column; its height (``-H`` /
    ``Height``) only when it has no area, so a file with FL1-A and FL2-H
    keeps FL2; its width (``-W`` / ``Width``) never. A name without one of
    these suffixes (a mass channel, a label) is kept. Order is kept. Falls
    back to every channel if the rule would leave none."""
    fl = [str(c) for c in (channels or [])]
    kinds: dict[str, set] = {}
    parsed = []
    for c in fl:
        m = _PULSE_RE.match(c.strip())
        if m is None:
            parsed.append((c, None, None))
            continue
        base, kind = m.group(1).strip().upper(), m.group(2).upper()[0]
        parsed.append((c, base, kind))
        kinds.setdefault(base, set()).add(kind)
    keep = [c for c, base, kind in parsed
            if kind is None or kind == 'A'
            or (kind == 'H' and 'A' not in kinds[base])]
    return keep or fl
