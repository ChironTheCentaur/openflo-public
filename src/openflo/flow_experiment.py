"""The multi-sample analysis class, ``FlowExperiment``: load a folder of FCS
files, process them, cluster them together and compare conditions.

Core module: no Tk, and it does not import ``openflo.pipeline``, which imports
this name back, so ``pipeline.FlowExperiment`` and ``from openflo import
FlowExperiment`` are this class.

``matplotlib.pyplot`` is imported inside the plotting methods, so importing
this module does not load it.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .cluster_labels import cluster_label
from .flow_sample import FlowSample
from .label_alignment import concatenate
from .som_kernel import FLOWSOM_DEFAULT_METACLUSTERS

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from typing import Any

    from matplotlib.figure import Figure

log = logging.getLogger('openflo.pipeline')


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT (multi-sample)
# ══════════════════════════════════════════════════════════════════════════════

class FlowExperiment:
    """
    Batch-process multiple FCS files.

    Usage
    -----
        exp = FlowExperiment('/path/to/fcs/')
        exp.exclude_pattern('unstained|bead|fmo')
        exp.run_all(k=30, umap=True)
        exp.compare_conditions(
            groupA=['sample_1','sample_2'], groupB=['sample_3','sample_4'],
            label_a='Group A',              label_b='Group B')
        exp.export_all('results/')
    """

    def __init__(self, source: str | os.PathLike[str] | list[str]
                 | tuple[str, ...]) -> None:
        self.samples = {}
        self.cluster_params = None       # set by run_all
        self.joint_clustering = None     # set by cluster_jointly
        for p in self._resolve(source):
            try:
                s = FlowSample(p)
                self.samples[s.name] = s
            except Exception as e:  # noqa: BLE001 - logged; one unreadable file must not stop the others loading
                log.warning(f"  [!] {p}: {e}")
        log.info(f"\nLoaded {len(self.samples)} sample(s).")

    @staticmethod
    def _resolve(source):
        if isinstance(source, (list, tuple)):
            return list(source)
        if os.path.isdir(source):
            return sorted([os.path.join(source, f)
                           for f in os.listdir(source)
                           if f.lower().endswith('.fcs')])
        raise ValueError("source must be a directory or list of paths.")

    def exclude_pattern(self, pattern: str) -> FlowExperiment:
        """Drop every sample whose name matches regex ``pattern`` (case-insensitive).
        Modifies the experiment in place and returns it for chaining."""
        rx  = re.compile(pattern, re.IGNORECASE)
        rem = [n for n in self.samples if rx.search(n)]
        for n in rem:
            del self.samples[n]
        log.info(f"Excluded {len(rem)}: {rem}")
        return self

    def keep_pattern(self, pattern: str) -> FlowExperiment:
        """Keep only samples whose name matches regex ``pattern`` (case-insensitive),
        dropping the rest in place; returns the experiment for chaining."""
        rx  = re.compile(pattern, re.IGNORECASE)
        rem = [n for n in self.samples if not rx.search(n)]
        for n in rem:
            del self.samples[n]
        log.info(f"Kept {len(self.samples)}, removed {len(rem)}.")
        return self

    def run_all(self, qc: bool = False, compensate: bool = True,
                wsp_path: str | os.PathLike[str] | None = None,
                transform: bool = True, transform_method: str = 'logicle',
                cluster: bool = True, k: int = 30, umap: bool = False,
                resolution: float = 1.0) -> FlowExperiment:
        """Run the processing steps on every sample in turn: optional QC,
        compensation (from ``wsp_path`` if given, else automatic), transform,
        clustering and UMAP. Records the cluster settings; returns self."""
        for name, s in self.samples.items():
            log.info(f"\n── {name}")
            if qc:           s.run_qc()
            if compensate:
                if wsp_path: s.compensate_from_wsp(wsp_path)
                else:        s.auto_compensate()
            if transform:    s.apply_transform(method=transform_method)
            if cluster:      s.cluster(k=k, resolution=resolution)
            if umap:         s.run_umap()
        if cluster:
            # What compare_conditions clusters the compared samples with.
            self.cluster_params = {'method': 'phenograph', 'k': k,
                                   'resolution': float(resolution)}
        return self

    def combined_frequencies(self, label_col: str = 'cluster',
                             names: Iterable[str] | None = None,
                             ) -> pd.DataFrame:
        """Every sample's (or the named samples') frequencies per label of
        `label_col`, one table."""
        pick = (self.samples.values() if names is None
                else [self.samples[n] for n in names])
        frames = [s.cluster_frequencies(label_col) for s in pick
                  if label_col in s.data.columns]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def cluster_jointly(self, names: Iterable[str] | None = None,
                        method: str | None = None, k: int | None = None,
                        max_events: int | None = None, seed: int = 42,
                        n_jobs: int = 1, resolution: float | None = None,
                        n_metaclusters: int = FLOWSOM_DEFAULT_METACLUSTERS,
                        label_col: str = 'shared_cluster',
                        ) -> dict[str, np.ndarray]:
        """Cluster the named samples (default: all) TOGETHER, once, and
        write each sample's label from that one clustering to its
        `label_col` column. Returns ``{name: labels}``.

        Each sample's own clustering (FlowSample.cluster, run_all) numbers
        its clusters arbitrarily, so one id names different populations in
        different samples and cannot be compared across them. This pools an
        equal share of every sample's events (at most `max_events` in all;
        by default as many as the largest sample holds, or the method's own
        cap), clusters the pool with `method` / `k` / `resolution` (default:
        what run_all used, else PhenoGraph k=30 at resolution 1.0), and gives every other event its
        nearest pooled event's label (FlowSOM: its SOM node's), exactly as
        openflo-run's group comparisons do
        (shared_clustering.shared_cluster_labels).
        Channels are matched across samples by antibody label.

        Records the clustering in ``self.joint_clustering`` (samples,
        label_col, settings), which compare_conditions reads."""
        from .shared_clustering import cluster_settings, shared_cluster_labels
        names = (list(self.samples) if names is None
                 else list(dict.fromkeys(names)))
        missing = [n for n in names if n not in self.samples]
        if missing:
            raise KeyError(f"not in this experiment: {missing}")
        if not names:
            raise ValueError("no samples to cluster")
        p = dict(getattr(self, 'cluster_params', None) or {})
        method = method or p.get('method', 'phenograph')
        k = int(k or p.get('k', 30))
        resolution = float(resolution if resolution is not None
                           else p.get('resolution', 1.0))
        task = cluster_settings(k, n_jobs, max_events, 1.0, seed,
                                {'method': method, 'resolution': resolution,
                                 'n_metaclusters': n_metaclusters})
        samples = [self.samples[n] for n in names]
        log.info("Clustering %d sample(s) together (%s, k=%d) so cluster "
                 "ids mean the same population in every sample.",
                 len(samples), method, k)
        labels, channels, _cols, _display = shared_cluster_labels(
            samples, task, names=names)
        for s, lab in zip(samples, labels, strict=True):
            s.data[label_col] = lab
        self.joint_clustering = {'samples': tuple(names),
                                 'label_col': label_col, 'method': method,
                                 'k': k, 'resolution': resolution,
                                 'channels': list(channels)}
        return dict(zip(names, labels, strict=True))

    def compare_conditions(self, groupA: Collection[str],
                            groupB: Collection[str],
                            label_a: str = 'A', label_b: str = 'B',
                            plot: bool = True,
                            **cluster_kw: Any) -> pd.DataFrame:
        """Mean % of events per cluster in each condition (+ sd, n).

        A cluster id is compared ACROSS samples here, so it must come from
        one clustering of all the compared samples. The ids of each
        sample's own clustering do not: PhenoGraph, Leiden and FlowSOM
        number clusters arbitrarily, and with CD4 at 80 % in A and 20 % in
        B, each sample's largest cluster was id 0, so the table read 80 % vs
        80 % for "cluster 0" (CD4 in A, CD8 in B). So unless
        ``self.joint_clustering`` (cluster_jointly, or openflo-run's shared
        clustering) already covers every compared sample, the compared
        samples are first clustered together with cluster_jointly
        (`cluster_kw`: method, k, max_events, seed, ...; passing any forces
        a new joint clustering), and the frequencies come from those
        labels. To compare several pairs on one set of ids, call
        ``cluster_jointly()`` once on all samples first."""
        names = [n for n in dict.fromkeys(list(groupA) + list(groupB))
                 if n in self.samples]
        jc = getattr(self, 'joint_clustering', None)
        if (jc and not cluster_kw and set(names) <= set(jc['samples'])
                and all(jc['label_col'] in self.samples[n].data.columns
                        for n in names)):
            label_col = jc['label_col']
        else:
            if not names:
                log.warning("  [!] None of the compared samples is in this "
                            "experiment.")
                return pd.DataFrame()
            self.cluster_jointly(names, **cluster_kw)
            label_col = cluster_kw.get('label_col', 'shared_cluster')
        freq = self.combined_frequencies(label_col, names)
        if freq.empty:
            log.warning("  [!] No cluster data — run clustering first.")
            return freq
        freq['condition'] = freq['sample'].apply(
            lambda n: label_a if n in groupA else (label_b if n in groupB else 'other'))
        freq = freq[freq['condition'] != 'other']
        # A sample where a cluster has no events contributes no ROW: the
        # frequencies come from value_counts(), which omits empty categories.
        # Averaging the rows directly therefore averaged only the samples the
        # cluster appeared in — a population found in 1 of 3 samples at 10%
        # was reported as the group's 10%, when the group mean is 3.3%. Absent
        # means 0%, not "not measured", so the table is squared off first and
        # every sample in the condition contributes a value.
        wide = freq.pivot_table(index=['condition', 'sample'],
                                columns='cluster', values='pct_total',
                                aggfunc='sum', fill_value=0.0)
        tall: pd.DataFrame
        if wide.empty:
            tall = pd.DataFrame(
                freq[['condition', 'sample', 'cluster', 'pct_total']])
        else:
            tall = pd.DataFrame(
                {'pct_total': wide.stack()}).reset_index()
        summary = (tall.groupby(['condition','cluster'])['pct_total']
                       .agg(['mean','std','size']).reset_index()
                       .rename(columns={'mean':'mean_pct', 'std':'sd_pct',
                                        'size':'n_samples'}))
        # Name each cluster (incl. the -1 noise bucket) so the comparison table
        # never carries a bare -1 next to real populations.
        summary.insert(2, 'population',
                       np.array([cluster_label(c) for c in summary['cluster']]))
        if plot:
            import matplotlib.pyplot as plt  # lazy: see module-top comment
            clusters = sorted(summary['cluster'].unique())
            a_v = (summary[summary.condition==label_a]
                          .set_index('cluster')['mean_pct']
                          .reindex(clusters, fill_value=0))
            b_v = (summary[summary.condition==label_b]
                          .set_index('cluster')['mean_pct']
                          .reindex(clusters, fill_value=0))
            x, w = np.arange(len(clusters)), 0.35
            _, ax = plt.subplots(figsize=(max(8, len(clusters)*0.5), 5))
            ax.bar(x-w/2, a_v, w, label=label_a, color='steelblue', alpha=0.8)
            ax.bar(x+w/2, b_v, w, label=label_b, color='coral',     alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([cluster_label(c) for c in clusters], rotation=45,
                               ha='right')
            ax.set_ylabel('% of total events')
            ax.set_title(f'{label_a} vs {label_b} — cluster frequencies')
            ax.legend()
            plt.tight_layout()
        return summary

    def plot_all(self, x: str, y: str, color_by: str = 'cluster',
                 ncols: int = 3, sample_n: int | None = 30_000) -> Figure:
        """A grid figure with one ``x`` vs ``y`` scatter per sample, ``ncols`` wide,
        coloured by ``color_by``; unused axes are hidden. Returns the Figure."""
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        names  = list(self.samples.keys())
        nrows  = (len(names)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(6*ncols, 5*nrows))
        axes = np.array(axes).flatten()
        # i defaults to -1 so that when `names` is empty, the post-loop
        # range(i+1, len(axes)) hides every axis (otherwise i is unbound).
        i = -1
        for i, name in enumerate(names):
            self.samples[name].plot(x, y, color_by=color_by,
                                    sample_n=sample_n, ax=axes[i])
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        return fig

    def concatenate_group(self, names: Iterable[str]) -> pd.DataFrame:
        """Concatenate the named samples' data into one DataFrame; names not loaded
        are skipped."""
        return concatenate([self.samples[n] for n in names if n in self.samples])

    def export_all(self, out_dir: str | os.PathLike[str] = '.') -> FlowExperiment:
        """Write each sample's processed CSV (and cluster stats CSV when it has
        clusters) into ``out_dir``, creating it if needed; returns self."""
        os.makedirs(out_dir, exist_ok=True)
        for name, s in self.samples.items():
            s.export_csv(os.path.join(out_dir, f"{name}_processed.csv"))
            if 'cluster' in s.data.columns:
                s.export_stats(os.path.join(out_dir, f"{name}_stats.csv"))
        return self

