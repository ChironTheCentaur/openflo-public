"""Positive-gate thresholds from FMO controls: ``FMOGater``.

Core module: no Tk, and it does not import ``openflo.pipeline``, which imports
this name back, so ``pipeline.FMOGater`` and ``from openflo import FMOGater``
are this class.
"""

import logging
import os
import re

import numpy as np

from .channel_transforms import to_linear
from .flow_sample import FlowSample
from .gating_helpers import fmo_threshold

log = logging.getLogger('openflo.pipeline')


# ══════════════════════════════════════════════════════════════════════════════
# FMO GATE THRESHOLDER
# ══════════════════════════════════════════════════════════════════════════════

class FMOGater:
    """
    Calculate positive thresholds from FMO control files.

    Usage
    -----
        gater = FMOGater()
        gater.add_fmo('Comp-FL1-A', 'fmo_fl1.fcs')  # CD901b
        gater.add_fmo('Comp-FL2-A',           'fmo_fl2.fcs')   # CD902
        gater.add_fmo('Comp-FL3-A',        'fmo_fl3.fcs')   # CD903
        thresholds = gater.compute(percentile=99.0)
        sample.apply_threshold_gates(thresholds)
    """

    def __init__(self):
        self.fmos        = {}
        self._is_fallback = {}   # channel -> True if using unstained instead of FMO

    def add_fmo(self, channel_name, fcs_path, is_fallback=False):
        """
        channel_name: detector name this FMO (or fallback unstained) controls.
        is_fallback:  True when the file is an unstained control rather than a
                      proper FMO — noted in compute() output.
        """
        self.fmos[channel_name]         = FlowSample(fcs_path)
        self._is_fallback[channel_name] = is_fallback
        return self

    def add_fmos_from_dir(self, directory, pattern=r'fmo[_-]?(\w+)'):
        """Load every .fcs file in ``directory`` whose name matches ``pattern`` as an FMO.

        The regex's first group gives the channel key. Returns self."""
        rx = re.compile(pattern, re.IGNORECASE)
        for fname in os.listdir(directory):
            if not fname.lower().endswith('.fcs'):
                continue
            m = rx.search(fname)
            if m:
                ch  = m.group(1)
                self.fmos[ch] = FlowSample(os.path.join(directory, fname))
                log.info(f"  [FMO] '{ch}' ← {fname}")
        return self

    def prepare(self, wsp_path=None, transform_method='logicle'):
        """
        Compensate and transform all FMO samples so thresholds are in the
        same data space as the experimental samples.  Call before compute().
        """
        for sample in self.fmos.values():
            done = list(sample._transformed_channels())
            if done:
                # Prepared before: back to linear, so the compensation below
                # replaces the earlier one instead of meeting logicle values.
                sample.retransform(done, method='linear')
            if wsp_path:
                sample.compensate_from_wsp(wsp_path)
            else:
                sample.auto_compensate()
            sample.apply_transform(method=transform_method)
        return self

    def compute(self, percentile=99.0):
        """
        Returns {channel_name: threshold} where channel_name is the key
        passed to add_fmo() — typically the compensated detector name.
        The threshold is the p`percentile` of that channel in the FMO:
        99 by default (1 % of FMO events above the gate, the usual choice
        and the app's FMO dialog default).
        """
        thresholds = {}
        for ch, sample in self.fmos.items():
            col = self._find_col(sample, ch)
            if col is None:
                log.info(f"  [FMO] '{ch}' not found in {sample.name} — skipped.")
                continue
            # The app's FMO dialog cut (gating_helpers.fmo_threshold): finite
            # values only. dropna() kept +/-inf, and one +inf made the cut NaN.
            try:
                val = fmo_threshold(sample.data[col].to_numpy(), percentile)
            except ValueError:
                log.info(f"  [FMO] '{ch}' has no finite values in "
                         f"{sample.name} — skipped.")
                continue
            thresholds[ch] = val
            tag = ' [UNSTAINED fallback]' if self._is_fallback.get(ch) else ''
            # The cut is in the transformed space the samples are gated in;
            # logged bare ('threshold=0.487', at 1,369 linear) it read as an
            # intensity. The linear value beside it is the one to compare
            # with an MFI.
            spec = (getattr(sample, 'data_transforms', None) or {}).get(col)
            scale = ''
            if spec and spec.get('method', 'linear') != 'linear':
                lin = float(to_linear(np.array([val]), spec)[0])
                scale = f" {spec['method']} (= {lin:,.0f} linear)"
            log.info(f"  [FMO] {ch}: threshold={val:.3f}{scale} "
                     f"(p{percentile}){tag}")
        return thresholds

    @staticmethod
    def _find_col(sample, hint):
        """Match by exact name, then by stripping common comp prefixes, then substring."""
        if hint in sample.data.columns:
            return hint
        stripped = re.sub(r'^[Cc]omp-', '', hint)
        for col in sample.data.columns:
            if col == stripped or col.lower() == stripped.lower():
                return col
        for col in sample.data.columns:
            if hint.lower() in col.lower() or stripped.lower() in col.lower():
                return col
        return None

