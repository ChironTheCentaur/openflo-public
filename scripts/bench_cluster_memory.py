#!/usr/bin/env python
"""Peak memory (RSS) of one FlowSample clustering call on generated data,
the measurement behind cluster_caps._MEMORY_FIT.

    python scripts/bench_cluster_memory.py phenograph 100000
    python scripts/bench_cluster_memory.py leiden 50000
    python scripts/bench_cluster_memory.py leiden-fast 200000
    python scripts/bench_cluster_memory.py phenograph 50000 100   # k=100

Prints the method, the events, the process's peak RSS in GB (Linux and
macOS report ru_maxrss differently: this reads it as KB, as Linux does) and
the time. Run each size in its own process, since the peak is per process.
Generated Gaussian mixture, 12 channels, 8 populations, k=30 unless a third argument gives k, one core;
every event is clustered (max_events=None).
"""
import resource
import sys
import time

import numpy as np
import pandas as pd

from openflo.pipeline import FlowSample

method, n = sys.argv[1], int(sys.argv[2])
k = int(sys.argv[3]) if len(sys.argv) > 3 else 30
rng = np.random.default_rng(0)
centers = rng.normal(0, 4, (8, 12))
lab = rng.integers(0, 8, n)
X = centers[lab] + rng.normal(0, 1, (n, 12))
s = FlowSample.from_dataframe(pd.DataFrame(X, columns=[f'FL{i}-A' for i in range(1, 13)]), name='gen')
chans = list(s.data.columns)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
t = time.time()
if method == 'phenograph':
    s.cluster(channels=chans, k=k, max_events=None, n_jobs=1)
else:
    s.run_leiden(channels=chans, n_neighbors=k, max_events=None,
                 fast_graph=method == 'leiden-fast')
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
print(method, n, k, f'{peak:.2f}', f'base {base:.2f}', f'{time.time() - t:.0f}s', flush=True)
