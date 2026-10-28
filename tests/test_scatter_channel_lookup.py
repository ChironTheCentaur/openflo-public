"""One FSC/SSC channel lookup for the auto-clean steps, FlowSample and the CLI.

The lookup was written three ways (best-practices finding B6): the auto-clean
helper and FlowSample._find_scatter_col took the first channel ENDING in the
suffix, while the CLI took the first with '-A' ANYWHERE in its name. On
['FSC-H', 'FSC-Area'] the CLI drew FSC-Area while the library used FSC-H.
All three now call pipeline.find_scatter_channel.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import openflo.pipeline as fp
from openflo import cli


def test_find_scatter_channel_prefers_the_suffix_then_falls_back():
    cols = ['FL1-A', 'fsc-h', 'FSC-A', 'SSC-H', 'SSC-A', 'Time']
    assert fp.find_scatter_channel(cols, 'FSC') == 'FSC-A'
    assert fp.find_scatter_channel(cols, 'FSC', '-H') == 'fsc-h'
    assert fp.find_scatter_channel(cols, 'ssc', '-a') == 'SSC-A'
    assert fp.find_scatter_channel(['FSC-H', 'SSC-A'], 'FSC') == 'FSC-H'
    assert fp.find_scatter_channel(['FSC-H', 'SSC-A'], 'FSC',
                                   fallback=False) is None
    assert fp.find_scatter_channel(['FL1-A'], 'FSC') is None
    df = pd.DataFrame({'SSC-A': [1.0], 'FSC-A': [2.0]})
    assert fp.find_scatter_channel(df, 'FSC') == 'FSC-A'


def test_flowsample_and_the_cli_pick_the_same_channel_on_every_name_set():
    names = ['FSC-A', 'FSC-H', 'FSC-W', 'FSC-Area', 'SSC-A', 'SSC-H',
             'SSC-B-A', 'FL1-A']
    rng = np.random.default_rng(6)
    for cols in itertools.chain(
            itertools.permutations(names[:4], 2),
            (list(rng.permutation(names)[:k]) for k in (3, 5, 8)
             for _ in range(20))):
        cols = list(dict.fromkeys(cols)) + ['FL2-A']
        s = fp.FlowSample.from_dataframe(
            pd.DataFrame({c: np.arange(4.0) for c in cols}), name='syn')
        for prefix in ('FSC', 'SSC'):
            lib = s._find_scatter_col(prefix, '-A')
            assert lib == fp.find_scatter_channel(cols, prefix), cols
            # The CLI looks only among the sample's scatter channels.
            assert cli.find_scatter_channel(s.scatter_channels, prefix) == lib


def test_the_cli_has_no_lookup_of_its_own():
    assert cli.find_scatter_channel is fp.find_scatter_channel
    assert not hasattr(cli, '_find_scatter_channel')
    assert not hasattr(fp, '_autoclean_find_scatter')
