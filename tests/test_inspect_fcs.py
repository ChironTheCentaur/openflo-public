"""python -m openflo.inspect_fcs — the FCS header dump (README: "FCS header dump").

The module is a top-level script (it reads sys.argv and exits at import),
so it is driven with runpy under a patched argv.
"""
from __future__ import annotations

import runpy
import sys

import numpy as np
import pytest

CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
MARKERS = ['', '', 'CD901b', 'CD902', 'CD903']
SPILL = '2,FL1-A,FL2-A,1,0.05,0.02,1'


@pytest.fixture
def fcs(tmp_path):
    import flowio
    ev = np.random.default_rng(0).normal(1000, 10, (50, 5)).astype(np.float32)
    p = tmp_path / 'x.fcs'
    with open(p, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), CHANNELS,
                          opt_channel_names=MARKERS,
                          metadata_dict={'SPILL': SPILL})
    return str(p)


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['inspect_fcs', *argv])
    sys.modules.pop('openflo.inspect_fcs', None)
    runpy.run_module('openflo.inspect_fcs', run_name='__main__')


def test_dumps_channels_markers_and_spillover(fcs, monkeypatch, capsys):
    _run(monkeypatch, fcs)
    out = capsys.readouterr().out
    assert 'channel_count: 5' in out
    for name in CHANNELS:
        assert f"'pnn': '{name}'" in out
    for m in MARKERS:
        if m:
            assert f"'pns': '{m}'" in out
    spill_line = out.split('Spillover keys:')[1].splitlines()[0]
    assert 'spill' in spill_line.lower()
    assert f'Value: {SPILL}' in out


def test_no_argument_prints_usage_and_exits(monkeypatch):
    with pytest.raises(SystemExit) as ei:
        _run(monkeypatch)
    assert 'usage' in str(ei.value.code)


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT: the 'Text keys (PnN/PnS/P*N)' section filters on keys starting "
    "with '$P', but flowio>=1.0 returns lower-case keys without '$' "
    "('p1n', 'p1s'), so the section is always empty"))
def test_text_keys_section_lists_pnn_keys(fcs, monkeypatch, capsys):
    _run(monkeypatch, fcs)
    out = capsys.readouterr().out
    section = out.split('Text keys (PnN/PnS/P*N):')[1].split('Spillover keys:')[0]
    assert 'p1n' in section.lower()
