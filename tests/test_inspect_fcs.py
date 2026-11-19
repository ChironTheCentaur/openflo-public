"""python -m openflo.inspect_fcs — the FCS header dump (README: "FCS header dump").

The module runs under `python -m`, so the __main__ path is driven with runpy
under a patched argv; main() is also called directly.
"""
from __future__ import annotations

import runpy
import sys

import numpy as np
import pytest

from tests._fcs_writer import write_float_fcs

CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
MARKERS = ['', '', 'CD901b', 'CD902', 'CD903']
SPILL = '2,FL1-A,FL2-A,1,0.05,0.02,1'


@pytest.fixture
def fcs(tmp_path):
    ev = np.random.default_rng(0).normal(1000, 10, (50, 5)).astype(np.float32)
    p = tmp_path / 'x.fcs'
    write_float_fcs(p, ev, CHANNELS, opt_channel_names=MARKERS, metadata_dict={'SPILL': SPILL})
    return str(p)


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['inspect_fcs', *argv])
    sys.modules.pop('openflo.inspect_fcs', None)
    try:
        runpy.run_module('openflo.inspect_fcs', run_name='__main__')
    except SystemExit as e:
        if e.code:          # the usage message; 0 is a clean finish
            raise


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


def test_text_keys_section_lists_pnn_keys(fcs, monkeypatch, capsys):
    _run(monkeypatch, fcs)
    out = capsys.readouterr().out
    section = out.split('Text keys (PnN/PnS/P*N):')[1].split('Spillover keys:')[0]
    assert 'p1n' in section.lower()


def test_main_takes_the_path_as_an_argument(fcs, capsys):
    from openflo.inspect_fcs import main
    assert main([fcs]) == 0
    assert 'channel_count: 5' in capsys.readouterr().out


def test_main_without_a_path_prints_usage_and_exits():
    from openflo.inspect_fcs import main
    with pytest.raises(SystemExit) as ei:
        main([])
    assert 'usage' in str(ei.value.code)
