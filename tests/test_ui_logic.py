"""Tests for openflo.ui_logic — the first pure logic extracted from gui.py.

Headless (no Tk): these run on CI without a display, which is the whole point
of pulling the logic out of the widget callbacks.
"""
from __future__ import annotations

from openflo.ui_logic import (
    filter_choices,
    format_channel,
    has_real_gates,
    resolve_channel,
    resolve_choice,
    short_label,
    should_use_dark,
)


def test_short_label():
    assert short_label('CD3') == 'CD3'                       # short → unchanged
    assert short_label('x' * 30, width=10) == 'x' * 9 + '…'  # elided to width
    assert short_label('exactly_ten', width=11) == 'exactly_ten'

CHANNELS = ['FSC-A', 'SSC-A', 'CD3', 'CD4', 'CD8']


def test_filter_choices():
    assert filter_choices('', CHANNELS) == CHANNELS          # empty → all
    assert filter_choices('cd', CHANNELS) == ['CD3', 'CD4', 'CD8']
    assert filter_choices('CD4', CHANNELS) == ['CD4']
    assert filter_choices('zzz', CHANNELS) == CHANNELS       # no match → all
    assert filter_choices('ssc', CHANNELS) == ['SSC-A']      # case-insensitive


def test_resolve_choice():
    assert resolve_choice('cd4', CHANNELS) == 'CD4'          # exact (ci)
    assert resolve_choice('cd', CHANNELS) == 'CD3'           # first substring
    assert resolve_choice('', CHANNELS, fallback='CD8') == 'CD8'
    assert resolve_choice('zzz', CHANNELS, fallback='CD3') == 'CD3'  # revert
    assert resolve_choice('zzz', CHANNELS) == ''             # no fallback


def test_has_real_gates():
    assert has_real_gates({}) is False
    assert has_real_gates({'a': {'kind': 'autoclean'}}) is False     # negative
    assert has_real_gates({'g': {'kind': 'rect'}}) is True
    # a positive gate alongside an autoclean still counts
    assert has_real_gates({'a': {'kind': 'autoclean'},
                           'g': {'kind': 'polygon'}}) is True
    # a gate with no 'kind' is treated as a real (positive) gate
    assert has_real_gates({'g': {}}) is True


def test_format_and_resolve_channel_roundtrip():
    labels = {'BV421-A': 'CD11b', 'APC-A': 'CD34', 'FSC-A': 'FSC-A'}
    # distinct label → "Label (DET)"; no/identity label → bare detector
    assert format_channel('BV421-A', labels) == 'CD11b (BV421-A)'
    assert format_channel('FSC-A', labels) == 'FSC-A'        # label == det
    assert format_channel('PE-A', labels) == 'PE-A'          # no label
    assert format_channel('PE-A', None) == 'PE-A'
    # resolve recovers the detector from the display string
    assert resolve_channel('CD11b (BV421-A)') == 'BV421-A'
    assert resolve_channel('FSC-A') == 'FSC-A'               # no parens
    assert resolve_channel('') is None
    # round-trips for every labelled detector
    for det in labels:
        assert resolve_channel(format_channel(det, labels)) == det


def test_should_use_dark():
    assert should_use_dark(True, 'light') is True      # toggle wins
    assert should_use_dark(False, 'midnight') is True  # midnight implies dark
    assert should_use_dark(False, 'light') is False
    assert should_use_dark(False, 'dark') is False     # dark chrome, white plot
    assert should_use_dark(False, None) is False
