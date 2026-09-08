"""Tests for the staining-panel reader and group pair-scatter outputs
(cli.read_staining_panel / find_panel_xlsx / parse_pairs /
save_group_pair_scatters). All headless (Agg)."""
from __future__ import annotations

import os

import pytest

from openflo import cli
from openflo.pipeline import FlowSample

# ── helpers ───────────────────────────────────────────────────────────────────

def _write_panel(path, rows):
    """Write a tiny .xlsx with the given rows (list of cell-lists)."""
    openpyxl = pytest.importorskip('openpyxl')
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    wb.save(path)


DATA_CHANNELS = ['FSC-A', 'SSC-A', 'BV421-A', 'APC-A',
                 'PE-Cy7-A', 'Time']


# ── read_staining_panel ───────────────────────────────────────────────────────

def test_panel_maps_fluorophores_to_detectors(tmp_path):
    p = tmp_path / 'panel.xlsx'
    # Mirrors the real sheet layout: a blank lead column, CD in col B,
    # fluorophore in col C, with a blank header row.
    _write_panel(p, [[None, None, None],
                     [None, 'CD11b', 'BV421'],
                     [None, 'CD45', 'PE/Cy7'],
                     [None, 'CD34', 'APC']])
    m = cli.read_staining_panel(str(p), DATA_CHANNELS)
    assert m == {'BV421-A': 'CD11b',
                 'PE-Cy7-A': 'CD45',
                 'APC-A': 'CD34'}


def test_panel_column_order_insensitive(tmp_path):
    p = tmp_path / 'panel.xlsx'
    _write_panel(p, [['BV421', 'CD11b'],      # fluor first
                     ['APC', 'CD34']])
    m = cli.read_staining_panel(str(p), DATA_CHANNELS)
    assert m == {'BV421-A': 'CD11b', 'APC-A': 'CD34'}


def test_panel_unmatched_fluor_is_dropped(tmp_path):
    p = tmp_path / 'panel.xlsx'
    _write_panel(p, [['CD19', 'FITC'],        # no FITC channel in the data
                     ['CD11b', 'BV421']])
    m = cli.read_staining_panel(str(p), DATA_CHANNELS)
    assert m == {'BV421-A': 'CD11b'}      # FITC row silently skipped


def test_panel_bad_file_returns_empty(tmp_path):
    p = tmp_path / 'not_a_panel.xlsx'
    p.write_text('garbage', encoding='utf-8')
    assert cli.read_staining_panel(str(p), DATA_CHANNELS) == {}


# ── find_panel_xlsx (ancestor walk) ───────────────────────────────────────────

def test_find_panel_in_ancestor(tmp_path):
    root = tmp_path / 'study'
    day = root / 'day3'
    day.mkdir(parents=True)
    panel = root / 'panel.xlsx'
    _write_panel(panel, [['CD11b', 'BV421']])
    # Searching from the day sub-folder should still find the ancestor panel.
    found = cli.find_panel_xlsx([str(day)])
    assert found is not None
    assert os.path.samefile(found, str(panel))


def test_find_panel_prefers_panel_named(tmp_path):
    d = tmp_path
    _write_panel(d / 'random.xlsx', [['a', 'b']])
    _write_panel(d / 'panel.xlsx', [['CD11b', 'BV421']])
    found = cli.find_panel_xlsx([str(d)])
    assert os.path.basename(found).lower().startswith('panel')


def test_find_panel_ignores_lock_files(tmp_path):
    (tmp_path / '~$panel.xlsx').write_text('lock', encoding='utf-8')
    assert cli.find_panel_xlsx([str(tmp_path)]) is None


# ── parse_pairs ───────────────────────────────────────────────────────────────

def test_parse_pairs_basic():
    assert cli.parse_pairs('CD34/CD11b,CD11b/CD45') == [
        ('CD34', 'CD11b'), ('CD11b', 'CD45')]


def test_parse_pairs_separators_and_blank():
    assert cli.parse_pairs('CD34 vs CD11b') == [('CD34', 'CD11b')]
    assert cli.parse_pairs('') is None
    assert cli.parse_pairs('  ') is None


# ── save_group_pair_scatters ──────────────────────────────────────────────────

def _labelled_sample(path, name):
    s = FlowSample(path)
    s.apply_transform()
    s.set_labels({'BV421-A': 'CD11b', 'APC-A': 'CD34', 'PE-Cy7-A': 'CD45'})
    s.name = name
    return s


def test_pair_scatters_emit_overlay_and_grid(tmp_path, synthetic_fcs):
    samples = [_labelled_sample(synthetic_fcs, 'A'),
               _labelled_sample(synthetic_fcs, 'B')]
    out = tmp_path / 'grp'
    cli.save_group_pair_scatters(samples, 'Grp', str(out))
    produced = sorted(os.listdir(out))
    # 3 default pairs × {overlay, grid}
    assert len(produced) == 6
    for x, y in cli.DEFAULT_SCATTER_PAIRS:
        for kind in ('overlay', 'grid'):
            assert f'Grp_{x}_{y}_{kind}.png' in produced


def test_pair_scatters_skip_unlabelled_pair(tmp_path, synthetic_fcs):
    # No CD labels → no pair resolves → nothing written. The synthetic FCS
    # carries CD labels via opt_channel_names, so reset them to identity.
    s = FlowSample(synthetic_fcs)
    s.apply_transform()
    s.channel_labels = {c: c for c in s.channel_labels}
    s.name = 'A'
    out = tmp_path / 'grp'
    cli.save_group_pair_scatters([s], 'Grp', str(out))
    assert not os.path.exists(out) or os.listdir(out) == []


def test_resolve_pair_missing_channel(synthetic_fcs):
    s = _labelled_sample(synthetic_fcs, 'A')
    assert cli._resolve_pair(s, 'CD34', 'CD11b') != (None, None)
    assert cli._resolve_pair(s, 'CD34', 'CD999') == (None, None)
