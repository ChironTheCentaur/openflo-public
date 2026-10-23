"""openflo-run --dpi: every PNG the CLI writes takes it, default 300.

The CLI wrote its figures at a hard-coded 150 dpi while the app's
Preferences default for exports is 300. The CLI does not read the
Preferences file (a script gives the same files on every machine), so the
default is the Preferences' own default, and --dpi changes it.

Each figure is drawn for real on a small synthetic sample and the PNG's
pHYs resolution is read back.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import openflo.cli as cli
from openflo.cli_args import DEFAULT_DPI, build_parser
from openflo.pipeline import FlowSample


def _png_dpi(path):
    from PIL import Image
    with Image.open(path) as im:
        return round(im.info['dpi'][0])


def _sample(path, name):
    s = FlowSample(path)
    s.apply_transform()
    s.set_labels({'FL1-A': 'CD901b', 'FL2-A': 'CD902', 'FL3-A': 'CD903'})
    s.name = name
    fl1 = s.data[s._resolve('FL1-A')].to_numpy()
    s.data['cluster'] = (fl1 > np.median(fl1)).astype(int)
    return s


def test_parser_dpi_defaults_to_the_preferences_default():
    from openflo.theme import EXPORT_DPI
    assert build_parser().parse_args([]).dpi == DEFAULT_DPI == EXPORT_DPI == 300
    assert build_parser().parse_args(['--dpi', '150']).dpi == 150
    for bad in ('10', 'high', '5000'):
        with pytest.raises(SystemExit):
            build_parser().parse_args(['--dpi', bad])


@pytest.mark.parametrize('dpi', [None, 150])
def test_sample_plots_take_the_dpi(tmp_path, synthetic_fcs, dpi):
    s = _sample(synthetic_fcs, 'sample_a')
    kw = {} if dpi is None else {'dpi': dpi}
    cli.save_plots(s, str(tmp_path), **kw)
    want = dpi or 300
    for kind in ('scatter', 'fsc_ssc', 'heatmap'):
        assert _png_dpi(tmp_path / f'sample_a_{kind}.png') == want, kind


@pytest.mark.parametrize('dpi', [None, 150])
def test_pair_scatters_take_the_dpi(tmp_path, synthetic_fcs, dpi):
    samples = [_sample(synthetic_fcs, 'sample_a'),
               _sample(synthetic_fcs, 'sample_b')]
    kw = {} if dpi is None else {'dpi': dpi}
    cli.save_group_pair_scatters(samples, 'Grp', str(tmp_path), **kw)
    x, y = cli.DEFAULT_SCATTER_PAIRS[0]
    for kind in ('overlay', 'grid'):
        assert _png_dpi(tmp_path / f'Grp_{x}_{y}_{kind}.png') == (dpi or 300)


@pytest.mark.parametrize('workers', [1, 2])
def test_group_umaps_take_the_dpi_in_and_out_of_workers(tmp_path,
                                                        synthetic_fcs,
                                                        monkeypatch, workers):
    """_draw_group_plots hands the dpi to its worker processes too."""
    monkeypatch.setattr(cli, 'save_group_pair_scatters', lambda *a, **k: None)
    jobs = [(g, [_sample(synthetic_fcs, f'{g}_s')], g, g)
            for g in ('G1', 'G2')]
    cli._draw_group_plots(jobs, str(tmp_path), 42, None, workers,
                          lambda m: None, dpi=150)
    assert _png_dpi(tmp_path / 'G1_umap.png') == 150
    assert _png_dpi(tmp_path / 'G2_umap.png') == 150
    import matplotlib.pyplot as plt
    plt.close('all')


def test_group_umap_defaults_to_300(tmp_path, synthetic_fcs):
    cli.run_group_umap([_sample(synthetic_fcs, 'sample_a')], 'G1',
                       str(tmp_path))
    assert _png_dpi(tmp_path / 'G1_umap.png') == 300
    import matplotlib.pyplot as plt
    plt.close('all')


def test_unmix_reference_spectra_png_takes_the_dpi(tmp_path):
    spectra = np.array([[1.0, 0.5, 0.1], [0.1, 0.6, 1.0]])
    qc = {'condition_number': 2.0, 'similarity': np.eye(2),
          'similar_pairs': [], 'ssm': np.zeros((2, 2)), 'worst_spread': []}
    for sub, kw in (('d', {}), ('e', {'dpi': 150})):
        os.makedirs(tmp_path / sub)
        cli._write_unmix_qc(str(tmp_path / sub), qc, spectra, ['F1', 'F2'],
                            ['D1', 'D2', 'D3'], **kw)
    assert _png_dpi(tmp_path / 'd' / 'reference_spectra.png') == 300
    assert _png_dpi(tmp_path / 'e' / 'reference_spectra.png') == 150
    assert os.path.isfile(tmp_path / 'd' / 'spectral_qc.md')
