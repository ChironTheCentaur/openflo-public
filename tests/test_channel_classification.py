"""Scatter, time and instrument parameters are never fluorescence channels.

`FlowSample._classify_channels` puts every parameter that is not scatter (or
an analysis column) on the fluorescence list, and `apply_transform` logicle-
transforms that list, which also makes it the clustering marker set. The test
was a case-SENSITIVE substring match on 'FSC' / 'SSC' / 'Time' / 'time', so
Beckman's 'FS INT' / 'SS INT' / 'TIME', a lower-case 'fsc-a', Sony's back-
scatter 'BSC-A' and CyTOF's 'Event_length' were all called fluorescence:
TIME (0..1000 s) became 0.11..0.45 logicle units and was clustered on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample, _is_excluded, _is_scatter

# Names as cytometers write $PnN.
NOT_FLUOR = {
    'BD': ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W', 'Time'],
    'CytoFLEX': ['FSC-Width', 'SSC-Width', 'VSSC-A', 'VSSC-H',
                 'Violet SSC-A'],
    'Beckman Navios / Gallios / FC500': [
        'FS INT', 'FS PEAK', 'FS TOF', 'SS INT', 'SS PEAK', 'SS TOF', 'TIME',
        'FS Lin', 'SS Log'],
    'Cytek': ['SSC-B-A', 'SSC-B-H'],
    'Sony': ['BSC-A', 'BSC-H', 'BSC-W'],
    'Attune': ['BSSC-A', 'VSSC1-A'],
    'Guava': ['FSC-HLin', 'SSC-HLin'],
    'CyTOF': ['Event_length', 'Center', 'Offset', 'Width', 'Residual'],
    'other spellings': ['fsc-a', 'ssc-a', 'time', 'Time (s)', 'TIMESTAMP',
                        'FSC_A', 'FSC.A', 'FSC', 'Trigger Pulse Width'],
}
FLUOR = [
    'FITC-A', 'PE-A', 'PerCP-Cy5-5-A', 'PE-Cy5-A', 'FL2-A', 'FL9-A',
    'BV786-A', 'BV605-A', 'BUV395-A', 'BUV737-A', 'Alexa Fluor 488-A',
    'PE-CF594-A', 'PE-Texas Red-A', 'DAPI-A', 'DAPI-W', 'Hoechst Red-A',
    'Comp-FITC-A', 'Pacific Blue-A', 'AmCyan-A', 'Qdot 605-A',
    'FL1 INT', 'FL1 PEAK', 'FL10 INT', 'FL1 Log', 'FL1-A',
    'UV1-A', 'V16-A', 'B14-A', 'YG1-A', 'R8-A',                    # Cytek
    'BL1-A', 'RL1-A', 'VL1-A', 'YL1-A',                            # Attune
    'PB450-A', 'KO525-A', 'ECD-A', 'PC5.5-A', 'PC5-A', 'FL2-A750-A',
    'GRN-HLin', 'RED2-HLin',                                       # Guava
    '89Y', '141Pr', 'Ir191Di', 'Bi209Di',                          # CyTOF
    'BB515-A', 'RB545-A', 'Spark NIR 685-A', 'NovaFluor Blue 610-A',
    'cFluor V420-A', 'StarBright UltraViolet 400-A', 'Super Bright 436-A',
    'eFluor 450-A', 'Zombie NIR-A', 'B530-A', 'Y586-A', 'SSEA4-A', 'BSA-A',
]


@pytest.mark.parametrize('name', [n for ns in NOT_FLUOR.values() for n in ns])
def test_scatter_time_and_instrument_names_are_not_fluorescence(name):
    s = FlowSample.from_dataframe(pd.DataFrame({name: [1.0, 2.0],
                                                'FITC-A': [3.0, 4.0]}))
    assert name not in s.fluor_channels, (
        f'{name!r} would be logicle-transformed and clustered as a marker')
    assert _is_scatter(name)


@pytest.mark.parametrize('name', FLUOR)
def test_fluorescence_names_stay_fluorescence(name):
    assert not _is_scatter(name) and not _is_excluded(name)
    s = FlowSample.from_dataframe(pd.DataFrame({name: [1.0, 2.0]}))
    assert s.fluor_channels == [name]


@pytest.mark.parametrize('name', ['Time', 'TIME', 'time'])
def test_time_is_never_a_qc_detector_in_any_case(name):
    """AcquisitionQC judges every numeric column `_is_excluded` lets through;
    'TIME' slipped past the case-sensitive list."""
    assert _is_excluded(name)


def test_a_beckman_file_keeps_time_and_scatter_linear(tmp_path):
    """Through the loader: FCS -> apply_transform. Only FL1 INT changes."""
    import flowio
    n = 500
    rng = np.random.default_rng(0)
    cols = ['TIME', 'FS INT', 'SS INT', 'FL1 INT']
    ev = np.column_stack([np.linspace(0.0, 1000.0, n),
                          rng.uniform(2e4, 2e5, n), rng.uniform(1e4, 1e5, n),
                          rng.lognormal(7, 1, n)]).astype(np.float32)
    path = tmp_path / 'navios.fcs'
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.ravel().tolist(), cols)
    s = FlowSample(str(path))
    s.apply_transform()
    assert s.fluor_channels == ['FL1 INT']
    assert set(s.data_transforms) == {'FL1 INT'}
    for i, c in enumerate(cols[:3]):
        np.testing.assert_allclose(s.data[c].to_numpy(), ev[:, i], rtol=1e-6)
    assert s.data['FL1 INT'].max() < 1.1          # logicle units
