"""The CLI's FMO threshold is reported with its intensity, not bare.

FMOGater.compute takes the cut on the FMO's transformed values (the space the
samples are gated in) and logged it bare: 'FL1-A: threshold=0.487 (p99.5)',
where the cut is at 1,369 in intensity. The log now names the transform and the
linear value the cut corresponds to.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import openflo.pipeline as fp


def test_the_logged_threshold_names_its_scale_and_linear_value(caplog):
    rng = np.random.default_rng(0)
    lin = rng.lognormal(np.log(300), 0.6, 20000)
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame({'FSC-A': rng.normal(5e4, 5e3, lin.size), 'FL1-A': lin}),
        name='fmo')
    s.apply_transform()
    g = fp.FMOGater()
    g.fmos['FL1-A'] = s
    with caplog.at_level(logging.INFO, logger=fp.log.name):
        cut = g.compute(percentile=99.5)['FL1-A']
    want = float(np.percentile(lin, 99.5))
    # The cut itself is unchanged: logicle, where the samples are gated.
    assert cut == float(np.percentile(s.data['FL1-A'], 99.5)) < 1.1
    line, = [r.getMessage() for r in caplog.records if '[FMO] FL1-A' in
             r.getMessage()]
    shown = float(line.split('(= ')[1].split(' linear')[0].replace(',', ''))
    assert 'logicle' in line and abs(shown - want) / want < 0.01, line


def test_the_fmo_percentile_defaults_to_99_on_every_surface(tmp_path):
    """The app's FMO dialog starts at 99; the CLI and FMOGater used 99.5."""
    import inspect

    import flowio

    from openflo import cli
    from openflo.cli_args import build_parser
    rng = np.random.default_rng(1)
    lin = rng.lognormal(np.log(300), 0.6, 20000)
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame({'FSC-A': rng.normal(5e4, 5e3, lin.size), 'FL1-A': lin}),
        name='fmo')
    s.apply_transform()
    g = fp.FMOGater()
    g.fmos['FL1-A'] = s
    cut = g.compute()['FL1-A']
    assert cut == float(np.percentile(s.data['FL1-A'], 99.0))
    assert cut != float(np.percentile(s.data['FL1-A'], 99.5))
    assert build_parser().parse_args([]).fmo_percentile == 99.0
    assert inspect.signature(cli.run).parameters[
        'fmo_percentile'].default == 99.0
    # build_fmo_thresholds, called without a percentile, on an FMO file.
    ev = np.column_stack([rng.normal(5e4, 5e3, lin.size), lin])
    with open(tmp_path / 'expt_fmo1.fcs', 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).ravel().tolist(),
                          ['FSC-A', 'FL1-A'])
    got = cli.build_fmo_thresholds(str(tmp_path), {'FL1-A': 'fmo1'}, 'set')
    ref = fp.FMOGater()
    ref.add_fmo('FL1-A', str(tmp_path / 'expt_fmo1.fcs'))
    ref.prepare()
    assert got == ref.compute(percentile=99.0)
    assert got != ref.compute(percentile=99.5)
