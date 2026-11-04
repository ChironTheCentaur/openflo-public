"""phenograph's progress prints go to the log, not stdout.

phenograph prints seven progress lines per call ("Finding 15 nearest
neighbors ...", "PhenoGraph completed in 0.47 seconds") straight to stdout,
so every clustering, FlowExperiment.cluster_jointly included, wrote them
unlabelled and --quiet could not silence them.
cluster_graph.phenograph_prints_to_log sends them to the
``openflo.cluster_graph`` logger for OpenFlo's own calls, on the calling
thread only: progress at DEBUG, anything else (phenograph's Louvain failure
diagnostics) at WARNING. Other callers and other threads still print.
"""
import importlib
import logging
import threading

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('phenograph')

LOGGER = 'openflo.cluster_graph'


def _sample(rng, name, f_hi, n=1500):
    """Two synthetic populations on FL1-A / FL2-A, f_hi of them FL1-A high."""
    from openflo.pipeline import FlowSample
    n1 = int(n * f_hi)
    n2 = n - n1
    df = pd.DataFrame({
        'FSC-A': rng.normal(5e4, 5e3, n),
        'FL1-A': np.r_[rng.normal(0.8, 0.03, n1), rng.normal(0.3, 0.03, n2)],
        'FL2-A': np.r_[rng.normal(0.3, 0.03, n1), rng.normal(0.8, 0.03, n2)]})
    s = FlowSample.from_dataframe(df, name=name)
    s.fluor_channels = ['FL1-A', 'FL2-A']
    return s


def _experiment():
    from openflo.pipeline import FlowExperiment
    rng = np.random.default_rng(5)
    exp = FlowExperiment.__new__(FlowExperiment)
    exp.samples = {nm: _sample(rng, nm, f)
                   for nm, f in (('A1', .8), ('B1', .2))}
    return exp


def _pg_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == LOGGER and r.levelno == logging.DEBUG
            and r.getMessage().startswith('phenograph: ')]


def test_joint_clustering_prints_nothing_and_logs_phenograph(capsys, caplog):
    exp = _experiment()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        labels = exp.cluster_jointly(method='phenograph', k=15)
    out = capsys.readouterr().out
    assert out == '', out
    lines = _pg_lines(caplog)
    assert any('nearest neighbors' in m for m in lines), lines
    assert any(m.startswith('phenograph: PhenoGraph completed in')
               for m in lines), lines
    # Still a real clustering: no cluster mixes the two planted populations.
    a1 = labels['A1']
    assert not set(a1[:1200]) & set(a1[1200:])


def test_single_sample_cluster_prints_nothing(capsys, caplog):
    s = _sample(np.random.default_rng(6), 'S1', .5)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        s.cluster(channels=['FL1-A', 'FL2-A'], k=15, n_jobs=1)
    assert capsys.readouterr().out == ''
    assert _pg_lines(caplog)


def test_other_callers_and_threads_still_print(capsys):
    """Outside the switch phenograph prints as before, and while the switch
    is on, a print from another thread is not captured."""
    from openflo.cluster_graph import install_phenograph_print_to_log, phenograph_prints_to_log
    install_phenograph_print_to_log()
    pgc = importlib.import_module('phenograph.cluster')
    pgc.print('direct caller', flush=True)
    seen = []
    with phenograph_prints_to_log():
        t = threading.Thread(target=lambda: pgc.print('other thread'))
        t.start()
        t.join()
        pgc.print('to the log')
        seen.append(capsys.readouterr().out)
    pgc.print('after')
    out = capsys.readouterr().out
    assert seen == ['direct caller\nother thread\n']
    assert out == 'after\n'


@pytest.mark.parametrize('text', [
    'Could not find Louvain code, tried: /nowhere/louvain',
    'stdout from convert: ',
    'stderr from convert: synthetic.bin: cannot open file'])
def test_diagnostics_reach_the_log_at_warning(capsys, caplog, text):
    """phenograph's Louvain diagnostics are the only clue when its convert
    binary fails; they must not be hidden at DEBUG with the progress."""
    from openflo.cluster_graph import (
        install_phenograph_print_to_log,
        phenograph_prints_to_log,
    )
    install_phenograph_print_to_log()
    pgc = importlib.import_module('phenograph.core')
    with caplog.at_level(logging.DEBUG, logger=LOGGER), \
            phenograph_prints_to_log():
        pgc.print(text)
        pgc.print('Running Louvain modularity optimization', flush=True)
    assert capsys.readouterr().out == ''
    got = [(r.levelno, r.getMessage()) for r in caplog.records
           if r.name == LOGGER]
    assert got == [(logging.WARNING, f'phenograph: {text.strip()}'),
                   (logging.DEBUG, 'phenograph: Running Louvain modularity '
                                   'optimization')]
