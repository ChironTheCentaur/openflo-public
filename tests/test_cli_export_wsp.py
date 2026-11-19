"""`openflo-run --export-wsp`: each sample carries the SPILL the pipeline
applied to it, and its gates are named 'Comp-<channel>'.

The pipeline compensates every sample with its own FCS SPILL
(auto_compensate). In FlowJo a gate on the bare channel name is evaluated on
the RAW parameter, and a matrix inside a <Sample> is applied as given. So the
export must put each file's own matrix in its sample and name its gates on
the compensated parameter -- or FlowJo gates data OpenFlo never gated.
"""
import re
import xml.etree.ElementTree as ET

import numpy as np

from openflo import cli_wsp
from tests._fcs_writer import write_float_fcs

T = '{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}'
D = '{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}'
SPILLS = {'alpha': '2,FL1-A,FL2-A,1,0.1,0.02,1',
          'beta': '2,FL1-A,FL2-A,1,0,0,1'}         # an identity SPILL next to a real one


def _fcs(path, spill):
    rng = np.random.default_rng(3)
    ev = rng.uniform(100, 5000, size=(50, 3)).astype(np.float32)
    write_float_fcs(path, ev, ['FSC-A', 'FL1-A', 'FL2-A'], metadata_dict={'SPILL': spill})


def test_cli_export_carries_each_files_own_spill_and_comp_names(tmp_path):
    trial = tmp_path / 'trial'
    trial.mkdir()
    for name, spill in SPILLS.items():
        _fcs(trial / f'{name}.fcs', spill)
    gates = [{'id': 'r', 'parent_id': None, 'kind': 'interval', 'channel': 'FL1-A',
              'lo': 100.0, 'hi': 1e5, 'name': 'FL1+'},
             {'id': 's', 'parent_id': 'r', 'kind': 'interval', 'channel': 'FSC-A',
              'lo': 100.0, 'hi': 1e5, 'name': 'Big'}]
    out = tmp_path / 'p.wsp'
    cli_wsp.export_pipeline_workspace(str(out), [str(trial)],
                                      [{'name': 'G', 'samples': ['alpha', 'beta']}], {}, gates)
    samples = {s.find('SampleNode').get('name'): s
               for s in ET.parse(out).getroot().iter('Sample')}
    assert sorted(samples) == ['alpha', 'beta']
    for name, spill in SPILLS.items():
        m = samples[name].find(f'{T}spilloverMatrix')
        assert m is not None, f'{name}: no per-sample matrix'
        vals = [float(c.get(f'{T}value')) for sp in m.findall(f'{T}spillover')
                for c in sp.findall(f'{T}coefficient')]
        assert vals == [float(v) for v in spill.split(',')[3:]], name
        dims = [d.get(f'{D}name') for d in samples[name].iter(f'{D}fcs-dimension')]
        assert dims == ['Comp-FL1-A', 'FSC-A'], name


def test_cli_export_writes_no_matrix_for_a_nan_spill(tmp_path, capsys):
    """A file whose $SPILL has a NaN cell loads uncompensated in the run, so
    its sample in the .wsp carries no matrix and its gates name the bare
    channel. The export still succeeds, the other file keeps its own
    matrix, and no 'nan' is written anywhere in the workspace outside the
    file's own keyword copy (<Keywords> lists the FCS TEXT as read)."""
    trial = tmp_path / 'trial'
    trial.mkdir()
    _fcs(trial / 'alpha.fcs', SPILLS['alpha'])
    _fcs(trial / 'gamma.fcs', '2,FL1-A,FL2-A,1,nan,0.02,1')
    gates = [{'id': 'r', 'parent_id': None, 'kind': 'interval', 'channel': 'FL1-A',
              'lo': 100.0, 'hi': 1e5, 'name': 'FL1+'}]
    out = tmp_path / 'p.wsp'
    cli_wsp.export_pipeline_workspace(str(out), [str(trial)],
                                      [{'name': 'G', 'samples': ['alpha', 'gamma']}], {}, gates)
    said = capsys.readouterr().out
    assert 'write failed' not in said
    assert f'wrote 2 sample(s) + spillover -> {out}' in said
    assert 'gamma.fcs: spillover read failed' in said and 'not a finite number' in said
    root = ET.parse(out).getroot()
    written = [(el.tag, v) for el in root.iter() if el.tag != 'Keyword'
               for v in el.attrib.values()]
    assert written and not [tv for tv in written
                            if re.search(r'(?i)\b(nan|inf)\b', tv[1])]
    samples = {s.find('SampleNode').get('name'): s
               for s in ET.parse(out).getroot().iter('Sample')}
    assert sorted(samples) == ['alpha', 'gamma']
    assert samples['gamma'].find(f'{T}spilloverMatrix') is None
    dims = [d.get(f'{D}name') for d in samples['gamma'].iter(f'{D}fcs-dimension')]
    assert dims == ['FL1-A']
    m = samples['alpha'].find(f'{T}spilloverMatrix')
    vals = [float(c.get(f'{T}value')) for sp in m.findall(f'{T}spillover')
            for c in sp.findall(f'{T}coefficient')]
    assert vals == [1.0, 0.1, 0.02, 1.0]
