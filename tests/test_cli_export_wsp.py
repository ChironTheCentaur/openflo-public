"""`openflo-run --export-wsp`: each sample carries the SPILL the pipeline
applied to it, and its gates are named 'Comp-<channel>'.

The pipeline compensates every sample with its own FCS SPILL
(auto_compensate). In FlowJo a gate on the bare channel name is evaluated on
the RAW parameter, and a matrix inside a <Sample> is applied as given. So the
export must put each file's own matrix in its sample and name its gates on
the compensated parameter -- or FlowJo gates data OpenFlo never gated.
"""
import xml.etree.ElementTree as ET

import numpy as np

from openflo import cli

T = '{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}'
D = '{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}'
SPILLS = {'alpha': '2,FL1-A,FL2-A,1,0.1,0.02,1',
          'beta': '2,FL1-A,FL2-A,1,0,0,1'}         # an identity SPILL next to a real one


def _fcs(path, spill):
    import flowio
    rng = np.random.default_rng(3)
    ev = rng.uniform(100, 5000, size=(50, 3)).astype(np.float32)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.ravel().tolist(), ['FSC-A', 'FL1-A', 'FL2-A'],
                          metadata_dict={'SPILL': spill})


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
    cli._export_pipeline_workspace(str(out), [str(trial)],
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
