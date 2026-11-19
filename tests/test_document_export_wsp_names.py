"""Document.export_wsp(names_only=True): the phone's Save as FlowJo
workspace. Each sample names its FCS file by name alone, so the workspace,
kept in the folder with those files, finds them there. Generated data only."""
import shutil
import xml.etree.ElementTree as ET

from openflo.document import Document
from openflo.document_rpc import METHODS
from openflo.wsp_writer import _wsp_dataset_uri
from tests.test_document import gated_doc, spill_fcs


def _uris(path):
    return [d.get('uri') for d in ET.parse(path).getroot().iter('DataSet')]


def test_each_sample_names_its_file_alone(tmp_path):
    doc, *_ = gated_doc(tmp_path)
    doc.open_fcs(spill_fcs(tmp_path / 'raw' / 'b, c d.fcs', seed=2))
    r = doc.export_wsp(str(tmp_path / 'a.wsp'), names_only=True)
    assert (r['samples'], r['not_written'], r['warnings']) == (2, [], [])
    assert _uris(tmp_path / 'a.wsp') == ['file:s1.fcs', 'file:b,%20c%20d.fcs']
    # Without it, the absolute path, as the desktop writes it.
    doc.export_wsp(str(tmp_path / 'abs.wsp'))
    assert _uris(tmp_path / 'abs.wsp')[0] == _wsp_dataset_uri(str(tmp_path / 'raw' / 's1.fcs'))
    assert _uris(tmp_path / 'abs.wsp')[0].endswith('/raw/s1.fcs')


def test_kept_beside_its_files_it_opens_with_the_same_counts(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    doc.export_wsp(str(tmp_path / 'out.wsp'), names_only=True)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    shutil.copy(tmp_path / 'out.wsp', elsewhere / 'out.wsp')
    shutil.copy(tmp_path / 'raw' / 's1.fcs', elsewhere / 's1.fcs')
    back = Document()
    r = back.import_wsp(str(elsewhere / 'out.wsp'))
    assert r['samples'] == [name] and r['unresolved'] == []
    assert back.samples()[0]['path'] == str(elsewhere / 's1.fcs')

    def counts(d):
        return {x['population']: x['count']
                for x in d.stats(name, channels=[], stats=(), all_events=False)}
    assert counts(back) == counts(doc) and len(counts(doc)) == 3


def test_different_files_of_one_name_are_written_with_a_warning(tmp_path):
    doc = Document()
    a = doc.open_fcs(spill_fcs(tmp_path / 'one' / 'x.fcs', seed=1))
    b = doc.open_fcs(spill_fcs(tmp_path / 'two' / 'x.fcs', seed=2))
    r = doc.export_wsp(str(tmp_path / 'a.wsp'), names_only=True)
    assert r['samples'] == 2
    assert r['warnings'] == [f'{a} and {b} come from different files named x.fcs: '
                             'FlowJo finds one file for all of them']
    # The same file opened twice is one file: nothing to warn of.
    same = Document()
    same.open_fcs(str(tmp_path / 'one' / 'x.fcs'))
    same.open_fcs(str(tmp_path / 'one' / 'x.fcs'))
    assert same.export_wsp(str(tmp_path / 'b.wsp'), names_only=True)['warnings'] == []
    # Absolute paths tell the two apart: no warning either.
    assert doc.export_wsp(str(tmp_path / 'c.wsp'))['warnings'] == []


def test_the_rpc_offers_it():
    assert 'export_wsp' in METHODS


def test_a_name_alone_is_found_beside_the_workspace_not_where_the_app_runs(tmp_path, monkeypatch):
    doc, name, *_ = gated_doc(tmp_path)
    doc.export_wsp(str(tmp_path / 'raw' / 'out.wsp'), names_only=True)
    # Another file of that name where the process runs: never taken.
    cwd = tmp_path / 'cwd'
    spill_fcs(cwd / 's1.fcs', seed=9, n=500)
    monkeypatch.chdir(cwd)
    back = Document()
    r = back.import_wsp(str(tmp_path / 'raw' / 'out.wsp'))
    assert r['samples'] == [name]
    assert back.samples()[0]['path'] == str(tmp_path / 'raw' / 's1.fcs')


def test_the_fcs_folder_chosen_wins_for_a_name_alone_as_for_a_full_path(tmp_path):
    from openflo.compare import FcsLocator
    spill_fcs(tmp_path / 'ws' / 's1.fcs', seed=1)
    chosen = spill_fcs(tmp_path / 'chosen' / 's1.fcs', seed=2)
    loc = FcsLocator(str(tmp_path / 'ws' / 'a.wsp'), str(tmp_path / 'chosen'))
    assert loc.find('file:s1.fcs') == chosen
    # A full path from another computer (a Windows drive here) the same way.
    assert loc.find('file:/Z:/gone/s1.fcs') == chosen
    assert loc.ambiguous == []


def test_a_full_path_that_is_there_is_taken_as_written(tmp_path):
    from openflo.compare import FcsLocator
    elsewhere = spill_fcs(tmp_path / 'elsewhere' / 'only.fcs', seed=3)
    loc = FcsLocator(str(tmp_path / 'ws' / 'a.wsp'))
    assert loc.find(_wsp_dataset_uri(elsewhere)) == elsewhere
