"""Document.stats as the phone's Statistics page shows it: each
population's own name and depth, and each column's scale note, the one
the CSV export titles that column with. Generated data only."""
import csv

from openflo.document import Document
from tests.test_document import gated_doc


def test_a_name_holding_a_slash_is_one_name_at_its_depth(tmp_path):
    doc, name, cells, *_ = gated_doc(tmp_path)
    doc.add_gate(name, {'kind': 'threshold', 'channel': 'FL3-A', 'value': 0.5,
                        'name': 'FL3 hi/lo'}, cells)
    got = [(r['name'], r['depth']) for r in doc.stats(name, stats=['Median'])]
    assert got == [('All events', 0), ('cells', 1), ('FL1+', 2), ('FL2 mid', 2), ('FL3 hi/lo', 2)]


def test_each_note_is_the_one_the_csv_titles_the_column_with(tmp_path):
    from tests.test_geomean_flowjo_axis import _axis_xml, _fcs, _wsp
    fcs, _ev = _fcs(tmp_path / 'wsp.fcs')
    wsp = _wsp(tmp_path / 'w.wsp', fcs, [_axis_xml('Comp-FL1-A')])
    doc = Document()
    name = doc.import_wsp(wsp)['samples'][0]
    stats = ['GeoMean', 'Median', 'rCV']
    rows = doc.stats(name, channels=['FL1-A', 'FL2-A'], stats=stats)
    out = tmp_path / 'g.csv'
    doc.export_stats_csv(str(out), channels=['FL1-A', 'FL2-A'], stats=stats)
    with open(out, newline='', encoding='utf-8') as f:
        head = next(csv.reader(f))[5:]
    mine = [f"{st} {c} {rows[0]['scale'][c][st.lower()]}" for c in ('FL1-A', 'FL2-A') for st in stats]
    assert sorted(mine) == sorted(head)
    assert 'GeoMean FL1-A (linear, FlowJo axis: logicle W=1 M=4.42)' in mine
    assert all(r['scale'] == rows[0]['scale'] for r in rows)
