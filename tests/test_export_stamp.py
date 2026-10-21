"""Every exported file names the OpenFlo version that wrote it (and the
corrections applied) in its own metadata, never drawn over the data.

Each format is written, its metadata read back the way another program
would read it, and the content checked unchanged by the stamp. The version
is replaced by a test value so a stamp from anywhere else cannot pass.
Generated data only; channels FL1-A, FL2-A, ...."""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

from matplotlib.figure import Figure  # noqa: E402

from openflo import export_stamp  # noqa: E402

V = '9.8.7-test'
FIXES = ['fix-001', 'fix-002']


@pytest.fixture(autouse=True)
def _test_version(monkeypatch):
    monkeypatch.setattr(export_stamp, '_version', lambda: V)


def _fig():
    fig = Figure(figsize=(2, 1.5))
    ax = fig.add_subplot()
    ax.plot([0, 1, 2], [1, 3, 2])
    ax.set_xlabel('FL1-A')
    return fig


def _events(n=200):
    rng = np.random.default_rng(7)
    return pd.DataFrame({'FSC-A': rng.uniform(1e4, 9e4, n),
                         'FL1-A': rng.normal(2000, 300, n),
                         'FL2-A': rng.normal(800, 200, n)})


# ── the helper ───────────────────────────────────────────────────────────────

def test_export_metadata_names_version_and_corrections():
    assert export_stamp.export_metadata() == {
        'Software': f'OpenFlo {V}', 'OpenFlo version': V}
    meta = export_stamp.export_metadata(FIXES)
    assert meta['Corrections'] == 'fix-001; fix-002'
    assert export_stamp.stamp_text(FIXES) == \
        f'OpenFlo {V}; corrections: fix-001; fix-002'


def test_corrections_come_from_the_editor_attribute():
    class Ed:
        pass
    ed = Ed()
    assert export_stamp.corrections_for(ed) is None
    ed._applied_fixes = []
    assert export_stamp.corrections_for(ed) is None
    ed._applied_fixes = list(FIXES)
    assert export_stamp.corrections_for(ed) == FIXES


# ── figures ──────────────────────────────────────────────────────────────────

def test_png_text_chunks_and_pixels_unchanged(tmp_path, monkeypatch):
    from PIL import Image

    from openflo.theme import savefig_background
    out = tmp_path / 'plot.png'
    savefig_background(_fig(), str(out), corrections=FIXES)
    with Image.open(out) as im:
        text = dict(im.text)
    assert text['Software'] == f'OpenFlo {V}'
    assert text['OpenFlo version'] == V
    assert text['Corrections'] == 'fix-001; fix-002'
    # Metadata only: the export draws the same pixels as one without it.
    fig = _fig()
    a, b = io.BytesIO(), io.BytesIO()
    with monkeypatch.context() as m:
        m.setattr(export_stamp, 'savefig_metadata', lambda *a, **k: None)
        savefig_background(fig, b, dpi=50, fmt='png', corrections=FIXES)
    savefig_background(fig, a, dpi=50, fmt='png', corrections=FIXES)
    with Image.open(a) as im_a, Image.open(b) as im_b:
        assert 'OpenFlo version' in im_a.text
        assert 'OpenFlo version' not in im_b.text
    pa, pb = (np.asarray(Image.open(x).convert('RGBA')) for x in (a, b))
    assert np.array_equal(pa, pb)


def test_pdf_info_dictionary(tmp_path):
    from openflo.theme import savefig_background
    out = tmp_path / 'plot.pdf'
    savefig_background(_fig(), str(out), corrections=FIXES)
    raw = out.read_bytes()
    # The Info dictionary is written uncompressed.
    assert f'/Creator (OpenFlo {V})'.encode() in raw
    assert (f'/Keywords (OpenFlo {V}; corrections: fix-001; fix-002)'
            .encode()) in raw


def test_svg_metadata(tmp_path):
    from openflo.theme import savefig_background
    out = tmp_path / 'plot.svg'
    savefig_background(_fig(), str(out))
    root = ET.parse(out).getroot()
    dc = '{http://purl.org/dc/elements/1.1/}'
    creator = root.find(f'.//{dc}creator')
    assert creator is not None
    assert ''.join(creator.itertext()).strip() == f'OpenFlo {V}'
    desc = root.find(f'.//{dc}description')
    assert desc is not None and desc.text == f'OpenFlo {V}'


def test_no_metadata_for_formats_that_keep_none():
    for fmt in ('jpg', 'tif'):
        assert export_stamp.savefig_metadata(fmt=fmt) is None
    assert export_stamp.figure_format('a/b.PNG') == 'png'


def test_html_report_generator_tag():
    from openflo.report import build_html_report
    doc = build_html_report('r', corrections=FIXES)
    assert (f'<meta name="generator" content="OpenFlo {V}; corrections: '
            'fix-001; fix-002">') in doc


# ── PowerPoint ───────────────────────────────────────────────────────────────

def test_pptx_core_properties(tmp_path):
    from openflo.pptx_export import write_pptx
    from tests.test_pptx_export import check_package
    buf = io.BytesIO()
    _fig().savefig(buf, format='png', dpi=40)
    png = buf.getvalue()
    out = tmp_path / 'deck.pptx'
    write_pptx(str(out), [(png, 'S1: page')], (4, 3), corrections=FIXES)
    with zipfile.ZipFile(out) as z:
        core = ET.fromstring(z.read('docProps/core.xml'))
    cp = '{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}'
    dc = '{http://purl.org/dc/elements/1.1/}'
    assert core.findtext(f'{cp}version') == V
    assert core.findtext(f'{cp}keywords') == \
        f'OpenFlo {V}; corrections: fix-001; fix-002'
    assert core.findtext(f'{dc}description') == core.findtext(f'{cp}keywords')
    # The package still hangs together and carries the picture as given.
    _size, slides = check_package(str(out))
    assert slides == [('S1: page', png)]


# ── FCS ──────────────────────────────────────────────────────────────────────

def test_fcs_keywords_replace_a_carried_stamp(tmp_path):
    import openflo.pipeline as fp
    from openflo.fcs_export import write_fcs
    df = _events()
    out = tmp_path / 'pop.fcs'
    # A source written by an older OpenFlo carries its own stamp, which
    # source_keywords copies; the new file must name the version writing it.
    write_fcs(df, str(out), keywords={'OPENFLO_VERSION': '0.0.1',
                                      'OPENFLO_CORRECTIONS': 'old'})
    s = fp.FlowSample(str(out))
    assert s.metadata['openflo_version'] == V
    assert 'openflo_corrections' not in s.metadata
    np.testing.assert_allclose(s.raw[list(df.columns)].to_numpy(),
                               df.to_numpy(), rtol=1e-6)
    write_fcs(df, str(out), corrections=FIXES)
    assert fp.FlowSample(str(out)).metadata['openflo_corrections'] == \
        'fix-001; fix-002'


def test_pipeline_write_fcs_is_stamped(tmp_path):
    import openflo.pipeline as fp
    df = _events()
    out = tmp_path / 'ev.fcs'
    fp.write_fcs(str(out), df)
    s = fp.FlowSample(str(out))
    assert s.metadata['openflo_version'] == V
    np.testing.assert_allclose(s.raw[list(df.columns)].to_numpy(),
                               df.to_numpy(), rtol=1e-6)


# ── Excel ────────────────────────────────────────────────────────────────────

def test_workbook_properties_and_sheets_unchanged(tmp_path):
    import openpyxl

    from openflo.table_export import about_rows, write_workbook
    df = pd.DataFrame({'Sample': ['S1', 'S2'], 'Population': ['P1', 'P1'],
                       'Count': [10, 20], '%Parent': [12.5, 40.0]})
    out = tmp_path / 'stats.xlsx'
    write_workbook(str(out), {'Stats': df}, about=about_rows(FIXES),
                   corrections=FIXES)
    wb = openpyxl.load_workbook(out)
    assert wb.properties.creator == f'OpenFlo {V}'
    assert wb.properties.keywords == \
        f'OpenFlo {V}; corrections: fix-001; fix-002'
    custom = {p.name: p.value for p in wb.custom_doc_props.props}
    assert custom == {'OpenFlo version': V, 'Corrections': 'fix-001; fix-002'}
    pd.testing.assert_frame_equal(pd.read_excel(out, sheet_name='Stats'), df)
    about = pd.read_excel(out, sheet_name='About')
    rows = dict(zip(about['Item'], about['Value'], strict=True))
    assert rows['Saved by'] == f'OpenFlo {V}'
    assert rows['Corrections'] == 'fix-001; fix-002'


# ── CSV (events export) ──────────────────────────────────────────────────────

def test_events_csv_stays_plain_and_its_record_names_the_version(tmp_path):
    """The CSV stays a plain table (OpenFlo's own loader and every analysis
    tool read its first line as the header); the record beside it says
    which OpenFlo wrote it."""
    import json

    import openflo.pipeline as fp
    df = _events()
    out = tmp_path / 'S1_processed.csv'
    df.to_csv(out, index=False)
    side = fp.write_transforms_sidecar(str(out), {}, df.columns, len(df),
                                       corrections=FIXES)
    with open(side, encoding='utf-8') as fh:
        rec = json.load(fh)
    assert rec['openflo_version'] == V
    assert rec['corrections'] == 'fix-001; fix-002'
    assert fp.read_transforms_sidecar(str(out), df.columns, len(df)) == {}
    assert out.read_text(encoding='utf-8').splitlines()[0] == \
        'FSC-A,FL1-A,FL2-A'
    back, _notes = fp.read_events_csv(str(out))
    pd.testing.assert_frame_equal(back, df)


# ── FlowJo workspace and Gating-ML ───────────────────────────────────────────

_STAMP = re.compile(r'<!-- Written by OpenFlo (.*?) -->\n')


def test_wsp_comment_and_matrix_unchanged(tmp_path):
    import openflo.pipeline as fp
    chans = ['FL1-A', 'FL2-A']
    mtx = np.array([[1.0, 0.12], [0.03, 1.0]])
    out = tmp_path / 'comp.wsp'
    fp.write_compensation_matrix(str(out), mtx, chans)
    text = out.read_text(encoding='utf-8')
    lines = text.splitlines()
    assert lines[0].startswith('<?xml')
    assert lines[1] == f'<!-- Written by OpenFlo {V} -->'
    m = fp.WspReader(str(out)).get_matrix()
    assert list(m['channels']) == chans
    np.testing.assert_allclose(m['matrix'], mtx)

    w = fp.WspWriter()
    w.corrections = FIXES
    w.set_compensation(chans, mtx)
    w.write(str(out))
    assert _STAMP.search(out.read_text(encoding='utf-8')).group(1) == \
        f'{V}; corrections: fix-001; fix-002'


def test_gatingml_comment_and_gates_unchanged(tmp_path):
    from openflo.gatingml import read_gatingml, write_gatingml
    gate = {'id': 'r', 'parent_id': None, 'kind': 'rect', 'name': 'R',
            'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
            'x0': 100.0, 'x1': 900.0, 'y0': 50.0, 'y1': 700.0}
    out = tmp_path / 'gates.xml'
    write_gatingml(str(out), [gate], corrections=FIXES)
    text = out.read_text(encoding='utf-8')
    assert _STAMP.search(text).group(1) == \
        f'{V}; corrections: fix-001; fix-002'
    bare = tmp_path / 'bare.xml'
    bare.write_text(_STAMP.sub('', text), encoding='utf-8')
    assert read_gatingml(str(out)) == read_gatingml(str(bare))
    (got,), _w = read_gatingml(str(out))
    assert (got['x0'], got['x1'], got['y0'], got['y1']) == \
        (100.0, 900.0, 50.0, 700.0)


def test_xml_comment_is_well_formed():
    c = export_stamp.xml_comment(['a--b'])
    assert '--' not in c[4:-3]
    ET.fromstring('<r>' + c + '</r>')
