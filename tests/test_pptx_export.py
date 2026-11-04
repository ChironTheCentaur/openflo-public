"""Layout editor → Export to PowerPoint: a .pptx with a slide per page (and,
in a batch, per sample), each the rendered page filling a slide of the
page's size. Written with zipfile alone; checked here part by part: every
XML part parses, every relationship points at a part that exists, and the
slides carry the pictures in order.

Generated pictures: small PNGs drawn with matplotlib."""
from __future__ import annotations

import io
import posixpath
import xml.etree.ElementTree as ET
import zipfile

import pytest

from openflo.pptx_export import EMU_PER_INCH, slide_title, write_pptx

P = '{http://schemas.openxmlformats.org/presentationml/2006/main}'
R = '{http://schemas.openxmlformats.org/package/2006/relationships}'


def _png(text):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    fig = Figure(figsize=(2, 1.5))
    fig.text(0.5, 0.5, text)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=40)
    return buf.getvalue()


def check_package(path):
    """``(slide size in EMU, [(slide name, picture bytes)])`` after checking
    the package hangs together."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        assert '[Content_Types].xml' in names
        for n in names:
            if n.endswith(('.xml', '.rels')):
                ET.fromstring(z.read(n))            # well formed
        for n in [n for n in names if n.endswith('.rels')]:
            base = posixpath.dirname(posixpath.dirname(n))
            for rel in ET.fromstring(z.read(n)).iter(R + 'Relationship'):
                target = posixpath.normpath(
                    posixpath.join(base, rel.get('Target')))
                assert target in names, (n, rel.get('Target'))
        types = z.read('[Content_Types].xml').decode()
        pres = ET.fromstring(z.read('ppt/presentation.xml'))
        sz = pres.find(P + 'sldSz')
        rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(
            z.read('ppt/_rels/presentation.xml.rels')).iter(R + 'Relationship')}
        slides = []
        for sid in pres.find(P + 'sldIdLst'):
            part = 'ppt/' + rels[sid.get(
                '{http://schemas.openxmlformats.org/officeDocument/2006/'
                'relationships}id')]
            assert f'PartName="/{part}"' in types
            name = ET.fromstring(z.read(part)).find(P + 'cSld').get('name')
            srels = ET.fromstring(z.read(part.replace(
                'slides/', 'slides/_rels/') + '.rels'))
            img = [r.get('Target') for r in srels.iter(R + 'Relationship')
                   if r.get('Type').endswith('/image')]
            slides.append((name, z.read(posixpath.normpath(
                posixpath.join('ppt/slides', img[0])))))
        return (int(sz.get('cx')), int(sz.get('cy'))), slides


def test_slides_in_order_at_the_page_size(tmp_path):
    pics = [_png('one'), _png('two & <three>')]
    out = tmp_path / 'deck.pptx'
    n = write_pptx(str(out), [(pics[0], 's1: Page 1'),
                              (pics[1], 's2: "Page" & <2>')], (11.0, 8.5))
    assert n == 2
    size, slides = check_package(out)
    assert size == (11 * EMU_PER_INCH, int(8.5 * EMU_PER_INCH))
    assert slides == [('s1: Page 1', pics[0]),
                      ('s2: "Page" & <2>', pics[1])]


def test_refuses_no_slides_and_non_png(tmp_path):
    with pytest.raises(ValueError):
        write_pptx(str(tmp_path / 'a.pptx'), [], (11, 8.5))
    with pytest.raises(ValueError):
        write_pptx(str(tmp_path / 'b.pptx'), [(b'GIF89a...', 'x')], (11, 8.5))


def test_slide_title():
    assert slide_title('s1', 'Page 1') == 's1: Page 1'
    assert slide_title('s1', '') == 's1'


def test_layout_editor_exports_a_page_and_a_batch(tmp_path, monkeypatch):
    from tests.test_layout_editor import _editor_or_skip, _load
    root, ed = _editor_or_skip()
    try:
        import openflo.ui_layout_editor as ule
        from openflo.async_task import wait_for_job
        _load(ed, 's1', 1, 400, 300.0)
        _load(ed, 's2', 2, 300, 500.0)
        win = ule.LayoutEditorWindow(ed)
        win.update()
        win._add_text()
        win._new_page()
        win._add_text()
        win.layout.page = 'A4 portrait'      # the first page sets the size
        one = tmp_path / 'one.pptx'
        monkeypatch.setattr(ule.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(one))
        win._export_page()
        size, slides = check_package(one)
        assert len(slides) == 1
        assert size == (int(8.27 * EMU_PER_INCH), int(11.69 * EMU_PER_INCH))

        many = tmp_path / 'many.pptx'
        monkeypatch.setattr(ule.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(many))
        win._export_batch()
        assert wait_for_job(win._busy)
        size, slides = check_package(many)
        assert [nm for nm, _png in slides] == [
            's1: Page 1', 's1: Page 2', 's2: Page 1', 's2: Page 2']
        assert size == (11 * EMU_PER_INCH, int(8.5 * EMU_PER_INCH))
        assert all(png.startswith(b'\x89PNG') for _nm, png in slides)
        win.destroy()
    finally:
        root.destroy()
