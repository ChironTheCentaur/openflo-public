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


def test_names_lose_characters_xml_cannot_hold(tmp_path):
    """A control character in a sample or page name (an FCS keyword can
    carry one) would leave the slide part unreadable, and PowerPoint then
    refuses the whole deck: it is dropped, the rest of the name kept."""
    out = tmp_path / 'deck.pptx'
    write_pptx(str(out), [(_png('a'), 's\x01one\x1f: Page\ufffe 1'),
                          (_png('b'), 'tab\tkept')], (11.0, 8.5))
    _size, slides = check_package(out)
    # A tab is legal XML and is kept (written as a character reference).
    assert [nm for nm, _p in slides] == ['sone: Page 1', 'tab\tkept']
    with zipfile.ZipFile(out) as z:
        alt = ET.fromstring(z.read('ppt/slides/slide1.xml')).find(
            f'.//{P}cNvPr[@id="2"]').get('descr')
    assert alt == 'sone: Page 1'


def test_xml_safe_drops_exactly_what_xml_cannot_hold():
    """fcs_text.xml_safe (shared with the .wsp writer): the edges of each
    dropped range go, their neighbours stay."""
    from openflo.fcs_text import xml_safe
    gone = '\x00\x08\x0b\x0c\x0e\x1f\ud800\udfff\ufffe\uffff'
    kept = '\t\n\r \ud7ff\ue000\ufffd\U0001f600'
    assert xml_safe(gone + kept) == kept


def test_a_name_with_bytes_that_are_not_utf8(tmp_path):
    """A file name with bytes that are not UTF-8 reaches Python as lone
    surrogates (os.fsdecode's surrogateescape on POSIX; decoded the same way
    here so Windows, whose fsdecode is strict UTF-8, checks it too); the deck
    is still written, the name without them."""
    import os
    name = os.path.splitext(
        b'sample_\xff.fcs'.decode('utf-8', 'surrogateescape'))[0]
    assert any(0xD800 <= ord(c) <= 0xDFFF for c in name)
    out = tmp_path / 'deck.pptx'
    write_pptx(str(out), [(_png('a'), f'{name}: Page 1')], (11.0, 8.5))
    _size, slides = check_package(out)
    assert [nm for nm, _p in slides] == ['sample_: Page 1']


def test_a_failure_part_way_leaves_no_file(tmp_path, monkeypatch):
    """A deck that fails while being written leaves nothing at the chosen
    path (no half-written zip) and no partial file beside it; an existing
    file there is left as it was."""
    import openflo.pptx_export as px
    real = px._slide
    calls = []

    def slide(*a):
        calls.append(a)
        if len(calls) == 2:
            raise OSError('disk full')
        return real(*a)
    monkeypatch.setattr(px, '_slide', slide)
    out = tmp_path / 'deck.pptx'
    with pytest.raises(OSError):
        write_pptx(str(out), [(_png('a'), 'x'), (_png('b'), 'y')], (11, 8.5))
    assert list(tmp_path.iterdir()) == []
    out.write_bytes(b'old deck')
    with pytest.raises(OSError):
        calls.clear()
        write_pptx(str(out), [(_png('a'), 'x'), (_png('b'), 'y')], (11, 8.5))
    assert [p.name for p in tmp_path.iterdir()] == ['deck.pptx']
    assert out.read_bytes() == b'old deck'


@pytest.mark.parametrize('size', [(0.99, 8.5), (11.0, 0.5), (56.01, 8.5),
                                  (11.0, 60.0)])
def test_refuses_a_slide_size_powerpoint_cannot_open(tmp_path, size):
    out = tmp_path / 'deck.pptx'
    with pytest.raises(ValueError, match='1 to 56 inches'):
        write_pptx(str(out), [(_png('a'), 'x')], size)
    assert not out.exists()


@pytest.mark.parametrize('size', [(1.0, 1.0), (56.0, 56.0)])
def test_the_limits_themselves_are_written(tmp_path, size):
    out = tmp_path / 'deck.pptx'
    write_pptx(str(out), [(_png('a'), 'x')], size)
    got, _slides = check_package(out)
    assert got == (int(size[0] * EMU_PER_INCH), int(size[1] * EMU_PER_INCH))


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
