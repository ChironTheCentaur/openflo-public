"""A PowerPoint (.pptx) file of pictures, one per slide, with no dependency:
the Office Open XML parts written with ``zipfile``. Tk-free.

The layout editor exports each page (and, in a batch, each sample's pages)
as a slide holding that page drawn at the export resolution, so a deck is
rebuilt from OpenFlo instead of by swapping pictures into slides by hand.
The slide is the page's size, so the picture fills it. Each slide's title
(the sample and page) is its name in PowerPoint's outline and its alt text.
The document properties (File > Info) name the OpenFlo version that wrote it.
"""
from __future__ import annotations

import zipfile
from collections.abc import Iterable
from xml.sax.saxutils import escape, quoteattr

from .export_stamp import export_metadata, stamp_text

EMU_PER_INCH = 914400

_NS_P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
_NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
_NS_R = ('http://schemas.openxmlformats.org/officeDocument/2006/'
         'relationships')
_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
_PKG_REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
_CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'


def _rels(items):
    body = ''.join(f'<Relationship Id="{rid}" Type="{_REL}/{kind}" '
                   f'Target="{target}"/>' for rid, kind, target in items)
    return f'{_DECL}<Relationships xmlns="{_PKG_REL}">{body}</Relationships>'


def _content_types(n):
    over = [('/ppt/presentation.xml', 'presentationml.presentation.main'),
            ('/ppt/slideMasters/slideMaster1.xml',
             'presentationml.slideMaster'),
            ('/ppt/slideLayouts/slideLayout1.xml',
             'presentationml.slideLayout'),
            ('/ppt/theme/theme1.xml', 'theme'),
            ('/docProps/core.xml', None),
            ('/docProps/app.xml', None)]
    over += [(f'/ppt/slides/slide{i}.xml', 'presentationml.slide')
             for i in range(1, n + 1)]

    def ctype(part, kind):
        if part == '/docProps/core.xml':
            return 'application/vnd.openxmlformats-package.core-properties+xml'
        if part == '/docProps/app.xml':
            return ('application/vnd.openxmlformats-officedocument.'
                    'extended-properties+xml')
        return f'application/vnd.openxmlformats-officedocument.{kind}+xml'
    body = ('<Default Extension="rels" ContentType="application/'
            'vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Default Extension="png" ContentType="image/png"/>'
            + ''.join(f'<Override PartName="{p}" ContentType="{ctype(p, k)}"/>'
                      for p, k in over))
    return f'{_DECL}<Types xmlns="{_CT}">{body}</Types>'


def _presentation(n, cx, cy):
    ids = ''.join(f'<p:sldId id="{256 + i}" r:id="rId{i + 2}"/>'
                  for i in range(n))
    return (f'{_DECL}<p:presentation xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" '
            f'xmlns:p="{_NS_P}" saveSubsetFonts="1">'
            '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/>'
            f'</p:sldMasterIdLst><p:sldIdLst>{ids}</p:sldIdLst>'
            f'<p:sldSz cx="{cx}" cy="{cy}"/><p:notesSz cx="{cy}" cy="{cx}"/>'
            '</p:presentation>')


_EMPTY_TREE = ('<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
               '<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>'
               '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" '
               'y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
               '</p:spTree></p:cSld>')

_MASTER = (f'{_DECL}<p:sldMaster xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" '
           f'xmlns:p="{_NS_P}">{_EMPTY_TREE}'
           '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" '
           'accent1="accent1" accent2="accent2" accent3="accent3" '
           'accent4="accent4" accent5="accent5" accent6="accent6" '
           'hlink="hlink" folHlink="folHlink"/>'
           '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/>'
           '</p:sldLayoutIdLst></p:sldMaster>')

_LAYOUT = (f'{_DECL}<p:sldLayout xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" '
           f'xmlns:p="{_NS_P}" type="blank" preserve="1">'
           + _EMPTY_TREE.replace('<p:cSld>', '<p:cSld name="Blank">')
           + '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>'
           '</p:sldLayout>')


def _theme():
    colours = (('dk1', '000000'), ('lt1', 'FFFFFF'), ('dk2', '1F2937'),
               ('lt2', 'E5E7EB'), ('accent1', '1F77B4'),
               ('accent2', 'FF7F0E'), ('accent3', '2CA02C'),
               ('accent4', 'D62728'), ('accent5', '9467BD'),
               ('accent6', '8C564B'), ('hlink', '1F77B4'),
               ('folHlink', '9467BD'))
    clr = ''.join(f'<a:{k}><a:srgbClr val="{v}"/></a:{k}>' for k, v in colours)
    font = ('<a:latin typeface="Calibri"/><a:ea typeface=""/>'
            '<a:cs typeface=""/>')
    fill = '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    line = f'<a:ln w="9525">{fill}</a:ln>'
    effect = '<a:effectStyle><a:effectLst/></a:effectStyle>'
    return (f'{_DECL}<a:theme xmlns:a="{_NS_A}" name="OpenFlo">'
            f'<a:themeElements><a:clrScheme name="OpenFlo">{clr}'
            '</a:clrScheme><a:fontScheme name="OpenFlo">'
            f'<a:majorFont>{font}</a:majorFont><a:minorFont>{font}'
            '</a:minorFont></a:fontScheme><a:fmtScheme name="OpenFlo">'
            f'<a:fillStyleLst>{fill * 3}</a:fillStyleLst>'
            f'<a:lnStyleLst>{line * 3}</a:lnStyleLst>'
            f'<a:effectStyleLst>{effect * 3}</a:effectStyleLst>'
            f'<a:bgFillStyleLst>{fill * 3}</a:bgFillStyleLst>'
            '</a:fmtScheme></a:themeElements></a:theme>')


def _slide(title, cx, cy):
    t = quoteattr(str(title))
    return (f'{_DECL}<p:sld xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" '
            f'xmlns:p="{_NS_P}"><p:cSld name={t}><p:spTree><p:nvGrpSpPr>'
            '<p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>'
            '</p:grpSpPr><p:pic><p:nvPicPr>'
            f'<p:cNvPr id="2" name="Picture 1" descr={t}/>'
            '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>'
            '<p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/>'
            '<a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr>'
            f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/>'
            '</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '</p:spPr></p:pic></p:spTree></p:cSld>'
            '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


def _core(corrections=None):
    """docProps/core.xml: title, creator and the export stamp (keywords,
    description, and the OpenFlo version as the document version)."""
    meta = export_metadata(corrections)
    stamp = escape(stamp_text(corrections))
    return (f'{_DECL}<cp:coreProperties xmlns:cp="http://schemas.'
            'openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/">'
            '<dc:title>OpenFlo layout</dc:title><dc:creator>OpenFlo'
            f'</dc:creator><cp:keywords>{stamp}</cp:keywords>'
            f'<dc:description>{stamp}</dc:description>'
            f'<cp:version>{escape(meta["OpenFlo version"])}</cp:version>'
            '</cp:coreProperties>')


def write_pptx(path: str, slides: Iterable[tuple[bytes, str]],
               size_in: tuple[float, float], corrections=None) -> int:
    """Write ``slides`` (``(PNG bytes, title)`` each, in order) to ``path``
    as a deck whose slides are ``size_in`` (width, height) inches, each
    picture filling its slide; ``corrections`` go in the document
    properties beside the version. Returns the number of slides. Raises
    ``ValueError`` for no slides or a picture that is not a PNG."""
    slides = list(slides)
    if not slides:
        raise ValueError("no slides to write")
    for png, _t in slides:
        if not bytes(png[:8]) == b'\x89PNG\r\n\x1a\n':
            raise ValueError("every slide picture must be a PNG")
    cx = int(round(size_in[0] * EMU_PER_INCH))
    cy = int(round(size_in[1] * EMU_PER_INCH))
    n = len(slides)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', _content_types(n))
        z.writestr('_rels/.rels', _rels([
            ('rId1', 'officeDocument', 'ppt/presentation.xml'),
            ('rId2', 'metadata/core-properties', 'docProps/core.xml'),
            ('rId3', 'extended-properties', 'docProps/app.xml')]).replace(
                f'{_REL}/metadata/core-properties',
                'http://schemas.openxmlformats.org/package/2006/'
                'relationships/metadata/core-properties'))
        z.writestr('docProps/core.xml', _core(corrections))
        z.writestr('docProps/app.xml', (
            f'{_DECL}<Properties xmlns="http://schemas.openxmlformats.org/'
            'officeDocument/2006/extended-properties">'
            f'<Application>OpenFlo</Application><Slides>{n}</Slides>'
            '</Properties>'))
        z.writestr('ppt/presentation.xml', _presentation(n, cx, cy))
        z.writestr('ppt/_rels/presentation.xml.rels', _rels(
            [('rId1', 'slideMaster', 'slideMasters/slideMaster1.xml')]
            + [(f'rId{i + 2}', 'slide', f'slides/slide{i + 1}.xml')
               for i in range(n)]
            + [(f'rId{n + 2}', 'theme', 'theme/theme1.xml')]))
        z.writestr('ppt/slideMasters/slideMaster1.xml', _MASTER)
        z.writestr('ppt/slideMasters/_rels/slideMaster1.xml.rels', _rels([
            ('rId1', 'slideLayout', '../slideLayouts/slideLayout1.xml'),
            ('rId2', 'theme', '../theme/theme1.xml')]))
        z.writestr('ppt/slideLayouts/slideLayout1.xml', _LAYOUT)
        z.writestr('ppt/slideLayouts/_rels/slideLayout1.xml.rels', _rels([
            ('rId1', 'slideMaster', '../slideMasters/slideMaster1.xml')]))
        z.writestr('ppt/theme/theme1.xml', _theme())
        for i, (png, title) in enumerate(slides, 1):
            z.writestr(f'ppt/slides/slide{i}.xml', _slide(title, cx, cy))
            z.writestr(f'ppt/slides/_rels/slide{i}.xml.rels', _rels([
                ('rId1', 'slideLayout', '../slideLayouts/slideLayout1.xml'),
                ('rId2', 'image', f'../media/image{i}.png')]))
            z.writestr(f'ppt/media/image{i}.png', bytes(png))
    return n


def slide_title(sample: str, page: str) -> str:
    """``'sample: page'``, the slide's name and alt text."""
    return f'{sample}: {page}' if page else str(sample)
