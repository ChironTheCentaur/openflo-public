"""The OpenFlo version (and the corrections applied) in an exported file's own
metadata. Tk-free.

A figure pasted into a paper or a table sent to a collaborator must say which
OpenFlo made it, so that once a bug is fixed anyone can tell whether a file
predates the fix. The stamp goes where each format keeps facts about itself,
never over the data: PNG text chunks, the PDF Info dictionary, SVG and PPTX
document properties, FCS TEXT keywords, workbook properties, an XML comment
in a .wsp or Gating-ML file.

``corrections`` is a list of fix ids or labels: the fixes applied to the
analysis that made the file. The editor holds them in ``_applied_fixes``
(see `corrections_for`); until a fix registry fills it, it is None and the
stamp says the version alone.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

# The FCS keywords, the names batch_normalize has written since its first
# normalized copies, used by every FCS export.
FCS_VERSION_KEY = 'OPENFLO_VERSION'
FCS_CORRECTIONS_KEY = 'OPENFLO_CORRECTIONS'


def version() -> str:
    """The OpenFlo version exports are stamped with: the one a saved session
    records (:func:`openflo.result_stamp.app_version`, cached, since it can
    start git and a batch export stamps many files)."""
    from .result_stamp import app_version
    return app_version()


def corrections_for(owner: Any) -> list[str] | None:
    """The corrections ``owner`` (an editor) says were applied, from its
    ``_applied_fixes``; None when it has none. The one place the stamp
    learns of them, so the fix registry has a single attribute to fill."""
    fixes = getattr(owner, '_applied_fixes', None)
    if not fixes:
        return None
    if isinstance(fixes, str):
        fixes = [fixes]
    return [str(f) for f in fixes if str(f).strip()] or None


def _corrections_text(corrections: Iterable[Any] | None) -> str:
    if corrections is None or isinstance(corrections, str):
        items = [corrections] if corrections else []
    else:
        items = list(corrections)
    return '; '.join(str(c).strip() for c in items if str(c).strip())


def export_metadata(corrections: Iterable[Any] | None = None) -> dict[str, str]:
    """``{'Software': 'OpenFlo <v>', 'OpenFlo version': <v>}``, plus
    ``'Corrections': 'id; id'`` when any were applied."""
    v = version()
    meta = {'Software': f'OpenFlo {v}', 'OpenFlo version': v}
    fixes = _corrections_text(corrections)
    if fixes:
        meta['Corrections'] = fixes
    return meta


def stamp_text(corrections: Iterable[Any] | None = None) -> str:
    """The stamp on one line: ``OpenFlo <v>`` or
    ``OpenFlo <v>; corrections: a; b``."""
    meta = export_metadata(corrections)
    text = meta['Software']
    if 'Corrections' in meta:
        text += f"; corrections: {meta['Corrections']}"
    return text


def figure_format(path: Any, fmt: str | None = None) -> str:
    """The format matplotlib will save ``path`` in: ``fmt``, else the file
    extension, else the ``savefig.format`` setting (a file object)."""
    if fmt:
        return str(fmt).lower()
    if isinstance(path, (str, os.PathLike)):
        ext = os.path.splitext(os.fspath(path))[1].lstrip('.').lower()
        if ext:
            return ext
    import matplotlib
    return str(matplotlib.rcParams['savefig.format']).lower()


# The background every figure export starts with. PNG, SVG, PDF, TIFF and
# WebP keep it see-through (EPS too, though PostScript has no partial
# transparency, so a Translucent EPS comes out opaque); a format in
# SOLID_FORMATS cannot, so its export is written on a solid colour instead.
DEFAULT_EXPORT_BACKGROUND = 'Transparent'
# The Background choices every export offers, the default first.
EXPORT_BACKGROUNDS = ('Transparent', 'White', 'Dark', 'Translucent')
SOLID_FORMATS = frozenset({'jpg', 'jpeg'})


def keeps_transparency(path: Any = None, fmt: str | None = None) -> bool:
    """Whether a figure saved to ``path`` / ``fmt`` can have a transparent
    background (False for JPEG)."""
    return figure_format(path, fmt) not in SOLID_FORMATS


def solid_face(fig: Any) -> tuple[float, float, float]:
    """The opaque colour behind ``fig`` when its format cannot be
    transparent: the figure's own face, or white when that face is itself
    see-through."""
    from matplotlib.colors import to_rgba
    r, g, b, a = to_rgba(fig.get_facecolor())
    return (r, g, b) if a > 0 else (1.0, 1.0, 1.0)


def background_kwargs(fig: Any, path: Any = None,
                      fmt: str | None = None) -> dict[str, Any]:
    """The ``savefig`` arguments for the default export background:
    transparent where the format allows it, else the figure's solid face
    (``solid_face``). For exports that call ``savefig`` directly; the
    windows go through ``theme.savefig_background``."""
    if keeps_transparency(path, fmt):
        return {'transparent': True}
    return {'facecolor': solid_face(fig)}


def savefig_metadata(path: Any = None, fmt: str | None = None,
                     corrections: Iterable[Any] | None = None
                     ) -> dict[str, str] | None:
    """The ``metadata`` argument for ``savefig`` (or ``PdfPages``) in the
    format of ``path`` / ``fmt``; None for a format that keeps none (JPEG,
    TIFF), which matplotlib would refuse.

    PNG: one text chunk per key (``Software`` is the standard one). PDF and
    PostScript: Creator, with the stamp in Keywords. SVG: the Dublin Core
    creator and description."""
    fmt = figure_format(path, fmt)
    meta = export_metadata(corrections)
    if fmt == 'png':
        return dict(meta)
    if fmt == 'pdf':
        return {'Creator': meta['Software'], 'Keywords': stamp_text(corrections)}
    if fmt in ('ps', 'eps'):
        return {'Creator': stamp_text(corrections)}
    if fmt in ('svg', 'svgz'):
        return {'Creator': meta['Software'],
                'Description': stamp_text(corrections)}
    return None


def fcs_keywords(corrections: Iterable[Any] | None = None) -> dict[str, str]:
    """TEXT keywords for an exported FCS file."""
    meta = export_metadata(corrections)
    out = {FCS_VERSION_KEY: meta['OpenFlo version']}
    if 'Corrections' in meta:
        out[FCS_CORRECTIONS_KEY] = meta['Corrections']
    return out


def xml_comment(corrections: Iterable[Any] | None = None) -> str:
    """``<!-- Written by OpenFlo <v> -->`` for an XML export. A comment, so
    FlowJo and Gating-ML readers skip it."""
    # '--' cannot appear inside an XML comment.
    text = stamp_text(corrections).replace('--', '- -')
    return f'<!-- Written by {text} -->'


def set_workbook_properties(wb, corrections: Iterable[Any] | None = None) -> None:
    """Stamp an openpyxl workbook's document properties (File > Info in
    Excel): creator and keywords, and the version and corrections as custom
    properties."""
    meta = export_metadata(corrections)
    wb.properties.creator = meta['Software']
    wb.properties.keywords = stamp_text(corrections)
    try:
        from openpyxl.packaging.custom import StringProperty
        for key in ('OpenFlo version', 'Corrections'):
            if key in meta:
                wb.custom_doc_props.append(
                    StringProperty(name=key, value=meta[key]))
    except Exception:  # noqa: BLE001 - an older openpyxl has no custom properties; the core ones stay
        pass
