"""Instrument files OpenFlo cannot read, and how to get FCS out of each.

Some instruments save their data in their own undocumented formats, which no
open reader can read. When the user opens or drops one, OpenFlo names the
program and says how to export FCS from it, instead of a parse error or
silently skipping it. ``VENDOR_FORMATS`` is the one table; the status line,
the dialog, the FCS reader and the command line all word their message from
it (``vendor_message``, ``vendor_summary``).

A file is recognised by its extension only, never by its content: a real FCS
file is read as FCS whatever it is called, and a file with one of these
extensions is never parsed.

Sources (checked 2026-11):

- MACSQuant ``.mqd``: MACSQuantify v3 writes MQD or FCS 3.1; export type FCS
  3.1 under Edit > Software > Export > FCS (De Novo Software FAQ "MACSQuant
  3.0.1", which warns off the "FCS 3.1 compatible" and "Custom" types;
  FlowJo docs "MACSQuant").
- NovoExpress ``.ncf``: the NovoExpress experiment file; Sample (tab) >
  Import and Export (group) > Export FCS File (Agilent NovoExpress Software
  Guide).
- Sony ``.sraw``: spectral raw data, always written with an ``.fcs`` file of
  the same name (FCS Express manual, "Spectral data file analysis").
- Attune CytPix ``.acs``: the image data of the Attune Cytometric Software;
  right-click the experiment, Export FCS Files (Thermo Fisher Attune NxT
  data management guide; FCS Express FAQ on the Attune CytPix, whose
  separate "FCS and Image Data" export is what writes ``.acs``).
  Chemometec writes ``.acs`` too, so that message says "looks like".

Core module: standard library only. No Tk.
"""
from __future__ import annotations

import os
from typing import NamedTuple


class VendorFormat(NamedTuple):
    """One instrument format: what the file is (``"a MACSQuant file"``) and
    the sentence that says how to get FCS from it. ``shared`` marks an
    extension other programs use too: the message then says the file only
    looks like this kind, and how to get FCS if it came from elsewhere."""
    what: str
    export: str
    shared: bool = False


# The sentence a ``shared`` extension's message ends with.
OTHER_PROGRAM = ("If it came from another instrument, export FCS from that "
                 "program.")


# Extension (lower case, with the dot) -> what it is and how to export FCS.
VENDOR_FORMATS: dict[str, VendorFormat] = {
    '.mqd': VendorFormat(
        'a MACSQuant file',
        "In MACSQuantify, export the data as FCS 3.1 (not 'FCS 3.1 "
        "compatible'), then open the FCS file."),
    '.ncf': VendorFormat(
        'a NovoExpress file',
        "In NovoExpress, choose Sample > Import and Export > Export FCS "
        "File, then open the FCS file."),
    '.sraw': VendorFormat(
        'a Sony spectral raw file',
        "Open the .fcs file of the same name that the Sony software saved "
        "next to it, or export FCS from the Sony software."),
    # Chemometec image cytometers also write .acs (FCS Express lists both).
    '.acs': VendorFormat(
        'an Attune CytPix file',
        "In the Attune software, right-click the experiment and choose "
        "Export FCS Files, then open those files.",
        shared=True),
}
# The table's extensions as a Tk file-dialog pattern, so a picker shows
# these files and the user gets the message instead of an empty folder.
VENDOR_FILE_PATTERNS = ' '.join(f'*{ext}' for ext in VENDOR_FORMATS)

# Not listed: Attune .tmo. FCS Express lists it as an Attune format, but no
# public page found says what it holds or how to export FCS from it.


def vendor_format(path) -> VendorFormat | None:
    """The ``VENDOR_FORMATS`` entry for `path`'s extension (any case), or
    None for any other file."""
    ext = os.path.splitext(str(path))[1].lower()
    return VENDOR_FORMATS.get(ext)


def vendor_message(path) -> str | None:
    """The message for one instrument file `path`, or None when it is not
    one: "This is a MACSQuant file. OpenFlo can't read it directly. In
    MACSQuantify, export the data as FCS 3.1 (not 'FCS 3.1 compatible'),
    then open the FCS file." """
    fmt = vendor_format(path)
    if fmt is None:
        return None
    if fmt.shared:
        ext = os.path.splitext(str(path))[1].lower()
        lead = f"This looks like {fmt.what} ({ext})."
    else:
        lead = f"This is {fmt.what}."
    return f"{lead} OpenFlo can't read it directly. {_how_to_export(fmt)}"


def _how_to_export(fmt: VendorFormat) -> str:
    """`fmt`'s export sentence, and for a ``shared`` extension the one for
    a file from another program."""
    return f"{fmt.export} {OTHER_PROGRAM}" if fmt.shared else fmt.export


def vendor_files(paths) -> list[str]:
    """The paths among `paths` that are instrument files, in order."""
    return [p for p in paths if vendor_format(p) is not None]


def vendor_summary(paths, cap: int = 8) -> str:
    """One message for the instrument files among `paths` ('' when there
    are none): the single-file message for one file; for several, one
    paragraph per format naming its files (at most `cap` names each)."""
    by_ext: dict[str, list[str]] = {}
    for p in vendor_files(paths):
        ext = os.path.splitext(str(p))[1].lower()
        by_ext.setdefault(ext, []).append(os.path.basename(str(p)))
    if not by_ext:
        return ''
    if sum(len(v) for v in by_ext.values()) == 1:
        (ext, (name,)), = by_ext.items()
        return f"{name}: {vendor_message(name)}"
    parts = []
    for ext, names in by_ext.items():
        fmt = VENDOR_FORMATS[ext]
        shown = ', '.join(names[:cap])
        if len(names) > cap:
            shown += f" and {len(names) - cap} more"
        kind = fmt.what.split(' ', 1)[1]
        parts.append(f"{len(names)} {kind}(s) not opened: {shown}. "
                     f"OpenFlo can't read them directly. "
                     f"{_how_to_export(fmt)}")
    return "\n\n".join(parts)


def vendor_status(paths) -> str:
    """A one-line status for the instrument files among `paths` ('' when
    none): the single-file message, or a count with a pointer to the
    dialog that lists them."""
    found = vendor_files(paths)
    if not found:
        return ''
    if len(found) == 1:
        return vendor_summary(found)
    return (f"Not opened: {len(found)} instrument file(s) OpenFlo can't "
            "read directly. Export them as FCS first.")
