"""Headless smoke test for the FCS metadata inspector dialog."""

import importlib
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import tkinter as tk  # noqa: E402

import pytest  # noqa: E402


def _make_editor():
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:
        pytest.skip(str(e))
    gui = importlib.import_module("openflo.gui")
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(
        root, fcs_dir=None, labels_str="", on_save=None, primary=False
    )
    ed.withdraw()
    return root, ed


def test_dialog_constructs_without_file():
    root, ed = _make_editor()
    try:
        from openflo.ui_inspect import FcsInspectorDialog

        d = FcsInspectorDialog(ed)
        assert d.summary_var.get() == "No file loaded."
        d.destroy()
    finally:
        ed.destroy()
        root.destroy()


def test_metadata_helper_reads_synthetic(tmp_path):
    from openflo import synthetic
    from openflo.ui_inspect import read_fcs_metadata

    paths = synthetic.make_immunophenotyping_dataset(
        str(tmp_path), n=300, seed=1
    )
    assert paths, "synthetic dataset produced no files"

    meta = read_fcs_metadata(paths[0])
    assert meta["channel_count"] > 0
    assert meta["channels"], "no channels parsed"
    assert meta["text"], "no TEXT keywords parsed"
    # event_count should reflect the n we asked for (or at least be a positive
    # int when the parser reports it).
    if isinstance(meta["event_count"], int):
        assert meta["event_count"] > 0
    # First channel carries a $PnN detector name.
    assert meta["channels"][0]["pnn"]


def test_dialog_loads_synthetic(tmp_path):
    root, ed = _make_editor()
    try:
        from openflo import synthetic
        from openflo.ui_inspect import FcsInspectorDialog

        paths = synthetic.make_immunophenotyping_dataset(
            str(tmp_path), n=300, seed=2
        )
        d = FcsInspectorDialog(ed)
        d._load(paths[0])
        assert d.chan_tv.get_children(), "channel table empty after load"
        assert d.kw_tv.get_children(), "keyword table empty after load"
        d.destroy()
    finally:
        ed.destroy()
        root.destroy()
