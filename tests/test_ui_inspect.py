"""Headless smoke test for the FCS metadata inspector dialog."""

from tests.conftest import editor_or_skip


def test_dialog_constructs_without_file():
    root, ed = editor_or_skip()
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
    # The file was written with n=300 events over these 11 detectors.
    assert meta["event_count"] == 300
    assert meta["channel_count"] == 11
    assert [c["index"] for c in meta["channels"]] == list(range(1, 12))
    assert [c["pnn"] for c in meta["channels"]] == [
        "FSC-A", "FSC-H", "SSC-A", "LiveDead-A", "FL6-A", "FL4-A", "FL2-A",
        "FL5-A", "FL7-A", "FL8-A", "Time"]
    assert meta["text"]["par"] == "11" and meta["text"]["tot"] == "300"


def test_dialog_loads_synthetic(tmp_path):
    root, ed = editor_or_skip()
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


def test_metadata_helper_reads_voltage_and_gain(tmp_path):
    """The channel table's voltage and gain columns come from each
    channel's $PnV / $PnG keyword. flowio hands TEXT keys back lowercase
    without the '$' (``p1v``), so a lookup that only tried ``$P1V`` / ``P1V``
    left both columns blank for every file."""
    import numpy as np
    import pandas as pd

    from openflo.fcs_export import write_fcs
    from openflo.ui_inspect import read_fcs_metadata

    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.uniform(0, 1000, size=(50, 3)),
                      columns=["FSC-A", "FL1-A", "FL2-A"])
    path = tmp_path / "volts.fcs"
    # FL2-A is written with no voltage, so its voltage cell stays blank.
    # (flowio writes a $PnG for every channel, so each gets an explicit one.)
    write_fcs(df, path, voltages={"FSC-A": 310, "FL1-A": 455},
              keywords={"P1G": "1.5", "P2G": "4", "P3G": "2"})

    chans = read_fcs_metadata(str(path))["channels"]
    assert [c["pnn"] for c in chans] == ["FSC-A", "FL1-A", "FL2-A"]
    assert [c["pnv"] for c in chans] == ["310", "455", ""]
    assert [c["gain"] for c in chans] == ["1.5", "4", "2"]
