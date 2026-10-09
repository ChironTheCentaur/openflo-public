"""A transform record is only ever trusted for the CSV it was written with.

The writers wrote the CSV first and the record second, so when the record
write failed, the PREVIOUS record stayed beside the new CSV and was trusted:
measured, a channel re-saved on asinh was inverted with the old logicle
record (median 123.5 where 19,922 is right). The old record is now removed
before the CSV is written, the new one is written atomically, and a failed
write leaves no ``.json.tmp`` behind.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.test_editor_tools import _fcs, _loaded
from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
    settle,
)


def _fail_record_writes(monkeypatch):
    """Make every transform-record write fail mid-file (the disk filled up,
    say); every other json.dump still works."""
    real = json.dump

    def dump(obj, fp, *a, **k):
        if str(getattr(fp, 'name', '')).endswith('_transforms.json.tmp'):
            fp.write('{"format": "openflo-data-')        # a partial file
            raise OSError('No space left on device')
        return real(obj, fp, *a, **k)
    monkeypatch.setattr(json, 'dump', dump)
    return real


def _leftovers(folder):
    return sorted(p for p in os.listdir(folder)
                  if p.endswith(('_transforms.json', '.json.tmp')))


def test_a_record_for_another_row_count_is_not_trusted(tmp_path, monkeypatch):
    """The column list alone does not bind a record to its CSV: another
    sample's export with the same panel matches it. Replaced by such a CSV
    (another row count), the old record -- asinh -- must be ignored, and the
    load must say so."""
    import pandas as pd

    from openflo import editor_load
    isolate_home(monkeypatch, tmp_path)
    shown = capture_warnings(monkeypatch)
    a = _loaded(_fcs(tmp_path / 'a.fcs', 1))
    a.retransform(['FL1-A'], method='asinh')
    out = tmp_path / 'panel.csv'
    a.export_csv(str(out))                       # record: FL1-A asinh
    b = _loaded(_fcs(tmp_path / 'b.fcs', 2))     # logicle, same columns
    keep = b.data.iloc[: len(b.data) - 7]        # and another row count
    assert list(keep.columns) == list(pd.read_csv(out).columns)
    keep.to_csv(out, index=False)                # replaced outside OpenFlo
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    try:
        ed = _new_editor(root)
        monkeypatch.setattr(editor_load.filedialog, 'askopenfilenames',
                            lambda **k: (str(out),))
        ed._load_processed_data()
        settle(root)
        s = ed._samples['panel']
        assert s.data_transforms['FL1-A']['method'] == 'logicle'   # not asinh
        (title, msg), = shown
        assert 'ignored' in msg and 'rows' in msg, msg
    finally:
        root.destroy()


def test_export_csv_never_leaves_a_stale_record(tmp_path, monkeypatch):
    from openflo.pipeline import read_transforms_sidecar
    isolate_home(monkeypatch, tmp_path)
    s = _loaded(_fcs(tmp_path / 'x.fcs', 3))
    out = tmp_path / 'x_processed.csv'
    s.export_csv(str(out))                               # record: logicle
    assert _leftovers(tmp_path) == ['x_processed_transforms.json']
    s.retransform(['FL1-A'], method='asinh')
    _fail_record_writes(monkeypatch)
    with pytest.raises(OSError):
        s.export_csv(str(out))                           # CSV now asinh
    assert _leftovers(tmp_path) == []                    # no stale, no .tmp
    import pandas as pd
    df = pd.read_csv(out)
    assert read_transforms_sidecar(str(out), df.columns, len(df)) is None


def test_a_failed_session_record_write_is_not_trusted_on_restore(
        tmp_path, monkeypatch):
    """Saved on logicle, then re-saved on asinh with the record write
    failing: the restore must not invert the asinh values with the old
    logicle record."""
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    try:
        ed = _new_editor(root)
        ed._on_loaded('s1', _loaded(_fcs(tmp_path / 's1.fcs', 1)))
        path = tmp_path / 'sess.flowsession'
        ed._write_session(str(path))
        data_dir = tmp_path / 'sess_data'
        assert _leftovers(data_dir) == ['s1_transforms.json']
        ed._apply_channel_transforms({'FL1-A': 'asinh'})
        asinh = ed._samples['s1'].data['FL1-A'].to_numpy(float).copy()
        real_dump = _fail_record_writes(monkeypatch)
        ed._write_session(str(path))
        assert ed._sidecar_failures == ['s1']             # reported, as before
        assert _leftovers(data_dir) == []                 # no stale, no .tmp
        monkeypatch.setattr(json, 'dump', real_dump)
        ed2 = _new_editor(root)
        ed2._queue_processed_loads = lambda items, front_names=(): [
            ed2._load_csv_worker(nm, p, {}) for nm, p in items]
        ed2._load_session_path(str(path))
        settle(root)
        s = ed2._samples['s1']
        # Scale unknown, values as saved (to the CSV text round trip's last
        # bit): never inverted with the old logicle record.
        assert s.data_transforms['FL1-A']['method'] == 'unknown'
        np.testing.assert_allclose(s.data['FL1-A'].to_numpy(float), asinh,
                                   rtol=1e-12)
    finally:
        root.destroy()
