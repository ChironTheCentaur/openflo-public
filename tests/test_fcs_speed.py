"""FCS writing and reading got faster without changing a byte or a value.

* Export (speed audit item 8): the event block is converted in numpy instead
  of through a Python list, and the file must be byte-for-byte what the list
  path wrote.
* Integer files with unused high bits (complaints item 29): the bits are
  masked with numpy instead of FlowIO's byte-at-a-time loop, and every value
  must equal plain ``flowio.FlowData``'s.

Each has a negative partner showing the comparison can fail.
"""
import time

import flowio
import numpy as np
import pandas as pd
import pytest

from openflo import fcs_read
from openflo.fcs_export import fcs_event_block, write_fcs

# ── export ────────────────────────────────────────────────────────────────────


def _frame(seed=0, n=5000):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 5)) * 1000,
                      columns=[f'FL{i}-A' for i in range(1, 6)])
    # Values that stress the double -> float32 rounding: halfway cases,
    # subnormals, beyond float32's range, and non-finite cells.
    df.iloc[0, 0] = 1e300
    df.iloc[1, 1] = -1e300
    df.iloc[2, 2] = np.nan
    df.iloc[3, 3] = 1e-300
    df.iloc[4, 4] = 3.4028235677973366e38
    df.iloc[5, 0] = 1.0 + 2.0 ** -24          # exactly halfway: ties to even
    df.iloc[6, 1] = np.inf
    return df


def _old_write(df, path, labels=None):
    """The export as it was: ``mat.flatten().tolist()`` into create_fcs."""
    channels = [str(c) for c in df.columns]
    mat = np.nan_to_num(df.to_numpy(dtype=float), nan=0.0, posinf=0.0,
                        neginf=0.0)
    opt = [labels.get(c, '') for c in channels] if labels else None
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, mat.flatten().tolist(), channels,
                          opt_channel_names=opt)


def test_export_is_byte_identical_to_the_list_path(tmp_path):
    df = _frame()
    labels = {'FL1-A': 'Marker 1'}
    write_fcs(df, str(tmp_path / 'new.fcs'), channel_labels=labels)
    _old_write(df, str(tmp_path / 'old.fcs'), labels)
    assert (tmp_path / 'new.fcs').read_bytes() == (tmp_path / 'old.fcs').read_bytes()


def test_pipeline_and_synthetic_writers_are_byte_identical(tmp_path):
    from openflo import synthetic
    from openflo.pipeline import write_fcs as pipeline_write
    df = _frame(1)
    pipeline_write(str(tmp_path / 'p.fcs'), df)
    synthetic._write_fcs(str(tmp_path / 's.fcs'), df)
    _old_write(df, str(tmp_path / 'old.fcs'))
    old = (tmp_path / 'old.fcs').read_bytes()
    assert (tmp_path / 'p.fcs').read_bytes() == old
    assert (tmp_path / 's.fcs').read_bytes() == old


def test_one_float32_step_changes_the_file(tmp_path):
    """Negative partner: a value one float32 step away gives a different
    file, so the byte comparison above would see any rounding change."""
    df = _frame()
    df2 = df.copy()
    v = np.float32(df2.iloc[10, 2])
    df2.iloc[10, 2] = float(np.nextafter(v, np.float32(np.inf)))
    write_fcs(df, str(tmp_path / 'a.fcs'))
    write_fcs(df2, str(tmp_path / 'b.fcs'))
    assert (tmp_path / 'a.fcs').read_bytes() != (tmp_path / 'b.fcs').read_bytes()


def test_event_block_does_not_build_python_floats():
    """The block is made without a Python list: a float64 matrix converts to
    a float32 array of the same length, in row order."""
    mat = np.arange(12, dtype=float).reshape(3, 4)
    block = fcs_event_block(mat)
    assert block.typecode == 'f' and len(block) == 12
    assert list(block) == [float(x) for x in range(12)]
    fortran = np.asfortranarray(mat)
    assert list(fcs_event_block(fortran)) == list(block)


# ── integer reading ───────────────────────────────────────────────────────────


def write_int_fcs(path, data, pnr, width=16, big_endian=False):
    """A minimal integer list-mode FCS 3.1 file (FlowIO only writes floats)."""
    n, p = data.shape
    text = {'$BYTEORD': '4,3,2,1' if big_endian else '1,2,3,4',
            '$DATATYPE': 'I', '$MODE': 'L', '$NEXTDATA': '0',
            '$PAR': str(p), '$TOT': str(n), '$BEGINANALYSIS': '0',
            '$ENDANALYSIS': '0', '$BEGINSTEXT': '0', '$ENDSTEXT': '0'}
    for i in range(p):
        text[f'$P{i + 1}N'] = f'FL{i + 1}-A'
        text[f'$P{i + 1}B'] = str(width)
        text[f'$P{i + 1}R'] = str(pnr[i])
        text[f'$P{i + 1}E'] = '0,0'
    dt = ('>' if big_endian else '<') + {8: 'u1', 16: 'u2', 32: 'u4'}[width]
    body = data.astype(dt).tobytes()

    def build(begin):
        t = dict(text, **{'$BEGINDATA': str(begin),
                          '$ENDDATA': str(begin + len(body) - 1)})
        return ('/' + ''.join(f'{k}/{v}/' for k, v in t.items())).encode()

    begin = 58
    for _ in range(4):           # the offsets' own digits move the data start
        begin = 58 + len(build(begin))
    txt = build(begin)
    header = (b'FCS3.1    ' + f'{58:>8}{57 + len(txt):>8}{begin:>8}'
              f'{begin + len(body) - 1:>8}{0:>8}{0:>8}'.encode())
    with open(path, 'wb') as fh:
        fh.write(header + txt + body)


CASES = [
    # width, $PnR per channel, big-endian
    (16, [1024, 1024, 1024, 1024], False),          # 10-bit data in 16 bits
    (16, [1000, 65536, 262144 // 64, 3], True),     # mixed; 1000 rounds up to 1024
    (8, [16, 256, 100], False),
    (32, [2 ** 20, 2 ** 32, 2 ** 18], True),
]


@pytest.mark.parametrize('width, pnr, big', CASES)
def test_masked_integer_values_equal_flowio(tmp_path, width, pnr, big):
    rng = np.random.default_rng(width)
    data = rng.integers(0, 2 ** width, size=(3000, len(pnr)), dtype=np.uint64)
    path = str(tmp_path / 'int.fcs')
    write_int_fcs(path, data, pnr, width, big)
    ours = fcs_read.FlowData(path)
    theirs = flowio.FlowData(path)
    assert ours.events.typecode == theirs.events.typecode
    assert ours.events == theirs.events
    got = np.asarray(ours.events).reshape(-1, len(pnr))
    limits = [1 << (int(r) - 1).bit_length() for r in pnr]
    assert (got < np.array(limits, dtype=np.uint64)).all()


def test_the_test_files_really_carry_high_bits(tmp_path):
    """Negative partner: the stored words have bits above $PnR set, so a
    reader that skipped the mask would return different values."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2 ** 16, size=(3000, 4), dtype=np.uint64)
    assert (data >= 1024).any()
    path = str(tmp_path / 'int.fcs')
    write_int_fcs(path, data, [1024] * 4)
    masked = np.asarray(fcs_read.FlowData(path).events).reshape(-1, 4)
    assert not np.array_equal(masked, data)
    np.testing.assert_array_equal(masked, data & 1023)


def test_unmasked_and_float_files_are_left_to_flowio(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.integers(0, 2 ** 16, size=(500, 3), dtype=np.uint64)
    path = str(tmp_path / 'full.fcs')
    write_int_fcs(path, data, [65536] * 3)
    assert fcs_read.FlowData(path).events == flowio.FlowData(path).events
    write_fcs(_frame(), str(tmp_path / 'f.fcs'))
    assert (fcs_read.FlowData(str(tmp_path / 'f.fcs')).events
            == flowio.FlowData(str(tmp_path / 'f.fcs')).events)


def test_the_flowio_hook_is_still_there():
    """The override relies on FlowIO's private reader method; if a FlowIO
    upgrade renames it, the fast path silently stops being used. Fail
    instead."""
    assert '_FlowData__parse_int_data' in vars(flowio.FlowData)
    assert '_FlowData__parse_int_data' in vars(fcs_read.FlowData)


def test_masking_is_much_faster_than_flowio(tmp_path):
    """200k events x 20 channels, 10 bits in 16: FlowIO's byte loop takes
    about a second; the numpy mask, milliseconds. Timed against FlowIO on the
    same machine, so a slow runner slows both."""
    rng = np.random.default_rng(2)
    data = rng.integers(0, 2 ** 16, size=(200_000, 20), dtype=np.uint64)
    path = str(tmp_path / 'big.fcs')
    write_int_fcs(path, data, [1024] * 20)

    def best(reader):
        times = []
        for _ in range(2):
            t = time.perf_counter()
            reader(path)
            times.append(time.perf_counter() - t)
        return min(times)

    assert best(fcs_read.FlowData) * 5 < best(flowio.FlowData)


def test_flowsample_loads_masked_values(tmp_path, monkeypatch):
    """The app's loader uses the fast reader and sees the masked values."""
    from openflo.pipeline import FlowSample
    calls = []
    real = fcs_read.FlowData._FlowData__parse_int_data

    def spy(self, *a):
        calls.append(1)
        return real(self, *a)

    monkeypatch.setattr(fcs_read.FlowData, '_FlowData__parse_int_data', spy)
    rng = np.random.default_rng(3)
    data = rng.integers(0, 2 ** 16, size=(400, 3), dtype=np.uint64)
    path = str(tmp_path / 'sample.fcs')
    write_int_fcs(path, data, [1024, 1024, 1024])
    s = FlowSample(path)
    np.testing.assert_array_equal(s.raw[['FL1-A', 'FL2-A', 'FL3-A']].to_numpy(),
                                  (data & 1023).astype(float))
    assert calls, 'FlowSample did not read through openflo.fcs_read'
