"""Two samples must never be handed the same processed-data sidecar.

`safe_sidecar_name` mapped every character outside `[alnum-_]` to `_`, which is
not injective: `Tube 01 Rep A` and `Tube_01_Rep_A` produced the
same stem. `_write_session` wrote `<stem>.csv` with no uniqueness check and
recorded that one path on BOTH entries — and `_restore_session` PREFERS the
sidecar over the raw FCS, because it is the copy carrying clusters, embeddings
and compensated values.

The result was silent data substitution: the first sample came back holding the
second sample's events, clusters and compensated values. Nothing raised,
nothing was logged, and the saved session looked complete.

Two defences. The stem now carries a digest of the original name whenever
sanitisation changed it, so distinct names cannot converge. And the writer
refuses to reuse a stem, reporting it through the existing `_sidecar_failures`
channel — which already means "this sample reloads without its computed
columns", a loss the user is told about — rather than overwriting.

These drive the real `_write_session`, not a copy of its loop.
"""
import json
import os
import types

import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

try:
    import tkinter as _tk  # noqa: F401

    from openflo.gui import ViewGateEditorWindow as V
except (ImportError, RuntimeError) as exc:                # pragma: no cover
    pytest.skip(f'openflo.gui not importable: {exc}', allow_module_level=True)

from openflo.paths import safe_sidecar_name  # noqa: E402
from tests.test_session import _session_stub  # noqa: E402

COLLIDING = ('Tube_01_Rep_A', 'Tube 01 Rep A')


def _use_samples(stub, samples):
    """Point every per-sample map in the stub at OUR samples, so the writer
    emits an entry for each of them and nothing else."""
    names = list(samples)
    stub._samples = samples
    stub._sample_order = names
    stub._sample_colors = {n: '#1f77b4' for n in names}
    stub._sample_plot_enabled = {n: True for n in names}
    stub._sample_trial = {n: 'Day 1' for n in names}
    stub._sample_is_comp = {}
    stub._sample_gates = {n: {} for n in names}
    stub._sample_gate_order = {n: [] for n in names}
    stub._active_sample = names[0]
    return stub


def _sample(level):
    """A sample carrying computed columns, so the writer gives it a sidecar.
    `level` marks which sample the events on disk actually came from."""
    return types.SimpleNamespace(
        path=f'/data/{level}.fcs',
        data=pd.DataFrame({'FSC-A': [1.0, 2.0, 3.0],
                           'CD3': [level, level, level],
                           'cluster': [0, 1, 0]}))


def _write(tmp_path, samples):
    stub = _session_stub()
    _use_samples(stub, samples)
    stub._session_state = lambda: V._session_state(stub)
    stub._has_computed_columns = V._has_computed_columns
    stub._sidecar_safe_name = V._sidecar_safe_name
    path = str(tmp_path / 'sess.flowsession')
    written = V._write_session(stub, path)
    return stub, written, tmp_path / 'sess_data'


def test_the_two_names_no_longer_share_a_stem():
    first, second = COLLIDING
    assert safe_sidecar_name(first) != safe_sidecar_name(second)


def test_each_colliding_sample_keeps_its_own_data(tmp_path):
    """The end-to-end statement: save a session holding both samples, then
    read back what actually landed on disk for each one."""
    first, second = COLLIDING
    _stub, written, data_dir = _write(
        tmp_path, {first: _sample(1.0), second: _sample(9.0)})

    entries = {e['name']: e for e in written['samples']}
    assert 'processed_csv' in entries[first]
    assert 'processed_csv' in entries[second]
    assert entries[first]['processed_csv'] != entries[second]['processed_csv'], (
        'both entries recorded the same sidecar: '
        f"{entries[first]['processed_csv']!r}")

    assert len(list(data_dir.glob('*.csv'))) == 2, (
        f'expected one sidecar per sample, got '
        f'{sorted(p.name for p in data_dir.glob("*.csv"))}')

    for name, expected in ((first, 1.0), (second, 9.0)):
        stem = entries[name]['processed_csv'].rsplit('/', 1)[-1]
        on_disk = pd.read_csv(data_dir / stem)
        assert on_disk['CD3'].iloc[0] == pytest.approx(expected), (
            f'{name!r} would restore the other sample\'s events')


def test_the_session_file_records_distinct_sidecars(tmp_path):
    """Restore reads these pointers, so they are what actually decides which
    events each sample gets back."""
    first, second = COLLIDING
    _write(tmp_path, {first: _sample(1.0), second: _sample(9.0)})
    saved = json.loads((tmp_path / 'sess.flowsession').read_text())
    pointers = [s.get('processed_csv') for s in saved['samples']]
    assert len(set(pointers)) == len(pointers), (
        f'two samples point at one file: {pointers}')


def test_an_ordinary_name_still_gets_a_plain_sidecar(tmp_path):
    """The digest must not appear where it is not needed, or every existing
    session's fallback lookup breaks."""
    _stub, written, data_dir = _write(tmp_path, {'A': _sample(1.0)})
    entry = written['samples'][0]
    assert entry['processed_csv'].endswith('/A.csv'), entry['processed_csv']
    assert (data_dir / 'A.csv').is_file()


def test_a_repeated_stem_is_reported_rather_than_overwritten(tmp_path):
    """Defence in depth, driven through the real writer by forcing the naming
    function to collide. The second sample must not quietly take the first
    one's file."""
    samples = {'A': _sample(1.0), 'B': _sample(9.0)}
    stub = _use_samples(_session_stub(), samples)
    stub._session_state = lambda: V._session_state(stub)
    stub._has_computed_columns = V._has_computed_columns
    stub._sidecar_safe_name = staticmethod(lambda _n: 'same')
    written = V._write_session(stub, str(tmp_path / 'sess.flowsession'))

    entries = {e['name']: e for e in written['samples']}
    assert 'processed_csv' in entries['A']
    assert 'processed_csv' not in entries['B'], (
        "the skipped sample still points at the other sample's file")
    assert 'B' in stub._sidecar_failures, (
        'the collision was not reported through _sidecar_failures')
    on_disk = pd.read_csv(tmp_path / 'sess_data' / 'same.csv')
    assert on_disk['CD3'].iloc[0] == pytest.approx(1.0), (
        "the first sample's data was overwritten by the second")


def test_a_sidecar_written_by_an_older_version_is_still_found(tmp_path):
    """Backward compatibility. Sidecars written before the digest existed sit
    on disk under the plain substituted stem, and the fallback lookup is
    exactly what recovers clusters/UMAP when the recorded pointer was dropped.
    It must still find them."""
    from openflo.paths import legacy_sidecar_name

    name = 'Sample_A1 Stim_003'
    data_dir = tmp_path / 'last_session_data'
    data_dir.mkdir()
    legacy = data_dir / (legacy_sidecar_name(name) + '.csv')
    legacy.write_text('cluster,UMAP1\n0,0.1\n')
    assert legacy.name == 'Sample_A1_Stim_003.csv'      # the old scheme
    assert safe_sidecar_name(name) != legacy_sidecar_name(name)

    stub = types.SimpleNamespace(
        _session_dir=str(tmp_path), _session_data_dir=str(data_dir),
        _sidecar_safe_name=V._sidecar_safe_name)
    got = V._resolve_processed_csv(
        stub, {'processed_csv': None, 'path': '/raw/x.fcs'}, name)
    assert got and os.path.isfile(got), (
        'a sidecar from an older session became unreachable, so its clusters '
        'and embeddings would be silently lost on resume')
    assert os.path.basename(got) == legacy.name


def test_the_current_stem_wins_over_a_legacy_file(tmp_path):
    """When both exist, the digest-carrying one is this sample's own."""
    from openflo.paths import legacy_sidecar_name

    name = 'Sample_A1 Stim_003'
    data_dir = tmp_path / 'last_session_data'
    data_dir.mkdir()
    (data_dir / (legacy_sidecar_name(name) + '.csv')).write_text('cluster\n0\n')
    current = data_dir / (safe_sidecar_name(name) + '.csv')
    current.write_text('cluster\n1\n')

    stub = types.SimpleNamespace(
        _session_dir=str(tmp_path), _session_data_dir=str(data_dir),
        _sidecar_safe_name=V._sidecar_safe_name)
    got = V._resolve_processed_csv(
        stub, {'processed_csv': None, 'path': '/raw/x.fcs'}, name)
    assert os.path.basename(got) == current.name
