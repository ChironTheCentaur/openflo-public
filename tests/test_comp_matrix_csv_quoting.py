"""A channel name containing the separator must survive the file.

`write_compensation_matrix` built each line with `sep.join(...)` and the reader
split it back with `ln.split(sep)`. Neither quotes anything, so a channel name
containing the separator became two fields. A `$PnS` description like
`CD4 FL3, clone SK3` is entirely ordinary, and reading such a file back
failed with `ValueError: could not convert string to float: ' clone SK3'` —
the compensation matrix the user had just saved would not load.

Both sides now use the `csv` module, which quotes per RFC 4180. Files written
by earlier versions contain no quoting and parse identically.
"""
import numpy as np
import pytest

from openflo.pipeline import read_compensation_matrix, write_compensation_matrix

MATRIX = np.array([[1.0, 0.05], [0.02, 1.0]])


@pytest.mark.parametrize('channels, suffix', [
    pytest.param(['FL4-A', 'FL5-A'], '.csv', id='plain-csv'),
    pytest.param(['FL4-A', 'FL5-A'], '.tsv', id='plain-tsv'),
    pytest.param(['CD4 FL3, clone SK3', 'FL4-A'], '.csv', id='comma'),
    pytest.param(['CD8 "FL2"', 'FL5-A'], '.csv', id='quote'),
    pytest.param(['CD3\tFL2', 'FL5-A'], '.tsv', id='tab-in-tsv'),
    pytest.param(['CD4, clone SK3', 'CD8 "FL2"'], '.csv', id='comma-and-quote'),
])
def test_the_matrix_round_trips(tmp_path, channels, suffix):
    path = str(tmp_path / f'comp{suffix}')
    write_compensation_matrix(path, MATRIX, channels)
    back_channels, back_matrix = read_compensation_matrix(path)
    assert back_channels == channels
    assert back_matrix is not None
    np.testing.assert_allclose(back_matrix, MATRIX)


def test_a_file_from_an_earlier_version_still_reads(tmp_path):
    """Unquoted files are what every existing installation has on disk."""
    path = tmp_path / 'legacy.csv'
    path.write_text(',FL4-A,FL5-A\n'
                    'FL4-A,1.0,0.05\n'
                    'FL5-A,0.02,1.0\n', encoding='utf-8')
    channels, matrix = read_compensation_matrix(str(path))
    assert channels == ['FL4-A', 'FL5-A']
    assert matrix is not None
    np.testing.assert_allclose(matrix, MATRIX)


def test_the_leading_separator_of_a_tsv_is_still_significant(tmp_path):
    """The header row is aligned by an empty first cell; losing it would shift
    every channel by one column."""
    path = tmp_path / 'comp.tsv'
    write_compensation_matrix(str(path), MATRIX, ['FL4-A', 'FL5-A'])
    first = path.read_text(encoding='utf-8').splitlines()[0]
    assert first.startswith('\t'), repr(first)
