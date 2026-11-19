"""Test files are named after the feature they check, not the work batch
that added them.

A file such as ``test_batch_j_data_handling.py`` mixed unrelated features
(sample groups, the plate view, concatenation, population export and
downsampling) under the name of the batch of work that built them, next to
real feature files like ``test_batch_normalize.py``. Batch-named files are
split by feature, and no new one may be added.
"""
from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parent

# test_batch_<letter>_<words>.py; test_batch_normalize*.py is a feature name.
BATCH_NAMED = re.compile(r'^test_batch_[a-z]_\w+\.py$')

NOT_YET_SPLIT: set[str] = set()


def _batch_named(names):
    return {n for n in names if BATCH_NAMED.match(n)}


def test_no_new_test_file_is_named_after_a_work_batch():
    found = _batch_named(p.name for p in TESTS.glob('test_*.py'))
    assert found <= NOT_YET_SPLIT, (
        f"name these test files after the feature they check: "
        f"{sorted(found - NOT_YET_SPLIT)}")


def test_the_not_yet_split_list_only_names_files_that_exist():
    gone = {n for n in NOT_YET_SPLIT if not (TESTS / n).exists()}
    assert not gone, f"split or renamed, drop from NOT_YET_SPLIT: {sorted(gone)}"


def test_the_pattern_tells_batch_names_from_feature_names():
    assert _batch_named(['test_batch_j_data_handling.py',
                         'test_batch_normalize.py',
                         'test_batch_normalize_dialog.py',
                         'test_comp_apply_to_batch.py',
                         'test_plate_view.py']) == {
        'test_batch_j_data_handling.py'}
