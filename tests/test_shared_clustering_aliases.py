"""The shared clustering lives in openflo.shared_clustering; its old names on
openflo.cli still resolve to the same objects, each with a
DeprecationWarning that names the new one."""
from __future__ import annotations

import subprocess
import sys

import pytest

import openflo.cli as cli
from openflo import shared_clustering

OLD_TO_NEW = {
    '_cluster_settings': 'cluster_settings',
    '_cluster_by_method': 'cluster_by_method',
    '_equal_quota': 'equal_quota',
    '_fold_label': 'fold_label',
    '_fluor_tokens': 'fluor_tokens',
    '_label_key': 'label_key',
    '_align_shared_channels': 'align_shared_channels',
    'align_shared_channels': 'align_shared_channels',
    '_shared_cluster_labels': 'shared_cluster_labels',
    '_shared_pool_cap': 'shared_pool_cap',
}


@pytest.mark.parametrize('old,new', sorted(OLD_TO_NEW.items()))
def test_old_cli_name_is_the_new_object_with_a_warning(old, new):
    with pytest.warns(DeprecationWarning, match=f'use shared_clustering.{new}'):
        got = getattr(cli, old)
    assert got is getattr(shared_clustering, new)


def test_an_unknown_cli_name_still_raises():
    with pytest.raises(AttributeError):
        cli._no_such_name  # noqa: B018


def test_shared_clustering_does_not_import_the_cli_or_a_display():
    code = ('import sys, openflo.shared_clustering\n'
            'bad = [m for m in ("openflo.cli", "tkinter", "matplotlib.pyplot",\n'
            '                   "openflo.theme", "openflo.gui") if m in sys.modules]\n'
            'print(bad)\n')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '[]'
