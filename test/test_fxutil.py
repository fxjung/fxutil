import os
from pathlib import Path

import pandas as pd
import pytest

from fxutil.common import bunny, get_git_repo_path


def test_get_git_repo_path(tmpdir):
    repo_path = get_git_repo_path()
    assert repo_path is not None
    assert isinstance(repo_path, Path)

    os.chdir(tmpdir)
    with pytest.raises(ValueError, match="not part of a git repository"):
        get_git_repo_path()


def test_bunny():
    assert bunny(pd.Series([1, 1])) == 1
    assert bunny([1, 1]) == 1
    assert bunny((1, 1)) == 1
    assert bunny({1, 1}) == 1
