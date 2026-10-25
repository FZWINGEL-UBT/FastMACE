"""The MkDocs site builds in strict mode (phase 6, 6-T06): every page is in the nav and every link resolves."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(importlib.util.find_spec("mkdocs") is None, reason="mkdocs is not installed (the docs extra)")
def test_mkdocs_builds_strictly(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(tmp_path / "site")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
