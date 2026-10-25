"""NOTICE.md lists every vendored third-party file; each exists and its directory has a LICENSE (6-T02)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _listed() -> list[str]:
    text = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    return re.findall(r"^\| `(fastmace/[^`]+)` \|", text, flags=re.MULTILINE)


def test_notice_lists_vendored_files():
    listed = _listed()
    assert "fastmace/longrange/data/dftd3_params.npz" in listed
    assert "fastmace/kernels/oeq_streaming/factorized_projected.cuh" in listed


def test_every_listed_file_exists_with_licence():
    for name in _listed():
        path = ROOT / name
        assert path.is_file(), f"{name} listed in NOTICE.md is missing"
        assert (path.parent / "LICENSE").is_file(), f"{path.parent} has no LICENSE"
        assert (path.parent / "NOTICE.md").is_file(), f"{path.parent} has no NOTICE.md"


def test_every_vendored_directory_is_listed():
    """A directory with its own LICENSE inside the package holds vendored files, which must be in NOTICE.md."""
    listed_dirs = {str(Path(name).parent).replace("\\", "/") for name in _listed()}
    for licence in (ROOT / "fastmace").rglob("LICENSE"):
        directory = licence.parent.relative_to(ROOT).as_posix()
        assert directory in listed_dirs, f"{directory} has a LICENSE but no file of it is in NOTICE.md"
