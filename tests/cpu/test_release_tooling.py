"""The release tooling never publishes (plan task 6-T12; docs/development.md, "Releasing")."""

from __future__ import annotations

import re
from pathlib import Path

import fastmace

ROOT = Path(__file__).resolve().parents[2]


def _code_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]


def test_make_release_never_uploads():
    lines = _code_lines(ROOT / "tools" / "release" / "make_release.sh")
    heredoc = False
    for line in lines:
        if line.startswith("cat <<"):
            heredoc = True  # the printed instructions
        elif heredoc and line == "EOF":
            heredoc = False
        elif not heredoc:
            assert "upload" not in line, f"make_release.sh runs an upload: {line!r}"


def test_release_workflow_is_manual_and_does_not_publish():
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    triggers = re.search(r"^on:\n((?:  .*\n)+)", text, flags=re.MULTILINE).group(1)
    assert re.findall(r"^  (\w+):", triggers, flags=re.MULTILINE) == ["workflow_dispatch"]
    for word in ("twine upload", "pypi-publish", "id-token", "secrets."):
        assert word not in text


def test_version_and_changelog_agree():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+|rc\d+)?", fastmace.__version__)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in changelog
    if ".dev" not in fastmace.__version__:
        assert f"## [{fastmace.__version__}] - " in changelog
