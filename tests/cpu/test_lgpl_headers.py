"""The ASE 3.27 ports carry their LGPL-2.1-or-later SPDX header and are listed in NOTICE.md (6-T02b, owner D8.2)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPDX = "# SPDX-License-Identifier: LGPL-2.1-or-later"
SECTION = "## ASE 3.27 ports (LGPL-2.1-or-later)"

# Every ASE 3.27 port (docs/licences.md, open question 3). A new port adds itself here and to NOTICE.md.
EXPECTED = {
    "fastmace/integrators/langevin.py",
    "fastmace/integrators/baoab.py",
    "fastmace/integrators/berendsen.py",
    "fastmace/integrators/csvr.py",
    "fastmace/integrators/nose_hoover.py",
    "fastmace/integrators/mtk.py",
    "fastmace/integrators/nve.py",
    "fastmace/optimize/fire.py",
    "fastmace/optimize/lbfgs.py",
    "fastmace/optimize/cell.py",
    "fastmace/reference/integrators.py",
    "fastmace/kernels/triton/fused_integrator.py",
    "fastmace/kernels/triton/nose_hoover_chain.py",
}


def _notice_ports() -> set[str]:
    text = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    section = text.split(SECTION, 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `(fastmace/[^`]+)` \|", section, flags=re.MULTILINE))


def _marked() -> set[str]:
    return {path.relative_to(ROOT).as_posix() for path in (ROOT / "fastmace").rglob("*.py")
            if SPDX in path.read_text(encoding="utf-8").splitlines()[:3]}


def test_every_port_has_header_and_provenance():
    for name in sorted(EXPECTED):
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        assert lines[0] == SPDX, f"{name} does not start with {SPDX!r}"
        assert lines[1].startswith("# Provenance:") and "ASE 3.27" in lines[1], f"{name} has no provenance line"


def test_marked_files_are_exactly_the_listed_ports():
    assert _marked() == EXPECTED
    assert _notice_ports() == EXPECTED


def test_licence_expression_and_text():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "MIT AND LGPL-2.1-or-later"' in pyproject
    text = (ROOT / "LICENSES" / "LGPL-2.1-or-later.txt").read_text(encoding="utf-8")
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in text and "Version 2.1, February 1999" in text
