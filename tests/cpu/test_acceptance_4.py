"""``tools/acceptance/phase4_local.sh`` (4-T13): it parses, its preflight files exist, and every job it lists is a
case it handles."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "acceptance" / "phase4_local.sh"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
def test_the_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_the_preflight_files_exist():
    text = SCRIPT.read_text(encoding="utf-8")
    block = re.search(r"PREFLIGHT_FILES=\((.*?)\)", text, re.S).group(1)
    files = block.split()
    assert files and all((ROOT / name).is_file() for name in files), [n for n in files if not (ROOT / n).is_file()]


def test_every_job_of_the_runbook_has_a_case():
    text = SCRIPT.read_text(encoding="utf-8")
    cases = set()
    for line in re.findall(r"^\s{2}([a-z0-9|-]+)\)", text, re.M):
        cases.update(line.split("|"))
    assert {"regression", "regression-npt", "ase-speed", "preflight", "optimisers", "minima", "protocol-start",
            "protocol-resume", "nve-drift", "trajectories-cpu", "trajectories-reproducible", "restart-across",
            "cpu-acceptance", "acceptance-tier", "slow-tier", "targeted", "compat"} <= cases
