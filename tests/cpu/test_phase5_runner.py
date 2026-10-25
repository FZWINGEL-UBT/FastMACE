"""Failure handling and round-4 option provenance for the phase-5 local runner."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "acceptance" / "phase5_local.sh"


def _fake_environment(tmp_path, outcomes):
    """Return a HOME/PATH pair with harmless runner dependencies."""
    home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    python = home / "environments" / "mace_cueq_oeq" / "bin" / "python"
    fake_bin.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    (home / "models").mkdir()
    (fake_bin / "nvidia-smi").write_text(
        "#!/bin/sh\nprintf 'fake-gpu, 0, 0 %%, 0 MiB\\n'\n", encoding="utf-8"
    )
    calls = home / "calls"
    fake_python = """#!/bin/sh
set -eu
count=0
if [ -f "$FAKE_CALLS" ]; then count=$(cat "$FAKE_CALLS"); fi
count=$((count + 1))
printf '%s\\n' "$count" > "$FAKE_CALLS"
printf '%s\\n' "$*" >> "$FAKE_ARGS"
case "$count" in
    """ + "\n".join(f"  {index}) exit {status} ;;" for index, status in enumerate(outcomes, 1)) + """
  *) exit 0 ;;
esac
"""
    python.write_text(fake_python, encoding="utf-8")
    for executable in (fake_bin / "nvidia-smi", python):
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    env.update(
        HOME=str(home),
        PATH=os.pathsep.join((str(fake_bin), env.get("PATH", ""))),
        FASTMACE_4_COMMIT="baseline-for-test",
        RUN_DATE="runner-test",
        FAKE_CALLS=str(calls),
        FAKE_ARGS=str(home / "args"),
    )
    return env, home


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_phase5_runner_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs POSIX bash")
def test_regression_stops_after_first_failed_child(tmp_path):
    env, home = _fake_environment(tmp_path, [37, 0, 0])
    result = subprocess.run(
        ["bash", str(SCRIPT), str(ROOT), "regression"],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 37, result.stdout + result.stderr
    assert (home / "calls").read_text(encoding="utf-8").strip() == "1"


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs POSIX bash")
def test_regression_passes_round4_option_allowlist_to_each_child(tmp_path):
    env, home = _fake_environment(tmp_path, [0, 0, 0])
    result = subprocess.run(
        ["bash", str(SCRIPT), str(ROOT), "regression"],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = (home / "args").read_text(encoding="utf-8").splitlines()
    assert len(args) == 3
    for invocation in args:
        assert "--allow-option-changes grouped_launch fx_e3nn_linear polar_graph_sums" in invocation


def test_adaptation_production_preserves_both_piece_records_before_checking():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'cp "$root/adapt-production/$name/metadata.json"' in text
    assert 'evidence=$p/evidence' in text
    assert '"$evidence/npt_anneal_NbCl5_short-start"' in text
    assert '"$evidence/npt_anneal_NbCl5_short-resume"' in text
    assert '"$evidence/swap_anneal_NaNbOCl4_short-start"' in text
    assert '"$evidence/swap_anneal_NaNbOCl4_short-resume"' in text
    assert '"$evidence/nvt_anneal_NaCl-start"' in text
    assert '"$evidence/nvt_anneal_NaCl-resume"' in text
