"""The annealing-benchmark minima tools (plan task 4-T08, SPEC decision 46): ``tools/validation/anneal_minima.py``
(manifest, distances, rows, pass rule) and the 4-T08 options of ``tools/bench/optimizer_frames.py``."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _tool(relative):
    path = REPOSITORY_ROOT / "tools" / relative
    spec = importlib.util.spec_from_file_location(f"_tool_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


minima = _tool("validation/anneal_minima.py")
frames_tool = _tool("bench/optimizer_frames.py")


# --- manifest ---------------------------------------------------------------------------------------------


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_manifest_lists_the_five_files_of_every_crystal_and_the_summary():
    entries = minima.read_manifest(minima.MANIFEST)
    assert len(entries) == 5 * len(minima.CRYSTALS) + 1
    for crystal in minima.CRYSTALS:
        for name in minima.FILES:
            assert f"{crystal}/{name}" in entries
    assert "comparison-summary.json" in entries
    assert all(len(digest) == 64 for digest in entries.values())


def test_verify_accepts_matching_files(tmp_path):
    manifest = tmp_path / "m.sha256"
    a = _write(tmp_path / "data" / "NaCl" / "captured.traj", b"frames")
    manifest.write_text(f"{a}  ./NaCl/captured.traj\n")
    assert minima.verify(tmp_path / "data", manifest) == 1


def test_verify_refuses_a_changed_file(tmp_path):
    manifest = tmp_path / "m.sha256"
    a = _write(tmp_path / "data" / "NaCl" / "captured.traj", b"frames")
    manifest.write_text(f"{a}  ./NaCl/captured.traj\n")
    (tmp_path / "data" / "NaCl" / "captured.traj").write_bytes(b"other frames")
    with pytest.raises(ValueError, match="NaCl/captured.traj"):
        minima.verify(tmp_path / "data", manifest)


def test_verify_refuses_a_missing_file(tmp_path):
    manifest = tmp_path / "m.sha256"
    manifest.write_text(f"{'0' * 64}  ./NaCl/captured.traj\n")
    with pytest.raises(FileNotFoundError, match="NaCl/captured.traj"):
        minima.verify(tmp_path / "data", manifest)


def test_fetch_copies_from_the_research_layout_and_verifies(tmp_path):
    research = tmp_path / "rm"
    lines = []
    for crystal in ("NaCl", "NbCl5"):
        for name, source in minima.research_sources(research, crystal).items():
            digest = _write(source, f"{crystal} {name}".encode())
            lines.append(f"{digest}  ./{crystal}/{name}")
    summary = minima.research_summary(research)
    lines.append(f"{_write(summary, b'{}')}  ./comparison-summary.json")
    manifest = tmp_path / "m.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    destination = tmp_path / "data"
    assert minima.fetch(research, destination, manifest, crystals=("NaCl", "NbCl5")) == len(lines)
    assert (destination / "NbCl5" / "lbfgs.traj").read_bytes() == b"NbCl5 lbfgs.traj"
    assert "anneal-NaCl-rtx5060ti-20261010" in str(minima.research_sources(research, "NaCl")["captured.traj"])
    assert "anneal-production-rtx5060ti-20261010" in str(minima.research_sources(research, "NbCl5")["quench.csv"])


# --- distances --------------------------------------------------------------------------------------------


def test_distances_use_the_minimum_image():
    cell = np.diag([10.0, 10.0, 10.0])
    a = np.array([[0.05, 5.0, 5.0], [5.0, 5.0, 5.0]])
    b = np.array([[9.98, 5.0, 5.0], [5.0, 5.0, 5.0]])
    largest, largest_shifted, rms_shifted = minima.distances(cell, a, b)
    assert largest == pytest.approx(0.07)
    assert largest_shifted == pytest.approx(0.035)
    assert rms_shifted == pytest.approx(0.035)


def test_distances_without_shift_see_a_rigid_translation():
    cell = np.array([[8.0, 0.0, 0.0], [1.0, 8.0, 0.0], [0.0, 0.0, 9.0]])
    rng = np.random.default_rng(1)
    a = rng.random((12, 3)) @ cell
    b = a + np.array([0.2, -0.1, 0.05])
    largest, largest_shifted, rms_shifted = minima.distances(cell, a, b)
    assert largest == pytest.approx(np.linalg.norm([0.2, -0.1, 0.05]))
    assert largest_shifted == pytest.approx(0.0, abs=1e-12)
    assert rms_shifted == pytest.approx(0.0, abs=1e-12)


# --- pass rule (SPEC decision 46) -------------------------------------------------------------------------


def _rows(override=None, crystals=("NaCl", "NbCl5"), n=21):
    rows = []
    for crystal in crystals:
        for index in range(n):
            row = {"crystal": crystal, "frame": index, "ase_converged": True, "gpu_converged": True,
                   "dE_meV_atom": 0.001, "max_distance_A": 0.001}
            row.update((override or {}).get((crystal, index), {}))
            rows.append(row)
    return rows


def test_judge_passes_when_every_clause_holds():
    verdict = minima.judge(_rows(), crystals=("NaCl", "NbCl5"))
    assert verdict["passed"]
    assert verdict["failures"] == []
    assert verdict["crystals"]["NbCl5"]["same_structure"] == 21
    assert verdict["crystals"]["NbCl5"]["max_abs_dE_meV_atom"] == pytest.approx(0.001)


@pytest.mark.parametrize("side", ["ase_converged", "gpu_converged"])
def test_judge_fails_an_unconverged_frame(side):
    verdict = minima.judge(_rows({("NbCl5", 3): {side: False}}), crystals=("NaCl", "NbCl5"))
    assert not verdict["passed"]
    assert any("NbCl5 frame 3" in f and "converge" in f for f in verdict["failures"])


def test_judge_energy_clause_with_21_of_21():
    verdict = minima.judge(_rows(), crystals=("NaCl", "NbCl5"))
    assert verdict["crystals"]["NaCl"]["same_energy"] == 21
    assert verdict["over_limit"] == []


def test_judge_allows_one_energy_difference_above_the_limit_per_crystal():
    # decision 46 as amended: |dE| <= 0.025 meV/atom in at least 20 of 21 frames per crystal
    rows = _rows({("NaCl", 7): {"dE_meV_atom": -0.0251}, ("NbCl5", 12): {"dE_meV_atom": 0.188}})
    verdict = minima.judge(rows, crystals=("NaCl", "NbCl5"))
    assert verdict["passed"]
    assert verdict["crystals"]["NaCl"]["same_energy"] == 20
    assert verdict["crystals"]["NbCl5"]["same_energy"] == 20
    # the frames over the limit stay listed
    assert any("NaCl frame 7" in f and "0.0251" in f for f in verdict["over_limit"])
    assert any("NbCl5 frame 12" in f and "0.188" in f for f in verdict["over_limit"])


def test_judge_fails_two_energy_differences_above_the_limit_in_one_crystal():
    rows = _rows({("NaCl", 7): {"dE_meV_atom": -0.0251}, ("NaCl", 8): {"dE_meV_atom": 0.03}})
    verdict = minima.judge(rows, crystals=("NaCl", "NbCl5"))
    assert not verdict["passed"]
    assert verdict["crystals"]["NaCl"]["same_energy"] == 19
    assert any("NaCl" in f and "19/21" in f and "dE" in f for f in verdict["failures"])
    assert len(verdict["over_limit"]) == 2


def test_judge_accepts_the_energy_limit_itself():
    verdict = minima.judge(_rows({("NaCl", 7): {"dE_meV_atom": 0.025}}), crystals=("NaCl", "NbCl5"))
    assert verdict["passed"]
    assert verdict["over_limit"] == []


def test_judge_allows_one_different_structure_per_crystal():
    rows = _rows({("NbCl5", 4): {"max_distance_A": 0.14}, ("NaCl", 2): {"max_distance_A": 0.3}})
    verdict = minima.judge(rows, crystals=("NaCl", "NbCl5"))
    assert verdict["passed"]
    assert verdict["crystals"]["NbCl5"]["same_structure"] == 20


def test_judge_fails_two_different_structures_in_one_crystal():
    rows = _rows({("NbCl5", 4): {"max_distance_A": 0.14}, ("NbCl5", 9): {"max_distance_A": 0.11}})
    verdict = minima.judge(rows, crystals=("NaCl", "NbCl5"))
    assert not verdict["passed"]
    assert any("NbCl5" in f and "19/21" in f for f in verdict["failures"])


def test_judge_fails_missing_frames_and_crystals():
    rows = [r for r in _rows() if not (r["crystal"] == "NbCl5" and r["frame"] == 20)]
    verdict = minima.judge(rows, crystals=("NaCl", "NbCl5", "TaCl5"))
    assert not verdict["passed"]
    assert any("NbCl5" in f and "20 of 21" in f for f in verdict["failures"])
    assert any("TaCl5" in f and "0 of 21" in f for f in verdict["failures"])


# --- rows from the two sides' records ---------------------------------------------------------------------


def _side_records(tmp_path, *, positions_shift=0.0, energy_shift=0.0):
    cell = np.diag([6.0, 6.0, 6.0])
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
    ase = {"side": "ase", "structures": {"NaCl": {"traj": "x", "frames": {
        "0": {"converged": True, "steps": 10, "seconds": 1.0, "energy_eV_atom": -1.0, "fmax_eV_A": 0.009,
              "ms_per_step": 100.0}}}}}
    gpu = {"side": "gpu", "structures": {"NaCl": {"traj": "x", "runs": {"graph-skinauto": {"frames": {
        "0": {"converged": True, "steps": 11, "seconds": 0.5, "energy_eV_atom": -1.0 + energy_shift,
              "fmax_eV_A": 0.008, "ms_per_step": 45.0, "neighbour_rebuilds": 1, "recaptures": 0}}}}}}}
    (tmp_path / "ase.json").write_text(json.dumps(ase))
    (tmp_path / "gpu.json").write_text(json.dumps(gpu))
    np.savez(tmp_path / "ase.npz", **{"NaCl/0": positions})
    np.savez(tmp_path / "gpu.npz", **{"NaCl/graph-skinauto/0": positions + positions_shift})
    return cell


def test_rows_pair_the_two_sides_frame_by_frame(tmp_path):
    cell = _side_records(tmp_path, positions_shift=0.01, energy_shift=2e-8)
    rows = minima.side_rows(tmp_path / "ase.json", tmp_path / "gpu.json", run="graph-skinauto",
                            cells={"NaCl": cell})
    assert len(rows) == 1
    row = rows[0]
    assert (row["crystal"], row["frame"]) == ("NaCl", 0)
    assert row["dE_meV_atom"] == pytest.approx(2e-5)
    assert row["max_distance_A"] == pytest.approx(0.01 * np.sqrt(3))
    assert row["max_distance_shift_removed_A"] == pytest.approx(0.0, abs=1e-12)
    assert (row["ase_steps"], row["gpu_steps"]) == (10, 11)
    assert row["ase_converged"] and row["gpu_converged"]


def test_rows_leave_out_frames_only_one_side_has(tmp_path):
    cell = _side_records(tmp_path)
    gpu = json.loads((tmp_path / "gpu.json").read_text())
    gpu["structures"]["NaCl"]["runs"]["graph-skinauto"]["frames"]["5"] = {"converged": True}
    (tmp_path / "gpu.json").write_text(json.dumps(gpu))
    rows = minima.side_rows(tmp_path / "ase.json", tmp_path / "gpu.json", run="graph-skinauto",
                            cells={"NaCl": cell})
    assert [r["frame"] for r in rows] == [0]


# --- optimizer_frames.py: the 4-T08 options ---------------------------------------------------------------


def test_frames_all_and_skin_auto_parse():
    args = frames_tool.parse_args(["--side", "gpu", "--output", "o", "--checkpoint", "c", "--traj", "A=a",
                                   "--frames", "all", "--skins", "auto", "--resume"])
    assert args.frames == ["all"]
    assert args.skins == ["auto"]
    assert args.resume


def test_frame_indices():
    assert frames_tool.frame_indices(["all"], 21) == list(range(21))
    assert frames_tool.frame_indices(["0", "10", "20"], 21) == [0, 10, 20]
    with pytest.raises(ValueError, match="all"):
        frames_tool.frame_indices(["all", "3"], 21)
    with pytest.raises(IndexError):
        frames_tool.frame_indices(["21"], 21)


def test_resume_reads_the_previous_record_and_minima(tmp_path):
    output = tmp_path / "run" / "ase"
    assert frames_tool.load_previous(output) == (None, {})
    output.parent.mkdir()
    output.with_suffix(".json").write_text(json.dumps({"side": "ase", "structures": {"NaCl": {"frames": {"3": {}}}}}))
    np.savez(output.with_suffix(".npz"), **{"NaCl/3": np.zeros((2, 3))})
    meta, saved = frames_tool.load_previous(output)
    assert meta["structures"]["NaCl"]["frames"] == {"3": {}}
    assert saved["NaCl/3"].shape == (2, 3)


def test_skin_key_names_auto():
    assert frames_tool.run_key("graph", "auto") == "graph-skinauto"
    assert frames_tool.run_key("compiled", 0.6) == "compiled-skin0.6"


def test_git_commit_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setattr(frames_tool.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    monkeypatch.setenv("FASTMACE_COMMIT", "abc123")
    assert frames_tool.git_commit() == "abc123"


# --- anneal_frame_probe.py: where two relaxations part ----------------------------------------------------

probe = _tool("validation/anneal_frame_probe.py")


def test_divergence_finds_the_first_step_above_each_threshold():
    cell = np.diag([10.0, 10.0, 10.0])
    a = np.zeros((5, 2, 3)) + 5.0
    b = a.copy()
    for step, offset in enumerate([0.0, 5e-6, 5e-4, 0.05, 0.5]):
        b[step, 1, 0] += offset
    largest, first = probe.divergence(cell, a, b)
    assert largest == pytest.approx([0.0, 5e-6, 5e-4, 0.05, 0.5])
    assert first == {"1e-06": 1, "0.0001": 2, "0.01": 3, "0.1": 4}


def test_divergence_over_the_shorter_trajectory_and_minimum_image():
    cell = np.diag([4.0, 4.0, 4.0])
    a = np.zeros((3, 1, 3)) + [[0.01, 2.0, 2.0]]
    b = np.zeros((2, 1, 3)) + [[3.99, 2.0, 2.0]]
    largest, first = probe.divergence(cell, a, b)
    assert largest == pytest.approx([0.02, 0.02])
    assert first["0.1"] is None
