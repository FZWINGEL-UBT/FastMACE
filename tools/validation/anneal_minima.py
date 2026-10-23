"""The annealing-benchmark minima: GPU LBFGS against ASE's LBFGS on the 210 quench frames (plan task 4-T08,
SPEC decision 46).

The frames are the research engine's ``captured.traj`` of NaCl and the nine production crystals (21 each,
3x3x3, ResearchModule ``Experiments/Langevin_Thermostat/results/anneal-*-rtx5060ti-20261010/``). Its minima
are recorded beside them, not judged: the 10 October BFGS quench (``quenched.traj``, ``quench.csv``) and the
13 October ASE LBFGS re-quench (``quench-optimizer-comparison-rtx5060ti-20261013/<crystal>/lbfgs.traj``,
``comparison.csv``). The trajectories are not in ResearchModule's git, so ``fetch`` copies them once from a
checkout into a data directory and checks them against the manifest committed here
(``anneal_minima.sha256``); every later step reads only that directory.

    python3 tools/validation/anneal_minima.py fetch --research /mnt/g/Margraf/ResearchModule \\
        --data ~/fastmace-data/anneal-minima
    python3 tools/validation/anneal_minima.py verify --data ~/fastmace-data/anneal-minima

The runs are ``tools/bench/optimizer_frames.py``, one process per side and crystal (``--frames all
--resume``): ``--side ase`` (ASE's LBFGS on ``fm.calc``) and ``--side gpu --modes graph --skins auto``
(``Simulation(optimizer=fm.optimize.LBFGS())``, the plan's mode and skin). ``evaluate`` pairs them frame by
frame and applies decision 46's pass rule:

    python3 tools/validation/anneal_minima.py evaluate --data ~/fastmace-data/anneal-minima \\
        --runs results/4-T08-<machine>-<date>

- every one of the 210 relaxations converged, on both sides;
- per frame |dE| <= 0.025 meV/atom;
- the same structure (every atom within 0.1 A, minimum image) in at least 20 of 21 frames per crystal.

It writes ``frames.csv`` and ``summary.json`` into the runs directory. Needs ASE and NumPy only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

CRYSTALS = ("NaCl", "NaNbCl6", "NaTaCl6", "NaNbO3", "NaTaO3", "NbCl3O", "TaCl3O", "NbCl2O", "NbCl5", "TaCl5")
FILES = ("captured.traj", "quenched.traj", "quench.csv", "lbfgs.traj", "comparison.csv")
FRAMES_PER_CRYSTAL = 21
MANIFEST = Path(__file__).with_suffix(".sha256")
RESULTS = Path("Experiments/Langevin_Thermostat/results")
QUENCH = "quench-min500K-bin100K-n2-fmax0.01"
COMPARISON = "quench-optimizer-comparison-rtx5060ti-20261013"

# SPEC decision 46
DE_LIMIT_MEV_ATOM = 0.025
DISTANCE_LIMIT_A = 0.1
MIN_SAME_STRUCTURE = 20


def research_sources(research: Path, crystal: str) -> dict[str, Path]:
    """Where each of ``FILES`` of ``crystal`` is in a ResearchModule checkout."""
    results = Path(research) / RESULTS
    run = "anneal-NaCl-rtx5060ti-20261010" if crystal == "NaCl" else "anneal-production-rtx5060ti-20261010"
    quench = results / run / crystal / QUENCH
    comparison = results / COMPARISON / crystal
    return {"captured.traj": quench / "captured.traj", "quenched.traj": quench / "quenched.traj",
            "quench.csv": quench / "quench.csv", "lbfgs.traj": comparison / "lbfgs.traj",
            "comparison.csv": comparison / "comparison.csv"}


def research_summary(research: Path) -> Path:
    return Path(research) / RESULTS / COMPARISON / "summary.json"


def read_manifest(path: Path) -> dict[str, str]:
    """``sha256sum`` lines -> {relative path: digest}."""
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.strip().removeprefix("./")] = digest
    return entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(data: Path, manifest: Path = MANIFEST) -> int:
    """Check every file of ``manifest`` in ``data``; returns how many. Raises on a missing or changed file."""
    entries = read_manifest(manifest)
    for name, digest in entries.items():
        path = Path(data) / name
        if not path.is_file():
            raise FileNotFoundError(f"{name}: missing from {data}")
        if _sha256(path) != digest:
            raise ValueError(f"{name}: sha256 differs from {manifest.name}")
    return len(entries)


def fetch(research: Path, data: Path, manifest: Path = MANIFEST, crystals=CRYSTALS) -> int:
    """Copy the files from a ResearchModule checkout (read only) into ``data``, then ``verify``."""
    for crystal in crystals:
        (Path(data) / crystal).mkdir(parents=True, exist_ok=True)
        for name, source in research_sources(research, crystal).items():
            shutil.copyfile(source, Path(data) / crystal / name)
    shutil.copyfile(research_summary(research), Path(data) / "comparison-summary.json")
    return verify(data, manifest)


def distances(cell, a, b) -> tuple[float, float, float]:
    """Per-atom distances from ``a`` to ``b`` (minimum image): the largest; and, with the rigid shift removed
    (the 13 October comparison's measure), the largest and the RMS."""
    import numpy as np

    cell = np.asarray(cell, dtype=float)
    fractional = np.linalg.solve(cell.T, (np.asarray(b) - np.asarray(a)).T).T
    fractional -= np.round(fractional)
    delta = fractional @ cell
    largest = float(np.linalg.norm(delta, axis=1).max())
    delta = delta - delta.mean(axis=0)
    shifted = np.linalg.norm(delta, axis=1)
    return largest, float(shifted.max()), float(np.sqrt((shifted ** 2).mean()))


def side_rows(ase_json: Path, gpu_json: Path, *, run: str, cells: dict) -> list[dict]:
    """One row per frame both sides relaxed: steps, convergence, time, dE (GPU - ASE) and the distances
    between the two minima. ``cells``: crystal -> its (fixed) cell."""
    import numpy as np

    ase = json.loads(Path(ase_json).read_text(encoding="utf-8"))
    gpu = json.loads(Path(gpu_json).read_text(encoding="utf-8"))
    ase_minima = np.load(Path(ase_json).with_suffix(".npz"))
    gpu_minima = np.load(Path(gpu_json).with_suffix(".npz"))
    rows = []
    for crystal, record in ase["structures"].items():
        theirs = gpu["structures"].get(crystal, {}).get("runs", {}).get(run, {}).get("frames", {})
        for index, a in sorted(record["frames"].items(), key=lambda item: int(item[0])):
            g = theirs.get(index)
            if g is None:
                continue
            largest, shifted, rms = distances(cells[crystal], ase_minima[f"{crystal}/{index}"],
                                              gpu_minima[f"{crystal}/{run}/{index}"])
            rows.append({"crystal": crystal, "frame": int(index),
                         "ase_converged": bool(a["converged"]), "gpu_converged": bool(g["converged"]),
                         "ase_steps": a["steps"], "gpu_steps": g["steps"],
                         "ase_eV_atom": a["energy_eV_atom"], "gpu_eV_atom": g["energy_eV_atom"],
                         "dE_meV_atom": 1000 * (g["energy_eV_atom"] - a["energy_eV_atom"]),
                         "max_distance_A": largest, "max_distance_shift_removed_A": shifted,
                         "rms_distance_shift_removed_A": rms,
                         "ase_fmax_eV_A": a["fmax_eV_A"], "gpu_fmax_eV_A": g["fmax_eV_A"],
                         "ase_s": a["seconds"], "gpu_s": g["seconds"],
                         "ase_warm_up": bool(a.get("warm_up", int(index) == 0)),
                         "gpu_warm_up": bool(g.get("warm_up", int(index) == 0)),
                         "gpu_neighbour_rebuilds": g.get("neighbour_rebuilds"),
                         "gpu_recaptures": g.get("recaptures")})
    return rows


def judge(rows, crystals=CRYSTALS, frames=FRAMES_PER_CRYSTAL) -> dict:
    """SPEC decision 46's pass rule over ``rows`` (``side_rows``'s keys)."""
    failures, by_crystal = [], {}
    for crystal in crystals:
        mine = [r for r in rows if r["crystal"] == crystal]
        same = sum(r["max_distance_A"] <= DISTANCE_LIMIT_A for r in mine)
        by_crystal[crystal] = {
            "frames": len(mine), "same_structure": same,
            "max_abs_dE_meV_atom": max((abs(r["dE_meV_atom"]) for r in mine), default=None),
            "max_distance_A": max((r["max_distance_A"] for r in mine), default=None)}
        if len(mine) != frames:
            failures.append(f"{crystal}: {len(mine)} of {frames} frames relaxed on both sides")
        for r in sorted(mine, key=lambda r: r["frame"]):
            for side in ("ase", "gpu"):
                if not r[f"{side}_converged"]:
                    failures.append(f"{crystal} frame {r['frame']}: {side} did not converge")
            if abs(r["dE_meV_atom"]) > DE_LIMIT_MEV_ATOM:
                failures.append(f"{crystal} frame {r['frame']}: |dE| {abs(r['dE_meV_atom']):.4g} meV/atom "
                                f"> {DE_LIMIT_MEV_ATOM}")
        if mine and same < MIN_SAME_STRUCTURE:
            failures.append(f"{crystal}: the same structure in {same}/{frames} frames, "
                            f"fewer than {MIN_SAME_STRUCTURE}")
    return {"passed": not failures, "failures": failures, "crystals": by_crystal,
            "rule": {"dE_limit_meV_atom": DE_LIMIT_MEV_ATOM, "distance_limit_A": DISTANCE_LIMIT_A,
                     "min_same_structure": MIN_SAME_STRUCTURE, "frames_per_crystal": frames}}


def reference_rows(data: Path, crystal: str, gpu_minima, run: str, gpu_frames: dict) -> list[dict]:
    """The GPU minima against the research engine's 13 October LBFGS and 10 October BFGS minima (recorded,
    not judged: another calculator stack)."""
    from ase.io import read

    directory = Path(data) / crystal
    lbfgs = read(directory / "lbfgs.traj", ":")
    bfgs = read(directory / "quenched.traj", ":")
    with open(directory / "comparison.csv", newline="") as handle:
        comparison = list(csv.DictReader(handle))
    rows = []
    for index, g in sorted(gpu_frames.items(), key=lambda item: int(item[0])):
        i = int(index)
        positions = gpu_minima[f"{crystal}/{run}/{index}"]
        cell = lbfgs[i].cell.array
        row = {"crystal": crystal, "frame": i, "md_frame": int(comparison[i]["frame"]),
               "phase": comparison[i]["phase"]}
        for name, atoms, energy, steps in (
                ("rm_lbfgs", lbfgs[i], float(comparison[i]["lbfgs_eV_atom"]), comparison[i]["lbfgs_steps"]),
                ("rm_bfgs", bfgs[i], float(comparison[i]["bfgs_eV_atom"]), comparison[i]["bfgs_steps"])):
            largest, shifted, _ = distances(cell, atoms.positions, positions)
            row[f"dE_vs_{name}_meV_atom"] = 1000 * (g["energy_eV_atom"] - energy)
            row[f"max_distance_vs_{name}_A"] = largest
            row[f"max_distance_shift_removed_vs_{name}_A"] = shifted
            row[f"{name}_steps"] = int(steps)
        rows.append(row)
    return rows


def evaluate(data: Path, runs: Path, run: str = "graph-skinauto", crystals=CRYSTALS) -> dict:
    """Pair ``runs/<crystal>/ase.json`` with ``runs/<crystal>/gpu.json``, judge, and write ``frames.csv``
    and ``summary.json`` into ``runs``."""
    import numpy as np
    from ase.io import read

    rows, references, timing = [], [], {}
    for crystal in crystals:
        ase_json, gpu_json = Path(runs) / crystal / "ase.json", Path(runs) / crystal / "gpu.json"
        if not (ase_json.is_file() and gpu_json.is_file()):
            continue
        cell = read(Path(data) / crystal / "captured.traj", index=0).cell.array
        mine = side_rows(ase_json, gpu_json, run=run, cells={crystal: cell})
        rows += mine
        gpu = json.loads(gpu_json.read_text(encoding="utf-8"))
        gpu_frames = gpu["structures"][crystal]["runs"][run]["frames"]
        references += reference_rows(data, crystal, np.load(gpu_json.with_suffix(".npz")), run, gpu_frames)
        timing[crystal] = {}
        for side in ("ase", "gpu"):
            warm = [r for r in mine if not r[f"{side}_warm_up"]]  # without the compiles and captures
            timing[crystal][side] = {
                "seconds": sum(r[f"{side}_s"] for r in mine), "steps": sum(r[f"{side}_steps"] for r in mine),
                "warm_up_frames": len(mine) - len(warm),
                "ms_per_step_warm": 1000 * sum(r[f"{side}_s"] for r in warm) / max(sum(r[f"{side}_steps"]
                                                                                    for r in warm), 1)}
        timing[crystal]["gpu"]["setup_s"] = gpu["structures"][crystal]["runs"][run].get("setup_s")
    verdict = judge(rows, crystals)
    summary = {"task": "4-T08", "decision": "SPEC decision 46", "run": run, "verdict": verdict,
               "timing": timing, "references": {
                   crystal: {key: max(abs(r[key]) for r in references if r["crystal"] == crystal)
                             for key in ("dE_vs_rm_lbfgs_meV_atom", "max_distance_vs_rm_lbfgs_A",
                                         "max_distance_shift_removed_vs_rm_lbfgs_A", "dE_vs_rm_bfgs_meV_atom",
                                         "max_distance_shift_removed_vs_rm_bfgs_A")}
                   for crystal in {r["crystal"] for r in references}}}
    merged = {(r["crystal"], r["frame"]): dict(r) for r in rows}
    for r in references:
        merged.setdefault((r["crystal"], r["frame"]), {}).update(r)
    table = [merged[key] for key in sorted(merged, key=lambda k: (crystals.index(k[0]), k[1]))]
    if table:
        fields = list(dict.fromkeys(name for row in table for name in row))
        with open(Path(runs) / "frames.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
    (Path(runs) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="copy the frames from a ResearchModule checkout and verify them")
    f.add_argument("--research", required=True, type=Path)
    f.add_argument("--data", required=True, type=Path)
    v = sub.add_parser("verify", help="check the data directory against the manifest")
    v.add_argument("--data", required=True, type=Path)
    e = sub.add_parser("evaluate", help="pair the two sides, judge, write frames.csv and summary.json")
    e.add_argument("--data", required=True, type=Path)
    e.add_argument("--runs", required=True, type=Path)
    e.add_argument("--run", default="graph-skinauto")
    args = p.parse_args(argv)
    if args.command == "fetch":
        print(f"{fetch(args.research, args.data.expanduser())} files copied and verified")
    elif args.command == "verify":
        print(f"{verify(args.data.expanduser())} files verified")
    else:
        verify(args.data.expanduser())
        summary = evaluate(args.data.expanduser(), args.runs, args.run)
        verdict = summary["verdict"]
        for crystal, c in verdict["crystals"].items():
            print(f"{crystal}: {c['frames']} frames, same structure {c['same_structure']}, "
                  f"max |dE| {c['max_abs_dE_meV_atom']} meV/atom, max distance {c['max_distance_A']} A")
        print("PASS" if verdict["passed"] else "FAIL:\n  " + "\n  ".join(verdict["failures"]))
        return 0 if verdict["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
