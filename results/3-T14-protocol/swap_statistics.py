"""3-T14: O <-> Cl swap statistics of a protocol run, per 100 K target bin and ramp.

    python results/3-T14-protocol/swap_statistics.py RUN_DIR OUT.json [--frames-per-bin 4]
        [--swaps-per-frame 50] [--seed 1] [--bin-K 100] [--min-K 250] [--no-energies]

- **Acceptance** from the thermo file's cumulative ``mc_swap_attempted`` / ``mc_swap_accepted``
  columns: the attempts and acceptances between consecutive rows, by the row's target and phase.
- **Attempted ΔE**, which the captured attempt graph does not keep: on ``--frames-per-bin`` saved
  frames per bin (evenly spaced), ``--swaps-per-frame`` uniform O–Cl pairs each. A swap is evaluated
  as the exchange of the two atoms' positions (the same configuration as exchanging their species,
  with the species array unchanged, so the calculator keeps its plan). Energies from ``fm.calc`` on
  the run's model in float32, as the run's MC energies (the move-energy gate chose float32).
  ΔE is reported in units of k_B T at the frame's target: median, 1st and 5th percentiles, minimum,
  and the mean Metropolis probability min(1, exp(−ΔE/kT)).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

K_B_EV = 8.617333262e-5


def _thermo(directory):
    from fastmace.io.output import read_thermo

    header, rows = read_thermo(directory)
    column = {name: header.index(name) for name in header}
    return [{"step": int(r[column["step"]]), "phase": r[column["phase"]], "target_K": float(r[column["target_K"]]),
             "attempted": int(r[column["mc_swap_attempted"]]), "accepted": int(r[column["mc_swap_accepted"]])}
            for r in rows if r]


def _bin(row, bin_K, min_K):
    if row["target_K"] < min_K or row["phase"] not in ("heating", "cooling"):
        return None
    low = math.floor(row["target_K"] / bin_K) * bin_K
    return f"{row['phase']} {low:g}-{low + bin_K:g} K"


def swap_ratios(atoms, calculator, first, second, count, kT, rng):
    """ΔE/kT of ``count`` uniform swaps of an atom of species ``first`` with one of ``second`` in ``atoms``,
    each evaluated as the exchange of the two atoms' positions from the unswapped frame."""
    import numpy as np

    atoms.calc = calculator
    base = atoms.get_potential_energy()
    one, other = np.flatnonzero(atoms.numbers == first), np.flatnonzero(atoms.numbers == second)
    positions = atoms.get_positions()
    ratios = []
    for _ in range(count):
        i, j = int(rng.choice(one)), int(rng.choice(other))
        trial = positions.copy()
        trial[[i, j]] = trial[[j, i]]
        atoms.set_positions(trial)
        ratios.append(float((atoms.get_potential_energy() - base) / kT))
    atoms.set_positions(positions)
    return ratios


def summary(ratios):
    """Median, 1st and 5th percentiles and minimum of ΔE/kT, and the mean Metropolis probability."""
    return {"sampled_swaps": len(ratios), "dE_over_kT_median": _percentile(ratios, 0.5),
            "dE_over_kT_p01": _percentile(ratios, 0.01), "dE_over_kT_p05": _percentile(ratios, 0.05),
            "dE_over_kT_min": min(ratios),
            "mean_metropolis_probability": sum(math.exp(-r) if r > 0 else 1.0 for r in ratios) / len(ratios)}


def _percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    below, above = math.floor(position), math.ceil(position)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("out")
    parser.add_argument("--frames-per-bin", type=int, default=4)
    parser.add_argument("--swaps-per-frame", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bin-K", type=float, default=100.0)
    parser.add_argument("--min-K", type=float, default=250.0)
    parser.add_argument("--no-energies", action="store_true", help="acceptance only: no model, no GPU")
    args = parser.parse_args(argv)
    run = Path(args.run)
    simulation = json.loads((run / "simulation.json").read_text(encoding="utf-8"))
    frame_every = simulation["output"]["trajectory_every"]

    rows = _thermo(run)
    bins: dict[str, dict] = {}
    for previous, row in zip(rows, rows[1:]):
        name = _bin(row, args.bin_K, args.min_K)
        if name is None:
            continue
        entry = bins.setdefault(name, {"attempted": 0, "accepted": 0, "frames": []})
        entry["attempted"] += row["attempted"] - previous["attempted"]
        entry["accepted"] += row["accepted"] - previous["accepted"]
        if row["step"] % frame_every == 0:
            entry["frames"].append(row)

    import numpy as np

    from fastmace import calc
    from fastmace.protocols.quench import read_frames_at

    chosen = {}
    for name, entry in bins.items():
        frames = entry["frames"]
        count = 0 if args.no_energies else min(args.frames_per_bin, len(frames))
        picks = [frames[round((j + 0.5) * len(frames) / count - 0.5)] for j in range(count)] if count else []
        entry["picked"] = picks
        for row in picks:
            chosen[row["step"] // frame_every] = row
    atoms_at = read_frames_at(run, sorted(chosen))[0] if chosen else {}
    model = simulation["model"]
    calculator = None if args.no_energies else calc(model["label"], head=model["head"], sha256=model["sha256"])
    rng = np.random.default_rng(args.seed)

    report = {"run": str(run), "frames_per_bin": args.frames_per_bin, "swaps_per_frame": args.swaps_per_frame,
              "seed": args.seed, "precision": "float32", "bins": {}}
    for name in sorted(bins, key=lambda n: (n.split()[0] != "heating", float(n.split()[1].split("-")[0])
                                            * (1 if n.startswith("heating") else -1))):
        entry = bins[name]
        ratios = []
        for row in entry["picked"]:
            atoms = atoms_at[row["step"] // frame_every].copy()
            ratios += swap_ratios(atoms, calculator, 8, 17, args.swaps_per_frame, K_B_EV * row["target_K"], rng)
        result = {"attempted": entry["attempted"], "accepted": entry["accepted"],
                  "acceptance": entry["accepted"] / entry["attempted"] if entry["attempted"] else None,
                  "sampled_frames": [row["step"] for row in entry["picked"]]}
        if ratios:
            result.update(summary(ratios))
        report["bins"][name] = result
        print(name, json.dumps(result), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with open(Path(args.out).with_suffix(".csv"), "w", newline="") as handle:
        fields = ["bin", "attempted", "accepted", "acceptance", "sampled_swaps", "dE_over_kT_median",
                  "dE_over_kT_p01", "dE_over_kT_p05", "dE_over_kT_min", "mean_metropolis_probability"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for name, result in report["bins"].items():
            writer.writerow({"bin": name, **result})


if __name__ == "__main__":
    main()
