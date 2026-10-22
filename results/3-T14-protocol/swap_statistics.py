"""3-T14: O <-> Cl swap statistics of a protocol run, per 100 K target bin and ramp.

    python results/3-T14-protocol/swap_statistics.py RUN_DIR OUT.json [--frames-per-bin 4]
        [--swaps-per-frame 50] [--seed 1] [--bin-K 100] [--min-K 250]

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
        count = min(args.frames_per_bin, len(frames))
        picks = [frames[round((j + 0.5) * len(frames) / count - 0.5)] for j in range(count)] if count else []
        entry["picked"] = picks
        for row in picks:
            chosen[row["step"] // frame_every] = row
    atoms_at, _ = read_frames_at(run, sorted(chosen))
    model = simulation["model"]
    calculator = calc(model["label"], head=model["head"], sha256=model["sha256"])
    rng = np.random.default_rng(args.seed)

    report = {"run": str(run), "frames_per_bin": args.frames_per_bin, "swaps_per_frame": args.swaps_per_frame,
              "seed": args.seed, "precision": "float32", "bins": {}}
    for name in sorted(bins, key=lambda n: (n.split()[0] != "heating", float(n.split()[1].split("-")[0])
                                            * (1 if n.startswith("heating") else -1))):
        entry = bins[name]
        ratios, probabilities = [], []
        for row in entry["picked"]:
            atoms = atoms_at[row["step"] // frame_every].copy()
            atoms.calc = calculator
            kT = K_B_EV * row["target_K"]
            base = atoms.get_potential_energy()
            oxygen = np.flatnonzero(atoms.numbers == 8)
            chlorine = np.flatnonzero(atoms.numbers == 17)
            positions = atoms.get_positions()
            for _ in range(args.swaps_per_frame):
                i, j = int(rng.choice(oxygen)), int(rng.choice(chlorine))
                trial = positions.copy()
                trial[[i, j]] = trial[[j, i]]
                atoms.set_positions(trial)
                ratio = (atoms.get_potential_energy() - base) / kT
                ratios.append(float(ratio))
                probabilities.append(math.exp(-ratio) if ratio > 0 else 1.0)
            atoms.set_positions(positions)
        result = {"attempted": entry["attempted"], "accepted": entry["accepted"],
                  "acceptance": entry["accepted"] / entry["attempted"] if entry["attempted"] else None,
                  "sampled_frames": [row["step"] for row in entry["picked"]], "sampled_swaps": len(ratios)}
        if ratios:
            result.update(dE_over_kT_median=_percentile(ratios, 0.5), dE_over_kT_p01=_percentile(ratios, 0.01),
                          dE_over_kT_p05=_percentile(ratios, 0.05), dE_over_kT_min=min(ratios),
                          mean_metropolis_probability=sum(probabilities) / len(probabilities))
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
