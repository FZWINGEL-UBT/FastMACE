"""3-T14, for owner decision D3 option (b): the attempted Nb <-> Ta swap ΔE/kT in a 50/50 Na(Nb,Ta)Cl6.

    python results/3-T14-protocol/nbta_swaps.py OUT_DIR [--seed 20261022] [--ps 2] [--frames 4]
        [--swaps-per-frame 50]

- **Structure:** ResearchModule's benchmark ``Structures/Benchmark/R2SCAN/NaNbCl6.cif`` at tag
  ``research-engine-baseline-20261017`` (copied to ``nbta/NaNbCl6_R2SCAN.cif``, blob
  ``c833e38c59f3cc4b8b4a3e82d45a228d2a104491``), 3×3×3 (864 atoms), half of the 108 Nb replaced by Ta at
  random (``numpy.random.default_rng(seed).choice``), written to ``OUT_DIR/NaNb0.5Ta0.5Cl6_3x3x3.extxyz``.
- **MD:** ``fm.md`` NVT at 600 K and at 1000 K, ``--ps`` ps each at 1 fs from the substituted structure,
  MH-1 ``matpes_r2scan``, hybrid float32, frames every 100 steps.
- **Swaps:** on ``--frames`` frames of the second half of each run, ``--swaps-per-frame`` uniform Nb–Ta
  pairs each, evaluated as in ``swap_statistics.py`` (the two atoms' positions exchanged; ``fm.calc``
  float32), ΔE in units of k_B T at the run's temperature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swap_statistics import K_B_EV, summary, swap_ratios

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "nbta" / "NaNbCl6_R2SCAN.cif"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--ps", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--swaps-per-frame", type=int, default=50)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from ase.io import read

    from fastmace import Output, calc, md
    from fastmace.config import EngineConfig
    from fastmace.io.output import read_frames

    atoms = read(SOURCE).repeat((3, 3, 3))
    niobium = np.flatnonzero(atoms.numbers == 41)
    rng = np.random.default_rng(args.seed)
    tantalum = np.sort(rng.choice(niobium, size=len(niobium) // 2, replace=False))
    numbers = atoms.numbers.copy()
    numbers[tantalum] = 73
    atoms.numbers = numbers
    atoms.set_masses()  # standard masses for the new species
    atoms.write(out / "NaNb0.5Ta0.5Cl6_3x3x3.extxyz")

    steps = round(args.ps * 1000)
    calculator = calc("mh1", head="matpes_r2scan")
    report = {"source": "ResearchModule Structures/Benchmark/R2SCAN/NaNbCl6.cif @ research-engine-baseline-20261017",
              "repeat": [3, 3, 3], "atoms": len(atoms), "substitution_seed": args.seed,
              "tantalum_indices": [int(i) for i in tantalum], "ps": args.ps, "timestep_fs": 1.0,
              "precision": "float32", "temperatures": {}}
    for temperature in (600.0, 1000.0):
        directory = out / f"md-{temperature:g}K"
        run = md(atoms, model="mh1", head="matpes_r2scan", temperature_K=temperature, steps=steps, timestep_fs=1.0,
                 seed=args.seed, engine=EngineConfig(backend="hybrid"),
                 output=Output(str(directory), trajectory_every=100, thermo_every=100, restart_every=None))
        print(run.summary, flush=True)
        run.simulation.close()
        frames = read_frames(directory)  # 1 + steps/100 frames
        second_half = frames[len(frames) // 2:]
        picks = [second_half[round((j + 0.5) * len(second_half) / args.frames - 0.5)] for j in range(args.frames)]
        ratios = []
        for frame in picks:
            frame = frame.copy()
            ratios += swap_ratios(frame, calculator, 41, 73, args.swaps_per_frame, K_B_EV * temperature, rng)
        result = summary(ratios)
        report["temperatures"][f"{temperature:g} K"] = result
        print(f"{temperature:g} K", json.dumps(result), flush=True)
    (out / "nbta_swaps.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
