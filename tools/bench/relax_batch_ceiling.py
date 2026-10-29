"""Batched-relaxation ceiling (relax-throughput B-T00): the step time of today's engine against atom count.

A batch of K frames of N atoms in one engine would cost at least one step of K·N atoms. This script times
the variable-cell LBFGS step (Energy_Hull's pass A settings: float64 with ``fp64_emulation_mantissa_bits=
"dynamic"``, ``radial_tables="off"``, ``FrechetCell(scalar_pressure_GPa=0)``) on supercells of one glass
frame, so the report can compare K × t(N) with t(K·N): the most a disjoint-union batch could gain over one
reused engine. Each size: build, ``--warm`` steps, then ``--steps`` timed steps at ``fmax=1e-9`` (no
convergence), the ms per step from the relaxation loop. A frame that converges early is re-rattled.

Output: ``<output>/ceiling.jsonl``, one row per size. Run on an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", required=True, help="an extxyz glass frame")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head", default="matpes_r2scan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", default="1x1x1,2x1x1,2x2x1,2x2x2,4x2x2,4x4x2")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--warm", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args(argv)

    import fastmace as fm  # before torch
    import torch
    from ase.io import read

    from fastmace.config import EngineConfig, RewriteConfig
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    args.output.mkdir(parents=True, exist_ok=True)
    model = fm.load("mh1", head=args.head, checkpoint=args.checkpoint)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    engine = EngineConfig(fp64_emulation_mantissa_bits="dynamic") if args.precision == "float64" else None
    record = fm.optimize.LBFGS(cell_filter=fm.optimize.FrechetCell(scalar_pressure_GPa=0.0))
    unit = read(args.frame)
    for text in args.repeats.split(","):
        repeat = tuple(int(value) for value in text.split("x"))
        atoms = unit.repeat(repeat)
        atoms.rattle(stdev=0.02, seed=1)  # off the periodic images' symmetry: a supercell is a new structure
        row = {"repeat": text, "copies": repeat[0] * repeat[1] * repeat[2], "natoms": len(atoms),
               "precision": args.precision, "gpu": gpu}
        try:
            started = time.perf_counter()
            field = ForceField(model, atoms, precision=args.precision, rewrites=RewriteConfig(radial_tables="off"))
            sim = Simulation(atoms, field, optimizer=record, seed=0, engine=engine)
            row["setup_s"] = time.perf_counter() - started
            sim.relax(fmax=1e-9, steps=args.warm)
            _, summary = sim.relax(fmax=1e-9, steps=args.steps)
            row.update({"steps": summary["steps"], "loop_s": summary["seconds"],
                        "ms_per_step": 1000 * summary["seconds"] / max(summary["steps"], 1),
                        "recaptures": summary["recaptures"]})
            if torch.cuda.is_available():
                row["peak_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
            sim.close()
        except Exception as error:  # an out-of-memory size ends the scan
            row["error"] = f"{type(error).__name__}: {error}"[:500]
        with open(args.output / "ceiling.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if "error" in row:
            break


if __name__ == "__main__":
    main()
