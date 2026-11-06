"""Symmetrix-XL under LAMMPS with the Benchmark_loup protocol, one fresh process (5 November 2026).

The counterpart of ``tools/bench/loup_md.py``: MACE-OMAT-0 medium (the Symmetrix JSON export), the structure
repeated, from rest (no initial velocities), skin --skin, 1 fs, NVE or Langevin 300 K (``fix langevin`` with damp
1 / friction, friction 0.002 per fs, ``zero yes``), float32 (``symmetrix/mace/float32/kk``, ``streamed_edges direct
profile speed`` and the JIT device artifact's arguments). ``run --warmup`` untimed, then ``run --steps pre no``;
the time per step is LAMMPS's "Loop time" of the second run. Writes one JSON record.

    sx_loup.py --lmp LMP --model-json JSON --jit-arguments "..." --structure PATH@10x10x10 --ensemble langevin \
        --workdir DIR --output out.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmp", type=Path, required=True)
    p.add_argument("--model-json", type=Path, required=True)
    p.add_argument("--jit-arguments", default="")
    p.add_argument("--structure", required=True, help="PATH@NXxNYxNZ")
    p.add_argument("--ensemble", choices=("nve", "langevin"), default="nve")
    p.add_argument("--skin", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wrap", default="", help="a command prefix for lmp, e.g. 'nsys profile -t cuda -o FILE' "
                                              "(6 November 2026, the Symmetrix-XL 0.1.2 kernel profile)")
    args = p.parse_args(argv)

    from ase.io import read, write

    path, repeat = args.structure.split("@")
    atoms = read(path).repeat(tuple(int(n) for n in repeat.split("x")))
    atoms.set_pbc(True)
    elements = sorted(set(atoms.get_chemical_symbols()))
    args.workdir.mkdir(parents=True, exist_ok=False)
    write(args.workdir / "data.lmp", atoms, format="lammps-data", specorder=elements, masses=True, units="metal",
          atom_style="atomic")
    thermostat = ("fix thermostat all langevin 300.0 300.0 0.500000 1 zero yes\n"
                  if args.ensemble == "langevin" else "")
    text = f"""units metal
atom_style atomic
atom_modify map yes
newton on
boundary p p p
read_data data.lmp
pair_style symmetrix/mace/float32/kk no_domain_decomposition streamed_edges direct profile speed {args.jit_arguments}
pair_coeff * * {args.model_json} {' '.join(elements)}
neighbor {args.skin} bin
neigh_modify every 1 delay 0 check yes
timestep 0.001
fix integrate all nve
{thermostat}thermo_style custom step temp pe etotal
thermo {max(args.steps, 1)}
run {args.warmup}
run {args.steps} pre no
"""
    (args.workdir / "in.md").write_text(text)
    command = [str(args.lmp), "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "neigh", "half", "newton", "on",
               "-in", "in.md", "-log", "log.lammps"]
    started = time.time()
    command = args.wrap.split() + command
    proc = subprocess.run(command, cwd=args.workdir, capture_output=True, text=True)
    wall = time.time() - started
    (args.workdir / "stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    log_path = args.workdir / "log.lammps"
    log = log_path.read_text() if log_path.exists() else proc.stdout
    loops = [(float(t), int(n)) for t, n in re.findall(r"Loop time of ([0-9.eE+-]+) on \d+ procs for (\d+) steps", log)]
    step_ms = 1e3 * loops[-1][0] / loops[-1][1] if len(loops) == 2 and loops[-1][1] else None
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    record = dict(tool="sx_loup", code="Symmetrix-XL (LAMMPS, Kokkos CUDA, float32)", gpu=gpu,
                  structure=args.structure, atoms=len(atoms), ensemble=args.ensemble, skin=args.skin,
                  warmup=args.warmup, steps=args.steps, returncode=proc.returncode, wall_s=wall, ms_per_step=step_ms,
                  us_per_atom_step=1e3 * step_ms / len(atoms) if step_ms else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record))
    return 0 if proc.returncode == 0 and step_ms else 1


if __name__ == "__main__":
    raise SystemExit(main())
