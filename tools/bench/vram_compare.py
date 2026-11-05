"""Device memory of one MD run: FastMACE or LAMMPS + Symmetrix-XL on the same cell (5 November 2026).

Samples the GPU's used memory (``nvidia-smi --query-gpu=memory.used``, every 0.1 s) before and during the run, so
the peak minus the idle baseline is the run's whole footprint: CUDA context, libraries, model, buffers and caches.
On WSL ``nvidia-smi`` cannot list per-process memory, so keep the desktop idle; the baseline is taken right before.
FastMACE also records PyTorch's own peak (allocated and reserved).

    vram_compare.py fastmace --structure PATH@10x10x10 --checkpoint CK --output out.json
    vram_compare.py symmetrix --structure PATH@10x10x10 --lmp LMP --model-json JSON --jit-arguments "..." \
        --workdir DIR --output out.json

Both run Langevin at 300 K, 1 fs, skin 0.5 Å from Maxwell-Boltzmann velocities (seed 20261105): ``--equilibrate``
steps, then ``--steps`` timed steps.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path


class Sampler:
    """Polls the GPU's used memory (MiB) on a thread."""

    def __init__(self, interval=0.1):
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def used():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.split()[0])

    def _run(self):
        while not self._stop.is_set():
            self.samples.append((time.time(), self.used()))
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def baseline(seconds=3.0):
    values = []
    for _ in range(int(seconds / 0.1)):
        values.append(Sampler.used())
        time.sleep(0.1)
    return statistics.median(values)


def cell(spec):
    import numpy as np
    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    path, repeat = spec.split("@")
    atoms = read(path).repeat(tuple(int(n) for n in repeat.split("x")))
    atoms.set_pbc(True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(20261105))
    Stationary(atoms)
    return atoms


def run_fastmace(args):
    import torch

    import fastmace as fm
    from fastmace.config import NeighbourConfig

    atoms = cell(args.structure)
    base = baseline()
    with Sampler() as sampler:
        loaded = fm.load(str(args.checkpoint), sha256=args.sha256)
        field = fm.ForceField(loaded, atoms, precision="float32", rewrites="auto")
        sim = fm.Simulation(atoms, field, integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0,
                                                                             friction_per_fs=0.002),
                            workload=fm.Workload(max_temperature_K=300.0), output=None, seed=1,
                            neighbours=NeighbourConfig(skin=0.5))
        sim.run(steps=args.equilibrate)
        torch.cuda.synchronize()
        started = time.perf_counter()
        sim.run(steps=args.steps)
        torch.cuda.synchronize()
        step_ms = 1e3 * (time.perf_counter() - started) / args.steps
        torch_peak = dict(allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                          reserved_MiB=torch.cuda.max_memory_reserved() / 2**20)
        sim.close()
    return dict(code="FastMACE", version=fm.__version__, natoms=len(atoms), step_ms=step_ms, torch_peak=torch_peak,
                **_memory(base, sampler))


def run_symmetrix(args):
    from ase.io import write

    atoms = cell(args.structure)
    args.workdir.mkdir(parents=True, exist_ok=False)
    elements = sorted(set(atoms.get_chemical_symbols()))
    write(args.workdir / "data.lmp", atoms, format="lammps-data", specorder=elements, masses=True, velocities=True,
          units="metal", atom_style="atomic")
    damp_ps = 1.0 / 0.002 / 1000.0  # friction 0.002 per fs, as the FastMACE side
    text = f"""units metal
atom_style atomic
atom_modify map yes
newton on
boundary p p p
read_data data.lmp
pair_style symmetrix/mace/float32/kk no_domain_decomposition streamed_edges direct profile speed {args.jit_arguments}
pair_coeff * * {args.model_json} {' '.join(elements)}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
timestep 0.001
fix integrate all nve
fix thermostat all langevin 300.0 300.0 {damp_ps:.6f} 20261105 zero yes
thermo_style custom step temp pe etotal
thermo {args.steps}
run {args.equilibrate}
run {args.steps} pre no
"""
    (args.workdir / "in.md").write_text(text)
    command = [str(args.lmp), "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "neigh", "half", "newton", "on",
               "-in", "in.md", "-log", "log.lammps"]
    base = baseline()
    with Sampler() as sampler:
        proc = subprocess.run(command, cwd=args.workdir, capture_output=True, text=True)
    (args.workdir / "stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    log = (args.workdir / "log.lammps").read_text() if (args.workdir / "log.lammps").exists() else proc.stdout
    loops = [(float(t), int(n)) for t, n in re.findall(r"Loop time of ([0-9.eE+-]+) on \d+ procs for (\d+) steps", log)]
    step_ms = 1e3 * loops[-1][0] / loops[-1][1] if len(loops) == 2 else None
    return dict(code="Symmetrix-XL (LAMMPS, Kokkos CUDA, float32)", natoms=len(atoms), returncode=proc.returncode,
                step_ms=step_ms, **_memory(base, sampler))


def _memory(base, sampler):
    used = [m for _, m in sampler.samples]
    return dict(baseline_MiB=base, peak_used_MiB=max(used), peak_over_baseline_MiB=max(used) - base,
                samples=len(used))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("code", choices=("fastmace", "symmetrix"))
    p.add_argument("--structure", required=True, help="PATH@NXxNYxNZ")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--equilibrate", type=int, default=100)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--sha256", default="d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a")
    p.add_argument("--lmp", type=Path)
    p.add_argument("--model-json", type=Path)
    p.add_argument("--jit-arguments", default="")
    p.add_argument("--workdir", type=Path)
    args = p.parse_args(argv)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    record = (run_fastmace if args.code == "fastmace" else run_symmetrix)(args)
    record.update(gpu=gpu, structure=args.structure, equilibrate=args.equilibrate, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
