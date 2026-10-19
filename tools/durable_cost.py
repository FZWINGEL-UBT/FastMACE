"""Cost of durable restart points (plan task 2-T10): fsync on versus off, in alternating blocks.

Each repeat writes one output chunk with ``io.output.ChunkedOutput`` (untimed: ``--frames``
frames and rows of ``--atoms`` atoms, a 10,000-step chunk at the default cadences), then times
the restart point: ``restart_point()`` (close and fsync both chunk files and the directory) and a
restart-sized ``io.durable.write_atomically``. "off" replaces ``os.fsync`` with a no-op, which is
the 1b behaviour. The chunk is fsynced right after it was written, with all of it still dirty, so
the time is an upper bound: in a run the kernel writes most of a chunk back during its 10,000 steps.

    python tools/durable_cost.py <directory> --out results/<run-id>/durable_cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

import fastmace  # noqa: F401  (before torch, SPEC §14)


def chunk_values(rng, natoms):
    return dict(positions=rng.random((natoms, 3)) * 20.0, momenta=rng.normal(size=(natoms, 3)),
                forces=rng.normal(size=(natoms, 3)), energy=np.array([-3.5 * natoms]),
                kinetic=np.array([0.04 * natoms]))


def one_restart_point(directory, atoms, header, values, frames, restart_bytes):
    from fastmace.io import durable
    from fastmace.io.output import ChunkedOutput

    directory.mkdir(parents=True)
    writer = ChunkedOutput(directory, atoms, header)
    for step in range(frames):
        writer.write(values, sample=(step * 20, step * 0.01, "md", 300.0), frame=True)
    start = time.perf_counter()
    writer.restart_point()
    durable.write_atomically(directory / "restart.npz", lambda handle: handle.write(restart_bytes))
    elapsed = time.perf_counter() - start
    writer.close()
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="scratch directory on the file system to measure")
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--blocks", type=int, default=6, help="alternating on/off blocks of each")
    parser.add_argument("--repeats", type=int, default=5, help="restart points per block")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from ase import Atoms

    rng = np.random.default_rng(0)
    atoms = Atoms("NaCl" * (args.atoms // 2), positions=np.zeros((args.atoms, 3)), cell=[20.0] * 3, pbc=True)
    values = chunk_values(rng, args.atoms)
    header = ["step", "time_ps", "phase", "target_K", "temperature_K", "epot_eV_atom", "ekin_eV_atom", "etot_eV_atom"]
    restart_bytes = rng.bytes(args.atoms * 3 * 8 * 6 + 64_000)  # six N x 3 float64 arrays plus records
    real_fsync = os.fsync
    times = {"on": [], "off": []}
    args.directory.mkdir(parents=True, exist_ok=True)
    run = 0
    for block in range(2 * args.blocks):
        mode = ("on", "off")[block % 2]
        os.fsync = real_fsync if mode == "on" else (lambda fd: None)
        try:
            for _ in range(args.repeats):
                path = args.directory / f"r{run:03d}"
                times[mode].append(one_restart_point(path, atoms, header, values, args.frames, restart_bytes))
                shutil.rmtree(path)
                run += 1
        finally:
            os.fsync = real_fsync
    chunk_bytes = None
    probe = args.directory / "probe"
    one_restart_point(probe, atoms, header, values, args.frames, restart_bytes)
    chunk_bytes = sum(p.stat().st_size for p in probe.iterdir() if p.name.endswith((".0.csv", ".0.traj")))
    shutil.rmtree(probe)

    def summary(values):
        return dict(median_ms=1e3 * statistics.median(values), min_ms=1e3 * min(values), max_ms=1e3 * max(values),
                    n=len(values))

    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repository), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    filesystem = subprocess.run(["stat", "-f", "-c", "%T", str(args.directory)], capture_output=True,
                                text=True).stdout.strip() if os.name == "posix" else None
    result = dict(machine=platform.node(), filesystem=filesystem, directory=str(args.directory),
                  fastmace_commit=commit, atoms=args.atoms, frames=args.frames, chunk_bytes=chunk_bytes,
                  restart_bytes=len(restart_bytes), on=summary(times["on"]), off=summary(times["off"]),
                  extra_median_ms=1e3 * (statistics.median(times["on"]) - statistics.median(times["off"])),
                  raw_s=times)
    print(json.dumps({k: v for k, v in result.items() if k != "raw_s"}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
