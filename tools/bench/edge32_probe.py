"""int32 ``edge_index`` falsifier (capacity bounds, 10 November 2026): run the real MD plan with
``fastmace.neighbors.skin.EDGE_INDEX_INT32`` off or on and save the trajectory's end state, then compare two saves.

- ``run OUT --side off|on``: SrTiO3 at REPEAT, OMAT-0 medium float32, 20 Langevin (NVT) steps and 20 LangevinBAOAB
  (NPT, full cell) steps from the same thermalised start, and two one-off evaluations (energy, forces, stress) at the
  start; saves positions, forces, energy, stress and cell after each,
  plus the plan text and the steady peak reserved.
- ``run ... --tag 2``: the same side again (``edge32-off2.pt``), the run-to-run control: a plan with atomic
  scatters is not bitwise between processes, so off/on is judged against off/off.
- ``run ... --reproducible --tag r``: the same in a reproducible plan (``fm.Workload(reproducible=True)``), where two
  processes give bitwise equal trajectories, so off/on is an exact test of the storage change (no one-off evaluations).
- ``compare A.pt B.pt``: bitwise equality of every saved tensor, else the largest difference.

One side per process (the switch is read when the skin list allocates). Locally through ``tools/ci/local_gpu.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent


def run(out: Path, side: str, repeat: str, steps: int, tag: str = "", reproducible: bool = False,
        model_label: str = "omat0-medium") -> None:
    from fastmace.model.compile import configure_allocator

    configure_allocator(enabled=True)
    import torch

    import fastmace as fm
    from fastmace.neighbors import skin

    skin.EDGE_INDEX_INT32 = side == "on"
    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(TOOLS / "bench"))
    import capacity_scan as scan
    from ab_frontends import thermalize

    name, cell = scan.base_cell("SrTiO3")
    start = cell.repeat(tuple(int(n) for n in repeat.lower().split("x")))
    start.set_pbc(True)
    thermalize(start, 300.0, 20261110)
    model = fm.load(model_label)
    # Nothing touches CUDA before the first Simulation: a reproducible plan then runs in this process (the planner
    # moves it to a child once CUDA is initialised, and a child would not see this process's module constants).
    saved, record = {}, dict(side=side, repeat=repeat, natoms=len(start), reproducible=reproducible, model=model_label)
    # One-off evaluations at the start positions, twice in this process: no trajectory in between, so off/on and
    # off/off2 compare the force field's own reproducibility (the MD end states also carry the trajectories' drift).
    single = None if reproducible else fm.ForceField(model, start.copy(), precision="float32")
    for k in range(0 if reproducible else 2):
        energy, forces, stress = single.evaluate(start, stress=True)
        saved[f"single{k}.energy"] = torch.tensor([energy], dtype=torch.float64)
        saved[f"single{k}.forces"] = torch.from_numpy(forces)
        saved[f"single{k}.stress"] = torch.from_numpy(stress)
    del single
    for label, integrator in (
            ("nvt", fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002)),
            ("npt", fm.integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002,
                                                 cell_mode="full"))):
        atoms = start.copy()
        field = fm.ForceField(model, atoms, precision="float32")
        extra = dict(workload=fm.Workload(max_temperature_K=300.0, reproducible=True)) if reproducible else {}
        sim = fm.Simulation(atoms, field, seed=20261110, integrator=integrator, **extra)
        record[f"{label}_runner"] = sim.plan.runner
        torch.cuda.reset_peak_memory_stats()
        sim.run(steps=steps)
        torch.cuda.synchronize()
        committed = sim.committed()  # public: also in a reproducible plan, whose engine runs in a child process
        for key in ("positions", "forces", "energy", "stress", "cell"):
            value = getattr(committed, key, None)
            if value is not None:
                saved[f"{label}.{key}"] = torch.as_tensor(value).clone()
        import gc

        dtype = set()
        for o in gc.get_objects():
            try:  # dead weak proxies raise ReferenceError on any access
                if type(o) is skin.SkinNeighbourList and getattr(o, "edge_index", None) is not None:
                    dtype.add(str(o.edge_index.dtype))
            except ReferenceError:
                continue
        dtype = sorted(dtype)
        record[label] = dict(peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, edge_index_dtype=dtype,
                             plan=sim.plan.describe().splitlines()[2:4])
        del sim, field
        torch.cuda.empty_cache()
    record["gpu"] = torch.cuda.get_device_name()
    torch.save(saved, out / f"edge32-{side}{tag}.pt")
    print(json.dumps(record))


def compare(a: Path, b: Path) -> dict:
    import torch

    sa, sb = torch.load(a), torch.load(b)
    out = {}
    for key in sorted(set(sa) | set(sb)):
        if key not in sa or key not in sb:
            out[key] = "missing"
            continue
        x, y = sa[key], sb[key]
        out[key] = "bitwise" if torch.equal(x, y) else f"max |d| {float((x.double() - y.double()).abs().max()):.3e}"
    out["all_bitwise"] = all(v == "bitwise" for v in out.values())
    print(json.dumps(out))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out", type=Path)
    r.add_argument("--side", choices=("off", "on"), required=True)
    r.add_argument("--repeat", default="10x10x10")
    r.add_argument("--steps", type=int, default=20)
    r.add_argument("--model", default="omat0-medium", help="label; reproducible plans qualify for MH-1 only")
    r.add_argument("--reproducible", action="store_true",
                   help="plan with fm.Workload(reproducible=True): bitwise across processes, so off/on is exact")
    r.add_argument("--tag", default="", help="file suffix (a repeat of one side: the run-to-run control)")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "run":
        args.out.mkdir(parents=True, exist_ok=True)
        run(args.out, args.side, args.repeat, args.steps, args.tag, args.reproducible, args.model)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
