"""Whether a production build uses SPLINE_W: builds Benchmark_loup's simulation (tools/bench/loup_md.py: SrTiO3,
Langevin 300 K, 1 fs, skin 0.5, from rest) with ``--constant`` set, runs a few steps, and prints every interaction's
``wprime_backward`` (its spline: intervals, certificate, whether its kernels are compiled and in use). Exits 1 when no
interaction uses the spline, so a packed job can stop before timing a build that does not exercise it."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


# Published SHA-256 of the mace-foundations checkpoints this check has been run on (mace_omat_0 release).
MODEL_SHA256 = {
    "omat0-medium": "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",  # as tools/bench/loup_md.py
    "omat0-small": "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5",  # mace-omat-0-small.model
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@4x4x4")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--constant", nargs="*", default=["fastmace.kernels.oeq_streaming.spline_forward:SPLINE_W=True"])
    p.add_argument("--model", choices=sorted(MODEL_SHA256), default="omat0-medium",
                   help="which checkpoint --checkpoint is: its published SHA-256 is checked before unpickling")
    p.add_argument("--sha256", default=None,
                   help="the checkpoint's SHA-256 instead of --model's (any other checkpoint)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--sources", type=Path, default=None, help="write each spline kernel's CUDA source here")
    p.add_argument("--compare-constant", nargs="*", default=None, metavar="MODULE:NAME=VALUE",
                   help="also build a second force field with these constants and compare its forces")
    p.add_argument("--rattle", type=float, default=0.0, help="compare forces on a copy rattled by this many A (seed 1)")
    p.add_argument("--require-compose", action="store_true",
                   help="exit 1 unless a first interaction composes its skip (kernels.triton.compose_skip)")
    args = p.parse_args(argv)

    import fastmace as fm
    from fastmace.config import NeighbourConfig

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ab_rewrites import constant_overrides
    from ase.io import read

    for (module, name), value in constant_overrides(args.constant).items():
        setattr(sys.modules[module], name, value)
    from fastmace.kernels.oeq_streaming.wprime_backward import WPrime

    path, repeat = args.structure.split("@")
    atoms = read(path).repeat(tuple(int(n) for n in repeat.split("x")))
    loaded = fm.load(str(args.checkpoint), sha256=args.sha256 or MODEL_SHA256[args.model])
    field = fm.ForceField(loaded, atoms, precision="float32", rewrites="auto")
    sim = fm.Simulation(atoms, field, integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0,
                                                                         friction_per_fs=0.002),
                        workload=fm.Workload(max_temperature_K=300.0), output=None, seed=1,
                        neighbours=NeighbourConfig(skin=0.5))
    sim.run(steps=args.steps)
    from cuda.bindings import driver

    def resources(kernel):
        attr = driver.CUfunction_attribute
        get = lambda a: int(driver.cuFuncGetAttribute(a, kernel._function)[1])  # noqa: E731
        return dict(registers=get(attr.CU_FUNC_ATTRIBUTE_NUM_REGS),
                    local_bytes=get(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
                    shared_bytes=get(attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES))

    rows = []
    for obj in gc.get_objects():
        if issubclass(type(obj), WPrime):  # type(): a weakproxy in gc's list must not be dereferenced
            spline = obj.spline or {}
            rows.append(dict(repr=obj.extra_repr(), attached=obj.spline is not None,
                             in_use=obj.spline_block is not None,
                             intervals=sum(g[2] for g in spline.get("grid", [])) or None,
                             certificate=(spline.get("certificate") or {}).get("worst"),
                             grid=spline.get("grid"),
                             forward_registers=obj.spline_block[1].registers if obj.spline_block else None,
                             forward_local_bytes=obj.spline_block[1].local_bytes if obj.spline_block else None,
                             forward_threads=obj.spline_block[1].threads if obj.spline_block else None,
                             forward_min_blocks=obj.spline_block[1].min_blocks if obj.spline_block else None,
                             forward_parts=obj.spline_block[1].parts if obj.spline_block else None,
                             forward_stage_edges=getattr(obj.spline_block[1], "stage_edges", None)
                             if obj.spline_block else None,
                             forward_block_order=getattr(obj.spline_block[1], "block_order", None)
                             if obj.spline_block else None,
                             forward_order_bound=(getattr(obj.spline_block[1], "order", None) is not None)
                             if obj.spline_block else None,
                             wprime=resources(obj.spline_block[2]) if obj.spline_block else None,
                             wprime_stage_quads=getattr(obj.spline_block[2], "stage_quads", None)
                             if obj.spline_block else None,
                             wprime_stage_sh=getattr(obj.spline_block[2], "stage_sh", None)
                             if obj.spline_block else None,
                             wprime_stage_edges=getattr(obj.spline_block[2], "stage_edges", None)
                             if obj.spline_block else None,
                             forward_prefetch=getattr(obj.spline_block[1], "prefetch", None)
                             if obj.spline_block else None))
            if args.sources and obj.spline_block:
                args.sources.mkdir(parents=True, exist_ok=True)
                k = len(rows) - 1
                (args.sources / f"spline_forward-{k}.cu").write_text(obj.spline_block[1].source)
                (args.sources / f"wprime-{k}.cu").write_text(obj.spline_block[2].source)
    used = any(r["in_use"] for r in rows)
    from fastmace.rewrites.mh1.modules import FirstInteractionReassociation

    reassociations = [obj for obj in gc.get_objects() if issubclass(type(obj), FirstInteractionReassociation)]
    composed = [getattr(obj, "compose_skip", None) is not None for obj in reassociations]
    joined = [bool(getattr(obj, "joined_layout", False)) for obj in reassociations]
    record = dict(constants=args.constant, structure=args.structure, wprime=rows, spline_in_use=used,
                  compose_skip=composed, compose_skip_in_use=any(composed), joined_layout_in_use=any(joined))
    if args.compare_constant is not None:  # forces of this build against a build with the other constants
        import torch

        import numpy as np

        probe = atoms.copy()
        if args.rattle:
            probe.positions += args.rattle * np.random.default_rng(1).standard_normal(probe.positions.shape)
        energy, forces, _ = field.evaluate(probe)
        energy_again, again, _ = field.evaluate(probe)  # the same build twice: the run-to-run spread (atomics)
        for (module, name), value in constant_overrides(args.compare_constant).items():
            setattr(sys.modules[module], name, value)
        other = fm.ForceField(loaded, atoms, precision="float32", rewrites="auto")
        energy_b, forces_b, _ = other.evaluate(probe)
        a, b = torch.as_tensor(forces, dtype=torch.float64), torch.as_tensor(forces_b, dtype=torch.float64)
        a2 = torch.as_tensor(again, dtype=torch.float64)
        record["compare"] = dict(constants=args.compare_constant, rattle=args.rattle,
                                 max_abs=float((a - b).abs().max()),
                                 max_force=float(a.abs().max()), bitwise=bool(torch.equal(a, b)),
                                 same_build_max_abs=float((a - a2).abs().max()),
                                 energy=float(torch.as_tensor(energy).sum()),
                                 energy_diff=float(torch.as_tensor(energy_b, dtype=torch.float64).sum()
                                                   - torch.as_tensor(energy, dtype=torch.float64).sum()),
                                 same_build_energy_diff=float(torch.as_tensor(energy_again, dtype=torch.float64).sum()
                                                              - torch.as_tensor(energy, dtype=torch.float64).sum()))
    print(json.dumps(record, indent=1, default=str))
    if args.output:
        args.output.write_text(json.dumps(record, indent=1, default=str) + "\n")
    if args.require_compose and not any(composed):
        return 1
    return 0 if used else 1


if __name__ == "__main__":
    raise SystemExit(main())
