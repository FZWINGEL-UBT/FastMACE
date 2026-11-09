"""Whether a production build uses ``rewrites.sc_generated`` and what it changes: builds Benchmark_loup's simulation
(SrTiO3, Langevin 300 K, 1 fs, skin 0.5) with ``--constant`` set, runs a few steps, reports whether product 0 runs
the generated kernels, then compares energy, forces and stress on a rattled copy against a build with
``--compare-constant`` (default: SC_GENERATED off), next to the same build's run-to-run spread. Exits 1 when the
generated kernels are not in use, so a packed job can stop before timing a build that does not exercise them."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ON = "fastmace.rewrites.sc_generated:SC_GENERATED=True"
OFF = "fastmace.rewrites.sc_generated:SC_GENERATED=False"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@4x4x4")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sha256", default="d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
                   help="the checkpoint's SHA-256 (default: OMAT-0 medium, as tools/bench/loup_md.py)")
    p.add_argument("--constant", nargs="*", default=[ON])
    p.add_argument("--compare-constant", nargs="*", default=[OFF])
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--rattle", type=float, default=0.05)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--source", type=Path, default=None, help="write the generated CUDA source here")
    args = p.parse_args(argv)

    import fastmace as fm
    from fastmace.config import NeighbourConfig, RewriteConfig

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ab_rewrites import constant_overrides, rewrite_overrides
    from ase.io import read

    import numpy as np
    import torch

    def set_constants(items):
        for (module, name), value in constant_overrides(items).items():
            setattr(sys.modules[module], name, value)

    set_constants(args.constant)
    path, repeat = args.structure.split("@")
    atoms = read(path).repeat(tuple(int(n) for n in repeat.split("x")))
    loaded = fm.load(str(args.checkpoint), sha256=args.sha256)
    overrides = rewrite_overrides(args.rewrites)
    rewrites = RewriteConfig(**overrides) if overrides else "auto"
    field = fm.ForceField(loaded, atoms, precision="float32", rewrites=rewrites)
    sim = fm.Simulation(atoms, field, integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0,
                                                                         friction_per_fs=0.002),
                        workload=fm.Workload(max_temperature_K=300.0), output=None, seed=1,
                        neighbours=NeighbourConfig(skin=0.5))
    sim.run(steps=args.steps)

    import gc

    from fastmace.kernels.cuda import sc_generated as kernels

    products = [obj for obj in gc.get_objects()
                if issubclass(type(obj), torch.nn.Module) and hasattr(type(obj), "sc_generated_base")]
    marked = [obj for obj in gc.get_objects()
              if issubclass(type(obj), torch.nn.Module) and getattr(obj, "sc_generated", False) is True]
    record = dict(constants=args.constant, rewrites=args.rewrites, structure=args.structure,
                  products_routed=len(products), interactions_marked=len(marked),
                  kernels=len(kernels._REGISTRY), in_use=bool(products and marked))
    if kernels._REGISTRY:
        from cuda.bindings import driver

        one = next(iter(kernels._REGISTRY.values()))
        attr = driver.CUfunction_attribute
        get = lambda f, a: int(driver.cuFuncGetAttribute(a, f)[1])  # noqa: E731
        record.update(forward_registers=get(one._forward, attr.CU_FUNC_ATTRIBUTE_NUM_REGS),
                      forward_local_bytes=get(one._forward, attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
                      backward_registers=get(one._backward, attr.CU_FUNC_ATTRIBUTE_NUM_REGS),
                      backward_local_bytes=get(one._backward, attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES))
        if args.source:
            args.source.write_text(one.source)

    if args.compare_constant is not None:
        probe = atoms.copy()
        if args.rattle:
            probe.positions += args.rattle * np.random.default_rng(1).standard_normal(probe.positions.shape)
        first = field.evaluate(probe)
        again = field.evaluate(probe)
        set_constants(args.compare_constant)
        other = fm.ForceField(loaded, atoms, precision="float32", rewrites=rewrites)
        third = other.evaluate(probe)

        def as64(x):
            return torch.as_tensor(np.asarray(x), dtype=torch.float64)

        def diff(i):
            a, a2, b = as64(first[i]), as64(again[i]), as64(third[i])
            return dict(max_abs=float((a - b).abs().max()), same_build_max_abs=float((a - a2).abs().max()),
                        max_value=float(a.abs().max()), bitwise=bool(torch.equal(a, b)))

        record["compare"] = dict(constants=args.compare_constant, rattle=args.rattle, energy=diff(0), forces=diff(1))
        if first[2] is not None and third[2] is not None:
            record["compare"]["stress"] = diff(2)
    text = json.dumps(record, indent=1, default=str)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return 0 if record["in_use"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
