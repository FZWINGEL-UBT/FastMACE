"""OPT-T06 on the GPU: a plan's model from a template copy against one from a fresh reload, bitwise.

    python gpu_copy_vs_load.py --checkpoint PATH [--label mh1-r2scan] --backend hybrid|cueq --dtype float32|float64
                               [--engine-steps 10] [--json OUT]

One backend per process (SPEC §8.7). Without --label the checkpoint is loaded as a trusted path (the tiny test model).

- **Model level:** ``ForceField.execution_model`` with the cache on (first plan: the load; second: a template copy)
  and with it off (a fresh reload, today's later plan). Each is evaluated through a private copy, as a force field
  does, on rattled NaCl 2x2x2; the first is evaluated twice to show the run-to-run noise.
- **Engine level** (``--engine-steps`` > 0): a second ``Simulation`` on a force field with the cache on (a template
  copy) and on one with it off (a reload). Their initial forces and positions after the steps are compared.
- Also the time of each execution model (untimed-quality: no idle requirement).

Not library code; ``tests/gpu/test_template_cache_gpu.py`` runs it on the tiny model.
"""

from __future__ import annotations

import argparse
import json
import time
import types

import fastmace  # noqa: F401  (before torch)


def _atoms():
    from ase.build import bulk

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=3)
    return atoms


def _evaluate(loaded, atoms):
    from fastmace.model import templates

    calculator = templates.private(loaded).calculator
    a = atoms.copy()
    a.calc = calculator
    return float(a.get_potential_energy()), a.get_forces().copy()


def _compare(one, two):
    import numpy as np

    return dict(bitwise=bool(one[0] == two[0] and np.array_equal(one[1], two[1])),
                energy=abs(one[0] - two[0]), max_force=float(np.abs(one[1] - two[1]).max()))


def _arrays_compare(one, two):
    import numpy as np

    return dict(bitwise=bool(np.array_equal(one, two)), max=float(np.abs(one - two).max()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label")
    p.add_argument("--backend", choices=("hybrid", "cueq"), required=True)
    p.add_argument("--dtype", choices=("float32", "float64"), required=True)
    p.add_argument("--engine-steps", type=int, default=10)
    p.add_argument("--json")
    args = p.parse_args(argv)
    from fastmace.config import EngineConfig
    from fastmace.model import compile as compiling
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField

    compiling.configure_allocator()  # before anything touches CUDA, as a planner's construction does
    if args.label:
        pristine = loading.load_pristine(args.label, checkpoint=args.checkpoint,
                                         sha256=loading.sha256_of(args.checkpoint))
    else:
        pristine = loading.load_pristine(args.checkpoint, trusted=True)
    import torch

    atoms = _atoms()

    def plan(cache):
        return types.SimpleNamespace(engine=EngineConfig(backend=args.backend, device="cuda:0",
                                                         model_template_cache=cache),
                                     precision=types.SimpleNamespace(dtype=args.dtype), allowances=())

    times = {}

    def timed(name, call):
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = call()
        torch.cuda.synchronize()
        times[name] = time.perf_counter() - started
        return value

    on = ForceField(pristine, precision=args.dtype)
    off = ForceField(pristine, precision=args.dtype)
    first = timed("first plan (load, plus template copy)", lambda: on.execution_model(plan(True)))
    copied = timed("later plan, cache on (template copy)", lambda: on.execution_model(plan(True)))
    reloaded = timed("later plan, cache off (reload)", lambda: off.execution_model(plan(False)))
    first_out = _evaluate(first, atoms)
    out = dict(backend=args.backend, dtype=args.dtype, label=args.label, gpu=torch.cuda.get_device_name(0),
               torch=torch.__version__, execution_model_s=times)
    out["model"] = {
        "first plan rerun (noise)": _compare(first_out, _evaluate(first, atoms)),
        "template copy vs reload": _compare(_evaluate(copied, atoms), _evaluate(reloaded, atoms)),
        "template copy vs first plan": _compare(_evaluate(copied, atoms), first_out),
        "reload vs first plan": _compare(_evaluate(reloaded, atoms), first_out),
    }
    del first, copied, reloaded
    allocated = torch.cuda.memory_allocated()
    out["template_bytes"] = on.release_templates()
    torch.cuda.synchronize()
    out["allocated_freed_by_release"] = allocated - torch.cuda.memory_allocated()
    if args.engine_steps > 0:
        from ase import units

        from fastmace.integrators import Langevin
        from fastmace.simulation import Simulation

        def second_simulation(cache):
            engine = EngineConfig(backend=args.backend, model_template_cache=cache)
            ff = ForceField(pristine, atoms, precision=args.dtype)
            integrator = Langevin(timestep_fs=0.5, temperature_K=300.0, friction_per_fs=0.02 * units.fs)
            results = []
            for _ in range(2):
                sim = Simulation(atoms.copy(), ff, integrator=integrator, seed=1, engine=engine)
                forces = sim._engine.state.forces.detach().cpu().numpy().copy()
                sim.run(steps=args.engine_steps)
                positions = sim._engine.state.positions.detach().cpu().numpy().copy()
                results.append((forces, positions, sim._plan.engine.backend))
                del sim
            return results, ff.template_count()

        with_cache, count_on = second_simulation(True)
        without, count_off = second_simulation(False)
        out["engine"] = dict(
            backend=with_cache[1][2], templates_kept=dict(on=count_on, off=count_off), steps=args.engine_steps,
            first_plans=dict(forces=_arrays_compare(with_cache[0][0], without[0][0]),
                             positions=_arrays_compare(with_cache[0][1], without[0][1])),
            second_plans_copy_vs_reload=dict(forces=_arrays_compare(with_cache[1][0], without[1][0]),
                                             positions=_arrays_compare(with_cache[1][1], without[1][1])))
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(out, handle, indent=1)
    return out


if __name__ == "__main__":
    main()
