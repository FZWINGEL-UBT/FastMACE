"""OPT-T06: the cost of a plan's execution model, reload and convert (today) against a private copy of a template.

    python clone_vs_reload.py CHECKPOINT [--label mh1-r2scan] [--backend cueq] [--dtype float64] [--device cpu]
                              [--repeat 3] [--json OUT]

Per repeat, in one process, alternating: ``loading.load`` as ``ForceField.execution_model`` calls it (with the
pristine hash), then ``compile.private_copy`` of the first load's model (the template), then a calculator shell
around that copy. Prints seconds per variant and the template's parameter and buffer bytes. Also checks that the
copy's energy and forces on a small NaCl cell match the template's bitwise (same device, same dtype).

Not library code.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time

import fastmace  # noqa: F401  (before torch)


def _sync(device):
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _bytes(model):
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


def _evaluate(calculator, atoms):
    atoms = atoms.copy()
    atoms.calc = calculator
    return atoms.get_potential_energy(), atoms.get_forces()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--label", default="mh1-r2scan")
    p.add_argument("--backend", default="cueq")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--json")
    args = p.parse_args()
    import numpy as np
    from ase.build import bulk

    from fastmace.model import compile as compiling
    from fastmace.model import loading

    digest = loading.sha256_of(args.checkpoint)
    options = dict(checkpoint=args.checkpoint, backend=args.backend, dtype=args.dtype, device=args.device,
                   sha256=digest)
    times = {"reload": [], "private_copy": [], "calculator_shell": []}
    started = time.perf_counter()
    template = loading.load(args.label, **options)
    first_load = time.perf_counter() - started
    for _ in range(args.repeat):
        _sync(args.device)
        started = time.perf_counter()
        reloaded = loading.load(args.label, **options)
        _sync(args.device)
        times["reload"].append(time.perf_counter() - started)
        del reloaded
        started = time.perf_counter()
        private = compiling.private_copy(template.model)
        _sync(args.device)
        times["private_copy"].append(time.perf_counter() - started)
        started = time.perf_counter()
        shell = copy.copy(template.calculator)
        shell.models = [private]
        shell.results = {}
        times["calculator_shell"].append(time.perf_counter() - started)

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.rattle(0.05, seed=1)
    e_template, f_template = _evaluate(template.calculator, atoms)
    e_again, f_again = _evaluate(template.calculator, atoms)  # run-to-run noise of the template itself
    e_copy, f_copy = _evaluate(shell, atoms)
    same = bool(e_template == e_copy and np.array_equal(f_template, f_copy))
    out = dict(label=args.label, backend=args.backend, dtype=args.dtype, device=args.device,
               first_load_s=first_load, template_bytes=_bytes(template.model),
               median_s={k: statistics.median(v) for k, v in times.items()}, each_s=times,
               copy_matches_template_bitwise=same, energy_difference=float(e_copy - e_template),
               max_force_difference=float(np.abs(f_copy - f_template).max()),
               template_rerun_bitwise=bool(e_again == e_template and np.array_equal(f_again, f_template)),
               template_rerun_max_force_difference=float(np.abs(f_again - f_template).max()))
    print(json.dumps({k: v for k, v in out.items() if k != "each_s"}, indent=1))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
