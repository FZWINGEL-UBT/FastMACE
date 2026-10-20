"""OPT-T06: which execution-model copies give bitwise the same energy and forces as a fresh load.

    python copy_bitwise.py CHECKPOINT [--label mh1-r2scan] [--backend e3nn] [--dtype float64] [--device cpu]

Compares, on a rattled NaCl cell: a second fresh load, private_copy(load), private_copy(private_copy(load)), and
a fresh load's private_copy against another fresh load's private_copy (today's production model per plan). Not
library code.
"""

import argparse
import copy
import json

import fastmace  # noqa: F401  (before torch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--label", default="mh1-r2scan")
    p.add_argument("--backend", default="e3nn")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    import numpy as np
    from ase.build import bulk

    from fastmace.model import compile as compiling
    from fastmace.model import loading

    options = dict(checkpoint=args.checkpoint, backend=args.backend, dtype=args.dtype, device=args.device,
                   sha256=loading.sha256_of(args.checkpoint))
    first = loading.load(args.label, **options)
    second = loading.load(args.label, **options)
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.rattle(0.05, seed=1)

    def run(calculator, model=None):
        if model is not None:
            calculator = copy.copy(calculator)
            calculator.models = [model]
            calculator.results = {}
        a = atoms.copy()
        a.calc = calculator
        return a.get_potential_energy(), a.get_forces()

    reference = run(first.calculator)
    once = compiling.private_copy(first.model)
    variants = {
        "second load": run(second.calculator),
        "private_copy(first)": run(first.calculator, once),
        "private_copy(private_copy(first))": run(first.calculator, compiling.private_copy(once)),
        "private_copy(second)": run(second.calculator, compiling.private_copy(second.model)),
        "first again": run(first.calculator),
    }
    once_ref = variants["private_copy(first)"]
    out = {}
    for name, (energy, forces) in variants.items():
        same = bool(energy == reference[0] and np.array_equal(forces, reference[1]))
        out[name] = dict(vs_first_load=dict(bitwise=same, max_force=float(np.abs(forces - reference[1]).max())),
                         vs_private_copy_first=dict(
                             bitwise=bool(energy == once_ref[0] and np.array_equal(forces, once_ref[1])),
                             max_force=float(np.abs(forces - once_ref[1]).max())))
    print(json.dumps(dict(backend=args.backend, dtype=args.dtype, device=args.device, results=out), indent=1))


if __name__ == "__main__":
    main()
