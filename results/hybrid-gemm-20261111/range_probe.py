"""HYBRID range check (CPU, 11 November 2026): the largest |A| and |B| of every matrix product in OMAT-0 medium's
energy, forces and stress, against the float16 limit of tf32x3.HYBRID's cross parts (operands below 32,752).

MACE's e3nn model in float32 on the CPU under a TorchDispatchMode that records max |left| and max |right| of every
aten mm / addmm / bmm / baddbmm (forward and the force/stress backward). That is a superset of the products FastMACE
routes to tf32x3 (linears, skips, readouts, radial MLPs and their input gradients). The bar: every max at or below
3,000 (10x margin). Also the model parameters' max |w|.

    python results/hybrid-gemm-20261111/range_probe.py --model M --root R --output FILE [--md-steps 300]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIMIT = 32752.0
BAR = 3000.0
OPS = {torch.ops.aten.mm.default: (0, 1), torch.ops.aten.addmm.default: (1, 2),
       torch.ops.aten.bmm.default: (0, 1), torch.ops.aten.baddbmm.default: (1, 2)}


class Recorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.rows = {}  # (op, shape a, shape b) -> [max |a|, max |b|, calls]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func in OPS:
            i, j = OPS[func]
            a, b = args[i], args[j]
            if a.is_floating_point() and a.numel() and b.numel():
                key = (str(func.__name__), tuple(a.shape), tuple(b.shape))
                row = self.rows.setdefault(key, [0.0, 0.0, 0])
                row[0] = max(row[0], float(a.detach().abs().max()))
                row[1] = max(row[1], float(b.detach().abs().max()))
                row[2] += 1
        return out


def evaluate(calc, atoms):
    atoms = atoms.copy()
    calc.reset()  # no cached results: every case is computed under the recorder
    atoms.calc = calc
    rec = Recorder()
    with rec:
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        s = atoms.get_stress()
    rows = [{"op": k[0], "a": list(k[1]), "b": list(k[2]), "max_a": v[0], "max_b": v[1], "calls": v[2]}
            for k, v in rec.rows.items()]
    rows.sort(key=lambda r: -max(r["max_a"], r["max_b"]))
    return {"atoms": len(atoms), "energy": float(e), "max_force": float(np.abs(f).max()),
            "max_stress": float(np.abs(s).max()), "max_a": max(r["max_a"] for r in rows),
            "max_b": max(r["max_b"] for r in rows), "top": rows[:8], "products": len(rows)}


def closest_pair(atoms):
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--root", required=True, help="directory holding Structures/Benchmark/R2SCAN")
    p.add_argument("--output", required=True)
    p.add_argument("--md-steps", type=int, default=300)
    args = p.parse_args()
    from ase.io import read
    from mace.calculators import MACECalculator
    torch.set_num_threads(max(1, torch.get_num_threads()))
    calc = MACECalculator(model_paths=args.model, device="cpu", default_dtype="float32")
    model = calc.models[0]
    weights = max(float(t.detach().abs().max()) for t in model.parameters() if t.numel())
    buffers = {n: float(t.detach().abs().max()) for n, t in model.named_buffers()
               if t.is_floating_point() and t.numel()}
    bench = Path(args.root) / "Structures" / "Benchmark" / "R2SCAN"
    base = {"SrTiO3": read(ROOT / "results" / "loup-a100-20261103" / "SrTiO3.cif") * (3, 3, 3),
            "NaCl": read(bench / "NaCl.cif") * (3, 3, 3), "NbCl5": read(bench / "NbCl5.cif") * (3, 3, 3)}
    cases = dict(base)
    for name in ("SrTiO3", "NaCl"):
        for label, linear in (("V0.7", 0.7 ** (1 / 3)), ("L0.7", 0.7)):
            a = base[name].copy()
            a.set_cell(a.cell * linear, scale_atoms=True)
            cases[f"{name} {label}"] = a
    a = base["NaCl"].copy()
    d = a.get_distances(0, range(1, len(a)), mic=True, vector=True)
    j = 1 + int(np.argmin(np.linalg.norm(d, axis=1)))
    v = d[j - 1] / np.linalg.norm(d[j - 1])
    a.positions[j] = a.positions[0] + 0.8 * v
    cases["NaCl pair 0.8"] = a
    out = {"model": str(args.model), "limit": LIMIT, "bar": BAR, "max_abs_parameter": weights,
           "max_abs_buffer": max(buffers.values()) if buffers else 0.0,
           "buffers_over_bar": {k: v for k, v in buffers.items() if v > BAR}, "torch": torch.__version__,
           "cases": {}}
    for name, atoms in cases.items():
        t = time.time()
        r = evaluate(calc, atoms)
        r["closest_pair"] = closest_pair(atoms)
        r["seconds"] = round(time.time() - t, 1)
        out["cases"][name] = r
        print(json.dumps({"case": name, **{k: r[k] for k in ("atoms", "closest_pair", "max_a", "max_b", "max_force",
                                                            "seconds")}}), flush=True)
    if args.md_steps:
        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        a = read(bench / "NaCl.cif") * (2, 2, 2)
        a.calc = calc
        MaxwellBoltzmannDistribution(a, temperature_K=2000, rng=np.random.default_rng(0))
        Langevin(a, 2 * units.fs, temperature_K=2000, friction=0.02, rng=np.random.default_rng(1)).run(args.md_steps)
        a.calc = None
        r = evaluate(calc, a)
        r["closest_pair"] = closest_pair(a)
        r["md"] = {"temperature_K": 2000, "steps": args.md_steps, "dt_fs": 2, "start": "NaCl 2x2x2"}
        out["cases"]["NaCl 2000 K frame"] = r
        print(json.dumps({"case": "NaCl 2000 K frame", "max_a": r["max_a"], "max_b": r["max_b"],
                          "closest_pair": r["closest_pair"]}), flush=True)
    worst = max(max(r["max_a"], r["max_b"]) for r in out["cases"].values())
    out["worst_operand"] = worst
    out["margin_to_limit"] = LIMIT / max(worst, out["max_abs_parameter"])
    out["passes_bar"] = worst <= BAR and out["max_abs_parameter"] <= BAR
    print(json.dumps({k: out[k] for k in ("max_abs_parameter", "max_abs_buffer", "worst_operand", "margin_to_limit",
                                          "passes_bar")}))
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
