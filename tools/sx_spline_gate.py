"""CPU only: the force/stress gate's error ratio with interaction 2's radial weights from a cubic spline.

sx-layout lab (7 November 2026, results/sx-layout-lab-20261107). The gate (``tools/force_stress_gate.py``) compares
a candidate's largest energy-per-atom, force and stress errors against the float64 reference with the float32
yardstick's (``fastmace.reference``), on the gate's own rattled and strained configurations (3 per cell mode,
hydrostatic and full, seed 20261015, rattle 0.05 A, strain 0.02), and passes at ratio <= ``selfcheck.RATIO`` (3).

Candidates: the float32 yardstick itself with interaction 2's ``conv_tp_weights`` replaced by an N-knot cubic Hermite
spline of w = (h c) W in r (float32 coefficients in the power basis of t, evaluated in float32, differentiable in r
through t), the spline the sx-layout lab's kernels evaluate. Interaction 1 keeps the network. MACE multiplies the
weights by the cutoff afterwards, so the replacement returns spline / cutoff (the same tensor MACE multiplies by).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import force_stress_gate as gate  # noqa: E402
import sx_spline_error as spl  # noqa: E402
from fastmace.planner import selfcheck  # noqa: E402
from fastmace.reference import calculators  # noqa: E402
from fastmace.rewrites import radial as rt  # noqa: E402


class Stash(torch.nn.Module):
    """The radial embedding, keeping its lengths, cutoff and pair inputs for the spline."""

    def __init__(self, inner, store):
        super().__init__()
        self.inner, self.store = inner, store

    def forward(self, edge_lengths, node_attrs, edge_index, atomic_numbers):
        out = self.inner(edge_lengths, node_attrs, edge_index, atomic_numbers)
        cutoff = out[1] if isinstance(out, tuple) else None
        self.store.update(r=edge_lengths, cutoff=cutoff, node_attrs=node_attrs, edge_index=edge_index)
        return out


class SplineWeights(torch.nn.Module):
    """w from the spline of the stashed lengths, divided by the stashed cutoff (MACE multiplies it back)."""

    def __init__(self, coef32, h, r_lo, columns, store):
        super().__init__()
        self.coef, self.h, self.r_lo, self.columns, self.store = coef32, h, r_lo, columns, store
        self.calls = 0

    def forward(self, edge_feats):
        self.calls += 1
        s = self.store
        r = s["r"].reshape(-1).to(torch.float32)
        species = s["node_attrs"][:, self.columns].argmax(dim=1)
        pair = species[s["edge_index"][0]] * len(self.columns) + species[s["edge_index"][1]]
        u = (r - self.r_lo) * (1.0 / self.h)
        i = torch.clamp(torch.floor(u.detach()).long(), 0, self.coef.shape[1] - 1)
        t = (u - i.to(u.dtype))[:, None]
        c = self.coef[pair, i]
        w = c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * c[:, 3]))
        cutoff = s["cutoff"]
        if cutoff is None:
            return w.to(edge_feats.dtype)
        return (w / cutoff.reshape(-1, 1).to(w.dtype)).to(edge_feats.dtype)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("output", type=Path)
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5", "results/loup-a100-20261103/SrTiO3.cif"])
    p.add_argument("--structures-root", default=str(Path.home() / "fastmace-baseline/1a-T18"))
    p.add_argument("--knots", type=int, nargs="*", default=[32, 256, 2048, 4096])
    p.add_argument("--configs", type=int, default=3)
    args = p.parse_args(argv)
    torch.set_num_threads(8)
    started = time.time()
    plain = torch.load(args.checkpoint, map_location="cpu", weights_only=False).float()
    numbers = [int(z) for z in plain.atomic_numbers]
    r_max = float(plain.r_max)
    report = dict(checkpoint=args.checkpoint, ratio_rule=selfcheck.RATIO, knots=args.knots, structures={})
    for spec in args.structures:
        name, source, repeat = gate.structure_spec(spec, (3, 3, 3), args.structures_root)
        from ase.io import read
        base = read(source).repeat(repeat)
        base.set_pbc(True)
        configs = [c for mode in gate.CELL_MODES
                   for c in gate.make_configurations(base, mode, args.configs, 20261015, 0.02, 0.05)]
        present = sorted(set(int(z) for z in base.numbers))
        columns = [numbers.index(z) for z in present]
        pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))]
        chain64 = rt._GenericChain(plain, plain.interactions[1], columns, torch.float64, torch.device("cpu"))
        ref = calculators.load(args.checkpoint, base, dtype="float64", device="cpu", trusted=True)
        yard = calculators.load(args.checkpoint, base, dtype="float32", device="cpu", trusted=True)
        model = yard.model
        store = {}
        model.radial_embedding = Stash(model.radial_embedding, store)
        original = model.interactions[1].conv_tp_weights
        f64 = [ref.evaluate(c) for c in configs]
        cand = {"float32 (yardstick)": None}
        for n in args.knots:
            coef, h = spl.spline_coefficients(chain64, pairs, rt.R_LO, r_max, n)
            cand[f"spline {n}"] = SplineWeights(coef.to(torch.float32), h, rt.R_LO, columns, store)
        values = {}
        for label, module in cand.items():
            model.interactions[1].conv_tp_weights = original if module is None else module
            values[label] = [yard.evaluate(c) for c in configs]
        model.interactions[1].conv_tp_weights = original
        natoms = len(base)

        def worst(rows):
            e = max(abs(a[0] - b[0]) / natoms * 1000 for a, b in zip(rows, f64))  # meV/atom
            f = max(np.abs(a[1] - b[1]).max() for a, b in zip(rows, f64))
            s = max(np.abs(a[2] - b[2]).max() for a, b in zip(rows, f64))
            return dict(energy_meV_per_atom=e, force_eV_per_A=f, stress_eV_per_A3=s)

        yard_err = {k: float(v) for k, v in worst(values["float32 (yardstick)"]).items()}
        result = dict(natoms=natoms, configs=len(configs), yardstick=yard_err, candidates={})
        for label, rows in values.items():
            err = {k: float(v) for k, v in worst(rows).items()}
            ratios = {k: (err[k] / yard_err[k] if yard_err[k] else None) for k in err}
            passes = {k: bool(err[k] <= max(selfcheck.RATIO * yard_err[k], selfcheck.FLOORS[k])) for k in err}
            calls = getattr(cand[label], "calls", None)
            result["candidates"][label] = dict(errors=err, ratios=ratios, passes=passes, verdict=all(passes.values()),
                                               spline_calls=calls)
            print(name, label, json.dumps({k: round(v, 3) if v is not None else None for k, v in ratios.items()}),
                  "PASS" if all(passes.values()) else "FAIL", flush=True)
        report["structures"][name] = result
    report["seconds"] = time.time() - started
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print("done", round(report["seconds"], 1), "s")


if __name__ == "__main__":
    main()
