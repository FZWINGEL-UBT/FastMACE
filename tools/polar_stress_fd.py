"""PolarMACE stress against central finite differences of the energy under strain (plan task 3b-T08).

One evaluation path per process (two backend families never share one):

  reference  fastmace.reference.calculators: MACE's calculator, plain cuEq, eager, with the record's
             corrections (the issue 1642 k-space stress fix); float64 is the gate's reference, float32
             its yardstick (float64 energy accumulation);
  stock      MACE 0.3.16's own calculator (plain cuEq, eager), no FastMACE code on the model: its
             energy is the model's definition, its stress the uncorrected one (issue 1642);
  cueq       FastMACE's one-off evaluation plan (fm.ForceField.evaluate) with backend cueq;
  hybrid     the same with backend hybrid (OEq convolutions; float32 only).

For each configuration it records the analytic stress and, for every step ``h``, the stress by
central differences: diagonal ii: strain e_ii = +-h, shear ij: e_ij = e_ji = +-h/2, sigma =
(E(+) - E(-)) / (2 h V), as tools/force_stress_gate.py and the reference's self-test.

``--frozen-kset`` is the control for k-vectors crossing the k-space cutoff: it replaces
fastmace.longrange.polar.compute_k_vectors_fixed, so the set of included k-vectors (graph_longrange's
bounds and its |k| <= cutoff test) stays the one of the configuration's own cell, while every
included k-vector still moves with the strain. The analytic stress is unchanged by it (the included
set is not differentiable), so finite differences of the frozen energy test the stress formula
without the cutoff's steps. FastMACE paths run eager with it (compile_mode none). ``--scan`` records
E(e_xx) at ``--scan-points`` strains in +-``--scan-max``, with the number of included k-vectors.

Usage (the local GPU through tools/ci/local_gpu.py run; docs/validation/3b-T08.md has the commands):

  python tools/polar_stress_fd.py --path reference --dtype float64 --structure NbCl5 --repeat 3 3 3 \\
      --output results/RUN/reference-f64-NbCl5.json [--frozen-kset] [--scan]

Benchmark names resolve under FASTMACE_STRUCTURES (the baseline export), labels under FASTMACE_MODELS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
import numpy as np

PATHS = ("reference", "stock", "cueq", "hybrid")
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
DEFAULT_STEPS = (3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
# The gate's configuration draw (tools/force_stress_gate.py defaults): first hydrostatic configuration.
GATE_SEED, GATE_STRAIN, GATE_RATTLE_A = 20261015, 0.02, 0.05


def _gate_module():
    import importlib.util

    location = Path(__file__).with_name("force_stress_gate.py")
    spec = importlib.util.spec_from_file_location("_force_stress_gate", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configurations(name, repeat, which):
    """{key: Atoms}: "ideal" (the relaxed benchmark supercell) and "gate" (the gate's hydrostatic-0 draw)."""
    from ase.io import read

    gate = _gate_module()
    _, source, repeat = gate.structure_spec(name, repeat, os.environ.get("FASTMACE_STRUCTURES", "."))
    base = read(source).repeat(repeat)
    base.set_pbc(True)
    out = {}
    if "ideal" in which:
        out["ideal"] = base.copy()
    if "gate" in which:
        out["gate"] = gate.make_configurations(base, "hydrostatic", 1, GATE_SEED, GATE_STRAIN, GATE_RATTLE_A)[0]
    return out


def strained(atoms, strain):
    copy = atoms.copy()
    copy.set_cell(atoms.cell.array @ (np.identity(3) + strain), scale_atoms=True)
    return copy


def voigt_strain(i, j, h):
    e = np.zeros((3, 3))
    if i == j:
        e[i, i] = h
    else:
        e[i, j] = e[j, i] = h / 2
    return e


# ------------------------------------------------------------------ the frozen k-set control

class KSetProbe:
    """Replaces polar.compute_k_vectors_fixed: counts the included k-vectors per call and, when frozen,
    keeps the included set of the first call after ``freeze_next()`` (eager only)."""

    def __init__(self, frozen):
        from fastmace.longrange import polar

        self.polar = polar
        self.original = polar.compute_k_vectors_fixed
        self.frozen = frozen
        self.mask = None
        self.pending = False
        self.last = {}
        polar.compute_k_vectors_fixed = self

    def freeze_next(self):
        self.mask, self.pending = None, True

    def __call__(self, call, cutoff, cell_vectors, r_cell_vectors):
        import torch

        k_vectors, k_norm2, k_batch, k0_mask = self.original(call, cutoff, cell_vectors, r_cell_vectors)
        live = k_batch == 0
        if self.pending:
            self.mask, self.pending = live.detach().clone(), False
        self.last = {"included": int(live.sum())}
        if self.frozen and self.mask is not None:
            mask = self.mask
            self.last.update(flipped=int((mask != live).sum()), frozen_included=int(mask.sum()))
            true_norm2 = torch.einsum("ni,ni->n", k_vectors, k_vectors)
            k_norm2 = torch.where(mask, true_norm2, (cutoff * cutoff).to(true_norm2.dtype))
            k_batch = (~mask).to(torch.int64)
        elif self.mask is not None:
            self.last["flipped"] = int((self.mask != live).sum())
        return k_vectors, k_norm2, k_batch, k0_mask


# ------------------------------------------------------------------ evaluators: atoms -> (energy, stress 3x3)

def reference_evaluator(args, atoms):
    from fastmace.reference import calculators
    from fastmace.reference.records import ReferencePolicyError

    reference = calculators.load(args.label, atoms, dtype=args.dtype, device="cuda:0")

    def evaluate(config):
        try:
            energy, _, stress = reference.evaluate(config)
        except ReferencePolicyError as error:
            if "overflowed" not in str(error):
                raise
            reference.rebase(config)
            energy, _, stress = reference.evaluate(config)
        return energy, stress

    def rebase(config):
        reference.rebase(config)

    return evaluate, rebase, {"reference": json.loads(json.dumps(reference.info(), default=str)),
                              "kspace": _kspace(reference.model)}


def stock_evaluator(args, atoms):
    from fastmace.model import loading
    from fastmace.reference import records

    stock = records.STOCK
    loaded = loading.load(args.label, backend=stock["backend"], dtype=args.dtype, device="cuda:0",
                          cueq_conv_fusion=stock["cueq_conv_fusion"])
    calculator = loaded.calculator

    def evaluate(config):
        config = config.copy()
        config.calc = calculator
        return float(config.get_potential_energy()), np.asarray(config.get_stress(voigt=False), float)

    return evaluate, lambda config: None, {"model": {"sha256": loaded.sha256}, "kspace": _kspace(loaded.model)}


def fastmace_evaluator(args, atoms):
    from fastmace import config as fm_config
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal

    loaded = loading.load_pristine(args.label)
    field = ForceField(loaded, atoms, precision=args.dtype)
    engine = dict(backend=args.path)
    if args.frozen_kset or args.probe or args.eager:
        engine["compile_mode"] = "none"
    planner = minimal.MinimalPlanner(engine=fm_config.EngineConfig(**engine))
    plan = planner.plan_evaluation(field, atoms)
    force_field = planner.build_evaluation(plan, field, atoms)

    def evaluate(config):
        energy, _, stress = force_field.evaluate(config, stress=True)
        return energy, np.asarray(stress, float)

    def rebase(config):
        force_field.reset_polar(np.asarray(config.cell.array, float))

    return evaluate, rebase, {"plan": plan.describe(), "polar": force_field.polar_info,
                              "kspace": _kspace(force_field._polar_model)}


def _kspace(model):
    """The k-space cutoff and the GTO widths (what a k-vector at the cutoff still weighs)."""
    info = {}
    try:
        info["kspace_cutoff"] = float(model.kspace_cutoff)
    except Exception as error:  # noqa: BLE001  (recorded, not needed)
        info["error"] = repr(error)
    for name, module in model.named_modules():
        sigma2 = getattr(module, "sigma2", None)
        if sigma2 is not None and "sigma2" not in info:
            info["sigma2"] = np.asarray(sigma2.detach().double().cpu()).reshape(-1).tolist()
            info["sigma2_module"] = name
    return info


EVALUATORS = {"reference": reference_evaluator, "stock": stock_evaluator, "cueq": fastmace_evaluator,
              "hybrid": fastmace_evaluator}


# ------------------------------------------------------------------ measurement

def fd_stress(evaluate, atoms, h, probe=None):
    """3x3 central-difference stress, the energies and the k-vectors that flip at each strain."""
    volume = atoms.get_volume()
    stress = np.zeros((3, 3))
    detail = []
    for i, j in VOIGT:
        energies, flips = [], []
        for sign in (1, -1):
            energies.append(evaluate(strained(atoms, voigt_strain(i, j, sign * h)))[0])
            flips.append(None if probe is None else probe.last.get("flipped"))
        stress[i, j] = stress[j, i] = (energies[0] - energies[1]) / (2 * h * volume)
        detail.append({"component": f"{i}{j}", "e_plus": energies[0], "e_minus": energies[1], "flipped": flips})
    return stress, detail


def measure(args, evaluate, rebase, probe, key, atoms):
    rebase(atoms)
    if probe is not None:
        probe.freeze_next()
    started = time.perf_counter()
    energy, analytic = evaluate(atoms)
    first_s = time.perf_counter() - started
    entry = {"natoms": len(atoms), "volume_A3": atoms.get_volume(), "cell": atoms.cell.array.tolist(),
             "energy_eV": energy, "analytic": analytic.tolist(), "first_evaluation_s": first_s,
             "kset": dict(probe.last) if probe is not None else None, "steps": []}
    print(key, "E", f"{energy:.10f}", "analytic diag", np.diag(analytic), flush=True)
    for h in args.steps:
        fd, detail = fd_stress(evaluate, atoms, h, probe)
        delta = fd - analytic
        entry["steps"].append({"h": h, "fd": fd.tolist(), "max_abs": float(np.abs(delta).max()),
                               "max_rel": float(np.abs(delta).max() / max(np.abs(analytic).max(), 1e-300)),
                               "detail": detail})
        print(key, f"h={h:g}", f"max|fd-analytic|={np.abs(delta).max():.3e}",
              "flips", [d["flipped"] for d in detail], flush=True)
    if args.scan:
        scan = []
        for e in np.linspace(-args.scan_max, args.scan_max, args.scan_points):
            energy_e, stress_e = evaluate(strained(atoms, voigt_strain(0, 0, float(e))))
            scan.append({"e_xx": float(e), "energy_eV": energy_e, "stress_xx": float(stress_e[0, 0]),
                         "kset": dict(probe.last) if probe is not None else None})
        entry["scan_xx"] = scan
        print(key, "scan done", flush=True)
    return entry


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", choices=PATHS, required=True)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--label", default="polar-1-s")
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", nargs=3, type=int, default=(3, 3, 3))
    p.add_argument("--configs", nargs="+", choices=("ideal", "gate"), default=["ideal", "gate"])
    p.add_argument("--steps", nargs="+", type=float, default=list(DEFAULT_STEPS))
    p.add_argument("--frozen-kset", action="store_true", help="Keep each configuration's included k-set fixed")
    p.add_argument("--probe", action="store_true", help="Count included and flipped k-vectors (FastMACE paths run eager)")
    p.add_argument("--eager", action="store_true", help="FastMACE paths: compile_mode none")
    p.add_argument("--scan", action="store_true", help="E(e_xx) on a fine strain grid")
    p.add_argument("--scan-max", type=float, default=2e-3)
    p.add_argument("--scan-points", type=int, default=81)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.path == "hybrid" and args.dtype != "float32":
        p.error("the hybrid backend runs float32 only")
    if args.path == "stock" and args.frozen_kset:
        p.error("the stock path does not use FastMACE's k-vector code; --frozen-kset needs another path")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch

    torch.set_num_threads(2)
    probe = KSetProbe(args.frozen_kset) if (args.frozen_kset or args.probe) else None
    configs = configurations(args.structure, args.repeat, args.configs)
    first = next(iter(configs.values()))
    started = time.perf_counter()
    evaluate, rebase, info = EVALUATORS[args.path](args, first)
    result = {"tool": "tools/polar_stress_fd.py", "argv": sys.argv[1:], "path": args.path, "dtype": args.dtype,
              "label": args.label, "structure": args.structure, "repeat": list(args.repeat),
              "frozen_kset": args.frozen_kset, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "load_s": time.perf_counter() - started, "info": info,
              "commit": os.environ.get("FASTMACE_SOURCE_COMMIT"), "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "python": sys.executable, "configs": {}}
    for key, atoms in configs.items():
        result["configs"][key] = measure(args, evaluate, rebase, probe, key, atoms)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1, default=str))
    result["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.output.write_text(json.dumps(result, indent=1, default=str))
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
