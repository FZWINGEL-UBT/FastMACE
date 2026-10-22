"""PolarMACE stress against central finite differences of the energy under strain (plan task 3b-T08).

One evaluation path per process (two backend families never share one):

  reference  fastmace.reference.calculators: MACE's calculator, plain cuEq, eager, with the record's
             corrections (the issue 1642 k-space stress fix); float64 is the gate's reference, float32
             its yardstick (float64 energy accumulation);
  stock      MACE 0.3.16's own calculator (plain cuEq, eager), no FastMACE code on the model: its
             energy is the model's definition, its stress the uncorrected one (issue 1642);
  cueq       FastMACE's one-off evaluation plan (fm.ForceField.evaluate) with backend cueq;
  hybrid     the same with backend hybrid (OEq convolutions; float32 only);
  research   1a-T19's polar reference, from the research engine's export (FASTMACE_BASELINE): its
             calculator (common.load_calculator, cuEq, MACE 0.3.16's conversion), polar_gpu enabled on the
             structure's unstrained cell with headroom 0.1 before configure_md_calculator, its one-graph
             strain, energies and stress through ASE (tools/force_stress_gate.py at 096b47b^,
             reference_child). This is the path that measured 4.4e-4 eV/A^3 on NbCl5 3x3x3.

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

PATHS = ("reference", "stock", "cueq", "hybrid", "research")
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


# ------------------------------------------------------------------ the included k-set on the host

# POLAR-1-S's float32 kspace_cutoff (1/A), as the float64 reference reads it.
POLAR_1_S_KSPACE_CUTOFF = 1.8467166423797607


def included_kset(cell, cutoff=POLAR_1_S_KSPACE_CUTOFF):
    """{(n1, n2, n3): (included, |k| / cutoff)} for graph_longrange's per-call k-vector set of ``cell``.

    graph_longrange.kspace.compute_k_vectors_flat in float64: bounds n_i = ceil(cutoff / (b_i . a_i/|a_i|)),
    the half-space parts (0,0,0), (0,0,0<c<n3), (0,0<b<n2,-n3<=c<n3), (0<a<n1,-n2<=b<n2,-n3<=c<n3), then
    |k| <= cutoff. The set depends on the cell only.
    """
    cell = np.asarray(cell, dtype=np.float64)
    reciprocal = 2 * np.pi * np.linalg.inv(cell.T)
    dots = np.einsum("ij,ij->i", reciprocal, cell / np.linalg.norm(cell, axis=1)[:, None])
    n1, n2, n3 = (int(n) for n in np.ceil(cutoff / dots))
    out = {}
    for a in range(0, n1 + 1):
        for b in range(-n2 - 1, n2 + 1):
            for c in range(-n3 - 1, n3 + 1):
                if a == 0 and (b < 0 or (b == 0 and c < 0)):
                    continue
                if a == 0 and b == 0:
                    in_bounds = c == 0 or 0 < c < n3
                elif a == 0:
                    in_bounds = 0 < b < n2 and -n3 <= c < n3
                else:
                    in_bounds = a < n1 and -n2 <= b < n2 and -n3 <= c < n3
                ratio = float(np.linalg.norm(np.array([a, b, c]) @ reciprocal) / cutoff)
                out[(a, b, c)] = (bool(in_bounds and ratio <= 1.0), ratio)
    return out


def kset_flips(cell, strain, cutoff=POLAR_1_S_KSPACE_CUTOFF):
    """The k-vectors whose inclusion differs between ``cell`` and ``cell (I + strain)``: [(n, before, after)]."""
    before = included_kset(cell, cutoff)
    after = included_kset(np.asarray(cell) @ (np.identity(3) + strain), cutoff)
    return [(n, before[n], after.get(n, (False, None))) for n in sorted(set(before) | set(after))
            if before.get(n, (False,))[0] != after.get(n, (False,))[0]]


def replay(args):
    """--replay: the flips of every configuration, component and step, on the host (no model, no GPU)."""
    result = {"tool": "tools/polar_stress_fd.py --replay", "cutoff": args.replay_cutoff, "configs": {}}
    for key, atoms in configurations(args.structure, args.repeat, args.configs).items():
        rows = []
        for h in args.steps:
            for i, j in VOIGT:
                for sign in (1, -1):
                    flips = kset_flips(atoms.cell.array, voigt_strain(i, j, sign * h), args.replay_cutoff)
                    rows.append({"h": h, "component": f"{i}{j}", "sign": sign,
                                 "flips": [{"n": list(n), "before": b, "after": a} for n, b, a in flips]})
                    if flips:
                        print(key, f"h={h:g}", f"{i}{j}", "+" if sign > 0 else "-",
                              [(n, round(b[1], 6), round(a[1], 6)) for n, b, a in flips], flush=True)
        result["configs"][key] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    print("wrote", args.output, flush=True)


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
        true_norm2 = torch.einsum("ni,ni->n", k_vectors, k_vectors)
        if self.mask is not None:
            flips = (self.mask != live).nonzero().flatten()
            # Each flipped k-vector: its integer coefficients, |k| / cutoff, and whether it is inside
            # graph_longrange's per-call bounds (False: the bounds, not the sphere, excluded it).
            within_cutoff = true_norm2 <= cutoff * cutoff
            self.last["flipped"] = int(len(flips))
            self.last["flips"] = [{"n": call.state.integer[i].tolist(),
                                   "k_over_cutoff": float(true_norm2[i].sqrt() / cutoff),
                                   "now_included": bool(live[i]), "within_cutoff": bool(within_cutoff[i])}
                                  for i in flips.tolist()]
        if self.frozen and self.mask is not None:
            mask = self.mask
            self.last["frozen_included"] = int(mask.sum())
            k_norm2 = torch.where(mask, true_norm2, (cutoff * cutoff).to(true_norm2.dtype))
            k_batch = (~mask).to(torch.int64)
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


def research_evaluator(args, atoms):
    """1a-T19's reference_child for a polar model (tools/force_stress_gate.py at 096b47b^), on ``base``'s cell."""
    from fastmace.bench import ab

    export = ab.load_baseline(os.environ["FASTMACE_BASELINE"])
    ab.activate(export)
    import langevin

    ab.check_origin(export, "langevin", langevin.__file__)
    from langevin import workflow as experiment

    md = experiment.parse_args(["--ensemble", "npt", "--backend", "auto", "--model", args.research_label,
                                "--checkpoint", args.checkpoint, "--dtype", "auto"])
    experiment.apply_polar_options(md)
    from langevin import polar_gpu, upstream_mace

    upstream_mace.use_single_graph_strain(md.single_graph_strain)
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MACE_BENCH_DEVICE="cuda:0",
                      MACE_BENCH_DTYPE=args.dtype, MACE_BENCH_CUEQ="1", MACE_BENCH_CHECKPOINT=args.checkpoint)
    import common

    upstream_mace.use_cueq_conv_fusion(None)
    calc = common.load_calculator(args.research_label, dtype=args.dtype, enable_cueq=True)
    model = calc.models[0]
    base = args.research_base
    polar_gpu.enable(model, base.cell.array, headroom=0.1)
    experiment.configure_md_calculator(calc, compute_stress=True)
    if getattr(model.forward, "__func__", None) is not type(model).forward:
        raise RuntimeError("the research reference does not run polar_gpu's corrected forward")

    seen = {"overflowed": False}

    def evaluate(config):
        probe = config.copy()
        probe.calc = calc
        values = float(probe.get_potential_energy()), np.asarray(probe.get_stress(voigt=False), float)
        seen["overflowed"] |= bool(polar_gpu.overflowed(model))  # recorded only: 1a-T19 did not check it
        return values

    def rebase(config):
        """1a-T19 sized the k-space state once, on the unstrained base cell; --research-rebase sizes it per
        configuration instead (as fastmace.reference's rebase)."""
        if args.research_rebase:
            polar_gpu.enable(model, np.asarray(config.cell.array, float), headroom=0.1)

    return evaluate, rebase, {"baseline_commit": export.commit, "single_graph_strain": bool(md.single_graph_strain),
                              "kspace": _kspace(model), "rebase": bool(args.research_rebase), "seen": seen}


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
              "hybrid": fastmace_evaluator, "research": research_evaluator}


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
            flips.append(None if probe is None else probe.last.get("flips"))
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
              "flips", [[None if f is None else len(f) for f in d["flipped"]] for d in detail], flush=True)
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
    p.add_argument("--path", choices=PATHS)
    p.add_argument("--replay", action="store_true",
                   help="No model: the k-vectors that change sides at each step, from the cell alone (host only)")
    p.add_argument("--replay-cutoff", type=float, default=POLAR_1_S_KSPACE_CUTOFF)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--label", default="polar-1-s")
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", nargs=3, type=int, default=(3, 3, 3))
    p.add_argument("--configs", nargs="+", choices=("ideal", "gate"), default=["ideal", "gate"])
    p.add_argument("--steps", nargs="+", type=float, default=list(DEFAULT_STEPS))
    p.add_argument("--frozen-kset", action="store_true", help="Keep each configuration's included k-set fixed")
    p.add_argument("--probe", action="store_true", help="Count included and flipped k-vectors (eager)")
    p.add_argument("--eager", action="store_true", help="FastMACE paths: compile_mode none")
    p.add_argument("--scan", action="store_true", help="E(e_xx) on a fine strain grid")
    p.add_argument("--scan-max", type=float, default=2e-3)
    p.add_argument("--scan-points", type=int, default=81)
    p.add_argument("--research-label", default="polar1-s", help="research path: the research engine's label")
    p.add_argument("--checkpoint", help="research path: the checkpoint (default FASTMACE_MODELS/MACE-POLAR-1-S.model)")
    p.add_argument("--research-rebase", action="store_true",
                   help="research path: size polar_gpu's k-space state on each configuration's cell (1a-T19 did not)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if (args.path is None) == (not args.replay):
        p.error("give --path or --replay")
    if args.path == "hybrid" and args.dtype != "float32":
        p.error("the hybrid backend runs float32 only")
    if args.path in ("stock", "research") and (args.frozen_kset or args.probe):
        p.error(f"the {args.path} path does not use FastMACE's k-vector code; --frozen-kset and --probe need another")
    if args.path == "research" and args.checkpoint is None:
        args.checkpoint = str(Path(os.environ.get("FASTMACE_MODELS", ".")) / "MACE-POLAR-1-S.model")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.replay:
        return replay(args)
    import torch

    torch.set_num_threads(2)
    probe = KSetProbe(args.frozen_kset) if (args.frozen_kset or args.probe) else None
    configs = configurations(args.structure, args.repeat, args.configs)
    args.research_base = configurations(args.structure, args.repeat, ["ideal"])["ideal"]
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
