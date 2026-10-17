"""Force and stress gate: the production path against the float64 calculator, by today's rule (SPEC §12.2, §12.3).

For rattled and strained supercells it evaluates, on the same configuration,

  f64         cuEquivariance float64 ASE calculator (reference)
  ase_f32     cuEquivariance float32 ASE calculator (ordinary float32 rounding, the yardstick)
  production  the production force field (hybrid FP32 by default, every default optimisation,
              compiled, padded edges as in the CUDA-graph path)

Rule: for forces and for stress, the production path's maximum error against float64 is at most
GATE_RATIO times that of ase_f32, or at most the absolute floor of SPEC §12.3 (1e-5 eV/A for forces,
1e-7 eV/A^3 for stress) where the float32 yardstick happens to be nearly exact. Energy per atom is
reported, not gated: the float32 yardstick with float64 accumulation of §12.1 does not exist yet.

Production paths (--engine):
  baseline  the research engine's force field from an exported baseline (--baseline DIR, with a
            COMMIT file), built as its main.py builds it
  fastmace  FastMACE's force field; refused until its ForceField exists (plan task 1a-T11)

Run it with the hybrid environment through the export's studies/hybrid_oeq/scripts/oeq_no_nvcc.sh
(or pass --backend cueq), and PYTHONPATH=<FastMACE checkout>. Output: <output>/metadata.json.

Copied from the research engine's tools/validation/npt_stress_reference.py (with the ratio summary of
tools/validation/roundoff_reference.py) and refactored: the rule has floors, the production path is
chosen by --engine, and the research engine is loaded only from the export given by path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GATE_RATIO = 2.0
# SPEC §12.3: absolute floors of the rule, eV/A for the maximum force component and eV/A^3 for the
# maximum stress component (decision 7).
FLOORS = {"forces": 1e-5, "stress": 1e-7}
QUANTITIES = ("forces", "stress")


def errors(reference, candidate):
    """Maximum and RMS absolute error of candidate against reference, and the reference's largest value."""
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    delta = np.abs(candidate - reference)
    return dict(max_abs=float(delta.max()), rms=float(np.sqrt((delta ** 2).mean())),
                reference_max_abs=float(np.abs(reference).max()))


def quantity_verdict(quantity, production_max_abs, ase_f32_max_abs, ratio=GATE_RATIO):
    """Today's rule for one quantity: production <= max(ratio x float32 calculator, floor)."""
    limit = max(ratio * ase_f32_max_abs, FLOORS[quantity])
    return dict(production_max_abs=production_max_abs, ase_f32_max_abs=ase_f32_max_abs,
                ratio=production_max_abs / ase_f32_max_abs if ase_f32_max_abs else None,
                floor=FLOORS[quantity], limit=limit, passed=bool(production_max_abs <= limit))


def _nonfinite(configs):
    """Every non-finite error value in ``configs``, as "configuration index/path/quantity" strings."""
    bad = []
    for index, config in enumerate(configs):
        for path in ("production", "ase_f32"):
            for quantity in QUANTITIES:
                if not np.isfinite(config[path][quantity]["max_abs"]):
                    bad.append(f"{index}/{path}/{quantity}")
            if not np.isfinite(config[path].get("energy_meV_atom", 0.0)):
                bad.append(f"{index}/{path}/energy_meV_atom")
    return bad


def summarize(configs, ratio=GATE_RATIO):
    """Worst configuration of each path, the verdict per quantity, and the overall verdict.

    A non-finite error anywhere (NaN or inf, in the production path or in the float32 yardstick)
    fails the gate before any maximum is taken: Python's max can drop a NaN depending on order,
    and an infinite yardstick would make the limit infinite.
    """
    bad = _nonfinite(configs)
    if bad:
        return {"passed": False, "nonfinite": bad}
    summary = {}
    for quantity in QUANTITIES:
        summary[quantity] = quantity_verdict(quantity,
                                             max(c["production"][quantity]["max_abs"] for c in configs),
                                             max(c["ase_f32"][quantity]["max_abs"] for c in configs), ratio)
    summary["energy_meV_atom"] = {path: max(c[path]["energy_meV_atom"] for c in configs)
                                  for path in ("production", "ase_f32")}
    summary["passed"] = all(summary[quantity]["passed"] for quantity in QUANTITIES)
    return summary


def random_strain(rng, amplitude, full):
    """Symmetric strain: isotropic plus (full cell) anisotropic and shear parts, uniform in +-amplitude."""
    if not full:
        return np.identity(3) * (1 + rng.uniform(-amplitude, amplitude))
    e = rng.uniform(-amplitude, amplitude, (3, 3))
    return np.identity(3) + 0.5 * (e + e.T)


def calculator_values(atoms, calc):
    probe = atoms.copy()
    probe.calc = calc
    return probe.get_potential_energy(), probe.get_forces(), probe.get_stress(voigt=False)


def baseline_values(engine, atoms):
    """Energy, forces and stress of the research engine's force field for a new configuration (fresh rebuild)."""
    import torch
    from langevin import gpu_langevin
    field, dyn = engine.field, engine.dyn
    field.set_cell(torch.tensor(atoms.cell.array, dtype=torch.float64, device=field.device))
    x = torch.tensor(atoms.positions, dtype=torch.float64, device=field.device)
    field.rebuild(x)
    field.neighbours.compact(x)
    count = int(field.neighbours.count)
    padded = gpu_langevin.round_up_edges(count, field.neighbours.capacity, dyn.edge_bucket)
    field.use_edges(padded)
    energy, forces, stress = field(x)
    return (float(energy.detach().reshape(-1)[0]), forces.detach().double().cpu().numpy(),
            stress.detach().double().cpu().numpy(), dict(edges=count, padded_edges=padded))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--engine", choices=("baseline", "fastmace"), default="baseline")
    p.add_argument("--baseline", type=Path, help="Exported research-engine directory with a COMMIT file")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model", default="mh1-r2scan", help="MACE model label (main.py --model)")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NaNbO3", "NbCl5"],
                   help="Benchmark names, or structure files with an optional repeat as PATH@NXxNYxNZ")
    p.add_argument("--repeat", nargs=3, type=int, default=(3, 3, 3))
    p.add_argument("--configs", type=int, default=3, help="Configurations per structure and cell mode")
    p.add_argument("--rattle-A", type=float, default=0.05)
    p.add_argument("--strain", type=float, default=0.02, help="Strain amplitude (uniform in +-strain)")
    p.add_argument("--backend", choices=("auto", "cueq"), default="auto",
                   help="auto: main.py's gpu default (hybrid FP32); cueq: cuEquivariance float64")
    p.add_argument("--dtype", choices=("auto", "float32", "float64"), default="auto")
    p.add_argument("--main-args", default="", help="Further main.py options for the production field")
    p.add_argument("--seed", type=int, default=20261015)
    args = p.parse_args(argv)
    if args.configs < 1 or args.rattle_A <= 0 or args.strain < 0:
        p.error("configs and rattle amplitude must be positive, strain nonnegative")
    if args.engine == "baseline" and args.baseline is None:
        p.error("--engine baseline needs --baseline DIR")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.engine == "fastmace":
        raise SystemExit("--engine fastmace: FastMACE has no ForceField yet (plan task 1a-T11)")
    import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)
    from fastmace.bench import ab
    export = ab.load_baseline(args.baseline)
    ab.activate(export)
    import langevin
    ab.check_origin(export, "langevin", langevin.__file__)
    from langevin import compile_cache
    from langevin import workflow as experiment

    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               text=True).strip():
        raise RuntimeError("GPU must be idle")
    md = experiment.parse_args(["--ensemble", "npt", "--backend", args.backend, "--model", args.model,
                                "--checkpoint", str(args.checkpoint), "--dtype", args.dtype]
                               + args.main_args.split())
    experiment.apply_polar_options(md)
    from langevin import upstream_mace
    # main.py's stress path, before any model is built. It is process-wide, so the reference
    # calculators take the same one-graph strain formula (exact up to rounding).
    upstream_mace.use_single_graph_strain(md.single_graph_strain)
    if md.backend == "hybrid":
        experiment.require_hybrid_wrapper()
    args.output.mkdir(parents=True, exist_ok=False)
    cache_info = compile_cache.configure(args.output)
    from langevin import gpu_langevin, polar_gpu
    cache_info["cuda_allocator"] = gpu_langevin.configure_allocator()  # before the first CUDA allocation
    import torch
    from ase import units
    from ase.io import read

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    flag_names = (
        "factor_species", "fold_cutoff", "sort_receivers", "density_first", "reassociate_first",
        "joint_radial_input", "fused_folded_backward", "segmented_first_sum", "gpu_neighbour_list",
        "block_linear", "grouped_linear", "merged_residual", "streaming_conv", "fused_radial_hidden", "pipelined")
    meta = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                engine=args.engine, baseline=export.as_dict(), python=sys.executable,
                checkpoint=str(checkpoint), checkpoint_sha256=experiment.sha256_file(checkpoint), tf32=False,
                model=args.model, backend=md.backend, dtype=md.dtype, compile_mode=md.compile_mode, skin=md.skin,
                rule=dict(ratio=GATE_RATIO, floors=FLOORS),
                protocol=(f"unrelaxed {'x'.join(map(str, args.repeat))} supercells, symmetric strain "
                          f"+-{args.strain:g} (hydrostatic: isotropic; full: anisotropic and shear), "
                          f"Gaussian rattle {args.rattle_A:g} A; float64 calculator as reference; gate: "
                          f"max error <= max({GATE_RATIO:g} x that of the float32 calculator, floor)"),
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                compile_cache=cache_info, structures={}, summary={})
    meta_path = args.output / "metadata.json"

    def save():
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    save()
    try:
        common, calc_md = experiment.load_md_calculator(md.dtype, md.backend == "hybrid", "cuda:0", checkpoint,
                                                        label=args.model, cueq_conv_fusion=md.cueq_conv_fusion)
        # The references keep MACE 0.3.16's conversion, as in every earlier gate run.
        upstream_mace.use_cueq_conv_fusion(None)
        calc64 = common.load_calculator(args.model, dtype="float64", enable_cueq=True)
        calc32 = common.load_calculator(args.model, dtype="float32", enable_cueq=True)
        meta["auto_options_off"] = experiment.adapt_options_to_model(md, calc_md.models[0])
        flags = meta["flags"] = {name: getattr(md, name) for name in flag_names}
        for calc in (calc64, calc32):
            experiment.configure_md_calculator(calc, compute_stress=True)
        experiment.configure_md_calculator(calc_md, compute_stress=False)  # the engine takes its own stress path
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["torch"] = torch.__version__
        for spec in args.structures:
            structure, source, repeat = experiment.structure_spec(spec, tuple(args.repeat))
            base = read(source).repeat(repeat)
            base.set_pbc(True)
            entry = meta["structures"][structure] = dict(
                natoms=len(base), source=str(source), repeat=list(repeat),
                source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(), configs=[])
            if polar_gpu.is_polar(calc64.models[0]):
                # MACE 0.3.16's PolarMACE stress misses the k-space cell dependence (MACE issue 1642);
                # the references take the corrected long-range path too.
                for reference in (calc64, calc32):
                    polar_gpu.enable(reference.models[0], base.cell.array, headroom=0.1)
                entry["polar_references"] = "polar_gpu (corrected stress)"
            for cell_mode in ("hydrostatic", "full"):
                rng = np.random.default_rng(args.seed)
                engine = experiment.GPUEngine(base.copy(), calc_md, timestep_fs=md.timestep_fs,
                                              friction=md.friction, seed=0, skin=md.skin,
                                              compile_mode=md.compile_mode, device="cuda:0", ensemble="npt",
                                              cell_mode=cell_mode, **flags)
                for index in range(args.configs):
                    atoms = base.copy()
                    atoms.set_cell(atoms.cell.array @ random_strain(rng, args.strain, cell_mode == "full"),
                                   scale_atoms=True)
                    atoms.positions += args.rattle_A * rng.standard_normal(atoms.positions.shape)
                    e64, f64, s64 = calculator_values(atoms, calc64)
                    e32, f32, s32 = calculator_values(atoms, calc32)
                    ep, fp, sp, edges = baseline_values(engine, atoms)
                    n = len(atoms)
                    row = dict(cell_mode=cell_mode, index=index, edges=edges,
                               reference_pressure_GPa=float(-np.trace(s64) / 3 / units.GPa),
                               ase_f32=dict(forces=errors(f64, f32), stress=errors(s64, s32),
                                            energy_meV_atom=abs(e32 - e64) / n * 1e3),
                               production=dict(forces=errors(f64, fp), stress=errors(s64, sp),
                                               energy_meV_atom=abs(ep - e64) / n * 1e3))
                    entry["configs"].append(row)
                    save()
                    print(structure, cell_mode, index,
                          "stress err production/f32 %.2e/%.2e eV/A3," % (row["production"]["stress"]["max_abs"],
                                                                          row["ase_f32"]["stress"]["max_abs"]),
                          "force err production/f32 %.2e/%.2e eV/A" % (row["production"]["forces"]["max_abs"],
                                                                        row["ase_f32"]["forces"]["max_abs"]),
                          flush=True)
                del engine
                torch.cuda.empty_cache()
            meta["summary"][structure] = summarize(entry["configs"])
            save()
            print(structure, "summary", json.dumps(meta["summary"][structure]), flush=True)
        meta["passed"] = all(s["passed"] for s in meta["summary"].values())
        meta["status"] = "complete"
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta["compile_cache"]["stats"] = compile_cache.cache_stats()
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
