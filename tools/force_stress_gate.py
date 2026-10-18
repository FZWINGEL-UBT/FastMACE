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

Processes (plan task 1a-T19; conductor decisions "after T10" and "Second outside review"). The driver
(``run``, the default) builds no model and imports neither torch nor the research engine. It writes
the configurations to files and starts one fresh child process per job, one after another:

  reference   one reference calculator (float64 or float32) in a process that builds no production
              engine; it evaluates its configurations one at a time (--reference-scope calculator:
              one process per structure and calculator; configuration: one per configuration and
              calculator). Every reference runs before the production process starts.
  production  the production path (--engine), one process for all structures:
                baseline  the research engine's force field from the export, built as its main.py
                          builds it (needs the hybrid launcher for the hybrid backend)
                fastmace  FastMACE's force field through the fixed-plan planner (no launcher)

A research-engine process that had built the hybrid engine moved the float64 POLAR-1-S reference by
up to 7.8e-7 eV/A (1a-T10), so no reference shares a process with a production engine.

The references are today's: the research engine's benchmark calculators (common.load_calculator)
from the export given by --baseline (with a COMMIT file and Structures/Benchmark), with MACE 0.3.16's
cuEq conversion, main.py's one-graph strain formula and, for PolarMACE, the corrected long-range
stress (polar_gpu). --reference-command and --production-command give each kind of child its
interpreter, with an optional launcher prefix (shell words), for example
  --reference-command "$ENVS/mace_cueq_latest/bin/python"
  --production-command "bash $B/.../oeq_no_nvcc.sh $ENV $CACHE $ENV/bin/python"
Children inherit the driver's environment (PYTHONPATH=<FastMACE checkout>). Output:
<output>/metadata.json, with the configurations, job files, values and logs of every child beside it.

Copied from the research engine's tools/validation/npt_stress_reference.py (with the ratio summary of
tools/validation/roundoff_reference.py) and refactored: the rule has floors, the production path is
chosen by --engine, the research engine is loaded only from the export given by path, and every
evaluation runs in a child process of its own kind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GATE_RATIO = 2.0
# SPEC §12.3: absolute floors of the rule, eV/A for the maximum force component and eV/A^3 for the
# maximum stress component (decision 7).
FLOORS = {"forces": 1e-5, "stress": 1e-7}
QUANTITIES = ("forces", "stress")
CELL_MODES = ("hydrostatic", "full")
# Reference calculators, in the order they run: (name in the rows, dtype).
REFERENCES = (("f64", "float64"), ("ase_f32", "float32"))
REFERENCE_SCOPES = ("calculator", "configuration")
# The research engine's benchmark crystals (workflow.NAMES); a name resolves to its r2SCAN CIF in the export.
BENCHMARK_NAMES = ("NaCl", "NaNbCl6", "NaTaCl6", "NaNbO3", "NaTaO3",
                   "NbCl3O", "TaCl3O", "NbCl2O", "NbCl5", "TaCl5")
BENCHMARK_DIRECTORY = Path("Structures", "Benchmark", "R2SCAN")
# Every engine option main.py passes to GPUEngine (species_skip was missing before 1a-T19).
FLAG_NAMES = (
    "factor_species", "fold_cutoff", "sort_receivers", "density_first", "reassociate_first",
    "joint_radial_input", "fused_folded_backward", "segmented_first_sum", "gpu_neighbour_list",
    "block_linear", "grouped_linear", "merged_residual", "species_skip", "streaming_conv", "fused_radial_hidden",
    "speculative_graph", "pipelined")


class GateJobError(RuntimeError):
    """A child process of the gate failed or left no result."""


# ------------------------------------------------------------------ the rule (unchanged since 1a-T18)


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


# ------------------------------------------------------------------ configurations (driver, no torch)


def structure_spec(spec, repeat, export_root):
    """(name, path, repeat) of a structure argument, as the research engine's workflow.structure_spec.

    "NaCl" -> the export's benchmark r2SCAN CIF, repeated by ``repeat``; "path/file.cif" -> that file,
    repeated by ``repeat``; "path/file.xyz@1x1x1" -> that file with its own repeat.
    """
    path, _, own = str(spec).partition("@")
    repeat = tuple(repeat)
    if own:
        try:
            repeat = tuple(int(n) for n in own.lower().split("x"))
        except ValueError:
            repeat = ()
        if len(repeat) != 3 or min(repeat) < 1:
            raise ValueError(f"{spec}: the repeat after @ must be NXxNYxNZ with positive integers")
    if path in BENCHMARK_NAMES:
        file = Path(export_root) / BENCHMARK_DIRECTORY / f"{path}.cif"
        if not file.is_file():
            raise FileNotFoundError(f"{spec}: the baseline export has no {BENCHMARK_DIRECTORY / (path + '.cif')}")
        return path, file.resolve(), repeat
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"{spec}: neither a benchmark structure ({', '.join(BENCHMARK_NAMES)}) nor a file")
    return file.stem, file.resolve(), repeat


def make_configurations(base, cell_mode, count, seed, strain, rattle_A):
    """The gate's configurations of one structure and cell mode, as earlier gate runs drew them.

    One generator per cell mode from ``seed``; for each configuration a random strain (scaling the
    atoms with the cell), then a Gaussian rattle.
    """
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(count):
        atoms = base.copy()
        atoms.set_cell(atoms.cell.array @ random_strain(rng, strain, cell_mode == "full"), scale_atoms=True)
        atoms.positions += rattle_A * rng.standard_normal(atoms.positions.shape)
        configs.append(atoms)
    return configs


def config_key(cell_mode, index):
    return f"{cell_mode}-{index}"


def write_configurations(path, base, configs_by_mode):
    """One structure's base cell and configurations ({cell mode: [Atoms]}) in an .npz file."""
    arrays = dict(numbers=np.asarray(base.numbers), base_cell=np.asarray(base.cell.array, float),
                  base_positions=np.asarray(base.positions, float))
    for cell_mode, configs in configs_by_mode.items():
        for index, atoms in enumerate(configs):
            if not np.array_equal(atoms.numbers, base.numbers):
                raise ValueError("a configuration must keep the base structure's species and order")
            arrays[f"{config_key(cell_mode, index)}/cell"] = np.asarray(atoms.cell.array, float)
            arrays[f"{config_key(cell_mode, index)}/positions"] = np.asarray(atoms.positions, float)
    np.savez(path, **arrays)


def read_configurations(path):
    """(base Atoms, {key: Atoms}) of a file from ``write_configurations``; keys "<cell mode>-<index>"."""
    from ase import Atoms

    with np.load(path) as data:
        numbers = data["numbers"]
        base = Atoms(numbers=numbers, positions=data["base_positions"], cell=data["base_cell"], pbc=True)
        keys = sorted({name.rsplit("/", 1)[0] for name in data.files if "/" in name}, key=_key_order)
        configs = {key: Atoms(numbers=numbers, positions=data[f"{key}/positions"], cell=data[f"{key}/cell"],
                              pbc=True) for key in keys}
    return base, configs


def _key_order(key):
    cell_mode, index = key.rsplit("-", 1)
    return CELL_MODES.index(cell_mode), int(index)


# ------------------------------------------------------------------ jobs and child processes


def reference_jobs(structures, keys, scope="calculator"):
    """The reference jobs, in order: per structure, per calculator, all keys (scope "calculator") or one key each.

    ``structures`` are names, ``keys`` maps a name to its configuration keys. Each job is one child process.
    """
    if scope not in REFERENCE_SCOPES:
        raise ValueError(f"reference scope must be one of {REFERENCE_SCOPES}, not {scope!r}")
    jobs = []
    for structure in structures:
        for name, dtype in REFERENCES:
            groups = [list(keys[structure])] if scope == "calculator" else [[key] for key in keys[structure]]
            for group in groups:
                suffix = "" if scope == "calculator" else f"-{group[0]}"
                jobs.append(dict(kind="reference", name=f"reference-{structure}-{name}{suffix}", structure=structure,
                                 reference=name, dtype=dtype, keys=group))
    return jobs


def child_argv(command, script, job_file):
    """The argv of a child: the command's shell words (interpreter, optional launcher prefix), script, job."""
    words = shlex.split(command) if isinstance(command, str) else list(command)
    if not words:
        raise ValueError("empty child command")
    kind = json.loads(Path(job_file).read_text())["kind"]
    return [*words, str(script), kind, "--job", str(job_file)]


def run_child(argv, log_path, env=None):
    """Run one child to completion, output to ``log_path``; returns its exit code."""
    with open(log_path, "w") as log:
        return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              env=env).returncode


def gpu_processes():
    """PIDs of the compute processes on the GPU (nvidia-smi)."""
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    return [int(line) for line in output.split() if line.strip()]


def load_values(path):
    """{key: (energy, forces, stress)} of a child's values file."""
    with np.load(path) as data:
        keys = sorted({name.rsplit("/", 1)[0] for name in data.files})
        return {key: (float(data[f"{key}/energy"]), data[f"{key}/forces"].astype(float),
                      data[f"{key}/stress"].astype(float)) for key in keys}


def save_values(path, values):
    """Write {key: (energy, forces, stress)} (a child's result) to an .npz file."""
    arrays = {}
    for key, (energy, forces, stress) in values.items():
        arrays[f"{key}/energy"] = np.asarray(float(energy))
        arrays[f"{key}/forces"] = np.asarray(forces, float)
        arrays[f"{key}/stress"] = np.asarray(stress, float)
    np.savez(path, **arrays)


def assemble_row(key, natoms, f64, f32, production, extra=None):
    """One configuration's row of metadata.json from the three (energy, forces, stress) evaluations."""
    from ase import units

    cell_mode, index = key.rsplit("-", 1)
    (e64, f64_forces, s64), (e32, f32_forces, s32), (ep, fp, sp) = f64, f32, production
    return dict(cell_mode=cell_mode, index=int(index), **(extra or {}),
                reference_pressure_GPa=float(-np.trace(s64) / 3 / units.GPa),
                ase_f32=dict(forces=errors(f64_forces, f32_forces), stress=errors(s64, s32),
                             energy_meV_atom=abs(e32 - e64) / natoms * 1e3),
                production=dict(forces=errors(f64_forces, fp), stress=errors(s64, sp),
                                energy_meV_atom=abs(ep - e64) / natoms * 1e3))


# ------------------------------------------------------------------ the driver


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--engine", choices=("baseline", "fastmace"), default="baseline")
    p.add_argument("--baseline", required=True, type=Path,
                   help="Exported research-engine directory with a COMMIT file and Structures/Benchmark "
                        "(the references are the research engine's calculators)")
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
    p.add_argument("--main-args", default="",
                   help="Further main.py options: the baseline's production field, and the references' strain path")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--reference-scope", choices=REFERENCE_SCOPES, default="calculator",
                   help="calculator: one reference process per structure and calculator; configuration: one "
                        "per configuration and calculator")
    p.add_argument("--reference-command", default=sys.executable,
                   help="Interpreter (with an optional launcher prefix) of the reference processes")
    p.add_argument("--production-command", default=sys.executable,
                   help="Interpreter (with an optional launcher prefix) of the production process")
    args = p.parse_args(argv)
    if args.configs < 1 or args.rattle_A <= 0 or args.strain < 0:
        p.error("configs and rattle amplitude must be positive, strain nonnegative")
    if args.engine == "fastmace" and args.main_args.strip():
        p.error("--main-args are main.py options for the baseline engine; FastMACE plans its own options")
    return args


def _repository_commit():
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = root / ".source_commit"
        return marker.read_text().strip() if marker.is_file() else None


def _read_export(path):
    """The export's root and commit (fastmace.bench.ab.load_baseline: refuses an export without COMMIT)."""
    from fastmace.bench import ab
    return ab.load_baseline(path)


def main(argv=None, runner=run_child, gpu_check=gpu_processes):
    """The driver: configurations, then every reference process, then the production process, then the rule.

    ``runner(argv, log_path)`` runs one child and returns its exit code; ``gpu_check()`` lists GPU
    compute processes, which must be none before each child (tests pass fakes for both).
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("reference", "production"):
        return child_main(argv)
    if argv and argv[0] == "run":
        argv = argv[1:]
    args = parse_args(argv)
    from ase.io import read

    export = _read_export(args.baseline)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if gpu_check():
        raise RuntimeError("GPU must be idle")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for sub in ("configs", "jobs", "values", "logs"):
        (output / sub).mkdir()
    meta = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                engine=args.engine, baseline=export.as_dict(), fastmace_commit=_repository_commit(),
                python=sys.executable, checkpoint=str(checkpoint), checkpoint_sha256=_sha256(checkpoint), tf32=False,
                model=args.model, backend=args.backend, dtype=args.dtype,
                rule=dict(ratio=GATE_RATIO, floors=FLOORS),
                processes_rule=("every reference in a fresh process that builds no production engine, before the "
                                f"production process (scope {args.reference_scope!r}); conductor decision after T10"),
                protocol=(f"unrelaxed {'x'.join(map(str, args.repeat))} supercells, symmetric strain "
                          f"+-{args.strain:g} (hydrostatic: isotropic; full: anisotropic and shear), "
                          f"Gaussian rattle {args.rattle_A:g} A; float64 calculator as reference; gate: "
                          f"max error <= max({GATE_RATIO:g} x that of the float32 calculator, floor)"),
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                structures={}, processes=[], summary={})
    meta_path = output / "metadata.json"

    def save():
        meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    save()
    try:
        keys, files = {}, {}
        for spec in args.structures:
            structure, source, repeat = structure_spec(spec, args.repeat, export.root)
            if structure in keys:
                raise ValueError(f"structure {structure} given twice")
            base = read(source).repeat(repeat)
            base.set_pbc(True)
            configs = {mode: make_configurations(base, mode, args.configs, args.seed, args.strain, args.rattle_A)
                       for mode in CELL_MODES}
            files[structure] = output / "configs" / f"{structure}.npz"
            write_configurations(files[structure], base, configs)
            keys[structure] = [config_key(mode, index) for mode in CELL_MODES for index in range(args.configs)]
            meta["structures"][structure] = dict(natoms=len(base), source=str(source), repeat=list(repeat),
                                                 source_sha256=_sha256(source), configurations=str(files[structure]),
                                                 configs=[])
        save()
        # md_dtype is the production path's --dtype; a reference job's own dtype is its calculator's.
        common = dict(model=args.model, checkpoint=str(checkpoint), baseline=str(export.root), backend=args.backend,
                      md_dtype=args.dtype, main_args=args.main_args)
        jobs = reference_jobs(list(keys), keys, args.reference_scope)
        jobs.append(dict(kind="production", name=f"production-{args.engine}", engine=args.engine,
                         structures={s: dict(configurations=str(files[s]), keys=keys[s]) for s in keys}))
        script = Path(__file__).resolve()
        references, production = {}, None
        for job in jobs:
            job.update(common, values=str(output / "values" / f"{job['name']}.npz"),
                       result=str(output / "values" / f"{job['name']}.json"))
            if job["kind"] == "reference":
                job["configurations"] = str(files[job["structure"]])
            job_file = output / "jobs" / f"{job['name']}.json"
            job_file.write_text(json.dumps(job, indent=2) + "\n")
            command = args.reference_command if job["kind"] == "reference" else args.production_command
            child = child_argv(command, script, job_file)
            busy = gpu_check()
            record = dict(name=job["name"], kind=job["kind"], argv=child, gpu_processes_before=busy)
            meta["processes"].append(record)
            if busy:
                raise GateJobError(f"GPU busy before {job['name']} (compute processes {busy})")
            started = time.perf_counter()
            record["exit_code"] = runner(child, output / "logs" / f"{job['name']}.log")
            record["wall_s"] = time.perf_counter() - started
            result = Path(job["result"])
            record["result"] = json.loads(result.read_text()) if result.is_file() else None
            save()
            if record["exit_code"] != 0 or not Path(job["values"]).is_file():
                raise GateJobError(f"{job['name']} exited with {record['exit_code']}; see logs/{job['name']}.log")
            if job["kind"] == "reference":
                references.setdefault((job["structure"], job["reference"]), {}).update(load_values(job["values"]))
            else:
                production = load_values(job["values"])
        production_info = meta["processes"][-1]["result"] or {}
        for structure in keys:
            entry = meta["structures"][structure]
            for key in keys[structure]:
                where = f"{structure}/{key}"
                extra = dict(edges=production_info.get("edges", {}).get(where))
                entry["configs"].append(assemble_row(key, entry["natoms"], references[structure, "f64"][where],
                                                     references[structure, "ase_f32"][where], production[where],
                                                     extra))
            meta["summary"][structure] = summarize(entry["configs"])
            print(structure, "summary", json.dumps(meta["summary"][structure]), flush=True)
        meta["passed"] = all(s["passed"] for s in meta["summary"].values())
        meta["status"] = "complete"
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return meta


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ child processes


def child_main(argv):
    p = argparse.ArgumentParser(prog="force_stress_gate.py reference|production")
    p.add_argument("kind", choices=("reference", "production"))
    p.add_argument("--job", required=True, type=Path)
    args = p.parse_args(argv)
    job = json.loads(args.job.read_text())
    if job["kind"] != args.kind:
        raise SystemExit(f"{args.job} is a {job['kind']} job, not {args.kind}")
    result = dict(name=job["name"], kind=job["kind"], pid=os.getpid(), host=os.uname().nodename,
                  python=sys.executable, started_utc=datetime.now(timezone.utc).isoformat(),
                  hybrid_launcher=bool(os.environ.get("OEQ_NVCC_ATTEMPTS")))
    started = time.perf_counter()
    try:
        if job["kind"] == "reference":
            values = reference_child(job, result)
        elif job["engine"] == "fastmace":
            values = fastmace_production_child(job, result)
        else:
            values = baseline_production_child(job, result)
        save_values(job["values"], values)
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["wall_s"] = time.perf_counter() - started
        Path(job["result"]).write_text(json.dumps(result, indent=2, default=str) + "\n")


def _activate_baseline(job):
    import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)
    from fastmace.bench import ab
    export = ab.load_baseline(job["baseline"])
    ab.activate(export)
    import langevin
    ab.check_origin(export, "langevin", langevin.__file__)
    from langevin import workflow as experiment
    md = experiment.parse_args(["--ensemble", "npt", "--backend", job["backend"], "--model", job["model"],
                                "--checkpoint", job["checkpoint"], "--dtype", job["md_dtype"]]
                               + job["main_args"].split())
    experiment.apply_polar_options(md)
    return experiment, md


def _torch_record(torch, result):
    result.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  peak_memory_bytes=int(torch.cuda.max_memory_reserved()))


def reference_child(job, result):
    """One reference calculator on its configurations, one at a time; this process builds no production engine."""
    experiment, md = _activate_baseline(job)
    from langevin import polar_gpu, upstream_mace
    # main.py's stress path, before any model is built: the references take the same one-graph strain
    # formula as the production path (exact up to rounding), as in every earlier gate run.
    upstream_mace.use_single_graph_strain(md.single_graph_strain)
    # The calculator as workflow.load_md_calculator loads it, without the hybrid and with MACE 0.3.16's
    # cuEq conversion (upstream_mace.use_cueq_conv_fusion(None)), as in every earlier gate run.
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MACE_BENCH_DEVICE="cuda:0",
                      MACE_BENCH_DTYPE=job["dtype"], MACE_BENCH_CUEQ="1", MACE_BENCH_CHECKPOINT=job["checkpoint"])
    import common
    import torch
    common.set_threads(1)
    torch.set_num_interop_threads(1)
    upstream_mace.use_cueq_conv_fusion(None)
    started = time.perf_counter()
    calc = common.load_calculator(job["model"], dtype=job["dtype"], enable_cueq=True)
    experiment.configure_md_calculator(calc, compute_stress=True)
    result["load_s"] = time.perf_counter() - started
    base, configs = read_configurations(job["configurations"])
    if polar_gpu.is_polar(calc.models[0]):
        # MACE 0.3.16's PolarMACE stress misses the k-space cell dependence (MACE issue 1642); the
        # references take the corrected long-range path, sized for the unstrained cell.
        polar_gpu.enable(calc.models[0], base.cell.array, headroom=0.1)
        result["polar_reference"] = "polar_gpu (corrected stress)"
    values = {}
    for key in job["keys"]:
        values[f"{job['structure']}/{key}"] = calculator_values(configs[key], calc)
        print(job["name"], key, flush=True)
    result.update(single_graph_strain=bool(md.single_graph_strain), dtype=job["dtype"])
    _torch_record(torch, result)
    return values


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


def baseline_production_child(job, result):
    """The research engine's production force field, built as main.py builds it, on every configuration."""
    experiment, md = _activate_baseline(job)
    from langevin import compile_cache, upstream_mace
    upstream_mace.use_single_graph_strain(md.single_graph_strain)
    if md.backend == "hybrid":
        experiment.require_hybrid_wrapper()
    result["compile_cache"] = compile_cache.configure(Path(job["values"]).parent / f"{job['name']}-cache")
    from langevin import gpu_langevin
    result["compile_cache"]["cuda_allocator"] = gpu_langevin.configure_allocator()
    import torch
    started = time.perf_counter()
    _, calc_md = experiment.load_md_calculator(md.dtype, md.backend == "hybrid", "cuda:0", Path(job["checkpoint"]),
                                               label=job["model"], cueq_conv_fusion=md.cueq_conv_fusion)
    result["auto_options_off"] = experiment.adapt_options_to_model(md, calc_md.models[0])
    flags = result["flags"] = {name: getattr(md, name) for name in FLAG_NAMES}
    experiment.configure_md_calculator(calc_md, compute_stress=False)  # the engine takes its own stress path
    result.update(load_s=time.perf_counter() - started, backend=md.backend, dtype=md.dtype,
                  compile_mode=md.compile_mode, skin=md.skin, edges={}, setup_s={})
    values = {}
    for structure, entry in job["structures"].items():
        base, configs = read_configurations(entry["configurations"])
        for cell_mode in CELL_MODES:
            started = time.perf_counter()
            engine = experiment.GPUEngine(base.copy(), calc_md, timestep_fs=md.timestep_fs, friction=md.friction,
                                          seed=0, skin=md.skin, compile_mode=md.compile_mode, device="cuda:0",
                                          ensemble="npt", cell_mode=cell_mode, **flags)
            result["setup_s"][f"{structure}/{cell_mode}"] = time.perf_counter() - started
            for key in entry["keys"]:
                if key.rsplit("-", 1)[0] != cell_mode:
                    continue
                energy, forces, stress, edges = baseline_values(engine, configs[key])
                values[f"{structure}/{key}"] = (energy, forces, stress)
                result["edges"][f"{structure}/{key}"] = edges
                print(job["name"], structure, key, flush=True)
            del engine
            torch.cuda.empty_cache()
    result["compile_cache"]["stats"] = compile_cache.cache_stats()
    _torch_record(torch, result)
    return values


def fastmace_values(engine, atoms):
    """Energy, forces and stress of a FastMACE engine's force field for a new configuration (fresh rebuild).

    The engine's force field on padded edges, as its captured stage B evaluates it (same bucket).
    """
    import torch

    from fastmace.neighbors.buckets import round_up_edges
    field = engine.force_field
    field.set_cell(torch.tensor(atoms.cell.array, dtype=torch.float64, device=field.device))
    x = torch.tensor(atoms.positions, dtype=torch.float64, device=field.device)
    field.rebuild(x)
    field.compact(x)
    count = int(field.edge_count)
    padded = round_up_edges(count, field.edge_capacity, engine.edge_bucket)
    field.use_edges(padded)
    out = field(x)
    return (float(out.energy.detach().reshape(-1)[0]), out.forces.detach().double().cpu().numpy(),
            out.stress.detach().double().cpu().numpy(), dict(edges=count, padded_edges=padded))


def fastmace_production_child(job, result):
    """FastMACE's production force field through the fixed-plan planner (NPT, so with stress), on every configuration.

    Needs no launcher: the planner's loading prepares the OEq runtime in this process before MACE is imported.
    """
    import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)
    from fastmace import config
    from fastmace.planner.fixed import FixedPlanner, auto_options_off, enabled_options

    planner = FixedPlanner(engine=config.EngineConfig(backend="hybrid" if job["backend"] == "auto" else "cueq"),
                           precision=config.PrecisionConfig(dtype=job["md_dtype"]))
    hardware = planner.hardware()  # the allocator before the first CUDA allocation
    import torch

    from fastmace import _patches
    from fastmace.model.compile import CompiledModelCache
    started = time.perf_counter()
    loaded = planner.load(job["model"], checkpoint=job["checkpoint"])
    result.update(load_s=time.perf_counter() - started, backend=loaded.backend, dtype=loaded.dtype,
                  checkpoint_sha256=loaded.sha256, edges={}, setup_s={}, plans={}, force_fields={})
    cache = CompiledModelCache()
    values = {}
    for structure, entry in job["structures"].items():
        base, configs = read_configurations(entry["configurations"])
        for cell_mode in CELL_MODES:
            started = time.perf_counter()
            plan = planner.plan(loaded, base, config.IntegratorConfig(ensemble="npt", cell_mode=cell_mode, seed=0),
                                hardware)
            engine = planner.build(plan, loaded, base.copy(), temperature_K=0.0, compile_cache=cache)
            result["setup_s"][f"{structure}/{cell_mode}"] = time.perf_counter() - started
            result["plans"][f"{structure}/{cell_mode}"] = dict(
                describe=plan.describe(), enabled=list(enabled_options(plan)), auto_options_off=auto_options_off(plan))
            result["force_fields"][f"{structure}/{cell_mode}"] = dict(engine.force_field.info())
            for key in entry["keys"]:
                if key.rsplit("-", 1)[0] != cell_mode:
                    continue
                energy, forces, stress, edges = fastmace_values(engine, configs[key])
                values[f"{structure}/{key}"] = (energy, forces, stress)
                result["edges"][f"{structure}/{key}"] = edges
                print(job["name"], structure, key, flush=True)
            del engine
            torch.cuda.empty_cache()
    result["patches"] = [str(record) for record in _patches.applied()]
    _torch_record(torch, result)
    return values


if __name__ == "__main__":
    main()
