"""Force, stress and energy gate: the production path against the float64 reference (SPEC §12.1-§12.3).

For rattled and strained supercells it evaluates, on the same configuration,

  f64         the float64 reference of fastmace.reference (MACE's calculator, cuEquivariance float64,
              with the reference record's corrections: 1b-T13)
  ase_f32     the float32 yardstick of fastmace.reference: the same calculator in float32 with its
              energy accumulated in float64 (§12.1), so that it measures the forward pass, not the
              float32 sum of the E0 table
  production  the production force field (hybrid FP32 by default, every default optimisation,
              compiled, padded edges as in the CUDA-graph path)

Rule (fastmace.planner.selfcheck's ratio rule, 1b-T16): for energy per atom, forces and stress, the
production path's largest error against float64 is at most GATE_RATIO times that of ase_f32, or at
most the quantity's floor of SPEC §12.3 (0.01 meV/atom, 1e-5 eV/A, 1e-7 eV/A^3) where the yardstick
happens to be nearly exact. Any NaN or inf fails. With the research engine as production
(--engine baseline), energy is reported and not gated: that engine sums its energy in float32.

Processes (plan task 1a-T19; conductor decisions "after T10" and "Second outside review"). The driver
(``run``, the default) builds no model and imports neither torch nor the research engine. It writes
the configurations to files and starts one fresh child process per job, one after another:

  reference   one reference calculator (float64 or float32) in a process that builds no production
              engine; it evaluates its configurations one at a time (--reference-scope calculator:
              one process per structure and calculator; configuration: one per configuration and
              calculator). Every reference runs before the production process starts.
  production  the production path (--engine), one process for all structures:
                fastmace  FastMACE's force field through the minimal planner of 1b (--planner minimal,
                          the default; the public path of Simulation and fm.md) or the fixed-plan
                          planner of main.py (--planner fixed); no launcher
                baseline  the research engine's force field from the export (--baseline), built as its
                          main.py builds it (needs the hybrid launcher for the hybrid backend)

A research-engine process that had built the hybrid engine moved the float64 POLAR-1-S reference by
up to 7.8e-7 eV/A (1a-T10), so no reference shares a process with a production engine. The first
float64 reference of a PolarMACE run checks its stress against finite differences of its energy on
its structure's first hydrostatic configuration (ReferenceCalculator.check_stress_fix, k-set frozen),
so a reference without the k-space correction of MACE issue 1642 cannot pass unnoticed (1a-T19).

Structures: benchmark names resolve to r2SCAN CIFs under --structures-root (a directory holding
Structures/Benchmark/R2SCAN, such as the baseline export, which is the default root when --baseline
is given), or files with an optional repeat. The checkpoint is the label's release file: the children
load the label with FASTMACE_MODELS set to --checkpoint's directory and its SHA-256 as the trust
decision. --reference-command and --production-command give each kind of child its interpreter, with
an optional launcher prefix (shell words), for example
  --reference-command "$ENVS/mace_cueq_latest/bin/python"
  --production-command "bash $B/.../oeq_no_nvcc.sh $ENV $CACHE $ENV/bin/python"
Children inherit the driver's environment (PYTHONPATH=<FastMACE checkout>). Output:
<output>/metadata.json, with the configurations, job files, values and logs of every child beside it.

Copied from the research engine's tools/validation/npt_stress_reference.py (with the ratio summary of
tools/validation/roundoff_reference.py) and refactored: the rule has floors and gates energy, the
references are the library's, the production path is chosen by --engine, the research engine is
loaded only from the export given by path, and every evaluation runs in a child process of its own
kind (1a-T19, rebuilt on fastmace.reference and selfcheck in 1b-T23).
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

# numpy and fastmace.errors only: the driver still imports neither torch nor MACE.
from fastmace.planner import selfcheck

GATE_RATIO = selfcheck.RATIO
# This driver's names for selfcheck's quantities; floors of SPEC §12.3 (decision 7): meV/atom for energy
# per atom, eV/A for the maximum force component, eV/A^3 for the maximum stress component.
SELFCHECK_NAMES = {"energy": "energy_meV_per_atom", "forces": "force_eV_per_A", "stress": "stress_eV_per_A3"}
QUANTITIES = tuple(SELFCHECK_NAMES)
FLOORS = {quantity: selfcheck.FLOORS[name] for quantity, name in SELFCHECK_NAMES.items()}
PLANNERS = ("minimal", "fixed")
CELL_MODES = ("hydrostatic", "full")
# Reference calculators, in the order they run: (name in the rows, dtype).
REFERENCES = (("f64", "float64"), ("ase_f32", "float32"))
REFERENCE_SCOPES = ("calculator", "configuration")
# Self-test of a float64 polar reference: its stress against central finite differences of its energy
# (strain +-FD_STRAIN, six Voigt components) on the first configuration, before any polar row counts.
# MACE 0.3.16's uncorrected PolarMACE stress is off by 2e-5 eV/A^3 on NaCl 3x3x3, the corrected one
# agrees to 7.5e-9 (ResearchModule studies/other_models/README.md); the tolerance lies between.
FD_STRAIN = 1e-4
FD_TOLERANCE = 1e-6  # eV/A^3
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
# The self-test runs on the first float64 reference's own structure, its first hydrostatic configuration.
# Until 4-T11 it ran on NaCl 3x3x3 whatever the run's structures (decision 26): POLAR-1-S on NbCl5 3x3x3
# differed from finite differences by 4.4e-4 eV/A^3, a k-vector crossing the hard k-space cutoff under
# the strain (3b-T08). check_stress_fix now holds the unstrained cell's k-set, 8.3e-9 there.
FD_KEY = "hydrostatic-0"
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


def rewrite_overrides(items):
    """{field: value} of ``FIELD=VALUE`` items, each a RewriteConfig field; "on"/"off"/"auto" or true/false."""
    import dataclasses

    from fastmace.config import RewriteConfig

    fields = {f.name: f for f in dataclasses.fields(RewriteConfig)}
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in fields or not value:
            raise ValueError(f"--rewrites {item!r}: expected FIELD=VALUE with a RewriteConfig field")
        default = fields[name].default
        if isinstance(default, bool):
            if value.lower() not in ("true", "false"):
                raise ValueError(f"--rewrites {item!r}: {name} takes true or false")
            out[name] = value.lower() == "true"
        else:
            out[name] = value
    return out


def engine_overrides(items):
    """{field: value} of ``FIELD=VALUE`` items, each a boolean EngineConfig field (true/false), e.g.
    model_template_cache=false (2-T23: the OMOL-0 XL production crash with and without OPT-T06's cache)."""
    import dataclasses

    from fastmace.config import EngineConfig

    fields = {f.name: f for f in dataclasses.fields(EngineConfig)}
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in fields or name == "backend" or not isinstance(fields[name].default, bool):
            raise ValueError(f"--engine-config {item!r}: expected FIELD=true|false with a boolean EngineConfig field")
        if value.lower() not in ("true", "false"):
            raise ValueError(f"--engine-config {item!r}: {name} takes true or false")
        out[name] = value.lower() == "true"
    return out


class GateJobError(RuntimeError):
    """A child process of the gate failed or left no result."""


# ------------------------------------------------------------------ the rule (selfcheck's, 1b-T16)


def errors(reference, candidate):
    """Maximum and RMS absolute error of candidate against reference, and the reference's largest value."""
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    delta = np.abs(candidate - reference)
    return dict(max_abs=float(delta.max()), rms=float(np.sqrt((delta ** 2).mean())),
                reference_max_abs=float(np.abs(reference).max()))


def quantity_verdict(quantity, production_max_abs, ase_f32_max_abs, ratio=GATE_RATIO):
    """The rule for one quantity: production <= max(ratio x float32 yardstick, floor) (selfcheck.verdict)."""
    limit = max(ratio * ase_f32_max_abs, FLOORS[quantity])
    return dict(production_max_abs=production_max_abs, ase_f32_max_abs=ase_f32_max_abs,
                ratio=production_max_abs / ase_f32_max_abs if ase_f32_max_abs else None,
                floor=FLOORS[quantity], limit=limit, passed=bool(production_max_abs <= limit))


def _value(config, path, quantity):
    if quantity == "energy":
        return config[path]["energy_meV_atom"]
    return config[path][quantity]["max_abs"]


def selfcheck_rows(configs):
    """The configurations' errors in fastmace.planner.selfcheck's row layout (path "yardstick" is ase_f32)."""
    return [{"production": {SELFCHECK_NAMES[q]: _value(c, "production", q) for q in QUANTITIES},
             "yardstick": {SELFCHECK_NAMES[q]: _value(c, "ase_f32", q) for q in QUANTITIES}} for c in configs]


def summarize(configs, ratio=GATE_RATIO, gated=QUANTITIES):
    """Worst configuration of each path, the verdict per quantity, and the overall verdict.

    The verdict is ``selfcheck.verdict``'s: a non-finite error anywhere (NaN or inf, in the production
    path or in the float32 yardstick, energy included) fails the gate before any maximum is taken.
    Quantities outside ``gated`` are reported with their verdict but do not decide the result.
    """
    names = {name: quantity for quantity, name in SELFCHECK_NAMES.items()}
    result = selfcheck.verdict(selfcheck_rows(configs), ratio)
    if result["nonfinite"]:
        nonfinite = []
        for entry in result["nonfinite"]:
            index, path, name = entry.split("/")
            nonfinite.append(f"{index}/{'ase_f32' if path == 'yardstick' else path}/{names[name]}")
        return {"passed": False, "nonfinite": nonfinite}
    summary = {}
    for quantity in QUANTITIES:
        verdict = quantity_verdict(quantity, max(_value(c, "production", quantity) for c in configs),
                                   max(_value(c, "ase_f32", quantity) for c in configs), ratio)
        if verdict["passed"] != result["quantities"][SELFCHECK_NAMES[quantity]]["passed"]:
            raise AssertionError(f"{quantity}: this driver's verdict differs from selfcheck.verdict")
        summary[quantity] = dict(verdict, gated=quantity in gated)
    summary["passed"] = all(summary[quantity]["passed"] for quantity in gated)
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
                                 reference=name, dtype=dtype, keys=group, implementation="fastmace.reference"))
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


def strained(atoms, strain):
    """A copy of ``atoms`` with the symmetric ``strain`` applied to cell and positions."""
    copy = atoms.copy()
    copy.set_cell(atoms.cell.array @ (np.identity(3) + strain), scale_atoms=True)
    return copy


def finite_difference_stress(energy, atoms, h=FD_STRAIN):
    """3x3 stress (eV/A^3, ASE sign convention) of ``energy(atoms)`` by central differences in the strain.

    Diagonal component ii: strain e_ii = +-h; shear ij: e_ij = e_ji = +-h/2, so in both cases
    sigma = (E(+) - E(-)) / (2 h V).
    """
    volume = atoms.get_volume()
    stress = np.zeros((3, 3))
    for i, j in VOIGT:
        e = np.zeros((3, 3))
        if i == j:
            e[i, i] = h
        else:
            e[i, j] = e[j, i] = h / 2
        value = (energy(strained(atoms, e)) - energy(strained(atoms, -e))) / (2 * h * volume)
        stress[i, j] = stress[j, i] = value
    return stress


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
    p.add_argument("--planner", choices=PLANNERS, default="minimal",
                   help="FastMACE production: the minimal planner of 1b (Simulation, fm.md) or main.py's fixed one")
    p.add_argument("--baseline", type=Path,
                   help="Exported research-engine directory with a COMMIT file (needed for --engine baseline; "
                        "the default --structures-root)")
    p.add_argument("--structures-root", type=Path,
                   help="Directory holding Structures/Benchmark/R2SCAN for benchmark names (default: --baseline)")
    p.add_argument("--checkpoint", required=True, type=Path, help="The label's release file")
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
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="RewriteConfig fields for FastMACE's minimal planner (e.g. gated_rows=on); others stay auto")
    p.add_argument("--engine-config", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="Boolean EngineConfig fields for FastMACE's minimal planner (e.g. model_template_cache=false)")
    p.add_argument("--reference-command", default=sys.executable,
                   help="Interpreter (with an optional launcher prefix) of the reference processes")
    p.add_argument("--production-command", default=sys.executable,
                   help="Interpreter (with an optional launcher prefix) of the production process")
    args = p.parse_args(argv)
    if args.configs < 1 or args.rattle_A <= 0 or args.strain < 0:
        p.error("configs and rattle amplitude must be positive, strain nonnegative")
    if args.engine == "fastmace" and args.main_args.strip():
        p.error("--main-args are main.py options for the baseline engine; FastMACE plans its own options")
    try:
        args.rewrites = rewrite_overrides(args.rewrites)
    except ValueError as exc:
        p.error(str(exc))
    try:
        args.engine_config = engine_overrides(args.engine_config)
    except ValueError as exc:
        p.error(str(exc))
    if args.engine_config and (args.engine != "fastmace" or args.planner != "minimal"):
        p.error("--engine-config are EngineConfig fields of FastMACE's minimal planner (--engine fastmace)")
    if args.rewrites and (args.engine != "fastmace" or args.planner != "minimal"):
        p.error("--rewrites are RewriteConfig fields of FastMACE's minimal planner (--engine fastmace)")
    if args.engine == "baseline" and args.baseline is None:
        p.error("--engine baseline builds the research engine from its export: give --baseline DIR")
    if args.baseline is None and args.structures_root is None:
        p.error("give --structures-root DIR (or --baseline DIR) for the benchmark structures")
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

    export = _read_export(args.baseline) if args.baseline is not None else None
    structures_root = args.structures_root if args.structures_root is not None else export.root
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    checkpoint_sha256 = _sha256(checkpoint)
    energy_gated = args.engine == "fastmace"
    gated = QUANTITIES if energy_gated else ("forces", "stress")
    if gpu_check():
        raise RuntimeError("GPU must be idle")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for sub in ("configs", "jobs", "values", "logs"):
        (output / sub).mkdir()
    meta = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                engine=args.engine, planner=args.planner if args.engine == "fastmace" else None,
                baseline=export.as_dict() if export else None, structures_root=str(structures_root),
                fastmace_commit=_repository_commit(),
                python=sys.executable, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha256, tf32=False,
                model=args.model, backend=args.backend, dtype=args.dtype,
                rule=dict(ratio=GATE_RATIO, floors=FLOORS, gated=list(gated), source="fastmace.planner.selfcheck"),
                energy_gated=energy_gated,
                references="fastmace.reference (1b-T13): float64 reference, float32 yardstick with float64 energy",
                processes_rule=("every reference in a fresh process that builds no production engine, before the "
                                f"production process (scope {args.reference_scope!r}); conductor decision after T10"),
                protocol=(f"unrelaxed {'x'.join(map(str, args.repeat))} supercells, symmetric strain "
                          f"+-{args.strain:g} (hydrostatic: isotropic; full: anisotropic and shear), "
                          f"Gaussian rattle {args.rattle_A:g} A; float64 reference; gate: max error <= "
                          f"max({GATE_RATIO:g} x that of the float32 yardstick, floor), for "
                          f"{', '.join(gated)}"),
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                structures={}, processes=[], summary={})
    meta_path = output / "metadata.json"

    def save():
        meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    save()
    try:
        keys, files = {}, {}
        for spec in args.structures:
            structure, source, repeat = structure_spec(spec, args.repeat, structures_root)
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
        common = dict(model=args.model, label=args.model, checkpoint=str(checkpoint),
                      checkpoint_sha256=checkpoint_sha256, baseline=str(export.root) if export else None,
                      backend=args.backend, md_dtype=args.dtype, main_args=args.main_args, rewrites=args.rewrites,
                      engine_config=args.engine_config)
        jobs = reference_jobs(list(keys), keys, args.reference_scope)
        # One finite-difference self-test, by the first float64 reference (it acts for polar models only), on
        # its structure's first hydrostatic configuration.
        fd_job = next(job for job in jobs if job["dtype"] == "float64")
        fd_job["fd_check"] = str(files[fd_job["structure"]])
        jobs.append(dict(kind="production", name=f"production-{args.engine}", engine=args.engine,
                         planner=args.planner,
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
        # Finite-difference self-tests of the polar references, against the production stress too.
        for process in meta["processes"]:
            check = (process.get("result") or {}).get("fd_stress_check")
            if check:
                where = f"{check['structure']}/{check['key']}"
                if where in production:
                    check["production_max_abs"] = float(np.abs(production[where][2]
                                                               - np.array(check["fd_stress"])).max())
                meta.setdefault("fd_stress_checks", {})[where] = check
        for structure in keys:
            entry = meta["structures"][structure]
            for key in keys[structure]:
                where = f"{structure}/{key}"
                extra = dict(edges=production_info.get("edges", {}).get(where))
                entry["configs"].append(assemble_row(key, entry["natoms"], references[structure, "f64"][where],
                                                     references[structure, "ase_f32"][where], production[where],
                                                     extra))
            meta["summary"][structure] = summarize(entry["configs"], gated=gated)
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
                  # oeq_no_nvcc.sh's exports, read before FastMACE's in-process OEq runtime sets some of them
                  hybrid_launcher=all(os.environ.get(name) for name in ("OEQ_NVCC_ATTEMPTS", "CUDA_HOME", "CUDACXX")))
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


def _models_environment(job):
    """This process's environment with FASTMACE_MODELS at the checkpoint's directory: the label resolves to it."""
    return {**os.environ, "FASTMACE_MODELS": str(Path(job["checkpoint"]).parent)}


def _evaluate_reference(reference, atoms):
    """(energy, forces, stress) of one configuration; a strained PolarMACE cell outside the reference's k-space
    headroom resizes it for that cell first (as selfcheck does)."""
    from fastmace.reference.records import ReferencePolicyError

    try:
        return reference.evaluate(atoms)
    except ReferencePolicyError as error:
        if not (reference.is_polar and "overflowed" in str(error)):
            raise
        reference.rebase(atoms)
        return reference.evaluate(atoms)


def reference_child(job, result):
    """One reference calculator (fastmace.reference, 1b-T13) on its configurations, one at a time; this process
    builds no production engine."""
    # Expandable segments only place memory differently: a float64 POLAR-1-M reference on NaNbO3 3x3x3 ran
    # out of memory with 12.5 GiB reserved but unallocated (leto, 18 October 2026).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch

    import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)
    from fastmace.reference import calculators
    torch.set_num_threads(1)
    base, configs = read_configurations(job["configurations"])
    started = time.perf_counter()
    # The label's record (family, head, model type) with the checkpoint's hash as the trust decision; MACE's
    # calculator, MACE 0.3.16's cuEq conversion and the record's corrections in the safe order (calculators.load).
    reference = calculators.load(job["label"], base, dtype=job["dtype"], device="cuda:0",
                                 sha256=job["checkpoint_sha256"], environ=_models_environment(job))
    loaded_sha256 = reference.loaded.sha256 if reference.loaded is not None else None
    if loaded_sha256 != job["checkpoint_sha256"]:
        raise GateJobError(f"the reference loaded {loaded_sha256}, not the checkpoint {job['checkpoint_sha256']}")
    result["load_s"] = time.perf_counter() - started
    result["reference"] = json.loads(json.dumps(reference.info(), default=str))
    values = {}
    for key in job["keys"]:
        values[f"{job['structure']}/{key}"] = _evaluate_reference(reference, configs[key])
        print(job["name"], key, flush=True)
    if reference.is_polar and job.get("fd_check"):
        # The self-test: a reference without the k-space stress correction must not pass unnoticed. The
        # long-range state is sized for its configuration's cell, then goes back to the base cell.
        _, fd_configs = read_configurations(job["fd_check"])
        reference.rebase(fd_configs[FD_KEY])
        try:
            check = reference.check_stress_fix(fd_configs[FD_KEY], tolerance=FD_TOLERANCE, h=FD_STRAIN)
        finally:
            reference.rebase(base)
        result["fd_stress_check"] = dict(structure=job["structure"], key=FD_KEY, **check)
        print(job["name"], "finite-difference stress check", FD_KEY, f"{check['max_abs']:.2e}", flush=True)
    result.update(dtype=job["dtype"], role=reference.role, record_id=reference.record.id)
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


def _fastmace_build(job, result):
    """``build(structure, base, cell_mode) -> (engine, plan record)`` through the planner the job names."""
    import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)
    from fastmace import config

    backend = "hybrid" if job["backend"] == "auto" else "cueq"
    precision = "float32" if job["md_dtype"] in ("auto", "float32") else "float64"
    if job.get("planner", "fixed") == "fixed":
        from fastmace.planner.fixed import FixedPlanner, auto_options_off, enabled_options

        planner = FixedPlanner(engine=config.EngineConfig(backend=backend),
                               precision=config.PrecisionConfig(dtype=job["md_dtype"]))
        hardware = planner.hardware()  # the allocator before the first CUDA allocation
        from fastmace.model.compile import CompiledModelCache
        started = time.perf_counter()
        loaded = planner.load(job["model"], checkpoint=job["checkpoint"])
        cache = CompiledModelCache()

        def build(structure, base, cell_mode):
            plan = planner.plan(loaded, base, config.IntegratorConfig(ensemble="npt", cell_mode=cell_mode, seed=0),
                                hardware)
            engine = planner.build(plan, loaded, base.copy(), temperature_K=0.0, compile_cache=cache)
            return engine, dict(describe=plan.describe(), enabled=list(enabled_options(plan)),
                                auto_options_off=auto_options_off(plan))
    else:
        from fastmace.model import loading
        from fastmace.model.forcefield import ForceField
        from fastmace.planner import minimal
        from fastmace.planner.fixed import auto_options_off, enabled_options
        from fastmace.planner.workload import Workload

        planner = minimal.MinimalPlanner(engine=config.EngineConfig(backend=backend, **job.get("engine_config", {})))
        started = time.perf_counter()
        # fm.load: prepares the OEq runtime before MACE is imported (1b-T08b), the hash as the trust decision.
        loaded = loading.load_pristine(job["model"], checkpoint=job["checkpoint"], sha256=job["checkpoint_sha256"])
        hardware = planner.hardware()

        def build(structure, base, cell_mode):
            # One ForceField per structure: its species superset is the structure's (owner decision D3).
            rewrites = config.RewriteConfig(**job["rewrites"]) if job.get("rewrites") else "auto"
            force_field = ForceField(loaded, base, precision=precision, rewrites=rewrites)
            plan = planner.plan(force_field, base, config.IntegratorConfig(ensemble="npt", cell_mode=cell_mode,
                                                                           seed=0, temperature_K=300.0),
                                Workload(max_temperature_K=300.0), hardware)
            engine = planner.build(plan, force_field, base.copy(), temperature_K=0.0)
            return engine, dict(describe=plan.describe(), enabled=list(enabled_options(plan)),
                                auto_options_off=auto_options_off(plan), backend=plan.engine.backend,
                                float64_energy=plan.precision.float64_energy)
    result.update(load_s=time.perf_counter() - started, planner=job.get("planner", "fixed"),
                  checkpoint_sha256=loaded.sha256)
    if loaded.sha256 != job.get("checkpoint_sha256", loaded.sha256):
        raise GateJobError(f"production loaded {loaded.sha256}, not the checkpoint {job['checkpoint_sha256']}")
    return build


def fastmace_production_child(job, result):
    """FastMACE's production force field through the job's planner (NPT, so with stress), on every configuration.

    Needs no launcher: loading prepares the OEq runtime in this process before MACE is imported.
    """
    build = _fastmace_build(job, result)
    import torch

    from fastmace import _patches
    result.update(edges={}, setup_s={}, plans={}, force_fields={})
    values = {}
    for structure, entry in job["structures"].items():
        base, configs = read_configurations(entry["configurations"])
        for cell_mode in CELL_MODES:
            started = time.perf_counter()
            engine, plan = build(structure, base, cell_mode)
            result["setup_s"][f"{structure}/{cell_mode}"] = time.perf_counter() - started
            result["plans"][f"{structure}/{cell_mode}"] = plan
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
