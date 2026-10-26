#!/usr/bin/env python3
"""Bounded FastMACE 1.0.0 vs NVIDIA ALCHEMI 0.2.0 comparison.

The suite runs every timed side in a fresh process.  It is intended to be
invoked under ``tools/ci/local_gpu.py run --quiet``; the suite itself refuses
an unreserved run unless ``--allow-unreserved`` is given for development.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = "post1-alchemi-20261025"
MODEL_LABEL = "OMAT-0 medium"
MODEL_SHA256 = "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a"
DT_FS = 1.0
TEMPERATURE_K = 300.0
FRICTION_PER_FS = 0.01
SKIN_A = 0.3
SEED = 0
CASES = {"NaCl216": ("NaCl.cif", 3, 216), "NbCl5_1944": ("NbCl5.cif", 3, 1944)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(*arrays) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def structure(name: str, root: Path):
    import numpy as np
    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    filename, repeat, expected = CASES[name]
    atoms = read(root / filename).repeat((repeat,) * 3)
    atoms.set_pbc(True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.default_rng(SEED))
    Stationary(atoms)
    if len(atoms) != expected:
        raise RuntimeError(f"{name}: expected {expected} atoms, got {len(atoms)}")
    return atoms


def input_record(atoms) -> dict:
    from ase import units

    velocities = atoms.get_velocities()
    masses = atoms.get_masses()
    return {
        "natoms": len(atoms),
        "cell_A": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
        "masses_amu_sha256": array_digest(masses),
        "positions_A_sha256": array_digest(atoms.positions),
        "velocities_A_per_internal_time_sha256": array_digest(velocities),
        "complete_input_sha256": array_digest(
            atoms.numbers, masses, atoms.positions, velocities, atoms.cell.array, atoms.pbc
        ),
        "velocity_units": "angstrom per ASE internal time unit",
        "fs_per_internal_time": float(1.0 / units.fs),
        "initial_temperature_K": float(atoms.get_temperature()),
    }


def set_tf32_off(torch) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except AttributeError:
        pass


class GPUSampler:
    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:
            self.samples.append({"error": f"{type(exc).__name__}: {exc}"})
            return

        def sample():
            while not self._stop.is_set():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.samples.append(
                        {
                            "t_monotonic_s": time.monotonic(),
                            "temperature_C": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                            "gpu_util_percent": util.gpu,
                            "memory_util_percent": util.memory,
                            "memory_used_bytes": memory.used,
                            "power_W": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                        }
                    )
                except Exception as exc:
                    self.samples.append({"error": f"{type(exc).__name__}: {exc}"})
                    return
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        numeric = [s for s in self.samples if "gpu_util_percent" in s]
        result = {"sample_interval_s": self.interval_s, "n": len(numeric), "samples": self.samples}
        for key in ("temperature_C", "gpu_util_percent", "memory_util_percent", "memory_used_bytes", "power_W"):
            values = [float(s[key]) for s in numeric]
            if values:
                result[key] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
        return result


def versions(torch) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    names = [
        "fastmace",
        "mace-torch",
        "nvalchemi-toolkit",
        "nvalchemi-toolkit-ops",
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu13",
        "warp-lang",
        "ase",
        "numpy",
    ]
    out = {"python": platform.python_version(), "torch": torch.__version__}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def common_record(args, atoms, torch) -> dict:
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    actual_sha = sha256(checkpoint)
    if actual_sha != MODEL_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 is {actual_sha}, expected {MODEL_SHA256}")
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "running",
        "started_utc": utc_now(),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "engine": args.engine,
        "kind": args.kind,
        "case": args.case,
        "round": args.round,
        "model": MODEL_LABEL,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "dt_fs": DT_FS,
        "temperature_K": TEMPERATURE_K,
        "friction_per_fs": FRICTION_PER_FS,
        "skin_A": SKIN_A,
        "seed": SEED,
        "tf32": False,
        "versions": versions(torch),
        "input": input_record(atoms),
        "vendor_mace_pin_override": {
            "vendor_extra": "nvalchemi-toolkit[mace]==0.2.0 pins mace-torch==0.3.15",
            "used": "nvalchemi-toolkit[cu13]==0.2.0 base plus explicit mace-torch==0.3.16",
        },
    }


def finite_result(energy, forces) -> dict:
    import numpy as np

    energy = float(energy)
    forces = np.asarray(forces)
    return {
        "energy_eV": energy,
        "forces_eV_A": forces.tolist(),
        "finite": bool(math.isfinite(energy) and np.isfinite(forces).all()),
        "force_max_abs_eV_A": float(np.max(np.abs(forces))),
        "force_rms_eV_A": float(np.sqrt(np.mean(forces**2))),
    }


def validate_reference(args, atoms, record, torch):
    from mace.calculators import MACECalculator

    start = time.perf_counter()
    calc = MACECalculator(
        model_paths=str(args.checkpoint.expanduser()), device="cuda", default_dtype="float64", enable_cueq=False
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    torch.cuda.synchronize()
    record.update(
        validation=finite_result(energy, forces),
        setup_and_evaluate_s=time.perf_counter() - start,
        precision={"weights": "float64", "energy": "float64", "forces": "float64"},
        implementation="MACECalculator 0.3.16, upstream e3nn reference, uncompiled",
    )


def validate_fastmace(args, atoms, record, torch):
    import fastmace as fm

    start = time.perf_counter()
    loaded = fm.load(str(args.checkpoint.expanduser()), sha256=MODEL_SHA256)
    field = fm.ForceField(loaded, atoms, precision="float32")
    energy, forces, _ = field.evaluate(atoms, stress=False)
    torch.cuda.synchronize()
    plan = field.evaluation_plan(atoms)
    record.update(
        validation=finite_result(energy, forces),
        setup_and_evaluate_s=time.perf_counter() - start,
        precision={"forces": plan.precision.dtype, "energy_accumulation": "float64"},
        implementation="FastMACE 1.0.0 one-off evaluation plan",
        plan=plan.describe(full=True),
        effective_cutoff_A=float(loaded.model.r_max.item()),
    )


def make_alchemi(args, atoms, torch):
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import NVTLangevin
    from nvalchemi.dynamics.base import DynamicsStage
    from nvalchemi.hooks import NeighborListHook
    from nvalchemi.models.mace import MACEWrapper

    device = torch.device("cuda:0")
    wrapper = MACEWrapper.from_checkpoint(
        # 0.2.0 passes checkpoint_path to MACE 0.3.16's download_mace_mp_checkpoint,
        # which passes non-preset values to urllib.request.urlopen; a file:// URI
        # loads the local checkpoint file without network access or vendor edits.
        checkpoint_path=args.checkpoint.expanduser().resolve(strict=True).as_uri(),
        device=device,
        enable_cueq=True,
        dtype=torch.float32,
        compile_model=True,
    )
    data = AtomicData(
        atomic_numbers=torch.as_tensor(atoms.numbers, dtype=torch.long, device=device),
        atomic_masses=torch.as_tensor(atoms.get_masses(), dtype=torch.float32, device=device),
        positions=torch.as_tensor(atoms.positions, dtype=torch.float32, device=device),
        velocities=torch.as_tensor(atoms.get_velocities(), dtype=torch.float32, device=device),
        forces=torch.zeros((len(atoms), 3), dtype=torch.float32, device=device),
        energy=torch.zeros((1, 1), dtype=torch.float32, device=device),
        cell=torch.as_tensor(atoms.cell.array[None], dtype=torch.float32, device=device),
        pbc=torch.as_tensor(atoms.pbc[None], dtype=torch.bool, device=device),
    )
    batch = Batch.from_data_list([data], device=device)
    hook = NeighborListHook(wrapper.model_config.neighbor_config, skin=SKIN_A, stage=DynamicsStage.BEFORE_COMPUTE)
    dynamics = NVTLangevin(
        model=wrapper,
        dt=DT_FS,
        temperature=TEMPERATURE_K,
        friction=FRICTION_PER_FS,
        random_seed=SEED,
        hooks=[hook],
    )
    return wrapper, hook, dynamics, batch


def alchemi_initial_compute(dynamics, batch):
    from nvalchemi.dynamics.base import DynamicsStage

    dynamics._open_hooks()
    try:
        dynamics._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
        outputs = dynamics.compute(batch)
    finally:
        dynamics._close_hooks()
    return outputs


def validate_alchemi(args, atoms, record, torch):
    start = time.perf_counter()
    wrapper, hook, dynamics, batch = make_alchemi(args, atoms, torch)
    outputs = alchemi_initial_compute(dynamics, batch)
    torch.cuda.synchronize()
    energy = outputs["energy"].reshape(-1)[0].item()
    forces = outputs["forces"].detach().cpu().numpy()
    config = wrapper.model_config.neighbor_config
    record.update(
        validation=finite_result(energy, forces),
        setup_and_evaluate_s=time.perf_counter() - start,
        precision={
            "weights": "float32", "energy": str(outputs["energy"].dtype), "forces": str(outputs["forces"].dtype)
        },
        implementation="ALCHEMI MACEWrapper(enable_cueq=True, compile_model=True)",
        effective_cutoff_A=float(config.cutoff),
        neighbor_skin_A=float(hook.skin),
        neighbor_search_radius_A=float(config.cutoff + hook.skin),
        neighbor_method=getattr(hook, "_neighbor_list_method", None),
    )


def block_times(call, step_count, torch, steps: int, blocks: int) -> tuple[list[float], list[int]]:
    values, completed = [], []
    for _ in range(blocks):
        before_step = int(step_count())
        start = time.perf_counter()
        call(steps)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        actual = int(step_count()) - before_step
        values.append(1000.0 * elapsed / actual if actual > 0 else float("nan"))
        completed.append(actual)
    return values, completed


def temperature_from_momenta(numbers, masses, momenta) -> float:
    from ase import Atoms

    probe = Atoms(numbers=numbers)
    probe.set_masses(masses)
    probe.set_momenta(momenta)
    return float(probe.get_temperature())


def temperature_from_velocities(numbers, masses, velocities) -> float:
    from ase import Atoms

    probe = Atoms(numbers=numbers)
    probe.set_masses(masses)
    probe.set_velocities(velocities)
    return float(probe.get_temperature())


def compile_counters(torch) -> dict:
    counters = torch._dynamo.utils.counters
    return {
        group: dict(values)
        for group, values in counters.items()
        if group in ("frames", "stats", "recompiles", "graph_break")
    }


def counter_delta(before: dict, after: dict) -> dict:
    result = {}
    for group in set(before) | set(after):
        changes = {}
        for key in set(before.get(group, {})) | set(after.get(group, {})):
            delta = after.get(group, {}).get(key, 0) - before.get(group, {}).get(key, 0)
            if delta:
                changes[key] = delta
        if changes:
            result[group] = changes
    return result


def timing_fastmace(args, atoms, record, torch):
    import fastmace as fm
    from fastmace.config import NeighbourConfig

    sampler = GPUSampler()
    sampler.start()
    setup_start = time.perf_counter()
    loaded = fm.load(str(args.checkpoint.expanduser()), sha256=MODEL_SHA256)
    field = fm.ForceField(loaded, atoms, precision="float32")
    sim = fm.Simulation(
        atoms,
        field,
        integrator=fm.integrators.Langevin(
            timestep_fs=DT_FS, temperature_K=TEMPERATURE_K, friction_per_fs=FRICTION_PER_FS
        ),
        neighbours=NeighbourConfig(skin=SKIN_A),
        seed=SEED,
    )
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - setup_start
    warm_start = time.perf_counter()
    sim.run(steps=args.warmup)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - warm_start
    before = compile_counters(torch)
    ms, completed = block_times(
        lambda n: sim.run(steps=n), lambda: sim.committed().step, torch, args.steps, args.blocks
    )
    after = compile_counters(torch)
    state = sim.committed()
    delta = counter_delta(before, after)
    invalid = [i for i, x in enumerate(ms) if not math.isfinite(x) or completed[i] != args.steps]
    if delta.get("recompiles") or delta.get("stats") or delta.get("frames"):
        invalid.extend(i for i in range(len(ms)) if i not in invalid)
    record.update(
        setup_s=setup_s,
        warmup_steps=args.warmup,
        warmup_s=warmup_s,
        timed_steps_per_block=args.steps,
        timed_blocks=args.blocks,
        ms_per_step=ms,
        median_ms_per_step=statistics.median(ms),
        invalid_block_indices=sorted(set(invalid)),
        valid=not invalid,
        completed_timed_steps=sum(completed),
        final_step=state.step,
        final_temperature_K=temperature_from_momenta(state.species, state.masses, state.momenta),
        final_energy_eV=state.energy,
        plan=sim.plan.describe(full=True),
        plan_tier=sim.plan.tier,
        plan_backend=sim.plan.engine.backend,
        precision={"forces": sim.plan.precision.dtype, "energy_accumulation": "float64"},
        effective_cutoff_A=float(loaded.model.r_max.item()),
        neighbor_skin_A=SKIN_A,
        compile_counter_delta_during_timing=delta,
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    sim.close()
    record["gpu_samples"] = sampler.stop()


def timing_alchemi(args, atoms, record, torch):
    sampler = GPUSampler()
    sampler.start()
    setup_start = time.perf_counter()
    wrapper, hook, dynamics, batch = make_alchemi(args, atoms, torch)
    alchemi_initial_compute(dynamics, batch)
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - setup_start
    warm_start = time.perf_counter()
    batch = dynamics.run(batch, n_steps=args.warmup)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - warm_start
    before = compile_counters(torch)

    def run(n):
        nonlocal batch
        batch = dynamics.run(batch, n_steps=n)

    ms, completed = block_times(run, lambda: dynamics.step_count, torch, args.steps, args.blocks)
    after = compile_counters(torch)
    delta = counter_delta(before, after)
    forces = batch.forces.detach().cpu()
    energy = batch.energy.detach().cpu()
    positions = batch.positions.detach().cpu()
    velocities = batch.velocities.detach().cpu().numpy()
    invalid = [i for i, x in enumerate(ms) if not math.isfinite(x) or completed[i] != args.steps]
    finite = bool(torch.isfinite(forces).all() and torch.isfinite(energy).all() and torch.isfinite(positions).all())
    if not finite or delta.get("recompiles") or delta.get("stats") or delta.get("frames"):
        invalid.extend(i for i in range(len(ms)) if i not in invalid)
    config = wrapper.model_config.neighbor_config
    record.update(
        setup_s=setup_s,
        warmup_steps=args.warmup,
        warmup_s=warmup_s,
        timed_steps_per_block=args.steps,
        timed_blocks=args.blocks,
        ms_per_step=ms,
        median_ms_per_step=statistics.median(ms),
        invalid_block_indices=sorted(set(invalid)),
        valid=not invalid,
        completed_timed_steps=sum(completed),
        final_step=int(dynamics.step_count),
        final_temperature_K=temperature_from_velocities(atoms.numbers, atoms.get_masses(), velocities),
        final_energy_eV=float(energy.reshape(-1)[0]),
        finite_final_state=finite,
        implementation="ALCHEMI MACEWrapper(enable_cueq=True, compile_model=True)",
        precision={"weights": "float32", "energy": str(batch.energy.dtype), "forces": str(batch.forces.dtype)},
        effective_cutoff_A=float(config.cutoff),
        neighbor_skin_A=float(hook.skin),
        neighbor_search_radius_A=float(config.cutoff + hook.skin),
        neighbor_method=getattr(hook, "_neighbor_list_method", None),
        compile_counter_delta_during_timing=delta,
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    record["gpu_samples"] = sampler.stop()


def worker(args) -> int:
    if args.engine == "fastmace":
        import fastmace  # noqa: F401 -- required before torch for soft file locks
        from fastmace.model.compile import configure_allocator

        configure_allocator()
    import torch

    set_tf32_off(torch)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    atoms = structure(args.case, args.structures)
    record = common_record(args, atoms, torch)
    sampler = None
    try:
        if args.kind == "validate":
            sampler = GPUSampler()
            sampler.start()
            {"reference": validate_reference, "fastmace": validate_fastmace, "alchemi": validate_alchemi}[
                args.engine
            ](args, atoms, record, torch)
            record["gpu_samples"] = sampler.stop()
            sampler = None
        elif args.engine == "fastmace":
            timing_fastmace(args, atoms, record, torch)
        elif args.engine == "alchemi":
            timing_alchemi(args, atoms, record, torch)
        else:
            raise ValueError("reference is validation-only")
        record["status"] = "complete"
        return_code = 0
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return_code = 1
        raise
    finally:
        if sampler is not None:
            record["gpu_samples"] = sampler.stop()
        record["finished_utc"] = utc_now()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        summary_row = {k: v for k, v in record.items() if k not in ("gpu_samples", "validation")}
        print("RESULT " + json.dumps(summary_row), flush=True)
    return return_code


def reserved() -> bool:
    marker = Path(os.path.expanduser(os.environ.get("FASTMACE_LOCAL_GPU_MARKER", "~/.fastmace-local-gpu")))
    try:
        holder = json.loads((marker / "holder.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(holder.get("quiet"))


def run_child(base: list[str], output: Path) -> None:
    if output.exists():
        try:
            if json.loads(output.read_text()).get("status") == "complete":
                return
        except (OSError, ValueError):
            pass
    completed = subprocess.run(base + ["--output", str(output)], text=True, capture_output=True)
    log = output.with_suffix(".log")
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"child failed ({completed.returncode}); see {log}")


def compare_validation(output_dir: Path, case: str) -> dict:
    import numpy as np

    records = {
        engine: json.loads((output_dir / f"validate-{case}-{engine}.json").read_text())
        for engine in ("reference", "fastmace", "alchemi")
    }
    reference = records["reference"]["validation"]
    ref_forces = np.asarray(reference["forces_eV_A"])
    result = {"case": case, "reference_finite": reference["finite"], "sides": {}}
    for engine in ("fastmace", "alchemi"):
        current = records[engine]["validation"]
        difference = np.asarray(current["forces_eV_A"]) - ref_forces
        result["sides"][engine] = {
            "finite": current["finite"],
            "energy_error_eV": current["energy_eV"] - reference["energy_eV"],
            "energy_error_eV_atom": (
                (current["energy_eV"] - reference["energy_eV"]) / records[engine]["input"]["natoms"]
            ),
            "force_max_abs_error_eV_A": float(np.max(np.abs(difference))),
            "force_rms_error_eV_A": float(np.sqrt(np.mean(difference**2))),
        }
    result["passed"] = all(side["finite"] for side in result["sides"].values()) and result["reference_finite"]
    return result


def suite(args) -> int:
    if not args.allow_unreserved and not reserved():
        raise RuntimeError("suite requires a quiet local_gpu.py reservation")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--checkpoint",
        str(args.checkpoint),
        "--structures",
        str(args.structures),
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--blocks",
        str(args.blocks),
    ]
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "started_utc": utc_now(),
        "protocol": {"warmup_steps": args.warmup, "timed_blocks": args.blocks, "steps_per_block": args.steps},
        "validation": [],
        "timing_order": [],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for case in CASES:
        for engine in ("reference", "fastmace", "alchemi"):
            output = output_dir / f"validate-{case}-{engine}.json"
            run_child(common + ["--kind", "validate", "--engine", engine, "--case", case], output)
        comparison = compare_validation(output_dir, case)
        manifest["validation"].append(comparison)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if not comparison["passed"]:
            raise RuntimeError(f"{case}: finite static validation failed")
    for round_number, order in ((1, ("fastmace", "alchemi")), (2, ("alchemi", "fastmace"))):
        for case in CASES:
            for engine in order:
                output = output_dir / f"timing-r{round_number}-{case}-{engine}.json"
                manifest["timing_order"].append({"round": round_number, "case": case, "engine": engine})
                (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                run_child(
                    common
                    + ["--kind", "timing", "--engine", engine, "--case", case, "--round", str(round_number)],
                    output,
                )
    manifest.update(status="complete", finished_utc=utc_now())
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--kind", choices=("validate", "timing"), required=True)
    worker_parser.add_argument("--engine", choices=("reference", "fastmace", "alchemi"), required=True)
    worker_parser.add_argument("--case", choices=tuple(CASES), required=True)
    worker_parser.add_argument("--round", type=int, default=0)
    worker_parser.add_argument("--output", type=Path, required=True)
    suite_parser = commands.add_parser("suite")
    suite_parser.add_argument("--output-dir", type=Path, required=True)
    suite_parser.add_argument("--allow-unreserved", action="store_true")
    for command in (worker_parser, suite_parser):
        command.add_argument("--checkpoint", type=Path, default=Path("~/models/mace-omat-0-medium.model"))
        command.add_argument(
            "--structures",
            type=Path,
            default=Path("~/fastmace-baseline/1a-T18/Structures/Benchmark/R2SCAN"),
        )
        command.add_argument("--warmup", type=int, default=300)
        command.add_argument("--steps", type=int, default=300)
        command.add_argument("--blocks", type=int, default=4)
    return root


def main() -> int:
    args = parser().parse_args()
    args.checkpoint = args.checkpoint.expanduser()
    args.structures = args.structures.expanduser()
    return worker(args) if args.command == "worker" else suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
