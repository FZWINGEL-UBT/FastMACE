"""The gate on the user's structure (SPEC §12.3; plan task 1b-T16).

The gate is opt-in (owner decision D1): ``check="skip"``, the default, runs nothing and records
``skipped``; ``check="gate"`` runs it; ``check="allow_unavailable"`` runs it too, but a gate whose
reference cannot run is recorded and the run goes on. It has two halves, which ``Simulation`` calls
around the build (``docs/design/contracts-1b.md``):

1. ``references(plan, force_field, atoms, integrator, *, check, seed)``, after the plan and before
   any production engine: the copies (2 rattled by 0.05 Å; for a variable-cell integrator also 2
   strained by up to ±2 %, then rattled), and on each the float64 reference and the float32
   yardstick with float64 energy accumulation of §12.1 (``fastmace.reference.calculators``), one
   calculator at a time, freed before the next and before the engine is built. A reference that
   cannot run (out of memory, or failing) makes the gate ``unavailable``: ``GateUnavailable``
   under ``check="gate"``, before any engine is built.
2. ``production(engine, references, *, check) -> GateRecord``, through the plan's engine in the mode
   it runs: the registry snapshots the runtime state, the gate loads each copy, rebuilds its
   neighbour list, replays stages C and B (captured graphs in graph mode) and reads energy, forces
   and stress from the state; then the registry restores the state, bit for bit. A failed gate
   raises ``GateFailed`` (the §9.4 fallback is phase 2).

The rule is §12.2's ratio rule: for each quantity the production path's largest error against the
float64 reference, over all copies, is at most twice the yardstick's, or at most the quantity's
floor (``FLOORS``). Any NaN or inf fails before a maximum is taken: Python's ``max`` can drop a NaN
depending on order, and an infinite yardstick would make the limit infinite. An NVT plan has no
stress path; its stress is recorded as "stress: not in plan".

References and process state. Production engines patch process-wide state that moved a float64
reference by up to 7.8e-7 eV/Å (1a "after T10"). So the references run in this process only while
no production engine has been built in it (``engine.base.production_engine_built``); afterwards
they run in a child process in the same environment (``python -m fastmace.planner.selfcheck``),
which builds no production engine. The record says which way they ran. For a hybrid plan the OEq
runtime is prepared before a reference imports MACE.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from fastmace.errors import FastMACEError, GateFailed, GateUnavailable

CHECK_MODES = ("skip", "gate", "allow_unavailable")
STATUSES = ("passed", "passed-reduced", "failed", "unavailable", "skipped")  # "passed-reduced": phase 2
SKIPPED_REASON = 'opt-in: check="gate"'
RATIO = 2.0
QUANTITIES = ("energy_meV_per_atom", "force_eV_per_A", "stress_eV_per_A3")
# The absolute floors of §12.3: 0.01 meV/atom, 1e-5 eV/Å (largest force component), 1e-7 eV/Å^3
# (largest stress component, about 16 kPa).
FLOORS = MappingProxyType({"energy_meV_per_atom": 0.01, "force_eV_per_A": 1e-5, "stress_eV_per_A3": 1e-7})
NO_STRESS = "stress: not in plan"
_NAMES = MappingProxyType({"energy_meV_per_atom": "energy per atom", "force_eV_per_A": "force",
                           "stress_eV_per_A3": "stress"})
_UNITS = MappingProxyType({"energy_meV_per_atom": "meV/atom", "force_eV_per_A": "eV/Å", "stress_eV_per_A3": "eV/Å^3"})
COPIES = 2  # rattled copies, and strained copies for a variable cell
RATTLE_A = 0.05
STRAIN = 0.02
ROLES = (("reference", "float64"), ("yardstick", "float32"))
_SEED_SALT = 0x6A7E  # the copies' generator is not the integrator's noise stream


class GateJobError(FastMACEError, RuntimeError):
    """The gate's reference child process failed or left no result."""


def check_mode(check: Any) -> str:
    """``check`` if it is one of ``CHECK_MODES``; else ``ValueError`` naming them."""
    if check not in CHECK_MODES:
        raise ValueError(f"check={check!r} is not one of {CHECK_MODES}; the gate is opt-in: check='gate' (SPEC §12.3)")
    return check


# --------------------------------------------------------------------------- the record


@dataclasses.dataclass(frozen=True)
class GateRecord:
    """One gate result (SPEC §12.3). ``to_metadata()`` gives the run report's keys and the rest.

    The report reads ``status``, ``reason``, ``copies`` and the production path's largest error per
    quantity: ``energy_meV_per_atom``, ``force_eV_per_A`` and ``stress_eV_per_A3`` (``None`` where
    not measured, the stress of an NVT plan included).
    """

    status: str
    check: str
    reason: str | None = None
    copies: str | None = None
    energy_meV_per_atom: float | None = None
    force_eV_per_A: float | None = None
    stress_eV_per_A3: float | None = None
    quantities: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    nonfinite: tuple[str, ...] = ()
    mode: str | None = None
    references: Mapping[str, Any] | None = None
    seconds: Mapping[str, float] = dataclasses.field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"gate status {self.status!r} is not one of {STATUSES}")

    @property
    def passed(self) -> bool:
        return self.status in ("passed", "passed-reduced")

    def to_metadata(self) -> dict[str, Any]:
        return _plain(dataclasses.asdict(self))


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def skipped_record() -> GateRecord:
    """The record of a gate that was not requested (owner decision D1)."""
    return GateRecord(status="skipped", check="skip", reason=SKIPPED_REASON)


# --------------------------------------------------------------------------- the rule


def copy_errors(reference, candidate, natoms: int) -> dict[str, float | None]:
    """Errors of one copy against the reference: energy per atom (meV), largest force and stress component.

    ``reference`` and ``candidate`` are ``(energy, forces, stress)``; a ``None`` stress on either side
    gives ``None`` (not measured).
    """
    energy = abs(float(candidate[0]) - float(reference[0])) / natoms * 1e3
    forces = float(np.abs(np.asarray(candidate[1], float) - np.asarray(reference[1], float)).max())
    stress = None
    if candidate[2] is not None and reference[2] is not None:
        stress = float(np.abs(np.asarray(candidate[2], float) - np.asarray(reference[2], float)).max())
    return {"energy_meV_per_atom": energy, "force_eV_per_A": forces, "stress_eV_per_A3": stress}


def _nonfinite(rows):
    bad = []
    for index, row in enumerate(rows):
        for path in ("production", "yardstick"):
            for quantity in QUANTITIES:
                value = row[path][quantity]
                if value is not None and not math.isfinite(value):
                    bad.append(f"{index}/{path}/{quantity}")
    return bad


def verdict(rows, ratio: float = RATIO) -> dict[str, Any]:
    """The ratio rule over the copies' errors (``copy_errors`` per path): per quantity and overall.

    Each quantity passes when the production path's largest error is at most
    ``max(ratio x the yardstick's largest error, floor)``. A non-finite error anywhere fails first.
    """
    bad = _nonfinite(rows)
    if bad:
        return {"passed": False, "nonfinite": bad, "quantities": {}}
    quantities = {}
    for quantity in QUANTITIES:
        production = [row["production"][quantity] for row in rows if row["production"][quantity] is not None]
        if not production:
            quantities[quantity] = {"passed": True, "note": NO_STRESS}
            continue
        worst = max(production)
        yardstick = max(row["yardstick"][quantity] for row in rows if row["yardstick"][quantity] is not None)
        limit = max(ratio * yardstick, FLOORS[quantity])
        quantities[quantity] = {"production": worst, "yardstick": yardstick, "ratio": worst / yardstick if yardstick
                                else None, "floor": FLOORS[quantity], "limit": limit, "passed": bool(worst <= limit)}
    return {"passed": all(q["passed"] for q in quantities.values()), "nonfinite": [], "quantities": quantities}


# --------------------------------------------------------------------------- the copies


@dataclasses.dataclass(frozen=True)
class GateCopy:
    kind: str  # "rattled" or "strained"
    atoms: Any


def _integrator_config(integrator, seed=0):
    from fastmace.config import IntegratorConfig

    if isinstance(integrator, IntegratorConfig):
        return integrator
    if hasattr(integrator, "to_config"):
        return integrator.to_config(seed=seed)
    raise TypeError(f"the gate needs an IntegratorConfig or an fm.integrators record, got {type(integrator).__name__}")


def _strain(rng, amplitude, full):
    """Symmetric strain: isotropic (hydrostatic) or isotropic, anisotropic and shear (full), uniform in ±amplitude."""
    if not full:
        return np.identity(3) * (1 + rng.uniform(-amplitude, amplitude))
    e = rng.uniform(-amplitude, amplitude, (3, 3))
    return np.identity(3) + 0.5 * (e + e.T)


def make_copies(atoms, integrator, seed: int) -> list[GateCopy]:
    """The gate's copies of ``atoms``: 2 rattled (Gaussian, 0.05 Å); a variable cell adds 2 strained and rattled.

    Strains as 1a-T19's gate driver draws them (``tools/force_stress_gate.py``), within ±2 %;
    isotropic for ``cell_mode="hydrostatic"``. One generator from ``seed``.
    """
    config = _integrator_config(integrator, seed)
    rng = np.random.default_rng([int(seed), _SEED_SALT])
    copies = []
    for _ in range(COPIES):
        copy = atoms.copy()
        copy.positions += RATTLE_A * rng.standard_normal(copy.positions.shape)
        copies.append(GateCopy("rattled", copy))
    if config.ensemble == "npt":
        for _ in range(COPIES):
            copy = atoms.copy()
            copy.set_cell(copy.cell.array @ _strain(rng, STRAIN, config.cell_mode == "full"), scale_atoms=True)
            copy.positions += RATTLE_A * rng.standard_normal(copy.positions.shape)
            copies.append(GateCopy("strained", copy))
    return copies


def describe_copies(copies, cell_mode: str | None = None) -> str:
    """"2 rattled (0.05 Å)" or "2 rattled (0.05 Å), 2 strained (±2 %, hydrostatic)"."""
    rattled = sum(copy.kind == "rattled" for copy in copies)
    strained = sum(copy.kind == "strained" for copy in copies)
    text = f"{rattled} rattled ({RATTLE_A:g} Å)"
    if strained:
        mode = cell_mode or ("hydrostatic" if all(_isotropic(copy.atoms, copies[0].atoms) for copy in copies
                                                  if copy.kind == "strained") else "full")
        text += f", {strained} strained (±{STRAIN * 100:g} %, {mode})"
    return text


def _isotropic(strained, rattled):
    strain = np.linalg.solve(rattled.cell.array, strained.cell.array)
    return bool(np.allclose(strain, strain[0, 0] * np.identity(3), atol=1e-12))


# --------------------------------------------------------------------------- references


@dataclasses.dataclass
class References:
    """What ``references`` computed, for ``production``: the copies and each one's reference values.

    ``values[i]`` maps "reference" and "yardstick" to ``(energy, forces, stress)``; ``ran`` is
    "in-process" or "child process"; ``unavailable`` is the record when no reference could run.
    """

    check: str
    copies: list = dataclasses.field(default_factory=list)
    description: str | None = None
    values: list = dataclasses.field(default_factory=list)
    ran: str | None = None
    info: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    seconds: float = 0.0
    unavailable: GateRecord | None = None

    @property
    def skipped(self) -> bool:
        return self.check == "skip"


def _job(plan, force_field, copies):
    """Everything a reference needs, as JSON (for a child process too)."""
    model = force_field.model
    polar = None if plan.polar is None else dataclasses.asdict(plan.polar)
    return {
        "source": model.label or str(model.path), "head": None if model.label else model.head,
        "sha256": model.sha256, "charge": model.charge, "spin_multiplicity": model.spin_multiplicity,
        "family": model.family, "device": plan.engine.device, "backend": plan.engine.backend, "polar": polar,
        "copies": [{"kind": copy.kind, "numbers": copy.atoms.numbers.tolist(),
                    "positions": copy.atoms.positions.tolist(), "cell": copy.atoms.cell.array.tolist()}
                   for copy in copies],
    }


def _atoms(entry):
    from ase import Atoms

    return Atoms(numbers=entry["numbers"], positions=entry["positions"], cell=entry["cell"], pbc=True)


def _polar_config(job):
    if job["polar"] is None:
        return None
    from fastmace.config import PolarConfig

    options = dict(job["polar"])
    if options.get("dipole_cap") is not None:
        options["dipole_cap"] = tuple(options["dipole_cap"])
    return PolarConfig(**options)


def _load_reference(job, dtype, atoms):
    """One corrected reference calculator (``fastmace.reference.calculators.load``) for ``job``'s model."""
    from fastmace.reference import calculators

    return calculators.load(job["source"], atoms, dtype=dtype, device=job["device"], head=job["head"],
                            family=job["family"], polar=_polar_config(job), sha256=job["sha256"],
                            charge=job["charge"], spin_multiplicity=job["spin_multiplicity"])


def _free(device):
    gc.collect()
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


def _evaluate_one(reference, atoms):
    from fastmace.reference.records import ReferencePolicyError

    try:
        return reference.evaluate(atoms)
    except ReferencePolicyError as error:
        if not (reference.is_polar and "overflowed" in error.what):
            raise
        reference.rebase(atoms)  # a strained cell outside the k-space headroom: size it for this cell
        return reference.evaluate(atoms)


def evaluate_references(job) -> tuple[list[dict], dict]:
    """The reference and the yardstick on every copy of ``job``: one calculator at a time, freed after use."""
    entries = [_atoms(entry) for entry in job["copies"]]
    values = [{} for _ in entries]
    info = {}
    if job.get("backend") == "hybrid":
        from fastmace.model import loading

        info["oeq_runtime"] = loading.prepare_hybrid_if_possible()  # before a reference imports MACE
    for role, dtype in ROLES:
        reference = _load_reference(job, dtype, entries[0])
        try:
            for index, atoms in enumerate(entries):
                energy, forces, stress = _evaluate_one(reference, atoms)
                values[index][role] = (float(energy), np.asarray(forces, float), np.asarray(stress, float))
            info[role] = json.loads(json.dumps(reference.info(), default=str))
        finally:
            del reference
            _free(job["device"])
    return values, info


def _reason(error) -> str:
    text = getattr(error, "what", None) or (str(error).splitlines() or [""])[0]
    kind = "out of memory" if "out of memory" in text.lower() or isinstance(error, MemoryError) else "failed"
    return f"reference {kind}: {type(error).__name__}: {text}" if text else f"reference {kind}: {type(error).__name__}"


def references(plan, force_field, atoms, integrator, *, check: str, seed: int) -> References:
    """The gate's first half: copies and their references, before any production engine (see module docstring).

    ``check="skip"`` computes nothing. ``GateUnavailable`` under ``check="gate"`` when no
    reference could run; under ``check="allow_unavailable"`` that is recorded in the result.
    """
    from fastmace.engine import base
    from fastmace.reference.calculators import ReferenceOrderError

    check_mode(check)
    if check == "skip":
        return References(check=check)
    integrator = plan.integrator if integrator is None else integrator
    config = _integrator_config(integrator, seed)
    copies = make_copies(atoms, config, seed)
    description = describe_copies(copies, config.cell_mode if config.ensemble == "npt" else None)
    job = _job(plan, force_field, copies)
    started = time.perf_counter()
    ran = "child process" if base.production_engine_built() else "in-process"
    try:
        values, info = run_child(job) if ran == "child process" else evaluate_references(job)
    except ReferenceOrderError:
        raise  # a correction is not the active code path: a defect, not an unavailable reference
    except Exception as error:  # out of memory, or a reference failing: the gate is unavailable (§12.3)
        record = GateRecord(status="unavailable", check=check, reason=_reason(error), copies=description,
                            references={"ran": ran}, seconds={"references_s": time.perf_counter() - started})
        if check == "gate":
            raise GateUnavailable(
                "the gate's reference could not run", why=record.reason,
                next_step='Free GPU memory and retry, or pass check="allow_unavailable" to run without the gate '
                          "(recorded).", gate_record=record) from error
        return References(check=check, copies=copies, description=description, ran=ran,
                          unavailable=record)
    return References(check=check, copies=copies, description=description, values=values, ran=ran, info=info,
                      seconds=time.perf_counter() - started)


# --------------------------------------------------------------------------- the child process


def _scratch_directory():
    from fastmace import _cache

    root = _cache.cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="gate-", dir=root))


def _child_environment():
    import fastmace

    env = dict(os.environ)
    checkout = str(Path(fastmace.__file__).resolve().parents[1])  # the same fastmace as this process
    env["PYTHONPATH"] = os.pathsep.join(part for part in (checkout, env.get("PYTHONPATH")) if part)
    return env


def run_child(job, *, timeout_s: float | None = None) -> tuple[list[dict], dict]:
    """``evaluate_references(job)`` in a fresh process of this environment, which builds no production engine."""
    directory = _scratch_directory()
    try:
        job_path, result_path = directory / "job.json", directory / "result.npz"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        argv = [sys.executable, "-m", "fastmace.planner.selfcheck", str(job_path), str(result_path)]
        done = subprocess.run(argv, env=_child_environment(), capture_output=True, text=True, timeout=timeout_s,
                              check=False)
        if done.returncode != 0 or not result_path.is_file():
            tail = "\n".join((done.stderr or done.stdout or "").strip().splitlines()[-5:])
            raise GateJobError(f"the reference child process failed (exit code {done.returncode})", why=tail or None)
        return _read_result(result_path)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _write_result(path, values, info):
    arrays = {"info": np.array(json.dumps(info))}
    for index, entry in enumerate(values):
        for role, (energy, forces, stress) in entry.items():
            arrays[f"{index}/{role}/energy"] = np.array(energy, dtype=np.float64)
            arrays[f"{index}/{role}/forces"] = forces
            arrays[f"{index}/{role}/stress"] = stress
    np.savez(path, **arrays)


def _read_result(path):
    with np.load(path, allow_pickle=False) as data:
        info = json.loads(str(data["info"]))
        count = 1 + max(int(key.split("/")[0]) for key in data.files if key != "info")
        values = [{role: (float(data[f"{index}/{role}/energy"]), data[f"{index}/{role}/forces"].copy(),
                          data[f"{index}/{role}/stress"].copy()) for role, _ in ROLES} for index in range(count)]
    return values, info


def _child_main(argv) -> int:
    job_path, result_path = argv
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    values, info = evaluate_references(job)
    info["process"] = "child"
    _write_result(result_path, values, info)
    return 0


# --------------------------------------------------------------------------- production


def _read_outputs(engine, stress: bool):
    """Energy, forces and (``stress``) the stress the last stage B wrote into the engine's state, as host copies."""
    state = engine.state
    energy = float(state.energy.detach().reshape(-1)[0])
    forces = state.forces.detach().double().cpu().numpy().copy()
    values = state.stress.detach().double().cpu().numpy().copy() if stress else None
    return energy, forces, values


def _host_attributes(obj):
    """The non-tensor attributes of ``obj`` (by reference), which a neighbour rebuild reassigns."""
    import torch

    return {name: value for name, value in vars(obj).items() if not torch.is_tensor(value)}


def _restore_attributes(obj, saved):
    for name in [name for name in vars(obj) if name not in saved and not _is_tensor(getattr(obj, name))]:
        delattr(obj, name)
    for name, value in saved.items():
        setattr(obj, name, value)


def _is_tensor(value):
    import torch

    return torch.is_tensor(value)


def _quiescent(engine):
    if getattr(engine, "_pipe_pending", None) is not None or getattr(engine, "_pending", None):
        raise RuntimeError("the gate runs on a quiescent engine: before the first step or after flush()")


def _replay(engine, copies):
    """Each copy through the engine's stages C and B (captured in graph mode); the state restored afterwards.

    Returns the outputs per copy and the notes (a neighbour capacity that grew).
    """
    import torch

    from fastmace.engine.registry import _identity

    _quiescent(engine)
    field, state = engine.force_field, engine.state
    neighbours = field.neighbours
    stress = state.stress is not None and field.compute_stress
    snapshot = engine.registry.snapshot()
    host = _host_attributes(neighbours)
    edges_before = int(field.data["edge_index"].shape[1])
    sizes_used = set(engine.edge_sizes_used)
    grew = False
    outputs = []
    try:
        for copy in copies:
            with torch.no_grad():
                state.positions.copy_(torch.as_tensor(copy.atoms.positions, dtype=state.positions.dtype,
                                                      device=state.positions.device))
                if engine.variable_cell:
                    state.cell.copy_(torch.as_tensor(copy.atoms.cell.array, dtype=state.cell.dtype,
                                                     device=state.cell.device))
            if engine.variable_cell:
                field.set_cell(state.cell)
            if field.rebuild(state.positions):  # a grown capacity moved the buffers: every graph is stale
                grew = True
                engine._clear_graphs()
            engine._run("C")
            edges = engine._padded(int(field.edge_count))
            engine._use_edges(edges)
            engine._run(("B", edges))
            outputs.append(_read_outputs(engine, stress))
    finally:
        notes = []
        if not grew:
            engine.registry.restore(snapshot)
            _restore_attributes(neighbours, host)
        else:
            # The neighbour buffers moved, so they cannot be restored in place; every other buffer is.
            # Their policy is "reset by a rebuild": rebuild the list at the restored positions and cell.
            current = engine.registry.tensors()
            with torch.no_grad():
                for name, (identity, value) in snapshot.values.items():
                    if name in current and _identity(current[name]) == identity:
                        current[name].copy_(value)
            if engine.variable_cell:
                field.set_cell(state.cell)
            field.rebuild(state.positions)
            field.compact(state.positions)
            neighbours.rebuilds = host["rebuilds"]
            notes.append("the neighbour capacity grew on a gate copy: the list was rebuilt at the committed "
                         "positions with the larger capacity, and the graphs are captured again")
        field.use_edges(edges_before)
        engine.edge_sizes_used.clear()
        engine.edge_sizes_used.update(sizes_used)
    return outputs, notes


def production(engine, references: References, *, check: str) -> GateRecord:
    """The gate's second half: the production path on the copies, the ratio rule, and the record.

    Raises ``GateFailed`` when the gate fails. Returns the ``skipped`` record for ``check="skip"``
    and the ``unavailable`` record for ``check="allow_unavailable"`` when no reference ran.
    """
    check_mode(check)
    if check != references.check:
        raise ValueError(f"production(check={check!r}) differs from the references' check={references.check!r}")
    if references.skipped:
        return skipped_record()
    if references.unavailable is not None:
        return references.unavailable
    started = time.perf_counter()
    outputs, notes = _replay(engine, references.copies)
    natoms = len(references.copies[0].atoms)
    rows = []
    for output, values in zip(outputs, references.values, strict=True):
        reference, yardstick = values["reference"], values["yardstick"]
        if output[2] is None:
            yardstick = (yardstick[0], yardstick[1], None)
        rows.append({"production": copy_errors(reference, output, natoms),
                     "yardstick": copy_errors(reference, yardstick, natoms)})
    result = verdict(rows)
    worst = {quantity: result["quantities"].get(quantity, {}).get("production") for quantity in QUANTITIES}
    if result["nonfinite"]:
        reason = "non-finite values: " + ", ".join(result["nonfinite"])
    else:
        failed = [(quantity, q) for quantity, q in result["quantities"].items() if not q["passed"]]
        reason = "; ".join(f"{_NAMES[quantity]} error {q['production']:.3g} {_UNITS[quantity]} above the limit "
                           f"{q['limit']:.3g} (2 x float32 yardstick {q['yardstick']:.3g}, floor {q['floor']:g})"
                           for quantity, q in failed) or None
    mode = ("captured replay of stages C, B" if getattr(engine, "capture", False) else "eager stages C, B")
    record = GateRecord(
        status="passed" if result["passed"] else "failed", check=check, reason=reason,
        copies=references.description, quantities=result["quantities"], nonfinite=tuple(result["nonfinite"]),
        mode=mode, references={"ran": references.ran, "record_id": references.info["reference"]["record_id"],
                               **references.info},
        seconds={"references_s": references.seconds, "production_s": time.perf_counter() - started},
        notes=tuple(notes), **worst)
    if not record.passed:
        raise GateFailed(f"the gate on this structure failed: {reason}",
                         why="the production path's error against the float64 reference exceeds the ratio rule "
                             "of SPEC §12.3 (no fallback in phase 1b).", gate_record=record)
    return record


__all__ = ["CHECK_MODES", "FLOORS", "GateCopy", "GateRecord", "References", "check_mode", "copy_errors",
           "describe_copies", "evaluate_references", "make_copies", "production", "references", "run_child",
           "skipped_record", "verdict"]


if __name__ == "__main__":
    sys.exit(_child_main(sys.argv[1:]))
