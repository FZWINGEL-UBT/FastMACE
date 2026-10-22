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
which builds no production engine. On a CUDA plan every GPU reference step runs in a child (2-T05):
a GPU out-of-memory can leave the CUDA context unusable (on the RTX 5060 Ti it surfaced as "CUDA
driver error: device not ready", and later CUDA calls in the process failed), and the child's exit
frees its memory before the engine is built. A GPU reference child that fails with a CUDA error, a
non-zero exit or a kill is a "device failure", and the ladder goes on. The record says which way
they ran. For a hybrid plan the OEq runtime is prepared before a reference imports MACE.

Resources (SPEC §12.3 "Resources"; plan task 2-T05). The references try, in order, until one runs:

1. the plan's device on the production cell;
2. after an out-of-memory on a GPU: the CPU in float64 on the production cell, in a child process
   (so a kill for host memory is a recorded failure), if the estimated time is within
   ``GateConfig.cpu_float64_time_limit_s`` and the estimated memory within ``MemAvailable``. The
   child calibrates first: two periodic boxes cut from the structure, timed, give a rate per edge
   (plus one per atom x volume for PolarMACE's k-space), measured in that process and never stored;
3. the reduced cell (``reduced_cells``) on the plan's device, walking down in size after an
   out-of-memory; its status is ``passed-reduced``;
4. the reduced cell on the CPU, as in step 2.
A cell is never larger than one that ran out of memory. A reference failing for any other reason
makes the gate ``unavailable`` at once (decision 3). On a reduced cell the production path is
evaluated eagerly by a separate force field converted from the pristine model, with the plan's
rewrites and precision, and the replay is checked on the production cell: the engine's captured
replay against a direct evaluation of the same force field at the same state, within the floors.
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
STATUSES = ("passed", "passed-reduced", "failed", "unavailable", "skipped")
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
# Resources (2-T05). Atoms match a translated atom of the same species within this distance when the
# primitive cell is found; a 0.05 Å rattle stays far inside, a disordered structure far outside.
REPEAT_TOLERANCE_A = 0.5
# After an out-of-memory on a reduced cell, the next one tried has at most this fraction of its atoms.
REDUCED_SHRINK = 0.75
# The CPU step runs only if its estimated memory times this margin is within MemAvailable.
MEMORY_MARGIN = 1.25
# A CPU reference child caps its own data segment (RLIMIT_DATA) at the smaller of MemAvailable less
# this reserve and its estimate times MEMORY_CAP_FACTOR: on WSL2 a host out-of-memory takes down
# the whole VM, not just the child (2-T05, 20 October 2026).
MEMORY_RESERVE_BYTES = 2 * 2**30
MEMORY_CAP_FACTOR = 3.0
# The calibration boxes: cubes of this many cutoffs on a side, and one of twice the volume.
CALIBRATION_SIDE_CUTOFFS = 2.0
OUTCOMES = ("ran", "out of memory", "device failure", "failed", "declined", "skipped")


class GateJobError(FastMACEError, RuntimeError):
    """The gate's reference child process failed or left no result; ``returncode`` is its exit code."""

    def __init__(self, what, why=None, next_step=None, *, returncode=None, **kwargs):
        super().__init__(what, why, next_step, **kwargs)
        self.returncode = returncode


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
    # 2-T05: the device the references ran on; the reduced cell's repeat of the primitive cell
    # (``passed-reduced``) or None on the production cell; each tried step (device, cell, repeat,
    # atoms, seconds, outcome, reason); the replay check on the production cell under a reduced gate.
    device: str | None = None
    reduced_cell: tuple[int, int, int] | None = None
    steps: tuple[Mapping[str, Any], ...] = ()
    replay: Mapping[str, Any] | None = None

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


def _nonfinite(rows, quantities=QUANTITIES):
    bad = []
    for index, row in enumerate(rows):
        for path in ("production", "yardstick"):
            for quantity in quantities:
                value = row[path][quantity]
                if value is not None and not math.isfinite(value):
                    bad.append(f"{index}/{path}/{quantity}")
    return bad


def verdict(rows, ratio: float = RATIO, quantities=QUANTITIES) -> dict[str, Any]:
    """The ratio rule over the copies' errors (``copy_errors`` per path): per quantity and overall.

    Each quantity passes when the production path's largest error is at most
    ``max(ratio x the yardstick's largest error, floor)``. A non-finite error anywhere fails first.
    ``quantities`` selects the quantities judged (T15's reduced gate judges the failing one).
    """
    selected = tuple(quantities)
    unknown = [quantity for quantity in selected if quantity not in QUANTITIES]
    if unknown:
        raise ValueError(f"unknown gate quantities {unknown}; they are {QUANTITIES}")
    bad = _nonfinite(rows, selected)
    if bad:
        return {"passed": False, "nonfinite": bad, "quantities": {}}
    quantities = {}
    for quantity in selected:
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


def make_copies(atoms, integrator, seed: int, count: int = COPIES) -> list[GateCopy]:
    """The gate's copies of ``atoms``: 2 rattled (Gaussian, 0.05 Å); a variable cell adds 2 strained and rattled.

    Strains as 1a-T19's gate driver draws them (``tools/force_stress_gate.py``), within ±2 %;
    isotropic for ``cell_mode="hydrostatic"``. One generator from ``seed``. ``count`` copies of each
    kind (T15's reduced gate takes one).
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"count must be a positive int, got {count!r}")
    config = _integrator_config(integrator, seed)
    rng = np.random.default_rng([int(seed), _SEED_SALT])
    copies = []
    for _ in range(count):
        copy = atoms.copy()
        copy.positions += RATTLE_A * rng.standard_normal(copy.positions.shape)
        copies.append(GateCopy("rattled", copy))
    if config.ensemble == "npt":
        for _ in range(count):
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


# --------------------------------------------------------------------------- the reduced cell


@dataclasses.dataclass(frozen=True)
class ReducedCell:
    """A supercell ``repeat`` of the primitive cell of a structure that is ``primitive_repeat`` of it."""

    primitive_repeat: tuple[int, int, int]
    repeat: tuple[int, int, int]
    atoms: Any = dataclasses.field(compare=False, repr=False)

    @property
    def natoms(self) -> int:
        return len(self.atoms)


def heights(cell) -> tuple[float, float, float]:
    """The three heights of a cell (its volume over the area of each pair of the other vectors), Å."""
    cell = np.asarray(cell, float)
    volume = abs(float(np.linalg.det(cell)))
    return tuple(volume / float(np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))) for i in range(3))


def _divisors(n):
    return [d for d in range(n, 1, -1) if n % d == 0]


def _nearest(points, sites, cell, tolerance):
    """For each of ``points`` the index of the nearest of ``sites`` (fractional, periodic), or None if one is
    farther than ``tolerance`` Å."""
    diff = points[:, None, :] - sites[None, :, :]
    diff -= np.round(diff)
    distance = np.linalg.norm(diff @ cell, axis=-1)
    nearest = distance.argmin(axis=1)
    return None if distance[np.arange(len(nearest)), nearest].max() > tolerance else nearest


def _translation(fractional, numbers, cell, shift, tolerance):
    """The permutation taking each atom to the atom of its species at its position + ``shift`` (fractional),
    or None when some atom has none within ``tolerance`` Å or two atoms share one."""
    permutation = np.full(len(numbers), -1)
    for z in np.unique(numbers):
        index = np.flatnonzero(numbers == z)
        sites = fractional[index]
        moved = sites + shift
        if _nearest(moved[:1], sites, cell, tolerance) is None:  # most shifts fail on the first atom
            return None
        for start in range(0, len(index), 256):
            nearest = _nearest(moved[start:start + 256], sites, cell, tolerance)
            if nearest is None:
                return None
            permutation[index[start:start + len(nearest)]] = index[nearest]
    if len(np.unique(permutation)) != len(permutation):
        return None
    return permutation


def _primitive(atoms, tolerance=REPEAT_TOLERANCE_A):
    """``(repeat, primitive atoms)``: the largest repeat along each cell vector and one cell of it.

    Along each vector the largest n is taken for which a translation by 1/n of it maps the structure
    onto itself (same species, within ``tolerance``). The translations' orbits must all have
    n1 n2 n3 atoms; one atom of each, shifted into the first cell, is the primitive content.
    """
    from ase import Atoms

    cell = np.asarray(atoms.cell.array, float)
    numbers = np.asarray(atoms.numbers)
    natoms = len(numbers)
    fractional = atoms.get_scaled_positions(wrap=True)
    lengths = np.linalg.norm(cell, axis=1)
    repeat, permutations = [1, 1, 1], []
    for axis in range(3):
        for n in _divisors(natoms):
            if lengths[axis] / n < 1.0:  # no crystal repeats within an Ångström
                continue
            shift = np.zeros(3)
            shift[axis] = 1.0 / n
            permutation = _translation(fractional, numbers, cell, shift, tolerance)
            if permutation is not None:
                repeat[axis] = n
                permutations.append(permutation)
                break
    total = repeat[0] * repeat[1] * repeat[2]
    if total == 1:
        return (1, 1, 1), atoms
    parent = list(range(natoms))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for permutation in permutations:
        for i, j in enumerate(permutation):
            a, b = root(i), root(int(j))
            if a != b:
                parent[max(a, b)] = min(a, b)
    orbits = {}
    for i in range(natoms):
        orbits.setdefault(root(i), []).append(i)
    if any(len(members) != total for members in orbits.values()):
        return (1, 1, 1), atoms
    representatives = sorted(min(members) for members in orbits.values())
    counts = np.asarray(repeat, float)
    inner = (fractional[representatives] * counts) % 1.0  # fractional in the primitive cell
    primitive = Atoms(numbers=numbers[representatives], cell=cell / counts[:, None], scaled_positions=inner,
                      pbc=True)
    return tuple(repeat), primitive


def primitive_repeat(atoms, tolerance: float = REPEAT_TOLERANCE_A) -> tuple[int, int, int]:
    """How many primitive cells the structure holds along each of its cell vectors ((1, 1, 1): none smaller)."""
    return _primitive(atoms, tolerance)[0]


def reduced_cells(atoms, cutoff: float, min_height_cutoffs: float = 2.0) -> list[ReducedCell]:
    """The reduced cells of ``atoms``, largest first (SPEC §12.3; plan task 2-T05).

    Each is a diagonal supercell m1 x m2 x m3 of the primitive cell (``primitive_repeat``) with fewer
    atoms than the production cell and every height at least ``min_height_cutoffs`` x ``cutoff``.
    The first is "the" reduced cell; the others, one per size, are the walk after an out-of-memory.
    Of equal sizes the one with the largest smallest height is kept, then the larger first repeats.
    """
    repeat, primitive = _primitive(atoms)
    if repeat == (1, 1, 1):
        return []
    base = heights(primitive.cell.array)
    least = [max(1, math.ceil(min_height_cutoffs * cutoff / h - 1e-9)) for h in base]
    total = repeat[0] * repeat[1] * repeat[2]
    best = {}
    for m1 in range(least[0], repeat[0] + 1):
        for m2 in range(least[1], repeat[1] + 1):
            for m3 in range(least[2], repeat[2] + 1):
                size = m1 * m2 * m3
                if size >= total:
                    continue
                key = (min(m * h for m, h in zip((m1, m2, m3), base, strict=True)), (m1, m2, m3))
                if size not in best or key > best[size]:
                    best[size] = key
    return [ReducedCell(repeat, best[size][1], primitive.repeat(best[size][1])) for size in sorted(best, reverse=True)]


def reduced_cell(atoms, cutoff: float, min_height_cutoffs: float = 2.0) -> ReducedCell | None:
    """The reduced cell of ``atoms`` (the first of ``reduced_cells``), or None."""
    cells = reduced_cells(atoms, cutoff, min_height_cutoffs)
    return cells[0] if cells else None


# --------------------------------------------------------------------------- the CPU estimate


def edge_count(atoms, cutoff: float) -> int:
    """The directed edges of ``atoms`` within ``cutoff`` (the model's message-passing graph)."""
    try:
        from matscipy.neighbours import neighbour_list
    except ImportError:  # pragma: no cover - matscipy comes with MACE
        from ase.neighborlist import neighbor_list as neighbour_list
    return int(len(neighbour_list("i", atoms, float(cutoff))))


def calibration_boxes(atoms, cutoff: float):
    """Two periodic cubes cut from ``atoms`` (sides 2 cutoffs and 2^(1/3) times that) for the CPU calibration.

    Their contents are the structure's atoms in the cube, periodic images included; where the
    cube's faces meet, atoms may sit closer than in the structure, which changes the numbers, not the
    cost per edge. An odd electron count drops one atom of odd atomic number (PolarMACE's parity
    rule for a neutral singlet, §10.4).
    """
    from ase import Atoms

    cell = np.asarray(atoms.cell.array, float)
    inverse = np.linalg.inv(cell)
    boxes = []
    for side in (CALIBRATION_SIDE_CUTOFFS * cutoff, CALIBRATION_SIDE_CUTOFFS * cutoff * 2 ** (1 / 3)):
        corners = np.array([[x, y, z] for x in (0, side) for y in (0, side) for z in (0, side)]) @ inverse
        ranges = [range(math.floor(corners[:, k].min()) - 1, math.ceil(corners[:, k].max()) + 1) for k in range(3)]
        positions, numbers = [], []
        for i in ranges[0]:
            for j in ranges[1]:
                for k in ranges[2]:
                    moved = atoms.positions + np.array([i, j, k]) @ cell
                    inside = np.all((moved >= 0) & (moved < side), axis=1)
                    positions.append(moved[inside])
                    numbers.append(atoms.numbers[inside])
        positions, numbers = np.concatenate(positions), np.concatenate(numbers)
        if int(numbers.sum()) % 2:
            odd = np.flatnonzero(numbers % 2)
            keep = np.ones(len(numbers), bool)
            keep[odd[-1]] = False
            positions, numbers = positions[keep], numbers[keep]
        if len(numbers) < 2:
            raise ValueError(f"a {side:.3g} Å calibration box holds {len(numbers)} atoms")
        boxes.append(Atoms(numbers=numbers, positions=positions, cell=np.identity(3) * side, pbc=True))
    return boxes


def _features(atoms, cutoff):
    return {"edges": edge_count(atoms, cutoff), "atoms": len(atoms),
            "atoms_volume": float(len(atoms) * atoms.get_volume())}


def estimate_cost(points, target, *, polar: bool) -> dict[str, float | None]:
    """Seconds and bytes of one evaluation of ``target`` from the calibration ``points``.

    Each term is extrapolated on its own and the largest counts: the points' largest cost per edge
    times the target's edges, per atom times its atoms, and with PolarMACE (whose k-space grows with
    atoms x volume) per atom x volume times its atoms x volume; plus the two-point fit of
    cost = a edges + b atoms (x volume for PolarMACE) where it has no negative term. Conservative by
    design: a wrong low estimate of memory can take down a WSL2 VM. ``None`` where a point has none.
    """
    result = {}
    for key in ("seconds", "bytes"):
        if any(point.get(key) is None for point in points):
            result[key] = None
            continue
        second = "atoms_volume" if polar else "atoms"
        terms = [max(point[key] / point["edges"] for point in points) * target["edges"],
                 max(point[key] / point["atoms"] for point in points) * target["atoms"]]
        if polar:
            terms.append(max(point[key] / point["atoms_volume"] for point in points) * target["atoms_volume"])
        if len(points) >= 2:
            matrix = np.array([[point["edges"], point[second]] for point in points[:2]], float)
            try:
                a, b = np.linalg.solve(matrix, np.array([point[key] for point in points[:2]], float))
            except np.linalg.LinAlgError:
                a = b = math.nan
            if math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0:
                terms.append(a * target["edges"] + b * target[second])
        result[key] = float(max(terms))
    return result


def _memory_available() -> int | None:
    """``MemAvailable`` of /proc/meminfo in bytes, or None where there is none."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _status_kib(field):
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _reset_peak() -> bool:
    """Reset this process's peak resident size (VmHWM), where Linux allows it."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5")
        return True
    except OSError:
        return False


def _calibrate(reference, boxes, cutoff):
    """Time (and, on Linux, the peak memory of) one evaluation of each box, after a warm-up on the first."""
    if reference.is_polar:
        reference.rebase(boxes[0])
    reference.evaluate(boxes[0])  # warm-up: lazy initialisation is not a per-edge cost
    points = []
    for box in boxes:
        if reference.is_polar:
            reference.rebase(box)
        resident = _status_kib("VmRSS") if _reset_peak() else None
        started = time.perf_counter()
        reference.evaluate(box)
        seconds = time.perf_counter() - started
        peak = _status_kib("VmHWM") if resident is not None else None
        points.append({**_features(box, cutoff), "seconds": seconds,
                       "bytes": None if peak is None else max(peak - resident, 0)})
    return points


# --------------------------------------------------------------------------- references


@dataclasses.dataclass
class References:
    """What ``references`` computed, for ``production``.

    ``copies`` are the production cell's copies (the replay runs on them); ``reference_copies`` the
    copies the references ran on: the same, or the reduced cell's (``reduced``). ``values[i]`` maps
    "reference" and "yardstick" to ``(energy, forces, stress)`` of ``reference_copies[i]``; ``ran``
    is "in-process" or "child process"; ``device`` where they ran; ``steps`` each tried step;
    ``unavailable`` is the record when no reference could run.
    """

    check: str
    copies: list = dataclasses.field(default_factory=list)
    description: str | None = None
    values: list = dataclasses.field(default_factory=list)
    ran: str | None = None
    info: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    seconds: float = 0.0
    unavailable: GateRecord | None = None
    device: str | None = None
    reduced: ReducedCell | None = None
    reference_copies: list = dataclasses.field(default_factory=list)
    steps: tuple = ()
    quantities: tuple = QUANTITIES
    # What the reduced cell's production force field is built from (not part of the record).
    plan: Any = dataclasses.field(default=None, repr=False, compare=False)
    force_field: Any = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def skipped(self) -> bool:
        return self.check == "skip"


def _copy_entries(copies):
    return [{"kind": copy.kind, "numbers": copy.atoms.numbers.tolist(), "positions": copy.atoms.positions.tolist(),
             "cell": copy.atoms.cell.array.tolist()} for copy in copies]


def _job(plan, force_field, copies, *, device=None, cell="production", repeat=None):
    """Everything a reference needs, as JSON (for a child process too)."""
    model = force_field.model
    polar = None if plan.polar is None else dataclasses.asdict(plan.polar)
    return {
        "source": model.label or str(model.path), "head": None if model.label else model.head,
        "sha256": model.sha256, "charge": model.charge, "spin_multiplicity": model.spin_multiplicity,
        "family": model.family, "device": device or plan.engine.device, "backend": plan.engine.backend,
        "polar": polar, "cell": cell, "repeat": None if repeat is None else list(repeat),
        "copies": _copy_entries(copies),
    }


def cpu_job(plan, force_field, cells, atoms, gate) -> dict:
    """The CPU step's job: calibrate on boxes cut from ``atoms``, then the first of ``cells`` within the limits.

    ``cells``: ``(name, repeat, cell atoms, copies)``, largest first.
    """
    cutoff = float(force_field.traits.r_max)
    job = _job(plan, force_field, cells[0][3], device="cpu")
    job.pop("copies")
    job["cutoff"] = cutoff
    job["calibration"] = _copy_entries([GateCopy("box", box) for box in calibration_boxes(atoms, cutoff)])
    job["limits"] = {"time_limit_s": float(gate.cpu_float64_time_limit_s), "memory_margin": MEMORY_MARGIN}
    job["candidates"] = [{"cell": name, "repeat": None if repeat is None else list(repeat),
                          **_features(cell_atoms, cutoff), "copies": _copy_entries(copies)}
                         for name, repeat, cell_atoms, copies in cells]
    return job


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


def _choose(job, points):
    """The CPU step's choice: ``(index or None, estimates, why none)``, the first candidate whose estimate is
    within the time and memory limits."""
    limits = job["limits"]
    available = _memory_available()
    polar = job["polar"] is not None
    estimates = []
    for index, candidate in enumerate(job["candidates"]):
        one = estimate_cost(points, candidate, polar=polar)
        evaluations = len(candidate["copies"]) * len(ROLES)
        seconds = one["seconds"] * evaluations
        memory = one["bytes"]
        estimate = {"cell": candidate["cell"], "repeat": candidate["repeat"], "atoms": candidate["atoms"],
                    "edges": candidate["edges"], "seconds": seconds, "bytes": memory,
                    "memory_available": available}
        estimates.append(estimate)
        within_time = seconds <= limits["time_limit_s"]
        within_memory = memory is None or available is None or memory * limits["memory_margin"] <= available
        if within_time and within_memory:
            return index, estimates, None
    last = estimates[-1]
    parts = []
    if last["seconds"] > limits["time_limit_s"]:
        parts.append(f"estimated {last['seconds']:.3g} s above the limit of {limits['time_limit_s']:g} s "
                     "(GateConfig.cpu_float64_time_limit_s)")
    if last["bytes"] is not None and last["memory_available"] is not None \
            and last["bytes"] * limits["memory_margin"] > last["memory_available"]:
        parts.append(f"estimated {last['bytes'] / 2**30:.3g} GiB x {limits['memory_margin']:g} above the "
                     f"{last['memory_available'] / 2**30:.3g} GiB available")
    return None, estimates, "; ".join(parts)


def cap_memory(growth_bytes) -> int | None:
    """Cap this process's data segment (``RLIMIT_DATA``) at its current size plus ``growth_bytes``.

    An allocation beyond it fails (``MemoryError``, or torch's "can't allocate memory") instead of
    exhausting the host. Returns the limit in bytes, or None where there is no such limit.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - not POSIX
        return None
    data = _status_kib("VmData")
    if data is None:
        return None
    limit = int(data + max(growth_bytes, 0))
    hard = resource.getrlimit(resource.RLIMIT_DATA)[1]
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))
    return limit


def _cap_cpu_child(estimate_bytes=None):
    """The CPU reference child's cap: MemAvailable less the reserve, and at most the estimate x the factor."""
    available = _memory_available()
    growth = None if available is None else available - MEMORY_RESERVE_BYTES
    if estimate_bytes is not None:
        estimated = MEMORY_CAP_FACTOR * estimate_bytes
        growth = estimated if growth is None else min(growth, estimated)
    if growth is None:
        return None
    limit = cap_memory(growth)
    if limit is not None:
        sys.stderr.write(f"gate-child: memory capped at {limit / 2**30:.3g} GiB (data segment)\n")
    return limit


def _evaluate_cpu(job):
    """The CPU step: calibrate the float64 reference, choose a candidate within the limits, evaluate it.

    In a child (``job["cap"]``) the data segment is capped first at MemAvailable less the reserve,
    then, once a candidate is chosen, at most at its estimate x ``MEMORY_CAP_FACTOR``; the measured
    peak resident size of the evaluations is recorded next to the estimate.
    """
    boxes = [_atoms(entry) for entry in job["calibration"]]
    info = {}
    capped = job.get("cap", False)
    info["cpu"] = {"memory_cap_bytes": _cap_cpu_child() if capped else None}
    reference = _load_reference(job, "float64", boxes[0])
    try:
        points = _calibrate(reference, boxes, job["cutoff"])
        chosen, estimates, why = _choose(job, points)
        info["cpu"].update({"calibration": points, "estimates": estimates, "chosen": chosen})
        sys.stderr.write("gate-child: cpu estimates " + json.dumps(
            [{key: e[key] for key in ("atoms", "seconds", "bytes", "memory_available")} for e in estimates]) + "\n")
        if chosen is None:
            info["declined"] = why
            return [], info
        if capped:
            info["cpu"]["memory_cap_bytes"] = _cap_cpu_child(estimates[chosen]["bytes"])
        entries = [_atoms(entry) for entry in job["candidates"][chosen]["copies"]]
        if reference.is_polar:
            reference.rebase(entries[0])
        resident = _status_kib("VmRSS") if _reset_peak() else None
        values = [{} for _ in entries]
        for index, atoms in enumerate(entries):
            energy, forces, stress = _evaluate_one(reference, atoms)
            values[index]["reference"] = (float(energy), np.asarray(forces, float), np.asarray(stress, float))
        peak = _status_kib("VmHWM") if resident is not None else None
        info["cpu"]["measured_peak_bytes"] = None if peak is None else max(peak - resident, 0)
        info["reference"] = json.loads(json.dumps(reference.info(), default=str))
    finally:
        del reference
        _free("cpu")
    reference = _load_reference(job, "float32", entries[0])
    try:
        for index, atoms in enumerate(entries):
            energy, forces, stress = _evaluate_one(reference, atoms)
            values[index]["yardstick"] = (float(energy), np.asarray(forces, float), np.asarray(stress, float))
        info["yardstick"] = json.loads(json.dumps(reference.info(), default=str))
    finally:
        del reference
        _free("cpu")
    return values, info


def evaluate_references(job) -> tuple[list[dict], dict]:
    """The reference and the yardstick on every copy of ``job``: one calculator at a time, freed after use.

    A CPU step's job (``cpu_job``) calibrates and chooses first; it returns no values and
    ``info["declined"]`` when no candidate is within the limits.
    """
    info = {}
    if job.get("backend") == "hybrid":
        from fastmace.model import loading

        info["oeq_runtime"] = loading.prepare_hybrid_if_possible()  # before a reference imports MACE
    if "candidates" in job:
        values, more = _evaluate_cpu(job)
        info.update(more)
        return values, info
    entries = [_atoms(entry) for entry in job["copies"]]
    values = [{} for _ in entries]
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


def is_out_of_memory(error) -> bool:
    """Did ``error`` come from memory: CUDA or host out-of-memory, also in a child (its text or a kill)?"""
    if isinstance(error, MemoryError):
        return True
    import torch

    if isinstance(error, getattr(torch, "OutOfMemoryError", ())) or isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    text = " ".join(str(part) for part in (getattr(error, "what", None) or str(error), getattr(error, "why", None))
                    if part)
    if "out of memory" in text.lower():
        return True
    # A child killed by SIGKILL is what the kernel's out-of-memory killer leaves.
    return isinstance(error, GateJobError) and "exit code -9" in text


def _is_cuda_error(error) -> bool:
    """A CUDA runtime or driver error (an ``AcceleratorError``, or its text), also from a child's stderr."""
    text = " ".join(str(part) for part in (getattr(error, "what", None) or str(error), getattr(error, "why", None))
                    if part)
    return (type(error).__name__ == "AcceleratorError" or "AcceleratorError" in text or "CUDA error" in text
            or "CUDA driver error" in text)


def _reason(error, kind=None) -> str:
    text = getattr(error, "what", None) or (str(error).splitlines() or [""])[0]
    kind = kind or ("out of memory" if is_out_of_memory(error) else "failed")
    reason = f"reference {kind}: {type(error).__name__}" + (f": {text}" if text else "")
    why = getattr(error, "why", None)
    if isinstance(error, GateJobError) and why:  # the child's last lines: the real error text
        reason += " (" + " | ".join(line.strip() for line in str(why).splitlines() if line.strip()) + ")"
    return reason


def _run_step(job, *, child: bool):
    """One step's references, in a child process or in this one."""
    return run_child(job) if child else evaluate_references(job)


class _Ladder:
    """The steps of SPEC §12.3 "Resources" in order (see the module docstring); ``run`` gives the result."""

    def __init__(self, plan, force_field, atoms, config, gate, seed, count):
        from fastmace.engine import base

        self.plan, self.force_field, self.atoms, self.config = plan, force_field, atoms, config
        self.gate, self.seed, self.count = gate, seed, count
        self.device = str(plan.engine.device)
        self.gpu = self.device.startswith("cuda")
        self.child = base.production_engine_built()
        # On a CUDA plan every GPU reference step runs in a child: a GPU out-of-memory can leave the CUDA
        # context unusable (2-T05), and the child's exit frees its memory before the engine is built.
        self.plan_child = self.child or self.gpu
        self.cutoff = float(force_field.traits.r_max)
        self.steps = []
        self.smallest_out_of_memory = None  # atoms of the smallest cell that ran out of memory

    def _record(self, device, cell, repeat, natoms, seconds, outcome, reason=None, **extra):
        self.steps.append({"device": device, "cell": cell, "repeat": None if repeat is None else list(repeat),
                           "atoms": natoms, "seconds": seconds, "outcome": outcome, "reason": reason, **extra})

    def _attempt(self, job, device, cell, repeat, natoms, *, child):
        """Run one step; ``(values, info)`` when it ran, else None (recorded). Non-memory failures raise."""
        from fastmace.reference.calculators import ReferenceOrderError

        started = time.perf_counter()
        process = {"process": "child" if child else "in-process"}
        if child and self.child and device.startswith("cuda"):
            process["gpu_free_bytes"] = _gpu_free_bytes(device)  # a production engine holds the rest
        try:
            values, info = _run_step(job, child=child)
        except ReferenceOrderError:
            raise  # a correction is not the active code path: a defect, not an unavailable reference
        except Exception as error:  # noqa: BLE001 - out of memory, or a reference failing (§12.3)
            seconds = time.perf_counter() - started
            if isinstance(error, GateJobError) and "ReferenceOrderError" in str(error.why or ""):
                raise ReferenceOrderError("a correction of the reference record is not the active code path in the "
                                          "reference child", why=str(error.why)) from error
            if is_out_of_memory(error):
                outcome = "out of memory"
            elif device.startswith("cuda") and (isinstance(error, GateJobError) or _is_cuda_error(error)):
                # A GPU reference child that failed: an out-of-memory can leave the CUDA context unusable
                # and surface as another error (2-T05 on the RTX 5060 Ti), so the ladder goes on.
                outcome = "device failure"
            else:
                outcome = "failed"
            reason = _reason(error, outcome)
            self._record(device, cell, repeat, natoms, seconds, outcome, reason,
                         exit_code=getattr(error, "returncode", None), **process)
            if outcome == "failed":
                raise _Unavailable(reason) from error
            self.smallest_out_of_memory = min(natoms, self.smallest_out_of_memory or natoms)
            return None
        seconds = time.perf_counter() - started
        if info.get("declined"):
            self._record(device, cell, repeat, natoms, seconds, "declined", info["declined"],
                         estimates=info.get("cpu", {}).get("estimates"), **process)
            return None
        if "candidates" in job:
            chosen = job["candidates"][info["cpu"]["chosen"]]
            cell, repeat, natoms = chosen["cell"], chosen["repeat"], chosen["atoms"]
        estimates = (info.get("cpu") or {}).get("estimates")
        extra = {"estimate": estimates[info["cpu"]["chosen"]]} if estimates else {}
        if estimates:
            extra["measured_peak_bytes"] = info["cpu"].get("measured_peak_bytes")
            extra["memory_cap_bytes"] = info["cpu"].get("memory_cap_bytes")
        self._record(device, cell, repeat, natoms, seconds, "ran", **extra, **process)
        return values, info

    def _copies(self, atoms):
        if self.count == COPIES:  # the default gate: the call 1b's tests wrap
            return make_copies(atoms, self.config, self.seed)
        return make_copies(atoms, self.config, self.seed, count=self.count)

    def _cpu(self, cells):
        """The CPU step with ``cells`` (name, repeat, atoms, copies); None when it did not run."""
        limit = self.gate.cpu_float64_time_limit_s
        natoms = len(cells[0][2])
        if limit <= 0:
            self._record("cpu", cells[0][0], cells[0][1], natoms, 0.0, "skipped",
                         "GateConfig.cpu_float64_time_limit_s=0: the CPU is never tried")
            return None
        job = cpu_job(self.plan, self.force_field, cells, self.atoms, self.gate)
        return self._attempt(job, "cpu", cells[0][0], cells[0][1], natoms, child=True)

    def run(self):
        """``(values, info, device, reduced, reference copies)`` of the first step that ran; raises
        ``_Unavailable`` when none did."""
        copies = self._copies(self.atoms)
        natoms = len(self.atoms)
        job = _job(self.plan, self.force_field, copies)
        result = self._attempt(job, self.device, "production", None, natoms, child=self.plan_child)
        if result is not None:
            return (*result, self.device, None, copies)
        if self.gpu:
            result = self._cpu([("production", None, self.atoms, copies)])
            if result is not None:
                return (*result, "cpu", None, copies)
        if self.gate.reduced_cell == "off":
            self._record(self.device, "reduced", None, None, 0.0, "skipped",
                         'GateConfig.reduced_cell="off": no reduced cell is tried')
            raise _Unavailable(self._summary())
        cells = reduced_cells(self.atoms, self.cutoff, self.gate.reduced_cell_min_height_cutoffs)
        if not cells:
            self._record(self.device, "reduced", None, None, 0.0, "skipped",
                         f"no supercell of the primitive cell (repeat {primitive_repeat(self.atoms)}) is smaller "
                         f"than the production cell with every height at least "
                         f"{self.gate.reduced_cell_min_height_cutoffs:g} x the cutoff {self.cutoff:g} Å")
            raise _Unavailable(self._summary())
        # The reduced cells on the plan's device, walking down after each out-of-memory.
        limit = self.smallest_out_of_memory
        for cell in cells:
            if limit is not None and cell.natoms > limit:
                continue
            reduced_copies = self._copies(cell.atoms)
            job = _job(self.plan, self.force_field, reduced_copies, cell="reduced", repeat=cell.repeat)
            result = self._attempt(job, self.device, "reduced", cell.repeat, cell.natoms, child=self.plan_child)
            if result is not None:
                return (*result, self.device, cell, reduced_copies)
            limit = REDUCED_SHRINK * cell.natoms
        if self.gpu:
            # The CPU, with no cell larger than the smallest one out of memory, largest first.
            fitting = [cell for cell in cells if cell.natoms <= self.smallest_out_of_memory]
            if fitting:
                candidates = [("reduced", cell.repeat, cell.atoms, self._copies(cell.atoms)) for cell in fitting]
                result = self._cpu(candidates)
                if result is not None:
                    values, info = result
                    index = info["cpu"]["chosen"]
                    return values, info, "cpu", fitting[index], candidates[index][3]
        raise _Unavailable(self._summary())

    def _summary(self):
        parts = []
        for step in self.steps:
            where = f"{step['cell']} cell" if step["repeat"] is None else \
                f"reduced cell {'x'.join(str(m) for m in step['repeat'])}"
            parts.append(f"{step['device']}, {where}: {step['outcome']}"
                         + (f" ({step['reason']})" if step["reason"] and step["outcome"] != "out of memory" else ""))
        first = next((step["reason"] for step in self.steps if step["outcome"] == "out of memory"), None)
        head = first or "no reference ran"
        return f"{head}; tried " + "; ".join(parts)


def _split_steps(steps):
    """The steps without their timings, and the timings as ``step_<i>_s`` (the record's ``seconds``),
    so a record without ``seconds`` is the same in every run."""
    plain = tuple({key: value for key, value in step.items() if key != "seconds"} for step in steps)
    return plain, {f"step_{index}_s": float(step["seconds"]) for index, step in enumerate(steps, 1)}


class _Unavailable(Exception):
    """No step of the ladder ran (internal; ``references`` turns it into the record)."""


def references(plan, force_field, atoms, integrator, *, check: str, seed: int, gate=None, count: int = COPIES,
               quantities=QUANTITIES) -> References:
    """The gate's first half: copies and their references, before any production engine (see module docstring).

    ``check="skip"`` computes nothing. ``GateUnavailable`` under ``check="gate"`` when no
    reference could run; under ``check="allow_unavailable"`` that is recorded in the result.
    ``gate``: the ``GateConfig`` (default ``GateConfig()``); ``count`` and ``quantities``: T15's
    reduced gate (one copy, the failing quantity).
    """
    from fastmace.config import GateConfig

    check_mode(check)
    if check == "skip":
        return References(check=check)
    gate = GateConfig() if gate is None else gate
    integrator = plan.integrator if integrator is None else integrator
    config = _integrator_config(integrator, seed)
    cell_mode = config.cell_mode if config.ensemble == "npt" else None
    ladder = _Ladder(plan, force_field, atoms, config, gate, seed, count)
    started = time.perf_counter()
    ran = "child process" if ladder.plan_child else "in-process"
    try:
        values, info, device, reduced, reference_copies = ladder.run()
    except _Unavailable as error:
        steps, step_seconds = _split_steps(ladder.steps)
        copies = ladder._copies(atoms)
        description = describe_copies(copies, cell_mode)
        record = GateRecord(status="unavailable", check=check, reason=str(error), copies=description,
                            references={"ran": ran},
                            seconds={"references_s": time.perf_counter() - started, **step_seconds}, steps=steps)
        if check == "gate":
            raise GateUnavailable(
                "the gate's reference could not run", why=record.reason,
                next_step='Free GPU memory and retry, or pass check="allow_unavailable" to run without the gate '
                          "(recorded).", gate_record=record) from error.__cause__
        return References(check=check, copies=copies, description=description, ran=ran, unavailable=record,
                          steps=steps, quantities=tuple(quantities))
    copies = ladder._copies(atoms) if reduced is not None else reference_copies
    description = describe_copies(reference_copies, cell_mode)
    return References(check=check, copies=copies, description=description, values=values, ran=ran, info=info,
                      seconds=time.perf_counter() - started, device=device, reduced=reduced,
                      reference_copies=reference_copies, steps=tuple(ladder.steps), quantities=tuple(quantities),
                      plan=plan, force_field=force_field)


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


def _child_command(job_path, result_path):
    """The reference child's command: this Python, ``python -m fastmace.planner.selfcheck job result``."""
    return [sys.executable, "-m", "fastmace.planner.selfcheck", str(job_path), str(result_path)]


def _gpu_free_bytes(device):
    """Free memory on ``device`` (``torch.cuda.mem_get_info``), or None."""
    try:
        import torch

        return int(torch.cuda.mem_get_info(torch.device(device))[0])
    except Exception:  # noqa: BLE001 - informational only
        return None


def run_child(job, *, timeout_s: float | None = None) -> tuple[list[dict], dict]:
    """``evaluate_references(job)`` in a fresh process of this environment, which builds no production engine."""
    directory = _scratch_directory()
    try:
        job_path, result_path = directory / "job.json", directory / "result.npz"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        argv = _child_command(job_path, result_path)
        done = subprocess.run(argv, env=_child_environment(), capture_output=True, text=True, timeout=timeout_s,
                              check=False)
        if done.returncode != 0 or not result_path.is_file():
            tail = "\n".join((done.stderr or done.stdout or "").strip().splitlines()[-8:])
            raise GateJobError(f"the reference child process failed (exit code {done.returncode})", why=tail or None,
                               returncode=done.returncode)
        return _read_result(result_path)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def resource_limit_text() -> str:
    """" (capped at X GiB)" when this process's data segment is capped, else ""."""
    try:
        import resource

        soft = resource.getrlimit(resource.RLIMIT_DATA)[0]
    except (ImportError, ValueError):  # pragma: no cover - not POSIX
        return ""
    return "" if soft == resource.RLIM_INFINITY else f" (capped at {soft / 2**30:.3g} GiB)"


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
        count = 1 + max((int(key.split("/")[0]) for key in data.files if key != "info"), default=-1)
        values = [{role: (float(data[f"{index}/{role}/energy"]), data[f"{index}/{role}/forces"].copy(),
                          data[f"{index}/{role}/stress"].copy()) for role, _ in ROLES} for index in range(count)]
    return values, info


def _child_main(argv) -> int:
    job_path, result_path = argv
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    job["cap"] = True  # a CPU reference child caps its own memory (_evaluate_cpu)
    try:
        values, info = evaluate_references(job)
    except (MemoryError, RuntimeError) as error:
        if not (isinstance(error, MemoryError) or "can't allocate memory" in str(error)):
            raise
        limit = resource_limit_text()
        sys.stderr.write(f"gate-child: {job['device']}: out of memory{limit}: {type(error).__name__}: "
                         f"{str(error).splitlines()[0][:200] if str(error) else ''}\n")
        return 3
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


def _replay(engine, copies, *, direct: bool = False):
    """Each copy through the engine's stages C and B (captured in graph mode); the state restored afterwards.

    Returns the outputs per copy, with ``direct`` also a direct evaluation of the same force field at
    each copy's state (``MACEForceField.evaluate``, uncaptured), and the notes (a neighbour capacity
    that grew).
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
    outputs, directs = [], []
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
            if direct:
                capacity = field.edge_capacity
                directs.append(field.evaluate(copy.atoms, stress=stress))
                if field.edge_capacity != capacity:
                    grew = True
                    engine._clear_graphs()
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
    if direct:
        return outputs, directs, notes
    return outputs, notes


def _replay_check(replayed, directs, natoms, quantities, limits):
    """The captured replay against a direct evaluation of the same force field, per quantity within ``limits``.

    The direct evaluation is ``MACEForceField.evaluate``: the same (compiled) model function,
    uncaptured, on a freshly built unpadded neighbour list. ``limits`` are the gate's own limits from
    the reduced cell's comparison, max(2 x the float32 yardstick's error, floor): a float32 plan's
    replay differs from a direct evaluation by float32 rounding (padded edges, OEq's atomic sums;
    4.8e-5 and 6.0e-5 eV/Å on 1944-atom POLAR-1-L and OMOL-0 XL, 2-T05), which the floors alone
    (made for float64-sized errors) do not admit; the full gate absorbs the same rounding in its
    production error. A broken capture (forces off by 1e-3 and more) still fails.
    """
    rows = [copy_errors(direct, replay, natoms) for replay, direct in zip(replayed, directs, strict=True)]
    result = {"cell": "production", "copies": len(rows),
              "compared": "captured replay against a direct evaluation of the same force field (the model function "
                          "uncaptured, on a fresh unpadded neighbour list)",
              "bound": "the gate's limit per quantity: max(2 x float32 yardstick, floor)",
              "bound_source": "the reduced cell's comparison with the reference (the comparison that ran)",
              "quantities": {}}
    passed = True
    for quantity in quantities:
        values = [row[quantity] for row in rows if row[quantity] is not None]
        if not values:
            continue
        limit = limits.get(quantity, FLOORS[quantity])
        if not all(math.isfinite(value) for value in values):
            result["quantities"][quantity] = {"difference": None, "limit": limit, "passed": False}
            passed = False
            continue
        worst = max(values)
        ok = bool(worst <= limit)
        result["quantities"][quantity] = {"difference": worst, "limit": limit, "passed": ok}
        passed = passed and ok
    result["passed"] = passed
    return result


def _reduced_outputs(references: References, engine, stress: bool):
    """The production path on the reduced cell's copies: a separate force field converted from the pristine
    model with the plan's backend, precision and rewrites (the engine's resolution), evaluated eagerly."""
    from fastmace.config import PolarConfig
    from fastmace.model.forces import MACEForceField

    plan, force_field = references.plan, references.force_field
    copies = references.reference_copies
    loaded = force_field.execution_model(plan)
    reduced = None
    try:
        reduced = MACEForceField(
            loaded, copies[0].atoms, rewrites=engine.force_field.resolution,
            engine=dataclasses.replace(plan.engine, compile_mode="none"), neighbours=plan.neighbours,
            polar=plan.polar or PolarConfig(), compute_stress=stress, check=False,
            float64_energy=plan.precision.float64_energy, species=getattr(engine.force_field, "superset", None))
        return [reduced.evaluate(copy.atoms, stress=stress) for copy in copies]
    finally:
        del reduced, loaded
        _free(plan.engine.device)


def production(engine, references: References, *, check: str, quantities=None) -> GateRecord:
    """The gate's second half: the production path on the copies, the ratio rule, and the record.

    Raises ``GateFailed`` when the gate fails. Returns the ``skipped`` record for ``check="skip"``
    and the ``unavailable`` record for ``check="allow_unavailable"`` when no reference ran. On a
    reduced cell (``references.reduced``) the status is ``passed-reduced``, and the replay check runs
    on the production cell. ``quantities`` defaults to the references' (T15's reduced gate).
    """
    check_mode(check)
    if check != references.check:
        raise ValueError(f"production(check={check!r}) differs from the references' check={references.check!r}")
    if references.skipped:
        return skipped_record()
    if references.unavailable is not None:
        return references.unavailable
    quantities = tuple(references.quantities if quantities is None else quantities)
    started = time.perf_counter()
    stress = engine.state.stress is not None and engine.force_field.compute_stress
    replay = None
    if references.reduced is None:
        outputs, notes = _replay(engine, references.copies)
    else:
        replayed, directs, notes = _replay(engine, references.copies, direct=True)
        replay = (replayed, directs)
        outputs = _reduced_outputs(references, engine, stress)
    natoms = len(references.reference_copies[0].atoms)
    rows = []
    for output, values in zip(outputs, references.values, strict=True):
        reference, yardstick = values["reference"], values["yardstick"]
        if output[2] is None:
            yardstick = (yardstick[0], yardstick[1], None)
        rows.append({"production": copy_errors(reference, output, natoms),
                     "yardstick": copy_errors(reference, yardstick, natoms)})
    result = verdict(rows, quantities=quantities)
    if replay is not None:
        limits = {quantity: q["limit"] for quantity, q in result["quantities"].items() if "limit" in q}
        replay = _replay_check(*replay, len(references.copies[0].atoms), quantities, limits)
    worst = {quantity: result["quantities"].get(quantity, {}).get("production") for quantity in QUANTITIES}
    if result["nonfinite"]:
        reasons = ["non-finite values: " + ", ".join(result["nonfinite"])]
    else:
        failed = [(quantity, q) for quantity, q in result["quantities"].items() if not q["passed"]]
        reasons = [f"{_NAMES[quantity]} error {q['production']:.3g} {_UNITS[quantity]} above the limit "
                   f"{q['limit']:.3g} (2 x float32 yardstick {q['yardstick']:.3g}, floor {q['floor']:g})"
                   for quantity, q in failed]
    if replay is not None and not replay["passed"]:
        reasons.append("replay on the production cell differs from a direct evaluation: " + ", ".join(
            f"{_NAMES[quantity]} {q['difference']:.3g} {_UNITS[quantity]} above the limit {q['limit']:.3g}"
            if q["difference"] is not None else f"{_NAMES[quantity]} non-finite"
            for quantity, q in replay["quantities"].items() if not q["passed"]))
    passed = result["passed"] and (replay is None or replay["passed"])
    status = ("passed" if references.reduced is None else "passed-reduced") if passed else "failed"
    mode = ("captured replay of stages C, B" if getattr(engine, "capture", False) else "eager stages C, B")
    if references.reduced is not None:
        mode = f"eager force field on the reduced cell; {mode} on the production cell"
    record = GateRecord(
        status=status, check=check, reason="; ".join(reasons) or None,
        copies=references.description, quantities=result["quantities"], nonfinite=tuple(result["nonfinite"]),
        mode=mode, references={"ran": references.ran, "record_id": references.info["reference"]["record_id"],
                               **references.info},
        seconds={"references_s": references.seconds, "production_s": time.perf_counter() - started,
                 **_split_steps(references.steps)[1]},
        notes=tuple(notes), device=references.device,
        reduced_cell=None if references.reduced is None else tuple(references.reduced.repeat),
        steps=_split_steps(references.steps)[0], replay=replay, **worst)
    if not record.passed:
        raise GateFailed(f"the gate on this structure failed: {record.reason}",
                         why="the production path's error against the float64 reference exceeds the ratio rule "
                             "of SPEC §12.3, or its replay differs from a direct evaluation (the §9.4 fallback "
                             "acts on this error).", gate_record=record)
    return record


__all__ = ["CHECK_MODES", "FLOORS", "GateCopy", "GateRecord", "ReducedCell", "References", "calibration_boxes",
           "check_mode", "copy_errors", "cpu_job", "describe_copies", "edge_count", "estimate_cost",
           "evaluate_references", "heights", "is_out_of_memory", "make_copies", "primitive_repeat", "production",
           "reduced_cell", "reduced_cells", "references", "run_child", "skipped_record", "verdict"]


if __name__ == "__main__":
    sys.exit(_child_main(sys.argv[1:]))
