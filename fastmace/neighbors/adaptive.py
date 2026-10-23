"""Rule 11 (SPEC §9.3): the skin and the pair-buffer headroom from the workload.

The skin list stays valid while no atom has moved more than skin / 2 since its build
(``SkinNeighbourList.needs_rebuild``), so the skin sets the rebuild rate. Rule 11 sizes it from the
expected displacement per rebuild interval, with no per-GPU or per-crystal table (SPEC §1.1 goal 2):

- **The fastest atom.** Each Cartesian velocity component of an atom of mass m at temperature T is
  normal with deviation sigma = sqrt(k_B T / m). The speed |v| / sigma of an atom then follows the
  Maxwell (chi, 3 degrees of freedom) distribution, whose tail is
  P(|v| > z sigma) = erfc(z / sqrt 2) + sqrt(2 / pi) z exp(-z^2 / 2). Of N atoms, about one is faster
  than the z_N with N P = 1. The lightest mass and the highest temperature give the largest sigma.
- **Ballistic over a rebuild interval.** Over a few tens of femtoseconds, short against the
  vibration periods of the ions, an atom keeps its velocity, so the fastest atom moves
  d = z_N sigma dt per step. The rebuild test then fires after skin / (2 d) steps.
- **The target interval.** The skin is the one that makes this interval ``TARGET_REBUILD_STEPS``
  steps, and at least the floor (phase 1a's 0.3 Å): skin = max(floor, 2 n* d). At 300 K and 0.5 fs
  the floor holds for the ionic crystals of the test set (NaCl, NbCl5, NaNbO3), and NaCl's measured
  rebuild interval there (about 25 steps, docs/validation/1a-noise-floor.md) matches the model's 24.
- **Displacement moves** (§10.4) add 2 delta_max: the displaced atom still has the MD budget of
  skin / 2 and moves at most delta_max on top, so a trial rarely needs a rebuild (§8.5 step 4).

The pair buffers hold every candidate within cutoff + skin, ``capacity_headroom`` above the first
build. Between rebuilds atoms cross the candidate sphere's surface; over one interval the shell of
pairs within skin / 2 of radius R = cutoff + skin turns over, a relative change of about
3 (skin / 2) / R. The headroom covers that, and at least the floor (1.15). ``Rule11.recapture_budget``
is the number of padded edge buckets the count can visit inside that capacity, above and below the
first count, 2 ceil((headroom - 1) / bucket fraction) + 1. The plan only records it: the run's
re-capture budget is 3-T04's rule (the graph variant cap, ``FixedPlanner._phase_3_fields``), since one
new bucket re-captures several stage graphs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ase import units as ase_units

from ..config import DEFAULT_SKIN_A

# eV / u in Å^2 / fs^2: 1.602176634e-19 J / 1.66053906660e-27 kg = 9.64853e7 m^2 / s^2.
EV_PER_U_IN_A2_PER_FS2 = 1.602176634e-19 / 1.66053906660e-27 * 1e-10
BOLTZMANN_EV_PER_K = 8.617333262e-5
# The rebuild interval the skin aims at, in steps (n*). At 300 K and 0.5 fs it keeps the ionic
# crystals on the 0.3 Å floor; above, it trades a larger candidate set for fewer rebuilds.
TARGET_REBUILD_STEPS = 16
# Phase 1a's pair-buffer headroom (``NeighbourConfig.capacity_headroom``), the headroom floor.
DEFAULT_HEADROOM = 1.15
# Rule 11 for optimisers (4-T05): the skin in optimiser steps of the largest length, maxstep. The rebuild test
# fires once an atom has moved skin / 2, so 3 maxsteps keep at least 1.5 full steps between rebuilds in a
# relaxation's first, largest steps. Measured on MH-1 NaCl and NbCl5 quench frames (docs/validation/4-T05.md):
# 0.6 A halves the rebuilds of 0.3 A and the re-captures with them, and 1.0 A gains nothing more.
OPTIMIZER_SKIN_STEPS = 3.0


def optimizer_skin(maxstep_A: float, floor_skin_A: float = DEFAULT_SKIN_A) -> float:
    """Rule 11's skin for an optimiser whose largest per-atom step is ``maxstep_A``: ``OPTIMIZER_SKIN_STEPS``
    of them, at least the floor (Angstrom)."""
    return max(float(floor_skin_A), OPTIMIZER_SKIN_STEPS * float(maxstep_A))


def maxwell_tail(z: float) -> float:
    """P(|v| > z sigma) for a 3-D Maxwell speed in units of the per-component deviation sigma."""
    return math.erfc(z / math.sqrt(2.0)) + math.sqrt(2.0 / math.pi) * z * math.exp(-0.5 * z * z)


def fastest_speed_quantile(natoms: int) -> float:
    """z_N with natoms * P(|v| > z_N sigma) = 1: the speed, in sigma, that about one of the atoms exceeds.

    At least sqrt(3), the root-mean-square speed, so that one or two atoms get a sensible value.
    """
    if natoms < 1:
        raise ValueError(f"natoms must be positive, got {natoms!r}")
    low, high = 0.0, 40.0
    for _ in range(200):  # bisection: maxwell_tail decreases from 1 at z = 0
        middle = 0.5 * (low + high)
        if natoms * maxwell_tail(middle) > 1.0:
            low = middle
        else:
            high = middle
    return max(math.sqrt(3.0), 0.5 * (low + high))


def thermal_speed(temperature_K: float, mass_u: float) -> float:
    """sigma = sqrt(k_B T / m) in Å/fs: the deviation of one velocity component."""
    if mass_u <= 0:
        raise ValueError(f"mass must be positive, got {mass_u!r}")
    return math.sqrt(BOLTZMANN_EV_PER_K * temperature_K / mass_u * EV_PER_U_IN_A2_PER_FS2)


def _up(value: float, step: float = 0.01) -> float:
    """``value`` rounded up to ``step``, so that plans print short numbers (round() removes float noise)."""
    return round(math.ceil(round(value / step, 9)) * step, 9)


@dataclass(frozen=True)
class Rule11:
    """Rule 11's sizes and the numbers they come from (all lengths in Å)."""

    skin_A: float  # the planned skin: md_skin_A plus 2 delta_max
    md_skin_A: float  # max(floor, 2 n* d): the MD part
    capacity_headroom: float
    recapture_budget: int | None  # None without edge buckets (fraction 0)
    lightest_mass_u: float
    max_temperature_K: float
    timestep_fs: float
    natoms: int
    speed_quantile: float  # z_N
    step_displacement_A: float  # d = z_N sigma dt
    max_displacement_A: float | None

    def text(self) -> str:
        """One line for the plan's decisions and the report."""
        out = (f"skin {self.skin_A:g} A, headroom {self.capacity_headroom:g} (fastest of {self.natoms} atoms at "
               f"{self.max_temperature_K:g} K, lightest mass {self.lightest_mass_u:g} u, {self.timestep_fs:g} fs: "
               f"{self.step_displacement_A:.4f} A per step, {TARGET_REBUILD_STEPS} steps per rebuild")
        if self.max_displacement_A is not None:
            out += f"; plus 2 x {self.max_displacement_A:g} A for displacement moves"
        return out + ")"


def rule_11(*, max_temperature_K: float, lightest_mass_u: float, timestep_fs: float, natoms: int,
            cutoff_A: float | None,
            edge_bucket_fraction: float, max_displacement_A: float | None = None,
            floor_skin_A: float = DEFAULT_SKIN_A, floor_headroom: float = DEFAULT_HEADROOM,
            target_rebuild_steps: int = TARGET_REBUILD_STEPS) -> Rule11:
    """Skin, headroom and re-capture budget for the workload (module docstring).

    ``cutoff_A=None`` (a model without a known cutoff) keeps the headroom at its floor.
    """
    if timestep_fs <= 0 or (cutoff_A is not None and cutoff_A <= 0):
        raise ValueError(f"timestep and cutoff must be positive, got {timestep_fs!r} fs and {cutoff_A!r} A")
    z = fastest_speed_quantile(natoms)
    step = z * thermal_speed(max_temperature_K, lightest_mass_u) * timestep_fs
    md_skin = max(floor_skin_A, _up(2.0 * target_rebuild_steps * step))
    skin = md_skin if max_displacement_A is None else _up(md_skin + 2.0 * max_displacement_A)
    headroom = floor_headroom if cutoff_A is None else max(
        floor_headroom, _up(1.0 + 3.0 * (0.5 * skin) / (cutoff_A + skin)))
    budget = (None if edge_bucket_fraction <= 0 else
              2 * math.ceil(round((headroom - 1.0) / edge_bucket_fraction, 9)) + 1)
    return Rule11(skin_A=skin, md_skin_A=md_skin, capacity_headroom=headroom, recapture_budget=budget,
                  lightest_mass_u=float(lightest_mass_u), max_temperature_K=float(max_temperature_K),
                  timestep_fs=float(timestep_fs), natoms=int(natoms), speed_quantile=z, step_displacement_A=step,
                  max_displacement_A=None if max_displacement_A is None else float(max_displacement_A))


# ---------------------------------------------------------------------------------------------------- D3 list
#
# Rule 11 with D3 on (SPEC §9.3, §11.2; plan task 3b-T02, conductor's choice 4). The D3 list has its own skin
# and rebuild test (``neighbors.d3_list``). A D3 build enumerates about 40 times the MACE list's candidates,
# so its skin aims at a longer rebuild interval than the MACE skin's:
#
# - **Displacement:** as for the MACE skin, the fastest atom moves d = z_N sigma dt per step, ballistic over
#   the interval: 2 n d.
# - **Strain** (a barostat): the rebuild test allows a pair at cutoff + skin to come within the cutoff only
#   while 2 max|y - y_build| <= cutoff + skin - cutoff / (1 - |F - I|_F). A Frobenius strain f over the
#   interval costs cutoff f / (1 - f) of the skin. The barostat's cell momentum p_eps is thermal with
#   <p_eps^2> = k_B T W per component (ASE's heuristic mass W), so the linear strain rate p_eps / W has the
#   deviation sqrt(k_B T / W); the bound is STRAIN_QUANTILE deviations, times sqrt(3) for a hydrostatic
#   strain (F - I = eps I) and 3 for a full cell (nine components). It is ballistic over the interval too.
#   (The melt-quench volume swing of about 16 % spans the whole ramp, not one interval.)
# - **The target interval** ``D3_TARGET_REBUILD_STEPS`` and the floor ``D3_FLOOR_SKIN_A`` are provisional:
#   3b-T06 chooses the skin on the production ramp.
#
# Rule 12 with D3 on: the stored pairs at uniform density (``d3_list.estimated_pairs``), their buffer at the
# capacity headroom, and the build's transient memory; above ``D3_MEMORY_WARN_FRACTION`` of the GPU memory
# the plan warns.

D3_TARGET_REBUILD_STEPS = 50  # provisional (3b-T06)
D3_FLOOR_SKIN_A = 1.0  # provisional (3b-T06)
STRAIN_QUANTILE = 3.0  # deviations of the barostat's strain rate
D3_MEMORY_WARN_FRACTION = 0.25
ASE_TIME_PER_FS = ase_units.fs  # ASE time units per fs (ASE's time unit sqrt(u A^2 / eV) is 10.18 fs)


def barostat_mass(P_tau_fs: float, natoms: int, mass_factor: float = 1.0) -> float:
    """ASE's heuristic barostat mass for P_tau (``integrators.baoab.barostat_mass``), from P_tau in fs."""
    if not P_tau_fs > 0:
        raise ValueError(f"P_tau must be positive, got {P_tau_fs!r} fs")
    return ((P_tau_fs / 4.0) / 1.66 * natoms ** (1.0 / 6.0)) ** 2 * mass_factor


def barostat_strain_rate(*, temperature_K: float, natoms: int, P_tau_fs: float, mass_factor: float = 1.0,
                         cell_mode: str = "hydrostatic") -> float:
    """The bound on the Frobenius strain rate |dF/dt|_F per fs of the barostat at ``temperature_K``."""
    if cell_mode not in ("hydrostatic", "full"):
        raise ValueError(f"cell_mode must be 'hydrostatic' or 'full', got {cell_mode!r}")
    mass = barostat_mass(P_tau_fs, natoms, mass_factor)
    deviation = math.sqrt(BOLTZMANN_EV_PER_K * temperature_K / mass) * ASE_TIME_PER_FS  # per fs
    return STRAIN_QUANTILE * (math.sqrt(3.0) if cell_mode == "hydrostatic" else 3.0) * deviation


@dataclass(frozen=True)
class Rule11D3:
    """Rule 11's D3 skin and the numbers it comes from (lengths in Å)."""

    skin_A: float
    displacement_term_A: float  # 2 n d
    strain_term_A: float  # cutoff f / (1 - f)
    cutoff_A: float
    target_rebuild_steps: int
    step_displacement_A: float  # d
    strain_per_step: float  # Frobenius strain rate times the timestep (0 without a barostat)
    max_temperature_K: float
    timestep_fs: float

    def text(self) -> str:
        """One line for the plan's decisions and the report."""
        out = (f"D3 skin {self.skin_A:g} A for {self.target_rebuild_steps} steps per rebuild at "
               f"{self.max_temperature_K:g} K and {self.timestep_fs:g} fs: displacement "
               f"{self.displacement_term_A:.3f} A "
               f"({self.step_displacement_A:.4f} A per step)")
        if self.strain_per_step > 0:
            out += f", barostat strain {self.strain_term_A:.3f} A ({self.strain_per_step:.2e} per step)"
        return out + f", cutoff {self.cutoff_A:g} A"


def rule_11_d3(*, max_temperature_K: float, lightest_mass_u: float, timestep_fs: float, natoms: int,
               cutoff_A: float, strain_rate_per_fs: float = 0.0,
               target_rebuild_steps: int = D3_TARGET_REBUILD_STEPS, floor_skin_A: float = D3_FLOOR_SKIN_A) -> Rule11D3:
    """The D3 list's skin: max(floor, 2 n d + cutoff f / (1 - f)) (the D3 block above)."""
    if timestep_fs <= 0 or cutoff_A <= 0 or strain_rate_per_fs < 0 or target_rebuild_steps < 1:
        raise ValueError(f"invalid rule 11 D3 inputs: timestep {timestep_fs!r} fs, cutoff {cutoff_A!r} A, strain rate "
                         f"{strain_rate_per_fs!r} per fs, {target_rebuild_steps!r} steps")
    z = fastest_speed_quantile(natoms)
    step = z * thermal_speed(max_temperature_K, lightest_mass_u) * timestep_fs
    strain_step = strain_rate_per_fs * timestep_fs
    strain = strain_step * target_rebuild_steps
    if strain >= 0.5:
        raise ValueError(f"the barostat strain over {target_rebuild_steps} steps would be {strain:.3f}; "
                         "shorten the timestep or lengthen the barostat time")
    displacement_term = 2.0 * target_rebuild_steps * step
    strain_term = cutoff_A * strain / (1.0 - strain)
    skin = max(floor_skin_A, _up(displacement_term + strain_term))
    return Rule11D3(skin_A=skin, displacement_term_A=displacement_term, strain_term_A=strain_term,
                    cutoff_A=float(cutoff_A), target_rebuild_steps=int(target_rebuild_steps),
                    step_displacement_A=step, strain_per_step=strain_step,
                    max_temperature_K=float(max_temperature_K), timestep_fs=float(timestep_fs))


@dataclass(frozen=True)
class Rule12D3:
    """Rule 12's D3 list memory estimate (bytes)."""

    pairs: int  # stored pairs at uniform density
    capacity: int  # at the headroom
    list_bytes: int
    build_bytes: int  # transient, during a build
    gpu_memory_bytes: int | None
    warn: bool

    def text(self) -> str:
        """One line for the plan's decisions and the report."""
        out = (f"D3 list about {self.pairs / 1e6:.2f} M pairs, {self.list_bytes / 2**20:.0f} MiB at capacity "
               f"{self.capacity}, builds about {self.build_bytes / 2**20:.0f} MiB transient")
        if self.gpu_memory_bytes:
            out += (f", {(self.list_bytes + self.build_bytes) / self.gpu_memory_bytes:.1%} of "
                    f"{self.gpu_memory_bytes / 2**30:.1f} GiB")
        return out


def rule_12_d3(*, natoms: int, volume_A3: float, radius_A: float, headroom: float = DEFAULT_HEADROOM,
               gpu_memory_bytes: int | None = None) -> Rule12D3:
    """The D3 list's stored pairs, buffer and build memory (the D3 block above)."""
    from .d3_list import BUILD_BYTES_PER_CANDIDATE, BYTES_PER_PAIR, CAPACITY_MULTIPLE, estimated_pairs

    pairs = estimated_pairs(natoms, volume_A3, radius_A)
    capacity = int(math.ceil(pairs * headroom / CAPACITY_MULTIPLE) * CAPACITY_MULTIPLE)
    list_bytes = capacity * BYTES_PER_PAIR + 6 * 8 * natoms  # pairs, and the per-atom offsets and positions
    build_bytes = int(pairs * BUILD_BYTES_PER_CANDIDATE)
    warn = bool(gpu_memory_bytes) and list_bytes + build_bytes > D3_MEMORY_WARN_FRACTION * gpu_memory_bytes
    return Rule12D3(pairs=int(round(pairs)), capacity=capacity, list_bytes=int(list_bytes), build_bytes=build_bytes,
                    gpu_memory_bytes=gpu_memory_bytes, warn=warn)
