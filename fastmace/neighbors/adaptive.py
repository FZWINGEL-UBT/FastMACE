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

from ..config import DEFAULT_SKIN_A

# eV / u in Å^2 / fs^2: 1.602176634e-19 J / 1.66053906660e-27 kg = 9.64853e7 m^2 / s^2.
EV_PER_U_IN_A2_PER_FS2 = 1.602176634e-19 / 1.66053906660e-27 * 1e-10
BOLTZMANN_EV_PER_K = 8.617333262e-5
# The rebuild interval the skin aims at, in steps (n*). At 300 K and 0.5 fs it keeps the ionic
# crystals on the 0.3 Å floor; above, it trades a larger candidate set for fewer rebuilds.
TARGET_REBUILD_STEPS = 16
# Phase 1a's pair-buffer headroom (``NeighbourConfig.capacity_headroom``), the headroom floor.
DEFAULT_HEADROOM = 1.15


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
