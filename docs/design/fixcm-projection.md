# The Langevin `fixcm` projection with unequal masses

Plan task 3-T22, 22 October 2026. Found by 3-T12
(`side-session-notes/…/3-T12-finding-fixcm-unequal-masses.md`). The measurements are in
[validation/3-T22-fixcm.md](../validation/3-T22-fixcm.md).

`IntegratorConfig.fixcm_projection` chooses how `fixcm` removes the centre-of-mass part of the
Langevin noise:
- `"mass_weighted"`, the projection derived below. It is the default under NVT `fixcm`, for MD and MC
  (decision 37, 3-T22c; SPEC §10.4);
- `"ase"`, ASE's projection bit for bit. It stays as an explicit option for ASE noise-stream equivalence,
  for A/Bs against the research engine, which uses it, and for the phase 3 no-regression A/B;
- `"auto"`, the field's default, which the planner resolves (below).

**`"auto"`.** The planner resolves it in every plan and records the decision
`IntegratorConfig.fixcm_projection`:
- `"mass_weighted"` under NVT `fixcm`: code `fixcm-projection-md` for MD, `fixcm-projection-mc` when the
  workload declares MC moves. MC's `redraw_all` draws (3N − 3) kT, which only this projection holds;
- `"ase"` (code `fixcm-projection-ase`) for NPT, where `LangevinBAOAB` has no `fixcm`, and for `fixcm`
  off. Neither changes anything;
- an explicit value is kept (`fixcm-projection-given`).

Outside a plan (a bare `Langevin`, the reference integrators), `"auto"` reads as `"ase"`. `MCDriver.attach`
refuses `redraw_all` under `fixcm` with any other projection (code `mc-redraw-all-ase-projection`).

**Reported temperatures** (3-T22c). Every kinetic temperature FastMACE reports comes from
`fastmace.io.trajectory.kinetic_temperature`, over `degrees_of_freedom(natoms, integrator)`:
- 3N − 3 under NVT `fixcm` with `"mass_weighted"`;
- 3N otherwise, which includes `"ase"`, as `Atoms.get_temperature` counts it.

`Simulation.degrees_of_freedom` gives a run's count from its plan. The thermo rows (`temperature_K`)
use it. The report and the CLI only show what the rows hold.

`fastmace.ase.Langevin` follows the default. Its log, ASE's `MDLogger`, reads a copy of each frame that
carries ASE's `FixCom`, so the logged temperature counts 3N − 3 as well. The trajectory frames stay
unconstrained, because a stored `FixCom` would change the momenta read back. `fixcm_projection="ase"` is
ASE's `Langevin` exactly. The thermostat's target is unchanged.

**Resume.** A run keeps the projection it ran with:
- a record from before the option has no field and reads as `"ase"`. That covers the `simulation.json`
  integrator record and the plan in the restart file;
- an `"auto"` record whose stored plan resolved to `"ase"` (an MD run from 3-T22b) keeps `"ase"`;
- an `"auto"` record that resolves to what it ran with stays as it is.

**The fused Triton stage** implements both projections bit for bit against the PyTorch stage (3-T22b).
For `"mass_weighted"`:
- the noise kernel also stores rnd_pos·m;
- the two centre-of-mass sums stay PyTorch reductions, in `Langevin.first_half`'s order;
- the drift kernel subtracts sum/M, one float64 division each, with M the registered derived buffer
  `fused.total_mass`.

The public `fm.integrators.Langevin` record and `fastmace.ase.Langevin` take `fixcm_projection` too.

## The scheme

`fastmace.integrators.langevin.Langevin` is ASE's `Langevin`, the second-order scheme of
Vanden-Eijnden and Ciccotti. Each step draws ξᵢ, ηᵢ ~ N(0, 1) for every atom i and Cartesian component,
and forms

- rnd_posᵢ = c₅ σᵢ ηᵢ,
- rnd_velᵢ = c₃ σᵢ ξᵢ − c₄ σᵢ ηᵢ,

with σᵢ = √(2γkT/mᵢ) and c₃, c₄, c₅ depending only on dt and γ. Per atom and component, the pair
(rnd_posᵢ, rnd_velᵢ) is therefore Gaussian with covariance S/mᵢ, where S is a 2 × 2 matrix that is the same
for every atom.

Two facts about the deterministic part of the step matter below:
- the forces sum to zero (a translation-invariant potential), so Σ mᵢ (Fᵢ/mᵢ) = 0;
- the friction multiplies every velocity by the same factor.

So, without noise, the step keeps the total momentum P and moves the centre of mass only by V dt.
`fixcm` asks the noise to do the same: Σ mᵢ rnd_posᵢ = 0 and Σ mᵢ rnd_velᵢ = 0.

## The conditional Gaussian

For a Gaussian z ~ N(0, C) and the linear constraint Bz = 0, the conditional distribution is that of
(I − C Bᵀ (B C Bᵀ)⁻¹ B) z. Here, per Cartesian component, C = S ⊗ diag(1/m) and B = I₂ ⊗ mᵀ. That gives
C Bᵀ = S ⊗ 1 and B C Bᵀ = M S, with M = Σ mᵢ. The projector is therefore I − I₂ ⊗ (1 mᵀ / M):

- rnd_posᵢ ← rnd_posᵢ − Σⱼ mⱼ rnd_posⱼ / M,
- rnd_velᵢ ← rnd_velᵢ − Σⱼ mⱼ rnd_velⱼ / M.

Both take the same mass-weighted mean from every atom, and there is no scale factor. `redraw_all` (3-T09)
removes the momentum in the same way. For equal masses the two means are ASE's plain means. The two
choices then differ only by ASE's √(N/(N−1)) factor.

## Why it samples the right distribution

Write xᵢ = X + yᵢ and vᵢ = V + uᵢ with Σ mᵢ yᵢ = Σ mᵢ uᵢ = 0. The forces depend only on y, and the friction
is the same for every atom. So the run without `fixcm` splits into two independent processes:
- an internal process, driven by the projected noise above;
- a free centre-of-mass process, driven by the rest of the noise. The projected part of a Gaussian is
  independent of the constrained part (Bz).

With `"mass_weighted"`, a run started at rest is therefore that internal process, number for number. It is
the run without `fixcm` seen from its centre of mass. `tests/cpu/test_fixcm_projection.py` checks this
step by step, with the same random numbers, to 1e-11.

Without `fixcm` the scheme samples the canonical distribution, to its own time-step accuracy. Its internal
part is Maxwell–Boltzmann on the subspace Σ mᵢ uᵢ = 0. In particular:

- ⟨Σᵢ mᵢ uᵢ²⟩ = (3N − 3) kT: the kinetic temperature counts **3N − 3** degrees of freedom;
- per species s, ⟨Σ_{i∈s} mᵢ uᵢ²⟩ = (3Nₛ − 3 Mₛ/M) kT. Per component, the covariance of √mᵢ uᵢ is
  kT (I − √m √mᵀ / M), whose trace over species s is Nₛ − Mₛ/M.

These degree-of-freedom counts are the "fixed centre of mass" convention of the validation tables.

## What ASE's projection does

ASE subtracts the plain mean from rnd_pos and P/(N mᵢ) from rnd_vel, then scales both by √(N/(N−1)).

**Positions.** For unequal masses the plain mean does not remove the centre-of-mass displacement:
Σ mᵢ (rᵢ − r̄) = Σ mᵢ rᵢ − M r̄ ≠ 0.

**Velocities.** Σ mᵢ rnd_velᵢ = 0 holds, but every atom gives up the same momentum P/N. The ideal is
mᵢ P/M, so the correction to the velocity is P/(N mᵢ) instead of P/M. The light atoms get the largest
corrections.

Per component, the momentum noise of atom i then has variance

  N/(N−1) · mᵢ s (1 − 2/N + m̄/(N mᵢ)),

against the conditional value mᵢ s (1 − mᵢ/(N m̄)), where s is its unconstrained variance per unit mass
and m̄ = M/N. For an atom much lighter than the mean, the term m̄/(N mᵢ) dominates the excess. The light
species is heated beyond the target, and the forces pass part of that heat to the other species. The size
grows with the mass contrast and shrinks roughly as 1/N.

**Equal masses.** ASE's variance is exactly the unconstrained one: N/(N−1) · (1 − 1/N) = 1. That is
3N kT in total, so ASE's kinetic temperature is right over 3N degrees of freedom and reads N/(N−1) high
over 3N − 3.

## The research engine and ASE

- **Research engine:** the frozen engine has the same projection, copied from ASE:
  - `Experiments/Langevin_Thermostat/langevin/gpu_langevin.py`, `GPULangevin._stage_a`, at
    ResearchModule HEAD d37df508;
  - its CPU path, `workflow.py`, runs ASE's own `Langevin(fixcm=True)`.

  Every research `fixcm` run has the bias in this note.
- **ASE:** ASE 3.29's `ase/md/langevin.py` (lines 187–194) is the same.
