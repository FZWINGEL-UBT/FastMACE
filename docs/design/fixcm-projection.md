# The Langevin `fixcm` projection with unequal masses

Plan task 3-T22, 22 October 2026. Found by 3-T12
(`side-session-notes/…/3-T12-finding-fixcm-unequal-masses.md`). The measurements are in
[validation/3-T22-fixcm.md](../validation/3-T22-fixcm.md).

`IntegratorConfig.fixcm_projection` chooses how `fixcm` removes the centre-of-mass part of the
Langevin noise. There are two choices:
- `"ase"`, the default, is ASE's projection bit for bit. The ASE noise-stream equivalence tests need it;
- `"mass_weighted"` is the projection derived below.

Whether the default changes is the owner's decision (plan checkpoint 2b). The fused Triton stage
implements only `"ase"`, so `"mass_weighted"` runs the PyTorch stage.

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
