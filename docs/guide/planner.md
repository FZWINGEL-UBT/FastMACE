# The planner

Every `Simulation` (and so every `fm.md`, `fastmace md` and `fastmace.ase`
run) starts with a **plan**. The planner reads the model, the structure, the
integrator, the workload, the installed software and the GPU, and decides:

- the **tier**: the CUDA-graph engine (fastest), the compiled tier without
  graphs, the eager hybrid, or the reference path;
- the **backend**: the cuEquivariance/OpenEquivariance hybrid, or plain
  cuEquivariance;
- the **precision** of the force evaluation (float32 by default; integration
  is always float64);
- which **rewrites** are on, and the reason for each one that is off;
- the neighbour-list skin and buffer sizes.

Rules are general: there are no per-GPU or per-crystal tuning tables. Every
rewrite is exact to rounding and is checked against the original MACE
module when it is applied.

## Read the plan

```python
model = fm.load("mh1", head="matpes_r2scan")
sim = fm.Simulation(atoms, model, integrator=..., seed=1)
print(sim.plan)                  # str(plan): about 30 lines, by section
print(sim.plan.describe(full=True))   # every decision with its rule and reason
```

`str(plan)` gives a short overview: the tier, backend and precision, the
hard rules that decided them, the rewrites that are on and the ones that are
off with a short reason. `describe(full=True)` lists every decision. The
same text is in the run report and in `metadata.json`.

`plan.decision(key)` returns `(item, choice, reason)` for one decision, and
`plan.overrides()` lists the options you set explicitly.

## Influence the plan

- **Workload:** `fm.Workload(max_temperature_K=..., species_changes=...,
  mc_moves=..., host_callbacks_every=...)` tells the planner what the run
  will do. `fm.md` infers it from its arguments. A schedule above
  `max_temperature_K` is refused. For runs with MC moves or species
  changes the skin and buffer headroom are sized from it; plain MD keeps
  the 0.3 Å skin.
- **Mode:** `Simulation(mode=...)` asks for a tier; `None` lets the planner
  choose.
- **Rewrites and precision** live on the force field:
  `fm.ForceField(model, atoms, precision="float32", rewrites="auto")`.
- An option the planner cannot honour raises `PlanRefused` with every
  reason, before the first step. Nothing is substituted silently.

## The gate

`check="gate"` evaluates the planned engine on your structure (and rattled
copies) against a float64 reference before the first step. If it fails, the
fallback searches for the closest plan that passes, turning off rewrites one
by one, and records what it did. The gate is opt-in; the default is
`check="skip"`. The record is `sim.gate`.

## Runtime adaptation

During a run the planner watches re-captures and buffer overflows. Past a
budget it may switch tier at a committed step, or fall back after a capture
error, and the report's `runtime` line records it. A resume rebuilds the
recorded revision exactly.

## When the run did not get the graph engine

The report says why and what would change it: a structure that is not 3D
periodic, a missing OpenEquivariance (`fastmace doctor` names the piece), a
model outside the rewrite registry, padded edges that are not exactly zero,
a failed self-check, or a fallback after a failed gate.

The full rule list is on
[Planner, gate and fallback](../reference/planner.md). The
[planner design note](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/docs/design/planner-5.md)
in the repository explains the design.

See also: [Planner, gate and fallback](../reference/planner.md),
[Engine and neighbour list](../reference/engine.md),
[Rewrites and precision](../reference/rewrites.md).
