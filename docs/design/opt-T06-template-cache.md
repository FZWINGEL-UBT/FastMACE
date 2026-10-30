# OPT-T06: converted-model template cache (design note)

**Status: design for review, 21 October 2026.** Plan `plans/opt-1.md`, OPT-T06. Nothing is implemented yet.

## Problem

Every plan reloads, re-hashes and converts the checkpoint (`ForceField.execution_model` → `loading.load`, decision
24). That costs 3.1–3.7 s per plan on the GPU (2-T19, MH-1 hybrid and plain cuEq). Since 1a loaded once per process,
1b's second and later plans are 2.9 s slower than 1a's. The same call runs:

- once per plan in `MinimalPlanner.build` and `build_evaluation`;
- **a second time inside a gated plan**: the gate's reduced-cell path (`planner/selfcheck.py:_reduced_outputs`)
  converts again;
- once for every fallback candidate (§9.4 step 1: "built from a pristine copy of the loaded model").

## The measured gain (before building anything)

GPU, from 2-T19's per-stage JSON (`results/2-T19-local-20261020`, RTX 5060 Ti, MH-1, warm). Each plan already makes
a private deep copy of its execution model (`compile.private_copy` in `MACEForceField`), so both numbers are on
record per plan:

| Backend, structure | execution model (reload, hash, conversion) | `private_copy` of the converted model |
| --- | ---: | ---: |
| hybrid float32, NaCl (3 processes) | 3.46–3.48 s | 0.70–0.71 s |
| hybrid float32, NbCl5 (later plan) | 3.17–3.28 s | 0.53–0.55 s |
| plain cuEq float32 (MACE first), NaCl | 3.19 s | 0.50 s |
| hybrid, `Simulation` with gate, NaCl | 3.69 s | 0.57 s |

A copy of a cached template therefore costs about 0.5–0.7 s where the reload costs 3.2–3.7 s: **about 2.7 s saved
per later plan, per fallback candidate, and per gated plan's reduced-cell model.**

CPU (`results/opt-T06-cpu-20261021`, `clone_vs_reload.py`, MH-1 float64) shows the gain is backend- and device-dependent:

| Backend, device | reload | `private_copy` |
| --- | ---: | ---: |
| e3nn, CPU | 0.39 s | 0.74 s |
| plain cuEq, CPU (naive kernels) | 41.6 s | 41.1 s |

- **e3nn:** there is no conversion, so the copy loses.
- **cuEq on the CPU:** a deepcopy re-runs the constructors of cuEquivariance's `SegmentedPolynomial`, which pickles
  through `__reduce__`, and so rebuilds the naive kernels. It costs as much as the reload.
- So the cache applies to the **hybrid and plain cuEq backends on CUDA** only. Elsewhere it would be no gain, or a
  loss. That is a rule from the backend and the device, not a table.
- A GPU check of the copy against the reload for plain cuEq float64 and hybrid float64 is still open (below).

**Bitwise check** (`copy_bitwise.py`, e3nn float64, CPU, one thread):

- the second load, `private_copy` of the first load, a copy of that copy, and `private_copy` of the second load
  all agree bitwise;
- only the process's very first load differs from all of them, by 3.5e-18 eV/Å in the forces, reproducibly.

Today's production model of every plan is a private copy of a fresh load, and a copy of a cached template is
bitwise that too.

## Design

### What is cached

One **template** per conversion key: the converted `LoadedModel` exactly as `loading.load` returns it, with its
calculator, model, provenance fields, `patches` and `atomic_energies`. It is never handed to a plan and never
run.

A plan receives an **execution copy**:

- `dataclasses.replace(template, calculator=shell)`;
- `shell` is a shallow `copy.copy` of the template's calculator, with `models = [compile.private_copy(template.model)]`
  and `results = {}`.

Downstream only reads `calc.models[0]`, `r_max`, the unit factors and `_atoms_to_batch` (`model/forces.py`,
`reference/calculators.py`), so a shallow shell around a private model is enough. No module, parameter or buffer is
shared between two plans, or between a plan and the template. `private_copy` already re-applies the OEq and fused-cuEq
adapters that a deepcopy drops; every force field relies on that today.

**First plan.** The first `execution_model` call for a key loads as today and returns that load, so the first plan
is exactly today's. The question is when the template is made. Templates are made only where caching applies (CUDA,
cuEq or hybrid).

- **Alternative A: defer the template.** The first plan gets the fresh load. A second plan with the
  same key loads once more, and only that second load becomes the template. That costs nothing in single-plan
  processes, saves from the third plan on, and loses the saving on the second plan.
- **Alternative B: copy at once.** The first plan also keeps a `private_copy` of its load as the template: +0.6 s on
  the first plan of every process, −2.7 s on every later plan.
  This favours `fm.calc`, the gate's reduced cell (always a second conversion under `check="gate"`) and fallbacks.

A gated plan is the common case (the gate's reduced cell calls `execution_model` again within the same plan), so
**B is the better default when the gate is on, and A when it is off.** Proposal: B, the simpler rule, because
`check="gate"` is `Simulation`'s recommended path and fallback depends on it. Conductor or owner choice.

### Key

The conversion depends on, and the key holds:

- checkpoint SHA-256 (already verified by the pristine load), head, `model_type`;
- backend, dtype, device;
- the resolved `cueq_conv_fusion` (`_patches.conv_fusion_for`), `allow_naive_cueq`, `charge` and
  `spin_multiplicity` (the last two only feed `check_embedding`, but a key that covers them is simpler to prove);
- the process patches in force (`_patches.applied()` at conversion time, as names and values). A hit requires the
  same patches now, so a patch change since the conversion is a miss;
- versions: mace-torch, torch, cuequivariance and cuequivariance-ops, and OEq's build id (the 2-T04 hash gates already
  make a version change within a process impossible, but the key costs nothing).

Not in the key, on purpose:

- **Structure, species and rewrites:** templates are pristine, and rewrites act on the plan's own copy.
- **Self-check results:** checkpoint 2 kept self-checks out of any cache, and every plan still runs all of them on its
  copy.

### Ownership, lifetime, bound

- **Owner: the `ForceField`.** It already holds the pristine model and is plan-free (owner decision 12), so the
  templates live and die with it. No module-level state: no global cache and no switch (AGENTS.md). Two `ForceField`
  objects of one checkpoint do not share templates.
  - **Superseded by setup-across-compositions F3 (30 October 2026).** The templates now belong to the pristine
    `LoadedModel` (`LoadedModel.template_cache`). Every `ForceField` made from one `fm.load` result shares them, so
    a new composition copies the conversion instead of converting again.
    - Each plan still gets its own private copy, the mutation guard still applies, and nothing is module-level.
    - `ForceField.template_count()` and `release_templates()` act on the shared cache.
    - Two separate `fm.load` results still do not share.
- **Bound:** at most 2 templates per `ForceField`, least recently used out. For MH-1 that is ~52 MB of parameters and
  buffers per template (measured on CPU; the GPU size is the same order).
- **Clearing:** `ForceField.release_templates()` (new public method; it frees the templates and returns the bytes
  freed). Callers:
  - the gate's resource ladder (2-T05) before an out-of-memory retry;
  - the subprocess runner hand-off;
  - users who keep a `ForceField` and want the memory back.
- **Opt-out:** a config field (`EngineConfig` or `PlannerConfig`, e.g. `template_cache: Literal["on", "off"] = "on"`)
  so that a bisection can compare against today. It needs T01-style contract ownership, so the conductor names the
  field.
- **Subprocess runner:** the child has its own `ForceField` and its own templates. Nothing crosses the process
  boundary (§8.7, the restart write).

### Self-checks and the gate

Unchanged: every record's self-check runs on the plan's own copy against MACE's own model, and the gate runs as
today. The template is pristine, the state before any rewrite (§9.4), and the copy is the same model bitwise. So
the checks see what they see today.

### Fallback

Every candidate is "built from a pristine copy of the loaded model" (§9.4 step 1). With the cache:

- a candidate with the same backend and dtype is a hit: a copy, 0.5–0.7 s instead of 3.2–3.7 s;
- a candidate that changes backend or precision is a miss: it converts, and its template joins the cache (the bound
  drops the oldest);
- the eager diagnosis (§9.4 step 2) and the bisection both need a pristine model per attempt, so they gain the most.

### Integrity

- On a hit the checkpoint is not re-read, so a file changed on disk after the first conversion is not noticed. The
  run continues with the model whose hash was verified. Today the next plan would refuse. Provenance stays correct
  either way, because the template carries the verified digest.
- **Mutation guard:** the template must never change. A cheap fingerprint is taken when it is stored and checked on
  every hit: parameter and buffer count, dtypes and devices, plus the `id` and `data_ptr` of each tensor. A mismatch
  is a programming error and raises. The full numeric hash is too slow on every hit, but the tests check it.

## Tests (written first)

CPU (fast tier, fake converted models where the real conversion needs CUDA):

1. **Key:** each key field changed alone misses; identical options hit. A changed `_patches.applied()` misses.
2. **Copy, not share:** two execution models of one key share no tensor storage, no module object and no calculator
   `results` dict with each other or with the template.
3. **A → B → A isolation (§8.7):** the existing isolation tests (`tests/cpu/test_isolation.py`), run with the cache on
   and off. Results must be identical, and bitwise on CPU float64 against main.
4. **Precision and backend change:** float32 then float64, and cuEq then e3nn, in one `ForceField`: misses, two
   templates, bound respected; a third key evicts the least recently used.
5. **Fallback:** a monkeypatched self-check failure (2-T04's fault injection in tests) re-plans. The candidate's
   execution model comes from the template (a hit), and the failing plan's rewrites are not visible in it.
6. **Mutation guard:** a test mutates a template tensor in place, and the next hit raises.
7. **Lifetime:** after `del force_field`, a weakref to the template is dead; `release_templates()` empties the cache
   and reports the bytes.
8. **Rule:** e3nn, or any backend on the CPU, never stores a template.
9. **Default path bitwise unchanged:** a default plan's forces and trajectory on CPU float64 are bitwise main's.

GPU (fast tier, short, untimed):

10. **Equality of copy and load:** for hybrid float32 and float64 and plain cuEq float32 and float64, a plan built
    from a template copy gives bitwise the same initial forces and 10 steps as a plan built from a fresh load. This
    is the GPU version of `copy_bitwise.py`.
11. **Memory:** `torch.cuda.memory_allocated` after `release_templates()` returns to the level before the first plan,
    within the allocator's rounding.
12. **Setup:** a second `Simulation` on the same `ForceField` has `execution model` ≤ 1 s, against 3.2–3.7 s
    (reported, not asserted against a time).

## Files (proposal)

| File | Part |
| --- | --- |
| `fastmace/model/forcefield.py` | `execution_model` uses the template cache; `release_templates()` |
| `fastmace/model/templates.py` (new) | the key, the bounded cache, the copy, the fingerprint |
| config field | opt-out, named by the conductor (T01-style contract) |
| `tests/cpu/test_template_cache.py`, `tests/gpu/test_template_cache_gpu.py` | tests 1–12 |
| `docs/validation/opt-T06-template-cache.md` | the setup measurement after the change |

`loading.py` stays unchanged: the cache sits above it.

## Open questions for the conductor

1. Template timing: B (copy at the first plan) or A (defer to the second conversion)? The recommendation is B, see
   "First plan".
2. Where the opt-out field goes, and its name.
3. A short untimed murphy probe for test 10 (and the plain cuEq float64 copy time). A few minutes of GPU, when
   murphy is free.
4. `installed_mace_version()` ("small fix alongside"): 2-T19b (merged, 42e683d) already memoises it per `sys.path`, so
   a resolve reads the metadata once per process. Nothing is left unless a stricter "once per process, ignoring
   `sys.path`" is wanted. I'd leave it.
