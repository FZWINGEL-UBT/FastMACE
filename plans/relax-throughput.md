# Relaxation throughput: one engine for many frames, then many frames in one engine

Owner request, 29 October 2026, from ResearchModule's Energy_Hull step 3a (relaxing melt-quenched glass frames):
write the plan for (A) reusing engines across frames and (B) several frames in one engine. Plan only; nothing is
built until the owner says so.

## Why

Energy_Hull relaxes hundreds of small, same-composition frames with variable cell (`relax_set.py`, which wraps
`GPURelaxer`; gpu-LBFGS + `FrechetCell(scalar_pressure_GPa=0)`, float64 with `fp64_emulation_mantissa_bits="dynamic"`,
`radial_tables="off"`, fmax 0.01 eV/Å). One 112-atom NaTaOCl4 glass frame on leto (RTX 4000 Ada), first worker on
the GPU:

| Part | Time |
| --- | ---: |
| relaxation loop (765 LBFGS steps, 42 ms per step) | 32 s |
| everything else: ForceField, plan, build, compile, capture | about 78 s |
| wall | 110 s |

- **Setup:** every frame gets a fresh engine. `GPURelaxer` reuses an engine for the same atomic numbers by
  `update(cell=, positions=)`, but that path is for fixed-cell engines. With a cell filter, the caller builds anew
  (relax_set.py: "host cell writes are for fixed-cell engines"). So 70 % of the wall time is setup that the next
  frame (same species, same atom count) repeats.
- **The loop:** 112 atoms use a small fraction of the GPU. Four worker processes per GPU raise the throughput to
  about 3 frames per minute on two GPUs, which suggests the GPU is latency-bound, not compute-bound.

Part A removes the repeated setup; part B fills the GPU. A is small and comes first; B is a larger change and
starts only if B-T00's measurement says it pays.

## Part A: one engine across frames (variable cell)

**Idea.** Allow `update(cell=, positions=)` on an engine built with a cell-filter optimiser, at a canonical point.
The next `relax()` then starts from a fresh optimiser and cell-filter state, as a new engine would. MD with a
barostat keeps the refusal (the barostat owns the cell there).

What a new structure on the same engine must reset or rebuild (A-T00 checks the list against the code):

- **Optimiser state:** LBFGS history (`s`, `y`, `rho`), FIRE velocity, `a` and `N_min`, step counters, the
  initial energy.
- **Cell-filter reference:** FrechetCell's reference cell (the deformation gradient's origin) and `exp_cell_factor`
  must be the new start cell, since the filter's degrees of freedom are relative to it. A stale reference gives a
  different optimisation path and a different minimum.
- **Neighbour list:** rebuilt at the new cell and positions. If the edge count leaves the padded bucket or a
  capacity grows, the graphs are re-captured, which is recorded (as for the fixed-cell update today).
- **Cell-dependent inputs:** strain graph inputs, PBC shifts, D3 lists, `reference_volume`.
- **Not reset:** the compiled model, the plan, the captured graphs when the sizes fit, the ForceField.

| Task | What | Stop rule | Status |
| --- | --- | --- | --- |
| **A-T00** | Measurement, no production code (`tools/bench/relax_reuse.py`). (a) Per-frame breakdown of `GPURelaxer`-style setup (ForceField, plan, build, compile, capture, first force call) on 84- and 112-atom NaMOCl4 glass frames and on NbCl5 3×3×3, fixed and variable cell, float64 dynamic and float32. (b) The fixed-cell `update(cell=, positions=)` path's cost on the same frames (neighbour rebuild, re-captures). (c) Locate where the variable-cell refusal fires and list everything A-T01 must reset. Report `docs/validation/relax-A-T00.md` | setup ≥ 30 % of a frame's wall time on the glass frames; otherwise report and stop | not started |
| **A-T01** | `Simulation.update(cell=, positions=)` on cell-filter optimiser engines: reset the optimiser and cell-filter state at the canonical point, rebuild the lists, re-capture only on a size change. The barostat refusal stays. Tests first (CPU): (1) relax frame B on an engine built for frame A equals a fresh engine's relax of B (same steps; energies, cell and positions to 1e-10 relative in float64); (2) the same for FIRE; (3) a larger cell that grows the edge bucket re-captures once and still matches; (4) NPT MD still refuses a host cell write; (5) a second `relax()` without `update` still continues the optimiser history | CPU suite | not started |
| **A-T02** | `GPURelaxer` reuses the engine for variable cell too (same atomic numbers in the same order), with `builds` and the re-captures in its summary; `protocols.quench` and `fm.relax` use it. `docs/` note | CPU suite | not started |
| **A-T03** | GPU check on the RTX 5060 Ti and RTX 4000 Ada (RTX A4000 if free): 40 glass frames per size, fresh against reused, same process. Equality as in A-T01 (float64), the step counts, wall per frame | equal results; per-frame wall ≤ loop time + 5 s | not started |

Expected gain: about 3× per frame on the glass frames (110 s → about 35 s), more with float32. Every
variable-cell relaxation of a series gains: quenches, step 2's competing phases with several polymorphs, DFT
preparation.

## Part B: several frames in one engine (batched relaxation)

**Idea.** One force call for K structures: a disjoint union graph (each structure keeps its own cell and PBC
shifts; no edges between structures), per-structure energies and stresses by segment sums over a batch index, and
K independent optimiser states updated by batched vector operations. A converged structure is frozen (zero step,
masked) and its slot refilled with the next frame (continuous batching), so the batch stays full until the queue
ends.

Why it can pay: at 84–112 atoms the step time is mostly launches and synchronisation, so K structures cost far less
than K steps. Where it stops paying is a measurement, not a guess (B-T00).

What it touches:

- **Neighbour list:** a batch index per atom, per-structure cells and shifts, one CSR over the union (receiver sort
  per structure keeps the streaming convolution's assumptions).
- **Model call:** MACE itself batches; FastMACE's engine, rewrites and captured graphs assume one structure (one
  cell, one stress, one strain). Per-structure strain, so per-structure stress through the single-graph strain path.
- **Long-range terms:** D3 per structure; PolarMACE k-space out of scope (refused).
- **Optimiser:** LBFGS and FIRE with per-structure history and masks; FrechetCell per structure. Results must equal
  K separate relaxations to rounding (sums over atoms change order, so not bitwise).
- **Padding:** fixed K slots and padded atom and edge counts per slot, so the CUDA graphs are captured once.
  Structures of different sizes share a batch only within a padded slot size.
- **API:** `fm.relax_many(frames, model=, optimizer=, fmax=, steps=, batch=K)`, returning per-frame relaxed
  structures and summaries in input order; `GPURelaxer` can use it for a quench.

| Task | What | Stop rule | Status |
| --- | --- | --- | --- |
| **B-T00** | Falsifier, no production code (`tools/bench/relax_batch_ceiling.py`): the force-call time of today's engine against atom count (one 84- and 112-atom glass frame, and supercells of them up to 32×, float64 dynamic and float32, both GPUs), giving the ceiling of a K-batch (time of K·N atoms against K × time of N). Report `docs/validation/relax-B-T00.md` | a K = 16 batch of 84-atom frames ≥ 3× the throughput of A's reused single engine (four workers per GPU as the baseline too); otherwise report and stop | not started |
| **B-T01** | Batched neighbour list: batch index, per-structure cells and shifts, padded slots; CPU tests against K separate lists (identical edge sets per structure) | CPU suite | not started |
| **B-T02** | Batched force call in the engine: per-structure energy, forces, stress (per-structure strain), D3 per structure; every exact rewrite either supports the batch or is switched off with a recorded reason. CPU tests against K single calls (float64 to 1e-12 relative) | CPU suite | not started |
| **B-T03** | Batched LBFGS/FIRE with FrechetCell per structure, convergence masks, slot refill. CPU tests: K batched relaxations equal K single ones (steps; energies to 1e-10 relative in float64) | CPU suite | not started |
| **B-T04** | `fm.relax_many` and the `GPURelaxer` path; docs | CPU suite | not started |
| **B-T05** | Gates: force/stress gate on batched calls (NaCl, NbCl5, NaNbO3 and the glass frames), equality with single relaxations on 100 glass frames, throughput on both GPUs against A's reused engine with four workers | gates pass; throughput ≥ B-T00's stop rule | not started |

## Rules

- General changes only (any crystal and composition). Per-GPU defaults only as SPEC decision 66 allows (same-process
  A/Bs in both build orders, exact device names).
- Tests first; no default changes without the owner. Local GPU work inside `tools/ci/local_gpu.py run` reservations;
  workstation work under ResearchModule's `AGENTS.md` (idle check before every job, GPU limit).
- Out of scope here, recorded for later: a two-stage precision relaxation (float32 to fmax ≈ 0.05 eV/Å, then float64
  to 0.01), which shortens the float64 loop but changes no engine code.
