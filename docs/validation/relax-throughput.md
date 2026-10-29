# Relaxation throughput: one engine across glass frames (A-T00 to A-T03), batching ceiling (B-T00)

Plan: [plans/relax-throughput.md](../../plans/relax-throughput.md). Owner request, 29 October 2026: Energy_Hull step
3a relaxes hundreds of melt-quenched glass frames (84 and 112 atoms, same species per MD run) with variable cell,
each on a fresh engine. This report measures that, adds engine reuse for cell-filter relaxations, and measures
the ceiling of batching several frames in one engine.

## Run

| | |
| --- | --- |
| Machine, GPU | murphy, NVIDIA RTX 4000 Ada Generation (one GPU, idle check before the job, watchdog every 30 s) |
| Environment | `/mnt/homes/zwingel/environments/mace_cueq_oeq` through ResearchModule's `oeq_no_nvcc.sh` (hybrid backend, as Energy_Hull ran it on murphy); torch 2.14.1+cu130, mace-torch 0.3.16, cuequivariance 0.12.0, openequivariance 0.7.0 |
| Model | MH-1 (`mace-mh-1.model`), head `matpes_r2scan` |
| Settings (Energy_Hull pass A) | float64, `fp64_emulation_mantissa_bits="dynamic"`, `radial_tables="off"`, no D3, gpu-LBFGS with `FrechetCell(scalar_pressure_GPa=0)`, fmax 0.01 eV/Å, at most 5000 steps |
| Frames | [results/relax-throughput/frames.csv](../../results/relax-throughput/frames.csv): 20 of Energy_Hull's 340 glass frames, 5 per group (Nb-84, Ta-84, Nb-112, Ta-112), evenly spaced by name |
| FastMACE | `19f4cd3a` (the code of this branch; later commits change tests, tools and docs only) |
| Baseline | the fresh-engine path of the same commit, which is what `relax_set.py` runs (a fresh `ForceField` and `Simulation` per frame); the research engine has no variable-cell gpu-LBFGS to compare |
| Script | `tools/bench/relax_reuse.py --blocks fresh,reused,reused,fresh` (one process, both orders), report `tools/bench/relax_reuse_report.py` |
| Time | 29 October 2026, 20:47 to 21:57 CET |
| Raw data | [results/relax-throughput/murphy-20261029/](../../results/relax-throughput/murphy-20261029/) (`rows.jsonl`, `frames.npz`, `run.json`, `log.txt`) |

## A-T00: where a frame's time goes today

Fresh engine per frame, first fresh block (block 0), 20 frames: wall 1164 s, of which setup 598 s (51 %) and
relaxation loop 566 s. Median setup per frame 32 s, almost all of it the engine build (31.8 s: compile and
autotune of the float64 path); plan 0.0 s, capture 0.3 s. The stop rule (setup ≥ 30 % of a frame's wall time)
holds, so A-T01 went ahead.

Not measured here, from A-T00's list: float32, NbCl5 3×3×3, the fixed-cell `update` cost on its own. The reset
list (part (c)) is in the code and tests of A-T01: the LBFGS/FIRE history is reset by `integrator.start` at the
canonical point already; what was missing was the cell filter's reference cell.

## A-T01, A-T02: what changed

- `FrechetCellFilter.reset_reference()` (`fastmace/optimize/cell.py`) restarts the filter from the state's current
  cell as construction does: `orig_cell` and its inverse, `log_deform = 0`, `x`'s cell rows, `cell_ok`.
  `OptimizerStepper.start` calls it, so every canonical point after a host write starts the filter afresh.
- `GraphEngine.reset_cell` refuses a host cell only for a barostat (variable cell without an optimiser); a
  cell-filter relaxation accepts it. `Simulation.update(cell=, positions=)` documents the same.
- `GPURelaxer` (`fastmace/protocols/quench.py`) keeps its engine across frames with the same atomic numbers for
  variable cell too, takes `rewrites=`, and reports `engine_built` per frame.
- Tests `tests/cpu/test_update_cell_relax.py` (8): frame B on frame A's engine equals a fresh engine (LBFGS, and
  FIRE at 0.5 GPa) to 1e-10 relative; the filter's state equals a fresh filter's; a larger cell with more edges
  still matches; a barostat still refuses; a second `relax()` without `update` continues the history;
  `GPURelaxer` builds once and matches fresh engines (gpu-LBFGS, gpu-FIRE). Run on leto (GPU hidden, 8 workers):
  all 8 pass, with the 68 existing relaxation and `update` tests (`test_update_cell.py`,
  `test_relax_simulation.py`, `test_relax_cell_simulation.py`, `test_fire_simulation.py`,
  `test_protocol_relaxations.py`, `test_optimize_cell.py`) passing in the first run (75 passed; the one failure
  was this branch's FIRE test leaving the filter's range, fixed by capping its steps).

## A-T03: fresh against reused on the GPU

| Block | Mode | Frames | Engine builds | Wall s | Setup s | Loop s | Steps | Frames/hour |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | fresh | 20 | 20 | 1164 | 598 | 566 | 15995 | 61.9 |
| 1 | reused | 20 | 4 | 699 | 131 | 568 | 16003 | 103.1 |
| 2 | reused | 20 | 4 | 697 | 128 | 568 | 16012 | 103.4 |
| 3 | fresh | 20 | 20 | 1534 | 952 | 583 | 15983 | 46.9 |

- **Speed-up 1.67×** on these 20 frames (61.9 → 103 frames per hour), with four engine builds (one per group of
  five). A reused frame that does not build costs its loop and nothing else (wall = loop to 0.1 s; the plan's
  limit is loop + 5 s). For Energy_Hull's 340 frames in four groups the builds vanish: about **127 frames per
  hour** (loop 28.4 s per frame), **2.0×** block 0.
- **Block 3 is slower than block 0** because 5 of its 20 fresh engines failed setup and went through the
  fallback (58 to 75 s each, `setup_parts_s.fallback`), after about 50 engines built in the process. Reused
  blocks build 8 engines in all and never fell back. The failure's cause is not recorded by the script (the
  plan's `fallbacks` record was not saved); a follow-up. Block 0 is the fair "before".

### Equality (against block 0)

| Block | Mode | Same steps | max abs dE meV/atom | max d cell length Å | max dV/V | max atom d Å | within limits |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | reused | 18/20 | 0.0031 | 0.0037 | 2.5e-05 | 0.014 | 18/20 |
| 2 | reused | 18/20 | 0.0075 | 0.0067 | 0.00049 | 0.039 | 18/20 |
| 3 | fresh | 18/20 | 0.011 | 0.0084 | 0.00045 | 0.045 | 19/20 |

Limits (4-T07): cell lengths 1e-3 Å, volume 0.01 %, energy 0.025 meV/atom.

- On 18 of 20 frames, reused and fresh engines agree to rounding: same step count, |dE| ≤ 3e-6 meV/atom, cell
  lengths ≤ 4e-6 Å.
- Two frames (`H3a-nvtsample-wei-Ta-s3-leeloo_t0097.0ps`, 84 atoms, and `H3a-nvtsample-pbca-Nb-s1-leeloo_t0101.0ps`,
  112 atoms) end 1 to 12 steps apart and exceed the cell-length limit (up to 0.0067 Å). The **second fresh block
  differs from the first on the same two frames** (13 steps, 0.0084 Å on the Nb frame; 0.00088 Å on the Ta frame),
  so this is the GPU's run-to-run non-determinism on flat energy landscapes, not the reuse. Energies stay within
  the limit on every frame and in every block (≤ 0.011 meV/atom).
- The 4-T07 cell-length limit is therefore not met by fresh-against-fresh either on these settings; reuse adds no
  difference of its own. The CPU tests show the reuse path itself is exact (1e-10 relative, float64).

## B-T00: the batching ceiling

`tools/bench/relax_batch_ceiling.py` on murphy, same environment and settings, 29 October 2026, 21:57 to 22:05
CET; raw data [results/relax-throughput/ceiling-murphy-20261029/](../../results/relax-throughput/ceiling-murphy-20261029/).
The variable-cell LBFGS step of today's engine on supercells of one 84-atom Nb frame
(`H3a-nvtmelt-wei-Nb-s2-leeloo_t0081.0ps`, rattled 0.02 Å), 30 warm steps, then 200 timed steps. A K-batch of
84-atom frames costs at least one step of K·84 atoms, so K × t(84) / t(K·84) is the most it could gain over one
reused engine.

| Copies K | Atoms | ms per step | K × t(84) ms | Ceiling gain | Peak GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 84 | 32.1 | 32 | 1.00× | 0.35 |
| 2 | 168 | 42.6 | 64 | 1.51× | 0.66 |
| 4 | 336 | 61.5 | 128 | 2.09× | 0.83 |
| 8 | 672 | 103.6 | 257 | 2.48× | 1.31 |
| 16 | 1344 | 193.0 | 514 | **2.66×** | 1.80 |
| 32 | 2688 | 365.3 | 1027 | 2.81× | 2.86 |

**Stop rule fails: K = 16 gives at most 2.66× (needs ≥ 3×), and K = 32 only 2.81×.** Above about 600 atoms the
step time grows linearly with the atoms, so the GPU is compute-bound on the float64 path well before K = 16.
The ceiling is an upper bound: a real batch adds masks, padded slots and the wait for its slowest frame, and the
plan's baseline also counts four worker processes per GPU on A's reused engine, which takes part of the same
headroom. Part B (B-T01 to B-T05) stops here, as the plan says. Not measured: float32, 112-atom frames, the
RTX 5060 Ti.

## Status and what is left

- A-T00 partly done (glass frames, float64 dynamic, murphy; not float32, NbCl5 or the RTX 5060 Ti).
- A-T01, A-T02 done (CPU tests on leto).
- A-T03 partly done: murphy only, 20 frames rather than 40 per size; the RTX 5060 Ti and RTX A4000 not run (one
  GPU for this task).
- B-T00 done on murphy (float64, 84 atoms): stop rule fails; B-T01 to B-T05 not started.
- Follow-ups: the fallback after many fresh engines in one process (block 3); run-to-run differences on two frames
  at fmax 0.01 against the 4-T07 cell-length limit.
