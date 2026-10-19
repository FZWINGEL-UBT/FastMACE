# 1b-T04: runtime-state registry and complete capture restore

Plan task 1b-T04 (SPEC §8.1 "Runtime-state registry", "Capture warm-ups leave no trace",
§12.2 "Runtime state"). FastMACE branch `fastmace/1b-T04`; no research-engine baseline is
involved (the registry changes no numbers).

## What the checks are

- **Registry against mutations** (`fastmace.engine.registry.unregistered_mutations`): every
  tensor reachable from the engine whose version counter a stage moves (A, C, B, the pipelined
  ACP and its forced redo, the speculative ACB, a rebuild) must be registered; an unregistered one
  is reported by its attribute path. Eager engines only: captured replays do not move version
  counters.
- **Warm-ups leave no trace**: with every sticky flag set, the full registered state (every
  `capture` buffer) is bitwise identical before and after the warm-up runs of every stage.

## CPU

### Local PC, `fastmace_cpu` (19 October 2026)

WSL2 Ubuntu, `~/environments/fastmace_cpu` (torch 2.14.1+cpu, MACE 0.3.16; no `graph_longrange`,
so the polar cases skip), FastMACE `2d664e9`, single files serially (1b-T24 was timing):
`test_registry` 35 passed, 3 skipped; `test_engine` 30; `test_forces` 39 (20 skipped);
`test_polar` 19 (14 skipped); `test_integrators` 32 (8 skipped); `test_planner` 79 (20 skipped);
`test_research_surface` 15; `test_layering` 32; `test_contracts` 72. Ruff clean.

Trajectories against `main` before the task (`origin/main` 36f9532, after 1b-T03 and 1b-T05):
tiny MH-1 (`tests/_support` stand-in, Na/Cl), NaCl 2×2×2 rattled, 600 K, 60 pipelined steps,
NVT and NPT, CPU float64: positions, forces and cell **bitwise equal**.

### leto, `mace_cueq_latest` (19 October 2026, 03:32 CEST)

leto (52 CPUs, 250 GB), CPUs only (`CUDA_VISIBLE_DEVICES=`, `nice -n 19`; another user's GPU job
was running and the GPU was not used), `/mnt/homes/zwingel/environments/mace_cueq_latest`
(Python 3.10.12, torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12.0, graph-longrange 0.4.0),
FastMACE `2d664e9` copied to `/mnt/homes/zwingel/fastmace-t04`:

```bash
nice -n 19 $E/bin/python -m pytest -n 8 -p no:cacheprovider -q \
  --basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-t04-cpu tests/cpu
```

Result: **1192 passed, 176 skipped** in 314 s. `tests/cpu/test_registry.py` alone: 38 passed,
no skips (the polar cases, NVT and NPT, run here).

## GPU: murphy (19 October 2026, 05:12 CEST)

- Machine: murphy, NVIDIA RTX 4000 Ada Generation 20 GB, driver 595.58.03; idle before every
  job (no compute processes of anyone).
- Environments: pytest in NAS `mace_cueq_latest` (torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance
  0.12.0); hybrid children in NAS `mace_cueq_oeq` (`FASTMACE_HYBRID_PYTHON`, OEq 0.7.0).
- FastMACE: branch `fastmace/1b-T04` after the sync with main 36cee51 (merge `44f9655`, plus
  the test fixes below), copied to `/mnt/homes/zwingel/fastmace-t04`; models from
  `/mnt/homes/zwingel/ResearchModule/Models`.
- Structure: NaCl 3×3×3 (216 atoms), 300 K, 0.5 fs, seed 20261018. Each case is planned and built
  by `FixedPlanner` from `main.py`'s defaults in a fresh process, then runs 20 steps.

```bash
$E/bin/python -m pytest -p no:cacheprovider -rs -v -s \
  --basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-t04-gpu tests/gpu/test_registry_gpu.py
```

Result: **4 passed** (197 s). JSON: `results/1b-T04-registry-murphy-20261019/`.

| Case | Backend, precision | Graphs recaptured | Registered tensors | Warm-up mismatches | Unregistered mutations |
| --- | --- | --- | --- | --- | --- |
| MH-1 `matpes_r2scan` NVT | hybrid float32, pipelined | ACP, B 7168, AC, C | 33 | none | none |
| MH-1 NPT BAOAB, full cell | hybrid float32, pipelined | ACP, B 7168, AC, C | 39 | none | none |
| POLAR-1-S NVT | hybrid float32, pipelined | AC, C, ACP, B 7168 | 40 | none | none |
| MH-1 plain cuEq | cueq float64, pipelined | ACP, B 7168, AC, C | 33 | none | none |

Sticky flags set to 1 before the recaptures: `engine.flags`, `pipe_flags`, `spec_flags`,
`rebuild_needed`, and for POLAR-1-S also `force_field.polar.overflow`; all came back unchanged,
at the same addresses. POLAR-1-S had no graphs left after its 20 steps (a neighbour rebuild that
grew the capacity cleared them), so the test recaptures ACP and the current B bucket explicitly.

The same job also ran the four modes of `tests/gpu/test_engine_gpu.py::test_capture_warmups_leave_the_registered_state`
(Lennard-Jones; checked, pipelined, speculative, NPT): 4 passed.

**Test fix found on the GPU.** The first run failed for both hybrid MH-1 cases while the test
recaptured B: `During CUDA graph capture, autograd node 'torch::autograd::AccumulateGrad' has a
stale reference to the default stream`. The cause was the test's own "before" copies,
`tensor.clone()` of the force field's positions leaf (`requires_grad=True`), made on the default
stream. A probe on main and on the branch (recapturing every graph after 20 steps, without those
copies) passed on both. The copies are now `detach().clone()`, as `Registry.snapshot` already
does. Anyone copying registered buffers (the gate, 1b-T16) has to detach them as well.
