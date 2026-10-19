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

## GPU

Pending: MH-1 NaCl 3×3×3 hybrid pipelined NVT, MH-1 NPT BAOAB (full cell), POLAR-1-S and MH-1
plain cuEq float64 (`tests/gpu/test_registry_gpu.py`), on the GPU the conductor assigns.
