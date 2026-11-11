# The skin list's `edge_index` in int32 (10–11 November 2026)

The compacted `edge_index` of `fastmace.neighbors.skin.SkinNeighbourList` stored as int32 instead of int64
(`skin.EDGE_INDEX_INT32`, default `False` on branch `fastmace/edge-index-int32`). The candidate list was already
int32. The rung-(B) bound put the saving at 0.8 kB/atom (0.37 GiB at SrTiO3 36³;
[rung-b](../rung-b-20261109/README.md)).

OMAT-0 medium float32, the default plan, SrTiO3, unless stated otherwise. Ada = leto's RTX 4000 Ada 20 GB (driver
595.99.02, NAS `mace_cueq_oeq`). 5060 Ti = the local RTX 5060 Ti 16 GB (WSL, `mace_cueq_oeq`, the default allocator,
through `tools/ci/local_gpu.py`).

## Summary

| Check | Result |
| --- | --- |
| Same-process A/B, SrTiO3 10³, 1 fs, skin 0.5, 6 × 256 steps, b/a (ab / ba order) | Ada **1.0001 / 0.9987**; 5060 Ti **1.0022 / 0.9959**: not slower |
| Memory, Ada SrTiO3 34³ (196,520 atoms, MD profile) | peak reserved 13.40 → **12.95 GiB** (73.2 → 70.7 kB/atom), peak allocated 10.50 → 10.36 GiB |
| Cube-step limits | unchanged (the 0.45 GiB does not reach the next SrTiO3 cube on the Ada or the 5060 Ti) |
| Reproducible plan (MH-1 `mh1-r2scan`, SrTiO3 6³, 20 steps NVT and NPT) | off/off **bitwise**; off/on forces 3.8e-5 (NVT) and 2.5e-5 (NPT) eV/Å, stress 1.3e-8, energy bitwise: a **rounding change** (the scatter order in the int32 index path) |
| Production plan | off/off differs by the plan's atomics (about 3e-6 eV/Å single point, 1e-4 after MD), off/on the same: no visible difference |
| 3× float32 gate (`tools/force_stress_gate.py`, Ada, `--production-constant ...EDGE_INDEX_INT32=True`) | **passed**: SrTiO3 3×3×3 forces 0.79, stress 0.47, energy 1.3e-3 (floor 0.01); NaCl forces 0.91, stress 1.00, energy 1.3e-4 |

Decision 66: a rounding change needs the gate, and it passed; the A/B is not slower on either GPU. The A100 has no
A/B yet. Adoption (default on, or kept opt-in for 1 % of memory that moves no limit) is the conductor's call.

## Files

- `ada/ab-10x10x10-{ab,ba}.json`, `ada/ab-run.log`: the Ada A/B (`tools/bench/ab_edge32_leto.sh`, source `8ff6f43c`).
- `local/ab-10x10x10-{ab,ba}.json`, `local/ab-run.log`: the 5060 Ti A/B (`tools/bench/ab_edge32_local.sh`).
- `ada/prof34-run.log`: the 34³ memory profiles (`tools/bench/prof_edge32_leto.sh`).
- `ada/reproducible-run.log`: the reproducible MH-1 comparison (`tools/bench/edge32_leto.sh`, `tools/bench/edge32_probe.py`,
  source `79cceac0`).
- `ada/gate.log`: the gate (`tools/bench/gate_edge32_leto.sh`, source `79cceac0`).
