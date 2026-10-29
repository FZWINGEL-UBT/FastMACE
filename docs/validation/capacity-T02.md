# CAP-T02: rule 12 keeps MH-1's fused path, and the estimate predicts the measured peak

Plan task CAP-T02 ([plans/capacity.md](../../plans/capacity.md)), decided by the owner on 29 October 2026: "It
should only do this if memory would not be enough" (SPEC §18, decision 67). The capacity scan
([capacity-16gb.md](capacity-16gb.md)) had found that rule 12 predicted about 1.8 times the measured peak, and from
about 13k atoms turned off `streaming_conv` with `chunked_conv_backward`, `fused_folded_forward` and
`wprime_backward`, because its term (+8 bytes per edge) "saved about 7 MiB".

## Verdict

**New limits on the RTX 5060 Ti 16 GB (WSL2), MH-1 float32, default plan, 220 MD steps:**

| Cell | Before | After | ms per step at the new limit |
| --- | --- | --- | --- |
| NaCl | 21,952 atoms (0.70M edges) | **27,000 atoms (0.86M edges), +23 %** | 288 |
| NbCl5 | 27,000 atoms (0.71M edges); 28,800 with CAP-T01 | **30,600 atoms (0.80M edges), +13 %** | 291 |

The next sizes fail during the build: NaCl at 32,768 atoms, NbCl5 at 36,720.

The fused path is also faster on large cells:

- NaCl at 21,952 atoms: 236 against 274 ms per step.
- NbCl5 at 25,200 atoms: 242 against 291 ms per step (`streaming_conv=on` against the old default).

**Cells that failed before now run.** With the fused path kept, these cells pass:

| Cell | Before | Now |
| --- | --- | --- |
| NaCl float64, 8,000 atoms | failed during the warm-up | 825 ms per step |
| SrTiO3 float32, 14,700 atoms | failed in the steady phase | 203 ms per step |

**The estimate now predicts the peak:**

| Data | Before | After |
| --- | --- | --- |
| Capacity cells (NaCl, NbCl5, SrTiO3, 2,016 to 30,600 atoms, float32 and float64) | 1.5–2.3 × the measured peak | 1.05–1.34 × |
| The 60 earlier small-cell cases (5-T03) | 1.05–1.48 × | 1.05–1.27 × |

- The capacity cells were measured on the RTX 5060 Ti under WSL2 (23 cases) and the RTX 4000 Ada under native
  Linux (5 cases).
- The small-cell cases stay within their [1.0, 1.5] bound.
- At the new limits rule 12 warns, but turns nothing off.

**Gate:** passed on NaCl, NbCl5 and NaNbO3; numerics are unchanged, since only the plan's choices at large sizes change.

## Why turning the fused path off made things worse

Under WSL2, device memory outside PyTorch's allocator grew to 2–7 GiB on large cells
([capacity-T03.md](capacity-T03.md)). The memory probe (`tools/bench/graph_memory_probe.py`) and the bisection
found the cause:

- **Plain CUDA graphs, cuBLAS and Triton kernels add nothing.** Their device memory stays at about 1.15 GiB with
  and without expandable segments, including 20,000-node graphs.
- **A kernel that spills to local memory reserves its stack for every resident thread.** In the probe, a kernel
  with 17,232 bytes of stack per thread costs 0.89 GiB: 17,232 × 1,536 threads × 36 SMs. The driver keeps that
  reservation after the kernel and after the graph are gone, and lowering `cudaLimitStackSize` does not give it
  back.
- **The bisection on NaCl float64 at 5,832 atoms**, steady-state memory outside the allocator:

  | Plan | Outside the allocator | ms per step |
  | --- | --- | --- |
  | Default | 1.80 GiB | 617 |
  | `fp64_emulation=off` | 1.41 GiB | 1,320 |
  | `segment_gemm=off` | 1.80 GiB | 620 |
  | **`streaming_conv=off`** | **6.70 GiB** | 693 |

  With the streaming convolution off, the hybrid backend runs OEq's own convolution kernels. OEq compiles them at
  run time with NVRTC, so `cuobjdump` cannot read them, but the reservation implies about 100 KB of stack per
  thread in float64.
- **The libraries are not the cause.** The precompiled kernels' stacks are at most 48 B (cuEquivariance) and 1 KiB
  (cuBLAS triangular solves). Inductor's fused kernels reach at most 5.3 KB.

So rule 12's drop at about 13k atoms put the step on a path that needs more memory: more per-edge activations, and
under WSL2 several GiB of stack. That is where the limits of the capacity scan sat.

## What changed

- **Measured terms** (`fastmace/rewrites/memory.py`). `chunked_conv_backward` and `radial_tables` declare the
  per-edge tensors they do not keep: −521 and −576 floats per edge. These are the changes in the peak when each is
  turned off alone (NbCl5, 14,976 atoms, float32):

  | Record turned off | Peak change per edge | ms per step |
  | --- | --- | --- |
  | `chunked_conv_backward` (with `fused_folded_backward`) | +2,086 B | 163 instead of 145 |
  | `radial_tables` | +2,306 B | 181 instead of 145 |
  | `streaming_conv` (with its dependants) | +1,981 B | |
  | `fused_folded_forward` | about 0 | |
  | `wprime_backward` | about 0 | |

- **Fitted term.** MH-1's `streaming_conv` declares the rest of the fused path's difference from the estimate's
  stock structure: −2,000 floats per edge in float32 and −673 in float64, plus its 8 bytes of int32 indices. It is
  fitted so that every capacity cell stays at or above 1.05 times its measured peak. The estimate's stock per-edge
  structure was 18.9 kB per edge against 5.9 kB measured. Its per-atom term, 286 kB against 268 kB measured, was
  already right.
- **Drop rule** (`planner.minimal._rule_12`). A rewrite is turned off for memory only while the estimate does not
  fit, and only if turning it off with its dependants lowers the estimate. With the fused path's terms negative,
  turning `streaming_conv` off would raise the estimate, so rule 12 keeps it.
- **SPEC:**
  - decision 67 in §18 and §19;
  - rule 12's row in §9.3: "only when the estimate with them on does not fit", plus measured or fitted per-path
    terms;
  - decision 51's bullet, amended to match.
- **Tests:**
  - `tests/cpu/test_memory_fit.py` checks the 28 capacity cases
    (`tests/_support/data/rule12_capacity.json`), each between 1.0 and 1.4 times its peak.
  - `tests/cpu/test_memory_estimate.py` checks the new terms' signs and records, and the guard: a group whose
    removal does not lower the estimate stays on.

**Memory still outside the allocator:** with the fused path on, 3–4 GiB under WSL2 on the largest cells (other
kernels' stacks). It is still a WSL2 cost, and the next lever there (CAP-T04).

## Checks

- **Capacity:** the scans above (`results/cap-t02-20261029/wsl/t02/`), default plan from commit `af07f921`.
- **Native Linux** (leto, RTX 4000 Ada 20 GB, commit `b576989d` with the new terms), estimate against the
  measured peak:

  | Cell | Predicted / peak | ms per step |
  | --- | --- | --- |
  | NaCl float32, 21,952 atoms | 1.26 | 259 |
  | NbCl5 float32, 25,200 atoms | 1.19 | 282 |
  | NbCl5 float32, 30,600 atoms | 1.19 | 344 |
  | SrTiO3 float32, 13,720 atoms | 1.21 | 212 |
  | NaCl float64, 5,832 atoms | 1.32 | 564 |

  All fit, and nothing was turned off. leto's GPU was used from 19:46 to 19:58 UTC, with the idle check.
- **Force/stress gate** (local RTX 5060 Ti, default plan, commit `af07f921`): passed on NaCl, NbCl5 and NaNbO3.
- **Golden plan texts and targeted CPU tests on leto** (GPU hidden, `-n 8`, commit `7547c11c`):
  - The golden plan texts did not change.
  - 586 passed and 28 skipped, in `test_memory_fit`, `test_memory_estimate`, `test_fallback_memory`,
    `test_plan_text`, `test_planner`, `test_minimal_planner`, `test_report`, `test_rewrites_mh1`,
    `test_rewrites_fusion`, `test_config` and `test_docs_options`.
  - Four rule 12 tests were updated for decision 67. Turning `factor_species` off now raises the estimate, because
    it takes the fused path with it, so the tests' tight plans turn off `reassociate_first` instead.

## Setup

- **Local:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130),
  `tools/ci/local_gpu.py run` reservations with the idle check. MH-1 (`mh1-r2scan`), Langevin NVT, 20 warm-up and
  200 timed steps, one process per point (`tools/bench/capacity_scan.py`).
- **leto:** RTX 4000 Ada 20 GB (driver 595.99.02), NAS `mace_cueq_oeq`
  (`results/capacity-16gb-20261029/ws.sh`).
- **Raw data:** `results/cap-t02-20261029/`:
  - `wsl/t02/`: the new limits;
  - `wsl/terms/`: the per-record measurements;
  - `wsl/bisect/` and `wsl/bisect64/`: the outside-memory bisection, with `-scon` for `streaming_conv=on` at the
    failing sizes;
  - `wsl/probe/`: the plain-CUDA probes;
  - `leto/`: the Linux points.
