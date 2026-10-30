# CAP-T07b: recompute_conv_weights with the w-rebuilding kernels

Plan task CAP-T07b ([plans/capacity.md](../../plans/capacity.md)), the follow-up recommended in
[capacity-T08-inventory.md](capacity-T08-inventory.md) section 2. The conductor's instruction (29 October 2026) was:

- reimplement `recompute_conv_weights` as the store-free fused forward plus `WRebuildBackward` (one path part, four
  edges per batch);
- keep decision 67's planning rule: on only where the estimate floor says the plan cannot fit, plus the
  out-of-memory rebuild ([capacity-T07.md](capacity-T07.md));
- update rule 12's term.

## Verdict

**`recompute_conv_weights` is now this implementation wherever `fused_folded_forward` is on, MH-1's default.** It
wins everywhere it was measured:

- It is faster than CAP-T07's chunked recompute: 0.93–0.94× its step time.
- It saves more memory: the force call's peak per edge drops by 5,126 B, against 3,409 B with the chunked recompute.
- It raises the limit further on the 16 GB RTX 5060 Ti, float32: NbCl5 goes from 36,720 to **49,248 atoms (+34 %)**,
  and NaCl from 27,000 to **46,656 atoms (+73 %)**.

Its cost against recompute off is 1–4 % of the step time. The planning rule is unchanged (decision 67): the record
stays a memory rescue and is not a default.

**New limits on the RTX 5060 Ti 16 GB (WSL2), MH-1 float32, default plan, 220 MD steps:**

| Cell | Before (CAP-T07) | After | ms per step at the new limit | First failure |
| --- | --- | --- | --- | --- |
| NbCl5 | 36,720 atoms | **49,248 atoms (6×6×19), 1.29M edges at the cutoff** | 524 | 51,840 (6×6×20): out of memory at build |
| NaCl | 27,000 atoms (CAP-T02; not rescanned in CAP-T07) | **46,656 atoms (18³), 1.49M edges at the cutoff** | 527 | 54,872 (19³): out of memory at build |

- Rule 12 planned the 36,720- to 49,248-atom NbCl5 cells and the 32,768- to 46,656-atom NaCl cells with recompute
  on, because their estimate divided by 1.34 did not fit (`memory.short_by_far`).
- NaCl at 32,768 atoms, the size that failed in CAP-T02, now runs at 343 ms per step with 9.92 GiB reserved.
- At 49,248 atoms the estimate with recompute was 18.35 GiB and the run reserved at most 14.32 GiB, 1.28× below the
  estimate and inside decision 67's band.

**Exactness: rounding, and the gate passed.**

- The forward is unchanged bitwise: the message and density equal the stored-weights path's.
- The rebuilt weights are bitwise the stored ones (the same k-ordered `fmaf` chain).
- `dsh` and the distance gradient are summed over the 32-channel slices with atomic additions. Their order differs
  from the stored-weights kernel's, so they agree to float32 rounding, not bitwise. The record's exactness changes
  from "exact" to "rounding".
- The force/stress gate passed with recompute on (see Checks).

## Step time

`tools/ab_rewrites.py`, same process, MH-1 hybrid float32 NVT at 300 K, everything else the default plan. Median step
in ms. Order `ab` uses the abab block pattern and order `ba` uses abba. Ratios are b/a, with the range over block
pairs in brackets.

**Rebuild (b) against CAP-T07's chunked recompute (a)**, both `recompute_conv_weights=on`; a sets
`fastmace.rewrites.radial:RECOMPUTE_KERNEL='chunked'`:

| Cell | Blocks | order ab: a / b, b/a | order ba: a / b, b/a |
| --- | --- | --- | --- |
| NbCl5 3×3×3, 1,944 atoms | 12 × 300 steps | 19.34 / 18.18, **0.940** (0.933–0.963) | 19.49 / 18.39, **0.944** (0.927–0.967) |
| NbCl5 4×4×8, 9,216 atoms | 8 × 100 steps | 95.62 / 89.89, **0.940** (0.927–0.951) | 95.90 / 89.48, **0.933** (0.924–0.944) |

**Rebuild (b) against recompute off (a):**

| Cell | Blocks | order ab: a / b, b/a | order ba: a / b, b/a |
| --- | --- | --- | --- |
| NbCl5 3×3×3, 1,944 atoms | 12 × 300 steps | 17.67 / 18.22, **1.031** (1.025–1.035) | 18.02 / 18.67, **1.036** (1.026–1.043) |
| NbCl5 4×4×8, 9,216 atoms | 8 × 100 steps | 88.21 / 89.86, **1.019** (1.012–1.029) | 88.25 / 89.41, **1.013** (1.008–1.021) |

Every run was valid: nothing compiled while timed, and no other process used the GPU.

- **Against CAP-T07's chunked recompute**, which cost 1.086–1.097× against off, the cost falls to 1.013–1.036×.
- **The cost shrinks with size.** The rebuild work is per edge, as in the stored-weights backward. The remaining cost
  is the second radial-table pass and the `w` build in the backward kernel.

## Memory

NbCl5 5×5×18 (32,400 atoms), one process per row, 20 warm-up and 200 timed steps (`tools/bench/capacity_scan.py`):

| | Off (CAP-T07) | Chunked recompute (CAP-T07) | Rebuild (this task) |
| --- | --- | --- | --- |
| Largest reserved memory | 13.80 GiB | 11.07 GiB | **9.51 GiB** |
| Force call's bytes per edge | 15,582 | 12,173 | **10,456** |
| Saved per edge against off | | 3,409 | **5,126** |
| ms per step (separate processes) | 313 | 338 | 322 |

The reserved memory here is the allocator's largest reservation over the run, in GiB (2^30 bytes). capacity-T07.md
printed the same runs as 13.92 and 11.12. Those were the capture records' MiB divided by 1,000, not GiB.

**Rule 12's term needs no new number.** It declares the full weights: 1,280 floats, 5,120 B per edge. The rebuild
saves 5,126 B per edge, while the chunked recompute reached two thirds of that.

- The chunked recompute did not reach the full saving because it still had a `w` transient, and without the stored
  weights the peak moved to other tensors.
- With the rebuild, `w` never exists in memory in either pass.
- `fastmace.rewrites.memory.recompute_conv_weights` keeps its value; its docstring now records the measurement.
- Without `fused_folded_forward` (the chunked path), the term still overstates the saving by about 1.5×, as before.

## What changed

- **Fused forward** (`fastmace/kernels/oeq_streaming/fused_forward.py`):
  - `FusedFoldedForward(..., store_w=False)` renders the kernel with `STORE_W = false`: it builds `w` on chip, as
    before, and does not write it out.
  - New custom op `fastmace::fused_folded_message` (message only, with a fake).
- **Backward** (`fastmace/kernels/oeq_streaming/wrebuild_backward.py`):
  - New custom op `fastmace::wrebuild_backward`. It recomputes the radial basis `a` from the edge lengths with the
    forward's table kernel, bitwise. It then runs `WRebuildBackward`, which builds `w` per 32-channel slice in
    shared memory and returns `dx`, `dsh` and the distance gradient.
- **`wprime_backward`** (`fastmace/kernels/oeq_streaming/wprime_backward.py`):
  - New method `WPrime.attach_rebuild(kernels, fused)` compiles both kernels; the launch is `REBUILD_PARTS = 1`,
    `REBUILD_BATCH = 4`.
  - With them attached, `convolution(..., recompute_rows)` runs the message-only forward and saves an empty
    placeholder, and its backward calls `wrebuild_backward`.
- **Record** (`fastmace/rewrites/radial`):
  - `attach_recompute` attaches the rebuild in every interaction with `fused_folded_forward`. Others keep the
    chunked cuBLAS recompute.
  - The module constant `RECOMPUTE_KERNEL` ("rebuild" or "chunked") exists for the same-process A/Bs only.
  - Exactness changes to "rounding". The self-check runs through the interaction's fused forward, so it checks the
    rebuild.
- **Docs:** `docs/reference/rewrites.md` (the row, which also gains its missing exactness column), the CHANGELOG
  entry and the plan row.

## Checks

- **GPU kernel test** (`tests/gpu/test_recompute_conv_weights_cuda.py`, local RTX 5060 Ti): passed, 10 cases. A
  rebuild case was added per layout (`ir_mul`, `mul_ir`):
  - message and density are bitwise the stored-weights path's;
  - `dsh` and the distance gradient agree to rounding;
  - `dx` max relative difference 2.5e-7 and 1.9e-7;
  - the CUDA-graph replay agrees.
- **Force/stress gate, recompute on** (`tools/force_stress_gate.py --engine fastmace --rewrites
  recompute_conv_weights=on`, commit `5bf1ea8b`): **passed** on all three cells. Error against float64, with its
  ratio to the float32 ASE calculator's error in brackets:

  | Cell | energy (eV) | forces (eV/Å) | stress (eV/Å³) |
  | --- | ---: | ---: | ---: |
  | NaCl | 1.2e-4 (0.17) | 3.6e-6 (0.96) | 3.6e-8 (0.40) |
  | NbCl5 | 5.7e-5 (0.08) | 8.4e-5 (1.07) | 4.0e-8 (0.40) |
  | NaNbO3 | 6.1e-4 (0.50) | 1.3e-4 (1.01) | 2.3e-7 (0.77) |

- **CPU tests on leto** (GPU hidden, `-n 8`, commit `5bf1ea8b`): 702 passed, 30 skipped; the golden plan texts are
  unchanged.
  - The files: planner, memory estimate and fit, rewrites, report, config, docs options, plan texts, fallback, rules,
    `test_memory_rescue_retry` and `test_reproducible`.
- **NVE drift:** not run. The per-edge values match the stored-weights path to rounding, as with
  `wprime_backward` itself.

## Setup

- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130).
- **GPU access:** `tools/ci/local_gpu.py run` reservations, with the idle check before every job.
- **Source:** a pinned copy of FastMACE `5bf1ea8b`. No baseline, since both sides are FastMACE.
- **Model:** MH-1 (`mh1-r2scan`), hybrid backend, float32, Langevin NVT at 300 K.
- **Raw records** in `results/cap-t07b-20261030/`:
  - the A/Bs: `ab/`;
  - the gate: `gate-metadata.json` and `gate-summary.txt`;
  - the capacity points and scans: `points/`;
  - the run script: `local.sh`.
