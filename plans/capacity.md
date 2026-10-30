# Capacity: more atoms per GPU on the 16 GB card

Owner request, 29 October 2026 (through the conductor): optimise FastMACE so that larger cells fit on the local
RTX 5060 Ti 16 GB. Step time comes second but still counts. The starting point is the capacity scan
([report](../docs/validation/capacity-16gb.md), `tools/bench/capacity_scan.py`): NVT MD through `fm.Simulation`,
MH-1 (`mh1-r2scan`), the default plan, 200 steps after 20 warm-up steps, one fresh process per size.

Owner, same day: the work targets float32. The largest cells will be float32 runs. float64 is checked once at
the end, on the final plan.

## What the scan found

| Structure (float32) | Largest passing | Edges at 6 Å | Peak reserved | ms/step | Limit |
| --- | --- | --- | --- | --- | --- |
| NaCl | 21,952 | 702k | 12.98 GiB | 274 | first capture of the force graph (B), 27,000 atoms |
| NbCl5 | 27,000 | 710k | 14.32 GiB | 320 | capture of a second, larger B variant in the steady phase, 28,800 atoms |
| NaCl, `streaming_conv=on` | 27,000 | 864k | 13.06 GiB | 275 | first B capture, 32,768 atoms |
| NbCl5, `streaming_conv=on` | 30,600 | 804k | 13.08 GiB | 292 | 36,720 atoms |

For float64 the default plan reaches NaCl 5,832 atoms (187k edges) and NbCl5 7,128 atoms (187k edges). Both
fail while capturing the second B variant (warm-up), with 5.9 GiB still free after the build.

Where the memory goes at the limit:

- **Allocator, NaCl float32 at 27,000 atoms, in the failing first B capture**:
  - 8.7 GiB live and 15.1 GiB reserved, so about 6 GiB sits cached or fragmented.
  - Model activations inside the force call: 5.5 GiB.
  - Outside FastMACE (MACE and e3nn): 1.0 GiB.
  - `segment_gemm` joined products: 0.82 GiB.
  - The registry snapshot taken around the capture: 0.49 GiB.
  - The Langevin noise batch: 0.31 GiB.
  - The radial table: 0.21 GiB.
  - Neighbour-list buffers: 0.17 GiB.
- **Outside PyTorch's allocator**: 1.13 GiB at the start (the context and the desktop). It grows to 2.2 GiB
  in float32 runs and 1.4-1.8 GiB in float64 runs (libraries, graph executables, compiled modules).
- **Peak against size.** Least squares over the default plan's passing float32 points where rule 12 dropped
  nothing:
  - NaCl, NbCl5 and SrTiO3, at 32, 26 and 74 edges per atom.
  - Fit: 0.23 GiB + 5.9 kB per edge + 268 kB per atom (`max_allocated`, residuals at most 0.07 GiB).
  - At the NaCl limit (22k atoms, 0.70M edges), the atoms' share (5.5 GiB) is larger than the edges' share
    (3.9 GiB).
  - 268 kB per atom is about eight node-feature tensors of MH-1's width (512 channels x 16 components, float32)
    kept for the backward.
- **Failure mode**: every failure is a raw driver allocation during a graph capture. It leaves a sticky CUDA
  error, so the fallback's replanning then fails too. The run crashes instead of being refused.

Rule 12 predicts about 1.8 times the measured peak at these sizes. At 21,952 NaCl atoms it predicts 23 GiB
against 13 GiB measured. Above about 13k atoms it therefore turns off `streaming_conv` (term: +8 B per edge,
"saves about 7 MiB") together with `chunked_conv_backward`, `fused_folded_forward` and `wprime_backward`. That
lowers the measured limit and slows large cells. The run then warns anyway.

## Rules for every task

- **Exactness.** Each change is an exact or rounding-class rewrite, or an engine or planner option.
- **Gates.** The force/stress gate runs on the local GPU. NVE drift runs too if numerics change.
- **Capacity.** The new maximum (atoms and edges) is measured with `tools/bench/capacity_scan.py` on NaCl and
  NbCl5 (and SrTiO3 when it helps), in float32.
- **Speed.** A same-process A/B in both build orders runs on NaCl, NbCl5 and NaNbO3 at normal sizes, and at the
  new large sizes.
- **Defaults.** A change that raises the limit and costs no step time at normal sizes becomes the default.
  Otherwise the planner switches it on above a threshold it computes from free memory and its estimate (general,
  not a per-structure table). A per-GPU default by exact device name is allowed where A/Bs show it faster
  (decision 66). Anything that needs a SPEC §18 decision goes to the conductor first.
- **CPU tests** run on leto only.
- **Records.** Every candidate is written up in `docs/validation/capacity-*.md`, including the ones that did
  not help. Each experiment is committed and pushed.

## Tasks, in order of expected gain per effort

| Task | What | Expected gain (float32 unless noted) | Risk to step time | Status |
| --- | --- | --- | --- | --- |
| **CAP-T01** | **One force graph near the limit.** When the edge count outgrows the captured padded size, release the existing B graphs and their pool before the warm-up and capture of the larger one, so only one set of activations lives at a time. Engine option `EngineConfig.single_force_graph` (`"auto"`: on when rule 12's estimate of two live force graphs does not fit the free memory, after the first capture's measured `capture_memory`). No numerics change | removes the second-variant failure: NbCl5 28,800 → about 30-31k (until the first capture limits); float64 NaCl about 5.8k → about 9-10k | one extra capture each time the padded edge size grows (rare after the first rebuilds; measured as part of the A/B); none below the threshold | **done, default `"fit"`**: NbCl5 float32 27,000 → 28,800 atoms (0.71M → 0.76M edges); A/B at normal sizes 0.997/1.004 (NaCl), 1.000/0.996 (NbCl5); gate passed. NaCl float64 and SrTiO3 not helped: their limit is the driver failing to map memory near a full card, not a second graph. [report](../docs/validation/capacity-T01.md) |
| **CAP-T02** | **Rule 12 must not drop `streaming_conv` and its dependants for memory.** The drop saves nothing (+8 B per edge against the fused convolution) and takes `chunked_conv_backward`, `fused_folded_forward` and `wprime_backward` with it. Measured with the rewrite kept on explicitly: NaCl 21,952 → 27,000 atoms, NbCl5 27,000 → 30,600, and faster per edge. Also recalibrate rule 12's `SAFETY` and per-path terms for the fused MH-1 paths (1.8x over at these sizes) so the estimate predicts the limit | **+23 % (NaCl) and +13 % (NbCl5) atoms, measured**; step time better at large sizes | none at normal sizes (rule 12 does not act there) | **done (decision 67)**: rule 12 keeps the fused path unless dropping it lowers the estimate; the estimate predicts 1.05-1.34x of the measured peak (was 1.5-2.3x). NaCl float32 21,952 → 27,000 atoms, NbCl5 → 30,600; NaCl float64 8,000 and SrTiO3 14,700 atoms now run. [report](../docs/validation/capacity-T02.md) |
| CAP-T03 | **Memory outside the allocator.** It grows from 1.13 to 2.2 GiB during the build. Find what holds it (cuBLAS and cuBLASLt workspaces, Triton and Inductor modules, OEq and cuEq handles, graph executables) with `cudaMemGetInfo` around each build step; then cap or share what can be (one cuBLAS workspace config, lazy module loading) | up to about 1 GiB, about 7 % atoms | none expected | **done, nothing to gain in FastMACE**: the extra memory is WSL2's, tied to live CUDA graphs (2-7 GiB; 0.3 GiB on native Linux for the same points); a wait after `empty_cache` was tried and reverted. Large cells: native Linux. [report](../docs/validation/capacity-T03.md) |
| CAP-T04 | **Fragmentation at the peak.** Reserved minus allocated is 1.6-1.9 GiB at the passing limit and about 6 GiB at the failing capture. Check that the B capture's warm-up does not leave pool blocks the capture cannot reuse (warm-up of the first capture outside the pool, then `empty_cache`), and whether a larger `max_split_size_mb` or a reserve-then-release before the first capture helps | about 1-2 GiB, about 7-15 % | none | **done, nothing to recover**: no inactive splits with expandable segments; the reserved-allocated gap is cached memory at different moments, and the capture options do not change it. NbCl5 float32 limit 32,400 atoms (the scan had skipped it). [report](../docs/validation/capacity-T04.md) |
| CAP-T05 | **Capture transients.** Around every capture the registry snapshots every captured buffer on the device (0.49 GiB at 27,000 atoms, at the moment of the peak). Snapshot to pinned host memory, or only the buffers the warm-up writes | about 0.5 GiB, about 3-4 % | a host copy per capture (captures are rare) | implemented on `claude/cap-t05-snapshot`: snapshot buffers of 1 MiB or more in pinned host memory; build peak −0.51 GiB on NbCl5. **Done** ([capacity-T05.md](../docs/validation/capacity-T05.md)): same-process A/B b/a 0.998–1.007 (no step-time cost), about 5 s more per build with captures (pinned host copies) |
| CAP-T06 | **Noise batch.** Langevin draws 512 steps of float64 noise ahead (`allocate_noise`, 0.31 GiB at 27,000 atoms). A batch sized from memory (for example 64 steps) near the limit, physics unchanged (same float64 stream) | about 0.27 GiB, about 2 % | the batch refill is amortised over fewer steps; A/B | **done** ([capacity-T06.md](../docs/validation/capacity-T06.md)): the batch capped at 64 MiB (`NOISE_BATCH_BYTES`), same stream, restarts with another batch size resume bitwise; −0.53 GiB at NbCl5 51,840 atoms (3.6 % of the reserved peak, no new grid step); step time 1.000–1.005 at 9,216 atoms |
| CAP-T07 | **Per-edge activations.** Profile what the force graph keeps per edge (5.9 kB per edge, about 1,500 floats). Then recompute cheap per-edge tensors in the backward (spherical harmonics, radial basis, the radial MLP's hidden layers) or chunk the remaining edge work above a size the planner computes from free memory. Exact if recomputed by the same kernels (bitwise) | about 40 % of the total on NaCl, 60 % on SrTiO3 | recompute costs time, so it switches on only above the threshold | **done** ([capacity-T07.md](../docs/validation/capacity-T07.md)): `recompute_conv_weights`, exact, −2.8 GiB at 32,400 atoms, +9–10 % step time. On only when memory would not be enough (rule 12 for cells that cannot fit; otherwise a rebuild after an out-of-memory error). NbCl5 float32 limit 34,200 → 36,720 atoms. **CAP-T07b done** ([capacity-T07b.md](../docs/validation/capacity-T07b.md)): reimplemented with the store-free fused forward and the w-rebuilding backward, rounding, gate passed; 0.93–0.94× the chunked recompute's step time, 1.01–1.04× against off; 5,126 B per edge saved (the full term). Limits: NbCl5 49,248 atoms, NaCl 46,656 |
| CAP-T08 | **Per-node activations.** 268 kB per atom, roughly eight node-feature tensors (512 x 16 float32) kept for the backward (product basis, symmetric contraction, `segment_gemm` products, the gate's rows). Recompute per interaction in the backward (checkpointing), or keep fewer of them, above the threshold | **the largest lever on NaCl and NbCl5: about 55-60 % of the total** | recompute: one more forward per interaction | open, falsifiers run ([capacity-T08-inventory.md](../docs/validation/capacity-T08-inventory.md)): 174.5 kB/atom saved plus 134-139 kB/atom backward working set (compiled plan); interaction-boundary checkpointing lowers the peak by 3.6 % at most (stopped); the lever is a split inside interaction 1. Also stopped there: w rebuilt in the backward (1.16-1.20x, but better than CAP-T07's mechanism), smem spilling (no spills), FFMA2 (not on sm_120) |
| CAP-T09 | **Neighbour list.** The pair capacity is 2x the cutoff edges (candidates within cutoff + skin, times headroom 1.15). 104 B per slot: 0.14 GiB at 1.4M slots. int32 indices save about 45 MB | about 0.3 % | none | **done on `fastmace/cap-nlist`** ([capacity-nlist.md](../docs/validation/capacity-nlist.md)), exact, default: int32 candidate indices, no stored candidate shifts (60 instead of 104 B per slot), chunked GPU build (a rebuild's temporaries 172 → 51 B per pair, outside the graph pool). NbCl5 49,248 → 51,840 atoms; NaCl 49,248 on both sides (0.52 GiB lower peak there); same-process A/B 1.00 |
| CAP-T10 | **Clean refusal.** A capture predicted to exceed the free memory is refused before it starts (decision 51's refusal path), because an out-of-memory error inside a capture leaves a sticky CUDA error and the fallback cannot recover. Needs T02's calibrated estimate | no atoms; a clean error instead of a crash | none | after T02 |
| CAP-T11 | float64 check: the final plan's float64 limit on NaCl and NbCl5 | | | at the end |

Owner, 29 October: develop and measure on one structure (NbCl5) for speed; the other cells before a default
change only when the conductor asks.

Order of work: T01 (done), then T03, T04, T05 and T06 (measure-first, small), then T08 and T07 (the large
levers, after one per-tensor profile of the force graph's saved tensors). T02 and T10 wait for the conductor's decision on rule 12.
