# Memory profile at the Ada capacity limit: OMAT-0 medium, SrTiO3 (9 November 2026)

Why FastMACE fit fewer atoms than Symmetrix-XL's streaming path: FastMACE about 85 kB/atom on the RTX 4000 Ada,
Symmetrix-XL about 65 kB/atom (486,680 SrTiO3 atoms on a 32 GB RTX 5090 and 1.25 M on an A100 80 GB, from a
one-force-call probe). The profile below breaks the peak down by owner, finds one wasted buffer (fixed here,
`tabled_outer_sum.SHAPE_CARRIER`), and ranks what is left.

Source main `4ffe4f6f` (+ the profiler; + the fix where noted). GPU: leto's NVIDIA RTX 4000 Ada Generation (20 GB,
19.54 GiB usable, driver 595.99.02), NAS `mace_cueq_oeq`, hybrid backend, float32, OMAT-0 medium, default plan
(rescue rung `recompute_first_table` at these sizes), `expandable_segments:True`. Each case ran alone on the GPU
(idle check before each, GPU processes logged every 30 s: only the case's own processes).

## Summary

- **At 36³ (233,280 atoms) the MD run reserves 86.6 kB/atom at its peak** (18.81 GiB; plus 0.23 GiB outside the
  allocator). The slope between 32³ and 36³ is 83.7 kB/atom with an intercept of 0.64 GiB: almost everything scales
  with the atoms.
- **A single force call (no graphs, no MD) allocates 62.7 kB/atom at its peak, about Symmetrix-XL's 65 kB/atom**, but
  reserves 83.0 kB/atom: the allocator's segments hold 4.4 GiB more than the live tensors.
- **In MD the force pass lives in the CUDA graph's private pool: 70.4 kB/atom reserved for a live peak of
  50.6 kB/atom.** The difference (19.9 kB/atom) is packing loss inside the pool's one expandable segment; every page of
  it is used at some point of the replay, so `empty_cache` cannot return it. The neighbour list and the tables sit
  beside it in the default pool (16.1 kB/atom).
- **The largest single block was a wasted one:** the recomputing backward of `kernels.triton.tabled_outer_sum` (the
  first interaction's fused table and outer sum) allocated `[edges, 64]` float32 only to carry a shape; the kernel
  evaluates those values itself and never reads them. 4.12 GiB, 18.9 kB/atom, live at the force peak, and the request
  that fragmented the single call's pool most. Fixed: a stride-0 view carries the shape (`SHAPE_CARRIER`).
- **With the fix: 75.9 kB/atom reserved at 36³ (16.49 GiB, −2.32 GiB); the Ada's SrTiO3 limit rises from 233,280 to
  253,265 atoms** (37³, 1,563 ms/step). 38³ fails in the steady phase, in the neighbour list's rebuild sort, beside
  the 15.4 GiB graph pool.

## Method

[`tools/bench/memory_profile.py`](../../tools/bench/memory_profile.py) records the caching allocator's full history
(`torch.cuda.memory._record_memory_history`, Python stacks) from before the model loads, marks the phases in the trace
(a marker allocation of a reserved size at each phase start), runs the case, saves the snapshot and replays the trace:

- the peak of allocated bytes overall, per phase and per memory pool (default, a CUDA graph's private pool, or a pool
  released before the end), with the live blocks at each peak by owner (the innermost FastMACE frame, `OWNERS`) and
  by allocating line;
- every growth of the reserve, with the bytes that were free when it grew and the allocation that needed it (growth
  while the request fitted in the free bytes is fragmentation);
- the memory outside the allocator (`mem_get_info`: context, library workspaces).

Cases: `md` is the capacity scan's setup (`fm.Simulation`, Langevin NVT, graph tier, CUDA graphs, the planner's rescue
rungs; 5 warm-up and 5 steps); `single` is one forces + stress call through the ASE calculator (compiled tier, no CUDA
graphs, no MD), the shape of Symmetrix-XL's probe. Jobs: [jobs/leto.sh](jobs/leto.sh), [jobs/profile.sh](jobs/profile.sh);
`--diff` gives slopes between two sizes, `--reanalyse` re-reads a saved snapshot.

Runs: r1 and r2 (the same four cases, r2 with snapshots; identical peaks), r3 with the fix, r4 NaCl with the fix.
Profiles and summaries: [r2/](r2/), [r3/](r3/), [r4/](r4/). The snapshots (`snapshot-*.pickle.gz`, 0.1–0.3 GB
each) are not committed; they stay under `/mnt/homes/zwingel/fm-memprofile/runs/` on leto (and the PC's worktree).

## Where the memory is: MD, SrTiO3 36³ (233,280 atoms, 17.26 M edges, 74 per atom)

The peak of reserved memory is in the steady phase (a neighbour-list rebuild beside the captured force graph). kB/atom
= bytes / 233,280 / 1000.

| Owner | Before the fix: GiB | kB/atom | With the fix: GiB | kB/atom |
|---|---:|---:|---:|---:|
| **CUDA graph private pool (force pass), reserved** | 15.31 | **70.4** | 12.98 | **59.7** |
| · live at the pool's peak (the force backward) | 10.99 | 50.6 | 9.99 | 46.0 |
| ·· `tabled_outer_sum` shape carrier `[E, 64]` (wasted) | 4.12 | 18.9 | – | – |
| ·· gradient of the per-slot sums S `[3N, 16·64]` | 2.67 | 12.3 | (not at the peak) | |
| ·· 3×TF32 GEMM outputs in the backward (four, 5,120 floats/atom together) | | | 4.45 | 20.5 |
| ·· saved generic-interaction output `[N, 2048]` | | | 1.78 | 8.2 |
| ·· edge spherical harmonics `[E, 16]` (saved) | 1.03 | 4.7 | 1.03 | 4.7 |
| ·· w′ backward blocks `[E, 16]` (+ `[E, 7]`) | 1.03 | 4.7 | 1.48 | 6.8 |
| ·· tabled backward's `grad_y` `[E, 16]` | 1.03 | 4.7 | (not at the peak) | |
| ·· the rest (atom-level compiled buffers) | 1.11 | 5.1 | 1.25 | 5.8 |
| · packing slack in the pool's expandable segment | 4.32 | 19.9 | 2.99 | 13.8 |
| · the second captured variant (`B` at the padded edge size) | 0.06 | 0.3 | 0.06 | 0.3 |
| **Default pool, reserved** | 3.51 | **16.1** | 3.51 | **16.1** |
| · neighbour list, persistent (skin list: candidates, edges, shifts) | 1.40 | 6.4 | 1.40 | 6.4 |
| · neighbour list, rebuild temporaries (`build_gpu` sort and gather) | 0.99 | 4.5 | 0.99 | 4.5 |
| · pair search screen codes | 0.22 | 1.0 | 0.22 | 1.0 |
| · w′ power/value, spline, radial and composed-skip tables (fixed size) | 0.37 | 1.6 | 0.37 | 1.6 |
| · integrator and state (positions, momenta, noise, forces) | 0.10 | 0.5 | 0.10 | 0.5 |
| · model weights, private copies, species tables, other | 0.29 | 1.3 | 0.29 | 1.3 |
| · slack | 0.14 | 0.6 | 0.14 | 0.6 |
| **Outside the allocator** (context, cuBLAS / cuEq / CUTLASS workspaces) | 0.23 | 1.1 | 0.23 | 1.1 |
| **Device memory in use at the peak** | 19.04 | **87.6** | 16.72 | **77.0** |
| Symmetrix-XL's probe (one force call) | | ~65 | | ~65 |

- The 3×TF32 split copies (`tf32x3_gemms`' operand splits) are 0.01 GiB; the GEMM *outputs* above are the backward's
  gradients, not split copies.
- The build's peak (eager first force call and the warm-up, default pool, released by `empty_cache` before the
  capture) is 13.36 GiB allocated (12.36 with the fix), below the steady peak.
- Before the fix the overall allocated peak (13.36 GiB, 61.5 kB/atom) was the eager first force call in the build;
  the graph pool's live peak (the same force pass, captured) equals it to 0.01 GiB.

**Single force call** (ASE calculator, compiled tier; [r2/summary-single-SrTiO3-233280.txt](r2/summary-single-SrTiO3-233280.txt)):

| | Before the fix | With the fix |
|---|---:|---:|
| Peak allocated | 13.63 GiB, 62.7 kB/atom | 12.82 GiB, 59.0 kB/atom |
| Peak reserved | 18.04 GiB, 83.0 kB/atom | 15.70 GiB, 72.2 kB/atom |
| Reserve growth while the request fitted in the free bytes | 6.40 GiB | |

The live memory of one force call is Symmetrix-XL's (59–63 against ~65 kB/atom; we do not know whether its probe
counts allocated or reserved bytes). What MD adds is the graph pool's packing slack, and the neighbour list and its
rebuild temporaries held beside the captured force pass.

Slopes 32³ → 36³ (MD, before the fix; [r2/slope-md.txt](r2/slope-md.txt)): peak reserved 83.7 kB/atom (intercept
0.64 GiB), peak allocated 59.0 kB/atom (intercept 0.54 GiB); the force pass 50.6 kB/atom (no intercept), the neighbour
list 7.5 kB/atom; the tables, weights and workspaces are the fixed part (about 0.9 GiB with the context).

## The three largest items

1. **The shape-only `[E, 64]` buffer, 18.9 kB/atom: freed** (this change). `tabled_outer_sum_backward` with
   `TABLED_FORWARD` (the recomputing backward, on for the Ada and the 5060 Ti) passed `r.new_empty((E, columns))` to
   a kernel that evaluates `a` from the table and never loads it. A stride-0 `expand` view now carries the shape;
   nothing else changes (no rounding change, the same kernel arguments but `a`'s row stride). Measured: the 36³
   reserve −2.32 GiB (86.6 → 75.9 kB/atom), the single call −0.81 GiB allocated and −2.34 GiB reserved, the SrTiO3
   limit 233,280 → 253,265.
2. **The graph pool's packing slack, 19.9 → 13.8 kB/atom: not freeable by releasing memory.** The pool is one
   expandable segment; when a capture's request is larger than any hole, the segment grows at its end. Every page is
   written by some kernel during the replay, so neither `empty_cache` nor pre-sizing the pool returns it (pre-sizing
   one block gives the same best-fit packing). What would reduce it: fewer and smaller transient tensors in the
   backward (each removed one also shrinks the holes), or a capture-time allocation order that places long-lived
   tensors first. Expected at most 13.8 kB/atom (about +18 % atoms); uncertain, not tried.
3. **The backward's 3×TF32 GEMM outputs, 20.5 kB/atom (four `[N·slots, …]` gradients), and before the fix the
   gradient of the per-slot sums S, 12.3 kB/atom: streamable, not recomputable.** They are gradients the backward
   produces, not saved activations, so a recompute rung cannot remove them; chunking the backward over node blocks (or
   fusing the composed-skip backward with the outer-sum backward) would. That is the streaming path's job (left to
   the tiled path). The saved activations a recompute rung could drop are 17.8 kB/atom of the pool's 46.0 at its peak
   (the generic interaction's output 8.2, the edge spherical harmonics 4.7, smaller ones 4.9).

Next limiter after the fix: the neighbour list's rebuild (`skin.py:424`, `torch.sort(key).indices`: int64 keys and
indices for every candidate, 4.5 kB/atom of temporaries) beside the graph pool; 38³ failed there.

## NaCl (343,000 atoms, 35³, with the fix)

[r4/](r4/): MD peak allocated 14.63 GiB (45.8 kB/atom, the build's eager first force call), peak reserved 18.71 GiB
(58.6 kB/atom, steps); single call 14.90 GiB allocated (46.7 kB/atom), 17.73 GiB reserved (55.5). NaCl has fewer
edges per atom than SrTiO3, so the edge-scaled items are smaller and the atom-scaled ones dominate. At the force pass's
peak (graph pool, 12.44 GiB live): the interaction's `[N, 2048]` outputs (`tabled_outer_sum_forward` and the saved
`lean_forward` output, 2.62 GiB each), the backward's 3×TF32 GEMM gradients (2.45 + 1.47 + 1.14 + 0.33 GiB, the
5,120 floats per atom of the SrTiO3 table; the profiler files them under "3xTF32 split copies" because they are
allocated in `tf32x3_gemm.py`, but they are the GEMMs' outputs), edge spherical harmonics and a `[N, 512]` output
(0.65 GiB each). 21.4 GiB of the build's reserve growth came while the request fitted in the free bytes.

**The NaCl limit stays at 343,000 atoms**: 36³ (373,248) fails in the build's first force call on one 7.12 GiB request,
`empty_strided_cuda((N, 5120))` in the compiled backward ([p1/points/](p1/points/)): the plan at that size runs the
interaction's backward through `streaming_conv_tabled_recompute_backward`, which takes the message gradient as one
dense `[N, 5120]` tensor, so Inductor builds it beside the four GEMM gradient blocks (20.5 kB/atom twice). The
blocked w′ backward (`streaming_conv_wprime_backward_blocks`, SrTiO3's path at 36³) takes the four blocks as they are.
That buffer is the first target of the selective-recompute rung (conductor's item 2).

## Limits with the fix

`capacity_scan.py point`, 20 warm-up and 200 timed steps, each point alone on the GPU (idle check before each):

| GPU | Crystal | Before (main `4ffe4f6f`) | With the fix | First failing size |
|---|---|---|---|---|
| RTX 4000 Ada 20 GB (leto) | SrTiO3 | 233,280 (36³), 1,385 ms | **253,265** (37³), 1,563 ms, peak 17.96 GiB | 274,360: steady, the neighbour list's rebuild sort (396 MiB) beside the 15.4 GiB graph pool; 296,595 also fails |
| RTX 4000 Ada 20 GB (leto) | NaCl | 343,000 (35³), 1,533 ms | 343,000 (unchanged) | 373,248: build, the dense `[N, 5120]` message gradient (7.12 GiB) |
| RTX 5060 Ti 16 GB (local PC, WSL2) | SrTiO3 | 163,840 (32³) | 163,840 (unchanged), 1,035 ms, peak 14.46 GiB | 179,685: Triton module load out of memory in the capture warm-up |
| RTX 5060 Ti 16 GB (local PC, WSL2) | NaCl | 195,112 (29³) | 195,112 (unchanged), 801 ms, peak 14.11 GiB | 216,000: Triton module load out of memory in the capture warm-up |

- On the 5060 Ti the first failing sizes now fail on the driver's own allocation (Triton's `cuModuleLoad` during the
  capture's warm-up), which cannot use the allocator's cached blocks. Releasing the cache before the warm-up
  (`EngineConfig.empty_cache_before_warmup`, tried and reverted) only moved the failure to the first eager force call
  (Triton load of `tabled_outer_sum_backward` at 13.96 GiB allocated; NaCl a WSL2 mapping failure): no limit gain,
  REJECTED ([5060ti/warmup-empty-cache/](5060ti/warmup-empty-cache/)).
- The 5060 Ti's step times are not comparable with the 8 November table (another day, WSL2).

## Gates

- **Speed** (work removal, one not-slower A/B): leto, `ab_rewrites.py`, Benchmark_loup SrTiO3 10×10×10, warm-up 200,
  6 blocks of 256 steps, b = `SHAPE_CARRIER` True: **b/a 1.0009 / 1.0002** (ab / ba; 21.28 ms per MD step), a tie
  ([ab/](ab/)).
- **Bitwise gradients**: `tests/gpu/test_round2_kernels_cuda.py::test_recomputing_backward_allocates_no_value_rows`
  runs the recomputing backward (plain and `POWER_BACKWARD`) with both settings in one process: the gradients are
  bitwise equal, and the new setting's peak stays below `E·K·4` bytes while the old one's reaches it. The whole file
  passes on the new source (10 passed); on the old source the new test fails as it should ([t2/](t2/)).
- **CPU suite** (leto, merge tier, the branch tree without `results/`): 5,501 passed, 1 failed, then fixed and
  re-run: `test_layering.py` wanted `SHAPE_CARRIER` on the module-switch allow list (added; the file now passes,
  32 passed). Ruff: the changed files pass; the 66 findings on leto are in 51 files this change does not touch
  ([suite-summary.txt](suite-summary.txt)).
