# 1a-T13b: the compile at T19's first new edge bucket

Follow-up to 1a-T13, 18 October 2026. Branch `fastmace/1a-T13b` from main
`43ce026`. The research engine is ResearchModule tag
`research-engine-baseline-20261017` (`d2faa9fd`, NAS export
`/mnt/homes/zwingel/fastmace-baseline/1a-T18`). The machine is leto: RTX 4000
Ada, driver 595.58.03, NAS `mace_cueq_oeq` (torch 2.14.1+cu130, MACE 0.3.16,
cuEquivariance 0.12.0, OEq 0.7.0). Reproducers and their outputs are in
`results/1a-T13b-bucket-compile-20261018/`.

## What T19 saw

On murphy, in a same-process A/B with MH-1 `matpes_r2scan` (hybrid FP32,
NVT, NaCl 3×3×3), the edge count grew from 7168 into the next bucket (7680)
in timed block 4. FastMACE's block then took 38 ms/step, against 3.5 ms in
the other blocks. Dynamo counted one new graph (unique_graphs 5→6). The
research engine's block took 3.67 ms/step: a recapture, no compile.

## Cause: the research engine's process-wide `mask_head` patch, not the engine

**FastMACE alone does not compile at a later bucket.** Each engine below ran
alone in its own process, at 900 K so that the edge count grows quickly. The
initial exact count is 6912 (216 atoms × 32 neighbours on the ideal
lattice).

| Run | Bucket | Warm-up graphs | New sizes after warm-up (step) | Graphs at the end | Step at the new size |
| --- | --- | --- | --- | --- | --- |
| FastMACE (planner), `fm2` | 512 (5 %) | 4 | 7680 (66) | 4 | 57 ms (capture) |
| research engine, `bl2` | 512 (5 %) | 4 | 7680 (66) | 4 | 53 ms (capture) |
| FastMACE, `fm1` | 256 | 4 | 7168 (29), 7424, 7680 | 5 | 16.4 s at 7168 |
| research engine, `bl1` | 256 | 4 | 7168 (29), 7424, 7680 | 5 | 22.7 s at 7168 |

With the production 5 % bucket, warm-up sees two edge counts: the exact
6912 for the initial forces and the padded 7168 for the first capture. Dynamo
then makes the edge dimension dynamic. On both sides the next bucket needs a
capture but no compile. With a 256-edge bucket, 6912 is itself a padded size,
so warm-up sees only one shape. Both engines then compile once at the first
new size, the same way.

**In one process, the second engine's build breaks the first's guards.**
`repro_ab.py` builds FastMACE first and the research engine second. The
hybrid must load before MACE is imported, so this is the only possible
order. Like T19's harness, it clears FastMACE's `post_grad_custom_pre_pass`
before the research engine installs its own. Then the two engines alternate
blocks of 150 steps. In the first block that reaches 7680, FastMACE takes
159 ms/step and dynamo goes from 5 to 6 graphs. The research engine stays at
3.7 ms/step with no new graph. Dynamo's recompile log
(`ab_fb.recompiles.txt`) shows two guard failures:

- `___check_obj_id(G['__import_mace_dot_modules_dot_blocks'].mask_head.__code__, …)`.
  The research engine's set-up installs its own capture-safe `mask_head` into
  `mace.modules.blocks`, replacing the one FastMACE's `_patches` installed.
  FastMACE's compiled entries guard on that function.
- The size guard of the static entry, `data['shifts']` 6912 vs 7680, which is
  expected.

A CUDA-graph replay never enters dynamo, so the failed guard has no effect
while FastMACE replays its graphs. It surfaces at FastMACE's next eager call
of the model, the warm-up of the first capture at a new padded size. That is
T19's first new bucket. The research engine, built second, compiled under
the final `mask_head`, so its guards still hold.

The other candidates were checked:

- **Dynamo settings and the cache limit:** the same on both sides. FastMACE
  only raises `cache_size_limit` to 64, as the research engine does.
- **Shapes in warm-up and captures:** the same on both sides (6912, then
  7168).
- **`mark_static`:** not used anywhere.
- **The compiled-model cache:** each A/B side loads its own model, so the
  cache shares nothing between them.
- **The cleared `post_grad_custom_pre_pass`:** it is not a dynamo guard, so it
  does not cause the recompile. It does mean that FastMACE's recompiled graph
  at 7680 is built with the research engine's addmm pass, not FastMACE's.

## Consequences

- **FastMACE is unchanged.** Its behaviour equals the research engine's in
  production, one engine per process.
- **Regression test:** `test_no_compile_at_a_later_edge_bucket_{plain_cueq,hybrid}`
  in `tests/gpu/test_engine_gpu.py`. It runs MH-1 NaCl 3×3×3 at 900 K through
  the planner, each case in a fresh process. The edge count crosses from 7168
  into 7680 at step 66, and the dynamo counters stay at their warm-up values
  (2 graphs for plain cuEq float64, 4 for the hybrid).
- **For T19's same-process A/B,** any block in which a side captures a new
  padded size after the other side's build is affected. Two ways out:
  - make both sides share one `mask_head` object (install FastMACE's
    capture-safe function where the research engine's set-up installs its
    own); or
  - treat steps with a first capture as warm-up, and compare only blocks
    without one.
  Setup costs and memory already come from single-engine processes
  (decisions after wave 1b).
