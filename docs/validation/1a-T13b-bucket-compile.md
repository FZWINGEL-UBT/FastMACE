# 1a-T13b: the compile at T19's first new edge bucket

Follow-up to 1a-T13, 18 October 2026. Branch `fastmace/1a-T13b` from main
`43ce026`. The research engine is ResearchModule tag
`research-engine-baseline-20261017` (`d2faa9fd`, export `1a-T18`). The
environment is `mace_cueq_oeq` (torch 2.14.1+cu130, MACE 0.3.16,
cuEquivariance 0.12.0, OEq 0.7.0). Reproducers and their outputs are in
`results/1a-T13b-bucket-compile-20261018/`.

Runs:
- `fmc`, `blc`, `abc_before`, `abc_after`: the PC's RTX 5060 Ti, each with
  fresh Inductor, Triton and FastMACE caches (`repro_local.sh`).
- `fm1`, `bl1`, `fm2`, `bl2`, `ab_fb`: leto, sharing a warm cache, before the
  GPU was given back to T19.

## What T19 saw

On murphy, in a same-process A/B with MH-1 `matpes_r2scan` (hybrid FP32,
NVT, NaCl 3×3×3), the edge count grew from 7168 into the next bucket (7680).
FastMACE's block then took 38–80 ms/step, against 3.5 ms in the other
blocks, and dynamo counted one new graph (5→6). The research engine's block
only recaptured.

## Cause: FastMACE did not put back its `mask_head` patch after the research engine replaced it

Compiled MACE models guard on the function in `mace.modules.blocks.mask_head`
(`___check_obj_id(G[...blocks].mask_head.__code__)`). Both engines replace
MACE's `mask_head` with a capture-safe copy of their own, but differently:

- **The research engine** installs its copy before *every* capture
  (`GPULangevin._capture` calls `use_capture_safe_mask_head()`).
- **FastMACE** installed its copy once. `_patches.set_capture_safe_mask_head`
  returned early when its record existed, even when another function had
  replaced it since.

In one process, every capture of the research engine leaves its own
function in place. FastMACE's compiled entries then fail their guard, but a
CUDA-graph replay never enters dynamo, so nothing happens while FastMACE
only replays. The failure surfaces at FastMACE's next eager call of the
model: the warm-up of the first capture at a new padded size, which is
T19's first new bucket. Build order does not matter, because the research
engine reinstalls its function at each of its own captures.

### Evidence

**Each engine alone (fresh caches, PC; 900 K, 5 % buckets, 300 steps).**
Both engines pass the same edge counts to the force call. The initial
forces run at the exact 6912 (216 atoms × 32 neighbours on the ideal
lattice). Then come three warm-up calls and a capture at 7168, again for the
first pipelined step, then three warm-up calls and a capture at 7680 (step
66).

| Run | Graphs after warm-up | Graphs at the end | Step at 7680 |
| --- | --- | --- | --- |
| FastMACE (planner), `fmc` | 4 | 4 | 47 ms (capture) |
| research engine, `blc` | 4 | 4 | 53 ms (capture) |

Warm-up sees 6912 and 7168, so dynamo makes the edge dimension dynamic in
both engines. The exact initial count does not equal the padded size. A cold
cache does not make FastMACE compile at the next bucket.

**Same-process A/B (fresh caches, PC; FastMACE built first, as the hybrid
requires).** The research engine's set-up then refuses to install its addmm
pass over FastMACE's in `post_grad_custom_pre_pass`, so the reproducer
clears that slot between the builds, as T19's harness does. Blocks of 150
steps alternate.

| Run | FastMACE block at 7680 | Research engine block at 7680 | Graphs |
| --- | --- | --- | --- |
| before the fix, `abc_before` | 142.8 ms/step | 4.13 ms/step | 5 → 6 (FastMACE) |
| after the fix, `abc_after` | 4.03 ms/step | 3.98 ms/step | 5 → 5 |

The recompile log (`abc_before.recompiles.txt`) names the failed guard: the
`mask_head.__code__` check of FastMACE's entries.

**Other candidates, ruled out:**
- **Dynamo settings:** the same on both sides. `cache_size_limit` is raised
  to 64, as the research engine does; nothing uses `mark_static`.
- **Warm-up shapes:** the same on both sides (above).
- **The compiled-model cache:** each side loads its own model.
- **Cold or warm Inductor cache:** single-engine runs with fresh caches
  behave like the warm ones.
- **The cleared `post_grad_custom_pre_pass`:** it is not a dynamo guard.

With a 256-edge bucket (leto, `fm1`/`bl1`), 6912 is itself a padded size,
so warm-up sees only one shape. Both engines then compile once at the first
new size, alike.

## Fix

`fastmace._patches.set_capture_safe_mask_head(True)` now puts FastMACE's
function back when another one is in place, keeping its record and MACE's
preserved original. The graph engine calls it before every capture, so
FastMACE now does what the research engine does.

Tests:
- `tests/cpu/test_engine.py::test_capture_safe_mask_head_is_put_back_after_another_install`.
- `tests/gpu/test_engine_gpu.py::test_no_compile_at_a_later_edge_bucket_{plain_cueq,hybrid}`:
  MH-1 NaCl 3×3×3 at 900 K through the planner in a fresh process with fresh
  compiler caches. After warm-up it installs a foreign `mask_head`, as the
  research engine does, and crosses into the next bucket. It asserts that no
  graph is compiled after warm-up.
  **This test has not run yet:** the local GPU run was stopped when the owner
  needed the GPU. The earlier leto version of the test, without the foreign
  `mask_head`, passed for both backends.

## For T19

With the fix, same-process A/B blocks that capture a new padded size no
longer compile on FastMACE's side. The research engine still recompiles once
during its own build, when it replaces FastMACE's `mask_head`. That falls in
its warm-up, as before.
