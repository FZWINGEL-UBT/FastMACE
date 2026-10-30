# CAP-T08: recomputing the node-local segments (`recompute_node_segments`)

Plan task CAP-T08 ([plans/capacity.md](../../plans/capacity.md)), idea 2 of the owner's capacity list of
30 October 2026, through the conductor: cut the per-atom state the force call keeps, which is most of its memory at
the limit ([capacity-T08-inventory.md](capacity-T08-inventory.md)).

## Verdict

**The rewrite.** `RewriteConfig.recompute_node_segments` runs each interaction's `gated_rows` chain and product
block under non-reentrant `torch.utils.checkpoint`:

- the forward keeps only the segments' inputs;
- the backward rebuilds their intermediates when it reaches them;
- the compiled model stays one graph.

**Class and default.** Rounding, deterministic, off by default. Rule 12 turns it on only when the plan does not fit
with `recompute_conv_weights` on (`planner.memory.MEMORY_RESCUE`, in that order).

**New limits** on the 16 GB RTX 5060 Ti, MH-1 float32:

| Structure | Before (main) | After | Change |
| --- | ---: | ---: | ---: |
| NbCl5 | 51,840 atoms (6×6×20) | **69,984 atoms (6×6×27)** | +35 % |
| NaCl | 49,248 atoms (18×18×19) | **67,200 atoms (20×20×21)** | +36 % |

At the same NbCl5 cell of 51,840 atoms the peak is 3.3 GB lower: 9.60 against 12.93 GB allocated, 11.88 against
15.34 GB reserved.

**Cost.** In the same-process A/B with the record forced on, b/a is 1.095–1.096 in both build orders, at 1,944 and
at 9,216 atoms. Planned on only when memory requires it, that cost is paid only by cells that would otherwise not
run.

**The gate passed** on NaCl, NbCl5 and NaNbO3 with the record on.

**One region per interaction lost.** Checkpointing `gated_rows` → reshape → product block as one region kept
35 kB per atom fewer, but:

- its recomputation set the same compiled peak;
- it cost 1.24 times the step time of separate regions;
- NbCl5 67,392 atoms no longer fitted.

So the record keeps each segment its own region (`SEGMENT_REGION = "separate"`). The combined region stays in the
code as an A/B constant for the atom-tiling idea below.

## How it was found

### Where the peak is (falsifier)

**Tool.** `tools/bench/segment_peaks.py` (new), on NbCl5 5×5×12 (21,600 atoms, 572,970 edges), MH-1 hybrid float32,
`recompute_conv_weights=on` (as at the limit):

- in the eager tier, markers at each marked module's forward and backward give the peak of every interval;
- a `saved_tensors_hooks` pack attributes each saved storage to its module.

**What the eager force call saves per atom:**

| Module | Saved per atom |
| --- | ---: |
| `gated_rows`, each interaction | 51.2 kB |
| product block, each interaction | 32.8 kB |
| interaction 1's other parts | 22.9 kB |
| interaction 0's other parts | 8.2 kB |

So the node-local segments hold 168 of the 199 kB per atom.

**The eager peaks**, above the call's start:

| Interval | Peak |
| --- | ---: |
| `interaction1.gated_rows` forward | 5.70 GiB |
| `product1` backward | 5.53 GiB |
| `interaction1.gated_rows` backward | 5.48 GiB |
| interaction 1's convolution backward | 5.24 GiB |

Interaction 0's saved segments (1.69 GiB) are still alive during interaction 1's convolution backward. So the saving
needs both interactions' segments; interaction 1 alone reached 0.47 GiB.

### Four variants of the segment forward

The variants run in the compiled default plan unless named eager. Both interactions' `gated_rows` and product blocks
change; the peak is measured above the force call's start.

| Variant | Peak | Force call |
| --- | ---: | ---: |
| none | 4.79 GiB | 210 ms |
| eager, both interactions, atom-chunked autograd function (8 chunks) | 4.56 GiB (eager none: 5.70) | 442 ms (eager none: 348) |
| autograd function with `autograd.grad` in its backward (8 chunks): the graph breaks | 4.16 GiB | 393 ms (1.87×) |
| opaque custom op with a registered backward, compiled chunks (8 chunks) | 3.56 GiB | 319 ms (1.52×) |
| **per-chunk `checkpoint` inside the compiled graph**, 8 chunks | 3.49 GiB | 247 ms (1.17×) |

**The custom op was dropped.** Its backward needs autograd's dispatch keys forced back inside the op body.
`checkpoint` saves more and costs less.

### Chunk count

`checkpoint` per chunk, the same cell:

| Chunks | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Peak above the call's start | 3.50 GiB | 3.25 GiB | 3.51 GiB | 3.49 GiB | 3.48 GiB |
| Force call | 230 ms | 248 ms | 248 ms | 247 ms | 233 ms |

- **One chunk saves as much as 4, 8 or 16.** Recomputation does the work, not chunking. The rebuilt segment is never
  the peak here, which is interaction 1's convolution backward.
- **2 chunks gave 0.24 GiB more, once.** That point was not repeated.
- **The rewrite therefore recomputes each segment whole,** with no chunk count and no plan parameter (the conductor's
  decision). Chunking would come back only if a larger cell showed a single segment's recomputation setting a new
  peak. The limit points below did not show that.
- **At 9,216 atoms** the saving was 59 kB per atom (with 4 chunks), against 64 kB per atom at 21,600.

## The rewrite

- **Record:** `fastmace.rewrites.segments`, record `recompute_node_segments`.
  - `requires=("gated_rows",)`, rounding, deterministic, default off.
  - Applied after every record that replaces the segments' modules or forwards.
- **Mechanism:** `modules.recompute(module)` keeps the forward before as `recomputed_inner`, a bound method, so a deep
  copy rebinds it to the copy. It installs `recomputed_forward`, which calls `checkpoint(recomputed_inner, ...,
  use_reentrant=False, preserve_rng_state=False)`.
- **Self-check:** each recomputed segment against its stored forward, values and input gradients. The product block's
  probe is its interaction's reshaped `gated_rows` probe.
- **Memory term:** `rewrites.memory.recompute_node_segments` is 0.20 of the stock per-node floats less per node. It is
  fitted on the compiled peaks: 64.2 kB per atom at 21,600 atoms and 63.3 kB at 9,216, against MH-1's 317,440 B per
  node in float32.
- **Planning order:** `MEMORY_RESCUE = ("recompute_conv_weights", "recompute_node_segments")`.
  - Rule 12 tries them in that order. Each is tried only while the estimate still does not fit, for a cell that
    cannot fit (`short_by_far`).
  - `fallback.memory_rescue` turns on one record per out-of-memory rebuild, in that order.
  - `Simulation` rebuilds while the rescue has a record left. Before this it rebuilt once.
- **Config:** `RewriteConfig.recompute_node_segments` (Toggle), with the fixed planner's rule
  `on_by_rule_12_when_the_plan_does_not_fit_with_recompute_conv_weights`.
- **Golden plan texts:** they change only in the count of rewrites listed off, one more each. The default plans are
  otherwise unchanged.

## Step time

`tools/ab_rewrites.py --prepare`, same process, MH-1 hybrid float32 NVT at 300 K.

- **Side a:** the default plan, with the record off.
- **Side b:** `recompute_node_segments=on`.

Median step in ms; ratios b/a, with the range over block pairs in brackets.

| Cell | Blocks | order ab (abab): a / b, b/a | order ba (abba): a / b, b/a |
| --- | --- | --- | --- |
| NbCl5 3×3×3, 1,944 atoms | 12 × 300 steps | 17.76 / 19.47, **1.096** (1.079–1.134) | 18.16 / 19.88, **1.095** (1.077–1.108) |
| NbCl5 4×4×8, 9,216 atoms | 8 × 100 steps | 89.92 / 98.43, **1.095** (1.085–1.113) | 91.21 / 99.85, **1.095** (1.069–1.107) |

Every run was valid: nothing compiled while timed, and no other process used the GPU.

## The limit points

`tools/bench/capacity_scan.py point`, default plan, 200 steps:

- **This branch:** rule 12 turned on `recompute_conv_weights` and then `recompute_node_segments` at every point
  (`short_by_far`).
- **main** (`88c9ba95`): `recompute_conv_weights` alone.

| Cell | Atoms | Side | Status | ms/step | Peak allocated | Peak reserved |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| NbCl5 6×6×20 | 51,840 | main | ok | 590.7 | 12.93 GB | 15.34 GB |
| NbCl5 6×6×20 | 51,840 | this branch | ok | 597.9 | 9.60 GB | 11.88 GB |
| NbCl5 6×6×21 | 54,432 | this branch | ok | 632.9 | 10.06 GB | 12.34 GB |
| NbCl5 6×6×22 | 57,024 | this branch | ok | 664.3 | 10.53 GB | 12.93 GB |
| NbCl5 6×6×23 | 59,616 | this branch | ok | 694.9 | 10.99 GB | 13.47 GB |
| NbCl5 6×6×24 | 62,208 | this branch | ok | 726.3 | 11.45 GB | 14.06 GB |
| NbCl5 6×6×26 | 67,392 | this branch | ok | 807.2 | 12.38 GB | 15.19 GB |
| NbCl5 6×6×27 | 69,984 | this branch | ok | 827.0 | 12.84 GB | 15.74 GB |
| NbCl5 6×6×28 | 72,576 | this branch | out of memory at capture | | 13.23 GB | 16.22 GB |
| NbCl5 6×6×30 | 77,760 | this branch | out of memory at capture | | 14.15 GB | 15.18 GB |
| NaCl 18×19×19 | 51,984 | this branch | ok | 603.2 | 9.79 GB | 12.09 GB |
| NaCl 19×19×19 | 54,872 | this branch | ok | 651.1 | 10.32 GB | 12.74 GB |
| NaCl 19×19×20 | 57,760 | this branch | ok | 680.8 | 10.84 GB | 13.43 GB |
| NaCl 20×20×20 | 64,000 | this branch | ok | 749.4 | 11.98 GB | 14.84 GB |
| NaCl 20×20×21 | 67,200 | this branch | ok | 787.1 | 12.56 GB | 15.53 GB |
| NaCl 20×20×22 | 70,400 | this branch | out of memory at capture | | 13.10 GB | 16.22 GB |
| NaCl 21×21×21 | 74,088 | this branch | out of memory at capture | | 13.77 GB | 14.95 GB |
| NaCl 21×21×23 | 81,144 | this branch | out of memory | | 14.98 GB | 16.14 GB |

- **The step time grows with the cell, about 12–13 µs per added atom.** At 51,840 atoms the two sides are within 1.2 %.
  These come from separate processes at the memory limit, not from an A/B, so the A/B above is the record's cost.
- **Failures above the limit are out of memory at capture.** They are recoverable allocator errors, not sticky ones,
  so the run reports the error.

## One region per interaction (rejected)

The conductor's follow-up: one checkpoint region per interaction that starts after the convolution and spans
`gated_rows` → reshape → product block. It saves only the region's inputs: the chain's two inputs, the skip term
and the species. The convolution stays outside. `modules.join_region` builds it, selected by
`segments.SEGMENT_REGION = "combined"`. Both versions ran at commit `bc1df9ab` on the same machine, one after the
other.

**Saved tensors**, compiled default plan, `recompute_conv_weights` on, NbCl5 5×5×12 (21,600 atoms),
`tools/bench/saved_tensors.py`:

| Region | Saved per atom | Peak above the call's start |
| --- | ---: | ---: |
| separate | 97.5 kB (1.96 GiB) | 3.49 GiB |
| combined | 62.3 kB (1.25 GiB) | 3.51 GiB |

Combined keeps 0.71 GiB less, but the compiled peak does not move: the recomputation of the region in the backward
takes it back.

**Where the peak is**, eager, `tools/bench/segment_peaks.py`, same cell, above the call's start:

| Record | Peak | Where |
| --- | ---: | --- |
| off | 5.70 GiB | `interaction1.gated_rows` forward |
| separate | 5.88 GiB | `interaction1.gated_rows` backward (its recomputation) |
| combined | 4.64 GiB | interaction 1's region backward (its recomputation) |

- **The peak does not move into the convolution backward.** With combined, interaction 1's convolution backward
  peaks at 3.89 GiB, 0.75 GiB below the region's recomputation.
- In eager the combined region lowers the peak by 1.06 GiB against off; compiled it lowers it no more than separate.
  So in the compiled graph the recomputed region's working set sets the peak, not the saved state.

**Step time**, same-process A/B, both sides with the record on, a = separate, b = combined:

| Cell | order ab: a / b, b/a | order ba: a / b, b/a |
| --- | --- | --- |
| NbCl5 3×3×3, 1,944 atoms | 19.92 / 24.69 ms, **1.240** | 19.79 / 24.61 ms, **1.243** |
| NbCl5 4×4×8, 9,216 atoms | 97.02 / 121.24 ms, **1.250** | 97.78 / 122.10 ms, **1.249** |

Against the default plan with the record off, combined costs 1.354–1.356 at both cells and in both orders; separate
costs 1.095.

**At the limit**, default plan, rule 12 deciding:

| Cell | Atoms | Separate | Combined |
| --- | ---: | --- | --- |
| NbCl5 6×6×26 | 67,392 | ok, 785.4 ms/step, 14.15 GiB reserved | out of memory at capture (CUDA context lost) |
| NbCl5 6×6×28 | 72,576 | out of memory | out of memory |
| NaCl 20×20×21 | 67,200 | ok, 787.1 ms/step | failed in build (the gate's reference ran out) |
| NaCl 20×20×22 | 70,400 | out of memory | out of memory |

**Gate and replay with combined:** passed (energy ratios 0.17, 0.08 and 0.50 for NaCl, NbCl5 and NaNbO3).
Capture was clean. Runs are not bitwise equal across runs, as with separate.

**Decision:** separate. Combined fits fewer atoms and costs 1.24 times more. Rule 12's term stays at 0.20 of the
stock per-node floats, the separate version's measured saving.

**The pending idea, not started:** atom-tiled forward and backward of the combined block. The numbers that bear on
it:

- the combined region's recomputation is the compiled peak: 3.51 GiB above the call's start at 21,600 atoms;
- its saved state is already 35 kB per atom below separate's;
- tiling the recomputation over atoms would leave that saving in place and cut the region's working set by the
  tile count.

The per-chunk `checkpoint` earlier did not show this: with separate segments the peak was elsewhere. The conductor
takes it to the owner.

## Exactness and capture

- **Gate:** `tools/force_stress_gate.py --rewrites recompute_node_segments=on`, against MACE's float64 model. The
  values are production error divided by ASE float32's error, and all are within the gate's limits:

  | Cell | Energy | Forces | Stress |
  | --- | ---: | ---: | ---: |
  | NaCl | 0.16 | 0.95 | 0.38 |
  | NbCl5 | 0.08 | 1.05 | 0.39 |
  | NaNbO3 | 0.49 | 1.01 | 0.81 |

- **Capture and replay:** `results/cap-t08-chunks-20261030/replay.py`, NbCl5 3×3×3, 200 steps.
  - **Method:** two runs each with the record off and on, in one process. The captured force graphs replay every
    step.
  - **Capture:** with the record on, the graphs captured and replayed without error.
  - **Bitwise across runs:** neither setting gives bitwise equal runs. The default float32 plan has
    non-deterministic records: atomic sender additions in the streaming convolution's and `wprime_backward`'s
    backward. The largest position difference after 200 steps was 5.4 × 10⁻⁷ Å with the record off and
    7.4 × 10⁻⁷ Å with it on, the same class.
  - **Determinism of the record itself:** it adds no atomics. It is declared deterministic and recomputes in a fixed
    order.
  - **No reproducible check:** the reproducible plan (plain cuEq) has no `gated_rows`, so the record does not apply
    there.
- **CPU, eager:** the recomputed segments give bitwise the stored forward's values and gradients, and the tiny MH-1
  model's energy and forces are bitwise unchanged (`tests/cpu/test_recompute_node_segments.py`). The rounding class
  comes from Inductor fusing the recomputed segments differently.

## Tests

- **CPU, on murphy:** leto had another user's heavy job and coraline another user's active job. On murphy the busy
  check was 0, the GPU hidden, `-n 8`, and the golden plan texts regenerated first. At commit `4db6f986`:
  **1,026 passed, 54 skipped.**
  - The files: `test_layering`, `test_recompute_node_segments`, `test_memory_estimate`, `test_memory_rescue_retry`,
    `test_memory_fit`, `test_plan_text`, `test_config`, `test_docs_options`, `test_planner`, `test_minimal_planner`,
    the four `test_fallback_*`, `test_rules_meta`, `test_rewrites_registry`, `test_rewrites_fusion`,
    `test_rewrites_mh1`, `test_rewrites_linear`, `test_rewrites_shared_fields`, `test_versioned_registry`,
    `test_layout_records`, `test_research_surface`, `test_report`, `test_provenance`, `test_reproducible`,
    `test_isolation`, `test_forces`, `test_gate_epilogue` and `test_wprime_backward`.
  - The first run, at `6cd84d99`, had four failures. The switch map, the auto-rule tables and the report's stub
    plan did not yet list the new field.
- **New tests:**
  - `test_recompute_node_segments.py`:
    - the mechanism against the stored forward, with fewer tensors saved;
    - deep-copy rebinding;
    - the record's fields, its order in `MEMORY_RESCUE` and its memory term;
    - the record on the tiny model: every segment recomputed, its self-check, the self-check failing for rows out of
      order;
    - energy and forces bitwise.
  - `test_memory_estimate.py::test_rule_12_turns_on_recompute_node_segments_only_after_recompute_conv_weights`.
  - `test_memory_rescue_retry.py`: one record per rebuild in order, and `Simulation` rebuilding again while a record
    is left.

## Setup

- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130).
- **GPU access:** `tools/ci/local_gpu.py run` reservations, with the idle check before every job. The GPU was shared
  with the neighbour-list session's reservations.
- **Sources:**
  - FastMACE `ce8c18a9` for the GPU runs. Its library code is the final branch's; the later commits change tests,
    docs and a reason's wording.
  - main `88c9ba95` for the before side.
  - The falsifier and the variants ran at `531b6586`, `fa718d19` and `ce8c18a9` (the tool's variants).
  - The region comparison ran at `bc1df9ab`, whose only later change is the default `SEGMENT_REGION`.
- **Model:** MH-1 (`mh1-r2scan`), hybrid backend, float32.
- **Records:** `results/cap-t08-chunks-20261030/`:
  - `segments/`: the tool's records;
  - `ab/`, `points/`, `gate/` and `replay.json` (separate regions, `ce8c18a9`);
  - `region/`: the region comparison (`ab/`, `points/`, `saved/`, `timeline/`, `gate/`, `replay.json`);
  - the run scripts and logs.
