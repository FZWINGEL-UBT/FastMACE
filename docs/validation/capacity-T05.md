# CAP-T05: capture snapshots in pinned host memory

Plan task CAP-T05 ([plans/capacity.md](../../plans/capacity.md)).

**What it does.** Around every capture and every warm-up, the graph engine snapshots each captured buffer and later
restores the MD state from that snapshot.

- On the device, those copies sat beside the warm-up's activations at the force stage's memory peak. The largest are
  the neighbour list and the noise batch: about 0.5 GiB at 27,000 NaCl atoms.
- Since commit `2a7c5279` (merged into main with CAP-T07), `Registry.snapshot(offload_bytes=...)` copies buffers of
  1 MiB or more to pinned host memory (`fastmace.engine.graph.CAPTURE_SNAPSHOT_OFFLOAD_BYTES`), and `restore` copies
  them back bit for bit.
- Other snapshots (MC moves, events, the gate) are unchanged.
- The implementation measured the build peak 0.51 GiB lower on NbCl5.

This report adds the same-process A/B the plan asked for.

## Verdict

- **Step time: no change.** Against no offload, the median step is 0.998–1.007×, in both build orders, at 1,944 and
  9,216 NbCl5 atoms. The snapshot runs only around captures and warm-ups, never in a replayed step.
- **Setup: about 5 s longer per build.** Building plus capturing the first step's stages took about 5 s more with the
  offload: 18.2 against 12.9 s at 1,944 atoms, 19.8 against 15.1 s at 9,216 atoms.
  - This comes from one comparable sample per cell (see below).
  - The likely cause is the pinned host allocations (`torch.empty(..., pin_memory=True)`) and the synchronous copies.
  - Captures at a new padded edge size later in a run pay it again, but they are rare.
- **Recommendation: keep it as it is.** The memory it frees sits at the force stage's peak, and its cost is a few
  seconds per build or capture, not per step.
  - If setup time ever matters, there is an obvious follow-up: offload only when the force graphs are planned near
    the memory limit (`force_graphs` `"fit"` releasing graphs), or keep the pinned buffers across captures.
  - Neither is done here.

## A/B

`tools/ab_rewrites.py --prepare`, same process, MH-1 hybrid float32 NVT at 300 K, default plan on both sides.

- **Side a: no offload.** It sets `fastmace.engine.graph:CAPTURE_SNAPSHOT_OFFLOAD_BYTES = 2**62` while it is built
  and while the first step's stages are captured.
- **Side b: the default,** 1 MiB.

The new `--prepare` flag captures the first step's stages inside each side's build, with that side's constants set,
as `Simulation` does (`engine.prepare`). Without it the captures would happen lazily in the warm-up, after the
constants were restored, and both sides would offload.

Captures at new padded edge sizes in the warm-up use the default on both sides. Median step in ms. Order `ab` uses the
abab block pattern and order `ba` uses abba. Ratios are b/a, with the range over block pairs in brackets.

| Cell | Blocks | order ab: a / b, b/a | order ba: a / b, b/a |
| --- | --- | --- | --- |
| NbCl5 3×3×3, 1,944 atoms | 12 × 300 steps | 17.68 / 17.75, **1.004** (0.988–1.021) | 18.02 / 17.98, **0.998** (0.995–1.006) |
| NbCl5 4×4×8, 9,216 atoms | 8 × 100 steps | 87.81 / 88.41, **1.007** (0.986–1.022) | 87.83 / 88.21, **1.004** (0.986–1.013) |

Every run was valid: nothing compiled while timed, and no other process used the GPU.

**Setup time (build plus `prepare`, seconds)** is in the records' `setup_s`. The side built first also compiles, so
only the side built second is comparable across the two orders:

| Cell | b (offload) built second, order ab | a (no offload) built second, order ba |
| --- | ---: | ---: |
| NbCl5 3×3×3 | 18.2 | 12.9 |
| NbCl5 4×4×8 | 19.8 | 15.1 |

## What changed in this task

- `tools/ab_rewrites.py`: the `--prepare` flag; the record's `prepare` field.
- The run script `results/cap-t05-20261030/local.sh`.
- The A/B records in `results/cap-t05-20261030/ab/`.
- This report and the plan row.

## Checks

- `tests/cpu/test_ab_rewrites_tool.py` on leto (GPU hidden, commit `6275833a`): 2 passed.
- Ruff is clean on `tools/ab_rewrites.py`.
- The offload's own tests (`tests/cpu/test_snapshot_offload.py`, `tests/gpu/test_snapshot_offload_cuda.py`) were
  merged with the implementation and are unchanged.

## Setup

- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130).
- **GPU access:** `tools/ci/local_gpu.py run` reservation, with the idle check before every job.
- **Source:** a pinned copy of FastMACE `6275833a` (main `26039927` plus the `--prepare` flag). No baseline, since both
  sides are FastMACE.
- **Model:** MH-1 (`mh1-r2scan`), hybrid backend, float32, Langevin NVT at 300 K.
