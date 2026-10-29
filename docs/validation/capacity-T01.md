# CAP-T01: one force graph near the memory limit

Plan task CAP-T01 ([plans/capacity.md](../../plans/capacity.md)), owner request of 29 October 2026. The capacity
scan ([capacity-16gb.md](capacity-16gb.md)) found that several limits on the 16 GB card were the capture of a second
force graph: a larger padded edge size during the warm-up or the steady phase, beside the first graph's pool.

## Verdict

**`EngineConfig.force_graphs`, default `"auto"`:**

- The planner resolves `"auto"` to **`"fit"`** when the plan captures CUDA graphs, and to `"keep"` otherwise.
- `"fit"` is the default because it raises one measured limit and costs nothing at normal sizes. It is exact,
  and the force/stress gate passed.

**Effects:**

- **NbCl5, MH-1, float32:** the largest size that runs 220 MD steps goes from 27,000 atoms (0.71M edges) to
  **28,800 atoms (0.76M edges), +6.7 %**. 30,600 atoms fails during the build, at the first capture.
- **Normal sizes:** a same-process A/B of `keep` against `fit` in both build orders shows no change:

  | Cell | b/a, order ab | b/a, order ba |
  | --- | --- | --- |
  | NaCl, 216 atoms | 0.997 | 1.004 |
  | NbCl5, 1,944 atoms | 1.000 | 0.996 |

  Below these sizes `fit` never acts: no force variant is too large for the free memory.
- **Near the limit there is a cost.** At NbCl5 27,000 atoms, which `keep` also runs, `fit` released twice.
  - It ran at 334.5 against 319.7 ms per step (+4.6 %).
  - The two recaptures fell inside the timed 200 steps.
  - After a release the padded size stays at the largest one captured: 782,848 edges against about 710k at
    the cutoff.
  - The 15 % margin in the estimate is conservative here. A tighter margin is a follow-up only if limits matter
    more than these few percent.

**Not helped:** NaCl float64 (8,000 atoms) and SrTiO3 float32 (14,700 atoms) still fail with `fit` and with
`"one"`. Their limit is not a second graph:

- **The second variant is cheap.** At NaCl float64, 5,832 atoms, the graph pool holds 6.4 GB. The second variant
  was captured beside the first (the warm-up runs inside the pool) and added only about 0.1 GB reserved.
- **The failure is a driver mapping error.** It happens during a capture, once the card is nearly full: "CUDA
  driver error: device not ready / Failed to create GPU mapping".
  - With launch blocking (`CUDA_LAUNCH_BLOCKING=1`, `--diagnose`), NaCl float64 at 8,000 atoms passes under both
    `fit` and `one`.
  - Device memory outside PyTorch's allocator rises from 1.4 to 7.0 GiB during the warm-up, and CUDA then reports
    0 bytes free.
- **Without expandable segments both cells run, but spill into system RAM:**

  | Cell | ms per step | Normal, extrapolated |
  | --- | --- | --- |
  | NaCl float64, 8,000 atoms | 1,802 | about 850 |
  | SrTiO3 float32, 14,700 atoms | 444 | about 260 |

  PyTorch reserves 16.35 GiB on a 15.93 GiB card, so the WSL/WDDM driver pages GPU memory into system RAM. On this
  machine that is the real limit: the driver fails to map new device memory or starts paging. Those cells need
  less memory at the peak (CAP-T03 to CAP-T08), not fewer graphs.
- **`empty_cache_before_capture=False` changes nothing:** both cells still fail.

## Design

`GraphEngine._limit_force_graphs` runs before a force-graph capture (`B`, `ACB`, `ACPB`) while another force graph
is captured.

- **`"one"`** synchronises, releases every captured graph and the pool (`invalidate_graphs`, which retires the
  pool), and returns the blocks to the device (`empty_cache`).
- **`"fit"`** does the same only when the new size would not fit:
  - Estimate: the largest bytes per edge the graph pool held after a force capture (segments of the pool in
    `torch.cuda.memory_snapshot()`), times the new padded size, times `FORCE_GRAPH_MARGIN` (1.15).
  - It compares the estimate with `cudaMemGetInfo` after `empty_cache`.
  - Without a measurement (the first capture, the CPU) it keeps the variant.
- **After a release** `_padded` never returns less than the captured size (`force_graph_floor`). A count that
  falls back replays the larger graph instead of capturing again. Padding is exact: padded edges lie beyond the
  cutoff and contribute zero.
- **Records:** releases count as re-captures (`recapture_log`, the budget), and `info()` reports
  `force_graphs`, `force_graph_releases` and `force_graph_floor`.

## Checks

- **CPU tests** (`tests/cpu/test_force_graphs.py`, replay engine):
  - `keep` holds several force variants; `one` holds one, never pads below the captured size, and gives the same
    trajectory.
  - `fit` releases only when the injected free memory is too small, and acts as `keep` without a measurement.
  - The estimate's margin is checked.
  - The planner resolves `"auto"` and keeps an explicit value.
- **Merge tier on leto** (`-m "not acceptance"`, `-n 8`, GPU hidden; commit `5b7a7da6`):
  - 4,609 passed, 202 skipped.
  - 16 failed, all in `test_colab_driver.py`: `git config` exits 128 in a test repository on the NAS, the known
    environment problem.
  - The golden plan texts are unchanged.
- **Force/stress gate**, local RTX 5060 Ti, default plan (`fit`): passed on NaCl, NbCl5 and NaNbO3
  (`results/cap-t01-20261029/ab/gate.log`). NVE drift was not run: the numerics do not change.
- **A/Bs** (`tools/ab_rewrites.py --a-engine force_graphs=keep --b-engine force_graphs=fit`, 12 blocks of 300
  steps, ABAB):
  - NaCl and NbCl5 in both build orders, as above.
  - NaNbO3 was stopped by owner request: develop on one structure, NbCl5, for speed.

## Setup

- **Machine:** Fabi-PC (WSL2), RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130), MH-1
  (`mh1-r2scan`).
- **FastMACE:** branch `claude/cap-t01-force-graphs` (main `33b25c93` plus CAP-T01). The engine code is the same
  in all GPU runs here (`6d5e148c` to `762527e6`); only tools changed between them.
- **Capacity points:** `tools/bench/capacity_scan.py`, 20 warm-up and 200 timed steps of Langevin NVT, one
  process per size, inside `tools/ci/local_gpu.py run` reservations with the idle check.
- **Raw data:** `results/cap-t01-20261029/`:
  - `points/` holds the capacity points, by suffix:
    - none: default plan;
    - `-fit`: the NbCl5 scan;
    - `-fit2`: the pool measurement;
    - `-one`: `force_graphs=one`;
    - `-onediag`: `--diagnose` (launch blocking, fallback refused) with the default `fit`; the `ENGINE`
      override did not reach diag jobs then;
    - `-onediag2`: `--diagnose` with `one`;
    - `-noexp`: `expandable_segments=off`;
    - `-noempty`: `empty_cache_before_capture=False`.
  - `ab/` holds the gate log and metadata and the A/B JSON.
