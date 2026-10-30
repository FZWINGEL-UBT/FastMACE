# CAP-T06: a smaller noise batch

Plan task CAP-T06 ([plans/capacity.md](../../plans/capacity.md)). Owner, 30 October 2026, through the conductor:
shrink the prefetched Langevin noise so that it scales less with the atom count, with the noise stream bitwise
unchanged, restarts included.

## Verdict

**Exact, and the default.**
- **The cap.** The drawn-ahead noise batch now holds at most 64 MiB of float64:
  `fastmace.integrators.langevin.NOISE_BATCH_BYTES`, applied by `noise_steps`.
- **Batch size by cell.** `IntegratorConfig.noise_batch` (256) is still the most steps per batch. Below about 5,500
  Langevin atoms nothing changes. At 51,840 atoms the batch holds 26 steps instead of 256.
- **The stream.** The numbers and their order do not depend on the batch size.

**NbCl5 at its limit (51,840 atoms): the peak is 0.53 GiB lower.**
- Allocated: 12.93 against 13.50 GB. The difference, 572,313,600 bytes, is exactly 230 steps × 6 × 51,840 × 8 B.
- Reserved: 15.34 against 15.90 GB.
- That is 3.6 % of the reserved peak. It is not enough for the next grid step: 54,432 atoms (6×6×21) still runs out of
  memory, at capture.

**Step time: no change.** In the same-process A/B, in both build orders, b/a is 1.000–1.005 at 9,216 atoms, where the
batch is 151 steps instead of 256. At 1,944 atoms both sides draw 256 steps; there the A/B (0.990–1.004) measures
the noise floor.

**Restarts.** A restart written with another batch size resumes bitwise. This covers every restart written before
this change, because each of them holds a 256-step batch.

**Bitwise: a reproducible float64 trajectory is the same before and after, across restarts too.**
- **Cell and runs.** MH-1 on NbCl5 2×2×2 (288 atoms), 300 steps, with a restart written at step 150. Each run is a
  fresh process.
- **The cap.** On the branch it is set to 16 steps of this cell, so the batch is far shorter than a restart's unused
  rows.
- **Result.** Positions, momenta and step are bitwise those of main (256-step batch) in all three cases:
  - uninterrupted;
  - resumed from main's restart, whose 106 unused rows span 7 refills;
  - resumed from the branch's own restart.
- **Not run: the cap binding by itself.** That needs more than 5,461 atoms. The reproducible float64 plan ran out of
  memory at 6,300 and at 9,216 atoms on the 16 GB GPU. The CPU tests cover the cap's own sizes with the same code.

## What changed

- **`noise_steps(limit, numbers)`** gives `min(limit, NOISE_BATCH_BYTES // (8 · numbers))`, and at least 1.
  - Langevin draws 6N numbers per step, BAOAB 3N plus the cell numbers, and CSVR 2.
  - `NOISE_BATCH_BYTES` is an integer constant, like `CAPTURE_SNAPSHOT_OFFLOAD_BYTES`. The A/B sets it to 2⁶² for the
    side "as before".
- **`RandomStream.queued`** (host, float64) holds drawn rows that come before the generator's next draw but are not
  in the batch. `refill_noise` takes them first.
  - It puts them at the end of the batch and starts the cursor there.
  - With nothing queued, it draws the batch's steps from the generator, in its order, as before.
- **Restarts with another batch size.**
  - **Loading.** A restart whose batch has another length but the same layout gives its unused rows to the queue
    (`io.restart.load_state`, and `allocate_noise` when the restart was loaded before the integrator was built).
  - **Writing.** A restart written while rows are queued stores all unused rows from the start, with `noise_next`
    0, so any batch size reads it.
  - The format is unchanged; a restart with nothing queued is written as before.
- **Rejected events** (`events.Transaction`) restore the queue with the generator and the cursor.

Why the stream is the same:

- numpy's `Generator.standard_normal` and `normal` fill an array in order and keep no state between calls. Drawing
  `k` steps then `m` steps gives the same numbers as drawing `k + m` at once.
- CSVR's draw is a per-step loop, so it was never batch-dependent.

## Step time

`tools/ab_rewrites.py --prepare`, same process, MH-1 hybrid float32 NVT at 300 K, default plan on both sides.

- **Side a:** `fastmace.integrators.langevin:NOISE_BATCH_BYTES=2**62` (256 steps, as before) while it is built and its
  first step's stages are captured.
- **Side b:** the default.

Median step in ms. Ratios are b/a, with the range over block pairs in brackets.

| Cell | Batch, a / b | Blocks | order ab (abab): a / b, b/a | order ba (abba): a / b, b/a |
| --- | --- | --- | --- | --- |
| NbCl5 3×3×3, 1,944 atoms | 256 / 256 | 12 × 300 steps | 17.71 / 17.54, **0.990** (0.990–0.997) | 17.73 / 17.81, **1.004** (0.998–1.012) |
| NbCl5 4×4×8, 9,216 atoms | 256 / 151 | 8 × 100 steps | 88.10 / 88.12, **1.000** (0.985–1.009) | 87.93 / 88.38, **1.005** (0.997–1.013) |

Every run was valid: nothing compiled while timed, and no other process used the GPU.

- **At 1,944 atoms** both sides run the same code. The two orders bracket 1, so the spread is the noise floor of the
  same-process A/B.
- **At 9,216 atoms** one refill (a host draw and one copy) serves 151 steps instead of 256. The draws are the same
  numbers, so the per-step cost does not change. The once-per-batch overhead is spread over fewer steps, and is not
  visible.

## The limit points

`tools/bench/capacity_scan.py point`, NbCl5 float32 default plan, 200 steps:

| Cell | Side | Status | Peak allocated | Peak reserved |
| --- | --- | --- | ---: | ---: |
| 6×6×20, 51,840 atoms | before (main `fbd8a6ed`) | ok | 13.50 GB | 15.90 GB |
| 6×6×20, 51,840 atoms | after | ok | 12.93 GB | 15.34 GB |
| 6×6×21, 54,432 atoms | after | out of memory at capture | 13.53 GB at failure | 16.05 GB |

The two 51,840-atom points are not a timing comparison: they come from separate processes at the memory limit.

## Tests

- **CPU, on leto** (busy check 0, GPU hidden, `-n 8`, commit `6762eeb5`): **551 passed, 9 skipped**.
  - The files: `test_layering`, `test_integrators`, `test_events`, `test_restart`, `test_restart_compat`,
    `test_restart_v3`, `test_restart_versions`, `test_canonical_restart`, `test_csvr_resume`,
    `test_nve_csvr_berendsen`, `test_reproducible`, `test_engine`, `test_fix_atoms`, `test_fixcm_projection`,
    `test_integrator_qualification`, `test_isolation`, `test_mc_displace`, `test_mc_evaluator`, `test_config` and
    `test_docs_options`.
  - The new tests, run on main first: 11 failed there, as they should. The twelfth, the refusal of another layout,
    passes on main too.
- **New tests:**
  - `test_noise_steps_cap_the_batch_bytes`.
  - `test_a_capped_batch_draws_ases_stream`: Langevin and full-cell BAOAB with the cap lowered, against
    numpy's stream.
  - `test_a_restart_resumes_with_another_noise_batch`: Langevin and full-cell BAOAB, written with 7 steps and
    resumed with 2, 5 and 64. A second restart, written while rows are still queued, resumes with 7. All are bitwise
    equal to the uninterrupted run.
  - `test_a_restart_loaded_before_the_integrator_resumes_with_another_batch`.
  - `test_a_batch_of_another_layout_is_refused`.
  - `test_a_rejected_event_restores_the_queued_noise_rows`.
- **One bug found by the first leto run:** `state.py` imports numpy for type checking only. The queue now imports it
  where it uses it (`6762eeb5`). The first GPU bitwise run used the commit before that fix. Its resume from main's
  restart failed with that error, so the bitwise runs were repeated on the fixed commit.

## Setup

- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130).
- **GPU access:** `tools/ci/local_gpu.py run` reservations, with the idle check before every job.
- **Sources:** pinned copies of FastMACE `03c33f74` (A/B and points; its only difference from the final branch is the
  numpy import in the queue, which these runs do not reach) and `a8955b83` (bitwise runs), against main `fbd8a6ed`.
  The first bitwise attempt ran on `03c33f74`. Its resume from main's restart failed on the missing import, and its
  9,216-atom float64 runs ran out of memory. Its records are not used.
- **Model:** MH-1 (`mh1-r2scan`), hybrid backend, float32. The bitwise runs use the reproducible plan instead:
  plain cuEq, float64.
- **Records:**
  - `results/cap-t06-20261030/`: the run script `local.sh`, the bitwise driver `bitwise.py`, the A/B records, the
    points and the bitwise records;
  - the restart directories stay in `~/fastmace-capacity/cap-t06-b/` (WSL).
