# Tiled call speed: the host state's write copies on a side stream (10 November 2026)

**Task** (conductor, 10 November 2026): keep improving the tiled and wavefront paths. Setup as
[tiled-speed-20261108](../tiled-speed-20261108/README.md): local RTX 5060 Ti 16 GB (WSL2), `mace_cueq_oeq`
(torch 2.14.1+cu130), `omat0-medium`, float32, SrTiO3 58×58×58 (975,560 atoms), every run through
`tools/ci/local_gpu.py run` from a `git archive` tree. Source `f34f35e8` (main `5249d8d8` plus the switch).

## The switch

`tiled.TILE_COPY_STREAM` (exact; None: on for `TILED_SPEED_GPUS`). With the state on the host and overlap, every
tile's deferred writes (sweep 1: x and sc; sweep 2: dE/dx and dE/dsc) were copied to pinned host memory on the
compute stream, so the next tile's kernels queued behind the link. The wavefront profile at d2435c8e (job 23) had
0.44 s of device-to-host copies in a 7.72 s call, the GPU busy 7.10 s. Now the compute stream clones the values and
device indices (the slot's outputs and index buffers hold the next tile before a side-stream copy would run), a side
stream copies the clones, and the host applies the same values in the same order after the copy's event.

## Result

| A/B, 975,560 atoms (`tools/bench/tiled_ab.py`, 4 blocks of 2 calls, 3 warm-up calls per side) | b/a (ab / ba) | Medians |
| --- | --- | --- |
| Wavefront (host state, aliased), off against on | **0.962 / 0.964** | 7.80 → 7.51 µs/atom |
| Host state, plain schedule, off against on | **0.960 / 0.959** | 6.99 → 6.70 µs/atom |

Pairs: wavefront 0.956–0.969, host state 0.954–0.965. Energies agree to 1e-9 relative (the GPU's atomics).

**Profile** (`results/tiled-speed-20261108/jobs/wprof.py`, job 36, the defaults): wall 7.24 s, GPU busy 7.15 s
(job 23: 7.72 s and 7.10 s). The copies are as long as before (device-to-host 0.444 s, host-to-device 0.562 s), now
beside the compute; the clones add 0.052 s of device-to-device copies. The wavefront call is GPU-bound: 7.42 µs/atom
wall against 4.79 µs/atom for the whole-system call (1.55×, was 1.61×).

**Tests.** `tests/gpu/test_tiled_speed_cuda.py`, 3 passed (job 35): the switch on against off on the host-state
wavefront (OMAT-0 medium, 1,080 atoms in tiles of 256), forces within float32 atomics, and against the whole-system
call. CPU: the switch stays off without CUDA (`test_host_worker_is_bitwise_the_call_that_writes_on_the_calling_thread`).

Job 35 ran the GPU test and the wavefront's ab order; its unit was stopped by systemd at 09:48:59 (not by this
session) during the ba order. Job 36 ran the rest.

## Records

- `ab/`: the four A/B records.
- `profiles/`: `job35/gputest.log`, `job36/wprof-58.json` and `wprof.log`, the jobs' logs.
- `jobs/`: `job35.sh`, `job36.sh` (launched by the `launch.sh` of tiled-speed-20261108 as system units behind the
  local GPU reservation).
