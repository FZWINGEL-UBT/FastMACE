# Capacity on the RTX 5060 Ti 16 GB: atoms and edges in NVT MD

Owner request, 29 October 2026, through the conductor: how many atoms and edges FastMACE can run on the local
RTX 5060 Ti 16 GB. The follow-up work to raise the limit is [plans/capacity.md](../../plans/capacity.md).

## Verdict

**Default plan, largest size that runs 20 warm-up steps plus 200 timed steps of Langevin NVT without running
out of memory:**

| Structure | Model | Precision | Max atoms | Max edges at 6 Å (candidates at 6.3 Å) | Peak allocated / reserved | ms per step | First failing size and limit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NaCl | MH-1 | float32 | **21,952** (14³) | **0.70M** (1.04M) | 11.13 / 12.98 GiB | 274 | 27,000 atoms (0.86M edges): first capture of the force graph |
| NbCl5 | MH-1 | float32 | **27,000** (5×5×15) | **0.71M** (0.94M) | 12.21 / 14.32 GiB | 320 | 28,800 atoms (0.76M edges): capture of a second, larger force-graph variant in the steady phase |
| SrTiO3 | MH-1 | float32 | **13,720** (14³) | **1.02M** (1.25M) | 13.20 / 14.57 GiB | 244 | 14,700 atoms (14×14×15, about 1.09M edges): out of memory in the steady phase (as for NbCl5, a capture at a larger padded edge size) |
| SrTiO3 | OMAT-0 medium | float32 | **13,720** (14³) | **1.02M** (1.25M) | 13.63 / 14.36 GiB | 166 | 14,700 atoms (about 1.09M edges): out of memory in the steady phase, as for MH-1 |
| NaCl | MH-1 | float64 | **5,832** (9³) | **0.19M** (0.28M) | 6.67 / 7.29 GiB | 616 | 8,000 atoms (0.26M edges): second force-graph variant, during the warm-up |
| NbCl5 | MH-1 | float64 | **7,128** (3×3×11) | **0.19M** (0.25M) | 6.77 / 7.67 GiB | 685 | 9,504 atoms (0.25M edges): second force-graph variant, during the warm-up |

- **The limit is set by edges and atoms together.** A fit over the default plan's passing float32 points (NaCl,
  NbCl5 and SrTiO3 at 32, 26 and 74 edges per atom, without rule 12's drops) gives
  `max_allocated` ≈ 0.23 GiB + 5.9 kB per edge + 268 kB per atom, residuals at most 0.07 GiB. On NaCl and
  NbCl5 the per-atom (node) share is the larger one.
- **The steady-state step is not what fails first.** Every failure is a CUDA-graph capture of the force stage
  (B). Either it is the first capture during setup, or it is a capture at a larger padded edge size during the
  warm-up or the steady phase. Both happen with the cache holding most of the card. A capture needs its own
  pool beside what the earlier graph keeps.
- **Rule 12 does not predict the limit.**
  - For MH-1 it predicts about 1.8 times the measured peak at these sizes: 23.1 GiB at 21,952 NaCl atoms
    against 12.98 GiB reserved.
  - It warns from about 13k atoms. It also turns off `streaming_conv` and its dependants (`chunked_conv_backward`,
    `fused_folded_forward`, `wprime_backward`), which saves about 7 MiB. That lowers the limit and slows the
    step (see "Candidate" below).
  - For OMAT-0 medium it underestimates: 12.8 GiB predicted against 14.36 GiB reserved at 13,720 atoms.
- **A failure is a crash, not a clean refusal.** An out-of-memory error inside a capture is a driver error
  ("device not ready / Failed to create GPU mapping", or `CUDA_ERROR_OUT_OF_MEMORY` at a stream synchronize).
  It leaves the CUDA context with a sticky error. The fallback then replans, calls `cudaMemGetInfo` and raises
  the sticky error from there.

## Candidate: keep `streaming_conv` on (rule 12 not allowed to drop it)

Same scan with `RewriteConfig(streaming_conv="on")`. An explicit setting is never dropped by rule 12. No other
change.

| Structure, float32 | Default plan | `streaming_conv=on` | Change |
| --- | --- | --- | --- |
| NaCl | 21,952 atoms, 0.70M edges, 274 ms | **27,000 atoms, 0.86M edges, 275 ms** | +23 % atoms; 32,768 fails in the first capture |
| NbCl5 | 27,000 atoms, 0.71M edges, 320 ms | **30,600 atoms, 0.80M edges, 292 ms** | +13 % atoms; 36,720 fails in the build |

At equal size the candidate is faster once the default plan has dropped the rewrites:

- NbCl5 at 25,200 atoms: 238 ms against 291 ms.
- NaCl at 27,000 atoms (the candidate): 275 ms, the same as the default plan at its smaller limit of 21,952
  atoms (274 ms).

At sizes where rule 12 drops nothing the plans are identical, for example NbCl5 at 14,976 atoms: 145 against
148 ms, run to run. Changing rule 12's drop rule and its fitted `SAFETY` factor is SPEC text (§9.3 rule 12,
decision 51). It is task CAP-T02 and waits for the conductor.

## Where the memory goes at the limit

The `--diagnose` mode records the allocator's history and refuses the fallback, so the first failure is the one
raised. These figures are for NaCl float32 at 27,000 atoms, default plan, failing in the first force-graph capture:

- **Allocator:** 8.7 GiB live and 15.1 GiB reserved, so about 6 GiB is cached or fragmented when the driver
  mapping fails.
- **Live, by allocation site:**

  | Allocation site | Live |
  | --- | --- |
  | Model activations in the force call | 5.5 GiB |
  | MACE and e3nn code outside FastMACE | 1.0 GiB |
  | `segment_gemm` joined products | 0.82 GiB |
  | The registry snapshot taken around each capture | 0.49 GiB |
  | The Langevin noise batch (512 steps of float64 noise drawn ahead) | 0.31 GiB |
  | The radial table | 0.21 GiB |
  | Neighbour-list buffers | 0.17 GiB |

- **Outside PyTorch's allocator:** 1.13 GiB at the start (the CUDA context and the desktop's share). It grows
  to 2.2 GiB after the float32 build and to 1.4–1.8 GiB in float64 (library workspaces, compiled modules,
  graph executables).
- **The float64 failures:** NaCl float64 at 8,000 atoms still has 5.9 GiB free after the build. It fails while
  capturing the second force-graph variant, at 8.85 GiB reserved: the earlier graph's pool and the new graph's
  warm-up and capture are live together.

## Desktop headroom

The desktop's share of the card, as Windows' `nvidia-smi` reports it, varied between 0.6 and 3.5 GiB during the
runs (recorded per point). CUDA in WSL reported **14.80 GiB free of 15.93 GiB at the start of every run**,
whatever that share. The measured limits therefore come with 14.8 GiB usable.

On a headless 16 GB card, about 15.93 − 0.4 (context) ≈ 15.5 GiB would be usable, about 0.7 GiB more. At the
fitted marginal cost per atom, the corrected limits are:

| Structure, precision | Marginal cost per atom | Measured limit | Headless estimate |
| --- | --- | --- | --- |
| NaCl, float32 | 0.46 MB | 21,952 | about 23,500 (+7 %) |
| NbCl5, float32 | 0.42 MB | 27,000 | about 28,800 (+6 %) |
| SrTiO3, float32 | 0.70 MB | 13,720 | about 14,800 (+8 %) |

These are estimates, not measurements. The float64 limits are set by the second-variant capture, not by the
total, so no correction is given for them.

## Comparison with symmetrix-xl's SrTiO3 numbers

symmetrix-xl (bonan-group/symmetrix-xl, README) reports single-GPU float32 capacity with MACE-OMAT-0 medium on
cubic SrTiO3 (a = 3.905 Å, 6.0 Å cutoff, 0.5 Å skin). This scan uses the same cell and model with FastMACE's
0.3 Å skin.

| GPU | ML-IAP + cuEquivariance | symmetrix-xl standard streaming | symmetrix-xl tiled streaming |
| --- | --- | --- | --- |
| A100 80 GB | 24,565 | 1,250,235 | 11,240,455 |
| RTX 5090 32 GB | 8,640 | 486,680 | 4,152,920 |

FastMACE reaches 13,720 atoms on 16 GB. Scaled linearly by memory to 16 GB:

- ML-IAP + cuEquivariance does about 4,300 atoms, so FastMACE fits about 3x more.
- symmetrix-xl's standard streaming does about 243k atoms, about 18x FastMACE.
- symmetrix-xl's tiled streaming does about 2.1M atoms, about 150x FastMACE.

The gap is the activation memory FastMACE keeps for the force backward: 5.9 kB per edge and 268 kB per atom.
symmetrix-xl's streamed execution does not hold that per-edge state (docs/streamed_edge_execution.md in their
repository). CAP-T07 and CAP-T08 in the plan are the corresponding work here.

Their numbers come from a capacity probe (one force call, "not sustained 20-step molecular-dynamics results");
these are 220 MD steps including every capture. Their skin is 0.5 Å, against 0.3 Å here.

## Setup

- **Machine:** the local PC (Fabi-PC): RTX 5060 Ti 16 GB (sm_120), driver 610.88, WSL2. Environment
  `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130), backend hybrid (the default plan's).
- **FastMACE:** `claude/capacity-16gb`, tool commits 68947395 to 8f9a8b8b. The library code is main d438ab01's
  throughout; only `tools/bench/capacity_scan.py` and the run scripts changed between them.
- **Model:** `mh1-r2scan` (MH-1, matpes_r2scan head), and `omat0-medium` for SrTiO3. Default plan (rule 12 as
  it decides), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (FastMACE's).
- **Structures:**
  - NaCl and NbCl5: the 1a-T18 baseline export's R2SCAN cells.
  - SrTiO3: cubic Pm-3m, a = 3.905 Å, built in the tool.
  - Supercells are near-isotropic repeats (`round(L / a_i)` per axis for growing L).
- **Run:** `fm.Simulation` with `fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300,
  friction_per_fs=0.002)`, Maxwell-Boltzmann momenta at 300 K, seed 1. Each size runs in a fresh process: 20
  warm-up steps (the captures), then 200 timed steps. A size passes when all 220 steps run.
- **Scan:**
  - Sizes grow by 1.5x in atoms until one fails, then bisect between the largest passing and smallest failing
    size.
  - For SrTiO3, 14×14×15 and 14×15×15 were added by hand, since the isotropic rule only gives cubes.
  - Each 13-minute reservation went through `tools/ci/local_gpu.py run`, with an idle check before each job (no
    compute process, under 10 % utilisation in both `nvidia-smi` views for 30 s).
- **Timing:** ms per step is from the scan, not from a quiet-machine A/B. It comes from one process per size
  and serves to compare sizes, not plans; the same-process A/B rule applies to any default change.
- **Edges:** pairs within the 6 Å cutoff at the start; candidates within cutoff + skin (0.3 Å). The pair-buffer
  capacity is 1.15 × candidates, rounded up to 256. The force graph runs on the cutoff count padded up to the
  edge bucket (5 % of the initial count).
- **Raw data:** `results/capacity-16gb-20261029/`:
  - one JSON per point;
  - the scan summaries and the run log;
  - `local.sh` and `scan_loop.sh`.

  The points have these suffixes:
  - `-sc`: `streaming_conv=on`;
  - `-omat`: OMAT-0 medium;
  - `-diag*`: diagnostics with blocking launches;
  - `-mem`: memory outside the allocator.
