# 2-T05: gate resource policy and statuses (acceptance 3)

Plan task 2-T05 ([plans/phase-2.md](../../plans/phase-2.md)); SPEC §12.3 "Status" and "Resources".
Branch `fastmace/2-T05`, FastMACE commit `13ceb5c` (code and tests; this note and the results
come in the commit after it). No research-engine baseline is involved.

## What the gate does now

When the gate is requested (`check="gate"` or `"allow_unavailable"`), its references try, in order:

1. the plan's device on the production cell;
2. after an out-of-memory or a device failure on a GPU: the CPU in float64 on the production cell,
   in a child process, if the estimated time is within `GateConfig.cpu_float64_time_limit_s`
   (default 300 s) and the estimated memory × 1.25 within `MemAvailable`;
3. the reduced cell (the largest diagonal supercell of the structure's primitive cell that is
   smaller than the production cell and at least `reduced_cell_min_height_cutoffs` × the cutoff
   in every height) on the plan's device, walking down in size after an out-of-memory (each next
   cell at most 3/4 of the atoms of the last that failed);
4. the reduced cell on the CPU, with no cell larger than the smallest that ran out of memory.

A cell is never larger than one that ran out of memory. A reference failing for any other reason
is `unavailable` at once (decision 3). On a reduced cell the status is `passed-reduced`. The
production path is then evaluated eagerly on the reduced cell's copies by a separate force field
built from the pristine model with the plan's rewrites. On the production cell, the captured replay
is compared with a direct evaluation of the same force field. That direct evaluation is
`MACEForceField.evaluate`: the same model function, uncaptured, on a fresh unpadded neighbour list.
Each quantity must agree within the gate's own limit for it, max(2 × float32 yardstick, floor),
taken from the reduced cell's comparison.

The record (`GateRecord`) holds the device, the reduced cell's repeat, every tried step (device,
cell, atoms, process, outcome, reason, exit code, estimate, measured peak, memory cap) with its
time in `seconds["step_<i>_s"]`, and the replay block with its bound and the bound's source. The
report line reads `passed-reduced (reduced cell 2×3×5 of the primitive cell on cuda:0, replay
checked on the production cell; …)`.

## Result on the RTX 5060 Ti (acceptance 3)

Local PC, NVIDIA GeForce RTX 5060 Ti (16 GiB, WSL2, 15 GB RAM), 20 October 2026, attempt-5
(04:07–04:16), under the local GPU reservation (`tools/ci/local_gpu.py`, 2-T18).
Parent `~/environments/mace_cueq_latest` (pytest); test child `~/environments/mace_cueq_oeq`
(hybrid OEq, float32 production). NbCl5 with 1944 atoms (the primitive cell 3×3×6, the size of
1b-T03's 3×3×3; the same cell shape was not checked), rattled 0.03 Å, NVT 300 K,
`check="allow_unavailable"`. Raw: [results/2-T05-rtx5060ti-20261020/attempt-5/](../../results/2-T05-rtx5060ti-20261020/attempt-5/)
(`cases.jsonl`, `pytest.log`, `free.log`, `junit.xml`).

| Model | Step | Device, cell | Outcome | Time |
| --- | --- | --- | --- | ---: |
| POLAR-1-L | 1 | GPU, production (1944 atoms) | out of memory (child exit 1) | 27 s |
| | 2 | CPU, production | out of memory under the cap (child exit 3) | 129 s |
| | 3 | GPU, reduced 3×3×5 (1620 atoms) | out of memory | 28 s |
| | 4 | GPU, reduced 2×3×5 (1080 atoms) | **ran** | 55 s |
| OMOL-0 XL | 1 | GPU, production (1944 atoms) | out of memory (child exit 1) | 22 s |
| | 2 | CPU, production | declined: 2520 s > 300 s, 24.8 GiB × 1.25 > 9.78 GiB available | 73 s |
| | 3 | GPU, reduced 3×3×5 (1620 atoms) | **ran** | 40 s |

| Model | Status | Energy (meV/atom) | Force (eV/Å) | Replay difference, force (eV/Å) | Gate time |
| --- | --- | --- | --- | --- | ---: |
| POLAR-1-L | `passed-reduced` | 5.1e-5 (limit 0.01) | 1.9e-4 (limit 5.7e-4) | 4.3e-5 (limit 5.7e-4) | 328 s |
| OMOL-0 XL | `passed-reduced` | 1.2e-4 (limit 0.01) | 1.7e-4 (limit 3.6e-4) | 6.2e-5 (limit 3.6e-4) | 176 s |

Both end on a reduced cell on the GPU. On this machine the CPU float64 step declines or caps every
large cell: the CPU references run cuEquivariance's naive kernels, which are slow and need a lot of
memory (below). Lowest available host memory during the run: 4.6 GB; largest reference child:
9.0 GB resident (the capped CPU child of POLAR-1-L).

## Why GPU references run in a child process

On a CUDA plan every GPU reference step runs in a child process (conductor decision, 20 October
2026; the conductor amends SPEC §12.3). A GPU out-of-memory can leave the CUDA context unusable,
so the gate cannot try the next step in the same process. Attempt-2, with in-process references:

- POLAR-1-L: the float64 reference failed at 13.5 GB with `CUDA driver error: device not ready`,
  with no "out of memory" in the text;
- OMOL-0 XL: after its reference failed, building the production engine in the same process
  failed with `Sticky error detected … CUDA_ERROR_OUT_OF_MEMORY from cuStreamIsCapturing`.

A GPU reference child that fails with a CUDA error, a non-zero exit or a kill is a "device failure"
and the ladder goes on, with the child's error text and exit code in the record. Each GPU step
costs a process start and a model load (22–55 s here, in `seconds["step_<i>_s"]`); the gate is
opt-in.

## The WSL crash of attempt-3, and the memory cap

Attempt-3 (02:48–02:51) took down the whole WSL2 VM, not just the child: available memory fell
from 8.4 GB to 0.34 GB in 30 s during POLAR-1-L's CPU float64 step, while the GPU held about 1 GB.
The conductor restarted WSL (`wsl --shutdown`, about 02:59; other sessions saw
`HCS_E_CONNECTION_TIMEOUT`). The estimate was not what let it through: the calibration itself ran
out of memory before any estimate existed. Probes since (`estimate_probe.py`,
`polar_cpu_probe.py`, each under a cgroup `MemoryMax` and the cap):

- POLAR-1-L float64 on the CPU needs more than 6.5 GiB above the loaded model even on its 36-atom
  primitive cell (killed at both 7 GiB and 11 GiB cgroup limits);
- MH-1 (`mh1-r2scan`): the calibration boxes (56 and 106 atoms) took 33 s and 94 s and 0.11 and
  2.24 GB of peak growth. The estimate for 1944-atom MH-1 is 41 GB and 1733 s per evaluation:
  declined.

So every CPU reference child now caps its own data segment (`RLIMIT_DATA`, which follows the
resident size closely for torch: +3.6 GiB data for +3.2 GiB resident on MH-1) before the model
loads, at `MemAvailable` − 2 GiB. Once a cell is chosen, the cap is lowered to at most 3 × its
estimate. An allocation past the cap fails inside the child, which exits with code 3. The record
says "out of memory (capped at X GiB)", and the ladder goes on. The estimate takes the largest of
the per-edge, per-atom and (PolarMACE) per atom × volume extrapolations from the two calibration
boxes and their two-point fit. The measured peak of the evaluation is recorded next to it. In
attempt-4 and attempt-5 the cap stopped POLAR-1-L's CPU child at about 9 GB resident.

## Attempts

| Attempt | What happened |
| --- | --- |
| aborted-1 | started before the conductor's "machine free" reached this session; stopped at its request (sibling suites) |
| attempt-2 | in-process GPU references: sticky CUDA errors, both cases failed (above) |
| attempt-3 | the CPU calibration exhausted host memory; WSL went down |
| attempt-4 | the ladder walked as designed; both failed the replay check against the 1e-5 eV/Å floor (float32 rounding: 4.8e-5, 6.0e-5) |
| attempt-5 | replay bounded by the gate's own limit: both `passed-reduced` |

## Tests

- CPU (`tests/cpu/test_gate_resources.py`): reduced-cell rule (cubic, triclinic, rattled, small, disordered),
  the estimate, the ladder with simulated out-of-memory and device failures at each step, real
  child processes that exit 1, are killed, or allocate past a 256 MiB cap, `passed-reduced` end to
  end on a tiny model (NVT and NPT, state and trajectory unchanged), the replay check's bound, the
  CPU step's calibration and decision, T15's reduced gate (one copy, a subset of quantities), the
  report line.
- GPU (`tests/gpu/test_gate_resources_gpu.py`, `acceptance` tier): the two cases above.
