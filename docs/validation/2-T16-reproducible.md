# 2-T16: reproducible mode, plain cuEq MH-1 (phase 2 acceptance item 6)

Plan task 2-T16; SPEC §9.3 rule 5 and "`reproducible=True`", §8.7 "Reproducible processes",
owner decisions 13 and 14, §18 risk "plain cuEq float64 is not bitwise reproducible".

## What was built

- `fastmace/planner/reproducible.py`: the allow-list, `qualify`, `prepare_process`,
  `check_repeatable`.
- **Rule 5** (`planner/minimal.py`, `planner/fixed.py`). A reproducible plan:
  - is plain cuEq with the unfused convolution (`cueq_conv_fusion` off under `auto`);
  - is refused for an explicit hybrid backend or fused convolution, for any family other than
    MH-1, for any `cuequivariance_ops` version other than 0.12.0 on CUDA, and for any rewrite
    record in use whose `deterministic` is not True. Each refusal names its component.
  - On plain cuEq every hybrid-only rewrite is off already, `reassociate_first` included, so
    neither `edge_outer_sum` nor its segmented forward is used. No deterministic segmented variant
    was written (conductor answer).
- **The process** (decision 14):
  - The first in-process reproducible plan sets these together, before CUDA starts:
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `TORCHINDUCTOR_DETERMINISTIC=1`,
    `CUEQUIVARIANCE_OPS_U1D_ALGOS=1,1`, `torch.use_deterministic_algorithms(True)` and
    `torch._inductor.config.deterministic`.
  - `ProcessRecord.reproducible` records it. A plan of the other kind then runs in a child
    (`runner-subprocess-reproducible`, `runner-subprocess-not-reproducible`), and the runner gives
    its children the three variables.
  - The compile-cache key carries the deterministic flag.
- **The build check.** The applied records must be declared deterministic, and two evaluations of
  the start positions must be bitwise equal; otherwise `ReproducibilityError` names the kernel.
- **Records.** Every rewrite record declares `deterministic`. `streaming_conv`,
  `generic_streaming`, `segmented_first_sum` and `generic_segmented_first_sum` are False; the rest
  are True.
- **Report.** The plan records `reproducible` and `reproducible.environment`. The report's "why"
  line names the reliance on the undocumented cuEq internal.

## cuEquivariance's run-time kernel choice

- `cuequivariance_ops` 0.12.0 JIT-compiles every `uniform_1d` problem in several
  (schedules, warps per schedule) variants. This covers the unfused `ChannelWiseTensorProduct` and
  the `SymmetricContraction`.
- The library times the variants with CUDA events and keeps the fastest. The default candidate
  list, decoded from `libcue_ops.so`'s read-only data, is (1,1) (1,2) (1,4) (2,2) (4,2) (6,4).
- `CUEQUIVARIANCE_OPS_U1D_ALGOS` replaces that list. The library reads it with `getenv`, splits
  it with `getline` and parses it with `strtol`, as the disassembly shows. It is not documented.
  The family is therefore qualified only on 0.12.0, with the build check, and the plan and report
  record the reliance (conductor answer Q2).
- The Python side offers no pin: `SegmentedPolynomial(options=...)` stores its options and never
  uses them.

### Probe (murphy, 20 October 2026)

**Setup:**
- machine murphy, NVIDIA RTX 4000 Ada (sm_89);
- environment `mace_cueq_latest`: torch 2.14.1+cu130, cuequivariance 0.12.0, cuequivariance_ops 0.12.0;
- FastMACE main 574b7f6 with the flags set by hand;
- MH-1 `mh1-r2scan`, NaCl 3x3x3 (216 atoms), float64, plain cuEq, unfused convolution;
- Langevin 300 K, 2 fs, 200 steps, seed 20261020;
- each run a fresh process, with `CUBLAS_WORKSPACE_CONFIG`, `TORCHINDUCTOR_DETERMINISTIC` and
  the deterministic flag set.

The GPU was shared with our own untimed jobs (two full-tier merge trials and another worker's
test), with the conductor's approval. No other user had a process on the GPU.

| `CUEQUIVARIANCE_OPS_U1D_ALGOS` | run 1 | run 2 | run 3 |
| --- | --- | --- | --- |
| unset (cuEq's six candidates) | `3fbd04281fb7e8d5` | `66c3aa8f81d16371` | `66c3aa8f81d16371` |
| `1,1` | `66c3aa8f81d16371` | `66c3aa8f81d16371` | `66c3aa8f81d16371` |

The table gives the first 16 hex digits of sha256(positions ‖ momenta) after 200 steps. The
energy was −1481.0775182701777 eV in every run.

- Unset, the deterministic flags are not enough: one of three processes picked another variant
  and diverged bitwise.
- Pinned, all three runs agree, and with the variant the timing usually picks.

## Acceptance 6

`tests/gpu/test_reproducible_gpu.py` setup:
- MH-1 `mh1-r2scan`, Langevin 300 K, 2 fs, seed 20261020, through `fm.Simulation` with
  `Workload(reproducible=True)`;
- every run a fresh process;
- per case, three runs of 500 steps with a restart write at step 250, and the first run's restart
  resumed in a fresh process for the last 250 steps;
- pass means positions and momenta bitwise equal across all four, and every process reproducible:
  plain cuEq, unfused convolution, in-process, qualified, the three variables and the flag set.

### murphy

- RTX 4000 Ada (sm_89), environment `mace_cueq_latest` (torch 2.14.1+cu130, cuequivariance_ops
  0.12.0), FastMACE efa23a3.
- The GPU had no process at the start (05:34). Other jobs of ours could run on it meanwhile; no
  other user's process was on it.

| Case | Result | Final energy (eV) | Seconds (run 1, 2, 3, resume) |
| --- | --- | --- | --- |
| NaCl 3x3x3, float64 | bitwise equal | −1479.6391301040 | 106.8, 79.7, 78.8, 64.1 |
| NaCl 3x3x3, float32 | bitwise equal | −1479.6391706467 | 61.4, 33.6, 32.8, 41.3 |
| NbCl5 2x2x2, float64 | bitwise equal | −3048.3407240580 | 112.6, 86.1, 85.9, 67.6 |
| NbCl5 2x2x2, float32 | bitwise equal | −3048.3408415318 | 60.8, 34.0, 33.0, 41.6 |
| runner trigger: CUDA started, then `reproducible=True`, 100 steps | runs in a child, bitwise equal to a fresh reproducible process with a restart write at step 0 | | |

5 passed in 1189 s.

The first attempt (fead58f) failed in every case:
- `MinimalPlanner.plan` detects the hardware with `torch.cuda.get_device_properties`, which
  starts CUDA, so the "reproducible after CUDA started" trigger sent each plan to a child, and
  each child did the same.
- efa23a3 prepares the process before the hardware detection.

### Local RTX 5060 Ti

- NVIDIA GeForce RTX 5060 Ti (sm_120) under WSL2, environment `mace_cueq_latest` (torch
  2.14.1+cu130, cuequivariance_ops 0.12.0), FastMACE 344bda5.
- The run held the local GPU reservation (`tools/ci/local_gpu.py`) and was untimed.

| Case | Result | Final energy (eV) | Seconds (run 1, 2, 3, resume) |
| --- | --- | --- | --- |
| NaCl 3x3x3, float64 | bitwise equal | −1479.6391301040 | 137.3, 91.1, 87.8, 70.1 |
| NaCl 3x3x3, float32 | bitwise equal | −1479.6392836571 | 71.4, 42.3, 42.0, 47.8 |
| NbCl5 2x2x2, float64 | bitwise equal | −3048.3407240580 | 131.1, 97.6, 95.3, 72.7 |
| NbCl5 2x2x2, float32 | bitwise equal | −3048.3408837318 | 76.7, 41.9, 41.3, 45.8 |
| runner trigger | runs in a child, bitwise equal to a fresh reproducible process | | |

5 passed in 1425 s.

The two machines agree to the printed digits in float64 and differ in float32 (for example NaCl
−1479.6391706 against −1479.6392837 eV). That is expected: §9.3 promises nothing across GPU models.

**Verdict: acceptance 6 holds.** Plain cuEq MH-1 is qualified for `reproducible=True` on both
machines, under the pinned `cuequivariance_ops` 0.12.0.

## Commands

CPU (murphy's CPUs, because the local PC was reserved for 2-T13's timing):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 $E/bin/python -m pytest -n 8 -p no:cacheprovider -q -rf tests/cpu
```

Result: 2241 passed, 180 skipped, at fead58f (environment `mace_cueq_latest`). After efa23a3, the three files it touches pass (102 tests). The new test file
`tests/cpu/test_reproducible.py` (27 tests) also passed alone in `fastmace_cpu`, before the
reservation.

GPU, on murphy, from the copy `/mnt/homes/zwingel/fastmace-branches/2-T16/branch`, with the
environment block of `docs/validation/phase-0.md`, plus `FASTMACE_MODELS` and
`TORCHINDUCTOR_COMPILE_THREADS=4`:

```bash
$E/bin/python -m pytest -n 0 -p no:cacheprovider -s -v -rf tests/gpu/test_reproducible_gpu.py
```
