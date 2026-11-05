# Phase 2: noise floor of the local RTX 5060 Ti

Plan task 2-T12 repeats 1a-T18's noise-floor measurement on the local PC
(RTX 5060 Ti 16 GB, WSL2), so that later local timings (2-T13, 2-T19, 2-T20,
2-T23) can be judged against this machine's own floor rather than murphy's.
The method is the same: two identical sides in one process, alternating
timed blocks. Any ratio away from 1 is noise.

## Result

| Crystal (3×3×3) | Atoms | Sides | Processes | Step time (ms) | B/A, 12 blocks | B/A, 4-block windows | Per-pair ratios | Block spread | Side A median across processes |
| --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| NaCl | 216 | FastMACE ×2 | 3 | 3.534–3.594 | 1.0002–1.0108 | 0.9960–1.0071 | 0.9896–1.0190 | 5.1–7.5 % | 1.35 % |
| NaCl | 216 | research engine ×2 | 1 | 3.562–3.565 | 0.9990 | 0.9977–1.0006 | 0.9897–1.0082 | 5.7–6.4 % | – |
| NbCl5 | 1944 | FastMACE ×2 | 3 | 27.028–27.141 | 0.9958–0.9994 | 0.9928–1.0015 | 0.9919–1.0091 | 3.5–4.3 % | 0.30 % |
| NbCl5 | 1944 | research engine ×2 | 1 | 27.085–27.139 | 0.9980 | 0.9972–0.9992 | 0.9965–1.0001 | 3.6–3.7 % | – |

**The local floor for a 12-block, same-process A/B is 1.1 % on NaCl and
0.4 % on NbCl5.** These are the largest deviations of B/A from 1 over the
FastMACE processes. In 4-block windows (the SPEC §12.5 protocol) the floor
is 0.7 % on both crystals. That is about seven times murphy's 0.16 % on
NaCl, and three times on NbCl5. A local A/B effect smaller than these
figures cannot be told apart from noise. Smaller effects need murphy when
it is idle, or more processes.

**This floor applies to every same-process A/B tool on this machine:**
`tools/ab_step_time.py`, `tools/ab_fastmace.py` (1b against 1a) and
`tools/ab_frontends.py`. The conductor decided this (2-T12, answer 4).
`ab_fastmace.py` cannot take one version twice, so no separate floor was
measured with it.

Columns, as in 1a-T18's report:

- step times are the side medians;
- the spread is (max − min)/median of one side's blocks in one process;
- "B/A, 12 blocks" is the ratio of the side medians per process;
- the 4-block windows split each process's 12 block pairs into three runs
  of 4 pairs;
- the last column is the spread of side A's median over the three
  FastMACE processes.

## What the numbers show

- **Each process carries an offset that alternation does not remove.**
  Within one process the three 4-block windows mostly share a sign:
  - NaCl: p1 1.0059 / 1.0025 / 1.0046; p2 (ABBA) 1.0037 / 1.0047 / 1.0071;
    p3 0.9960 / 0.9984 / 1.0040;
  - NbCl5: p1 all below 1, p2 all below 1, p3 mixed.

  So most of the 12-block deviation is a fixed per-process offset between
  two identical engines, not block-to-block jitter. On NaCl the side built
  second was slower in all three FastMACE processes (by 0.02–1.1 %). On
  NbCl5 it was faster in all four processes, the research engine's
  included. murphy showed the same kind of offset on NaNbO3 (1a-T18), but
  only 0.07–0.16 %. Locally it is up to seven times larger.
- **What this means for local A/B work.** Run several processes and swap
  the build order between them, as 1b-T23 did. Then judge the mean against
  the floor, not one process.
- **NaCl is the noisy case.** NaCl has 216 atoms at 3.5 ms per step, so
  the step is short enough for host-side jitter to matter. On a Windows
  desktop under WSL2 that jitter is larger than on a dedicated Linux
  workstation.
- **Blocks drift, ratios mostly do not.** As on murphy, the NaCl blocks
  slow down as the system heats from 0 to 300 K (p1 side A: 3.455 ms in the
  first block, 3.58–3.69 ms later), which gives the 5–7 % block spread.
  Both sides drift together.
- **The research engine looks quieter, but that rests on one process.** Its
  12-block ratios (0.10 % NaCl, 0.20 % NbCl5) lie inside the FastMACE range.
  A single process cannot show the process-to-process offset, so it does not
  set a lower floor.
- **1b-T23 in this light.** 1b-T23's local 1b/1a ratios were NaCl
  1.0006–1.0044 and NbCl5 0.9982–1.0015. Both lie within this floor
  (1.1 % and 0.4 %). Locally, then, there is no measurable regression on
  either crystal.

## Validity

- **Same trajectory.** After 300 + 12 × 300 steps the two sides'
  temperatures differ by at most 7.8e-5 K and their potential energies by at
  most 1.0e-6 eV/atom (research engine, NbCl5). Both sides made the same
  number of neighbour rebuilds (FastMACE: NaCl 154, NbCl5 149; research
  engine 155 and 150) and used the same edge buckets (NaCl 7168/7680, NbCl5
  51200/53760/56320).
- **Nothing compiled during timing** (`compile_during_timing.valid`, no
  recompiles). Dynamo counters were the same in every FastMACE process (5
  and 9 unique graphs after NaCl and NbCl5).
- **No other GPU process.** WSL's `nvidia-smi` listed no compute process
  before any process (`idle-checks.log`) or during timing
  (`other_gpu_processes` empty).
- **Desktop idle.** Windows' `nvidia-smi.exe` shows desktop programs too.
  Before every process it reported 0 % utilization (at most 2 % in WSL's
  view) and 4432–4437 MiB in use. That is the desktop's usual footprint,
  unchanged across the 24 minutes, so no desktop program started on the GPU.
  During the runs the monitor (`monitor.log`, every 10 s) saw utilization
  and memory only from our own processes. At least 9.7 GB of WSL memory
  stayed available.
- **Machine reserved.** The conductor kept all other local suites and GPU
  work off the machine for the run.

**One oddity: the WSL load average.** It stood at about 326 during the whole
run, and before it. Yet the CPU was 100 % idle, and in this distribution's
process list no task was running or in uninterruptible sleep (28 tasks out
of about 1080 kernel threads). The load therefore comes from tasks outside
this distribution's view, in the WSL VM's kernel. It took no CPU time (CPU
idle before each process), but its cause is unknown. Windows' own CPU load
was 15 % before the run (the desktop's programs). A reboot of WSL
(`wsl --shutdown`) before the next timing campaign would show whether the
offsets above shrink. That is a follow-up for the owner, not done here.

## Configuration

| Item | Value |
| --- | --- |
| Machine | PC, WSL2 Ubuntu (24 logical CPUs, 15.6 GB RAM), NVIDIA GeForce RTX 5060 Ti 16 GB, driver 610.88 |
| Environment | `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130), FastMACE from the checkout via `PYTHONPATH` |
| Backend, precision | cuEq/OEq hybrid, float32, TF32 off |
| Model | `mh1-r2scan` (MH-1, head `matpes_r2scan`), `~/models/mace-mh-1.model` |
| FastMACE sides | `GraphEngine` through the fixed-plan planner (`ab_step_time.py --sides fastmace fastmace`), FastMACE's in-process OEq runtime |
| Research-engine sides | `GPULangevin`, `main.py`'s gpu default, launched through the export's `oeq_no_nvcc.sh` |
| Crystals | NaCl and NbCl5 r²SCAN benchmark CIFs, 3×3×3, NVT 300 K, seed 1 |
| Protocol | 300 warm-up steps per side, then 12 timed blocks of 300 steps per side, alternating |
| Processes | p1 FastMACE ABAB, p2 FastMACE ABBA, p3 FastMACE ABAB, p4 research engine ABAB, one after another, 23:13–23:37 UTC, 19 October 2026 (≈ 6 min per FastMACE process, 7 min for p4) |
| Settings | `TORCHINDUCTOR_COMPILE_THREADS=4`, caches under `~/fastmace-cache/2-T12` (empty at the start), launched detached with `setsid -f` |
| FastMACE commit | `b0f24d5` (main). `fastmace_commit` is `null` in the metadata, because WSL's git cannot resolve this worktree's Windows `.git` path |
| Baseline | ResearchModule tag `research-engine-baseline-20261017`, commit `d2faa9fd`, export `~/fastmace-baseline/1a-T18` |

Setup times are in `summary.json`. As on murphy, they are not a fair
comparison: the side built first also pays for the first compilation.

## Reproducing

```bash
setsid -f bash results/2-T12-local-noise-floor-20261020/run.sh floor ~/fastmace-acceptance/<run-id> >/dev/null 2>&1
python results/2-T12-local-noise-floor-20261020/summarize.py <directory with the nf-*/metadata.json>
```

`run.sh smoke <dir>` runs one short NaCl A/B first (20 steps, 2 blocks) to
check that the FastMACE side builds. `run.sh` holds the checkout path of
this worktree; change `FM` for another checkout. Run it with the desktop
idle, with nothing else on the machine, and with at least 10 GB of WSL
memory available. It takes about 25 minutes.

## Data

`results/2-T12-local-noise-floor-20261020/`:

- `nf-fm-p1-abab/`, `nf-fm-p2-abba/`, `nf-fm-p3-abab/` and
  `nf-base-p4-abab/metadata.json`: every block time, observables, engine
  info, dynamo counters and provenance;
- `smoke/metadata.json`: the smoke run;
- `summary.json`: the derived statistics above (`summarize.py`);
- `idle-checks.log`, `monitor.log` and `run.log`: the idle record and the
  timeline;
- `run.sh`: the launcher.
