# Ideas queue: folded α, power-basis tabled backward, Estrin w′ and a sweep of chunks and cross-checks (6 November 2026)

The conductor's own queue from the H200 ideas plan ([plans/h200-ideas-20261105.md](../../plans/h200-ideas-20261105.md)):

| Idea | What it is |
| --- | --- |
| 34 | `FOLD_ALPHA` |
| 39 | `ESTRIN` |
| 40 | `POWER_BACKWARD` |

It also covers a sweep of the tabled kernels' chunk constants and cross-checks of switches that are on for only one
of the two Modal GPUs.

**Two switches are adopted:**
- `FOLD_ALPHA` on the H200 and the A100-SXM4-80GB;
- `POWER_BACKWARD` on the H200.

**`ESTRIN` stays off.** It is faster on the A100 alone, but `FOLD_ALPHA` alone is faster still, and the two together
gain nothing.

**Setup.**

| | |
| --- | --- |
| Workload | OMAT-0 medium, float32 hybrid, SrTiO3 10×10×10 (5,000 atoms; Benchmark_loup's cell) |
| GPUs | Modal H200 and A100-SXM4-80GB; A100 80GB PCIe landings retried |
| Profiles | `second`, `third`, `fourth`, `fzwingel` |
| Timing | same-process MD-step A/B through `tools/modal/hopper_sweep.sh` (`AB=`, b = the named constants), both build orders (ab / ba) |
| Gate | `tools/modal/const_gate.sh`: `tools/force_stress_gate.py` on SrTiO3 3×3×3, NaCl, NbCl5, default plan against `--production-constant`, in one job |

**Sources.**
- The ideas branch at:
  - `aa6dad8c` (`ESTRIN`);
  - `716699ca` (`POWER_BACKWARD`);
  - `a2bd212f` (`FOLD_ALPHA`, all A/B switches off by default).
- `61086292`, for the A/Bs against the new defaults.
- From `a2bd212f` on, jobs ran from a code-only tarball (13 MB; see "Running many Modal jobs").

## Outcome

b/a below 1 means b is faster.

| Switch | H200 MD b/a | A100-SXM4-80GB MD b/a | Gate | Verdict |
| --- | --- | --- | --- | --- |
| `FOLD_ALPHA` (34): `FACTORED_SCALE`'s per-path α folded into the w′ value table's rows at build (float64), so the kernel's terms need no α multiply and scalar paths stay scalar | 0.9963 / 0.9954 | 0.9911 / 0.9863 (slow card, 10.0 ms base), 0.9895 / 0.9893 | as the default on both (worst max-error ratio: H200 1.12 SrTiO3 stress, within the limit; A100 1.049) | **default on both** |
| `POWER_BACKWARD` (40): the tabled outer-sum backward from a power-basis copy of the Chebyshev table (`pcoef`), value and slope by dual Horner | 0.9954 / 0.9957 | 1.0020 / 0.9977 (tie) | as the default on the H200 (worst 1.016, forces) | **default on the H200** |
| `ESTRIN` (39): w and w′ by Estrin's scheme (three FMAs deep instead of five) | 1.0030 / 1.0012 | 0.9961 / 0.9963 alone; with `FOLD_ALPHA` vs neither 1.0002 / 1.0028; b = `FOLD_ALPHA` on, `ESTRIN` off vs `ESTRIN` on: 0.9967 / 0.9902, 0.9884 / 0.9907 | as the default on the A100 (worst 1.011) | **off** (`FOLD_ALPHA` is the better A100 choice) |
| `POWER_BACKWARD` + `FOLD_ALPHA` together vs neither | 0.9938 / 0.9985; at `61086292`, b = both off: 1.0073 / 1.0089 | – | – | the two add up to about 0.8 % on the H200 |

### Sweeps and cross-checks (all rejected or ties)

| A/B (b) | H200 | A100-SXM4-80GB |
| --- | --- | --- |
| tabled `CHUNK=8` | 1.0046 / 1.0004 | 1.0040 / 0.9966 (earlier batch) |
| tabled `CHUNK=32` | 1.0018 / 0.9966 | 1.0057 / 0.9980 |
| tabled `FORWARD_CHUNK=16` | 1.0003 / 1.0015 | 0.9999 / 0.9993 |
| tabled `FORWARD_CHUNK=64` | 1.0058 / 1.0060 | 1.0098 / 1.0042 |
| w′ `STAGE_DOUT_CHUNK=16` | 1.0074 / 1.0026 | 1.0082 / 1.0011 |
| `SEGMENT_ROWS` (A100-only) on the H200 | 1.0008 / 1.0008 | – |
| `BACKWARD_WALK` (A100-only) on the H200 | 1.0000 / 0.9999 | – |
| `LANE_INDICES` (H200-only) on the A100 | – | 0.9969 / 1.0001 |
| radial-table forward 8 warps (H200-only) on the A100 | – | 1.0051 / 0.9949 |

No switch changes on a cross-check: each is a tie or slower on the other GPU.

### MD step after this round

| GPU | MD step |
| --- | --- |
| H200 | about 4.92–4.94 ms (the `61086292` A/B's a side) |
| A100-SXM4-80GB | about 9.67–9.71 ms (b side of the two `FOLD_ALPHA`-vs-`ESTRIN` A/Bs) |

Both are taken from the A/B jobs, not from a separate benchmark.

## Gate details

Max absolute error of production against the float64 reference (`production_max_abs`): default (a) and b in one job.
Every check that passes for the default passes for b. NbCl5 stress fails for both, on every A100 and H200 run of this
plan; it is the A100 plan's accepted failure (`docs/current/defaults-by-gpu.md`).

| Run | Worst b / default | Largest changes |
| --- | --- | --- |
| `ESTRIN`, A100 | 1.011 (NaCl energy) | SrTiO3 stress 0.955 |
| `POWER_BACKWARD`, H200 | 1.016 (SrTiO3 forces) | NaCl energy 0.955, NbCl5 stress 0.953 |
| `FOLD_ALPHA`, H200 | 1.122 (SrTiO3 stress 2.74e-7 vs 2.44e-7; passed) | NbCl5 stress 0.972 |
| `FOLD_ALPHA`, A100 | 1.049 (SrTiO3 forces and stress) | NbCl5 stress 0.976 |

## Running many Modal jobs from WSL

Three relaunches of this batch died before the cause was found:

1. **Memory.** `modal run fastmace_modal.py::job --commit X` archives the whole commit. That is 635 MB, mostly
   `docs/design/conversations/` and `results/`. Each client then held about 9 GB, and five clients filled WSL's
   40 GB.
   - Earlier this day this showed up as `Wsl/Service/E_UNEXPECTED` crashes and an OOM kill.
   - Jobs now take `--tarball` with a code-only archive (13 MB). It drops `docs/` and `runs/`, and drops `results/`
     except `results/loup-a100-20261103/`, whose `SrTiO3.cif` the scripts read.
   - It keeps `.source_commit`, so the record still names the commit.
2. **WSL power-off.** WSL powers its VM off about 17 s after the last `wsl.exe` session ends. That kills detached
   jobs and systemd user units.
   - Batches run as system transient units (`sudo systemd-run --uid=...`).
   - A Windows-side `wsl.exe ... sleep` keeps the VM up.

## Runs (`runs/`)

Each folder is a job's `out/`. Gate folders keep logs and `metadata.json`, not the per-configuration `values/`.

| Folder | Commit | Content |
| --- | --- | --- |
| `estrin-a100`, `estrin-h200` | `aa6dad8c` | `ESTRIN` MD A/B |
| `gate-estrin-a100` | `a2bd212f` | `ESTRIN` gate |
| `power-h200`, `power-a100`, `gate-power-h200` | `a2bd212f` | `POWER_BACKWARD` MD A/B and gate |
| `fold-h200`, `fold-a100`, `fold2-a100`, `gate-fold-h200`, `gate-fold-a100` | `a2bd212f` | `FOLD_ALPHA` MD A/B and gates |
| `fold-estrin-a100`, `powerfold-h200` | `a2bd212f` | combinations |
| `foldonestrin-a100`, `foldnoestrin-a100`, `foldnoestrin2-a100`, `newoff-h200` | `61086292` | against the new defaults |
| `b5-*` | `a2bd212f` | sweeps and cross-checks |
| `tosc8-a100` | `75471662` | tabled `CHUNK=8` on the A100 (earlier batch) |
