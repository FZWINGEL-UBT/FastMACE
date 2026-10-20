# 2-T19: setup time per stage, 1b against 1a

**Status: measured, 20 October 2026.** Plan task 2-T19, branch `fastmace/2-T19`. Local RTX 5060 Ti only;
murphy's GPU was busy throughout (two of our own jobs at 100 %), so no new murphy timing. The self-check cache
is **not built**: the self-checks are no part of the gap (conductor's rule: build it only if they are material).

## Summary

| Question | Answer |
| --- | --- |
| Is 1b's setup longer than 1a's? | **Not on a process's first plan.** Warm MH-1 NaCl, both sides on the same file system: load + setup 16.74 s (1a) against 16.75 s (1b). |
| Where did 1b-T23's 4–10 s come from? | (1) **Accounting:** 1a's `load_s` includes the checkpoint conversion (3.4–4.3 s), 1b's does not (0.18 s); 1b converts in each plan's build, so its "setup" carries it. (2) **Locally, the file system:** 1b ran from `/mnt/d` (WSL 9p), the 1a export from ext4; `inspect` and `importlib.metadata` stat calls cost seconds on 9p. (3) 1b-T23's local idle check logged a **load average of 136**: its local setup numbers come from a busy machine. |
| What remains | **+2.9 s on every later plan in a process** (NbCl5 after NaCl): the per-plan reload, hash and conversion (decision 24), 3.1–3.5 s, which 1a did once per process. |
| Self-check share of the gap | **0.** Self-checks take 1.35 s (1b) and 1.43 s (1a), warm, on the first structure; 0.68 s on both on a later one; 6.2–6.4 s on both in a cold first process. |
| Largest stages | Cold: the first compiled call (30–36 s), captures (22 s), the OEq extension build (20 s). With `check="gate"`: the gate's references (24–47 s) and production half (10 s), as long as the rest of setup. |

## Method

- Probe `results/2-T19-probes/setup_stages.py` (not library code). It is 1b-T11's method: timers, CUDA-synchronised, are
  wrapped around the stages of one side's setup without changing what runs. Its `--path ab` runs exactly
  `tools/ab_fastmace.py`'s load and build of one side per process, so "load" and "setup" are 1b-T23's `load_s` and
  `setup_s`. `--path simulation` runs `fastmace.Simulation` (1b) with `check="gate"` or `"skip"`.
- MH-1 `matpes_r2scan`, NaCl then NbCl5 3×3×3, hybrid float32 NVT, 300 warm-up steps. 1a is the release at tag
  `fastmace-1a` (bb35363), exported as `fastmace_1a` (tree sha256 `b817ed07…`, the same as 1b-T23's). 1b is main at
  `6a005bb`.
- One version per process, under `tools/ci/local_gpu.py run`, on an otherwise idle PC (the conductor kept the other
  workers off it). Every process's machine state is in `<name>.idle.txt`: load average 0.6–2.3, no other GPU
  process.
- **Warm:** one cache root (Inductor, Triton, OEq, compile) shared by all processes of a set, filled by one process
  per side first, then 1a 1b 1b 1a 1a 1b. **Cold:** a fresh cache root per process, 1a 1b 1b 1a.
- Two warm sets: `warm-*` (1b from the worktree on `/mnt/d`, 9p) and `ext4-*` (1b from an ext4 copy of the same
  commit, `git archive`). The 1a export is on ext4 in both.
- Values are medians of 3 processes (2 for cold, gate), with min–max. Setup differences of seconds are far above the
  local step-time noise floor (2-T12: 0.4–1.1 %), but setup's own spread is given per row.

Environment: PC (WSL2, Ubuntu), RTX 5060 Ti 16 GB, driver 610.88, `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130,
cuEquivariance 0.12.0, OEq 0.7.0, MACE 0.3.16).

## Results (seconds)

### Warm, both sides on ext4 (`ext4-*`): the like-for-like comparison

| Stage | NaCl 1a | NaCl 1b | NbCl5 1a | NbCl5 1b |
| --- | ---: | ---: | ---: | ---: |
| load (per process; 1a converts here) | 3.73 | 0.39 | – | – |
| execution model (reload, hash, conversion; 1b per plan) | – | 3.51 | – | 3.14 |
| plan | 0.00 | 0.01 | 0.00 | 0.00 |
| rewrites: apply | 0.20 | 0.20 | 0.15 | 0.14 |
| **rewrites: self-checks** | **1.43** | **1.35** | **0.68** | **0.68** |
| private copy | 0.66 | 0.52 | 0.53 | 0.52 |
| first compiled call | 10.45 | 10.39 | 12.92 | 12.79 |
| setup, rest | 0.25 | 0.33 | 0.08 | 0.07 |
| setup | 13.01 (12.86–13.14) | 16.34 (16.23–16.53) | 14.38 (14.20–14.59) | 17.28 (17.26–17.43) |
| **load + setup** | **16.74** (16.54–16.91) | **16.75** (16.52–16.91) | 14.38 | 17.28 |
| warm-up (captures) | 10.34 (9.27) | 10.37 (9.33) | 8.61 (0.58) | 8.59 (0.62) |

Load per process for 1a: torch.load plus conversion 3.40 s, sha256 0.02 s. ("load + setup" adds the process's load to
its first structure only.)

### Warm, 1b from `/mnt/d` (`warm-*`): the conditions of 1b-T23's local run

| Stage | NaCl 1a | NaCl 1b | NbCl5 1a | NbCl5 1b |
| --- | ---: | ---: | ---: | ---: |
| execution model | – | 3.47 | – | 3.26 |
| plan | 0.00 | 0.40 | 0.00 | 0.00 |
| rewrites: apply | 0.18 | 0.80 | 0.14 | 0.15 |
| rewrites: self-checks | 1.44 | 1.47 | 0.72 | 0.71 |
| first compiled call | 10.78 | 12.19 | 13.53 | 13.79 |
| setup, rest | 0.57 | 6.12 (3.75–6.40) | 0.15 | 0.40 |
| load + setup | 17.29 | 25.93 (23.14–26.27) | 15.21 | 18.78 |

The extra 1b time on NaCl sits in plan, apply, first compiled call and "rest". A finer run (`detail-*`) put the "rest"
in `Registry.resolve` (11 s in 1b, 2.6 s in 1a under the extra timers). A CPU profile of the first `resolve`
(`profile-resolve-9p.txt`, `-ext4.txt`): from `/mnt/d` it took 1.6–6.8 s, from ext4 0.16 s. The time is in
`posix.stat` and `lstat`. Two sources:

- the lazy imports of the Triton kernel modules (`fastmace/kernels/triton/fused_radial.py` through
  `rewrites/mh1._hidden_support`): `torch.library`'s custom-op registration calls `inspect.getframeinfo`, whose
  `getmodule` stats every loaded module's file;
- `installed_mace_version()`, called once per record (19 times per resolve), scans the package metadata of every
  `sys.path` entry: 0.6 s on 9p, 0.012 s on ext4.

On ext4 the two versions are equal, so this is a file-system effect of the checkout's location, not 1b's code
path. The same stats would cost time on a slow NFS checkout. On murphy, 1b-T23 ran both sides from NFS, and its load
+ setup agree (below).

### Cold, 1b from `/mnt/d` (`cold-*`, n = 2)

| Stage | NaCl 1a | NaCl 1b | NbCl5 1a | NbCl5 1b |
| --- | ---: | ---: | ---: | ---: |
| prepare_hybrid (OEq extension build, per process) | 20.06 | 19.74 | – | – |
| load | 3.62 | 0.40 | – | – |
| execution model | – | 3.59 | – | 2.85 |
| rewrites: self-checks | 6.41 | 6.22 | 0.69 | 0.70 |
| first compiled call | 30.48 | 31.99 | 35.72 | 36.15 |
| setup | 38.49 | 48.54 | 37.53 | 40.70 |
| load + setup | 42.11 | 48.94 | 37.53 | 40.70 |
| warm-up (captures) | 24.24 (23.21) | 23.42 (22.40) | 8.50 (0.50) | 8.62 (0.60) |

Cold self-checks: 6.2–6.4 s on the first structure, equal on both sides. 1b-T11 measured 3.8 s on murphy. The
first eager calls build kernels and library handles there; with the checks off, that work would move to the first
compiled call rather than disappear (1b-T11: `check=False` saved 1.2–1.3 s warm).

### `Simulation` with the gate, and MACE first (1b from `/mnt/d`, warm, NaCl)

| Run | n | plan | build | gate (references + production) | provenance | process start to built |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `check="gate"` | 2 | 0.23 | 22.52 | 45.34 (33.45–57.23): 35.22 + 10.12 | 0.39 | 83.2 (69.7–96.8) |
| `check="skip"` | 1 | 0.19 | 23.86 | – | 0.48 | 40.3 |
| MACE imported first, `check="gate"` | 1 | 0.62 | 26.29 | 30.90: 26.13 + 4.78 | 0.56 | 72.0 |

- The gate's reference half is the largest setup stage when it runs. Its spread across two processes (24–47 s)
  is larger than any other stage's.
- With MACE imported first, the plan is plain cuEq (`backend-cueq-mace-imported-first`, 1b-T23 finding 8).
  Here it was not slower than FastMACE first (72 s against 70–97 s), with n = 1 and 2. 1b-T23's 48 against 34 s
  is not reproduced as a gap. This should be re-measured with more processes if it matters.

### murphy (from 1b-T23's JSON, no new run)

1b-T23's single-version controls (`results/1b-T23-murphy-20261019/regression-control-*`), both sides from the NAS
checkout:

| | load | NaCl setup | NaCl load + setup | NbCl5 setup |
| --- | ---: | ---: | ---: | ---: |
| 1a | 4.26 | 15.30 | 19.56 | 15.53 |
| 1b | 0.18 | 19.54 | 19.72 | 16.17 |

Load + setup agree within 0.2 s on the first plan. The NbCl5 difference (+0.6 s) is smaller than locally. The same
per-plan reload applies; murphy's conversion is faster, or partly cached. Local, from the same JSON:
1a 3.54 + 14.36 = 17.90 s against 1b 0.18 + 24.01 = 24.19 s (9p and load average 136, above).

## The self-check cache

Not built, under the conductor's rule. The numbers for checkpoint 2:

- **Share of the gap: 0.** The self-checks cost the same in 1a and 1b.
- **Possible saving, independent of the gap:** about 1.4 s per process warm (8 % of a warm 16.7 s load + setup),
  0.7 s per later structure, and up to 6 s in a cold first process. Part of the cold saving would move to the first
  compiled call.
- The safety trade-off is unchanged: the checks are a function of the key only when everything that determines them
  is in the key. The structure-dependent checks (generic, `reassociate_first`) would stay uncached unless shown to be
  species-set invariant, and only passes with a margin of threshold/10 would be cached.

## Found elsewhere, reported, not fixed (conductor's Q4)

1. **Per-plan reload, hash and conversion** (decision 24): 3.1–3.5 s for every plan after the first in a process
   (NbCl5 +2.9 s warm against 1a). It also hits every revision or fallback re-plan. A per-process cache of the
   converted model, keyed by checkpoint hash, backend and dtype, with a deep copy per plan, would remove it. That is
   a design question for the instance-isolation rule (§8.7), not a T19 change.
2. **File-system-bound resolution:** the first `Registry.resolve` stats every loaded module file (custom-op
   registration through `inspect`), and `installed_mace_version()` runs once per record. 0.1 s on ext4, 1.6–6.8 s
   from WSL's 9p. Calling `installed_mace_version()` once per resolve (or once per process) is a small general fix.
   The `inspect` cost belongs to `torch.library` and would need the kernel modules imported earlier or their
   registration made cheaper.
3. **Gate references:** 24–47 s, the largest single setup stage under `check="gate"`, with a large spread.
4. **Measurement hygiene for later setup comparisons:** put both sides on the same file system and in the same
   cache state, and record the load average (1b-T23's local run: 136).

## Files

- `results/2-T19-probes/`: `setup_stages.py` (probe), `mace_first.py`, `run_setup.sh` (the run), `summarize.py`
  (tables), `profile_resolve.py`.
- `results/2-T19-local-20261020/`: `setup.json` per process, `<name>.idle.txt`, `summary.json`, `run_setup.log`,
  `ext4.sh`, the two resolve profiles.

Reproduce (WSL, idle PC; `run_setup.sh RUN_ROOT PYTHON CHECKPOINT RENAMED_DIR STRUCTURES_DIR`):

```bash
python3 tools/ci/local_gpu.py run --purpose "2-T19 setup timing" -- bash results/2-T19-probes/run_setup.sh ~/fastmace-t19/run ~/environments/mace_cueq_oeq/bin/python ~/models/mace-mh-1.model ~/fastmace-t19/versions ~/fastmace-baseline/1a-T18
```
