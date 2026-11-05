# 1b-T28: the software block once per process

1b-T21b found that `Simulation.run` rebuilt the whole `metadata.json` record at the end of every
call, and that `provenance.fastmace_block` cost about 0.13 s per call from ext4 and up to about 13 s
from the `/mnt/d` checkout. Almost all of that is `git_commit`, which ran `git rev-parse` and
`git status` over WSL's 9P file system.

## Change

- `fastmace_block(device)` builds its static part once per process and device: the git commit and
  dirtiness, the kernel probes (Triton, NVRTC, streaming), the package versions, and the GPU name
  and capability. The patches in force and the runtime are still read at every call, and every
  call returns a new dictionary. Keys, order and meaning are unchanged, except that `git_dirty` may
  now be `"unknown"`.
- `git_commit` reads `HEAD` from the `.git` files without starting git (`head_commit`). It handles
  loose and packed refs, a detached `HEAD`, and a worktree's `.git` file with `commondir`. A Windows
  `gitdir: D:/...` maps to `/mnt/d/...` under WSL, so worktrees made by Windows git now record their
  commit (they recorded `None` before). A copy without `.git` (the NAS copy) gives its
  `.source_commit`. `git status` runs once per process and checkout with a 2 s timeout, and records
  `"unknown"` when the timeout runs out. It never runs in the step loop.
- `Simulation` builds the block during setup, as the setup stage `provenance` (0.07 s locally), so
  its cost shows in the report's setup time. The per-call `metadata.json` write (atomic, with fsync)
  stays, so `sim.metadata()` and `RunResult.metadata` stay current.

Measured on the local PC under WSL, `fastmace_cpu` environment:

| Checkout | `git_commit` first call | Later calls | `git rev-parse HEAD` (Windows git) |
| --- | --- | --- | --- |
| T28 worktree on `/mnt/d` | 0.021 s, `(34bef6c…, None)` | 0.009 s | `34bef6c…` |
| main checkout on `/mnt/d` | 2.04 s (the `git status` timeout), `(34bef6c…, "unknown")` | 0.028 s | `34bef6c…` |

In the same process, `fastmace_block` took 0.07–0.14 s on its first call and 0 s on later calls.
Before this change, `git_commit` took 11.7 s per call from the main checkout (1b-T21b).

## A/B on the RTX 5060 Ti

Hardware and software: local PC (WSL2), NVIDIA GeForce RTX 5060 Ti, `mace_cueq_oeq` (torch
2.14.1+cu130), hybrid cuEq/OEq, float32.

Workload: MH-1 r²SCAN (`mh1-r2scan`), NaCl 3×3×3 (216 atoms), Langevin NVT, 300 K, 0.5 fs.

Procedure:
- `tools/ab_frontends.py --sides simulation ase --warmup 300 --block 300 --blocks 12`: 12
  alternating blocks of 300 steps in one process, with `TORCHINDUCTOR_COMPILE_THREADS=4`.
- No compute processes were on the GPU, only desktop programs.
- Both sides ran from `/mnt/d`. Before: the main checkout `/mnt/d/Margraf/FastMACE` at `34bef6c`.
  After: the T28 worktree on `/mnt/d` at `795b1e8`.
- Both runs were on 19 October 2026. The JSON is in `results/1b-T28-local-20261019-2157/`.

| Run | `Simulation` ms/step (spread) | ASE shim ms/step | Ratio ASE/`Simulation` | Setup, `Simulation` side |
| --- | --- | --- | --- | --- |
| before (`34bef6c`) | 20.76 (0.34) | 21.00 | 1.012 | 38.4 s |
| after (`795b1e8`) | **3.649** (0.075) | 3.668 | 1.005 | 20.0 s (`provenance` 0.07 s) |

The before run's per-call cost was about 5 s (20.8 against 3.65 ms/step over 300 steps). 1b-T21b
measured about 13 s per call from the same checkout. The cost varies with how quickly `git status`
answers over 9P. After the change, step time is the same as from ext4 (1b-T21b: 3.62–3.67 ms/step),
and the run's `metadata.json` records the worktree's commit (`795b1e8…`) instead of `None`.
