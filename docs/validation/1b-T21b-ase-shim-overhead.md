# 1b-T21b: the ASE shim's cost per `run()` call

1b-T23 (acceptance item 4b) measured `fastmace.ase.Langevin` about 0.26 s slower per `run()` call
than `Simulation.run` (+20 % in 300-step blocks), with equal per-step time.

## Where the time went

CPU profile (tiny MH-1-family model, 10 calls of `run(1)`):

| Part | Per call | Needed per call? |
| --- | --- | --- |
| a second `metadata.json` record (`Simulation.metadata()` for the shim's `"ase"` block) | ≈0.25 s, almost all in `provenance.fastmace_block`: `importlib.metadata` version scans, `git_commit`, the NVRTC probe | no |
| `metadata.json` write and `report.txt` | a few ms | yes (end of each `run`) |
| `committed()`: final state into `atoms` | one device-to-host copy; the `Simulation` has already synchronised | yes: ASE leaves `atoms` at the final step |
| chunk end, report update, observer sync, sink drain | 0 without observers; the files go through `Simulation.add_sink` since 1b-T21 | — |

**Fix** (`fastmace/ase/md.py`, no change to `simulation.py`): the shim adds its `"ase"` block to the
`metadata.json` the `Simulation` has just written (read, merge, atomic write), and `dyn.metadata`
reads that file, as `fm.md`'s `RunResult.metadata` does. CPU test
`test_a_run_call_costs_what_simulation_run_costs`: per call the shim builds as many software blocks
as `Simulation.run` (one), writes `metadata.json` once more and reads `committed()` once.

## A/B on the RTX 5060 Ti

Local PC (WSL2), NVIDIA GeForce RTX 5060 Ti, `mace_cueq_oeq` (torch 2.14.1+cu130), hybrid
cuEq/OEq, float32, MH-1 r²SCAN (`mh1-r2scan`), NaCl 3×3×3 (216 atoms), Langevin NVT 300 K,
0.5 fs. `tools/ab_frontends.py --sides simulation ase --warmup 300 --block 300 --blocks 12`,
alternating blocks in one process, `TORCHINDUCTOR_COMPILE_THREADS=4`, GPU idle (desktop only).
Before: FastMACE `0477cc4` (main); after: `c38bb56` (this branch on that main). Runs one after another,
19 October 2026; JSON in `results/1b-T21b-local-20261019-1620/`.

| FastMACE source | Side | `Simulation` ms/step | ASE shim ms/step | Ratio | Shim extra per 300-step call |
| --- | --- | --- | --- | --- | --- |
| export on ext4 | before | 3.619 | 3.673 | 1.0149 | +16 ms |
| export on ext4 | after | 3.670 | 3.669 | 0.9998 | 0 ms |
| checkout on `/mnt/d` | before | 48.08 | 91.41 | 1.90 | +13 s |
| checkout on `/mnt/d` | after | 4.117 | 4.059 | 0.986 | −17 ms (noise) |

The shim now costs what `Simulation.run` costs. The local GPU has no noise floor of its own
(1b-T23: per-pair ratios 0.987–1.014), so differences of about 1 % are noise.

## Finding for `Simulation` (not changed here)

`Simulation.run` itself rebuilds the full metadata record at the end of every call, so it pays
`fastmace_block` once per call too: ≈0.13 s per call from the `/mnt/d` worktree (4.12 against
3.67 ms/step over 300 steps), and ≈13 s per call from the `/mnt/d/Margraf/FastMACE` checkout,
where `git_commit` succeeds and is slow over WSL's 9P file system (the before row: 48 ms/step for
`Simulation` alone). The software block does not change within a process; caching it in
`provenance` (or writing `metadata.json` only at restart points and at the end of a run) would
remove this from `Simulation`, `fm.md`, the CLI and the shim alike. Proposed as a follow-up for the
owner of `provenance.py`/`simulation.py`.
