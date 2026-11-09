# dm_qe transfer labs, 9 November 2026

See [the assessment](../../docs/validation/speed/dmqe-transfer-labs-20261109.md) for methods, numbers and limits.
Production and baseline code are FastMACE `a7a8211e`; all candidates live in new standalone lab tools. No default
or production implementation changed. Branch: `fastmace/dm-qe-ideas-20261109`.

## Valid measurements

| Evidence | Scope |
| --- | --- |
| `leto-outer-final-0/` | Ada, ideal SrTiO3, shared and register forward / butterfly backward; validated graph replay |
| `leto-outer-final-0.07/` | Ada, jitter 0.07 A; includes exact generated CUDA source and NVRTC resource log |
| `local-outer-final/lab/` | RTX 5060 Ti, jitter 0.07 A, same lab and parity checks |
| `murphy-species/lab/species_plan.json` | Isolated static plan, current stable sort and counting sort, unchanged forward/input-gradient consumers |
| `local-species/lab/species_plan.json` | Local isolated plan, both capture orders, byte-identical outputs and lifecycle checks |
| `leto-force/species_force.json` | Full hybrid OMAT-0 force call including stress, 1,080 atoms, ideal/rattle 0.05 A, independent models, both build orders |
| `local-geometry-final/lab/dmqe_geometry_lab.json` | Compiled installed MACE ZBL versus analytical value/derivative; energy-preserving and force-only graph variants |
| `cpu-geometry/dmqe_geometry_lab.json` | Small float64/autograd and float32 output-contract smoke check with non-unit upstream |
| `local-hybrid-profile/breakdown/` | Current float32 hybrid SrTiO3 5,000-atom profile, 24.78 ms per MD step |
| `provenance.json` | Production/baseline commit, model SHA, environment scope, final tool hashes |

Raw full-force outputs are not bitwise because unchanged graph replays are nondeterministic. The isolated plan's
outputs and input gradients are byte-identical. No full-model `force_stress_gate.py` or drift qualification was
performed for the new rounding kernels. H200/A100 timings were not run or inferred from these GPUs.

Some early raw JSON records say `fastmace_commit: unknown`/null: WSL cannot follow this Windows worktree's native
git pointer. They were run from an archive of `a7a8211e` plus the standalone lab file. The separate provenance is
supplied without rewriting those raw records. Later NAS runs use the archive's `.source_commit` marker.

## Discarded or diagnostic attempts

- `leto-outer-0/`, `leto-outer-0.07/`: invalid first harness. Int64 pair IDs were read as int32 by CUDA and a stream
  handle was cached before capture. Their apparent timings are invalid. `leto-outer-v1-source.py` is preserved.
- `failures/leto-outer-v2.log`: the ordinary layout passed but the JOINED backward test omitted its layout argument.
- `failures/leto-outer-v3.log`: replay validation broadcast an `[E,1]` density against `[E]`, requesting a huge
  temporary. Size-checked flattened comparisons fix the harness. No capacity result is implied by this failure.
- `leto-outer-v4-*.json`: valid shared-accumulator controls, superseded by the final combined shared/register lab.
- `local-geometry/`, `local-geometry-v2/`: numerical checks passed; graph timing failed on donated-buffer/stale-stream
  contracts. `local-geometry-v3/` is a valid force-only precursor; the final run adds retained-energy timing and
  alternating replay order. Use the final result for conclusions.
- `local-profile/`: plain cuEq float64 control; excluded from the float32 hybrid comparison.
- `local-hybrid-profile/launch-failed-path.log`: first system-unit launch lacked WSL's PATH; corrected before the
  valid profile. Subsequent unit launchers preserve PATH and HOME.
- `local-force/run.log`: current 5060 Ti defaults do not use `COMPOSE_SKIP`, so the tool refused a misleading test.
- `failures/murphy-force-idle.json`: first guard conservatively rejected a usage-reporter heartbeat. The background
  command was inspected; a longer sample distinguishes its idle reporting from a computational job.
- `failures/murphy-force-v2-idle.json`: another user's CPU/GPU job started; the guard skipped our force test. Leto
  was checked independently and used instead. Coraline was occupied and no experiment ran there.

## Reproduction

Use approved environments and the repository GPU reservation/idle checks. The tools do not install packages.
From the checkout, representative commands (all output directories must be fresh where the tool requires it):

```bash
PYTHONPATH=. python tools/dmqe_outer_lab.py --jitter 0.07 --rounds 31 --output OUT/outer
PYTHONPATH=. python tools/dmqe_species_lab.py --counts 1000,1000,3000 --input-width 192 \
  --channels 128 --sizes 1,3,5,7 --blocks 10 --rounds 100 --output OUT/species
PYTHONPATH=. python tools/dmqe_species_force_lab.py --checkpoint CHECKPOINT --model omat0-medium \
  --structure results/loup-a100-20261103/SrTiO3.cif@6x6x6 --stress --output OUT/force
PYTHONPATH=. python tools/dmqe_geometry_lab.py --device cuda --cells 10 --rounds 31 \
  --upstream-scale 0.37 --output OUT/geometry
```

The full-force lab is fixed-species only. Its source-checked private method replacement is an experimental
mechanism, not a production cache. A production implementation must refresh derived buffers capturably and in
place before MC trials and on acceptance/rejection, restart and other lifecycle changes.

Verification: eight focused CPU invalidation tests pass; all six new Python files pass targeted Ruff; numerical
and captured-output parity checks pass in the final GPU lab runs. No full CPU suite is needed for these new lab
tools. All NAS evidence remains under `/mnt/homes/zwingel/fm-dmqe-20261109`.
