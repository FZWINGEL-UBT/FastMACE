# 1b-T14: canonical restart points on the GPU

Plan task 1b-T14 (SPEC §10.6 "Canonical restart points", §12.2 restart row).
Test: `tests/gpu/test_canonical_restart_gpu.py`. JSON:
`results/1b-T14-rtx5060ti-20261019/`.

- **Machine:** local PC, WSL2.
- **GPU:** RTX 5060 Ti, idle before the run (`idle_check.txt`). Available
  RAM never fell below 10.2 GB (`free.log`).
- **Environments:** pytest in `mace_cueq_latest`; hybrid children in
  `mace_cueq_oeq` (`FASTMACE_HYBRID_PYTHON`). Both from the checkout with
  `PYTHONPATH`. Both have torch 2.14.1+cu130 and mace-torch 0.3.16;
  cuequivariance-torch 0.12.0 and OpenEquivariance 0.7.0.
- **FastMACE:** branch `fastmace/1b-T14` at 7df77ff (`commit.txt`). No
  research-engine baseline is involved: the comparison is FastMACE against
  itself.
- **Model:** MH-1 r²SCAN (`mh1-r2scan`), NaCl 3×3×3 (216 atoms).
- **Run:** 2 fs time step, friction 0.02, the target ramped from 300 to 600 K,
  the fixed planner's plan (pipelined mode).

Each run writes a restart file at the restart cadence through
`write_engine_restart`, which makes every write a canonical point. An
interrupted run builds a fresh engine from the file at the listed steps
(`resume_engine`); the uninterrupted run keeps its engine. Both runs use the
same cadence.

## Plain cuEq float64

500 steps, a restart every 100, resumed at every restart (100, 200, 300, 400).
Positions are compared every 50 steps.

| Comparison | Max position difference |
| --- | --- |
| interrupted vs uninterrupted | 3.7e-13 Å |
| second uninterrupted run vs first (noise floor) | 3.9e-13 Å |

The limit is 1e-12 Å. Plain cuEq is not bitwise reproducible even without an
interruption, because it picks kernel variants at run time (SPEC §18). The
resumed run is as close to the uninterrupted one as two uninterrupted runs
are to each other. **Passed.**

## Hybrid (cuEq + OEq, float32)

20,000 steps, a restart every 2,000, resumed at step 10,000, seeds 20261001
and 20261101. Temperature and E_pot are sampled every 20 steps and binned by
the 50 K target.

| Difference | interrupted vs whole, seed 20261001 | interrupted vs whole, seed 20261101 | between seeds (whole runs) |
| --- | --- | --- | --- |
| mean temperature | 3.4e-5 K | 1.3e-5 K | 2.7 K |
| mean E_pot | 4.3e-6 meV/atom | 2.7e-6 meV/atom | 0.48 meV/atom |
| E_pot bin means, RMS | 3.4e-5 meV/atom | 1.8e-5 meV/atom | 3.5 meV/atom |
| E_pot bin means, max | 8.7e-5 meV/atom | 4.0e-5 meV/atom | 8.6 meV/atom |

The criterion, as in 1a-T17: a resumed run differs from the uninterrupted one
by no more than twice the seed-to-seed difference. The resumed runs are about
five orders of magnitude closer than that. Over 20,000 steps the float32
trajectories have not yet separated much from their uninterrupted
counterparts, which makes the hybrid check stricter than a purely statistical
one. **Passed.**
