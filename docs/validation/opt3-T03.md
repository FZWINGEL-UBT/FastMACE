# OPT3-T03: the integrator stage in four kernels (opt-3)

21 October 2026. Branch `fastmace/opt3-T03` on main 62296d4 (round 3 baseline 62aa6d4 plus the plan).
Local RTX 5060 Ti (WSL2), `mace_cueq_oeq` (torch 2.14.1+cu130, Triton 3.8.0), MH-1 r²SCAN, NaCl 3×3×3
and NbCl5 3×3×3; Modal L4 below. Results: `results/opt3-T03-local-20261021/`.

## What it does

`EngineConfig.fused_integrator`, on by default since the owner's decision of 21 October 2026 (branch
`fastmace/opt3-T03-default`; SPEC fourteenth amendment); it was opt-in when measured here. On a fixed-cell CUDA `Langevin` engine the
AC stage (and the pipelined ACP) runs as four Triton kernels instead of about 45 PyTorch kernels:

1. noise terms `rnd_pos`, `rnd_vel` and `rnd_vel * m` per element; clears the displacement maximum;
2. (PyTorch: the two centre-of-mass sums, `fixcm` only);
3. per atom: centre-of-mass correction, half kick, drift, `x_old`, the new velocity, the saved velocity
   (ACP) and the squared displacement from the reference positions into an exact atomic maximum;
4. per candidate pair: the cutoff test (then PyTorch's cumsum);
5. per candidate pair: the stable compaction (destinations from the one cumsum), `count`, the rebuild flag,
   `_flags` and, pipelined, `_pipe_flags` with the hard and early limits.

Anything else (NPT/`LangevinBAOAB`, CPU, no Triton, non-float64 positions) keeps the PyTorch stage;
`info()["fused_integrator"]` is `True` or the reason. The "C" graph after a rebuild keeps PyTorch's
compaction.

## Exactness: bitwise, not rounding class

The noise is not Philox: `Langevin.prepare_step` copies the host-drawn NumPy batch (ASE's order) into
`xi`/`eta`, and the kernels read those, so the random stream is unchanged by construction. Every float64
operation is PyTorch's, in its order. What that took (`tools/opt3_t03_order_probe.py` and a synthetic
check against `GraphEngine`'s own stage methods):

- coefficients from float64 device tensors (a Python float kernel argument is float32: 12000 of 12000
  elements differed);
- FMA contraction off (`enable_fp_fusion=False`; with it, 55 of 12000 differed);
- a tensor divided by a Python number is PyTorch's `x * (1 / s)` on CUDA (velocities `/ dt` and the
  centre-of-mass `/ n` differed in a few hundred bytes until the kernel did the same);
- `sum(dim=1)` over three float64 components is `(x + z) + y` on CUDA, for both the pair distances and
  the displacements; `x ** 2` is `x * x`;
- the two centre-of-mass sums and the cumsum stay PyTorch's own reductions.

## Tests

- CPU (`tests/cpu/test_fused_integrator.py`): the option is carried by the plan; without CUDA it is a
  no-op (pipelined and checked, the same trajectory bitwise, `info()` says why); the compaction
  identity `cumsum(1 - inside) == i + 1 - cumsum(inside)`. With `tests/cpu/test_config.py`: 42 passed.
- GPU stage level (`tests/gpu/test_fused_integrator_gpu.py`, production hybrid FP32 plan, float32 model
  shifts): after 60 steps, five noise draws, ACP and AC each run through PyTorch and fused from the
  same buffers; positions, velocities, `x_old`, the noise terms, the compacted edges and shifts, the
  count, every flag and the saved velocities are byte-identical in all 20 comparisons (`fixcm` on and
  off), with displacements set to give no flag, the early flag only and the hard flag.
- GPU trajectories (`reproducible=True`, plain cuEq float64, NaCl at 1000 K and 2 fs, 600 steps, one fresh
  process each): pipelined (162 rebuilds), checked (136), a save/resume at step 300 (83 after resume)
  and NPT (`LangevinBAOAB`, which keeps the PyTorch stage; 134). Positions and momenta bitwise equal with
  the option off and on in all four.
- ruff clean.

## Timing (local RTX 5060 Ti, noise floor 1.1 % NaCl, 0.4 % NbCl5)

`tools/ab_rewrites.py --a-engine fused_integrator=false --b-engine fused_integrator=true`, same process,
OPT-T08's records on both sides, 300 K, 300 warm-up steps, 12 blocks of 300 steps, one structure and one
build order per reservation; ratio of medians on/off:

| Structure | order ab | order ba | all 12 pair ratios |
| --- | ---: | ---: | --- |
| NaCl | **0.9758** | **0.9758** | 0.969–0.988 (ab), 0.959–0.984 (ba): every pair faster |
| NbCl5 | 0.9972 | 0.9929 | 0.990–1.007 (ab), 0.989–1.001 (ba) |

NaCl −2.4 % in both orders (estimate was ~3 %): about 0.07 ms of the 3.0 ms step. NbCl5 −0.3 to −0.7 %,
at its noise floor; its step is dominated by the force call.

## Modal L4

`tools/modal/opt3_t03.sh` from commit daf3f66, NVIDIA L4 (driver 610.57.04), the same protocol, both
orders (patterns abab and abba) in one container; labelled "Modal L4, not murphy". 20.4 min, about
$0.27 of GPU time (`results/modal-opt3-T03-L4-20261021/`).

| Structure | order ab | order ba | pair ratios |
| --- | ---: | ---: | --- |
| NaCl | **0.9807** | **0.9797** | 0.976–0.985 (ab), 0.968–0.982 (ba): every pair faster |
| NbCl5 | 0.9990 | 0.9975 | 0.986–1.009 (ab), 0.991–1.005 (ba) |

NaCl −1.9 to −2.0 % (0.08 ms of 4.2 ms), NbCl5 within noise. The same picture as locally: an exact
gain on small cells, nothing to lose on large ones.

## Recommendation

Exact and reproducible-mode bitwise, −2 to −2.4 % on NaCl on two GPUs, neutral on NbCl5: a candidate
for the default (owner decision; it would be `True` with the PyTorch stage kept for NPT and the CPU).


## The default (OPT3-T03-default, 22 October 2026 UTC)

Branch `fastmace/opt3-T03-default` from main 5c48e26, pinned `125465d`, local RTX 5060 Ti; results in
`results/opt3-T03-default/`.

- Targeted CPU tests (`-n 2`): `test_fused_integrator`, `test_config`, `test_rewrites_species`,
  `test_report`, `test_planner`, `test_overlap_rebuild`, `test_step_graph`, `test_canonical_restart`,
  `test_engine`, `test_reference_policy`, `test_versioned_registry`, `test_simulation`: 442 passed, 20
  skipped. ruff clean.
- Force gate (`tools/force_stress_gate.py --engine fastmace`, default plan, NaCl and NbCl5): **passed**.
  Ratios NaCl energy 0.05, forces 1.15, stress 0.31; NbCl5 0.19, 1.00, 0.67 (OPT-DEF-2: 0.05, 1.16, 0.35;
  0.19, 1.01, 0.65). The integrator stage does not enter the force call, so this is the expected no-change.
- `tests/gpu/test_simulation_gpu.py`: 2 passed.
- Reproducible mode, the default against `fused_integrator=False` (plain cuEq float64, NaCl at 1000 K,
  600 steps, pipelined with 162 rebuilds and checked with 136): positions and momenta bitwise equal.
