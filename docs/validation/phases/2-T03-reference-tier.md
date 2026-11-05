# 2-T03: the reference tier for MD

Plan task 2-T03 (SPEC §8.4, §9.3 rule 1, §9.4 step 5, §12.1), 20 October 2026.

`fm.Simulation(..., mode="reference")` runs MD on the reference tier. `ReferenceForceField` evaluates
through the corrected reference calculator (`fastmace.reference.calculators`, the versioned record),
and FastMACE's own integrator, noise stream, output and restart drive it on an uncaptured, unpadded
`GraphEngine`. The plan has every rewrite off and the record's stock backend (plain cuEq, unfused
convolution, uncompiled), and it claims no compile family. The report's plan line reads
`plan: reference, plain cuEq (unfused convolution), float32, 0 rewrites on, 14 off`.

## CPU (tiny MH-1-class model, NaCl, float64)

`tests/cpu/test_reference_tier.py`, WSL `fastmace_cpu`:

| Check | Result |
| --- | --- |
| Reference tier against the graph tier with every rewrite off, NVT, 50 steps | max \|Δx\| = 4.9e-17 Å (bound 1e-10 Å) |
| The same, NPT (full-cell BAOAB), 10 steps, positions and cell | within 1e-10 Å |
| Interrupted against uninterrupted (restart every 2 steps) | bitwise equal |
| Gate in reference mode | passed |
| One evaluation against `calculators.load(...).evaluate` | bitwise equal |

## GPU (murphy)

Machine murphy, NVIDIA RTX 4000 Ada Generation, environment `mace_cueq_latest` (torch 2.14.1+cu130,
MACE 0.3.16), plain cuEq, float32, model `mh1-r2scan`, FastMACE commit dea10a4. The run was untimed:
another of our sessions had a process on the GPU, and no other user did. Raw result:
`results/2-T03-murphy-20261020/2-T03-reference-tier-mh1-nacl.json`.

MH-1 on NaCl 3×3×3 (216 atoms), NVT Langevin at 300 K, 2 fs, friction 0.02 per ASE time unit,
200 steps, the same starting velocities and seed, and each tier in a fresh process
(`tests/gpu/test_reference_tier_gpu.py`). The means are taken over steps 51 to 200.

| Quantity | Reference tier | Graph tier | Limit |
| --- | --- | --- | --- |
| Mean temperature (K) | 207.2514 | 207.2514 | difference < 15.8 K |
| Temperature spread (K) | 15.622 | 15.622 | |
| Mean potential energy (eV/atom) | −6.8609983 | −6.8609983 | difference < 1 meV/atom |
| Final positions, max \|Δx\| | 7.4e-7 Å | | |
| Gate | passed (4.2e-5 meV/atom, 2.9e-6 eV/Å) | skipped | |

The two float32 trajectories stay within 7.4e-7 Å after 200 steps, so they agree far more closely
than the statistical bound requires.

## "Fits" (§9.4 step 5)

`MinimalPlanner.reference_fits(plan, force_field, atoms, references=None)` decides by a rule, with
no table:
- **With a gate result:** the tier fits if the gate's reference ran on the plan's device on the
  production cell. It does not fit if the reference could not run, ran on another device, or ran
  on a reduced cell only.
- **Without a gate result:** one trial evaluation decides, in this process on the CPU. On CUDA it
  runs in a child process (`selfcheck.run_child`), because an out-of-memory error leaves a
  process's CUDA context unusable (2-T05). Any error, non-zero exit or kill of the child means
  "does not fit".
