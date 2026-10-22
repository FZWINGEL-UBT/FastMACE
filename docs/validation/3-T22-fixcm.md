# 3-T22: the Langevin `fixcm` projection with unequal masses, measurements

Plan task 3-T22, 22 October 2026. The derivation is in
[design/fixcm-projection.md](../design/fixcm-projection.md). The raw numbers are in
`results/3-T22/bias.json`, written by `tools/fixcm_bias.py`.

## Setup

- **Machine:** the local PC (WSL2), CPU only, environment `fastmace_cpu`, float64, 2 threads, niced. No GPU
  and no research-engine baseline were involved.
- **Integrator:** FastMACE `Langevin` at 1 fs, γ = 0.05/fs (0.02/fs in the MC check), with the starting
  velocities Maxwell–Boltzmann at the target and no centre-of-mass motion.
- **Projections:** `fixcm_projection="ase"` (today's default), `"mass_weighted"` (the new option) and
  `fixcm=False` (the control).
- **Sampling:** 2000 steps of equilibration, then every step, with batch means over 20 batches (± is one
  standard error).
- **Force fields:**
  - **springs:** a translation-invariant network of harmonic springs, k = 1 eV/Å², between the pairs within
    3.2 Å (NaCl and Li/Cs), 3.6 Å (NaNbOCl4) and 3.8 Å (NbCl5). The momentum marginal of the canonical
    distribution is Maxwell–Boltzmann for any potential, so this tests the thermostat itself;
  - **`-mace`:** a tiny random MH-1-like MACE model through the planner and the graph engine, on the
    eager CPU stages: 3-T12's setting.
- **Structures:**
  - NaCl 3×3×3 (216 atoms);
  - NbCl5 with 1944 atoms: the 36-atom primitive cell of `tests/_support/structures.py` × 3×3×6, the same
    count and composition as the 72-atom cell × 3×3×3;
  - NaNbOCl4 I4/m (Ma 2026, ResearchModule `Structures/NaMOCl4`, read only) × 3×3×3 (756 atoms);
  - Li/Cs NaCl: the cations of NaCl alternately Li and Cs, 2×1×1 (16 atoms) and 3×3×3 (216 atoms).
- **Conventions:**
  - **3N:** Σ m v² / (3 Nₛ k_B) per species, and over 3N in total. ASE's `fixcm` holds this one at the
    target;
  - **fixed centre of mass:** Σ m v² / ((3 Nₛ − 3 Mₛ/M) k_B) per species, and over 3N − 3 in total. That
    is the correct count at zero total momentum, and the one `"mass_weighted"` and `redraw_all` hold.
  - Without `fixcm` the total momentum is not fixed, so the 3N convention is the right one for that
    control.

## Summary

Each projection is read in its own convention: `ase` and off over 3N, `mass_weighted` at a fixed centre
of mass. Deviations from the target are in K.

| Case | Target | `ase`: lightest / heaviest species | off: lightest / heaviest | `mass_weighted`: lightest / heaviest |
| --- | --- | --- | --- | --- |
| Li/Cs 16, springs | 600 K | Li +233.4 / Cs −26.0 | Li +0.1 / Cs −0.7 | Li +0.1 / Cs −0.5 |
| Li/Cs 16, tiny MACE | 600 K | Li +262.7 / Cs −17.5 | Li −8.4 / Cs +8.3 | Li −7.9 / Cs +6.8 |
| Li/Cs 216, springs | 600 K | Li +16.4 / Cs −0.9 | Li −0.2 / Cs +0.8 | Li −0.1 / Cs +0.9 |
| Li/Cs 216, springs | 1500 K | Li +40.9 / Cs −2.5 | Li −0.5 / Cs +1.9 | Li −0.4 / Cs +2.0 |
| NaCl 216, springs | 600 K | Na +0.8 / Cl −1.0 | Na +0.1 / Cl −0.5 | Na +0.1 / Cl −0.5 |
| NaNbOCl4 756, springs | 600 K | O +1.1 / Nb −0.8 | O −0.2 / Nb −0.3 | O −0.1 / Nb −0.4 |
| NaNbOCl4 756, springs | 1500 K | O +2.7 / Nb −2.1 | O −0.4 / Nb −0.9 | O −0.2 / Nb −1.0 |
| NbCl5 1944, springs | 1500 K | Cl 0.0 / Nb −0.7 | Cl −0.2 / Nb −0.3 | Cl −0.2 / Nb −0.3 |

**What the numbers show:**

1. **`ase` is biased toward the light species, and the bias falls with N.**
   - Li/Cs: Li is +39 % at 16 atoms and +2.7 % at 216, at both temperatures.
   - The three production crystals at 3×3×3 show at most +0.2 % (O in NaNbOCl4 at 1500 K, 1.4σ) and
     are otherwise within the noise.
   - The total stays at the target over 3N only where the masses are close. Li/Cs 216 reads +0.8 %
     total.
2. **`mass_weighted` puts every species at the target at fixed centre of mass**, at every size and both
   temperatures, and agrees with `fixcm=False` species by species.
   - With the tiny MACE model, `mass_weighted` and off share the same small Li-low, Cs-high pattern
     (Li −8 K, Cs +7 K, 2σ each). The projection doesn't cause it; I read it as the scheme's own
     time-step error with the light Li at 1 fs.
3. **The conventions differ by N/(N−1).** At equal masses `ase` holds 3N kT, so over 3N − 3 it reads
   N/(N−1) high: +0.46 % at 216 atoms, +0.13 % at 756, +0.05 % at 1944.
4. **The MC check** (3-T12's addendum): (Na,K)Cl 16 atoms, tiny MACE, 2000 K, γ = 0.02/fs, Na↔K swaps every
   10 fs with `redraw_all`, real Metropolis decisions, 85 % accepted.
   - With `ase`, MC lowers the kinetic temperature by 4.7 %, 1896.9 ± 5.3 K against 1990.3 ± 15.0 K over
     3N (5.9σ): every accepted block draws (3N − 3) kT, and the thermostat holds 3N kT.
   - With `mass_weighted`, MC on and off agree: 1999.9 ± 5.5 K and 1986.2 ± 13.8 K over 3N − 3 (1.0σ),
     both at the 2000 K target.

## Research engine and ASE

- **Research engine:** the frozen engine uses the same projection:
  - `gpu_langevin.py`, `GPULangevin._stage_a`, at ResearchModule HEAD d37df508;
  - its CPU path runs ASE's `Langevin(fixcm=True)`, so it has the same projection too.

  Every research `fixcm` run carries the bias above.
- **ASE:** ASE 3.29's `Langevin` is the same.

## Recommendation (owner checkpoint 2b)

1. **MC with `redraw_all`: use `"mass_weighted"` now.** The planner should choose it whenever a move set
   uses `redraw_all` under `fixcm`. This is a general rule: the two conventions must agree. Otherwise every
   accepted block takes a 1/N fraction of the kinetic energy away.
2. **The MD default: `"mass_weighted"` is correct for every system, but it should not become the default
   yet.** The fused Triton stage, part of the round-3 speed-up, implements only ASE's projection, so today
   switching would turn the fused stage off. Teaching the kernel the mass-weighted means is a small change
   (two sums and no factor). After that, switching the default costs nothing.
3. **What a default switch changes:**
   - ASE noise-stream equivalence (`"ase"` stays for those tests);
   - A/B trajectories against the research engine, which stay comparable only statistically;
   - the reported temperature at equal masses, by N/(N−1) under the 3N count. The temperature output
     should then count 3N − 3 degrees of freedom.
4. **Until then, production MD with `"ase"` is acceptable.** On the three production crystals at 3×3×3 the
   bias is at most 0.2 % per species. Small cells with a large mass contrast (Li with Cs, or H) are not
   acceptable with `"ase"`.

## Measurements

The tables below come from `results/3-T22/bias.json` (per species, mean ± standard error, K).

#### 3N convention

| Case | Target | Projection | Steps | Total | Per species (K) |
| --- | --- | --- | --- | --- | --- |
| LiCs-16 | 600K | `ase` | 937,500 | 669.8 ± 0.7 | Li 833.4 ± 1.8; Cl 635.9 ± 0.9; Cs 574.0 ± 0.8 |
| LiCs-16 | 600K | off | 937,500 | 600.1 ± 0.5 | Li 600.1 ± 1.3; Cl 600.5 ± 0.9; Cs 599.3 ± 0.8 |
| LiCs-16 | 600K | `mass_weighted` | 937,500 | 562.6 ± 0.5 | Li 595.2 ± 1.2; Cl 575.1 ± 0.9; Cs 505.0 ± 0.7 |
| LiCs-16 | 1500K | `ase` | 937,500 | 1674.6 ± 1.8 | Li 2085.4 ± 4.5; Cl 1588.5 ± 2.3; Cs 1435.9 ± 2.0 |
| LiCs-16 | 1500K | off | 937,500 | 1500.4 ± 1.3 | Li 1500.5 ± 3.2; Cl 1501.3 ± 2.2; Cs 1498.5 ± 2.1 |
| LiCs-16 | 1500K | `mass_weighted` | 937,500 | 1406.6 ± 1.4 | Li 1488.2 ± 3.1; Cl 1437.8 ± 2.1; Cs 1262.5 ± 1.9 |
| LiCs-216 | 600K | `ase` | 69,440 | 604.7 ± 0.5 | Li 616.4 ± 1.0; Cl 601.7 ± 0.7; Cs 599.1 ± 1.3 |
| LiCs-216 | 600K | off | 69,440 | 599.8 ± 0.5 | Li 599.8 ± 1.0; Cl 599.3 ± 0.7; Cs 600.8 ± 1.3 |
| LiCs-216 | 600K | `mass_weighted` | 69,440 | 597.1 ± 0.5 | Li 599.5 ± 1.0; Cl 597.4 ± 0.7; Cs 593.9 ± 1.3 |
| LiCs-216 | 1500K | `ase` | 69,440 | 1511.7 ± 1.2 | Li 1540.9 ± 2.7; Cl 1504.2 ± 1.7; Cs 1497.5 ± 3.3 |
| LiCs-216 | 1500K | off | 69,440 | 1499.5 ± 1.3 | Li 1499.5 ± 2.6; Cl 1498.2 ± 1.8; Cs 1501.9 ± 3.3 |
| LiCs-216 | 1500K | `mass_weighted` | 69,440 | 1492.6 ± 1.3 | Li 1498.7 ± 2.6; Cl 1493.5 ± 1.7; Cs 1484.5 ± 3.3 |
| NaCl-216 | 600K | `ase` | 69,440 | 599.9 ± 0.5 | Na 600.8 ± 0.7; Cl 599.0 ± 0.7 |
| NaCl-216 | 600K | off | 69,440 | 599.8 ± 0.5 | Na 600.1 ± 0.7; Cl 599.5 ± 0.8 |
| NaCl-216 | 600K | `mass_weighted` | 69,440 | 597.0 ± 0.5 | Na 597.9 ± 0.7; Cl 596.1 ± 0.7 |
| NaCl-216 | 1500K | `ase` | 69,440 | 1499.7 ± 1.3 | Na 1502.0 ± 1.6; Cl 1497.4 ± 1.9 |
| NaCl-216 | 1500K | off | 69,440 | 1499.3 ± 1.3 | Na 1500.0 ± 1.7; Cl 1498.6 ± 1.9 |
| NaCl-216 | 1500K | `mass_weighted` | 69,440 | 1492.4 ± 1.3 | Na 1494.6 ± 1.6; Cl 1490.2 ± 1.9 |
| NaNbOCl4-756 | 600K | `ase` | 60,000 | 600.4 ± 0.2 | O 601.1 ± 0.8; Na 599.7 ± 0.6; Cl 600.7 ± 0.3; Nb 599.2 ± 0.8 |
| NaNbOCl4-756 | 600K | off | 60,000 | 600.1 ± 0.2 | O 599.8 ± 0.8; Na 599.1 ± 0.6; Cl 600.5 ± 0.3; Nb 599.7 ± 0.8 |
| NaNbOCl4-756 | 600K | `mass_weighted` | 60,000 | 599.3 ± 0.2 | O 599.6 ± 0.8; Na 598.7 ± 0.6; Cl 599.9 ± 0.3; Nb 597.7 ± 0.8 |
| NaNbOCl4-756 | 1500K | `ase` | 60,000 | 1500.9 ± 0.5 | O 1502.7 ± 1.9; Na 1499.1 ± 1.6; Cl 1501.7 ± 0.8; Nb 1497.9 ± 1.9 |
| NaNbOCl4-756 | 1500K | off | 60,000 | 1500.3 ± 0.5 | O 1499.6 ± 2.0; Na 1497.8 ± 1.6; Cl 1501.3 ± 0.8; Nb 1499.1 ± 1.9 |
| NaNbOCl4-756 | 1500K | `mass_weighted` | 60,000 | 1498.3 ± 0.5 | O 1499.0 ± 2.0; Na 1496.6 ± 1.6; Cl 1499.6 ± 0.8; Nb 1494.3 ± 1.9 |
| NbCl5-1944 | 600K | `ase` | 60,000 | 599.9 ± 0.2 | Cl 600.0 ± 0.3; Nb 599.6 ± 0.4 |
| NbCl5-1944 | 600K | off | 60,000 | 599.9 ± 0.2 | Cl 599.9 ± 0.3; Nb 599.7 ± 0.4 |
| NbCl5-1944 | 600K | `mass_weighted` | 60,000 | 599.6 ± 0.2 | Cl 599.7 ± 0.3; Nb 599.1 ± 0.4 |
| NbCl5-1944 | 1500K | `ase` | 60,000 | 1499.9 ± 0.6 | Cl 1500.0 ± 0.7; Nb 1499.3 ± 1.0 |
| NbCl5-1944 | 1500K | off | 60,000 | 1499.8 ± 0.6 | Cl 1499.8 ± 0.7; Nb 1499.7 ± 1.1 |
| NbCl5-1944 | 1500K | `mass_weighted` | 60,000 | 1499.0 ± 0.6 | Cl 1499.2 ± 0.7; Nb 1498.1 ± 1.0 |
| LiCs-16-mace | 600K | `ase` | 60,000 | 670.9 ± 3.0 | Li 862.7 ± 6.0; Cl 619.2 ± 4.0; Cs 582.5 ± 4.4 |
| LiCs-16-mace | 600K | off | 60,000 | 599.5 ± 2.1 | Li 591.6 ± 4.4; Cl 598.9 ± 3.8; Cs 608.3 ± 4.5 |
| LiCs-16-mace | 600K | `mass_weighted` | 60,000 | 561.2 ± 2.1 | Li 587.2 ± 4.3; Cl 573.2 ± 3.9; Cs 511.1 ± 3.4 |

#### fixed-centre-of-mass convention

| Case | Target | Projection | Steps | Total | Per species (K) |
| --- | --- | --- | --- | --- | --- |
| LiCs-16 | 600K | `ase` | 937,500 | 714.4 ± 0.8 | Li 840.3 ± 1.8; Cl 663.8 ± 0.9; Cs 681.4 ± 0.9 |
| LiCs-16 | 600K | off | 937,500 | 640.1 ± 0.6 | Li 605.1 ± 1.3; Cl 626.9 ± 0.9; Cs 711.5 ± 1.0 |
| LiCs-16 | 600K | `mass_weighted` | 937,500 | 600.1 ± 0.6 | Li 600.1 ± 1.2; Cl 600.4 ± 0.9; Cs 599.5 ± 0.9 |
| LiCs-16 | 1500K | `ase` | 937,500 | 1786.2 ± 1.9 | Li 2102.8 ± 4.6; Cl 1658.2 ± 2.4; Cs 1704.7 ± 2.4 |
| LiCs-16 | 1500K | off | 937,500 | 1600.4 ± 1.4 | Li 1512.9 ± 3.2; Cl 1567.2 ± 2.3; Cs 1779.0 ± 2.5 |
| LiCs-16 | 1500K | `mass_weighted` | 937,500 | 1500.4 ± 1.5 | Li 1500.6 ± 3.1; Cl 1501.0 ± 2.2; Cs 1498.8 ± 2.2 |
| LiCs-216 | 600K | `ase` | 69,440 | 607.6 ± 0.5 | Li 616.7 ± 1.0; Cl 603.6 ± 0.7; Cs 606.2 ± 1.4 |
| LiCs-216 | 600K | off | 69,440 | 602.6 ± 0.5 | Li 600.2 ± 1.0; Cl 601.2 ± 0.7; Cs 607.9 ± 1.4 |
| LiCs-216 | 600K | `mass_weighted` | 69,440 | 599.8 ± 0.5 | Li 599.9 ± 1.0; Cl 599.3 ± 0.7; Cs 600.9 ± 1.3 |
| LiCs-216 | 1500K | `ase` | 69,440 | 1518.7 ± 1.3 | Li 1541.9 ± 2.7; Cl 1508.9 ± 1.7; Cs 1515.2 ± 3.4 |
| LiCs-216 | 1500K | off | 69,440 | 1506.4 ± 1.3 | Li 1500.5 ± 2.6; Cl 1502.9 ± 1.8; Cs 1519.6 ± 3.3 |
| LiCs-216 | 1500K | `mass_weighted` | 69,440 | 1499.5 ± 1.3 | Li 1499.6 ± 2.6; Cl 1498.2 ± 1.7; Cs 1502.0 ± 3.3 |
| NaCl-216 | 600K | `ase` | 69,440 | 602.7 ± 0.5 | Na 603.0 ± 0.7; Cl 602.4 ± 0.8 |
| NaCl-216 | 600K | off | 69,440 | 602.6 ± 0.5 | Na 602.2 ± 0.7; Cl 602.9 ± 0.8 |
| NaCl-216 | 600K | `mass_weighted` | 69,440 | 599.8 ± 0.5 | Na 600.1 ± 0.7; Cl 599.5 ± 0.7 |
| NaCl-216 | 1500K | `ase` | 69,440 | 1506.7 ± 1.3 | Na 1507.4 ± 1.6; Cl 1505.9 ± 1.9 |
| NaCl-216 | 1500K | off | 69,440 | 1506.3 ± 1.3 | Na 1505.5 ± 1.7; Cl 1507.1 ± 1.9 |
| NaCl-216 | 1500K | `mass_weighted` | 69,440 | 1499.3 ± 1.3 | Na 1500.1 ± 1.6; Cl 1498.6 ± 1.9 |
| NaNbOCl4-756 | 600K | `ase` | 60,000 | 601.2 ± 0.2 | O 601.4 ± 0.8; Na 600.1 ± 0.6; Cl 601.4 ± 0.3; Nb 601.1 ± 0.8 |
| NaNbOCl4-756 | 600K | off | 60,000 | 600.9 ± 0.2 | O 600.2 ± 0.8; Na 599.6 ± 0.6; Cl 601.3 ± 0.3; Nb 601.5 ± 0.8 |
| NaNbOCl4-756 | 600K | `mass_weighted` | 60,000 | 600.1 ± 0.2 | O 599.9 ± 0.8; Na 599.1 ± 0.6; Cl 600.6 ± 0.3; Nb 599.6 ± 0.8 |
| NaNbOCl4-756 | 1500K | `ase` | 60,000 | 1502.9 ± 0.5 | O 1503.5 ± 1.9; Na 1500.3 ± 1.6; Cl 1503.5 ± 0.8; Nb 1502.6 ± 1.9 |
| NaNbOCl4-756 | 1500K | off | 60,000 | 1502.2 ± 0.5 | O 1500.4 ± 2.0; Na 1498.9 ± 1.6; Cl 1503.1 ± 0.8; Nb 1503.8 ± 1.9 |
| NaNbOCl4-756 | 1500K | `mass_weighted` | 60,000 | 1500.3 ± 0.5 | O 1499.8 ± 2.0; Na 1497.8 ± 1.6; Cl 1501.4 ± 0.8; Nb 1499.0 ± 1.9 |
| NbCl5-1944 | 600K | `ase` | 60,000 | 600.3 ± 0.2 | Cl 600.3 ± 0.3; Nb 600.2 ± 0.4 |
| NbCl5-1944 | 600K | off | 60,000 | 600.2 ± 0.2 | Cl 600.2 ± 0.3; Nb 600.4 ± 0.4 |
| NbCl5-1944 | 600K | `mass_weighted` | 60,000 | 599.9 ± 0.2 | Cl 599.9 ± 0.3; Nb 599.8 ± 0.4 |
| NbCl5-1944 | 1500K | `ase` | 60,000 | 1500.6 ± 0.6 | Cl 1500.6 ± 0.7; Nb 1500.9 ± 1.0 |
| NbCl5-1944 | 1500K | off | 60,000 | 1500.5 ± 0.6 | Cl 1500.4 ± 0.7; Nb 1501.3 ± 1.1 |
| NbCl5-1944 | 1500K | `mass_weighted` | 60,000 | 1499.8 ± 0.6 | Cl 1499.8 ± 0.7; Nb 1499.7 ± 1.0 |
| LiCs-16-mace | 600K | `ase` | 60,000 | 715.6 ± 3.2 | Li 869.8 ± 6.1; Cl 646.3 ± 4.2; Cs 691.6 ± 5.2 |
| LiCs-16-mace | 600K | off | 60,000 | 639.4 ± 2.3 | Li 596.5 ± 4.4; Cl 625.2 ± 3.9; Cs 722.2 ± 5.3 |
| LiCs-16-mace | 600K | `mass_weighted` | 60,000 | 598.6 ± 2.2 | Li 592.1 ± 4.3; Cl 598.4 ± 4.1; Cs 606.8 ± 4.0 |

#### MC on and off, both conventions

| Case | Target | Projection | MC | Steps | Total (3N) | Per species, 3N (K) |
| --- | --- | --- | --- | --- | --- | --- |
| NaK-16-mace-mc | 2000K | `ase` | mc-off | 40,000 | 1990.3 ± 15.0 | Na 2027.9 ± 23.2; Cl 1963.9 ± 24.9; K 2005.7 ± 30.2 |
| NaK-16-mace-mc | 2000K | `ase` | mc-on | 40,000 | 1896.9 ± 5.3 | Na 1936.4 ± 12.0; Cl 1892.1 ± 9.6; K 1866.9 ± 12.0 |
| NaK-16-mace-mc | 2000K | `mass_weighted` | mc-off | 40,000 | 1862.0 ± 13.0 | Na 1880.2 ± 24.2; Cl 1858.5 ± 19.9; K 1850.9 ± 27.1 |
| NaK-16-mace-mc | 2000K | `mass_weighted` | mc-on | 40,000 | 1874.9 ± 5.2 | Na 1906.3 ± 10.6; Cl 1869.8 ± 9.8; K 1853.6 ± 11.2 |

| Case | Target | Projection | MC | Steps | Total (3N − 3) | Per species, fixed centre of mass (K) |
| --- | --- | --- | --- | --- | --- | --- |
| NaK-16-mace-mc | 2000K | `ase` | mc-off | 40,000 | 2123.0 ± 16.0 | Na 2119.5 ± 24.3; Cl 2104.1 ± 26.6; K 2164.8 ± 32.6 |
| NaK-16-mace-mc | 2000K | `ase` | mc-on | 40,000 | 2023.3 ± 5.6 | Na 2023.8 ± 12.6; Cl 2027.2 ± 10.3; K 2015.0 ± 12.9 |
| NaK-16-mace-mc | 2000K | `mass_weighted` | mc-off | 40,000 | 1986.2 ± 13.8 | Na 1965.1 ± 25.3; Cl 1991.2 ± 21.3; K 1997.7 ± 29.3 |
| NaK-16-mace-mc | 2000K | `mass_weighted` | mc-on | 40,000 | 1999.9 ± 5.5 | Na 1992.4 ± 11.1; Cl 2003.3 ± 10.5; K 2000.7 ± 12.1 |

NaK-16-mace-mc/2000K/ase/mc-on: accepted 14312 of 16800

NaK-16-mace-mc/2000K/mass_weighted/mc-on: accepted 14227 of 16800
