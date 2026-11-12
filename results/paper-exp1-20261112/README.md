# Paper experiment 1: consistency of the packed forward table (12 November 2026)

Branch `fastmace/paper-exp1-20261112`, built on main `c2ac5f53`. The branch adds tools and records only: no code
outside `tools/` changes and no default changes. Model: OMAT-0 medium (`mace-omat-0-medium.model`, SHA-256
`d4b14be9…0307a`, the same file locally, on the workstations and on Alex), float32 plans. Times are CET.

**Question (paper §5.1).** With `spline_forward.PACKED8`, on for all four GPU families, the forward pass reads a
packed quadratic copy of the spline table, while the w′ reverse kernel reads the float32 cubic for the weight values
and slopes. The forces are therefore cubic Jacobians applied to adjoints from the packed forward, and they are not
the exact gradient of either evaluator. Do the forces still conserve energy?

**Answer so far.** The packed forward (a) cannot be told apart from the consistent cubic (b): every measure sits at
the float32 floor that the autograd control (c) sets. That covers directional derivatives, knot crossings,
closed-loop work and 100 ps of NVE at 300 K. Making the two consistent would cost a 10 % slower MD step on SrTiO3
(6 % on NbCl5) on the RTX 5060 Ti. Nothing argues for changing a default. The A100 and RTX 4000 Ada records are
pending (see the end).

## Configurations

`tools/run_manifest.py` (`CONFIGS`, `apply_config`). Each process sets one configuration before anything is built.

| | Selection | What runs (from each record's manifest, `wprime_attached` and `switches`) |
| --- | --- | --- |
| a | today's defaults | `SPLINE_W`, `R128`, `STAGE_EDGES`, `HORNER`, `TABLE_W`: the spline forward reads the packed table (`spline_packed` True, `forward_packed8` True); the w′ kernel reads the float32 cubic `spline_table` |
| b | `fastmace.kernels.oeq_streaming.spline_forward:PACKED8=False` | the same, except that the forward reads the float4 cubic (`spline_packed` False): forward and reverse read one table |
| c | `RewriteConfig(radial_tables="off")` | no radial tables, so no `wprime_backward`, no spline and no packed table (`wprime_attached` empty); forces by autograd through the radial MLP. This is the control and the float32 noise floor |

Switches that go with them:
- `PACKED8` needs `STAGE_EDGES` (`SplineForward.packed8 = packed8_bias is not None and stage_edges`).
- `PACKED8` is skipped where `SHARED_LOCATOR` is on (off everywhere today).
- The spline needs `wprime_backward`, `TABLE_W`, `HORNER` and float2 tables, and that w′ needs `radial_tables`.
- So (c) turns off the whole chain: `radial_tables`, `wprime_backward`, `SPLINE_W`, `R128`, `STAGE_EDGES` and
  `PACKED8`. Its rewrite set otherwise matches (a)'s; see `plan.enabled` in each manifest.
- `EDGE_FORCE` is off on every GPU, so the forces come from the same w′ kernel in the evaluation plan and the MD plan.

The spline grid is the same in (a) and (b), certificate 0.48:
- seven uniform bands between 0.3 and 6.0 Å, with 157, 112, 87, 124, 54, 31 and 166 intervals;
- knot spacing from 1.3 mÅ (0.3–0.5 Å) to 21 mÅ (2.5–6 Å).

## Tools (new, under `tools/`)

- `run_manifest.py` writes the manifest next to every record. It holds:
  - the FastMACE revision, checkpoint SHA-256, structure and size;
  - the Python, PyTorch, cuEquivariance, OpenEquivariance, ASE and MACE versions;
  - the resolved plan: `describe(full=True)`, enabled rewrites, and the neighbour settings (algorithm, skin, int32
    edges, sorting);
  - the engine and coordinate path (`engine_info`: graph engine, pipelined, captured, padded edges);
  - every constant of the spline and w′ kernel modules, with its value on this device;
  - what each w′ kernel attached.

  The callers add the start state, temperatures, rebuild count, and the timing schedule and estimator.
- `consistency_probe.py`: measurements 1–3.
  - Evaluation is `fm.ForceField.evaluate`: compiled, the graph plan's rewrite set, and the energy summed in
    float64.
  - Configuration: the gate's first hydrostatic configuration (seed 20261015, strain 0.02, rattle 0.05 Å) of SrTiO3
    3×3×3 (135 atoms) and of SrTiO3 10×10×10.
- `nve_energy.py`: the NVE driver.
  - `equilibrate` runs Langevin and saves the state (npz with positions, momenta, masses and cell, plus a JSON
    manifest).
  - `nve` starts from a saved state and logs E_pot + E_kin and E_kin every 100 steps (CSV).
  - The drift is a least-squares slope with its ordinary standard error. The ± values quoted here are the standard
    error inflated for the residuals' lag-1 autocorrelation.
- Job scripts and helpers are in `jobs/`:
  - `local_common.sh` holds the local guards: `wrap_frac.py` (memory fraction 0.9 after FastMACE's allocator
    settings), and a watchdog on GPU-process memory > 14.8 GiB, card > 16,000 MiB, new soft-lockup or RCU-stall
    lines, and 3× the expected wall time per step.
  - The Alex and Ada files are `alex_*.sh` and `ada_*.sh`.
  - `analyse_probe.py` and `analyse_speed.py` make the tables below.

## 1–3. Directional derivatives, knot crossings, closed loop (RTX 5060 Ti, source `8c5286a3`)

Records: `local-5060ti/probe/` (`probe-{a,b,c}.json` with manifests, `tables.md`, `summary.json`).

**Directional derivatives.**
- Setup: 20 random unit displacement fields d (unit 2-norm over 3N). The mismatch is e = (E(x+hd) − E(x−hd))/2h −
  (−F·d).
- Measures:
  - |e|/|F|: |F| is the 2-norm over 3N, which bounds |F·d|.
  - |e|/|F·d|: the specified measure. It is large for 5,000 atoms because some F·d fall near zero.
  - The mean of e over directions, as a bias.
- Steps: 1e-3, 3e-4 and 1e-4 Å were specified. I added 1e-2 and 3e-3 Å because at smaller h the central difference
  is float32 energy noise: about 1e-5 eV per evaluation, so the error goes as 1/h.

Median |e|/|F| (a / b / c):

| h (Å) | 135 atoms | 5,000 atoms |
| --- | --- | --- |
| 1e-2 | 2.4e-5 / 2.9e-5 / 4.2e-5 | 4.1e-5 / 3.5e-5 / 6.9e-5 |
| 3e-3 | 6.8e-5 / 1.3e-4 / 1.1e-4 | 7.9e-5 / 7.2e-5 / 1.3e-4 |
| 1e-3 | 2.2e-4 / 2.9e-4 / 4.2e-4 | 3.0e-4 / 4.8e-4 / 6.3e-4 |
| 3e-4 | 8.9e-4 / 1.3e-3 / 1.5e-3 | 1.4e-3 / 9.6e-4 / 1.0e-3 |
| 1e-4 | 2.4e-3 / 2.2e-3 / 4.0e-3 | 4.1e-3 / 3.8e-3 / 3.0e-3 |

Median |e|/|F·d| at the specified steps (a / b / c):
- 135 atoms: 1e-3 5.4e-3 / 1.2e-2 / 1.2e-2; 1e-4 7.8e-2 / 6.2e-2 / 1.1e-1.
- 5,000 atoms: 1e-3 8.0e-2 / 7.5e-2 / 8.8e-2.

The bias is zero within its standard error for all three at h ≥ 1e-3. Full tables, with maxima and biases, are in
`local-5060ti/probe/tables.md`.

**Knot crossings** (135 atoms, atom 132).
- The line is 0.05 Å in steps of 1e-4 Å. Its direction is the best of 400 random (atom, direction) candidates:
  120 knot crossings by the moved atom's edges.
- The trend is a degree-8 Chebyshev fit.

| | largest step jump about the trend (eV) | rms jump (eV) | E(end) − E(start) (eV) | −∫F·dR (eV) | ΔE − work (eV) | max \|E − cumulative work\| (eV) |
| --- | --- | --- | --- | --- | --- | --- |
| a | 2.6e-5 | 8.6e-6 | −0.016757 | −0.016764 | +6.5e-6 | 2.6e-5 |
| b | 2.9e-5 | 8.7e-6 | −0.016757 | −0.016764 | +6.0e-6 | 3.3e-5 |
| c | 3.2e-5 | 9.5e-6 | −0.016743 | −0.016764 | +2.0e-5 | 4.3e-5 |

No knot shows a step above the float32 jitter of E (about 1e-5 eV). The economized quadratic's discontinuities at the
knots are below it.

**Closed loop** (the same atom, an equilateral triangle of side 0.02 Å in a random plane, 600 steps):

| | ∮F·dR (eV) | Σ\|segment work\| (eV) |
| --- | --- | --- |
| a | +1.33e-8 | 0.0809 |
| b | +1.15e-8 | 0.0809 |
| c | +1.18e-8 | 0.0809 |

**Reach.** These probes resolve a gradient error of about 3e-5 of |F| and a curl of about 1e-8 eV per loop (1.5e-7 of
the path work). The packed forward's inconsistency is below both.

## 4. Energy conservation (SrTiO3 10×10×10, 5,000 atoms)

### RTX 5060 Ti, 300 K (source `236d5f4b`)

- Start state: 20 ps Langevin at 300 K (1 fs, friction 0.01 /fs, configuration a, seed 1), from Maxwell-Boltzmann
  momenta. The temperature over the last tenth was 299.6 K. File `local-5060ti/nve300/state-300K.npz`, SHA-256
  `dcf267b3…29f8`; the same file starts the Ada runs.
- NVE: 100 ps at 1 fs, energy every 100 steps (1,001 samples).
- Neighbour list: GPU, skin 0.3 Å, pipelined graph engine with early rebuilds.

| | drift (meV/atom/ps) | RMS about the line (meV/atom) | range (meV/atom) | mean T (K) | T first / last tenth (K) | rebuilds | wall ms/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a | +6.0e-7 ± 9.6e-7 | 6.51e-4 | 4.64e-3 | 302.5 | 302.7 / 302.6 | 13,467 | 16.8 |
| b | +3.3e-6 ± 1.1e-6 | 6.65e-4 | 4.69e-3 | 302.5 | 302.7 / 302.5 | 13,457 | 18.3 |

- Both drifts are negligible: at most 3.3e-4 meV/atom over 100 ps, half the RMS fluctuation.
- (a) is not worse than (b).
- The wall ms/step includes sampling every 100 steps; it is not the A/B.
- Records: `local-5060ti/nve300/` (JSON manifests, CSV energy logs, step logs).

### RTX 4000 Ada (leto), 300 K and 1,500 K

Pending: jobs started 12 November 12:59 CET, records in `ada-leto/`. leeloo was to take 1,500 K but was busy with
another session's jobs, so 1,500 K runs on leto after 300 K.

### A100-SXM4-80GB (NHR Alex), (a), (b), (c) at 300 K and 1,500 K

Pending: Alex jobs 4988651 (300 K: 20 ps Langevin, 3 × 100 ps NVE) and 4988652 (1,500 K: 20 ps Langevin from
Maxwell-Boltzmann momenta, 3 × 50 ps NVE), queued on priority since 11:38 CET. Records will go to `alex-a100/`.

## 5. Speed: the price of consistency

Paired same-process MD-step A/B with `tools/ab_rewrites.py`:
- Side b is `PACKED8=False`.
- Benchmark_loup settings: 1 fs, skin 0.5, warm-up 200, 6 blocks of 256 steps per side, patterns abab and abba.
- Both build orders, on SrTiO3 10×10×10 and NbCl5 3×3×3.
- Every block time is in `ab-{ab,ba}.json`.

**RTX 5060 Ti** (source `236d5f4b`, `local-5060ti/speed/`):

| structure | order | a median / mean (min–max), ms | b median / mean (min–max), ms | b/a | pair ratios |
| --- | --- | --- | --- | --- | --- |
| SrTiO3 (5,000) | ab | 15.988 / 15.974 (15.715–16.137) | 17.622 / 17.640 (17.475–17.928) | 1.1022 | 1.092–1.125 |
| SrTiO3 (5,000) | ba | 15.942 / 15.967 (15.857–16.158) | 17.561 / 17.612 (17.442–17.915) | 1.1016 | 1.095–1.109 |
| NbCl5 (1,944) | ab | 3.468 / 3.518 (3.392–3.848) | 3.691 / 3.746 (3.631–4.058) | 1.0642 | 1.055–1.085 |
| NbCl5 (1,944) | ba | 3.399 / 3.478 (3.381–3.895) | 3.623 / 3.700 (3.614–4.089) | 1.0659 | 1.050–1.075 |

On the 5060 Ti, consistency costs 10 % of the MD step on SrTiO3 and 6.5 % on NbCl5.

The A100 (Alex job 4988653) and RTX 4000 Ada (leto) A/Bs are pending.
