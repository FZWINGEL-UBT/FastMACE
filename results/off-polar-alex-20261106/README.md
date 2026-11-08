# MACE-OFF and POLAR-1 on the A100 (off-polar-alex, 6–7 November 2026)

Branch `fastmace/off-polar-alex`. Goal (conductor brief): bring every MACE-OFF and MACE-POLAR-1 model up to the
current A100 speeds. A per-GPU default needs a same-process MD-step A/B in both build orders (decision 66), and a
rounding change needs `tools/force_stress_gate.py` no worse than the default.

**Where it ran.**

- Timings: NVIDIA A100-SXM4-80GB only.
  - NHR@FAU Alex: Slurm `-C a100_80`, one GPU per job, NAS-equivalent environment `jobs/env.sh`.
  - Modal: profiles `fifth`, `third`, `fourth`; `tools/modal/offpolar_tasks.sh` at the commit each job names in
    `modal/<job>/out/command.log`.
- New ideas were built and tried first locally (RTX 5060 Ti, `mace_cueq_oeq_polar`) and on leto (RTX 4000 Ada).
- Every run: backend hybrid, float32. Baseline is FastMACE `main` as merged into the branch at each job's commit;
  the source commit is in each `run.log` line.
- Structures:
  - MACE-OFF: water box `water_216` (648 atoms) for timing; gates on urea, naphthalene, paracetamol (OFF23-large:
    paracetamol 2×2×2) and water_216, seed 20261015.
  - POLAR-1: NaCl (216 atoms) and NbCl5 (1,944 atoms).

b/a = candidate MD-step time over baseline time, two values = the two build orders (ab / ba), below 1 is faster.

## 1. Per-model speed on the A100-SXM4-80GB

### MACE-OFF (water_216, ms per MD step)

| Model | Before the branch | 6 Nov defaults | 7 Nov defaults | 7 Nov vs 6 Nov (b/a) | Overall |
| --- | --- | --- | --- | --- | --- |
| OFF23-small | 2.59 | 2.57 (`grouped_launch`) | **1.45** | 0.566 / 0.563 | 1.79× |
| OFF23-medium | 4.50 | 4.36 (`tf32x3_gemms`) | **2.61** | 0.599 / 0.598 | 1.72× |
| OFF23b-medium | 6.23 | 6.08 (`tf32x3_gemms`) | **3.89** | 0.640 / 0.635 | 1.60× |
| OFF24-medium | 6.19 | 6.07 (`tf32x3_gemms`) | **3.90** | 0.640 / 0.645 | 1.59× |
| OFF23-large | 12.12 | 11.69 (`tf32x3_gemms`) | **6.97** | 0.598 / 0.592 | 1.74× |

"Before" and "6 Nov" from the Alex `final-f80-*` A/Bs (a = `tf32x3_gemms` and `grouped_launch` off); "7 Nov" from the
Modal `offall-ab*` A/Bs (a = the 6 November defaults, b = the three switches below). The two sources ran on
different A100-SXM4-80GB machines on different days; each ratio is same-process.

The 7 November defaults, each also measured alone (Alex `stockskip-m80`, `offtables-m80`, `radtf32-m80`; Modal
`off-ab2`):

| Switch | OFF23-small | OFF23-medium | OFF23b-medium | OFF24-medium | OFF23-large |
| --- | --- | --- | --- | --- | --- |
| Stock first block's species skip (`STOCK_FIRST_SKIP_GPUS`), exact | 0.709 / 0.714 | 0.749 / 0.749 | 0.826 / 0.821 | 0.823 / 0.826 | 0.707 / 0.714 |
| Radial tables with the stock first block (`DECISION_66_STOCK_FIRST_TABLES_GPUS`), certified | 0.933 / 0.929 | 0.942 / 0.945 | 0.923 / 0.933 | 0.934 / 0.938 | 0.977 / 0.971 |
| Radial last layer as 3xTF32 (`RADIAL_TF32X3_GPUS`), precision change | 0.962 / 0.971 | 0.930 / 0.933 | 0.915 / 0.914 | 0.916 / 0.915 | 0.919 / 0.919 |
| All three (`offall`) | 0.566 / 0.563 | 0.599 / 0.598 | 0.640 / 0.635 | 0.640 / 0.645 | 0.598 / 0.592 |

**OFF23-small also takes `tf32x3_gemms` since 7 November** (in place of `grouped_launch`): on top of the three
switches above 0.966 / 0.947 (Modal `small-tf32x3-ab`, an A100 80GB PCIe, base 1.45 ms as on the SXM4). Its gate,
which failed at 2× on 6 November, passes at 3× (`small-tf32x3-gate`, A100-SXM4-80GB: paracetamol forces 2.55,
stress 2.77; default 1.53, 1.30). So OFF23-small's step is about 1.40 ms, 1.85× faster than before the branch.

**RTX 5060 Ti** (local, `local/offlocal-ab-20261107`, against that GPU's defaults): stock skip + OFF tables
(`offst`) OFF23-small 0.677 / 0.671, -medium 0.737 / 0.735, -large 0.793 / 0.811, so both are listed for the 5060 Ti
too. Adding the radial 3xTF32 path (Triton there, not CUTLASS) is slower (`offall` 0.734 / 0.742, 0.806 / 0.792,
0.837 / 0.811), so `RADIAL_TF32X3_GPUS` stays the A100. The RTX 4000 Ada was not measured (every workstation busy).

### POLAR-1 (ms per MD step)

| Model | Cell | Before the branch | 6 Nov defaults | 7 Nov defaults | 7 Nov vs 6 Nov (b/a) | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| POLAR-1-S | NaCl | 3.63 | 2.75 | **2.59** | 0.943 / 0.942 | 1.40× |
| POLAR-1-S | NbCl5 | 13.08 | 9.93 | **9.06** | 0.912 / 0.913 | 1.44× |
| POLAR-1-M | NaCl | 6.33 | 5.13 | **4.98** | 0.971 / 0.969 | 1.27× |
| POLAR-1-M | NbCl5 | 28.31 | 20.96 | **20.14** | 0.961 / 0.962 | 1.41× |
| POLAR-1-L | NaCl | 11.15 | 9.44 | **9.25** | 0.979 / 0.985 | 1.21× |
| POLAR-1-L | NbCl5 | 49.68 | 37.81 | **36.91** | 0.976 / 0.977 | 1.35× |

- "Before": a side of the Alex `screen-s80-polar1-s` `tf32x3` and `stack-s80-polar1-{m,l}` `tf32pair` A/Bs.
- "6 Nov defaults": `tf32x3_gemms` on S, M and L; `wprime_backward` + `fused_folded_forward` on M. Includes changes
  that reached `main` in between (round-7 lab, segmented sums), so 6 Nov vs before is not one switch.
- "7 Nov": `SPIN_BATCHED` (Modal `spin-ab`, a = the 6 November defaults).

## 2. Ideas table

| Idea | Bound (what it can save) | Built | A/B (A100-SXM4-80GB, MD step b/a) | Gate | Decision |
| --- | --- | --- | --- | --- | --- |
| `tf32x3_gemms` on POLAR-1 and OFF (6 Nov) | CUTLASS 3xTF32 for the e3nn products | existing | POLAR NbCl5 0.792–0.847, NaCl 0.906–0.940; OFF 0.963–0.981; OFF23-small on the 7 Nov defaults 0.966 / 0.947 | pass (OFF23-small failed at 2×: forces 2.19; passes at 3×: 2.55 / 2.77) | default, A100 (d66), every OFF model since 7 Nov |
| `wprime_backward` + `fused_folded_forward` on POLAR-1 (6 Nov) | the w′ backward | existing | M on tf32x3: NaCl 0.976 / 0.977, NbCl5 0.920 / 0.920; L NbCl5 1.003 / 1.002 | pass | default POLAR-1-M, A100; L rejected (mixed cells) |
| `grouped_launch` on OFF23-small (6 Nov) | launches | existing | 0.989 / 0.995; OFF23-large 1.008 / 1.009 | exact | default OFF23-small 6 Nov; replaced by `tf32x3_gemms` 7 Nov |
| Stock first block's species skip (`STOCK_FIRST_SKIP`) | e3nn's one-hot skip_tp, ≈ 0.8 ms of OFF23-small's 2.57 ms (module profile) | `generic/modules.py` | 0.71–0.83, every OFF model; with the tables: 5060 Ti 0.67–0.81, RTX 4000 Ada OFF23-small 0.592 / 0.591, -medium 0.688 / 0.683, OFF23b 0.720 / 0.725, OFF24 0.722 / 0.728, OFF23-large 0.749 / 0.745 | exact | **default, A100, 5060 Ti, Ada** |
| Radial tables with MACE-OFF's stock first block | the radial MLP per edge | `rewrites/radial` (`StockTableWeights`) | 0.92–0.98, every OFF model (Ada and 5060 Ti: with the row above) | certified | **default, A100, 5060 Ti, Ada** |
| Radial last layer as 3xTF32 (`RADIAL_TF32X3`) | [E, 64] @ [64, 3808] on SIMT FP32, 1.0 + 1.3 ms of OFF23-large's 12.1 ms | `skinny_linear` | 0.91–0.97, every OFF model | pass on all five alone; with the two above stress up to 2.50× (OFF24-medium paracetamol) under 3× | **default, A100, MACE-OFF only** (eligible products; MH-1 and OMAT-0 not measured) |
| `SPIN_BATCHED`: POLAR's α and β field features in one call | half the k-space launches of the field features | `longrange/polar.py` | S NaCl 0.943 / 0.942, NbCl5 0.912 / 0.913; M 0.971 / 0.969, 0.961 / 0.962; L 0.979 / 0.985, 0.976 / 0.977. 5060 Ti S 0.943 / 0.945, 0.885 / 0.882; M 0.977 / 0.971, 0.949 / 0.951; L 0.994 / 0.987, 0.966 / 0.975. Ada S NaCl 0.953 / 0.953, NbCl5 0.897 / 0.897; M 0.983 / 0.982, 0.957 / 0.953; L 0.995 / 0.994, 0.975 / 0.975 | pass at 3× (S, M leto; L Modal) | adopted: plain default `True` (pure removal) |
| `SMALL_MATMULS`: mm/addmm with K ≤ 8 or an output edge ≤ 4 as a fused multiply-sum | 111 of 655 kernels per POLAR-S step (5060 Ti) | `rewrites/inductor.py` | 5060 Ti POLAR-S NaCl 0.951 / 0.951, NbCl5 1.00 / 1.00; OFF23-small 1.07 / 1.045 (slower); POLAR-M NbCl5 0.998 / 0.999, NaCl invalid (b recompiles while timing, also with 1200 warm-up steps); POLAR-L NbCl5 1.000 / 1.000, NaCl 0.979 (invalid) / 0.968 | pass (POLAR-S, leto) | rejected: slower on OFF, a tie on every large POLAR cell, the small-cell gains not measurable cleanly; not sent to the A100 |
| `FUSED_FOURIER` / joint backward on POLAR-1 (megakernel lane) | k-space Fourier launches | megakernel branch | 5060 Ti S NaCl 1.107 / 1.086 (joint 1.050 / 1.054), NbCl5 1.32 / 1.34 (1.30 / 1.32); M NaCl 1.05 / 1.05 (1.04 / 1.02), NbCl5 1.15 / 1.16 (1.13 / 1.15) | — | rejected (slower); L not run |
| Codex R1: column-split input gradient (`SPLIT_COLUMNS`) | OFF23-large's 512-row chunks launch 8 CTAs on 36 SMs; kernel alone 0.88× at [512, 3808] | `skinny_linear` (fixed one-wave rule, no table) | 5060 Ti OFF23-large 0.951 / 0.953 (32-column tiles 1.07 / 1.06); OFF24 with the split forced on every chunk (lab, before the one-wave rule) 1.07 / 1.08, split32 1.006; all five, rule-based (8 November): OFF23-large 0.950 / 0.954; OFF23-small 1.004 / 1.006, -medium 0.993 / 1.005, OFF23b 0.999 / 0.999, OFF24 1.002 / 1.005 (their chunks are over the one-wave rule's 576 rows, so they run the one-axis kernel: noise); A100-SXM4-80GB (Alex 4961878, 8 November; OFF23-large's 640-row chunks split): OFF23-large 0.999 / 0.998, OFF23-small 0.993 / 1.001, -medium 0.999 / 1.005, OFF23b 1.000 / 1.004, OFF24 1.002 / 0.998; POLAR-L NbCl5 1.000 / 1.000 | exact (bitwise, GPU test: 10 passed on the 5060 Ti) | **default on the RTX 5060 Ti** (d66); A100 measured and left off (a tie; a tiling change needs a speed-up in both orders); Ada not measured |
| Codex R7: channel strips | same as R1 | not built | — | — | folded into R1 |
| Codex R2 / R6: five-term backward, factored TP adjoint | stub bound unusable: zeroed gradients change the MD, the stub came out slower (1.05–1.15) | R6 compile-only SASS falsifier | — | — | not adopted; no valid bound on file |
| Codex R3 / R5 / R8, ideas (1)(2) | fused POLAR kernels | — | — | — | handed to the polar-megakernel lane |
| Codex R4: CPU-side locality | ≈ 11 % of host time, hidden under CUDA graphs | not built | — | — | not built (below bound) |
| `FOLD_UPDATE`: POLAR's field-update MLP folded at setup (Codex N1, N4) | the update MLP's last linear and `tp_out`'s per-path products, the species half of its input | `rewrites/polar/update_fold.py` | Alex A100 S NaCl 0.965 / 0.971, NbCl5 1.000 / 1.000; M NaCl 0.969 / 0.970, NbCl5 1.001 / 1.000; L NaCl 0.972 / 0.971, NbCl5 1.000 / 1.000. RTX 4000 Ada (murphy) S NaCl 0.966 / 0.966, NbCl5 1.000 / 0.999; M 0.972 / 0.970, 1.001 / 1.000; L 0.977 / 0.975, 1.000 / 1.000. RTX 5060 Ti S NaCl 0.948 / 0.963, NbCl5 1.002 / 0.998; M 0.974 / 0.960, 1.001 / 0.990; L 0.981 / 0.971, 1.002 / 0.997 | pass at 3× on S, M, L (Alex A100, 4957296; candidate within the default's spread) | adopted: plain default `True` (pure removal) |
| `SHARED_GEOMETRY`: one unmasked k-space geometry per call | duplicate geometry kernels | `longrange/polar.py` | S 0.990 / 1.006 (NaCl), M and L 0.996–1.000 | exact | rejected (tie; S slower in one order) |
| `E3NN_BLOCK_LINEAR` on POLAR-1-M/L | e3nn Linear launches | `rewrites/linear` | M 0.998–1.002, L 0.991–1.000; S 1.000–1.010 | rounding | rejected (tie) |
| `fused_folded_forward` alone on POLAR-1 | — | existing | S NaCl 1.235 / 1.240, NbCl5 1.40 / 1.40; M 1.13–1.19; L NaCl 1.079 | — | rejected (slower) |
| EDGE_SH, FUSED_FIRST_BACKWARD for MH-1's class on POLAR-1 | — | existing | EDGE_SH tie (S, M, L within 0.4 %); FUSED_FIRST S NaCl 1.006 / 1.006, M 1.005 / 1.004 | — | rejected |
| OFF screens: species GEMM, chunked fused forward, EDGE_SH / HEAD_SLICE / GRAPH_SUM off | — | existing | species GEMM 1.009–1.020; chunked forward 1.12–1.14 (medium); the others ties or slower | — | rejected; GRAPH_SUM off up to 0.5 % faster on OFF23-small/-medium, a tie elsewhere: left on, flagged |

## 3. Force and stress gates

`tools/force_stress_gate.py`, ratio rule 3× the float32 yardstick (`fastmace.planner.selfcheck.RATIO`, 3.0 since
7 November 2026, owner; was 2×). Floors: energy 0.01 meV/atom, forces 1e-5 eV/Å, stress 1e-7 eV/Å³. The tables in
`gate_summary.txt` and `modal_gate_summary.txt` mark PASS/FAIL against the old 2× rule.

- **All three OFF switches together** (`modal/offall-gates-*`): the worst cells are stress on paracetamol,
  OFF24-medium 2.50×, OFF23-medium 2.13×, OFF23-large 2.10× (and naphthalene 2.04×); the largest absolute stress
  difference is 6e-7 eV/Å³ (about 1 bar). Forces stay at or below 1.69×. All pass at 3×.
- **3xTF32 radial alone**: passes on all five OFF models at 2× (Alex `gate-radtf-*`, Modal `radtf32-gates`).
- **Stock skip + tables** (leto, RTX 4000 Ada, `leto3`): OFF23-medium and OFF24-medium pass.
- **`SPIN_BATCHED`**: POLAR-1-S and -M pass on leto (ratios like the default's); POLAR-1-L passes on Modal A100-SXM4-80GB: NbCl5 forces 0.56, stress 0.66, energy 0.78 (default 0.42 / 0.41 / 0.55); NaCl at the floors (`modal/spin-gate-l`; the 60-minute first try timed out, `modal/spin-gate-l-timeout60`).

## 4. Notes

- **NVT only.** Every timing is an NVT Langevin MD step (`tools/ab_rewrites.py`). NPT and stress-driven cell
  updates were not timed; the gates cover stress, so the new defaults are accurate there, but their speed-up under
  NPT is not measured.
- **POLAR-1-M gate noise.** The default against itself varies more on POLAR-1-M than on the other models: NbCl5
  stress 2.75× in one run (`gate-g80tf-polar1-m/gate-default.log`) and 1.21× in another
  (`gate-g80tp-polar1-m/gate-default.log`), with the same code. The ratio is a stress difference of order 1e-7
  eV/Å³ over a yardstick of the same order, so a single POLAR-1-M cell near the limit is noise, and a candidate is
  read against the default's own spread.
- **A100 80GB PCIe vs SXM4.** Modal sometimes hands out an A100 80GB PCIe; the task script accepts any 80 GB A100.
  The first try of `off-ab`, `sharedgeo-ab` and `e3nnlin-ab` landed on PCIe cards and stopped at setup without a
  ratio; all were run again on SXM4 (`*-ab2`), so every timing in this folder is from an A100-SXM4-80GB. Two gate
  runs (`offall-gates-b`, `-c`) ran on PCIe cards: the gate compares numbers, not times, and the PCIe card has the
  same GA100 chip. The defaults list the SXM4 card and, as `GPUModels`, cover every A100 variant (owner,
  4 and 7 November 2026); the PCIe card's lower clocks and power limit were not timed with these switches.
- **The 3xTF32 radial switch is scoped to MACE-OFF.** `folded_linear` is also MH-1's, POLAR-1's and OMAT-0's
  radial product; the GPU list applies only where the rewrite marks the product eligible
  (`generic.apply_fused_radial_backward` on a model with MACE-OFF's stock first block, and `StockTableWeights`).
  `RADIAL_TF32X3 = True` still forces every product for A/Bs.

## 5. Local detailed profiles (RTX 5060 Ti, 7 November 2026)

`local-profiles-20261107/<case>/{module,step}` (`summary.txt`): `tools/module_profile.py` (per source, one force
call) and a graphed-MD-step profile (wall, GPU busy union, idle gaps, kernels per step), RTX 5060 Ti,
`mace_cueq_oeq_polar`, commit `49f2ff2d` tree. OFF: water_216, default vs stock skip + OFF tables (the 3xTF32 radial
switch is not listed for this GPU). POLAR: NaCl and NbCl5, default vs `SPIN_BATCHED`.

| Case | Default ms/step | Candidate ms/step | Busy / wall (candidate) | Kernels/step |
| --- | --- | --- | --- | --- |
| OFF23-small | 4.04 | 2.84 | 2.26 / 2.84 | 238 → 200 |
| OFF23-medium | 8.19 | 6.20 | 5.60 / 6.20 | 306 → 266 |
| OFF23b-medium | 12.42 | 8.81 | 9.06 / 8.81 | 370 → 325 |
| OFF24-medium | 10.21 | 7.84 | 7.91 / 7.84 | 329 → 291 |
| OFF23-large | 26.99 | 21.25 | 19.58 / 21.25 | 434 → 396 |
| POLAR-1-S NaCl / NbCl5 | 3.35 / 21.85 | 2.93 / 17.50 | | 655 → 602 / 577 → 530 |
| POLAR-1-M NaCl / NbCl5 | 5.86 / 43.81 | 5.73 / 41.52 | | 912 → 858 / 859 → 813 |
| POLAR-1-L NaCl / NbCl5 | 11.94 / 77.77 | 11.96 / 75.52 | | 1131 → 1079 / 959 → 913 |

Single profiled windows, not A/Bs: the A/B ratios in sections 1 and 2 are the evidence. What they show:

- **The GPU is busy.** Idle gaps are 0.2–0.9 ms per step in every case. Launch overhead is not the limit, even
  for OFF23-small and POLAR-1-S NaCl.
- **OFF, after stock skip + tables, without 3xTF32 radial:** the radial last layer's input gradient
  (`skinny_linear._input_grad`, SIMT FP32) is the largest single source. OFF23-large: 11.1 of 26.7 ms per force
  call; OFF24-medium: 2.5 of 9.8 ms. The OEQ streaming convolution follows (7.4 and 2.8 ms). On the A100 this
  input gradient is the product `RADIAL_TF32X3` moves to tensor cores; a local A/B of it on the RTX 5060 Ti is queued
  (`offlocal-ab`).
- **POLAR:** the two Inductor graphs (forward and backward) hold 30.6 of 42.5 ms per force call on POLAR-1-M NbCl5
  and nearly all of POLAR-1-S NaCl. The field-update MLP and `tp_out` products that `FOLD_UPDATE` removes, and the
  Fourier geometry adjoints, are inside them.

## 6. Files

- `runs/`: the Alex jobs (`<kind>-<set><gpu>-<model>-<jobid>`), with `ab-*.log`, `gate-*.log` and the A/B JSON.
- `modal/<job>/out/`: the Modal jobs; `command.log` names the commit and the tasks.
- `jobs/`: the Alex job scripts and `env.sh`; `sets/`: the A/B sets.
- `profile-*`: the A100 module and step profiles (`categorize.py` groups the kernels).
- `ab_summary.txt`, `gate_summary.txt` (Alex), `modal_ab_summary.txt`, `modal_gate_summary.txt` (Modal):
  regenerated by `summarize.sh`.
- `murphy-ab/`: the RTX 4000 Ada packed A/B (murphy, 8 November 2026; mace_cueq_oeq, source 657c80b3):
  `spinoff` (a = `SPIN_BATCHED` off), `fold` (b = `FOLD_UPDATE`) on POLAR-1 S/M/L, `offst` (b = stock-first skip
  and tables) on all five OFF models.
- `local/`: the RTX 5060 Ti runs (WSL, `~/fm-runs`): `offlocal-ab-20261107` (OFF skip and tables),
  `fold-ab-20261107`, `polar-switches-ab-20261107` (fused Fourier, joint backward, spin, small matmuls; M/L reruns),
  `r1-split-20261107b` (Codex R1 split and stub lab), `split-columns-ab-20261107` (R1's GPU bitwise test and the
  rule-based MD A/B on all five OFF models; `run.sh` is the first launch, whose A/B part failed on an unquoted
  PATH, `gputest2.log` the rerun's test).
- `leto-gates/`: the workstation force/stress gates (leto, murphy; RTX 4000 Ada).
