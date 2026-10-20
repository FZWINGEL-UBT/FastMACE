# 2-T20a: step timing breakdown of FastMACE (baseline of optimisation round 1)

Plan task 2-T20a (owner request, 20 October 2026) and the baseline of `plans/opt-1.md`. It measures where
one MD step of current FastMACE goes. The method is the research engine's step breakdown of 15 October
2026 (ResearchModule `Experiments/Langevin_Thermostat/studies/further_optimizations/2026-10-15-step-breakdown.md`),
with the same three measurements and the same categories, so the tables compare row by row. The
optimisation round added profiling columns: achieved FLOP/s, grid and occupancy, idle gaps inside the
force graph, and bytes moved per operator.

## Setup

| Item | Value |
| --- | --- |
| Machine | local PC, WSL2 Ubuntu, 15 GB RAM |
| GPU | NVIDIA GeForce RTX 5060 Ti 16 GB (sm_120, 36 SMs, max SM clock 3,090 MHz), driver 610.88 |
| Environment | `~/environments/mace_cueq_oeq`: torch 2.14.1+cu130, triton 3.8.0, mace-torch 0.3.16, cuequivariance 0.12.0, openequivariance 0.7.0, e3nn 0.4.4, ase 3.27.0 |
| FastMACE | commit `67940f6` (branch `fastmace/2-T20a`, on main `05ecbe6`) |
| Comparison | research engine, 15 October 2026 (before `grouped_linear` and `merged_residual` were adopted on 16 October), same card, same environment |
| Model | MH-1 (`mh1-r2scan`, `mace-mh-1.model`), hybrid cuEq/OEq, float32, TF32 off |
| Plan | minimal planner defaults (what `Simulation` and `fm.md` build): pipelined graph engine with `factor_species`, `fold_cutoff`, `density_first`, `reassociate_first`, `segmented_first_sum`, `joint_radial_input`, `fused_folded_backward`, `gpu_neighbour_list`, `sort_receivers`, `block_linear`, `grouped_linear`, `merged_residual`, `streaming_conv`; float64 energy accumulation |
| Run | 3×3×3 cells, 1000 K (Maxwell–Boltzmann start, seed 20261007), 0.5 fs, Langevin NVT. One fresh process per crystal: 300 equilibration steps, two uninstrumented windows of 1,000 steps (the second counts), 1,000 steps with CUDA events per stage, 10 eager force calls under the profiler, and 10 profiled replays of the force graph. Then, per crystal, one process under Nsight Compute 2026.2 (one eager force call, every kernel, `--cache-control none --clock-control none`) and one under Nsight Systems 2026.1 (10 replays of the force graph, `--cuda-graph-trace=node`) |
| Conditions | local GPU reservation held (`tools/ci/local_gpu.py run`); idle check before every process (no WSL compute process, utilisation ≤ 5 % in both nvidia-smi views); nothing else on the card during the timed passes (`monitor.log`) |
| Raw data | `results/2-T20a-step-breakdown-20261020/{NaCl,NbCl5}/`; the tool is `tools/step_breakdown.py`, the launcher `run.sh` |

## 1. Step stages (GPU, ms per step)

| Stage | NaCl, research engine | NaCl, FastMACE | NbCl5, research engine | NbCl5, FastMACE |
| --- | ---: | ---: | ---: | ---: |
| **Total (GPU span)** | **3.81** | **3.71** | **29.37** | **27.43** |
| B: forces + second half kick | 3.49 | 3.39 | 28.96 | 26.98 |
| ACP: half kick, positions, neighbour check, compaction | 0.14 | 0.14 | 0.17 | 0.17 |
| Neighbour-list rebuild (amortised) | 0.12 (1.46 ms × 83/1000) | 0.13 (1.60 ms × 81/1000) | 0.16 (1.94 ms × 80/1000) | 0.17 (2.13 ms × 80/1000) |
| Noise upload | 0.01 | 0.01 | 0.02 | 0.02 |
| GPU idle between stages | 0.04 | 0.04 | 0.07 | 0.08 |
| Uninstrumented wall time (two windows) | 3.78 | 3.77, 3.79 | 29.18 | 27.11, 27.13 |

Padded edges: NaCl 8,192 (sizes used 7,168–8,192), NbCl5 56,320 (51,200–56,320). The instrumented wall time
equals the GPU span within 0.01 ms, so the host never limits the step, as in the research engine. No graph
was captured during the timed windows.

## 2. Force stage by model part (kernel ms per force call)

| Part | NaCl, research | NaCl, FastMACE | NbCl5, research | NbCl5, FastMACE |
| --- | ---: | ---: | ---: | ---: |
| Atom-level linear layers (GEMM) | 1.66 | 1.59 | 10.17 | 9.88 |
| Atom-level elementwise (gates, products, residuals, readouts) | 0.41 | 0.36 | 8.38 | 7.35 |
| Radial networks: folded last layer, forward GEMM | 0.18 | 0.19 | 1.21 | 1.23 |
| Radial networks: folded last layer, input gradient (Triton) | 0.16 | 0.16 | 0.83 | 0.84 |
| Radial networks: other GEMMs | 0.18 | 0.18 | 0.87 | 0.93 |
| Radial networks: layer norms and activations | 0.13 | 0.13 | 0.96 | 0.96 |
| Interaction 2 convolution, streaming forward | 0.09 | 0.09 | 0.84 | 0.86 |
| Interaction 2 convolution, streaming backward | 0.27 | 0.27 | 2.11 | 2.18 |
| Edge geometry, radial basis, gathers | 0.06 | 0.07 | 0.56 | 0.58 |
| Atom-level symmetric contraction (cuEq) | 0.08 | 0.08 | 1.34 | 1.20 |
| Interaction 1, reassociated edge sums | 0.06 | 0.06 | 0.51 | 0.50 |
| Copies and fills | 0.02 | 0.02 | 0.11 | 0.10 |
| **Kernel sum (eager)** | **3.32** | **3.20** | **27.88** | **26.61** |
| B graph replay | 3.49 | 3.39 | 28.87 | 26.98 |
| Kernels per force call (eager) | – | 307 | – | 279 |
| Kernels per B replay | – | 315 | – | 287 |

Shares of FastMACE's kernel time: the atom-level GEMMs are 50 % on NaCl and 37 % on NbCl5. The atom-level
elementwise kernels are 11 % and 28 %. All per-edge work together (radial networks, convolutions, geometry, first interaction) is 36 % and 30 %.

### What changed since 15 October

- **Atom-level elementwise: −0.05 ms on NaCl, −1.03 ms on NbCl5 (−12 %); atom GEMMs −0.07 and −0.29 ms.**
  These are the two rewrites the research engine adopted on 16 October, after its breakdown:
  - `merged_residual` runs `linear_1` and `linear_res` as one product per output segment;
  - `grouped_linear` puts same-irrep input segments into one product.

  Both remove GEMM launches and the per-l slice copies and concatenations around them.
- **Symmetric contraction: −0.14 ms on NbCl5.** Not attributed. cuEq chooses its kernel variant at run
  time (the ncu process saw other `segmented_polynomial_*` variants than the profiler process).
- **Everything per edge is unchanged within ±0.07 ms.** That covers the radial networks, both
  convolutions, geometry and the first interaction. FastMACE copied these kernels.
- **FastMACE's own engine adds no stage time.** ACP, the rebuilds, the noise upload and the idle time
  match the research engine. The float64 energy accumulation (FastMACE only) is inside the
  elementwise rows and does not show as a separate cost.
- **Net:** the step is 2.6 % shorter on NaCl and 6.6 % shorter on NbCl5 than the research engine's
  15 October default. These figures come from two runs on different days, not an A/B.

## 3. Profiling columns

### GEMM efficiency (profiler: FLOPs from the operator shapes, time of all its kernels incl. split-K reduce and memset)

The nominal FP32 peak is 36 SMs × 128 lanes × 2 × 3,090 MHz = **28.5 TFLOPS**.

| GEMM group | NaCl TFLOPS | NaCl share of peak | NbCl5 TFLOPS | NbCl5 share of peak |
| --- | ---: | ---: | ---: | ---: |
| Atom-level linear layers | 8.7 | 31 % | 12.6 | 44 % |
| Radial folded last layer, forward `[E, 68] @ [68, 1280]` | 7.7 | 27 % | 8.0 | 28 % |
| Radial other GEMMs (`[E, 64] @ [64, 64]`, `[E, 128] @ [128, 10]`, ...) | 3.4 | 12 % | 4.6 | 16 % |

Atom-level GEMMs by shape (the row count is atoms × (2l + 1): the irrep components are folded into rows,
so M is 216, 648, 1,080 or 1,512 on NaCl):

| Shape `[M, K] @ [K, N]` | NaCl µs per call (invocations) | NaCl TFLOPS | NbCl5 shape | NbCl5 µs (inv.) | NbCl5 TFLOPS |
| --- | ---: | ---: | --- | ---: | ---: |
| l = 3: `[7N, 512] @ [512, 512]` | 274 (4) | 11.6 | `[13608, 512]` | 1,889 (4) | 15.1 |
| l = 1: `[3N, 512] @ [512, 512]` | 256 (8) | 10.6 | `[5832, 512]` | 1,920 (8) | 12.7 |
| l = 0: `[N, 512] @ [512, 512]` | 214 (11) | **5.8** | `[1944, 512]` | 964 (11) | 11.6 |
| l = 2: `[5N, 512] @ [512, 512]` | 205 (4) | 11.0 | `[9720, 512]` | 1,482 (4) | 13.8 |

Nsight Compute: the atom GEMM kernels launch 14 to 272 blocks on NaCl (0.04–0.41 waves on 36 SMs) and 16 to 428
on NbCl5 (up to 1.2 waves). Their achieved occupancy is 20 % (NaCl) and 23 % (NbCl5), the occupancy of
cuBLAS's 128×64 SIMT tiles. NaCl's l = 0 GEMMs at 5.8 TFLOPS are the underfilled grids of
`plans/opt-1.md` OPT-T03. The large NbCl5 GEMMs at 13–15 TFLOPS are near this card's practical SGEMM
ceiling (section 4).

### Bytes moved per model part (Nsight Compute, one eager force call; real clocks, no cache flush)

| Part | NaCl DRAM MB | NaCl L2 MB | NbCl5 DRAM MB | NbCl5 L2 MB | NbCl5 DRAM GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Atom-level linear layers (GEMM) | 150 | 1,061 | 1,132 | 6,978 | 109 |
| Atom-level elementwise | 204 | 374 | 2,442 | 3,390 | **370** |
| Interaction 2 convolution backward | 95 | 173 | 776 | 1,311 | 378 |
| Interaction 2 convolution forward | 44 | 69 | 355 | 436 | 414 |
| Radial folded layer forward | 24 | 234 | 293 | 1,768 | 228 |
| Radial folded layer input gradient | 40 | 88 | 324 | 636 | 375 |
| Symmetric contraction | 46 | 78 | 437 | 552 | 378 |
| Radial elementwise | 38 | 83 | 403 | 602 | 365 |
| Radial other GEMMs | 30 | 78 | 296 | 845 | 296 |
| Other (edge geometry, first interaction, copies) | 19 | 85 | 297 | 631 | – |
| **Total** | **691** | **2,324** | **6,754** | **17,149** | – |

The card's DRAM bandwidth is 448 GB/s. On NbCl5 the atom-level elementwise kernels move 2.4 GB per force call
at 370 GB/s, close to that limit: they are memory-bound. Fusing them into the GEMM epilogues (OPT-T02) saves
traffic, not arithmetic. On NaCl most of the working set stays in the 32 MB L2 (691 MB from DRAM against
2.3 GB through L2), so NaCl is bound by launches and small grids, not by bandwidth. Nsight Compute times
each kernel alone (5.1 ms against 3.2 ms of eager kernel time on NaCl, 26.5 against 26.6 ms on NbCl5), so
its durations show proportions, not step time. Its join with the profiler's operators aligned 268 of 273
kernels on NaCl and 251 of 255 on NbCl5 (`summary.json`, `ncu`). The rest are cuEq variants chosen at run
time, which are categorised by name.

### Idle gaps inside the force graph

| Measure | NaCl | NbCl5 |
| --- | ---: | ---: |
| B replay span (profiler) | 3.40 ms | 26.92 ms |
| Kernel time in it | 3.21 ms | 26.80 ms |
| Gaps between consecutive kernels, profiler | 0.19 ms (5.5 %) | 0.14 ms (0.5 %) |
| of which < 2 µs | 276 gaps, 0.08 ms | 255 gaps, 0.04 ms |
| of which 2–5 µs | 37 gaps, 0.10 ms | 31 gaps, 0.09 ms |
| of which ≥ 5 µs | 0.2 per replay | 0.6 per replay |
| Gaps between graph nodes, Nsight Systems | 0.20 ms | NBCL5_NSYS |

The gaps are about 0.6 µs per kernel boundary, with a 2–5 µs tail mostly before cuBLAS GEMMs that follow a
Triton copy kernel. One Nsight Systems replay on NaCl had a single 121 µs gap; it is the only gap above 25 µs
in 10 replays and is counted in the table. Removing every gap would gain at most 5.5 % on NaCl and 0.5 % on
NbCl5. That agrees with the 8–16 % ceiling of a persistent kernel in `plans/opt-1.md`.

## 4. Reading for the optimisation round

- **Atom-level GEMMs are near the hardware on large cells, not on small ones.**
  - NbCl5 runs them at 12.6 TFLOPS and its largest shapes at 15.1 TFLOPS. This card's SGEMM ceiling is
    PEAK_LINE.
  - NaCl runs them at 8.7 TFLOPS, with the l = 0 shapes at 5.8 TFLOPS on less than half a wave.
  - So the remaining GEMM gain on large cells has to come from less arithmetic or less traffic. On small
    cells, fuller grids (grouping, split-K by a shape rule: OPT-T03) can still help.
- **The next largest item is memory traffic in the atom-level elementwise kernels.** They take 7.35 ms
  (28 %) on NbCl5 at 83 % of the DRAM bandwidth. They are the target of OPT-T02 (epilogue fusion).
- **Launch gaps and the fixed part are small:** 0.19 ms of gaps on NaCl (5.5 %), 0.14 ms on NbCl5
  (0.5 %). This confirms that a megakernel is not this round's lever.
- **OPT-T01 (tensor cores):** see `docs/validation/opt-T01-tensor-cores.md` (its step 2 uses this run's
  `gemms.csv` shapes and strides).

## Reproducing

```bash
setsid -f python3 tools/ci/local_gpu.py run --purpose "2-T20a step breakdown" -- bash results/2-T20a-step-breakdown-20261020/run.sh breakdown ~/fastmace-acceptance/<run-id> >/dev/null 2>&1
```

`run.sh smoke <dir>` checks the tool and both profilers in about 8 minutes. `run.sh nsys <dir> <structure>`
repeats one structure's Nsight Systems pass. The full run takes about 18 minutes. Most of that is Nsight
Compute, which replays every kernel about 10 times. The `.ncu-rep` and `.nsys-rep` files (30 MB) stay
outside git in the run directory.

The worktree's `.git` file holds a Windows path that WSL's git cannot read. The run's commit is therefore
taken from an untracked `.source_commit` file at the checkout root (`tools/ab_step_time.fastmace_commit`).
