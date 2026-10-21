# OPT-T09: where the step goes after optimisation round 1, and what to optimise next

Owner request, 21 October 2026. A timing breakdown of the MD step in the presumed new default configuration,
to choose the next optimisations. Measurement only: no library change.

**Configuration:**
- **Code:** main `6638185`, pinned copy `~/fastmace-runs/src-6638185`.
- **Defaults:** the hybrid float32 default plan, which includes `gated_rows` and `segment_gemm`.
- **Plus OPT-T08:** `species_tables`, `fold_readout_tail` and `fold_product_linear` switched on (assumed
  default, pending murphy). The plan records all five as on.
- **Run:** MH-1 `matpes_r2scan`, 3×3×3 cells, 0.5 fs, pipelined GraphEngine with CUDA graphs. NVT at 1000 K
  with the research probe's velocity seed, as in 2-T20a and the 15 October breakdown, so the tables compare
  row by row.

**Machine:**
- local RTX 5060 Ti (sm_120, 36 SMs, nominal FP32 peak 28.5 TFLOPS at 3.09 GHz; measured cuBLAS FP32 peak
  16.7 TFLOPS at 4096³, OPT-T01). DRAM peak 448 GB/s (128-bit GDDR7, 28 Gbps).
- Driver 610.88, WSL2, `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130, Triton 3.8.0, cuEquivariance
  0.12.0, OpenEquivariance 0.7.0).
- 12:09–12:39 UTC 21 October 2026, under the local GPU reservation.
- Before every process, utilisation was 0–1 % in both the Windows and WSL views, with no other compute
  process. A 5 s monitor of Windows `nvidia-smi` (utilisation, memory, SM clock, temperature) ran throughout,
  and showed 0 % between processes. Run log and monitor: `results/opt-T09-step-breakdown-6638185/`.

**Method:** `tools/step_breakdown.py` (2-T20a), three processes per structure:
- **Main pass**, with three kinds of number:
  - **Un-profiled:** 1000 MD steps of wall time, and CUDA events around every stage the host launches.
  - **PyTorch profiler, 10 eager calls:** the force call's kernels, attributed to their operators.
  - **PyTorch profiler, 10 replays:** the force graph's kernels and gaps.
- **ncu:** one eager force call, every kernel, real clocks and caches. ncu serialises kernels and adds
  overhead, so its kernel sums are 5–41 % above the profiler's (NaNbO3 5 %, NbCl5 11 %, NaCl 41 %), and are used for ratios: the
  forward/backward split and the roofline.
- **nsys:** 10 replays of the force graph with every graph node traced (the gaps between nodes).

The full tables are `results/opt-T09-step-breakdown-6638185/tables.md`, produced by `analyse.py` in that
directory; the per-structure CSVs are next to it.

## 1. Totals

| Per MD step | NaCl (216 atoms, 8,192 padded edges) | NbCl5 (1,944; 56,320) | NaNbO3 (1,080; 82,688) |
| --- | ---: | ---: | ---: |
| **Wall time** (un-profiled, 1000 steps) | **3.196 ms** | **21.461 ms** | **18.810 ms** |
| GPU span (stage events, un-profiled) | 3.229 ms | 21.629 ms | 18.863 ms |
| B graph (forces + second half kick, un-profiled) | 2.872 ms | 21.122 ms | 18.365 ms |
| Force-graph replay: span / kernel time (profiler) | 2.937 / 2.781 ms | 21.202 / 21.112 ms | 18.346 / 18.233 ms |
| Kernels per replay | 264 | 230 | 241 |
| Gaps inside a replay (nsys, every node) | **147 µs** (5.0 % of the replay) | 109 µs (0.5 %) | 124 µs (0.7 %) |

## 2. Where the time goes

### Stages (un-profiled CUDA events, ms per step)

| Stage | NaCl | NbCl5 | NaNbO3 |
| --- | ---: | ---: | ---: |
| B: forces + second half kick | 2.872 (88.9 %) | 21.143 (97.8 %) | 18.365 (97.4 %) |
| ACP: Langevin half kick, positions, neighbour check, edge compaction (integrator, thermostat) | 0.150 (4.6 %) | 0.182 (0.8 %) | 0.178 (0.9 %) |
| Neighbour-list rebuild, amortised | 0.124 (3.8 %) | 0.165 (0.8 %) | 0.213 (1.1 %) |
| Noise upload | 0.017 (0.5 %) | 0.038 (0.2 %) | 0.029 (0.2 %) |
| Host side: GPU idle between stages | 0.068 (2.1 %) | 0.101 (0.5 %) | 0.077 (0.4 %) |

So outside the force call, NaCl spends 0.36 ms per step (11 %): the integrator and thermostat graph,
neighbour rebuilds, the noise upload and the host gaps. Add the 147 µs of gaps inside the replay, and 16 %
of NaCl's step is not force kernels. On the large cells it is under 3 %.

### The force call by part

Kernel ms per force call from the profiler's eager calls. The forward/backward split comes from ncu's
durations, with the boundary at the energy gradient's seed (`ones_like`).

| Part | NaCl | NbCl5 | NaNbO3 | forward / backward (NaCl, NbCl5, NaNbO3) |
| --- | ---: | ---: | ---: | --- |
| Atom-level linears (cuBLAS, `segment_gemm` Triton) | **1.303 (47 %)** | **9.001 (43 %)** | 4.900 (27 %) | 49/51, 53/47, 50/50 |
| Radial networks and embeddings (incl. the folded last layer and its Triton input gradient) | 0.658 (24 %) | 3.899 (19 %) | **5.999 (33 %)** | 42/58, 40/60, 52/48 |
| Convolution (interaction 1 reassociated sums; interaction 2 streaming OEq) | 0.425 (15 %) | 3.560 (17 %) | 4.638 (25 %) | 18/82, 29/71, 28/72 |
| Atom-level elementwise (gates, products, residuals, readouts, species tables) | 0.230 (8 %) | 2.707 (13 %) | 1.185 (7 %) | 44/56, 44/56, 40/60 |
| Symmetric contraction (cuEq) | 0.078 (3 %) | 1.181 (6 %) | 0.571 (3 %) | not split (the ncu join does not name these kernels) |
| Edge geometry, radial basis, gathers (edge prep) | 0.057 (2 %) | 0.486 (2 %) | 0.768 (4 %) | 53/47, 76/24, 77/23 |
| Copies and fills | 0.018 (1 %) | 0.108 (1 %) | 0.133 (1 %) | – |
| **Force call kernels** | **2.769** | **20.942** | **18.194** | **40/60, 45/55, 45/55** |

The readout is inside the elementwise and linear rows. After `fold_readout_tail` it is a few µs, too small
for its own row.

## 3. Roofline: the top kernels (ncu)

Shares are of ncu's kernel time for one force call (3.90, 23.26 and 19.13 ms). "Bound" compares ncu's SM and
memory speed-of-light percentages: both below 60 % means latency or occupancy bound. The complete top-15
tables are in `tables.md`.

**NaCl:**

| Kernel | Share | DRAM (% of 448 GB/s) | FP32 (% of 28.5 TFLOPS) | Occupancy | Bound |
| --- | ---: | ---: | ---: | ---: | --- |
| OEq streaming convolution backward | 12.5 % | 197 GB/s (44 %) | 1.7 (6 %) | 63 % | memory (SOL 88 %) |
| atom GEMMs 512 × 512 (l = 1–3; cuBLAS SIMT) | 17.5 % together | 74–99 GB/s | 11.0–11.5 (39–41 %) | 23 % | compute / occupancy |
| folded radial last layer, Triton input gradient | 6.1 % | 171 GB/s (38 %) | 5.4 (19 %) | 24 % | memory (SOL 76 %) |
| radial last layer forward `[8192, 68] @ [68, 1280]` (MAGMA) | 4.5 % | 166 GB/s (37 %) | 9.2 (32 %) | 48 % | memory (SOL 87 %) |
| symmetric contraction (4 launches) | 4.3 % | 184 GB/s (41 %) | 4.7 (16 %) | 57 % | memory (SOL 78 %) |
| OEq streaming convolution forward | 2.8 % | **400 GB/s (89 %)** | 2.3 (8 %) | 46 % | memory, at the roofline |
| l = 0 GEMM `[216, 512] @ [512, 512]` | 2.0 % | 92 GB/s | 5.9 (21 %) | 18 % | latency (0.8 waves) |

**NbCl5:**

| Kernel | Share | DRAM | FP32 | Occupancy | Bound |
| --- | ---: | ---: | ---: | ---: | --- |
| `segment_gemm` forward (`joined_products`) | 13.0 % | 195 GB/s (43 %) | 10.2 (36 %) | 24 % | balanced (SOL 62 / 68) |
| `segment_gemm` input gradient | 10.2 % | 169 GB/s (38 %) | 13.0 (46 %) | 24 % | compute (SOL 71) |
| OEq streaming convolution backward | 7.6 % | 372 GB/s (83 %) | 3.4 (12 %) | 66 % | memory |
| folded radial last layer, Triton input gradient | 7.5 % | 208 GB/s (46 %) | 5.4 (19 %) | 25 % | memory (SOL 93 %: L2/L1, not DRAM) |
| radial last layer forward (MAGMA) | 5.5 % | 227 GB/s (51 %) | 9.1 (32 %) | 49 % | memory (SOL 91 %) |
| symmetric contraction | 4.9 % | 344 GB/s (77 %) | 6.7 (24 %) | 70 % | memory |
| OEq streaming convolution forward | 3.5 % | **413 GB/s (92 %)** | 2.2 (8 %) | 49 % | memory, at the roofline |
| radial layer norm + SiLU (fused Triton) | 3.3 % | 213 GB/s (47 %) | – | 80 % | memory (SOL 79 %) |
| cuBLAS l = 1 GEMM `[5832, 512] @ [512, 512]` | 2.1 % | 71 GB/s | 12.8 (45 %) | 23 % | compute |

**NaNbO3 (82,688 edges, the edge-heaviest):**
- **Convolution:** the streaming convolution backward (14.2 %, 368 GB/s, 82 %) and forward (6.2 %,
  414 GB/s, 92 %) are both at DRAM's roofline.
- **Radial last layer forward:** 10.4 %, 241 GB/s.
- **Its Triton input gradient:** 6.2 %, 394 GB/s, 88 %.
- **Radial elementwise:** the layer-norm and SiLU kernels and the radial basis gathers run at 61–82 % of
  DRAM.

**What the roofline says:**
- **The edge side is at DRAM bandwidth.** The convolution forward and backward, the radial last layer and
  its gradient, and the radial elementwise kernels run at 50–92 % of DRAM peak. They are memory-bound, so
  only less traffic can lower them. The largest traffic is the per-edge tensor-product weights,
  `[edges, 1280]` float32. That is 42 MB on NaCl, 288 MB on NbCl5 and 423 MB on NaNbO3. The radial last
  layer writes it, the convolution forward reads it, the backward reads it again and writes its gradient,
  and the Triton input gradient reads that.
- **The atom-level GEMMs** run at 10–13 TFLOPS: 36–46 % of nominal peak, 65–78 % of cuBLAS's measured
  16.7 TFLOPS at 4096³. They are compute- or occupancy-bound (4 warps, 23–25 % occupancy from 128–160
  registers). The 15 October study and OPT-T01 found no faster FP32 kernel for these shapes without TF32.
  The headroom is small, about 1.2–1.5× at best on the large shapes. On NaCl the l = 0 products (0.8 waves)
  run at 6 TFLOPS.
- **Symmetric contraction:** memory-bound at 77–85 % on the large cells.

## 4. Compared with the earlier breakdowns (same card)

| Per force call (ms) | 15 Oct, research engine | 2-T20a, FastMACE before round 1 | now | change since 2-T20a |
| --- | ---: | ---: | ---: | ---: |
| **NaCl** atom linears | 1.66 | 1.59 | 1.30 | −0.29 |
| NaCl atom elementwise | 0.41 | 0.36 | 0.23 | −0.13 |
| NaCl radial | 0.65 | 0.66 | 0.66 | 0 |
| NaCl convolution | 0.42 | 0.43 | 0.43 | 0 |
| NaCl force kernels / wall per step | 3.32 / 3.81 (span) | 3.20 / 3.79 | **2.77 / 3.20** | **−15.6 %** wall |
| **NbCl5** atom linears | 10.17 | 9.88 | 9.00 | −0.88 |
| NbCl5 atom elementwise | 8.38 | 7.35 | 2.71 | **−4.65** |
| NbCl5 radial | 3.87 | 3.96 | 3.90 | 0 |
| NbCl5 convolution | 3.46 | 3.54 | 3.56 | 0 |
| NbCl5 symmetric contraction | 1.34 | 1.20 | 1.18 | 0 |
| NbCl5 force kernels / wall per step | 27.88 / 29.37 (span) | 26.61 / 27.13 | **20.94 / 21.46** | **−20.9 %** wall |
| **NaNbO3** atom linears | 5.97 | – | 4.90 | −1.07 against 15 Oct |
| NaNbO3 atom elementwise | 4.35 | – | 1.19 | −3.16 against 15 Oct |
| NaNbO3 radial / convolution | 6.13 / 4.56 | – | 6.00 / 4.64 | 0 |
| NaNbO3 force kernels / span per step | 22.67 / 23.51 | – | **18.19 / 18.86** | **−20 %** against 15 Oct |

2-T20a: `results/2-T20a-step-breakdown-20261020/` on main. The 15 October numbers come from ResearchModule's
`studies/further_optimizations/2026-10-15-step-breakdown.md`. Its step figure is the GPU span, while the
FastMACE rows use wall time; the two agree within 1–2 %. The OPT-T02 trace (`results/opt-T02-trace/`) is from
murphy and is not in this table.

**What shrank:** the atom-level elementwise work, by 3–5 ms on the large cells. These were the joins, copies
and gate glue that `gated_rows` and `segment_gemm` removed. The atom-level linears also shrank, by 0.3 ms on
NaCl and 0.9–1.1 ms on the large cells (OPT-T08's species tables and folds, `segment_gemm`).

**What did not change:** the edge side. That is the radial networks, the convolution and the symmetric
contraction, at 3.6–11.2 ms per force call.

**What dominates now:**
- **NaCl:** the atom-level linears (47 %), then the radial networks (24 %) and the convolution (15 %). Also
  16 % of the step outside the force kernels.
- **NbCl5:** the linears (43 %), then radial (19 %) and convolution (17 %).
- **NaNbO3:** the edge side. Radial (33 %) and convolution (25 %) together are 58 % of the force kernels.

## 5. Candidates for the next round, ranked

Ceilings are fractions of the measured step: wall time for NaCl, force kernels for the large cells. All
candidates stay general: no per-GPU or per-shape tables, one crystal at a time, TF32 off.

| # | Candidate | Ceiling: NaCl / NbCl5 / NaNbO3 | Class | Effort | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | **Fuse the radial network's last layer into the streaming convolution** (forward and backward). The `[edges, 1280]` tensor-product weights would be formed in registers per edge tile and never stored. The backward would form their gradient in the same kernel and contract it with the hidden activations. | ≈ 12 % / 12 % / **20 %** (the last layer forward, its Triton input gradient and part of the convolution traffic) | rounding | high (OEq / NVRTC kernel work; the hidden activations become the operand) | Edge-side kernels are at DRAM's roofline, so less traffic is the only lever. This is the largest item for edge-heavy cells. |
| 2 | **Fewer bytes per edge on the radial side.** Fuse the radial MLP's layer norm, SiLU and hidden GEMMs (`[E, 64]` products at 6–8 TFLOPS, 48–73 % of DRAM), forward and backward, into one kernel per layer pair. | ≈ 5 % / 4 % / 8 % | rounding | medium–high | The research engine's fused radial hidden block (opt-in `fused_radial_hidden`) gave no gain after the 15 October fix. Size the headroom of a forward + backward fusion first. |
| 3 | **Triton input gradient of the folded radial layer:** 25 % occupancy and 19 % of FP32 on NaCl and NbCl5. Choose its tiles by the shared shape rule (waves over SMs), not one fixed configuration. | ≈ 3 % / 4 % / 2 % | rounding | low | Its SOL is high in L1/L2, not DRAM, so better tiling or split-K could help. Measure in the step (the lesson of OPT-T03b). |
| 4 | **NaCl's fixed costs:** fold the ACP graph and the B graph into one graph per step, and trim kernel count (264 launches per replay, 147 µs of gaps). | NaCl ≈ 4–6 %; large cells < 1 % | exact | medium | PDL stays forbidden (wrong forces in graphs). OPT-T03b showed that removing launches pays on WSL2, not on murphy, so measure on both. |
| 5 | **Atom GEMMs on small cells:** NaCl's l = 0 products run at 0.8 waves and 6 TFLOPS. Split-K or stream-K by the shape rule. | NaCl ≈ 2 % | rounding | medium | Grouping by weight shape (OPT-T03b) gained only on sm_120. |
| 6 | **Symmetric contraction:** memory-bound at 77–85 % of DRAM on the large cells. A fused input (no `[N, 16, 512]` materialisation) would need cuEq changes. | ≈ 1 % / 3 % / 2 % | rounding | high (cuEquivariance) | Low priority. |
| 7 | **The remaining atom-level elementwise:** NbCl5 2.7 ms, mostly gate products and their backward at 60 % of DRAM. | ≈ 2 % / 5 % / 2 % | rounding | medium | Stage C (gate in the GEMM) failed. Epilogue fusion of the gate's backward needs atomics (stage D, rejected). Only Inductor-level fusions are left. |
| 8 | **Atom GEMMs, large shapes:** 10–13 TFLOPS against cuBLAS's 16.7 at 4096³. | ≈ 5 % / 8 % / 4 % if a 1.2× kernel existed | rounding | high, unlikely | The 15 October study and OPT-T01/T03 found none without TF32. Not recommended. |

Not candidates (screened earlier, `docs/rewrites.md`):
- tensor cores without TF32 (OPT-T01, T01b);
- the gate inside the GEMM (stage C);
- graph-branch concurrency (OPT-T03 candidate 1);
- exact skips (OPT-T04);
- mirror-edge radial sharing and tabulated radial networks (15 October screens).

**Recommendation:**
1. Start with candidate 1. It is the largest ceiling, and it targets the part of the step that round 1 did
   not touch: the edge side, at DRAM's roofline.
2. In parallel, take the low-effort candidate 3.
3. Candidate 4 matters only for small cells. Size it on murphy too before building, since its gain may be
   WSL2-specific.
