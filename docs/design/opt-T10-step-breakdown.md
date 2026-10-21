# OPT-T10: where the step goes on the new default, and what to do next

Owner request, 21 October 2026: "After you merged do another detailed kernel timing of the current best
configuration". Measurement only, with no library change.

**Configuration:** main `072fa0a` (the OPT-DEF-2 merge), the plain default plan with no extra switches. That
plan now includes:
- `gated_rows` and `segment_gemm`;
- OPT-T08's `species_tables`, `fold_readout_tail` and `fold_product_linear`;
- `overlap_rebuild`;
- `folded_backward_tiles`.

The run is MH-1 `matpes_r2scan`, hybrid float32, 3×3×3 cells, 0.5 fs, the pipelined GraphEngine, NVT at
1000 K with the research probe's seed, as in OPT-T09 (`docs/design/opt-T09-step-breakdown.md`) and 2-T20a.

**Machines:**
- **Local RTX 5060 Ti** (sm_120, 36 SMs; FP32 peak 28.5 TFLOPS nominal, 16.7 measured with cuBLAS; DRAM
  448 GB/s):
  - driver 610.88, WSL2, `~/environments/mace_cueq_oeq`;
  - pinned copy `~/fastmace-runs/src-072fa0a`;
  - 18:41–19:11 UTC 21 October 2026, one local GPU reservation per structure;
  - 0 % utilisation in both views before every process;
  - the Windows `nvidia-smi` monitor runs throughout.
- **Modal L4** (sm_89, 23 GB, 72 W), "Modal L4, not murphy":
  - driver 610.57.04, the recipe-C image (`tools/modal/`);
  - main pass only (no ncu or nsys);
  - 13 min, about $0.17 of GPU list price.

**Method:** `tools/step_breakdown.py`, as in OPT-T09. The main pass gives:
- un-profiled wall times and stage events;
- the PyTorch profiler's per-kernel attribution of 10 eager force calls;
- 10 profiled force-graph replays.

Locally, ncu (one eager force call; its durations are used only as ratios) and nsys (every graph node) follow.

Raw data and tables: `results/opt-T10-step-breakdown-072fa0a/{local,modal-L4}/` (`tables.md`, per-structure
CSVs, `analyse.py`).

## 1. Totals

| Per MD step, local, un-profiled | NaCl (216 atoms) | NbCl5 (1,944) | NaNbO3 (1,080) |
| --- | ---: | ---: | ---: |
| **Wall time, OPT-T10** | **3.028 ms** | **21.434 ms** | **18.788 ms** |
| Wall time, OPT-T09 (`6638185` + OPT-T08 on) | 3.196 ms | 21.461 ms | 18.810 ms |
| Change | **−5.3 %** | −0.1 % | −0.1 % |
| B graph (forces + second half kick) | 2.827 ms | 21.273 ms | 18.556 ms |
| Force-graph replay, span / kernel time (profiler) | 2.874 / 2.720 ms | 21.188 / 21.098 ms | 18.452 / 18.344 ms |
| Gaps between graph nodes (nsys) | 174 µs (239 gaps) | 110 µs (215 gaps) | 121 µs (223 gaps) |
| Kernels per replay | 264 | 230 | 240 |

| Per MD step, Modal L4, un-profiled | NaCl | NbCl5 | NaNbO3 |
| --- | ---: | ---: | ---: |
| Wall time | 4.264 ms (1.41× local) | 33.126 ms (1.55×) | 29.589 ms (1.57×) |
| B graph / eager force kernels | 4.043 / 3.293 ms | 33.293 / 28.658 ms | 29.601 / 25.604 ms |

On the L4, the replayed force graph is 4–5 ms slower than the eager calls' kernels on the large cells. Locally
the two agree within 0.3 ms. This fits a power- or clock-limited card (72 W, 70 °C idle in the smoke) under
sustained replay; it is not a FastMACE effect. L4 numbers are comparable only within one container.

## 2. Where the time goes

### Stages (local, un-profiled CUDA events, ms per step)

| Stage | NaCl | NbCl5 | NaNbO3 |
| --- | ---: | ---: | ---: |
| B: forces + second half kick | 2.826 | 21.295 | 18.556 |
| ACP: half kick, positions, neighbour check, compaction (integrator, thermostat) | 0.144 | 0.175 | 0.183 |
| Neighbour-list rebuild, amortised | 0.163 | 0.593 | 0.817 |
| Noise upload | 0.009 | 0.024 | 0.013 |
| GPU idle between stages | **−0.111** | **−0.511** | **−0.713** |

The idle time is negative because `overlap_rebuild` now runs the rebuild on a side stream beside the force
graph. Its event span (0.59–0.82 ms on the large cells) overlaps B and is mostly hidden. The rebuild's own
span grew by 0.4–0.6 ms there, because it shares the GPU with the force kernels, and the B stage grew by
0.15–0.19 ms. Net, the large cells are unchanged. NaCl saves its 0.068 ms of host idle and more.

### The force call by part (local, profiler, eager; forward/backward from ncu)

| Part | NaCl | NbCl5 | NaNbO3 | forward / backward (NaCl, NbCl5, NaNbO3) |
| --- | ---: | ---: | ---: | --- |
| Atom-level linears (cuBLAS, `segment_gemm` Triton) | **1.306 (48 %)** | **9.025 (43 %)** | 4.891 (27 %) | 49/51, 49/51, 50/50 |
| Radial networks and embeddings | 0.613 (22 %) | 3.880 (18 %) | **6.048 (33 %)** | 45/55, 60/40, 53/47 |
| Convolution (interaction 1 sums; interaction 2 streaming OEq) | 0.426 (16 %) | 3.570 (17 %) | 4.623 (25 %) | 31/69, 42/58, 27/73 |
| Atom-level elementwise | 0.247 (9 %) | 2.746 (13 %) | 1.191 (7 %) | 44/56 throughout |
| Symmetric contraction (cuEq) | 0.077 (3 %) | 1.155 (5 %) | 0.588 (3 %) | not reliably split |
| Edge geometry, radial basis, gathers | 0.057 (2 %) | 0.524 (2 %) | 0.768 (4 %) | 54/46, 74/26, 77/23 |
| Copies and fills | 0.018 (1 %) | 0.112 (1 %) | 0.133 (1 %) | – |
| **Force call kernels** | **2.745** | **21.013** | **18.242** | 44/56, 50/50, 46/54 |

**The L4** (profiler, eager, same parts):

| | NaCl | NbCl5 | NaNbO3 |
| --- | --- | --- | --- |
| Linears | 1.367 ms (42 %) | 10.49 (37 %) | 5.75 (22 %) |
| Radial | 0.748 (23 %) | 5.77 (20 %) | 8.40 (33 %) |
| Convolution | 0.671 (20 %) | 5.35 (19 %) | 6.92 (27 %) |
| Elementwise | 0.309 (9 %) | 4.26 (15 %) | 2.11 (8 %) |
| Symmetric contraction | 0.112 (3 %) | 1.84 (6 %) | 1.22 (5 %) |

The memory-bound edge kernels slow down most on the L4 (300 GB/s against 448): the streaming backward takes
1.5× as long. The shares stay close to the local card's.

## 3. Roofline: the top kernels (local ncu)

ncu's durations vary between processes by up to 2× for single kernels (with real clocks and caches). For
example, the NbCl5 streaming forward took 822 µs in OPT-T09's ncu run and 1,681 µs now, while the profiler
gives 0.87 ms both times. So the shares and rates below are for one force call in one process. The complete
top-15 tables are in `results/opt-T10-step-breakdown-072fa0a/local/tables.md`.

| Kernel | Share (NaCl / NbCl5 / NaNbO3) | DRAM (% of 448 GB/s) | FP32 (% of 28.5 TFLOPS) | Occupancy | Bound |
| --- | --- | --- | --- | --- | --- |
| OEq streaming convolution backward | 10.1 / 8.2 / **14.7 %** | 44 / 82 / 80 % | 7–12 % | 63–66 % | memory |
| OEq streaming convolution forward | 5.2 / 7.5 / 5.9 % | 45 / 47 / **93 %** | 4–8 % | 47–50 % | memory |
| `segment_gemm` forward + input gradient (Triton) | – / **21.5** / 10.6 % | 17–51 % | **45–46 %** | 24 % | compute |
| atom GEMMs 512 × 512, cuBLAS (NaCl: l = 0–3) | 18.8 / 3.8 / 2.8 % | 15–21 % | 39–45 % (l = 0: 21 %) | 18–25 % | compute / occupancy |
| radial last layer forward `[E, 68] @ [68, 1280]` (MAGMA) | 5.1 / 8.9 / 10.2 % | 33–55 % | 21–32 % | 47–50 % | memory |
| folded layer Triton input gradient (`folded_backward_tiles`) | 5.7 / 3.9 / 6.3 % | 43 / 83 / 88 % | 20 / 38 / 42 % | 25–33 % | memory |
| symmetric contraction (cuEq) | 4.7 / 4.0 / – | 40 / 81 % | 15–27 % | 59–77 % | memory |
| radial layer norm + SiLU (fused Triton), forward and backward | 1.7 / 1.7 / 5.5 % | 37–81 % | – | 16–83 % | memory |

**What the roofline says:**
- **The edge side is at DRAM bandwidth on the large cells.** These are the convolution, the radial last
  layer and its gradient, and the radial elementwise kernels: 47–93 % of peak. Moving fewer bytes is the
  only lever, and the fused radial convolution that would have removed `[E, 1280]` failed its gate (OPT2-T01:
  in-kernel weight generation at 18 % of FMA peak).
- **The atom-level GEMMs run at 39–46 % of nominal FP32 peak:** 70–80 % of cuBLAS's measured 16.7 TFLOPS,
  compute- or occupancy-bound at 23–25 % occupancy. They are the largest single part on NaCl and NbCl5.
- **On NaCl, small-cell effects remain:** the l = 0 products at 21 % of peak (0.8 waves), 174 µs of gaps
  between graph nodes (6 % of the replay), 264 launches.

## 4. Compared with OPT-T09

Per force call, ms, local profiler; the OPT-T09 run had OPT-T08 switched on.

| | NaCl T09 → T10 | NbCl5 T09 → T10 | NaNbO3 T09 → T10 |
| --- | --- | --- | --- |
| Folded layer input gradient | 0.178 → **0.124** (−0.054) | 0.843 → 0.847 | 1.167 → 1.156 |
| Atom-level linears | 1.303 → 1.306 | 9.001 → 9.025 | 4.900 → 4.891 |
| Convolution | 0.425 → 0.426 | 3.560 → 3.570 | 4.638 → 4.623 |
| Radial (all) | 0.658 → 0.613 | 3.899 → 3.880 | 5.999 → 6.048 |
| Elementwise | 0.230 → 0.247 | 2.707 → 2.746 | 1.185 → 1.191 |
| Force kernels | 2.769 → 2.745 | 20.942 → 21.013 | 18.194 → 18.242 |
| Wall per step | 3.196 → **3.028** | 21.461 → 21.434 | 18.810 → 18.788 |

**What shrank:**
- On NaCl, the folded layer's input gradient (`folded_backward_tiles`: 128-row tiles where the shape rule
  gives 2–8 waves, so only NaCl's 8,192 edges qualify).
- The stage overheads (`overlap_rebuild`): host idle 0.068 → −0.111 ms, and B 2.872 → 2.827 ms.
- NaCl's step is 5.3 % shorter.

**What did not change:** the large cells. Both new defaults act on small-cell costs, which the large cells do
not have; their differences are within ±0.2 %.

**Relative to the 15 October research engine** (OPT-T09 §4): NaCl −20 % (3.81 → 3.03 ms), NbCl5 −27 %
(29.37 → 21.43), NaNbO3 −20 % (23.51 → 18.79).

**What dominates now:**
- **NaCl:** the atom linears (48 %), radial (22 %) and the convolution (16 %).
- **NbCl5:** the linears (43 %), radial (18 %), convolution (17 %) and elementwise (13 %).
- **NaNbO3:** the edge side, radial plus convolution at 58 %.

## 5. Next candidates, ranked

Ceilings are fractions of the measured local step. All are general: no per-GPU or per-shape tables, one
crystal at a time, TF32 off.

| # | Candidate | Ceiling: NaCl / NbCl5 / NaNbO3 | Class | Effort | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | **Radial MLP fusion** (OPT2-T02, in progress): the `[E, 64]` hidden layers, layer norms and SiLU and their backward in fewer passes | ≈ 4 / 4 / 8 % | rounding | medium–high | The radial elementwise and hidden GEMMs are 0.30 / 1.80 / 3.08 ms per call at 37–81 % of DRAM. The largest open item for edge-heavy cells. |
| 2 | **Atom GEMMs, small cells:** NaCl's l = 0 products at 0.8 waves (21 % of peak) and 264 launches per replay. Split-K or stream-K by the shared rule (OPT2-T05, in progress). | ≈ 2–3 % NaCl | rounding | medium | The grouped launch (OPT-T03b) gained only on sm_120. |
| 3 | **NaCl fixed costs:** 174 µs of node gaps (6 % of the replay); the `step_graph` variant was neutral (OPT2-T04). | ≈ 3–5 % NaCl | exact | medium | Launch-count reduction by Inductor-level fusion of the 95 elementwise kernels. |
| 4 | **Symmetric contraction:** memory-bound at 81 % on NbCl5. | ≈ 1 / 3 / 2 % | rounding | high (cuEquivariance) | Low priority. |
| 5 | **Atom GEMMs, large shapes:** 70–80 % of cuBLAS's measured peak. | ≤ 5 / 8 / 4 % if a 1.2× FP32 kernel existed | rounding | high, unlikely | No faster FP32 kernel found (OPT-T01, OPT-T03, 15 October). |

**Not candidates**, screened in `docs/rewrites.md`:
- tensor cores without TF32;
- the gate inside the GEMM (stage C);
- graph-branch concurrency;
- exact skips (OPT-T04);
- the fused radial last layer + convolution (OPT2-T01, rejected at gate 1). A backward-only variant that
  contracts `dw` with the folded weights inside the convolution, so `dw` is never written, meets the same
  limit: it executes the same E × 1280 × 65 multiply-adds inside the convolution's receiver- or edge-owned
  layout, where OPT2-T01 measured 18 % of FMA peak;
- lower-precision storage of `w` (fails the force criterion, 14 October);
- per-size Inductor combo kernels (size-specific).

**Recommendation:**
- **Finish OPT2-T02 (radial MLP fusion).** It is the only open edge-side lever with a measured ceiling above
  noise on the large cells.
- **Finish OPT2-T05 (small-cell split-K).**
- **On NaCl,** the node-gap structure (candidate 3) is the largest exact item.
- **Beyond these,** the FP32 SIMT ceiling of the atom GEMMs and the DRAM bandwidth of the edge side bound
  what software rewrites of this model can still gain on this card.
