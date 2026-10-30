# Three small speed items: pair repulsion, constexpr D, SH gradient (30 October 2026; none adopted)

The fourth list of [plans/capacity-ideas-20261030.md](../../plans/capacity-ideas-20261030.md), approved by the owner
through the conductor. MH-1 (`mh1-r2scan`) hybrid float32, default plan, local RTX 5060 Ti (sm_120, WSL2, driver
610.88), torch 2.14.1+cu130, environment `mace_cueq_oeq`.

## Verdict

| Item | Result |
| --- | --- |
| **A. Fused pair repulsion (ZBL)** | **Stopped: already fused in the compiled step.** At most about 15 µs per step on NaCl, against the 100 µs threshold. No kernel written. |
| **B. `segment_gemm` with D as `tl.constexpr`** | **Exact, neutral in speed, not adopted (reverted):** same-process A/B b/a 0.992–1.004. The only same-sign pair is NaNbO3 at 8,640 atoms, +0.2 % in both orders, inside its block range (0.973–1.011). Not adopted (conductor, 30 October): no gain, and the constexpr adds compiled variants (setup time); the code change is reverted. No tail gap between the per-irrep launches. |
| **C. SH gradient on chip in `wprime_backward`** | **Not worth building:** the SH backward is one kernel of 1.2–8.1 µs per force call, under 0.1 % of the step. |

## The profile

The 28 October numbers that motivated A and C came from `tools/bench/fp32_parts.py`, which runs the eager tier: there,
`pair_repulsion_fn` is 59 forward and 58 backward launches (157 µs on NaCl). The production step is the compiled graph
tier, so it was profiled again:

- `tools/step_breakdown.py`'s main pass: 10 eager calls of the compiled force model under torch.profiler, by kernel;
- its nsys pass: 10 force-graph replays with every graph node traced;
- Inductor's generated code logged (`TORCH_LOGS=output_code`), whose "Source Nodes" comments name the model
  variables each kernel computes.

NaCl, NbCl5 and NaNbO3 3×3×3, 1000 K, 1 fs, as on 28 October. FastMACE `2aaeb55d` (`main` `bd540ccd` plus the run
script; no library change). Records: `results/speed-small-20261030/profile/`.

| 3×3×3 | atoms | padded edges | kernels per force call | kernel time per force call |
| --- | ---: | ---: | ---: | ---: |
| NaCl | 216 | 8,192 | 183 | 2.30 ms |
| NbCl5 | 1,944 | 58,880 | 188 | 17.6 ms |
| NaNbO3 | 1,080 | 82,688 | 199 | 12.8 ms |

## A. Pair repulsion

In the compiled step ZBL has no kernels of its own:

- **Forward.** `phi` and `v_edges` and the scatter to nodes are one kernel. Its source nodes are `exp`, `exp_1`,
  `exp_2`, `exp_3`, `phi` and `v_edges`. `Z^a_exp` is computed in the geometry kernel, together with the lengths and
  the spherical harmonics.
- **Backward.** ZBL is folded into the edge kernel that also runs the envelope and radial backward, plus three
  node-sized kernels of about 2 µs each.

| µs per force call | NaCl | NbCl5 | NaNbO3 |
| --- | ---: | ---: | ---: |
| forward ZBL kernel (`add_exp_mul_scatter_add`) | 3.3 | 7.5 | 17.0 |
| backward edge kernel with ZBL (`…exp…neg_pow…where`) | 4.0 | 25.3 | 43.3 |
| radial-table backward kernel that also has an `exp` | 1.5 | 3.5 | 4.6 |
| node-sized kernels (3 launches) | 5.7 | 6.0 | 5.7 |
| **sum, an upper bound** | **14.5** | **42.3** | **70.6** |

The bound counts these kernels in full, but they also carry the envelope, radial and scatter work. A separate exact
ZBL kernel would therefore save launches the compiled step does not have, and none of the work. The threshold was
≥ 100 µs per step on NaCl, so item A stops before the kernel.

## B. D as `tl.constexpr`

**Change (measured at `54942faf`, then reverted).** In `fastmace/kernels/triton/segment_gemm.py`, D (the irrep size 2l + 1) is a `tl.constexpr` of the
gathered-row GEMM. Triton compiles one variant per irrep (D = 3, 5, 7; D = 1 was already specialised), and `r // D`
and `r % D` are by a constant. The arithmetic is the same, and the results are bitwise equal.

**Launches.** The joined products launch once per irrep.

- NbCl5 3×3×3: groups of 3 launches (D = 3, 5, 7; D = 1 is on cuBLAS by the shape rule), grids 92/152/213 × 8, four
  groups per force call.
- NaNbO3: groups of 2 launches.
- Between consecutive launches the GPU is idle for 0.16–0.19 µs (nsys, graph nodes). Each launch is 20 or more waves
  on 36 SMs, so there is no tail gap worth co-scheduling.
- These GEMMs take 4.85 of 17.3 ms on NbCl5 (28 %) and 2.0 of 13.5 ms on NaNbO3.

**Same-process A/B** (`tools/bench/ab_constexpr_d.py`).

- Method: `tools/ab_rewrites.py` with side a's `segment_gemm._kernel` replaced by `main`'s runtime-D kernel, while side
  a builds or steps. Both variants were compiled in the A/B process: the Triton cache holds the runtime-D kernel
  (`arith.remsi` by an argument) and the three constexpr ones (by a constant).
- Each cell was run in both build orders (ab, pattern abab; ba, pattern abba).
- ms per step; b/a is the ratio of medians, and the bracket is the range of the block ratios:

| cell | atoms | order | a | b | b/a |
| --- | ---: | --- | ---: | ---: | --- |
| NaNbO3 | 1,920 | ab (abab) | 24.352 | 24.329 | 0.9991 (0.995–1.000) |
| NaNbO3 | 1,920 | ba (abba) | 24.291 | 24.269 | 0.9991 (0.998–1.001) |
| NaNbO3 | 8,640 | ab (abab) | 116.048 | 116.288 | 1.0021 (0.989–1.011) |
| NaNbO3 | 8,640 | ba (abba) | 114.973 | 115.228 | 1.0022 (0.973–1.010) |
| NbCl5 | 1,944 | ab (abab) | 17.715 | 17.571 | 0.9919 (0.991–1.002) |
| NbCl5 | 1,944 | ba (abba) | 17.634 | 17.703 | 1.0040 (1.001–1.006) |
| NbCl5 | 9,216 | ab (abab) | 87.630 | 87.654 | 1.0003 (1.000–1.001) |
| NbCl5 | 9,216 | ba (abba) | 87.548 | 87.450 | 0.9989 (0.998–0.999) |

No recompilation during the timed blocks; no other GPU process. Averaged over the two orders: NbCl5 0.998 (1,944
atoms) and 1.000 (9,216); NaNbO3 0.999 (1,920) and 1.002 (8,640). The integer division the change removes was not
on the critical path; presumably these GEMMs are bound by their loads and `tl.dot`, not by the row addressing
(not profiled).

**Tests.**

- `tests/gpu/test_segment_gemm_constexpr_gpu.py` (at `54942faf`, removed with the revert) passed (3 tests): the joined products and input gradients are bitwise
  equal to the runtime-D kernel for D = 1, 3, 5, 7, with a full and a ragged last tile, and D is a constexpr of the
  kernel.
- Gate (`tools/force_stress_gate.py`, NaCl, NbCl5, NaNbO3): passed.
- Targeted CPU tests on murphy (GPU hidden, `-n 8`): 113 passed, 18 skipped (`FASTMACE_BASELINE` was not set),
  including `test_layering.py`.

## C. SH gradient on chip

The spherical-harmonics backward is one Inductor kernel (`…mul_select_select_backward`). It reads the two `dsh`
[E, 16] tensors, from interaction 0's OEq convolution and interaction 1's `wprime_backward`, with the edge geometry,
and writes the edge-vector gradient.

| per force call | NaCl | NbCl5 | NaNbO3 |
| --- | ---: | ---: | ---: |
| SH backward kernel | 1.2 µs | 6.1 µs | 8.1 µs |
| one `dsh` (64 B per edge) | 0.5 MB | 3.8 MB | 5.3 MB |
| share of the force call's kernel time | 0.05 % | 0.03 % | 0.06 % |

Each `dsh` is written once and read once, with 15 MB of traffic in all on NbCl5. The kernel's effective rate, about
1.2 TB/s, shows that the reads come from L2. Writing the edge force inside `wprime_backward` would save this kernel
and one `dsh` write: about 10 µs per step at most on these cells, under 0.1 %. Item C stops.

## Records

`results/speed-small-20261030/`:

- `local.sh`: the jobs;
- `jobs.log`;
- `profile/<structure>/`: `ops.csv`, `kernel_sequence.csv`, `summary.json`, the nsys GPU trace CSV (compressed), and the
  generated code (`<structure>.code.log`, compressed);
- `ab/`: the A/B JSONs;
- `gate-summary.txt`.
