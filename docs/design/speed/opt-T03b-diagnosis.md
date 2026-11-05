# OPT-T03 candidate 2 (`grouped_launch`): why the bench gain became a full-step loss, and the fix

Owner request, 21 October 2026: find out exactly why the isolated block-linear bench's 1.30–1.34× (sm_120)
turned into a full-step loss, and make it work in the step if the cause is clear. Branch `fastmace/opt-T03b`:
first version `cd71b5c`, fix `f2531cb`. Still opt-in (`RewriteConfig.grouped_launch`, `never_automatic`).

**Result:** two causes, both in my first version, not in the kernel. Fixed, the grouped launch makes the MH-1
NaCl step 1.8–2.3 % faster on sm_120 (same-process A/B, both orders). On murphy (sm_89) it gains nothing:
NaCl 0.25–0.46 % slower, NbCl5 neutral. The force gate passes. It stays opt-in: no gain on both GPUs.

## 1. Method

`tools/step_breakdown.py` (2-T20a) on the local RTX 5060 Ti (sm_120, 36 SMs, driver 610.88, WSL2,
`~/environments/mace_cueq_oeq`, torch 2.14.1+cu130, Triton 3.8.0), MH-1 `matpes_r2scan` hybrid float32, NVT
1000 K, 3×3×3 cells, pinned source copies. The main pass gives each force call's kernels attributed to their
operators, with grid, registers and the profiler's occupancy estimate (eager calls of the compiled model), and
10 replays of the force graph (kernel time, span and gaps per replay). Each pass is one process: grouping off
and on are separate processes, so these tables locate the cost; the A/Bs in §4 measure it.

## 2. Diagnosis (first version `cd71b5c`, NaCl)

| Per force call | off | on (`cd71b5c`) |
| --- | ---: | ---: |
| Atom-level GEMMs (cuBLAS) | 1.601 ms, 131 kernels | 0.318 ms, 59 kernels |
| Grouped launches | – | 1.289 ms, 12 launches |
| GEMMs + grouped together | 1.601 ms | 1.607 ms |
| Atom-level elementwise | 0.241 ms, 95 kernels | 0.350 ms, 99 kernels |
| Eager kernel time | 3.116 ms, 295 kernels | 3.208 ms, 240 kernels |
| Force-graph replay: kernel time / span / gaps | 3.086 / 3.270 ms / 181 µs | 3.180 / 3.288 ms / 105 µs |

**Cause 1: stragglers in launches that mix K.**
- The six launches that group a layer's four 512 × 512 per-l products (448 tiles) take about 108 µs each. The
  same four products took about 169 µs in sequence on cuBLAS. So these launches save about 0.36 ms per force
  call, about the bench's gain.
- The six launches with 544 tiles group products of different K (inferred from the cuBLAS products they
  replaced), for example an l = 0 product with K = 2048
  (the `grouped_linear` inputs: four 512-channel paths concatenated) next to K = 128–384 products. All tiles
  have the same 64 × 64 shape, so the K = 2048 tiles run 4–16 times longer than the others and set the
  launch's length. cuBLAS had split those products' K eight ways (`[16, 1, 8]` grids with `splitKreduce`).
  These six launches take 638 µs against 269 µs of cuBLAS, a loss of about 0.37 ms, which cancels the gain.
- Grouped kernel: 160 registers, 4 warps, estimated occupancy 25 %; the same configuration as in the bench.

**Cause 2: the flat output buffer.**
- The first version returned all products of a launch as one flat tensor, cut into views. Autograd then built
  their gradient through `slice_backward` into the whole buffer (2.1 M elements). Inductor's fused consumer
  kernels ran over that buffer instead of each segment: `triton_poi_fused_add_mul_sigmoid_slice_slice_backward_*`
  alone takes 50 µs, and the new `*_grouped_products_slice_view_*` kernels 11–20 µs each.
- Atom-level elementwise time rose by 0.109 ms per force call.

**Not a cause:**
- **Operand layout:** the operands were not strided differently from the bench. The kernel takes the l = 0 rows
  as a strided view, and the other rows are the same contiguous copies cuBLAS reads.
- **Backward recompute:** no recompute kernels appear in the trace. The grouped launches come in even counts
  (6 + 6), consistent with one input-gradient launch per forward launch, with the contiguous transposes.
- **Launch gaps:** the replays' gaps fell from 181 to 105 µs (fewer launches), so they were not the loss.

## 3. Fix (`f2531cb`)

- **Separate outputs:** the custom op `fastmace::grouped_products` returns one tensor per product
  (`List[Tensor]`), and its backward takes the products' gradients as they come. No flat buffer, no
  `slice_backward`.
- **Same weight shape per launch:** in `block_products`, only small products with the same weight shape
  (K, N) share a launch, so every tile does the same work in the forward and in the input gradient (where K is
  the forward's N). Products of other shapes, such as the K = 2048 l = 0 products, keep cuBLAS and its split-K.
  The rule is general: it uses only shapes, with no table of GPUs or shapes.

NaCl with the fix (`f2531cb`), against off in the same breakdown:

| Per force call | off | on (`f2531cb`) |
| --- | ---: | ---: |
| GEMMs + grouped together | 1.601 ms | 1.588 ms (0.920 ms in 10 grouped launches) |
| Atom-level elementwise | 0.241 ms | 0.233 ms |
| Eager kernel time | 3.116 ms | 3.071 ms |
| Force-graph replay: kernel time / span / gaps | 3.086 / 3.270 ms / 181 µs | 3.048 / 3.169 ms / 118 µs |

### Where the fixed version's gain comes from (ncu and nsys, `f2531cb`, NaCl)

ncu (one eager force call, real clocks and caches, as in 2-T20a) for one layer's four 512 × 512 per-l
products, input rows 216, 648, 1080 and 1512:

| | Duration | FP32 rate | Achieved / theoretical occupancy | Registers | DRAM read |
| --- | ---: | ---: | ---: | ---: | ---: |
| cuBLAS, four launches in sequence (off) | 19.6 + 33.5 + 49.1 + 67.3 = **169.5 µs** | 6.0 / 11.1 / 11.7 / 12.1 TFLOPS | 17–23 % | 80–128 | – |
| one grouped launch, 448 tiles (on) | **174 µs** | 10.8 TFLOPS | 23.8 / 25 % | 160 | 13–14 MB |

So in the step, the grouped kernel runs the four products in about the time cuBLAS takes for them in
sequence. Its operands are the same as cuBLAS's: the l = 0 rows a strided view, the others contiguous. The
bench's 1.34× (144 µs against 193 µs) was measured on replays of the same buffers, with the 11 MB of
operands warm in the 32 MB L2. In the step they come from DRAM (13–14 MB read per launch).

nsys (10 replays of the force graph, every node traced):

| Per replay | off | on (`f2531cb`) |
| --- | ---: | ---: |
| Kernels | 304 | 255 |
| Gaps between nodes | 265 gaps, **199 µs** | 239 gaps, **114 µs** |

The gain in the step is therefore mainly the launches it removes: 85 µs fewer idle gaps per force call,
plus about 40 µs less kernel time (3.086 → 3.048 ms; not attributed per product). A faster grouped GEMM
would add to this. Tiles other than the
shared 64 × 64 × 32 configuration were not tried tonight.

## 4. Full-step A/B

`tools/ab_rewrites.py`, a = the defaults (`gated_rows`, `segment_gemm`), b = a + `grouped_launch`, 12 blocks of
300 steps, both build orders, same process.

| b/a | local sm_120, a-b / b-a (`f2531cb`) | murphy sm_89 (`f2531cb`) |
| --- | --- | --- |
| NaCl | **0.9766 / 0.9817** | 1.0046 / 1.0025 |
| NbCl5 | 1.0044 / 0.9963 | 1.0008 / 1.0019 |

First version (`cd71b5c`) for comparison: local NaCl 1.0179 / 1.0103, NbCl5 1.0041 / 0.9971; murphy NaCl
1.0697, NbCl5 1.0146 (a-b).

NbCl5 does not take the grouped path. Only its l = 0 products (1944 rows) are small by the shape rule, and a
single product is not grouped, so its b/a is the noise floor.

**Against the OPT-T03 design rule** (≥ 10 % of NaCl's atom-level GEMM time, no NbCl5 loss): the gain is about
0.07 ms per step, 4–5 % of NaCl's 1.6 ms of GEMMs, with no NbCl5 loss. It falls short of the 10 % bar but
is a measured full-step gain above the local noise floor (1.1 %).

murphy (RTX 4000 Ada, sm_89, 48 SMs, driver 595.58.03, NAS `mace_cueq_oeq` through the hybrid launcher), pinned
`b-f2531cb`, GPU empty and load 0.05–1.25 before every job, 23:09–23:27 UTC:
- **GPU test:** `tests/gpu/test_grouped_gemm_cuda.py` passed.
- **Force gate** (`--rewrites gated_rows=on segment_gemm=on grouped_launch=on`): passed. NaCl ratios: energy
  1.31, forces 1.16, stress 0.72. NbCl5: 0.23, 1.01, 0.79.
- On 48 SMs, NbCl5's l = 0 and l = 1 products (1944 and 5832 rows) are both small by the shape rule and share a
  launch. NbCl5 is neutral there too.

murphy's Nsight tools (2021.3) do not support CUDA 13, so the sm_89 result is not profiled. A plausible
reading, not measured: the launch gaps the fix removes are what pays on WSL2. Native Linux on murphy has
smaller gaps per graph node, and the grouped GEMM itself is no faster than cuBLAS in the step (§3).

## 5. Verdict

- **Cause of the loss:** found and fixed (§2–3).
- **sm_120:** NaCl −1.8 to −2.3 %, NbCl5 neutral.
- **sm_89:** NaCl +0.25 to +0.46 %, NbCl5 neutral.
- **Decision:** no gain on both GPUs, so `grouped_launch` stays opt-in (`never_automatic`) and is not proposed
  as a default.
- **What could still make it pay:** a grouped GEMM faster than cuBLAS at these shapes, cold in DRAM (other
  tiles or split-K inside the launch, by the shared rule). That is unbuilt.

Raw data: `results/opt-T03b-grouped-launch/local-diag/` (step breakdowns; ncu tables for `f2531cb`),
`results/opt-T03b-grouped-launch/local-ab/f2531cb/`, `results/opt-T03b-grouped-launch/murphy/t03b-f2531cb/`.
The first version's runs are in `results/opt-T03b-grouped-launch/local/` and `murphy/t03b-cd71b5c/` on main.
