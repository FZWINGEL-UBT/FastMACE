# Ideas queue 2: segmented density sums, G1's ceiling, rcfold on the A100 and the step's tail (6 November 2026)

The conductor's second queue from the H200 ideas plan ([plans/h200-ideas-20261105.md](../../plans/h200-ideas-20261105.md)).
These are the recorded ideas not tried before today:

| Idea | What it is |
| --- | --- |
| 5(b) | Density sums as a segmented sum over receiver-sorted edges (`RECEIVER_SUM`) |
| 5(d) | The ZBL / geometry forward kernel's config |
| 13 | Step-invariant fills (the S zero fill) |
| 35 | Chunked S kept in L2 |
| G1 | CUTLASS 3xTF32 with pre-split weights |
| 28 | `rcfold` (fold at rendering) on the A100 |

**One switch is adopted:** `RECEIVER_SUM` on the A100 80GB, both the SXM4 and the PCIe names. The owner said
relative results carry over between the two, so A100 runs no longer retry PCIe landings.

**The others are measured and not built.** Each bound is under the A/B's noise (about 0.5 %), or the build is out of
proportion to the ceiling.

**Setup.**

| | |
| --- | --- |
| Workload | OMAT-0 medium, float32 hybrid, SrTiO3 10×10×10 (5,000 atoms; Benchmark_loup's cell) |
| GPUs | Modal H200 and A100-SXM4-80GB |
| Timing | same-process MD-step A/B through `tools/modal/hopper_sweep.sh` (`AB=`), both build orders (ab / ba) |
| Gate | `tools/modal/const_gate.sh` (`tools/force_stress_gate.py` on SrTiO3 3×3×3, NaCl, NbCl5; default against `--production-constant`) |
| Sources | `66062fda` (G1 bench, rc-all lab), `067a950e` (`RECEIVER_SUM`, off by default), `7ffcaf7a` (breakdowns) |

## Outcome

### 5(b) `RECEIVER_SUM`: adopted on the A100

**What it replaces.** The generic interactions' densities were `scatter_sum(edge_density, receivers)`. Inductor
fused the step's three density sums into one atomic kernel per edge: 74.8 µs on the A100-SXM4, 44.8 µs on the H200
(`runs/breakdown-*`).

**How it works.** `fastmace/kernels/triton/receiver_sum.py` (`fastmace::receiver_sum`):
- each program takes 1,024 receiver-sorted edges;
- it sums their runs of equal receivers with `tl.associative_scan`;
- it adds one atomic per run end.

The backward is the same gather as `scatter_sum`'s. With the switch on, the A100 breakdown shows two
`receiver_sum` launches at 17.4 µs in total, and the fused scatter is gone (`runs/segdens-breakdown-a100`).

| GPU | MD b/a (ab, ba) | Gate | Decision |
| --- | --- | --- | --- |
| A100-SXM4-80GB | 0.9998 / 0.9918; repeat 0.9990 / 0.9985 | passes the same checks as the default; worst b/default 1.036 (NaCl stress); NbCl5 stress fails on both sides (the accepted A100 failure) | **on** (`RECEIVER_SUM_GPUS`, with the A100 80GB PCIe) |
| H200 | 0.9992 / 1.0012 | – | off (a tie) |

### G1: pre-split weights, ceiling only

This is a ceiling, not the real kernel. The ceiling cubins put B's raw bits into both 3xTF32 fragments, so no B split
is left in the mainloop (`g1/`, `tools/bench/g1_ceiling_bench.py`). This counts the split's ALU work alone: it is
the most pre-split weights could save. The bench timed this plan's GEMM shapes, alternating rounds:

| GPU | ceiling / plain | Today's TF32×3 GEMMs in the step | Ceiling saving on the step |
| --- | --- | --- | --- |
| H200 | 0.894 | ≈ 925 µs of 4.76 ms kernel time | ≈ 2 % |
| A100-SXM4 | 0.915 | (similar share) | ≈ 1.5–2 % |

**Not built.** A real kernel would hold two B tiles (hi and lo) in shared memory and load two B fragments per step.
That brings back some of the cost the ceiling drops, so the realistic gain is about half the ceiling (≈ 1 %). The
work is a new CUTLASS warp-level MMA and its iterators. Recorded as the largest open item of this queue.

### 28: rcfold on the A100, not ported

All variants ran in one forward-lab job (`runs/g1-rcall-a100`):

| Variant | Forward (µs) | Note |
| --- | --- | --- |
| round 13's base (`A_LD128` and `TP_NOFLUSH` off) | 2615 | |
| `rc-all` | 2468 | 0.944 of round 13's base |
| today's default base | 2503 | |

`rc-all` is about 1.4 % of the forward faster than today's default base: about 35 µs, or 0.35 % of the A100 step.
`rcfold` is rounding (4e-8 to 3e-7 relative), and porting it to the current render is real work. The gain is below
what the MD A/B can resolve. Not ported.

### 5(d), 13, 35: bounded from the breakdowns, not built

- **5(d), the ZBL / geometry forward:** one fused Inductor kernel at 33.1 µs on the A100 (87 registers, 25 %
  occupancy). It streams about 20 edge arrays (≈ 30 MB), so it is near its bandwidth floor. A better config could
  save about 10 µs (0.1 %).
- **13, the S zero fill:** `[15000, 1024]` at 33.0 µs on the A100, about 13 µs on the H200.
  - Dropping it needs run-aligned chunks: each slot written once by its owner, empty slots zeroed in the kernel.
    That is a planner change for at most 0.34 % (A100).
- **35, chunked S in L2:** a redesign of the tabled forward and the GEMM that reads S. Its bound is the S traffic
  (61 MB written, then read), and the fill above is part of that. Not attempted at this budget.

**MD step after this queue:** H200 about 4.93 ms (unchanged). On the A100-SXM4, `RECEIVER_SUM` saves about 0.3 % on
the MD step (about 9.68 ms before).

## Overhead at a bigger cell: H200, SrTiO3 16×16×16 (owner's request)

The same step breakdown as `runs/breakdown-h200` (main `7ffcaf7a`, 10×10×10), here at main `16472d77`. The cell is
16×16×16: 20,480 atoms, 4.1× the atoms (`runs/breakdown-h200-16x16x16`).

| | 10×10×10 | 16×16×16 | ratio |
| --- | --- | --- | --- |
| Padded edges | 373,760 | 1,591,296 | 4.26 |
| Wall per MD step | 4.974 ms | 20.015 ms | 4.02 |
| Wall per atom-step | 0.995 µs | 0.977 µs | 0.98 |
| Force-graph kernel time | 4.735 ms | 18.323 ms | 3.87 |
| Kernel time per padded edge | 12.7 ns | 11.5 ns | 0.91 |
| Gaps inside the graph (133 kernels) | 53 µs (1.1 %) | 48 µs (0.3 %) | – |
| Wall minus kernel time | 0.24 ms (4.8 %) | 1.69 ms (8.5 %) | 7.0 |
| Neighbour rebuild (GPU) | 5.5 ms × 24 per 200 steps | 29.3 ms × 26 per 200 steps | 5.3 per rebuild |

**Launch overhead is not what limits us.** The graph's gaps stay at about 50 µs per step, which is 1 % at 5,000
atoms. The kernels also get about 9 % more efficient per edge at 4× the size: the small cell underfills the GPU
slightly.

**The neighbour-list rebuild grows faster than the cell.** It is 5.3× per rebuild for 4.1× the atoms, and its
host time equals its GPU time, so the host waits on it.
- At 10×10×10 it mostly overlaps the force call: the stages overlap by 0.62 ms of the 0.66 ms per step.
- At 16×16×16 it costs 3.8 ms per step against 3.5 ms of overlap. The time outside the kernels grows from 4.8 % to
  8.5 % of the step.

For large cells, the rebuild is the next target: its scaling, and the host wait. The force kernels are not.

`runs/`: one folder per Modal job, token-clean. Gate value dumps are dropped.

| Folder | What it holds |
| --- | --- |
| `g1-h200`, `g1-rcall-a100` | GEMM shapes, the G1 bench, and (A100) the rc-all forward labs |
| `breakdown-h200`, `breakdown-a100` | step breakdowns at `7ffcaf7a` |
| `segdens-*`, `segdens2-a100` | `RECEIVER_SUM` MD A/Bs |
| `segdens-breakdown-a100` | breakdown with the switch on |
| `breakdown-h200-16x16x16` | H200 step breakdown of SrTiO3 16×16×16 at `16472d77` |
| `gate-segdens-a100` | the gate |
