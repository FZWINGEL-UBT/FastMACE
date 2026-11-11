# Inference speed, third round: the first interaction and two evidence updates (11 November 2026)

Source: an owner request (11 November 2026) for new ideas to make the force call faster, without training or MD
tricks. Read: `results/step-limiters-20261109/` (the per-kernel limiter tables of the H200 and the A100 with
PACKED8), the captured w′ and spline-forward sources under `results/a100-colab-ideas-20261111/` and
`results/half-far-wprime-20261111/`, `fastmace/kernels/triton/tabled_outer_sum.py`, and the ledger (keyword search
and the catalogue in `plans/notes/rejected-ledger-20261111/`).

**Nothing below is built or measured.** Sizes are my estimates from the profiles on file.

## Where this round looked

The two interaction-2 edge kernels and the GEMMs have had fifteen rounds. Most of what I tried there is already in
the ledger or dies on the limits found today (see "Worked through and dropped"). The first interaction's two tabled
kernels have had far fewer: they are 844 µs of the A100's 6,512 µs force call (13 %) and 323 µs of the H200's 3,258
(10 %).

| # | Idea | Helps | Estimated size | First test |
| --- | --- | --- | --- | --- |
| 18 | First interaction's forward by slot, sums in registers | A100 most, H200 | A100 up to 4.6 % of the force call, H200 up to 1.9 % | Falsifier: today's kernel without its slot loads and stores |
| 19 | The first radial table packed like PACKED8 | Both tabled kernels, every GPU | 1.5–3 % | Falsifier: one load per column instead of six |
| 20 | w′ accumulators started from their first product | H200 | at most about 1 % | Offline census on sm_90 |

## 18. The first interaction's forward by slot, with its sums in registers

**The evidence.** On the A100 the tabled outer-sum forward takes 400 µs, 3.5× the H200's 112 µs, against about 2×
for every other kernel. Its limiter is waiting on global loads (long-scoreboard stalls 10 per issued instruction,
issue slots 28 % used) at 23 % active warps, because Triton compiles the one-warp program to 128 registers on
sm_80. A register cap was the only thing tried, and it tied (`tabled-forward-maxnreg`).

Per multiply-add it is the slowest edge kernel by a wide margin (my count: 1,344 multiply-adds per edge here,
16,384 in the spline forward):

| Kernel | A100 | H200 |
| --- | ---: | ---: |
| Tabled outer-sum forward | 0.80 ps per multiply-add | 0.22 |
| Interaction 2's spline forward | 0.21 | 0.10 |

**Why.** Each program takes 32 edges and adds `y aᵀ` into the slot block `S[receiver, sender species]`, 1,024
floats in global memory, which it loads and stores for every run of edges. A separate zero fill (15–33 µs) and a
`scatter_add` (40 µs on the H200) sit beside it.

**The idea.** Give the kernel the spline forward's shape. One block per slot (the edges are already sorted by
receiver and sender species), one thread per radial column, the thread's 16 sums in registers over the slot's
edges, one store at the end. Per edge a thread does one table evaluation and 16 multiply-adds. No block is read
back, nothing needs zeroing, and the output is already the per-slot sum.

**Size.** If it reached the spline forward's cost per multiply-add, the A100's 400 µs would become about 100 and
the H200's 112 about 50: up to 4.6 % and 1.9 % of the force call, before the fill and scatter kernels. Half of that
is a fair expectation.

**Closest entries.** `tabled-forward-maxnreg` (register cap, tie), `kr-tabled-warps` (two warps, slower on the
H200), `packed-coef` (float4 coefficient loads, slower). None changes who owns the sums.

**Not for the backward.** There each edge needs a sum over the columns, so a warp per edge is the right shape, as
in w′.

## 19. The first radial table packed like PACKED8

**The evidence.** The tabled backward is L1-bound on both GPUs (load unit 48–58 % busy), and the limiter report
concludes that it "needs fewer loads, not better ones". Per edge it reads six coefficient rows of 64 columns, 384
loads. `PACKED8` did this for interaction 2's forward table (8 bytes per weight, a quadratic with a shared exponent)
and was worth 5–10 % of the MD step on the A100, the Ada and the 5060 Ti.

**The idea.** The same packing for the first table: one or two loads per column instead of six, in the backward
and in the forward. It is a rounding change under the table's certificate, like PACKED8.

**Size.** If loads are half of the backward's time and fall to a third, the backward loses about 30 %: 130 µs on the
A100 and 60 µs on the H200, about 2 % of the force call on each. The forward gains less on the H200 and more on
the A100, where it waits on loads.

**Closest entries.** `packed8-table` (interaction 2 only), `packed-coef` (wider loads of the same floats).

## 20. w′ accumulators started from their first product

The w′ channel loop zeroes about 25 accumulators per channel step and then adds into them: 25 register clears among
497 instructions on sm_80 (`census_half_far.json`). Writing each sum's first term as a plain product drops the
clear. The compiler cannot do this itself because `0 + a·b` and `a·b` differ in the sign of zero. On the H200,
where w′ is 77 % issue-bound, that is at most 5 % of the loop, about 1 % of the force call. Nothing on the A100,
where the FMA pipe is the limit. The spline forward already keeps its sums across edges and has nothing to gain.

## Two evidence updates for earlier ideas

- **GEMMs (idea 4 of the first list).** Every large 3×TF32 launch lists `math_pipe_throttle` as its top stall,
  with the tensor pipe 45–57 % busy. So their time follows the number of tensor instructions, not memory. They are
  1,004 µs on the A100 (15 %) and 560 µs on the H200 (17 %), in 24 launches. One FP16 pass in place of the two cross
  passes cuts a third of those instructions. This is the largest open lever on both GPUs, and its accuracy stop
  measured a cuBLAS stand-in (my review of this morning).
- **A100 forward.** Since PACKED8 the spline forward is FMA-bound on the A100 (84 %), not L2-bound as my first list
  said. Both big edge kernels, 54 % of the A100's force call, now wait on the same pipe. The half-precision far-edge
  idea therefore applies to both, within the register limit below.

## Worked through and dropped

- **Integer multiply as a third pipe on the A100.** As far as I know integer multiplies run on the same FMA pipe
  as FFMA, so there is nothing to offload to.
- **Double-half arithmetic on the idle FP16 pipe** (two float16 words per value, about 21 bits). Accurate sums need
  two registers per packed pair of accumulators, which is the register wall that stopped the FP64 pipe: w′ keeps 6
  blocks per SM only up to about 85 registers.
- **CG coefficients out of the w′ loop by scaling each dout row.** Already done: 36 coefficient multiplies are
  left for 86 terms. Removing more needs split accumulators, and the formulation search put that family at about
  1 %.
- **A lower-rank basis for the first table.** `rad-low-rank` measured rank 65 of 65 on MH-1. Not re-measured on
  OMAT-0.
- **A grouped launch for the four per-l GEMMs.** `SEGMENT_GROUPED` was built and tied.
- **A wgmma 3×TF32 GEMM on the H200.** Its bound was 1.07× today's GEMMs (`ideas-r14`, item 33).
- **The post-convolution linear computed inside w′'s staging on the idle tensor pipe.** Each receiver would stream
  655 KB of weights for 20 KB of output.
- **Keeping each GEMM block's weight tile across row tiles.** The GEMMs are bound by the tensor pipe, not by loads.
