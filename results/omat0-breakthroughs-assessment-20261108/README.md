# Assessment of the "breakthrough mechanisms" report for OMAT-0 medium (8 November 2026)

This assesses the external report `plans/notes/omat0-breakthroughs-20261108/REPORT.md` (uncommitted, written by another agent). The report claims the H200 force call can drop from 3.40 ms to 1.25 ms.

Inputs used:

- the H200 breakdown of the main defaults, [h200-fastpath round 4](../h200-fastpath-20261108/round4/breakdown/);
- the [optimisation ledger](../../docs/optimisation-ledger.md);
- one new lab bound, `tools/spline_table_bound.py`.

## The lab bound: how much of the spline kernels is the table read

The tool captures the real arguments of `SplineForward` and `WPrimeBackward` from one eager force call, after 200 equilibration steps. It then times each kernel in alternating rounds, comparing the real inputs with a "one-slab" input: every edge inside the cutoff reads the same table slab. The one-slab outputs are wrong; only the time counts. The difference is the most that any change to the table traffic can save in that kernel.

Run details:

- GPU: RTX 5060 Ti, local.
- Model: OMAT-0 medium, SHA `d4b14be9…`.
- Structure: SrTiO3 10×10×10, 373,760 edges, of which 370,288 are inside the cutoff.
- Switches, forced as on the H200: `SPLINE_W`, `R128`, `SPLINE_VECTOR=4`.
- Timing: medians of 21 rounds.
- Result file: [spline_table_bound_5060ti.json](spline_table_bound_5060ti.json).

| Kernel | Real (L2 flushed / warm) | One slab (flushed / warm) | One slab ÷ real |
| --- | --- | --- | --- |
| spline forward | 3,935 / 3,927 µs | 2,060 / 2,050 µs | 0.524 / 0.522 |
| w′ | 5,508 / 5,461 µs | 4,631 / 4,567 µs | 0.841 / 0.836 |

On this GPU the forward spends about half its time on table reads. w′ spends about 16 %. The H200 has not been measured; its share may differ, because its L2 bandwidth is relatively higher.

## Verdicts

| # | Proposal | Verdict | Why |
| --- | --- | --- | --- |
| 1 | 64-channel h(r) spline, w = h·W_last in the kernel | **Rejected (arithmetic)** | See "Proposal 1" below. |
| 2 | Persistent atom megakernel | **Rejected (premise)** | See "Proposal 2" below. |
| 3 | Undirected edge pairs in w′ | **Closed** | w′'s whole table share is 16 % on the 5060 Ti, and pairs could save at most half of it. Messages and adjoints differ per direction, so only the table read is shared. `B-mirror-share` (ledger, research answers of 5 November) gained nothing beyond the mirror order. "w once per mirror pair" was dropped at its bound (`ideas-r3`). |
| 4 | Interaction-1 outer sum fused with the species projection | **Not tried (small bound)** | See "Proposal 4" below. |
| 5 | Multi-timescale integrator | **Not tried** | It is not exact, so it needs the owner's decision. Its velocity extrapolation also needs ∇F·v, a Hessian-vector product that costs about one more backward pass. |

### Proposal 1: 64-channel h(r) spline

The contraction w = h·W_last costs 64 × 1,280 = 81,920 FMAs per edge. The report counts 640, which is the count for one thread out of 128.

Over all edges that is 30.6 G FMA per pass. At the full SIMT rate that takes at least 0.91 ms on the H200, against an 806 µs forward. On the 5060 Ti it takes about 2.5 ms, against a saving of at most 1.9 ms. The backward pass needs the same work again for dh = Wᵀ·dw.

The tensor-core version of this factorisation is the a·W product that SPLINE_W replaced; that change took the forward from 1,269 to 959 µs (ledger `sx-layout-spline`).

### Proposal 2: persistent atom megakernel

The 3×TF32 GEMMs are not launch-bound. They take about 26 µs per launch, against graph launch gaps of about 1–2 µs.

One irreps linear is about 2 × 5,000 × 16 × 128 × 128 = 2.6 GFLOP. The report counts 0.15 GFLOP for all 18 GEMMs together.

The report also uses the H200's sparse TF32 peak (989 TFLOPS); the dense peak is about 495. At about 165 TFLOPS of effective 3×TF32 throughput, the GEMM floor is around 300 µs of the 535 µs, not 45 µs. Tile changes were already measured slower in round 5. The ledger has `launch-megakernel`, `launch-het-megakernel` and `launch-cublasdx`.

### Proposal 4: fused outer sum and species projection

Removing the [15000, 1024] buffer saves one 61 MB write and one read in each direction. At HBM speed that is about 50 µs, roughly 1.5 % of the H200 force call.

The fused kernel would also lose the CUTLASS tiling of the species GEMM (about 106 µs on its own). Low priority, and several days of work.

### The floor model

The A100 "measured" column is not a measurement: its total of 7,360 µs is the whole A100 MD step from the defaults-by-GPU notes. The H200 floors assume 12 TB/s of L2 read bandwidth throughout.

## What is new

The forward's table share is the one new piece of evidence: about half the kernel on the 5060 Ti. Any idea that cuts the bytes per edge, or improves slab reuse, without adding arithmetic is worth bounding on the H200. Examples:

- the w′ over mirror-ordered edges, still open in the ledger at about 2 %;
- a receiver order that puts neighbouring receivers with the same slabs on one SM.

The 64-channel spline adds 81,920 FMAs per edge and is not one of these.
