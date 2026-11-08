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

## The H200 measurement and the table-byte ideas (conductor, 8 November 2026)

### The H200 bound

Two runs of `tools/modal/spline_table_bound_job.sh` on Modal profile `sixth` (H200; sources `18f65eb0`, `e917306e`), H200 defaults, 31 rounds each. Log: [h200-run.log](h200-run.log). JSON: [first run](spline_table_bound_h200.json), [second run, with slab-sorted](spline_table_bound_h200_sorted.json).

| Kernel | Real | One slab | Slab-sorted (forward) |
| --- | --- | --- | --- |
| spline forward | 785 / 793 µs | 433 / 432 µs (× 0.551 / 0.545) | 680 µs (× 0.858 flushed / 0.861 warm) |
| w′ | 1,086 / 1,098 µs | 1,056 / 1,069 µs (× 0.973 / 0.974) | – |

Two results come out of this:

- **The forward is table-bound.** About 350 µs of its 790 µs is table traffic, roughly 10 % of the force call.
- **w′ is not.** Its table share is under 3 % (about 30 µs). That leaves nothing for the undirected-pair idea (proposal 3).

### Three ideas for reading fewer table bytes

Bounds from `tools/spline_table_ideas_bound.py` (CPU, [table_ideas_bound.json](table_ideas_bound.json)) and the H200 slab-sorted variant above.

| Idea | Bound | Verdict |
| --- | --- | --- |
| **Compressed table** (float16 or bfloat16 coefficients, float32 c0) | Fails the production certificate in every variant. Scores are slope-dominated: c2, c3 in float16 (12 B) 3.16; c1–c3 in float16 (10 B) 52; all float16 (8 B) 52; c2, c3 in bfloat16 20; c1–c3 in bfloat16 418. The certificate passes at ≤ 1; production scores 0.49. | **Rejected** |
| **Receiver order and slab reuse** | See below. | **Next build** |
| **Smaller knot grid per pair** | Each pair could drop to 390–563 of 736 intervals at the production target (pair worst scores 0.32–0.49). Bytes per edge stay the same (one 20 KB slab per edge), so only reuse can change. The modelled L1 hit rate rises by only 1–3 points (natural / CSR 0.061 → 0.073; slab-sorted 0.544 → 0.579). | **Dropped** (bound) |

#### Receiver order and slab reuse

The L1 model runs on SrTiO3 10×10×10 with σ = 0.05 Å per component, 6 resident blocks per SM, and an LRU of 10 slabs per SM. Hit rates:

| Receiver order | Each receiver's edges in CSR order | Sorted by slab |
| --- | --- | --- |
| Natural | 0.061 | 0.544 |
| Morton | 0.079 | 0.583 |
| Morton, each SM a contiguous run | 0.078 | 0.571 |
| Species + Morton, each SM a contiguous run | 0.175 | 0.747 |

Receiver order alone (Morton) barely matters. The order of edges within each receiver matters a lot: co-resident receivers then walk the shells in step.

Measured on the H200, slab-sorted edges take the forward from 792 to 680 µs: **−112 µs, about 3.3 % of the force call**. The change is rounding-class (message max|d|/max|a| 8.5e-7 against the real order).

Grouping receivers by species per SM (a block → receiver permutation) is modelled to raise the hit rate further, from 0.54 to 0.75. It has not been measured.

## Recommendation

Build the slab-sorted edge order: within each receiver, sort edges by (species pair, length) when the neighbour list is rebuilt. Steps:

1. MD-step A/B on the H200, both build orders. Between rebuilds lengths drift by at most the skin, so the order stays almost sorted.
2. The force/stress gate, since the change is rounding-class.
3. Then the species-grouped block order on top, as its own switch.

Compressed tables and per-pair grids are closed. w′ has no table-traffic headroom left.
