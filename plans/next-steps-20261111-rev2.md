# What to try next, revision 2 (11 November 2026, after the first queue's results)

Source: an owner request (11 November 2026) to rewrite the queue with the nested radial basis and the H200 `GEOM`
A/B at the top. This file supersedes `plans/next-steps-20261111.md`, which stays as the record of the first queue.
State read: main `4d444f9b`; `results/first-interaction-20261111/`, `results/census-c345-20261111/`,
`results/hybrid-gemm-20261111/`; the second outside note `D:\Margraf\inference-more-20261111\ideas.md` with its
probes. **Sizes without a measurement behind them are my estimates.**

Profile used for every share (`results/step-limiters-20261109/`, OMAT-0 medium, SrTiO3 10×10×10): kernel time of one
force call 3,258 µs on the H200 and 6,512 µs on the A100. Lab bars: 68 µs on the H200, 130 µs on the A100.

## The short version

1. **One kernel idea is worth a lane: the nested radial basis** for the first interaction (64 columns near, 32
   beyond 3.15 Å). Its ceiling is 5.4 % of the A100's force call and 4.1 % of the H200's. It removes work, where
   the two layout ideas for the same kernels were tested and stopped today.
2. **One Modal H200 pack should carry two switches that already exist:** `tf32x3.HYBRID` and `GEOM`. Neither has
   been measured there.
3. **Three packed jobs from the first queue are not on main yet:** `BINNED_SCREEN` on the Ada, larger receiver
   caps for the A100's uncaptured tiled calls, shuffled atoms.
4. **Everything else in the first queue's kernel section is closed** (below). The other two new outside ideas
   should not be built.

## What the first queue produced

| Item | Outcome | Record |
| --- | --- | --- |
| `BINNED_SCREEN` on the RTX 5060 Ti | Merged, default | `results/binned-screen-5060ti-20261111/` |
| `tf32x3.HYBRID` on the A100 family | Merged, default: MD step 0.983–0.990, gate passes | `results/hybrid-gemm-20261111/` |
| C1, first interaction's forward by slot | Stopped: the falsifier saved 202 µs on the A100, the best correct kernel kept 76 | `results/first-interaction-20261111/` |
| C2, first table packed | Stopped by a bound: the falsifier saved 168 µs on the A100, but packed slopes need 9,000–15,000 pieces and 71–91 MB | same |
| C3, one decode for two edges | Stopped: spills on sm_90, at most 18 instructions saved against a bar of 27 | `results/census-c345-20261111/` |
| C4, joint rescaling | Stopped: 25 → 10 products, about 97 µs on the A100 and 37 µs on the H200, under both bars | same |
| C5, accumulators from their first product | Stopped: no instruction drop; the compiler already emits one instruction | same |
| Texture interpolation with a correction | Closed by argument | ledger `outside-texture-interp` |
| Coordinate precision (tiled idea 11) | Default by size on both paths; costs 2.8 % on the A100 tiled call at 58³ | `results/tiled-speed-20261111/` |

## 1. The nested radial basis (first priority)

**The idea** (outside note, idea 1). The first interaction sums `Y ⊗ a(r)` over 64 radial columns. For each species
pair, rotate the columns with an orthonormal basis whose first 32 directions span the radial functions beyond
3.15 Å to 1e-7. Compose the rotation into the node matrices at setup. Near edges use all 64 columns; far edges
evaluate and accumulate only the first 32.

**Why it is credible.**
- **It removes work.** For 84 % of SrTiO3's edges it halves the table loads, the series evaluation, the outer
  product and the slot updates. The table does not grow.
- **The columns are what cost.** Today's one-load-per-column falsifier saved 168 µs on the A100.
- **The matrices already have the right shape.** With `COMPOSE_SKIP` the composed weight is per receiver species,
  with one block of columns per sender species (`fastmace/rewrites/mh1/modules.py`, `_composed`). A basis per
  species pair composes into it without regrouping the GEMM.
- **The band edge is continuous.** The prefix is seeded with the value and slope at 3.15 Å.

**Size.** Ceiling 353 µs on the A100 (5.4 %) and 135 µs on the H200 (4.1 %), if every per-edge cost scales with the
columns. I expect about 3–4.5 % on the A100 and about 3 % on the H200.

**Steps, each with its stop rule.**

| Step | What | Where | Stop if |
| --- | --- | --- | --- |
| 1 | Certificate: transform the real table and the composed matrices, evaluate in float32, and run the radial certificate against the original chain. Take the rank at a tolerance that leaves room for the table's own fit error. | CPU | The far rank passes 32 for any pair at the tolerance the certificate needs |
| 2 | Timing bound: today's lab (`tools/first_interaction_lab.py`) with 32 columns for every edge, forward and backward. The gain is 0.84 × the difference from 64 columns. | 5060 Ti, then Alex A100 | Under 130 µs on the A100 |
| 3 | The real kernels: a per-edge width branch (every lane works on the same edge, so no edge partition is needed), the rotated table, the composed matrices | lane | Not faster in both orders, or the gate fails |

**Open points for the lane.**
- **Other chemistries.** Rank 29 is for Sr, Ti and O. The prefix width has to come from the table at setup, per
  pair; the kernel's far width is then the largest over the pairs present.
- **The 5060 Ti.** `COMPOSE_SKIP` is off there, so its weights have no receiver-species dependence and a per-pair
  basis does not fit as is.
- **The list order.** The lab found the keys sorted only up to the neighbour list's device count. A per-edge branch
  does not depend on it; a partition into near and far launches would.
- **sm_80.** The A100's forward compiles to 128 registers because of its scalar atomics. The branch should not add
  to that.

## 2. One Modal H200 pack: `HYBRID` and `GEOM`

| Switch | State | Expected on the H200 | Stop rule |
| --- | --- | --- | --- |
| `tf32x3.HYBRID` | Default on the A100; the sm_90 cubin exists; not measured on the H200 | About 2 % if the A100's lab ratio (0.873) carries over; GEMMs are 560 µs there | Not faster in both orders, or the gate fails |
| `spline_forward.GEOM` | Default on the A100 (0.976 / 0.981); slower on the Ada (1.02); listed as not measured on the H200 | 1–2.5 % if it carries over | The same |

`GEOM` belongs here because the second outside idea (below) has the same operation count as `GEOM` and would need
new kernels. The existing switch answers the question first.

## 3. Packed jobs still open from the first queue

| # | Item | Resource | Stop rule |
| --- | --- | --- | --- |
| B2 | `BINNED_SCREEN` on the RTX 4000 Ada, SrTiO3 16×16×16 and 20×20×20 | leto or murphy after the idle check | Slower in either order |
| B3 | Receiver caps 32,768 and 65,536 for the A100's host-state and wavefront tiled calls | Alex, `tiled_ab.py --b-receivers` | Within noise |
| B4 | Shuffled against ASE-ordered atoms at 24×24×24 | Local 5060 Ti, one run | Shuffled within 3 % |

I found no record of these on main. They may be queued with the conductor.

## 4. Only after a result above

| Item | Depends on | Note |
| --- | --- | --- |
| All three GEMM products in one FP16 pass | `HYBRID` on the H200 | 1.5 tensor instructions per step against `HYBRID`'s 2. The A100 lab gained 13 % from 3 → 2, so expect less than the count suggests. |
| Compact angular products (second outside idea) | `GEOM` slower on the H200 | Build only if a census then shows at most 80 registers. |

## 5. The capacity path

Unchanged from the first queue: CUDA-graph tile bodies for host state and wavefront (first test: Nsight Systems
idle share of an A100 wavefront call), 512 state columns, kept lists as stencil bitmasks, harmonics inside the edge
kernels. The A100 397.5M-atom rerun with the heap trim is still pending. A new lane, `fastmace/billion-atoms`,
started today; I have not read it.

## Review of the three new outside ideas

Note: `D:\Margraf\inference-more-20261111\ideas.md` (outside the checkout). I checked it against today's lab and the
projection code; I did not rerun its probes.

- **Idea 1, nested radial basis:** sound; section 1 above.
- **Idea 2, angular products in a compact basis:** the note's own count is 66 against `GEOM`'s 64 for the four
  paths. It needs 52 accumulators in a forward that already sits at its 80-register cap, and it widens each sender
  vector from three components to ten (18 MB more to gather at 5,000 atoms). Measure `GEOM` on the H200 instead.
- **Idea 3, one shared denominator:** it takes the series from six coefficients per column to five. Today's
  falsifier removed five of the six loads and saved 168 µs on the A100 and an estimated 33–49 µs on the H200, so
  removing one is worth about 34 µs and 7–10 µs: under both bars. Its screen also used 3e-7, looser than the
  table's 1e-7, and 0.15 % of the pieces still failed. Stop at the bound.

## Closed today (not to be retried without new evidence)

Half-precision far edges; half-systems on two streams; the formulation search; the wavefront row pool on the
device; the texture idea; C1 to C5 above; the shared denominator (bound). The 43 older rejections reviewed in
`plans/notes/rejected-ledger-20261111/review-old-and-resources.md` also stand.

## What today's labs established (for the next idea)

- **w′ on the A100:** about 85 registers and 27 KB of shared memory per block, or it drops from 6 to 5 blocks per
  SM. The spline forward runs at exactly 80 registers.
- **First interaction's forward on sm_80:** 64 scalar atomics and 128 registers; without the atomics 72 registers.
  No correct kernel avoids writing the 61 MB of slot blocks.
- **First table:** values pack (1,402 pieces at 1e-7, 6.4 MB); slopes do not.
- **Pairing edges in the forward** costs a 27-instruction header per edge before it saves anything.
- **GEMMs** follow their tensor-instruction count only partly: 3 → 2 gave 0.873, not 0.67.

## Housekeeping

- Copy `D:\Margraf\inference-more-20261111\` into `plans/notes/`, as was done for the first outside note.
- Ledger: a row for the nested basis (open), one for the compact angular products (waits on `GEOM`), one for the
  shared denominator (stopped by the bound). In the working tree I read, the index rows of ideas 18–20 and the two
  outside ideas (lines 638–642) still said OPEN; they should carry today's stops.
- This file and `plans/notes/conductor-prompt-20261111-next-steps-rev2.md` are untracked.
