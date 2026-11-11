# Review of two flagged subsets of the rejected-optimisation catalogue (11 November 2026)

Source: an owner request (11 November 2026) to review the 25 rows "timed before 5 November whose switch still exists"
and the 20 rows flagged `resources` in [summary.md](summary.md). 43 rows, 2 in both. Read: each row's ledger entry,
the linked report where the entry was thin, and `docs/current/defaults-by-gpu.md` (main `b951e804`).
**Nothing was run.** Verdicts are mine; sizes marked "my estimate" are not measured.

## Result

- **None of the 43 rejections needs to be reopened as it stands.** 19 sit on code paths that today's OMAT-0 default
  no longer runs or were adopted later in another form, 22 lost for a reason that does not depend on the base, and
  2 were tested too narrowly to carry their conclusion.
- **The old switches are not a re-screen pool.** BUTTERFLY flipped on 9 November because w′ around it changed. The
  switches rejected before 5 November mostly belong to the fused forward, the stored per-edge weights and the first
  interaction's old table path, which SPLINE_W, TABLE_W and TABLED_FORWARD replaced on all four GPUs.
- **One side finding is worth more than the two lists:** `BINNED_SCREEN` is off on the RTX 4000 Ada and the
  RTX 5060 Ti, the two GPUs that run the large whole-system cells (next section).

## Side finding: the O(N²) neighbour screen on the Ada and the 5060 Ti

Found while checking `omat0-atom-order`, whose reason ("x stays in L2") holds only at 5,000 atoms.

- **The record.** The rebuild's tile screen is O(N²). On the Ada it takes 1.26 ms at 5,000 atoms, 17.8 ms at 20,480
  and 201 ms at 69,120 (`results/ideas-queue2-20261106/README.md`). `BINNED_SCREEN` replaces it with a cell list:
  201 → 1.26 ms at 69,120 atoms, the same candidate list bit for bit.
- **The gap.** It is on by default for the H200 and the A100 only (`BINNED_SCREEN_GPUS`), from 10,000 atoms. For the
  Ada and the 5060 Ti the defaults table says "not measured". The Ada and 5060 Ti screen of 10–11 November could not
  see it: its cells have 5,000 atoms, below the switch's threshold.
- **Where it costs.** The whole-system capacity runs. Per-atom step time grows with size before any rescue rung is
  taken (`results/omat0-ada-capacity-20261108/README.md`): Ada SrTiO3 4.91 µs/atom at 69,120 atoms and 5.48 at
  148,955; Ada NaCl 3.29 at 125,000 and 4.20 at 314,432.
- **Size (my estimate).** Scaled as N², the screen is about 0.93 s per rebuild at 148,955 atoms and about 3.7 s at the
  Ada's 296,595-atom limit. At one rebuild per 18 steps (the 5060 Ti's rate at 5,000 atoms; larger cells rebuild no
  less often) that is about 6 % of the step at 148,955 atoms and 10–18 % at the Ada's limits (SrTiO3 296,595, NaCl
  314,432 atoms). It explains a third to a half of the measured per-atom growth; the rest is not traced.
- **First test.** `BINNED_SCREEN=True` against the default on the Ada and the 5060 Ti at SrTiO3 16×16×16 and one
  size near 30×30×30. The switch is bitwise, so one not-slower A/B decides it (decision 66).

## Two rows tested too narrowly

| ID (ledger line) | What was tested | What it does not cover | Worth doing |
| --- | --- | --- | --- |
| `omat0-atom-order` (497) | Fused forward at 5,000 atoms on the 5060 Ti: ASE repeat order 6,724 µs, Morton 6,737, shuffled 6,679 | Cells whose x no longer fits L2 (x is 307 MB at 150,000 atoms). In ASE repeat order the senders of consecutive receivers stay within about 8–14 MB (my count), so benchmark crystals should not gain. A shuffled or species-sorted input would read every sender row from DRAM. | One run: repeat order against a shuffled copy at 24×24×24 on the 5060 Ti. If shuffled is clearly slower, sort atoms spatially at setup. |
| `omat0-cutlass-tile` (464) | 32×64 and 64×32 CUTLASS tiles on the A100 when the products were 5,000 × 128 (158 blocks on 108 SMs): 1.012–1.017 | The H200 (132 SMs), and today's product shapes, most of which have 35,000 rows and many waves | Nothing: the wave effect it aimed at is gone for the large shapes, and the bound is under 1 %. |

## The 25 old rows

| Verdict | Rows | Why |
| --- | --- | --- |
| Adopted later in another form | `omat0-slope-vector` (456), `conv-receiver-backward` (200) | The q-major table with vector loads is `SPLINE_VECTOR` (4 on the H200 and A100, 2 on the Ada and 5060 Ti). Receiver-owned edges came back as w′ `STAGE_DOUT`: 32-edge blocks walk runs of one receiver and stage its dout row once. |
| Path not in today's default | `omat0-stream-w` (457), `omat0-stream-store-w` (461), `omat0-tc-presplit` (465) | They tune the stored per-edge weights and the fused forward's tensor-core product. `SPLINE_W` and w′ `TABLE_W` are on for all four GPUs, so OMAT-0 medium stores no w and runs no fused forward. |
| Path not in today's default | `omat0-first-full-rows` (452), `omat0-column-block` (453), `omat0-eos-chunks` (462), `omat0-first-table-columns` (473) | They tune the first interaction's radial-table and `edge_outer_sum` path. `TABLED_FORWARD` + `FUSED_FIRST_BACKWARD` are on for all four GPUs. All four were ties or slower even then. |
| Path not in today's default | `omat0-species-gemm-tc` (467), `omat0-species-gemm-cutlass` (468) | On the A100 the species skip now runs as 3×TF32 under `COMPOSE_SKIP`; the species GEMM is off there and on for the Ada and 5060 Ti. |
| Path not in today's default | `rad-contiguous-folded-weight` (156), `rad-joint-fused-backward` (158) | Research-engine probes of the radial MLP, which the radial tables replaced. |
| Holds, independent of the base | `lin-inplace-block-linear` (90), `lin-split-k` (102), `prec-bf16x3-facto` (266), `omat0-wprime-aprime` (472) | Slower for a structural reason: cuBLAS batched tiles; an extra launch for a 0.3 % ceiling; BF16 tensor cores only 3–5× FP32 with 7–19× the error; 61 GFLOP of new per-edge products against a slope that has since halved. |
| Holds, by rule | `lin-gemm-autotune` (94), `prec-tf32` (255) | Per-shape tuning is excluded by the production rule; TF32 fails accuracy by 300–1,500× (a closed direction). |
| Holds, ties | `eng-bucket-width` (347), `fp32-followups-20261029` (227, 337, 374, 375, 376) | Launch constants, bucket widths and rebuild options within about 1 % on MH-1 cells. Padded edges are skipped at the top of the edge loop, so narrower buckets have little to save. |

## The 20 resource losses

| Verdict | Rows | Why |
| --- | --- | --- |
| Tested again on the 9 November base, still slower | `omat0-wprime-maxreg-a100` (484), `wprime-smem-spill` (557) | A100 w′ capped at 72 registers 1.095, at 64 1.300; H200 capped 1.18–3.0×. Any cap spills. |
| The trade loses, not the layout | `wprime-receiver-adjoint` (575) | It removes 14 FP32 instructions per channel step and adds 43 shared loads (FFMA 290 → 280, FMUL 57 → 53, LDS 52 → 95). The compact layout that was not built still drops w′ from 6 to 4 blocks per SM. |
| The trade loses, not the layout | `fp64-second-pipe` (569) | Doubles take register pairs. w′ runs 128-thread blocks at 80 registers, 6 per SM; by my arithmetic 85 registers still fit 6, and the smallest offload needs 96. Free conversions change nothing (1.178 against 1.186). |
| The trade loses, not the layout | `deferred-cg-forward` (567), `cx-n1-cartesian-wprime` (600), `rad-k64-addr` (167), `conv-chunked-recompute` (208), `lin-split-k` (102) | 76 accumulators cannot live in 80 registers (the selective form became `GEOM`); a larger hot loop for fewer registers; a rank-1 update that rewrites the whole output; an extra GEMM for less traffic. |
| Adopted later in another form | `omat0-wprime-piece-order` (469) | Sorting edges by table slab lost the receiver order (1.513). `SORT_BY_LENGTH` sorts within each receiver and is on for all four GPUs. |
| Fused forward only, not in today's OMAT-0 default | `omat0-tc-presplit` (465), `omat0-fp16-tc` (477), `omat0-hybrid16` (487), `ideas-r9` (504), `ideas-r11` (505), `kr-x-nodep` (514) | Warp counts, register caps and x gathers of the tensor-core forward. They would matter again only for models that keep the fused forward (POLAR-1, MACE-OFF), where none was measured. `ideas-r9` also holds a persistent w′ variant: 0.987 on the H200 and 1.113 on the A100 against the 6 November kernel, under the lab bar. |
| Holds, closed direction or moved baseline | `prec-setmaxnreg` (290), `ext-lunatp` (1011) | The sm_120 hybrid GEMM layout was inferior regardless of registers. LunaTP lost on 27 October against kernels that have since gained several rounds; the A100 and Ada were never measured. |
| Tested too narrowly | `omat0-atom-order` (497), `omat0-cutlass-tile` (464) | See above. |

## What the resource rows say about w′ on the A100

Every w′ idea in this list died on the same two limits, so they are worth stating once (my reading of
`results/other-ways-20261109/README.md` §5 and `results/packed-table-lab-20261109/README.md`):

- **Registers:** 80 used by 128-thread blocks; about 85 keep 6 blocks per SM (my arithmetic); 96 drop to 5 and cost
  more than a second FMA pipe gives.
- **Shared memory:** 23 KB per block used; about 27 KB keep 6 blocks per SM (164 KB / 6).

A new w′ variant that needs more than about 5 registers or 4 KB per block starts with an occupancy loss of a sixth.
This also bounds the one-kernel form of the half-precision far-edge idea: its lab kernel fitted in 80 registers
without spills, the production build did not.

## Ledger rows whose status no longer matches the code or the record

- `omat0-slope-vector` (456): REJECTED in the index; the idea is the default `SPLINE_VECTOR`.
- `omat0-stream-w` (457), `omat0-stream-store-w` (461), `omat0-eos-chunks` (462): REJECTED in the index, PENDING in
  their entries; all three are moot under today's defaults.
- `BINNED_SCREEN`: "not measured" for the Ada and the 5060 Ti in `docs/current/defaults-by-gpu.md` §0.2 and listed
  under "Open: not measured per family".
