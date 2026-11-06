# Research answers: mirror-pair w′, split-domain tables, wave-aware tile rule (5 November 2026)

Two research answers to the H200 prompt (`reports/2026-11-05-h200-research-prompt.md`) left four ideas worth a cheap
falsifier. This covers all four. **None is adopted.** One layout (the w′ kernel over edges stored in mirror order)
stays open, with a bound of about 2 % of the H200 step after copy costs.

**Workload:** OMAT-0 medium, float32 hybrid, SrTiO3 10×10×10 (5,000 atoms, 373,760 edges; Benchmark_loup's cell).
**GPUs:**
- Modal profile `third` (fzwingel54): H200 and two A100-SXM4-80GB cards (the Modal hybrid image);
- local RTX 5060 Ti (`mace_cueq_oeq`, H200 plan through `tools/bench/gpu_alias`).

**Tools:**
- `tools/omat0_kernel_lab.py --kernels wprime` (`30c0df85`, `382a85d4`, `7a7b3a76`);
- `tools/bench/tile_rule_bench.py`;
- `tools/radial_degree_lab.py`.

## Outcome

| Idea | Stop rule | Result | Verdict |
| --- | --- | --- | --- |
| Mirror-pair w′: share w and w′ between an edge and its reverse | drop if sharing ≥ 0.93 of base | sharing adds nothing beyond the order's cache reuse; the exact permuted kernel is 0.95 (H200) and 1.04–1.06 (A100) | sharing dead; physical mirror order open |
| Pair-shared r-only kernels | only if mirror pairing works | pairing gives no arithmetic saving, and 0.56 % of the edges have no mirror | dead |
| Split-domain tables (degree 3/4 on [1.0 Å, r_max)) | continue if degree 4 needs no more pieces than degree 5 | degree 4: 3.1× (int. 0) and 3.2× (int. 1) the pieces; degree 3 hits the piece cap | dead |
| Wave-aware tile rule for the CUTLASS GEMMs | drop below 30 µs per force call | ≤ 5.6 µs on the H200 | dead |

## Mirror-pair w′

`SYMMETRIC_PAIRS` already makes an edge (j → i) and its mirror (i → j) read the same table row. sh differs only by
the sign (−1)^l. The answers proposed computing w and w′ once per mirror pair.

**Lab variants** (each a ratio of w′ kernel time to `base`):
- `B-mirror-order`: every per-edge array physically reordered so that mirrors are adjacent. Key: (min, max) of the
  two atoms, stable. Timing bound only; the outputs come out in that order.
- `B-mirror-share`: the same order, with the value table and Horner skipped on odd edges. This bounds sharing within
  a pair.
- `F-half-table`: the table skipped on odd edges, in receiver order.
- `F-no-table-horner`: the table skipped on every edge.
- `M-mirror-perm`: the exact form. The data stays in receiver order, and warp k processes edge `perm[k]`. dsh and
  dL/dr are bitwise; dx differs by the atomics' order (≤ 4.4e-7 relative).
- `M-mirror-hybrid`: the 4-byte edge arrays (senders, receivers, r, pair, dr) physically in mirror order, with only
  the 64-byte sh/dsh rows read through `perm`.
- `B-sender-order`: the edges sorted by sender (RTX 5060 Ti only).

**Mirror structure:** 185,831 atom pairs have both edges and 2,098 have only one (0.56 % of the edges, the cutoff
boundary), so a kernel cannot assume strict pairs.

| Variant | H200 | A100-SXM4 (card 1 / card 2) | RTX 5060 Ti |
| --- | ---: | ---: | ---: |
| base (µs) | 1,564 / 1,596 / 1,570 | 3,697 / 3,149 | 6,935–6,944 |
| B-mirror-order | 0.897 / 0.887 / 0.899 | 0.852 / 0.961 | 0.893–0.895 |
| B-mirror-share | 1.197 | | |
| F-half-table | 1.190 | | 0.807 |
| F-no-table-horner | 0.737 | | 0.576 |
| M-mirror-perm | 0.947 / 0.953 | 1.038 / 1.057 | 0.852–0.855 |
| M-mirror-hybrid | 0.951 | 1.037 | 0.857 |
| B-sender-order | | | 1.267 |

The H200's `B-mirror-share` and `F-half-table` compile to 93 registers. Their branch around the table costs more on
Hopper than it saves (1.19–1.20), so on that GPU only `F-no-table-horner` bounds the table: 26 % of the kernel. On
the RTX 5060 Ti, `B-mirror-share` (0.893, first run) equals `B-mirror-order`.

What this shows:
- **Sharing buys nothing beyond the order.** In mirror order, the partner's table row and lengths are already cache
  hits, so skipping its table work saves nothing. Sharing the arithmetic, or a pair-shared kernel, has nothing left
  to gain.
- **The gain is the locality of the sh/dsh rows.** Reading them through a permutation loses it on the datacenter
  GPUs, and the hybrid shows the edge scalars are not the cost. The RTX 5060 Ti alone gains from the exact form
  (0.855).
- **The open option is to store the w′ inputs in mirror order:**
  - What it takes: gather sh (and r, pair, senders, receivers) into mirror order once per force call, run the w′,
    and scatter dsh and dL/dr back. The permutation is built once per neighbour-list rebuild.
  - Bound on the H200: about 160 µs per force call from the kernel, less about 35 µs for the 64-byte-row copies
    (4 × 24 MB). That is about 2 % of the 5.5 ms step.
  - On the A100 the gain varies too much between cards (0.852 and 0.961) to plan on.
  - Not built yet.
- **Building the permutation per call costs too much.** The key build and int64 sort take 290 µs on the RTX 5060 Ti
  (int32 sort alone: 131 µs), so the permutation must come from the neighbour-list rebuild.

## Split-domain tables

`tools/radial_degree_lab.py` fits each interaction's tables with `fastmace.rewrites.radial.fit` at degrees 3, 4 and
5, on the default domain [0.3 Å, r_max) and on [1.0 Å, r_max), and certifies them (`certify`, CPU arithmetic):

| Interaction | Domain | Degree 3 | Degree 4 | Degree 5 (default) |
| --- | --- | --- | --- | --- |
| 0 | [0.3, 6) | 4,096 pieces (cap), fit error 1.5e-3 | 2,120, cert. 0.31 | 893, cert. 0.33 |
| 0 | [1.0, 6) | 4,093, fit error 4.6e-4 | 1,636, cert. 0.19 | 533, cert. 0.19 |
| 1 | [0.3, 6) | 4,096 (cap), fit error 1.7e-4 | 1,687, cert. 0.31 | 612, cert. 0.32 |
| 1 | [1.0, 6) | 4,096 (cap) | 1,230, cert. 0.20 | 382, cert. 0.22 |

- **Degree 4 saves no loads:** 5 terms pad to the same 6 as degree 5 (float2 rows), and it needs about three times
  the pieces (table 22 + 17 MB instead of 7 + 5 MB).
- **Degree 3 fails:** it reaches the piece cap without reaching the tolerance.
- **Side result:** degree 5 on [1.0 Å, r_max) needs 40 % fewer pieces. The table falls from 20 MB to 12 MB, which
  would matter only for an L2-limited GPU, and it needs a separate path below 1.0 Å.

## Wave-aware tile rule

`tools/bench/gemm_shapes.py` records every CUTLASS 3xTF32 GEMM of two force calls. `tools/bench/tile_rule_bench.py`
times each shape with the 64×64, 32×64 and 64×32 tiles under CUDA graphs, in alternating rounds, and checks that all
three are bitwise. On the H200 (132 SMs), choosing the best tile only where the 64×64 grid fills fewer than two waves
saves 5.56 µs over the recorded calls (two force calls). The large-M shapes are 2–13 % slower with the smaller tiles,
except one: M 15,000, K 384, N 128, rc (0.87 with 64×32), which already fills 3.6 waves.

## Runs (`runs/`)

| Directory | What |
| --- | --- |
| `mirror-h200/` | H200, `30c0df85`: lab base, B-mirror-order, B-mirror-share, F-half-table, F-no-table-horner; `shapes.json`, `tiles.json` |
| `perm-h200/`, `perm-a100/` | `382a85d4`: base, M-mirror-perm, B-mirror-order |
| `hyb-h200/`, `hyb-a100/` | `7a7b3a76`: base, M-mirror-hybrid, M-mirror-perm, B-mirror-order |
| `rtx5060ti/` | the local lab runs (H200 plan by alias) and `radial-degree.json` |

Modal cost on profile `third`: these five jobs ran with $8.88 metered at the start; it was $9.33 before the last
two.

## Ideas session, revised H200 plan (6 November 2026)

The ideas session's revised plan (`plans/h200-ideas-20261105.md` at `29d88d15` on its branch) proposed ideas 10–13
from a CPU-side SASS census of the H200 kernels. Each got a cheap falsifier on the H200 (Modal `third`) first.

| Idea | Falsifier | H200 | RTX 5060 Ti (H200 plan) | Verdict |
| --- | --- | ---: | ---: | --- |
| 10: de-phase the forward's warps | F-skew: odd warps start after `__nanosleep` (bitwise) | 1.094 (100 ns), 1.103 (300 ns), 1.114 (1 µs) | 1.043 (100 ns) to 1.534 (16 µs) | dead: any skew costs |
| 12: receiver-staged dout in the w′ | FF-dout-one-row: every edge reads receiver 0's dout | 1.001 | 0.999 | dead: dout locality is worth nothing |
| census: the forward's W split | F-wfrag-no-wsplit: W fragments used unsplit | 0.853 (H200 base 1,685 µs) | 0.982–0.993 | bound only, see below |
| W split stored once | `fused_forward.TC_WFRAG_SPLIT` (lab `tc-wfrag-split`), bitwise | 1.070 | 1.144 (parts 4 instead of 2: 100 KB shared memory) | rejected (off) |
| 11: W-stationary wgmma | register arithmetic | | | dead without a build |
| 2, 4, 7 (conditional) | the earlier bounds | | | decided before: 2 adopted (DEGREE_PARTS), 4 and 7 dropped |

- **W split, stored once (`TC_WFRAG_SPLIT`):**
  - What: the wfrag slice holds every fragment as a float4 of high parts and a float4 of low parts per lane, so the
    batches load two float4 and run no split instructions.
  - On the H200 it keeps the plan's launch rule (2 parts, 12 warps, 33 blocks per slice; 224 KB instead of 175 KB of
    shared memory) and is bitwise, yet takes 1.070 of the forward.
  - So the second shared load per fragment costs more than the LOP3 + FADD it saves: the forward is limited by
    shared-memory loads, not by the split's ALU work.
  - `tests/gpu/test_fused_folded_forward_cuda.py` checks it bitwise (passed on the RTX 5060 Ti). The switch stays
    off, for A/Bs.
- **Larger batches (W reuse over 16 edges):** not viable. The product's mma tiles hard-code 8 edges per fragment.
  A second n-tile doubles the accumulators past the 168 registers that 12 warps allow, the same warp loss that sank
  WGMMA.
- **Idea 11 (W-stationary wgmma): dead on register count.** One part's W slice is about 5 paths × 64 × 32 floats per
  warp, roughly 320 registers per lane before the high/low split.

Runs: `runs/ideas-r2-h200/` (`2472c0e0`: F-skew, FF-dout-one-row), `runs/ideas-r3-h200/` (`614f4779`:
F-wfrag-no-wsplit), `runs/ideas-r4-h200/` (`7e4da9e9`: tc-wfrag-split), `runs/rtx5060ti/ideas-lab/`. Modal `third`
stood at $9.90 metered before the last job.

## Ideas session, third revision: w once per mirror pair, lane-parallel index loads (6 November 2026)

The third revision proposed two ideas:

- **Idea 14:** compute w = a·Wᵀ once per mirror pair. Both edges of a pair have the same length, so they have the
  same radial input.
- **Idea 15(a):** load a batch's receiver and sender indices with lanes 0..B−1 at once and broadcast them by
  shuffle, so the x gathers stop waiting on a serial chain of index loads.

One H200 job (Modal `third`, `d8c8fbc4`) ran the falsifiers. A second job at the same commit re-ran the
dense-product bench, because the first job lost its output to an unescaped `$OUT`.

| Check | H200 | Reading |
| --- | ---: | --- |
| Mirror lengths (`mirror_order`) | 185,831 pairs, 100 % bitwise-equal, largest \|dr\| 0; 2,098 edges unpaired | idea 14's premise holds |
| Forward base (lab) | 1,673.4 µs | |
| F-tp-stored-w (the TP reads w per edge, no a·W) | 932.1 µs (0.557) | the forward's floor with w given |
| F-no-tp | 1,142.2 µs (0.683) | |
| Pair product `[186,880, 64] @ [64, 1,280]`, CUTLASS 3xTF32 (`tools/bench/pair_w_bench.py`) | 750 µs (rel. err 3.4e-7) | FP32 1,001 µs; a copy of w alone 451 µs (4.24 TB/s) |
| **x-lane-idx** (idea 15a, bitwise) | **1,617.2 µs (0.966)** | RTX 5060 Ti (H200 plan) 1.024 |

- **Idea 14 is dead.**
  - The pair-shared forward needs the pair product plus the stored-w forward: 750 + 932 = 1,682 µs, against
    1,673 µs today.
  - Writing and re-reading 957 MB of w costs what halving a·W saves.
  - No ordering of the two steps gets under the HBM floor: copying w alone takes 451 µs.
- **Idea 15(a) is adopted on the H200.**
  - Switch: `fused_forward.LANE_INDICES` (None: `LANE_INDICES_GPUS`), applied as a transform of the rendered kernel
    (`lane_indices`). Not with WGMMA.
  - Correctness: bitwise on the SIMT and the tensor-core kernels (`tests/gpu/test_fused_folded_forward_cuda.py`,
    passed on the RTX 5060 Ti).
  - MD-step A/B (`37b6dd66`; OMAT-0 medium, SrTiO3 10×10×10; `hopper_sweep.sh` AB `laneidx`):
    - **0.9897** (5.6688 → 5.6107 ms) in one build order;
    - **0.9913** (5.6777 → 5.6281 ms) in the other.
  - So under decision 66 it is the default for `NVIDIA H200` only.
  - The RTX 5060 Ti lost in the lab (1.024) and stays off. The A100 is not measured.

Runs: `runs/ideas-r5-h200/` (lab falsifiers, mirror check), `runs/ideas-r6-h200/` (pair product),
`runs/laneidx-h200/` (MD A/B). Modal `third` stood at $11.57 metered before the A/B.

## Ideas session, fifth revision: owner-pass forward and shared W fragments (6 November 2026)

The fifth revision (`fastmace/h200-ideas-20261105`, `433b4e19`) reopened idea 14 as 14(A) and ranked idea 17 next.

- **14(A), the owner pass:**
  - Each mirror pair runs the fused forward once, for its owner edge, and stores w into its partner's slot.
  - A second, product-free pass then serves the partner edges from the stored w.
- **Idea 17:** a warp group of G warps shares each W fragment across its G batches.

Three lab variants (`74838161`, Modal `third`) decide whether to build them:

- `store-w`: the base with STORE_W on, bitwise.
- `F-w-half`: both m-tiles read one W fragment. That is G = 2's W traffic, without its extra a loads and barriers.
- `F-w-quarter`: additionally, paired path groups read one W fragment. That is G = 4's W traffic.

The base includes LANE_INDICES on the H200.

| Variant | H200 | A100-SXM4-80GB |
| --- | ---: | ---: |
| base | 1,632.9 µs | 3,107.6 µs |
| store-w | 1.102 | 1.059 (20 B spill) |
| F-tp-stored-w | 0.506 | 0.613 |
| F-w-half | 0.878 | 0.876 |
| F-w-quarter | 0.817 | 0.813 |
| **14(A) estimate: ½ store-w + ½ F-tp-stored-w** | **0.804** | **0.836** |

Both falsifiers clear the plan's thresholds on both GPUs.

- **14(A): build.**
  - The threshold is ≤ 0.85. The H200 estimate of 0.804 is about −320 µs per forward, before the second launch and
    pass 2's message reads.
  - It is rounding class, so a default needs the force/stress gate as well as the MD A/Bs.
- **Idea 17: build, starting at G = 2.**
  - G = 2: F-w-half is 0.878, below the 0.92 threshold.
  - G = 4: F-w-quarter is 0.06 below F-w-half, above the 0.04 margin, so G = 4 is worth building too. It needs
    idea 18's registers.
  - Idea 17 is bitwise, so a default needs only the MD A/Bs.
- **Order:** idea 17 first. It is a render change inside one kernel, and it makes 14(A)'s pass 1 cheaper. 14(A)
  follows, with the rebuild's pairing, the second pass and the gate.

Runs: `runs/ideas-r7-h200/`, `runs/ideas-r7-a100/`. Modal `third` stood at $12.19 metered before these two jobs.

### Idea 17 at G = 2: `W_GROUP` (6 November 2026)

**What it does.** `fused_forward.W_GROUP` (None: `W_GROUP_GPUS`) pairs the warps of a block.
- In the k-loop, each warp of a pair loads and splits one W m-tile once. It uses the tile for both warps' batches,
  taking its partner's a fragments from the partner's `a_s`.
- It writes its C fragments into the owning warp's `r_s`.
- Each pair has one named barrier id, used twice per batch:
  - a `bar.red.or` vote after staging, which also tells whether the pair still has work;
  - a `bar.sync` after the `r_s` writes.
- A warp that runs out of chunks joins its partner's k-loops with empty batches.
- The result is bitwise.

**Where it can be built.** It needs:
- the TF32 product with `wfrag`;
- no TC_TP and no WGMMA;
- an even warp count.

**Checks.** `tests/gpu/test_fused_folded_forward_cuda.py` tests it on an uneven neighbourhood, natively on the RTX
5060 Ti and under the H200 alias. sm_90 at 12 warps: 168 registers, no spills. The background build was commit
`429cebe3`.

| | H200 | A100-SXM4-80GB |
| --- | ---: | ---: |
| lab forward, base | 1,631.9 µs | 3,805.2 µs |
| lab `w-group` | **0.980** (bitwise) | not built: the plan runs 11 warps |
| lab F-w-half (bound) | 0.878 | 0.878 |
| MD step (b = on), both build orders | **0.9946 / 0.9942** (5.862 → 5.830, 5.873 → 5.839 ms) | 0.9997 / 1.0018 (switch inert) |

**What follows.**
- **Default on the H200** (decision 66). Off elsewhere.
- **The real G = 2 keeps about a sixth of its bound.** The extra a loads (2 LDS.32 per n-tile and k-step) and the
  two barriers per batch take the rest. G = 4 would double both, so it is not worth building without a different a
  path.
- **The A100-SXM4** would need an even launch rule (10 or 12 warps) to try it. Not measured.

Runs: `runs/wgroup-h200/`, `runs/wgroup-a100/`.

### H200 defaults after round 6: gate and Benchmark_loup at `7f7ef7db` (6 November 2026)

Main `7f7ef7db` changes two H200 defaults: `W_GROUP` is on, and `BUTTERFLY` is off (kernel-review session,
`results/sx012-profile-20261106`). One H200 job (Modal `fourth`) checked the result.

**Force/stress gate:** passes. Ratios to the float32 calculator's error, with each limit:

| Crystal | Energy (absolute floor 10 meV) | Forces (limit 2) | Stress (limit 2) |
| --- | ---: | ---: | ---: |
| NaCl | 3.00 (0.39 meV) | 0.76 | 0.92 |
| NbCl5 | 3.82 (0.90 meV) | 0.98 | **1.96** |
| SrTiO3 | 11.5 (3.1 meV) | 0.94 | 0.93 |

- NbCl5 stress has the least headroom: 1.96 of 2, against 1.93 at `62556704`. Any change of summation order
  (14(A)) must recheck it.

**Benchmark_loup (Langevin, OMAT-0 medium, SrTiO3 10×10×10):** **5.495 ms per step, 1.099 µs per atom and step.**

- On 5 November (`62556704`) it was 5.73–5.78 ms.
- Since then: DEGREE_PARTS, LANE_INDICES, W_GROUP, BUTTERFLY off, and the round-5 changes.

Run: `runs/main-7f7ef7db-h200/`.

### Round 7: owner pass rejected, drift audit, lab at `bcdeb501` (6 November 2026)

**Idea 14(A), the owner-pass forward, is rejected.**

- **Where it is:** built behind `OWNER_PASS` on branch `fastmace/owner-pass` (`f104e196`), CPU suite green. It is
  off by default, and the branch is parked, not merged.
- **MD-step A/Bs** (SrTiO3 10×10×10, b = owner pass):

  | GPU | b/a, order ab | b/a, order ba |
  | --- | ---: | ---: |
  | H200 | 1.0445 | 1.0460 |
  | A100-SXM4 | 1.0664 | 1.0656 |

- **Gate (H200):** fails. NbCl5 stress is 2.076 of 2, because the owner order changes the summation order.
- **Lab split (H200, `--owner-pass`, 15 rounds):**
  - The message is bitwise against the one-pass kernel in owned-first order.
  - One pass on the owned-first order: 1594.5 µs (0.996 of 1601.2).
  - Pass 1: 1078.8 µs (0.674). Pass 2: 660.3 µs (0.412). Both: 1737.0 µs (1.085).
  - Pass 1 without its w stores: 982.0 µs, so the stores cost ≈ 97 µs.
  - Pass 2 without its w loads: 445.0 µs, so the reloads cost ≈ 215 µs.
  - Even with both free, the two passes would take 1427 µs (0.89). Storing and reloading w costs more than the
    k-loop work it saves.
- **Idea 19** (w′ in mirror-pair order) needed 14(A)'s pairs, so it is closed with it.

**Drift audit at main `451143be`.** Every per-GPU default was re-measured as an MD-step A/B in both build orders
(b = the switch flipped against today's default).

- **H200:** all 21 switches still hold.
  - Strongest, with the switch off:
    - no block messages: 1.076 / 1.079;
    - no vector a: 1.074 / 1.070;
    - no species skip: 1.070 / 1.068;
    - no fused first interaction: 1.068 / 1.070;
    - no Horner: 1.047 / 1.048.
  - Off switches stay off:
    - SH_LDS4, BACKWARD_WALK and SEGMENT_ROWS are ties (one order each way, within 0.5 %);
    - SPECIES_GEMM (1.009 / 1.005), TABLE_PAIRS (1.006 / 1.009), HOIST_SCALE (1.006 / 1.010) and BLOCK_LDG
      (1.009 / 1.011) are slower when on.
- **A100-SXM4:**
  - Still faster: DEGREE_PARTS (off: 1.010 / 1.002), HORNER (off: 1.033 / 1.030) and TC_WFRAG (off:
    1.023 / 1.015).
  - BUTTERFLY on is slower (1.029 / 1.018). LANE_INDICES ties (1.000 / 0.991).
  - No default changes.

**Kernel lab at `bcdeb501`.** Forward kernel, 15 rounds, Modal. All variants are bitwise unless their name starts
with `F-`, which marks a bound.

- **Storing w (the cost of 14(A)'s pass 1):**

  | GPU | store-w | store-w-cs (streaming stores) | store-w-v4 (16-byte stores) |
  | --- | ---: | ---: | ---: |
  | H200 | 1.157 | 1.289 | 1.149 |
  | A100 | 1.055 | 1.048 | 1.033 |

  None comes close to paying for itself.
- **W_GROUP's a path (H200):**
  - Each warp splitting its own a (F-wg-own-a): 0.971.
  - The same with its own a loads (F-wg-own-a-ld): 0.998.
  - Without the two named barriers (F-wg-no-bar): 0.979.
  - So a pre-split a in shared memory could gain up to ≈ 3 %, and the barriers cost ≈ 2 %.
- **A100 with a fragment-sized W region (idea 21):**
  - tc-warps12 (12 warps instead of 11): 0.958.
  - w-group-12 (W_GROUP at 12 warps): 0.905.
  - It is built as the `WFRAG_REGION` switch.
- **3xTF32 bias (idea 20):** the truncating split makes w's error proportional to w.

  | Variant | H200 slope | H200 time | A100 slope | A100 time |
  | --- | ---: | ---: | ---: | ---: |
  | today's split (instrumented) | −7.7e-7 | 1.157 | −1.05e-6 | 1.054 |
  | round-to-nearest split | −3.7e-7 | 1.260 | −5.2e-7 | 1.226 |
  | B1 correction | −4.5e-7 | 1.173 | −6.6e-7 | 1.057 |
  | SIMT FP32 | ≈ 0 | 1.587 | ≈ 0 | 1.615 |

  - The times include the instrument, which costs ≈ 1.16 on the H200 and 1.05 on the A100.
  - Against the instrumented split, rounding costs ≈ 9 % (H200) and 16 % (A100). B1 costs ≈ 1.5 % and 0.3 %.
  - The CUTLASS 3xTF32 GEMM has a similar slope (−1.6e-7 to −1.9e-7 at K = 128).
  - The gate passes today, so neither fix is adopted. B1 is the cheap one if a gate ever needs it.
- **ncu stall reasons for idea 18 (the compact TP):**
  - The first job could not lock the clocks on Modal (`LockingClocksFailed`).
  - `tools/modal/ncu_forward.sh` now passes `--clock-control none`. The rerun is pending.

Runs: `runs/owner-pass/`, `runs/drift-451143be/`, `runs/lab-bcdeb501/`.

### Round 8: A100 at 12 warps, A_SPLIT, ncu stall reasons (6 November 2026)

**`WFRAG_REGION` (idea 21) is default on the A100-SXM4, and with it `W_GROUP`.**

- **What it does:** with TC_WFRAG the W slice is read only in fragment order. So its shared region needs
  `CHANNEL_BLOCK` floats per (path, k), not `TC_STRIDE`. The smaller region lets the launch rule fit 12 warps on the
  A100-SXM4 instead of 11, and W_GROUP needs an even count.
- **Accuracy:** bitwise (GPU test).
- **MD-step A/Bs** (SrTiO3 10×10×10, b = switch on, commit `8aa8c0e7`):

  | GPU | Switch | b/a, order ab | b/a, order ba |
  | --- | --- | ---: | ---: |
  | A100-SXM4 | WFRAG_REGION | 0.9871 | 0.9887 |
  | A100-SXM4 | WFRAG_REGION + W_GROUP | 0.9718 | 0.9734 |
  | H200 | WFRAG_REGION | 1.0014 | 1.0010 |
  | H200 | WFRAG_REGION + W_GROUP (already on) | 0.9997 | 1.0004 |

- **On the H200 it is a tie.** The H200 plan has 12 warps already, limited by registers (168 × 384 threads), so the
  freed shared memory buys nothing. It stays off there.
- **Commit:** `ff4d1a89` (branch `fastmace/a-split`).
- **A100-SXM4 gate at `ff4d1a89`:**
  - Energy and forces pass. Ratios to the float32 calculator's error: forces NaCl 0.74, NbCl5 0.99, SrTiO3 1.05.
  - NbCl5 stress is 1.59e-7 eV/Å³ against a limit of 1.23e-7 (ratio 2.60). This is the plan's accepted failure from
    3xTF32 and the tensor-core forward (`docs/current/defaults-by-gpu.md`). Rounds 4 and 5 measured 1.57e-7 there,
    and both switches are bitwise.
- **A100-SXM4 Benchmark_loup** (Langevin, SrTiO3 10×10×10): **10.85 ms per step, 2.169 µs per atom and step**.

**`A_SPLIT` is rejected.**

- **Idea:** with W_GROUP each warp of a pair splits both batches' a into TF32 parts, so every a value is split twice.
  A_SPLIT splits it once, at staging, into the a tile (high parts) and a per-warp low-part tile. On the H200 that
  costs 24 KB more shared memory, which fits.
- **Accuracy:** bitwise (GPU test).
- **H200 MD-step A/B:** 1.0311 / 1.0346, slower.
- **Why:** each k-step now loads two shared values per a fragment instead of one plus a split. The forward is not
  ALU-bound there (next item), so trading ALU work for shared loads loses. The switch stays in the code, off
  everywhere.

**ncu stall reasons of the H200 forward** (base and F-skew-100ns, `--clock-control none`, commit `6868f0d1`):

- **Occupancy and issue:**
  - 12 warps per SM (18.75 % theoretical occupancy, limited by registers and shared memory), 2.91 active warps per
    scheduler.
  - One or more eligible warps in 53.9 % of cycles. 5.40 warp cycles per issued instruction.
- **Stalls per issued instruction:**

  | Stall | Cycles |
  | --- | ---: |
  | math pipe throttle | 0.50 |
  | long scoreboard | 0.37 |
  | short scoreboard | 0.26 |
  | barrier | 0.23 |
  | no instruction | 0.18 |
  | MIO throttle | 0.07 |

- **Idea 18 (a compact, rolled TP against instruction-fetch stalls) is closed.** `no_instruction` is only 0.18 of
  the 5.40 cycles (≈ 3 %), so its bound is about 3 % of the forward, before its own costs.
- **F-skew-100ns** changes nothing (0.999).
- **The largest stall is the math pipe** (HMMA and FP32). ncu counts 127.6 M fused and 197.3 M non-fused FP32
  instructions. Most of the non-fused ones are the 3xTF32 splits (`x − hi`), which cannot fuse.

Runs: `runs/round8/` (`rg-h200`, `rg-a100`, `as-h200`, `r11-ncu`, `gate-a100-ff4d1a89`).

### Round 9: more warps, part phases, x-load bounds, a persistent w′ (6 November 2026)

**Nothing adopted.** The round tested the H200 ideas plan's revision 9 (ideas 22, 23 and 25) with lab bounds on Modal.
Lab variants are in `tools/omat0_kernel_lab.py` (branch `fastmace/round9-lab`, commits `98851d50`, `ab76a9d5`,
`50ebcdf9`, `e97fd7e6`). The forward lab runs on SrTiO3 10×10×10, OMAT-0 medium, the GPU's own plan.

**More warps per SM (idea 22(ii)) do not pay, even without the x loads.**

| Variant (H200, base 1600.5 µs, 168 registers) | Registers | Spill stores | Ratio |
| --- | ---: | ---: | ---: |
| `wg12`: W_GROUP + fragment-sized region, 12 warps (bitwise) | 168 | 0 | 0.994 |
| `wg14` (bitwise) | 128 | 20 B | 1.029 |
| `wg16` (bitwise) | 128 | 52 B | 1.017 |
| `F-no-xload` (plan render) | 165 | 0 | 0.904 |
| `F-no-xload-wg12` | 165 | 0 | 0.905 |
| `F-no-xload-wg14` | 128 | 24 B | 0.915 |
| `F-no-xload-wg16` | 128 | 24 B | 0.900 |
| `F-l1-28` (shared request padded to 224 KB: 28 KB of L1; bitwise) | 168 | 0 | 0.995 |

- At 14 and 16 warps ptxas caps the kernel at 128 registers and spills. Without the x loads the extra warps still
  gain nothing over 12 (0.915 and 0.900 against 0.905). Idea 22(ii) is closed.
- Less L1 costs nothing (F-l1-28 0.995), so a larger carve-out is free if a later idea needs it.

**The x loads cost about 2–3 % on the H200, not 10 %.** F-no-xload replaces x by a constant, which lets the compiler
fold the TP's x products, so it overstates the loads. Two bounds keep x opaque: `F-x-row0` (every edge reads node 0's
row, an L1 hit) and `F-x-rcv` (each edge reads its receiver's row).

| GPU | Base | F-no-xload | F-x-row0 | F-x-rcv |
| --- | ---: | ---: | ---: | ---: |
| H200 | 1606.4 µs | 0.905 | 0.977 | 1.071 |
| A100-SXM4 | 3321.9 µs | 0.768 | 0.844 | 0.941 |

- F-x-row0 and F-x-rcv spill 24 B on both GPUs.
- **Atom order and the lab's L2 flush change nothing** (`tools/modal/order_lab.sh`, base and F-x-row0, 5 rounds):

  | GPU | Order | Base, flush 256 MB | Base, no flush | F-x-row0, flush / no flush |
  | --- | --- | ---: | ---: | ---: |
  | H200 | repeat | 1600.2 | 1613.9 | 0.982 / 0.985 |
  | H200 | Morton | 1582.7 | 1611.6 | 0.982 / 0.983 |
  | H200 | random | 1583.4 | 1612.6 | 0.982 / 0.988 |
  | A100 PCIe (SXM4 plan) | repeat | 2863.1 | 2886.7 | 0.977 / 0.975 |
  | A100 PCIe (SXM4 plan) | Morton | 2886.7 | 2919.4 | 0.975 / 0.971 |
  | A100 PCIe (SXM4 plan) | random | 2872.3 | 2900.0 | 0.974 / 0.974 |

  Round 4's Morton rejection (5060 Ti) holds on the H200 and the A100 PCIe: x stays in L2 in any order.
- **Modal's `A100-80GB` is either card.** This job and the w′ job ran on an A100 80GB PCIe with the SXM4 plan by
  alias. On it F-x-row0 is 0.977, against 0.844 on the SXM4 job, whose base was also slower (3,322 against 2,863 µs).
  A rerun on an SXM4 (after two PCIe draws) gave base 3,425.7 µs, F-x-row0 0.988 and F-no-xload 0.897. So the
  first SXM4 job's 0.844 came from that one machine, and the x gathers cost about 1–2 % on the A100 as well. Modal's
  SXM4 hosts also ran the forward slower than its PCIe hosts in the lab (3,322 and 3,426 against 2,863–2,919 µs), so
  absolute A100 lab times from Modal are only comparable within one job.

**Part phases (idea 25).** One part's blocks only (the others return at once):

| Variant (H200) | Registers | µs | HMMA time (plan estimate) |
| --- | ---: | ---: | ---: |
| F-part0-only | 132 | 625.9 | ≈ 432 |
| F-part1-only | 168 | 1026.8 | ≈ 650 |

Both parts run far above their tensor time (+194 and +377 µs). By the plan's own rule ("if both run equally far
above, 22 is the better route") idea 25 is not built.

**ncu on the H200 forward** (pass 3 of `tools/modal/ncu_forward.sh`, `STALLS=1`: every stall reason, the pipes and
the stalls by SASS line; base, part 0 only, part 1 only):

- Stalls per issued instruction:

  | Stall | Base | Part 0 | Part 1 |
  | --- | ---: | ---: | ---: |
  | wait (fixed latency) | 1.44 | 1.43 | 1.23 |
  | selected | 1.00 | 1.00 | 1.00 |
  | not selected | 0.91 | 0.79 | 1.06 |
  | math pipe throttle | 0.50 | 0.69 | 0.57 |
  | dispatch | 0.41 | 0.50 | 0.41 |
  | long scoreboard | 0.37 | 0.34 | 0.15 |
  | short scoreboard | 0.26 | 0.25 | 0.15 |
  | barrier | 0.23 | 0.25 | 0.19 |
  | no instruction | 0.18 | 0.09 | 0.20 |
  | MIO throttle | 0.07 | 0.04 | 0.06 |

- Pipes (percent of peak while active): tensor 38.1, ALU 29.3, FMA 24.5, LSU 24.2.
- Stall samples by opcode (base): HMMA 25.2 %, LOP3 10.7 % (the 3xTF32 splits), LDS 8.4 %, FMUL 6.5 %, STS 5.4 %,
  LEA 5.4 %, FADD 5.3 %, SHFL 4.5 %, FFMA 4.1 %. The two hottest lines are `B2R.RESULT` after W_GROUP's
  `bar.red.or` vote (3.5 % together), then the receivers' `SHFL.IDX` (3.0 %).
- The largest single cost is "wait", the fixed-latency dependency stall: the HMMA and split chains within a warp.
  Not selected (0.91) means another warp issued instead, which more warps cannot raise further.
- Files: `runs/round9/r9-ncu/` (`stall-summary.txt`, `ncu-stalls.csv`, `ncu-source-sass.csv.gz`).

**A persistent w′ (idea 23) is closed.** One-warp blocks, a persistent grid and the next edge's indices prefetched
(dsh and dr bitwise, dx in another atomic order):

| Variant | H200 (base 1632.4 µs) | A100 PCIe, SXM4 plan (base 3275.8 µs) |
| --- | ---: | ---: |
| wp-blk32 | 1.005 | 1.113 |
| wp-persist | 1.203 | 1.301 |
| wp-persist-pf | 1.264 | 1.484 |
| wp-persist128 | 1.051 | 1.192 |
| wp-persist128-pf | 0.987 | 1.422 |

None reaches the plan's 0.97 bar. The persistent grid loses the hardware's block balancing.

Runs: `runs/round9/` (`r9-lab-h200`, `r9-ncu`, `wp9-h200`, `wp9-a100`, `x9-h200`, `x9-a100`, `ord-h200`, `ord-a100`, `sxm9-b`).

### Round 11: x gathers after the k-loop (`X_LATE`), adopted; stage-ahead rejected (6 November 2026)

**`X_LATE` (idea 27) is default on the H200 and the A100-SXM4.**

- **What it does:** with W_GROUP the x gathers and their index shuffles sat in the staging, so x and the receivers
  stayed live in registers through the whole k-loop, although only the TP reads them. X_LATE moves them to just
  before the C-write barrier, after the k-loop. That frees about 6 registers and shortens the k-loop's waits.
- **`K_PIPE`** loads the next k-step's a values and first W fragment one k-step ahead.
- **Accuracy:** bitwise (GPU test, both switches and together; and in the lab on both GPUs).
- **Built as** `fused_forward.X_LATE` / `K_PIPE` (`X_LATE_GPUS`, `K_PIPE_GPUS`): text transforms of the W_GROUP
  render, with checked pattern counts. Where the patterns do not apply (no W_GROUP; K_PIPE without VEC_A; A_SPLIT),
  the kernel is built without them.
- **Lab** (round-11 variants from `tools/lab_round11.py`, written by the ideas session; base = the plan):

  | Variant | H200 | A100-SXM4 |
  | --- | ---: | ---: |
  | xlate | 0.970, 0.961 (two jobs) | 0.961 |
  | xlate-kp | 0.977 | 0.958 |
  | xlate-kpk4 (unrolled by 4) | 0.983 | 0.986 |
  | xmid (gathers before the last two k-steps) | 0.998 | spills on sm_80, not run |
  | xmid-kp | 0.994 | spills on sm_80, not run |

- **MD-step A/Bs** (SrTiO3 10×10×10, commit `d74ca073`, b = switch on):

  | GPU | Switch | b/a, order ab | b/a, order ba |
  | --- | --- | ---: | ---: |
  | H200 | X_LATE | 0.9884 | 0.9882 |
  | H200 | X_LATE + K_PIPE | 0.9934 | 0.9927 |
  | A100-SXM4 | X_LATE | 0.9937 | 0.9937 |
  | A100-SXM4 | X_LATE + K_PIPE | 0.9910 | 0.9920 |

- **K_PIPE on top of X_LATE**, A100-SXM4 (commit `fc0d7531`, X_LATE on both sides): 0.9980 / 0.9990. Both orders
  are below 1, but by about 0.15 %, within run-to-run noise. K_PIPE stays off; on the H200 it is slower than
  X_LATE alone.
- **CPU suite** on leto at `fc0d7531`: 5150 passed, 202 skipped.
- **Commit:** `fc0d7531` (branch `fastmace/round11-lab`).

**Stage-ahead (idea 26) is rejected.** It copies the next batch's a, sh and indices by cp.async while the current
batch computes, with two sh buffers and two index halves per warp (+13,824 B of shared memory). On the H200 it is
bitwise but slower, and it spills:

| Variant (H200, base 1608.1 µs) | Spill | Ratio |
| --- | ---: | ---: |
| stage-ahead | 24 B | 1.033 |
| stage-ahead-noidx (a and sh only) | 48 B | 1.012 |
| stage-ahead-xlate-kpk4 | 48 B | 1.032 |
| stage-ahead-xmid-kp | 48 B | 1.034 |

The first stage-ahead job died with CUDA_ERROR_ILLEGAL_ADDRESS. The transforms had dropped the source text that the
lab's launcher reads to recognise VEC_A's padded a rows, so the lab passed a compacted a and the kernel read past it.
`tools/lab_round11.py` now refuses any variant that loses that text.

Runs: `runs/round11/` (`r11-h200` (the crash), `r11-a100`, `r11b-h200`, `r11c-h200`, `ab27-h200-a`, `ab27-a100-a`, `kp-a100-c`).


### Round 12: CG factors folded into W (`CG_FOLD`), adopted; TP in the next k-loop rejected (6 November 2026)

Ideas session revision 12 (`tools/lab_round12.py`), kernel lab at `54488d00` (OMAT-0 medium forward kernel, SrTiO3,
15 rounds, L2 flush 256 MB; Modal H200 on profile `fourth`, Modal A100-SXM4-80GB on `third`). Runs in
`runs/round12/`.

| Variant | What it does | H200 (base 1541.6 µs) | A100-SXM4 (base 2757.5 µs) | Messages |
| --- | --- | --- | --- | --- |
| `iii` | 22(iii): the TP of batch n inside batch n+1's k-loop | 1.109 | 1.075 | bitwise |
| `iii-nf` | the same, with a no-flush fast path | 1.068 | 1.029 | bitwise |
| `iii-u1` | the same, unroll 1 | 1.164 | 1.086 | bitwise |
| `iii-br` | the same, branch form | 1.214 | 1.091 | bitwise |
| `fold` | 24: CG factors folded into W (part 1: one factor per path; part 0: w x0 first) | **0.969** (154 regs) | **0.951** (64 B local) | max rel 3–7e-7 |
| `iii-fold` | both | 1.093 | 1.044 | max rel 3–7e-7 |
| `iii-nf-fold` | both, no-flush path | 1.039 | 1.018 | max rel 3–7e-7 |

- **22(iii) rejected:** every form is slower on both GPUs.
- **`CG_FOLD` (idea 24):** ported as `fastmace/kernels/oeq_streaming/cg_fold.py` and switched by
  `fused_forward.CG_FOLD` / `CG_FOLD_GPUS`. It applies only to the message kernel (STORE_W false, as in the MD path)
  with the two DEGREE_PARTS parts, and builds without it elsewhere. Check script: `tools/modal/cg_fold_check.sh`, at
  `361a8e33`.
  - **MD step (SrTiO3 10x10x10, b = CG_FOLD):** H200 0.9938 / 0.9959; A100-SXM4-80GB 0.9877 / 0.9898 (both build orders).
  - **Force/stress gate (OMAT-0 medium; SrTiO3 3x3x3, NaCl, NbCl5; default and CG_FOLD in the same job):**

    | GPU | Plan | SrTiO3 F / S ratio | NaCl F / S | NbCl5 F / S |
    | --- | --- | --- | --- | --- |
    | H200 | default | 0.95 / 0.97 | 0.74 / 1.00 | 0.97 / **2.10 (fail)** |
    | H200 | CG_FOLD | 0.99 / 0.93 | 0.78 / 0.92 | 0.98 / **2.06 (fail)** |
    | A100-SXM4 | default | 1.04 / 1.21 | 0.76 / 1.16 | 1.03 / **2.42 (fail)** |
    | A100-SXM4 | CG_FOLD | 1.06 / 1.25 | 0.75 / 1.13 | 0.99 / **2.38 (fail)** |

    CG_FOLD gives the default's result: NbCl5 stress fails for both, at about 1.3–1.5e-7 eV/Å³ against a limit of
    about 1.3e-7. That is near the gate's absolute floor of 1e-7 and is the A100's accepted exception. **New:** the
    H200 default also failed NbCl5 stress in this job (2.10). Its earlier margin was small, and the float32 ASE
    reference value moves between jobs (6.3–6.5e-8). CG_FOLD changes none of this, so it was adopted
    ("make them default if they are faster and accurate").
  - **Default since this commit on the H200 and A100-SXM4-80GB** (decision 66).
