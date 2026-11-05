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
