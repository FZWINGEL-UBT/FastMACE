# H200 ideas plan: ideas 2, 3, 4, 6, 7 and 8 measured (5 November 2026)

This covers the H200 ideas plan (`plans/h200-ideas-20261105.md`, branch `fastmace/h200-ideas-20261105`), except
ideas 1 and 9, which are in [w′ pairs](../wprime-pairs-20261105/README.md). Each idea got a cheap bound first and
was built only if the bound left room.

**Workload:** OMAT-0 medium, float32 hybrid, Benchmark_loup (SrTiO3 10×10×10, 5,000 atoms, 373,760 edges), 1 fs
Langevin. **Profile:** Modal `third` (fzwingel54). **Environment:** the Modal hybrid image (recipe C plus OEq). The
local runs used `mace_cueq_oeq` on the RTX 5060 Ti.

## Outcome

| Idea | What | Verdict |
| --- | --- | --- |
| 2 | `fused_forward.DEGREE_PARTS`: path parts by input irrep dimension | **adopted** for the H200 and the A100-SXM4-80GB (decision 66) |
| 3 | piece-sorted radial-table evaluation | dead (the sort costs more than the kernel saves) |
| 4 | S tile: CG·sh once per edge instead of once per lane | dropped (bound 0.966 on the H200) |
| 6 | one launch per BlockLinear (`segment_rows.SEGMENT_GROUPED`) | rejected (MD step neutral on the A100) |
| 7 | 1-D bulk copies for the forward's a/sh staging | dropped (staging bound 3–4.5 %) |
| 8 | PDL between our own GEMMs | dropped (≤ 4.7 µs per group of four) |

## Idea 2: DEGREE_PARTS (adopted)

`fused_forward.degree_parts` puts the paths of the same input irrep dimension into one part: OMAT-0's scalar-input
paths and its vector-input paths. It replaces `balanced_parts`' cost balance, so each part gathers only its own x
rows. The result is bitwise, because every output slot keeps its terms and edge order. The bitwise block in
`tests/gpu/test_fused_folded_forward_cuda.py` passed on the RTX 5060 Ti, with both the SIMT and the tensor-core
product. The switch arrived in `3ed00b73`, and its defaults in `891bf370`.

MD step b/a, b = on, ab / ba, `tools/ab_rewrites.py`:

| GPU | ab | ba | Base step | Verdict |
| --- | ---: | ---: | ---: | --- |
| H200 | 0.9818 | 0.9845 | 5.77 ms | on |
| A100-SXM4-80GB | 0.9920 | 0.9932 | 11.12 ms | on |
| RTX 5060 Ti | 0.9980 | 1.0032 | 24.71 ms | off (mixed) |

Gate with the new default (`891bf370`): the force/stress gate passes on NaCl, NbCl5 and SrTiO3 3×3×3, both on the
H200 and on an A100 80GB PCIe run under the SXM4 plan (Modal assigned a PCIe card). Benchmark_loup with the new
default: H200 5.504 ms per step (1.101 µs per atom-step), A100 80GB PCIe 10.89 ms. CPU merge tier on leto at
`891bf370`: 5,150 passed, 202 skipped.

## Kernel-lab bounds of the fused forward

`tools/omat0_kernel_lab.py --kernels forward` (`6ebc795d` and `891bf370`): each variant's ratio to the base kernel.
The H200 base was 1,777.9 µs at `6ebc795d` and 1,691.3 µs at `891bf370`. The A100-SXM4 base was 3,171.3 µs; the
A100 80GB PCIe base was 3,253.2 µs.

| Variant | H200 | A100-SXM4 | A100 PCIe | RTX 5060 Ti |
| --- | ---: | ---: | ---: | ---: |
| F-no-xload (no sender-feature gathers) | 0.834 | 0.812 | | 0.950 |
| F-no-aload4 (no float4 a staging) | 0.965 | 0.991 | | |
| F-no-shload4 (no float4 sh staging) | 0.995 | 0.993 | | 1.017 |
| F-no-ash4 (neither) | 0.955 | 0.971 | | |
| F-no-tp (no tensor product) | 0.619 | 0.613 | | 0.517 |
| F-tp-no-coef (no CG-coefficient multiplies) | 0.910; 0.971 at `891bf370` | 0.879 | 0.945 | 0.977 |
| F-s-tile (idea 4's per-lane product, with S read from shared memory) | 0.966 at `891bf370` | | 0.951 | |

What the bounds decided:
- **Idea 7 dropped:** the a/sh staging is worth 3–4.5 %, so bulk copies cannot pay for their mbarriers.
- **Idea 2 built:** the x gathers (17–19 % of the kernel) were the largest target.
- **Idea 4 dropped:** once DEGREE_PARTS was on, the coefficient multiplies cost only 2.9 % on the H200. The S tile's
  per-lane product bounds at 3.4 % (≈ 57 µs), before the cost of building the tile.

## Idea 3: piece-sorted radial tables (dead)

`tools/bench/radial_sort_bench.py` (RTX 5060 Ti; synthetic OMAT-0-shaped tables, 9 pairs, 624 and 896 pieces): the
kernel on edges sorted by (pair, piece) took 0.866 of the receiver-order kernel. But the sort alone took 155 µs,
against 509 µs for the whole kernel. Total ratio > 1.

## Ideas 6 and 8: GEMM launches (`tools/bench/grouped_gemm_bound.py`, `34c4bc1e`)

The bound compares one CUDA graph of four 3xTF32 products (M = 5,000 × (2l + 1), l = 0–3) against one launch over
their concatenated rows, and against the four launched with programmatic stream serialization (PDL, no wait;
unsafe, timing only). Saving per group of four, in µs:

| K×N | H200 grouped | H200 PDL | A100 PCIe grouped | A100 PCIe PDL |
| --- | ---: | ---: | ---: | ---: |
| 128×128 | 15.2 | 1.8 | 26.0 | −1.3 |
| 128×384 | 14.8 | 2.5 | 35.9 | 2.6 |
| 384×128 | 27.2 | 4.7 | 51.7 | 4.5 |
| 256×128 | 20.7 | 2.7 | 37.3 | −0.5 |

Idea 8 is below the plan's 3 µs stop rule on both GPUs.

Idea 6 was built as `segment_rows.SEGMENT_GROUPED` (`f8facf08`). It runs all of a BlockLinear's blocks in one launch
of CUTLASS's gathered `species_into` kernel, one "species" per block, so each element's arithmetic is unchanged.
`tests/gpu/test_segment_rows_cuda.py` on the A100-SXM4-80GB showed it bitwise against the per-block launches,
forward and input gradient, and under graph replay. The 7-atom expectation in the new test was wrong and is fixed
in the next commit.

MD step on the A100-SXM4-80GB, where SEGMENT_ROWS is the default: b/a 0.9987 / 1.0004, neutral. The bound's
"joined" launch used `mm_into` on contiguous rows. The gathered kernel over the real BlockLinears does not convert
the launch saving into step time. **Rejected** and kept off. On the H200, SEGMENT_ROWS itself is off (1.0006 /
1.0000), so the switch does not act there.

## Runs (`runs/`)

| Directory | What |
| --- | --- |
| `degree-parts-{h200,a100}/` | `tools/modal/hopper_sweep.sh`, `AB=degparts` at `3ed00b73` |
| `degree-parts-rtx5060ti/` | local `ab_rewrites.py`, both orders |
| `bounds-{h200,a100}/` | lab bounds at `6ebc795d` (A100-SXM4-80GB) |
| `bounds-rtx5060ti-lab.json` | local lab bounds |
| `gate-bounds-h200/`, `gate-bounds-a100pcie/` | at `891bf370`: gate, Benchmark_loup, lab F-tp-no-coef/F-s-tile, grouped/PDL bound |
| `segment-grouped-a100/` | at `f8facf08`: `test_segment_rows_cuda.py`, the SEGMENT_GROUPED A/B |

Modal cost on profile `third`: about $1.5 for these jobs (metered $7.58 before, $8.14 after the H200 gate run).
Seven earlier launches were lost when the local WSL clients got SIGTERM and WSL then restarted. Their apps were
stopped by hand, and their stale in-flight entries were cleared from the budget guard.
