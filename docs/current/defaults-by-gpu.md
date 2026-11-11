# Defaults by GPU: what runs today and why

Source: main `7d18028d` (10 November 2026; §0 regenerated from the code, §1 compiled at `eca7ba5d` on 5 November and
extended row by row since; Ada and 5060 Ti cells re-measured by ada-5060-screen, 11 November 2026). The code is the authority: `fastmace/planner/fixed.py` (auto rules),
the `*_GPUS` sets next to each kernel switch, and [switch-map.md](switch-map.md) (every config field and its
`auto` rule). History of every attempt, adopted or not: [optimisation ledger](../optimisation-ledger.md).
How the A100 OMAT-0 path fits together: [omat0-a100.md](omat0-a100.md).

Conventions. b/a = candidate time over baseline time, same-process A/B, alternating blocks; two values = the two
build orders (ab / ba); below 1 is faster. GPUs: **5060 Ti** = NVIDIA GeForce RTX 5060 Ti (local, sm_120);
**Ada** = NVIDIA RTX 4000 Ada Generation (murphy, leto, leeloo; sm_89); **A100** = every A100 model where the set
is a `GPUModels` (owner, 4 November 2026: "turn on a100 speedups no matter which version of it it runs on";
`fastmace/kernels/gpu_models.py`), and an H200 entry covers the H200 NVL too (owner, 7 November 2026: a GPU
tested on one variant counts for every variant unless they differ significantly in performance; the H100 SXM and
PCIe cards stay separate). Otherwise the exact name given. A switch that only removes work (no new or replaced kernel, no change of tiling,
launch shape, precision or memory kept alive) is on for every GPU after one not-slower A/B on any GPU (owner,
7 November 2026). Any other GPU (L4, RTX A4000, RTX PRO 6000, …) gets
none of the switches below: decision 66 (owner, 29 October 2026) enables a per-GPU default only where same-process
whole-step A/Bs in both build orders measured a speed-up, by device name, never by compute capability.
Benchmark_loup = OMAT-0 medium, SrTiO3 5,000 atoms unless stated. "on" = default on; "–" = off (not measured, or
measured slower: the cell says which).

## 0. Every switch by GPU family (main `7d18028d`, 10 November 2026)

One row per switch set, read from the code at `7d18028d` (Ada and 5060 Ti cells: the ada-5060-screen merge): the `*_GPUS` set beside each switch and
`fastmace/planner/fixed.py`. Cells: **on** = default on for that family; **off** = measured and not adopted (the value
says why: b/a ab / ba, b = the switch on unless stated); **not measured** = no MD-step A/B for that family is recorded
in the code comments, this page, the ledger or the results it cites; **n/a** = cannot run there (the reason is given).
Evidence and reports per switch: §1 and the [ledger](../optimisation-ledger.md). Class: **bitwise** / **exact** (the
same result, or the same up to the order of atomics); **rounding** (needs the 3× float32 force/stress gate,
`tools/force_stress_gate.py`); **precision** (a TF32-class change, owner-approved); **certified** (a table with a
certificate). Commits are where the set last gained a family.

### 0.1 MH-1, POLAR-1 and MACE-OFF plans

| Switch (set) | H200 | A100 | Ada | 5060 Ti | Class | Commits |
| --- | --- | --- | --- | --- | --- | --- |
| `grouped_launch`, MH-1 (`DECISION_66_GROUPED_LAUNCH_GPUS`) | not measured | not measured | off (0.5 % slower, OPT-T03b) | on | exact | d66, 29 Oct |
| `wprime_backward`, MH-1 (`DECISION_66_WPRIME_GPUS`) | not measured | on | on | on | rounding | `70593240` |
| `gate_epilogue`, MH-1 (`DECISION_66_GATE_EPILOGUE_GPUS`) | not measured | not measured | on | on | rounding | `57cbb0ac` |
| unpadded gated input-gradient kernel (`gated_products.UNPADDED_GPUS`, exact names) | not measured | not measured | on | on | bitwise | `70a48290` |
| `segment_gemm` column tile order (`tile_rule.COLUMN_ORDER_GPUS`, exact) | not measured | off (1.0017 / 1.0015) | on | on | bitwise | `34576e69` |
| 128-row tiles, 8 warps (`tile_rule.LARGE_TILE_GPUS`, exact) | not measured | off (A100-SXM4-40GB 1.021 / 1.021) | on | off (1.020 / 1.007) | bitwise | `f2261a09` |
| `tf32x3_gemms`, MH-1 (`DECISION_66_TF32X3_GPUS`) | not measured | on | off (Triton tf32x3 slower) | off (Triton tf32x3 slower) | precision | `41619478` |
| float64 D2 `w_tables`, D3 `save_transcendentals` (`DECISION_65_GPUS`, exact; D1 suspended) | not measured | off (D2, D3 slower) | on | on | rounding | `fbfc95f7` |
| POLAR-1 `wprime_backward` + `fused_folded_forward` (`POLAR_DECISION_66_WPRIME_GPUS`; A100 by label, `POLAR_DECISION_66_WPRIME_LABEL_GPUS`) | not measured | on, M only (L: NbCl5 1.003 / 1.002) | on, M and L | on, M only by label (0.9597 / 0.9589, 0.9380 / 0.9415; L: NbCl5 1.0098 / 1.0097) | rounding | `fbe91675`, `f7f223c1`, ada-5060-screen |
| POLAR-1 `gate_epilogue` (`POLAR_DECISION_66_GATE_EPILOGUE_GPUS`) | not measured | off (tie) | on | on (with the 5060 Ti set, NbCl5: S 0.9927 / 0.9846, L 0.9869 / 0.9885; NaCl mixed) | rounding | `fbe91675`, ada-5060-screen |
| POLAR `SPIN_BATCHED`, `FOLD_UPDATE`, `COMPOSE_HEADS`, `PRUNE_DEAD_DEGREES` (`True`, every GPU) | on (pure-removal rule; not measured) | on | on | on | rounding (gates pass) | `657c80b3`, `0d09dc40`, `5176ad1d`, `540fa159` |
| POLAR k-space `MEGAKERNEL` (`polar.MEGAKERNEL_GPUS`, from `MEGAKERNEL_MIN_WORK`) | not measured | on | on (NbCl5: S 0.8964 / 0.8964, M 0.9490 / 0.9517, L 0.9678 / 0.9684) | on | rounding | `b155b642`, `b4ec7053`, `d1048f04` |
| `tf32x3_gemms` by label: OFF ×5, POLAR-1 S, M, L (`DECISION_66_TF32X3_MODEL_GPUS`) | not measured | on | not measured | not measured | precision | `91e428a1`, `1ffd7f34`, `f7f223c1` |
| `CUEQ_PROJECTED` (`cueq_projection.CUEQ_PROJECTED_GPUS`; not OFF23-large, POLAR-1-S, -M on the A100: `DECISION_66_CUEQ_PROJECTED_EXCLUDED`) | on | on | off (0.9992 / 0.9987 off again: slower on) | off (from the 5060 Ti candidate set: removing it 1.0000 / 0.9997, 0.9920 / 0.9993) | rounding | `b8d977f4` |
| Stock first block's species skip, MACE-OFF (`generic.modules.STOCK_FIRST_SKIP_GPUS`) | not measured | on | on | on | exact | `90cac1b3`, `42c063ce` |
| Radial tables with the stock first block, MACE-OFF (`DECISION_66_STOCK_FIRST_TABLES_GPUS`) | not measured | on | on | on | certified | `90cac1b3`, `42c063ce` |
| Radial last layer as 3×TF32, MACE-OFF (`skinny_linear.RADIAL_TF32X3_GPUS`) | not measured | on | not measured | off (Triton 3×TF32 slower) | precision | `90cac1b3` |
| Column-split input gradient (`skinny_linear.SPLIT_COLUMNS_GPUS`) | not measured | off (tie, 0.993–1.005) | on (OFF23-large 0.9556 / 0.9583; OFF24 no split) | on | bitwise | `8e733265`, `1aec1931`, `edfc0115` |
| Model exclusions (`*_MODEL_OFF_GPUS`): `SORT_BY_LENGTH` off for OFF23-small, -large (A100); w′ `STAGE_DOUT` off for POLAR-1-M (A100, Ada) and POLAR-1-L (Ada); `DEGREE_PARTS` off for POLAR-1-M (A100) | not measured (no OFF or POLAR A/B on the H200) | on (the exclusions) | w′ `STAGE_DOUT` off for POLAR-1-M and -L (leave-one-out from the Ada set: M NaCl 0.9868 / 0.9858, L NbCl5 0.9986 / 0.9987 off; the set without it against the defaults: M 0.9986 / 1.0006, 0.9999 / 0.9982, L 1.0005 / 1.0005) | none needed for POLAR-1-M (measured with `STAGE_DOUT` on: 0.9659 / 0.9758, 0.9273 / 0.9389); OFF23-small, -large not measured | rounding / bitwise | `c619d306`, ada-5060-screen |
| `EDGE_SH` in MH-1's class (`generic.modules.EDGE_SH_MH1_GPUS`, empty) | not measured | off (POLAR-1 tie, S, M, L within 0.4 %; MH-1 not measured) | not measured | not measured | rounding | – |
| `FUSED_FIRST_BACKWARD` in MH-1's class (`mh1.modules.FUSED_FIRST_MH1_GPUS`, empty) | not measured | off (POLAR-1 slower: S NaCl 1.006 / 1.006, M 1.005 / 1.004) | not measured | not measured | rounding | – |
| e3nn `o3.Linear` as BlockLinear (`linear.E3NN_BLOCK_LINEAR_GPUS`, empty) | not measured | not measured | not measured | not measured | not recorded | – |

### 0.2 OMAT-0 and the generic family

| Switch (set) | H200 | A100 | Ada | 5060 Ti | Class | Commits |
| --- | --- | --- | --- | --- | --- | --- |
| `generic_chunked_conv_backward` (`DECISION_66_GENERIC_CHUNKED_GPUS`) | not measured | not measured | on | on | rounding | `37d7a6a3` |
| `generic_radial_tables` (`DECISION_66_GENERIC_RADIAL_TABLES_GPUS`) | on | on | on | on | certified | `27d27f21` (H200) |
| `generic_fused_folded_forward` + `generic_wprime_backward`, OMAT-0 medium (`DECISION_66_GENERIC_WPRIME_GPUS`) | on | on | on | on | rounding | `27d27f21` (H200) |
| OMAT-0 small: the pair above + `tf32x3_gemms` (`DECISION_66_GENERIC_WPRIME_LABEL_GPUS`, `…_TF32X3_MODEL_GPUS["omat0-small"]`) | on | on | not measured | not measured | precision + rounding | `979785a7`, `5797d298` |
| `generic_fold_product_linear` (`DECISION_66_GENERIC_FOLD_PRODUCT_GPUS`), with `FOLD_FIRST_DENSITY_LINEAR` | on | on | on | on (alone 0.9377 / 0.9348, 0.9222 / 0.9166) | rounding | `27d27f21` (H200), ada-5060-screen |
| `tf32x3_gemms`, OMAT-0 medium (`DECISION_66_TF32X3_GPUS`, `…_OMAT0_GPUS`) | on | on | on (Triton backend, with the compose group) | off (Triton tf32x3 slower) | precision | `27d27f21`, `1cd369e4` |
| species GEMM (`generic.modules.SPECIES_GEMM_GPUS`) | off (with `species_gemm.CUTLASS` 1.0141 / 1.0175) | off (0.992 / 0.991; its skip runs as 3×TF32) | on | on | exact | `b48c10de` |
| `FUSED_FIRST_BACKWARD` + `TABLED_FORWARD` | on | on | on | on | rounding | `27d27f21`, `5ddec837` |
| `EDGE_SH`, generic (`EDGE_SH_GPUS`) | on | on | on | on | rounding | `27d27f21` |
| `HEAD_SLICE` (`energy.HEAD_SLICE_GPUS`) | on | on | on | on | bitwise | `27d27f21` |
| `GRAPH_SUM` (`energy.GRAPH_SUM_GPUS`) | on | on | off (1.001 / 0.999) | off (removing it from the candidate set 0.9971 / 0.9947, 1.0001 / 0.9982) | rounding | `27d27f21` |
| `SYMMETRIC_PAIRS` (`radial.SYMMETRIC_PAIRS_GPUS`) | on | on | on | off (mixed: removing it from the candidate set 0.9984 / 0.9965, 1.0024 / 1.0033; adding it to the adopted set 0.9926 / 0.9960, 0.9986 / 1.0002: SrTiO3 disagrees) | bitwise | `27d27f21` |
| radial-table forward with 8 warps (`radial_table.WARPS_GPUS`) | on | not measured | off (tie 0.9975 / 0.9990, 1.0019 / 0.9997) | off (NbCl5 1.0172 / 1.0071) | launch shape (exactness not recorded) | `62556704` |
| w′ `TABLE_W`, `BLOCK_MESSAGE`, `FACTORED`, `HORNER` | on | on | on | on | rounding | `27d27f21`, `03125b0c`, `1cd369e4` |
| w′ `SCALAR_PATHS` | on | on | off (tie 0.9984 / 0.9984, 1.0001 / 0.9987) | off (SrTiO3 1.0015 / 1.0066) | rounding | `27d27f21` |
| w′ `MAX_REGISTERS` 80 (`MAX_REGISTERS_GPUS`) | not measured | not measured | off (1.0529 / 1.0537, 1.0416 / 1.0401) | on | bitwise (register cap) | `03125b0c` |
| w′ `FOLD_ALPHA` | on | on | off (1.0077 / 1.0082, 1.0079 / 1.0065) | off (1.0018 / 1.0017, 1.0092 / 1.0005) | rounding | `c0813688` |
| w′ `STAGE_DOUT` | on | on (not POLAR-1-M) | on, not POLAR-1-M, -L (alone 0.9699 / 0.9694, 0.9848 / 0.9806) | on (with w′ `STAGE_EDGES`, `STAGE_SH`) | bitwise | `6561cc13`, `ec25c9c4`, ada-5060-screen |
| w′ `STAGE_SH` | on | on | on (re-screen: off again from the Ada set 1.0035 / 1.0028, 1.0078 / 1.0056; the 8 Nov reading flipped) | on (with `STAGE_DOUT`) | bitwise | `b8d977f4`, `ec25c9c4`, ada-5060-screen |
| w′ `STAGE_EDGES` (`wprime_backward.STAGE_EDGES_GPUS`; `STAGE_EDGES_MODEL_GPUS` for omat0-small) | medium off (tie 0.9969 / 1.0010); small on | on | on (with `STAGE_DOUT`, `STAGE_SH`: 0.9639 / 0.9643, 0.9779 / 0.9758) | on (the three left out of the candidate set 1.0529 / 1.0503, 1.0334 / 1.0341) | bitwise | `8aaace6b`, `387ec7bb`, `ec25c9c4`, ada-5060-screen |
| w′ `SPLINE_VECTOR` 4 | on | on | off (0.9918 / 0.9906 with 2: 4 slower) | off (removing it from the candidate set 0.9963 / 0.9941, 0.9957 / 0.9997) | bitwise | `78a25fd6`, `b8d977f4` |
| w′ `BUTTERFLY` | on | on | on (re-screen: 0.9840 / 0.9849, 0.9913 / 0.9931; the 5 Nov reading flipped) | on | bitwise | `bcca4189`, `ec25c9c4` |
| w′ `REDUX` | off (1.0206 / 1.0202) | off (1.018–1.029) | off (1.009 / 1.008, 1.011 / 1.008) | on | rounding (fixed point) | `6c2f2967` |
| fused forward `SKIP_ZERO_COLUMN` | on | on | on | off (1.000) | exact | `02a81e13` |
| fused forward `TENSOR_CORES` | on | on | off (slower: TF32 = 2× FP32) | off (slower) | precision | `27d27f21` |
| fused forward `VEC_SH` | on | on | not measured | on | bitwise | `27d27f21` |
| fused forward `VEC_A`, `TC_WFRAG` | on | on | n/a (tensor-core forward off) | n/a (tensor-core forward off) | bitwise | `27d27f21` |
| fused forward `DEGREE_PARTS` | on | on (not POLAR-1-M) | off (tie 0.9990 / 0.9990, 1.0007 / 0.9994) | off (removing it from the candidate set 0.9962 / 1.0012, 0.9931 / 0.9990) | bitwise | `891bf370` |
| fused forward `WS_FORWARD` | on | n/a (sm_90 wgmma) | n/a | n/a | rounding | `aa544c69` |
| fused forward `LANE_INDICES` | on | not measured | not measured | not measured | bitwise | `44f548bf` |
| fused forward `W_GROUP` | on | on | not measured | not measured | bitwise | `240b2019`, `ff4d1a89` |
| fused forward `WFRAG_REGION` | off (tie 1.0014 / 1.0010) | on | not measured | not measured | bitwise | `ff4d1a89` |
| fused forward `X_LATE` | on | on | not measured | not measured | bitwise | `fc0d7531` |
| fused forward `CG_FOLD` | on | on | not measured | not measured | rounding | `1388c3a1` |
| fused forward `A_LD128` | on | on | not measured | not measured | bitwise | `31c394b4` |
| fused forward `TP_NOFLUSH` | off (0.9992 / 1.0018) | on | not measured | not measured | bitwise | `31c394b4` |
| fused forward `SH_LDS4` (needs `VEC_SH`) | off (1.0014 / 1.0027) | off (1.0008 / 0.9949) | n/a (`VEC_SH` off) | on | bitwise | `9a6df1f5` |
| CUTLASS `tf32x3.HYBRID` (`cutlass.tf32x3.HYBRID_GPUS`; TF32 big·big + one FP16 cross mma in every 3×TF32 entry) | not measured | on (0.9896 / 0.9877 SrTiO3, 0.9832 / 0.9846 NbCl5) | n/a | n/a | rounding (gate passes) | review-retries 2026-11-11 |
| CUTLASS `LINEAR_DIV` (`cutlass.linear_div.LINEAR_DIV_GPUS`) | on | on | not measured | not measured | bitwise (kernel) | `5dc5d799` |
| CUTLASS `SKIP_SELECT` (`cutlass.skip_select.SKIP_SELECT_GPUS`) | on | on | not measured | not measured | bitwise (kernel) | `6561cc13` |
| `SEGMENT_ROWS` (`cutlass.segment_rows.SEGMENT_ROWS_GPUS`) | off (1.0006 / 1.0000) | on | off (tie with the CUTLASS backend on sm_89: 0.9967 / 0.9981, 0.9984 / 1.0016) | n/a (CUTLASS cubins for sm_80 / sm_90 only) | rounding (3×TF32) | `6c2f2967` |
| tabled `POWER_BACKWARD` | on | off (tie 1.0020 / 0.9977; 1.0006 / 1.0028 on the 8 Nov defaults) | on (alone 0.9935 / 0.9929, 0.9935 / 0.9933) | on (alone 0.9917 / 0.9946, 0.9944 / 0.9941) | rounding | `8a1ed833`, `ec25c9c4`, ada-5060-screen |
| `RECEIVER_SUM` (`receiver_sum.RECEIVER_SUM_GPUS`) | off (tie 0.9992 / 1.0012) | on | off (mixed by structure: removing it from the Ada set 0.9967 / 0.9964 SrTiO3, 1.0087 / 1.0068 NbCl5) | on (removing it from the candidate set 1.0041 / 1.0006, 1.0034 / 1.0049) | rounding | `7a2f6b6f`, ada-5060-screen |
| `BACKWARD_WALK` (`edge_outer_sum.BACKWARD_WALK_GPUS`) | off (1.0000 / 1.0001) | on | off (kernel 1.03–1.11) | off (kernel 0.94, near the DRAM floor) | rounding | `03125b0c` |
| `BINNED_SCREEN` from 10,000 atoms (`pair_search.BINNED_SCREEN_GPUS`) | on | on | not measured | not measured | bitwise | `ac01df97` |
| `SPLINE_W` (`spline_forward.SPLINE_W_GPUS`) | on | on | on (with `HORNER`) | on (with `R128`, `STAGE_EDGES`: 0.8162 / 0.8183, 0.8548 / 0.8541) | rounding (certificate) | `b6da7a6f`, `b8d977f4`, `1cd369e4`, ada-5060-screen |
| `R128` (`R128_GPUS`) | on | on | on | on (with `SPLINE_W`) | bitwise | `b6da7a6f`, `b8d977f4`, `1cd369e4`, ada-5060-screen |
| `R128` with `MIN_BLOCKS` 8 for omat0-small (`R128_MIN_BLOCKS_MODEL_GPUS`) | on | not measured | not measured | not measured | bitwise (register cap) | `e2206bb6` |
| spline forward `STAGE_EDGES` | on | on | on | on (with `SPLINE_W`) | bitwise | `b8d977f4`, `1cd369e4`, ada-5060-screen |
| spline forward `PREFETCH_AHEAD` (off for omat0-small on the A100: `PREFETCH_AHEAD_MODEL_OFF_GPUS`) | off (1.0046 / 1.0067) | on | off (1.0037 / 1.0051, 1.0063 / 1.0073) | off (NbCl5 1.0190 / 1.0073) | bitwise | `8aaace6b`, `50480105` |
| spline forward `BLOCK_ORDER` | on | on | off (tie 0.9961 / 0.9947, 1.0012 / 1.0000) | off (removing it from the candidate set 0.9988 / 0.9969, 0.9944 / 1.0013) | bitwise | `5e7d961f` |
| spline forward `PACKED8` | on | on | on (0.9036 / 0.9030, 0.9346 / 0.9357) | on (0.9222 / 0.9246, 0.9456 / 0.9535) | rounding | `e3049399`, `ec25c9c4`, ada-5060-screen |
| spline forward `GEOM` (`spline_forward.GEOM_GPUS`) | not measured | on | off (1.0226 / 1.0209, 1.0156 / 1.0159) | not measured (needs `SPLINE_W` + `STAGE_EDGES`, on from ada-5060-screen) | rounding | `48000458` |
| `COMPOSE_SKIP` | on | on | on (with OMAT-0 tf32x3) | off (needs `tf32x3_gemms`: tf32x3 + compose NbCl5 1.0412 / 1.0396) | rounding | `b8d977f4`, `1cd369e4` |
| `COMPOSE_FAST` | on | on | on | n/a (with `COMPOSE_SKIP`, off) | bitwise | `b8d977f4`, `1cd369e4` |
| `COUNT_PLAN` | on | on | off (tie 1.0000 / 1.0004) | n/a (with `COMPOSE_SKIP`, off) | bitwise | `b8d977f4` |
| `JOINED_LAYOUT` | on | on | off (tie 1.0006 / 0.9986) | n/a (with `COMPOSE_SKIP`, off) | bitwise | `8aaace6b` |
| `GENERIC_FOLD_READOUT_TAIL` (`species.GENERIC_FOLD_READOUT_TAIL_GPUS`) | on | on | on | on (removing it from the candidate set 1.0037 / 1.0048, 1.0071 / 1.0097) | rounding | `b8d977f4`, `1cd369e4`, ada-5060-screen |
| `SORT_BY_LENGTH` (`neighbors.skin.SORT_BY_LENGTH_GPUS`) | on | on (not OFF23-small, -large) | on (0.9794 / 0.9815, 0.9890 / 0.9908) | on (0.9860 / 0.9868, 0.9967 / 0.9887) | rounding | `ec716912`, `ec25c9c4`, ada-5060-screen |
| `SC_GENERATED` (`sc_generated.SC_GENERATED_GPUS`) | on | on | on (0.9785 / 0.9775, 0.9738 / 0.9749) | off (routed only under `tf32x3_gemms`, off on the 5060 Ti; with SC_FACTORED a tie in the adopted set 0.9995 / 0.9998, 0.9994 / 1.0026) | rounding | `9225793e`, `ec25c9c4` |
| `SC_FACTORED` (`sc_generated.SC_FACTORED_GPUS`) | off (lab bound under 1 %, sc0-invariant-basis; no MD A/B) | on | on (0.9978 / 0.9978, 0.9907 / 0.9906) | n/a (needs `SC_GENERATED`, off here) | rounding | `48000458`, `63153034` |

### 0.3 Tiled mode, plain constants and the memory-rescue rungs

| Switch (set) | H200 | A100 | Ada | 5060 Ti | Class | Commits |
| --- | --- | --- | --- | --- | --- | --- |
| Tiled speed switches: `TILE_PREFETCH_DEVICE`, `TILE_VIRIAL`, `TILE_GRAPH_CACHE`, `TILE_FILTER32`, `TILE_EPILOGUE`, `TILE_PINNED_SUPERSET`, `TILE_GRAPH_CACHE_WAVE`, `TILE_HOST_WORKER` (`tiled.TILED_SPEED_GPUS`) | not measured | not measured | not measured | on | exact; `TILE_VIRIAL`, `TILE_EPILOGUE` rounding | `b9362cc9`, `3fdbad6c` |
| Two sweep-2 lanes (`tiled.LANE_GPUS`, exact) | not measured | not measured | on | off (1.05× per pair) | exact | `d14d07a8` |
| `VIRIAL_ELEMENTWISE` (`rewrites.engine.VIRIAL_ELEMENTWISE_GPUS`; stress calls with single_graph_strain, every model family; measured on OMAT-0 medium with tools/virial_ab.py, the force call with stress) | not measured | off (bound 0.4 %; 18x18x18 0.9957 / 0.9944) | off (order-dependent: 1.0041 / 0.9805, rerun 1.0045 / 0.9810) | on (0.9675 / 0.9648) | rounding | `cde06eb0` |
| Backward chunks ×4 in tiles (`tiled.TILE_CHUNK_SCALE`, exact) | not measured | not measured (×1) | on | on | exact | `42c975c6` |
| Every GPU (`True`): `edge_outer_sum.STRIDED_ROWS`, `langevin.NOISE_DRAW_AHEAD`, `runtime.TABLE_GRADIENT_PER_CHUNK`, `tabled_outer_sum.SHAPE_CARRIER`, `skin.SORT_KEYS_ONLY`, `tiled.TILE_FILL_SKIP`; `generic.TILE_BLOCK_ROWS` (`None`: on in every tile) | on | on | on | on | exact (`TILE_BLOCK_ROWS` rounding) | `9ed7f559`, `349f03e3`, `4e7d0b5b`, `d4d7a08a`, `3fdbad6c`, `cfec5551` |
| Memory-rescue rungs (`planner.memory.MEMORY_RESCUE`, in order): `recompute_conv_weights`, `generic_recompute_conv_weights`, `recompute_first_table`, `generic_recompute_first_table`, `generic_recompute_first_layer`, `recompute_node_segments`, `gate_epilogue` off, `recompute_node_region` | taken only when the plan does not fit (rule 12, decision 67) | same | same | same | the generic rungs rounding, float32 only (`RESCUE_FLOAT32_ONLY`); `generic_recompute_conv_weights` never beside `generic_wprime_backward` (`RESCUE_NOT_BESIDE`) | `7d18028d` (§19 "Rung B beside w′") |

Off on every GPU (empty sets; measured where named, nowhere else): `FIRST_DIV` (1.031–1.035), `LINEAR_DIV_SPLIT`
(H200 0.9972 / 1.0025), `SEGMENT_GROUPED` (H200), `PREFETCH`, `TC_TP` (A100, round 4), `TC_WFRAG_SPLIT`, `A_SPLIT`
(H200 1.0311 / 1.0346), `K_PIPE` (H200 with X_LATE 0.9934 / 0.9927), `WGMMA` (H200), `FACTOR_W` (H200 0.9969 /
0.9995), `CURSOR32` (H200 1.0016 / 1.0033), `SHARED_LOCATOR` (H200 0.9966 / 0.9974), `BLOCK_LDG` (5060 Ti, plain loads
faster with FACTORED; H200 tie 1.0012 / 1.0001), `ESTRIN` (A100, H200 1.0030 / 1.0012), `TABLE_PAIRS`, `HOIST_SCALE`
(H200), `STAGE_QUADS` (H200 0.9960 / 0.9984), `PACKED_COEF` (H200 1.0072 / 1.0079), `DECISION_66_GROUPED_LAUNCH_MODEL_GPUS`
(OFF on the A100, replaced by `tf32x3_gemms`), `FOLD_ALPHA_MODEL_OFF_GPUS` (omat0-small on the H200, within the
spread). The step-limiters re-screen of 9 November re-measured several of these on the H200 against that day's
defaults ([rescreen-20261109](../../results/step-limiters-20261109/README.md#round-3-results-no-winner)).

## Open: not measured per family

Every switch above with a family cell "not measured". The session `ada-5060-screen` (10-11 November 2026,
[results](../../results/ada-5060-screen-20261110/README.md)) measured the Ada and 5060 Ti cells of the OMAT-0
switches and the POLAR / OFF cells named there. A cell needs:

- **A/B only** (bitwise or exact): a same-process MD-step A/B in both build orders. A pure removal is on after one
  not-slower A/B (owner, 7 November 2026); anything else (a new kernel, tiling, launch shape, memory kept alive) needs
  a speed-up in both orders (decision 66).
- **A/B + gate** (rounding or precision): the same A/B, plus the 3× float32 force/stress gate
  (`tools/force_stress_gate.py`; owner, 7 November 2026). A precision change also needs the owner's approval.

| Switch | Not measured on | Needs |
| --- | --- | --- |
| `grouped_launch`, MH-1 | H200, A100 | A/B only (exact) |
| `wprime_backward`, MH-1 | H200 | A/B + gate |
| `gate_epilogue`, MH-1 | H200, A100 | A/B + gate |
| unpadded gated kernel | H200, A100 | A/B only (bitwise) |
| column tile order | H200 | A/B only (bitwise) |
| 128-row tiles | H200 | A/B only (bitwise) |
| `tf32x3_gemms`, MH-1 | H200 | A/B + gate (precision) |
| float64 D2, D3 | H200 | A/B + gate |
| POLAR-1 `wprime_backward` + fused forward | H200 | A/B + gate |
| POLAR-1 `gate_epilogue` | H200 | A/B + gate |
| POLAR `SPIN_BATCHED`, `FOLD_UPDATE`, `COMPOSE_HEADS`, `PRUNE_DEAD_DEGREES` | H200 | none to stay on (pure-removal rule); an A/B would only confirm |
| POLAR k-space `MEGAKERNEL` | H200 | A/B + gate |
| `tf32x3_gemms` by label (OFF, POLAR-1) | H200, Ada, 5060 Ti | A/B + gate (precision) |
| stock first skip and stock-first tables (MACE-OFF) | H200 | A/B only (exact) for the skip; A/B + certificate for the tables |
| radial 3×TF32 (MACE-OFF) | H200, Ada | A/B + gate (precision) |
| `SPLIT_COLUMNS` | H200 | A/B only (bitwise) |
| OFF / POLAR model exclusions | H200 | A/B (as their base switches) |
| `EDGE_SH` in MH-1's class | H200, Ada, 5060 Ti (MH-1 nowhere) | A/B + gate |
| `FUSED_FIRST_BACKWARD` in MH-1's class | H200, Ada, 5060 Ti (MH-1 nowhere) | A/B + gate |
| e3nn `o3.Linear` as BlockLinear | every family | A/B (class not recorded) |
| `generic_chunked_conv_backward` | H200, A100 | A/B + gate |
| OMAT-0 small pair | Ada, 5060 Ti | A/B + gate (precision) |
| radial-table forward 8 warps | A100 | A/B (exactness not recorded) |
| w′ `MAX_REGISTERS` 80 | H200, A100 | A/B only (bitwise) |
| fused forward `VEC_SH` | Ada | A/B only (bitwise) |
| fused forward `DEGREE_PARTS`, `W_GROUP`, `X_LATE`, `A_LD128` | Ada, 5060 Ti (`DEGREE_PARTS` measured: off on both) | A/B only (bitwise) |
| fused forward `LANE_INDICES` | A100, Ada, 5060 Ti | A/B only (bitwise) |
| fused forward `WFRAG_REGION`, `TP_NOFLUSH` | Ada, 5060 Ti | A/B only (bitwise) |
| fused forward `CG_FOLD` | Ada, 5060 Ti | A/B + gate |
| CUTLASS `LINEAR_DIV`, `SKIP_SELECT` | Ada, 5060 Ti | A/B only (bitwise at kernel level) |
| `BINNED_SCREEN` | Ada, 5060 Ti | A/B only (bitwise) |
| `R128` `MIN_BLOCKS` 8 for omat0-small | A100, Ada, 5060 Ti | A/B only (bitwise) |
| spline `GEOM` | H200, 5060 Ti (now that SPLINE_W + STAGE_EDGES are on there) | A/B + gate |
| `SC_FACTORED` | 5060 Ti only once SC_GENERATED is on there (it needs `tf32x3_gemms`; H200: lab bound under 1 %, no MD A/B) | A/B + gate |
| `VIRIAL_ELEMENTWISE` | H200; other families than OMAT-0 medium on the 5060 Ti | stress-call A/B (tools/virial_ab.py) + gate |
| tiled speed switches | H200, A100, Ada | A/B only, except `TILE_VIRIAL`, `TILE_EPILOGUE` (A/B + gate) |
| two sweep-2 lanes, tile backward chunks ×4 | H200, A100 | A/B only (exact) |

## 1. Per-GPU switches (decisions 65 and 66): evidence

### MH-1 (`mh1-r2scan`) plans

| Switch (set) | 5060 Ti | Ada | A100 | Exactness | Evidence (MD step b/a) | Decided | Report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `grouped_launch` (`DECISION_66_GROUPED_LAUNCH_GPUS`) | on | – (lost 0.5 % in OPT-T03b) | – | exact | 5060 Ti NaCl 0.971 / 0.975, NbCl5 1.002 / 0.998, NaNbO3 1.001 / 0.999 | d66, 29 Oct | [decision66-grouped-launch](../validation/speed/decision66-grouped-launch.md) |
| `wprime_backward` (`DECISION_66_WPRIME_GPUS`) | on | on | on (A100-SXM4-40GB not measured) | rounding | A100-SXM4-80GB NaNbO3 0.815 / 0.812, NbCl5 0.889 / 0.889; 5060 Ti NaNbO3 0.937 / 0.940, NbCl5 0.970 / 0.971, NaCl 0.946 / 0.935; Ada NaNbO3 0.938 / 0.939, NbCl5 0.964 / 0.964, NaCl 0.934 / 0.934 | OPT9-T05, owner 29 Oct (d66) | [opt9-T03](../validation/speed/opt9-T03.md) |
| `gate_epilogue` with K-outer `gated_products` (`DECISION_66_GATE_EPILOGUE_GPUS`) | on | on | – (not measured) | rounding | 5060 Ti NbCl5 1,944 0.983 / 0.982, 9,216 0.958 / 0.957; NaNbO3 1,920 0.982 / 0.988, 8,640 0.972 / 0.968; NaCl 1.000 / 1.005. Ada NbCl5 0.976 / 0.975, 0.953 / 0.953; NaNbO3 0.982 / 0.981, 0.968 / 0.969; NaCl 0.998 / 1.003 | speed-def E2, owner 31 Oct | [speed-def](../validation/speed/speed-def.md) |
| unpadded gated input-gradient kernel (`gated_products.UNPADDED_GPUS`, exact names) | on | on | – (not measured; RTX A4000 keeps `gate_epilogue` off) | bitwise | Ada NbCl5 21,600 atoms 0.947 / 0.951; 5060 Ti NbCl5 14,400 0.959 / 0.956 (threshold ≥ 3 %) | d66 | [gated-unpadded](../validation/speed/gated-unpadded.md) |
| `segment_gemm` column tile order (`tile_rule.COLUMN_ORDER_GPUS`, exact) | on | on | – (A100-SXM4-80GB NbCl5 9,216 1.0017 / 1.0015) | bitwise | 5060 Ti NbCl5 9,216 0.940 / 0.938, NaNbO3 8,640 0.959 / 0.955 | d66, 29 Oct | [segment-gemm-order](../validation/speed/segment-gemm-order.md) |
| 128-row tiles, 8 warps (`tile_rule.LARGE_TILE_GPUS`, exact) | – (NbCl5 1,944 1.020 / 1.007) | on | – (A100-SXM4-40GB 9,216 1.021 / 1.021) | bitwise | Ada NbCl5 9,216 0.987 / 0.986, 1,944 0.989 / 0.989, NaNbO3 0.995 / 0.994 | d66, 29 Oct | [ab-constants-audit](../validation/speed/ab-constants-audit-20261030.md) "tiles128" |
| `tf32x3_gemms` on CUTLASS (`DECISION_66_TF32X3_GPUS`; `…_OMAT0_GPUS` = A100 80GB PCIe, kept for the record) | – (Triton tf32x3 slower) | on for OMAT-0 medium only (`DECISION_66_TF32X3_OMAT0_GPUS`, Triton backend; carries COMPOSE_SKIP); MH-1 – (slower) | on, MH-1 and OMAT-0 | precision change (owner-approved) | A100-SXM4-80GB MH-1 NaCl 0.887 / 0.882, NbCl5 0.772 / 0.770; OMAT-0 0.948 / 0.947; A100 PCIe OMAT-0 4.61 → 3.31 µs/atom/step with the SXM4's switches; Ada OMAT-0 (8 Nov): alone 1.0438 / 1.0400 (slower), with the compose group 0.9902 / 0.9898; off again from the Ada set 1.0117 / 1.0124 | owner 3 Nov; every A100 4 Nov; Ada d66, 8 Nov | [cutlass-tf32x3](../validation/omat0/cutlass-tf32x3.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| float64: D1 56 mantissa bits, D2 `w_tables`, D3 `save_transcendentals` (`DECISION_65_GPUS`, exact) | D2, D3 on; D1 suspended | D2, D3 on; D1 suspended | – (D2, D3 slower) | rounding | float64 package + D1 + D2 + D3 ABBA: local NaCl 0.806, NaNbO3 0.782, NbCl5 0.797; Ada NaCl 0.804, NbCl5 0.802. D1's `auto` is `"dynamic"` everywhere since 28 Oct (a graph re-captured during a run computed wrong energies) | d65, 28 Oct | [ideas-20261027](../../plans/ideas-20261027.md), [switch-map](switch-map.md) |

### POLAR-1 (`polar1-s`, `polar1-m`, `polar1-l`) plans

POLAR-1 runs MH-1's interaction class, so it takes MH-1's class-level defaults (section 2) and, since 5 November 2026,
`species_tables`. MH-1's label-gated switches above apply to POLAR-1 only where listed here.

| Switch (set) | 5060 Ti | Ada | A100 | Exactness | Evidence (MD step b/a) | Decided | Report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `wprime_backward` + `fused_folded_forward` (`POLAR_DECISION_66_WPRIME_GPUS`), M and L only | – (not measured) | on | – (not measured) | rounding | Ada POLAR-1-M NaCl 0.963 / 0.965, NbCl5 0.970 / 0.971; with `gate_epilogue` 0.966 / 0.966, 0.957 / 0.958; force gate passes | owner 5 Nov (d66) | [polar-reuse](../validation/polar/polar-reuse.md) |
| `gate_epilogue` (`POLAR_DECISION_66_GATE_EPILOGUE_GPUS`) | – (not measured) | on | – (A100-SXM4-80GB tie: S 0.9996–1.0005, M 0.9987–1.0005) | rounding | Ada POLAR-1-S NbCl5 0.994 / 0.993, NaCl 0.998 / 1.001; POLAR-1-M NbCl5 0.989 / 0.989, NaCl 1.002 / 0.999 | owner 5 Nov (d66) | [polar-reuse](../validation/polar/polar-reuse.md) |
| `longrange.polar.SPIN_BATCHED`: α and β field features in one call (plain default, every GPU) | on (S NaCl 0.943 / 0.945, NbCl5 0.885 / 0.882; M 0.977 / 0.971, 0.949 / 0.951) | on (S NaCl 0.953 / 0.953, NbCl5 0.897 / 0.897; M 0.983 / 0.982, 0.957 / 0.953; L 0.995 / 0.994, 0.975 / 0.975) | on | rounding (gate passes) | A100-SXM4-80GB S NaCl 0.943 / 0.942, NbCl5 0.912 / 0.913; M 0.971 / 0.969, 0.961 / 0.962; L 0.979 / 0.985, 0.976 / 0.977; force gates pass for S, M, L | off-polar-alex, 7 Nov (pure removal) | [off-polar-alex](../../results/off-polar-alex-20261106/README.md) |
| `rewrites.polar.update_fold.FOLD_UPDATE`: the field-update MLP folded at setup (plain default, every GPU) | on (S NaCl 0.948 / 0.963, NbCl5 1.002 / 0.998; M 0.974 / 0.960, 1.001 / 0.990; L 0.981 / 0.971, 1.002 / 0.997) | on (S NaCl 0.966 / 0.966; M 0.972 / 0.970; L 0.977 / 0.975; NbCl5 1.000) | on | rounding (gate passes) | A100-SXM4-80GB S NaCl 0.965 / 0.971, M 0.969 / 0.970, L 0.972 / 0.971, NbCl5 1.000 on all three; force gates pass for S, M, L | off-polar-alex, 8 Nov (pure removal) | [off-polar-alex](../../results/off-polar-alex-20261106/README.md) |
| `rewrites.polar.head_compose.COMPOSE_HEADS`: head projections composed into their dot products, P1 (plain default, every GPU) | on (NaCl S 0.906 / 0.909, M 0.876 / 0.872, L 0.883 / 0.879; NbCl5 S 1.001 / 1.005, M 0.997 / 1.000, L 1.002 / 0.998) | on | on (off (ab): NaCl 1.087-1.127, NbCl5 1.104-1.181) | rounding (gate passes) | force gates pass for S, M (NaCl, NbCl5), L NaCl (A4000), L NbCl5 (A100-80GB) | polar-megakernel, 8 Nov (pure removal) | [POLAR-1 megakernel](../design/research/2026-11-07-polar-megakernel.md) |
| `longrange.polar.MEGAKERNEL` + `kspace_megakernel.CUDA`: k-space field call as one CUDA C++ megakernel (`MEGAKERNEL_GPUS`, cells from `MEGAKERNEL_MIN_WORK` = 1e6 atoms x k) | on (NbCl5 S 0.893 / 0.896, M 0.957 / 0.957, L 0.972 / 0.973; NaCl 0.998–1.012, below the threshold: off) | – (not measured) | on (A100 family; NbCl5 S 0.9846 / 0.9854, M 0.9918 / 0.9954, L 0.9855 / 0.9964; NaCl below the threshold: off) | rounding (gate passes) | force gates pass for S, M (NaCl, NbCl5), L NaCl (A4000), L NbCl5 (A100-80GB); A100-SXM4 S, M (NaCl, NbCl5) | polar-megakernel, 8 Nov (d66); A100 off-polar-a100, 8 Nov (d66) | [POLAR-1 megakernel](../design/research/2026-11-07-polar-megakernel.md), [off-polar-a100](../../results/off-polar-a100-20261108/README.md) |
| `rewrites.polar.head_liveness.PRUNE_DEAD_DEGREES`: dead head degrees pruned, R5 (plain default, every GPU) | on (L NaCl 0.979 / 0.975, NbCl5 0.998 / 1.000) | on (NaCl 0.974 / 0.973, NbCl5 1.000 / 1.001) | on (off (ab): NaCl 1.087-1.141, NbCl5 1.104-1.217) | exact (readout) / rounding (fold) | gates pass | polar-megakernel, 8 Nov (pure removal) | [POLAR-1 megakernel](../design/research/2026-11-07-polar-megakernel.md) |
| `tf32x3_gemms` (`DECISION_66_TF32X3_MODEL_GPUS`, S, M and L) | – | – | on | precision change | A100-SXM4-80GB S NaCl 0.940 / 0.938, NbCl5 0.847 / 0.847; M 0.931 / 0.930, 0.842 / 0.843; L 0.906 / 0.907, 0.792 / 0.793; force gates pass | off-polar-alex, 6 Nov (d66) | [off-polar-alex](../../results/off-polar-alex-20261106/README.md) |
| `wprime_backward` + `fused_folded_forward` on the A100 (`POLAR_DECISION_66_WPRIME_LABEL_GPUS`), M only | – | (row above) | on, M | rounding | A100-SXM4-80GB on top of `tf32x3_gemms`: M NaCl 0.976 / 0.977, NbCl5 0.920 / 0.920 (gate passes). L stays off: NaCl 0.956 / 0.957 but NbCl5 1.003 / 1.002 (mixed-cell rule) | off-polar-alex, 6 Nov (d66) | [off-polar-alex](../../results/off-polar-alex-20261106/README.md) |
| w′ `STAGE_DOUT` and fused-forward `DEGREE_PARTS` **off** for POLAR-1-M (`wprime_backward.STAGE_DOUT_MODEL_OFF_GPUS`, `fused_forward.DEGREE_PARTS_MODEL_OFF_GPUS`) | – | – | off, M (on for every other model) | bitwise | A100-SXM4-80GB, b = off: STAGE_DOUT NaCl 0.9682 / 0.9770, NbCl5 0.9920 / 0.9911; DEGREE_PARTS NaCl 0.9825 / 0.9911, NbCl5 0.9967 / 0.9962. `BUTTERFLY` stays on (off: NaCl 0.988 / 0.996, NbCl5 1.002 / 1.002); `SORT_BY_LENGTH` stays on (S, M, L ties); L's w′ retried and still off (NaCl 0.964, NbCl5 1.009 / 1.012) | off-polar-a100-r2, 9 Nov (d66, pure removal) | [off-polar-a100-r2](../../results/off-polar-a100-r2-20261109/README.md) |

POLAR-1-S (one interaction, reassociated) never takes `wprime_backward`, `fused_folded_forward` or `w_tables`: off
with that reason, refused when asked for. `grouped_launch` and the round-5 switches (EDGE_SH, TABLE_W,
FUSED_FIRST_BACKWARD for MH-1's class) stay off for POLAR-1: on the A100-SXM4-80GB EDGE_SH is a tie (S, M, L within
0.4 %) and FUSED_FIRST_BACKWARD is slower (S NaCl 1.006 / 1.006, M NaCl 1.005 / 1.004).

### MACE-OFF (`off23-small`, `off23-medium`, `off23-large`, `off23b-medium`, `off24-medium`) plans

MACE-OFF's first interaction is the stock `RealAgnosticInteractionBlock`, so the generic family's records act on the
later interactions only, except the species skip and the radial tables, which take the stock first block too since
7 November 2026. Measured on NHR@FAU Alex and Modal, A100-SXM4-80GB, water box of 648 atoms (6–7 November 2026):

| Switch (set) | 5060 Ti | Ada | A100 | Exactness | Evidence (MD step b/a) | Decided | Report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tf32x3_gemms` (`DECISION_66_TF32X3_MODEL_GPUS`): every OFF model | – | – | on | precision change | 0.968 / 0.966, 0.967 / 0.963, 0.980 / 0.976, 0.980 / 0.981; force gates pass. OFF23-small since 7 Nov (its gate passes at 3×: paracetamol forces 2.55, stress 2.77): 0.966 / 0.947 on top of the rows below | off-polar-alex, 6–7 Nov (d66) | [off-polar-alex](../../results/off-polar-alex-20261106/README.md) |
| `grouped_launch` (`DECISION_66_GROUPED_LAUNCH_MODEL_GPUS`, empty since 7 Nov) | – | – | – | exact | OFF23-small 0.989 / 0.995 (on 6–7 Nov, then replaced by `tf32x3_gemms`); OFF23-large slower (1.008 / 1.009) | off-polar-alex, 6–7 Nov (d66) | [off-polar-alex](../../results/off-polar-alex-20261106/README.md) |
| Stock first block's species skip (`generic.modules.STOCK_FIRST_SKIP_GPUS`), every OFF model | on (with the tables row below: OFF23-small 0.677 / 0.671, -medium 0.737 / 0.735, -large 0.793 / 0.811) | on (with the tables row below: OFF23-small 0.592 / 0.591, -medium 0.688 / 0.683, OFF23b 0.720 / 0.725, OFF24 0.722 / 0.728, OFF23-large 0.749 / 0.745) | on | exact | A100-SXM4-80GB: OFF23-small 0.709 / 0.714, -medium 0.749 / 0.749, OFF23b 0.826 / 0.821, OFF24 0.823 / 0.826, OFF23-large 0.707 / 0.714 | off-polar-alex, 7 Nov (d66); Ada 8 Nov | [off-polar-alex](../../results/off-polar-alex-20261106/README.md) |
| Radial tables with the stock first block (`DECISION_66_STOCK_FIRST_TABLES_GPUS`), every OFF model | on (measured with the row above) | on (measured with the row above) | on | certified | A100-SXM4-80GB: 0.929–0.977 on all five; gates pass | off-polar-alex, 7 Nov (d66) | [off-polar-alex](../../results/off-polar-alex-20261106/README.md) |
| Radial last layer as 3xTF32 (`skinny_linear.RADIAL_TF32X3_GPUS`), MACE-OFF's radial products only | – (Triton 3xTF32 slower: both sets against the same default, OFF23-small 0.734 / 0.742 with it vs 0.677 / 0.671 without, -medium 0.806 / 0.792 vs 0.737 / 0.735) | – | on | precision change | A100-SXM4-80GB alone: 0.914–0.971 on all five; all three rows above together 0.563–0.645 (OFF23-small 2.58 → 1.45 ms, OFF23-large 11.7 → 6.97 ms); gates pass under the 3× rule (worst: OFF24-medium paracetamol stress 2.50) | off-polar-alex, 7 Nov (d66) | [off-polar-alex](../../results/off-polar-alex-20261106/README.md) |
| Column-split input gradient of the folded radial layer (`skinny_linear.SPLIT_COLUMNS_GPUS`), where the split grid fits in one wave | on | – (not measured) | – (measured, a tie) | exact (bitwise) | 5060 Ti OFF23-large 0.950 / 0.954; other OFF models 0.993–1.006 (the split does not fire); A100 all five 0.993–1.005 (OFF23-large 0.999 / 0.998: a tie, left off; a tiling change needs a speed-up in both orders) | off-polar-alex, 8 Nov (d66) | [off-polar-alex](../../results/off-polar-alex-20261106/README.md) |
| `SORT_BY_LENGTH` **off** for OFF23-small and OFF23-large (`neighbors.skin.SORT_BY_LENGTH_MODEL_OFF_GPUS`; MACE-OFF has no spline tables for the length order to help) | – | – | off, small and large (on for the other three) | rounding (back to the gated configuration of 8 Nov) | A100-SXM4-80GB, b = off: OFF23-small 0.9830 / 0.9739, OFF23-large 0.9951 / 0.9900; OFF23-medium, OFF23b and OFF24 ties (0.997–1.003) | off-polar-a100-r2, 9 Nov (d66, pure removal) | [off-polar-a100-r2](../../results/off-polar-a100-r2-20261109/README.md) |

Measured and left as they are: EDGE_SH (on; off is 0.3–1 % slower), HEAD_SLICE (on; off is a tie or slower),
GRAPH_SUM (on; off is up to 0.5 % faster on OFF23-small and -medium, a tie elsewhere: left on, flagged), the species
GEMM (slower, 1.009–1.020), the chunked fused forward (slower on the medium models, 1.12–1.14).

### OMAT-0 and the generic family

| Switch (set) | 5060 Ti | Ada | A100 | Exactness | Evidence (MD step b/a) | Decided | Report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `generic_chunked_conv_backward` (`DECISION_66_GENERIC_CHUNKED_GPUS`) | on | on | – (not measured; stays on there only as the carrier of `generic_wprime_backward`, chunking never runs; 1.31× slower) | rounding | Ada NaCl 8,000 0.953 / 0.953, SrTiO3 6,655 0.916 / 0.916; 5060 Ti NaCl 5,832 0.972 / 0.972, SrTiO3 3,645 0.949 / 0.951; RTX A4000 (6 MB L2) 3–4× slower | d66, 2 Nov | [omat0-capacity](../validation/omat0/omat0-capacity.md) |
| `generic_radial_tables` (`DECISION_66_GENERIC_RADIAL_TABLES_GPUS`) | on | on | on (A100-SXM4-40GB not measured) | certified | Ada NaCl 0.864 / 0.862, SrTiO3 0.845 / 0.844; 5060 Ti NaCl 0.849 / 0.848, SrTiO3 0.821 / 0.821; tiled Ada 975,560 atoms 0.861 / 0.862, 5060 Ti 320,000 0.912 / 0.882; A100 PCIe SrTiO3 0.902 / 0.901, A100-SXM4-80GB NaCl 8,000 0.917 / 0.917 | owner 2 Nov; A100s 3 Nov | [generic-radial-tables](../validation/omat0/generic-radial-tables.md), [loup-a100](../validation/omat0/loup-a100.md) |
| `generic_fused_folded_forward` + `generic_wprime_backward` (`DECISION_66_GENERIC_WPRIME_GPUS`) | on | on | on | rounding | both on: 5060 Ti 0.839 / 0.831, Ada 0.832 / 0.833, A100-SXM4-80GB 0.895 / 0.894; the fused forward alone 1.29× slower (stays off without wprime) | owner 3 Nov | [generic-wprime](../validation/omat0/generic-wprime.md) |
| `generic_fold_product_linear` (`DECISION_66_GENERIC_FOLD_PRODUCT_GPUS`) | – (not measured) | on | on | rounding | Ada SrTiO3 10×10×10 0.9986 / 0.9960, NbCl5 0.9911 / 0.9849; A100-SXM4-80GB 0.9928 / 0.9931, 0.9805 / 0.9828 | d66, 4–5 Nov | [generic-fold-product](../validation/omat0/generic-fold-product.md) |
| species GEMM (`generic.modules.SPECIES_GEMM_GPUS`, exact) | on | on | – (0.992 / 0.991: its skip runs as 3×TF32; own mma kernel no better) | exact | Benchmark_loup Ada 0.955 / 0.952, 5060 Ti 0.950 / 0.953 | owner 4 Nov ("Yes both") | [ledger: omat0-species-gemm](../optimisation-ledger.md#omat0-species-gemm-the-species-skip-as-one-grouped-gemm) |
| `TABLE_W`: w and its slope from the value table (`wprime_backward.TABLE_W_GPUS`) | on | on | on | rounding | Benchmark_loup 5060 Ti 0.888 / 0.892, Ada 0.928 / 0.931, A100-SXM4-80GB 0.943 / 0.943 | d66, 4 Nov | `results/omat0-kernel-lab-20261103` §16 |
| `BLOCK_MESSAGE` (`BLOCK_MESSAGE_GPUS`); `BLOCK_LDG` (`BLOCK_LDG_GPUS`, empty) | on (plain loads) | on | on | rounding | vs TABLE_W: Ada 0.974 / 0.971, A100-SXM4-80GB 0.983 / 0.977, 5060 Ti 0.976 / 0.975 with `__ldg` (1.060 without); with FACTORED the 5060 Ti is faster with plain loads, so no GPU keeps `__ldg` | d66, 4 Nov; LDG dropped 5 Nov | [ledger](../optimisation-ledger.md#omat0-block-message-the-conv-message-in-linear_1s-block-layout) |
| `FACTORED` TP backward (`FACTORED_GPUS`) | on (since round 2, with plain loads, HORNER, 80 registers) | on | on | rounding | Ada 0.983 / 0.980; A100-SXM4-80GB 0.938 / 0.940 (w′ kernel 0.90 / 0.73); 5060 Ti SrTiO3 0.979 / 0.982, NbCl5 0.986 / 0.989. `FACTORED_SCALE` rejected (A100 1.025 / 1.024) | owner 4 Nov; 5060 Ti 5 Nov | [omat0-tp-backward](../validation/omat0/omat0-tp-backward.md) |
| `SYMMETRIC_PAIRS` mirror table rows (`radial.SYMMETRIC_PAIRS_GPUS`) | – | on | on | bitwise | on top of FACTORED: Ada 0.996 / 0.995, A100-SXM4-80GB 0.998 / 0.990 | owner 4 Nov | [omat0-tp-backward](../validation/omat0/omat0-tp-backward.md) §4 |
| `HORNER` (`HORNER_GPUS`), `SCALAR_PATHS` (A100), `MAX_REGISTERS` (5060 Ti: 80) | HORNER + 80 regs on | HORNER on (OMAT-0 medium needs it for SPLINE_W; alone a tie, 1.004 / 1.001) | HORNER + SCALAR_PATHS on | rounding | A100-SXM4-80GB SrTiO3 0.986 / 0.983, NbCl5 0.994 / 0.978; 5060 Ti (with FACTORED, no LDG) 0.979 / 0.982, 0.986 / 0.989; Ada (8 Nov): with SPLINE_W 0.8396 / 0.8304; both off again from the Ada set 1.2003 / 1.2015 | A100 round 2, 5 Nov; Ada d66, 8 Nov | [round-2 notes](../../plans/ideas-20261105-round2.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| zero-column skip in the fused forward, degree-5 tables (`fused_forward.SKIP_ZERO_COLUMN_GPUS`) | – (1.000) | on | on | exact | Ada 0.981, A100-SXM4-80GB 0.986 (vs degree 5 alone; both vs degree 10 0.896 / 0.895); A100 PCIe 4.61 → 3.31 µs/atom/step with the SXM4 set | owner 4 Nov | `results/omat0-kernel-lab-20261103` |
| tensor-core fused forward (`TENSOR_CORES_GPUS`) | – (slower: TF32 = 2× FP32) | – (slower) | on | precision (TF32 products) | A100-SXM4-80GB 0.933 / 0.933; loup 3.17 → 2.92 (NVE), 3.12 → 2.90 µs/atom/step (Langevin); energy error 1.39 meV vs 0.73 (floor 10 meV) | owner 4 Nov | [omat0-a100](omat0-a100.md) §3.5 |
| `VEC_SH` (`VEC_SH_GPUS`), `VEC_A` and `TC_WFRAG` (A100) | VEC_SH on | – | all three on | bitwise | 5060 Ti VEC_SH SrTiO3 10×10×10 0.977 / 0.977, NbCl5 0.984 / 0.984; A100-SXM4-80GB VEC_SH + VEC_A 0.973 / 0.976, 0.976 / 0.981; + TC_WFRAG 0.960 / 0.954, 0.970 / 0.969. `PREFETCH` and `TC_TP` rejected (sets empty) | A100 round 4, 5 Nov | [round-4 notes](../../plans/ideas-20261105-round2.md#round-4-the-fused-forwards-staging-5-november-2026) |
| `DEGREE_PARTS` (`DEGREE_PARTS_GPUS`): fused-forward path parts by input degree, so each part gathers only its own x rows | – (not measured) | – (not measured) | on (H200 too; not POLAR-1-M: `DEGREE_PARTS_MODEL_OFF_GPUS`) | bitwise | A100-SXM4-80GB SrTiO3 10×10×10 0.9920 / 0.9932; H200 0.9818 / 0.9845 (b = on) | d66, 5 Nov | [h200-ideas-20261105](../../results/h200-ideas-20261105/) idea 2 |
| `SH_LDS4` (`SH_LDS4_GPUS`; needs VEC_SH): the TP's sh reads as float4 shared loads | on | – (VEC_SH off) | – (A100-SXM4-80GB 1.0008 / 0.9949) | bitwise | 5060 Ti SrTiO3 10×10×10 0.9932 / 0.9902; H200 1.0014 / 1.0027 (off) | d66, 6 Nov | [kernel review](../validation/kernel-review-20261105.md) |
| `WFRAG_REGION` (`WFRAG_REGION_GPUS`; fragment-sized W region, 12 warps) + `W_GROUP` (`W_GROUP_GPUS`; warp pairs share W fragments) | – | – | both on (A100-SXM4-80GB) | bitwise | A100-SXM4-80GB SrTiO3 10×10×10: WFRAG_REGION 0.9871 / 0.9887, with W_GROUP 0.9718 / 0.9734; H200 WFRAG_REGION a tie (1.0014 / 1.0010), W_GROUP on there since 6 Nov (0.9946 / 0.9942). `A_SPLIT` rejected (H200 1.0311 / 1.0346) | d66, 6 Nov | [research answers](../../results/research-answers-20261105/README.md#round-8-a100-at-12-warps-a_split-ncu-stall-reasons-6-november-2026) |
| `X_LATE` (`X_LATE_GPUS`; the W_GROUP forward's x gathers after the k-loop) | – | – | on (H200, A100-SXM4-80GB) | bitwise | MD step H200 0.9884 / 0.9882, A100-SXM4 0.9937 / 0.9937; `K_PIPE` (a and W one k-step ahead) off: H200 with X_LATE 0.9934 / 0.9927 | d66, 6 Nov | [research answers](../../results/research-answers-20261105/README.md#round-11-x-gathers-after-the-k-loop-x_late-adopted-stage-ahead-rejected-6-november-2026) |
| `CG_FOLD` (`CG_FOLD_GPUS`; the message kernel's CG factors folded into W) | – | – | on (H200, A100-SXM4-80GB) | rounding (gate as the default) | MD step H200 0.9938 / 0.9959, A100-SXM4 0.9877 / 0.9898 | d66, 6 Nov | [research answers](../../results/research-answers-20261105/README.md#round-12-cg-factors-folded-into-w-cg_fold-adopted-tp-in-the-next-k-loop-rejected-6-november-2026) |
| `A_LD128` (`A_LD128_GPUS`; a read with one LDS.128 per k-step) | – | – | on (H200, A100-SXM4-80GB) | bitwise | MD step H200 0.9925 / 0.9956; A100-SXM4 with TP_NOFLUSH 0.9909 / 0.9923 | d66, 6 Nov | [research answers](../../results/research-answers-20261105/README.md#round-13-one-lds128-for-a-a_ld128-and-the-tp-without-flush-checks-tp_noflush-adopted-6-november-2026) |
| `TP_NOFLUSH` (`TP_NOFLUSH_GPUS`; TP without flush checks for full in-row batches) | – | – | on (A100-SXM4-80GB) | bitwise | MD step A100-SXM4 0.9936 / 0.9954; H200 0.9992 / 1.0018 (off) | d66, 6 Nov | [research answers](../../results/research-answers-20261105/README.md#round-13-one-lds128-for-a-a_ld128-and-the-tp-without-flush-checks-tp_noflush-adopted-6-november-2026) |
| `LINEAR_DIV` (`cutlass.linear_div.LINEAR_DIV_GPUS`; interaction linear / (density + 1) in the GEMM epilogue, written into its slice) | – | – | on (H200, A100-SXM4-80GB) | bitwise (kernel level) | MD step H200 0.9912 / 0.9934, A100-SXM4 0.9896 / 0.9891 | d66, 6 Nov | [research answers](../../results/research-answers-20261105/README.md#revision-14-gemm-epilogue-linear_div-and-warp-specialized-forward-ws_forward-adopted-ideas-30-32-33-36-measured-6-november-2026) |
| `WS_FORWARD` (`WS_FORWARD_GPUS`; warp-specialized wgmma forward, producer/consumer warps) | – | – | on (H200) | rounding (gate as the default) | MD step H200 0.9584 / 0.9610 | d66, 6 Nov | [research answers](../../results/research-answers-20261105/README.md#revision-14-gemm-epilogue-linear_div-and-warp-specialized-forward-ws_forward-adopted-ideas-30-32-33-36-measured-6-november-2026) |
| `STAGE_DOUT` (`wprime_backward.STAGE_DOUT_GPUS`; the receiver's dout row in shared memory in the w′ backward) | – | – | on (H200, A100-SXM4-80GB; not POLAR-1-M: `STAGE_DOUT_MODEL_OFF_GPUS`) | bitwise (dsh, dL/dr; dx up to the atomics' order) | MD step H200 0.9616 / 0.9638, A100-SXM4 0.9896 / 0.9920 | d66, 6 Nov | [research answers](../../results/research-answers-20261105/README.md#revision-14-step-2-stage_dout-and-skip_select-adopted-first_div-rejected-graph-break-removed-6-november-2026) |
| `SKIP_SELECT` (`cutlass.skip_select.SKIP_SELECT_GPUS`; species-skip products with the own-species select in the epilogue) | – | – | on (H200, A100-SXM4-80GB) | bitwise (kernel level) | MD step H200 0.9876 / 0.9919, A100-SXM4 0.9843 / 0.9858; `FIRST_DIV` rejected (1.031–1.035) | d66, 6 Nov | [research answers](../../results/research-answers-20261105/README.md#revision-14-step-2-stage_dout-and-skip_select-adopted-first_div-rejected-graph-break-removed-6-november-2026) |
| `FOLD_ALPHA` (`wprime_backward.FOLD_ALPHA_GPUS`; per-path α folded into the w′ value table) | – | – | on (H200, A100-SXM4-80GB) | rounding (gate as the default) | MD step H200 0.9963 / 0.9954, A100-SXM4 0.9895 / 0.9893; `ESTRIN` (idea 39) off: A100 0.9961 / 0.9963 alone, but `FOLD_ALPHA` beats it (0.9967 / 0.9902, 0.9884 / 0.9907) and the two together gain nothing | d66, 6 Nov | [ideas queue](../../results/ideas-queue-20261106/README.md) |
| `POWER_BACKWARD` (`tabled_outer_sum.POWER_BACKWARD_GPUS`; tabled backward by dual Horner over a power-basis table) | – | – | on (H200) | rounding (gate as the default) | MD step H200 0.9954 / 0.9957; A100-SXM4 a tie (1.0020 / 0.9977); on the A100 defaults of 8 Nov alone 0.9919 / 0.9969, on top of the ten H200 switches 1.0006 / 1.0028 (off) | d66, 6 Nov | [ideas queue](../../results/ideas-queue-20261106/README.md) |
| `RECEIVER_SUM` (`receiver_sum.RECEIVER_SUM_GPUS`; the generic density sums as a segmented sum over receiver-sorted edges) | – | – | on (A100-SXM4-80GB, A100 80GB PCIe) | rounding (gate as the default) | MD step A100-SXM4 0.9998 / 0.9918, 0.9990 / 0.9985; H200 a tie (0.9992 / 1.0012) | d66, 6 Nov | [ideas queue 2](../../results/ideas-queue2-20261106/README.md) |
| `BINNED_SCREEN` (`pair_search.BINNED_SCREEN_GPUS`; the neighbour-list rebuild's pair screen as a cell list, from `BINNED_MIN_ATOMS` = 10,000 atoms) | – | – | on (H200, A100-SXM4-80GB, A100 80GB PCIe) | bitwise | SrTiO3 16×16×16 MD H200 0.9767 / 0.9833, A100-SXM4 0.9765 / 0.9776; 10×10×10 H200 a tie, A100 1.0038 / 1.0044 (hence the threshold) | d66, 6 Nov | [ideas queue 2](../../results/ideas-queue2-20261106/README.md) |
| `SPLINE_W` (`spline_forward.SPLINE_W_GPUS`; interaction 2's message in Symmetrix-XL's layout with w from a per-model certified cubic spline, the w′ backward on the same spline, density alone from the radial table) + `R128` (`R128_GPUS`; one receiver per 128-thread block, `__launch_bounds__(128, 6)`) | – | on (with HORNER) | on (A100 family) | rounding (certificate 0.48, gate passes) | MD step H200 0.8789 / 0.8775 (Benchmark_loup 4.92 → 4.31 ms); `R128` on top 0.9838 / 0.9848; `PARTS` (degree parts) 0.9938 / 0.9904, off; A100-SXM4 (Alex, 8 Nov) `SPLINE_W` 0.8654 / 0.8661, with `R128` 0.8491 / 0.8469, `R128` off again from all ten 1.0325 / 1.0311; Ada (8 Nov) with HORNER 0.8396 / 0.8304, with `R128` 0.8328 / 0.8312; `R128` off again from the Ada set 1.0017 / 1.0004 | d66, 7 Nov; A100 d66, 8 Nov; Ada d66, 8 Nov | [spline-w H200](../../results/spline-w-h200-20261107/README.md), [a100-omat0-best](../../results/a100-omat0-best-20261108/README.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| `COMPOSE_SKIP` (`compose_skip.COMPOSE_SKIP_GPUS`; interaction 1's reassociated projection, first linear and receiver-species skip composed per species, `C_l,z = R_l L_l B_l,z` in float64, through `species_mm`'s CUTLASS launch; the interaction then only divides by density + 1) | – | on (with OMAT-0 tf32x3) | on (A100 family) | rounding (gate passes) | MD step H200 0.9698 / 0.9738 against the H200 defaults (4.12 → 4.00 ms); measured and off: `STAGE_QUADS` 0.9960 / 0.9984, `PACKED_COEF` 1.0072 / 1.0079, `FACTOR_W` 0.9969 / 0.9995, `CURSOR32` 1.0016 / 1.0033, `SHARED_LOCATOR` 0.9966 / 0.9974; A100-SXM4 (8 Nov) 0.8027 / 0.9546; Ada (8 Nov) off again from the Ada set 1.0572 / 1.0585 | d66, 7 Nov; A100 d66, 8 Nov; Ada d66, 8 Nov | [h200-compose-quads](../../results/h200-compose-quads-20261107/README.md), [a100-omat0-best](../../results/a100-omat0-best-20261108/README.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| `SPLINE_VECTOR` (`wprime_backward.SPLINE_VECTOR_GPUS`; with SPLINE_W the w′ kernel reads the spline's four terms per weight as one float4 instead of two float2) | – | – (2; 4 is slower) | on (A100 family) | bitwise | MD step H200 0.9739 / 0.9793 against the H200 defaults (SPLINE_W, R128); R128 with `MIN_BLOCKS=5` 0.9925 / 0.9937 (not combined, cap stays 6); A100-SXM4 (8 Nov) 2 again from all ten 1.0122 / 1.0115; Ada (8 Nov) 2 again from the Ada set 0.9918 / 0.9906 (4 slower) | d66, 7 Nov; A100 d66, 8 Nov; Ada d66, 8 Nov | [spline-w H200](../../results/spline-w-h200-20261107/README.md), [a100-omat0-best](../../results/a100-omat0-best-20261108/README.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| `COMPOSE_FAST` (`compose_skip.COMPOSE_FAST_GPUS`; with COMPOSE_SKIP the composed weights are buffers bound once per radial table and the four degree products one joined `species_mm`, the division a view) | – | on | on (A100 family) | bitwise | MD step H200 0.9614 / 0.9656 against the H200 defaults (4.015 → 3.860 ms); measured and off: `LINEAR_DIV_SPLIT` 0.9972 / 1.0025; A100-SXM4 (8 Nov) off again from all ten 1.0240 / 1.0235; Ada (8 Nov) off again from the Ada set 1.0105 / 1.0104 | d66, 8 Nov; A100 d66, 8 Nov; Ada d66, 8 Nov | [h200-fastpath](../../results/h200-fastpath-20261108/README.md), [a100-omat0-best](../../results/a100-omat0-best-20261108/README.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| `CUEQ_PROJECTED` (`cueq_projection.CUEQ_PROJECTED_GPUS`; cuEq SymmetricContraction's weight x projection kept as a buffer instead of an einsum per call) | – | – (0.9992 / 0.9987 off again) | on (A100 family), except OFF23-large, POLAR-1-S and -M (`RewriteConfig.cueq_projected`, `planner.fixed.DECISION_66_CUEQ_PROJECTED_EXCLUDED`) | rounding (gate passes; OFF small, medium, 23b, 24 and POLAR-1-L on the A100) | MD step H200 0.9902 / 0.9954; A100-SXM4 (8 Nov) 0.9480 / 0.9945, off again from all ten 1.0010 / 1.0020; OFF water_216 small 0.9894 / 0.9930, medium 0.9896 / 0.9997, 23b 0.9915 / 0.9916, 24 0.9932 / 0.9981, large 1.0006 / 0.9996 (excluded); POLAR-1-L 0.994-0.999, S and M ties or slower on NaCl (excluded); Ada (8 Nov) off again from the Ada set 0.9992 / 0.9987 (slower on, off) | d66, 8 Nov; A100 d66, 8 Nov; Ada d66, 8 Nov | [h200-fastpath](../../results/h200-fastpath-20261108/README.md), [a100-omat0-best](../../results/a100-omat0-best-20261108/README.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md), [off-polar-a100](../../results/off-polar-a100-20261108/README.md) |
| `SC_GENERATED` (`sc_generated.SC_GENERATED_GPUS`; interaction 0's 1 / (density + 1) and product 0's cuEq symmetric contraction as one generated straight-line kernel pair, `RewriteConfig.sc_generated`) | – | – | on (H200, A100 family) | rounding (gate passes on the H200 and the A100) | MD step H200 0.9760 / 0.9731 (3.48 → 3.40 ms); A100 80GB PCIe 0.9685 / 0.9652 (7.50 → 7.27 ms) | d66, 9 Nov | [sc0-op](../../results/sc0-op-20261109/README.md) |
| `GENERIC_FOLD_READOUT_TAIL` (`species.GENERIC_FOLD_READOUT_TAIL_GPUS`; the generic product's species skip composed with the following readout linear) | – | on | on (A100 family) | rounding (gate passes) | MD step H200 0.9855 / 0.9901; all four of 8 November together 0.9497 / 0.9509 (4.018 → 3.816 ms); A100-SXM4 (8 Nov) 0.9861 / 0.9908, off again from all ten 1.0097 / 1.0072; all ten 0.7471 / 0.7469 (9.85 → 7.36 ms); Ada (8 Nov) 0.9974 / 0.9962 against the Ada defaults, off again from the Ada set 1.0020 / 1.0026 | d66, 8 Nov; A100 d66, 8 Nov; Ada d66, 8 Nov | [h200-fastpath](../../results/h200-fastpath-20261108/README.md), [a100-omat0-best](../../results/a100-omat0-best-20261108/README.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| `STAGE_EDGES` (`spline_forward.STAGE_EDGES_GPUS`; with SPLINE_W and R128 the forward stages 32-edge tiles in shared memory, cutoff / sender / lookup once per edge) | – | on | on (A100 family) | bitwise | MD step H200 0.9864 / 0.9856 against the H200 defaults of 8 November round 1 (3.780 → 3.728 ms); A100-SXM4 (8 Nov) off again from all ten 1.0050 / 1.0044; Ada (8 Nov) off again from the Ada set 1.0018 / 1.0028 | d66, 8 Nov; A100 d66, 8 Nov; Ada d66, 8 Nov | [h200-fastpath](../../results/h200-fastpath-20261108/README.md), [a100-omat0-best](../../results/a100-omat0-best-20261108/README.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| `STAGE_SH` (`wprime_backward.STAGE_SH_GPUS`; each 32-lane w′ cohort's edge features in shared memory once per edge) | – | – (0.9903 / 0.9905 off again) | on (A100 family) | bitwise | MD step H200 0.9955 / 0.9950; with STAGE_EDGES 0.9819 / 0.9811; A100-SXM4 (8 Nov) off again from all ten 1.0079 / 1.0077; POLAR-1-M (the only OFF or POLAR plan with wprime_backward) NaCl 0.9811 / 0.9914, NbCl5 0.9968 / 0.9977; Ada (8 Nov) off again from the Ada set 0.9903 / 0.9905 (slower on, off) | d66, 8 Nov; A100 d66, 8 Nov; Ada d66, 8 Nov | [h200-fastpath](../../results/h200-fastpath-20261108/README.md), [a100-omat0-best](../../results/a100-omat0-best-20261108/README.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| `COUNT_PLAN` (`compose_skip.COUNT_PLAN_GPUS`; COMPOSE_SKIP's species plan by a counting sort instead of an int64 radix sort per force call) | – | – (tie) | on (A100 family) | bitwise | MD step H200 0.9831 / 0.9818 against the H200 defaults of 8 November round 2; measured and off: R128 shape with `MIN_BLOCKS` 7 1.0097 / 1.0103, 8 1.0515 / 1.0507; A100-SXM4 (8 Nov) off again from all ten 1.0115 / 1.0130; Ada (8 Nov) off again from the Ada set 1.0000 / 1.0004, a tie, off | d66, 8 Nov; A100 d66, 8 Nov; Ada d66, 8 Nov | [h200-fastpath](../../results/h200-fastpath-20261108/README.md), [a100-omat0-best](../../results/a100-omat0-best-20261108/README.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| `JOINED_LAYOUT` (`compose_skip.JOINED_LAYOUT_GPUS`; with COMPOSE_FAST the first interaction's sums written d-major, so the four paths are row groups of one matrix read in place by `species_gemm.species_mm_rows`, one gradient tensor back) | – | – (tie) | on (A100 family) | bitwise | MD step H200 0.9805 / 0.9815 against the H200 defaults of 8 November round 3 (3.718 → 3.646 ms); measured and off: w′ `STAGE_EDGES` 0.9969 / 1.0010 (tie), forward `PREFETCH_AHEAD` 1.0046 / 1.0067; Ada (8 Nov) off again from the Ada set 1.0006 / 0.9986, a tie, off; A100 (Alex pack6, 8 Nov) 0.9782 / 0.9784 |  d66, 8 Nov; Ada d66, 8 Nov | [h200-fastpath](../../results/h200-fastpath-20261108/README.md), [omat0-ada-capacity](../../results/omat0-ada-capacity-20261108/README.md) |
| w′ `STAGE_EDGES` (`wprime_backward.STAGE_EDGES_GPUS`; with STAGE_DOUT, STAGE_SH and SPLINE_W the w′ chunk's edges staged per block) and forward `PREFETCH_AHEAD` (`spline_forward.PREFETCH_AHEAD_GPUS`; with STAGE_EDGES the table rows of the edge two positions on hinted into L1) | – | – | on (A100 family) | bitwise | MD step A100 (Alex pack6 against the A100 defaults of main 11297d45): w′ `STAGE_EDGES` 0.9643 / 0.9853, `PREFETCH_AHEAD` 0.9810 / 0.9943, with `JOINED_LAYOUT` all three 0.9590 / 0.9641; H200 off (0.9969 / 1.0010, 1.0046 / 1.0067) | d66, 8 Nov | [round6-a100](../../results/h200-fastpath-20261108/round6-a100/run.log) |
| `SORT_BY_LENGTH` (`neighbors.skin.SORT_BY_LENGTH_GPUS`; with `sort_receivers` each receiver's candidates, within its sender group, ordered by length at the skin-list build, so neighbouring receivers read the same spline-table slabs) | – | – | on (A100 family, except OFF23-small and -large: `SORT_BY_LENGTH_MODEL_OFF_GPUS`; alone 0.9924 / 0.9936, with `BLOCK_ORDER` 0.9920 / 0.9904; gate passes: forces 0.92 / 0.93 / 0.96, stress 0.64 / 1.02 / 1.15) | rounding (gate passes) | MD step H200 0.9807 / 0.9791 on top of the H200 defaults with `JOINED_LAYOUT` (3.585 → 3.514 ms); forward kernel 792 → 680 µs; gate forces 0.83 / 0.92 / 0.97, stress 0.48 / 0.94 / 1.07 | d66, 8 Nov | [slab-sorted](../../results/slab-sorted-h200-20261108/README.md) |
| `BLOCK_ORDER` (`spline_forward.BLOCK_ORDER_GPUS`; the SPLINE_W forward's blocks walk the receivers grouped by species, an int32 order the force field binds once) | – | – | on (A100 family) | bitwise | MD step H200 0.9753 / 0.9724 on top of `SORT_BY_LENGTH` (3.514 → 3.42 ms); lab forward 717 → 624 µs; A100 alone 0.9953 / 0.9945 (Alex pack 7) | d66, 8 Nov | [slab-sorted](../../results/slab-sorted-h200-20261108/README.md) |
| `PACKED8` (`spline_forward.PACKED8_GPUS`; with STAGE_EDGES the SPLINE_W forward reads an 8-byte copy of the spline table: the cubic in t − ½ economized to a quadratic, fields of 22 / 20 / 16 bits under one 6-bit exponent, decoded exactly as mantissas; the w′ kernel keeps the float32 table) | – | – | on (H200, A100-SXM4-80GB) | rounding (gate as the default) | MD step H200, SrTiO3 10×10×10: 300 K 0.9930 / 0.9945, 1500 K 0.9818 / 0.9768; A100-SXM4: 300 K 0.9557 / 0.9549, 1500 K 0.9587 / 0.9558 (its forward is L2-bound, step-limiters-20261109); kernel lab forward crystal 1.004, 0.3 Å disordered 0.858; Ada not measured | d66, 9 Nov | [packed-table-lab](../../results/packed-table-lab-20261109/README.md) |
| `GEOM` (`spline_forward.GEOM_GPUS`; with STAGE_EDGES the SPLINE_W forward computes the 54 CG sums of each edge's geometry shared by several paths once per edge) | – | – | on (A100 family) | rounding (gate passes) | A100-SXM4-80GB (Alex job 4973813, a = the A100 defaults): 0.9756 / 0.9806; kernel lab forward 0.867 / 0.868; gate with SC_FACTORED forces 0.863 / 0.913 / 0.985, stress 0.727 / 0.896 / 1.343 | d66, 10 Nov | [other-ways](../../results/other-ways-20261109/README.md#4-selective-shared-geometry-in-the-interaction-2-forward-faster-on-the-a100-follow-up-3) |
| `SC_FACTORED` (`sc_generated.SC_FACTORED_GPUS`; with SC_GENERATED the contraction through memoised CG pair products in a cheaper basis, the change of basis folded into the weights once at routing) | – | – | on (A100 family) | rounding (gate passes) | A100-SXM4-80GB (Alex job 4973813, a = the A100 defaults): 0.9889 / 0.9881; kernel lab pair 0.650 (226.0 → 147.0 µs); 5060 Ti lab 1.003 (memory-bound); H200 not adopted (sc0-invariant-basis: under 1 % there) | d66, 10 Nov | [other-ways](../../results/other-ways-20261109/README.md#3-factored-symmetric-contraction-faster-on-the-a100-follow-up-2) |
| `VIRIAL_ELEMENTWISE` (`rewrites.engine.VIRIAL_ELEMENTWISE_GPUS`; with single_graph_strain the strained shifts and positions as elementwise products, so the strain gradient is a reduction, not a split-K [3, E] x [E, 3] GEMM; stress calls only) | on | – (not measured) | – (A100-SXM4-80GB: bound 0.4 %; 18x18x18 0.9957 / 0.9944) | rounding (gate passes on the 5060 Ti and the A100) | force call with stress, SrTiO3 10x10x10: 5060 Ti 0.9675 / 0.9648 | 10 Nov | [virial-elementwise](../../results/virial-elementwise-20261110/README.md) |
| fused first-interaction backward (`FUSED_FIRST_BACKWARD_GPUS`) + tabled forward (`TABLED_FORWARD_GPUS`) | on | on | on | rounding | backward alone SrTiO3 / NbCl5: 5060 Ti 0.962 / 0.963, 0.972 / 0.969; Ada 0.962 / 0.963, 0.979 / 0.975; A100-SXM4-80GB 0.969 / 0.966, 0.983 / 0.984. Both: 5060 Ti 0.945 / 0.945, 0.948 / 0.955; Ada 0.949 / 0.944, 0.978 / 0.968; A100 0.966 / 0.962, 0.970 / 0.982 | A100 round 2, 5 Nov | [round-2 notes](../../plans/ideas-20261105-round2.md) |
| run-walking `edge_outer_sum` backward (`edge_outer_sum.BACKWARD_WALK_GPUS`) | – (kernel 0.94, near DRAM floor) | – (kernel 1.03–1.11) | on | rounding | A100-SXM4-80GB SrTiO3 0.995 / 0.987, NbCl5 0.995 / 0.996 (kernel 0.802) | A100 round 2, 5 Nov | [round-2 notes](../../plans/ideas-20261105-round2.md) |
| `EDGE_SH` Triton edge spherical harmonics (`EDGE_SH_GPUS`) | on | on | on | rounding (5.5e-7 / 1.4e-7 of float64 e3nn) | SrTiO3 / NbCl5: 5060 Ti 0.985 / 0.961, 0.990 / 0.993; Ada 0.989 / 0.987, 0.998 / 0.997; A100 PCIe 0.983 / 0.985, 0.995 / 0.988 | A100 round 5, 5 Nov | `results/round5-20261105` |
| `HEAD_SLICE` one-head readout as a slice (`energy.HEAD_SLICE_GPUS`) | on | on | on (A100-SXM4-80GB) | bitwise | SrTiO3 / NbCl5: 5060 Ti 0.996 / 0.998, 0.989 / 0.993; Ada 0.999 / 0.997, 0.993 / 0.992; A100 0.992 / 0.987, 0.975 / 0.979 | d66, 5 Nov | [round 5](../../plans/ideas-20261105-round2.md) |
| `GRAPH_SUM` one-graph energy sums as reductions (`energy.GRAPH_SUM_GPUS`) | – (tie) | – (1.001 / 0.999) | on (A100-SXM4-80GB) | rounding | A100 0.997 / 0.998, 0.992 / 0.998 | d66, 5 Nov | [round 5](../../plans/ideas-20261105-round2.md) |
| `SEGMENT_ROWS` in-place BlockLinear 1:1 products (`segment_rows.SEGMENT_ROWS_GPUS`) | – | – | on (A100-SXM4-80GB) | rounding (3xTF32) | A100 SrTiO3 0.985 / 0.984, NbCl5 0.981 / 0.988 | d66, 5 Nov | [round 5](../../plans/ideas-20261105-round2.md) |
| w′ end-of-edge reduction: `BUTTERFLY` (`BUTTERFLY_GPUS`), `REDUX` (`REDUX_GPUS`) | both on | – (REDUX 1.008–1.011) | BUTTERFLY on (H200, A100-SXM4-80GB); REDUX – (1.018–1.029) | BUTTERFLY bitwise; REDUX fixed point | 5060 Ti REDUX 0.998 / 0.999, 0.999 / 0.998; BUTTERFLY 9 Nov, SrTiO3 10×10×10 / NbCl5: H200 0.9863 / 0.9861, 0.9936 / 0.9903; A100-SXM4 0.9910 / 0.9892, 0.9942 / 0.9973 (gate passes; earlier off: A100 1.006, H200 0.9659 / 0.9677 against older w′ defaults) | d66, 5 and 9 Nov | [round 5](../../plans/ideas-20261105-round2.md) |
| `FOLD_FIRST_DENSITY_LINEAR` (True; follows `generic_fold_product_linear`) | – (fold off) | on | on | rounding | A100-SXM4-80GB SrTiO3 0.978 / 0.977, NbCl5 0.954 / 0.956; Ada 0.954 / 0.955, 0.920 / 0.920 | d66, 5 Nov | [round 6](../../plans/ideas-20261105-round2.md#round-6-skipping-work-around-the-big-kernels-5-november-2026) |

The A100 plan's one accepted gate failure: OMAT-0 NbCl5 stress 1.8e-7 eV/Å³ against a limit of 1.2–1.5e-7, from
the 3×TF32 GEMMs and the tensor-core forward together (each passes alone); the owner kept both (5 November 2026;
[omat0-tp-backward](../validation/omat0/omat0-tp-backward.md) §3.1).

### RTX 4000 Ada and RTX 5060 Ti re-screen (ada-5060-screen, 10–11 November 2026)

Every OMAT-0 medium switch that was on for the H200 or the A100 but never measured on these two GPUs, plus GEOM,
SC_FACTORED, VIRIAL_ELEMENTWISE and SEGMENT_ROWS, as same-process MD-step A/Bs in both build orders (Benchmark_loup
SrTiO3 10×10×10 and NbCl5), then leave-one-outs from the combined set, the 3× float32 gate and the other models.
Full record: [results/ada-5060-screen-20261110](../../results/ada-5060-screen-20261110/README.md).

- **Ada** (murphy; against main `4e9388c8` with the registry fix): PACKED8, SORT_BY_LENGTH, SC_GENERATED, SC_FACTORED,
  BUTTERFLY, w′ STAGE_DOUT, STAGE_SH, w′ STAGE_EDGES, POWER_BACKWARD on; set against today's Ada defaults SrTiO3
  0.8255 / 0.8194 (21.4 → 17.7 ms), NbCl5 0.8546 / 0.8558; gate passes. BUTTERFLY and STAGE_SH flipped from their
  5 and 8 November off readings. Other models: POLAR-1-M NaCl was slower (1.0061 / 1.0077) through w′ STAGE_DOUT,
  now off for POLAR-1-M and -L on the Ada (the set without it: ties, M 0.9986 / 1.0006, L 1.0005 / 1.0005);
  OFF23-medium a tie (only SORT_BY_LENGTH runs there); gates pass.
- **5060 Ti** (local): SPLINE_W, R128, spline STAGE_EDGES, PACKED8, SORT_BY_LENGTH, GENERIC_FOLD_READOUT_TAIL,
  POWER_BACKWARD, RECEIVER_SUM, w′ STAGE_EDGES / STAGE_DOUT / STAGE_SH, `generic_fold_product_linear` on; set
  against today's 5060 Ti defaults SrTiO3 0.6427 / 0.6478 (23.9 → 15.4 ms), NbCl5 0.6911 / 0.6906; gate passes
  (2×2×2 supercells: the float64 reference at 3×3×3 failed under WSL). POLAR-1: `gate_epilogue` for S, M, L and the
  w′ pair for POLAR-1-M by label; gates pass.

### Mixed cells under decision 66 (6 November 2026)

The owner left this rule to the conductor ("you can decide yourself"). It answers the kernel review's question:
does a per-GPU default have to win on every benchmark cell, or in aggregate?

A switch can be a GPU's default when both of these hold on that GPU's benchmark cells:

1. **A win:** on at least one cell, it is faster in both build orders.
2. **No loss:** on no cell is it slower in both build orders.

A cell where one order is faster and the other slower, each within 0.5 %, is a tie and counts as neither.

So `grouped_launch` (RTX 5060 Ti) and `gate_epilogue` (RTX 5060 Ti, RTX 4000 Ada) stay on:

| Switch | Win | Ties |
| --- | --- | --- |
| `grouped_launch` | NaCl | NbCl5 1.002 / 0.998, NaNbO3 1.001 / 0.999 |
| `gate_epilogue` | NbCl5 and NaNbO3 | NaCl 1.000 / 1.005 (5060 Ti) and 0.998 / 1.003 (Ada) |

A switch that loses on a cell in both orders needs a per-cell rule (a planner condition) or stays off.

### NVIDIA H200 (5 November 2026)

§0 has the H200 column; this section keeps the H200 evidence of 5–6 November 2026. The H200 (Modal, sm_90) is in these sets, every one from MD-step A/Bs in both
build orders on OMAT-0 medium, Benchmark_loup (b = switch off unless noted, so b/a above 1 means the switch is
faster): planner `DECISION_66_GENERIC_RADIAL_TABLES_GPUS` 2.030 / 2.027, `GENERIC_WPRIME` 1.771 / 1.768,
`TF32X3_OMAT0_GPUS` 1.068 / 1.069, `GENERIC_FOLD_PRODUCT` 1.033 / 1.032; w′ `FACTORED` 1.160 / 1.159, `TABLE_W`
1.142 / 1.141, `BLOCK_MESSAGE` 1.066 / 1.065, `HORNER` 1.046 / 1.046, `SCALAR_PATHS` 1.030 / 1.030; fused forward `TENSOR_CORES` 1.107 / 1.105, `SKIP_ZERO_COLUMN` 1.092 / 1.090, `VEC_A` 1.047 /
1.046, `TC_WFRAG` 1.022 / 1.024, `VEC_SH` 1.003 / 1.003, `DEGREE_PARTS` (b = on) 0.9818 / 0.9845; generic
`FUSED_FIRST_BACKWARD` 1.051 / 1.052, `EDGE_SH` 1.019 / 1.018; tabled `TABLED_FORWARD` (b = on, after the recapture
fix) 0.9838 / 0.9871; energy `HEAD_SLICE` 1.016 / 1.017, `GRAPH_SUM` 1.005 / 1.004; radial `SYMMETRIC_PAIRS` 1.005 /
1.004, radial-table forward 8 warps (`WARPS_GPUS`; b = on) 0.9946 / 0.9950. Spline forward `SPLINE_W` (b = on, 7 November) 0.8789 / 0.8775, with `R128` on top 0.9838 / 0.9848. `SPLINE_VECTOR` 4 on top (b = 4) 0.9739 / 0.9793. `COMPOSE_SKIP` on top (b = on) 0.9698 / 0.9738. On top of that (8 November): `COMPOSE_FAST` 0.9614 / 0.9656, `CUEQ_PROJECTED` 0.9902 / 0.9954, `GENERIC_FOLD_READOUT_TAIL` 0.9855 / 0.9901 (b = on). Then `STAGE_EDGES` 0.9864 / 0.9856 and `STAGE_SH` 0.9955 / 0.9950 (b = on). Then `COUNT_PLAN` 0.9831 / 0.9818 (b = on). Then `JOINED_LAYOUT` 0.9805 / 0.9815 (b = on), then the neighbour list's `SORT_BY_LENGTH` 0.9807 / 0.9791 and the spline forward's `BLOCK_ORDER` 0.9753 / 0.9724 (b = on); w′ `STAGE_EDGES` (0.9969 / 1.0010) and `PREFETCH_AHEAD` (1.0046 / 1.0067) off. `BUTTERFLY` was on from 5 November (b/a 0.994 / 0.994, b = on), off from 6 November (the A/B at `d599cc70` gave
0.9659 / 0.9677 with b = off, [sx012 profile](../../results/sx012-profile-20261106/README.md)) and is on again since
9 November (`bcca4189`: 0.9863 / 0.9861, b = on). Off there: `BACKWARD_WALK`,
`SEGMENT_ROWS`, `SPECIES_GEMM` (also with `species_gemm.CUTLASS`: 1.0141 / 1.0175), CUTLASS tile 32x64, `TABLE_PAIRS`,
`HOIST_SCALE`, `LINEAR_DIV_SPLIT` (0.9972 / 1.0025), `WGMMA`, `SEGMENT_GROUPED`, `SH_LDS4` (1.0014 / 1.0027), tabled kernels at 2 warps (forward 1.0147 /
1.0168, backward 1.0270 / 1.0295).
Sources: [h200-defaults](../../results/h200-defaults-20261105/README.md), [hopper sweep](../../results/hopper-sweep-h200-20261105/README.md),
[h200-ideas](../../results/h200-ideas-20261105/README.md), [h200-wgmma](../../results/h200-wgmma-20261105/README.md),
[wprime-pairs](../../results/wprime-pairs-20261105/README.md).

Two of these change precision: `TENSOR_CORES` (3xTF32 a·W product) and `TF32X3_OMAT0_GPUS` (CUTLASS 3xTF32 GEMMs).
On the A100 the owner approved both explicitly (3 and 4 November 2026).

**H200 approval (owner, 6 November 2026):** "go ahead and make them default if they are faster and accurate."
Both pass that test, so both stay on:
- Speed (MD step, both build orders): `TENSOR_CORES` 1.107 / 1.105, `TF32X3_OMAT0` 1.068 / 1.069.
- Accuracy: the force/stress gate with every H200 default on passes. The ratios to the float32 calculator's error
  (limit 2) are:

  | | NaCl | NbCl5 | SrTiO3 |
  | --- | ---: | ---: | ---: |
  | Force | 0.74 | 0.97 | 0.99 |
  | Stress | 1.02 | 1.93 | 0.93 |

  NbCl5 stress has the least headroom ([h200-defaults](../../results/h200-defaults-20261105/README.md)).
- This closes the kernel review's open item ([kernel-review-20261105](../validation/kernel-review-20261105.md)).

**Current H200 figures (main `7f7ef7db`, 6 November 2026):**
- Benchmark_loup: 5.495 ms per step (1.099 µs per atom and step).
- Force/stress gate: passes; NbCl5 stress 1.96 of 2.
- Source: [round 6](../../results/research-answers-20261105/README.md#h200-defaults-after-round-6-gate-and-benchmark_loup-at-7f7ef7db-6-november-2026).

`LANE_INDICES` (idea 15a, bitwise) joined the H200 set on 6 November 2026: MD step 0.9897 / 0.9913 (b = on)
([ledger: ideas-r3](../optimisation-ledger.md#ideas-r3)).

### OMAT-0 small (`omat0-small`, 8–9 November 2026)

OMAT-0 small has had a label since 8 November 2026 (`labels.EXTRA_LABELS`: in the table, recognised by SHA-256, but
not one of the 19 acceptance labels). It runs the per-GPU switches as they are, plus these per-model defaults:

| Default | Table | GPUs | Evidence (Benchmark_loup, b/a ab / ba) |
| --- | --- | --- | --- |
| 3×TF32 GEMMs and the generic w′ backward with its fused forward (decision 66 pair) | `DECISION_66_TF32X3_MODEL_GPUS`, `DECISION_66_GENERIC_WPRIME_LABEL_GPUS` | H200, A100 family | H200 (b = without): 2.360 / 2.370 (5.41 → 2.29 ms); gate force 1.01, stress 1.22 (NbCl5). A100-SXM4 (Alex 4967329, b = with): 0.3911 / 0.3916 (10.72 → 4.19 ms); A100 gate (4967421) force 1.03, stress 1.33 (NbCl5) |
| w′ `STAGE_EDGES` on (bitwise) | `wprime_backward.STAGE_EDGES_MODEL_GPUS` (on top of `STAGE_EDGES_GPUS`) | H200 | b = off: 1.0072 / 1.0114 |
| Spline forward R128 shape with `MIN_BLOCKS` 8 instead of 6 (bitwise) | `spline_forward.R128_MIN_BLOCKS_MODEL_GPUS` | H200 | b = 8: 0.9873 / 0.9883, 0.9923 / 0.9952; as the default, b = 6: 1.0046 / 1.0057. Medium keeps 6 (8: 1.0515 / 1.0507) |
| Spline forward `PREFETCH_AHEAD` off (bitwise) | `spline_forward.PREFETCH_AHEAD_MODEL_OFF_GPUS` | A100 family | Alex 4967540 b = off 0.9902 / 0.9908; as the default, b = on 1.0068 / 1.0073 (4967601) |

The kernels read the table label from the rewrites' `Context.inputs["model_label"]`, through WPrime's `model_label`.
Small's step (step breakdown, 9 November 2026): H200 2.12 ms (5.41 ms without the pair), A100-SXM4 4.19 ms (10.72 ms without), 4.17 ms with PREFETCH_AHEAD off (A/B).
w′ `FOLD_ALPHA` off measured within the run-to-run spread (0.4–0.7 % either way), so it stays on
(`FOLD_ALPHA_MODEL_OFF_GPUS` is empty). The other GPUs keep the pair off until they have their own A/B. Source:
[h200-omat0-small](../../results/h200-omat0-small-20261108/README.md).

### Tiled (capacity) mode

| Switch | 5060 Ti | Ada | A100 | Evidence | Report |
| --- | --- | --- | --- | --- | --- |
| `TileConfig.compiled_graphs` None (`tiled.COMPILED_GRAPH_GPUS`, rounding: the atomics' order): compiled tile bodies in CUDA graphs behind a memory guard (device state) | off (30³ 1.007 / 0.990, 40³ 0.998) | off (40³ 1.0015 with graphs on; 58³ guard off, 1.0010 / 1.0007) | on (0.569 / 0.571; gate 3x passed, Alex 4982466) | 975,560 SrTiO3, R 16,384 budget; the guard measures the first compiled call and captures where one pool per slot fits | [tiled-speed-20261111](../../results/tiled-speed-20261111/README.md) |
| `TileConfig.lanes` `auto` (`tiled.LANE_GPUS`, exact): two sweep-2 tiles on disjoint SM halves | – (1.05× per pair) | on (1.09× per pair) | – | under `cudaMallocAsync` | [tiled-concurrent-tiles](../validation/tiled/tiled-concurrent-tiles.md) |
| `TileConfig.backward_chunk_scale` `auto` (`tiled.TILE_CHUNK_SCALE`, exact) | ×4 | ×4 | ×1 | 1.08M SrTiO3, wavefront, R 8,192: 5060 Ti −10.5 %, Ada −3.2 %; RTX PRO 6000 tiles do not chunk | `results/tiled-ideas-20261102` |
| `tiled.TILE_PREFETCH_DEVICE` (`TILED_SPEED_GPUS`, exact): prefetch with the state on the device | on | on | on | OMAT-0 SrTiO3 975,560, device state: 0.939 / 0.941 | [tiled-speed](../../results/tiled-speed-20261108/README.md) |
| `generic.TILE_BLOCK_ROWS` (every GPU, rounding): w′ block message for the receivers only | on | on | on | 5060 Ti device state 0.918 / 0.921; wavefront 0.931 / 0.937 (1.030 / 1.060 while host-bound) | [tiled-speed](../../results/tiled-speed-20261108/README.md) |
| `tiled.TILE_VIRIAL` (`TILED_SPEED_GPUS`, rounding): strain gradient from the edge gradients | on | on | on | 0.970 / 0.955; wavefront 0.994 / 0.959; gate 12/12 (5060 Ti); Ada and A100 gates pass with all nine off and all on | [tiled-speed](../../results/tiled-speed-20261108/README.md) |
| `tiled.TILE_GRAPH_CACHE` (`TILED_SPEED_GPUS`, exact): tile graphs kept for the call (25 % of free memory less the workspace; wavefront: `TILE_GRAPH_CACHE_WAVE`) | on | on | on | 0.948 / 0.960; all four 0.795 / 0.787 (8.38 → 6.66 s) | [tiled-speed](../../results/tiled-speed-20261108/README.md) |
| `tiled.TILE_FILTER32` (`TILED_SPEED_GPUS`, exact): kept lists' cutoff test in float32, float64 near the cutoff | on | on | on | 0.987 / 0.993; wavefront 0.981 / 1.012 | [tiled-speed](../../results/tiled-speed-20261108/README.md) |
| `tiled.TILE_EPILOGUE` (`TILED_SPEED_GPUS`, rounding): sweep masks, objective, virial and edge vectors compiled | on | on | on | 0.980 / 0.981; wavefront 0.985 / 0.983 (1.012 / 1.009 while host-bound); gate 12/12 (5060 Ti); Ada and A100 gates pass with all nine off and all on | [tiled-speed](../../results/tiled-speed-20261108/README.md) |
| `tiled.TILE_FILL_SKIP` (every GPU, exact): no list-cache fill for graph-cached tiles | on | on | on | 0.961 / 0.966 | [tiled-speed](../../results/tiled-speed-20261108/README.md) |
| `tiled.TILE_PINNED_SUPERSET` (`TILED_SPEED_GPUS`, exact): kept lists pinned on the host, up to 4 GiB of allocator blocks, returned when the lists go | on | on | on | 0.986 / 0.990; wavefront 0.989 / 0.990 (1.012 / 1.028 while host-bound) | [tiled-speed](../../results/tiled-speed-20261108/README.md) |
| `tiled.TILE_GRAPH_CACHE_WAVE` (`TILED_SPEED_GPUS`, exact): the graph cache under the wavefront, search pass to sweep 3 | on | on | on | wavefront 0.824 / 0.817 | [tiled-speed](../../results/tiled-speed-20261108/README.md) |
| `tiled.TILE_HOST_WORKER` (`TILED_SPEED_GPUS`, exact): host-state writes applied on a worker thread | on | on | on | wavefront 0.975 / 0.982; host state 0.969 / 0.966 | [tiled-speed](../../results/tiled-speed-20261108/README.md) |
| `tiled.TILE_COPY_STREAM` (`TILED_SPEED_GPUS`, exact): host-state write copies on a side stream from device clones | on | on | on | wavefront 0.962 / 0.964; host state 0.960 / 0.959 | [tiled-speed-20261110](../../results/tiled-speed-20261110/README.md) |
| `TileConfig.state_dtype` (every GPU, aliased host state): `"auto"`, the call's dtype while the host state fits; else the kept lists go, then float16 (fails the 3x gate: a capacity fallback) | auto | auto | auto | f16 store 0.983 / 0.977; host state halved | [tiled-host-state-dtype](../validation/tiled/tiled-host-state-dtype.md) |

`TILED_SPEED_GPUS` (10 November 2026): the RTX 5060 Ti, the RTX 4000 Ada and every A100. Every switch above marked `TILED_SPEED_GPUS`, off against on as one set, 975,560 SrTiO3 atoms, both orders: Ada device state 0.868 / 0.868, wavefront 0.772 / 0.769; A100-SXM4-80GB device state 0.792 / 0.789, wavefront 0.782 / 0.785 ([tiled-speed-20261110](../../results/tiled-speed-20261110/README.md)). The switches were not measured one by one on these GPUs. The 3× float32 force/stress gate (R = 48, compiled), all nine off and all on, passes on the Ada and the A100-SXM4 (the rounding switches `TILE_VIRIAL` and `TILE_EPILOGUE`, decision 66).

## 2. Default MH-1 float32 plan (GPU-independent)

Compiled for the ledger at main `8a269a91` (28 October 2026); rule names checked against `fastmace/planner/fixed.py` at `eca7ba5d`.

`auto` rules as in [switch-map](switch-map.md) §1–§4 and `fastmace/planner/fixed.py` on main `8a269a91`.
"RE default" means the research engine's `main.py` default as listed in SPEC §6.2 ("defaults as in `main.py` on 16
October 2026"), carried over in phase 1a.

RewriteConfig, on in the default MH-1 float32 plan:

| Field | `auto` rule | Made default by |
| --- | --- | --- |
| `factor_species` | `on_for_hybrid_graph_engine_if_model_supports` | RE default (9 Oct 2026, production decision); SPEC §6.2 |
| `fold_cutoff` | `on_with_factor_species_if_model_supports` | RE default (9 Oct 2026); SPEC §6.2 |
| `density_first` | `on_with_fold_cutoff_if_model_supports` | RE default (10 Oct 2026); SPEC §6.2 |
| `reassociate_first` | `on_with_stage1_fold_cutoff_if_model_supports` | RE default (10 Oct 2026); SPEC §6.2 |
| `joint_radial_input` | `on_with_fold_cutoff_if_model_supports` | RE default (14 Oct 2026); SPEC §6.2 |
| `fused_folded_backward` | `on_with_stage1_fold_cutoff_if_model_supports` | RE default (14 Oct 2026); SPEC §6.2 |
| `folded_backward_tiles` | `on_with_fused_folded_backward` | Owner, 21 Oct 2026 (OPT-DEF-2), SPEC §19 thirteenth amendment |
| `segmented_first_sum` | `on_with_reassociate_first_and_sort_receivers` | RE default (14 Oct 2026); SPEC §6.2 |
| `streaming_conv` | `on_for_hybrid_with_stage1_fold_cutoff_and_sort_receivers_if_model_supports` | RE default (14 Oct 2026); SPEC §6.2 |
| `block_linear` | `on_for_hybrid_graph_engine` | RE default (14 Oct 2026); SPEC §6.2 |
| `grouped_linear`, `merged_residual` | `on_with_block_linear` | RE default (16 Oct 2026); SPEC §6.2 |
| `gated_rows` | `on_with_merged_residual` | Owner, 20 Oct 2026 night, §19 twelfth amendment |
| `segment_gemm` | `on_with_gated_rows_float32_unless_reproducible` | Owner, 20 Oct 2026, §19 twelfth amendment |
| `species_tables` | `on_unless_reproducible_if_model_supports` | Owner, 21 Oct 2026 (OPT-DEF-2), thirteenth amendment; float64 by decision 63 |
| `fold_product_linear` | `on_with_species_tables_and_block_linear_if_model_supports` | as `species_tables` |
| `fold_readout_tail` | `on_with_block_linear_unless_reproducible_if_model_supports` | as `species_tables` |
| `chunked_conv_backward` | `on_with_streaming_conv_and_fused_folded_backward` | Decision 60 (27 Oct 2026; §19 "Optimisation round 6") |
| `radial_tables` | `on_with_fold_cutoff_unless_reproducible` | Decision 62 (27 Oct 2026); class `certified` by decision 61; float64 by decision 63 |
| `fused_folded_forward` | `on_for_mh1_float32_unless_reproducible_and_for_omat0_with_wprime_on_decision_66_gpus` | Decision 64 (27 Oct 2026); OMAT-0 with wprime, decision 66 (3 Nov 2026) |
| `single_graph_strain` | `on` (acts only where stress is computed) | RE default (16 Oct 2026); SPEC §6.2 |

RewriteConfig, off or not applicable in the MH-1 float32 plan: `species_skip` (generic family only,
`on_with_graph_engine_if_model_supports`, RE 15 Oct 2026); `grouped_launch` (generic family only, decision 57);
`fx_e3nn_linear`, `polar_graph_sums` (POLAR-1 only, decision 58); `polar_uvu` (`True`, acts on POLAR models only, RE
16 Oct 2026); `fused_radial_hidden`, `direct_node_feats`, `folded_constants`, `combo_kernels`, `tf32x3_gemms`
(`never_automatic`); `inplace_block_linear` (`False`, refused if set); `w_tables`, `save_transcendentals` (float64
only, decision 65); `selfcheck_cache` (`"use"`, inert, not built).

EngineConfig: `backend` hybrid (`backend_hybrid_for_graph_engine`; RE production decision 9 Oct 2026);
`compile_mode` `"default"` (RE, 8–9 Oct 2026, no autotuning); `pipelined` (`on_with_graph_engine_unless_speculative`,
RE 14 Oct); `async_output` (`on_with_graph_engine`, RE 15 Oct); `speculative_graph` `False` (OPT-IN);
`first_derivative_only` `True`, `unfuse_full_addmm` `True`, `expandable_segments` (`on_for_torch_2_14_unless_alloc_conf_set`)
(RE 7–8 Oct); `dynamic_edges` `True` (decision 59); `model_template_cache` `True` (conductor on merge, twelfth
amendment mentions it); `overlap_rebuild` `True` (OPT-DEF-2, thirteenth amendment); `step_graph` `False` (OPT-IN);
`fused_integrator` `True` (owner 21 Oct 2026, fourteenth amendment); `cueq_conv_fusion` (plain cuEq only,
`on_for_plain_cueq_unless_reproducible`, RE 16 Oct); `recapture_budget` `"auto"`, `graph_variant_cap` 64,
`dynamo_cache_size_limit` 64, `empty_cache_before_capture` `True`, `warmup_in_graph_pool` `True` (constants).
The CUDA-graph tier is the default tier (rule 9 as amended by decision 38); `AdaptationConfig.adapt` `"auto"`
(decisions 52–53); `CalibrationConfig.calibrate` `False` (decision 55); `RunnerConfig.persistent` `True` (POSIX,
conductor decision 20 Oct 2026).

PrecisionConfig: `dtype` float32 (`float32_for_hybrid_float64_for_cueq`); `tf32` `False` (never default, SPEC §1.2);
`float64_energy` `True` (1b-T01/T03).

NeighbourConfig: `skin` `"auto"` (0.3 Å for MD workloads; rule 11 for MC, species-change and displacement workloads,
3-T05); `sort_receivers` (`on_for_hybrid_graph_engine`, RE 10 Oct); `gpu_neighbour_list` (`on_with_graph_engine`,
RE 14 Oct); `edge_multiple` 256, `edge_bucket_fraction` 0.05, `capacity_headroom` 1.15, `headroom` `"auto"`,
`d3_skin` `"auto"` (rule 11, 3b-T02), `edge_index_int32` `"auto"`: on on the RTX 4000 Ada, the RTX 5060 Ti and every A100 (decision 66, 11 Nov; b/a 1.0001 / 0.9987, 1.0022 / 0.9959 and A100 0.9960 / 0.9951, gate passed on the Ada and A100; 8 B per edge, 0.8 kB/atom), off on the H200 until measured. `IntegratorConfig.noise_batch` 256, the next batch drawn ahead on a background thread (`langevin.NOISE_DRAW_AHEAD`, every GPU, bitwise; A100 PCIe step 0.97 / 0.96 against off; round 6, 5 Nov); `fixcm_projection` `mass_weighted` under
NVT fixcm (decision 37, correctness).

## 3. Default MH-1 float64 plan: extra switches

| Field / part | Rule | Made default by |
| --- | --- | --- |
| Hybrid backend for non-reproducible float64 plans | planner backend choice | Decision 63(a) |
| `fp64_emulation` (cuBLAS INT8 emulation, eager strategy) | `on_for_float64_cuda_with_reduced_rate_fp64_unless_reproducible` (measured FP64 rate < 1/8) | Decision 63 |
| `species_tables`, `fold_product_linear`, `fold_readout_tail` in float64 | rules without `float32` | Decision 63(c) |
| `radial_tables` in float64 (degree 16, tolerance 3e-13) | `on_with_fold_cutoff_unless_reproducible` | Decision 63(d) |
| `chunked_conv_backward` float64 chunk rule (4 L2s) | library rule inside `l2_chunk_edges` | f64-T03 (`79fb1115`), no decision number |
| `fp64_emulation_mantissa_bits` | `dynamic_decision_65_d1_suspended_after_recapture_defect` (was `56_for_mh1_float64_on_measured_gpus_decision_65_else_dynamic`) | Decision 65(a), suspended by the owner on 28 Oct 2026 (auto = `"dynamic"` everywhere; explicit requests stay); now on main |
| `w_tables` | `on_for_mh1_float64_on_measured_gpus_decision_65` | Decision 65(b) |
| `save_transcendentals` | `on_for_mh1_float64_on_measured_gpus_decision_65` | Decision 65(c) |
| Not in float64: `segment_gemm`, `grouped_launch`, `fused_folded_forward` (Triton/NVRTC float32 only) | — | — |

## 4. Speed reference (MH-1, 3×3×3)

MD step time in ms, NVT 300 K unless stated. Rows from different days or processes are not A/B comparisons.

| Date | Engine and plan | GPU | NaCl | NaNbO3 | NbCl5 | Commit | Source |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 2026-10-07 | RE: ASE path, FP32 | local | 41.4 | – | – | – | `ResearchModule:LT/studies/optimization_history/report.md` |
| 2026-10-08 | RE: cuEq FP32 / hybrid FP32 | local | 9.668 / 7.075 | 84.587 / 55.920 | 72.286 / 55.650 | – | `ResearchModule:LT/studies/hybrid_oeq/rtx5060ti.md` |
| 2026-10-14 | RE: best default after streaming conv + `ir_mul` | local | ≈ 3.75 (3.71) | ≈ 23.1 (22.9) | ≈ 28.8 (28.6) | – | `ResearchModule:LT/docs/roadmap.md` item 13 / `status.md` |
| 2026-10-16 | RE: + grouped linear + merged residual | murphy | 3.55 | 25.74 | 30.54 | – | `ResearchModule:LT/studies/other_models/README.md` |
| 2026-10-20 | FM before round 1 (2-T20a) | local / murphy | 3.71 / 3.68 | – | 27.43 / 31.58 | not recorded | [2-T20a](../validation/phases/2-T20a-step-breakdown.md), [opt-1 plan](../../plans/opt-1.md) |
| 2026-10-21 | FM after round 2 (OPT-T10, 1000 K, wall) | local / L4 | 3.028 / 4.264 | 18.788 / 29.589 | 21.434 / 33.126 | `072fa0a` | [OPT-T10](../design/speed/opt-T10-step-breakdown.md) |
| 2026-10-25 | FM 0.x default, median of 4 blocks | L4 (Colab) | 3.95 | 31.5 | 35.2 | `c1f4de9c` | [benchmarks](../benchmarks.md) |
| 2026-10-27 | FM default + `radial_tables` (before `fused_folded_forward`) | leto | 2.80 | 18.03 | 21.75 | `0cba0d7a` | [opt6-T05](../validation/speed/opt6-T05.md) |
| 2026-10-27 | FM default + `fused_folded_forward` (today's float32 default), side b, both orders | local | 2.654 / 2.709 | 14.56 / 14.58 | 18.67 / 18.62 | `9eb1ab1e` (pinned) | [opt8-T02](../validation/speed/opt8-T02.md) |
| 2026-10-27 | same | murphy | 2.811 / 2.809 | 17.37 / 17.50 | 21.54 / 21.62 | `059b2cfa` | [opt8-T02](../validation/speed/opt8-T02.md) |
| 2026-10-27 | same, order a-b only | L4 (Colab) | – | 26.06 | 31.47 | `9eb1ab1e` | [opt8-T02](../validation/speed/opt8-T02.md) |
| 2026-10-27 | FM float32 default, step breakdown at 1000 K | local | 2.78 | 15.31 | 19.21 | branch `opt7-plan` | branch fastmace/opt7-plan: plans/opt-7.md §1 |
| 2026-10-27 | FM float64: stock → decision-63 package | local | 81.9 → 46.9 | 516 → 255 | 652 → 275 | `7562a425` | [f64-T01](../validation/speed/f64-T01.md) |
| 2026-10-27 | FM float64: stock → package | leto | 76.3 → 44.8 | – | 579 → 233 | not recorded | [f64-T01](../validation/speed/f64-T01.md) |
| 2026-10-28 | FM float64: package + D1 + D2 + D3, ABBA ratio | local / Ada | 0.806 / 0.804 | 0.782 / – | 0.797 / 0.802 | not recorded | [ideas](../../plans/ideas-20261027.md) |
| 2026-10-26 | FM 1.0.0, OMAT-0 medium float32 | local | 1.651 / 1.652 | – | 9.901 / 9.921 | 1.0.0 | [ALCHEMI comparison](../validation/comparisons/alchemi-comparison-20261025.md) |

A100 float32 numbers for the MH-1 default are not recorded (ledger, "Open or untried"). OMAT-0 on the A100: [omat0-a100.md](omat0-a100.md).
