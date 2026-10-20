# Rewrite screens: adopted and rejected

SPEC §6.2 asks for this record of every screen, adopted or not, so that work is not repeated. The adopted
rewrites are the registry records (`fastmace/rewrites/`, SPEC §6.2 catalogue). This file lists FastMACE's
own screens and summarises the research engine's rejected ones.

## FastMACE screens

| Date | Screen | Exactness | Result | Decision | Report |
| --- | --- | --- | --- | --- | --- |
| 2026-10-20 | OPT-T02 stage A, `gated_rows`: MH-1's residual products, gate and `linear_2` on per-segment rows, without the joins and copies between them | exact (bitwise on the CPU and in the reproducible GPU mode) | same-process A/B, 12 blocks, both orders: murphy (sm_89) NaCl −4.4 %, NbCl5 −9.0 %; local RTX 5060 Ti (sm_120) NaCl −3.7 %, NbCl5 −10.6 %. Force gate passed (NaCl, NbCl5) | **Adopted**, default on wherever `merged_residual` is (owner decision, 20 October 2026; planner rule `on_with_merged_residual`). The check of all models (the 19 labels through the gate) runs in the combined acceptance after the round; a model that fails there gets `gated_rows` switched off | `docs/validation/opt-T02a-gated-rows.md` |
| 2026-10-20 | OPT-T02 stage B, `segment_gemm`: in the `gated_rows` chains, `linear_2`'s products write straight into its joined output (Triton gathered-row GEMM where the shared shape rule picks it, cuBLAS otherwise), and the interaction's `reshape_irreps` becomes a view | rounding (another GEMM), deterministic | same-process A/B on top of `gated_rows`, 12 blocks, both orders: murphy (sm_89) NbCl5 0.952/0.948, NaCl 0.997/0.996; local (sm_120) NbCl5 0.958/0.957, NaCl 0.997/0.992 (NaCl takes the stock path). Force gate passed | **Adopted**, default on with `gated_rows` in float32 plans that are not reproducible (owner decision, 20 October 2026; planner rule `on_with_gated_rows_float32_unless_reproducible`); float64 and `reproducible=True` keep the stock path. The 19-label gate in the combined acceptance after the round checks all models; a model that fails gets it switched off | `docs/design/opt-T02-BC-go-no-go.md` |
| 2026-10-20 | OPT-T01: FP32-accurate tensor-core GEMMs (Triton `tf32x3` 3×TF32; cuBLAS BF16×9; CUTLASS 3×TF32) | rounding (3×TF32 error 5.3e-7 against float64, below cuBLAS IEEE's 1.3e-6) | sm_89: tf32x3 slower than cuBLAS SIMT FP32 on 89 of 92 production shapes; per force call NbCl5 atom GEMMs 15.6 ms against 10.4 ms, NaCl 3.3 against 1.9 ms. Peak 4096³: cuBLAS FP32 16.3, TF32 38.6, tf32x3 8.4 TFLOPS, so three TF32 products cannot beat FP32 (ceiling 12.9). BF16×9 exists only on compute capability 10.0/10.3. | **Rejected** (sm_89); sm_120 rerun for the record pending | `docs/validation/opt-T01-tensor-cores.md` |
| 2026-10-20 | OPT-T01b: tensor-core work beside cuBLAS's SIMT FP32 GEMM (column split or duplicate on two streams, one captured graph) | speed only | murphy (sm_89, 48 SMs). With the FP32-accurate partner, Triton tf32x3, the combined rate is 0.77–0.85× of cuBLAS FP32 alone on 9720/13608 × 512 × 512 (best split 75/25), 0.98× on 1944 × 512 × 384 and 0.82× on 216 × 512 × 512. Duplicate (both whole products at once): 0.75–0.79× on the large shapes. Only 13–29 % of the shorter part overlaps. Even 1×TF32 (not FP32-accurate) only reaches 1.10–1.36×, mostly because it is 2.3× faster alone. | **Rejected** (rule: < 1.10×). Both GEMMs fill every SM (hundreds of tiles on 48 SMs), so the second stream only gets the first one's tail. Co-resident warps share the schedulers' issue slots and the register file, and tf32x3 spends its own slots on its three products and the operand splits. | `results/opt-T01b-tensor-core-overlap/murphy/`, `tools/gemm_overlap_bench.py` |
| 2026-10-20 | OPT-T03 candidates 1–2: a block linear's independent per-l products on 2 or 4 CUDA-graph branches (cuBLAS), or in one grouped Triton launch | exact (streams); rounding (grouped) | murphy (sm_89), one MH-1 block linear (4 × 512 → 512), graph replays: streams 0.85–1.11×, grouped 1.00/1.16× on NaCl forward/input gradient (7 % together) and 0.97/1.03× on NbCl5. The four products together already run at 9.6–11.1 TFLOPS on NaCl | **Not built** on sm_89 (rule: ≥ 10 % of NaCl's atom GEMM time, no NbCl5 loss); sm_120 half pending | `docs/design/opt-T03-small-m.md` §6 |
| 2026-10-20 | OPT-T02 stage D: gate backward fused into `linear_2`'s input-gradient epilogue | rounding | not built: its per-atom reduction needs atomics or whole-atom tiles | **Rejected** (conductor): non-deterministic gradients would break the reproducible mode and the float64 bitwise promise | `docs/design/opt-T02-fusion.md` |

## The research engine's rejected screens (10–16 October 2026)

ResearchModule `Experiments/Langevin_Thermostat/studies/further_optimizations/` and
`studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`; local RTX 5060 Ti unless noted. Each failed for the
reason given, so it should not be repeated without new evidence.

| Idea | Result | Why it failed |
| --- | --- | --- |
| GEMM-only autotuning (Inductor `max_autotune_gemm`) | −0.1 to −1.0 % (slower); setup 141–756 s | per-shape tuning cost, no gain after species factoring; excluded by SPEC §9.1 anyway |
| TF32 | changes the accuracy | production decision: FP32 with TF32 off (SPEC §1) |
| Grouped node GEMMs | ceiling 0.1 ms | Triton IEEE-FP32 GEMMs slower than cuBLAS for these shapes |
| In-place / strided-batched block linear | exact; +0.1 to +0.3 ms per step | cuBLAS's batched tiles make the l = 3 product 415 against 305 µs; tile choice per GPU would be tuning |
| Contiguous folded radial weight | within noise, twice | – |
| Inductor `decompose_mm_pass` for the joint radial layer | no gain | – |
| Fused radial hidden block (Triton) | no gain after the 15 October fix | kept as opt-in record `fused_radial_hidden` |
| Fused gate/residual VJP (stand-alone kernel) | ceiling 2–2.5 % | the gate already runs at memory bandwidth; a gain needs fusion into the GEMMs (OPT-T02 now) |
| Joint-layer Triton input gradient | < 0.1 ms; slower on small cells | – |
| Segmented force accumulation | 21 µs | below the 0.1 ms bar |
| Activation memory budget < 1 (recompute in the backward) | +0.4 to +5.9 ms | – |
| Programmatic dependent launch (`triton.enable_pdl`) | wrong forces in CUDA-graph replays | **never enable** (SPEC §6.2) |
| Radial sharing between mirror edges | ceiling already slower (+1.1 ms NaNbO3) | – |
| Tabulated radial networks | ceiling 2.9–5.1 %, approximate | would spend force-error budget; not built |
| Species + Morton atom order | no gain | features and weights already L2-resident |
| Skin 0.2 Å, 2 % edge buckets | no gain | – |
| Speculative AC+B graph | −4 to −15 % (slower) | kept as opt-in `speculative_graph` |
| L2-resident edge chunks with unpatched OEq | much slower | OEq 0.7 always returns full-size outputs |
| Receiver-owned streaming backward; register caps | slower | – |
| OpenEquivariance 0.7 symmetric contraction | exact, 27–79× slower compiled | the hybrid keeps cuEq's contraction |
| Reduced Clebsch–Gordan basis | ≈ 1 % of a 0.7 ms kernel | no measurable gain |
| MACE develop since 0.3.16 | PRs 1498, 1641, 1711, 1718 already taken | nothing else left |
