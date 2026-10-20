# Rewrite screens: adopted and rejected

SPEC §6.2 asks for this record of every screen, adopted or not, so that work is not repeated. The adopted
rewrites are the registry records (`fastmace/rewrites/`, SPEC §6.2 catalogue). This file lists FastMACE's
own screens and summarises the research engine's rejected ones.

## FastMACE screens

| Date | Screen | Exactness | Result | Decision | Report |
| --- | --- | --- | --- | --- | --- |
| 2026-10-20 | OPT-T01: FP32-accurate tensor-core GEMMs (Triton `tf32x3` 3×TF32; cuBLAS BF16×9; CUTLASS 3×TF32) | rounding (3×TF32 error 5.3e-7 against float64, below cuBLAS IEEE's 1.3e-6) | sm_89: tf32x3 slower than cuBLAS SIMT FP32 on 89 of 92 production shapes; per force call NbCl5 atom GEMMs 15.6 ms against 10.4 ms, NaCl 3.3 against 1.9 ms. Peak 4096³: cuBLAS FP32 16.3, TF32 38.6, tf32x3 8.4 TFLOPS, so three TF32 products cannot beat FP32 (ceiling 12.9). BF16×9 exists only on compute capability 10.0/10.3. | **Rejected** (sm_89); sm_120 rerun for the record pending | `docs/validation/opt-T01-tensor-cores.md` |
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
