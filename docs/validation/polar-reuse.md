# PolarMACE with the OMAT-0 and MH-1 optimisations (polar-T01, 5 November 2026)

Task: test MACE-POLAR-1 (PolarMACE) in FastMACE and make it as fast as possible, starting by reusing what rounds
1–5 built for MH-1 and OMAT-0. Branch `fastmace/polar-T01`, from main `eca7ba5d`.

**Verdict (first step):**
- PolarMACE already loads and runs in FastMACE (SPEC §4.3, §11.1: POLAR-1 S/M/L on MH-1's interaction class, plus
  the polar path, `polar_uvu`, `fx_e3nn_linear` and `polar_graph_sums`). The CPU tests now cover the rest of the
  MH-1 rewrite package on a tiny PolarMACE, long range included: all of it matches MACE to float64 rounding.
- POLAR-1 gets the rewrites that follow MH-1's *class*, but none of those gated on MH-1's *label*. Decisions 64 and
  66 enabled them only where they were measured, and POLAR-1 was never measured. So POLAR-1 runs without
  `fused_folded_forward`, `wprime_backward`, `gate_epilogue`, `grouped_launch` and `tf32x3_gemms`. It also misses
  every per-GPU kernel flag that only acts inside those two kernels (TABLE_W, BLOCK_MESSAGE, FACTORED, HORNER,
  SCALAR_PATHS, VEC_SH, VEC_A, TC_WFRAG, TENSOR_CORES, SKIP_ZERO_COLUMN).
- An explicit `on` of each of those five is accepted for POLAR-1, so they can be measured without code changes. On
  leto (RTX 4000 Ada, §4), `wprime_backward` + `fused_folded_forward` + `gate_epilogue` make POLAR-1-M's MD step
  3.4 % (NaCl) and 4.2 % (NbCl5) faster and pass the force gate. POLAR-1-S has a single interaction, so only
  `gate_epilogue` applies (−0.7 % on NbCl5, neutral on NaCl).
- Three items need code: EDGE_SH (attached only by the generic family's streaming record, so MH-1 lacks it too),
  the species-table family (pinned to `ScaleShiftMACE.forward`), and a profile of PolarMACE's long-range part,
  which none of the MH-1 work touches.

## 1. What PolarMACE is

`mace.modules.extensions.PolarMACE(ScaleShiftMACE)`, MACE 0.3.16, needs `graph_longrange==0.4.0` (the `[polar]`
extra). Its local part is MH-1's: `RealAgnosticResidualNonLinearInteractionBlock` with the cutoff passed
(`apply_cutoff=False`); POLAR-1-S has one interaction, M two, L three (`tests/cpu/test_planner.py` `LABEL_FACTS`).
On top of that, its forward:

- feeds every layer's node features into `lr_source_maps` (spin-charge sources), the last layer's into the Fukui
  source map, and the stack of all layers into `layer_feature_mixer`;
- builds a k-space grid (`compute_k_vectors_flat`), precomputes the field geometry, and runs a fixed number of
  self-consistent recursion steps for the charge density and the electrostatic potential;
- adds per-graph inputs: `total_charge`, `total_spin`, `fermi_level`, `external_field`.

FastMACE's polar path (`fastmace.longrange.polar`, SPEC §11.1) replaces the k-vector list with a fixed-size,
capturable one, fixes the stress geometry (MACE issue 1642) and carries PR 1711's uvu GEMM and PR 1762's options.

What this means for reuse: everything inside the interactions carries over unchanged. A rewrite that changes what
the model does *after* an interaction (the product's linear, the last readout) must keep every layer's node features
intact, because the long-range heads read them.

## 2. The optimisations and their reuse

POLAR-1's resolved plan today (`FixedPlanner`, NaCl, NVT, float32 hybrid; the same on the RTX 5060 Ti, the RTX 4000
Ada and the A100-SXM4-80GB): on: the MH-1 package (factor_species, fold_cutoff, density_first, reassociate_first,
joint_radial_input, fused_folded_backward, segmented_first_sum), streaming_conv, chunked_conv_backward,
radial_tables, block_linear, grouped_linear, merged_residual, gated_rows, segment_gemm, polar_uvu, fx_e3nn_linear,
polar_graph_sums, CUDA graphs (graph tier). Pinned by `tests/cpu/test_polar_reuse.py`.

| Optimisation | Switch (default) | MH-1 / OMAT-0 status | POLAR-1 today | Reuse | Why |
| --- | --- | --- | --- | --- | --- |
| MH-1 package (factored radial, folded cutoff, density first, reassociated first interaction, segmented first sum, fused folded backward) | `RewriteConfig` fields, `auto` | default for MH-1's class | **on** | as is | same interaction class; CPU tests pass |
| Streaming convolution, chunked backward | `streaming_conv`, `chunked_conv_backward` | default (hybrid) | **on** | as is | class-level |
| Block linears, `gated_rows`, `segment_gemm` | `block_linear` …, `segment_gemm` (float32) | default | **on** | as is | CPU tests pass (float64) |
| Radial tables (certified) | `radial_tables` | default with fold_cutoff, float32 | **on** | as is | certified on the tiny PolarMACE |
| CUDA graphs, pipelined steps, GPU neighbour list | `EngineConfig`, `NeighbourConfig` | default | **on** | as is | the polar path is capturable (§11.1) |
| `fused_folded_forward` | `auto`: MH-1 (decision 64), OMAT-0 with wprime (66) | default for MH-1 and OMAT-0 | off | **as is, needs a measurement** | POLAR-1-S alone 1.004, neutral (OPT8-T02, 27 Oct); not measured with wprime, which is how OMAT-0 gains (alone 29 % slower on the A100) |
| `wprime_backward` | `auto`: MH-1 and OMAT-0 on the decision-66 GPUs | default; MH-1 −11 to −19 % on the A100 | off | **as is, needs a measurement** | same kernels as MH-1; label-gated only |
| TABLE_W, BLOCK_MESSAGE, FACTORED, HORNER, SCALAR_PATHS, MAX_REGISTERS | `kernels.oeq_streaming.wprime_backward` per-GPU lists | default on the listed GPUs | off (no wprime) | as is, comes with `wprime_backward` | model-agnostic kernel flags |
| VEC_SH, VEC_A, TC_WFRAG, TENSOR_CORES, SKIP_ZERO_COLUMN | `kernels.oeq_streaming.fused_forward` per-GPU lists | default on the listed GPUs | off (no fused forward) | as is, comes with `fused_folded_forward` | model-agnostic kernel flags |
| `gate_epilogue` | `auto`: MH-1 on RTX 5060 Ti and RTX 4000 Ada | default there | off | **as is, needs a measurement** | label-gated only |
| `grouped_launch` | `auto`: generic family; MH-1 on the RTX 5060 Ti | default there | off | as is, needs a measurement (5060 Ti only) | MH-1 lost 0.5 % on sm_89 |
| `tf32x3_gemms` | `auto`: MH-1 and OMAT-0 on the A100 | default there | off | as is, needs an A100 measurement | label-gated only |
| `w_tables`, `save_transcendentals` | `auto`: float64 MH-1 (decision 65) | default for float64 MH-1 | off | as is, needs a float64 measurement | label-gated only; POLAR-1 float64 is the gate's reference |
| EDGE_SH (Triton edge spherical harmonics) | `rewrites.generic.modules.EDGE_SH_GPUS` | default for OMAT-0 on 5060 Ti, 4000 Ada, A100 | off | **needs adaptation** | attached only in `generic_streaming`; the MH-1 class (MH-1 too) never gets it. PolarMACE calls `spherical_harmonics` the same way |
| `species_tables`, `fold_product_linear` | `auto`, MH-1 class | default for MH-1 | refused | **needs adaptation** | pinned to `ScaleShiftMACE.forward`; PolarMACE's first input is still `node_embedding(node_attrs)`, so the tables apply in principle; needs PolarMACE's forward pinned and a check that every layer's features stay exact |
| `fold_readout_tail` | `auto`, MH-1 class | default for MH-1 | refused | **probably does not apply** | it folds the last product into the readout; PolarMACE also reads the last features in the Fukui, source and mixer heads |
| `species_residual`, `fused_radial_hidden`, `direct_node_feats`, `folded_constants`, `combo_kernels` | opt-in | rejected or neutral on MH-1 | off | not pursued | no gain on MH-1 |
| Generic-family records (`generic_*`, species skip, generic radial tables, generic fold) | generic family | OMAT-0 defaults | n/a | does not apply | PolarMACE is not the generic family; MH-1's own records cover the same ground |
| Rejected lab items (TC_TP, 3xFP16, BLOCK_LDG, PREFETCH, launch screens) | — | rejected | — | does not apply | rejected on OMAT-0 |

## 3. CPU checks (this branch)

`tests/cpu/test_polar_reuse.py` (19 tests), on a tiny PolarMACE (Na, Cl, O; Fe absent) in float64, every record
through the registry with its self-check, and the whole model's energy and forces (long range included) against
MACE:

- MH-1 package + `polar_uvu` on e3nn, and + block linears + `gated_rows` on cuEq (naive CPU kernels): self-checks at
  most 2e-15, energy 2.4e-15 and forces 1.2e-14 relative;
- `radial_tables` in float32: certified, energy 4e-7 and forces 1.3e-5 relative (MH-1's tiny model: 3e-8 / 3.5e-5);
- the planner's POLAR-1 defaults on the three decision-66 GPUs as in §2, each label-gated rewrite accepted when
  asked, and an explicit `species_tables` refused with the pin reason.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ~/environments/fastmace_cpu/bin/python -m pytest -n 2 --dist=load --maxschedchunk=1 tests/cpu/test_polar_reuse.py
```

`19 passed`. These need `graph-longrange==0.4.0` in the CPU environment (installed into `fastmace_cpu` on
5 November 2026 for this task); without it they skip, as the other PolarMACE tests do.

## 4. First GPU measurement: leto, RTX 4000 Ada

leto, NVIDIA RTX 4000 Ada Generation (driver 595.99.02), environment `mace_cueq_oeq` from the checkout
(`PYTHONPATH`), hybrid backend (cuEq + OEQ streaming), float32, FastMACE source `1ff585fa` (this branch; the
driver `tools/polar_reuse_ab.sh`), checkpoints `MACE-POLAR-1-S.model` and `MACE-POLAR-1-M.model`. No baseline engine
is involved: side a is FastMACE's default POLAR-1 plan, side b the same plan with the named fields set to `on`. The
idle check passed before each job (12:20:44Z and 12:37:38Z, 5 November 2026; GPU at 2 MiB, 0 %, no user processes).
Same-process MD-step A/Bs, NaCl and NbCl5 3×3×3, 300 warm-up steps, 8 blocks of 200, both build orders (`ab` with
pattern abab, `ba` with abba); every A/B reported valid. Raw JSON: `results/polar-reuse-leto-20261105/`.

Step time b/a (lower is faster; the two numbers are the two orders):

| Model | b adds | NaCl | NbCl5 |
| --- | --- | --- | --- |
| POLAR-1-S | `gate_epilogue` | 0.9984, 1.0005 (2.669 ms) | 0.9935, 0.9927 (22.39 ms) |
| POLAR-1-M | `wprime_backward` | 0.9676, 0.9667 | 0.9782, 0.9781 |
| POLAR-1-M | `wprime_backward` + `fused_folded_forward` | 0.9629, 0.9646 | 0.9701, 0.9706 |
| POLAR-1-M | `gate_epilogue` | 1.0019, 0.9991 | 0.9893, 0.9893 |
| POLAR-1-M | all three | **0.9659, 0.9659** (6.15 → 5.94 ms) | **0.9574, 0.9580** (50.90 → 48.75 ms) |

- **POLAR-1-M gains 3.4 % (NaCl) and 4.2 % (NbCl5)** with the three label-gated rewrites, the same direction as
  MH-1. The pieces add up: wprime 2–3 %, the fused forward a further 0.5–0.8 %, the gate epilogue 1 % on NbCl5 and
  nothing on NaCl.
- **Force gate (POLAR-1-M, all three on): passed** on NaCl and NbCl5 for energy, forces and stress (forces NaCl
  3.4e-6 eV/Å against the float32 ASE reference's 3.8e-6, NbCl5 8.8e-4 against 1.6e-3; stress ratios 0.55 and 0.53).
- **POLAR-1-S:** `gate_epilogue` is 0.6–0.7 % faster on NbCl5 and neutral on NaCl. `wprime_backward` and
  `fused_folded_forward` act only on interactions after the first, and S has only one, so they do not apply. The
  planner nevertheless accepts an explicit `on` for S, and the build then fails (`RewriteError: wprime_backward: no
  interaction with float32 radial tables and the chunked streaming convolution`); the S gate run therefore stopped at
  its production job. This also explains decision 64's neutral 1.004 for POLAR-1-S's fused forward.
- Not measured yet: POLAR-1-L, the RTX 5060 Ti and the A100 (where `tf32x3_gemms` also applies), `grouped_launch`.

## 5. Plan for the rest

1. **Label-gated rewrites (no code).** Measure on POLAR-1 S, M and L, NaCl and NbCl5 3×3×3, both build orders, with
   the force gate: `wprime_backward` + `fused_folded_forward` on the RTX 5060 Ti, RTX 4000 Ada and A100;
   `gate_epilogue` on the 5060 Ti and 4000 Ada; `tf32x3_gemms` on the A100; `grouped_launch` on the 5060 Ti. Where
   they win, the owner decides whether `_mh1` in `planner.fixed` should take the POLAR-1 labels (a decision-66
   amendment), as decision 58 did for `fx_e3nn_linear`. Done for POLAR-1-M and S on the RTX 4000 Ada (§4):
   M wins with all three. If the owner extends the gate, `wprime_backward` and `fused_folded_forward` should go to
   models with at least two interactions only, not to POLAR-1-S.
2. **Plan-time refusal.** The planner accepts `wprime_backward=on` (and the fused forward) for a one-interaction
   model, and the build then raises. It should refuse at plan time with the reason, or the `auto` rule should skip
   such models, so an extended gate cannot break POLAR-1-S.
3. **EDGE_SH for MH-1's class.** Attach `EdgeSH` in MH-1's `streaming_convolutions` as `generic_streaming` does, with
   a CPU test and an A/B on MH-1 and POLAR-1 (it would change MH-1's default too, so it needs the owner).
4. **Species tables on PolarMACE.** Pin `PolarMACE.forward` next to `ScaleShiftMACE.forward` in
   `rewrites.species`, check that the first input is species-only and that every layer's features stay within the
   threshold, then measure. `fold_readout_tail` stays refused unless it keeps the last layer's features.
5. **Profile the long-range part.** With the local part at MH-1 speed, the k-space recursion, the float64 sums and
   the self-consistent steps become a larger share. Nsight profile of one POLAR-1-S force call on NbCl5, split into
   local and long range, before any long-range rewrite is designed.
6. **CPU environment.** Add `graph-longrange==0.4.0` to the CPU CI recipe (plan task 0-T02's environment), so the
   PolarMACE CPU tests run in CI instead of skipping.
