# PolarMACE with the OMAT-0 and MH-1 optimisations (polar-T01, 5 November 2026)

Task: test MACE-POLAR-1 (PolarMACE) in FastMACE and make it as fast as possible, starting by reusing what rounds
1–5 built for MH-1 and OMAT-0. Branch `fastmace/polar-T01`, from main `eca7ba5d`.

**Verdict:**
- PolarMACE loads and runs in FastMACE (SPEC §4.3, §11.1: POLAR-1 S/M/L on MH-1's interaction class, plus the polar
  path, `polar_uvu`, `fx_e3nn_linear` and `polar_graph_sums`). The CPU tests cover the MH-1 rewrite package, the
  linears, the radial tables and now the species tables on a tiny PolarMACE, long range included: all within float64
  rounding of MACE (float32 noise for the tables).
- Until this task POLAR-1 got the rewrites that follow MH-1's *class* but none gated on MH-1's *label*
  (`fused_folded_forward`, `wprime_backward`, `gate_epilogue`, `grouped_launch`, `tf32x3_gemms`). Measured on leto
  (RTX 4000 Ada, §4), the first three make POLAR-1-M's MD step 3.4 % (NaCl) and 4.2 % (NbCl5) faster and pass the
  force gate. **Now default on the RTX 4000 Ada** (owner, 5 November 2026): the pair for POLAR-1-M and -L,
  `gate_epilogue` for every POLAR-1. The RTX 5060 Ti and the A100 are not measured and keep them off.
- POLAR-1-S has one interaction, which is reassociated; `wprime_backward`, `fused_folded_forward` and `w_tables` act
  only on later ones. The planner used to accept them and the build failed; they are now off (auto) or refused
  (explicit) at plan time.
- Built in this task, off until measured (A/B constants; leto runs queued): EDGE_SH and FUSED_FIRST_BACKWARD for
  MH-1's class (MH-1 lacked both too). TABLE_W already works on MH-1 as an A/B constant. `species_tables` now takes
  PolarMACE (default on by its class rule, as for MH-1); the two layer folds stay refused.

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
| `fused_folded_forward` | `auto`: MH-1 (decision 64), OMAT-0 and POLAR-1-M/L with wprime (66) | default for MH-1 and OMAT-0 | **on** for M/L on the Ada; never for S | as is | POLAR-1-S was neutral (1.004) because it has no second interaction; M/L with wprime, §4 |
| `wprime_backward` | `auto`: MH-1 and OMAT-0 on the decision-66 GPUs; POLAR-1-M/L on the Ada | default; MH-1 −11 to −19 % on the A100 | **on** for M/L on the Ada | as is | §4; 5060 Ti and A100 not measured |
| FACTORED, SCALAR_PATHS, MAX_REGISTERS | `kernels.oeq_streaming.wprime_backward` per-GPU lists | default on the listed GPUs | with `wprime_backward` | as is | model-agnostic kernel flags |
| TABLE_W (+ HORNER), BLOCK_MESSAGE | `TABLE_W`, `BLOCK_MESSAGE` | default for the generic family only (`table_w_default(generic)`) | off | TABLE_W: the A/B constant works on MH-1's class, measurement queued; BLOCK_MESSAGE needs `gated_rows` / `residual_linear` to take grouped tuples | correction: an earlier draft said these come with `wprime_backward`; they do not |
| VEC_SH, TC_WFRAG, TENSOR_CORES | `kernels.oeq_streaming.fused_forward` per-GPU lists | default on the listed GPUs | with the fused forward | as is | model-agnostic kernel flags |
| VEC_A, SKIP_ZERO_COLUMN | fused forward | A100; Ada and A100 | off in effect | VEC_A needs a kernel change (k = h + 1 with the bias column); SKIP_ZERO_COLUMN does not apply (MH-1's last column is the bias, not zero) | from the OMAT-0 audit |
| `gate_epilogue` | `auto`: MH-1 on RTX 5060 Ti and RTX 4000 Ada; POLAR-1 on the Ada | default there | **on** on the Ada | as is | §4: NbCl5 0.6–1.1 % faster, NaCl neutral |
| `grouped_launch` | `auto`: generic family; MH-1 on the RTX 5060 Ti | default there | off | as is, needs a measurement (5060 Ti only) | MH-1 lost 0.5 % on sm_89 |
| `tf32x3_gemms` | `auto`: MH-1 and OMAT-0 on the A100 | default there | off | as is, needs an A100 measurement | label-gated only |
| `w_tables`, `save_transcendentals` | `auto`: float64 MH-1 (decision 65) | default for float64 MH-1 | off | as is, needs a float64 measurement | label-gated only; POLAR-1 float64 is the gate's reference |
| EDGE_SH (Triton edge spherical harmonics) | `EDGE_SH_GPUS` (generic), `EDGE_SH_MH1_GPUS` (MH-1 class, empty) | default for OMAT-0 on 5060 Ti, 4000 Ada, A100 | off | **built** (MH-1's streaming record attaches it), measurement queued | POLAR-1's harmonics are l ≤ 3, normalised, component |
| FUSED_FIRST_BACKWARD (+ TABLED_FORWARD) | `FUSED_FIRST_BACKWARD_GPUS` (generic), `FUSED_FIRST_MH1_GPUS` (MH-1 class, empty) | default for OMAT-0 on 5060 Ti, Ada, A100 | off | **built** (MH-1's reassociated first interaction), measurement queued | the folded weight [W, b] reads the table's leading h + 1 columns |
| `species_tables` | `auto`, MH-1 class | default for MH-1 | **on** (class rule) | **adapted**: `PolarMACE.forward` pinned | CPU within float64 rounding; GPU A/B queued |
| `fold_product_linear` | `auto`, MH-1 class | default for MH-1 | refused | does not apply | it changes layer 0's features, which PolarMACE's long-range heads read |
| `fold_readout_tail` | `auto`, MH-1 class | default for MH-1 | refused | does not apply | it changes the last layer's features, which the Fukui, source and mixer heads read |
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

## 5. Status and what is left

Done in this task (5 November 2026):
1. **Label-gated rewrites:** measured on the RTX 4000 Ada (§4); default there for POLAR-1 (`POLAR_DECISION_66_*` in
   `planner.fixed`; owner: "yes turn it on for 2 interaction models"). SPEC §19 entry, defaults-by-gpu and the ledger
   updated.
2. **Plan-time refusal** of `fused_folded_forward`, `wprime_backward` and `w_tables` on one-interaction models.
3. **EDGE_SH for MH-1's class:** `generic.modules.attach_edge_sh`, applied by MH-1's streaming record on
   `EDGE_SH_MH1_GPUS` (empty) or with the `EDGE_SH` constant.
4. **Species tables on PolarMACE:** `PolarMACE.forward` pinned in `rewrites.species` (the polar path's hash); the
   folds stay refused.
5. **FUSED_FIRST_BACKWARD for MH-1's class** (from the OMAT-0 audit): `mh1.modules.fused_first_message`, on
   `FUSED_FIRST_MH1_GPUS` (empty) or with the `FUSED_FIRST_BACKWARD` constant.
6. **CPU environment:** the CPU CI and the `fastmace_cpu` recipe install `[polar]`.

Left:
1. **Measurements queued on leto** (RTX 4000 Ada, `tools/polar_reuse_ab.sh`): POLAR-1-L A/Bs and its force gate at
   2×2×2 (the float64 reference of 3×3×3 NbCl5 runs out of the 20 GB); POLAR-1-M species tables, EDGE_SH, TABLE_W,
   FUSED_FIRST_BACKWARD with the gate and a `tools/module_profile.py` profile (local against long range); MH-1
   EDGE_SH, TABLE_W, FUSED_FIRST_BACKWARD. Each switch becomes a default only where these win (decision 66).
2. **The RTX 5060 Ti and the A100:** not measured. Modal jobs are paused (owner, 5 November 2026);
   `tools/modal/polar_a100.sh` is ready. The Modal image needs `MODAL_IMAGE_BUILDER_VERSION=2025.06` (or a Python
   3.11 image) since the workspace's builder moved to 2026.10.
3. **From the OMAT-0 audit, not built:** VEC_A for k = h + 1 (kernel change), BLOCK_MESSAGE for MH-1's consumers,
   HEAD_SLICE for multi-head checkpoints with a fixed head.
4. **Long range:** design long-range rewrites only after the profile in item 1.
