# 2-T04: unknown MACE versions, always-on treatments, the self-check reaction

Plan task 2-T04 (`plans/phase-2.md`), SPEC §6.3, §6.4, §9.3 rule 13, §12.1 and §18 decision 2:
MACE 0.3.16 is the minimum and the only qualified version. Another version runs through the
versioned registry, and nothing is patched that was not hashed.

## The rule

Every decision is made by source hashes (sha256 of `inspect.getsource`), never by the version
string. The version only appears in reasons and in the reference record's id.

| What | Sources hash | Sources do not hash |
| --- | --- | --- |
| A rewrite record (`rewrites/registry.py`, `qualify`) | enabled as resolved | off, with the reason logged; the planner continues |
| `float64_energy` (`model/energy.py`) | patched forward | the force precision is promoted to float64 (even when float32 was asked for), and the float64 model sums its energy natively. The reference tier is the last resort |
| PolarMACE forward (`longrange/polar.py`) | patched (PR 1762 port, stress fix, float64 energy) | PolarMACE refused: the stress fix (MACE issue 1642) needs the patched code |
| `capture_safe_mask_head` | installed | MACE's mask kept; a multi-head model cannot be captured, so the eager tier |
| `first_derivative_only`, `single_graph_strain`, `unfuse_full_addmm` | installed | not installed; MACE's or torch's own function stays |
| `cueq_conv_fusion` | installed | not installed; MACE's own conversion rule stays |

`fastmace.rewrites.versions.treatments()` lists these decisions with their reasons.

**Planner** (`MinimalPlanner._common`):
- The `force_precision` decision is the effective precision. A promotion carries a
  `Reason` with code `precision-promoted-unqualified-energy`, and the requested precision is kept
  in `plan.requested`. This happens at revision 0, before anything is built, so no `Plan.revise`
  is involved.
- The run report prints the promotion as a downgrade ("why") line.
- PolarMACE without its hashed forward is refused (rule 1, code `polar-unqualified-source`).
- The default never promotes: on MACE 0.3.16 every hash matches (tested).

Still open: `capture_support` is not yet a planner decision (`fixed.py` belongs to T02). Without
it, a multi-head model on a MACE with an unhashed mask fails at capture (`UnqualifiedPatchError`),
and the fallback (T15) takes it to eager. On 0.3.15 the mask hashes.

**Hash gates.** Every process-wide patch in `fastmace._patches` checks each source it replaces
against `SOURCE_PINS` and raises `UnqualifiedPatchError` otherwise.
`_patches.install_if_qualified(name)` is for callers that can run with MACE's own function: it
skips the patch and logs why. The CUDA-graph capture path calls the setter itself, because it
cannot run without the capture-safe mask.

**Force field.** `MACEForceField.energy_treatment` is one of three values:
- `"patched"`: the qualified forwards;
- `"native"`: a float64 model with an unqualified forward;
- `"off"`.

A float32 model with an unqualified forward is refused (`EnergyReductionError`); the planner
promotes before that happens. `fm.ForceField.identity` takes the effective precision, so a
resume on the same MACE re-plans the same way.

Known limitation: the identity assumes `float64_energy=True`. A plan with
`PrecisionConfig(float64_energy=False)`, which exists for A/B against the research engine only,
runs at float32 on a MACE whose float64 energy forward does not hash, but its identity records
float64. Fixing it needs the plan's decided precision in `Simulation`, which records the identity.

**Reference records.** A version other than 0.3.16 gets an "unqualified stock" record
(`record_for(version, family)`, `record.qualified` is False):
- stock equivalence only;
- PR 1498's strain is stock only when its source hashes;
- the yardstick's float64 accumulation only when the float64 forward hashes. Otherwise the
  `float32_sum_unqualified` entry keeps MACE's own float32 sum, with
  `info()["energy_bound"]` set to the looser bound, which the gate record carries under
  `references.yardstick` and the report's gate line names. Production then runs float64 (the
  promotion), so its own energy error stays small;
- PolarMACE is refused.

`content_digest()` is the digest without the version, for 2-T09's resume rule. The 0.3.16
record ids are unchanged.

## Rule 13: the self-check reaction

`Registry.apply` raises `SelfCheckError` on a failed check, carrying:
- `rewrite` and `result`;
- `applied`, the records applied before it;
- `reaction`, a `Reaction` from `Registry.disable_with_dependants`: the failed record plus every
  enabled record that requires it, transitively, their config fields, and the resolution without
  them.

If the failed record had installed a process-wide patch, that patch is undone.
`fastmace.planner.reactions.selfcheck_revision(plan, error)` returns the next plan revision:
- the fields are `"off"` (`False` for a bool field);
- the decisions carry codes `selfcheck-failed` and `selfcheck-dependant`, rule 13;
- the reason reads `rule 13: rewrite X failed its self-check (...); X, Y off`;
- step 0.

The revision is built again from the pristine model (decision 24). The fallback (2-T15) calls
it. 2-T04 does not touch `planner/minimal.py`.

Per-record self-check results:
- `MACEForceField.selfcheck_records()` (`registry.selfcheck_records`);
- `Plan.selfchecks` (`SelfCheck` records) on the plan that `MinimalPlanner.build` attaches to the
  engine (`engine.plan`). `Simulation` (T15) and provenance (T17) pick it up from there.

## MACE 0.3.15 against the pins

mace-torch 0.3.15 is the newest release before 0.3.16 (PyPI has nothing newer). On murphy, with
its wheel on `PYTHONPATH` in front of `mace_cueq_latest` (torch 2.14.1, graph_longrange 0.4.0):

| Source | 0.3.15 | Users |
| --- | --- | --- |
| `mace.modules.blocks:RealAgnosticResidualNonLinearInteractionBlock.forward` | differs | `factor_species`, `fold_cutoff` (MH-1) |
| `mace.modules.utils:get_symmetric_displacement` | differs | `single_graph_strain` |
| `mace.cli.convert_e3nn_cueq:run` | differs | `cueq_conv_fusion` |
| `mace.modules.extensions:PolarMACE.forward` and the other polar sources in MACE | missing | PolarMACE |
| `mace.modules.field_blocks:SparseUvuTensorProduct.*` | missing | `polar_uvu` |
| `MACE.forward`, `ScaleShiftMACE.forward` | match | `float64_energy` |
| `mace.modules.irreps_tools:mask_head` | match | `capture_safe_mask_head` |
| `compute_forces`, `compute_forces_virials`, `RadialMLP.forward`, the generic blocks, graph_longrange | match | |

So on 0.3.15, MH-1 runs with its factored and folded rewrites off, MACE's own strain and the
patched float64 energy, with no precision promotion. The promotion and the PolarMACE refusal are
tested with fake hashes (`tests/cpu/test_versioned_registry.py`).

## The other-MACE environment and its run

`~/environments/fastmace_mace_other` (local WSL, 20 October 2026) is a copy of
`mace_cueq_latest` (torch 2.14.1+cu130, cuEquivariance 0.12.0, graph_longrange 0.4.0) with
mace-torch 0.3.15 instead of 0.3.16:
- the dry run changed nothing else;
- `pip check` is clean;
- the list is `~/environments/fastmace_mace_other.requirements.txt`, which differs from
  `mace_cueq_latest`'s in the MACE line only.

FastMACE runs there from the checkout (`PYTHONPATH`), as on every GPU environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=$PWD ~/environments/fastmace_mace_other/bin/python -m pytest -n 0 tests/cpu/test_unknown_mace.py
```

Both tests pass there: 2 passed, and `test_versioned_registry.py`'s fake-hash tests skip off
0.3.16. The tiny MH-1 `fm.md` run on NaCl completes with `check="gate"` and the gate passes. Every
unhashed record is off in the built force field, and every patch in force hashes.

Two compatibility fixes came out of that run:
- `_patches.cueq_conv_fusion(enabled, device)` patches nothing on an unhashed converter when
  MACE's own rule already gives the asked-for fusion on that device (the CPU reference);
- `loading` recognises a cuEquivariance calculator of a MACE that does not record
  `_enable_cueq` (0.3.15) by its modules.

Still open: plain cuEq on the GPU asks for the fused convolution on `cuda:0`, which MACE's own
rule does not give. On an unhashed converter that conversion raises `UnqualifiedPatchError`. The
planner's `cueq_conv_fusion` decision is `fixed.py`'s (T02), and the hybrid default does not
convert through this path.

## GPU check: MH-1 on NaCl, MACE 0.3.15 (local RTX 5060 Ti)

Run on 20 October 2026: local PC (WSL2), RTX 5060 Ti 16 GB, `fastmace_mace_other` (mace-torch
0.3.15, torch 2.14.1+cu130, cuEquivariance 0.12.0), FastMACE at 67c09ff. MH-1 `matpes_r2scan` on
NaCl 3x3x3 (216 atoms), NVT 300 K, 200 steps, `check="gate"`, plain cuEq (no OEq in this
environment) with `cueq_conv_fusion="off"`. The run did not complete. It failed twice, on
grounds outside the hash rules:

1. **No compilation on 0.3.15.** `model/compile.py` imports
   `mace.tools.compile.configure_autograd_for_compile`, which MACE 0.3.15 lacks (`ImportError`
   at build).
2. **No CUDA graph capture of the uncompiled 0.3.15 model.** With `compile_mode="none"`, MACE
   0.3.15's `RealAgnosticResidualNonLinearInteractionBlock.forward` (unhashed, so its MH-1
   rewrites are off) calls `torch.nonzero(node_attrs)` inside the capture
   (`cudaErrorStreamCaptureUnsupported`).

So on 0.3.15 the GPU path that is left is the eager tier (2-T02, `mode="eager"`). Two pieces
reach it: 2-T15's fallback takes a capture failure to eager, and a planner rule leaves
compilation off when MACE lacks its compile API. The GPU part of acceptance 1 is re-run after
2-T02 and 2-T15 merge. The CPU part, where nothing is compiled or captured, passes (see above).
