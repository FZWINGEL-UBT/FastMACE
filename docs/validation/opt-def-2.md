# OPT-DEF-2: three more defaults (owner decision, 21 October 2026)

The owner's words: "Make all 3 defaults". These are the OPT-T08 records (`species_tables`,
`fold_product_linear`, `fold_readout_tail`), `EngineConfig.overlap_rebuild` (OPT2-T04) and
`folded_backward_tiles` (OPT2-T03). SPEC: thirteenth amendment of draft 0.12.

## What changed

| Switch | Before | Now |
| --- | --- | --- |
| `RewriteConfig.species_tables` | `never_automatic` | `on_float32_unless_reproducible_if_model_supports` |
| `RewriteConfig.fold_product_linear` | `never_automatic` | `on_with_species_tables_and_block_linear_if_model_supports` (off in float64 and reproducible plans, and with `species_tables` off) |
| `RewriteConfig.fold_readout_tail` | `never_automatic` | `on_with_block_linear_float32_unless_reproducible_if_model_supports` |
| `RewriteConfig.folded_backward_tiles` | `never_automatic` | `on_with_fused_folded_backward` (stage 2 follows `fold_cutoff`: MH-1 class only) |
| `EngineConfig.overlap_rebuild` | `False` | `True` (acts only on pipelined fixed-cell CUDA steps) |
| `EngineConfig.step_graph` | `False` | unchanged |

**The models' scope.** New `rewrite_support` keys `species_tables` and `fold_readout_tail` carry each
record's own applicability, so stage 2 drops a record where the model does not take it, with the
model's reason. Of the 19 labels:
- `species_tables` and `fold_product_linear` apply to the MH-1 labels.
- `fold_readout_tail` applies to those and OMOL-0.
- `folded_backward_tiles` applies to the MH-1 class (MH-1, OMOL-0, POLAR-1).
- The research-engine plan comparison leaves them out, each with its own scope (as in `32018ad`).

## Float64 and the reproducible mode (follow each record's exactness)

| Switch | Class | In float64 and `reproducible=True` plans | Evidence |
| --- | --- | --- | --- |
| `fold_readout_tail`, `fold_product_linear` | rounding | **off** (as `segment_gemm`) | CPU float64 within 1e-12 relative of the default |
| `species_tables` | exact on the CPU, **not bitwise on the GPU** | **off** | Reproducible GPU mode, plain cuEq float64, local RTX 5060 Ti, main `f993713`, first force: off against off bitwise; `species_tables` on against off: energy bitwise, forces differ by 7.9e-16 (NaCl) and 5.6e-15 eV/Å (NbCl5). The tables come from GEMMs over the 89 identity rows, which round differently from GEMMs over n atoms (`results/opt-def-2/c1-repro.json`) |
| `folded_backward_tiles` | exact | **on** | BLOCK_K unchanged, so every row reduces in the same order; the kernel screen found the tiles bitwise equal (`docs/validation/opt2-T03.md`) |
| `overlap_rebuild` | exact | **on** | Reproducible GPU mode, NaCl 1000 K, 600 steps, 122 early rebuilds beside the force graph: bitwise (`docs/validation/opt2-T04.md`) |

An explicit `"on"` is kept in float64. `fold_product_linear="on"` without `species_tables` is refused
there.

## Results the decision rests on

| Switch | RTX 5060 Ti (sm_120) | Modal L4 (sm_89) | Modal A100 (sm_80) |
| --- | --- | --- | --- |
| OPT-T08, all three | NaCl 0.908 / 0.904, NbCl5 0.924 / 0.928; force gate passed | NaCl 0.918 / 0.920, NbCl5 0.922 / 0.920; force gate passed | NaCl 0.907 / 0.896, NbCl5 0.907 / 0.906 |
| `overlap_rebuild` | NaCl 1000 K 0.964 / 0.967; NbCl5 0.993–0.999 | NaCl 300 K 0.996 / 0.990, 1000 K 0.988 / 0.987; NbCl5 0.996 / 1.003 | – |
| `folded_backward_tiles` | NaCl 0.983 / 0.981 | NaCl 1.005 / 1.000, NbCl5 1.000 / 1.001 (neutral) | – |

All entries are same-process A/B ratios (on/off) in both build orders. Sources:
`docs/validation/opt-T08-linear.md`, `opt2-T03.md`, `opt2-T04.md`, `results/modal-*`.

## Checks of this change

| Check | Where | Result |
| --- | --- | --- |
| Targeted CPU tests (planner, minimal planner, config, report, capabilities, rewrites, overlap, tiles, fallback, provenance, restart, research surface, simulation) | local `fastmace_cpu`, `-n 2` | 763 passed, 45 skipped. The 19-label default-plan and research-engine comparisons pass with each default's own scope |
| Force gate with the new defaults (no `--rewrites`; the production plan has the three records on through `auto`) | local RTX 5060 Ti, pinned `953aa72`, 18:20–18:25 UTC | **passed**. NaCl energy 0.05, forces 1.16, stress 0.35; NbCl5 0.19, 1.01, 0.65 |
| `tests/gpu/test_smoke.py`, `tests/gpu/test_simulation_gpu.py` | same | 4 passed |

Raw data: `results/opt-def-2/` (the gate's value arrays stay outside git).
