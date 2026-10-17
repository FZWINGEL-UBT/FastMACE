# Phase 1a: generic rewrites on released models (plan task 1a-T08)

The stacked generic rewrites (every record of `fastmace.rewrites.generic` that `auto` enables
for the model) were applied to MACE-MP-0 medium and MACE-OFF23 small and compared with the same
model unrewritten, on one small structure each. Both models agree to rounding in float64, far
below the SPEC §6.4 per-model level (1e-10 relative), and to 1e-6 in float32 (level 1e-4).

## Setup

- murphy, NVIDIA RTX 4000 Ada Generation (driver 595.58.03), 17 October 2026.
- Plain cuEq: NAS `mace_cueq_latest` (torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance-torch
  0.12.0, e3nn 0.4.4), convolution fused (`cueq_conv_fusion`). Hybrid: NAS `mace_cueq_oeq`
  (same plus OpenEquivariance 0.7.0), in a fresh interpreter.
- FastMACE: main 19d732b plus branch `fastmace/1a-T08`. The reassociation and streaming
  modules belong to plan task 1a-T07 (`fastmace.rewrites.mh1`), not yet merged; this run used
  an uncommitted stand-in copied from the baseline (`_FirstInteractionReassociation`,
  `_probe_first_tensor_product` with a tolerance in its coefficient check, `_streaming_message`,
  `_attach_streaming_kernels`). Repeat with T07's module after both merge.
- Baseline for the CPU bitwise tests: ResearchModule tag `research-engine-baseline-20261017`
  (d2faa9fd).
- Test: `tests/gpu/test_rewrites_generic_gpu.py`. Each model is loaded twice and the second
  copy rewritten through the registry with its self-checks; edges sorted by receiver.
- Structures: NaCl 1×1×1 cubic, rattled 0.05 Å (8 atoms, 256 edges) for MP-0; an ethanol
  molecule rattled 0.03 Å (9 atoms, 72 edges, no stress) for MACE-OFF.
- Raw results: `results/1a-T08-generic-murphy-20261017/agreement.jsonl`.

## Result

Maximum relative error, rewritten against unrewritten, max |Δ| / max |reference|:

| Model | Backend | Precision | Records applied (interactions) | Energy | Forces | Stress |
| --- | --- | --- | --- | --- | --- | --- |
| mp0-medium | cueq | float64 | species skip (0, 1), reassociate (0), fused backward (1) | 1.3e-16 | 1.6e-15 | 3.2e-15 |
| mp0-medium | hybrid | float64 | the same plus streaming (1) | 0 | 2.0e-15 | 7.3e-15 |
| mp0-medium | hybrid | float32 | the same plus streaming (1) | 0 | 7.0e-7 | 1.3e-6 |
| off23-small | cueq | float64 | species skip (1), fused backward (1) | 0 | 4.1e-16 | — |
| off23-small | hybrid | float64 | species skip (1), fused backward (1), streaming (1) | 0 | 3.8e-16 | — |
| off23-small | hybrid | float32 | species skip (1), fused backward (1), streaming (1) | 0 | 2.2e-7 | — |

Every per-module self-check passed (worst 1.4e-15 in float64, 6.5e-7 in float32).

## MACE-OFF: the first block is never rewritten

MACE-OFF's first interaction is a `RealAgnosticInteractionBlock`, which is not a generic class.
The research engine's `generic_rewrites` reassociates only an index-0 block *inside* its loop over
generic blocks, so for MACE-OFF it never reassociates, although `rewrite_support` reports
`reassociate_first` as supported ("ok (GENERIC_REASSOCIATE)", kept in
`fastmace.model.capabilities` for plan equality in 1a-T15). Counted with the research engine and
with FastMACE on the `off` tiny model and on `MACE-OFF23_small.model`: one interaction
rewritten (`interactions.1`), none reassociated. FastMACE's `generic_reassociate` says so in its
verdict: "the first interaction RealAgnosticInteractionBlock is not a generic class". The other
three generic rewrites act on MACE-OFF's later blocks, as the table shows.
