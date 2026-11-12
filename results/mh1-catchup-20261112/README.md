# MH-1 catch-up on the A100 and the H200 (fastmace/mh1-catchup, 12 November 2026)

MH-1's tuning stopped around 27–29 October. Since then, two weeks of H200 and A100 work went into OMAT-0 medium. This
run asks three things:

- what MH-1 (`mh1-r2scan`, and the `mh1-omat` / `mh1-mp` heads, which share its plan) gets today;
- which November OMAT-0 changes can reach MH-1's interaction class;
- what is worth an A/B.

Source: main `236fa60c`. Every GPU result below names its job.

## 1. Which plan MH-1 gets today (offline, `plan_dump.py` → `plan_dump.json`)

The fixed planner with the CPU tests' stand-in traits and the GPU name set. The three MH-1 heads get the same plan.

| Field | H200 | A100 (SXM4-80GB, 80GB PCIe) | Ada | OMAT-0 medium, H200 / A100 |
| --- | --- | --- | --- | --- |
| `tf32x3_gemms` | **off**: "not a measured GPU, decision 66" | on | off | on / on |
| `wprime_backward` | **off**: "not a measured GPU" | on | on | on / on |
| `gate_epilogue` | **off**: "not a measured GPU" | off (not measured; under `tf32x3_gemms` it is rerouted anyway, `fixed.py:1585`) | on | n/a |
| `grouped_launch` | off (5060 Ti only) | off (tf32x3 reroutes) | off | — |
| `cueq_projected` | off (it follows tf32x3) | on | off | on / on |
| `sc_generated` | off | "on", but a no-op for MH-1 (needs a first-interaction `compose_skip`) | off | on / on |
| `edge_index_int32` | **off** (H200 not measured for any model) | on | on | off / on |
| `fused_folded_forward`, `radial_tables` | on | on | on | on |

So on the H200 MH-1 runs the late-October GPU-independent plan, plus the fused forward and the radial tables. It has
no 3×TF32 products and no w′ kernel. Those two were the A100's largest MH-1 wins (A100-SXM4-80GB MD step
`tf32x3_gemms` NaCl 0.887 / 0.882, NbCl5 0.772 / 0.770; `wprime_backward` NaNbO3 0.815 / 0.812, NbCl5 0.889 / 0.889).

### MH-1 per-GPU sets that lack the H200

| Set | Members | H200 | Note |
| --- | --- | --- | --- |
| `DECISION_66_WPRIME_GPUS` | A100 (every model), 5060 Ti, Ada | missing | the H200 has OMAT-0's w′ |
| `DECISION_66_GATE_EPILOGUE_GPUS` | 5060 Ti, Ada | missing (A100 too) | moot wherever `tf32x3_gemms` is on |
| `DECISION_66_TF32X3_GPUS` | A100 | missing | the H200 is in `…_OMAT0_GPUS`, OMAT-0 medium only |
| `DECISION_66_GROUPED_LAUNCH_GPUS` | 5060 Ti | missing (A100 too) | rerouted by `tf32x3_gemms` |
| `gated_products.UNPADDED_GPUS` (exact) | 5060 Ti, Ada | missing (A100 too) | Triton gated kernel; only without `tf32x3_gemms` |
| `tile_rule.COLUMN_ORDER_GPUS` (exact) | 5060 Ti, Ada | missing (A100 measured off) | Triton `segment_gemm`; only without `tf32x3_gemms` |
| `tile_rule.LARGE_TILE_GPUS` (exact) | Ada | missing (A100 measured off) | as above |
| `DECISION_66_EDGE_INDEX_INT32_GPUS` | 5060 Ti, Ada, A100 | missing | any model |

## 2. Which kernel switches reach MH-1 (code map)

MH-1's w′ module comes from `attach_wprime` (`rewrites/radial/__init__.py:775`), built with `generic=False`.
OMAT-0's comes from `attach_generic_wprime` (`:879`), with `generic=True` and `_attach_spline`. `TABLE_W`'s default
is `bool(generic) and GPU in TABLE_W_GPUS` (`wprime_backward.py:1822`), so MH-1 never gets `TABLE_W`, and nothing that
needs it.

| November OMAT-0 change | Reaches MH-1? | Why |
| --- | --- | --- |
| w′ `TABLE_W` (value table, no stored w) | no, by default; **yes as the constant `TABLE_W=True`** | `generic=False`; nothing is missing in the data: `value_table(table.coef, weight)` needs only MH-1's radial table and folded weight. Gate passed on the 5060 Ti for MH-1 (4 Nov, ledger `omat0-table-w`) |
| `HORNER`, `FOLD_ALPHA` | only with `TABLE_W` | they follow `TABLE_W` |
| `SKIP_OUTSIDE` + `HOIST_DX` (A100, H200) | only with `TABLE_W` | `skip_outside = table_w and …` (`wprime_backward.py:1370`) |
| `LATE_PULLBACK` (H200) | no | needs `TABLE_W`, `HORNER` and `wprime_pullback.supported`, which declines MH-1's schedule without the block layout (render below: off even with `TABLE_W`) |
| `WEIGHTED_VECTOR` (A100) | **yes, today** (A100), unmeasured on MH-1 | FACTORED + A100 set; 5 of 10 paths |
| `STAGE_DOUT`, `STAGE_SH`, `FACTORED`, `SCALAR_PATHS`, `BUTTERFLY` | yes on the A100; on the H200 only once w′ is on | device sets |
| w′ `STAGE_EDGES`, `SPLINE_VECTOR`, spline w | no | spline only through `_attach_spline` (generic); a port needs MH-1 wiring of `_attach_spline`, `BLOCK_MESSAGE` and `block_spec` |
| spline forward `PACKED8`, `SPLINE_W`, `R128`, `GEOM`, `BLOCK_ORDER`, `PREFETCH_AHEAD` | no | spline forward is generic-only (as above) |
| `COMPOSE_SKIP`, `COMPOSE_FAST`, `COUNT_PLAN`, `JOINED_LAYOUT` | no | `precision.py:102-113`: tf32x3 route **and** the generic density first interaction; MH-1 has none |
| CUTLASS `tf32x3.HYBRID` | **yes, today** on the A100 (unmeasured on MH-1); on the H200 once `tf32x3_gemms` is on | every CUTLASS 3×TF32 GEMM |
| `LINEAR_DIV`, `SEGMENT_ROWS` | yes on the A100 (`SEGMENT_ROWS` A100 only) | BlockLinear on CUTLASS |
| `SKIP_SELECT`, `SC_GENERATED`/`SC_FACTORED`, `RECEIVER_SUM`, `GENERIC_FOLD_READOUT_TAIL`, `EDGE_SH` | no | generic-only modules (MH-1 has no `SpeciesSkip`, no `_density_sum`, `EDGE_SH_MH1_GPUS` empty) |
| `BACKWARD_WALK` | yes on the A100 (MH-1's reassociated first interaction uses `edge_outer_sum`) | A100 set |
| `SORT_BY_LENGTH`, `BINNED_SCREEN`, `HEAD_SLICE`, `GRAPH_SUM` | yes (device sets) | — |
| fused forward `TENSOR_CORES`, `VEC_SH`, `TC_WFRAG`, `WFRAG_REGION`, `DEGREE_PARTS`, `W_GROUP`, `X_LATE`, `A_LD128`, `LANE_INDICES` (H200) | yes where the shapes fit | device sets; `TENSOR_CORES` was measured on MH-1 (A100 0.97–0.98) |
| fused forward `CG_FOLD`, `TP_NOFLUSH`, `WS_FORWARD` | no | `not store_w`: MH-1's forward stores w; with `TABLE_W` the message-only forward could take them |

## 3. Offline SASS census of MH-1's w′ (`render_mh1.py`, `census_sass.py`)

MH-1's second interaction: `128x0e+128x1o` ⊗ `1x0e+1x1o+1x2e+1x3o`, ten uvu paths. This is the same problem as
POLAR-1-M. Sources were rendered by the production `WPrimeBackward.__init__` with the GPU name patched
(`sources/`). They were compiled with NVRTC for sm_80 / sm_90 (`sass/`, `census_sm80.json`, `census_sm90.json`).
Instructions are per edge and lane.

| Variant | Flags | sm_80 instr. / regs | sm_90 instr. / regs |
| --- | --- | --- | --- |
| slope (stored w; A100 today) | FACTORED, SCALAR_PATHS, BUTTERFLY, STAGE_DOUT, STAGE_SH 4, WEIGHTED_VECTOR (A100) | 2,168 / 96 | 2,283 / 96 |
| `TABLE_W=True` | + HORNER, FOLD_ALPHA, SKIP_OUTSIDE, HOIST_DX | 2,120 / **127** | 2,192 / 96 |

`TABLE_W` changes the kernel in three ways:

- **Loads:** it drops the per-edge w loads. The channel pass's memory instructions go from 112 to 82 (sm_80) and from
  109 to 78 (sm_90).
- **Forward:** the forward no longer stores w, which is E × 2,560 floats per interaction.
- **Instructions:** −2.2 % on sm_80, −4.0 % on sm_90.

On sm_80 it costs 31 registers (96 → 127), so occupancy falls. Only an A/B can say whether the traffic saved outweighs
that on the A100. The A100 pack carries `TABLE_W` both with and without `SKIP_OUTSIDE`.

## 4. Jobs

- Alex 4990788 (`jobs/alex_mh1_round1.sh`) is one job, per the owner via the conductor: one long A100 job per round.
  It runs these in sequence:
  - `jobs/alex_mh1_base.sh`: A100 baseline MD step times (NaCl, NaNbO3, NbCl5 3×3×3, NbCl5 6×6×6), the NVRTC
    sources, and the torch 2.15 CUDA-graph attribution (NaCl, NaNbO3, NbCl5);
  - `jobs/alex_mh1_ab.sh` pack 1: `tablew`, `tablew_noskip`, `noweighted`, `nohybrid`, then gate `tablew`;
  - pack 2: `nowalk`, `nosegrows`, `noint32`, `nosort`, then gate `default` (today's A100 MH-1 plan).

  It replaces 4990696, 4990707 and 4990708, which were cancelled while still pending.
- leto, RTX 4000 Ada (`jobs/ada_tablew.sh`): the workstation screen before the A100 job. It runs MH-1 with
  `TABLE_W=True` against the Ada defaults (NaCl, NbCl5, both orders), then the 3× gate.
- H200 (`h200/launch_h200.sh`; prepared, not launched; Modal at $29.49 of the $30 cap):
  - two A/B packs, NbCl5 and NaCl 3×3×3, each with `tf32`, `wp`, `tf32wp`, `tf32wptw`, `gepi`, `wpgepi`, `int32`;
  - one gate pack: default, `tf32wp`, `tf32wptw`, `wpgepi`;
  - cost: about $5 expected, about $11 worst case reserved.
  - The gate needs `tools/jobs/gate_pack.sh` v2, which passes `--rewrites` bodies and takes comma-separated STRUCTURES.

Results: pending (sections 5 onward).
