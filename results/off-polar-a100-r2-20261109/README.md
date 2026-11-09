# MACE-OFF and POLAR-1 on the A100: the new OMAT-0 switches (9 November 2026)

Branch `fastmace/off-polar-a100-r2`. Task (owner): OMAT-0 medium and small gained new defaults since the last OFF and
POLAR round ([off-polar-a100](../off-polar-a100-20261108/README.md)). Check which of them reach MACE-OFF and POLAR-1
on the A100, measure them, and update OFF's and POLAR's defaults where they do not pay.

- Machine: NHR@FAU Alex, NVIDIA A100-SXM4-80GB (driver 615.71.10), 16 cores, one GPU per job.
- Environment: `fm-offpolar/env` (Python 3.10.12 venv, `requirements-hybrid.txt`), backend hybrid (cuEq + OEq),
  float32, model labels `off23-small`, `off23-medium`, `off23-large`, `off23b-medium`, `off24-medium`, `polar1-s`,
  `polar1-m`, `polar1-l`.
- Protocol: `tools/jobs/alex_ab_pack.sh` v2 (`tools/ab_rewrites.py --pack`), the same-process MD step A/B in both
  build orders (ab = abab, ba = abba). OFF: water_216 (648 atoms), warm-up 500, blocks 8 x 1000 steps. POLAR: NaCl
  and NbCl5 3x3x3, warm-up 200, blocks 6 x 256. b/a < 1: b faster.
- Screen source: FastMACE main 6f03bd02, side a = its A100 defaults, side b = one switch reverted (or forced on).
  Confirmation source: this branch, c619d306, side a = its defaults, side b = main's defaults forced back.
- Baseline: none needed (FastMACE against FastMACE, one process per order).

## Which new switches reach OFF and POLAR

Code reading of the planner and the kernels at 6f03bd02 (every switch with an A100 entry added since 8 November):

| Switch | Reaches | Why |
|---|---|---|
| `neighbors.skin.SORT_BY_LENGTH` (rounding) | all OFF and POLAR | every plan with `sort_receivers` |
| `wprime_backward.BUTTERFLY`, `STAGE_DOUT` (bitwise) | POLAR-1-M | the only OFF or POLAR plan with `wprime_backward` on the A100 |
| `fused_forward.DEGREE_PARTS` (bitwise) | POLAR-1-M | the only one with `fused_folded_forward` on the A100 |
| `SC_GENERATED`, `JOINED_LAYOUT`, spline `STAGE_EDGES`, `PREFETCH_AHEAD`, `PACKED8`, `BLOCK_ORDER`, w′ `STAGE_EDGES`, `SHAPE_CARRIER`, `RECEIVER_SUM`, the `omat0-small` label entries | none | the generic family's spline w′ path, the generic first block or an OMAT-0 label |
| `BINNED_SCREEN` | none of the benchmark cells | from 10k atoms |

POLAR-1-L's `wprime_backward` + `fused_folded_forward` was retried as well: it lost on NbCl5 on 6 November, and the
w′ kernels have changed since.

## Screen (A100-SXM4-80GB, main 6f03bd02, b/a ab / ba)

| Switch (b) | Model | Structure | b/a ab / ba | Decision |
|---|---|---|---|---|
| SORT_BY_LENGTH off | OFF23-small | water_216 | 0.9830 / 0.9739 | **off** for this model (pure removal, faster) |
| | OFF23-large | water_216 | 0.9951 / 0.9900 | **off** for this model |
| | OFF23-medium | water_216 | 0.9970 / 1.0024 | stays on (tie) |
| | OFF23b-medium | water_216 | 1.0025 / 1.0006 | stays on |
| | OFF24-medium | water_216 | 1.0002 / 1.0028 | stays on |
| | POLAR-1-S | NaCl, NbCl5 | 1.0015 / 1.0010, 0.9997 / 1.0004 | stays on (tie) |
| | POLAR-1-M | NaCl, NbCl5 | 0.9924 / 1.0097, 0.9996 / 1.0006 | stays on (mixed) |
| | POLAR-1-L | NaCl, NbCl5 | 0.9993 / 1.0027, 0.9979 / 1.0003 | stays on (tie) |
| STAGE_DOUT off | POLAR-1-M | NaCl, NbCl5 | 0.9682 / 0.9770, 0.9920 / 0.9911 | **off** for this model (bitwise, faster in all four) |
| DEGREE_PARTS off | POLAR-1-M | NaCl, NbCl5 | 0.9825 / 0.9911, 0.9967 / 0.9962 | **off** for this model (bitwise, faster in all four) |
| BUTTERFLY off | POLAR-1-M | NaCl, NbCl5 | 0.9875 / 0.9959, 1.0020 / 1.0022 | stays on (faster on NaCl, slower on NbCl5 in both orders) |
| wprime + fused forward on | POLAR-1-L | NaCl, NbCl5 | 0.9629 (invalid) / 0.9643, 1.0089 / 1.0123 | stays off (NbCl5 slower, as on 6 November) |

POLAR-1-S rows for BUTTERFLY, STAGE_DOUT and DEGREE_PARTS (0.992-1.000) are controls: S has no w′ or fused forward,
so they show the job's spread. MACE-OFF has no spline tables, so the length sort's slab locality buys nothing there
and only its build costs; on OFF23-small (the cheapest step) that is 1.7-2.6 %.

## Change

The kernels' per-model maps (the convention of `PREFETCH_AHEAD_MODEL_OFF_GPUS`), each on the A100 family:

- `neighbors.skin.SORT_BY_LENGTH_MODEL_OFF_GPUS`: `off23-small`, `off23-large`. The label reaches the list as
  `SkinNeighbourList(model_label=...)` from the force field's table label.
- `wprime_backward.STAGE_DOUT_MODEL_OFF_GPUS`: `polar1-m`. The label reaches `WPrimeBackward` through `WPrime`, which
  the MH-1-class attach (`rewrites.radial.attach_wprime`) now builds with the context's model label.
- `fused_forward.DEGREE_PARTS_MODEL_OFF_GPUS`: `polar1-m`. The label reaches `FusedFoldedForward` from the MH-1
  record (`rewrites.mh1.fused_folded_forwards`) and from WPrime's message and block forwards.

All three are removals of a default on these models only; the result is the previous configuration (gated for
SORT_BY_LENGTH off: the A100 OFF and POLAR gates of 8 November; bitwise for the other two), so no new gate.

## Confirmation (A100-SXM4-80GB, branch c619d306, b = main's defaults forced back)

b/a > 1: the branch's defaults are faster. MD step in ms is side a's (the branch's) median.

| Model | Structure | Branch, ms (ab / ba) | b/a = main / branch (ab / ba) | b forces |
|---|---|---|---|---|
| OFF23-small | water_216 | 1.493 / 1.504 | 1.0251 / 1.0074 | `SORT_BY_LENGTH=True` |
| OFF23-large | water_216 | 6.945 / 6.916 | 0.9972 / 1.0047 | `SORT_BY_LENGTH=True` |
| POLAR-1-M | NaCl | 4.138 / 4.104 | 1.0129 / 1.0306 | `STAGE_DOUT=True DEGREE_PARTS=True` |
| POLAR-1-M | NbCl5 | 16.098 / 16.126 | 1.0145 / 1.0142 | `STAGE_DOUT=True DEGREE_PARTS=True` |

OFF23-small and POLAR-1-M confirm in both orders: 0.7-2.5 % and 1.3-3.1 %. OFF23-large is a tie here (the
screen had 0.5-1 % for the removal in both orders); a pure removal needs only a not-slower A/B (decision 66), so
it stays off for large as well. The confirmation also shows that the model label reaches the neighbour list, the
w′ backward and the fused forward in the real force field (side b differs from side a only through the forced
constants).

## Files

- `runs/r5-off-4970847`, `runs/r5-polsm-4970848`, `runs/r5-poll-4970849`: the screen.
- `runs/r6-off-4972687`, `runs/r6-polm-4972688`: the confirmation.
- `runs/jobs/`, `runs/logs/`: job files and Slurm logs.
