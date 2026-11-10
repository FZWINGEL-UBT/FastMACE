# MACE-OFF and POLAR-1 on the A100: the H200 switches and tonight's POLAR defaults (8 November 2026)

Branch `fastmace/off-polar-a100`. Task (conductor, from the owner): the best A100 configuration for all five MACE-OFF
models and POLAR-1 S/M/L, with the H200 path's new switches applied where they reach these models, and tonight's
POLAR defaults (COMPOSE_HEADS, PRUNE_DEAD_DEGREES, the k-space megakernel) measured on the A100.

- Machine: NHR@FAU Alex, NVIDIA A100-SXM4-80GB (driver 615.71.10), 16 cores, one GPU per job.
- Environment: `fm-offpolar/env` (Python 3.10.12 venv, `requirements-hybrid.txt`), backend hybrid (cuEq + OEq),
  float32, model labels `off23-small`, `off23-medium`, `off23-large`, `off23b-medium`, `off24-medium`, `polar1-s`,
  `polar1-m`, `polar1-l`.
- Protocol: `tools/jobs/alex_ab_pack.sh` (`tools/ab_rewrites.py --pack`), the same-process MD step A/B in both build
  orders (ab = abab, ba = abba). OFF: water_216 (648 atoms), warm-up 500, blocks 8 x 1000 steps. POLAR: NaCl and
  NbCl5 3x3x3, warm-up 200, blocks 6 x 256. b/a < 1: b faster.
- Source of the screens: FastMACE 1bd4f3be (main before the A100 family got the H200 switches), side a = its A100
  defaults, side b = one switch forced on (or off for the POLAR plain defaults). Gates: `tools/force_stress_gate.py`
  (3x rule, rule-13 revisions fail, 2baa8634), seed 20261015; OFF on urea, naphthalene, paracetamol and water_216,
  POLAR on NaCl and NbCl5 3x3x3.
- Baseline: none needed (FastMACE against FastMACE, one process per order).

## Which of the ten H200 switches the OFF and POLAR plans take

Main 0d0eefae turned the ten H200 switches on for the A100 family. On the A100 none of OFF's or POLAR's plans
reaches eight of them (code reading of the planner and the records, confirmed by the `ctrl` pack entry, which forces
all eight on: OFF 0.996-1.007, POLAR-1-M 0.986-1.002, POLAR-1-S 0.995-0.999, ties):

| Switch | Needs | OFF (all five) | POLAR-1 S / M / L |
|---|---|---|---|
| COMPOSE_SKIP, COMPOSE_FAST, COUNT_PLAN | a reassociated generic density first interaction | not reached (stock first block) | not reached (MH-1-class first interaction, density before linear) |
| GENERIC_FOLD_READOUT_TAIL | the generic fold | not reached (first two interactions not both generic) | not reached (not ScaleShiftMACE) |
| SPLINE_W, R128, STAGE_EDGES, SPLINE_VECTOR | the generic w' backward with spline tables | not reached (no wprime_backward) | not reached (the POLAR w' attach builds no spline) |
| STAGE_SH | wprime_backward | not reached | **POLAR-1-M only** (the only POLAR label with wprime_backward on the A100) |
| CUEQ_PROJECTED | tf32x3_gemms and cuEq's SymmetricContraction | **taken** (tf32x3_gemms is on for all five) | **taken** (S, M, L) |

## A/B results (A100-SXM4-80GB, b/a, ab / ba)

| Switch (b) | Model | Structure | b/a ab / ba | Decision |
|---|---|---|---|---|
| CUEQ_PROJECTED on | OFF23-small | water_216 | 0.9894 / 0.9930 | on (gate passes) |
| | OFF23-medium | water_216 | 0.9896 / 0.9997 | on (gate passes) |
| | OFF23-large | water_216 | 1.0006 / 0.9996 | **excluded** (tie) |
| | OFF23b-medium | water_216 | 0.9915 / 0.9916 | on (gate passes) |
| | OFF24-medium | water_216 | 0.9932 / 0.9981 | on (gate passes) |
| | POLAR-1-S | NaCl, NbCl5 | 1.0042 / 1.0075, 0.9983 / 0.9998 | **excluded** (slower on NaCl) |
| | POLAR-1-M | NaCl, NbCl5 | 0.9878 / 1.0058, 0.9985 / 0.9993 | **excluded** (NaCl ba slower) |
| | POLAR-1-L | NaCl, NbCl5 | 0.9942 / 0.9967, 0.9990 / 0.9977 | on (gate: see below) |
| STAGE_SH on | POLAR-1-M | NaCl, NbCl5 | 0.9811 / 0.9914, 0.9968 / 0.9977 | on (bitwise; kept as main has it) |
| MEGAKERNEL on | POLAR-1-S | NaCl, NbCl5 | 1.0169 / 1.0149, 0.9846 / 0.9854 | A100 added to `MEGAKERNEL_GPUS` (NbCl5 only: NaCl is under `MEGAKERNEL_MIN_WORK`) |
| | POLAR-1-M | NaCl, NbCl5 | 0.9996 / 1.0096, 0.9918 / 0.9954 | as above |
| | POLAR-1-L | NaCl, NbCl5 | 0.9910 (invalid) / 0.9962, 0.9855 / 0.9964 | as above |
| COMPOSE_HEADS off | POLAR-1-S | NaCl, NbCl5 | 1.0866 / –, 1.1041 / – | stays on (every GPU) |
| | POLAR-1-M | NaCl, NbCl5 | 1.1274 / 1.1372, 1.1806 / 1.1804 | stays on |
| | POLAR-1-L | NaCl, NbCl5 | 1.1032 / –, 1.1645 / – | stays on |
| PRUNE_DEAD_DEGREES off | POLAR-1-S | NaCl, NbCl5 | 1.0873 / –, 1.1041 / – | stays on (every GPU) |
| | POLAR-1-M | NaCl, NbCl5 | 1.1289 / 0.9949, 1.1810 / 1.0027 | stays on (the ba side apparently built with pruning on; ab counts) |
| | POLAR-1-L | NaCl, NbCl5 | 1.1406 / –, 1.2171 / – | stays on |

– : the job's time limit ended before the ba process (S and L, first round); only CUEQ_PROJECTED and MEGAKERNEL were
rerun in ba. COMPOSE_HEADS and PRUNE_DEAD_DEGREES are plain defaults on every GPU already; their A100 ab numbers
confirm them (8-22 % slower without).

## Gates (A100-SXM4-80GB, 3x rule; ratio = production error / float32 reference error, worst per quantity)

| Gate | Model | Result |
|---|---|---|
| CUEQ_PROJECTED on | OFF23-small | pass (paracetamol forces 2.28, stress 2.38). The 1bd4f3be default **fails**: paracetamol stress 3.13 |
| | OFF23-medium | pass (paracetamol stress 2.48; default 2.66) |
| | OFF23b-medium | pass (paracetamol stress 1.04; default 1.10) |
| | OFF24-medium | pass (paracetamol stress 1.96; default 2.97) |
| | POLAR-1-L | pass (branch tip b4ec7053, where it is the default: NaCl forces 0.88, stress 1.16; NbCl5 0.38, 0.28) |
| MEGAKERNEL on | POLAR-1-S | pass (NaCl stress 1.56, NbCl5 0.72) |
| | POLAR-1-M | pass (NbCl5 forces 1.21, stress 1.74) |

The POLAR-1-L megakernel gate passed on the A100-80GB before (polar-megakernel, NbCl5). No gate here had a rule-13
revision of its plan. The first POLAR-1-L gate (4966344) died on the Alex group file quota (Inductor cache write);
after cleaning `fm-offpolar` (about 125K files) it was rerun as 4967061.

## Before and after (branch tip b4ec7053, A100-SXM4-80GB)

One pack per model group on the branch's final tree: side a = its A100 defaults (CUEQ_PROJECTED by
`RewriteConfig.cueq_projected`, STAGE_SH, the megakernel from `MEGAKERNEL_MIN_WORK`), side b = `before`
(`CUEQ_PROJECTED=False`, `STAGE_SH=False`, `MEGAKERNEL=False`: the 1bd4f3be A100 configuration for these models,
since none of the other H200 switches reaches them). b/a > 1: the new defaults are faster. MD step in ms is side a's
median.

| Model | Structure | After, ms (ab / ba) | b/a = before / after (ab / ba) | What changed |
|---|---|---|---|---|
| OFF23-small | water_216 | 1.491 / 1.489 | 1.0033 / 0.9989 | CUEQ_PROJECTED |
| OFF23-medium | water_216 | 2.604 / 2.606 | 1.0046 / 1.0026 | CUEQ_PROJECTED |
| OFF23-large | water_216 | 6.923 / 6.915 | 1.0043 / 1.0041 | nothing (control: excluded) |
| OFF23b-medium | water_216 | 3.850 / 3.868 | 1.0109 / 0.9980 | CUEQ_PROJECTED |
| OFF24-medium | water_216 | 3.848 / 3.851 | 1.0096 / 1.0037 | CUEQ_PROJECTED |
| POLAR-1-S | NaCl | 2.289 / 2.285 | 0.9919 / 0.9956 | nothing (control) |
| POLAR-1-S | NbCl5 | 7.772 / 7.765 | 1.0114 / 1.0108 | megakernel |
| POLAR-1-M | NaCl | 4.228 / 4.191 | 0.9984 / 1.0180 | STAGE_SH |
| POLAR-1-M | NbCl5 | 16.382 / 16.395 | 1.0108 / 1.0097 | megakernel, STAGE_SH |
| POLAR-1-L | NaCl | 7.707 / 7.700 | 1.0030 / 1.0048 | CUEQ_PROJECTED |
| POLAR-1-L | NbCl5 | 29.055 / 29.049 | 1.0040 / 1.0044 | CUEQ_PROJECTED, megakernel |

The rows where nothing changed (OFF23-large, POLAR-1-S NaCl) show this job's spread: about ±0.5-0.8 %. Against
that, the gains are about 1 % on POLAR NbCl5 (the megakernel) and 0.3-1 % elsewhere. The A100 has little left from
the H200 path for these models: the large H200 wins (spline w', composed skip, the tail fold) need the generic MACE
family, which neither OFF (stock first block) nor POLAR (MH-1-class interactions) is.

## Files

- `runs/h200a100-off-4963396`, `runs/h200a100-polsm-4963397`, `runs/h200a100-poll-4963398`: round 1 packs
  (`cueqproj`, `stagesh`, `mega`, `headsoff`, `pruneoff`, `ctrl`).
- `runs/h200a100-pols-ba-4964663`, `runs/h200a100-poll-ba-4964664`: the ba order for S and L.
- `runs/gate-cueq-*-4964665`, `runs/gate-mega-*-4964666`: gates (default and candidate);
  `runs/gate-cueq-polar1-l-4967061` (and the failed 4966344).
- `runs/final-*-49663*`: before and after.
- `runs/jobs/`, `runs/logs/`: job files and Slurm logs.
