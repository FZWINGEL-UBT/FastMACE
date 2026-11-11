# ada-5060-screen: OMAT-0 switches on the RTX 4000 Ada and the RTX 5060 Ti (10–11 November 2026)

Decision 66 screen of every OMAT-0 medium switch that was on for the H200 or the A100 but had never been measured on
the RTX 4000 Ada (murphy, leto) or the local RTX 5060 Ti. Main's later additions were screened too: GEOM, SC_FACTORED
and VIRIAL_ELEMENTWISE. The adopted sets were then checked on POLAR-1 and MACE-OFF.

Branch `fastmace/ada-5060-screen`. Code changes: `ec25c9c4`, `63153034`, `d1048f04`, `edfc0115`, `9a669585`.

**Method**

- **Offline census.** Census of the kernels' instructions and registers for sm_89 and sm_120 (`census/`).
- **Packed A/Bs.** Same-process MD-step A/Bs with `tools/ab_rewrites.py --pack`, in both build orders (ab with
  abab, ba with abba).
- **Structures.** Benchmark_loup: SrTiO3 10×10×10 (5,000 atoms) and NbCl5 3×3×3 (1,944 atoms).
- **Settings.** OMAT-0 medium, float32, 1 fs, skin 0.5 Å, warm-up 200, 6 blocks of 256 steps.
- **Leave-one-outs.** Each set was cut by leave-one-outs from the combined set.
- **Gate.** The 3× float32 force/stress gate (`tools/force_stress_gate.py`) was run on the final set.
- **Software.** Hybrid backend, environment `mace_cueq_oeq` (torch 2.14.1+cu130).
- **Trees.** The workstations used the trees `src-screen` (`edfc0115`: main `4e9388c8` with the registry fix and
  this branch's earlier changes). The local runs used a WSL copy of the same tree.

**How to read b/a**

- b/a is side b over side a.
- Two values are the two build orders (ab / ba).
- Below 1 is faster.
- In a leave-one-out, a is the set and b is the set with one switch off, so a value above 1 means the switch helps.

Raw lines: [ada-lines.txt](ada-lines.txt) (Ada packs A1–A7, X1–X10), [local-ba-lines.txt](local-ba-lines.txt) and
[local-polar-ba-lines.txt](local-polar-ba-lines.txt) (5060 Ti stage 1). Job scripts: [jobs/](jobs/).

## RTX 4000 Ada

Stage 1: each switch alone against today's Ada defaults (b = on).

| Switch | SrTiO3 | NbCl5 | Reading |
| --- | --- | --- | --- |
| spline `PACKED8` | 0.9036 / 0.9030 | 0.9346 / 0.9357 | on |
| `SORT_BY_LENGTH` | 0.9794 / 0.9815 | 0.9890 / 0.9908 | on |
| `SC_GENERATED` | 0.9785 / 0.9775 | 0.9738 / 0.9749 | on |
| `SC_FACTORED` (on top of SC_GENERATED) | 0.9978 / 0.9978 | 0.9907 / 0.9906 | on |
| w′ `BUTTERFLY` | 0.9840 / 0.9849 | 0.9913 / 0.9931 | on (flip, below) |
| w′ `STAGE_DOUT` | 0.9699 / 0.9694 | 0.9848 / 0.9806 | on |
| w′ `STAGE_EDGES` + `STAGE_DOUT` + `STAGE_SH` | 0.9639 / 0.9643 | 0.9779 / 0.9758 | on |
| tabled `POWER_BACKWARD` | 0.9935 / 0.9929 | 0.9935 / 0.9933 | on |
| `RECEIVER_SUM` | 0.9931 / 0.9936 | 0.9998 / 0.9993 | off: mixed in the set (below) |
| `GRAPH_SUM` | 0.9965 / 0.9970 | 1.0010 / 1.0000 | off (tie) |
| fused forward `DEGREE_PARTS` | 0.9990 / 0.9990 | 1.0007 / 0.9994 | off (tie) |
| spline `BLOCK_ORDER` | 0.9961 / 0.9947 | 1.0012 / 1.0000 | off (tie; from the set 1.0058 / 0.9991, 0.9980 / 0.9986) |
| w′ `SCALAR_PATHS` | 0.9984 / 0.9984 | 1.0001 / 0.9987 | off (tie) |
| radial table 8 warps | 0.9975 / 0.9990 | 1.0019 / 0.9997 | off (tie) |
| spline `PREFETCH_AHEAD` 2 | 1.0037 / 1.0051 | 1.0063 / 1.0073 | off (slower) |
| w′ `FOLD_ALPHA` | 1.0077 / 1.0082 | 1.0079 / 1.0065 | off (slower) |
| w′ `MAX_REGISTERS` 80 | 1.0529 / 1.0537 | 1.0416 / 1.0401 | off (slower) |

Leave-one-out from the combined set (b = without the switch; above 1 means the switch helps):

| Removed | SrTiO3 | NbCl5 |
| --- | --- | --- |
| `PACKED8` | 1.1061 / 1.1028 | 1.0700 / 1.0722 |
| `SC_GENERATED` | 1.0149 / 1.0171 | 1.0355 / 1.0434 |
| `SORT_BY_LENGTH` | 1.0141 / 1.0150 | 1.0045 / 1.0062 |
| `BUTTERFLY` | 1.0057 / 1.0061 | 1.0058 / 1.0034 |
| w′ `STAGE_DOUT` + `STAGE_SH` + `STAGE_EDGES` | 1.0228 / 1.0200 | 1.0261 / 1.0232 |
| `STAGE_SH` + w′ `STAGE_EDGES` | 1.0035 / 1.0028 | 1.0078 / 1.0056 |
| w′ `STAGE_EDGES` | 1.0032 / 0.9981 | 1.0043 / 1.0064 |
| `POWER_BACKWARD` | 1.0125 / 1.0052 | 1.0087 / 1.0074 |
| `RECEIVER_SUM` | 0.9967 / 0.9964 | 1.0087 / 1.0068 |
| `GRAPH_SUM` | 0.9996 / 0.9965 | 0.9992 / 0.9988 |
| w′ `SCALAR_PATHS` | 0.9992 / 0.9979 | 1.0001 / 0.9997 |

**`RECEIVER_SUM` is mixed.** It depends on the structure: it helps NbCl5 in both orders and slows SrTiO3 in both. It
stays off. This is the conductor's reading of 11 November.

The set was run first with BLOCK_ORDER, which exposed the registry use-after-free fixed in `fastmace/spline-registry-fix`
(merged at `1299b32e`). The rows above are from runs after the fix.

**Final set against today's Ada defaults (X10)**

| Structure | b/a (ab / ba) | Step time |
| --- | --- | --- |
| SrTiO3 | 0.8255 / 0.8194 | 21.4 → 17.7 ms |
| NbCl5 | 0.8546 / 0.8558 | |

The set is `PACKED8`, `SORT_BY_LENGTH`, `SC_GENERATED`, `SC_FACTORED`, `BUTTERFLY`, w′ `STAGE_DOUT`, `STAGE_SH`, w′
`STAGE_EDGES` and `POWER_BACKWARD`.

Gate (3× float32, 3×3×3 supercells). Every ratio passes:

| Structure | Energy | Forces | Stress |
| --- | --- | --- | --- |
| NaCl | 0.399 | 0.875 | 0.896 |
| NbCl5 | 1.561 | 0.951 | 1.165 |
| SrTiO3 | 3.855 (passes: the limit is the absolute floor of 0.01 eV) | 0.787 | 0.517 |

### Rescreen flips

Two switches that measured off on the Ada before now measure on against today's defaults.

- **w′ `BUTTERFLY`.** On 5 November it read 1.008 / 1.007 and 1.006 / 1.000 against that day's w′ defaults. Today
  it is faster alone and in the set.
- **`STAGE_SH`.** On 8 November it read "off again from the Ada set 0.9903 / 0.9905". Today, removing it with w′
  `STAGE_EDGES` from the set is slower: 1.0035 / 1.0028 and 1.0078 / 1.0056.

Both flips follow changed w′ defaults: STAGE_DOUT and w′ STAGE_EDGES are now on. The census shows the cause for
BUTTERFLY: w′ drops from 1306 to 928 instructions and from 94 to 80 registers on sm_89.

### GEOM, SC_FACTORED, VIRIAL, SEGMENT_ROWS

| Switch | SrTiO3 | NbCl5 | Reading |
| --- | --- | --- | --- |
| spline `GEOM` | 1.0226 / 1.0209 | 1.0156 / 1.0159 | off |
| `SC_FACTORED` (on SC_GENERATED) | 0.9978 / 0.9978 | 0.9907 / 0.9906 | on |
| `VIRIAL_ELEMENTWISE` | 1.0041 / 0.9805 (rerun 1.0045 / 0.9810) | – | off (order-dependent) |
| CUTLASS tf32x3 on sm_89 | 1.0025 / 0.9998 | 0.9998 / 1.0009 | tie |
| CUTLASS tf32x3 + `SEGMENT_ROWS` | 0.9967 / 0.9981 | 0.9984 / 1.0016 | off (tie) |

- **GEOM with PACKED8.** GEOM is also slower on top of PACKED8: PACKED8 with GEOM reads 0.9120 / 0.9139 and
  0.9429 / 0.9396, against PACKED8 alone 0.9013 / 0.9058 and 0.9332 / 0.9358.
- **VIRIAL_ELEMENTWISE.** It was measured with `tools/virial_ab.py` on the force call with stress. Its gate passes:
  NaCl 0.916 / 0.953 / 0.961, NbCl5 1.062 / 0.948 / 1.215.
- **SEGMENT_ROWS.** It needs the CUTLASS backend built for sm_89. That build is a tie, so it was not adopted. On the
  5060 Ti it cannot run, because the CUTLASS cubins are built for sm_80 and sm_90 only.

### Other models

[models-firing.md](models-firing.md) lists, from the code, which of the nine switches run for POLAR-1 and MACE-OFF:

- **POLAR-1-M and POLAR-1-L:** SORT_BY_LENGTH, BUTTERFLY, STAGE_DOUT and STAGE_SH.
- **OFF23-medium:** SORT_BY_LENGTH only.
- **The others** never run for these models.

The A/Bs record only `RewriteConfig` fields in their plan text, so the plan text cannot show this.

The whole Ada set against today's Ada defaults:

| Model | Structure | b/a | Gate (energy / forces / stress) |
| --- | --- | --- | --- |
| POLAR-1-M | NaCl | 1.0061 / 1.0077 (slower) | 1.357 / 0.928 / 0.515 |
| POLAR-1-M | NbCl5 | 0.9994 / 0.9985 | 0.753 / 0.687 / 0.784 |
| POLAR-1-L | NbCl5 | 1.0033 / 1.0021 (slightly slower) | – |
| OFF23-medium | water_216 | 1.0002 / 0.9992 | 0.731 / 0.966 / 0.929 |

**POLAR-1-M leave-one-out on NaCl.** Here a is the set, and b turns one switch off.

- **STAGE_DOUT off:** 0.9868 / 0.9858. It is the culprit, as on the A100.
- **BUTTERFLY off:** 1.0044 / 1.0050. It helps.
- **STAGE_SH off:** 0.9960 / 0.9979. Within noise.
- **The five switches that do not run** read 0.994–1.003. That is the drift of this run.

The Ada joins `STAGE_DOUT_MODEL_OFF_GPUS["polar1-m"]`.

**POLAR-1-L leave-one-out on NbCl5.** Every switch is within 0.15 %. STAGE_DOUT off reads 0.9986 / 0.9987.

**Confirmation on leto** (11 November, 06:12–06:53Z): the set without STAGE_DOUT, against today's Ada defaults.

| Model | Structure | b/a |
| --- | --- | --- |
| POLAR-1-M | NaCl | 0.9986 / 1.0006 |
| POLAR-1-M | NbCl5 | 0.9999 / 0.9982 |
| POLAR-1-L | NbCl5 | 1.0005 / 1.0005 |

Both are ties, so the slowdown is gone. The Ada also joins `STAGE_DOUT_MODEL_OFF_GPUS["polar1-l"]`.

## RTX 5060 Ti

**Stage 1 (L1–L9).** Each pack was measured against today's 5060 Ti defaults; see local-ba-lines.txt.

| Pack | SrTiO3 | NbCl5 |
| --- | --- | --- |
| SPLINE_W + R128 + spline STAGE_EDGES | 0.8162 / 0.8183 | 0.8548 / 0.8541 |
| PACKED8 on top | 0.9222 / 0.9246 | 0.9456 / 0.9535 |
| `generic_fold_product_linear` | 0.9377 / 0.9348 | 0.9222 / 0.9166 |
| tf32x3 | 1.0715 / 1.0577 | 1.1449 / 1.1451 |
| tf32x3 + compose | – | 1.0412 / 1.0396 |

tf32x3 is slower, so `COMPOSE_SKIP`, `COMPOSE_FAST`, `COUNT_PLAN`, `JOINED_LAYOUT` and the SC pair stay out.
SC_GENERATED is routed only under tf32x3.

**Stage 2: candidate set D against the defaults.**

| Set | SrTiO3 | NbCl5 |
| --- | --- | --- |
| D with the fold | 0.6407 / 0.6399 | 0.6866 / 0.6873 |
| D without the fold | 0.7009 / 0.7006 | 0.7784 / 0.7800 |

**Leave-one-out from D** (b = without the switch; above 1 means it helps):

| Removed | SrTiO3 | NbCl5 | Reading |
| --- | --- | --- | --- |
| `PACKED8` | 1.0851 / 1.0785 | 1.0509 / 1.0621 | keep |
| fold | 1.0950 / 1.0917 | 1.1331 / 1.1376 | keep |
| w′ `STAGE_EDGES` + `STAGE_DOUT` + `STAGE_SH` | 1.0529 / 1.0503 | 1.0334 / 1.0341 | keep |
| `POWER_BACKWARD` | 1.0126 / 1.0070 | 1.0068 / 1.0091 | keep |
| `GENERIC_FOLD_READOUT_TAIL` | 1.0037 / 1.0048 | 1.0071 / 1.0097 | keep |
| `RECEIVER_SUM` | 1.0041 / 1.0006 | 1.0034 / 1.0049 | keep |
| `SORT_BY_LENGTH` | 1.0052 / 1.0029 | 1.0058 / 0.9966 | keep (adopted set: 1.0095 / 1.0100, 0.9988 / 1.0036) |
| `SC_GENERATED` | 0.9969 / 1.0027 | 0.9971 / 0.9986 | drop |
| `CUEQ_PROJECTED` | 1.0000 / 0.9997 | 0.9920 / 0.9993 | drop |
| `BLOCK_ORDER` | 0.9988 / 0.9969 | 0.9944 / 1.0013 | drop |
| `GRAPH_SUM` | 0.9971 / 0.9947 | 1.0001 / 0.9982 | drop |
| `DEGREE_PARTS` | 0.9962 / 1.0012 | 0.9931 / 0.9990 | drop |
| `SPLINE_VECTOR` 4 (b = 2) | 0.9963 / 0.9941 | 0.9957 / 0.9997 | drop |
| `SYMMETRIC_PAIRS` | 0.9984 / 0.9965 | 1.0024 / 1.0033 | drop (mixed) |
| + `SC_FACTORED` | 0.9966 / 0.9976 | 0.9990 / 0.9943 | see stage 3 |

**Stage 3 (X11, X12, gate).** The adopted set against today's 5060 Ti defaults:

| Structure | b/a (ab / ba) | Step time |
| --- | --- | --- |
| SrTiO3 | 0.6427 / 0.6478 | 23.9 → 15.4 ms |
| NbCl5 | 0.6911 / 0.6906 | 4.95 → 3.42 ms |

From the adopted set:

| Change | SrTiO3 | NbCl5 | Reading |
| --- | --- | --- | --- |
| SC_GENERATED + SC_FACTORED removed | 0.9995 / 0.9998 | 0.9994 / 1.0026 | tie: not adopted (they do not run without tf32x3) |
| `SYMMETRIC_PAIRS` added | 0.9926 / 0.9960 | 0.9986 / 1.0002 | off |

**SYMMETRIC_PAIRS disagrees between runs.** Its SrTiO3 reading is the opposite of stage 2's, so the two runs
disagree and it stays off.

The adopted set: `SPLINE_W`, `R128`, spline `STAGE_EDGES`, `PACKED8`, `SORT_BY_LENGTH`, `GENERIC_FOLD_READOUT_TAIL`,
`POWER_BACKWARD`, `RECEIVER_SUM`, w′ `STAGE_EDGES` / `STAGE_DOUT` / `STAGE_SH` and `generic_fold_product_linear`.

Gate (3× float32), 2×2×2 supercells. Every ratio passes:

| Structure | Energy | Forces | Stress |
| --- | --- | --- | --- |
| NaCl | 1.725 | 1.355 | 1.572 |
| NbCl5 | 0.545 | 0.855 | 0.527 |
| SrTiO3 | 1.142 | 0.398 | 0.535 |

**Why 2×2×2.** The POLAR-1-M gate at 3×3×3 failed in its float64 NbCl5 reference with "CUDA driver error: device
not ready / Failed to create GPU mapping" (WSL). The 16 GiB card is not to be oversubscribed.

**POLAR-1 on the 5060 Ti**

Stage 1, against today's defaults:

| Candidate | NaCl | NbCl5 |
| --- | --- | --- |
| POLAR-1-M w′ pair | 0.9597 / 0.9589 | 0.9380 / 0.9415 |
| POLAR-1-L w′ pair | 0.9373 / 0.9348 | 1.0098 / 1.0097 (slower: not adopted) |
| POLAR-1-S gate_epilogue | 1.0043 / 0.9992 | 0.9897 / 0.9909 |
| POLAR-1-M gate_epilogue | 1.0032 / 0.9969 | 0.9839 / 0.9944 |
| POLAR-1-L gate_epilogue | 1.0000 / 0.9991 | 0.9863 / 0.9874 |

The candidate defaults with the 5060 Ti set on. With the pair, POLAR-1-M's w′ also takes the staging switches.

| Model | NaCl | NbCl5 | Gate (2×2×2), energy / forces / stress |
| --- | --- | --- | --- |
| POLAR-1-M: pair + epilogue | 0.9659 / 0.9758 | 0.9273 / 0.9389 | NaCl 1.145 / 0.923 / 0.318, NbCl5 0.705 / 1.029 / 0.928 |
| POLAR-1-S: epilogue | 1.0080 / 0.9971 | 0.9927 / 0.9846 | NaCl 0.722 / 0.729 / 0.391, NbCl5 0.605 / 0.682 / 0.794 |
| POLAR-1-L: epilogue | 0.9988 / 1.0027 | 0.9869 / 0.9885 | NaCl 0.417 / 1.043 / 1.078, NbCl5 0.680 / 1.279 / 0.585 |

Adopted: `POLAR_DECISION_66_GATE_EPILOGUE_GPUS` gains the 5060 Ti, and `POLAR_DECISION_66_WPRIME_LABEL_GPUS["polar1-m"]`
gains it too. S and L pass by the mixed-cell rule.

**The 5060 Ti set on the other models.** Without the pair, only SORT_BY_LENGTH runs.

| Model | Structure | b/a | Gate |
| --- | --- | --- | --- |
| POLAR-1-M | NaCl | 1.0039 / 0.9983 | – |
| POLAR-1-M | NbCl5 | 1.0006 / 0.9998 | – |
| OFF23-medium | water_216 | 0.9998 / 0.9907 | 1.020 / 0.983 / 1.093 |

## Earlier in this branch

- **POLAR k-space `MEGAKERNEL` on the Ada** (`d1048f04`). NbCl5: S 0.8964 / 0.8964, M 0.9490 / 0.9517, L
  0.9678 / 0.9684. The gate passes for S, M and L.
- **`SPLIT_COLUMNS` on the Ada** (`edfc0115`). OFF23-large 0.9556 / 0.9583.

## Registry fix

The use-after-free that BLOCK_ORDER exposed in five-engine A/B processes was fixed separately and merged at
`be193ea0` / `1299b32e`:

- **Before the fix.** SplineForward's registry replaced the entry on an equal build. An engine that had captured
  earlier then launched a freed object.
- **Fix and tests.** `tests/cpu/test_spline_forward_registry.py` and `tests/gpu/test_spline_registry_engines_cuda.py`.
  The GPU test passes on murphy.
- **Memcheck.** compute-sanitizer 2026.2 aborts this stack in the host process with "double free or corruption"
  before any kernel is checked, so it is impractical here.

## Lessons

- **Local GPU jobs.** Long local jobs must run as a `systemd-run --user` unit with `/usr/lib/wsl/lib` on PATH and
  LD_LIBRARY_PATH, plus a `wsl` keep-alive. Detached `nohup` launches died when the `wsl` call returned.
- **Workstation jobs.** Run one sequential chain per host. Separately queued jobs of the same session raced in
  `queue.sh`.
