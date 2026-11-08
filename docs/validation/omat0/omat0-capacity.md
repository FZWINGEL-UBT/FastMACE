# MACE-OMAT-0 medium: the generic folded path and its memory rescue (2 November 2026)

The owner's task, through the conductor:
- carry the whole-system capacity work over to MACE-OMAT-0 medium (`omat0-medium`);
- measure the default plan's limit and a memory ledger;
- find the generic analogue of each MH-1 memory record;
- build the best one or two, with rule 12's terms and a calibrated rescue ladder.

The conductor's conditions:
- (A), the folded and chunked streaming convolution, is default-on only where it passes the accuracy gate and its
  same-process A/Bs, in both build orders, are no slower (decision 66, exact device names);
- (B), the weight recompute, stays a rescue rung: off by default, float32 only, refusing float64 cleanly.

## Verdict

**The default plan's limit rose 3.2–4.1 times on both GPUs**, MACE-OMAT-0 medium float32 NVT. The method is the
same both times: `capacity_scan.py scan`, 20 warm-up and 200 timed steps, the largest passing cubic repeat.

| GPU (provider) | Crystal | Before: atoms, ms/step | After: atoms, ms/step | Ratio | Plan at the new limit |
|---|---|---|---|---:|---|
| RTX 4000 Ada 20 GB (leeloo, workstation) | SrTiO3 | 16,875, 234 | **69,120**, 974 | 4.1 | (A) + (B) |
| RTX 4000 Ada 20 GB (leeloo, workstation) | NaCl | 39,304, 354 | **125,000**, 965 | 3.2 | (A) + (B) |
| RTX 5060 Ti 16 GB (local PC, WSL2) | SrTiO3 | 13,720, 227 | **53,240**, 840 (practical 46,305, 656) | 3.9 (3.4) | (A) + (B) |
| RTX 5060 Ti 16 GB (local PC, WSL2) | NaCl | 32,768 raw, 561 / 27,000 practical, 196 | **85,184**, 623 | 2.6 raw, 3.2 practical | (A) + (B) |

- "Practical" on the 5060 Ti is the largest size without the WSL2 slowdown.
  - Before, NaCl at 32,768 ran 2.9 times slower per atom than at 27,000.
  - After, NaCl's time per atom is flat up to 85,184 (7.0–7.6 µs per atom), while SrTiO3's rises 11 % from 46,305 to
    53,240.
- At the old limits, (A) alone makes the step faster:
  - NaCl at 46,656 atoms runs at 335 ms on the Ada, against 354 ms at the old limit's 39,304;
  - SrTiO3 at 20,480 atoms runs at 256 ms.
- Rule 12 planned every point by itself: (A) by default, and (B) above about 29k SrTiO3 and 64k NaCl atoms on the
  5060 Ti (34k and 74k on the Ada). No point fell back or rebuilt.
- The first failing sizes:
  - out of memory at build on the Ada (SrTiO3 78,125, NaCl 140,608);
  - on the 5060 Ti, out of memory in the capture at NaCl 97,336 (WSL2's "device not ready / Failed to create GPU
    mapping") and in the build at SrTiO3 60,835.
- Rule 12's estimate against these points' reserved peaks:
  - 0.96–1.04 on the Ada;
  - 0.96–1.08 on the 5060 Ti.
  - The lowest are the first (B) points, SrTiO3 34,295 (Ada, 0.964) and 29,160 (5060 Ti, 0.958). Both are inside
    decision 51's 10 % margin.

## Baseline and ledger

**Baseline, the default plan:** main 1ef6c0ec; `capacity_scan.py scan`, 20 warm-up and 200 timed steps; the limit
is the largest passing cubic repeat.

| GPU (provider) | Crystal | Limit (atoms) | ms/step | Peak reserved | First failing size |
|---|---|---:|---:|---:|---:|
| RTX 4000 Ada 20 GB (leeloo, workstation) | SrTiO3, a = 3.905 Å | 16,875 (15³) | 234 | 16.62 GiB | 20,480 |
| RTX 4000 Ada 20 GB (leeloo) | NaCl | 39,304 (17³) | 354 | 17.60 GiB | 46,656 |
| RTX 5060 Ti 16 GB (local PC, WSL2) | SrTiO3 | 13,720 (14³) | 227 | 13.93 GiB | 16,875 |
| RTX 5060 Ti 16 GB (local PC, WSL2) | NaCl | 32,768 (16³) raw; about 27,000 practical | 561 at 32,768, 196 at 27,000 | 14.80 GiB | 39,304 |

- The 5060 Ti's NaCl limit at 32,768 runs 2.9 times slower per atom than at 27,000, which looks like WSL2 spilling
  device memory to the host. So about 27,000 atoms is the practical limit there.
- Rule 12 already fitted OMAT-0's default plan: 0.98–1.06 times the reserved peak at every passing point.

**Ledger** (`memory_ledger.py`, trillian RTX A4000):

| Site at the force call's peak (SrTiO3, 8,640 atoms, 640k edges) | GiB | Per unit |
|---|---:|---:|
| the second interaction's tensor-product weights `[E, 1280]` (kept for the backward) | 3.05 | 5.12 kB per edge |
| `streaming_conv_backward`'s weight gradient `[E, 1280]` (transient) | 3.05 | 5.12 kB per edge |
| the radial networks' hidden activations, 8 × `[E, 64]` | 1.22 | 2.0 kB per edge |
| the messages `[N, 5120]` and two `[N, 2048]` buffers | 0.30 | 37 kB per atom |
| live peak | 7.64 | |

- The weights and their gradient are 80 % of the peak. NaCl (13,824 atoms) has the same sites.
- **The analogues of MH-1's records:**

| MH-1 record | Generic analogue | Frees at SrTiO3 8,640 (bound) | Class | Built |
|---|---|---:|---|---|
| chunked_conv_backward | interaction 2's streaming convolution on MH-1's folded path | 3.05 GiB, the weight gradient | rounding | yes (A) |
| recompute_conv_weights / wprime_backward | the folded path without stored weights | 2.9 GiB | rounding | yes (B) |
| none | checkpoint the radial networks' hidden layers | ≤ 1.1 GiB | exact or rounding | no |
| first_table | interaction 1 is already reassociated; its per-edge tensors are the hidden rows above | – | – | no |
| node_segments / node_region | recompute the node-local product | ≤ 0.3 GiB (≤ 4 %) | rounding | no |
| gate_epilogue | none: no gated non-linearity in the residual block | – | – | – |

## (A) `generic_chunked_conv_backward`

**What it does:**
- A generic interaction with the streaming convolution and a split radial network gets MH-1's folded layer:
  `GenericFoldedRadial`, the weight `[W, 0]`, applied to `[h c, c]`.
- Its convolution is MH-1's `folded_streaming_convolution`. The backward runs in L2-sized edge chunks, and each
  chunk's weight gradient goes straight into the last radial layer's input gradient.
- So the `[E, 1280]` gradient is never formed. The cost: the folded input `[E, 65]` is saved.

**Class and rule:**
- Rounding class: `(h c) W` against `(h W) c`. For OMAT-0 and MP-0 the cutoff is applied inside the radial
  embedding, so the rounding difference is only the zero column.
- Field `chunked_conv_backward`, MH-1's. Stage 2 keeps it to `DECISION_66_GENERIC_CHUNKED_GPUS`.

**Gate:** the stacked generic rewrites against the unrewritten model, with every cell forced into chunks of 37 edges:
- OMAT-0 medium: forces 2.8e-6 (float32) and 3.9e-15 (float64) relative;
- MP-0 medium: 1.5e-6 and 1.9e-15;
- thresholds 1e-4 and 1e-10.

**A/Bs** (`ab_rewrites.py`, 12 blocks of 300 steps, both build orders; b/a, below 1 is faster):

| GPU | NaCl | SrTiO3 |
|---|---|---|
| RTX 4000 Ada (leeloo) | 0.953 / 0.953 (8,000 atoms) | 0.916 / 0.916 (6,655) |
| RTX 5060 Ti (local) | 0.972 / 0.972 (5,832) | 0.949 / 0.951 (3,645) |
| A100 | not measured: Colab had no compute units | |

On by default on the two measured GPUs, off elsewhere. On the RTX A4000 (6 MB L2, 256-edge chunks) the matrix's
step times were 3–4 times the default's (with a CPU suite running beside them), so it stays off there.

## (B) `generic_recompute_conv_weights`

**What it does:**
- On (A)'s path, the forward builds the weights on chip without storing them: MH-1's
  `FusedFoldedForward(store_w=False)`, float32.
- The backward builds each chunk's weights again with cuBLAS into a chunk-sized slot
  (`streaming_conv_folded_recompute_backward`), before the chunk's streaming backward and input gradient.
- It keeps `[h c, c]` in place of `[E, 1280]`.

**Class and rule:**
- Rounding class.
- Field `recompute_conv_weights`, MH-1's, default off.
- It is in `MEMORY_RESCUE` after MH-1's own rung, and only rule 12 or the out-of-memory rebuild turns it on.
- Never in float64 (`RESCUE_FLOAT32_ONLY`); an explicit "on" in float64 is refused.

**Gate:** OMAT-0 medium forces 3.4e-6 relative, MP-0 medium 8.1e-7 (float32, forced 37-edge chunks).

**Cost, against (A) alone:**

| GPU | NaCl | SrTiO3 |
|---|---|---|
| RTX 4000 Ada | 1.045 / 1.048 | 1.088 / 1.090 |
| RTX 5060 Ti | 1.113 / 1.110 | 1.148 / 1.147 |

## Rule 12

**Terms** (`rewrites.memory`):
- (A): `-(weight_numel - (P + 1))` floats per edge: the gradient is gone, the folded input is saved.
- (B): `-weight_numel` per edge, the stored weights.
- (B)'s stage factor: `RESCUE_RESERVE["generic_recompute_conv_weights"] = 1.0`.
- Two fixes found on the way:
  - Rule 12's family check now counts the radial records (MH-1's own `recompute_conv_weights`) for MH-1's class
    only. Before, a generic model with the field on got MH-1's keep fraction too, which halved its estimate.
  - The out-of-memory rebuild maps a rung to its field.

**Matrix** (`est_matrix.sh`, 30 steps after 10, each stage's rewrites set; the memory probe off): SrTiO3 and NaCl on
trillian (RTX A4000), NaNbO3 held out on leeloo (RTX 4000 Ada).

| Stage | Estimate / reserved peak |
|---|---:|
| default, (A) off | 1.008–1.036 |
| default with (A) | 1.018–1.064; NaNbO3 0.995–1.017 |
| (A) + (B) | 1.002–1.110 |

- NaNbO3, held out against SrTiO3 and NaCl's first fit (a factor of 0.975), came out at 0.980–1.014.
- The committed factor, 1, keeps every cell, the small ones included, at or above its peak.
- The reserved peak per stage:

| Cell | Default | (A) | (A) + (B) |
|---|---:|---:|---:|
| SrTiO3 8,640 | 8.30 GiB | 5.49 | 2.48 |
| SrTiO3 13,720 | 13.03 GiB | 8.56 | 3.81 |
| NaCl 13,824 | 6.68 GiB | 4.46 | 2.35 |

- On the 16 GB A4000, SrTiO3 ran at 40,000 atoms with (B) (10.62 GiB), and NaCl at 74,088 (11.42 GiB).

**Tests:** `tests/cpu/test_memory_fit.py` keeps every matrix case in its stage's band: [1.0, 1.05], [0.99, 1.08],
[1.0, 1.12].

## Second round: `generic_recompute_first_table` (2–3 November 2026, branch `fastmace/omat0-capacity-2`)

With `generic_radial_tables` on (`generic-radial-tables.md`), the largest kept tensor per edge is the tables'
output, `[h c, c]`: 65 floats per interaction for OMAT-0 medium. The ledger ranked dropping it as the first remaining
lever. Its bound was 520 B per edge for both interactions, 31 % (SrTiO3) and 21 % (NaCl) of the live peak.

**What it does:**
- The table output is no longer kept for the backward, only the lengths, which the tables keep anyway.
- In the backward the table is evaluated again (same kernel, same input), and the table's gradient is consumed per
  chunk.
- First interaction (reassociated): under a checkpoint.
- The other interactions: inside (A)'s chunked convolution. This needs (B), because without (B) they keep the
  weights, not the table output.

**Class and rule:**
- Exact: the same table kernel on the same input.
- It is a rescue rung after (B), in `MEMORY_RESCUE`, default off.
- Rule 12's terms (`rewrites.memory`):
  - `generic_recompute_first_table`: −65 floats per edge (first interaction only);
  - `generic_recompute_first_table_with_weights`: −130 floats per edge, with (B).

**Two fixes on the way:**
- Rule 12 counted `generic_radial_tables` against what the estimate never held. It now removes the networks' input
  and saved hidden units, not the real saved tensors (`f651530c`). A `generic_*` record now counts for generic models
  wherever it lives (`e99b222f`).
- The GPU gate's tables child had no `species_columns`.
  - Without them, a per-pair (Agnesi) radial is fitted for every element pair of the model: 89² pairs for OMAT-0,
    on the CPU in float64.
  - The leto gate was killed at 128 GB of host memory.
  - It now passes the structure's species, as the planner does (`5d83b59a`).
  - After the fix, `tests/gpu/test_rewrites_generic_gpu.py` passes 4 of 4 in 88 s on leto (`omat2-gate4-leto`).
  - Against the kept tables: forces 3.3e-7 relative, energy equal (MP-0 medium, float32).

**Cost (`ab_rewrites.py --b ... recompute_first_table=on`, b/a in both build orders, 1 fs).** Side a is the stage's
rewrites; side b adds the rung.
- s1 = (A) + tables.
- s2 = (A) + (B) + tables.

| GPU (provider) | Stage | NaCl 10×10×10 | SrTiO3 11×11×11 |
|---|---|---|---|
| RTX 4000 Ada (leto, workstation) | s1 | 1.011 / 1.008 | 1.015 / 1.014 |
| RTX 4000 Ada (leto, workstation) | s2 | 1.051 / 1.050 | 1.063 / 1.061 |
| RTX 5060 Ti (local PC, WSL2) | s1 | 1.010 / 1.007 | 1.019 / 1.017 |
| RTX 5060 Ti (local PC, WSL2) | s2 | 1.042 / 1.042 | 1.038 / 1.037 |

- With (B), each interaction evaluates its table once more per chunk, which costs 4–6 %.
- Without (B), only the first interaction's table is evaluated again, which costs 1–2 %.
- Neither is a default; rule 12 turns the rung on only when the plan would not fit otherwise.

**Tables matrix (`est_matrix_tables.sh`):** the reserved peaks with tables on, at s1 / s2:

| Cell | s1 | s2 |
|---|---:|---:|
| SrTiO3 24,565 (leeloo) | 11.79 GiB | 3.91 GiB |
| SrTiO3 53,240 (leeloo) | – | 8.06 GiB |
| SrTiO3 87,880 (leeloo) | – | 13.20 GiB (985 ms/step) |

The memory the rung frees is not measured yet. The capacity scan with the rung, and the rung's points in rule 12's
matrix, are still to do. The merge tier runs before any merge.

## Third round: two capacity fixes and the new Ada defaults (8 November 2026, branch `fastmace/omat0-ada-capacity`)

A rescan on the 5060 Ti found the limit lower than on 2 November: SrTiO3 34,295 atoms (was 53,240), NaCl 64,000 (was
85,184), with only 3.5 GiB in use at the last passing size. Two bugs, both since `generic_wprime_backward`
(3 November):
- `attach_generic_wprime` raised when rule 12 took rung (B), `generic_recompute_conv_weights`: every interaction is
  then on the recompute path and none is a candidate. The planner fell back to the rewrites-off terminal plan, which
  ran out of memory. It now applies to none and returns no names.
- Rule 12 had no memory term for `generic_wprime_backward` (which, with the fused folded forward, stores no
  [E, 1280] weights). It predicted about 20.8 GiB at SrTiO3 40,000 (real about 4.1 GiB) and took (B) beyond about
  35,000 atoms. The term is now (B)'s, with a `CONTEXT_TERMS` entry.

**5060 Ti with both fixes** (`capacity_scan.py scan`, as above):

| Crystal | 2 November: atoms, ms/step | 8 November: atoms, ms/step, µs/atom/step | Peak, estimate (GiB) | Rung |
|---|---|---|---|---|
| SrTiO3 | 53,240, 840 | **163,840**, 1,286, 7.85 | 14.59, 17.47 | recompute_first_table |
| NaCl | 85,184, 623 | **195,112**, 809, 4.15 | 14.11, 17.01 | recompute_first_table |

At 53,240 SrTiO3 the step is now 291 ms (5.47 µs per atom). First failures: SrTiO3 179,685, NaCl 216,000. The
estimate is 1.18–1.21 of the peak where the near-limit probe corrected it.

**RTX 4000 Ada with the new defaults** (spline w and the composed skip on for OMAT-0 medium, ledger `ada-omat0-best`):
pending. The scan and an old/new peak comparison at the same sizes are queued on leto.

[Results](../../../results/omat0-ada-capacity-20261108/README.md).

## Records

`results/omat0-capacity-20261102/`:
- the scripts: `scan_ws.sh`, `ledger_ws.sh`, `ab_ws.sh`, `ab_local.sh`, `est_matrix.sh`, `matrix_cases.py`,
  `band.py`, `terms.py`, `gpu_tests.sh`, `cpu_suite.sh`;
- the matrix cases, `cases-*.json`, also in `tests/_support/data/rule12_generic_rescue.json`.

Points, A/B records and ledgers are on the NAS, `/mnt/homes/zwingel/fastmace-capacity/omat-*`.

Second round: the scripts are in `results/omat0-capacity-2-20261102/` (`ab_local.sh`, `ab_ws.sh`, `est_matrix_tables.sh`,
`saved_probe.py`). The records are on the NAS, `/mnt/homes/zwingel/fastmace-capacity/omat2-*`:
- the A/Bs, `omat2-ab3-*`;
- the gates, `omat2-gate*`;
- the matrix, `omat2-matrix-*`.

The local A/Bs are in `/mnt/homes/zwingel/fastmace-capacity/local-5060ti/omat2-ab3-*-local*`.
