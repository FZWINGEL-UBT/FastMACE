# What the agents reported beside their records (11 November 2026)

Collected from the final messages of the eleven Haiku subagents that built the catalogue. These are their
observations while reading `docs/optimisation-ledger.md` and its linked reports. **None is verified.** Each is a
place where the index row and the fuller record, or the status and the numbers, do not obviously agree.

## Index row and fuller record disagree

| ID (ledger line) | What was reported |
| --- | --- |
| `omat0-stream-w` | The index says REJECTED with step numbers; the fuller record says PENDING with lab numbers only. |
| `omat0-stream-store-w` (461), `omat0-eos-chunks` | The fuller record says PENDING; only the index row has step numbers. |
| `omat0-slope-vector` (456) | The fuller record says the A100 is still open; the index gives A100 numbers. The Ada figure is 1.027 in the index and 0.9909 / 1.0276 in the record. |
| `ideas-r5` (501) | The index says 14(A) REJECTED; its own record says 14(A) BUILDING. The rejection is recorded under `ideas-r7`. |
| `ideas-r7` (502) | The research-answers link is cut off, so its anchor is missing; the file exists. |
| `ideas-r8` (503) | The Evidence cell is garbled after the link. |
| `virial-elementwise` (573) | The row says the Ada was not measured; row 626 (11 November) reports an Ada measurement (1.0041 / 0.9805). |
| `rad-radial-entry` (163) | NbCl5 1.0008 / 1.0060 in the index, 1.001 / 1.006 in opt-4. |
| `prec-tf32x3-triton` (260) | The report header says the sm_120 rerun is still to do; two other documents say it is done. The sm_120 numbers are not in the report. |
| `prec-fp16x3-other-gpus` (277) | Status REJECTED, but the opt-5 plan and the fp16x3 note §3.7 still call it a candidate pending owner permission. |
| `launch-regional-persistence` | The 23 October design note conflicts with the 10 November status (the entry records this). |
| `setup-max-autotune`, `setup-activation-budget` | The records themselves flag a conflict with the roadmap and with the switch map (9–20 %). |

## Status and numbers pull in different directions

| ID (ledger line) | What was reported |
| --- | --- |
| `lin-gemm-autotune` | REJECTED, yet its V2 numbers show a 4.98–5.72 % gain; the record explains it by a slowdown of the hybrid default. |
| `prec-fp16x3-standalone` (275) | The Effect cell says "ties or loses to cuBLAS IEEE"; the record also gives 1.78–1.88× faster than tf32x3. |
| `prec-fp32-baselines` | One cuBLASLt run (493.9 µs) is below torch.mm (501.0 µs), but the reason given is that no FP32 alternative beat cuBLAS. |
| `prec-d1-56bits` | SUSPENDED after wrong energies on CUDA-graph re-capture, not a speed reject; the quoted ratios (0.86–0.88) are faster. |
| `eng-bucket-width` | The index says "+0.4 % to noise"; the record also gives a 3.04 % gain for 2 %. |
| `model-polar-phase-recurrence` | Status says "tie", but the S NbCl5 ratios are 1.075 and 0.988. |
| `model-mh1-fused-first` (422) | "Slower" rests on ratios of 1.001–1.006, with one 0.970 order read as noise. |
| `launch-megakernel` | The entry says the 6–12 % NaCl gaps meet its reopening condition, yet the status stays NOT BUILT. |
| `launch-sched-iso` | opt-5 cites 0.86 for `sched_iso 6x2`, which the probe table does not show. |
| `sc-opt3-t02` (245) | The reason "memory-bound at its traffic floor" is undercut by a 9 November note (H200 FP32-issue bound). |
| `forward-ghoist` | Rejected, while its A100 re-screen is faster (0.912); closed as superseded by `GEOM`. |
| `wprime-zero-byte` | STOPPED (bound), but its H200 numbers are timed probes. |
| Row 598 | Says "slower", but both quoted ratios lie within 0.99–1.01. |
| Row 596 | No reason in the index; the record gives only "slower". |
| `conv-fused-radial-conv` | The sign convention of its numbers is ambiguous (the report says positive means faster). |

## Records that could not be read

- **ResearchModule reports** (about 30 rows across the October areas): the evidence is under `ResearchModule:`,
  outside this checkout, so `record_exists` is null and only the ledger text was used.
- **No fuller record in the ledger:** `eng-rebuild-cost`; `codex1109-fused-outer-sum` (its link is a folder of CSV
  and JSON files without a README); rows 525, 526, 528 and 529 (their records are sections of
  `results/step-limiters-20261109/README.md`); `elt-gate-backward-epilogue` (137, only a `#screen-table` anchor).
- **Filled from the linked plan, not the ledger:** `omat0-producer-consumer`, `omat0-wprime-maxreg-a100`,
  `omat0-wprime-split`, `omat0-mma-chain`, `omat0-hybrid16` (`parts/part-08-fix.jsonl`).
- **Named code not under `fastmace/` or `tools/`:** `formulation_search.py`, `FIRST_LAYER_REGION` (removed),
  `empty_cache_before_warmup`, the scripts of `breakthroughs-codex-20261108`, `certified-skip` and
  `tc-overlap-bound` (they sit under `results/`), `tiled-lazy-keys` (reverted).
