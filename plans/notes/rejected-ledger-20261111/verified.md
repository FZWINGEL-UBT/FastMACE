# The 27 agent notes, checked against the records (11 November 2026)

Source: `agent-notes.md` in this folder (unverified reports from the subagents that built the catalogue), on main
`f31d4f50`. Each row of its two tables is checked against the ledger's own entry, the linked results README or
report, SPEC §19 and the code's GPU lists. The line numbers in `agent-notes.md` are from an older ledger, so IDs are
used here. Only defects confirmed against a record were fixed, on branch `fastmace/docs-ledger-1111`.

Counts: **14 confirmed and fixed**, **11 not confirmed**, **2 unclear**.

## Index row and fuller record disagree (12 rows)

| # | ID | Verdict | What the record says, and what changed |
| --- | --- | --- | --- |
| 1 | `omat0-stream-w` | confirmed, fixed | Kernel-lab README §3 (5060 Ti step 0.987) and §10 (A100-SXM4 1.0048, "rejected on the A100 too"). Appendix Status PENDING → REJECTED, with step numbers; index unchanged. |
| 2 | `omat0-stream-store-w`, `omat0-eos-chunks` | confirmed, fixed | README §3: stcs 1.001 / 1.001 (Ada), 0.991 (5060 Ti); eos 1.002 / 1.004 (Ada). Both appendix entries PENDING → REJECTED, with step numbers. |
| 3 | `omat0-slope-vector` | confirmed, fixed | README §3: d5-slopes2 1.027 / 1.070 / 1.040 (Ada / 5060 Ti / A100); slopes2 0.991 at degree 10 (the record's 0.9909). Appendix "the A100 is open" → REJECTED on all three; both figures kept with their degree; index and entry now name `SPLINE_VECTOR` (`SPLINE_VECTOR_GPUS`) as the later default in that direction. |
| 4 | `ideas-r5` | confirmed, fixed | Record said "14(A): BUILDING" and "Not built on the A100-SXM4"; round 7 rejected 14(A) and `ff4d1a89` made `W_GROUP` default on the A100-SXM4 (`W_GROUP_GPUS`). Both lines updated with links. |
| 5 | `ideas-r7` | confirmed, fixed | The index link stopped at `...lab-at-bcdeb`; completed to the README heading "Round 7 ... lab at `bcdeb501` (6 November 2026)". |
| 6 | `ideas-r8` | confirmed, fixed | The cut tail of row 5 (`501-6-november-2026) |`) sat after row 6's last cell; removed. |
| 7 | `virial-elementwise` | confirmed, fixed | `ada-5060-screen-20261110` README: Ada 1.0041 / 0.9805, rerun 1.0045 / 0.9810, off. Status "H200, Ada not measured" → Ada measured, order-dependent, off; H200 not measured. |
| 8 | `rad-radial-entry` | not confirmed | 1.0008 / 1.0060 rounds to opt-4's 1.001 / 1.006; the entry already says so. No change. |
| 9 | `prec-tf32x3-triton` | confirmed, fixed | `results/opt-T01-gemm-precision-20261020/local/summary.json` (RTX 5060 Ti, 21:47 UTC, `3d301ec`) and `opt-1.md` show the rerun ran. A dated note in the opt-T01 report header points to it; the ledger's CONFLICT line now says this. |
| 10 | `prec-fp16x3-other-gpus` | confirmed, fixed | `opt-5.md` §3.11 (26 Oct) rejected it on the A100, A4000 and Ada; `plans/opt-5.md` and the fp16x3 note §3.7 still called it a candidate. Dated notes added to both; ledger CONFLICT line updated. |
| 11 | `launch-regional-persistence` | not confirmed | NOT BUILT in index and entry; the entry already records the 23 October design note's wording and that no owner close-out exists. Nothing to fix without an owner decision. |
| 12 | `setup-max-autotune`, `setup-activation-budget` | unclear | Both records sit in `ResearchModule` (outside this checkout; not to be written). The entries already flag the conflicts. For the budget, NaNbO3 23.2 → 27.4 ms is +18 %, so "9–20 %" may be the same data in percent. |

## Status and numbers pull apart (15 rows)

| # | ID | Verdict | What the record says, and what changed |
| --- | --- | --- | --- |
| 13 | `lin-gemm-autotune` | not confirmed | The 5 % gain is on the V2 cuEq replay; on the 10 October hybrid default it is 0.1–1.0 % slower, and that is the rejection. Consistent. |
| 14 | `prec-fp16x3-standalone` | not confirmed | The comparison that decides is against cuBLAS IEEE (ties or loses); 1.78–1.88× faster than tf32x3 is a different baseline. |
| 15 | `prec-fp32-baselines` | not confirmed | 493.9 µs is one first run; the repeated cuBLASLt figure is 501.9 µs against `torch.mm` 501.0. |
| 16 | `prec-d1-56bits` | confirmed, fixed (a different defect) | SUSPENDED for correctness is right, and the faster ratios do not contradict it. But the appendix entry still said DEFAULT and called the suspension branch unmerged; it is on main since `aa5850dc` (`fixed.py` rule `dynamic_decision_65_d1_suspended_after_recapture_defect`). Status rewritten to match the index. |
| 17 | `eng-bucket-width` | confirmed, fixed | The index Effect showed only "+0.4 % to noise". It now also gives the 2 % force-replay gain (NaNbO3 3.04 %, NbCl5 0.70 %) and says the MD step gained nothing. |
| 18 | `model-polar-phase-recurrence` | not confirmed | NbCl5 1.075 / 0.988 flips with the order; with the other pairs inside ±0.8 % that is a tie, as stated. |
| 19 | `model-mh1-fused-first` | not confirmed | Every pair except one L NbCl5 order is above 1 in both orders (1.001–1.006); "slower" (not faster) holds. |
| 20 | `launch-megakernel` | not confirmed | The entry records that the POLAR-1 gaps meet the reopen condition and that the conductor kept the full step NOT BUILT (7 Nov). A recorded decision, not a mismatch. |
| 21 | `launch-sched-iso` | unclear | `opt-5.md` gives 0.86 for `sched_iso` 6×2; the probe report's table has `sched_iso` rows for 4×4 only (0.72). No source for 0.86 found. The rejection does not depend on it; the entry already notes this. |
| 22 | `sc-opt3-t02` | confirmed in part, fixed | The October screen is MH-1 on the RTX 5060 Ti (memory SOL 72–96 %). `step-limiters-20261109` is OMAT-0 on the A100 (FMA-bound) and H200 (near issue-bound). The Why line now says which case each holds for. |
| 23 | `forward-ghoist` | confirmed, fixed | `a100-colab-ideas-20261111` README: A100 lab 0.912, built as `GHOIST`, MD 0.977–0.989, "superseded by `GEOM`" (`GEOM_GPUS` holds both A100 models). Status REJECTED → REJECTED on the 5060 Ti, SUPERSEDED by `GEOM` on the A100 family. |
| 24 | `wprime-zero-byte` | not confirmed | "Bound" here means a timed falsifier (operand loads replaced by one integer op); the timed probes are that bound. |
| 25 | Row 598 = `h200-cutlass-tile` | not confirmed | `h200-fastpath-20261108` README: 1.0023 / 1.0030 and 1.0014 / 1.0035, above 1 in both orders, "slower: closed". |
| 26 | Row 596 = `r128-min-blocks` | not confirmed | The same README gives the reason: a register cap for 7 or 8 blocks per SM, 1.0097 / 1.0103 and 1.0515 / 1.0507, "the cap stays 6". The index status carries these numbers. |
| 27 | `conv-fused-radial-conv` | confirmed, fixed | The report's convention (positive = faster) is in the entry, not the index; the index Effect now states it. |

Not covered: the section "Records that could not be read" in `agent-notes.md` (ResearchModule rows, entries without
a fuller record, names outside `fastmace/` and `tools/`). It lists gaps, not contradictions.
