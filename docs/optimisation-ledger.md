# Optimisation ledger

Status: compiled on 28 October 2026 from FastMACE main `8a269a91` (read in the worktree at `312942df`, which is
main plus two tool commits), the unmerged branches named in the entries (`fastmace/opt7-plan`, `fastmace/f64-T04`,
`fastmace/fp64-d1-relax-fix`, `fastmace/radial-table-cache`, `fastmace/opt2-T01`, `opt2-T02`, `opt2-T02c`,
`opt2-T05`, `opt-T01`, `opt-T01b`, `opt-T02`, `opt-T02c`, `opt-T03`) and the ResearchModule Langevin documents
(`Experiments/Langevin_Thermostat/`, cited as `ResearchModule:LT/...`, where `LT/` stands for
`Experiments/Langevin_Thermostat/`). The research engine is ResearchModule's `--engine gpu`
(`langevin/gpu_langevin.py`), frozen for FastMACE's A/B comparisons as tag `research-engine-baseline-20261017`.

This file is the index and the verdict list. It does not replace the evidence: the per-task reports, plans and
SPEC decisions it links to remain authoritative for numbers and methods. Where two sources disagree, the entry
keeps both values and marks them "CONFLICT:". Numbers are copied as reported; "not recorded" means no source gives
one.

Maintenance rule: every new optimisation attempt adds or updates its index row and its appendix entry in the same
commit as its report. A change of a default cites its SPEC §18 decision (and the §19 amendment that records it)
in the row's "Decision/date" column. Rows are never deleted: a reverted or superseded attempt changes its status.

Conventions used throughout:

- b/a is side b (candidate) time over side a (baseline) time in a same-process A/B with alternating blocks
  (12 blocks of 300 MD steps unless stated); two values are the two build orders (a-b / b-a). Below 1 is faster.
- GPUs: local = RTX 5060 Ti 16 GB (sm_120, WSL2, desktop shares the GPU); murphy, leto, leeloo = RTX 4000 Ada
  Generation (sm_89); L4 = Colab or Modal L4 (sm_89); A100 = Colab or Modal A100 (sm_80); coraline = RTX A4000
  (sm_86, per `ResearchModule:LT/studies/hybrid_oeq/coraline.md`); trillian = RTX A4000 (sm_86).
- Structures are 3×3×3 supercells: NaCl 216 atoms, NaNbO3 1,080, NbCl5 1,944. MH-1 is `mh1-r2scan`
  (`matpes_r2scan` head), hybrid cuEq/OEq, float32, TF32 off, NVT 300 K, unless stated.
- Noise floors of the A/B tool: local 1.1 % (NaCl) and 0.4 % (NbCl5) ([2-T12](validation/2-T12-local-noise-floor.md));
  murphy 0.16 % ([1a noise floor](validation/1a-noise-floor.md)).
- Engine column: RE = research engine, FM = FastMACE, both = an RE idea carried into FastMACE.
- Status vocabulary: DEFAULT; DEFAULT (scoped: …); OPT-IN (a plain off-by-default switch); NEVER-AUTOMATIC (a
  switch whose `auto` rule is `never_automatic`, only an explicit "on" enables it); REJECTED (built or measured,
  then not adopted); CUT (stopped at a falsifier before a record was built); NOT BUILT (dropped on paper, not
  started, or absorbed); SUPERSEDED; ON BRANCH (exists only on an unmerged branch, not judged); PENDING (an open
  owner decision or an unfinished check); MEASUREMENT (profile, parity or comparison, no switch).

## How optimisation work is documented today

| Document type | Where | Authoritative for |
| --- | --- | --- |
| Round plans | [`plans/opt-1.md`](../plans/opt-1.md) … [`opt-6.md`](../plans/opt-6.md), [`opt-8.md`](../plans/opt-8.md); `plans/opt-7.md` on branch `fastmace/opt7-plan` only | Candidates, ceilings, stop rules and (in opt-2, opt-3, opt-4 §2, opt-6, opt-8) a task status table. [`plans/opt-4.md`](../plans/opt-4.md) §2 is the only main-branch list of every round 1–3 idea with its status. |
| Ideas round | [`plans/ideas-20261027.md`](../plans/ideas-20261027.md) | Float64 ideas D1–D5 and float32 ideas F0–F5, falsifiers, per-GPU confirmation, the text behind decision 65. No separate report. |
| Task reports | [`docs/validation/`](validation/) (`opt-T0x`, `opt2-T0x`, `opt3-T0x`, `opt4-T0x`, `opt5-T01`, `opt6-T0x`, `opt8-T0x`, `f64-T01`, phase and comparison reports) | Measured numbers, methods, raw-record paths, per-task verdicts. |
| Round reports | [`validation/opt-1.md`](validation/opt-1.md), [`validation/opt-5.md`](validation/opt-5.md), [`validation/opt-def-2.md`](validation/opt-def-2.md) | Round summaries. Rounds 2, 3, 4, 6, 8 and the ideas round have none. |
| Design notes | [`docs/design/`](design/) (`opt-T02-BC-go-no-go`, `opt-T02-C-gate-prologue`, `opt-T03b-diagnosis`, `opt-T06/T07/T08`, `opt-T09/T10-step-breakdown`, `opt3-T01/T02/T05`, the megakernel and fp16x3 notes) | Designs, breakdowns, analyses that ended without a build. |
| SPEC §18 | [`SPEC.md`](../SPEC.md) §18 "Risks and open questions" | Numbered owner decisions. Numbered items present: 1–7, 39, 50, 57–59, 62–65. Other decision numbers (for example 37, 38, 52–55, 60, 61) are cited elsewhere but are not numbered items of §18. |
| SPEC §19 | [`SPEC.md`](../SPEC.md) §19 "Revision history" | Amendments; the twelfth to fourteenth record the unnumbered round 1–3 adoptions; the fifteenth `grouped_launch`; the sixteenth decision 58; the seventeenth decision 59; "Optimisation round 6" decisions 60 and 61. Nothing for 62–65. |
| SPEC §6.1 / §6.2 | [`SPEC.md`](../SPEC.md) §6.1 (rewrite record, exactness classes incl. `certified`), §6.2 catalogue | The catalogue of rewrites with the research engine's gains (the only place for many RE numbers). Lacks rows for later records (see defects). |
| Screen ledger | [`docs/rewrites.md`](rewrites.md) | The most complete per-screen table (RE rejected screens; FastMACE screens to round 8). Some rows are stale. |
| Switch map | [`docs/design/switch-map.md`](design/switch-map.md) | Every config field, its default and its `auto` rule; mirrors `fastmace/planner/fixed.py`, which is the final authority for defaults. No "measured" column. |
| Benchmarks | [`docs/benchmarks.md`](benchmarks.md) | Published step-time table (25 October 2026, Colab L4). Out of date. |
| Branch-only documents | `fastmace/opt2-T01`, `opt2-t02`/`opt2-t02c`, `opt2-T05`, `opt-T02`, `opt-T03`, `opt7-plan`, `f64-T04`, `fp64-d1-relax-fix`, `radial-table-cache` | Evidence of rejected or unmerged work (cited as "branch fastmace/x: path"). |
| ResearchModule | `ResearchModule:LT/docs/roadmap.md`, `LT/docs/status.md`, `LT/docs/configuration*.md`, `LT/studies/*` (dated reports), `AGENTS.md` | The research engine's history, 1–16 October 2026, and the production rules that still apply to `main.py`. |
| Screen adoption rule | [`plans/phase-2.md`](../plans/phase-2.md) 2-T20 ("Speed screens land off") | Screens are rewrite records merged "off"; adoption needs the 19-label gate, a same-process gain beyond the local floor on MH-1 and a generic model, and a murphy confirmation; the owner turns them on. The optimisation rounds replaced it in practice (see defect 29). |

Unmerged branches with optimisation content (none is an ancestor of main `8a269a91`; last commits from
`git log -3 --format='%cs %s'`):

| Branch | Last commit | Content | Ledger entries |
| --- | --- | --- | --- |
| `fastmace/opt-T01` | 2026-10-20 `4bad42a7` "OPT-T01: report (rejected on sm_89) and docs/rewrites.md" | tensor-core bench launchers; the report itself is on main | [prec-tf32x3-triton](#prec-tf32x3-triton) |
| `fastmace/opt-T01b` | 2026-10-20 `7648e749` "docs/rewrites.md: OPT-T03 concurrency candidates not built on sm_89" | overlap bench | [prec-tc-overlap](#prec-tc-overlap) |
| `fastmace/opt-T02` | 2026-10-20 `5f1f54d4` "opt-T02 design: review answers (A approved, shared tile rule, D rejected)" | `docs/design/opt-T02-fusion.md` | [lin-fusion-mechanisms](#lin-fusion-mechanisms), [elt-gate-backward-epilogue](#elt-gate-backward-epilogue), [elt-density-prologue](#elt-density-prologue) |
| `fastmace/opt-T02c` | 2026-10-20 `3dc0ca59` "OPT-T02 stage C: launchers for the murphy and local runs" | `gate_prologue` record | [elt-gate-prologue](#elt-gate-prologue) |
| `fastmace/opt-T03` | 2026-10-20 `baaaa543` "opt-T03 note: wording of the reading" | `docs/design/opt-T03-small-m.md` | [lin-concurrent-branches](#lin-concurrent-branches), [lin-split-k](#lin-split-k) |
| `fastmace/opt2-T01` | 2026-10-21 `dfd74525` "OPT2-T01 rejected at gate 1 …" | fused radial + conv prototype and design note | [conv-fused-radial-conv](#conv-fused-radial-conv) |
| `fastmace/opt2-T02` | 2026-10-21 `deaa5f90` "OPT2-T02: local A/B, NaCl -1.5 %, NbCl5 +1.0 %: not adopted …" | `fused_radial_mlp` record | [rad-fused-radial-mlp](#rad-fused-radial-mlp) |
| `fastmace/opt2-T02c` | 2026-10-21 `65fad58c` "OPT2-T02c: gate passed, NbCl5 A/B 1.0008 / 1.0060: stopped, record stays opt-in" | entry kernels, design note | [rad-radial-entry](#rad-radial-entry), [rad-fused-radial-mlp-recompute](#rad-fused-radial-mlp-recompute) |
| `fastmace/opt2-T05` | 2026-10-21 `ff014456` "OPT2-T05: validation report, local results; rejected on sm_120" | `split_k` record and report | [lin-split-k](#lin-split-k), [lin-split-k-triton](#lin-split-k-triton) |
| `fastmace/opt7-plan` | 2026-10-27 `799ad6de` "OPT7: wide loads documented as eligible, not built (owner, 27 October 2026)" | round 7 plan, report, results | [conv-opt7-share](#conv-opt7-share), [ext-lunatp](#ext-lunatp), [conv-wide-loads](#conv-wide-loads) |
| `fastmace/f64-T04` | 2026-10-27 `eb6c56fc` "f64-T04: double-float falsifier on sm_120 …" | `docs/validation/f64-T04.md`, format-rate bench | [prec-double-float](#prec-double-float), [prec-ozaki-fp8](#prec-ozaki-fp8) |
| `fastmace/fp64-d1-relax-fix` | 2026-10-28 `bbd8bd23` "Suspend decision 65 D1: fp64_emulation_mantissa_bits auto = dynamic everywhere (owner, 28 October 2026)" | D1 suspension and relaxation fixes | [prec-d1-relax-fix](#prec-d1-relax-fix), [prec-d1-56bits](#prec-d1-56bits) |
| `fastmace/radial-table-cache` | 2026-10-28 `54b6f13f` "radial_tables: cache the fitted and certified tables under FASTMACE_CACHE/radial-tables" | table cache | [rad-table-cache](#rad-table-cache) |

### Known documentation defects

Merged from the four extraction slices and this compilation. Suggested fixes are suggestions only; none is applied
here.

| # | Defect | Where | Evidence | Suggested fix |
| --- | --- | --- | --- | --- |
| 1 | Default status of `radial_tables` contradicts | [rewrites.md](rewrites.md) OPT6-T05 row vs [switch-map](design/switch-map.md) §1 and SPEC decisions 62/63 | rewrites.md: "Opt-in (`never_automatic`, float32 only)"; switch-map and SPEC: default on, float32 and float64 | Update the rewrites.md row to "default (decisions 62, 63)". |
| 2 | `chunked_conv_backward` has two rows | [rewrites.md](rewrites.md) | An "Opt-in (default off)" row (OPT6-T02) and a later "Default on" row (decision 60) | Mark the first row superseded. |
| 3 | `fused_folded_forward` called opt-in | [rewrites.md](rewrites.md) OPT8 row vs switch-map §1 and decision 64 | rewrites.md: "opt-in record (owner), not a default"; switch-map: default for MH-1 float32 (commit `e0143c6a`) | Update the row. |
| 4 | Decision 65 D1 suspended only on an unmerged branch | SPEC decision 65, switch-map lines 90–91, ideas file; branch `fastmace/fp64-d1-relax-fix` | `bbd8bd23` (28 Oct) sets `fp64_emulation_mantissa_bits` auto = dynamic everywhere after wrong energies after a graph re-capture; main still documents 56 bits; ResearchModule AGENTS.md already says to pass `"dynamic"` | Merge the branch or record the suspension in SPEC §18/§19 and the switch map. |
| 5 | Decisions 60 and 61 not in §18 | SPEC §18 | Recorded only in §19 "Optimisation round 6" and §6.1/§6.2 | Add numbered items 60 and 61. |
| 6 | No §19 amendments for decisions 62–65 | SPEC §19 | Amendments end at round 6 / 6-T15 | Add amendments. |
| 7 | §6.2 catalogue incomplete | SPEC §6.2 | No rows for `fused_folded_forward`, `w_tables`, `save_transcendentals`, `tf32x3_gemms`, `fp64_emulation*` and the float64 package | Add rows. |
| 8 | Round 1–3 adoptions have no numbered decision | SPEC §18 | `gated_rows`, `segment_gemm`, `model_template_cache`, OPT-T08 records, `overlap_rebuild`, `folded_backward_tiles`, `fused_integrator` are unnumbered owner decisions in §19 amendments 12–14; `persistent` is a conductor decision | Give them numbers or a cross-reference table in §18. |
| 9 | Rule-name drift | SPEC decision 62; [opt-def-2](validation/opt-def-2.md) | Decision 62 names `on_with_fold_cutoff_float32_unless_reproducible`; code uses `on_with_fold_cutoff_unless_reproducible` after decision 63. OPT-DEF-2 names `on_float32_unless_reproducible_if_model_supports` and `on_with_block_linear_float32_unless_reproducible_if_model_supports`; switch-map now drops `float32` | Add a note in both places. |
| 10 | Switch-map counts stale | [switch-map](design/switch-map.md) "Counts" (line 366) | Says `overlap_rebuild` and `step_graph` were added "both opt-in" (`overlap_rebuild` is default `True`); counts stop at phase 5; OPT-T06 and OPT-T07 both claim "71" fields | Recount; fix the wording. |
| 11 | Misplaced rows in the switch map | switch-map §2, §10 | Two rule definitions (`on_for_mh1_float64_on_measured_gpus_decision_65`, `56_for_mh1_…_else_dynamic`) sit in the §2 options table; five rule rows sit in §10's `FallbackConfig` table | Move them to "Auto resolution rules". |
| 12 | `docs/benchmarks.md` out of date | [benchmarks.md](benchmarks.md) | Still the 25 October L4 table (before opt-6 and opt-8) and "A full table on this GPU is still to be measured"; MH-1 is now about 2.7 / 14.6 / 18.6 ms locally (NaCl / NaNbO3 / NbCl5) | Re-measure and replace; link this ledger's speed reference. |
| 13 | Missing round reports | `docs/validation/` | No `opt-2.md`, `opt-3.md` (planned as opt-2 step 6), `opt-4.md` (planned OPT4-REP), no round 6, round 8 or ideas-round report; plan rows name `opt5-T02.md` … `opt5-T05.md`, which do not exist (consolidated in [opt-5](validation/opt-5.md)) | Write short round reports or point the plans at the existing ones. |
| 14 | Result documents only on branches | branches `fastmace/opt2-T01`, `opt2-t02`/`opt2-t02c`, `opt2-T05`, `opt-T02`, `opt-T03`, `f64-T04`, `opt7-plan` | `docs/design/opt2-T01-fused-radial-conv.md`, `opt2-T02-radial-mlp-fusion.md`, `opt2-T05-split-k.md` (+ validation), `opt-T02-fusion.md`, `opt-T03-small-m.md`, `docs/validation/f64-T04.md`, `plans/opt-7.md`, `docs/validation/opt7-T00-T02.md`; cited from main (switch-map row 55/57, rewrites.md, opt-T02a report) | Cherry-pick the documents (not the code) onto main. |
| 15 | Cited document missing everywhere | [plans/opt-1.md](../plans/opt-1.md) | `opt-T01-survey.md` is cited but is in no branch | Remove the citation or point to [opt-T01-tensor-cores](validation/opt-T01-tensor-cores.md) §1. |
| 16 | Records described as opt-in that are not on main | rewrites.md, opt-4 §2 | `gate_prologue`, `split_k`, `fused_radial_mlp` are "opt-in records" but exist only on their branches, not in `fastmace/config.py` | Say "on branch, not merged". |
| 17 | Stale headers in task reports | [opt-T01-tensor-cores](validation/opt-T01-tensor-cores.md), [opt-1](validation/opt-1.md), [opt-T08-linear](validation/opt-T08-linear.md), [opt2-T03](validation/opt2-T03.md), [opt2-T04](validation/opt2-T04.md), [opt-T02a](validation/opt-T02a-gated-rows.md) | sm_120 rerun "still to do" (done, numbers not in the doc); combined A+B "estimate … pending" (measured); "murphy not measured, adoption rule not met" (adopted on L4/A100); "murphy A/B pending … `False` by default" (defaults); rule `never_automatic` in the body vs default on in the header | Add dated status lines. |
| 18 | Stale plan statuses | [opt-4](../plans/opt-4.md), [opt-5](../plans/opt-5.md), [opt-6](../plans/opt-6.md), [ideas](../plans/ideas-20261027.md) | opt-4: "Nothing in it is implemented yet", `grouped_launch` "opt-in", no done notes for T01/T06/T07; opt-5: "Nothing … implemented yet", T01–T05 without status, T08 follow-up says standalone 3×FP16 "remains a candidate" (rejected in opt-5.md §3.11; same in the fp16x3 design note §3.7); opt-6 T03 "not started" without "superseded"; ideas: "Nothing is committed" | Update the status lines. |
| 19 | `step_graph` "neutral on both" | [opt-2](../plans/opt-2.md) | Only the local card measured it; the murphy A/B that the report made a condition never ran | Say "local only". |
| 20 | `inplace_block_linear` called opt-in | switch-map §1, SPEC §6.2 | Code refuses `True` (`fastmace/planner/fixed.py:722`, "not carried over in 1a"); the 2-T20 screen of it has no result | Call it "not carried over (refused)". |
| 21 | `selfcheck_cache` wording | switch-map §1 | "no effect until 2-T19"; [2-T19](validation/2-T19-setup.md) recommended not to build it; no later note whether the field stays inert | State "not built (2-T19); field inert". |
| 22 | Carried-over rewrites have RE numbers only | SPEC §6.2; FastMACE validation | `sort_receivers`, `density_first`, `block_linear`, `streaming_conv`, `species_skip`, `pipelined`, … were never re-measured one at a time in FastMACE; `async_output` and `single_graph_strain` have no speed number anywhere | Accept, or run one ablation pass. |
| 23 | RE rejected screens duplicated with different numbers | switch-map §7, rewrites.md, SPEC §6.2 | `ACTIVATION_MEMORY_BUDGET`: "9–20 % slower" (switch-map) vs "+0.4 to +5.9 ms" (rewrites.md); `JOINT_FUSED_BACKWARD`: only rewrites.md adds "slower on small cells" | Keep one table (this ledger) and link it. |
| 24 | Rough cross-process times cited as results | [1a-T13](validation/1a-T13-engine.md) | Single-run "sanity" times (3.76 / 3.53 ms) appear in later summaries | Cite [phase-1a](validation/phase-1a.md) instead. |
| 25 | OPT-T05 close-out undecided | [opt-4](../plans/opt-4.md) §2 vs [heterogeneous megakernel note](design/2026-10-23-heterogeneous-megakernel.md) | opt-4 recommends closing it; the design note still lists it as "nearer-term work" | Record the owner's call. |
| 26 | Number variants (rounding or measurement sets) | several | `gated_rows` −3.7 / −10.6 % (plan, SPEC, rewrites) vs −3.5 to −3.9 / −10.5 to −10.7 % (report); `overlap_rebuild` NaCl 1000 K −3.3 to −3.7 % (report) vs −3.3 to −3.6 % (plan, SPEC); OPT-T06 0.34–0.80 vs 2.6–3.9 s vs per-backend 2.58–3.94 s; OPT2-T02c 1.0008 / 1.0060 vs 1.001 / 1.006; OPT6-T02 two measurement sets merged into ranges in §6.2; decision 64 quotes averaged b/a (NbCl5 0.967, NaCl 0.996) | Quote both orders, as the reports do. |
| 27 | Naming inconsistencies | design notes, branches, task IDs | Stage C is "gate in the GEMM's epilogue" in the design and [opt-T02b](validation/opt-T02b-segment-gemm.md) but was built as a prologue (`gate_prologue`); branch names mix `fastmace/opt2-T0x` and bare `opt2-t02`, `opt-t08`, `opt3-t01`; task IDs mix `OPT-T0x`, `OPT2-T0x`, `OPT-T03b`, `OPT-DEF-2`, `OPT4-T05b` (follow-up) and `T05b` (default); OPT-T09/T10 are measurements numbered like tasks; ideas use D/F letters | Use this ledger's IDs as aliases. |
| 28 | Plan idea differs from the build without a note | [opt-3](../plans/opt-3.md) OPT3-T03; OPT-T02 B design | OPT3-T03's plan describes in-kernel Philox; the build reads host-drawn NumPy noise. OPT-T02 B's design included "the density scale in the prologue"; the built `segment_gemm` report does not mention it | Add one line each. |
| 29 | Adoption rule loosened silently | [opt-3](../plans/opt-3.md), round 2 | OPT-T08, OPT2-T03, OPT2-T04 and OPT3-T03 were adopted on sm_120 plus L4 (and A100), not murphy; the round-1 "both GPUs" rule became "gain on sm_120, no loss on the L4"; OPT2-T03 (L4 neutral) was adopted under that reading | State the rule in SPEC. |
| 30 | Decision 65 wording | SPEC decision 65 | "where same-process A/Bs in both orders measured a speed-up", but D1 is process-wide and was measured ABBA in separate processes | Reword. |
| 31 | `docs/rewrites.md` gaps | [rewrites.md](rewrites.md) | No rows for f64-T01/T03/T04 (emulation, float64 tables, INT8-slice float32, paired float32), ideas D1–D5/F0–F5, the opt-5 §3.10 receiver-backward and cuBLASLt A/Bs, round 7 | Add rows or link this ledger. |
| 32 | Round 7 not on main | branch `fastmace/opt7-plan` | Plan and report (LunaTP, atomics, traffic split, wide `float2` loads "eligible, documented, not built") exist only there; the plan says the `w`/`dw` lever is "owned by another session" without naming round 8 | Merge the documents; cross-link [opt-8](../plans/opt-8.md). |
| 33 | `empty_cache` before capture not reconciled | switch-map §2 vs `ResearchModule:LT/studies/optimization_history/report.md` §8 | FastMACE default `empty_cache_before_capture=True`; the RE rejected `empty_cache()` after each capture (no peak reduction); no document says why the library kept a variant | One sentence in the switch map. |
| 34 | ResearchModule page headers stale | `ResearchModule:LT/docs/status.md`, `LT/docs/roadmap.md` | Titled "9 October 2026" but contain material to 15–27 October; roadmap "Decisions" still says keep 0.5 fs (1 fs since 22 October); roadmap item 8 says the hybrid environment "is coraline-only" | Re-date; add the 1 fs decision. |
| 35 | ResearchModule "production remains cuEq float64" statements | `ResearchModule:LT/studies/hybrid_oeq/README.md`, `rtx5060ti.md`, `coraline.md`, `optimization_v2/report.md`, `remaining_optimizations/report.md`, `optimization_history/report.md` | True on 8–9 October, superseded by the 9 October hybrid FP32 decision; only some pages carry a dated preface | Add dated prefaces. |
| 36 | Segmented first sum contradiction | `ResearchModule:LT/docs/roadmap.md` item 12; `LT/studies/further_optimizations/2026-10-14-third-round.md` §3 vs §4b | "Not part of the default" and "Adopted 14 October 2026 (auto)" in the same item | Keep the adoption only. |
| 37 | Fused radial hidden "adopted" wording kept | `ResearchModule:LT/docs/roadmap.md` item 13, `STREAMED_EDGES_RESULTS.md`, `further_optimizations/README.md`, `studies/catalog.md` | Reported adopted on 14 October; it never attached; corrected 15 October next to the old wording | Strike the old wording. |
| 38 | Receiver sorting status drift | `ResearchModule:LT/studies/edge_ordering/README.md`, `LT/docs/status.md` | "Sorting is optional, whole-MD benefit is unqualified" predates the 10 October adoption; status.md lists both rows without a cross-reference | Add a dated note. |
| 39 | `max-autotune` open item | `ResearchModule:LT/docs/roadmap.md` item 6 | Lists a coraline `max-autotune` check as open; the 12 October gear-switch screen found `max-autotune` fails with outer CUDA graphs and rejected `max-autotune-no-cudagraphs` | Close the item. |
| 40 | "Best step" figures unlabelled | ResearchModule roadmap, AGENTS.md, configuration vs `status.md` | ≈ 3.75 / 23.1 / 28.8 ms vs 3.71 / 22.9 / 28.6 ms (the `ir_mul` A/B controls), both local | Label the source run. |
| 41 | ResearchModule screen tables incomplete | `ResearchModule:LT/docs/status.md` "Completed optimization screens"; `LT/studies/catalog.md`; `LT/studies/radial_ceiling/` | status.md table stops at 14 October; catalog.md (last edit 15 October) lacks the 12 October gear-switch runs, the 13 October speculative screen, the 16 October upstream survey, OEq contraction, OMOL optimisation, grouped/merged linears; `radial_ceiling/` has no README | Add rows or point to this ledger. |
| 42 | Mixed metrics side by side | ResearchModule AGENTS.md, `further_optimizations` tables | Force-replay percentages (V2 GEMM-only 4.98–5.72 %, early sorting 1.65–4.45 %, idea 4 buckets) quoted next to MD-step percentages | Label the metric. |
| 43 | Benchmark_MACE README stale | `ResearchModule:Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/README.md` | cuEquivariance benchmark "pending" because of cuEq 0.1.0 / MACE 0.3.14; all stacks moved to MACE 0.3.16 | Mark superseded. |
| 44 | cuEq baselines fused or not | older ResearchModule reports | All `--backend cueq` runs before 16 October were unfused (`cuda:0`), the GPU-benchmark scripts (device `cuda`) fused; reports quoting cuEq baselines do not say which | Label per report where known. |
| 45 | Hardware label (checked, resolved) | [opt-5](validation/opt-5.md) §3.11 | Places an RTX A4000 (sm_86) on coraline; the slice flagged this because AGENTS.md names the A4000 only for trillian. `ResearchModule:LT/studies/hybrid_oeq/coraline.md` records coraline as an RTX A4000, so the label is right | None needed; AGENTS.md could name coraline's GPU. |

## Current state

### Default float32 MH-1 plan

`auto` rules as in [switch-map](design/switch-map.md) §1–§4 and `fastmace/planner/fixed.py` on main `8a269a91`.
"RE default" means the research engine's `main.py` default as listed in SPEC §6.2 ("defaults as in `main.py` on 16
October 2026"), carried over in phase 1a.

RewriteConfig, on in the default MH-1 float32 plan:

| Field | `auto` rule | Made default by |
| --- | --- | --- |
| `factor_species` | `on_for_hybrid_graph_engine_if_model_supports` | RE default (9 Oct 2026, production decision); SPEC §6.2 |
| `fold_cutoff` | `on_with_factor_species_if_model_supports` | RE default (9 Oct 2026); SPEC §6.2 |
| `density_first` | `on_with_fold_cutoff_if_model_supports` | RE default (10 Oct 2026); SPEC §6.2 |
| `reassociate_first` | `on_with_stage1_fold_cutoff_if_model_supports` | RE default (10 Oct 2026); SPEC §6.2 |
| `joint_radial_input` | `on_with_fold_cutoff_if_model_supports` | RE default (14 Oct 2026); SPEC §6.2 |
| `fused_folded_backward` | `on_with_stage1_fold_cutoff_if_model_supports` | RE default (14 Oct 2026); SPEC §6.2 |
| `folded_backward_tiles` | `on_with_fused_folded_backward` | Owner, 21 Oct 2026 (OPT-DEF-2), SPEC §19 thirteenth amendment |
| `segmented_first_sum` | `on_with_reassociate_first_and_sort_receivers` | RE default (14 Oct 2026); SPEC §6.2 |
| `streaming_conv` | `on_for_hybrid_with_stage1_fold_cutoff_and_sort_receivers_if_model_supports` | RE default (14 Oct 2026); SPEC §6.2 |
| `block_linear` | `on_for_hybrid_graph_engine` | RE default (14 Oct 2026); SPEC §6.2 |
| `grouped_linear`, `merged_residual` | `on_with_block_linear` | RE default (16 Oct 2026); SPEC §6.2 |
| `gated_rows` | `on_with_merged_residual` | Owner, 20 Oct 2026 night, §19 twelfth amendment |
| `segment_gemm` | `on_with_gated_rows_float32_unless_reproducible` | Owner, 20 Oct 2026, §19 twelfth amendment |
| `species_tables` | `on_unless_reproducible_if_model_supports` | Owner, 21 Oct 2026 (OPT-DEF-2), thirteenth amendment; float64 by decision 63 |
| `fold_product_linear` | `on_with_species_tables_and_block_linear_if_model_supports` | as `species_tables` |
| `fold_readout_tail` | `on_with_block_linear_unless_reproducible_if_model_supports` | as `species_tables` |
| `chunked_conv_backward` | `on_with_streaming_conv_and_fused_folded_backward` | Decision 60 (27 Oct 2026; §19 "Optimisation round 6") |
| `radial_tables` | `on_with_fold_cutoff_unless_reproducible` | Decision 62 (27 Oct 2026); class `certified` by decision 61; float64 by decision 63 |
| `fused_folded_forward` | `on_for_mh1_float32_unless_reproducible` | Decision 64 (27 Oct 2026) |
| `single_graph_strain` | `on` (acts only where stress is computed) | RE default (16 Oct 2026); SPEC §6.2 |

RewriteConfig, off or not applicable in the MH-1 float32 plan: `species_skip` (generic family only,
`on_with_graph_engine_if_model_supports`, RE 15 Oct 2026); `grouped_launch` (generic family only, decision 57);
`fx_e3nn_linear`, `polar_graph_sums` (POLAR-1 only, decision 58); `polar_uvu` (`True`, acts on POLAR models only, RE
16 Oct 2026); `fused_radial_hidden`, `direct_node_feats`, `folded_constants`, `combo_kernels`, `tf32x3_gemms`
(`never_automatic`); `inplace_block_linear` (`False`, refused if set); `w_tables`, `save_transcendentals` (float64
only, decision 65); `selfcheck_cache` (`"use"`, inert, not built).

EngineConfig: `backend` hybrid (`backend_hybrid_for_graph_engine`; RE production decision 9 Oct 2026);
`compile_mode` `"default"` (RE, 8–9 Oct 2026, no autotuning); `pipelined` (`on_with_graph_engine_unless_speculative`,
RE 14 Oct); `async_output` (`on_with_graph_engine`, RE 15 Oct); `speculative_graph` `False` (OPT-IN);
`first_derivative_only` `True`, `unfuse_full_addmm` `True`, `expandable_segments` (`on_for_torch_2_14_unless_alloc_conf_set`)
(RE 7–8 Oct); `dynamic_edges` `True` (decision 59); `model_template_cache` `True` (conductor on merge, twelfth
amendment mentions it); `overlap_rebuild` `True` (OPT-DEF-2, thirteenth amendment); `step_graph` `False` (OPT-IN);
`fused_integrator` `True` (owner 21 Oct 2026, fourteenth amendment); `cueq_conv_fusion` (plain cuEq only,
`on_for_plain_cueq_unless_reproducible`, RE 16 Oct); `recapture_budget` `"auto"`, `graph_variant_cap` 64,
`dynamo_cache_size_limit` 64, `empty_cache_before_capture` `True`, `warmup_in_graph_pool` `True` (constants).
The CUDA-graph tier is the default tier (rule 9 as amended by decision 38); `AdaptationConfig.adapt` `"auto"`
(decisions 52–53); `CalibrationConfig.calibrate` `False` (decision 55); `RunnerConfig.persistent` `True` (POSIX,
conductor decision 20 Oct 2026).

PrecisionConfig: `dtype` float32 (`float32_for_hybrid_float64_for_cueq`); `tf32` `False` (never default, SPEC §1.2);
`float64_energy` `True` (1b-T01/T03).

NeighbourConfig: `skin` `"auto"` (0.3 Å for MD workloads; rule 11 for MC, species-change and displacement workloads,
3-T05); `sort_receivers` (`on_for_hybrid_graph_engine`, RE 10 Oct); `gpu_neighbour_list` (`on_with_graph_engine`,
RE 14 Oct); `edge_multiple` 256, `edge_bucket_fraction` 0.05, `capacity_headroom` 1.15, `headroom` `"auto"`,
`d3_skin` `"auto"` (rule 11, 3b-T02). `IntegratorConfig.noise_batch` 256; `fixcm_projection` `mass_weighted` under
NVT fixcm (decision 37, correctness).

### Default float64 MH-1 plan: extra switches

| Field / part | Rule | Made default by |
| --- | --- | --- |
| Hybrid backend for non-reproducible float64 plans | planner backend choice | Decision 63(a) |
| `fp64_emulation` (cuBLAS INT8 emulation, eager strategy) | `on_for_float64_cuda_with_reduced_rate_fp64_unless_reproducible` (measured FP64 rate < 1/8) | Decision 63 |
| `species_tables`, `fold_product_linear`, `fold_readout_tail` in float64 | rules without `float32` | Decision 63(c) |
| `radial_tables` in float64 (degree 16, tolerance 3e-13) | `on_with_fold_cutoff_unless_reproducible` | Decision 63(d) |
| `chunked_conv_backward` float64 chunk rule (4 L2s) | library rule inside `l2_chunk_edges` | f64-T03 (`79fb1115`), no decision number |
| `fp64_emulation_mantissa_bits` 56 | `56_for_mh1_float64_on_measured_gpus_decision_65_else_dynamic` (RTX 5060 Ti, RTX 4000 Ada by exact name) | Decision 65(a); suspended on branch `fastmace/fp64-d1-relax-fix` (auto = dynamic everywhere, owner 28 Oct 2026) |
| `w_tables` | `on_for_mh1_float64_on_measured_gpus_decision_65` | Decision 65(b) |
| `save_transcendentals` | `on_for_mh1_float64_on_measured_gpus_decision_65` | Decision 65(c) |
| Not in float64: `segment_gemm`, `grouped_launch`, `fused_folded_forward` (Triton/NVRTC float32 only) | — | — |

### Current speed reference

MD step time in ms, NVT 300 K unless stated. Rows from different days or processes are not A/B comparisons.

| Date | Engine and plan | GPU | NaCl | NaNbO3 | NbCl5 | Commit | Source |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 2026-10-07 | RE: ASE path, FP32 | local | 41.4 | – | – | – | `ResearchModule:LT/studies/optimization_history/report.md` |
| 2026-10-08 | RE: cuEq FP32 / hybrid FP32 | local | 9.668 / 7.075 | 84.587 / 55.920 | 72.286 / 55.650 | – | `ResearchModule:LT/studies/hybrid_oeq/rtx5060ti.md` |
| 2026-10-14 | RE: best default after streaming conv + `ir_mul` | local | ≈ 3.75 (3.71) | ≈ 23.1 (22.9) | ≈ 28.8 (28.6) | – | `ResearchModule:LT/docs/roadmap.md` item 13 / `status.md` |
| 2026-10-16 | RE: + grouped linear + merged residual | murphy | 3.55 | 25.74 | 30.54 | – | `ResearchModule:LT/studies/other_models/README.md` |
| 2026-10-20 | FM before round 1 (2-T20a) | local / murphy | 3.71 / 3.68 | – | 27.43 / 31.58 | not recorded | [2-T20a](validation/2-T20a-step-breakdown.md), [opt-1 plan](../plans/opt-1.md) |
| 2026-10-21 | FM after round 2 (OPT-T10, 1000 K, wall) | local / L4 | 3.028 / 4.264 | 18.788 / 29.589 | 21.434 / 33.126 | `072fa0a` | [OPT-T10](design/opt-T10-step-breakdown.md) |
| 2026-10-25 | FM 0.x default, median of 4 blocks | L4 (Colab) | 3.95 | 31.5 | 35.2 | `c1f4de9c` | [benchmarks](benchmarks.md) |
| 2026-10-27 | FM default + `radial_tables` (before `fused_folded_forward`) | leto | 2.80 | 18.03 | 21.75 | `0cba0d7a` | [opt6-T05](validation/opt6-T05.md) |
| 2026-10-27 | FM default + `fused_folded_forward` (today's float32 default), side b, both orders | local | 2.654 / 2.709 | 14.56 / 14.58 | 18.67 / 18.62 | `9eb1ab1e` (pinned) | [opt8-T02](validation/opt8-T02.md) |
| 2026-10-27 | same | murphy | 2.811 / 2.809 | 17.37 / 17.50 | 21.54 / 21.62 | `059b2cfa` | [opt8-T02](validation/opt8-T02.md) |
| 2026-10-27 | same, order a-b only | L4 (Colab) | – | 26.06 | 31.47 | `9eb1ab1e` | [opt8-T02](validation/opt8-T02.md) |
| 2026-10-27 | FM float32 default, step breakdown at 1000 K | local | 2.78 | 15.31 | 19.21 | branch `opt7-plan` | branch fastmace/opt7-plan: plans/opt-7.md §1 |
| 2026-10-27 | FM float64: stock → decision-63 package | local | 81.9 → 46.9 | 516 → 255 | 652 → 275 | `7562a425` | [f64-T01](validation/f64-T01.md) |
| 2026-10-27 | FM float64: stock → package | leto | 76.3 → 44.8 | – | 579 → 233 | not recorded | [f64-T01](validation/f64-T01.md) |
| 2026-10-28 | FM float64: package + D1 + D2 + D3, ABBA ratio | local / Ada | 0.806 / 0.804 | 0.782 / – | 0.797 / 0.802 | not recorded | [ideas](../plans/ideas-20261027.md) |
| 2026-10-26 | FM 1.0.0, OMAT-0 medium float32 | local | 1.651 / 1.652 | – | 9.901 / 9.921 | 1.0.0 | [ALCHEMI comparison](validation/alchemi-comparison-20261025.md) |

A100 float32 numbers for the current default are not recorded (see "Open or untried").

## Index by area

Each ID links to its appendix entry. RE-only evidence is given as `ResearchModule:` paths; `LT/` is
`Experiments/Langevin_Thermostat/`.

### Atom-level linears and GEMMs

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [lin-block-linear](#lin-block-linear) | cuEq `Linear` as per-l block GEMMs | `block_linear` | both | DEFAULT | RE 2026-10-14; SPEC §6.2 | RE NaCl −11.8 %, NaNbO3 −1.5 %, NbCl5 −0.9 % (local) | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [lin-grouped-linear](#lin-grouped-linear) | Same-irrep input segments in one product | `grouped_linear` | both | DEFAULT | RE 2026-10-16 (user) | with merged residual: murphy NaCl 3.68 → 3.55 ms, NbCl5 31.91 → 30.54 ms | `ResearchModule:LT/studies/other_models/README.md` |
| [lin-merged-residual](#lin-merged-residual) | `linear_1` and `linear_res` as one product | `merged_residual` | both | DEFAULT | RE 2026-10-16 | included in grouped linear's −2 to −5 % | SPEC §6.2 |
| [lin-unfuse-addmm](#lin-unfuse-addmm) | Keep Inductor from fusing mm + full add into addmm | `unfuse_full_addmm` | both | DEFAULT | RE 2026-10-08 | coraline FP32 11.00/11.08 vs 11.37/11.41 ms; local −3.7 % | `ResearchModule:LT/docs/configuration/workstations.md` |
| [lin-inplace-block-linear](#lin-inplace-block-linear) | In-place / strided-batched block linear | `inplace_block_linear` (refused) | both | REJECTED | RE 2026-10-14/15 | +0.10 / +0.30 / +0.24 ms; batched 1,192 vs 992 µs | `ResearchModule:LT/studies/further_optimizations/2026-10-15-fourth-round.md` |
| [lin-grouped-node-gemms](#lin-grouped-node-gemms) | Grouped node GEMMs (RE) | none | RE | NOT BUILT | 2026-10-14 | ceiling ≈ 0.10 ms NaCl | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [lin-cueq-fused-tp](#lin-cueq-fused-tp) | cuEq `fused_tp` for linears | probe | RE | REJECTED | 2026-10-07/08 | 24.6 → 24.4 ms per force call | `ResearchModule:LT/docs/configuration/local.md` |
| [lin-magma](#lin-magma) | MAGMA GEMM diagnosis | none | RE | MEASUREMENT | 2026-10-10 | node-level linears, inherent FP32 compute | `ResearchModule:LT/studies/further_optimizations/README.md` |
| [lin-gemm-autotune](#lin-gemm-autotune) | GEMM-only and broad Inductor autotuning | `--gemm-autotune`, `max_autotune_gemm` | RE | REJECTED | 2026-10-09, 2026-10-10 | V2 −4.98 to −5.72 %; on hybrid default −0.1 to −1.0 %, setup 140.7–756.3 s | `ResearchModule:LT/studies/optimization_v2/report.md` |
| [lin-gated-rows](#lin-gated-rows) | Residual products, gate, `linear_2` as per-irrep rows | `gated_rows` | FM | DEFAULT (scoped: MH-1 class) | owner 2026-10-20, §19 12th | murphy NaCl 0.957 / 0.954, NbCl5 0.909 / 0.911 | [opt-T02a](validation/opt-T02a-gated-rows.md) |
| [lin-segment-gemm](#lin-segment-gemm) | Gathered-row Triton GEMM into the joined output | `segment_gemm` | FM | DEFAULT (scoped: MH-1 class, float32, not reproducible) | owner 2026-10-20, §19 12th | murphy NbCl5 0.9517 / 0.9479, local 0.9577 / 0.9567 | [opt-T02b](validation/opt-T02b-segment-gemm.md) |
| [lin-t02-combined](#lin-t02-combined) | gated_rows + segment_gemm vs fastmace-1b | none | FM | MEASUREMENT | 2026-10-21 | local NbCl5 −14.6 to −14.9 %, murphy −13.5 to −13.6 % | [opt-1 plan](../plans/opt-1.md) |
| [lin-fusion-mechanisms](#lin-fusion-mechanisms) | Inductor templates, cuBLASLt epilogues, CUTLASS EVT, cuBLASDx | none | FM | NOT BUILT | 2026-10-20 | not measured | branch fastmace/opt-T02: docs/design/opt-T02-fusion.md |
| [lin-concurrent-branches](#lin-concurrent-branches) | Per-l products on concurrent graph branches | none | FM | REJECTED | 2026-10-20 | murphy 0.85–1.11×; local 0.68–0.98× | branch fastmace/opt-T03: docs/design/opt-T03-small-m.md |
| [lin-grouped-launch-v1](#lin-grouped-launch-v1) | One Triton launch for small l-products, first version | `grouped_launch` (`cd71b5c`) | FM | SUPERSEDED | 2026-10-20/21 | full step local NaCl 1.0179 / 1.0103; murphy 1.0697 | [opt-T03b diagnosis](design/opt-T03b-diagnosis.md) |
| [lin-grouped-launch](#lin-grouped-launch) | Grouped launch, equal (K, N) only | `grouped_launch` | FM | DEFAULT (scoped: generic MACE family, float32, not reproducible) | decision 57, 2026-10-24 | generic NaCl 0.946–0.955; MH-1 sm_120 0.977 / 0.982, sm_89 1.005 / 1.003 | [opt4-T04](validation/opt4-T04.md) |
| [lin-split-k](#lin-split-k) | Split-K for small-cell products | `split_k` (branch) | FM | REJECTED | 2026-10-21 | local NaCl 1.0028 / 1.0049 | branch fastmace/opt2-T05: docs/validation/opt2-T05-split-k.md |
| [lin-split-k-triton](#lin-split-k-triton) | Triton split-K with epilogue reduction | none | FM | NOT BUILT | 2026-10-21 | ceiling ≈ 0.3 % NaCl | branch fastmace/opt2-T05 |
| [lin-checkpoint-scan](#lin-checkpoint-scan) | Scan checkpoints for exact zero skips | none | FM | NOT BUILT | 2026-10-21 | < 0.5 % together | [opt-T04](validation/opt-T04-checkpoint-scan.md) |
| [lin-zero-channels](#lin-zero-channels) | Drop MH-1's 24 zero l = 1 channels | none | FM | NOT BUILT | 2026-10-21 | < 0.1 ms NbCl5 (estimate) | [opt-T04](validation/opt-T04-checkpoint-scan.md) |
| [lin-unused-heads](#lin-unused-heads) | Single-head model / skip unused readout heads | `select_head.py` (RE probe) | both | REJECTED | 2026-10-08; 2026-10-21; 2026-10-24 | 9.89 vs 9.93 ms; ≈ 0.05 ms NbCl5 | [opt-T04](validation/opt-T04-checkpoint-scan.md) |
| [lin-species-tables](#lin-species-tables) | Species-only atom products as gathered tables | `species_tables` | FM | DEFAULT (scoped: MH-1 labels, not reproducible) | owner 2026-10-21 (OPT-DEF-2); decision 63 | local NaCl 0.956, NbCl5 0.979 | [opt-T08](validation/opt-T08-linear.md) |
| [lin-fold-product-linear](#lin-fold-product-linear) | Fold `products.0.linear` into its consumers | `fold_product_linear` | FM | DEFAULT (scoped: MH-1 labels, not reproducible) | OPT-DEF-2; decision 63 | on top of C1: NaCl −3.4 %, NbCl5 −4.4 % | [opt-T08](validation/opt-T08-linear.md) |
| [lin-fold-readout-tail](#lin-fold-readout-tail) | Fold the last linear tail into the readout | `fold_readout_tail` | FM | DEFAULT (scoped: MH-1 labels and OMOL-0, not reproducible) | OPT-DEF-2; decision 63 | NaCl 0.977, NbCl5 0.991 | [opt-T08](validation/opt-T08-linear.md) |
| [lin-opt-t08-package](#lin-opt-t08-package) | C1 + C2 + C3 together (OPT-DEF-2) | the three fields | FM | DEFAULT | owner 2026-10-21, §19 13th | local NaCl 0.908 / 0.904, NbCl5 0.924 / 0.928; L4, A100 similar | [opt-def-2](validation/opt-def-2.md) |
| [lin-t08-c6](#lin-t08-c6) | No gradient w.r.t. `node_attrs` | none | FM | NOT BUILT | 2026-10-21 | absorbed by C1 | [OPT-T08 audit](design/opt-T08-linear-audit.md) |
| [lin-t08-c4](#lin-t08-c4) | One product for `skip_tp` and `linear_up` | none | FM | NOT BUILT | 2026-10-21 | absorbed by C2, C3 | [OPT-T08 audit](design/opt-T08-linear-audit.md) |
| [lin-t08-c5](#lin-t08-c5) | Drop the `node_feats` descriptor concatenation | none | FM | NOT BUILT | 2026-10-21 | absorbed by C3 | [OPT-T08 audit](design/opt-T08-linear-audit.md) |
| [lin-t08-c7](#lin-t08-c7) | Drop unused heads of `readouts.0.linear` | none | FM | NOT BUILT | 2026-10-21 | absorbed | [OPT-T08 audit](design/opt-T08-linear-audit.md) |
| [lin-t08-headslice](#lin-t08-headslice) | C2 with a one-head slice | none | FM | NOT BUILT | 2026-10-21 | ≈ 10 µs NaCl, 40 µs NbCl5 | [OPT-T08 audit](design/opt-T08-linear-audit.md) |
| [lin-t08-generic](#lin-t08-generic) | C1/C2 for residual, density, OFF families | none | FM | NOT BUILT | 2026-10-21 | not measured | [OPT-T08 audit](design/opt-T08-linear-audit.md) |
| [lin-t08-paper](#lin-t08-paper) | Other linear folds checked on paper | none | FM | NOT BUILT | 2026-10-21 | not measured | [OPT-T08 audit](design/opt-T08-linear-audit.md) |
| [lin-forward-audit](#lin-forward-audit) | Further consecutive-linear folds (round 5) | none | FM | NOT BUILT | 2026-10-26 | none found | [opt-5](validation/opt-5.md) |
| [lin-faster-fp32](#lin-faster-fp32) | A 1.2× faster FP32 SIMT GEMM | none | FM | NOT BUILT | 2026-10-21 | ceiling ≈ 5 / 8 / 4 % | [OPT-T09](design/opt-T09-step-breakdown.md) |
| [lin-layout-copies](#lin-layout-copies) | Remove block-linear row gathers, joins, fallback copy | none | FM | NOT BUILT | 2026-10-21 | ≈ 20–50 µs each | [opt3-T05 design](design/opt3-T05-layout-audit.md) |
| [lin-cublaslt](#lin-cublaslt) | cuBLASLt as preferred BLAS | `TORCH_BLAS_PREFER_CUBLASLT`, `PREFERRED_BLAS` | both | REJECTED | 2026-10-08, 2026-10-14, 2026-10-26 | full step NaCl −1.2 %, NaNbO3 +1.0 % | [opt-5](validation/opt-5.md) §3.10 |
| [lin-cublaslt-heuristic](#lin-cublaslt-heuristic) | Per-shape cuBLASLt algorithm search | none | FM | REJECTED | 2026-10-26 | 505.23 vs 427.02 µs, unqualified, shape-keyed | [opt-5](validation/opt-5.md) §3.9 |
| [lin-svd](#lin-svd) | Low-rank atom linears (truncated SVD) | none | FM | REJECTED | 2026-10-27 | rank 509–512 of 512 | [ideas](../plans/ideas-20261027.md) |

### Atom-level elementwise and gates

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [elt-density-first](#elt-density-first) | Density division before `linear_1` | `density_first` | both | DEFAULT | RE 2026-10-10 | NaCl −5.7 %, NaNbO3 −4.5 %, NbCl5 −6.8 % | `ResearchModule:LT/studies/further_optimizations/README.md` §6 |
| [elt-gate-vjp](#elt-gate-vjp) | Fused gate/residual VJP | none | both | NOT BUILT | RE 2026-10-10/14; OPT3-T01 2026-10-21 | removable 0.13–0.25 ms NbCl5 (0.6–1.1 %) | [opt3-T01](design/opt3-T01-gate-vjp-analysis.md) |
| [elt-gate-prologue](#elt-gate-prologue) | Gate inside `segment_gemm`'s prologue (stage C) | `gate_prologue` (branch) | FM | REJECTED | 2026-10-20 | local NbCl5 1.0196 / 1.0157 | [stage C note](design/opt-T02-C-gate-prologue.md) |
| [elt-gate-backward-epilogue](#elt-gate-backward-epilogue) | Gate backward in the input-gradient epilogue (stage D) | none | FM | REJECTED | 2026-10-20 | not measured (est. 1.0–1.4 ms NbCl5) | [rewrites](rewrites.md) |
| [elt-density-prologue](#elt-density-prologue) | Density division in the residual GEMM prologue (stage E) | none | FM | NOT BUILT | 2026-10-20 | est. 0.34 ms NbCl5 | [BC go/no-go](design/opt-T02-BC-go-no-go.md) |
| [elt-gated-segment-gemm](#elt-gated-segment-gemm) | Gate scaling in the `segment_gemm` prologue (round 5) | none | FM | CUT | 2026-10-26 | NbCl5 forward replay 0.9626× (< 5 % bar) | [opt-5](validation/opt-5.md) |
| [elt-direct-node-feats](#elt-direct-node-feats) | Skip the LAMMPS `[node_feats, node_attrs]` join | `direct_node_feats` | FM | NEVER-AUTOMATIC | 2026-10-21 | local NaCl 0.9921 / 0.9948, NbCl5 1.0020 / 1.0020 | [opt3-T05](validation/opt3-T05-layout-audit.md) |

### Radial networks and tables

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [rad-factor-species](#rad-factor-species) | Species-factored radial input | `factor_species` | both | DEFAULT | RE 2026-10-09 | NaCl −18.4 %, NaNbO3 −19.3 %, NbCl5 −14.3 % | `ResearchModule:LT/studies/species_factoring/README.md` |
| [rad-fold-cutoff](#rad-fold-cutoff) | Cutoff folded into the last radial layer | `fold_cutoff` | both | DEFAULT | RE 2026-10-09 | NaCl −11.1 %, NaNbO3 −17.3 %, NbCl5 −14.9 % | `ResearchModule:LT/studies/species_factoring/README.md` |
| [rad-joint-radial-input](#rad-joint-radial-input) | One joint radial input | `joint_radial_input` | both | DEFAULT | RE 2026-10-14 | NaCl −3.5 %, NaNbO3 −7.4 %, NbCl5 −4.7 % | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [rad-fused-folded-backward](#rad-fused-folded-backward) | Triton input gradient of the folded layer (+ generic) | `fused_folded_backward` | both | DEFAULT | RE 2026-10-14; generic 2026-10-15 | MH-1 −0.6 to −1.4 %; generic −3 to −5 % large cells | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [rad-folded-backward-tiles](#rad-folded-backward-tiles) | 128-row tiles for the folded input gradient | `folded_backward_tiles` | FM | DEFAULT | OPT-DEF-2, 2026-10-21 | local NaCl 0.983 / 0.981; L4 neutral | [opt2-T03](validation/opt2-T03.md) |
| [rad-fused-radial-hidden](#rad-fused-radial-hidden) | Triton fused radial hidden block | `fused_radial_hidden` | both | NEVER-AUTOMATIC | RE 2026-10-14, corrected 2026-10-15 | −0.06 / 0 / +0.15 ms | `ResearchModule:LT/studies/further_optimizations/2026-10-15-step-breakdown.md` |
| [rad-tabulation-ceiling](#rad-tabulation-ceiling) | Radial-network tabulation (RE ceiling) | probe | RE | SUPERSEDED | RE 2026-10-10 | ceiling 4.4 / 5.1 / 2.9 %; reopened as `radial_tables` | `ResearchModule:LT/studies/further_optimizations/README.md` |
| [rad-weight-sharing](#rad-weight-sharing) | Same-species radial weight sharing | probe | RE | NOT BUILT | 2026-10-10 | ceiling 4.3 / 5.5 / 4.2 % | `ResearchModule:LT/studies/further_optimizations/README.md` |
| [rad-mirror-edge](#rad-mirror-edge) | Mirror-edge radial sharing | none | RE | NOT BUILT | 2026-10-15 | ceiling already +1.1 ms NaNbO3 | `ResearchModule:LT/studies/further_optimizations/2026-10-15-fourth-round.md` |
| [rad-contiguous-folded-weight](#rad-contiguous-folded-weight) | Contiguous folded weight | `CONTIGUOUS_FOLDED_WEIGHT` | RE | REJECTED | 2026-10-14 | full step within noise twice | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [rad-decompose-mm](#rad-decompose-mm) | Inductor `decompose_mm_pass` | `DECOMPOSE_SMALL_MM` | RE | REJECTED | 2026-10-14 | no gain | `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` |
| [rad-joint-fused-backward](#rad-joint-fused-backward) | Triton input gradient of the joint layer | `JOINT_FUSED_BACKWARD` | RE | REJECTED | 2026-10-14 | NaCl −0.03 ms; slower on small cells | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [rad-triton-folded-forward](#rad-triton-folded-forward) | Triton forward for the folded layer (RE) | probe | RE | REJECTED | 2026-10-14 | 2.0–64 ms vs 1.9 ms cuBLAS | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [rad-cuda-gemm](#rad-cuda-gemm) | Hand-written CUDA GEMM for the folded forward | probe | RE | REJECTED | 2026-10-14 | 2.78 vs 1.88 / 1.57 ms | `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` |
| [rad-fused-radial-mlp](#rad-fused-radial-mlp) | Fused radial MLP, store variant | `fused_radial_mlp` (branch) | FM | REJECTED | 2026-10-21 | local NaCl 0.985 / 0.985, NbCl5 1.011 / 1.009 | branch fastmace/opt2-t02c: docs/design/opt2-T02-radial-mlp-fusion.md |
| [rad-fused-radial-mlp-recompute](#rad-fused-radial-mlp-recompute) | Fused radial MLP, recompute variant | `fused_radial_mlp` (first) | FM | SUPERSEDED | 2026-10-21 | stopped at microbench (spills) | branch fastmace/opt2-t02c |
| [rad-radial-entry](#rad-radial-entry) | Radial MLP entry kernels (OPT2-T02c) | `fused_radial_mlp` + joint input (branch) | FM | REJECTED | 2026-10-21 | NbCl5 1.0008 / 1.0060 | branch fastmace/opt2-t02c |
| [rad-fused-mlp-retry](#rad-fused-mlp-retry) | Second radial-MLP fusion attempt | none | FM | NOT BUILT | 2026-10-24 | not measured | [opt-4](../plans/opt-4.md) §3 |
| [rad-folded-constants](#rad-folded-constants) | Folded `[W, b]` as buffers | `folded_constants` | FM | NEVER-AUTOMATIC | 2026-10-21 | measured only with `direct_node_feats` | [opt3-T05](validation/opt3-T05-layout-audit.md) |
| [rad-write-bound](#rad-write-bound) | Write-bound Triton kernel for the folded forward | none | FM | CUT | 2026-10-24 | best Triton 199 / 1,331 / 1,969 µs vs limit 150 / 1,000 / 1,450 | [opt4-T01](validation/opt4-T01.md) |
| [rad-k64-addr](#rad-k64-addr) | cuBLAS K = 64 plus rank-1 update | none | FM | REJECTED | 2026-10-24 | ≈ 2× slower | [opt4-T01](validation/opt4-T01.md) |
| [rad-radial-last-gemm](#rad-radial-last-gemm) | `64 + tail` streaming-store kernel | planned `radial_last_gemm` | FM | CUT | 2026-10-26 | 160.7 / 1,440.3 / 2,435.8 µs vs targets | [opt-5](validation/opt-5.md) |
| [rad-stub-ceiling](#rad-stub-ceiling) | Radial-table accuracy premise and stub ceiling | none | FM | MEASUREMENT | 2026-10-27 | stub b/a NaCl 0.897, NbCl5 0.928, NaNbO3 0.839 | [opt6-T04](validation/opt6-T04.md) |
| [rad-radial-tables](#rad-radial-tables) | Certified Chebyshev radial tables | `radial_tables` | FM | DEFAULT (scoped: `fold_cutoff`, not reproducible) | decisions 61, 62, 2026-10-27 | local NaCl 0.901 / 0.898, NbCl5 0.915 / 0.916, NaNbO3 0.811 / 0.807 | [opt6-T05](validation/opt6-T05.md) |
| [rad-values-only-fit](#rad-values-only-fit) | Values-only table fit | none | FM | REJECTED | 2026-10-27 | certificate 1.03–1.07 | [opt6-T05](validation/opt6-T05.md) |
| [rad-tables-omol](#rad-tables-omol) | `radial_tables` self-check failure on OMOL-0 XL | `radial_tables` | FM | PENDING | 2026-10-27 | errors 1.4–2.2e-4 > 1e-4 | [opt8-T02](validation/opt8-T02.md) |
| [rad-f64-tables](#rad-f64-tables) | Float64 radial tables | `radial_tables` (float64) | FM | DEFAULT (scoped: float64, not reproducible) | decision 63(d) | step 0.90 NaCl, 0.91 NbCl5 | [f64-T01](validation/f64-T01.md) §4 |
| [rad-f64-r03](#rad-f64-r03) | 0.3 Å lower bound in float64 tables (bug) | `R_LO_64` | FM | SUPERSEDED | 2026-10-27 | certificate 3e10 on CUDA | [f64-T01](validation/f64-T01.md) §4 |
| [rad-table-cache](#rad-table-cache) | Cache fitted, certified tables under `FASTMACE_CACHE` | none | FM | ON BRANCH | 2026-10-28 | fit + certify 42 of 87 s of a float64 build (trillian) | branch fastmace/radial-table-cache: commit `54b6f13f` |
| [rad-w-tables](#rad-w-tables) | Per-piece folded `w` tables (D2) | `w_tables` | FM | DEFAULT (scoped: float64 MH-1, RTX 5060 Ti and RTX 4000 Ada) | decision 65(b) | local NbCl5 0.911 / 0.912, NaNbO3 0.855 / 0.855; A100 slower | [ideas](../plans/ideas-20261027.md) |
| [rad-w-tables-auto](#rad-w-tables-auto) | D2/D3 by measured FP64 rate instead of GPU names | none | FM | NOT BUILT | 2026-10-28 | not measured | [ideas](../plans/ideas-20261027.md) |
| [rad-w-tables-r1](#rad-w-tables-r1) | Keep only w-table pieces with r ≥ 1 Å | none | FM | NOT BUILT | 2026-10-28 | not measured | [ideas](../plans/ideas-20261027.md) |
| [rad-low-rank](#rad-low-rank) | Certified low-rank radial / w basis (D4, F1) | none | FM | REJECTED | 2026-10-27 | rank 65 of 65 | [ideas](../plans/ideas-20261027.md) |
| [rad-f4](#rad-f4) | Piecewise w-tables in float32 (F4) | none | FM | SUPERSEDED | 2026-10-27 | not measured | [ideas](../plans/ideas-20261027.md) |
| [rad-f0](#rad-f0) | Float32 input-gradient tiles in the chunked backward (F0) | none | FM | REJECTED | 2026-10-27 | stub ceiling NbCl5 0.5 %, NaNbO3 2.6 % | [ideas](../plans/ideas-20261027.md) |
| [rad-f64-reshape](#rad-f64-reshape) | Reshaped folded forward product in float64 | none | FM | REJECTED | 2026-10-27 | 24.6–36.0 vs 25.1 ms | [f64-T01](validation/f64-T01.md) |

### Interaction-2 convolution (streaming, OEq)

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [conv-sort-receivers](#conv-sort-receivers) | Receiver-sorted edges | `sort_receivers` | both | DEFAULT | RE 2026-10-10 | NaNbO3 −3.3 %, NbCl5 −4.9 %, NaCl +0.4 % | `ResearchModule:LT/studies/further_optimizations/README.md` §1 |
| [conv-segmented-reduction](#conv-segmented-reduction) | Segmented reduction operator (idea 5) | probe | RE | NOT BUILT | 2026-10-09 | sorted 1.24–3.37 %; with permutation −80 % | `ResearchModule:LT/studies/edge_ordering/README.md` |
| [conv-oeq-schedules](#conv-oeq-schedules) | OEq launch schedules warp4/warp5/blocks2 | probe | RE | REJECTED | 2026-10-09 | −2.790 to +0.052 % | `ResearchModule:LT/studies/optimization_v2/report.md` |
| [conv-shared-weight-ceiling](#conv-shared-weight-ceiling) | Streamed-edges shared-weight ceilings | probe | RE | MEASUREMENT | 2026-10-14 | 11.4 / 16.0 / 9.1 % | `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` |
| [conv-l2-chunks-re](#conv-l2-chunks-re) | L2-resident chunks with unpatched OEq | probe | RE | REJECTED | 2026-10-14 | NaNbO3 9.79 → 13.97–18.58 ms | `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` |
| [conv-own-triton-conv](#conv-own-triton-conv) | Own fused Triton convolution | probe | RE | REJECTED | 2026-10-14 | NaNbO3 12.12 → 16.63 ms fwd+bwd | `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` |
| [conv-oeq-weight-prologue](#conv-oeq-weight-prologue) | OEq weight prologue (per-edge weight generation) | probe | RE | NOT BUILT | 2026-10-14 | 2.90–4.16 vs 1.51 ms | `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` |
| [conv-streaming-conv](#conv-streaming-conv) | Receiver-streaming OEq convolution (PR 223) (+ generic) | `streaming_conv` | both | DEFAULT | RE 2026-10-14; generic 2026-10-15 | NaCl −3.1 %, NaNbO3 −7.7 %, NbCl5 −5.7 %; generic −16 to −28 % | `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` |
| [conv-ir-mul](#conv-ir-mul) | `ir_mul` layout in streaming kernels | none | both | DEFAULT | RE 2026-10-14 | NaCl −2.0 %, NaNbO3 −1.4 %, NbCl5 −1.7 % | `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` |
| [conv-mul-ir](#conv-mul-ir) | Old `mul_ir` layout with transposes | `STREAMING_MUL_IR` | RE | SUPERSEDED | 2026-10-14 | not recorded | [switch-map](design/switch-map.md) §7 |
| [conv-bwd-variants](#conv-bwd-variants) | Streaming backward `__restrict__`, block sizes, register caps | probe | RE | REJECTED | 2026-10-14 | 1.7–5.6× slower | `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` |
| [conv-receiver-backward](#conv-receiver-backward) | Receiver-owned streaming backward | `STREAMING_RECEIVER_BACKWARD` | both | REJECTED | RE 2026-10-14; FM 2026-10-26 | NaNbO3 +9.4 %, NaCl +7.2 %, NbCl5 +4.5 % | [opt-5](validation/opt-5.md) §3.10 |
| [conv-cueq-conv-fusion](#conv-cueq-conv-fusion) | MACE PR 1641 fused cuEq convolution | `cueq_conv_fusion` | both | DEFAULT (scoped: plain cuEq backend, not reproducible) | RE 2026-10-16 | float64 −2 to −6 %, float32 −29 to −40 % | `ResearchModule:LT/studies/further_optimizations/2026-10-16-upstream-mace-survey.md` |
| [conv-fused-radial-conv](#conv-fused-radial-conv) | Radial last layer fused into streaming conv (OPT2-T01) | none (branch) | FM | REJECTED | 2026-10-21 | NaCl −9.6 %, NbCl5 +4.5 %, NaNbO3 −2.0 % (gate 40 %) | branch fastmace/opt2-T01: docs/design/opt2-T01-fused-radial-conv.md |
| [conv-dw-in-conv](#conv-dw-in-conv) | Backward-only `dw` contraction inside the conv | none | FM | NOT BUILT | 2026-10-21 | not measured | [OPT-T10](design/opt-T10-step-breakdown.md) |
| [conv-oeq-launch-shape](#conv-oeq-launch-shape) | Streaming forward launch shape | none | FM | CUT | 2026-10-24 | 93.85 % DRAM; ceiling 0.26 % | [opt4-T06](validation/opt4-T06.md) |
| [conv-l2-hints](#conv-l2-hints) | L2 eviction hints in the streaming forward | none | FM | NOT BUILT | 2026-10-24 | ≤ 0.26 % | [opt4-T06](validation/opt4-T06.md) |
| [conv-radial-conv-reopen](#conv-radial-conv-reopen) | Radial producer / conv fusion via cuBLASDx | none | FM | NOT BUILT | 2026-10-26 | not measured | [opt-5](validation/opt-5.md) §3.12 |
| [conv-opt6-t01](#conv-opt6-t01) | L2-resident edge chunks falsifier | none | FM | MEASUREMENT | 2026-10-26 | NbCl5 −1.06 ms, NaNbO3 −0.93 ms subgraph | [opt6-T01](validation/opt6-T01.md) |
| [conv-chunked-recompute](#conv-chunked-recompute) | Chunked forward with recompute | none | FM | REJECTED | 2026-10-26 | slower everywhere | [opt6-T01](validation/opt6-T01.md) |
| [conv-chunked-conv-backward](#conv-chunked-conv-backward) | Chunked conv backward, `dw` in L2 | `chunked_conv_backward` | FM | DEFAULT | decision 60, 2026-10-27 | NbCl5 0.9725 / 0.9669, NaNbO3 0.9694 / 0.9642, NaCl 1.0077 / 1.0087 | [opt6-T02](validation/opt6-T02.md) |
| [conv-chunked-nacl](#conv-chunked-nacl) | NaCl +0.8 % under chunked backward | same | FM | PENDING | 2026-10-27 | NaCl 1.0077 / 1.0087 | [opt6-T02](validation/opt6-T02.md) |
| [conv-chunked-ada](#conv-chunked-ada) | sm_89 check of the quarter-L2 chunk rule | same | FM | PENDING | 2026-10-27 | not measured | [opt6-T02](validation/opt6-T02.md) |
| [conv-chunked-forward](#conv-chunked-forward) | Chunked forward for low-degree cells (OPT6-T03, F3) | none | FM | SUPERSEDED | 2026-10-27 | T01: NbCl5 forward 1.90 → 1.39 ms | [opt-6](../plans/opt-6.md) |
| [conv-f64-chunks](#conv-f64-chunks) | Float64 chunks of 4 L2s | `l2_chunk_edges` rule | FM | DEFAULT (scoped: float64) | f64-T03, no decision number | NbCl5 0.90, NaNbO3 0.85 | [f64-T01](validation/f64-T01.md) |
| [conv-sobek](#conv-sobek) | Sobek per-receiver reassociation | none | FM | REJECTED | 2026-10-27 | 4× more FMA (paper) | [opt8-T01](validation/opt8-T01.md) |
| [conv-opt8-ceilings](#conv-opt8-ceilings) | Register-only stubs and L2-kept products | none | FM | MEASUREMENT | 2026-10-27 | not valid bounds | [opt8-T01](validation/opt8-T01.md) |
| [conv-full-fusion](#conv-full-fusion) | Table → `w` tile → conv, forward and backward | none | FM | REJECTED | 2026-10-27 | subgraph 0.80–0.83 (stop rule 0.75) | [opt8-T01](validation/opt8-T01.md) |
| [conv-edge-count-chunks](#conv-edge-count-chunks) | Edge-count chunking instead of receiver chunks | none | FM | NOT BUILT | 2026-10-27 | rc = 3 0.90 vs 0.83 | [opt8-T01](validation/opt8-T01.md) |
| [conv-fused-folded-forward](#conv-fused-folded-forward) | Fused folded forward storing `w` (T01b → T02) | `fused_folded_forward` | FM | DEFAULT (scoped: MH-1 labels, float32, not reproducible) | decision 64, 2026-10-27 | local NaNbO3 0.928 / 0.927, NbCl5 0.970 / 0.964, NaCl 0.991 / 1.000 | [opt8-T02](validation/opt8-T02.md) |
| [conv-tc-wtile](#conv-tc-wtile) | Tensor-core w-tile in the fused backward (F2) | none | FM | NOT BUILT | 2026-10-27 | ceiling 2.4 % NbCl5 | [ideas](../plans/ideas-20261027.md) |
| [conv-opt7-share](#conv-opt7-share) | Convolution share of the step after tables | none | FM | MEASUREMENT | 2026-10-27 | backward 9.4–18.6 % (MH-1), 14.8–24.6 % (OMAT-0) | branch fastmace/opt7-plan: plans/opt-7.md |
| [conv-opt7-atomics](#conv-opt7-atomics) | Cost of the sender-gradient atomics (T01a) | none | FM | MEASUREMENT | 2026-10-27 | ≤ 5 % of the backward | branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md |
| [conv-source-owned](#conv-source-owned) | Source-owned backward (T01b) | none | FM | NOT BUILT | 2026-10-27 | stop rule (a) not met | branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md |
| [conv-opt7-traffic](#conv-opt7-traffic) | Traffic split of forward and backward (T02) | none | FM | MEASUREMENT | 2026-10-27 | `w`/`dw` 49–67 % of L2 bytes | branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md |
| [conv-path-grouping](#conv-path-grouping) | Path-group traffic reduction | none | FM | NOT BUILT | 2026-10-27 | inferred loss (receiver-owned 2.10 vs 1.66 ms) | branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md |
| [conv-deriv-activity](#conv-deriv-activity) | Derivative activity specialisation | none | FM | NOT BUILT | 2026-10-27 | only 1/16 of `dsh` discardable | branch fastmace/opt7-plan: plans/opt-7.md |
| [conv-wide-loads](#conv-wide-loads) | `float2` loads in the streaming backward | none (transform in results) | FM | NOT BUILT | owner 2026-10-27 "document it" | local 0.931–0.971, murphy 0.951–1.003 (kernel) | branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md |

### Interaction 1 (reassociated)

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [int1-reassociate-first](#int1-reassociate-first) | Reassociated first interaction + Triton backward (+ generic, OMOL) | `reassociate_first` | both | DEFAULT | RE 2026-10-10; generic 2026-10-15; OMOL 2026-10-16 | NaCl −4.9 %, NaNbO3 −7.2 %, NbCl5 −4.9 %; OMOL NbCl5 −8.9 % | `ResearchModule:LT/studies/further_optimizations/README.md` §7 |
| [int1-segmented-first-sum](#int1-segmented-first-sum) | Run-segmented sum of the reassociated interaction (+ generic) | `segmented_first_sum` | both | DEFAULT | RE 2026-10-14; generic record 1b-T11 2026-10-18 | NaNbO3 −1.6 %; generic fix closed 1.4–1.6 % gap | [1b-T11](validation/1b-T11-mp0-setup.md) |
| [int1-lm-major](#int1-lm-major) | (l, m)-major outer sums | none (reverted) | FM | REJECTED | 2026-10-21 | 0.03 % of the step | [opt3-T05](validation/opt3-T05-layout-audit.md) |

### Symmetric contraction

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [sc-oeq-contraction](#sc-oeq-contraction) | OEq 0.7 SymmetricContraction instead of cuEq | none | both | REJECTED | RE 2026-10-16; FM 2026-10-21 | 27–79× slower | `ResearchModule:LT/studies/further_optimizations/2026-10-16-oeq-symmetric-contraction.md` |
| [sc-reduced-cg](#sc-reduced-cg) | Reduced Clebsch–Gordan basis (mace-reforge) | none | RE | REJECTED | 2026-10-16 | 0.03 % of NaNbO3 step | `ResearchModule:LT/studies/further_optimizations/2026-10-16-upstream-mace-survey.md` |
| [sc-naive-guard](#sc-naive-guard) | Refuse naive cuEq fallback (MACE PR 1614) | guard | both | DEFAULT | RE 2026-10-16 | guard, not a speed-up | `ResearchModule:LT/studies/further_optimizations/2026-10-16-upstream-mace-survey.md` |
| [sc-fused-input](#sc-fused-input) | Contraction with fused input (OPT-T09 rank 6) | none | FM | SUPERSEDED | 2026-10-21 | ceiling 1 / 3 / 2 % | [OPT-T09](design/opt-T09-step-breakdown.md) |
| [sc-opt3-t02](#sc-opt3-t02) | Symmetric contraction screen (OPT3-T02) | none | FM | NOT BUILT | 2026-10-21 | 72–96 % of memory SOL; no option ≥ 3 % | [opt3-T02](design/opt3-T02-symmetric-contraction.md) |
| [sc-tf32-bwd](#sc-tf32-bwd) | TF32 backward of interaction 0's contraction | none | FM | NOT BUILT | 2026-10-24 | fails on OMAT-0 (forces 3.92) | [opt4-T08](validation/opt4-T08-precision-sensitivity.md) |
| [sc-emulated-gemm](#sc-emulated-gemm) | Contraction as one emulated GEMM (D5) | none | FM | NOT BUILT | 2026-10-27 | 7.7 GB traffic per NbCl5 call | [ideas](../plans/ideas-20261027.md) |

### Precision and tensor cores

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [prec-dtype](#prec-dtype) | Float32 forces, TF32 off | `dtype`, `tf32` | both | DEFAULT | RE 2026-10-09 | see hybrid backend | [switch-map](design/switch-map.md) §3 |
| [prec-float64-energy](#prec-float64-energy) | Energy accumulated in float64 | `float64_energy` | FM | DEFAULT | 1b-T01/T03 | 0.3 % faster, inside noise | [1b-T03](validation/1b-T03-energy.md) |
| [prec-tf32](#prec-tf32) | TF32 matrix products (global, selective) | `allow_tf32` | both | REJECTED | RE 2026-10-08/09/12 | NaCl −19.9 % but force error 1,020–1,500× | `ResearchModule:LT/studies/remaining_optimizations/report.md` |
| [prec-fp32-emulation-re](#prec-fp32-emulation-re) | cuBLAS FP32 emulation on tensor cores (RE) | none | RE | REJECTED | 2026-10-07 | unavailable on GeForce Blackwell | `ResearchModule:LT/studies/optimization_history/report.md` §8 |
| [prec-torch-modes](#prec-torch-modes) | torch FP32 matmul precision modes | none | FM | REJECTED | 2026-10-20 | no FP32-accurate mode | [opt-T01](validation/opt-T01-tensor-cores.md) §1 |
| [prec-bf16x9](#prec-bf16x9) | cuBLAS BF16×9 FP32 emulation | none | FM | REJECTED | 2026-10-20 | sm_100/103 only | [opt-T01](validation/opt-T01-tensor-cores.md) §1 |
| [prec-tf32x3-triton](#prec-tf32x3-triton) | 3×TF32 GEMMs (RE probe; Triton `tf32x3`) | bench | both | REJECTED | RE 2026-10-08; FM 2026-10-20/23 | murphy 1.78× / 1.49× slower; A100 0 of 92 shapes | [opt-T01](validation/opt-T01-tensor-cores.md) |
| [prec-cutlass-ada](#prec-cutlass-ada) | CUTLASS 3×TF32 on sm_89 / sm_120 | none | FM | NOT BUILT | 2026-10-20 | same ceiling as Triton | [opt-T01](validation/opt-T01-tensor-cores.md) §3 |
| [prec-cutlass-a100](#prec-cutlass-a100) | CUTLASS 3×TF32 on the A100 (stage 2) | would sit behind `tf32x3_gemms` | FM | NOT BUILT | owner 2026-10-24 | GEMM time NaCl 0.75×, NbCl5 0.55× | [opt-T01](validation/opt-T01-tensor-cores.md) §5, §7 |
| [prec-bf16x6](#prec-bf16x6) | Triton BF16×6 / BF16×3 | none | FM | REJECTED | 2026-10-23 | never beats cuBLAS IEEE at K = 512 | [opt-T01](validation/opt-T01-tensor-cores.md) §6 |
| [prec-tc-overlap](#prec-tc-overlap) | Tensor-core GEMM beside SIMT on two streams | bench | FM | REJECTED | 2026-10-20 | 0.77–0.98× of cuBLAS alone | [opt-1](validation/opt-1.md) |
| [prec-tf32x3-gemms](#prec-tf32x3-gemms) | 3×TF32 atom GEMMs as a record (+ tile rule) | `tf32x3_gemms` | FM | NEVER-AUTOMATIC | 2026-10-24; tile rule 2026-10-26 | accuracy pass; A100 slower (NaCl 3.1 → 4.2 ms) | [opt4-T03](validation/opt4-T03.md) |
| [prec-hybrid-dispatcher](#prec-hybrid-dispatcher) | Libra hybrid dispatcher speed target | planned `hybrid_tc_simt_gemms` | FM | NOT BUILT | 2026-10-26 | not measured | [opt-5 plan](../plans/opt-5.md) |
| [prec-deepseek-x3](#prec-deepseek-x3) | SIMT-split producer + pure MMA consumer | probe `x3_pipe` | FM | CUT | 2026-10-26 | gain 0.85 on sm_120 | [opt5-T01](validation/opt5-T01-scheduler-probe.md) |
| [prec-precision-screen](#prec-precision-screen) | Per-stage bf16/fp16/TF32 sensitivity | none | FM | REJECTED | 2026-10-24 | fp16 misses 2–1,100×, bf16 7 to > 30,000× | [opt4-T08](validation/opt4-T08-precision-sensitivity.md) |
| [prec-zbl-fp16](#prec-zbl-fp16) | ZBL stage in fp16 storage | none | FM | NOT BUILT | 2026-10-24 | passes, share ≈ 0 | [opt4-T08](validation/opt4-T08-precision-sensitivity.md) |
| [prec-fp16-edge-weights](#prec-fp16-edge-weights) | FP16 / BF16 storage of per-edge weights | probe | RE | REJECTED | 2026-10-14 | fails by 3–110× | `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` |
| [prec-fp16x3-probe](#prec-fp16x3-probe) | Exponent-shifted 3×FP16 co-execution probe | probe | FM | REJECTED | 2026-10-26 | probe gain 1.22× on sm_120; no kernel realised it | [opt-5](validation/opt-5.md) §3.1 |
| [prec-fp16x3-standalone](#prec-fp16x3-standalone) | Standalone Triton 3×FP16 GEMM on sm_120 | bench | FM | REJECTED | 2026-10-26 | ties or loses to cuBLAS IEEE | [opt-5](validation/opt-5.md) §3.2 |
| [prec-fp16x3-rowcol](#prec-fp16x3-rowcol) | Row/column-normalised 3×FP16 | bench | FM | REJECTED | 2026-10-26 | accurate; 860.3 vs 522.3 µs | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) |
| [prec-fp16x3-other-gpus](#prec-fp16x3-other-gpus) | Row/col 3×FP16 on A100, A4000, Ada | planned `fp16x3_gemms` | FM | REJECTED | 2026-10-26 | 1.12–1.71× of cuBLAS | [opt-5](validation/opt-5.md) §3.11 |
| [prec-delta-gemm](#prec-delta-gemm) | Time-step delta GEMM | none | FM | REJECTED | 2026-10-26 | 482–6,130× worse error | [opt-5](validation/opt-5.md) §3.3 |
| [prec-channel-split](#prec-channel-split) | 50/50 channel-magnitude split | none | FM | REJECTED | 2026-10-26 | 411–1,343× worse | [opt-5](validation/opt-5.md) §3.3 |
| [prec-hybrid-gemm](#prec-hybrid-gemm) | Fused SIMT + 3×FP16 NVRTC GEMM | bench | FM | REJECTED | 2026-10-26 | NbCl5 shapes +6.1 to +19.8 % | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.2 |
| [prec-packed-b](#prec-packed-b) | Larger tiles, pre-split packed B | bench | FM | REJECTED | 2026-10-26 | slower; accuracy fails | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.3 |
| [prec-hybrid-profile](#prec-hybrid-profile) | Nsight / power profile of the hybrids | none | FM | MEASUREMENT | 2026-10-26 | issue 58–62 % vs 76.6 % | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.4 |
| [prec-decoupled](#prec-decoupled) | Subgroup-barrier decoupling | bench | FM | REJECTED | 2026-10-26 | +0.7 to +5.3 % vs coupled | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.5 |
| [prec-mbarrier](#prec-mbarrier) | Concurrent kernels / `mbarrier` rings | none | FM | NOT BUILT | 2026-10-26 | not measured | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.5 |
| [prec-two-kernel-split](#prec-two-kernel-split) | Two-kernel FP32/MMA column split | bench | FM | REJECTED | 2026-10-26 | 734.1 vs 501.0 µs | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6 |
| [prec-bk32-swizzle](#prec-bk32-swizzle) | BK = 32 staging, block swizzling | bench | FM | REJECTED | 2026-10-26 | shape-dependent, far behind cuBLAS | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6 |
| [prec-fp32-baselines](#prec-fp32-baselines) | FP32 baseline options (cuBLASLt, transposed B) | screen | FM | REJECTED | 2026-10-26 | 501.0 / 501.9 / 697.7 µs | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6 |
| [prec-tensor-only-sweep](#prec-tensor-only-sweep) | Tensor-only 3×FP16 tile sweep | bench | FM | REJECTED | 2026-10-26 | best 795.9 vs 501.0 µs | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6 |
| [prec-presplit-epilogue](#prec-presplit-epilogue) | Producer-side A split + epilogue fusion | bench | FM | REJECTED | 2026-10-26 | 1,737 vs 501.0 µs total | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6 |
| [prec-setmaxnreg](#prec-setmaxnreg) | 4+4 warpgroup `setmaxnreg` | bench | FM | REJECTED | 2026-10-26 | 716.3 vs 530.6 µs | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.7 |
| [prec-cutlass-sm120](#prec-cutlass-sm120) | CUTLASS SM120 FP16 GEMM as base | bench | FM | REJECTED | 2026-10-26 | 194.5 vs 174.5 µs | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.7 |
| [prec-producer-split](#prec-producer-split) | Fused producer A split (Triton) | bench | FM | REJECTED | 2026-10-26 | 682.1 vs 503.7 µs | [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.7 |
| [prec-gemmul8](#prec-gemmul8) | GEMMul8 Ozaki II emulation | bench | FM | REJECTED | 2026-10-26 | invalid benchmark | [opt-5](validation/opt-5.md) §3.9 |
| [prec-int8-f32](#prec-int8-f32) | INT8-slice float32 GEMMs | bench | FM | REJECTED | 2026-10-27 | 599–965 vs 518 µs | [f64-T01](validation/f64-T01.md) §6 |
| [prec-f64-hybrid](#prec-f64-hybrid) | Hybrid backend for float64 | backend choice | FM | DEFAULT (scoped: float64, not reproducible) | decision 63(a) | plain cuEq 2.6× / 3.3× slower | [f64-T01](validation/f64-T01.md) |
| [prec-fp64-emulation](#prec-fp64-emulation) | cuBLAS FP64 emulation, eager | `fp64_emulation` | FM | DEFAULT (scoped: float64, FP64 rate < 1/8, not reproducible) | decision 63 | NaCl 0.80, NbCl5 0.55 | [f64-T01](validation/f64-T01.md) §2 |
| [prec-emu-performant](#prec-emu-performant) | Emulation "performant" strategy | env | FM | REJECTED | 2026-10-27 | 0.86 / 0.67 vs eager | [f64-T01](validation/f64-T01.md) §2 |
| [prec-emu-40](#prec-emu-40) | 40 emulation mantissa bits | env | FM | REJECTED | 2026-10-27 | error 6e-14 / 6e-12 | [f64-T01](validation/f64-T01.md) §2 |
| [prec-f64-rewrites](#prec-f64-rewrites) | OPT-T08 folds in float64 | three fields | FM | DEFAULT (scoped: float64, not reproducible) | decision 63(c) | step 0.91 | [f64-T01](validation/f64-T01.md) §3 |
| [prec-f64-triton-rewrites](#prec-f64-triton-rewrites) | `segment_gemm`/`grouped_launch` in float64 | — | FM | NOT BUILT | 2026-10-27 | not measured | [f64-T01](validation/f64-T01.md) §3 |
| [prec-f64-package](#prec-f64-package) | Float64 package in the default plan | decision-63 parts | FM | DEFAULT (scoped: float64) | decision 63 | local NaCl 0.57, NbCl5 0.42, NaNbO3 0.49 | [f64-T01](validation/f64-T01.md) |
| [prec-double-float](#prec-double-float) | Paired-float32 (double-float) arithmetic | none | FM | REJECTED | 2026-10-27 | 1.94× native FP64 (bar 2.0×) | branch fastmace/f64-T04: docs/validation/f64-T04.md |
| [prec-d1-56bits](#prec-d1-56bits) | Fixed 56 mantissa bits (D1) | `fp64_emulation_mantissa_bits` | FM | DEFAULT (scoped: float64 MH-1, two named GPUs; suspended ON BRANCH) | decision 65(a), 2026-10-28 | force graph 0.86–0.88 | [ideas](../plans/ideas-20261027.md) |
| [prec-d1-relax-fix](#prec-d1-relax-fix) | D1 fix: small step products element-wise; auto = dynamic | none | FM | ON BRANCH | owner 2026-10-28 | correctness | branch fastmace/fp64-d1-relax-fix: commits `bc292dc6`, `7b3c1d88`, `bbd8bd23` |
| [prec-d1-48](#prec-d1-48) | 48 or 52 mantissa bits | explicit count | FM | REJECTED | 2026-10-27 | errors 4–20e-15 | [ideas](../plans/ideas-20261027.md) |
| [prec-save-transcendentals](#prec-save-transcendentals) | No recompute of transcendentals (D3) | `save_transcendentals` | FM | DEFAULT (scoped: float64 MH-1, two named GPUs) | decision 65(c) | local 0.968–0.983; A100 0.994–1.023 | [ideas](../plans/ideas-20261027.md) |
| [prec-d123](#prec-d123) | D1 + D2 + D3 together | — | FM | DEFAULT (scoped: as decision 65) | decision 65 | ABBA 0.782–0.806 | [ideas](../plans/ideas-20261027.md) |
| [prec-ozaki-fp8](#prec-ozaki-fp8) | FP8/FP4 Ozaki emulation for float64 | none | FM | REJECTED | 2026-10-27 | not recorded | [ideas](../plans/ideas-20261027.md) |

### Launches, graphs, scheduling and megakernels

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [launch-cuda-graphs](#launch-cuda-graphs) | CUDA graphs with padded edge sizes; graph tier | tier `mode` | both | DEFAULT | RE 2026-10-07; decision 38 | NaCl 30.0 → 16.5 ms (−45 %) | `ResearchModule:LT/studies/optimization_history/report.md` §3 |
| [launch-compiled-tier](#launch-compiled-tier) | Compiled (non-graph) tier | tier `mode` | FM | DEFAULT (scoped: inexact padding, fallback) | decision 38, 2026-10-22 | MH-1 21.7 vs graph 4.2 ms | [3-T06](validation/3-T06.md) |
| [launch-adaptation](#launch-adaptation) | Runtime graph → compiled switch | `AdaptationConfig.adapt` | FM | DEFAULT | decisions 52, 53 | storm 27.1 vs 219.2 ms | [phase-5](validation/phase-5.md) |
| [launch-calibration](#launch-calibration) | Opt-in backend calibration | `CalibrationConfig.calibrate` | FM | OPT-IN | decision 55 | kept hybrid (3.95 vs 7.25 ms) | [5-T09](validation/5-T09.md) |
| [launch-host-stop](#launch-host-stop) | Remove the last host stop (spin / lagged pipeline) | none | RE | REJECTED | 2026-10-07/08 | 10.14 vs 9.77 ms | `ResearchModule:LT/studies/optimization_history/report.md` §8 |
| [launch-speculative](#launch-speculative) | Speculative combined AC+B graph | `speculative_graph` | both | OPT-IN | RE 2026-10-13 | real −4.0 to −15.3 % (slower) | `ResearchModule:LT/studies/further_optimizations/2026-10-13-speculative-graph-screen.md` |
| [launch-pipelined](#launch-pipelined) | Pipelined steps | `pipelined` | both | DEFAULT | RE 2026-10-14 | NaCl −3.0 %, NaNbO3 −0.8 % | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §4c |
| [launch-step-graph](#launch-step-graph) | One ACPB graph per pipelined step | `step_graph` | FM | OPT-IN | 2026-10-21 | NaCl 1000 K 0.9973 / 0.9999 | [opt2-T04](validation/opt2-T04.md) |
| [launch-b-fuse](#launch-b-fuse) | Fuse kernels inside the B replay | none | FM | SUPERSEDED | 2026-10-21 | 147 µs of gaps | [opt2-T04](validation/opt2-T04.md) |
| [launch-combo-kernels](#launch-combo-kernels) | Inductor combo kernels (+ aggressive fusion) | `combo_kernels`; RE `INDUCTOR_COMBO` | both | NEVER-AUTOMATIC | RE 2026-10-14; FM 2026-10-21 | NaCl 0.991 / 0.982, NbCl5 1.024 / 1.027 | [opt3-T04](validation/opt3-T04.md) |
| [launch-combo-persub](#launch-combo-persub) | Combo kernels with per-sub-kernel blocks | screen | FM | REJECTED | 2026-10-21 | NbCl5 1.023 | [opt3-T04](validation/opt3-T04.md) |
| [launch-combo-sizerule](#launch-combo-sizerule) | Combo kernels on small cells only | none | FM | NOT BUILT | 2026-10-21, 2026-10-24 | ≈ 1 % NaCl | [opt3-T04](validation/opt3-T04.md) |
| [launch-memsets](#launch-memsets) | cuBLAS memsets, per-step constants, eager zero-fills | none | FM | NOT BUILT | 2026-10-21 | not timed | [opt3-T04](validation/opt3-T04.md) |
| [launch-pdl](#launch-pdl) | Inductor programmatic dependent launch | `INDUCTOR_PDL` | RE | REJECTED | 2026-10-15 | wrong forces in graph replays | `ResearchModule:LT/studies/further_optimizations/2026-10-15-fourth-round.md` §3 |
| [launch-batching](#launch-batching) | Disconnected-graph batching (idea 7) | probe | RE | REJECTED | 2026-10-09 | 2.32 % (2 × NaCl) | `ResearchModule:LT/studies/batching/README.md` |
| [launch-megakernel](#launch-megakernel) | Complete force megakernel | none | FM | NOT BUILT | 2026-10-20 | ceiling 8–16 % NaCl, 1–2 % NbCl5 | [megakernel research](design/2026-10-20-megakernel-research.md) |
| [launch-het-megakernel](#launch-het-megakernel) | Heterogeneous persistent megakernel; co-execution gate | probe | FM | REJECTED | owner 2026-10-24 | sm_120 gain 1.04, A100 0.92 | [opt4-T07](validation/opt4-T07-megakernel-probe.md) |
| [launch-sched-iso](#launch-sched-iso) | Scheduler-isolated co-execution | probe | FM | REJECTED | 2026-10-26 | sched_iso 0.72 | [opt5-T01](validation/opt5-T01-scheduler-probe.md) |
| [launch-regional-persistence](#launch-regional-persistence) | Regional persistence of the fused atom block (OPT-T05) | none | FM | PENDING | 2026-10-20; close-out recommended 2026-10-24 | gaps 100–152 µs (0.5–5 %) | [opt-4](../plans/opt-4.md) §2 |
| [launch-cublasdx](#launch-cublasdx) | cuBLASDx block GEMM as megakernel primitive | bench | FM | REJECTED | 2026-10-26 | ratio 1.062 | [opt-5](validation/opt-5.md) §3.12 |
| [launch-two-streams](#launch-two-streams) | Independent branches on two streams | probe | FM | REJECTED | 2026-10-26 | 0.985–1.151× | [opt-5](validation/opt-5.md) §3.12 |
| [launch-profiles](#launch-profiles) | Step breakdowns and profiles (RE ncu, 15 Oct, 2-T20a, OPT-T09, OPT-T10, OPT4-BASE, float32 after tables, float64) | none | both | MEASUREMENT | 2026-10-09 … 2026-10-27 | see entry | [2-T20a](validation/2-T20a-step-breakdown.md), [OPT-T10](design/opt-T10-step-breakdown.md) |

### Integrator, engine and neighbour list

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [eng-gpu-integrator](#eng-gpu-integrator) | GPU-resident Langevin integrator | `--engine gpu`; FastMACE engine | both | DEFAULT | RE 2026-10-07/08 | NaCl 41.4 → 30.0 ms (with skin list) | `ResearchModule:LT/studies/optimization_history/report.md` |
| [eng-skin-list](#eng-skin-list) | Skin neighbour list, 0.3 Å | `skin` | both | DEFAULT | RE 2026-10-07 | in 41.4 → 30.0 ms | `ResearchModule:LT/studies/optimization_history/report.md` §2 |
| [eng-skin-rescreens](#eng-skin-rescreens) | Skin 0.2 / 0.4 Å, first skin/bucket screen | `skin` | RE | REJECTED | 2026-10-08, 12, 14 | within noise | `ResearchModule:LT/studies/further_optimizations/2026-10-12-gear-switch-screen.md` |
| [eng-edge-buckets](#eng-edge-buckets) | 5 % edge buckets and capacity constants | `edge_bucket_fraction` etc. | both | DEFAULT | RE 2026-10-07 | memory −15 %, no speed cost | `ResearchModule:LT/docs/roadmap.md` item 1 |
| [eng-bucket-width](#eng-bucket-width) | 2 % / 3 % / 10 % buckets; stand-alone padding | `edge_bucket_fraction` | both | REJECTED | 2026-10-09, 12, 14; 2026-10-24 | +0.4 % to noise | `ResearchModule:LT/studies/edge_buckets/README.md` |
| [eng-batched-noise](#eng-batched-noise) | 256-step noise batches, one sync per step | `noise_batch` | both | DEFAULT | RE 2026-10-07 | −5 to −8 % | `ResearchModule:LT/studies/optimization_history/report.md` §6 |
| [eng-first-derivative](#eng-first-derivative) | First-derivative-only compiled forces | `first_derivative_only` | both | DEFAULT | RE 2026-10-08 | NaNbO3 −30 % | `ResearchModule:LT/docs/roadmap.md` items 3a/3c |
| [eng-warmup-pool](#eng-warmup-pool) | Warm-ups inside the graph pool | `warmup_in_graph_pool` | both | DEFAULT | RE 2026-10-07/08 | peak 9.4 → 4.9 GiB | `ResearchModule:LT/studies/optimization_history/report.md` §7 |
| [eng-expandable](#eng-expandable) | Expandable allocator segments | `expandable_segments` | both | DEFAULT | RE 2026-10-08 | 7.4 → 4.1 GiB, same speed | `ResearchModule:LT/studies/optimization_history/report.md` §7 |
| [eng-empty-cache](#eng-empty-cache) | `empty_cache()` after each capture | none | RE | REJECTED | 2026-10-07 | no peak reduction | `ResearchModule:LT/studies/optimization_history/report.md` §8 |
| [eng-gpu-neighbour-list](#eng-gpu-neighbour-list) | GPU neighbour list (Triton pair search) | `gpu_neighbour_list` | both | DEFAULT | RE 2026-10-14 | rebuild 1.4–1.7 vs 4–24 ms; step −1.6 to −4.6 % | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [eng-species-morton](#eng-species-morton) | Species + Morton atom order | probe | RE | REJECTED | 2026-10-14 | no gain | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [eng-force-scatter](#eng-force-scatter) | Segmented force scatter | none | RE | NOT BUILT | 2026-10-14 | ≈ 21 µs | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [eng-baoab-1fs](#eng-baoab-1fs) | BAOAB at 1 fs (as a speed item) | none | RE | NOT BUILT | 2026-10-14 | excluded (timestep change) | `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` |
| [eng-lbfgs](#eng-lbfgs) | LBFGS for initial relaxation and quench | `--quench-optimizer` | RE | DEFAULT (scoped: ResearchModule `main.py`) | user 2026-10-13 | 1.1 h vs ≈ 31 h | `ResearchModule:LT/studies/annealing/README.md` |
| [eng-overlap-rebuild](#eng-overlap-rebuild) | Early rebuild on a side stream | `overlap_rebuild` | FM | DEFAULT | OPT-DEF-2, 2026-10-21 | NaCl 1000 K 0.9635 / 0.9674 | [opt2-T04](validation/opt2-T04.md) |
| [eng-noise-in-graph](#eng-noise-in-graph) | Noise selection inside the graph | none | FM | NOT BUILT | 2026-10-21 | 0.061 ms host, hidden | [opt2-T04](validation/opt2-T04.md) |
| [eng-fused-integrator](#eng-fused-integrator) | Fused Triton integrator stage | `fused_integrator` | FM | DEFAULT | owner 2026-10-21, §19 14th | NaCl 0.9758 / 0.9758 | [opt3-T03](validation/opt3-T03.md) |
| [eng-fixcm](#eng-fixcm) | Mass-weighted fixcm projection in the fused stage | `fixcm_projection` | FM | DEFAULT | decision 37 (correctness) | 0.09 %, below floor | [3-T22b](validation/3-T22b-fused-projection.md) |
| [eng-fused-nhc](#eng-fused-nhc) | Fused NHC / MTK stage | none | FM | DEFAULT (scoped: NHC and MTK integrators) | conductor, 2026-10-23 | NHC −4.4 %, MTK full −13 % | [4-T03b](validation/4-T03b.md) |
| [eng-rule11](#eng-rule11) | Adaptive skin and headroom (rule 11) | `skin`, `headroom`, `d3_skin` | FM | DEFAULT (scoped: MC, species-change, displacement workloads) | 3-T05, 2026-10-22 | inside the floor | [3-T05](validation/3-T05.md) |
| [eng-d3-cost](#eng-d3-cost) | GPU D3(BJ) cost | dispersion | FM | MEASUREMENT | decision 42 | +24 % at 1000 K (target missed) | [3b-T06](validation/3b-T06.md) |
| [eng-species-change](#eng-species-change) | Species-change events vs fixed species | workload | FM | MEASUREMENT | 2026-10-22 | rate 0.946–0.992 | [3-T20](validation/3-T20.md) |
| [eng-incremental-nl](#eng-incremental-nl) | Incremental neighbour list (F5) | none | FM | NOT BUILT | 2026-10-27 | ≤ 1 % | [ideas](../plans/ideas-20261027.md) |
| [eng-on-nl](#eng-on-nl) | O(N) neighbour-list rebuild for large cells | none | FM | NOT BUILT | owner 2026-10-28 (future) | not measured | [future NL plan](../plans/future-neighbour-list.md) |
| [eng-parity](#eng-parity) | Parity and regression A/Bs (1a, 1b, 2, Ada, 5) | none | FM | MEASUREMENT | 2026-10-18 … 2026-10-23 | see entry | [phase-1a](validation/phase-1a.md) |
| [eng-bucket-compile](#eng-bucket-compile) | 1a-T13b: compile at the first new bucket (fix) | none | FM | DEFAULT | 2026-10-18 (`ae3e758`) | invalid blocks before the fix | [1a-T13b](validation/1a-T13b-bucket-compile.md) |
| [eng-static-relax](#eng-static-relax) | Static-relaxation GPU screen (threads, optimiser, EOS fmax, mixed precision, concurrency) | various | RE | MEASUREMENT | 2026-10-01 | see entry | `ResearchModule:Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/README.md` |

### Setup, compilation and start-up

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [setup-torch-compile](#setup-torch-compile) | torch.compile `default` mode inside outer graphs | `compile_mode` | both | DEFAULT | RE 2026-10-08 | coraline NaCl 17.57 → 11.13 ms | `ResearchModule:LT/studies/kernel_fusion/report.md` |
| [setup-stack](#setup-stack) | torch 2.14.1 + cuEq 0.12.0 (recipe C) | environment | both | DEFAULT | user 2026-10-08 | local 12.6 → 10.5 ms | `ResearchModule:LT/studies/optimization_history/report.md` §5 |
| [setup-max-autotune](#setup-max-autotune) | `max-autotune` compilation | `--compile-mode` | RE | REJECTED | 2026-10-08, 2026-10-12 | fails with outer graphs on the hybrid | `ResearchModule:LT/studies/further_optimizations/2026-10-12-gear-switch-screen.md` |
| [setup-activation-budget](#setup-activation-budget) | Activation memory budget < 1 | `ACTIVATION_MEMORY_BUDGET` | RE | REJECTED | 2026-10-15 | +0.4 to +5.9 ms | `ResearchModule:LT/studies/further_optimizations/2026-10-15-fourth-round.md` §2 |
| [setup-oeq-seed](#setup-oeq-seed) | Reuse a prebuilt OEq host extension | `OEQ_EXTENSION_SEED` | RE | OPT-IN | 2026-10-09 | import 19.6 → 1.5–1.7 s | `ResearchModule:LT/docs/roadmap.md` item 9 |
| [setup-template-cache](#setup-template-cache) | Converted-model template cache | `model_template_cache` | FM | DEFAULT (scoped: CUDA cuEq and hybrid) | conductor 2026-10-21, §19 12th | −2.2 to −3.3 s per later plan | [opt-T06](validation/opt-T06-template-cache.md) |
| [setup-mace-version](#setup-mace-version) | `installed_mace_version()` once per process | none | FM | SUPERSEDED | 2026-10-21 | not measured | [OPT-T06 design](design/opt-T06-template-cache.md) |
| [setup-persistent-worker](#setup-persistent-worker) | Persistent subprocess worker | `RunnerConfig.persistent` | FM | DEFAULT (scoped: POSIX) | conductor 2026-10-20 | `run(100)` ≈ 70 s → 10 s | [opt-T07](validation/opt-T07-persistent-worker.md) |
| [setup-dynamic-edges](#setup-dynamic-edges) | Dynamic edge dimension (eval plans; graph-tier MD) | `dynamic_edges` | FM | DEFAULT | 2-T13; decision 59 | starts 7–9 s shorter warm, 16–22 s cold | [opt4-T05](validation/opt4-T05.md) |
| [setup-compiler-caches](#setup-compiler-caches) | Compiler caches under `FASTMACE_CACHE` | process setup | FM | DEFAULT | §19 17th, 2026-10-24 | cold-warm gap 20 vs 33 s | [opt4-T05](validation/opt4-T05.md) |
| [setup-other-items](#setup-other-items) | OEq build, portable artifacts, self-check cost, load, graph variants | none | FM | NOT BUILT | 2026-10-24 | below the build rule | [opt4-T05](validation/opt4-T05.md) |
| [setup-selfcheck-cache](#setup-selfcheck-cache) | Qualification cache to skip self-checks | `selfcheck_cache` | FM | NOT BUILT | 2-T19, 2026-10-20 | ≤ 1.4 s per process warm | [2-T19](validation/2-T19-setup.md) |
| [setup-gap](#setup-gap) | Setup gap diagnosis (1b-T11) | none | FM | MEASUREMENT | 2026-10-18/19 | artefact of cold caches | [1b-T11](validation/1b-T11-mp0-setup.md) |

### Output

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [out-async-output](#out-async-output) | Asynchronous output | `async_output` | both | DEFAULT | RE 2026-10-15 | NaCl −1.9 %, NaNbO3 −0.2 % (wall) | `ResearchModule:LT/studies/further_optimizations/2026-10-15-fourth-round.md` §4 |
| [out-ase-shim](#out-ase-shim) | ASE shim metadata once; software block once per process | none (fixes) | FM | DEFAULT | 1b-T21b, 1b-T28 | `/mnt/d` 48.08 → 4.117 ms/step | [1b-T21b](validation/1b-T21b-ase-shim-overhead.md) |

### Other models (generic family, POLAR, OMOL, OFF)

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [model-generic-rewrites](#model-generic-rewrites) | Generic streaming and reassociation for non-MH-1 models | `streaming_conv`, `reassociate_first` | both | DEFAULT | user 2026-10-15 | −16 to −28 % | `ResearchModule:LT/studies/other_models/README.md` |
| [model-species-skip](#model-species-skip) | Species-restricted skip connection | `species_skip` | both | DEFAULT (scoped: generic family) | user 2026-10-15 | −7 to −39 % | `ResearchModule:LT/studies/other_models/README.md` |
| [model-generic-density-first](#model-generic-density-first) | Density-first for the generic family | `GENERIC_DENSITY_FIRST` | RE | REJECTED | 2026-10-15 | no gain | [switch-map](design/switch-map.md) §7 |
| [model-polar-uvu](#model-polar-uvu) | PolarMACE uvu reassociation (MACE PR 1711) | `polar_uvu` | both | DEFAULT (scoped: POLAR models) | user 2026-10-16 | 4.2–5.5× (e.g. NbCl5 574 → 114.6 ms) | `ResearchModule:LT/studies/other_models/README.md` |
| [model-spin-restricted](#model-spin-restricted) | Spin-restricted PolarMACE | `SPIN_RESTRICTED` | RE | REJECTED | RE | changes the model (max force 0.47 → 30.5 eV/Å) | [switch-map](design/switch-map.md) §7 |
| [model-fx-e3nn](#model-fx-e3nn) | TorchScript e3nn `Linear` back to fx | `fx_e3nn_linear` | FM | DEFAULT (scoped: POLAR-1-S/M/L, hybrid float32, not reproducible) | decision 58 | with graph sums NaCl 0.7634 / 0.7589, NbCl5 0.8225 / 0.8231 | [opt4-T02](validation/opt4-T02.md) |
| [model-polar-graph-sums](#model-polar-graph-sums) | Deterministic per-graph sums for PolarMACE | `polar_graph_sums` | FM | DEFAULT (scoped: as `fx_e3nn_linear`) | decision 58 | alone −8 / −9 %; none on top | [opt4-T02](validation/opt4-T02.md) |
| [model-ffwd-polar](#model-ffwd-polar) | `fused_folded_forward` on POLAR-1-S | explicit on | FM | REJECTED | decision 64 scope | b/a 1.004 | [opt8-T02](validation/opt8-T02.md) |
| [model-ffwd-omol](#model-ffwd-omol) | `fused_folded_forward` on OMOL-0 XL | explicit on | FM | REJECTED | decision 64 scope | b/a 1.035 | [opt8-T02](validation/opt8-T02.md) |
| [model-gate-t03d](#model-gate-t03d) | Gate tool applies rule 13 (OMOL-0 on A100) | tooling | FM | DEFAULT | 2026-10-24 (`3e06d243`) | tooling | [opt4-T03](validation/opt4-T03.md) §9 |
| [model-other-models](#model-other-models) | Benchmark models, MACE-OFF, POLAR, OMOL on the GPU engine | `--model` | RE | MEASUREMENT | 2026-10-15/16 | 2.9–12.2× vs stock ASE (ten models) | `ResearchModule:LT/studies/other_models/README.md` |

### Alternative backends and external comparisons

| ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ext-hybrid](#ext-hybrid) | cuEq/OEq hybrid backend | `backend` | both | DEFAULT | user 2026-10-09 | NaCl −26.8 %, NaNbO3 −33.9 %, NbCl5 −23.0 % | `ResearchModule:LT/studies/hybrid_oeq/rtx5060ti.md` |
| [ext-jax-mliap](#ext-jax-mliap) | MACE-JAX and LAMMPS ML-IAP | separate envs | RE | REJECTED | 2026-10-08/09 | ML-IAP 3.4× slower; JAX fails accuracy | `ResearchModule:LT/studies/alternative_backends/README.md` |
| [ext-symmetrix](#ext-symmetrix) | Symmetrix-XL engine; LAMMPS rematch | separate venv | both | REJECTED | 2026-10-14; 2026-10-23 | 1.89–2.50× slower; rematch 1.07–2.0× | `ResearchModule:LT/studies/symmetrix_xl/README.md` |
| [ext-lunatp](#ext-lunatp) | LunaTP tensor product | scratch venv | FM | REJECTED | 2026-10-27 | backward 1.39–1.53× slower on large cells | branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md |
| [ext-torchsim](#ext-torchsim) | TorchSim 0.6.2 comparison | none | FM | MEASUREMENT | 2026-10-25 | FastMACE 3.8–10× faster | [TorchSim](validation/torchsim-comparison-20261025.md) |
| [ext-alchemi](#ext-alchemi) | NVIDIA ALCHEMI 0.2.0 comparison | none | FM | MEASUREMENT | 2026-10-26 | 2.74–9.62× faster | [ALCHEMI](validation/alchemi-comparison-20261025.md) |
| [ext-stock-ase](#ext-stock-ase) | Stock ASE comparison (phase 4) | none | FM | MEASUREMENT | phase 4 (date not recorded) | 10–23× NaCl, 6–12× NbCl5 | [phase-4](validation/phase-4.md) |

## Chronology

Step times are MH-1 float32 MD steps (NaCl / NaNbO3 / NbCl5, 3×3×3) on the GPU named; rows from different sessions
are not A/B comparisons.

| Date | Round or phase | Tried | Adopted | Milestone |
| --- | --- | --- | --- | --- |
| 2026-10-01 | Benchmark_MACE_GPU_OPTIMIZATIONS | Static relaxations on RTX A4000: threads, BFGS/LBFGS/FIRE, EOS fmax, mixed precision, TF32, concurrency | One CPU thread; FIRE and TF32 rejected | ASE float64 cuEq timed step 31.44 ms (NaCl, leeloo) |
| 2026-10-07 | RE phase 0 | GPU integrator, skin list, CUDA graphs, torch.compile, first-derivative forces, 5 % buckets, torch 2.14 + cuEq 0.12, batched noise, cuBLAS FP32 emulation, `empty_cache` | Integrator, skin list, graphs, buckets, batched noise | Local NaCl 41.4 → 30.0 → 16.5 → 12.6 → 10.0 ms |
| 2026-10-08 | RE phase 0 | Compile adoption, unfused addmm, expandable segments, host-stop removal, single head, `fused_tp`, cuBLASLt, max-autotune, TF32, TF32x3; hybrid screens | Compile `default`, unfused addmm, allocator, stack recipe C | Local cuEq FP32 9.7 ms; hybrid 7.075 / 55.920 / 55.650 ms |
| 2026-10-09 | RE phase 1 (eight ideas, V2) and phase 2 | Hybrid production decision, OEq seed, buckets 2–3 %, segmented reduction, ncu, batching, JAX / ML-IAP, GEMM autotune, OEq schedules, species factoring, cutoff folding, sorting (force screen) | Hybrid FP32 (user), factoring and folding | Local 5.163 / 35.938 / 40.540 ms |
| 2026-10-10 | RE phase 2 | Sorting in MD, tabulation and sharing ceilings, density-first, reassociated first interaction, gate VJP ceiling; autotune on hybrid | Sorting, density-first, reassociation (`main.py` default 10 Oct) | Local 4.647 / 30.918 / 34.345 ms |
| 2026-10-12 | RE gear-switch screen | Ablations, skin 0.2, buckets, max-autotune on the hybrid | Nothing | Desktop-loaded, 5–12 % inflated |
| 2026-10-13 | RE | Speculative AC+B graph; LBFGS quench | LBFGS (user); speculative stays opt-in | Local ≈ 4.65 / 30.9 / 34.4 ms |
| 2026-10-14 | RE third round and streamed edges | Joint radial input, fused folded backward, GPU NL, segmented sum, block linear, pipelined, contiguous weight, many rejects; Symmetrix-XL; chunks, own conv, OEq PR 223 streaming, `ir_mul`, cuBLASLt, combo, FP16 weights, skin 0.4, fused radial hidden | Joint input, fused backward, GPU NL, segmented sum, block linear, pipelined, streaming conv, `ir_mul` | Local ≈ 3.75 / 23.1 / 28.8 ms; leto 3.952 / 27.489 / 34.121 ms |
| 2026-10-15 | RE fourth round and other models | Step breakdown, async output, in-place block linear, activation budget, PDL, mirror edges, generic rewrites rounds 1–2, fused radial hidden fix | Async output; generic streaming, reassociation, species skip, fused radial backward | Murphy 3.781 / 25.861 / 32.443 ms |
| 2026-10-16 | RE upstream and OMOL | Grouped / merged linears, PolarMACE uvu, OEq contraction, reduced CG, single-graph strain, cuEq conv fusion, naive guard | Grouped and merged linears, uvu, strain, cuEq conv fusion | Murphy 3.55 / 25.74 / 30.54 ms |
| 2026-10-17 | FM phase 1a | Rewrites and engine features carried over; research engine frozen as `research-engine-baseline-20261017` | All RE defaults | – |
| 2026-10-18 | FM phase 1a/1b | Parity A/Bs (1a-T19), bucket-compile fix, generic segmented sum (1b-T11), setup diagnosis | Generic segmented sum | Parity within 0.33 % (local), 0.16 % floor (murphy) |
| 2026-10-19 | FM phase 1b | ASE shim overhead, software block | Both fixes | – |
| 2026-10-20 | FM phase 2 and round 1 | 2-T13 dynamic edges (eval), 2-T19 setup, 2-T20a breakdown; OPT-T01 tensor-core survey, OPT-T01b overlap, OPT-T02 A–E, OPT-T03, OPT-T06, OPT-T07 | `gated_rows`, `segment_gemm` (owner, night); template cache; persistent worker | Before round 1: local 3.71 / – / 27.43 ms |
| 2026-10-21 | Round 1 close, round 2, round 3 | OPT-T03b, OPT-T04, OPT-T08, OPT-T09; OPT2-T01 to T05; OPT-T10; OPT3-T01 to T05 | OPT-DEF-2 (species tables, two folds, tiles, overlap rebuild); fused integrator (owner "Yes") | After round 2: local 3.028 / 18.788 / 21.434 ms (1000 K) |
| 2026-10-22 | FM phase 3 | Fixcm projection in the fused stage, rule 11, compiled tier, species-change cost | Rule 11 (scoped), fixcm (decision 37), compiled tier rules (decision 38) | – |
| 2026-10-23 | FM phases 3b–5 | GPU D3 cost, fused NHC/MTK, adaptation, calibration; CUTLASS 3×TF32 and BF16×6 on Colab | Fused NHC/MTK; adaptation (decisions 52–53); calibration opt-in (55) | – |
| 2026-10-24 | Round 4 | Radial write-bound kernel, PolarMACE fx/graph sums, 3×TF32 record, grouped launch on generic, dynamic edges in MD, compiler caches, OEq launch shape, megakernel gate, precision screen | Decisions 57, 58, 59; compiler caches; `tf32x3_gemms` never automatic | Round-4 baseline local 3.083 / 18.409 / 21.928 ms (1000 K, untimed) |
| 2026-10-25 | FM 1.0 packaging | Colab L4 table, TorchSim comparison | – | L4 3.95 / 31.5 / 35.2 ms |
| 2026-10-26 | Round 5 | Scheduler isolation, DeepSeek split, radial `64 + tail`, gated segment GEMM, tf32x3 tile rule, 3×FP16 (probe, standalone, fused, decoupled, other GPUs), delta GEMM, channel split, cuBLASLt, GEMMul8, receiver backward, cuBLASDx, two streams; ALCHEMI | tf32x3 tile rule (inside the opt-in record) | – |
| 2026-10-27 | Round 6, float64 package, round 7, round 8, ideas | L2 chunks, chunked backward, radial tables; float64 emulation, tables, folds, double-float, INT8 slices; LunaTP, atomics, traffic, wide loads; Sobek, full fusion, fused folded forward; ideas F0–F5, D4, D5, SVD | Decisions 60, 61, 62, 63, 64 | Local 2.654 / 14.56 / 18.67 ms (float32); float64 NaCl 81.9 → 46.9 ms |
| 2026-10-28 | Ideas round close | D1 (56 bits), D2 `w_tables`, D3 `save_transcendentals`; D1 suspended on a branch; radial-table cache on a branch | Decision 65 | Float64 package + D1–D3 0.78–0.81 of the package |

## Closed directions

Do not retry these unless the stated condition changes.

| Direction | Evidence | Reopen only if |
| --- | --- | --- |
| 16-bit storage or arithmetic on the force path (FP16/BF16 per-edge weights, per-stage bf16/fp16, delta GEMMs, channel splits) | [prec-fp16-edge-weights](#prec-fp16-edge-weights) fails by 3–110×; [prec-precision-screen](#prec-precision-screen) fp16 misses by 2–1,100×, bf16 by 7 to > 30,000×; [prec-delta-gemm](#prec-delta-gemm), [prec-channel-split](#prec-channel-split) 411–6,130× worse | the accuracy gate changes |
| TF32 (1×) anywhere on the force path | [prec-tf32](#prec-tf32): force error 1,020–1,500× (NaCl); SPEC §1.2 never default | never (owner rule) |
| 3×TF32 GEMMs on sm_89, sm_120 and A100 via Triton | [prec-tf32x3-triton](#prec-tf32x3-triton): slower on 89 of 92 shapes (sm_89), 0 of 92 faster (A100) | the A100 or another GPU with TF32 ≥ 3× FP32 becomes a production target (then CUTLASS stage 2, [prec-cutlass-a100](#prec-cutlass-a100)) |
| 3×FP16 / hybrid SIMT + MMA GEMMs on consumer GPUs | [prec-fp16x3-other-gpus](#prec-fp16x3-other-gpus) 1.12–1.71× of cuBLAS; [prec-hybrid-gemm](#prec-hybrid-gemm), [prec-decoupled](#prec-decoupled), [prec-packed-b](#prec-packed-b) | a GPU with BF16/FP16 dense ≥ 6× FP32 and a kernel better than Triton's |
| Heterogeneous megakernel / co-execution | [launch-het-megakernel](#launch-het-megakernel) gate fails on sm_120 (gain 1.04) and A100 (0.92); [launch-sched-iso](#launch-sched-iso) | a GPU where TF32 MMA exceeds FFMA rate and the gate passes |
| Complete force megakernel, persistent scheduling, two-stream branches | [launch-megakernel](#launch-megakernel) ceiling 8–16 % NaCl, 1–2 % NbCl5; [launch-cublasdx](#launch-cublasdx) primitive 1.062× slower; [launch-two-streams](#launch-two-streams) | graph gaps grow well above 5 % of a step |
| Inductor programmatic dependent launch | [launch-pdl](#launch-pdl): wrong forces in captured-graph replays (ratios 1e4–1e5) | never enable (SPEC §6.2) |
| Inductor combo kernels, aggressive fusion, `decompose_mm_pass`, `max-autotune`, GEMM autotuning | [launch-combo-kernels](#launch-combo-kernels) NbCl5 +2.4 to +2.7 %; [rad-decompose-mm](#rad-decompose-mm); [setup-max-autotune](#setup-max-autotune) fails with outer graphs; [lin-gemm-autotune](#lin-gemm-autotune) slower, 141–756 s setup | per-shape autotuning becomes acceptable (production rule excludes it) |
| Own FP32 GEMMs, split-K, concurrent l-branches, cuBLASLt, cuBLASDx, hand-written CUDA GEMM | [lin-split-k](#lin-split-k), [lin-concurrent-branches](#lin-concurrent-branches), [lin-cublaslt](#lin-cublaslt), [lin-cublaslt-heuristic](#lin-cublaslt-heuristic), [rad-cuda-gemm](#rad-cuda-gemm), [lin-faster-fp32](#lin-faster-fp32) | a new FP32 kernel source beats cuBLAS at 4096³ |
| Full fused convolution forward + backward (radial weights built in registers) | [conv-fused-radial-conv](#conv-fused-radial-conv) (gate 1 missed by far); [conv-full-fusion](#conv-full-fusion) 0.80–0.83 against the 0.75 stop rule; [conv-own-triton-conv](#conv-own-triton-conv) | the backward no longer has to rebuild `w` (for example tensor-core w-tiles above 3 %) |
| Low-rank radial or atom-linear bases | [rad-low-rank](#rad-low-rank) rank 65 of 65; [lin-svd](#lin-svd) rank 509–512 of 512 | a different model architecture |
| Radial MLP fusion as an opaque op | [rad-fused-radial-mlp](#rad-fused-radial-mlp), [rad-radial-entry](#rad-radial-entry); [rad-fused-radial-hidden](#rad-fused-radial-hidden) | moot under `radial_tables` (the hidden layers are gone) |
| Write-bound or split kernels for the folded radial forward | [rad-write-bound](#rad-write-bound), [rad-radial-last-gemm](#rad-radial-last-gemm), [rad-k64-addr](#rad-k64-addr); superseded by [conv-fused-folded-forward](#conv-fused-folded-forward) | – |
| Receiver-owned or source-owned streaming backward; LunaTP | [conv-receiver-backward](#conv-receiver-backward) +4.5 to +10.6 %; [conv-opt7-atomics](#conv-opt7-atomics) ≤ 5 %; [ext-lunatp](#ext-lunatp) 1.39–1.53× slower | a data-centre GPU (H100-class bandwidth balance) becomes a target |
| OEq launch-shape tuning, L2 hints, streaming backward register/block tuning | [conv-oeq-launch-shape](#conv-oeq-launch-shape) ceiling 0.26 %; [conv-bwd-variants](#conv-bwd-variants); [conv-oeq-schedules](#conv-oeq-schedules) | – |
| Symmetric contraction rewrites (OEq contraction, reduced CG, fused input, TF32 backward, emulated GEMM) | [sc-opt3-t02](#sc-opt3-t02) at 72–96 % of memory SOL; [sc-oeq-contraction](#sc-oeq-contraction) 27–79× slower; [sc-emulated-gemm](#sc-emulated-gemm) | cuEquivariance changes its kernel |
| Fused gate VJP, gate prologue/epilogue fusions | [elt-gate-vjp](#elt-gate-vjp) 0.6–1.1 % ceiling; [elt-gate-prologue](#elt-gate-prologue) NbCl5 +1.6 to +2.0 %; [elt-gated-segment-gemm](#elt-gated-segment-gemm) | – |
| Edge-bucket width and skin re-tuning | [eng-bucket-width](#eng-bucket-width), [eng-skin-rescreens](#eng-skin-rescreens): all within noise | – |
| Disconnected-graph batching and concurrent crystals | [launch-batching](#launch-batching) 0.5–2.3 %; production runs crystals sequentially (user) | the production rule changes |
| Float64 by double-float, INT8 slices for float32, 40/48/52 mantissa bits, performant strategy | [prec-double-float](#prec-double-float) 1.94× (bar 2.0×); [prec-int8-f32](#prec-int8-f32); [prec-emu-40](#prec-emu-40), [prec-d1-48](#prec-d1-48), [prec-emu-performant](#prec-emu-performant) | INT8:FP32 rate ≥ 10× (data-centre GPUs) |
| Speculative AC+B graph | [launch-speculative](#launch-speculative) slower with all gates passing; `pipelined` won | – |

## Open or untried

| Item | State | Where |
| --- | --- | --- |
| D1 suspension (`fp64_emulation_mantissa_bits` auto = dynamic everywhere) and the small-product fix | On branch `fastmace/fp64-d1-relax-fix`; its GPU checks are open (ResearchModule AGENTS.md) | [prec-d1-relax-fix](#prec-d1-relax-fix) |
| Radial-table cache under `FASTMACE_CACHE/radial-tables` | On branch `fastmace/radial-table-cache`, not merged; no timing beyond the commit message | [rad-table-cache](#rad-table-cache) |
| `save_transcendentals` in float32 | Never measured | [prec-save-transcendentals](#prec-save-transcendentals) |
| F5 incremental neighbour list; O(N) rebuild for large cells | Not screened; future work (owner, 28 October 2026) | [eng-incremental-nl](#eng-incremental-nl), [eng-on-nl](#eng-on-nl) |
| Wide `float2` loads in the streaming backward | Eligible, documented, not built (owner, 27 October 2026); build recipe in the round 7 report | [conv-wide-loads](#conv-wide-loads) |
| OPT-T05 regional persistence | Close-out recommended in opt-4 §2; no owner decision recorded | [launch-regional-persistence](#launch-regional-persistence) |
| CUTLASS 3×TF32 stage 2 | Not built; waits for the A100 to become a production target | [prec-cutlass-a100](#prec-cutlass-a100) |
| Round 7 T01(b) source-owned backward and the T02 hand-off | Not built (stop rule); the `w`/`dw` lever was handed to the post-radial-tables work, which became round 8 | [conv-source-owned](#conv-source-owned), [conv-opt7-traffic](#conv-opt7-traffic) |
| A100 float32 numbers for the current default | Not recorded; A100 numbers exist only for OPT-T08 (0.896–0.907) and float64 | [Current speed reference](#current-speed-reference) |
| NaCl +0.8 % under `chunked_conv_backward` | Not profiled | [conv-chunked-nacl](#conv-chunked-nacl) |
| Quarter-L2 chunk rule on sm_89 | No dedicated check | [conv-chunked-ada](#conv-chunked-ada) |
| OMOL-0 XL `radial_tables` self-check failure; `tools/ab_rewrites.py` does not re-plan by rule 13 | Owner decision open | [rad-tables-omol](#rad-tables-omol) |
| `direct_node_feats` and `folded_constants` defaults | Owner call never taken (exact, no resolvable gain) | [elt-direct-node-feats](#elt-direct-node-feats), [rad-folded-constants](#rad-folded-constants) |
| `step_graph` murphy A/B | Never run; the report said drop it if neutral there | [launch-step-graph](#launch-step-graph) |
| `selfcheck_cache` field | Not built; whether the field is removed or stays inert is not recorded | [setup-selfcheck-cache](#setup-selfcheck-cache) |
| D2/D3 by measured FP64 rate; w-table pieces r ≥ 1 Å | Not built | [rad-w-tables-auto](#rad-w-tables-auto), [rad-w-tables-r1](#rad-w-tables-r1) |
| `w_tables` memory for more species | Refused above 2 GiB (about 4 species) | [rad-w-tables](#rad-w-tables) |
| Wide loads `double2` variant for float64 | Untested | [conv-wide-loads](#conv-wide-loads) |
| Gate VJP real-kernel test, symmetric contraction fused input | Not built (ceilings below 3 %) | [elt-gate-vjp](#elt-gate-vjp), [sc-opt3-t02](#sc-opt3-t02) |
| cuBLAS memsets, per-step constants, eager zero-fills | "For a later round" | [launch-memsets](#launch-memsets) |
| Carried-over rewrites re-measured one at a time in FastMACE | Never done; gains are research-engine numbers | defect 22 |
| `max-autotune` coraline check (ResearchModule roadmap item 6) | Listed open, superseded by the 12 October rejection | [setup-max-autotune](#setup-max-autotune) |

## Appendix: Entries

Each entry has the same fields. "RE" numbers were measured on the research engine (local RTX 5060 Ti unless
stated), not in FastMACE. Paths beginning `LT/` are `ResearchModule:Experiments/Langevin_Thermostat/`.

### Atom-level linears and GEMMs

<a id="lin-block-linear"></a>
#### lin-block-linear: block linear layers
- Switch: RE `--block-linear` / `BLOCK_LINEAR` (`_BlockLinear`); FM `RewriteConfig.block_linear` (`on_for_hybrid_graph_engine`)
- Engine: both
- Date/round: RE third round, 14 October 2026; FM 1a-T06
- What: cuEq `Linear` (internal shared weights) replaced by per-l (per-irrep) GEMMs with extracted weights and a contiguous transposed backward.
- Status: DEFAULT (RE `auto` = on for hybrid since 14 October 2026; FM default).
- Measured: RE NaCl 4.386 → 3.870 ms (11.8 %), NaNbO3 25.83 → 25.44 (1.5 %), NbCl5 31.01 → 30.74 ms (0.9 %). Not re-measured alone in FastMACE.
- Why: per-l GEMMs avoid cuEq's naive `Linear` path.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §4b; [switch-map](design/switch-map.md) §1; SPEC §6.2.

<a id="lin-grouped-linear"></a>
#### lin-grouped-linear: grouped block linear
- Switch: RE `--grouped-linear` (`GROUPED_BLOCK_LINEAR`); FM `RewriteConfig.grouped_linear` (`on_with_block_linear`)
- Engine: both
- Date/round: RE 16 October 2026 (user decision); FM 1a-T06
- What: same-irrep input segments in one product.
- Status: DEFAULT.
- Measured: RE murphy, with `merged_residual`: MH-1 NaCl 3.68 → 3.55 ms (−3.5 to −4.6 %), NaNbO3 26.50 → 25.74 (−2.3 to −2.9 %), NbCl5 31.91 → 30.54 (−4.2 %); OMOL-0 −3.7 to −5.1 %. SPEC §6.2: MH-1 −2 to −5 %, OMOL −4 %. FM 2-T20a attributes to it and `merged_residual`: atom-level elementwise −0.05 ms NaCl, −1.03 ms NbCl5 (−12 %), atom GEMMs −0.07 / −0.29 ms against the RE's 15 October breakdown (different days, not an A/B).
- Why: fewer, larger products.
- Sources: `ResearchModule:LT/studies/other_models/README.md` (OMOL-0 optimization, item 2); `ResearchModule:LT/docs/roadmap.md` item 12; SPEC §6.2; [2-T20a](validation/2-T20a-step-breakdown.md) §2.

<a id="lin-merged-residual"></a>
#### lin-merged-residual: merged residual linear
- Switch: RE `--merged-residual` (`MERGED_RESIDUAL_LINEAR`); FM `RewriteConfig.merged_residual` (`on_with_block_linear`)
- Engine: both
- Date/round: RE 16 October 2026; FM 1a-T06
- What: `linear_1` and `linear_res` as one product per output segment (needs `density_first`).
- Status: DEFAULT (acts only where `density_first` applies).
- Measured: RE "included above" in `grouped_linear`'s −2 to −5 %.
- Why: one product instead of two.
- Sources: SPEC §6.2; [switch-map](design/switch-map.md) §1; see [lin-grouped-linear](#lin-grouped-linear).

<a id="lin-unfuse-addmm"></a>
#### lin-unfuse-addmm: unfused full-matrix addmm
- Switch: RE `gpu_langevin.UNFUSE_FULL_ADDMM = True`; FM `EngineConfig.unfuse_full_addmm` (`True`)
- Engine: both
- Date/round: RE 8 October 2026; FM 1a-T01/T11
- What: keeps Inductor (torch 2.14) from fusing `mm + full-matrix add` into `addmm`, which cost two 31.6 MB DtoD copies per step.
- Status: DEFAULT.
- Measured: coraline FP32 11.00/11.08 (on) vs 11.37/11.41 ms (off); FP64 123.7 vs 124.3–124.4 ms; local alternating A/B 3.7 % lower FP32 NaCl step. SPEC §6.2: "avoids 2 × 32 MB copies per step". Not measured in FastMACE.
- Why: the fused addmm forced copies.
- Sources: `ResearchModule:LT/docs/configuration/workstations.md`; `ResearchModule:LT/docs/roadmap.md` item 6; `ResearchModule:LT/studies/reproduction/report.md`; SPEC §6.2.

<a id="lin-inplace-block-linear"></a>
#### lin-inplace-block-linear: in-place / strided-batched block linear
- Switch: RE `INPLACE_BLOCK_LINEAR` (screen); FM `RewriteConfig.inplace_block_linear` (`False`; `True` refused, `fastmace/planner/fixed.py:722`)
- Engine: both
- Date/round: RE third round 14 October 2026 (strided batched GEMMs) and fourth round 15 October 2026 (in-place)
- What: block linear as strided-batched cuBLAS products, in place.
- Status: REJECTED (not carried over; the FM field exists for CLI parity only).
- Measured: third round: strided batched block-linear GEMMs 1,192 µs vs 992 µs. Fourth round: +0.10 / +0.30 / +0.24 ms (NaCl / NaNbO3 / NbCl5); rewrites.md: exact, +0.1 to +0.3 ms per step (cuBLAS batched tiles make the l = 3 product 415 vs 305 µs). A FastMACE timing screen was listed as 2-T20 candidate 2; no result found.
- Why: slower.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md`; `2026-10-15-fourth-round.md` §1; [rewrites](rewrites.md); [switch-map](design/switch-map.md) §1; [phase-1a plan](../plans/phase-1a.md); [phase-2 plan](../plans/phase-2.md) 2-T20.

<a id="lin-grouped-node-gemms"></a>
#### lin-grouped-node-gemms: grouped node GEMMs
- Switch: none
- Engine: RE
- Date/round: third round, 14 October 2026; repeated in docs/rewrites.md
- What: group node-level GEMMs.
- Status: NOT BUILT.
- Measured: ceiling ≈ 0.10 ms on NaCl.
- Why: ceiling too small.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §2, §5; [rewrites](rewrites.md).

<a id="lin-cueq-fused-tp"></a>
#### lin-cueq-fused-tp: cuEquivariance `fused_tp` for linear layers
- Switch: probe (`method="fused_tp"`)
- Engine: RE
- Date/round: 7–8 October 2026
- What: cuEq's fused tensor-product method for the linears.
- Status: REJECTED.
- Measured: 24.6 → 24.4 ms per force call; under compilation already fused.
- Why: no gain.
- Sources: `ResearchModule:LT/docs/configuration/local.md`; `ResearchModule:LT/studies/optimization_history/report.md` §8.

<a id="lin-magma"></a>
#### lin-magma: MAGMA GEMM diagnosis (item 10, idea 2)
- Switch: none
- Engine: RE
- Date/round: 10 October 2026
- What: diagnose the MAGMA GEMMs seen in the hybrid profile.
- Status: MEASUREMENT.
- Measured: node-level equivariant linears, inherent FP32 compute; nothing built.
- Why: –
- Sources: `ResearchModule:LT/studies/further_optimizations/README.md`.

<a id="lin-gemm-autotune"></a>
#### lin-gemm-autotune: GEMM-only and broad Inductor autotuning (V2)
- Switch: `--gemm-autotune` (probe), `max_autotune_gemm`
- Engine: RE
- Date/round: 9 October 2026 (V2); re-screened on the hybrid default 10 October 2026
- What: Inductor autotuning of GEMMs only, or broadly.
- Status: REJECTED (10 October: slower on the factored default; no lookup table allowed).
- Measured: V2 (cuEq FP32 force replay) GEMM-only +4.981 to +5.723 % (gain); broad over GEMM-only +0.040 / +0.114 %; first execution 606–627 s vs 31–35 s; repayment 119,446–121,067 NaNbO3 replays. On the 10 October hybrid default: NaCl −0.1 %, NaNbO3 −0.7 %, NbCl5 −1.0 % (slower); setup 140.7–756.3 s. FM rewrites.md: −0.1 to −1.0 % (slower), setup 141–756 s.
- Why: no gain on the rewritten model; per-shape tuning excluded by the production rule.
- Sources: `ResearchModule:LT/studies/optimization_v2/report.md` (Compiler policies); `ResearchModule:LT/studies/further_optimizations/README.md` §3; [rewrites](rewrites.md).

<a id="lin-gated-rows"></a>
#### lin-gated-rows: `gated_rows` (OPT-T02 stage A)
- Switch: `RewriteConfig.gated_rows` (`on_with_merged_residual`)
- Engine: FM
- Date/round: 20 October 2026, round 1
- What: keeps MH-1's residual products, gate and `linear_2` as per-irrep rows `[atoms·(2l+1), channels]`, removing the `[atoms, 9728]` join and the slice copies around MACE's `GatedEquivariantBlock`.
- Status: DEFAULT (scoped: MH-1 class with `merged_residual` and `GatedEquivariantBlock`; float32 and float64, exact). Owner decision 20 October 2026 night, SPEC §19 twelfth amendment; no §18 number.
- Measured: b/a murphy NaCl 0.957 / 0.954, NbCl5 0.909 / 0.911 (−4.4 %, −9.0 %); local (clean rerun, pinned `85d741d`) NaCl 0.961 / 0.965, NbCl5 0.895 / 0.893. Murphy NbCl5 31.40 → 28.55 ms. Bitwise on CPU float64 and in the reproducible GPU mode. Force gate NaCl 1.52 / 1.16 / 0.68, NbCl5 0.23 / 0.99 / 0.78. Design estimate 5–7 % NbCl5. CONFLICT (rounding): plan, SPEC §6.2 and rewrites.md give local −3.7 % NaCl / −10.6 % NbCl5; the report gives −3.5 to −3.9 % / −10.5 to −10.7 %.
- Why: exact and a clear gain on both GPUs.
- Sources: [opt-T02a](validation/opt-T02a-gated-rows.md); [opt-1](validation/opt-1.md); [opt-1 plan](../plans/opt-1.md); [rewrites](rewrites.md); SPEC §6.2, §19; [switch-map](design/switch-map.md).

<a id="lin-segment-gemm"></a>
#### lin-segment-gemm: `segment_gemm` (OPT-T02 stage B)
- Switch: `RewriteConfig.segment_gemm` (`on_with_gated_rows_float32_unless_reproducible`)
- Engine: FM
- Date/round: 20 October 2026, round 1
- What: `linear_2`'s per-l products write straight into the joined `[atoms, 16·512]` output via a gathered-row Triton IEEE GEMM (`fastmace/kernels/triton/segment_gemm.py`) where the shared shape rule picks it (rows ≥ 2 waves of 64-row tiles; 64×64×32, 4 warps, 3 stages), cuBLAS otherwise; `reshape_irreps` becomes a view (`IrrepsView`); NaCl-sized chains skip the joined op.
- Status: DEFAULT (scoped: MH-1 class, float32, not reproducible; float64 and `reproducible=True` keep the stock path). Owner decision 20 October 2026, §19 twelfth amendment.
- Measured: b/a on top of A: murphy NbCl5 0.9517 / 0.9479, NaCl 0.9972 / 0.9961; local NbCl5 0.9577 / 0.9567, NaCl 0.9974 / 0.9916. NbCl5 murphy 28.53 → 27.15 ms, local 24.60 → 23.56 ms. Tile within 2–5 % of best of five per product. Force gate NaCl 1.29 / 1.13 / 0.72, NbCl5 0.24 / 1.01 / 0.78. Without `IrrepsView` (`6bd203d`) NbCl5 1.000 / 1.000.
- Why: rounding class but deterministic; clear NbCl5 gain, NaCl neutral.
- Sources: [opt-T02b](validation/opt-T02b-segment-gemm.md); [BC go/no-go](design/opt-T02-BC-go-no-go.md); [opt-1](validation/opt-1.md); [rewrites](rewrites.md).

<a id="lin-t02-combined"></a>
#### lin-t02-combined: round 1 end-to-end (A + B) against `fastmace-1b`
- Switch: none
- Engine: FM
- Date/round: 21 October 2026 morning (combined acceptance item 8)
- What: frozen main `4f90a92` against `fastmace-1b`, plans differing only in `gated_rows` and `segment_gemm`.
- Status: MEASUREMENT.
- Measured: local NaCl −3.6 to −4.0 %, NbCl5 −14.6 to −14.9 %; murphy NaCl −3.7 to −4.1 %, NbCl5 −13.5 to −13.6 %. Earlier estimate from separate A/Bs: murphy NaCl ≈ −4.7 %, NbCl5 ≈ −13.6 %; local ≈ −4.2 %, ≈ −14.4 %. CONFLICT/stale: [opt-1](validation/opt-1.md) still calls it "an estimate … pending". Phase 2 item 8 reports the same: 3.6–4.0 % / 14.6–14.9 % locally, 3.7–4.1 % / 13.5–13.6 % on murphy.
- Why: –
- Sources: [opt-1 plan](../plans/opt-1.md) "Morning checkpoint"; [opt-1](validation/opt-1.md); [phase-2](validation/phase-2.md).

<a id="lin-fusion-mechanisms"></a>
#### lin-fusion-mechanisms: alternative fusion mechanisms (OPT-T02 design)
- Switch: none
- Engine: FM
- Date/round: 20 October 2026, round 1
- What: fuse gate, product and residual into GEMMs by Inductor mm templates under `max_autotune`, cuBLASLt bias/activation epilogues, CUTLASS example 36 + epilogue visitor trees, or cuBLASDx device GEMMs in NVRTC.
- Status: NOT BUILT (dropped on paper).
- Measured: not measured.
- Why: Inductor needs per-shape autotuning (SPEC §9.1; 140–756 s setup); cuBLASLt has no sigmoid-from-other-operand or gathered rows; CUTLASS was a fallback only if Triton could not match cuBLAS; cuBLASDx needs MathDx headers (environment change).
- Sources: branch fastmace/opt-T02: docs/design/opt-T02-fusion.md §3; [opt-1 plan](../plans/opt-1.md).

<a id="lin-concurrent-branches"></a>
#### lin-concurrent-branches: per-l products on concurrent graph branches (OPT-T03 candidate 1)
- Switch: none (bench `tools/block_linear_concurrency_bench.py`)
- Engine: FM
- Date/round: 20 October 2026, round 1
- What: a block linear's independent l-products on 2 or 4 streams forked and joined inside the captured force graph.
- Status: REJECTED (not built into the model).
- Measured: murphy, one MH-1 block linear's graph replays: 2 streams 0.94× / 1.11× (NaCl forward / input gradient), 4 streams 0.85× / 1.01×; NbCl5 0.95–1.02×. Local 0.68–0.98×. Design expectation −0.4 to −0.6 ms (10–15 %) NaCl.
- Why: branches cost more than they overlap.
- Sources: branch fastmace/opt-T03: docs/design/opt-T03-small-m.md §2, §6; [rewrites](rewrites.md); [opt-1](validation/opt-1.md).

<a id="lin-grouped-launch-v1"></a>
#### lin-grouped-launch-v1: `grouped_launch`, first version (`cd71b5c`)
- Switch: `RewriteConfig.grouped_launch`
- Engine: FM
- Date/round: 20–21 October 2026, round 1, OPT-T03 candidate 2
- What: all small l-products of a block linear in one Triton IEEE launch, forward and input gradient, one flat output buffer.
- Status: SUPERSEDED by [lin-grouped-launch](#lin-grouped-launch) (`f2531cb`); first rejected in the model.
- Measured: bench murphy 1.00 / 1.16× NaCl (7 % together), 0.97 / 1.03× NbCl5; local 1.34 / 1.30× NaCl, 0.97 / 0.98× NbCl5. Full step: local NaCl 1.0179 / 1.0103, NbCl5 1.0041 / 0.9971; murphy NaCl 1.0697, NbCl5 1.0146 (a-b only). Force gate (murphy) NaCl 1.83 / 1.12 / 0.79, NbCl5 0.23 / 1.02 / 0.84.
- Why: mixed K = 2048 with K = 128–384 (straggler tiles where cuBLAS split K); the flat output made consumers read the whole buffer (+0.109 ms elementwise); the bench gain came from L2-warm operands.
- Sources: [opt-T03b diagnosis](design/opt-T03b-diagnosis.md) §2, §4; [rewrites](rewrites.md); [opt-1](validation/opt-1.md).

<a id="lin-grouped-launch"></a>
#### lin-grouped-launch: `grouped_launch` (OPT-T03b fix, OPT4-T04)
- Switch: `RewriteConfig.grouped_launch` (`on_with_block_linear_float32_unless_reproducible_for_generic_mace_family`)
- Engine: FM
- Date/round: 21 October 2026 (fix `f2531cb`); 24 October 2026 round 4 (OPT4-T04)
- What: separate outputs per product, and only products of the same weight shape (K, N) share a launch; others keep cuBLAS split-K.
- Status: DEFAULT (scoped: generic MACE family MP-0, MP-0b3, MPA-0, OMAT-0, MatPES; float32, not reproducible). Decision 57 (24 October 2026); §19 fifteenth amendment. MH-1, MACE-OFF, POLAR-1, OMOL-0 off.
- Measured: MH-1 (round 1): local NaCl 0.9766 / 0.9817, NbCl5 1.0044 / 0.9963; murphy NaCl 1.0046 / 1.0025, NbCl5 1.0008 / 1.0019. ncu: grouped 174 µs vs cuBLAS 169.5 µs for four 512×512 products; gaps per replay 199 → 114 µs, kernels 304 → 255. Force gate (murphy) NaCl 1.31 / 1.16 / 0.72, NbCl5 0.23 / 1.01 / 0.79. Generic (round 4): NaCl b/a local 0.946–0.955 (five labels, both orders; e.g. omat0-medium 0.9529 / 0.9509), L4 omat0-medium 0.9593 / 0.9568; NbCl5 1.0008 / 0.9984. Gate omat0-medium NaCl 1.57 / 1.14 / 0.64, NbCl5 1.38 / 1.13 / 1.03.
- Why: removed launches pay on WSL2 only for MH-1; generic 216-atom products fill few SMs (4.4 TFLOPS), one launch saves ≈ 0.08 ms/step.
- Sources: [opt-T03b diagnosis](design/opt-T03b-diagnosis.md); [opt-1](validation/opt-1.md); [opt4-T04](validation/opt4-T04.md); SPEC decision 57; [switch-map](design/switch-map.md).

<a id="lin-split-k"></a>
#### lin-split-k: split-K for small-cell atom products (OPT-T03 candidate 3, OPT2-T05)
- Switch: `RewriteConfig.split_k` (`never_automatic`, branch `fastmace/opt2-T05` only)
- Engine: FM
- Date/round: round 1 (not built, ceiling ≈ 0.1 ms NaCl, 214 → ~120 µs estimated); round 2, 21 October 2026
- What: `tile_rule.split_k_parts(m, n, k, sms)` splits a product along K into a power of two as one batched cuBLAS product plus a sum; on NaCl only the l = 0 `[216, 512] @ [512, 512]` products split (2 parts).
- Status: REJECTED (record opt-in on its branch, not on main).
- Measured: bench `[216,512]@[512,512]` cuBLAS 20.8 µs, 2 parts 18.5, 4 parts 19.4 µs; `[216,2048]@[2048,512]` 45.8 vs 56.9–62.1 µs. Local A/B (pinned `2c6db72`) NaCl 1.0028 / 1.0049 (3.1089 → 3.1175 ms), NbCl5 1.0037 (a-b only). Force gate NaCl 0.14 / 1.13 / 0.36, NbCl5 0.11 / 1.02 / 0.78. Ceiling ≈ 0.3 % of the NaCl step (plan said ~2 %). murphy not run.
- Why: the extra launch and partial-sum traffic cost about what the split saves.
- Sources: branch fastmace/opt2-T05: docs/validation/opt2-T05-split-k.md, docs/design/opt2-T05-split-k.md; branch fastmace/opt-T03: docs/design/opt-T03-small-m.md; [opt-4](../plans/opt-4.md) §2.

<a id="lin-split-k-triton"></a>
#### lin-split-k-triton: Triton split-K with an epilogue reduction
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, round 2
- What: a split-K kernel that reduces in its epilogue.
- Status: NOT BUILT.
- Measured: ceiling ≈ 0.3 % of the NaCl step.
- Why: not worth building.
- Sources: branch fastmace/opt2-T05: docs/validation/opt2-T05-split-k.md ("Not done").

<a id="lin-checkpoint-scan"></a>
#### lin-checkpoint-scan: checkpoint scan for exact skips (OPT-T04)
- Switch: none (`tools/checkpoint_scan.py`)
- Engine: FM
- Date/round: 21 October 2026, round 1 (CPU only)
- What: scan 10 benchmark and 5 MACE-OFF checkpoints for zero weight blocks, zero linear paths, zero per-species contraction blocks, unused heads.
- Status: NOT BUILT (screened; conductor 21 October 2026).
- Measured: zeros 4.9–36 % of parameters+buffers, almost all in `U_matrix_*` (93.8–99.6 % zero; cuEq already sparse); no zero linear paths or species blocks; MH-1 learned zeros 14,171 of 6.44 M (0.22 %).
- Why: the only exact skips are worth under 0.5 % together.
- Sources: [opt-T04](validation/opt-T04-checkpoint-scan.md); [rewrites](rewrites.md); [opt-1](validation/opt-1.md).

<a id="lin-zero-channels"></a>
#### lin-zero-channels: MH-1 product 0's 24 zero l = 1 channels
- Switch: none
- Engine: FM
- Date/round: 21 October 2026 (OPT-T04 finding); confirmed by the 27 October SVD probe (`products.0.linear` l = 1 rank 471 of 512)
- What: drop the 24 of 512 exactly-zero l = 1 channels before `products.0.linear`.
- Status: NOT BUILT.
- Measured: estimate under 0.1 ms per NbCl5 step; SVD probe: known zero channels, < 0.5 %.
- Why: below the noise floor; needs its own record, self-check and pins.
- Sources: [opt-T04](validation/opt-T04-checkpoint-scan.md); [ideas](../plans/ideas-20261027.md).

<a id="lin-unused-heads"></a>
#### lin-unused-heads: single-head model and unused readout heads
- Switch: RE `select_head.py` (probe); FM none
- Engine: both
- Date/round: RE 7–8 October 2026; FM OPT-T04 21 October 2026; cited in round 4 §1, 24 October 2026
- What: drop the five unused output heads (RE: a single-head MH-1 copy; FM: compute only the plan's head in the linear 512 → 6 and non-linear 512 → 96 → 6 readouts).
- Status: REJECTED (RE no gain; FM below threshold, partly absorbed by OPT-T08 C2/C7).
- Measured: RE 9.89 vs 9.93 ms, identical forces. FM estimate about 0.05 ms, 0.2 % of the NbCl5 step; round 4: the single-head checkpoint run had 5–6 fewer kernels and about 0.05 ms less (2.899 / 21.161 / 18.353 ms vs 3.083 / 21.928 / 18.409 ms, cross-process).
- Why: below noise.
- Sources: `ResearchModule:LT/docs/configuration/local.md`; `ResearchModule:LT/studies/optimization_history/report.md` §8; [opt-T04](validation/opt-T04-checkpoint-scan.md); [opt-4](../plans/opt-4.md) §1.

<a id="lin-species-tables"></a>
#### lin-species-tables: `species_tables` (OPT-T08 C1)
- Switch: `RewriteConfig.species_tables` (`on_unless_reproducible_if_model_supports`; was `on_float32_unless_reproducible_if_model_supports` at OPT-DEF-2)
- Engine: FM
- Date/round: 21 October 2026, round 1 (audit 20–21 October)
- What: species-only atom-level products (node embedding, source/target embeddings and their `ProjectedEmbedding` projections, interaction 0's `skip_tp`, `linear_up` and the `linear_res` half of the merged l0 product) evaluated once on the 89 identity rows and gathered by species each step. Also absorbs C6 (no gradient w.r.t. `node_attrs`) and the phase-1a "folded species embedding" candidate (estimate ≈ 10 µs, 0.3 % NaCl).
- Status: DEFAULT (scoped: MH-1 labels, not reproducible; float64 since decision 63). Owner decision 21 October 2026 "Make all 3 defaults" (OPT-DEF-2), §19 thirteenth amendment.
- Measured: audit estimate 4.1 % NaCl / 2.1 % NbCl5 of B (local), 3.8 % / 2.1 % (murphy). Local A/B (off first) NaCl 0.956, NbCl5 0.979. Bitwise on CPU; GPU reproducible mode forces differ by 7.9e-16 (NaCl), 5.6e-15 eV/Å (NbCl5).
- Why: removes species-only GEMMs from every step.
- Sources: [OPT-T08 audit](design/opt-T08-linear-audit.md); [opt-T08](validation/opt-T08-linear.md); [opt-def-2](validation/opt-def-2.md); [phase-1a plan](../plans/phase-1a.md).

<a id="lin-fold-product-linear"></a>
#### lin-fold-product-linear: `fold_product_linear` (OPT-T08 C3)
- Switch: `RewriteConfig.fold_product_linear` (`on_with_species_tables_and_block_linear_if_model_supports`)
- Engine: FM
- Date/round: 21 October 2026, round 1
- What: fold `products.0.linear` (W_p) into its linear consumers: `x1 @ W_c` → `SC0 @ (W_p W_c) + table(sc0 @ W_c)`; folded weights computed in float64 at apply time.
- Status: DEFAULT (scoped: MH-1 labels; off in reproducible plans and without `species_tables`; float64 since decision 63).
- Measured: audit estimate 3.0 % NaCl / 2.4 % NbCl5 (local), 2.9 % / 2.5 % (murphy). Local C1 + C3 (off first): NaCl 0.921, NbCl5 0.935 (C3 on top of C1: −3.4 % / −4.4 %). CPU float64 within 1e-12 energy, 1e-10 forces (relative).
- Why: removes two 512×512 products per direction.
- Sources: [OPT-T08 audit](design/opt-T08-linear-audit.md) §2; [opt-T08](validation/opt-T08-linear.md); [opt-def-2](validation/opt-def-2.md).

<a id="lin-fold-readout-tail"></a>
#### lin-fold-readout-tail: `fold_readout_tail` (OPT-T08 C2)
- Switch: `RewriteConfig.fold_readout_tail` (`on_with_block_linear_unless_reproducible_if_model_supports`)
- Engine: FM
- Date/round: 21 October 2026, round 1
- What: fold the last layer's linear tail (`products.1.linear`, layer-1 `skip_tp`) into the non-linear readout's first linear; keeps all 96 columns.
- Status: DEFAULT (scoped: MH-1 labels and OMOL-0; off in reproducible plans; float64 since decision 63).
- Measured: audit estimate 2.0 % NaCl / 1.2 % NbCl5 (local), 1.6 % / 1.2 % (murphy); the built record saves about 10 µs (NaCl) / 40 µs (NbCl5) less than estimated. Local A/B (off first) NaCl 0.977, NbCl5 0.991.
- Why: exact to rounding in float64, measurable gain.
- Sources: [OPT-T08 audit](design/opt-T08-linear-audit.md) §2; [opt-T08](validation/opt-T08-linear.md); [opt-def-2](validation/opt-def-2.md).

<a id="lin-opt-t08-package"></a>
#### lin-opt-t08-package: OPT-T08 C1 + C2 + C3 and the OPT-DEF-2 default change
- Switch: `species_tables`, `fold_product_linear`, `fold_readout_tail` (with OPT-DEF-2 also `folded_backward_tiles`, `overlap_rebuild`)
- Engine: FM
- Date/round: 21 October 2026, round 1 → adopted in round 2 (OPT-DEF-2)
- What: all three records together; OPT-DEF-2 carried out the owner's "Make all 3 defaults" with per-record float64/reproducible handling.
- Status: DEFAULT (owner decision 21 October 2026; §19 thirteenth amendment; no §18 number).
- Measured: local NaCl 0.908 / 0.904 (3.403 → 3.088 ms), NbCl5 0.924 / 0.928 (23.20 → 21.42 ms); Modal L4 NaCl 0.918 / 0.920, NbCl5 0.922 / 0.920; Modal A100 NaCl 0.907 / 0.896, NbCl5 0.907 / 0.906. Force gate local NaCl 0.05 / 1.13 / 0.33, NbCl5 0.19 / 1.02 / 0.65; passed on the L4. Audit estimate −9 % NaCl / −5.7 % NbCl5 of B (local), −8.3 % / −5.8 % (murphy); murphy never measured. OPT-DEF-2 gate with the new defaults (pinned `953aa72`) NaCl 0.05 / 1.16 / 0.35, NbCl5 0.19 / 1.01 / 0.65; 763 CPU tests passed.
- Why: −7 to −10 % on three GPU types.
- Sources: [opt-T08](validation/opt-T08-linear.md); [opt-def-2](validation/opt-def-2.md); [rewrites](rewrites.md); [opt-2 plan](../plans/opt-2.md); [opt-4](../plans/opt-4.md) §2.

<a id="lin-t08-c6"></a>
#### lin-t08-c6: no gradient with respect to `node_attrs` (C6)
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, OPT-T08 audit
- What: stop MACE's `node_attrs.requires_grad_(True)` path so the four `[n, 512] → 89` backward products are not computed.
- Status: NOT BUILT separately (absorbed by C1: tables take no gradient).
- Measured: estimate 0.4 % NaCl (14 µs), 0.2 % NbCl5 (53 µs).
- Why: comes with C1.
- Sources: [OPT-T08 audit](design/opt-T08-linear-audit.md) §2, §4.

<a id="lin-t08-c4"></a>
#### lin-t08-c4: one product for layer-1 `skip_tp` and `linear_up` (C4)
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, OPT-T08 audit
- What: concatenate the two weights along N.
- Status: NOT BUILT (absorbed by C2 and C3).
- Measured: estimate ≈ 0.3 % NaCl, ≈ 0.1 % NbCl5 (about 10 / 30 µs).
- Why: C2 and C3 replace both products.
- Sources: [OPT-T08 audit](design/opt-T08-linear-audit.md) §2.

<a id="lin-t08-c5"></a>
#### lin-t08-c5: no `node_feats` descriptor concatenation (C5)
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, OPT-T08 audit
- What: drop `torch.cat(node_feats_list)`, which FastMACE never reads.
- Status: NOT BUILT (absorbed by C3).
- Measured: estimate ≈ 0.1 % NaCl, ≈ 0.2 % NbCl5.
- Why: gone with C3.
- Sources: [OPT-T08 audit](design/opt-T08-linear-audit.md) §2.

<a id="lin-t08-c7"></a>
#### lin-t08-c7: unused heads of `readouts.0.linear` (C7)
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, OPT-T08 audit
- What: drop the five unused heads' columns of the 512 → 6 linear readout.
- Status: NOT BUILT (absorbed by C3's consumer fold / C2).
- Measured: estimate < 0.1 % (under 5 µs).
- Why: absorbed.
- Sources: [OPT-T08 audit](design/opt-T08-linear-audit.md) §2.

<a id="lin-t08-headslice"></a>
#### lin-t08-headslice: C2 with a one-head slice (512 → 16)
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, OPT-T08
- What: fold the readout tail for the plan's one head only.
- Status: NOT BUILT ("left for later").
- Measured: difference to the built C2 about 10 µs NaCl, 40 µs NbCl5.
- Why: needs a fixed head; small.
- Sources: [OPT-T08 audit](design/opt-T08-linear-audit.md) §2.

<a id="lin-t08-generic"></a>
#### lin-t08-generic: C1/C2 for the residual, density and OFF families
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, OPT-T08
- What: tabulate or compose the element-dependent `skip_tp` per species so C1/C2 apply beyond MH-1.
- Status: NOT BUILT.
- Measured: not measured.
- Why: those families' `skip_tp` is a tensor product with `node_attrs`; folding as built would add a product.
- Sources: [OPT-T08 audit](design/opt-T08-linear-audit.md) §3.

<a id="lin-t08-paper"></a>
#### lin-t08-paper: other linear folds checked on paper
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, OPT-T08 audit
- What: fold `linear_res` into `linear_up`; `products.*.linear` into the symmetric contraction; `linear_2` into the gate or contraction; further merges of the merged residual products; backward-only merges.
- Status: NOT BUILT (dropped on paper).
- Measured: not measured.
- Why: more arithmetic (K = 512 vs 128), non-commuting operations, nonlinear on both sides, or already merged.
- Sources: [OPT-T08 audit](design/opt-T08-linear-audit.md) §1, §2.

<a id="lin-forward-audit"></a>
#### lin-forward-audit: forward-graph / MACE linear audit (round 5)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5 §3.9(3)
- What: look for further consecutive-linear folds beyond OPT-T08.
- Status: NOT BUILT (no new fold).
- Measured: not measured.
- Why: nonlinear boundaries around `linear_2` block it.
- Sources: [opt-5](validation/opt-5.md) §3.9.

<a id="lin-faster-fp32"></a>
#### lin-faster-fp32: faster FP32 kernel for large atom GEMMs (OPT-T09 rank 8 / OPT-T10 rank 5)
- Switch: none
- Engine: FM
- Date/round: 21 October 2026
- What: a 1.2× faster FP32 SIMT GEMM for the large 512×512 shapes (10–13 TFLOPS achieved vs cuBLAS's 16.7 at 4096³).
- Status: NOT BUILT ("not recommended", "high, unlikely").
- Measured: ceiling ≈ 5 % / 8 % / 4 % (NaCl / NbCl5 / NaNbO3) if such a kernel existed.
- Why: no faster FP32 kernel found without TF32 (15 October study, OPT-T01, OPT-T03).
- Sources: [OPT-T09](design/opt-T09-step-breakdown.md) §5; [OPT-T10](design/opt-T10-step-breakdown.md) §5.

<a id="lin-layout-copies"></a>
#### lin-layout-copies: remaining layout copies (OPT3-T05)
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, round 3
- What: remove the block-linear backward row gathers and forward joins; let `joined_products`' cuBLAS fallback write via `torch.mm(out=strided slice)`.
- Status: NOT BUILT.
- Measured: NaCl row gathers 19.1 + 3.3 µs, joins 17.9 + 3.4 µs; NbCl5 33.8 / 15.8 µs; fallback copy about 26 µs (NbCl5) / 47 µs (NaNbO3).
- Why: not exact: needs a gathered-row GEMM (rounding; slower on NaCl) or a different cuBLAS kernel choice.
- Sources: [opt3-T05 design](design/opt3-T05-layout-audit.md).

<a id="lin-cublaslt"></a>
#### lin-cublaslt: cuBLASLt as preferred BLAS library
- Switch: RE `TORCH_BLAS_PREFER_CUBLASLT=1` (coraline), `PREFERRED_BLAS` / `--blas cublaslt`; FM screen `preferred_blas_library("cublaslt")`
- Engine: both
- Date/round: RE 8 October 2026 (coraline), 14 October 2026 (streamed-edges round); FM 26 October 2026 (round 5 §3.10 and OPT5-T09c)
- What: prefer cuBLASLt for the FP32 GEMMs.
- Status: REJECTED (not carried over; owner, 26 October 2026: about 1 % on some cases is not enough).
- Measured: RE coraline torch 2.14 11.29/11.33 vs 11.26/11.32 ms; torch 2.5.1 10.95/10.94 vs 10.95/10.96 ms. RE 14 October: isolated weighted 6.53 → 5.84 ms; full step NaNbO3 23.45/23.17 vs 23.21/23.44, NaCl 3.764/3.783 vs 3.896/3.869 ms (pairs in the source's order; its verdict: slower in the full step). FM MD A/B (32 MiB workspace): NaCl 3.009 → 2.974 ms (−1.2 %), NaNbO3 20.232 → 20.435 (+1.0 %); block ratios straddle 1; NaCl 8-atom and NbCl5 runs invalid. FM GEMM screen [13608]: `torch.mm` 501.0, cuBLASLt 501.9 (493.9 first run), transposed-B view 697.7 µs.
- Why: faster in isolation, no consistent gain in the step.
- Sources: `ResearchModule:LT/docs/configuration/workstations.md`; `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`; [opt-5](validation/opt-5.md) §3.10; [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6(3); [switch-map](design/switch-map.md) §7.

<a id="lin-cublaslt-heuristic"></a>
#### lin-cublaslt-heuristic: native cuBLASLt heuristic search and custom tile dispatcher (OPT5-T09i)
- Switch: none (`tools/bench/opt5_t09_cublaslt_bench.cu`)
- Engine: FM
- Date/round: 26 October 2026, round 5 §3.9(1)
- What: pick a cuBLASLt algorithm per shape.
- Status: REJECTED for production (exploratory, not integrated).
- Measured: [13608] SGEMM 505.23 µs vs one cuBLASLt candidate 427.02 µs (single non-alternating run; accuracy sampled every 997th element only).
- Why: unqualified prototype; a shape-keyed cache breaks the generality rule.
- Sources: [opt-5](validation/opt-5.md) §3.9.

<a id="lin-svd"></a>
#### lin-svd: low-rank atom-level linear layers
- Switch: none (`svd.py`)
- Engine: FM
- Date/round: 27 October 2026, ideas round
- What: truncated SVD of MH-1's atom weights.
- Status: REJECTED (falsified).
- Measured: rank 509–512 of 512 (or 255–256 of 256); `products.0.linear` l = 1 471 of 512 (known zero channels, < 0.5 %); radial layers full rank 64.
- Why: no truncated-SVD route.
- Sources: [ideas](../plans/ideas-20261027.md) "Low-rank atom-level linear layers".

### Atom-level elementwise and gates

<a id="elt-density-first"></a>
#### elt-density-first: density division before `linear_1`
- Switch: RE `--density-first` / `DENSITY_BEFORE_LINEAR`; FM `RewriteConfig.density_first` (`on_with_fold_cutoff_if_model_supports`)
- Engine: both
- Date/round: RE second round, 10 October 2026 (roadmap item 11.1); FM 1a-T07
- What: divide by the edge density before `linear_1` (bias-free `linear_1`, MH-1 class).
- Status: DEFAULT (RE hybrid default 10 October 2026, user).
- Measured: RE NaCl 5.210 → 4.915 ms (5.7 %), NaNbO3 35.041 → 33.449 (4.5 %), NbCl5 38.599 → 35.958 ms (6.8 %); 12 October ablation +0.6 % / +0.9 %. SPEC §6.2: ≈ −11 % together with `reassociate_first`. Not measured alone in FastMACE.
- Why: the division moves to the smaller tensor.
- Sources: `ResearchModule:LT/studies/further_optimizations/README.md` §6; SPEC §6.2; [switch-map](design/switch-map.md) §1.

<a id="elt-gate-vjp"></a>
#### elt-gate-vjp: fused gate/residual VJP (RE item 11.3, OPT3-T01)
- Switch: RE `force_replay_ceiling.py --variant gate`; FM none
- Engine: both
- Date/round: RE 10 October 2026, re-assessed 14 October; FM 21 October 2026, round 3
- What: one backward kernel per l-block reading grad, products and gate once and writing product and gate gradients.
- Status: NOT BUILT.
- Measured: RE ceiling 5.1 % (NaCl), 3.6 % (NaNbO3), 4.7 % (NbCl5); realistic 2–2.5 %; 14 October: the gate is already one fused Inductor kernel at about bandwidth (≈ 200 µs forward, ≈ 244 µs backward per interaction). FM (OPT-T10 trace): NbCl5 gate kernels at 380–430 GB/s; removable 55–110 MB per force call = 0.13–0.25 ms, 0.6–1.1 % of NbCl5's force call. Plan estimate 3–6 %.
- Why: already fused at DRAM bandwidth; a custom op would likely lose Inductor's neighbouring fusions (the OPT2-T02 lesson).
- Sources: `ResearchModule:LT/studies/further_optimizations/README.md` §8; `2026-10-14-third-round.md` §5; [opt3-T01](design/opt3-T01-gate-vjp-analysis.md); [opt-3 plan](../plans/opt-3.md); [rewrites](rewrites.md).

<a id="elt-gate-prologue"></a>
#### elt-gate-prologue: `gate_prologue` (OPT-T02 stage C)
- Switch: `RewriteConfig.gate_prologue` (`never_automatic`; branch `fastmace/opt-T02c` only)
- Engine: FM
- Date/round: 20 October 2026, round 1
- What: apply the sigmoid gate inside `segment_gemm`'s Triton GEMM as it loads the A rows. The design put it in the residual GEMM's epilogue; the build moved it to the prologue.
- Status: REJECTED (conductor, per rule: NbCl5 ≤ 0.985 both orders on both GPUs); code unmerged.
- Measured: local, pinned `3dc0ca5`, on top of A + B: NbCl5 1.0196 / 1.0157 (23.58 → 24.04, 23.38 → 23.75 ms), NaCl 0.9979 / 0.9970; values bitwise equal to `segment_gemm`. murphy not measured. Estimate 0.6–0.9 ms (2–3 %) NbCl5.
- Why: the per-atom gate gather in the K loop cost the GEMMs more than the elementwise kernel it removed.
- Sources: [stage C note](design/opt-T02-C-gate-prologue.md) §2–5; [opt-1](validation/opt-1.md); [rewrites](rewrites.md); [BC go/no-go](design/opt-T02-BC-go-no-go.md); branch fastmace/opt-T02c.

<a id="elt-gate-backward-epilogue"></a>
#### elt-gate-backward-epilogue: gate backward in `linear_2`'s input-gradient epilogue (stage D)
- Switch: none
- Engine: FM
- Date/round: 20 October 2026, round 1
- What: fuse the gate backward (`d pre` and the per-atom reduction `d g`) into the epilogue of B's backward GEMM.
- Status: REJECTED before building (conductor).
- Measured: not measured; estimate ≈ 1.0–1.4 ms NbCl5.
- Why: the per-atom reduction needs atomics or whole-atom tiles, breaking determinism and the float64 bitwise promise.
- Sources: [rewrites](rewrites.md); branch fastmace/opt-T02: docs/design/opt-T02-fusion.md §3, §7; [stage C note](design/opt-T02-C-gate-prologue.md) §2.

<a id="elt-density-prologue"></a>
#### elt-density-prologue: density division in the `ResidualBlockLinear` prologue (stage E)
- Switch: none
- Engine: FM
- Date/round: 20 October 2026, round 1 design
- What: move the density division into the residual GEMM's prologue without B.
- Status: NOT BUILT (absorbed: "already fused with the slice copies by Inductor; it disappears with A/B").
- Measured: estimate 0.34 ms NbCl5 (0.03 NaCl). B's design planned "the density scale in the prologue"; the built `segment_gemm` report does not mention it.
- Why: part of B / already fused.
- Sources: branch fastmace/opt-T02: docs/design/opt-T02-fusion.md §3; [BC go/no-go](design/opt-T02-BC-go-no-go.md) §4.

<a id="elt-gated-segment-gemm"></a>
#### elt-gated-segment-gemm: gate scaling in the `segment_gemm` prologue (OPT5-T04)
- Switch: none (planned `gated_segment_gemm`)
- Engine: FM
- Date/round: 26 October 2026, round 5 candidate 3 (Tacker/Libra complementary fusion)
- What: fuse gate scaling into the `segment_gemm` prologue.
- Status: CUT at the wave-1 falsifier (−3.7 % < 5 %).
- Measured: NbCl5 forward-only replay 184.33 → 177.44 µs (0.9626×, bitwise equal); NaCl 44.38 → 35.40 µs in isolation.
- Why: below the 5 % bar; the backward still needs the gated rows (round 1 stage C regressed NbCl5 +1.6 to +2.0 %).
- Sources: [opt-5](validation/opt-5.md) §1.

<a id="elt-direct-node-feats"></a>
#### elt-direct-node-feats: `direct_node_feats` (OPT3-T05)
- Switch: `RewriteConfig.direct_node_feats` (`never_automatic`; requires `fold_cutoff`)
- Engine: FM
- Date/round: 21 October 2026, round 3
- What: without LAMMPS, the folded MH-1 forward skips the `[node_feats, node_attrs]` join for `handle_lammps`, so the streaming convolution no longer copies its strided `[n, 512]` input forward and backward.
- Status: NEVER-AUTOMATIC (MH-1 family, exact); the default was left to the owner and never taken.
- Measured: jointly with `folded_constants`: NaCl pure data movement 59.4 → 47.8 µs per force call; force gate NaCl 0.05 / 1.15 / 0.32, NbCl5 0.19 / 1.00 / 0.70; local (pinned `3aff996`) NaCl 0.9921 / 0.9948, NbCl5 1.0020 / 1.0020; Modal L4 NaCl 1.0034 / 1.0005, NbCl5 1.0008 / 0.9950. Expected about 11 µs NaCl (0.4 %), 31 µs NbCl5, 50 µs NaNbO3.
- Why: exact but no gain resolved above noise.
- Sources: [opt3-T05](validation/opt3-T05-layout-audit.md); [opt3-T05 design](design/opt3-T05-layout-audit.md); [opt-3 plan](../plans/opt-3.md); [rewrites](rewrites.md).

### Radial networks and tables

<a id="rad-factor-species"></a>
#### rad-factor-species: species-factored radial input (roadmap item 9)
- Switch: RE `--factor-species` / `FACTOR_SPECIES_RADIAL`; FM `RewriteConfig.factor_species` (`on_for_hybrid_graph_engine_if_model_supports`)
- Engine: both
- Date/round: RE 9 October 2026; FM 1a-T07
- What: the species-embedding columns of the radial networks' first layer are computed per atom; the per-edge GEMM shrinks from E×1034×64 to E×10×64.
- Status: DEFAULT (part of the 9 October production decision; MH-1 class, graph engine + hybrid).
- Measured: RE local hybrid MD step NaCl 7.112 → 5.805 ms (18.4 %), NaNbO3 55.485 → 44.785 (19.3 %), NbCl5 55.447 → 47.527 ms (14.3 %); worst force ratio 1.228 / 1.048; 24/24 replays. SPEC §6.2: "part of the MH-1 package". Not measured alone in FastMACE.
- Why: the species part of the input is per atom, not per edge.
- Sources: `ResearchModule:LT/studies/species_factoring/README.md`; `ResearchModule:LT/docs/roadmap.md` item 9; SPEC §6.2.

<a id="rad-fold-cutoff"></a>
#### rad-fold-cutoff: cutoff folding (roadmap item 9a)
- Switch: RE `--fold-cutoff` / `FOLD_RADIAL_CUTOFF`; FM `RewriteConfig.fold_cutoff` (`on_with_factor_species_if_model_supports`)
- Engine: both
- Date/round: RE 9 October 2026; FM 1a-T07
- What: `(hW + b)c = [hc, c][W; b]`, one GEMM with 65 inputs; interaction forward copy hash-checked against MACE 0.3.16. Gates most later MH-1 records.
- Status: DEFAULT.
- Measured: RE factored → folded NaCl 5.807 → 5.163 (11.1 %), NaNbO3 43.432 → 35.938 (17.3 %), NbCl5 47.622 → 40.540 ms (14.9 %); with factoring 27–35 % faster than the plain hybrid. 12 October ablation: all rewrites removed +39.9 % (NaCl). SPEC §6.2 gives no number ("—").
- Why: one GEMM instead of a GEMM and a scale.
- Sources: `ResearchModule:LT/studies/species_factoring/README.md` (Follow-up: cutoff folding); roadmap item 9a; SPEC §6.2.

<a id="rad-joint-radial-input"></a>
#### rad-joint-radial-input: joint radial input (roadmap item 12)
- Switch: RE `--joint-radial-input` / `JOINT_RADIAL_INPUT`; FM `RewriteConfig.joint_radial_input` (`on_with_fold_cutoff_if_model_supports`)
- Engine: both
- Date/round: RE third round, 14 October 2026; FM 1a-T07
- What: one joint radial input instead of separate per-network inputs.
- Status: DEFAULT.
- Measured: RE NaCl 4.716 → 4.549 ms (3.5 %), NaNbO3 30.76 → 28.48 (7.4 %), NbCl5 34.38 → 32.78 ms (4.7 %). With fused folded backward and GPU neighbour list: 6.2 % (4.41 ms), 15.2 % (26.2 ms), 7.5 % (31.6 ms); SPEC §6.2 "−6 to −15 %".
- Why: one input GEMM for all radial networks.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §2–4; SPEC §6.2.

<a id="rad-fused-folded-backward"></a>
#### rad-fused-folded-backward: fused folded-layer backward (+ GENERIC_FUSED_RADIAL_BACKWARD)
- Switch: RE `--fused-folded-backward` / `FUSED_FOLDED_BACKWARD` (`langevin/skinny_linear.py`), generic `GENERIC_FUSED_RADIAL_BACKWARD`; FM `RewriteConfig.fused_folded_backward` (`on_with_stage1_fold_cutoff_if_model_supports`), one field for both
- Engine: both
- Date/round: RE 14 October 2026 (MH-1), 15 October 2026 (generic, user); FM 1a-T07/T08
- What: Triton input gradient of the folded last radial layer (skinny linear).
- Status: DEFAULT.
- Measured: RE MH-1 NaCl 4.647 → 4.618 (0.6 %), NaNbO3 30.74 → 30.31 (1.4 %), NbCl5 34.51 → 34.29 ms (0.6 %); kernel 1.15 ms vs cuBLAS 2.0 ms. Generic: −3 to −5 % on large cells. Tiling follow-up: [rad-folded-backward-tiles](#rad-folded-backward-tiles).
- Why: the skinny K = 65 product is poorly served by cuBLAS.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md`; `ResearchModule:LT/studies/other_models/README.md` (Second round); SPEC §6.2.

<a id="rad-folded-backward-tiles"></a>
#### rad-folded-backward-tiles: `folded_backward_tiles` (OPT2-T03)
- Switch: `RewriteConfig.folded_backward_tiles` (`on_with_fused_folded_backward`)
- Engine: FM
- Date/round: 21 October 2026, round 2 (OPT-T09 rank 3)
- What: the folded layer's Triton input gradient picks 128-row tiles with 8 warps where the 64-row grid has 2–8 waves over the SMs, else the fixed 64 × 32 tiles; BLOCK_K stays 32, so results are bitwise identical.
- Status: DEFAULT (OPT-DEF-2, §19 thirteenth amendment); MH-1 class (MH-1, OMOL-0, POLAR-1); exact, also in float64 and reproducible plans; in practice changes only small cells.
- Measured: kernel NaCl 170.2 → 134.8 µs (ncu, 0.79×); local A/B (pinned `27e7718`) NaCl 0.983 / 0.981 (3.109 → 3.056 ms), 12/12 pairs faster; Modal L4 NaCl 1.005 / 1.000, NbCl5 1.000 / 1.001 (`results/modal-round2-L4-1176db8/tiles-*`: 1.0053 / 0.9998, 1.0000 / 1.0006). OPT-T10: folded input gradient 0.178 → 0.124 ms on NaCl. Plan ceiling 2–4 %. murphy never measured.
- Why: exact, NaCl gain on sm_120, neutral on the L4.
- Sources: [opt2-T03](validation/opt2-T03.md); [opt-def-2](validation/opt-def-2.md); [opt-2 plan](../plans/opt-2.md); [rewrites](rewrites.md).

<a id="rad-fused-radial-hidden"></a>
#### rad-fused-radial-hidden: fused radial hidden block
- Switch: RE `--fused-radial-hidden` / `FUSED_RADIAL_HIDDEN` (`langevin/fused_radial.py`); FM `RewriteConfig.fused_radial_hidden` (`never_automatic`; needs `fold_cutoff`)
- Engine: both
- Date/round: RE 14 October 2026 (reported adopted), corrected 15 October 2026; referenced in FM round 2
- What: Triton fusion of one LayerNorm/SiLU/Linear block of MH-1's radial hidden stack.
- Status: NEVER-AUTOMATIC (RE `auto` = off since 15 October; SPEC §6.2 "opt-in, no gain"). The 14 October "adoption" was a no-op: the block never attached to MH-1.
- Measured: 14 October (invalid, identical configurations) NaNbO3 −1.2 %; fixed 15 October: NaCl −0.06 ms, NaNbO3 0, NbCl5 +0.15 ms. Ceiling (stub) 4.8 % / 4.5 % / 2.3 %. Not re-timed in FastMACE.
- Why: no gain after the fix; conflicts with `fused_radial_mlp`.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-15-step-breakdown.md`; `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`; [rewrites](rewrites.md); [switch-map](design/switch-map.md); branch fastmace/opt2-t02c: docs/design/opt2-T02-radial-mlp-fusion.md §1.

<a id="rad-tabulation-ceiling"></a>
#### rad-tabulation-ceiling: radial-network tabulation (RE ceiling)
- Switch: `force_replay_ceiling.py --variant hidden|all`
- Engine: RE
- Date/round: 10 October 2026
- What: replace the radial networks by tables (ceiling stub).
- Status: SUPERSEDED (rejected as "approximate" by the RE; reopened as [rad-radial-tables](#rad-radial-tables) on 27 October 2026).
- Measured: ceiling NaCl 4.4 %, NaNbO3 5.1 %, NbCl5 2.9 %; FM rewrites.md: "ceiling 2.9–5.1 %, approximate".
- Why: considered approximate at the time.
- Sources: `ResearchModule:LT/studies/further_optimizations/README.md` §4–5; `ResearchModule:LT/studies/radial_ceiling/` (probes only); [rewrites](rewrites.md).

<a id="rad-weight-sharing"></a>
#### rad-weight-sharing: same-species radial weight sharing
- Switch: `force_replay_ceiling.py`
- Engine: RE
- Date/round: 10 October 2026
- What: share radial weights between edges of the same species pair.
- Status: NOT BUILT.
- Measured: ceiling 4.3 %, 5.5 %, 4.2 % (NaCl / NaNbO3 / NbCl5).
- Why: not pursued.
- Sources: `ResearchModule:LT/studies/further_optimizations/README.md` §4–5.

<a id="rad-mirror-edge"></a>
#### rad-mirror-edge: mirror-edge radial sharing
- Switch: none
- Engine: RE
- Date/round: fourth round, 15 October 2026
- What: share radial work between an edge and its mirror.
- Status: NOT BUILT.
- Measured: ceiling already slower, NaNbO3 +1.1 ms.
- Why: slower at the ceiling.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-15-fourth-round.md` §5; [rewrites](rewrites.md).

<a id="rad-contiguous-folded-weight"></a>
#### rad-contiguous-folded-weight: contiguous folded weight
- Switch: `CONTIGUOUS_FOLDED_WEIGHT` (probe)
- Engine: RE
- Date/round: 14 October 2026 (twice)
- What: store the folded radial weight contiguously.
- Status: REJECTED (not carried over).
- Measured: NaCl 1.5 %, NaNbO3 0.5 %, NbCl5 −0.7 %; second A/B no gain; microbenchmark 1.86 → 1.52 ms; full step "within noise, twice".
- Why: no full-step gain.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §4b; `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`; [switch-map](design/switch-map.md) §7; [rewrites](rewrites.md).

<a id="rad-decompose-mm"></a>
#### rad-decompose-mm: Inductor `decompose_mm_pass`
- Switch: `DECOMPOSE_SMALL_MM`
- Engine: RE
- Date/round: 14 October 2026 (third round / streamed-edges round)
- What: Inductor's small-mm decomposition for the joint radial layer.
- Status: REJECTED (not carried over).
- Measured: NaNbO3 23.05/23.22 vs 23.45/23.02 ms; "fires but no gain".
- Why: no gain.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`; [switch-map](design/switch-map.md) §7.

<a id="rad-joint-fused-backward"></a>
#### rad-joint-fused-backward: Triton input gradient of the joint radial layer
- Switch: `JOINT_FUSED_BACKWARD`
- Engine: RE
- Date/round: 14 October 2026
- What: fused Triton input gradient of the joint first radial layer.
- Status: REJECTED (not carried over).
- Measured: NaCl −0.03 ms; NaNbO3 noise; microbenchmark NaNbO3 159 → 122 µs, NaCl 19 → 44 µs. switch-map: "below 0.1 ms"; rewrites.md adds "slower on small cells".
- Why: below 0.1 ms, slower on small cells.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §2, §5; [switch-map](design/switch-map.md) §7; [rewrites](rewrites.md).

<a id="rad-triton-folded-forward"></a>
#### rad-triton-folded-forward: Triton forward for the folded layer (RE)
- Switch: probe
- Engine: RE
- Date/round: 14 October 2026
- What: Triton kernel for the folded `[E, 65] @ [65, 1280]` forward.
- Status: REJECTED.
- Measured: 2.0–64 ms vs 1.9 ms cuBLAS.
- Why: slower.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §5.

<a id="rad-cuda-gemm"></a>
#### rad-cuda-gemm: hand-written CUDA GEMM for the folded forward
- Switch: probe
- Engine: RE
- Date/round: 14 October 2026, streamed-edges round
- What: own CUDA GEMM.
- Status: REJECTED.
- Measured: 2.78 ms vs 1.88 (cuBLAS transposed), 1.57 ms (contiguous).
- Why: slower than cuBLAS.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`.

<a id="rad-fused-radial-mlp"></a>
#### rad-fused-radial-mlp: radial MLP fusion, store variant (OPT2-T02)
- Switch: `RewriteConfig.fused_radial_mlp` (`never_automatic`; branches `opt2-t02` / `opt2-t02c` only)
- Engine: FM
- Date/round: 21 October 2026, round 2 (OPT-T09 rank 2)
- What: one Triton kernel per direction for the radial MLP hidden layers (LayerNorm, SiLU, hidden GEMMs) and their backward; the store variant keeps LayerNorm inputs; 32 rows / 8 warps; D = 128 refused.
- Status: REJECTED (record opt-in on its branch, not on main).
- Measured: microbench store 32×8: NaCl 86.9 → 53.5, NbCl5 517.3 → 348.7, NaNbO3 909.9 → 514.8 µs per stack. Force gate NaCl 0.09 / 1.24 / 0.38, NbCl5 0.06 / 1.01 / 0.79. Local A/B (pinned `215d8b8`) NaCl 0.985 / 0.985 (3.419 → 3.367 ms), NbCl5 1.011 / 1.009 (23.204 → 23.450 ms). Plan ceiling 4–8 %. Branch commit `deaa5f90`: "NaCl −1.5 %, NbCl5 +1.0 %: not adopted".
- Why: around the opaque op Inductor loses two fusions (species gathers + add into the first LayerNorm; last SiLU into the `[h·c, c]` cat), about 0.3 ms on NbCl5.
- Sources: branch fastmace/opt2-t02c: docs/design/opt2-T02-radial-mlp-fusion.md; [opt-4](../plans/opt-4.md) §2.

<a id="rad-fused-radial-mlp-recompute"></a>
#### rad-fused-radial-mlp-recompute: radial MLP fusion, recompute variant
- Switch: `RewriteConfig.fused_radial_mlp` (first version, branch)
- Engine: FM
- Date/round: 21 October 2026, round 2
- What: the same fusion with the backward recomputing LayerNorm inputs (32 rows, 4 warps).
- Status: SUPERSEDED by the store variant; stopped at the microbenchmark.
- Measured: per stack NaCl 86.8 → 92.1, NbCl5 539.9 → 541.7, NaNbO3 930.9 → 794.9 µs (saving per force call −11 / −4 / +272 µs); backward spilled 188 registers. Stop rule 0.15 ms on NbCl5.
- Why: register spills.
- Sources: branch fastmace/opt2-t02c: docs/design/opt2-T02-radial-mlp-fusion.md.

<a id="rad-radial-entry"></a>
#### rad-radial-entry: radial MLP entry kernels (OPT2-T02c)
- Switch: `fused_radial_mlp` with `joint_radial_input` (branch `opt2-t02c` only)
- Engine: FM
- Date/round: 21 October 2026, round 2
- What: the fused op takes the joint first layer's column view and the two species tables, gathers and adds them, runs the stack and writes `[h·c, c]` directly (16 rows, 8 warps, 4 elements per thread).
- Status: REJECTED (conductor's rule: NbCl5 not below 1.0 in both orders; NaCl not run).
- Measured: force gate NaCl 0.093 / 1.162 / 0.355, NbCl5 0.053 / 1.015 / 0.70; local A/B (pinned `2515146`) NbCl5 1.0008 / 1.0060 (23.284 → 23.302 ms). CONFLICT (rounding): [opt-4](../plans/opt-4.md) gives "1.001 / 1.006".
- Why: the stack's isolated −0.34 ms did not reach the step (open causes: 16-row backward, split first-layer reads, L2 pressure).
- Sources: branch fastmace/opt2-t02c: docs/design/opt2-T02-radial-mlp-fusion.md "OPT2-T02c"; [opt-4](../plans/opt-4.md) §2.

<a id="rad-fused-mlp-retry"></a>
#### rad-fused-mlp-retry: second radial-MLP fusion attempt
- Switch: none
- Engine: FM
- Date/round: 24 October 2026, round 4 "Not proposed"
- What: retry OPT2-T02.
- Status: NOT BUILT (dropped on paper).
- Measured: not measured.
- Why: OPT2-T02's cause (opaque op costs Inductor's neighbouring fusions) stands.
- Sources: [opt-4](../plans/opt-4.md) §3.

<a id="rad-folded-constants"></a>
#### rad-folded-constants: `folded_constants` (OPT3-T05)
- Switch: `RewriteConfig.folded_constants` (`never_automatic`; requires `fold_cutoff`)
- Engine: FM
- Date/round: 21 October 2026, round 3
- What: the folded layer's `[W, b]` `[1280, 65]` and its halves are buffers built once; the op `fastmace::folded_linear_input_grad_parts` runs the same kernel on the given halves.
- Status: NEVER-AUTOMATIC (MH-1 family, exact).
- Measured: only jointly with `direct_node_feats` (see [elt-direct-node-feats](#elt-direct-node-feats)); removes the `[1280, 65]` cat and `[1280, 64]`, `[1280]` copies (5.2–5.6 µs per force call).
- Why: exact, no resolvable gain.
- Sources: [opt3-T05](validation/opt3-T05-layout-audit.md); [switch-map](design/switch-map.md).

<a id="rad-write-bound"></a>
#### rad-write-bound: write-bound Triton kernel for the folded radial forward (OPT4-T01)
- Switch: none (`tools/radial_forward_bench.py`, `tools/radial_forward_probe.py`)
- Engine: FM
- Date/round: 24 October 2026, round 4 candidate 1
- What: one-tile-K Triton GEMM for `[E, 65] @ [65, 1280]` (Inductor pads to K = 68 for cuBLAS), IEEE FP32 `tl.dot` on 64 columns plus a rank-1 cutoff column, streamed output; 36 tile configurations, K steps 16/32/64, persistent TMA-store variant.
- Status: CUT at the falsifying microbenchmark (float32, MH-1).
- Measured: local graph replays, µs per call NaCl / NbCl5 / NaNbO3: limit ≤ 150 / ≤ 1,000 / ≤ 1,450; cuBLAS K = 68 183–186 / 1,270–1,286 / 1,897–1,913; cuBLAS K = 64 165 / 1,104 / 1,630; best Triton 199 / 1,331 / 1,969; persistent TMA 190 / 1,306 / 1,913; `fill_` 113–114 / 793–799 / 1,168–1,171. Triton `tl.dot` 7.2–7.5 TFLOPS vs 16.7 peak. Realistic ceiling 0.8 % (NbCl5) / 1.4 % (NaNbO3).
- Why: compute-bound, not write-bound; even cuBLAS at K = 64 misses the limits.
- Sources: [opt4-T01](validation/opt4-T01.md); [opt-4](../plans/opt-4.md) §3; [rewrites](rewrites.md).

<a id="rad-k64-addr"></a>
#### rad-k64-addr: cuBLAS K = 64 plus a separate rank-1 update
- Switch: none
- Engine: FM
- Date/round: 24 October 2026, OPT4-T01 sub-variant
- What: split the K = 65 product into cuBLAS K = 64 plus `addr_` for the cutoff column.
- Status: REJECTED.
- Measured: 376 / 2,650 / 3,936 µs (NaCl / NbCl5 / NaNbO3), about 2× today's 183 / 1,284 / 1,892 µs.
- Why: the rank-1 update re-reads and re-writes the whole `[E, 1280]` output.
- Sources: [opt4-T01](validation/opt4-T01.md).

<a id="rad-radial-last-gemm"></a>
#### rad-radial-last-gemm: `64 + tail` streaming-store kernel (OPT5-T03)
- Switch: none (planned `radial_last_gemm`)
- Engine: FM
- Date/round: 26 October 2026, round 5 candidate 5
- What: Triton `64 + 1` column split with `.cs` / `evict_first` streaming stores.
- Status: CUT at the wave-1 falsifier.
- Measured: Triton 160.7 / 1,440.3 / 2,435.8 µs vs cuBLAS K = 65 199.8 / 1,339.4 / 1,984.1 and the cuBLAS K = 64 floor 150.3 / 1,052.6 / 1,546.3 µs (NaCl / NbCl5 / NaNbO3); targets 150 / 1,000 / 1,450.
- Why: even cuBLAS at K = 64 misses the targets (repeats OPT4-T01).
- Sources: [opt-5](validation/opt-5.md) §1.

<a id="rad-stub-ceiling"></a>
#### rad-stub-ceiling: radial tables, accuracy premise and stub ceiling (OPT6-T04)
- Switch: none (`tools/bench/opt6_radial_stub.py`)
- Engine: FM
- Date/round: 27 October 2026, round 6 (owner, from an external proposal)
- What: NumPy reproduction of MACE's radial chain, float32 Chebyshev table accuracy against the network, and a stub replacing the chain with the table op's traffic.
- Status: MEASUREMENT (passed, led to OPT6-T05).
- Measured: re-implementation equals MACE to 1.1e-14; table error / float32 network error 0.13–0.57 (24 of 24 cases). Stub per force call b/a NaCl 0.897 / 0.898, NbCl5 0.928 / 0.928, NaNbO3 0.839 / 0.843 (−0.28 / −1.50 / −3.0 ms).
- Why: answers the research engine's "approximate" rejection.
- Sources: [opt6-T04](validation/opt6-T04.md).

<a id="rad-radial-tables"></a>
#### rad-radial-tables: `radial_tables` (OPT6-T05)
- Switch: `RewriteConfig.radial_tables` (`on_with_fold_cutoff_unless_reproducible`)
- Engine: FM
- Date/round: 27 October 2026, round 6
- What: per interaction and ordered species pair, the chain r → `[h c, c]` and the edge density as degree-10 piecewise Chebyshev series fitted in float64 at setup, certified against the float32 network; one Triton op with a recomputing backward. Replaces Bessel, Agnesi, embeddings, first and hidden layers.
- Status: DEFAULT (scoped: plans with `fold_cutoff`, not reproducible; MH-1 factored class). Decision 62 (float32), decision 61 (class `certified`), decision 63 (float64, see [rad-f64-tables](#rad-f64-tables)).
- Measured: certificates 0.15–0.30; gate NaCl 0.08 / 0.98 / 0.36, NbCl5 0.11 / 1.05 / 0.41, NaNbO3 0.32 / 1.00 / 0.60; NVE drift unchanged; local b/a NaCl 0.901 / 0.898, NbCl5 0.915 / 0.916, NaNbO3 0.811 / 0.807; leto 0.914 / 0.914, 0.904 / 0.904, 0.832 / 0.831. leto default step with tables: NaCl 2.80, NbCl5 21.75, NaNbO3 18.03 ms (`0cba0d7a`).
- Why: one table lookup for most of the radial chain.
- Sources: [opt6-T05](validation/opt6-T05.md); SPEC decisions 62, 63 and §19 (61); [switch-map](design/switch-map.md). CONFLICT: [rewrites](rewrites.md) still lists it as opt-in (defect 1).

<a id="rad-values-only-fit"></a>
#### rad-values-only-fit: values-only table fit
- Switch: none
- Engine: FM
- Date/round: 27 October 2026, OPT6-T05 development
- What: fit only values and differentiate the series for r-derivatives.
- Status: REJECTED (replaced by fitting derivatives too).
- Measured: tiny CPU model certificate 1.03–1.07 (derivative of `w`) → 0.06–0.13 with a derivative tolerance.
- Why: the derivative error failed the certificate.
- Sources: [opt6-T05](validation/opt6-T05.md).

<a id="rad-tables-omol"></a>
#### rad-tables-omol: `radial_tables` self-check failure on OMOL-0 XL
- Switch: `RewriteConfig.radial_tables`
- Engine: FM
- Date/round: 27 October 2026 (found in OPT8-T02)
- What: the default plan on `omol0-xl` NbCl5 fails the `radial_tables` self-check.
- Status: PENDING (rule 13 should re-plan in production; `tools/ab_rewrites.py` does not).
- Measured: relative errors 1.4–2.2e-4 (limit 1e-4) in interactions 1 and 2.
- Why: open owner decision.
- Sources: [opt8-T02](validation/opt8-T02.md) "Decisions for the owner" 2.

<a id="rad-f64-tables"></a>
#### rad-f64-tables: float64 radial tables
- Switch: `RewriteConfig.radial_tables` (`DEGREE_64 = 16`, `TOLERANCE_64 = 3e-13`, `CERTIFY_BOUND_64 = 1e-12`, `R_LO_64 = 0.25` Å)
- Engine: FM
- Date/round: 27 October 2026, f64-T01 §4
- What: tables in the model's dtype; derivative series fitted from exact network derivatives; fixed-bound certificate.
- Status: DEFAULT (scoped: float64 with `fold_cutoff`, not reproducible). Decision 63(d).
- Measured: step 0.90 (NaCl), 0.91 (NbCl5); certificates 0.17–0.52 at 3e-13 (1e-12 tolerance failed 1.20–1.96); forces 4.1e-15 / 2.9e-14 against native.
- Why: removes the radial hidden layers' share (9 % / 7 %).
- Sources: [f64-T01](validation/f64-T01.md) §4; SPEC decision 63.

<a id="rad-f64-r03"></a>
#### rad-f64-r03: 0.3 Å lower bound in float64 tables (bug)
- Switch: constant `R_LO_64`
- Engine: FM
- Date/round: 27 October 2026, f64-T01 §4
- What: Triton passes scalar bounds as float32; 0.3 rounds to 0.30000001, so an edge at exactly 0.3 Å fell below the domain.
- Status: SUPERSEDED (by 0.25 Å, exact in float32).
- Measured: certificate 3e10 on CUDA vs 0.29 on CPU.
- Why: correctness.
- Sources: [f64-T01](validation/f64-T01.md) §4.

<a id="rad-table-cache"></a>
#### rad-table-cache: cache of fitted and certified radial tables
- Switch: none (cache under `FASTMACE_CACHE/radial-tables`)
- Engine: FM
- Date/round: 28 October 2026
- What: `tabulate_radials` stores the tables under a key of model state, species set, dtype, r_max, fit constants and module source (atomic writes) and reuses them; the self-check still runs on cached tables; a damaged entry is a miss.
- Status: ON BRANCH (`fastmace/radial-table-cache`, commit `54b6f13f`; not merged into main `8a269a91`).
- Measured: commit message: half of a float64 MH-1 build was fitting and certifying the tables (trillian profile: 42 of 87 s, certify 30.5 s). Cached build time not recorded.
- Why: setup time.
- Sources: branch fastmace/radial-table-cache: commit `54b6f13f` (`fastmace/rewrites/radial/__init__.py`, `tests/cpu/test_radial_table_cache.py`).

<a id="rad-w-tables"></a>
#### rad-w-tables: `w_tables` (ideas D2)
- Switch: `RewriteConfig.w_tables` (`on_for_mh1_float64_on_measured_gpus_decision_65`; needs `radial_tables`, `chunked_conv_backward`, ≤ 3 species; `W_TABLE_MAX_BYTES` 2 GiB)
- Engine: FM
- Date/round: 27–28 October 2026, ideas round
- What: fold `W` into each table piece's Chebyshev coefficients so `w(r)` and its r-derivative are evaluated directly (17 terms), skipping `w = a Wᵀ` and `da = dw W`; edges sorted by (pair, piece).
- Status: DEFAULT (scoped: float64 MH-1, not reproducible, on "NVIDIA GeForce RTX 5060 Ti" and "NVIDIA RTX 4000 Ada Generation" only). Decision 65(b). Rounding, non-deterministic.
- Measured: same-process b/a local NbCl5 0.911 / 0.912, NaNbO3 0.855 / 0.855, NaCl 0.942 / 0.944; murphy 0.918 / 0.918, 0.870 / 0.870, 0.944 / 0.945; A100 1.063 / 1.065, 1.103 / 1.102, 1.268 / 1.269 (slower). Self-check 7.7e-16 / 6.2e-16; tables 486 MiB (NbCl5), 1,297 MiB (NaNbO3).
- Why: float64 folded products are FP64-bound; w-tables need 3.8× fewer FMA.
- Sources: [ideas](../plans/ideas-20261027.md) D2, "Per-GPU confirmation", "Decision 65"; SPEC decision 65(b).

<a id="rad-w-tables-auto"></a>
#### rad-w-tables-auto: tie D2/D3 to the measured FP64-rate rule
- Switch: would be a rule on `kernels.triton.fp64_rate`
- Engine: FM
- Date/round: 28 October 2026, ideas "Per-GPU confirmation"
- What: enable D2/D3 where the FP64 rate is below 1/8 of FP32 instead of naming GPUs.
- Status: NOT BUILT (owner chose the exact GPU-name list, decision 65).
- Measured: not measured.
- Why: owner rule "only where measured".
- Sources: [ideas](../plans/ideas-20261027.md).

<a id="rad-w-tables-r1"></a>
#### rad-w-tables-r1: keep only w-table pieces MD reaches (r ≥ 1 Å)
- Switch: none
- Engine: FM
- Date/round: 28 October 2026, ideas D2 memory note
- What: drop pieces below 1 Å to cut table memory (supersets grow quadratically).
- Status: NOT BUILT (future note).
- Measured: not measured (memory 0.5–1.3 GB; refused above 2 GiB, about 4 species).
- Why: memory caveat.
- Sources: [ideas](../plans/ideas-20261027.md) "D2 step-level probe".

<a id="rad-low-rank"></a>
#### rad-low-rank: certified low-rank radial basis (D4, float64) and low-rank w basis (F1, float32)
- Switch: none (`wrank.py`)
- Engine: FM
- Date/round: 27 October 2026, ideas D4 and F1
- What: replace the 65-wide folded input `(a, W)` by `(b, V)` with k < 65.
- Status: REJECTED (falsified).
- Measured: interaction 1 rank 65 / 65 at 1e-7 / 1e-13 (NaNbO3, NbCl5); k = 64 truncation leaves 8e-4 / 3e-5 of output scale.
- Why: full rank.
- Sources: [ideas](../plans/ideas-20261027.md) D4, F1, "Falsified here".

<a id="rad-f4"></a>
#### rad-f4: piecewise w-tables in float32, forward only (F4)
- Switch: none
- Engine: FM
- Date/round: 27 October 2026, ideas F4
- What: D2's tables replacing the float32 cuBLAS folded GEMM.
- Status: SUPERSEDED by decision 64 (`fused_folded_forward`).
- Measured: not measured (coefficients 56 KB per piece do not fit shared memory beside the convolution).
- Why: the fused forward already removed the GEMM it targeted.
- Sources: [ideas](../plans/ideas-20261027.md) F4.

<a id="rad-f0"></a>
#### rad-f0: float32 input-gradient tiles inside the chunked backward (F0)
- Switch: none (`skinny_linear.tile_choice` / chunk rule)
- Engine: FM
- Date/round: 27 October 2026, ideas F0
- What: split-K or two chunks per side-stream launch for the `[1536,1280]×[1280,65]` input-gradient products (4.4 TFLOPS).
- Status: REJECTED (dropped at the stub ceiling).
- Measured: stub ceiling NbCl5 0.5 % (18.70/18.86 → 18.68/18.69 ms), NaNbO3 2.6 % (14.56/14.53 → 14.17/14.16 ms).
- Why: the side stream already hides the input gradient.
- Sources: [ideas](../plans/ideas-20261027.md) F0.

<a id="rad-f64-reshape"></a>
#### rad-f64-reshape: reshaped folded forward product in float64
- Switch: none
- Engine: FM
- Date/round: 27 October 2026, f64-T03
- What: transposed, column-block-split, inner dimension padded to 128.
- Status: REJECTED.
- Measured: 24.6–36.0 ms vs 25.1 ms as is.
- Why: no gain.
- Sources: [f64-T01](validation/f64-T01.md) "f64-T03".

### Interaction-2 convolution (streaming, OEq)

<a id="conv-sort-receivers"></a>
#### conv-sort-receivers: receiver-sorted edges (roadmap item 10)
- Switch: RE `--sort-receivers`; FM `NeighbourConfig.sort_receivers` (`on_for_hybrid_graph_engine`)
- Engine: both
- Date/round: RE force screen 9 October 2026, MD screen and adoption 10 October 2026; FM 1a-T09/T15
- What: candidate edges ordered stably by receiver at each rebuild (CPU sort, later a GPU key); needed by `streaming_conv` and `segmented_first_sum`.
- Status: DEFAULT (RE hybrid default 10 October 2026, user).
- Measured: 9 October force replay: cuEq NaNbO3 4.45 %, NbCl5 2.14 %; hybrid NaNbO3 1.65 %, NbCl5 3.65 %. 10 October MD on the folded hybrid: NaCl 5.179 → 5.201 (−0.4 %), NaNbO3 36.041 → 34.845 (3.3 %), NbCl5 40.621 → 38.619 ms (4.9 %). 12 October ablation: removing it NaCl −3.3 %, NaNbO3 +1.0 / +2.2 %. SPEC §6.2: −3 to −5 % on large crystals.
- Why: locality for the convolution.
- Sources: `ResearchModule:LT/studies/further_optimizations/README.md` §1; `ResearchModule:LT/studies/remaining_optimizations/report.md`; `2026-10-12-gear-switch-screen.md`; SPEC §6.2.

<a id="conv-segmented-reduction"></a>
#### conv-segmented-reduction: segmented reduction (idea 5, synthetic operator)
- Switch: probe
- Engine: RE
- Date/round: 9 October 2026
- What: segmented reduction over sorted edges.
- Status: NOT BUILT (full-model port declined).
- Measured: already sorted, combined 1.24 % (NaNbO3), 3.37 % (NbCl5); with permutation −80.05 %, −82.11 %.
- Why: the permutation cost dominates; sorting was adopted instead.
- Sources: `ResearchModule:LT/studies/remaining_optimizations/report.md` (Ideas 4–5); `ResearchModule:LT/studies/edge_ordering/README.md`.

<a id="conv-oeq-schedules"></a>
#### conv-oeq-schedules: OEq launch schedules warp4 / warp5 / blocks2 (V2)
- Switch: probe
- Engine: RE
- Date/round: 9 October 2026
- What: alternative OEq launch schedules.
- Status: REJECTED.
- Measured: warp4 NaNbO3 −2.790 / −2.035 %; warp5 −1.338 / −1.412 %; blocks2 −0.055 / +0.052 %.
- Why: slower or no change.
- Sources: `ResearchModule:LT/studies/optimization_v2/report.md` (OpenEquivariance schedules).

<a id="conv-shared-weight-ceiling"></a>
#### conv-shared-weight-ceiling: streamed-edges ceilings (steps 1–3)
- Switch: none
- Engine: RE
- Date/round: 14 October 2026
- What: ceilings for shared-weight streamed edges.
- Status: MEASUREMENT.
- Measured: shared-weight ceiling 11.4 % / 16.0 % / 9.1 %.
- Why: –
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` (Steps 1–4.1); `STREAMED_EDGES.md`.

<a id="conv-l2-chunks-re"></a>
#### conv-l2-chunks-re: L2-resident chunks with unpatched OEq (step 4.1)
- Switch: probe
- Engine: RE
- Date/round: 14 October 2026
- What: run the convolution in L2-sized edge chunks.
- Status: REJECTED (later reopened differently as [conv-opt6-t01](#conv-opt6-t01) and [conv-chunked-conv-backward](#conv-chunked-conv-backward)).
- Measured: chunked conv NaNbO3 9.79 → 13.97–18.58 ms (1.4–2× slower).
- Why: launch and copy overhead with stock OEq.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`; [rewrites](rewrites.md).

<a id="conv-own-triton-conv"></a>
#### conv-own-triton-conv: own fused Triton convolution (step 4.3)
- Switch: probe
- Engine: RE
- Date/round: 14 October 2026
- What: own Triton forward and backward convolution.
- Status: REJECTED.
- Measured: forward NaNbO3 4.55 vs ≈ 4.3 ms; forward + backward NaCl 2.01 → 1.39 ms but NaNbO3 12.12 → 16.63, NbCl5 11.33 → 12.77 ms.
- Why: slower on large cells.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`.

<a id="conv-oeq-weight-prologue"></a>
#### conv-oeq-weight-prologue: OEq weight prologue (option 4.2)
- Switch: probe
- Engine: RE
- Date/round: 14 October 2026
- What: generate per-edge weights inside the OEq kernel.
- Status: NOT BUILT (slower in benchmark).
- Measured: per-edge weight generation 2.90–4.16 ms vs 1.51 ms GEMM.
- Why: slower than the GEMM.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` (Option 4.2 check).

<a id="conv-streaming-conv"></a>
#### conv-streaming-conv: receiver-streaming OEq convolution (OEq PR 223, + GENERIC_STREAMING)
- Switch: RE `--streaming-conv` / `STREAMING_CONVOLUTION` (`langevin/oeq_streaming/`, NVRTC via `cuda.bindings`), generic `GENERIC_STREAMING`; FM `RewriteConfig.streaming_conv` (`on_for_hybrid_with_stage1_fold_cutoff_and_sort_receivers_if_model_supports`)
- Engine: both
- Date/round: RE 14 October 2026; generic 15 October 2026; FM 1a-T07/T08
- What: receiver-streaming convolution kernels instead of OEq's stock convolution.
- Status: DEFAULT (`auto` with hybrid + folded + sorted). Not bitwise reproducible (atomic sums).
- Measured: standalone NaNbO3 8.22 → 6.35 ms; MD NaCl 4.057 → 3.932 (3.1 %), NaNbO3 25.38 → 23.44 (7.7 %), NbCl5 31.10 → 29.33 ms (5.7 %). Generic models with reassociation: −16 to −28 %.
- Why: the streaming schedule removes OEq's stock gather/scatter overhead.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` (OpenEquivariance PR 223); `ResearchModule:LT/studies/other_models/README.md`; SPEC §6.2.

<a id="conv-ir-mul"></a>
#### conv-ir-mul: `ir_mul` layout for streaming kernels
- Switch: none (inside streaming conv)
- Engine: both
- Date/round: RE 14 October 2026
- What: `ir_mul` instead of `mul_ir` layout, no transposes.
- Status: DEFAULT.
- Measured: NaCl −0.08 ms (2.0 %), NaNbO3 −0.32 ms (1.4 %), NbCl5 −0.49 ms (1.7 %).
- Why: fewer transposes.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`.

<a id="conv-mul-ir"></a>
#### conv-mul-ir: old `mul_ir` layout with transposes
- Switch: `STREAMING_MUL_IR`
- Engine: RE
- Date/round: 14 October 2026
- What: the earlier layout of the streaming kernels.
- Status: SUPERSEDED (by `ir_mul`); not carried over.
- Measured: not recorded beyond "superseded".
- Why: –
- Sources: [switch-map](design/switch-map.md) §7.

<a id="conv-bwd-variants"></a>
#### conv-bwd-variants: streaming backward variants
- Switch: probe
- Engine: RE
- Date/round: 14 October 2026
- What: `__restrict__`, block sizes 64–512, register caps 32 / 40 / 48.
- Status: REJECTED.
- Measured: register caps 1.7–5.6× slower; no gain from the others. Floor: backward within 0.05–0.45 ms of a bare copy (NaNbO3 2.42 ms copy).
- Why: already near the copy floor.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`; [rewrites](rewrites.md).

<a id="conv-receiver-backward"></a>
#### conv-receiver-backward: receiver-owned streaming backward
- Switch: RE `STREAMING_RECEIVER_BACKWARD`; FM kernel kept behind `receiver_backward` (not a config switch)
- Engine: both
- Date/round: RE 14 October 2026; FM re-screen 26 October 2026 (round 5 §3.10); profiled again 27 October 2026 (round 7 T02)
- What: one block per receiver owns its edges in the backward.
- Status: REJECTED (default unchanged).
- Measured: RE NaNbO3 ≈ 4.4 vs 2.7 ms. FM MD ms/step edge → receiver: NaCl 1×1×1 0.838 → 0.927 (+10.6 %), NaCl 3×3×3 3.004 → 3.221 (+7.2 %), NaNbO3 20.175 → 22.065 (+9.4 %), NbCl5 22.691 → 23.713 (+4.5 %); an earlier 3.4 % "win" was a 54-atom, non-alternating, possibly video-contaminated run. Round 7: NbCl5 927 MB of L2 (−15 %), 1,174 MB L1 loads (−47 %), yet 2.10 vs 1.66 ms; NaNbO3 1,502 MB (−10 %), 3.31 vs 3.14 ms; 69–72 % of memory throughput, 5× the reductions.
- Why: one block per receiver serialises a receiver's edges and loses parallelism.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`; [opt-5](validation/opt-5.md) §3.10; [switch-map](design/switch-map.md) §7; branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md.

<a id="conv-cueq-conv-fusion"></a>
#### conv-cueq-conv-fusion: cuEq convolution fusion (MACE PR 1641 backport)
- Switch: RE `--cueq-conv-fusion`; FM `EngineConfig.cueq_conv_fusion` (`on_for_plain_cueq_unless_reproducible`)
- Engine: both
- Date/round: RE 16 October 2026; FM 1a-T05/T15
- What: MACE's fused cuEq convolution for the plain-cuEq backend (MACE 0.3.16 fused only for the device string `cuda`; the RE re-applies the adapter to its deep copies).
- Status: DEFAULT (scoped: plain cuEq backend, off in reproducible plans; hybrid unaffected). RE `--backend cueq` default (user, 16 October 2026).
- Measured: RE float64 MD step −2.3 to −6.0 %, float32 −29 to −40 % (RTX 5060 Ti and leto); exact to rounding. Not measured in FastMACE. The 16 October upstream survey found that MACE develop since 0.3.16 offered nothing further: PRs 1498, 1641, 1711 and 1718 were already taken ([rewrites](rewrites.md)).
- Why: upstream fused kernel.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-16-upstream-mace-survey.md`; ResearchModule AGENTS.md; [switch-map](design/switch-map.md) §2.

<a id="conv-fused-radial-conv"></a>
#### conv-fused-radial-conv: fused radial last layer + streaming convolution (OPT2-T01)
- Switch: none (prototype `tools/fused_conv_bench.py`, branch `fastmace/opt2-T01` only)
- Engine: FM
- Date/round: 21 October 2026, round 2 (OPT-T09 rank 1)
- What: form the `[E, 1280]` tensor-product weights in registers per edge tile inside the receiver-owned streaming convolution (NVRTC/CUDA C), forward and backward.
- Status: REJECTED at gate 1 (conductor's stop rule); backward (gate 2) and record not built.
- Measured: local, pinned `aed96b4`, forward GEMM + conv vs fused best (CS 16, R 8, T 4): NaCl 229 vs 251 µs (−9.6 %), NbCl5 1,944 vs 1,857 µs (+4.5 %), NaNbO3 3,045 vs 3,105 µs (−2.0 %) (positive = faster in the report's table); gate 1 needed a 40 % saving on NbCl5 and NaNbO3. Weight generation at about 18 % of FP32 FMA peak. Bitwise equal message. Plan ceiling ~12 % NaCl/NbCl5, ~20 % NaNbO3; realistic 6 / 6 / 11 %.
- Why: the receiver-owned layout forbids a register-tiled GEMM micro-kernel; the saved `w` traffic is spent again on in-kernel arithmetic.
- Sources: branch fastmace/opt2-T01: docs/design/opt2-T01-fused-radial-conv.md §6 (commit `dfd74525`); [opt-4](../plans/opt-4.md) §2; [OPT-T10](design/opt-T10-step-breakdown.md) §3, §5.

<a id="conv-dw-in-conv"></a>
#### conv-dw-in-conv: backward-only `dw` contraction inside the convolution
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, OPT-T10 "not candidates"
- What: contract `dw` with the folded weights inside the convolution so `dw` is never written.
- Status: NOT BUILT (dropped on paper). A related chunked-backward idea was built as OPT6-T02.
- Measured: not measured.
- Why: same E × 1280 × 65 multiply-adds in the layout where OPT2-T01 reached 18 % of FMA peak.
- Sources: [OPT-T10](design/opt-T10-step-breakdown.md) §5.

<a id="conv-oeq-launch-shape"></a>
#### conv-oeq-launch-shape: OEq streaming forward launch shape on NbCl5 (OPT4-T06)
- Switch: none
- Engine: FM
- Date/round: 24 October 2026, round 4 candidate 6
- What: Nsight Compute and a launch sweep (block 32–512 threads, `--maxrregcount` 64 / 72) of the interaction-2 streaming forward.
- Status: CUT (no shape-rule lever).
- Measured: local NbCl5 802 µs, 93.85 % of DRAM peak (414 GB/s); variants 800.8–830.7 µs (512 threads +3.6 %, register cap 64 +2.1 %). Ceiling 56 µs per force call = 0.26 % of the step (plan estimated ≤ 2 %; its "47 % of DRAM" was an ncu clock artefact).
- Why: already DRAM-bound.
- Sources: [opt4-T06](validation/opt4-T06.md).

<a id="conv-l2-hints"></a>
#### conv-l2-hints: L2 eviction-priority hints in the streaming forward
- Switch: none
- Engine: FM
- Date/round: 24 October 2026, OPT4-T06 residual lever
- What: `evict_first` on `w` loads, `evict_last` on `x` in the vendored template to recover about 41 MB of re-reads.
- Status: NOT BUILT.
- Measured: bounded by the 56 µs (0.26 %) ceiling.
- Why: below any useful gain; template change.
- Sources: [opt4-T06](validation/opt4-T06.md) "Levers left".

<a id="conv-radial-conv-reopen"></a>
#### conv-radial-conv-reopen: radial-producer / convolution fusion via cuBLASDx (round 5 reopen)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5 §3.12
- What: fuse radial weight production into the convolution using cuBLASDx.
- Status: NOT BUILT (not reopened; partly superseded by round 8).
- Measured: not measured (prior: OPT2-T01 at 18 % of FMA peak).
- Why: its primitive failed gate 1 ([launch-cublasdx](#launch-cublasdx)); the SIMT form had failed twice.
- Sources: [opt-5](validation/opt-5.md) §3.12.

<a id="conv-opt6-t01"></a>
#### conv-opt6-t01: L2-resident edge chunks falsifier (OPT6-T01)
- Switch: none (`tools/bench/opt6_l2_chunked_conv.py`)
- Engine: FM
- Date/round: 26 October 2026, round 6
- What: chunk receiver-sorted edges so `w`/`dw` stay in L2: chunked backward (`mixed`), chunked forward (`store`, `recompute`, `pipe`), combinations.
- Status: MEASUREMENT (exact; NbCl5 passes −1.06 ms, NaNbO3 narrow miss −0.93 vs 1.0 ms, NaCl 0; led to OPT6-T02).
- Measured: local subgraph ms: NbCl5 base 4.567 → backward-only 3.960 (−0.61) → `combo1536` 3.511; NaNbO3 7.081 → 6.154 (−0.93); NaCl 0.505 → 0.490. Optimum chunk 1,536 edges; 2,048 halves the gain; ≤ 768 launch-bound (217 × 256-edge chunks: NbCl5 backward 7.5 ms). Pipelined 4,096-edge forward chunks: NbCl5 forward 1.90 → 1.39 ms; NaNbO3 loses at every size.
- Why: backward chunking removes the `[E, 1280]` `dw` round trip.
- Sources: [opt6-T01](validation/opt6-T01.md); [opt-6 plan](../plans/opt-6.md).

<a id="conv-chunked-recompute"></a>
#### conv-chunked-recompute: chunked forward with recompute (no stored `w`)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, OPT6-T01 variant
- What: rebuild `w` by GEMM in the backward instead of storing it.
- Status: REJECTED.
- Measured: slower everywhere.
- Why: the extra GEMM costs more than the traffic it saves.
- Sources: [opt6-T01](validation/opt6-T01.md).

<a id="conv-chunked-conv-backward"></a>
#### conv-chunked-conv-backward: `chunked_conv_backward` (OPT6-T02)
- Switch: `RewriteConfig.chunked_conv_backward` (`on_with_streaming_conv_and_fused_folded_backward`)
- Engine: FM
- Date/round: 27 October 2026, round 6
- What: one autograd op over the folded last layer and the streaming convolution; backward in 1,536-edge (quarter-L2) chunks into two `dw` slots, Triton input gradient on a side stream; the `[E, 1280]` `dw` is never allocated; weights ≤ 4 L2s run unchunked.
- Status: DEFAULT (decision 60, recorded in §19 "Optimisation round 6" and §6.2, not §18); exact (non-deterministic atomics); MH-1 class; float32 and float64 (float64 chunk rule [conv-f64-chunks](#conv-f64-chunks)).
- Measured: local b/a at opt-in commit `08b0d004`: NaCl 1.0057 / 1.0111, NbCl5 0.9771 / 0.9801, NaNbO3 0.9635 / 0.9651; with the size rule at `2ea19f0d`: NaCl 1.0077 / 1.0087, NbCl5 0.9725 / 0.9669, NaNbO3 0.9694 / 0.9642. Gate NaCl 0.04 / 1.16 / 0.31, NbCl5 0.19 / 1.01 / 0.65, NaNbO3 0.35 / 1.08 / 1.02. SPEC §6.2 merges the two sets into "NbCl5 −2.0 to −3.3 %, NaNbO3 −3.1 to −3.6 %".
- Why: keeps `dw` in L2 between producer and consumer.
- Sources: [opt6-T02](validation/opt6-T02.md); SPEC §19, §6.2; [switch-map](design/switch-map.md). CONFLICT: [rewrites](rewrites.md) has an "opt-in" row and a "default" row (defect 2).

<a id="conv-chunked-nacl"></a>
#### conv-chunked-nacl: NaCl +0.8 % under `chunked_conv_backward`
- Switch: as above
- Engine: FM
- Date/round: 27 October 2026
- What: residual NaCl slowdown although the size rule keeps NaCl unchunked.
- Status: PENDING (source not identified, not profiled).
- Measured: NaCl 1.0077 / 1.0087.
- Why: unknown.
- Sources: [opt6-T02](validation/opt6-T02.md) "Not done".

<a id="conv-chunked-ada"></a>
#### conv-chunked-ada: sm_89 check of the L2 chunk rule
- Switch: as above
- Engine: FM
- Date/round: 27 October 2026
- What: verify the quarter-L2 rule on the RTX 4000 Ada (48 MB L2 → 2,432 edges).
- Status: PENDING (later Ada A/Bs of OPT6-T05 and OPT8 ran with it on, but no dedicated rule check).
- Measured: not measured.
- Why: workstation pause at the time.
- Sources: [opt6-T02](validation/opt6-T02.md) "Not done".

<a id="conv-chunked-forward"></a>
#### conv-chunked-forward: chunked forward for low-degree cells (OPT6-T03, ideas F3)
- Switch: none
- Engine: FM
- Date/round: round 6 (planned); ideas F3 27 October 2026
- What: device-side receiver range or fixed edge ranges with `+=` at boundary receivers, behind a degree rule.
- Status: SUPERSEDED (never started; `fused_folded_forward` removed the `w` read differently).
- Measured: OPT6-T01: NbCl5 forward 1.90 → 1.39 ms with pipelined 4,096-edge chunks; NaNbO3 loses at every chunk size.
- Why: only low-degree cells gain.
- Sources: [opt-6 plan](../plans/opt-6.md) T03; [opt6-T01](validation/opt6-T01.md); [ideas](../plans/ideas-20261027.md) F3.

<a id="conv-f64-chunks"></a>
#### conv-f64-chunks: float64 chunks for `chunked_conv_backward` (f64-T03)
- Switch: `oeq_streaming.l2_chunk_edges` rule (no switch of its own)
- Engine: FM
- Date/round: 27 October 2026, f64-T03 (on main via `79fb1115`)
- What: float64 chunks hold `UNCHUNKED_L2_MULTIPLE` (4) L2s (13,056 edges on 32 MiB) instead of a quarter L2 (768 edges).
- Status: DEFAULT (scoped: float64; library rule, no decision number).
- Measured: microbench NbCl5 input gradient 768-edge 33.6 ms, 12,288-edge 15.3 ms, unchunked 19.5 ms; MD step NbCl5 272.4/273.0 → 244.8/245.1 ms (0.90), NaNbO3 253.6/252.6 → 214.4/213.8 (0.85).
- Why: emulated products need large chunks.
- Sources: [f64-T01](validation/f64-T01.md) "f64-T03".

<a id="conv-sobek"></a>
#### conv-sobek: Sobek per-receiver reassociation (arXiv 2607.18074)
- Switch: none
- Engine: FM
- Date/round: 27 October 2026, round 8
- What: accumulate per receiver before the channel products so no per-edge weight tensor exists.
- Status: REJECTED on arithmetic (on paper).
- Measured: 333k FMA per edge (65 × 5,120) vs 83k for the tile approach (4×).
- Why: the per-edge weight is a 65-wide linear map, not a short basis.
- Sources: [opt-8 plan](../plans/opt-8.md); [opt8-T01](validation/opt8-T01.md).

<a id="conv-opt8-ceilings"></a>
#### conv-opt8-ceilings: register-only `w`/`dw` stubs and L2-kept products (OPT8-T01a)
- Switch: none (`tools/bench/opt8_fused_conv.py`)
- Engine: FM
- Date/round: 27 October 2026, round 8 T01(a)
- What: bound the fused-convolution gain by stubbing `w`/`dw` traffic and keeping products in L2.
- Status: MEASUREMENT (not valid bounds).
- Measured: NaNbO3 `gemm_l2` 2.10 ms vs fused forward 1.61 ms; stubs NaNbO3 forward 0.33 ms, backward 0.86 ms.
- Why: cuBLAS on L2 chunks is slower than the prototype's own product, so the "ceilings" overstate the floor.
- Sources: [opt8-T01](validation/opt8-T01.md).

<a id="conv-full-fusion"></a>
#### conv-full-fusion: table output → `w` tile → streaming convolution, forward and backward (OPT8-T01b)
- Switch: none
- Engine: FM
- Date/round: 27 October 2026, round 8 T01(b)
- What: FP32 SIMT kernels generated from the streaming schedule building `w` on chip in forward and backward (`da = dw W` on chip).
- Status: REJECTED (stop rule ≥ 25 % not met).
- Measured: local subgraph fused/chain NaCl 0.80–0.82, NbCl5 0.818–0.822 (3.34 vs 4.07 ms), NaNbO3 0.825–0.827 (5.18 vs 6.27 ms); forward halves, backward 1.12–1.13× the chain. Message bitwise; `dx` ≤ 8.0e-7, `da` ≤ 3.9e-6 relative.
- Why: the backward must rebuild `w` (three FFMA-bound products, issue-bound at IPC 2.6–2.9); the chain backward already keeps `dw` in L2.
- Sources: [opt8-T01](validation/opt8-T01.md); [opt-8 plan](../plans/opt-8.md).

<a id="conv-edge-count-chunks"></a>
#### conv-edge-count-chunks: edge-count chunking instead of receiver chunks (`rc`)
- Switch: none
- Engine: FM
- Date/round: 27 October 2026, round 8 T01
- What: a general edge-count chunk rule to remove the receiver-chunk tail effect.
- Status: NOT BUILT.
- Measured: rc = 3 vs rc = 2 on NaNbO3: 0.90 vs 0.83.
- Why: would not close the gap to 0.75.
- Sources: [opt8-T01](validation/opt8-T01.md) "Not done".

<a id="conv-fused-folded-forward"></a>
#### conv-fused-folded-forward: `fused_folded_forward` (OPT8-T01d → T02–T04)
- Switch: `RewriteConfig.fused_folded_forward` (`on_for_mh1_float32_unless_reproducible`)
- Engine: FM
- Date/round: 27 October 2026, round 8
- What: an NVRTC kernel generated from the streaming schedule computes `w = a @ [W, b].T` and the streaming message, storing `w` for the unchanged chunked backward; launch rule from shared memory and SM count. Prototype T01d (`fusedw`): forward 0.66 / 0.65 (NbCl5 / NaNbO3); forward + backward NbCl5 0.831–0.839, NaNbO3 0.826–0.827, NaCl 0.952–0.961; `w`, message, `dsh`, `da` bitwise.
- Status: DEFAULT (scoped: MH-1 labels `mh1-r2scan`, `mh1-omat`, `mh1-mp`; float32; not reproducible; with `chunked_conv_backward`; refused in float64). Decision 64.
- Measured: local b/a NaNbO3 0.928 / 0.927, NbCl5 0.970 / 0.964, NaCl 0.991 / 1.000; murphy NaNbO3 0.958 / 0.958, NbCl5 0.983 / 0.982, NaCl 0.980 / 0.983; L4 NaNbO3 0.977, NbCl5 0.991 (one order). Gate NaCl 0.085 / 0.97 / 0.36, NbCl5 0.11 / 1.06 / 0.35, NaNbO3 0.32 / 1.01 / 0.67. NVE drift unchanged. Side-b step times: see the speed reference. SPEC decision 64 quotes averaged b/a (NbCl5 0.967, NaCl 0.996).
- Why: removes the cuBLAS folded GEMM and the forward's `w` read.
- Sources: [opt8-T01](validation/opt8-T01.md); [opt8-T02](validation/opt8-T02.md); SPEC decision 64; [switch-map](design/switch-map.md). CONFLICT: [rewrites](rewrites.md) calls it opt-in (defect 3).

<a id="conv-tc-wtile"></a>
#### conv-tc-wtile: tensor-core (FP16×3) w-tile in the fused backward (OPT8 option, ideas F2)
- Switch: none
- Engine: FM
- Date/round: 27 October 2026
- What: tensor-core products for the fused backward's three w-tile products.
- Status: NOT BUILT (dropped on paper).
- Measured: ideal ceiling ≈ 0.47 ms, 2.4 % of the NbCl5 step, ≈ 3 % NaNbO3.
- Why: below 3 % before issue-slot and accuracy penalties (OPT5-T06/T08 lost on sm_120).
- Sources: [ideas](../plans/ideas-20261027.md) F2; [opt8-T01](validation/opt8-T01.md).

<a id="conv-opt7-share"></a>
#### conv-opt7-share: what the convolution costs after radial tables (round 7 §1)
- Switch: none (`tools/step_breakdown.py`)
- Engine: FM
- Date/round: 27 October 2026, round 7
- What: kernel share of the vendored streaming kernels in the float32 default step (with `chunked_conv_backward`), 1000 K, local.
- Status: MEASUREMENT.
- Measured: step ms, conv forward %, conv backward %, rebuild ms/step: MH-1 + tables NaCl 2.78, 3.3, 9.4, 0.21; NbCl3O 5.82, 4.7, 10.4, 0.25; NbCl5 19.21, 4.7, 10.3, 0.61; NaNbO3 15.31, 7.5, 18.6, 0.68. OMAT-0 medium NaCl 1.71, 5.3, 14.8, 0.15; NbCl3O 3.32, 8.4, 18.8, 0.29; NbCl5 9.99, 9.1, 21.9, 0.42; NaNbO3 11.63, 11.2, 24.6, 0.80. Ceilings (backward 2× faster): MH-1 NbCl5 1.054, NaNbO3 1.103, OMAT-0 1.123 / 1.140; backward free 1.115 / 1.229 / 1.280 / 1.326. In float64 the convolution is about 6 % of the step.
- Why: bounds round 7.
- Sources: branch fastmace/opt7-plan: plans/opt-7.md §1.

<a id="conv-opt7-atomics"></a>
#### conv-opt7-atomics: the sender-gradient atomics (OPT7-T01a)
- Switch: none (`t01a.py`)
- Engine: FM
- Date/round: 27 October 2026, round 7
- What: rewrite the edge-owned backward's four `atomicAdd` into `dx` as `racy` (non-atomic read-modify-write) or `drop` (no `dx`), timing only.
- Status: MEASUREMENT (stop rule ≥ 25 % of the backward not met).
- Measured: MH-1 ms base / racy / drop: NaCl 0.232 / 0.247 (1.07) / 0.236 (1.02); NbCl3O 0.602 / 0.645 (1.07) / 0.601 (1.00); NbCl5 1.809 / 1.905 (1.05) / 1.771 (0.98); NaNbO3 2.839 / 2.980 (1.05) / 2.737 (0.96). OMAT-0: racy 1.02–1.07, drop 0.95–0.99.
- Why: Blackwell's L2 reductions make the atomics nearly free: at most 5 % of the backward, 1 % of a step.
- Sources: branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md.

<a id="conv-source-owned"></a>
#### conv-source-owned: source-owned streaming backward (OPT7-T01b)
- Switch: none
- Engine: FM
- Date/round: 27 October 2026, round 7
- What: walk a CSR by sender, accumulate `dx` on chip, gather `dout` by receiver; drop the l = 0 `dsh` column.
- Status: NOT BUILT (stop rule (a) not met; LunaTP's source-owned backward also lost, see [ext-lunatp](#ext-lunatp)).
- Measured: not built.
- Why: the atomics it would remove cost ≤ 5 %; walking by sender gathers rows out of order.
- Sources: branch fastmace/opt7-plan: plans/opt-7.md; docs/validation/opt7-T00-T02.md.

<a id="conv-opt7-traffic"></a>
#### conv-opt7-traffic: where the bytes go (OPT7-T02)
- Switch: none (`ncu_probe.py`)
- Engine: FM
- Date/round: 27 October 2026, round 7
- What: Nsight Compute L2 and L1 traffic split of forward and backward (DRAM counters unavailable under WSL).
- Status: MEASUREMENT. The `w`/`dw` lever was handed to the post-radial-tables work owned by another session (it became round 8).
- Measured: NbCl5 forward / backward, NaNbO3 forward / backward: time 0.76 / 1.66, 1.17 / 3.14 ms; memory throughput 94 / 84, 94 / 81 %; L2 bytes 422 / 1,094, 642 / 1,673 MB; of which `w` (+ `dw`) 268 MB (64 %) / 537 MB (49 %), 430 MB (67 %) / 861 MB (51 %); L1 global loads 490 / 2,204, 786 / 3,535 MB; L2 reductions 0 / 0.84 M, 0 / 1.35 M. The plan's rule (non-`w` ≥ 30 %) passes for the backward; its known remedy is receiver ownership ([conv-receiver-backward](#conv-receiver-backward)).
- Why: decides between path grouping and the `w`/`dw` lever.
- Sources: branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md.

<a id="conv-path-grouping"></a>
#### conv-path-grouping: path-group traffic reduction (round 7 hypothesis 2)
- Switch: none
- Engine: FM
- Date/round: 27 October 2026, round 7
- What: group tensor-product paths to save reloads of `x`, `sh` and `dout` and accumulator registers.
- Status: NOT BUILT (conclusion is an inference from the receiver-owned kernel).
- Measured: not measured directly; LunaTP's node-centric forward, which groups paths, is within 0–11 % of the streaming forward.
- Why: needs a receiver-owned loop to keep its accumulators, which loses more parallelism than it saves.
- Sources: branch fastmace/opt7-plan: plans/opt-7.md §2; docs/validation/opt7-T00-T02.md.

<a id="conv-deriv-activity"></a>
#### conv-deriv-activity: derivative activity specialisation (round 7 hypothesis 3)
- Switch: none (activity switches exist in `schedule.py:136–186` for double backward)
- Engine: FM
- Date/round: 27 October 2026, round 7
- What: skip derivatives MD does not consume.
- Status: NOT BUILT.
- Measured: not measured; in MD all three first derivatives are consumed; only the l = 0 column of `dsh` (1/16 of `dsh`) is discardable.
- Why: too small.
- Sources: branch fastmace/opt7-plan: plans/opt-7.md §2; docs/validation/opt7-T00-T02.md.

<a id="conv-wide-loads"></a>
#### conv-wide-loads: wider (`float2`) loads in the streaming backward
- Switch: none (source transform `wide.py` in `results/opt7-T00-T02-20261027/`, not in FastMACE)
- Engine: FM
- Date/round: 27 October 2026, round 7 (owner request)
- What: each lane of an edge's 32-lane group handles two adjacent channels per iteration; per-edge weight, output-gradient and sender-feature elements read with one 8-byte `float2` load (136 load sites); weight gradients stored as `float2` (10 sites); sender atomics stay scalar.
- Status: NOT BUILT (owner, 27 October 2026: "lets just document it"; eligible). Branch commit `799ad6de`.
- Measured: kernel ratios, local MH-1 / OMAT-0 / murphy MH-1: NaCl 0.931 / 0.935 / 0.951; NbCl3O 0.962 / 0.964 / 1.001; NbCl5 0.971 / 0.977 / 1.003; NaNbO3 0.961 / 0.964 / 0.998. Every local round 0.91–0.99; murphy above NaCl within ±0.4 %; NbCl5 on Ada +0.2 to +0.4 % in all five rounds. Step level 0.3–1.7 % on sm_120, about 0 on Ada except NaCl (≈ 0.5 %). `dw` bitwise; `dx`, `dsh` within 2.4e-7.
- Why: below the owner's rule (about 1 % on only some cases does not change a default). Build recipe and risks (≈ 140 more values per lane; edits the vendored template's output) in the report.
- Sources: branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md "Wider loads".

### Interaction 1 (reassociated)

<a id="int1-reassociate-first"></a>
#### int1-reassociate-first: reassociated first interaction with fused Triton backward (+ GENERIC_REASSOCIATE, OMOL)
- Switch: RE `--reassociate-first` / `REASSOCIATE_FIRST_INTERACTION` (`langevin/edge_outer_sum.py`), generic `GENERIC_REASSOCIATE`; FM `RewriteConfig.reassociate_first` (`on_with_stage1_fold_cutoff_if_model_supports`), one field for both records
- Engine: both
- Date/round: RE 10 October 2026 (MH-1, item 11.2); generic 15 October 2026; OMOL-0 16 October 2026; FM 1a-T07 (MH-1), 1a-T08 (generic)
- What: aggregate by receiver and sender species, then project with node GEMMs; removes the first OEq convolution and the E×512 weights; the Triton backward reads the gathered gradient once.
- Status: DEFAULT (RE hybrid default 10 October 2026, user). The stage-1 coupling is open question 2 in the switch map.
- Measured: RE plain prototype 1.9 % / 2.6 % / 0.6 %; fused NaCl 4.886 → 4.647 (4.9 %), NaNbO3 33.327 → 30.918 (7.2 %), NbCl5 36.131 → 34.345 ms (4.9 %); with density-first 10.8 % / 11.8 % / 11.0 %. Backward kernel 2.19 → 0.43 ms (NaNbO3). Warm setup +10 s. 12 October ablation +4.3 % / +6.1 %. OMOL-0: NbCl5 175.9 → 160.2 ms (−8.9 %). Generic: −16 to −28 % with `streaming_conv`. FM: agreement only (1a-T08: float64 rounding, float32 ≤ 1e-6); no speed number of its own. A100 OMOL-0 failed this record's self-check inside the gate tool (1.2–1.3e-4 vs 1e-4), see [model-gate-t03d](#model-gate-t03d).
- Why: species-only first inputs make the per-edge weights unnecessary.
- Sources: `ResearchModule:LT/studies/further_optimizations/README.md` §7; `ResearchModule:LT/studies/other_models/README.md`; [1a-T08](validation/1a-T08-generic-rewrites.md); [1a-T07-T08 GPU](validation/1a-T07-T08-gpu.md); SPEC §6.2.

<a id="int1-segmented-first-sum"></a>
#### int1-segmented-first-sum: segmented first-interaction sum (+ generic record)
- Switch: RE `--segmented-first-sum` / `SEGMENTED_FIRST_SUM`; FM `RewriteConfig.segmented_first_sum` (`on_with_reassociate_first_and_sort_receivers`), driving the MH-1 record and, since 1b-T11, `generic_segmented_first_sum` plus sender-species grouping
- Engine: both
- Date/round: RE 14 October 2026 (screened twice); FM 1a-T07 (MH-1); generic record 1b-T11 (`efd617f`, 18 October 2026)
- What: run-segmented Triton sum of the reassociated first interaction on receiver-sorted, sender-grouped edges.
- Status: DEFAULT (RE `auto` with reassociation + sorting after the second screen).
- Measured: RE first screen 0.9 % / 1.0 % / 0.2 %, nothing on top of the other three; second screen NaCl 0.5 %, NaNbO3 25.27 → 24.87 ms (1.6 %), NbCl5 −0.2 %. FM 1b-T11 (murphy, MP-0 medium): without the generic record FastMACE was 1.4–1.6 % slower than the RE (B graph NaCl 1.0231, NbCl5 1.0261); after the fix NaCl 1.0011 / 0.9992, NbCl5 0.9991 / 0.9996.
- Why: avoids atomics on sorted runs.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §3, §4b; [1b-T11](validation/1b-T11-mp0-setup.md); [phase-1a](validation/phase-1a.md). CONFLICT (ResearchModule roadmap item 12 and third-round §3 say "not part of the default"; §4b adopts it; defect 36).

<a id="int1-lm-major"></a>
#### int1-lm-major: `lm_major_outer_sum` (OPT3-T05)
- Switch: none (commits `4dcaa30`, `37ef837`, reverted in `fce27e8`)
- Engine: FM
- Date/round: 21 October 2026, round 3
- What: store the reassociated first interaction's outer sums (l, m)-major so each l's GEMM operand is a view.
- Status: REJECTED (reverted).
- Measured: NbCl5 interaction-1 layout kernels 93.2 → 85.9 µs (input transposes 43.6 → 0; joins 15.8 → 37.8; gradient slices 33.8 → 48.1), 0.03 % of the step.
- Why: the transpose moved to the join and gradient slices instead of disappearing.
- Sources: [opt3-T05](validation/opt3-T05-layout-audit.md); [opt3-T05 design](design/opt3-T05-layout-audit.md); [rewrites](rewrites.md).

### Symmetric contraction

<a id="sc-oeq-contraction"></a>
#### sc-oeq-contraction: OEq SymmetricContraction instead of cuEq
- Switch: none
- Engine: both
- Date/round: RE 16 October 2026; FM screened again in OPT3-T02, 21 October 2026
- What: OpenEquivariance 0.7.0's symmetric contraction.
- Status: REJECTED.
- Measured: RE 27–79× slower compiled; exact. FM OPT3-T02: ceiling ≤ 1 %, likely negative.
- Why: much slower.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-16-oeq-symmetric-contraction.md`; [opt3-T02](design/opt3-T02-symmetric-contraction.md); [rewrites](rewrites.md).

<a id="sc-reduced-cg"></a>
#### sc-reduced-cg: reduced Clebsch–Gordan basis (mace-reforge)
- Switch: none
- Engine: RE
- Date/round: 16 October 2026, upstream survey
- What: reduced CG basis for the contraction.
- Status: REJECTED.
- Measured: ≈ 1 % of the cuEq kernel (≈ 1 % of a 0.7 ms kernel), 0.03 % of the NaNbO3 step.
- Why: nothing to gain.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-16-upstream-mace-survey.md`; [rewrites](rewrites.md).

<a id="sc-naive-guard"></a>
#### sc-naive-guard: naive-fallback guard (MACE PR 1614)
- Switch: guard (no flag)
- Engine: both
- Date/round: RE 16 October 2026
- What: refuse runs where cuEq's symmetric contraction or convolution fell back to naive kernels (cuEq's `Linear` layers are naive by design and not checked).
- Status: DEFAULT (guard, not a speed-up).
- Measured: not applicable.
- Why: protects against silent slow paths.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-16-upstream-mace-survey.md`; ResearchModule AGENTS.md.

<a id="sc-fused-input"></a>
#### sc-fused-input: symmetric contraction with fused input (OPT-T09 rank 6)
- Switch: none
- Engine: FM
- Date/round: 21 October 2026
- What: avoid materialising the `[N, 16, 512]` contraction input (needs cuEq changes).
- Status: SUPERSEDED (by OPT3-T02's screen; not built).
- Measured: ceiling ≈ 1 % / 3 % / 2 % (NaCl / NbCl5 / NaNbO3).
- Why: low priority in round 2; screened in round 3.
- Sources: [OPT-T09](design/opt-T09-step-breakdown.md) §5.

<a id="sc-opt3-t02"></a>
#### sc-opt3-t02: cuEquivariance symmetric contraction screen (OPT3-T02)
- Switch: none (tooling fix kept in `tools/step_breakdown.py`)
- Engine: FM
- Date/round: 21 October 2026, round 3
- What: profile and roofline the cuEq `uniform_1d` contraction, then screen other cuEq methods, OEq 0.7.0's contraction, fusing its input materialisation or output side, exact zero paths, reassociation, lower-precision input.
- Status: NOT BUILT (screened; build rule needs a ≥ 3 % ceiling).
- Measured: share of the step NaCl 2.5 %, NbCl5 5.4 %, NaNbO3 3.1 %; kernels at 72–96 % of memory speed-of-light (NbCl5 forward 68–71 MB at up to 422 GB/s); ceilings: other cuEq methods ≤ 1 %, OEq ≤ 1 % (likely negative), fused input ≈ 2.7 % NbCl5 at best (not buildable as an epilogue across four per-l products), output side < 0.5 %, zero paths 0, reassociation n/a, lower precision excluded.
- Why: memory-bound at its traffic floor.
- Sources: [opt3-T02](design/opt3-T02-symmetric-contraction.md); [opt-3 plan](../plans/opt-3.md); [rewrites](rewrites.md).

<a id="sc-tf32-bwd"></a>
#### sc-tf32-bwd: TF32 backward of interaction 0's symmetric contraction (OPT4-T08)
- Switch: none
- Engine: FM
- Date/round: 24 October 2026, round 4
- What: TF32 in the i0 symmetric-contraction backward only.
- Status: NOT BUILT (MH-1 only; not general).
- Measured: MH-1 gate 1.01 / 1.25 / 0.89; omat0-medium NaCl forces 3.92 (fail); i1 version stress 2.38 (NaNbO3).
- Why: fails on another family; production uses cuEquivariance's kernel, which has no TF32 switch.
- Sources: [opt4-T08](validation/opt4-T08-precision-sensitivity.md).

<a id="sc-emulated-gemm"></a>
#### sc-emulated-gemm: symmetric contraction as an emulated GEMM (ideas D5)
- Switch: none
- Engine: FM
- Date/round: 27 October 2026
- What: degree-≤ 3 monomials then one dense INT8-emulated product.
- Status: NOT BUILT (dropped on paper).
- Measured: 969 monomials per (atom, channel) → 7.7 GB per NbCl5 force call, ≈ 34 ms of traffic vs 28 ms for the whole float64 contraction.
- Why: traffic exceeds the time it would save.
- Sources: [ideas](../plans/ideas-20261027.md) D5.

### Precision and tensor cores

<a id="prec-dtype"></a>
#### prec-dtype: float32 forces with TF32 off
- Switch: FM `PrecisionConfig.dtype` (`float32_for_hybrid_float64_for_cueq`), `PrecisionConfig.tf32` (`False`)
- Engine: both
- Date/round: RE production decision 9 October 2026; FM 1a-T01
- What: float32 forces for the hybrid, float64 for plain cuEq; TF32 never default (SPEC §1.2). A float32 path is accepted if its error against float64 is at most 2× that of the float32 calculator.
- Status: DEFAULT.
- Measured: not measured as an A/B in FastMACE; FM 1a-T13 rough single-run times (sanity only): hybrid float32 3.76 / 3.53 ms (checked / pipelined) vs plain cuEq float64 77.8 ms on NaCl.
- Why: production decision.
- Sources: [switch-map](design/switch-map.md) §3; [1a-T13](validation/1a-T13-engine.md); ResearchModule AGENTS.md.

<a id="prec-float64-energy"></a>
#### prec-float64-energy: energy accumulated in float64
- Switch: `PrecisionConfig.float64_energy` (`True`)
- Engine: FM
- Date/round: 1b-T01, 1b-T03
- What: accumulate the energy in float64 (correctness).
- Status: DEFAULT (always on; `False` only for A/B against the RE).
- Measured: "not slower; in both orders it is 0.3 % faster", inside NaCl's block-pair spread (0.990–1.015).
- Why: correctness.
- Sources: [1b-T03](validation/1b-T03-energy.md); SPEC §6.2.

<a id="prec-tf32"></a>
#### prec-tf32: TF32 (global, compiled, selective)
- Switch: `run_variant.py tf32`; `allow_tf32` / `fp32_precision`; selective radial layers
- Engine: both (RE screens; FM keeps TF32 off)
- Date/round: RE 8–9 October 2026 (idea 2), re-checked 12 October; static relaxations 1 October 2026
- What: TF32 matrix products.
- Status: REJECTED (failed accuracy; off by design).
- Measured: coraline uncompiled −27 %, compiled crashed, force error 100–1300× FP32. Local alternating separate-process force: NaCl 8.419 → 6.737 ms (19.90 %), NaNbO3 85.541 → 79.542 ms (7.07 %); TF32 max error / FP32 error 1,020–1,500× (NaCl), 301–327× (NaNbO3). Selective `interactions.0.conv_tp_weights.net.0` ratio 1026.81 / 1016.48 (limit 2). 12 October ablation TF32 −4.0 % NaCl (within drift). Static relaxations (Benchmark_MACE): slower, B₀ shift ≈ 100 %.
- Why: accuracy.
- Sources: `ResearchModule:LT/studies/remaining_optimizations/report.md` (Idea 2); `ResearchModule:LT/studies/compiled_precision/README.md`; `ResearchModule:LT/docs/configuration/workstations.md`; `ResearchModule:Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/README.md`; [rewrites](rewrites.md).

<a id="prec-fp32-emulation-re"></a>
#### prec-fp32-emulation-re: cuBLAS FP32 emulation on tensor cores (RE)
- Switch: none
- Engine: RE
- Date/round: 7 October 2026
- What: cuBLAS's FP32 emulation on tensor cores.
- Status: REJECTED (not available on GeForce Blackwell; data-centre Blackwell only).
- Measured: run `gpu-langevin-cublas-emulation-NaCl333-rtx5060ti-torch214-20261007T214911Z` exists; no gain reported. (FastMACE later uses cuBLAS float64 emulation, a different feature.)
- Why: unavailable.
- Sources: `ResearchModule:LT/studies/optimization_history/report.md` §8; `ResearchModule:LT/docs/configuration/workstations.md`.

<a id="prec-torch-modes"></a>
#### prec-torch-modes: torch FP32 matmul precision modes (OPT-T01 survey)
- Switch: none
- Engine: FM
- Date/round: 20 October 2026, round 1
- What: torch 2.14.1 `fp32_precision` / `set_float32_matmul_precision` as an FP32-accurate tensor-core route.
- Status: REJECTED (survey).
- Measured: not measured (code reading): only `ieee`, `tf32`, `none`; `bf16x3`/`tf32x3` "unknown precision", `bf16` refused for CUDA; Inductor emits only tf32 or ieee.
- Why: no FP32-accurate mode exists.
- Sources: [opt-T01](validation/opt-T01-tensor-cores.md) §1; [opt-1 plan](../plans/opt-1.md).

<a id="prec-bf16x9"></a>
#### prec-bf16x9: cuBLAS BF16×9 FP32 emulation
- Switch: none
- Engine: FM
- Date/round: 20 October 2026, round 1
- What: `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` / `CUBLAS_EMULATE_SINGLE_PRECISION`.
- Status: REJECTED (unavailable on sm_89 and sm_120).
- Measured: not measured; cuBLAS documents compute capability 10.0 and 10.3 only.
- Why: absent on both production GPUs.
- Sources: [opt-T01](validation/opt-T01-tensor-cores.md) §1; [opt-1](validation/opt-1.md); [rewrites](rewrites.md); [megakernel research](design/2026-10-20-megakernel-research.md).

<a id="prec-tf32x3-triton"></a>
#### prec-tf32x3-triton: 3×TF32 GEMMs (RE probe; Triton `tl.dot(input_precision="tf32x3")`)
- Switch: none (RE probe; FM bench `tools/gemm_precision_bench.py`)
- Engine: both
- Date/round: RE 8 October 2026; FM 20 October 2026 (sm_89), 20 October 21:47 UTC (sm_120 rerun), 23 October (A100)
- What: atom-level and radial FP32 GEMMs (all 92 production shapes of an MH-1 force call, forward and input gradients) as 3×TF32 instead of cuBLAS SIMT FP32.
- Status: REJECTED (sm_89, sm_120, A100).
- Measured: RE TF32x3 1.9–3.4× slower on the three large shapes. FM murphy: 4096³ peak cuBLAS IEEE 16.3, 1×TF32 38.6, Triton IEEE 13.5, tf32x3 8.4 TFLOPS; tf32x3 slower on 89 of 92 shapes; per force call NaCl atom linears 3,295 vs 1,856 µs (1.78×), NbCl5 15,573 vs 10,433 µs (1.49×); worst error 5.3e-7 vs IEEE 1.3e-6. Local sm_120: faster on 13 of 92; NbCl5 GEMMs 24.9 vs 12.5 ms, NaCl 4.3 vs 2.6 ms per call; peak 16.7 / 25.0 / 7.1 TFLOPS (FP32 / TF32 / tf32x3). Colab A100: 0 of 92 faster; peak 14.8 / 132.2 / 18.2 (Triton IEEE) / 16.8 TFLOPS; NaCl 7,776 vs 3,911 µs, NbCl5 16,929 vs 10,946 µs. Under cancellation tf32x3 about 4× less accurate than IEEE (1.9e-4 / 1.7e-4 / 2.6e-8 vs 4.8e-5 / 4.6e-5 / 6.5e-9).
- Why: three TF32 products cannot beat SIMT FP32 when TF32 ≤ 3× FP32 (sm_89 ceiling 38.6 / 3 ≈ 12.9 < 16.3); on the A100 Triton's kernel is the limit. CONFLICT: the report's header says the sm_120 rerun is "still to do", while rewrites.md and opt-1.md report it done (13 of 92); the sm_120 numbers are not in the report itself.
- Sources: `ResearchModule:LT/docs/configuration/workstations.md` ("Matrix-multiplication shapes"); [opt-T01](validation/opt-T01-tensor-cores.md) §1–4, §6; [opt-1](validation/opt-1.md); [rewrites](rewrites.md); [opt-4](../plans/opt-4.md) §2.

<a id="prec-cutlass-ada"></a>
#### prec-cutlass-ada: CUTLASS 3×TF32 on sm_89 / sm_120
- Switch: none
- Engine: FM
- Date/round: 20 October 2026, round 1
- What: CUTLASS `OpMultiplyAddFastF32` (example 27) as the C++ 3×TF32 alternative.
- Status: NOT BUILT.
- Measured: not measured.
- Why: same arithmetic and ceiling as Triton's tf32x3 on these cards.
- Sources: [opt-T01](validation/opt-T01-tensor-cores.md) §3.

<a id="prec-cutlass-a100"></a>
#### prec-cutlass-a100: CUTLASS 3×TF32 on the A100 (OPT-T01 §5/§7, OPT4-T03 stage 2)
- Switch: none (standalone C++ bench; would sit behind `tf32x3_gemms` as a custom op with an nvcc toolchain in the package)
- Engine: FM
- Date/round: 23 October 2026 (Colab bench); stage 2 decided 24 October 2026
- What: CUTLASS v4.8.1 `OpMultiplyAddFastF32` on the 92 production shapes with production layouts, tiles 64×64×16 and 64×128×16 by a shape rule.
- Status: NOT BUILT (owner, 24 October 2026: the A100 is Colab-only; stage 2 waits until it is a production target).
- Measured: Colab A100-SXM4-40GB, summed GEMM µs per force call, CUTLASS vs cuBLAS IEEE: NaCl 1,940 vs 2,600 (0.75), NbCl5 5,716 vs 10,313 (0.55); 83 of 92 shapes won; peak 40.8 vs 19.0 TFLOPS; rerun 0.75× / 0.54×. Accuracy: Gaussian worst 6.6e-7 / 3.4e-7 / 9.1e-7; under cancellation 1.8e-3 / 8.6e-4 / 4.2e-7 vs IEEE 4.1e-4 / 2.1e-4 / 1.4e-7: worse than IEEE on 91 of 92 shapes, 7× median, 14× worst.
- Why: fast but only FP32-class on typical data; sm_80 is not a production GPU.
- Sources: [opt-T01](validation/opt-T01-tensor-cores.md) §5, §7; [opt4-T03](validation/opt4-T03.md) §8; [opt-4](../plans/opt-4.md) §2.

<a id="prec-bf16x6"></a>
#### prec-bf16x6: Triton BF16×6 (and BF16×3) stage-1 screen
- Switch: none
- Engine: FM
- Date/round: 23 October 2026 (Colab A100, G4, L4), OPT-T01 §6
- What: Triton `tl.dot` with `input_precision="bf16x6"` / `"bf16x3"` against cuBLAS IEEE and tf32x3.
- Status: REJECTED (G4, L4; the A100 would need a CUTLASS-class kernel; RTX 5060 Ti and RTX 4000 Ada not run).
- Measured: TFLOPS at 4096³ / K = N = 512 M = 13608 / M = 216: bf16x6 A100 19.2 / 17.4 / 1.49, G4 42.8 / 34.1 / 3.67, L4 5.77 / 6.60 / 1.81 vs cuBLAS IEEE 19.0 / 17.5 / 3.25, 74.0 / 57.0 / 7.69, 12.5 / 12.5 / 2.57; never beats cuBLAS IEEE at K = 512 (A100 tie 17.37 vs 17.51). BF16 dense ÷ 6×FP32 at 4096³: A100 2.29, G4 0.83, L4 1.05. bf16x3 not FP32-class (cancellation 1.0e-2 / 7.8e-3 / 1.3e-6). bf16x6 bug for K < 16 (triton#12252) would hit the K = 10 layer.
- Why: needs BF16 dense ≥ 6× FP32 and a better kernel than Triton's.
- Sources: [opt-T01](validation/opt-T01-tensor-cores.md) §1, §6; [opt-4](../plans/opt-4.md) §2.

<a id="prec-tc-overlap"></a>
#### prec-tc-overlap: tensor-core GEMM beside cuBLAS SIMT on two streams (OPT-T01b)
- Switch: none (`tools/gemm_overlap_bench.py`; branch `fastmace/opt-T01b`)
- Engine: FM
- Date/round: 20 October 2026, round 1
- What: split or duplicate a product between cuBLAS SIMT FP32 and a tensor-core GEMM on two streams in one captured graph.
- Status: REJECTED (rule: < 1.10×).
- Measured: murphy, Triton tf32x3 partner: 0.77–0.85× of cuBLAS alone on 9720/13608 × 512 × 512 (best split 75/25), 0.98× on 1944 × 512 × 384, 0.82× on 216 × 512 × 512; duplicate 0.75–0.79×; only 13–29 % of the shorter part overlaps; even 1×TF32 only 1.10–1.36×. Local not measured.
- Why: both GEMMs fill every SM, so the second stream only gets the tail.
- Sources: [rewrites](rewrites.md); [opt-1](validation/opt-1.md); `results/opt-T01b-tensor-core-overlap/murphy/`.

<a id="prec-tf32x3-gemms"></a>
#### prec-tf32x3-gemms: `tf32x3_gemms` record (OPT4-T03) with the `uses_tf32x3_tile` rule (OPT5-T05)
- Switch: `RewriteConfig.tf32x3_gemms` (`never_automatic`; requires `block_linear`, conflicts with `grouped_launch`; refused off float32 and below compute capability 8.0); sub-rule `uses_tf32x3_tile(k, n)`: min(K, N) ≥ 32 on 3×TF32, else SIMT IEEE
- Engine: FM
- Date/round: 24 October 2026, round 4 candidate 3 (stage 1); tile rule 26 October 2026, round 5 candidate 4
- What: custom op `fastmace::tf32x3_mm` plus `segment_gemm`'s `tf32x3` path for all 52 position-dependent atom GEMMs (MH-1), 106 for OMOL-0; round 5 keeps sub-tile products (K ∈ {6, 16}, N ∈ {1, 6}; < 0.4 % of FLOPs) on cuBLAS IEEE.
- Status: NEVER-AUTOMATIC (accuracy pass; never a default; the tile rule is adopted inside the record). A100 A/B of the tile rule not run.
- Measured: force gate local MH-1 NaCl forces 1.09, NbCl5 1.00, OMOL-0 XL NaCl 1.26 (limit 2). Real-operand worst cancellation-scaled error 4.7–5.7e-7 vs cuBLAS IEEE 6.4e-7–1.0e-6; up to 6.9× worse on K = 6 products. Energy offset +2–3e-4 meV/atom. NVE drift A100 NaCl 1.4e-7 vs default 1.1e-6 meV/atom/ps, NbCl5 −5.2e-6 vs −3.0e-6. Speed (A100 drift wall time, not an A/B): NaCl 3.1 → 4.2 ms, NbCl5 13.7 → 15.7 ms (slower). Tile rule: worst per-product error ratio 6.90× → 1.73× (MH-1 NaCl), 6.90× → 1.52× (NbCl5), 2.74× → 1.30× (OMOL-0 XL); gate energy ratio 0.45 → 0.28 (NaCl) / 0.32 (NbCl5); forces 1.18 / 1.00.
- Why: answers the accuracy question; Triton `tf32x3` is slower than cuBLAS SIMT everywhere; a gain needs CUTLASS ([prec-cutlass-a100](#prec-cutlass-a100)).
- Sources: [opt4-T03](validation/opt4-T03.md); [opt-5](validation/opt-5.md) §1–§2; [opt-4](../plans/opt-4.md) §2, §5; [switch-map](design/switch-map.md).

<a id="prec-hybrid-dispatcher"></a>
#### prec-hybrid-dispatcher: Libra hybrid dispatcher and SIMT small-K side stream (OPT5-T05 speed part)
- Switch: none (planned `hybrid_tc_simt_gemms`)
- Engine: FM
- Date/round: 26 October 2026, round 5 candidate 4
- What: route large atom GEMMs to pipelined/CUTLASS 3×TF32 and overlap the small-K skip on a second stream; A100 same-process A/B.
- Status: NOT BUILT (only the accuracy rule landed).
- Measured: not measured.
- Why: depended on OPT5-T02's sm_80 kernel, never built; the A100 is not a production GPU.
- Sources: [opt-5 plan](../plans/opt-5.md) §3, §5.

<a id="prec-deepseek-x3"></a>
#### prec-deepseek-x3: SIMT-split producer and pure-MMA consumer (OPT5-T02)
- Switch: none (probe mode `x3_pipe`; planned name `deepseek_x3`)
- Engine: FM
- Date/round: 26 October 2026, round 5 candidate 1
- What: SIMT warps split FP32 tiles into big/small TF32 in shared memory (`bar.arrive`), MMA warps issue pure `mma.sync`.
- Status: CUT on sm_120 (mechanism confirmed; sm_80 kernel not built).
- Measured: MMA throughput +35.6 % (45.2 → 61.3 FLOP/cycle/SM, 4×4), +21.3 % (2×6), registers 205 → 194; gain 0.85 (4×4). Triton 3×TF32 on `[5832,512]@[512,512]` 497.5 µs vs cuBLAS IEEE 229.6 µs; inline `cvt.rna.tf32` before `tl.dot(tf32)` 543.2 µs (double-rounded).
- Why: on sm_120 TF32 = FP32 rate, so the producer cost exceeds the 3×TF32 gain.
- Sources: [opt5-T01](validation/opt5-T01-scheduler-probe.md); [opt-5](validation/opt-5.md) §1.

<a id="prec-precision-screen"></a>
#### prec-precision-screen: per-stage precision sensitivity screen (OPT4-T08)
- Switch: none (`tools/precision_screen.py`)
- Engine: FM
- Date/round: 24 October 2026, round 4 (owner request)
- What: one force-call stage at a time in bf16/fp16 (output, gradient, saved tensors, autocast, cast) or TF32 (forward/backward), scored with the force gate; MH-1 (NaCl, NbCl5, NaNbO3) and omat0-medium.
- Status: REJECTED as a route (no stage with a real share passes).
- Measured: fp16 misses by 2–1,100×, bf16 by 7× to > 30,000×; passes only ZBL (fp16 storage, ≈ 0 share) and MH-1 i0 symmetric-contraction TF32 backward (1.01 / 1.25 / 0.89; fails on omat0-medium, forces 3.92).
- Why: the error is not concentrated; any 16-bit stage on the force path breaks the gate.
- Sources: [opt4-T08](validation/opt4-T08-precision-sensitivity.md).

<a id="prec-zbl-fp16"></a>
#### prec-zbl-fp16: ZBL pair repulsion in fp16 storage
- Switch: none
- Engine: FM
- Date/round: 24 October 2026, OPT4-T08
- What: fp16/bf16 storage of the ZBL stage's outputs and gradients.
- Status: NOT BUILT (passes the gate, nothing to gain).
- Measured: worst 1.02 / 1.03 / 1.09 (E / F / S); share ≈ 0.
- Why: costs nothing today.
- Sources: [opt4-T08](validation/opt4-T08-precision-sensitivity.md).

<a id="prec-fp16-edge-weights"></a>
#### prec-fp16-edge-weights: FP16 / BF16 storage of per-edge weights
- Switch: probe
- Engine: RE
- Date/round: 14 October 2026, streamed-edges round
- What: store the per-edge weights in 16 bits.
- Status: REJECTED (failed accuracy).
- Measured: FP16 adds 8.3e-5 / 8.5e-4 / 5.5e-4 eV/Å vs allowed ≈ 5e-6 / 2.2e-4 / 1.6e-4; BF16 6.1e-4 / 7.4e-3 / 4.1e-3 (fails by 3–110×).
- Why: accuracy.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`.

<a id="prec-fp16x3-probe"></a>
#### prec-fp16x3-probe: exponent-shifted 3×FP16 in the co-execution probe (OPT5-T06a)
- Switch: none (`tools/megakernel_probe.py --kinds fp16`)
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: `big = fp16(x)`, `small = fp16((x − big)·2048)`, 3 FP16 MMAs at 512 FLOP/cycle/SM, beside SIMT warps.
- Status: REJECTED as a production route on sm_120 (probe passed; no production kernel realised it, superseded by OPT5-T08).
- Measured: x3 fp16 4×4 keep 0.92, gain 1.22×; gemm upper bound 1.28×; 6×2 1.10×.
- Why: the fused GEMM ([prec-hybrid-gemm](#prec-hybrid-gemm)) lost to cuBLAS.
- Sources: [opt-5](validation/opt-5.md) §3.1; [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §1–§2.

<a id="prec-fp16x3-standalone"></a>
#### prec-fp16x3-standalone: standalone Triton 3×FP16 GEMM on sm_120 (OPT5-T06b)
- Switch: none (`tools/bench/opt5_t06_t07_bench.py`, `triton_fp16x3_kernel`)
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: tile-normalised standalone 3×FP16 Triton GEMM.
- Status: REJECTED on sm_120.
- Measured: µs vs cuBLAS IEEE / Triton tf32x3: NaCl [1512,512,512] 77.48 vs 77.46 / 142.03; NbCl5 [5832] 276.50 vs 227.90 / 491.57; [13608] 612.30 vs 490.95; OMOL [1944,1024,1024] 354.25 vs 324.33. 1.78–1.88× faster than tf32x3. MH-1 NaCl force call ΔE 3.77e-5 eV, max ΔF 4.14e-7 eV/Å.
- Why: the 3×FP16 ceiling on sm_120 is 0.67× FP32 SIMT.
- Sources: [opt-5](validation/opt-5.md) §3.2; [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §2.

<a id="prec-fp16x3-rowcol"></a>
#### prec-fp16x3-rowcol: per-row / per-column normalised 3×FP16 (OPT5-T08 path A)
- Switch: none (`triton_fp16x3_rowcol_kernel`)
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: per-row A and per-column B power-of-two normalisation to avoid FP16 subnormals.
- Status: REJECTED (speed; accuracy qualified on MH-1 NaCl operands).
- Measured: worst cancellation error 5.02e-7 vs IEEE 6.44e-7 and tf32x3 5.59e-7; max ratio 1.519× vs 1.727×; 3 of 52 products above IEEE. Sustained [13608] 860.3 µs vs cuBLAS 522.3 µs.
- Why: normalisation overhead.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.1, §3.4.

<a id="prec-fp16x3-other-gpus"></a>
#### prec-fp16x3-other-gpus: standalone row/col 3×FP16 on A100, RTX A4000 and RTX 4000 Ada
- Switch: none (`tools/bench/fp16x3_gemm_screen.py`; planned `RewriteConfig.fp16x3_gemms`, not built)
- Engine: FM
- Date/round: 26 October 2026, round 5 §3.11
- What: best of four Triton tiles on production `gemms.csv` shapes, weighted per force call.
- Status: REJECTED (stop rule ≤ 0.90× cuBLAS fails on all three GPUs).
- Measured: FP16×3 / cuBLAS NaCl / NbCl5: A100 1.69× / 1.12×; RTX A4000 (sm_86, coraline) 1.62× / 1.43×; RTX 4000 Ada (sm_89, leeloo) 1.71× / 1.33×; worst cancellation error 0.63–0.78× IEEE.
- Why: per-K-tile normalisation, split and rescale outweigh the FP16 tensor rate on tall-narrow K = N = 512 shapes. CONFLICT: [opt-5 plan](../plans/opt-5.md) (T08 follow-up) and the fp16x3 note §3.7 still call it "a candidate, pending owner permission".
- Sources: [opt-5](validation/opt-5.md) §3.11; [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §2.

<a id="prec-delta-gemm"></a>
#### prec-delta-gemm: time-step delta GEMM (OPT5-T07a)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: Y_t = Y_{t−1} + ΔX_t W in 1×TF32 or 2×FP16 across 1 fs steps.
- Status: REJECTED (accuracy).
- Measured: per-step feature change ‖ΔX‖/‖X‖ = 20.8 % (median); 1×TF32 delta 1,838× (median) to 6,130× (max) worse than IEEE; 2×FP16 482× worse.
- Why: equivariant features change about 21 % per 1 fs step.
- Sources: [opt-5](validation/opt-5.md) §1, §3.3; [opt-5 plan](../plans/opt-5.md).

<a id="prec-channel-split"></a>
#### prec-channel-split: 50/50 channel-magnitude split (OPT5-T07b)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: top-50 % channels by weight norm in IEEE FP32, the tail in 1×TF32 or 2×FP16.
- Status: REJECTED (accuracy).
- Measured: 1,343× (1×TF32 tail) and 411× (2×FP16 tail) worse than IEEE (median).
- Why: MACE weight row norms are nearly uniform; the tail carries about 40 % of the product norm.
- Sources: [opt-5](validation/opt-5.md) §3.3.

<a id="prec-hybrid-gemm"></a>
#### prec-hybrid-gemm: fused single-kernel SIMT + 3×FP16 NVRTC GEMM (OPT5-T08a)
- Switch: none (`tools/bench/opt5_t08_hybrid_gemm.py`; `hybrid_gemm_64_32`, `hybrid_gemm_6x2_64_128`)
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: 4 SIMT warps (64×64 FP32) + 4 MMA warps (64×32 3×FP16) sharing an A tile; 6 + 2 variant on 64×128; double-buffered, 128 registers, 2 blocks/SM.
- Status: REJECTED (no general cuBLAS win; accuracy fails on real operands).
- Measured: NaCl [1512] 67.67–72.99 µs vs cuBLAS 73.75–74.37 (−1 to −9 %); NbCl5 [5832] 6×2 251.76–254.30 vs 233.95–237.19 (+6.1–7.6 %); [13608] 579.89–580.66 vs 475.61–484.20 (+19.8 %); OMOL [1944,1024,1024] +3.9–5.7 %.
- Why: wins only on the smallest shape; lost SIMT efficiency and barriers dominate.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.2; [opt-5](validation/opt-5.md) §3.2.

<a id="prec-packed-b"></a>
#### prec-packed-b: larger tiles and pre-split (packed) B hybrids (OPT5-T08b)
- Switch: none (`tools/bench/opt5_t08_large_gemm.py`)
- Engine: FM
- Date/round: 26 October 2026, round 5 (owner: faster for more atoms)
- What: B packed once into hi/lo FP16, 128×128 row partitions, 64×128 / 128×128 column partitions with aligned loads, `cp.async` staging.
- Status: REJECTED.
- Measured: including packing, µs vs cuBLAS: [13608] 616.55 / 596.65 vs 527.83; [32768,512,512] 1,545 vs 1,261.27; real MH-1 operands: 36 of 52 (row) and 28 of 38 (column) products > 2× IEEE cancellation error; reciprocal 2^±30 / 2^±20 scaling gives non-finite output.
- Why: no general speed win; a fixed FP16 split loses FP32's exponent range.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.3; [opt-5](validation/opt-5.md) §3.4.

<a id="prec-hybrid-profile"></a>
#### prec-hybrid-profile: Nsight and sustained-power profile of the hybrids (OPT5-T08c)
- Switch: none (`tools/bench/opt5_t08_profile.py`)
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: why the hybrid loses.
- Status: MEASUREMENT (confirms the rejection).
- Measured: [13608] cuBLAS issue 76.6 %, FMA 70.2 %, barrier stall 0.13; hybrids issue 58–62 %, FMA 50–52 %, tensor 23–25 %, barrier 0.97–1.92. All at the 180 W cap; 3 × cuBLAS FP16 one-pass 523.6 µs ≈ FP32 522.3 µs; hybrids +18 % energy per GEMM.
- Why: barrier coupling and lost SIMT efficiency; the power cap removes headroom.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.4; [opt-5](validation/opt-5.md) §3.5.

<a id="prec-decoupled"></a>
#### prec-decoupled: subgroup-barrier decoupling (`col64_aligned_decoupled`, OPT5-T08d)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5 (external Deep Research design 1)
- What: independent SIMT and MMA pipelines with named PTX barriers and private A staging.
- Status: REJECTED.
- Measured: barrier stall 1.357 → 0.880, but +0.7 to +5.3 % slower than the coupled kernel on every shape (e.g. [13608] 615.07 vs 584.15 µs; cuBLAS 517.81).
- Why: duplicate staging and LSU work outweigh the recovered sync time.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.5; [opt-5](validation/opt-5.md) §3.6.

<a id="prec-mbarrier"></a>
#### prec-mbarrier: separate concurrent kernels / `mbarrier` producer-consumer rings (OPT5-T08e)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: deeper decoupling designs.
- Status: NOT BUILT (untested hypotheses).
- Measured: not measured (the two-kernel column split was measured separately, [prec-two-kernel-split](#prec-two-kernel-split)).
- Why: the T08d result lowers their value; GEMM-level hybrid work on sm_120 closed.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.5; [opt-5 plan](../plans/opt-5.md).

<a id="prec-two-kernel-split"></a>
#### prec-two-kernel-split: two-kernel concurrent FP32 / MMA column split (OPT5-T09a)
- Switch: none (`tools/bench/opt5_t09_screens.py`)
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: cuBLAS FP32 on some columns, Triton FP16×3 on the rest, on two streams.
- Status: REJECTED.
- Measured: [13608] full cuBLAS 501.0 µs; best split (384 + 128) sequential 731.2, concurrent 734.1 µs.
- Why: the MMA slice is slower per column than FP32.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6(1).

<a id="prec-bk32-swizzle"></a>
#### prec-bk32-swizzle: BK = 32 staging and block swizzling (OPT5-T09b)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: halve staging and barrier frequency; A-sharing swizzled rasterisation.
- Status: REJECTED (no general gain).
- Measured: BK32 619.7 vs 610.4 µs ([13608]); swizzle 252.8 vs 266.9 ([5832]) but 78.0 vs 68.1 ([1512]); cuBLAS 521.8.
- Why: within noise or shape-dependent; far behind cuBLAS.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6(2).

<a id="prec-fp32-baselines"></a>
#### prec-fp32-baselines: FP32 baseline options (OPT5-T09c)
- Switch: none (`preferred_blas_library("cublaslt")` in a screen)
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: alternative FP32 GEMM paths (cuBLASLt preference, transposed B).
- Status: REJECTED (see also [lin-cublaslt](#lin-cublaslt)).
- Measured: `torch.mm` 501.0, cuBLASLt 501.9 (493.9 first run), transposed-B view 697.7 µs.
- Why: no FP32 alternative beat cuBLAS.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6(3).

<a id="prec-tensor-only-sweep"></a>
#### prec-tensor-only-sweep: tensor-only FP16×3 Triton configuration sweep (OPT5-T09d)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: six tile configurations of the row/col-normalised kernel.
- Status: REJECTED.
- Measured: best (64,128,32,4,3) 795.9 µs vs cuBLAS 501.0 µs ([13608]).
- Why: on-the-fly normalisation costs more than tensor cores return.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6(4).

<a id="prec-presplit-epilogue"></a>
#### prec-presplit-epilogue: producer-side A split and epilogue fusion (OPT5-T09e)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: GEMM on pre-split FP16 hi/lo operands with global scales; fused bias + sigmoid gate + residual epilogue.
- Status: REJECTED (conditional on free producer emission, which is not available).
- Measured: pre-split GEMM 505.9 µs (ties cuBLAS 501.0) but the unfused A split 1,231 µs → 1,737 µs total; fused epilogue 866 µs vs naive chain 1,669 µs.
- Why: recurring operand preparation is the barrier.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.6(5).

<a id="prec-setmaxnreg"></a>
#### prec-setmaxnreg: 4 + 4 warpgroup `setmaxnreg` variant (sm_120a, OPT5-T09f)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: SIMT warpgroup `setmaxnreg.inc 232`, MMA `dec 64`.
- Status: REJECTED.
- Measured: 716.3 vs 1,002.7 µs against its spilling control; 6 + 2 packed 606.6, cuBLAS 530.6 µs ([13608]).
- Why: the 4 + 4 / BM = 128 layout is inferior regardless of registers.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.7(1).

<a id="prec-cutlass-sm120"></a>
#### prec-cutlass-sm120: CUTLASS GeForce SM120 FP16 GEMM as a base (OPT5-T09g)
- Switch: none
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: check whether CUTLASS's SM120 dense GEMM (TMA, `mbarrier`) is a faster FP16 base for a 3-pass kernel.
- Status: REJECTED (not pursued).
- Measured: 194.5 µs vs cuBLAS FP16 174.5 µs ([13608], FP16 → FP32).
- Why: slower base; at most a few-percent ceiling.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.7(2).

<a id="prec-producer-split"></a>
#### prec-producer-split: fused producer A split in Triton (OPT5-T09h)
- Switch: none (`tools/bench/opt5_t09_producer.py`)
- Engine: FM
- Date/round: 26 October 2026, round 5
- What: one kernel splitting FP32 A into FP16 hi/lo with per-row scales.
- Status: REJECTED.
- Measured: split 174.6 µs (≈ plain copy 172.8); split + pre-split GEMM 682.1 vs cuBLAS 503.7 µs; GEMM alone 507.6 µs.
- Why: even with free emission the GEMM only ties cuBLAS.
- Sources: [fp16x3 note](design/2026-10-26-fp16x3-and-hybrid-gemm.md) §3.7(3).

<a id="prec-gemmul8"></a>
#### prec-gemmul8: GEMMul8 Ozaki-II emulation (OPT5-T09j)
- Switch: none (`tools/bench/opt5_t09_gemmul8_bench.cu`)
- Engine: FM
- Date/round: 26 October 2026, round 5 §3.9
- What: Ozaki-II FP32 emulation with weight-encoding reuse.
- Status: REJECTED (not qualified: the benchmark reused the activation encoding, not the weight; eight moduli not qualified as FP32-accurate).
- Measured: numbers kept in `gemmul8_summary.json`, declared invalid as reuse evidence.
- Why: flawed benchmark; no accuracy gate.
- Sources: [opt-5](validation/opt-5.md) §3.9.

<a id="prec-int8-f32"></a>
#### prec-int8-f32: INT8-slice float32 GEMMs (FlashSO(2), arXiv 2610.35164)
- Switch: none (`int8emu.py`)
- Engine: FM
- Date/round: 27 October 2026, f64-T01 §6
- What: emulate float32 GEMMs with p INT8 slices and 128-wide block scales.
- Status: REJECTED on sm_120.
- Measured: 13608×512×512 SGEMM 518 µs (error 4.1e-7) vs 3 slices 599 µs (2.4e-6), 4 slices 965 µs (5.7e-8); one INT8 product only ≈ 2× SGEMM (the paper's GH200 ≈ 30×).
- Why: the INT8:FP32 rate must exceed the product count (≥ 10) for float32 accuracy.
- Sources: [f64-T01](validation/f64-T01.md) §6.

<a id="prec-f64-hybrid"></a>
#### prec-f64-hybrid: hybrid backend for float64 runs
- Switch: planner backend choice; ResearchModule `main.py --backend hybrid --dtype float64`
- Engine: FM
- Date/round: 27 October 2026, decision 63(a)
- What: use the hybrid backend for non-reproducible float64 plans.
- Status: DEFAULT (scoped: float64, not reproducible).
- Measured: plain cuEq float64 2.6× (NaCl) and 3.3× (NbCl5) slower than the full combination; f64-T01 profile: plain cuEq 117 / 891 ms per step, 1.4× slower than hybrid.
- Why: plain cuEq's radial folded layer is 243 vs 74 ms per NbCl5 call.
- Sources: [f64-T01](validation/f64-T01.md) §1; SPEC decision 63.

<a id="prec-fp64-emulation"></a>
#### prec-fp64-emulation: cuBLAS FP64 emulation on INT8 tensor cores (eager strategy)
- Switch: `EngineConfig.fp64_emulation` (`on_for_float64_cuda_with_reduced_rate_fp64_unless_reproducible`; measured `fp64_rate_ratio` < 1/8)
- Engine: FM
- Date/round: 27 October 2026, f64-T01 §2
- What: `CUBLAS_EMULATE_DOUBLE_PRECISION=1`, `CUBLAS_EMULATION_STRATEGY=eager`, set before the first float64 product; the build checks that the emulated kernel runs.
- Status: DEFAULT (scoped: float64, reduced-rate FP64 GPUs (RTX 5060 Ti measured 1/74), not reproducible; off on A100/H100). Decision 63.
- Measured: step NaCl 82.3 → 65.5 ms (0.80), NbCl5 658 → 354 ms (0.55) eager; default strategy 0.86 / 0.67. GEMM 13608×512×512 native 20.7 ms → default 8.2, eager 9.8. Forces vs native ≤ 8.2e-15. A100 forced emulation 5× slower than native (293 vs 59 s, NbCl5 3 ps).
- Why: the INT8:FP64 rate is about 100× on consumer GPUs; accuracy better than native DGEMM.
- Sources: [f64-T01](validation/f64-T01.md) §2; SPEC decision 63; [switch-map](design/switch-map.md).

<a id="prec-emu-performant"></a>
#### prec-emu-performant: emulation default ("performant") strategy
- Switch: `CUBLAS_EMULATION_STRATEGY=performant` (cuBLAS default)
- Engine: FM
- Date/round: 27 October 2026, f64-T01 §2
- What: let cuBLAS emulate only large products.
- Status: REJECTED (eager chosen).
- Measured: NaCl 70.0 ms (0.86), NbCl5 428 ms (0.67) vs eager 65.5 / 354 ms.
- Why: eager also emulates the radial 1280×65 products.
- Sources: [f64-T01](validation/f64-T01.md) §2.

<a id="prec-emu-40"></a>
#### prec-emu-40: 40 mantissa bits
- Switch: `CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=40`
- Engine: FM
- Date/round: 27 October 2026, f64-T01 §2
- What: fewer emulation bits.
- Status: REJECTED (not pursued).
- Measured: 13608×512×512 3.8 ms; error 6e-14 / 6e-12 (normal / spread) vs native 6e-16 / 2.7e-15.
- Why: accuracy is lost fast.
- Sources: [f64-T01](validation/f64-T01.md) §2.

<a id="prec-f64-rewrites"></a>
#### prec-f64-rewrites: rounding-class rewrites in float64
- Switch: `species_tables`, `fold_readout_tail`, `fold_product_linear` (rules renamed without `float32`)
- Engine: FM
- Date/round: 27 October 2026, f64-T01 §3
- What: allow the OPT-T08 folds in float64 plans.
- Status: DEFAULT (scoped: float64, not reproducible). Decision 63(c).
- Measured: step 0.91 on NaCl and NbCl5; forces 1.1e-15 / 5.7e-15 vs native.
- Why: same folds as float32, no accuracy cost.
- Sources: [f64-T01](validation/f64-T01.md) §3; SPEC decision 63.

<a id="prec-f64-triton-rewrites"></a>
#### prec-f64-triton-rewrites: `segment_gemm` / `grouped_launch` in float64
- Switch: `segment_gemm`, `grouped_launch`
- Engine: FM
- Date/round: 27 October 2026, f64-T01 §3
- What: Triton GEMM rewrites in float64.
- Status: NOT BUILT (not tried).
- Measured: not measured.
- Why: Triton GEMMs are float32-only; emulated cuBLAS is faster anyway.
- Sources: [f64-T01](validation/f64-T01.md) §3.

<a id="prec-f64-package"></a>
#### prec-f64-package: the float64 package in the default plan (f64-T02)
- Switch: combination of [prec-f64-hybrid](#prec-f64-hybrid), [prec-fp64-emulation](#prec-fp64-emulation), [prec-f64-rewrites](#prec-f64-rewrites), [rad-f64-tables](#rad-f64-tables)
- Engine: FM
- Date/round: 27 October 2026, f64-T02 (`7562a425`)
- What: all decision-63 parts on by default.
- Status: DEFAULT (scoped: float64). Decision 63.
- Measured: local stock → package NaCl 81.9 → 46.9 ms (0.57), NbCl5 652 → 275 (0.42), NaNbO3 516 → 255 (0.49); forces ≤ 4.0e-14 eV/Å; NVE NaCl 20 ps drift −1.154743e-7 vs −1.154744e-7. leto: NaCl 76.3 → 44.8 (0.59), NbCl5 579 → 233 (0.40). A100 (no emulation): tables + rewrites 0.85.
- Why: float64 accuracy at about half the cost.
- Sources: [f64-T01](validation/f64-T01.md) "The package in the default float64 plan"; SPEC decision 63.

<a id="prec-double-float"></a>
#### prec-double-float: paired-float32 ("double-float") arithmetic (f64-T01 §5, f64-T04)
- Switch: none (`df64_bench.py`; NVRTC df64 microbenchmark)
- Engine: FM
- Date/round: 27 October 2026, f64-T01 §5 (first screen) and f64-T04 (falsifier)
- What: carry float64 as two float32 on FP32 units for FP64-bound non-GEMM kernels (streaming convolution, symmetric contraction); step 2 (df64 streaming forward) only if step 1 passes.
- Status: REJECTED (step 1 fails; step 2 not run). The f64-T04 report is on branch `fastmace/f64-T04` (commit `eb6c56fc`), not on main.
- Measured: first screen 422 vs 200 G FMA/s (2.1×) but accurate only to 1.9e-6 (the compiler broke the error terms); only 10–13 % of the step outside GEMMs and tables. f64-T04: accurate df64 (Joldes–Muller–Popescu) 1.94× native FP64 throughput (bar 2.0×), error 1.3e-13 to 4.6e-13 (bar 1e-13); sloppy addition 2.80× but no error bound.
- Why: misses both bars.
- Sources: [f64-T01](validation/f64-T01.md) §5 and "f64-T03" last paragraph; branch fastmace/f64-T04: docs/validation/f64-T04.md.

<a id="prec-d1-56bits"></a>
#### prec-d1-56bits: fixed 56 mantissa bits in cuBLAS's float64 emulation (ideas D1)
- Switch: `EngineConfig.fp64_emulation_mantissa_bits` (`"auto"` | `"dynamic"` | 8–128; auto rule `56_for_mh1_float64_on_measured_gpus_decision_65_else_dynamic`)
- Engine: FM
- Date/round: 27–28 October 2026, ideas round
- What: `CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=56` instead of cuBLAS's dynamic choice.
- Status: DEFAULT (scoped: float64 MH-1, not reproducible, on "NVIDIA GeForce RTX 5060 Ti" and "NVIDIA RTX 4000 Ada Generation", where decision 63's emulation is on). Decision 65(a). CONFLICT: suspended the same day on the unmerged branch `fastmace/fp64-d1-relax-fix` (`bbd8bd23`, auto = dynamic everywhere; [prec-d1-relax-fix](#prec-d1-relax-fix)); ResearchModule AGENTS.md tells users to pass `"dynamic"`; SPEC, switch-map and the ideas file on main still describe 56 bits.
- Measured: ABBA force-graph ratios (separate processes; process-wide setting) local NbCl5 0.877, NaNbO3 0.881, NaCl 0.862; murphy 0.876 / 0.884 / 0.858. GEMM input-gradient shape 56320×65×1280 0.66 / 0.68; errors identical to dynamic. Forces 2–5e-15 vs the package. NVE NaCl drift −1.1547444e-7 vs −1.1547428e-7.
- Why: dynamic mode pays an exponent pre-pass or picks more bits than needed.
- Sources: [ideas](../plans/ideas-20261027.md) D1, "Per-GPU confirmation", "Decision 65"; SPEC decision 65(a); [switch-map](design/switch-map.md) lines 90–91; branch fastmace/fp64-d1-relax-fix.

<a id="prec-d1-relax-fix"></a>
#### prec-d1-relax-fix: D1 relaxation fix and suspension
- Switch: `fp64_emulation_mantissa_bits` auto rule (changed on the branch); optimiser step code
- Engine: FM
- Date/round: 28 October 2026 (owner)
- What: `bc292dc6`: the optimiser step's small float64 products as element-wise sums, no FP64 GEMM/TRSM (with 56 fixed bits they were rounded to about 8 bits inside captured graphs and float64 variable-cell relaxations diverged); `7b3c1d88`: keep LBFGS's triangular solves as TRSM (the doubling solve lost ASE's step agreement on cell rows); `bbd8bd23`: suspend D1, auto = dynamic everywhere, because a CUDA graph re-captured during a run computed wrong energies and forces (NbOCl3 3×3×3 float64 relaxation on trillian: +0.34 eV/atom from the step after the first re-capture); an explicit bit count still applies.
- Status: ON BRANCH (`fastmace/fp64-d1-relax-fix`, not merged); its GPU checks are open (ResearchModule AGENTS.md).
- Measured: correctness only; no timing recorded.
- Why: correctness of float64 relaxations and quenches.
- Sources: branch fastmace/fp64-d1-relax-fix: commits `bc292dc6`, `7b3c1d88`, `bbd8bd23` (CHANGELOG.md, docs/design/switch-map.md, fastmace/planner/fixed.py); ResearchModule AGENTS.md "FastMACE float64 relaxations".

<a id="prec-d1-48"></a>
#### prec-d1-48: 48 or 52 mantissa bits
- Switch: `fp64_emulation_mantissa_bits` (explicit count)
- Engine: FM
- Date/round: 27 October 2026, ideas round
- What: fewer fixed bits.
- Status: REJECTED ("not recommended").
- Measured: errors 4–20e-15 vs native DGEMM 1–3e-15 on spread operands; 48 bits 1.4× force error (NbCl5), 3× stress error (NaNbO3).
- Why: no longer float64-grade.
- Sources: [ideas](../plans/ideas-20261027.md) "Not recommended for float64".

<a id="prec-save-transcendentals"></a>
#### prec-save-transcendentals: `save_transcendentals` (ideas D3)
- Switch: `RewriteConfig.save_transcendentals` (`on_for_mh1_float64_on_measured_gpus_decision_65`)
- Engine: FM
- Date/round: 27–28 October 2026, ideas round
- What: a custom AOTAutograd min-cut partitioner (`custom_partitioner_fn`) that leaves exp/tanh/sigmoid off the recomputable list.
- Status: DEFAULT (scoped: float64 MH-1, not reproducible, on the two named GPUs). Decision 65(c). Exact (bitwise on CPU).
- Measured: b/a local NbCl5 0.969 / 0.968, NaNbO3 0.980 / 0.983, NaCl 0.975 / 0.977; murphy 0.971 / 0.972, 0.982 / 0.983, 0.979 / 0.979; A100 1.010 / 1.010, 1.022 / 1.023, 0.994 / 0.995. Partitioner pickling bug fixed (`SaveTranscendentals` published by name). Not measured in float32.
- Why: the FP64 pipe is 84–89 % busy, so recomputation is not free in float64.
- Sources: [ideas](../plans/ideas-20261027.md) D3, "Decision 65"; SPEC decision 65(c).

<a id="prec-d123"></a>
#### prec-d123: the three float64 ideas together
- Switch: D1 + D2 + D3
- Engine: FM
- Date/round: 27–28 October 2026
- What: the combination on top of the decision-63 package.
- Status: DEFAULT (scoped: as decision 65; the D1 part is suspended on a branch).
- Measured: ABBA local NbCl5 0.797, NaNbO3 0.782, NaCl 0.806; Ada NbCl5 0.802, NaCl 0.804; ≈ 0.34 / 0.38 / 0.46 of stock float64. NVE drift unchanged.
- Why: independent float64 bottlenecks.
- Sources: [ideas](../plans/ideas-20261027.md) "The three float64 ideas together"; SPEC decision 65.

<a id="prec-ozaki-fp8"></a>
#### prec-ozaki-fp8: FP8 / FP4 Ozaki emulation for float64
- Switch: none
- Engine: FM
- Date/round: 27 October 2026, ideas (mention)
- What: Ozaki schemes on FP8 or FP4.
- Status: REJECTED ("rejected or parked earlier").
- Measured: not recorded (leto has FP8 163 TFLOPS, no FP4, per f64-T01; branch `fastmace/f64-T04` adds `tools/bench/format_rates.py`, dense GEMM rates by format, for the arXiv 2609.37693 follow-up).
- Why: not recorded beyond "rejected or parked".
- Sources: [ideas](../plans/ideas-20261027.md) "Not recommended for float64"; [f64-T01](validation/f64-T01.md); branch fastmace/f64-T04: commit `61697b4e`.

### Launches, graphs, scheduling and megakernels

<a id="launch-cuda-graphs"></a>
#### launch-cuda-graphs: CUDA graphs with padded edge sizes; the graph tier
- Switch: RE `cuda_graph=True` in `GPULangevin` (stages AC/ACP and B); FM no field, the planner's tier (`mode`)
- Engine: both
- Date/round: RE 7 October 2026; FM 1a-T13 (graph engine), 3-T06, decision 38
- What: capture the whole step as CUDA graphs per padded edge count with static buffers; padding edges beyond the cutoff contribute exactly zero. FM replays AC, C, ACP and B as captured graphs.
- Status: DEFAULT (rule 9 amended by decision 38: host callbacks keep the graph tier).
- Measured: RE local NaCl 30.0 → 16.5 ms (−45 %, "single largest step"); coraline uncompiled graphs 17.57 ms FP32 vs ASE 29.5 ms; 12 October ablation: removing graphs +295 % (NaCl). FM 1a-T13: node counts equal to the RE (B 240/238 plain NVT; 315/317 hybrid). 3-T06: a callback every step costs the graph tier 25 % on NaCl (3.87 vs 3.09 ms/step).
- Why: removes launch overhead.
- Sources: `ResearchModule:LT/studies/optimization_history/report.md` §3; `ResearchModule:LT/studies/kernel_fusion/report.md`; `2026-10-12-gear-switch-screen.md`; [1a-T13](validation/1a-T13-engine.md); [3-T06](validation/3-T06.md); SPEC decision 38.

<a id="launch-compiled-tier"></a>
#### launch-compiled-tier: compiled (non-graph) tier
- Switch: tier `mode` (rules 4, 9)
- Engine: FM
- Date/round: 3-T06 / decision 38, 22 October 2026
- What: a compiled tier for inexact padding and as a fallback.
- Status: DEFAULT (scoped: plans with inexact padding or after adaptation; the graph tier is the default tier).
- Measured: whole run MH-1 21.7 vs graph 4.2 ms/step, MP-0 15.7 vs 3.7; with host callbacks compiled is 4.3–4.7× slower on NaCl and 1.3–1.5× on NbCl5. 4-T05 relaxations: graph 2.8–3.9× faster than compiled on NaCl, 1.1× on NbCl5.
- Why: correctness where padding is inexact.
- Sources: [3-T06](validation/3-T06.md); [4-T05](validation/4-T05.md); SPEC decision 38.

<a id="launch-adaptation"></a>
#### launch-adaptation: runtime adaptation (graph → compiled switch)
- Switch: `AdaptationConfig.adapt` (`"auto"`), `trigger` "recapture-time", `recapture_time_fraction` 0.2, `max_switches` 1
- Engine: FM
- Date/round: 5-T07, decisions 52–53, 23 October 2026
- What: one switch to the compiled tier when re-capture eats more than 20 % of wall time.
- Status: DEFAULT.
- Measured: phase 5 item 6: a re-capture "storm" switched at step 1, 27.1 vs 219.2 ms/step unswitched (2,000 steps); production profiles 0 switches, re-capture fraction 0.02–0.65 %.
- Why: protects against re-capture storms.
- Sources: [phase-5](validation/phase-5.md) item 6; SPEC decisions 52, 53; [switch-map](design/switch-map.md) §12.

<a id="launch-calibration"></a>
#### launch-calibration: opt-in backend calibration
- Switch: `CalibrationConfig.calibrate` (`False`), `margin` 0.05
- Engine: FM
- Date/round: 5-T09, decision 55, 23 October 2026
- What: pick a candidate only if faster by more than 5 % in every pair.
- Status: OPT-IN.
- Measured: smoke kept hybrid: 3.95 / 4.01 ms vs cuEq 7.25 / 7.24 ms (NaCl, untimed smoke).
- Why: optional per-machine choice.
- Sources: [5-T09](validation/5-T09.md); [phase-5](validation/phase-5.md) item 9; SPEC decision 55.

<a id="launch-host-stop"></a>
#### launch-host-stop: remove the last host stop (polling, one-step-lagged pipeline)
- Switch: none (code removed)
- Engine: RE
- Date/round: 7–8 October 2026
- What: spin-wait on a CUDA event; or decide rebuild and padded size from step n−1's flags with GPU-side checks.
- Status: REJECTED (no gain; later revived differently as [launch-pipelined](#launch-pipelined)).
- Measured: spin 10.14 vs 9.77 ms (A/B medians); lagged pipeline 9.86 vs 9.83 ms with 92 vs 62 rebuilds.
- Why: no gain.
- Sources: `ResearchModule:LT/docs/configuration/local.md`; `ResearchModule:LT/studies/optimization_history/report.md` §8.

<a id="launch-speculative"></a>
#### launch-speculative: speculative combined AC+B graph
- Switch: RE `--speculative-graph on`; FM `EngineConfig.speculative_graph` (`False`; `SPECULATIVE_SLOTS` = 1 is a constant; turns `pipelined` off)
- Engine: both
- Date/round: RE 13 October 2026; FM 1a-T13
- What: one graph per padded size launched without host sync, validated next step, with rollback and redo of stage B.
- Status: OPT-IN (built, all correctness gates pass, slower).
- Measured: host-gap ceiling (12 October) 4.35 vs 5.25 ms (NaCl, −17 %), 30.63 vs 33.45 ms (NaNbO3, −8.4 %). Real: two in flight −4.0 to −15.3 %, one in flight −5.5 to −7.3 % at 1000 K; −3.7 % NaCl at 300 K (rewrites.md: "−4 to −15 % (slower)"). FM 1a-T13: speculative vs pipelined 1.2e-13 Å (agreement only).
- Why: slower than the pipelined step.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-13-speculative-graph-screen.md`; [rewrites](rewrites.md); [1a-T13](validation/1a-T13-engine.md).

<a id="launch-pipelined"></a>
#### launch-pipelined: pipelined steps
- Switch: RE `--pipelined`; FM `EngineConfig.pipelined` (`on_with_graph_engine_unless_speculative`), constants `pipeline_edge_margin` 0.0, `pipeline_rebuild_fraction` 0.8
- Engine: both
- Date/round: RE 14 October 2026; FM 1a-T13
- What: launch B before the host reads the flags; redo B when needed; early rebuild at 80 % of the skin limit.
- Status: DEFAULT. Exact.
- Measured: RE NaCl 4.024 → 3.896–3.913 ms (3.0 %), NaNbO3 24.89–24.95 → 24.70–24.73 (0.8 %), NbCl5 noise; a first version with a 0.5 % margin was slower on NaNbO3 (25.07 → 25.64). Host gap 0.36–0.51 ms/step. FM 1a-T13: pipelined and checked trajectories agree (2.5e-13 Å float64); rough NaCl 3.53 vs 3.76 ms (single runs, sanity only).
- Why: hides the host round trip.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §4c, §5; [1a-T13](validation/1a-T13-engine.md); SPEC §6.2.

<a id="launch-step-graph"></a>
#### launch-step-graph: `step_graph` (OPT2-T04)
- Switch: `EngineConfig.step_graph` (`False`)
- Engine: FM
- Date/round: 21 October 2026, round 2
- What: one `("ACPB", edges)` CUDA graph per pipelined step (ACP, the flags' copy to the host, an external event node, then B).
- Status: OPT-IN (exact; "stays off").
- Measured: local NaCl 1000 K 0.9973 / 0.9999 (pairs 0.984–1.042), NbCl5 1000 K 1.0015 / 1.0039. murphy A/B never run (outage); not in the L4 round-2 run. CONFLICT: [opt-2 plan](../plans/opt-2.md) says "neutral on both"; only the local card measured it.
- Why: the saved replay launch was already hidden behind the running B graph.
- Sources: [opt2-T04](validation/opt2-T04.md); [opt-2 plan](../plans/opt-2.md); [opt-4](../plans/opt-4.md) §2.

<a id="launch-b-fuse"></a>
#### launch-b-fuse: fusing kernels inside the B replay (NaCl graph gaps)
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, round 2
- What: reduce the gaps between the force replay's 264 kernels by fusing kernels.
- Status: SUPERSEDED (out of the engine task's scope; taken up as OPT3-T04, [launch-combo-kernels](#launch-combo-kernels)).
- Measured: 147 µs of gaps between 264 kernels (OPT-T09).
- Why: a model-compilation change, not an engine one.
- Sources: [opt2-T04](validation/opt2-T04.md); [OPT-T09](design/opt-T09-step-breakdown.md) §5.

<a id="launch-combo-kernels"></a>
#### launch-combo-kernels: Inductor combo kernels (and aggressive fusion)
- Switch: RE `INDUCTOR_COMBO` / `--inductor-combo`; FM `RewriteConfig.combo_kernels` (`never_automatic`)
- Engine: both
- Date/round: RE streamed-edges round, 14 October 2026; FM 21 October 2026, round 3 (OPT3-T04)
- What: compile with `combo_kernels=True, combo_kernels_pointwise_only=True` to cut launches inside B.
- Status: NEVER-AUTOMATIC (FM record kept; rejected as a default).
- Measured: RE combo-only NaCl −0.17 ms, NaNbO3 +0.64 ms; combo + aggressive NaNbO3 +0.66 ms; aggressive alone noise (switch-map: −0.07 to −0.17 ms on NaCl, +0.64 ms on NaNbO3). FM: NaCl kernels per replay 264/266 → 211, gaps 156–157 → 127–129 µs; local (pinned `e3614cc`) NaCl 0.991 / 0.982, NbCl5 1.024 / 1.027; Modal L4 NaCl 0.991 / 0.990, NbCl5 1.019 / 1.020. Reproducible mode: positions, momenta, forces bitwise, potential energy differs by 2.1e-6 eV (rounding). Plan estimate 3–5 % NaCl, exact.
- Why: NbCl5 slower on both GPUs and not exact (energy).
- Sources: `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`; [opt3-T04](validation/opt3-T04.md); [opt-3 plan](../plans/opt-3.md); [rewrites](rewrites.md); [switch-map](design/switch-map.md) §7.

<a id="launch-combo-persub"></a>
#### launch-combo-persub: combo kernels with `combo_kernel_per_subkernel_blocks`
- Switch: none (uncommitted screening copy `c03a511+per_subkernel_blocks`)
- Engine: FM
- Date/round: 21 October 2026, round 3
- What: combo kernels where each sub-kernel keeps its own block size.
- Status: REJECTED.
- Measured: local, one order: NaCl 2.978 → 2.942 ms (0.988, 12/12), NbCl5 21.325 → 21.806 ms (1.023, 0/12).
- Why: did not remove the NbCl5 loss.
- Sources: [opt3-T04](validation/opt3-T04.md).

<a id="launch-combo-sizerule"></a>
#### launch-combo-sizerule: combo kernels on small cells only
- Switch: would be `combo_kernels` with a size rule
- Engine: FM
- Date/round: 21 October 2026 (OPT3-T04, OPT-T10 "per-size Inductor combo kernels"); 24 October 2026 (round 4 "Not proposed")
- What: enable combo kernels only when a rule from atoms and SM count says the cell is small.
- Status: NOT BUILT.
- Measured: remaining gain about 1 % NaCl (round 4 expected 1–2 % NaCl).
- Why: would not make the record exact; size-specific; small.
- Sources: [opt3-T04](validation/opt3-T04.md); [OPT-T10](design/opt-T10-step-breakdown.md) §5; [opt-4](../plans/opt-4.md) §3.

<a id="launch-memsets"></a>
#### launch-memsets: cuBLAS memsets, per-step constants, eager zero-fills
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, round 3 ("for a later round")
- What: (a) remove the 25 cuBLAS memsets before small GEMMs; (b) precompute the species index, `arange` and `ones_like` seed as buffers; (c) fuse eager zero-fills before scatter accumulations into their consumers.
- Status: NOT BUILT.
- Measured: memsets 0.3–0.5 µs each plus gaps; 12 eager `fill_`/`copy_` kernels; not timed.
- Why: (a) needs another GEMM or cuBLAS path; (b) touches MC species changes; (c) overlaps OPT3-T05.
- Sources: [opt3-T04](validation/opt3-T04.md) "Not attempted".

<a id="launch-pdl"></a>
#### launch-pdl: programmatic dependent launch
- Switch: `INDUCTOR_PDL` (`triton.enable_pdl`)
- Engine: RE (not carried over)
- Date/round: fourth round, 15 October 2026
- What: Inductor's programmatic dependent launch.
- Status: REJECTED ("never enable", SPEC §6.2).
- Measured: correct without graphs; wrong forces in CUDA-graph replays, gate ratios 127,219 / 25,546 / 29,161 (switch-map: 1e4–1e5).
- Why: wrong forces.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-15-fourth-round.md` §3; [switch-map](design/switch-map.md) §7; [rewrites](rewrites.md).

<a id="launch-batching"></a>
#### launch-batching: disconnected-graph batching (idea 7)
- Switch: probe
- Engine: RE
- Date/round: 9 October 2026
- What: run several structures as one disconnected graph.
- Status: REJECTED (integrator port declined; production runs crystals sequentially, user rule).
- Measured: 2 × 216-atom NaCl 17.068 → 16.672 ms (2.32 %); 4 × 33.973 → 33.804 ms (0.50 %).
- Why: small gain; excluded by the production rule.
- Sources: `ResearchModule:LT/studies/remaining_optimizations/report.md` (Idea 7); `ResearchModule:LT/studies/batching/README.md`.

<a id="launch-megakernel"></a>
#### launch-megakernel: complete force megakernel (and the external report)
- Switch: none
- Engine: FM
- Date/round: 20 October 2026 (external report; round 1 exclusion)
- What: one persistent kernel for the whole force call. The external report recommended one fused atom block first, a persistence budget of 0–15 %, 3×TF32 10–22 %, and noted BF16×9 is unavailable on sm_89 / sm_120.
- Status: NOT BUILT (excluded from round 1; its items became round 1 tasks and later falsifiers OPT4-T07, OPT5-MK1/MK2).
- Measured: ceiling from the 2-T20a fit 8–16 % NaCl, 1–2 % NbCl5; the report budgets 0–15 % "with regression possible"; gaps about 0.15 ms of 2.77 ms on NaCl.
- Why: months of work for a small ceiling; the step is per-size arithmetic, not launch overhead.
- Sources: [opt-1 plan](../plans/opt-1.md); [opt-1](validation/opt-1.md) §3; [megakernel research](design/2026-10-20-megakernel-research.md).

<a id="launch-het-megakernel"></a>
#### launch-het-megakernel: heterogeneous persistent megakernel and its co-execution gate (OPT4-T07)
- Switch: none (`tools/megakernel_probe.py`)
- Engine: FM
- Date/round: recorded 23 October 2026 (round 1 plan "Long-term direction", design note); gate run 24 October 2026 (round 4 candidate 7)
- What: one persistent kernel scheduling CUDA-core and tensor-core work; gated by a single-SM probe with FFMA warps beside `mma.sync` TF32/BF16 warps (variants reg, smem, gemm, x3 with in-kernel 3×TF32 split); pass = keep ≥ 0.90 and gain ≥ 1.10.
- Status: REJECTED (gate fails on sm_120 and sm_80, sm_89 at threshold; heterogeneous part closed by the owner on 24 October 2026; homogeneous persistence stays under OPT-T05).
- Measured: x3 tf32 4×4: sm_120 keep 0.91, gain 1.04; L4 keep 0.92, gain 1.10; A100 keep 0.58, gain 0.92. Best realistic sm_120 gain 1.04–1.07.
- Why: TF32 MMA rate equals FP32 FFMA on sm_120; on A100 tensor cores alone beat the mix; the L4 pass is a no-traffic ceiling.
- Sources: [opt4-T07](validation/opt4-T07-megakernel-probe.md); [heterogeneous megakernel note](design/2026-10-23-heterogeneous-megakernel.md); [opt-1 plan](../plans/opt-1.md).

<a id="launch-sched-iso"></a>
#### launch-sched-iso: scheduler-isolated and asymmetric co-execution (OPT5-T01)
- Switch: none (`tools/megakernel_probe.py --sched-modes`)
- Engine: FM
- Date/round: 26 October 2026, round 5 candidate 2
- What: pin SIMT and MMA warps to disjoint warp schedulers (`w % 4`); asymmetric ratios 6×2, 7×1, 2×6; named barriers.
- Status: REJECTED (sm_120 closed).
- Measured: sm_120 gemm 4×4 interleaved gain 1.05, sched_iso 0.72; 6×2 1.02, 7×1 1.01 (keep 0.97–0.99). Threshold 1.10. [opt-5](validation/opt-5.md) also cites 0.86 for sched_iso 6×2 (not in the probe report table).
- Why: each sub-partition owns private FFMA and tensor units; isolation starves them.
- Sources: [opt5-T01](validation/opt5-T01-scheduler-probe.md); [opt-5](validation/opt-5.md) §1.

<a id="launch-regional-persistence"></a>
#### launch-regional-persistence: regional persistence of the fused atom block (OPT-T05)
- Switch: none
- Engine: FM
- Date/round: planned 20 October 2026 (round 1, optional, last); close-out recommended 24 October 2026
- What: run OPT-T02's fused atom block as a persistent region (tiled tasks, dependency counters, reusable workspace); must beat the fused block and give ≥ 10 % whole-step gain on both GPUs.
- Status: PENDING (never started; the precondition "C landing" failed; [opt-4](../plans/opt-4.md) §2 recommends closing it; CONFLICT: the 23 October design note still lists it as "nearer-term work"; no owner close-out recorded).
- Measured: not measured; later gap data 100–152 µs per replay (0.5–5 %), below its 10 % stopping rule.
- Why: stage C was rejected; gaps too small.
- Sources: [opt-1 plan](../plans/opt-1.md); [opt-1](validation/opt-1.md); [opt-4](../plans/opt-4.md) §2; [heterogeneous megakernel note](design/2026-10-23-heterogeneous-megakernel.md).

<a id="launch-cublasdx"></a>
#### launch-cublasdx: cuBLASDx FP32 block GEMM as a megakernel primitive (OPT5-MK1)
- Switch: none (`tools/bench/cublasdx_block_bench.cu`)
- Engine: FM
- Date/round: 26 October 2026, round 5 §3.12 (external megakernel proposal)
- What: 2,048 independent 64×64×64 FP32 block GEMMs, cuBLASDx vs cuBLAS strided-batched.
- Status: REJECTED (fails the within-5 % primitive gate).
- Measured: 149.48 µs (7.18 TFLOPS) vs 140.72 µs (7.63 TFLOPS), ratio 1.062.
- Why: the primitive is already slower before any fusion.
- Sources: [opt-5](validation/opt-5.md) §3.12.

<a id="launch-two-streams"></a>
#### launch-two-streams: independent branches on two streams / persistent SM scheduler (OPT5-MK2)
- Switch: none (`tools/bench/stream_branch_probe.py`)
- Engine: FM
- Date/round: 26 October 2026, round 5 §3.12
- What: atom GEMM and edge radial elementwise concurrently vs sequentially (Mirage-style scheduling gate).
- Status: REJECTED.
- Measured: concurrent / sequential NaCl-like 1.151×, NbCl5-like 0.995×, NaNbO3-like 0.985×; graph-gap ceiling ≈ 0.5 % on NbCl5.
- Why: no reliable critical-path gain.
- Sources: [opt-5](validation/opt-5.md) §3.12.

<a id="launch-profiles"></a>
#### launch-profiles: step breakdowns and profiles
- Switch: none (`run_ncu.py`, `tools/step_breakdown.py`)
- Engine: both
- Date/round: 9 October to 27 October 2026
- What: profiles that defined the rounds.
- Status: MEASUREMENT.
- Measured:
  - RE idea 3 (8–9 October): matrix products remain ≈ 23 ms on both large crystals after hybrid conversion.
  - RE idea 6, Nsight Compute (9 October): 342 NaNbO3 graph nodes; large kernels 79–95 % DRAM; GEMMs 49–70 % DRAM, 47 % / 24 % occupancy (MAGMA / CUTLASS).
  - RE 15 October breakdown: GPU span 3.81 / 23.51 / 29.37 ms (NaCl / NaNbO3 / NbCl5); B (forces) 3.49 / 23.05 / 28.96 ms; idle 0.04–0.07 ms; node GEMMs at 11–13 TFLOPS vs 13.5–16.2 standalone ceiling.
  - FM 2-T20a (20 October, round 1 baseline): murphy NaCl 3.68, NbCl5 31.58 ms; local NaCl 3.71, NbCl5 27.43 ms; atom GEMMs 43–50 % of the force call; local NaCl force call 3.32 ms kernel time, atom FP32 GEMMs 1.59 ms at 8.7 TFLOPS (0.8–2.7 waves); NbCl5 atom GEMMs 9.88 ms at 12.6 TFLOPS; fit 0.32 ms fixed + 11.0 µs/atom + 0.14 µs/edge. Neighbour rebuild amortised 0.13 ms/step NaCl (1.60 ms × 81/1000), 0.17 ms NbCl5 (2.13 ms × 80/1000). CONFLICT (minor): the plan says atom GEMMs 1.66 ms (the RE's 15 October number), the opt-1 summary 1.59 ms.
  - FM OPT-T09 (21 October, main `6638185`, 1000 K, local): wall NaCl 3.196, NbCl5 21.461, NaNbO3 18.810 ms; gaps 147 / 109 / 124 µs; vs 2-T20a NaCl −15.6 %, NbCl5 −20.9 %.
  - FM OPT-T10 (21 October, main `072fa0a`): local wall NaCl 3.028 (−5.3 % vs OPT-T09), NbCl5 21.434, NaNbO3 18.788 ms; L4 4.264 / 33.126 / 29.589 ms; gaps 174 / 110 / 121 µs; vs the 15 October RE NaCl −20 %, NbCl5 −27 %, NaNbO3 −20 %.
  - FM OPT4-BASE (24 October, main `ccab528`, multi-head checkpoint, 1000 K, untimed): NaCl 3.083, NbCl5 21.928, NaNbO3 18.409 ms; omat0-medium NaCl 1.710 ms (atom linears at 4.4 TFLOPS, 15 % of peak); polar1-s NaCl 3.914 ms with 1,234 kernels and 568 µs gaps per replay; omol0-xl NaCl 12.735 ms, linears 78 %. A first run used the single-head checkpoint (2.899 / 21.161 / 18.353 ms).
  - FM float32 after radial tables (27 October, main `aaeac537`, ideas probe 2): GPU span per step NbCl5 19.53, NaNbO3 15.88 ms; input gradient `da = dw W` 2.31 / 3.22 ms at ≈ 4.4 TFLOPS (15 % of peak); folded products 17 % / 28 % of eager kernel time. Conclusion: no float32 idea of the list survives.
  - FM float64 (27 October, f64-T01 §1): force call NaCl 74 ms, NbCl5 591 ms; atom GEMMs 61 % / 63 %; radial folded layer 14 % / 13 %; DGEMMs at 0.31–0.35 TFLOPS (70–75 % of FP64 peak, 1/64 of FP32); plain cuEq float64 117 / 891 ms per step.
- Why: –
- Sources: `ResearchModule:LT/studies/remaining_optimizations/report.md` (Ideas 3, 6); `ResearchModule:LT/studies/further_optimizations/2026-10-15-step-breakdown.md`; [2-T20a](validation/2-T20a-step-breakdown.md); [opt-1 plan](../plans/opt-1.md); [OPT-T09](design/opt-T09-step-breakdown.md); [OPT-T10](design/opt-T10-step-breakdown.md); [opt-4](../plans/opt-4.md) §1; [ideas](../plans/ideas-20261027.md); [f64-T01](validation/f64-T01.md) §1.

### Integrator, engine and neighbour list

<a id="eng-gpu-integrator"></a>
#### eng-gpu-integrator: GPU-resident Langevin integrator
- Switch: RE `--engine gpu` (`GPULangevin`); FM `fastmace.integrators.Langevin` and the graph engine
- Engine: both
- Date/round: RE 7 October 2026 (engine), wired into `main.py` 8 October 2026
- What: ASE's Langevin step reimplemented term for term on GPU tensors; positions, velocities and forces stay on the GPU; same NumPy RNG stream as ASE.
- Status: DEFAULT (RE default 8 October 2026; superseded as `main.py` default by `--engine fastmace` on 19 October 2026, which was built from it).
- Measured: NaCl 3×3×3 FP32 local: ASE 41.4 → 30.0 ms/step (together with the skin list).
- Why: removes host round trips.
- Sources: `ResearchModule:LT/studies/optimization_history/report.md` §1–2; `ResearchModule:LT/docs/roadmap.md`.

<a id="eng-skin-list"></a>
#### eng-skin-list: skin neighbour list
- Switch: RE skin 0.3 Å (`SkinNeighbourList`); FM `NeighbourConfig.skin` (`"auto"` = 0.3 Å for MD)
- Engine: both
- Date/round: RE 7 October 2026
- What: store pairs within cutoff + 0.3 Å; rebuild only when an atom moved more than skin/2 (every 15–30 steps), GPU compaction to the true cutoff each step.
- Status: DEFAULT.
- Measured: part of 41.4 → 30.0 ms (NaCl local, not separated).
- Why: rare rebuilds.
- Sources: `ResearchModule:LT/studies/optimization_history/report.md` §2.

<a id="eng-skin-rescreens"></a>
#### eng-skin-rescreens: skin and size-step re-screens
- Switch: skin, bucket width
- Engine: RE
- Date/round: 7–8 October 2026 (first local screen); 12 October 2026 (0.2 Å); 14 October 2026 (0.2 Å, 0.4 Å)
- What: other skins and finer edge-size steps.
- Status: REJECTED (keep 0.3 Å).
- Measured: first screen: all within 1.5 %. 12 October: skin 0.2 Å −1.8 % (NaCl), −0.9 % (NaNbO3), +49 % rebuilds. 14 October: 0.2 Å NaNbO3 ≈ 0.3 ms slower; 0.4 Å NaNbO3 23.40/23.31 vs 23.39/23.03 ms (87 → 65 rebuilds), repeat +0.21 ms NaNbO3, −0.08 ms NbCl5 (net about zero). rewrites.md: "Skin 0.2 Å, 2 % edge buckets: no gain".
- Why: within noise.
- Sources: `ResearchModule:LT/studies/optimization_history/report.md` §8; `ResearchModule:LT/docs/configuration/local.md`; `2026-10-12-gear-switch-screen.md`; `2026-10-14-third-round.md` §5; `ResearchModule:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`; [rewrites](rewrites.md).

<a id="eng-edge-buckets"></a>
#### eng-edge-buckets: 5 % edge buckets and capacity constants (roadmap item 1)
- Switch: RE `EDGE_BUCKET_FRACTION = 0.05`, 256-edge granularity; FM `NeighbourConfig.edge_multiple` (256), `edge_bucket_fraction` (0.05), `capacity_headroom` (1.15), `headroom` ("auto"); `EngineConfig.graph_variant_cap` (64), `dynamo_cache_size_limit` (64), `recapture_budget` ("auto")
- Engine: both
- Date/round: RE 7 October 2026 (commit `d6cf288`); FM budget and cap 3-T01/3-T04
- What: padded edge counts step by 5 % of the initial count instead of 256 edges, bounding graphs and compiled variants; re-capture policy.
- Status: DEFAULT (constants, not auto-tuned).
- Measured: NaCl uses 3 sizes instead of 5; compiled FP32 11.08–11.19 ms (no cost), uncompiled +1 %; reserved memory −15 % (FP32), −10 % (FP64); setup 227 vs 269 s. No FastMACE A/B of the constants.
- Why: fewer graphs and compilations.
- Sources: `ResearchModule:LT/docs/roadmap.md` item 1; `ResearchModule:LT/studies/kernel_fusion/report.md`; [switch-map](design/switch-map.md) §2, §4.

<a id="eng-bucket-width"></a>
#### eng-bucket-width: edge-bucket width 2 % / 3 % / 10 % and stand-alone padding
- Switch: `edge_bucket_fraction`
- Engine: both
- Date/round: RE 8–9 October 2026 (idea 4), 12 and 14 October 2026; FM round 4 "Not proposed" (24 October 2026)
- What: finer or coarser buckets; pad edges independently of other changes.
- Status: REJECTED (keep 5 %).
- Measured: force replay NaNbO3 2 % 83.414 vs 5 % 86.030 ms (3.04 %), NbCl5 0.70 %; 20 ps heating total process time −0.49 to −1.56 %. 12 October NaNbO3 focused A/B 33.37 vs 33.50 ms (+0.4 %); 10 % buckets −3.4 % single pass, not pursued. 14 October NaNbO3 2 %: control processes differ by 0.84 ms, no gain. Round 4: not measured (screened earlier).
- Why: no MD-step gain.
- Sources: `ResearchModule:LT/studies/remaining_optimizations/report.md` (Ideas 4–5); `ResearchModule:LT/studies/edge_buckets/README.md`; `2026-10-12-gear-switch-screen.md`; `2026-10-14-third-round.md` §5; [opt-4](../plans/opt-4.md) §3.

<a id="eng-batched-noise"></a>
#### eng-batched-noise: batched noise and one host synchronisation per step
- Switch: RE 256-step noise batches, combined A/C stage; FM `IntegratorConfig.noise_batch` (256)
- Engine: both
- Date/round: RE 7 October 2026
- What: one NumPy draw and one H2D copy per 256 steps (same numbers as ASE); two GPU stages merged so both flags are read in one copy.
- Status: DEFAULT.
- Measured: local NaCl 10.55–10.85 → 10.0 ms (−5 to −8 %).
- Why: fewer host transfers.
- Sources: `ResearchModule:LT/studies/optimization_history/report.md` §6; `ResearchModule:LT/docs/configuration/local.md`.

<a id="eng-first-derivative"></a>
#### eng-first-derivative: first-derivative-only compiled forces
- Switch: RE `FIRST_DERIVATIVE_ONLY`; FM `EngineConfig.first_derivative_only` (`True`)
- Engine: both
- Date/round: RE 7 October 2026 (roadmap items 3a/3c), default 8 October 2026
- What: compiled force path without the second-derivative graph that MD never uses.
- Status: DEFAULT.
- Measured: NaNbO3 coraline compiled 94.2 vs 134.7 ms (−30 %), peak reserved 11.02 vs 9.62 GiB (+14.6 %, previously +44 %); compiled setup 94 s (was 227 s); round-off ratios 1.06 / 1.07 (NaCl), 1.01 / 1.00 (NaNbO3). Not measured in FastMACE.
- Why: MD needs first derivatives only.
- Sources: `ResearchModule:LT/docs/roadmap.md` items 3, 3a, 3c; `ResearchModule:LT/studies/kernel_fusion/report.md`.

<a id="eng-warmup-pool"></a>
#### eng-warmup-pool: warm-ups inside the graph pool
- Switch: RE in-pool capture warm-ups; FM `EngineConfig.warmup_in_graph_pool` (`True`)
- Engine: both
- Date/round: RE 7–8 October 2026
- What: eager warm-ups of later captures run inside the shared graph pool on the capture stream.
- Status: DEFAULT (memory, not speed).
- Measured: NaNbO3 compiled peak 9.4 → 4.9 GiB; no speed difference in a same-day A/B.
- Why: memory.
- Sources: `ResearchModule:LT/studies/optimization_history/report.md` §7; `ResearchModule:LT/docs/roadmap.md` item 5; [switch-map](design/switch-map.md) §2.

<a id="eng-expandable"></a>
#### eng-expandable: expandable allocator segments
- Switch: RE `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`configure_allocator()`, torch ≥ 2.14); FM `EngineConfig.expandable_segments` (`on_for_torch_2_14_unless_alloc_conf_set`, process-wide)
- Engine: both
- Date/round: RE 8 October 2026
- What: allocator blocks grow in place when new edge sizes appear.
- Status: DEFAULT (memory).
- Measured: NbCl5 20 ps heating 7.4 → 4.1 GiB, same speed; coraline A/B 11.26/11.32 (on) vs 11.30/11.28 ms (off), reserved 0.59 vs 0.96 GiB.
- Why: memory.
- Sources: `ResearchModule:LT/studies/optimization_history/report.md` §7; `ResearchModule:LT/docs/configuration/workstations.md`.

<a id="eng-empty-cache"></a>
#### eng-empty-cache: `empty_cache()` after each capture
- Switch: none (RE); note: FastMACE has `EngineConfig.empty_cache_before_capture` (`True`), a different placement whose rationale is not recorded
- Engine: RE
- Date/round: 7 October 2026
- What: release cached blocks after each capture.
- Status: REJECTED.
- Measured: no peak reduction (the peak is during warm-up).
- Why: no effect.
- Sources: `ResearchModule:LT/docs/roadmap.md` items 3a, 5; `ResearchModule:LT/studies/optimization_history/report.md` §8; [switch-map](design/switch-map.md) §2.

<a id="eng-gpu-neighbour-list"></a>
#### eng-gpu-neighbour-list: GPU neighbour list (Triton pair search)
- Switch: RE `--gpu-neighbour-list` / `GPU_NEIGHBOUR_LIST` (`langevin/pair_search.py`); FM `NeighbourConfig.gpu_neighbour_list` (`on_with_graph_engine`)
- Engine: both
- Date/round: RE third round, 14 October 2026; FM 1a-T09
- What: build the neighbour list on the GPU with in-graph compaction; exact (same pairs).
- Status: DEFAULT.
- Measured: rebuild 1.4–1.7 ms vs 4–24 ms (CPU 24.2 ms NaNbO3); step NaCl 4.710 → 4.549 (3.4 %), NaNbO3 31.00 → 29.58 (4.6 %), NbCl5 34.38 → 33.83 ms (1.6 %). FM 2-T20a: rebuild amortised 0.13 ms/step NaCl, 0.17 ms NbCl5.
- Why: removes the CPU rebuild.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md`; roadmap item 12; [2-T20a](validation/2-T20a-step-breakdown.md) §1.

<a id="eng-species-morton"></a>
#### eng-species-morton: species + Morton atom order
- Switch: probe
- Engine: RE
- Date/round: 14 October 2026
- What: reorder atoms by species and Morton code.
- Status: REJECTED.
- Measured: NaNbO3 25.75/26.11 vs 26.05/25.88 ms.
- Why: no gain.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §5; [rewrites](rewrites.md).

<a id="eng-force-scatter"></a>
#### eng-force-scatter: segmented force scatter
- Switch: none
- Engine: RE
- Date/round: 14 October 2026
- What: segmented force accumulation.
- Status: NOT BUILT.
- Measured: ≈ 21 µs.
- Why: too small.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §5; [rewrites](rewrites.md).

<a id="eng-baoab-1fs"></a>
#### eng-baoab-1fs: BAOAB at 1 fs (as a speed item)
- Switch: none
- Engine: RE
- Date/round: 14 October 2026
- What: a larger timestep via BAOAB.
- Status: NOT BUILT (excluded as a speed item: a timestep change). Production later moved to 1 fs by a separate user decision on 22 October 2026.
- Measured: not measured.
- Why: changes the protocol.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-14-third-round.md` §5; ResearchModule AGENTS.md (timestep).

<a id="eng-lbfgs"></a>
#### eng-lbfgs: LBFGS for the initial relaxation and quenches
- Switch: `--quench-optimizer` (default LBFGS)
- Engine: RE (FastMACE has its own `gpu-LBFGS`, a different implementation)
- Date/round: 13 October 2026 (user decision)
- What: LBFGS (memory 100) instead of BFGS for the anneal's relaxation and quenches.
- Status: DEFAULT (scoped: ResearchModule `main.py`).
- Measured: all ten crystals quench in 1.1 h vs about 31 h with BFGS; NbCl5/TaCl5 20–25× faster; same minima.
- Why: fewer force calls.
- Sources: `ResearchModule:LT/studies/annealing/README.md`.

<a id="eng-overlap-rebuild"></a>
#### eng-overlap-rebuild: `overlap_rebuild` (OPT2-T04)
- Switch: `EngineConfig.overlap_rebuild` (`True`; acts on pipelined fixed-cell CUDA steps)
- Engine: FM
- Date/round: 21 October 2026, round 2 (OPT-T09 rank 4)
- What: a pipelined step's early neighbour rebuild runs on a side stream beside the force graph in flight; hard rebuilds and NPT keep the current path.
- Status: DEFAULT (OPT-DEF-2, §19 thirteenth amendment); exact, also in reproducible plans.
- Measured: probe: early rebuild 4.19 ms per call, 76 per 1000 steps (0.32 ms/step) at 1000 K. Local NaCl 1000 K 0.9635 / 0.9674; NbCl5 1000 K 0.9994 / 0.9927; with `step_graph` at 300 K NaCl 0.9839 / 0.9807. Modal L4 NaCl 300 K 0.996 / 0.990, 1000 K 0.988 / 0.987, NbCl5 0.996 / 1.003. Bitwise in reproducible mode (600 steps, 122 early rebuilds beside). OPT-T10: host idle 0.068 → −0.111 ms. CONFLICT (rounding): report "−3.3 to −3.7 %", plan and SPEC "−3.3 to −3.6 %" / "−0.4 to −3.6 %".
- Why: exact; the gain scales with the rebuild rate.
- Sources: [opt2-T04](validation/opt2-T04.md); [opt-def-2](validation/opt-def-2.md); [opt-2 plan](../plans/opt-2.md); [rewrites](rewrites.md). CONFLICT: switch-map "Counts" calls it opt-in (defect 10).

<a id="eng-noise-in-graph"></a>
#### eng-noise-in-graph: noise selection inside the graph
- Switch: none
- Engine: FM
- Date/round: 21 October 2026, round 2
- What: draw or select the Langevin noise inside the captured graph.
- Status: NOT BUILT.
- Measured: 0.061 ms host time per step (`integrator.prepare_step`), hidden like the replays.
- Why: already hidden behind the GPU.
- Sources: [opt2-T04](validation/opt2-T04.md) "Not built".

<a id="eng-fused-integrator"></a>
#### eng-fused-integrator: `fused_integrator` (OPT3-T03)
- Switch: `EngineConfig.fused_integrator` (`True`)
- Engine: FM
- Date/round: measured 21 October 2026; default 22 October 2026 UTC; round 3
- What: fixed-cell CUDA Langevin's AC/ACP stage (half kick, drift, `x_old`, velocities, displacement maximum, cutoff test, stable compaction, flags) as four Triton kernels around PyTorch's centre-of-mass sums and cumsum instead of about 45 PyTorch kernels; noise stays the host-drawn NumPy batch (the plan's in-kernel Philox was not built).
- Status: DEFAULT (owner decision 21 October 2026 "Yes", §19 fourteenth amendment); also in reproducible plans; NPT, CPU and no-Triton keep the PyTorch stage.
- Measured: local NaCl 0.9758 / 0.9758 (−2.4 %, about 0.07 ms), NbCl5 0.9972 / 0.9929; Modal L4 NaCl 0.9807 / 0.9797, NbCl5 0.9990 / 0.9975. Stage-level byte-identical; reproducible trajectories bitwise. Force gate NaCl 0.05 / 1.15 / 0.31, NbCl5 0.19 / 1.00 / 0.67. Plan estimate ≈ 3 % NaCl. Ada check (leeloo, 21 October): main `5c48e26` + `fused_integrator=True` vs `fastmace-2`: NaCl 3.360 → 2.994 ms (0.891) and 3.405 → 3.020 (0.887); NbCl5 26.661 → 24.229 (0.909) and 26.462 → 24.105 (0.911) (includes all changes since `fastmace-2`).
- Why: exact, small-cell gain on two GPUs, neutral on large cells.
- Sources: [opt3-T03](validation/opt3-T03.md); [opt-3 plan](../plans/opt-3.md); [rewrites](rewrites.md); [Ada leeloo](validation/ada-leeloo-20261022.md).

<a id="eng-fixcm"></a>
#### eng-fixcm: fixcm projection in the fused stage (3-T22b)
- Switch: `IntegratorConfig.fixcm_projection` (`"auto"` = `mass_weighted` under NVT fixcm)
- Engine: FM
- Date/round: 22 October 2026; decision 37
- What: mass-weighted centre-of-mass projection inside the fused integrator stage (a correctness change; speed checked).
- Status: DEFAULT (`mass_weighted_under_nvt_fixcm_else_ase`).
- Measured: plain cuEq float64 MH-1 NaCl, 600 K, local, 10 alternating blocks: 6220.0 µs (ase) vs 6214.7 µs (mass_weighted), 0.09 %, below the 1.1 % floor; fused stage alone 151.0 vs 133.9 µs (eager, launch-bound). Hybrid production plan not timed.
- Why: correctness at no cost.
- Sources: [3-T22b](validation/3-T22b-fused-projection.md); SPEC decision 37.

<a id="eng-fused-nhc"></a>
#### eng-fused-nhc: fused Nose-Hoover chain / MTK stage (4-T03b)
- Switch: no separate field (part of the NHC/MTK integrators' fused stage)
- Engine: FM
- Date/round: 23 October 2026 (on 4-T03 `356d605`, main `352b33c`)
- What: the NHC chain and MTK chain updates in fused kernels instead of PyTorch device-scalar operations.
- Status: DEFAULT (scoped: NHC and MTK integrators; conductor rule "adopt if it passes").
- Measured: stages A + B, harmonic stand-in, NaCl, local: NHC 77 vs 374 µs, MTK hydrostatic 407 vs 983 µs, MTK full 458 vs 1,459 µs (Langevin yardstick 78 µs). MH-1 end to end (600 K, 1 fs, 10 × 300 steps): NHC 6.21 vs 6.50 ms (−4.4 %), MTK full 7.00 vs 8.07 ms (−13 %).
- Why: launch-bound scalar code.
- Sources: [4-T03b](validation/4-T03b.md).

<a id="eng-rule11"></a>
#### eng-rule11: rule 11, adaptive skin and headroom (3-T05)
- Switch: `NeighbourConfig.skin` (`"auto"`), `headroom` (`"auto"`), `d3_skin` (`"auto"`, 3b-T02)
- Engine: FM
- Date/round: 22 October 2026 (3-T05, SPEC §9.3 rule 11)
- What: skin = max(0.3 Å, 2 · 16 · d) from the fastest atom's ballistic displacement; headroom from the same; no tables.
- Status: DEFAULT (scoped: MC, species-change and displacement workloads; MD-only workloads keep 0.3 Å and 1.15).
- Measured: every b/a inside its machine's floor (local 0.988–1.004); NbCl5 faster by 0.1–0.3 % in 6 of 7 runs. D3 skin (3b-T06): 2.04–2.07 Å at 1000 K is the measured optimum; cost flat ±0.1 ms/step from 1.5 to 2.5 Å.
- Why: robustness without a speed cost.
- Sources: [3-T05](validation/3-T05.md); [3b-T06](validation/3b-T06.md); [switch-map](design/switch-map.md) §4.

<a id="eng-d3-cost"></a>
#### eng-d3-cost: GPU D3(BJ) cost (3b-T06)
- Switch: dispersion option; `NeighbourConfig.d3_skin`
- Engine: FM
- Date/round: 23 October 2026; SPEC §11.2 cost target, decision 42
- What: cost of FastMACE's GPU D3 on the production ramp.
- Status: MEASUREMENT (target missed and recorded; decision 42: the phase continues).
- Measured: dense NVT 1000 K: +24 % (4.95 ms on a 20.68 ms step, budget 2.07 ms); NPT anneal at 0 GPa ≈ 15 % amortised over 270 ps; passes ≈ 4.2 ms/step at 1.8 million pairs, builds 0.73 ms/step at 1000 K.
- Why: –
- Sources: [3b-T06](validation/3b-T06.md).

<a id="eng-species-change"></a>
#### eng-species-change: species-change events vs fixed species (3-T20)
- Switch: none (`Workload(species_changes=True)`)
- Engine: FM
- Date/round: 22 October 2026, phase 3 acceptance item 1
- What: speed cost of a superset/species-change plan with host swap events every 100 steps.
- Status: MEASUREMENT (passes, threshold rate ≥ 0.90).
- Measured: rate fixed/events 0.946–0.960 NaCl, 0.986–0.992 NbCl5; 16 of 16 runs valid (local and leeloo, both orders).
- Why: –
- Sources: [3-T20](validation/3-T20.md).

<a id="eng-incremental-nl"></a>
#### eng-incremental-nl: incremental neighbour list (ideas F5)
- Switch: none
- Engine: FM
- Date/round: 27 October 2026
- What: re-search only cells of moved atoms.
- Status: NOT BUILT (never screened).
- Measured: rebuild 0.17 ms × rate, ≤ 1 %.
- Why: not worth a falsifier.
- Sources: [ideas](../plans/ideas-20261027.md) F5.

<a id="eng-on-nl"></a>
#### eng-on-nl: O(N) neighbour-list rebuild for very large cells
- Switch: none
- Engine: FM
- Date/round: 28 October 2026 (owner: future work, not scheduled)
- What: replace the O(N² × images) tile pair search (`fastmace/neighbors/pair_search.py`) by an O(N) rebuild for very large cells.
- Status: NOT BUILT (nothing built or measured; numbers in the plan are estimates).
- Measured: not measured.
- Why: rebuilds are rare with the skin; matters only for very large cells.
- Sources: [future NL plan](../plans/future-neighbour-list.md).

<a id="eng-parity"></a>
#### eng-parity: parity and regression A/Bs
- Switch: none
- Engine: FM
- Date/round: 18 October to 23 October 2026
- What: FastMACE against the research engine and against its own releases.
- Status: MEASUREMENT.
- Measured:
  - Phase 1a (1a-T19, murphy, 18 October), FastMACE/RE: MH-1 NVT NaNbO3 0.9996–0.9998, NbCl5 1.0000–1.0003, NaCl 1.0002 (r3), after 1a-T13b 0.9996 / 0.9950; NPT 1.0023; MP-0 medium 1.0139 (NaCl) / 1.0164 (NbCl5) (fixed in 1b-T11); POLAR-1-S 1.0003; OMOL-0 XL NbCl5 0.9999. Local: MH-1 NVT within 0.33 % on all three crystals (e.g. r1 NaCl 3.628 / 3.628, NaNbO3 22.979 / 22.992, NbCl5 27.035 / 27.060 ms), NPT 1.0016. An A/B tool bug left the RE without `species_skip` (FastMACE looked 12–15 % faster); that run was discarded.
  - Phase 1b vs 1a: NbCl5 on murphy 0.9996–1.0004, within 0.18 % locally; NaCl 1b 0.4–0.6 % faster on murphy, 0.06–0.44 % slower locally; accepted (decision 27).
  - Phase 2 vs `fastmace-1b`: see [lin-t02-combined](#lin-t02-combined).
  - Main vs `fastmace-2` on leeloo (21 October): see [eng-fused-integrator](#eng-fused-integrator).
  - Phase 5 timed regression: freeze / baseline NaCl 0.9987 and 1.0021, NbCl5 1.0031 and 0.9940 ("no gain and no regression"); [benchmarks](benchmarks.md) quotes MH-1 NaCl 2.87 ms, NbCl5 21.0–21.3 ms from it.
- Why: –
- Sources: [phase-1a](validation/phase-1a.md); [1a noise floor](validation/1a-noise-floor.md); [2-T12](validation/2-T12-local-noise-floor.md); [phase-1b](validation/phase-1b.md); [phase-2](validation/phase-2.md); [Ada leeloo](validation/ada-leeloo-20261022.md); [phase-5](validation/phase-5.md).

<a id="eng-bucket-compile"></a>
#### eng-bucket-compile: 1a-T13b compile at the first new edge bucket (fix)
- Switch: none (fix `ae3e758`)
- Engine: FM
- Date/round: 18 October 2026
- What: in a same-process A/B the RE reinstalled its capture-safe `mask_head` and FastMACE did not, so FastMACE compiled at its first new bucket; the fix reinstalls it before each capture.
- Status: DEFAULT (fix).
- Measured: invalid timed blocks before the fix (pair ratios up to 10.36 and 21.80; one block 38.0 / 80.5 ms/step vs about 3.6).
- Why: correctness of timing and of setup cost.
- Sources: [1a-T13b](validation/1a-T13b-bucket-compile.md); [phase-1a](validation/phase-1a.md).

<a id="eng-static-relax"></a>
#### eng-static-relax: Benchmark_MACE_GPU_OPTIMIZATIONS static-relaxation screen
- Switch: various (threads, optimiser, EOS fmax, precision, concurrency)
- Engine: RE (precursor, ASE static relaxations)
- Date/round: 1 October 2026, RTX A4000, MH-1 r²SCAN
- What and status per item:
  - CPU threads: one thread sufficient; 8 slows float32. DEFAULT (scoped: that benchmark; 1 thread).
  - Optimiser: BFGS ≈ LBFGS force-evaluation counts; FIRE about 2× slower and fails the oxygen reference. REJECTED (FIRE).
  - EOS fmax 0.03: EOS 14–19 % faster, B₀ shifts up to ≈ 3 % vs 0.02. REJECTED (not adopted as reported).
  - Mixed precision (float32 pre-relax to fmax 0.01, float64 refinement): ≈ 1.14× speed-up, max B₀ shift 0.021 %. MEASUREMENT (no adoption statement).
  - TF32: slower, B₀ shift ≈ 100 %. REJECTED (see [prec-tf32](#prec-tf32)).
  - Two concurrent jobs: float32 ≈ 1.19×, memory 0.9 → 1.6 GiB; float64 inconsistent. MEASUREMENT.
  - cuEquivariance: PENDING in the README (installed 0.1.0 rejects MACE 0.3.14 `Linear(use_fallback=...)`); stale since all stacks moved to MACE 0.3.16 (defect 43).
- Status: MEASUREMENT (mixed; see the items).
- Measured: as listed. ASE + MACE calculator, float64 cuEq timed step on leeloo (cooling start): 31.44 ms (NaCl), from `optimization_history/launch_history.md`.
- Why: –
- Sources: `ResearchModule:Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/README.md`; `ResearchModule:LT/studies/optimization_history/launch_history.md`.

### Setup, compilation and start-up

<a id="setup-torch-compile"></a>
#### setup-torch-compile: torch.compile kernel fusion in `default` mode
- Switch: RE `--compile-mode default` (`MACEForces(compile_mode="default")`); FM `EngineConfig.compile_mode` (`"default"`)
- Engine: both
- Date/round: RE 7 October 2026 (opt-in), adopted 8 October 2026
- What: Inductor fuses MACE's elementwise chains inside the outer CUDA graphs (no Inductor cudagraphs, `fullgraph=False`); compile once per model, recompile limit 64.
- Status: DEFAULT (after 2 ps, memory and round-off gates; production decision 9 October 2026 excludes autotuning).
- Measured: coraline NaCl FP32 17.57 → 11.13 ms (−37 %, 1.58×), FP64 131 → 118.5 ms (−9.5 %); local 16.5 → 12.6 ms (torch 2.7.1). Idle profile: light work 63.6 % of FP32 kernel time. 12 October ablation: no compile +53.8 % (NaCl).
- Why: fuses elementwise chains.
- Sources: `ResearchModule:LT/studies/kernel_fusion/report.md`; `ResearchModule:LT/studies/optimization_history/report.md` §4.

<a id="setup-stack"></a>
#### setup-stack: newer software stack (torch 2.14.1 + cuEquivariance 0.12.0, recipe C)
- Switch: environment `mace_cueq_latest`; compile cache `MACE_COMPILE_CACHE`
- Engine: both
- Date/round: 7–8 October 2026 (user approval 8 October)
- What: torch 2.7.1 → 2.13 → 2.14.1 with cuEq 0.6.1 → 0.12.0.
- Status: DEFAULT (workstation default environment).
- Measured: local NaCl compiled 12.6 (2.7.1 / cuEq 0.6.1) → 10.9 (2.13) → 10.5 ms (2.14.1 / cuEq 0.12); setup cached 44 → 26 s. Coraline: torch 2.14 FP32 11.26–11.32 ms vs torch 2.5.1 10.95 ms (gap later traced to the fused addmm); FP64 123.7 vs 117.5 ms (open diagnostic).
- Why: faster kernels and compiler.
- Sources: `ResearchModule:LT/studies/optimization_history/report.md` §5; `ResearchModule:LT/docs/configuration/workstations.md`; roadmap item 6.

<a id="setup-max-autotune"></a>
#### setup-max-autotune: `max-autotune` compilation
- Switch: `--compile-mode max-autotune` / `run_variant.py max-autotune`; `max-autotune-no-cudagraphs`
- Engine: RE
- Date/round: 7–8 October 2026 (original); 12 October 2026 (on the hybrid)
- What: Inductor autotuning of GEMMs and more (68-SM heuristic overridden on small GPUs).
- Status: REJECTED (`max-autotune` fails outright with the outer CUDA graphs; `no-cudagraphs` too slow to set up; autotuning excluded by the production decision). CONFLICT: ResearchModule roadmap item 6 still lists a coraline check as open (defect 39).
- Measured: original: local first try no gain, 13 min vs 24 s setup; coraline −7 % (10.2 vs 10.95 ms, NaCl FP32); local with the addmm fix −6.3 %, round-off passed. Hybrid (12 October): `max-autotune-no-cudagraphs` NaCl 5.21 → 5.09 ms (−2.3 %), setup 120–222 s vs 52 s; NaNbO3 34.1 vs 33.4 ms (slower), setup 333 s.
- Why: Inductor cudagraph trees conflict with the outer capture; setup cost.
- Sources: `ResearchModule:LT/studies/optimization_history/report.md` §8; `ResearchModule:LT/docs/configuration/workstations.md`; `ResearchModule:LT/docs/roadmap.md` item 6; `2026-10-12-gear-switch-screen.md`.

<a id="setup-activation-budget"></a>
#### setup-activation-budget: activation memory budget 0.5 / 0.9
- Switch: `ACTIVATION_MEMORY_BUDGET`
- Engine: RE (not carried over)
- Date/round: fourth round, 15 October 2026
- What: activation memory budget below 1 (recompute in the backward).
- Status: REJECTED.
- Measured: +0.4 to +5.9 ms (e.g. NaNbO3 23.2 → 27.4 ms at 0.5). CONFLICT: switch-map says "0.5 and 0.9 were 9–20 % slower".
- Why: recomputation costs time.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-15-fourth-round.md` §2; [switch-map](design/switch-map.md) §7; [rewrites](rewrites.md).

<a id="setup-oeq-seed"></a>
#### setup-oeq-seed: OEq host-extension seed reuse
- Switch: `OEQ_EXTENSION_SEED` (`seed_oeq_extension.py`)
- Engine: RE (FastMACE prepares OEq in-process; a separate `fastmace-oeq-ext` wheel came with 6-T15)
- Date/round: 9 October 2026
- What: reuse a prebuilt OEq host extension instead of rebuilding per process.
- Status: OPT-IN (start-up only).
- Measured: import 19.6 s → 1.5–1.7 s per process, identical library hash.
- Why: start-up time.
- Sources: `ResearchModule:LT/docs/roadmap.md` item 9; `ResearchModule:LT/studies/hybrid_production/README.md`.

<a id="setup-template-cache"></a>
#### setup-template-cache: converted-model template cache (OPT-T06)
- Switch: `EngineConfig.model_template_cache` (`True`; CUDA cuEq and hybrid only)
- Engine: FM
- Date/round: 20–21 October 2026, round 1
- What: the first plan's converted model is kept as a pristine template (copied at once), keyed by checkpoint hash, options, versions and the two conversion patches; later plans, gate reduced-cell models and fallback candidates get a `private_copy` instead of a reload. LRU bound 2 per `ForceField`; mutation guard; `release_templates()`.
- Status: DEFAULT (scoped: CUDA cuEq and hybrid). Conductor decision on merge; §19 twelfth amendment mentions it; no §18 number.
- Measured: murphy MH-1, later plan copy vs reload: hybrid fp32 0.78 vs 3.94 s, hybrid fp64 0.80 vs 2.88 s, cuEq fp32 0.56 vs 3.87 s, cuEq fp64 0.34 vs 2.58 s (−2.2 to −3.3 s per plan); run `20261021c` copy 0.32–0.83 s, reload 2.52–3.79 s. Memory 25.8 MB (fp32) / 51.5 MB (fp64) per template. CPU: e3nn copy loses (0.74 vs 0.39 s), cuEq CPU no gain (41.1 vs 41.6 s). CONFLICT (minor): [opt-1](validation/opt-1.md) says "0.34–0.80 s against 2.6–3.9 s".
- Why: seconds per later plan at negligible memory.
- Sources: [opt-T06](validation/opt-T06-template-cache.md); [OPT-T06 design](design/opt-T06-template-cache.md); [opt-1](validation/opt-1.md).

<a id="setup-mace-version"></a>
#### setup-mace-version: `installed_mace_version()` once per process
- Switch: none
- Engine: FM
- Date/round: 20–21 October 2026, OPT-T06 "small fix alongside"
- What: call `installed_mace_version()` once per process, not per record.
- Status: SUPERSEDED (2-T19b, `42e683d`, already memoises it per `sys.path`).
- Measured: not measured.
- Why: nothing left to do.
- Sources: [OPT-T06 design](design/opt-T06-template-cache.md) (open question 4); [opt-1 plan](../plans/opt-1.md).

<a id="setup-persistent-worker"></a>
#### setup-persistent-worker: persistent subprocess worker (OPT-T07)
- Switch: `RunnerConfig.persistent` (`True`; POSIX only, Windows keeps fresh children)
- Engine: FM
- Date/round: 20–21 October 2026, round 1
- What: a `runner="subprocess"` simulation keeps its setup child as a worker serving every `run()` call, keeping its compiled engine and graphs; restart write unchanged; a failure ends the worker; ends on close, GC, exit or parent death (30 s grace).
- Status: DEFAULT (scoped: POSIX). Conductor decision on the design answers, 20 October 2026; no §18 number.
- Measured: murphy reproducible plain cuEq float64 NaCl, `run(100)` worker calls 44.5, 10.2, 10.2 s vs fresh children 48.6, 61.5, 58.1 s, bitwise equal; hybrid → cuEq case 37.8, 9.6, 9.6 s vs 90.3, 70.1, 68.3 s (2-T07), 2.17e-13 Å vs a fresh process. CPU: calls after the first ≤ 0.18 s vs about 5 s.
- Why: removes the per-call recompile.
- Sources: [opt-T07](validation/opt-T07-persistent-worker.md); [OPT-T07 design](design/opt-T07-persistent-worker.md); [opt-1](validation/opt-1.md).

<a id="setup-dynamic-edges"></a>
#### setup-dynamic-edges: dynamic edge dimension (2-T13, OPT4-T05)
- Switch: `EngineConfig.dynamic_edges` (`True` since decision 59)
- Engine: FM
- Date/round: 2-T13 (merged `f0bde74`, 20 October 2026) for evaluation and compiled-tier plans; OPT4-T05, 24 October 2026, for graph-tier MD
- What: mark the edge dimension dynamic so `fm.calc` / evaluation plans compile once, and so B's padded first capture reuses the first compile.
- Status: DEFAULT (decision 59); exact; off for plans that compile nothing.
- Measured: 2-T13 LBFGS relaxation (41 steps, edge counts 2062/2060/2048): plain cuEq call 2 33.3 s → 0.016 s, hybrid 23.1 s → 0.024 s; median per call unchanged (13.05 → 12.58 ms cuEq, 14.77 → 14.68 ms hybrid); static vs dynamic per call within noise. OPT4-T05 local `fm.md` 300 steps, warm MH-1 NaCl 35.2 → 26.7 s, NbCl5 39.7 → 31.3 s, omat0 NaCl 28.7 → 21.6 s; cold 86.7 → 65.4, 92.9 → 70.5, 71.7 → 55.4 s (every graph-tier start had compiled the model twice: guard on `shifts` 6912 vs 7168). Reproducible trajectories bitwise equal. Decision 59 text: starts 7–9 s shorter warm, 16–22 s cold.
- Why: avoids a second full compile.
- Sources: [2-T13](validation/2-T13-calc-compile.md); [opt4-T05](validation/opt4-T05.md); SPEC decision 59.

<a id="setup-compiler-caches"></a>
#### setup-compiler-caches: compiler caches under `FASTMACE_CACHE` (OPT4-T05b)
- Switch: none (process setup: `FixedPlanner.configure_process` → `kernels.triton.cache.configure_defaults`)
- Engine: FM
- Date/round: 24 October 2026 (owner "Yes")
- What: set `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` under the cache root and enable FX-graph and AOT caches unless the caller set them.
- Status: DEFAULT (conformance fix, SPEC §14; §19 seventeenth amendment; no decision number of its own).
- Measured: GPU smoke: one compile; Inductor 52 MB, Triton 57 MB under `FASTMACE_CACHE`. Cold vs warm compile difference ≈ 20 s with the option, ≈ 33 s without (MH-1 NaCl).
- Why: caches landed in `$TMPDIR` (forbidden on workstations) and did not persist.
- Sources: [opt4-T05](validation/opt4-T05.md); SPEC §19.

<a id="setup-other-items"></a>
#### setup-other-items: other setup items screened (OPT4-T05)
- Switch: none
- Engine: FM
- Date/round: 24 October 2026, round 4 candidate 5
- What: OEq extension build (cold 16 s), torch portable compile-cache artifacts, self-check cost (MH-1 3.7–4.0 s warm, `segment_gemm` check 2.3 s), checkpoint load and conversion (3 + 4 s), fewer graph variants (ACP 0.04–0.4 s).
- Status: NOT BUILT (OEq wheel deferred to the 1.0 plan and built in 6-T15; artifacts superseded by the cache location; self-checks kept by rule; others below the ≥ 5 s warm / ≥ 15 s cold rule).
- Measured: as listed.
- Why: below the build rule or covered by another fix.
- Sources: [opt4-T05](validation/opt4-T05.md) "Verdict", "Not built".

<a id="setup-selfcheck-cache"></a>
#### setup-selfcheck-cache: qualification cache for rewrite self-checks (2-T19)
- Switch: `RewriteConfig.selfcheck_cache` (`"use"`)
- Engine: FM
- Date/round: field from 2-T01; 2-T19 decision 20 October 2026
- What: a cache to skip rewrite self-checks at setup.
- Status: NOT BUILT (2-T19 recommends no cache; the field is inert).
- Measured: warm load + setup 16.74 s (1a) vs 16.75 s (1b), MH-1 NaCl, ext4; self-checks 1.35 s (1b) / 1.43 s (1a) warm first structure, 0.68 s later, 6.2–6.4 s cold; the cache "would save at most about 1.4 s per process warm". Remaining gap +2.9 s on every later plan (per-plan reload, hash, conversion; decision 24). Largest cold stages: first compiled call 30–36 s, captures 22 s, OEq extension build 20 s.
- Why: too little to save.
- Sources: [2-T19](validation/2-T19-setup.md); [switch-map](design/switch-map.md) §1.

<a id="setup-gap"></a>
#### setup-gap: setup gap diagnosis (1b-T11)
- Switch: none
- Engine: FM
- Date/round: 18–19 October 2026, after 1a-T19 reported setup 4–33 s longer
- What: diagnose FastMACE's longer setup.
- Status: MEASUREMENT (no change: a measurement artefact; the FastMACE-only controls used a cold Inductor cache).
- Measured: self-checks 0.1–1.1 s per structure warm, about 3.5 s in a cold first process; with equal cache state setup equal or shorter. 1a-T19 raw figures (murphy): MH-1 NaCl 27.3 vs 45.2 s, OMOL-0 XL NaCl 36.0 vs 68.6 s (RE / FastMACE).
- Why: –
- Sources: [1b-T11](validation/1b-T11-mp0-setup.md); [phase-1a](validation/phase-1a.md).

### Output

<a id="out-async-output"></a>
#### out-async-output: asynchronous output
- Switch: RE `--async-output` (`GPULangevin.snapshot()`); FM `EngineConfig.async_output` (`on_with_graph_engine`)
- Engine: both
- Date/round: RE fourth round, 15 October 2026; FM 1a-T13/T14
- What: asynchronous trajectory and sample output.
- Status: DEFAULT. Bit-identical to synchronous output.
- Measured: RE production wall time NaCl −0.09 ms (−1.9 %), NbCl2O −1.1 %, NaNbO3 −0.06 ms (−0.2 %); 100/100 snapshots bit-identical. SPEC §6.2 records only "bit-identical"; no FastMACE step-time number.
- Why: hides output I/O.
- Sources: `ResearchModule:LT/studies/further_optimizations/2026-10-15-fourth-round.md` §4; [1a-T13](validation/1a-T13-engine.md); SPEC §6.2.

<a id="out-ase-shim"></a>
#### out-ase-shim: ASE shim per-`run()` overhead (1b-T21b) and software block once per process (1b-T28)
- Switch: none (fixes in `fastmace/ase/md.py`, `provenance.fastmace_block`)
- Engine: FM
- Date/round: 1b-T21b (`c38bb56`, 19 October 2026), 1b-T28
- What: the ASE shim stopped rebuilding a second metadata record per `run()`; the static software block (git commit, kernel probes, versions) is built once per process and `git_commit` reads `.git` without starting git.
- Status: DEFAULT (fixes).
- Measured: 1b-T21b (local, MH-1 hybrid NaCl, 0.5 fs, 12 blocks): ext4 shim / Simulation 1.0149 → 0.9998 (+16 ms → 0 per 300-step call); `/mnt/d` checkout 1.90 → 0.986 (48.08 → 4.117 ms/step Simulation). `fastmace_block` ≈ 0.13 s per call from ext4, ≈ 13 s from `/mnt/d`. 1b-T28: before, about 5 s per call (20.8 vs 3.65 ms/step over 300 steps); after, the same as ext4 (3.62–3.67 ms/step).
- Why: per-call metadata cost.
- Sources: [1b-T21b](validation/1b-T21b-ase-shim-overhead.md); [1b-T28](validation/1b-T28-software-block.md).

### Other models (generic family, POLAR, OMOL, OFF)

<a id="model-generic-rewrites"></a>
#### model-generic-rewrites: generic rewrites for other architectures, round 1
- Switch: RE `GENERIC_STREAMING`, `GENERIC_REASSOCIATE` (through `--streaming-conv`, `--reassociate-first`); FM `streaming_conv`, `reassociate_first`
- Engine: both
- Date/round: 15 October 2026 (user decision)
- What: exact streaming convolution and reassociated first interaction for the non-MH-1 architectures (MP-0, MPA-0, OMAT-0 and the others).
- Status: DEFAULT (`auto` for non-MH-1 models).
- Measured: −16 to −28 % on all seven non-MH-1 models (e.g. mp0-medium −20.9 / −28.1 / −25.3 %).
- Why: same mechanisms as for MH-1. Details: [conv-streaming-conv](#conv-streaming-conv), [int1-reassociate-first](#int1-reassociate-first).
- Sources: `ResearchModule:LT/studies/other_models/README.md` (Exact rewrites for the other architectures); SPEC §6.2.

<a id="model-species-skip"></a>
#### model-species-skip: species-restricted skip connection (GENERIC_SPECIES_SKIP)
- Switch: RE `--species-skip`; FM `RewriteConfig.species_skip` (`on_with_graph_engine_if_model_supports`, any backend; switch-map open question 3)
- Engine: both
- Date/round: 15 October 2026 (user decision); FM 1a-T08
- What: skip connection only for the species present (generic MP-0/density family); exact.
- Status: DEFAULT (scoped: generic family).
- Measured: −7 to −39 % (e.g. mpa0-medium NbCl5 17.85 → 10.85 ms, −39.2 %); with the fused radial gradient mpa0/mp0-large −27 to −52 % against the morning default.
- Why: skips work for absent species.
- Sources: `ResearchModule:LT/studies/other_models/README.md` (Second round); SPEC §6.2; [phase-1a](validation/phase-1a.md).

<a id="model-generic-density-first"></a>
#### model-generic-density-first: density-first for the generic family
- Switch: `GENERIC_DENSITY_FIRST`
- Engine: RE (not carried over)
- Date/round: 15 October 2026
- What: `density_first` for the generic family.
- Status: REJECTED ("stays a screen option (off)"; switch-map open question 1).
- Measured: "gave no gain"; no number in FastMACE documents.
- Why: no gain.
- Sources: ResearchModule AGENTS.md; [switch-map](design/switch-map.md) §7.

<a id="model-polar-uvu"></a>
#### model-polar-uvu: PolarMACE uvu reassociation (MACE PR 1711 backport)
- Switch: RE `--polar-uvu` (`REASSOCIATE_UVU`); FM `RewriteConfig.polar_uvu` (`True`)
- Engine: both
- Date/round: RE 16 October 2026 (user); FM 1a-T10
- What: PolarMACE's uvu tensor product in GEMM form.
- Status: DEFAULT (scoped: POLAR models).
- Measured: with the MH-1 rewrites a further 4.2–5.5× for POLAR (e.g. POLAR-1-L NbCl5 574 → 114.6 ms); 12–22× faster than stock ASE. FM 1a-T19: POLAR-1-S at parity with the RE (1.0003 on NaCl and NbCl5).
- Why: GEMM form of a slow tensor product.
- Sources: `ResearchModule:LT/studies/other_models/README.md` (PolarMACE on the GPU); SPEC §6.2; [phase-1a](validation/phase-1a.md).

<a id="model-spin-restricted"></a>
#### model-spin-restricted: spin-restricted PolarMACE
- Switch: `polar_gpu.SPIN_RESTRICTED`
- Engine: RE (not carried over)
- Date/round: research engine (date not recorded)
- What: spin-restricted PolarMACE evaluation.
- Status: REJECTED ("never enable": changes the model).
- Measured: accuracy only: POLAR-1-L NbCl3O max force 0.47 → 30.5 eV/Å.
- Why: worse model.
- Sources: [switch-map](design/switch-map.md) §7.

<a id="model-fx-e3nn"></a>
#### model-fx-e3nn: `fx_e3nn_linear` (OPT4-T02a)
- Switch: `RewriteConfig.fx_e3nn_linear` (`on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints`)
- Engine: FM
- Date/round: 24 October 2026, round 4 candidate 2
- What: give each TorchScript `o3.Linear` restored from a checkpoint e3nn's own fx graph, so dynamo traces PolarMACE's forward as one frame (31 compiled graphs → 5).
- Status: DEFAULT (scoped: POLAR-1-S, -M, -L; hybrid float32; not reproducible; no-op on MH-1). Decision 58; §19 sixteenth amendment.
- Measured: POLAR-1-S force-call kernels NaCl 3.60 → 2.73 ms (−24 %), NbCl5 25.26 → 20.76 ms (−18 %); with `polar_graph_sums`, same-process b/a NaCl 0.7634 / 0.7589, NbCl5 0.8225 / 0.8231 (local); L4 NaCl 0.8184 / 0.8108, NbCl5 0.8395 / 0.8366. Gate passed S, M, L (e.g. POLAR-1-L NbCl5 2×2×2: 0.88 / 0.96 / 0.75).
- Why: the TorchScript break inside a loop made dynamo skip the whole forward frame.
- Sources: [opt4-T02](validation/opt4-T02.md); SPEC decision 58.

<a id="model-polar-graph-sums"></a>
#### model-polar-graph-sums: `polar_graph_sums` (OPT4-T02b)
- Switch: `RewriteConfig.polar_graph_sums` (same rule as `fx_e3nn_linear`)
- Engine: FM
- Date/round: 24 October 2026, round 4 candidate 2
- What: `scatter_sum(x, batch)[batch]` → `x.sum(0, keepdim=True).expand_as(x)` (and mean/expand forms) at eight sites of `PolarMACE.forward`, removing the float64 serial `index_put` backward.
- Status: DEFAULT (scoped: as `fx_e3nn_linear`). Decision 58.
- Measured: alone NaCl 3.60 → 3.29 ms (−8 %), NbCl5 25.26 → 22.93 ms (−9 %) force-call kernels; no gain on top of `fx_e3nn_linear` (2.74 / 20.70 ms both).
- Why: kept for determinism (no atomics, no `index_put`).
- Sources: [opt4-T02](validation/opt4-T02.md); SPEC decision 58.

<a id="model-ffwd-polar"></a>
#### model-ffwd-polar: `fused_folded_forward` on POLAR-1-S
- Switch: `fused_folded_forward` (explicit on)
- Engine: FM
- Date/round: 27 October 2026, round 8 T03
- What: A/B on POLAR-1-S NbCl5.
- Status: REJECTED (off for non-MH-1 by decision 64).
- Measured: b/a 1.004 (pairs 0.99–1.01); the probe saw the kernel only in self-checks (not investigated).
- Why: neutral.
- Sources: [opt8-T02](validation/opt8-T02.md) "MH-1 family".

<a id="model-ffwd-omol"></a>
#### model-ffwd-omol: `fused_folded_forward` on OMOL-0 XL
- Switch: `fused_folded_forward` (explicit on)
- Engine: FM
- Date/round: 27 October 2026, round 8 T03
- What: A/B on OMOL-0 XL NbCl5 (folded width 129, 5 parts, 7 warps).
- Status: REJECTED (off by decision 64).
- Measured: b/a 1.035 (pairs 1.031–1.044), slower.
- Why: the wider `a` costs the in-kernel product more than the `w` read saves.
- Sources: [opt8-T02](validation/opt8-T02.md).

<a id="model-gate-t03d"></a>
#### model-gate-t03d: gate tool applies rule 13 (OPT4-T03d)
- Switch: none (tooling: `tools/force_stress_gate.py` `build_with_rule13`)
- Engine: FM
- Date/round: 24 October 2026 (merged `3e06d243`)
- What: the gate tool re-plans after a self-check failure as production does, recording `rule13_revisions`.
- Status: DEFAULT (tooling; not a speed item).
- Measured: A100 OMOL-0 XL NaCl gate passed 1.14 / 1.14 / 0.89 (rule 13 path not exercised).
- Why: OMOL-0 on the A100 failed `reassociate_first`'s self-check (1.2–1.3e-4 vs 1e-4) inside the tool, not in production. `tools/ab_rewrites.py` still lacks this ([rad-tables-omol](#rad-tables-omol)).
- Sources: [opt4-T03](validation/opt4-T03.md) §9.

<a id="model-other-models"></a>
#### model-other-models: other models on the research engine
- Switch: `main.py --model` (benchmark model table, MACE-OFF labels, `polar1-s|m|l`, `omol0-xl`)
- Engine: RE
- Date/round: 15–16 October 2026
- What: run any benchmark model label; the MH-1 radial rewrites switch off for other architectures; MACE-POLAR-1 and MACE-OMOL-0 take the MH-1 rewrites; MACE-OFF with the generic rewrites.
- Status: MEASUREMENT.
- Measured: all ten benchmark models pass the accuracy gate, 2.9–12.2× faster than stock ASE; MACE-OFF default 3.9–16.8× faster than stock ASE and 2.1–3.2× faster than the plain GPU engine; 15 October default NVT at 300 K (all-models run, MH-1) NaCl 3.53, NaNbO3 23.79, NbCl5 28.58 ms. OMOL-0 collapses in NaCl MD (as in stock ASE).
- Why: –
- Sources: `ResearchModule:LT/studies/other_models/README.md`; ResearchModule AGENTS.md.

### Alternative backends and external comparisons

<a id="ext-hybrid"></a>
#### ext-hybrid: cuEq/OEq hybrid backend (idea 1)
- Switch: RE `--backend hybrid` (`--backend auto` = hybrid for the gpu engine), environment `mace_cueq_oeq`, `oeq_no_nvcc.sh`; FM `EngineConfig.backend` (`backend_hybrid_for_graph_engine`; OEq prepared in process)
- Engine: both
- Date/round: RE screens 8 October 2026; production decision 9 October 2026 (user); `main.py` default 10 October 2026; FM 1a-T00, 1a-T15
- What: OpenEquivariance 0.7.0 fused convolution replaces cuEq's gather / tensor-product / scatter; cuEq symmetric contractions and linears kept.
- Status: DEFAULT.
- Measured: local FP32 captured step cuEq → hybrid NaCl 9.668 → 7.075 (26.8 %), NaNbO3 84.587 → 55.920 (33.9 %), NbCl5 72.286 → 55.650 ms (23.0 %); peak allocated −29.1 to −50.9 %. Coraline: 11.323 → 9.187 (18.9 %), 96.152 → 73.382 (23.7 %), 80.313 → 68.124 ms (15.2 %). 12 October ablation: cuEq FP32 backend +91.9 %, e3nn +510 % (NaCl). FM 5-T09 smoke: hybrid 3.95 / 4.01 ms vs cuEq 7.25 / 7.24 ms (NaCl).
- Why: fused convolution.
- Sources: `ResearchModule:LT/studies/hybrid_oeq/rtx5060ti.md`, `coraline.md`; `ResearchModule:LT/studies/remaining_optimizations/proposal.md` §1; `ResearchModule:LT/docs/status.md`; [5-T09](validation/5-T09.md); [switch-map](design/switch-map.md) §2.

<a id="ext-jax-mliap"></a>
#### ext-jax-mliap: MACE-JAX and LAMMPS ML-IAP (idea 8)
- Switch: separate environments `mace_jax_screen_20261008`, `mace_mliap_screen_20261008`
- Engine: RE
- Date/round: 8–9 October 2026
- What: alternative MACE backends.
- Status: REJECTED (ML-IAP slower; fused JAX failed accuracy).
- Measured: ML-IAP force call NaCl 29.34–31.83 ms, NaNbO3 129.32–129.89 ms vs PyTorch 8.71 / 86.92 ms; fused cuEq JAX NaCl 8.5–8.7 ms, NaNbO3 62–64 ms but force ratios 2.8–3.5 (NaCl) and 14–41 (NaNbO3).
- Why: slower or inaccurate.
- Sources: `ResearchModule:LT/studies/remaining_optimizations/report.md` (Idea 8); `ResearchModule:LT/studies/alternative_backends/README.md`.

<a id="ext-symmetrix"></a>
#### ext-symmetrix: Symmetrix-XL engine (roadmap item 13) and LAMMPS rematch
- Switch: separate venvs `symmetrix-xl-eval-20261014`, `~/environments/symmetrix_xl_lammps`
- Engine: both (RE comparison 14 October; FastMACE rematch 23 October)
- Date/round: 14 October 2026 (leto); 23 October 2026 (LAMMPS rematch, MACE-OMAT-0 medium, since its LAMMPS pair styles refuse MH-1)
- What: compare against Symmetrix-XL.
- Status: REJECTED (not adopted).
- Measured: leto MD step ours 3.98 / 27.43 / 34.01 ms vs Symmetrix-XL 9.96 / 52.42 / 64.28 ms (1.89–2.50× slower). Rematch vs FastMACE (OMAT-0 medium, local): 2.0× / 1.07× / 1.38× slower (AGENTS.md: native GPU path 1.1–2.5× slower per step; WSL needs `SYMMETRIX_A1_CUBLAS=0`).
- Why: slower.
- Sources: `ResearchModule:LT/studies/symmetrix_xl/README.md`; ResearchModule AGENTS.md.

<a id="ext-lunatp"></a>
#### ext-lunatp: LunaTP tensor product (OPT7-T00 and fairness follow-up)
- Switch: scratch venv `~/environments/lunatp_screen` (copy of `mace_cueq_oeq` with `e3nn==0.6.0` and `LunaEquivariance` 0.1.0 at `5a24ffe`)
- Engine: FM
- Date/round: 27 October 2026, round 7 (arXiv 2610.18393)
- What: map MH-1's second-interaction convolution (`uvu`, per-edge weights) onto LunaTP and time forward and backward against the vendored streaming kernels on recorded operands (CUDA-graph replays, L2 flush).
- Status: REJECTED (stop rule ≥ 1.3× on NaNbO3 not met).
- Measured: ours / LunaTP / ratio (ms), forward and backward: MH-1 NaCl 0.104 / 0.116 / 1.11, 0.240 / 0.195 / 0.81; NbCl3O 0.280 / 0.294 / 1.05, 0.615 / 0.669 / 1.09; NbCl5 0.853 / 0.936 / 1.10, 1.792 / 2.735 / 1.53; NaNbO3 1.236 / 1.347 / 1.09, 2.840 / 3.955 / 1.39; OMAT-0 similar (NaCl backward 0.76). Within 1.1e-6 relative. `create_graph` 0.8–1.2 ms per rebuild. Fairness follow-up: the NaCl 0.76–0.81 was an artefact of subtracting input copies (direct: 0.251 vs 0.240 ms, 1.05); parameter sweep best (vector width 2) 0.92 NaCl, 1.09 NbCl3O, 1.22 NaNbO3, 1.50 NbCl5. Nsight: LunaTP executes ≈ 25 % fewer instructions but moves 1.5× the L2 bytes and 2.6× the L1 loads at 27–32 % occupancy (128 registers). sm_89 and A100 not measured.
- Why: the backward is bound by bytes on this GPU; LunaTP's published gains are against stock OEq on an H100.
- Sources: branch fastmace/opt7-plan: docs/validation/opt7-T00-T02.md, plans/opt-7.md (commit `1bbb23fa`).

<a id="ext-torchsim"></a>
#### ext-torchsim: TorchSim 0.6.2 comparison
- Switch: none
- Engine: FM
- Date/round: 25 October 2026, local, main `5a3166c`
- What: step-time comparison.
- Status: MEASUREMENT (recorded, not judged).
- Measured: MH-1 NaCl 2.90 vs best TorchSim 22.2 ms (7.7×), NbCl5 21.3 vs 81.5 ms (3.8×); OMAT-0 medium NaCl 1.62 vs 16.3 (10×), NbCl5 9.79 vs 39.4 (4.0×).
- Why: –
- Sources: [TorchSim](validation/torchsim-comparison-20261025.md).

<a id="ext-alchemi"></a>
#### ext-alchemi: NVIDIA ALCHEMI 0.2.0 comparison
- Switch: environment `fastmace_alchemi_compare`
- Engine: FM
- Date/round: 26 October 2026, local, FastMACE 1.0.0, OMAT-0 medium float32
- What: step-time comparison.
- Status: MEASUREMENT.
- Measured: NaCl 1.651 / 1.652 vs 15.724 / 15.895 ms (9.53× / 9.62×); NbCl5 9.901 / 9.921 vs 27.399 / 27.230 ms (2.77× / 2.74×); setup 23–32 s vs 324–326 s; peak memory NbCl5 1,018 vs 1,474 MiB.
- Why: –
- Sources: [ALCHEMI](validation/alchemi-comparison-20261025.md) §3.

<a id="ext-stock-ase"></a>
#### ext-stock-ase: stock ASE comparison (phase 4)
- Switch: none
- Engine: FM
- Date/round: phase 4 (date not recorded)
- What: FastMACE against the stock ASE calculator.
- Status: MEASUREMENT.
- Measured: FastMACE 10–23× faster on NaCl, 6–12× on NbCl5.
- Why: –
- Sources: [phase-4](validation/phase-4.md).
