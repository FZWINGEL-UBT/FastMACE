# FastMACE preprint draft (arXiv style)

Source: owner requests of 11 and 12 November 2026: a new paper in arXiv style on its own branch
(`fastmace/paper-arxiv-20261111`); a comparison with the Symmetrix-XL paper (arXiv 2610.01036) and "Update our
paper and make it overall higher quality"; copies of all references; and, on 12 November, "Review this review
and edit the paper" for an outside review of commit `7a7a4fd`. It replaces the ChatGPT-written draft reviewed in
`plans/notes/paper-review-20261111.md`.

## Build

```bash
latexmk -pdf main.tex
```

`main.tex` uses the standard `article` class with common packages (no external style file), and `references.bib`
with `unsrtnat`.

The built `main.pdf` is committed (owner request, 11 November 2026), so the paper can be read without a LaTeX
installation. Rebuild it and commit it together with any change to `main.tex` or `references.bib`. The other
build outputs are ignored by `paper/.gitignore`.

## The third revision (12 November 2026): answer to the outside review

The review made eight findings. All eight were checked against the code and the raw records and hold. The paper
was changed as follows.

| Finding | Check | Change in the paper |
| --- | --- | --- |
| 1. The setup test does not cover the packed forward table | True. `encode_packed8` runs no error check; the only fallback is for the exponent range; the 0.16 ratio is from an offline tool with Sr, Ti, O hard-coded; no self-check or test sees the packed table | Section 5.1 separates the two checks and says the packed values are unchecked for other checkpoints and species. Abstract and limitations say so |
| 2. Today's default has no dynamics validation; the reverse uses the cubic for values and slopes | True (`wprime_backward.py:1414-1433`) | Section 5.1 states that all three adjoints use the cubic. Section 7 has its own paragraph on the missing energy-conservation record |
| 3. The largest capacity runs predate the coordinate fix; `state_dtype="auto"` can pick float16 | True. The fix is commit `cf76159a` of 11 November; the capacity sources `7e4fbf4c` and `d036568b` do not contain it; their records show 1.3 meV/Å | Section 6.2 and Table 10 label the runs; Section 6.1 explains the automatic precision; Section 8.6 calls them capacity demonstrations |
| 4. The Symmetrix-XL comparison is a cold run without rebuilds | True. The log goes from 0 to 18.26 K with 0 neighbour-list builds | Abstract, Section 8.1, Table 7, Section 8.4 and the conclusion name the protocol; cold and thermal runs are separated |
| 5. Estimator and uncertainty | True. Table 4's A100 NbCl5 value is a median of block means (1.67 ms); the mean is 1.84 ms because the first block takes 2.7 ms | Sections 3.2, 8.1 and 8.2 define the estimator; Table 4 has a footnote; Section 8.3 says what "both orders" does not show |
| 6. Validation details | True: mixed-precision energy yardstick; the spline is refined and scored on the same points; distances below 0.3 Å are clamped without warning; 18 configurations | Section 3.1 (yardstick, configurations), Section 5.1 (sample reuse, domain) |
| 7. Configuration-specific provenance | True. The ALCHEMI study used Python 3.12.3, cuEquivariance 0.13.0, ASE 3.29.0; the 214,375-atom row predates the cell-list screen on that GPU | Appendix C lists the deviations and a protocol per record; Table 9 marks the row; the availability statement says what a public archive needs |
| 8. Ablations and workload coverage | True | The contribution claim is narrowed; tiled paired ratios are added (Table 6); Sections 8.3 and 8.7 say what is missing |

The checks also found three errors of the previous revision, now corrected: the FastMACE side of the 6 November
comparison has two processes on file, not three; the records table gave commit `4c574d2` for the TorchSim and
ALCHEMI comparisons (TorchSim is `5a3166c`, ALCHEMI is not recorded); and the protocol paragraph claimed one
skin and start state for comparisons that used others.

Further changes asked for by the review:

- **Structure.** The paper is organised around three claims (reassociation, streamed execution, reduced-state
  tiling). The adoption chronology and the origins table moved to Appendices A and B; attribution and the AI
  disclosure stay.
- **Methods.** A cost model for the first interaction; how the symmetric contraction is generated, with an
  example and its checks; the definitions and instruction shapes of the hybrid product; the wavefront's
  dependency construction, pseudocode, correctness argument, periodic boundaries and what still grows with size.
- **Literature.** The NVIDIA TF32x3 blog and the MACE-POLAR-1 paper are cited; Symmetrix-XL's downstream
  validation is attributed to MACE-MPA-0 in float64; the general-engine comparison is labelled unbatched; the
  sentence that a native evaluator "inherits" back ends from LAMMPS is replaced.
- **Open experiments.** Table 13 lists the five experiments the review ranks.

## The fourth revision (12 November 2026): answer to the second review

A second outside review, of commit `399069ec`, closed most reporting findings and raised six new ones. All six
were checked and hold.

| Finding | Check | Change in the paper |
| --- | --- | --- |
| 1. The packed-path forces are not "the gradient of the cubic model" | True. The reverse applies cubic Jacobians to adjoints from the packed forward | Section 5.1 gives the force as −J_c^T ∇φ(m_p) and says it is the gradient of neither energy; reporting the cubic energy would not help |
| 2. The OMAT-0 small comparison mixes protocols | True. FastMACE's 2.12 ms is from a thermal start, 200 steps after 100, with 13 and 14 rebuilds (`results/h200-omat0-small-20261108/defaults2/breakdown/summary.json`) | Table 7 labels the row "mixed" and brackets the quotient; Section 8.4 no longer uses it |
| 3. The wavefront storage account omits the partial-adjoint accumulator | True (`tiled.py` `_window`, `_close_now`) | Algorithm 2 names the accumulator; Section 6.1 gives M = 2,560 N_live + 2,048 N_open and the census values; "halves" is qualified |
| 4. "Thermalised" overstates the RTX 5060 Ti runs | True. The Symmetrix-XL run moves from 187 to 233 K in its timed window | Abstract, Sections 8.1 and 8.4 and the conclusion say "thermal initial velocities and neighbour rebuilds"; the sentence that the force call does not depend on the protocol is replaced |
| 5. Dates do not identify the coordinate path | True. `git merge-base` shows that no source revision in Table 10 contains the fix, including the two rows dated 11 November | Table 10's caption and Section 6.2 say that every row used float32 coordinates; the corrected path's paired cost on the A100 (1.028 / 1.029) is given |
| 6. Appendix C assigns the paired protocol too broadly | True for all four named studies, and also for OMAT-0 small and the tiled path | Appendix C lists seven deviations; Tables 5, 9 and 11 point to them |

Also repaired: the NVIDIA blog entry now links to the article itself.

A third review, of commit `f25e81be`, found the six findings resolved and no new substantive issue. It asked
for two clarifications, both made: the force expression in Section 5.1 is scoped to this message's
contribution, with the direct term written out, and its heading says only that the reverse is not consistent
with the evaluated energy; and the memory formula in Section 6.1 counts allocated peak capacities, since both
pools are preallocated. Its recommendation: put further effort into the experiments of Table 13, not into
another wording revision.

A fourth review, of commit `956824f3`, found one more reporting error, in text that had not changed: the paper
called the symmetric contraction "the only kernel FastMACE generates per model". That is wrong. The streaming
kernels are also rendered per model, from the coupling paths, the layout and the spline grid
(`spline_forward.py` `_render`, `wprime_backward.py` `render`). Section 4.3, Section 5.2 and the design table
now distinguish the synthesised contraction from the template-rendered streaming kernels, and state the
difference from Symmetrix-XL that does hold: FastMACE keeps PyTorch's execution of the model and its automatic
differentiation.

## The fifth pass (12 November 2026): eight more reporting findings

A fifth review, of commit `ac3fb74f`, re-read unchanged sections against the raw records. Its six "unresolved
earlier findings" are limitations the paper already states and need experiments, not edits. Its eight
reporting findings were checked and all hold.

| Finding | Check | Change in the paper |
| --- | --- | --- |
| Table 1 labels a PCIe profile as A100-SXM4 | True. `step-limiters-20261109/a100-packed8/gpu.txt` says A100 80GB PCIe (source `0df22417`); the SXM4 profile totals 6,208.5 µs in another configuration | Column relabelled; caption gives both revisions and the distinction |
| The H200 NbCl5 time has another schedule and a mean-median gap | True. Baseline side of the butterfly study at `e3049399`, eight blocks; medians 1.087 / 1.093, means 1.150 / 1.174, first block 1.59 / 1.65 ms | Footnote in Table 4, text in Section 8.2, entry in Appendix C |
| The hybrid paragraph overstates its probes | True. `range_probe.py` runs the reference MACE calculator, not the rewritten graph; `hybrid_lab.py` measures max \|Ĉ−C\| / (\|A\|\|B\|) on synthetic Gaussian operands | Section 5.3 says the production margin is not established, prints the error definition and names the operands |
| Hard restrictions are missing | True: periodic boundaries in three directions (`forces.py`), two-interaction ScaleShiftMACE only and no joint embeddings or PolarMACE for the tiled path (`tiled.py`), three cutoff-wide bins per axis for per-tile lists (`tile_list.py`) | New paragraph "Supported domain" in Section 3 |
| Figure 4 counts a second-interaction kernel with the first interaction | True. The 6 November profile's 90 µs `radial_table_forward` feeds the second interaction | Bar and prose changed to 0.34 ms and 1.71 ms; total unchanged |
| Float16 "fails the rule by factors of 12 to 51" uses the wrong denominator | True. Those are ratios to the float32 reference error; against the limit they are 4 to 17 | Sentence corrected |
| Operand splitting attributed to Ootomo and Yokota | True. Their paper credits Markidis et al. (2018) with the split and four products | Sections 5.3 and 9 credit Markidis et al. for the split and Ootomo and Yokota for the refinement to three products; the Markidis paper is cited and saved in `references/` |
| Appendix C gives one protocol for all capacity rows | True. The A100's 2.04 s is from a paired study at `d7d9e7ce`, four blocks of two calls | The capacity row is split into whole-system MD, device state and host state, with revisions |

## Where the numbers come from

Every number in the paper is taken from a record in this repository or from a cited paper. Nothing was measured
for the paper. Appendix C of the paper lists the record, revision and protocol behind each table and figure.
Additional sources used for the third revision:

| Paper item | Record |
| --- | --- |
| Cost model of the first interaction (Section 4.1) | `fastmace/rewrites/mh1/modules.py`, `results/first-interaction-20261111/README.md`; the operation counts are arithmetic on the model's sizes |
| Generated contraction (Section 4.3) | `fastmace/kernels/cuda/sc_generated.py`, `sc_factored.py`, `results/sc0-bound-20261109/README.md`, `results/other-ways-20261109/README.md` |
| Setup test, domain, packed table (Section 5.1) | `fastmace/rewrites/radial/__init__.py`, `spline.py`, `fastmace/kernels/triton/radial_table.py`, `fastmace/kernels/oeq_streaming/spline_forward.py`, `wprime_backward.py`, `tools/spline_packed_certificate.py` |
| Hybrid product (Section 5.3) | `fastmace/kernels/cutlass/hybrid_f16.h`, `results/hybrid-gemm-20261111/README.md` |
| Wavefront (Section 6.1) | `fastmace/model/tiled.py`, `docs/validation/tiled/tiled-host-state-dtype.md`, `results/tiled-speed-20261110/README.md`, `results/tiled-speed-20261111/README.md` |
| Coordinates (Section 6.2) | `results/tiled-speed-20261111/README.md` |
| Acceptance rule (Section 3.1) | `tools/force_stress_gate.py`, `fastmace/reference/calculators.py`, `results/sc0-op-20261109/h200/gate-metadata.json` |
| Block statistics (Table 4) | `results/hybrid-gemm-20261111/alex-ab/*.json`, `results/sc0-op-20261109/h200/ab-*.json` |
| Protocol of the comparisons (Table 7) | `results/sx012-profile-20261106/symmetrix-h200/work-langevin-*/log.lammps`, `tools/bench/loup_md.py`, the two RTX 5060 Ti comparison records |
| Symmetrix-XL's own numbers | arXiv 2610.01036, read as the arXiv HTML on 11 November 2026; a copy of the PDF is in `references/` |

## Open items before submission

- **The five experiments of Table 13.** None was run. The brief for them is
  `plans/notes/conductor-prompt-20261112-paper-experiments.md` in the main checkout (owner request of
  12 November 2026); it is not committed there yet and is for the conductor to pick up. It also lists the code
  items the reviews raised.
- **A public repository or archive** with checkpoint and structure identities, resolved configurations and
  commands. The availability statement says this is needed.
- **Author list, affiliation, acknowledgements, release plan.** Marked `TODO (owner)` in `main.tex`.
- **Bibliography.** Titles and authors of all 38 entries are confirmed. Still unconfirmed are publication
  details whose publisher pages could not be read: the proceedings names and DOIs of PyTorch 2, Triton and the
  SC '23 paper, the publisher and year of the Trefethen book, and the journal fields of MACE-MP-0, MACE-OFF and
  the NequIP infrastructure paper.
- **Reference copies.** `references/` holds 29 papers and one book excerpt, plus snapshots of the software
  references; `references/README.md` lists source and checksum. Check the licences before the repository
  becomes public.
- **MACE-POLAR-1 settings.** The record summaries do not state the total charge and spin of the timed runs.
- **Provenance.** `docs/current/omat0-a100.md` stops at 8 November and lists eight parts with no recorded origin.
- **Inconsistencies found in the records and code comments** (not changed, they belong to the conductor):
  several READMEs label the gate's energy error "meV" or "eV" where the tool reports meV/atom; comments in
  `spline_forward.py`, SPEC.md and `defaults-by-gpu.md` say the reverse keeps the float32 table "for its slopes",
  while it also takes its values from it; `spline_forward.py` calls the packed table "a certified
  approximation" although no setup check exists; the spline has 731 intervals in one record and 736 in another.
