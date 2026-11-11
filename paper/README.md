# FastMACE preprint draft (arXiv style)

Source: owner requests of 11 November 2026: a new paper in arXiv style on its own branch
(`fastmace/paper-arxiv-20261111`), then a comparison with the Symmetrix-XL paper (arXiv 2610.01036), then
"Update our paper and make it overall higher quality". It replaces the ChatGPT-written draft reviewed in
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

## What the second revision changed

- **Symmetrix-XL.** The paper now says what Symmetrix-XL does and where it is ahead, in the introduction, in each
  mechanism section, in the validation section and in a design table. Three statements were corrected: its
  reverse pass also recomputes the radial factors, it also gates its splines, and it runs under ASE, on CPUs and
  on many GPUs, not only under LAMMPS.
- **Accuracy.** Table 3 gives absolute errors, with the energy. The energy passes through the absolute floor, not
  through the factor of three; the paper says so.
- **Comparisons.** Every same-machine run against Symmetrix-XL from November is in Table 6, with GPU memory, and
  a stage breakdown is in Figure 5. The published Symmetrix-XL numbers are quoted with the caveat that the sizes
  and protocols differ.
- **New material from existing records.** NbCl5 step times, step time against system size, a 320-million-atom
  float32 call, step times for other models, table sizes and test ratios, setup times, software versions.
- **Presentation.** A contributions list, an algorithm listing for the reverse kernel, three new figures (MD step,
  wavefront, stage breakdown), an expanded related-work section, an appendix of records.
- **Bibliography.** 36 entries, each marked VERIFIED, SECONDARY or UNVERIFIED in `references.bib`. Copies of the
  references were added afterwards in `references/` (owner request of 11 November 2026), and TorchSim and the
  ALCHEMI Toolkit got the citations their projects ask for.

## Where the numbers come from

Every number in the paper is taken from a record in this repository or from the Symmetrix-XL paper. Nothing was
measured for the paper. Appendix A of the paper has the same list with source revisions.

| Paper item | Record |
| --- | --- |
| Kernel-time profile (Table 1) | `results/step-limiters-20261109/` (`h200/limiters.csv`, `a100-packed8/limiters.csv`) |
| Routes per GPU (Table 2) | `docs/current/defaults-by-gpu.md` |
| Absolute errors (Table 3) | `results/sc0-op-20261109/h200/gate.log` (H200), `results/hybrid-gemm-20261111/alex-gate/gate-hybrid.log` (A100); threshold history in `docs/current/omat0-a100.md` §3.7 and §3.8.3; protocol in `results/sc0-op-20261109/h200/gate-metadata.json` |
| Step times (Table 4) | `results/sc0-op-20261109/README.md` (H200), `results/hybrid-gemm-20261111/README.md` (A100), `results/ada-5060-screen-20261110/` (Ada: `ada-lines.txt`, pack X10; 5060 Ti: `README.md`, stage 3), `results/step-limiters-20261109/h200-butterfly/run.log` (H200, NbCl5) |
| A100 history (Figure 4) | `docs/validation/omat0/loup-a100.md` (3 Nov), `docs/current/omat0-a100.md` (4–8 Nov), `results/hybrid-gemm-20261111/README.md` (11 Nov) |
| Paired ratios (Table 5) | `docs/current/omat0-a100.md` §3, `docs/current/defaults-by-gpu.md`, the ledger rows of 8–11 November |
| Symmetrix-XL runs (Table 6) | `docs/validation/omat0/loup-a100.md`, `results/symmetrix-xl-h200-loup-20261105/`, `results/sx012-profile-20261106/`, `results/symmetrix-xl-h200-omat0-small-20261109/`, `results/vram-compare-rtx5060ti-20261105/`, `results/nacl-compare-rtx5060ti-20261105/` |
| Stage breakdown (Figure 5) | `results/sx012-profile-20261106/README.md` §2, Table 1 |
| General engines (Table 7) | `docs/validation/phases/phase-4.md` (ASE, the Langevin row; the model is MH-1 per `results/4-T13-local-20261024-speed/`), `docs/validation/comparisons/` (TorchSim, ALCHEMI) |
| Neighbour search, size (Table 8) | `results/binned-screen-5060ti-20261111/README.md`, `results/shuffled-order-5060ti-20261111/README.md`, `results/rung-b-20261109/README.md` |
| Capacity (Table 9) | `results/rung-b-20261109/README.md`, `results/tiled-speed-20261111/README.md`, `docs/validation/tiled/tiled-host-state-dtype.md` |
| Other models (Table 10) | `results/off-polar-a100-20261108/README.md`, `docs/current/defaults-by-gpu.md` (OMAT-0 small) |
| Design (Table 11) | `docs/current/omat0-a100.md` §6 and the Symmetrix-XL paper |
| Origins (Table 12) | `docs/current/omat0-a100.md`; for 9–11 November the ledger and the result READMEs; `fastmace/kernels/oeq_streaming/wprime_backward.py` (the reverse kernel's vendored body) |
| Radial tables (Section 5) | `fastmace/rewrites/radial/`, `results/first-interaction-20261111/README.md`, `results/packed-table-lab-20261109/packed_certificate.json`, `results/spline-w-h200-20261107/` |
| Energy drift | `docs/validation/omat0/generic-radial-tables.md` §5, `docs/validation/tiled/tiled-host-state-dtype.md` |
| Setup times | `results/spline-w-h200-20261107/loup-*.json`, `results/sx012-profile-20261106/fastmace-h200-breakdown/loup/`, `results/omat0-a100-profile-20261104-4b70e3f2/loup/` |
| Symmetrix-XL's own numbers | arXiv 2610.01036, read as the arXiv HTML on 11 November 2026 (Table 1, Sections 3–7, Appendices A and B). The values of its Figure 2 exist only in the image and are not used. |

## Open items before submission

- **A matched benchmark against Symmetrix-XL.** Both engines on one A100 at 5,000 and at about one million
  atoms, with moving atoms, plus the LAMMPS interface with cuEquivariance as baseline. The paper names this as the
  main missing experiment.
- **A 100 ps NVE run on the 5,000-atom benchmark** with today's defaults, which include the packed forward
  table. The only whole-system drift record is 216 atoms for 10 ps from 2 November.
- **Downstream validation.** None exists. Symmetrix-XL reproduces Matbench Discovery and three other workflows.
- **A consolidated benchmark** of one revision on all four GPUs.
- **Author list, affiliation, acknowledgements, release plan.** Marked `TODO (owner)` in `main.tex`. The author
  and affiliation there are taken from the commit identity and need confirming.
- **Bibliography.** Titles and authors of all 36 entries are now confirmed, most of them from the downloaded
  copies in `references/`. What is still unconfirmed are publication details whose publisher pages could not be
  read: the proceedings names and DOIs of PyTorch 2, Triton and the SC '23 paper, the publisher and year of the
  Trefethen book, and the journal fields of MACE-MP-0 (J. Chem. Phys.), MACE-OFF (JACS) and the NequIP
  infrastructure paper (Digital Discovery).
- **Reference copies.** `references/` holds 28 papers and one book excerpt, plus README and citation snapshots
  of the software references; `references/README.md` lists the source and checksum of each. Only Drautz's Phys.
  Rev. B paper is missing (paywalled). The folder is 158 MB, of which one PDF is 84 MB. Check the licences before
  the repository becomes public.
- **Provenance.** `docs/current/omat0-a100.md` stops at 8 November and lists eight parts with no recorded origin
  (its §7). Table 12 marks three of them "not recorded". The LLM research reports are credited generically; decide
  how to name them.
- **Record inconsistencies found while checking.** Several READMEs label the gate's energy error "meV" or "eV"
  where the tool reports meV/atom (for example `results/sc0-op-20261109/README.md`). The spline has 731 intervals
  in the 7 November record and 736 in the 9 November one. `results/first-interaction-20261111/README.md` says the
  packed table "takes the slope from" its quadratic, while the code and SPEC say the reverse pass keeps the
  float32 cubic; the paper follows the code.
