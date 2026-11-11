# FastMACE preprint draft (arXiv style)

Source: an owner request (11 November 2026) for a new paper in arXiv style, on its own branch
(`fastmace/paper-arxiv-20261111`). It replaces the ChatGPT-written draft reviewed in
`plans/notes/paper-review-20261111.md` and addresses that review as far as the records allow.

## Build

```bash
latexmk -pdf main.tex
```

`main.tex` uses the standard `article` class with common packages (no external style file), and `references.bib`
with `unsrtnat`. Build outputs are ignored by `paper/.gitignore`.

## Where the numbers come from

Every number in the paper is taken from a record in this repository. Nothing was measured for the paper.

| Paper item | Record |
| --- | --- |
| Kernel-time profile (Table 1) | `results/step-limiters-20261109/` (`h200/limiters.csv`, `a100-packed8/limiters.csv`) |
| Routes per GPU (Table 2) | `docs/current/defaults-by-gpu.md` |
| Accuracy ratios (Table 3) | `results/sc0-op-20261109/README.md` (H200) and `results/hybrid-gemm-20261111/README.md` (A100), gate sections; threshold history in `docs/current/omat0-a100.md` §3.7 and §3.8.3 |
| Step times (Table 4) | `results/sc0-op-20261109/README.md` (H200), `results/hybrid-gemm-20261111/README.md` (A100), `results/ada-5060-screen-20261110/` (Ada: `ada-lines.txt`, pack X10; 5060 Ti: `README.md`, stage 3) |
| A100 history (Figure 2) | `docs/validation/omat0/loup-a100.md` (3 Nov), `docs/current/omat0-a100.md` (4–8 Nov), `results/hybrid-gemm-20261111/README.md` (11 Nov) |
| Paired ratios (Table 5) | `docs/current/omat0-a100.md` §3, `docs/current/defaults-by-gpu.md`, the ledger rows of 8–11 November |
| Engine comparisons (Table 6) | `docs/validation/phases/phase-4.md` (ASE), `docs/validation/comparisons/` (TorchSim, ALCHEMI), `docs/validation/omat0/loup-a100.md` and `docs/current/omat0-a100.md` §6 (Symmetrix-XL) |
| Neighbour search | `results/binned-screen-5060ti-20261111/README.md` |
| Capacity (Table 7) | `results/rung-b-20261109/README.md`, `results/tiled-speed-20261111/README.md`, `docs/validation/tiled/tiled-host-state-dtype.md` |
| Energy drift | `docs/validation/omat0/generic-radial-tables.md` §5 |
| Origins (Table 8) | `docs/current/omat0-a100.md`; for 9–11 November the ledger and the result READMEs |

## Open items before submission

- **A consolidated benchmark.** One revision, all four GPUs, the reference implementation and current versions of
  the other engines, with accuracy numbers from the same run. The paper says this is missing.
- **Author list, affiliation, acknowledgements, release plan.** Marked `TODO (owner)` in `main.tex`. The author
  and affiliation there are taken from the commit identity and need confirming.
- **Bibliography.** Three entries are verified against arXiv (marked `VERIFIED` in `references.bib`). The others
  are from memory: check authors, venues and years.
- **Provenance.** `docs/current/omat0-a100.md` stops at 8 November and lists eight parts with no recorded origin
  (its §7). Table 8 marks three of them "not recorded". The LLM research reports are credited generically; decide
  how to name them.
- **Energy drift with the packed forward table.** Not measured; the paper says so.
- **The comparison with stock ASE** names the model as MH-1 from the step times; the record does not state it on
  the lines read.
