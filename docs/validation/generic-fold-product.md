# The C3 fold for the generic family: `generic_fold_product_linear` (4–5 November 2026)

Owner, 4 November 2026: "Yes, port it" (idea 2 of the seven OMAT-0 A100 proposals), then on 5 November "go ahead".
Branch `fastmace/generic-fold-product`.

**Verdict: default on the RTX 4000 Ada and every A100 (decision 66, `planner.fixed.DECISION_66_GENERIC_FOLD_PRODUCT_GPUS`).**
The fold is faster in both build orders on both measured GPUs. It is a rounding-class change. The RTX 5060 Ti was not
measured and keeps it off unless `fold_product_linear="on"` is asked for.

## 1. What it does

MH-1's C3 fold (`lin-fold-product-linear`, OPT-T08) needs `species_tables` and so never applied to the generic
family. For a generic model whose first product has no skip term (the density family, MACE-OMAT-0, whose density first
interaction returns `sc=None`), the new record `generic_fold_product_linear` folds `products.0.linear` (W_p) into its
three consumers and replaces it with `Identity`:
- **Interaction 1's `linear_up`:** becomes `compose(W_p, linear_up)`.
- **Readout 0's linear:** becomes `compose(W_p, readout)`.
- **Interaction 1's `SpeciesSkip`:** becomes `compose_species_skip(W_p, skip)`, which per species is `A @ W_s`,
  computed in float64 at apply time.

The record shares the config field `fold_product_linear` with MH-1's record and needs `block_linear` and
`generic_species_skip`. OMAT-0's first density block applies `skip_tp(message, node_attrs)` to the message, after
geometry-dependent processing. That skip is not folded and is never replaced by a species table; the fold touches
only the second interaction's skip, whose input is `products.0`'s output.

On the real OMAT-0 medium default plan (leto probe, 4 November), all the consumers have the types the fold needs:
- `products.0.linear` is a BlockLinear (128x0e+128x1o).
- `interactions.1.skip_tp` is a SpeciesSkip over the 0e and 1o segments.
- `interactions.1.linear_up` and `readouts.0.linear` are BlockLinears.

## 2. Checks

| Check | Result |
|---|---|
| CPU, tiny density model (`tests/cpu/test_generic_fold_product.py`) | float64 energy, forces and stress within 1e-10 relative of the stock model; the self-check catches a wrong skip matrix; MP-0 (first skip term) and MH-1 are not taken by the record |
| Targeted CPU tests on leto (`df0d088f`) | 319 passed, 29 skipped; ruff clean |
| Force gate with the fold, RTX 4000 Ada (`rtx4000ada-df0d088f/gate-fold.log`) | pass. NaCl energy r1.24, forces r0.77, stress r0.69; NbCl5 r1.01, r0.98, r0.66; SrTiO3 3×3×3 r1.69, r0.73, r0.67 |
| Force gate with the fold, A100-SXM4-80GB (`a100-sxm4-df0d088f/gate-fold.log`) | NaCl and SrTiO3 pass (stress r1.29 and r1.51). NbCl5 stress 1.59e-7 r2.09 misses: the A100 plan's accepted exception (default plan r2.5–3.0; tf32x3 GEMMs + tensor-core forward, `omat0-tp-backward.md` §3.1). Forces r0.80–1.05 |

Ratios are max |error| against float64 MACE divided by the ASE float32 calculator's error; the gate's limit is 2 (or
its floor).

## 3. MD step A/B

`tools/modal/fold_product_ab.sh`, `tools/ab_rewrites.py`: OMAT-0 medium, side a the default plan (where decision 66
kept the fold off), side b with `fold_product_linear=on`; 1 fs, skin 0.5 Å, 200 warm-up steps, 8 blocks of 200 steps,
in one process, abab and abba.

| GPU | SrTiO3 10×10×10 (5,000 atoms), abab / abba | NbCl5 (1,944 atoms), abab / abba |
|---|---|---|
| RTX 4000 Ada (leto, `mace_cueq_oeq`) | 0.9986 / 0.9960 (30.41 → 30.37 ms) | 0.9911 / 0.9849 (6.07 → 6.02 ms) |
| A100-SXM4-80GB (Modal) | 0.9928 / 0.9931 (14.37 → 14.27 ms) | 0.9805 / 0.9828 (3.39 → 3.33 ms) |

The gain is small, as expected: the fold removes one 128-channel GEMM pair and its backward per step. On the Ada's
SrTiO3 it is within noise (0.14 % in one order), but below 1 in both orders on both structures, which is decision
66's rule.

## 4. Runs and provenance

All in `results/generic-fold-product-20261104/` (README there), source `df0d088f`:
- **leto:** RTX 4000 Ada, NAS `mace_cueq_oeq` (torch 2.14.1+cu130), FastMACE from a `git archive` copy. The launcher
  ran the idle check (no compute process on the GPU, no other user's active process) before the job; its output was
  not saved.
- **Modal:** A100-SXM4-80GB, driver 610.57.04, profile `second`, `MODAL_IMAGE_BUILDER_VERSION=2025.06`; $1.05
  (`modal.json`, ledger).

## 5. Not done

- The RTX 5060 Ti and the A100 80GB PCIe were not measured. The PCIe card takes the default through the "every A100"
  list.
- Other density-family models (MPA-0, MP-0b3) take the same rule on the same GPUs but were not measured; the fold only
  removes work there too.
- No tiled-mode measurement.
