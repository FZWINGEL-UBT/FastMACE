# Generic fold of `products.0.linear` (4–5 November 2026)

Report: `docs/validation/generic-fold-product.md`. Model MACE-OMAT-0 medium (checkpoint sha256 d4b14be9…), float32
default plan with `fold_product_linear=on` on side b. Source `df0d088f` (branch `fastmace/generic-fold-product`).
Script: `tools/modal/fold_product_ab.sh` (force gate on NaCl, NbCl5, SrTiO3 3×3×3; MD-step A/Bs on SrTiO3 10×10×10
and NbCl5, abab and abba).

| Folder | GPU | What |
|---|---|---|
| `rtx4000ada-df0d088f` | NVIDIA RTX 4000 Ada Generation (leto, NAS `mace_cueq_oeq`, idle check by the launcher) | gate passes; MD step b/a SrTiO3 0.9986 / 0.9960, NbCl5 0.9911 / 0.9849 |
| `a100-sxm4-df0d088f` | NVIDIA A100-SXM4-80GB (Modal, driver 610.57.04, profile `second`, $1.05) | gate: NbCl5 stress r2.09 (the accepted A100 exception), the rest passes; MD step b/a SrTiO3 0.9928 / 0.9931, NbCl5 0.9805 / 0.9828 |
