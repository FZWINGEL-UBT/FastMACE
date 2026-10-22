# Sources of the shipped protocol data

## `NaNbOCl4_I4m_Ma2026.cif`

- The Ma 2026 I4/m polymorph of NaNbOCl4, as owner decision D3 (SPEC §18 decision 33) names it.
- Copied byte for byte (LF line endings) from the ResearchModule repository:
  `G:/Margraf/ResearchModule/Structures/NaMOCl4/NaNbOCl4_I4m_Ma2026.cif`, at tag
  `research-engine-baseline-20261017` (blob `fe8a09b47883b91b65125ba76dab056e0d6e0c6c`; the file was
  last changed in commit `ed6f3610`, 8 October 2026). SHA-256 of the copy:
  `b1cddd834ce21c1277de0cde8c42cd12c1bbae542bf322644b4b5c6b47ae0cca`.
- 28 atoms, Na4Nb4Cl16O4; a 3×3×3 supercell has 756 atoms.
- The file is written as P1 with a = 8.304 Å, b = 9.875 Å, c = 8.02 Å and γ = 89.52°. That is not a
  tetragonal I4/m metric (a ≠ b, γ ≠ 90°). It is used as given (conductor, 22 October 2026).

## `swap_anneal_NaNbOCl4.toml` and `swap_anneal_NaNbOCl4_short.toml`

- The swap-MC annealing protocol of D3 (plan task 3-T14), for `fastmace run`.
- The short variant has the same stages and settings with a schedule about a tenth as long. It is
  the validation run for a paid GPU (the Modal L4), not a production protocol.
