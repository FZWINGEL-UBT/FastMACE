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

## `NbCl5_R2SCAN.cif`

- NbCl5 (C2/m), the cell of the R2SCAN benchmark set, for the NPT melt-quench protocol of owner
  decision D1 (SPEC §18 decision 40).
- Copied byte for byte (LF line endings) from the ResearchModule repository:
  `G:/Margraf/ResearchModule/Structures/Benchmark/R2SCAN/NbCl5.cif`, at tag
  `research-engine-baseline-20261017` (blob `f872cdc8d28ce6689060de3ae28766b155645a82`; the file was
  last changed in commit `6f079e7f`, 24 September 2026). SHA-256 of the copy:
  `1d662303cbc693ec1beb1b0af9adccbf0392d0919cddfe4628c5dc5dc1f8ae99`.
- 72 atoms, Nb12Cl60 (a = 18.964 Å, b = 18.178 Å, c = 6.007 Å, β = 90.47°); a 3×3×3 supercell has
  1944 atoms.

## `npt_anneal_NbCl5.toml` and `npt_anneal_NbCl5_short.toml`

- The NPT melt-quench protocol of D1 (plan task 3b-T07), for `fastmace run`.
- The short variant has the same stages and settings with a schedule a tenth as long. It is the
  smoke run, not a production protocol.
