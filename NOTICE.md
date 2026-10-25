# Third-party notices

FastMACE is distributed under the MIT licence (`LICENSE`). The files below,
shipped in the `fastmace` package, come from third parties and keep their own
licences. Each lives in a directory with that licence's text in a `LICENSE`
file and a `NOTICE.md` with the details of the copy.
`tests/cpu/test_notice.py` checks that every file in the table exists and that
its directory has a `LICENSE`.

| File | Origin | Licence | Changes |
| --- | --- | --- | --- |
| `fastmace/kernels/oeq_streaming/factorized_projected.cuh` | OpenEquivariance PR 223 (P. Fuchs), commit `ef051c3f9efa2d1daf74cfbe3beaad152235f47d`, <https://github.com/PASSIONLab/OpenEquivariance/pull/223>; © 2025 The Regents of the University of California, through Lawrence Berkeley National Laboratory | BSD-3-Clause | adapted (`ir_mul` node offsets) |
| `fastmace/kernels/oeq_streaming/schedule.py` | as above | BSD-3-Clause | adapted (accumulator slots, layout) |
| `fastmace/kernels/oeq_streaming/jinja_env.py` | as above, the PR's `templates/jinja_utils.py` | BSD-3-Clause | adapted (CUDA only) |
| `fastmace/longrange/data/dftd3_params.npz` | torch-dftd 0.5.1, `torch_dftd/nn/params/dftd3_params.npz`, <https://github.com/pfnet/torch-dftd>; © 2021 Preferred Networks, Inc. Holds S. Grimme's DFT-D3 reference data (see `docs/licences.md`) | MIT | none (SHA-256 `52b7b35a957df55b813035ffde3a6b5a41dec585bdb1e281fb8ba67a729f3d92`) |

Licence texts: `fastmace/kernels/oeq_streaming/LICENSE` (BSD-3-Clause,
OpenEquivariance) and `fastmace/longrange/data/LICENSE` (MIT, torch-dftd).

## Ported code inside FastMACE's own modules

- **Ported code.** Several modules re-implement or port code paths of
  MACE (MIT, © 2022 ACEsuit/mace; for example `fastmace/rewrites/engine.py`
  holds mace-torch 0.3.16's `get_symmetric_displacement` with PR 1498's
  one-graph path; also PRs 1614, 1641, 1711 and the PolarMACE stress fix)
  and ASE 3.27 (LGPL-2.1-or-later; integrators and optimisers, ported for
  agreement with ASE to rounding). PR 1762's PolarMACE change is applied at
  run time to the installed MACE source and is not shipped. Whether the ASE
  ports count as derived works of LGPL code is an open question for the owner
  (`docs/licences.md`, "Open questions").
- **Protocol data** (`fastmace/protocols/data/`, see its `SOURCES.md`): two
  structure files copied from the author's ResearchModule repository and the
  author's own protocol files.
