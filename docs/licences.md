# Licences

## FastMACE

FastMACE is distributed under the **MIT licence** (`LICENSE`; SPEC §14, owner
decision 4). The few third-party files shipped inside the package keep their
own licences; they are listed with origin and licence in `NOTICE.md` at the
top of the repository:

- the vendored OpenEquivariance receiver-streaming kernels
  (`fastmace/kernels/oeq_streaming/`, BSD-3-Clause, Lawrence Berkeley
  National Laboratory / University of California);
- the D3 reference tables `fastmace/longrange/data/dftd3_params.npz`
  (from torch-dftd 0.5.1, MIT, Preferred Networks); see the assessment below.

## Runtime dependencies

FastMACE depends on these packages; it does not copy them. Each keeps its own
licence, which applies when you install it. Values are read from the installed
package metadata (`METADATA`, licence files) of the qualified stack in the
local WSL environment `~/environments/mace_cueq_oeq` on 25 October 2026;
simple-dftd3 and tad-dftd3 from `~/environments/mace_cueq_latest` (they are not
in the hybrid environment).

| Package (version checked) | Needed for | Licence (from metadata) |
| --- | --- | --- |
| torch 2.14.1+cu130 | base | `License-Expression: Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`; PyTorch's own code is BSD-3-Clause style, the others are bundled third-party parts. The CUDA wheels pull in NVIDIA runtime libraries under NVIDIA's licences. |
| mace-torch 0.3.16 | base | MIT (classifier and `LICENSE.md`, © 2022 ACEsuit/mace) |
| e3nn 0.4.4 (through MACE) | base | MIT |
| ase 3.27.0 | base | `LGPL-2.1-or-later` |
| numpy 2.2.6 | base | BSD (classifier, `LICENSE.txt`) |
| cuequivariance 0.12.0, cuequivariance-torch 0.12.0 | base | Apache-2.0 (licence file; the `License:` field is empty) |
| cuequivariance-ops-cu13 / cuequivariance-ops-torch-cu13 0.12.0 | fused cuEq kernels, chosen by the user's CUDA build | NVIDIA proprietary: "LICENSE AGREEMENT FOR NVIDIA MATH LIBRARIES SOFTWARE DEVELOPMENT KITS" (`Other/Proprietary License`). FastMACE depends on them and never vendors or redistributes them. |
| triton 3.8.0 (through torch) | Triton kernels | MIT |
| filelock 3.32.3, PyYAML 6.0.3 | base / input files | filelock: `MIT`; PyYAML: MIT |
| tomli (Python 3.10 only) | TOML input files | not installed in `mace_cueq_oeq`, so not verified here (**owner/6-T02 follow-up**) |
| openequivariance 0.7.0 | `[oeq]` | `BSD-3-Clause` |
| jinja2 3.1.6, cuda-bindings 13.3.1 | streaming kernels (template, NVRTC) | jinja2: BSD; cuda-bindings: `LicenseRef-NVIDIA-SOFTWARE-LICENSE` (NVIDIA) |
| graph-longrange 0.4.0 | `[polar]` | MIT (`LICENSE.md`; the metadata has no licence field) |
| torch-dftd 0.5.1 | `[d3]`, `[d3-reference]` | MIT (`LICENSE`, © 2021 Preferred Networks, Inc.) |
| dftd3 1.6.0 (simple-dftd3) | `[d3-reference]` only | `LGPL-3.0-or-later` |
| tad-dftd3 0.7.0 | test reference only, not an extra | Apache-2.0 |

None of the LGPL packages (ASE, simple-dftd3) is copied into FastMACE; they
are imported as separate installed libraries, which the LGPL permits for an
MIT program.

## Model weights

FastMACE never ships or redistributes model weights. `fm.load` downloads a
checkpoint from its official source (or uses a local copy), checks its SHA-256
and shows its licence once per process as a `ModelLicenceWarning`
(`fastmace/model/labels.py`, `fastmace/model/loading.py`). Several MACE
foundation models, MACE-MH-1 among them, are under the Academic Software
License (non-commercial use only); the label table records each one. The
weights' licence binds you independently of FastMACE's MIT licence.

## Assessment: may the release ship `dftd3_params.npz`?

*Conductor's assessment for the owner (task 6-T02); not legal advice, and the
owner decides.*

**What the file is.** `dftd3_params.npz` is byte-identical to
`torch_dftd/nn/params/dftd3_params.npz` of torch-dftd 0.5.1 (SHA-256 in
`NOTICE.md`; `tests/cpu/test_d3_kernel.py` checks it). It holds numbers only:
the D3 C6 reference values with their reference coordination numbers, the
cutoff radii R0, covalent radii and r2/r4 by atomic number.

**torch-dftd's licence.** torch-dftd is MIT (© 2021 Preferred Networks), and
the file is part of its released package. MIT allows copying and
redistribution, also under another licence, provided the copyright notice and
permission notice are kept. FastMACE keeps both: `fastmace/longrange/data/LICENSE`
(torch-dftd's text) and `NOTICE.md` files, all included in the wheel through
`pyproject.toml`'s package data. **As far as torch-dftd's licence goes,
shipping it is permitted.**

**Provenance of the numbers.** The values are S. Grimme's DFT-D3 reference
data (Grimme, Antony, Ehrlich, Krieg, J. Chem. Phys. 132, 154104 (2010); BJ
damping: J. Comput. Chem. 32, 1456 (2011)). They originate in the original program's coefficient file (`pars.f`,
distributed with DFT-D3 by the Grimme group, Universität Bonn); exactly how
torch-dftd derived its file was not traced (**uncertain**). Public sources show:

- The original DFT-D3 program is distributed under the GNU GPL, version 1 or
  later; the repackaged dftd3-lib states exactly that, and conda's packaging
  of the original lists it as GPL-1.0+.
- simple-dftd3 (the Grimme group's current implementation, which carries the
  same reference data) is LGPL-3.0-or-later.
- We found no separate licence placed on the reference numbers as a dataset.

**Open questions for the owner.**

1. Whether a table of physical reference values (computed C6 coefficients)
   is copyrightable at all is jurisdiction-dependent. In the EU a database
   right may also apply. If it is protected, the upstream licence of the data
   is the GPL (original program) or LGPL (simple-dftd3), and torch-dftd's MIT
   relicensing of the converted file would only be valid if those terms
   allowed it. They would not, for a copyleft licence. That is torch-dftd's
   question, not one FastMACE can settle.
2. Low-risk options, for the owner to choose: (a) ship as now, with torch-dftd's
   MIT notice plus an attribution to Grimme et al. (done in the data
   `NOTICE.md` and here), relying on torch-dftd's licence, as many MIT/BSD
   D3 implementations (torch-dftd itself, and others built on the same
   tables) do; (b) not ship the file and fetch it at first use from the
   installed `torch-dftd` (making `[d3]` a requirement of the native D3 path,
   against SPEC §4.1); (c) ask the Grimme group (or Preferred Networks) for a
   statement about the data's licence before 1.0.
3. Ported ASE code: the FastMACE integrators and optimisers port ASE 3.27's
   algorithms to the GPU for agreement to rounding. If any port is a close
   enough translation to be a derived work of LGPL-2.1-or-later code, that
   file would be LGPL, not MIT. **Decided (owner D8.2, 25 October 2026; task
   6-T02b):** the ports are marked LGPL-2.1-or-later. The 13 files are:
   `fastmace/integrators/{langevin,baoab,berendsen,csvr,nose_hoover,mtk,nve}.py`,
   `fastmace/optimize/{fire,lbfgs,cell}.py`, `fastmace/reference/integrators.py`
   and `fastmace/kernels/triton/{fused_integrator,nose_hoover_chain}.py`.
   Each carries `# SPDX-License-Identifier: LGPL-2.1-or-later` and a line
   naming the ASE source it ports; NOTICE.md lists them, the licence text is
   `LICENSES/LGPL-2.1-or-later.txt`, and `pyproject.toml` declares
   `license = "MIT AND LGPL-2.1-or-later"`. `tests/cpu/test_lgpl_headers.py`
   keeps the headers, NOTICE.md and the licence expression in step. Found by
   searching `fastmace/` for ASE provenance in docstrings ("ASE 3.27's ...",
   "port"); the `fastmace/ase/` shims mirror ASE's signatures and stay MIT.

No change to what ships was made by this task.
