# Phase 6: Release

Status: **draft**, 25 October 2026, written by the conductor after the phase 5 acceptance (tag `fastmace-5`,
`d304834c`). D1–D7 below are open; tasks that need none of them may start.
Spec: [SPEC.md](../SPEC.md) §16 row "6. Release", and the sections it cites:
- §14: name, licence, extras, prebuilt OEq host-extension wheels, caches;
- §15: documentation;
- §4.1: support policy (Python 3.10–3.12, the two newest torch minors);
- §6.3: upstreaming (MACE: linear rewrites, reassociation; OEq: the no-`nvcc` host build);
- §18: the PyPI name check, the re-check of competitors before 1.0.

Acceptance (§16): **1.0 tagged; docs build; `pip install fastmace[oeq]` on a clean machine without a compiler.**
Estimate: 2–3 weeks (§16).

## Inventory (25 October 2026)

- **Name:** `fastmace` is free on PyPI (`/pypi/fastmace/json` 404). On GitHub the only repository of that name is
  ours (`FZWINGEL-UBT/FastMACE`). **The API reports it as public** (ResearchModule's AGENTS.md says private). D1.
- **Version:** `fastmace.__version__ = "0.0.1.dev0"`, dynamic in `pyproject.toml`. MIT, `LICENSE` present.
- **Dependencies:** `torch` unpinned, `mace-torch==0.3.16`, `cuequivariance-torch>=0.12`, `ase>=3.23`; extras
  `oeq`, `polar`, `d3`, `d3-reference`, `full`, `dev`. §4.1 asks for pinned, CI-tested builds of the two newest
  torch minors at release time; only 2.14.1+cu130 is qualified.
- **OEq:** `fastmace/kernels/_oeq_runtime.py` builds the host extension in-process (g++, ninja, runtime header
  wheels, `nvcc` sentinel). No prebuilt wheel exists.
- **Vendored files:** OEq streaming kernels (BSD-3, `LICENSE` and `NOTICE.md` present); `dftd3_params.npz`
  (torch-dftd 0.5.1, MIT, `LICENSE` and `NOTICE.md` present). §14 leaves "may the release ship it" to this phase.
- **Docs:** `README.md` (27 lines), `docs/index.md`, `quickstart.md`, `rewrites.md`, `development.md`,
  `docs/design/` (27 internal notes), `docs/validation/` (reports). No docs builder, no API reference, no examples
  directory, no benchmark tables outside the validation reports.
- **CI:** `cpu.yml` (hosted, off since 20 October 2026), `gpu.yml` and `gpu-tier.yml` (run by hand).
- **Carried over from phase 5 (open findings):** slow tier out of memory in one process on 16 GiB; integrator label
  in plan records; extended-energy `run_fresh`; slow POLAR cold compile; OMOL-0 XL on the A100; NAS sync.

## Owner decisions

- **D1 Repository visibility.** (a) keep it public, as GitHub reports now; (b) make it private until 1.0 is
  tagged. The ResearchModule's AGENTS.md is corrected either way.
- **D2 PyPI.** (a) reserve the name now with `0.1.0` from `fastmace-5`, release 1.0 later; (b) TestPyPI only
  until 1.0; (c) no PyPI in phase 6 (GitHub releases only). Publishing is irreversible (a version can never be
  re-uploaded).
- **D3 Supported torch builds.** (a) 2.14.1+cu130 only, documented; (b) add the next torch minor if it exists at
  release time, with its own qualification (Colab L4 + local); (c) §4.1 as written (two newest minors).
- **D4 Prebuilt OEq wheels.** (a) one wheel per supported torch/CUDA build and Python 3.10–3.12, Linux x86_64 only,
  built in manylinux containers (on Modal or locally), published on PyPI as a separate package
  (`fastmace-oeq-ext`); (b) attached to GitHub releases and found with `--find-links`; (c) drop the promise and
  ship the in-process toolchain build only (changes §16's acceptance).
- **D5 Upstream PRs.** (a) open them (MACE: block/grouped linear and first-interaction reassociation; OEq: the
  no-`nvcc` host build) from the owner's GitHub account, after the conductor drafts them; (b) drafts only, owner
  opens them; (c) skip in 1.0.
- **D6 Docs site.** (a) MkDocs Material on GitHub Pages; (b) Sphinx with autodoc; (c) Markdown in the repository
  only (the acceptance's "docs build" is then a link check).
- **D7 Hosted CPU CI.** (a) turn `cpu.yml` back on for the release branch and tags; (b) keep it off and run the
  suite locally before each tag.
- **D8 Licence questions from 6-T02** (`docs/licences.md`). (1) `dftd3_params.npz`: the numbers come from Grimme's
  DFT-D3 (GPL-1+; simple-dftd3 LGPL-3+); no separate data licence was found, so torch-dftd's MIT relabelling may
  not hold. (a) ship as now with the notice; (b) do not ship it, fetch it from an installed torch-dftd at first
  use; (c) ask the Grimme group or PFN first. (2) The ASE 3.27 ports in `integrators/` and `optimize/` may be
  LGPL derived works: (a) owner reviews; (b) mark those files LGPL-2.1+ and say so in NOTICE.md.

## Tasks

| Task | Size | GPU | After | Files (main ones) |
| --- | --- | --- | --- | --- |
| 6-T01 Packaging audit | S | no | — | `pyproject.toml`, `MANIFEST.in` (if needed), `tests/cpu/test_packaging.py` (new) |
| 6-T02 Licences and notices | S | no | — | `NOTICE.md` (new, top level), `docs/licences.md` (new), `fastmace/longrange/data/NOTICE.md` |
| 6-T03 Clean-install check (no compiler) | M | smoke | T01; D4 | `tools/release/clean_install.sh` (new), `docs/validation/6-T03.md` |
| 6-T04 Prebuilt OEq host-extension wheels | L | yes (smoke) | D3, D4 | `tools/release/oeq_wheels/` (new), `fastmace/kernels/_oeq_runtime.py` (prefer the wheel), `fastmace/cli/doctor.py` |
| 6-T05 User documentation | M–L | no | D6 | `docs/` (user guide, port-your-ASE-script, models, planner, restart, MC, NPT + D3, FAQ), `README.md` |
| 6-T06 API reference and docs build | M | no | T05; D6 | docs config, `docs/api/`, link check |
| 6-T07 Examples | M | smoke | T05 | `examples/` (new): NVT anneal, NPT + D3 melt–quench, swap MC, relaxations, `fm.calc` port |
| 6-T08 Benchmark tables | M | yes (timed) | — | `docs/benchmarks.md`, `tools/bench/release_table.py` (new); RTX 5060 Ti timed, L4 untimed-labelled |
| 6-T09 Competitor re-check (§18) | S | no | — | `docs/design/2026-10-positioning.md` (TorchSim 0.6.2 result of 25 October included) |
| 6-T10 Upstream PR drafts | M | no | D5 | `docs/upstream/` (new): MACE linear rewrites, reassociation; OEq no-`nvcc` build |
| 6-T11 Phase 5 follow-ups | M | yes | — | slow-tier POLAR references in a child; integrator label in plan records; extended-energy `run_fresh` |
| 6-T12 Release engineering | S–M | no | T01; D2, D7 | `.github/workflows/release.yml` (new), `CHANGELOG.md` (new), `docs/development.md` (release steps) |
| 6-T13 Acceptance and 1.0 | M | yes | all | `docs/validation/phase-6-runbook.md`, `docs/validation/phase-6.md`; tag `v1.0.0` |

**Ready now (no decision needed):** T01, T02, T08, T09, T11. T05 and T07 may begin on content; only the docs
builder waits for D6.

## Rules carried over (phase 5)

Tests and merges, GPU reservations (`tools/ci/local_gpu.py run`), timing (alternating blocks, RTX 5060 Ti only),
Colab through `tools/colab/` with each VM stopped, workstations off until the owner says they are back, workers
push their own `fastmace/6-Tnn` branches. **Nothing is published (PyPI, GitHub release, upstream PR, docs site)
without the owner's decision for that item.**

## Acceptance runbook outline (6-T13)

1. CPU suite green at the release candidate; lint clean.
2. `python -m build` gives an sdist and a wheel; `twine check` passes; the wheel contains the vendored kernels, D3
   tables, protocol data and every licence and notice.
3. Clean machine (fresh Colab L4 VM, and a fresh WSL venv without `g++`/`ninja` on `PATH`):
   `pip install "fastmace[oeq]"` from the built artefacts, `fastmace doctor` reports the prebuilt extension, the
   NaCl MD smoke runs on the hybrid backend.
4. Docs build without warnings; links checked.
5. Examples run (smoke length) on the RTX 5060 Ti.
6. No regression against `fastmace-5`: plan corpus and bitwise trajectories as in phase 5 item 3–4.
7. Tag `v1.0.0` (owner).
