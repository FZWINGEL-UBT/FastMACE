# Runbook: phase 6 acceptance (1.0)

Plan task 6-T13, 25 October 2026. Outline: `plans/phase-6.md` ("Acceptance runbook outline (6-T13)"); SPEC §16
row 6: **1.0 tagged; docs build; `pip install fastmace[oeq]` on a clean machine without a compiler.** Items and
pass rules were fixed before any run. The tag itself is the owner's step (item 7); this task prepares it.

## Freeze and baseline

- **Freeze:** the release-candidate commit of `fastmace/6-T13` named in `phase-6.md` (version `1.0.0`). Fixes
  needed during the acceptance are listed there with the items they rerun.
- **Baseline:** the tag `fastmace-5` (`6722f29`), exported with `python -m fastmace.bench.versions export`.
- **Hardware:** local RTX 5060 Ti (sm_120) in WSL2 Ubuntu 24.04; one fresh Colab L4 VM (sm_89) as the other
  machine. Workstations are off. Colab budget: at most 6 compute units.
- **Environments:** `~/environments/fastmace_cpu` (CPU suite, build, docs), `~/environments/mace_cueq_oeq`
  (hybrid GPU), `~/environments/mace_cueq_latest` (CPU float64 trajectories); fresh uv venvs under `~/tmp/6-T13`
  for the clean installs.

## Items and pass rules

| # | Item | How | Pass |
| --- | --- | --- | --- |
| 1 | CPU suite and lint | `ruff check fastmace tests`; `pytest -n 8 tests/cpu tests/gpu` in `fastmace_cpu` with `CUDA_VISIBLE_DEVICES=''`, one suite at a time | ruff clean; no failure, no error |
| 2 | Build | `python -m build --sdist --wheel` from a `git archive` of the freeze; `twine check --strict`; list the wheel | both named `1.0.0`; twine passes; the wheel holds `fastmace/kernels/oeq_streaming/*.cuh` with its `LICENSE` and `NOTICE.md`, `fastmace/longrange/data/dftd3_params.npz` with `LICENSE` and `NOTICE.md`, the `fastmace/protocols/data/*` files, and the dist-info licences `LICENSE` and `LICENSES/LGPL-2.1-or-later.txt`; `License-Expression: MIT AND LGPL-2.1-or-later`; the sdist holds the same data files and `NOTICE.md` |
| 3a | Clean install, local | `tools/release/clean_install.sh` with `CLEAN_INSTALL_FASTMACE=<item 2 wheel>`, the `fastmace-oeq-ext` wheels of 6-T04 (rebuilt only if missing), Python 3.10, 3.11, 3.12; check stage through `tools/ci/local_gpu.py run`. Plus the same once with the sdist on 3.10 | for every run: `uv pip check` clean; `doctor` reports `host extension: prebuilt`; NaCl hybrid MD smoke reaches step 20 finite with the installed (not checkout) `fastmace 1.0.0`; the sentinel log of g++/c++/ninja/nvcc is empty; `CLEAN INSTALL OK` |
| 3b | Clean install, other machine | a fresh Colab L4 VM through `tools/colab/fastmace_colab.py --gpu L4 --timeout-min 40`; the command runs `clean_install.sh` on the VM's own Python (`python3`) with the item 2 wheel and the matching `fastmace-oeq-ext` wheel shipped in the payload | as 3a, on that Python; the VM stops itself; spend recorded |
| 4 | Docs build | `mkdocs build --strict` in `fastmace_cpu` into `~/tmp` | exit 0 and no `WARNING` line in the log |
| 5 | Examples | the five smoke commands of `examples/README.md` in `mace_cueq_oeq` on the RTX 5060 Ti, one reservation | each exits 0 |
| 6a | Plan corpus | `tools/plan_corpus.py run` on the `fastmace-5` archive and on the freeze archive, then `compare` | no difference other than the 6-T11 integrator label (`integrator` decision of non-Langevin/BAOAB schemes); the corpus plans only Langevin/BAOAB, so expected: no difference at all |
| 6b | Bitwise trajectories | `tools/trajectory_versions.py run --against 5 --side 5\|6 --positions`: CPU float64 NVT and NPT (with and without `--dispersion d3bj`), reproducible GPU NVT and NPT; `compare --bitwise` | every comparison `equal`/`passed: true` |
| 7 | Tag `v1.0.0` | owner | not judged here |

The 6-T14 change (explicit checkpoint in the gate's reference child) and the 6-T04 change (prebuilt OEq loader)
do not change trajectories; any other difference in item 6 fails it.

Not judged: timings (no fastmace/ change that affects step time since `fastmace-5`; `docs/benchmarks.md` holds
6-T08's tables), PyPI upload, docs deployment (owner's steps).

## Order

1. Item 1 (CPU, alone), then items 2 and 4 (CPU).
2. Item 3a install stages (no GPU), then GPU reservations: 3a checks, item 5, item 6b GPU sides.
3. Item 6a and 6b CPU sides between suites (never alongside item 1).
4. Item 3b on Colab in parallel with the local GPU work.
