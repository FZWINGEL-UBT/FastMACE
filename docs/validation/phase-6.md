# Phase 6 acceptance (1.0)

Plan task 6-T13, 25 October 2026. Runbook (items and pass rules fixed before the runs):
[phase-6-runbook.md](phase-6-runbook.md). SPEC §16 row 6: **1.0 tagged; docs build; `pip install fastmace[oeq]` on
a clean machine without a compiler.** Records: `results/6-T13-20261025/`.

**Verdict: pass, with one finding for the owner (F1).** Every judged item passes. The clean install works without
a compiler on the local PC (Python 3.10, 3.11 and 3.12, wheel and sdist) and on a fresh Colab L4 VM (Python 3.12
and 3.10). Colab's own system Python is 3.13, though, and `fastmace-oeq-ext` (D4: CPython 3.10 to 3.12) has no
wheel for it. On that Python, `pip install "fastmace[oeq]"` without a compiler quietly runs plain cuEquivariance
instead of the hybrid (F1). The tag `v1.0.0` is the owner's step. It is not blocked, provided that 1.0 states
Python 3.10 to 3.12 for the no-compiler hybrid.

## Freeze

- **Freeze commit:** `63a6c0c9529d387fb65a06b6d4d7fb352fa30572` (`fastmace/6-T13`). Version `1.0.0`, CHANGELOG
  entry `[1.0.0] - 2026-10-25`, classifier "Production/Stable". The `fastmace/` tree is
  `1dc1074b`. It is the same in the built artefacts' commit `cc24c3bf`, which differs from the freeze only in
  tools, tests and the runbook.
- **Baseline:** tag `fastmace-5`, commit `d304834c` (the tag object is `6722f29`; the runbook's first draft named
  the tag object, and it was corrected before item 6 ran).
- **Changes made during the acceptance** (all on `fastmace/6-T13`):
  1. `NOTICE.md` (the LGPL-2.1 notice of D8.2) was in neither the wheel nor the sdist: item 2's first build failed.
     Now in `license-files`; `tests/cpu/test_packaging.py` checks every licence and notice. Item 2 rerun.
  2. `docs/release/oeq-wheels.md` linked two excluded pages (mkdocs INFO, broken on the site). They now point to
     GitHub. Item 4 rerun.
  3. The known flake (`test_a_stopped_job_keeps_its_results_and_junit`) was not load. It came from the suite's launch:
     under `setsid nohup ... &`, SIGHUP and SIGINT start out ignored, and bash cannot trap a signal that was ignored
     on entry. Raising the timeout alone (first fix) failed both cases in the first suite run. The test now starts
     the script with default signal dispositions; the script's header documents the limitation. Item 1 rerun.
  4. Tools: release `5` in `ab_fastmace.py`/`trajectory_versions.py` (test `test_acceptance_6.py`),
     `clean_install.sh` installs a given artefact and refuses a checkout on `PYTHONPATH`, and the runners
     `tools/acceptance/phase6_*.sh` and `tools/release/{acceptance_*,colab_clean_install,check_dist}.sh`.

## Verdicts

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | CPU suite and lint | **pass**: ruff clean; 4,646 passed, 553 skipped, 0 failed (33 min, `-n 8`) | `cpu-suite-2-tail.log`; the first run (2 failed, the flake above) is `cpu-suite-1-tail.log` |
| 2 | Build | **pass**: `fastmace-1.0.0-py3-none-any.whl` and `fastmace-1.0.0.tar.gz`, `twine check --strict` PASSED; the vendored kernel `factorized_projected.cuh` with LICENSE and NOTICE.md, `dftd3_params.npz` with LICENSE and NOTICE.md, the seven protocol data files; the dist-info has LICENSE, NOTICE.md and LICENSES/LGPL-2.1-or-later.txt; `License-Expression: MIT AND LGPL-2.1-or-later` | `contents.txt`, `sha256.txt`, `twine.log` |
| 3a | Clean install, local | **pass**: wheel on 3.10.12, 3.11.16 and 3.12.3, sdist on 3.10.12: `uv pip check` clean, doctor `host extension: prebuilt (fastmace-oeq-ext 0.7.0.post1 …); no compiler needed`, NaCl hybrid MD to step 20, finite, the installed `fastmace 1.0.0` (from `site-packages`), empty sentinel log, `CLEAN INSTALL OK` | `clean-install/` |
| 3b | Clean install, other machine | **pass on 3.12 and 3.10** (uv-managed Python on the VM): as 3a, sentinel logs empty. **Not on the VM's system Python 3.13** (F1): doctor says `fastmace-oeq-ext is not installed`, and the smoke planned `backend: cueq` | `colab-l4-py312/`, `colab-l4-py313/` |
| 4 | Docs build | **pass**: `mkdocs build --strict`, exit 0, no WARNING, no excluded-link INFO | `mkdocs.log` |
| 5 | Examples | **pass**: 01 (72 s), 02 (83 s), 03 (113 s), 04 (143 s), 05 (79 s), all exit 0 | `examples/` |
| 6a | Plan corpus | **pass**: 228 baseline / 228 freeze cases, 6 refused each, **no difference at all** | `corpus/compare.txt` |
| 6b | Bitwise trajectories | **pass**: CPU float64 NVT/NPT with and without D3(BJ), reproducible GPU (plain cuEq float64) NVT/NPT: all `equal: true`, `passed: true` against `fastmace-5` | `traj/` |
| 7 | Tag `v1.0.0` | owner | |

The runbook expected no difference in item 6, and none appeared. The 6-T11 integrator label touches only
non-Langevin/BAOAB schemes, which the corpus does not plan, and 6-T14 and 6-T04 do not change trajectories.

## Runs

| Run | Hardware | Environment | Source |
| --- | --- | --- | --- |
| CPU suite 1 | local PC, WSL2 Ubuntu 24.04 | `fastmace_cpu` (torch 2.14.1+cpu) | archive of `a675a147` |
| CPU suite 2, lint | same | same | archive of `63a6c0c9` |
| Build, docs | same | `fastmace_cpu` (build 1.6.1, twine 7.0.0, mkdocs-material 9.7.7) | archive of `cc24c3bf` |
| Clean installs (3a) | RTX 5060 Ti (sm_120), driver 610.88 | fresh uv venvs `~/tmp/6-T13/clean-*`: torch 2.14.1+cu130, mace-torch 0.3.16, OEq 0.7.0, fastmace-oeq-ext 0.7.0.post1 (6-T04's local wheels) | `cc24c3bf` artefacts, script from the same archive |
| Colab 3b, run 1 | Colab L4 (sm_89), driver 580.82.07, `fm-l4-20261025-183925-e0e40e` | VM system Python 3.13.16 | `3e8b2b38`; 0.23 CU |
| Colab 3b, run 2 | Colab L4, same driver, `fm-l4-20261025-191815-75c17f` | uv-managed Python 3.12.3 and 3.10.22 | `16e402b0`; 1.03 CU (the cache pack step took 26 min) |
| Examples (5), trajectories (6b GPU) | RTX 5060 Ti, `tools/ci/local_gpu.py run` | `mace_cueq_oeq` (examples), `mace_cueq_latest` (reproducible) | archive of `63a6c0c9` |
| Plan corpus (6a), CPU trajectories | local CPU | `fastmace_cpu`, `mace_cueq_oeq` with `CUDA_VISIBLE_DEVICES=` | archives of `fastmace-5` and `cc24c3bf`/`63a6c0c9` |

Colab spend: 1.39 CU (31.19 → 29.80) of the 6 allowed. Both VMs stopped themselves. Nothing was uploaded to PyPI,
no tag was made, and the docs were not deployed.

## Open findings

**F1 follow-up (6-T15):** the owner selected option (b), adding a CPython 3.13 wheel. The wheel, extras'
markers and documentation have been updated and a fresh Python 3.13 installation passed the local hybrid
GPU smoke with no g++/Ninja/nvcc calls. See [6-T15.md](6-T15.md) for the artefact, evidence and qualification
boundary. The original Colab Python 3.13 failure below remains the acceptance-run history.

- **F1 (owner decision, before or with the tag): Python 3.13.** Colab's system Python, and the current default of
  several distributions, is 3.13. `fastmace` declares `requires-python >=3.10` and the cp313 classifier is absent,
  but its `[oeq]` marker skips `fastmace-oeq-ext` on 3.13. With no prebuilt extension and no compiler the planner
  chose plain cuEquivariance (the sentinel log stayed empty: no build was even tried). That is correct per rule 6,
  but slower, and visible only in doctor and the plan record. Options: (a) ship 1.0 as is and say "no-compiler hybrid: CPython 3.10 to 3.12" in the
  install docs (D4's scope); (b) add a cp313 `fastmace-oeq-ext` wheel (`PYTHONS` in
  `tools/release/oeq_wheels/build_in_container.sh`, `requires-python <3.14`, the marker in `pyproject.toml`) with
  its own clean-install check; (c) cap `fastmace` at `<3.13` for 1.0. Not changed here: it widens D4.
- **F2: Colab driver pack step.** After a fresh-venv job, `remote.sh pack` packed the new uv and compile caches
  for 26 minutes, about 0.6 CU of the 1.03, and one poll timed out. Excluding `uv-cache` (or `$FM/tmp`) from the
  cache delta would avoid it. This is ops-colab's code, not the library.
- **F3: OEq warning (cosmetic, known since 6-T04).** OpenEquivariance still prints "falling back to JIT
  compilation" with the prebuilt extension.
- **Carried over:** phase 5's open findings that 6-T11 did not close (slow tier in one process on 16 GiB, slow
  POLAR cold compile, OMOL-0 XL on the A100, NAS sync); workstation clean install (workstations off).

## Steps left for the owner

1. F1 option (b) was selected and implemented in 6-T15; the 3.13 wheel is merged. The complete CPU suite
   passed after 6-T15: 4,655 passed, 553 skipped, now 13 min 51 s with 6-T16's scheduling.
   A fresh Colab L4 Python 3.13 clean install also passed with the prebuilt hybrid and no C++ build calls.
2. `fastmace/6-T13`, 6-T15 and 6-T16 are merged. The owner has authorized tagging `v1.0.0`
   (`docs/development.md`, "Releasing"), publication and docs deployment.
3. `tools/release/make_release.sh v1.0.0`, upload `fastmace-oeq-ext` 0.7.0.post1 (6-T04's wheels) before
   `fastmace` 1.0.0, since `[oeq]` depends on it, then upload `fastmace`. Optionally deploy the docs.
