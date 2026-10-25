# FastMACE 1.0.0 release validation

The owner authorized merging the Python 3.13 wheel, final validation, the release tag, PyPI uploads and
documentation deployment. The additional CPU-suite speed investigation was completed before publication.

## Final checks

- **Library:** the `fastmace/` tree remains `1dc1074bf7bad1d0eee6c0f24ef28337b86aee2e`, the accepted
  phase-6 library. Subsequent changes are Python-version packaging metadata, tests, release tools and docs.
- **CPU suite:** 4,655 passed, 553 skipped, zero failures/errors, 831.34 s. Eight workers, OMP2,
  `--dist=load --maxschedchunk=1`. All 5,199 prior cases are retained, plus nine Python 3.13 metadata cases.
  See [6-T16.md](6-T16.md), including the 2.4× scheduling improvement and comparison limits.
- **Lint:** `ruff check fastmace tests` passed.
- **Local Python 3.13:** installed wheel, prebuilt host extension, hybrid MH-1 NaCl MD, 20 steps, finite output;
  no C++ compiler/Ninja/nvcc calls. See [6-T15.md](6-T15.md).
- **Fresh Colab L4 Python 3.13:** the installed FastMACE 1.0.0 wheel selected the prebuilt extension and hybrid
  backend on CPython 3.13.16, reached step 20 with finite output, and left the blocked-compiler log empty.
  GPU: NVIDIA L4, sm_89, driver 580.82.07. Source `68a159f3`, wheels from 6-T15. Exit 0; VM stopped.
- **Build:** the merged-source wheel and sdist built; strict twine checks passed. The final publication
  artefacts are rebuilt from the annotated tag by `tools/release/make_release.sh`.
- **Documentation:** `mkdocs build --strict` passed with the GitHub Pages site URL configured.
- **Earlier acceptance:** the five GPU examples, 228-case unchanged plan corpus and bitwise trajectory
  checks remain the evidence in [phase-6.md](phase-6.md); no runtime library change requires repeating them.

The Colab run took 316 s. Its immediate account-balance readings were both 29.80 CU; a later account query
reported 29.68 CU with no active assignments. The raw ledger is retained; delayed billing is not reported
as a free run. Compile caches for this one-off clean install were kept under its temporary root, outside
the driver's persistent cache-pack directory.

Records: `results/release-1.0.0-20261025/`, plus the linked 6-T15 and 6-T16 records. The earlier default-scheduler
CPU run was intentionally interrupted for the speed investigation and is not counted as passed evidence.

## Qualification limits

Linux x86_64, CPython 3.10–3.13, torch 2.14.1+cu130 and OpenEquivariance 0.7.0. The new Python 3.13 check is
an installation/ABI and short-MD qualification, not a full scientific validation across every model on 3.13.
The prebuilt extension removes g++, Ninja and nvcc requirements for its host build; Triton's C launcher
still needs a C compiler, and runtime GPU-kernel compilation still occurs. Workstation/NAS qualification
remains deferred until the owner says those machines are available.

## Publication

**Complete, 25 October 2026.** Annotated tag `v1.0.0` points to
`f45985c5a5aba75747421fc58058ee8c3dd07364`. The four extension wheels were uploaded before FastMACE.
Anonymous PyPI metadata confirms every published file's SHA-256 matches the checked local artefact.

- [FastMACE 1.0.0](https://pypi.org/project/fastmace/1.0.0/): wheel and sdist.
  Wheel SHA-256 `919a84cfddefd2df458beb8789fc5b17b7b35d336d3a0e36ce2771c8f9d1d9c9`;
  sdist SHA-256 `c1125ae1d2ddd66fae00049342950aac4901548ae47075a70b055c2366f12bbc`.
- [fastmace-oeq-ext 0.7.0.post1](https://pypi.org/project/fastmace-oeq-ext/0.7.0.post1/):
  CPython 3.10, 3.11, 3.12 and 3.13 manylinux_2_28_x86_64 wheels. Hashes: `pypi-publication.json`.
- [GitHub release](https://github.com/FZWINGEL-UBT/FastMACE/releases/tag/v1.0.0): published stable,
  six uploaded artefacts, not a draft or prerelease.
- [Documentation](https://fzwingel-ubt.github.io/FastMACE/): GitHub Pages reports `built`; the root,
  installation guide, API reference and generated stylesheet each returned HTTP 200.

A second fresh Python 3.13.15 environment installed `fastmace[oeq]==1.0.0` from the public PyPI index,
without `--find-links` or a local FastMACE source. Neither distribution had direct-URL metadata.
`uv pip check` passed; the GPU smoke selected hybrid, loaded the prebuilt extension, reached step 20
with finite output, and left the blocked-compiler log empty. See `pypi-installation.json`, `pypi-gpu.log`
and `pypi-install.log`. The publishing credential was entered at Twine's hidden prompt, never placed in a
repository file or shell command, and cleared from session storage after both uploads.
