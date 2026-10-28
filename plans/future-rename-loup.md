# Future work: rename FastMACE to loup

Status: **planned, not started** (owner, 28 October 2026: "Go ahead and also document this as future work").
Start it only after the open branches have merged (`fastmace/docs-20261028`, `fastmace/fp64-d1-relax-fix`,
`fastmace/fp32-ideas-20261028`, `fastmace/suite-speed` and any started since). The rename touches nearly
every file, so every branch still open would conflict. Two steps need the owner's go-ahead at the time they
are taken: renaming the GitHub repository and publishing on PyPI.

## Scope (counted on main, 28 October 2026)

| Area | Files | Mentions of "fastmace" |
| --- | --- | --- |
| package `fastmace/` | 150 | 1,524 |
| `tests/` | 260 | 3,424 |
| `tools/` | 128 | 1,505 |
| `docs/` | 188 | 1,996 |
| `plans/` | 18 | 582 |
| `examples/` | 7 | 35 |
| `results/` | 2,896 | 33,261 |

There are also 29 environment variables `FASTMACE_*`, for example `FASTMACE_CACHE`, `FASTMACE_MODELS`,
`FASTMACE_SOFT_FILE_LOCKS` and `FASTMACE_OEQ_PREBUILT`.

`results/` stays as it is. It is history and correctly names the FastMACE that produced it. The validation
reports, the CHANGELOG entries up to the rename and the SPEC decision log keep the old name in their text too.

PyPI: `loup` was free on 28 October 2026. `fastmace` 1.0.0 and `fastmace-oeq-ext` are published there.

## What must keep working

1. **Existing installs and imports.** PyPI cannot rename a package, so:
   - publish `loup` and `loup-oeq-ext`;
   - publish a last `fastmace` release that only depends on `loup` and makes `import fastmace` (and its
     submodules) resolve to `loup`, with a `DeprecationWarning`.
2. **Saved runs.** Restart files, `simulation.json`, `metadata.json`, protocol-run records and any pickled or
   recorded module paths name `fastmace`. Old runs must still `--resume` and `--quench`:
   - the loaders accept the old names;
   - tests resume a restart written by 1.0.0 (keep one as a fixture).
   This is the main risk of the rename.
3. **The CLI.** The command `fastmace` becomes `loup`, with `fastmace` kept as an alias that warns, for at
   least one minor version.
4. **Environment variables.** Read `LOUP_*` first, then `FASTMACE_*` with a deprecation warning, for the same
   period. `fastmace doctor` (`loup doctor`) reports which spelling was used.
5. **Default output folder.** `<structure stem>.fastmace/` becomes `<stem>.loup/`. A resume still finds an old
   folder given by path.
6. **Caches.** The compile, template and OEq caches may simply rebuild under the new name. Say so in the
   CHANGELOG, because the first runs after upgrading take longer.

## ResearchModule

- `main.py --engine fastmace`, `langevin/fastmace_engine.py`, `--fastmace-api` and the Modal and run scripts
  move to the new name.
- `fastmace` stays an accepted engine alias, because every `run.json` so far records it and `--resume` and
  `--quench` read it.
- `AGENTS.md`, the memory notes, the environment recipes and the PYTHONPATH instructions are updated in the
  same change.
- The research engine baseline tag and the past results keep their names.

## Steps when started

1. Merge or close every open branch; announce a freeze to the running sessions.
2. Rename the package directory and every import, the distribution name, the console scripts and the
   environment variables, with the compatibility layers above.
3. Rename `fastmace-oeq-ext` to `loup-oeq-ext` with the same build recipe (6-T04).
4. Tests:
   - the full CPU suite;
   - resuming a 1.0.0 restart;
   - `import fastmace` through the shim;
   - the old CLI name and old environment variables;
   - one GPU check on the RTX 5060 Ti and one on an Ada workstation.
5. Docs site, README, SPEC title and the docs build under the new name.
6. **Owner go-ahead:** rename the GitHub repository (GitHub redirects the old URLs) and move the local and NAS
   checkouts (`D:\Margraf\FastMACE`, `/mnt/homes/zwingel/FastMACE`).
7. **Owner go-ahead:** publish `loup`, `loup-oeq-ext` and the `fastmace` shim release on PyPI.
8. ResearchModule follow-up as listed above.

## Effort

About a day for the rename, the compatibility layers, the shim and the tests, plus a few hours for
ResearchModule and the docs site.
