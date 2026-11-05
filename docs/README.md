# FastMACE documents

The map of every document in the repository. The user site (MkDocs, `mkdocs.yml`) is built from the pages in the
first table; everything else is for developers and is read on GitHub.

## User site

| Page | What it covers |
| --- | --- |
| [index.md](index.md), [install.md](install.md), [quickstart.md](quickstart.md), [faq.md](faq.md) | Start here |
| [guide/](guide/) | Running MD, ASE scripts, MC, NPT with D3, relaxations, restarts, models, the planner |
| [reference/](reference/index.md) | Every option, CLI flag and TOML key with its default (checked by `tests/cpu/test_docs_options.py`) |
| [api/](api/index.md) | API reference generated from the code |
| [benchmarks.md](benchmarks.md) | Published step-time table (25 October 2026, Colab L4; newer numbers: [current/defaults-by-gpu.md](current/defaults-by-gpu.md) §4) |
| [development.md](development.md) | Environments, test tiers, CI, GPU runners, releasing, NAS, Modal, Colab |
| [release/oeq-wheels.md](release/oeq-wheels.md), [licences.md](licences.md) | Prebuilt OEq extension; third-party licences |

## What runs today

| Question | Document |
| --- | --- |
| Which switches are on, on which GPU, and why (decision, measured b/a, report)? | [current/defaults-by-gpu.md](current/defaults-by-gpu.md) |
| Every config field, its default and its `auto` rule | [current/switch-map.md](current/switch-map.md) (read by `tests/cpu/test_config.py`); the code (`fastmace/planner/fixed.py`) is final |
| How OMAT-0 runs on the A100, part by part, and where each idea came from | [current/omat0-a100.md](current/omat0-a100.md) |
| The contract: architecture, rules, decisions (§18), revisions (§19) | [../SPEC.md](../SPEC.md) (read only the sections you need: `grep -n '^##' SPEC.md`) |
| Who does what, environments, how to record results | [../AGENTS.md](../AGENTS.md), [../plans/README.md](../plans/README.md), [../plans/CONDUCTOR.md](../plans/CONDUCTOR.md) |
| How an agent runs jobs on the NHR@FAU Alex cluster (SSH, storage, Slurm) | [nhr-cluster.md](nhr-cluster.md) |

## History and evidence

| Question | Document |
| --- | --- |
| Every optimisation ever tried (adopted, rejected, cut, open), by area, with chronology and screen table | [optimisation-ledger.md](optimisation-ledger.md) |
| Phase 0–6 task reports, acceptance reports, runbooks | [validation/phases/](validation/phases/README.md) |
| MH-1 step-time rounds 1–10, float64 package, precision screens | [validation/speed/](validation/speed/README.md); designs in [design/speed/](design/speed/README.md) |
| Memory capacity and the rescue ladder (rule 12) | [validation/capacity/](validation/capacity/README.md) |
| The tiled path (up to 149M atoms) | [validation/tiled/](validation/tiled/README.md) |
| OMAT-0 and the generic family (A100 kernel rounds) | [validation/omat0/](validation/omat0/README.md) |
| PolarMACE (POLAR-1): MH-1 and OMAT-0 reuse, POLAR-1 defaults | [validation/polar/](validation/polar/README.md) |
| Comparisons with TorchSim, ALCHEMI, other machines | [validation/comparisons/](validation/comparisons/README.md) |
| Phase and round plans, idea lists, future work | [../plans/README.md](../plans/README.md) |
| Raw run records (JSON, logs, scripts) | `../results/<run-id>/` |
| Patches proposed upstream (MACE, OEq) | [upstream/](upstream/README.md) |
| Research notes and prompts for outside agents | [design/research/](design/research/README.md) |
| Conversation transcripts (provenance; large zips) | [design/conversations/](design/conversations/README.md) |

## Design notes (architecture as built)

| Note | Topic |
| --- | --- |
| [interfaces](design/interfaces.md), [isolation](design/isolation.md), [physics](design/physics.md), [metadata-schema](design/metadata-schema.md) | Core interfaces, process isolation, physics conventions, output metadata |
| [contracts-1b](design/contracts-1b.md), [contracts-2](design/contracts-2.md), [contracts-3](design/contracts-3.md), [plan-1b](design/plan-1b.md) | Phase contracts and the phase-1b plan text (read by `tests/cpu/test_report.py`) |
| [planner-5](design/planner-5.md), [strain-per-model](design/strain-per-model.md) | Planner (phase 5), strain handling per model |
| [d3](design/d3.md), [fixcm-projection](design/fixcm-projection.md), [oeq-runtime](design/oeq-runtime.md) | D3 dispersion, fixcm projection, the OEq runtime (named by `fastmace doctor`) |
| [tiled-backend](design/tiled-backend.md), [tiled-wavefront](design/tiled-wavefront.md) | Tiled capacity path and its wavefront mode |
| [2026-10-positioning](design/2026-10-positioning.md) | Positioning against other MD codes |

## Where new documents go

Reports: `docs/validation/<topic>/`, with a row in that folder's `README.md`. A changed default: also
[current/defaults-by-gpu.md](current/defaults-by-gpu.md) and the [ledger](optimisation-ledger.md). Raw records:
`results/<run-id>/`. Design notes: `docs/design/` (architecture), `docs/design/research/` (dated research notes).
