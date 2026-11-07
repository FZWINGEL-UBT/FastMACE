# Packed Modal jobs

These files are the job tooling behind AGENTS.md "Using resources". The design is in
[resource framework §3.4–3.5](../../docs/process/resource-framework.md).

| File | What it does |
| --- | --- |
| `ab_pack.sh` | Container template: one baseline (the GPU's default plan) and N switches, each A/B in both build orders |
| `gate_pack.sh` | Container template: the default gate once, then one `tools/force_stress_gate.py` gate per switch |
| `profile.sh` | Container template: one `tools/step_breakdown.py` profile, default plan or one switch |
| `launch.py` | Manifest-driven launcher: packs rows, sets timeouts, launches under the WSL limits |
| `slim_tar.py` | Builds the code-only source tarball (about 13 MB) that every job uses |
| `p90.json` | p90 wall times per template and GPU; the launcher sets each timeout to twice the p90 |

Templates are versioned (`v1` in their second line). To change one, add a new version. A running batch keeps its
own copy, because the launcher copies the templates into the batch folder before it launches anything.

## A batch

Write the manifest. `timeout` is in minutes; leave it empty to use twice the p90. The optional `kind` column takes
`ab` (default), `gate` or `profile`, and the optional `name` column sets the switch's short name.

```csv
idea,switch,gpu,profile,timeout,kind,name
i41,fastmace.kernels.triton.tabled_outer_sum:CHUNK=8,H200,fourth,,ab,tosc8
i42,fastmace.kernels.triton.tabled_outer_sum:CHUNK=32,H200,fourth,,ab,tosc32
```

Check the manifest and the packs, then launch the batch from WSL:

```bash
python3 tools/jobs/launch.py plan batch7.csv --batch b7
python3 tools/jobs/launch.py start batch7.csv --batch b7 --commit <sha>
python3 ~/fm-runs/b7/tools/jobs/launch.py status ~/fm-runs/b7
```

The launcher enforces these rules:

- **Packs.** Rows with the same kind, GPU and profile share one container, with at most 8 switches. A pack with a
  single switch is refused unless you pass `--allow-single` because that switch blocks a decision.
- **Slim source.** The slim tarball of `--commit` is built inside the batch folder. The commit must contain
  `tools/jobs`.
- **WSL limits.**
  - At most 6 Modal clients run at once, counting every `fastmace_modal.py` client on the machine.
  - Launches are at least 25 s apart.
  - The batch runs as a `sudo systemd-run` system unit named `fm-batch-<id>`.
  - A hidden Windows `wsl.exe sleep` keeps the VM up.
- **Retries.** A failed job (non-zero exit or a wrong GPU) is retried once. A second failure is left for a person.
  Budget refusals and Modal's rate limit wait 90 s and relaunch without using up the retry.
- **Ledger.** Each client sets `MODAL_PROFILE`, `FASTMACE_BATCH` and `FASTMACE_IDEA`. The budget guard writes them
  to `~/.fastmace-modal/modal-ledger.csv`.
- **Stale reservations.** Before the first launch, the batch drops in-flight budget entries whose client has ended.
  Every entry also expires after its function timeout, which is the command timeout plus 10 min.

`--env KEY=VALUE` passes a template variable to every job, for example `CELL=16x16x16` or `BLOCKS=4`. The allowed
variables are `MODEL`, `CHECKPOINT`, `STRUCTURE`, `STRUCTURES`, `CELL`, `BLOCKS`, `ALIAS`, `BACKEND` and `STEPS`.
