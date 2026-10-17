# Conductor hand-over

Read this first if you are the conductor session for FastMACE. It was
written on 17 October 2026 by the conductor that ran phases 0 and 1a waves
0–1 from a ResearchModule session. From here on, the conductor runs in a
session opened in this repository (`D:\Margraf\FastMACE`).

## Your job

You own `plans/`, the phase status and `main`. You start one worker per
plan task, review its branch, run the combined trial, merge, record your
decisions in the phase plan, push and sync the NAS checkout. You write
little code yourself: small integration fixes, `pyproject.toml`, plans,
SPEC amendments.

**The owner has delegated routine decisions** (17 October 2026): merge
reviewed, green branches and start the next tasks without asking. Stop and
ask only for open SPEC §18 decisions, changes to scientific protocol,
anything destructive outside your own branches and worktrees, and the
phase 1a checkpoint 3 (after T19: `main.py` default and shims).

## State on 17 October 2026

- **Phase 0:** done (`docs/validation/phase-0.md`).
- **Phase 1a:** waves 0 and 1 are merged (T00–T05, T09, T14, T16, T18).
  Their decisions are in `plans/phase-1a.md`, in the sections "Conductor
  decisions after wave 0", "after wave 1" and "after wave 1b". Read them
  before briefing anyone.
- **Next: wave 2.** T10 is merged (see "after T10" in the plan). T06 (rewrite registry and linear family) comes first,
  then T07 and T08 in parallel. T10 (PolarMACE) needs only T03 and T05, so
  it can start now. It also fixes the tiny polar model; see the wave 1
  decisions. See "Order and parallelism" in the plan.
- **Critical path:** T06 → T07 → T11 → T13 → T15 → T17 → T19.

## Starting workers

- Preferred: app sessions with `cwd` this repository and their own
  worktree, one per task, branch `fastmace/<phase>-T<nn>`. From a session in
  this repository, proposal cards work too.
- The app starts sessions only in a turn the owner started. Turns started
  by a worker's report are refused ("this turn runs an automated event").
  So after merging, start the next workers in the owner's next turn, or ask
  the owner to say "start the next tasks". Do **not** use background
  subagents to get around this. The permission classifier blocked that as an
  "Auto-Mode Bypass" on 17 October 2026. Background subagents are fine only
  in a turn the owner started, when sessions are unavailable for another
  reason. In that case create their worktrees yourself
  (`git worktree add -b fastmace/1a-Tnn D:/Margraf/FastMACE-wt/1a-Tnn main`),
  and say "work only there" in the brief.
- **Never create worktrees of ResearchModule.** It is on a USB hard disk,
  and parallel worktrees there time out. Workers only read it, at
  `G:\Margraf\ResearchModule`.
- **The permission classifier refused one brief** that had a test import
  research-engine code from a path in an environment variable ("Create RCE
  Surface"). Briefs that copy expected values from the research engine's
  tests and source are accepted. T04 and T09 load the baseline through
  `FASTMACE_BASELINE` in their tests and were accepted. Keep new briefs close
  to those.
- At most four workers at once. Give each GPU task its own machine:
  local RTX 5060 Ti (WSL), leto or murphy, both RTX 4000 Ada, after an idle
  check. Timing runs (T18 noise floor, T19) need an otherwise idle GPU and
  nothing else running.
- A good brief contains: read first (`AGENTS.md`, the plan's working
  principles, the decision sections, the task, the named SPEC sections only,
  the design docs it depends on); the sources in ResearchModule; the files it
  owns and the ones it must not touch; tests first; how to run the CPU suite
  (`fastmace_cpu`, `-n 2` while siblings run) and GPU checks (which machine,
  environment, `--basetemp` on the NAS, `FASTMACE_HYBRID_PYTHON` for hybrid
  child processes); the commit and push rules; and what the final report must
  contain. The briefs of waves 0–1 are good templates; their reports are in
  the decision sections.

## Reviewing and merging

1. Read the report fully. Check the diff stat and that each branch touches
   only the files it owns.
2. Trial merge all ready branches in a scratch worktree:
   `git worktree add --detach D:/Margraf/fastmace-trial main`, then
   `git merge --no-ff origin/fastmace/<a> origin/fastmace/<b> ...`.
3. In WSL, from the trial worktree:
   `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<trial> FASTMACE_BASELINE=$HOME/fastmace-baseline/20261017 ~/environments/fastmace_cpu/bin/python -m pytest -n 8 -q tests/cpu tests/gpu`,
   then the same with `~/environments/mace_cueq_latest/bin/python` on
   `tests/cpu`, then `~/environments/fastmace_cpu/bin/ruff check fastmace tests`.
   Run once more, alone, any test that passes only in the full run:
   first-model-per-process rounding exists (`docs/development.md`).
4. If green: merge each branch with `git merge --no-ff` in the main
   checkout, check that the tree equals the trial tree, remove the trial
   worktree and push.
5. Record decisions and answers to the workers' questions in the phase
   plan, update `plans/README.md`, push.
6. Sync the NAS copy with the recipe in `docs/development.md` ("NAS
   checkout"). First the hash check that nothing was edited on the NAS, then
   `git archive | ssh murphy tar -x`, then `.source_commit`.

Pushing works with a plain `git push`: `.git/config` has a credential
helper that uses the `FZWINGEL-UBT` gh account. The active gh account
(`FZWINGEL`) cannot see this repository, so use
`GH_TOKEN=$(gh auth token --user FZWINGEL-UBT) gh ...` for single `gh`
commands. Never print the token.

## Environments in one place

- `~/environments/fastmace_cpu` (WSL): FastMACE's own venv, editable from
  `/mnt/d/Margraf/FastMACE`. Used for the CPU suite, ruff and wheel builds.
- `~/environments/mace_cueq_latest` and `mace_cueq_oeq` (WSL), and the NAS
  environments of the same names on the workstations: run FastMACE through
  `PYTHONPATH`, never installed there. ResearchModule's AGENTS.md governs
  them.
- The baseline export (tag `research-engine-baseline-20261017`) is at
  `~/fastmace-baseline/20261017` (WSL) and
  `/mnt/homes/zwingel/fastmace-baseline/20261017` (NAS).
- Model checkpoints: `/mnt/homes/zwingel/ResearchModule/Models` (NAS) and
  `~/models` (WSL). Pass them as paths or through `FASTMACE_MODELS`.
- WSL has 15 GB RAM: check `free -g` before suites, and never run two
  `-n 8` suites at once.
