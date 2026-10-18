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

## State on 18 October 2026

- **Phase 0:** done (`docs/validation/phase-0.md`).
- **Phase 1a:** 14 of 20 tasks are merged: T00–T10, T14, T16 and T18.
  Their decisions are in `plans/phase-1a.md`, in every "Conductor decisions
  after …" section. Read them all before briefing anyone.
- **T07/T08 GPU confirmation is open:** the GPU suite on the merged tree, and
  T08's murphy agreement run repeated with T07's real modules. See "after
  T07/T08" in the plan.
- **Next: T17 and T19** (prepare-then-wait, given their go after T15; see "after T15").
  All dependencies of both are merged. Then T13 (graph engine), T15
  (planner), T17 (`main.py` on FastMACE) and T19 (acceptance).
- **Critical path:** T17 → T19.

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
- **Ready list, one batch start (owner decision, 18 October 2026).** In
  every report-driven turn, review, merge and then update the "Ready to
  start" list in `plans/README.md`: each task whose dependencies are merged,
  with its brief ready. End the turn with one question, "Start all ready?".
  The owner's single reply ("start all ready") starts every task on the
  list in that turn, at most four at once, one GPU per task. No supported
  setting lets report-driven turns start sessions (research of 18 October
  2026; anthropics/claude-code#100913 and #89783 track it). A watcher script
  with an owner-written queue (`claude --bg`) is the hands-off alternative,
  to reconsider for phase 1b.
- **Prepare, then wait.** Start a dependent task early, in the owner's
  turn. It does its preparation, reports "prepared, waiting", and gets a go
  message after its dependency merges; messaging an existing session works
  from any turn. T17 and T19 run this way.
- **ResearchModule worktrees go on NVMe.** ResearchModule is on a USB hard
  disk: an app worktree there took more than 5 minutes and got stuck. For a
  task that must change ResearchModule (such as T17), the conductor creates
  the worktree on D: itself:
  `git -C G:/Margraf/ResearchModule -c safe.directory='*' worktree add -b fastmace/1a-Tnn D:/Margraf/ResearchModule-wt/1a-Tnn main`
  (about 2.5 minutes). It then starts the session with `cwd` that folder and
  `use_worktree: false`. Every git command there needs
  `-c safe.directory='*'`, because the repository's `.git` is on a file
  system without ownership. Never create app worktrees of ResearchModule,
  and never several in parallel. Workers that only read the research engine
  read it at `G:\Margraf\ResearchModule`.
- The app setting "When you ask for an unrelated task → Always start in a
  new session" (owner, 18 October 2026) concerns the owner's own requests.
  It does not let report-driven turns start sessions.
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
